"""Cosmos policy retrieval-augmented inference server for FFW BG2 bimanual.

Companion to ``77_fk_ffw_bg2_cosmos_control.ipynb``. Drop-in replacement for
``deploy_realrobot.py`` once a RAG checkpoint exists. Key differences vs the
non-RAG deploy script:

  1. **Pool indexing on startup** — uses the same active-arm classification,
     sub-bucket layout (task × arm × current_grip), KD-tree feature, and
     idle-slot override as the offline pair builder. All shared rules live
     in :mod:`cosmos_policy.datasets.retrieval_pair_utils`.

  2. **Per-request retrieval.** Given a 16-frame proprio history, classify
     the active arm, look up the chunk-start grip, query the bucket's
     KD-tree with active-arm chunk-start xyz (action component zeroed at
     inference — future action unknown), pull top-1.

  3. **Residual action with idle override.** Matches the pair-build logic
     exactly: idle arm gets cur_proprio[idle] broadcast into cur_action,
     ret_proprio, and ret_action. The model predicts a normalized residual;
     we unnormalize via delta_stats.json and add it to the (raw) retrieved
     action before publishing.

  4. **Custom image sequence builder.** The RAG model expects the state_t=8
     latent layout from ``policy_video2world_model_realrobot_ret.py``. We
     build it directly here because the cosmos ``image_sequence`` builder
     is hard-coded to PushT/RoboTwin layouts (which have a ret_proprio slot
     — ours doesn't).

Usage:

  # ── Default: OC abs R10 (Predict2.5 RF) ────────────────────────────────
  # _variant_settings decodes "_img" → use_ret_image, "_abs" → no_residual,
  # "_open_cabinet" → r_arm_only + per-task stats (per_task/open_cabinet/...).
  python -m cosmos_policy.experiments.robot.real_ffw.deploy_realrobot_ret \
      --backbone predict2p5 --variant top10_img_abs_open_cabinet \
      --task "open cabinet" \
      --pool-root /mnt/ddn/dataset/real_robot/h5_human_clean \
      --ckpt-path <학습 ckpt> --cpc-ip 192.168.0.175

  # ── Predict2 (EDM) OC abs R10 ──────────────────────────────────────────
  python -m cosmos_policy.experiments.robot.real_ffw.deploy_realrobot_ret \
      --backbone predict2 --variant top10_img_abs_open_cabinet \
      --task "open cabinet" --ckpt-path <ckpt>

  # ── Retrieval dropout 0.2 OC abs ───────────────────────────────────────
  python -m cosmos_policy.experiments.robot.real_ffw.deploy_realrobot_ret \
      --backbone predict2p5 --variant top10_img_abs_open_cabinet_drop02 \
      --task "open cabinet" --ckpt-path <ckpt>

  # ── PB single-task (20D, per_task/place_bottle_in_plastic_box stats) ───
  python -m cosmos_policy.experiments.robot.real_ffw.deploy_realrobot_ret \
      --variant top10_img_abs_place_bottle --task "place bottle in plastic box" \
      --ckpt-path <ckpt>

  # ── Full 2-task 20D (legacy) — manual stats path needed for 2-task base ─
  python -m cosmos_policy.experiments.robot.real_ffw.deploy_realrobot_ret \
      --variant top10_img_abs --task "place bottle in plastic box" \
      --dataset-stats-path <2task_stats.json> --delta-stats-path <delta.json> \
      --ckpt-path <ckpt>

  # ── Unseen condition: R-first place (override TASK_FIRST_ACTIVE) ───────
  # initial-arm is ignored for r_arm_only variants (forced R).
  python -m cosmos_policy.experiments.robot.real_ffw.deploy_realrobot_ret \
      --variant top10_img_abs_place_bottle --task "place bottle in plastic box" \
      --pool-root /mnt/ddn/dataset/real_robot/h5_human_unseen_clean \
      --initial-arm R --ckpt-path <ckpt>

Knobs:
  --backbone           {predict2p5, predict2}. Predict2.5 RF or Predict2 EDM.
  --variant            training variant; decodes _img/_abs/_open_cabinet/
                       _place_bottle/_drop0N suffixes into deploy flags.
  --task               one of {"place bottle in plastic box", "open cabinet",
                               "close cabinet"} — must match a T5 cache key.
  --pool-root          dir with {task}/{embodiment}/data/episode*.hdf5.
  --per-task-stats-root  parent dir for per-task stats (auto-resolved when
                       variant indicates single-task).
  --dataset-stats-path / --delta-stats-path  override auto-resolved stats.
  --initial-arm        "L"|"R"|None. Overrides TASK_FIRST_ACTIVE for unseen
                       deploy conditions (ignored for r_arm_only variants).
  --no-residual-actions / --no-ret-image / --r-arm-only / --no-r-arm-only
                       explicit overrides for variant-decoded flags.
  --ckpt-path          .pt file produced by convert_dcp_to_pt.py
  FLIP_TRIGGER_MM=X    env var; opposite-arm predicted motion threshold for
                       one-time active-arm flip (default 30 mm). r_arm_only
                       skips flip entirely.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from PIL import Image

from cosmos_policy.datasets.retrieval_pair_utils import (
    CHUNK_SIZE,
    POOL_EXCLUDE_TAIL_FRAMES,
    TASK_FIRST_ACTIVE,
    active_arm_xyz_slice,
    apply_grip_fallback,
    apply_idle_override,
    build_pool_index,
    classify_active_arm,
    classify_chunks,
    idle_arm_slot,
    list_episode_h5,
)
from cosmos_policy.experiments.robot.real_ffw.deploy_realrobot import (
    ACTION_DIM,
    DEFAULT_CHUNK_PORT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATASET_STATS_PATH,
    DEFAULT_REALSENSE_FPS,
    DEFAULT_REALSENSE_H,
    DEFAULT_REALSENSE_W,
    DEFAULT_REQUEST_IP,
    DEFAULT_REQUEST_PORT,
    DEFAULT_T5_EMBEDDINGS_PATH,
    PROPRIO_DIM,
    _ObservationSource,
)

LOGGER = logging.getLogger(__name__)

# ─── retrieval-specific constants ──────────────────────────────────────────
# Per-variant inference-config + ckpt path. Pick one via
# --backbone {predict2p5, predict2} --variant <name>. The variant name encodes
# the training setup:
#   top10[_img][_abs][_open_cabinet|_place_bottle][_drop0{1,2,3,4,5}]
# Conventions:
#   "_img"   → state_t=8 layout (ret_image at slot 1), use_ret_image=True
#   "_abs"   → absolute baseline (use_residual_actions=False)
#   "_open_cabinet" → single-task R-arm only (action_dim=10)
#   "_place_bottle" → single-task 20D
#   "_drop0N"       → retrieval dropout at training (deploy semantics unchanged;
#                     same R10 OC abs setup as parent)
# Predict2 (EDM) backbone currently only has open_cabinet variants registered.
_PREDICT2P5 = "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_"
_PREDICT2   = "cosmos_predict2_2b_480p_realrobot_bg2_ret_"

def _inf_cfg(prefix: str, name: str) -> str:
    return f"{prefix}{name}__inference_only"

_PREDICT2P5_VARIANTS = [
    # base 20-D (2-task) variants
    "top10", "top10_abs", "top10_img", "top10_img_abs",
    # single-task OC R10
    "top10_img_open_cabinet",
    "top10_img_abs_open_cabinet",
    # single-task PB 20D
    "top10_img_place_bottle",
    "top10_img_abs_place_bottle",
    # OC abs dropout variants
    "top10_img_abs_open_cabinet_drop01",
    "top10_img_abs_open_cabinet_drop02",
    "top10_img_abs_open_cabinet_drop03",
    "top10_img_abs_open_cabinet_drop04",
    "top10_img_abs_open_cabinet_drop05",
    # legacy aliases kept for compatibility with prior train tags
    "top1", "top3", "top1_abs", "top3_abs",
]
_PREDICT2_VARIANTS = [
    "top10_img_open_cabinet",
    "top10_img_abs_open_cabinet",
]

_VARIANT_CONFIGS: dict[tuple[str, str], str] = {
    **{("predict2p5", v): _inf_cfg(_PREDICT2P5, v) for v in _PREDICT2P5_VARIANTS},
    **{("predict2",   v): _inf_cfg(_PREDICT2,   v) for v in _PREDICT2_VARIANTS},
}

_VARIANT_CKPT_PATHS: dict[tuple[str, str], str] = {
    (bk, va): (
        f"/home/rilab/cosmos_policy/ckpt/rag/"
        f"model_{bk}_{va}_iter10000.pt"
    ) for (bk, va) in _VARIANT_CONFIGS
}


def _variant_settings(variant: str) -> dict:
    """Decode a variant name into deploy-time semantic flags.

    Returns: dict with keys
      use_residual_actions: bool
      use_ret_image:        bool
      r_arm_only:           bool      (open_cabinet single-task)
      per_task_dir:         Optional[str]  ('open_cabinet' / 'place_bottle_in_plastic_box' / None)
    """
    s = variant
    abs_mode = "_abs" in s
    use_img = "_img" in s
    is_oc = "_open_cabinet" in s
    is_pb = "_place_bottle" in s
    per_task = (
        "open_cabinet" if is_oc
        else "place_bottle_in_plastic_box" if is_pb
        else None
    )
    return dict(
        use_residual_actions=not abs_mode,
        use_ret_image=use_img,
        r_arm_only=is_oc,
        per_task_dir=per_task,
    )


DEFAULT_BACKBONE = "predict2p5"
DEFAULT_VARIANT = "top10_img_abs_open_cabinet"
DEFAULT_CONFIG_NAME = _VARIANT_CONFIGS[(DEFAULT_BACKBONE, DEFAULT_VARIANT)]
DEFAULT_CKPT_PATH = _VARIANT_CKPT_PATHS[(DEFAULT_BACKBONE, DEFAULT_VARIANT)]
DEFAULT_POOL_ROOT = "/home/rilab/cosmos_policy/ckpt/rag/h5_human_clean"
DEFAULT_DELTA_STATS_PATH = "/home/rilab/cosmos_policy/ckpt/rag/delta_stats.json"
DEFAULT_T5_EMBEDDINGS_PATH = "/home/rilab/cosmos_policy/ckpt/rag/t5_embeddings.pkl"
DEFAULT_DINO_CACHE_PATH = "/home/rilab/cosmos_policy/ckpt/rag/dino_first_frames.npz"
# Per-task stats roots used when variant indicates a single-task setup.
# The eval scripts and training configs read the same per-task dir layout:
#   <root>/<task_dir>/{dataset_statistics.json, delta_stats.json}
DEFAULT_PER_TASK_STATS_ROOT = "/home/rilab/cosmos_policy/ckpt/rag/per_task"


def _per_task_stats_paths(root: str, per_task_dir: str) -> tuple[str, str]:
    """Return (dataset_stats_path, delta_stats_path) for a per-task subdir."""
    return (
        f"{root}/{per_task_dir}/dataset_statistics.json",
        f"{root}/{per_task_dir}/delta_stats.json",
    )
NUM_DUPLICATES_PER_IMAGE = 4
# state_t depends on use_ret_image: 7 without ret_image latent, 8 with it.
STATE_T_NO_IMG = 7
STATE_T_WITH_IMG = 8
STATE_T = STATE_T_NO_IMG  # default; build_ret_data_batch(use_ret_image=True) returns the 8-slot layout
FINAL_IMAGE_SIZE = 224

# user-facing task name (with spaces, matches T5 cache key) → on-disk task dir
TASKS: dict[str, str] = {
    "place bottle in plastic box": "place_bottle_in_plastic_box",
    "open cabinet": "open_cabinet",
    "close cabinet": "close_cabinet",
}

# ─── DeployRetConfig ───────────────────────────────────────────────────────


@dataclass
class DeployRetConfig:
    """Subset of fields needed for retrieval-augmented inference."""

    suite: str = "real_robot"
    model_family: str = "cosmos"
    config: str = DEFAULT_CONFIG_NAME
    ckpt_path: str = DEFAULT_CKPT_PATH
    config_file: str = "cosmos_policy/config/config.py"
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""

    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = False
    num_wrist_images: int = 0
    use_proprio: bool = True
    flip_images: bool = False
    use_variance_scale: bool = False
    use_jpeg_compression: bool = False

    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    predict_future_states: bool = True
    predict_values: bool = False

    num_denoising_steps_action: int = 5
    chunk_size: int = DEFAULT_CHUNK_SIZE
    num_open_loop_steps: int = DEFAULT_CHUNK_SIZE
    state_t: int = STATE_T

    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    trained_with_image_aug: bool = True

    use_residual_actions: bool = True
    # When True, build state_t=8 layout with retrieved-human image latent
    # injected at slot 1 (before ret_action).
    use_ret_image: bool = False
    # When True, the model operates on R-arm only (action_dim=proprio_dim=10).
    # Pre-inference: slice 20-D cur_proprio / ret_action raw to [10:20].
    # Post-inference: pad predicted (16, 10) back to (16, 20) using
    # cur_proprio_raw[L_DIMS_20] for the idle L-arm. Single-task open_cabinet.
    r_arm_only: bool = False
    shift: float = 5.0
    seed: int = 195
    randomize_seed: bool = False
    deterministic: bool = True
    resize_size: int = FINAL_IMAGE_SIZE

    task: str = "place bottle in plastic box"
    realsense_width: int = DEFAULT_REALSENSE_W
    realsense_height: int = DEFAULT_REALSENSE_H
    realsense_fps: int = DEFAULT_REALSENSE_FPS
    cpc_ip: str = DEFAULT_REQUEST_IP
    request_port: int = DEFAULT_REQUEST_PORT
    chunk_port: int = DEFAULT_CHUNK_PORT
    dataset_stats_path: str = DEFAULT_DATASET_STATS_PATH
    t5_text_embeddings_path: str = DEFAULT_T5_EMBEDDINGS_PATH

    pool_root: str = DEFAULT_POOL_ROOT
    delta_stats_path: str = DEFAULT_DELTA_STATS_PATH
    dino_cache_path: str = DEFAULT_DINO_CACHE_PATH

    # Override the task's default first active arm (TASK_FIRST_ACTIVE).
    # Use when the deploy condition differs from the training data — e.g.
    # place_bottle R-first instead of the trained L-first.
    initial_arm: Optional[str] = None    # "L" | "R" | None=use task default

    # If set, dump retrieved image (frame 0) + ret_action_raw + cur_proprio
    # per chunk for offline inspection. None disables logging.
    log_ret_dir: Optional[str] = None


# ─── DINO CLS extractor (inline copy of robotwin_ret's helper to avoid
# pulling in the heavier robotwin retrieval module). ───────────────────────


class _DinoCLSExtractor:
    """timm DINOv2 ViT-B/14 → 768-D CLS embedding for a single RGB frame."""

    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m",
                 input_size: int = 224, amp_dtype: str = "bfloat16",
                 device: Optional[str] = None):
        import timm
        from timm.data import create_transform, resolve_model_data_config

        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = {
            "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16,
        }[amp_dtype]
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=input_size,
        ).eval().to(self.device)
        data_cfg = resolve_model_data_config(self.model)
        data_cfg["input_size"] = (3, input_size, input_size)
        self.transform = create_transform(**data_cfg, is_training=False)

    @torch.no_grad()
    def embed(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"expected (H,W,3) uint8, got {frame.shape} {frame.dtype}")
        x = self.transform(Image.fromarray(frame)).unsqueeze(0).to(self.device, non_blocking=True)
        if self.device.startswith("cuda") and self.amp_dtype != torch.float32:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                feat = self.model.forward_features(x)
        else:
            feat = self.model.forward_features(x)
        cls = feat[:, 0, :] if feat.ndim == 3 else feat
        return cls[0].float().cpu().numpy()


# ─── Pool indexing ─────────────────────────────────────────────────────────


class PoolIndex:
    """Thin wrapper over :func:`retrieval_pair_utils.build_pool_index` that
    keeps the matched chunks' h5 files open across requests so we don't
    re-open them per inference."""

    def __init__(self, pool_root: str, embodiment: str = "bg2_bimanual",
                 dino_cache_path: Optional[str] = None,
                 dino_device: Optional[str] = None):
        self.pool_root = pool_root
        self.embodiment = embodiment
        self._open_h5: dict[str, h5py.File] = {}
        # Single global episode lock — set once via DINO top-1 on the first
        # observed image and never replaced. Within that episode we still pick
        # the chunk nearest the current proprio xyz.
        self._locked_episode_h5: Optional[str] = None
        # Monotonic pointer per active arm: once we picked chunk_start = t for
        # arm X, subsequent retrieves on X cannot pick t' < t inside the locked
        # episode. Prevents the "ret jumps back and forth" jitter that makes
        # the robot wobble between chunks. Reset to {} on episode lock change
        # or via reset_monotonic() so a new task trial starts fresh.
        self._last_pcs_per_arm: dict[str, int] = {}
        self._dino_cache_path = dino_cache_path
        self._dino_device = dino_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dino_extractor: Optional["_DinoCLSExtractor"] = None
        self._dino_pool_unit: Optional[np.ndarray] = None  # (n_episodes, 768)
        self._dino_episode_paths: list[str] = []           # (n_episodes,) aligned with above
        all_chunks = []
        for task_h5 in TASKS.values():
            files = list_episode_h5(pool_root, task_h5, embodiment)
            chunks = classify_chunks(files, task_h5,
                                     exclude_tail_frames=POOL_EXCLUDE_TAIL_FRAMES)
            LOGGER.info("Pool %-30s : %d chunks (%d episodes)",
                        task_h5, len(chunks), len(files))
            all_chunks.extend(chunks)
        self.buckets = build_pool_index(all_chunks)
        for key, b in sorted(self.buckets.items()):
            LOGGER.info("  bucket %s: %d entries", key, len(b["items"]))

        # Build / load DINO first-frame cache for episode locking.
        self._load_or_build_dino_cache()

    # ─── DINO cache ────────────────────────────────────────────────────────

    def _all_pool_episodes(self) -> list[str]:
        """Same enumeration order as the constructor — first frame per ep."""
        eps: list[str] = []
        for task_h5 in TASKS.values():
            eps.extend(list_episode_h5(self.pool_root, task_h5, self.embodiment))
        return eps

    def _load_or_build_dino_cache(self) -> None:
        episodes = self._all_pool_episodes()
        if self._dino_cache_path and os.path.exists(self._dino_cache_path):
            data = np.load(self._dino_cache_path, allow_pickle=False)
            cached_paths = [str(p) for p in data["paths"].tolist()]
            if cached_paths == episodes:
                self._dino_pool_unit = data["feats"].astype(np.float32)
                self._dino_episode_paths = cached_paths
                LOGGER.info("DINO first-frame cache loaded from %s (%d eps)",
                            self._dino_cache_path, len(cached_paths))
                return
            LOGGER.info("DINO cache episode list changed — rebuilding")

        LOGGER.info("Building DINO first-frame cache over %d episodes...", len(episodes))
        ext = self._get_dino_extractor()
        feats = []
        for fp in episodes:
            with h5py.File(fp, "r") as h5:
                first_jpeg = bytes(h5["observation/third_eye/rgb"][0])
            img = np.asarray(Image.open(io.BytesIO(first_jpeg)).convert("RGB"))
            feats.append(ext.embed(img).astype(np.float32))
        feats_arr = np.stack(feats, axis=0)
        norms = np.linalg.norm(feats_arr, axis=1, keepdims=True)
        self._dino_pool_unit = (feats_arr / np.clip(norms, 1e-8, None)).astype(np.float32)
        self._dino_episode_paths = episodes
        if self._dino_cache_path:
            os.makedirs(os.path.dirname(self._dino_cache_path) or ".", exist_ok=True)
            np.savez(self._dino_cache_path,
                     feats=self._dino_pool_unit,
                     paths=np.array(episodes))
            LOGGER.info("DINO cache saved to %s", self._dino_cache_path)

    def _get_dino_extractor(self) -> "_DinoCLSExtractor":
        if self._dino_extractor is None:
            self._dino_extractor = _DinoCLSExtractor(device=self._dino_device)
        return self._dino_extractor

    def pick_episode_by_dino(self, image_rgb: np.ndarray) -> str:
        """Top-1 pool episode by cosine sim of DINO-CLS features.

        Both query and pool first-frames are 640x365 (CROP_RULE.md applied at
        h5 build / RealSense capture time), so we feed straight into DINO.
        """
        q = self._get_dino_extractor().embed(image_rgb).astype(np.float32)
        q_unit = q / max(float(np.linalg.norm(q)), 1e-8)
        sims = self._dino_pool_unit @ q_unit
        return self._dino_episode_paths[int(np.argmax(sims))]

    def reset(self) -> None:
        """Clear episode lock + monotonic per-arm pointers. Call between trials
        so the next retrieve() re-runs DINO top-1 on the incoming image and a
        fresh monotonic sequence begins for each arm."""
        self._locked_episode_h5 = None
        self._last_pcs_per_arm = {}

    def reset_monotonic(self, arm: Optional[str] = None) -> None:
        """Reset only the monotonic pointer (keep DINO episode lock). Pass
        ``arm`` to reset just one side (e.g. on active-arm flip)."""
        if arm is None:
            self._last_pcs_per_arm = {}
        else:
            self._last_pcs_per_arm.pop(arm, None)

    def ensure_keys(self, requested_keys: list[tuple[str, str, int]]) -> None:
        fell_back = apply_grip_fallback(self.buckets, requested_keys)
        for key in sorted(fell_back):
            LOGGER.warning("retrieval bucket %s empty — falling back to (%s, %s, %d)",
                           key, key[0], key[1], 1 - key[2])

    def _get_h5(self, h5_path: str) -> h5py.File:
        if h5_path not in self._open_h5:
            self._open_h5[h5_path] = h5py.File(h5_path, "r")
        return self._open_h5[h5_path]

    def retrieve(self, task: str, active_arm: str, current_grip: int,
                 query_xyz_3d: np.ndarray,
                 query_image_rgb: Optional[np.ndarray] = None) -> dict:
        """DINO-locked single-episode retrieval.

        On the first call we set ``_locked_episode_h5`` to the pool episode
        whose first frame has the highest DINO-CLS cosine similarity to
        ``query_image_rgb``. All subsequent calls reuse that episode and only
        pick the chunk within it that is nearest to ``query_xyz_3d``.
        """
        key = (task, active_arm, current_grip)
        if key not in self.buckets:
            self.ensure_keys([key])
        bucket = self.buckets[key]
        items = bucket["items"]
        xyz_slice = bucket["xyz_slice"]

        if self._locked_episode_h5 is None and query_image_rgb is not None:
            self._locked_episode_h5 = self.pick_episode_by_dino(query_image_rgb)
            LOGGER.info("episode lock (DINO top-1): %s", self._locked_episode_h5)

        # Filter bucket items to the locked episode; fall back to full bucket
        # if the locked episode has no chunks in this (arm, grip) bucket.
        if self._locked_episode_h5 is not None:
            candidates = [it for it in items if it.h5_path == self._locked_episode_h5]
            if not candidates:
                LOGGER.warning("locked episode %s has no chunks in bucket %s — falling back",
                               self._locked_episode_h5, key)
                candidates = items
        else:
            candidates = items

        # Monotonic constraint within the same active arm (causal retrieval):
        # cannot rewind past the last pcs we already picked for this arm.
        last_pcs = self._last_pcs_per_arm.get(active_arm, 0)
        mono = [it for it in candidates if it.chunk_start >= last_pcs]
        if not mono:
            # Relax (rare — locked ep ran out of forward chunks for this arm
            # or the bucket was empty after grip-fallback redirect).
            mono = candidates
        candidates = mono

        # Pick nearest by xyz within the (filtered, monotonic) candidate set.
        dists = [float(np.linalg.norm(query_xyz_3d - it.proprio_start[xyz_slice]))
                 for it in candidates]
        best = int(np.argmin(dists))
        entry = candidates[best]
        dist = dists[best]
        # Advance the per-arm pointer.
        self._last_pcs_per_arm[active_arm] = entry.chunk_start

        h5 = self._get_h5(entry.h5_path)
        cs = entry.chunk_start
        ret_jpegs = h5["observation/third_eye/rgb"][cs : cs + CHUNK_SIZE]
        ret_action_raw = h5["action"][cs : cs + CHUNK_SIZE].astype(np.float32)
        ret_proprio_raw = h5["proprio"][cs].astype(np.float32)
        return dict(
            ret_jpegs=[bytes(b) for b in ret_jpegs],
            ret_action_raw=ret_action_raw,
            ret_proprio_raw=ret_proprio_raw,
            dist=dist,
            pool_h5=entry.h5_path,
            pool_chunk_start=cs,
        )


# ─── data batch builder ────────────────────────────────────────────────────


def _decode_jpeg_to_rgb(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def _resize_uint8(img: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))


def build_ret_data_batch(
    cur_image_rgb: np.ndarray,
    cur_proprio_normalized: np.ndarray,
    ret_jpegs: list[bytes],
    ret_action_normalized: np.ndarray,
    t5_embedding: torch.Tensor,
    device: torch.device,
    use_ret_image: bool = False,
) -> dict:
    """Assemble the data_batch dict the RealRobot ret model expects.

    Layout (use_ret_image=False, state_t=7, chunk_duration=25):
      0 blank | 1 ret_action | 2 cur_image | 3 cur_proprio |
      4 pred_action | 5 future_image | 6 future_proprio   (NCF=4)

    Layout (use_ret_image=True, state_t=8, chunk_duration=29):
      0 blank | 1 ret_image | 2 ret_action | 3 cur_image | 4 cur_proprio |
      5 pred_action | 6 future_image | 7 future_proprio   (NCF=5)

    Each non-blank latent gets NUM_DUPLICATES_PER_IMAGE frames.
    """
    n_dup = NUM_DUPLICATES_PER_IMAGE
    cur_frames = np.broadcast_to(cur_image_rgb[None], (n_dup,) + cur_image_rgb.shape).copy()
    future_frames = cur_frames.copy()              # placeholder; model generates the future

    blank_single = np.zeros((1, *cur_image_rgb.shape), dtype=np.uint8)
    blank_dup = np.zeros((n_dup, *cur_image_rgb.shape), dtype=np.uint8)

    if use_ret_image:
        # Decode first n_dup of the 16 retrieved JPEGs and stack into (n_dup, H, W, 3).
        from PIL import Image as _PILImage
        import io as _io
        ret_decoded = np.stack(
            [np.asarray(_PILImage.open(_io.BytesIO(bytes(ret_jpegs[i]))).convert("RGB"))
             for i in range(n_dup)],
            axis=0,
        )
        # Resize each frame to match cur_image_rgb spatial size (PIL keeps any aspect).
        if ret_decoded.shape[1:3] != cur_image_rgb.shape[:2]:
            ret_decoded = np.stack([_resize_uint8(im, cur_image_rgb.shape[0]) for im in ret_decoded], axis=0)
        image_list = [
            blank_single,             # latent 0
            ret_decoded,              # latent 1 — ret_image
            blank_dup,                # latent 2 — ret_action injection target
            cur_frames,               # latent 3 — cur_image
            blank_dup,                # latent 4 — cur_proprio injection target
            blank_dup,                # latent 5 — pred_action
            future_frames,            # latent 6 — future_image
            blank_dup,                # latent 7 — future_proprio
        ]
        idx = dict(
            ret_image_start=1, ret_image_end=2,
            ret_action=2,
            cur_image=3, cur_proprio=4,
            action=5,
            future_image=6, future_proprio=7,
        )
        num_conditional_frames = 5
        has_ret_image_val = 1.0
    else:
        del ret_jpegs  # not used in state_t=7 layout
        image_list = [
            blank_single,             # latent 0
            blank_dup,                # latent 1 (ret_action injection target)
            cur_frames,               # latent 2
            blank_dup,                # latent 3 (cur_proprio injection target)
            blank_dup,                # latent 4 (pred action)
            future_frames,            # latent 5
            blank_dup,                # latent 6 (future proprio)
        ]
        idx = dict(
            ret_image_start=-1, ret_image_end=-1,
            ret_action=1,
            cur_image=2, cur_proprio=3,
            action=4,
            future_image=5, future_proprio=6,
        )
        num_conditional_frames = 4
        has_ret_image_val = 0.0

    images = np.concatenate(image_list, axis=0)
    images_resized = np.stack([_resize_uint8(im, FINAL_IMAGE_SIZE) for im in images], axis=0)
    video = (
        torch.from_numpy(images_resized)  # (T, H, W, C)
        .permute(3, 0, 1, 2)              # (C, T, H, W)
        .unsqueeze(0)                     # (1, C, T, H, W) — matches training
        .to(torch.uint8)
        .to(device)
    )

    def _t(v: int) -> torch.Tensor:
        return torch.tensor([v], dtype=torch.int64, device=device)

    return {
        "dataset_name": "video_data",
        "video": video,
        "t5_text_embeddings": t5_embedding.to(dtype=torch.bfloat16, device=device),
        "t5_text_mask": torch.ones(1, 512, dtype=torch.int64, device=device),
        "fps": torch.tensor([16], dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros(1, 1, FINAL_IMAGE_SIZE, FINAL_IMAGE_SIZE,
                                    dtype=torch.bfloat16, device=device),
        "num_conditional_frames": num_conditional_frames,
        "proprio": torch.from_numpy(cur_proprio_normalized).reshape(1, -1)
            .to(dtype=torch.bfloat16, device=device),
        "retrieved_actions": torch.from_numpy(ret_action_normalized)
            .reshape(1, *ret_action_normalized.shape)
            .to(dtype=torch.bfloat16, device=device),
        # latent indices — matches RealRobotRetDataset layout
        "current_proprio_latent_idx": _t(idx["cur_proprio"]),
        "current_image_latent_idx": _t(idx["cur_image"]),
        "current_image2_latent_idx": _t(-1),
        "current_wrist_image_latent_idx": _t(-1),
        "current_wrist_image2_latent_idx": _t(-1),
        "future_image_latent_idx": _t(idx["future_image"]),
        "future_image2_latent_idx": _t(-1),
        "future_proprio_latent_idx": _t(idx["future_proprio"]),
        "future_wrist_image_latent_idx": _t(-1),
        "future_wrist_image2_latent_idx": _t(-1),
        "action_latent_idx": _t(idx["action"]),
        "value_latent_idx": _t(-1),
        # ret_image: start/end activate ret_video masking in the model
        # (-1 disables for state_t=7).
        "retrieved_video_start_latent_idx": _t(idx["ret_image_start"]),
        "retrieved_video_end_latent_idx": _t(idx["ret_image_end"]),
        "retrieved_action_latent_idx": _t(idx["ret_action"]),
        "retrieved_state_latent_idx": _t(-1),
        "has_ret_data": torch.ones(1, device=device),
        "has_ret_image": torch.full((1,), has_ret_image_val, device=device),
        "has_current_image": torch.ones(1, device=device),
    }


# ─── normalization helpers ─────────────────────────────────────────────────


# Mixed action representation (must match RealRobotRetDataset):
#   xyz dims  → FRAME delta, residual learning + delta_xyz stats
#   non-xyz   → ABS direct, no residual, dataset_stats actions stats
#
# Layout depends on whether the model is R-arm only (action_dim=10) or full
# bimanual (action_dim=20). Both share the same 20-D raw layout from h5; the
# R10 path slices to [10:20] before feeding the model.
XYZ_DIMS_DEPLOY = np.array([0, 1, 2, 10, 11, 12], dtype=np.int64)
NON_XYZ_DIMS_DEPLOY = np.array([3, 4, 5, 6, 7, 8, 9,
                                 13, 14, 15, 16, 17, 18, 19], dtype=np.int64)
ROT_DIMS_DEPLOY = np.array([3, 4, 5, 6, 7, 8,
                             13, 14, 15, 16, 17, 18], dtype=np.int64)
GRIP_DIMS_DEPLOY = np.array([9, 19], dtype=np.int64)

# R-arm only layout (action_dim=10, indices into the [10:20] slice of the
# raw 20-D layout, i.e. local 0..9 with R_xyz at 0..2, R_rot6d at 3..8,
# R_grip at 9).
XYZ_DIMS_R10 = np.array([0, 1, 2], dtype=np.int64)
NON_XYZ_DIMS_R10 = np.array([3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
ROT_DIMS_R10 = np.array([3, 4, 5, 6, 7, 8], dtype=np.int64)
GRIP_DIMS_R10 = np.array([9], dtype=np.int64)

# Raw 20-D L/R slices used for r_arm_only pad-back.
L_DIMS_20 = np.arange(0, 10, dtype=np.int64)
R_DIMS_20 = np.arange(10, 20, dtype=np.int64)


def _dims_for(r_arm_only: bool) -> dict:
    """Return the dim-aware index arrays the build/reconstruct paths need."""
    if r_arm_only:
        return dict(
            xyz=XYZ_DIMS_R10, non_xyz=NON_XYZ_DIMS_R10,
            rot=ROT_DIMS_R10, grip=GRIP_DIMS_R10, action_dim=10,
        )
    return dict(
        xyz=XYZ_DIMS_DEPLOY, non_xyz=NON_XYZ_DIMS_DEPLOY,
        rot=ROT_DIMS_DEPLOY, grip=GRIP_DIMS_DEPLOY, action_dim=20,
    )


def _normalize_proprio(p_raw: np.ndarray, stats: dict) -> np.ndarray:
    return 2.0 * ((p_raw - stats["proprio_min"]) /
                  (stats["proprio_max"] - stats["proprio_min"] + 1e-8)) - 1.0


def _build_ret_action_input(ret_action_raw: np.ndarray,
                             cur_proprio_raw: np.ndarray,
                             dataset_stats: dict,
                             delta_stats: dict,
                             dims: Optional[dict] = None) -> np.ndarray:
    """Mirror of RealRobotRetDataset._build_ret_action_for_inject (mixed mode):
      xyz : ret_xyz_fd normalized by delta_stats.ret_xyz_fd_min/max
      rot : cur_proprio[rot] normalized via dataset_stats.proprio, broadcast
            across the chunk (robot rot stats are near-zero range, human rot
            saturates if pushed through them).
      grip: ret_action[grip] normalized via dataset_stats.actions (binary).

    ``dims`` is the output of ``_dims_for(r_arm_only)``. Defaults to 20-D
    bimanual when omitted (legacy callers).
    """
    if dims is None:
        dims = _dims_for(r_arm_only=False)
    xyz_dims = dims["xyz"]; rot_dims = dims["rot"]; grip_dims = dims["grip"]

    out = np.zeros_like(ret_action_raw, dtype=np.float32)
    # xyz: retrieved frame-delta
    xyz = ret_action_raw[:, xyz_dims]
    xyz_fd = np.diff(xyz, axis=0, prepend=xyz[0:1])
    r_min = np.asarray(delta_stats["ret_xyz_fd_min"], dtype=np.float32)
    r_max = np.asarray(delta_stats["ret_xyz_fd_max"], dtype=np.float32)
    out[:, xyz_dims] = (
        2.0 * ((xyz_fd - r_min) / (r_max - r_min + 1e-8)) - 1.0
    ).astype(np.float32)
    # rot: cur_proprio broadcast
    p_min = np.asarray(dataset_stats["proprio_min"], dtype=np.float32)
    p_max = np.asarray(dataset_stats["proprio_max"], dtype=np.float32)
    cur_p_norm = (2.0 * ((cur_proprio_raw - p_min) / (p_max - p_min + 1e-8)) - 1.0).astype(np.float32)
    out[:, rot_dims] = cur_p_norm[rot_dims][None, :]
    # grip: ret retrieved, normalized via action stats
    a_min_all = np.asarray(dataset_stats["actions_min"], dtype=np.float32)
    a_max_all = np.asarray(dataset_stats["actions_max"], dtype=np.float32)
    a_min = a_min_all[grip_dims]
    a_max = a_max_all[grip_dims]
    out[:, grip_dims] = np.clip(
        2.0 * ((ret_action_raw[:, grip_dims] - a_min) / (a_max - a_min + 1e-8)) - 1.0,
        -1.0, 1.0,
    ).astype(np.float32)
    return out


def _reconstruct_eef_chunk(
    pred_norm: np.ndarray,
    ret_action_raw: np.ndarray,
    cur_proprio_raw: np.ndarray,
    dataset_stats: dict,
    delta_stats: dict,
    use_residual_actions: bool = True,
    dims: Optional[dict] = None,
) -> np.ndarray:
    """Reconstruct absolute EEF chunk (D-dim, D=10 or 20) from prediction.

    Modes:
      use_residual_actions=True (mixed): xyz residual FRAME + non-xyz direct ABS
        xyz: cur_proprio[xyz] + cumsum(ret_xyz_fd + pred_xyz_fd_residual)
        non-xyz: direct unnormalize
      use_residual_actions=False (absolute baseline): all channels direct ABS
        every dim: unnormalize via dataset_stats actions_min/max

    ``dims`` selects the index layout; defaults to 20-D bimanual when None.
    Returns (16, D) absolute EEF pose ready to publish or pad-back to 20-D.
    """
    if dims is None:
        dims = _dims_for(r_arm_only=False)
    xyz_dims = dims["xyz"]; non_xyz_dims = dims["non_xyz"]; grip_dims = dims["grip"]

    eef = np.zeros_like(pred_norm, dtype=np.float32)
    a_min_all = np.asarray(dataset_stats["actions_min"], dtype=np.float32)
    a_max_all = np.asarray(dataset_stats["actions_max"], dtype=np.float32)

    if use_residual_actions:
        # xyz reconstruction (FRAME delta residual + cumsum)
        d_min = np.asarray(delta_stats["delta_xyz_min"], dtype=np.float32)
        d_max = np.asarray(delta_stats["delta_xyz_max"], dtype=np.float32)
        pred_xyz_resid = 0.5 * (pred_norm[:, xyz_dims] + 1.0) * (d_max - d_min) + d_min
        ret_xyz = ret_action_raw[:, xyz_dims]
        ret_xyz_fd = np.diff(ret_xyz, axis=0, prepend=ret_xyz[0:1])
        final_xyz_fd = pred_xyz_resid + ret_xyz_fd
        cur_xyz = cur_proprio_raw[xyz_dims]
        eef[:, xyz_dims] = cur_xyz[None] + np.cumsum(final_xyz_fd, axis=0)

        # non-xyz (rot, grip) direct ABS unnormalize
        a_min = a_min_all[non_xyz_dims]
        a_max = a_max_all[non_xyz_dims]
        pred_non_xyz = 0.5 * (pred_norm[:, non_xyz_dims] + 1.0) * (a_max - a_min) + a_min
        eef[:, non_xyz_dims] = pred_non_xyz
    else:
        # Absolute baseline: every dim direct unnormalize
        eef[:] = 0.5 * (pred_norm + 1.0) * (a_max_all - a_min_all) + a_min_all

    # Binarize grip
    eef[:, grip_dims] = (eef[:, grip_dims] >= 0.5).astype(np.float32)
    return eef


def _opposite_arm_predicted_motion_mm(
    pred_norm: np.ndarray,
    opposite_arm: str,
    delta_stats: dict,
) -> float:
    """Sum of |predicted xyz frame-delta residual| (in mm) on the opposite arm
    over the 16-step chunk. Used as the bucket-flip trigger.

    A large value means the model is predicting motion on the currently-idle
    arm — i.e. transition is happening.
    """
    d_min = delta_stats["delta_xyz_min"]
    d_max = delta_stats["delta_xyz_max"]
    pred_xyz_resid = 0.5 * (pred_norm[:, XYZ_DIMS_DEPLOY] + 1.0) * (d_max - d_min) + d_min
    # XYZ_DIMS_DEPLOY = [0,1,2,10,11,12]  → first 3 = L, last 3 = R
    opp_slc = slice(3, 6) if opposite_arm == "R" else slice(0, 3)
    return float(np.linalg.norm(pred_xyz_resid[:, opp_slc], axis=1).sum() * 1000.0)


# ─── main run loop ─────────────────────────────────────────────────────────


def run(cfg: DeployRetConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        extract_action_chunk_from_latent_sequence,
        get_model,
        get_t5_embedding_from_cache,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )
    from cosmos_policy.experiments.robot.real_ffw.zmq_pubsub import ZMQPubSub

    if cfg.deterministic:
        os.environ.setdefault("DETERMINISTIC", "True")

    LOGGER.info("Loading T5 cache from %s", cfg.t5_text_embeddings_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    t5_emb = get_t5_embedding_from_cache(cfg.task)

    LOGGER.info("Loading dataset stats from %s", cfg.dataset_stats_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    LOGGER.info("Loading delta stats from %s", cfg.delta_stats_path)
    with open(cfg.delta_stats_path, "r") as f:
        delta_stats = {k: np.asarray(v, dtype=np.float32) for k, v in json.load(f).items()}

    LOGGER.info("Loading model checkpoint from %s", cfg.ckpt_path)
    model, _ = get_model(cfg)
    device = next(model.parameters()).device
    LOGGER.info("Model ready on %s", device)

    LOGGER.info("Building pool index from %s …", cfg.pool_root)
    pool_index = PoolIndex(cfg.pool_root, dino_cache_path=cfg.dino_cache_path)

    obs_source = _ObservationSource(cfg=cfg)
    sub_request = ZMQPubSub(role="sub", ip=cfg.cpc_ip, port=cfg.request_port,
                            hwm=1, conflate=True, verbose=True)
    pub_chunk = ZMQPubSub(role="pub", port=cfg.chunk_port, hwm=1, verbose=True)

    # Optional per-chunk retrieval logging (cur_image + ret_image + ret_action +
    # cur_proprio + pred_action). Writes one subdir per chunk so the operator
    # can scrub through deploys offline.
    log_ret_dir: Optional[Path] = None
    if cfg.log_ret_dir:
        log_ret_dir = Path(cfg.log_ret_dir)
        log_ret_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("retrieval logging → %s", log_ret_dir)

    task_h5 = TASKS[cfg.task]
    if cfg.initial_arm in ("L", "R"):
        prev_active = cfg.initial_arm
        LOGGER.info("initial_arm override: %s (task default would be %s)",
                    prev_active, TASK_FIRST_ACTIVE[task_h5])
    else:
        prev_active = TASK_FIRST_ACTIVE[task_h5]
    already_flipped = False                            # max one flip per episode
    FLIP_TRIGGER_MM = float(os.environ.get("FLIP_TRIGGER_MM", "30.0"))
    n_chunks = 0
    LOGGER.info("Deploy_ret server ready.  FLIP_TRIGGER_MM=%.1f  start_active=%s",
                FLIP_TRIGGER_MM, prev_active)
    try:
        while True:
            obs_source.poll_once()
            req = sub_request.recv(nonblock=True)
            if req is None:
                time.sleep(0.005)
                continue
            if not isinstance(req, dict) or req.get("type") != "request":
                LOGGER.warning("Bad request: %r", req)
                continue

            # 16-frame proprio history needed for active-arm classification.
            # Fall back to broadcasting a single proprio if the control side
            # hasn't been upgraded to send a history yet (active arm will
            # then be stuck at TASK_FIRST_ACTIVE until the std exceeds the
            # threshold — flag this once for visibility).
            if "proprio_history" in req:
                history = np.asarray(req["proprio_history"], dtype=np.float32)
                assert history.shape == (CHUNK_SIZE, PROPRIO_DIM), history.shape
            else:
                if not obs_source.update_proprio_from_request(req):
                    continue
                snap_p = obs_source.snapshot()
                if snap_p is None:
                    continue
                history = np.tile(snap_p.proprio[None], (CHUNK_SIZE, 1))

            snap = obs_source.snapshot()
            if snap is None:
                LOGGER.warning("Request received but no image yet")
                continue

            # 1) Active arm = prev_active (latched; we do NOT use std-rule on
            # past history here — that would never flip). We rely on the
            # opposite-arm predicted-motion trigger below to flip once.
            # When r_arm_only=True the active arm is force-locked to R since
            # the model has no L-arm output channels — no flip trigger ever
            # fires.
            if cfg.r_arm_only:
                active_arm = "R"
            else:
                active_arm = prev_active
            cur_proprio_raw_20 = history[0]
            xyz_slice = active_arm_xyz_slice(active_arm)
            query_xyz = cur_proprio_raw_20[xyz_slice]
            # current_grip = chunk-start grip of active arm
            grip_idx_active = 9 if active_arm == "L" else 19
            current_grip = 1 if float(cur_proprio_raw_20[grip_idx_active]) >= 0.5 else 0

            # 2) Retrieve from active arm's bucket
            t0 = time.time()
            ret = pool_index.retrieve(task_h5, active_arm, current_grip, query_xyz,
                                      query_image_rgb=snap.primary_image)
            t_ret = time.time() - t0
            ret_action_raw_20 = ret["ret_action_raw"]                       # (16, 20) raw human

            # 2.5) Dim-aware slice: R10 model takes [10:20], full model takes 20-D.
            if cfg.r_arm_only:
                cur_proprio_raw = cur_proprio_raw_20[R_DIMS_20]
                ret_action_raw = ret_action_raw_20[:, R_DIMS_20]
            else:
                cur_proprio_raw = cur_proprio_raw_20
                ret_action_raw = ret_action_raw_20
            dims = _dims_for(cfg.r_arm_only)

            # 3) Normalize inputs for model
            cur_proprio_norm = _normalize_proprio(cur_proprio_raw, dataset_stats).astype(np.float32)
            if cfg.use_residual_actions:
                ret_action_norm = _build_ret_action_input(
                    ret_action_raw, cur_proprio_raw, dataset_stats, delta_stats,
                    dims=dims,
                )
            else:
                # Absolute baseline mirror of RealRobotRetDataset:
                # xyz/grip from retrieved (real signal); rot from cur_proprio
                # broadcast across the chunk to avoid saturation on near-zero
                # robot rot stats.
                a_min = np.asarray(dataset_stats["actions_min"], dtype=np.float32)
                a_max = np.asarray(dataset_stats["actions_max"], dtype=np.float32)
                ret_norm_all = np.clip(
                    2.0 * ((ret_action_raw - a_min) / (a_max - a_min + 1e-8)) - 1.0,
                    -1.0, 1.0,
                ).astype(np.float32)
                ret_action_norm = ret_norm_all.copy()
                ret_action_norm[:, dims["rot"]] = cur_proprio_norm[dims["rot"]][None, :]

            # 4) Build batch + generate
            data_batch = build_ret_data_batch(
                cur_image_rgb=snap.primary_image,
                cur_proprio_normalized=cur_proprio_norm,
                ret_jpegs=ret["ret_jpegs"],
                ret_action_normalized=ret_action_norm,
                t5_embedding=t5_emb,
                device=device,
                use_ret_image=cfg.use_ret_image,
            )

            t1 = time.time()
            with torch.inference_mode():
                samples = model.generate_samples_from_batch(
                    data_batch,
                    n_sample=1,
                    num_steps=cfg.num_denoising_steps_action,
                    seed=cfg.seed,
                    is_negative_prompt=False,
                    use_variance_scale=cfg.use_variance_scale,
                    return_orig_clean_latent_frames=False,
                    shift=cfg.shift,
                )
            t_inf = time.time() - t1

            # 5) Extract predicted residual chunk + reconstruct EEF
            # action_latent_idx shifts to 5 when ret_image latent is inserted.
            _action_idx = 5 if cfg.use_ret_image else 4
            action_indices = torch.full((1,), _action_idx, dtype=torch.int64, device=device)
            pred_norm = (
                extract_action_chunk_from_latent_sequence(
                    samples,
                    action_shape=(cfg.chunk_size, dims["action_dim"]),
                    action_indices=action_indices,
                )
                .to(torch.float32)
                .cpu()
                .numpy()[0]
            )
            final_eef_d = _reconstruct_eef_chunk(
                pred_norm=pred_norm,
                ret_action_raw=ret_action_raw,
                cur_proprio_raw=cur_proprio_raw,
                dataset_stats=dataset_stats,
                delta_stats=delta_stats,
                use_residual_actions=cfg.use_residual_actions,
                dims=dims,
            )

            # Pad R-arm-only output back to (16, 20) using cur_proprio L-arm
            # state held constant. Bimanual path is already 20-D.
            if cfg.r_arm_only:
                final_eef = np.zeros((cfg.chunk_size, 20), dtype=np.float32)
                final_eef[:, R_DIMS_20] = final_eef_d
                final_eef[:, L_DIMS_20] = cur_proprio_raw_20[L_DIMS_20][None, :]
            else:
                final_eef = final_eef_d

            # 6) Flip 0/1 trigger: if predicted motion on the OPPOSITE arm is
            # large, the model is signalling transition → flip active arm
            # (once per episode). Next cycle will retrieve from the new bucket.
            # Skipped for r_arm_only — the model has no L-arm output.
            did_flip = False
            opp_mm = 0.0
            if not cfg.r_arm_only:
                opposite = "R" if active_arm == "L" else "L"
                opp_mm = _opposite_arm_predicted_motion_mm(pred_norm, opposite, delta_stats)
                if not already_flipped and opp_mm >= FLIP_TRIGGER_MM:
                    LOGGER.info("FLIP trigger fired: opposite arm pred motion = %.1f mm >= %.1f → switching %s→%s",
                                opp_mm, FLIP_TRIGGER_MM, active_arm, opposite)
                    prev_active = opposite
                    already_flipped = True
                    did_flip = True
                    # Fresh monotonic pointer for the new arm so it starts from
                    # the locked ep's beginning of the new active-arm phase.
                    pool_index.reset_monotonic(arm=opposite)

            # Per-chunk retrieval log: dump cur img + ret img + raw arrays.
            if log_ret_dir is not None:
                chunk_dir = log_ret_dir / f"chunk_{n_chunks:05d}"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(snap.primary_image).save(chunk_dir / "cur_image.jpg",
                                                          quality=85)
                # Dump first + middle + last ret JPEGs (compact)
                for sample_i, frame_i in enumerate([0, 7, 15]):
                    p_ = chunk_dir / f"ret_image_f{frame_i:02d}.jpg"
                    with open(p_, "wb") as fh:
                        fh.write(ret["ret_jpegs"][frame_i])
                np.savez_compressed(
                    chunk_dir / "data.npz",
                    cur_proprio=cur_proprio_raw_20.astype(np.float32),
                    ret_action_raw=ret["ret_action_raw"].astype(np.float32),
                    ret_proprio_raw=ret["ret_proprio_raw"].astype(np.float32),
                    pred_norm=pred_norm.astype(np.float32),
                    final_eef=final_eef.astype(np.float32),
                    meta=dict(
                        active_arm=active_arm,
                        current_grip=current_grip,
                        ret_pool=Path(ret["pool_h5"]).name,
                        ret_chunk_start=int(ret["pool_chunk_start"]),
                        ret_dist_mm=float(ret["dist"] * 1000),
                        opp_arm_pred_mm=float(opp_mm),
                        did_flip=bool(did_flip),
                        already_flipped=bool(already_flipped),
                    ),
                )

            pub_chunk.send({
                "type": "chunk",
                "actions": final_eef,
                "req_ts": float(req.get("ts", time.time())),
                "pub_ts": time.time(),
                "retrieval_dist_m": ret["dist"],
                "active_arm": active_arm,
                "current_grip": current_grip,
                "ret_pool": Path(ret["pool_h5"]).name,
                "ret_chunk_start": ret["pool_chunk_start"],
                "opp_arm_pred_mm": opp_mm,
                "flipped_this_cycle": did_flip,
                "already_flipped": already_flipped,
                "inference_sec": t_inf,
                "retrieval_sec": t_ret,
            })
            n_chunks += 1
            LOGGER.info("chunk #%d  active=%s grip=%d  ret_dist=%.1fmm  opp_mm=%.1f%s  inf=%.0fms",
                        n_chunks, active_arm, current_grip,
                        ret["dist"] * 1000, opp_mm,
                        " FLIPPED" if did_flip else (" (locked)" if already_flipped else ""),
                        t_inf * 1000)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted.")
    finally:
        sub_request.close()
        pub_chunk.close()
        obs_source.close()


def _parse_args() -> DeployRetConfig:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Shortcut: pick backbone + variant → auto fills --config, --ckpt-path,
    # --r-arm-only, --use-ret-image, --no-residual-actions and (if the variant
    # is single-task) the per-task dataset/delta stats paths.
    p.add_argument("--backbone", choices=sorted({bk for (bk, _) in _VARIANT_CONFIGS}),
                   default=None,
                   help="auto-select inference config + default ckpt path. "
                        "Overridden by explicit --config/--ckpt-path.")
    p.add_argument("--variant", default=None,
                   choices=sorted({va for (_, va) in _VARIANT_CONFIGS}),
                   help="training variant name. Decodes _img/_abs/_open_cabinet/"
                        "_place_bottle/_drop0N suffixes into deploy-time flags.")
    p.add_argument("--config", default=None)
    p.add_argument("--ckpt-path", default=None)
    p.add_argument("--dataset-stats-path", default=None,
                   help=f"override stats path. Default: per-task if the variant is "
                        f"single-task ({DEFAULT_PER_TASK_STATS_ROOT}/<task>), else "
                        f"{DEFAULT_DATASET_STATS_PATH}.")
    p.add_argument("--delta-stats-path", default=None,
                   help="override delta_stats. Default: per-task or "
                        f"{DEFAULT_DELTA_STATS_PATH}.")
    p.add_argument("--per-task-stats-root", default=DEFAULT_PER_TASK_STATS_ROOT,
                   help="root dir for per-task {dataset,delta}_statistics.json. "
                        "Used when the chosen variant is single-task.")
    p.add_argument("--pool-root", default=DEFAULT_POOL_ROOT)
    p.add_argument("--t5-embeddings-path", default=DEFAULT_T5_EMBEDDINGS_PATH)
    p.add_argument("--task", default="open cabinet",
                   choices=list(TASKS.keys()))
    p.add_argument("--cpc-ip", default=DEFAULT_REQUEST_IP)
    p.add_argument("--request-port", type=int, default=DEFAULT_REQUEST_PORT)
    p.add_argument("--chunk-port", type=int, default=DEFAULT_CHUNK_PORT)
    p.add_argument("--num-denoising-steps-action", type=int, default=5)
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--initial-arm", default=None, choices=[None, "L", "R"],
                   help="override task default first-active arm. Ignored when "
                        "r_arm_only (active arm is forced to R).")
    # Override knobs — usually inferred from the variant name; CLI flags let
    # the user force a specific behavior independently of the auto-resolve.
    p.add_argument("--no-residual-actions", action="store_true",
                   help="absolute baseline: model output is ABS direct. Auto-set "
                        "when variant contains '_abs'.")
    p.add_argument("--no-ret-image", action="store_true",
                   help="disable state_t=8 ret_image latent. Auto-cleared when "
                        "variant lacks '_img'.")
    p.add_argument("--r-arm-only", action=argparse.BooleanOptionalAction, default=None,
                   help="force R-arm-only deploy (action_dim=10). Auto-set when "
                        "variant contains '_open_cabinet'.")
    p.add_argument("--log-ret-dir", default=None,
                   help="If set, dump retrieved JPEGs + actions per chunk to this "
                        "dir for offline inspection. E.g. /mnt/ddn/tmp/deploy_ret_log")
    args = p.parse_args()

    # Resolve backbone/variant shortcut
    bk = args.backbone or DEFAULT_BACKBONE
    va = args.variant or DEFAULT_VARIANT
    if (bk, va) not in _VARIANT_CONFIGS:
        raise SystemExit(
            f"Unknown (backbone, variant) = ({bk!r}, {va!r}). Valid combos:\n"
            + "\n".join(f"  {b}/{v}" for (b, v) in sorted(_VARIANT_CONFIGS))
        )
    if args.config is None:
        args.config = _VARIANT_CONFIGS[(bk, va)]
    if args.ckpt_path is None:
        args.ckpt_path = _VARIANT_CKPT_PATHS[(bk, va)]

    # Decode variant semantics; honor explicit overrides where provided.
    settings = _variant_settings(va)
    use_residual = settings["use_residual_actions"] and not args.no_residual_actions
    use_ret_image = settings["use_ret_image"] and not args.no_ret_image
    r_arm_only = (
        settings["r_arm_only"] if args.r_arm_only is None else bool(args.r_arm_only)
    )

    # Stats paths: per-task if variant is single-task and user didn't override.
    if settings["per_task_dir"] is not None:
        ds_default, delta_default = _per_task_stats_paths(
            args.per_task_stats_root, settings["per_task_dir"],
        )
    else:
        ds_default, delta_default = DEFAULT_DATASET_STATS_PATH, DEFAULT_DELTA_STATS_PATH
    dataset_stats_path = args.dataset_stats_path or ds_default
    delta_stats_path = args.delta_stats_path or delta_default

    LOGGER.info(
        "variant=%s/%s  use_residual=%s use_ret_image=%s r_arm_only=%s",
        bk, va, use_residual, use_ret_image, r_arm_only,
    )
    LOGGER.info("  config=%s", args.config)
    LOGGER.info("  ckpt=%s", args.ckpt_path)
    LOGGER.info("  dataset_stats=%s", dataset_stats_path)
    LOGGER.info("  delta_stats=%s", delta_stats_path)

    return DeployRetConfig(
        config=args.config,
        ckpt_path=args.ckpt_path,
        dataset_stats_path=dataset_stats_path,
        delta_stats_path=delta_stats_path,
        pool_root=args.pool_root,
        t5_text_embeddings_path=args.t5_embeddings_path,
        task=args.task,
        cpc_ip=args.cpc_ip,
        request_port=args.request_port,
        chunk_port=args.chunk_port,
        num_denoising_steps_action=args.num_denoising_steps_action,
        seed=args.seed,
        initial_arm=args.initial_arm,
        use_residual_actions=use_residual,
        use_ret_image=use_ret_image,
        r_arm_only=r_arm_only,
        log_ret_dir=args.log_ret_dir,
    )


if __name__ == "__main__":
    run(_parse_args())
