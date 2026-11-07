"""DINO-CLS + proprio-delta retrieval for RoboTwin RAG inference.

Loads a pool of pool-embodiment episodes (one or more split dirs under the
same task — typically clean_50 + randomized_500 for one robot family) and,
on each inference step, returns the top-1 matched (head, L wrist, R wrist,
proprio, action_chunk) window from that pool.

Score per pool frame:
    score = w_dino * cos(q_dino, p_dino)
          - w_prop      * mean(||q_dprop - p_dprop||)         # direction
          - w_prop_curr * ||q_curr  - p_curr||                # absolute pose
where the delta term is the uniform mean L2 over the K-1 consecutive
proprio deltas of the query (current EE + last K-1 history entries) against
the corresponding K-1 pool deltas ending at the candidate frame, and the
current term is the L2 between the query's current normalized proprio and
each pool frame's normalized proprio. The two together capture both
direction-of-motion and absolute-pose similarity. Proprio is normalized to
[-1,1] with the policy's training dataset_stats before differencing so that
DINO cosine and proprio L2 terms live on the same magnitude scale.

Pool frames are loaded into memory once at startup (head + L/R wrist + 16d
EE proprio + 16d EE action). DINO CLS features are loaded from the
robotwin_rag_tools cache:
    {dino_cache_root}/dino_cls/{task}/{split}/episode{N}.npz["head"]
    -> (T, 768) float32

Usage on the server side (see robotwin_ret_zmq_server.py):
    retrieval = RoboTwinDinoProprioRetrieval(...)
    retrieval.reset()                                # clears proprio history at episode boundary
    out = retrieval.get_retrieved(query_head, query_proprio, dataset_stats)
    obs.update(out)                                  # adds retrieved_head/L/R/proprio/actions
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import time
from collections import deque
from typing import Optional, Sequence

import h5py
import numpy as np
import torch

# Reuse the EE pool loader + 16d action builder from the existing RAG dataset.
from cosmos_policy.datasets.robotwin_dataset import _build_ee_action_16d  # noqa: F401
from cosmos_policy.datasets.dataset_utils import resize_images

# Raw EE (16-d quat) → 20-d state (rot6) conversion for the 20d_rot6 proprio
# scoring path. Mirrors robotwin_rag_tools/robotwin_rag/state_actions.raw_to_state
# so this server stays self-contained (no sibling-repo dependency).
from scipy.spatial.transform import Rotation as _Rotation


def _raw_to_state_20d(raw_16d: np.ndarray) -> np.ndarray:
    """(..., 16) raw [L_pos|L_quat_wxyz|L_grip|R_pos|R_quat_wxyz|R_grip]
    → (..., 20) state [L_pos|L_rot6|L_grip|R_pos|R_rot6|R_grip].

    The 6-d rotation (Zhou et al.) is the flattened first two columns of the
    rotation matrix. It is sign-flip immune (q ≡ -q give the same matrix)
    which fixes the cross-embodiment quat-hemisphere artifacts the 16-d quat
    path showed at retrieval scoring time.
    """
    raw = np.asarray(raw_16d, dtype=np.float64)
    flat_shape = raw.shape[:-1]
    flat = raw.reshape(-1, 16)
    out = np.empty((flat.shape[0], 20), dtype=np.float32)
    # L arm
    out[:, 0:3] = flat[:, 0:3]
    mat = _Rotation.from_quat(flat[:, 3:7], scalar_first=True).as_matrix()  # (N, 3, 3)
    out[:, 3:9] = mat[:, :, :2].reshape(-1, 6)
    out[:, 9] = flat[:, 7]
    # R arm
    out[:, 10:13] = flat[:, 8:11]
    mat = _Rotation.from_quat(flat[:, 11:15], scalar_first=True).as_matrix()
    out[:, 13:19] = mat[:, :, :2].reshape(-1, 6)
    out[:, 19] = flat[:, 15]
    return out.reshape(*flat_shape, 20)


def _decode_jpeg_sequence(byte_array_seq) -> np.ndarray:
    """Decode (T,) uint8 jpeg byte arrays -> (T, H, W, 3) uint8 RGB.

    See robotwin_dataset._decode_jpeg_sequence for why no BGR->RGB conversion
    is needed: RoboTwin's HDF5 JPEGs round-trip cleanly via cv2 so the
    decoded bytes already carry the original RGB pixel order.
    """
    import cv2

    frames: list[np.ndarray] = []
    for raw in byte_array_seq:
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("jpeg decode failed in retrieval pool loader")
        frames.append(img)
    return np.stack(frames, axis=0)


def _episode_num_from_path(p: str) -> int:
    return int(os.path.splitext(os.path.basename(p))[0].replace("episode", ""))


def _load_pool_episode(hdf5_path: str, final_image_size: int) -> dict:
    """Load one episode: head + L/R wrist (resized) + 16d proprio + 16d actions."""
    with h5py.File(hdf5_path, "r") as f:
        left_endpose = f["/endpose/left_endpose"][()]
        right_endpose = f["/endpose/right_endpose"][()]
        left_gripper = f["/endpose/left_gripper"][()]
        right_gripper = f["/endpose/right_gripper"][()]
        ee_at_t = _build_ee_action_16d(
            left_endpose, right_endpose, left_gripper, right_gripper
        )
        proprio = ee_at_t.astype(np.float32)
        # Action = next-step proprio (one-step left shift; final step holds last value).
        actions = np.concatenate([ee_at_t[1:], ee_at_t[-1:]], axis=0).astype(np.float32)

        head = _decode_jpeg_sequence(f["/observation/head_camera/rgb"][()])
        left_wrist = _decode_jpeg_sequence(f["/observation/left_camera/rgb"][()])
        right_wrist = _decode_jpeg_sequence(f["/observation/right_camera/rgb"][()])

    head = resize_images(head, final_image_size)
    left_wrist = resize_images(left_wrist, final_image_size)
    right_wrist = resize_images(right_wrist, final_image_size)
    T = len(head)
    assert len(actions) == T, (
        f"frame/action mismatch in {hdf5_path}: T={T} actions={len(actions)}"
    )
    return {
        "head": head,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "proprio": proprio,
        "actions": actions,
        "T": T,
    }


def _load_pool_dino(dino_cache_root: str, task: str, split: str, ep_num: int) -> np.ndarray:
    """Load (T, 768) head DINO CLS features for one pool episode."""
    npz_path = os.path.join(dino_cache_root, "dino_cls", task, split, f"episode{ep_num}.npz")
    with np.load(npz_path) as z:
        return z["head"].astype(np.float32)


# ---------------------------------------------------------------------------
# Lightweight DINOv2 ViT-B/14 CLS extractor (mirrors robotwin_rag.features)
# ---------------------------------------------------------------------------


class _DinoCLSExtractor:
    """Lazy-loaded DINOv2 ViT-B/14 CLS extractor for query-time embedding.

    Mirrors robotwin_rag_tools/robotwin_rag/features.DinoCLSExtractor so that
    pool features (extracted offline by that tool) and query features (extracted
    here online) live in the same embedding space. Avoids importing from the
    sibling repo so this server has no extra path dependency — the only price
    is duplicating ~30 lines.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        input_size: int = 224,
        amp_dtype: str = "bfloat16",
        device: Optional[str] = None,
    ):
        import timm
        from timm.data import create_transform, resolve_model_data_config

        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[amp_dtype]
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=input_size
        ).eval().to(self.device)
        data_cfg = resolve_model_data_config(self.model)
        data_cfg["input_size"] = (3, input_size, input_size)
        self.transform = create_transform(**data_cfg, is_training=False)

    @torch.no_grad()
    def embed(self, frame: np.ndarray) -> np.ndarray:
        """Embed a single (H, W, 3) uint8 RGB frame -> (768,) float32 CLS token."""
        from PIL import Image as _Image

        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"expected (H,W,3) uint8, got {frame.shape} {frame.dtype}")
        x = self.transform(_Image.fromarray(frame)).unsqueeze(0).to(self.device, non_blocking=True)
        if self.device.startswith("cuda") and self.amp_dtype != torch.float32:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                feat = self.model.forward_features(x)
        else:
            feat = self.model.forward_features(x)
        cls = feat[:, 0, :] if feat.ndim == 3 else feat
        return cls[0].float().cpu().numpy()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class RoboTwinDinoProprioRetrieval:
    """Joint DINO-CLS + proprio-history retrieval over a multi-split pool.

    All pool data is held in CPU memory; only DINO inference happens on GPU.
    """

    def __init__(
        self,
        dataset_root: str,
        task: str,
        pool_splits: Sequence[str],
        # Optional multi-task pool. When set (e.g. progression-eval mode),
        # the pool spans every (task, split) combination listed here and
        # each pool episode is tagged with its source task. ``task`` then
        # becomes the *query* task (used for SAM3 prompts, T5 lookup, etc.)
        # while ``pool_tasks`` controls which task data lives in the pool.
        # None / empty → backward-compat single-task pool (= [task]).
        pool_tasks: Optional[Sequence[str]] = None,
        dino_cache_root: str = "/mnt/ddn/robotwin_rag_tools/cache",
        chunk_size: int = 16,
        final_image_size: int = 224,
        ret_image_subsample: int = 4,
        proprio_history_window: int = 4,
        w_dino: float = 1.0,
        # Direction-of-motion (proprio delta) penalty weight. Operates on K-1
        # consecutive deltas of normalized proprio — magnitudes are small
        # (~0.01–0.1) so this typically wants to be ~5–10× larger than the
        # legacy absolute-proprio weight.
        w_prop: float = 0.5,
        # Current absolute proprio penalty weight. Operates on
        # ||q_curr − p_curr|| in normalized [-1,1] space (typical magnitude
        # ~5–8 in 16d), so values around 0.5 keep it on the same scale as
        # the DINO term.
        w_prop_curr: float = 0.5,
        # Per-dim weight on gripper channels (idx depends on proprio_repr) in
        # the proprio L2 (both curr and delta). 1.0 = uniform; >1 emphasizes
        # gripper-state agreement to disambiguate pre-grasp / post-grasp /
        # passive-arm-home cases where grip is the only discriminative dim.
        # Applied via sqrt-scaling on normalized proprio so downstream L2
        # becomes a weighted L2 with squared-term contribution = w_grip_curr.
        w_grip_curr: float = 1.0,
        # Per-dim weight on xyz channels (only used when proprio_repr ==
        # "20d_rot6"). 6-d rot6 has 2× more dims than 3-d pos at ~unit
        # magnitude, so raw-L2 rotation dominates by default. ``w_pos_curr=4``
        # multiplies xyz squared-contribution by 4 (sqrt-scale 2.0 on the 3
        # xyz dims per arm). 1.0 = no-op. Matches training-side --pos-weight.
        w_pos_curr: float = 1.0,
        # Proprio representation for scoring only — stored ``retrieved_proprio``
        # output stays 16-d quat for model compatibility regardless.
        # "16d_quat" (default): inference-legacy path, normalize via dataset_stats,
        # raw L2 in [-1, 1] space. Has cross-embodiment quat-hemisphere artifacts.
        # "20d_rot6": convert pool / query proprio to 20-d state (rot6) via
        # raw_to_state, skip dataset_stats normalize (rot6 values are bounded),
        # raw L2 in 20-d state space. Sign-flip immune, matches training-side
        # ``run_retrieval.py --proprio-repr 20d_rot6``.
        proprio_repr: str = "16d_quat",
        top_k: int = 1,
        n_initial_filter: int = 40,   # 0 = no filter; otherwise keep N closest episodes
        # Phase-alignment penalty weight. When > 0, the score gets an extra
        # term −w · |Q_cum(t) − P_cum(E, k)| where Q_cum is the query's
        # cumulative proprio path length since reset() and P_cum(E, k) is the
        # pool episode's cumulative path length up to frame k (both in the
        # query's normalized proprio space). 0 = legacy (no phase term).
        phase_alignment_weight: float = 0.0,
        # Per-step retrieval diagnostic dump. When set, every get_retrieved()
        # call appends a JSONL record with the picked frame + top-K candidate
        # score breakdown, and saves the query head image + picked pool head
        # image as PNGs. Use to diagnose backstep / wrong-frame matches.
        debug_dump_dir: Optional[str] = None,
        debug_dump_top_k: int = 10,
        # Home-pose transform (cross-embodiment alignment). When the query arm
        # is near the QUERY-side home pose, shift it toward the POOL-side home
        # pose before proprio scoring. Per-arm independent; uses a Gaussian
        # decay so mid-task queries (arms far from home) are unaffected.
        # JSON path holding {aloha_home_proprio, ur5_home_proprio, home_offset}
        # produced by the home-pose precompute. Empty/None disables transform.
        home_pose_json: Optional[str] = None,
        # When ``home_pose_json`` is empty, try to auto-resolve a JSON inside
        # this directory by (task, pool_family) — e.g. pool_splits=
        # ["ur5_clean_50", "ur5_randomized_500"] → family "ur5" → look up
        # ``<dir>/<task>__aloha-vs-ur5.json``. None disables auto-resolution.
        home_pose_auto_dir: Optional[str] = "/mnt/ddn/cosmos_policy/home_poses",
        # Decay scale: σ_arm = β * ||offset_arm||. Larger β + larger ``p`` ⇒
        # plateau near home and a sharper falloff outside. Defaults give
        # α≈0.61 at d=0.5*||offset|| and α≈0 by d=||offset||.
        home_transform_beta: float = 0.7,
        # Decay exponent: α = exp(-(d/σ)^p). p=2 is a Gaussian (smooth bell);
        # p≥4 is a super-Gaussian — flat near home, then a knee-then-drop.
        # p=4 picks the home-aligned pool frames at near-home queries while
        # leaving mid-task queries (d > σ) untouched.
        home_transform_p: float = 4.0,
        # Soft forward bias inside an episode we've already picked from:
        #   bonus = w_fwd * tanh((local_idx - last_picked[ep]) / chunk_size)
        # bounded by ±w_fwd. Tips score landscape toward forward progress
        # in flat regions; small backsteps (recovery) stay possible since
        # tanh saturates softly. 0 = legacy (no bias); 0.5 is a good first
        # pass since mid-task score differences are typically ~0.05–0.2.
        forward_bias_weight: float = 0.0,
        # Mirror training-NPZ force-fr=0 at inference: for the first
        # ``force_initial_pool_fr0_n_acts`` act() calls after reset(), restrict
        # the pool candidate set to frame=0 entries only. 0 = disable.
        force_initial_pool_fr0_n_acts: int = 0,
        # Per-chunk advancement brake: penalize candidates whose local_idx
        # exceeds (last_picked + advance_brake_max_step) within the same
        # episode. Targets training-NPZ-like slow pool_fr advancement (≈
        # chunk_size * P_step / Q_step ≈ 9 frames per 16-step query chunk for
        # the aloha→ur5 ratio). 0 = disable.
        advance_brake_weight: float = 0.0,
        advance_brake_max_step: int = 10,
        # When True, skip the per-step DINO embed() call entirely and use a
        # zero query vector instead. Saves ~10–30 ms per act() on Cosmos-2B
        # at the cost of removing the DINO term from both the initial-scene
        # filter and per-step scoring. Object_init (SAM3) + proprio + delta
        # still drive retrieval. Default False keeps full DINO behavior.
        disable_dino: bool = False,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        assert pool_splits, "pool_splits must be non-empty"
        self.dataset_root = dataset_root
        self.task = task   # query task (drives SAM3 prompts, T5 lookup, etc.)
        self.pool_splits = list(pool_splits)
        # Multi-task pool composition: tuple of task names that contribute to
        # the pool. Defaults to a single-task pool of ``[task]`` for backward
        # compat. Per-pool-episode metadata gains a ``task`` field below.
        if pool_tasks:
            self.pool_tasks: list[str] = list(dict.fromkeys(pool_tasks))  # de-dup, preserve order
        else:
            self.pool_tasks = [task]
        self.dino_cache_root = dino_cache_root
        self.chunk_size = int(chunk_size)
        self.final_image_size = int(final_image_size)
        self.ret_image_subsample = int(ret_image_subsample)
        self.n_ret_frames_per_camera = self.chunk_size // self.ret_image_subsample
        self.proprio_history_window = int(max(1, proprio_history_window))
        self.w_dino = float(w_dino)
        self.w_prop = float(w_prop)
        self.w_prop_curr = float(w_prop_curr)
        self.w_grip_curr = float(max(0.0, w_grip_curr))
        self.w_pos_curr = float(max(0.0, w_pos_curr))
        if proprio_repr not in ("16d_quat", "20d_rot6"):
            raise ValueError(f"proprio_repr must be '16d_quat' or '20d_rot6'; got {proprio_repr!r}")
        self.proprio_repr = str(proprio_repr)
        # Sqrt-scale vector applied element-wise to normalized proprio. L2 of
        # (sqrt_w * x) reproduces sqrt(sum(w_i * x_i^2)) — a weighted L2. Grip
        # / pos indices depend on the chosen proprio representation:
        #   16d_quat → grip [7, 15];                pos weighting unused
        #   20d_rot6 → grip [9, 19], pos [0:3, 10:13]
        if self.proprio_repr == "16d_quat":
            dim_w = np.ones(16, dtype=np.float32)
            dim_w[[7, 15]] = self.w_grip_curr
        else:
            dim_w = np.ones(20, dtype=np.float32)
            dim_w[[9, 19]] = self.w_grip_curr
            dim_w[[0, 1, 2, 10, 11, 12]] = self.w_pos_curr
        self._proprio_dim_sqrt_w = np.sqrt(dim_w).astype(np.float32)
        self.top_k = int(max(1, top_k))
        self.n_initial_filter = int(max(0, n_initial_filter))
        self.phase_alignment_weight = float(max(0.0, phase_alignment_weight))

        # Diagnostic dump state.
        self.debug_dump_dir = (
            os.path.abspath(debug_dump_dir) if debug_dump_dir else None
        )
        self.debug_dump_top_k = int(max(1, debug_dump_top_k))
        self._debug_step_count = 0
        self._debug_pool_seen: set[tuple[int, int]] = set()
        if self.debug_dump_dir is not None:
            os.makedirs(os.path.join(self.debug_dump_dir, "query_head"), exist_ok=True)
            os.makedirs(os.path.join(self.debug_dump_dir, "pool_head"), exist_ok=True)

        # Home-pose transform.
        self._home_transform_enabled = False
        self._aloha_home: Optional[np.ndarray] = None
        self._ur5_home: Optional[np.ndarray] = None
        self._home_offset: Optional[np.ndarray] = None
        self._sigma_L: float = 0.0
        self._sigma_R: float = 0.0
        self._home_transform_beta = float(max(1e-6, home_transform_beta))
        self._home_transform_p = float(max(1.0, home_transform_p))
        self._forward_bias_weight = float(max(0.0, forward_bias_weight))
        self._force_initial_pool_fr0_n_acts = int(max(0, force_initial_pool_fr0_n_acts))
        self._n_acts_since_reset = 0
        self._advance_brake_weight = float(max(0.0, advance_brake_weight))
        self._advance_brake_max_step = int(max(0, advance_brake_max_step))
        self._disable_dino = bool(disable_dino)
        # Per-episode last-picked local index, populated as the rollout picks
        # frames. Cleared on reset(). Used by ``_compute_forward_bias``.
        self._last_picked_local_idx_per_ep: dict[int, int] = {}
        # Auto-resolve home_pose_json if not provided. Pool family is the first
        # split's prefix before ``_clean`` / ``_randomized`` (e.g. "ur5",
        # "arx-x5"). Filename convention: ``<task>__aloha-vs-<family>.json``.
        if not home_pose_json and home_pose_auto_dir and pool_splits:
            fam = pool_splits[0]
            for tag in ("_clean", "_randomized"):
                if tag in fam:
                    fam = fam.split(tag)[0]
                    break
            cand = os.path.join(home_pose_auto_dir, f"{task}__aloha-vs-{fam}.json")
            if os.path.isfile(cand):
                home_pose_json = cand
                if verbose:
                    print(f"[home-transform] auto-resolved → {cand}", flush=True)
            elif verbose:
                print(
                    f"[home-transform] no auto-resolved JSON (looked for {cand})",
                    flush=True,
                )
        if home_pose_json:
            with open(home_pose_json) as fh:
                hp = json.load(fh)
            self._aloha_home = np.asarray(hp["aloha_home_proprio"], dtype=np.float32)
            self._ur5_home = np.asarray(hp["ur5_home_proprio"], dtype=np.float32)
            self._home_offset = (self._ur5_home - self._aloha_home).astype(np.float32)
            self._sigma_L = float(self._home_transform_beta * np.linalg.norm(self._home_offset[:8]))
            self._sigma_R = float(self._home_transform_beta * np.linalg.norm(self._home_offset[8:]))
            self._home_transform_enabled = True
            if verbose:
                print(
                    f"[home-transform] enabled — beta={self._home_transform_beta} "
                    f"p={self._home_transform_p} "
                    f"||L_offset||={np.linalg.norm(self._home_offset[:8]):.3f} σ_L={self._sigma_L:.3f}  "
                    f"||R_offset||={np.linalg.norm(self._home_offset[8:]):.3f} σ_R={self._sigma_R:.3f}",
                    flush=True,
                )

        # Pool data layout — see _build_or_load_pool_cache() for cache schema.
        # After the call:
        #   self._pool_data: list[{"split", "ep_num", "T"}]    per-episode metadata
        #   self._frame_episode: (N,) int32                    abs frame -> episode index
        #   self._frame_local_idx: (N,) int32                  abs frame -> local index in ep
        #   self._dino_pool: (N, 768) float32                  in-RAM (used every score call)
        #   self._proprio_pool_raw: (N, 16) float32            in-RAM
        #   self._all_head, _all_left_wrist, _all_right_wrist: h5py datasets (lazy I/O)
        #   self._all_actions: (N, 16) float32                 in-RAM
        #   self._cache_h5: open h5py.File handle (kept for image-dataset access)
        self._cache_h5: Optional[h5py.File] = None
        self._build_or_load_pool_cache(verbose=verbose)

        # Pre-normalize for fast cosine similarity.
        norms = np.linalg.norm(self._dino_pool, axis=1, keepdims=True)
        self._dino_pool_unit = (self._dino_pool / np.clip(norms, 1e-8, None)).astype(np.float32)

        # Cumulative episode-frame offsets so subset windowing knows which
        # absolute pool indices belong to which episode.
        self._ep_starts = np.concatenate(
            [[0], np.cumsum([e["T"] for e in self._pool_data])]
        ).astype(np.int64)

        # Per-episode start fingerprint (frame 0 of each episode) used for the
        # initial-scene filter. Indices into the flat pool arrays.
        ep0_idx = self._ep_starts[:-1]                                       # (n_ep,)
        self._ep_start_dino_unit = self._dino_pool_unit[ep0_idx].copy()      # (n_ep, 768)
        self._ep_start_proprio_raw = self._proprio_pool_raw[ep0_idx].copy()  # (n_ep, 16)

        # Lazy DINO model — built on first call so that cold-start doesn't
        # block model loading on the policy side.
        self._dino_model: Optional[_DinoCLSExtractor] = None
        self._dino_device = device

        # Past proprio buffer for the current rollout (raw 16d); reset() clears.
        self._proprio_history_raw: deque[np.ndarray] = deque(maxlen=self.proprio_history_window)

        # Cached normalized pool proprio (lazily built once dataset_stats is known).
        self._proprio_pool_norm: Optional[np.ndarray] = None
        self._cached_stats_signature: Optional[tuple] = None

        # Phase-alignment: cumulative proprio path length per pool frame in the
        # query's normalized space (lazy, signature-cached) and the running
        # query cumulative motion since reset(). Both are unused when
        # phase_alignment_weight == 0.
        self._pool_cum_motion: Optional[np.ndarray] = None
        self._cached_phase_signature: Optional[tuple] = None
        self._q_cum_motion_since_reset: float = 0.0
        self._q_prev_norm: Optional[np.ndarray] = None

        # Initial-scene filter state — set on the first act of each episode
        # using the current obs as the scene fingerprint, then frozen for the
        # rest of the rollout.
        # _allowed_frame_idx: int32 array of absolute pool-frame indices (or
        #                     None when filtering is disabled / all-pool mode)
        # _allowed_episodes : int array of episode indices kept by the filter
        # _needs_initial_filter: True between reset() and the first act
        self._allowed_frame_idx: Optional[np.ndarray] = None
        self._allowed_episodes: Optional[np.ndarray] = None
        self._needs_initial_filter = self.n_initial_filter > 0

        if verbose:
            print(
                f"[RoboTwinDinoProprioRetrieval] total pool frames={len(self._dino_pool_unit)} "
                f"across {len(self._pool_data)} episodes; w_dino={self.w_dino} "
                f"w_prop={self.w_prop} w_prop_curr={self.w_prop_curr} "
                f"w_grip_curr={self.w_grip_curr} w_pos_curr={self.w_pos_curr} "
                f"proprio_repr={self.proprio_repr} "
                f"history_window={self.proprio_history_window} "
                f"n_initial_filter={self.n_initial_filter} "
                f"phase_alignment_weight={self.phase_alignment_weight} "
                f"forward_bias_weight={self._forward_bias_weight}"
            )

    # --------------------------------------------------------------------- API

    def reset(self) -> None:
        """Clear past-proprio buffer + arm initial-scene filter for next act.

        Called on episode boundary ('reset' op). The actual filter is computed
        on the FIRST act after reset (using that obs as the scene fingerprint)
        and then frozen for the rest of the rollout.
        """
        self._proprio_history_raw.clear()
        self._allowed_frame_idx = None
        self._allowed_episodes = None
        self._needs_initial_filter = self.n_initial_filter > 0
        self._q_cum_motion_since_reset = 0.0
        self._q_prev_norm = None
        self._debug_step_count = 0
        self._last_picked_local_idx_per_ep = {}
        self._n_acts_since_reset = 0

    def get_retrieved(
        self,
        query_head_image: np.ndarray,   # (H, W, 3) uint8 — current step head camera
        query_proprio: np.ndarray,      # (16,) float32 — current step raw proprio
        dataset_stats: dict,            # policy training stats with proprio_min/max
        query_proprio_history: Optional[np.ndarray] = None,
        # (M, 16) raw proprio captured at SIM-STEP rate, ordered oldest -> most-recent
        # (last row = current). When provided, the server-supplied history wins and
        # the internal chunk-spaced deque is ignored, so K-window matching has the
        # same temporal sample rate as the pool. Pass None to fall back to legacy
        # chunk-spaced behavior (one append per act call).
    ) -> dict:
        """Return retrieved conditioning at the current step.

        Output keys (raw / un-normalized; cosmos_utils handles normalization):
            retrieved_head:        (n_ret, H, W, 3) uint8
            retrieved_left_wrist:  (n_ret, H, W, 3) uint8
            retrieved_right_wrist: (n_ret, H, W, 3) uint8
            retrieved_proprio:     (16,) float32 raw
            retrieved_actions:     (chunk_size, 16) float32 raw
            retrieval_meta:        dict with split/ep/frame for diagnostics
        """
        if self._disable_dino:
            # Skip the embed() call entirely; downstream scoring (initial
            # filter + per-step DINO term) multiplies this by ``self.w_dino``
            # so zeros contribute nothing without needing extra guards.
            q_dino_unit = np.zeros(
                self._dino_pool_unit.shape[1], dtype=np.float32
            )
        else:
            if self._dino_model is None:
                self._dino_model = _DinoCLSExtractor(device=self._dino_device)

            q_dino = self._dino_model.embed(np.ascontiguousarray(query_head_image)).astype(np.float32)
            q_dino_unit = q_dino / max(float(np.linalg.norm(q_dino)), 1e-8)

        proprio_pool_norm = self._get_proprio_pool_norm(dataset_stats)

        # ── Home-pose transform (no-op when home_pose_json wasn't given) ────
        # Shift query into UR5 space when the corresponding arm is near aloha
        # home. Applied per-arm with Gaussian decay so mid-task is unchanged.
        # Original (untransformed) query proprio is preserved for the deque /
        # debug dump; the transformed copy drives proprio scoring only.
        query_proprio_orig = np.asarray(query_proprio, dtype=np.float32)
        query_proprio_for_score = self._apply_home_transform(query_proprio_orig)

        if query_proprio_history is not None and len(query_proprio_history) > 0:
            # Sim-step history supplied by the caller. Truncate to K from the
            # current end so we never use stale entries when the buffer is
            # longer than the configured window.
            K = self.proprio_history_window
            hist = np.asarray(query_proprio_history, dtype=np.float32)
            if hist.ndim != 2 or hist.shape[1] != int(self._proprio_pool_raw.shape[1]):
                raise ValueError(
                    f"query_proprio_history must be (M, {self._proprio_pool_raw.shape[1]}); "
                    f"got {hist.shape}"
                )
            hist = hist[-K:]
            # Apply home transform to each row independently — α(d) is
            # per-frame so the buffer of "near home → mid-task" frames will
            # smoothly transition from full-shift to identity, preserving
            # the direction-of-motion signal in the relevant region.
            hist = self._apply_home_transform_batch(hist)
            if self.proprio_repr == "20d_rot6":
                q_hist_norm = _raw_to_state_20d(hist).astype(np.float32)
            else:
                p_min = np.asarray(dataset_stats["proprio_min"], dtype=np.float32)
                p_max = np.asarray(dataset_stats["proprio_max"], dtype=np.float32)
                q_hist_norm = (
                    2.0 * ((hist - p_min) / (p_max - p_min + 1e-8)) - 1.0
                ).astype(np.float32)
            # Keep the chunk-spaced deque in sync with what the caller is
            # tracking, so n_allowed_episodes / diag math doesn't drift if the
            # caller stops sending history mid-rollout.
            self._proprio_history_raw.append(np.asarray(query_proprio, dtype=np.float32).copy())
        else:
            # Legacy chunk-spaced fallback (one append per act call). Append
            # raw to deque, then transform the materialized history before
            # normalization so the score uses UR5-aligned values.
            self._proprio_history_raw.append(np.asarray(query_proprio, dtype=np.float32).copy())
            q_hist_norm = self._normalize_history(dataset_stats)        # (K, 16)

        # ---- Initial-scene filter (first act after reset) ------------------
        # Score every pool *episode* by start-frame DINO cosine + start-frame
        # proprio L2 vs. this query. Keep top n_initial_filter episodes for
        # the rest of the rollout. Drops scoring cost ~13× for 40/550 keep.
        # Pass the home-transformed proprio so the ep0-vs-query L2 is computed
        # in UR5 space (relevant when query is at aloha home).
        if self._needs_initial_filter:
            self._compute_initial_filter(q_dino_unit, query_proprio_for_score, dataset_stats)
            self._needs_initial_filter = False

        # Force pool_fr=0 for the first N acts after reset (training-NPZ
        # parity). Restricts candidate frames to local_idx==0 only; intersects
        # with prefilter ``_allowed_frame_idx`` when active. Applied as a
        # per-call local override (``allowed`` below) so the prefilter's
        # persistent ``_allowed_frame_idx`` isn't narrowed permanently.
        # No-op when the threshold is 0 (default).
        force_fr0_active = (
            self._force_initial_pool_fr0_n_acts > 0
            and self._n_acts_since_reset < self._force_initial_pool_fr0_n_acts
        )

        # Current absolute proprio (last row of normalized history) — used by
        # the w_prop_curr penalty term. Always defined since K >= 1.
        q_curr_norm = q_hist_norm[-1]                                          # (D,)

        # Weighted views for proprio L2 (curr + delta). Multiplying both pool
        # and query by sqrt(weights) turns plain L2 into the desired weighted
        # L2 with squared-term contribution = w_grip_curr on grip dims. The
        # unweighted ``proprio_pool_norm`` is preserved for the phase term.
        sqrt_w = self._proprio_dim_sqrt_w
        proprio_pool_norm_w = proprio_pool_norm * sqrt_w
        q_curr_norm_w = q_curr_norm * sqrt_w
        q_hist_norm_w = q_hist_norm * sqrt_w

        # ---- Update query cumulative motion since reset (chunk-rate) -------
        # Each get_retrieved call adds ||q_curr − q_prev|| in the query's
        # normalized proprio space. This is a chunk-rate approximation of the
        # query's accumulated path length — used by the phase-alignment term.
        if self.phase_alignment_weight > 0.0 and self._q_prev_norm is not None:
            self._q_cum_motion_since_reset += float(
                np.linalg.norm(q_curr_norm - self._q_prev_norm)
            )
        self._q_prev_norm = q_curr_norm.copy()

        # Phase-alignment penalty: |Q_cum(t) − P_cum(E, k)|. 0 when disabled.
        phase_penalty_full: Optional[np.ndarray] = None
        if self.phase_alignment_weight > 0.0:
            pool_cum = self._get_pool_cum_motion(dataset_stats)
            phase_penalty_full = np.abs(
                pool_cum - self._q_cum_motion_since_reset
            ).astype(np.float32)

        # ---- Score (filtered or full pool) ---------------------------------
        # Build the per-call allowed set: start from the persistent prefilter
        # (``_allowed_frame_idx``), then optionally narrow to fr=0 candidates
        # for the first N acts.
        allowed = self._allowed_frame_idx
        if force_fr0_active:
            fr0_idx = np.where(self._frame_local_idx == 0)[0].astype(np.int64)
            if allowed is not None and len(allowed) > 0:
                fr0_idx = np.intersect1d(fr0_idx, np.asarray(allowed, dtype=np.int64))
            if len(fr0_idx) > 0:
                allowed = fr0_idx
        if allowed is not None:
            dino_score_subset = self._dino_pool_unit[allowed] @ q_dino_unit
            prop_penalty_subset = self._proprio_delta_l2_episodes(
                proprio_pool_norm_w, q_hist_norm_w, self._allowed_episodes
            )[allowed]
            curr_penalty_subset = np.linalg.norm(
                proprio_pool_norm_w[allowed] - q_curr_norm_w, axis=1
            ).astype(np.float32)
            phase_penalty_subset = (
                phase_penalty_full[allowed]
                if phase_penalty_full is not None
                else np.zeros_like(curr_penalty_subset)
            )
            scores_subset = (
                self.w_dino * dino_score_subset
                - self.w_prop * prop_penalty_subset
                - self.w_prop_curr * curr_penalty_subset
                - self.phase_alignment_weight * phase_penalty_subset
            )
            # Per-chunk T5 / SAM3 ep-level biases (mirrors
            # render_progression_runtime_video.py). No-op for exact-match
            # pools (subclass sets self._exact_pool=True and zeros these out
            # at init; arrays are also missing in legacy / single-task mode).
            scores_subset = scores_subset + self._per_chunk_ep_biases(allowed)
            forward_bias_subset = self._compute_forward_bias(allowed)
            scores_subset = scores_subset + forward_bias_subset
            advance_brake_subset = self._compute_advance_brake(allowed)
            scores_subset = scores_subset - advance_brake_subset
            top_n = min(self.top_k, len(scores_subset))
            order = np.argpartition(-scores_subset, top_n - 1)[:top_n]
            local_pick = int(order[0]) if self.top_k == 1 else int(
                order[int(np.random.randint(top_n))]
            )
            best = int(allowed[local_pick])
            scores = scores_subset       # for diagnostic logging only
            dino_score = dino_score_subset
            prop_penalty = prop_penalty_subset
            curr_penalty = curr_penalty_subset
            phase_penalty = phase_penalty_subset
            best_for_diag = local_pick
            allowed_for_diag = allowed
        else:
            dino_score = self._dino_pool_unit @ q_dino_unit
            prop_penalty = self._proprio_delta_l2(proprio_pool_norm_w, q_hist_norm_w)
            curr_penalty = np.linalg.norm(
                proprio_pool_norm_w - q_curr_norm_w, axis=1
            ).astype(np.float32)
            phase_penalty = (
                phase_penalty_full
                if phase_penalty_full is not None
                else np.zeros_like(curr_penalty)
            )
            scores = (
                self.w_dino * dino_score
                - self.w_prop * prop_penalty
                - self.w_prop_curr * curr_penalty
                - self.phase_alignment_weight * phase_penalty
            )
            all_idx = np.arange(scores.shape[0], dtype=np.int64)
            scores = scores + self._per_chunk_ep_biases(all_idx)
            forward_bias_full = self._compute_forward_bias(all_idx)
            scores = scores + forward_bias_full
            advance_brake_full = self._compute_advance_brake(all_idx)
            scores = scores - advance_brake_full
            top_n = min(self.top_k, len(scores))
            order = np.argpartition(-scores, top_n - 1)[:top_n]
            best = int(order[0]) if self.top_k == 1 else int(
                order[int(np.random.randint(top_n))]
            )
            best_for_diag = best
            allowed_for_diag = None

        ep_index = int(self._frame_episode[best])
        local_idx = int(self._frame_local_idx[best])
        # Track per-episode last-picked local index for the forward-bias term
        # on the next call. Always keep the *most recent* pick (not the max),
        # so a deliberate small backstep can re-anchor the bias center.
        self._last_picked_local_idx_per_ep[ep_index] = local_idx
        ep = self._pool_data[ep_index]
        T = int(ep["T"])
        abs_start = int(self._ep_starts[ep_index])

        # 4 distinct stride-sampled frames per camera (model conditioning input
        # — mirrors RoboTwinRetDataset).
        sampled_local = [
            max(0, min(local_idx + i * self.ret_image_subsample, T - 1))
            for i in range(self.n_ret_frames_per_camera)
        ]
        sampled_abs = np.asarray(
            [abs_start + i for i in sampled_local], dtype=np.int64
        )
        # h5py fancy indexing requires STRICTLY increasing indices. Near
        # episode end the min(..., T-1) clamp produces duplicates (e.g.
        # [185, 186, 186, 186]). Read the unique frames once, then expand
        # back to the original 4-row layout the model expects.
        unique_abs, inverse = np.unique(sampled_abs, return_inverse=True)
        ret_head = np.asarray(self._all_head[unique_abs.tolist()])[inverse]
        ret_left = np.asarray(self._all_left_wrist[unique_abs.tolist()])[inverse]
        ret_right = np.asarray(self._all_right_wrist[unique_abs.tolist()])[inverse]

        # Full 16-frame contiguous window (logging only — not fed to the
        # model). Pad with the last frame if the window runs past episode end
        # so logger gets a fixed chunk_size length per call.
        last_local = min(local_idx + self.chunk_size, T)
        win_lo, win_hi = abs_start + local_idx, abs_start + last_local
        ret_head_full = np.asarray(self._all_head[win_lo:win_hi])
        ret_left_full = np.asarray(self._all_left_wrist[win_lo:win_hi])
        ret_right_full = np.asarray(self._all_right_wrist[win_lo:win_hi])
        n_pad = self.chunk_size - ret_head_full.shape[0]
        if n_pad > 0:
            ret_head_full = np.concatenate(
                [ret_head_full, np.repeat(ret_head_full[-1:], n_pad, axis=0)], axis=0
            )
            ret_left_full = np.concatenate(
                [ret_left_full, np.repeat(ret_left_full[-1:], n_pad, axis=0)], axis=0
            )
            ret_right_full = np.concatenate(
                [ret_right_full, np.repeat(ret_right_full[-1:], n_pad, axis=0)], axis=0
            )

        # Retrieved action chunk (raw); pad with last action if window runs off
        # end. Actions live in RAM so plain numpy slicing.
        ret_actions = self._all_actions[win_lo:win_hi].copy()
        if len(ret_actions) < self.chunk_size:
            pad = np.repeat(ret_actions[-1:], self.chunk_size - len(ret_actions), axis=0)
            ret_actions = np.concatenate([ret_actions, pad], axis=0)
        ret_proprio = self._proprio_pool_raw[abs_start + min(local_idx, T - 1)].copy()

        # ── Diagnostic dump (no-op when debug_dump_dir is None) ─────────────
        if self.debug_dump_dir is not None:
            try:
                self._dump_debug(
                    query_head_image=query_head_image,
                    query_proprio=query_proprio,
                    proprio_pool_norm_w=proprio_pool_norm_w,
                    q_curr_norm_w=q_curr_norm_w,
                    scores=scores,
                    dino_score=dino_score,
                    prop_penalty=prop_penalty,
                    curr_penalty=curr_penalty,
                    phase_penalty=phase_penalty,
                    picked_local_idx=best_for_diag,
                    picked_abs=best,
                    allowed=allowed_for_diag,
                )
            except Exception as e:
                # Diagnostic must never break inference.
                print(f"[retrieval/debug-dump] failed: {e}", flush=True)
        self._debug_step_count += 1
        self._n_acts_since_reset += 1

        return {
            "retrieved_head": ret_head,
            "retrieved_left_wrist": ret_left,
            "retrieved_right_wrist": ret_right,
            # Full chunk_size contiguous window for logging (not used by model).
            "retrieved_head_full": ret_head_full,
            "retrieved_left_wrist_full": ret_left_full,
            "retrieved_right_wrist_full": ret_right_full,
            "retrieved_proprio": ret_proprio.astype(np.float32),
            "retrieved_actions": ret_actions.astype(np.float32),
            "retrieval_meta": {
                "split": ep["split"],
                "ep_num": int(ep["ep_num"]),
                "frame": local_idx,
                "score": float(scores[best_for_diag]),
                "dino_score": float(dino_score[best_for_diag]),
                "prop_penalty": float(prop_penalty[best_for_diag]),
                "curr_penalty": float(curr_penalty[best_for_diag]),
                "phase_penalty": float(phase_penalty[best_for_diag]),
                "q_cum_motion": float(self._q_cum_motion_since_reset),
                "history_len": len(self._proprio_history_raw),
                "n_allowed_episodes": int(len(self._allowed_episodes))
                if self._allowed_episodes is not None
                else int(len(self._pool_data)),
            },
        }

    # ----------------------------------------------------------------- internals

    # ── Packed-cache schema (single .h5 per (task, splits, image_size)) ──
    # /head, /left_wrist, /right_wrist : (N_total, H, H, 3) uint8  chunks=(1,H,H,3)  lzf
    # /proprio, /actions               : (N_total, 16)     float32  chunks=(256,16)  lzf
    # /dino                            : (N_total, 768)    float32  chunks=(256,768) lzf
    # /episode_starts                  : (n_ep + 1,)       int64
    # /episode_splits                  : (n_ep,)           |S64    (utf-8 bytes)
    # /episode_nums                    : (n_ep,)           int32
    # /episode_Ts                      : (n_ep,)           int32
    # attrs: schema_version, task, pool_splits (csv), final_image_size,
    #        n_episodes, n_total_frames, created_at
    _CACHE_SCHEMA_VERSION = 2   # v2 adds per-episode ``task`` (multi-task pool)

    def _pool_cache_path(self) -> str:
        # Multi-task pool: include every contributing task in the cache key.
        # Single-task pool collapses to the legacy ``<task>`` filename stem.
        tasks_sorted = sorted(self.pool_tasks)
        key = (
            f"{','.join(tasks_sorted)}__"
            f"{','.join(sorted(self.pool_splits))}__"
            f"sz{self.final_image_size}"
        )
        digest = hashlib.sha1(key.encode()).hexdigest()[:10]
        if len(self.pool_tasks) == 1:
            stem = self.pool_tasks[0]
        else:
            stem = f"multi{len(self.pool_tasks)}_{self.pool_tasks[0]}_etc"
        cache_dir = os.path.join(self.dino_cache_root, "pool_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(
            cache_dir,
            f"{stem}__sz{self.final_image_size}__{digest}.h5",
        )

    def _build_or_load_pool_cache(self, verbose: bool = True) -> None:
        cache_path = self._pool_cache_path()
        if os.path.isfile(cache_path):
            if verbose:
                print(f"[pool-cache] hit  → {cache_path}", flush=True)
            self._open_packed_cache(cache_path)
            return
        # Concurrent-safe build: take an exclusive flock on a sidecar lockfile
        # so multiple servers asking for the same cache serialize. The first
        # one builds the .h5; the rest wake up and find it already there.
        # Avoids ``BlockingIOError`` and ``FileNotFoundError`` from racing on
        # the .tmp file.
        import fcntl
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        lock_path = cache_path + ".lock"
        with open(lock_path, "w") as lf:
            if verbose:
                print(f"[pool-cache] acquiring build lock → {lock_path}", flush=True)
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                # Re-check after grabbing the lock — peer may have built it
                # while we were waiting.
                if os.path.isfile(cache_path):
                    if verbose:
                        print(
                            f"[pool-cache] another process built it → {cache_path}",
                            flush=True,
                        )
                    self._open_packed_cache(cache_path)
                    return
                if verbose:
                    print(
                        f"[pool-cache] miss → building (one-time, ~minutes): {cache_path}",
                        flush=True,
                    )
                self._build_packed_cache(cache_path, verbose=verbose)
                self._open_packed_cache(cache_path)
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

    def _build_packed_cache(self, cache_path: str, verbose: bool = True) -> None:
        """Walk hdf5 + dino npz files once, stream into a single lzf-compressed h5."""
        from tqdm import tqdm

        # Enumerate (task, split, ep_num, hdf5_path) deterministically across
        # every pool_task × pool_split combination. Single-task pool degenerates
        # to the legacy single iteration since ``self.pool_tasks == [self.task]``.
        episodes: list[tuple[str, str, int, str]] = []
        for ptask in self.pool_tasks:
            for split in self.pool_splits:
                split_dir = os.path.join(self.dataset_root, ptask, split, "data")
                assert os.path.isdir(split_dir), (
                    f"missing pool split dir: {split_dir!r} "
                    f"(task={ptask!r} split={split!r})"
                )
                files = sorted(glob.glob(os.path.join(split_dir, "*.hdf5")))
                assert files, f"no hdf5 files in {split_dir}"
                for f in files:
                    episodes.append((ptask, split, _episode_num_from_path(f), f))

        H = self.final_image_size
        tmp_path = cache_path + ".tmp"
        t0 = time.time()
        try:
            with h5py.File(tmp_path, "w") as out:
                # Resizeable datasets so we can append episode-by-episode without
                # holding everything in RAM simultaneously.
                head_ds = out.create_dataset(
                    "head", shape=(0, H, H, 3), maxshape=(None, H, H, 3),
                    dtype=np.uint8, chunks=(1, H, H, 3), compression="lzf",
                )
                left_ds = out.create_dataset(
                    "left_wrist", shape=(0, H, H, 3), maxshape=(None, H, H, 3),
                    dtype=np.uint8, chunks=(1, H, H, 3), compression="lzf",
                )
                right_ds = out.create_dataset(
                    "right_wrist", shape=(0, H, H, 3), maxshape=(None, H, H, 3),
                    dtype=np.uint8, chunks=(1, H, H, 3), compression="lzf",
                )
                proprio_ds = out.create_dataset(
                    "proprio", shape=(0, 16), maxshape=(None, 16),
                    dtype=np.float32, chunks=(256, 16), compression="lzf",
                )
                actions_ds = out.create_dataset(
                    "actions", shape=(0, 16), maxshape=(None, 16),
                    dtype=np.float32, chunks=(256, 16), compression="lzf",
                )
                dino_ds = out.create_dataset(
                    "dino", shape=(0, 768), maxshape=(None, 768),
                    dtype=np.float32, chunks=(256, 768), compression="lzf",
                )

                splits_list: list[bytes] = []
                ep_nums_list: list[int] = []
                Ts_list: list[int] = []
                tasks_list: list[bytes] = []
                running = 0
                iterator = tqdm(episodes, desc="pack pool") if verbose else episodes
                for ptask, split, ep_num, f in iterator:
                    ep = _load_pool_episode(f, H)
                    dino = _load_pool_dino(self.dino_cache_root, ptask, split, ep_num)
                    T = ep["T"]
                    if dino.shape[0] != T:
                        raise RuntimeError(
                            f"DINO/HDF5 length mismatch for {split}/episode{ep_num}: "
                            f"dino={dino.shape[0]} hdf5={T}"
                        )

                    new_n = running + T
                    for ds, arr in (
                        (head_ds, ep["head"]),
                        (left_ds, ep["left_wrist"]),
                        (right_ds, ep["right_wrist"]),
                        (proprio_ds, ep["proprio"]),
                        (actions_ds, ep["actions"]),
                        (dino_ds, dino),
                    ):
                        ds.resize(new_n, axis=0)
                        ds[running:new_n] = arr
                    splits_list.append(split.encode("utf-8"))
                    ep_nums_list.append(int(ep_num))
                    Ts_list.append(int(T))
                    tasks_list.append(ptask.encode("utf-8"))
                    running = new_n

                ep_starts = np.concatenate([[0], np.cumsum(Ts_list)]).astype(np.int64)
                out.create_dataset("episode_splits", data=np.array(splits_list, dtype="S64"))
                out.create_dataset("episode_nums", data=np.array(ep_nums_list, dtype=np.int32))
                out.create_dataset("episode_Ts", data=np.array(Ts_list, dtype=np.int32))
                out.create_dataset("episode_starts", data=ep_starts)
                # NEW in schema v2: per-episode task name (multi-task pool).
                out.create_dataset("episode_tasks", data=np.array(tasks_list, dtype="S64"))

                out.attrs["schema_version"] = self._CACHE_SCHEMA_VERSION
                out.attrs["task"] = self.task
                out.attrs["pool_tasks"] = ",".join(self.pool_tasks)
                out.attrs["pool_splits"] = ",".join(self.pool_splits)
                out.attrs["final_image_size"] = H
                out.attrs["n_episodes"] = len(episodes)
                out.attrs["n_total_frames"] = running
                out.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            os.replace(tmp_path, cache_path)
        except Exception:
            # Don't leave a half-written file masquerading as a valid cache.
            if os.path.isfile(tmp_path):
                os.remove(tmp_path)
            raise

        sz_mb = os.path.getsize(cache_path) / 1e6
        elapsed = time.time() - t0
        print(
            f"[pool-cache] built {running} frames across {len(episodes)} episodes "
            f"in {elapsed:.1f}s — {sz_mb:.1f} MB at {cache_path}",
            flush=True,
        )

    def _open_packed_cache(self, cache_path: str) -> None:
        h5 = h5py.File(cache_path, "r")
        # v1 caches predate the multi-task `episode_tasks` field; the reader
        # below already falls back to `[self.task] * n_episodes` when the
        # dataset is missing, so v1 stays readable for single-task pools.
        # Reject anything newer than the current schema (forward-incompatible).
        cache_sv = int(h5.attrs.get("schema_version", -1))
        cache_fis = int(h5.attrs.get("final_image_size", -1))
        if cache_sv < 1 or cache_sv > self._CACHE_SCHEMA_VERSION:
            h5.close()
            raise RuntimeError(
                f"pool cache schema version unsupported "
                f"(file={cache_sv}, supported=[1..{self._CACHE_SCHEMA_VERSION}]) "
                f"— delete {cache_path} and retry"
            )
        if cache_sv < self._CACHE_SCHEMA_VERSION:
            print(
                f"[pool-cache] opening v{cache_sv} cache (current={self._CACHE_SCHEMA_VERSION}); "
                f"single-task fallback in use → {cache_path}",
                flush=True,
            )
        if cache_fis != self.final_image_size:
            h5.close()
            raise RuntimeError(
                f"pool cache final_image_size mismatch: "
                f"file={cache_fis} vs requested={self.final_image_size}"
            )
        self._cache_h5 = h5

        # Small arrays → load full into RAM (used every score call).
        self._proprio_pool_raw = h5["proprio"][:].astype(np.float32)
        self._all_actions = h5["actions"][:].astype(np.float32)
        self._dino_pool = h5["dino"][:].astype(np.float32)
        self._ep_starts = h5["episode_starts"][:].astype(np.int64)

        # Image datasets → keep as h5py refs (lazy I/O at retrieval time).
        self._all_head = h5["head"]
        self._all_left_wrist = h5["left_wrist"]
        self._all_right_wrist = h5["right_wrist"]

        # Per-episode metadata
        splits = [s.decode("utf-8") for s in h5["episode_splits"][:]]
        ep_nums = h5["episode_nums"][:].tolist()
        Ts = h5["episode_Ts"][:].tolist()
        # NEW in schema v2: per-episode task name. Old single-task v1 caches
        # have been rebuilt under the v2 schema-version bump so this dataset
        # is guaranteed to exist; for safety we still fall back to ``self.task``
        # if it's missing.
        if "episode_tasks" in h5:
            tasks = [t.decode("utf-8") for t in h5["episode_tasks"][:]]
        else:
            tasks = [self.task] * len(splits)
        self._pool_data = [
            {"task": t, "split": s, "ep_num": int(n), "T": int(T)}
            for t, s, n, T in zip(tasks, splits, ep_nums, Ts)
        ]

        # Derived flat arrays (frame -> episode/local index lookup).
        self._frame_episode = np.repeat(
            np.arange(len(self._pool_data), dtype=np.int32),
            np.asarray(Ts, dtype=np.int32),
        )
        self._frame_local_idx = np.concatenate(
            [np.arange(t, dtype=np.int32) for t in Ts]
        )
        # Per-frame task name (parallel to ``_frame_episode``) — used by the
        # progression-style prefilter (T5 / SAM3 task-aware scoring).
        self._frame_task = np.repeat(
            np.asarray(tasks, dtype=object),
            np.asarray(Ts, dtype=np.int32),
        )

        print(
            f"[pool-cache] loaded {self._dino_pool.shape[0]} frames across "
            f"{len(self._pool_data)} episodes (image data is lazy-mmap'd from h5)",
            flush=True,
        )

    def __del__(self):
        try:
            if getattr(self, "_cache_h5", None) is not None:
                self._cache_h5.close()
        except Exception:
            pass

    def _get_pool_cum_motion(self, dataset_stats: dict) -> np.ndarray:
        """Per-frame cumulative proprio path length (in query-normalized space).

        Same shape as ``self._proprio_pool_norm``: each pool frame holds the
        sum of ``||p[i]-p[i-1]||`` for i in 1..local_idx of its episode.
        Frame 0 of each episode is 0.0. Cached on the dataset_stats signature.
        """
        sig = (
            tuple(dataset_stats["proprio_min"].tolist()),
            tuple(dataset_stats["proprio_max"].tolist()),
        )
        if self._pool_cum_motion is not None and self._cached_phase_signature == sig:
            return self._pool_cum_motion
        proprio_pool_norm = self._get_proprio_pool_norm(dataset_stats)
        cum = np.zeros(proprio_pool_norm.shape[0], dtype=np.float32)
        for ei in range(len(self._pool_data)):
            s, e = int(self._ep_starts[ei]), int(self._ep_starts[ei + 1])
            seg = proprio_pool_norm[s:e]
            if seg.shape[0] < 2:
                continue
            step = np.linalg.norm(seg[1:] - seg[:-1], axis=1)
            cum[s:e] = np.concatenate([[0.0], np.cumsum(step)]).astype(np.float32)
        self._pool_cum_motion = cum
        self._cached_phase_signature = sig
        return cum

    def _get_proprio_pool_norm(self, dataset_stats: dict) -> np.ndarray:
        """Cache pool proprio in normalized space; rebuilt only when stats change.

        16d_quat: ``2*(raw - p_min)/(p_max - p_min) - 1`` over (N, 16).
        20d_rot6: raw_to_state(raw) → (N, 20) state, no normalization
                  (rot6 components are already bounded ~[-1, 1]).
        """
        if self.proprio_repr == "20d_rot6":
            # No dataset_stats dependence — cache once.
            if self._proprio_pool_norm is None:
                self._proprio_pool_norm = _raw_to_state_20d(self._proprio_pool_raw).astype(np.float32)
            return self._proprio_pool_norm
        sig = (
            tuple(dataset_stats["proprio_min"].tolist()),
            tuple(dataset_stats["proprio_max"].tolist()),
        )
        if self._proprio_pool_norm is not None and self._cached_stats_signature == sig:
            return self._proprio_pool_norm
        p_min = np.asarray(dataset_stats["proprio_min"], dtype=np.float32)
        p_max = np.asarray(dataset_stats["proprio_max"], dtype=np.float32)
        self._proprio_pool_norm = (
            2.0 * ((self._proprio_pool_raw - p_min) / (p_max - p_min + 1e-8)) - 1.0
        ).astype(np.float32)
        self._cached_stats_signature = sig
        return self._proprio_pool_norm

    def _normalize_history(self, dataset_stats: dict) -> np.ndarray:
        """Stack proprio history in scoring space. Output is (K, 16) for the
        16d_quat path and (K, 20) for the 20d_rot6 path. K = buffer length."""
        hist_raw = np.stack(list(self._proprio_history_raw), axis=0).astype(np.float32)
        # Apply home transform to each row before downstream conversion so the
        # legacy and supplied-history paths share the same alignment step.
        hist_raw = self._apply_home_transform_batch(hist_raw)
        if self.proprio_repr == "20d_rot6":
            return _raw_to_state_20d(hist_raw).astype(np.float32)
        p_min = np.asarray(dataset_stats["proprio_min"], dtype=np.float32)
        p_max = np.asarray(dataset_stats["proprio_max"], dtype=np.float32)
        return 2.0 * ((hist_raw - p_min) / (p_max - p_min + 1e-8)) - 1.0

    def _compute_initial_filter(
        self,
        q_dino_unit: np.ndarray,
        query_proprio_raw: np.ndarray,
        dataset_stats: dict,
    ) -> None:
        """Pick top-N episodes by start-frame DINO + proprio similarity to the
        first query of the rollout. Sets ``self._allowed_episodes`` and the
        flat-index lookup ``self._allowed_frame_idx``.
        """
        if self.proprio_repr == "20d_rot6":
            q_prop_norm = _raw_to_state_20d(
                np.asarray(query_proprio_raw, dtype=np.float32)[None, :]
            )[0].astype(np.float32)
            ep0_prop_norm = _raw_to_state_20d(self._ep_start_proprio_raw).astype(np.float32)
        else:
            p_min = np.asarray(dataset_stats["proprio_min"], dtype=np.float32)
            p_max = np.asarray(dataset_stats["proprio_max"], dtype=np.float32)
            q_prop_norm = (
                2.0 * ((np.asarray(query_proprio_raw, dtype=np.float32) - p_min) / (p_max - p_min + 1e-8)) - 1.0
            )
            ep0_prop_norm = (
                2.0 * ((self._ep_start_proprio_raw - p_min) / (p_max - p_min + 1e-8)) - 1.0
            ).astype(np.float32)

        ep_dino_score = self._ep_start_dino_unit @ q_dino_unit                        # (n_ep,)
        ep_prop_pen = np.linalg.norm(ep0_prop_norm - q_prop_norm[None, :], axis=1)    # (n_ep,)
        ep_score = self.w_dino * ep_dino_score - self.w_prop * ep_prop_pen

        n_keep = min(self.n_initial_filter, len(ep_score))
        keep = np.argpartition(-ep_score, n_keep - 1)[:n_keep]
        keep = keep[np.argsort(-ep_score[keep])].astype(np.int32)                     # ranked order

        self._allowed_episodes = keep
        # Build flat absolute indices for fast subset scoring.
        ep_set = set(int(e) for e in keep.tolist())
        mask = np.fromiter(
            (int(e) in ep_set for e in self._frame_episode), dtype=bool, count=len(self._frame_episode)
        )
        self._allowed_frame_idx = np.flatnonzero(mask).astype(np.int32)

        print(
            f"[retrieval] initial-scene filter: kept {n_keep}/{len(self._pool_data)} episodes "
            f"({len(self._allowed_frame_idx)} frames). top-3 by start-score:",
            flush=True,
        )
        for ei in keep[:3]:
            ep = self._pool_data[int(ei)]
            print(
                f"  - {ep['split']}/episode{ep['ep_num']}  "
                f"start-dino={ep_dino_score[ei]:+.3f}  "
                f"start-prop-L2={ep_prop_pen[ei]:.3f}",
                flush=True,
            )

    def _proprio_delta_l2_episodes(
        self,
        proprio_pool_norm: np.ndarray,
        q_hist_norm: np.ndarray,
        allowed_episodes: np.ndarray,
    ) -> np.ndarray:
        """Same semantics as _proprio_delta_l2 but only fills entries for
        frames in ``allowed_episodes``. All other positions get +inf.
        """
        N = proprio_pool_norm.shape[0]
        K = q_hist_norm.shape[0]
        out = np.full(N, np.inf, dtype=np.float32)

        if K < 2:
            # No motion direction available from query — neutralize the
            # proprio term for all allowed frames so DINO drives the score.
            for ei in allowed_episodes:
                s, e = int(self._ep_starts[int(ei)]), int(self._ep_starts[int(ei) + 1])
                out[s:e] = 0.0
            return out

        Kd = K - 1
        q_delta = q_hist_norm[1:] - q_hist_norm[:-1]                            # (Kd, D)
        neutral = float(np.linalg.norm(q_delta, axis=1).mean())                 # frame-0 fallback

        for ei in allowed_episodes:
            s, e = int(self._ep_starts[int(ei)]), int(self._ep_starts[int(ei) + 1])
            out[s:e] = self._delta_l2_for_segment(
                proprio_pool_norm[s:e], q_delta, Kd, neutral
            )
        return out

    def _proprio_delta_l2(
        self, proprio_pool_norm: np.ndarray, q_hist_norm: np.ndarray
    ) -> np.ndarray:
        """Direction-of-motion penalty: uniform mean L2 between query proprio
        deltas and pool proprio deltas in normalized space.

        Query history (K, 16) yields K-1 deltas. For each pool frame n with
        local episode index k, compare against the K-1 deltas of pool frames
        ending at n (truncated at episode start when k < K-1). Cross-episode
        windows are avoided by walking episodes.

        When the query has fewer than 2 frames, returns zeros (no motion
        signal yet — the DINO term carries the score).
        """
        N = proprio_pool_norm.shape[0]
        K = q_hist_norm.shape[0]
        if K < 2:
            return np.zeros(N, dtype=np.float32)

        Kd = K - 1
        q_delta = q_hist_norm[1:] - q_hist_norm[:-1]
        neutral = float(np.linalg.norm(q_delta, axis=1).mean())

        diffs = np.empty(N, dtype=np.float32)
        ep_starts = np.concatenate([[0], np.cumsum([e["T"] for e in self._pool_data])])
        for ei in range(len(self._pool_data)):
            s, e = int(ep_starts[ei]), int(ep_starts[ei + 1])
            diffs[s:e] = self._delta_l2_for_segment(
                proprio_pool_norm[s:e], q_delta, Kd, neutral
            )
        return diffs

    @staticmethod
    def _delta_l2_for_segment(
        seg: np.ndarray, q_delta: np.ndarray, Kd: int, neutral: float
    ) -> np.ndarray:
        """Per-frame uniform-mean L2 between query deltas and the last Kd pool
        deltas ending at each frame of one episode segment.

        Frames with fewer than Kd preceding deltas use the available subset
        (truncated). Frame 0 (no preceding delta) gets ``neutral`` so it
        isn't artificially preferred by zero penalty.
        """
        from numpy.lib.stride_tricks import sliding_window_view

        T_e, D = seg.shape
        if T_e < 2:
            return np.full(T_e, neutral, dtype=np.float32)

        seg_delta = seg[1:] - seg[:-1]                                          # (T_e - 1, D)
        # Left-pad with Kd zero-deltas so the sliding window over the resulting
        # (T_e - 1 + Kd, D) array covers exactly T_e frames, with frame k's
        # window = seg_d_padded[k : k + Kd]. Padding entries get masked out.
        seg_d_padded = np.concatenate(
            [np.zeros((Kd, D), dtype=seg_delta.dtype), seg_delta], axis=0
        )
        win = sliding_window_view(seg_d_padded, Kd, axis=0).transpose(0, 2, 1)[:T_e]
        l2 = np.linalg.norm(win - q_delta[None, :, :], axis=2)                  # (T_e, Kd)

        eff_Kd = np.minimum(Kd, np.arange(T_e))                                 # (T_e,)
        col = np.arange(Kd)
        mask = col[None, :] >= (Kd - eff_Kd[:, None])                           # (T_e, Kd)
        masked_l2 = np.where(mask, l2, 0.0)
        denom = np.maximum(eff_Kd, 1).astype(np.float32)
        per_frame = (masked_l2.sum(axis=1) / denom).astype(np.float32)

        zero_mask = eff_Kd == 0
        if zero_mask.any():
            per_frame[zero_mask] = neutral
        return per_frame

    # ───────────────────── forward-bias helper ──────────────────────────────

    def _compute_forward_bias(self, abs_frames: np.ndarray) -> np.ndarray:
        """Per-candidate soft forward-progress bonus inside any episode we've
        already picked from.

        ``bonus = w_fwd * tanh((local_idx - last_picked[ep]) / chunk_size)``

        Episodes we haven't picked from yet get 0 (free to enter). Bias is
        bounded by ±w_fwd so it tips flat score landscapes toward forward
        progress without overwhelming larger DINO/proprio differences.
        """
        n = len(abs_frames)
        if self._forward_bias_weight <= 0.0 or not self._last_picked_local_idx_per_ep:
            return np.zeros(n, dtype=np.float32)
        ep_per = self._frame_episode[abs_frames]
        local_per = self._frame_local_idx[abs_frames].astype(np.float32)
        # Per-ep last-picked lookup vector; NaN where we haven't picked yet.
        last_arr = np.full(len(self._pool_data), np.nan, dtype=np.float64)
        for ep_idx, idx in self._last_picked_local_idx_per_ep.items():
            last_arr[int(ep_idx)] = float(idx)
        last_per = last_arr[ep_per]
        valid = ~np.isnan(last_per)
        delta = local_per - np.where(valid, last_per, 0.0).astype(np.float32)
        bonus = self._forward_bias_weight * np.tanh(delta / float(self.chunk_size))
        return np.where(valid, bonus, 0.0).astype(np.float32)

    # ───────────────────── advancement-brake helper ─────────────────────────
    def _compute_advance_brake(self, abs_frames: np.ndarray) -> np.ndarray:
        """Per-candidate penalty when local_idx exceeds (last_picked + max_step)
        within the same episode.

        Linear penalty on the excess:
            excess = max(0, local_idx - last_picked - max_step)
            brake  = brake_weight * excess
        Episodes we haven't picked from yet are exempt (return 0). Returns a
        non-negative array meant to be SUBTRACTED from the score.
        """
        n = len(abs_frames)
        if self._advance_brake_weight <= 0.0 or not self._last_picked_local_idx_per_ep:
            return np.zeros(n, dtype=np.float32)
        ep_per = self._frame_episode[abs_frames]
        local_per = self._frame_local_idx[abs_frames].astype(np.float32)
        last_arr = np.full(len(self._pool_data), np.nan, dtype=np.float64)
        for ep_idx, idx in self._last_picked_local_idx_per_ep.items():
            last_arr[int(ep_idx)] = float(idx)
        last_per = last_arr[ep_per]
        valid = ~np.isnan(last_per)
        excess = (
            local_per
            - np.where(valid, last_per, 0.0).astype(np.float32)
            - float(self._advance_brake_max_step)
        )
        excess = np.maximum(0.0, excess)
        penalty = self._advance_brake_weight * excess
        return np.where(valid, penalty.astype(np.float32), 0.0).astype(np.float32)

    # ───────────────────── per-chunk ep-level biases ───────────────────────
    def _per_chunk_ep_biases(self, abs_frames: np.ndarray) -> np.ndarray:
        """Add per-pool-ep T5 + SAM3 biases to per-chunk score.

        Mirrors ``render_progression_runtime_video.py`` per-chunk formula::
            scores += w_t5 * t5_per_ep[ep] + w_sam3_eff * sam3_eff[ep]

        At CHUNK 0 (first action call after reset, ``_n_acts_since_reset==0``)
        we substitute the xy-emphasized SAM3 (``_sam3_xy_per_ep``) weighted
        by ``chunk0_w_sam3_xy`` — anchors the first ep choice on object
        image-position match. For chunks k>=1 we use the default greedy
        SAM3 (``_sam3_per_ep``) weighted by ``w_sam3``.

        No-op on exact-match pools (subclass sets ``_exact_pool=True`` and
        skips per-ep array population in the prefilter short-circuit).
        """
        n = int(abs_frames.shape[0])
        if getattr(self, "_exact_pool", False):
            return np.zeros(n, dtype=np.float32)
        t5_per_ep = getattr(self, "_t5_per_ep", None)
        sam3_per_ep = getattr(self, "_sam3_per_ep", None)
        sam3_xy_per_ep = getattr(self, "_sam3_xy_per_ep", None)
        if t5_per_ep is None and sam3_per_ep is None and sam3_xy_per_ep is None:
            return np.zeros(n, dtype=np.float32)

        is_chunk0 = int(getattr(self, "_n_acts_since_reset", 1)) == 0
        w_t5 = float(getattr(self, "w_t5", 0.0))
        if is_chunk0:
            w_sam3 = float(getattr(self, "chunk0_w_sam3_xy", 0.0))
            sam3_array = sam3_xy_per_ep if sam3_xy_per_ep is not None else sam3_per_ep
        else:
            w_sam3 = float(getattr(self, "w_sam3", 0.0))
            sam3_array = sam3_per_ep
        if w_t5 == 0.0 and w_sam3 == 0.0:
            return np.zeros(n, dtype=np.float32)

        ep_per = self._frame_episode[abs_frames]
        out = np.zeros(n, dtype=np.float32)
        if t5_per_ep is not None and w_t5 != 0.0:
            out = out + w_t5 * t5_per_ep[ep_per]
        if sam3_array is not None and w_sam3 != 0.0:
            out = out + w_sam3 * sam3_array[ep_per]
        return out.astype(np.float32)

    # ───────────────────── home-pose transform helpers ──────────────────────

    def _home_transform_alphas(self, proprio_raw: np.ndarray) -> tuple[float, float]:
        """Return (alpha_L, alpha_R) for a single (16,) raw proprio."""
        if not self._home_transform_enabled:
            return 0.0, 0.0
        p = self._home_transform_p
        d_L = float(np.linalg.norm(proprio_raw[:8] - self._aloha_home[:8]))
        d_R = float(np.linalg.norm(proprio_raw[8:] - self._aloha_home[8:]))
        a_L = float(np.exp(-((d_L / max(self._sigma_L, 1e-8)) ** p)))
        a_R = float(np.exp(-((d_R / max(self._sigma_R, 1e-8)) ** p)))
        return a_L, a_R

    def _apply_home_transform(self, proprio_raw: np.ndarray) -> np.ndarray:
        """Per-arm super-Gaussian-decayed shift toward UR5 home space.

        For each arm independently:
            d   = ||q_arm - aloha_home_arm||
            α   = exp(-(d/σ)^p)              # p=2 Gaussian, p=4 super-Gaussian
            q'  = q + α * (ur5_home - aloha_home)   on that arm only

        Mid-task arms are far from home → α≈0 → identity. Near home (start,
        return) → α≈1 → full shift to UR5 space so the L2 against ur5 pool
        frames is well-defined. p>2 gives a flat near-home plateau and a
        sharper falloff so the transform engages cleanly inside d<σ and
        vanishes cleanly past it.
        """
        if not self._home_transform_enabled:
            return np.asarray(proprio_raw, dtype=np.float32)
        p = np.asarray(proprio_raw, dtype=np.float32).copy()
        a_L, a_R = self._home_transform_alphas(p)
        p[:8] += a_L * self._home_offset[:8]
        p[8:] += a_R * self._home_offset[8:]
        return p

    def _apply_home_transform_batch(self, proprio_raw: np.ndarray) -> np.ndarray:
        """Vectorized variant for an (M, 16) history buffer."""
        if not self._home_transform_enabled:
            return np.asarray(proprio_raw, dtype=np.float32)
        exp_p = self._home_transform_p
        p = np.asarray(proprio_raw, dtype=np.float32).copy()
        d_L = np.linalg.norm(p[:, :8] - self._aloha_home[:8], axis=1)
        d_R = np.linalg.norm(p[:, 8:] - self._aloha_home[8:], axis=1)
        a_L = np.exp(-((d_L / max(self._sigma_L, 1e-8)) ** exp_p)).astype(np.float32)
        a_R = np.exp(-((d_R / max(self._sigma_R, 1e-8)) ** exp_p)).astype(np.float32)
        p[:, :8] += a_L[:, None] * self._home_offset[:8]
        p[:, 8:] += a_R[:, None] * self._home_offset[8:]
        return p

    # ───────────────────────────── diagnostic dump ──────────────────────────

    def _dump_debug(
        self,
        query_head_image: np.ndarray,
        query_proprio: np.ndarray,
        proprio_pool_norm_w: np.ndarray,
        q_curr_norm_w: np.ndarray,
        scores: np.ndarray,
        dino_score: np.ndarray,
        prop_penalty: np.ndarray,
        curr_penalty: np.ndarray,
        phase_penalty: np.ndarray,
        picked_local_idx: int,
        picked_abs: int,
        allowed: Optional[np.ndarray],
    ) -> None:
        """Append per-step retrieval diagnostics to ``debug_dump_dir``.

        Layout::
            <dir>/retrieval.jsonl   one JSON line per call (full breakdown)
            <dir>/query_head/<step:06d>.png
            <dir>/pool_head/<split>_ep<ep>_f<frame>.png  (one per unique frame)
        """
        import cv2

        step = int(self._debug_step_count)

        # ---- top-K diagnostic ---------------------------------------------
        K = min(self.debug_dump_top_k, int(scores.shape[0]))
        top = np.argpartition(-scores, K - 1)[:K]
        top = top[np.argsort(-scores[top])]
        if allowed is not None:
            top_abs = allowed[top]
        else:
            top_abs = top
        top_breakdown = []
        for local_i, abs_i in zip(top.tolist(), top_abs.tolist()):
            local_i = int(local_i); abs_i = int(abs_i)
            ep_index = int(self._frame_episode[abs_i])
            local_idx = int(self._frame_local_idx[abs_i])
            ep = self._pool_data[ep_index]
            top_breakdown.append({
                "split": ep["split"],
                "ep_num": int(ep["ep_num"]),
                "frame": local_idx,
                "score": float(scores[local_i]),
                "dino_score": float(dino_score[local_i]),
                "prop_penalty": float(prop_penalty[local_i]),
                "curr_penalty": float(curr_penalty[local_i]),
                "phase_penalty": float(phase_penalty[local_i]),
                # Sum-of-weighted-contributions (sign-aligned with the scoring
                # formula) so consumers can verify component magnitudes.
                "score_check": float(
                    self.w_dino * dino_score[local_i]
                    - self.w_prop * prop_penalty[local_i]
                    - self.w_prop_curr * curr_penalty[local_i]
                    - self.phase_alignment_weight * phase_penalty[local_i]
                ),
                # Raw 16-d proprio so consumers can eyeball what each
                # candidate actually looks like next to the query.
                "proprio_raw": self._proprio_pool_raw[abs_i].tolist(),
            })

        # ---- per-dim weighted diff (picked vs query) -----------------------
        picked_norm_w = proprio_pool_norm_w[picked_abs]
        per_dim_w_diff = (q_curr_norm_w - picked_norm_w).tolist()

        # ---- save query head image -----------------------------------------
        try:
            qpath = os.path.join(
                self.debug_dump_dir, "query_head", f"{step:06d}.png"
            )
            cv2.imwrite(qpath, cv2.cvtColor(query_head_image, cv2.COLOR_RGB2BGR))
        except Exception as e:
            print(f"[retrieval/debug-dump] query_head save failed: {e}", flush=True)

        # ---- save picked pool head image (once per unique (ep, frame)) -----
        ep_index = int(self._frame_episode[picked_abs])
        local_idx = int(self._frame_local_idx[picked_abs])
        key = (ep_index, local_idx)
        if key not in self._debug_pool_seen:
            try:
                ep = self._pool_data[ep_index]
                abs_start = int(self._ep_starts[ep_index])
                pool_img = np.asarray(self._all_head[abs_start + local_idx])
                fname = f"{ep['split']}_ep{int(ep['ep_num'])}_f{local_idx}.png"
                ppath = os.path.join(self.debug_dump_dir, "pool_head", fname)
                cv2.imwrite(ppath, cv2.cvtColor(pool_img, cv2.COLOR_RGB2BGR))
                self._debug_pool_seen.add(key)
            except Exception as e:
                print(f"[retrieval/debug-dump] pool_head save failed: {e}", flush=True)

        # ---- JSONL record --------------------------------------------------
        q_raw = np.asarray(query_proprio, dtype=np.float32)
        record = {
            "step": step,
            "q_proprio_raw": q_raw.tolist(),
            "q_curr_norm_w": q_curr_norm_w.tolist(),
            "q_cum_motion": float(self._q_cum_motion_since_reset),
            "history_len": len(self._proprio_history_raw),
            "n_allowed": int(allowed.shape[0]) if allowed is not None else int(scores.shape[0]),
            "picked": top_breakdown[0],
            "top": top_breakdown,
            "per_dim_weighted_diff": per_dim_w_diff,
        }
        if self._home_transform_enabled:
            a_L, a_R = self._home_transform_alphas(q_raw)
            q_shifted = self._apply_home_transform(q_raw)
            record["home_transform"] = {
                "alpha_L": a_L,
                "alpha_R": a_R,
                "q_proprio_shifted": q_shifted.tolist(),
                "d_L_to_aloha_home": float(np.linalg.norm(q_raw[:8] - self._aloha_home[:8])),
                "d_R_to_aloha_home": float(np.linalg.norm(q_raw[8:] - self._aloha_home[8:])),
            }
        try:
            with open(os.path.join(self.debug_dump_dir, "retrieval.jsonl"), "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[retrieval/debug-dump] jsonl append failed: {e}", flush=True)
