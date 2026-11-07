"""Single source of truth for the real_robot retrieval pair pipeline.

Both the offline pair builder (``scripts/build_retrieval_pair_h5.py``) and the
online deploy server (``deploy_realrobot_ret.py``) consume the same active-arm
classification, sub-bucket layout, idle-slot override, KD-tree feature, and
DINO episode-lock logic. Anything that affects which pool chunk gets matched
(or what residual the model learns) lives here.

The 20-D action/state layout is fixed:
    L_xyz (3) | L_rot6d (6) | L_grip (1) | R_xyz (3) | R_rot6d (6) | R_grip (1)
"""
from __future__ import annotations

import glob
import io
import os
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np
from scipy.spatial import cKDTree


# ─── constants ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 16
STRIDE = 1
THRESH_STD = 0.005           # 5 mm carry-over threshold for active-arm decision
TOP_K = 1
POOL_EXCLUDE_TAIL_FRAMES = 30   # drop the human "return-home" tail (~3 s @ 10 Hz)

TASKS = ("open_cabinet", "place_bottle_in_plastic_box", "close_cabinet")
TASK_FIRST_ACTIVE: dict[str, str] = {
    "open_cabinet": "R",
    "place_bottle_in_plastic_box": "L",
    "close_cabinet": "R",
}

# 20-D layout offsets
SLOT_L = slice(0, 10)
SLOT_R = slice(10, 20)
SLOT_L_XYZ = slice(0, 3)
SLOT_R_XYZ = slice(10, 13)
SLOT_L_GRIP = 9
SLOT_R_GRIP = 19

# Matching weights — score = W_XYZ * xyz_dist + W_ACTION * action_chunk_dist
ACTION_FEATURE_DIM = CHUNK_SIZE * 3      # 16 frames * 3 xyz
W_XYZ = 1.0
W_ACTION = 0.1 / float(np.sqrt(ACTION_FEATURE_DIM))    # ≈ 0.0144


# ─── data types ────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """One stride-1 chunk's bookkeeping. Full proprio/action/jpegs are
    streamed from h5 on demand via :func:`load_chunk_arrays`."""

    h5_path: str
    episode_basename: str
    task: str
    chunk_start: int
    proprio_start: np.ndarray        # (20,) float32 — chunk_start frame proprio
    active_arm: str                  # 'L' or 'R'
    current_grip: int                # active arm's grip at chunk_start (0 or 1)
    action_xyz_chunk: np.ndarray     # (16, 3) float32 — active arm xyz across the 16 action frames


# ─── active arm + grip classification ──────────────────────────────────────


def classify_active_arm(
    chunk_proprio: np.ndarray,
    task: str,
    prev_active: Optional[str] = None,
) -> tuple[str, int]:
    """Apply the train-time active-arm rule to a (16, 20) chunk of proprio.

    Returns ``(active_arm, current_grip)`` where ``current_grip`` is the
    chunk-start grip (0/1) of the active arm.
    """
    L_xyz = chunk_proprio[:, SLOT_L_XYZ]
    R_xyz = chunk_proprio[:, SLOT_R_XYZ]
    L_std = float(L_xyz.std(axis=0).max())
    R_std = float(R_xyz.std(axis=0).max())
    if max(L_std, R_std) < THRESH_STD:
        active = prev_active or TASK_FIRST_ACTIVE[task]
    else:
        active = "L" if L_std > R_std else "R"
    grip_idx = SLOT_L_GRIP if active == "L" else SLOT_R_GRIP
    current_grip = 1 if float(chunk_proprio[0, grip_idx]) >= 0.5 else 0
    return active, current_grip


def active_arm_xyz_slice(active_arm: str) -> slice:
    return SLOT_L_XYZ if active_arm == "L" else SLOT_R_XYZ


def idle_arm_slot(active_arm: str) -> slice:
    return SLOT_R if active_arm == "L" else SLOT_L


# ─── chunk enumeration ─────────────────────────────────────────────────────


def classify_chunks(
    h5_files: list[str],
    task: str,
    exclude_tail_frames: int = 0,
) -> list[Chunk]:
    """Walk every (h5, chunk_start) pair stride-1 and classify active arm +
    current grip + active-arm xyz across the chunk's 16 *action* frames
    (used as a soft matching component at pair-build time).

    ``exclude_tail_frames`` clips the last N frames per episode before
    chunking. The pair builder uses ``POOL_EXCLUDE_TAIL_FRAMES`` on the
    human pool side to drop return-home motion; the query side passes 0.
    """
    out: list[Chunk] = []
    for fp in h5_files:
        with h5py.File(fp, "r") as f:
            proprio = f["proprio"][:]
            action = f["action"][:]
        T_full = len(proprio)
        T = max(0, T_full - exclude_tail_frames)
        if T < CHUNK_SIZE:
            continue
        ep_basename = os.path.basename(fp)
        prev_active = TASK_FIRST_ACTIVE[task]
        for cs in range(0, T - CHUNK_SIZE + 1, STRIDE):
            chunk_proprio = proprio[cs : cs + CHUNK_SIZE]
            active, current_grip = classify_active_arm(chunk_proprio, task, prev_active)
            xyz_slice = active_arm_xyz_slice(active)
            action_xyz_chunk = action[cs : cs + CHUNK_SIZE, xyz_slice].astype(np.float32)
            out.append(Chunk(
                h5_path=fp,
                episode_basename=ep_basename,
                task=task,
                chunk_start=cs,
                proprio_start=proprio[cs].astype(np.float32),
                active_arm=active,
                current_grip=current_grip,
                action_xyz_chunk=action_xyz_chunk,
            ))
            prev_active = active
    return out


# ─── KD-tree feature ───────────────────────────────────────────────────────


def chunk_feature(proprio_start_xyz_3d: np.ndarray,
                  action_xyz_chunk_or_zeros: np.ndarray) -> np.ndarray:
    """Build the 51-D weighted feature vector used by the pool KD-tree.

    Pass an all-zeros (16, 3) array for ``action_xyz_chunk_or_zeros`` at
    inference time when future actions are unknown — pool entries are
    indexed in the same metric space, so the xyz term dominates the match.
    """
    xyz_part = W_XYZ * proprio_start_xyz_3d.astype(np.float32)
    action_part = W_ACTION * action_xyz_chunk_or_zeros.reshape(-1).astype(np.float32)
    return np.concatenate([xyz_part, action_part])


def build_pool_index(pool_chunks: list[Chunk]) -> dict[tuple[str, str, int], dict]:
    """Bucket pool chunks by (task, active_arm, current_grip) and build one
    KD-tree per bucket over :func:`chunk_feature` vectors.

    Returned dict shape::

        {(task, arm, grip): {
            "tree": cKDTree, "items": list[Chunk], "xyz_slice": slice
        }}
    """
    buckets: dict[tuple[str, str, int], list[Chunk]] = {}
    for c in pool_chunks:
        buckets.setdefault((c.task, c.active_arm, c.current_grip), []).append(c)
    out: dict[tuple[str, str, int], dict] = {}
    for key, items in buckets.items():
        xyz_slice = active_arm_xyz_slice(key[1])
        pts = np.stack([
            chunk_feature(c.proprio_start[xyz_slice], c.action_xyz_chunk)
            for c in items
        ], axis=0)
        out[key] = {"tree": cKDTree(pts), "items": items, "xyz_slice": xyz_slice}
    return out


def apply_grip_fallback(
    pool_index: dict[tuple[str, str, int], dict],
    requested_keys: list[tuple[str, str, int]],
) -> set[tuple[str, str, int]]:
    """Patch ``pool_index`` so that for every key in ``requested_keys`` we
    have a non-empty bucket; if the (task, arm, grip) bucket is missing we
    point it at (task, arm, 1-grip). Returns the set of buckets that hit
    the fallback (for logging)."""
    fell_back: set[tuple[str, str, int]] = set()
    for key in requested_keys:
        if key in pool_index:
            continue
        task, arm, grip = key
        alt = (task, arm, 1 - grip)
        if alt not in pool_index:
            raise RuntimeError(f"no pool at all for ({task}, {arm}); neither {key} nor {alt}")
        pool_index[key] = pool_index[alt]
        fell_back.add(key)
    return fell_back


# ─── per-chunk I/O ─────────────────────────────────────────────────────────


def load_chunk_arrays(h5_path: str, chunk_start: int) -> tuple[np.ndarray, np.ndarray, list[bytes]]:
    """Load (proprio, action, image_jpegs) for one chunk.

    proprio: (16, 20) float32
    action:  (16, 20) float32
    jpegs:   list[bytes] length 16  (raw JPEG payloads from third_eye/rgb)
    """
    with h5py.File(h5_path, "r") as f:
        proprio = f["proprio"][chunk_start : chunk_start + CHUNK_SIZE]
        action = f["action"][chunk_start : chunk_start + CHUNK_SIZE]
        rgb = f["observation/third_eye/rgb"][chunk_start : chunk_start + CHUNK_SIZE]
        jpegs = [bytes(b) for b in rgb]
    return proprio, action, jpegs


# ─── idle-slot override ────────────────────────────────────────────────────


def apply_idle_override(
    cur_proprio: np.ndarray,
    cur_action: np.ndarray,
    ret_proprio: np.ndarray,
    ret_action: np.ndarray,
    active_arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero out the idle arm's residual by broadcasting the chunk-start
    idle-arm state into ``cur_action[:, idle]``, ``ret_proprio[idle]``, and
    ``ret_action[:, idle]``.

    Inputs are NOT mutated — returns copies.
    """
    idle_slot = idle_arm_slot(active_arm)
    idle_state = cur_proprio[idle_slot]
    cur_action = cur_action.copy()
    ret_proprio = ret_proprio.copy()
    ret_action = ret_action.copy()
    cur_action[:, idle_slot] = idle_state
    ret_proprio[idle_slot] = idle_state
    ret_action[:, idle_slot] = idle_state
    return cur_action, ret_proprio, ret_action


# ─── filesystem helpers ────────────────────────────────────────────────────


def list_episode_h5(root: str, task: str, embodiment: str = "bg2_bimanual") -> list[str]:
    return sorted(glob.glob(f"{root}/{task}/{embodiment}/data/*.hdf5"))


# ─── DINO first-frame episode lock ─────────────────────────────────────────


class DinoCLSExtractor:
    """timm DINOv2 ViT-B/14 → 768-D CLS embedding for a single RGB frame.

    Shared by the deploy server and the pair builder so that the episode-
    lock decision is identical at train and test time.
    """

    def __init__(self, model_name: str = "vit_base_patch14_dinov2.lvd142m",
                 input_size: int = 224, amp_dtype: str = "bfloat16",
                 device: Optional[str] = None):
        import torch
        import timm
        from timm.data import create_transform, resolve_model_data_config

        self.input_size = input_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.amp_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[amp_dtype]
        self._torch = torch
        self.model = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=input_size,
        ).eval().to(self.device)
        data_cfg = resolve_model_data_config(self.model)
        data_cfg["input_size"] = (3, input_size, input_size)
        self.transform = create_transform(**data_cfg, is_training=False)

    def embed(self, frame: np.ndarray) -> np.ndarray:
        from PIL import Image
        torch = self._torch
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"expected (H,W,3) uint8, got {frame.shape} {frame.dtype}")
        x = self.transform(Image.fromarray(frame)).unsqueeze(0).to(
            self.device, non_blocking=True
        )
        with torch.no_grad():
            if str(self.device).startswith("cuda") and self.amp_dtype != torch.float32:
                with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                    feat = self.model.forward_features(x)
            else:
                feat = self.model.forward_features(x)
        cls = feat[:, 0, :] if feat.ndim == 3 else feat
        return cls[0].float().cpu().numpy()


def _read_first_frame_rgb(h5_path: str) -> np.ndarray:
    from PIL import Image
    with h5py.File(h5_path, "r") as h5:
        first_jpeg = bytes(h5["observation/third_eye/rgb"][0])
    return np.asarray(Image.open(io.BytesIO(first_jpeg)).convert("RGB"))


def build_dino_episode_cache(
    episode_paths: list[str],
    cache_path: Optional[str] = None,
    extractor: Optional[DinoCLSExtractor] = None,
) -> tuple[np.ndarray, list[str]]:
    """Embed each episode's first frame with DINO CLS, return unit-norm
    matrix (N, 768) + the parallel episode path list.

    If ``cache_path`` exists and its stored ``paths`` matches ``episode_paths``
    exactly, the cache is reused. Otherwise (re)computes and saves.
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=False)
        cached = [str(p) for p in data["paths"].tolist()]
        if cached == episode_paths:
            return data["feats"].astype(np.float32), cached

    ext = extractor or DinoCLSExtractor()
    feats: list[np.ndarray] = []
    for fp in episode_paths:
        img = _read_first_frame_rgb(fp)
        feats.append(ext.embed(img).astype(np.float32))
    feats_arr = np.stack(feats, axis=0)
    norms = np.linalg.norm(feats_arr, axis=1, keepdims=True)
    unit = (feats_arr / np.clip(norms, 1e-8, None)).astype(np.float32)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.savez(cache_path, feats=unit, paths=np.array(episode_paths))
    return unit, list(episode_paths)


def pick_top_k_episodes_by_dino(
    query_first_frame_rgb: np.ndarray,
    pool_unit: np.ndarray,
    pool_paths: list[str],
    k: int,
    extractor: Optional[DinoCLSExtractor] = None,
) -> list[tuple[str, float]]:
    """Cosine-sim top-K episode pick. Returns list of (path, sim) sorted by
    similarity desc."""
    ext = extractor or DinoCLSExtractor()
    q = ext.embed(query_first_frame_rgb).astype(np.float32)
    q_unit = q / max(float(np.linalg.norm(q)), 1e-8)
    sims = pool_unit @ q_unit
    order = np.argsort(-sims)[:k]
    return [(pool_paths[int(i)], float(sims[int(i)])) for i in order]


def nearest_chunk_in_episode(
    ep_chunks: list[Chunk],
    active_arm: str,
    current_grip: int,
    query_xyz_3d: np.ndarray,
    grip_fallback: bool = True,
    min_chunk_start: int = 0,
) -> Optional[tuple[Chunk, float]]:
    """Pick the xyz-nearest chunk from a single human episode's chunks
    that matches (active_arm, current_grip) **and has chunk_start >= min_chunk_start**.

    The ``min_chunk_start`` argument enforces monotonic (causal) retrieval —
    once the query side has consumed up to human-chunk t, subsequent queries
    cannot go back to t' < t inside the same locked episode. This eliminates
    the "human timeline jumps backwards" artifact observed when raw xyz
    nearest picks a chunk earlier in time.

    Returns (chunk, dist_m) or None if no candidate (even after grip
    fallback OR after relaxing the monotonic constraint).
    """
    def _filter(arr: list[Chunk]) -> list[Chunk]:
        return [c for c in arr if c.chunk_start >= min_chunk_start]

    primary = [c for c in ep_chunks
               if c.active_arm == active_arm and c.current_grip == current_grip]
    candidates = _filter(primary)
    if not candidates and grip_fallback:
        # try grip-flipped within monotonic constraint
        alt = [c for c in ep_chunks if c.active_arm == active_arm]
        candidates = _filter(alt)
    if not candidates:
        # monotonic constraint produced empty — relax it (rare, e.g. when the
        # locked episode ran out of forward chunks for this arm).
        candidates = primary if primary else [c for c in ep_chunks if c.active_arm == active_arm]
        if not candidates:
            return None
    xyz_slice = active_arm_xyz_slice(active_arm)
    dists = np.array(
        [float(np.linalg.norm(query_xyz_3d - c.proprio_start[xyz_slice]))
         for c in candidates]
    )
    best = int(np.argmin(dists))
    return candidates[best], float(dists[best])
