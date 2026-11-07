"""RealRobot retrieval-augmented dataset.

Consumes the pre-built pairs.hdf5 from
``cosmos_policy/scripts/build_retrieval_pair_h5.py`` and emits one batch
sample per (robot query chunk, top-1 human retrieval) pair.

Latent layout (state_t = 8, chunk_duration = 1 + 7 * 4 = 29):

    latent 0 : blank sentinel
    latent 1 : ret image          (4 frames sampled from the 16-frame chunk)
    latent 2 : ret action         (blank image; 16 actions injected)
    latent 3 : cur image          (4 frames sampled from the robot chunk)
    latent 4 : cur proprio        (blank image; 1 proprio injected)
    latent 5 : pred action        (blank image; model generates)
    latent 6 : future image       (4 copies of the chunk-end frame; model fills)
    latent 7 : future proprio     (blank image; model fills)

min_num_conditional_frames = 5  (latents 0-4 are conditioning).

Compared to PushT/RoboTwin retrieval datasets:
  * Single camera (third_eye), so only 3 image latents.
  * ret_proprio slot removed — trajectory information is captured by ret_action.
  * Idle-arm slot is already overridden in the pair h5 (cur_proprio[idle_slot]
    broadcast into both cur_action and ret_action), so the residual is
    automatically 0 on idle dimensions and the model never sees a learning
    signal for the idle hand.
"""
from __future__ import annotations

import io
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_utils import preprocess_image


DEFAULT_PAIR_H5 = "/mnt/ddn/dataset/real_robot/h5_retrieval/pairs.hdf5"
DEFAULT_T5_EMB = "/mnt/ddn/dataset/real_robot/h5/t5_embeddings.pkl"
DEFAULT_DATASET_STATS = (
    "/mnt/ddn/dataset/real_robot/h5/_cosmos_cache/"
    "bg2_bimanual__2tasks_e75b909e/dataset_statistics.json"
)
DEFAULT_DELTA_STATS_DIR = "/mnt/ddn/dataset/real_robot/h5_retrieval"

CHUNK_SIZE = 16
NUM_DUPLICATES = 4

# task → human-readable command (matches t5_embeddings.pkl keys)
TASK_TO_COMMAND = {
    "open_cabinet": "open cabinet",
    "place_bottle_in_plastic_box": "place bottle in plastic box",
}

# Mixed action representation (per user spec 2026-05-24):
#   xyz dims  → FRAME (frame-to-frame delta, chunk-internal prepend self)
#               residual = cur_xyz_fd - ret_xyz_fd  → normalize via delta_stats
#   rot dims  → ABS, direct prediction (no residual)
#               target = cur_rot                    → normalize via dataset_stats
#   grip dims → ABS, direct prediction (no residual)
#               target = cur_grip                   → normalize via dataset_stats
# 20-D layout: L_xyz(3) | L_rot6d(6) | L_grip(1) | R_xyz(3) | R_rot6d(6) | R_grip(1)
XYZ_DIMS = np.array([0, 1, 2, 10, 11, 12], dtype=np.int64)
NON_XYZ_DIMS = np.array([3, 4, 5, 6, 7, 8, 9,
                         13, 14, 15, 16, 17, 18, 19], dtype=np.int64)
# Within NON_XYZ_DIMS:
#   ROT_DIMS  (L_rot6d + R_rot6d) — robot stats are nearly-zero range here,
#     so human ret rot saturates after normalize; we override with cur_proprio.
#   GRIP_DIMS — binary {0,1}, same distribution on human and robot; keep
#     the human ret value (real retrieval signal).
ROT_DIMS = np.array([3, 4, 5, 6, 7, 8,
                     13, 14, 15, 16, 17, 18], dtype=np.int64)
GRIP_DIMS = np.array([9, 19], dtype=np.int64)


def _action_to_mixed(action: np.ndarray, xyz_dims: np.ndarray) -> np.ndarray:
    """Convert a (16, D) raw action chunk to mixed representation:
    xyz dims → chunk-internal frame delta, non-xyz dims → ABS as-is.
    Frame 0's xyz delta is 0 (prepend self).
    `xyz_dims` are indices into the trailing dim (10D or 20D context)."""
    out = action.copy().astype(np.float32)
    xyz = action[:, xyz_dims]
    out[:, xyz_dims] = np.diff(xyz, axis=0, prepend=xyz[0:1])
    return out

# 4 image frames sampled from a 16-frame chunk for image latents
SAMPLE_FRAMES_4 = [0, 5, 10, 15]


def _decode_jpeg(u8_arr: np.ndarray) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(u8_arr.tobytes())).convert("RGB"))


def _decode_chunk_4(jpegs_per_frame, indices=SAMPLE_FRAMES_4) -> np.ndarray:
    """Decode `len(indices)` JPEGs from a 16-frame chunk → (n, H, W, 3) uint8."""
    return np.stack([_decode_jpeg(jpegs_per_frame[i]) for i in indices], axis=0)


def _duplicate_single(img_hwc: np.ndarray, n: int) -> np.ndarray:
    """Repeat a single image n times along axis 0 → (n, H, W, 3)."""
    return np.broadcast_to(img_hwc[None], (n,) + img_hwc.shape).copy()


class RealRobotRetDataset(Dataset):
    """Retrieval-augmented dataset for the real_robot bg2 bimanual task family.

    Each pair in the input h5 already contains:
      * cur (robot)    : 16 JPEGs + chunk-start proprio + 16-frame action chunk
      * ret (human)    : 16 JPEGs + chunk-start proprio + 16-frame action chunk
      * idle-slot override applied so idle-hand residual is identically 0.
    """

    def __init__(
        self,
        pair_h5_path: str = DEFAULT_PAIR_H5,
        t5_text_embeddings_path: str = DEFAULT_T5_EMB,
        dataset_stats_path: str = DEFAULT_DATASET_STATS,
        delta_stats_dir: str = DEFAULT_DELTA_STATS_DIR,
        chunk_size: int = CHUNK_SIZE,
        final_image_size: int = 224,
        num_duplicates_per_image: int = NUM_DUPLICATES,
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_residual_actions: bool = True,
        predict_future_states: bool = True,
        retrieval_dropout_prob: float = 0.0,
        # When True, insert a ret_image latent slot before ret_action.
        # Layout becomes state_t=8:
        #   blank | ret_image | ret_action | cur_image | cur_proprio |
        #   pred_action | future_image | future_proprio
        use_ret_image: bool = False,
        # If provided, restrict the dataset to pair rows whose task field is in
        # this list (e.g. ["open_cabinet"]). None → use all pairs.
        tasks_filter: Optional[list] = None,
        # If True, slice every action/proprio output field to R-arm dims only
        # (indices 10..19 in the 20-D action). Single-task open_cabinet
        # optimization since L-arm is fully idle and learning it wastes capacity.
        r_arm_only: bool = False,
        # Roots used when the pair_h5 is manifest-only (no cur_*/ret_* arrays
        # baked in). Dataset fetches image / proprio / action lazily from these
        # h5 trees instead. Detected via pair_h5 attrs["manifest_only"].
        robot_root: str = "/mnt/ddn/dataset/real_robot/h5",
        human_root: str = "/mnt/ddn/dataset/real_robot/h5_human_clean",
        embodiment: str = "bg2_bimanual",
        # Compat shims for LIBERO base config inheritance
        use_proprio: bool = True,
        use_third_person_images: bool = True,
        use_wrist_images: bool = False,
        is_train: Optional[bool] = None,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert chunk_size == CHUNK_SIZE, f"chunk_size must be {CHUNK_SIZE}"
        assert num_duplicates_per_image == NUM_DUPLICATES, (
            f"num_duplicates_per_image must be {NUM_DUPLICATES}"
        )

        self.pair_h5_path = pair_h5_path
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.num_duplicates_per_image = num_duplicates_per_image
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_residual_actions = use_residual_actions
        self.predict_future_states = predict_future_states
        self.retrieval_dropout_prob = retrieval_dropout_prob
        self.use_ret_image = use_ret_image
        self.use_proprio = use_proprio
        self.use_third_person_images = use_third_person_images
        self.robot_root = robot_root
        self.human_root = human_root
        self.embodiment = embodiment
        self.r_arm_only = r_arm_only
        # R-arm indices in the 20-D action vector (used to slice raw h5 reads).
        self._r_arm_dims = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.int64)
        # Per-dim indices in the dataset's *output* dim (10D if r_arm_only, else 20D).
        if r_arm_only:
            self.xyz_dims     = np.array([0, 1, 2], dtype=np.int64)
            self.rot_dims     = np.array([3, 4, 5, 6, 7, 8], dtype=np.int64)
            self.grip_dims    = np.array([9], dtype=np.int64)
            self.non_xyz_dims = np.array([3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        else:
            self.xyz_dims     = XYZ_DIMS.copy()
            self.rot_dims     = ROT_DIMS.copy()
            self.grip_dims    = GRIP_DIMS.copy()
            self.non_xyz_dims = NON_XYZ_DIMS.copy()
        self.debug = debug

        # Open pair h5 lazily — h5py file handle is not fork-safe. We keep
        # a single reusable handle per worker, opened on first __getitem__.
        self._pair_h5 = None
        self._robot_h5_cache: dict = {}
        self._human_h5_cache: dict = {}
        with h5py.File(pair_h5_path, "r") as f:
            self.manifest_only = bool(f.attrs.get("manifest_only", False))
            self.top_k = int(f.attrs.get("top_k", 1))
            # Optional task filter (single-task / subset training).
            if tasks_filter is not None:
                wanted = {t.encode() if isinstance(t, str) else t for t in tasks_filter}
                task_arr = f["task"][:]
                mask = np.isin(task_arr, list(wanted))
                self._row_indices = np.where(mask)[0].astype(np.int64)
                self.N = int(self._row_indices.shape[0])
                print(f"[RealRobotRetDataset] {self.N} pairs (tasks_filter={tasks_filter}) "
                      f"in {pair_h5_path}  (manifest_only={self.manifest_only}, top_k={self.top_k})")
            else:
                self._row_indices = None
                self.N = int(f["task"].shape[0])
                print(f"[RealRobotRetDataset] {self.N} pairs in {pair_h5_path}  "
                      f"(manifest_only={self.manifest_only}, top_k={self.top_k})")
            for k, v in f.attrs.items():
                print(f"  attr {k} = {v}")

        # Dataset stats (computed from robot training h5; reuse instead of
        # recomputing from pair h5 to keep the same scale as the non-RAG model).
        with open(dataset_stats_path, "r") as fp:
            stats = json.load(fp)
        self.dataset_stats = {k: np.asarray(v) for k, v in stats.items()}
        print(f"[RealRobotRetDataset] dataset_stats loaded from {dataset_stats_path}")

        # Delta stats: only needed for residual (mixed) mode. Absolute mode
        # uses dataset_stats actions_min/max directly.
        os.makedirs(delta_stats_dir, exist_ok=True)
        self.delta_stats_path = os.path.join(delta_stats_dir, "delta_stats.json")
        self.delta_stats = None
        if self.use_residual_actions:
            self.delta_stats = self._load_or_compute_delta_stats()
        else:
            print("[RealRobotRetDataset] absolute mode (use_residual_actions=False) — "
                  "delta_stats skipped, using dataset_stats for all channels.")

        # T5 text embeddings — same file as the non-RAG model.
        with open(t5_text_embeddings_path, "rb") as fp:
            self.t5_text_embeddings = pickle.load(fp)
        print(f"[RealRobotRetDataset] t5 embeddings: {list(self.t5_text_embeddings.keys())}")

        # epoch_length mirrors PushT / RoboTwin convention.
        self.epoch_length = self.N

    # ──────────────────────────────────────────────────────────────────────
    # Lazy h5 handle
    # ──────────────────────────────────────────────────────────────────────

    def _get_h5(self) -> h5py.File:
        if self._pair_h5 is None:
            self._pair_h5 = h5py.File(self.pair_h5_path, "r")
        return self._pair_h5

    def _get_robot_h5(self, task: str, ep_basename: str) -> h5py.File:
        key = (task, ep_basename)
        h = self._robot_h5_cache.get(key)
        if h is None:
            path = os.path.join(self.robot_root, task, self.embodiment, "data", ep_basename)
            h = h5py.File(path, "r")
            self._robot_h5_cache[key] = h
        return h

    def _get_human_h5(self, task: str, ep_basename: str) -> h5py.File:
        key = (task, ep_basename)
        h = self._human_h5_cache.get(key)
        if h is None:
            path = os.path.join(self.human_root, task, self.embodiment, "data", ep_basename)
            h = h5py.File(path, "r")
            self._human_h5_cache[key] = h
        return h

    def __getstate__(self):
        # Drop the unpicklable h5 handles so workers can re-open them cleanly.
        state = self.__dict__.copy()
        state["_pair_h5"] = None
        state["_robot_h5_cache"] = {}
        state["_human_h5_cache"] = {}
        return state

    # ──────────────────────────────────────────────────────────────────────
    # Delta statistics for residual learning
    # ──────────────────────────────────────────────────────────────────────

    def _load_or_compute_delta_stats(self) -> dict:
        """Mixed-representation stats over all pairs:
          delta_xyz_min/max: (cur_xyz_fd - ret_xyz_fd) per xyz dim (frame delta residual)
                             — used to normalize the model's xyz target / output
          ret_xyz_fd_min/max: ret_xyz_fd per xyz dim
                              — used to normalize the ret_action xyz channels fed to the model
        Non-xyz (rot, grip) channels are NOT residual-learned; they use
        dataset_statistics.json (ABS) directly, so no entries here.
        """
        if os.path.exists(self.delta_stats_path):
            with open(self.delta_stats_path, "r") as fp:
                d = json.load(fp)
            print(f"[RealRobotRetDataset] delta_stats loaded from {self.delta_stats_path}")
            return {k: np.asarray(v) for k, v in d.items()}

        print("[RealRobotRetDataset] computing mixed delta_stats over all pairs (one pass)…")
        with h5py.File(self.pair_h5_path, "r") as f:
            cur_act = f["cur_action"][:]                       # (N, 16, 20) raw
            ret_act = f["ret_action"][:]                       # (N, 16, 20) or (N, K, 16, 20)
        # For top-K we use ALL K retrievals in the stats (more representative).
        if ret_act.ndim == 4:                                  # (N, K, 16, 20) → (N*K, 16, 20)
            K = ret_act.shape[1]
            ret_act = ret_act.reshape(-1, *ret_act.shape[2:])
            cur_act = np.repeat(cur_act, K, axis=0)
        # Convert to mixed: xyz → frame delta, non-xyz → ABS as-is
        N = cur_act.shape[0]
        cur_xyz_fd = np.zeros((N, CHUNK_SIZE, len(XYZ_DIMS)), dtype=np.float32)
        ret_xyz_fd = np.zeros((N, CHUNK_SIZE, len(XYZ_DIMS)), dtype=np.float32)
        for i in range(N):
            c_xyz = cur_act[i, :, XYZ_DIMS].T   # (16, 6)  numpy fancy-index transpose
            r_xyz = ret_act[i, :, XYZ_DIMS].T
            cur_xyz_fd[i] = np.diff(c_xyz, axis=0, prepend=c_xyz[0:1])
            ret_xyz_fd[i] = np.diff(r_xyz, axis=0, prepend=r_xyz[0:1])
        xyz_resid = (cur_xyz_fd - ret_xyz_fd).reshape(-1, len(XYZ_DIMS))
        ret_xyz_flat = ret_xyz_fd.reshape(-1, len(XYZ_DIMS))
        d = {
            "xyz_dims": XYZ_DIMS.astype(np.int64),
            "non_xyz_dims": NON_XYZ_DIMS.astype(np.int64),
            "delta_xyz_min":  xyz_resid.min(axis=0),
            "delta_xyz_max":  xyz_resid.max(axis=0),
            "delta_xyz_mean": xyz_resid.mean(axis=0),
            "delta_xyz_std":  xyz_resid.std(axis=0),
            "ret_xyz_fd_min": ret_xyz_flat.min(axis=0),
            "ret_xyz_fd_max": ret_xyz_flat.max(axis=0),
        }
        with open(self.delta_stats_path, "w") as fp:
            json.dump({k: v.tolist() for k, v in d.items()}, fp, indent=2)
        print(f"[RealRobotRetDataset] mixed delta_stats saved to {self.delta_stats_path}")
        print(f"  delta_xyz_min  = {np.round(d['delta_xyz_min'], 4).tolist()}")
        print(f"  delta_xyz_max  = {np.round(d['delta_xyz_max'], 4).tolist()}")
        print(f"  ret_xyz_fd_min = {np.round(d['ret_xyz_fd_min'], 4).tolist()}")
        print(f"  ret_xyz_fd_max = {np.round(d['ret_xyz_fd_max'], 4).tolist()}")
        return d

    # ──────────────────────────────────────────────────────────────────────
    # Normalization helpers
    # ──────────────────────────────────────────────────────────────────────

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        if not self.normalize_actions:
            return action
        a_min = self.dataset_stats["actions_min"]
        a_max = self.dataset_stats["actions_max"]
        return 2.0 * ((action - a_min) / (a_max - a_min + 1e-8)) - 1.0

    def _normalize_proprio(self, proprio: np.ndarray) -> np.ndarray:
        if not self.normalize_proprio:
            return proprio
        p_min = self.dataset_stats["proprio_min"]
        p_max = self.dataset_stats["proprio_max"]
        return 2.0 * ((proprio - p_min) / (p_max - p_min + 1e-8)) - 1.0

    def _normalize_xyz_delta(self, xyz_resid: np.ndarray) -> np.ndarray:
        """Normalize xyz frame-delta residual (cur_xyz_fd - ret_xyz_fd) to [-1, 1]."""
        d_min = self.delta_stats["delta_xyz_min"]
        d_max = self.delta_stats["delta_xyz_max"]
        return 2.0 * ((xyz_resid - d_min) / (d_max - d_min + 1e-8)) - 1.0

    def _normalize_ret_xyz_fd(self, ret_xyz_fd: np.ndarray) -> np.ndarray:
        """Normalize ret's xyz frame-delta to [-1, 1] (separate stat from residual)."""
        r_min = self.delta_stats["ret_xyz_fd_min"]
        r_max = self.delta_stats["ret_xyz_fd_max"]
        return 2.0 * ((ret_xyz_fd - r_min) / (r_max - r_min + 1e-8)) - 1.0

    def _build_target_action(self, cur_action: np.ndarray,
                             ret_action: np.ndarray) -> np.ndarray:
        """Build the mixed-representation target the model learns.
          xyz dims:   residual = cur_xyz_fd - ret_xyz_fd, normalized by delta_xyz stats
          non-xyz:    direct cur (rot/grip ABS), normalized by dataset_stats action stats
        Output shape (16, D) float32 in [-1, 1] (approx). D = 10 if r_arm_only else 20.
        """
        cur_mixed = _action_to_mixed(cur_action, self.xyz_dims)
        ret_mixed = _action_to_mixed(ret_action, self.xyz_dims)
        target = np.zeros_like(cur_mixed, dtype=np.float32)
        # xyz channels: residual normalized
        xyz_resid = cur_mixed[:, self.xyz_dims] - ret_mixed[:, self.xyz_dims]
        target[:, self.xyz_dims] = self._normalize_xyz_delta(xyz_resid).astype(np.float32)
        # non-xyz channels: cur ABS normalized by dataset_stats actions
        a_min = self.dataset_stats["actions_min"][self.non_xyz_dims]
        a_max = self.dataset_stats["actions_max"][self.non_xyz_dims]
        target[:, self.non_xyz_dims] = (
            2.0 * ((cur_action[:, self.non_xyz_dims] - a_min) / (a_max - a_min + 1e-8)) - 1.0
        ).astype(np.float32)
        return target

    def _build_ret_action_for_inject(self, ret_action: np.ndarray,
                                     cur_proprio_raw: np.ndarray) -> np.ndarray:
        """Ret action fed to the model (latent injection). Same 20-D shape.
          xyz : ret_xyz_fd normalized via delta_stats (real retrieval signal)
          rot : broadcast cur_proprio[rot] across the chunk. Robot rot stats
                are near-zero range on several dims (~0.001), so normalizing
                human rot through them saturates to ±50-400. Overriding with
                cur_proprio keeps this slot inside the reconstructable range.
          grip: ret_action[grip] normalized via dataset_stats (binary {0,1};
                same distribution on human and robot — no saturation).
        """
        ret_mixed = _action_to_mixed(ret_action, self.xyz_dims)
        out = np.zeros_like(ret_mixed, dtype=np.float32)
        # xyz: retrieved frame-delta
        out[:, self.xyz_dims] = self._normalize_ret_xyz_fd(ret_mixed[:, self.xyz_dims]).astype(np.float32)
        # rot: cur_proprio override, broadcast across chunk
        cur_p_norm = self._normalize_proprio(cur_proprio_raw).astype(np.float32)
        out[:, self.rot_dims] = cur_p_norm[self.rot_dims][None, :]
        # grip: retrieved value, normalized via action stats (binary, no saturate)
        a_min = self.dataset_stats["actions_min"][self.grip_dims]
        a_max = self.dataset_stats["actions_max"][self.grip_dims]
        out[:, self.grip_dims] = np.clip(
            2.0 * ((ret_action[:, self.grip_dims] - a_min) / (a_max - a_min + 1e-8)) - 1.0,
            -1.0, 1.0,
        ).astype(np.float32)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Core
    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> dict:
        if self.debug:
            idx = 0
        i = idx % self.N
        # Map back to absolute pair_h5 row when a task filter is active.
        if self._row_indices is not None:
            i = int(self._row_indices[i])
        f = self._get_h5()

        # ── 1) Fetch raw chunk data ─────────────────────────────────────
        # Either from baked-in pair h5, or via manifest lookup into robot/human h5.
        if self.manifest_only:
            task = f["task"][i].decode()
            query_ep = bytes(f["query_episode"][i]).decode()
            query_cs = int(f["query_chunk_start"][i])

            pool_ep_field = f["pool_episode"][i]
            pool_cs_field = f["pool_chunk_start"][i]
            if pool_ep_field.shape == ():  # K=1 scalar
                pool_ep_b = pool_ep_field
                pool_cs = int(pool_cs_field)
            else:
                k_idx = int(np.random.randint(pool_ep_field.shape[0]))
                pool_ep_b = pool_ep_field[k_idx]
                pool_cs = int(pool_cs_field[k_idx])
            pool_ep = bytes(pool_ep_b).decode()

            rh = self._get_robot_h5(task, query_ep)
            cur_jpegs = rh["observation/third_eye/rgb"][query_cs:query_cs + self.chunk_size]
            cur_proprio_raw = np.asarray(rh["proprio"][query_cs], dtype=np.float32)
            cur_action_raw = np.asarray(rh["action"][query_cs:query_cs + self.chunk_size], dtype=np.float32)

            hh = self._get_human_h5(task, pool_ep)
            ret_jpegs = hh["observation/third_eye/rgb"][pool_cs:pool_cs + self.chunk_size]
            ret_action_raw = np.asarray(hh["action"][pool_cs:pool_cs + self.chunk_size], dtype=np.float32)
        else:
            cur_jpegs = f["cur_image_jpegs"][i]
            ret_image_full = f["ret_image_jpegs"][i]
            cur_proprio_raw = f["cur_proprio"][i]
            cur_action_raw = f["cur_action"][i]
            ret_action_full = f["ret_action"][i]
            if ret_action_full.ndim == 3:
                k_idx = int(np.random.randint(ret_action_full.shape[0]))
                ret_action_raw = ret_action_full[k_idx]
                ret_jpegs = f["ret_image_jpegs"][i][k_idx]
            else:
                ret_action_raw = ret_action_full
                ret_jpegs = ret_image_full
            task = f["task"][i].decode()

        # Slice raw 20-D data to R-arm only 10-D (single-task open_cabinet).
        # All downstream normalize/build steps run in the dataset's output dim.
        if self.r_arm_only:
            cur_proprio_raw = np.asarray(cur_proprio_raw)[self._r_arm_dims]
            cur_action_raw  = np.asarray(cur_action_raw)[:, self._r_arm_dims]
            ret_action_raw  = np.asarray(ret_action_raw)[:, self._r_arm_dims]

        # ── 2) Decode images ────────────────────────────────────────────
        cur_frames = _decode_chunk_4(cur_jpegs)             # (4, H, W, 3) uint8
        ret_frames = _decode_chunk_4(ret_jpegs)
        future_frame = _decode_jpeg(cur_jpegs[15])           # (H, W, 3)

        # Optional retrieval dropout
        has_ret_data = 1
        if self.retrieval_dropout_prob > 0 and np.random.random() < self.retrieval_dropout_prob:
            ret_frames = np.zeros_like(ret_frames)
            has_ret_data = 0

        command = TASK_TO_COMMAND[task]

        cur_proprio_norm = self._normalize_proprio(cur_proprio_raw).astype(np.float32)

        # ── 3) Target + ret input ───────────────────────────────────────
        if self.use_residual_actions:
            # Mixed: xyz residual FRAME, non-xyz direct ABS.
            target_action = self._build_target_action(cur_action_raw, ret_action_raw)
            if has_ret_data:
                ret_action_for_inject = self._build_ret_action_for_inject(
                    ret_action_raw, cur_proprio_raw,
                )
            else:
                ret_action_for_inject = np.zeros_like(target_action)
        else:
            # Absolute baseline. xyz/grip come from the retrieved chunk (real
            # human signal); rot is replaced by cur_proprio broadcast across
            # the chunk because robot rot stats are near-zero range on several
            # dims so normalizing human rot through them saturates.
            target_action = self._normalize_action(cur_action_raw).astype(np.float32)
            if has_ret_data:
                ret_norm_all = np.clip(
                    self._normalize_action(ret_action_raw), -1.0, 1.0,
                ).astype(np.float32)
                cur_p_norm = self._normalize_proprio(cur_proprio_raw).astype(np.float32)
                ret_action_for_inject = ret_norm_all.copy()
                ret_action_for_inject[:, self.rot_dims] = cur_p_norm[self.rot_dims][None, :]
            else:
                ret_action_for_inject = np.zeros_like(target_action)

        # Future proprio = robot's chunk-end proprio (approx by cur_action[15]
        # absolute, normalized by dataset_stats).
        future_proprio_norm = self._normalize_action(cur_action_raw[15]).astype(np.float32)

        # ── 4) Build image sequence per latent layout ───────────────────
        # Layout (use_ret_image=False, state_t=7):
        #   0  blank                                (1 frame)
        #   1  ret action         (NUM_DUPLICATES blank, action injected)
        #   2  cur image          (NUM_DUPLICATES frames)
        #   3  cur proprio        (NUM_DUPLICATES blank, proprio injected)
        #   4  pred action        (NUM_DUPLICATES blank)
        #   5  future image       (NUM_DUPLICATES frames, model fills)
        #   6  future proprio     (NUM_DUPLICATES blank, model fills)
        #
        # Layout (use_ret_image=True, state_t=8):
        #   0  blank                                (1 frame)
        #   1  ret image          (NUM_DUPLICATES frames, retrieved human image)
        #   2  ret action         (NUM_DUPLICATES blank, action injected)
        #   3  cur image          (NUM_DUPLICATES frames)
        #   4  cur proprio        (NUM_DUPLICATES blank, proprio injected)
        #   5  pred action        (NUM_DUPLICATES blank)
        #   6  future image       (NUM_DUPLICATES frames, model fills)
        #   7  future proprio     (NUM_DUPLICATES blank, model fills)
        #
        # chunk_duration = 1 + state_t_minus_one * NUM_DUPLICATES = 25 or 29.
        image_list = []
        cur_seq = 0
        blank_single = np.zeros_like(cur_frames[0])               # (H, W, 3)
        blank_dup = _duplicate_single(blank_single, self.num_duplicates_per_image)

        # 0: blank (1 frame)
        image_list.append(blank_single[None])
        cur_seq += 1

        # 1 (optional): ret image — only when use_ret_image=True
        if self.use_ret_image:
            image_list.append(ret_frames)
            retrieved_video_start_latent_idx = cur_seq
            retrieved_video_end_latent_idx = cur_seq + 1
            cur_seq += 1
        else:
            retrieved_video_start_latent_idx = -1
            retrieved_video_end_latent_idx = -1

        # ret action (blank, action injected at training time)
        image_list.append(blank_dup.copy())
        retrieved_action_latent_idx = cur_seq
        cur_seq += 1

        # cur image
        image_list.append(cur_frames)
        current_image_latent_idx = cur_seq
        cur_seq += 1

        # cur proprio (blank, proprio injected)
        image_list.append(blank_dup.copy())
        current_proprio_latent_idx = cur_seq
        cur_seq += 1

        # pred action (blank, model generates)
        image_list.append(blank_dup.copy())
        action_latent_idx = cur_seq
        cur_seq += 1

        # future image (4 copies of chunk-end frame, model fills)
        future_dup = _duplicate_single(future_frame, self.num_duplicates_per_image)
        image_list.append(future_dup)
        future_image_latent_idx = cur_seq
        cur_seq += 1

        # future proprio (blank, model fills)
        image_list.append(blank_dup.copy())
        future_proprio_latent_idx = cur_seq
        cur_seq += 1

        images = np.concatenate(image_list, axis=0)    # (29, H, W, 3)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )

        # t5 emb lookup
        t5_emb_full = self.t5_text_embeddings[command]
        t5_emb = torch.squeeze(t5_emb_full)

        # next_action_chunk: for value-head compatibility; we don't use it but
        # the cosmos pipeline expects the field. Use the same chunk again.
        next_action_chunk = target_action.copy()

        return {
            "video": images,
            "actions": target_action,
            "t5_text_embeddings": t5_emb,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": cur_proprio_norm,
            "future_proprio": future_proprio_norm,
            "__key__": idx,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": -1,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": -1,
            "current_wrist_image2_latent_idx": -1,
            "current_image_latent_idx": current_image_latent_idx,
            "current_image2_latent_idx": -1,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": -1,
            "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": future_image_latent_idx,
            "future_image2_latent_idx": -1,
            "value_function_return": float("-100"),
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": float("-100"),
            # retrieval-specific (injected by the model into latents 1-2)
            "retrieved_video_start_latent_idx": retrieved_video_start_latent_idx,
            "retrieved_video_end_latent_idx": retrieved_video_end_latent_idx,
            "retrieved_action_latent_idx": retrieved_action_latent_idx,
            "retrieved_state_latent_idx": -1,    # ret_proprio slot removed per spec
            "retrieved_actions": ret_action_for_inject,    # (16, 20) normalized
            "retrieved_proprio": np.zeros_like(cur_proprio_norm),  # unused (slot=-1)
            "has_ret_data": has_ret_data,
            "has_ret_image": has_ret_data,
            "has_current_image": 1,
        }
