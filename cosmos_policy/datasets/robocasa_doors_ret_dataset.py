# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""RoboCasa dataset with RoboCasa→MG retrieval-augmented conditioning.

Query set  : ``RoboCasa-Cosmos-Policy/success_only/kitchen_doors/<Folder>/<date>/*.hdf5``
Pool set   : ``robocasa-MG_3000_hdf5/kitchen_doors/<Folder>/demos.hdf5`` (JPEG obs)
Pair files : ``robocasa-retrieval/kitchen_doors/<Folder>/retrieval_results_robocasa_mg_task_<idx>.npz``
             (PushT-style; see features.md)

Each training sample is conditioned on the retrieved pool frame's 3 cameras,
proprio and action chunk. Latent layout (matches the model's ``state_t=15``):

    latent  0 : blank sentinel
    latent  1 : ret wrist           ┐
    latent  2 : ret left agent-view │  retrieval condition
    latent  3 : ret right agent-view┘
    latent  4 : ret proprio         (blank image; proprio injected)
    latent  5 : ret action          (blank image; action chunk injected)
    latent  6 : cur wrist           ┐
    latent  7 : cur left agent-view │  current condition
    latent  8 : cur right agent-view┘
    latent  9 : cur proprio         (blank image; proprio injected)
    latent 10 : pred action         ← generated (blank image)
    latent 11 : future wrist        ┐
    latent 12 : future left         │  generated future frames
    latent 13 : future right        ┘
    latent 14 : future proprio      ← generated

See ``pusht_dataset_ret.py`` for the analogous PushT version and
``robocasa_dataset.py`` for the non-retrieval baseline.
"""

from __future__ import annotations

import glob
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from cosmos_policy.datasets.dataset_common import (
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    decode_jpeg_bytes_dataset,
    decode_single_jpeg_frame,
    get_hdf5_files,
    preprocess_image,
    rescale_data,
)
from cosmos_policy.utils.utils import duplicate_array

np.set_printoptions(precision=3, linewidth=np.inf)


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

# Open door is Single + Double; both folders are pooled into the same dataset.
DEFAULT_OPEN_DOOR_FOLDERS: list[str] = ["OpenSingleDoor", "OpenDoubleDoor"]

# Maps folder → list of task_indices that live inside that folder (used to
# locate the per-task retrieval_pairs npz files).
FOLDER_TO_TASK_IDX: dict[str, list[int]] = {
    "OpenSingleDoor": [11, 12],          # microwave + cabinet door
    "OpenDoubleDoor": [8],               # cabinet doors
    "CloseSingleDoor": [3, 4],
    "CloseDoubleDoor": [0],
}


class RoboCasaDoorsRetDataset(Dataset):
    """RoboCasa query + MG retrieval pool for door-task policies.

    The retrieval lookup key is ``(folder, query_demo_key, t_query)``. Since
    retrieval pairs were built with ``query_stride=4``, ``__getitem__``
    searches nearby t values to cover steps that weren't directly indexed
    (same offset ladder as ``PushTRetDataset``).
    """

    # Offsets to probe when the requested query t isn't in the retrieval
    # lookup (stride=4 build → need up to ±3 tolerance).
    _RETRIEVAL_OFFSETS: tuple[int, ...] = (0, -1, 1, -2, 2, -3, 3, -4, 4)

    def __init__(
        self,
        query_data_dir: str,
        pool_data_dir: str,
        pair_glob: str,
        task_split: "list[str]" = tuple(DEFAULT_OPEN_DOOR_FOLDERS),
        chunk_size: int = 32,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        num_duplicates_per_image: int = 4,
        retrieval_top_k_choice: int = 1,
        retrieval_dropout_prob: float = 0.0,
        use_residual_actions: bool = False,
        return_value_function_returns: bool = False,
        gamma: float = 0.99,
        max_num_query_episodes: int = -1,
        rollout_data_dir: str = "",  # unused; kept for LazyConfig merge compat
        **kwargs,
    ):
        """
        Args
        ----
        query_data_dir : str
            Root of the RoboCasa query set (e.g.
            ``.../RoboCasa-Cosmos-Policy/success_only``). Folders under
            ``kitchen_doors/`` match ``task_split``.
        pool_data_dir : str
            Root of the MG pool set (e.g.
            ``.../robocasa-MG_3000_hdf5``). Each folder contains one
            ``demos.hdf5``.
        pair_glob : str
            Glob pattern to all pair npz files, e.g.
            ``.../robocasa-retrieval/kitchen_doors/*/retrieval_results_robocasa_mg_task_*.npz``.
        task_split : list[str]
            Folder names under ``kitchen_doors/`` (e.g. ``["OpenSingleDoor",
            "OpenDoubleDoor"]``).
        chunk_size : int
            Action chunk length (= number of action/future frames). Must be a
            multiple of ``num_duplicates_per_image``.
        retrieval_top_k_choice : int
            If 1, always pick rank-1 match. If >1, sample uniformly from
            top-K. Mirrors the PushT ret dataset.
        retrieval_dropout_prob : float
            Probability of dropping retrieval (zero ret_image/proprio/action +
            mask=0) for augmentation.
        use_residual_actions : bool
            If True, the model predicts the delta between the ground-truth
            action chunk and the retrieved action chunk (both in normalized
            space). The dataset returns ``actions = gt_action_chunk -
            ret_actions``. When retrieval is dropped (``ret_mask = 0``), the
            ret_action is zeroed so delta falls back to the absolute
            ground-truth action.
        """
        assert chunk_size % num_duplicates_per_image == 0, (
            f"chunk_size ({chunk_size}) must be divisible by "
            f"num_duplicates_per_image ({num_duplicates_per_image})"
        )
        self.query_data_dir = query_data_dir
        self.pool_data_dir = pool_data_dir
        self.pair_glob = pair_glob
        self.task_split = list(task_split)
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.num_duplicates_per_image = num_duplicates_per_image
        self.retrieval_top_k_choice = retrieval_top_k_choice
        self.retrieval_dropout_prob = retrieval_dropout_prob
        self.use_residual_actions = use_residual_actions
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.action_dim = 7  # Same convention as RoboCasaDataset baseline.

        # ── 1. Load query episodes (RoboCasa) ────────────────────────────────
        self.data: dict[int, dict] = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.unique_commands: set[str] = set()
        self._episode_to_folder: dict[int, str] = {}
        self._episode_to_demo_key: dict[int, str] = {}

        print(f"[RoboCasaDoorsRetDataset] loading query demos for {self.task_split} …")
        for folder in self.task_split:
            files = sorted(glob.glob(
                os.path.join(query_data_dir, "kitchen_doors", folder, "*", "*.hdf5")
            ))
            for fp in files:
                # Parent dir name is the date; combine with demo name for
                # uniqueness across dates (mirrors extract_features_robocasa.py).
                date = Path(fp).parent.name
                with h5py.File(fp, "r") as f:
                    sorted_keys = sorted(
                        f["data"].keys(), key=lambda x: int(x.split("_")[1])
                    )
                    for demo_key in sorted_keys:
                        if (max_num_query_episodes > 0 and
                                self.num_episodes >= max_num_query_episodes):
                            break
                        self._load_query_episode(f, folder, date, demo_key)
            print(f"  {folder}: {self.num_episodes} episodes loaded so far")

        self._build_step_index_mapping()

        # ── 2. Load pool demos (MG) — JPEG bytes in RAM ──────────────────────
        self.base_data: dict[tuple[str, str], dict] = {}
        print(f"[RoboCasaDoorsRetDataset] loading pool (MG) demos …")
        for folder in self.task_split:
            demos_path = os.path.join(
                pool_data_dir, "kitchen_doors", folder, "demos.hdf5"
            )
            if not os.path.exists(demos_path):
                raise FileNotFoundError(f"pool file missing: {demos_path}")
            with h5py.File(demos_path, "r") as f:
                for demo_key in tqdm(
                    sorted(f["data"].keys(), key=lambda x: int(x.split("_")[1])),
                    desc=f"pool {folder}",
                ):
                    self._load_pool_demo(f, folder, demo_key)
        print(f"  pool demos total: {len(self.base_data)}")

        # ── 3. Load retrieval lookup (merge all npz files) ───────────────────
        self._retrieval_lookup: dict[tuple[str, str, int], list[tuple[str, str, int, float]]] = {}
        pair_paths = sorted(glob.glob(self.pair_glob))
        print(f"[RoboCasaDoorsRetDataset] loading retrieval pairs from {len(pair_paths)} file(s) …")
        for pp in pair_paths:
            self._load_retrieval_npz(pp)
        print(f"  total retrieval entries: {len(self._retrieval_lookup):,}")

        # ── 4. Text embeddings (reason1 or t5; single pickle, same format) ───
        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as fp:
                self.t5_text_embeddings = pickle.load(fp)
        else:
            self.t5_text_embeddings = {}

        # ── 5. Dataset stats (computed on QUERY data; applied to both) ───────
        self.dataset_stats = load_or_compute_dataset_statistics(
            data_dir=self.query_data_dir,
            data=self.data,
            calculate_dataset_statistics_func=calculate_dataset_statistics,
        )

        if self.normalize_actions or self.normalize_proprio:
            if self.normalize_actions:
                self.data = rescale_data(self.data, self.dataset_stats, "actions")
                self._normalize_pool("actions")
            if self.normalize_proprio:
                self.data = rescale_data(self.data, self.dataset_stats, "proprio")
                self._normalize_pool("proprio")
            self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                data_dir=self.query_data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )

        self._calculate_epoch_structure()
        print(f"[RoboCasaDoorsRetDataset] ready — {self.num_episodes} query eps / "
              f"{self.num_steps} steps, {len(self.base_data)} pool demos, "
              f"{len(self._retrieval_lookup):,} ret keys.")

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #

    def _load_query_episode(self, hdf5_file, folder: str, date: str, demo_key: str):
        obs = hdf5_file[f"data/{demo_key}/obs"]
        left = obs["robot0_agentview_left_rgb"][:]          # (T, H, W, 3) uint8
        right = obs["robot0_agentview_right_rgb"][:]
        wrist = obs["robot0_eye_in_hand_rgb"][:]
        actions = hdf5_file[f"data/{demo_key}/actions"][:, : self.action_dim].astype(np.float32)
        # RoboCasa query: build 9-dim proprio from gripper_states + base_ee_pose.
        gripper = obs["gripper_states"][:].astype(np.float32)               # (T, 2)
        base_ee = obs["base_ee_pose"][:].astype(np.float32)                 # (T, 7)
        proprio = np.concatenate([gripper, base_ee], axis=1)                # (T, 9)
        command = str(hdf5_file[f"data/{demo_key}"].attrs["task_description"])
        self.unique_commands.add(command)
        num_steps = len(actions)
        returns = (
            compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
            if self.return_value_function_returns
            else None
        )
        # Globally unique demo identifier = "<date>__<demo_key>" (matches the
        # key written into retrieval_results_* npz files by
        # extract_features_robocasa.py).
        unique_demo = f"{date}__{demo_key}"
        self.data[self.num_episodes] = dict(
            left_primary_images=left,
            right_primary_images=right,
            wrist_images=wrist,
            proprio=proprio,
            actions=actions,
            command=command,
            num_steps=num_steps,
            returns=returns.copy() if returns is not None else None,
            folder=folder,
            demo_key=unique_demo,
        )
        self._episode_to_folder[self.num_episodes] = folder
        self._episode_to_demo_key[self.num_episodes] = unique_demo
        self.num_episodes += 1
        self.num_steps += num_steps

    def _load_pool_demo(self, hdf5_file, folder: str, demo_key: str):
        """Cache MG pool JPEGs (without decoding) + raw proprio/action."""
        obs = hdf5_file[f"data/{demo_key}/obs"]
        # vlen JPEG datasets — keep as python list of numpy byte arrays to
        # avoid holding an open HDF5 reference during training.
        left_jpegs  = [bytes(b) for b in obs["robot0_agentview_left_rgb"][:]]
        right_jpegs = [bytes(b) for b in obs["robot0_agentview_right_rgb"][:]]
        wrist_jpegs = [bytes(b) for b in obs["robot0_eye_in_hand_rgb"][:]]
        actions = hdf5_file[f"data/{demo_key}/actions"][:, : self.action_dim].astype(np.float32)
        # MG pool: robot_states already 9-dim [gripper(2), pos(3), quat(4)].
        proprio = hdf5_file[f"data/{demo_key}/robot_states"][:].astype(np.float32)
        self.base_data[(folder, demo_key)] = dict(
            left_jpegs=left_jpegs,
            right_jpegs=right_jpegs,
            wrist_jpegs=wrist_jpegs,
            actions=actions,
            proprio=proprio,
        )

    def _load_retrieval_npz(self, npz_path: str):
        """Parse one retrieval_results_robocasa_mg_task_<idx>.npz.

        Infer the folder from the config inside the npz so we don't rely on
        parsing filenames.
        """
        data = np.load(npz_path, allow_pickle=True)
        config = {kv.split("=", 1)[0]: kv.split("=", 1)[1] for kv in data["config"] if "=" in kv}
        folder = config.get("folder")
        assert folder is not None, f"{npz_path}: missing folder in config"

        query_ids = data["query_ids"]
        match_ids = data["match_ids"]
        match_sims = data["match_sims"].astype(np.float32)
        added = 0
        for qi, qid in enumerate(query_ids):
            _, demo_key, t_str = str(qid).split("/")
            key = (folder, demo_key, int(t_str))
            matches = []
            for ki in range(match_ids.shape[1]):
                mid = str(match_ids[qi, ki])
                _, m_demo, m_t = mid.split("/")
                matches.append((folder, m_demo, int(m_t), float(match_sims[qi, ki])))
            self._retrieval_lookup[key] = matches
            added += 1
        print(f"  {os.path.basename(npz_path)}: folder={folder} entries={added:,}")

    def _normalize_pool(self, field: str):
        """Apply the same min/max stats used for query data to pool arrays."""
        lo = self.dataset_stats[f"{field}_min"]
        hi = self.dataset_stats[f"{field}_max"]
        rng = np.maximum(hi - lo, 1e-8)
        for key, entry in self.base_data.items():
            arr = entry[field]
            entry[field] = (2.0 * ((arr - lo) / rng) - 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Index / epoch helpers (same shape as RoboCasaDataset)
    # ------------------------------------------------------------------ #

    def _build_step_index_mapping(self):
        self._step_to_episode_map: dict[int, tuple[int, int]] = {}
        self._total_steps = 0
        for ep_idx, ep in self.data.items():
            for i in range(ep["num_steps"]):
                self._step_to_episode_map[self._total_steps] = (ep_idx, i)
                self._total_steps += 1

    def _calculate_epoch_structure(self):
        result = calculate_epoch_structure(
            num_steps=self.num_steps,
            rollout_success_total_steps=0,
            rollout_failure_total_steps=0,
            demonstration_sampling_prob=1.0,
            success_rollout_sampling_prob=0.0,
        )
        self.adjusted_demo_count = result["adjusted_demo_count"]
        self.adjusted_success_rollout_count = 0
        self.adjusted_failure_rollout_count = 0
        self.epoch_length = result["epoch_length"]

    # ------------------------------------------------------------------ #
    # Retrieval lookup
    # ------------------------------------------------------------------ #

    def _find_retrieval_match(
        self, folder: str, demo_key: str, step_idx: int
    ) -> "tuple[str, str, int] | None":
        """Return (pool_folder, pool_demo, pool_t) or None if nothing nearby.

        Probes ``step_idx ± offset`` since retrieval was built with stride>1.
        """
        for offset in self._RETRIEVAL_OFFSETS:
            key = (folder, demo_key, step_idx + offset)
            matches = self._retrieval_lookup.get(key)
            if matches:
                choice_pool = matches[: self.retrieval_top_k_choice]
                m_folder, m_demo, m_t, _sim = random.choice(choice_pool)
                return (m_folder, m_demo, m_t)
        return None

    def _load_pool_frame(self, folder: str, demo_key: str, t: int) -> dict:
        """Decode a single pool frame (wrist + left + right) on demand."""
        entry = self.base_data[(folder, demo_key)]
        T = len(entry["left_jpegs"])
        t = max(0, min(t, T - 1))
        return dict(
            wrist=decode_single_jpeg_frame(entry["wrist_jpegs"][t]),
            left=decode_single_jpeg_frame(entry["left_jpegs"][t]),
            right=decode_single_jpeg_frame(entry["right_jpegs"][t]),
        )

    def _get_pool_action_chunk(self, folder: str, demo_key: str, start: int) -> np.ndarray:
        entry = self.base_data[(folder, demo_key)]
        T = len(entry["actions"])
        idxs = [max(0, min(start + i, T - 1)) for i in range(self.chunk_size)]
        return entry["actions"][idxs]

    def _get_pool_proprio_at(self, folder: str, demo_key: str, t: int) -> np.ndarray:
        entry = self.base_data[(folder, demo_key)]
        T = len(entry["proprio"])
        return entry["proprio"][max(0, min(t, T - 1))]

    # ------------------------------------------------------------------ #
    # Dataset API
    # ------------------------------------------------------------------ #

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, idx):
        global_step_idx = idx % self.num_steps
        episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
        ep = self.data[episode_idx]

        folder = ep["folder"]
        demo_key = ep["demo_key"]
        num_steps = ep["num_steps"]
        future_frame_idx = min(relative_step_idx + self.chunk_size, num_steps - 1)

        # ── Retrieval lookup ────────────────────────────────────────────
        ret_hit = self._find_retrieval_match(folder, demo_key, relative_step_idx)
        ret_dropped = (
            self.retrieval_dropout_prob > 0 and random.random() < self.retrieval_dropout_prob
        )
        if ret_hit is None or ret_dropped:
            ret_frames = None
            ret_actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
            ret_proprio = np.zeros(ep["proprio"].shape[1], dtype=np.float32)
            ret_mask = 0
        else:
            m_folder, m_demo, m_t = ret_hit
            # Retrieval side is future-aligned (mirrors the query side where
            # `future_image` sits at `t + chunk_size`):
            #   ret_image / ret_proprio   live at m_t + chunk_size
            #   ret_actions                span   [m_t, m_t + chunk_size]
            m_t_future = m_t + self.chunk_size
            ret_frames = self._load_pool_frame(m_folder, m_demo, m_t_future)
            ret_actions = self._get_pool_action_chunk(m_folder, m_demo, m_t)
            ret_proprio = self._get_pool_proprio_at(m_folder, m_demo, m_t_future)
            ret_mask = 1

        # ── Build the image_list in the fixed latent order ──────────────
        image_list: list[np.ndarray] = []
        current_sequence_idx = 0

        # latent 0: blank sentinel (1 frame, no duplication)
        ref_img = ep["left_primary_images"][relative_step_idx]
        image_list.append(np.expand_dims(np.zeros_like(ref_img), axis=0))
        current_sequence_idx += 1

        def _push_image(arr: np.ndarray) -> int:
            nonlocal current_sequence_idx
            image_list.append(duplicate_array(arr, total_num_copies=self.num_duplicates_per_image))
            idx_out = current_sequence_idx
            current_sequence_idx += 1
            return idx_out

        def _push_blank() -> int:
            return _push_image(np.zeros_like(ref_img))

        # latent 1-3: retrieval frames (fall back to zeros when dropped).
        if ret_frames is None:
            ret_wrist_image_latent_idx = _push_blank()
            ret_left_image_latent_idx = _push_blank()
            ret_right_image_latent_idx = _push_blank()
        else:
            ret_wrist_image_latent_idx = _push_image(ret_frames["wrist"])
            ret_left_image_latent_idx = _push_image(ret_frames["left"])
            ret_right_image_latent_idx = _push_image(ret_frames["right"])

        # latent 4: ret proprio (injected)
        ret_proprio_latent_idx = _push_blank()
        # latent 5: ret action (injected)
        ret_action_latent_idx = _push_blank()

        # latent 6-8: current frames
        current_wrist_image_latent_idx = _push_image(ep["wrist_images"][relative_step_idx])
        current_image_latent_idx = _push_image(ep["left_primary_images"][relative_step_idx])
        current_image2_latent_idx = _push_image(ep["right_primary_images"][relative_step_idx])

        # latent 9: cur proprio (injected)
        current_proprio_latent_idx = _push_blank()

        # latent 10: predicted action (generated)
        action_latent_idx = _push_blank()

        # latent 11-13: future frames (generated)
        future_wrist_image_latent_idx = _push_image(ep["wrist_images"][future_frame_idx])
        future_image_latent_idx = _push_image(ep["left_primary_images"][future_frame_idx])
        future_image2_latent_idx = _push_image(ep["right_primary_images"][future_frame_idx])

        # latent 14: future proprio (generated)
        future_proprio_latent_idx = _push_blank()

        images = np.concatenate(image_list, axis=0)
        if np.isnan(images).any() or np.isinf(images).any():
            raise ValueError(f"Invalid image data for sample {idx}")
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        if torch.isnan(images).any() or torch.isinf(images).any():
            raise ValueError(f"Invalid processed image tensor for sample {idx}")

        # ── Action chunk (padded when near end of episode) ──────────────
        remaining = num_steps - relative_step_idx
        if remaining >= self.chunk_size:
            action_chunk = ep["actions"][relative_step_idx : relative_step_idx + self.chunk_size]
        else:
            pad = np.tile(ep["actions"][-1], (self.chunk_size - remaining, 1))
            action_chunk = np.concatenate([ep["actions"][relative_step_idx:], pad], axis=0)
        action_chunk = action_chunk.astype(np.float32)

        # Residual-action learning: predict delta in the (already-normalized)
        # action space. When retrieval is dropped, ret_actions is zeros, so
        # delta falls back to the absolute GT action chunk.
        target_actions = (
            action_chunk - ret_actions if self.use_residual_actions else action_chunk
        )

        proprio = ep["proprio"][relative_step_idx]
        future_proprio = ep["proprio"][future_frame_idx]

        command = ep["command"]
        t5_emb = (
            torch.squeeze(self.t5_text_embeddings[command])
            if command in self.t5_text_embeddings
            else torch.zeros(512, 1024)
        )

        sample = {
            "video": images,
            "command": command,
            "actions": target_actions.astype(np.float32),
            "t5_text_embeddings": t5_emb,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio.astype(np.float32),
            "future_proprio": future_proprio.astype(np.float32),
            "__key__": idx,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 1,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "value_function_return": float("-100"),
            # Latent indices (fixed layout).
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": -1,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "current_image2_latent_idx": current_image2_latent_idx,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
            "future_image2_latent_idx": future_image2_latent_idx,
            # Retrieval-specific.
            "ret_wrist_image_latent_idx": ret_wrist_image_latent_idx,
            "ret_left_image_latent_idx": ret_left_image_latent_idx,
            "ret_right_image_latent_idx": ret_right_image_latent_idx,
            "ret_proprio_latent_idx": ret_proprio_latent_idx,
            "ret_action_latent_idx": ret_action_latent_idx,
            "ret_proprio": ret_proprio.astype(np.float32),
            "ret_actions": ret_actions.astype(np.float32),
            "ret_mask": ret_mask,
        }
        return sample
