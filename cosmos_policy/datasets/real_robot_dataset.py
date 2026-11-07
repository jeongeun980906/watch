# SPDX-License-Identifier: Apache-2.0
"""RealRobot dataset loader — single third-eye camera, 20-D rot6d action.

This is a thin subclass of :class:`RoboTwinDataset` that swaps in our custom
HDF5 schema (see ``scripts/convert_real_robot_npz_to_h5.py``):

    /observation/third_eye/rgb   (T,)     |S<jpeg_max_len>   JPEG bytes
    /proprio                     (T, 20)  float32            measured state
    /action                      (T, 20)  float32            commanded action
    /t                           (T,)     float32

Differences from RoboTwinDataset:

* **Single camera.** The native RoboTwin model has three image slots (head +
  left wrist + right wrist). We only record one camera, so we feed the SAME
  third-eye stream into all three slots — easier than ripping out the wrist
  slots from the model definition, and the policy still sees a perfectly
  valid (if slightly redundant) observation.

* **20-D rot6d action / proprio.** Action and proprio are pre-built in the h5
  conversion step as ``[L_xyz(3), L_rot6d(6), L_grip(1), R_xyz(3), R_rot6d(6),
  R_grip(1)]`` (see CROP_RULE.md / convert_real_robot_npz_to_h5.py). We just
  read them directly — no quaternion repacking.

* **No action shifting.** RoboTwinDataset constructs action[t] = endpose[t+1]
  because RoboTwin only stores ``endpose`` (the achieved pose). We have both
  ``qpos_real`` (= proprio) and ``qpos_cmd`` (= the actual commanded action at
  step t), so the h5 already separates them — no left-shift needed.

Layout on disk (same shape RoboTwin uses, so the multi-task discovery helpers
keep working):

    <dataset_root>/<task>/<embodiment>/data/episode{N:04d}.hdf5

Pair this dataset with a model config that has ``action_dim=20, proprio_dim=20``
— see ``_REALROBOT_MODEL_CONFIG`` in ``robotwin_experiment_configs.py``.
"""
from __future__ import annotations

import glob
import hashlib
import os
import pickle
from typing import Optional, Sequence

import h5py
import numpy as np
from tqdm import tqdm

import torch

from cosmos_policy.datasets.dataset_common import (
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    determine_sample_type,
    get_action_chunk_with_padding,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    preprocess_image,
    rescale_data,
    resize_images,
)
from cosmos_policy.datasets.robotwin_dataset import (
    RoboTwinDataset,
    _decode_jpeg_sequence,
    _humanize_task_name,
    _strip_jpeg_buffers,
)


DEFAULT_REALROBOT_EMBODIMENT = "bg2_bimanual"

# Action / proprio dim emitted to the trainer.
# Must match what the model config (action_dim / proprio_dim) expects.
ACTION_DIM = 20
PROPRIO_DIM = 20


def _resolve_task_list(tasks):
    """Same semantics as RoboTwinDataset._resolve_task_list: None | list | path-to-file."""
    if tasks is None:
        return None
    if isinstance(tasks, str):
        if os.path.isfile(tasks):
            resolved = []
            with open(tasks, "r") as fp:
                for line in fp:
                    line = line.split("#", 1)[0].strip()
                    if line:
                        resolved.append(line)
            assert resolved, f"task list file {tasks} has no entries"
            return resolved
        return [tasks]
    resolved = [t.strip() for t in tasks if t and t.strip()]
    assert resolved
    return resolved


def _collect_realrobot_files(dataset_root: str, tasks, embodiment: str) -> list[str]:
    assert os.path.isdir(dataset_root), f"dataset_root not a dir: {dataset_root}"
    if tasks is None:
        tasks = sorted(
            name for name in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, name, embodiment, "data"))
        )
        assert tasks, f"no tasks with embodiment={embodiment!r} under {dataset_root}"
    files: list[str] = []
    for t in tasks:
        task_dir = os.path.join(dataset_root, t, embodiment, "data")
        assert os.path.isdir(task_dir), f"missing dir: {task_dir}"
        task_files = sorted(glob.glob(os.path.join(task_dir, "*.hdf5")))
        assert task_files, f"no .hdf5 files in {task_dir}"
        files.extend(task_files)
    return files


def _task_name_from_file(file_path: str) -> str:
    """Path shape: <root>/<task>/<embodiment>/data/episode{N}.hdf5 → -4."""
    return file_path.split(os.sep)[-4]


def _split_cache_dir(dataset_root: str, tasks: list[str], embodiment: str) -> str:
    tasks_key = ",".join(sorted(tasks))
    digest = hashlib.sha1(tasks_key.encode()).hexdigest()[:8]
    name = f"{embodiment}__{len(tasks)}tasks_{digest}"
    path = os.path.join(dataset_root, "_cosmos_cache", name)
    os.makedirs(path, exist_ok=True)
    return path


class RealRobotDataset(RoboTwinDataset):
    """Single-camera bimanual real-robot dataset with 20-D rot6d action.

    Subclasses :class:`RoboTwinDataset` purely to reuse ``__getitem__`` and all
    the index/stats plumbing. The constructor is rewritten from scratch because
    the on-disk schema is different (see module docstring).
    """

    def __init__(
        self,
        chunk_size: int = 16,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 8,
        return_value_function_returns: bool = False,
        gamma: float = 0.998,
        debug: bool = False,
        # Required: multi-task root entrypoint.
        dataset_root: Optional[str] = None,
        tasks: Optional[Sequence[str] | str] = None,
        embodiment: str = DEFAULT_REALROBOT_EMBODIMENT,
        stats_dir: Optional[str] = None,
        # Same future-prediction toggle as RoboTwinDataset.
        predict_future_states: bool = False,
        # Lazy decode — keep h5 JPEG bytes in RAM, decode per __getitem__ call.
        # Default True: this dataset is small (50 episodes) but JPEG bytes still
        # average ~30MB/episode → eager decode would multiply that by ~6x.
        lazy_video_decompression: bool = True,
        # Same defer mechanism as parent for joint-stats workflows.
        defer_finalization: bool = False,
        # Compat shims for LIBERO base config inheritance — silently absorbed.
        use_third_person_images: bool = True,
        use_wrist_images: bool = True,
        is_train: Optional[bool] = None,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 1.0,
        success_rollout_sampling_prob: float = 0.0,
        treat_success_rollouts_as_demos: bool = False,
        data_dir: Optional[str] = None,    # accepted but unused (we require dataset_root)
        # R-arm-only mode (single-task open_cabinet etc.): after loading the
        # raw 20-D h5 data, slice [10:20] so the model trains on action_dim=10.
        # Model config must set action_dim=proprio_dim=10 to match.
        r_arm_only: bool = False,
        **kwargs,
    ):
        assert dataset_root is not None, "RealRobotDataset requires --dataset_root"

        # Store knobs (parallel to RoboTwinDataset.__init__).
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.predict_future_states = bool(predict_future_states)
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.debug = debug
        self.embodiment = embodiment
        self.lazy_video_decompression = lazy_video_decompression
        self.r_arm_only = bool(r_arm_only)

        self.dataset_root = dataset_root
        self.tasks = _resolve_task_list(tasks)
        hdf5_files = _collect_realrobot_files(
            dataset_root=dataset_root, tasks=self.tasks, embodiment=embodiment,
        )
        if self.tasks is None:
            self.tasks = sorted({_task_name_from_file(p) for p in hdf5_files})
        # Stats dir: r_arm_only gets a separate sub-cache so 10-D stats don't
        # collide with the 20-D stats of the same task list.
        suffix = "_r10" if self.r_arm_only else ""
        self.data_dir = stats_dir or (_split_cache_dir(
            dataset_root=dataset_root, tasks=self.tasks, embodiment=embodiment,
        ) + suffix)
        if suffix and not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

        if os.environ.get("DEBUGGING", "False").lower() == "true":
            hdf5_files = hdf5_files[:1]

        self.data: dict = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.unique_commands: set = set()

        for file in tqdm(hdf5_files, desc="Loading RealRobot episodes"):
            with h5py.File(file, "r") as f:
                proprio = np.asarray(f["proprio"][()], dtype=np.float32)
                actions = np.asarray(f["action"][()], dtype=np.float32)
                assert proprio.shape[-1] == PROPRIO_DIM, (
                    f"proprio last-dim {proprio.shape[-1]} != {PROPRIO_DIM} in {file}"
                )
                assert actions.shape == proprio.shape, (
                    f"action/proprio shape mismatch in {file}: {actions.shape} vs {proprio.shape}"
                )
                if self.r_arm_only:
                    proprio = proprio[:, 10:20]   # R xyz(3) + R rot6d(6) + R grip(1)
                    actions = actions[:, 10:20]

                third_raw = f["observation/third_eye/rgb"][()]

                if self.lazy_video_decompression:
                    third_jpeg = _strip_jpeg_buffers(third_raw)
                    # Same JPEG stream feeds every camera slot the parent
                    # __getitem__ expects (head + left wrist + right wrist).
                    # That's fine semantically — model just sees one view three
                    # times, no zero tokens, no architecture change.
                    images_jpeg = third_jpeg
                    left_wrist_images_jpeg = third_jpeg
                    right_wrist_images_jpeg = third_jpeg
                    images = left_wrist_images = right_wrist_images = None
                    episode_num_steps = len(third_jpeg)
                else:
                    decoded = resize_images(_decode_jpeg_sequence(third_raw), final_image_size)
                    images = decoded
                    left_wrist_images = decoded
                    right_wrist_images = decoded
                    images_jpeg = left_wrist_images_jpeg = right_wrist_images_jpeg = None
                    episode_num_steps = len(decoded)

                assert episode_num_steps == len(actions), (
                    f"frame/action mismatch in {file}: {episode_num_steps} vs {len(actions)}"
                )

            task_name = _task_name_from_file(file)
            command = _humanize_task_name(task_name)
            self.unique_commands.add(command)

            if self.return_value_function_returns:
                returns = compute_monte_carlo_returns(
                    episode_num_steps, terminal_reward=1.0, gamma=self.gamma,
                )
            else:
                returns = None

            episode_entry: dict = dict(
                file_path=file,
                task_name=task_name,
                proprio=proprio,
                actions=actions,
                command=command,
                num_steps=episode_num_steps,
                returns=returns.copy() if returns is not None else None,
                success=True,
                is_lazy_video=self.lazy_video_decompression,
            )
            if self.lazy_video_decompression:
                episode_entry["images_jpeg"] = images_jpeg
                episode_entry["left_wrist_images_jpeg"] = left_wrist_images_jpeg
                episode_entry["right_wrist_images_jpeg"] = right_wrist_images_jpeg
            else:
                episode_entry["images"] = images
                episode_entry["left_wrist_images"] = left_wrist_images
                episode_entry["right_wrist_images"] = right_wrist_images

            self.data[self.num_episodes] = episode_entry
            self.num_episodes += 1
            self.num_steps += episode_num_steps

        # Index map + stats + rescale — mirrors RoboTwinDataset.__init__ tail.
        self._build_step_index_mapping()

        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as fp:
                self.t5_text_embeddings = pickle.load(fp)
        else:
            self.t5_text_embeddings = None

        self._finalized = False
        if defer_finalization:
            self.dataset_stats = None
            self.dataset_stats_post_norm = None
        else:
            self.dataset_stats = load_or_compute_dataset_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )
            if self.normalize_actions or self.normalize_proprio:
                if self.normalize_actions:
                    self.data = rescale_data(self.data, self.dataset_stats, "actions")
                if self.normalize_proprio:
                    self.data = rescale_data(self.data, self.dataset_stats, "proprio")
                self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                    data_dir=self.data_dir,
                    data=self.data,
                    calculate_dataset_statistics_func=calculate_dataset_statistics,
                )
            self._finalized = True

        # No rollout path — same as RoboTwinDataset.
        self._rollout_success_total_steps = 0
        self._rollout_failure_total_steps = 0
        self._rollout_total_steps = 0
        self.rollout_episode_metadata: dict = {}
        self.rollout_data: dict = {}
        self.rollout_num_episodes = 0
        self.rollout_num_steps = 0
        self.demonstration_sampling_prob = 1.0
        self.success_rollout_sampling_prob = 0.0

        result = calculate_epoch_structure(
            num_steps=self.num_steps,
            rollout_success_total_steps=0,
            rollout_failure_total_steps=0,
            demonstration_sampling_prob=1.0,
            success_rollout_sampling_prob=0.0,
        )
        self.adjusted_demo_count = result["adjusted_demo_count"]
        self.adjusted_success_rollout_count = result["adjusted_success_rollout_count"]
        self.adjusted_failure_rollout_count = result["adjusted_failure_rollout_count"]
        self.epoch_length = result["epoch_length"]

    # ------------------------------------------------------------------- API
    # Override RoboTwinDataset.__getitem__ to drop the two wrist-camera slots.
    # We only have one camera (third_eye), so the latent layout shrinks to:
    #
    #   no-future case (predict_future_states=False):
    #     slot 0: blank             (repeats=1)
    #     slot 1: current proprio   (repeats=num_duplicates_per_image)
    #     slot 2: current primary   (repeats=num_duplicates_per_image)
    #     slot 3: action            (repeats=num_duplicates_per_image)
    #     → state_t=4, chunk_duration = 1 + 3 * num_duplicates_per_image
    #
    #   with-future case (predict_future_states=True):
    #     +slot 4: future proprio   (repeats=num_duplicates_per_image)
    #     +slot 5: future primary   (repeats=num_duplicates_per_image)
    #     → state_t=6, chunk_duration = 1 + 5 * num_duplicates_per_image
    #
    # Wrist indices are emitted as -1 — matches PushT's pattern and the model
    # already special-cases that value to skip the wrist token block.
    def __getitem__(self, idx: int) -> dict:
        if self.debug:
            idx = 0

        sample_type = determine_sample_type(
            idx, self.adjusted_demo_count, self.adjusted_success_rollout_count
        )
        assert sample_type == "demo", (
            f"RealRobotDataset only supports demo sampling, got {sample_type}"
        )

        global_step_idx = idx % self.num_steps
        episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
        episode_data = self.data[episode_idx]

        future_frame_idx = min(
            relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1
        )

        # ── segment list (no wrist slots) ──
        frames: list[np.ndarray] = []
        repeats: list[int] = []
        segment_idx = 0

        action_latent_idx = -1
        value_latent_idx = -1
        current_proprio_latent_idx = -1
        current_image_latent_idx = -1
        future_proprio_latent_idx = -1
        future_image_latent_idx = -1
        # Wrist slots are unused for this dataset.
        current_wrist_image_latent_idx = -1
        current_wrist_image2_latent_idx = -1
        future_wrist_image_latent_idx = -1
        future_wrist_image2_latent_idx = -1

        ref_image = self._get_frame(episode_data, "images", relative_step_idx)

        # 0: blank
        frames.append(np.zeros_like(ref_image))
        repeats.append(1)
        segment_idx += 1

        # 1: current proprio (blank image placeholder, real proprio injected later)
        if self.use_proprio:
            current_proprio_latent_idx = segment_idx
            frames.append(np.zeros_like(ref_image))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
            proprio = episode_data["proprio"][relative_step_idx]
        else:
            proprio = np.zeros_like(episode_data["proprio"][relative_step_idx])

        # 2: current primary (only camera — third_eye)
        current_image_latent_idx = segment_idx
        frames.append(self._get_frame(episode_data, "images", relative_step_idx))
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # 3: action
        action_latent_idx = segment_idx
        frames.append(np.zeros_like(ref_image))
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # 4-5 (optional): future proprio + future primary
        if self.predict_future_states:
            if self.use_proprio:
                future_proprio_latent_idx = segment_idx
                frames.append(np.zeros_like(ref_image))
                repeats.append(self.num_duplicates_per_image)
                segment_idx += 1
                future_proprio = episode_data["proprio"][future_frame_idx]
            else:
                future_proprio = np.zeros_like(episode_data["proprio"][future_frame_idx])

            future_image_latent_idx = segment_idx
            frames.append(self._get_frame(episode_data, "images", future_frame_idx))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
        else:
            future_proprio = np.zeros_like(episode_data["proprio"][future_frame_idx])

        # Optional value slot (unused — return_value_function_returns defaults False).
        if self.return_value_function_returns:
            value_latent_idx = segment_idx
            frames.append(np.zeros_like(ref_image))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # ── preprocess + temporal expand (same as parent) ──
        all_unique_images = np.stack(frames, axis=0)
        all_unique_images = preprocess_image(
            all_unique_images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        lengths = torch.as_tensor(repeats, dtype=torch.long, device=all_unique_images.device)
        all_images = torch.repeat_interleave(all_unique_images, lengths, dim=1)

        # ── action chunks ──
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )
        next_relative_step_idx = min(
            relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1
        )
        next_action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        if self.return_value_function_returns:
            value_function_return = episode_data["returns"][future_frame_idx]
            next_future_frame_idx = min(
                next_relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1
            )
            next_value_function_return = episode_data["returns"][next_future_frame_idx]
        else:
            value_function_return = float("-100")
            next_value_function_return = float("-100")

        if self.t5_text_embeddings is None:
            raise RuntimeError(
                "t5_text_embeddings_path was not provided. Build embeddings "
                "with cosmos_policy.datasets.save_robotwin_t5_text_embeddings first."
            )
        t5_emb = torch.squeeze(self.t5_text_embeddings[episode_data["command"]])

        return {
            "video": all_images,
            "command": episode_data["command"],
            "actions": action_chunk,
            "t5_text_embeddings": t5_emb,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio,
            "future_proprio": future_proprio,
            "__key__": idx,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_wrist_image2_latent_idx": current_wrist_image2_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_wrist_image2_latent_idx": future_wrist_image2_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
        }
