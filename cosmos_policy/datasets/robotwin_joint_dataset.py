# SPDX-License-Identifier: Apache-2.0
"""RoboTwin 2.0 dataloader, *joint-space* (qpos) action variant.

Mirrors ``RoboTwinDataset`` but reads ``/joint_action/vector`` (T, 14) directly
instead of building a 16-dim EE pose. Action / proprio dim therefore changes:

    EE  variant: 16 = [L_xyz(3) + L_quat(4) + L_grip(1) + R_xyz(3) + R_quat(4) + R_grip(1)]
    qpos variant: 14 = [L_arm(6) + L_grip(1) + R_arm(6) + R_grip(1)]

The 14-dim layout is exactly what RoboTwin's
``TASK_ENV.take_action(action, action_type='qpos')`` expects, so an inference
server emitting joint actions can hand chunks straight to the env.

Same one-step left-shift convention as the EE variant
(``action_t == proprio_{t+1}``, terminal step holds last value).

Cache + stats live alongside the EE variant under a ``__joint`` suffix so the
two reps never share a ``dataset_statistics.json``.
"""

from __future__ import annotations

import os
import pickle
from typing import Optional, Sequence

import h5py
import numpy as np
from tqdm import tqdm

# Reuse the EE dataset's module-level helpers (no edits to that file).
from cosmos_policy.datasets.dataset_common import (
    build_demo_step_index_mapping,
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    get_hdf5_files,
    rescale_data,
    resize_images,
)
from cosmos_policy.datasets.robotwin_dataset import (
    DEFAULT_EMBODIMENT,
    RoboTwinDataset,
    _collect_robotwin_files,
    _decode_jpeg_sequence,
    _humanize_task_name,
    _resolve_task_list,
    _split_cache_dir,
    _task_name_from_file,
)

JOINT_ACTION_DIM = 14  # L_arm(6)+L_grip(1)+R_arm(6)+R_grip(1) for aloha-agilex


def _build_joint_action_14d(
    left_arm: np.ndarray,       # (T, 6)
    left_gripper: np.ndarray,   # (T,)
    right_arm: np.ndarray,      # (T, 6)
    right_gripper: np.ndarray,  # (T,)
) -> np.ndarray:
    """Pack [L_arm(6), L_grip(1), R_arm(6), R_grip(1)] = (T, 14).

    Matches the layout consumed by RoboTwin's ``take_action(..., action_type='qpos')``
    (see ``envs/_base_task.py:take_action`` — left_arm_dim = jointstate_len - 1
    so for 6-DOF aloha-agilex the action vector is exactly this 14-dim shape).
    """
    out = np.concatenate(
        [
            left_arm.astype(np.float32),
            left_gripper.astype(np.float32)[:, None],
            right_arm.astype(np.float32),
            right_gripper.astype(np.float32)[:, None],
        ],
        axis=-1,
    )
    assert out.shape[-1] == JOINT_ACTION_DIM, (
        f"expected {JOINT_ACTION_DIM}-dim joint action, got {out.shape}"
    )
    return out


class RoboTwinJointDataset(RoboTwinDataset):
    """Joint-space (qpos) variant of ``RoboTwinDataset``.

    Inherits ``__getitem__`` and all post-load machinery from the parent
    (the per-sample dict shape is identical — only ``actions`` / ``proprio``
    last-dim changes from 16 to 14). ``__init__`` is overridden to read
    ``/joint_action/vector`` instead of ``/endpose/...``.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
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
        dataset_root: Optional[str] = None,
        tasks: Optional[Sequence[str] | str] = None,
        embodiment: str = DEFAULT_EMBODIMENT,
        stats_dir: Optional[str] = None,
        # Signature-compatible (unused) flags from the LIBERO/ALOHA base config.
        use_third_person_images: bool = True,
        use_wrist_images: bool = True,
        is_train: Optional[bool] = None,
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 1.0,
        success_rollout_sampling_prob: float = 0.0,
        treat_success_rollouts_as_demos: bool = False,
        **kwargs,
    ):
        # NOTE: deliberately not calling super().__init__() because the parent
        # __init__ hard-codes EE extraction. We re-implement the same flow but
        # read /joint_action/vector instead.
        assert (data_dir is not None) or (dataset_root is not None), (
            "must pass either data_dir or dataset_root"
        )

        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.debug = debug
        self.embodiment = embodiment

        if dataset_root is not None:
            self.dataset_root = dataset_root
            self.tasks = _resolve_task_list(tasks)
            hdf5_files = _collect_robotwin_files(
                dataset_root=dataset_root,
                tasks=self.tasks,
                embodiment=embodiment,
            )
            if self.tasks is None:
                self.tasks = sorted({_task_name_from_file(p) for p in hdf5_files})
            # Suffix the cache dir so EE and joint stats never collide.
            ee_dir = stats_dir or _split_cache_dir(
                dataset_root=dataset_root, tasks=self.tasks, embodiment=embodiment
            )
            self.data_dir = ee_dir if stats_dir else f"{ee_dir}__joint"
            os.makedirs(self.data_dir, exist_ok=True)
        else:
            self.dataset_root = None
            self.tasks = None
            self.data_dir = stats_dir or data_dir
            hdf5_files = get_hdf5_files(data_dir, is_train=is_train)
        assert len(hdf5_files) > 0, (
            f"No HDF5 files found (data_dir={data_dir!r}, dataset_root={dataset_root!r}, "
            f"tasks={tasks!r}, embodiment={embodiment!r})"
        )

        if os.environ.get("DEBUGGING", "False").lower() == "true":
            hdf5_files = hdf5_files[:1]

        self.data: dict = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.unique_commands: set = set()

        for file in tqdm(hdf5_files, desc="Loading RoboTwin joint episodes"):
            with h5py.File(file, "r") as f:
                # ---- actions / proprio: native joint-space (qpos) ----
                # /joint_action/vector is (T, 14):
                #   [L_arm(6), L_grip(1), R_arm(6), R_grip(1)]
                # Same one-step left-shift convention as the EE variant:
                # action[t] = proprio[t+1]; final step holds last value.
                if "/joint_action/vector" in f:
                    qpos_at_t = np.asarray(f["/joint_action/vector"][()], dtype=np.float32)
                else:
                    # Fallback: build from the per-arm components if `vector` is
                    # absent in older dumps.
                    qpos_at_t = _build_joint_action_14d(
                        f["/joint_action/left_arm"][()],
                        f["/joint_action/left_gripper"][()],
                        f["/joint_action/right_arm"][()],
                        f["/joint_action/right_gripper"][()],
                    )
                assert qpos_at_t.shape[-1] == JOINT_ACTION_DIM, (
                    f"expected joint vector last-dim {JOINT_ACTION_DIM}, "
                    f"got {qpos_at_t.shape} in {file}"
                )
                proprio = qpos_at_t
                actions = np.concatenate([qpos_at_t[1:], qpos_at_t[-1:]], axis=0)

                # ---- images (identical to EE variant) ----
                images = _decode_jpeg_sequence(f["/observation/head_camera/rgb"][()])
                left_wrist_images = _decode_jpeg_sequence(
                    f["/observation/left_camera/rgb"][()]
                )
                right_wrist_images = _decode_jpeg_sequence(
                    f["/observation/right_camera/rgb"][()]
                )
                images = resize_images(images, final_image_size)
                left_wrist_images = resize_images(left_wrist_images, final_image_size)
                right_wrist_images = resize_images(right_wrist_images, final_image_size)

                episode_num_steps = len(images)
                assert episode_num_steps == len(actions), (
                    f"Frame/action length mismatch in {file}: "
                    f"{episode_num_steps} frames vs {len(actions)} actions"
                )

            task_name = _task_name_from_file(file)
            command = _humanize_task_name(task_name)
            self.unique_commands.add(command)

            if self.return_value_function_returns:
                returns = compute_monte_carlo_returns(
                    episode_num_steps, terminal_reward=1.0, gamma=self.gamma
                )
            else:
                returns = None

            self.data[self.num_episodes] = dict(
                file_path=file,
                task_name=task_name,
                images=images,
                left_wrist_images=left_wrist_images,
                right_wrist_images=right_wrist_images,
                proprio=proprio,
                actions=actions,
                command=command,
                num_steps=episode_num_steps,
                returns=returns.copy() if returns is not None else None,
                success=True,
                is_lazy_video=False,
            )
            self.num_episodes += 1
            self.num_steps += episode_num_steps

        self._build_step_index_mapping()

        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as fp:
                self.t5_text_embeddings = pickle.load(fp)
        else:
            self.t5_text_embeddings = None

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

        # No rollout path (mirrors RoboTwinDataset).
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
