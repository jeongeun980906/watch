"""RoboTwin RAG dataset, *joint-space* (qpos) action variant.

Mirrors :class:`RoboTwinRetDataset` 1:1 except:

* Episode loader reads ``/joint_action/{left_arm,left_gripper,right_arm,
  right_gripper}`` and packs them into the 14-d aloha-agilex layout
  (consumed by ``TASK_ENV.take_action(action_type='qpos')``) — same shape
  as :func:`cosmos_policy.datasets.robotwin_joint_dataset._build_joint_action_14d`.
* ``_ACTION_DIM`` is 14 instead of 16 — the inner ``_get_retrieved`` zero
  fallback and stats arrays size off this.
* Cache dir is suffixed ``__joint`` so EE and joint variants of the same
  (task, query, pool) tuple never share a ``dataset_statistics.json`` /
  ``delta_dataset_statistics_*.json``.

Pool features (image cache, retrieval npz, SAM3 object_init) are
representation-agnostic and the same files used by the EE config still
apply — only the per-frame action/proprio extraction changes.
"""

from __future__ import annotations

import h5py
import numpy as np

from cosmos_policy.datasets.dataset_utils import resize_images
from cosmos_policy.datasets.robotwin_dataset import _decode_jpeg_sequence
from cosmos_policy.datasets.robotwin_joint_dataset import (
    JOINT_ACTION_DIM,
    _build_joint_action_14d,
)
from cosmos_policy.datasets.robotwin_rag_dataset import RoboTwinRetDataset


def _load_episode_frames_14d(hdf5_path: str, final_image_size: int) -> dict:
    """Joint-space sibling of ``robotwin_rag_dataset._load_episode_frames_16d``.

    Returns the same dict schema (``proprio`` / ``actions`` / ``images`` /
    ``left_wrist_images`` / ``right_wrist_images`` / ``num_steps``) but with
    ``proprio`` / ``actions`` shaped ``(T, 14)`` derived from
    ``/joint_action/*``. The action chunk shifts by one step (action at t is
    qpos at t+1) to mirror the EE loader's ``actions = ee[1:] + ee[-1:]``
    convention so the residual / chunk semantics stay aligned.
    """
    with h5py.File(hdf5_path, "r") as f:
        left_arm = f["/joint_action/left_arm"][()]
        right_arm = f["/joint_action/right_arm"][()]
        left_gripper = f["/joint_action/left_gripper"][()]
        right_gripper = f["/joint_action/right_gripper"][()]
        qpos = _build_joint_action_14d(
            left_arm, left_gripper, right_arm, right_gripper
        )
        proprio = qpos
        actions = np.concatenate([qpos[1:], qpos[-1:]], axis=0)

        head = _decode_jpeg_sequence(f["/observation/head_camera/rgb"][()])
        left_wrist = _decode_jpeg_sequence(f["/observation/left_camera/rgb"][()])
        right_wrist = _decode_jpeg_sequence(f["/observation/right_camera/rgb"][()])

    head = resize_images(head, final_image_size)
    left_wrist = resize_images(left_wrist, final_image_size)
    right_wrist = resize_images(right_wrist, final_image_size)
    num_steps = len(head)
    assert len(actions) == num_steps, (
        f"frame/action mismatch in {hdf5_path}: {num_steps} frames vs "
        f"{len(actions)} actions"
    )
    return {
        "images": head,
        "left_wrist_images": left_wrist,
        "right_wrist_images": right_wrist,
        "proprio": proprio,
        "actions": actions,
        "num_steps": num_steps,
    }


class RoboTwinJointRetDataset(RoboTwinRetDataset):
    """Joint-space (qpos) variant of :class:`RoboTwinRetDataset`.

    Inherits all retrieval / batching / normalization machinery from the
    parent. Only swaps the per-frame loader, action/proprio dim, and
    cache-dir suffix via the three subclass hooks.
    """

    _LOAD_EPISODE_FN = staticmethod(_load_episode_frames_14d)
    _ACTION_DIM = JOINT_ACTION_DIM
    _CACHE_DIR_SUFFIX = "__joint"
