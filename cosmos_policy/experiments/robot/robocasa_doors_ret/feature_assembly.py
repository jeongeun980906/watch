# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Canonical query-feature composition for RoboCasa→MG retrieval.

The same layout is used at training time when assembling pool features
(see ``/mnt/ddn/robocasa_retrieval/build_retrieval_index.py``) and at
eval time for online query features (see ``retrieval.py``). Keep both
sides in sync — the pool features and the query features must live in
the *same* weighted L2-normed feature space for FAISS L2 to be
meaningful.

Layout (3460 dims per query row; see features.md):

    [ action(320) | time(2) | image_cur(1536) | proprio(11) |
      image_past(1536) | velocity(55) ]

Each group is L2-normalized per row, then scaled by ``sqrt(weight)``.
Zero-filling a whole group (e.g. action at eval) makes it contribute
a constant ``w_group`` to every pool distance → no ranking effect.
"""

from __future__ import annotations

import numpy as np

# Window / chunk must match the pool features.
N_HIST: int = 4
H_CHUNK: int = 32

# Per-timestep dims.
DIM_DINO_CLS: int = 768
DIM_PROPRIO: int = 11
DIM_ACTION: int = 10   # dpos(3) + drot6d(6) + gripper(1)

DEFAULT_WEIGHTS: dict[str, float] = {
    "action": 1.0,
    "time": 1.0,
    "image": 0.7,
    "proprio": 0.7,
    "past_image": 0.2,
    "velocity": 0.2,
}

# Per-group dims (flattened).
DIM_ACTION_GROUP: int = H_CHUNK * DIM_ACTION                     # 320
DIM_TIME_GROUP: int = 2                                          # sin/cos
DIM_IMAGE_GROUP: int = 2 * DIM_DINO_CLS                          # 1536
DIM_PROPRIO_GROUP: int = DIM_PROPRIO                             # 11
DIM_PAST_IMAGE_GROUP: int = 2 * DIM_DINO_CLS                     # 1536
DIM_VELOCITY_GROUP: int = (N_HIST + 1) * DIM_PROPRIO             # 55
FEATURE_DIM: int = (
    DIM_ACTION_GROUP
    + DIM_TIME_GROUP
    + DIM_IMAGE_GROUP
    + DIM_PROPRIO_GROUP
    + DIM_PAST_IMAGE_GROUP
    + DIM_VELOCITY_GROUP
)

# Group slice offsets (useful for per-group analysis / ablation).
OFFSET_ACTION = 0
OFFSET_TIME = OFFSET_ACTION + DIM_ACTION_GROUP
OFFSET_IMAGE = OFFSET_TIME + DIM_TIME_GROUP
OFFSET_PROPRIO = OFFSET_IMAGE + DIM_IMAGE_GROUP
OFFSET_PAST_IMAGE = OFFSET_PROPRIO + DIM_PROPRIO_GROUP
OFFSET_VELOCITY = OFFSET_PAST_IMAGE + DIM_PAST_IMAGE_GROUP


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize last axis of a (..., D) array in place-safe manner."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


# ---------------------------------------------------------------------------
# Query feature builder (one row at a time or batched)
# ---------------------------------------------------------------------------

def build_query_feature(
    *,
    cur_cls_left: np.ndarray,   # (768,)  — DINO CLS at time t
    cur_cls_right: np.ndarray,  # (768,)
    past_cls_left: np.ndarray | None,   # (768,) at t-N, or None (zero-padded)
    past_cls_right: np.ndarray | None,
    cur_proprio: np.ndarray,           # (11,)
    proprio_window: np.ndarray,        # (N+1, 11)  zero-padded at the head
    weights: dict[str, float] = DEFAULT_WEIGHTS,
    include_action: bool = False,
    action_chunk: np.ndarray | None = None,   # (H, 10) when include_action=True
    include_time: bool = False,
    progress: float | None = None,            # [0, 1]
) -> np.ndarray:
    """Compose a single FEATURE_DIM vector (float32).

    At training / pool-build time, set ``include_action=True`` and
    ``include_time=True`` with real values. At eval time these two
    groups are typically zero-filled (unknown future / unknown T).
    """
    # action
    if include_action and action_chunk is not None:
        assert action_chunk.shape == (H_CHUNK, DIM_ACTION), (
            f"action_chunk must be ({H_CHUNK}, {DIM_ACTION}); got {action_chunk.shape}"
        )
        a = l2_normalize(action_chunk.reshape(-1).astype(np.float32))
        a = a * np.sqrt(weights["action"])
    else:
        a = np.zeros(DIM_ACTION_GROUP, dtype=np.float32)

    # time
    if include_time and progress is not None:
        tf = np.array(
            [np.sin(np.pi * progress), np.cos(np.pi * progress)], dtype=np.float32
        )
        tf = l2_normalize(tf) * np.sqrt(weights["time"])
    else:
        tf = np.zeros(DIM_TIME_GROUP, dtype=np.float32)

    # image (current)
    img = np.concatenate([cur_cls_left, cur_cls_right]).astype(np.float32)
    img = l2_normalize(img) * np.sqrt(weights["image"])

    # proprio
    prop = l2_normalize(cur_proprio.astype(np.float32)) * np.sqrt(weights["proprio"])

    # past image (zeros when history too short)
    if past_cls_left is not None and past_cls_right is not None:
        past_img = np.concatenate([past_cls_left, past_cls_right]).astype(np.float32)
        past_img = l2_normalize(past_img) * np.sqrt(weights["past_image"])
    else:
        past_img = np.zeros(DIM_PAST_IMAGE_GROUP, dtype=np.float32)

    # velocity (flattened proprio window, pre-padded with zeros by caller)
    assert proprio_window.shape == (N_HIST + 1, DIM_PROPRIO), (
        f"proprio_window must be ({N_HIST + 1}, {DIM_PROPRIO}); got {proprio_window.shape}"
    )
    vel = l2_normalize(proprio_window.reshape(-1).astype(np.float32))
    vel = vel * np.sqrt(weights["velocity"])

    out = np.concatenate([a, tf, img, prop, past_img, vel]).astype(np.float32)
    assert out.shape == (FEATURE_DIM,)
    return out


def build_initial_scene_vec(
    patch_left: np.ndarray,   # (256, 768)
    patch_right: np.ndarray,  # (256, 768)
) -> np.ndarray:
    """Avg-pool patch tokens per camera + concat + L2-normalize. Mirrors
    ``build_retrieval_index.py::_build_initial_scene_index``."""
    left = patch_left.mean(axis=0).astype(np.float32)
    right = patch_right.mean(axis=0).astype(np.float32)
    return l2_normalize(np.concatenate([left, right]))
