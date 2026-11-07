# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""15-latent layout assembly for RoboCasa door retrieval eval.

Layout (matches ``RoboCasaDoorsRetDataset.__getitem__`` and
``policy_video2world_model_robocasa_ret.py``):

    latent  0 : blank sentinel
    latent  1 : ret wrist            ┐
    latent  2 : ret left             │  retrieval condition
    latent  3 : ret right            ┘
    latent  4 : ret proprio          (blank image; proprio injected by model)
    latent  5 : ret action           (blank image; action chunk injected)
    latent  6 : cur wrist            ┐
    latent  7 : cur left             │  current condition
    latent  8 : cur right            ┘
    latent  9 : cur proprio          (blank image; proprio injected)
    latent 10 : pred action          ← generated
    latent 11 : future wrist         ┐
    latent 12 : future left          │  generated future frames
    latent 13 : future right         ┘
    latent 14 : future proprio       ← generated

state_t = 15, chunk_duration = 1 + 14 * num_duplicates.

This module provides the helper that builds the image sequence and the
latent-index dict that the retrieval-augmented model's
``get_data_and_condition`` expects as ``data_batch``.
"""

from __future__ import annotations

import numpy as np

from cosmos_policy.utils.utils import duplicate_array

# Canonical latent indices. Use these constants everywhere so training and
# eval never drift out of sync.
LATENT_BLANK              = 0
LATENT_RET_WRIST          = 1
LATENT_RET_LEFT           = 2
LATENT_RET_RIGHT          = 3
LATENT_RET_PROPRIO        = 4
LATENT_RET_ACTION         = 5
LATENT_CUR_WRIST          = 6
LATENT_CUR_LEFT           = 7
LATENT_CUR_RIGHT          = 8
LATENT_CUR_PROPRIO        = 9
LATENT_PRED_ACTION        = 10
LATENT_FUTURE_WRIST       = 11
LATENT_FUTURE_LEFT        = 12
LATENT_FUTURE_RIGHT       = 13
LATENT_FUTURE_PROPRIO     = 14

STATE_T = 15
NUM_CONDITIONAL_LATENTS = 10   # latents 0-9 conditioned


def build_image_sequence(
    cur_wrist: np.ndarray,
    cur_left: np.ndarray,
    cur_right: np.ndarray,
    ret_wrist: np.ndarray | None,
    ret_left: np.ndarray | None,
    ret_right: np.ndarray | None,
    num_duplicates: int = 4,
) -> tuple[np.ndarray, dict[str, int]]:
    """Build the 15-latent image sequence + index dict for one eval step.

    Future frames are blank — the model generates them. If the current
    chunk has no retrieval (e.g. dropped), pass ``None`` for the ret
    images and the corresponding latents become blank.

    Returns
    -------
    images : np.ndarray, (1 + 14·num_duplicates, H, W, 3) uint8
    indices : dict mapping sample-dict-key → latent index (compatible with
              ``RoboCasaDoorsRetDataset.__getitem__`` output).
    """
    # All images share a common H/W; use cur_left as the template for blanks.
    ref = cur_left
    blank_dup = duplicate_array(np.zeros_like(ref), total_num_copies=num_duplicates)

    def dup(img: np.ndarray) -> np.ndarray:
        return duplicate_array(img, total_num_copies=num_duplicates)

    ret_wrist = blank_dup if ret_wrist is None else dup(ret_wrist)
    ret_left  = blank_dup if ret_left  is None else dup(ret_left)
    ret_right = blank_dup if ret_right is None else dup(ret_right)

    image_list = [
        np.expand_dims(np.zeros_like(ref), axis=0),   # latent 0 — blank sentinel (1 frame)
        ret_wrist,                                     # 1
        ret_left,                                      # 2
        ret_right,                                     # 3
        blank_dup.copy(),                              # 4  ret proprio (inject)
        blank_dup.copy(),                              # 5  ret action (inject)
        dup(cur_wrist),                                # 6
        dup(cur_left),                                 # 7
        dup(cur_right),                                # 8
        blank_dup.copy(),                              # 9  cur proprio (inject)
        blank_dup.copy(),                              # 10 pred action (generated)
        blank_dup.copy(),                              # 11 future wrist (generated)
        blank_dup.copy(),                              # 12 future left (generated)
        blank_dup.copy(),                              # 13 future right (generated)
        blank_dup.copy(),                              # 14 future proprio (generated)
    ]

    indices = {
        "ret_wrist_image_latent_idx":     LATENT_RET_WRIST,
        "ret_left_image_latent_idx":      LATENT_RET_LEFT,
        "ret_right_image_latent_idx":     LATENT_RET_RIGHT,
        "ret_proprio_latent_idx":         LATENT_RET_PROPRIO,
        "ret_action_latent_idx":          LATENT_RET_ACTION,
        "current_wrist_image_latent_idx": LATENT_CUR_WRIST,
        "current_image_latent_idx":       LATENT_CUR_LEFT,
        "current_image2_latent_idx":      LATENT_CUR_RIGHT,
        "current_proprio_latent_idx":     LATENT_CUR_PROPRIO,
        "action_latent_idx":              LATENT_PRED_ACTION,
        "future_wrist_image_latent_idx":  LATENT_FUTURE_WRIST,
        "future_image_latent_idx":        LATENT_FUTURE_LEFT,
        "future_image2_latent_idx":       LATENT_FUTURE_RIGHT,
        "future_proprio_latent_idx":      LATENT_FUTURE_PROPRIO,
        "value_latent_idx":               -1,          # no value prediction
    }
    return np.concatenate(image_list, axis=0), indices
