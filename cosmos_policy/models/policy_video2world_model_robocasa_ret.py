# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Cosmos Policy model for RoboCasa-Doors retrieval-augmented training.

Latent layout (``state_t=15``, ``chunk_duration=57``):
    0  : blank sentinel
    1  : ret wrist            ┐
    2  : ret left agent-view  │ retrieval condition  (real images from MG pool)
    3  : ret right agent-view ┘
    4  : ret proprio          (blank image; ret proprio injected)
    5  : ret action           (blank image; ret action chunk injected)
    6  : cur wrist            ┐
    7  : cur left             │ current condition
    8  : cur right            ┘
    9  : cur proprio          (blank image; cur proprio injected — base model)
    10 : pred action          ← generated
    11 : future wrist         ┐
    12 : future left          │ future state prediction  (generated)
    13 : future right         ┘
    14 : future proprio       ← generated

``min_num_conditional_frames = 10`` makes the base model treat latents 0-9 as
conditioning, so ret-image latents and ret-inject latents are automatically in
the condition region — no custom mask logic is needed here. This model only
has to add two extra injections on top of the RoboCasa baseline:

1. ``ret_actions`` → ``ret_action_latent_idx`` (latent 5)
2. ``ret_proprio`` → ``ret_proprio_latent_idx``  (latent 4)

Compare with ``policy_video2world_model_pusht_ret.py`` — the PushT variant
additionally rewrites the condition mask per-sample to support retrieval
dropout. We skip that here because the dataset zeros out ret arrays when
retrieval is dropped, which produces the same effective conditioning signal
(zero-valued "silent" latents) without the mask bookkeeping.
"""

from __future__ import annotations

import torch

from cosmos_policy.models.policy_text2world_model import (
    replace_latent_with_action_chunk,
    replace_latent_with_proprio,
)
from cosmos_policy.models.policy_video2world_model import (
    CosmosPolicyVideo2WorldConfig,
    CosmosPolicyVideo2WorldModel,
)
from cosmos_policy.models.policy_video2world_model_rectified_flow import (
    CosmosPolicyVideo2WorldConfigRectifiedFlow,
    CosmosPolicyVideo2WorldModelRectifiedFlow,
)


# ---------------------------------------------------------------------------
# Shared retrieval injections
# ---------------------------------------------------------------------------

def _inject_ret_action(x0: torch.Tensor, data_batch: dict) -> None:
    """Write ``ret_actions`` into x0 at ``ret_action_latent_idx``.

    Silent no-op when the index is missing or all -1 (e.g. non-retrieval batch).
    When retrieval is dropped at the dataset level, ``ret_actions`` is zeros,
    so this still runs but writes zeros — equivalent to not injecting.
    """
    if "ret_actions" not in data_batch or "ret_action_latent_idx" not in data_batch:
        return
    idx = data_batch["ret_action_latent_idx"]
    if torch.all(idx == -1):
        return
    replace_latent_with_action_chunk(
        x0,
        data_batch["ret_actions"].to(x0.dtype),
        action_indices=idx,
    )


def _inject_ret_proprio(x0: torch.Tensor, data_batch: dict) -> None:
    if "ret_proprio" not in data_batch or "ret_proprio_latent_idx" not in data_batch:
        return
    idx = data_batch["ret_proprio_latent_idx"]
    if torch.all(idx == -1):
        return
    replace_latent_with_proprio(
        x0,
        data_batch["ret_proprio"].to(x0.dtype),
        proprio_indices=idx,
    )


# ---------------------------------------------------------------------------
# EDM / WAN2.1 variant
# ---------------------------------------------------------------------------

class CosmosPolicyRoboCasaRetConfig(CosmosPolicyVideo2WorldConfig):
    """EDM config — inherits all base settings, nothing to override."""
    pass


class CosmosPolicyRoboCasaRetModel(CosmosPolicyVideo2WorldModel):
    """EDM policy model with RoboCasa→MG retrieval conditioning."""

    def __init__(self, config: CosmosPolicyRoboCasaRetConfig):
        super().__init__(config)

    def get_data_and_condition(self, data_batch):
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        # Inject into both the conditioning tensor (gt_frames) and the latent
        # state used for forward/velocity targets.
        _inject_ret_action(condition.gt_frames, data_batch)
        _inject_ret_proprio(condition.gt_frames, data_batch)
        _inject_ret_action(latent_state, data_batch)
        _inject_ret_proprio(latent_state, data_batch)
        return raw_state, latent_state, condition


# ---------------------------------------------------------------------------
# Rectified Flow / Predict2.5 variant (this is what open_door baseline uses)
# ---------------------------------------------------------------------------

class CosmosPolicyRoboCasaRetConfigRectifiedFlow(CosmosPolicyVideo2WorldConfigRectifiedFlow):
    """RF config — inherits all base settings, nothing to override."""
    pass


class CosmosPolicyRoboCasaRetModelRectifiedFlow(CosmosPolicyVideo2WorldModelRectifiedFlow):
    """Rectified Flow policy model with RoboCasa→MG retrieval conditioning."""

    def __init__(self, config: CosmosPolicyRoboCasaRetConfigRectifiedFlow):
        super().__init__(config)

    def get_data_and_condition(self, data_batch):
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        # Inject into condition.gt_frames (bf16 conditioning tensor) AND
        # latent_state (fp32 x0 used as velocity target). Matches PushT ret.
        _inject_ret_action(condition.gt_frames, data_batch)
        _inject_ret_proprio(condition.gt_frames, data_batch)
        _inject_ret_action(latent_state, data_batch)
        _inject_ret_proprio(latent_state, data_batch)
        return raw_state, latent_state, condition
