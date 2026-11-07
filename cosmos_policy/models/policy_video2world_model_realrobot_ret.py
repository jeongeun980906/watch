"""Cosmos Policy model with real_robot retrieval conditioning.

Single-camera (third_eye) variant for the FFW BG2 bimanual setup. Pairs with
``RealRobotRetDataset`` which produces the latent layout below.

Sequence layout (chunk_size=16, num_dup=4, predict_future_states=True;
ret_image latent dropped to keep RAG layout closer to plain non-RAG fit):
  latent 0  : blank input          (1 frame)               — sentinel
  latent 1  : retrieved action     (4 frames → 1 latent)   — condition (injected)
  latent 2  : current frame        (4 frames → 1 latent)   — condition
  latent 3  : current state        (4 frames → 1 latent)   — condition (proprio injected)
  latent 4  : predicted action     (4 frames → 1 latent)   — generated
  latent 5  : predicted frame      (4 frames → 1 latent)   — generated
  latent 6  : predicted state      (4 frames → 1 latent)   — generated
  ──────────────────────────────────────────────────────────────────────
  state_t = 7   chunk_duration = 1 + 6 * 4 = 25

Conditioning strategy:
  - min_num_conditional_frames=4 covers latents 0-3 (blank + ret_action +
    cur_frame + cur_state).
  - ``_apply_ret_mask`` re-asserts ret_action slot mask based on per-sample
    ``has_ret_data`` flag. ret_video idx fields are -1 so the ret_video
    branch is a no-op.
  - Latent 4 (predicted action) is GENERATED at both training and inference.
  - Latents 5-6 (future frame + future state) are GENERATED targets.

Differences from ``policy_video2world_model_pusht_ret``:
  * Single camera (one cur_image latent, one future image latent — no
    left/right wrist).
  * ret_image latent dropped — ret_action alone preserves the retrieval
    signal while removing an extra mask=1 latent slot that was disrupting
    optimization on the small realrobot dataset.
  * No retrieved_state slot. ``retrieved_state_latent_idx`` is always -1 in
    the data_batch, so ``_inject_retrieved_state`` is a no-op.

We only ship the Rectified Flow / Predict2.5 backbone; the EDM/WAN2.1 path
is not needed for our use case.
"""
from __future__ import annotations

from cosmos_policy.models.policy_video2world_model import (
    CosmosPolicyVideo2WorldModel,
    CosmosPolicyVideo2WorldConfig,
)
from cosmos_policy.models.policy_video2world_model_pusht_ret import (
    _apply_ret_mask,
    _find_condition_in_closure,
    _inject_retrieved_actions,
    _inject_retrieved_state,
)
from cosmos_policy.models.policy_video2world_model_rectified_flow import (
    CosmosPolicyVideo2WorldModelRectifiedFlow,
    CosmosPolicyVideo2WorldConfigRectifiedFlow,
)


# ── EDM / Predict2 (non-RF) ────────────────────────────────────────────────


class CosmosPolicyRealRobotRetConfig(CosmosPolicyVideo2WorldConfig):
    """Config for real_robot-retrieval EDM/Predict2 policy model."""
    pass


class CosmosPolicyRealRobotRetModel(CosmosPolicyVideo2WorldModel):
    """Predict2 (EDM) policy model with retrieval conditioning for real_robot.

    Same control flow as the RF variant — only the base diffusion backbone
    differs.
    """

    def __init__(self, config: CosmosPolicyRealRobotRetConfig):
        super().__init__(config)

    def get_data_and_condition(self, data_batch):
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        _apply_ret_mask(condition, data_batch)
        _inject_retrieved_actions(condition.gt_frames, data_batch)
        _inject_retrieved_state(condition.gt_frames, data_batch)   # no-op (idx=-1)
        _inject_retrieved_actions(latent_state, data_batch)
        _inject_retrieved_state(latent_state, data_batch)          # no-op (idx=-1)
        return raw_state, latent_state, condition

    def get_x0_fn_from_batch(self, data_batch, guidance, **kwargs):
        import os
        result = super().get_x0_fn_from_batch(data_batch, guidance, **kwargs)
        x0_fn = result[0] if isinstance(result, tuple) else result
        condition = _find_condition_in_closure(x0_fn)
        if condition is not None:
            _apply_ret_mask(condition, data_batch)
            if os.environ.get("RET_INJECT_INFERENCE", "0") == "1":
                _inject_retrieved_actions(condition.gt_frames, data_batch)
                _inject_retrieved_state(condition.gt_frames, data_batch)
        return result


# ── Rectified Flow / Predict2.5 ────────────────────────────────────────────


class CosmosPolicyRealRobotRetConfigRectifiedFlow(CosmosPolicyVideo2WorldConfigRectifiedFlow):
    """Config for real_robot-retrieval RF policy model. Inherits base settings."""
    pass


class CosmosPolicyRealRobotRetModelRectifiedFlow(CosmosPolicyVideo2WorldModelRectifiedFlow):
    """Rectified Flow policy model with retrieval conditioning for real_robot.

    Identical control flow to PushT ret model — the only difference is the
    latent layout, which is driven by the data_batch's latent_idx fields.
    """

    def __init__(self, config: CosmosPolicyRealRobotRetConfigRectifiedFlow):
        super().__init__(config)

    def get_data_and_condition(self, data_batch):
        raw_state, latent_state, condition = super().get_data_and_condition(data_batch)
        _apply_ret_mask(condition, data_batch)
        _inject_retrieved_actions(condition.gt_frames, data_batch)
        _inject_retrieved_state(condition.gt_frames, data_batch)   # no-op (idx=-1)
        _inject_retrieved_actions(latent_state, data_batch)
        _inject_retrieved_state(latent_state, data_batch)          # no-op (idx=-1)
        return raw_state, latent_state, condition

    # _apply_ret_loss_mask override removed — clean-replace hook now sets
    # vt = 0 on mask=1 slots, so the velocity MSE at those slots is naturally
    # tiny and there's no extra capacity drain to mask out. Forcing the loss
    # to exactly 0 there was actually hurting the action loss signal mix
    # during early training. Falls back to base no-op.

    def get_velocity_fn_from_batch(self, data_batch, guidance=1.5, **kwargs):
        import os
        result = super().get_velocity_fn_from_batch(data_batch, guidance, **kwargs)
        condition = _find_condition_in_closure(result)
        if condition is not None:
            _apply_ret_mask(condition, data_batch)
            if os.environ.get("RET_INJECT_INFERENCE", "0") == "1":
                _inject_retrieved_actions(condition.gt_frames, data_batch)
                _inject_retrieved_state(condition.gt_frames, data_batch)
        return result
