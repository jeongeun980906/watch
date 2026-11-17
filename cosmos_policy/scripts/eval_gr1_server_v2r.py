# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Vid2Robot-style eval server: GR1 one-shot eval for models trained with BOTH `no_ret_action=True` and
`no_future=True` (config ..._gr1_ret_v6_448_state_v2r_align_4node). The demo conditions the policy through its
VIDEO only, and the action chunk is the only generated slot, so the serve-time sample uses that layout
(state_t = K+3 = 11, 41 pixel frames), the inference twin of EgoVerseGr1RetV2StateDataset(no_ret_action=True,
no_future=True).build_sample. Retrieval, HTTP API, action decode and CLI are inherited from eval_gr1_server.
(The video + demo-action arm, ..._v2r_align_ra_4node, has the _actonly layout: use eval_gr1_server_actonly.)

Run (gr1_local_eval.sh: SERVER_MOD=eval_gr1_server_v2r):
  python -m cosmos_policy.scripts.eval_gr1_server_v2r --checkpoint <ckpt.pt> \
      --experiment cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_v2r_align_4node --port 8700 ...
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from cosmos_policy.scripts import eval_gr1_server as base

V2_K, V2_DT, NOW_S, NUM_DUP, CHUNK, DEMO_EMB = (
    base.V2_K, base.V2_DT, base.NOW_S, base.NUM_DUP, base.CHUNK, base.DEMO_EMB)
ROLE_CUR_STATE, ROLE_RET_VIDEO, ROLE_CUR_IMAGE, ROLE_ACTION = (
    base.ROLE_CUR_STATE, base.ROLE_RET_VIDEO, base.ROLE_CUR_IMAGE, base.ROLE_ACTION)
preprocess_image, _base_task = base.preprocess_image, base._base_task


class GR1PolicyV2R(base.GR1Policy):
    """Demo video only, action only: no_ret_action + no_future twin of GR1Policy._build_sample_v2."""

    def _build_sample_v2(self, imgs3, proprio, demo_ep, demo_w0, has_ret, task) -> dict:
        # Layout (state_t = K + 3 = 11):
        #   0      cur_state  (state57 chunk, injected)       cond  t=NOW
        #   1..8   ret_video  (native 32f demo, grid-3cam)    cond  t=NOW-8dt..NOW-dt
        #   9      cur_image  (grid-3cam env obs)             cond  t=NOW
        #   -- num_conditional_frames = 10 --
        #   10     action     (32x57, GENERATED)              GEN   t=NOW
        S = self.S
        T = V2_K + 3
        gm = self._grp(demo_ep) if has_ret else None
        m_clip = self._grid_demo_clip32(gm, demo_w0) if has_ret else np.zeros((CHUNK, S, S, 3), np.uint8)
        cur_grid = self._grid3(imgs3)

        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)
        # [state 1f | demo 32f | cur 4f | act 4f] = 41f
        images = np.concatenate([blank1, m_clip, np.repeat(cur_grid[None], NUM_DUP, axis=0), blank4], axis=0)
        assert images.shape[0] == 1 + (T - 1) * NUM_DUP, (images.shape, T)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)

        if self.state_mode:
            cur_state = self._norm(self._past_state_chunk(proprio), self._slo, self._shi).reshape(-1)
        else:
            cur_state = self._norm(base._state57_from_proprio(proprio)[None], self._slo, self._shi)[0]

        roles = np.empty(T, np.int64); tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        roles[1:1 + V2_K] = ROLE_RET_VIDEO
        tpos[1:1 + V2_K] = NOW_S - V2_DT * np.arange(V2_K, 0, -1)
        roles[1 + V2_K], tpos[1 + V2_K] = ROLE_CUR_IMAGE, NOW_S
        roles[2 + V2_K], tpos[2 + V2_K] = ROLE_ACTION, NOW_S

        cls = _base_task(task)
        t5 = (torch.squeeze(torch.as_tensor(self._t5[cls])) if cls in self._t5 else torch.zeros(512, 1024))

        zeros_act = np.zeros((CHUNK, 57), np.float32)
        return {
            "video": images,
            "command": str(cls),
            "actions": zeros_act,
            "action_mask": np.ones(57, np.float32),
            "retrieved_actions": zeros_act.copy(),              # no RET_ACTION slot: never injected
            "retrieved_action_mask": np.ones(57, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "embodiment": "gr1", "retrieved_embodiment": DEMO_EMB,
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": V2_K + 2,                 # 10
            "retrieved_video_start_latent_idx": 1,
            "retrieved_video_end_latent_idx": 1 + V2_K,         # 9
            "retrieved_action_latent_idx": -1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 1 + V2_K,               # 9
            "future_video_start_latent_idx": T,                 # empty range
            "future_video_end_latent_idx": T,
            "action_latent_idx": 2 + V2_K,                      # 10 = the only GEN slot
            "proprio": cur_state.astype(np.float32),
            "future_proprio": np.zeros(57, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": -1, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": has_ret, "has_ret_image": has_ret, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": zeros_act.copy(),
        }


def main() -> None:
    base.GR1Policy = GR1PolicyV2R           # base.main() instantiates GR1Policy -> subclass
    if "--experiment" not in " ".join(sys.argv):
        sys.argv += ["--experiment", "cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_v2r_align_4node"]
    base.main()


if __name__ == "__main__":
    main()
