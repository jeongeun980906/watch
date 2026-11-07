"""Video-only demo-conditioning eval server — one-shot GR1 eval for the model trained
with `no_ret_action=True` (config ..._gr1_ret_v6_448_state_vidonly). The retrieved demo
contributes VIDEO ONLY: the RET_ACTION latent slot is DROPPED, so the serve-time sample
must use the shrunken o=0 layout (state_t=19, 73 pixel frames), matching
EgoVerseGr1RetV2StateDataset(no_ret_action=True).

Base files are NOT modified — this subclasses GR1Policy and overrides `_build_sample_v2`
with the o=0 layout (the exact inference twin of the dataset's video-only build_sample).
Everything else (retrieval, HTTP API, action decode, CLI) is inherited verbatim.

Run:
  uv run --extra cu128 --group egoverse python -m cosmos_policy.scripts.eval_gr1_server_vidonly \
      --checkpoint <vidonly_ckpt_dir> --port 8701
  # defaults injected: --experiment ..._v6_448_state_vidonly, --pairs-dir egoverse_pairs_gr1_v6
  # client is UNCHANGED (standard one-shot-pool3 client hits this server).
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from cosmos_policy.scripts import eval_gr1_server as base

# module constants/helpers reused from the base server (all module-level there)
V2_K, V2_DT, NOW_S, NUM_DUP, CHUNK, DEMO_EMB = (
    base.V2_K, base.V2_DT, base.NOW_S, base.NUM_DUP, base.CHUNK, base.DEMO_EMB)
ROLE_CUR_STATE, ROLE_RET_VIDEO, ROLE_CUR_IMAGE, ROLE_ACTION, ROLE_FUTURE = (
    base.ROLE_CUR_STATE, base.ROLE_RET_VIDEO, base.ROLE_CUR_IMAGE, base.ROLE_ACTION, base.ROLE_FUTURE)
preprocess_image, _base_task = base.preprocess_image, base._base_task


class GR1PolicyVidOnly(base.GR1Policy):
    """GR1 one-shot policy whose retrieved-demo conditioning is VIDEO ONLY (no RET_ACTION
    slot). o=0 twin of GR1Policy._build_sample_v2."""

    def _build_sample_v2(self, imgs3, proprio, demo_ep, demo_w0, has_ret, task) -> dict:
        # o = 0: NO RET_ACTION slot. Layout (state_t = 2*K+3 = 19):
        #   0        cur_state                                cond  t=NOW
        #   1..8     ret_video (native 32f demo, grid-3cam)   cond  t=NOW-8dt..NOW-dt
        #   9        cur_image                                cond  t=NOW
        #   -- num_conditional_frames = K+2 = 10 --
        #   10       action    (GENERATED -> zeros)           GEN   t=NOW
        #   11..18   future    (GENERATED -> zeros)           GEN   t=NOW+dt..NOW+8dt
        S = self.S
        o = 0
        T = 2 * V2_K + 3                                          # 19
        gm = self._grp(demo_ep) if has_ret else None
        m_clip = self._grid_demo_clip32(gm, demo_w0) if has_ret \
            else np.zeros((CHUNK, S, S, 3), np.uint8)
        cur_grid = self._grid3(imgs3)

        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)
        # pixel packing (NO ret_act 4f): [state 1f | demo 32f | cur 4f | act 4f | future 32f] = 73f
        images = np.concatenate([
            blank1, m_clip,
            np.repeat(cur_grid[None], NUM_DUP, axis=0),
            blank4, np.zeros((CHUNK, S, S, 3), np.uint8),         # future GENERATED -> zeros
        ], axis=0)
        assert images.shape[0] == 1 + (T - 1) * NUM_DUP, (images.shape, T)   # 73
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)

        # retrieved-demo chunk: computed EXACTLY like the base (state/cmd_delta/action
        # branches) so the sample matches the dataset byte-for-byte; the model ignores it
        # here since retrieved_action_latent_idx=-1, but we keep it faithful.
        if has_ret and self.state_mode:
            m_state = gm.state("state57")
            ret32 = np.stack([m_state[min(max(demo_w0 + 1 + k, 0), len(m_state) - 1)]
                              for k in range(CHUNK)])
            ret_act = self._norm(ret32, self._mslo, self._mshi)
        elif has_ret and self.pred_target == "cmd_delta57" and not (
                self.plate_state_delta and task.startswith("PnPSinkToRightCounterPlate")):
            from cosmos_policy.datasets.egoverse_gr1_ret_v2_cmddelta_dataset import cmd_delta57 as _cd57
            m_abs = gm.action("abs_action57")
            rows = np.stack([m_abs[min(max(demo_w0 + k, 0), len(m_abs) - 1)]
                             for k in range(CHUNK + 1)])
            ret_act = self._norm(_cd57(rows), self._mcdlo, self._mcdhi)
        elif has_ret:
            m_act = gm.action("action57")
            ret32 = np.stack([m_act[min(max(demo_w0 + k, 0), len(m_act) - 1)]
                              for k in range(CHUNK)])
            ret_act = self._norm(ret32, self._mlo, self._mhi)
        else:
            ret_act = np.zeros((CHUNK, 57), np.float32)

        if self.state_mode:
            cur_state = self._norm(self._past_state_chunk(proprio), self._slo, self._shi).reshape(-1)
        else:
            cur_state = self._norm(base._state57_from_proprio(proprio)[None], self._slo, self._shi)[0]

        # slot roles + rope positions — o=0 (no RET_ACTION)
        roles = np.empty(T, np.int64); tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        roles[1:1 + V2_K] = ROLE_RET_VIDEO
        tpos[1:1 + V2_K] = NOW_S - V2_DT * np.arange(V2_K, 0, -1)
        roles[1 + V2_K], tpos[1 + V2_K] = ROLE_CUR_IMAGE, NOW_S
        roles[2 + V2_K], tpos[2 + V2_K] = ROLE_ACTION, NOW_S
        roles[3 + V2_K:], tpos[3 + V2_K:] = ROLE_FUTURE, NOW_S + V2_DT * np.arange(1, V2_K + 1)

        cls = _base_task(task)
        t5 = (torch.squeeze(torch.as_tensor(self._t5[cls]))
              if cls in self._t5 else torch.zeros(512, 1024))

        zeros_act = np.zeros((CHUNK, 57), np.float32)
        return {
            "video": images,
            "command": str(cls),
            "actions": zeros_act,
            "action_mask": np.ones(57, np.float32),
            "retrieved_actions": ret_act.astype(np.float32),      # unused (idx=-1), kept faithful
            "retrieved_action_mask": np.ones(57, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "embodiment": "gr1", "retrieved_embodiment": DEMO_EMB,
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": V2_K + 2,                   # 10
            "retrieved_video_start_latent_idx": 1,
            "retrieved_video_end_latent_idx": 1 + V2_K,           # 9
            "retrieved_action_latent_idx": -1,                    # DROPPED (video-only)
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 1 + V2_K,                 # 9
            "future_video_start_latent_idx": 3 + V2_K,            # 11
            "future_video_end_latent_idx": T,                     # 19
            "action_latent_idx": 2 + V2_K,                        # 10
            "proprio": cur_state.astype(np.float32),
            "future_proprio": np.zeros(57, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": 3 + V2_K, "future_image2_latent_idx": -1,   # 11
            "value_latent_idx": -1,
            "has_ret_data": has_ret, "has_ret_image": has_ret, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": zeros_act.copy(),
        }


def main() -> None:
    base.GR1Policy = GR1PolicyVidOnly       # base.main() instantiates GR1Policy -> subclass
    argstr = " ".join(sys.argv)
    if "--experiment" not in argstr:
        sys.argv += ["--experiment", "cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_vidonly"]
    if "--pairs-dir" not in argstr:
        sys.argv += ["--pairs-dir", "/mnt/ddn/dataset/egoverse_pairs_gr1_v6"]
    base.main()


if __name__ == "__main__":
    main()
