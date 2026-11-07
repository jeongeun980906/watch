"""GR1 POST-training PLAIN — non-retrieval policy + video prediction (abs_action57).

The post-training counterpart of ``egoverse_plain_dataset`` (mid). Identical to
``EgoVerseGr1RetV2StateDataset`` — the v6 ABS-COMMAND recipe (``target_group=
"abs_action57"``) — except the RETRIEVAL channel is removed: no ret_action slot,
no ret_video block. The policy is conditioned only on the current observation:

    slot:  cur_state | cur_image | action | future (K=8)
    role:      0     |     3     |   4    |   5
    time:     NOW    |    NOW    |  NOW   | NOW+dt .. NOW+8*dt
    cond:     yes    |    yes    |  GEN   |  GEN
    state_t = K+3 = 11          (retrieval layout: 20)
    chunk_duration = 41 frames  (retrieval layout: 77)
    num_conditional_frames = 2  (retrieval layout: 11)

Target and normalization are inherited UNCHANGED from the v6 state recipe:
  * ``actions``  = FUTURE absolute COMMAND chunk act[w0 .. w0+31] from
    ``target_group`` (abs_action57), normalized with ``target_stats``.
  * ``proprio``  = PAST state57 chunk s[w0-31 .. w0], normalized with
    ``state_stats``, flattened (32*57) and tiled into slot 0.
  * No per-frame jitter — v6 pairs already enumerate every timestep.

WHY THIS ARM EXISTS. Paired with the retrieval post-train it forms the demo
ablation the mid-training A/B needs in order to be readable at all:

  * ``mid=transfer x post=plain`` (warm-started from the transfer mid) asks
    whether the transfer pre-training benefit SURVIVES deployment without a
    demo — i.e. whether retrieval can be dropped from the serving path.
  * ``post=plain`` vs ``post=transfer`` at the same mid directly measures how
    much the retrieved demo contributes. ``gr1_data_eval_redesign.md`` §0 defect
    #3 reports demo-blindness Delta = 0.006 on this data, so if these two match,
    the demo contributes nothing and no mid-training comparison phrased as
    "which helps the transfer task" can be interpreted.

Layout is data-driven (``*_latent_idx`` keys), so NO model code changes. The
CLOSED-LOOP eval server DOES need a matching builder (see
``eval_gr1_server.py::_build_sample_plain``); offline eval does not.
"""
from __future__ import annotations

import numpy as np
import torch

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_gr1_ret_dataset import (
    ACTION_DIM,
    STATE_DIM,
    NUM_DUPLICATES,
)
from cosmos_policy.datasets.egoverse_gr1_ret_v2_dataset import DT, K, NATIVE
from cosmos_policy.datasets.egoverse_gr1_ret_v2_state_dataset import EgoVerseGr1RetV2StateDataset
from cosmos_policy.datasets.egoverse_ret_v2_dataset import (
    NOW_S,
    ROLE_ACTION,
    ROLE_CUR_IMAGE,
    ROLE_CUR_STATE,
    ROLE_FUTURE,
)


class EgoVerseGr1PlainStateDataset(EgoVerseGr1RetV2StateDataset):
    """Plain (non-retrieval) GR1 post-train dataset, abs_action57 target."""

    # ---- layout ----
    @property
    def state_t(self) -> int:            # cur_state + cur_image + action + K future
        return K + 3

    @property
    def chunk_duration(self) -> int:     # 41 pixel frames
        return 1 + (self.state_t - 1) * NUM_DUPLICATES

    def bucket_key(self, r: int):        # single shape -> one bucket
        return ("gr1", "plain")

    def __getitem__(self, idx: int) -> dict:
        if self.debug:
            idx = 0
        q = int(self.query_idx[idx % self.N])
        return self.build_sample(q, q, idx=idx, has_ret=0, aug=True, rand_delta=True)

    def build_sample(self, q: int, m: int = -1, idx: int = 0, has_ret: int = 0,
                     aug: bool = False, rand_delta: bool = False) -> dict:
        """``m`` / ``has_ret`` / ``rand_delta`` accepted for signature-compat with the
        retrieval dataset (offline batch eval calls it positionally) and IGNORED —
        there is no demo in this layout."""
        del m, has_ret, rand_delta
        gq = self._grp(q)
        qw0 = int(self.ww0[q])
        q_emb = self.wemb[q]
        q_state = gq.state(self.state_group)

        # target: FUTURE absolute COMMAND chunk act[qw0 .. qw0+31] (v6 abs_action57);
        # legacy (target_group=None) = achieved state chunk s[qw0+1 .. qw0+32].
        if self.target_group is not None:
            q_tgt = self._chunk(np.asarray(gq.action(self.target_group), np.float64), qw0)
        else:
            q_tgt = self._chunk(q_state, qw0 + 1)
        # slot 0: PAST state chunk s[qw0-31 .. qw0] (left-pad with s[0] when qw0 < 31)
        past = self._past_chunk(q_state, qw0, NATIVE).astype(np.float32)
        if self.normalize_state:
            _tstats = self.target_stats if (self.target_group and self.target_stats) else self.state_stats
            q_tgt = self._norm(q_tgt, _tstats, q_emb)
            past = self._norm(past, self.state_stats, q_emb)

        # video: [cur_state 1f | cur_image 4f | action 4f | future 32f] = 41f
        q_clip = self._grid_clip(gq, qw0)
        S = self.final_image_size
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)
        images = np.concatenate([
            blank1,                                              # cur_state
            np.repeat(q_clip[:1], NUM_DUPLICATES, axis=0),       # cur_image x4 -> 1 latent
            blank4,                                              # action
            q_clip,                                              # future video (K latents)
        ], axis=0)
        assert images.shape[0] == self.chunk_duration, (images.shape, self.chunk_duration)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=aug, stronger_image_aug=aug)

        T = self.state_t
        roles = np.empty(T, np.int64)
        tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        roles[1], tpos[1] = ROLE_CUR_IMAGE, NOW_S
        roles[2], tpos[2] = ROLE_ACTION, NOW_S
        roles[3:], tpos[3:] = ROLE_FUTURE, NOW_S + DT * np.arange(1, K + 1)

        cls = self.wcls[q]
        t5 = (torch.squeeze(self.t5_text_embeddings[cls]) if cls in self.t5_text_embeddings
              else torch.zeros(512, 1024))

        return {
            "video": images,
            "command": str(cls),
            "actions": q_tgt.astype(np.float32),             # (32,57) abs_action57 chunk (GEN)
            "action_mask": np.ones(ACTION_DIM, np.float32),
            # NO retrieved_actions — injection helpers return early when absent.
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": idx,
            "embodiment": q_emb, "retrieved_embodiment": q_emb,
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": 2,                     # cur_state + cur_image
            # retrieval fully disabled, future video ACTIVE
            "retrieved_video_start_latent_idx": -1,
            "retrieved_video_end_latent_idx": -1,
            "retrieved_action_latent_idx": -1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 1,
            "future_video_start_latent_idx": 3,
            "future_video_end_latent_idx": T,
            "action_latent_idx": 2,
            "proprio": past.reshape(-1).astype(np.float32),  # (32*57,) tiled into slot 0
            "future_proprio": np.zeros(STATE_DIM, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": 3, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": 0, "has_ret_image": 0, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_tgt.copy().astype(np.float32),
        }
