# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""GR1 post-training SHORT layout — NO video conditioning / NO video generation.

GR1 counterpart of EgoVerseRetShortDataset (egoverse_ret_short_dataset.py): the
demo-video and future-video blocks are stripped from the v2 GR1 layout, keeping
only retrieval-action + current-image conditioning and the generated 32x57 chunk.
Post-trains FROM the "short" mid checkpoints (cosmos_predict2p5_2b_egoverse_ret_
short_v11 / _short_state_v11), which were trained in exactly these layouts.

Layout (state variant, with_proprio=True, state_t=4):
    0  cur_state   (blank 1f; PAST state chunk tiled)        cond  ROLE_CUR_STATE
    1  ret_action  (blank 4f; demo chunk tiled)              cond  ROLE_RET_ACTION
    2  cur_image   (current 4-view grid x4 -> 1 latent)      cond  ROLE_CUR_IMAGE
    3  action      (blank 4f; GEN chunk)                     GEN   ROLE_ACTION
    -- num_conditional_frames = 3 --   pixels: 1+4*3 = 13

Layout (delta variant, with_proprio=False, state_t=3):
    0  ret_action  (blank 1f)                                cond  ROLE_RET_ACTION
    1  cur_image   (current 4-view grid x4 -> 1 latent)      cond  ROLE_CUR_IMAGE
    2  action      (blank 4f)                                GEN   ROLE_ACTION
    -- num_conditional_frames = 2 --   pixels: 1+4*2 = 9

Targets:
  * EgoVerseGr1RetShortStateDataset  -> abs_action57 chunk (v6 state recipe:
        pctl norm, demo cond = demo future state chunk, proprio = past chunk).
  * EgoVerseGr1RetShortHybridDataset -> hybrid delta (Plate*=legacy state-delta,
        else cmd-delta; norm inside _act_chunk), demo cond = demo hybrid chunk.

world_model_sample_mask=0 (policy). bucket_key single bucket (fixed shapes).
"""
from __future__ import annotations

import numpy as np
import torch

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_gr1_ret_dataset import ACTION_DIM, STATE_DIM
from cosmos_policy.datasets.egoverse_gr1_ret_v2_dataset import (
    NATIVE,
    NUM_DUPLICATES,
    EgoVerseGr1RetV2Dataset,
)
from cosmos_policy.datasets.egoverse_gr1_ret_v2_cmddelta_dataset import (
    EgoVerseGr1RetV2HybridDeltaDataset,
)
from cosmos_policy.datasets.egoverse_gr1_ret_v2_state_dataset import (
    EgoVerseGr1RetV2StateDataset,
)
from cosmos_policy.datasets.egoverse_ret_v2_dataset import (
    NOW_S,
    ROLE_ACTION,
    ROLE_CUR_IMAGE,
    ROLE_CUR_STATE,
    ROLE_RET_ACTION,
)


class _Gr1ShortPackMixin:
    """Short-layout packing + sample assembly shared by both target variants.

    Consumers must define ``with_proprio`` (bool) before use and provide the GR1
    base-dataset machinery (_grp/_grid_frame/wemb/wcls/t5_text_embeddings/...).
    """

    with_proprio: bool = True

    # ---- layout (overrides the v2 20-slot values) ----
    @property
    def state_t(self) -> int:
        return 4 if self.with_proprio else 3

    @property
    def chunk_duration(self) -> int:
        return 1 + (self.state_t - 1) * NUM_DUPLICATES

    def _pack_short(self, gq, qw0: int, aug: bool) -> tuple[np.ndarray, dict]:
        """(preprocessed pixel tensor, layout fields) for the short sequence."""
        S = self.final_image_size
        cur1 = self._grid_frame(gq, qw0)[None]                    # (1,S,S,3)
        cur4 = np.repeat(cur1, NUM_DUPLICATES, axis=0)
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)

        if self.with_proprio:
            T = 4
            images = np.concatenate([blank1, blank4, cur4, blank4], axis=0)
            roles = np.array([ROLE_CUR_STATE, ROLE_RET_ACTION, ROLE_CUR_IMAGE, ROLE_ACTION], np.int64)
            lay = dict(retrieved_action_latent_idx=1, current_image_latent_idx=2,
                       action_latent_idx=3, current_proprio_latent_idx=0,
                       num_conditional_frames=3)
        else:
            T = 3
            images = np.concatenate([blank1, cur4, blank4], axis=0)
            roles = np.array([ROLE_RET_ACTION, ROLE_CUR_IMAGE, ROLE_ACTION], np.int64)
            lay = dict(retrieved_action_latent_idx=0, current_image_latent_idx=1,
                       action_latent_idx=2, current_proprio_latent_idx=-1,
                       num_conditional_frames=2)

        assert images.shape[0] == self.chunk_duration, (images.shape, T)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=aug, stronger_image_aug=aug)
        lay.update(slot_roles=roles, t_positions=np.full(T, NOW_S, np.float32), T=T)
        return images, lay

    def _short_sample(self, q: int, m: int, idx: int, has_ret: int,
                      q_tgt: np.ndarray, m_ret: np.ndarray, proprio: np.ndarray,
                      images: np.ndarray, lay: dict) -> dict:
        S = self.final_image_size
        cls = self.wcls[q]
        t5 = (torch.squeeze(self.t5_text_embeddings[cls]) if cls in self.t5_text_embeddings
              else torch.zeros(512, 1024))
        T = lay["T"]
        return {
            "video": images,
            "command": str(cls),
            "actions": q_tgt.astype(np.float32),
            "action_mask": np.ones(ACTION_DIM, np.float32),
            "retrieved_actions": m_ret.astype(np.float32),
            "retrieved_action_mask": np.ones(ACTION_DIM, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": idx,
            "embodiment": self.wemb[q], "retrieved_embodiment": self.wemb[m],
            "t_positions": lay["t_positions"], "slot_roles": lay["slot_roles"],
            "num_conditional_frames": lay["num_conditional_frames"],
            # NO video blocks: demo/future video ranges disabled
            "retrieved_video_start_latent_idx": -1,
            "retrieved_video_end_latent_idx": -1,
            "retrieved_action_latent_idx": lay["retrieved_action_latent_idx"],
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": lay["current_image_latent_idx"],
            "future_video_start_latent_idx": T,
            "future_video_end_latent_idx": T,
            "action_latent_idx": lay["action_latent_idx"],
            "proprio": proprio.astype(np.float32),
            "future_proprio": np.zeros(STATE_DIM, np.float32),
            "current_proprio_latent_idx": lay["current_proprio_latent_idx"],
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": -1, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": has_ret, "has_ret_image": 0, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_tgt.copy().astype(np.float32),
        }


class EgoVerseGr1RetShortStateDataset(_Gr1ShortPackMixin, EgoVerseGr1RetV2StateDataset):
    """SHORT layout, abs_action57 target (v6 state recipe). with_proprio=True."""

    with_proprio = True

    def build_sample(self, q: int, m: int, idx: int = 0, has_ret: int = 1,
                     aug: bool = False, rand_delta: bool = False) -> dict:
        gq = self._grp(q); gm = self._grp(m)
        qw0 = int(self.ww0[q]); mw0 = int(self.ww0[m])
        q_emb, m_emb = self.wemb[q], self.wemb[m]

        q_state = gq.state(self.state_group)
        m_state = gm.state(self.state_group)
        if self.target_group is not None:
            q_tgt = self._chunk(np.asarray(gq.action(self.target_group), np.float64), qw0)
        else:
            q_tgt = self._chunk(q_state, qw0 + 1)
        m_ret = self._chunk(m_state, mw0 + 1)
        if self.normalize_state:
            _tstats = self.target_stats if (self.target_group and self.target_stats) else self.state_stats
            q_tgt = self._norm(q_tgt, _tstats, q_emb)
            m_ret = self._norm(m_ret, self.state_stats, m_emb) if has_ret \
                else np.zeros((NATIVE, STATE_DIM), np.float32)
        elif not has_ret:
            m_ret = np.zeros((NATIVE, STATE_DIM), np.float32)
        past = self._past_chunk(q_state, qw0, NATIVE).astype(np.float32)
        if self.normalize_state:
            past = self._norm(past, self.state_stats, q_emb)

        images, lay = self._pack_short(gq, qw0, aug)
        return self._short_sample(q, m, idx, has_ret, q_tgt, m_ret,
                                  past.reshape(-1), images, lay)


class EgoVerseGr1RetShortHybridDataset(_Gr1ShortPackMixin, EgoVerseGr1RetV2HybridDeltaDataset):
    """SHORT layout, hybrid delta target (_act_chunk normalizes internally).
    with_proprio=True (2026-07-20): every GR1 post-train feeds current state — the
    delta short now carries the PAST state chunk in slot 0 (state_t=4), identical
    input to the abs short so the two differ only by target."""

    with_proprio = True

    def _past_state_chunk(self, gq, qw0: int, emb) -> np.ndarray:
        """PAST state chunk s[qw0-31 .. qw0], gr1-normalized, flattened (32*57,)."""
        st = gq.state(self.state_group)
        idx = np.arange(qw0 - NATIVE + 1, qw0 + 1)
        past = np.asarray(st[np.clip(idx, 0, len(st) - 1)], np.float32)
        if self.normalize_state:
            past = self._norm(past, self.state_stats, emb)
        return past.reshape(-1)

    def __getitem__(self, idx: int) -> dict:
        if self.debug:
            idx = 0
        r = idx % self.N
        q = int(self.query_idx[r])
        ms = self.match_idx[r]; ms = ms[ms >= 0]
        has_ret = 1 if len(ms) else 0
        m = int(ms[np.random.randint(len(ms))]) if has_ret else q

        gq = self._grp(q); gm = self._grp(m)
        qw0 = int(self.ww0[q]); mw0 = int(self.ww0[m])

        q_tgt = self._act_chunk(gq, qw0)
        m_ret = self._act_chunk(gm, mw0) if has_ret \
            else np.zeros((NATIVE, ACTION_DIM), np.float32)

        proprio = self._past_state_chunk(gq, qw0, self.wemb[q])
        images, lay = self._pack_short(gq, qw0, aug=True)
        return self._short_sample(q, m, idx, has_ret, q_tgt, m_ret,
                                  proprio, images, lay)


# The state variant keeps EgoVerseGr1RetV2StateDataset.__getitem__ (delegates to
# build_sample with aug=True), which resolves to the SHORT build_sample above.
