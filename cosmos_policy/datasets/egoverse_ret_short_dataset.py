"""EgoVerse mid-training SHORT — action-only compact layout (NO video cond / NO video gen).

Ablation of the v3 compact layout (egoverse_ret_v3_dataset / _v3_state_dataset):
strip BOTH the retrieved-video conditioning block AND the generated future-video
block, keeping only the retrieval-action + current-image conditioning and the
generated action/state chunk. The model still runs the wan2.1 VAE on the packed
image tensor, but only the ONE current-image latent carries real pixels; every
other slot is a tiled action/state/proprio latent (blank pixels overwritten by
replace_latent_with_action_chunk / replace_latent_with_proprio). Result: a tiny
state_t=3 (or 4 with proprio) sequence — a much cheaper "policy head" run that
tests whether the action target trains without the expensive video objective.

Layout (with_proprio=False, state_t=3):
    0  ret_action   (blank 1f;  demo action/state chunk injected)   cond   role RET_ACTION
    1  cur_image    (first query frame x4 -> 1 latent)              cond   role CUR_IMAGE
    2  action       (blank 4f;  query action/state chunk)           GEN    role ACTION
    -- num_conditional_frames = 2 --

Layout (with_proprio=True, state_t=4):
    0  cur_state    (blank 1f;  proprio chunk injected)             cond   role CUR_STATE
    1  ret_action   (blank 4f;  demo action/state chunk injected)   cond   role RET_ACTION
    2  cur_image    (first query frame x4 -> 1 latent)              cond   role CUR_IMAGE
    3  action       (blank 4f;  query action/state chunk)           GEN    role ACTION
    -- num_conditional_frames = 3 --

Pixel packing keeps the causal-VAE contract ``1 + 4*(state_t-1)`` frames (first
latent = 1 frame, the rest = 4 frames each). The current image occupies a 4-frame
slot (repeated), so it maps to a single clean latent token — exactly the
"current image x4 -> one token" the design calls for.

``state_target``:
  * False -> DELTA action57 chunk (base ``_normalize`` + action57_stats.json), like
    EgoVerseRetV3Dataset. ``proprio`` = current absolute state57 (single frame).
  * True  -> ABSOLUTE state57 FUTURE chunk (``_norm_state`` + robust state stats),
    like EgoVerseRetV3StateDataset. ``proprio`` = PAST state57 chunk.

state_t is CONSTANT (3/4) regardless of embodiment (no per-side video length), but
the action/retrieved-action chunk lengths still vary by embodiment (92 human /
60 MD), so BucketedDistributedSampler is still used to keep batches same-shape.
world_model_sample_mask=0 so the action latent gets the velocity/L1 loss (see
docs/egoverse_cosmos2p5_pipeline.md pitfall; demo_sample_action_l1_loss logging).
"""
from __future__ import annotations

import numpy as np
import torch

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_ret_dataset import (
    ACTION_DIM,
    CLASS_TO_COMMAND,
    NUM_DUPLICATES,
)
from cosmos_policy.datasets.egoverse_ret_v2_dataset import (
    NOW_S,
    ROLE_ACTION,
    ROLE_CUR_IMAGE,
    ROLE_CUR_STATE,
    ROLE_RET_ACTION,
    ROLE_RET_VIDEO,
    STATE_DIM,
)
from cosmos_policy.datasets.egoverse_ret_v3_state_dataset import EgoVerseRetV3StateDataset


class EgoVerseRetShortDataset(EgoVerseRetV3StateDataset):
    """Action-only compact dataset (see module doc). Subclasses the STATE dataset so
    both the delta (_normalize/_action_native) and absolute-state (_norm_state/
    _state_rows) machinery are available; ``state_target`` selects which."""

    def __init__(self, *args, state_target: bool = False, with_proprio: bool = True,
                 n_blank_video: int = 0, **kw):
        # For the DELTA variant we don't need the robust state-stats file (proprio is a
        # single current-state frame); only load state stats when they'll be used.
        if not state_target and "robust_state_stats" not in kw:
            kw["robust_state_stats"] = True   # proprio still normalized with robust stats
        super().__init__(*args, **kw)
        self.state_target = bool(state_target)
        self.with_proprio = bool(with_proprio)
        # n_blank_video: insert this many BLANK (zero-pixel) RET_VIDEO latents into the
        # conditioning prefix at PAST t_positions. Trains the backbone to attend past a
        # video slot in the prefix (role RET_VIDEO + non-NOW rope) WITHOUT real video —
        # a cheap structural bridge so the mid transfers to the full-video post. with_proprio only.
        self.n_blank_video = int(n_blank_video)

    def __getitem__(self, idx: int) -> dict:
        r = int(idx) % self.N
        q = int(self.query_idx[r])
        m = self._choose_demo(r)
        return self.build_sample(q, m, idx=idx, aug=True)

    def build_sample(self, q: int, m: int, idx: int = 0, aug: bool = False) -> dict:
        nf_q, _ = self._native(q)
        nf_m, _ = self._native(m)
        fr_q = self._frames_native(q, nf_q)
        fr_m = self._frames_native(m, nf_m)

        # ---- targets + proprio ------------------------------------------------
        if self.state_target:
            span_q = int(self.ww1[q] - self.ww0[q])
            q_tgt = self._norm_state(self._state_rows(q, fr_q + 1), self.wemb[q])   # future state chunk
            m_tgt = self._norm_state(self._state_rows(m, fr_m + 1), self.wemb[m])   # demo future state chunk
            proprio = self._norm_state(self._state_rows(q, fr_q - span_q + 1), self.wemb[q]).reshape(-1)  # past chunk
        else:
            q_a, _ = self._action_native(q, fr_q); q_tgt = self._normalize(q_a, self.wemb[q])   # delta chunk
            m_a, _ = self._action_native(m, fr_m); m_tgt = self._normalize(m_a, self.wemb[m])   # demo delta chunk
            try:
                st = np.asarray(self._grp(q).state("state57")[int(self.ww0[q])], np.float32)
                proprio = self._norm_state(st, self.wemb[q])                                     # current state (57,)
            except Exception:
                proprio = np.zeros(STATE_DIM, np.float32)

        # current image (first query frame), full-frame at final_image_size
        cur_img = self._grid_clip(q, fr_q[:1])                       # (1, S, S, 3)

        # ---- pixel packing: 1 + 4*(T-1) frames (causal VAE), cur_image x4 -> 1 latent
        S = self.final_image_size
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)
        cur4 = np.repeat(cur_img, NUM_DUPLICATES, axis=0)           # (4, S, S, 3)

        ret_video_start = -1
        ret_video_end = -1
        if self.with_proprio:
            nbv = self.n_blank_video
            T = 4 + nbv
            # [cur_state 1f | ret_action 4f | (blank video 4f)*nbv | cur_image 4f | action 4f]
            images = np.concatenate([blank1, blank4] + [blank4] * nbv + [cur4, blank4], axis=0)
            roles = np.array(
                [ROLE_CUR_STATE, ROLE_RET_ACTION] + [ROLE_RET_VIDEO] * nbv
                + [ROLE_CUR_IMAGE, ROLE_ACTION], np.int64)
            retrieved_action_latent_idx = 1
            if nbv:
                ret_video_start, ret_video_end = 2, 2 + nbv     # range [2, 2+nbv)
            current_image_latent_idx = 2 + nbv
            action_latent_idx = 3 + nbv
            current_proprio_latent_idx = 0
            n_cond = 3 + nbv
        else:
            T = 3
            # [ret_action 1f | cur_image 4f | action 4f]
            images = np.concatenate([blank1, cur4, blank4], axis=0)
            roles = np.array([ROLE_RET_ACTION, ROLE_CUR_IMAGE, ROLE_ACTION], np.int64)
            retrieved_action_latent_idx = 0
            current_image_latent_idx = 1
            action_latent_idx = 2
            current_proprio_latent_idx = -1
            n_cond = 2

        assert images.shape[0] == 1 + 4 * (T - 1), (images.shape, T)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=aug, stronger_image_aug=aug)

        # conditioning + action are "now"; blank video latents sit at PAST times so the
        # backbone trains non-NOW rope rotations (idx 2..2+nbv -> NOW_S-nbv .. NOW_S-1).
        tpos = np.full(T, NOW_S, np.float32)
        for j in range(ret_video_end - ret_video_start if ret_video_start >= 0 else 0):
            tpos[ret_video_start + j] = NOW_S - float((ret_video_end - ret_video_start) - j)

        cls = self.wcls[q]
        cmd = f"{CLASS_TO_COMMAND.get(cls, CLASS_TO_COMMAND['other'])} [{self.wemb[q]}]"
        t5 = (torch.squeeze(self.t5_text_embeddings[cmd]) if cmd in self.t5_text_embeddings
              else torch.zeros(512, 1024))

        return {
            "video": images,
            "actions": q_tgt,                                # (nf_q,57) GEN target (delta or future state)
            "action_mask": np.ones(ACTION_DIM, np.float32),
            "retrieved_actions": m_tgt,                      # (nf_m,57) cond (demo chunk)
            "retrieved_action_mask": np.ones(ACTION_DIM, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": idx,
            "embodiment": self.wemb[q], "retrieved_embodiment": self.wemb[m],
            # v2 core
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": n_cond,
            # latent indices — blank ret-video range when n_blank_video>0, else disabled
            "retrieved_video_start_latent_idx": ret_video_start,
            "retrieved_video_end_latent_idx": ret_video_end,
            "retrieved_action_latent_idx": retrieved_action_latent_idx,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": current_image_latent_idx,
            "future_video_start_latent_idx": T,              # empty range -> no future video
            "future_video_end_latent_idx": T,
            "action_latent_idx": action_latent_idx,
            "proprio": proprio,
            "future_proprio": np.zeros(STATE_DIM, np.float32),
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": -1, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": 1, "has_ret_image": 0, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_tgt.copy(),
        }
