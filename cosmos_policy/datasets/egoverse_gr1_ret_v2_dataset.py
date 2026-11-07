"""GR1 post-training dataset in the mid-v2 LENGTH-GENERAL layout (docs/egoverse_mid_v2_design.md).

Same contract as EgoVerseRetV2Dataset (compact sequence, per-token slot ROLES +
CONTINUOUS-TIME rope positions), instantiated for the GR1 pairs at their native
32f@20Hz: k = g = 8 latents, T = 20, num_conditional_frames = 11.

    0     cur_state  (state57, gr1-normalized)        cond   t=NOW
    1     ret_action (32x57 demo chunk, MD-normalized) cond  t=NOW
    2..9  ret_video  (native 32f demo, grid-3cam)     cond   t=NOW-8*dt..NOW-dt
    10    cur_image  (grid-3cam)                      cond   t=NOW
    -- num_conditional_frames = 11 --
    11    action     (32x57, GENERATED)               GEN    t=NOW
    12..19 future    (native 32f query, GENERATED)    GEN    t=NOW+dt..NOW+8*dt

This is EXACTLY the format the mid-v2 model was probed with (iter-1000 warm-start
GR1 action_l1 0.330 == in-distribution mid-MD 0.333). Warm-start from the mid-v2
checkpoint; the layout/time-axis transfer is zero-gap by design.
world_model_sample_mask=0 (policy). Exposes bucket_key() (single bucket) so
scripts/train.py picks the BucketedDistributedSampler branch.
"""
from __future__ import annotations

import numpy as np
import torch

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_gr1_ret_dataset import (
    ACTION_DIM,
    STATE_DIM,
    CENTERTOP,
    EgoVerseGr1RetDataset,
    FRONT,
    SIDELEFT,
    SIDERIGHT,
    NUM_DUPLICATES,
    _decode,
)
from cosmos_policy.datasets.egoverse_ret_v2_dataset import (
    NOW_S,
    ROLE_ACTION,
    ROLE_CUR_IMAGE,
    ROLE_CUR_STATE,
    ROLE_FUTURE,
    ROLE_RET_ACTION,
    ROLE_RET_VIDEO,
)

NATIVE = 32                    # GR1 window @20Hz
K = NATIVE // NUM_DUPLICATES   # 8 latents per video block
DT = 4.0 / 20.0                # seconds per latent


class EgoVerseGr1RetV2Dataset(EgoVerseGr1RetDataset):
    """GR1 pairs rendered in the mid-v2 compact role/timestamp layout (see module doc)."""

    # 4-view (center_top in BR) was the 2026-07-16 default; set use_center_top=False to
    # revert to the original 3-view grid (BR quadrant blank) — center_top is a
    # low-information ceiling/upper-cabinet view for these tasks (user 2026-07-20).
    def __init__(self, *args, use_center_top: bool = True, **kw):
        super().__init__(*args, **kw)
        self.use_center_top = bool(use_center_top)

    # ---- layout ----
    @property
    def state_t(self) -> int:            # 1+1+8+1+1+8
        return 2 * K + 4

    @property
    def chunk_duration(self) -> int:     # 77 pixel frames
        return 1 + (self.state_t - 1) * NUM_DUPLICATES

    def bucket_key(self, r: int):        # single shape -> one bucket
        return ("gr1", "md")

    # ---- grid-3cam ----
    def _grid_frame(self, g, i: int) -> np.ndarray:
        S = self.final_image_size
        h = S // 2
        i = min(max(i, 0), g.n_frames - 1)
        canvas = np.zeros((S, S, 3), np.uint8)
        # 4-view grid (2026-07-16): the (h,h) quadrant was zeros; center_top fills it.
        # Zarrs without images.center_top fall back to a black tile (try/except below),
        # so pre-backfill episodes and old runs stay readable. use_center_top=False ->
        # original 3-view (BR quadrant stays blank).
        tiles = [((0, 0), FRONT), ((0, h), SIDELEFT), ((h, 0), SIDERIGHT)]
        if getattr(self, "use_center_top", True):
            tiles.append(((h, h), CENTERTOP))
        for (r0, c0), cam in tiles:
            try:
                b = g.frame(i, cam)
                if b:
                    canvas[r0:r0 + h, c0:c0 + h] = _decode(b, h)
            except Exception:
                pass
        return canvas

    def _grid_clip(self, g, w0: int) -> np.ndarray:
        return np.stack([self._grid_frame(g, w0 + k) for k in range(NATIVE)])



    def _act_chunk(self, g, w0: int) -> np.ndarray:
        """Action-target hook: native chunk of self.action_group rows by default."""
        return self._chunk(g.action(self.action_group), w0)

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

        # per-frame t+k lockstep delta (train every timestep; same rule as the base class)
        def _room(g, w0):
            top = min(g.action_rows(self.action_group), g.n_frames - 1)
            return top - (w0 + NATIVE)
        dmax = min(self.per_frame_stride - 1, _room(gq, qw0), _room(gm, mw0))
        delta = np.random.randint(0, dmax + 1) if dmax > 0 else 0
        qw0 += delta; mw0 += delta

        # actions (native 32, normalized per-embodiment) — _act_chunk is a hook so
        # subclasses can redefine the action target (cmd-delta, 2026-07-18)
        q_act = self._act_chunk(gq, qw0)
        m_act = self._act_chunk(gm, mw0)
        q_emb, m_emb = self.wemb[q], self.wemb[m]
        if self.normalize_actions:
            q_act = self._norm(q_act, self.act_stats, q_emb)
            m_act = self._norm(m_act, self.act_stats, m_emb) if has_ret \
                else np.zeros((NATIVE, ACTION_DIM), np.float32)

        # cur_state
        cur_state = self._row(gq.state(self.state_group), qw0).astype(np.float32)
        if self.normalize_state:
            cur_state = self._norm(cur_state[None], self.state_stats, q_emb)[0]

        # video: [state 1f | ret_act 4f | demo 32f | cur 4f | act 4f | future 32f] = 77f
        q_clip = self._grid_clip(gq, qw0)
        m_clip = self._grid_clip(gm, mw0) if has_ret else np.zeros_like(q_clip)
        S = self.final_image_size
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)
        images = np.concatenate([
            blank1, blank4, m_clip,
            np.repeat(q_clip[:1], NUM_DUPLICATES, axis=0),
            blank4, q_clip,
        ], axis=0)
        assert images.shape[0] == self.chunk_duration
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=True, stronger_image_aug=True)

        T = self.state_t
        roles = np.empty(T, np.int64); tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        roles[1], tpos[1] = ROLE_RET_ACTION, NOW_S
        roles[2:2 + K] = ROLE_RET_VIDEO
        tpos[2:2 + K] = NOW_S - DT * np.arange(K, 0, -1)
        roles[2 + K], tpos[2 + K] = ROLE_CUR_IMAGE, NOW_S
        roles[3 + K], tpos[3 + K] = ROLE_ACTION, NOW_S
        roles[4 + K:], tpos[4 + K:] = ROLE_FUTURE, NOW_S + DT * np.arange(1, K + 1)

        cls = self.wcls[q]
        t5 = (torch.squeeze(self.t5_text_embeddings[cls]) if cls in self.t5_text_embeddings
              else torch.zeros(512, 1024))

        return {
            "video": images,
            "command": str(cls),
            "actions": q_act.astype(np.float32),
            "action_mask": np.ones(ACTION_DIM, np.float32),
            "retrieved_actions": m_act.astype(np.float32),
            "retrieved_action_mask": np.ones(ACTION_DIM, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": idx,
            "embodiment": q_emb, "retrieved_embodiment": m_emb,
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": K + 3,                     # 11
            "retrieved_video_start_latent_idx": 2,
            "retrieved_video_end_latent_idx": 2 + K,
            "retrieved_action_latent_idx": 1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 2 + K,
            "future_video_start_latent_idx": 4 + K,
            "future_video_end_latent_idx": T,
            "action_latent_idx": 3 + K,                          # 11
            "proprio": cur_state.astype(np.float32),
            "future_proprio": np.zeros(STATE_DIM, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": 4 + K, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": has_ret, "has_ret_image": has_ret, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_act.copy().astype(np.float32),
        }
