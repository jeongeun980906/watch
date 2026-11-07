"""EgoVerse mid-training PLAIN — non-retrieval policy + video prediction.

Mid-training arm of the "plain mid -> retrieval post-train" ablation. Identical
to the v3 mid (``egoverse_ret_v3_dataset`` / ``_v3_state_dataset``) except the
RETRIEVAL channel is removed: no ret_action slot, no ret_video block. What
remains is plain policy learning on the current observation, jointly with video
prediction from the current frame:

    conditioning : cur_state (proprio) + cur_image
    generated    : action chunk + future video

    slot:  cur_state | cur_image | action | future (g)
    role:      0     |     3     |   4    |   5
    time:     NOW    |    NOW    |  NOW   | NOW+dt .. NOW+g*dt
    cond:     yes    |    yes    |  GEN   |  GEN
    T = g + 3 ;  num_conditional_frames = 2

WHY THE FUTURE-VIDEO BLOCK STAYS (do not "simplify" it away). Two independent
reasons, both about what post-training inherits:

  1. Role 4 must stay DISCRIMINATIVE. ``egoverse_v2_pe_slot_rationale.md`` §2.1
     argues the role embedding is forced to be learned because several slots
     share t=NOW. That argument covers the CONDITIONING slots, but the action
     slot is identified by role only if there is more than one GENERATED block.
     Drop the future video and action becomes "the single noisy slot, and also
     the last one" — the net can solve mid-training without ever reading role 4,
     so role 4 degenerates to "the thing I generate". Post-training then has TWO
     generated blocks (action=role 4, future=role 5) and has to learn the
     distinction from scratch, which is exactly the transfer the ablation is
     supposed to be measuring. Keeping future video forces role 4 vs role 5 to
     mean different things during mid-training.
  2. The FUTURE time axis (t > NOW) is only exercised by this block. Without it
     every token sits at t=NOW and post-training's spread-out timestamps are
     pure extrapolation.

REMAINING TRANSFER GAP (accepted, and the point of the ablation). Role 1
(RET_ACTION) and role 2 (RET_VIDEO) are still never trained here, and the PAST
time axis (t < NOW) is still never exercised, so post-training meets both for
the first time with zero-init ``role_embedder`` rows. Gate the run with the
post-train first-iter ``action_l1`` probe from that doc (v1's broken layout
scored 0.49, the v2 contract 0.19-0.21).

``state_target`` selects the target space, matching the v3 variants:
  * False -> DELTA action57 chunk (``_normalize`` + action57_stats.json);
    ``proprio`` = current absolute state57 (57,).
  * True  -> ABSOLUTE state57 FUTURE chunk (``_norm_state`` + robust stats);
    ``proprio`` = PAST state57 chunk (flattened).

Pairs are still iterated query-by-query from the pair manifest so the query
distribution is IDENTICAL to the retrieval mids (controlled comparison); the
matched demo index is simply ignored.
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
    ROLE_FUTURE,
    STATE_DIM,
)
from cosmos_policy.datasets.egoverse_ret_v3_state_dataset import EgoVerseRetV3StateDataset


class EgoVersePlainDataset(EgoVerseRetV3StateDataset):
    """Plain (non-retrieval) policy + video-prediction dataset — see module doc.

    Subclasses the V3 STATE dataset so both the delta (``_normalize`` /
    ``_action_native``) and absolute-state (``_norm_state`` / ``_state_rows``)
    machinery are available, and so video packing is v3 FULL-FRAME (no 2x2 grid)
    — matching ``cosmos_predict2p5_2b_egoverse_ret_v3_448_v11``, the transfer mid
    this is meant to be compared against.
    """

    def __init__(self, *args, state_target: bool = False, **kw):
        # proprio is normalized with the robust state stats even in the DELTA variant.
        if not state_target and "robust_state_stats" not in kw:
            kw["robust_state_stats"] = True
        super().__init__(*args, **kw)
        self.state_target = bool(state_target)

    def __getitem__(self, idx: int) -> dict:
        r = int(idx) % self.N
        q = int(self.query_idx[r])
        return self.build_sample(q, q, idx=idx, aug=True)

    def build_sample(self, q: int, m: int = -1, idx: int = 0, aug: bool = True) -> dict:
        """``m`` is accepted for signature-compat with the ret datasets and IGNORED."""
        del m
        nf_q, dt_q = self._native(q)
        g = nf_q // NUM_DUPLICATES              # future-video latents
        T = g + 3

        fr_q = self._frames_native(q, nf_q)
        q_video = self._grid_clip(q, fr_q)      # (nf_q, S, S, 3) full-frame primary cam

        # ---- target + proprio -------------------------------------------------
        if self.state_target:
            span_q = int(self.ww1[q] - self.ww0[q])
            q_tgt = self._norm_state(self._state_rows(q, fr_q + 1), self.wemb[q])
            proprio = self._norm_state(
                self._state_rows(q, fr_q - span_q + 1), self.wemb[q]
            ).reshape(-1)                                            # past state chunk
        else:
            q_a, _ = self._action_native(q, fr_q)
            q_tgt = self._normalize(q_a, self.wemb[q])               # delta chunk
            try:
                st = np.asarray(self._grp(q).state("state57")[int(self.ww0[q])], np.float32)
                proprio = self._norm_state(st, self.wemb[q])         # current state (57,)
            except Exception:
                proprio = np.zeros(STATE_DIM, np.float32)

        # ---- pixel packing: [cur_state 1f | cur_image 4f | action 4f | future 4g]
        S = self.final_image_size
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)
        images = np.concatenate([
            blank1,                                                  # cur_state
            np.repeat(q_video[:1], NUM_DUPLICATES, axis=0),          # cur_image x4 -> 1 latent
            blank4,                                                  # action
            q_video,                                                 # future video (g latents)
        ], axis=0)
        assert images.shape[0] == 1 + 4 * (T - 1), (images.shape, T)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=aug, stronger_image_aug=aug)

        # ---- roles + timestamps ----
        roles = np.empty(T, np.int64)
        tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        roles[1], tpos[1] = ROLE_CUR_IMAGE, NOW_S
        roles[2], tpos[2] = ROLE_ACTION, NOW_S
        roles[3:], tpos[3:] = ROLE_FUTURE, NOW_S + dt_q * np.arange(1, g + 1)

        cls = self.wcls[q]
        cmd = f"{CLASS_TO_COMMAND.get(cls, CLASS_TO_COMMAND['other'])} [{self.wemb[q]}]"
        t5 = (torch.squeeze(self.t5_text_embeddings[cmd]) if cmd in self.t5_text_embeddings
              else torch.zeros(512, 1024))

        return {
            "video": images,
            "actions": q_tgt,                                # (nf_q,57) GEN target
            "action_mask": np.ones(ACTION_DIM, np.float32),
            # NO retrieved_actions / retrieved_action_mask — the injection helpers in
            # policy_video2world_model_pusht_ret.py return early when the key is absent.
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": idx,
            "embodiment": self.wemb[q], "retrieved_embodiment": self.wemb[q],
            # v2 core
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": 2,                     # cur_state + cur_image
            # latent indices — retrieval fully disabled, future video ACTIVE
            "retrieved_video_start_latent_idx": -1,
            "retrieved_video_end_latent_idx": -1,
            "retrieved_action_latent_idx": -1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 1,
            "future_video_start_latent_idx": 3,
            "future_video_end_latent_idx": T,
            "action_latent_idx": 2,
            "proprio": proprio,
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
            "next_action_chunk": q_tgt.copy(),
        }

    def bucket_key(self, r: int) -> tuple[bool]:
        """Only the QUERY length varies (no demo tensors) -> 2 buckets, not 4."""
        return (self._is_md(self.wemb[int(self.query_idx[r])]),)
