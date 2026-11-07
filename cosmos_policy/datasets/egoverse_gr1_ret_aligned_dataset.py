"""GR1 post-train dataset ALIGNED to the egoverse_ret mid-training latent layout.

Motivation: warm-starting GR1 post-train from the egoverse_ret (3 s) checkpoint is
only meaningful if the latent layout matches — mid-training put the action at slot
2R+3=49 inside a state_t=50, R=23 layout, while the first GR1 layout used state_t=35
with the action at 31, so nothing the DiT learned about WHERE to denoise actions
transferred. This dataset reproduces the mid layout EXACTLY (same slots, same
chunk_duration, same sample-dict keys) and maps the GR1 data into it:

  * 3 cams are ALL kept, packed as a 2x2 grid into ONE 224x224 frame
    (front top-left, sideleft top-right, sideright bottom-left, black bottom-right)
    so the video stays single-stream like mid-training. (112x112 per cam.)
  * temporal UPSAMPLE 32f@20Hz -> 92f (idx = floor(arange(92)*32/92)): the ret demo
    clip, the query future clip, the action chunk and the ret action chunk all
    become 92-length, so ret_video spans latents 1..23 and the generated future
    spans 26..48 exactly like the 3 s mid window. At inference, downsample the
    generated (92,57) chunk back to 32 via idx = floor((arange(32)+0.5)*92/32).

Layout (identical to egoverse_ret num_frames=92, R=23, state_t=50):
    0        blank sentinel                                   cond
    1  .. 23 ret_video   (grid-3cam demo, 92f upsampled)      cond
    24       ret_action  (blank; 92x57 injected)              cond
    25       cur_image   (grid-3cam first query frame)        cond
    -- num_conditional_frames = 26 --
    26 .. 48 future video (grid-3cam query clip, 92f)         GEN
    49       action      (blank; model generates 92x57)       GEN

State: mid has NO state slots, but GR1 control needs the current state, so cur_state
(normalized state57) is TILED into slot 0 — the blank sentinel mid always saw as zeros —
via the model's proprio injection (current_proprio_latent_idx=0). Slot indices are
untouched; smoke-gated (init action_l1 must stay < 0.8). ret_state stays dropped.
world_model_sample_mask=0 (policy: the action IS generated — see
docs/egoverse_cosmos2p5_pipeline.md pitfalls).
"""
from __future__ import annotations

import numpy as np
import torch

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_gr1_ret_dataset import (
    ACTION_DIM,
    EgoVerseGr1RetDataset,
    FRONT,
    SIDELEFT,
    SIDERIGHT,
    NUM_DUPLICATES,
    _decode,
)

NUM_FRAMES = 92                      # mid-training temporal length (R=23 latents)
R = NUM_FRAMES // NUM_DUPLICATES     # 23
NATIVE = 32                          # native GR1 steps @20Hz
# layout="upsample": 32 -> 92 by frame repetition (idx order preserved)
UPSAMPLE_IDX = (np.arange(NUM_FRAMES) * NATIVE) // NUM_FRAMES      # (92,) in [0,31]
# inverse map for inference-time extraction of the native 32-step chunk
DOWNSAMPLE_IDX = ((np.arange(NATIVE) + 0.5) * NUM_FRAMES / NATIVE).astype(np.int64)  # (32,) in [0,91]
# layout="pad" (preferred): first 32 entries are the native clip/chunk, rest ZERO-PADDED.
PAD_EXTRACT_IDX = np.arange(NATIVE)                                # (32,) first rows


class EgoVerseGr1RetAlignedDataset(EgoVerseGr1RetDataset):
    """GR1 pairs rendered in the egoverse_ret mid-training layout (see module doc).

    layout="pad" (default): native 32f clips/chunks ZERO-PADDED to the 92-slot span
      (video: real frames then black; actions: real 32 rows then zeros; extraction =
      first 32 rows). No temporal resampling.
    layout="upsample": 32f repeated to 92f (UPSAMPLE_IDX / DOWNSAMPLE_IDX).
    """

    def __init__(self, *args, layout: str = "pad", **kw):
        super().__init__(*args, **kw)
        assert layout in ("pad", "upsample")
        self.layout = layout

    # ---- static layout: EXACTLY the mid-training values ----
    @property
    def state_t(self) -> int:               # 2R+4 = 50
        return 2 * R + 4

    @property
    def chunk_duration(self) -> int:        # 8R+13 = 197
        return 1 + (self.state_t - 1) * NUM_DUPLICATES

    # ---- grid-3cam packing ----
    def _grid_frame(self, g, i: int) -> np.ndarray:
        """One 2x2-grid frame: 3 cams at S/2 each, bottom-right black."""
        S = self.final_image_size
        h = S // 2
        i = min(max(i, 0), g.n_frames - 1)
        canvas = np.zeros((S, S, 3), np.uint8)
        canvas[:h, :h] = _decode(g.frame(i, FRONT), h)
        canvas[:h, h:] = _decode(g.frame(i, SIDELEFT), h)
        canvas[h:, :h] = _decode(g.frame(i, SIDERIGHT), h)
        return canvas

    def _grid_clip_92(self, g, w0: int) -> np.ndarray:
        """(92,S,S,3) grid clip. pad: native 32f then black; upsample: 32f repeated to 92."""
        native = {int(k): self._grid_frame(g, w0 + int(k)) for k in range(NATIVE)}
        if self.layout == "pad":
            S = self.final_image_size
            clip = np.zeros((NUM_FRAMES, S, S, 3), np.uint8)
            for k in range(NATIVE):
                clip[k] = native[k]
            return clip
        return np.stack([native[int(k)] for k in UPSAMPLE_IDX])

    def _span92(self, a32: np.ndarray) -> np.ndarray:
        """(32,D) chunk -> (92,D): zero-pad (pad mode) or repeat (upsample mode)."""
        if self.layout == "pad":
            out = np.zeros((NUM_FRAMES, a32.shape[1]), np.float32)
            out[:NATIVE] = a32
            return out
        return a32[UPSAMPLE_IDX].astype(np.float32)

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
        C = self.chunk_size                                    # 32 native steps

        # per-frame t+k lockstep delta (same rule as the base class)
        def _room(g, w0):
            top = min(g.action_rows(self.action_group), g.n_frames - 1)
            return top - (w0 + C)
        dmax = min(self.per_frame_stride - 1, _room(gq, qw0), _room(gm, mw0))
        delta = np.random.randint(0, dmax + 1) if dmax > 0 else 0
        qw0 += delta; mw0 += delta

        # ---- actions: native 32 -> 92 span (pad: zero-tail / upsample: repeat) ----
        q_act32 = self._chunk(gq.action(self.action_group), qw0)   # (32,57)
        m_act32 = self._chunk(gm.action(self.action_group), mw0)
        q_emb, m_emb = self.wemb[q], self.wemb[m]
        if self.normalize_actions:
            q_act32 = self._norm(q_act32, self.act_stats, q_emb)
            m_act32 = self._norm(m_act32, self.act_stats, m_emb)
        q_act = self._span92(q_act32)                              # (92,57)
        m_act = (self._span92(m_act32) if has_ret
                 else np.zeros((NUM_FRAMES, ACTION_DIM), np.float32))

        # ---- current state: normalized state57, tiled into slot 0 (blank sentinel) ----
        cur_state = self._row(gq.state(self.state_group), qw0).astype(np.float32)
        if self.normalize_state:
            cur_state = self._norm(cur_state[None], self.state_stats, q_emb)[0]

        # ---- video: grid-3cam clips, 92f ----
        q_video = self._grid_clip_92(gq, qw0)                      # (92,S,S,3) query/future
        m_video = (self._grid_clip_92(gm, mw0) if has_ret
                   else np.zeros_like(q_video))                    # ret demo

        # ---- frame packing: EXACT mid order (blank | ret92 | blank4 | cur4 | q92 | blank4) ----
        blank1 = np.zeros_like(q_video[:1])
        blank_dup = np.repeat(blank1, NUM_DUPLICATES, axis=0)
        cur = 0; image_list = []
        image_list.append(blank1); cur += 1                                   # 0 blank
        image_list.append(m_video); ret_vid_start = cur; cur += R             # 1..23 ret video
        ret_vid_end = cur
        image_list.append(blank_dup.copy()); ret_act_idx = cur; cur += 1      # 24 ret action
        image_list.append(np.repeat(q_video[:1], NUM_DUPLICATES, axis=0))     # 25 cur image
        q_img_idx = cur; cur += 1
        image_list.append(q_video.copy()); q_vid_start = cur; cur += R        # 26..48 future (GEN)
        q_vid_end = cur
        image_list.append(blank_dup.copy()); q_act_idx = cur; cur += 1        # 49 action (GEN)

        images = np.concatenate(image_list, axis=0)                # (197,S,S,3)
        images = preprocess_image(
            images, final_image_size=self.final_image_size, normalize_images=False,
            use_image_aug=True, stronger_image_aug=True,
        )

        cls = self.wcls[q]
        command = f"{cls}"
        t5 = (torch.squeeze(self.t5_text_embeddings[command])
              if command in self.t5_text_embeddings else torch.zeros(512, 1024))

        S = self.final_image_size
        return {
            "video": images,
            "command": command,
            "actions": q_act,                                     # (92,57) GEN target
            "action_mask": np.ones(ACTION_DIM, np.float32),
            "retrieved_actions": m_act,                           # (92,57) cond
            "retrieved_action_mask": np.ones(ACTION_DIM, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": idx,
            "embodiment": q_emb,
            "retrieved_embodiment": m_emb,
            # latent indices — IDENTICAL to mid-training
            "retrieved_video_start_latent_idx": ret_vid_start,    # 1
            "retrieved_video_end_latent_idx": ret_vid_end,        # 24 (exclusive)
            "retrieved_action_latent_idx": ret_act_idx,           # 24
            "retrieved_state_latent_idx": -1,                     # no state slots (like mid)
            "current_image_latent_idx": q_img_idx,                # 25
            "future_video_start_latent_idx": q_vid_start,         # 26
            "future_video_end_latent_idx": q_vid_end,             # 49 (exclusive)
            "action_latent_idx": q_act_idx,                       # 49
            # cur_state tiled into slot 0 (the blank sentinel — mid saw zeros there)
            "proprio": cur_state.astype(np.float32),
            "future_proprio": np.zeros(57, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": q_vid_start, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": has_ret, "has_ret_image": has_ret, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_act.copy(),
        }
