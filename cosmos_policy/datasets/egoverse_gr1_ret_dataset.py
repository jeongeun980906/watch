"""EgoVerse mimicdroid-GR1 multimodal retrieval dataset for Cosmos Predict-2.5.

Ports the Cosmos-3 GR1 posttrain DATA (`egoverse_pairs_gr1`: 3-cam front/sideleft/
sideright, action57 + state57, demo↔query future-pairs) into the 2.5 multimodal
image+state+action prediction framework (structure mirrors robocasa_doors_ret).

Instead of generating a future VIDEO, the model here (like realrobot/robocasa ret)
CONDITIONS on the retrieved demo (video+state+action) + current observation
(3 images + state), and GENERATES: the action chunk + the chunk-END observation
(3 future images). future_state and value prediction are DISABLED (image + action only).

Latent layout (state_t = 35, chunk_duration = 1 + 34*4 = 137):
    0                     blank sentinel
    1 .. 8                ret_video front      (32f demo clip -> 8 latents)   cond
    9 .. 16               ret_video sideleft   (32f)                          cond
    17 .. 24              ret_video sideright  (32f)                          cond
    25                    ret_state            (blank, state injected)        cond
    26                    ret_action           (blank, action injected)       cond
    27                    cur_image front      (1 obs frame)                  cond
    28                    cur_image sideleft                                  cond
    29                    cur_image sideright                                 cond
    30                    cur_state            (blank, state injected)        cond
    -- num_conditional_frames = 31 --
    31                    pred_action          (blank, generated)             GEN
    32                    future_image front   (chunk-END obs frame)          GEN
    33                    future_image sideleft                               GEN
    34                    future_image sideright                              GEN
    (future_state latent REMOVED — no state/value prediction)

front->current_image / sideleft->current_image2 / sideright->current_wrist_image
(same for future_*). The 3 ret-video spans are laid CONTIGUOUSLY and covered by the
single `retrieved_video_start/end_latent_idx` span (the model's `_apply_ret_mask`
masks that whole block); per-cam spans are also emitted for eval/clarity.

PER-FRAME t+k (docs/gr1_inference_retrieval_spec.md §6b): pairs have STRIDE=4
boundaries; to learn EVERY timestep, one delta in {0..min(per_frame_stride-1,
room_q, room_c)} is drawn and applied to BOTH query and demo reads in lockstep.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_zread import CENTERTOP, EgoZarrEpisode, FRONT, SIDELEFT, SIDERIGHT
from cosmos_policy.utils.utils import duplicate_array

# Episode root for the GR1/MimicDroid post-train zarrs. EGOVERSE_GR1_HUM_ROOT overrides it outside the
# cluster layout (e.g. the unpacked HF dataset Jeongeun/mimicdroid-gr1).
HUM_ROOT = os.environ.get("EGOVERSE_GR1_HUM_ROOT", "/mnt/ddn/dataset/egoverse")
DEFAULT_MANIFEST = "/mnt/ddn/dataset/egoverse_pairs_gr1"
DEFAULT_T5_EMB = f"{DEFAULT_MANIFEST}/t5_embeddings.pkl"
DEFAULT_ACTION_STATS = f"{DEFAULT_MANIFEST}/action57_stats.json"
DEFAULT_STATE_STATS = f"{DEFAULT_MANIFEST}/state57_stats.json"

NUM_DUPLICATES = 4
ACTION_DIM = 57
STATE_DIM = 57
CAMS = (FRONT, SIDELEFT, SIDERIGHT)


def _decode(jpeg: bytes, size: int) -> np.ndarray:
    """JPEG bytes -> (size,size,3) uint8; zeros on empty/broken frame."""
    if not jpeg:
        return np.zeros((size, size, 3), np.uint8)
    try:
        img = Image.open(io.BytesIO(jpeg)).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.BILINEAR)
        return np.asarray(img)
    except Exception:
        return np.zeros((size, size, 3), np.uint8)


class EgoVerseGr1RetDataset(Dataset):
    def __init__(
        self,
        manifest_dir: str = DEFAULT_MANIFEST,
        t5_text_embeddings_path: str = DEFAULT_T5_EMB,
        action_stats_path: str = DEFAULT_ACTION_STATS,
        state_stats_path: str = DEFAULT_STATE_STATS,
        chunk_size: int = 32,
        per_frame_stride: int = 4,           # GR1 pairs stride 4 -> train every timestep
        final_image_size: int = 224,
        num_duplicates_per_image: int = NUM_DUPLICATES,
        normalize_actions: bool = True,
        normalize_state: bool = True,
        use_residual_actions: bool = False,  # GR1 action57 is already deltas -> predict absolute
        retrieval_dropout_prob: float = 0.0,
        state_group: str = "state57",
        action_group: str = "action57",
        epoch_length: Optional[int] = None,
        debug: bool = False,
        **_ignored,
    ):
        super().__init__()
        assert num_duplicates_per_image == NUM_DUPLICATES
        self.manifest_dir = manifest_dir
        self.chunk_size = int(chunk_size)
        self.per_frame_stride = max(1, int(per_frame_stride))
        self.final_image_size = final_image_size
        self.num_duplicates_per_image = num_duplicates_per_image
        self.normalize_actions = normalize_actions
        self.normalize_state = normalize_state
        self.use_residual_actions = use_residual_actions
        self.retrieval_dropout_prob = retrieval_dropout_prob
        self.state_group = state_group
        self.action_group = action_group
        self.debug = debug

        W = np.load(os.path.join(manifest_dir, "windows.npz"), allow_pickle=True)
        self.wep = W["ep"].astype(str)
        self.ww0 = W["w0"].astype(np.int64)
        self.wcls = W["cls"].astype(str)
        self.wemb = W["emb"].astype(str)
        P = np.load(os.path.join(manifest_dir, "pairs.npz"), allow_pickle=True)
        self.query_idx = P["query_idx"].astype(np.int64)
        self.match_idx = P["match_idx"].astype(np.int64)     # (N, K) padded -1
        self.N = len(self.query_idx)
        self.epoch_length = epoch_length or self.N

        # t5 embeddings optional: missing file / missing key -> zeros fallback
        # (real t5 regen for the GR1 command vocab can be done later).
        if os.path.exists(t5_text_embeddings_path):
            with open(t5_text_embeddings_path, "rb") as fp:
                self.t5_text_embeddings = pickle.load(fp)
        else:
            print(f"[EgoVerseGr1RetDataset] t5 file absent ({t5_text_embeddings_path}); zeros fallback")
            self.t5_text_embeddings = {}
        self._zcache: dict[str, EgoZarrEpisode] = {}
        self.act_stats = self._load_or_compute_stats(action_stats_path, self.action_group, "action")
        self.state_stats = self._load_or_compute_stats(state_stats_path, self.state_group, "state")
        print(f"[EgoVerseGr1RetDataset] pairs={self.N} chunk={self.chunk_size} "
              f"per_frame_stride={self.per_frame_stride} state_t={self.state_t} "
              f"chunk_duration={self.chunk_duration}", flush=True)

    # ---- static layout sizes ----
    @property
    def state_t(self) -> int:
        # 1 blank + 3*(chunk/4) ret video + 2 ret(state,action) + 3 cur img + 1 cur state
        #        + 1 pred action + 3 future img   (GENERATE = action + 3-cam future image ONLY;
        #        future_state prediction was removed, so NO trailing future-state latent)
        return 1 + 3 * (self.chunk_size // NUM_DUPLICATES) + 2 + 3 + 1 + 1 + 3

    @property
    def chunk_duration(self) -> int:
        return 1 + (self.state_t - 1) * NUM_DUPLICATES

    # ---- zarr / normalization ----
    def __getstate__(self):
        s = self.__dict__.copy(); s["_zcache"] = {}; return s

    def __setstate__(self, s):
        self.__dict__.update(s); self._zcache = {}

    def _grp(self, wi: int) -> EgoZarrEpisode:
        ep = self.wep[wi]
        g = self._zcache.get(ep)
        if g is None:
            g = EgoZarrEpisode(os.path.join(HUM_ROOT, ep))
            self._zcache[ep] = g
        return g

    def _load_or_compute_stats(self, path: str, group: str, kind: str) -> dict:
        """PER-EMBODIMENT min-max stats {emb: {lo,hi}, "_global": {...}} (OLD flat file -> _global)."""
        if os.path.exists(path):
            d = json.load(open(path))
            if "lo" in d and "hi" in d:
                return {"_global": {"lo": np.asarray(d["lo"], np.float32), "hi": np.asarray(d["hi"], np.float32)}}
            return {k: {"lo": np.asarray(v["lo"], np.float32), "hi": np.asarray(v["hi"], np.float32)}
                    for k, v in d.items()}
        print(f"[EgoVerseGr1RetDataset] computing PER-EMB {group} {kind} stats ...", flush=True)
        rng = np.random.default_rng(0)
        sel = rng.choice(self.N, min(4000, self.N), replace=False)
        by_emb: dict = {}
        for r in sel:
            wi = int(self.query_idx[r]); g = self._grp(wi); w0 = int(self.ww0[wi])
            arr = g.action(group) if kind == "action" else g.state(group)
            by_emb.setdefault(str(self.wemb[wi]), []).append(arr[max(0, w0): w0 + self.chunk_size])

        def _mmax(A):
            lo = np.percentile(A, 1, axis=0); hi = np.percentile(A, 99, axis=0)
            return lo, np.where(hi - lo < 1e-4, lo + 1.0, hi)
        stats = {}
        for emb, chunks in by_emb.items():
            lo, hi = _mmax(np.concatenate(chunks, 0))
            stats[emb] = {"lo": lo.astype(np.float32), "hi": hi.astype(np.float32)}
        glo, ghi = _mmax(np.concatenate([c for cs in by_emb.values() for c in cs], 0))
        stats["_global"] = {"lo": glo.astype(np.float32), "hi": ghi.astype(np.float32)}
        json.dump({k: {"lo": v["lo"].tolist(), "hi": v["hi"].tolist()} for k, v in stats.items()},
                  open(path, "w"))
        self._zcache = {}
        return stats

    def _norm(self, a: np.ndarray, stats: dict, emb: str) -> np.ndarray:
        s = stats.get(str(emb), stats["_global"])                    # PER-EMBODIMENT
        lo, hi = s["lo"], s["hi"]
        return np.clip(2.0 * (a - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0).astype(np.float32)

    # ---- row/frame access with clamping ----
    @staticmethod
    def _row(arr: np.ndarray, i: int) -> np.ndarray:
        return arr[min(max(i, 0), len(arr) - 1)]

    def _chunk(self, arr: np.ndarray, start: int) -> np.ndarray:
        n = len(arr); out = np.stack([arr[min(max(start + k, 0), n - 1)] for k in range(self.chunk_size)])
        return out

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, idx: int) -> dict:
        if self.debug:
            idx = 0
        r = idx % self.N
        q = int(self.query_idx[r])
        ms = self.match_idx[r]; ms = ms[ms >= 0]
        ret_dropped = self.retrieval_dropout_prob > 0 and random.random() < self.retrieval_dropout_prob
        has_ret = 0 if (len(ms) == 0 or ret_dropped) else 1
        m = int(ms[random.randrange(len(ms))]) if has_ret else q

        gq = self._grp(q); gm = self._grp(m)
        qw0 = int(self.ww0[q]); mw0 = int(self.ww0[m])
        C = self.chunk_size; S = self.final_image_size

        # v6: NO stochastic per-frame jitter. The v6 pairs enumerate EVERY
        # timestep as an explicit row, so one epoch covers every episode at
        # every t exactly once (robot-dataset standard; user rule 2026-07-14).
        # The old delta jitter existed only to fill sparse stride-4 rows.

        q_state = gq.state(self.state_group); q_act = gq.action(self.action_group)
        m_state = gm.state(self.state_group); m_act = gm.action(self.action_group)

        # ---- build image_list in fixed latent order ----
        image_list: list[np.ndarray] = []
        cur_latent = 0
        zero = np.zeros((S, S, 3), np.uint8)

        # latent 0: blank sentinel (1 frame)
        image_list.append(np.expand_dims(zero, 0)); cur_latent += 1

        def _push_image(arr: np.ndarray) -> int:
            nonlocal cur_latent
            image_list.append(duplicate_array(arr, total_num_copies=self.num_duplicates_per_image))
            i = cur_latent; cur_latent += 1; return i

        def _push_blank() -> int:
            return _push_image(zero)

        def _push_video(frames: np.ndarray) -> tuple[int, int]:
            """frames: (C,S,S,3) real demo clip -> C/4 latents. Returns (start, end)."""
            nonlocal cur_latent
            image_list.append(frames)
            start = cur_latent; cur_latent += frames.shape[0] // self.num_duplicates_per_image
            return start, cur_latent

        def _read_frame(g, i, cam):
            return _decode(g.frame(min(max(i, 0), g.n_frames - 1), cam), S)

        def _read_clip(g, w0, cam):
            return np.stack([_read_frame(g, w0 + k, cam) for k in range(C)])

        # ret videos (3 cams, 32f). Zeros when retrieval dropped.
        ret_spans = {}
        for name, cam in (("front", FRONT), ("sideleft", SIDELEFT), ("sideright", SIDERIGHT)):
            clip = _read_clip(gm, mw0, cam) if has_ret else np.broadcast_to(zero, (C, S, S, 3)).copy()
            ret_spans[name] = _push_video(clip)
        retrieved_video_start_latent_idx = ret_spans["front"][0]
        retrieved_video_end_latent_idx = ret_spans["sideright"][1]     # covers all 3 (contiguous)

        # ret state / ret action (injected)
        retrieved_state_latent_idx = _push_blank()
        ret_action_latent_idx = _push_blank()

        # current images (3 cams, single obs frame)
        current_image_latent_idx = _push_image(_read_frame(gq, qw0, FRONT))
        current_image2_latent_idx = _push_image(_read_frame(gq, qw0, SIDELEFT))
        current_wrist_image_latent_idx = _push_image(_read_frame(gq, qw0, SIDERIGHT))
        # current state (injected)
        current_proprio_latent_idx = _push_blank()

        # predicted action (generated)
        action_latent_idx = _push_blank()
        # future images (3 cams, chunk-END obs frame)
        fut = qw0 + C
        future_image_latent_idx = _push_image(_read_frame(gq, fut, FRONT))
        future_image2_latent_idx = _push_image(_read_frame(gq, fut, SIDELEFT))
        future_wrist_image_latent_idx = _push_image(_read_frame(gq, fut, SIDERIGHT))
        # future state prediction REMOVED (predict image + action only) -> no future-state latent
        future_proprio_latent_idx = -1

        images = np.concatenate(image_list, axis=0)
        images = preprocess_image(
            images, final_image_size=S, normalize_images=False,
            use_image_aug=True, stronger_image_aug=True,
        )

        # ---- action / state values ----
        act_chunk = self._chunk(q_act, qw0)
        ret_act_chunk = self._chunk(m_act, mw0) if has_ret else np.zeros((C, ACTION_DIM), np.float32)
        q_emb, m_emb = self.wemb[q], self.wemb[m]                     # per-embodiment normalization
        if self.normalize_actions:
            act_chunk = self._norm(act_chunk, self.act_stats, q_emb)
            if has_ret:
                ret_act_chunk = self._norm(ret_act_chunk, self.act_stats, m_emb)
        target_actions = (act_chunk - ret_act_chunk) if self.use_residual_actions else act_chunk

        cur_state = self._row(q_state, qw0).astype(np.float32)
        fut_state = np.zeros(STATE_DIM, np.float32)   # future-state prediction removed (unused)
        ret_state = self._row(m_state, mw0).astype(np.float32) if has_ret else np.zeros(STATE_DIM, np.float32)
        if self.normalize_state:
            cur_state = self._norm(cur_state[None], self.state_stats, q_emb)[0]
            if has_ret:
                ret_state = self._norm(ret_state[None], self.state_stats, m_emb)[0]

        cls = self.wcls[q]
        command = f"{cls}"
        t5 = (torch.squeeze(self.t5_text_embeddings[command])
              if command in self.t5_text_embeddings else torch.zeros(512, 1024))

        sample = {
            "video": images,
            "command": command,
            "actions": target_actions.astype(np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "proprio": cur_state.astype(np.float32),
            "future_proprio": fut_state.astype(np.float32),
            "__key__": idx,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            # 0 = POLICY sample (action is GENERATED/predicted). MUST be 0: the model at
            # get_data_and_condition sets the action latent's condition mask = world_model_sample_mask
            # (policy_video2world_model_rectified_flow.py:934). With 1 the action latent becomes a
            # CONDITIONING INPUT (ground-truth action fed in) so it is never predicted → action loss=0.
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "value_function_return": float("-100"),
            # generated / current latent indices
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": -1,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "current_image2_latent_idx": current_image2_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_wrist_image2_latent_idx": -1,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
            "future_image2_latent_idx": future_image2_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_wrist_image2_latent_idx": -1,
            # retrieval
            "retrieved_video_start_latent_idx": retrieved_video_start_latent_idx,
            "retrieved_video_end_latent_idx": retrieved_video_end_latent_idx,
            "ret_video_front_start_latent_idx": ret_spans["front"][0],
            "ret_video_front_end_latent_idx": ret_spans["front"][1],
            "ret_video_sideleft_start_latent_idx": ret_spans["sideleft"][0],
            "ret_video_sideleft_end_latent_idx": ret_spans["sideleft"][1],
            "ret_video_sideright_start_latent_idx": ret_spans["sideright"][0],
            "ret_video_sideright_end_latent_idx": ret_spans["sideright"][1],
            "retrieved_action_latent_idx": ret_action_latent_idx,
            "retrieved_state_latent_idx": retrieved_state_latent_idx,
            "retrieved_actions": ret_act_chunk.astype(np.float32),
            "retrieved_proprio": ret_state.astype(np.float32),
            "ret_proprio": ret_state.astype(np.float32),
            "has_ret_data": has_ret,
            "has_ret_image": has_ret,
            "ret_mask": has_ret,
        }
        return sample
