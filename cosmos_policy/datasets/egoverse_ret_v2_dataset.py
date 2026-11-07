"""EgoVerse mid-training V2 — length-GENERAL compact layout (docs/egoverse_mid_v2_design.md).

Per-token ROLE ids + CONTINUOUS-TIME rope positions replace fixed slot indices:
"what a slot is" = role embedding, "when it is" = timestamp (seconds, now=NOW_S).
Sequences are COMPACT (native lengths, no padding, no resampling): human window
92f@30Hz -> 23 latents, MD 60f@20Hz -> 15. Cross-embodiment pairs give 4 sequence
shapes; BucketedDistributedSampler serves same-shape batches.

Latent layout (k = demo latents, g = query latents; T = k+g+4):
    0        cur_state   (state57 tiled; role CUR_STATE)          cond   t=NOW
    1        ret_action  (demo action chunk tiled; role RET_ACT)  cond   t=NOW
    2..1+k   ret_video   (native demo frames, grid-3cam)          cond   t=NOW-k*dt..NOW-dt
    2+k      cur_image   (first query frame, grid-3cam)           cond   t=NOW
    -- num_conditional_frames = k+3 --
    3+k      action      (query action chunk; GENERATED)          GEN    t=NOW
    4+k..T-1 future      (native query frames; GENERATED)         GEN    t=NOW+dt..NOW+g*dt

GR1 post-train reuses this class with k=g=8 (32f@20Hz) — same contract, no changes.
world_model_sample_mask=0 (policy — see docs/egoverse_cosmos2p5_pipeline.md pitfalls).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Sampler

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_ret_dataset import (
    ACTION_DIM,
    CLASS_TO_COMMAND,
    NUM_DUPLICATES,
    EgoVerseRetDataset,
    _decode,
)

NOW_S = 4.0          # "now" on the time axis (>= max window span 3 s; rope is shift-invariant)
STATE_DIM = 57
# slot-role ids (net role_embedder has 8 entries; 6 used)
ROLE_CUR_STATE, ROLE_RET_ACTION, ROLE_RET_VIDEO, ROLE_CUR_IMAGE, ROLE_ACTION, ROLE_FUTURE = 0, 1, 2, 3, 4, 5


class EgoVerseRetV2Dataset(EgoVerseRetDataset):
    """Compact role/timestamp mid-training dataset (see module doc). `num_frames` is
    ignored per-sample: each side uses its NATIVE length (92 human / 60 MD / 32 GR1)."""

    def __init__(self, *args, state_stats_path: str | None = None,
                 same_emb_top1: bool = False, **kw):
        super().__init__(*args, **kw)
        # same_emb_top1: INTRA-embodiment retrieval. The reference demo is the
        # best-ranked match whose embodiment EQUALS the query's (pinned per row — no
        # random draw among same-emb candidates), instead of the default lock on the
        # manifest's global top-1 embodiment (cross-embodiment ~51% of v10 rows).
        # Rows with no same-embodiment match in the K ranked columns are DROPPED
        # (v10: 0.2% of rows) — falling back to the query window itself would hand
        # the model its own answer as the demo. False => unchanged default behaviour.
        self.same_emb_top1 = bool(same_emb_top1)
        self._pinned = self._pin_same_emb_top1() if self.same_emb_top1 else None
        if self._pinned is not None:
            # demo embodiment == query embodiment by construction (bucket_key uses it)
            self._m_emb = self.wemb[self.query_idx].astype(object)
        else:
            # per-pair demo-embodiment lock so the sampler can bucket by (q_emb, m_emb):
            # each pair only samples matches whose embodiment == its FIRST valid match's.
            first = np.full(self.N, -1, np.int64)
            memb = np.empty(self.N, dtype=object)
            for r in range(self.N):
                ms = self.match_idx[r]; ms = ms[ms >= 0]
                if len(ms):
                    first[r] = int(ms[0]); memb[r] = str(self.wemb[int(ms[0])])
                else:
                    memb[r] = str(self.wemb[int(self.query_idx[r])])
            self._m_emb = memb
        # state57 stats (per-embodiment min-max; compute+cache if absent)
        sp = state_stats_path or os.path.join(self.manifest_dir, "state57_stats.json")
        self.state_stats = self._load_or_compute_state_stats(sp)

    # ---- demo (reference) selection ----
    def _pin_same_emb_top1(self) -> np.ndarray:
        """Row filter + per-row demo index for `same_emb_top1` (see __init__).

        match_idx columns are ranked best-first, so the FIRST column whose window
        carries the query's embodiment is the same-embodiment top-1."""
        n0 = self.N
        _, codes = np.unique(self.wemb, return_inverse=True)
        codes = codes.astype(np.int32)
        mi = self.match_idx
        ok = (mi >= 0) & (codes[np.clip(mi, 0, None)] == codes[self.query_idx][:, None])
        pinned = mi[np.arange(len(mi)), ok.argmax(1)]
        keep = ok.any(1)
        self.query_idx = self.query_idx[keep]
        self.match_idx = mi[keep]
        self.N = int(keep.sum())
        if self.epoch_length == n0:
            self.epoch_length = self.N
        print(f"[EgoVerseRetV2] same_emb_top1: kept {self.N}/{n0} rows "
              f"({n0 - self.N} had no same-embodiment match)", flush=True)
        return pinned[keep].astype(np.int64)

    def _choose_demo(self, r: int) -> int:
        """Demo (retrieved) window index for row r: the pinned same-embodiment top-1
        under `same_emb_top1`, else a random match of the row's locked embodiment."""
        pinned = getattr(self, "_pinned", None)     # subclasses may skip this __init__
        if pinned is not None:
            return int(pinned[r])
        ms = self.match_idx[r]; ms = ms[ms >= 0]
        lock = self._m_emb[r]
        cand = [int(m) for m in ms if str(self.wemb[int(m)]) == lock]
        return int(np.random.choice(cand)) if cand else int(self.query_idx[r])

    # ---- helpers ----
    @staticmethod
    def _is_md(emb: str) -> bool:
        return "mimicdroid" in str(emb)

    def _native(self, wi: int) -> tuple[int, float]:
        """(native frame count divisible by 4, latent dt seconds) for a window."""
        if self._is_md(self.wemb[wi]):
            return 60, 4.0 / 20.0
        return 92, 4.0 / 30.0                      # 90-frame window +2 dup (lossless)

    def _frames_native(self, wi: int, nf: int) -> np.ndarray:
        w0 = int(self.ww0[wi]); span = int(self.ww1[wi] - self.ww0[wi])
        fr = (w0 + np.linspace(0, span - 1, nf)).round().astype(np.int64)
        return np.clip(fr, 0, self._grp(wi).n_frames - 1)

    def _grid_frame(self, g, i: int) -> np.ndarray:
        """2x2 grid: primary cam TL; side cams (MD/GR1 3-cam eps) TR/BL; BR black.
        Single-cam (human) episodes -> TL only."""
        S = self.final_image_size; h = S // 2
        canvas = np.zeros((S, S, 3), np.uint8)
        b = g.frame(int(i))
        canvas[:h, :h] = _decode(b, h) if b else 0
        for (r0, c0), cam in (((0, h), "images.sideleft"), ((h, 0), "images.sideright")):
            try:
                bb = g.frame(int(i), cam)
                if bb:
                    canvas[r0:r0 + h, c0:c0 + h] = _decode(bb, h)
            except Exception:
                pass
        return canvas

    def _grid_clip(self, wi: int, fr: np.ndarray) -> np.ndarray:
        g = self._grp(wi)
        uniq = {int(j): self._grid_frame(g, int(j)) for j in np.unique(fr)}
        return np.stack([uniq[int(j)] for j in fr])

    def _action_native(self, wi: int, fr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """native-length action57 chunk (base _action57 is fixed num_frames)."""
        g = self._grp(wi)
        a = g.action("action57")
        out = a[np.clip(fr, 0, len(a) - 1)].astype(np.float32)
        return out, np.ones(ACTION_DIM, np.float32)

    def _load_or_compute_state_stats(self, path: str) -> dict:
        if os.path.exists(path):
            d = json.load(open(path))
            return {k: {"lo": np.asarray(v["lo"], np.float32), "hi": np.asarray(v["hi"], np.float32)}
                    for k, v in d.items()}
        print(f"[EgoVerseRetV2] computing state57 stats -> {path}", flush=True)
        rng = np.random.default_rng(0)
        by: dict = {}
        for r in rng.choice(self.N, min(3000, self.N), replace=False):
            wi = int(self.query_idx[int(r)])
            try:
                s = self._grp(wi).state("state57")
            except Exception:
                continue
            by.setdefault(str(self.wemb[wi]), []).append(np.asarray(s[int(self.ww0[wi])], np.float32))
        stats = {}
        allv = []
        for emb, rows in by.items():
            A = np.stack(rows); allv.append(A)
            lo = np.percentile(A, 1, 0); hi = np.percentile(A, 99, 0)
            stats[emb] = {"lo": lo.astype(np.float32),
                          "hi": np.where(hi - lo < 1e-4, lo + 1.0, hi).astype(np.float32)}
        A = np.concatenate(allv, 0)
        lo = np.percentile(A, 1, 0); hi = np.percentile(A, 99, 0)
        stats["_global"] = {"lo": lo.astype(np.float32),
                            "hi": np.where(hi - lo < 1e-4, lo + 1.0, hi).astype(np.float32)}
        try:
            json.dump({k: {"lo": v["lo"].tolist(), "hi": v["hi"].tolist()} for k, v in stats.items()},
                      open(path, "w"))
        except OSError:
            pass
        self._zcache = {}
        return stats

    def _norm_state(self, s: np.ndarray, emb: str) -> np.ndarray:
        st = self.state_stats.get(str(emb), self.state_stats["_global"])
        return np.clip(2.0 * (s - st["lo"]) / (st["hi"] - st["lo"] + 1e-8) - 1.0, -1.0, 1.0).astype(np.float32)

    # ---- sample ----
    def __getitem__(self, idx: int) -> dict:
        r = int(idx) % self.N
        q = int(self.query_idx[r])
        # bucket-consistent demo choice: only matches with the locked embodiment
        m = self._choose_demo(r)

        nf_q, dt_q = self._native(q)
        nf_m, dt_m = self._native(m)
        k, g = nf_m // NUM_DUPLICATES, nf_q // NUM_DUPLICATES
        T = k + g + 4

        fr_q = self._frames_native(q, nf_q)
        fr_m = self._frames_native(m, nf_m)
        q_video = self._grid_clip(q, fr_q)
        m_video = self._grid_clip(m, fr_m)
        q_act, _ = self._action_native(q, fr_q)
        m_act, _ = self._action_native(m, fr_m)
        q_act = self._normalize(q_act, self.wemb[q])
        m_act = self._normalize(m_act, self.wemb[m])

        # cur_state (query state57 at window start)
        try:
            st = np.asarray(self._grp(q).state("state57")[int(self.ww0[q])], np.float32)
        except Exception:
            st = np.zeros(STATE_DIM, np.float32)
        cur_state = self._norm_state(st, self.wemb[q])

        # ---- pixel packing: [state 1f | ret_act 4f | demo 4k | cur 4f | action 4f | future 4g]
        S = self.final_image_size
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)
        images = np.concatenate([
            blank1, blank4, m_video,
            np.repeat(q_video[:1], NUM_DUPLICATES, axis=0),
            blank4, q_video,
        ], axis=0)
        assert images.shape[0] == 1 + 4 * (T - 1), (images.shape, T)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=True, stronger_image_aug=True)

        # ---- roles + timestamps ----
        roles = np.empty(T, np.int64)
        tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        roles[1], tpos[1] = ROLE_RET_ACTION, NOW_S
        roles[2:2 + k] = ROLE_RET_VIDEO
        tpos[2:2 + k] = NOW_S - dt_m * np.arange(k, 0, -1)          # ends at NOW-dt
        roles[2 + k], tpos[2 + k] = ROLE_CUR_IMAGE, NOW_S
        roles[3 + k], tpos[3 + k] = ROLE_ACTION, NOW_S
        roles[4 + k:], tpos[4 + k:] = ROLE_FUTURE, NOW_S + dt_q * np.arange(1, g + 1)

        cls = self.wcls[q]
        cmd = f"{CLASS_TO_COMMAND.get(cls, CLASS_TO_COMMAND['other'])} [{self.wemb[q]}]"
        t5 = (torch.squeeze(self.t5_text_embeddings[cmd]) if cmd in self.t5_text_embeddings
              else torch.zeros(512, 1024))

        return {
            "video": images,
            "actions": q_act,                                # (nf_q,57) GEN target
            "action_mask": np.ones(ACTION_DIM, np.float32),
            "retrieved_actions": m_act,                      # (nf_m,57) cond
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
            "num_conditional_frames": k + 3,
            # latent indices
            "retrieved_video_start_latent_idx": 2,
            "retrieved_video_end_latent_idx": 2 + k,
            "retrieved_action_latent_idx": 1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 2 + k,
            "future_video_start_latent_idx": 4 + k,
            "future_video_end_latent_idx": T,
            "action_latent_idx": 3 + k,
            "proprio": cur_state, "future_proprio": np.zeros(STATE_DIM, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": 4 + k, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": 1, "has_ret_image": 1, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_act.copy(),
        }

    def bucket_key(self, r: int) -> tuple[bool, bool]:
        """(query_is_md, demo_is_md) — determines sequence shape T."""
        return (self._is_md(self.wemb[int(self.query_idx[r])]), self._is_md(self._m_emb[r]))


class BucketedDistributedSampler(Sampler):
    """Distributed sampler yielding SAME-BUCKET contiguous blocks of `batch_size`
    so every DataLoader batch has one sequence shape. Buckets interleave
    proportionally to their size; per-epoch shuffle within buckets."""

    def __init__(self, dataset: EgoVerseRetV2Dataset, batch_size: int,
                 num_replicas: int = 1, rank: int = 0, shuffle: bool = True, seed: int = 0,
                 drop_last: bool = True, **_ignored):
        # drop_last accepted for DistributedSampler API-compat; incomplete same-shape
        # blocks are ALWAYS dropped here regardless (bucket blocks must be full).
        self.ds = dataset; self.bs = int(batch_size)
        self.num_replicas = int(num_replicas); self.rank = int(rank)
        self.shuffle = shuffle; self.seed = seed; self.epoch = 0
        keys = [dataset.bucket_key(r) for r in range(dataset.N)]
        self.buckets: dict = {}
        for r, kk in enumerate(keys):
            self.buckets.setdefault(kk, []).append(r)

    def set_epoch(self, e: int):
        self.epoch = e

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        blocks = []
        gbs = self.bs * self.num_replicas                     # one global block per step
        for kk, idxs in self.buckets.items():
            arr = np.array(idxs)
            if self.shuffle:
                rng.shuffle(arr)
            nblk = len(arr) // gbs
            for b in range(nblk):
                blocks.append(arr[b * gbs:(b + 1) * gbs])
        order = rng.permutation(len(blocks)) if self.shuffle else np.arange(len(blocks))
        out = []
        for bi in order:
            blk = blocks[bi]                                  # (gbs,)
            out.extend(blk[self.rank * self.bs:(self.rank + 1) * self.bs].tolist())
        return iter(out)

    def __len__(self):
        gbs = self.bs * self.num_replicas
        return sum((len(v) // gbs) for v in self.buckets.values()) * self.bs
