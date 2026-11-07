"""EgoVerse mid-training V3 — STATE-CHUNK prediction variant.

Identical to EgoVerseRetV3Dataset (v2 compact layout, full-frame video, roles +
continuous-time rope) EXCEPT the action semantics: every tensor that carried
action57 DELTA chunks now carries state57 ABSOLUTE chunks (per-emb min-max
normalized), keeping tensor keys and model code unchanged:

  * "actions"/"next_action_chunk"  : FUTURE state chunk s[t+1 .. t+CHUNK] of the
        query window (states at fr+1 for the same native frame tiling fr the
        delta variant read actions at; CHUNK = native nf 92/60).
  * "retrieved_actions"            : the matched demo's FUTURE state chunk from
        its w0 (states at fr_m+1), demo-emb normalized.
  * "proprio" (slot 0)             : the PAST STATE CHUNK s[t-CHUNK+1 .. t]
        (states at fr - span + 1, ending at the window start = "now"), tiled
        into latent slot 0 exactly like the action chunk (flattened to
        (CHUNK*57,) so replace_latent_with_proprio tiles the full chunk).
        Early-window convention: indices < 0 are LEFT-PADDED by repeating the
        episode's first frame s[0].

Everything else (video slots, roles, t_positions, t5/command,
num_conditional_frames, world_model_sample_mask=0) is unchanged.

Normalization: per-emb min-max from state57_stats.json. If ROBUST_STATE_NORM
(set from the saturation gate: >20% of normalized target values |x|>0.9 for any
embodiment) the dataset instead uses per-dim mean +/- 4*std ranges computed over
the manifest's episodes, cached at state57_stats_staterobust.json.
"""
from __future__ import annotations

import json
import os

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
    ROLE_RET_ACTION,
    ROLE_RET_VIDEO,
    STATE_DIM,
)
from cosmos_policy.datasets.egoverse_ret_v3_dataset import EgoVerseRetV3Dataset

# Saturation gate result (2026-07-10, 200 target chunks): min-max saturation
# |x|>0.9 was mimicdroid 49.69% / MECKA 7.80% / scale 7.33% / aria 6.08% -> the
# mimicdroid embodiment exceeds the 20% gate (constant head9 dims normalize to
# -1.0 under the lo+1 degenerate-range guard), so the MID state dataset uses
# ROBUST normalization (mean +/- 4*std). After robust: mimicdroid 0.85% /
# MECKA 0.70% / scale 0.21% / aria 0.05%.
ROBUST_STATE_NORM = True
ROBUST_SUFFIX = "_staterobust.json"


def compute_robust_stats(rows_by_emb: dict[str, np.ndarray], guard: float = 1e-4) -> dict:
    """{emb: (N,57) rows} -> {emb: {lo,hi}} with lo/hi = mean -/+ max(4*std, guard).

    Symmetric guard keeps constant dims centered at 0 after the standard
    2*(x-lo)/(hi-lo)-1 min-max mapping (instead of saturating at -1 like the
    degenerate hi=lo+1 min-max guard does)."""
    stats = {}
    for emb, A in rows_by_emb.items():
        A = np.asarray(A, np.float64)
        mean = A.mean(0)
        half = np.maximum(4.0 * A.std(0), guard)
        stats[emb] = {"lo": (mean - half).astype(np.float32), "hi": (mean + half).astype(np.float32)}
    allv = np.concatenate([np.asarray(v, np.float64) for v in rows_by_emb.values()], 0)
    mean = allv.mean(0)
    half = np.maximum(4.0 * allv.std(0), guard)
    stats["_global"] = {"lo": (mean - half).astype(np.float32), "hi": (mean + half).astype(np.float32)}
    return stats


def load_stats_json(path: str) -> dict:
    d = json.load(open(path))
    return {k: {"lo": np.asarray(v["lo"], np.float32), "hi": np.asarray(v["hi"], np.float32)}
            for k, v in d.items()}


def dump_stats_json(stats: dict, path: str) -> None:
    json.dump({k: {"lo": v["lo"].tolist(), "hi": v["hi"].tolist()} for k, v in stats.items()},
              open(path, "w"))


class EgoVerseRetV3StateDataset(EgoVerseRetV3Dataset):
    """V3 (full-frame) mid dataset with STATE-CHUNK targets (see module doc)."""

    def __init__(self, *args, robust_state_stats: bool = ROBUST_STATE_NORM,
                 robust_stats_file: str | None = None, no_ret_action: bool = False,
                 video_only: bool = False, blank_ret_video: bool = False,
                 no_future: bool = False, delta_target: bool = False,
                 blank_retrieval: bool = False, blank_video_slots: int = -1,
                 blank_retrieval_prob: float = 0.0, blank_retrieval_slots: int = 2,
                 cap_ret_video: int = -1, **kw):
        super().__init__(*args, **kw)
        self.robust_state_stats = bool(robust_state_stats)
        # delta_target: emit DELTA action57 chunks (base _normalize + action57_stats.json) as
        # the GEN target / demo cond, and a SINGLE current state57 as proprio — instead of the
        # default ABSOLUTE state57 FUTURE chunks. Lets the same blank/capped/vidonly/actonly
        # layout machinery produce a DELTA mid (matches EgoVerseRetV3Dataset target semantics).
        # False => byte-identical to the existing state variant.
        self.delta_target = bool(delta_target)
        # blank_video_slots: when >=0 (and a blank-video mode is on), CAP the RET_VIDEO block
        # to this many blank latents instead of the demo's full k=nf_m//4 (~15-23). Tests
        # whether the transfer-seeding STRUCTURE needs the whole video block or just a couple
        # of ret slots (roles 1/2 + a shallow past axis) — a much cheaper mid (state_t ~= g+4).
        # The post-train still uses its own full k=8 block; the mid only needs to have EXERCISED
        # the ret_video role + a past-time rotation, not matched the post's exact depth.
        self.blank_video_slots = int(blank_video_slots)
        # blank_retrieval: keep the FULL transfer layout (RET_ACTION + RET_VIDEO slots,
        # roles 1/2, past time axis at NOW-k*dt, action positioned AFTER the video block)
        # but ZERO the demo CONTENT. The slots stay CONDITIONING (has_ret_data/image=1), so
        # the mid learns the retrieval STRUCTURE with no retrieval information — still BC
        # (action generated from cur obs only). A post-train transfer run then fills these
        # blanks with real demos, so ONLY the content changes and the action-generation
        # timing / roles 1,2 / past-rope are already warm. Isolates "does mid need real demo
        # CONTENT, or just the STRUCTURE?" — the fix for the plain-mid->transfer-post gap
        # (a pure no-demo mid seeds a transfer post no better than base; see
        # docs/egoverse_plain_vs_transfer_ablation.md).
        self.blank_retrieval = bool(blank_retrieval)
        # blank_retrieval_prob: retrieval DROPOUT — this fraction of TRAINING rows is served
        # without a demo (0.2 => 20% BC-style / 80% retrieval-conditioned), so the model must
        # be able to act with AND without one. blank_retrieval=True (constant) takes
        # precedence; eval (build_sample(aug=False)) NEVER drops, so offline metrics stay
        # deterministic and fully conditioned.
        #
        # Dropped rows are genuinely SHORTER, not zero-padded: the RET_VIDEO block is capped
        # to blank_retrieval_slots (2, the bv2 layout) instead of the demo's full k=nf_m//4,
        # so a human-query row goes state_t 50 -> 29 / n_cond 26 -> 5. That is free compute:
        # ~0.2*(1-29/50) of the sequence disappears. Mixing lengths is already the norm here
        # (md vs human native lengths give 4 buckets / 4 different T with state_t=50 in the
        # config), state_t is only a generation default, and num_conditional_frames travels
        # per sample in the batch.
        #
        # The mask is DETERMINISTIC per row (fixed seed, epoch-independent) — NOT a coin flip
        # in __getitem__. BucketedDistributedSampler computes bucket_key ONCE at init, so a
        # random per-call drop would put T=29 and T=50 samples in one batch and the collate
        # would fail. Epoch-varying masks are also unusable: persistent_workers=True means
        # set_epoch never reaches the worker copies of the dataset. Consequence: a dropped
        # row is ALWAYS demo-free. Harmless in practice — rows are ~1 s-strided windows, so
        # every episode still appears in both conditions.
        self.blank_retrieval_prob = float(blank_retrieval_prob)
        self.blank_retrieval_slots = int(blank_retrieval_slots)
        # cap_ret_video: keep the demo CONTENT but shrink its RET_VIDEO block to this many
        # latents (blank_video_slots does the same shape change with the content ZEROED).
        # Needed to score a blankret_bv* checkpoint on REAL demos: that model only ever saw
        # a k-latent ret block, so the demo clip is TEMPORALLY SUBSAMPLED to 4*k frames over
        # the same window rather than truncated — same 3 s span, coarser stride.
        self.cap_ret_video = int(cap_ret_video)
        self._drop = (np.random.default_rng(20260812).random(self.N) < self.blank_retrieval_prob
                      if self.blank_retrieval_prob > 0 else np.zeros(self.N, bool))
        if self.blank_retrieval_prob > 0:
            print(f"[EgoVerseRetV3State] retrieval dropout p={self.blank_retrieval_prob} -> "
                  f"{int(self._drop.sum())}/{self.N} rows demo-free "
                  f"(ret_video capped to {self.blank_retrieval_slots} latents)", flush=True)
        # no_ret_action: DROP the RET_ACTION slot from the layout (demo conditioning =
        # VIDEO only). Mid learns to read demo video WITHOUT the action channel; the
        # post-train then ADDS ret_action back (1 token) — the reverse of the failed
        # no-video->video direction (expensive video learned in mid, cheap token added in post).
        self.no_ret_action = bool(no_ret_action)
        # video_only: PURE video->video mid — drop BOTH the RET_ACTION conditioning slot
        # AND the ACTION generation slot. Conditioning = demo VIDEO (ret_video) + current
        # image (+ proprio); the ONLY generated target is FUTURE_VIDEO. No action touches the
        # EDM/attention at all. The action policy is added back entirely in the post-train
        # (action_latent_idx flips from -1 to a real slot), warm-started from this mid.
        self.video_only = bool(video_only)
        # blank_ret_video: demo VIDEO blanked, demo ACTION (ret_action) kept -> action-only
        # demo conditioning (mirror of video_only). Full layout preserved for clean transfer.
        self.blank_ret_video = bool(blank_ret_video)
        # no_future: DROP the FUTURE_VIDEO block entirely (no g future latents in the
        # sequence). The action slot becomes the only generated target -> action-only mid
        # WITHOUT wasting compute on masked-but-processed future latents. state_t collapses
        # from k+g+4 to k+4 (much faster). Pair with blank_ret_video for a cheap action mid.
        self.no_future = bool(no_future)
        if robust_stats_file is not None:
            # explicit precomputed stats (e.g. state57_stats_pctl.json: percentile
            # p0.5/p99.5, corrupt-eps-excluded) — must already exist, never computed here
            assert os.path.exists(robust_stats_file), f"missing stats: {robust_stats_file}"
            self.state_stats = load_stats_json(robust_stats_file)
            print(f"[EgoVerseRetV3State] EXPLICIT state norm <- {robust_stats_file}", flush=True)
        elif self.robust_state_stats:
            base = kw.get("state_stats_path") or os.path.join(self.manifest_dir, "state57_stats.json")
            rpath = base[:-len(".json")] + ROBUST_SUFFIX if base.endswith(".json") else base + ROBUST_SUFFIX
            self.state_stats = self._load_or_compute_robust_state_stats(rpath)
            print(f"[EgoVerseRetV3State] ROBUST state norm (mean+/-4std) <- {rpath}", flush=True)

    # ---- robust stats (mean +/- 4*std over the manifest's episodes) ----
    def _load_or_compute_robust_state_stats(self, path: str, max_eps_per_emb: int = 200) -> dict:
        if os.path.exists(path):
            return load_stats_json(path)
        print(f"[EgoVerseRetV3State] computing ROBUST state57 stats -> {path}", flush=True)
        rng = np.random.default_rng(0)
        # unique episodes per embodiment across ALL manifest windows
        eps_by_emb: dict[str, list[int]] = {}
        seen: set[tuple[str, str]] = set()
        for wi in range(len(self.wep)):
            key = (str(self.wemb[wi]), str(self.wep[wi]))
            if key not in seen:
                seen.add(key)
                eps_by_emb.setdefault(key[0], []).append(wi)
        rows_by_emb: dict[str, np.ndarray] = {}
        for emb, wis in eps_by_emb.items():
            sel = rng.choice(len(wis), min(max_eps_per_emb, len(wis)), replace=False)
            chunks = []
            for si in sel:
                try:
                    chunks.append(np.asarray(self._grp(wis[int(si)]).state("state57"), np.float64))
                except Exception:
                    continue
            if chunks:
                rows_by_emb[emb] = np.concatenate(chunks, 0)
        stats = compute_robust_stats(rows_by_emb)
        try:
            dump_stats_json(stats, path)
        except OSError:
            pass
        self._zcache = {}
        return stats

    # ---- state chunk access ----
    def _state_rows(self, wi: int, idx: np.ndarray) -> np.ndarray:
        """state57 rows at absolute frame indices idx; idx<0 clips to s[0]
        (= left-pad by repeating the episode's first frame)."""
        s = self._grp(wi).state("state57")
        return np.asarray(s[np.clip(idx, 0, len(s) - 1)], np.float32)

    # ---- sample (clone of EgoVerseRetV2Dataset.__getitem__, state-chunk semantics) ----
    def bucket_key(self, r: int):
        """(query_is_md, demo_is_md, dropped) — the sequence shape T. A dropped row's
        RET_VIDEO block is capped to blank_retrieval_slots, so its T no longer depends on
        the demo's native length -> collapse the demo axis for those rows (one bucket)."""
        base = super().bucket_key(r)
        if not self._drop[int(r)]:
            return base + (False,)
        return (base[0], False, True)

    def __getitem__(self, idx: int) -> dict:
        r = int(idx) % self.N
        q = int(self.query_idx[r])
        m = self._choose_demo(r)
        return self.build_sample(q, m, idx=idx, aug=True)

    def build_sample(self, q: int, m: int, idx: int = 0, aug: bool = False) -> dict:
        """Deterministic sample for an explicit (query window q, demo window m) —
        the offline batch eval (eval_demos + build_sample contract of the base
        dataset) uses this with aug=False; __getitem__ delegates with aug=True."""
        nf_q, dt_q = self._native(q)
        nf_m, dt_m = self._native(m)
        k, g = nf_m // NUM_DUPLICATES, nf_q // NUM_DUPLICATES
        # retrieval dropout: this row is served demo-free -> SHORTEN the ret_video block
        # (bucket_key accounts for it, so a batch never mixes the two lengths).
        drop = bool(aug and not self.blank_retrieval and self._drop[int(idx) % self.N])
        if drop:
            k = self.blank_retrieval_slots
        if self.cap_ret_video >= 0 and not (self.blank_retrieval or self.blank_ret_video or drop):
            k = self.cap_ret_video                # REAL demo, shorter block
            nf_m = k * NUM_DUPLICATES             # -> re-read the demo at 4k frames (same span)
        if self.blank_video_slots >= 0:
            assert self.blank_retrieval or self.blank_ret_video, \
                "blank_video_slots requires blank_retrieval or blank_ret_video (video must be blank to cap it)"
            k = self.blank_video_slots            # cap the (blank) ret_video block
        o = 0 if (self.no_ret_action or self.video_only) else 1   # RET_ACTION cond slot?
        a = 0 if self.video_only else 1                           # ACTION gen slot?
        fu = 0 if self.no_future else 1                           # FUTURE_VIDEO block?
        gf = g * fu                                               # future latents (0 if dropped)
        T = k + gf + 2 + o + a

        fr_q = self._frames_native(q, nf_q)
        fr_m = self._frames_native(m, nf_m)
        q_video = self._grid_clip(q, fr_q)
        m_video = self._grid_clip(m, fr_m)

        # TARGET semantics ----------------------------------------------------
        span_q = int(self.ww1[q] - self.ww0[q])
        if self.delta_target:
            # DELTA action57 chunks (like EgoVerseRetV3Dataset): GEN target + demo cond,
            # proprio = SINGLE current state57 (tiled at slot 0).
            q_a, _ = self._action_native(q, fr_q); q_state = self._normalize(q_a, self.wemb[q])
            m_a, _ = self._action_native(m, fr_m); m_state = self._normalize(m_a, self.wemb[m])
            try:
                st = np.asarray(self._grp(q).state("state57")[int(self.ww0[q])], np.float32)
                past_state = self._norm_state(st, self.wemb[q])          # (57,)
            except Exception:
                past_state = np.zeros(STATE_DIM, np.float32)
        else:
            # ABSOLUTE state57 FUTURE chunks (default).
            q_state = self._norm_state(self._state_rows(q, fr_q + 1), self.wemb[q])
            m_state = self._norm_state(self._state_rows(m, fr_m + 1), self.wemb[m])
            past_idx = fr_q - span_q + 1
            past_state = self._norm_state(self._state_rows(q, past_idx), self.wemb[q])
        # ----------------------------------------------------------------------

        # BLANK-DEMO: zero the retrieved CONTENT (video pixels + demo chunk) while
        # keeping the layout/roles/tpos/has_ret intact -> structure without information.
        # Blank ret_video is rebuilt at the (possibly capped) k length, not zeros_like the
        # full-demo clip, so blank_video_slots can shrink the block.
        _S = self.final_image_size
        _blank_vid = np.zeros((k * NUM_DUPLICATES, _S, _S, 3), np.uint8)
        if self.blank_retrieval or drop:
            m_video = _blank_vid
            m_state = np.zeros_like(m_state)
        # blank_ret_video: zero ONLY the demo VIDEO (ret_video slot stays present but blank)
        # -> demo conditioning = ret_action (m_state) ONLY. Mirror of video_only. Pair with
        # mask_loss_for_action_future_state_prediction=True so there is no video objective.
        if self.blank_ret_video:
            m_video = _blank_vid

        # ---- pixel packing: [state 1f | ret_act 4f | demo 4k | cur 4f | action 4f | future 4g]
        S = self.final_image_size
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)
        images = np.concatenate(
            [blank1] + ([blank4] if o else []) + [
                m_video,
                np.repeat(q_video[:1], NUM_DUPLICATES, axis=0),
            ] + ([blank4] if a else []) + ([q_video] if fu else []),
            axis=0)
        assert images.shape[0] == 1 + 4 * (T - 1), (images.shape, T)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=aug, stronger_image_aug=aug)

        # ---- roles + timestamps (o = 1 with RET_ACTION, 0 without) ----
        roles = np.empty(T, np.int64)
        tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        if o:
            roles[1], tpos[1] = ROLE_RET_ACTION, NOW_S
        roles[1 + o:1 + o + k] = ROLE_RET_VIDEO
        tpos[1 + o:1 + o + k] = NOW_S - dt_m * np.arange(k, 0, -1)
        roles[1 + o + k], tpos[1 + o + k] = ROLE_CUR_IMAGE, NOW_S
        if a:
            roles[2 + o + k], tpos[2 + o + k] = ROLE_ACTION, NOW_S
        if fu:
            roles[2 + o + k + a:], tpos[2 + o + k + a:] = ROLE_FUTURE, NOW_S + dt_q * np.arange(1, g + 1)

        cls = self.wcls[q]
        cmd = f"{CLASS_TO_COMMAND.get(cls, CLASS_TO_COMMAND['other'])} [{self.wemb[q]}]"
        t5 = (torch.squeeze(self.t5_text_embeddings[cmd]) if cmd in self.t5_text_embeddings
              else torch.zeros(512, 1024))

        return {
            "video": images,
            "actions": q_state,                              # (nf_q,57) FUTURE state chunk (GEN target)
            "action_mask": np.ones(ACTION_DIM, np.float32),
            "retrieved_actions": m_state,                    # (nf_m,57) demo FUTURE state chunk (cond)
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
            "num_conditional_frames": k + 2 + o,
            # latent indices (shift by o: 1 with RET_ACTION at slot 1, 0 without)
            "retrieved_video_start_latent_idx": 1 + o,
            "retrieved_video_end_latent_idx": 1 + o + k,
            "retrieved_action_latent_idx": 1 if o else -1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 1 + o + k,
            "future_video_start_latent_idx": 2 + o + k + a,
            "future_video_end_latent_idx": T,
            "action_latent_idx": (2 + o + k) if a else -1,
            # PAST state chunk tiled into slot 0 (flattened: replace_latent_with_proprio
            # tiles a (B,D) vector — D = nf_q*57 here, same mechanism as the action chunk).
            "proprio": past_state.reshape(-1),
            "future_proprio": np.zeros(STATE_DIM, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": (2 + o + k + a) if fu else -1, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": 1, "has_ret_image": 1, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_state.copy(),
        }
