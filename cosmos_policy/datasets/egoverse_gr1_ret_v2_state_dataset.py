"""GR1 post-training dataset (mid-v2 layout) — STATE-CHUNK prediction variant.

Identical to EgoVerseGr1RetV2Dataset (compact k=g=8 layout, roles + continuous-
time rope, 2x2 grid-3cam) EXCEPT the action semantics: every tensor that carried
action57 DELTA chunks now carries state57 ABSOLUTE chunks (per-emb normalized),
keeping tensor keys and model code unchanged:

  * "actions"/"next_action_chunk" : FUTURE state chunk s[qw0+1 .. qw0+32]
        (absolute state57 of the query episode), gr1-normalized.
  * "retrieved_actions"           : the matched demo's FUTURE state chunk from
        its w0, s[mw0+1 .. mw0+32], mimicdroid-normalized.
  * "proprio" (slot 0)            : the PAST STATE CHUNK s[qw0-31 .. qw0]
        (ending at the current state), tiled into latent slot 0 exactly like
        the action chunk (flattened to (32*57,)). Early-window convention:
        indices < 0 are LEFT-PADDED by repeating the episode's first frame s[0].

Everything else (video slots, roles, t_positions, t5/command, per-frame t+k
lockstep delta, num_conditional_frames=11, world_model_sample_mask=0) unchanged.

Normalization: per-emb from state57_stats.json, or ROBUST (mean +/- 4*std over
the manifest's episodes, cached at state57_stats_staterobust.json) when the
saturation gate triggered — see ROBUST_STATE_NORM below.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_gr1_ret_dataset import ACTION_DIM, STATE_DIM
from cosmos_policy.datasets.egoverse_gr1_ret_v2_dataset import (
    DT,
    K,
    NATIVE,
    NUM_DUPLICATES,
    EgoVerseGr1RetV2Dataset,
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
from cosmos_policy.datasets.egoverse_ret_v3_state_dataset import (
    ROBUST_SUFFIX,
    compute_robust_stats,
    dump_stats_json,
    load_stats_json,
)

# Saturation gate result (2026-07-10, 200 target chunks): min-max saturation
# |x|>0.9 was gr1 68.61% (constant head9 dims normalize to -1.0 under the lo+1
# degenerate-range guard, plus 1/99-pct tail clipping) -> exceeds the 20% gate,
# so the GR1 state dataset uses ROBUST normalization (mean +/- 4*std). After
# robust: gr1 0.26%.
ROBUST_STATE_NORM = True


def resolve_state_stats(manifest_dir: str, robust: bool = ROBUST_STATE_NORM) -> tuple[str, dict]:
    """(path, stats) for the state normalization the STATE datasets train with.
    Shared with the inference server so train/serve normalization always match."""
    base = os.path.join(manifest_dir, "state57_stats.json")
    path = base[:-len(".json")] + ROBUST_SUFFIX if robust else base
    return path, load_stats_json(path)


class EgoVerseGr1RetV2StateDataset(EgoVerseGr1RetV2Dataset):
    """GR1 v2 dataset with STATE-CHUNK targets (see module doc)."""

    def __init__(self, *args, robust_state_stats: bool = ROBUST_STATE_NORM,
                 target_group: str | None = None, target_stats_path: str | None = None,
                 no_ret_action: bool = False, no_future: bool = False,
                 ret_action_noise_std: float = 0.0, **kw):
        # ret_action_noise_std: ROBUSTNESS AUG. When >0, perturb the (already
        # normalized ~[-1,1]) retrieved_actions conditioning with Gaussian noise
        # whose sigma is drawn per-sample from U(0, ret_action_noise_std). Trains
        # the policy to tolerate a noisy demo-action prior; matched at eval by the
        # server's --ret-action-noise. Only applied when has_ret (never on blanks).
        self.ret_action_noise_std = float(ret_action_noise_std)
        # no_ret_action: DROP the RET_ACTION conditioning slot from the layout so the
        # retrieved demo contributes VIDEO ONLY (query still predicts action + future
        # video). Mirrors EgoVerseRetV3StateDataset.no_ret_action; state_t 20->19,
        # num_conditional 11->10, chunk_duration 77->73, retrieved_action_latent_idx=-1.
        # MUST be set BEFORE super().__init__(): the base ctor reads self.state_t (which
        # this flag shrinks) in its startup print.
        self.no_ret_action = bool(no_ret_action)
        # no_future: DROP the FUTURE_VIDEO block (K latents) from the layout -> the ACTION
        # slot becomes the ONLY generated target (action-only post-train; the demo VIDEO
        # conditioning is untouched, so ICL survives). Mirrors
        # EgoVerseRetV3StateDataset.no_future: the future latents are not merely masked out
        # of the loss, they are not in the sequence at all, so the step is cheaper too.
        # state_t 20->12, chunk_duration 77->45, num_conditional_frames unchanged (11).
        # Like no_ret_action this MUST be set BEFORE super().__init__() (the base ctor
        # reads self.state_t in its startup print).
        self.no_future = bool(no_future)
        super().__init__(*args, **kw)
        # v6 (user 2026-07-14): predict ABSOLUTE COMMANDED targets (abs_action57,
        # site frame, row t = command at t -> state t+1) instead of achieved
        # state chunks — executing predictions reproduces the demo's tracking
        # dynamics and removes the achieved-vs-commanded double lag.
        self.target_group = target_group
        self.target_stats = None
        if target_group is not None and target_stats_path:
            import json as _json
            with open(target_stats_path) as _f:
                _d = _json.load(_f)
            self.target_stats = {k: {"lo": np.asarray(v["lo"], np.float32),
                                     "hi": np.asarray(v["hi"], np.float32)}
                                 for k, v in _d.items()}
        self.robust_state_stats = bool(robust_state_stats)
        if self.robust_state_stats:
            base = kw.get("state_stats_path") or os.path.join(self.manifest_dir, "state57_stats.json")
            rpath = base[:-len(".json")] + ROBUST_SUFFIX if base.endswith(".json") else base + ROBUST_SUFFIX
            self.state_stats = self._load_or_compute_robust_state_stats(rpath)
            print(f"[EgoVerseGr1RetV2State] ROBUST state norm (mean+/-4std) <- {rpath}", flush=True)

    @property
    def state_t(self) -> int:
        # video-only demo conditioning drops the RET_ACTION latent, action-only drops the
        # whole FUTURE_VIDEO block (chunk_duration, which is defined off state_t in the
        # base, shrinks 77->73 / 77->45 automatically).
        return (EgoVerseGr1RetV2Dataset.state_t.fget(self)
                - (1 if self.no_ret_action else 0)
                - (K if self.no_future else 0))

    def _load_or_compute_robust_state_stats(self, path: str, max_eps_per_emb: int = 200) -> dict:
        if os.path.exists(path):
            return load_stats_json(path)
        print(f"[EgoVerseGr1RetV2State] computing ROBUST state57 stats -> {path}", flush=True)
        rng = np.random.default_rng(0)
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
                    chunks.append(np.asarray(self._grp(wis[int(si)]).state(self.state_group), np.float64))
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

    @staticmethod
    def _past_chunk(arr: np.ndarray, t: int, n: int) -> np.ndarray:
        """PAST chunk s[t-n+1 .. t] (inclusive of the current state t); indices < 0
        left-pad by repeating the episode's first frame s[0]."""
        idx = np.arange(t - n + 1, t + 1)
        return np.asarray(arr[np.clip(idx, 0, len(arr) - 1)])

    def __getitem__(self, idx: int) -> dict:
        if self.debug:
            idx = 0
        r = idx % self.N
        q = int(self.query_idx[r])
        ms = self.match_idx[r]; ms = ms[ms >= 0]
        has_ret = 1 if len(ms) else 0
        m = int(ms[np.random.randint(len(ms))]) if has_ret else q
        return self.build_sample(q, m, idx=idx, has_ret=has_ret, aug=True, rand_delta=True)

    def build_sample(self, q: int, m: int, idx: int = 0, has_ret: int = 1,
                     aug: bool = False, rand_delta: bool = False) -> dict:
        """Deterministic sample for an explicit (query window q, demo window m) —
        the offline batch eval (eval_demos + build_sample contract of the base
        dataset) uses the defaults (no aug, no random per-frame delta);
        __getitem__ delegates with the training randomness enabled."""
        gq = self._grp(q); gm = self._grp(m)
        qw0 = int(self.ww0[q]); mw0 = int(self.ww0[m])

        # v6: NO stochastic per-frame jitter — the v6 pairs enumerate every
        # timestep as an explicit row (1 epoch = full coverage; user rule
        # 2026-07-14). rand_delta is accepted for API compat but ignored.

        # STATE-CHUNK semantics -------------------------------------------------
        q_state = gq.state(self.state_group)
        m_state = gm.state(self.state_group)
        q_emb, m_emb = self.wemb[q], self.wemb[m]
        # target: v6 default = FUTURE absolute COMMAND chunk act[qw0 .. qw0+31]
        # (abs_action57; act[t] produces s[t+1] — same horizon as the state chunk);
        # legacy (target_group=None) = achieved state chunk s[qw0+1 .. qw0+32].
        if self.target_group is not None:
            q_tgt = self._chunk(np.asarray(gq.action(self.target_group), np.float64), qw0)
        else:
            q_tgt = self._chunk(q_state, qw0 + 1)
        # demo conditioning: demo's FUTURE state chunk s[mw0+1 .. mw0+32]
        m_ret = self._chunk(m_state, mw0 + 1)
        if self.normalize_state:
            _tstats = self.target_stats if (self.target_group and self.target_stats) else self.state_stats
            q_tgt = self._norm(q_tgt, _tstats, q_emb)
            m_ret = self._norm(m_ret, self.state_stats, m_emb) if has_ret \
                else np.zeros((NATIVE, STATE_DIM), np.float32)
        elif not has_ret:
            m_ret = np.zeros((NATIVE, STATE_DIM), np.float32)
        # robustness aug: perturb the normalized demo-action prior with per-sample
        # Gaussian noise, sigma ~ U(0, ret_action_noise_std). Only on real retrievals.
        if self.ret_action_noise_std > 0.0 and has_ret:
            _sig = float(np.random.uniform(0.0, self.ret_action_noise_std))
            m_ret = (m_ret + np.random.normal(0.0, _sig, m_ret.shape)).astype(np.float32)
        # slot 0: PAST state chunk s[qw0-31 .. qw0] (left-pad with s[0] when qw0 < 31)
        past = self._past_chunk(q_state, qw0, NATIVE).astype(np.float32)
        if self.normalize_state:
            past = self._norm(past, self.state_stats, q_emb)
        # ------------------------------------------------------------------------

        # video layout (o = RET_ACTION slot present, fu = FUTURE_VIDEO block present):
        #   [state 1f | (ret_act 4f) | demo 32f | cur 4f | act 4f | (future 32f)]
        #   -> 77f (o=1,fu=1)  73f (o=0, video-only demo cond)  45f (fu=0, action-only)
        o = 0 if self.no_ret_action else 1
        fu = 0 if self.no_future else 1
        q_clip = self._grid_clip(gq, qw0)
        m_clip = self._grid_clip(gm, mw0) if has_ret else np.zeros_like(q_clip)
        S = self.final_image_size
        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUPLICATES, axis=0)
        images = np.concatenate(
            [blank1] + ([blank4] if o else []) + [
                m_clip,
                np.repeat(q_clip[:1], NUM_DUPLICATES, axis=0),
                blank4,
            ] + ([q_clip] if fu else []), axis=0)
        assert images.shape[0] == self.chunk_duration
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=aug, stronger_image_aug=aug)

        T = self.state_t
        roles = np.empty(T, np.int64); tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        if o:
            roles[1], tpos[1] = ROLE_RET_ACTION, NOW_S
        roles[1 + o:1 + o + K] = ROLE_RET_VIDEO
        tpos[1 + o:1 + o + K] = NOW_S - DT * np.arange(K, 0, -1)
        roles[1 + o + K], tpos[1 + o + K] = ROLE_CUR_IMAGE, NOW_S
        roles[2 + o + K], tpos[2 + o + K] = ROLE_ACTION, NOW_S
        if fu:
            roles[3 + o + K:], tpos[3 + o + K:] = ROLE_FUTURE, NOW_S + DT * np.arange(1, K + 1)

        cls = self.wcls[q]
        t5 = (torch.squeeze(self.t5_text_embeddings[cls]) if cls in self.t5_text_embeddings
              else torch.zeros(512, 1024))

        return {
            "video": images,
            "command": str(cls),
            "actions": q_tgt.astype(np.float32),             # (32,57) FUTURE state chunk (GEN target)
            "action_mask": np.ones(ACTION_DIM, np.float32),
            "retrieved_actions": m_ret.astype(np.float32),   # (32,57) demo FUTURE state chunk (cond)
            "retrieved_action_mask": np.ones(ACTION_DIM, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": idx,
            "embodiment": q_emb, "retrieved_embodiment": m_emb,
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": K + 2 + o,             # 11 (o=1) / 10 (o=0)
            "retrieved_video_start_latent_idx": 1 + o,
            "retrieved_video_end_latent_idx": 1 + o + K,
            "retrieved_action_latent_idx": 1 if o else -1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 1 + o + K,
            "future_video_start_latent_idx": (3 + o + K) if fu else T,   # start==end==T -> empty
            "future_video_end_latent_idx": T,
            "action_latent_idx": 2 + o + K,                  # 11 (o=1) / 10 (o=0)
            # PAST state chunk tiled into slot 0 (flattened (32*57,) — same tiling
            # mechanism replace_latent_with_proprio uses for the action chunk).
            "proprio": past.reshape(-1).astype(np.float32),
            "future_proprio": np.zeros(STATE_DIM, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": (3 + o + K) if fu else -1, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": has_ret, "has_ret_image": has_ret, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_tgt.copy().astype(np.float32),
        }
