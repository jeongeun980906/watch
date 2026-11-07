"""EgoVerse cross-embodiment retrieval-augmented dataset (Cosmos Predict-2.5).

Consumes the v9 retrieval manifest at ``/mnt/ddn/dataset/egoverse_pairs_v9/``
(pairs.npz + windows.npz) and emits one sample per (query window, retrieved
neighbour) pair. Reads frames/actions directly from the per-episode zarr stores
(``/mnt/ddn/dataset/egoverse`` for human aria/scale/MECKA, ``egoverse_eva`` for
the eva robot).

Design (confirmed):
  * NO proprio / state slots.
  * Embodiment tag via TEXT: command = f"{class_phrase} [{embodiment}]" -> T5.
  * Native action57 (v9 is human-only aria/scale/MECKA, all 57 dims valid):
        head9 | L_eef9 | L_hand15 | R_eef9 | R_hand15
      The eva grippers (old 59-D union's L_grip1/R_grip1) are dropped; eva
      (action20) if ever loaded scatters L_eef9/R_eef9 only (grips masked).
  * Full-frame video (longer T): num_frames sampled evenly from the 5 s window
    (must be divisible by NUM_DUPLICATES=4). 1 latent slot = 4 frames.

Latent layout (R = num_frames // 4):
    0          : blank sentinel
    1 .. R      : ret_video           (conditioning, retrieved demo)
    R+1        : ret_action           (blank; action injected)        cond
    R+2        : query_image          (first query frame, observation) cond
    R+3 .. 2R+2: query_video          (GENERATED future video)
    2R+3       : query_action         (blank; model generates)
  => chunk_duration = 1 + R + 1 + 1 + R + 1 = 2R + 4 latents.
  min_num_conditional_frames covers latents 0 .. R+2 (= R+3 latents).
"""
from __future__ import annotations

import io
import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_zread import EgoZarrEpisode

# Episode root (one dir per episode: images.front_1 + state57/action57 side by side).
# EGOVERSE_HUM_ROOT overrides it for setups outside the cluster's /mnt/ddn layout.
HUM_ROOT = os.environ.get("EGOVERSE_HUM_ROOT", "/mnt/ddn/dataset/egoverse")
EVA_ROOT = "/mnt/ddn/dataset/egoverse_eva"
HE_ROOT = "/mnt/ddn/dataset/egoverse_he"    # Humanoid-Everyday (embodiment 'he')
DEFAULT_MANIFEST = "/mnt/ddn/dataset/egoverse_pairs_v9"
DEFAULT_T5_EMB = "/mnt/ddn/dataset/egoverse_pairs_v9/t5_embeddings.pkl"
DEFAULT_ACT_STATS = "/mnt/ddn/dataset/egoverse_pairs_v9/action57_stats.json"

NUM_DUPLICATES = 4
ACTION_DIM = 57
TIPS = [4, 8, 12, 16, 20]

# native action57 slices (v9 is human-only: aria/scale/MECKA). The eva gripper dims
# (Lgrip/Rgrip) of the old 59-D union are dropped — eva is never referenced by v9 pairs.
SL_HEAD = slice(0, 9)
SL_LEEF = slice(9, 18)
SL_LHAND = slice(18, 33)
SL_REEF = slice(33, 42)
SL_RHAND = slice(42, 57)

# activity class -> readable command (matches t5_embeddings.pkl keys, sans tag).
# v9 lang_taxonomy classes (20) — the cls values in egoverse_pairs_v9/windows.npz.
# Regenerate t5_embeddings.pkl whenever this vocabulary changes.
CLASS_TO_COMMAND = {
    "assemble": "assembling parts",
    "bagging_groceries": "bagging groceries",
    "clean_wipe": "wiping and cleaning a surface",
    "cup_on_saucer": "placing a cup on a saucer",
    "dishwasher_other": "handling dishwasher items",
    "do_dishes": "doing the dishes",
    "fill_scoop": "filling and scooping",
    "fold": "folding cloth",
    "load_dishwasher": "loading the dishwasher",
    "pick_place": "picking up and placing an object",
    "put_object_in_container": "putting an object into a container",
    "seal_wrap": "sealing and wrapping",
    "sort_bins": "sorting objects into bins",
    "stack": "stacking objects",
    "tidy_kitchen": "tidying the kitchen",
    "tie": "tying",
    "unload_dishwasher": "unloading the dishwasher",
    "unpack_grocery": "unpacking groceries",
    "utensils_in_drawer": "putting utensils in a drawer",
    # MimicDroid (GR1 play) subgoal classes — v8 MD rows brought into v9
    "door_close": "closing a cabinet door",
    "door_open": "opening a cabinet door",
    "faucet_on": "turning on the faucet",
    "faucet_off": "turning off the faucet",
    "other": "manipulating objects",   # also covers MD cls "none"
}
EMB_TAGS = ["aria", "scale", "MECKA", "eva", "human", "mimicdroid"]


def _commands_with_tags() -> list[str]:
    """All (class, embodiment) command strings — for T5 pre-computation."""
    return [f"{c} [{e}]" for c in CLASS_TO_COMMAND.values() for e in EMB_TAGS]


def _getb(x):
    while isinstance(x, np.ndarray):
        x = x.item()
    return bytes(x)


def _decode(jpeg_bytes, size: int = None) -> np.ndarray:
    img = Image.open(io.BytesIO(_getb(jpeg_bytes))).convert("RGB")
    if size is not None and img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.array(img)


class EgoVerseRetDataset(Dataset):
    def __init__(
        self,
        manifest_dir: str = DEFAULT_MANIFEST,
        t5_text_embeddings_path: str = DEFAULT_T5_EMB,
        action_stats_path: str = DEFAULT_ACT_STATS,
        num_frames: int = 152,          # 5 s ~ full; 28 for 1 s coarse. div by 4.
        final_image_size: int = 224,
        normalize_images: bool = False,   # keep uint8; model normalizes (matches realrobot)
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        retrieval_dropout_prob: float = 0.0,
        epoch_length: Optional[int] = None,
        debug: bool = False,
        pairs_name: str = "pairs.npz",   # e.g. "pairs_clean.npz" (corrupt-ep-filtered)
        # DATA-SCALING knob: path to a query-episode-set JSON (see
        # build_egoverse_query_episode_sets.py). Keeps only the rows whose QUERY episode is
        # listed; None = the whole manifest. The set is PINNED ON DISK rather than derived
        # from a seed here, so which episodes an arm trained on stays auditable after the
        # fact and can be inspected or hand-edited before launching.
        # Episode-level, not row-level, on purpose — pairs.npz enumerates ~1 s-strided
        # windows (v10 median 74 rows/episode), so a random fraction of ROWS would still
        # touch nearly every episode and would not reduce the data at all. Retrieval is
        # untouched: match_idx keeps pointing at the full demo pool, so the axis is purely
        # query-side data volume.
        query_episode_set: str | None = None,
        # The experiment config inherits from cosmos_predict2p5_2b_480p_libero, whose
        # LIBERODataset node hydra-merges its kwargs onto ours (chunk_size, use_proprio,
        # num_duplicates_per_image, gamma, rollout_data_dir, ...). They don't apply to
        # this loader (action chunk == num_frames; no proprio; fixed 4 dup/latent), so
        # absorb and ignore them — same contract realrobot's dataset relies on.
        **_ignored,
    ):
        super().__init__()
        assert num_frames % NUM_DUPLICATES == 0, "num_frames must be divisible by 4"
        self.manifest_dir = manifest_dir
        self.num_frames = int(num_frames)
        self.R = self.num_frames // NUM_DUPLICATES
        self.final_image_size = final_image_size
        self.normalize_images = normalize_images
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.retrieval_dropout_prob = retrieval_dropout_prob
        self.debug = debug

        W = np.load(os.path.join(manifest_dir, "windows.npz"), allow_pickle=True)
        self.wep = W["ep"].astype(str)
        self.ww0 = W["w0"].astype(np.int64)
        # native window length per row (eg=150 @30Hz, md=100 @20Hz → both 5 s). Used so the
        # frame span matches real time across embodiments (fallback 150 if w1 absent).
        self.ww1 = W["w1"].astype(np.int64) if "w1" in W.files else (self.ww0 + 150)
        self.wisev = W["isev"].astype(bool)
        self.wemb = W["emb"].astype(str)
        self.wcls = W["cls"].astype(str)
        P = np.load(os.path.join(manifest_dir, pairs_name), allow_pickle=True)
        self.query_idx = P["query_idx"].astype(np.int64)
        self.match_idx = P["match_idx"].astype(np.int64)   # (N, K) padded -1
        if query_episode_set is not None:
            keep = self._query_episode_mask(query_episode_set)
            self.query_idx = self.query_idx[keep]
            self.match_idx = self.match_idx[keep]
        self.N = len(self.query_idx)
        self.epoch_length = epoch_length or self.N

        with open(t5_text_embeddings_path, "rb") as fp:
            self.t5_text_embeddings = pickle.load(fp)

        self._zcache: dict[str, zarr.Group] = {}
        self.act_stats = self._load_or_compute_action_stats(action_stats_path)
        print(f"[EgoVerseRetDataset] pairs={self.N} num_frames={self.num_frames} "
              f"R={self.R} chunk_duration={2 * self.R + 4}")

    def _query_episode_mask(self, path: str) -> np.ndarray:
        """Row mask keeping only rows whose QUERY episode is in the set file at `path`.

        The file is the source of truth (JSON: {"episodes": [...], ...}), so every rank and
        dataloader worker selects the identical rows with no seed or broadcast involved.
        Episodes listed but absent from this manifest are an error, not a silent no-op —
        that means the set was built against a different pairs dir.
        """
        with open(path) as fp:
            spec = json.load(fp)
        want = set(spec["episodes"])
        qep = self.wep[self.query_idx]
        have = set(np.unique(qep).tolist())
        missing = want - have
        assert not missing, (
            f"{path}: {len(missing)} episodes are not query episodes of {self.manifest_dir} "
            f"(e.g. {sorted(missing)[:3]}) — the set was built for a different manifest")
        mask = np.isin(qep, np.asarray(sorted(want)))
        print(f"[EgoVerseRetDataset] query_episode_set={os.path.basename(path)}: "
              f"kept {len(want)}/{len(have)} episodes, {int(mask.sum())}/{len(mask)} rows")
        return mask

    # ---------- zarr access (fork-safe: reopened per worker) ----------
    def __getstate__(self):
        s = self.__dict__.copy(); s["_zcache"] = {}; return s

    def __setstate__(self, s):
        self.__dict__.update(s); self._zcache = {}

    def _grp(self, wi: int) -> EgoZarrEpisode:
        # embodiment 'he' (Humanoid-Everyday) routes to HE_ROOT; otherwise the isev
        # flag picks EVA_ROOT vs HUM_ROOT (unchanged).
        if str(self.wemb[wi]) == "he":
            root, tag = HE_ROOT, "he|"
        elif self.wisev[wi]:
            root, tag = EVA_ROOT, "E|"
        else:
            root, tag = HUM_ROOT, "H|"
        key = tag + self.wep[wi]
        g = self._zcache.get(key)
        if g is None:
            g = EgoZarrEpisode(os.path.join(root, self.wep[wi]))
            self._zcache[key] = g
        return g

    # ---------- frame indices for a window ----------
    def _frame_idx(self, wi: int, n: int) -> np.ndarray:
        """num_frames absolute indices sampled evenly across the window's NATIVE 5 s span
        (eg=150 frames @30Hz, md=100 frames @20Hz → both 5 s of real time)."""
        w0 = int(self.ww0[wi]); span = int(self.ww1[wi] - self.ww0[wi])   # 150 (eg) or 100 (md)
        fr = (w0 + np.linspace(0, span - 1, self.num_frames)).round().astype(np.int64)
        return np.clip(fr, 0, n - 1)

    # ---------- action57 + mask ----------
    def _action57(self, wi: int, fr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (num_frames, 57) action chunk + (57,) mask for absolute frame ids `fr`.
        Human is native action57 (all dims valid). eva (action20) is scattered into the
        eef slots WITHOUT grippers (57 has no grip dims) — v9 never references eva."""
        g = self._grp(wi)
        out = np.zeros((self.num_frames, ACTION_DIM), np.float32)
        mask = np.zeros(ACTION_DIM, np.float32)
        if self.wisev[wi]:
            a = g.action("action20")                        # (n-1, 20): Leef9|Lg|Reef9|Rg
            ch = a[np.clip(fr, 0, len(a) - 1)]
            out[:, SL_LEEF] = ch[:, 0:9]; out[:, SL_REEF] = ch[:, 10:19]   # grips dropped
            for s in (SL_LEEF, SL_REEF):
                mask[s] = 1.0
        else:
            a = g.action("action57")                        # (n-1,57): head9|Leef9|Lh15|Reef9|Rh15
            out[:] = a[np.clip(fr, 0, len(a) - 1)]           # native 57, all dims valid
            mask[:] = 1.0
        return out, mask

    # ---------- frames ----------
    def _video(self, wi: int) -> tuple[np.ndarray, np.ndarray]:
        g = self._grp(wi); n = g.n_frames
        fr = self._frame_idx(wi, n)
        frames = []
        for j in fr:
            b = g.frame(int(j))
            frames.append(_decode(b, self.final_image_size) if b else None)   # None = fill slot
        valid = next((f for f in frames if f is not None), None)
        if valid is None:                                   # whole window empty (shouldn't happen)
            valid = np.zeros((self.final_image_size, self.final_image_size, 3), np.uint8)
        last = valid
        for k, f in enumerate(frames):                      # forward-fill empties with nearest prior
            if f is None:
                frames[k] = last
            else:
                last = f
        vid = np.stack(frames, axis=0)                      # (num_frames,H,W,3)
        return vid, fr

    # ---------- normalization ----------
    def _normalize(self, a: np.ndarray, emb: str) -> np.ndarray:
        s = self.act_stats.get(str(emb), self.act_stats["_global"])   # PER-EMBODIMENT min-max
        lo, hi = s["lo"], s["hi"]
        return np.clip(2.0 * (a - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0).astype(np.float32)

    def _load_or_compute_action_stats(self, path: str) -> dict:
        """PER-EMBODIMENT min-max stats: {emb: {"lo","hi"}, "_global": {...}}. Each embodiment
        (aria/scale/MECKA/mimicdroid/eva) is normalized in its OWN action range — different rigs
        / motion scales must not share one range (mirrors cosmos-framework's per-emb normalization)."""
        if os.path.exists(path):
            d = json.load(open(path))
            if "lo" in d and "hi" in d:                       # OLD flat (global) file -> treat as _global
                return {"_global": {"lo": np.asarray(d["lo"], np.float32), "hi": np.asarray(d["hi"], np.float32)}}
            return {k: {"lo": np.asarray(v["lo"], np.float32), "hi": np.asarray(v["hi"], np.float32)}
                    for k, v in d.items()}
        print("[EgoVerseRetDataset] computing PER-EMB action57 stats ...")
        rng = np.random.default_rng(0)
        sel = rng.choice(self.N, min(6000, self.N), replace=False)
        by_emb: dict = {}
        for r in sel:
            wi = int(self.query_idx[r]); g = self._grp(wi)
            n = g.action_rows("action20" if self.wisev[wi] else "action57") + 1
            fr = self._frame_idx(wi, n)
            a, _ = self._action57(wi, fr)
            by_emb.setdefault(str(self.wemb[wi]), []).append(a)

        def _mmax(A):
            lo = np.percentile(A, 1, axis=0); hi = np.percentile(A, 99, axis=0)
            return lo, np.where(hi - lo < 1e-4, lo + 1.0, hi)
        stats = {}
        for emb, chunks in by_emb.items():
            lo, hi = _mmax(np.concatenate(chunks, 0))
            stats[emb] = {"lo": lo.astype(np.float32), "hi": hi.astype(np.float32)}
        glo, ghi = _mmax(np.concatenate([c for cs in by_emb.values() for c in cs], 0))
        stats["_global"] = {"lo": glo.astype(np.float32), "hi": ghi.astype(np.float32)}   # fallback
        json.dump({k: {"lo": v["lo"].tolist(), "hi": v["hi"].tolist()} for k, v in stats.items()},
                  open(path, "w"))
        self._zcache = {}
        return stats

    def __len__(self) -> int:
        return self.epoch_length

    def eval_demos(self, r: int, topk: int = 3) -> tuple[int, list[tuple[int, str]]]:
        """DETERMINISTIC eval demo set for query row r = EVERY stored matching pair (all valid
        matches in match_idx[r]), each labeled same/cross-embodiment. Matches are stored
        best-first, so `topk>0` restricts to same-top-k + cross-top-k; `topk=0` (default) =
        ALL matches. Returns (q, [(demo_window, kind)]) — the batch eval scores the query
        against each and averages (vs the random single-demo the training __getitem__ draws)."""
        q = int(self.query_idx[r])
        emb = self.wemb[q]
        ms = [int(m) for m in self.match_idx[r] if m >= 0]
        same = [m for m in ms if self.wemb[m] == emb]
        cross = [m for m in ms if self.wemb[m] != emb]
        if topk > 0:
            same, cross = same[:topk], cross[:topk]
        demos = [(m, "same") for m in same] + [(m, "cross") for m in cross]
        return q, (demos or [(q, "self")])

    def __getitem__(self, idx: int) -> dict:
        if self.debug:
            idx = 0
        r = idx % self.N
        q = int(self.query_idx[r])
        # training: sample ONE retrieved neighbour at random (variety/augmentation)
        ms = self.match_idx[r]; ms = ms[ms >= 0]
        m = q if len(ms) == 0 else int(ms[np.random.randint(len(ms))])
        return self.build_sample(q, m, idx)

    def build_sample(self, q: int, m: int, idx: int = 0) -> dict:
        """Assemble the full data_batch sample for an explicit (query window q, demo window m).
        __getitem__ picks m randomly; the eval passes each top-k demo explicitly (eval_demos)."""
        # ── frames + actions ──
        q_video, q_fr = self._video(q)                            # (num_frames,H,W,3)
        q_action, q_mask = self._action57(q, q_fr)
        m_video, m_fr = self._video(m)
        m_action, m_mask = self._action57(m, m_fr)

        # normalize actions — EACH item by its OWN embodiment's stats (query vs retrieved-demo may differ)
        q_action = self._normalize(q_action, self.wemb[q]) * q_mask[None, :]
        m_action = self._normalize(m_action, self.wemb[m]) * m_mask[None, :]

        # retrieval dropout
        has_ret = 1
        if self.retrieval_dropout_prob > 0 and np.random.random() < self.retrieval_dropout_prob:
            m_video = np.zeros_like(m_video); m_action = np.zeros_like(m_action); has_ret = 0

        # ── latent image sequence ──
        # Causal temporal tokenizer: latent 0 = frame 0, every later latent = 4 frames.
        # => pixel frames = 1 + 4*(num_latents-1) = 8R+13; latents = 2R+4.
        blank1 = np.zeros_like(q_video[:1])                       # 1 frame  -> latent 0
        blank_dup = np.repeat(blank1, NUM_DUPLICATES, axis=0)     # 4 frames -> 1 latent
        cur = 0; image_list = []

        image_list.append(blank1); cur += 1                       # latent 0: blank
        image_list.append(m_video); ret_vid_start = cur; cur += self.R   # 1..R: ret video (cond)
        ret_vid_end = cur
        image_list.append(blank_dup.copy()); ret_act_idx = cur; cur += 1  # R+1: ret action (injected)
        image_list.append(np.repeat(q_video[:1], NUM_DUPLICATES, axis=0)); q_img_idx = cur; cur += 1  # R+2: query first frame (cond)
        image_list.append(q_video.copy()); q_vid_start = cur; cur += self.R  # R+3..2R+2: query video (GEN)
        q_vid_end = cur
        image_list.append(blank_dup.copy()); q_act_idx = cur; cur += 1  # 2R+3: query action (GEN)

        images = np.concatenate(image_list, axis=0)               # (8R+13, H, W, 3)
        images = preprocess_image(
            images, final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug, stronger_image_aug=self.use_stronger_image_aug,
        )

        # text command + embodiment tag
        cls = self.wcls[q]
        cmd = f"{CLASS_TO_COMMAND.get(cls, CLASS_TO_COMMAND['other'])} [{self.wemb[q]}]"
        t5 = torch.squeeze(self.t5_text_embeddings[cmd])

        return {
            "video": images,
            "actions": q_action,                       # (num_frames, 57) target (masked)
            "action_mask": q_mask.astype(np.float32),  # (57,) valid dims for query embodiment
            "retrieved_actions": m_action,             # (num_frames, 57) cond (masked to ret emb)
            "retrieved_action_mask": m_mask.astype(np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "__key__": idx,
            "embodiment": self.wemb[q],
            "retrieved_embodiment": self.wemb[m],
            # latent indices (single-camera, no proprio/state)
            "retrieved_video_start_latent_idx": ret_vid_start,
            "retrieved_video_end_latent_idx": ret_vid_end,
            "retrieved_action_latent_idx": ret_act_idx,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": q_img_idx,
            "future_video_start_latent_idx": q_vid_start,
            "future_video_end_latent_idx": q_vid_end,
            "action_latent_idx": q_act_idx,
            # proprio unused; dummy tensors so the base model's pass-through keys resolve
            "proprio": np.zeros((self.num_frames, 1), np.float32),
            "future_proprio": np.zeros((self.num_frames, 1), np.float32),
            # unused single-cam / proprio / value slots (-1 disables special handling)
            "current_proprio_latent_idx": -1,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            # First GENERATED query-video latent. Used ONLY by the diagnostic
            # future_image_l1/mse metric (model rectified_flow L479) so the dashboard
            # shows a real video-prediction signal; the training/sampling paths that
            # read this idx are all gated behind mask_*_prediction flags (False) +
            # rollout/value masks (0), so it has ZERO effect on the loss.
            "future_image_latent_idx": q_vid_start, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": has_ret, "has_ret_image": has_ret, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": q_action.copy(),
        }


if __name__ == "__main__":
    import sys
    nf = int(sys.argv[1]) if len(sys.argv) > 1 else 28
    # stub t5 if missing (self-test only)
    if not os.path.exists(DEFAULT_T5_EMB):
        stub = {c: torch.zeros(512, 1024) for c in _commands_with_tags()}
        pickle.dump(stub, open("/tmp/_t5_stub.pkl", "wb"))
        t5p = "/tmp/_t5_stub.pkl"
    else:
        t5p = DEFAULT_T5_EMB
    ds = EgoVerseRetDataset(num_frames=nf, t5_text_embeddings_path=t5p,
                            action_stats_path="/tmp/_act59_stats.json",
                            use_image_aug=False, use_stronger_image_aug=False)
    s = ds[0]
    print("video", s["video"].shape, s["video"].dtype)
    print("actions", s["actions"].shape, "mask sum", s["action_mask"].sum())
    print("retrieved_actions", s["retrieved_actions"].shape, "ret mask sum", s["retrieved_action_mask"].sum())
    print("emb", s["embodiment"], "ret_emb", s["retrieved_embodiment"])
    print("latent idx: ret_vid", s["retrieved_video_start_latent_idx"], s["retrieved_video_end_latent_idx"],
          "ret_act", s["retrieved_action_latent_idx"], "q_img", s["current_image_latent_idx"],
          "q_vid", s["future_video_start_latent_idx"], s["future_video_end_latent_idx"], "q_act", s["action_latent_idx"])
    print("t5", s["t5_text_embeddings"].shape)
