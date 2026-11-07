# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""RoboMIND offline eval in the v3 full-frame STATE-chunk layout.

Third held-out embodiment set next to EgoDex and HE, so the SAME batch eval
(`scripts/run_batch_8gpu.sh` -> `eval_egoverse_ret_batch.py`) produces the same metrics
(MSEN / acc@tau / dtws / FID / FVD / LPIPS) on RoboMIND data.

This is the class form of what `robomind_invariance/tools/rm_feats.py` did by patching an
EgoVerse dataset at runtime: the manifest (`manifest_mid*/{windows,pairs}.npz`, built by
`tools/rm_pairs.py` from the DTW matches) is already in the EgoVerse schema, so only the
*accessors* differ — frames come from RoboMIND h5/mp4 episodes instead of zarr, and state57
rows from `rm_state57*.pkl` instead of the manifest store.

  * src_hz           : RoboMIND is recorded at 10 Hz (the LeRobot fps=30 is only metadata:
                       at 30 Hz the franka / UR grippers would close faster than the hardware
                       can). Every episode is resampled to 30 Hz on load — states linearly
                       interpolated, frames nearest — and ALL indices (ww0/ww1, _grp().state,
                       n_frames) live on that 30 Hz grid, so the model gets real 3.07 s clips
                       at the DT it was trained with (4/30 per latent).
  * frames="fixed"   : 92 CONSECUTIVE (30 Hz) frames from w0 = a real 3.07 s clip. DEFAULT.
    frames="resample": the rm_feats rule — the whole motion span squeezed into 92 frames
                       (median 1.4x speed-up). Keep only to reproduce the feature analysis.
  * t5_mode="raw"    : the episode's own sentence ("{lang} [{emb}]", t5_embeddings_raw*.pkl).
                       REQUIRED for the v2 manifest, whose extra classes have no entry in
                       t5_commands.json — "task" mode would silently hand them a ZERO T5.
    t5_mode="task"   : "{task command} [{emb}]" (v1 manifest).
  * gello's stored JPEGs have swapped channels; they are corrected on read (same rule and
    same set as rm_feats.BGR_STORED).

State normalization uses the per-embodiment robust stats inside `rm_state57*.pkl` (franka /
ur / gello), exactly like the feature runs.
"""
from __future__ import annotations

import io
import json
import os
import pickle
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from cosmos_policy.datasets.egoverse_ret_v3_state_dataset import EgoVerseRetV3StateDataset

RM_ROOT = "/mnt/ddn/tmp/robomind_invariance"
RM_TAG = "_v2"
# Balanced multi-anchor window set:
#   tools/rm_pairs_multi.py --stride 30 --max-anchors 12 --balance 2049
# 2049 windows from v2's 700 episodes, 683 per embodiment (franka / ur episodes are shorter,
# so they get more anchors each), every (episode, w0) distinct, all 700 episodes represented.
# That covers the n=2048 the EgoDex / HE evals use. Alternatives, selectable via manifest_dir:
#   manifest_mid_v2      616 windows, one anchor per episode (the original rm_pairs.py set)
#   manifest_mid_v2_x4  2112 windows, stride 46 but embodiment-skewed (gello 1000 / franka 544)
RM_MANIFEST = f"{RM_ROOT}/manifest_mid{RM_TAG}_bal"
# action57 stats are only read by the DELTA path (delta_target); the state eval never uses
# them. Point at the egoverse file so the base ctor does not try to COMPUTE stats from a
# store RoboMIND does not have.
RM_ACT_STATS = "/mnt/ddn/dataset/egoverse_pairs_he/action57_stats.json"
RM_T5_RAW = f"{RM_ROOT}/t5_embeddings_raw{RM_TAG}.pkl"     # per-episode sentence (t5_mode="raw")
RM_T5_TASK = f"{RM_ROOT}/t5_embeddings.pkl"                # "{task command} [{emb}]" (t5_mode="task")
NF = 92                       # native window length (matches the EgoVerse/EgoDex windows)
DT = 4.0 / 30.0               # seconds per latent on the resampled 30 Hz grid
BGR_STORED = {"gello"}        # TienKung camera_top JPEGs are stored channel-swapped
MODEL_HZ = 30.0               # rate the model's DT assumes
# Recording rate per embodiment. franka / ur verified from gripper closing speed (at 30 Hz the
# full close would take 0.1-0.3 s, faster than Franka Hand / Robotiq allow); gello assumed.
SRC_HZ = {"franka": 10.0, "ur": 10.0, "gello": 10.0}


def _upsample(s: np.ndarray, ratio: float) -> np.ndarray:
    """Linearly interpolate a (T, D) state sequence onto a grid `ratio` times denser."""
    s = np.asarray(s, np.float32)
    if ratio == 1.0 or len(s) < 2:
        return s
    t = np.arange(int(round((len(s) - 1) * ratio)) + 1) / ratio
    i0 = np.minimum(np.floor(t).astype(np.int64), len(s) - 2)
    w = (t - i0)[:, None].astype(np.float32)
    return (1 - w) * s[i0] + w * s[i0 + 1]


def _load_frames_subset(r: dict, idxs: list[int], size: int) -> dict[int, np.ndarray]:
    """Recorded frames `idxs` of an h5 episode, resized to (size, size) — the per-frame rule of
    _load_all_frames + _frames_of_episode (RGB, BGR_STORED swap, bilinear resize)."""
    import h5py
    out = {}
    with h5py.File(r["src"]["path"]) as h:
        ds = h["observations/rgb_images/camera_top"]
        for i in idxs:
            f = np.array(Image.open(io.BytesIO(ds[i].tobytes())).convert("RGB"))
            if r["emb"] in BGR_STORED:
                f = np.ascontiguousarray(f[:, :, ::-1])
            out[i] = np.array(Image.fromarray(f).resize((size, size), Image.BILINEAR))
    return out


SIDE_CAMS = (("camera_left", (0, 1)), ("camera_right", (1, 0)))      # (camera, (row, col) quadrant)


def _grid_frames(r: dict, S: int) -> list[np.ndarray]:
    """franka_3rgb episode as 2x2 grid frames (EgoVerseRetV2Dataset._grid_frame layout).
    Side-camera mp4s share the top camera's chunk/file/from_timestamp for every episode
    (checked on all 11,718 franka_3rgb episodes), so only the camera name changes."""
    h = S // 2
    small = lambda f: np.array(Image.fromarray(f).resize((h, h), Image.BILINEAR))     # noqa: E731
    top = _load_all_frames(r)
    sides = {}
    for cam, _ in SIDE_CAMS:
        path = r["src"]["path"].replace("observation.images.camera_top", f"observation.images.{cam}")
        sides[cam] = _load_all_frames({**r, "src": {**r["src"], "path": path}})
    out = []
    for i, f in enumerate(top):
        c = np.zeros((S, S, 3), np.uint8)
        c[:h, :h] = small(f)
        for cam, (ri, ci) in SIDE_CAMS:
            sf = sides[cam]
            if sf:
                c[ri * h:(ri + 1) * h, ci * h:(ci + 1) * h] = small(sf[min(i, len(sf) - 1)])
        out.append(c)
    return out


def _load_all_frames(r: dict) -> list[np.ndarray]:
    """Every RGB frame of one RoboMIND episode (rm_feats.load_all_frames, verbatim rule)."""
    s = r["src"]
    if s["kind"] == "h5":
        import h5py
        with h5py.File(s["path"]) as h:
            ds = h["observations/rgb_images/camera_top"]
            fr = [np.array(Image.open(io.BytesIO(ds[i].tobytes())).convert("RGB")) for i in range(len(ds))]
        if r["emb"] in BGR_STORED:
            fr = [np.ascontiguousarray(f[:, :, ::-1]) for f in fr]
        return fr
    import av
    out: list[np.ndarray] = []
    with av.open(s["path"]) as c:
        st = c.streams.video[0]
        c.seek(int(s["t0"] / st.time_base), stream=st, backward=True)
        for fr in c.decode(st):
            if float(fr.pts * st.time_base) < s["t0"] - 0.5 / s["fps"]:
                continue
            out.append(fr.to_ndarray(format="rgb24"))
            if len(out) >= r["n"]:
                break
    return out


class _RMEpisode:
    """EgoZarrEpisode stand-in for one RoboMIND episode.

    The parent classes AND the batch eval reach episodes through `dataset._grp(wi)` — e.g.
    eval_egoverse_ret_batch.py:543 takes `_grp(q).state("state57")[w0]` as the FK state0.
    Overriding only the frame/state helpers left that path pointing at EgoVerse zarr stores
    ("egoverse_eva/franka/bread_on_table/000#a0/images.front_1/zarr.json" -> FileNotFoundError,
    every window skipped), so the adapter is the thing to hand out instead.
    """

    def __init__(self, ds: "RoboMindRetV3StateDataset", wi: int):
        self._ds, self._e = ds, int(ds._epi[wi])
        self.n_frames = len(ds._states[self._e])          # 30 Hz grid

    def state(self, name: str = "state57") -> np.ndarray:
        assert name == "state57", f"RoboMIND has no state group {name!r}"
        return np.asarray(self._ds._states[self._e], np.float32)

    def action(self, name: str) -> np.ndarray:
        raise KeyError(
            f"RoboMIND stores no recorded action group ({name!r}); the state eval derives "
            "action57 from the state57 chunks")

    def frame(self, i: int, cam: str = "images.front_1") -> np.ndarray:
        """Decoded RGB frame (the callers that need pixels go through _grid_clip instead)."""
        f = self._ds._frames_of_episode(self._e)
        return f[self._ds._real_frame(self._e, i, len(f))]


class RoboMindRetV3StateDataset(EgoVerseRetV3StateDataset):
    """RoboMIND eval dataset — EgoVerse v3 state layout, RoboMIND accessors (see module doc)."""

    def __init__(
        self,
        manifest_dir: str = RM_MANIFEST,
        root: str = RM_ROOT,
        tag: str = RM_TAG,
        action_stats_path: str = RM_ACT_STATS,
        # MUST stay an explicit parameter: LazyCall bakes the signature defaults into the
        # experiment config, and anything NOT in the signature is inherited from the parent
        # experiment instead — the libero default leaked in and collided with the value this
        # ctor passed to super() ("got multiple values for keyword argument").
        t5_text_embeddings_path: str = RM_T5_RAW,
        t5_mode: str = "raw",
        frames: str = "fixed",
        cache_episodes: int = 3,
        num_frames: int = NF,
        final_image_size: int = 448,
        use_image_aug: bool = False,          # eval: no aug
        use_stronger_image_aug: bool = False,
        retrieval_dropout_prob: float = 0.0,
        src_hz: dict | None = None,          # per-emb recording rate; None -> SRC_HZ
        # 2x2 grid for episodes that HAVE side views (only franka_3rgb: top/left/right; UR and
        # TienKung are 1rgb in every RoboMIND release) — the EgoVerseRetV2Dataset._grid_frame rule:
        # TL = top, TR = left, BL = right, BR = black, each quadrant final_image_size // 2.
        # Single-view episodes stay full-frame. Default off = the v3 full-frame layout.
        grid_multicam: bool = False,
        **kw,
    ):
        assert t5_mode in ("raw", "task") and frames in ("fixed", "resample")
        self.rm_root, self.rm_tag = root, tag
        t5_path = t5_text_embeddings_path
        if t5_mode == "task" and t5_path == RM_T5_RAW:      # follow t5_mode unless overridden
            t5_path = f"{root}/t5_embeddings.pkl"
        assert "robomind" in t5_path, (
            f"t5_text_embeddings_path={t5_path!r} is not a RoboMIND pickle — the parent "
            "experiment's value leaked in; pass it explicitly or fix the signature default")
        super().__init__(
            manifest_dir=manifest_dir,
            t5_text_embeddings_path=t5_path,
            action_stats_path=action_stats_path,
            num_frames=num_frames,
            final_image_size=final_image_size,
            use_image_aug=use_image_aug,
            use_stronger_image_aug=use_stronger_image_aug,
            retrieval_dropout_prob=retrieval_dropout_prob,
            robust_state_stats=False,          # stats come from rm_state57*.pkl below
            **kw,
        )
        self.t5_mode, self.frames_mode = t5_mode, frames
        self.grid_multicam = bool(grid_multicam)
        self._eps = pickle.load(open(f"{root}/episodes{tag}.pkl", "rb"))
        ST = pickle.load(open(f"{root}/rm_state57{tag}.pkl", "rb"))
        hz = {**SRC_HZ, **(src_hz or {})}
        # 30 Hz grid: ratio per episode, states upsampled, window anchors rescaled
        self._ratio = np.array([MODEL_HZ / hz[e["emb"]] for e in self._eps], np.float64)
        self._states = [_upsample(s, r) for s, r in zip(ST["states"], self._ratio)]
        stats = {k.replace("rm_", ""): v for k, v in ST["stats"].items()}      # franka / ur / gello
        self.state_stats = {**stats, "_global": next(iter(stats.values()))}
        W = np.load(os.path.join(manifest_dir, "windows.npz"), allow_pickle=True)
        self._epi = W["episode_id"].astype(np.int64)
        rw = self._ratio[self._epi]
        self.ww0 = np.round(self.ww0 * rw).astype(np.int64)
        self.ww1 = np.round(self.ww1 * rw).astype(np.int64)
        if frames == "fixed":
            # The parent takes the PAST state chunk at fr_q - (w1 - w0) + 1, which ends at "now"
            # only when fr_q covers exactly w1 - w0 frames (EgoVerse: span 90 -> 92 frames).
            # The manifest's w1 - w0 is the motion half-length (34..914 frames), so with 92
            # consecutive frames the past chunk ended 7-30 cm (median) away from the current
            # state, or ran into the future. Fixed mode reads 92 frames -> span must be 92.
            self.ww1 = self.ww0 + num_frames
        self._cmd = json.load(open(f"{root}/t5_commands.json"))["task_command"]
        self._cache: OrderedDict[int, list[np.ndarray]] = OrderedDict()
        self._cache_n = int(cache_episodes)
        print(f"[RoboMind] windows={len(self._epi)} eps={len(self._eps)} frames={frames} "
              f"t5={t5_mode} emb={sorted(set(self.wemb.tolist()))} src_hz={hz}", flush=True)

    # ---------- fork-safe: the decoded-frame cache never crosses a worker boundary ----------
    def __getstate__(self):
        s = super().__getstate__() if hasattr(super(), "__getstate__") else self.__dict__.copy()
        s = dict(s); s["_cache"] = OrderedDict()
        return s

    def __setstate__(self, s):
        self.__dict__.update(s)
        self._cache = OrderedDict()
        self._zcache = {}

    # ---------- RoboMIND accessors (replace the zarr ones) ----------
    def _grp(self, wi: int) -> _RMEpisode:
        """RoboMIND episode adapter — replaces the parent's EgoZarrEpisode lookup."""
        return _RMEpisode(self, wi)

    def _frames_of(self, wi: int) -> list[np.ndarray]:
        return self._frames_of_episode(int(self._epi[wi]))

    def _frames_of_episode(self, e: int) -> list[np.ndarray]:
        if e not in self._cache:
            S = self.final_image_size
            r = self._eps[e]
            if self.grid_multicam and r["src"]["kind"] == "mp4":
                fr = _grid_frames(r, S)
            else:
                fr = [np.array(Image.fromarray(f).resize((S, S), Image.BILINEAR))
                      for f in _load_all_frames(r)]
            self._cache[e] = fr
            while len(self._cache) > self._cache_n:
                self._cache.popitem(last=False)
        self._cache.move_to_end(e)
        return self._cache[e]

    def _native(self, wi: int):
        return NF, DT

    def _frames_native(self, wi: int, nf: int) -> np.ndarray:
        n = len(self._states[int(self._epi[wi])])       # 30 Hz grid
        w0 = int(self.ww0[wi])
        if self.frames_mode == "fixed":                 # real 3.07 s at 30 Hz
            fr = w0 + np.arange(nf, dtype=np.int64)
        else:                                           # rm_feats rule: span -> nf frames
            span = int(self.ww1[wi] - self.ww0[wi])
            fr = (w0 + np.linspace(0, span - 1, nf)).round().astype(np.int64)
            return np.clip(fr, 0, n - 1)
        # NOT clipped at n-1: the parent derives the past chunk as fr - span + 1, so clipping
        # first froze the tail of the past chunk at (n-1) - span + 1 whenever w0 + 92 > n
        # (short franka/ur episodes). _grid_clip / _state_rows clip on read (= hold last frame).
        return np.maximum(fr, 0)

    def _real_frame(self, e: int, i: int, n_real: int) -> int:
        """30 Hz grid index -> nearest recorded frame of episode e."""
        return min(max(int(round(int(i) / self._ratio[e])), 0), n_real - 1)

    def _grid_clip(self, wi: int, fr: np.ndarray) -> np.ndarray:
        e = int(self._epi[wi])
        r = self._eps[e]
        if r["src"]["kind"] == "h5" and e not in self._cache:
            # Training reads random windows: decoding the whole h5 episode (gello ~580 JPEGs)
            # per sample is the bottleneck. JPEGs decode independently -> read only the ~31
            # recorded frames this 92-frame window touches. Same decode + resize as the full path.
            n_real = int(r["n"])
            real = [self._real_frame(e, j, n_real) for j in fr]
            got = _load_frames_subset(r, sorted(set(real)), self.final_image_size)
            return np.stack([got[k] for k in real])
        f = self._frames_of_episode(e)
        return np.stack([f[self._real_frame(e, j, len(f))] for j in fr])

    def _state_rows(self, wi: int, idx: np.ndarray) -> np.ndarray:
        s = self._states[int(self._epi[wi])]
        return np.asarray(s[np.clip(idx, 0, len(s) - 1)], np.float32)

    def __getitem__(self, idx: int) -> dict:
        """Eval-only dataset: the parent's __getitem__ hard-codes aug=True (training crop + strong
        color jitter), which viz_v2 (dataset[idx]) then fed to the model. Follow use_image_aug."""
        r = int(idx) % self.N
        return self.build_sample(int(self.query_idx[r]), self._choose_demo(r), idx=idx,
                                 aug=bool(self.use_image_aug))

    # ---------- text conditioning ----------
    def build_sample(self, q: int, m: int, idx: int = 0, *args, **kw) -> dict:
        sample = super().build_sample(q, m, idx, *args, **kw)
        key = self._t5_key(q)
        emb = self.t5_text_embeddings.get(key)
        assert emb is not None, f"T5 key missing ({self.t5_mode}): {key!r}"
        sample["t5_text_embeddings"] = torch.squeeze(torch.as_tensor(emb))
        return sample

    def _t5_key(self, wi: int) -> str:
        if self.t5_mode == "raw":
            e = self._eps[int(self._epi[wi])]
            return f"{e['lang']} [{e['emb']}]"
        return f"{self._cmd[str(self.wcls[wi])]} [{self.wemb[wi]}]"
