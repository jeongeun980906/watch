# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Online demo retrieval for GR1 closed-loop evaluation — GRIP-AUGMENTED ABS-TRAJ (v4).

Port of ``81_future_pairs_dth_grip.py`` (the canonical GR1 v4 pair builder, which
extended ``78_future_pairs_abstraj.py`` with a grip channel and a DTH demo pool).
The GR1 policy is a *conditioning* policy: each generated action chunk is conditioned
on a retrieved demo, re-matched online from the robot's PAST window — run live instead
of offline.

Descriptor (identical to 81, so the live match reproduces the trained v4 pairs):

    past proprio window (≤WIN=32 frames of base-rel ee_pose L/R + gripper)
        -> per-frame BOTH-hand ABSOLUTE 6D pose + scaled grip
           [Lpos3, L6D, gL*0.4, Rpos3, R6D, gR*0.4]  (20-d, METERS; L6D = rotation
           cols 0,1; NO z-score, NO start-relative; g in [0,1], GRIP_SCALE=0.4 from
           the dthpool sweep), subsampled to SEQ=24 frames
        -> task-gated mean-pose L2 prefilter vs the DTH pool (SHORTLIST=150)
        -> banded abs-traj DTW rerank
        -> top-1 demo window (ep, w0=future-start boundary)

The grip value: from state57 it is the hand15 inverse lerp (convention doc §3c/§3f,
``g = clip(((hand15-OPEN15)@axis)/(axis@axis), 0, 1)``); from live proprio it is the
commanded gripper in [0,1] directly — IDENTICAL by construction, because state57's
hand15 is written as the OPEN/CLOSED lerp of that same commanded g.

``w0`` is the future-start boundary: the demo clip used to condition is the demo's
FUTURE ``action57[w0 : w0+FUT]`` (+ video), read by the server's reader. Matching on
the PAST while conditioning on the demo's FUTURE is the future-pair setup. The DTH /
taskdemo descriptor POOLS are precomputed at 20-d by ``81_future_pairs_dth_grip.py``
(``dth_pool_1x_v4.npz`` — the SAME windows/descriptors used to build the training
pairs) and ``83_build_taskdemo_pool_v4.py`` (``taskdemo_pool_v4.npz``); this module
only does the live query side. The legacy 18-d play pools (``play_pool_{1x,2x}.npz``)
are NOT compatible with the 20-d query descriptor.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as _R

# 81_future_pairs_dth_grip.py knobs (keep in sync with the pool builders).
WIN = 32           # past-window length (frames @ 20 Hz)
FUT = 32           # future/conditioning chunk span
SEQ = 24           # abs-traj DTW sequence length (subsampled from the past window)
SHORTLIST = 150    # mean-pose L2 shortlist size before the abs-traj DTW rerank
DTW_BAND = 6       # Sakoe-Chiba band
GRIP_SCALE = 0.4   # grip-channel weight (dthpool sweep, S=0.4)
ROT_SCALE = 0.25   # v6: rot6D downweight (position-dominant matching)
VEL_SCALE = 20.0   # v6: per-frame delta-pos weight (1 cm/f -> 0.2)

# hand15 OPEN/CLOSED anchors (action57_state57_convention.md §3c) — the state57 grip
# inverse-lerp (§3f) uses axis = CLOSED15 - OPEN15.
OPEN15 = np.array([0.0005, 0.0958, -0.034, 0.0285, 0.0222, -0.1263, 0.0085, 0.0217, -0.1285,
                   -0.0116, 0.0222, -0.1282, -0.0314, 0.0229, -0.1259], float)
CLOSED15 = np.array([0.0052, 0.0775, -0.0701, 0.0266, 0.044, -0.0916, 0.0085, 0.0441, -0.1016,
                     -0.0097, 0.044, -0.0937, -0.0278, 0.044, -0.0919], float)
_GRIP_AXIS = CLOSED15 - OPEN15


def grip_from_state57(s: np.ndarray, side: str) -> np.ndarray:
    """(T,57) state57 -> (T,) g in [0,1] via the hand15 inverse lerp (§3c/§3f)."""
    h = s[:, 18:33] if side == "L" else s[:, 42:57]
    return np.clip((h - OPEN15) @ _GRIP_AXIS / (_GRIP_AXIS @ _GRIP_AXIS), 0, 1)

# GR1 task -> required play subgoal (type, src, dst); dst is None or a tuple of allowed
# values — IDENTICAL to 78_future_pairs_abstraj.TASK2REQ.
TASK2REQ: dict[str, tuple] = {
    "TurnOnFaucet": ("faucet_on", None, None),
    "CloseLeftCabinetDoor": ("door_close", None, None),
    "PnPSinkToCabinet": ("pick_place", "the sink", ("the cabinet",)),
    "PnPSinkToRightCounterPlate": ("pick_place", "the sink", ("the plate", "the counter")),
}

# state57 layout: head9(0:9) | Leef9(9:18) | Lhand15(18:33) | Reef9(33:42) | Rhand15(42:57)
# Leef9/Reef9 = abs9 = [pos3, R[:,0]3, R[:,1]3]. The abs-traj feature picks the pose part.
_S_SLICES = (slice(9, 12), slice(12, 18), slice(33, 36), slice(36, 42))


def _rmat(quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(T,4) xyzw quats -> ((T,3,3) rot matrices, (T,) validity)."""
    q = np.asarray(quat_xyzw, float)
    nb = np.linalg.norm(q, axis=1)
    bad = (nb < 1e-6) | ~np.isfinite(q).all(1)
    qn = q.copy()
    qn[bad] = [0, 0, 0, 1]
    qn /= np.linalg.norm(qn, axis=1, keepdims=True)
    return _R.from_quat(qn).as_matrix(), ~bad


def _subsample(f: np.ndarray, seq: int = SEQ) -> np.ndarray:
    """linspace-subsample a (T,D) sequence to (seq,D) — 78's index scheme."""
    idx = np.linspace(0, len(f) - 1, seq).round().astype(int)
    return f[idx].astype(np.float32)


def traj_from_state57(s: np.ndarray, e: int, win: int = WIN, seq: int = SEQ) -> np.ndarray:
    """GRIP-AUGMENTED abs both-hand trajectory (seq,20) over the PAST window [e-pl, e].

    ``s`` is (T,57) state57; ``pl = min(win, e)`` (shorter near the start). Identical to
    ``81_future_pairs_dth_grip.frames20``+``traj20``: per frame
    ``[Lpos3, L6D, gL*GRIP_SCALE, Rpos3, R6D, gR*GRIP_SCALE]`` in meters, no z-score;
    g recovered from the hand15 block by the §3c inverse lerp.
    """
    pl = max(min(win, e), 1)
    seg = s[e - pl:e]
    gL = grip_from_state57(seg, "L")[:, None] * GRIP_SCALE
    gR = grip_from_state57(seg, "R")[:, None] * GRIP_SCALE
    # v6 26-d: [pos, rot6D*ROT_SCALE, grip, dpos*VEL_SCALE] per arm — identical to
    # 81c frames20 (vel = frame diff computed BEFORE windowing, first row 0).
    dl = np.vstack([np.zeros((1, 3)), np.diff(s[:, _S_SLICES[0]], axis=0)])[e - pl:e] * VEL_SCALE
    dr = np.vstack([np.zeros((1, 3)), np.diff(s[:, _S_SLICES[2]], axis=0)])[e - pl:e] * VEL_SCALE
    f = np.concatenate([seg[:, _S_SLICES[0]], ROT_SCALE * seg[:, _S_SLICES[1]], gL, dl,
                        seg[:, _S_SLICES[2]], ROT_SCALE * seg[:, _S_SLICES[3]], gR, dr], 1)
    return _subsample(f, seq)


def traj_from_proprio(pL, pR, quatL, quatR, gripL, gripR, seq: int = SEQ) -> np.ndarray:
    """GRIP-AUGMENTED abs both-hand trajectory (seq,20) from a buffered proprio window.

    pL/pR: (T,3) base-relative ee positions; quatL/quatR: (T,4) xyzw; gripL/gripR: (T,)
    commanded gripper in [0,1]. The 6D rotation is columns 0,1 of the rotation matrix —
    byte-identical to state57's abs9 pose — and the live grip equals the state57 hand15
    inverse-lerp BY CONSTRUCTION (hand15 is written as the OPEN/CLOSED lerp of this same
    commanded g), so the live query matches the pool (built from state57) exactly.
    """
    pL = np.asarray(pL, float)
    pR = np.asarray(pR, float)
    RL, _ = _rmat(quatL)
    RR, _ = _rmat(quatR)
    gL = np.clip(np.asarray(gripL, float), 0.0, 1.0)[:, None] * GRIP_SCALE
    gR = np.clip(np.asarray(gripR, float), 0.0, 1.0)[:, None] * GRIP_SCALE
    dl = np.vstack([np.zeros((1, 3)), np.diff(pL, axis=0)]) * VEL_SCALE
    dr = np.vstack([np.zeros((1, 3)), np.diff(pR, axis=0)]) * VEL_SCALE
    f = np.concatenate([pL, ROT_SCALE * RL[:, :, 0], ROT_SCALE * RL[:, :, 1], gL, dl,
                        pR, ROT_SCALE * RR[:, :, 0], ROT_SCALE * RR[:, :, 1], gR, dr], 1)
    return _subsample(f, seq)


def dtw(p: np.ndarray, q: np.ndarray, band: int = DTW_BAND) -> float:
    """Banded DTW between two (n,D) sequences, normalized by path length (78's `dtw`)."""
    Da = np.linalg.norm(np.asarray(p)[:, None] - np.asarray(q)[None], axis=2)
    n, mm = Da.shape
    acc = np.full((n + 1, mm + 1), 1e18)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        jlo, jhi = max(1, i - band), min(mm, i + band)
        pr, row = acc[i - 1], acc[i]
        for j in range(jlo, jhi + 1):
            row[j] = Da[i - 1, j - 1] + min(pr[j], row[j - 1], pr[j - 1])
    return float(acc[n, mm] / (n + mm))


class PlayDemoPool:
    """Precomputed play-demo abs-traj pool + the 78-style gated DTW matcher.

    Pool fields (row-aligned): T (N,SEQ,18) abs-traj, Mpose (N,18) per-window mean pose,
    ep, w0(=boundary), sgtype/src/dst (subgoal), prel (= boundary - subgoal start).
    """

    def __init__(self, pool_path: str | Path, shortlist: int = SHORTLIST) -> None:
        d = np.load(Path(pool_path), allow_pickle=True)
        self.T = np.asarray(d["T"], np.float32)
        self.Mpose = np.asarray(d["Mpose"], np.float32)
        self.ep = d["ep"].astype(str)
        self.w0 = np.asarray(d["w0"], np.int64)
        self.sgtype = d["sgtype"].astype(str)
        self.src = d["src"].astype(str)
        self.dst = d["dst"].astype(str)
        self.prel = np.asarray(d["prel"], np.int64)
        self.shortlist = int(shortlist)
        self._task_pool: dict[str, np.ndarray] = {}

    def _req(self, task: str, i: int) -> bool:
        t, src, dst = TASK2REQ[task]
        if self.sgtype[i] != t:
            return False
        if src is not None and self.src[i] != src:
            return False
        if dst is None:
            return True
        return self.dst[i] in dst  # dst is a tuple of allowed values

    def _pool_for(self, task: str) -> np.ndarray:
        if task not in self._task_pool:
            self._task_pool[task] = (
                np.array([i for i in range(len(self.ep)) if self._req(task, i)], dtype=np.int64)
                if task in TASK2REQ else np.arange(len(self.ep))
            )
        return self._task_pool[task]

    def match(self, qt: np.ndarray, qm: np.ndarray, task: str, past_len: int, topk: int = 1) -> list[dict]:
        """Top demos for a query abs-traj ``qt`` (mean pose ``qm``). ``past_len`` < WIN
        forces the early-in-subgoal play pool (prel ≤ WIN), exactly like 78."""
        pool = self._pool_for(task)
        if pool.size == 0:
            return []
        if past_len < WIN:
            early = pool[self.prel[pool] <= WIN]
            if early.size:
                pool = early
        l2 = np.linalg.norm(self.Mpose[pool] - qm, axis=1)
        cand = pool[np.argsort(l2)[: self.shortlist]]
        dd = np.array([dtw(qt, self.T[c]) for c in cand])
        order = cand[np.argsort(dd)]
        dds = np.sort(dd)
        out: list[dict] = []
        seen: set[str] = set()
        for c, dv in zip(order, dds):
            if self.ep[c] in seen:
                continue
            seen.add(self.ep[c])
            out.append({"ep": str(self.ep[c]), "w0": int(self.w0[c]), "dist": float(dv)})
            if len(out) >= topk:
                break
        return out


class TaskDemoPool:
    """Per-task TaskDemos abs-traj pool — same matcher as PlayDemoPool but gated by the
    TASK name (each TaskDemo is one whole task, so no subgoal type/src/dst gate)."""

    def __init__(self, pool_path: str | Path, shortlist: int = SHORTLIST) -> None:
        d = np.load(Path(pool_path), allow_pickle=True)
        self.T = np.asarray(d["T"], np.float32)
        self.Mpose = np.asarray(d["Mpose"], np.float32)
        self.ep = d["ep"].astype(str)
        self.w0 = np.asarray(d["w0"], np.int64)
        self.task = d["task"].astype(str)
        self.shortlist = int(shortlist)
        self._task_pool: dict[str, np.ndarray] = {}

    def _pool_for(self, task: str) -> np.ndarray:
        if task not in self._task_pool:
            sel = np.where(self.task == task)[0]
            self._task_pool[task] = sel if sel.size else np.arange(len(self.ep))
        return self._task_pool[task]

    def match(self, qt: np.ndarray, qm: np.ndarray, task: str, past_len: int, topk: int = 1) -> list[dict]:
        pool = self._pool_for(task)
        if pool.size == 0:
            return []
        warmup = past_len < WIN
        if warmup:
            # t=0/warmup: training (77_gr1_t0_taskdemo) pairs every t=0 query with a
            # taskdemo OPENING (w0=0); an ungated mean-pose match here picks mid-demo
            # windows the model never sees at t=0 (measured: Cabinet demo_1@140).
            # The pool's earliest stored boundary is 4 (no w0=0 rows), so match on the
            # earliest windows and return w0=0 to reproduce the training convention.
            early = pool[self.w0[pool] <= 4]
            if early.size:
                pool = early
        l2 = np.linalg.norm(self.Mpose[pool] - qm, axis=1)
        cand = pool[np.argsort(l2)[: self.shortlist]]
        dd = np.array([dtw(qt, self.T[c]) for c in cand])
        order = cand[np.argsort(dd)]
        dds = np.sort(dd)
        out: list[dict] = []
        seen: set[str] = set()
        for c, dv in zip(order, dds):
            if self.ep[c] in seen:
                continue
            seen.add(self.ep[c])
            out.append({"ep": str(self.ep[c]), "w0": 0 if warmup else int(self.w0[c]),
                        "dist": float(dv)})
            if len(out) >= topk:
                break
        return out


class DthDemoPool:
    """DTH demo pool (``dth_pool_1x_v4.npz``, 20-d grip-augmented) — the t>0 pool that
    replaced the play pool in the v4 pairs. Task-gated by the DTH episode's own task
    (each ``dthdemo_<Task>_*`` episode is one whole task), matcher identical to the
    training builder ``81_future_pairs_dth_grip``: mean-pose L2 shortlist -> banded DTW
    -> episode-dedup top-k.

    SAME-scene exclusion: training pairs exclude demo windows whose scene_seed equals
    the query episode's. At SERVE time the eval scenes are HELD-OUT seeds (never in the
    DTH pool's non-training-scene selection), so the exclusion is a no-op online —
    ``exclude_seed`` stays None. It is exposed for the offline train<->serve consistency
    gate, which replays TRAINING episodes through this matcher.
    """

    def __init__(self, pool_path: str | Path, shortlist: int = SHORTLIST) -> None:
        d = np.load(Path(pool_path), allow_pickle=True)
        self.T = np.asarray(d["T"], np.float32)
        self.Mpose = np.asarray(d["Mpose"], np.float32)
        self.ep = d["ep"].astype(str)
        self.w0 = np.asarray(d["w0"], np.int64)
        self.task = d["task"].astype(str)
        self.seed = np.asarray(d["seed"], np.int64)
        self.shortlist = int(shortlist)
        self._task_pool: dict[str, np.ndarray] = {}

    def _pool_for(self, task: str) -> np.ndarray:
        if task not in self._task_pool:
            sel = np.where(self.task == task)[0]
            self._task_pool[task] = sel if sel.size else np.arange(len(self.ep))
        return self._task_pool[task]

    def match(self, qt: np.ndarray, qm: np.ndarray, task: str, past_len: int,
              topk: int = 1, exclude_seed: int | None = None) -> list[dict]:
        pool = self._pool_for(task)
        if exclude_seed is not None:
            pool = pool[self.seed[pool] != int(exclude_seed)]
        if pool.size == 0:
            return []
        l2 = np.linalg.norm(self.Mpose[pool] - qm, axis=1)
        cand = pool[np.argsort(l2)[: self.shortlist]]
        dd = np.array([dtw(qt, self.T[c]) for c in cand])
        order = cand[np.argsort(dd)]
        dds = np.sort(dd)
        out: list[dict] = []
        seen: set[str] = set()
        for c, dv in zip(order, dds):
            if self.ep[c] in seen:
                continue
            seen.add(self.ep[c])
            out.append({"ep": str(self.ep[c]), "w0": int(self.w0[c]), "dist": float(dv)})
            if len(out) >= topk:
                break
        return out


_POOL_KINDS = {"play": PlayDemoPool, "taskdemo": TaskDemoPool, "dth": DthDemoPool}


class OnlineRetriever:
    """Server-side helper: a rolling proprio buffer + per-chunk abs-traj demo retrieval.

    The env client streams per-step proprio; on each action chunk the server calls
    :meth:`retrieve` to (re)select the conditioning demo from the recent past window.
    ``kind`` picks the pool: ``"taskdemo"`` (per-task TaskDemos, used at t=0) or
    ``"play"`` (the SinkEnvForPlay subgoal-gated pool, used for t>0).
    """

    def __init__(self, pool_path: str | Path, kind: str = "taskdemo", **_ignored) -> None:
        self.pool = _POOL_KINDS[kind](pool_path)
        # WIN+1: the newest buffered frame is the CURRENT boundary state t — training
        # queries match on the strict past [e-32, e) and feed state[e] only as
        # cur_state, so retrieve() drops the newest frame and uses the previous WIN.
        self._buf: deque = deque(maxlen=WIN + 1)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, pL, pR, quatL, quatR, gripL, gripR) -> None:
        """Append one proprio step. Grip (commanded, [0,1]) feeds the 20-d descriptor's
        grip channel — identical to the state57 hand15 inverse-lerp by construction."""
        self._buf.append((np.asarray(pL, float), np.asarray(pR, float),
                          np.asarray(quatL, float), np.asarray(quatR, float),
                          float(gripL), float(gripR)))

    def query(self) -> tuple[np.ndarray, np.ndarray, int] | None:
        """(qt, qm, past_len) from the buffered STRICT past [t-WIN, t) — the 20-d
        query descriptor, exposed for external matchers (e.g. within-episode one-shot)."""
        if len(self._buf) < 1:
            return None
        buf = list(self._buf)
        if len(buf) >= 2:
            # strict past [t-WIN, t): exclude the current boundary frame (training
            # convention — builder 81's traj20(f, e) slices f[e-pl:e], excluding e).
            buf = buf[:-1][-WIN:]
        pL = np.stack([b[0] for b in buf])
        pR = np.stack([b[1] for b in buf])
        qL = np.stack([b[2] for b in buf])
        qR = np.stack([b[3] for b in buf])
        gL = np.array([b[4] for b in buf])
        gR = np.array([b[5] for b in buf])
        qt = traj_from_proprio(pL, pR, qL, qR, gL, gR)
        return qt, qt.mean(0), len(buf)

    def retrieve(self, task: str, topk: int = 1) -> list[dict]:
        """Match the buffered past window against the pool. Works from a SINGLE frame
        (t=0): a 1-frame buffer subsamples to a constant zero-motion abs-traj, so the
        mean-pose term selects the same-task demo whose opening pose is closest."""
        q = self.query()
        if q is None:
            return []
        qt, qm, past_len = q
        return self.pool.match(qt, qm, task, past_len=past_len, topk=topk)


def episode_windows(state57: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Descriptor windows of ONE episode for within-episode one-shot matching.

    Returns (w0s, T): boundaries w0 ∈ [WIN, act_rows-SPAN] stride 4, SPAN=36 — the EXACT
    grid of the 81/85 pair builders (w0+36 <= rows manifest validity), so one-shot
    matching at eval only picks boundaries that exist in training pairs. (The looser
    FUT=32 cap let the server pick a demo's last 4-frame boundary that training never
    conditioned on — off-by-4 tail mismatch found by the v5 verification suite.)"""
    SPAN = 36
    rows = len(state57) - 1
    w0s = np.arange(WIN, rows - SPAN + 1, 4, dtype=np.int64)
    T = np.stack([traj_from_state57(state57, int(e)) for e in w0s]) if len(w0s) else \
        np.zeros((0, SEQ, 20), np.float32)
    return w0s, T


def match_within(qt: np.ndarray, w0s: np.ndarray, T: np.ndarray) -> int:
    """Best boundary within one episode's windows (banded DTW over all of them)."""
    if len(w0s) == 0:
        return 0
    dd = np.array([dtw(qt, T[i]) for i in range(len(w0s))])
    return int(w0s[int(np.argmin(dd))])


def match_within_idx(qt: np.ndarray, w0s: np.ndarray, T: np.ndarray) -> int:
    """Index (not w0) of the best window — for multi-episode window unions (v6
    one-shot task pin: 3 taskdemos' windows concatenated)."""
    if len(w0s) == 0:
        return -1
    dd = np.array([dtw(qt, T[i]) for i in range(len(w0s))])
    return int(np.argmin(dd))
