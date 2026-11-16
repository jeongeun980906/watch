# SPDX-License-Identifier: Apache-2.0
"""Inference server for the Sprout in-context post-train (``cosmos_predict2p5_2b_sprout_icl_c32*``): a human VR demo
conditions every chunk, retrieved online from a pinned demo pool the way the GR1 MimicDroid server does it
(``eval_gr1_server.py`` one-shot pin, ``--pinned-advance mono --pin-lock-ep``).

Same zmq protocol and the same robot client as the context-free server (``sprout_policy_server.py``,
``sprout_real/policy.py``); the client adds ``--demo-tag``. Everything below the policy (IK, chunk handover, checks)
is the context-free one.

    {"endpoint": "reset", "data": {"instruction": "...", "tag": 0, "demo": optional}}
        pins the demo pool of the task and tag and starts a new timeline:
            tag 0     the 20 training ghosts of the task      (<root>/human_ghost/<task>/episode_*.h5)
            tag 1..3  the 3 demos of that scene variant        (<root>/human_context/<task>/tag<k>/*.h5,
                                                                quest_capture/pc/to_sprout_context.py)
        "demo": a file name in the pool (e.g. "episode_3" or a human episode) to start from; default the first one
        -> {"status": "ok", "pool": [...], "hand": ...}
    {"endpoint": "get_action", "data": {"observation": {...context-free fields...,
        "icl": {"history": (L, 29) float32  the measured state29 at 15 Hz since the policy started, newest last,
                "t": frames since the policy started (15 Hz, x --speed)}}}}
        -> [{part: (1, C, d)}, {"ms": ..., "demo": file, "h": demo frame}]

The demo of a chunk is (file, h): the slots get the demo's state57 s_h[h+1 .. h+32] and its frames h and h+32,
exactly as the training dataset builds them (``SproutIclDataset._plain_build_per_view`` is called with the demo
swapped in, as ``sprout_policy_server`` swaps in the robot's live frame). Training paired robot frame t with the
ghost's frame t (+-6 jitter), so:

* chunk 0: h = 0 of the start demo (the GR1 t = 0 convention);
* later chunks: the robot's strict past (its working hand over the last <= 32 frames, torso frame) is matched by
  banded DTW against every demo window ending at a candidate h. The descriptor is the training pairs'
  (``quest_capture/pc/sprout_pairs.py``: start-relative wrist path + per-step delta, z-scored, 40 steps; the human
  hand in the body frame, a mirrored demo with y flipped). The candidates are a monotonic band
  [last + --mono-floor, last + --mono-band] in demo frames (GR1 ``mono``: 8, 48; sprout_icl_r29_server.py: 8, 96), whatever the wall time since the
  last chunk (the robot holds or slows while inferring, so elapsed time is not progress; ``--band elapsed`` scales
  the band by it instead: [last + lo x elapsed, last + hi x elapsed + pad]);
  the demo is the start one for the whole run (``--lock start``: the client's ``--demo``, else the pool's
  first); ``--lock first`` lets the first re-match pick it from the pool (GR1 ``--pin-lock-ep``). ``--advance lockstep`` instead moves h by the elapsed frames.

Launch::

    CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=12371 \\
    PYTHONPATH=. uv run --extra cu128 --group robocasa --python 3.10 \\
      python -m cosmos_policy.scripts.sprout_icl_server \\
        --experiment cosmos_predict2p5_2b_sprout_icl_c32 --checkpoint <.../sprout_icl/model_000004000.pt> --port 5555

    # no GPU: sample builder against training samples, and the retrieval replayed on robot episodes
    python -m cosmos_policy.scripts.sprout_icl_server --experiment cosmos_predict2p5_2b_sprout_icl_c32 --selftest
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time

import numpy as np

from cosmos_policy.datasets.sprout_dataset import DIM29, SPROUT_ROOT, SproutIclDataset, make
from cosmos_policy.datasets.tabletop_sim_dataset import TabletopEpisode
from cosmos_policy.scripts import sprout_policy_server as base

# robot task -> (working arm, its EEF xyz in state29); the other arm is idle (policy.py IDLE_ARM)
ARM = {"sprout_cup_to_drawer": ("R", slice(19, 22)), "sprout_open_drawer": ("L", slice(9, 12))}
WIN = 32            # past window (the training chunk; sprout_pairs.WIN)
TS, BAND = 40, 6    # DTW steps and Sakoe-Chiba band (sprout_pairs.TS / BAND)


# ---------------------------------------------------------------- descriptor (= quest_capture/pc/sprout_pairs.py)
def _r6(x):
    a, b = x[..., :3], x[..., 3:6]
    return np.stack([a, b, np.cross(a, b)], -1)


def _resample(a, m=TS):
    return a[np.linspace(0, len(a) - 1, m).round().astype(int)]


def wrist_seq(p):
    """(L,3) path -> (TS,6) start-relative position + per-step delta, z-scored per dim."""
    a = _resample(p - p[0])
    d = np.vstack([np.zeros((1, 3)), np.diff(a, axis=0)])
    w = np.concatenate([a, d], 1)
    return ((w - w.mean(0)) / (w.std(0) + 1e-6)).astype(np.float32)


def dtw_batch(q, C, band=BAND):
    """q (T,D), C (N,T,D) -> (N,) banded DTW cost (mean over path length)."""
    N, T = len(C), len(q)
    D = np.full((N, T + 1, T + 1), np.inf, np.float32)
    D[:, 0, 0] = 0
    for i in range(1, T + 1):
        for j in range(max(1, i - band), min(T, i + band) + 1):
            c = np.linalg.norm(C[:, j - 1] - q[i - 1], axis=1)
            D[:, i, j] = c + np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    return D[:, T, T] / (2 * T)


def human_path(s57: np.ndarray, hand: str, mirrored: bool) -> np.ndarray:
    """state57 -> (T,3) working-hand path in the body frame: relative to the head horizontally, rotated by the head
    yaw (sprout_pairs.load_human). A mirrored demo (other hand, mirrored ghost) gets y flipped to match the robot arm."""
    s = np.asarray(s57, np.float64)
    i = 33 if hand == "R" else 9
    hp = s[:, 0:3]
    fwd = _r6(s[:, 3:9])[:, :, 2]
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    d = s[:, i:i + 3] - np.c_[hp[:, :2], np.zeros(len(hp))]
    c, sn = np.cos(-yaw), np.sin(-yaw)
    p = np.c_[c * d[:, 0] - sn * d[:, 1], sn * d[:, 0] + c * d[:, 1], d[:, 2]]
    if mirrored:
        p[:, 1] *= -1
    return p


# ---------------------------------------------------------------- demo pool + online retrieval
class Demo:
    def __init__(self, path: str, hand: str, mirrored: bool):
        self.path, self.name = path, os.path.basename(path)[:-3]
        self.ep = TabletopEpisode(path)
        self.n = int(self.ep.n_frames)
        if DEMO29 and MATCH29:          # the retargeted demo's own torso-frame EEF: the robot history's frame
            import h5py
            with h5py.File(path, "r") as f:
                side = str(f.attrs.get("retarget_side", "right" if hand == "R" else "left"))
                x = f["state29"][()].astype(np.float64)
            a = 19 if side == "right" else 9
            self.p = x[:, a:a + 3].copy()
            self.g = x[:, a + 9].copy()     # the gripper (retarget.py: the followed robot episode's)
            self.b = base_cmd(x)            # the base command (retarget.py: the person's body, robot-like bursts)
            if mirrored:                # the demo's other arm: mirror its y onto the robot's working arm
                self.p[:, 1] *= -1
        else:
            self.p = human_path(self.ep.state("state57"), hand, mirrored)
            self.g = self.b = None


# --demo29: the demos are the retargeted copies (quest_capture/pc/retarget.py), the model's demo slot reads "state29"
DEMO29 = False
MATCH29 = True      # with DEMO29: retrieve on the retargeted torso-frame EEF (False: the state57 head-frame wrist)
BASE_REL = False     # ..._r29rel*: the predicted poses are relative to the current pose (and so is the demo chunk)
BASE_ABS_OUT = True  # ..._r29absin*: only the state and the demo are absolute, the model outputs velocity commands
BASE_ABS = False    # ..._r29abs*: head9 = the start-frame torso pose (state, target, demo); the server integrates the
                    # client's velocity commands into that pose and turns the predicted poses back into commands
GRIP_W = 12.0       # ... plus the gripper (0..1, not z-scored) x this, as a 7th descriptor channel


IDLE_HOLD, IDLE_M = False, 0.02  # the robot still (EEF path < 2 cm, gripper and base unchanged): h holds
POS_W = 64.0        # ... plus the torso-frame EEF position (m) x this, 3 channels (retargeted demos share the frame)
PROG_W = 0.0        # ... plus this x |advance - elapsed| / elapsed (a pace prior; 0 off)
BASE_W = 12.0       # ... plus the base command (vx, vy m/s, yaw rate rad/s; not z-scored) x this, 3 more channels


def grip_seq(g, w):
    return (w * _resample(np.asarray(g, np.float64))[:, None]).astype(np.float32)


def base_cmd(x29):
    """state29 head9 = [vx, vy, vz, rot6d(Rz(vyaw / 15))] -> (T, 3) [vx, vy, vyaw]."""
    x = np.asarray(x29, np.float64)
    return np.c_[x[:, 0], x[:, 1], np.arctan2(x[:, 4], x[:, 3]) * 15.0]


def base_seq(b, w):
    return (w * _resample(np.asarray(b, np.float64))).astype(np.float32)


def human_dirs() -> tuple[str, str]:
    return ("human_ghost_r29", "human_context_r29") if DEMO29 else ("human_ghost", "human_context")


def pool_files(root: str, task: str, tag: int) -> list[str]:
    hg, hc = human_dirs()
    if int(tag) == 0:
        fs = glob.glob(os.path.join(root, hg, task, "*.h5"))
        return sorted(fs, key=lambda f: int(re.search(r"_(\d+)\.h5$", f).group(1)))
    return sorted(glob.glob(os.path.join(root, hc, task, f"tag{int(tag)}", "*.h5")))


class Retriever:
    """GR1 one-shot pin (eval_gr1_server.py ``--pinned-advance mono --pin-lock-ep``) on the Sprout descriptor."""

    def __init__(self, root: str, task: str, tag: int, demo: str | None = None, advance: str = "mono",
                 lo: float = 0.5, hi: float = 1.5, pad: int = 8, lock: str = "start", exclude: str | None = None,
                 band: str = "gr1", floor: int = 8, cap: int = 48):
        import h5py

        files = [f for f in pool_files(root, task, tag) if os.path.basename(f)[:-3] != exclude]
        if not files:
            raise FileNotFoundError(f"no demos for {task} tag {tag} under {root}")
        self.arm, self.rslice = ARM[task]
        self.demos = []
        for f in files:
            with h5py.File(f, "r") as h:
                hand = str(h.attrs.get("hand", self.arm))
                mir = bool(h.attrs.get("mirrored", False))
            self.demos.append(Demo(f, hand, mir))
        names = [d.name for d in self.demos]
        if demo is not None and demo not in names:
            raise KeyError(f"demo {demo!r} not in the pool: {names}")
        self.start = names.index(demo) if demo is not None else 0
        self.task, self.tag = task, int(tag)
        self.advance, self.lo, self.hi, self.pad, self.lock = advance, lo, hi, pad, lock
        self.band, self.floor, self.cap = band, int(floor), int(cap)
        assert lock in ("start", "first", "none"), lock
        self.cur, self.h, self.t, self.locked, self.chunk = self.start, 0, 0.0, lock == "start", 0

    def names(self) -> list[str]:
        return [d.name for d in self.demos]

    def next(self, history: np.ndarray | None, t: float) -> tuple[Demo, int, float]:
        """(demo, h, DTW cost or nan) for the chunk observed at t (demo-speed frames since the start)."""
        if self.chunk == 0 or history is None or len(history) < 3:
            h, cost = 0 if self.chunk == 0 else self.h, float("nan")
        else:
            el = max(0.0, t - self.t)
            if self.advance == "lockstep":
                h, cost = int(round(self.h + el)), float("nan")
            else:
                h, cost = self._match(history, el)
        d = self.demos[self.cur]
        self.h = int(min(max(h, 0), d.n - 1))
        self.t, self.chunk = t, self.chunk + 1
        return d, self.h, cost

    def _match(self, history: np.ndarray, el: float) -> tuple[int, float]:
        hist = np.asarray(history, np.float64)[:-1][-WIN:]                     # strict past, as the pairs
        q = hist[:, self.rslice]
        if IDLE_HOLD and len(q) > 1:      # the robot has not moved since the last chunk: the demo does not advance
            path = np.linalg.norm(np.diff(q, axis=0), axis=1).sum()
            g = hist[:, self.rslice.stop + 6]
            moving_base = np.abs(base_cmd(hist)).max() > 1e-3
            if path < IDLE_M and np.ptp(g) < 0.05 and not moving_base:
                return self.h, float("nan")
        L = len(q)
        qs = wrist_seq(q)
        use_g = GRIP_W > 0 and self.demos[self.cur].g is not None
        use_b = BASE_W > 0 and self.demos[self.cur].b is not None
        if use_g:
            qs = np.concatenate([qs, grip_seq(hist[:, self.rslice.stop + 6], GRIP_W)], 1)
        if use_b:
            qs = np.concatenate([qs, base_seq(base_cmd(hist), BASE_W)], 1)
        use_p = POS_W > 0 and self.demos[self.cur].g is not None
        if use_p:                          # the EEF itself (torso frame, not start-relative / z-scored): start vs end
            qs = np.concatenate([qs, (POS_W * _resample(q)).astype(np.float32)], 1)
        if self.band == "gr1":
            lo, hi = self.h + self.floor, self.h + self.cap
        else:
            lo = self.h + max(1, int(round(self.lo * el)))
            hi = self.h + int(round(self.hi * el)) + self.pad
        cand = [self.cur] if self.locked else range(len(self.demos))
        rows, C = [], []
        for k in cand:
            d = self.demos[k]
            for h in range(max(lo, 2), min(hi, d.n - 1) + 1):
                rows.append((k, h))
                c = wrist_seq(d.p[max(0, h - L):h])
                if use_g:
                    c = np.concatenate([c, grip_seq(d.g[max(0, h - L):h], GRIP_W)], 1)
                if use_b:
                    c = np.concatenate([c, base_seq(d.b[max(0, h - L):h], BASE_W)], 1)
                if use_p:
                    c = np.concatenate([c, (POS_W * _resample(d.p[max(0, h - L):h])).astype(np.float32)], 1)
                C.append(c)
        if not rows:                       # band past the demo's end: stay on the demo, at its end side
            return min(lo, self.demos[self.cur].n - 1), float("nan")
        cost = dtw_batch(qs, np.stack(C))
        if PROG_W > 0 and el > 0:          # prefer advancing about the elapsed frames when the descriptor is ambiguous
            hh = np.array([r[1] for r in rows], np.float64)
            cost = cost + PROG_W * np.abs(hh - (self.h + el)) / el
        j = int(np.argmin(cost))
        self.cur = rows[j][0]
        self.locked = self.lock != "none"
        return rows[j][1], float(cost[j])


# ---------------------------------------------------------------- sample builder
class IclSampleBuilder(base.SampleBuilder):
    """Robot observation + (demo, h) -> training-format sample, through ``SproutIclDataset``."""

    def __init__(self, root: str, chunk: int):
        self.ds = make(root, cls=SproutIclDataset, chunk_size=chunk, jitter=0, other_prob=0.0, demo29=DEMO29,
                       base_abs=BASE_ABS, base_abs_out=BASE_ABS_OUT, base_rel=BASE_REL)
        self.root = root
        self.chunk = chunk
        w0 = np.asarray(self.ds.ww0)
        self.slot, self.task = {}, {}
        for q in np.flatnonzero(w0 == 0):
            c = str(self.ds.wcls[q])
            self.slot.setdefault(c, int(q))
            self.task.setdefault(c, str(self.ds.wep[q]).split("/")[0])
        emb = str(self.ds.wemb[next(iter(self.slot.values()))])
        tgt = self.ds.target_stats if self.ds.target_stats else self.ds.state_stats
        s = tgt.get(emb, tgt["_global"])
        self.lo, self.hi = np.asarray(s["lo"], np.float64), np.asarray(s["hi"], np.float64)

    def build(self, image, state29: np.ndarray, instruction: str, demo=None, h: int = 0) -> dict:
        if demo is None:
            raise RuntimeError("no demo pinned: send reset with the instruction and tag first")
        ds = self.ds
        ds._demo = lambda q, qw0, aug: (demo, int(h))        # the training code, the pinned demo swapped in
        try:
            return super().build(image, state29, instruction)
        finally:
            del ds._demo


class IclPolicy(base.Policy):
    builder_cls = IclSampleBuilder

    def __init__(self, a):
        super().__init__(a)
        self.a, self.r = a, None

    def info(self) -> dict:
        return {**super().info(), "icl": True}

    def reset(self, req: dict | None = None) -> dict:
        req = req or {}
        ins = str(req.get("instruction", ""))
        if ins not in self.b.task:
            raise KeyError(f"unknown instruction {ins!r}; ICL tasks: {sorted(self.b.task)}")
        a = self.a
        self.r = Retriever(self.b.root, self.b.task[ins], int(req.get("tag", 0)), req.get("demo"),
                           a.advance, a.band_lo, a.band_hi, a.band_pad, a.lock,
                           band=a.band, floor=a.mono_floor, cap=a.mono_band)
        self.ins = ins
        self.pose, self.pose_t = (np.zeros(2), 0.0), 0.0
        self.carry = (0.0, 0.0, 0.0)
        print(f"[sprout-icl] reset: {self.r.task} tag {self.r.tag}, pool {self.r.names()}, start "
              f"{self.r.names()[self.r.start]}", flush=True)
        return {"status": "ok", "pool": self.r.names(), "start": self.r.names()[self.r.start], "arm": self.r.arm}

    def sample(self, obs: dict) -> tuple[dict, dict]:
        if self.r is None:
            raise RuntimeError("no demo pool: the client must send reset (policy.py --demo-tag) first")
        image, state, ins = base.parse(obs)
        if ins != self.ins:
            raise KeyError(f"instruction {ins!r} is not the one reset with ({self.ins!r})")
        icl = obs.get("icl") or {}
        hist = icl.get("history")
        hist = None if hist is None else np.asarray(hist, np.float64).reshape(-1, DIM29)
        d, h, cost = self.r.next(hist, float(icl.get("t", 0.0)))
        if BASE_ABS:
            state = self._abs_state(state, hist, float(icl.get("t", 0.0)))
        print(f"[sprout-icl] chunk {self.r.chunk - 1}: t {float(icl.get('t', 0.0)):6.1f} -> {d.name} h {h} "
              f"(cost {cost:.3f})", flush=True)
        return self.b.build(image, state, ins, d.ep, h), {"demo": d.name, "h": int(h), "cost": cost}


def _abs_state(self, state, hist, t):
    """Integrate the velocity commands since the last chunk (the history's rows for frames [last t, t)) into the
    start-frame torso pose; the model's state gets that pose as head9."""
    from cosmos_policy.datasets import sprout_icl_aug as A
    (xy, q), t0 = self.pose, self.pose_t
    n = int(round(t - t0))
    if hist is not None and n > 0 and len(hist) > 1:
        rows = np.asarray(hist, np.float64)[:-1][-n:]          # the commands at frames t - n .. t - 1
        pxy, pq = A.poses(rows, xy, q)
        xy, q = pxy[-1], pq[-1]
    self.pose, self.pose_t = (xy, q), t
    out = np.asarray(state, np.float64).copy()
    out[:9] = A.abs_head(np.asarray(xy)[None], np.array([q]))[0]
    return out


ABS_SMOOTH = int(os.environ.get("SPROUT_ABS_SMOOTH", -1))  # base_abs: -1 = robot-speed bursts (burst_cmd); 0 = a
                                                            # constant velocity to the chunk end; N > 1 = moving-average window (frames) over the
                                                            # predicted poses before differencing; 0 = straight line
                                                            # from the current pose to the chunk's end pose
ABS_DEAD_M = float(os.environ.get("SPROUT_ABS_DEAD_M", 0.05))       # ... and no base command when the chunk moves
ABS_DEAD_DEG = float(os.environ.get("SPROUT_ABS_DEAD_DEG", 5.0))    # less than this


BURST_V = float(os.environ.get("SPROUT_BURST_V", 0.38))          # m/s, the robot's median base speed when it moves
BURST_W = float(os.environ.get("SPROUT_BURST_DEG_S", 60.0))      # deg/s, its median turn rate
BURST_EXEC = int(os.environ.get("SPROUT_BURST_EXEC", 24))        # frames of a chunk the client plays (its --replan)
BURST_START = int(os.environ.get("SPROUT_BURST_START", 6))       # earliest burst frame: the client's chunk crossfade
                                                                 # (--blend-s 0.4 s = 6 frames) also fades the base velocity
# Per request the client may override these (options: burst_v, burst_deg_s, burst_exec, burst_start, dead_m, dead_deg)
# and ask for xy_gain: the step it SENDS is the predicted one x xy_gain (longer at the same speed, inside burst_exec);
# info["base_track"] is the same burst without the gain: what the client keeps in the history (the tracked pose).
BURST_KEYS = ("burst_v", "burst_deg_s", "burst_exec", "burst_start", "dead_m", "dead_deg", "grip_wait")  # + carry
GRIP_WAIT = True       # bursts start only after the chunk's predicted gripper change (place, release, then step back)
CARRY = True           # what the gripper wait cut is added to the next chunk's step / turn
GRIP_EPS = 0.1         # a gripper change: the predicted gripper moves more than this over the frames played


def grip_settle(res, E):
    """The frame after the last predicted gripper change within the first E frames (0: none)."""
    out = 0
    for k in ("left_gripper", "right_gripper"):
        if k not in res:
            continue
        g = np.asarray(res[k][0], np.float64).reshape(-1)[:E]
        if len(g) > 1 and np.ptp(g) > GRIP_EPS:
            lo, hi = g.min() + 0.1 * np.ptp(g), g.max() - 0.1 * np.ptp(g)
            moving = np.nonzero((g > lo) & (g < hi))[0]               # frames in the middle 80 % of the change
            if len(moving):
                out = max(out, int(moving[-1]) + 2)
    return out


def burst_cmd(b, xy0, q0, gain=1.0, burst_v=None, burst_deg_s=None, burst_exec=None, burst_start=None, dead_m=None,
              dead_deg=None, not_before=0, extra=(0.0, 0.0, 0.0)):
    """Predicted poses (n, 9) -> (base commands as the robot drives, what was left out). The chunk's net step (current
    body frame; + extra: the previous chunk's left-out part) x gain goes at burst_v, then its net turn at burst_deg_s,
    each as one burst inside the burst_exec frames the client plays: from where the prediction starts to move (not
    before burst_start; earlier when it would not fit), but never before not_before (the predicted gripper change:
    release, then step back). What does not fit after not_before is cut and returned as (dx, dy, dyaw), predicted,
    without the gain. Below dead_m / dead_deg: nothing."""
    from cosmos_policy.datasets import sprout_icl_aug as A
    V = BURST_V if burst_v is None else float(burst_v)
    W = BURST_W if burst_deg_s is None else float(burst_deg_s)
    E = BURST_EXEC if burst_exec is None else int(burst_exec)
    S = BURST_START if burst_start is None else int(burst_start)
    DM = ABS_DEAD_M if dead_m is None else float(dead_m)
    DD = ABS_DEAD_DEG if dead_deg is None else float(dead_deg)
    n = len(b)
    E = max(1, min(E, n))
    bx, bq = A.head_pose_abs(b)
    c, s_ = np.cos(-q0), np.sin(-q0)
    rel = (bx - np.asarray(xy0)[None]) @ np.array([[c, s_], [-s_, c]])        # body frame of now
    dq = np.angle(np.exp(1j * (np.unwrap(np.r_[q0, bq])[1:] - q0)))
    p0, pyaw = rel[-1], dq[-1]                                                  # the prediction alone
    ex, eq = np.asarray(extra[:2], np.float64), float(extra[2])       # the left-out part, less what the prediction
    if np.linalg.norm(ex) > 1e-6:                                       # already does in the same direction
        u = ex / np.linalg.norm(ex)
        ex = u * max(0.0, np.linalg.norm(ex) - max(0.0, float(p0 @ u)))
    if abs(eq) > 1e-6:
        eq = np.sign(eq) * max(0.0, abs(eq) - max(0.0, np.sign(eq) * pyaw))
    d0, dyaw = p0 + ex, pyaw + eq
    cmd = np.zeros((n, 9))
    cmd[:, 3] = cmd[:, 7] = 1.0
    move = np.linalg.norm(d0) >= DM
    turn = abs(np.degrees(dyaw)) >= DD
    if not (move or turn):
        return cmd, (0.0, 0.0, 0.0)
    d = d0 * float(gain)
    m_xy = int(np.ceil(np.linalg.norm(d) / V * A.FPS)) if move else 0
    m_q = int(np.ceil(abs(np.degrees(dyaw)) / W * A.FPS)) if turn else 0
    pm, pt = np.linalg.norm(p0) >= 1e-3, abs(pyaw) >= 1e-3
    prog = np.maximum(np.linalg.norm(rel, axis=1) / max(np.linalg.norm(p0), 1e-6) if pm else 0,
                      np.abs(dq) / max(abs(pyaw), 1e-6) if pt else 0)
    t0 = int(np.argmax(prog >= 0.25)) if (pm or pt) else 0                     # the predicted motion's start
    t0 = max(t0, S)
    t0 = min(t0, E - (m_xy + m_q))                                             # earlier when it would not fit ...
    t0 = max(t0, int(not_before), 0)                                           # ... but not before the gripper change
    room = E - t0
    k_xy, k_q = m_xy, m_q
    if m_xy + m_q > room:                                                       # cut to the frames left
        f = max(room, 0) / (m_xy + m_q)
        k_xy = int(np.floor(m_xy * f)) if move else 0
        k_q = max(room, 0) - k_xy if turn else 0
    fx = k_xy / m_xy if m_xy else 0.0
    fq = k_q / m_q if m_q else 0.0
    if k_xy:
        cmd[t0:t0 + k_xy, 0:2] = d * fx / k_xy * A.FPS                          # constant body-frame velocity
    if k_q:
        step = dyaw * fq / k_q
        a = t0 + k_xy
        cmd[a:a + k_q, 3], cmd[a:a + k_q, 4] = np.cos(step), np.sin(step)
        cmd[a:a + k_q, 6], cmd[a:a + k_q, 7] = -np.sin(step), np.cos(step)
    left_xy = d0 * (1 - fx) if move else np.zeros(2)
    left_q = dyaw * (1 - fq) if turn else 0.0
    return cmd, (float(left_xy[0]), float(left_xy[1]), float(left_q))


def smooth_abs(b, xy0, q0, w=None):
    """Moving average of the predicted (x, y, yaw) over the chunk, the current pose prepended (so the first steps
    start from it); returned as head9 rows."""
    from cosmos_policy.datasets import sprout_icl_aug as A
    w = ABS_SMOOTH if w is None else w
    bx, bq = A.head_pose_abs(b)
    if w == 0:                          # constant velocity: the current pose to the chunk end in a straight line
        n = len(bq)
        f = (np.arange(1, n + 1) / n)[:, None]
        q1 = q0 + np.angle(np.exp(1j * (bq[-1] - q0)))
        return A.abs_head(np.asarray(xy0)[None] + f * (bx[-1] - np.asarray(xy0))[None], q0 + f[:, 0] * (q1 - q0))
    if w <= 1:
        return b
    P = np.c_[np.vstack([np.asarray(xy0)[None], bx]), np.unwrap(np.r_[q0, bq])]
    k = w // 2
    pad = np.vstack([np.repeat(P[:1], k, 0), P, np.repeat(P[-1:], k, 0)])
    S = np.stack([np.convolve(pad[:, i], np.ones(w) / w, mode="valid") for i in range(3)], 1)[1:]
    return A.abs_head(S[:, :2], S[:, 2])


def _abs_act(self, obs, options=None):
    res, info = base.Policy.act(self, obs, options)
    if BASE_ABS and BASE_ABS_OUT:
        from cosmos_policy.datasets import sprout_icl_aug as A
        xy, q = (np.zeros(2), 0.0) if BASE_REL else self.pose      # base_rel: the poses are relative to now
        b = np.asarray(res["base"][0], np.float64)
        track = None
        if ABS_SMOOTH < 0:                 # bursts at the robot's base speed (it does not move below ~0.35 m/s)
            kw = {k: (options or {})[k] for k in BURST_KEYS if (options or {}).get(k) is not None}
            gain = float((options or {}).get("xy_gain") or 1.0)
            wait = kw.pop("grip_wait", GRIP_WAIT)
            E = int(kw.get("burst_exec", BURST_EXEC))
            kw["not_before"] = grip_settle(res, E) if wait else 0
            carry = bool((options or {}).get("carry", CARRY))
            kw["extra"] = getattr(self, "carry", (0.0, 0.0, 0.0)) if carry else (0.0, 0.0, 0.0)
            info["base_after"], info["base_carry"] = kw["not_before"], list(kw["extra"])
            cmd, _ = burst_cmd(b, xy, q, gain, **kw)
            track, left = burst_cmd(b, xy, q, 1.0, **kw)                # the tracked (ungained) burst
            self.carry = left                                                # cut by the gripper wait: the next chunk's
        else:
            b = smooth_abs(b, xy, q)
            cmd = A.cmd_from_abs(b, xy, q)
        bx, bq = A.head_pose_abs(b)
        if ABS_SMOOTH >= 0 and np.linalg.norm(bx[-1] - xy) < ABS_DEAD_M and abs(np.angle(np.exp(1j * (bq[-1] - q)))) < np.radians(ABS_DEAD_DEG):
            cmd[:, :2] = 0.0                                  # a still base: the robot stands still between moves
            cmd[:, 3:9] = [1, 0, 0, 0, 1, 0]
        res["base"] = cmd.astype(np.float32)[None]
        info["base_track"] = (cmd if track is None else track).astype(np.float32)
        info["pose"] = [float(xy[0]), float(xy[1]), float(np.degrees(q))]
    return res, info


IclPolicy._abs_state = _abs_state
IclPolicy.act = _abs_act


def warm_up(p: IclPolicy) -> None:
    ds, b = p.b.ds, p.b
    ins, q = next(iter(b.slot.items()))
    g = ds._grp(q)
    w0 = int(ds.ww0[q])
    state = np.asarray(g.state(ds.state_group)[w0], np.float64)
    p.reset({"instruction": ins, "tag": 0})
    obs = {"video": {"front": g.frame(w0, "images.front_1")}, "state": {k: state[base.SLICES[k]] for k in base.PARTS},
           "language": {base.LANGUAGE: [[ins]]}}
    t = time.monotonic()
    p.act(obs)
    p.r = None
    print(f"[sprout-icl] warm-up inference {(time.monotonic() - t) * 1000:.0f} ms (not a client request)", flush=True)


# ---------------------------------------------------------------- offline checks
def selftest(b: IclSampleBuilder, n: int = 4) -> None:
    """(1) server sample with the own ghost at h = t == the training sample (jitter 0), every field but the
    generated slots; (2) the retrieval replayed on robot episodes."""
    import torch

    ds = b.ds
    rng = np.random.default_rng(0)
    dup = 4
    for q in rng.choice(np.arange(ds.N), n, replace=False):
        q = int(ds.query_idx[q])
        g = ds._grp(q)
        w0 = int(ds.ww0[q])
        ep = str(ds.wep[q])
        ref = ds._plain_build(q, 0, False, None)
        demo = TabletopEpisode(os.path.join(b.root, human_dirs()[0], ep + ".h5"))
        got = b.build(g.frame(w0, "images.front_1"), g.state(ds.state_group)[w0], str(ds.wcls[q]), demo, w0)
        del ds._zcache[str(ds.wep[b.slot[str(ds.wcls[q])]])]
        nc = int(got["num_conditional_frames"])
        for k, v in ref.items():
            if k in ("actions", "next_action_chunk", "__key__"):
                continue
            w = got[k]
            if k == "video":
                v, w = torch.as_tensor(v).float(), torch.as_tensor(w).float()
                cond = 1 + (nc - 1) * dup
                assert (v[:, :cond] - w[:, :cond]).abs().max().item() == 0, k
                assert (w[:, cond + dup:] == 0).all(), "future slot not black"
            elif torch.is_tensor(v) or isinstance(v, np.ndarray):
                assert torch.equal(torch.as_tensor(v).float(), torch.as_tensor(w).float()), k
            else:
                assert v == w, (k, v, w)
        print(f"  {ep} frame {w0}: equal to the training sample (own ghost, h = t)")
    print("sample builder ok")


def replay_check(root: str, task: str, tag: int, robot_eps: list[str], step: int = 20, exclude_own: bool = True,
                 **kw) -> None:
    """The robot episode's own state29 as the live history, a chunk every `step` frames: the h picked vs t."""
    import h5py

    for ep in robot_eps:
        with h5py.File(os.path.join(root, task, ep + ".h5"), "r") as f:
            s = f["state29"][:].astype(np.float64)
        r = Retriever(root, task, tag, exclude=ep if (exclude_own and tag == 0) else None, **kw)
        rows = []
        for t in range(0, len(s) - 1, step):
            d, h, c = r.next(s[max(0, t - WIN):t + 1], float(t))
            rows.append((t, h, d.name, c))
        err = [abs(h - t) for t, h, _, _ in rows]
        print(f"  {task} {ep} tag {tag}: demo {rows[-1][2]}, |h - t| median {np.median(err):.0f} max {max(err)} frames "
              f"| " + " ".join(f"{t}->{h}" for t, h, _, _ in rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="cosmos_predict2p5_2b_sprout_icl_c32[_base|_mix30|_mix30_base]")
    ap.add_argument("--checkpoint")
    ap.add_argument("--config", default="cosmos_policy/config/config.py")
    ap.add_argument("--root", default=SPROUT_ROOT, help="the training store (manifest, h5, human_ghost, human_context)")
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--shift", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sdpa", choices=("math", "auto"), default="math")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    r = ap.add_argument_group("retrieval (GR1 one-shot pin)")
    r.add_argument("--advance", choices=("mono", "lockstep"), default="mono",
                   help="mono: DTW re-match inside the monotonic band; lockstep: h moves by the elapsed frames")
    r.add_argument("--band", choices=("gr1", "elapsed"), default="gr1",
                   help="gr1: [last + --mono-floor, last + --mono-band] per chunk; elapsed: scaled by the wall time")
    r.add_argument("--mono-floor", type=int, default=8, help="--band gr1: least advance per chunk (demo frames)")
    r.add_argument("--mono-band", type=int, default=48, help="--band gr1: most advance per chunk (demo frames; "
                   "sprout_icl_r29_server.py: 96)")
    r.add_argument("--band-lo", type=float, default=0.5, help="band start: last h + this x elapsed frames (>= 1)")
    r.add_argument("--band-hi", type=float, default=1.5, help="band end: last h + this x elapsed frames + --band-pad")
    r.add_argument("--band-pad", type=int, default=8)
    r.add_argument("--lock", choices=("start", "first", "none"), default="start",
                   help="start: one demo for the whole run, the start one (--demo from the client, else the pool's "
                   "first); first: the first re-match picks it from the pool (GR1 --pin-lock-ep); none: re-match "
                   "over the whole pool every chunk")
    ap.add_argument("--demo29", action="store_true",
                    help="demos = the retargeted copies (human_ghost_r29 / human_context_r29, \"state29\"); for the "
                    "..._r29* experiments")
    ap.add_argument("--selftest", action="store_true", help="no GPU: sample builder + retrieval replay")
    a = ap.parse_args()
    global DEMO29, BASE_ABS
    DEMO29 = bool(a.demo29 or "_r29" in a.experiment)
    BASE_ABS = "_r29abs" in a.experiment or "_r29rel" in a.experiment
    global BASE_REL
    BASE_REL = "_r29rel" in a.experiment
    global BASE_ABS_OUT
    BASE_ABS_OUT = "_r29absin" not in a.experiment
    m = re.search(r"_sprout_icl_c(\d+)", a.experiment)
    if not m:
        ap.error("--experiment must be a ..._sprout_icl_c<chunk> config")
    if a.selftest:
        selftest(IclSampleBuilder(a.root, int(m.group(1))))
        kw = dict(advance=a.advance, lo=a.band_lo, hi=a.band_hi, pad=a.band_pad, lock=a.lock,
                  band=a.band, floor=a.mono_floor, cap=a.mono_band)
        for task in ARM:
            hc = os.path.join(a.root, human_dirs()[1], task)
            for tag in [0] + sorted({int(x[3:]) for x in os.listdir(hc)} if os.path.isdir(hc) else []):
                replay_check(a.root, task, tag, ["episode_10", "episode_15"], **kw)
        return
    if not a.checkpoint:
        ap.error("--checkpoint is required")
    import torch

    if a.sdpa == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    p = IclPolicy(a)
    if int(m.group(1)) != p.chunk:
        raise SystemExit(f"name says chunk {m.group(1)}, config {p.chunk}")
    warm_up(p)
    base.serve(p, a.host, a.port)


if __name__ == "__main__":
    main()
