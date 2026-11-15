# SPDX-License-Identifier: Apache-2.0
"""Demo-consistent augmentation for the Sprout ICL post-train: the human demo and the robot target change TOGETHER.

The own-ghost ICL data has one behaviour per scene, so the robot image and state already give the answer and the
model learns to ignore the demo (counterfactual check 2026-11-14: a yaw-reversed demo leaves the predicted turn
unchanged in all 8 checkpoints). Here each training sample can be transformed so that only the demo tells what the
robot does, by factors that compose (applied in this order):

* ``tempo`` r: the robot chunk and the demo window are read at t + r k and h + r k (k = 0..32); velocities x r.
  The future frame is read at t + r 32 too, so the sample stays consistent (its future loss is kept).
* ``yaw`` (s, d): the body yaw rate becomes s x the original + d per frame, in the robot chunk (base rot6d), the
  current base command, and the demo (head yaw; body-frame steps and everything relative to the body kept, as
  quest_capture/pc/make_yaw_reversed.py). s < 0 reverses the turn.
* ``wrist`` w: the working hand turns by an extra rotation Exp(a_k w) in the body frame, a_k ramping 0 -> 1 over the
  chunk, in the robot target and in the demo's same hand (positions kept).
* ``arm``: the other arm does the task: robot target, current state and demo are mirrored (y -> -y, R -> M R M, L/R
  swapped, vy and vyaw negated); the new idle arm is put at its real reset pose. The images stay (the robot image
  still shows the original arm: an inconsistency the future loss is dropped for).
* ``full``: the same mirror and the images flipped as well (a consistent mirrored world).

Samples whose future frame contradicts the new target (yaw, wrist, arm) carry ``future_loss_weight = 0`` and the
model drops their future-image loss (CosmosPolicyRealRobotRetModelRectifiedFlow._apply_ret_loss_mask).

Sprout arm mirror (URDF, checked by FK 2026-11-14): with the joint map (+, -, +, -, +, -) the left arm is the exact
mirror of the right one up to a constant offset in the gripper frame: R_L = M R_R M, p_L = M p_R + R_L (-0.0431, 0, 0).
"""
from __future__ import annotations

import numpy as np

M = np.diag([1.0, -1.0, 1.0])
GRIP_OFF = {"left": np.array([-0.0431, 0.0, 0.0]), "right": np.array([0.0431, 0.0, 0.0])}
EEF = {"left": 9, "right": 19}            # state29 / abs_action29: [p(3), R[:,0](3), R[:,1](3)], grip at +9
H_EEF = {"left": 9, "right": 33}          # state57: head 0:9 | Leef 9:18 | Ltips 18:33 | Reef 33:42 | Rtips 42:57
H_TIPS = {"left": 18, "right": 42}
TASK_ARM = {"sprout_cup_to_drawer": "right", "sprout_open_drawer": "left"}
OTHER = {"left": "right", "right": "left"}


# ---------------------------------------------------------------- rotations
def r6(x):
    """(..., 6) [col0, col1] -> (..., 3, 3)."""
    a, b = x[..., :3], x[..., 3:6]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], -1)


def m6(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def rz(t):
    t = np.asarray(t, np.float64)
    c, s = np.cos(t), np.sin(t)
    R = np.zeros(t.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 1, 0], R[..., 1, 1], R[..., 2, 2] = c, -s, s, c, 1
    return R


def exp_so3(w):
    """(..., 3) axis-angle -> (..., 3, 3)."""
    w = np.asarray(w, np.float64)
    th = np.linalg.norm(w, axis=-1, keepdims=True)[..., None]
    k = w / np.maximum(th[..., 0], 1e-12)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -k[..., 2], k[..., 1], k[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 0], -k[..., 1], k[..., 0]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


# ---------------------------------------------------------------- resampling
def lerp_rows(x, f, rot_at=()):
    """x (T, D) at fractional rows f (clamped), linear; the rot6d blocks starting at `rot_at` re-orthonormalised."""
    f = np.clip(np.asarray(f, np.float64), 0, len(x) - 1)
    i0 = np.floor(f).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    u = (f - i0)[:, None]
    out = (1 - u) * x[i0] + u * x[i1]
    for at in rot_at:
        out[:, at:at + 6] = m6(r6(out[:, at:at + 6]))
    return out


def base_angle(b):
    """head9 [vx, vy, vz, rot6d(Rz(a))] rows -> a (rad per frame)."""
    return np.arctan2(b[..., 4], b[..., 3])


def set_base(b, ang, scale_v=1.0, flip_y=False):
    b = b.copy()
    b[..., 0:3] *= scale_v
    if flip_y:
        b[..., 1] *= -1
    c, s = np.cos(ang), np.sin(ang)
    b[..., 3], b[..., 4], b[..., 5], b[..., 6], b[..., 7], b[..., 8] = c, s, 0, -s, c, 0
    return b


# ---------------------------------------------------------------- robot (29-D rows)
ROBOT_ROT = (EEF["left"] + 3, EEF["right"] + 3)


def robot_tempo(A, t0, r, n):
    """Rows t0 + r k (k < n) of the per-frame episode A (T, 29), the commands x r."""
    x = lerp_rows(A, t0 + r * np.arange(n), ROBOT_ROT)
    return set_base(x, base_angle(x) * r, scale_v=r)


def robot_yaw(x, s, d):
    return set_base(x, s * base_angle(x) + d)


def robot_wrist(x, side, w, ramp):
    """Working hand rotated by Exp(a w) in the torso frame, a per row (`ramp`), position kept."""
    x = x.copy()
    at = EEF[side]
    R = r6(x[:, at + 3:at + 9])
    x[:, at + 3:at + 9] = m6(exp_so3(np.asarray(ramp)[:, None] * np.asarray(w)[None]) @ R)
    return x


def robot_mirror(x, side, idle):
    """The other arm does what `side` did (exact Sprout mirror); `side` goes to its reset pose idle[side] (10,)."""
    x = x.copy()
    o = OTHER[side]
    a, b = EEF[side], EEF[o]
    p, R = x[:, a:a + 3], r6(x[:, a + 3:a + 9])
    Rm = M @ R @ M
    x[:, b:b + 3] = p @ M + Rm @ GRIP_OFF[o]
    x[:, b + 3:b + 9] = m6(Rm)
    x[:, b + 9] = x[:, a + 9]
    x[:, a:a + 10] = idle[side]
    return set_base(x, -base_angle(x), flip_y=True)


# ---------------------------------------------------------------- human (57-D rows)
def head_pose(s):
    """(T, 57) -> head xy (T, 2), yaw (T,) from the camera forward axis (sprout_pairs.load_human)."""
    z = r6(s[:, 3:9])[:, :, 2]
    return s[:, 0:2], np.unwrap(np.arctan2(z[:, 1], z[:, 0]))


HUMAN_ROT = (3, H_EEF["left"] + 3, H_EEF["right"] + 3)


def human_tempo(S, h, r, n):
    """Rows h + r k, k = 0..n (n + 1 rows: row 0 is the window start h)."""
    return lerp_rows(S, h + r * np.arange(n + 1), HUMAN_ROT)


def human_yaw(s, s_rate, d):
    """Body yaw psi'(i) = psi(0) + s (psi(i) - psi(0)) + d i; body-frame steps and everything relative to the body
    kept (the body path is integrated again). Row 0 unchanged."""
    s = np.asarray(s, np.float64)
    p, psi = head_pose(s)
    i = np.arange(len(psi))
    psi2 = psi[0] + s_rate * (psi - psi[0]) + d * i
    v = np.einsum("tij,tj->ti", rz(-psi[1:])[:, :2, :2], np.diff(p, axis=0))
    p2 = np.vstack([p[:1], p[:1] + np.cumsum(np.einsum("tij,tj->ti", rz(psi2[1:])[:, :2, :2], v), 0)])
    D = rz(psi2 - psi)
    off = np.c_[p2 - np.einsum("tij,tj->ti", D[:, :2, :2], p), np.zeros(len(p))]
    out = s.copy()
    for at in (0, H_EEF["left"], H_EEF["right"]):
        out[:, at:at + 3] = np.einsum("tij,tj->ti", D, s[:, at:at + 3]) + off
        out[:, at + 3:at + 9] = m6(np.einsum("tij,tjk->tik", D, r6(s[:, at + 3:at + 9])))
    return out


def human_wrist(s, side, w, ramp):
    """The `side` hand rotated by Exp(a w) in the body frame (head yaw), position and fingertips (hand frame) kept."""
    s = s.copy()
    at = H_EEF[side]
    _, psi = head_pose(s)
    E = exp_so3(np.asarray(ramp)[:, None] * np.asarray(w)[None])
    B = rz(psi)
    s[:, at + 3:at + 9] = m6(B @ E @ np.swapaxes(B, -1, -2) @ r6(s[:, at + 3:at + 9]))
    return s


def human_mirror(s):
    """y -> -y about the waist frame, R -> M R M, hands swapped (fingertips are in the hand frame: tip -> M tip)."""
    s = np.asarray(s, np.float64).copy()
    out = s.copy()
    out[:, 0:3] = s[:, 0:3] @ M
    out[:, 3:9] = m6(M @ r6(s[:, 3:9]) @ M)
    for side in ("left", "right"):
        a, b = H_EEF[side], H_EEF[OTHER[side]]
        out[:, b:b + 3] = s[:, a:a + 3] @ M
        out[:, b + 3:b + 9] = m6(M @ r6(s[:, a + 3:a + 9]) @ M)
        ta, tb = H_TIPS[side], H_TIPS[OTHER[side]]
        out[:, tb:tb + 15] = (s[:, ta:ta + 15].reshape(-1, 5, 3) @ M).reshape(-1, 15)
    return out


# ---------------------------------------------------------------- sampling the factors
DEFAULT = dict(p_tempo=0.3, tempo=(0.7, 1.4), p_yaw=0.3, yaw_scale=(0.5, 1.5), p_yaw_flip=0.5, yaw_add=0.02,
               p_wrist=0.3, wrist=0.5, p_arm=0.2, p_full=0.1)


def draw(cfg: dict, rng) -> dict:
    """One sample's factors; {} = none."""
    c = {**DEFAULT, **(cfg or {})}
    f = {}
    if rng.random() < c["p_tempo"]:
        lo, hi = c["tempo"]
        f["tempo"] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    if rng.random() < c["p_yaw"]:
        s = rng.uniform(*c["yaw_scale"]) * (-1 if rng.random() < c["p_yaw_flip"] else 1)
        f["yaw"] = (float(s), float(rng.uniform(-c["yaw_add"], c["yaw_add"])))
    if rng.random() < c["p_wrist"]:
        f["wrist"] = rng.uniform(-c["wrist"], c["wrist"], 3)
    u = rng.random()
    if u < c["p_full"]:
        f["mirror"] = "full"
    elif u < c["p_full"] + c["p_arm"]:
        f["mirror"] = "arm"
    return f


def apply(f: dict, A, st, t0, S, h, n, side, idle, demo29: bool = False, full_demo: bool = False):
    """Factors f on one sample. A (T, 29) robot actions, st (T, 29) robot states, t0 the robot frame, S (Th, 57) the
    demo, h its window start, n the chunk. -> target (n, 29), current state (29,), demo rows h+1.. (n, 57),
    demo end row offset, robot future row offset, future_loss_weight, flip images."""
    r = f.get("tempo", 1.0)
    tgt = robot_tempo(A, t0, r, n) if r != 1.0 else np.asarray(A[np.clip(t0 + np.arange(n), 0, len(A) - 1)], np.float64)
    cur = np.asarray(st[min(t0, len(st) - 1)], np.float64)[None].copy()
    if r != 1.0:
        cur = set_base(cur, base_angle(cur) * r, scale_v=r)
    if demo29:      # S is the retargeted demo (T, 29): the robot's own transforms apply to it
        dem = robot_tempo(S, h, r, n + 1) if r != 1.0 else np.asarray(S[np.clip(h + np.arange(n + 1), 0, len(S) - 1)], np.float64)
    else:
        dem = human_tempo(S, h, r, n)
    consistent = True
    if "yaw" in f:
        s, d = f["yaw"]
        tgt, cur = robot_yaw(tgt, s, d), robot_yaw(cur, s, d)
        dem = robot_yaw(dem, s, d) if demo29 else human_yaw(dem, s, d)
        consistent = False
    if "wrist" in f:
        w = f["wrist"]
        tgt = robot_wrist(tgt, side, w, (np.arange(n) + 1) / n)
        dem = (robot_wrist(dem, side, w, np.arange(n + 1) / n) if demo29
               else human_wrist(dem, side, w, np.arange(n + 1) / n))
        consistent = False
    m = f.get("mirror")
    if m:
        tgt, cur = robot_mirror(tgt, side, idle), robot_mirror(cur, side, idle)
        dem = robot_mirror(dem, side, idle) if demo29 else human_mirror(dem)
        consistent = consistent and m == "full"
    return (tgt, cur[0], dem if full_demo else dem[1:], int(round(r * n)), int(round(r * n)), 1.0 if consistent else 0.0,
            m == "full")


# ---------------------------------------------------------------- absolute base (base_abs)
# head9 as the torso pose in the episode's start frame, [x, y, 0, rot6d(Rz(yaw))], instead of the velocity command
# [vx, vy, 0, rot6d(Rz(vyaw / 15))]. The command at row t moves the torso from t to t + 1 (the robot's convention):
# a STATE row t is pose(t), an ACTION / target row t is pose(t + 1), a demo row h is pose(h) of the demo.
FPS = 15.0


def poses(cmd, xy0=(0.0, 0.0), q0=0.0):
    """(T, >=9) velocity commands -> torso xy (T + 1, 2), yaw (T + 1,) from (xy0, q0): row k = before command k."""
    cmd = np.asarray(cmd, np.float64)
    n = len(cmd)
    xy, q = np.zeros((n + 1, 2)), np.zeros(n + 1)
    xy[0], q[0] = xy0, q0
    vyaw = np.arctan2(cmd[:, 4], cmd[:, 3])                # per-frame yaw step (rad)
    for t in range(n):
        c, s = np.cos(q[t]), np.sin(q[t])
        xy[t + 1] = xy[t] + np.array([c * cmd[t, 0] - s * cmd[t, 1], s * cmd[t, 0] + c * cmd[t, 1]]) / FPS
        q[t + 1] = q[t] + vyaw[t]
    return xy, q


def abs_head(xy, q):
    xy, q = np.asarray(xy, np.float64), np.asarray(q, np.float64)
    c, s, z = np.cos(q), np.sin(q), np.zeros(len(q))
    return np.stack([xy[:, 0], xy[:, 1], z, c, s, z, -s, c, z], 1)


def head_pose_abs(h9):
    """[x, y, 0, rot6d] rows -> xy (n, 2), yaw (n,)."""
    h9 = np.asarray(h9, np.float64)
    return h9[:, 0:2].copy(), np.arctan2(h9[:, 4], h9[:, 3])


def cmd_from_abs(h9, xy0, q0):
    """Predicted absolute poses (n, 9), the current pose (xy0, q0) -> the velocity command head (n, 9) that drives
    the torso through them (the inverse of `poses`)."""
    xy, q = head_pose_abs(h9)
    xy = np.vstack([np.asarray(xy0, np.float64)[None], xy])
    q = np.unwrap(np.r_[q0, q])
    d = np.diff(xy, axis=0) * FPS
    c, s = np.cos(-q[:-1]), np.sin(-q[:-1])
    v = np.c_[c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]]
    dq = np.diff(q)
    z = np.zeros(len(dq))
    return np.stack([v[:, 0], v[:, 1], z, np.cos(dq), np.sin(dq), z, -np.sin(dq), np.cos(dq), z], 1)


def window_abs(cmd_rows, xy0, q0):
    """Command rows (n, >=9) applied from the pose (xy0, q0) -> the n poses AFTER each command, as head9."""
    xy, q = poses(cmd_rows, xy0, q0)
    return abs_head(xy[1:], q[1:])


def rel_from_abs(h9, ref9):
    """Absolute head9 rows -> relative to the pose ref9 (its torso frame): [dx, dy, 0, rot6d(Rz(dyaw))]."""
    xy, q = head_pose_abs(h9)
    rx, rq = head_pose_abs(np.asarray(ref9, np.float64)[None])
    d = xy - rx
    c, s = np.cos(-rq[0]), np.sin(-rq[0])
    return abs_head(np.c_[c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], q - rq[0])
