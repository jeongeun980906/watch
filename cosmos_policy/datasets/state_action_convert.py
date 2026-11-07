"""STATE57 <-> ACTION57 conversion for the EgoVerse convention (numpy-only, no torch).

Layout (both spaces): head9(0:9) | Leef9(9:18) | Lhand15(18:33) | Reef9(33:42) | Rhand15(42:57)
  * state57  = ABSOLUTE pose per frame:  pose9 = [pos3, R[:,0], R[:,1]]
  * action57 = per-step deltas t -> t+1 expressed in frame t's LOCAL frame
    (docs: cosmos-framework/docs/action57_state57_convention.md §3a):
        dpos_t  = R_t^T (P_{t+1} - P_t)
        drot_t  = first two columns of R_t^T R_{t+1}
    hand15 rows carry the NEXT frame's wrist-local fingertips: human converters store
    tips at t+1 in the eef(t+1) frame, robot converters store hand15(g_{t+1}) — both
    equal state57[t+1]'s hand15 block.

`derived_action57_from_states` below is the EXACT inverse of the forward integrator
`_fk_skeleton` in cosmos_policy/scripts/eval_egoverse_ret.py
(P_{t+1} = P_t + R_t @ dpos_t ; R_{t+1} = R_t @ rot6d(drot_t)), so state-chunk models
can be scored with the SAME delta-action metric pipeline as delta-action models.
"""
from __future__ import annotations

import numpy as np

# (pos, rot6d) slices per integrated joint + hand15 slices (wrist-local, copied as-is)
POSE_SLICES = ((slice(0, 3), slice(3, 9)),      # head (world / head-frame origin)
               (slice(9, 12), slice(12, 18)),   # L wrist (head-frame)
               (slice(33, 36), slice(36, 42)))  # R wrist (head-frame)
HAND_SLICES = (slice(18, 33), slice(42, 57))    # Lhand15 | Rhand15


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """6-D rot [col0(3), col1(3)] -> 3x3 (columns) via Gram-Schmidt; batched over
    leading axes. Same math as eval_egoverse_ret._rot6d_to_mat."""
    d6 = np.asarray(d6, np.float64)
    c0, c1 = d6[..., :3], d6[..., 3:6]
    b0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-8)
    c1p = c1 - (b0 * c1).sum(-1, keepdims=True) * b0
    b1 = c1p / (np.linalg.norm(c1p, axis=-1, keepdims=True) + 1e-8)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=-1)


def derived_action57_from_states(states: np.ndarray, state0: np.ndarray) -> np.ndarray:
    """EXACT inverse of `_fk_skeleton`: absolute state57 chunk -> delta action57 chunk.

    states: (T,57) absolute state57 (PHYSICAL units, de-normalized) for frames t+1..t+T.
    state0: (57,) absolute state57 at the chunk boundary frame t ("now").
    Returns (T,57) float64 action57 with, over seq = [state0, states...]:
        dpos_t = R_t^T (P_{t+1} - P_t);  drot_t = (R_t^T R_{t+1})[:, :2]
    for head + both wrists, and hand15 rows = the NEXT frame's (seq[1:]) hand15
    (the stored action57 convention — see module doc).
    """
    states = np.asarray(states, np.float64)
    seq = np.concatenate([np.asarray(state0, np.float64).reshape(1, 57), states], axis=0)
    T = states.shape[0]
    out = np.zeros((T, 57), np.float64)
    for ps, rs in POSE_SLICES:
        P = seq[:, ps]                                     # (T+1,3)
        R = rot6d_to_mat(seq[:, rs])                       # (T+1,3,3)
        out[:, ps] = np.einsum("tji,tj->ti", R[:-1], P[1:] - P[:-1])   # R_t^T (P_{t+1}-P_t)
        dR = np.einsum("tji,tjk->tik", R[:-1], R[1:])                   # R_t^T R_{t+1}
        out[:, rs.start:rs.start + 3] = dR[:, :, 0]
        out[:, rs.start + 3:rs.start + 6] = dR[:, :, 1]
    for hs in HAND_SLICES:
        out[:, hs] = seq[1:, hs]
    return out


def fk_skeleton_state(states_norm: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                      state0: np.ndarray) -> dict:
    """STATE-chunk counterpart of `_fk_skeleton`: each frame IS the absolute state57
    (head world pose + head-frame wrists + wrist-local hand15) — NO integration.
    De-normalizes with the STATE min-max (lo, hi), prepends state0 (the current state)
    as frame 0 so paths/shapes match the delta renderer: head_p/L_P/R_P (T+1,3),
    *_R (T+1,3,3), *_F (T,5,3), world cm."""
    denom = np.clip(np.asarray(hi, np.float64) - np.asarray(lo, np.float64), 1e-8, None)
    s = (np.asarray(states_norm, np.float64)[:, :57] + 1.0) / 2.0 * denom + lo
    s = np.concatenate([np.asarray(state0, np.float64).reshape(1, 57), s], axis=0)   # (T+1,57)
    T1 = s.shape[0]; M = 100.0
    (hps, hrs), (lps, lrs), (rps, rrs) = POSE_SLICES
    HP = s[:, hps] * M
    HR = rot6d_to_mat(s[:, hrs])
    out = {"head_p": HP, "head_R": HR}
    for side, (ps, rs), hs in (("L", (lps, lrs), HAND_SLICES[0]),
                               ("R", (rps, rrs), HAND_SLICES[1])):
        P = s[:, ps] * M                                   # head-frame wrist pos
        R = rot6d_to_mat(s[:, rs])
        hand = s[:, hs].reshape(T1, 5, 3) * M              # wrist-local fingertips
        F = P[:, None] + np.einsum("tij,tnj->tni", R, hand)
        out[side + "_P"] = HP + np.einsum("tij,tj->ti", HR, P)
        out[side + "_R"] = np.einsum("tij,tjk->tik", HR, R)
        out[side + "_F"] = (HP[:, None] + np.einsum("tij,tnj->tni", HR, F))[1:]
    return out
