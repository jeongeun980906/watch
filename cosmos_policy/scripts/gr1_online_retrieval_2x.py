"""No-velocity (speed-invariant) retrieval descriptor for the 2x demo pool.

The 2x GR1 post-train (egoverse_pairs_gr1_v6_2x) re-matched with a NO-VELOCITY
descriptor — pos + rot6D(x0.25) + grip (20-d) — because a 2x-sped demo has doubled
frame-to-frame velocity that would mismatch the 1x online robot. This module provides
drop-in no-vel replacements for gr1_online_retrieval.traj_from_state57 /
traj_from_proprio and installs them onto the base module so that BOTH descriptor
producers used at serve time pick up the change with NO edit to the base file:

  * episode_windows(state57)   -> traj_from_state57   (pinned-demo windows T)
  * OnlineRetriever.query()    -> traj_from_proprio   (live robot query q)

Both look the two functions up as module globals at call time, so rebinding them on
the base module is sufficient. Call install() once before constructing the server.

Usage: imported and install()-ed by eval_gr1_server_2x.py — not run directly.
"""
from __future__ import annotations

import numpy as np

from cosmos_policy.scripts import gr1_online_retrieval as _ret


def traj_from_state57_novel(s: np.ndarray, e: int, win: int = _ret.WIN,
                            seq: int = _ret.SEQ) -> np.ndarray:
    """(seq,20) no-vel past-window descriptor: [Lpos3, L6D*ROT, gL, Rpos3, R6D*ROT, gR].
    Identical to gr1_online_retrieval.traj_from_state57 with the velocity block dropped."""
    pl = max(min(win, e), 1)
    seg = s[e - pl:e]
    gL = _ret.grip_from_state57(seg, "L")[:, None] * _ret.GRIP_SCALE
    gR = _ret.grip_from_state57(seg, "R")[:, None] * _ret.GRIP_SCALE
    S = _ret._S_SLICES
    f = np.concatenate([seg[:, S[0]], _ret.ROT_SCALE * seg[:, S[1]], gL,
                        seg[:, S[2]], _ret.ROT_SCALE * seg[:, S[3]], gR], 1)
    return _ret._subsample(f, seq)


def traj_from_proprio_novel(pL, pR, quatL, quatR, gripL, gripR,
                            seq: int = _ret.SEQ) -> np.ndarray:
    """(seq,20) no-vel live-query descriptor from a buffered proprio window.
    Identical to gr1_online_retrieval.traj_from_proprio with the velocity block dropped."""
    pL = np.asarray(pL, float)
    pR = np.asarray(pR, float)
    RL, _ = _ret._rmat(quatL)
    RR, _ = _ret._rmat(quatR)
    gL = np.clip(np.asarray(gripL, float), 0.0, 1.0)[:, None] * _ret.GRIP_SCALE
    gR = np.clip(np.asarray(gripR, float), 0.0, 1.0)[:, None] * _ret.GRIP_SCALE
    R = _ret.ROT_SCALE
    f = np.concatenate([pL, R * RL[:, :, 0], R * RL[:, :, 1], gL,
                        pR, R * RR[:, :, 0], R * RR[:, :, 1], gR], 1)
    return _ret._subsample(f, seq)


_installed = False


def install() -> None:
    """Rebind the base module's descriptor producers to the no-vel versions (idempotent)."""
    global _installed
    if _installed:
        return
    _ret.traj_from_state57 = traj_from_state57_novel
    _ret.traj_from_proprio = traj_from_proprio_novel
    _installed = True
    print("[gr1_online_retrieval_2x] installed NO-VEL descriptors (20-d pos+rot+grip)", flush=True)
