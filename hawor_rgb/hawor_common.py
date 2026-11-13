"""Shared geometry for the HaWoR -> WATCH state57/action57 conversion (numpy only).

state57 layout: head 0:9, left eef 9:18 (pos 9:12, 6D rot 12:18), left hand15 18:33, right eef 33:42
(pos 33:36, rot 36:42), right hand15 42:57. 6D rot = [R[:,0], R[:,1]]; hand15 = 5 fingertips in the wrist frame.
action57 (human): pos = R_w(t)^T (p(t+1) - p(t)), rot = 6D of R_w(t)^T R_w(t+1); one row fewer than state.
Same conversion as responses/yw5w/scripts/rgb_pairing_score.py (hawor_state, HAWOR_SMOOTH=1), on whole episodes.
Smoothing: the tracked state57 went through EgoVerse's smooth_keypoints (median filter + Savitzky-Golay per channel,
aria strong 7/15/3, others weak 5/9/2), so the HaWoR joints get the same filter in the world frame before the wrist
pose and fingertips are derived.
"""
import numpy as np
from scipy.signal import medfilt, savgol_filter

TIPS = [4, 8, 12, 16, 20]
SIDES = {"L": (9, 12, 18), "R": (33, 36, 42)}          # pos, rot, hand15 offsets
FGRID = np.arange(250, 1601, 25)


SMOOTH = {"aria": dict(med_k=7, sg_win=15, sg_poly=3)}      # EgoVerse process_egoverse.py strength per source
SMOOTH_DEFAULT = dict(med_k=5, sg_win=9, sg_poly=2)


def smooth_channels(x, med_k=7, sg_win=15, sg_poly=3):
    """EgoVerse smooth_poses._smooth_channels (verbatim): median filter then Savitzky-Golay along time, per channel."""
    T = x.shape[0]
    if T < 5:
        return x
    mk = min(med_k, T if T % 2 == 1 else T - 1)
    if mk % 2 == 0:
        mk -= 1
    out = np.empty_like(x)
    for c in range(x.shape[1]):
        v = medfilt(x[:, c], kernel_size=max(mk, 1))
        w = min(sg_win, T if T % 2 == 1 else T - 1)
        if w % 2 == 0:
            w -= 1
        if w >= sg_poly + 2:
            v = savgol_filter(v, w, sg_poly)
        out[:, c] = v
    return out


def rot6(r6):
    a, b = r6[..., 0:3], r6[..., 3:6]
    c0 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (c0 * b).sum(-1, keepdims=True) * c0; c1 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=-1)


def six(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], -1)


def kabsch(P, Q):
    """R, t minimizing |R P + t - Q| (rows are points)."""
    mp, mq = P.mean(0), Q.mean(0); H = (P - mp).T @ (Q - mq); U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def at_focal(J, tz, f, F0):
    """HaWoR camera-frame joints with the focal length f instead of F0 (depth scales with f, x/y do not)."""
    J = J.copy(); J[..., 2] += (tz * (f / F0 - 1.0))[..., None]; return J


def tracked_wrist_in_head(st, side):
    """Tracked wrist position in the head frame, (N,3): R_h^T (p_w - p_h)."""
    p, _, _ = SIDES[side]; Rh = rot6(st[:, 3:9])
    return np.einsum("nji,nj->ni", Rh, st[:, p:p + 3] - st[:, 0:3])


def interp_rows(x, ok, n_out, rows):
    """Values x (k,...) known at source rows `rows[ok]` -> (n_out,...) by linear interpolation; edges held."""
    t = np.arange(n_out); y = x.reshape(len(x), -1)[ok]; r = rows[ok]
    out = np.empty((n_out, y.shape[1]))
    for k in range(y.shape[1]): out[:, k] = np.interp(t, r, y[:, k])
    return out.reshape((n_out,) + x.shape[1:])


def convert(st, h, fit, min_cov=0.3, source=None, smooth=True):
    """Tracked state57 (N,57) + HaWoR episode output -> HaWoR state57 (N,57), action57 (N-1,57), coverage.

    The head pose (0:9) is kept from the recording: the camera is fixed to the head, so HaWoR's camera frame is
    the head frame up to the per-source fit (f, Rc, tc). Hands are overwritten; a hand observed in fewer than
    `min_cov` of the frames is held at its tracked first-frame pose (it barely appears in the video).
    """
    N = len(st); S = st.copy(); ph = st[:, 0:3]; Rh = rot6(st[:, 3:9])
    f, Rc, tc = fit; F0 = float(h["F0"]); rows = np.asarray(h["frames"]).astype(np.int64)
    sel = rows < N; rows = rows[sel]; cov = {}
    for side, (p, r, hnd) in SIDES.items():
        J = np.asarray(h[f"joints_{side}"], np.float64)[sel]; tz = np.asarray(h[f"trans_{side}"], np.float64)[sel, 2]
        ok = np.isfinite(J[:, 0, 0]) & np.isfinite(tz); cov[side] = float(ok.mean()) if len(ok) else 0.0
        if ok.sum() < max(2, min_cov * len(ok)):
            S[:, p:p + 3] = st[0, p:p + 3]; S[:, r:r + 6] = st[0, r:r + 6]; S[:, hnd:hnd + 15] = st[0, hnd:hnd + 15]
            continue
        J = at_focal(np.nan_to_num(J), np.nan_to_num(tz), f, F0) @ Rc.T + tc             # head frame
        J = interp_rows(J, ok, N, rows)
        Jw = ph[:, None] + np.einsum("nij,nkj->nki", Rh, J)                                 # world frame
        if smooth:
            Jw = smooth_channels(Jw.reshape(N, 63), **SMOOTH.get(source, SMOOTH_DEFAULT)).reshape(N, 21, 3)
        x = Jw[:, 9] - Jw[:, 0]; x /= np.linalg.norm(x, axis=1, keepdims=True)
        z = np.cross(Jw[:, 5] - Jw[:, 0], Jw[:, 17] - Jw[:, 0]); z /= np.linalg.norm(z, axis=1, keepdims=True)
        y = np.cross(z, x); Rw = np.stack([x, y, z], -1)                                    # wrist frame in world
        S[:, p:p + 3] = Jw[:, 0]
        S[:, r:r + 6] = six(Rw)
        S[:, hnd:hnd + 15] = np.einsum("nji,nkj->nki", Rw, Jw[:, TIPS] - Jw[:, :1]).reshape(N, 15)
    A = np.zeros((N - 1, 57))
    for p, r, _ in SIDES.values():
        Rw = rot6(S[:, r:r + 6])
        A[:, p:p + 3] = np.einsum("nji,nj->ni", Rw[:-1], S[1:, p:p + 3] - S[:-1, p:p + 3])
        A[:, r:r + 6] = six(np.einsum("nji,njk->nik", Rw[:-1], Rw[1:]))
    return S, A, cov
