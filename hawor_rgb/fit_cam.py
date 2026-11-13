"""Per-source camera fit for HaWoR: focal length f and one rigid camera -> head transform (Rc, tc) per source.

HaWoR's camera-frame depth scales with the assumed focal length F0; x and y do not. For f on a grid, the wrists
are rescaled and one Kabsch transform (no scale) maps them onto the tracked wrists in the head frame; the f with
the smallest median wrist error wins. Fit on one half of the calibration episodes and report the error on the
other half (split by episode), then refit on all of them for the output.
Then, with that camera fit, one constant wrist-frame rotation C per source and hand: HaWoR's wrist frame (x: wrist ->
middle MCP, z: palm normal) differs from the tracked state57 wrist-frame convention by a fixed rotation
(R_hawor = R_tracked C, chordal mean over the calibration episodes); to_state57.py applies C^T so that the 6D
rotation and the wrist-frame fingertips follow the tracked convention (and the v10 normalization statistics).
Calibration = the first --per-source episodes of each source (sorted ids) that HaWoR has finished; frames are
subsampled to at most --frames-per-episode per episode so long episodes do not dominate.

usage (python >= 3.11 with zarr>=3 and numpy; watch repo on PYTHONPATH):
  fit_cam.py --hawor-out $HAWOR_OUT --state-root <v10-state episodes> --manifest <v10 manifest> --out cam_fit.json
"""
import argparse, glob, json, os

import numpy as np
import zarr

from hawor_common import FGRID, at_focal, convert, kabsch, tracked_wrist_in_head, wrist_rot_offset


def load_pairs(h_path, st, max_frames, rng):
    h = np.load(h_path); F0 = float(h["F0"]); rows = h["frames"].astype(np.int64); keep = rows < len(st)
    out = []
    for s in "LR":
        J = h[f"joints_{s}"][keep].astype(np.float64); tz = h[f"trans_{s}"][keep, 2].astype(np.float64)
        ok = np.isfinite(J[:, 0, 0]) & np.isfinite(tz)
        w = tracked_wrist_in_head(st[rows[keep]], s)
        idx = np.where(ok)[0]
        if len(idx) > max_frames: idx = np.sort(rng.choice(idx, max_frames, replace=False))
        out.append((J[idx, 0], tz[idx], w[idx], F0))
    return out


def stack(data, f):
    P = np.concatenate([at_focal(J[:, None], tz, f, F0)[:, 0] for d in data for J, tz, _, F0 in d])
    Q = np.concatenate([w for d in data for _, _, w, _ in d]); return P, Q


def fit(data):
    best = None
    for f in FGRID:
        P, Q = stack(data, f); R, t = kabsch(P, Q); e = np.median(np.linalg.norm(P @ R.T + t - Q, axis=1))
        if best is None or e < best[0]: best = (float(e), float(f), R, t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hawor-out", required=True); ap.add_argument("--state-root", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--per-source", type=int, default=60); ap.add_argument("--frames-per-episode", type=int, default=400)
    a = ap.parse_args()
    W = np.load(f"{a.manifest}/windows.npz", allow_pickle=True)
    src = dict(zip(W["ep"].astype(str), W["emb"].astype(str)))
    done = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{a.hawor_out}/*.npz"))
    rng = np.random.default_rng(0); res = {}
    for s in ("aria", "scale", "MECKA"):
        eps = [e for e in done if src.get(e) == s][:a.per_source]
        if len(eps) < 4: print(f"{s}: only {len(eps)} finished episodes, skipped"); continue
        data = []; sts = {}
        for e in eps:
            st = np.asarray(zarr.open_array(f"{a.state_root}/{e}/state57/state", mode="r")[:], np.float64); sts[e] = st
            data.append(load_pairs(f"{a.hawor_out}/{e}.npz", st, a.frames_per_episode, rng))
        halves = [data[0::2], data[1::2]]; held = []
        for k in (0, 1):                                   # fit on one half, evaluate on the other
            _, f, R, t = fit(halves[k]); P, Q = stack(halves[1 - k], f)
            held.append(np.linalg.norm(P @ R.T + t - Q, axis=1))
        held = np.concatenate(held)
        e_all, f, R, t = fit(data)
        pairs = []
        for e in eps:                                      # wrist-frame convention offset, with this camera fit
            S, _, cov = convert(sts[e], np.load(f"{a.hawor_out}/{e}.npz"), (f, R, t), source=s)
            if min(cov.values()) >= 0.3: pairs.append((sts[e], S))
        C, resid = wrist_rot_offset(pairs)
        res[s] = dict(f=f, R=R.tolist(), t=t.tolist(), episodes=len(eps), fit_err_mm=round(e_all * 1000, 1),
                      wrist_rot={k: v.tolist() for k, v in C.items()}, wrist_rot_residual_deg={k: round(v, 1) for k, v in resid.items()},
                      heldout_err_mm_med_p90=[round(float(np.median(held)) * 1000, 1), round(float(np.percentile(held, 90)) * 1000, 1)])
        print(f"{s:6s} f={f:.0f} t={np.round(t * 1000).tolist()} mm  held-out wrist error median {res[s]['heldout_err_mm_med_p90'][0]} mm, "
              f"p90 {res[s]['heldout_err_mm_med_p90'][1]} mm  ({len(eps)} episodes)  wrist-frame residual after C "
              f"L {resid['L']:.1f} / R {resid['R']:.1f} deg")
    json.dump(res, open(a.out, "w"), indent=1); print("->", a.out)


if __name__ == "__main__":
    main()
