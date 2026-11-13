"""Matching descriptors of every v10 window from an episode root's state57/action57 (tracked or HaWoR).

Index-aligned to <manifest>/windows.npz (N rows). Writes into --out:
  desc_V9.npy (N,40,66) f16, desc_LF.npy (N,) f32   70_3s feat_and_lfrac: DTW sequence + left-hand share
  desc_M.npy  (N,750) f16                             01_3s M  (abs + delta + grasp blocks): cross-source shortlist
  desc_E.npy  (N,3)   f32, desc_G.npy (N,60) f16      01_3s E (extent) and G (grip closedness): same-source shortlist
  desc_valid.npy (N,) bool, pinch.json
The functions are the originals (70_3s.py / 01_3s.py) with one change of input: 01_3s read the raw
obs_ee_pose_transformed + keypoints, here the same quantities come from state57 (eef pos, 6D rot, fingertips in
the wrist frame), so the descriptor of a HaWoR root and of the tracked root are computed identically.

usage (python >= 3.11, zarr>=3, numpy): build_desc.py --root <episodes> --manifest <v10 manifest> --out <dir> [--procs 48]
"""
import argparse, json, os
from collections import Counter, defaultdict
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import zarr

HUMAN = {"MECKA", "scale", "aria"}
WIN = 90
# ---------------- 70_3s.py (verbatim) ----------------
SEQ_TS = 40; FEATD = 66
HANDS = [(slice(9, 12), slice(12, 18), slice(18, 33)), (slice(33, 36), slice(36, 42), slice(42, 57))]
_BW = 33; FING_IDX = list(range(18, 33)) + list(range(18 + _BW, 33 + _BW)); FING_W = 0.25
IDX = np.linspace(0, WIN - 1, SEQ_TS).round().astype(int)


def feat_and_lfrac(st, at, w):
    if w + WIN > len(st) or w + WIN > len(at): return None, None
    s = st[w:w + WIN]; ac = at[w:w + WIN]; cols = []
    for ps, rs, hs in HANDS:
        pos = s[:, ps] - s[0, ps]; rot = s[:, rs]; d9 = ac[:, ps.start:ps.start + 9]; hand = s[:, hs]
        cols += [pos, rot, d9, hand]
    F = np.concatenate(cols, 1)[IDX].astype(np.float32)
    Fn = (F - F.mean(0)) / (F.std(0) + 1e-6); Fn[:, FING_IDX] *= FING_W
    rL = np.linalg.norm(s[:, 9:12] - s[0, 9:12], axis=1).max()
    rR = np.linalg.norm(s[:, 33:36] - s[0, 33:36], axis=1).max()
    return Fn, float(rL / (rL + rR + 1e-6))
# ---------------- 01_3s.py (verbatim blocks) ----------------
TS = 30


def iso(a):
    a0 = a - a.mean(0); return (a0 / (a0.std() + 1e-6)).flatten()


def zf(a): return ((a - a.mean(0)) / (a.std(0) + 1e-6)).flatten()


def sub(a, m=TS):
    idx = np.linspace(0, len(a) - 1, m).round().astype(int); return a[idx]


def span(p): return float(np.linalg.norm(p.max(0) - p.min(0)))


def blocks(fr, w0):
    w1 = w0 + WIN; pR = fr["pR"][w0:w1]; pL = fr["pL"][w0:w1]; RR = fr["RR"][w0:w1]; R0 = RR[0].T
    aR = (R0 @ (pR - pR[0]).T).T; aL = (R0 @ (pL - pL[0]).T).T
    aO = np.einsum('nij,nj->ni', np.transpose(RR, (0, 2, 1)), pL - pR)
    dR = np.diff(aR, axis=0); dL = np.diff(aL, axis=0)
    Mabs = iso(np.concatenate([sub(aR), sub(aL), sub(aO)], axis=1)); Mdel = iso(np.concatenate([sub(dR), sub(dL)], axis=1))
    Mhand = iso(sub(np.concatenate([fr["fingL"][w0:w1], fr["fingR"][w0:w1]], axis=1)))
    M = np.concatenate([Mabs, Mdel, Mhand]).astype(np.float32)
    E = np.log(np.array([np.linalg.norm(dR, axis=1).sum(), np.linalg.norm(dL, axis=1).sum(), span(aO)]) + 1e-3)
    G = zf(sub(np.stack([fr["cL"][w0:w1], fr["cR"][w0:w1]], axis=1)))
    return M, E, G
# -------------------------------------------------------------
ROOT = None; PINCH = None


def rot6(r6):
    a, b = r6[:, 0:3], r6[:, 3:6]
    c0 = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b - (c0 * b).sum(1, keepdims=True) * c0; c1 = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.stack([c0, c1, np.cross(c0, c1)], axis=2)


def load(e):
    st = np.asarray(zarr.open_array(f"{ROOT}/{e}/state57/state", mode="r")[:], np.float64)
    at = np.asarray(zarr.open_array(f"{ROOT}/{e}/action57/action", mode="r")[:], np.float64)
    return st, at


def pinch_job(e):
    try:
        st, _ = load(e)
    except Exception:  # noqa: BLE001
        return None
    out = {}
    for side, o in (("left", 18), ("right", 42)):
        t = st[:, o:o + 15].reshape(-1, 5, 3); val = np.isfinite(t).all((1, 2)) & ~np.all(t.reshape(len(t), -1) == 0, 1)
        if val.sum() >= 20: out[side] = np.linalg.norm(t[val, 0] - t[val, 1], axis=1)
    return out


def work(args):
    e, rows, w0s = args
    try:
        st, at = load(e)
    except Exception as ex:  # noqa: BLE001
        return e, None, repr(ex)[:80]
    tL = st[:, 18:33].reshape(-1, 5, 3); tR = st[:, 42:57].reshape(-1, 5, 3)
    (l10, l90), (r10, r90) = PINCH["left"], PINCH["right"]
    fr = dict(pL=st[:, 9:12], pR=st[:, 33:36], RR=rot6(st[:, 36:42]),
              fingL=np.linalg.norm(tL, axis=2), fingR=np.linalg.norm(tR, axis=2),
              cL=((l90 - np.linalg.norm(tL[:, 0] - tL[:, 1], axis=1)) / (l90 - l10)).clip(0, 1),
              cR=((r90 - np.linalg.norm(tR[:, 0] - tR[:, 1], axis=1)) / (r90 - r10)).clip(0, 1))
    res = []
    for i, w in zip(rows, w0s):
        F, lf = feat_and_lfrac(st, at, int(w))
        if F is None or int(w) + WIN > len(st): continue
        M, E, G = blocks(fr, int(w)); res.append((i, F, lf, M, E, G))
    return e, res, None


def init(root, pinch):
    global ROOT, PINCH
    ROOT, PINCH = root, pinch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); init(a.root, None)
    W = np.load(f"{a.manifest}/windows.npz", allow_pickle=True)
    ep, w0, emb = W["ep"].astype(str), W["w0"], W["emb"].astype(str); N = len(ep)
    hum = np.where(np.isin(emb, list(HUMAN)))[0]
    first = {}
    for i in hum: first.setdefault(ep[i], emb[i])
    have = {e for e in first if os.path.exists(f"{a.root}/{e}/state57/state/zarr.json")}
    print(f"{len(have)}/{len(first)} human episodes present under {a.root}", flush=True)
    cnt = Counter(); sel = []                               # pinch p10/p90: 00_grasp_templates rule
    for e in sorted(have)[::5]:
        if cnt[first[e]] < 60: cnt[first[e]] += 1; sel.append(e)
    with Pool(a.procs, initializer=init, initargs=(a.root, None)) as p:
        rs = [r for r in p.map(pinch_job, sel) if r]
    pinch = {s: [float(x) for x in np.percentile(np.concatenate([r[s] for r in rs if s in r]), [10, 90])] for s in ("left", "right")}
    json.dump(pinch, open(f"{a.out}/pinch.json", "w")); print("pinch", pinch, flush=True)
    groups = defaultdict(list)
    for i in hum:
        if ep[i] in have: groups[ep[i]].append(i)
    jobs = [(e, r, w0[r]) for e, r in groups.items()]
    mm = lambda n, sh, dt: np.lib.format.open_memmap(f"{a.out}/{n}", mode="w+", dtype=dt, shape=(N,) + sh)
    V9, LF, M, E, G = mm("desc_V9.npy", (SEQ_TS, FEATD), np.float16), mm("desc_LF.npy", (), np.float32), \
        mm("desc_M.npy", (750,), np.float16), mm("desc_E.npy", (3,), np.float32), mm("desc_G.npy", (2 * TS,), np.float16)
    VL = np.zeros(N, bool); errs = []
    with Pool(a.procs, initializer=init, initargs=(a.root, pinch)) as p:
        for k, (e, res, err) in enumerate(p.imap_unordered(work, jobs, chunksize=8)):
            if err: errs.append((e, err)); continue
            for i, F, lf, m, ee, g in res: V9[i] = F; LF[i] = lf; M[i] = m; E[i] = ee; G[i] = g; VL[i] = True
            if k % 2000 == 0: print(f"  {k}/{len(jobs)}", flush=True)
    for x in (V9, LF, M, E, G): x.flush()
    np.save(f"{a.out}/desc_valid.npy", VL)
    print(f"done: {VL.sum():,} valid windows of {len(hum):,} human windows, {len(errs)} episode errors {errs[:3]}")


if __name__ == "__main__":
    main()
