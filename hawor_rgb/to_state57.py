"""HaWoR episode outputs -> a WATCH episode root whose state57/action57 come from the images.

For every episode with <HAWOR_OUT>/<ep>.npz, writes <OUT_ROOT>/<ep>/state57/state (N,57) and action57/action
(N-1,57) as zarr v3 (float64, chunks of 200 rows, zstd: the layout the training reader expects), and symlinks
the video (images.*) from the video root and meta/ from the tracked state root. The result is a drop-in
EGOVERSE_HUM_ROOT. Hands come from HaWoR (hawor_common.convert); the head pose is the recorded one.
Also writes <OUT_ROOT>/_coverage.json (per-episode share of frames with a HaWoR estimate, per hand).

usage (python >= 3.11, zarr>=3, numpy; cwd = this directory or it on PYTHONPATH):
  to_state57.py --hawor-out $HAWOR_OUT --cam-fit cam_fit.json --manifest <v10 manifest> \
      --state-root <v10-state episodes> --video-root $EGOVERSE_HUM_ROOT --out-root $HAWOR_ROOT [--procs 32]
Re-running skips finished episodes (state and action both present).
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_v] = "1"
import numpy as np
import zarr
from zarr.codecs import ZstdCodec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hawor_common import convert  # noqa: E402

A = None


def write(path, x):
    zarr.create_array(store=path, shape=x.shape, chunks=(200, x.shape[1]), dtype="float64",
                      compressors=ZstdCodec(), fill_value=0.0, overwrite=True)[:] = x


def work(ep):
    o = f"{A.out_root}/{ep}"
    if os.path.exists(f"{o}/state57/state/zarr.json") and os.path.exists(f"{o}/action57/action/zarr.json"):
        return ep, None, "skip"
    try:
        st = np.asarray(zarr.open_array(f"{A.state_root}/{ep}/state57/state", mode="r")[:], np.float64)
        h = np.load(f"{A.hawor_out}/{ep}.npz"); f = FITS[SRC[ep]]
        S, Ac, cov = convert(st, h, f, A.min_cov, source=SRC[ep], smooth=not A.no_smooth, wrist_rot=WROT[SRC[ep]])
        os.makedirs(o, exist_ok=True)
        for g in ("state57", "action57"): zarr.open_group(f"{o}/{g}", mode="a")
        write(f"{o}/state57/state", S); write(f"{o}/action57/action", Ac)
        for d in glob.glob(f"{A.video_root}/{ep}/images.*") + [f"{A.state_root}/{ep}/meta"]:
            lnk = f"{o}/{os.path.basename(d)}"
            if os.path.exists(d) and not os.path.lexists(lnk): os.symlink(os.path.abspath(d), lnk)
        return ep, cov, None
    except Exception as e:  # noqa: BLE001
        return ep, None, repr(e)[:200]


def main():
    global A, FITS, SRC, WROT
    ap = argparse.ArgumentParser()
    for k in ("hawor-out", "cam-fit", "manifest", "state-root", "video-root", "out-root"): ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--procs", type=int, default=32); ap.add_argument("--min-cov", type=float, default=0.3)
    ap.add_argument("--no-smooth", action="store_true", help="skip the EgoVerse keypoint smoothing (default: smoothed, like tracked state57)")
    A = ap.parse_args()
    W = np.load(f"{A.manifest}/windows.npz", allow_pickle=True)
    SRC = dict(zip(W["ep"].astype(str), W["emb"].astype(str)))
    CF = json.load(open(A.cam_fit))
    FITS = {s: (v["f"], np.array(v["R"]), np.array(v["t"])) for s, v in CF.items()}
    WROT = {s: {k: np.array(m) for k, m in v["wrist_rot"].items()} for s, v in CF.items() if "wrist_rot" in v}
    missing = [s for s in FITS if s not in WROT]
    if missing: raise SystemExit(f"cam_fit.json has no wrist_rot for {missing}: re-run fit_cam.py (this version)")
    eps = sorted(e for e in (os.path.basename(p)[:-4] for p in glob.glob(f"{A.hawor_out}/*.npz")) if SRC.get(e) in FITS)
    os.makedirs(A.out_root, exist_ok=True); cov, err = {}, {}
    with Pool(A.procs) as p:
        for k, (ep, c, e) in enumerate(p.imap_unordered(work, eps, chunksize=4)):
            if c is not None: cov[ep] = c
            elif e != "skip": err[ep] = e
            if k % 1000 == 0: print(f"  {k}/{len(eps)}", flush=True)
    old = json.load(open(f"{A.out_root}/_coverage.json")) if os.path.exists(f"{A.out_root}/_coverage.json") else {}
    old.update(cov); json.dump(old, open(f"{A.out_root}/_coverage.json", "w"))
    print(f"converted {len(cov)}, failed {len(err)} {list(err.items())[:3]}; "
          f"mean coverage L {np.mean([c['L'] for c in old.values()]):.2f} R {np.mean([c['R'] for c in old.values()]):.2f}")
    if err: json.dump(err, open(f"{A.out_root}/_failed.json", "w"), indent=1); sys.exit(1)


if __name__ == "__main__":
    main()
