"""MimicDroid play episodes (the 20 `source == "md"` episodes of the v10 manifest) as EgoVerse episode dirs.

Their video is not in the EgoVerse R2 bucket, so earlier rebuttal mids dropped them (query_eps_nomd.json). This
rebuilds them from the MimicDroid release (HF Rutav/MimicDroidDataset, training/<dir>/demo_im128_notp.hdf5) the
way cosmos_policy/repro/egoverse_pair_builder/10_mimicdroid_convert.py did for WATCH: `images.front_1` = JPEG q92
of `robot0_agentview_center_image` (128x128), one frame per inner chunk. The array is sharded (one file per
episode) like the EgoVerse human videos. state57 / action57 / meta and the group attributes are copied
unchanged from the v10-state episode, so the episodes are exactly the ones WATCH trained on.

Check per episode: frame count == state57 length and the hdf5 eef positions equal state57 (both hands).

usage: build_md_play.py --hdf5-root <MimicDroidDataset/training> --manifest <v10 manifest> --state-root <v10 episodes>
                        --out <dir> [--procs 8]
"""
import argparse, io, os, shutil
from multiprocessing import Pool

import h5py
import numpy as np
import zarr
from PIL import Image

IMG_KEY = "robot0_agentview_center_image"
JPEG_Q = 92


def build(job):
    ep, h5, st_root, out = job
    dirnum, key = ep.split("__", 1)
    src, dst = f"{st_root}/{ep}", f"{out}/{ep}"
    if os.path.exists(f"{dst}/images.front_1/zarr.json"):
        return ep, "skip", 0
    tmp = f"{dst}.tmp"; shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, symlinks=False)                 # zarr.json + state57 + action57 + meta
    st = zarr.open_group(tmp, mode="r")["state57/state"][:]
    o = h5py.File(h5, "r")["data"][key]["obs"]
    n = o[IMG_KEY].shape[0]
    assert n == len(st), f"{ep}: hdf5 {n} frames, state57 {len(st)}"
    for k, sl in (("robot0_base_to_left_eef_pos", slice(9, 12)), ("robot0_base_to_right_eef_pos", slice(33, 36))):
        err = float(np.abs(np.asarray(o[k]) - st[:, sl]).max())
        assert err < 1e-9, f"{ep}: {k} differs from state57 by {err}"
    enc = np.empty(n, dtype=object)
    for i0 in range(0, n, 2000):                             # read in blocks (n x 128 x 128 x 3 is ~1 GB)
        blk = np.asarray(o[IMG_KEY][i0:i0 + 2000])
        for j, im in enumerate(blk):
            b = io.BytesIO(); Image.fromarray(im).save(b, "JPEG", quality=JPEG_Q); enc[i0 + j] = b.getvalue()
    g = zarr.open_group(tmp, mode="r+")
    a = g.create_array(name="images.front_1", shape=(n,), chunks=(1,), shards=(n,),
                       dtype="variable_length_bytes", overwrite=True)
    a[:] = enc
    os.replace(tmp, dst)
    return ep, "ok", n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5-root", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--state-root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    W = np.load(f"{a.manifest}/windows.npz", allow_pickle=True)
    eps = sorted(set(W["ep"].astype(str)[W["source"].astype(str) == "md"].tolist()))
    os.makedirs(a.out, exist_ok=True)
    jobs = [(e, f"{a.hdf5_root}/{e.split('__')[0]}/demo_im128_notp.hdf5", a.state_root, a.out) for e in eps]
    for _, h5, _, _ in jobs:
        assert os.path.exists(h5), f"missing {h5}"
    tot = 0
    with Pool(a.procs) as p:
        for ep, s, n in p.imap_unordered(build, jobs):
            tot += n; print(f"{ep} {s} {n} frames", flush=True)
    print(f"{len(eps)} MimicDroid play episodes in {a.out} ({tot:,} frames written)")


if __name__ == "__main__":
    main()
