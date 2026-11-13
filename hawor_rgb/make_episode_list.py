"""Episode list for the HaWoR job: every human episode of the v10 manifest (train and held-out), longest first,
so the long episodes start early and the tail of the job is made of short ones.

usage: make_episode_list.py --manifest <v10 manifest dir> --video-root $EGOVERSE_HUM_ROOT --out episodes.txt
       [--sources aria,scale,MECKA] [--limit N]   (--limit: smoke test)
"""
import argparse, json, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--sources", default="aria,scale,MECKA")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    W = np.load(f"{a.manifest}/windows.npz", allow_pickle=True)
    ep, emb = W["ep"].astype(str), W["emb"].astype(str)
    eps = sorted(set(ep[np.isin(emb, a.sources.split(","))].tolist()))
    n, missing = {}, []
    for e in eps:
        m = f"{a.video_root}/{e}/images.front_1/zarr.json"
        if not os.path.exists(m): missing.append(e); continue
        n[e] = int(json.load(open(m))["shape"][0])
    order = sorted(n, key=lambda e: -n[e])
    if a.limit: order = order[:a.limit]
    open(a.out, "w").write("\n".join(order) + "\n")
    tot = sum(n[e] for e in order)
    print(f"{len(order)} episodes, {tot:,} frames ({tot / 30 / 3600:.0f} h of video) -> {a.out}")
    if missing:
        print(f"WARNING: {len(missing)} manifest episodes have no images.front_1 under {a.video_root} "
              f"(e.g. {missing[:3]}); they are not in the list")


if __name__ == "__main__":
    main()
