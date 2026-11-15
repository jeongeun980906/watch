"""Training-time demo alignment = the inference retrieval: for every robot frame t, the demo frame h that
``sprout_icl_server.Retriever`` (retargeted torso-frame EEF + gripper, GR1 band, own ghost pinned) picks when a chunk
starts at t. Chunks re-match every 32 frames at inference, so each start offset (t mod 32) is its own replay.

    <root>/manifest/ghost_match_r29.json    {"<task>/<episode>": [h(t) for t in 0..T-1], ...}

SproutIclDataset(match_table=True) reads it: the demo window of a sample starting at t is h(t) (no jitter).

    PYTHONPATH=. python -m cosmos_policy.scripts.sprout_match_table
"""
from __future__ import annotations

import glob
import json
import os
import time

import h5py
import numpy as np

from cosmos_policy.datasets.sprout_dataset import SPROUT_ROOT
from cosmos_policy.scripts import sprout_icl_server as srv

KW = dict(advance="mono", lo=0.5, hi=1.5, pad=8, lock="start", band="gr1", floor=8, cap=96)


def table(root: str, task: str, ep: str, chunk: int = srv.WIN) -> list[int]:
    with h5py.File(os.path.join(root, task, ep + ".h5"), "r") as f:
        s = f["state29"][()].astype(np.float64)
    n = len(s)
    h = np.zeros(n, int)
    for off in range(chunk):
        r = srv.Retriever(root, task, 0, ep, **KW)
        if off:                                    # the replay starting at frame off: its first chunk is matched
            r.next(s[:1], 0.0)                     # (frame 0 is always h = 0)
        for t in range(off, n, chunk):
            _, hh, _ = r.next(s[max(0, t - srv.WIN):t + 1], float(t))
            h[t] = hh
    return h.tolist()


def main():
    srv.DEMO29, srv.MATCH29 = True, True
    root = SPROUT_ROOT
    out, t0 = {}, time.time()
    for task in srv.ARM:
        for f in sorted(glob.glob(os.path.join(root, "human_ghost_r29", task, "*.h5"))):
            ep = os.path.basename(f)[:-3]
            h = table(root, task, ep)
            out[f"{task}/{ep}"] = h
            d = np.abs(np.asarray(h) - np.arange(len(h)))
            print(f"{task}/{ep}: |h - t| median {np.median(d):.0f} p90 {np.percentile(d, 90):.0f} ({time.time() - t0:.0f} s)",
                  flush=True)
    p = os.path.join(root, "manifest", "ghost_match_r29.json")
    json.dump(out, open(p, "w"))
    print(f"{len(out)} episodes -> {p} (grip weight {srv.GRIP_W})")


if __name__ == "__main__":
    main()
