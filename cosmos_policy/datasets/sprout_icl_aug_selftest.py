"""Checks of sprout_icl_aug (CPU):

    PYTHONPATH=. python -m cosmos_policy.datasets.sprout_icl_aug_selftest
"""
from __future__ import annotations

import collections
import importlib.util
import os

import numpy as np

from cosmos_policy.datasets import sprout_icl_aug as A
from cosmos_policy.datasets.sprout_dataset import SPROUT_ROOT

YAWREV = os.path.expanduser("~/WAM/WATCH/quest_capture/pc/make_yaw_reversed.py")


def _ds(cf):
    from cosmos_policy.datasets.sprout_dataset import SproutIclAugDataset, make

    return make(SPROUT_ROOT, cls=SproutIclAugDataset, chunk_size=32, jitter=6, other_prob=0.0, cf_aug=cf)


def main():
    rng = np.random.default_rng(0)
    ds = _ds({"p_tempo": 0.3})
    C = 32
    eps = [int(np.flatnonzero(np.array([str(e).startswith(t) for e in ds.wep]))[0]) for t in A.TASK_ARM]
    for q in eps:
        g = ds._grp(q)
        task = str(ds.wep[q]).split("/")[0]
        side = A.TASK_ARM[task]
        Aa = np.asarray(g.action("abs_action29"), np.float64)
        st = np.asarray(g.state("state29"), np.float64)
        gh = ds._human(str(ds.wep[q]))
        S = np.asarray(gh.state("state57"), np.float64)
        t0, h = 40, 40
        # (1) no factor = the plain sample's rows
        tgt, cur, dem, dend, fend, w, flip = A.apply({}, Aa, st, t0, S, h, C, side, ds.idle)
        assert np.abs(tgt - Aa[t0:t0 + C]).max() < 1e-6 and np.abs(cur - st[t0]).max() < 1e-9
        assert np.abs(dem - S[h + 1:h + 1 + C]).max() < 1e-6 and dend == fend == C and w == 1 and not flip
        # (2) tempo 1 and yaw (1, 0) and wrist 0 are identities
        assert np.abs(A.robot_tempo(Aa, t0, 1.0, C) - Aa[t0:t0 + C]).max() < 1e-6
        seq = S[h:h + C + 1]
        assert np.abs(A.human_yaw(seq, 1.0, 0.0) - seq).max() < 1e-6
        assert np.abs(A.human_wrist(seq, side, np.zeros(3), np.ones(C + 1)) - seq).max() < 1e-5
        # (3) yaw -1 == make_yaw_reversed.reverse_yaw (pivot 0)
        spec = importlib.util.spec_from_file_location("myr", YAWREV)
        myr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(myr)
        assert np.abs(A.human_yaw(seq, -1.0, 0.0) - myr.reverse_yaw(seq, 0)).max() < 1e-6
        # yaw on the robot: angles scaled, everything else kept
        ty = A.robot_yaw(Aa[t0:t0 + C], -1.0, 0.01)
        assert np.allclose(A.base_angle(ty), -A.base_angle(Aa[t0:t0 + C]) + 0.01, atol=1e-9)
        assert np.abs(ty[:, 9:] - Aa[t0:t0 + C, 9:]).max() == 0
        # (4) mirror: twice = the working arm back; base vy / yaw flipped twice = back
        x = Aa[t0:t0 + C]
        mm = A.robot_mirror(A.robot_mirror(x, side, ds.idle), A.OTHER[side], ds.idle)
        a = A.EEF[side]
        assert np.abs(mm[:, a:a + 10] - x[:, a:a + 10]).max() < 1e-6, np.abs(mm[:, a:a + 10] - x[:, a:a + 10]).max()
        assert np.abs(mm[:, 0:9] - x[:, 0:9]).max() < 1e-6
        hm = A.human_mirror(A.human_mirror(seq))
        assert np.abs(hm - seq).max() < 1e-5, np.abs(hm - seq).max()
        # (5) wrist: the rotation between original and new at the chunk end is |w|, position kept
        wv = np.array([0.3, -0.2, 0.1])
        tw = A.robot_wrist(x, side, wv, (np.arange(C) + 1) / C)
        R0, R1 = A.r6(x[-1, a + 3:a + 9]), A.r6(tw[-1, a + 3:a + 9])
        ang = np.arccos(np.clip((np.trace(R1 @ R0.T) - 1) / 2, -1, 1))
        assert abs(ang - np.linalg.norm(wv)) < 1e-6 and np.abs(tw[:, a:a + 3] - x[:, a:a + 3]).max() == 0
        print(f"  {task}: identities, yaw = make_yaw_reversed, mirror involution, wrist angle ok")

    # (6) clipping by the normalisation stats, per factor
    _ts = ds.target_stats if (ds.target_group and ds.target_stats) else ds.state_stats
    clip = collections.defaultdict(list)
    for _ in range(300):
        q = int(ds.query_idx[rng.integers(ds.N)])
        g = ds._grp(q)
        task = str(ds.wep[q]).split("/")[0]
        f = A.draw({}, rng)
        Aa = np.asarray(g.action("abs_action29"), np.float64)
        st = np.asarray(g.state("state29"), np.float64)
        S = np.asarray(ds._human(str(ds.wep[q])).state("state57"), np.float64)
        t0 = int(ds.ww0[q])
        tgt, cur, dem, *_ = A.apply(f, Aa, st, t0, S, min(t0, len(S) - 1), 32, A.TASK_ARM[task], ds.idle)
        s = _ts.get(str(ds.wemb[q]), _ts["_global"])
        hs = ds.human_stats.get("quest", ds.human_stats["_global"])
        zt = np.abs(2 * (tgt - np.asarray(s["lo"])) / (np.asarray(s["hi"]) - np.asarray(s["lo"]) + 1e-8) - 1)
        zh = np.abs(2 * (dem - np.asarray(hs["lo"])) / (np.asarray(hs["hi"]) - np.asarray(hs["lo"]) + 1e-8) - 1)
        ct, ch = float((zt > 4).mean()), float((zh > 4).mean())
        for k in (sorted(f) or ["none"]):
            clip[k if k != "mirror" else f"mirror_{f['mirror']}"].append((ct, ch))
    for k, v in sorted(clip.items()):
        v = np.array(v)
        print(f"  beyond the |z| <= 4 cap [{k:12s}] n={len(v):3d}: target {v[:, 0].mean() * 100:5.2f}% of values, "
              f"demo {v[:, 1].mean() * 100:5.2f}%")

    # (7) the dataset path: augmented samples have the parent's shapes
    from cosmos_policy.datasets.sprout_dataset import SproutIclDataset, make

    base = make(SPROUT_ROOT, cls=SproutIclDataset, chunk_size=32, jitter=6, other_prob=0.0)
    ref = base[0]
    seen = collections.Counter()
    for i in range(24):
        s = ds[int(rng.integers(len(ds)))]
        for k, v in ref.items():
            if hasattr(v, "shape"):
                assert tuple(s[k].shape) == tuple(v.shape), (k, s[k].shape, v.shape)
        assert np.isfinite(np.asarray(s["actions"])).all() and np.isfinite(np.asarray(s["retrieved_actions"])).all()
        seen[(s["cf_aug"], float(s["future_loss_weight"]))] += 1
    print("  dataset samples (factors, future_loss_weight):", dict(seen))
    off = _ds(None)
    import torch

    torch.manual_seed(0); np.random.seed(0)
    a = off[5]
    torch.manual_seed(0); np.random.seed(0)
    b = base[5]
    for k, v in b.items():
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            assert torch.equal(torch.as_tensor(a[k]).float(), torch.as_tensor(v).float()), k
    print("  cf_aug off == SproutIclDataset sample")
    print("sprout_icl_aug selftest ok")


if __name__ == "__main__":
    main()
