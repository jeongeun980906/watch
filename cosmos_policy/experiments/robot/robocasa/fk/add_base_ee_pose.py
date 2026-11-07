"""Copy success_only/ → success_only2/ and add `obs.base_ee_pose (T, 7)` per demo.

`base_ee_pose` = [eef_pos_base(3), eef_quat_base_xyzw(4)] computed via FK from
`obs.joint_states(T, 7)`.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

from cosmos_policy.experiments.robot.robocasa.fk.fk_panda_base import PandaFK

SRC_ROOT = Path("/mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only")
DST_ROOT = Path("/mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only2")


def _augment_hdf5(hdf5_path: Path, fk: PandaFK) -> int:
    """Open the (already-copied) HDF5 in append mode and add `obs/base_ee_pose`.

    Returns the number of demos processed.
    """
    with h5py.File(hdf5_path, "a") as f:
        demo_names = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        for d in demo_names:
            obs = f[f"data/{d}/obs"]
            if "base_ee_pose" in obs:
                del obs["base_ee_pose"]
            qpos = obs["joint_states"][:]  # (T, 7)
            pos, quat = fk.fk_batch(qpos)
            base_ee_pose = np.concatenate([pos, quat], axis=1).astype(np.float32)
            obs.create_dataset("base_ee_pose", data=base_ee_pose, compression="gzip")
        return len(demo_names)


def process_one(src_hdf5: Path, dst_hdf5: Path, fk: PandaFK, *, overwrite: bool) -> None:
    rel = src_hdf5.relative_to(SRC_ROOT)
    if dst_hdf5.exists() and not overwrite:
        print(f"  SKIP  (exists) {rel}")
        return
    dst_hdf5.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"  COPY  {rel}  ({src_hdf5.stat().st_size / 1e9:.2f} GB)")
    shutil.copy2(src_hdf5, dst_hdf5)
    t1 = time.time()
    n_demos = _augment_hdf5(dst_hdf5, fk)
    t2 = time.time()
    print(
        f"    → copied in {t1-t0:.1f}s, augmented {n_demos} demos in {t2-t1:.1f}s; "
        f"total {t2-t0:.1f}s"
    )


def mirror_non_hdf5(src_root: Path, dst_root: Path) -> None:
    """Copy every non-hdf5 sibling (json/pkl/etc.) preserving dir structure."""
    for p in src_root.rglob("*"):
        if p.is_dir() or p.suffix == ".hdf5":
            continue
        rel = p.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(p, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="substring filter on the source path (e.g. 'kitchen_drawer/OpenDrawer')")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-non-hdf5", action="store_true")
    args = ap.parse_args()

    DST_ROOT.mkdir(parents=True, exist_ok=True)
    fk = PandaFK()

    hdf5s = sorted(SRC_ROOT.rglob("*.hdf5"))
    if args.only:
        hdf5s = [p for p in hdf5s if args.only in str(p)]
    print(f"processing {len(hdf5s)} hdf5 file(s)")

    for i, src in enumerate(hdf5s, 1):
        rel = src.relative_to(SRC_ROOT)
        dst = DST_ROOT / rel
        print(f"[{i}/{len(hdf5s)}] {rel}")
        process_one(src, dst, fk, overwrite=args.overwrite)

    if not args.skip_non_hdf5:
        print("mirroring non-hdf5 sibling files …")
        mirror_non_hdf5(SRC_ROOT, DST_ROOT)

    print("done.")


if __name__ == "__main__":
    main()
