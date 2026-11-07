"""Compare raw-HDF5 FK(qpos) vs MG_3000 `base_to_eef` for a drawer task.

Checks that scale and shape of the eef-pose-in-base-frame trajectory look similar.
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cosmos_policy.experiments.robot.robocasa.fk.fk_panda_base import PandaFK

RAW_HDF5 = Path(
    "/mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only/kitchen_drawer/"
    "OpenDrawer/2024-05-03/demo_gentex_im128_randcams.hdf5"
)
MG_ROOT = Path("/mnt/ddn/dataset/robocasa-MG_3000")
MG_TASK = "open the left drawer"


def load_raw_fk_episode(hdf5_path: Path, language_task: str | None = None, demo_idx: int = 0):
    """Pick the `demo_idx`-th demo whose `task_description` == language_task."""
    with h5py.File(hdf5_path, "r") as f:
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1]))
        if language_task is not None:
            def _desc(d):
                v = f["data"][d].attrs.get("task_description", "")
                return v.decode() if isinstance(v, bytes) else v
            demos = [d for d in demos if _desc(d) == language_task]
            if not demos:
                raise RuntimeError(f"no demo with task_description={language_task!r} in {hdf5_path}")
        chosen = demos[demo_idx]
        g = f["data"][chosen]
        task_desc = g.attrs.get("task_description", "")
        if isinstance(task_desc, bytes):
            task_desc = task_desc.decode()
        qpos = g["obs"]["joint_states"][:]          # (T, 7)
        ee_pos_world = g["obs"]["ee_pos"][:]        # (T, 3)
        ee_ori_world = g["obs"]["ee_ori"][:]        # (T, 3)
    fk = PandaFK()
    pos_base, quat_base = fk.fk_batch(qpos)
    return dict(
        demo=chosen,
        task_description=task_desc,
        qpos=qpos,
        fk_pos_base=pos_base,
        fk_quat_base=quat_base,
        ee_pos_world=ee_pos_world,
        ee_ori_world=ee_ori_world,
        T=len(qpos),
    )


def load_mg_episode(mg_root: Path, task_name: str):
    tasks = (pd.read_parquet(mg_root / "meta/tasks.parquet")
               .reset_index().rename(columns={"index": "task_name"}))
    task_idx = int(tasks.loc[tasks["task_name"] == task_name, "task_index"].iloc[0])

    for data_f in sorted((mg_root / "data/chunk-000").glob("*.parquet")):
        df = pd.read_parquet(data_f, columns=["task_index", "episode_index", "frame_index"])
        hit = df[df["task_index"] == task_idx]
        if len(hit):
            ep_idx = int(hit.iloc[0]["episode_index"])
            full = pd.read_parquet(data_f)
            ep = (full[full["episode_index"] == ep_idx]
                    .sort_values("frame_index").reset_index(drop=True))
            state = np.stack(ep["observation.state"].to_numpy())
            return dict(
                episode_index=ep_idx,
                state=state,
                eef_pos_base=state[:, 0:3],
                eef_quat_base=state[:, 3:7],
                T=len(ep),
            )
    raise RuntimeError(f"no MG episode for task {task_name}")


def plot_comparison(raw, mg, out_path: Path):
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(
        f"FK(qpos) vs MG base_to_eef  —  '{MG_TASK}'\n"
        f"(raw: T={raw['T']}, MG: T={mg['T']}, ep={mg['episode_index']})"
    )

    t_raw = np.arange(raw["T"])
    t_mg = np.arange(mg["T"])

    for i, axis_name in enumerate(["x", "y", "z"]):
        ax = axes[i, 0]
        ax.plot(t_raw, raw["fk_pos_base"][:, i], label=f"raw FK.{axis_name}",  color="tab:blue")
        ax.plot(t_mg,  mg["eef_pos_base"][:, i],  label=f"MG state.{axis_name}", color="tab:orange")
        ax.set_title(f"EEF base-frame  pos.{axis_name}")
        ax.set_ylabel("m")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for i, (lbl_raw, lbl_mg) in enumerate([("fk_quat_base", "eef_quat_base")]):
        pass
    for i, cname in enumerate(["qx", "qy", "qz", "qw"]):
        if i < 3:
            ax = axes[i, 1]
        else:
            ax = axes[2, 1]  # overlay qw on the same panel as qz for space
        ax.plot(t_raw, raw["fk_quat_base"][:, i],  label=f"raw FK.{cname}", color="tab:blue", linestyle="--" if i == 3 else "-")
        ax.plot(t_mg,  mg["eef_quat_base"][:, i],  label=f"MG state.{cname}", color="tab:orange", linestyle="--" if i == 3 else "-")
        if i < 3:
            ax.set_title(f"EEF base-frame  quat.{cname}")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    axes[2, 1].set_title("EEF base-frame  quat.qz + qw")
    axes[2, 1].legend(fontsize=8)

    axes[2, 0].set_xlabel("timestep")
    axes[2, 1].set_xlabel("timestep")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    print(f"saved {out_path}")
    return fig


def main():
    print(f"loading raw HDF5 demo with task_description='{MG_TASK}' + running FK …")
    raw = load_raw_fk_episode(RAW_HDF5, language_task=MG_TASK, demo_idx=0)
    print(f"  demo={raw['demo']}  desc={raw['task_description']!r}  T={raw['T']}")
    print(f"  FK pos_base  min/max: {raw['fk_pos_base'].min(0).round(3)} "
          f"/ {raw['fk_pos_base'].max(0).round(3)}")

    print("loading MG_3000 episode …")
    mg = load_mg_episode(MG_ROOT, MG_TASK)
    print(f"  MG ep={mg['episode_index']}  T={mg['T']}")
    print(f"  MG pos_base  min/max: {mg['eef_pos_base'].min(0).round(3)} "
          f"/ {mg['eef_pos_base'].max(0).round(3)}")

    out = Path("/mnt/ddn/tmp/fk_vs_mg_drawer.png")
    plot_comparison(raw, mg, out)


if __name__ == "__main__":
    main()
