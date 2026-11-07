# SPDX-License-Identifier: Apache-2.0
"""Precompute T5 text embeddings for the real-robot dataset.

Task names follow the same convention as RoboTwin (humanized folder name —
see ``_humanize_task_name`` in robotwin_dataset.py). For real-robot we
usually have a single task ("bg2 tabletop"); this script just walks the
``<dataset_root>/<task>/<embodiment>/data/`` layout and embeds every task it
finds with the requested embodiment.

Output: ``<dataset_root>/t5_embeddings.pkl``  (dict[str, torch.Tensor])

Usage:
    uv run --extra cu128 --group robocasa --python 3.10 \\
        -m cosmos_policy.datasets.save_realrobot_t5_text_embeddings \\
        --dataset_root /mnt/ddn/dataset/real_robot/h5
"""
from __future__ import annotations

import argparse
import os

from cosmos_policy.datasets.real_robot_dataset import DEFAULT_REALROBOT_EMBODIMENT
from cosmos_policy.datasets.robotwin_dataset import _humanize_task_name
from cosmos_policy.datasets.t5_embedding_utils import (
    generate_t5_embeddings,
    save_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute T5 text embeddings for all real-robot tasks.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/ddn/dataset/real_robot/h5",
        help="Root containing <task>/<embodiment>/data/*.hdf5",
    )
    parser.add_argument(
        "--embodiment",
        type=str,
        default=DEFAULT_REALROBOT_EMBODIMENT,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to save t5_embeddings.pkl (default: dataset_root).",
    )
    return parser.parse_args()


def _collect_tasks(dataset_root: str, embodiment: str) -> list[str]:
    tasks = sorted(
        name for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name, embodiment, "data"))
    )
    assert tasks, f"no tasks with embodiment={embodiment!r} under {dataset_root}"
    return tasks


def main() -> None:
    args = parse_args()
    tasks = _collect_tasks(args.dataset_root, args.embodiment)
    commands = sorted({_humanize_task_name(t) for t in tasks})
    print(f"Found {len(commands)} task(s) with embodiment={args.embodiment!r}:")
    for c in commands:
        print(f"  - {c!r}")

    t5_text_embeddings = generate_t5_embeddings(commands)
    out_dir = args.out_dir or args.dataset_root
    os.makedirs(out_dir, exist_ok=True)
    save_embeddings(t5_text_embeddings, out_dir)
    print(f"saved embeddings under {out_dir}")


if __name__ == "__main__":
    main()
