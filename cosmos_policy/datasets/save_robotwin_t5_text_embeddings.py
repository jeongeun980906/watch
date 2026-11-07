# SPDX-License-Identifier: Apache-2.0
"""
Precomputes T5 text embeddings for every RoboTwin task under `dataset_root`.

Instructions are deterministic per task (humanized folder name — see
`_humanize_task_name` in robotwin_dataset.py), so we don't need to load any
HDF5s: we just enumerate task folders that have the requested embodiment and
embed their names. One pkl covers every training split.

Output: `<dataset_root>/t5_embeddings.pkl`  (dict[str, torch.Tensor])

Usage:
    uv run -m cosmos_policy.datasets.save_robotwin_t5_text_embeddings \
        --dataset_root /mnt/ddn/dataset/RoboTwin2.0/dataset
"""

import argparse
import os

from cosmos_policy.datasets.robotwin_dataset import (
    DEFAULT_EMBODIMENT,
    _humanize_task_name,
)
from cosmos_policy.datasets.t5_embedding_utils import (
    generate_t5_embeddings,
    save_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute T5 text embeddings for all RoboTwin tasks."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/ddn/dataset/RoboTwin2.0/dataset",
        help="RoboTwin 2.0 root containing <task>/<embodiment>/data/*.hdf5",
    )
    parser.add_argument(
        "--embodiment",
        type=str,
        default=DEFAULT_EMBODIMENT,
        help="Only embed tasks that have this embodiment subdir. "
             "Kept as a safety net — keeps us from embedding tasks you "
             "haven't actually downloaded the data for.",
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
    assert tasks, (
        f"no tasks under {dataset_root} have an '{embodiment}/data/' subdir"
    )
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
    print(f"Saved embeddings under {out_dir}")


if __name__ == "__main__":
    main()
