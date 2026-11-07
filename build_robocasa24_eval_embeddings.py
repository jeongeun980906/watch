"""Split the 24.8 GB RoboCasa reason1 embeddings pickle into per-task subsets.

`run_robocasa_eval.py` loads the whole `--t5_text_embeddings_path` pickle and moves every
tensor to the eval GPU. For the all-24-task dataset that is 242 instructions x 103 MB =
24.8 GB per process — several minutes of Lustre reads and 24.8 GB of GPU memory that a
single-task eval does not need (e.g. TurnOffMicrowave uses exactly one instruction).

This writes one pickle per task containing only that task's instructions, so each eval
process loads 0.1-4 GB instead.

Usage:
    .venv/bin/python build_robocasa24_eval_embeddings.py \
        --src /mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only/reason1_embeddings.pkl \
        --task-instructions /mnt/ddn/tmp/robocasa24_bc_base/task_instructions.json \
        --out-dir /mnt/ddn/tmp/robocasa24_bc_base/eval_embeddings
"""

import argparse
import json
import os
import pickle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="/mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only/reason1_embeddings.pkl",
        help="Full embeddings pickle (instruction -> tensor).",
    )
    parser.add_argument(
        "--task-instructions",
        default="/mnt/ddn/tmp/robocasa24_bc_base/task_instructions.json",
        help="JSON map {task_name: [instruction, ...]} built from the training HDF5s.",
    )
    parser.add_argument(
        "--out-dir",
        default="/mnt/ddn/tmp/robocasa24_bc_base/eval_embeddings",
        help="Directory to write <TaskName>.pkl into.",
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Subset of tasks (default: all).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.task_instructions) as f:
        task_instructions: dict[str, list[str]] = json.load(f)
    if args.tasks:
        missing = [t for t in args.tasks if t not in task_instructions]
        if missing:
            raise SystemExit(f"Unknown tasks: {missing}")
        task_instructions = {t: task_instructions[t] for t in args.tasks}

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading {args.src} (this reads ~25 GB, expect several minutes)...", flush=True)
    with open(args.src, "rb") as f:
        full = pickle.load(f)
    print(f"Loaded {len(full)} instructions.", flush=True)

    for task, instructions in sorted(task_instructions.items()):
        absent = [i for i in instructions if i not in full]
        if absent:
            raise SystemExit(f"{task}: {len(absent)} instructions missing from {args.src}: {absent[:3]}")
        subset = {i: full[i] for i in instructions}
        out_path = os.path.join(args.out_dir, f"{task}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(subset, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_gb = os.path.getsize(out_path) / 1e9
        print(f"{task:24s} {len(subset):3d} instr  {size_gb:5.2f} GB -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
