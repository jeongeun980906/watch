#!/bin/bash
# HaWoR over all EgoVerse episodes: run this ONCE PER NODE, on as many nodes as you have (4, 8, 16, ...).
# There is no rank or world size: the workers share a claim-based queue in $HAWOR_OUT (see hawor_worker.py), so
# nodes can be added or stopped at any time and re-running resumes. Guide: docs/egoverse_hawor_rgb.md
#
# Required env:
#   HAWOR_REPO        HaWoR checkout with weights/ and _DATA/ (MANO) in place (guide section 2)
#   HAWOR_PY          python of the HaWoR venv
#   EGOVERSE_HUM_ROOT episode root with images.front_1 per episode (same as training)
#   HAWOR_OUT         shared output dir (one <ep>.npz per episode)
#   EPISODES          episode list (make_episode_list.py), the same file on every node
# Optional env:
#   NGPU (all GPUs of the node)  WORKERS_PER_GPU (3; HaWoR needs ~6 GB, 3-6 fit on an 80 GB GPU)
#   STRIDE (1)  SEG (600)  STALE (3600)  TMPDIR (node-local scratch for JPEG segments, ~0.2 GB per worker)
#   MAX_FRAMES (0 = all; smoke test only)
set -euo pipefail
: "${HAWOR_REPO:?}" "${HAWOR_PY:?}" "${EGOVERSE_HUM_ROOT:?}" "${HAWOR_OUT:?}" "${EPISODES:?}"
WATCH_REPO=$(cd "$(dirname "$0")/.." && pwd)
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-3}
STRIDE=${STRIDE:-1}; SEG=${SEG:-600}; STALE=${STALE:-3600}; MAX_FRAMES=${MAX_FRAMES:-0}
mkdir -p "$HAWOR_OUT/_logs"
host=$(hostname)
echo "[hawor] $host: $NGPU GPUs x $WORKERS_PER_GPU workers, stride $STRIDE, out $HAWOR_OUT"
cd "$HAWOR_REPO"
pids=()
for ((g = 0; g < NGPU; g++)); do
  for ((w = 0; w < WORKERS_PER_GPU; w++)); do
    CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$WATCH_REPO:${PYTHONPATH:-}" OMP_NUM_THREADS=4 \
      "$HAWOR_PY" "$WATCH_REPO/hawor_rgb/hawor_worker.py" --episodes "$EPISODES" --video-root "$EGOVERSE_HUM_ROOT" \
      --out "$HAWOR_OUT" --stride "$STRIDE" --seg "$SEG" --stale "$STALE" --max-frames "$MAX_FRAMES" \
      > "$HAWOR_OUT/_logs/${host}_gpu${g}_w${w}.log" 2>&1 &
    pids+=($!)
    sleep 2                                   # stagger model loading
  done
done
echo "[hawor] $host: ${#pids[@]} workers started; logs $HAWOR_OUT/_logs/${host}_*.log"
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
echo "[hawor] $host: all workers exited ($fail with an error)"
