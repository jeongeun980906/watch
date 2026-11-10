#!/bin/bash
# Sprout real-robot post-train (context-free, pv0 style) on 2 nodes x 8 GPUs. Six runs, one 2-node group each.
# Run this script ONCE PER NODE of a group with the group's MASTER_ADDR/MASTER_PORT and NODE_RANK=0..1.
# Guide: docs/real-robot-context-free.md
#
#   ARM=c32        cosmos_predict2p5_2b_sprout_3task_pv0_c32        chunk 32, from the WATCH mid   (ours)
#   ARM=c16        cosmos_predict2p5_2b_sprout_3task_pv0_c16        chunk 16, from the WATCH mid   (ours)
#   ARM=c32_within cosmos_predict2p5_2b_sprout_3task_pv0_c32_within chunk 32, from the within-video mid (baseline)
#   ARM=c16_within cosmos_predict2p5_2b_sprout_3task_pv0_c16_within chunk 16, from the within-video mid (baseline)
#   ARM=c32_base   cosmos_predict2p5_2b_sprout_3task_pv0_c32_base   chunk 32, no mid (Cosmos-Predict2.5 base)
#   ARM=c16_base   cosmos_predict2p5_2b_sprout_3task_pv0_c16_base   chunk 16, no mid
#   all: batch 8/GPU -> global 128, 4K iterations (49 epochs of the 10,358 windows), a checkpoint every 1K
#
# Required env: IMAGINAIRE_OUTPUT_ROOT, MASTER_ADDR, NODE_RANK, SPROUT_ROOT (the HF dataset, h5 + manifest/)
#               SPROUT_MID_CKPT (the WATCH mid .pt) for c32/c16, SPROUT_WITHIN_CKPT (the within-video mid .pt)
#               for *_within
# Optional env: MASTER_PORT (12351)  NNODES (2)  NPROC_PER_NODE (8)  WANDB_MODE (online)  PY (python)
#               EXTRA_OVERRIDES   extra hydra overrides after the experiment (e.g. "trainer.max_iter=20")
set -euo pipefail

ARM=${ARM:-c32}
case "$ARM" in c32|c16|c32_within|c16_within|c32_base|c16_base) ;;
  *) echo "ARM must be c32, c16, c32_within, c16_within, c32_base or c16_base (got $ARM)" >&2; exit 1 ;; esac
EXP=cosmos_predict2p5_2b_sprout_3task_pv0_${ARM}
BATCH_SIZE=8; GBATCH=128
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-12351}
WANDB_MODE=${WANDB_MODE:-online}
PY=${PY:-python}
: "${IMAGINAIRE_OUTPUT_ROOT:?set IMAGINAIRE_OUTPUT_ROOT}"
: "${MASTER_ADDR:?set MASTER_ADDR (node 0 of this group)}"
: "${NODE_RANK:?set NODE_RANK (0..NNODES-1)}"
: "${SPROUT_ROOT:?set SPROUT_ROOT (the downloaded sprout-real-3task dataset)}"

need_file() { [ -f "$1" ] || { echo "[$ARM] missing $1" >&2; exit 1; }; }
for f in windows.npz pairs.npz tasks.json t5_embeddings.pkl abs_action29_stats_pctl.json state29_stats_pctl.json; do
  need_file "$SPROUT_ROOT/manifest/$f"
done
for t in sprout_cup_to_drawer sprout_open_drawer sprout_cup_close_drawer; do
  n=$(ls "$SPROUT_ROOT/$t"/episode_*.h5 2>/dev/null | wc -l)
  [ "$n" -eq 20 ] || { echo "[$ARM] $SPROUT_ROOT/$t has $n episodes, expected 20" >&2; exit 1; }
done
case "$ARM" in
  c32|c16)   : "${SPROUT_MID_CKPT:?set SPROUT_MID_CKPT (WATCH mid .pt)}"; need_file "$SPROUT_MID_CKPT"; export SPROUT_MID_CKPT ;;
  *_within)  : "${SPROUT_WITHIN_CKPT:?set SPROUT_WITHIN_CKPT (within-video mid .pt)}"; need_file "$SPROUT_WITHIN_CKPT"
             export SPROUT_WITHIN_CKPT ;;
esac

cd "$(dirname "$0")/.."   # cosmos_policy repo root

ngpu=$(nvidia-smi -L | wc -l)
[ "$ngpu" -ge "$NPROC_PER_NODE" ] || { echo "node has $ngpu GPUs < NPROC_PER_NODE=$NPROC_PER_NODE" >&2; exit 1; }
gbatch=$((NNODES * NPROC_PER_NODE * BATCH_SIZE))
[ "$gbatch" -eq "$GBATCH" ] || echo "WARNING: global batch $gbatch != $GBATCH; the 4K budget of $EXP assumes $GBATCH" >&2

mkdir -p "$IMAGINAIRE_OUTPUT_ROOT/pod_logs"
LOG="$IMAGINAIRE_OUTPUT_ROOT/pod_logs/sprout_${ARM}_node${NODE_RANK}.log"
echo "[sprout2node:$ARM] $EXP  node $NODE_RANK/$NNODES  gpus/node $NPROC_PER_NODE  global batch $gbatch  log $LOG"

export IMAGINAIRE_OUTPUT_ROOT WANDB_MODE SPROUT_ROOT
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # base ckpt + VAE must already be in the HF cache

# every arm needs both (the configs resolve the base when they load): fail here with the fix, not in torchrun
$PY - <<'PYEOF' || exit 1
import sys
from huggingface_hub import hf_hub_download
repo = "nvidia/Cosmos-Predict2.5-2B"
files = ("base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt", "tokenizer.pth")
missing = []
for f in files:
    try:
        hf_hub_download(repo, f, local_files_only=True)
    except Exception:
        missing.append(f)
if missing:
    sys.exit(f"[sprout] not in the HF cache (HF_HOME): {missing}\n"
             f"  accept the license on https://huggingface.co/{repo}, `hf auth login`, then\n"
             f"  hf download {repo} {' '.join(files)}")
PYEOF

# shellcheck disable=SC2086
$PY -m torch.distributed.run \
  --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py -- \
  experiment="$EXP" \
  dataloader_train.batch_size="$BATCH_SIZE" trainer.grad_accum_iter=1 \
  job.wandb_mode="$WANDB_MODE" \
  ${EXTRA_OVERRIDES:-} 2>&1 | tee -a "$LOG"
