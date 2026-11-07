#!/bin/bash
# Random same-task reference (randref) and noisy-label (conf50) chains on 2 nodes x 8 GPUs:
# mid-training, then two post-trains.
# Run this script ONCE PER NODE with the same MASTER_ADDR/MASTER_PORT and NODE_RANK=0 / 1.
# Full guide: docs/egoverse_randref_from_scratch.md
#
#   STAGE=mid  (default)  cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k
#                         batch 2/GPU -> global 32, 15K iters
#   STAGE=randref_q50     cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_q50_nomd_8k (randref, q50 scale)
#                         batch 2/GPU -> global 32, 8K iters (= q50 scaling arm 4K @ global 64)
#   STAGE=conf50          cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_15k     (noisy task labels, full)
#                         batch 2/GPU -> global 32, 15K iters
#   STAGE=conf50_q50      cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_q50_nomd_8k  (noisy task labels, q50)
#                         batch 2/GPU -> global 32, 8K iters (= q50 scaling arm 4K @ global 64)
#   STAGE=gr1             cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_${MID}_2node   (GR1 v6 ICL)
#                         batch 4/GPU -> global 64, 6K iters (= reference 4K @ 3 nodes), warm start GR1_BASE_CKPT
#   STAGE=pv0             cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_${MID}_2node
#                         batch 16/GPU -> global 256, 12K iters (= reference 8K @ 3 nodes), warm start TABLETOP_RANDREF_CKPT
#
# Required env (all stages): IMAGINAIRE_OUTPUT_ROOT, MASTER_ADDR, NODE_RANK
#   mid, randref_q50: EGOVERSE_RANDREF_MANIFEST (v10-state manifest + pairs_randref.npz + query_eps_nomd.json
#        + query_eps_q50_nomd.json)
#        EGOVERSE_HUM_ROOT         (episode root: images.front_1 + state57 + action57 per episode)
#   conf50, conf50_q50: EGOVERSE_CONF50_MANIFEST (v10-state manifest + pairs_conf50_full.npz +
#        query_eps_nomd.json + query_eps_q50_nomd.json), EGOVERSE_HUM_ROOT
#   gr1: EGOVERSE_GR1_V6_MANIFEST  (<mimicdroid-gr1>/manifest)
#        EGOVERSE_GR1_HUM_ROOT     (<mimicdroid-gr1>/egoverse, unpacked tars)
#        GR1_BASE_CKPT             (mid .pt to warm-start from: randref or conf50)
#   pv0: TABLETOP_RANDREF_DATA     (<tabletopsim-watch>/data)
#        TABLETOP_RANDREF_MANIFEST (<tabletopsim-watch>/manifest)
#        TABLETOP_RANDREF_CKPT     (mid .pt to warm-start from: randref or conf50)
# Optional env:
#   MID (randref)        gr1/pv0: post-train config to use, ..._{randref,randref_q50,conf50,conf50_q50}_2node
#                        (the warm-start .pt itself comes from GR1_BASE_CKPT / TABLETOP_RANDREF_CKPT)
#   MASTER_PORT (12341)  NNODES (2)  NPROC_PER_NODE (8)  WANDB_MODE (online)  PY (python)
#   EXTRA_OVERRIDES      extra hydra overrides appended after the experiment (e.g. "trainer.max_iter=20")
set -euo pipefail

STAGE=${STAGE:-mid}
MID=${MID:-randref}   # gr1/pv0 only: which mid the post-train config is named after (randref | randref_q50 | conf50 | conf50_q50)
case "$MID" in randref|randref_q50|conf50|conf50_q50) ;; *) echo "MID must be randref, randref_q50, conf50 or conf50_q50 (got $MID)" >&2; exit 1 ;; esac
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-12341}
WANDB_MODE=${WANDB_MODE:-online}
PY=${PY:-python}
: "${IMAGINAIRE_OUTPUT_ROOT:?set IMAGINAIRE_OUTPUT_ROOT}"
: "${MASTER_ADDR:?set MASTER_ADDR (node 0 hostname/IP)}"
: "${NODE_RANK:?set NODE_RANK (0..NNODES-1)}"

need_file() { [ -f "$1" ] || { echo "[$STAGE] missing $1" >&2; exit 1; }; }
need_dir()  { [ -d "$1" ] || { echo "[$STAGE] missing directory $1" >&2; exit 1; }; }

case "$STAGE" in
  mid|randref_q50)
    if [ "$STAGE" = mid ]; then EXP=cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k; QSET=query_eps_nomd.json
    else EXP=cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_q50_nomd_8k; QSET=query_eps_q50_nomd.json; fi
    BATCH_SIZE=2; GBATCH=32
    : "${EGOVERSE_RANDREF_MANIFEST:?set EGOVERSE_RANDREF_MANIFEST}"; : "${EGOVERSE_HUM_ROOT:?set EGOVERSE_HUM_ROOT}"
    for f in windows.npz pairs_randref.npz "$QSET" t5_embeddings.pkl action57_stats.json \
             state57_stats.json state57_stats_staterobust.json; do need_file "$EGOVERSE_RANDREF_MANIFEST/$f"; done
    need_dir "$EGOVERSE_HUM_ROOT"
    export EGOVERSE_RANDREF_MANIFEST EGOVERSE_HUM_ROOT ;;
  conf50|conf50_q50)
    if [ "$STAGE" = conf50 ]; then EXP=cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_15k; QSET=query_eps_nomd.json
    else EXP=cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_q50_nomd_8k; QSET=query_eps_q50_nomd.json; fi
    BATCH_SIZE=2; GBATCH=32
    : "${EGOVERSE_CONF50_MANIFEST:?set EGOVERSE_CONF50_MANIFEST}"; : "${EGOVERSE_HUM_ROOT:?set EGOVERSE_HUM_ROOT}"
    for f in windows.npz pairs_conf50_full.npz "$QSET" t5_embeddings.pkl action57_stats.json \
             state57_stats.json state57_stats_staterobust.json; do need_file "$EGOVERSE_CONF50_MANIFEST/$f"; done
    need_dir "$EGOVERSE_HUM_ROOT"
    export EGOVERSE_CONF50_MANIFEST EGOVERSE_HUM_ROOT ;;
  gr1)
    EXP=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_${MID}_2node; BATCH_SIZE=4; GBATCH=64
    : "${EGOVERSE_GR1_V6_MANIFEST:?set EGOVERSE_GR1_V6_MANIFEST}"; : "${EGOVERSE_GR1_HUM_ROOT:?set EGOVERSE_GR1_HUM_ROOT}"
    : "${GR1_BASE_CKPT:?set GR1_BASE_CKPT (randref mid .pt)}"
    for f in windows.npz pairs.npz t5_embeddings.pkl action57_stats.json state57_stats_pctl.json \
             abs_action57_stats_pctl.json; do need_file "$EGOVERSE_GR1_V6_MANIFEST/$f"; done
    need_dir "$EGOVERSE_GR1_HUM_ROOT"; need_file "$GR1_BASE_CKPT"
    export EGOVERSE_GR1_V6_MANIFEST EGOVERSE_GR1_HUM_ROOT GR1_BASE_CKPT ;;
  pv0)
    EXP=cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_${MID}_2node; BATCH_SIZE=16; GBATCH=256
    : "${TABLETOP_RANDREF_DATA:?set TABLETOP_RANDREF_DATA}"; : "${TABLETOP_RANDREF_MANIFEST:?set TABLETOP_RANDREF_MANIFEST}"
    : "${TABLETOP_RANDREF_CKPT:?set TABLETOP_RANDREF_CKPT (randref mid .pt)}"
    for f in windows.npz t5_embeddings.pkl action57_stats.json abs_action20_stats_pctl.json \
             state20_stats_pctl.json; do need_file "$TABLETOP_RANDREF_MANIFEST/$f"; done
    need_dir "$TABLETOP_RANDREF_DATA"; need_file "$TABLETOP_RANDREF_CKPT"
    export TABLETOP_RANDREF_DATA TABLETOP_RANDREF_MANIFEST TABLETOP_RANDREF_CKPT ;;
  *) echo "STAGE must be mid, randref_q50, conf50, conf50_q50, gr1 or pv0 (got $STAGE)" >&2; exit 1 ;;
esac

cd "$(dirname "$0")/.."   # cosmos_policy repo root

ngpu=$(nvidia-smi -L | wc -l)
[ "$ngpu" -ge "$NPROC_PER_NODE" ] || { echo "node has $ngpu GPUs < NPROC_PER_NODE=$NPROC_PER_NODE" >&2; exit 1; }
gbatch=$((NNODES * NPROC_PER_NODE * BATCH_SIZE))
[ "$gbatch" -eq "$GBATCH" ] || echo "WARNING: global batch $gbatch != $GBATCH; the iteration budget of $EXP assumes $GBATCH" >&2

mkdir -p "$IMAGINAIRE_OUTPUT_ROOT/pod_logs"
LOG="$IMAGINAIRE_OUTPUT_ROOT/pod_logs/${STAGE}_node${NODE_RANK}.log"
echo "[2node:$STAGE] $EXP  node $NODE_RANK/$NNODES  gpus/node $NPROC_PER_NODE  global batch $gbatch  log $LOG"

export IMAGINAIRE_OUTPUT_ROOT WANDB_MODE
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # base ckpt + VAE must already be in the HF cache (guide step 5.3)

# shellcheck disable=SC2086
$PY -m torch.distributed.run \
  --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py -- \
  experiment="$EXP" \
  dataloader_train.batch_size="$BATCH_SIZE" trainer.grad_accum_iter=1 \
  job.wandb_mode="$WANDB_MODE" \
  ${EXTRA_OVERRIDES:-} 2>&1 | tee -a "$LOG"
