#!/bin/bash
# Rebuttal runs on 4 nodes x 8 GPUs: two mid-trainings (full query set) and four post-trainings.
# Every stage uses the global batch of its reference, so the reference iteration count and LR schedule are
# used unchanged. Run this script ONCE PER NODE with the same MASTER_ADDR/MASTER_PORT and NODE_RANK=0..3.
# Guide: docs/egoverse_rebuttal_4node.md (data download: docs/egoverse_randref_from_scratch.md sections 1-5)
#
#   STAGE=randref  cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_4node
#                  batch 2/GPU -> global 64, 8K iters (= WATCH v10 state mid, 4 x 8 x 2)
#   STAGE=conf50   cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_4node
#                  batch 2/GPU -> global 64, 8K iters
#   STAGE=rematch|hawor_pairs|hawor_full
#                  cosmos_predict2p5_2b_egoverse_ret_v3_448_state_${STAGE}_nomd_4node (RGB-only hands, docs/egoverse_hawor_rgb.md)
#                  batch 2/GPU -> global 64, 8K iters
#   STAGE=gr1      cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_${MID}_4node   (GR1 v6 ICL)
#                  batch 3/GPU -> global 96, 4K iters on the 8K schedule (= reference 3 x 8 x 4, read at 4K)
#   STAGE=pv0      cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_${MID}_4node
#                  batch 12/GPU -> global 384, 8K iters (= reference 3 x 8 x 16)
#
# Required env (all stages): IMAGINAIRE_OUTPUT_ROOT, MASTER_ADDR, NODE_RANK
#   randref: EGOVERSE_RANDREF_MANIFEST (v10-state manifest + pairs_randref.npz + query_eps_nomd.json), EGOVERSE_HUM_ROOT
#   conf50:  EGOVERSE_CONF50_MANIFEST  (v10-state manifest + pairs_conf50_full.npz + query_eps_nomd.json), EGOVERSE_HUM_ROOT
#   rematch/hawor_pairs: EGOVERSE_HAWOR_MANIFEST, EGOVERSE_HUM_ROOT (tracked); hawor_full: + EGOVERSE_HAWOR_ROOT
#   gr1:     EGOVERSE_GR1_V6_MANIFEST, EGOVERSE_GR1_HUM_ROOT, GR1_BASE_CKPT (mid .pt @ 8K of the MID arm)
#   pv0:     TABLETOP_RANDREF_DATA, TABLETOP_RANDREF_MANIFEST, TABLETOP_RANDREF_CKPT (mid .pt @ 8K of the MID arm)
# Optional env:
#   MID (randref)        gr1/pv0: which mid the post-train config is named after
#                        (randref | conf50 | rematch | hawor_pairs | hawor_full)
#   MASTER_PORT (12341)  NNODES (4)  NPROC_PER_NODE (8)  WANDB_MODE (online)  PY (python)
#   EXTRA_OVERRIDES      extra hydra overrides appended after the experiment (e.g. "trainer.max_iter=20")
set -euo pipefail

STAGE=${STAGE:-randref}
MID=${MID:-randref}
case "$MID" in randref|conf50|rematch|hawor_pairs|hawor_full) ;;
  *) echo "MID must be randref, conf50, rematch, hawor_pairs or hawor_full (got $MID)" >&2; exit 1 ;; esac
NNODES=${NNODES:-4}
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
  randref)
    EXP=cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_4node; BATCH_SIZE=2; GBATCH=64
    : "${EGOVERSE_RANDREF_MANIFEST:?set EGOVERSE_RANDREF_MANIFEST}"; : "${EGOVERSE_HUM_ROOT:?set EGOVERSE_HUM_ROOT}"
    for f in windows.npz pairs_randref.npz query_eps_nomd.json t5_embeddings.pkl action57_stats.json \
             state57_stats.json state57_stats_staterobust.json; do need_file "$EGOVERSE_RANDREF_MANIFEST/$f"; done
    need_dir "$EGOVERSE_HUM_ROOT"
    export EGOVERSE_RANDREF_MANIFEST EGOVERSE_HUM_ROOT ;;
  conf50)
    EXP=cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_4node; BATCH_SIZE=2; GBATCH=64
    : "${EGOVERSE_CONF50_MANIFEST:?set EGOVERSE_CONF50_MANIFEST}"; : "${EGOVERSE_HUM_ROOT:?set EGOVERSE_HUM_ROOT}"
    for f in windows.npz pairs_conf50_full.npz query_eps_nomd.json t5_embeddings.pkl action57_stats.json \
             state57_stats.json state57_stats_staterobust.json; do need_file "$EGOVERSE_CONF50_MANIFEST/$f"; done
    need_dir "$EGOVERSE_HUM_ROOT"
    export EGOVERSE_CONF50_MANIFEST EGOVERSE_HUM_ROOT ;;
  rematch|hawor_pairs|hawor_full)
    EXP=cosmos_predict2p5_2b_egoverse_ret_v3_448_state_${STAGE}_nomd_4node; BATCH_SIZE=2; GBATCH=64
    : "${EGOVERSE_HAWOR_MANIFEST:?set EGOVERSE_HAWOR_MANIFEST}"; : "${EGOVERSE_HUM_ROOT:?set EGOVERSE_HUM_ROOT}"
    PF=pairs_hawor.npz; [ "$STAGE" = rematch ] && PF=pairs_rematch.npz
    for f in windows.npz $PF query_eps_nomd.json t5_embeddings.pkl action57_stats.json \
             state57_stats.json state57_stats_staterobust.json; do need_file "$EGOVERSE_HAWOR_MANIFEST/$f"; done
    if [ "$STAGE" = hawor_full ]; then              # state/action from HaWoR: read the HaWoR episode root
      : "${EGOVERSE_HAWOR_ROOT:?set EGOVERSE_HAWOR_ROOT (hawor_rgb/to_state57.py output)}"
      need_dir "$EGOVERSE_HAWOR_ROOT"; EGOVERSE_HUM_ROOT=$EGOVERSE_HAWOR_ROOT
    fi
    need_dir "$EGOVERSE_HUM_ROOT"
    export EGOVERSE_HAWOR_MANIFEST EGOVERSE_HUM_ROOT ;;
  gr1)
    EXP=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_${MID}_4node; BATCH_SIZE=3; GBATCH=96
    : "${EGOVERSE_GR1_V6_MANIFEST:?set EGOVERSE_GR1_V6_MANIFEST}"; : "${EGOVERSE_GR1_HUM_ROOT:?set EGOVERSE_GR1_HUM_ROOT}"
    : "${GR1_BASE_CKPT:?set GR1_BASE_CKPT (mid .pt)}"
    for f in windows.npz pairs.npz t5_embeddings.pkl action57_stats.json state57_stats_pctl.json \
             abs_action57_stats_pctl.json; do need_file "$EGOVERSE_GR1_V6_MANIFEST/$f"; done
    need_dir "$EGOVERSE_GR1_HUM_ROOT"; need_file "$GR1_BASE_CKPT"
    export EGOVERSE_GR1_V6_MANIFEST EGOVERSE_GR1_HUM_ROOT GR1_BASE_CKPT ;;
  pv0)
    EXP=cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_${MID}_4node; BATCH_SIZE=12; GBATCH=384
    : "${TABLETOP_RANDREF_DATA:?set TABLETOP_RANDREF_DATA}"; : "${TABLETOP_RANDREF_MANIFEST:?set TABLETOP_RANDREF_MANIFEST}"
    : "${TABLETOP_RANDREF_CKPT:?set TABLETOP_RANDREF_CKPT (mid .pt)}"
    for f in windows.npz t5_embeddings.pkl action57_stats.json abs_action20_stats_pctl.json \
             state20_stats_pctl.json; do need_file "$TABLETOP_RANDREF_MANIFEST/$f"; done
    need_dir "$TABLETOP_RANDREF_DATA"; need_file "$TABLETOP_RANDREF_CKPT"
    export TABLETOP_RANDREF_DATA TABLETOP_RANDREF_MANIFEST TABLETOP_RANDREF_CKPT ;;
  *) echo "STAGE must be randref, conf50, rematch, hawor_pairs, hawor_full, gr1 or pv0 (got $STAGE)" >&2; exit 1 ;;
esac

cd "$(dirname "$0")/.."   # cosmos_policy repo root

ngpu=$(nvidia-smi -L | wc -l)
[ "$ngpu" -ge "$NPROC_PER_NODE" ] || { echo "node has $ngpu GPUs < NPROC_PER_NODE=$NPROC_PER_NODE" >&2; exit 1; }
gbatch=$((NNODES * NPROC_PER_NODE * BATCH_SIZE))
[ "$gbatch" -eq "$GBATCH" ] || echo "WARNING: global batch $gbatch != $GBATCH; the iteration budget of $EXP assumes $GBATCH" >&2

mkdir -p "$IMAGINAIRE_OUTPUT_ROOT/pod_logs"
LOG="$IMAGINAIRE_OUTPUT_ROOT/pod_logs/${STAGE}_${MID}_node${NODE_RANK}.log"
echo "[4node:$STAGE] $EXP  node $NODE_RANK/$NNODES  gpus/node $NPROC_PER_NODE  global batch $gbatch  log $LOG"

export IMAGINAIRE_OUTPUT_ROOT WANDB_MODE
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # base ckpt + VAE must already be in the HF cache (from-scratch guide 5.3)

# shellcheck disable=SC2086
$PY -m torch.distributed.run \
  --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py -- \
  experiment="$EXP" \
  dataloader_train.batch_size="$BATCH_SIZE" trainer.grad_accum_iter=1 \
  job.wandb_mode="$WANDB_MODE" \
  ${EXTRA_OVERRIDES:-} 2>&1 | tee -a "$LOG"
