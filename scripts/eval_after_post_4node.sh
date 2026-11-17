#!/bin/bash
# Closed-loop eval of a 4-node post-train on the same nodes, right after it finishes. Called by
# scripts/train_egoverse_rebuttal_4node.sh when EVAL=1 (STAGE=gr1 or pv0); runs ONCE PER NODE, like the training.
#
#   1. node 0 converts the final DCP checkpoint to .pt and writes $EVAL_OUT/_markers/convert.done;
#      the other nodes wait for that file (never for a process pattern).
#   2. every node runs its share on its 8 GPUs:
#        gr1: the 37 L1-L3 atoms of gr1_local_eval.sh MODE=full, split by ATOM_SHARD=$NODE_RANK/$NNODES
#        pv0: 10 tasks (EASY 5, OOD 4, full box_into_pot) x 50 trials, each task cut into 3 episode ranges
#             (30 jobs; job j goes to node j % NNODES, one GPU per job), protocol of the rebuttal Tabletop tables
#             (seed 195, EXEC 16, guidance 1.5, 35 steps)
#      and writes $EVAL_OUT/_markers/node<rank>.done.
#   3. node 0 waits for every node and writes $EVAL_OUT/summary.txt.
#
#   --check   only verify the eval environment (the launcher runs this before training, so a missing path
#             fails in seconds, not after the post-train)
#
# Env (set by the launcher): STAGE (gr1|pv0), EXP, IMAGINAIRE_OUTPUT_ROOT, NODE_RANK, NNODES, NPROC_PER_NODE.
# Env (eval):
#   gr1: the variables of docs/gr1_eval_rebuttal_4node.md §1.4 (CLIENT_PY, FW, MIMICDROID_TASKDEMOS, POOL,
#        PLAY_POOL, PAIRS_DIR, EGOVERSE_GR1_HUM_ROOT)
#   pv0: SIM_PY (Tabletop-Sim-backup venv python), TABLETOP_RANDREF_MANIFEST
#   EVAL_OUT ($IMAGINAIRE_OUTPUT_ROOT/eval), EVAL_ITER (4000 gr1 / 8000 pv0), EVAL_WAIT_H (12, wait limit)
set -uo pipefail

: "${STAGE:?}" "${EXP:?}" "${IMAGINAIRE_OUTPUT_ROOT:?}" "${NODE_RANK:?}"
NNODES=${NNODES:-4}
NG=${NPROC_PER_NODE:-8}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case "$STAGE" in
  gr1) EVAL_ITER=${EVAL_ITER:-4000} ;;
  pv0) EVAL_ITER=${EVAL_ITER:-8000} ;;
  *) echo "[eval] EVAL=1 is for STAGE=gr1 or pv0 (got $STAGE)" >&2; exit 1 ;;
esac
EVAL_OUT=${EVAL_OUT:-$IMAGINAIRE_OUTPUT_ROOT/eval}
MK=$EVAL_OUT/_markers
CK=$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/cosmos_v2_finetune/$EXP/checkpoints/model_$(printf %09d "$EVAL_ITER").pt
WAIT_S=$(( ${EVAL_WAIT_H:-12} * 3600 ))

need() { [ -e "$1" ] || { echo "[eval] missing $2: $1" >&2; exit 1; }; }
check_env() {
  case "$STAGE" in
    gr1)
      for v in CLIENT_PY FW MIMICDROID_TASKDEMOS POOL PLAY_POOL PAIRS_DIR EGOVERSE_GR1_HUM_ROOT; do
        [ -n "${!v:-}" ] || { echo "[eval] set $v (docs/gr1_eval_rebuttal_4node.md §1.4)" >&2; exit 1; }
        need "${!v}" "$v"
      done ;;
    pv0)
      for v in SIM_PY TABLETOP_RANDREF_MANIFEST; do
        [ -n "${!v:-}" ] || { echo "[eval] set $v (docs/egoverse_vid2robot_4node.md §1.3, scripts/setup_tabletop_sim.sh)" >&2; exit 1; }
        need "${!v}" "$v"
      done ;;
  esac
}
check_env
[ "${1:-}" = "--check" ] && { echo "[eval] $STAGE eval environment OK (node $NODE_RANK)"; exit 0; }

wait_file() {   # path, what
  local t=0
  until [ -e "$1" ]; do
    sleep 30; t=$((t + 30))
    [ $t -ge $WAIT_S ] && { echo "[eval] node $NODE_RANK: gave up waiting for $2 ($1)" >&2; exit 1; }
  done
}

mkdir -p "$MK"
echo "[eval] node $NODE_RANK/$NNODES  stage $STAGE  exp $EXP  ckpt $CK  out $EVAL_OUT  $(date -u +%FT%TZ)"

# ---- 1. convert (node 0), the others wait ----
if [ "$NODE_RANK" = 0 ]; then
  rm -f "$MK"/node*.done
  if [ ! -f "$CK" ]; then
    (cd "$REPO" && IMAGINAIRE_OUTPUT_ROOT=$IMAGINAIRE_OUTPUT_ROOT ${PY:-python} convert_dcp_to_pt.py --tag "$EXP" --iter "$EVAL_ITER") \
      || { echo "[eval] conversion failed" >&2; exit 1; }
  fi
  echo "$CK" > "$MK/convert.done"
else
  wait_file "$MK/convert.done" "the converted checkpoint"
fi
need "$CK" "checkpoint"

# ---- 2. this node's share ----
GPU_LIST=$(seq -s ' ' 0 $((NG - 1)))
case "$STAGE" in
  gr1)
    (cd "$REPO" && CK=$CK EXP=$EXP MODE=full GPUS="$GPU_LIST" PORT0=8700 MPORT0=29600 \
       ATOM_SHARD=$NODE_RANK/$NNODES OUT=$EVAL_OUT/shard$NODE_RANK ./gr1_local_eval.sh) \
      > "$EVAL_OUT/shard$NODE_RANK.log" 2>&1
    ;;
  pv0)
    TASKS_ALL=(aloha_dish_drainer aloha_handover_box aloha_shoes_table aloha_lift_box aloha_box_into_pot_easy
               aloha_dish_drainer_new aloha_handover_box_new aloha_shoes_table_new aloha_lift_box_new aloha_box_into_pot)
    STARTS=(0 17 34); COUNTS=(17 17 16)                     # 50 trials = 3 episode ranges
    jobs=()                                                 # "task start count" of this node
    j=0
    for t in "${TASKS_ALL[@]}"; do
      for p in 0 1 2; do
        [ $((j % NNODES)) = "$NODE_RANK" ] && jobs+=("$t ${STARTS[$p]} ${COUNTS[$p]}")
        j=$((j + 1))
      done
    done
    for g in $(seq 0 $((NG - 1))); do                       # GPU g runs jobs g, g+NG, ...
      ( for ((k = g; k < ${#jobs[@]}; k += NG)); do
          read -r t s n <<< "${jobs[$k]}"
          (cd "$REPO" && CKPT=$CK EXP=$EXP SIM_PY=$SIM_PY TASKS=$t TRIALS=$n START=$s GPU=$g \
             PORT=$((8010 + g)) MPORT=$((29010 + g)) OUT=$EVAL_OUT/$t/ep$s ./tabletop_sim_eval.sh) \
            > "$EVAL_OUT/${t}_ep$s.log" 2>&1 || echo "[eval] node $NODE_RANK: job FAILED $t ep$s (log $EVAL_OUT/${t}_ep$s.log)"
        done ) &
    done
    wait
    ;;
esac
touch "$MK/node$NODE_RANK.done"
echo "[eval] node $NODE_RANK done $(date -u +%FT%TZ)"
[ "$NODE_RANK" = 0 ] || exit 0

# ---- 3. node 0: merge ----
for r in $(seq 0 $((NNODES - 1))); do wait_file "$MK/node$r.done" "node $r"; done
case "$STAGE" in
  gr1)
    {
      echo "# $EXP"; echo "# $CK"; echo "# $(date -u +%FT%TZ)  ($NNODES shards)"
      tot_s=0; tot_j=0
      for f in "$EVAL_OUT"/shard*/logs/task_*.log; do
        [ -e "$f" ] || continue
        s=$(tr -d '\0' < "$f" | grep -acE ' -> SUCCESS'); j=$(tr -d '\0' < "$f" | grep -acE ' -> (SUCCESS|fail)')
        tot_s=$((tot_s + s)); tot_j=$((tot_j + j))
        printf '%-46s %s/%s %s\n' "$(basename "$f" .log | sed 's/^task_//')" "$s" "$j" "$([ "$j" -ge 10 ] && echo ✓)"
      done | sort
      n_atoms=$(ls "$EVAL_OUT"/shard*/logs/task_*.log 2>/dev/null | wc -l)
      echo; echo "atoms $n_atoms/37"
      grep -h "ATOM FAILED" "$EVAL_OUT"/shard*.log 2>/dev/null || true
    } > "$EVAL_OUT/summary.txt"
    # TOTAL over all atom logs (the loop above runs in a pipe, so recount here)
    python3 - "$EVAL_OUT" >> "$EVAL_OUT/summary.txt" <<'PY'
import glob, re, sys
s = j = 0
for f in glob.glob(f"{sys.argv[1]}/shard*/logs/task_*.log"):
    t = open(f, "rb").read().replace(b"\0", b"").decode("utf-8", "replace")
    s += len(re.findall(r" -> SUCCESS", t)); j += len(re.findall(r" -> (?:SUCCESS|fail)", t))
print(f"TOTAL {s}/{j} ({100 * s / max(j, 1):.1f}%)")
PY
    ;;
  pv0)
    python3 - "$EVAL_OUT" > "$EVAL_OUT/summary.txt" <<'PY'
import glob, json, sys
out = sys.argv[1]
easy = ["aloha_dish_drainer", "aloha_handover_box", "aloha_shoes_table", "aloha_lift_box", "aloha_box_into_pot_easy"]
ood = ["aloha_dish_drainer_new", "aloha_handover_box_new", "aloha_shoes_table_new", "aloha_lift_box_new"]
full = ["aloha_box_into_pot"]
ep = {}
for f in glob.glob(f"{out}/*/ep*/results.json"):
    for r in json.load(open(f))["results"]:
        ep[(r["task"], int(r["episode"]))] = bool(r["success"])
print(f"# {out}  (50 trials/task, seed 195, EXEC 16)")
for name, tasks in (("EASY", easy), ("OOD", ood), ("FULL", full)):
    S = N = 0
    for t in tasks:
        v = [ep[k] for k in ep if k[0] == t]; s, n = sum(v), len(v); S += s; N += n
        print(f"{name:4s} {t:26s} {s}/{n}{'' if n == 50 else '  (INCOMPLETE)'}")
    print(f"{name:4s} TOTAL {S}/{N} ({100 * S / max(N, 1):.1f}%)\n")
PY
    grep -h "job FAILED" "$EVAL_OUT"/*.log 2>/dev/null >> "$EVAL_OUT/summary.txt" || true
    ;;
esac
cat "$EVAL_OUT/summary.txt"
echo "[eval] summary -> $EVAL_OUT/summary.txt"
