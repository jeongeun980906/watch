#!/usr/bin/env bash
# Tabletop-Sim closed-loop eval: policy server (this venv) + dm_control client (sim venv).
#
# The two halves CANNOT share a process — Tabletop-Sim pins numpy<2 / mujoco 3.2.0 while
# cosmos_policy is on numpy 2.x — so this script starts the server, waits for it, runs the
# client, and tears the server down.
#
#   CKPT=/mnt/ddn/tmp/tabletop_sim/.../model_000008000.pt ./tabletop_sim_eval.sh
#   TASKS="aloha_lift_box aloha_dish_drainer" TRIALS=50 ./tabletop_sim_eval.sh
#   OOD=1 ./tabletop_sim_eval.sh          # the 4 *_new generalization tasks
#   GPU=3 PORT=8013 MPORT=29013 ...        # several evals on one node: distinct GPU, PORT and MPORT
#
# Env: MANIFEST (the Tabletop manifest: T5 + action/state stats), SIM_PY (Tabletop-Sim-backup venv python),
#      TABLETOP_RANDREF_DATA / TABLETOP_RANDREF_MANIFEST (read by the *_4node configs at import), START (first
#      episode index; per-episode seeding makes resumes exact), EXEC (16 = the rebuttal tables), GUID, STEPS.
#
# Throughput: ~50 s/episode (env.step is 130 ms, dominated by rendering 3 cameras) plus
# ~12 inferences. 5 tasks x 50 trials is roughly 3.5 h on one GPU.
set -euo pipefail

CKPT=${CKPT:?set CKPT to a .pt from convert_dcp_to_pt.py}
EXP=${EXP:-cosmos_predict2p5_2b_tabletop_sim_5task_short_state}
PORT=${PORT:-8010}
MPORT=${MPORT:-29010}
GPU=${GPU:-0}
MANIFEST=${MANIFEST:-${TABLETOP_RANDREF_MANIFEST:-/mnt/ddn/dataset/tabletopsim-watch/manifest}}
START=${START:-0}
TRIALS=${TRIALS:-50}
SEED=${SEED:-195}
OUT=${OUT:-/mnt/ddn/tmp/tabletop_eval/$(basename "$(dirname "$(dirname "${CKPT}")")")_$(basename "${CKPT}" .pt)}
SIM_PY=${SIM_PY:-/mnt/ddn/Tabletop-Sim/.venv/bin/python}
GRID4=${GRID4:-0}          # must match how the model was trained (run #1 is 3-view)
EXEC=${EXEC:-16}           # --exec-steps. 16 = the rebuttal / blankbc / cotrain / pv2 tables; "" = the
                           # client default (the whole 32-step chunk).
GUID=${GUID:-}             # --guidance. 비우면 서버 기본값 1.5.
STEPS=${STEPS:-}           # --num-steps. 비우면 서버 기본값 35.

if [ "${OOD:-0}" = "1" ]; then
  TASKS=${TASKS:-"aloha_dish_drainer_new aloha_handover_box_new aloha_shoes_table_new aloha_lift_box_new"}
else
  TASKS=${TASKS:-"aloha_dish_drainer aloha_handover_box aloha_shoes_table aloha_lift_box aloha_box_into_pot_easy"}
fi

mkdir -p "${OUT}"
# A stale server on PORT would silently answer with the WRONG checkpoint.
for p in "${PORT}" "${MPORT}"; do
  if (exec 3<>/dev/tcp/127.0.0.1/$p) 2>/dev/null; then
    echo "ERROR: port $p is busy (old server?). Pick PORT/MPORT or kill it." >&2; exit 1
  fi
done
echo "[eval] exp=${EXP}"
echo "[eval] ckpt=${CKPT}"
echo "[eval] tasks=${TASKS} trials=${TRIALS} start=${START} seed=${SEED} exec=${EXEC:-default} out=${OUT}"

server_flags=()
client_flags=()
if [ "${GRID4}" = "1" ]; then server_flags+=(--grid4); client_flags+=(--grid4); fi
if [ -n "${EXEC}" ]; then client_flags+=(--exec-steps "${EXEC}"); fi
if [ -n "${GUID}" ]; then server_flags+=(--guidance "${GUID}"); fi
if [ -n "${STEPS}" ]; then server_flags+=(--num-steps "${STEPS}"); fi

# torch.distributed is initialised even for one process: RANK/WORLD_SIZE/MASTER_* are required.
CUDA_VISIBLE_DEVICES="${GPU}" RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT="${MPORT}" \
PYTHONPATH=. HF_HOME=${HF_HOME:-/mnt/ddn/hf_cache} \
TABLETOP_RANDREF_MANIFEST=${TABLETOP_RANDREF_MANIFEST:-${MANIFEST}} TABLETOP_RANDREF_DATA=${TABLETOP_RANDREF_DATA:-} \
  uv run --extra cu128 --group robocasa --python 3.10 \
  python -m cosmos_policy.scripts.tabletop_sim_eval_server \
  --experiment "${EXP}" --checkpoint "${CKPT}" --manifest-dir "${MANIFEST}" --port "${PORT}" \
  ${server_flags[@]+"${server_flags[@]}"} > "${OUT}/server.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

echo -n "[eval] waiting for server"
for _ in $(seq 1 180); do
  if grep -q "server ready" "${OUT}/server.log" 2>/dev/null; then echo " up"; break; fi
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo " DIED"; tail -30 "${OUT}/server.log"; exit 1
  fi
  sleep 5; echo -n "."
done
grep -E "latent contract|action space|manifest=" "${OUT}/server.log" || true

MUJOCO_GL=egl "${SIM_PY}" tabletop_sim_eval_client.py \
  --server "http://127.0.0.1:${PORT}" \
  --tasks ${TASKS} --episodes "${TRIALS}" --start-episode "${START}" --seed "${SEED}" \
  --out-dir "${OUT}" --save-video \
  ${client_flags[@]+"${client_flags[@]}"} 2>&1 | tee "${OUT}/client.log"

echo "[eval] done -> ${OUT}"
