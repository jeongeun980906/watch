#!/bin/bash
# Online (simulation) evaluation of the RoboCasa all-24-task Cosmos Policy baseline.
#
# Runs one `run_robocasa_eval.py` process per GPU, pulling tasks off a flock'd queue
# until all 24 atomic tasks are done, then prints a per-task success-rate summary.
#
# Prereqs (already set up on this node — see notes at the bottom):
#   * robocasa-cosmos-policy installed into .venv (editable, --no-deps) + kitchen assets
#   * per-task embedding subsets built by build_robocasa24_eval_embeddings.py
#
# Usage:
#   ./robocasa24_eval.sh                       # 20K ckpt, GPUs 0,1, 50 trials/task
#   GPUS=0,1,2,3 TRIALS=25 ./robocasa24_eval.sh
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}   # this repo
T5_RUN=/mnt/ddn/tmp/robocasa24_bc_t5/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5
CKPT=${CKPT:-$T5_RUN/checkpoints/model_000020000.pt}
EXP=${EXP:-cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5__inference}
DATA=${DATA:-/mnt/ddn/dataset/RoboCasa-Cosmos-Policy/success_only}
OUT=${OUT:-/mnt/ddn/tmp/robocasa24_bc_t5/eval_20000}
GPUS=${GPUS:-0,1}
TRIALS=${TRIALS:-50}
SEED=${SEED:-195}
NOTE=${NOTE:-ckpt20k-seed${SEED}}
# EMB_FILE: one pickle for every task. Correct whenever the whole file is small enough to load
# per task — notably t5_embeddings.pkl (464 MB), the default here. Takes precedence over EMB_DIR.
EMB_FILE=${EMB_FILE:-$DATA/t5_embeddings.pkl}
# EMB_DIR / FULL_EMB: fallback for Cosmos-Reason1 policies, whose pickle is 24.8 GB and gets
# moved onto the eval GPU in its entirety — hence the per-task subsets built by
# build_robocasa24_eval_embeddings.py. Unused while EMB_FILE is set.
EMB_DIR=${EMB_DIR:-$OUT/eval_embeddings}
FULL_EMB=${FULL_EMB:-$DATA/reason1_embeddings.pkl}

# PnP first: those have 17-39 instructions (1.7-4 GB embedding load) and the longest
# horizons, so starting them early keeps the tail of a multi-GPU run short.
TASKS=${TASKS:-"PnPCounterToCab PnPCabToCounter PnPCounterToSink PnPSinkToCounter PnPStoveToCounter \
PnPCounterToStove PnPCounterToMicrowave PnPMicrowaveToCounter TurnOnStove TurnOffStove \
CloseDrawer OpenDrawer CloseSingleDoor OpenSingleDoor TurnSinkSpout CloseDoubleDoor OpenDoubleDoor \
CoffeePressButton CoffeeServeMug CoffeeSetupMug TurnOffMicrowave TurnOnMicrowave TurnOffSinkFaucet TurnOnSinkFaucet"}

LOGS=$OUT/logs
mkdir -p "$LOGS"
QF=$LOGS/queue.txt
# Longest-first so the 30-40 instruction PnP tasks start before the 1-instruction ones.
printf '%s\n' $TASKS > "$QF"
: > "$QF.lock"

next_task() { flock "$QF.lock" -c "head -1 $QF; sed -i '1d' $QF"; }

run_one() {  # $1 = gpu id, $2 = task
    local gpu=$1 task=$2
    # EMB_FILE: one pickle for every task. Correct for T5-conditioned models — the whole
    # t5_embeddings.pkl is 464 MB, so there is nothing to gain from splitting it.
    # Otherwise fall back to the per-task reason1 subsets (the full reason1 pickle is
    # 24.8 GB and run_robocasa_eval moves all of it onto the eval GPU).
    local emb="${EMB_FILE:-}"
    if [ -z "$emb" ]; then
        emb="$EMB_DIR/$task.pkl"
        [ -f "$emb" ] || emb="$FULL_EMB"
    fi
    [ -f "$emb" ] || { echo "ERROR: no embeddings file for $task ($emb)" >&2; return 1; }
    BASE_DATASETS_DIR=/mnt/ddn/dataset \
    HF_HOME=/mnt/ddn/hf_cache HF_HUB_OFFLINE=1 \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES=$gpu \
    "$REPO/.venv/bin/python" -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
        --config "$EXP" \
        --ckpt_path "$CKPT" \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True --num_wrist_images 1 \
        --use_proprio True --normalize_proprio True --unnormalize_actions True \
        --dataset_stats_path "$DATA/dataset_statistics.json" \
        --t5_text_embeddings_path "$emb" \
        --trained_with_image_aug True \
        --chunk_size 32 --num_open_loop_steps 16 \
        --task_name "$task" --num_trials_per_task "$TRIALS" \
        --run_id_note "$NOTE" --local_log_dir "$LOGS" \
        --seed "$SEED" --randomize_seed False --deterministic True \
        --use_variance_scale False --use_jpeg_compression True --flip_images True \
        --num_denoising_steps_action 5 --num_denoising_steps_future_state 1 --num_denoising_steps_value 1 \
        --data_collection False \
        > "$LOGS/$task.log" 2>&1
}

cd "$REPO"
echo ">>> robocasa24 eval start $(date -u +%FT%TZ)  CKPT=$CKPT  GPUS=$GPUS  TRIALS=$TRIALS"
for g in ${GPUS//,/ }; do
    (
        while task=$(next_task); [ -n "$task" ]; do
            echo ">>> [gpu $g] $task ($(date -u +%FT%TZ))"
            run_one "$g" "$task" || echo ">>> [gpu $g] $task FAILED (see $LOGS/$task.log)"
        done
    ) &
done
wait

S=$OUT/summary.txt
{
    echo "==== robocasa all-24 eval: $NOTE ($(date -u +%FT%TZ)) ===="
    echo "ckpt: $CKPT"
    echo "trials/task: $TRIALS  seed: $SEED"
    for task in $TASKS; do
        rate=$(grep -h "^Success rate:" "$LOGS/$task.log" 2>/dev/null | tail -1 | awk '{print $3}')
        printf '%-24s %s\n' "$task" "${rate:-FAILED}"
    done
} | tee "$S"
echo ">>> done. summary: $S"

# ---------------------------------------------------------------------------
# One-time setup that produced the state this script assumes:
#   git clone https://github.com/moojink/robocasa-cosmos-policy.git /mnt/ddn/robocasa-cosmos-policy
#   # kitchen assets reused from /mnt/ddn/mimicdroid-robocasa (identical download URLs):
#   #   objects/ textures/ generative_textures/ symlinked, fixtures/ rsync'd
#   uv pip install --python .venv/bin/python --no-deps -e /mnt/ddn/robocasa-cosmos-policy
#   uv pip install --python .venv/bin/python lxml
#   .venv/bin/python /mnt/ddn/robocasa-cosmos-policy/robocasa/scripts/setup_macros.py
#   .venv/bin/python build_robocasa24_eval_embeddings.py
# ---------------------------------------------------------------------------
