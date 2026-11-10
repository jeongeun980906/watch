#!/usr/bin/env bash
# Sprout real-robot post-train, pv0 style (native 29-D, 15 Hz, ego view). See
# cosmos_policy/config/experiment/sprout_experiment_configs.py.
#
#   once: T5 embedding of the instructions, written next to the manifest (the 11B encoder; CPU works, ~25 GB RAM)
#       ./sprout_train.sh t5
#   iterations: SPROUT_ITER (default 4000), checkpoints SPROUT_SAVE_ITER (default 1000)
#   check the config only:
#       ./sprout_train.sh --dryrun
#   smoke, then full:
#       ./sprout_train.sh trainer.max_iter=20 dataloader_train.batch_size=2
#       ./sprout_train.sh                                                  # ours, chunk 32, from the WATCH mid
#       TAG=cosmos_predict2p5_2b_sprout_3task_pv0_c16 ./sprout_train.sh       # ours, chunk 16
#       TAG=cosmos_predict2p5_2b_sprout_3task_pv0_c32_base ./sprout_train.sh  # no mid (also _c16_base)
#
# Extra args are hydra overrides. The recipe's batch is 16 per GPU (pv0); on a smaller GPU lower it with
# dataloader_train.batch_size=N.
set -euo pipefail
cd "$(dirname "$0")"

export SPROUT_ROOT=${SPROUT_ROOT:-$HOME/WAM/dataset/sprout_ego29_3task}
export SPROUT_MID_CKPT=${SPROUT_MID_CKPT:-$HOME/WAM/WATCH/ckpt/watch-mid-v10/egoverse_ret_v3_448_state_4node/model_000008000.pt}
export IMAGINAIRE_OUTPUT_ROOT=${IMAGINAIRE_OUTPUT_ROOT:-$HOME/WAM/outputs/sprout}
PY=(uv run --extra cu128 --group robocasa --python 3.10)

if [[ "${1:-}" == "t5" ]]; then
  exec "${PY[@]}" python -m cosmos_policy.datasets.save_tabletop_sim_t5_text_embeddings \
    --manifest_dir "${SPROUT_ROOT}/manifest" --no_ood --no_paraphrase
fi

TAG=${TAG:-cosmos_predict2p5_2b_sprout_3task_pv0_c32}
GPUS=${GPUS:-1}
PORT=${PORT:-12361}
DRY=()
if [[ "${1:-}" == "--dryrun" ]]; then DRY=(--dryrun); shift; fi

[[ -f "${SPROUT_ROOT}/manifest/t5_embeddings.pkl" ]] || { echo "no T5 embeddings: run ./sprout_train.sh t5" >&2; exit 1; }
[[ "$TAG" == *_base* || -f "$SPROUT_MID_CKPT" ]] || { echo "no mid checkpoint at $SPROUT_MID_CKPT" >&2; exit 1; }
echo "[sprout] TAG=${TAG} GPUS=${GPUS} DATA=${SPROUT_ROOT} OUT=${IMAGINAIRE_OUTPUT_ROOT}"

"${PY[@]}" torchrun --nproc_per_node="${GPUS}" --master_port="${PORT}" -m cosmos_policy.scripts.train \
  --config=cosmos_policy/config/config.py ${DRY[@]+"${DRY[@]}"} -- \
  experiment="${TAG}" ${@+"$@"}
