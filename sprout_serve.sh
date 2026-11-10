#!/usr/bin/env bash
# Sprout real-robot policy server (cosmos_policy/scripts/sprout_policy_server.py) for sprout_real/policy.py.
#
#   ./sprout_serve.sh <checkpoint.pt> [extra server flags]
#   EXP=cosmos_predict2p5_2b_sprout_3task_pv0_c16 ./sprout_serve.sh .../sprout_pv0_c16/model_000004000.pt
#   ./sprout_serve.sh --selftest                     # no GPU: the sample builder against training samples
#
# EXP must be the config the checkpoint was trained with (default chunk 32, ours); the server refuses a mismatch of
# chunk size, action dim or state_t. SPROUT_ROOT is the training store (manifest + h5,
# `hf download Jeongeun/sprout-real-3task --repo-type dataset --local-dir ...`). STEPS (default 35) sets the
# sampling steps: on an RTX 5090 a request takes 0.42 s + 0.08 s per step (35: 3.3 s, 10: 1.25 s, 5: 0.84 s);
# policy.py sets its exec size from it (1.5 x the inference left to play), else holds while inferring (sync).
set -euo pipefail
cd "$(dirname "$0")"

export SPROUT_ROOT=${SPROUT_ROOT:-$HOME/WAM/dataset/sprout_ego29_3task}
EXP=${EXP:-cosmos_predict2p5_2b_sprout_3task_pv0_c32}
PORT=${PORT:-5555}
GPU=${GPU:-0}
STEPS=${STEPS:-35}
PY=(uv run -q --extra cu128 --group robocasa --python 3.10 --with pyzmq --with msgpack --with msgpack-numpy)

if [[ "${1:-}" == "--selftest" ]]; then
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 PYTHONPATH=. exec "${PY[@]}" \
    python -m cosmos_policy.scripts.sprout_policy_server --experiment "$EXP" --selftest
fi
CKPT=${1:?usage: ./sprout_serve.sh <checkpoint.pt> [server flags]}
shift
[[ -f "$CKPT" ]] || { echo "no checkpoint at $CKPT" >&2; exit 1; }
[[ -f "$SPROUT_ROOT/manifest/t5_embeddings.pkl" ]] || { echo "no store at $SPROUT_ROOT (manifest/t5_embeddings.pkl)" >&2; exit 1; }

# `uv run --with` puts the extra packages in an overlay env, where Transformer Engine no longer finds the venv's
# cuDNN by itself; CUDNN_HOME points it there. One process, but torch.distributed is initialised: RANK etc. required.
CUDNN_HOME=${CUDNN_HOME:-$PWD/.venv/lib/python3.10/site-packages/nvidia/cudnn} \
CUDA_VISIBLE_DEVICES=$GPU RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=${MPORT:-12370} \
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} PYTHONPATH=. \
  exec "${PY[@]}" python -m cosmos_policy.scripts.sprout_policy_server \
  --experiment "$EXP" --checkpoint "$CKPT" --port "$PORT" --num-steps "$STEPS" ${@+"$@"}
