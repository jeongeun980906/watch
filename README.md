# WATCH: Learning Transferable World-Action Models from Task-Paired Human Videos

Training code for WATCH: cross-video pre-training on task-paired human videos (EgoVerse), then robot
post-training on five settings. The code is built on
[Cosmos Policy](https://github.com/NVlabs/cosmos-policy) with the Cosmos-Predict2.5-2B backbone.

- [1. Setup](#1-setup)
- [2. Data and checkpoints](#2-data-and-checkpoints)
- [3. Pre-training (main model)](#3-pre-training-main-model)
- [4. Post-training: the five settings](#4-post-training-the-five-settings)
- [5. Evaluation](#5-evaluation)
- [6. Rebuttal experiments](#6-rebuttal-experiments)
- [7. Repository layout](#7-repository-layout)

---

## 1. Setup

Python 3.10, CUDA 12.8, torch 2.7. Use the Docker image ([SETUP.md](SETUP.md)) or uv directly:

```bash
git clone https://github.com/jeongeun980906/watch.git && cd watch
uv sync --extra cu128 --group egoverse --python 3.10
source .venv/bin/activate
python -c "import torch, megatron.core, numcodecs; print(torch.__version__, torch.cuda.device_count())"
```

Every run starts from the Cosmos-Predict2.5-2B post-trained checkpoint and its Wan2.1 VAE. Accept the model
license on Hugging Face, then cache both files on every node (or on a shared `HF_HOME`):

```bash
hf download nvidia/Cosmos-Predict2.5-2B \
    base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt tokenizer.pth
```

T5 is not needed. Text embeddings are precomputed in each manifest (`t5_embeddings.pkl`).

## 2. Data and checkpoints

| what | where | used by |
|---|---|---|
| EgoVerse v10 state/action arrays + pairing manifest (`pairs.npz`, 1,045,007 query windows) | HF `Jeongeun/egoverse-v10-state` | pre-training |
| EgoVerse episode videos (13,449 human episodes, ~3.9 TB) | EgoVerse R2 bucket, via [`EgoVerse-backup/watch_download`](https://github.com/jeongeun980906/EgoVerse-backup/tree/backup-snapshot/watch_download) | pre-training |
| MimicDroid GR1 post-training data (v6 manifest + zarr episodes) | HF `Jeongeun/mimicdroid-gr1` | settings 1, 3 |
| MimicDroid GR1 with 2x-speed references (v6 2x manifest + episodes) | HF `Jeongeun/mimicdroid-gr1-2x` | setting 2 |
| TableTop-Sim (ALOHA, 5 tasks) | HF `Jeongeun/tabletopsim-watch` | setting 4 |
| RoboCasa (50 human demos per task, 24 tasks) | HF `nvidia/RoboCasa-Cosmos-Policy` ([ROBOCASA.md](ROBOCASA.md)) | setting 5 |
| **WATCH pre-trained model** (8K) | HF `Jeongeun/cosmos-policy-egoverse-mid-v10`, `egoverse_ret_v3_448_state_4node/model_000008000.pt` | all post-training |
| within-video baseline (8K) | same repo, `egoverse_ret_v3_448_state_blankret_bv2_v10_4node/model_000008000.pt` | baseline |

Download and assembly of the EgoVerse data (state arrays + video, then linking them into one episode root and
checking completeness) are written out step by step in
[docs/egoverse_randref_from_scratch.md](docs/egoverse_randref_from_scratch.md), steps 1, 3 and 4. For the
main model, run step 4 with `--pairs pairs.npz`.

The post-training datasets download the same way:

```bash
for d in mimicdroid-gr1 mimicdroid-gr1-2x tabletopsim-watch; do
  hf download Jeongeun/$d --repo-type dataset --local-dir $ROOT/$d
done
cd $ROOT/mimicdroid-gr1 && mkdir -p egoverse && ls data/*.tar | xargs -P 8 -I{} tar -xf {} -C egoverse
```

If `mimicdroid-gr1-2x` also ships episode tars, unpack them the same way and point `EGOVERSE_GR1_HUM_ROOT` at
that `egoverse/` for setting 2.

Paths are passed through environment variables. The defaults are the cluster paths.

| env var | points to |
|---|---|
| `EGOVERSE_V10_MANIFEST` | `<egoverse-v10-state>/manifest` |
| `EGOVERSE_HUM_ROOT` | the assembled EgoVerse episode root (video + `state57`/`action57` per episode) |
| `EGOVERSE_GR1_V6_MANIFEST`, `EGOVERSE_GR1_HUM_ROOT` | `<mimicdroid-gr1>/manifest`, `<mimicdroid-gr1>/egoverse` (unpacked tars) |
| `EGOVERSE_GR1_V6_2X_MANIFEST` | `<mimicdroid-gr1-2x>/manifest` (setting 2) |
| `TABLETOP_ROOT` | TableTop-Sim root: episodes, with the manifest at `$TABLETOP_ROOT/manifest` |
| `BASE_DATASETS_DIR` | the directory that holds `RoboCasa-Cosmos-Policy/` |
| `GR1_BASE_CKPT`, `TABLETOP_BASE_CKPT` | warm-start `.pt` for the GR1 and TableTop-Sim post-training configs |

With the HF layout of `tabletopsim-watch` (`data/` and `manifest/` side by side), point `TABLETOP_ROOT` at
`data/` and link the manifest into it: `ln -s ../manifest <tabletopsim-watch>/data/manifest`.

## 3. Pre-training (main model)

Experiment: **`cosmos_predict2p5_2b_egoverse_ret_v3_448_state`**
(`cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py`).

- 448 px front camera, 92-frame (3 s) windows at 30 Hz, absolute `state57` chunks.
- Retrieval layout v3: each query window is trained with one of its up to 20 motion-matched references
  from `pairs.npz`, drawn uniformly per sample. The reference's video and actions fill the reference
  slots. The model predicts the query's actions and future video.
- 4 nodes x 8 GPUs, batch 2 per GPU (global 64). LR 1e-4, 500-step warmup, decay over a 50K cycle to 0.3
  of the peak. **The paper's model is the 8K checkpoint**. The config's `max_iter` is 20K, so stop at, or
  read, iteration 8,000.

```bash
# on every node (NODE_RANK = 0..3)
export EGOVERSE_V10_MANIFEST=$ROOT/egoverse-v10-state/manifest EGOVERSE_HUM_ROOT=$ROOT/egoverse-video
export IMAGINAIRE_OUTPUT_ROOT=/shared/runs/watch_mid HF_HUB_OFFLINE=1
python -m torch.distributed.run --nnodes=4 --nproc_per_node=8 --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=12341 \
    -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py -- \
    experiment=cosmos_predict2p5_2b_egoverse_ret_v3_448_state \
    dataloader_train.batch_size=2 trainer.grad_accum_iter=1 trainer.max_iter=8000
```

Expected in the rank-0 log: `[EgoVerseRetDataset] pairs=1045007 num_frames=92 R=23 chunk_duration=50`, and
`ROBUST state norm ... state57_stats_staterobust.json`. A fresh EgoVerse download has no video for the 20
MimicDroid play episodes (1.55% of query rows). To train without them, add
`dataloader_train.dataset.query_episode_set=$EGOVERSE_V10_MANIFEST/query_eps_nomd.json` (file on HF
`Jeongeun/egoverse-v10-randref`).

Convert the DCP checkpoint to a `.pt` for post-training:

```bash
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/watch_mid \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state --iter 8000
export WATCH_MID=/shared/runs/watch_mid/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state/checkpoints/model_000008000.pt
```

Alternatively, use the released `model_000008000.pt` from section 2.

## 4. Post-training: the five settings

All five keep the pre-training objective and slot layout, start from the WATCH model, and update the full
transformer. Post-training predicts future video and actions, without the future-proprioception or rollout
terms of the original Cosmos Policy recipe.

| # | setting (paper) | experiment | data | nodes x GPUs x batch | iterations |
|---|---|---|---|---|---|
| 1 | MimicDroid, in-context (GR1) | `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state` | GR1 v6 | 3 x 8 x 4 (96) | 8K |
| 2 | MimicDroid, in-context, 2x-speed references | `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state` | GR1 v6 2x | 3 x 8 x 4 (96) | 8K |
| 3 | MimicDroid, context-free | `cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6` | GR1 v6 | 3 x 8 x 4 (96) | 8K |
| 4 | TableTop-Sim, context-free (ALOHA) | `cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0` | TableTop-Sim | 3 x 8 x 16 (384) | 8K |
| 5 | RoboCasa Kitchen-H50, context-free (Franka) | `cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2` | RoboCasa | 3 x 8 x 25 | 20K read (50K schedule) |

- **In-context (1, 2):** the reference slots hold a retrieved demonstration of the same task (its video and
  actions). For setting 2 the reference plays at 2x speed.
- **Context-free (3-5):** the reference slots are removed, so the policy sees only its own observation.
  - Setting 3 uses the same GR1 data as setting 1.
  - Setting 4 is the per-view (`pv0`), 224 px, `a20` action layout.
  - Setting 5 uses the bridge layout: a zero-pixel reference prefix so that the EgoVerse slot layout
    carries over, and no future proprioception.
- **Baselines** (post-training the original video model) are the `_base` variants of the same
  configs, for example `..._gr1_plain_state_v6_base`. Swapping the warm start for the within-video
  checkpoint gives the within-video baseline.

Launch (one command per node; `NNODES=3`). The warm start is the WATCH `.pt`:

```bash
COMMON="--nnodes=3 --nproc_per_node=8 --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=12341
        -m cosmos_policy.scripts.train --config=cosmos_policy/config/config.py --"

# 1-3: MimicDroid GR1
export EGOVERSE_GR1_V6_MANIFEST=$ROOT/mimicdroid-gr1/manifest EGOVERSE_GR1_HUM_ROOT=$ROOT/mimicdroid-gr1/egoverse
export EGOVERSE_GR1_V6_2X_MANIFEST=$ROOT/mimicdroid-gr1-2x/manifest   # setting 2 only
export GR1_BASE_CKPT=$WATCH_MID
python -m torch.distributed.run $COMMON experiment=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state
#   ... experiment=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state
#   ... experiment=cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6

# 4: TableTop-Sim
export TABLETOP_ROOT=$ROOT/tabletopsim-watch/data TABLETOP_BASE_CKPT=$WATCH_MID
python -m torch.distributed.run $COMMON experiment=cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0

# 5: RoboCasa (warm start passed as an override)
export BASE_DATASETS_DIR=$ROOT
python -m torch.distributed.run $COMMON experiment=cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2 \
    checkpoint.load_path=$WATCH_MID
```

Use a separate `IMAGINAIRE_OUTPUT_ROOT` per run. Checkpoints are written collectively (DCP), so the output
root must be on storage that every node can see. After training, convert the evaluated iteration with
`convert_dcp_to_pt.py --tag <experiment> --iter <N>`.

## 5. Evaluation

The evaluation is closed-loop: a policy server (this repo's venv) talks to a simulator client in its own
environment, because the simulators pin different numpy/mujoco versions.

| setting | script | simulator side |
|---|---|---|
| 1-3 MimicDroid | `gr1_local_eval.sh`: server `cosmos_policy.scripts.eval_gr1_server`, one per GPU | MimicDroid robocasa venv (`CLIENT_PY`) + `cosmos-framework` GR1 client (`FW`) |
| 4 TableTop-Sim | `tabletop_sim_eval.sh`: server `cosmos_policy.scripts.tabletop_sim_eval_server` + `tabletop_sim_eval_client.py` | Tabletop-Sim venv (`SIM_PY`) |
| 5 RoboCasa | `robocasa24_eval.sh` (`EXP=..._t5_bridge_b2__inference`) | robocasa-cosmos-policy + kitchen assets ([ROBOCASA.md](ROBOCASA.md)) |

```bash
CK=<model_XXXXXX.pt> EXP=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state OUT=<dir> MODE=full GPUS="0 1" ./gr1_local_eval.sh
CKPT=<model_XXXXXX.pt> EXP=cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0 TRIALS=50 ./tabletop_sim_eval.sh
CKPT=<model_000020000.pt> EXP=cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2__inference \
    DATA=$ROOT/RoboCasa-Cosmos-Policy/success_only OUT=<dir> GPUS=0,1,2,3 TRIALS=50 ./robocasa24_eval.sh
```

- **RoboCasa:** 24 tasks x 50 trials with seed 195, `--deterministic True`. The score is the mean of the
  per-task success rates, not episodes pooled across tasks.
- **Pitfalls and the server protocol:** see
  [docs/egoverse_cosmos2p5_pipeline.md](docs/egoverse_cosmos2p5_pipeline.md), sections 5-7.

## 6. Rebuttal experiments

[docs/egoverse_randref_from_scratch.md](docs/egoverse_randref_from_scratch.md) goes from an empty machine
to two reviewer controls on 2 nodes, with `scripts/train_egoverse_randref_2node.sh`:

- **random same-task references (randref):** WATCH's references replaced by uniform draws from the same
  source and fine task class, so the motion matching is removed. Full run and an optional q50 run.
- **noisy task labels (conf50):** half of the episodes relabelled to a sibling class before cross-source
  pairing. Full run and an optional q50 run.

Both come with their GR1 v6 ICL and TableTop-Sim pv0 post-training.

On 4-node groups, [docs/egoverse_rebuttal_4node.md](docs/egoverse_rebuttal_4node.md) launches the full randref
and conf50 mids and their four post-trains with `scripts/train_egoverse_rebuttal_4node.sh`, keeping the global
batch and schedule of every reference run (16 nodes: both mids side by side, then the four post-trains, ~31 h).

## 6b. Real robot (Sprout), context-free post-training

[docs/real-robot-context-free.md](docs/real-robot-context-free.md) post-trains on Sprout real-robot
teleoperation (3 tasks x 20 episodes, 29-D, 15 Hz, one ego camera; HF `Jeongeun/sprout-real-3task`) in the
TableTop-Sim pv0 layout, with `scripts/train_sprout_2node.sh`: chunk 32 and 16, each from the WATCH mid
and from the within-video baseline mid (both public, HF `Jeongeun/cosmos-policy-watch-mid`) and from the
Cosmos-Predict2.5-2B base, six 2-node groups side by side (12 nodes, ~2 h). Checkpoints at 2K, 3K and 4K are kept.
`sprout_serve.sh` serves a returned checkpoint to the robot client over GR00T's zmq protocol, building each sample
with the training dataset's code (`./sprout_serve.sh --selftest` checks it against training samples, no GPU).

## 7. Repository layout

```
cosmos_policy/
  config/experiment/          experiment configs (pre-training: egoverse_ret_*, post-training:
                              egoverse_gr1_ret_*, tabletop_sim_*, cosmos_policy_* for RoboCasa)
  datasets/                   EgoVerse retrieval datasets, GR1, TableTop-Sim and RoboCasa loaders
  scripts/train.py            training entry point
  scripts/eval_gr1_server.py, scripts/tabletop_sim_eval_server.py   policy servers for closed-loop eval
  scripts/sprout_policy_server.py   policy server for the Sprout robot (sprout_serve.sh)
  experiments/robot/robocasa/ RoboCasa eval
  _src/                       Cosmos (imaginaire / predict2 / predict2.5) backbone code
scripts/train_egoverse_randref_2node.sh   2-node launcher for the rebuttal runs
scripts/train_egoverse_rebuttal_4node.sh  4-node launcher for the rebuttal runs
convert_dcp_to_pt.py          DCP checkpoint -> .pt
build_egoverse_query_episode_sets.py      episode subsets for the data-scaling arms (q10-q75)
gr1_local_eval.sh, tabletop_sim_eval.sh, tabletop_sim_eval_client.py, robocasa24_eval.sh,
build_robocasa24_eval_embeddings.py       closed-loop evaluation
docs/                         rebuttal guide, pipeline notes
```

The package also contains configs and loaders for embodiments that WATCH does not use (inherited from Cosmos
Policy). They are kept because the WATCH configs import and extend them.

## License

[LICENSE](LICENSE) (Apache 2.0) and [ATTRIBUTIONS.md](ATTRIBUTIONS.md), inherited from Cosmos Policy.
