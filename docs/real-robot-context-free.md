# Real robot (Sprout), context-free post-training: 6 runs on 12 nodes (6 groups of 2)

Six post-trainings on the Sprout real-robot data, pv0 style (Tabletop `..._a20_pv0` recipe, no retrieval context),
one 2-node group each, all at the same time. **Send back the 2K, 3K and 4K checkpoints of every run** (section 6).

| group | `ARM` | experiment | chunk | warm start | `MASTER_PORT` |
|---|---|---|---|---|---|
| A | `c32` | `cosmos_predict2p5_2b_sprout_3task_pv0_c32` | 32 (2.13 s) | WATCH mid (ours) | 12351 |
| B | `c16` | `cosmos_predict2p5_2b_sprout_3task_pv0_c16` | 16 (1.07 s) | WATCH mid (ours) | 12352 |
| C | `c32_within` | `cosmos_predict2p5_2b_sprout_3task_pv0_c32_within` | 32 | within-video mid (baseline) | 12353 |
| D | `c16_within` | `cosmos_predict2p5_2b_sprout_3task_pv0_c16_within` | 16 | within-video mid (baseline) | 12354 |
| E | `c32_base` | `cosmos_predict2p5_2b_sprout_3task_pv0_c32_base` | 32 | Cosmos-Predict2.5-2B base | 12355 |
| F | `c16_base` | `cosmos_predict2p5_2b_sprout_3task_pv0_c16_base` | 16 | Cosmos-Predict2.5-2B base | 12356 |

The WATCH mid and the within-video mid are trained on the same EgoVerse data with the same recipe; the
within-video one has a blank reference (no cross-video task pairing), so A/B vs C/D isolates the pairing and
C/D vs E/F the human-video mid itself.

All six: 2 nodes x 8 GPUs x batch 8 = **global 128**, **4,000 iterations**, lr 1e-4 with 100 warmup steps then
linear to 0.5e-4 at 4K, **a checkpoint every 1,000**. Estimated ~1.5-2 h per run (section 7), so the whole plan is
~2 h on 12 nodes.

Launcher: [`scripts/train_sprout_2node.sh`](../scripts/train_sprout_2node.sh), once per node, `ARM=...`, `NODE_RANK=0..1`.

---

## 1. Code and environment

```bash
git clone https://github.com/jeongeun980906/watch.git
cd watch
grep -c "sprout_3task_pv0" cosmos_policy/config/experiment/sprout_experiment_configs.py   # >= 1
ls scripts/train_sprout_2node.sh
```

Python env, base checkpoint + VAE and W&B exactly as sections 5.2-5.4 of
[`egoverse_randref_from_scratch.md`](egoverse_randref_from_scratch.md):

```bash
uv sync --extra cu128 --group egoverse --python 3.10 && source .venv/bin/activate
export HF_HOME=/shared/hf_cache                      # same on all nodes
hf auth login                                        # a token of the account that accepted the license (below)
hf download nvidia/Cosmos-Predict2.5-2B \
    base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt tokenizer.pth   # ~4 GB; ALL groups need both
ls $HF_HOME/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/*/base/post-trained/*_ema_bf16.pt \
   $HF_HOME/hub/models--nvidia--Cosmos-Predict2.5-2B/snapshots/*/tokenizer.pth   # both must exist
```

**Every group needs the Cosmos-Predict2.5-2B base and its VAE in the HF cache, also A and B (WATCH mid):** the
configs resolve `hf://nvidia/Cosmos-Predict2.5-2B/...` when they are imported, and the launcher runs with
`HF_HUB_OFFLINE=1`, so a missing file stops every run at startup (`LocalEntryNotFoundError`). The repo is gated:
accept the NVIDIA Open Model License on https://huggingface.co/nvidia/Cosmos-Predict2.5-2B with the account whose
token is used, once. With a shared `HF_HOME` one download serves all nodes.

T5 is not needed: the embeddings of the 3 instructions are in the dataset's `manifest/t5_embeddings.pkl`.

## 2. Data and the two mids (Hugging Face)

```bash
export ROOT=/shared/data                              # any storage all nodes see
hf download Jeongeun/sprout-real-3task --repo-type dataset --local-dir $ROOT/sprout-real-3task      # 216 MB, private
hf download Jeongeun/cosmos-policy-watch-mid --local-dir $ROOT/watch-mid                            # 2 x 3.9 GB, public
ls $ROOT/watch-mid/*/model_000008000.pt               # egoverse_ret_v3_448_state_4node (WATCH)
                                                      # egoverse_ret_v3_448_state_blankret_bv2_v10_4node (within-video)

ls $ROOT/sprout-real-3task/*/episode_*.h5 | wc -l     # 60
ls $ROOT/sprout-real-3task/manifest                   # abs_action29_stats_pctl.json pairs.npz state29_stats_pctl.json
                                                      # t5_embeddings.pkl tasks.json windows.npz
```

The dataset card (`README.md` in the dataset) has the 29-D layout. One camera is used: the ego view `images/front_1`.

## 3. Common env (every node of every group)

```bash
cd /path/to/watch && source .venv/bin/activate
export HF_HOME=/shared/hf_cache WANDB_MODE=online     # or offline / disabled
export SPROUT_ROOT=$ROOT/sprout-real-3task
export SPROUT_MID_CKPT=$ROOT/watch-mid/egoverse_ret_v3_448_state_4node/model_000008000.pt                   # A, B
export SPROUT_WITHIN_CKPT=$ROOT/watch-mid/egoverse_ret_v3_448_state_blankret_bv2_v10_4node/model_000008000.pt  # C, D
```

## 4. Short test (once, on one group, before the six runs)

```bash
MASTER_ADDR=<node 0> MASTER_PORT=12351 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_smoke \
  EXTRA_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20" \
  ARM=c32 NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh
```

It must reach iteration 20 and write `iter_000000020`. Then `rm -rf /shared/runs/sprout_smoke`.

## 5. Launch: six groups at the same time

Each group: its own 2 nodes, its node 0 as `MASTER_ADDR`, its own port and **its own output root** (never share
one: checkpoints and `latest_checkpoint.txt` collide). Run on both nodes of the group with `NODE_RANK=0` and `1`.

```bash
# group A: chunk 32, WATCH mid
MASTER_ADDR=<A node 0> MASTER_PORT=12351 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_pv0_c32 \
  ARM=c32 NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group B: chunk 16, WATCH mid
MASTER_ADDR=<B node 0> MASTER_PORT=12352 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_pv0_c16 \
  ARM=c16 NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group C: chunk 32, within-video mid
MASTER_ADDR=<C node 0> MASTER_PORT=12353 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_pv0_c32_within \
  ARM=c32_within NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group D: chunk 16, within-video mid
MASTER_ADDR=<D node 0> MASTER_PORT=12354 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_pv0_c16_within \
  ARM=c16_within NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group E: chunk 32, base
MASTER_ADDR=<E node 0> MASTER_PORT=12355 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_pv0_c32_base \
  ARM=c32_base NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group F: chunk 16, base
MASTER_ADDR=<F node 0> MASTER_PORT=12356 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_pv0_c16_base \
  ARM=c16_base NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh
```

Resume after a crash: launch again with the same output root, node count and batch.

### What to check (in `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/sprout_<ARM>_node0.log`)

1. Launcher line: `global batch 128`. A `WARNING: global batch ...` line means the group is not 2 x 8.
2. Dataset (rank 0):
   ```
   [TabletopSim] T5 coverage OK (3 instructions)
   [TabletopSim] SproutPlainEndDataset root=.../sprout-real-3task windows=10358 target=abs_action29 state_t=4 chunk_duration=13
   [EgoVerseGr1RetDataset] pairs=10358 chunk=32 ...        (chunk=16 for groups B, D and F)
   ```
3. Warm start: groups A, B load `.../watch-mid/egoverse_ret_v3_448_state_4node/model_000008000.pt`; C, D
   `.../watch-mid/egoverse_ret_v3_448_state_blankret_bv2_v10_4node/model_000008000.pt`; E, F the
   Cosmos-Predict2.5-2B base. The action/proprio dims are 29 (the mid was 57-D, but the actions are tiled into the
   latent, so there is no shape change and no missing key).
4. LR: 100 warmup steps to 1e-4, 0.5e-4 at 4K. Loss finite, iteration time stable after ~50 iterations.

## 6. Outputs: send back 2K, 3K and 4K of every run

Checkpoints are written at 1K, 2K, 3K and 4K (DCP folders `iter_00000X000`). Convert 2K, 3K and 4K of each run to
`.pt` (18 files, ~3.9 GB each):

```bash
for arm in c32 c16 c32_within c16_within c32_base c16_base; do
  for it in 2000 3000 4000; do
    IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_pv0_${arm} \
      python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_sprout_3task_pv0_${arm} --iter $it
  done
done
```

Upload all 18 to the private model repo `Jeongeun/cosmos-policy-sprout-real`, one folder per run:

```bash
for arm in c32 c16 c32_within c16_within c32_base c16_base; do
  d=/shared/runs/sprout_pv0_${arm}/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_sprout_3task_pv0_${arm}/checkpoints
  for it in 2000 3000 4000; do
    hf upload Jeongeun/cosmos-policy-sprout-real $d/model_$(printf %09d $it).pt \
      sprout_pv0_${arm}/model_$(printf %09d $it).pt
  done
done
```

Expected in the repo: `sprout_pv0_{c32,c16,c32_within,c16_within,c32_base,c16_base}/model_00000{2,3,4}000.pt`.
Please also send the six
`pod_logs/sprout_<ARM>_node0.log` files (loss curves) or the W&B run links. 1K is not needed.

Why three checkpoints: with 60 episodes one epoch is 81 iterations at global 128, so 2K / 3K / 4K are 25 / 37 / 49
epochs (Tabletop pv0 saw ~59 at 8K). Which one transfers best is read on the robot.

## 7. Estimated time

Rule of section 8 of the 2-node guide (~0.12 s per 1K tokens per GPU + 0.6 s): 8 samples x 4 latents x 196 tokens
= 6.3K tokens per GPU per step, ~1.35 s/iter, 4,000 iterations ~1.5 h, ~1.7 h with checkpoint writes. Chunk 16 has
the same token count (the 16 or 32 actions are tiled into the same action latent).

---

## Files

| file | content |
|---|---|
| `scripts/train_sprout_2node.sh` | 2-node launcher, `ARM=c32|c16|c32_within|c16_within|c32_base|c16_base` |
| `cosmos_policy/config/experiment/sprout_experiment_configs.py` | the six experiments |
| `cosmos_policy/datasets/sprout_dataset.py` | 29-D, 15 Hz, one ego camera, per-view layout |
| `cosmos_policy/datasets/tabletop_sim_dataset.py` | per-view camera list and latent dt made settable (Tabletop unchanged) |
| `sprout_train.sh` | single-machine launcher (smoke tests, `--dryrun`, T5 embeddings) |
