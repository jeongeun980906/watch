# Real robot (Sprout), in-context post-training with retargeted demos: 6 runs on 12 nodes (6 groups of 2)

Training only. Same launcher and dataset as `docs/real-robot-icl.md`; these arms read the retargeted ghosts
(`human_ghost_r29/`) and place the demo window with the inference retrieval (`manifest/ghost_match_r29.json`).

| group | `ARM` | experiment | base representation | warm start | port |
|---|---|---|---|---|---|
| K | `icl_r29abs` | `cosmos_predict2p5_2b_sprout_icl_c32_r29abs_cfaug` | absolute | WATCH mid (ours) | 12361 |
| L | `icl_r29abs_within` | `cosmos_predict2p5_2b_sprout_icl_c32_r29abs_cfaug_within` | absolute | within-video mid | 12362 |
| M | `icl_r29abs_base` | `cosmos_predict2p5_2b_sprout_icl_c32_r29abs_cfaug_base` | absolute | none (Cosmos base) | 12363 |
| N | `icl_r29rel` | `cosmos_predict2p5_2b_sprout_icl_c32_r29rel_cfaug` | chunk-relative | WATCH mid (ours) | 12364 |
| O | `icl_r29rel_within` | `cosmos_predict2p5_2b_sprout_icl_c32_r29rel_cfaug_within` | chunk-relative | within-video mid | 12365 |
| P | `icl_r29rel_base` | `cosmos_predict2p5_2b_sprout_icl_c32_r29rel_cfaug_base` | chunk-relative | none (Cosmos base) | 12366 |

All: global batch 64, 4K iterations, a checkpoint every 1K, demo-consistent augmentation on.

## 1. Code and environment

As `docs/real-robot-icl.md` section 1 (same venv, base checkpoint + VAE in the HF cache, W&B). Check the code has
these arms:

```bash
grep -c "icl_r29rel" scripts/train_sprout_2node.sh                                          # >= 1
grep -c "_r29rel_cfaug" cosmos_policy/config/experiment/sprout_experiment_configs.py        # >= 1
```

## 2. Data (Hugging Face)

Update the dataset copy (only new files are fetched) and the mids:

```bash
export ROOT=/shared/data
hf download Jeongeun/sprout-real-3task --repo-type dataset --local-dir $ROOT/sprout-real-3task      # private
hf download Jeongeun/cosmos-policy-watch-mid --local-dir $ROOT/watch-mid                            # if not there yet

ls $ROOT/sprout-real-3task/human_ghost_r29/*/episode_*.h5 | wc -l     # 40
ls $ROOT/sprout-real-3task/manifest | grep -E "r29|anchor|retarget"  # ghost_match_r29.json human_anchor.json retarget_calib.json
```

## 3. Common env (every node of every group)

```bash
cd /path/to/watch && source .venv/bin/activate
export HF_HOME=/shared/hf_cache WANDB_MODE=online
export SPROUT_ROOT=$ROOT/sprout-real-3task
export SPROUT_MID_CKPT=$ROOT/watch-mid/egoverse_ret_v3_448_state_4node/model_000008000.pt                    # K, N
export SPROUT_WITHIN_CKPT=$ROOT/watch-mid/egoverse_ret_v3_448_state_blankret_bv2_v10_4node/model_000008000.pt # L, O
```

## 4. Short test (once, on one group)

```bash
MASTER_ADDR=<node 0> MASTER_PORT=12361 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_r29_smoke \
  EXTRA_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20" \
  ARM=icl_r29rel NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh
```

It must reach iteration 20 and write `iter_000000020`. Then `rm -rf /shared/runs/sprout_r29_smoke`.

## 5. Launch: six groups at the same time

Each group: its own 2 nodes, its node 0 as `MASTER_ADDR`, its own port and its own output root.

```bash
# group K: absolute base, WATCH mid
MASTER_ADDR=<K node 0> MASTER_PORT=12361 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_icl_r29abs \
  ARM=icl_r29abs NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group L: absolute base, within-video mid
MASTER_ADDR=<L node 0> MASTER_PORT=12362 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_icl_r29abs_within \
  ARM=icl_r29abs_within NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group M: absolute base, no mid
MASTER_ADDR=<M node 0> MASTER_PORT=12363 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_icl_r29abs_base \
  ARM=icl_r29abs_base NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group N: chunk-relative base, WATCH mid
MASTER_ADDR=<N node 0> MASTER_PORT=12364 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_icl_r29rel \
  ARM=icl_r29rel NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group O: chunk-relative base, within-video mid
MASTER_ADDR=<O node 0> MASTER_PORT=12365 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_icl_r29rel_within \
  ARM=icl_r29rel_within NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group P: chunk-relative base, no mid
MASTER_ADDR=<P node 0> MASTER_PORT=12366 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_icl_r29rel_base \
  ARM=icl_r29rel_base NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh
```

Resume after a crash: launch again with the same output root, node count and batch.

### What to check (in `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/sprout_<ARM>_node0.log`)

1. Launcher line: `global batch 64`.
2. Dataset (rank 0):
   ```
   [TabletopSim] task_filter=['sprout_cup_to_drawer', 'sprout_open_drawer'] -> windows 10358 -> 6158
   [SproutIcl] windows=6158 jitter=+-6 other_prob=0.0 state_t=7 chunk_duration=25
   [SproutIclAug] cf_aug={'p_arm': 0.0}
   ```
3. Warm start: K and N load `.../egoverse_ret_v3_448_state_4node/model_000008000.pt`, L and O
   `.../egoverse_ret_v3_448_state_blankret_bv2_v10_4node/model_000008000.pt`, M and P the Cosmos-Predict2.5-2B base.
4. Loss finite, iteration time stable after ~50 iterations.

## 6. Outputs: send back 2K, 3K and 4K of every run

```bash
for arm in icl_r29abs icl_r29abs_within icl_r29abs_base icl_r29rel icl_r29rel_within icl_r29rel_base; do
  rep=${arm#icl_}; rep=${rep%%_*}; mid=${arm#icl_$rep}
  tag=cosmos_predict2p5_2b_sprout_icl_c32_${rep}_cfaug${mid}
  for it in 2000 3000 4000; do
    IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_${arm} python convert_dcp_to_pt.py --tag $tag --iter $it
  done
  d=/shared/runs/sprout_${arm}/cosmos_policy/cosmos_v2_finetune/$tag/checkpoints
  for it in 2000 3000 4000; do
    hf upload Jeongeun/cosmos-policy-sprout-real $d/model_$(printf %09d $it).pt sprout_${arm}/model_$(printf %09d $it).pt
  done
done
```

Expected in the repo: `sprout_icl_r29{abs,rel}{,_within,_base}/model_00000{2,3,4}000.pt` (18 files). Please also send
the six `pod_logs/sprout_<ARM>_node0.log` files or the W&B run links.

## 7. Estimated time

As `docs/real-robot-icl.md` section 7: ~1.25 s/iter, 4,000 iterations ~1.4 h, ~1.6 h with checkpoint writes.

---

## Files

| file | content |
|---|---|
| `scripts/train_sprout_2node.sh` | 2-node launcher, `ARM=icl_r29abs[_within\|_base]`, `ARM=icl_r29rel[_within\|_base]` |
| `cosmos_policy/config/experiment/sprout_experiment_configs.py` | the six `sprout_icl_c32_r29{abs,rel}_cfaug*` experiments |
| `cosmos_policy/datasets/sprout_dataset.py` | `SproutIclAugDataset` (`demo29`, `match_table`, `base_abs`, `base_rel`), `AbsEpisode` |
| `cosmos_policy/datasets/sprout_icl_aug.py` | the augmentation factors; absolute / relative base helpers |
| `cosmos_policy/scripts/sprout_match_table.py` | builds `manifest/ghost_match_r29.json` |
| `quest_capture/pc/retarget.py` | builds `human_ghost_r29/` and the manifest files (in the capture repo) |
