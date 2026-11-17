# Vid2Robot-style baseline mids (4-node groups)

Two mid-trainings with the [Vid2Robot](https://arxiv.org/abs/2403.12943) recipe on the WATCH data and backbone. The demo
(prompt) video and the current image condition the policy, and the action chunk is the only generated slot. There is no
demo-action slot and no future-video prediction. Every command is in this file, in order.

| arm | `STAGE` | experiment | loss |
|---|---|---|---|
| v2r | `v2r` | `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v2r_4node` | flow loss |
| v2r + alignment | `v2r_align` | `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v2r_align_4node` | flow loss + TCC + video-video SigLIP |

- **Layout:** `[state | demo video k | current image | action]`, T = k + 3 (human 26, MimicDroid demo 18). The dataset
  flags are `no_ret_action=True, no_future=True`.
- **Alignment losses:** Vid2Robot Sec. II-E (2) and (3), between the demo video and the query video of each pair.
  - TCC (temporal cycle consistency) on per-frame embeddings.
  - SigLIP on one embedding per video, gathered over all 32 GPUs. Pairs of the same fine task class are not used as
    negatives.
  - Features: output of DiT block 14, averaged over the spatial tokens of each latent frame, then the projection heads
    in `net.align_heads`.
  - Query-video features come from a second forward of the same net, with the query video as clean conditioning in the
    demo-video role.
  - Weights: TCC 1.0, SigLIP 1.0, TCC temperature 0.1 (`_V2R_ALIGN_KW`). The video-text loss (4) is not used, because
    WATCH already conditions on T5 text.
- **Data:** WATCH v10 (`pairs.npz`, every query including the 20 MimicDroid play episodes).
- **Launch:** like the other 4-node mids (4 x 8 x 2 = global 64, 8K iterations, the `_scaled_scheduler` schedule).

**Starting point:** the cluster setup of the randref / conf50 4-node runs (data, training environment, base checkpoint
in the HF cache).

## 0. Environment

On every node:

```bash
export ROOT=<the randref/conf50 data root>
export V10=$ROOT/egoverse-v10-state
export EGOVERSE_V10_MANIFEST=$V10/manifest          # windows.npz, pairs.npz, t5_embeddings.pkl, *_stats.json
export EGOVERSE_HUM_ROOT=$ROOT/egoverse-video
export WATCH=<this repository, the training checkout>
export HF_HOME=<the shared HF cache of the randref runs>
```

## 1. MimicDroid play episodes (once)

The MimicDroid video is not in the EgoVerse R2 bucket. Download it and link it into the episode root. This is the same
data as section 1 of [`egoverse_hawor_rgb_md.md`](egoverse_hawor_rgb_md.md); skip the download if `$MDPLAY` exists.

```bash
export MDPLAY=$ROOT/egoverse-mimicdroid-play
$WATCH/.venv/bin/hf download Jeongeun/egoverse-mimicdroid-play --repo-type dataset --local-dir $MDPLAY
cd $MDPLAY && (cd data && sha256sum -c ../sha256.txt)
mkdir -p episodes && for t in data/*.tar; do tar -xf $t -C episodes; done
for d in $MDPLAY/episodes/*/; do ln -sfn ${d%/} $EGOVERSE_HUM_ROOT/$(basename $d); done
ls -d $EGOVERSE_HUM_ROOT/*__SinkEnvForPlay_D_demo_*/images.front_1 | wc -l     # 20
```

## 2. Mid-training (two 4-node groups at once)

```bash
cd $WATCH && source .venv/bin/activate
export HF_HOME WANDB_MODE=online EGOVERSE_V10_MANIFEST EGOVERSE_HUM_ROOT
```

### 2.1 Short test (each arm)

```bash
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/v2r_smoke EXTRA_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20" \
  MASTER_ADDR=<node 0> MASTER_PORT=12346 STAGE=v2r_align NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
rm -rf /shared/runs/v2r_smoke
```

The `v2r_align` log prints a line every 50 iterations (the first at iteration 0):
`[align] iter N: flow ...  tcc ...  vv ...  valid ...  scale ... bias ...`.

### 2.2 Launch

```bash
# group A
MASTER_ADDR=<A node 0> MASTER_PORT=12346 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_v2r_4node \
  STAGE=v2r NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
# group B
MASTER_ADDR=<B node 0> MASTER_PORT=12347 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_v2r_align_4node \
  STAGE=v2r_align NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

To resume after a crash, launch again with the same `IMAGINAIRE_OUTPUT_ROOT` and node count.

### 2.3 Convert to `.pt`

```bash
for A in v2r v2r_align; do
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_${A}_4node \
    python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state_${A}_4node --iter 8000
done
```

## 3. What to check

| after | check |
|---|---|
| 1 | `sha256sum -c` all OK; 20 MimicDroid episodes with video in `$EGOVERSE_HUM_ROOT` (the launcher refuses to start otherwise) |
| 2 | pod log `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/<stage>_<mid>_node0.log`: `Training with 32 GPUs.`, `global batch 64` (no `WARNING: global batch`) |
| 2 | dataset line `[EgoVerseRetDataset] pairs=1045007`; LR warmup 500 to 1e-4, 0.894e-4 at 8K; loss finite, iteration time stable |
| 2 (`v2r_align`) | `[align]` lines: `valid` mostly `2/2`; `tcc` falls from ~0.09 (uniform-alignment level); `vv` falls from ~0.7 |

## Files

| file | role |
|---|---|
| `cosmos_policy/datasets/egoverse_ret_v3_state_dataset.py` | `no_ret_action`, `no_future`; `align_query_video` emits the query video for the alignment losses |
| `cosmos_policy/models/policy_video2world_model_align.py` | TCC + SigLIP on top of the flow loss; the second (query-video) forward |
| `cosmos_policy/networks/align_dit.py` | `MinimalV1LVGDiTAlign`: the DiT with the projection heads (`align_heads.*`; ignored by a plain `.pt` warm start) |
| `cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py` | `_v2r_4node`, `_V2R_ALIGN_KW` |
| `scripts/train_egoverse_rebuttal_4node.sh` | `STAGE=v2r / v2r_align` |
