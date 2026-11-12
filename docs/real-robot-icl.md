# Real robot (Sprout), in-context post-training: 3 runs on 6 nodes (3 groups of 2)

Three post-trainings on the Sprout real-robot data where a **human demonstration of the same moment is in the
context**: the GR1 v6 ICL layout (`..._egoverse_gr1_ret_v6_448_state`), shrunk for one 224 ego camera. One 2-node
group per run, all at the same time. **Send back the 2K, 3K and 4K checkpoints of every run** (section 6).

| group | `ARM` | experiment | demo | warm start | `MASTER_PORT` |
|---|---|---|---|---|---|
| G | `icl` | `cosmos_predict2p5_2b_sprout_icl_c32` | own ghost | WATCH mid (ours) | 12357 |
| H | `icl_base` | `cosmos_predict2p5_2b_sprout_icl_c32_base` | own ghost | Cosmos-Predict2.5-2B base | 12358 |
| I | `icl_mix30` | `cosmos_predict2p5_2b_sprout_icl_c32_mix30` | own ghost, 30% another episode's | WATCH mid | 12359 |

All three: chunk 32, **global batch 64** (half of the context-free runs: 2 nodes x 8 GPUs x batch 4),
**4,000 iterations**, lr 1e-4 with 100 warmup steps then linear to 0.5e-4 at 4K, **a checkpoint every 1,000**.
Estimated ~1.5 h per run (section 7).

Launcher: [`scripts/train_sprout_2node.sh`](../scripts/train_sprout_2node.sh), once per node, `ARM=...`, `NODE_RANK=0..1`.
It sets the per-GPU batch to 64 / (nodes x GPUs), so a 1-node group (`NNODES=1`, batch 8) gives the same run.

## What the model sees

The human demos are "ghosts": a person wearing a Quest 3 followed each robot episode, replayed as a ghost hand in VR,
at the robot's speed, and the recording is trimmed to the robot episode's length, so human frame t is robot frame t.
At test time a person demonstrates in the same scene first and the robot follows, so the training demo of a robot
window is the ghost of **its own episode** at the same t.

```
idx role         time          content
 0  CUR_STATE    NOW           current state29 (tiled into the latent)
 1  RET_ACTION   NOW           human state57 of the demo, s_h[h+1 .. h+32] (tiled)
 2  RET_VIDEO    NOW - 8 dt    human frame h        (demo window start)
 3  RET_VIDEO    NOW - dt      human frame h + 32   (demo window end)
 4  CUR_IMAGE    NOW           robot ego frame t
-- num_conditional_frames = 5 --
 5  ACTION       NOW           32 x 29 abs_action29          GENERATED
 6  FUTURE       NOW + dt      robot ego frame t + 32        GENERATED
state_t = 7, chunk_duration = 25, dt = 4/15 s
```

- `h = t + j`, with j uniform in [-6, 6] frames during training. A test demo is never in lockstep with the robot.
- Roles and continuous-time positions follow v6, whose 8 demo latents sat at NOW - 8 dt .. NOW - dt.
- Changes against v6:
  - one 224 camera (196 tokens per latent, 1,372 in all; v6 had 15,680);
  - two demo frames instead of the demo's 8 latents;
  - the chunk-end future frame instead of 8 future latents (the pv0 future);
  - the current state only, not the past 32-step state chunk (pv0 and the robot server already work this way).
- `icl_mix30` uses, for 30% of samples, a retrieved ghost of **another** episode instead
  (`manifest/human_pairs.npz`: 32-frame windows, wrist-path DTW in the body frame plus gripper timing). The aim is
  robustness to a demo that differs from the scene.
- Data: `sprout_cup_to_drawer` and `sprout_open_drawer` only, 40 episodes and 6,158 windows.
  `sprout_cup_close_drawer` has no human demos and is left out.
- At global 64 one epoch is 96 iterations, so 1K / 2K / 3K / 4K are 10 / 21 / 31 / 42 epochs.

---

## 1. Code and environment

Same as sections 1 of [`real-robot-context-free.md`](real-robot-context-free.md) (env, Cosmos-Predict2.5-2B base and
VAE in the HF cache for every group, W&B). Check the code has the ICL arms:

```bash
grep -c "sprout_icl_c32" cosmos_policy/config/experiment/sprout_experiment_configs.py   # >= 1
grep -c "icl_mix30" scripts/train_sprout_2node.sh                                      # >= 1
```

## 2. Data (Hugging Face)

The same dataset as the context-free runs, which now also holds the human ghosts. Download again (or update an
existing copy; only new files are fetched):

```bash
export ROOT=/shared/data
hf download Jeongeun/sprout-real-3task --repo-type dataset --local-dir $ROOT/sprout-real-3task      # ~580 MB, private
hf download Jeongeun/cosmos-policy-watch-mid --local-dir $ROOT/watch-mid                            # if not there yet

ls $ROOT/sprout-real-3task/human_ghost/*/episode_*.h5 | wc -l      # 40
ls $ROOT/sprout-real-3task/manifest | grep human                  # human_ghost.json human_pairs.npz human_state57_stats.json
```

The dataset card (`README.md` in the dataset) describes `human_ghost/`.

## 3. Common env (every node of every group)

```bash
cd /path/to/watch && source .venv/bin/activate
export HF_HOME=/shared/hf_cache WANDB_MODE=online
export SPROUT_ROOT=$ROOT/sprout-real-3task
export SPROUT_MID_CKPT=$ROOT/watch-mid/egoverse_ret_v3_448_state_4node/model_000008000.pt      # G, I
```

## 4. Short test (once, on one group)

```bash
MASTER_ADDR=<node 0> MASTER_PORT=12357 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_icl_smoke \
  EXTRA_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20" \
  ARM=icl NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh
```

It must reach iteration 20 and write `iter_000000020`. Then `rm -rf /shared/runs/sprout_icl_smoke`.

## 5. Launch: three groups at the same time

Each group: its own 2 nodes, its node 0 as `MASTER_ADDR`, its own port and its own output root.

```bash
# group G: own ghost, WATCH mid
MASTER_ADDR=<G node 0> MASTER_PORT=12357 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_icl \
  ARM=icl NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group H: own ghost, base
MASTER_ADDR=<H node 0> MASTER_PORT=12358 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_icl_base \
  ARM=icl_base NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh

# group I: 30% another episode's ghost, WATCH mid
MASTER_ADDR=<I node 0> MASTER_PORT=12359 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_icl_mix30 \
  ARM=icl_mix30 NODE_RANK=<0..1> bash scripts/train_sprout_2node.sh
```

Resume after a crash: launch again with the same output root, node count and batch.

### What to check (in `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/sprout_<ARM>_node0.log`)

1. Launcher line: `global batch 64`.
2. Dataset (rank 0):
   ```
   [TabletopSim] task_filter=['sprout_cup_to_drawer', 'sprout_open_drawer'] -> windows 10358 -> 6158
   [TabletopSim] T5 coverage OK (2 instructions)
   [SproutIcl] windows=6158 jitter=+-6 other_prob=0.0 state_t=7 chunk_duration=25     (other_prob=0.3 for group I)
   ```
3. Warm start: G and I load `.../watch-mid/egoverse_ret_v3_448_state_4node/model_000008000.pt`, H the
   Cosmos-Predict2.5-2B base.
4. LR: 100 warmup steps to 1e-4, 0.5e-4 at 4K. Loss finite, iteration time stable after ~50 iterations.

## 6. Outputs: send back 2K, 3K and 4K of every run

```bash
for arm in icl icl_base icl_mix30; do
  tag=cosmos_predict2p5_2b_sprout_icl_c32${arm#icl}
  for it in 2000 3000 4000; do
    IMAGINAIRE_OUTPUT_ROOT=/shared/runs/sprout_${arm} python convert_dcp_to_pt.py --tag $tag --iter $it
  done
  d=/shared/runs/sprout_${arm}/cosmos_policy/cosmos_v2_finetune/$tag/checkpoints
  for it in 2000 3000 4000; do
    hf upload Jeongeun/cosmos-policy-sprout-real $d/model_$(printf %09d $it).pt sprout_${arm}/model_$(printf %09d $it).pt
  done
done
```

Expected in the repo: `sprout_{icl,icl_base,icl_mix30}/model_00000{2,3,4}000.pt` (9 files, ~3.9 GB each). Please
also send the three `pod_logs/sprout_<ARM>_node0.log` files or the W&B run links.

## 7. Estimated time

Rule of section 8 of the 2-node guide (~0.12 s per 1K tokens per GPU + 0.6 s): 4 samples x 7 latents x 196 tokens
= 5.5K tokens per GPU per step, ~1.25 s/iter, 4,000 iterations ~1.4 h, ~1.6 h with checkpoint writes.

---

## Files

| file | content |
|---|---|
| `scripts/train_sprout_2node.sh` | 2-node launcher, `ARM=icl|icl_base|icl_mix30` (global 64) |
| `cosmos_policy/config/experiment/sprout_experiment_configs.py` | the three `sprout_icl_c32*` experiments |
| `cosmos_policy/datasets/sprout_dataset.py` | `SproutIclDataset`: ghost demo slots, roles, jitter, other-episode mix |
