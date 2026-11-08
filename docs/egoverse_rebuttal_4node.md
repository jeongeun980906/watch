# Rebuttal runs on 4-node groups: 2 mid-trainings, then 4 post-trainings (16 nodes)

This is the 4-node version of the randref and conf50 runs in
[`egoverse_randref_from_scratch.md`](egoverse_randref_from_scratch.md) (the 2-node guide). Data, environment and
checkpoints are the same; only the launch changes. On 4 nodes x 8 GPUs every stage can use the global batch of
its reference run, so the reference iteration count and LR schedule are kept as they are. No step budget is
rescaled, unlike the 2-node runs.

| # | stage | experiment | warm start | what it answers |
|---|---|---|---|---|
| 1 | mid | `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_4node` | Cosmos-Predict2.5-2B base | random same-task references (y6hQ Q2) |
| 2 | mid | `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_4node` | Cosmos-Predict2.5-2B base | noisy task labels (yw5w Q2a) |
| 3 | post | `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_randref_4node` | run 1 @ 8K | GR1 v6 ICL from the randref mid |
| 4 | post | `cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_randref_4node` | run 1 @ 8K | Tabletop pv0 from the randref mid |
| 5 | post | `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_conf50_4node` | run 2 @ 8K | GR1 v6 ICL from the conf50 mid |
| 6 | post | `cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_conf50_4node` | run 2 @ 8K | Tabletop pv0 from the conf50 mid |

Both mids use the full query set (`query_eps_nomd.json`, 1,028,812 rows: all of WATCH's except MimicDroid). The
q50 variants of the 2-node guide are not part of this plan.

Launcher: [`scripts/train_egoverse_rebuttal_4node.sh`](../scripts/train_egoverse_rebuttal_4node.sh), run once per
node with `STAGE=randref | conf50 | gr1 | pv0` (and `MID=randref | conf50` for the post-trains).

**Schedule on 16 nodes (four groups of 4 nodes):**

| phase | group A (4 nodes) | group B (4 nodes) | group C (4 nodes) | group D (4 nodes) | est. time |
|---|---|---|---|---|---|
| 1: pretraining | run 1, randref mid | run 2, conf50 mid | idle | idle | ~23 h |
| (between) | convert run 1 @ 8K to `.pt` | convert run 2 @ 8K to `.pt` | | | ~0.5 h |
| 2: post-training | run 3, GR1 from randref | run 4, pv0 from randref | run 5, GR1 from conf50 | run 6, pv0 from conf50 | ~7 h |
| **total** | | | | | **~31 h** |

Each group is an independent 4-node job: its own node 0 as `MASTER_ADDR`, its own `MASTER_PORT`, its own
`IMAGINAIRE_OUTPUT_ROOT`. Phase 2 starts when both mids are converted (or, per chain, as soon as its own mid is:
runs 3 and 4 need only run 1, runs 5 and 6 only run 2).

---

## 1. Data and environment (same as the 2-node guide)

Follow [`egoverse_randref_from_scratch.md`](egoverse_randref_from_scratch.md):

| step | section of the 2-node guide | needed for |
|---|---|---|
| state/action arrays + manifest | 1 | runs 1, 2 |
| random-reference manifest (`pairs_randref.npz`, `query_eps_nomd.json`) | 2 | run 1 |
| conf50 manifest (`pairs_conf50_full.npz`), copied into the same manifest dir | 9.2 | run 2 |
| video (EgoVerse R2, low file-count download) | 3 | runs 1, 2 |
| episode root + check, **once with `pairs_randref.npz` and once with `pairs_conf50_full.npz`** | 4, 9.3 | runs 1, 2 |
| code, Python env, base checkpoint + VAE in the HF cache, W&B | 5 | all |
| GR1 v6 and Tabletop post-training data (`mimicdroid-gr1`, `tabletopsim-watch`) | 7.1 | runs 3-6 |

The conf50 check expects 13,440 episodes and the randref check 13,295; the 145 extra are held-out episodes that
WATCH's own `pairs.npz` also uses as references (9.3). One `EGOVERSE_HUM_ROOT` built for conf50 covers both.

---

## 2. What changes against the references

| | reference | 4-node rebuttal run |
|---|---|---|
| **mid (runs 1, 2)** | `..._ret_v3_448_state` (WATCH v10 state mid) | `..._{randref,conf50}_nomd_4node` |
| references | `pairs.npz` | `pairs_randref.npz` (run 1) / `pairs_conf50_full.npz` (run 2) |
| query rows | 1,045,007 (incl. MimicDroid) | 1,028,812 (MimicDroid dropped, 1.55%) |
| nodes x GPUs x batch = global | 4 x 8 x 2 = 64 | 4 x 8 x 2 = **64** |
| iterations / samples | 8,000 / 512K | 8,000 / 512K |
| LR | 1e-4, warmup 500, linear to 0.3 over 50K (0.894 at 8K) | same |
| checkpoint every | 1,000 | 1,000 |
| **GR1 v6 ICL (runs 3, 5)** | `..._egoverse_gr1_ret_v6_448_state` | `..._gr1_ret_v6_448_state_{randref,conf50}_4node` |
| nodes x GPUs x batch = global | 3 x 8 x 4 = 96 | 4 x 8 x **3** = **96** |
| iterations | 8K schedule, ICL read at 4,000 (384K samples) | **4,000** on the same 8K schedule |
| LR | 1e-4, warmup 500, linear to 0.3 over 8K (0.673 at 4K) | same |
| checkpoint every | 1,000 | 1,000 |
| **pv0 (runs 4, 6)** | `..._tabletop_sim_5task_plain_end_a20_pv0` | `..._a20_pv0_{randref,conf50}_4node` |
| nodes x GPUs x batch = global | 3 x 8 x 16 = 384 | 4 x 8 x **12** = **384** |
| iterations / samples | 8,000 / 3.07M | 8,000 / 3.07M |
| LR | 1e-4, warmup 200, linear to 0.5 over 8K | same |
| checkpoint every | 1,600 | 1,600 |

The per-GPU batch of the post-trains is smaller than in the references (3 instead of 4, 12 instead of 16), so the
global batch, the samples per step and the schedule are identical; only the split of each step over GPUs
differs. A `--dryrun` config diff against the references shows only the
references / query set (mids), `load_path`, `batch_size` and the job name.

Compared with the 2-node runs: the 2-node mid trains 15K iterations at global 32 (480K samples, stretched
schedule); the 4-node mid trains 8K at global 64 (512K samples, the reference schedule). The 4-node runs are the
closer match to WATCH; do not mix 2-node and 4-node checkpoints in one comparison.

---

## 3. Launch

Every run is one 4-node group: the 4 nodes of the group run the same command with `NODE_RANK=0..3`, and the
group's node 0 is its rendezvous host. Runs that go at the same time (two in phase 1, four in phase 2) are separate
groups, so each needs its own `MASTER_ADDR` (that group's node 0), its own `MASTER_PORT` and its own
`IMAGINAIRE_OUTPUT_ROOT`. Never give two groups the same output root: checkpoints and `latest_checkpoint.txt`
collide. `IMAGINAIRE_OUTPUT_ROOT` must be on storage that all 4 nodes of the group see (DCP checkpoints are written
and read collectively).

| group | phase 1 | phase 2 | `MASTER_PORT` |
|---|---|---|---|
| A | run 1 (randref mid) | run 3 (GR1, randref) | 12341 |
| B | run 2 (conf50 mid) | run 4 (pv0, randref) | 12342 |
| C | | run 5 (GR1, conf50) | 12343 |
| D | | run 6 (pv0, conf50) | 12344 |

Common env, on every node of every group:

```bash
cd /path/to/cosmos_policy && source .venv/bin/activate
export HF_HOME=/shared/hf_cache            # where step 5.3 of the 2-node guide cached the checkpoints
export WANDB_MODE=online                   # or offline / disabled
# per group:
export MASTER_ADDR=<node 0 of this group> MASTER_PORT=<port of this group, table above>
```

### 3.1 Short test (once, on one group, before the long runs)

```bash
export EGOVERSE_RANDREF_MANIFEST=$ROOT/egoverse-v10-state/manifest EGOVERSE_HUM_ROOT=$ROOT/egoverse-video
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/rebuttal4_smoke EXTRA_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20" \
  STAGE=randref NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

Then delete `/shared/runs/rebuttal4_smoke`.

### 3.2 Phase 1, group A: run 1 (randref mid)

On the 4 nodes of group A (`MASTER_ADDR=<A node 0> MASTER_PORT=12341`):

```bash
export EGOVERSE_RANDREF_MANIFEST=$ROOT/egoverse-v10-state/manifest
export EGOVERSE_HUM_ROOT=$ROOT/egoverse-video
export IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_randref_4node
STAGE=randref NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

### 3.3 Phase 1, group B: run 2 (conf50 mid), at the same time as run 1

On the 4 nodes of group B (`MASTER_ADDR=<B node 0> MASTER_PORT=12342`):

```bash
export EGOVERSE_CONF50_MANIFEST=$ROOT/egoverse-v10-state/manifest     # same dir, pairs_conf50_full.npz copied in (9.2)
export EGOVERSE_HUM_ROOT=$ROOT/egoverse-video
export IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_conf50_4node
STAGE=conf50 NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

### 3.4 Between the phases: convert the mids to `.pt`

On a node that sees `/shared/runs`, after each mid has written `iter_000008000`:

```bash
for m in randref conf50; do
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_${m}_4node \
    python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state_${m}_nomd_4node --iter 8000
done
export RANDREF_MID=/shared/runs/egoverse_ret_v3_448_state_randref_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_4node/checkpoints/model_000008000.pt
export CONF50_MID=/shared/runs/egoverse_ret_v3_448_state_conf50_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_4node/checkpoints/model_000008000.pt
```

### 3.5 Phase 2: runs 3-6, one group each, all at the same time

On every node of every group (with `RANDREF_MID` / `CONF50_MID` from 3.4):

```bash
# GR1 v6 ICL data
export EGOVERSE_GR1_V6_MANIFEST=$ROOT/mimicdroid-gr1/manifest EGOVERSE_GR1_HUM_ROOT=$ROOT/mimicdroid-gr1/egoverse
# Tabletop pv0 data
export TABLETOP_RANDREF_DATA=$ROOT/tabletopsim-watch/data TABLETOP_RANDREF_MANIFEST=$ROOT/tabletopsim-watch/manifest
```

Then, per group:

```bash
# group A, run 3: GR1 v6 ICL from randref
MASTER_ADDR=<A node 0> MASTER_PORT=12341 GR1_BASE_CKPT=$RANDREF_MID \
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/gr1_posttrain_v6_448_state_randref_4node \
  STAGE=gr1 MID=randref NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh

# group B, run 4: pv0 from randref
MASTER_ADDR=<B node 0> MASTER_PORT=12342 TABLETOP_RANDREF_CKPT=$RANDREF_MID \
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/tabletop_pv0_randref_4node \
  STAGE=pv0 MID=randref NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh

# group C, run 5: GR1 v6 ICL from conf50
MASTER_ADDR=<C node 0> MASTER_PORT=12343 GR1_BASE_CKPT=$CONF50_MID \
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/gr1_posttrain_v6_448_state_conf50_4node \
  STAGE=gr1 MID=conf50 NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh

# group D, run 6: pv0 from conf50
MASTER_ADDR=<D node 0> MASTER_PORT=12344 TABLETOP_RANDREF_CKPT=$CONF50_MID \
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/tabletop_pv0_conf50_4node \
  STAGE=pv0 MID=conf50 NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

The four post-trains read the same post-training data at the same time. GR1 v6 is light on I/O (~0.5 s CPU per
sample) and pv0 very light (~0.02 s), so shared storage is not expected to be the bottleneck.

`MID` selects the config name; the warm start itself is the `.pt` in `GR1_BASE_CKPT` / `TABLETOP_RANDREF_CKPT`.
Check that the two agree (the log prints the experiment name and the checkpoint it loads).

After training, convert with `convert_dcp_to_pt.py --tag <experiment> --iter 4000` (GR1 v6) or `--iter 8000`
(pv0), and evaluate exactly like the reference arms.

---

## 4. What to check

In `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/<stage>_<mid>_node0.log`:

1. `Training with 32 GPUs.` and, from the launcher, `global batch 64` (mids), `96` (GR1) or `384` (pv0). A
   `WARNING: global batch ...` line means the node count or `NPROC_PER_NODE` is not 4 x 8.
2. Dataset lines (rank 0):
   ```
   mids : [EgoVerseRetDataset] query_episode_set=query_eps_nomd.json: kept 13295/13315 episodes, 1028812/1045007 rows
          [EgoVerseRetDataset] pairs=1028812 num_frames=92 R=23 chunk_duration=50
          [EgoVerseRetV3State] ROBUST state norm (mean+/-4std) <- .../manifest/state57_stats_staterobust.json
   GR1  : [EgoVerseGr1RetDataset] pairs=61686 chunk=32 per_frame_stride=1 state_t=20 chunk_duration=77
   pv0  : [TabletopSim] TabletopSimPlainEndImageDataset root=.../tabletopsim-watch/data windows=52043 ... state_t=8 chunk_duration=29
   ```
3. Warm start: the mids load the base checkpoint in non-strict mode (only the policy additions differ, as in
   6.4 of the 2-node guide); the post-trains load the mid `.pt` with no missing or unexpected keys.
4. LR (the reference curves):

   | stage | warmup | LR at the last step |
   |---|---|---|
   | mid | 500 steps to 1e-4 | 0.894e-4 at 8K (0.965e-4 at 3K) |
   | GR1 v6 | 500 steps to 1e-4 | 0.673e-4 at 4K |
   | pv0 | 200 steps to 1e-4 | 0.5e-4 at 8K |

   A mid that ends near 0.89e-4 at 8K is correct here (it was the sign of a wrong schedule only in the 15K
   2-node runs).
5. Loss finite, iteration time stable after ~50 iterations, GPU utilization high (see 6.4 of the 2-node guide
   for the data-loading check).

Resume after a crash by launching again with the same `IMAGINAIRE_OUTPUT_ROOT` and the same node count and batch.

---

## 5. Estimated wall-clock time

Estimated with the rule of section 8 of the 2-node guide (step time from tokens per GPU per step; ~0.12 s per
1K tokens + 0.6 s), not measured on the cluster; add 10-20% for communication and checkpoint writes.

| run | tokens / GPU / step | est. s/iter | iterations | est. time on 4 x 8 GPUs |
|---|---|---|---|---|
| 1, 2: mid | 2 x 50 x 784 = 78K | ~10.4 | 8,000 | ~23 h each |
| 3, 5: GR1 v6 ICL | 3 x 20 x 784 = 47K | ~6.2 | 4,000 | ~7 h each |
| 4, 6: pv0 | 12 x 8 x 196 = 19K | ~2.9 | 8,000 | ~6.5 h each |
| **16 nodes: phase 1 (runs 1 and 2 side by side) + phase 2 (runs 3-6 side by side)** | | | | **~31 h** (23 h + ~0.5 h conversion + 7 h) |
| for reference: all six one after another on a single 4-node group | | | | ~73 h |

Phase 1 is the long part. Phase 2 is set by the GR1 runs (~7 h); the pv0 groups finish ~0.5 h earlier.

---

## Files

| file | content |
|---|---|
| `scripts/train_egoverse_rebuttal_4node.sh` | launcher, `STAGE=randref|conf50|gr1|pv0`, `MID=randref|conf50` |
| `cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py` | `..._randref_nomd_4node`, `..._conf50_nomd_4node` |
| `cosmos_policy/config/experiment/egoverse_gr1_ret_experiment_configs.py` | `..._gr1_ret_v6_448_state_{randref,conf50}_4node` |
| `cosmos_policy/config/experiment/tabletop_sim_experiment_configs.py` | `..._a20_pv0_{randref,conf50}_4node` |
| `docs/egoverse_randref_from_scratch.md` | data download, environment, 2-node versions, conf50 manifest details |
