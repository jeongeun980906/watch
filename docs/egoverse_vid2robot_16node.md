# Vid2Robot-style baseline on 16 nodes: mid (16) → GR1 post + eval (8), pv0 post + eval (8)

The same baseline as [`egoverse_vid2robot_4node.md`](egoverse_vid2robot_4node.md) (mid recipe, aux losses, data,
standard posts, eval protocol), laid out for 16 nodes. Setup and environment are those of that guide; this file only
lists what differs and the commands.

**Not the WATCH recipe.** 128 GPUs cannot hold the WATCH mid's global batch 64, and 64 GPUs cannot hold the GR1
post's 96. The mid and the GR1 post therefore run a larger global batch over fewer iterations, with the same number of
samples and the LR schedule kept in samples. The peak LR is unchanged. pv0 keeps its recipe exactly. Results of this
layout are labelled as such; the controlled comparison is the 4-/8-node guide.

## 0. Overview

| run | experiment | nodes | per GPU | global | iters | schedule |
|---|---|---|---|---|---|---|
| mid | `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v2r_align_16node` | 16 | 1 | 128 | 4K | warmup 250, cycle 25K |
| GR1 v6 ICL post + eval | `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_v2r_align16_8node` | 8 | 2 | 128 | 3K | warmup 375, cycle 6K |
| Tabletop pv0 post + eval | `cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_v2r_align16_4node` | 8 | 6 | 384 | 8K | the pv0 reference |

For comparison, the WATCH recipe (4-/8-node guide): mid 64 x 8K (warmup 500, cycle 50K); GR1 96 x 4K (warmup 500,
cycle 8K); pv0 as above. `MID=v2r_align16` names the posts from the 16-node mid; the pv0 experiment keeps its
`_4node` name because its recipe is the 4-node one.

**Evals** (`scripts/eval_after_post_4node.sh`, on the 8 nodes of each post):
- GR1: the 37 L1-L3 atoms, split over 8 nodes (5 or 4 atoms per node), at iteration 3000.
- Tabletop: 10 tasks x 50 trials, each task cut into 6 episode ranges (60 jobs, one GPU each), at iteration 8000.
  Seeding is per episode, so the split gives the same episodes as the 4-node one.

| step | section | nodes | time |
|---|---|---|---|
| 1. setup | [4-node guide §1](egoverse_vid2robot_4node.md#1-setup-once-before-training) | 1 | ~30 min |
| 2. mid | 2 | 16 | ~5-6 h |
| 3. GR1 post + eval, pv0 post + eval, at once | 3 | 2 x 8 = 16 | GR1 ~2.5 h + eval ~1.5 h; pv0 ~3.5 h + eval ~30 min |

The 16 nodes of the mid are then split into the two 8-node groups of step 3.

**Environment:** [4-node guide §0.3](egoverse_vid2robot_4node.md#03-common-environment) on every node, including the
eval variables. **Setup:** [4-node guide §1](egoverse_vid2robot_4node.md#1-setup-once-before-training) (MimicDroid play
episodes, GR1 eval, Tabletop-Sim), unchanged.

---

## 2. Mid-training (16 nodes)

```bash
cd $WATCH && source .venv/bin/activate && export WANDB_MODE=online    # or offline / disabled
```

Short test (optional):

```bash
IMAGINAIRE_OUTPUT_ROOT=$RUNS/v2r16_smoke EXTRA_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20" \
  MASTER_ADDR=<node 0> MASTER_PORT=12346 NNODES=16 STAGE=v2r_align NODE_RANK=<0..15> bash scripts/train_egoverse_rebuttal_4node.sh
rm -rf $RUNS/v2r16_smoke
```

Launch:

```bash
MASTER_ADDR=<node 0> MASTER_PORT=12346 IMAGINAIRE_OUTPUT_ROOT=$RUNS/egoverse_ret_v3_448_state_v2r_align_16node \
  NNODES=16 STAGE=v2r_align NODE_RANK=<0..15> bash scripts/train_egoverse_rebuttal_4node.sh
```

The first line must read `..._v2r_align_16node  node <r>/16  gpus/node 8  batch 1/GPU  global batch 128`. To resume
after a crash, launch again with the same `IMAGINAIRE_OUTPUT_ROOT` and `NNODES=16`.

Convert to `.pt`:

```bash
IMAGINAIRE_OUTPUT_ROOT=$RUNS/egoverse_ret_v3_448_state_v2r_align_16node \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v2r_align_16node --iter 4000
export MIDPT=$RUNS/egoverse_ret_v3_448_state_v2r_align_16node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v2r_align_16node/checkpoints/model_000004000.pt
ls -la $MIDPT
```

---

## 3. Post-training + eval (two 8-node groups at once)

Each group trains, then evaluates on the same 8 nodes. Run on every node of the group, `NODE_RANK=0..7`.

```bash
cd $WATCH && source .venv/bin/activate && export WANDB_MODE=online
# group A (8 nodes): GR1 v6 ICL -> GR1 L1-L3 eval at iteration 3000
EVAL=1 MASTER_ADDR=<A node 0> MASTER_PORT=12347 GR1_BASE_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=$RUNS/gr1_posttrain_v6_448_state_v2r_align16_8node \
  NNODES=8 STAGE=gr1 MID=v2r_align16 NODE_RANK=<0..7> bash scripts/train_egoverse_rebuttal_4node.sh
# group B (8 nodes): Tabletop pv0 -> Tabletop eval at iteration 8000
EVAL=1 MASTER_ADDR=<B node 0> MASTER_PORT=12348 TABLETOP_RANDREF_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=$RUNS/tabletop_pv0_v2r_align16_8node \
  NNODES=8 STAGE=pv0 MID=v2r_align16 NODE_RANK=<0..7> bash scripts/train_egoverse_rebuttal_4node.sh
```

- First lines: `[eval] gr1 eval environment OK` / `[eval] pv0 eval environment OK`, then
  `batch 2/GPU  global batch 128` (GR1) / `batch 6/GPU  global batch 384` (pv0). The log prints the checkpoint it
  loads; check that it is `$MIDPT`.
- After training: `[eval] node <r>/8 ...`, then each node's share. Logs and `summary.txt` are as in the
  [4-node guide §3](egoverse_vid2robot_4node.md#3-post-training--eval-two-4-node-groups-at-once).
- **Eval only:** the same command with `EVAL_ONLY=1` instead of `EVAL=1`.

---

## 4. What to check, and what to send back

| after | check |
|---|---|
| 2 | `Training with 128 GPUs.`, `batch 1/GPU  global batch 128`; LR warmup 250 steps to 1e-4; `[align]` lines with `valid` mostly `1/1`, `tcc` and `vv` falling |
| 3 | `Training with 64 GPUs.`; both load `$MIDPT`; GR1 `state_t=20`, pv0 `state_t=8` |
| 3 (eval) | GR1 `atoms 37/37`, no `ATOM FAILED`; Tabletop every task `/50` (no `INCOMPLETE`), no `job FAILED` |

**Send back:**
- `$RUNS/gr1_posttrain_v6_448_state_v2r_align16_8node/eval/` (`summary.txt`, `shard*/logs/`, `shard*.log`)
- `$RUNS/tabletop_pv0_v2r_align16_8node/eval/` (`summary.txt`, `*/ep*/results.json`, `*/ep*/config.txt`, `*_ep*.log`)

## Files

| file | role |
|---|---|
| `cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py` | `..._v2r_align_16node` (mid, global 128 x 4K) |
| `cosmos_policy/config/experiment/egoverse_gr1_ret_experiment_configs.py` | `..._v2r_align16_8node`, `..._v2r_align_8node` (GR1 post, global 128 x 3K) |
| `cosmos_policy/config/experiment/tabletop_sim_experiment_configs.py` | `..._pv0_v2r_align16_4node` (pv0 reference recipe) |
| `scripts/train_egoverse_rebuttal_4node.sh` | picks the 16-node mid / 8-node GR1 config from `NNODES`; batch per GPU = global / GPUs |
| `scripts/eval_after_post_4node.sh` | eval split by `NNODES` (GR1 atoms, Tabletop episode ranges) |
