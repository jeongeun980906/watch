# Vid2Robot-style baseline: mid → standard post-trainings + eval (4-node groups)

A controlled comparison of the pre-training formulation. The mid is trained with the
[Vid2Robot](https://arxiv.org/abs/2403.12943) recipe; backbone, training data and budget are those of WATCH. The
post-trainings and evals are the **standard** WATCH ones (GR1 v6 ICL, Tabletop pv0), unchanged; only the warm-start
mid differs. Each post-train group runs its closed-loop eval on its own 4 nodes as soon as its training ends
(`EVAL=1`). Every command is in this file, in order.

## 0. Overview

### 0.1 The runs

| run | experiment | nodes |
|---|---|---|
| mid | `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v2r_align_4node` | 4 |
| GR1 v6 ICL post + eval | `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_v2r_align_4node` (standard recipe) | 4 |
| Tabletop pv0 post + eval | `cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_v2r_align_4node` (standard recipe) | 4 |

**Mid.** The demo video and the current image condition the policy, and the action chunk is the only generated slot.
There is no demo-action slot and no future-video prediction (dataset `no_ret_action=True, no_future=True`). The layout
is `[state | demo video k | current image | action]`: T = k + 3 (human 26, MimicDroid demo 18).

**Auxiliary losses** (Vid2Robot Sec. II-E (2) and (3), between the demo video and the query video of each pair):
- TCC (temporal cycle consistency) on per-frame embeddings.
- SigLIP on one embedding per video, gathered over all 32 GPUs. Pairs of the same fine task class are not used as
  negatives.
- Features: output of DiT block 14, averaged over the spatial tokens of each latent frame, then the heads in
  `net.align_heads`.
- The query video goes through a second forward of the same net, as clean conditioning in the demo-video role.
- Weights: TCC 1.0, SigLIP 1.0, TCC temperature 0.1.
- The video-text loss (4) is not used, because WATCH already conditions on T5 text.

**Mid data and budget:** WATCH v10 (`pairs.npz`, every query including the 20 MimicDroid play episodes), 4 x 8 x 2 =
global 64, 8K iterations.

**Transfer to the standard post layout.** Position comes from the slot timestamps (continuous-time RoPE,
`t_positions`) and the slot role embeddings, not from the slot index. The mid's slots therefore have the same role and
time as in the WATCH layout:
- state, current image, action: t = NOW
- demo video: NOW − k·dt … NOW − dt

The post-training adds the demo-action slot (t = NOW) and the future-video slots (t > NOW). The mid never saw these.

**Post-trains:** the standard recipes, warm-started from the mid `.pt`. The `align_heads.*` keys are ignored, because
the `.pt` warm start is strict=False.
- GR1: global 96, read at 4K on the 8K schedule
- pv0: global 384, 8K iterations (= the pv0 reference 3 x 8 x 16)

**Evals** (`scripts/eval_after_post_4node.sh`, started by the launcher on the same 4 nodes):
1. Node 0 converts the final checkpoint to `.pt`. The other nodes wait for a marker file.
2. Every node runs its share on its 8 GPUs:
   - GR1: the 37 L1-L3 atoms of `gr1_local_eval.sh MODE=full`, split over the 4 nodes, 10 episodes each.
   - Tabletop: 10 tasks x 50 trials, in 30 jobs of about 17 episodes each. The tasks are EASY (5), OOD (4 `*_new`) and
     the full `aloha_box_into_pot`. The protocol is that of the rebuttal Tabletop tables: seed 195, `EXEC=16`,
     guidance 1.5, 35 steps.
3. Node 0 merges everything into `$IMAGINAIRE_OUTPUT_ROOT/eval/summary.txt`.

### 0.2 Steps and nodes

| step | section | nodes | time |
|---|---|---|---|
| 1. setup: MimicDroid play episodes, GR1 eval, Tabletop-Sim | 1 | 1 | ~30 min |
| 2. mid | 2 | 4 | ~15-17 h |
| 3. GR1 post + eval, pv0 post + eval, at once | 3 | 2 x 4 = 8 | GR1 ~5 h + eval ~3 h; pv0 ~7 h + eval ~1 h |

Each group runs on its own; the GR1 eval does not wait for the pv0 training.

**Starting point:** the cluster setup of the randref / conf50 4-node runs: data, training environment, base checkpoint in
the HF cache, GR1 and Tabletop post-training data.

### 0.3 Common environment

On every node, for every step:

```bash
export ROOT=<the randref/conf50 data root>
export V10=$ROOT/egoverse-v10-state
export EGOVERSE_V10_MANIFEST=$V10/manifest          # windows.npz, pairs.npz, t5_embeddings.pkl, *_stats.json
export EGOVERSE_HUM_ROOT=$ROOT/egoverse-video
export EGOVERSE_GR1_V6_MANIFEST=$ROOT/mimicdroid-gr1/manifest EGOVERSE_GR1_HUM_ROOT=$ROOT/mimicdroid-gr1/egoverse
export TABLETOP_RANDREF_DATA=$ROOT/tabletopsim-watch/data TABLETOP_RANDREF_MANIFEST=$ROOT/tabletopsim-watch/manifest
export WATCH=<this repository, the training checkout>
export HF_HOME=<the shared HF cache of the randref runs>
export RUNS=/shared/runs                            # IMAGINAIRE_OUTPUT_ROOTs, on storage all nodes share
# eval (section 1.2 / 1.3)
export SIM_PY=$ROOT/Tabletop-Sim-backup/.venv/bin/python
# + the GR1 eval variables of docs/gr1_eval_rebuttal_4node.md §1.4 (CLIENT_PY, FW, MIMICDROID_TASKDEMOS, POOL,
#   PLAY_POOL, PAIRS_DIR; EGOVERSE_GR1_HUM_ROOT is the one above)
```

---

## 1. Setup (once, before training)

The evals start automatically after the post-trains, so their environment must exist before step 3. The launcher
checks it before training starts (`EVAL=1`) and stops within seconds if anything is missing.

### 1.1 MimicDroid play episodes

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

### 1.2 GR1 eval

[`gr1_eval_rebuttal_4node.md`](gr1_eval_rebuttal_4node.md) §1:
- repositories: cosmos-framework (`FW`, the GR1 sim client) and the MimicDroid robocasa fork with its venv (`CLIENT_PY`)
- data: TaskDemos, `mimicdroid-gr1` with its eval pools
- the environment variables of §1.4

If this was set up for the randref eval, it is reused as is. It must be on the shared filesystem: all 4 nodes of the
GR1 group run eval workers.

### 1.3 Tabletop-Sim (simulator client), from GitHub

The client runs in its own venv: Tabletop-Sim pins `numpy<2` / `mujoco==3.2.0`, the policy server (the training venv)
is on numpy 2.x, and the two talk over HTTP. Use the fork
[jeongeun980906/Tabletop-Sim-backup](https://github.com/jeongeun980906/Tabletop-Sim-backup) at commit `606319a`. It
renders the `front` camera that the training data has; the original repository does not, so its views would differ
from training.

On one node (shared filesystem), with network access to GitHub and PyPI:

```bash
cd $WATCH && bash scripts/setup_tabletop_sim.sh $ROOT
```

The script does the following. Run these by hand instead if needed; finished steps are skipped on a re-run:

```bash
git clone https://github.com/jeongeun980906/Tabletop-Sim-backup.git $ROOT/Tabletop-Sim-backup
cd $ROOT/Tabletop-Sim-backup && git checkout 606319a
git submodule update --init --recursive            # object meshes (kevinzakka/mujoco_scanned_objects), ~2 GB
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt      # mujoco 3.2.0, dm_control 1.0.21, ..., `-e .`
uv pip install --python .venv/bin/python 'numpy==1.26.4'           # numpy < 2
```

Run the `uv` commands inside the Tabletop-Sim checkout. In the WATCH repository, uv reads its `pyproject.toml`
(`override-dependencies = numpy>=2`) and silently keeps numpy 2.

**Check** (needs EGL, i.e. headless rendering on the GPU node). The script ends with it; run it once on a node of the
pv0 group:

```bash
MUJOCO_GL=egl $SIM_PY -c "import tabletop; e=tabletop.env('aloha_lift_box','ee_6d_pos'); \
  print(sorted(e.reset().observation['images']))"
# ['back', 'front', 'wrist_left', 'wrist_right']
```

The policy server needs nothing extra:
- the training venv (`$WATCH/.venv`; `tabletop_sim_eval.sh` uses it as is, `SERVER_PY` overrides it)
- the HF cache (base model / VAE)
- `$TABLETOP_RANDREF_MANIFEST` (T5 embeddings, action/state stats)

Optional smoke test of the whole Tabletop eval, with any pv0 `.pt` at hand (2 episodes, ~5 min):

```bash
cd $WATCH && CKPT=<a pv0 .pt> EXP=cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0 SIM_PY=$SIM_PY \
  TASKS=aloha_lift_box TRIALS=2 OUT=/tmp/tt_smoke ./tabletop_sim_eval.sh
```

---

## 2. Mid-training (one 4-node group)

```bash
cd $WATCH && source .venv/bin/activate && export WANDB_MODE=online    # or offline / disabled
```

### 2.1 Short test

```bash
IMAGINAIRE_OUTPUT_ROOT=$RUNS/v2r_smoke EXTRA_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20" \
  MASTER_ADDR=<node 0> MASTER_PORT=12346 STAGE=v2r_align NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
rm -rf $RUNS/v2r_smoke
```

The log prints a line every 50 iterations (the first at iteration 0):
`[align] iter N: flow ...  tcc ...  vv ...  valid ...  scale ... bias ...`.

### 2.2 Launch

```bash
MASTER_ADDR=<node 0> MASTER_PORT=12346 IMAGINAIRE_OUTPUT_ROOT=$RUNS/egoverse_ret_v3_448_state_v2r_align_4node \
  STAGE=v2r_align NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

To resume after a crash, launch again with the same `IMAGINAIRE_OUTPUT_ROOT` and node count.

### 2.3 Convert to `.pt`

```bash
IMAGINAIRE_OUTPUT_ROOT=$RUNS/egoverse_ret_v3_448_state_v2r_align_4node \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v2r_align_4node --iter 8000
export MIDPT=$RUNS/egoverse_ret_v3_448_state_v2r_align_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v2r_align_4node/checkpoints/model_000008000.pt
ls -la $MIDPT
```

---

## 3. Post-training + eval (two 4-node groups at once)

Each group trains, then evaluates on the same nodes. Run on every node of the group, `NODE_RANK=0..3`. The
environment of 0.3 must be set on every node, including the eval variables.

```bash
cd $WATCH && source .venv/bin/activate && export WANDB_MODE=online
# group A: GR1 v6 ICL -> GR1 L1-L3 eval
EVAL=1 MASTER_ADDR=<A node 0> MASTER_PORT=12347 GR1_BASE_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=$RUNS/gr1_posttrain_v6_448_state_v2r_align_4node \
  STAGE=gr1 MID=v2r_align NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
# group B: Tabletop pv0 -> Tabletop eval
EVAL=1 MASTER_ADDR=<B node 0> MASTER_PORT=12348 TABLETOP_RANDREF_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=$RUNS/tabletop_pv0_v2r_align_4node \
  STAGE=pv0 MID=v2r_align NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

- First lines: `[eval] gr1 eval environment OK` / `[eval] pv0 eval environment OK`, then training. The log prints the
  checkpoint it loads; check that it is `$MIDPT`.
- After training: `[eval] node <r>/4 ...`, then each node's share. Logs:
  - `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/<stage>_v2r_align_eval_node<r>.log`
  - GR1: `$IMAGINAIRE_OUTPUT_ROOT/eval/shard<r>/` (as `gr1_local_eval.sh`)
  - Tabletop: `$IMAGINAIRE_OUTPUT_ROOT/eval/<task>/ep<start>/` (as `tabletop_sim_eval.sh`)
- Node 0 ends with the merged `$IMAGINAIRE_OUTPUT_ROOT/eval/summary.txt`:
  - GR1: one line per atom, `atoms 37/37`, `TOTAL s/n`
  - Tabletop: per task `s/50`, EASY / OOD / FULL totals
- The short test of 2.1 works here too, but run it without `EVAL=1`: its 20-iteration checkpoint is not the one the eval
  reads.

**Eval only** (e.g. after an eval crash; training is skipped, the existing checkpoint is reused): the same command
with `EVAL_ONLY=1` instead of `EVAL=1`. To redo only some GR1 atoms or Tabletop jobs, run them by hand as in
[`gr1_eval_rebuttal_4node.md`](gr1_eval_rebuttal_4node.md) §2.2 / `tabletop_sim_eval.sh` (`TASKS`, `START`, `TRIALS`).

---

## 4. What to check, and what to send back

| after | check |
|---|---|
| 1 | 20 MimicDroid episodes with video; GR1 eval setup of §1 of the eval guide; Tabletop-Sim check prints the 4 cameras |
| 2, 3 | pod log `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/<stage>_<mid>_node0.log`: `Training with 32 GPUs.`, `global batch 64 / 96 / 384` (no `WARNING: global batch`) |
| 2 | dataset line `[EgoVerseRetDataset] pairs=1045007`; LR warmup 500 to 1e-4, 0.894e-4 at 8K; loss finite, iteration time stable |
| 2 | `[align]` lines: `valid` mostly `2/2`; `tcc` falls from ~0.09 (uniform-alignment level); `vv` falls from ~0.7 |
| 3 | both load `$MIDPT`; GR1 `state_t=20`, pv0 `state_t=8` (the standard layouts) |
| 3 (eval) | GR1 `atoms 37/37`, no `ATOM FAILED`; Tabletop every task `/50` (no `INCOMPLETE`), no `job FAILED`; every Tabletop `server.log` prints `latent contract: PLAIN_END+PER_VIEW(nb=0) ... state_t=8` |

**Send back:**
- `$RUNS/gr1_posttrain_v6_448_state_v2r_align_4node/eval/` (`summary.txt`, `shard*/logs/`, `shard*.log`)
- `$RUNS/tabletop_pv0_v2r_align_4node/eval/` (`summary.txt`, `*/ep*/results.json`, `*/ep*/config.txt`, `*_ep*.log`)

## Files

| file | role |
|---|---|
| `cosmos_policy/datasets/egoverse_ret_v3_state_dataset.py` | `no_ret_action`, `no_future`; `align_query_video` (the query clip for the alignment losses) |
| `cosmos_policy/models/policy_video2world_model_align.py` | TCC + SigLIP on top of the flow loss; the second (query-video) forward |
| `cosmos_policy/networks/align_dit.py` | `MinimalV1LVGDiTAlign`: the DiT with the heads (`align_heads.*`; ignored by the plain `.pt` warm start of the post-trains) |
| `cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py` | mid (`_v2r_4node`; `v2r` without the aux losses is also defined) |
| `cosmos_policy/config/experiment/egoverse_gr1_ret_experiment_configs.py`, `tabletop_sim_experiment_configs.py` | standard GR1 / pv0 post-trains with tag `v2r_align` |
| `scripts/train_egoverse_rebuttal_4node.sh` | `STAGE=v2r_align`; `STAGE=gr1 / pv0` with `MID=v2r_align`; `EVAL=1`, `EVAL_ONLY=1` |
| `scripts/eval_after_post_4node.sh` | convert, 4-node eval split, merge |
| `scripts/setup_tabletop_sim.sh` | Tabletop-Sim client from GitHub |
| `gr1_local_eval.sh` (`ATOM_SHARD`), `tabletop_sim_eval.sh` | the evals |
