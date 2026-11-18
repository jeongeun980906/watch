# RGB-only hand trajectories with MimicDroid play: smoothing → pairs → mid → post-training + eval (4-node groups)

This guide redoes the HaWoR run of [`egoverse_hawor_rgb.md`](egoverse_hawor_rgb.md) from the state conversion
(keypoint smoothing) onward, this time **with the 20 MimicDroid play episodes** in the mid. The query set is then
WATCH's full query set, not `query_eps_nomd.json`. Every command is in this file, in order.

**Starting point:** sections 1-3.1 of `egoverse_hawor_rgb.md` are done: the HaWoR environment, the HaWoR output
`$WORK/hawor/<ep>.npz` for every episode, and `$WORK/cam_fit.json` (with `wrist_rot`). Nothing written by the
earlier run is overwritten; every new output has its own name (`_md`).

## 0. Overview

Arm: `STAGE=hawor_full_md`. It is the same as `hawor_full` (pairs and `state57`/`action57` from HaWoR) plus these
MimicDroid changes:

- **video:** the MimicDroid play video is not in the EgoVerse R2 bucket. It was rebuilt from the MimicDroid release
  and is downloaded from HF `Jeongeun/egoverse-mimicdroid-play` (section 1).
- **state/action:** HaWoR is not run on MimicDroid. Its hands are simulator hands, and its `state57`/`action57`
  are the simulator values, the same as in WATCH.
- **pairs:** the MimicDroid query rows of the released `pairs.npz` (16,195 rows) are added verbatim to the HaWoR
  pairs, the same way v10 brought them in. The query set is `query_eps_md.json` (section 3.4).

| step | section | where | time on 4 nodes |
|---|---|---|---|
| 1. MimicDroid play episodes (1.9 GB) | 1 | 1 node | ~5 min |
| 2. `state57`/`action57` from HaWoR (smoothed) | 2 | 1 node, CPU | ~1 h |
| 3. pairs from HaWoR + MimicDroid rows | 3 | 1 node (CPU + 1 GPU for T5) | ~1.5 h |
| 4. mid-training | 4 | one 4-node group | ~23 h |
| 5. two post-trainings + their evals (GR1, Tabletop) | 5 | two 4-node groups at once | GR1 ~5 h + eval ~3 h; pv0 ~7 h + eval ~1 h |

### 0.1 Common environment

On every node, for every step (as in `egoverse_hawor_rgb.md` §0.3):

```bash
export ROOT=<the randref/conf50 data root>
export V10=$ROOT/egoverse-v10-state
export EGOVERSE_HUM_ROOT=$ROOT/egoverse-video
export QSET=$V10/manifest/query_eps_nomd.json
export WATCH=<this repository, the training checkout>
export WORK=$ROOT/hawor_rgb                                  # the earlier run's directory (hawor/, cam_fit.json)
export HF_HOME=<the shared HF cache of the randref runs>
export EGOVERSE_HAWOR_ROOT=$WORK/episodes_hawor_md           # new episode root of this guide
export EGOVERSE_HAWOR_MANIFEST=$WORK/manifest_md             # new manifest dir of this guide
export MDPLAY=$ROOT/egoverse-mimicdroid-play
```

---

## 1. MimicDroid play episodes

```bash
cd $ROOT && $WATCH/.venv/bin/hf download Jeongeun/egoverse-mimicdroid-play --repo-type dataset --local-dir $MDPLAY
cd $MDPLAY && (cd data && sha256sum -c ../sha256.txt)
mkdir -p episodes && for t in data/*.tar; do tar -xf $t -C episodes; done
ls episodes | wc -l          # 20
```

Each episode has `images.front_1` (128x128, 20 Hz) and the `state57`/`action57`/`meta` of the v10-state episode.
The data was built with `hawor_rgb/build_md_play.py` from `Rutav/MimicDroidDataset`; to rebuild it instead of downloading:

```bash
$WATCH/.venv/bin/hf download Rutav/MimicDroidDataset --repo-type dataset \
    --include "training/0*/demo_im128_notp.hdf5" --local-dir $ROOT/MimicDroidDataset     # 4.1 GB
uv run --no-project --python 3.11 --with h5py --with "zarr>=3" --with numpy --with pillow \
  python $WATCH/hawor_rgb/build_md_play.py --hdf5-root $ROOT/MimicDroidDataset/training --manifest $V10/manifest \
    --state-root $V10/episodes --out $MDPLAY/episodes --procs 8
```

---

## 2. HaWoR → `state57`/`action57` (one node, CPU)

The conversion of `egoverse_hawor_rgb.md` §3.2 into the new root, with the default keypoint smoothing (EgoVerse
`smooth_keypoints`, the same as the tracked data):

```bash
cd $WATCH/hawor_rgb
UVZ="uv run --no-project --python 3.11 --with zarr>=3 --with numpy --with scipy"
$UVZ python to_state57.py --hawor-out $WORK/hawor --cam-fit $WORK/cam_fit.json --manifest $V10/manifest \
    --state-root $V10/episodes --video-root $EGOVERSE_HUM_ROOT --out-root $EGOVERSE_HAWOR_ROOT --procs 32
```

Then add the episodes that are not converted. Link the MimicDroid play episodes **first**, so that a MimicDroid
directory of `$EGOVERSE_HUM_ROOT` without video is not linked in their place. After that, link the tracked
directories of the human episodes without HaWoR output:

```bash
for d in $MDPLAY/episodes/*/; do ln -sfn ${d%/} $EGOVERSE_HAWOR_ROOT/$(basename $d); done
for d in $EGOVERSE_HUM_ROOT/*/; do e=$(basename $d); [ -e $EGOVERSE_HAWOR_ROOT/$e ] || ln -s $d $EGOVERSE_HAWOR_ROOT/$e; done
ls -d $EGOVERSE_HAWOR_ROOT/*__SinkEnvForPlay_D_demo_*/images.front_1 | wc -l     # 20
```

**Check:** `converted N, failed 0`, `mean coverage L ≥ 0.85, R ≥ 0.85`, 20 MimicDroid episodes with video.

---

## 3. Pairs: `pairs_hawor_md.npz`

The same matcher as `egoverse_hawor_rgb.md` §4, run on the new root. It matches human queries only.

```bash
cd $WATCH/hawor_rgb
UVZ="uv run --no-project --python 3.11 --with zarr>=3 --with numpy --with scipy"
UVF="uv run --no-project --python 3.11 --with numpy --with faiss-cpu"
# 3.1 descriptors (CPU, ~15 min with 48 processes)
$UVZ python build_desc.py --root $EGOVERSE_HAWOR_ROOT --manifest $V10/manifest --out $WORK/desc_hawor_md --procs 48
# 3.2 episode text embeddings: reuse $WORK/temb.npz of the earlier run; if it is missing:
[ -f $WORK/temb.npz ] || $WATCH/.venv/bin/python t5_text.py --root $V10/episodes --out $WORK/temb.npz
# 3.3 matcher (~1 h on one node with 48 processes; --shard i/n over n nodes as in the earlier guide)
$UVF python match.py --desc $WORK/desc_hawor_md --temb $WORK/temb.npz --manifest $V10/manifest \
    --out $WORK/pairs_hawor_md_0.npz --shard 0/1 --procs 48
```

### 3.4 Manifest dir with the MimicDroid rows

```bash
python3 finalize_manifest.py --shards "$WORK/pairs_hawor_md_*.npz" --base $V10/manifest --query-set $QSET \
    --name hawor_md --out $EGOVERSE_HAWOR_MANIFEST --md-from $V10/manifest/pairs.npz
```

`--md-from` appends the 16,195 MimicDroid query rows of the released `pairs.npz` verbatim and writes
`query_eps_md.json` (= `query_eps_nomd.json` + the 20 MimicDroid episodes).

**Check:** it prints `MimicDroid: 16,195 query rows from 20 episodes appended`, train query rows = the human rows
of the matcher + 16,195, ~6 cross refs/query, and `query set query_eps_md.json: 13315/13315 episodes present`.

---

## 4. Mid-training (one 4-node group)

Experiment `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_hawor_full_md_4node`: the `hawor_full` recipe
(4 x 8 x 2 = global 64, 8K iterations, cycle 50K, warmup 500) with `pairs_hawor_md.npz` and `query_eps_md.json`.

```bash
cd $WATCH && source .venv/bin/activate
export HF_HOME WANDB_MODE=online
export EGOVERSE_HAWOR_MANIFEST EGOVERSE_HUM_ROOT EGOVERSE_HAWOR_ROOT
```

### 4.1 Short test

```bash
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/hawor_md_smoke EXTRA_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20" \
  MASTER_ADDR=<node 0> MASTER_PORT=12343 STAGE=hawor_full_md NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
rm -rf /shared/runs/hawor_md_smoke
```

### 4.2 Launch

```bash
# the launcher sets EGOVERSE_HUM_ROOT=$EGOVERSE_HAWOR_ROOT and refuses to start without the 20 MimicDroid videos
MASTER_ADDR=<node 0> MASTER_PORT=12343 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_hawor_full_md_4node \
  STAGE=hawor_full_md NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

Resume after a crash: launch again with the same `IMAGINAIRE_OUTPUT_ROOT` and node count.

### 4.3 Convert the mid to `.pt`

```bash
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_hawor_full_md_4node \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state_hawor_full_md_4node --iter 8000
MIDPT=/shared/runs/egoverse_ret_v3_448_state_hawor_full_md_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_hawor_full_md_4node/checkpoints/model_000008000.pt
ls -la $MIDPT      # ~3.9 GB
```

---

## 5. Post-training + eval: GR1 v6 ICL and Tabletop pv0 (two 4-node groups at once)

Same recipe as the randref/conf50 post-trains, warm-started from `$MIDPT` (§4.3). Each group runs its
closed-loop eval on its own 4 nodes as soon as its training ends (`EVAL=1`, `scripts/eval_after_post_4node.sh`).

| post-train | experiment | 4 nodes | iterations | eval at |
|---|---|---|---|---|
| GR1 v6 ICL | `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_hawor_full_md_4node` | batch 3/GPU = global 96 | 4K on the 8K schedule | 4K: 37 L1-L3 atoms x 10 episodes, split over the 4 nodes |
| Tabletop pv0 | `cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_hawor_full_md_4node` | batch 12/GPU = global 384 | 8K | 8K: 10 tasks x 50 trials (EASY 5, OOD 4, full `aloha_box_into_pot`), seed 195, `EXEC=16` |

### 5.1 Eval setup (once, before the post-trains)

The launcher checks the eval environment before training starts and stops within seconds if anything is missing.
- GR1: [`egoverse_vid2robot_4node.md`](egoverse_vid2robot_4node.md) §1.2 (= [`gr1_eval_rebuttal_4node.md`](gr1_eval_rebuttal_4node.md) §1; reuse it if it is
  already set up).
- Tabletop-Sim: [`egoverse_vid2robot_4node.md`](egoverse_vid2robot_4node.md) §1.3 (`scripts/setup_tabletop_sim.sh`; reuse it if `$SIM_PY` exists).

Both must be on the shared filesystem: every node of a group runs eval workers.

### 5.2 Launch

```bash
cd $WATCH && source .venv/bin/activate && export HF_HOME WANDB_MODE=online
export EGOVERSE_GR1_V6_MANIFEST=$ROOT/mimicdroid-gr1/manifest EGOVERSE_GR1_HUM_ROOT=$ROOT/mimicdroid-gr1/egoverse
export TABLETOP_RANDREF_DATA=$ROOT/tabletopsim-watch/data TABLETOP_RANDREF_MANIFEST=$ROOT/tabletopsim-watch/manifest
export SIM_PY=$ROOT/Tabletop-Sim-backup/.venv/bin/python
# + the GR1 eval variables of gr1_eval_rebuttal_4node.md §1.4 (CLIENT_PY, FW, MIMICDROID_TASKDEMOS, POOL, PLAY_POOL,
#   PAIRS_DIR; EGOVERSE_GR1_HUM_ROOT is the one above)
# group A: GR1 v6 ICL -> GR1 L1-L3 eval
EVAL=1 MASTER_ADDR=<A node 0> MASTER_PORT=12344 GR1_BASE_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/gr1_posttrain_v6_448_state_hawor_full_md_4node \
  STAGE=gr1 MID=hawor_full_md NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
# group B: Tabletop pv0 -> Tabletop eval
EVAL=1 MASTER_ADDR=<B node 0> MASTER_PORT=12345 TABLETOP_RANDREF_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/tabletop_pv0_hawor_full_md_4node \
  STAGE=pv0 MID=hawor_full_md NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

- First lines: `[eval] gr1 eval environment OK` / `[eval] pv0 eval environment OK`, then training. The log prints the
  experiment and the checkpoint it loads; check that it is `$MIDPT`.
- After training, node 0 converts the final checkpoint to `.pt` (the others wait for it), then every node runs its
  share: `[eval] node <r>/4 ...`. Logs:
  - `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/<stage>_hawor_full_md_eval_node<r>.log`
  - GR1: `$IMAGINAIRE_OUTPUT_ROOT/eval/shard<r>/` (as `gr1_local_eval.sh`)
  - Tabletop: `$IMAGINAIRE_OUTPUT_ROOT/eval/<task>/ep<start>/` (as `tabletop_sim_eval.sh`)
- Node 0 ends with the merged `$IMAGINAIRE_OUTPUT_ROOT/eval/summary.txt`:
  - GR1: one line per atom, `atoms 37/37`, `TOTAL s/n`
  - Tabletop: per task `s/50`, EASY / OOD / FULL totals
- **Eval only** (e.g. after an eval crash; training is skipped, the existing checkpoint is reused): the same command
  with `EVAL_ONLY=1` instead of `EVAL=1`. To redo single GR1 atoms or Tabletop jobs, run them by hand as in
  [`gr1_eval_rebuttal_4node.md`](gr1_eval_rebuttal_4node.md) §2.2 / `tabletop_sim_eval.sh` (`TASKS`, `START`, `TRIALS`).

**Send back:**
- `/shared/runs/gr1_posttrain_v6_448_state_hawor_full_md_4node/eval/` (`summary.txt`, `shard*/logs/`, `shard*.log`)
- `/shared/runs/tabletop_pv0_hawor_full_md_4node/eval/` (`summary.txt`, `*/ep*/results.json`, `*/ep*/config.txt`, `*_ep*.log`)

---

## 6. What to check

| after | check |
|---|---|
| 1 | `sha256sum -c` all OK; 20 episode dirs |
| 2 | `converted N, failed 0`; mean coverage L/R ≥ 0.85; 20 MimicDroid episodes with `images.front_1` in `$EGOVERSE_HAWOR_ROOT` |
| 3 | `finalize_manifest.py`: 16,195 MimicDroid rows appended; `query_eps_md.json: 13315/13315 episodes present` |
| 4 | dataset lines: `query_episode_set=query_eps_md.json: kept 13315/...`; launcher's first lines read episodes from `$EGOVERSE_HAWOR_ROOT` |
| 4, 5 | pod log `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/<stage>_<mid>_node0.log`: `Training with 32 GPUs.`, `global batch 64 / 96 / 384`; LR and loss as in `egoverse_hawor_rgb.md` §7 |
| 5 | post-trains load `$MIDPT` with no missing or unexpected keys |
| 5 (eval) | GR1 `atoms 37/37`, no `ATOM FAILED`; Tabletop every task `/50` (no `INCOMPLETE`), no `job FAILED`; every Tabletop `server.log` prints `latent contract: PLAIN_END+PER_VIEW(nb=0) ... state_t=8` |

## Files

| file | role |
|---|---|
| `hawor_rgb/build_md_play.py` | MimicDroid play episodes with video, from the MimicDroid release |
| `hawor_rgb/finalize_manifest.py` | `--md-from`: MimicDroid query rows + `query_eps_md.json` |
| `cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py` | the mid (`hawor_full_md`) |
| `cosmos_policy/config/experiment/egoverse_gr1_ret_experiment_configs.py`, `tabletop_sim_experiment_configs.py` | GR1 / pv0 post-trains for `MID=hawor_full_md` |
| `scripts/train_egoverse_rebuttal_4node.sh` | `STAGE=hawor_full_md`; `STAGE=gr1 / pv0` with `MID=hawor_full_md`; `EVAL=1`, `EVAL_ONLY=1` |
| `scripts/eval_after_post_4node.sh` | `EVAL=1`: convert, 4-node eval split, merge |
| `gr1_local_eval.sh` (`ATOM_SHARD`), `tabletop_sim_eval.sh`, `scripts/setup_tabletop_sim.sh` | the evals; Tabletop-Sim client from GitHub |
