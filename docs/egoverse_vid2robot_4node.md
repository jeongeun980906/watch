# Vid2Robot-style baseline: mid → standard post-trainings → eval (4-node groups)

A controlled comparison of the pre-training formulation. The mid is trained with the
[Vid2Robot](https://arxiv.org/abs/2403.12943) recipe; backbone, training data and budget are those of WATCH. The
post-trainings and evals are the **standard** WATCH ones (GR1 v6 ICL, Tabletop pv0), unchanged; only the warm-start
mid differs. Every command is in this file, in order.

## 0. Overview

### 0.1 The runs

| run | experiment | nodes |
|---|---|---|
| mid | `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v2r_align_4node` | 4 |
| GR1 v6 ICL post | `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_v2r_align_4node` (standard recipe) | 4 |
| Tabletop pv0 post | `cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_v2r_align_4node` (standard recipe) | 4 |

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

### 0.2 Steps and nodes

| step | section | nodes | time |
|---|---|---|---|
| 1. MimicDroid play episodes | 1 | 1 (CPU) | ~5 min |
| 2. mid | 2 | 4 | ~15-17 h |
| 3. GR1 and pv0 post-trains at once | 3 | 2 x 4 = 8 | ~7 h |
| 4. eval: GR1 L1-L3, Tabletop pv0 | 4, 5 | 2 nodes of 8 GPUs | GR1 ~6-7 h, pv0 ~3 h |

The GR1 eval can start as soon as the GR1 post-train is converted; it does not wait for pv0.

**Starting point:** the cluster setup of the randref / conf50 4-node runs: data, training environment, base checkpoint in
the HF cache, GR1 and Tabletop post-training data. The GR1 eval setup is §1 of
[`gr1_eval_rebuttal_4node.md`](gr1_eval_rebuttal_4node.md); if it was done for the randref eval, it is reused as is.

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
```

---

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

## 3. Post-training (two 4-node groups at once)

```bash
cd $WATCH && source .venv/bin/activate && export WANDB_MODE=online
# group A: GR1 v6 ICL
MASTER_ADDR=<A node 0> MASTER_PORT=12347 GR1_BASE_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=$RUNS/gr1_posttrain_v6_448_state_v2r_align_4node \
  STAGE=gr1 MID=v2r_align NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
# group B: Tabletop pv0
MASTER_ADDR=<B node 0> MASTER_PORT=12348 TABLETOP_RANDREF_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=$RUNS/tabletop_pv0_v2r_align_4node \
  STAGE=pv0 MID=v2r_align NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

Check that each log prints the checkpoint it loads, and that it is `$MIDPT`.

Convert:

```bash
IMAGINAIRE_OUTPUT_ROOT=$RUNS/gr1_posttrain_v6_448_state_v2r_align_4node \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_v2r_align_4node --iter 4000
IMAGINAIRE_OUTPUT_ROOT=$RUNS/tabletop_pv0_v2r_align_4node \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_v2r_align_4node --iter 8000
```

---

## 4. GR1 eval, L1-L3 (one 8-GPU node)

Setup: [`gr1_eval_rebuttal_4node.md`](gr1_eval_rebuttal_4node.md) §1 (repos, robocasa venv, data, the environment
variables of §1.4). The post-train has the standard layout, so the standard eval server is used.

```bash
cd $WATCH
export V2R_CK=$RUNS/gr1_posttrain_v6_448_state_v2r_align_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_v2r_align_4node/checkpoints/model_000004000.pt
export V2R_EXP=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_v2r_align_4node
export RES=$ROOT/gr1_v2r_eval && mkdir -p $RES
```

Smoke test (one episode, ~5 min; pass criteria of §1.5 of the eval guide). Delete `$RES/smoke` afterwards:

```bash
CK=$V2R_CK EXP=$V2R_EXP OUT=$RES/smoke MODE=full EPISODES=1 \
  ONLY_TASKS="PnPSinkToCabinetL2" ONLY_DEMOS="1" GPUS="0" ./gr1_local_eval.sh
```

All L1-L3 atoms (37 atoms, 10 episodes each):

```bash
CK=$V2R_CK EXP=$V2R_EXP OUT=$RES/gr1_v6_v2r_align_4node_4000_full MODE=full \
  GPUS="0 1 2 3 4 5 6 7" PORT0=8700 MPORT0=29600 nohup ./gr1_local_eval.sh > $RES/v2r_align.log 2>&1 &
```

A failed atom (`ATOM FAILED`) is re-run as in §2.2 of the eval guide.

**Check:**
- The run log prints `[eval] mode=full  atoms=37` and `[eval] server=eval_gr1_server`.
- `[eval] exp=` and `[eval] ckpt=` are the `v2r_align` ones.
- §3 of the eval guide (f3r2 spawn, Close\* at `x/10`, no `SERVER DIED`).

---

## 5. Tabletop pv0 eval (one 8-GPU node)

### 5.1 Tabletop-Sim setup (once)

The simulator client runs in its own venv: Tabletop-Sim pins `numpy<2` / `mujoco==3.2.0`, the policy server is on
numpy 2.x, and the two talk over HTTP. Use the `Tabletop-Sim-backup` fork. It renders the `front` camera that the
training data has; the original repository does not, so its views would differ from training.

```bash
cd $ROOT && git clone https://github.com/jeongeun980906/Tabletop-Sim-backup.git && cd Tabletop-Sim-backup
git submodule update --init --recursive              # object meshes, ~2 GB
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python 'numpy<2'
export SIM_PY=$ROOT/Tabletop-Sim-backup/.venv/bin/python
```

The node needs EGL (headless rendering). **Check:**

```bash
MUJOCO_GL=egl $SIM_PY -c "import tabletop; e=tabletop.env('aloha_lift_box','ee_6d_pos'); \
  print(sorted(e.reset().observation['images']))"
# ['back', 'front', 'wrist_left', 'wrist_right']
```

The policy server uses the training venv (`$WATCH/.venv`), the HF cache (base model / VAE) and the Tabletop manifest
`$TABLETOP_RANDREF_MANIFEST` (T5 embeddings, action/state stats).

### 5.2 Run

The protocol is that of the rebuttal Tabletop tables: 50 trials per task, seed 195, `EXEC=16` (the
`tabletop_sim_eval.sh` default), guidance 1.5, 35 steps. It covers 10 tasks: EASY (5 training tasks), OOD (4 `*_new`
tasks) and the full `aloha_box_into_pot`. Each task runs as its own job, one GPU each, in two waves on 8 GPUs (~50-130 s
per episode).

```bash
cd $WATCH
export PV_CK=$RUNS/tabletop_pv0_v2r_align_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_v2r_align_4node/checkpoints/model_000008000.pt
export PV_EXP=cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_v2r_align_4node
export PV_RES=$ROOT/tabletop_v2r_eval && mkdir -p $PV_RES
TASKS_ALL=(aloha_dish_drainer aloha_handover_box aloha_shoes_table aloha_lift_box aloha_box_into_pot_easy \
           aloha_dish_drainer_new aloha_handover_box_new aloha_shoes_table_new aloha_lift_box_new aloha_box_into_pot)
# smoke test (2 episodes, ~5 min); delete $PV_RES/smoke afterwards
CKPT=$PV_CK EXP=$PV_EXP SIM_PY=$SIM_PY TASKS=aloha_lift_box TRIALS=2 OUT=$PV_RES/smoke ./tabletop_sim_eval.sh
# all tasks: GPU g runs tasks g and g+8
for g in 0 1 2 3 4 5 6 7; do
  ( for i in $g $((g + 8)); do
      t=${TASKS_ALL[$i]:-}; [ -n "$t" ] || continue
      CKPT=$PV_CK EXP=$PV_EXP SIM_PY=$SIM_PY TASKS=$t GPU=$g PORT=$((8010 + g)) MPORT=$((29010 + g)) \
        OUT=$PV_RES/$t ./tabletop_sim_eval.sh > $PV_RES/$t.log 2>&1
    done ) &
done; wait
```

To resume a task after a crash, run it again with `START=<first missing episode>` and a new `OUT`. Per-episode seeding
makes the resumed episodes identical to an uninterrupted run.

**Check:** every `server.log` prints `latent contract: PLAIN_END+PER_VIEW(nb=0) ... state_t=8`, and every task directory
has a `results.json` with 50 episodes.

---

## 6. What to check, and what to send back

| after | check |
|---|---|
| 1 | `sha256sum -c` all OK; 20 MimicDroid episodes with video in `$EGOVERSE_HUM_ROOT` (the mid launcher refuses to start otherwise) |
| 2, 3 | pod log `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/<stage>_<mid>_node0.log`: `Training with 32 GPUs.`, `global batch 64 / 96 / 384` (no `WARNING: global batch`) |
| 2 | dataset line `[EgoVerseRetDataset] pairs=1045007`; LR warmup 500 to 1e-4, 0.894e-4 at 8K; loss finite, iteration time stable |
| 2 | `[align]` lines: `valid` mostly `2/2`; `tcc` falls from ~0.09 (uniform-alignment level); `vv` falls from ~0.7 |
| 3 | both load `$MIDPT`; GR1 `state_t=20`, pv0 `state_t=8` (the standard layouts) |
| 4, 5 | checks of sections 4 and 5 |

**Send back:**
- GR1: `summary.txt` and `logs/` of `$RES/gr1_v6_v2r_align_4node_4000_full`, plus `$RES/v2r_align.log`
- Tabletop: `results.json`, `config.txt` and `client.log` of each `$PV_RES/<task>/`

## Files

| file | role |
|---|---|
| `cosmos_policy/datasets/egoverse_ret_v3_state_dataset.py` | `no_ret_action`, `no_future`; `align_query_video` (the query clip for the alignment losses) |
| `cosmos_policy/models/policy_video2world_model_align.py` | TCC + SigLIP on top of the flow loss; the second (query-video) forward |
| `cosmos_policy/networks/align_dit.py` | `MinimalV1LVGDiTAlign`: the DiT with the heads (`align_heads.*`; ignored by the plain `.pt` warm start of the post-trains) |
| `cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py` | mid (`_v2r_4node`; `v2r` without the aux losses is also defined) |
| `cosmos_policy/config/experiment/egoverse_gr1_ret_experiment_configs.py`, `tabletop_sim_experiment_configs.py` | standard GR1 / pv0 post-trains with tag `v2r_align` |
| `scripts/train_egoverse_rebuttal_4node.sh` | `STAGE=v2r_align`; `STAGE=gr1 / pv0` with `MID=v2r_align` |
| `gr1_local_eval.sh`, `tabletop_sim_eval.sh` | evals |
