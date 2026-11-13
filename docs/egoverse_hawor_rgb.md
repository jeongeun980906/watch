# RGB-only hand trajectories: HaWoR → pairs → mid-training → post-training (4-node groups)

This guide replaces WATCH's tracked hand trajectories with hands estimated from the egocentric video alone
([HaWoR](https://github.com/ThunderVVV/HaWoR), CVPR 2025), rebuilds the pairs from them, mid-trains WATCH on
them (one run), and post-trains that mid on GR1 v6 ICL and Tabletop pv0, and
evaluates GR1 (L1-L3). Every command is in this file, in order.

**Starting point:** the cluster setup of the randref / conf50 4-node runs. The data, the training environment,
the base checkpoint in the HF cache and the post-training data are already there and are reused unchanged.

## 0. Overview

### 0.1 The run

One arm, WATCH as in the paper with every hand trajectory taken from RGB (`STAGE=hawor_full`): the pairs are
matched on HaWoR trajectories, and the mid trains on HaWoR `state57`/`action57`. It is compared with the released
WATCH mid and its post-trains.

Kept from the recording: the **head pose** (the camera is fixed to the head; HaWoR's SLAM stage is not
run), the window table and task labels, and the normalization statistics of the v10 manifest.

### 0.2 Steps and time

| step | section | where | time on 4 nodes (8 / 16 nodes) |
|---|---|---|---|
| 1. HaWoR environment | 1 | once, shared filesystem | ~30 min |
| 2. HaWoR on every episode | 2 | **any number of nodes**, elastic | ~34 h stride 1, ~17 h stride 2 (17 / 8.5 h; 8.5 / 4.3 h) |
| 3. camera fit + `state57`/`action57` from HaWoR | 3 | 1 node, CPU | ~1 h |
| 4. pairs from HaWoR | 4 | 1 node (CPU + 1 GPU for T5) | ~1.5 h |
| 5. mid-training | 5 | one 4-node group | ~23 h |
| 6. two post-trainings (GR1, pv0) | 6 | two 4-node groups at once | ~7 h |
| 7. GR1 eval L1-L3; send the pv0 checkpoint | 7 | one 8-GPU node | ~6-7 h |

HaWoR rate: 0.12 s per frame per worker, measured on one RTX 5090 (H100 assumed equal). EgoVerse (human sources of
v10): 13,445 episodes, 32.4M frames ≈ 1,080 GPU-hours at one worker per GPU. Several workers per GPU (2.3) overlap the
CPU-bound detection with GPU inference; the times above are for one worker per GPU.

### 0.3 Common environment

On every node, for every step:

```bash
export ROOT=<the randref/conf50 data root>                 # holds egoverse-v10-state, egoverse-video, ...
export V10=$ROOT/egoverse-v10-state                        # episodes/ (tracked state57/action57) + manifest/
export EGOVERSE_HUM_ROOT=$ROOT/egoverse-video              # one dir per episode: images.front_1 + state57/action57 links
export QSET=$V10/manifest/query_eps_nomd.json              # copied in for randref
export WATCH=<this repository, the training checkout>
export WORK=$ROOT/hawor_rgb && mkdir -p $WORK              # everything this guide writes (~0.5 TB); shared by all nodes
export HF_HOME=<the shared HF cache of the randref runs>
```

---

## 1. HaWoR environment (once, on the shared filesystem)

```bash
cd $ROOT && git clone https://github.com/ThunderVVV/HaWoR.git && cd HaWoR
export HAWOR_REPO=$PWD HAWOR_PY=$ROOT/hawor_venv/bin/python
uv venv --python 3.10 $ROOT/hawor_venv && export VIRTUAL_ENV=$ROOT/hawor_venv
uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install pip setuptools wheel
uv pip install --no-build-isolation "chumpy @ git+https://github.com/mattloper/chumpy"
uv pip install "numpy<2" scipy opencv-python-headless natsort joblib yacs timm einops "smplx==0.1.28" ultralytics \
    scikit-image rich "pytorch-lightning==2.2.4" lightning-utilities "torchmetrics==1.4.0" huggingface_hub \
    omegaconf dill lapx numcodecs
mkdir -p weights/external weights/hawor/checkpoints
wget https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt -P weights/external/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/checkpoints/hawor.ckpt -P weights/hawor/checkpoints/
wget https://huggingface.co/ThunderVVV/HaWoR/resolve/main/hawor/model_config.yaml -P weights/hawor/
```

**MANO** (license, account at <https://mano.is.tue.mpg.de>): from `mano_v1_2.zip`, copy `MANO_RIGHT.pkl` to
`_DATA/data/mano/MANO_RIGHT.pkl` and `MANO_LEFT.pkl` to `_DATA/data_left/mano_left/MANO_LEFT.pkl`.
(`_DATA/data/mano_mean_params.npz` ships with the repository. DROID-SLAM, Metric3D and the infiller are not needed.)

**Check:**

```bash
cd $HAWOR_REPO && $HAWOR_PY -c "import torch, smplx, ultralytics; print(torch.__version__, torch.cuda.is_available())"
ls weights/external/detector.pt weights/hawor/checkpoints/hawor.ckpt _DATA/data/mano/MANO_RIGHT.pkl \
   _DATA/data_left/mano_left/MANO_LEFT.pkl
```

The T5-11B encoder of step 4 (~45 GB) also goes into the HF cache once:

```bash
$WATCH/.venv/bin/hf download google-t5/t5-11b config.json pytorch_model.bin spiece.model tokenizer.json
```

---

## 2. HaWoR on every episode (4, 8 or 16 nodes, changeable at any time)

### 2.1 Episode list

```bash
python3 $WATCH/hawor_rgb/make_episode_list.py --manifest $V10/manifest --video-root $EGOVERSE_HUM_ROOT \
    --out $WORK/episodes.txt
```

Expected: ~13,4xx episodes, ~32.4M frames. A `WARNING ... have no images.front_1` lists episodes whose video is not
downloaded; they are skipped by HaWoR and keep their tracked hands in `hawor_full` (section 3.2).

### 2.2 Smoke test (one node, ~5 min)

```bash
python3 $WATCH/hawor_rgb/make_episode_list.py --manifest $V10/manifest --video-root $EGOVERSE_HUM_ROOT \
    --out $WORK/episodes_smoke.txt --limit 4
HAWOR_OUT=$WORK/hawor_smoke EPISODES=$WORK/episodes_smoke.txt MAX_FRAMES=300 NGPU=1 WORKERS_PER_GPU=1 \
  HAWOR_REPO=$HAWOR_REPO HAWOR_PY=$HAWOR_PY bash $WATCH/hawor_rgb/launch_hawor.sh
grep -h "frames" $WORK/hawor_smoke/_logs/*.log     # 4 lines: "<ep> 300 frames ..s (0.1-0.2s/f) cov L 0.9x R 0.9x"
rm -rf $WORK/hawor_smoke
```

### 2.3 Full run

`launch_hawor.sh` runs **once per node** and starts `NGPU x WORKERS_PER_GPU` workers. The workers share a queue in
`$WORK/hawor`: an episode is claimed by creating `_claims/<ep>` (atomic) and is done when `<ep>.npz` exists. There
is no rank or world size, so:

- run the identical command on 4, 8 or 16 nodes, or on whatever is free; one k8s pod per node is fine, no
  rendezvous is needed;
- nodes can be added or stopped during the run; the episodes of a stopped node are reclaimed after `STALE`
  seconds (default 1 h) without a heartbeat;
- re-running resumes; finished episodes are skipped.

On every node:

```bash
export HAWOR_REPO=$ROOT/HaWoR HAWOR_PY=$ROOT/hawor_venv/bin/python
export HAWOR_OUT=$WORK/hawor EPISODES=$WORK/episodes.txt
export WORKERS_PER_GPU=3            # ~6 GB each; 3-6 fit on an 80 GB GPU
export TMPDIR=/local/scratch        # node-local disk for JPEG segments, ~0.2 GB per worker
# export STRIDE=2                   # every 2nd frame: half the time (conversion interpolates). Same value on every node.
bash $WATCH/hawor_rgb/launch_hawor.sh
```

Progress, from any node:

```bash
python3 $WATCH/hawor_rgb/hawor_status.py --episodes $WORK/episodes.txt --out $WORK/hawor --video-root $EGOVERSE_HUM_ROOT
# episodes 5210/13445 done, 96 running, 0 failed
# frames   12,345,678/32,400,000 (38.1%)
# rate     780 frames/s over the last hour -> ETA 7.1 h
```

After the first hour, compare the rate with `WORKERS_PER_GPU` and change it on the next (re)launch if the GPUs are
not saturated (`nvidia-smi`). Failed episodes are in `_failed/<ep>.txt`; delete the file to retry it.

Output: `$WORK/hawor/<ep>.npz` (`frames`, `joints_L/R` (n,21,3) in the camera frame, NaN where no hand,
`trans_L/R`, `F0`, `H`, `W`, `stride`), ~70 GB in total.

---

## 3. HaWoR → `state57`/`action57` (one node, CPU)

### 3.1 Per-source camera fit

HaWoR's depth scales with the assumed focal length, and the camera sits at a fixed offset on the head. One focal
length and one rigid camera → head transform per source are fitted against the tracked wrists of 60 finished
episodes per source (fit on one half, error on the other, then refit on all):

```bash
cd $WATCH/hawor_rgb
UVZ="uv run --no-project --python 3.11 --with zarr>=3 --with numpy --with scipy"
$UVZ python fit_cam.py --hawor-out $WORK/hawor --state-root $V10/episodes --manifest $V10/manifest \
    --out $WORK/cam_fit.json
```

Expected (from the 56-clip check of the rebuttal): f ≈ 325 (aria), 250 (scale), 275 (MECKA); held-out wrist error
median 40-55 mm. It can run as soon as ~60 episodes per source are finished.

### 3.2 Convert

```bash
$UVZ python to_state57.py --hawor-out $WORK/hawor --cam-fit $WORK/cam_fit.json --manifest $V10/manifest \
    --state-root $V10/episodes --video-root $EGOVERSE_HUM_ROOT --out-root $WORK/episodes_hawor --procs 32
export EGOVERSE_HAWOR_ROOT=$WORK/episodes_hawor
```

Per episode: camera joints → head frame (fit) → world (recorded head pose) → the same keypoint smoothing as the tracked data (EgoVerse `smooth_keypoints`; `--no-smooth` turns it off) → wrist frame (x: wrist → middle MCP, z:
palm normal) → `state57` (eef position, 6D rotation, 5 fingertips in the wrist frame), and `action57` exactly as
the data pipeline defines it. Frames without an estimate are interpolated; a hand seen in fewer than 30% of an
episode's frames is held at its first tracked pose. The video and `meta/` are symlinked, so
`$EGOVERSE_HAWOR_ROOT` is a drop-in `EGOVERSE_HUM_ROOT`.

Episodes without HaWoR output (no video, or failed) are not in `$EGOVERSE_HAWOR_ROOT`. Link their tracked
directories in so that training can read every episode it uses:

```bash
for d in $EGOVERSE_HUM_ROOT/*/; do e=$(basename $d); [ -e $EGOVERSE_HAWOR_ROOT/$e ] || ln -s $d $EGOVERSE_HAWOR_ROOT/$e; done
ls $EGOVERSE_HAWOR_ROOT | wc -l      # = ls $EGOVERSE_HUM_ROOT | wc -l
```

**Check:** `converted N, failed 0` and `mean coverage L ≥ 0.85, R ≥ 0.85`.

---

## 4. Pairs: `pairs_hawor.npz`
 The matcher is `71_3s.py`, the one that produced `egoverse_pairs_v10`
(`hawor_rgb/match.py`, verbatim `match()`); its inputs are rebuilt from `state57`:

| input | script | original |
|---|---|---|
| DTW sequence 40x66 + left-hand share | `build_desc.py` | `70_3s.py`, verbatim |
| shortlist descriptors M, E, G | `build_desc.py` | `01_3s.py` blocks, from `state57` instead of the raw keypoints |
| episode text embedding (same-source gate, cos ≥ 0.80) | `t5_text.py` | `02_ep_lang_act.py`, T5-11B on `task_description` |
| fine / coarse task buckets | `windows.npz` `cls` + `lang_taxonomy.py` | same |

```bash
cd $WATCH/hawor_rgb
UVZ="uv run --no-project --python 3.11 --with zarr>=3 --with numpy --with scipy"
UVF="uv run --no-project --python 3.11 --with numpy --with faiss-cpu"
# 4.1 descriptors (CPU, ~15 min with 48 processes)
$UVZ python build_desc.py --root $EGOVERSE_HAWOR_ROOT --manifest $V10/manifest --out $WORK/desc_hawor   --procs 48
# 4.2 episode text embeddings (1 GPU, ~2 min once T5-11B is cached)
$WATCH/.venv/bin/python t5_text.py --root $V10/episodes --out $WORK/temb.npz
# 4.3 matcher: 1.04M queries, ~1 h on one node with 48 processes
$UVF python match.py --desc $WORK/desc_hawor --temb $WORK/temb.npz --manifest $V10/manifest \
    --out $WORK/pairs_hawor_0.npz --shard 0/1 --procs 48
```

To spread 4.3 over n nodes, run `--shard i/n --out $WORK/pairs_hawor_$i.npz` on node i (i = 0..n-1).

```bash
# 4.4 manifest dir: v10 manifest links + query_eps_nomd.json + pairs_hawor.npz
python3 finalize_manifest.py --shards "$WORK/pairs_hawor_*.npz"   --base $V10/manifest --query-set $QSET \
    --name hawor   --out $WORK/manifest
export EGOVERSE_HAWOR_MANIFEST=$WORK/manifest
```

`finalize_manifest.py` removes the 150 held-out episodes as queries (as `73_v10.py` did; they stay available as
references), and prints refs per query, the cross-source share, the overlap with the released `pairs.npz`, and the
query-set check.

**Check:** it prints `query set query_eps_nomd.json: 13295/13295 episodes present`, ~14 refs/query, ~6
cross-source, 90%+ of rows with a cross-source reference.

---

## 5. Mid-training (one 4-node group)

The WATCH v10 state mid, launched like the randref 4-node mid: 4 nodes x 8 GPUs x batch 2 = global 64, 8K
iterations, the reference schedule (cycle 50K, warmup 500), MimicDroid-free query set. Only the pairs file
(`pairs_hawor.npz`) and the episode root (`$EGOVERSE_HAWOR_ROOT`) differ from WATCH.

Experiment: `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_hawor_full_nomd_4node`. The 4 nodes run the same
command with `NODE_RANK=0..3`, `MASTER_ADDR` = node 0, and an `IMAGINAIRE_OUTPUT_ROOT` on storage the 4 nodes share.

```bash
cd $WATCH && source .venv/bin/activate
export HF_HOME WANDB_MODE=online            # or offline / disabled
export EGOVERSE_HAWOR_MANIFEST=$WORK/manifest EGOVERSE_HUM_ROOT=$ROOT/egoverse-video EGOVERSE_HAWOR_ROOT=$WORK/episodes_hawor
```

### 5.1 Short test

```bash
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/hawor_smoke EXTRA_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20" \
  MASTER_ADDR=<node 0> MASTER_PORT=12343 STAGE=hawor_full NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
rm -rf /shared/runs/hawor_smoke
```

### 5.2 Launch

```bash
# the launcher sets EGOVERSE_HUM_ROOT=$EGOVERSE_HAWOR_ROOT for this stage
MASTER_ADDR=<node 0> MASTER_PORT=12343 IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_hawor_full_4node \
  STAGE=hawor_full NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

Resume after a crash: launch again with the same `IMAGINAIRE_OUTPUT_ROOT` and node count.

### 5.3 Convert the mid to `.pt`

After `iter_000008000`:

```bash
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_hawor_full_4node \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state_hawor_full_nomd_4node --iter 8000
MIDPT=/shared/runs/egoverse_ret_v3_448_state_hawor_full_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_hawor_full_nomd_4node/checkpoints/model_000008000.pt
ls -la $MIDPT      # ~3.9 GB
```

---

## 6. Post-training: GR1 v6 ICL and Tabletop pv0 (two 4-node groups at once)

Same recipe as the randref/conf50 post-trains, warm-started from `$MIDPT`.

| post-train | experiment | 4 nodes | iterations | read at |
|---|---|---|---|---|
| GR1 v6 ICL | `cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_hawor_full_4node` | batch 3/GPU = global 96 | 4K on the 8K schedule | 4K |
| Tabletop pv0 | `cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_hawor_full_4node` | batch 12/GPU = global 384 | 8K | 8K |

```bash
cd $WATCH && source .venv/bin/activate && export HF_HOME WANDB_MODE=online
export EGOVERSE_GR1_V6_MANIFEST=$ROOT/mimicdroid-gr1/manifest EGOVERSE_GR1_HUM_ROOT=$ROOT/mimicdroid-gr1/egoverse
export TABLETOP_RANDREF_DATA=$ROOT/tabletopsim-watch/data TABLETOP_RANDREF_MANIFEST=$ROOT/tabletopsim-watch/manifest
# group A: GR1 v6 ICL
MASTER_ADDR=<A node 0> MASTER_PORT=12344 GR1_BASE_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/gr1_posttrain_v6_448_state_hawor_full_4node \
  STAGE=gr1 MID=hawor_full NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
# group B: Tabletop pv0
MASTER_ADDR=<B node 0> MASTER_PORT=12345 TABLETOP_RANDREF_CKPT=$MIDPT \
  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/tabletop_pv0_hawor_full_4node \
  STAGE=pv0 MID=hawor_full NODE_RANK=<0..3> bash scripts/train_egoverse_rebuttal_4node.sh
```

The log prints the experiment and the checkpoint it loads; check that it is `$MIDPT`.

Convert for evaluation:

```bash
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/gr1_posttrain_v6_448_state_hawor_full_4node \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_hawor_full_4node --iter 4000
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/tabletop_pv0_hawor_full_4node \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_hawor_full_4node --iter 8000
```

---

## 7. Evaluation

### 7.1 GR1 v6 ICL: closed-loop eval, L1-L3 (one 8-GPU node)

Setup is [`gr1_eval_rebuttal_4node.md`](gr1_eval_rebuttal_4node.md) §1 unchanged (repos, robocasa venv, data,
environment variables of §1.4). If it was already set up for the randref eval, reuse it as is. Then, instead of the
randref checkpoint:

```bash
export HAWOR_CK=/shared/runs/gr1_posttrain_v6_448_state_hawor_full_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_hawor_full_4node/checkpoints/model_000004000.pt
export HAWOR_EXP=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_hawor_full_4node
export RES=$ROOT/gr1_hawor_eval && mkdir -p $RES
```

Smoke test (one episode, ~5 min), same pass criteria as §1.5 of the eval guide; delete `$RES/smoke` afterwards:

```bash
CK=$HAWOR_CK EXP=$HAWOR_EXP OUT=$RES/smoke MODE=full EPISODES=1 \
ONLY_TASKS="PnPSinkToCabinetL2" ONLY_DEMOS="1" GPUS="0" ./gr1_local_eval.sh
```

Launch all L1-L3 atoms (`MODE=full`: 12 seen-task atoms + 25 L2/L3 atoms = 37 atoms, 10 episodes each):

```bash
CK=$HAWOR_CK EXP=$HAWOR_EXP OUT=$RES/gr1_v6_hawor_full_4node_4000_full MODE=full \
GPUS="0 1 2 3 4 5 6 7" PORT0=8700 MPORT0=29600 \
  nohup ./gr1_local_eval.sh > $RES/hawor_full.log 2>&1 &
```

Rough time: ~6-7 h on 8 GPUs. A failed atom (`ATOM FAILED`) is re-run as in §2.2 of the eval guide.

**Check:** the run log prints `[eval] mode=full  atoms=37`, `[eval] exp=` / `[eval] ckpt=` are the HaWoR ones, and
§3 of the eval guide (f3r2 spawn, Close\* at `x/10`, no `SERVER DIED`).

**Send back:** `summary.txt` and `logs/` from `$RES/gr1_v6_hawor_full_4node_4000_full`, and `$RES/hawor_full.log`.

### 7.2 Tabletop pv0: send the checkpoint

The pv0 checkpoint is evaluated locally. Send the 8K `.pt` (~3.9 GB):

```
/shared/runs/tabletop_pv0_hawor_full_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_hawor_full_4node/checkpoints/model_000008000.pt
```

---

## 8. What to check

| after | check |
|---|---|
| 2 | `hawor_status.py`: `13445/13445 done, 0 failed` (or every failure explained); worker logs mostly `cov` ≥ 0.8 per hand |
| 3.1 | held-out wrist error median below ~60 mm per source; f in 250-350 |
| 3.2 | `converted N, failed 0`; mean coverage L/R ≥ 0.85; `$EGOVERSE_HAWOR_ROOT` has as many entries as `$EGOVERSE_HUM_ROOT` |
| 4 | `finalize_manifest.py`: full query set present; ~14 refs/query, ~6 cross-source |
| 5, 6 | pod log `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/<stage>_<mid>_node0.log`: `Training with 32 GPUs.`, `global batch 64 / 96 / 384` (no `WARNING: global batch`) |
| 5 | dataset lines: `query_episode_set=query_eps_nomd.json: kept 13295/...`, `ROBUST state norm ... state57_stats_staterobust.json`; episodes read from `$EGOVERSE_HAWOR_ROOT` (first launcher lines) |
| 5, 6 | LR: mid warmup 500 to 1e-4, 0.894e-4 at 8K; GR1 0.673e-4 at 4K; pv0 warmup 200, 0.5e-4 at 8K. Loss finite, iteration time stable |
| 6 | post-trains load the mid `.pt` with no missing or unexpected keys |

## Files

| file | role |
|---|---|
| `hawor_rgb/make_episode_list.py` | episode list, longest first |
| `hawor_rgb/launch_hawor.sh` | one per node; starts the elastic workers |
| `hawor_rgb/hawor_worker.py` | HaWoR camera-frame stage on whole episodes; claim-based queue |
| `hawor_rgb/hawor_status.py` | progress, ETA, failures |
| `hawor_rgb/fit_cam.py` | per-source focal + camera → head fit |
| `hawor_rgb/hawor_common.py` | conversion geometry |
| `hawor_rgb/to_state57.py` | HaWoR → `state57`/`action57` episode root |
| `hawor_rgb/build_desc.py` | 70_3s / 01_3s descriptors from any episode root |
| `hawor_rgb/t5_text.py` | episode text embeddings (T5-11B) |
| `hawor_rgb/match.py`, `lang_taxonomy.py` | 71_3s matcher, shardable |
| `hawor_rgb/finalize_manifest.py` | shards → `pairs_<name>.npz` + manifest dir |
| `cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py` | the mid experiment (`hawor_full`; `rematch` / `hawor_pairs` are defined but not run) |
| `cosmos_policy/config/experiment/egoverse_gr1_ret_experiment_configs.py`, `tabletop_sim_experiment_configs.py` | GR1 / pv0 post-trains for `MID=hawor_full` |
| `scripts/train_egoverse_rebuttal_4node.sh` | `STAGE=hawor_full` (mid), `STAGE=gr1 / pv0` with `MID=hawor_full` |
