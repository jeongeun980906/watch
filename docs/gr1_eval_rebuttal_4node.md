# GR1 v6 ICL closed-loop eval of the 4-node randref post-train (remaining L2/L3 atoms)

This guide evaluates run 3 of [`egoverse_rebuttal_4node.md`](egoverse_rebuttal_4node.md): the GR1 v6 ICL
post-train from the randref mid. It runs on the 8-GPU cluster eval node. The conf50 post-train (run 5) is
evaluated locally and is not part of this guide.

Part of the randref eval already ran on a single local RTX 5090. One GPU took ~45 min per atom on average, so
what is left goes to the cluster:

| job | checkpoint | atoms | what is left |
|---|---|---|---|
| A | randref, 4K | 22 | all L2/L3 atoms except `PnPSinkToCabinetL2` |
| B | randref, 4K | 1 | `PnPSinkToCabinetL2` demo_1 (the local run was cut off at 0/3) |

The protocol is the certified v2 flag set that `gr1_local_eval.sh` already carries; nothing changes:

- `--oneshot-pool3`, `--max-steps 400`.
- The per-task flags of `MODE=full` (`--approach-demo-t0` on the L2/L3 Close tasks, floor 16 on TurnOnFaucetL3, ...).
- The f3r2 spawn on `PnPSinkToCabinet*`: the client default (anchor F+3/R+2 cm, jitter ±1 cm), with no flag.
- **10 episodes per atom on every task, Close\* included** (the script's `EPISODES=10`).

---

## 1. Setup

The eval runs in two environments: the policy server in this repo's venv, and the GR1 sim client in a separate
MimicDroid robocasa venv, because robocasa pins numpy 1.23.3 and mujoco 3.2.6. All code comes from GitHub and
all data from HF.

| what | source | env / variable |
|---|---|---|
| policy server, `gr1_local_eval.sh` | GitHub `jeongeun980906/watch` (`main`) | this repo's `.venv` |
| GR1 sim client (`eval_gr1_client.py`) | GitHub `jeongeun980906/cosmos-framework-backup` (`backup-snapshot`) | `FW` |
| robocasa (MimicDroid fork) | GitHub `jeongeun980906/mimicdroid-robocasa-backup` (`latest`) | `CLIENT_PY` (its venv) |
| robosuite (GR1 whole-body IK) | GitHub `ShahRutav/robosuite` (`abs_robot`) | in the robocasa venv |
| env-init demos (TaskDemos hdf5) | HF `Rutav/MimicDroidDataset`, `TaskDemos/` | `MIMICDROID_TASKDEMOS` |
| zarr episodes (`taskdemo_*`, `dthdemo_*`) | HF `Jeongeun/mimicdroid-gr1`: `data/*.tar` + `eval/taskdemo_l2l3.tar` | `EGOVERSE_GR1_HUM_ROOT` |
| GR1 v6 manifest | HF `Jeongeun/mimicdroid-gr1`, `manifest/` | `PAIRS_DIR` |
| server pools | HF `Jeongeun/mimicdroid-gr1`, `eval/pools/` | `POOL`, `PLAY_POOL` |
| checkpoint | HF `watch-2026/watch-2026-posttrain` (private org, needs access), or the `.pt` from `convert_dcp_to_pt.py --iter 4000` after training | `CK` |
| Cosmos-Predict2.5-2B base + VAE | HF `nvidia/Cosmos-Predict2.5-2B` | `HF_HOME` |

`$ROOT` below is any directory with ~30 GB free (the kitchen assets are ~21 GB).

### 1.1 Policy server (this repo)

```bash
cd $ROOT && git clone https://github.com/jeongeun980906/watch.git && cd watch
uv sync --extra cu128 --group egoverse --python 3.10
export HF_HOME=$ROOT/hf_cache
.venv/bin/hf download nvidia/Cosmos-Predict2.5-2B \
    base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt tokenizer.pth
```

### 1.2 Sim client (robocasa venv)

```bash
cd $ROOT
git clone -b abs_robot https://github.com/ShahRutav/robosuite.git robosuite-abs
git clone -b latest https://github.com/jeongeun980906/mimicdroid-robocasa-backup.git
GIT_LFS_SKIP_SMUDGE=1 git clone -b backup-snapshot https://github.com/jeongeun980906/cosmos-framework-backup.git

cd mimicdroid-robocasa-backup
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -e ../robosuite-abs -e .
uv pip install --python .venv/bin/python mink==0.0.5 numba==0.56.4 "imageio[ffmpeg]" h5py opencv-python
uv pip install --python .venv/bin/python numpy==1.23.3 mujoco==3.2.6    # last: the line above can lift numpy to 2.x
yes | .venv/bin/python robocasa/scripts/download_kitchen_assets.py      # ~21 GB into robocasa/models/assets
.venv/bin/python -c "import numpy, mujoco, mink; print(numpy.__version__, mujoco.__version__)"   # 1.23.3 3.2.6
```

- `GIT_LFS_SKIP_SMUDGE=1` is needed because one LFS image in `cosmos-framework-backup` is missing on the
  LFS server. The eval does not use it.
- `mink` must stay at **0.0.5**. Newer versions drop the `WHOLE_BODY_MINK_IK` controller without an error,
  and the GR1 arm then does not move.
- `numpy` must stay at **1.23.3**. Installing `imageio[ffmpeg]` upgrades it to 2.x without saying so.

### 1.3 Data and checkpoint

```bash
cd $ROOT
H=watch/.venv/bin/hf
$H download Rutav/MimicDroidDataset --repo-type dataset --include "TaskDemos/*" --local-dir MimicDroidDataset   # 24 hdf5
$H download Jeongeun/mimicdroid-gr1 --repo-type dataset --local-dir mimicdroid-gr1
cd mimicdroid-gr1 && (cd eval && sha256sum -c sha256.txt)
mkdir -p egoverse && ls data/*.tar | xargs -P 8 -I{} tar -xf {} -C egoverse
tar -xf eval/taskdemo_l2l3.tar -C egoverse
ls -d egoverse/taskdemo_* | wc -l      # 37 = 12 seen + 25 L2/L3
cd $ROOT
$H download watch-2026/watch-2026-posttrain \
    gr1_posttrain_v6_448_state_randref_4node/model_000004000.pt sha256.txt --local-dir ckpt
(cd ckpt && sha256sum -c sha256.txt --ignore-missing)
```

### 1.4 Environment variables (every launch)

```bash
cd $ROOT/watch && source .venv/bin/activate
export HF_HOME=$ROOT/hf_cache
export FW=$ROOT/cosmos-framework-backup
export CLIENT_PY=$ROOT/mimicdroid-robocasa-backup/.venv/bin/python
export MIMICDROID_TASKDEMOS=$ROOT/MimicDroidDataset/TaskDemos
export EGOVERSE_GR1_HUM_ROOT=$ROOT/mimicdroid-gr1/egoverse
export PAIRS_DIR=$ROOT/mimicdroid-gr1/manifest
export POOL=$ROOT/mimicdroid-gr1/eval/pools/taskdemo_pool_v6.npz
export PLAY_POOL=$ROOT/mimicdroid-gr1/eval/pools/dth_pool_1x_v6.npz
export RANDREF_CK=$ROOT/ckpt/gr1_posttrain_v6_448_state_randref_4node/model_000004000.pt
export RANDREF_EXP=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_randref_4node
export RES=$ROOT/gr1_rebuttal4_eval && mkdir -p $RES
```

Without these variables the scripts fall back to the old cluster paths under `/mnt/ddn`.

### 1.5 Smoke test (one episode, ~5 min)

```bash
CK=$RANDREF_CK EXP=$RANDREF_EXP OUT=$RES/smoke MODE=full EPISODES=1 \
ONLY_TASKS="PnPSinkToCabinetL2" ONLY_DEMOS="1" GPUS="0" ./gr1_local_eval.sh
```

The smoke test passes when:

- `$RES/smoke/summary.txt` has a line `PnPSinkToCabinetL2_demo_1 x/1`.
- `logs/task_PnPSinkToCabinetL2_demo_1.log` shows `default spawn: anchor F+3/R+2 cm`.

Delete `$RES/smoke` afterwards.

---

## 2. Launch

Each `GPUS` entry is a physical GPU and runs one server and one client. The server listens on `PORT0+g` and binds
`MPORT0+g`. Two runs on the same node at the same time need **different `PORT0` and `MPORT0`, and disjoint
`GPUS`**. Otherwise both runs die with `EADDRINUSE`, or a client connects to the other run's server.

### 2.1 One node (8 GPUs): A + B together

A takes 7 GPUs and B takes 1 GPU, at the same time:

```bash
# A: randref, remaining L2/L3 atoms (22 atoms)
CK=$RANDREF_CK EXP=$RANDREF_EXP OUT=$RES/gr1_v6_randref_4node_4000_l2l3 MODE=full \
ONLY_TASKS="PnPSinkToRightCounterPlateL2 PnPSinkToRightCounterPlateL3 CloseLeftCabinetDoorL2 CloseLeftCabinetDoorL3 CloseRightCabinetDoorL2 TurnOnFaucetL3 PnPSinkToMicrowaveTopL3" \
GPUS="0 1 2 3 4 5 6" PORT0=8700 MPORT0=29600 \
  nohup ./gr1_local_eval.sh > $RES/randref_l2l3.log 2>&1 &

# B: randref, PnPSinkToCabinetL2 demo_1 (1 atom)
CK=$RANDREF_CK EXP=$RANDREF_EXP OUT=$RES/gr1_v6_randref_4node_4000_cabL2d1 MODE=full \
ONLY_TASKS="PnPSinkToCabinetL2" ONLY_DEMOS="1" \
GPUS="7" PORT0=8800 MPORT0=29700 \
  nohup ./gr1_local_eval.sh > $RES/randref_cabL2d1.log 2>&1 &
```

### 2.2 Re-running an atom that failed

When a line `[gpuN] ATOM FAILED <task>_<demo>` appears (the client crashed, not a task failure), re-run only
that atom into the same `OUT`:

```bash
CK=... EXP=... OUT=<same OUT> MODE=full ONLY_TASKS="<task>" ONLY_DEMOS="<n>" GPUS="0" PORT0=8750 MPORT0=29650 ./gr1_local_eval.sh
```

This overwrites only that atom's `logs/task_<task>_demo_<n>.log`. The other logs stay, and `summary.txt` is
rebuilt from all of them.

---

## 3. What to check

1. **The checkpoint:** the script checks that the server log has the basename of `CK`, and stops if it does
   not. Check also that the run log has the right `[eval] exp=` and `[eval] ckpt=` (randref).
2. **Atom counts:** the log prints `[eval] mode=full atoms=N`: A `atoms=22`, B `atoms=1`.
3. **f3r2 spawn:** each `logs/task_PnPSinkToCabinet*_demo_*.log` has
   `default spawn: anchor F+3/R+2 cm + jitter ±1.0 cm`. It must not have `--raw-spawn`.
4. **Close\* at 10 ep:** every Close atom in `summary.txt` reads `x/10 ✓`.
5. **Server crashes:** check for `SERVER DIED`, `server never came up` and `ATOM FAILED`. For a failed
   atom, see 2.2.

## 4. Estimated time

These are measured on the local RTX 5090, one server and one client per GPU; the H200 was not measured.

| atom | time per episode | time per atom (10 ep) |
|---|---|---|
| succeeds early (TurnOnFaucet, Close\*) | ~2 min | ~20-35 min |
| PnP, success | ~4.6 min (~275 s) | |
| PnP, failure (all 400 steps) | ~7.4 min (~446 s) | up to ~75 min |

The per-atom times give these rough totals:

| job | rough time |
|---|---|
| A (22 atoms on 7 GPUs) | ~4-5 h |
| B (1 atom) | ~1.3 h |
| A + B side by side on one node | ~4-5 h |

## 5. What to send back

From each `OUT`, send `summary.txt` and the `logs/` folder (`task_*.log`, `server_gpu*.log`), plus the run log
`$RES/*.log`. The `videos_<task>/` folders are optional; they are large.

---

## 6. What already ran locally (randref, merged with A + B)

Local results, run with the same protocol (`eval_mimicdroid/results/gr1_v6_randref_4node_4000/`):

| atom | randref |
|---|---|
| TurnOnFaucet demo_1/2/3 | 10/10, 10/10, 10/10 |
| CloseLeftCabinetDoor demo_1/2/3 | 10/10, 10/10, 10/10 |
| PnPSinkToCabinet demo_1/2/3 | 2/10, 1/10, 2/10 |
| PnPSinkToRightCounterPlate demo_1/2/3 | 4/10, 6/10, 0/10 |
| PnPSinkToCabinetL2 demo_2 | 0/10 |
| PnPSinkToCabinetL2 demo_3 | (in progress) |

The randref full result is these 14 atoms plus A (22) plus B (1), 37 atoms in all.
