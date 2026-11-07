# Rebuttal experiments from scratch: randref and noisy-label (conf50) runs

> Main model and paper post-training: see the [README](../README.md). This guide covers the two
> reviewer controls only (random same-task references, noisy task labels). The manifest builders it
> mentions (`responses/...`, `build_M.py`, `build_query_eps_*.py`) are offline tools kept with the
> analysis code and are not part of this repository; the manifests themselves are on Hugging Face.

This guide goes from an empty machine to the full random-reference (randref) chain: the randref
mid-training run and its GR1 v6 ICL and Tabletop pv0 post-trains. It uses two repositories:
`jeongeun980906/EgoVerse-backup` (data download, steps 3-4) and `jeongeun980906/watch-release` (training,
steps 5-7). Run each command from the root of the repository its step names.

Steps:

1. download the EgoVerse v10 state/action arrays and the retrieval manifest (Hugging Face),
2. download the random-reference manifest (Hugging Face),
3. download the episode videos (EgoVerse R2 bucket),
4. assemble the two into one episode root and verify it,
5. set up the training environment,
6. train `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k` on 2 nodes x 8 GPUs,
7. post-train the resulting mid on GR1 v6 (ICL) and Tabletop pv0 (section 7; time estimates in section 8).

Section 9 adds the noisy task-label run (conf50: full as the main run, q50 optional), which reuses steps 1, 3
and 5 and the same launcher.

Every step ends with a check. Do not skip the checks, and pay most attention to step 4: the video reader does
not raise on missing files, so a broken download trains without errors on blank frames.

---

## 0. What you end up with

```
$ROOT/
  egoverse-v10-state/              # HF Jeongeun/egoverse-v10-state (public)
    data/*.tar, data2/*.tar        #   13,469 per-episode tars (state57/action57/meta), 29.9 GB
    episodes/<episode>/            #   unpacked tars (step 1), ~47 GB
    manifest/                      #   retrieval manifest + normalization stats, 0.98 GB
      + pairs_randref.npz          #   copied in from step 2
      + query_eps_nomd.json        #   copied in from step 2
  egoverse-v10-randref/            # HF Jeongeun/egoverse-v10-randref (private, ask for access)
  egoverse-video/<episode>/        # R2 download (step 3), ~3.9 TB
    images.front_1/, obs_*/, ...   #   video + raw tracking arrays
    state57 -> ../../egoverse-v10-state/episodes/<episode>/state57    # symlinks (step 4)
    action57 -> ...
    meta -> ...
```

Training reads two paths:

| env var | points to | contents |
|---|---|---|
| `EGOVERSE_RANDREF_MANIFEST` | `$ROOT/egoverse-v10-state/manifest` | `windows.npz`, `pairs_randref.npz`, `query_eps_nomd.json`, `t5_embeddings.pkl`, `action57_stats.json`, `state57_stats*.json` |
| `EGOVERSE_HUM_ROOT` | `$ROOT/egoverse-video` | one directory per episode with `images.front_1` and `state57`/`action57` side by side |

**Disk:** plan for at least 4.1 TB on the volume that holds `egoverse-video` (3.9 TB of video plus headroom),
and ~80 GB for `egoverse-v10-state` (tars plus unpacked). You can delete the tars after step 1 if needed.
Training checkpoints are extra (step 6.6).

**MimicDroid is not part of this run.** The v10 manifest has 20 MimicDroid episodes (16,195 of 1,045,007
query rows, 1.55%). Their video is not in the EgoVerse R2 bucket, so a fresh download cannot train on them.
`query_eps_nomd.json` drops them as queries. That removes them completely, because no human query
references a MimicDroid window (checked by `build_query_eps_nomd.py`). All 13,295 episodes the run reads, as
query or as reference, are in `all_episodes.csv`.

---

## 1. State/action arrays and manifest (Hugging Face, ~30 GB)

```bash
export ROOT=/path/to/data          # pick a volume with >= 4.2 TB free
mkdir -p $ROOT && cd $ROOT

python3 -m venv ~/hfenv && ~/hfenv/bin/pip install -U huggingface_hub
~/hfenv/bin/hf auth login           # a token that can read Jeongeun/egoverse-v10-randref (step 2)

~/hfenv/bin/hf download Jeongeun/egoverse-v10-state --repo-type dataset --local-dir egoverse-v10-state
```

Unpack the episode tars into one directory. `data/` and `data2/` are only a storage split (HF allows at most
10,000 files per directory), not a train/test split. Each tar holds one `<episode>/` directory.

```bash
cd $ROOT/egoverse-v10-state && mkdir -p episodes
ls data/*.tar data2/*.tar | xargs -P 16 -I{} tar -xf {} -C episodes
```

**Check:**

```bash
ls episodes | wc -l                                   # 13469
diff <(ls episodes) <(sort episodes.txt) && echo same # every listed episode unpacked
ls episodes/2025-11-14-16-18-17-323000                # action57  meta  state57  zarr.json
ls manifest                                           # 14 files, incl. windows.npz pairs.npz t5_embeddings.pkl
                                                      # state57_stats_staterobust.json action57_stats.json
```

`state57_stats_staterobust.json` must be present. Without it the dataset recomputes the state normalization
on the first rank, which is slow and would not match the reference run.

---

## 2. Random-reference manifest (Hugging Face, 280 MB)

```bash
cd $ROOT
~/hfenv/bin/hf download Jeongeun/egoverse-v10-randref --repo-type dataset --local-dir egoverse-v10-randref
cp egoverse-v10-randref/manifest/pairs_randref.npz egoverse-v10-randref/manifest/query_eps_nomd.json \
   egoverse-v10-randref/manifest/query_eps_q50_nomd.json egoverse-v10-state/manifest/
```

The training manifest directory is then `$ROOT/egoverse-v10-state/manifest`. All the other files in it are the
unchanged v10 files that the WATCH reference run used.

**Check:**

```bash
python3 - <<'EOF'
import numpy as np, json, os
M = os.path.expandvars("$ROOT/egoverse-v10-state/manifest")
a, b = np.load(f"{M}/pairs_randref.npz"), np.load(f"{M}/pairs.npz")
assert a["match_idx"].shape == b["match_idx"].shape == (1045007, 20)
assert (a["query_idx"] == b["query_idx"]).all()                  # same queries as WATCH
q = json.load(open(f"{M}/query_eps_nomd.json"))
print(q["n_episodes"], q["n_rows"], q["episodes_per_embodiment"])  # 13295 1028812 {'MECKA': 6639, 'aria': 649, 'scale': 6007}
EOF
```

What `pairs_randref.npz` is (details in the HF README and `pairs_randref_check.json`): every reference slot of
WATCH's `pairs.npz` is replaced by a uniform draw from the windows with the same (source, fine task class).
Per query the number of references (10.52 on average) and their source/class mix (60.4% cross-source, 73.2%
same fine class) are identical to WATCH's. Only the motion ranking (DTW + handedness) is removed. 0.09% of the
links coincide with the original. Builder: `responses/y6hQ/scripts/build_randref_manifest.py`, seed 0.

---

## 3. Video (EgoVerse R2 bucket, ~3.9 TB)

`all_episodes.csv` lists the 13,449 human episodes of v10 (MECKA 6,718 / scale 6,074 / aria 657), with
columns `ep, emb, cls`. `fetch_raw.py` looks up each episode's current zarr path in the EgoVerse SQL episode
table and downloads it from R2.

### 3.1 Tools and credentials

```bash
git clone https://github.com/jeongeun980906/EgoVerse-backup.git
cd EgoVerse-backup

# AWS CLI (used only to read the R2/DB secrets)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && ./aws/install -i ~/aws-cli -b ~/bin

# public EgoVerse read-only keys: see EgoVerse-backup/README.md, section "AWS Configure"; region us-east-2
~/bin/aws configure
PATH=~/bin:$PATH ./egomimic/utils/aws/setup_secret.sh      # writes ~/.egoverse_env (R2 + DB, read-only)

# minimal Python env (3.11+)
python3.11 -m venv venv
venv/bin/pip install pandas boto3 sqlalchemy "psycopg[binary]" cloudpathlib numpy
```

### 3.2 Dry run: every episode must resolve

```bash
venv/bin/python watch_download/fetch_raw.py --list watch_download/all_episodes.csv --dry-run
# expected: resolved by source: {'MECKA': 6718, 'aria': 657, 'scale': 6074} | unresolved: {}
```

If anything is unresolved, stop here. Scale episodes were re-hashed on 2026-05-03. `fetch_raw.py` has a
fallback through `scale_task_id`, but `all_episodes.csv` has no such column, so the fallback never fires.
Unresolved scale episodes mean the DB hash changed again.

### 3.3 Full download

```bash
export VIDEO=$ROOT/egoverse-video
PATH=~/bin:$PATH nohup venv/bin/python watch_download/fetch_raw.py --list watch_download/all_episodes.csv \
    --full --out $VIDEO --workers 128 > fetch_all.log 2>&1 &
```

- The log prints one line per 500 episodes. Use `du -sh $VIDEO` for byte progress.
- Re-running skips files that already exist. boto3 downloads to a temporary name and renames on success, so
  an interrupted run never leaves a truncated file under the final name, though temporary files can be left
  behind. To resume, run the same command again.
- Sizes are uneven: MECKA ~0.14 GB, aria ~0.32 GB, and scale ~0.45 GB per episode on average, ranging from
  0.01 to 3.8 GB. Expect about 0.95 / 0.21 / 2.75 TB per source (estimated from 941 downloaded episodes).

### 3.4 Video storage formats (why the file counts look inconsistent)

`images.front_1` is a zarr v3 array of shape `(T,)`, with one JPEG per frame (variable-length bytes, zstd).
It is the only camera. The R2 episodes come in three on-disk layouts, depending on the source:

| source | `images.front_1/c/` | files per episode |
|---|---|---|
| aria | one shard `c/0` with every frame plus a trailing (offset, length) index | 1 |
| MECKA | shards of 1,000 frames: `c/0`, `c/1`, ... | ceil(T/1000), usually 2-4 |
| scale | either one shard `c/0`, or **one file per frame** `c/0` ... `c/T-1` | 1 or T |

`c/` is a directory in all three layouts. The cosmos_policy reader handles all of them. A scale episode stored
one file per frame contributes thousands of files, so the full set holds on the order of ten million small
files. Check the inode budget of the target filesystem before you start (`df -i`), and expect `du`/`ls` on
`$VIDEO` to be slow.

An episode is complete when all `ceil(shape / chunk_shape)` chunk files exist (`shape` and the outer
`chunk_grid.chunk_shape` come from `images.front_1/zarr.json`). Step 4 checks this for every episode.

---

## 4. Assemble the episode root and verify it

The R2 episodes do not have WATCH's `state57`/`action57`, and the loader reads video and state from the same
episode directory. `assemble.py` symlinks `state57`, `action57` and `meta` from the unpacked v10 episodes into
each video episode, then checks every episode that the run reads.

```bash
python3 watch_download/assemble.py \
    --video-root $ROOT/egoverse-video \
    --state-root $ROOT/egoverse-v10-state/episodes \
    --manifest   $ROOT/egoverse-v10-state/manifest \
    --pairs pairs_randref.npz --query-set query_eps_nomd.json \
    --link --check
```

Expected:

```
13295 episodes used by pairs_randref.npz restricted to query_eps_nomd.json
link: 39885 symlinks created; 0 episodes have no video directory yet
check: OK, all 13295 episodes complete (video chunks, state57, action57)
```

On failure the script exits with code 1 and lists every bad episode and the reason in
`$ROOT/egoverse-video/assemble_missing.txt`:

| reason | fix |
|---|---|
| `episode directory missing` / `video incomplete` | re-run step 3.3 (it resumes), then this step |
| `state57/state missing` | step 1 is incomplete for that episode, or `--link` was not passed |
| `video has N frames < state57 rows` | the R2 episode differs from the one v10 was built from; report it, do not train |

Notes:

- The symlinks are absolute. If you move `$ROOT`, delete them (`find $ROOT/egoverse-video -maxdepth 2 -type l
  -delete`) and run `--link` again.
- The R2 `zarr.json` at the episode root stays as it is. The loader does not read it.
- The run never reads the 154 episodes in `all_episodes.csv` that are not in the set (the held-out
  episodes). `--check` ignores them.

---

## 5. Training environment

### 5.1 Code

```bash
git clone https://github.com/jeongeun980906/watch-release.git
cd watch-release
```

The randref experiment and the `EGOVERSE_HUM_ROOT` override must be in your checkout:

```bash
grep -n "randref_nomd_15k" cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py   # 2+ hits
grep -n "EGOVERSE_HUM_ROOT" cosmos_policy/datasets/egoverse_ret_dataset.py                       # 1 hit
ls scripts/train_egoverse_randref_2node.sh
```

### 5.2 Python environment

Use the repo's Docker image (`SETUP.md`) or uv directly. Python 3.10, CUDA 12.8, torch 2.7:

```bash
# inside the container or on the host, from the cosmos_policy root
uv sync --extra cu128 --group egoverse --python 3.10
source .venv/bin/activate
python -c "import torch, megatron.core, numcodecs; print(torch.__version__, torch.cuda.device_count())"
```

The 448 state model needs about 21 GB per GPU at batch size 1 (measured on one GPU), so 40 GB or larger GPUs
are recommended for batch size 2. Use `--ipc=host` (or `--shm-size 32g`) in Docker, because each rank runs 4
dataloader workers.

### 5.3 Base checkpoint and VAE (Hugging Face)

The run starts from the Cosmos-Predict2.5-2B post-trained checkpoint and uses the Wan2.1 VAE shipped in the
same repo (`tokenizer.pth`; the resolved config has `vae_pth: hf://nvidia/Cosmos-Predict2.5-2B/tokenizer.pth`).
Accept the model license on Hugging Face with the account whose token you use, then cache the two files once on
every node (or on a shared `HF_HOME`), because the launch script runs with `HF_HUB_OFFLINE=1`:

```bash
export HF_HOME=/shared/hf_cache    # optional, same on all nodes
hf download nvidia/Cosmos-Predict2.5-2B \
    base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt tokenizer.pth
```

T5 is not needed, because the text embeddings are precomputed in `manifest/t5_embeddings.pkl`.

### 5.4 Weights & Biases

The experiment logs to W&B (`job.wandb_mode=online`). Either `wandb login` on every node, or launch with
`WANDB_MODE=offline` (or `disabled`). The launch script passes `WANDB_MODE` through as `job.wandb_mode`.

---

## 6. Training: `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k`

### 6.1 What the run is

It is the WATCH v10 state mid-training run (`cosmos_predict2p5_2b_egoverse_ret_v3_448_state`: Cosmos-Predict2.5-2B,
448 px full-frame front camera, 92 frames / 3 s windows, absolute state57 chunks, retrieval layout v3). Only the
references change, and the step budget is adapted to 2 nodes.

| | WATCH reference (already trained) | randref (this run) |
|---|---|---|
| experiment | `..._ret_v3_448_state` | `..._ret_v3_448_state_randref_nomd_15k` |
| references | `pairs.npz` (DTW + handedness top-20) | `pairs_randref.npz` (random, same source + fine class) |
| query rows | 1,045,007 (incl. MimicDroid) | 1,028,812 (MimicDroid dropped) |
| nodes x GPUs | 4 x 8 | **2 x 8** |
| batch / GPU | 2 | 2 |
| global batch | 64 | **32** |
| iterations | 8,000 | **15,000** |
| samples seen | 512,000 (0.490 epoch) | 480,000 (0.467 epoch) |
| LR | 1e-4, warmup 500, linear decay f_max 1.0 to f_min 0.3 | same |
| decay horizon (`cycle_lengths[0]`) | 50,000 | **93,750** (= 50,000 x 15,000 / 8,000) |
| LR multiplier at the last step | 0.894 | 0.891 |
| checkpoint every | 1,000 iters | 1,000 iters |
| init | Cosmos-Predict2.5-2B post-trained base | same |

The decay horizon is stretched by the same factor as the run length (`_scaled_scheduler`, the rule the
data-scaling arms also use). As a result the run ends at nearly the same point of its LR curve as the 8K
reference (multiplier 0.891 vs 0.894), instead of decaying further just because it has more steps. Peak LR and
warmup (500 steps) are unchanged. The small remaining gap comes from keeping warmup fixed. The multiplier after
warmup is `0.3 + 0.7 * (cycle - n) / (cycle - 500)`. Everything else
in the resolved config is identical to the reference. A full `--dryrun` config diff shows only the manifest
paths, `pairs_name`, `query_episode_set`, batch size, `cycle_lengths`, `max_iter` and the job name.

### 6.1b Optional: randref at q50 scale

`cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_q50_nomd_8k` is the cheaper version of the same control.
It is compared with the q50 data-scaling arm (`..._state_q50`, Fig. 9, 50% point) instead of the full WATCH mid.

| | q50 scaling arm (reference) | randref q50 |
|---|---|---|
| references | `pairs.npz` | `pairs_randref.npz` |
| query set | `query_eps_q50.json` (519,158 rows, incl. 10 MimicDroid eps) | `query_eps_q50_nomd.json` (511,746 rows) |
| nodes x GPUs, global batch | 4 x 8, 64 | 2 x 8, 32 |
| iterations / samples | 4,000 / 256K | 8,000 / 256K |
| decay horizon | 25,000 | 50,000 |
| LR multiplier at the last step | 0.900 | 0.894 |

Launch with `STAGE=randref_q50` (section 6.2, separate `IMAGINAIRE_OUTPUT_ROOT`). The manifest needs
`query_eps_q50_nomd.json` next to `pairs_randref.npz` (it is in the HF repo, step 2). The step 4 check with
`query_eps_nomd.json` already covers it. Estimated time: ~23 h. Post-train with `MID=randref_q50` (section 7),
converting with `--iter 8000`.

### 6.2 Launch (2 nodes x 8 GPUs)

Run the same script on both nodes. Node 0 is the rendezvous host.

```bash
# on BOTH nodes
cd /path/to/cosmos_policy && source .venv/bin/activate
export EGOVERSE_RANDREF_MANIFEST=$ROOT/egoverse-v10-state/manifest
export EGOVERSE_HUM_ROOT=$ROOT/egoverse-video
export IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_randref_2node
export MASTER_ADDR=<node-0 hostname or IP>
export MASTER_PORT=12341
export HF_HOME=/shared/hf_cache                # where step 5.3 cached the checkpoints
export WANDB_MODE=online                       # or offline / disabled

NODE_RANK=0 bash scripts/train_egoverse_randref_2node.sh   # on node 0
NODE_RANK=1 bash scripts/train_egoverse_randref_2node.sh   # on node 1
```

The script:

- checks that the manifest files exist, that `EGOVERSE_HUM_ROOT` exists, and that the node has 8 GPUs, and
  warns if the global batch is not 32;
- runs `torch.distributed.run --nnodes=2 --nproc_per_node=8 ... -m cosmos_policy.scripts.train
  --config=cosmos_policy/config/config.py -- experiment=..._randref_nomd_15k dataloader_train.batch_size=2
  trainer.grad_accum_iter=1 job.wandb_mode=$WANDB_MODE`;
- appends the output to `$IMAGINAIRE_OUTPUT_ROOT/pod_logs/<stage>_node<rank>.log`.

`IMAGINAIRE_OUTPUT_ROOT` should be on storage both nodes can see. Checkpoints are written collectively (DCP)
and resuming reads them from all ranks. With node-local disks, each node holds only its own shards and resume
breaks.

If you use the cluster's kubeflow PyTorchJob template instead of the script (as the 4-node reference run did),
copy the reference yaml and change only: job name / rdzvId / app label, `EXP`, `IMAGINAIRE_OUTPUT_ROOT`,
`nnodes: 2`, and the two env vars above. Keep `nproc_per_node=8` and `dataloader_train.batch_size=2`.

### 6.3 Short test before the long run

Run 20 iterations with the full launch first (same script, same nodes), using a separate output root so the test
checkpoint cannot be picked up by the real run:

```bash
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/randref_smoke EXTRA_OVERRIDES="trainer.max_iter=20 checkpoint.save_iter=20" \
  NODE_RANK=<0|1> bash scripts/train_egoverse_randref_2node.sh
```

Then delete `/shared/runs/randref_smoke`.

### 6.4 What to check in the first 30 minutes

In `pod_logs/mid_node0.log` (rank 0):

1. Dataset, printed once per rank:
   ```
   [EgoVerseRetDataset] query_episode_set=query_eps_nomd.json: kept 13295/13315 episodes, 1028812/1045007 rows
   [EgoVerseRetDataset] pairs=1028812 num_frames=92 R=23 chunk_duration=50
   [EgoVerseRetV3State] ROBUST state norm (mean+/-4std) <- .../manifest/state57_stats_staterobust.json
   ```
   If you see `computing ROBUST state57 stats`, the stats file is missing (step 1 check).
2. `Training with 16 GPUs.`
3. Base checkpoint loaded in non-strict mode. The only expected incompatible keys are the policy-specific
   additions (for example `role_embedder.weight` missing, and `crossattn_proj` keys unexpected).
4. Loss is finite and the iteration time is stable after the first ~50 iterations.
5. LR: it rises linearly over the first 500 iterations to 1e-4, then decays slowly: 0.981e-4 at iteration
   3,000, 0.948e-4 at 7,500, and 0.891e-4 at 15,000. If it is already near 0.89e-4 at 8K, the old 50K horizon
   is in effect, and you are not running the randref experiment.
6. GPU memory is stable and GPU utilization is high. Low utilization with workers at 100% CPU means data
   loading is the bottleneck: check the storage under `EGOVERSE_HUM_ROOT`, which serves two 92-frame 448 px
   videos per sample.

Measured on one local GPU with the real data: one sample (query and reference video, 2 x 92 JPEG decodes at
448, augmentation) takes ~1.6 s of CPU in a single process. Each rank has 4 workers.

### 6.5 Outputs and resume

```
$IMAGINAIRE_OUTPUT_ROOT/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k/
  config.yaml
  checkpoints/
    iter_000001000/ ... iter_000015000/      # DCP: model, optimizer, scheduler, trainer state
    latest_checkpoint.txt
```

To resume after a crash, launch again with the same `IMAGINAIRE_OUTPUT_ROOT`. The checkpointer reads
`latest_checkpoint.txt` and restores model, optimizer, scheduler and iteration. Do not change the batch size or
node count when resuming, because the sampler position and the epoch accounting assume global batch 32.

### 6.6 Converting a checkpoint for evaluation / post-training

Post-training configs load a `.pt` file, while training writes DCP directories:

```bash
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_randref_2node \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k --iter 15000
# -> .../checkpoints/model_000015000.pt   (~3.9 GB, same size as the reference model_000008000.pt)
```

DCP checkpoints include the optimizer state and are much larger than the `.pt`. With 15 checkpoints, budget the
disk accordingly, or delete intermediate `iter_*` directories you do not need.

### 6.7 Comparing with WATCH

The matched baseline is the existing WATCH v10 state checkpoint at iteration 8,000 (HF
`Jeongeun/cosmos-policy-egoverse-mid-v10`, `egoverse_ret_v3_448_state_4node/model_000008000.pt`). Keep these
differences in mind when you write up the comparison:

- randref sees 480K samples vs WATCH's 512K (0.94x), at global batch 32 vs 64;
- randref excludes MimicDroid (1.55% of rows), and WATCH includes it.

Downstream post-training and evaluation are unchanged: swap in the randref `.pt` as the mid checkpoint.

---

## 7. Post-training from the randref mid (GR1 v6 ICL, Tabletop pv0)

Both post-trains keep the reference recipe (data, layout, targets, loss, LR) and change only the warm-start mid,
from the WATCH v10 state mid 8K to the randref mid 15K. The references were trained on 3 nodes. On 2 nodes the
iteration count is scaled so that the number of samples is the same, and the LR decay horizon is scaled with
it, following the same rule as the mid.

| | GR1 v6 ICL reference | GR1 v6 randref | pv0 reference | pv0 randref |
|---|---|---|---|---|
| experiment | `..._egoverse_gr1_ret_v6_448_state` | `..._egoverse_gr1_ret_v6_448_state_randref_2node` | `..._tabletop_sim_5task_plain_end_a20_pv0` | `..._a20_pv0_randref_2node` |
| warm start | v10 state mid 8K | **randref mid 15K** | v10 state mid 8K | **randref mid 15K** |
| data | `mimicdroid-gr1` (61,686 query rows) | same | `tabletopsim-watch` (52,043 windows) | same |
| layout | state_t 20, chunk 77, 448, ICL (ret video + ret action) | same | state_t 8, chunk 29, 224 per-view, a20 | same |
| nodes x GPUs | 3 x 8 | **2 x 8** | 3 x 8 | **2 x 8** |
| batch / GPU, global | 4, 96 | 4, **64** | 16, 384 | 16, **256** |
| iterations | 8,000 schedule, ICL read at **4,000** | **6,000** (12K schedule) | 8,000 | **12,000** |
| samples | 384K (at 4K) | 384K | 3.07M | 3.07M |
| LR | 1e-4, warmup 500, linear to 0.3 over 8K (0.673 at 4K) | same, decay over **12K** (0.665 at 6K) | 1e-4, warmup 200, to 0.5 over 8K | same, decay over **12K** |
| checkpoint every | 1,000 | 1,000 | 1,600 | **2,400** |

GR1 v6 ICL is read at the reference's 4K checkpoint (3 nodes). The randref run therefore keeps the stretched
12K schedule and stops at 6K, which is the same sample count (384K) at nearly the same LR (0.665 vs 0.673). Pass
`EXTRA_OVERRIDES="trainer.max_iter=12000"` for the full 8K-equivalent.

A `--dryrun` config diff against the references shows only `load_path`, `max_iter`, `cycle_lengths`,
`save_iter` (pv0) and the job name.

### 7.1 Data (Hugging Face, private, ~7 GB)

```bash
cd $ROOT
~/hfenv/bin/hf download Jeongeun/mimicdroid-gr1    --repo-type dataset --local-dir mimicdroid-gr1     # 4.0 GB
~/hfenv/bin/hf download Jeongeun/tabletopsim-watch --repo-type dataset --local-dir tabletopsim-watch  # 3.1 GB
cd mimicdroid-gr1 && mkdir -p egoverse && ls data/*.tar | xargs -P 8 -I{} tar -xf {} -C egoverse
ls egoverse | wc -l          # 579
```

### 7.2 Convert the randref mid to `.pt`

The post-train configs load a `.pt` file:

```bash
IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_randref_2node \
  python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k --iter 15000
export RANDREF_MID=/shared/runs/egoverse_ret_v3_448_state_randref_2node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k/checkpoints/model_000015000.pt
```

### 7.3 Launch (same script, `STAGE=gr1` / `STAGE=pv0`)

Use a separate `IMAGINAIRE_OUTPUT_ROOT` per run, otherwise checkpoints and `latest_checkpoint.txt` collide.

```bash
# GR1 v6 ICL, on BOTH nodes (NODE_RANK=0 / 1)
export STAGE=gr1
export EGOVERSE_GR1_V6_MANIFEST=$ROOT/mimicdroid-gr1/manifest
export EGOVERSE_GR1_HUM_ROOT=$ROOT/mimicdroid-gr1/egoverse
export GR1_BASE_CKPT=$RANDREF_MID
export IMAGINAIRE_OUTPUT_ROOT=/shared/runs/gr1_posttrain_v6_448_state_randref_2node
NODE_RANK=<0|1> bash scripts/train_egoverse_randref_2node.sh

# Tabletop pv0, on BOTH nodes
export STAGE=pv0
export TABLETOP_RANDREF_DATA=$ROOT/tabletopsim-watch/data
export TABLETOP_RANDREF_MANIFEST=$ROOT/tabletopsim-watch/manifest
export TABLETOP_RANDREF_CKPT=$RANDREF_MID
export IMAGINAIRE_OUTPUT_ROOT=/shared/runs/tabletop_pv0_randref_2node
NODE_RANK=<0|1> bash scripts/train_egoverse_randref_2node.sh
```

Expected dataset lines in the log:

```
GR1 v6: [EgoVerseGr1RetDataset] pairs=61686 chunk=32 per_frame_stride=1 state_t=20 chunk_duration=77
pv0   : [TabletopSim] T5 coverage OK (10 instructions)
        [TabletopSim] TabletopSimPlainEndImageDataset root=.../tabletopsim-watch/data windows=52043 ... state_t=8 chunk_duration=29
```

The warm-start load should report no missing or unexpected keys (it is the same architecture as the reference
mid). After training, convert with `convert_dcp_to_pt.py --tag <experiment> --iter 6000` (GR1 v6) or `--iter 12000` (pv0), and evaluate exactly
like the reference arms.

---

## 8. Estimated wall-clock time

These numbers are estimated, not measured on the cluster. They come from two 1-GPU measurements of this
model family (state_t 12 at batch 1: 1.8 s/iter; state_t 50 at batch 1: 5.3-5.8 s/iter). Iteration time is
assumed to grow linearly with the tokens per GPU per step (about 0.12 s per 1K tokens plus 0.6 s). In FSDP
data-parallel runs the step time is close to the per-GPU compute time (the 3-node GR1 tabletop run measured
1.74 s/iter). Add 10-20% for multi-node communication and checkpoint writes. The first 30 minutes of each run
give the real number.

| stage | tokens / GPU / step | est. s/iter | iterations | est. time on 2 x 8 GPUs |
|---|---|---|---|---|
| randref mid | 2 x 50 x 784 = 78K | ~10.4 | 15,000 | **~43 h (~1.8 days)** |
| GR1 v6 ICL post | 4 x 20 x 784 = 63K | ~8.4 | 6,000 | **~14 h** |
| pv0 post | 16 x 8 x 196 = 25K | ~3.7 | 12,000 | **~12 h** |
| total, run one after another | | | | **~69 h (~2.9 days)**, plus ~1 h of DCP conversion and launch gaps |

The two post-trains only depend on the mid. With 4 nodes available they can run at the same time (2 nodes each),
which brings the total to ~57 h.

Data loading keeps up in all three stages: one sample costs ~1.6 s (mid), ~0.5 s (GR1 v6) and ~0.02 s (pv0)
of CPU in one process, against 4 (mid) or 8 (post) workers per GPU.

---

## 9. Noisy task-label run (conf50)

A second reviewer control uses the same data, the same environment and the same launcher as randref. Only the
references change: the task labels that gate cross-source pairing are corrupted. Run steps 1, 3 and 5 as above
(they are shared). Then do 9.1-9.4 instead of steps 2, 4 and 6. Post-training (step 7) is the same with
`MID=conf50`.

The **full** run is the main experiment and the one to run first. The **q50** run is optional. It is the
cheaper version and is compared with the q50 data-scaling arm (Fig. 9). Both use the same manifest file.

| | WATCH reference | conf50 full (main) | conf50 q50 (optional) |
|---|---|---|---|
| experiment | `..._ret_v3_448_state` / `..._state_q50` | `..._state_conf50_nomd_15k` | `..._state_conf50_q50_nomd_8k` |
| references | `pairs.npz` | `pairs_conf50_full.npz` | `pairs_conf50_full.npz` |
| query set | all (incl. MimicDroid) / `query_eps_q50.json` | `query_eps_nomd.json` (1,028,812 rows) | `query_eps_q50_nomd.json` (511,746 rows) |
| nodes x GPUs, global batch | 4 x 8, 64 | 2 x 8, 32 | 2 x 8, 32 |
| iterations | 8,000 / 4,000 | 15,000 | 8,000 |
| samples | 512K / 256K | 480K | 256K |
| decay horizon | 50,000 / 25,000 | 93,750 | 50,000 (= q50 arm's 25,000 x 64/32) |
| matched baseline | | WATCH v10 mid 8K (as in 6.7) | `..._state_q50` @ 4K |
| launch | | `STAGE=conf50` | `STAGE=conf50_q50` |

### 9.1 What `pairs_conf50_full.npz` is

The taxonomy labels enter the v10 pairing in one place only: the cross-source gate of the matcher (`71_3s.py`).
Same-source candidates are gated by T5 episode similarity, which needs no label. The noisy manifest therefore
starts from WATCH's `pairs.npz` and changes only cross-source references:

- **Noise:** 50% of the human episodes are drawn, and each drawn episode's fine task label is replaced by a
  different class of the same super-group (for example, a different dish-washing variant), with the class prior
  as weights. This is the confusion a language-to-taxonomy mapping produces. Labels are per episode and the
  seed is 0. Episodes whose super-group has no other class keep their label, so 30.7% of the human windows
  end up with a wrong fine label.
- **Where the gate does not change** (the query's noisy bucket is the same as its clean bucket): the original
  references stay, except those whose own label moved out of the bucket. Their scores are the original ones.
- **Where the gate changes** (the query's label was flipped): that source is re-matched in the new bucket with
  the matcher's cross branch: a shortlist of 40 windows, DTW plus handedness on the v9 descriptor, then keep
  `<= 1.6 x best`, at most 7. The selection is then applied again over all sources, and the merge with the
  same-source references takes the top 20 by score.
- **Unchanged:** every same-source reference and every MimicDroid row.
- **Check:** with clean labels the builder reproduces `pairs.npz` exactly (`match_idx` identical).
- **Changes:** 433,918 of 1,028,812 human queries (42%) change gate. 475,299 of 3,544,512 original cross-source
  references (13%) are dropped as no longer eligible. 558,039 query rows (54%) end up with a different reference
  set. Per human query (`pairs_conf50_full_check.json`):

  | | `pairs.npz` | `pairs_conf50_full.npz` |
  |---|---|---|
  | references | 10.37 | 9.94 |
  | cross-source references | 6.33 | 5.90 |
  | queries with a cross-source reference | 90.6% | 90.5% |
  | cross-source refs with the query's true fine class | 69.5% | 56.1% |
  | cross-source refs with the query's true super-group | 100% | 100% |
- **Pairing quality under this noise** (3,000 queries, Table 5 metrics, from the label-noise analysis): the
  share of cross-source references with the query's true fine class falls from 71.5% to 54.5%, the
  super-group stays at 100%, and motion agreement is unchanged (speed corr 0.203 -> 0.198, shape cos 0.334 -> 0.329).
- **Builder:** `responses/yw5w/scripts/noisy_manifest.py build conf50 <out> full`. The shortlist descriptor
  for re-matching is rebuilt from `state57` by `build_M.py`.
- **q50 rows:** the full file restricted to `query_eps_q50_nomd.json` is identical to a q50-only build, because
  labels are per episode and references are per query. So one file serves both runs.

### 9.2 Manifest (Hugging Face)

```bash
cd $ROOT
~/hfenv/bin/hf download Jeongeun/egoverse-v10-conf50 --repo-type dataset --local-dir egoverse-v10-conf50
cp egoverse-v10-conf50/manifest/pairs_conf50_full.npz \
   egoverse-v10-conf50/manifest/query_eps_nomd.json \
   egoverse-v10-conf50/manifest/query_eps_q50_nomd.json \
   egoverse-v10-state/manifest/
```

`EGOVERSE_CONF50_MANIFEST` is then `$ROOT/egoverse-v10-state/manifest`, the same directory as for randref.

**Check:**

```bash
python3 - <<'EOF'
import numpy as np, json, os
M = os.path.expandvars("$ROOT/egoverse-v10-state/manifest")
a, b = np.load(f"{M}/pairs_conf50_full.npz"), np.load(f"{M}/pairs.npz")
assert a["match_idx"].shape == b["match_idx"].shape == (1045007, 20)
assert (a["query_idx"] == b["query_idx"]).all()
for f in ("query_eps_nomd.json", "query_eps_q50_nomd.json"):
    q = json.load(open(f"{M}/{f}")); print(f, q["n_episodes"], q["n_rows"])
# query_eps_nomd.json 13295 1028812 ; query_eps_q50_nomd.json 6649 511746
EOF
```

### 9.3 Episode root

The video is the same as for randref (step 3). Run the step 4 check against the conf50 file, because the
re-matched references can point to episodes the randref file does not use:

```bash
python3 watch_download/assemble.py \
    --video-root $ROOT/egoverse-video --state-root $ROOT/egoverse-v10-state/episodes \
    --manifest $ROOT/egoverse-v10-state/manifest \
    --pairs pairs_conf50_full.npz --query-set query_eps_nomd.json --link --check
```

`query_eps_nomd.json` covers the q50 run too (q50 is a subset).

### 9.4 Launch

```bash
# on BOTH nodes
export EGOVERSE_CONF50_MANIFEST=$ROOT/egoverse-v10-state/manifest
export EGOVERSE_HUM_ROOT=$ROOT/egoverse-video
export MASTER_ADDR=<node-0 hostname or IP> MASTER_PORT=12341 HF_HOME=/shared/hf_cache WANDB_MODE=online

# main: full
export IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_conf50_2node
STAGE=conf50 NODE_RANK=<0|1> bash scripts/train_egoverse_randref_2node.sh

# optional: q50 (separate output root)
export IMAGINAIRE_OUTPUT_ROOT=/shared/runs/egoverse_ret_v3_448_state_conf50_q50_2node
STAGE=conf50_q50 NODE_RANK=<0|1> bash scripts/train_egoverse_randref_2node.sh
```

Expected dataset lines (rank 0):

```
full: [EgoVerseRetDataset] query_episode_set=query_eps_nomd.json: kept 13295/13315 episodes, 1028812/1045007 rows
q50 : [EgoVerseRetDataset] query_episode_set=query_eps_q50_nomd.json: kept 6649/13315 episodes, 511746/1045007 rows
```

The checks of 6.4 apply. LR for full is the same as randref (0.891e-4 at 15K). For q50 it reaches
0.894e-4 at 8K (the q50 arm ends at 0.900e-4 at 4K; the gap is the fixed 500-step warmup). Estimated time,
by the rule of section 8: full ~43 h, q50 ~23 h.

Convert for post-training as in 6.6, with the tags `cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_15k`
(`--iter 15000`) or `..._conf50_q50_nomd_8k` (`--iter 8000`).

### 9.5 Post-training

Same as section 7, with `MID=conf50` (or `MID=conf50_q50`) and the conf50 `.pt` as the warm start. The configs
`..._gr1_ret_v6_448_state_conf50{,_q50}_2node` and `..._pv0_conf50{,_q50}_2node` are the randref ones with only the
warm-start path and the job name changed.

```bash
export GR1_BASE_CKPT=<conf50 mid .pt>  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/gr1_posttrain_v6_448_state_conf50_2node
STAGE=gr1 MID=conf50 NODE_RANK=<0|1> bash scripts/train_egoverse_randref_2node.sh

export TABLETOP_RANDREF_CKPT=<conf50 mid .pt>  IMAGINAIRE_OUTPUT_ROOT=/shared/runs/tabletop_pv0_conf50_2node
STAGE=pv0 MID=conf50 NODE_RANK=<0|1> bash scripts/train_egoverse_randref_2node.sh
```

The env var names `GR1_BASE_CKPT` / `TABLETOP_RANDREF_CKPT` are shared with randref. Set them to the conf50
`.pt`, and use a separate `IMAGINAIRE_OUTPUT_ROOT` per run.

---

## 10. Troubleshooting

| symptom | cause / fix |
|---|---|
| `FileNotFoundError: .../images.front_1/zarr.json` | episode missing from `EGOVERSE_HUM_ROOT`; run step 4 `--check` |
| `AssertionError: ... episodes are not query episodes of ...` | `query_eps_nomd.json` does not match the manifest dir; `EGOVERSE_RANDREF_MANIFEST` points at the wrong directory |
| `missing .../pairs_randref.npz` from the launch script | step 2 copy not done into `egoverse-v10-state/manifest` |
| HF download error at start with `HF_HUB_OFFLINE=1` | step 5.3 not done on this node, or `HF_HOME` differs between nodes |
| NCCL timeout at start | the nodes cannot reach `MASTER_ADDR:MASTER_PORT`, or the two nodes were started with different env |
| slow first iterations, CPU-bound | storage throughput; see 6.4 item 6 |
| training runs but samples look blank | incomplete video; the reader returns empty frames for missing chunks. Run step 4 `--check` |

## Files

| file | purpose |
|---|---|
| `EgoVerse-backup/watch_download/all_episodes.csv` | the 13,449 human v10 episodes (`ep, emb, cls`) |
| `EgoVerse-backup/watch_download/fetch_raw.py` | DB lookup + R2 download (`--full` for video) |
| `EgoVerse-backup/watch_download/assemble.py` | link v10 state/action into the video episodes, and verify completeness |
| `scripts/train_egoverse_randref_2node.sh` | 2-node launcher, `STAGE=mid\|randref_q50\|conf50\|conf50_q50\|gr1\|pv0`, `MID=randref\|randref_q50\|conf50\|conf50_q50` for gr1/pv0 |
| `cosmos_policy/cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py` | `..._ret_v3_448_state_randref_nomd_15k` (mid), `..._randref_q50_nomd_8k` (optional q50 mid) |
| `cosmos_policy/cosmos_policy/config/experiment/egoverse_gr1_ret_experiment_configs.py` | `..._gr1_ret_v6_448_state_randref_2node` (GR1 v6 ICL) |
| `cosmos_policy/cosmos_policy/config/experiment/tabletop_sim_experiment_configs.py` | `..._a20_pv0_randref_2node` (pv0) |
| HF `Jeongeun/egoverse-v10-conf50` | `pairs_conf50_full.npz`, `query_eps_nomd.json`, `query_eps_q50_nomd.json` (builders: `responses/yw5w/scripts/noisy_manifest.py`, `build_M.py`, `build_query_eps_subset_nomd.py`) |
| `cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py` | `..._state_conf50_nomd_15k`, `..._state_conf50_q50_nomd_8k` (conf50 mids) |
| HF `Jeongeun/egoverse-v10-randref` | `pairs_randref.npz`, `query_eps_nomd.json` (builders: `build_randref_manifest.py`, `build_query_eps_nomd.py`) |
