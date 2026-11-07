# EgoVerse Cross-Embodiment Policy on Cosmos Predict-2.5 (cosmos_policy)

End-to-end pipeline for the EgoVerse demo-conditioned action policy built on
**Cosmos Predict2.5-2B (Rectified Flow)** in this repo (`cosmos_policy`).

Two stages:
1. **Mid-training** (`egoverse_ret`): retrieval-augmented policy pre-trained on paired
   human/robot demos — conditions on a retrieved demo (video+state+action) + current
   observation, generates a **future video + action chunk**.
2. **Post-training** (`egoverse_gr1_ret`): warm-started from a mid-training checkpoint,
   specialized to the **GR1 / MimicDroid** embodiment — conditions on the same signals,
   generates **action chunk + chunk-END image (3-cam)** for closed-loop robot control.

Sibling embodiments reuse the same machinery (`realrobot_ret`, `pusht_ret`, `robocasa_doors_ret`).


---

## 1. Environment setup

Cosmos_policy uses **uv** with a CUDA-12.8 torch build. The project venv is `.venv`
(torch 2.7.0+cu128). NEVER `pip install` into `.venv` (it drags torch to a CPU/cu130
build and breaks the model).

```bash
# create / restore the env (reads pyproject + uv.lock from the repo)
cd /mnt/ddn/cosmos_policy
uv sync --extra cu128 --group egoverse --python 3.10

# run anything in the env WITHOUT re-syncing
uv run --no-sync --extra cu128 --group egoverse --python 3.10 python <script>
```

- Offline model loading: `HF_HOME=/mnt/ddn/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.
  Base Predict2.5-2B + tokenizer are pre-seeded there so 4 nodes load offline.
- Datasets root: `BASE_DATASETS_DIR=/mnt/ddn/dataset`.
- Multi-node training runs in a k8s PyTorchJob; each pod builds a **pod-local** venv at
  `/workspace/.venv` via `UV_PROJECT_ENVIRONMENT` (4 pods must not uv-sync into one shared env).

---

## 2. Data: pairing & manifests

Descriptor + pair building lives in `/mnt/ddn/dataset/egoverse_pair_builder/` (env
`/mnt/ddn/EgoVerse/emimic/bin/python`). Episodes are zarr under `/mnt/ddn/dataset/egoverse/`.

### 2.1 Mid-training pairs (human↔robot retrieval)
- **v10 (3-second)** — `/mnt/ddn/dataset/egoverse_pairs_v10/` (+ `_test/`). Human WIN=90 @30Hz,
  MimicDroid WIN=60 @20Hz, STRIDE=30. ~1.09M windows. `t5_embeddings.pkl` + `action57_stats.json`.
  (v9 = 5-second predecessor at `egoverse_pairs_v9`.)
- Each manifest = `windows.npz` (ep, w0, cls, emb) + `pairs.npz` (query_idx, match_idx (N,K) padded -1).
- Descriptor: both-hand ABS-TRAJ 6D pose, no z-score, banded DTW (`78_future_pairs_abstraj.py`).

### 2.2 GR1 post-train pairs (query robot ↔ retrieved play/taskdemo)
- **1x demos only:** `/mnt/ddn/dataset/egoverse_pairs_gr1_1x/` — 232 episodes, all 3-cam,
  ~12.5k pairs. (2x → `_gr1_2x`; 5s/base → `egoverse_pairs_gr1`.)
- **MUST use `_gr1_1x` for the 1x recipe.** `egoverse_pairs_gr1` (no suffix) mixes in 12
  `taskdemo_*__2x` demos that lack `images.sideleft/sideright` → DataLoader FileNotFoundError.
  Side cams are added by `cosmos-framework/tools/add_concat_view_3cam.py` (`--twox` mode for 2x eps).
- t5 keyed by raw class (`CloseLeftCabinetDoor`, `PnPSinkToCabinet`, `PnPSinkToRightCounterPlate`,
  `TurnOnFaucet`, `play`); `action57_stats.json` / `state57_stats.json` (per-embodiment min-max;
  `mimicdroid` demos fall back to `_global` on both train & eval → self-consistent).

---

## 3. Model design (shared RF policy)

Class: `CosmosPolicyRealRobotRetModelRectifiedFlow`
(`cosmos_policy/models/policy_video2world_model_realrobot_ret.py`), a RAG subclass of the
Predict2.5 video2world rectified-flow model.

**Representation is TILING, not a learned head.** With `use_action_projection=False` /
`use_proprio_projection=False`, the action chunk `(chunk_size, 57)` and state `(57,)` are
**tiled** (flatten + repeat) directly into a single latent slot
(`replace_latent_with_action_chunk` / `replace_latent_with_proprio` in `policy_text2world_model.py`).
Set `use_action_projection=True` only to opt into an MLP encoder/decoder head — the EgoVerse
policies do NOT (all prior working configs tile).

**Latent layout** is data-driven via `*_latent_idx` fields the dataset emits. A latent is
CONDITIONING (kept clean, model input) if its `condition_video_input_mask` = 1, else GENERATED
(noised, predicted). The base prefix mask covers `[0, num_conditional_frames)`; the ret subclass
marks retrieved slots; and crucially:

> `policy_video2world_model_rectified_flow.py:~934` sets the ACTION latent's mask =
> **`world_model_sample_mask`**. So a sample with `world_model_sample_mask=1` conditions ON
> the ground-truth action (world-model mode) and the action is NEVER predicted; `=0` makes the
> action GENERATED (policy mode). **Policy datasets MUST emit `world_model_sample_mask=0`**
> (egoverse_ret, pusht_ret, realrobot_ret all do). Symptom of getting this wrong:
> `train/demo_sample_action_l1_loss` stuck at exactly 0 while image loss is large.

Per-component losses are logged by the ACTIVE callback `cosmos_policy/config/callbacks.py`
(`WandbCallback`) — `train/demo_sample_action_{l1,mse}_loss`, `.../future_image_l1`, etc.
(NOT `_src/predict2/callbacks/wandb_log.py`, which only logs the aggregate `edm_loss`).

---

## 4. Mid-training (egoverse_ret)


Config: `cosmos_policy/config/experiment/egoverse_ret_experiment_configs.py`
(`cosmos_predict2p5_2b_egoverse_ret_{1s,3s,5s}`).

- Dataset `EgoVerseRetDataset`; layout `R = num_frames//4`, `state_t = 2R+4`,
  conditioning prefix = `R+3` latents, `chunk_duration(pixel) = 8R+13`.
  - `_1s`: num_frames=28 (R=7, state_t=18). `_3s`: num_frames=92-lossless variant.
    `_5s`: num_frames=92 (R=23, state_t=50). **state_t ≤ 50** (Wan2.1 VAE caps latent T).
- CONDITION = retrieved demo (video 1..R + ret_state + ret_action) + current query image.
  GENERATE = future video (R+3..2R+2) + query action (2R+3, the LAST latent → safely generated).
- `chunk_size = num_frames` (one action per video frame), `action_loss_multiplier` up-weights
  the action latent (48 for 5s). `world_model_sample_mask=0` (policy). lr=1e-4.
- Launch (4-node × 8-GPU FSDP): `kube_yaml/cosmos-egoverse-ret-3s-4node.yaml` (and 5s variant).
  Checkpoints: DCP dirs `iter_000NNNNNN/` under `.../checkpoints/`, `save_iter` per config.

---

## 5. Post-training (egoverse_gr1_ret)

Config: `cosmos_policy/config/experiment/egoverse_gr1_ret_experiment_configs.py`
(`cosmos_predict2p5_2b_egoverse_gr1_ret_from{3s,5s}`).

**What post-train does:** take a trained mid-training checkpoint (broad human+robot retrieval
prior) and specialize it to the GR1/MimicDroid embodiment for closed-loop control, changing the
generation target from a full future video to a single chunk-END image + the action chunk.

- Warm-start: `checkpoint=dict(load_path=<mid-train model_000020000.pt>, strict_resume=False)`
  (differs in state_t / chunk / cams → non-strict loads the shared backbone).
- Dataset `EgoVerseGr1RetDataset`, manifest `egoverse_pairs_gr1_1x`, `chunk_size=32`,
  `per_frame_stride=4`, `final_image_size=224`. **`world_model_sample_mask=0`** (policy).
- **Latent layout (state_t=35, chunk_duration=137):**
  ```
  0 blank | 1-24 ret video(front/sideleft/sideright, 8 latents each) | 25 ret_state |
  26 ret_action | 27-29 cur image(3-cam) | 30 cur_state       <- conditioning (num_cond=31, idx 0-30)
  31 pred_action | 32-34 future image(3-cam)                   <- GENERATED
  ```
  Predicts **image + action only** — future-state and value prediction are disabled
  (`future_proprio_latent_idx=-1`, `value_latent_idx=-1`; both guarded in the model).
- Per-frame t+k: pairs are STRIDE-4 boundaries; one `delta∈{0..3}` is drawn per __getitem__ and
  applied to BOTH query and demo in lockstep, so every timestep is trained. The 32-step action
  chunk and 32-frame demo clips are CONSECUTIVE frames (stride 1).
- `action_loss_multiplier=16`, lr=1e-4, max_iter=20000, batch=4/GPU, save_iter=1000.
- Launch: `kube_yaml/cosmos-gr1-posttrain-2p5-from{3s,5s}-4node.yaml`
  (hardcode the EXP in the yaml — `GR1_EXP=... kubectl apply` does NOT propagate into the pod).
  Logs survive pod death at `/mnt/ddn/tmp/gr1_posttrain_2p5_from3s/pod_logs/worker_*.log`.
- When deleting checkpoints to restart, also wipe `wandb/` + `wandb_id.txt` (else a fresh run
  resumes the old wandb → step warnings).

---

## 6. Evaluation

### 6.1 Offline (in-distribution test + EgoDex held-out)
- `eval_egoverse_ret_batch.py` / `decode_dump.py` decode gen vs GT video + FK wrists, sharded
  across GPUs, saving JPEG-compressed npz.

### 6.2 Online (closed-loop GR1 / MimicDroid, robocasa)
- Server: `cosmos_policy/scripts/eval_gr1_server.py` — HTTP `/reset /push /predict`, builds the
  36-latent (now 35) data_batch, runs the RF model, extracts the action57 chunk
  (min-max de-norm → integrate to env actions). Speaks the same schema as the Cosmos3 server, so
  `cosmos-framework/scripts/eval_gr1_client.py` plugs in unchanged.
  ```
  uv run --no-sync --extra cu128 --group egoverse --python 3.10 \
    python -m cosmos_policy.scripts.eval_gr1_server --checkpoint <DCP dir> --port 8000 \
      --pool /mnt/ddn/dataset/egoverse_pairs_v2/taskdemo_pool.npz --pool-kind taskdemo \
      --play-pool /mnt/ddn/dataset/egoverse_pairs_v2/play_pool_1x.npz
  ```
- **Online retrieval** (`gr1_online_retrieval.py`, `OnlineRetriever`) reproduces the offline
  training pairs EXACTLY (78 abs-traj descriptor, verified 100% top-1 on 150 sampled pairs; the
  live env proprio path is byte-identical to the state57 descriptor). t=0 uses the taskdemo pool
  (forced `w0=0`), t>0 uses the play pool. Goal: ≥30% success.

---

## 7. Known pitfalls (learned the hard way)
- **Action never trains / `demo_sample_action_l1_loss=0`** → `world_model_sample_mask` must be 0
  for a policy dataset (see §3).
- **"NCCL/IB hang" at first collective** is usually a red herring — the checkpoint broadcast
  succeeds; the real cause is a worker dying without traceback (DataLoader FileNotFoundError on
  missing side-cams, or OOM/eviction). Don't chase SHARP/NVLS/NCCL_ALGO.
- **GR1 manifest** must be `_gr1_1x` (missing side-cams on 2x taskdemo episodes otherwise).
- **Never pip-install into `.venv`**; keep extra metric/eval deps in a separate env.
- viz FPS: 3s = 92 frames @30fps ≈ 3.07s; render at `--fps 30` (not 16).
