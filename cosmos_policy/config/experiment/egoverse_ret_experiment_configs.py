"""Predict2.5 retrieval-augmented configs for the EgoVerse cross-embodiment set.

Generates the FULL query video + the query action chunk jointly, conditioned on
a retrieved demo video + its action + the query's first frame. No proprio/state.
Embodiment is injected via TEXT ([aria]/[scale]/[MECKA]/[eva] appended to the
activity-class command → T5).

Latent layout (R = num_frames // 4), built by ``EgoVerseRetDataset``:
    0          : blank
    1 .. R      : ret_video        (conditioning)
    R+1        : ret_action        (conditioning, injected)
    R+2        : query_image       (conditioning, first query frame)
    R+3 .. 2R+2: query_video       (GENERATED)
    2R+3       : query_action      (GENERATED)
  state_t = 2R+4,  chunk_duration(pixel) = 8R+13,  conditioning prefix = R+3 latents.

Two variants:
  * ``..._1s``  num_frames=28  (R=7,  state_t=18, chunk_duration=69)   — coarse 1 s view
  * ``..._5s``  num_frames=92  (R=23, state_t=50, chunk_duration=197)  — full 5 s window

NOTE: the wan2.1 VAE caps latent T at 50 (its precomputed video_mean/std buffers
are length-50), so state_t <= 50 => num_frames <= 92. The 5 s window is sampled
to 92 frames (~18 fps) to stay at the cap while still spanning the whole 5 s.

Reuses ``CosmosPolicyRealRobotRetModelRectifiedFlow`` — the ret machinery is
driven entirely by the data_batch latent_idx fields, so no new model class is
needed. Action is native action57 (head9|Leef9|Lhand15|Reef9|Rhand15) — v9 is
human-only, so the old 59-D union's eva grippers are dropped — tiled into one
latent slot (57*152=8664 ≤ 16*28*28=12544 latent elements).
"""
from __future__ import annotations

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.callbacks.egoverse_debug_sample import EgoVerseDebugSample
from cosmos_policy.datasets.egodex_ret_dataset import EgoDexRetDataset
from cosmos_policy.datasets.egoverse_ret_dataset import EgoVerseRetDataset
from cosmos_policy.models.policy_video2world_model_realrobot_ret import (
    CosmosPolicyRealRobotRetModelRectifiedFlow,
)

NUM_DUPLICATES = 4

_DATASET_KW = dict(
    manifest_dir="/mnt/ddn/dataset/egoverse_pairs_v9",  # v9 TRAIN split (150-ep holdout removed)
    t5_text_embeddings_path="/mnt/ddn/dataset/egoverse_pairs_v9/t5_embeddings.pkl",
    action_stats_path="/mnt/ddn/dataset/egoverse_pairs_v9/action57_stats.json",
    final_image_size=224,
    normalize_images=False,
    use_image_aug=True,
    use_stronger_image_aug=True,
    retrieval_dropout_prob=0.0,
)

# v10 = 3 s pairing (human WIN_eg=90 @30Hz, MD WIN_md=60 @20Hz). Same lang_taxonomy
# class vocab as v9, so t5_embeddings.pkl is reused (copied into the v10 dir).
# v10 manifest dir (HF Jeongeun/egoverse-v10-state, manifest/). EGOVERSE_V10_MANIFEST overrides the cluster path.
_V10_MANIFEST = __import__("os").environ.get("EGOVERSE_V10_MANIFEST", "/mnt/ddn/dataset/egoverse_pairs_v10")
_DATASET_KW_V10 = {
    **_DATASET_KW,
    "manifest_dir": _V10_MANIFEST,
    "t5_text_embeddings_path": f"{_V10_MANIFEST}/t5_embeddings.pkl",
    "action_stats_path": f"{_V10_MANIFEST}/action57_stats.json",
}

# COTRAIN (2026-07-30): v10 queries {aria,scale,MECKA,mimicdroid} + eva-ROBOT (now action57,
# was 0 queries in v10) + HE (Humanoid-Everyday, egoverse_he, emb 'he'), all self-matched.
# For BLANK mids only (demo content zeroed -> no DTW pairing needed). Per-emb robust STATE
# stats auto-compute from the windows; action falls back to _global for he/eva.
_DATASET_KW_V10_COTRAIN = {
    **_DATASET_KW,
    "manifest_dir": "/mnt/ddn/dataset/egoverse_pairs_v10_cotrain",
    "t5_text_embeddings_path": "/mnt/ddn/dataset/egoverse_pairs_v10_cotrain/t5_embeddings.pkl",
    "action_stats_path": "/mnt/ddn/dataset/egoverse_pairs_v10_cotrain/action57_stats.json",
}

# v11 = v10 pairing logic, REBUILT on the scale-corrupt-cleaned data (82 sensor-glitch
# scale eps removed/trimmed: hand-tracking bursts that inflated action/state variance).
# Same lang_taxonomy vocab as v10, so t5_embeddings.pkl is reused (copied into the dir).
# Data is already clean at the source, so NO pairs_clean.npz / percentile-stats workaround
# is needed — action57_stats.json (delta p1/p99) and state57_stats_staterobust.json
# (mean+/-4std) are computed over the full clean v11 pairs.
_DATASET_KW_V11 = {
    **_DATASET_KW,
    "manifest_dir": "/mnt/ddn/dataset/egoverse_pairs_v11",
    "t5_text_embeddings_path": "/mnt/ddn/dataset/egoverse_pairs_v11/t5_embeddings.pkl",
    "action_stats_path": "/mnt/ddn/dataset/egoverse_pairs_v11/action57_stats.json",
}

_SCHEDULER = dict(
    cycle_lengths=[50_000, 100000000000000],
    warm_up_steps=[500, 0],
    f_start=[1e-6, 0.06],
    f_max=[1.0, 0.06],
    f_min=[0.3, 0.06],
)
_OPTIMIZER = dict(lr=1e-4)


def _layout(num_frames: int) -> dict:
    assert num_frames % NUM_DUPLICATES == 0
    R = num_frames // NUM_DUPLICATES
    state_t = 2 * R + 4
    assert state_t <= 50, (
        f"state_t={state_t} > 50: wan2.1 VAE video_mean/std cap latent T at 50, "
        f"so num_frames must be <= 92 (got {num_frames})")
    chunk_duration = 8 * R + 13          # (state_t-1)*4 + 1
    n_cond = R + 3                        # blank + ret_video(R) + ret_action + query_image
    return dict(
        state_t=state_t,
        chunk_duration=chunk_duration,
        n_cond=n_cond,
    )


def _model_override(num_frames: int) -> dict:
    lo = _layout(num_frames)
    return dict(
        state_t=lo["state_t"],
        min_num_conditional_frames=lo["n_cond"],
        max_num_conditional_frames=lo["n_cond"],
        conditional_frames_probs={lo["n_cond"]: 1.0},
        tokenizer=dict(chunk_duration=lo["chunk_duration"]),
        text_encoder_class="T5",
        net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
        # action: native action57 tiled into one latent slot (no learned projection).
        # 57*num_frames ≤ 16*28*28 latent elements (5s: 57*152=8664 ≤ 12544).
        use_action_projection=False,
        use_proprio_projection=False,
        action_dim=57,
        proprio_dim=1,           # unused (current_proprio_latent_idx=-1)
        chunk_size=num_frames,   # action chunk length == video length
        action_loss_multiplier=48,   # bumped 16->48: push action (esp. small translation deltas)
        # full-video generation: keep all loss-mask flags off so every generated
        # latent (query_video + query_action) gets velocity loss, action upweighted.
        mask_loss_for_action_future_state_prediction=False,
        mask_value_prediction_loss_for_policy_prediction=False,
        mask_current_state_action_for_value_prediction=False,
        mask_future_state_for_qvalue_prediction=False,
    )


def _dataset(num_frames: int) -> LazyDict:
    return L(EgoVerseRetDataset)(num_frames=num_frames, **_DATASET_KW)


def _experiment(num_frames: int, name: str, batch_size: int, dataset: LazyDict | None = None) -> LazyDict:
    ds = dataset if dataset is not None else _dataset(num_frames)
    return LazyDict(
        dict(
            defaults=[
                "/experiment/cosmos_predict2p5_2b_480p_libero",
                "_self_",
            ],
            trainer=dict(
                max_iter=50_000,
                logging_iter=5,
                # Input/output debug video dump: top rows = generated (per guidance),
                # bottom = GT layout, cols = batch. Override every_n=1 on the CLI for a
                # one-shot dump. Saves local mp4 under <job>/EgoVerseDebugSample/.
                callbacks=dict(
                    egoverse_debug_sample=L(EgoVerseDebugSample)(
                        every_n=1000,
                        guidance=[1.5],
                        num_sampling_step=20,
                        n_viz_sample=2,
                        do_x0_prediction=False,
                        save_s3=False,
                        fps=16,
                        run_at_start=True,
                    ),
                ),
            ),
            scheduler=_SCHEDULER,
            optimizer=_OPTIMIZER,
            model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
                config=dict(**_model_override(num_frames)),
            ),
            dataloader_train=L(DataLoader)(
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                dataset=ds,
                sampler=L(DistributedSampler)(
                    dataset=ds,
                    num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                    rank=L(parallel_state.get_data_parallel_rank)(),
                    shuffle=True,
                    seed=0,
                ),
                batch_size=batch_size,
                drop_last=True,
            ),
            job=dict(group="cosmos_v2_finetune", name=name),
            upload_reproducible_setup=False,
        )
    )


# Per-GPU batch_size. Memory is NOT the binding constraint under FSDP (model+opt+
# grad shard across all ranks): profiled on FSDP shard=2, 5s — batch=2 → 11GB peak,
# batch=16 → 44GB peak / 60GB reserved (fits, no OOM); activation ≈ 2.4GB/sample, so
# at 4node×8GPU=32 (model state ~0.4GB/GPU) an H100 80GB holds ≥16/GPU comfortably.
# The real limit is GLOBAL batch = batch_size × world_size. At 32 GPUs:
#   batch_size=4 → global 128 (default here),  =8 → global 256.
# If you raise it, scale optimizer.lr accordingly (base lr=1e-4 was for small batch).
cosmos_predict2p5_2b_egoverse_ret_1s = _experiment(
    28, "cosmos_predict2p5_2b_egoverse_ret_1s", batch_size=4,   # 1s is cheaper; can go higher
)
cosmos_predict2p5_2b_egoverse_ret_5s = _experiment(
    92, "cosmos_predict2p5_2b_egoverse_ret_5s", batch_size=4,   # global 128 on 32 GPUs
)
# 3 s pairing (v10). num_frames=92 spans the 90-frame @30Hz human window with ZERO
# frame loss (all 90 native frames present, 2 duplicated to fill 92; MD 60→92 upsampled)
# — no subsampling, unlike 5s which sampled 92 of 150. Same tuned layout (state_t=50).
cosmos_predict2p5_2b_egoverse_ret_3s = _experiment(
    92, "cosmos_predict2p5_2b_egoverse_ret_3s", batch_size=4,
    dataset=L(EgoVerseRetDataset)(num_frames=92, **_DATASET_KW_V10),
)

# OFFLINE EVAL on the held-out EgoDex embodiment (Apple Vision Pro human hands).
# Identical 5 s model/layout (state_t=50, action57), but the dataset is swapped to
# EgoDexRetDataset (reads egodex_pairs_top3.npz + egodex_egoverse zarr, egodex-native
# per-emb normalization, text aliased to the trained [human] tag). Load a trained
# egoverse-ret checkpoint and run eval_egoverse_ret / _batch against this experiment.
cosmos_predict2p5_2b_egodex_ret_5s_eval = _experiment(
    92, "cosmos_predict2p5_2b_egodex_ret_5s_eval", batch_size=1,
    dataset=L(EgoDexRetDataset)(num_frames=92),
)
# 3 s egodex eval (matches the v10 3s model). EgoDex is 30 Hz → 3s = 90-frame windows
# (egodex_pairs_top3_3s.npz, win=90 stride=90); num_frames=92 covers all 90 frames (lossless).
cosmos_predict2p5_2b_egodex_ret_3s_eval = _experiment(
    92, "cosmos_predict2p5_2b_egodex_ret_3s_eval", batch_size=1,
    dataset=L(EgoDexRetDataset)(
        num_frames=92, pairs_path="/mnt/ddn/dataset/egodex_pairs_top3_3s.npz"),
)


def _inference_only(parent: str, name: str) -> LazyDict:
    return LazyDict(dict(
        defaults=[f"/experiment/{parent}", "_self_"],
        job=dict(group="cosmos_v2_inference", name=name),
    ))


cosmos_predict2p5_2b_egoverse_ret_1s__inference_only = _inference_only(
    "cosmos_predict2p5_2b_egoverse_ret_1s",
    "cosmos_predict2p5_2b_egoverse_ret_1s__inference_only",
)
cosmos_predict2p5_2b_egoverse_ret_5s__inference_only = _inference_only(
    "cosmos_predict2p5_2b_egoverse_ret_5s",
    "cosmos_predict2p5_2b_egoverse_ret_5s__inference_only",
)


# ---------------------------------------------------------------------------
# V2: length-GENERAL compact layout — per-token slot-ROLE embedding + CONTINUOUS-
# TIME rope (docs/egoverse_mid_v2_design.md). Native lengths (human 92f / MD 60f,
# no resample/padding), 4 sequence shapes served by BucketedDistributedSampler.
# num_conditional_frames is PER-SAMPLE (k+3, from the data batch); the model-config
# min/max below are informational bounds only.
# ---------------------------------------------------------------------------
from cosmos_policy.datasets.egoverse_ret_v2_dataset import (
    BucketedDistributedSampler,
    EgoVerseRetV2Dataset,
)

_V2_DATASET = L(EgoVerseRetV2Dataset)(num_frames=92, **_DATASET_KW_V10)

_V2_MODEL_OVERRIDE = dict(
    state_t=50,                        # max bucket (human T=50); MD=34, cross=42
    min_num_conditional_frames=18,     # MD bucket k+3
    max_num_conditional_frames=26,     # human bucket k+3
    conditional_frames_probs=None,     # per-sample batch key takes precedence
    tokenizer=dict(chunk_duration=197),
    text_encoder_class="T5",
    net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
    use_action_projection=False,
    use_proprio_projection=False,
    action_dim=57,
    proprio_dim=57,                    # state57 tiled at latent 0
    chunk_size=92,                     # max native chunk (per-sample native length used)
    action_loss_multiplier=48,
    mask_loss_for_action_future_state_prediction=False,
    mask_value_prediction_loss_for_policy_prediction=False,
    mask_current_state_action_for_value_prediction=False,
    mask_future_state_for_qvalue_prediction=False,
)

cosmos_predict2p5_2b_egoverse_ret_v2 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=20_000, logging_iter=5),
        scheduler=_SCHEDULER,
        optimizer=_OPTIMIZER,
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_V2_DATASET,
            sampler=L(BucketedDistributedSampler)(
                dataset=_V2_DATASET,
                batch_size=4,          # MUST equal dataloader batch_size (same-shape blocks)
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_ret_v2"),
        upload_reproducible_setup=False,
    )
)

# ---------------------------------------------------------------------------
# V3: ablation — EXACT clone of the v2 experiment, but mid-training video is the
# primary camera decoded FULL-FRAME at 224 (no 2x2 grid padding, no side cams;
# EgoVerseRetV3Dataset). GR1 post-training keeps the grid, so v3-vs-v2 probes
# (tmp/v3_gate_probe.py [C] grid vs [C2] full-frame) measure the image-format
# transfer gap cleanly. Same model override / batch / schedule as v2.
# ---------------------------------------------------------------------------
from cosmos_policy.datasets.egoverse_ret_v3_dataset import EgoVerseRetV3Dataset

_V3_DATASET = L(EgoVerseRetV3Dataset)(num_frames=92, **_DATASET_KW_V10)

cosmos_predict2p5_2b_egoverse_ret_v3 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=20_000, logging_iter=5),
        scheduler=_SCHEDULER,
        optimizer=_OPTIMIZER,
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_V3_DATASET,
            sampler=L(BucketedDistributedSampler)(
                dataset=_V3_DATASET,
                batch_size=4,          # MUST equal dataloader batch_size (same-shape blocks)
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_ret_v3"),
        upload_reproducible_setup=False,
    )
)

# ---------------------------------------------------------------------------
# 448-resolution variants: 2x2 grid quadrant becomes 224/cam (native res) for v2;
# v3 full-frame gets 448. Base Predict2.5 is 480p-trained so 448 is in-distribution
# (net max_img_h/w=240 latent >> 448/8=56; rope len_h/len_w=120 tokens >> 28).
# Spatial tokens x4 vs 224 -> batch_size=1. Cost is measured by
# /mnt/ddn/tmp/res448_speed_probe.py before committing to a 448 mid retrain.
# ---------------------------------------------------------------------------


def _ret_448_experiment(name: str, dataset_cls, batch_size: int = 1,
                        base_kw: dict = _DATASET_KW_V10, model_override: dict | None = None,
                        max_iter: int = 20_000, scheduler: dict = _SCHEDULER,
                        **ds_kw) -> LazyDict:
    ds = L(dataset_cls)(num_frames=92, **{**base_kw, "final_image_size": 448, **ds_kw})
    return LazyDict(
        dict(
            defaults=[
                "/experiment/cosmos_predict2p5_2b_480p_libero",
                "_self_",
            ],
            trainer=dict(max_iter=max_iter, logging_iter=5),
            scheduler=scheduler,
            optimizer=_OPTIMIZER,
            model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
                config=dict(**(model_override or _V2_MODEL_OVERRIDE)),
            ),
            dataloader_train=L(DataLoader)(
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                dataset=ds,
                sampler=L(BucketedDistributedSampler)(
                    dataset=ds,
                    batch_size=batch_size,   # MUST equal dataloader batch_size
                    num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                    rank=L(parallel_state.get_data_parallel_rank)(),
                    shuffle=True,
                    seed=0,
                ),
                batch_size=batch_size,
                drop_last=True,
            ),
            job=dict(group="cosmos_v2_finetune", name=name),
            upload_reproducible_setup=False,
        )
    )


cosmos_predict2p5_2b_egoverse_ret_v2_448 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v2_448", EgoVerseRetV2Dataset)
cosmos_predict2p5_2b_egoverse_ret_v3_448 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448", EgoVerseRetV3Dataset)

# STATE-CHUNK prediction variant: exact clone of ..._ret_v3_448, dataset swapped to
# EgoVerseRetV3StateDataset — "actions"/"retrieved_actions" carry FUTURE state57
# ABSOLUTE chunks (not action deltas) and slot 0 carries the PAST state chunk
# (tiled like the action chunk). Model override / layout / batch UNCHANGED
# (tensor keys and shapes are identical to the delta variant).
from cosmos_policy.datasets.egoverse_ret_v3_state_dataset import EgoVerseRetV3StateDataset

cosmos_predict2p5_2b_egoverse_ret_v3_448_state = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state", EgoVerseRetV3StateDataset)

# CLEAN mid retrain (2026-07-12): corrupt-scale-eps-filtered pairs (pairs_clean.npz —
# 9.5% of pairs dropped: 95 sensor-glitch eps as query and/or conditioning demo) +
# PERCENTILE normalization (state57_stats_pctl.json: per-emb p0.5/p99.5 over
# corrupt-excluded frames, symmetric degenerate-dim guard). Fixes the precision
# floor: scale pos norm range 7m -> ~3m, scale L-hand 794 -> 0.19 (glitch eps had
# inflated the 4-sigma robust stats).
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_clean = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_clean", EgoVerseRetV3StateDataset,
    batch_size=2,   # 4 was too slow on the cluster (2026-07-13)
    pairs_name="pairs_clean.npz",
    robust_stats_file="/mnt/ddn/dataset/egoverse_pairs_v10/state57_stats_pctl.json")

# ============================ v11 (scale-cleaned rebuild) ============================
# v11 = the DELTA and STATE mid-training pair, rebuilt on the corrupt-scale-cleaned data.
# Same v3@448 model/layout/schedule as the v10 variants above; only the pairs dir + its
# freshly-computed clean stats differ (base_kw=_DATASET_KW_V11). Because the corruption is
# gone at the source, the state variant uses the DEFAULT robust stats (mean+/-4std over the
# full clean pairs, state57_stats_staterobust.json) — no pairs_clean.npz / pctl override.
#   * ..._v11        : DELTA action57 target (EgoVerseRetV3Dataset)
#   * ..._state_v11  : ABSOLUTE state57 target (EgoVerseRetV3StateDataset)
cosmos_predict2p5_2b_egoverse_ret_v3_448_v11 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_v11", EgoVerseRetV3Dataset,
    base_kw=_DATASET_KW_V11)
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11)
# BLANK-DEMO mid: identical FULL transfer layout to _state_v11 (RET_ACTION + RET_VIDEO
# slots, roles 1/2, past time axis) but the demo CONTENT is zeroed (blank_retrieval=True).
# Learns the retrieval STRUCTURE with no retrieval information -> still BC, but seeds a
# transfer post-train (unlike a pure no-demo plain mid, which seeds it no better than base).
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_v11 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_v11", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11, blank_retrieval=True)
# CAPPED blank-demo: same as blankret but the RET_VIDEO block is only k blank latents
# (blank_video_slots) instead of the full demo length. Tests whether the transfer-seeding
# structure needs the whole video block or just a couple of ret slots.
#
# ⚠️ The capped layout has a SMALLER sequence than the full transfer layout, so the model
# config MUST match the data (unlike the full/transfer mid which use _V2_MODEL_OVERRIDE
# state_t=50). Layout (max bucket = human query, g = num_frames//4 = 23; o=1 ret_action,
# a=1 action):
#     [cur_state | ret_action | ret_video(k) | cur_image | action | future(g)]
#   state_t (max)   = k + g + 4        = k + 27      (human bucket; MD bucket is smaller)
#   num_conditional = k + 3            (cur_state + ret_action + k ret_video + cur_image;
#                                       equals the action latent index -> first generated slot)
#   chunk_duration  = 1 + 4*(state_t-1)
# state_t is functional (video_noise_multiplier = sqrt(state_t)); a wrong state_t=50 scales
# the flow-matching noise by sqrt(50) instead of sqrt(31) (bv4). num_conditional_frames is
# per-sample from the data batch (overrides min/max here), but set consistently for hygiene.
def _blankret_capped_override(k: int) -> dict:
    state_t = k + 27
    n_cond = k + 3
    return dict(**{**_V2_MODEL_OVERRIDE,
                   "state_t": state_t,
                   "min_num_conditional_frames": n_cond,
                   "max_num_conditional_frames": n_cond,
                   "tokenizer": dict(chunk_duration=1 + 4 * (state_t - 1))})


cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv1_v11 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv1_v11", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11, blank_retrieval=True, blank_video_slots=1,
    model_override=_blankret_capped_override(1))   # state_t=28, chunk_duration=109, n_cond=4
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_v11 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_v11", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11, blank_retrieval=True, blank_video_slots=2,
    model_override=_blankret_capped_override(2))   # state_t=29, chunk_duration=113, n_cond=5
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv4_v11 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv4_v11", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11, blank_retrieval=True, blank_video_slots=4,
    model_override=_blankret_capped_override(4))   # state_t=31, chunk_duration=121, n_cond=7
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv8_v11 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv8_v11", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11, blank_retrieval=True, blank_video_slots=8,
    model_override=_blankret_capped_override(8))   # state_t=35, chunk_duration=137, n_cond=11

# ===================== v10 (noisy original) blank/bv variants =====================
# Same blank-demo layouts as the _v11 ones above, but on the ORIGINAL NOISY v10 data
# (_DATASET_KW_V10 -> egoverse_pairs_v10, default pairs.npz = 1.045M pairs WITH the
# scale-glitch episodes, robust state stats). v11 was the scale-corruption-cleaned rebuild;
# the user wants v10 back. This matches the REAL transfer mid (cosmos_..._ret_v3_448_state,
# which also uses _DATASET_KW_V10 + noisy pairs.npz), removing the v10-vs-v11 confound that
# existed when transfer=v10 was compared against blank=v11.
# NOTE: do NOT use pairs_clean.npz / state57_stats_pctl.json — the "clean" variant was
# determined to be the wrong version (removed scale eps backed up at
# /mnt/ddn/dataset/egoverse_removed_scale_corrupt/, clean_manifest.json).
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_v10 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_v10", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V10, blank_retrieval=True)
# SAME-EMBODIMENT top-1 reference mid (v10): identical to ..._ret_v3_448_state — the "Ours"
# v10 mid — in data (egoverse_pairs_v10 pairs.npz + robust state stats), layout (full transfer,
# state_t=50), model override, schedule and batch. The ONLY change is WHICH demo is served:
# same_emb_top1=True pins the reference to the best-ranked match whose embodiment EQUALS the
# query's (aria->aria, scale->scale, MECKA->MECKA, mimicdroid->mimicdroid), instead of the
# default lock on the row's global top-1 embodiment (cross-embodiment for ~51% of v10 rows).
# Rows without any same-embodiment match among the K=20 ranked columns are dropped (0.2%).
# Isolates the cross-embodiment axis: does retrieval mid-training need demos from ANOTHER
# embodiment, or does an intra-embodiment neighbour teach the same demo-reading behaviour?
# batch_size is left at the factory default 1 and overridden to 2 by the yaml, exactly as the
# reference run did (32 ranks x 2 = global batch 64) — the two are directly comparable at the
# iter-8000 readout the ladder uses.
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_sameemb_top1_v10 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_sameemb_top1_v10", EgoVerseRetV3StateDataset,
    base_kw=_DATASET_KW_V10, same_emb_top1=True)

# RETRIEVAL-DROPOUT mid (2026-08-12): identical to ..._ret_v3_448_state (the "Ours" mid)
# except a fraction of TRAINING rows is served demo-free — i.e. (1-p) retrieval-conditioned
# + p BC, drawn deterministically per row (rng seed 20260812, so the p=0.05 set is a SUBSET
# of the p=0.2 set and the two runs are nested rather than independent draws).
# Dropped rows are genuinely SHORTER: the RET_VIDEO block is capped to blank_retrieval_slots
# (2 = the bv2 layout), not zero-padded to the full k=nf_m//4. state_t therefore varies per
# row; BucketedDistributedSampler.bucket_key separates the two shapes so a batch never mixes
# them, and no model_override is needed. batch_size stays 1 like the Ours mid so the runs are
# directly comparable. Eval never drops (build_sample passes aug=False).
#
# p=0.2 (retdrop20, 8000 iters, 2026-08-13) measured worse than the Ours mid on the +ract
# readout: L0/step0/action ratio=gapE/gapM went 0.51 -> 1.19 (B, the match/cross-embodiment
# cell, 0.636 -> 1.005), and 0/28 layers kept ratio<1. Without ract it was slightly BETTER
# (mean ratio 2.63 -> 2.18). p=0.05 tests whether a lighter dropout keeps the no-ract gain
# without destroying the cross-embodiment alignment that ract exposes.
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_retdrop20_v10 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_retdrop20_v10", EgoVerseRetV3StateDataset,
    base_kw=_DATASET_KW_V10, blank_retrieval_prob=0.2)
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_retdrop5_v10 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_retdrop5_v10", EgoVerseRetV3StateDataset,
    base_kw=_DATASET_KW_V10, blank_retrieval_prob=0.05)

cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv1_v10 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv1_v10", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V10, blank_retrieval=True, blank_video_slots=1,
    model_override=_blankret_capped_override(1))   # state_t=28, chunk_duration=109, n_cond=4
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_v10 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_v10", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V10, blank_retrieval=True, blank_video_slots=2,
    model_override=_blankret_capped_override(2))   # state_t=29, chunk_duration=113, n_cond=5
# BC-mid scored on REAL demos (2026-08-13). Same checkpoint/layout as
# ..._blankret_bv2_v10 (state_t=29, n_cond=5) but the RET_VIDEO + RET_ACTION slots carry the
# ACTUAL matched demo instead of zeros: cap_ret_video=2 keeps the block at the 2 latents the
# model was trained with and TEMPORALLY SUBSAMPLES the demo clip to 8 frames over the same
# 3 s window (not a truncation). Purpose: the A/B/C control — a model trained demo-blind,
# asked at test time whether a demo helps. Scoring the BC config itself is vacuous: with
# blank_retrieval=True the demo index has zero effect on the input, so A/B/C are identical
# by construction.
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_bv2_realdemo_eval = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_bv2_realdemo_eval",
    EgoVerseRetV3StateDataset, base_kw=_DATASET_KW_V10, cap_ret_video=2,
    model_override=_blankret_capped_override(2))   # state_t=29, chunk_duration=113, n_cond=5

cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv4_v10 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv4_v10", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V10, blank_retrieval=True, blank_video_slots=4,
    model_override=_blankret_capped_override(4))   # state_t=31, chunk_duration=121, n_cond=7
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv8_v10 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv8_v10", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V10, blank_retrieval=True, blank_video_slots=8,
    model_override=_blankret_capped_override(8))   # state_t=35, chunk_duration=137, n_cond=11

# COTRAIN blank bv2: SAME as ..._blankret_bv2_v10 but the manifest adds eva-robot + HE as
# queries (human+robot cotraining). Blank recipe identical (state_t=29, n_cond=5).
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_cotrain = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_cotrain", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V10_COTRAIN, blank_retrieval=True, blank_video_slots=2,
    model_override=_blankret_capped_override(2))   # state_t=29, chunk_duration=113, n_cond=5

# DELTA-target ICL mid (v10): REAL retrieval (full transfer layout, state_t=50) with a DELTA
# action57 GEN target AND a current-state proprio slot (slot 0) — like the post-train. This is
# the delta counterpart of the state transfer mid; unlike the base ret_v3_448 delta mid it KEEPS
# cur_state conditioning (delta_target on EgoVerseRetV3StateDataset -> proprio = single state57).
cosmos_predict2p5_2b_egoverse_ret_v3_448_delta_v10 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_delta_v10", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V10, delta_target=True)   # real retrieval, state_t=50

# DELTA-target blank bv2 (v10): SAME capped blank layout as state_blankret_bv2_v10
# (state_t=29, 2 blank ret_video latents) but the GEN target is DELTA action57 (delta_target=True,
# proprio = single current state57), matching the DELTA transfer mid (cosmos_..._ret_v3_448).
# Pairs with a delta ICL/BC post — the delta counterpart of the state blank bv2.
cosmos_predict2p5_2b_egoverse_ret_v3_448_delta_blankret_bv2_v10 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_delta_blankret_bv2_v10", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V10, blank_retrieval=True, blank_video_slots=2,
    delta_target=True, model_override=_blankret_capped_override(2))   # state_t=29, chunk 113, n_cond=5

# VIDEO-ONLY mid: demo conditioning = RET_VIDEO only, NO RET_ACTION slot (o=0).
# state_t 50->49, num_conditional_frames k+2 (17..25), chunk_duration 197->193.
# Tests the REVERSE transfer direction: mid learns demo-VIDEO reading (the expensive
# part); post-train then ADDS ret_action (1 token) — expected to transfer cleanly
# (small addition to an already video-rich attention context, no 8-token flooding).
_V2_MODEL_OVERRIDE_NORA = {
    **_V2_MODEL_OVERRIDE, "state_t": 49,
    "min_num_conditional_frames": 17, "max_num_conditional_frames": 25,
    "tokenizer": dict(chunk_duration=193),
}
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_nora = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_nora", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11, no_ret_action=True,
    model_override=_V2_MODEL_OVERRIDE_NORA)

# PURE video->video mid: drop BOTH ret_action (cond) AND action (gen) slots; the ONLY
# generated target is FUTURE_VIDEO (demo video + current image -> future video). No action
# in the EDM at all. state_t 50->48 (T=k+g+2), chunk_duration 197->189, num_cond k+2.
# The action policy is added back entirely in the action post-train (warm-started here).
_V2_MODEL_OVERRIDE_VIDONLY = {
    **_V2_MODEL_OVERRIDE, "state_t": 48,
    "min_num_conditional_frames": 17, "max_num_conditional_frames": 25,
    "tokenizer": dict(chunk_duration=189),
}
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_vidonly = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_vidonly", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11, video_only=True,
    model_override=_V2_MODEL_OVERRIDE_VIDONLY)

# MIRROR of vidonly: demo conditioning = ret_ACTION only (ret_video blanked), and the ONLY
# objective is ACTION (mask_loss_for_action_future_state_prediction=True -> no future_video
# loss). Full layout kept (same as post) for a clean warm-start. state_t=50 (unchanged).
# actonly = ret_action-only conditioning (ret_video blank) + NO future_video block
# (no_future -> action is the only generated slot = no video objective, and the future
# latents aren't even in the sequence, so it's fast). T = k+4: full k=23 -> state_t=27
# (chunk 105), num_conditional_frames k+3 (18..26). "future action" (the action chunk) IS
# generated; only the future VIDEO objective is gone.
_V2_MODEL_OVERRIDE_ACTONLY = {
    **_V2_MODEL_OVERRIDE, "state_t": 27,
    "min_num_conditional_frames": 18, "max_num_conditional_frames": 26,
    "tokenizer": dict(chunk_duration=105),
}
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_actonly = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_actonly", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11, blank_ret_video=True, no_future=True,
    model_override=_V2_MODEL_OVERRIDE_ACTONLY)

# Same actonly but the blank RET_VIDEO block is capped to 4 latents (vs full ~23).
# k=4, no future -> T = 4+4 = 8 (state_t=8, chunk 29, num_cond 7) -> very cheap mid.
_V2_MODEL_OVERRIDE_ACTONLY_BV4 = {
    **_V2_MODEL_OVERRIDE, "state_t": 8,
    "min_num_conditional_frames": 7, "max_num_conditional_frames": 7,
    "tokenizer": dict(chunk_duration=29),
}
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_actonly_bv4 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_actonly_bv4", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11, blank_ret_video=True, blank_video_slots=4,
    no_future=True, model_override=_V2_MODEL_OVERRIDE_ACTONLY_BV4)

# bv2: only 2 blank ret_video latents (minimum-ish to seed RET_VIDEO role + past-rope).
# k=2, no future -> T = 2+4 = 6 (state_t=6, chunk 21, num_cond 5) -> cheapest actonly mid.
_V2_MODEL_OVERRIDE_ACTONLY_BV2 = {
    **_V2_MODEL_OVERRIDE, "state_t": 6,
    "min_num_conditional_frames": 5, "max_num_conditional_frames": 5,
    "tokenizer": dict(chunk_duration=21),
}
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_actonly_bv2 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_actonly_bv2", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V11, blank_ret_video=True, blank_video_slots=2,
    no_future=True, model_override=_V2_MODEL_OVERRIDE_ACTONLY_BV2)

# v10 (noisy original) versions of vidonly / actonly_bv2 — identical flags+overrides, only
# base_kw swapped to _DATASET_KW_V10 (egoverse_pairs_v10 noisy pairs.npz + robust stats),
# to match the v10 real transfer mid (no v10-vs-v11 confound).
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v10_vidonly = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v10_vidonly", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V10, video_only=True,
    model_override=_V2_MODEL_OVERRIDE_VIDONLY)
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v10_actonly_bv2 = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v10_actonly_bv2", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_DATASET_KW_V10, blank_ret_video=True, blank_video_slots=2,
    no_future=True, model_override=_V2_MODEL_OVERRIDE_ACTONLY_BV2)

# ===================== v11 SHORT (action-only, NO video) =====================
# Compact "policy-head" ablation of the v11 pair: strip the retrieved-video block AND
# the generated future-video block, keeping only [ret_action | cur_image(x4->1 latent) |
# action] (+ cur_state proprio for the state target). state_t collapses 50 -> 3/4, so
# the run is far cheaper — it tests whether the action/state target trains at all
# without the video objective. EgoVerseRetShortDataset (see its module doc); model is
# the SAME RF backbone driven purely by the data_batch latent_idx fields. Still 3 s v11
# data, still bucket-sampled (action-chunk length varies by embodiment), demo cond only.
from cosmos_policy.datasets.egoverse_ret_short_dataset import EgoVerseRetShortDataset


def _SHORT_MODEL_OVERRIDE(state_t: int, n_cond: int) -> dict:
    return dict(
        state_t=state_t,
        min_num_conditional_frames=n_cond,
        max_num_conditional_frames=n_cond,
        conditional_frames_probs=None,          # per-sample num_conditional_frames key
        tokenizer=dict(chunk_duration=1 + (state_t - 1) * NUM_DUPLICATES),   # 1+4*(T-1)
        text_encoder_class="T5",
        net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
        use_action_projection=False,
        use_proprio_projection=False,
        action_dim=57,
        proprio_dim=57,
        chunk_size=92,                          # max native action-chunk length
        action_loss_multiplier=48,
        mask_loss_for_action_future_state_prediction=False,
        mask_value_prediction_loss_for_policy_prediction=False,
        mask_current_state_action_for_value_prediction=False,
        mask_future_state_for_qvalue_prediction=False,
    )


def _short_experiment(name: str, state_target: bool, with_proprio: bool,
                      batch_size: int = 4, base_kw: dict = _DATASET_KW_V11,
                      n_blank_video: int = 0, **ds_kw) -> LazyDict:
    state_t = (4 if with_proprio else 3) + n_blank_video
    n_cond = state_t - 1
    ds = L(EgoVerseRetShortDataset)(
        num_frames=92, state_target=state_target, with_proprio=with_proprio,
        n_blank_video=n_blank_video,
        **{**base_kw, "final_image_size": 448, **ds_kw})
    return LazyDict(
        dict(
            defaults=["/experiment/cosmos_predict2p5_2b_480p_libero", "_self_"],
            trainer=dict(max_iter=20_000, logging_iter=5),
            scheduler=_SCHEDULER,
            optimizer=_OPTIMIZER,
            model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
                config=dict(**_SHORT_MODEL_OVERRIDE(state_t, n_cond)),
            ),
            dataloader_train=L(DataLoader)(
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                dataset=ds,
                sampler=L(BucketedDistributedSampler)(
                    dataset=ds,
                    batch_size=batch_size,       # MUST equal dataloader batch_size
                    num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                    rank=L(parallel_state.get_data_parallel_rank)(),
                    shuffle=True,
                    seed=0,
                ),
                batch_size=batch_size,
                drop_last=True,
            ),
            job=dict(group="cosmos_v2_finetune", name=name),
            upload_reproducible_setup=False,
        )
    )


# DELTA action57 target, minimal 3-slot layout (no proprio).
cosmos_predict2p5_2b_egoverse_ret_short_v11 = _short_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_short_v11", state_target=False, with_proprio=False)
# ABSOLUTE state57 target, 4-slot layout (current-state proprio conditioning).
cosmos_predict2p5_2b_egoverse_ret_short_state_v11 = _short_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_short_state_v11", state_target=True, with_proprio=True)

# BLANK-VIDEO bridge: short STATE layout + 1 BLANK ret-video latent (role RET_VIDEO at a
# PAST time) in the conditioning prefix (state_t=5, n_cond=4). Tests whether giving the
# mid the *structure* of a video slot (no real pixels) restores warm-start transfer to the
# full-video post (which A==base proved a pure no-video mid cannot seed). See probe A.
cosmos_predict2p5_2b_egoverse_ret_short_state_v11_bv1 = _short_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_short_state_v11_bv1",
    state_target=True, with_proprio=True, n_blank_video=1)

# ===================== v11 PLAIN (non-retrieval BC) =====================
# Mid-training arm of the "plain mid -> retrieval post-train" ablation: drop the
# retrieval channel ENTIRELY, leaving [cur_state | cur_image | action]. Same v11
# data, same query distribution, same RF backbone (layout is data-driven) — the
# only difference vs _short_v11 is that the ret_action slot is gone.
#
# The post-train side needs NO new config: point GR1_BASE_CKPT at this run's
# converted .pt and launch the existing retrieval post-train, e.g.
#   GR1_BASE_CKPT=/mnt/ddn/tmp/egoverse_plain_state_v11_4node/.../model_000008000.pt \
#     ... experiment=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_v11
#
# Read the TRANSFER-GAP WARNING in datasets/egoverse_plain_dataset.py before
# reading the result: roles 1/2/5 and the past/future time axes are untrained
# here by construction, so post-train warm-start is expected to be weaker than
# the v2 contract's 0.19-0.21 first-iter action_l1.
from cosmos_policy.datasets.egoverse_plain_dataset import EgoVersePlainDataset


# Plain layout is [cur_state | cur_image | action | future(g)] -> T = g+3, so the
# buckets are human g=23 -> T=26 and MD g=15 -> T=18 (vs the transfer mid's 50/34,
# which also carry the k retrieved-video latents). num_conditional_frames is a
# CONSTANT 2 here (cur_state + cur_image) — it no longer scales with the demo length.
_PLAIN_MODEL_OVERRIDE = dict(
    state_t=26,                        # max bucket (human T=26); MD=18
    min_num_conditional_frames=2,
    max_num_conditional_frames=2,
    conditional_frames_probs={2: 1.0},
    tokenizer=dict(chunk_duration=1 + 4 * (26 - 1)),   # 101
    text_encoder_class="T5",
    net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
    use_action_projection=False,
    use_proprio_projection=False,
    action_dim=57,
    proprio_dim=57,                    # state57 tiled at latent 0
    chunk_size=92,                     # max native chunk (per-sample native length used)
    action_loss_multiplier=48,         # same as the transfer mid — keep the arms matched
    mask_loss_for_action_future_state_prediction=False,
    mask_value_prediction_loss_for_policy_prediction=False,
    mask_current_state_action_for_value_prediction=False,
    mask_future_state_for_qvalue_prediction=False,
)


def _plain_experiment(name: str, state_target: bool, batch_size: int = 2,
                      base_kw: dict = _DATASET_KW_V11, **ds_kw) -> LazyDict:
    cfg = _ret_448_experiment(name, EgoVersePlainDataset, batch_size=batch_size,
                              base_kw=base_kw, state_target=state_target, **ds_kw)
    # _ret_448_experiment supplies the shared v11 scaffolding but wires _V2_MODEL_OVERRIDE
    # (state_t=50, the transfer layout). Swap in the plain layout's smaller sequence.
    cfg["model"] = L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
        config=dict(**_PLAIN_MODEL_OVERRIDE),
    )
    return cfg


# DELTA action57 target — counterpart of cosmos_predict2p5_2b_egoverse_ret_v3_448_v11.
cosmos_predict2p5_2b_egoverse_plain_v11 = _plain_experiment(
    "cosmos_predict2p5_2b_egoverse_plain_v11", state_target=False)
# ABSOLUTE state57 target — counterpart of ..._egoverse_ret_v3_448_state_v11.
cosmos_predict2p5_2b_egoverse_plain_state_v11 = _plain_experiment(
    "cosmos_predict2p5_2b_egoverse_plain_state_v11", state_target=True)


# OFFLINE EVAL of a v3@448 checkpoint on the held-out EgoDex embodiment: same
# model/schedule as cosmos_predict2p5_2b_egoverse_ret_v3_448, dataset swapped to
# EgoDexRetV3Dataset (egodex_pairs_top3_3s.npz, full-frame 448, egodex-native
# action57/state57 stats, [egodex]->[human] text alias). batch_size=1.
from cosmos_policy.datasets.egodex_ret_v3_dataset import EgoDexRetV3Dataset

_EGODEX_V3_448_DS = L(EgoDexRetV3Dataset)(num_frames=92, final_image_size=448)

cosmos_predict2p5_2b_egodex_ret_v3_448_eval = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=20_000, logging_iter=5),
        scheduler=_SCHEDULER,
        optimizer=_OPTIMIZER,
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_EGODEX_V3_448_DS,
            sampler=L(BucketedDistributedSampler)(
                dataset=_EGODEX_V3_448_DS,
                batch_size=1,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=1,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egodex_ret_v3_448_eval"),
        upload_reproducible_setup=False,
    )
)

# STATE-CHUNK twin of the EgoDex v3@448 offline eval: same model/schedule, dataset
# swapped to EgoDexRetV3StateDataset ("actions"/"retrieved_actions" carry FUTURE
# state57 ABSOLUTE chunks, slot 0 the PAST state chunk; egodex-native ROBUST state
# stats). Pairs with the ..._egoverse_ret_v3_448_state checkpoints.
from cosmos_policy.datasets.egodex_ret_v3_state_dataset import EgoDexRetV3StateDataset

_EGODEX_V3_448_STATE_DS = L(EgoDexRetV3StateDataset)(num_frames=92, final_image_size=448)

cosmos_predict2p5_2b_egodex_ret_v3_448_state_eval = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=20_000, logging_iter=5),
        scheduler=_SCHEDULER,
        optimizer=_OPTIMIZER,
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_EGODEX_V3_448_STATE_DS,
            sampler=L(BucketedDistributedSampler)(
                dataset=_EGODEX_V3_448_STATE_DS,
                batch_size=1,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=1,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_egodex_ret_v3_448_state_eval"),
        upload_reproducible_setup=False,
    )
)

# UNCONDITIONAL ("blank-BC style") twin of the EgoDex offline eval above. Same EgoDex
# plumbing (egodex_pairs_top3_3s.npz, egodex-native robust state stats, full-frame 448,
# [egodex]->[human] alias) and the same query windows, but the demo is served the way the
# BC mid (..._state_blankret_bv2_v10) was TRAINED: the RET_VIDEO block is capped to 2
# latents and its CONTENT -- demo pixels AND the demo state chunk -- is ZEROED
# (blank_retrieval=True, blank_video_slots=2 -> state_t=29, chunk_duration=113, n_cond=5).
# Roles / tpos / has_ret stay intact, so the model still sees the retrieval STRUCTURE with
# zero retrieval INFORMATION.
#
# This is the unconditional column of the 2x2 EgoDex table
# (ICL mid vs BC mid) x (conditional vs unconditional). The conditional column is the
# existing ..._egodex_ret_v3_448_state_eval -- real demo, k = nf_m//4 = 23, state_t=50 --
# i.e. the way the ICL mid was trained. Each checkpoint is therefore in-distribution in
# exactly ONE cell and out-of-distribution in the other:
#
#                   conditional (k=23, real)      unconditional (k=2, blank)
#   ICL  8K         its own contract              OOD for ICL
#   BC   (bv2) 8K   OOD for BC                    its own contract
#
# Read the OFF-DIAGONAL cells with that in mind: an off-diagonal deficit can be the layout
# mismatch (role embeddings never given gradient at that block length) rather than an
# inability to use, or to ignore, a demo.
#
# Video metrics stay comparable across the two columns. The generated FUTURE_VIDEO block is
# g = nf_q // NUM_DUPLICATES latents, decided by the QUERY window (92 frames on every EgoDex
# row) and not by the demo, so both columns generate the same 92 future frames and their
# FID / SSIM / PSNR / LPIPS / FVD sit on the same scale. Only the demo block length moves.
#
# EgoDexRetV3StateDataset forwards **kw up the MRO to EgoVerseRetV3StateDataset, and the
# flags are honoured in build_sample(aug=False) -- the path the offline batch eval uses --
# so no dataset code changes for this. (aug=False also means retrieval DROPOUT never fires,
# which is why the conditional column stays fully conditioned.)
_EGODEX_V3_448_STATE_BV2BLANK_DS = L(EgoDexRetV3StateDataset)(
    num_frames=92, final_image_size=448, blank_retrieval=True, blank_video_slots=2)

cosmos_predict2p5_2b_egodex_ret_v3_448_state_bv2blank_eval = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=20_000, logging_iter=5),
        scheduler=_SCHEDULER,
        optimizer=_OPTIMIZER,
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            # state_t=29, min/max_num_conditional_frames=5, tokenizer.chunk_duration=113 --
            # the bv2 layout the BC mid trained in. Matches T = k + g + 2 + o + a
            # = 2 + 23 + 2 + 1 + 1 = 29 that the blank dataset emits for EgoDex.
            config=dict(**_blankret_capped_override(2)),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_EGODEX_V3_448_STATE_BV2BLANK_DS,
            sampler=L(BucketedDistributedSampler)(
                dataset=_EGODEX_V3_448_STATE_BV2BLANK_DS,
                batch_size=1,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=1,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_egodex_ret_v3_448_state_bv2blank_eval"),
        upload_reproducible_setup=False,
    )
)

# Humanoid-Everyday (HE) offline eval — held-out embodiment, same metrics as the
# EgoDex eval. Dataset = HeRetV3StateDataset (EgoVerse manifest format, emb 'he',
# HE-native robust state stats, [he]->[human] text alias). Pairs built by
# egoverse_pair_builder/168_he_match.py (HE->HE, exact task-name gate + DTW rerank,
# same-episode demos excluded); manifest by 169_he_manifest.py.
# Pairs with the ..._egoverse_ret_v3_448_state checkpoints.
from cosmos_policy.datasets.he_ret_v3_state_dataset import HeRetV3StateDataset

_HE_V3_448_STATE_DS = L(HeRetV3StateDataset)(num_frames=92, final_image_size=448)

cosmos_predict2p5_2b_he_ret_v3_448_state_eval = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=20_000, logging_iter=5),
        scheduler=_SCHEDULER,
        optimizer=_OPTIMIZER,
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_HE_V3_448_STATE_DS,
            sampler=L(BucketedDistributedSampler)(
                dataset=_HE_V3_448_STATE_DS,
                batch_size=1,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=1,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_he_ret_v3_448_state_eval"),
        upload_reproducible_setup=False,
    )
)

# HE UNCONDITIONAL (blank-demo, bv2) twin of ..._he_ret_v3_448_state_eval -- the HE copy of
# ..._egodex_ret_v3_448_state_bv2blank_eval, for the BC-mid (blankret_bv2_v10) "no
# conditioning" cell. blank_retrieval=True + blank_video_slots=2 give the bv2 layout the
# BC mid trained in (k=2 blank demo latents, state_t=29); the query window, generated
# FUTURE_VIDEO block (92 frames) and metrics are identical to the conditional HE eval, so
# FID / FVD / LPIPS and the action metrics sit on the same scale. The two flags reach
# EgoVerseRetV3StateDataset through HeRetV3StateDataset's **kw (not shadowed).
_HE_V3_448_STATE_BV2BLANK_DS = L(HeRetV3StateDataset)(
    num_frames=92, final_image_size=448, blank_retrieval=True, blank_video_slots=2)

cosmos_predict2p5_2b_he_ret_v3_448_state_bv2blank_eval = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=20_000, logging_iter=5),
        scheduler=_SCHEDULER,
        optimizer=_OPTIMIZER,
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_blankret_capped_override(2)),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_HE_V3_448_STATE_BV2BLANK_DS,
            sampler=L(BucketedDistributedSampler)(
                dataset=_HE_V3_448_STATE_BV2BLANK_DS,
                batch_size=1,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=1,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_he_ret_v3_448_state_bv2blank_eval"),
        upload_reproducible_setup=False,
    )
)

# ============================== RoboMIND offline eval ==============================
# Third held-out embodiment set (franka / ur / gello doing the SAME tasks), same metrics as
# the EgoDex and HE evals. Dataset = RoboMindRetV3StateDataset (see its module doc): the
# manifest is already in the EgoVerse schema (robomind_invariance/tools/rm_pairs.py from the
# DTW matches); only the frame/state accessors differ (h5+mp4 episodes, rm_state57*.pkl).
# frames="fixed" = 92 CONSECUTIVE frames from w0 -> a real 3.07 s clip at 30 Hz, matching the
# DT=4/30 the model is told. t5_mode="raw" = the episode's own sentence, REQUIRED for the v2
# manifest (its extra task classes have no t5_commands.json entry).
# Pairs with the ..._egoverse_ret_v3_448_state checkpoints, like EgoDex/HE.
from cosmos_policy.datasets.robomind_ret_v3_state_dataset import RoboMindRetV3StateDataset

_RM_V3_448_STATE_DS = L(RoboMindRetV3StateDataset)(num_frames=92, final_image_size=448)

cosmos_predict2p5_2b_robomind_ret_v3_448_state_eval = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=20_000, logging_iter=5),
        scheduler=_SCHEDULER,
        optimizer=_OPTIMIZER,
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_RM_V3_448_STATE_DS,
            sampler=L(BucketedDistributedSampler)(
                dataset=_RM_V3_448_STATE_DS,
                batch_size=1,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=1,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_robomind_ret_v3_448_state_eval"),
        upload_reproducible_setup=False,
    )
)

# RoboMIND UNCONDITIONAL (blank-demo bv2) twin — the BC / cotrain mids' own contract.
_RM_V3_448_STATE_BV2BLANK_DS = L(RoboMindRetV3StateDataset)(
    num_frames=92, final_image_size=448, blank_retrieval=True, blank_video_slots=2)

cosmos_predict2p5_2b_robomind_ret_v3_448_state_bv2blank_eval = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=20_000, logging_iter=5),
        scheduler=_SCHEDULER,
        optimizer=_OPTIMIZER,
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_blankret_capped_override(2)),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_RM_V3_448_STATE_BV2BLANK_DS,
            sampler=L(BucketedDistributedSampler)(
                dataset=_RM_V3_448_STATE_BV2BLANK_DS,
                batch_size=1,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=1,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_robomind_ret_v3_448_state_bv2blank_eval"),
        upload_reproducible_setup=False,
    )
)

# RoboMIND-ONLY mid, same recipe as `..._egoverse_ret_v3_448_state` (Ours mid): same model /
# layout / init / aug / batch (4 nodes x 8 x bs2 = 64 via the yaml), data = all 700 RoboMIND
# episodes (10 Hz -> 30 Hz grid, 47,608 3 s windows, RoboMIND per-emb state stats, raw T5).
# Pairs = manifest_win3s_v2 with the demo-emb lock reordered so ~53% of rows condition on a
# cross-emb demo (EgoVerse v10 train: 51.5%); see robomind_invariance/GEN_10HZ_FIX.md.
# Schedule scaled down with the data (Ours: 8K iters used of a 50K linear cycle, warmup 500):
#   max_iter 1K, cycle 6.25K (1K = 16% of the cycle, as 8K/50K), warmup 62, save every 500.
_RM_TRAIN_KW = dict(
    manifest_dir="/mnt/ddn/tmp/robomind_invariance/manifest_win3s_v2_trainx",
    grid_multicam=True,          # franka_3rgb -> 2x2 grid (top/left/right); UR / TienKung are 1rgb -> full frame
    normalize_images=False,
    use_image_aug=True,
    use_stronger_image_aug=True,
    retrieval_dropout_prob=0.0,
)
_RM_SCHEDULER = dict(
    cycle_lengths=[6_250, 100000000000000],
    warm_up_steps=[62, 0],
    f_start=[1e-6, 0.06],
    f_max=[1.0, 0.06],
    f_min=[0.3, 0.06],
)
cosmos_predict2p5_2b_robomind_ret_v3_448_state = _ret_448_experiment(
    "cosmos_predict2p5_2b_robomind_ret_v3_448_state", RoboMindRetV3StateDataset,
    base_kw=_RM_TRAIN_KW, max_iter=1_000, scheduler=_RM_SCHEDULER)
cosmos_predict2p5_2b_robomind_ret_v3_448_state["checkpoint"] = dict(save_iter=500)

# x3 data (RM_TAG _v3: ~75 episodes per (task, emb) cell, ~2,100 episodes, v2's franka 25 included),
# same recipe as above; schedule stretched to 1.5K iters with the same shape
# (1.5K = 16% of a 9,375 cycle, warmup 94 = 62 x 1.5), save every 500.
_RM_TRAIN_KW_V3 = dict(
    _RM_TRAIN_KW,
    manifest_dir="/mnt/ddn/tmp/robomind_invariance/manifest_win3s_v3_trainx",
    tag="_v3",
    t5_text_embeddings_path="/mnt/ddn/tmp/robomind_invariance/t5_embeddings_raw_v3.pkl",
)
_RM_SCHEDULER_V3 = dict(_RM_SCHEDULER, cycle_lengths=[9_375, 100000000000000], warm_up_steps=[94, 0])
cosmos_predict2p5_2b_robomind_ret_v3_448_state_x3 = _ret_448_experiment(
    "cosmos_predict2p5_2b_robomind_ret_v3_448_state_x3", RoboMindRetV3StateDataset,
    base_kw=_RM_TRAIN_KW_V3, max_iter=1_500, scheduler=_RM_SCHEDULER_V3)
cosmos_predict2p5_2b_robomind_ret_v3_448_state_x3["checkpoint"] = dict(save_iter=500)

# ================== DATA-SCALING sweep: query episodes x matched schedule ==================
# "If we had only collected X% of the query data, how much worse is the mid?" Each arm trains
# on an episode-level random subset of the v10 queries for a proportionally shorter schedule,
# so all four points sit at the SAME number of epochs and the only thing that moves is how
# much data exists.
#
# Which episodes each arm gets is DECIDED UP FRONT and pinned to disk by
# ``build_egoverse_query_episode_sets.py`` (a random, per-embodiment stratified draw over
# EPISODES, nested q25 < q50 < q75 < full); the dataset just filters to the listed episodes via
# ``query_episode_set``. The sets are readable JSON next to the manifest, so what an arm trained
# on stays auditable after the fact instead of having to be replayed from a seed. Episode-level
# is essential — v10 has a median of 74 ~1s-strided query rows per episode, so a random 25% of
# ROWS would still touch nearly all 13,315 episodes and the sweep would measure nothing but step
# count. Retrieval is untouched (match_idx still points at the full demo pool), so the retrieval
# corpus is held fixed and the axis is purely query-side data volume.
#
#   arm    query rows   episodes   max_iter   samples (gbatch 64)   epochs
#   q25       267,254      3,330      2,000       128,000            0.48
#   q50       519,158      6,659      4,000       256,000            0.49
#   q75       777,873      9,988      6,000       384,000            0.49
#   full    1,045,007     13,315      8,000       512,000            0.49
#
# The LR schedule scales with the RUN LENGTH, not the peak LR: cycle_lengths shrinks by the
# same factor as max_iter (50K -> 12.5K/25K/37.5K) while f_max stays 1.0 and optimizer.lr stays
# 1e-4. Scaling the peak instead would confound "less data" with "less LR" and make the small
# arms underfit for a reason unrelated to data. This lands every arm at the same point on its
# own LR curve when it stops (final multiplier 0.913 / 0.900 / 0.896 vs the reference's 0.894)
# rather than leaving the short arms cut off at a much higher LR. warm_up_steps stays 500 for
# all arms — scaling that too would change the early dynamics arm-to-arm.
#
# The 100% anchor is ALREADY ON DISK and must NOT be rerun (rerunning it under a different
# recipe would break the ladder):
#   /mnt/ddn/tmp/egoverse_ret_v3_448_state_4node/cosmos_policy/cosmos_v2_finetune/
#     cosmos_predict2p5_2b_egoverse_ret_v3_448_state/checkpoints/iter_000008000
# Launch each arm exactly like that reference did — ``dataloader_train.batch_size=2`` on
# 4 nodes x 8 GPUs (context_parallel_size=1 -> global batch 64). Changing the node count or
# batch size changes the epoch budget and invalidates the comparison.
_SCALING_REF_ITER = 8_000     # steps the reference v10 state mid actually trained for


def _scaled_scheduler(max_iter: int) -> dict:
    """_SCHEDULER with the decay horizon scaled to the run length; peak LR untouched."""
    head, tail = _SCHEDULER["cycle_lengths"]
    return {**_SCHEDULER, "cycle_lengths": [head * max_iter // _SCALING_REF_ITER, tail]}


def _query_scaling_arm(pct: int, max_iter: int) -> LazyDict:
    return _ret_448_experiment(
        f"cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q{pct}", EgoVerseRetV3StateDataset,
        max_iter=max_iter, scheduler=_scaled_scheduler(max_iter),
        query_episode_set=f"{_DATASET_KW_V10['manifest_dir']}/query_eps_q{pct}.json")


# q10 (added 2026-09-21): same rule, 10% of query episodes (1,332 ep / 105,118 rows, stratified, nested
# q10 < q25 — query_eps_q10.json from build_egoverse_query_episode_sets.py --fracs 0.10, same seed/perm).
# 800 iters x gbatch 64 = 51,200 samples = 0.487 epoch; cycle 5,000; warmup 500 kept like the other arms.
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q10 = _query_scaling_arm(10, 800)
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q25 = _query_scaling_arm(25, 2_000)
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q50 = _query_scaling_arm(50, 4_000)
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q75 = _query_scaling_arm(75, 6_000)


# ======================= random same-task reference control (randref) =======================
# Reviewer control "randomly selected same-task references under matched training settings".
# Exact clone of ..._ret_v3_448_state (the WATCH v10 state mid) except for three things:
#   1. pairs_name="pairs_randref.npz": every reference slot is redrawn uniformly from the windows
#      with the same (source, fine task class) as the original, i.e. only the DTW+handedness
#      motion ranking is removed (built by responses/y6hQ/scripts/build_randref_manifest.py).
#   2. MimicDroid (20 eps, 16,195 query rows = 1.55%) is dropped via query_eps_nomd.json: its
#      video is not in the EgoVerse R2 bucket, so a fresh download cannot train on it. No human
#      query references a MimicDroid window, so dropping it as a query removes it completely.
#   3. Launch: 2 nodes x 8 GPUs, batch_size=2 -> global batch 32 (the reference used 4 nodes, 64).
#      max_iter 8K -> 15K, and the LR decay horizon is stretched by the same factor
#      (_scaled_scheduler: cycle 50K -> 93,750) so the run stops at nearly the same point of its LR
#      curve as the 8K reference (multiplier 0.891 vs 0.894; the gap is the fixed 500-step warmup).
#      Peak LR 1e-4 and warmup 500 are unchanged.
#      Samples seen: 15K x 32 = 480K vs the reference's 8K x 64 = 512K (0.94x).
# All manifest files live in ONE directory: the v10-state manifest plus pairs_randref.npz and
# query_eps_nomd.json (from the HF repo Jeongeun/egoverse-v10-randref). Point EGOVERSE_RANDREF_MANIFEST
# at it; the video+state episode root is EGOVERSE_HUM_ROOT (see egoverse_ret_dataset.HUM_ROOT).
import os as _os

_RANDREF_MANIFEST = _os.environ.get("EGOVERSE_RANDREF_MANIFEST", "/mnt/ddn/dataset/egoverse_pairs_v10_randref")
_RANDREF_ITER = 15_000

cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k", EgoVerseRetV3StateDataset,
    batch_size=2,
    base_kw={
        **_DATASET_KW_V10,
        "manifest_dir": _RANDREF_MANIFEST,
        "t5_text_embeddings_path": f"{_RANDREF_MANIFEST}/t5_embeddings.pkl",
        "action_stats_path": f"{_RANDREF_MANIFEST}/action57_stats.json",
    },
    max_iter=_RANDREF_ITER, scheduler=_scaled_scheduler(_RANDREF_ITER),
    pairs_name="pairs_randref.npz",
    query_episode_set=f"{_RANDREF_MANIFEST}/query_eps_nomd.json")

# randref at q50 scale (optional, the cheaper version): query rows of query_eps_q50 minus MimicDroid
# (query_eps_q50_nomd.json, 511,746 rows), 8K iters x 32 = 256K samples = the q50 scaling arm's
# 4K x 64, decay horizon 50K (= the q50 arm's 25K x 64/32). Compare with ..._state_q50 (Fig. 9).
cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_q50_nomd_8k = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_q50_nomd_8k", EgoVerseRetV3StateDataset,
    batch_size=2,
    base_kw={
        **_DATASET_KW_V10,
        "manifest_dir": _RANDREF_MANIFEST,
        "t5_text_embeddings_path": f"{_RANDREF_MANIFEST}/t5_embeddings.pkl",
        "action_stats_path": f"{_RANDREF_MANIFEST}/action57_stats.json",
    },
    max_iter=8_000, scheduler=_scaled_scheduler(8_000),
    pairs_name="pairs_randref.npz",
    query_episode_set=f"{_RANDREF_MANIFEST}/query_eps_q50_nomd.json")


# ======================= noisy task labels (conf50) =======================
# Reviewer question: robustness of the pairing to noisy task labels. Exact clone of the randref
# launch recipe above (2 nodes x 8 GPUs, batch 2 -> global 32, _scaled_scheduler, MimicDroid
# dropped), with pairs_name="pairs_conf50_full.npz": WATCH's pairs.npz with the CROSS-source
# references edited under corrupted episode labels (50% of episodes relabelled to a sibling class
# of the same super-group; built by responses/yw5w/scripts/noisy_manifest.py, seed 0). Same-source
# references (T5-gated, label-free) are unchanged; with clean labels the builder reproduces
# pairs.npz exactly. Manifest dir = v10-state manifest + pairs_conf50_full.npz +
# query_eps_nomd.json + query_eps_q50_nomd.json (EGOVERSE_CONF50_MANIFEST).
#   full : 15K iters x 32 = 480K samples (vs WATCH 8K x 64 = 512K), compare with the WATCH v10 mid.
#   q50  : query rows of query_eps_q50 minus MimicDroid (511,746 rows); 8K x 32 = 256K samples =
#          the q50 scaling arm's 4K x 64, decay horizon 50K (= the q50 arm's 25K x 2), compare
#          with ..._state_q50 (Fig. 9, 50% point). Rows of the full file restricted to q50 are
#          identical to a q50-only build (labels are per episode, refs per query).
_CONF50_MANIFEST = _os.environ.get("EGOVERSE_CONF50_MANIFEST", "/mnt/ddn/dataset/egoverse_pairs_v10_conf50")
_CONF50_KW = {
    **_DATASET_KW_V10,
    "manifest_dir": _CONF50_MANIFEST,
    "t5_text_embeddings_path": f"{_CONF50_MANIFEST}/t5_embeddings.pkl",
    "action_stats_path": f"{_CONF50_MANIFEST}/action57_stats.json",
}
_CONF50_Q50_ITER = 8_000

cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_15k = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_15k", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_CONF50_KW,
    max_iter=_RANDREF_ITER, scheduler=_scaled_scheduler(_RANDREF_ITER),
    pairs_name="pairs_conf50_full.npz",
    query_episode_set=f"{_CONF50_MANIFEST}/query_eps_nomd.json")

cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_q50_nomd_8k = _ret_448_experiment(
    "cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_q50_nomd_8k", EgoVerseRetV3StateDataset,
    batch_size=2, base_kw=_CONF50_KW,
    max_iter=_CONF50_Q50_ITER, scheduler=_scaled_scheduler(_CONF50_Q50_ITER),
    pairs_name="pairs_conf50_full.npz",
    query_episode_set=f"{_CONF50_MANIFEST}/query_eps_q50_nomd.json")


ALL_EGOVERSE_RET_CONFIGS = [
    cosmos_predict2p5_2b_egoverse_ret_v2,
    cosmos_predict2p5_2b_egoverse_ret_v3,
    cosmos_predict2p5_2b_egoverse_ret_v2_448,
    cosmos_predict2p5_2b_egoverse_ret_v3_448,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_clean,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_v11,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_v11,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv1_v11,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_v11,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv4_v11,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv8_v11,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_sameemb_top1_v10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_retdrop20_v10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_retdrop5_v10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_bv2_realdemo_eval,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_v10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv1_v10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_v10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv4_v10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv8_v10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_cotrain,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_delta_v10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_delta_blankret_bv2_v10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_nora,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_vidonly,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_actonly,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_actonly_bv4,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v11_actonly_bv2,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v10_vidonly,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v10_actonly_bv2,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q10,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q25,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q50,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q75,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_q50_nomd_8k,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_15k,
    cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_q50_nomd_8k,
    cosmos_predict2p5_2b_egoverse_ret_short_v11,
    cosmos_predict2p5_2b_egoverse_ret_short_state_v11,
    cosmos_predict2p5_2b_egoverse_ret_short_state_v11_bv1,
    cosmos_predict2p5_2b_egoverse_plain_v11,
    cosmos_predict2p5_2b_egoverse_plain_state_v11,
    cosmos_predict2p5_2b_egoverse_ret_1s,
    cosmos_predict2p5_2b_egoverse_ret_5s,
    cosmos_predict2p5_2b_egoverse_ret_3s,
    cosmos_predict2p5_2b_egodex_ret_5s_eval,
    cosmos_predict2p5_2b_egodex_ret_3s_eval,
    cosmos_predict2p5_2b_egodex_ret_v3_448_eval,
    cosmos_predict2p5_2b_egodex_ret_v3_448_state_eval,
    cosmos_predict2p5_2b_egodex_ret_v3_448_state_bv2blank_eval,
    cosmos_predict2p5_2b_he_ret_v3_448_state_eval,
    cosmos_predict2p5_2b_he_ret_v3_448_state_bv2blank_eval,
    cosmos_predict2p5_2b_robomind_ret_v3_448_state_eval,
    cosmos_predict2p5_2b_robomind_ret_v3_448_state_bv2blank_eval,
    cosmos_predict2p5_2b_robomind_ret_v3_448_state,
    cosmos_predict2p5_2b_robomind_ret_v3_448_state_x3,
    cosmos_predict2p5_2b_egoverse_ret_1s__inference_only,
    cosmos_predict2p5_2b_egoverse_ret_5s__inference_only,
]
