"""EgoVerse mimicdroid-GR1 multimodal retrieval training (Cosmos Predict-2.5).

Ports the Cosmos-3 GR1 posttrain DATA (egoverse_pairs_gr1: 3-cam, action57+state57,
demo↔query future-pairs) into the 2.5 image+state+action prediction framework
(`EgoVerseGr1RetDataset` + `CosmosPolicyRealRobotRetModelRectifiedFlow`).

Latent layout (state_t=36, chunk_duration=1+35*4=141):
  0 blank | 1-24 ret_video(front/sideleft/sideright, 32f each -> 8 latents) |
  25 ret_state | 26 ret_action | 27-29 cur_image(front/side/side) | 30 cur_state |
  -- 31 conditional latents --
  31 pred_action | 32-34 future_image(front/side/side, chunk-END) | 35 future_state.

CONDITION = retrieved demo video(3-cam)+state+action + current obs(3-cam)+state.
GENERATE  = action chunk(32) + chunk-END image(3-cam) + chunk-END state.
Per-frame t+k (stride 4) trains every timestep (see the dataset + docs §6b).
"""
from __future__ import annotations

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.datasets.egoverse_gr1_ret_dataset import EgoVerseGr1RetDataset
from cosmos_policy.models.policy_video2world_model_realrobot_ret import (
    CosmosPolicyRealRobotRetModelRectifiedFlow,
)

# v4 (2026-07-10): DTH demo pool (100 eps/task, non-training scenes, same-scene
# excluded) + grip-augmented 20-d matcher (S=0.4) + Cabinet central-weighted selection
# (L2 39/L7 10/L0 1). Server mirror: gr1_online_retrieval 20-d, dth_pool_1x_v4 +
# taskdemo_pool_v4. v3 and older manifests DELETED — this is the only valid one.
MANIFEST = "/mnt/ddn/dataset/egoverse_pairs_gr1_v4"
CHUNK = 32
# GENERATE = action + 3-cam future image ONLY (future_state + value prediction disabled).
# state_t = 1 blank + 3*(32/4)=24 ret video + 2 ret(state,action) + 3 cur img
#           + 1 cur state + 1 pred action + 3 future img = 35  (no future-state latent)
STATE_T = 35
CHUNK_DURATION = 1 + (STATE_T - 1) * 4       # 137
NUM_COND = 31                                 # latents 0..30 are conditioning

_GR1_RET_MODEL_OVERRIDE = dict(
    state_t=STATE_T,
    min_num_conditional_frames=NUM_COND,
    max_num_conditional_frames=NUM_COND,
    conditional_frames_probs={NUM_COND: 1.0},
    tokenizer=dict(chunk_duration=CHUNK_DURATION),
    text_encoder_class="T5",
    net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
    use_action_projection=False,
    use_proprio_projection=False,
    action_dim=57,
    proprio_dim=57,
    projection_hidden_dim=256,
    action_loss_multiplier=16,
    chunk_size=CHUNK,
)

_GR1_RET_DATASET = L(EgoVerseGr1RetDataset)(
    manifest_dir=MANIFEST,
    t5_text_embeddings_path=f"{MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST}/action57_stats.json",
    state_stats_path=f"{MANIFEST}/state57_stats.json",
    chunk_size=CHUNK,
    per_frame_stride=4,
    final_image_size=224,
    num_duplicates_per_image=4,
    use_residual_actions=False,
    retrieval_dropout_prob=0.0,
)


def _gr1_ret_experiment(ds_node, name: str, base_ckpt: str | None = None) -> LazyDict:
    # base_ckpt = warm-start from a trained EgoVerse-ret checkpoint (.pt). GR1 differs in
    # state_t(36 vs 50)/chunk(32 vs 92)/3-cam, so strict_resume=False skips the mismatched
    # heads and loads the shared backbone. None → train from the base Predict2.5-2B.
    ckpt = dict(load_path=base_ckpt, strict_resume=False) if base_ckpt else {}
    return LazyDict(
        dict(
            defaults=[
                "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
                "_self_",
            ],
            trainer=dict(max_iter=20_000),
            optimizer=dict(lr=1e-4),
            checkpoint=ckpt,
            model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
                config=dict(**_GR1_RET_MODEL_OVERRIDE),
            ),
            dataloader_train=L(DataLoader)(
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                dataset=ds_node,
                sampler=L(DistributedSampler)(
                    dataset=ds_node,
                    num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                    rank=L(parallel_state.get_data_parallel_rank)(),
                    shuffle=True,
                    seed=0,
                ),
                batch_size=4,             # state_t=36 is heavy; keep small (raise if it fits)
                drop_last=True,
            ),
            job=dict(group="cosmos_v2_finetune", name=name),
            upload_reproducible_setup=False,
        )
    )


cosmos_predict2p5_2b_egoverse_gr1_ret = _gr1_ret_experiment(
    _GR1_RET_DATASET, "cosmos_predict2p5_2b_egoverse_gr1_ret"
)

# ---------------------------------------------------------------------------
# MID-ALIGNED layout: identical latent layout to egoverse_ret 3s (state_t=50,
# R=23, ret_video 1-23, ret_action 24, cur_image 25, future video 26-48 GEN,
# action 49 GEN). 3 cams packed as a 2x2 grid in one 224 frame; 32f@20Hz
# upsampled to 92f. Model override copied VERBATIM from egoverse_ret
# _model_override(92) so the warm-start is a layout-exact continuation.
# See cosmos_policy/datasets/egoverse_gr1_ret_aligned_dataset.py.
# ---------------------------------------------------------------------------
from cosmos_policy.datasets.egoverse_gr1_ret_aligned_dataset import EgoVerseGr1RetAlignedDataset

_GR1_ALIGNED_MODEL_OVERRIDE = dict(
    state_t=50,
    min_num_conditional_frames=26,
    max_num_conditional_frames=26,
    conditional_frames_probs={26: 1.0},
    tokenizer=dict(chunk_duration=197),
    text_encoder_class="T5",
    net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
    use_action_projection=False,
    use_proprio_projection=False,
    action_dim=57,
    proprio_dim=57,                # cur_state tiled into slot 0 (blank sentinel)
    chunk_size=92,                 # 92-span chunk (native 32 zero-padded / upsampled) == video span
    action_loss_multiplier=48,     # mid 3s value
    mask_loss_for_action_future_state_prediction=False,
    mask_value_prediction_loss_for_policy_prediction=False,
    mask_current_state_action_for_value_prediction=False,
    mask_future_state_for_qvalue_prediction=False,
)

_GR1_ALIGNED_DATASET = L(EgoVerseGr1RetAlignedDataset)(
    manifest_dir=MANIFEST,
    t5_text_embeddings_path=f"{MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST}/action57_stats.json",
    state_stats_path=f"{MANIFEST}/state57_stats.json",
    chunk_size=32,                 # native GR1 steps (upsampled to 92 inside)
    per_frame_stride=4,
    final_image_size=224,
    num_duplicates_per_image=4,
    use_residual_actions=False,
    retrieval_dropout_prob=0.0,
)


def _gr1_aligned_experiment(name: str, base_ckpt: str | None = None) -> LazyDict:
    ckpt = dict(load_path=base_ckpt, strict_resume=False) if base_ckpt else {}
    return LazyDict(
        dict(
            defaults=[
                "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
                "_self_",
            ],
            trainer=dict(max_iter=20_000),
            optimizer=dict(lr=1e-4),
            checkpoint=ckpt,
            model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
                config=dict(**_GR1_ALIGNED_MODEL_OVERRIDE),
            ),
            dataloader_train=L(DataLoader)(
                num_workers=4,
                persistent_workers=True,
                pin_memory=True,
                dataset=_GR1_ALIGNED_DATASET,
                sampler=L(DistributedSampler)(
                    dataset=_GR1_ALIGNED_DATASET,
                    num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                    rank=L(parallel_state.get_data_parallel_rank)(),
                    shuffle=True,
                    seed=0,
                ),
                batch_size=4,
                drop_last=True,
            ),
            job=dict(group="cosmos_v2_finetune", name=name),
            upload_reproducible_setup=False,
        )
    )


# (instantiated below, after _CK3S is defined)

# GR1 post-training WARM-STARTED from the trained EgoVerse-ret checkpoints (iter 20000).
# load_path via env so the two share this file; strict_resume=False (see _gr1_ret_experiment).
_CK3S = "${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_3s_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_3s/checkpoints/model_000020000.pt}"
_CK5S = "${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_4node/cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_5s/checkpoints/model_000020000.pt}"
cosmos_predict2p5_2b_egoverse_gr1_ret_from3s = _gr1_ret_experiment(
    _GR1_RET_DATASET, "cosmos_predict2p5_2b_egoverse_gr1_ret_from3s", base_ckpt=_CK3S,
)
cosmos_predict2p5_2b_egoverse_gr1_ret_from5s = _gr1_ret_experiment(
    _GR1_RET_DATASET, "cosmos_predict2p5_2b_egoverse_gr1_ret_from5s", base_ckpt=_CK5S,
)
# MID-ALIGNED layout (state_t=50, action slot 49 — see block above), 3s warm-start.
# Default dataset layout="pad" (native 32f + zero-pad; no temporal resampling) + cur_state
# tiled into slot 0. The _up variant is the upsample fallback (smoke-compare on demand).
cosmos_predict2p5_2b_egoverse_gr1_ret_aligned_from3s = _gr1_aligned_experiment(
    "cosmos_predict2p5_2b_egoverse_gr1_ret_aligned_from3s", base_ckpt=_CK3S,
)
_ALIGNED_UP_DATASET = L(EgoVerseGr1RetAlignedDataset)(
    layout="upsample",
    manifest_dir=MANIFEST,
    t5_text_embeddings_path=f"{MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST}/action57_stats.json",
    state_stats_path=f"{MANIFEST}/state57_stats.json",
    chunk_size=32, per_frame_stride=4, final_image_size=224,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)
cosmos_predict2p5_2b_egoverse_gr1_ret_aligned_up_from3s = _gr1_aligned_experiment(
    "cosmos_predict2p5_2b_egoverse_gr1_ret_aligned_up_from3s", base_ckpt=_CK3S,
)
cosmos_predict2p5_2b_egoverse_gr1_ret_aligned_up_from3s["dataloader_train"]["dataset"] = _ALIGNED_UP_DATASET
cosmos_predict2p5_2b_egoverse_gr1_ret_aligned_up_from3s["dataloader_train"]["sampler"]["dataset"] = _ALIGNED_UP_DATASET

# ---------------------------------------------------------------------------
# V2 (CURRENT): GR1 post-train in the mid-v2 LENGTH-GENERAL layout (compact,
# k=g=8 native 32f@20Hz, slot roles + continuous-time rope; state_t=20,
# num_cond=11, action slot 11). Probe-verified zero transfer gap: mid-v2
# iter-1000 warm-start GR1 action_l1 0.330 == in-dist mid-MD 0.333.
# Warm-start from the mid-v2 checkpoint (convert DCP -> .pt first, like before).
# NOTE: the older from3s/from5s/aligned configs above are OBSOLETE (their base
# checkpoints were deleted 2026-07-07); kept only for config-registry stability.
# ---------------------------------------------------------------------------
from cosmos_policy.datasets.egoverse_gr1_ret_v2_dataset import EgoVerseGr1RetV2Dataset

_GR1_V2_MODEL_OVERRIDE = dict(
    state_t=20,
    min_num_conditional_frames=11,
    max_num_conditional_frames=11,
    conditional_frames_probs=None,        # per-sample batch key takes precedence
    tokenizer=dict(chunk_duration=77),
    text_encoder_class="T5",
    net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
    use_action_projection=False,
    use_proprio_projection=False,
    action_dim=57,
    proprio_dim=57,
    chunk_size=32,
    action_loss_multiplier=48,            # match mid-v2
    mask_loss_for_action_future_state_prediction=False,
    mask_value_prediction_loss_for_policy_prediction=False,
    mask_current_state_action_for_value_prediction=False,
    mask_future_state_for_qvalue_prediction=False,
)

_GR1_V2_DATASET = L(EgoVerseGr1RetV2Dataset)(
    manifest_dir=MANIFEST,
    t5_text_embeddings_path=f"{MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST}/action57_stats.json",
    state_stats_path=f"{MANIFEST}/state57_stats.json",
    chunk_size=32, per_frame_stride=4, final_image_size=224,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

_CKV2 = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_v2_4node/cosmos_policy/"
         "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v2/checkpoints/model_000020000.pt}")

cosmos_predict2p5_2b_egoverse_gr1_ret_v2 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=20_000),
        optimizer=dict(lr=1e-4),
        checkpoint=dict(load_path=_CKV2, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V2_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V2_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=8,             # state_t=20 is light (~2.5x smaller than 50)
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v2"),
        upload_reproducible_setup=False,
    )
)

# ---------------------------------------------------------------------------
# V3 @ 448: clone of the v2 experiment at final_image_size=448 — the 2x2 grid
# quadrant (S//2 in EgoVerseGr1RetV2Dataset._grid_frame) becomes a NATIVE 224
# decode per camera. Warm-started from the v3@448 mid run (full-frame primary
# cam at 448; GR1 post-train keeps the grid, mirroring the v2<-v2-mid pattern).
# Latent layout / roles / rope are UNCHANGED (only spatial size grows), so the
# same model override applies.
# ---------------------------------------------------------------------------
_GR1_V3_448_DATASET = L(EgoVerseGr1RetV2Dataset)(
    manifest_dir=MANIFEST,
    t5_text_embeddings_path=f"{MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST}/action57_stats.json",
    state_stats_path=f"{MANIFEST}/state57_stats.json",
    chunk_size=32, per_frame_stride=4, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

_CKV3_448 = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_4node/cosmos_policy/"
             "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448/checkpoints/model_000020000.pt}")

cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=20_000),
        optimizer=dict(lr=1e-4),
        checkpoint=dict(load_path=_CKV3_448, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V3_448_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V3_448_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            # 448 = 4x spatial tokens vs 224 (state_t=20 -> 20 latents x 56x56).
            # v2@224 ran batch=8; ~4x activation cost bounds this at ~2-4/GPU under
            # FSDP (mid@448 with state_t=50 needed batch=1). Set 2 CONSERVATIVELY;
            # raise to 4 only after a measured memory probe.
            batch_size=2,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448"),
        upload_reproducible_setup=False,
    )
)

# Ablation twin: SAME data/layout/448, but from the BASE Predict2.5-2B (no egoverse
# mid warm-start; empty checkpoint dict → the default experiment's pretrained load).
# Isolates what the v3@448 mid-training contributes to GR1 post-train.
cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448_base = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=20_000),
        optimizer=dict(lr=1e-4),
        checkpoint={},
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V3_448_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V3_448_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=2,             # same conservative 448 bound as the warm-start twin
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448_base"),
        upload_reproducible_setup=False,
    )
)

# ---------------------------------------------------------------------------
# STATE-CHUNK prediction variant @ 448: exact clone of the _v3_448 experiment,
# dataset swapped to EgoVerseGr1RetV2StateDataset — "actions"/"retrieved_actions"
# carry FUTURE state57 ABSOLUTE chunks s[w0+1..w0+32] (not action deltas) and
# slot 0 carries the PAST state chunk s[w0-31..w0] (tiled like the action chunk,
# left-padded with s[0] on early windows). Model override / layout / batch
# UNCHANGED (tensor keys and shapes are identical to the delta variant).
# Warm-start default = the FUTURE state-mid 20K checkpoint (same GR1_BASE_CKPT
# env pattern); the _base twin trains from the base Predict2.5-2B.
# ---------------------------------------------------------------------------
from cosmos_policy.datasets.egoverse_gr1_ret_v2_cmddelta_dataset import (
    EgoVerseGr1RetV2CmdDeltaDataset,
    EgoVerseGr1RetV2HybridDeltaDataset,
)
from cosmos_policy.datasets.egoverse_gr1_ret_v2_state_dataset import EgoVerseGr1RetV2StateDataset

_GR1_V3_448_STATE_DATASET = L(EgoVerseGr1RetV2StateDataset)(
    manifest_dir=MANIFEST,
    t5_text_embeddings_path=f"{MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST}/action57_stats.json",
    state_stats_path=f"{MANIFEST}/state57_stats.json",
    chunk_size=32, per_frame_stride=4, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

_CKV3_448_STATE = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_4node/cosmos_policy/"
                   "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state/checkpoints/model_000020000.pt}")

cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448_state = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=20_000),
        optimizer=dict(lr=1e-4),
        checkpoint=dict(load_path=_CKV3_448_STATE, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V3_448_STATE_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V3_448_STATE_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=2,             # same conservative 448 bound as the delta twin
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448_state"),
        upload_reproducible_setup=False,
    )
)

# Ablation twin: SAME state-chunk data/layout/448, from the BASE Predict2.5-2B
# (no mid warm-start; empty checkpoint dict -> the default experiment's pretrained load).
cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448_state_base = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=20_000),
        optimizer=dict(lr=1e-4),
        checkpoint={},
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V3_448_STATE_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V3_448_STATE_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=2,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448_state_base"),
        upload_reproducible_setup=False,
    )
)

# ---------------------------------------------------------------------------
# v5 post-train (2026-07-12): egoverse_pairs_gr1_v5 — regenerated GR1 episodes with
# the UNRECORDED approach bridge (demo-aligned t0, inactive arm parked like the
# source demo) + a 6th one-shot match slot (the episode's source_demo taskdemo,
# within-episode matched exactly like the eval server) so pinned one-shot eval is
# in-distribution. Warm-start default = the CLEAN state-mid 8K (pairs_clean +
# percentile normalization); override via GR1_BASE_CKPT.
# ---------------------------------------------------------------------------
MANIFEST_V5 = "/mnt/ddn/dataset/egoverse_pairs_gr1_v5"

_GR1_V5_448_STATE_DATASET = L(EgoVerseGr1RetV2StateDataset)(
    manifest_dir=MANIFEST_V5,
    t5_text_embeddings_path=f"{MANIFEST_V5}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V5}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V5}/state57_stats.json",
    chunk_size=32, per_frame_stride=4, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

_CKV5_448_STATE = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_clean_3node/cosmos_policy/"
                   "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_clean/checkpoints/model_000008000.pt}")

cosmos_predict2p5_2b_egoverse_gr1_ret_v5_448_state = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        # scheduler cycle == max_iter (8K): full cosine from f_max to f_min inside
        # the run — no LR cliff (the inherited 7K-cycle bug) and no flat tail.
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint=dict(load_path=_CKV5_448_STATE, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,          # 4 left the GPUs ~25% busy (448 JPEG decode bound)
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V5_448_STATE_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V5_448_STATE_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,   # precision-first setting (2026-07-13): b8 run underfit at 4K
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v5_448_state"),
        upload_reproducible_setup=False,
    )
)

# ---------------------------------------------------------------------------
# v6 post-train (2026-07-14): egoverse_pairs_gr1_v6 — one-shot-faithful data
# (tomato/corn-main, traj-proximity selection, GRASP_DWELL recipe, 4cm spawn
# law), EVERY-timestep pairs (no dataset jitter; 1 epoch = full coverage),
# t=0/approach FORCED taskdemo rows with negative-w0 frame0 padding, GROUND-
# TRUTH lockstep one-shot slot, and ABS-COMMAND targets (abs_action57, site
# frame — executing predictions reproduces the demo tracking dynamics).
# ---------------------------------------------------------------------------
# GR1 v6 manifest (HF Jeongeun/mimicdroid-gr1, manifest/); EGOVERSE_GR1_V6_MANIFEST overrides it.
MANIFEST_V6 = __import__("os").environ.get("EGOVERSE_GR1_V6_MANIFEST", "/mnt/ddn/dataset/egoverse_pairs_gr1_v6")

_GR1_V6_448_STATE_DATASET = L(EgoVerseGr1RetV2StateDataset)(
    manifest_dir=MANIFEST_V6,
    t5_text_embeddings_path=f"{MANIFEST_V6}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V6}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST_V6}/abs_action57_stats_pctl.json",
    robust_state_stats=False,   # percentile-hybrid 파일을 그대로 사용 (mean±4std 재계산 금지)
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

_CKV6_448_STATE = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_clean_3node/cosmos_policy/"
                   "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_clean/checkpoints/model_000008000.pt}")

cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint=dict(load_path=_CKV6_448_STATE, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_448_STATE_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_448_STATE_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state"),
        upload_reproducible_setup=False,
    )
)

# ---------------------------------------------------------------------------
# v6 ICL post-train FROM THE RANDREF MID (random same-task references), 2 nodes.
# Same data / layout / targets / loss as ..._v6_448_state; only the warm-start mid changes:
#   mid = cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k @ 15K (GR1_BASE_CKPT)
# The v6 reference was launched on 3 nodes x 8 GPUs x batch 4 = global 96 with an 8K schedule, and the
# ICL result is read at its 4K checkpoint (384K samples). On 2 nodes the global batch is 64, so the
# schedule is stretched by 96/64 (cycle 8K -> 12K) and the run stops at 6K = the reference's 4K point,
# on nearly the same point of the LR curve (multiplier 0.665 vs 0.673, not yet decayed). Peak LR 1e-4 and warmup 500
# are unchanged, as for the randref mid. For the full 8K-equivalent, pass trainer.max_iter=12000.
# Paths: EGOVERSE_GR1_V6_MANIFEST (manifest dir, e.g. <mimicdroid-gr1>/manifest) and
# EGOVERSE_GR1_HUM_ROOT (unpacked zarr episodes, egoverse_gr1_ret_dataset.HUM_ROOT).
# ---------------------------------------------------------------------------
import os as _os

_RANDREF_GR1_V6_MANIFEST = _os.environ.get("EGOVERSE_GR1_V6_MANIFEST", MANIFEST_V6)
_RANDREF_GR1_V6_CYCLE = 12_000  # reference 8K schedule x (96 / 64)
_RANDREF_GR1_V6_ITER = 6_000    # reference 4K checkpoint x (96 / 64)

_GR1_V6_448_STATE_RANDREF_DATASET = L(EgoVerseGr1RetV2StateDataset)(
    manifest_dir=_RANDREF_GR1_V6_MANIFEST,
    t5_text_embeddings_path=f"{_RANDREF_GR1_V6_MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{_RANDREF_GR1_V6_MANIFEST}/action57_stats.json",
    state_stats_path=f"{_RANDREF_GR1_V6_MANIFEST}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{_RANDREF_GR1_V6_MANIFEST}/abs_action57_stats_pctl.json",
    robust_state_stats=False,
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

_CK_RANDREF_MID = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_randref_2node/cosmos_policy/"
                   "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k/"
                   "checkpoints/model_000015000.pt}")

cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_randref_2node = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=_RANDREF_GR1_V6_ITER),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[_RANDREF_GR1_V6_CYCLE, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint=dict(load_path=_CK_RANDREF_MID, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_448_STATE_RANDREF_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_448_STATE_RANDREF_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_randref_2node"),
        upload_reproducible_setup=False,
    )
)

# v6 ICL post-train FROM THE NOISY-LABEL (conf50) MIDS and the randref q50 mid, 2 nodes: identical to ..._randref_2node above,
# only the warm-start mid and the job name change (GR1_BASE_CKPT overrides the default path).
#   _conf50_2node     <- cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_nomd_15k @ 15K
#   _conf50_q50_2node <- cosmos_predict2p5_2b_egoverse_ret_v3_448_state_conf50_q50_nomd_8k @ 8K
#   _randref_q50_2node <- cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_q50_nomd_8k @ 8K
import copy as _copy_conf50

for _tag, _mid, _it in (("conf50", "conf50_nomd_15k", 15_000), ("conf50_q50", "conf50_q50_nomd_8k", 8_000),
                        ("randref_q50", "randref_q50_nomd_8k", 8_000)):
    _n = f"cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_{_tag}_2node"
    _e = _copy_conf50.deepcopy(cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_randref_2node)
    _e["checkpoint"]["load_path"] = (
        f"${{oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_{_tag}_2node/cosmos_policy/"
        f"cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_{_mid}/checkpoints/model_{_it:09d}.pt}}")
    _e["job"]["name"] = _n
    globals()[_n] = _e

# 2x-POOL post-train (2026-07-29): IDENTICAL to ..._v6_448_state, but the retrieval
# demo pool is TEMPORALLY 2x-sped (dthdemo_*__2x / taskdemo_*__2x). Query/target are
# unchanged 1x GR1 (abs_action57). Matching descriptor = no-velocity pos+rot+grip
# (speed-invariant), re-matched over the 2x pool -> egoverse_pairs_gr1_v6_2x.
# Experiment: does a 2x-compressed conditioning demo still transfer (ICL) as well as 1x?
# GR1 v6 manifest with 2x-speed references; EGOVERSE_GR1_V6_2X_MANIFEST overrides it.
MANIFEST_V6_2X = __import__("os").environ.get("EGOVERSE_GR1_V6_2X_MANIFEST", "/mnt/ddn/dataset/egoverse_pairs_gr1_v6_2x")

_GR1_V6_2X_448_STATE_DATASET = L(EgoVerseGr1RetV2StateDataset)(
    manifest_dir=MANIFEST_V6_2X,
    t5_text_embeddings_path=f"{MANIFEST_V6_2X}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V6_2X}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6_2X}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST_V6_2X}/abs_action57_stats_pctl.json",
    robust_state_stats=False,
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint=dict(load_path=_CKV6_448_STATE, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_2X_448_STATE_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_2X_448_STATE_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state"),
        upload_reproducible_setup=False,
    )
)

# 2x-POOL FROM BASE (2026-07-31): same 2x data/layout as ..._v6_2x_448_state, but NO mid
# warm-start — trains from the HF base Predict2.5-2B (checkpoint={}). Baseline for the 2x run.
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state_base = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint={},   # NO mid warm-start: HF base Predict2.5-2B init
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_2X_448_STATE_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_2X_448_STATE_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state_base"),
        upload_reproducible_setup=False,
    )
)

# VIDEO-ONLY demo conditioning (2026-07-30): identical to ..._v6_448_state, but the
# retrieved demo contributes VIDEO ONLY — the RET_ACTION conditioning slot is DROPPED
# (dataset no_ret_action=True). The query still predicts action + future video. Ablation:
# is the demo's video enough, or does its action chunk help? state_t 20->19, num_cond
# 11->10, chunk_duration 77->73 (model override below MUST match the shrunken layout).
_GR1_V2_MODEL_OVERRIDE_VIDONLY = {
    **_GR1_V2_MODEL_OVERRIDE,
    "state_t": 19,
    "min_num_conditional_frames": 10,
    "max_num_conditional_frames": 10,
    "tokenizer": dict(chunk_duration=73),
}

_GR1_V6_448_STATE_VIDONLY_DATASET = L(EgoVerseGr1RetV2StateDataset)(
    manifest_dir=MANIFEST_V6,
    t5_text_embeddings_path=f"{MANIFEST_V6}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V6}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST_V6}/abs_action57_stats_pctl.json",
    robust_state_stats=False,
    no_ret_action=True,          # demo conditioning = VIDEO ONLY (drop RET_ACTION slot)
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_vidonly = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint=dict(load_path=_CKV6_448_STATE, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE_VIDONLY),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_448_STATE_VIDONLY_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_448_STATE_VIDONLY_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_vidonly"),
        upload_reproducible_setup=False,
    )
)

# VIDEO-ONLY FROM BASE (2026-07-31): same video-only (no_ret_action) layout as
# ..._v6_448_state_vidonly, but NO mid warm-start — trains from HF base Predict2.5-2B.
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_vidonly_base = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint={},   # NO mid warm-start: HF base Predict2.5-2B init
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE_VIDONLY),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_448_STATE_VIDONLY_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_448_STATE_VIDONLY_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_vidonly_base"),
        upload_reproducible_setup=False,
    )
)

# ============================================================================
# ACTION-ONLY post-train (2026-09-16) — ..._v6_448_state_actonly
# ----------------------------------------------------------------------------
# The 1x v6 ICL post-train (..._v6_448_state) conditions on the retrieved demo
# (ret_action + ret_video) and GENERATES (action + future video). This arm keeps the
# conditioning EXACTLY as-is -- ret_video stays, so in-context imitation is untouched --
# and removes ONLY the future-video generation target. Question: does the video
# objective actually buy anything for the ICL policy, or is it just paying for pixels?
#
# NOT the same as the SHORT arms (..._v6_448_short_state*), which also strip ret_video
# and therefore kill ICL itself; those confound "no video objective" with "no demo
# video". This arm separates the two.
#
# Implementation: the FUTURE_VIDEO latents are removed from the SEQUENCE, not merely
# masked out of the loss (EgoVerseGr1RetV2StateDataset(no_future=True); same meaning as
# EgoVerseRetV3StateDataset.no_future on the mid side and the Tabletop pv0_ap_* arms).
# The step gets cheaper as a side effect -- 8 fewer latents and no future clip decode.
#
#   0  CUR_STATE   1  RET_ACTION   2..9  RET_VIDEO(8)   10  CUR_IMAGE
#   -- num_conditional_frames = 11 (UNCHANGED) --
#   11 ACTION (GEN)   <- the only generated slot
#   state_t 20 -> 12,  chunk_duration 77 -> 45 (= 1 + 4*11)
#
# mask_loss_for_action_future_state_prediction stays FALSE, i.e. identical to
# _GR1_V2_MODEL_OVERRIDE. The Tabletop pv0_ap_* arms set it True, but here that would
# change the loss on the CONDITIONING slots relative to ..._v6_448_state and add a
# second changed variable; with T = num_conditional_frames + 1 the action slot is
# already the only generated one, so the flag buys nothing.
# action_loss_multiplier stays 48 (changing it would itself become the confound).
#
# Everything else -- data (MANIFEST_V6), abs_action57 target, pctl stats, 8K schedule
# with f_min=0.3, lr 1e-4, batch 4, warm-start ckpt -- is byte-identical to
# ..._v6_448_state, so the pair differs in exactly one thing: the future-video block.
_GR1_V2_MODEL_OVERRIDE_ACTONLY = {
    **_GR1_V2_MODEL_OVERRIDE,
    "state_t": 12,                        # 20 - 8 future latents
    "tokenizer": dict(chunk_duration=45),  # 1 + 4*11
    # min/max_num_conditional_frames stay 11 -- the conditioning prefix is unchanged.
}

_GR1_V6_448_STATE_ACTONLY_DATASET = L(EgoVerseGr1RetV2StateDataset)(
    manifest_dir=MANIFEST_V6,
    t5_text_embeddings_path=f"{MANIFEST_V6}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V6}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST_V6}/abs_action57_stats_pctl.json",
    robust_state_stats=False,
    no_future=True,              # DROP the FUTURE_VIDEO block (action is the only GEN slot)
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_actonly = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint=dict(load_path=_CKV6_448_STATE, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE_ACTONLY),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_448_STATE_ACTONLY_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_448_STATE_ACTONLY_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_actonly"),
        upload_reproducible_setup=False,
    )
)

# BASELINE (2026-07-16): NO mid-training — base Predict2.5-2B straight into the v6
# GR1 post-train recipe (same 8K schedule/data laws). Dataset = the RESTORED
# pre-v6.1 manifest (egoverse_pairs_gr1_v6_orig: v61 query rows filtered out of the
# rebuilt manifest; matches only ever referenced the unchanged dthdemo/taskdemo pool,
# so the filtered content equals the original v6 build exactly).
MANIFEST_V6_ORIG = "/mnt/ddn/dataset/egoverse_pairs_gr1_v6_orig"

_GR1_V6_ORIG_448_STATE_DATASET = L(EgoVerseGr1RetV2StateDataset)(
    manifest_dir=MANIFEST_V6_ORIG,
    use_center_top=False,   # 3-view revert (2026-07-20): drop low-info center_top
    t5_text_embeddings_path=f"{MANIFEST_V6_ORIG}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V6_ORIG}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6_ORIG}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST_V6_ORIG}/abs_action57_stats_pctl.json",
    robust_state_stats=False,   # percentile-hybrid 파일을 그대로 사용 (mean±4std 재계산 금지)
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

# DELTA-from-v11-mid (2026-07-18): post-train on the v6 recipe/data but with the
# DELTA action57 target (EgoVerseGr1RetV2Dataset), warm-started from the v11 mid
# (scale-cleaned DELTA mid, cosmos_predict2p5_2b_egoverse_ret_v3_448_v11 @ 8K).
# Stats files are the {emb:{lo,hi}} format the delta dataset expects (state norm
# uses the v6 pctl file). batch 2 (delta state_t=35 > state 20 — v3_448 precedent).
_GR1_V6_ORIG_448_DELTA_DATASET = L(EgoVerseGr1RetV2Dataset)(
    manifest_dir=MANIFEST_V6_ORIG,
    use_center_top=False,   # 3-view revert (2026-07-20): drop low-info center_top
    t5_text_embeddings_path=f"{MANIFEST_V6_ORIG}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V6_ORIG}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6_ORIG}/state57_stats_pctl.json",
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

_CKV11_MID = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_v11_4node/cosmos_policy/"
              "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_v11/checkpoints/model_000008000.pt}")

cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_v11 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.5],
            f_max=[1.0, 0.5],
            f_min=[0.5, 0.5],
        ),
        checkpoint=dict(load_path=_CKV11_MID, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_ORIG_448_DELTA_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_ORIG_448_DELTA_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=2,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_v11"),
        upload_reproducible_setup=False,
    )
)


# CMD-DELTA "cd" (2026-07-18, 확정 델타 정식화): target = COMMAND-space per-step
# deltas from abs_action57 (differentiate/integrate/execute in one space).
# Base = v11 clean-delta mid 8K; v6 recipe/data; stats = cmd_delta57_stats.json.
_GR1_V6_ORIG_448_CMDDELTA_DATASET = L(EgoVerseGr1RetV2CmdDeltaDataset)(
    manifest_dir=MANIFEST_V6_ORIG,
    use_center_top=False,   # 3-view revert (2026-07-20): drop low-info center_top
    t5_text_embeddings_path=f"{MANIFEST_V6_ORIG}/t5_embeddings.pkl",
    action_group="abs_action57",
    action_stats_path=f"{MANIFEST_V6_ORIG}/cmd_delta57_stats.json",
    state_stats_path=f"{MANIFEST_V6_ORIG}/state57_stats_pctl.json",
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cd = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.5],
            f_max=[1.0, 0.5],
            f_min=[0.5, 0.5],
        ),
        checkpoint=dict(load_path=_CKV11_MID, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_ORIG_448_CMDDELTA_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_ORIG_448_CMDDELTA_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=2,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cd"),
        upload_reproducible_setup=False,
    )
)

cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_base = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.5],
            f_max=[1.0, 0.5],
            f_min=[0.5, 0.5],
        ),
        checkpoint={},   # NO mid warm-start: HF base Predict2.5-2B init (same as *_base v3 configs)
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_ORIG_448_STATE_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_ORIG_448_STATE_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_base"),
        upload_reproducible_setup=False,
    )
)

# ---------------------------------------------------------------------------
# ROBUSTNESS: ret_action noise augmentation (2026-08-03). Train the demo-action
# prior to be noisy: per-sample Gaussian sigma ~ U(0, 0.2) on the normalized
# retrieved_actions (dataset ret_action_noise_std). Eval matches via the server
# --ret-action-noise {0, 0.1, 0.2}. Three warm-starts compared on IDENTICAL data
# (MANIFEST_V6): base = HF base (checkpoint={}); ret-mid = _CKV6_448_STATE default;
# BC-mid = GR1_BASE_CKPT overridden to the bv2 (blank-retrieval BC) 8k ckpt.
# _retnoise (env-switchable ckpt) serves ret-mid & BC-mid; _retnoise_base = base.
# Separate dataset instances per config (avoid shared config-node mutation).
_RETNOISE_STD = 0.2
_GR1_V6_448_STATE_RETNOISE_DATASET = L(EgoVerseGr1RetV2StateDataset)(
    manifest_dir=MANIFEST_V6,
    t5_text_embeddings_path=f"{MANIFEST_V6}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V6}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST_V6}/abs_action57_stats_pctl.json",
    robust_state_stats=False,
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
    ret_action_noise_std=_RETNOISE_STD,
)
_GR1_V6_448_STATE_RETNOISE_DATASET_B = L(EgoVerseGr1RetV2StateDataset)(
    manifest_dir=MANIFEST_V6,
    t5_text_embeddings_path=f"{MANIFEST_V6}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V6}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST_V6}/abs_action57_stats_pctl.json",
    robust_state_stats=False,
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
    ret_action_noise_std=_RETNOISE_STD,
)

# ret-mid / BC-mid: warm-start via _CKV6_448_STATE (GR1_BASE_CKPT env switch); mid scheduler
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_retnoise = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2", "_self_"],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(cycle_lengths=[8_000, 100000000000000], warm_up_steps=[500, 0],
                       f_start=[1e-6, 0.06], f_max=[1.0, 0.06], f_min=[0.3, 0.06]),
        checkpoint=dict(load_path=_CKV6_448_STATE, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(config=dict(**_GR1_V2_MODEL_OVERRIDE)),
        dataloader_train=L(DataLoader)(
            num_workers=8, prefetch_factor=4, persistent_workers=True, pin_memory=True,
            dataset=_GR1_V6_448_STATE_RETNOISE_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_448_STATE_RETNOISE_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0),
            batch_size=4, drop_last=True),
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_retnoise"),
        upload_reproducible_setup=False,
    )
)

# base: NO warm-start (HF base Predict2.5-2B); from-scratch scheduler (0.5 tail, as *_base)
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_retnoise_base = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2", "_self_"],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(cycle_lengths=[8_000, 100000000000000], warm_up_steps=[500, 0],
                       f_start=[1e-6, 0.5], f_max=[1.0, 0.5], f_min=[0.5, 0.5]),
        checkpoint={},   # NO warm-start: HF base init
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(config=dict(**_GR1_V2_MODEL_OVERRIDE)),
        dataloader_train=L(DataLoader)(
            num_workers=8, prefetch_factor=4, persistent_workers=True, pin_memory=True,
            dataset=_GR1_V6_448_STATE_RETNOISE_DATASET_B,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_448_STATE_RETNOISE_DATASET_B,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0),
            batch_size=4, drop_last=True),
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_retnoise_base"),
        upload_reproducible_setup=False,
    )
)

# 4-VIEW (2026-07-16): same v6 recipe/data(v6_orig), but the reader now fills the
# grid's BR quadrant with images.center_top (was black) — model input becomes
# front|sideleft|sideright|center_top. Requires zarrs backfilled with center_top
# (649/649 done 2026-07-16) and eval server --grid4.
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_4v = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.5],
            f_max=[1.0, 0.5],
            f_min=[0.5, 0.5],
        ),
        checkpoint=dict(load_path=_CKV6_448_STATE, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_ORIG_448_STATE_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_ORIG_448_STATE_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_4v"),
        upload_reproducible_setup=False,
    )
)

# CD-BASELINE (2026-07-19): cd 델타 post-train을 mid 없이 — base Predict2.5-2B에서
# 바로 (checkpoint={}). v11 delta mid의 기여를 cd 계열에서 분리하는 ablation.
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cd_base = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.5],
            f_max=[1.0, 0.5],
            f_min=[0.5, 0.5],
        ),
        checkpoint={},
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_ORIG_448_CMDDELTA_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_ORIG_448_CMDDELTA_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=2,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cd_base"),
        upload_reproducible_setup=False,
    )
)

# HYBRID-DELTA BASELINE "cdh_base" (2026-07-19): per-task delta target
# (Plate*=state-delta/측정리앵커, else cmd-delta), NO mid (checkpoint={}).
# 서버는 --pred-target cmd_delta57 --plate-state-delta 로 태스크별 디코드.
_GR1_V6_ORIG_448_HYBRID_DATASET = L(EgoVerseGr1RetV2HybridDeltaDataset)(
    manifest_dir=MANIFEST_V6_ORIG,
    use_center_top=False,   # 3-view revert (2026-07-20): drop low-info center_top
    t5_text_embeddings_path=f"{MANIFEST_V6_ORIG}/t5_embeddings.pkl",
    action_group="abs_action57",
    action_stats_path=f"{MANIFEST_V6_ORIG}/cmd_delta57_stats.json",   # unused (norm은 내부)
    cmd_delta_stats_path=f"{MANIFEST_V6_ORIG}/cmd_delta57_stats.json",
    state_delta_stats_path=f"{MANIFEST_V6_ORIG}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6_ORIG}/state57_stats_pctl.json",
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cdh_base = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.5],
            f_max=[1.0, 0.5],
            f_min=[0.5, 0.5],
        ),
        checkpoint={},
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_ORIG_448_HYBRID_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_ORIG_448_HYBRID_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=2,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cdh_base"),
        upload_reproducible_setup=False,
    )
)

# HYBRID-DELTA + MID ("cdh", 2026-07-19): cdh_base와 동일 타깃(Plate*=state-delta,
# else cmd-delta), v11 clean-delta mid 8K에서 warm-start — mid 기여를 하이브리드에서 측정.
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cdh = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.5],
            f_max=[1.0, 0.5],
            f_min=[0.5, 0.5],
        ),
        checkpoint=dict(load_path=_CKV11_MID, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_ORIG_448_HYBRID_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_ORIG_448_HYBRID_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=2,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cdh"),
        upload_reproducible_setup=False,
    )
)

# ==================== SHORT post-train (NO video; from short mids) ====================
# GR1 post-train counterparts of the "short" mid pair (cosmos_predict2p5_2b_egoverse_
# ret_short_v11 / _short_state_v11): video conditioning/generation stripped, state_t
# 20 -> 3/4. Datasets in egoverse_gr1_ret_short_dataset.py; same v6_orig manifest,
# targets identical to the full-model pair (state=abs_action57 pctl, short_cdh=hybrid).
from cosmos_policy.datasets.egoverse_gr1_ret_short_dataset import (
    EgoVerseGr1RetShortHybridDataset,
    EgoVerseGr1RetShortStateDataset,
)


def _GR1_SHORT_MODEL_OVERRIDE(state_t: int, n_cond: int) -> dict:
    return dict(
        state_t=state_t,
        min_num_conditional_frames=n_cond,
        max_num_conditional_frames=n_cond,
        conditional_frames_probs=None,
        tokenizer=dict(chunk_duration=1 + (state_t - 1) * 4),
        text_encoder_class="T5",
        net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
        use_action_projection=False,
        use_proprio_projection=False,
        action_dim=57,
        proprio_dim=57,
        chunk_size=32,
        action_loss_multiplier=48,
        mask_loss_for_action_future_state_prediction=False,
        mask_value_prediction_loss_for_policy_prediction=False,
        mask_current_state_action_for_value_prediction=False,
        mask_future_state_for_qvalue_prediction=False,
    )


_GR1_V6_ORIG_448_SHORT_STATE_DATASET = L(EgoVerseGr1RetShortStateDataset)(
    manifest_dir=MANIFEST_V6_ORIG,
    use_center_top=False,   # 3-view revert (2026-07-20): drop low-info center_top
    t5_text_embeddings_path=f"{MANIFEST_V6_ORIG}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V6_ORIG}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6_ORIG}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST_V6_ORIG}/abs_action57_stats_pctl.json",
    robust_state_stats=False,
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

_GR1_V6_ORIG_448_SHORT_HYBRID_DATASET = L(EgoVerseGr1RetShortHybridDataset)(
    manifest_dir=MANIFEST_V6_ORIG,
    use_center_top=False,   # 3-view revert (2026-07-20): drop low-info center_top
    t5_text_embeddings_path=f"{MANIFEST_V6_ORIG}/t5_embeddings.pkl",
    action_group="abs_action57",
    action_stats_path=f"{MANIFEST_V6_ORIG}/cmd_delta57_stats.json",   # unused (norm inside)
    cmd_delta_stats_path=f"{MANIFEST_V6_ORIG}/cmd_delta57_stats.json",
    state_delta_stats_path=f"{MANIFEST_V6_ORIG}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6_ORIG}/state57_stats_pctl.json",
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)

_CKSHORT_MID = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_short_v11_4node/cosmos_policy/"
                "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_short_v11/checkpoints/model_000008000.pt}")
_CKSHORT_STATE_MID = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_short_state_v11_4node/cosmos_policy/"
                      "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_short_state_v11/checkpoints/model_000008000.pt}")

# short STATE mid @ 10K (egoverse_ret_short_state_v11; NCCL-hang 재시작 후 완주, layout
# state_t=4 이므로 post-train과 정확히 일치 — proprio 슬롯 warm-start 유효).
_CKSHORT_STATE_MID_10K = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_short_state_v11_4node/cosmos_policy/"
                          "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_short_state_v11/checkpoints/model_000010000.pt}")


def _gr1_short_post(name: str, dataset, ckpt: str, state_t: int) -> LazyDict:
    return LazyDict(
        dict(
            defaults=[
                "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
                "_self_",
            ],
            trainer=dict(max_iter=8_000),
            optimizer=dict(lr=1e-4),
            scheduler=dict(
                cycle_lengths=[8_000, 100000000000000],
                warm_up_steps=[500, 0],
                f_start=[1e-6, 0.5],
                f_max=[1.0, 0.5],
                f_min=[0.5, 0.5],
            ),
            checkpoint=(dict(load_path=ckpt, strict_resume=False) if ckpt else {}),
            model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
                config=dict(**_GR1_SHORT_MODEL_OVERRIDE(state_t, state_t - 1)),
            ),
            dataloader_train=L(DataLoader)(
                num_workers=8,
                prefetch_factor=4,
                persistent_workers=True,
                pin_memory=True,
                dataset=dataset,
                sampler=L(DistributedSampler)(
                    dataset=dataset,
                    num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                    rank=L(parallel_state.get_data_parallel_rank)(),
                    shuffle=True,
                    seed=0,
                ),
                batch_size=4,
                drop_last=True,
            ),
            job=dict(group="cosmos_v2_finetune", name=name),
            upload_reproducible_setup=False,
        )
    )


# hybrid delta target, SHORT layout, NO mid (checkpoint={} -> HF base init):
# short_cdh의 no-mid 짝 — short 레이아웃에서 델타 mid 기여를 분리하는 ablation.
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_cdh_base = _gr1_short_post(
    "cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_cdh_base",
    _GR1_V6_ORIG_448_SHORT_HYBRID_DATASET, "", state_t=4)   # state_t=4: feeds current state (proprio)

# abs_action57 target, SHORT layout, NO mid (checkpoint={} -> HF base init):
# short_state의 no-mid 짝 (state mid 미기동이어도 독립 학습 가능).
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_base = _gr1_short_post(
    "cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_base",
    _GR1_V6_ORIG_448_SHORT_STATE_DATASET, "", state_t=4)


# ===== DELTA-mid -> ABS post-train (2026-07-20) =====
# v11 DELTA mid -> abs_action57 post-train. Two versions to compare the effect of
# the video objective given a delta pretrain:
#   * _state_dmid       : FULL v2 layout (VIDEO conditioning + future-video prediction),
#                         warm-started from the v11 FULL delta mid (_CKV11_MID).
#   * _short_state_dmid : SHORT layout (NO video), warm-started from the v11 SHORT
#                         delta mid (_CKSHORT_MID). Both predict abs_action57.
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_dmid = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        trainer=dict(max_iter=8_000),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            cycle_lengths=[8_000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        checkpoint=dict(load_path=_CKV11_MID, strict_resume=False),
        model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
            config=dict(**_GR1_V2_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            prefetch_factor=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=_GR1_V6_ORIG_448_STATE_DATASET,
            sampler=L(DistributedSampler)(
                dataset=_GR1_V6_ORIG_448_STATE_DATASET,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=2,
            drop_last=True,
        ),
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_dmid"),
        upload_reproducible_setup=False,
    )
)

# abs_action57 target, from the short STATE mid.
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state = _gr1_short_post(
    "cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state",
    _GR1_V6_ORIG_448_SHORT_STATE_DATASET, _CKSHORT_STATE_MID, state_t=4)

# abs_action57 SHORT post-train warm-started from the v11 SHORT DELTA mid (_CKSHORT_MID,
# egoverse_ret_short_v11 8K, state_t=3 no-proprio -> backbone transfers; post is state_t=4).
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_dmid = _gr1_short_post(
    "cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_dmid",
    _GR1_V6_ORIG_448_SHORT_STATE_DATASET, _CKSHORT_MID, state_t=4)

# abs_action57 SHORT post-train from the short STATE mid @ 10K (state_t=4 clean match).
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_smid = _gr1_short_post(
    "cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_smid",
    _GR1_V6_ORIG_448_SHORT_STATE_DATASET, _CKSHORT_STATE_MID_10K, state_t=4)
# hybrid delta target, from the short DELTA mid. NOTE (2026-07-20): dataset now
# feeds current state (state_t=4); the existing _CKSHORT_MID delta mid is state_t=3
# (no proprio) -> a state_t=4 delta mid must be trained before this warm-start is valid.
cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_cdh = _gr1_short_post(
    "cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_cdh",
    _GR1_V6_ORIG_448_SHORT_HYBRID_DATASET, _CKSHORT_MID, state_t=4)

# ---- SHORT post-train from the v10 448 STATE TRANSFER mid (2026-09-16) ----------------
# The existing SHORT arms warm-start from SHORT mids (or nothing). This one starts from
# the SAME mid the 1x v6 ICL post-train uses (..._egoverse_ret_v3_448_state @8K, "mid448
# v10"), so the SHORT family gets a third point on its own ladder:
#
#   short_state_base   no mid                    35.7% (132/370, 2026-07-21 full eval)
#   short_state_dmid   v11 SHORT delta mid @8K   36.2% (134/370, 2026-07-22 full eval)
#   short_state_tmid   v10 448 transfer mid @8K  <- this arm
#
# The mid is a FULL-layout (state_t=50) model and the post is SHORT (state_t=4); the
# backbone still transfers, exactly like short_state_smid (state_t=3 mid -> state_t=4
# post). Deliberately built on the EXISTING short contract -- MANIFEST_V6_ORIG, 3-view
# (use_center_top=False), f_min=0.5 -- so it is directly comparable to the two rows
# above. That is NOT the contract of ..._v6_448_state / ..._v6_448_state_actonly
# (MANIFEST_V6, 4-view, f_min=0.3): do not put this number in the same column as those
# without saying so.
_CKV10_448_STATE_MID = ("${oc.env:GR1_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_4node/cosmos_policy/"
                        "cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state/"
                        "checkpoints/model_000008000.pt}")

cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_tmid = _gr1_short_post(
    "cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_tmid",
    _GR1_V6_ORIG_448_SHORT_STATE_DATASET, _CKV10_448_STATE_MID, state_t=4)

# ===================== PLAIN post-train (non-retrieval BC), abs_action57 =====================
# The "post = plain BC" column of the mid/post factorial. Same v6 data, same
# abs_action57 target, same schedule and normalization as ..._gr1_ret_v6_448_state —
# the ONLY differences are the dataset (retrieval slots removed) and the resulting
# sequence length (state_t 20 -> 11, chunk_duration 77 -> 41, n_cond 11 -> 2).
#
#   ..._gr1_plain_state_v6        warm-started from a TRANSFER mid  -> the ⭐ arm:
#                                 "does the transfer pre-training benefit survive
#                                  deployment with NO demo?"
#   ..._gr1_plain_state_v6_base   no mid at all -> pure-BC floor.
#
# Run the retrieval baseline with the SAME GR1_BASE_CKPT so mid is held fixed and
# only the post-train layout varies:
#   GR1_BASE_CKPT=<mid.pt> ... experiment=cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state
#   GR1_BASE_CKPT=<mid.pt> ... experiment=cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6
#
# ⚠️ _CKV6_448_STATE's built-in default path (the state_clean mid) has NO checkpoint
#    on disk as of 2026-07-20 — neither a converted .pt nor a DCP dir. The only mid
#    currently materialized is the v11 DELTA mid at
#    /mnt/ddn/tmp/egoverse_ret_v3_448_v11_4node/.../model_000008000.pt
#    So GR1_BASE_CKPT MUST be set explicitly for the non-base arm; do not rely on
#    the default. (A delta-mid -> abs_action-post handoff is fine: actions are TILED,
#    not projected, and strict_resume=False — the _cd/_cdh configs already do exactly
#    this delta-mid -> cmd-delta-post crossing.)
from cosmos_policy.datasets.egoverse_gr1_plain_dataset import EgoVerseGr1PlainStateDataset

_GR1_PLAIN_MODEL_OVERRIDE = dict(
    **{**_GR1_V2_MODEL_OVERRIDE,
       "state_t": 11,                       # K+3 (retrieval layout: 20)
       "min_num_conditional_frames": 2,     # cur_state + cur_image
       "max_num_conditional_frames": 2,
       "conditional_frames_probs": {2: 1.0},
       "tokenizer": dict(chunk_duration=41),
       }
)

_GR1_V6_448_PLAIN_STATE_DATASET = L(EgoVerseGr1PlainStateDataset)(
    manifest_dir=MANIFEST_V6,
    t5_text_embeddings_path=f"{MANIFEST_V6}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST_V6}/action57_stats.json",
    state_stats_path=f"{MANIFEST_V6}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST_V6}/abs_action57_stats_pctl.json",
    robust_state_stats=False,   # percentile-hybrid 파일 그대로 (mean±4std 재계산 금지)
    chunk_size=32, per_frame_stride=1, final_image_size=448,
    num_duplicates_per_image=4, use_residual_actions=False, retrieval_dropout_prob=0.0,
)


def _gr1_plain_post(name: str, base_ckpt: str | None) -> LazyDict:
    return LazyDict(
        dict(
            defaults=[
                "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
                "_self_",
            ],
            trainer=dict(max_iter=8_000),
            optimizer=dict(lr=1e-4),
            scheduler=dict(
                cycle_lengths=[8_000, 100000000000000],
                warm_up_steps=[500, 0],
                f_start=[1e-6, 0.06],
                f_max=[1.0, 0.06],
                f_min=[0.3, 0.06],
            ),
            checkpoint=(dict(load_path=base_ckpt, strict_resume=False) if base_ckpt else {}),
            model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
                config=dict(**_GR1_PLAIN_MODEL_OVERRIDE),
            ),
            dataloader_train=L(DataLoader)(
                num_workers=8,
                prefetch_factor=4,
                persistent_workers=True,
                pin_memory=True,
                dataset=_GR1_V6_448_PLAIN_STATE_DATASET,
                sampler=L(DistributedSampler)(
                    dataset=_GR1_V6_448_PLAIN_STATE_DATASET,
                    num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                    rank=L(parallel_state.get_data_parallel_rank)(),
                    shuffle=True,
                    seed=0,
                ),
                batch_size=4,   # match the transfer post (gr1_ret_v6_448_state); plain is lighter
                drop_last=True,
            ),
            job=dict(group="cosmos_v2_finetune", name=name),
            upload_reproducible_setup=False,
        )
    )


# ⭐ mid=transfer x post=plain
cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6 = _gr1_plain_post(
    "cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6", _CKV6_448_STATE)
# floor: no mid at all
cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6_base = _gr1_plain_post(
    "cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6_base", None)


ALL_EGOVERSE_GR1_RET_CONFIGS = [
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_actonly,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_tmid,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_smid,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_dmid,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_dmid,
    cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6,
    cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6_base,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state_base,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_cdh_base,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_state,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_short_cdh,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cdh,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cdh_base,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cd_base,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_4v,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_cd,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_v11,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_base,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_randref_2node,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_conf50_2node,  # noqa: F821 (defined via globals())
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_conf50_q50_2node,  # noqa: F821
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_randref_q50_2node,  # noqa: F821
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_2x_448_state_base,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_vidonly,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_vidonly_base,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_retnoise,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state_retnoise_base,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v5_448_state,
    cosmos_predict2p5_2b_egoverse_gr1_ret,
    cosmos_predict2p5_2b_egoverse_gr1_ret_from3s,
    cosmos_predict2p5_2b_egoverse_gr1_ret_from5s,
    cosmos_predict2p5_2b_egoverse_gr1_ret_aligned_from3s,
    cosmos_predict2p5_2b_egoverse_gr1_ret_aligned_up_from3s,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v2,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448_base,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448_state,
    cosmos_predict2p5_2b_egoverse_gr1_ret_v3_448_state_base,
]
