"""Experiment configs for the real_robot bg2 bimanual retrieval-augmented
training stack (Predict2.5 / Rectified Flow only).

We ship three configs here:

  1. ``cosmos_predict2p5_2b_480p_realrobot_bg2``
       Base Predict2.5 single-camera setup that mirrors the existing
       Predict2 config ``cosmos_predict2_2b_480p_realrobot_bg2`` (state_t=6,
       action/proprio_dim=20). The RAG variant inherits from this for shared
       net + tokenizer + optimizer settings.

  2. ``cosmos_predict2p5_2b_480p_realrobot_bg2_ret_{top1,top3}{,_abs}``
       Retrieval-augmented training. Uses ``RealRobotRetDataset`` (consumes
       the pre-built ``pairs_top{1,3}.hdf5``) and the single-camera ret model
       class. Latent layout state_t=8 — see the model module docstring for
       details. Config layout (model override + explicit conditional_frames_probs)
       mirrors PushT pusht_ret_100 (residual) so the same training contract
       drives both setups.

  3. ``cosmos_predict2p5_2b_480p_realrobot_bg2_ret_*__inference_only``
       Inference shim (no training-only overrides) for each ret variant.
"""
from __future__ import annotations

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.datasets.real_robot_dataset import RealRobotDataset
from cosmos_policy.datasets.realrobot_ret_dataset import RealRobotRetDataset
from cosmos_policy.models.policy_video2world_model_realrobot_ret import (
    CosmosPolicyRealRobotRetModel,
    CosmosPolicyRealRobotRetModelRectifiedFlow,
)
from cosmos_policy.models.policy_video2world_model_rectified_flow import (
    CosmosPolicyVideo2WorldModelRectifiedFlow,
)


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

# Re-use the same training data the Predict2 base used; we only swap the
# backbone to 2.5 (Rectified Flow). Two tasks pooled together.
_REALROBOT_DATASET_KW = dict(
    dataset_root="/mnt/ddn/dataset/real_robot/h5",
    embodiment="bg2_bimanual",
    t5_text_embeddings_path="/mnt/ddn/dataset/real_robot/h5/t5_embeddings.pkl",
    chunk_size=16,
    final_image_size=224,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    predict_future_states=True,
    lazy_video_decompression=True,
    gamma=0.998,
)

realrobot_bg2_base_dataset = L(RealRobotDataset)(
    tasks=["place_bottle_in_plastic_box", "open_cabinet"],
    **_REALROBOT_DATASET_KW,
)

_RET_DATASET_KW = dict(
    t5_text_embeddings_path="/mnt/ddn/dataset/real_robot/h5/t5_embeddings.pkl",
    dataset_stats_path=(
        "/mnt/ddn/dataset/real_robot/h5/_cosmos_cache/"
        "bg2_bimanual__2tasks_e75b909e/dataset_statistics.json"
    ),
    delta_stats_dir="/mnt/ddn/dataset/real_robot/h5_retrieval",
    chunk_size=16,
    final_image_size=224,
    num_duplicates_per_image=4,
    normalize_proprio=True,
    normalize_actions=True,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_residual_actions=True,
    predict_future_states=True,
    retrieval_dropout_prob=0.0,
)

# Absolute baseline (no residual learning, all channels ABS direct)
_RET_DATASET_ABS_KW = {**_RET_DATASET_KW, "use_residual_actions": False}

# ret_image variants (state_t=8 layout): insert a ret_image latent slot at idx 1.
_RET_DATASET_IMG_KW     = {**_RET_DATASET_KW,     "use_ret_image": True}
_RET_DATASET_ABS_IMG_KW = {**_RET_DATASET_ABS_KW, "use_ret_image": True}

_PAIR_H5_PATHS = {
    5:  "/mnt/ddn/dataset/real_robot/h5_retrieval/pairs_top5_manifest.hdf5",
    10: "/mnt/ddn/dataset/real_robot/h5_retrieval/pairs_top10_manifest.hdf5",
    20: "/mnt/ddn/dataset/real_robot/h5_retrieval/pairs_top20_manifest.hdf5",
}

# 12 datasets — K ∈ {5, 10, 20} × img ∈ {off, on} × residual ∈ {True (mixed), False (abs)}.
realrobot_bg2_ret_top5_dataset          = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[5],  **_RET_DATASET_KW)
realrobot_bg2_ret_top5_abs_dataset      = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[5],  **_RET_DATASET_ABS_KW)
realrobot_bg2_ret_top5_img_dataset      = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[5],  **_RET_DATASET_IMG_KW)
realrobot_bg2_ret_top5_img_abs_dataset  = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[5],  **_RET_DATASET_ABS_IMG_KW)
realrobot_bg2_ret_top10_dataset         = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[10], **_RET_DATASET_KW)
realrobot_bg2_ret_top10_abs_dataset     = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[10], **_RET_DATASET_ABS_KW)
realrobot_bg2_ret_top10_img_dataset     = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[10], **_RET_DATASET_IMG_KW)
realrobot_bg2_ret_top10_img_abs_dataset = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[10], **_RET_DATASET_ABS_IMG_KW)
realrobot_bg2_ret_top20_dataset         = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[20], **_RET_DATASET_KW)
realrobot_bg2_ret_top20_abs_dataset     = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[20], **_RET_DATASET_ABS_KW)
realrobot_bg2_ret_top20_img_dataset     = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[20], **_RET_DATASET_IMG_KW)
realrobot_bg2_ret_top20_img_abs_dataset = L(RealRobotRetDataset)(pair_h5_path=_PAIR_H5_PATHS[20], **_RET_DATASET_ABS_IMG_KW)

# ── Single-task datasets (top10 with ret_image, mixed/abs × open_cabinet/place_bottle)
# Each uses its task-specific dataset_statistics + delta_stats, computed in the
# dataset's actual output dim (open_cabinet 10D R-only, place_bottle 20D full).
_PER_TASK_ROOT = "/mnt/ddn/dataset/real_robot/h5_retrieval/per_task"
_STATS_OC = f"{_PER_TASK_ROOT}/open_cabinet/dataset_statistics.json"
_STATS_PB = f"{_PER_TASK_ROOT}/place_bottle_in_plastic_box/dataset_statistics.json"
_DELTA_DIR_OC = f"{_PER_TASK_ROOT}/open_cabinet"
_DELTA_DIR_PB = f"{_PER_TASK_ROOT}/place_bottle_in_plastic_box"

# Drop shared dataset_stats_path / delta_stats_dir so we can override per-task.
_RET_DATASET_IMG_KW_NOSTATS     = {k: v for k, v in _RET_DATASET_IMG_KW.items()
                                    if k not in {"dataset_stats_path", "delta_stats_dir"}}
_RET_DATASET_ABS_IMG_KW_NOSTATS = {k: v for k, v in _RET_DATASET_ABS_IMG_KW.items()
                                    if k not in {"dataset_stats_path", "delta_stats_dir"}}

realrobot_bg2_ret_top10_img_open_cabinet_dataset = L(RealRobotRetDataset)(
    pair_h5_path=_PAIR_H5_PATHS[10],
    dataset_stats_path=_STATS_OC,
    delta_stats_dir=_DELTA_DIR_OC,
    tasks_filter=["open_cabinet"],
    r_arm_only=True,
    **_RET_DATASET_IMG_KW_NOSTATS,
)
realrobot_bg2_ret_top10_img_abs_open_cabinet_dataset = L(RealRobotRetDataset)(
    pair_h5_path=_PAIR_H5_PATHS[10],
    dataset_stats_path=_STATS_OC,
    delta_stats_dir=_DELTA_DIR_OC,
    tasks_filter=["open_cabinet"],
    r_arm_only=True,
    **_RET_DATASET_ABS_IMG_KW_NOSTATS,
)
realrobot_bg2_ret_top10_img_place_bottle_dataset = L(RealRobotRetDataset)(
    pair_h5_path=_PAIR_H5_PATHS[10],
    dataset_stats_path=_STATS_PB,
    delta_stats_dir=_DELTA_DIR_PB,
    tasks_filter=["place_bottle_in_plastic_box"],
    **_RET_DATASET_IMG_KW_NOSTATS,
)
realrobot_bg2_ret_top10_img_abs_place_bottle_dataset = L(RealRobotRetDataset)(
    pair_h5_path=_PAIR_H5_PATHS[10],
    dataset_stats_path=_STATS_PB,
    delta_stats_dir=_DELTA_DIR_PB,
    tasks_filter=["place_bottle_in_plastic_box"],
    **_RET_DATASET_ABS_IMG_KW_NOSTATS,
)

# OC abs with retrieval dropout — model learns cur_image fallback when ret is
# zeroed at training time. 0.1 ~ 0.5 prob.
def _drop_dataset(prob: float):
    kw = {**_RET_DATASET_ABS_IMG_KW_NOSTATS, "retrieval_dropout_prob": prob}
    return L(RealRobotRetDataset)(
        pair_h5_path=_PAIR_H5_PATHS[10],
        dataset_stats_path=_STATS_OC,
        delta_stats_dir=_DELTA_DIR_OC,
        tasks_filter=["open_cabinet"],
        r_arm_only=True,
        **kw,
    )
realrobot_bg2_ret_top10_img_abs_open_cabinet_drop01_dataset = _drop_dataset(0.1)
realrobot_bg2_ret_top10_img_abs_open_cabinet_drop02_dataset = _drop_dataset(0.2)
realrobot_bg2_ret_top10_img_abs_open_cabinet_drop03_dataset = _drop_dataset(0.3)
realrobot_bg2_ret_top10_img_abs_open_cabinet_drop04_dataset = _drop_dataset(0.4)
realrobot_bg2_ret_top10_img_abs_open_cabinet_drop05_dataset = _drop_dataset(0.5)


# ---------------------------------------------------------------------------
# Predict2.5 base (mirrors Predict2 cosmos_predict2_2b_480p_realrobot_bg2)
# ---------------------------------------------------------------------------

_REALROBOT_TRAINER = dict(max_iter=7_000, logging_iter=5)
_REALROBOT_SCHEDULER = dict(
    cycle_lengths=[7_000, 100000000000000],
    warm_up_steps=[500, 0],
    f_start=[1e-6, 0.06],
    f_max=[1.0, 0.06],
    f_min=[0.3, 0.06],
)

# Base layout (no retrieval):
#   0 blank | 1 proprio | 2 image | 3 action | 4 future_proprio | 5 future_image
#   state_t = 6  →  chunk_duration = 1 + 5 * 4 = 21
_REALROBOT_BASE_MODEL = dict(
    state_t=6,
    min_num_conditional_frames=3,
    max_num_conditional_frames=3,
    action_dim=20,
    proprio_dim=20,
    chunk_size=16,
    action_loss_multiplier=16,
    tokenizer=dict(chunk_duration=21),
    text_encoder_class="T5",
    # T5-XXL gives 1024-d crossattn input directly; matches PushT 2.5 setup.
    net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
)

cosmos_predict2p5_2b_480p_realrobot_bg2 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=_REALROBOT_TRAINER,
        scheduler=_REALROBOT_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(**_REALROBOT_BASE_MODEL),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=realrobot_bg2_base_dataset,
            sampler=L(DistributedSampler)(
                dataset=realrobot_bg2_base_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=8,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_realrobot_bg2",
        ),
        upload_reproducible_setup=False,
    )
)


# ---------------------------------------------------------------------------
# Predict2.5 + retrieval (the actual training target)
# ---------------------------------------------------------------------------

# Ret-aware layout (ret_image dropped — same-target abs RAG was failing to fit,
# diagnosing the ret_image latent as the disruptor since the ret_action signal
# alone preserves zero-shot generalization).
#   0 blank | 1 ret_action | 2 cur_image | 3 cur_proprio
#   | 4 pred_action | 5 future_image | 6 future_proprio
#   state_t = 7  →  chunk_duration = 1 + 6 * 4 = 25
#
# Conditioning: num_conditional_frames=4 covers slots 0-3 (blank + ret_action +
# cur_image + cur_proprio). Generation targets: 4 (pred_action), 5 (future_image),
# 6 (future_proprio).
_REALROBOT_RET_MODEL_OVERRIDE = dict(
    state_t=7,
    min_num_conditional_frames=4,
    max_num_conditional_frames=4,
    conditional_frames_probs={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0},
    tokenizer=dict(chunk_duration=25),                # 1 + 6 * num_duplicates_per_image(4)
    text_encoder_class="T5",
    net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
    # Match pusht_ret_100 projection setup: tiling (no learned projection)
    # for action/proprio latents, hidden_dim=256 carried for parity if a
    # future flip to projection=True is needed.
    use_action_projection=False,
    use_proprio_projection=False,
    action_dim=20,
    proprio_dim=20,
    projection_hidden_dim=256,
    # PushT pusht_ret_100 (Predict2.5 residual) uses multiplier=16. Mirror
    # that here — bumping the multiplier alone (24, 40) didn't move the raw
    # action MSE at fixed iter because Adam's second-moment normalization
    # cancels the gradient scaling. We dial the LR upward in
    # _REALROBOT_RET_OPTIMIZER below instead, since LR survives Adam's
    # normalization.
    action_loss_multiplier=16,
)

# RAG trains for 10K iters; the plain _REALROBOT_SCHEDULER cycles at 7K and
# would drop the LR factor to 0.06 from iter 7K onwards. Stretch the cycle so
# the LR stays alive through 10K (and well past, up to 20K) — at iter 10K the
# cosine schedule is ~0.66 of f_max, vs the 0.06 cliff with the plain config.
_REALROBOT_RET_SCHEDULER = dict(
    cycle_lengths=[20_000, 100000000000000],
    warm_up_steps=[500, 0],
    f_start=[1e-6, 0.06],
    f_max=[1.0, 0.06],
    f_min=[0.3, 0.06],
)

# LR dial history:
#   1e-4 (PushT mirror): baseline; train loss decreasing but R-arm plateau too high
#   2e-4 (2x): marginal improvement, plateau still too high
#   5e-5, 2e-5: dropping LR below the PushT default. The hypothesis is that
#     RAG attention mixing pumps batch-level noise into cur slot gradients,
#     so a higher LR oscillates around a poor basin instead of settling.
#     A smaller LR lets the optimizer integrate over more samples per step
#     and find a deeper minimum.
_REALROBOT_RET_OPTIMIZER = dict(lr=1e-4)


# Same model override but state_t=8 (ret_image slot at idx 1, all downstream
# indices shifted +1, num_conditional_frames=5 to cover blank+ret_image+
# ret_action+cur_image+cur_proprio).
_REALROBOT_RET_IMG_MODEL_OVERRIDE = dict(
    **{k: v for k, v in _REALROBOT_RET_MODEL_OVERRIDE.items()
       if k not in {"state_t", "min_num_conditional_frames",
                     "max_num_conditional_frames", "conditional_frames_probs",
                     "tokenizer"}},
    state_t=8,
    min_num_conditional_frames=5,
    max_num_conditional_frames=5,
    conditional_frames_probs={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0},
    tokenizer=dict(chunk_duration=29),  # 1 + 7 * num_duplicates_per_image(4)
)

# R-arm only variant of the img override: action_dim=10, proprio_dim=10.
# Used by single-task open_cabinet (L-arm fully idle → drop entirely).
_REALROBOT_RET_IMG_MODEL_OVERRIDE_R10 = {
    **_REALROBOT_RET_IMG_MODEL_OVERRIDE,
    "action_dim": 10,
    "proprio_dim": 10,
}


def _ret_experiment(ds_node, name: str, model_override: dict = None) -> LazyDict:
    """Build a Predict2.5 ret training experiment config."""
    override = model_override if model_override is not None else _REALROBOT_RET_MODEL_OVERRIDE
    return LazyDict(
        dict(
            defaults=[
                "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
                "_self_",
            ],
            trainer=dict(max_iter=10_000),
            scheduler=_REALROBOT_RET_SCHEDULER,
            optimizer=_REALROBOT_RET_OPTIMIZER,
            model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
                config=dict(**override),
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
                # Plain non-RAG runs batch=8 and fits fine; keep RAG at 8
                # too — batch=25 (PushT mirror) made each iter too slow on
                # our hardware without measurable fit improvement.
                batch_size=8,
                drop_last=True,
            ),
            job=dict(
                group="cosmos_v2_finetune",
                name=name,
            ),
            upload_reproducible_setup=False,
        )
    )


cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5 = _ret_experiment(
    realrobot_bg2_ret_top5_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_abs = _ret_experiment(
    realrobot_bg2_ret_top5_abs_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_abs",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img = _ret_experiment(
    realrobot_bg2_ret_top5_img_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img_abs = _ret_experiment(
    realrobot_bg2_ret_top5_img_abs_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img_abs",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10 = _ret_experiment(
    realrobot_bg2_ret_top10_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_abs = _ret_experiment(
    realrobot_bg2_ret_top10_abs_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_abs",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img = _ret_experiment(
    realrobot_bg2_ret_top10_img_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs = _ret_experiment(
    realrobot_bg2_ret_top10_img_abs_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20 = _ret_experiment(
    realrobot_bg2_ret_top20_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_abs = _ret_experiment(
    realrobot_bg2_ret_top20_abs_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_abs",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img = _ret_experiment(
    realrobot_bg2_ret_top20_img_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img_abs = _ret_experiment(
    realrobot_bg2_ret_top20_img_abs_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img_abs",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE,
)

# Single-task variants (top10 + ret_image, mixed/abs × {open_cabinet, place_bottle})
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet = _ret_experiment(
    realrobot_bg2_ret_top10_img_open_cabinet_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE_R10,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet = _ret_experiment(
    realrobot_bg2_ret_top10_img_abs_open_cabinet_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE_R10,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_place_bottle = _ret_experiment(
    realrobot_bg2_ret_top10_img_place_bottle_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_place_bottle",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_place_bottle = _ret_experiment(
    realrobot_bg2_ret_top10_img_abs_place_bottle_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_place_bottle",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE,
)
# OC abs + retrieval dropout variants
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop01 = _ret_experiment(
    realrobot_bg2_ret_top10_img_abs_open_cabinet_drop01_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop01",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE_R10,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop02 = _ret_experiment(
    realrobot_bg2_ret_top10_img_abs_open_cabinet_drop02_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop02",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE_R10,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop03 = _ret_experiment(
    realrobot_bg2_ret_top10_img_abs_open_cabinet_drop03_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop03",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE_R10,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop04 = _ret_experiment(
    realrobot_bg2_ret_top10_img_abs_open_cabinet_drop04_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop04",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE_R10,
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop05 = _ret_experiment(
    realrobot_bg2_ret_top10_img_abs_open_cabinet_drop05_dataset,
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop05",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE_R10,
)

# Back-compat alias: existing scripts/registrations still resolve.
cosmos_predict2p5_2b_480p_realrobot_bg2_ret = cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10


# ---------------------------------------------------------------------------
# Inference variants
# ---------------------------------------------------------------------------

cosmos_predict2p5_2b_480p_realrobot_bg2__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_realrobot_bg2__inference_only",
        ),
    )
)

def _inference_only(parent_exp: str, name: str) -> LazyDict:
    return LazyDict(dict(
        defaults=[f"/experiment/{parent_exp}", "_self_"],
        job=dict(group="cosmos_v2_inference", name=name),
    ))


cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_abs__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_abs",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_abs__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img_abs__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img_abs",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img_abs__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_abs__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_abs",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_abs__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_abs__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_abs",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_abs__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img_abs__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img_abs",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img_abs__inference_only",
)
# Single-task inference variants
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_place_bottle__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_place_bottle",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_place_bottle__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_place_bottle__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_place_bottle",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_place_bottle__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop01__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop01",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop01__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop02__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop02",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop02__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop03__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop03",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop03__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop04__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop04",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop04__inference_only",
)
cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop05__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop05",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop05__inference_only",
)
# Back-compat alias.
cosmos_predict2p5_2b_480p_realrobot_bg2_ret__inference_only = (
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10__inference_only
)


# ---------------------------------------------------------------------------
# Predict2 (EDM) + retrieval — OC single-task (R-arm only)
# Mirrors the Predict2.5 single-task OC ret configs but uses the Predict2 base
# (cosmos_predict2_2b_480p_realrobot_bg2) and CosmosPolicyRealRobotRetModel
# (EDM backbone) instead of the Rectified Flow class. Same dataset, same
# state_t=8 layout, same loss multiplier and LR. Lets us A/B the backbone at
# fixed retrieval setup.
# ---------------------------------------------------------------------------


def _ret_experiment_predict2(ds_node, name: str, model_override: dict) -> LazyDict:
    """Build a Predict2 (EDM) ret training experiment config."""
    return LazyDict(
        dict(
            defaults=[
                "/experiment/cosmos_predict2_2b_480p_realrobot_bg2",
                "_self_",
            ],
            trainer=dict(max_iter=10_000),
            scheduler=_REALROBOT_RET_SCHEDULER,
            optimizer=_REALROBOT_RET_OPTIMIZER,
            model=L(CosmosPolicyRealRobotRetModel)(
                config=dict(**model_override),
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
                batch_size=8,
                drop_last=True,
            ),
            job=dict(
                group="cosmos_v2_finetune",
                name=name,
            ),
            upload_reproducible_setup=False,
        )
    )


cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet = _ret_experiment_predict2(
    realrobot_bg2_ret_top10_img_open_cabinet_dataset,
    "cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE_R10,
)
cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet = _ret_experiment_predict2(
    realrobot_bg2_ret_top10_img_abs_open_cabinet_dataset,
    "cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet",
    model_override=_REALROBOT_RET_IMG_MODEL_OVERRIDE_R10,
)
cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet",
    "cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet__inference_only",
)
cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet",
    "cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet__inference_only",
)


# ---------------------------------------------------------------------------
# Predict2.5 RF — OC single-task R-arm only, NO retrieval baseline.
# 10-D action/proprio (R xyz(3) + R rot6d(6) + R grip(1)). Same backbone as
# the multi-task no-rag cosmos_predict2p5_2b_480p_realrobot_bg2 but with
# R-arm-only data slicing in RealRobotDataset.
# ---------------------------------------------------------------------------

realrobot_bg2_open_cabinet_r10_dataset = L(RealRobotDataset)(
    tasks=["open_cabinet"],
    r_arm_only=True,
    **_REALROBOT_DATASET_KW,
)

_REALROBOT_NORAG_MODEL_R10 = {
    **{k: v for k, v in _REALROBOT_BASE_MODEL.items()
       if k not in ("action_dim", "proprio_dim")},
    "action_dim": 10,
    "proprio_dim": 10,
}

cosmos_predict2p5_2b_480p_realrobot_bg2_open_cabinet_r10 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=_REALROBOT_TRAINER,
        scheduler=_REALROBOT_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(**_REALROBOT_NORAG_MODEL_R10),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=realrobot_bg2_open_cabinet_r10_dataset,
            sampler=L(DistributedSampler)(
                dataset=realrobot_bg2_open_cabinet_r10_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=8,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_realrobot_bg2_open_cabinet_r10",
        ),
        upload_reproducible_setup=False,
    )
)
cosmos_predict2p5_2b_480p_realrobot_bg2_open_cabinet_r10__inference_only = _inference_only(
    "cosmos_predict2p5_2b_480p_realrobot_bg2_open_cabinet_r10",
    "cosmos_predict2p5_2b_480p_realrobot_bg2_open_cabinet_r10__inference_only",
)


ALL_REALROBOT_RET_CONFIGS = [
    cosmos_predict2p5_2b_480p_realrobot_bg2,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_abs,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img_abs,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_abs,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_abs,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img_abs,
    cosmos_predict2p5_2b_480p_realrobot_bg2__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_abs__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top5_img_abs__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_abs__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_abs__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top20_img_abs__inference_only,
    # Single-task variants
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_place_bottle,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_place_bottle,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_place_bottle__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_place_bottle__inference_only,
    # OC abs + retrieval dropout variants
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop01,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop02,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop03,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop04,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop05,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop01__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop02__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop03__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop04__inference_only,
    cosmos_predict2p5_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet_drop05__inference_only,
    # Predict2 (EDM) OC single-task variants
    cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet,
    cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet,
    cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_open_cabinet__inference_only,
    cosmos_predict2_2b_480p_realrobot_bg2_ret_top10_img_abs_open_cabinet__inference_only,
    # Predict2.5 RF — OC R10 no-rag baseline
    cosmos_predict2p5_2b_480p_realrobot_bg2_open_cabinet_r10,
    cosmos_predict2p5_2b_480p_realrobot_bg2_open_cabinet_r10__inference_only,
]
