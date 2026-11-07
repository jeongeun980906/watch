# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""Experiment configs for RoboCasa open-door retrieval-augmented training.

Builds on ``cosmos_predict2p5_2b_480p_robocasa_open_door`` (the base
open-door experiment). Queries come from the RoboCasa success_only
split; the retrieval pool comes from the MG_3000 HDF5 dataset. Pair
files are the ``retrieval_results_robocasa_mg_task_<idx>.npz`` produced
by ``robocasa_retrieval/build_retrieval_index.py``.

Latent layout (``state_t=15``, ``chunk_duration=57``):
    see ``policy_video2world_model_robocasa_ret.py`` docstring.
"""

import os

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.datasets.robocasa_doors_ret_dataset import (
    DEFAULT_OPEN_DOOR_FOLDERS,
    RoboCasaDoorsRetDataset,
)
from cosmos_policy.models.policy_video2world_model_robocasa_ret import (
    CosmosPolicyRoboCasaRetModelRectifiedFlow,
)

BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")


# ---------------------------------------------------------------------------
# Dataset configs
# ---------------------------------------------------------------------------

robocasa_open_door_ret_dataset = L(RoboCasaDoorsRetDataset)(
    query_data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),
    pool_data_dir=os.path.join(BASE_DATASETS_DIR, "robocasa-MG_3000_hdf5"),
    pair_glob=os.path.join(
        BASE_DATASETS_DIR,
        "robocasa-retrieval",
        "kitchen_doors",
        "*",
        "retrieval_results_robocasa_mg_task_*.npz",
    ),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    task_split=list(DEFAULT_OPEN_DOOR_FOLDERS),  # OpenSingleDoor + OpenDoubleDoor
    chunk_size=32,
    use_image_aug=True,
    use_stronger_image_aug=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=False,
)

# Residual-action variant: identical except it returns (gt - ret) as the
# target action chunk. The model predicts the delta in the already-normalized
# action space.
robocasa_open_door_ret_dataset_residual = L(RoboCasaDoorsRetDataset)(
    query_data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),
    pool_data_dir=os.path.join(BASE_DATASETS_DIR, "robocasa-MG_3000_hdf5"),
    pair_glob=os.path.join(
        BASE_DATASETS_DIR,
        "robocasa-retrieval",
        "kitchen_doors",
        "*",
        "retrieval_results_robocasa_mg_task_*.npz",
    ),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    task_split=list(DEFAULT_OPEN_DOOR_FOLDERS),
    chunk_size=32,
    use_image_aug=True,
    use_stronger_image_aug=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=True,
)


# ---------------------------------------------------------------------------
# Shared model config pieces
# ---------------------------------------------------------------------------

_MODEL_OVERRIDE = dict(
    # Latent layout (see module docstring):
    #   0=blank, 1-3=ret images, 4=ret_proprio, 5=ret_action,
    #   6-8=cur images, 9=cur_proprio, 10=pred_action,
    #   11-13=future images, 14=future_proprio
    state_t=15,
    min_num_conditional_frames=10,     # latents 0-9 conditioned (blank + 9 cond)
    max_num_conditional_frames=10,
    conditional_frames_probs={10: 1.0},  # always condition on 10 frames
    tokenizer=dict(
        chunk_duration=57,             # 1 + 14 * 4
    ),
)


# ---------------------------------------------------------------------------
# Training experiments
# ---------------------------------------------------------------------------

cosmos_predict2p5_2b_480p_robocasa_open_door_ret = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_open_door",
            "_self_",
        ],
        trainer=dict(max_iter=10_000),
        model=L(CosmosPolicyRoboCasaRetModelRectifiedFlow)(
            config=dict(**_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_open_door_ret_dataset,
            sampler=L(DistributedSampler)(
                dataset=robocasa_open_door_ret_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_robocasa_open_door_ret",
        ),
    )
)

# Residual-action variant (model predicts gt - ret_action delta)
cosmos_predict2p5_2b_480p_robocasa_open_door_ret_residual = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_open_door",
            "_self_",
        ],
        trainer=dict(max_iter=10_000),
        model=L(CosmosPolicyRoboCasaRetModelRectifiedFlow)(
            config=dict(**_MODEL_OVERRIDE),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_open_door_ret_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robocasa_open_door_ret_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_robocasa_open_door_ret_residual",
        ),
    )
)


# ---------------------------------------------------------------------------
# Inference experiments
# ---------------------------------------------------------------------------

cosmos_predict2p5_2b_480p_robocasa_open_door_ret__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_open_door_ret",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_open_door_ret__inference",
        ),
    )
)

cosmos_predict2p5_2b_480p_robocasa_open_door_ret_residual__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_open_door_ret_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_open_door_ret_residual__inference",
        ),
    )
)


# Exported so cosmos_policy_experiment_configs.py can register these in bulk.
ALL_ROBOCASA_DOORS_RET_CONFIGS = [
    cosmos_predict2p5_2b_480p_robocasa_open_door_ret,
    cosmos_predict2p5_2b_480p_robocasa_open_door_ret_residual,
    cosmos_predict2p5_2b_480p_robocasa_open_door_ret__inference,
    cosmos_predict2p5_2b_480p_robocasa_open_door_ret_residual__inference,
]
