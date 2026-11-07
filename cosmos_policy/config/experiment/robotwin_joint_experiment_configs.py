"""Experiment configs for RoboTwin 2.0 — *joint-space* (qpos) action variant.

Mirrors ``robotwin_experiment_configs.py`` 1:1 but swaps:
  - dataset class:   RoboTwinDataset  -> RoboTwinJointDataset
  - action_dim:      16 (EE)          -> 14 (qpos)
  - proprio_dim:     16               -> 14
  - experiment name: ...               + "__qpos" suffix
  - cache dir:       <ee_cache>__joint (handled inside RoboTwinJointDataset)

Existing EE-side files are untouched. Configs self-register at import time —
``cosmos_policy/config/config_v2.py`` already calls
``import_all_modules_from_package("cosmos_policy.config.experiment", reload=True)``
before any ``register_configs()`` runs, so dropping this file in here is enough.
"""

from __future__ import annotations

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.datasets.robotwin_joint_dataset import RoboTwinJointDataset
from cosmos_policy.datasets.robotwin_joint_rag_dataset import RoboTwinJointRetDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel

BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")
ROBOTWIN_DATASET_ROOT = os.path.join(BASE_DATASETS_DIR, "RoboTwin2.0", "dataset")
ROBOTWIN_EMBODIMENT = os.environ.get("ROBOTWIN_EMBODIMENT", "aloha-agilex_clean_50")
ROBOTWIN_T5_PATH = os.path.join(ROBOTWIN_DATASET_ROOT, "t5_embeddings.pkl")
ROBOTWIN_RAG_NPZ_DIR = "/mnt/ddn/robotwin_rag_tools/retrieval_results"

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "robotwin_splits")
TRAIN_3TASKS_SPLIT = os.path.join(SPLITS_DIR, "train_3tasks.txt")
DEBUG_SINGLE_TASK_SPLIT = os.path.join(SPLITS_DIR, "debug_place_cans_plasticbox.txt")
DEBUG_BURGER_FRIES_SPLIT = os.path.join(SPLITS_DIR, "debug_place_burger_fries.txt")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

robotwin_qpos_train_3tasks_dataset = L(RoboTwinJointDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    tasks=TRAIN_3TASKS_SPLIT,
    embodiment=ROBOTWIN_EMBODIMENT,
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    chunk_size=16,
    final_image_size=224,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    return_value_function_returns=False,
    gamma=0.998,
)

robotwin_qpos_debug_place_burger_fries_dataset = L(RoboTwinJointDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    tasks=DEBUG_BURGER_FRIES_SPLIT,
    embodiment=ROBOTWIN_EMBODIMENT,
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    chunk_size=16,
    final_image_size=224,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    return_value_function_returns=False,
    gamma=0.998,
)

robotwin_qpos_debug_single_task_dataset = L(RoboTwinJointDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    tasks=DEBUG_SINGLE_TASK_SPLIT,
    embodiment=ROBOTWIN_EMBODIMENT,
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    chunk_size=16,
    final_image_size=224,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    return_value_function_returns=False,
    gamma=0.998,
)


# ---------------------------------------------------------------------------
# Model / tokenizer shape — same slot layout as EE variant, only dims change.
# ---------------------------------------------------------------------------

_ROBOTWIN_QPOS_MODEL_CONFIG = dict(
    state_t=10,
    min_num_conditional_frames=5,
    max_num_conditional_frames=5,
    # Joint-space: [L_arm(6) + L_grip(1) + R_arm(6) + R_grip(1)] = 14.
    action_dim=14,
    proprio_dim=14,
    chunk_size=16,
    action_loss_multiplier=32,
    tokenizer=dict(
        chunk_duration=37,
    ),
)

_ROBOTWIN_QPOS_SCHEDULER = dict(
    cycle_lengths=[20000, 100000000000000],
    warm_up_steps=[2000, 0],
    f_start=[1e-6, 0.06],
    f_max=[1.0, 0.06],
    f_min=[0.3, 0.06],
)


# ---------------------------------------------------------------------------
# 3-task training config (qpos)
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_train_3tasks__qpos = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=_ROBOTWIN_QPOS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_QPOS_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_qpos_train_3tasks_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_qpos_train_3tasks_dataset,
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
            name="cosmos_predict2_2b_480p_robotwin_train_3tasks__qpos",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_train_3tasks__qpos__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_train_3tasks__qpos",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_train_3tasks__qpos__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# Debug single-task config (qpos)
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__qpos = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=10000,
            logging_iter=5,
        ),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_QPOS_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=5000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_qpos_debug_single_task_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_qpos_debug_single_task_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__qpos",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__qpos__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__qpos",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__qpos__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# place_burger_fries — joint-pose baseline (non-RAG, aloha-only qpos training).
# Used as the comparison baseline for the EE-based RAG variants in
# robotwin_experiment_configs.py (aloha→arx absolute/residual etc.).
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries__qpos = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=7000,
            logging_iter=5,
        ),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_QPOS_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=5000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_qpos_debug_place_burger_fries_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_qpos_debug_place_burger_fries_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=4,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries__qpos",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries__qpos__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries__qpos",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries__qpos__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# place_burger_fries — joint-pose RAG (aloha → arx, qpos action space).
# Same retrieval npz as the EE variant (image+proprio matching is action-rep
# agnostic) — RoboTwinJointRetDataset's __joint cache suffix keeps stats
# files separate so EE absolute / EE residual / qpos absolute / qpos residual
# never overwrite each other inside the same split-cache dir.
# ---------------------------------------------------------------------------

robotwin_qpos_rag_debug_place_burger_fries_arx_dataset = L(RoboTwinJointRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_burger_fries",
    query_embodiment=ROBOTWIN_EMBODIMENT,            # aloha-agilex_clean_50
    pool_embodiment="arx-x5_clean_50",
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_burger_fries__q-aloha-agilex_clean_50__p-arx-x5_clean_50__prefilterN5.npz",
    ),
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    chunk_size=16,
    final_image_size=224,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    use_wrist_images=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    ret_image_subsample=4,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    return_value_function_returns=False,
    gamma=0.998,
)

robotwin_qpos_rag_debug_place_burger_fries_arx_dataset_residual = L(RoboTwinJointRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_burger_fries",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="arx-x5_clean_50",
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_burger_fries__q-aloha-agilex_clean_50__p-arx-x5_clean_50__prefilterN5.npz",
    ),
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    chunk_size=16,
    final_image_size=224,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    use_wrist_images=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    ret_image_subsample=4,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    use_residual_actions=True,
    return_value_function_returns=False,
    gamma=0.998,
)


cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=7000, logging_iter=5),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_QPOS_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_qpos_rag_debug_place_burger_fries_arx_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_qpos_rag_debug_place_burger_fries_arx_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx_residual = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(max_iter=7000, logging_iter=5),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_QPOS_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_qpos_rag_debug_place_burger_fries_arx_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_qpos_rag_debug_place_burger_fries_arx_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx_residual__inference_only",
        ),
    )
)


ALL_ROBOTWIN_QPOS_CONFIGS = [
    cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__qpos,
    cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__qpos__inference_only,
    cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries__qpos,
    cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries__qpos__inference_only,
    cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx,
    cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx__inference_only,
    cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx_residual,
    cosmos_predict2_2b_480p_robotwin_qpos_rag_debug_place_burger_fries_arx_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_train_3tasks__qpos,
    cosmos_predict2_2b_480p_robotwin_train_3tasks__qpos__inference_only,
]


# ---------------------------------------------------------------------------
# Self-register at import time. cosmos_policy_experiment_configs.register_configs
# only iterates ALL_*_CONFIGS lists it knows about; ours isn't in that loop, so
# we do our own cs.store() with the same arg shape.
# ---------------------------------------------------------------------------

_cs = ConfigStore.instance()
for _item in ALL_ROBOTWIN_QPOS_CONFIGS:
    _cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
