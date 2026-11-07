# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# This codebase constitutes NVIDIA proprietary technology and is strictly
# confidential. Any unauthorized reproduction, distribution, or disclosure
# of this code, in whole or in part, outside NVIDIA is strictly prohibited
# without prior written consent.
#
# For inquiries regarding the use of this code in other NVIDIA proprietary
# projects, please contact the Deep Imagination Research Team at
# dir@exchange.nvidia.com.
# -----------------------------------------------------------------------------

import os

from hydra.core.config_store import ConfigStore
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils import log
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path  # noqa: F401
from cosmos_policy.datasets.aloha_dataset import ALOHADataset
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.datasets.robocasa_dataset import RoboCasaDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel
from cosmos_policy.models.policy_video2world_model_rectified_flow import CosmosPolicyVideo2WorldModelRectifiedFlow
from cosmos_policy.modules.hybrid_edm_sde import HybridEDMSDE
from cosmos_policy.config.experiment.pusht_experiment_configs import ALL_PUSHT_CONFIGS  # noqa: F401
from cosmos_policy.config.experiment.robocasa_doors_ret_experiment_configs import (  # noqa: F401
    ALL_ROBOCASA_DOORS_RET_CONFIGS,
)
from cosmos_policy.config.experiment.gr1_tabletop_experiment_configs import (  # noqa: F401
    ALL_GR1_TABLETOP_CONFIGS,
)
from cosmos_policy.config.experiment.robotwin_experiment_configs import (  # noqa: F401
    ALL_ROBOTWIN_CONFIGS,
)
from cosmos_policy.config.experiment.realrobot_ret_experiment_configs import (  # noqa: F401
    ALL_REALROBOT_RET_CONFIGS,
)
from cosmos_policy.config.experiment.egoverse_ret_experiment_configs import (  # noqa: F401
    ALL_EGOVERSE_RET_CONFIGS,
)
from cosmos_policy.config.experiment.egoverse_gr1_ret_experiment_configs import (  # noqa: F401
    ALL_EGOVERSE_GR1_RET_CONFIGS,
)
from cosmos_policy.config.experiment.tabletop_sim_experiment_configs import (  # noqa: F401
    ALL_TABLETOP_SIM_CONFIGS,
)

cs = ConfigStore.instance()
val_sampling_size_override = dict(
    video_length=121,
    video_height=704,
    video_width=1280,
)
BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")


# *** Main checkpoint ***
libero_all_4_suites_dataset = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
# Dataset with precomputed Reason1 embeddings (for rectified flow / T2V models - faster training)
libero_all_4_suites_dataset_reason1 = L(LIBERODataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    chunk_size=16,
    use_image_aug=True,
    use_wrist_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "LIBERO-Cosmos-Policy", "all_episodes"
    ),
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)

cosmos_predict2_2b_480p_libero = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-102-Size-2B-Res-480-Fps-16-Note-HQ_V5_from_26",
            {"override /data_train": "mock"},
            {"override /model": "policy_fsdp"},
            {"override /tokenizer": "policy_wan2pt1_tokenizer"},
            {
                "override /callbacks": [
                    "basic",
                    "long",
                    "cluster_speed",
                    "wandb",
                    "wandb_callback_actions",
                ]
            },
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=100000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                    num_sampling_step=9,
                ),
            ),
            run_validation=False,
            logging_iter=5,
            max_iter=1000000,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 30K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[30000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                conditioner=dict(
                    text=dict(
                        # IMPORTANT: We don't want any text dropout; otherwise, the model may fail to follow language
                        dropout_rate=0.0,
                    ),
                ),
                state_t=9,  # Latent temporal dim (blank, proprio, wrist, primary, action, future proprio, future wrist, future primary, value)
                min_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                max_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                sigma_conditional=0.0,  # No noise on conditional latents
                conditioning_strategy="frame_replace",
                denoise_replace_gt_frames=True,
                tokenizer=dict(
                    chunk_duration=33,  # 1 blank + 32 images (4 proprio, 4 wrist image, 4 primary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 value)
                ),
                ema=dict(
                    enabled=False,
                ),
                input_data_key="video",
                sde=L(HybridEDMSDE)(
                    hybrid_sigma_distribution=True,
                    p_mean=1.3862943611198906,  # Copied from base model config
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                    uniform_lower=1.0,
                    uniform_upper=85.0,
                ),
                adjust_video_noise=True,
                resize_online=True,
                resolution="224",
                high_sigma_strategy="none",
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path("hf://nvidia/Cosmos-Predict2-2B-Video2World/model-480p-16fps.pt"),
            load_training_state=False,  # This means do not load train state from the base checkpoint above (load_path); but when resuming this job, will load train state
            strict_resume=False,
            save_iter=1000,
            load_ema_to_reg=True,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_all_4_suites_dataset,
            sampler=L(DistributedSampler)(
                dataset=libero_all_4_suites_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=30,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_libero",
        ),
        upload_reproducible_setup=False,
    )
)
# Inference version
cosmos_predict2_2b_480p_libero__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_libero__inference_only",
        ),
    )
)


# *** Cosmos Predict2.5 (T2V / Rectified Flow backbone) ***
cosmos_predict2p5_2b_480p_libero = LazyDict(
    dict(
        defaults=[
            "/experiment/Stage-c_pt_4-Index-2-Size-2B-Res-720-Fps-16-Note-rf_with_edm_ckpt",
            {"override /data_train": "mock"},
            {"override /model": "policy_rectified_flow_fsdp"},
            {"override /tokenizer": "policy_wan2pt1_tokenizer"},
            {
                "override /callbacks": [
                    "basic",
                    "long",
                    "cluster_speed",
                    "wandb",
                    "wandb_callback_actions",
                ]
            },
            "_self_",
        ],
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=100000,
                    save_s3=False,
                    use_negative_prompt=False,
                    guidance=[0],
                ),
            ),
            run_validation=False,
            logging_iter=5,
            max_iter=1000000,
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        optimizer=dict(
            lr=2e-4,
            weight_decay=0.1,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            # LR decay for ~43K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[42800, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.06, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                conditioner=dict(
                    text=dict(
                        # IMPORTANT: We don't want any text dropout; otherwise, the model may fail to follow language
                        dropout_rate=0.0,
                    ),
                ),
                state_t=9,  # Latent temporal dim (blank, proprio, wrist, primary, action, future proprio, future wrist, future primary, value)
                min_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                max_num_conditional_frames=4,  # 1 blank, 3 conditioning (proprio, wrist, primary)
                conditional_frame_timestep=0,  # Near-clean timestep for conditional frames (0=clean, 1=noise)
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0, 4: 1.0},
                conditioning_strategy="frame_replace",
                denoise_replace_gt_frames=True,
                tokenizer=dict(
                    vae_pth = "hf://nvidia/Cosmos-Predict2.5-2B/tokenizer.pth",
                    chunk_duration=33,  # 1 blank + 32 images (4 proprio, 4 wrist image, 4 primary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 value)
                ),
                ema=dict(
                    enabled=False,
                ),
                input_data_key="video",
                shift=5,
                action_loss_multiplier=16,
                use_dynamic_shift=False,
                train_time_distribution="uniform", #"logitnormal",#,
                train_time_weight="uniform",
                resolution="224",
                text_encoder_class="T5",
                text_encoder_config=None,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        checkpoint=dict(
            load_path=get_checkpoint_path("hf://nvidia/Cosmos-Predict2.5-2B/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt"),
            load_training_state=False,
            strict_resume=False,
            save_iter=1000,
            load_ema_to_reg=True,
            load_from_object_store=dict(
                enabled=False,
            ),
            save_to_object_store=dict(
                enabled=False,
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=libero_all_4_suites_dataset_reason1,
            sampler=L(DistributedSampler)(
                dataset=libero_all_4_suites_dataset_reason1,
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
            name="cosmos_predict2p5_2b_480p_libero",
        ),
        upload_reproducible_setup=False,
    )
)
# Inference version (T2V / rectified flow)
cosmos_predict2p5_2b_480p_libero__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_libero__inference_only",
        ),
    )
)


# *** Main checkpoint ***
robocasa_50_demos_per_task_dataset = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "all_episodes"
    ),  # All demo rollouts (successes + failures)
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.99,
)
# Dataset with precomputed Reason1 embeddings (for rectified flow / T2V models)
robocasa_50_demos_stove_only_dataset_reason1 = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),
    # Use precomputed Reason1 embeddings instead of T5 (stored under same key name for compatibility)
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    task_split = ['PnPStoveToCounter'],  # Filter to only include demos from the "PnPStoveToCounter" task for this ablation
    # rollout_data_dir= None,
    rollout_data_dir=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "all_episodes"
    ),
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2_2b_480p_robocasa_50_demos_per_task = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=10,  # Latent temporal dim (blank, proprio, wrist image, primary image, secondary image, action, future proprio, future wrist image, future primary image, future secondary image, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                tokenizer=dict(
                    chunk_duration=37,  # 1 blank + 40 images (4 proprio, 4 wrist image, 4 primary image, 4 secondary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 future secondary, 4 value)
                ),
                mask_value_prediction_loss_for_policy_prediction = True,  # Mask value prediction loss for the first 5 frames (policy prediction frames); only compute value prediction loss for future frames
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_per_task_dataset,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_per_task_dataset,
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
            name="cosmos_predict2_2b_480p_robocasa_50_demos_per_task",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robocasa_50_demos_per_task",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
        ),
    )
)

cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=11,  # Latent temporal dim (blank, proprio, wrist image, primary image, secondary image, action, future proprio, future wrist image, future primary image, future secondary image, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.0, 5: 1.0},
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 proprio, 4 wrist image, 4 primary image, 4 secondary image, 4 action, 4 future proprio, 4 future wrist, 4 future primary, 4 future secondary, 4 value)
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_stove_only_dataset_reason1,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_stove_only_dataset_reason1,
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
            name="cosmos_predict2p5_2b_480p_robocasa_50_demos_stove_only",
        ),
    )
)
# Inference version (T2V / rectified flow)
cosmos_predict2p5_2b_480p_robocasa_50_demos_stove_only__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_50_demos_stove_only",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_50_demos_stove_only__inference",
        ),
    )
)


robocasa_50_demos_kitchen_stove_only = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    task_split = ['PnPStoveToCounter'],  # Filter to only include demos from the "PnPStoveToCounter" task for this ablation
    rollout_data_dir= None,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=True,
    gamma=0.99,
)

cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=50_000,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                tokenizer=dict(
                    chunk_duration=41,
                ),
                # mask_value_prediction_loss_for_policy_prediction = True,  # Mask value prediction loss
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_kitchen_stove_only,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_kitchen_stove_only,
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
            name="cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove",
        ),
    )
)


robocasa_50_demos_kitchen_stove_only_no_pred = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),  # Successful demos
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    task_split = ['PnPStoveToCounter'],  # Filter to only include demos from the "PnPStoveToCounter" task for this ablation
    rollout_data_dir= None,
    predict_future_states=False,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove_no_pred = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=20_000,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=6,  # Latent temporal dim (blank, proprio, wrist image, primary image, secondary image, action) — no future states, no value
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, wrist image, primary image, secondary image)
                max_num_conditional_frames=5,
                tokenizer=dict(
                    chunk_duration=21,  # 1 blank + 20 images (4 proprio, 4 wrist image, 4 primary image, 4 secondary image, 4 action)
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_50_demos_kitchen_stove_only_no_pred,
            sampler=L(DistributedSampler)(
                dataset=robocasa_50_demos_kitchen_stove_only_no_pred,
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
            name="cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove_no_pred",
        ),
    )
)


# Train-Seen splits baseline (Cosmos Predict 2.5, with future-state prediction)
# Splits (see success_only/Splits.md):
#   OpenDoor (Single or Double) -> OpenSingleDoor, OpenDoubleDoor
#   CountertoX                  -> PnPCounterToCab, PnPCounterToMicrowave,
#                                  PnPCounterToSink, PnPCounterToStove
#   Closedrawer                 -> CloseDrawer
#   TurnOffStove                -> TurnOffStove
ROBOCASA_TRAIN_SEEN_TASKS = [
    "OpenSingleDoor",
    "OpenDoubleDoor",
    "PnPCounterToCab",
    "PnPCounterToMicrowave",
    "PnPCounterToSink",
    "PnPCounterToStove",
    "CloseDrawer",
    "TurnOffStove",
]

robocasa_train_seen_dataset_reason1 = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),
    # Predict 2.5 uses Reason1 text embeddings (stored under the same key for compatibility)
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    use_stronger_image_aug=True,
    task_split=ROBOCASA_TRAIN_SEEN_TASKS,
    rollout_data_dir=None,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2p5_2b_480p_robocasa_train_seen = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=50_000,
        ),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=11,  # 1 blank + proprio + wrist + primary + secondary + action + future proprio + future wrist + future primary + future secondary + value
                min_num_conditional_frames=5,  # 1 blank + 4 conditioning (proprio, wrist, primary, secondary)
                max_num_conditional_frames=5,
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.0, 5: 1.0},
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 each for 10 categories)
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_train_seen_dataset_reason1,
            sampler=L(DistributedSampler)(
                dataset=robocasa_train_seen_dataset_reason1,
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
            name="cosmos_predict2p5_2b_480p_robocasa_train_seen",
        ),
    )
)
# Inference version (T2V / rectified flow)
cosmos_predict2p5_2b_480p_robocasa_train_seen__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_train_seen",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_train_seen__inference",
        ),
    )
)


# OpenDoor-only baseline (Single + Double door, Cosmos Predict 2.5, with future-state prediction)
ROBOCASA_OPEN_DOOR_TASKS = [
    "OpenSingleDoor",
    "OpenDoubleDoor",
]

robocasa_open_door_dataset_reason1 = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=True,
    task_split=ROBOCASA_OPEN_DOOR_TASKS,
    rollout_data_dir=None,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2p5_2b_480p_robocasa_open_door = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=10_000,
        ),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.0, 5: 1.0},
                tokenizer=dict(
                    chunk_duration=41,
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_open_door_dataset_reason1,
            sampler=L(DistributedSampler)(
                dataset=robocasa_open_door_dataset_reason1,
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
            name="cosmos_predict2p5_2b_480p_robocasa_open_door",
        ),
    )
)
# Inference version (T2V / rectified flow)
cosmos_predict2p5_2b_480p_robocasa_open_door__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_open_door",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_open_door__inference",
        ),
    )
)


# All-24-tasks baseline (Cosmos Predict 2.5 / RF, no retrieval, with future-state prediction)
# task_split=None -> loads every HDF5 under success_only (all 24 RoboCasa atomic tasks)
robocasa_all_24_tasks_dataset_reason1 = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "reason1_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=True,
    task_split=None,  # All 24 tasks
    rollout_data_dir=None,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=50_000,
        ),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.0, 5: 1.0},
                tokenizer=dict(
                    chunk_duration=41,
                ),
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_all_24_tasks_dataset_reason1,
            sampler=L(DistributedSampler)(
                dataset=robocasa_all_24_tasks_dataset_reason1,
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
            name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks",
        ),
    )
)
# Inference version (T2V / rectified flow)
cosmos_predict2p5_2b_480p_robocasa_all_24_tasks__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks__inference",
        ),
    )
)


# All-24-tasks baseline, T5 text conditioning
# ------------------------------------------
# Identical data and schedule to the reason1 run above; only the text encoder changes.
#
# Why: the eval composes its instruction from held-out ("B" split) object instances, so it
# emits phrasings the training demos never contain (e.g. "pick the banana from the counter
# and place it in the cabinet"). reason1_embeddings.pkl (242 keys) does not cover those;
# t5_embeddings.pkl (342 keys) is a strict superset of it and covers every instruction the
# eval enumerates. It also makes run_robocasa_eval's cache-miss path *correct* — that path
# computes a T5 embedding, which only matches a T5-conditioned policy. Bonus: 464 MB
# instead of 24.8 GB, so eval no longer parks 24.8 GB of text embeddings on each GPU.
#
# Cost: the Predict2.5 base is pretrained with Reason1 conditioning fed through a
# 100352 -> 1024 crossattn projection. Turning that projection off discards those
# pretrained weights, so text conditioning is relearned during finetuning. This is the
# same arrangement the egoverse/pusht experiments use.
robocasa_all_24_tasks_dataset_t5 = L(RoboCasaDataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only"),
    t5_text_embeddings_path=os.path.join(
        BASE_DATASETS_DIR, "RoboCasa-Cosmos-Policy", "success_only", "t5_embeddings.pkl"
    ),
    chunk_size=32,
    use_image_aug=True,
    use_wrist_images=True,
    use_third_person_images=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,
    use_stronger_image_aug=True,
    task_split=None,  # All 24 tasks
    rollout_data_dir=None,
    demonstration_sampling_prob=1.0,
    success_rollout_sampling_prob=1.0,
    return_value_function_returns=False,
    gamma=0.99,
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=11,
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                conditional_frames_probs={0: 0, 1: 0, 2: 0, 3: 0, 4: 0.0, 5: 1.0},
                tokenizer=dict(
                    chunk_duration=41,
                ),
                # T5 embeddings are already 1024-dim, so they enter cross-attention directly
                # instead of through the base model's Reason1 (100352 -> 1024) projection.
                text_encoder_class="T5",
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=robocasa_all_24_tasks_dataset_t5,
            sampler=L(DistributedSampler)(
                dataset=robocasa_all_24_tasks_dataset_t5,
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
            name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5",
        ),
    )
)
# Inference version (T2V / rectified flow)
# --- BRIDGE layout arm ---------------------------------------------------------------------
# Proprio rides in the tokenizer blank (latent 0) and a zero-pixel retrieval prefix
# (RET_ACTION + one RET_VIDEO at a PAST time) is inserted, with per-latent slot_roles /
# t_positions — the contract the EgoVerse mid checkpoints were trained under. 11 -> 12 latents,
# 41 -> 45 frames. Everything that carries information in the 63.17% recipe is kept.
_ROBOCASA_BRIDGE_STATE_T = 12
_ROBOCASA_BRIDGE_CHUNK_DURATION = 45

# base (warm start 없음, stock layout) 에서 future proprio 슬롯만 제거한 대조 arm.
# b2 는 bridge 도입과 동시에 future proprio 를 뺐기 때문에 그 제거가 성능에 기여했는지 분리되지
# 않는다. 이 arm 은 bridge 없이 그 변수 하나만 바꾼다 — base 63.17% 와 직접 비교된다.
# 슬롯이 action 뒤에 있으므로 conditioning 경계(n_cond=5)는 그대로이고 latent 11 -> 10,
# 프레임 41 -> 37 로 줄어든다.
cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_nofp = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5", "_self_"],
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_nofp"),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=10,
                tokenizer=dict(chunk_duration=37),
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                conditional_frames_probs={5: 1.0},
            )
        ),
        dataloader_train=dict(
            dataset=dict(drop_future_proprio=True),
            sampler=dict(dataset=dict(drop_future_proprio=True)),
        ),
    )
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_nofp__inference = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_nofp", "_self_"],
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_nofp__inference"),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(config=dict(state_t=10)),
    )
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5", "_self_"],
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge"),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=_ROBOCASA_BRIDGE_STATE_T,
                tokenizer=dict(chunk_duration=_ROBOCASA_BRIDGE_CHUNK_DURATION),
            )
        ),
        dataloader_train=dict(
            dataset=dict(bridge_layout=True),
            sampler=dict(dataset=dict(bridge_layout=True)),
        ),
    )
)

# bridge variant: RET_ACTION + RET_VIDEO
# (11 latents). future proprio is dropped in every bridge variant.
cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5", "_self_"],
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2"),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=11,
                tokenizer=dict(chunk_duration=41),
                # Conditioning must END where the action slot begins. The stock value (5) was
                # written for the stock layout; inheriting it here left the SECONDARY camera one
                # slot past the boundary, so the model had to GENERATE a view it should have been
                # given. Initial losses look fine either way, but over 20K the extra generation
                # target is what pushed future_image to 0.33 against 0.097 for stock.
                min_num_conditional_frames=6,
                max_num_conditional_frames=6,
                conditional_frames_probs={6: 1.0},
            )
        ),
        dataloader_train=dict(
            dataset=dict(bridge_layout=True, bridge_ret_action=True, bridge_ret_video=1,
                         bridge_future_stagger=True),
            sampler=dict(dataset=dict(bridge_layout=True, bridge_ret_action=True, bridge_ret_video=1,
                                      bridge_future_stagger=True)),
        ),
    )
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2__inference = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2", "_self_"],
        job=dict(group="cosmos_v2_inference",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2__inference"),
    )
)

# bridge variant: RET_ACTION only
# (10 latents). future proprio is dropped in every bridge variant.
# b2 + future proprio 슬롯 복원 (b2p). b2 는 mimicdroid-GR1 계약을 따라 future proprio 를 뺐는데,
# 그 제거가 TurnOnMicrowave 회귀(v1 90% -> b2 45%)의 원인인지 분리하기 위한 arm.
# 슬롯은 action 뒤에 들어가므로 conditioning 경계(n_cond=6)는 그대로이고 latent 만 11 -> 12,
# 프레임 41 -> 45 로 늘어난다. future proprio 는 비디오가 아니라 stagger 에서 빼고 NOW+dt 고정.
cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2p = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5", "_self_"],
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2p"),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=12,
                tokenizer=dict(chunk_duration=45),
                min_num_conditional_frames=6,
                max_num_conditional_frames=6,
                conditional_frames_probs={6: 1.0},
            )
        ),
        dataloader_train=dict(
            dataset=dict(bridge_layout=True, bridge_ret_action=True, bridge_ret_video=1,
                         bridge_future_stagger=True, bridge_future_proprio=True),
            sampler=dict(dataset=dict(bridge_layout=True, bridge_ret_action=True, bridge_ret_video=1,
                                      bridge_future_stagger=True, bridge_future_proprio=True)),
        ),
    )
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2p__inference = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2p", "_self_"],
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2p__inference"),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(config=dict(state_t=12)),
    )
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b1 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5", "_self_"],
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b1"),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=10,
                tokenizer=dict(chunk_duration=37),
                # Conditioning must END where the action slot begins. The stock value (5) was
                # written for the stock layout; inheriting it here left the SECONDARY camera one
                # slot past the boundary, so the model had to GENERATE a view it should have been
                # given. Initial losses look fine either way, but over 20K the extra generation
                # target is what pushed future_image to 0.33 against 0.097 for stock.
                min_num_conditional_frames=5,
                max_num_conditional_frames=5,
                conditional_frames_probs={5: 1.0},
            )
        ),
        dataloader_train=dict(
            dataset=dict(bridge_layout=True, bridge_ret_action=True, bridge_ret_video=0,
                         bridge_future_stagger=True),
            sampler=dict(dataset=dict(bridge_layout=True, bridge_ret_action=True, bridge_ret_video=0,
                                      bridge_future_stagger=True)),
        ),
    )
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b1__inference = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b1", "_self_"],
        job=dict(group="cosmos_v2_inference",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b1__inference"),
    )
)

# bridge variant: no retrieval prefix
# (9 latents). future proprio is dropped in every bridge variant.
cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b0 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5", "_self_"],
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b0"),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=9,
                tokenizer=dict(chunk_duration=33),
                # Conditioning must END where the action slot begins. The stock value (5) was
                # written for the stock layout; inheriting it here left the SECONDARY camera one
                # slot past the boundary, so the model had to GENERATE a view it should have been
                # given. Initial losses look fine either way, but over 20K the extra generation
                # target is what pushed future_image to 0.33 against 0.097 for stock.
                min_num_conditional_frames=4,
                max_num_conditional_frames=4,
                conditional_frames_probs={4: 1.0},
            )
        ),
        dataloader_train=dict(
            dataset=dict(bridge_layout=True, bridge_ret_action=False, bridge_ret_video=0,
                         bridge_future_stagger=True),
            sampler=dict(dataset=dict(bridge_layout=True, bridge_ret_action=False, bridge_ret_video=0,
                                      bridge_future_stagger=True)),
        ),
    )
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b0__inference = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b0", "_self_"],
        job=dict(group="cosmos_v2_inference",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b0__inference"),
    )
)

# Same bridge contract at action_loss_multiplier=48 (the mimicdroid-GR1 post-train value)
# instead of RoboCasa's 16. With 12 latents the action slot goes from 16/27 = 59% of the masked
# loss to 48/59 = 81%, so the video and proprio slots are correspondingly quieter.
cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_m48 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge", "_self_"],
        job=dict(group="cosmos_v2_finetune",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_m48"),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(action_loss_multiplier=48)
        ),
    )
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_m48__inference = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_m48", "_self_"],
        job=dict(group="cosmos_v2_inference",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_m48__inference"),
    )
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge__inference = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge", "_self_"],
        job=dict(group="cosmos_v2_inference",
                 name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge__inference"),
    )
)

cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5__inference",
        ),
    )
)


# All-24-tasks, T5 conditioning, on the paper's training schedule
# ---------------------------------------------------------------
# The _t5 run above compresses training into 20K steps at global batch 600 (24 ranks x 25).
# The Cosmos Policy RoboCasa result was produced with 45K steps at global batch 800
# (32 ranks x 25) and LR decay ending at 30K — see ROBOCASA.md and the paper's own
# cosmos_predict2_2b_480p_robocasa_50_demos_per_task, which inherits lr=1e-4 /
# cycle_lengths=[30000, ...] from cosmos_predict2_2b_480p_libero.
#
# This variant matches steps, batch (via 4 nodes — batch_size stays 25), and LR schedule.
# lr is already 1e-4 in _t5, which is what the paper uses at batch 800, so it is unchanged.
#
# Deliberately NOT matched (these make it a different experiment, not a longer one):
#   * Predict2.5 / rectified flow here vs the paper's Predict2 / EDM
#   * no value-function head, and demos only — the paper mixes all_episodes rollouts 50/50
#     and trains with return_value_function_returns=True
# So this closes the training-budget gap; it does not by itself reproduce the paper number.
cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_paper_schedule = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5",
            "_self_",
        ],
        trainer=dict(
            max_iter=45_000,
        ),
        scheduler=dict(
            cycle_lengths=[30_000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_paper_schedule",
        ),
    )
)
cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_paper_schedule__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_paper_schedule",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_paper_schedule__inference",
        ),
    )
)


# *** Main checkpoint ***
aloha_cosmos_policy_dataset_185_demos = L(ALOHADataset)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed"),
    t5_text_embeddings_path=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed", "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    treat_demos_as_success_rollouts=True,  # Include demos as success rollouts
    demonstration_sampling_prob=0.5,
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
)
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=dict(
            # LR decay for 20K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[2000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                state_t=11,  # Latent temporal dim (blank, proprio, left wrist, right wrist, primary, action, future proprio, future left wrist, future right wrist, future primary, value)
                min_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)
                max_num_conditional_frames=5,  # 1 blank, 4 conditioning (proprio, left wrist, right wrist, primary)
                tokenizer=dict(
                    chunk_duration=41,  # 1 blank + 40 images (4 proprio, 4 left wrist image, 4 right wrist image, 4 primary image, 4 action, 4 future proprio, 4 future left wrist, 4 future right wrist, 4 future primary, 4 value)
                ),
            ),
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=aloha_cosmos_policy_dataset_185_demos,
            sampler=L(DistributedSampler)(
                dataset=aloha_cosmos_policy_dataset_185_demos,
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
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only",
        ),
    )
)


# ALOHA planning model
# Dataset: 648 rollouts from evaluations with Cosmos Policy, pi05, pi0, OpenVLA-OFT+, Diffusion Policy
# NOTE: This rollouts dataset is not released; you will need to replace `rollout_data_dir` below with your own rollouts dataset
aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset = L(
    ALOHADataset
)(
    data_dir=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed"),
    t5_text_embeddings_path=os.path.join(BASE_DATASETS_DIR, "ALOHA-Cosmos-Policy", "preprocessed", "t5_embeddings.pkl"),
    chunk_size=50,
    use_image_aug=True,
    use_stronger_image_aug=True,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    treat_demos_as_success_rollouts=False,  # Don't include demos as success rollouts because they have a fixed episode length + we want to focus on real policy rollouts
    demonstration_sampling_prob=0.1,  # Smaller demonstration sampling prob - more emphasis on rollouts
    success_rollout_sampling_prob=0.5,
    return_value_function_returns=True,
    gamma=0.998,  # Higher gamma for ALOHA because episodes can have up to 1.5-2.0K steps  # (s, a, s', v)
    rollout_data_dir=os.path.join(BASE_DATASETS_DIR, "PATH/TO/YOUR/ROLLOUTS/DATASET"),  # JPEG images
    use_jpeg_for_rollouts=True,  # JPEG images
)
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80",
            "_self_",
        ],
        checkpoint=dict(
            # Resume from 50K checkpoint of base Cosmos Policy run
            load_path=get_checkpoint_path(
                "hf://nvidia/Cosmos-Policy-ALOHA-Predict2-2B/Cosmos-Policy-ALOHA-Predict2-2B.pt"
            ),
        ),
        scheduler=dict(
            # LR decay for 15K steps in cycle #1, then decay by 5x and stay constant forever in cycle #2
            cycle_lengths=[15000, 100000000000000],
            warm_up_steps=[1500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset,
            sampler=L(DistributedSampler)(
                dataset=aloha_2025_09_18__648_rollouts__cosmos_policy__pi05__pi0__openvla_oft__diffusion_policy__dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=25,
            drop_last=True,
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                mask_current_state_action_for_value_prediction=True,  # Use input masking to mask out irrelevant inputs (current state and action) during value prediction
            ),
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func",
        ),
    )
)
# Inference version
cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func",
            "_self_",
        ],
        model=L(CosmosPolicyVideo2WorldModel)(
            config=dict(
                sde=L(HybridEDMSDE)(
                    sigma_max=80,
                    sigma_min=4,
                )
            )
        ),
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only",
        ),
    )
)


def register_configs():
    cs = ConfigStore.instance()
    # Register the experiments
    for _item in [
        # LIBERO
        cosmos_predict2_2b_480p_libero,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_libero__inference_only,
        # LIBERO (T2V / Rectified Flow)
        cosmos_predict2p5_2b_480p_libero,  # *** Main checkpoint (T2V backbone) ***
        cosmos_predict2p5_2b_480p_libero__inference_only,
        # RoboCasa
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference,
        cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove,  # *** Main checkpoint with smaller dataset (50 demos from a single task) for ablating data scaling ***
        cosmos_predict2_2b_480p_robocasa_50_demos_kitchen_stove_no_pred,
        # RoboCasa (T2V / Rectified Flow)
        # cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task,  # *** Main checkpoint (T2V backbone) ***
        # cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task__inference,
        cosmos_predict2p5_2b_480p_robocasa_train_seen,  # Train-Seen splits baseline (with future-state prediction)
        cosmos_predict2p5_2b_480p_robocasa_train_seen__inference,
        cosmos_predict2p5_2b_480p_robocasa_open_door,  # OpenDoor-only baseline (Single + Double)
        cosmos_predict2p5_2b_480p_robocasa_open_door__inference,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks,  # All-24-tasks baseline (no retrieval)
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks__inference,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5,  # Same, but T5 text conditioning
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5__inference,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_nofp,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_nofp__inference,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge__inference,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b0,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b0__inference,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b1,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b1__inference,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2__inference,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2p,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_b2p__inference,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_m48,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_bridge_m48__inference,
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_paper_schedule,  # 45K @ batch 800, decay 30K
        cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_paper_schedule__inference,
        # ALOHA
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80,  # *** Main checkpoint ***
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only,
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func,  # ALOHA planning model
        cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__resumeFrom50K_648_rollouts_Vsprime_value_func__inference_only,
        *ALL_PUSHT_CONFIGS,
        # RoboCasa GR1 tabletop (GR00T teleop sim, single ego camera, 44-dim actions)
        *ALL_GR1_TABLETOP_CONFIGS,
        # RoboCasa door retrieval (Predict2.5 / RF + RoboCasa→MG retrieval)
        *ALL_ROBOCASA_DOORS_RET_CONFIGS,
        # RoboTwin 2.0 (no proprio — see robotwin_experiment_configs.py)
        *ALL_ROBOTWIN_CONFIGS,
        # Real-robot bg2 bimanual + human retrieval pool (Predict2.5 + RF)
        *ALL_REALROBOT_RET_CONFIGS,
        # EgoVerse cross-embodiment retrieval (Predict2.5 + RF, full video+action gen)
        *ALL_EGOVERSE_RET_CONFIGS,
        # EgoVerse mimicdroid-GR1 multimodal retrieval (3-cam image+state+action gen)
        *ALL_EGOVERSE_GR1_RET_CONFIGS,
        # Tabletop-Sim bimanual ALOHA (57-D mapping, GR1 SHORT state layout)
        *ALL_TABLETOP_SIM_CONFIGS,
    ]:
        experiment_name = _item["job"]["name"]
        log.info(f"Registering experiment: {experiment_name}")
        cs.store(
            group="experiment",
            package="_global_",
            name=experiment_name,
            node=_item,
        )
