"""RoboCasa GR1 tabletop (GR00T teleop sim) experiments — Predict2.5 / rectified flow.

Dataset: nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim, 24 tabletop tasks x 1000 human teleop
demos on GR1ArmsAndWaistFourierHands, ~4.5M frames. See docs/robocasa_all24_t5.md for the
RoboCasa-side recipe these settings are derived from.

What differs from the RoboCasa all-24 configs, and why:

* **One camera.** The dataset has only `observation.images.ego_view`, so the latent sequence is
  1 blank + proprio + ego_view + action + future proprio + future ego_view = 5 content slots.
  With 4 duplicates per slot that is 1 + 5*4 = 21 frames, which the WAN2.1 tokenizer maps to
  1 + (21-1)//4 = 6 latents. Hence `state_t=6`, `chunk_duration=21` — both are computed by
  `gr1_tabletop_dataset.model_shape()` and were verified against the loader's actual output.
  (RoboCasa, with three cameras, emits 37 frames / 10 latents.)

* **44-dim actions.** GR1 sends left_arm 7 | left_hand 6 | left_leg 6 | neck 3 | right_arm 7 |
  right_hand 6 | right_leg 6 | waist 3. Legs and neck are identically zero on this embodiment,
  but all 44 are kept: they fit the action slot (44*32 = 1408 of 16*28*28 = 12544 elements) and
  slicing would add a mapping that training and eval could drift on.

* **40K iterations, decay to 30K.** The data is ~12.5x RoboCasa's (~4.5M vs ~360K frames), so
  RoboCasa's 20K x batch 600 would be 2.7 epochs here instead of 33. 40K gives ~5.3 epochs.
  Scaling iterations with the data 1:1 is not warranted — the RoboCasa base arm had already
  converged by 15K (63.33% at 15K vs 63.17% at 20K, 24 tasks x 50 trials).

* **save_iter 2500, not 1000.** 40K iterations at 1000 would write 40 checkpoints x ~11.6 GB =
  464 GB. The shared filesystem has hit 324 GB free during this project.

* **T5 conditioning.** meta/tasks.jsonl carries CamelCase task IDs; build_gr1_tabletop_t5_embeddings.py
  renders each as a sentence, encodes it, and keys the pickle by the raw ID. 24 fixed strings,
  50 MB, and no per-episode object names — the open-vocabulary miss that affects RoboCasa eval
  cannot occur here.
"""

import os

from megatron.core import parallel_state
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.datasets.gr1_tabletop_dataset import GR1TabletopDataset, model_shape
from cosmos_policy.models.policy_video2world_model_rectified_flow import CosmosPolicyVideo2WorldModelRectifiedFlow

GR1_TABLETOP_DIR = os.environ.get(
    "GR1_TABLETOP_DIR", "/mnt/ddn/dataset/gr1_tabletop_teleop"
)
STATE_T, CHUNK_DURATION = model_shape()  # (6, 21) — keep the config and loader from drifting apart

# The plain-BC arms mirror the mimicdroid-GR1 post-train contract exactly
# (datasets/egoverse_gr1_plain_dataset.py): blank1 carries the proprio, there is no separate
# proprio slot and no future-proprio slot, and only two latents are conditioning.
PLAIN_STATE_T, PLAIN_CHUNK_DURATION = model_shape(plain_bc_layout=True)  # (4, 13)
PLAIN_NUM_COND = 2

# BLANK-VIDEO bridge: plain-BC plus the zero-pixel retrieval prefix (RET_ACTION + one
# RET_VIDEO at a PAST time) that the ret-mid checkpoints were trained with. Conditioning is
# everything before the action slot.
BRIDGE_STATE_T, BRIDGE_CHUNK_DURATION = model_shape(
    plain_bc_layout=True, n_blank_ret_video=1, use_blank_ret_action=True
)  # (6, 21)
BRIDGE_NUM_COND = 4

gr1_tabletop_dataset = L(GR1TabletopDataset)(
    data_dir=os.path.join(GR1_TABLETOP_DIR, "LeRobot"),
    t5_text_embeddings_path=os.path.join(GR1_TABLETOP_DIR, "t5_embeddings_perep.pkl"),
    # v2 stats: constant (legs/neck) dims are widened SYMMETRICALLY so they normalise to 0
    # instead of -1. The v1 file is kept because the first 80K run and its 1200-episode eval
    # were produced with it.
    dataset_stats_path=os.path.join(GR1_TABLETOP_DIR, "dataset_statistics_v2.json"),
    chunk_size=32,
    use_proprio=True,
    normalize_proprio=True,
    normalize_actions=True,
    predict_future_states=True,
    use_image_aug=True,
    use_stronger_image_aug=True,
    num_duplicates_per_image=4,
    future_offset=16,
    # Inherited from the LIBERO dataloader this experiment extends; pinned to the only values
    # this dataset supports (one camera, demos only, no value function).
    use_wrist_images=False,
    use_third_person_images=False,
    rollout_data_dir=None,
    return_value_function_returns=False,
)

# Base arm: start from the stock Predict2.5 checkpoint (no EgoVerse warm-start).
cosmos_predict2p5_2b_480p_gr1_tabletop = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=85_000,  # the run stops at 80K via the yaml; leaves headroom to extend
        ),
        checkpoint=dict(
            # 80K at 2500 would write 32 checkpoints x 11 GB = 352 GB. 5000 keeps it at 16
            # checkpoints (~176 GB) and still gives a 5K grid for intermediate evals.
            save_iter=5000,
        ),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=STATE_T,
                # 1 blank + proprio + ego_view = 3 latents of conditioning before the action slot
                min_num_conditional_frames=3,
                max_num_conditional_frames=3,
                conditional_frames_probs={3: 1.0},
                tokenizer=dict(
                    chunk_duration=CHUNK_DURATION,
                ),
                text_encoder_class="T5",
                # T5 embeddings are 1024-d, so they enter cross-attention directly rather than
                # through the base model's Reason1 (100352 -> 1024) projection.
                net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
                # Weight of the action latent in the velocity loss. NOT set here originally,
                # which left it at the model default of 1 — with 6 latents and one action slot
                # that gives actions ~1/6 of the gradient. Every setup that works sets it:
                # 48 (the mimicdroid-GR1 post-train value) was tried first and the 40K sweep came
                # in at 10.4% against 19.8% for the old recipe at the same iteration. 16 is what
                # RoboCasa uses, and RoboCasa is the closer task: a single arm reaching for and
                # placing objects, with video prediction still carrying real signal. At 48 the
                # action slot takes 48/53 of the masked loss and the video/proprio slots are
                # effectively silenced.
                action_loss_multiplier=16,
                action_dim=44,
                proprio_dim=44,
                use_action_projection=False,
                use_proprio_projection=False,
            ),
        ),
        optimizer=dict(
            lr=1e-4,
        ),
        scheduler=dict(
            # Decay across the WHOLE budget, and land on the value cycle #2 holds.
            #
            # The inherited LIBERO/RoboCasa schedule is [30K, forever] with f_min=0.3 then a flat
            # 0.06 — cycle #1 stops at 0.3 and cycle #2 resumes at 0.06, a 5x drop at the seam.
            # On a 40K budget that seam lands at 30K, so the last quarter of training ran at
            # 6e-6 (1/17 of peak). Measured progress kept improving through it, but there is no
            # reason to spend a quarter of an 80K run at a near-frozen LR.
            #
            # f_min == cycle #2's f_max, so LR decays 1e-4 -> 6e-6 smoothly over the full 80K
            # with no discontinuity. (The same file's paper-schedule variant uses this shape.)
            cycle_lengths=[80_000, 100000000000000],
            warm_up_steps=[1000, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.06, 0.06],
        ),
        dataloader_train=L(DataLoader)(
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            dataset=gr1_tabletop_dataset,
            sampler=L(DistributedSampler)(
                dataset=gr1_tabletop_dataset,
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
            name="cosmos_predict2p5_2b_480p_gr1_tabletop",
        ),
    )
)

# --- plain-BC layout arm ------------------------------------------------------------------
# Same data, same 80K/LR/language recipe; only the latent contract changes to the reference
# plain-BC one. This is the arm the EgoVerse / mimicdroid-GR1 checkpoints were trained under,
# so a warm start has a layout it recognises.
cosmos_predict2p5_2b_480p_gr1_tabletop_plainbc = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_gr1_tabletop",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2p5_2b_480p_gr1_tabletop_plainbc",
        ),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=PLAIN_STATE_T,
                min_num_conditional_frames=PLAIN_NUM_COND,
                max_num_conditional_frames=PLAIN_NUM_COND,
                conditional_frames_probs={PLAIN_NUM_COND: 1.0},
                tokenizer=dict(chunk_duration=PLAIN_CHUNK_DURATION),
            )
        ),
        dataloader_train=dict(
            dataset=dict(plain_bc_layout=True, emit_slot_pe=True),
            sampler=dict(dataset=dict(plain_bc_layout=True, emit_slot_pe=True)),
        ),
    )
)

cosmos_predict2p5_2b_480p_gr1_tabletop_bridge = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_gr1_tabletop", "_self_"],
        job=dict(group="cosmos_v2_finetune", name="cosmos_predict2p5_2b_480p_gr1_tabletop_bridge"),
        model=L(CosmosPolicyVideo2WorldModelRectifiedFlow)(
            config=dict(
                state_t=BRIDGE_STATE_T,
                min_num_conditional_frames=BRIDGE_NUM_COND,
                max_num_conditional_frames=BRIDGE_NUM_COND,
                conditional_frames_probs={BRIDGE_NUM_COND: 1.0},
                tokenizer=dict(chunk_duration=BRIDGE_CHUNK_DURATION),
            )
        ),
        dataloader_train=dict(
            dataset=dict(plain_bc_layout=True, emit_slot_pe=True,
                         n_blank_ret_video=1, use_blank_ret_action=True),
            sampler=dict(dataset=dict(plain_bc_layout=True, emit_slot_pe=True,
                                      n_blank_ret_video=1, use_blank_ret_action=True)),
        ),
    )
)

cosmos_predict2p5_2b_480p_gr1_tabletop_bridge__inference = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2p5_2b_480p_gr1_tabletop_bridge", "_self_"],
        job=dict(group="cosmos_v2_inference",
                 name="cosmos_predict2p5_2b_480p_gr1_tabletop_bridge__inference"),
    )
)

cosmos_predict2p5_2b_480p_gr1_tabletop_plainbc__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_gr1_tabletop_plainbc",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_gr1_tabletop_plainbc__inference",
        ),
    )
)

cosmos_predict2p5_2b_480p_gr1_tabletop__inference = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2p5_2b_480p_gr1_tabletop",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2p5_2b_480p_gr1_tabletop__inference",
        ),
    )
)

ALL_GR1_TABLETOP_CONFIGS = [
    cosmos_predict2p5_2b_480p_gr1_tabletop,
    cosmos_predict2p5_2b_480p_gr1_tabletop_plainbc,
    cosmos_predict2p5_2b_480p_gr1_tabletop_bridge,
    cosmos_predict2p5_2b_480p_gr1_tabletop_bridge__inference,
    cosmos_predict2p5_2b_480p_gr1_tabletop_plainbc__inference,
    cosmos_predict2p5_2b_480p_gr1_tabletop__inference,
]
