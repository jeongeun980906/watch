# SPDX-License-Identifier: Apache-2.0
"""Tabletop-Sim (TwinVLA bimanual ALOHA) post-train — GR1 SHORT state layout.

5 sim tasks / 260 demos / 52,043 windows, converted to the egoverse 57-D convention by
``convert_tabletop_to_egoverse57.py`` (see ``cosmos_policy/datasets/tabletop_convert.py``
for the mapping and ``test_tabletop_action57_roundtrip.py`` for its round-trip proof).

Everything about the latent layout is inherited from the GR1 SHORT state stack so the
``egoverse_ret_short_state_v11`` mid warm-starts with no shape gaps::

    0  cur_state   past state57 chunk, tiled       cond
    1  ret_action  BLANK (no pair pool yet)        cond
    2  cur_image   448 2x2 grid x4 -> 1 latent     cond
    3  action      32 x 57  abs_action57           GEN
    -- num_conditional_frames = 3, chunk_duration = 13 --

Target is ABSOLUTE (`abs_action57` = the recorded GELLO command), not a delta: the
Tabletop HDF5 ships command and measured proprio as separate streams, so there is
nothing to reconstruct, and the decode at eval is a constant frame transform with no
integration. See the delta-accumulation number in the round-trip test (0.137 deg over
140 steps) for why the delta path is not worth building here.

3-view grid for run #1 (``use_center_top=False``) — the mid was trained with the BR
quadrant blank. ``center_top`` (the collaborator_pov camera) IS in the converted store,
so 4-view is a flag flip with no re-conversion.

Experiments::

    ..._tabletop_sim_5task_short_state        SHORT abs, warm-start from short state mid 8K
    ..._tabletop_sim_5task_short_state_base   same, no mid  (checkpoint={}) -> mid ablation

Launch::

    uv run --extra cu128 --group robocasa --python 3.10 \
      torchrun --nproc_per_node=8 -m cosmos_policy.scripts.train \
      --config=cosmos_policy/config/config.py -- \
      experiment=cosmos_predict2p5_2b_tabletop_sim_5task_short_state
"""
from __future__ import annotations

import copy

from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy.datasets.tabletop_sim_dataset import (
    TABLETOP_ROOT,
    TabletopSimPlainEndImageDataset,
    TabletopSimPlainStateDataset,
    TabletopSimShortStateDataset,
)
from cosmos_policy.models.policy_video2world_model_realrobot_ret import (
    CosmosPolicyRealRobotRetModelRectifiedFlow,
)

MANIFEST = f"{TABLETOP_ROOT}/manifest"
CHUNK = 32          # 32 @ 20 Hz = 1.6 s; must match the mid so the action slot lines up
STATE_T = 4         # SHORT state layout (with_proprio=True)

# short STATE mid (state_t=4 / chunk 32 / dim 57 all identical to this post-train, so
# expect missing_keys=[] on load — VERIFY that line at run start).
#
# 10K, not the 8K the GR1 configs name: only ``model_000010000.pt`` was ever converted in
# that run (the iter_000008000 DCP dir is there if the 8K arm is wanted —
# ``convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_short_state_v11 --iter 8000``
# with IMAGINAIRE_OUTPUT_ROOT set to the run root). 10K is the completed run.
_CKSHORT_STATE_MID = (
    "${oc.env:TABLETOP_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_short_state_v11_4node/"
    "cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_short_state_v11/"
    "checkpoints/model_000010000.pt}"
)


def _model_override(state_t: int, n_cond: int) -> dict:
    """Byte-for-byte the GR1 ``_GR1_SHORT_MODEL_OVERRIDE`` — the whole point of mapping
    Tabletop into 57-D is that this does not have to change."""
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
        chunk_size=CHUNK,
        action_loss_multiplier=48,
        mask_loss_for_action_future_state_prediction=False,
        mask_value_prediction_loss_for_policy_prediction=False,
        mask_current_state_action_for_value_prediction=False,
        mask_future_state_for_qvalue_prediction=False,
    )


# Normalization: pctl for BOTH streams, matching the GR1 SHORT config
# (robust_state_stats=False). Measured saturation |x|>0.9 is 73% on the target and 61%
# on proprio vs GR1's 62% / 24% — but the excess is entirely head9 (constant by
# construction) and hand15, which encodes a gripper that ALOHA teleop keeps at its stops.
# The eef blocks that actually carry the manipulation signal sit at 16-19%, below the 20%
# gate that pushed GR1's mid to robust stats. Flip robust_state_stats=True to revisit.
_TABLETOP_5TASK_DATASET = L(TabletopSimShortStateDataset)(
    manifest_dir=MANIFEST,
    dataset_root=TABLETOP_ROOT,
    use_center_top=False,
    t5_text_embeddings_path=f"{MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST}/action57_stats.json",
    state_stats_path=f"{MANIFEST}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST}/abs_action57_stats_pctl.json",
    robust_state_stats=False,
    chunk_size=CHUNK,
    per_frame_stride=1,
    final_image_size=448,
    num_duplicates_per_image=4,
    use_residual_actions=False,
    retrieval_dropout_prob=0.0,
    blank_retrieval=True,
    require_t5=True,
)


# 2026-08-23: 8K -> 20K. 8K 최종(plain_state_base)에서 teacher-forced 오차는 시연 스텝당
# 이동량의 25~30% 수준이라 접촉 파지에 부족했고, 롤아웃 성공률이 2/100 에 머물렀다.
# RoboCasa arm 들과 같은 20K 예산으로 맞춰 재측정한다. cycle_lengths 도 함께 올려
# 전 구간이 코사인 한 주기로 유지되게 한다 (LR 절벽 없음).
MAX_ITER = 20_000
# 20K x 1000 = arm 당 체크포인트 20개 x 11 GB = 220 GB. /mnt/ddn 이 99% 라 2000 으로 낮춘다
# (arm 당 10개, 110 GB). 20K 는 2000 의 배수라 최종 체크포인트는 그대로 남는다.
SAVE_ITER = 2_000


def _tabletop_short_post(name: str, dataset, ckpt: str, state_t: int = STATE_T,
                         model_override: dict | None = None) -> LazyDict:
    """GR1 ``_gr1_short_post`` with the Tabletop dataset swapped in.

    Schedule: MAX_ITER iters = one cosine cycle (no LR cliff), warmup 500, lr 1e-4,
    batch 4 @ 448. 52K windows / (4 x 8 GPUs) ~= 1.6K steps per epoch. MAX_ITER was 8K
    (~5 epochs) until 2026-08-23; now 20K (~12 epochs) to match the RoboCasa arms.
    """
    return LazyDict(
        dict(
            defaults=[
                # NOTE: p5, not p2. The rectified-flow model reads train_time_distribution
                # / shift / train_time_weight from its config, and only the Predict2.5
                # parent (realrobot_ret_experiment_configs.py) carries the RF config class.
                # Inheriting the p2 (EDM) twin resolves but dies at model init with
                # "Key 'train_time_distribution' not in 'CosmosPolicyVideo2WorldConfig'".
                "/experiment/cosmos_predict2p5_2b_480p_realrobot_bg2",
                "_self_",
            ],
            trainer=dict(max_iter=MAX_ITER),
            optimizer=dict(lr=1e-4),
            scheduler=dict(
                cycle_lengths=[MAX_ITER, 100000000000000],
                warm_up_steps=[500, 0],
                f_start=[1e-6, 0.5],
                f_max=[1.0, 0.5],
                f_min=[0.5, 0.5],
            ),
            checkpoint=dict(save_iter=SAVE_ITER,
                            **({"load_path": ckpt, "strict_resume": False} if ckpt else {})),
            model=L(CosmosPolicyRealRobotRetModelRectifiedFlow)(
                config=dict(**(model_override or _model_override(state_t, state_t - 1))),
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


def _inference_only(parent_name: str) -> LazyDict:
    return LazyDict(
        dict(
            defaults=[f"/experiment/{parent_name}", "_self_"],
            job=dict(group="cosmos_v2_inference", name=f"{parent_name}__inference_only"),
        )
    )


# Warm-started from the SHORT state mid — the arm this whole 57-D mapping exists for.
cosmos_predict2p5_2b_tabletop_sim_5task_short_state = _tabletop_short_post(
    "cosmos_predict2p5_2b_tabletop_sim_5task_short_state",
    _TABLETOP_5TASK_DATASET,
    _CKSHORT_STATE_MID,
)
cosmos_predict2p5_2b_tabletop_sim_5task_short_state__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_5task_short_state"
)

# No mid (HF base init). Always run alongside the arm above — it is the only way to
# separate "the mid transfers to a new embodiment" from "the task is just learnable".
cosmos_predict2p5_2b_tabletop_sim_5task_short_state_base = _tabletop_short_post(
    "cosmos_predict2p5_2b_tabletop_sim_5task_short_state_base",
    _TABLETOP_5TASK_DATASET,
    "",
)
cosmos_predict2p5_2b_tabletop_sim_5task_short_state_base__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_5task_short_state_base"
)


# ---------------------------------------------------------------------------
# PLAIN (BC) layout — structural twin of cosmos_predict2p5_2b_egoverse_gr1_plain_state_v6
#
#   slot:  cur_state | cur_image | action | future x8      state_t = 11
#   role:      0     |     3     |   4    |   5            chunk_duration = 41
#   cond:     yes    |    yes    |  GEN   |  GEN           num_conditional_frames = 2
#
# vs the SHORT arm: the retrieval channel is REMOVED (not blanked), and the query's own
# next 32 frames become a generation target. That future-video block is the auxiliary
# signal the GR1 BC arm trains with; it costs roughly 4x the per-iter time of SHORT.
# ---------------------------------------------------------------------------
PLAIN_STATE_T = 11


def _plain_model_override() -> dict:
    """_GR1_PLAIN_MODEL_OVERRIDE with nothing changed but the comments.

    action_loss_multiplier=48 comes straight from _GR1_V2_MODEL_OVERRIDE ("match
    mid-v2") — the action loss would otherwise be swamped by the 8 future-video latents
    now sharing the objective, which is exactly why the multiplier exists.
    """
    return dict(
        state_t=PLAIN_STATE_T,
        min_num_conditional_frames=2,          # cur_state + cur_image
        max_num_conditional_frames=2,
        conditional_frames_probs={2: 1.0},
        tokenizer=dict(chunk_duration=1 + (PLAIN_STATE_T - 1) * 4),   # 41
        text_encoder_class="T5",
        net=dict(use_crossattn_projection=False, crossattn_emb_channels=1024),
        use_action_projection=False,
        use_proprio_projection=False,
        action_dim=57,
        proprio_dim=57,
        chunk_size=CHUNK,
        action_loss_multiplier=48,
        mask_loss_for_action_future_state_prediction=False,
        mask_value_prediction_loss_for_policy_prediction=False,
        mask_current_state_action_for_value_prediction=False,
        mask_future_state_for_qvalue_prediction=False,
    )


# v10 STATE mid @8K — the canonical "state mid" this lab warm-starts from (it is also
# the transfer mid behind the RoboCasa-24 arm). GR1's plain arm names
# `..._v3_448_state_clean`, whose .pt no longer exists on disk; this is that lineage's
# live checkpoint. Carries a TRAINED net.role_embedder (8x2048, all rows nonzero), so
# the v2 role/continuous-time PE transfers rather than starting at zero.
_CKV10_448_STATE = (
    "${oc.env:TABLETOP_BASE_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_4node/"
    "cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state/"
    "checkpoints/model_000008000.pt}"
)

_TABLETOP_5TASK_PLAIN_DATASET = L(TabletopSimPlainStateDataset)(
    manifest_dir=MANIFEST,
    dataset_root=TABLETOP_ROOT,
    use_center_top=False,
    t5_text_embeddings_path=f"{MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST}/action57_stats.json",
    state_stats_path=f"{MANIFEST}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST}/abs_action57_stats_pctl.json",
    robust_state_stats=False,
    chunk_size=CHUNK,
    per_frame_stride=1,
    final_image_size=448,
    num_duplicates_per_image=4,
    use_residual_actions=False,
    retrieval_dropout_prob=0.0,
    require_t5=True,
)

cosmos_predict2p5_2b_tabletop_sim_5task_plain_state = _tabletop_short_post(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_state",
    _TABLETOP_5TASK_PLAIN_DATASET,
    _CKV10_448_STATE,
    model_override=_plain_model_override(),
)
cosmos_predict2p5_2b_tabletop_sim_5task_plain_state__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_state"
)

cosmos_predict2p5_2b_tabletop_sim_5task_plain_state_base = _tabletop_short_post(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_state_base",
    _TABLETOP_5TASK_PLAIN_DATASET,
    "",
    model_override=_plain_model_override(),
)
cosmos_predict2p5_2b_tabletop_sim_5task_plain_state_base__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_state_base"
)


# ---------------------------------------------------------------------------
# PLAIN-END: same roles as PLAIN, but the future block is the single chunk-END frame
# (t+32) instead of all 32 frames. state_t 11 -> 4, chunk_duration 41 -> 13 — i.e. the
# SHORT arm's sequence length, with the blank ret slot replaced by a real prediction
# target. Matches the GR1 v1 post-train's "action + chunk-END image" generation set.
# ---------------------------------------------------------------------------
END_STATE_T = 4
# PLAIN-END 의 action_loss_multiplier. PLAIN(48)과 갈라지는 값 — 위 _plain_end_model_override 주석 참고.
END_ACTION_LOSS_MULTIPLIER = 16

_TABLETOP_5TASK_PLAIN_END_DATASET = L(TabletopSimPlainEndImageDataset)(
    manifest_dir=MANIFEST,
    dataset_root=TABLETOP_ROOT,
    use_center_top=False,
    t5_text_embeddings_path=f"{MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST}/action57_stats.json",
    state_stats_path=f"{MANIFEST}/state57_stats_pctl.json",
    target_group="abs_action57",
    target_stats_path=f"{MANIFEST}/abs_action57_stats_pctl.json",
    robust_state_stats=False,
    chunk_size=CHUNK,
    per_frame_stride=1,
    final_image_size=448,
    num_duplicates_per_image=4,
    use_residual_actions=False,
    retrieval_dropout_prob=0.0,
    require_t5=True,
)


def _plain_end_model_override() -> dict:
    o = _plain_model_override()
    o["state_t"] = END_STATE_T
    o["tokenizer"] = dict(chunk_duration=1 + (END_STATE_T - 1) * 4)   # 13
    # 2026-08-23: 48 -> 16. 48 은 _GR1_V2_MODEL_OVERRIDE 에서 온 값이고, PLAIN 의 future
    # latent 8개가 목적함수를 나눠 가질 때 action 이 묻히지 않게 하려는 것이었다
    # (_plain_model_override docstring). PLAIN-END 는 future latent 이 1개뿐이라 영상 쪽
    # 손실 지분이 ~1/8 이고, 같은 48 을 쓰면 action 이 과대가중된다.
    o["action_loss_multiplier"] = END_ACTION_LOSS_MULTIPLIER
    return o


cosmos_predict2p5_2b_tabletop_sim_5task_plain_end = _tabletop_short_post(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end",
    _TABLETOP_5TASK_PLAIN_END_DATASET,
    _CKV10_448_STATE,
    model_override=_plain_end_model_override(),
)
cosmos_predict2p5_2b_tabletop_sim_5task_plain_end__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end"
)

cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_base = _tabletop_short_post(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_base",
    _TABLETOP_5TASK_PLAIN_END_DATASET, "",
    model_override=_plain_end_model_override(),
)
cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_base__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_base"
)


# ---- TASK 전용 모델 (2026-08-24) ------------------------------------------------------
# TwinVLA 는 task 당 전용 모델을 학습한다 — train_twinvla.sh 가 data_mix 로 단일 데이터셋
# 이름을 받고, mixtures.py 에 aloha mixture 가 없어 datasets.py 가 [(name, 1.0)] 으로
# 폴백한다. 체크포인트 경로도 checkpoints/${TASK_NAME}, eval 의 --unnorm_key 도 task 이름.
# 5 task 공유 모델(우리) vs task 전용(TwinVLA) 을 같은 축에서 보려고 만든 arm.
#
# box_into_pot_easy 는 60 데모 / 약 8.6K 윈도우다. 8K iter x batch 96 = 768K 샘플이면
# ~89 epoch — 5 task 공유(20K, ~37 epoch)보다 훨씬 많이 본다. 과적합 여부도 같이 읽힌다.
_TABLETOP_POT_PLAIN_END_DATASET = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_DATASET)
_TABLETOP_POT_PLAIN_END_DATASET.task_filter = ["aloha_box_into_pot_easy"]

POT_MAX_ITER = 8_000


def _pot_end(name: str, ckpt: str) -> LazyDict:
    e = _tabletop_short_post(name, _TABLETOP_POT_PLAIN_END_DATASET, ckpt,
                             model_override=_plain_end_model_override())
    e["trainer"] = dict(max_iter=POT_MAX_ITER)
    e["scheduler"] = dict(
        cycle_lengths=[POT_MAX_ITER, 100000000000000],
        warm_up_steps=[500, 0], f_start=[1e-6, 0.5], f_max=[1.0, 0.5], f_min=[0.5, 0.5],
    )
    return e


cosmos_predict2p5_2b_tabletop_sim_pot_plain_end_base = _pot_end(
    "cosmos_predict2p5_2b_tabletop_sim_pot_plain_end_base", "")
cosmos_predict2p5_2b_tabletop_sim_pot_plain_end_base__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_pot_plain_end_base")


# ---- NATIVE 20-D variant (2026-08-24) -------------------------------------------------
# 57-D egoverse 인코딩을 벗고 Tabletop 원본 ee_6d_pos 20-D 로 학습한다:
#   [ L_xyz(3) | L_rot6d(6) | L_grip(1) | R_xyz(3) | R_rot6d(6) | R_grip(1) ]
#
# WHY. 57-D 는 egoverse warm-start 계약(head9 | Leef9 | Lhand15 | Reef9 | Rhand15)을
# 맞추려던 것이다. 그 중 11개가 상수 채널이고(head9 전체 + hand15 의 dim 24/48),
# head9 는 decode_57_to_20 에서 아예 버려진다 — 모델이 그 차원을 예측하는 데 쓰는 용량은
# env.step 에 닿지도 않는다. 실측: abs_action20 의 상수 채널은 0개.
#
# 저장소에 20-D 는 없지만 decode_57_to_20 이 정확한 역변환이라 데이터셋이 온더플라이로
# 만든다 (test_tabletop_action57_roundtrip.py 14/14). 통계는 같은 대칭 확장 규칙으로
# 새로 계산해 manifest 에 넣었다 (abs_action20_stats_pctl.json / state20_stats_pctl.json).
#
# ⚠ warm-start arm(a20 + _CKV10_448_STATE)은 주의. use_action_projection=False 라
# 액션은 latent 볼륨에 타일링되는데, 57 값과 20 값은 타일링 패턴이 다르다. 백본의
# 시각/언어 부분은 그대로 전이되지만 액션 슬롯의 사전지식은 깨진다. 그래도 재는 이유는
# 그 손실보다 상수 채널 제거 이득이 큰지가 열린 질문이기 때문이다.
_A20_KW = dict(
    native20=True,
    target_stats_path=f"{MANIFEST}/abs_action20_stats_pctl.json",
    state_stats_path=f"{MANIFEST}/state20_stats_pctl.json",
)
_TABLETOP_5TASK_PLAIN_END_A20_DATASET = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_DATASET)
for _k, _v in _A20_KW.items():
    setattr(_TABLETOP_5TASK_PLAIN_END_A20_DATASET, _k, _v)


def _plain_end_a20_model_override() -> dict:
    o = _plain_end_model_override()
    o["action_dim"] = 20
    o["proprio_dim"] = 20
    return o


cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_base = _tabletop_short_post(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_base",
    _TABLETOP_5TASK_PLAIN_END_A20_DATASET, "",
    model_override=_plain_end_a20_model_override(),
)
cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_base__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_base")

cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20 = _tabletop_short_post(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20",
    _TABLETOP_5TASK_PLAIN_END_A20_DATASET, _CKV10_448_STATE,
    model_override=_plain_end_a20_model_override(),
)
cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20")


# ---- PER-VIEW (RoboCasa b2 스타일) ----------------------------------------------------
# 2x2 그리드 한 장 대신 카메라 3대가 각자 latent 슬롯을 갖는다. PE 규약은 b2
# (bridge_layout + bridge_ret_action + bridge_ret_video=1 + bridge_future_stagger) 를
# 글자 그대로 옮겼다 — 자세한 슬롯표는 tabletop_sim_dataset._plain_build_per_view 참고.
#
#   nb = zero-pixel prefix 슬롯 수.  2 = b2 와 동일(RET_ACTION+RET_VIDEO), 1, 0 = 없음.
#   state_t = 8 + nb,  chunk_duration = 1 + 4*(7+nb),  n_cond = 4 + nb (액션 슬롯 직전).
#
# b2 와 의도적으로 다른 점 (2026-08-26, 사용자 확정):
#   * value 슬롯 없음      — Tabletop 에 value head 가 없다
#   * future proprio 없음  — 슬롯도 값도 예측하지 않는다
#   * 카메라 순서 front -> sideleft -> sideright (egoverse per-view 빌더와 동일).
#     현재 카메라 3슬롯은 role/시각을 공유하므로 PE 는 순서에 영향받지 않는다.
def _pv_model_override(nb: int) -> dict:
    o = _plain_end_a20_model_override()
    o["state_t"] = 8 + nb
    o["tokenizer"] = dict(chunk_duration=1 + 4 * (7 + nb))
    o["min_num_conditional_frames"] = 4 + nb
    o["max_num_conditional_frames"] = 4 + nb
    o["conditional_frames_probs"] = {4 + nb: 1.0}
    return o


for _nb in (0, 1, 2):
    _ds = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_A20_DATASET)
    _ds.per_view = True
    _ds.per_view_blank = _nb
    # 224. 그리드 448 은 224 타일 3장을 붙인 것이라 뷰당 해상도는 원래 224 였고 우하단
    # 1/4 은 검게 버려지고 있었다. 뷰를 쪼개면 그 낭비가 사라지므로 224 가 등가이면서
    # 픽셀이 준다: 448^2 = 200704 -> 3 x 224^2 = 150528. RoboCasa 도 224 로 돈다.
    _ds.final_image_size = 224
    for _tag, _ck in (("_base", ""), ("", _CKV10_448_STATE)):
        _name = f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv{_nb}{_tag}"
        _e = _tabletop_short_post(_name, _ds, _ck, model_override=_pv_model_override(_nb))
        # 뷰 분리는 15K 까지만 (2026-08-26 사용자 지정). save_iter 2000 은 그대로 둔다 —
        # trainer.py:259 가 max_iter 를 save_iter 로 나눈 나머지가 0 이 아니면 종료 시
        # 한 번 더 저장하므로 iter_000015000 이 남는다.
        # batch 16. RoboCasa b2 는 25 지만 그 값을 그대로 쓰면 global batch 가 600 이 되어
        # 스텝당 픽셀이 기준선(그리드 bs4)의 6.25 배가 된다. 16 이면 global 384 로 4 배다.
        #   robocasa b2 : 224^2 x 41 x 25 = 51M px/step
        #   pv0         : 224^2 x 29 x 16 = 23M px/step
        #
        # 학습 윈도우 N = 52,043, global batch = 16 x 24 = 384 이라 1 epoch = 136 step.
        # 기준선인 그리드 20K(bs4, global 96)가 36.9 epoch. 8,000 step = 59.0 epoch 으로
        # 기준선의 1.6 배를 준다 — 에폭만 맞추면 5,000 step 인데 그건 너무 짧다는 판단.
        # save_iter 1,600 으로 5 개를 남겨서 hard eval 로 어느 지점이 제일 좋은지 고른다
        # (20-D 판에서 이미 18K 가 20K 보다 나은 구간이 보였다).
        # nb=2 만 10K (2026-08-27 사용자 지정). nb=0 은 이미 8K 로 학습이 끝났으므로
        # 건드리지 않는다. save_iter 가 ITER//5 라 10K 판은 2000 단위로 저장되고,
        # 그 중 iter_000008000 이 있으므로 pv0 과 같은 스텝에서의 직접 비교도 가능하다.
        _PV_ITER, _PV_BS = (10_000 if _nb == 2 else 8_000), 16
        _e["dataloader_train"]["batch_size"] = _PV_BS
        _e["trainer"]["max_iter"] = _PV_ITER
        _e["checkpoint"]["save_iter"] = _PV_ITER // 5      # 1,600 -> 5 개
        # cycle_lengths 는 max_iter 와 같아야 코사인이 끝까지 내려간다 (기본값은 20K 라
        # 그대로 두면 3K 에서 중간에 잘린다). warmup 도 비율을 유지해 500 x 3/20 = 75.
        # LR 은 1e-4 유지 (기준선과 같게 — batch 를 4 배 올렸지만 스케일링하지 않는다).
        _e["scheduler"]["cycle_lengths"] = [_PV_ITER, 100000000000000]
        _e["scheduler"]["warm_up_steps"] = [_PV_ITER // 40, 0]   # 500 x ITER/20K, 비율 유지
        globals()[_name] = _e
        globals()[f"{_name}__inference_only"] = _inference_only(_name)


# ---- 다른 egoverse mid 에서 warm-start (2026-08-28) ----------------------------------
# pv0 (per-view nb=0) 레이아웃은 그대로 두고 출발 체크포인트만 바꾼다. pv0 warm 이 EASY
# 76.0% 로 지금까지 최고였으므로, 그 판에서 mid 만 교체해 "어느 egoverse mid 가 Tabletop
# 으로 제일 잘 전이되는가" 를 가른다. 예산도 pv0 warm 과 똑같이 맞췄다 (8K, bs 16).
#
#   기존   _CKV10_448_STATE : ..._egoverse_ret_v3_448_state            @8K
#   blankbc                 : ..._egoverse_ret_v3_448_state_blankret_bv2_v10     @8K
#   cotrain                 : ..._egoverse_ret_v3_448_state_blankret_bv2_cotrain @8K
#
# blankbc 와 cotrain 은 레시피가 같고 (blank_retrieval=True, blank_video_slots=2,
# state_t=29, n_cond=5) manifest 만 다르다 — cotrain 쪽이 eva-robot + HE 를 쿼리로
# 추가한 human+robot 공동학습이다 (egoverse_ret_experiment_configs.py:571-575).
# 즉 이 둘의 차이는 "로봇 데이터를 mid 에 섞는 것이 Tabletop 전이에 도움이 되는가" 다.
#
# 셋 다 net.role_embedder(8x2048) 8행이 전부 학습돼 있어 (2026-08-28 확인) role /
# 연속시간 PE 가 0 이 아닌 상태에서 전이된다 — 갈아끼워도 계약은 그대로다.
_CK_BLANKBC = (
    "${oc.env:TABLETOP_BLANKBC_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_blankret_bv2_v10_4node/"
    "cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_v10/"
    "checkpoints/model_000008000.pt}"
)
_CK_COTRAIN = (
    "${oc.env:TABLETOP_COTRAIN_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_blankret_bv2_cotrain_4node/"
    "cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_blankret_bv2_cotrain/"
    "checkpoints/model_000008000.pt}"
)

# nb=0 과 nb=2 둘 다 만든다. 같은 mid·같은 8K 예산에서 prefix 유무만 갈리므로
# pv0_{tag} 대 pv2_{tag} 가 곧 "zero-pixel prefix 2개가 값을 하는가" 의 대조군이다.
# (기존 pv2 warm 은 10K 였어서 pv0 warm 8K 와 예산이 어긋나 있었다 — 여기서는 맞춘다.)
for _nb2, (_tag, _ck) in [(n, t) for n in (0, 2)
                          for t in (("blankbc", _CK_BLANKBC), ("cotrain", _CK_COTRAIN))]:
    _ds = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_A20_DATASET)
    _ds.per_view = True
    _ds.per_view_blank = _nb2
    _ds.final_image_size = 224
    _name = f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv{_nb2}_{_tag}"
    _e = _tabletop_short_post(_name, _ds, _ck, model_override=_pv_model_override(_nb2))
    _MID_ITER = 8_000                                  # pv0 warm(76.0%) 과 동일 예산
    _e["dataloader_train"]["batch_size"] = 16
    _e["trainer"]["max_iter"] = _MID_ITER
    _e["checkpoint"]["save_iter"] = _MID_ITER // 5
    _e["scheduler"]["cycle_lengths"] = [_MID_ITER, 100000000000000]
    _e["scheduler"]["warm_up_steps"] = [_MID_ITER // 40, 0]
    globals()[_name] = _e
    globals()[f"{_name}__inference_only"] = _inference_only(_name)


# ---- DATA-SCALING 사다리의 mid 를 pv0 로 내려보기 (2026-08-31) ----------------------
# pv0 warm(EASY 76.0%, 지금까지 최고) 판을 그대로 두고 출발 mid 만 데이터 스케일링 사다리의
# q25 / q50 / q75 로 바꾼다. 100% 지점은 기존 `..._a20_pv0` (= _CKV10_448_STATE, 이미 학습
# 완료 /mnt/ddn/tmp/tabletop_pv0/.../model_000008000.pt) 이므로 새로 돌리지 않는다.
# 위 blankbc/cotrain 블록과 완전히 같은 형태 — 레이아웃·예산 고정, mid 만 변수.
#
# 묻는 것: "mid 를 X% 데이터로만 학습했다면 Tabletop 전이가 얼마나 나빠지는가."
# mid 쪽 설계 근거(에피소드 단위 서브샘플, LR horizon 스케일)는 docs/egoverse_data_scaling.md.
#
# ⚠️ a20 은 액션을 20-D 로 타일링하므로 57-D mid 의 액션 슬롯 사전지식이 깨진다
#    (_A20_KW 주석 참고). 여기서는 문제가 되지 않는다 — 네 arm 이 **똑같은** 57->20 핸디캡을
#    같은 레이아웃 위에서 지므로 비교는 mid 대 mid 로 공정하고, blankbc/cotrain 도 이미
#    같은 판에서 mid 를 갈랐다.
#
# ⚠️ 이 체크포인트들은 mid 학습이 끝나야 존재한다. iter 수가 arm 마다 다르다(2K/4K/6K)는
#    점에 주의 — DCP -> .pt 변환 시 --iter 를 맞출 것:
#      IMAGINAIRE_OUTPUT_ROOT=/mnt/ddn/tmp/egoverse_ret_v3_448_state_q25_4node \
#        python convert_dcp_to_pt.py --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q25 \
#        --iter 2000
_SCALING_MID_ITER = {25: 2_000, 50: 4_000, 75: 6_000}


def _scaling_mid_ckpt(pct: int) -> str:
    """사다리 arm q<pct> 의 mid 체크포인트 (env TABLETOP_Q<pct>_CKPT 로 덮어쓸 수 있다)."""
    tag = f"cosmos_predict2p5_2b_egoverse_ret_v3_448_state_q{pct}"
    return (f"${{oc.env:TABLETOP_Q{pct}_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_q{pct}_4node/"
            f"cosmos_policy/cosmos_v2_finetune/{tag}/"
            f"checkpoints/model_{_SCALING_MID_ITER[pct]:09d}.pt}}")


for _pct in (25, 50, 75):
    _ds = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_A20_DATASET)
    _ds.per_view = True
    _ds.per_view_blank = 0
    _ds.final_image_size = 224
    _name = f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_q{_pct}"
    _e = _tabletop_short_post(_name, _ds, _scaling_mid_ckpt(_pct),
                              model_override=_pv_model_override(0))
    _Q_ITER = 8_000                                    # pv0 warm(76.0%) 과 동일 예산
    _e["dataloader_train"]["batch_size"] = 16
    _e["trainer"]["max_iter"] = _Q_ITER
    _e["checkpoint"]["save_iter"] = _Q_ITER // 5
    _e["scheduler"]["cycle_lengths"] = [_Q_ITER, 100000000000000]
    _e["scheduler"]["warm_up_steps"] = [_Q_ITER // 40, 0]
    globals()[_name] = _e
    globals()[f"{_name}__inference_only"] = _inference_only(_name)


# ---- SAME-EMB TOP-1 mid 를 pv0 로 내려보기 (2026-09-10) -----------------------------
# 위 blankbc/cotrain/q25-75 블록과 완전히 같은 형태 — pv0 레이아웃·예산(8K, bs 16) 고정,
# 출발 mid 만 same-embodiment top-1 retrieval mid 로 바꾼다.
#   기존 _CKV10_448_STATE : ..._egoverse_ret_v3_448_state                  @8K (cross-emb top1)
#   여기                  : ..._egoverse_ret_v3_448_state_sameemb_top1_v10 @8K (same-emb top1)
# 두 mid 는 데이터·레이아웃·스케줄이 같고 데모 선택 규칙만 다르므로 (레퍼런스를 쿼리와 같은
# 임베디먼트의 top-1 로 고정), 이 arm 은 "레퍼런스를 같은 임베디먼트로 강제한 mid 가
# Tabletop 으로 더 잘 전이되는가" 만 묻는다. 설계: docs/egoverse_same_emb_top1_mid.md.
#
# ⚠️ mid 학습이 8000 iter 을 찍어야 존재한다. DCP -> .pt 변환 먼저:
#      IMAGINAIRE_OUTPUT_ROOT=/mnt/ddn/tmp/egoverse_ret_v3_448_state_sameemb_top1_4node \
#        python convert_dcp_to_pt.py \
#        --tag cosmos_predict2p5_2b_egoverse_ret_v3_448_state_sameemb_top1_v10 --iter 8000
_CK_SEMID = (
    "${oc.env:TABLETOP_SEMID_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_sameemb_top1_4node/"
    "cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_sameemb_top1_v10/"
    "checkpoints/model_000008000.pt}"
)

_ds = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_A20_DATASET)
_ds.per_view = True
_ds.per_view_blank = 0
_ds.final_image_size = 224
_name = "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_semid"
_e = _tabletop_short_post(_name, _ds, _CK_SEMID, model_override=_pv_model_override(0))
_SEMID_ITER = 8_000                                # pv0 warm(76.0%) 과 동일 예산
_e["dataloader_train"]["batch_size"] = 16
_e["trainer"]["max_iter"] = _SEMID_ITER
_e["checkpoint"]["save_iter"] = _SEMID_ITER // 5
_e["scheduler"]["cycle_lengths"] = [_SEMID_ITER, 100000000000000]
_e["scheduler"]["warm_up_steps"] = [_SEMID_ITER // 40, 0]
globals()[_name] = _e
globals()[f"{_name}__inference_only"] = _inference_only(_name)


# ---- 단일 모달 mid (action->action / video->video) 를 pv0 로 내려보기 (2026-09-16) ----
# 위 blankbc / cotrain / q25-75 / semid 블록과 완전히 같은 형태 — pv0 레이아웃과 예산
# (8K, bs 16) 을 고정하고 출발 mid 만 "한 모달만 학습한" egoverse mid 로 바꾼다.
#
#   vidonly : ..._egoverse_ret_v3_448_state_v10_vidonly      @8K
#             ret_video -> future_video 만. EDM 안에 RET_ACTION / ACTION 슬롯이 아예 없다
#             (video_only=True, state_t=48, chunk 189).
#   actonly : ..._egoverse_ret_v3_448_state_v10_actonly_bv2  @8K
#             ret_action -> action 만. future_video 목적항이 없고 RET_VIDEO 는 blank 2슬롯
#             (blank_ret_video=True, blank_video_slots=2, no_future=True, state_t=6, chunk 21).
#
# 둘 다 _CKV10_448_STATE (= ..._ret_v3_448_state, 두 모달을 같이 학습한 transfer mid) 와
# 데이터(_DATASET_KW_V10), 스케줄, 8K 예산이 같고 EDM 의 목적항 조합만 다르다. 그래서 이
# 두 arm 이 묻는 것은 하나다 — "mid 에서 video 목적항만 / action 목적항만 학습해도
# Tabletop 으로 전이되는가, 둘을 같이 학습한 mid 대비 얼마나 손해인가."
# 대조군은 기존 pv0 warm (/mnt/ddn/tmp/tabletop_pv0, EASY 76.0%) 이라 다시 돌리지 않는다.
# mid 쪽 설계 근거: egoverse_ret_experiment_configs.py:627-695.
#
# 2026-09-16 확인 (role/PE 계약): 두 mid 모두 net.role_embedder(8x2048) 8행이 전부 nonzero다.
#     rownorm  vidonly 0.252 0.748 0.751 0.798 0.782 0.963 0.630 0.589
#              actonly 0.183 0.759 0.748 0.791 0.782 0.965 0.630 0.589
#              ref v10 0.233 0.754 0.759 0.810 0.789 0.970 0.630 0.589
# 즉 vidonly 는 ACTION(role 4) 슬롯을, actonly 는 FUTURE(role 5) 슬롯을 자기 mid 학습에서
# 한 번도 보지 않았는데도 그 행이 0 이 아니다 (ref mid 와 공통 조상에서 물려받은 값이며,
# 아무 arm 에서도 안 쓰이는 role 6/7 행은 셋이 byte 단위로 같다). 그래서 warm-start 계약은
# 다른 pv0 arm 과 동일하게 성립하고 missing_keys=[] 로 붙는다. 다만 "그 슬롯의 사전지식이
# 얕다" 는 점은 결과 해석에 반드시 들어가야 한다 — 특히 pv0 판의 GEN 슬롯은 ACTION 과
# FUTURE 둘 다이므로 두 arm 모두 자기가 안 배운 GEN 슬롯을 하나씩 안고 출발한다.
_CK_VIDONLY = (
    "${oc.env:TABLETOP_VIDONLY_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_v10_vidonly_4node/"
    "cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v10_vidonly/"
    "checkpoints/model_000008000.pt}"
)
_CK_ACTONLY = (
    "${oc.env:TABLETOP_ACTONLY_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_v10_actonly_bv2_4node/"
    "cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_v10_actonly_bv2/"
    "checkpoints/model_000008000.pt}"
)

for _tag, _ck in (("vidonly", _CK_VIDONLY), ("actonly", _CK_ACTONLY)):
    _ds = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_A20_DATASET)
    _ds.per_view = True
    _ds.per_view_blank = 0
    _ds.final_image_size = 224
    _name = f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_{_tag}"
    _e = _tabletop_short_post(_name, _ds, _ck, model_override=_pv_model_override(0))
    _MODAL_ITER = 8_000                                # pv0 warm(76.0%) 과 동일 예산
    _e["dataloader_train"]["batch_size"] = 16
    _e["trainer"]["max_iter"] = _MODAL_ITER
    _e["checkpoint"]["save_iter"] = _MODAL_ITER // 5
    _e["scheduler"]["cycle_lengths"] = [_MODAL_ITER, 100000000000000]
    _e["scheduler"]["warm_up_steps"] = [_MODAL_ITER // 40, 0]
    globals()[_name] = _e
    globals()[f"{_name}__inference_only"] = _inference_only(_name)


# ---- ACTION-ONLY POST-TRAIN x 출발 mid 3종 (2026-09-16) ----------------------------
# 지금까지의 pv0 arm 은 전부 post-train 에서 ACTION 과 FUTURE VIDEO 를 같이 예측했다.
# 여기서는 **post-train 의 목적항에서 future video 를 빼고 ACTION 만 예측**하게 하고,
# 출발 mid 를 3종으로 갈라 "mid 에서 무엇을 배웠는지가 action-only post-train 에
# 얼마나 남는가" 를 본다.
#
#   ap_base    : mid 없음 (checkpoint={}, HF Cosmos-Predict2.5-2B init)
#   ap_actonly : ..._egoverse_ret_v3_448_state_v10_actonly_bv2 @8K  (action->action 만)
#   ap_mid     : ..._egoverse_ret_v3_448_state                 @8K  (video+action 둘 다)
#
# ===== 구현: FUTURE 슬롯을 시퀀스에서 통째로 뺀다 ===================================
# 손실만 끄는 게 아니라 latent 자체가 없다 (2026-09-16 사용자 지정). mid 의
# EgoVerseRetV3StateDataset(no_future=True) 와 같은 의미다.
#
#     idx role         t     내용
#      0  CUR_STATE    NOW   blank 1프레임, proprio 가 안에 실림
#      1  CUR_IMAGE    NOW   front_1    (t)
#      2  CUR_IMAGE    NOW   sideleft   (t)
#      3  CUR_IMAGE    NOW   sideright  (t)
#     -- num_conditional_frames = 4 --
#      4  ACTION       NOW   GEN  <- 유일한 GEN 슬롯
#     state_t 8 -> 5,  chunk_duration 29 -> 17,  n_cond 4 (그대로)
#
# 데이터셋: TabletopSimPlainEndImageDataset(per_view=True, no_future=True) — future
# 프레임을 디코드조차 하지 않으므로 스텝이 pv0 보다 싸다.
# eval 서버: 체크포인트 config 의 state_t 가 5+nb 면 자동으로 같은 레이아웃을 만든다
# (tabletop_sim_eval_server.py, 이름 규칙이 아니라 state_t 로 판정).
#
# mask_loss_for_action_future_state_prediction = True (2026-09-18 확정).
#
# 이 플래그는 demo 샘플의 final_mask 를 ACTION 한 칸만 1 로 만든다
# (policy_video2world_model_rectified_flow.py:385-397). latent 별 손실 가중치:
#     True  : [0, 0, 0, 0, 16]          조건화 latent 는 손실에서 빠진다
#     False : [1, 1, 1, 1, 16]          조건화 프레임 재구성 항이 추가된다
#     pv0   : [1, 1, 1, 1, 16, 1, 1, 1] (CUR_STATE, CUR_IMG x3, ACTION, FUTURE x3)
#
# 2026-09-17 에 "False 가 pv0 과 조건화까지 같으니 더 공정하다" 고 보고 False 로 재학습
# 했는데, 50-trial eval 에서 **False 가 일관되게 더 나빴다** (ap_base 19.5% vs 25.0%,
# 8 task 중 6승 2무 0패). ACTION 슬롯의 gradient 는 두 경우 같고 False 는 "입력을 다시
# 그려내는" 보조 항만 더해지는데, 그게 도움이 안 되고 용량만 잡아먹은 것으로 보인다
# (pv0 의 추가 항은 FUTURE = 진짜 예측 과제라 성격이 다르다).
# 그래서 True 로 되돌린다. mask=False 판의 체크포인트/eval 은
# *_maskfalse_* 로 보존돼 있다.
#
# ⚠ 어느 쪽이든 결론은 같다: 두 판 다 19~25% 이고 pv0 는 64.6% 다. ap 가 무너지는
#   주 원인은 FUTURE 목적항 부재이고 이 플래그는 5pp 짜리 2차 변수다.
#
# action_loss_multiplier 는 END_ACTION_LOSS_MULTIPLIER(16) 그대로 둔다. ACTION 슬롯의
# gradient 스케일을 pv0 와 같게 유지하려는 것이다 (바꾸면 그것부터 교란 변수가 된다).
#
# ⚠ pv0 계열과 시퀀스 길이가 달라졌으므로 이 3 arm 은 **자기들끼리** 비교한다. pv0 warm
#   76.0 / vidonly / actonly 와 나란히 놓을 때는 "목적항 + 시퀀스 길이" 두 가지가 같이
#   다르다는 점을 반드시 명시할 것.
_AP_ARMS = (
    ("base",    ""),                    # mid 없음
    ("actonly", _CK_ACTONLY),           # action->action mid
    ("mid",     _CKV10_448_STATE),      # video+action mid (pv0 warm 과 같은 mid)
)


_AP_STATE_T = 5                                    # cur_state + cam3 + action (future 없음)


def _pv0_actpred_override() -> dict:
    o = _pv_model_override(0)                      # n_cond=4 등 pv0 값을 물려받는다
    o["state_t"] = _AP_STATE_T
    o["tokenizer"] = dict(chunk_duration=1 + 4 * (_AP_STATE_T - 1))   # 17
    o["mask_loss_for_action_future_state_prediction"] = True
    return o


for _aptag, _apck in _AP_ARMS:
    _ds = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_A20_DATASET)
    _ds.per_view = True
    _ds.per_view_blank = 0
    _ds.no_future = True
    _ds.final_image_size = 224
    _name = f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_ap_{_aptag}"
    _e = _tabletop_short_post(_name, _ds, _apck, model_override=_pv0_actpred_override())
    _AP_ITER = 8_000                                   # pv0 warm(76.0%) 과 동일 예산
    _e["dataloader_train"]["batch_size"] = 16
    _e["trainer"]["max_iter"] = _AP_ITER
    _e["checkpoint"]["save_iter"] = _AP_ITER // 5
    _e["scheduler"]["cycle_lengths"] = [_AP_ITER, 100000000000000]
    _e["scheduler"]["warm_up_steps"] = [_AP_ITER // 40, 0]
    globals()[_name] = _e
    globals()[f"{_name}__inference_only"] = _inference_only(_name)


# ---- NOFP 변형 (2026-08-27) ----------------------------------------------------------
# RoboCasa 의 stock 레이아웃에서 future proprio 를 뺀 arm
# (cosmos_predict2p5_2b_480p_robocasa_all_24_tasks_t5_nofp) 을 Tabletop 으로 그대로 옮긴다.
# 여기에 사용자 지정으로 value 슬롯까지 뺀다.
#
#     0 blank | 1 proprio | 2 front_1 | 3 sideleft | 4 sideright
#     -- num_conditional_frames = 5 --
#     5 action | 6 front_1(t+32) | 7 sideleft(t+32) | 8 sideright(t+32)
#     state_t = 9, chunk_duration = 33
#
# per-view(pv) arm 과 슬롯 인벤토리는 같지만 두 가지가 다르다:
#   * proprio 가 latent 0 에 얹히지 않고 자기 슬롯(1)을 갖는다 — RoboCasa stock 방식
#   * slot_roles / t_positions 를 아예 내보내지 않는다. stock RoboCasa 는 bridge_layout=False
#     라 bridge_pe 가 빈 dict 이고 모델의 기본 PE 를 쓴다 ("PE 기본").
# 즉 pv 판이 egoverse mid 계약(role/rope)을 얹은 것이라면 이 arm 은 RoboCasa 가 63.17% 를
# 낸 그 계약 그대로다. base(HF init) 만 만든다 — warm ckpt 는 role/rope 계약으로 학습돼
# 있어서 PE 기본 판에 얹을 근거가 없다.
def _nofp_model_override() -> dict:
    o = _plain_end_a20_model_override()
    o["state_t"] = 9
    o["tokenizer"] = dict(chunk_duration=33)          # 1 + 4*8
    o["min_num_conditional_frames"] = 5
    o["max_num_conditional_frames"] = 5
    o["conditional_frames_probs"] = {5: 1.0}
    return o


_ds = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_A20_DATASET)
_ds.nofp = True
_ds.final_image_size = 224                            # RoboCasa 도 224
_name = "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_nofp_base"
_e = _tabletop_short_post(_name, _ds, "", model_override=_nofp_model_override())
_NOFP_ITER, _NOFP_BS = 10_000, 16                     # pv2 와 동일 예산 (직접 비교용)
_e["dataloader_train"]["batch_size"] = _NOFP_BS
_e["trainer"]["max_iter"] = _NOFP_ITER
_e["checkpoint"]["save_iter"] = _NOFP_ITER // 5
_e["scheduler"]["cycle_lengths"] = [_NOFP_ITER, 100000000000000]
_e["scheduler"]["warm_up_steps"] = [_NOFP_ITER // 40, 0]
globals()[_name] = _e
globals()[f"{_name}__inference_only"] = _inference_only(_name)


# ---- BRIDGE 변형 (2026-08-23) --------------------------------------------------------
# PLAIN-END 에 zero-pixel RET 슬롯 2개를 넣어 egoverse mid 계약(k=1)을 맞춘다.
#
#   0 cur_state | 1 ret_action | 2 ret_video(t=NOW-dt) | 3 cur_image | 4 action | 5 future
#   n_cond = k+3 = 4        state_t = 6        chunk_duration = 21
#
# WHY. warm-start 체크포인트(_CKV10_448_STATE = egoverse_ret_v3_448_state 8K)는 RET 슬롯을
# 갖고 학습된 모델이다. RET 이 없는 PLAIN 레이아웃에 그대로 로드하면 백본이 한 번도 본 적 없는
# 구조를 받는다 — RoboCasa 가 bridge_ret_action / bridge_ret_video 를 넣어 피하려던 상황과
# 같다 (cosmos_policy_experiment_configs.py 의 ..._t5_bridge_b2).
# 슬롯 픽셀은 0 이고 역할(role)과 rope 시각만 남는다.
#
# 주의: RoboCasa 에서 bridge + warm start 조합은 base 를 한 번도 못 넘었다
# (base 63.17 vs b2 61.67 / b2cot 59.75 / b2p 59.50 / b2bbc 56.25). 기대값을 높게 잡지 말 것.
_TABLETOP_5TASK_PLAIN_END_BRIDGE_DATASET = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_DATASET)
_TABLETOP_5TASK_PLAIN_END_BRIDGE_DATASET.bridge_ret = True

BRIDGE_END_STATE_T = END_STATE_T + 2       # 4 -> 6


def _plain_end_bridge_model_override() -> dict:
    o = _plain_end_model_override()
    o["state_t"] = BRIDGE_END_STATE_T
    o["tokenizer"] = dict(chunk_duration=1 + (BRIDGE_END_STATE_T - 1) * 4)   # 21
    # 조건화는 action 슬롯 직전까지 = cur_state + ret_action + ret_video + cur_image = 4.
    # RoboCasa b2 가 이 경계를 stock 값 그대로 물려받아 카메라 하나를 GEN 으로 넘겼다가
    # future_image 가 0.33 vs 0.097 로 벌어졌던 곳이다 — 반드시 같이 올려야 한다.
    o["min_num_conditional_frames"] = 4
    o["max_num_conditional_frames"] = 4
    o["conditional_frames_probs"] = {4: 1.0}
    return o


cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge = _tabletop_short_post(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge",
    _TABLETOP_5TASK_PLAIN_END_BRIDGE_DATASET,
    _CKV10_448_STATE,
    model_override=_plain_end_bridge_model_override(),
)
cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge"
)
cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge_base = _tabletop_short_post(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge_base",
    _TABLETOP_5TASK_PLAIN_END_BRIDGE_DATASET, "",
    model_override=_plain_end_bridge_model_override(),
)
cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge_base__inference_only = _inference_only(
    "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge_base"
)


# ---- RANDREF mid 를 pv0 로 내려보기, 2 노드 (2026-11-06) -------------------------------
# pv0 (per-view nb=0, 224, a20, plain-end) 레이아웃/타깃/손실은 그대로, 출발 mid 만
# random same-task reference mid (..._egoverse_ret_v3_448_state_randref_nomd_15k @15K) 로 바꾼다.
# 기준 pv0 는 3 노드 x 8 GPU x bs 16 = global 384 로 8K (3.07M 샘플). 2 노드는 global 256 이라
# 같은 샘플 수 기준으로 스케줄을 384/256 배 늘린다 (cycle 8K -> 12K). 학습도 12K 까지 돈다
# (= 3 노드 기준 8K, 3.07M 샘플), 기준 pv0 와 같은 학습량.
# warmup 200 과 peak LR 1e-4 는 고정.
# save_iter 2,400 (5 개, 12K 포함).
# 경로: TABLETOP_RANDREF_DATA (= <tabletopsim-watch>/data), TABLETOP_RANDREF_MANIFEST
# (= <tabletopsim-watch>/manifest), mid 는 TABLETOP_RANDREF_CKPT.
import os as _os

_RR_TT_DATA = _os.environ.get("TABLETOP_RANDREF_DATA", TABLETOP_ROOT)
_RR_TT_MANIFEST = _os.environ.get("TABLETOP_RANDREF_MANIFEST", MANIFEST)
_RR_PV_CYCLE = 12_000  # 기준 8K 스케줄 x (384 / 256)
_RR_PV_ITER = 12_000   # = 3 노드 기준 8K (3.07M 샘플), 기준 pv0 와 같은 학습량
_CK_RANDREF_MID = (
    "${oc.env:TABLETOP_RANDREF_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_randref_2node/"
    "cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_randref_nomd_15k/"
    "checkpoints/model_000015000.pt}"
)

_ds = copy.deepcopy(_TABLETOP_5TASK_PLAIN_END_A20_DATASET)
_ds.per_view = True
_ds.per_view_blank = 0
_ds.final_image_size = 224
_ds.dataset_root = _RR_TT_DATA
_ds.manifest_dir = _RR_TT_MANIFEST
for _k in ("t5_text_embeddings_path", "action_stats_path", "state_stats_path", "target_stats_path"):
    setattr(_ds, _k, getattr(_ds, _k).replace(MANIFEST, _RR_TT_MANIFEST, 1))
_name = "cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_randref_2node"
_e = _tabletop_short_post(_name, _ds, _CK_RANDREF_MID, model_override=_pv_model_override(0))
_e["dataloader_train"]["batch_size"] = 16
_e["trainer"]["max_iter"] = _RR_PV_ITER
_e["checkpoint"]["save_iter"] = _RR_PV_ITER // 5
_e["scheduler"]["cycle_lengths"] = [_RR_PV_CYCLE, 100000000000000]
_e["scheduler"]["warm_up_steps"] = [200, 0]          # pv0 의 8_000 // 40, 스텝 수 고정
globals()[_name] = _e
globals()[f"{_name}__inference_only"] = _inference_only(_name)

# ---- noisy-label (conf50) mids 를 pv0 로, 2 노드: 위 randref_2node 와 같은 레시피, 출발 mid 와 이름만 다르다.
#   _conf50_2node     <- ..._egoverse_ret_v3_448_state_conf50_nomd_15k @15K
#   _conf50_q50_2node <- ..._egoverse_ret_v3_448_state_conf50_q50_nomd_8k @8K
#   _randref_q50_2node <- ..._egoverse_ret_v3_448_state_randref_q50_nomd_8k @8K
# mid 경로는 TABLETOP_RANDREF_CKPT 로 덮어쓴다 (randref 와 같은 env 이름).
for _tag, _mid, _it in (("conf50", "conf50_nomd_15k", 15_000), ("conf50_q50", "conf50_q50_nomd_8k", 8_000),
                        ("randref_q50", "randref_q50_nomd_8k", 8_000)):
    _name = f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_{_tag}_2node"
    _ck = (f"${{oc.env:TABLETOP_RANDREF_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_{_tag}_2node/"
           f"cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_{_mid}/"
           f"checkpoints/model_{_it:09d}.pt}}")
    _e = _tabletop_short_post(_name, copy.deepcopy(_ds), _ck, model_override=_pv_model_override(0))
    _e["dataloader_train"]["batch_size"] = 16
    _e["trainer"]["max_iter"] = _RR_PV_ITER
    _e["checkpoint"]["save_iter"] = _RR_PV_ITER // 5
    _e["scheduler"]["cycle_lengths"] = [_RR_PV_CYCLE, 100000000000000]
    _e["scheduler"]["warm_up_steps"] = [200, 0]
    globals()[_name] = _e
    globals()[f"{_name}__inference_only"] = _inference_only(_name)


# ---- 4-node pv0 from the two 4-node rebuttal mids (..._{randref,conf50}_nomd_4node @ 8K) ----
# 4 x 8 GPUs x batch 12 = global 384 = the pv0 reference (3 x 8 x 16): reference schedule unchanged
# (8K iters, cycle 8K, warmup 200, save every 1,600). TABLETOP_RANDREF_CKPT overrides the warm-start path.
# rematch / hawor_pairs / hawor_full: the RGB-only-hand mids (docs/egoverse_hawor_rgb.md), same recipe.
_PV0_4NODE_TAGS = ("randref", "conf50", "rematch", "hawor_pairs", "hawor_full")
for _tag in _PV0_4NODE_TAGS:
    _name = f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_{_tag}_4node"
    _ck = (f"${{oc.env:TABLETOP_RANDREF_CKPT,/mnt/ddn/tmp/egoverse_ret_v3_448_state_{_tag}_4node/"
           f"cosmos_policy/cosmos_v2_finetune/cosmos_predict2p5_2b_egoverse_ret_v3_448_state_{_tag}_nomd_4node/"
           f"checkpoints/model_000008000.pt}}")
    _e = _tabletop_short_post(_name, copy.deepcopy(_ds), _ck, model_override=_pv_model_override(0))
    _e["dataloader_train"]["batch_size"] = 12
    _e["trainer"]["max_iter"] = 8_000
    _e["checkpoint"]["save_iter"] = 1_600
    _e["scheduler"]["cycle_lengths"] = [8_000, 100000000000000]
    _e["scheduler"]["warm_up_steps"] = [200, 0]
    globals()[_name] = _e
    globals()[f"{_name}__inference_only"] = _inference_only(_name)


ALL_TABLETOP_SIM_CONFIGS = [
    cosmos_predict2p5_2b_tabletop_sim_5task_plain_end,
    cosmos_predict2p5_2b_tabletop_sim_5task_plain_end__inference_only,
    cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_base,
    cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_base__inference_only,
        cosmos_predict2p5_2b_tabletop_sim_pot_plain_end_base,
        cosmos_predict2p5_2b_tabletop_sim_pot_plain_end_base__inference_only,
        cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_base,
        cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_base__inference_only,
        cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20,
        cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20__inference_only,
        *[globals()[f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv{n}{t}{i}"]
          for n in (0, 1, 2) for t in ("_base", "") for i in ("", "__inference_only")],
        cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_nofp_base,
        cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_nofp_base__inference_only,
        *[globals()[f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv{n}_{t}{i}"]
          for n in (0, 2) for t in ("blankbc", "cotrain") for i in ("", "__inference_only")],
        *[globals()[f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_q{q}{i}"]
          for q in (25, 50, 75) for i in ("", "__inference_only")],
        *[globals()[f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_semid{i}"]
          for i in ("", "__inference_only")],
        *[globals()[f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_{t}_2node{i}"]
          for t in ("randref", "randref_q50", "conf50", "conf50_q50") for i in ("", "__inference_only")],
        *[globals()[f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_{t}_4node{i}"]
          for t in _PV0_4NODE_TAGS for i in ("", "__inference_only")],
        *[globals()[f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_{t}{i}"]
          for t in ("vidonly", "actonly") for i in ("", "__inference_only")],
        *[globals()[f"cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_a20_pv0_ap_{t}{i}"]
          for t in ("base", "actonly", "mid") for i in ("", "__inference_only")],
        cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge,
        cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge__inference_only,
        cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge_base,
        cosmos_predict2p5_2b_tabletop_sim_5task_plain_end_bridge_base__inference_only,
    cosmos_predict2p5_2b_tabletop_sim_5task_plain_state,
    cosmos_predict2p5_2b_tabletop_sim_5task_plain_state__inference_only,
    cosmos_predict2p5_2b_tabletop_sim_5task_plain_state_base,
    cosmos_predict2p5_2b_tabletop_sim_5task_plain_state_base__inference_only,
    cosmos_predict2p5_2b_tabletop_sim_5task_short_state,
    cosmos_predict2p5_2b_tabletop_sim_5task_short_state__inference_only,
    cosmos_predict2p5_2b_tabletop_sim_5task_short_state_base,
    cosmos_predict2p5_2b_tabletop_sim_5task_short_state_base__inference_only,
]
