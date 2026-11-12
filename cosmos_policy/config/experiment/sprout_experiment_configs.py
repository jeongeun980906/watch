# SPDX-License-Identifier: Apache-2.0
"""Sprout real-robot post-train, Tabletop pv0 style: per-view, no zero-pixel prefix, native 29-D, 15 Hz, ego view only.

Data: ``WAM/dataset/sprout_ego29_3task`` (``to_sprout29.py``), 60 episodes, 10,358 windows::

    sprout_cup_to_drawer     20  "Pick up the cup and put it in the drawer"   right arm (left at the reset pose)
    sprout_open_drawer       20  "Open the drawer"                            left arm (right at the reset pose)
    sprout_cup_close_drawer  20  "Pick up the cup and close drawer"           both arms

Layout (``sprout_dataset.py``)::

    0 cur_state | 1 cur_image (ego) | 2 action (GEN) | 3 future ego image t+C (GEN)
    state_t = 4, chunk_duration = 13, num_conditional_frames = 2

Two chunk sizes C: 32 frames (2.13 s at 15 Hz; the frame count of the mid and of Tabletop, whose 32 are 1.6 s at
20 Hz) and 16 (1.07 s). The future image is the chunk's last frame, t+C.

Recipe = ``..._tabletop_sim_5task_plain_end_a20_pv0`` (lr 1e-4, batch 16 per GPU, one cosine cycle, warmup 1/40),
but 4K iterations instead of 8K (60 episodes against Tabletop's 260) and a checkpoint every 1K. pv0 saw ~59 epochs of its 52K
windows (global batch 384); here, on 2 nodes x 8 GPUs x batch 8 = global 128 (scripts/train_sprout_2node.sh),
one epoch is 81 iterations and 2K / 3K / 4K are 25 / 37 / 49 epochs.
``SPROUT_ITER`` changes it. The action is 29 values tiled into the latent volume, as the a20 arms tile 20.

    ..._sprout_3task_pv0_c{32,16}          warm start from the WATCH mid (egoverse_ret_v3_448_state @ 8K)   = ours
    ..._sprout_3task_pv0_c{32,16}_within   from the within-video mid (..._state_blankret_bv2_v10 @ 8K): the same
                                           recipe with a blank reference, i.e. no cross-video pairing
    ..._sprout_3task_pv0_c{32,16}_base     no mid (Cosmos-Predict2.5 base)

Launch with ``sprout_train.sh`` (T5 embeddings once with ``sprout_train.sh t5``).
"""
from __future__ import annotations

import copy
import os

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy.config.experiment.tabletop_sim_experiment_configs import (
    _inference_only,
    _plain_end_model_override,
    _tabletop_short_post,
)
from cosmos_policy.datasets.sprout_dataset import DIM29, SPROUT_ROOT, SproutIclDataset, SproutPlainEndDataset

MANIFEST = f"{SPROUT_ROOT}/manifest"
ITER = int(os.environ.get("SPROUT_ITER", 4_000))
SAVE_ITER = int(os.environ.get("SPROUT_SAVE_ITER", 1_000))   # 1K, 2K, 3K, 4K: read on the robot to pick one
BATCH = 16
NB = 0          # pv0: no zero-pixel prefix
N_CAMS = 1      # ego view
STATE_T = 2 + NB + 2 * N_CAMS

# WATCH mid; the same file as Tabletop's _CKV10_448_STATE, downloaded from HF cosmos-policy-egoverse-mid-v10
_CK_WATCH_MID = (
    "${oc.env:SPROUT_MID_CKPT," + os.path.expanduser("~/WAM/WATCH/ckpt/watch-mid-v10/egoverse_ret_v3_448_state_4node/")
    + "model_000008000.pt}"
)

# within-video baseline mid (Tabletop's _CK_BLANKBC), HF Jeongeun/cosmos-policy-egoverse-mid-v10 (private)
_CK_WITHIN_MID = (
    "${oc.env:SPROUT_WITHIN_CKPT," + os.path.expanduser(
        "~/WAM/WATCH/ckpt/watch-mid-v10/egoverse_ret_v3_448_state_blankret_bv2_v10_4node/") + "model_000008000.pt}"
)

_SPROUT_PV0_DATASET = L(SproutPlainEndDataset)(
    manifest_dir=MANIFEST,
    dataset_root=SPROUT_ROOT,
    use_center_top=False,
    t5_text_embeddings_path=f"{MANIFEST}/t5_embeddings.pkl",
    action_stats_path=f"{MANIFEST}/abs_action29_stats_pctl.json",
    state_stats_path=f"{MANIFEST}/state29_stats_pctl.json",
    target_group="abs_action29",
    target_stats_path=f"{MANIFEST}/abs_action29_stats_pctl.json",
    robust_state_stats=False,
    chunk_size=32,
    per_frame_stride=1,
    final_image_size=224,
    num_duplicates_per_image=4,
    use_residual_actions=False,
    retrieval_dropout_prob=0.0,
    require_t5=True,
    per_view=True,
    per_view_blank=NB,
)


def _sprout_pv_override(chunk: int) -> dict:
    o = _plain_end_model_override()
    o["chunk_size"] = chunk
    o["action_dim"] = DIM29
    o["proprio_dim"] = DIM29
    o["state_t"] = STATE_T
    o["tokenizer"] = dict(chunk_duration=1 + 4 * (STATE_T - 1))
    n_cond = 1 + NB + N_CAMS                   # up to the action slot
    o["min_num_conditional_frames"] = n_cond
    o["max_num_conditional_frames"] = n_cond
    o["conditional_frames_probs"] = {n_cond: 1.0}
    return o


def _sprout_pv0(name: str, ckpt: str, chunk: int):
    ds = copy.deepcopy(_SPROUT_PV0_DATASET)
    ds.chunk_size = chunk
    e = _tabletop_short_post(name, ds, ckpt, model_override=_sprout_pv_override(chunk))
    e["dataloader_train"]["batch_size"] = BATCH
    e["trainer"]["max_iter"] = ITER
    e["checkpoint"]["save_iter"] = SAVE_ITER
    e["scheduler"]["cycle_lengths"] = [ITER, 100000000000000]
    e["scheduler"]["warm_up_steps"] = [ITER // 40, 0]
    return e


_NAMES = []
for _chunk in (32, 16):
  for _tag, _ck in (("", _CK_WATCH_MID), ("_within", _CK_WITHIN_MID), ("_base", "")):
    _name = f"cosmos_predict2p5_2b_sprout_3task_pv0_c{_chunk}{_tag}"
    globals()[_name] = _sprout_pv0(_name, _ck, _chunk)
    globals()[f"{_name}__inference_only"] = _inference_only(_name)
    _NAMES += [_name, f"{_name}__inference_only"]

# ---- ICL: human ghost demo in context (2026-11-11) --------------------------------------------------------------
# GR1 v6 (..._egoverse_gr1_ret_v6_448_state: demo video + demo state chunk -> action + future) shrunk to
# state_t = 7 for one 224 ego camera: demo = its first and last frame + its 32 x 57 human state chunk, future = the
# chunk-end frame (pv0). Layout and reasons: SproutIclDataset (sprout_dataset.py). 196 tokens per latent, 1,372 in all
# (v6: 784 x 20 = 15,680).
#
# Data: cup_to_drawer + open_drawer only (pnp has no human data), 6,158 windows; the demo is the Quest ghost of the
# query's own robot episode, at the same t +- 6 frames (``human_ghost/``, quest_capture/pc/to_sprout_ghost.py).
#   ..._sprout_icl_c32          WATCH mid (same as pv0)
#   ..._sprout_icl_c32_base     no mid
#   ..._sprout_icl_c32_mix30    WATCH mid, 30% of demos a retrieved ghost of ANOTHER episode
#   ..._sprout_icl_c32_mix30_base   no mid, 30% of demos from another episode
# Schedule (user, 2026-11-12): pv0's 4K iterations at HALF its global batch, i.e. global 64
# (scripts/train_sprout_2node.sh ARM=icl*: 2 nodes x 8 GPUs x 4, or 1 node x 8 x 8). One epoch of the 6,158 windows is
# 96 iterations, so 1K / 2K / 3K / 4K = 10 / 21 / 31 / 42 epochs. SPROUT_ICL_ITER / SPROUT_ICL_SAVE_ITER change it.
ICL_ITER = int(os.environ.get("SPROUT_ICL_ITER", 4_000))
ICL_SAVE_ITER = int(os.environ.get("SPROUT_ICL_SAVE_ITER", 1_000))
ICL_STATE_T = 7
ICL_N_COND = 5          # cur_state, ret_action, demo start, demo end, cur_image

_SPROUT_ICL_DATASET = L(SproutIclDataset)(**{**{k: v for k, v in _SPROUT_PV0_DATASET.items() if k != "_target_"},
                                             "jitter": 6, "other_prob": 0.0})


def _sprout_icl_override() -> dict:
    o = _plain_end_model_override()
    o["chunk_size"] = 32
    o["action_dim"] = DIM29
    o["proprio_dim"] = DIM29
    o["state_t"] = ICL_STATE_T
    o["tokenizer"] = dict(chunk_duration=1 + 4 * (ICL_STATE_T - 1))      # 25
    o["min_num_conditional_frames"] = ICL_N_COND
    o["max_num_conditional_frames"] = ICL_N_COND
    o["conditional_frames_probs"] = {ICL_N_COND: 1.0}
    return o


def _sprout_icl(name: str, ckpt: str, other_prob: float = 0.0):
    ds = copy.deepcopy(_SPROUT_ICL_DATASET)
    ds.other_prob = other_prob
    ds.chunk_size = 32
    e = _tabletop_short_post(name, ds, ckpt, model_override=_sprout_icl_override())
    e["dataloader_train"]["batch_size"] = BATCH
    e["trainer"]["max_iter"] = ICL_ITER
    e["checkpoint"]["save_iter"] = ICL_SAVE_ITER
    e["scheduler"]["cycle_lengths"] = [ICL_ITER, 100000000000000]
    e["scheduler"]["warm_up_steps"] = [ICL_ITER // 40, 0]
    return e


for _tag, _ck, _op in (("", _CK_WATCH_MID, 0.0), ("_base", "", 0.0), ("_mix30", _CK_WATCH_MID, 0.3), ("_mix30_base", "", 0.3)):
    _name = f"cosmos_predict2p5_2b_sprout_icl_c32{_tag}"
    globals()[_name] = _sprout_icl(_name, _ck, _op)
    globals()[f"{_name}__inference_only"] = _inference_only(_name)
    _NAMES += [_name, f"{_name}__inference_only"]

ALL_SPROUT_CONFIGS = [globals()[n] for n in _NAMES]
