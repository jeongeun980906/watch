"""Experiment configs for RoboTwin 2.0.

Pipeline-validation training uses three tasks — `place_cans_plasticbox`,
`place_can_basket`, `place_burger_fries` — with four additional tasks
(`place_burger_fries`, `place_bread_skillet`, `place_container_plate`,
`place_dual_shoes`) held out for test-time evaluation. A single-task debug
config (`place_cans_plasticbox` only) is provided for quick smoke training.

Embodiment selection goes through `ROBOTWIN_EMBODIMENT` (defaults to
`aloha-agilex_clean_50`, the embodiment currently on disk). Once ARX-X5 data
lands under `$BASE_DATASETS_DIR/RoboTwin2.0/dataset/<task>/arx-x5_*/data/`,
flip the env var (or override per-launch) to train on it without touching
this file.

Segment layout (matches `RoboTwinDataset.__getitem__` when `use_proprio=True`
and `return_value_function_returns=False`):
    0: blank                        (repeat 1)
    1: current proprio              (repeat num_duplicates_per_image)
    2: current left wrist           (repeat num_duplicates_per_image)
    3: current right wrist          (repeat num_duplicates_per_image)
    4: current primary (head)       (repeat num_duplicates_per_image)
    5: action placeholder           (repeat num_duplicates_per_image)
    6: future proprio               (repeat num_duplicates_per_image)
    7: future left wrist            (repeat num_duplicates_per_image)
    8: future right wrist           (repeat num_duplicates_per_image)
    9: future primary               (repeat num_duplicates_per_image)
-> state_t = 10, chunk_duration = 1 + 9 * 4 = 37, conditional_frames = 5
   (1 blank + proprio + 2 wrist + primary).

If value-function training is enabled later, append one slot and bump
`state_t` to 11 / `chunk_duration` to 41 (mirrors the ALOHA 185-demos config).
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from megatron.core import parallel_state
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler

from cosmos_policy._src.imaginaire.lazy_config import LazyCall as L
from cosmos_policy._src.imaginaire.lazy_config import LazyDict
from cosmos_policy._src.imaginaire.utils.checkpoint_db import get_checkpoint_path  # noqa: F401
from cosmos_policy.datasets.dataset_utils import calculate_dataset_statistics
from cosmos_policy.datasets.real_robot_dataset import RealRobotDataset
from cosmos_policy.datasets.robotwin_dataset import RoboTwinDataset, _split_cache_dir
from cosmos_policy.datasets.robotwin_rag_dataset import RoboTwinRetDataset
from cosmos_policy.models.policy_video2world_model import CosmosPolicyVideo2WorldModel

BASE_DATASETS_DIR = os.environ.get("BASE_DATASETS_DIR", ".")
ROBOTWIN_DATASET_ROOT = os.path.join(BASE_DATASETS_DIR, "RoboTwin2.0", "dataset")
# Flip to `arx-x5_clean_50` / `arx-x5_randomized_500` once that data is placed
# on disk. Kept as an env var so configs don't need to change per embodiment.
ROBOTWIN_EMBODIMENT = os.environ.get("ROBOTWIN_EMBODIMENT", "aloha-agilex_clean_50")
# Single T5 pkl covers every task (instructions are deterministic per folder name).
ROBOTWIN_T5_PATH = os.path.join(ROBOTWIN_DATASET_ROOT, "t5_embeddings.pkl")

SPLITS_DIR = os.path.join(
    os.path.dirname(__file__), "robotwin_splits"
)
TRAIN_3TASKS_SPLIT = os.path.join(SPLITS_DIR, "train_3tasks.txt")
TRAIN_5TASKS_SPLIT = os.path.join(SPLITS_DIR, "train_5tasks.txt")
TRAIN_10TASKS_SPLIT = os.path.join(SPLITS_DIR, "train_10tasks.txt")
# Progression cotrain split (30 tasks): 5 train + 5 heldout + 20 distractors.
# Mirrors POOL_5 in robotwin_rag_tools/scripts/progression_prefilter.py (29
# tasks) plus ``click_bell`` — POOL_5 intentionally excludes click_bell to
# keep one truly out-of-pool eval task, but for cotrain we want every known
# heldout in the train set. All 30 tasks have ur5_{clean_50, randomized_500}
# and aloha-agilex_clean_50 data on disk.
TRAIN_PROGRESSION_SPLIT = os.path.join(SPLITS_DIR, "train_progression.txt")
DEBUG_SINGLE_TASK_SPLIT = os.path.join(SPLITS_DIR, "debug_place_cans_plasticbox.txt")
DEBUG_BURGER_FRIES_SPLIT = os.path.join(SPLITS_DIR, "debug_place_burger_fries.txt")

# Cross-embodiment retrieval for RAG runs (query -> pool).
ROBOTWIN_RAG_POOL_EMBODIMENT = os.environ.get("ROBOTWIN_RAG_POOL_EMBODIMENT", "arx-x5_clean_50")
ROBOTWIN_RAG_NPZ_DIR = os.environ.get(
    "ROBOTWIN_RAG_NPZ_DIR", "/mnt/ddn/robotwin_rag_tools/retrieval_results"
)
_ROBOTWIN_RAG_NPZ_DIR_AC030 = "/mnt/ddn/robotwin_rag_tools/retrieval_results_ac030"
# v3: image=0.1, proprio_hist=0.05, grip_weight=3.0, no time, hist_window=8.
# Fixes the proprio_hist zero-bias home-jump that the earlier ac030 weights
# had. Paired with the ``prefN5_ac030_v3_k3`` pair-h5 build.
_ROBOTWIN_RAG_NPZ_DIR_AC030_V3 = "/mnt/ddn/robotwin_rag_tools/retrieval_results_ac030_lowimg_lowhist"

# Joint multi-task RAG training: same query (aloha-agilex_clean_50) and pool
# (arx-x5_clean_50 by default), one prefilterN5 retrieval npz per task. Each
# task is wrapped in its own RoboTwinRetDataset, combined with
# torch.utils.data.ConcatDataset so the trainer sees a uniform index space.
# Sample schema is identical across tasks (verified: video shape
# (3,57,224,224), actions (16,16), proprio (16,)).
#
# Stats are POOLED across all tasks: dataset_statistics.json, post-norm stats,
# and (residual mode only) delta_dataset_statistics live at the multi-task
# `_split_cache_dir` and are shared with the matching non-RAG joint config.
RAG_5TASKS = (
    "place_cans_plasticbox",
    "move_can_pot",
    "open_microwave",
    "grab_roller",
    "pick_dual_bottles",
)


def _load_or_compute_combined_dataset_stats(stats_dir, instances):
    """Combined raw-data dataset stats over the union of N instances' episodes.
    Cached as ``<stats_dir>/dataset_statistics.json`` so that joint RAG and
    joint non-RAG configs over the same task set share the file.
    """
    path = os.path.join(stats_dir, "dataset_statistics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            js = json.load(f)
        print(f"[joint_rag] loaded combined dataset statistics from: {path}")
        return {k: np.array(v) for k, v in js.items()}

    merged: dict = {}
    next_key = 0
    for ds in instances:
        for ep_data in ds.data.values():
            merged[next_key] = ep_data
            next_key += 1
    stats = calculate_dataset_statistics(merged)
    with open(path, "w") as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f, indent=4)
    print(f"[joint_rag] combined dataset statistics saved to: {path}")
    return {k: np.array(v) for k, v in stats.items()}


def _load_or_compute_combined_delta_stats(
    stats_dir, instances, pool_embodiment, npz_basenames, npz_dir=None,
):
    """Combined (gt - ret) min/max over the union of N instances' raw deltas.
    Cached at ``<stats_dir>/delta_dataset_statistics__pool-<combined_key>.json``
    where ``combined_key`` is keyed on pool_embodiment + sha1 of the sorted
    per-task npz basenames (so different task subsets / retrieval pools don't
    collide). ``npz_dir`` is folded into the hash *only when it differs from
    the legacy default* (``retrieval_results``); this preserves existing
    cache hits while letting sibling dirs (e.g. ``retrieval_results_ac030``)
    that share basenames stay distinct.
    """
    npz_dir_tag = "" if (
        npz_dir is None or os.path.basename(npz_dir.rstrip("/")) == "retrieval_results"
    ) else os.path.basename(npz_dir.rstrip("/"))
    hash_seed = "|".join(sorted(npz_basenames))
    if npz_dir_tag:
        hash_seed += f"||npz_dir={npz_dir_tag}"
    digest = hashlib.sha1(hash_seed.encode()).hexdigest()[:8]
    combined_key = f"{pool_embodiment}_joint{len(npz_basenames)}_{digest}"
    path = os.path.join(stats_dir, f"delta_dataset_statistics__pool-{combined_key}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            js = json.load(f)
        print(f"[joint_rag] loaded combined delta statistics from: {path}")
        return {k: np.array(v, dtype=np.float32) for k, v in js.items()}

    print(
        f"[joint_rag] computing combined delta (gt - ret) statistics over "
        f"{len(instances)} tasks (top-1 retrieval) ..."
    )
    pooled = []
    for ds in instances:
        d = ds.compute_raw_deltas()
        if d.size > 0:
            pooled.append(d)
    if not pooled:
        raise RuntimeError(
            "combined delta-stats computation produced no samples — retrieval "
            "lookup is empty across all tasks. Check retrieval_npz paths."
        )
    arr = np.concatenate(pooled, axis=0)
    delta_stats = {
        "delta_actions_min": arr.min(axis=0).astype(np.float32),
        "delta_actions_max": arr.max(axis=0).astype(np.float32),
    }
    with open(path, "w") as f:
        json.dump({k: v.tolist() for k, v in delta_stats.items()}, f, indent=4)
    print(f"[joint_rag] combined delta statistics saved to: {path}")
    return delta_stats


def _load_or_compute_combined_post_norm_stats(stats_dir, instances):
    """Combined post-normalization stats over the union of N instances'
    rescaled episodes. Must run AFTER each instance has been rescaled
    (i.e. after finalize())."""
    path = os.path.join(stats_dir, "dataset_statistics_post_norm.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            js = json.load(f)
        print(f"[joint_rag] loaded combined post-norm statistics from: {path}")
        return {k: np.array(v) for k, v in js.items()}

    merged: dict = {}
    next_key = 0
    for ds in instances:
        for ep_data in ds.data.values():
            merged[next_key] = ep_data
            next_key += 1
    stats = calculate_dataset_statistics(merged)
    with open(path, "w") as f:
        json.dump({k: v.tolist() for k, v in stats.items()}, f, indent=4)
    print(f"[joint_rag] combined post-norm statistics saved to: {path}")
    return {k: np.array(v) for k, v in stats.items()}


def _build_joint_rag_dataset(
    *,
    tasks,
    dataset_root,
    query_embodiment,
    pool_embodiment,
    pool_tag,
    npz_dir,
    npz_template,
    t5_text_embeddings_path,
    use_residual_actions=False,
    # When False, drop slots 11-14 (future block) → state_t=11 instead of 15.
    # The matching model config must use `_ROBOTWIN_RAG_MODEL_CONFIG_NO_FUTURE`.
    predict_future_states=True,
    retrieval_top_k_choice=1,
    retrieval_dropout_prob=0.0,
    chunk_size=16,
    final_image_size=224,
    use_image_aug=True,
    use_stronger_image_aug=True,
    num_duplicates_per_image=4,
    ret_image_subsample=4,
    return_value_function_returns=False,
    gamma=0.998,
    # Optional multi-pool composition. When given, each per-task
    # RoboTwinRetDataset spans every listed pool embodiment (e.g.
    # ``["arx-x5_clean_50", "arx-x5_randomized_500"]`` for the combined
    # 50-clean + 500-randomized pool). ``pool_embodiment`` (singular) stays
    # required so the legacy single-pool key path (delta_stats / cache
    # filename prefix) still resolves; it should equal
    # ``pool_embodiments[0]``. ``pool_tag`` must match what
    # ``run_retrieval.py`` wrote into the NPZ filename
    # (``<emb1>+<emb2>+…`` concatenation, no spaces).
    pool_embodiments=None,
    # Optional pair-h5 mode: when set, each per-task RoboTwinRetDataset reads
    # pool images / endpose / joint_action from per-query-ep h5 files at
    # ``<dataset_root>/<task>/<pair_h5_subdir>/data/episode<E>.hdf5`` instead
    # of opening original pool hdf5s. Path is resolved per task using
    # ``pair_h5_subdir.format(task=task)`` so the template can encode
    # query_split / pool_tag / prefilterN / out_tag once at the config level.
    # See build_retrieved_pair_h5.py for the output layout.
    pair_h5_subdir=None,
    # When True, each per-task RoboTwinRetDataset decodes every query + pool
    # episode's JPEGs at __init__ and serves __getitem__ from a file_path-keyed
    # in-memory dict. Forked DataLoader workers share the pages via Linux COW
    # so per-rank RAM stays flat at N_eps × 3 cam × T × bytes (~70 GB / rank
    # for clean-only 100 eps). Set on the clean-only residual configs so they
    # stay fast; pool_all (550) and pair-h5 paths leave it False.
    eager_pool_load=False,
    # Optional per-task query-episode drop list. Maps ``task -> tuple of
    # episode numbers`` to remove from ``query_files`` before metadata load,
    # so those eps are completely absent from training (not zero-filled at
    # retrieval time). Use it to drop queries whose pair_h5 was intentionally
    # skipped by ``build_retrieved_pair_h5.py --exclude-eps``.
    exclude_query_eps=None,
    # LazyConfig deep-merge from the libero parent injects dataset kwargs that
    # don't apply at the joint-helper level (e.g. ``normalize_images`` from
    # LIBERODataset). Absorb them silently — actual normalization flags for
    # each per-task RoboTwinRetDataset are hard-coded below.
    **_merge_kwargs,
):
    """Build N RoboTwinRetDataset instances (one per task) sharing pooled
    dataset / delta / post-norm stats, wrapped in a ConcatDataset."""
    tasks_list = list(tasks)
    combined_stats_dir = _split_cache_dir(
        dataset_root=dataset_root,
        tasks=tasks_list,
        embodiment=query_embodiment,
    )

    # Normalize the exclude map so .get(task, ()) always works below.
    exclude_query_eps_map: dict = dict(exclude_query_eps or {})

    # Phase 1: build deferred per-task instances (raw episodes + retrieval
    # lookup; no stats / no rescale yet).
    parts: list[RoboTwinRetDataset] = []
    npz_basenames: list[str] = []
    for task in tasks_list:
        npz_path = os.path.join(
            npz_dir,
            npz_template.format(
                task=task, query=query_embodiment, pool_tag=pool_tag,
            ),
        )
        npz_basenames.append(os.path.basename(npz_path).removesuffix(".npz"))
        pair_h5_root = None
        if pair_h5_subdir is not None:
            pair_h5_root = os.path.join(
                dataset_root, task, pair_h5_subdir.format(task=task), "data",
            )
        task_exclude = tuple(exclude_query_eps_map.get(task, ()))
        if task_exclude:
            print(
                f"[joint_rag] task={task} will exclude query eps "
                f"{sorted(task_exclude)} via exclude_query_eps"
            )
        parts.append(
            RoboTwinRetDataset(
                dataset_root=dataset_root,
                task=task,
                query_embodiment=query_embodiment,
                pool_embodiment=pool_embodiment,
                pool_embodiments=(
                    list(pool_embodiments) if pool_embodiments else None
                ),
                retrieval_npz_path=npz_path,
                t5_text_embeddings_path=t5_text_embeddings_path,
                chunk_size=chunk_size,
                final_image_size=final_image_size,
                use_image_aug=use_image_aug,
                use_stronger_image_aug=use_stronger_image_aug,
                use_proprio=True,
                use_wrist_images=True,
                normalize_proprio=True,
                normalize_actions=True,
                num_duplicates_per_image=num_duplicates_per_image,
                ret_image_subsample=ret_image_subsample,
                retrieval_top_k_choice=retrieval_top_k_choice,
                retrieval_dropout_prob=retrieval_dropout_prob,
                use_residual_actions=use_residual_actions,
                predict_future_states=predict_future_states,
                return_value_function_returns=return_value_function_returns,
                gamma=gamma,
                stats_dir=combined_stats_dir,
                defer_finalization=True,
                pair_h5_root=pair_h5_root,
                eager_pool_load=eager_pool_load,
                exclude_query_eps=task_exclude,
            )
        )

    # Phase 2: pool stats across all tasks, then finalize each instance.
    combined_stats = _load_or_compute_combined_dataset_stats(combined_stats_dir, parts)
    combined_delta_stats = None
    if use_residual_actions:
        combined_delta_stats = _load_or_compute_combined_delta_stats(
            combined_stats_dir, parts, pool_embodiment, npz_basenames,
            npz_dir=npz_dir,
        )
    for ds in parts:
        ds.finalize(combined_stats, combined_delta_stats)

    # Phase 3: post-norm stats from the now-rescaled merged data.
    combined_post_norm = _load_or_compute_combined_post_norm_stats(
        combined_stats_dir, parts,
    )
    for ds in parts:
        ds.set_post_norm_stats(combined_post_norm)

    return ConcatDataset(parts)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

robotwin_train_3tasks_dataset = L(RoboTwinDataset)(
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
    num_duplicates_per_image=4,  # WAN 2.1 tokenizer: 4 images per latent frame
    return_value_function_returns=False,
    gamma=0.998,
)

robotwin_train_5tasks_dataset = L(RoboTwinDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    tasks=TRAIN_5TASKS_SPLIT,
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

# No-future ablation of the baseline 5-task config. Drops the 4 future
# latent slots (state_t 10 → 6) so the trainer only learns action prediction
# from current obs — no future-state / future-image loss. Pair with
# _ROBOTWIN_MODEL_CONFIG_NO_FUTURE in the matching experiment LazyDict.
robotwin_train_5tasks_dataset_no_future = L(RoboTwinDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    tasks=TRAIN_5TASKS_SPLIT,
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
    predict_future_states=False,
)

robotwin_debug_single_task_dataset = L(RoboTwinDataset)(
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

# EE baseline (non-RAG, aloha-only EE actions) for place_burger_fries — the
# EE-space counterpart of robotwin_qpos_debug_place_burger_fries_dataset.
# Used as the absolute-EE baseline for the aloha→arx / aloha→piper /
# franka→ur5 RAG comparisons.
robotwin_debug_place_burger_fries_dataset = L(RoboTwinDataset)(
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

# Franka-only EE baseline (non-RAG) for place_burger_fries. Embodiment
# hardcoded so it doesn't depend on ROBOTWIN_EMBODIMENT (which other configs
# rely on for the aloha-query default). Counterpart to the
# franka_query_ur5_pool RAG residual variant.
robotwin_debug_place_burger_fries_franka_dataset = L(RoboTwinDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    tasks=DEBUG_BURGER_FRIES_SPLIT,
    embodiment="franka_clean_50",
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
# RAG dataset: debug single task (place_cans_plasticbox).
# Query = aloha-agilex_clean_50 (same as debug base config).
# Pool  = arx-x5_clean_50 (cross-embodiment conditioning).
# ---------------------------------------------------------------------------

robotwin_rag_debug_place_cans_plasticbox_dataset = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment=ROBOTWIN_RAG_POOL_EMBODIMENT,
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_cans_plasticbox__q-aloha-agilex_clean_50__p-arx-x5_clean_50__prefilterN5.npz",
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
    ret_image_subsample=4,  # 16-frame window -> 4 frames (exactly one latent)
    retrieval_top_k_choice=1,  # pick one of the top-3 retrieved episodes at random each step
    retrieval_dropout_prob=0.0,  # robustness to no-match eval
    return_value_function_returns=False,
    gamma=0.998,
)

# Residual variant — model predicts (gt_action - ret_action) delta normalized
# by per-component delta_min/max. delta_dataset_statistics.json is computed at
# first dataset load and cached next to dataset_statistics.json. At inference
# the server unnormalizes the delta with the same stats and adds raw
# ret_actions back.
robotwin_rag_debug_place_cans_plasticbox_dataset_residual = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment=ROBOTWIN_RAG_POOL_EMBODIMENT,
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_cans_plasticbox__q-aloha-agilex_clean_50__p-arx-x5_clean_50__prefilterN5.npz",
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

# Top-3 retrieval choice — same residual dataset as above but each step
# samples one of the top-3 retrieved frames (not always top-1). Trains the
# model to be robust to "good but not best" retrievals; matches what the
# inference server does when retrieval_top_k > 1.
robotwin_rag_debug_place_cans_plasticbox_dataset_residual_top3 = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment=ROBOTWIN_RAG_POOL_EMBODIMENT,
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_cans_plasticbox__q-aloha-agilex_clean_50__p-arx-x5_clean_50__prefilterN5.npz",
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
    retrieval_top_k_choice=3,
    retrieval_dropout_prob=0.0,
    use_residual_actions=True,
    return_value_function_returns=False,
    gamma=0.998,
)


# Multi-split pool variant — same query (aloha-agilex_clean_50) but pool
# spans BOTH arx-x5_clean_50 (50 ep) AND arx-x5_randomized_500 (500 ep) for
# 550 total candidates. The retrieval npz (produced by run_retrieval.py with
# --pool-splits arx-x5_clean_50 arx-x5_randomized_500) embeds the
# originating split per match_id so the dataset's (split, ep_num) lookup
# disambiguates correctly. Residual + image-variant npz to match the 'v2'
# config above, just with the larger pool.
robotwin_rag_debug_place_cans_plasticbox_dataset_residual_pool_all = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiments=["arx-x5_clean_50", "arx-x5_randomized_500"],
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_cans_plasticbox__q-aloha-agilex_clean_50__p-arx-x5_clean_50+arx-x5_randomized_500__prefilterN5.npz",
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


# v2 retrieval pairs: action_chunk raw L2 + tiny dino (1e-5) + tiny proprio
# (0.01) — minimizes per-step delta variance for spike-free residual learning.
# Paired with residual action learning.
robotwin_rag_debug_place_cans_plasticbox_dataset_residual_v2 = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment=ROBOTWIN_RAG_POOL_EMBODIMENT,
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_v2_place_cans_plasticbox__q-aloha-agilex_clean_50__p-arx-x5_clean_50__prefilterN5.npz",
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


# ---------------------------------------------------------------------------
# UR5 pool variants — same task / same query, but pool = ur5_clean_50.
# Retrieval npz produced by `scripts/run_retrieval.py --pool-split ur5_clean_50`.
# pool_embodiment is hardcoded (ignores ROBOTWIN_RAG_POOL_EMBODIMENT) so these
# always match their npz files.
# ---------------------------------------------------------------------------

robotwin_rag_debug_place_cans_plasticbox_ur5_dataset = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_cans_plasticbox__q-aloha-agilex_clean_50__p-ur5_clean_50__prefilterN5.npz",
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

robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_residual = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_cans_plasticbox__q-aloha-agilex_clean_50__p-ur5_clean_50__prefilterN5.npz",
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

robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_v2 = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    # v2 retrieval: image + proprio + action_chunk only (delta_action / time
    # dropped). action_chunk=5.0, image=0.1, proprio=0.2 — heavily future-action
    # dominated. Built by scripts/run_retrieval.py.
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_v2_place_cans_plasticbox__q-aloha-agilex_clean_50__p-ur5_clean_50__prefilterN5.npz",
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

robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_residual_v2 = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    # v2 retrieval: image + proprio + action_chunk only (delta_action / time
    # dropped). action_chunk=5.0, image=0.1, proprio=0.2 — heavily future-action
    # dominated. Built by scripts/run_retrieval.py.
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_v2_place_cans_plasticbox__q-aloha-agilex_clean_50__p-ur5_clean_50__prefilterN5.npz",
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

# v3 = v2 with the same H=16 horizon but a tighter prefilter (N=2 instead of
# N=5). Trades candidate diversity for cleaner per-query candidate set.
robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_residual_v3 = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_v3_place_cans_plasticbox__q-aloha-agilex_clean_50__p-ur5_clean_50__prefilterN2.npz",
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


# ---------------------------------------------------------------------------
# place_burger_fries — RAG datasets for the 3-experiment comparison
# (aloha→arx, aloha→piper, franka→ur5). All use the ``weighted`` retrieval
# variant (image + proprio + action_chunk + delta_action + time, DEFAULT_WEIGHTS).
# ---------------------------------------------------------------------------

robotwin_rag_debug_place_burger_fries_arx_dataset = L(RoboTwinRetDataset)(
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

robotwin_rag_debug_place_burger_fries_arx_dataset_residual = L(RoboTwinRetDataset)(
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

robotwin_rag_debug_place_burger_fries_piper_dataset_residual = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_burger_fries",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="piper_clean_50",
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_burger_fries__q-aloha-agilex_clean_50__p-piper_clean_50__prefilterN5.npz",
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

# aloha → ur5 (largest embodiment gap of the comparison set: ||L_offset||=0.78,
# quat 88°). Kept to show the upper-bound limit of cross-embodiment retrieval.
robotwin_rag_debug_place_burger_fries_ur5_dataset = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_burger_fries",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_burger_fries__q-aloha-agilex_clean_50__p-ur5_clean_50__prefilterN5.npz",
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

robotwin_rag_debug_place_burger_fries_ur5_dataset_residual = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_burger_fries",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_burger_fries__q-aloha-agilex_clean_50__p-ur5_clean_50__prefilterN5.npz",
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


robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_dataset_residual = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_burger_fries",
    query_embodiment="franka_clean_50",              # query hardcoded: franka, not aloha
    pool_embodiment="ur5_clean_50",
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_burger_fries__q-franka_clean_50__p-ur5_clean_50__prefilterN5.npz",
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

# Same as the franka_query_ur5_pool variant above but with arx-x5 as the pool.
# Mirrors the burger_fries aloha→arx baseline shape; query stays hardcoded
# franka so this config doesn't depend on ROBOTWIN_EMBODIMENT.
robotwin_rag_debug_place_burger_fries_franka_query_arx_pool_dataset_residual = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_burger_fries",
    query_embodiment="franka_clean_50",
    pool_embodiment="arx-x5_clean_50",
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_burger_fries__q-franka_clean_50__p-arx-x5_clean_50__prefilterN5.npz",
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


# Panda (Franka) as query, UR5 as pool. Query embodiment hardcoded so it
# doesn't depend on the ROBOTWIN_EMBODIMENT env var (which other configs in
# this file rely on for the aloha-query default).
robotwin_rag_debug_place_cans_plasticbox_panda_query_ur5_pool_dataset_residual = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment="panda_clean_50",
    pool_embodiment="ur5_clean_50",
    # Generated post-data-collection by:
    #   run_retrieval.py --query-split panda_clean_50 --pool-splits ur5_clean_50 \
    #                    --top-k 10 --prefilter-top-n 5
    # Uses the full ``weighted`` retrieval mix (image+proprio+action_chunk+
    # delta_action+time, default DEFAULT_WEIGHTS).
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_cans_plasticbox__q-panda_clean_50__p-ur5_clean_50__prefilterN5.npz",
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


robotwin_rag_debug_place_cans_plasticbox_piper_dataset_residual = L(RoboTwinRetDataset)(
    dataset_root=ROBOTWIN_DATASET_ROOT,
    task="place_cans_plasticbox",
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="piper_clean_50",
    # Full ``weighted`` retrieval (image+proprio+action_chunk+delta_action+
    # time, DEFAULT_WEIGHTS) against the piper pool. Generated post-data-
    # collection by run_retrieval.py with --pool-splits piper_clean_50
    # --prefilter-top-n 5.
    retrieval_npz_path=os.path.join(
        ROBOTWIN_RAG_NPZ_DIR,
        "retrieval_weighted_place_cans_plasticbox__q-aloha-agilex_clean_50__p-piper_clean_50__prefilterN5.npz",
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


# ---------------------------------------------------------------------------
# Joint 5-task RAG datasets (arx-x5_clean_50 pool, prefilterN5).
# Same model topology as the single-task RAG configs; only the dataset is a
# ConcatDataset over per-task RoboTwinRetDataset instances. Each task has its
# own dataset_statistics.json + delta_dataset_statistics_*.json (per-pool key).
# Use the residual variant for default training (matches debug_residual line).
# ---------------------------------------------------------------------------

robotwin_rag_train_5tasks_arx_dataset = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment=ROBOTWIN_RAG_POOL_EMBODIMENT,
    pool_tag="arx-x5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=False,
)

robotwin_rag_train_5tasks_arx_dataset_residual = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment=ROBOTWIN_RAG_POOL_EMBODIMENT,
    pool_tag="arx-x5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
)

robotwin_rag_train_5tasks_ur5_dataset_residual = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_tag="ur5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    eager_pool_load=True,
)

# ac030-weighted clean-only UR5 sibling of the residual config above.
# Same pool (50 ur5 clean), same eager preload, but retrieval npz comes from
# the ac030 dir (action_chunk=0.1, delta_action=0.1, proprio_hist=0.5).
# The cache key includes the npz_dir basename so its delta_stats file is
# distinct from the default-weights sibling.
robotwin_rag_train_5tasks_ur5_dataset_residual_ac030 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_tag="ur5_clean_50",
    npz_dir=_ROBOTWIN_RAG_NPZ_DIR_AC030,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    eager_pool_load=True,
)

# Non-residual UR5 5-task RAG (parallel to robotwin_rag_train_5tasks_arx
# which is non-residual + weighted NPZs). Same weighted NPZs as the
# residual variant but the model predicts absolute action, not (gt - ret).
robotwin_rag_train_5tasks_ur5_dataset = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_tag="ur5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=False,
    eager_pool_load=True,
)

# v3 = raw action-chunk L2 over H=16 with prefilterN=2 (per-task v3 NPZs).
# Mirrors robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_residual_v3 but
# joined across the 5 RAG tasks.
robotwin_rag_train_5tasks_ur5_dataset_residual_v3 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_tag="ur5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_v3_{task}__q-{query}__p-{pool_tag}__prefilterN2.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
)

# `no_future` ablation: same weighted NPZ pool as the canonical UR5/ARX
# residual datasets, but the dataset drops slots 11-14 (future block) so
# the model only learns action prediction. Pair with the corresponding
# `*_residual_no_future` training config (which uses
# `_ROBOTWIN_RAG_MODEL_CONFIG_NO_FUTURE`).
robotwin_rag_train_5tasks_ur5_dataset_residual_no_future = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_tag="ur5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    predict_future_states=False,
)

robotwin_rag_train_5tasks_arx_dataset_residual_no_future = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment=ROBOTWIN_RAG_POOL_EMBODIMENT,
    pool_tag="arx-x5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    predict_future_states=False,
)

# `no_prefilter` ablation: weighted retrieval but NPZs are the full-pool
# variants (no `__prefilterN*` suffix) — top-1 per query is picked from the
# entire pool rather than a pre-filtered top-N. Future-state prediction is
# left ON (same `_ROBOTWIN_RAG_MODEL_CONFIG`); pair with the corresponding
# training config.
robotwin_rag_train_5tasks_ur5_dataset_residual_no_prefilter = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_tag="ur5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
)

robotwin_rag_train_5tasks_arx_dataset_residual_no_prefilter = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment=ROBOTWIN_RAG_POOL_EMBODIMENT,
    pool_tag="arx-x5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
)

# Top-3 randomized retrieval choice. Same weighted NPZs / prefilterN5 as the
# canonical 5tasks residual datasets, but each step samples uniformly from the
# top-3 retrieved episodes instead of always using top-1. Trains the model to
# be robust to "good but not best" retrievals (matches what the inference
# server does when retrieval_top_k > 1).
robotwin_rag_train_5tasks_arx_dataset_residual_top3 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment=ROBOTWIN_RAG_POOL_EMBODIMENT,
    pool_tag="arx-x5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    retrieval_top_k_choice=3,
)

robotwin_rag_train_5tasks_ur5_dataset_residual_top3 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_tag="ur5_clean_50",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    retrieval_top_k_choice=3,
)

# Combined-pool variants. Same 5tasks joint setup but the pool is clean_50 +
# randomized_500 per embodiment (550 episodes/task). Pair retrieval (NPZ)
# was recomputed via run_retrieval.py over the larger pool, so each per-task
# NPZ basename carries the combined pool tag ``<emb>_clean_50+<emb>_randomized_500``.
robotwin_rag_train_5tasks_arx_dataset_residual_pool_all = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="arx-x5_clean_50",
    pool_embodiments=["arx-x5_clean_50", "arx-x5_randomized_500"],
    pool_tag="arx-x5_clean_50+arx-x5_randomized_500",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
)

robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_embodiments=["ur5_clean_50", "ur5_randomized_500"],
    pool_tag="ur5_clean_50+ur5_randomized_500",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
)

# Combined pool + top-3 randomized retrieval choice. Same combined pool NPZs
# as ..._pool_all but each step samples uniformly from the top-3 retrieved
# episodes instead of always top-1.
robotwin_rag_train_5tasks_arx_dataset_residual_pool_all_top3 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="arx-x5_clean_50",
    pool_embodiments=["arx-x5_clean_50", "arx-x5_randomized_500"],
    pool_tag="arx-x5_clean_50+arx-x5_randomized_500",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    retrieval_top_k_choice=3,
)

robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_embodiments=["ur5_clean_50", "ur5_randomized_500"],
    pool_tag="ur5_clean_50+ur5_randomized_500",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    retrieval_top_k_choice=3,
)

# ─── ac030 variants ─────────────────────────────────────────────────────────
# Same pool/query/template as ``_pool_all_top3`` but reads NPZs from
# retrieval_results_ac030/. Those NPZs were re-generated with weakened weights
# (action_chunk=0.1, delta_action=0.1, time excluded, proprio=1.0,
# proprio_hist=0.5) and a K=8 motion-delta history window, so the retrieval
# is dominated by proprio + motion deltas rather than action_chunk.
robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_ac030 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_embodiments=["ur5_clean_50", "ur5_randomized_500"],
    pool_tag="ur5_clean_50+ur5_randomized_500",
    npz_dir=_ROBOTWIN_RAG_NPZ_DIR_AC030,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    retrieval_top_k_choice=3,
)

# ─── pair-h5 variants ──────────────────────────────────────────────────────
# Same retrieval semantics as the corresponding non-pair-h5 dataset above,
# but pool images / endpose / joint_action are read from the per-query-ep
# h5 files built by ``build_retrieved_pair_h5.py``. Pool side never opens
# the original pool hdf5 files. ``pair_h5_subdir`` is templated per task
# to land at ``<dataset_root>/<task>/<query_split>__pair__<pool_tag>__prefN5[_<out_tag>]/data``.

# NOTE: repointed in-place from the legacy default-weighted retrieval +
# ``prefN5_k3`` pair-h5 to the new ac030_v3 setup (image=0.1, proprio_hist=0.05,
# grip_weight=3.0, no time, hist_window=8 — fixes the proprio_hist zero-bias
# home-jump that the earlier weights had). Pair-h5 dir built with out_tag
# ``ac030_v3_k3``. Experiment name (``..._pool_all_top3_pair_h5``) stays the
# same so the existing training shell / ckpt path don't need renaming.
robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_pair_h5 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_embodiments=["ur5_clean_50", "ur5_randomized_500"],
    pool_tag="ur5_clean_50+ur5_randomized_500",
    npz_dir=_ROBOTWIN_RAG_NPZ_DIR_AC030_V3,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    retrieval_top_k_choice=3,
    pair_h5_subdir=(
        ROBOTWIN_EMBODIMENT
        + "__pair__ur5_clean_50+ur5_randomized_500__prefN5_ac030_v3_k3"
    ),
    # Drop the query eps whose pair_h5 was intentionally skipped by
    # ``build_retrieved_pair_h5.py --exclude-eps`` (rot6 + pos_w=4 rebuild,
    # 2026-05-19). Without this, those eps would silently become
    # ``has_ret_data=0`` zero-filled samples in training.
    exclude_query_eps={
        "move_can_pot": (32, 45),
        "open_microwave": (34,),
        "grab_roller": (38,),
    },
)

robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_ac030_pair_h5 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_embodiments=["ur5_clean_50", "ur5_randomized_500"],
    pool_tag="ur5_clean_50+ur5_randomized_500",
    npz_dir=_ROBOTWIN_RAG_NPZ_DIR_AC030,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    retrieval_top_k_choice=3,
    pair_h5_subdir=(
        ROBOTWIN_EMBODIMENT
        + "__pair__ur5_clean_50+ur5_randomized_500__prefN5_ac030_k3"
    ),
)


# ─── K=1 eager-decode pair-h5 variants ─────────────────────────────────────
# Point at the ``_k1`` pair h5 subdirs built by ``build_retrieved_pair_h5.py
# --max-k 1``. With PAIR_H5_EAGER_DECODE=1 in the shell, RoboTwinRetDataset
# walks every (q_ep, k=0, t) JPEG triplet at __init__ and decodes them into
# uint8 numpy arrays in self._pool_imgs_cache; __getitem__ is then a pure
# dict.get. num_workers MUST be 0 — the cached arrays cannot survive being
# duplicated across DataLoader worker forks.
robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_pair_h5_k1 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_embodiments=["ur5_clean_50", "ur5_randomized_500"],
    pool_tag="ur5_clean_50+ur5_randomized_500",
    npz_dir=ROBOTWIN_RAG_NPZ_DIR,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    retrieval_top_k_choice=1,
    pair_h5_subdir=(
        ROBOTWIN_EMBODIMENT
        + "__pair__ur5_clean_50+ur5_randomized_500__prefN5_k1"
    ),
)

robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_ac030_pair_h5_k1 = L(_build_joint_rag_dataset)(
    tasks=RAG_5TASKS,
    dataset_root=ROBOTWIN_DATASET_ROOT,
    query_embodiment=ROBOTWIN_EMBODIMENT,
    pool_embodiment="ur5_clean_50",
    pool_embodiments=["ur5_clean_50", "ur5_randomized_500"],
    pool_tag="ur5_clean_50+ur5_randomized_500",
    npz_dir=_ROBOTWIN_RAG_NPZ_DIR_AC030,
    npz_template="retrieval_weighted_{task}__q-{query}__p-{pool_tag}__prefilterN5.npz",
    t5_text_embeddings_path=ROBOTWIN_T5_PATH,
    use_residual_actions=True,
    retrieval_top_k_choice=1,
    pair_h5_subdir=(
        ROBOTWIN_EMBODIMENT
        + "__pair__ur5_clean_50+ur5_randomized_500__prefN5_ac030_k1"
    ),
)


# ---------------------------------------------------------------------------
# Model / tokenizer shape shared by every RoboTwin config below
# (proprio on, no value: 10 slots / chunk_duration=37 / 5 conditional frames)
# ---------------------------------------------------------------------------

_ROBOTWIN_MODEL_CONFIG = dict(
    state_t=10,
    min_num_conditional_frames=5,
    max_num_conditional_frames=5,
    # Native absolute-EE signal: [xyz(3) + quat_wxyz(4) + grip(1)] * 2 arms = 16.
    # Tokenizer chunk_duration 37 derives from state_t=10,
    # num_duplicates_per_image=4, conditional_frames=5.
    action_dim=16,
    proprio_dim=16,
    chunk_size=16,
    action_loss_multiplier=16,
    tokenizer=dict(
        chunk_duration=37,
    ),
)

# Ablation for the non-RAG baseline: dataset.predict_future_states=False
# drops slots 6-9 (future block: proprio + head + L/R wrist) →
# state_t = 10 - 4 = 6, chunk_duration = 1 + 5 * 4 = 21. Conditional frames
# stay at 5 (current obs untouched).
_ROBOTWIN_MODEL_CONFIG_NO_FUTURE = dict(
    state_t=6,
    min_num_conditional_frames=5,
    max_num_conditional_frames=5,
    action_dim=16,
    proprio_dim=16,
    chunk_size=16,
    action_loss_multiplier=16,
    tokenizer=dict(
        chunk_duration=21,  # 1 + 5 * 4
    ),
)

# RAG variant: prepends 5 retrieved slots (ret head/Lw/Rw/proprio/action) to the
# base 10-slot layout, so conditional_frames grows from 5 to 10 and state_t/chunk
# shift accordingly. See RoboTwinRetDataset.__getitem__ docstring for the exact
# slot numbering.
_ROBOTWIN_RAG_MODEL_CONFIG = dict(
    state_t=15,
    min_num_conditional_frames=10,
    max_num_conditional_frames=10,
    action_dim=16,
    proprio_dim=16,
    chunk_size=16,
    action_loss_multiplier=16,
    tokenizer=dict(
        chunk_duration=57,  # 1 + 14 * 4
    ),
)

# Ablation: dataset.predict_future_states=False drops slots 11-14 (future
# block) → state_t = 15 - 4 = 11, chunk_duration = 1 + 10 * 4 = 41.
# Conditional-frames count stays at 10 (slots 0-9 are still all ret + current).
_ROBOTWIN_RAG_MODEL_CONFIG_NO_FUTURE = dict(
    state_t=11,
    min_num_conditional_frames=10,
    max_num_conditional_frames=10,
    action_dim=16,
    proprio_dim=16,
    chunk_size=16,
    action_loss_multiplier=16,
    tokenizer=dict(
        chunk_duration=41,  # 1 + 10 * 4
    ),
)

_ROBOTWIN_SCHEDULER = dict(
    # LR decay for 20K steps in cycle #1, then decay by 5x forever in cycle #2.
    cycle_lengths=[20000, 100000000000000],
    warm_up_steps=[2000, 0],
    f_start=[1e-6, 0.06],
    f_max=[1.0, 0.06],
    f_min=[0.3, 0.06],
)

# 5-task training (RAG and non-RAG) needs more iterations to converge than the
# 3-task pipeline run. cycle_lengths is scaled 3x relative to _ROBOTWIN_SCHEDULER
# so the LR decay matches max_iter.
_ROBOTWIN_5TASKS_SCHEDULER = dict(
    cycle_lengths=[60000, 100000000000000],
    warm_up_steps=[2000, 0],
    f_start=[1e-6, 0.06],
    f_max=[1.0, 0.06],
    f_min=[0.3, 0.06],
)

_ROBOTWIN_5TASKS_TRAINER = dict(
    max_iter=30000,
    logging_iter=5,
)


# ---------------------------------------------------------------------------
# 3-task training config (pipeline-validation run)
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_train_3tasks = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        scheduler=_ROBOTWIN_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_train_3tasks_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_train_3tasks_dataset,
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
            name="cosmos_predict2_2b_480p_robotwin_train_3tasks",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_train_3tasks__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_train_3tasks",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_train_3tasks__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# 5-task training config (non-RAG). Same task set as RAG_5TASKS so the two
# variants can be compared head-to-head. Uses the 3x trainer/scheduler shared
# with the RAG 5tasks runs.
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_train_5tasks = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_train_5tasks_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_train_5tasks_dataset,
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
            name="cosmos_predict2_2b_480p_robotwin_train_5tasks",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_train_5tasks__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_train_5tasks",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_train_5tasks__inference_only",
        ),
    )
)


# ── No-future ablation of the 5-task baseline ───────────────────────────
# Drops the 4 future latent slots (state_t 10 → 6, chunk_duration 37 → 21).
# Trainer only learns action prediction from current obs.
cosmos_predict2_2b_480p_robotwin_train_5tasks_no_future = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_MODEL_CONFIG_NO_FUTURE,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_train_5tasks_dataset_no_future,
            sampler=L(DistributedSampler)(
                dataset=robotwin_train_5tasks_dataset_no_future,
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
            name="cosmos_predict2_2b_480p_robotwin_train_5tasks_no_future",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_train_5tasks_no_future__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_train_5tasks_no_future",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_train_5tasks_no_future__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# Cross-embodiment co-training / pretrain-finetune datasets (non-RAG).
#
# Use ConcatDataset over per-embodiment RoboTwinDataset instances; each
# instance reads the 5tasks split for its own embodiment. Each split keeps
# its own normalization (per-embodiment dataset_statistics.json) — actions
# stay in [-1, 1] within each instance, the trainer sees a uniform 16D
# normalized action across the mixed batch.
#
# Layouts:
#   cotrain  (UR/ARX): {ur5,arx-x5}_{clean_50, randomized_500} + aloha clean
#   pretrain (UR/ARX): {ur5,arx-x5}_{clean_50, randomized_500} only
#   finetune (aloha) : aloha-agilex_clean_50 only (== robotwin_train_5tasks_dataset)
# ---------------------------------------------------------------------------


# Two separate joint-stats families, one per cross-embodiment pipeline:
#   * UR group  : (UR clean + UR rand + Aloha)  → cotrain_ur,  pretrain_ur,
#                                                 finetune_after_ur
#   * ARX group : (ARX clean + ARX rand + Aloha) → cotrain_arx, pretrain_arx,
#                                                  finetune_after_arx
# Each group's configs share one dataset_statistics.json so pretrain →
# finetune normalization is consistent within the pipeline. UR and ARX
# pipelines use *different* stats files (intentional — different action
# distributions, separately normalized).
_5TASKS_UR_SPLITS = [
    (TRAIN_5TASKS_SPLIT, "ur5_clean_50"),
    (TRAIN_5TASKS_SPLIT, "ur5_randomized_500"),
    (TRAIN_5TASKS_SPLIT, "aloha-agilex_clean_50"),
]
_5TASKS_ARX_SPLITS = [
    (TRAIN_5TASKS_SPLIT, "arx-x5_clean_50"),
    (TRAIN_5TASKS_SPLIT, "arx-x5_randomized_500"),
    (TRAIN_5TASKS_SPLIT, "aloha-agilex_clean_50"),
]
_5TASKS_UR_CACHE_LABEL = "5tasks_joint_ur_aloha"
_5TASKS_ARX_CACHE_LABEL = "5tasks_joint_arx_aloha"


def _build_joint_multi_embodiment_dataset_with_combined_stats(
    splits, *, cache_label, stats_splits=None, **_libero_parent_leftovers,
):
    """Build a ConcatDataset of RoboTwinDataset instances with **joint
    dataset_statistics** computed across instances.

    ``stats_splits`` (default = ``splits``) controls *which* instances are
    used to compute the joint stats — passing a superset lets multiple
    configs (cotrain UR, pretrain UR, finetune Aloha, …) share one global
    stats file even though each loads only its own data subset.

    Cache layout:
        {ROBOTWIN_DATASET_ROOT}/_cosmos_cache/_joint__{cache_label}/
            dataset_statistics.json
            dataset_statistics_post_norm.json

    Pass ``cache_label=_5TASKS_JOINT_CACHE_LABEL`` and
    ``stats_splits=_5TASKS_ALL_EMBODIMENT_SPLITS`` from every 5-task config
    to share one stats file.
    """
    cache_dir = os.path.join(
        ROBOTWIN_DATASET_ROOT, "_cosmos_cache", f"_joint__{cache_label}"
    )
    os.makedirs(cache_dir, exist_ok=True)

    stats_splits = list(stats_splits) if stats_splits is not None else list(splits)
    stats_path = os.path.join(cache_dir, "dataset_statistics.json")
    post_norm_path = os.path.join(cache_dir, "dataset_statistics_post_norm.json")

    def _make_instance(split_path, embodiment, defer):
        return RoboTwinDataset(
            dataset_root=ROBOTWIN_DATASET_ROOT,
            tasks=split_path,
            embodiment=embodiment,
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
            stats_dir=cache_dir,
            defer_finalization=defer,
            # Joint-multi-embodiment pools concat O(5k) episodes, which the
            # eager (decoded uint8) path can't fit in RAM. Retain JPEG bytes
            # and decode-per-getitem so workers stay within pod memory.
            lazy_video_decompression=True,
        )

    # --- Compute (or load) joint raw-data stats over stats_splits ----------
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            js = json.load(f)
        combined_stats = {k: np.array(v) for k, v in js.items()}
        print(f"[joint_emb] loaded combined dataset statistics from: {stats_path}")
    else:
        stats_parts = [_make_instance(p, e, defer=True) for p, e in stats_splits]
        combined_stats = _load_or_compute_combined_dataset_stats(cache_dir, stats_parts)
        del stats_parts  # release raw episode memory after stats are computed

    # --- Build the data-side instances and finalize with the shared stats ---
    parts: list[RoboTwinDataset] = [_make_instance(p, e, defer=True) for p, e in splits]
    for ds in parts:
        ds.finalize(combined_stats)

    # --- Joint post-norm stats over the (now-rescaled) data instances ------
    if os.path.exists(post_norm_path):
        with open(post_norm_path, "r") as f:
            js = json.load(f)
        combined_post_norm = {k: np.array(v) for k, v in js.items()}
        print(f"[joint_emb] loaded combined post-norm statistics from: {post_norm_path}")
    else:
        combined_post_norm = _load_or_compute_combined_post_norm_stats(cache_dir, parts)
    for ds in parts:
        ds.set_post_norm_stats(combined_post_norm)

    return ConcatDataset(parts)


# Back-compat alias (defaults to UR group; legacy callers won't hit this path).
def _build_multi_embodiment_dataset(splits):
    return _build_joint_multi_embodiment_dataset_with_combined_stats(
        splits,
        cache_label=_5TASKS_UR_CACHE_LABEL,
        stats_splits=_5TASKS_UR_SPLITS,
    )


# ── UR group: cotrain_ur / pretrain_ur / finetune_after_ur share UR stats ──
# Joint stats computed over (UR clean + UR rand + Aloha). Every config in this
# group writes/reads dataset_statistics.json under cache_label
# _5TASKS_UR_CACHE_LABEL, guaranteeing pretrain → finetune handoff has zero
# normalization drift inside the UR pipeline.

robotwin_cotrain_5tasks_ur5_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_5TASKS_SPLIT, "ur5_clean_50"),
        (TRAIN_5TASKS_SPLIT, "ur5_randomized_500"),
        (TRAIN_5TASKS_SPLIT, "aloha-agilex_clean_50"),
    ],
    cache_label=_5TASKS_UR_CACHE_LABEL,
    stats_splits=_5TASKS_UR_SPLITS,
)

robotwin_pretrain_5tasks_ur5_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_5TASKS_SPLIT, "ur5_clean_50"),
        (TRAIN_5TASKS_SPLIT, "ur5_randomized_500"),
    ],
    cache_label=_5TASKS_UR_CACHE_LABEL,
    stats_splits=_5TASKS_UR_SPLITS,
)

robotwin_finetune_5tasks_aloha_after_ur_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_5TASKS_SPLIT, "aloha-agilex_clean_50"),
    ],
    cache_label=_5TASKS_UR_CACHE_LABEL,
    stats_splits=_5TASKS_UR_SPLITS,
)


# ── ARX group: cotrain_arx / pretrain_arx / finetune_after_arx share ARX stats
# Joint stats computed over (ARX clean + ARX rand + Aloha). Separate stats
# file from the UR group — UR and ARX action distributions differ enough that
# pooling all three embodiments into one would dilute either side's range.

robotwin_cotrain_5tasks_arx_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_5TASKS_SPLIT, "arx-x5_clean_50"),
        (TRAIN_5TASKS_SPLIT, "arx-x5_randomized_500"),
        (TRAIN_5TASKS_SPLIT, "aloha-agilex_clean_50"),
    ],
    cache_label=_5TASKS_ARX_CACHE_LABEL,
    stats_splits=_5TASKS_ARX_SPLITS,
)

robotwin_pretrain_5tasks_arx_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_5TASKS_SPLIT, "arx-x5_clean_50"),
        (TRAIN_5TASKS_SPLIT, "arx-x5_randomized_500"),
    ],
    cache_label=_5TASKS_ARX_CACHE_LABEL,
    stats_splits=_5TASKS_ARX_SPLITS,
)

robotwin_finetune_5tasks_aloha_after_arx_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_5TASKS_SPLIT, "aloha-agilex_clean_50"),
    ],
    cache_label=_5TASKS_ARX_CACHE_LABEL,
    stats_splits=_5TASKS_ARX_SPLITS,
)


# ── UR-10 group: cotrain / pretrain / finetune over a 10-task UR5 pool ──────
# UR5 expands to 10 tasks (5 originals + 5 previously-held-out tasks); aloha
# remains on the original 5-task split (those are the only tasks with aloha
# demos in this experiment family). Joint stats are computed over the same
# pool every config in the group sees, so pretrain → finetune normalization
# stays drift-free.
_10TASKS_UR_SPLITS = [
    (TRAIN_10TASKS_SPLIT, "ur5_clean_50"),
    (TRAIN_10TASKS_SPLIT, "ur5_randomized_500"),
    (TRAIN_5TASKS_SPLIT, "aloha-agilex_clean_50"),
]
_10TASKS_UR_CACHE_LABEL = "10tasks_joint_ur_aloha"

robotwin_cotrain_10tasks_ur5_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_10TASKS_SPLIT, "ur5_clean_50"),
        (TRAIN_10TASKS_SPLIT, "ur5_randomized_500"),
        (TRAIN_5TASKS_SPLIT, "aloha-agilex_clean_50"),
    ],
    cache_label=_10TASKS_UR_CACHE_LABEL,
    stats_splits=_10TASKS_UR_SPLITS,
)

robotwin_pretrain_10tasks_ur5_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_10TASKS_SPLIT, "ur5_clean_50"),
        (TRAIN_10TASKS_SPLIT, "ur5_randomized_500"),
    ],
    cache_label=_10TASKS_UR_CACHE_LABEL,
    stats_splits=_10TASKS_UR_SPLITS,
)

robotwin_finetune_10tasks_aloha_after_ur_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_5TASKS_SPLIT, "aloha-agilex_clean_50"),
    ],
    cache_label=_10TASKS_UR_CACHE_LABEL,
    stats_splits=_10TASKS_UR_SPLITS,
)


# ── Progression group: cotrain / pretrain / finetune share progression stats ─
# Both embodiments share the progression-pool task split, so the joint dataset
# matches the full set of pool tasks a progression-style retrieval can sample
# from. Stats also span every (split, embodiment) pair in this group, so the
# pretrain (UR-only) → finetune (Aloha-only) handoff has zero normalization
# drift — mirrors the UR-10 group layout, but over POOL_5 (29 tasks) instead
# of 10.
_PROGRESSION_UR_SPLITS = [
    (TRAIN_PROGRESSION_SPLIT, "ur5_clean_50"),
    (TRAIN_PROGRESSION_SPLIT, "ur5_randomized_500"),
    (TRAIN_PROGRESSION_SPLIT, "aloha-agilex_clean_50"),
]
_PROGRESSION_UR_CACHE_LABEL = "progression_joint_ur_aloha"

robotwin_cotrain_progression_ur5_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_PROGRESSION_SPLIT, "ur5_clean_50"),
        (TRAIN_PROGRESSION_SPLIT, "ur5_randomized_500"),
        (TRAIN_PROGRESSION_SPLIT, "aloha-agilex_clean_50"),
    ],
    cache_label=_PROGRESSION_UR_CACHE_LABEL,
    stats_splits=_PROGRESSION_UR_SPLITS,
)

robotwin_pretrain_progression_ur5_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_PROGRESSION_SPLIT, "ur5_clean_50"),
        (TRAIN_PROGRESSION_SPLIT, "ur5_randomized_500"),
    ],
    cache_label=_PROGRESSION_UR_CACHE_LABEL,
    stats_splits=_PROGRESSION_UR_SPLITS,
)

robotwin_finetune_progression_aloha_after_ur_dataset = L(_build_joint_multi_embodiment_dataset_with_combined_stats)(
    splits=[
        (TRAIN_PROGRESSION_SPLIT, "aloha-agilex_clean_50"),
    ],
    cache_label=_PROGRESSION_UR_CACHE_LABEL,
    stats_splits=_PROGRESSION_UR_SPLITS,
)


# Co-train / pretrain / finetune trainer step counts (per user spec).
_ROBOTWIN_5TASKS_TRAINER_COTRAIN = dict(
    max_iter=20_000,
    logging_iter=5,
)
_ROBOTWIN_5TASKS_TRAINER_PRETRAIN = dict(
    max_iter=10_000,
    logging_iter=5,
)
_ROBOTWIN_5TASKS_TRAINER_FINETUNE = dict(
    max_iter=10_000,
    logging_iter=5,
)
# Progression group uses a shorter schedule: 7.5K pretrain + 7.5K finetune.
_ROBOTWIN_PROGRESSION_TRAINER_PRETRAIN = dict(
    max_iter=7_500,
    logging_iter=5,
)
_ROBOTWIN_PROGRESSION_TRAINER_FINETUNE = dict(
    max_iter=7_500,
    logging_iter=5,
)


def _multi_emb_experiment(name, dataset, trainer):
    """Helper to build a LazyDict experiment with a given dataset + trainer."""
    return LazyDict(
        dict(
            defaults=["/experiment/cosmos_predict2_2b_480p_libero", "_self_"],
            trainer=trainer,
            scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
            model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_MODEL_CONFIG),
            dataloader_train=L(DataLoader)(
                num_workers=12,
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
                batch_size=25,
                drop_last=True,
            ),
            job=dict(group="cosmos_v2_finetune", name=name),
        )
    )


def _inference_only(parent_name):
    return LazyDict(
        dict(
            defaults=[f"/experiment/{parent_name}", "_self_"],
            job=dict(group="cosmos_v2_inference", name=f"{parent_name}__inference_only"),
        )
    )


# UR5 co-training experiment (20K).
cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_ur5 = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_ur5",
    dataset=robotwin_cotrain_5tasks_ur5_dataset,
    trainer=_ROBOTWIN_5TASKS_TRAINER_COTRAIN,
)
cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_ur5__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_ur5"
)

# ARX co-training experiment (20K).
cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_arx = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_arx",
    dataset=robotwin_cotrain_5tasks_arx_dataset,
    trainer=_ROBOTWIN_5TASKS_TRAINER_COTRAIN,
)
cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_arx__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_arx"
)

# UR5 pretraining (10K). Phase 1 of pretrain→finetune.
cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_ur5 = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_ur5",
    dataset=robotwin_pretrain_5tasks_ur5_dataset,
    trainer=_ROBOTWIN_5TASKS_TRAINER_PRETRAIN,
)
cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_ur5__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_ur5"
)

# ARX pretraining (10K).
cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_arx = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_arx",
    dataset=robotwin_pretrain_5tasks_arx_dataset,
    trainer=_ROBOTWIN_5TASKS_TRAINER_PRETRAIN,
)
cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_arx__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_arx"
)

# Aloha finetune (10K). Phase 2 of pretrain → finetune. Split into UR vs ARX
# variants because each uses a *different* joint dataset_statistics (the UR
# group's stats vs ARX group's stats) — pairs with its matching pretrain ckpt
# so the action normalization carries over with zero drift. Load the pretrain
# checkpoint at launch via the trainer's load_path / ckpt_path CLI override.
cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_ur = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_ur",
    dataset=robotwin_finetune_5tasks_aloha_after_ur_dataset,
    trainer=_ROBOTWIN_5TASKS_TRAINER_FINETUNE,
)
cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_ur__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_ur"
)

cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_arx = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_arx",
    dataset=robotwin_finetune_5tasks_aloha_after_arx_dataset,
    trainer=_ROBOTWIN_5TASKS_TRAINER_FINETUNE,
)
cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_arx__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_arx"
)


# UR-10 cotrain (20K). UR5 spans 10 tasks; aloha keeps its 5-task slice.
cosmos_predict2_2b_480p_robotwin_cotrain_10tasks_ur5 = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_cotrain_10tasks_ur5",
    dataset=robotwin_cotrain_10tasks_ur5_dataset,
    trainer=_ROBOTWIN_5TASKS_TRAINER_COTRAIN,
)
cosmos_predict2_2b_480p_robotwin_cotrain_10tasks_ur5__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_cotrain_10tasks_ur5"
)

# Progression cotrain (20K). Both UR5 and aloha cover all 29 POOL_5 tasks so
# the model sees every task a progression-style retrieval pool can serve.
cosmos_predict2_2b_480p_robotwin_cotrain_progression_ur5 = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_cotrain_progression_ur5",
    dataset=robotwin_cotrain_progression_ur5_dataset,
    trainer=_ROBOTWIN_5TASKS_TRAINER_COTRAIN,
)
cosmos_predict2_2b_480p_robotwin_cotrain_progression_ur5__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_cotrain_progression_ur5"
)

# Progression pretrain (7.5K) — UR5 only, both clean and randomized splits
# cover the full POOL_5 progression task set. Phase 1 of pretrain→finetune.
cosmos_predict2_2b_480p_robotwin_pretrain_progression_ur5 = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_pretrain_progression_ur5",
    dataset=robotwin_pretrain_progression_ur5_dataset,
    trainer=_ROBOTWIN_PROGRESSION_TRAINER_PRETRAIN,
)
cosmos_predict2_2b_480p_robotwin_pretrain_progression_ur5__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_pretrain_progression_ur5"
)

# Progression finetune (7.5K) on aloha-agilex over the same POOL_5 split —
# phase 2 of pretrain→finetune. Shares the progression group's joint
# dataset_statistics with its matching pretrain so the action normalization
# carries over with zero drift. Load the pretrain checkpoint at launch via the
# trainer's checkpoint.load_path CLI override.
cosmos_predict2_2b_480p_robotwin_finetune_progression_aloha_after_ur = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_finetune_progression_aloha_after_ur",
    dataset=robotwin_finetune_progression_aloha_after_ur_dataset,
    trainer=_ROBOTWIN_PROGRESSION_TRAINER_FINETUNE,
)
cosmos_predict2_2b_480p_robotwin_finetune_progression_aloha_after_ur__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_finetune_progression_aloha_after_ur"
)

# ---------------------------------------------------------------------------
# Real-robot experiments (bg2 bimanual, single third-eye camera)
#
# Single third-eye camera (640×365 post-crop), 20-D rot6d action / proprio.
# Source: /mnt/ddn/dataset/real_robot/recordings_raw/*.npz, split into two
# tasks (the operator switched task halfway through the session):
#   place_bottle_in_plastic_box : eps 0..24  (first 25 npz files, sorted)
#   open_cabinet                : eps 25..49 (last 25)
# Total: 50 demo episodes / ~9.8K frames.
#
# Pairs RealRobotDataset (which drops the wrist slots) with
# _REALROBOT_MODEL_CONFIG (state_t=6, chunk_duration=21 — PushT `pred`-style
# layout minus the value slot).
# ---------------------------------------------------------------------------

_REALROBOT_MODEL_CONFIG = dict(
    # Slots: blank, proprio, primary, action, future_proprio, future_primary = 6.
    # Mirrors PushT `pred` (state_t=7) minus the value slot — we have no reward
    # signal, so the value head would be predicting noise. The future
    # (proprio + primary) slots stay as auxiliary prediction targets for
    # richer representation learning.
    state_t=6,
    min_num_conditional_frames=3,
    max_num_conditional_frames=3,
    # Rot6d: [L_xyz(3) + L_rot6d(6) + L_grip(1) + R_xyz(3) + R_rot6d(6) + R_grip(1)] = 20.
    action_dim=20,
    proprio_dim=20,
    chunk_size=16,
    action_loss_multiplier=16,
    tokenizer=dict(
        # 1 (blank) + 5 (proprio, primary, action, future_proprio, future_primary)
        # * num_duplicates_per_image(4) = 21.
        chunk_duration=21,
    ),
)

# Per-dataset trainer / scheduler. The recording set is small (~9.8K frames /
# 50 episodes); 7K iters is a safer margin than the PushT 5K default while
# still well within the convergence window for a single-task dataset of this
# size. Bump up if eval shows underfitting; drop if you see overfitting on
# the train split.
_REALROBOT_TRAINER = dict(
    max_iter=7_000,
    logging_iter=5,
)
_REALROBOT_SCHEDULER = dict(
    cycle_lengths=[7_000, 100000000000000],
    warm_up_steps=[500, 0],
    f_start=[1e-6, 0.06],
    f_max=[1.0, 0.06],
    f_min=[0.3, 0.06],
)

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
    # Matches the chunk_duration math above. Bigger = more redundant tokens.
    num_duplicates_per_image=4,
    # state_t=6 model config expects future_proprio + future_primary slots.
    predict_future_states=True,
    lazy_video_decompression=True,
    gamma=0.998,
)

# Both tasks together — the default multi-task config we'd train from. Stats
# are computed jointly over all 50 episodes so the action / proprio norm is
# the same scale for both tasks.
realrobot_bg2_all_dataset = L(RealRobotDataset)(
    tasks=["place_bottle_in_plastic_box", "open_cabinet"],
    **_REALROBOT_DATASET_KW,
)

# Single-task variants (kept for ablations / fast iteration). Each has its
# own dataset_statistics.json (separate _cosmos_cache subdir keyed on task
# list hash), so per-task norms are independent.
realrobot_bg2_place_bottle_dataset = L(RealRobotDataset)(
    tasks=["place_bottle_in_plastic_box"],
    **_REALROBOT_DATASET_KW,
)
realrobot_bg2_open_cabinet_dataset = L(RealRobotDataset)(
    tasks=["open_cabinet"],
    **_REALROBOT_DATASET_KW,
)


def _realrobot_experiment(name: str, dataset) -> LazyDict:
    return LazyDict(
        dict(
            defaults=["/experiment/cosmos_predict2_2b_480p_libero", "_self_"],
            trainer=_REALROBOT_TRAINER,
            scheduler=_REALROBOT_SCHEDULER,
            model=L(CosmosPolicyVideo2WorldModel)(config=_REALROBOT_MODEL_CONFIG),
            dataloader_train=L(DataLoader)(
                num_workers=4,
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
                batch_size=8,
                drop_last=True,
            ),
            job=dict(group="cosmos_v2_finetune", name=name),
        )
    )


# Default: train on both tasks at once with a shared stats file. T5 text
# embeddings hold a separate vector per task, so the model conditions on the
# correct language for each sample.
cosmos_predict2_2b_480p_realrobot_bg2 = _realrobot_experiment(
    name="cosmos_predict2_2b_480p_realrobot_bg2",
    dataset=realrobot_bg2_all_dataset,
)
cosmos_predict2_2b_480p_realrobot_bg2__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_realrobot_bg2"
)

# Per-task ablations.
cosmos_predict2_2b_480p_realrobot_bg2_place_bottle = _realrobot_experiment(
    name="cosmos_predict2_2b_480p_realrobot_bg2_place_bottle",
    dataset=realrobot_bg2_place_bottle_dataset,
)
cosmos_predict2_2b_480p_realrobot_bg2_place_bottle__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_realrobot_bg2_place_bottle"
)
cosmos_predict2_2b_480p_realrobot_bg2_open_cabinet = _realrobot_experiment(
    name="cosmos_predict2_2b_480p_realrobot_bg2_open_cabinet",
    dataset=realrobot_bg2_open_cabinet_dataset,
)
cosmos_predict2_2b_480p_realrobot_bg2_open_cabinet__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_realrobot_bg2_open_cabinet"
)


cosmos_predict2_2b_480p_robotwin_pretrain_10tasks_ur5 = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_pretrain_10tasks_ur5",
    dataset=robotwin_pretrain_10tasks_ur5_dataset,
    trainer=_ROBOTWIN_5TASKS_TRAINER_PRETRAIN,
)
cosmos_predict2_2b_480p_robotwin_pretrain_10tasks_ur5__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_pretrain_10tasks_ur5"
)

cosmos_predict2_2b_480p_robotwin_finetune_10tasks_aloha_after_ur = _multi_emb_experiment(
    name="cosmos_predict2_2b_480p_robotwin_finetune_10tasks_aloha_after_ur",
    dataset=robotwin_finetune_10tasks_aloha_after_ur_dataset,
    trainer=_ROBOTWIN_5TASKS_TRAINER_FINETUNE,
)
cosmos_predict2_2b_480p_robotwin_finetune_10tasks_aloha_after_ur__inference_only = _inference_only(
    "cosmos_predict2_2b_480p_robotwin_finetune_10tasks_aloha_after_ur"
)


# ---------------------------------------------------------------------------
# Debug config: single task (place_cans_plasticbox) for pipeline smoke tests.
# Smaller batch, shorter warmup, fewer dataloader workers so it fits a single
# GPU without arguing with the rest of the box.
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox = LazyDict(
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
            config=_ROBOTWIN_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_debug_single_task_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_debug_single_task_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox",
        ),
    )
)

# EE baseline (non-RAG, aloha-only) for place_burger_fries. Comparison
# baseline against the RAG variants (aloha→arx / aloha→piper / franka→ur5).
cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries = LazyDict(
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
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_debug_place_burger_fries_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_debug_place_burger_fries_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries__inference_only",
        ),
    )
)

# Franka-only EE no-RAG baseline. Mirrors the aloha non-RAG config above but
# with embodiment hardcoded to franka_clean_50 — the comparison baseline for
# cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_residual.
cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries_franka = LazyDict(
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
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_debug_place_burger_fries_franka_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_debug_place_burger_fries_franka_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries_franka",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries_franka__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries_franka",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries_franka__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__inference_only",
        ),
    )
)




# ---------------------------------------------------------------------------
# RAG debug config: single task, cross-embodiment retrieval conditioning.
# Mirrors the base debug config but swaps dataset + model config to the RAG
# variant (state_t=15, chunk_duration=57, conditional_frames=10).
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox = LazyDict(
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
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# Residual RAG: same model topology + same (head/L/R/proprio/action) RAG slot
# layout, only the action target changes from absolute to (gt - ret) delta.
# Produces a sibling .json (`delta_dataset_statistics.json`) inside the same
# split-cache dir on first run.
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual = LazyDict(
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
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual__inference_only",
        ),
    )
)


# Top-3 sample variant: identical training shape to the residual config above
# but the dataset draws from top-3 retrievals each step (retrieval_top_k_choice=3).
cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_top3 = LazyDict(
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
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_dataset_residual_top3,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_dataset_residual_top3,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_top3",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_top3__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_top3",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_top3__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# Residual RAG, v2 retrieval pairs: same as the residual config above but
# swaps the retrieval npz to the v2 variant (image+proprio+action_chunk,
# delta_action/time excluded) and bumps the base LR slightly.
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_v2 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=10000,
            logging_iter=5,
        ),
        optimizer=dict(
            lr=2e-4,
        ),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_dataset_residual_v2,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_dataset_residual_v2,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_v2",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_v2__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_v2",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_v2__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# Pool=all variant: same residual training + image npz, but pool spans both
# arx-x5_clean_50 + arx-x5_randomized_500 (550 ep). Forces the model to
# generalize over a much wider distribution of pool trajectories.
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_pool_all = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=dict(
            max_iter=10000,
            logging_iter=5,
        ),
        optimizer=dict(
            lr=2e-4,
        ),
        scheduler=dict(
            cycle_lengths=[20000, 100000000000000],
            warm_up_steps=[500, 0],
            f_start=[1e-6, 0.06],
            f_max=[1.0, 0.06],
            f_min=[0.3, 0.06],
        ),
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_dataset_residual_pool_all,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_dataset_residual_pool_all,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_pool_all",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_pool_all__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_pool_all",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_pool_all__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# UR5 pool variants — four configs (absolute / absolute_v2 / residual /
# residual_v2), each with a sibling __inference_only spec. Identical training
# shape to the arx-x5 versions; only the dataset (and thus the npz /
# pool_embodiment) differs.
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5 = LazyDict(
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
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_ur5_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_ur5_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5__inference_only",
        ),
    )
)


cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_v2 = LazyDict(
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
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_v2,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_v2,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_v2",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_v2__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_v2",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_v2__inference_only",
        ),
    )
)


cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual = LazyDict(
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
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual__inference_only",
        ),
    )
)


cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v2 = LazyDict(
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
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_residual_v2,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_residual_v2,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v2",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v2__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v2",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v2__inference_only",
        ),
    )
)


# v3 variant: identical training setup to v2 — only the retrieval NPZ basename
# differs (v3 uses H=8 action_chunk slice).
cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v3 = LazyDict(
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
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_residual_v3,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_ur5_dataset_residual_v3,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v3",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v3__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v3",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v3__inference_only",
        ),
    )
)


# ── place_burger_fries experiment configs ───────────────────────────────────
# Comparison set: aloha→arx (absolute + residual), aloha→piper (residual),
# franka→ur5 (residual). All with weighted retrieval npz.

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx = LazyDict(
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
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_RAG_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_rag_debug_place_burger_fries_arx_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_burger_fries_arx_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx_residual = LazyDict(
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
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_RAG_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_rag_debug_place_burger_fries_arx_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_burger_fries_arx_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx_residual__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_piper_residual = LazyDict(
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
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_RAG_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_rag_debug_place_burger_fries_piper_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_burger_fries_piper_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_piper_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_piper_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_piper_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_piper_residual__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5 = LazyDict(
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
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_RAG_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_rag_debug_place_burger_fries_ur5_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_burger_fries_ur5_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5_residual = LazyDict(
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
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_RAG_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_rag_debug_place_burger_fries_ur5_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_burger_fries_ur5_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5_residual__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_residual = LazyDict(
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
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_RAG_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_residual__inference_only",
        ),
    )
)

# ── franka query + arx-x5 pool ────────────────────────────────────────────
cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_arx_pool_residual = LazyDict(
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
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_RAG_MODEL_CONFIG),
        checkpoint=dict(save_iter=1000),
        dataloader_train=L(DataLoader)(
            num_workers=4, persistent_workers=True, pin_memory=True,
            dataset=robotwin_rag_debug_place_burger_fries_franka_query_arx_pool_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_burger_fries_franka_query_arx_pool_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True, seed=0,
            ),
            batch_size=16, drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_arx_pool_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_arx_pool_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_arx_pool_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_arx_pool_residual__inference_only",
        ),
    )
)


# ── panda query + UR5 pool (data still being collected) ────────────────────
cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_panda_query_ur5_pool_residual = LazyDict(
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
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_panda_query_ur5_pool_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_panda_query_ur5_pool_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_panda_query_ur5_pool_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_panda_query_ur5_pool_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_panda_query_ur5_pool_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_panda_query_ur5_pool_residual__inference_only",
        ),
    )
)


# ── piper pool variant (data still being collected; npz path placeholder) ───
cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_piper_residual = LazyDict(
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
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        checkpoint=dict(
            save_iter=1000,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_debug_place_cans_plasticbox_piper_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_debug_place_cans_plasticbox_piper_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_piper_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_piper_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_piper_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_piper_residual__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# Joint 5-task RAG training (absolute actions, arx-x5_clean_50 pool).
# Mirrors the single-task RAG debug config but swaps the dataset to a
# ConcatDataset over five RoboTwinRetDataset instances. Training-volume knobs
# (max_iter / scheduler) follow the 3-task non-RAG run since the joint set
# has comparable total-step counts.
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_arx_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_arx_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx__inference_only",
        ),
    )
)


cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_arx_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_arx_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual__inference_only",
        ),
    )
)


cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual__inference_only",
        ),
    )
)


# ── ac030-weighted clean-only UR5 sibling ────────────────────────────────
cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_ac030 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_ac030,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_ac030,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_ac030",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_ac030__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_ac030",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_ac030__inference_only",
        ),
    )
)


# Non-residual UR5 5tasks RAG (weighted NPZs, absolute-action prediction).
cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5__inference_only",
        ),
    )
)


cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_v3 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_v3,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_v3,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_v3",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_v3__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_v3",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_v3__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# `no_future` ablation: weighted RAG ckpt but with the future block dropped
# at training time (slots 11-14 not in the sequence, no future_state /
# future_image loss). Uses `_ROBOTWIN_RAG_MODEL_CONFIG_NO_FUTURE`
# (state_t=11, chunk_duration=41) to match the trimmed dataset.
# Inference must also pass predict_future_states=False so the layout
# matches what the model was trained on.
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_future = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG_NO_FUTURE,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_no_future,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_no_future,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_future",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_future__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_future",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_future__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_future = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG_NO_FUTURE,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_arx_dataset_residual_no_future,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_arx_dataset_residual_no_future,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_future",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_future__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_future",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_future__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# `no_prefilter` ablation training configs (weighted, full-pool top-1 NPZs).
# Same model shape as the canonical weighted residual (state_t=15) — only the
# retrieval-pair source differs (no prefilterN suffix).
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_prefilter = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_no_prefilter,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_no_prefilter,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_prefilter",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_prefilter__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_prefilter",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_prefilter__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_prefilter = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_arx_dataset_residual_no_prefilter,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_arx_dataset_residual_no_prefilter,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_prefilter",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_prefilter__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_prefilter",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_prefilter__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# Top-3 randomized retrieval choice (5tasks joint, residual). Same model /
# trainer / scheduler as the canonical ..._residual config — only the dataset
# differs (retrieval_top_k_choice=3 instead of 1).
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_top3 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_arx_dataset_residual_top3,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_arx_dataset_residual_top3,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_top3",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_top3__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_top3",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_top3__inference_only",
        ),
    )
)


cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_top3 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_top3,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_top3,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_top3",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_top3__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_top3",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_top3__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# Combined-pool (clean + randomized) variants of the 5tasks joint residual
# configs. Same model / trainer / scheduler — only the dataset (pool size,
# NPZ basename) differs.
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_arx_dataset_residual_pool_all,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_arx_dataset_residual_pool_all,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all__inference_only",
        ),
    )
)


cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all__inference_only",
        ),
    )
)


# ---------------------------------------------------------------------------
# Combined pool + top-3 randomized retrieval choice. Stacks the _pool_all
# and _top3 axes: bigger pool (550 ep/task) AND top-3 sampling at training.
# ---------------------------------------------------------------------------

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all_top3 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_arx_dataset_residual_pool_all_top3,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_arx_dataset_residual_pool_all_top3,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all_top3",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all_top3__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all_top3",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all_top3__inference_only",
        ),
    )
)


cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3__inference_only",
        ),
    )
)

# ─── ac030 variant: same shape, different NPZ dataset (weak ac/da + history) ─
cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_ac030,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_ac030,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030__inference_only",
        ),
    )
)

# ─── pair-h5 backed experiments ──────────────────────────────────────────
# Identical to the matching non-pair-h5 configs above but route their
# dataset side through ``pair_h5_root`` so the trainer never opens the
# original pool hdf5 files. Pre-build the pair h5s with
# ``build_retrieved_pair_h5.py`` before launching these.

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_pair_h5,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_pair_h5,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5 = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_libero",
            "_self_",
        ],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(
            config=_ROBOTWIN_RAG_MODEL_CONFIG,
        ),
        dataloader_train=L(DataLoader)(
            num_workers=12,
            persistent_workers=True,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_ac030_pair_h5,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_ac030_pair_h5,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5__inference_only",
        ),
    )
)

# ─── K=1 eager-decode experiments (pair_h5_k1) ───────────────────────────
# Use with PAIR_H5_EAGER_DECODE=1 and dataloader_train.num_workers=0.
cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5_k1 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2_2b_480p_libero", "_self_"],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_RAG_MODEL_CONFIG),
        dataloader_train=L(DataLoader)(
            num_workers=0,
            persistent_workers=False,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_pair_h5_k1,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_pair_h5_k1,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5_k1",
        ),
    )
)
cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5_k1__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5_k1",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5_k1__inference_only",
        ),
    )
)

cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5_k1 = LazyDict(
    dict(
        defaults=["/experiment/cosmos_predict2_2b_480p_libero", "_self_"],
        trainer=_ROBOTWIN_5TASKS_TRAINER,
        scheduler=_ROBOTWIN_5TASKS_SCHEDULER,
        model=L(CosmosPolicyVideo2WorldModel)(config=_ROBOTWIN_RAG_MODEL_CONFIG),
        dataloader_train=L(DataLoader)(
            num_workers=0,
            persistent_workers=False,
            pin_memory=True,
            dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_ac030_pair_h5_k1,
            sampler=L(DistributedSampler)(
                dataset=robotwin_rag_train_5tasks_ur5_dataset_residual_pool_all_top3_ac030_pair_h5_k1,
                num_replicas=L(parallel_state.get_data_parallel_world_size)(),
                rank=L(parallel_state.get_data_parallel_rank)(),
                shuffle=True,
                seed=0,
            ),
            batch_size=16,
            drop_last=True,
        ),
        job=dict(
            group="cosmos_v2_finetune",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5_k1",
        ),
    )
)
cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5_k1__inference_only = LazyDict(
    dict(
        defaults=[
            "/experiment/cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5_k1",
            "_self_",
        ],
        job=dict(
            group="cosmos_v2_inference",
            name="cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5_k1__inference_only",
        ),
    )
)


ALL_ROBOTWIN_CONFIGS = [
    cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox,
    cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__inference_only,
    cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries,
    cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries__inference_only,
    cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries_franka,
    cosmos_predict2_2b_480p_robotwin_debug_place_burger_fries_franka__inference_only,
    cosmos_predict2_2b_480p_robotwin_train_3tasks,
    cosmos_predict2_2b_480p_robotwin_train_3tasks__inference_only,
    cosmos_predict2_2b_480p_robotwin_train_5tasks,
    cosmos_predict2_2b_480p_robotwin_train_5tasks__inference_only,
    cosmos_predict2_2b_480p_robotwin_train_5tasks_no_future,
    cosmos_predict2_2b_480p_robotwin_train_5tasks_no_future__inference_only,
    cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_ur5,
    cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_ur5__inference_only,
    cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_arx,
    cosmos_predict2_2b_480p_robotwin_cotrain_5tasks_arx__inference_only,
    cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_ur5,
    cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_ur5__inference_only,
    cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_arx,
    cosmos_predict2_2b_480p_robotwin_pretrain_5tasks_arx__inference_only,
    cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_ur,
    cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_ur__inference_only,
    cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_arx,
    cosmos_predict2_2b_480p_robotwin_finetune_5tasks_aloha_after_arx__inference_only,
    cosmos_predict2_2b_480p_robotwin_cotrain_10tasks_ur5,
    cosmos_predict2_2b_480p_robotwin_cotrain_10tasks_ur5__inference_only,
    cosmos_predict2_2b_480p_robotwin_cotrain_progression_ur5,
    cosmos_predict2_2b_480p_robotwin_cotrain_progression_ur5__inference_only,
    cosmos_predict2_2b_480p_robotwin_pretrain_progression_ur5,
    cosmos_predict2_2b_480p_robotwin_pretrain_progression_ur5__inference_only,
    cosmos_predict2_2b_480p_robotwin_finetune_progression_aloha_after_ur,
    cosmos_predict2_2b_480p_robotwin_finetune_progression_aloha_after_ur__inference_only,
    cosmos_predict2_2b_480p_robotwin_pretrain_10tasks_ur5,
    cosmos_predict2_2b_480p_robotwin_pretrain_10tasks_ur5__inference_only,
    cosmos_predict2_2b_480p_robotwin_finetune_10tasks_aloha_after_ur,
    cosmos_predict2_2b_480p_robotwin_finetune_10tasks_aloha_after_ur__inference_only,
    cosmos_predict2_2b_480p_realrobot_bg2,
    cosmos_predict2_2b_480p_realrobot_bg2__inference_only,
    cosmos_predict2_2b_480p_realrobot_bg2_place_bottle,
    cosmos_predict2_2b_480p_realrobot_bg2_place_bottle__inference_only,
    cosmos_predict2_2b_480p_realrobot_bg2_open_cabinet,
    cosmos_predict2_2b_480p_realrobot_bg2_open_cabinet__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_v2,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_v2__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_top3,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_top3__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_pool_all,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_residual_pool_all__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_v2,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_v2__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v2,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v2__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v3,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_ur5_residual_v3__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_piper_residual,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_piper_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_panda_query_ur5_pool_residual,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_panda_query_ur5_pool_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx_residual,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_arx_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_piper_residual,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_piper_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5_residual,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_ur5_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_residual,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_ur5_pool_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_arx_pool_residual,
    cosmos_predict2_2b_480p_robotwin_rag_debug_place_burger_fries_franka_query_arx_pool_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_ac030,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_ac030__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_v3,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_v3__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_future,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_future__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_future,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_future__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_prefilter,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_no_prefilter__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_prefilter,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_no_prefilter__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_top3,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_top3__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_top3,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_top3__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all_top3,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_arx_residual_pool_all_top3__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5_k1,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_pair_h5_k1__inference_only,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5_k1,
    cosmos_predict2_2b_480p_robotwin_rag_train_5tasks_ur5_residual_pool_all_top3_ac030_pair_h5_k1__inference_only,
]
