# SPDX-License-Identifier: Apache-2.0
"""
RoboTwin 2.0 dataset with retrieval-augmented conditioning (cross-embodiment).

Mirrors :class:`RoboTwinDataset` for query loading but also loads a separate
*pool* embodiment of the same task and injects a retrieved trajectory as
conditioning, driven by a precomputed NPZ produced by robotwin_rag_tools.

Retrieval NPZ schema (cosine-sim, one row per query frame; see
/mnt/ddn/robotwin_rag_tools/retrieval_results/*.npz):

    query_ids:  (N,)    str  "{embodiment}/episode{N}/{frame}"
    match_ids:  (N, K)  str  "{embodiment}/episode{N}/{frame}"
    match_sims: (N, K)  float32  higher = better

Latent layout (use_wrist_images=True, chunk_size=16, num_duplicates_per_image=4):
Slot order inside each (ret/current/future) block matches the rest of the
codebase's `proprio -> L wrist -> R wrist -> head (-> action)` convention
(see RoboTwinDataset / libero).

     0: blank                            (1 frame)
     1: ret proprio (blank placeholder; value injected downstream)
     2: ret L wrist (1 latent)           - 4 frames, stride-4 sample of 16-frame window
     3: ret R wrist (1 latent)
     4: ret head    (1 latent)
     5: ret action  (blank placeholder; values injected downstream)
     6: current proprio (blank placeholder)
     7: current L wrist (4 dup)
     8: current R wrist (4 dup)
     9: current head    (4 dup)
    10: action placeholder
    11: future proprio (blank placeholder)
    12: future L wrist (4 dup)
    13: future R wrist (4 dup)
    14: future head    (4 dup)
  -> state_t = 15, chunk_duration = 1 + 14 * 4 = 57, conditional_frames = 5
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Per-episode image arrays (head + L wrist + R wrist, all uint8 224×224×3 ×
# num_frames) are the dataset's RAM hog — ~30–50 MB / ep × 600 ep / task × 5
# tasks easily eats 100s of GB per rank. To keep dataset_root construction
# bounded we lazy-load images in __getitem__ instead of decoding all of them
# at __init__. proprio + actions stay eager (tiny, KB-scale) because they're
# needed for dataset statistics, normalization, and (gt − ret) delta_stats.
_LAZY_IMAGE_CACHE_SIZE = int(os.environ.get("LAZY_IMAGE_CACHE_SIZE", "32"))

from cosmos_policy.datasets.dataset_common import (
    build_demo_step_index_mapping,
    calculate_epoch_structure,
    compute_monte_carlo_returns,
    determine_sample_type,
    get_action_chunk_with_padding,
    load_or_compute_dataset_statistics,
    load_or_compute_post_normalization_statistics,
)
from cosmos_policy.datasets.dataset_utils import (
    calculate_dataset_statistics,
    preprocess_image,
    rescale_data,
    resize_images,
)
from cosmos_policy.datasets.robotwin_dataset import (
    DEFAULT_EMBODIMENT,
    _build_ee_action_16d,
    _decode_jpeg_sequence,
    _humanize_task_name,
    _split_cache_dir,
)

np.set_printoptions(precision=3, linewidth=np.inf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _episode_num_from_path(file_path: str) -> int:
    """Extract integer N from '.../data/episode{N}.hdf5'."""
    name = os.path.basename(file_path)
    stem = os.path.splitext(name)[0]
    return int(stem.replace("episode", ""))


def _collect_task_files(dataset_root: str, task: str, embodiment: str) -> list[str]:
    task_dir = os.path.join(dataset_root, task, embodiment, "data")
    assert os.path.isdir(task_dir), (
        f"missing data dir for task={task!r} embodiment={embodiment!r}: {task_dir}"
    )
    files = sorted(glob.glob(os.path.join(task_dir, "*.hdf5")))
    assert files, f"no hdf5 files in {task_dir}"
    return files


def _load_episode_proprio_actions_16d(hdf5_path: str) -> dict:
    """Cheap: read 16-d EE proprio/actions + num_steps from an episode. No JPEG
    decode, no image resize → KB-scale memory per episode. Used at __init__
    so the dataset holds metadata-only for hundreds-to-thousands of episodes;
    image arrays are decoded lazily in :meth:`RoboTwinRetDataset._get_images`.
    """
    with h5py.File(hdf5_path, "r") as f:
        left_endpose = f["/endpose/left_endpose"][()]
        right_endpose = f["/endpose/right_endpose"][()]
        left_gripper = f["/endpose/left_gripper"][()]
        right_gripper = f["/endpose/right_gripper"][()]
        ee_at_t = _build_ee_action_16d(
            left_endpose, right_endpose, left_gripper, right_gripper
        )
        proprio = ee_at_t
        actions = np.concatenate([ee_at_t[1:], ee_at_t[-1:]], axis=0)
        # num_steps from head-camera JPEG count (cheap — only reads dataset
        # shape, doesn't materialize the JPEG bytes).
        num_steps = int(f["/observation/head_camera/rgb"].shape[0])
    assert len(actions) == num_steps, (
        f"frame/action mismatch in {hdf5_path}: {num_steps} frames vs {len(actions)} actions"
    )
    return {
        "proprio": proprio,
        "actions": actions,
        "num_steps": num_steps,
    }


def _load_episode_images(hdf5_path: str, final_image_size: int) -> dict:
    """Heavy: decode + resize all camera JPEG sequences (head + L/R wrist).
    Called on-demand from :meth:`RoboTwinRetDataset._get_images` so each
    rank only holds ~LAZY_IMAGE_CACHE_SIZE episodes' image arrays at any
    time instead of every episode in the (potentially 550-ep) pool."""
    with h5py.File(hdf5_path, "r") as f:
        head = _decode_jpeg_sequence(f["/observation/head_camera/rgb"][()])
        left_wrist = _decode_jpeg_sequence(f["/observation/left_camera/rgb"][()])
        right_wrist = _decode_jpeg_sequence(f["/observation/right_camera/rgb"][()])
    head = resize_images(head, final_image_size)
    left_wrist = resize_images(left_wrist, final_image_size)
    right_wrist = resize_images(right_wrist, final_image_size)
    return {
        "images": head,
        "left_wrist_images": left_wrist,
        "right_wrist_images": right_wrist,
    }


def _load_episode_frames_16d(hdf5_path: str, final_image_size: int) -> dict:
    """Legacy eager loader — kept for any callsite that still wants the full
    episode in one dict (proprio + actions + decoded images). Internally this
    just calls the split lazy/eager helpers. RoboTwinRetDataset itself no
    longer uses this; __init__ uses _load_episode_proprio_actions_16d and
    __getitem__ uses _load_episode_images via the LRU cache."""
    pa = _load_episode_proprio_actions_16d(hdf5_path)
    imgs = _load_episode_images(hdf5_path, final_image_size)
    return {**pa, **imgs}


def _parse_retrieval_id(rid: str) -> tuple[str, int, int]:
    parts = rid.split("/")
    assert len(parts) == 3, f"unexpected retrieval id: {rid!r}"
    embodiment = parts[0]
    ep_num = int(parts[1].replace("episode", ""))
    frame = int(parts[2])
    return embodiment, ep_num, frame


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RoboTwinRetDataset(Dataset):
    """RoboTwin RAG dataset: train on `query_embodiment`, conditioned on a
    retrieved trajectory from `pool_embodiment` (same task)."""

    # Subclass hooks for swapping the action representation. The default
    # 16-d EE space loads /endpose/* and packs (xyz+quat+grip)x2 = 16. The
    # joint-space variant (RoboTwinJointRetDataset) overrides these to load
    # /joint_action/* and pack (6arm+1grip)x2 = 14, with a "__joint" cache
    # suffix so its dataset_statistics.json never collides with the EE one.
    #
    # Loaders are split so the dataset can hold metadata + proprio/actions
    # for every episode (KB-scale) while decoding images only on demand
    # (MB-scale per ep, LRU-cached). Subclasses that need a different proprio
    # encoding override `_LOAD_PROPRIO_FN`; image loading stays generic.
    _LOAD_PROPRIO_FN = staticmethod(_load_episode_proprio_actions_16d)
    _LOAD_IMAGES_FN = staticmethod(_load_episode_images)
    # Legacy alias — preserved for any external caller (e.g. tests / scripts)
    # that still expects the eager combined loader. New code should use the
    # two split helpers above.
    _LOAD_EPISODE_FN = staticmethod(_load_episode_frames_16d)
    _ACTION_DIM = 16
    _CACHE_DIR_SUFFIX = ""

    def __init__(
        self,
        dataset_root: str,
        task: str,
        query_embodiment: str = DEFAULT_EMBODIMENT,
        pool_embodiment: str = "arx-x5_clean_50",
        pool_embodiments: Optional[list[str]] = None,
        retrieval_npz_path: str = "",
        chunk_size: int = 16,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        use_proprio: bool = True,
        use_wrist_images: bool = True,
        use_third_person_images: bool = True,  # signature compat
        num_duplicates_per_image: int = 4,
        return_value_function_returns: bool = False,
        gamma: float = 0.998,
        debug: bool = False,
        retrieval_top_k_choice: int = 1,
        retrieval_dropout_prob: float = 0.0,
        retrieval_image_only_dropout_prob: float = 0.0,
        ret_image_subsample: int = 4,
        force_zero_ret_image: bool = False,
        force_zero_ret_state: bool = False,
        # ── Residual learning ────────────────────────────────────────────────
        # When True, the action target becomes the (gt_action - ret_action)
        # delta — normalized by per-component delta_min/max and stored next to
        # dataset_statistics.json as `delta_dataset_statistics.json`. The model
        # learns to predict the small correction; at inference we add the raw
        # ret_action back. Caveat for EE 16d: naive subtraction is fine for
        # xyz/grip but only an approximation for the 4-d quaternion (assumes
        # small rotational deltas around sign-canonicalized w>=0 quats).
        use_residual_actions: bool = False,
        # When False, slots 11-14 (future_proprio + future L/R/head wrist)
        # are dropped from the latent sequence and their `*_latent_idx`s are
        # set to -1, so the trainer skips both future_state and future_image
        # losses. Reduces state_t from 15 → 11 and chunk_duration 57 → 41.
        # The model config must match (use the
        # `_ROBOTWIN_RAG_MODEL_CONFIG_NO_FUTURE` variant in the experiment
        # registry).
        predict_future_states: bool = True,
        stats_dir: Optional[str] = None,
        max_num_episodes: int = -1,
        # Drop these query episode numbers entirely from training. Use when a
        # specific (task, ep) had a known-bad cross-embodiment retrieval
        # (rotation mismatch / catastrophic jump) and we don't want the
        # policy to see the ep at all — neither as a normal training example
        # nor as a has_ret_data=0 fallback. Set in the experiment config
        # (e.g. ``exclude_query_eps=(38,)`` for grab_roller).
        exclude_query_eps: tuple = (),
        # When True, __init__ stops after raw episodes + retrieval lookup are
        # built — dataset_stats / delta_stats / rescale / post-norm are NOT run.
        # Caller is expected to compute pooled stats across multiple instances
        # and inject them via finalize() and set_post_norm_stats(). Used by the
        # joint-RAG dataset builder so per-task instances share combined stats.
        defer_finalization: bool = False,
        # ── Pre-baked retrieved pair h5 (option B from the pair-h5 plan) ──
        # When non-empty, points at a directory of per-query-ep h5 files
        # built by ``build_retrieved_pair_h5.py`` whose layout is:
        #     <pair_h5_root>/episode<E>.hdf5
        #     /observation/{head,left,right}_camera/rgb  (K, T, IMG_NUM) vlen<uint8>
        #     /endpose/*                                 (K, T, CHUNK_FRAMES, ...)
        #     /joint_action/*                            (K, T, CHUNK_FRAMES, ...)
        # When set, ``_get_retrieved`` reads pool-side data from these files
        # instead of decoding JPEGs from the original pool hdf5s via
        # ``_get_images``. pool_data is still loaded for delta-stats compute
        # (cheap, metadata-only) — only the image decode is short-circuited.
        pair_h5_root: Optional[str] = None,
        # ── Eager image preload (legacy non-pair-h5 path only) ───────────
        # When True, decode every QUERY + POOL episode's JPEGs into uint8
        # numpy arrays at __init__ and stash them keyed by file_path in
        # ``self._eager_image_cache``. ``_get_images`` then short-circuits
        # to a dict.get → no per-step JPEG decode. Forked DataLoader
        # workers share the cache via Linux COW (read-only access keeps
        # the pages shared). Only feasible for small pools (clean-only
        # ~100 eps total fits in ~70 GB / rank); for pool_all=550 the
        # decoded volume is too large and the pair-h5 path is preferred.
        # Ignored when ``pair_h5_root`` is set.
        eager_pool_load: bool = False,
        # Accepted-but-unused (LazyConfig deep-merge compat):
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 1.0,
        success_rollout_sampling_prob: float = 0.0,
        treat_success_rollouts_as_demos: bool = False,
        is_train: Optional[bool] = None,
        **kwargs,
    ) -> None:
        assert use_wrist_images, (
            "RoboTwinRetDataset currently requires use_wrist_images=True "
            "(layout injects L/R wrist slots for current+future+ret)"
        )
        assert chunk_size % num_duplicates_per_image == 0, (
            f"chunk_size ({chunk_size}) must be divisible by "
            f"num_duplicates_per_image ({num_duplicates_per_image})"
        )
        assert chunk_size % ret_image_subsample == 0, (
            f"chunk_size ({chunk_size}) must be divisible by "
            f"ret_image_subsample ({ret_image_subsample})"
        )
        ret_frames_per_latent = chunk_size // ret_image_subsample
        assert ret_frames_per_latent == num_duplicates_per_image, (
            f"ret_image_subsample={ret_image_subsample} yields "
            f"{ret_frames_per_latent} frames per ret-image latent, but tokenizer "
            f"expects {num_duplicates_per_image}. Layout assumes exactly 1 "
            f"latent per ret camera."
        )

        # ── Pool embodiment(s) ────────────────────────────────────────────
        # Legacy single-string ``pool_embodiment`` is still accepted; new
        # multi-pool configs should pass ``pool_embodiments=[...]`` to span
        # multiple splits (e.g. clean_50 + randomized_500). Internally we
        # always work with the list — match_ids in the retrieval npz embed
        # the originating split so dispatch is keyed by (split, ep_num).
        if pool_embodiments is None:
            pool_embodiments = [pool_embodiment]
        elif pool_embodiment and pool_embodiment != pool_embodiments[0]:
            # Legacy ``pool_embodiment`` ignored when explicit list given —
            # the explicit list wins. Warn so silent override is visible.
            print(
                f"[RoboTwinRetDataset] pool_embodiment={pool_embodiment!r} "
                f"shadowed by pool_embodiments={pool_embodiments!r}",
            )
        self.pool_embodiments = tuple(pool_embodiments)
        self.pool_embodiment = self.pool_embodiments[0]  # legacy attr, points at first

        self.dataset_root = dataset_root
        self.task = task
        self.query_embodiment = query_embodiment
        self.retrieval_npz_path = retrieval_npz_path
        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio = use_proprio
        self.use_wrist_images = use_wrist_images
        self.use_third_person_images = use_third_person_images
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.debug = debug
        self.retrieval_top_k_choice = int(retrieval_top_k_choice)
        self.retrieval_dropout_prob = float(retrieval_dropout_prob)
        self.retrieval_image_only_dropout_prob = float(retrieval_image_only_dropout_prob)
        self.ret_image_subsample = int(ret_image_subsample)
        self.force_zero_ret_image = bool(force_zero_ret_image)
        self.force_zero_ret_state = bool(force_zero_ret_state)
        self.use_residual_actions = bool(use_residual_actions)
        self.predict_future_states = bool(predict_future_states)
        self.max_num_episodes = int(max_num_episodes)
        self.exclude_query_eps: set[int] = set(int(e) for e in exclude_query_eps)
        # Pair-h5 mode: when set, pool images / proprio / actions all come
        # from per-query-ep h5 files at ``pair_h5_root/episode<E>.hdf5`` and
        # the original pool hdf5s are never opened. ``pool_data`` and
        # ``_retrieval_lookup`` stay empty, delta_stats is computed by
        # walking pair-h5 files directly.
        self.pair_h5_root: Optional[Path] = Path(pair_h5_root) if pair_h5_root else None
        self._use_pair_h5: bool = self.pair_h5_root is not None
        self.eager_pool_load: bool = bool(eager_pool_load) and not self._use_pair_h5
        # file_path → {images, left_wrist_images, right_wrist_images} when
        # eager_pool_load is on. Populated at __init__, read-only thereafter
        # so DataLoader worker forks share via COW.
        self._eager_image_cache: dict = {}
        # Two pair-h5 modes selected via env var:
        #   PAIR_H5_EAGER_DECODE=1 → decode ALL pool + query JPEGs into
        #     uint8 numpy at __init__ (no per-step decode, dict.get only).
        #     Use with ``num_workers=0`` so cached arrays don't multiply
        #     across DataLoader worker forks. Big RAM (~100 GB per rank
        #     for K=1, 5 task), fast steady-state step.
        #   PAIR_H5_EAGER_DECODE=0 (default) → lazy per-step decode,
        #     ``_pair_h5_files`` is LRU-bounded so the libhdf5 chunk-cache
        #     footprint can't grow unbounded. Safe to use with many
        #     workers and large K.
        self._pair_h5_eager = (
            os.environ.get("PAIR_H5_EAGER_DECODE", "0") not in ("", "0", "false", "False")
        )
        # Safe defaults for lazy (top-K) mode: bound both the h5 file
        # handle dict and the libhdf5 chunk cache so per-worker memory
        # plateaus immediately instead of creeping up over the first ~100
        # steps (the cause of the OOM-at-step-100 regression). Worst case
        # 8 handles × 12 datasets × 64 KiB ≈ 6 MB per worker.
        self._pair_h5_files: OrderedDict = OrderedDict()
        self._pair_h5_files_max = int(os.environ.get("PAIR_H5_FILES_MAX", "8"))
        self._pair_h5_rdcc_nbytes = int(os.environ.get("PAIR_H5_RDCC_NBYTES", str(64 * 1024)))
        self._pool_imgs_cache: dict = {}    # (q_ep, k, t) → {head, left, right}
        self._query_imgs_cache: dict = {}   # q_ep → {head, left, right}

        # Reuse RoboTwinDataset's split-cache so dataset_statistics.json is shared
        # with the base (non-RAG) config.
        self.data_dir = stats_dir or (
            _split_cache_dir(
                dataset_root=dataset_root,
                tasks=[task],
                embodiment=query_embodiment,
            )
            + self._CACHE_DIR_SUFFIX
        )

        # Pool-specific cache key for files whose values depend on the
        # retrieved trajectory distribution (currently only delta_stats).
        # In legacy mode hashing the npz basename uniquely identifies
        # (task, query, pool splits, retrieval scoring). In pair-h5 mode
        # the pair-h5 root path encodes the same — we hash it instead so
        # different pair builds (e.g. ac030 vs default) get distinct
        # delta_stats cache files.
        if self._use_pair_h5:
            pair_token = str(self.pair_h5_root)
            pair_digest = hashlib.sha1(pair_token.encode()).hexdigest()[:8]
            self._pool_cache_key = f"{self.pool_embodiment}_pair_{pair_digest}"
        else:
            npz_basename = os.path.basename(self.retrieval_npz_path).removesuffix(".npz")
            npz_digest = hashlib.sha1(npz_basename.encode()).hexdigest()[:8]
            self._pool_cache_key = f"{self.pool_embodiment}_{npz_digest}"

        # -- Load QUERY episodes (head + L/R wrist + 16d EE) ------------------
        query_files = _collect_task_files(dataset_root, task, query_embodiment)
        if os.environ.get("DEBUGGING", "False").lower() == "true":
            query_files = query_files[:1]
        if self.exclude_query_eps:
            before = len(query_files)
            query_files = [
                f for f in query_files
                if _episode_num_from_path(f) not in self.exclude_query_eps
            ]
            print(
                f"[RoboTwinRetDataset] task={task} dropped "
                f"{before - len(query_files)} query eps via exclude_query_eps="
                f"{sorted(self.exclude_query_eps)}"
            )
        if self.max_num_episodes > 0:
            query_files = query_files[: self.max_num_episodes]

        self.data: dict = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.unique_commands: set = set()
        command = _humanize_task_name(task)
        self.unique_commands.add(command)

        for file in tqdm(query_files, desc=f"Loading QUERY metadata ({query_embodiment})"):
            ep = self._LOAD_PROPRIO_FN(file)
            ep_num = _episode_num_from_path(file)
            returns = (
                compute_monte_carlo_returns(ep["num_steps"], terminal_reward=1.0, gamma=self.gamma)
                if self.return_value_function_returns
                else None
            )
            self.data[self.num_episodes] = dict(
                file_path=file,
                task_name=task,
                ep_num=ep_num,
                # NOTE: images / left_wrist_images / right_wrist_images are
                # NOT stored here. They get decoded on demand from file_path
                # via :meth:`_get_images` (LRU-cached).
                proprio=ep["proprio"],
                actions=ep["actions"],
                command=command,
                num_steps=ep["num_steps"],
                returns=returns.copy() if returns is not None else None,
                success=True,
                is_lazy_video=False,
            )
            self.num_episodes += 1
            self.num_steps += ep["num_steps"]

        # -- Load POOL episode metadata (proprio + actions + num_steps; no  --
        # -- images, those are lazy-loaded from file_path on demand).        --
        # Pool actions/proprio stay raw; normalized inline at retrieve time
        # with the QUERY's stats so injected values share the query's scale.
        # Each split is loaded independently and keyed by (split, ep_num) so
        # ep_num collisions across splits stay unambiguous.
        #
        # In pair-h5 mode we skip this entirely — pool data (images + proprio
        # + actions) is served from the per-query-ep pair h5 files at
        # ``pair_h5_root/episode<E>.hdf5``.
        self.pool_data: dict = {}
        if not self._use_pair_h5:
            for emb in self.pool_embodiments:
                pool_files = _collect_task_files(dataset_root, task, emb)
                if os.environ.get("DEBUGGING", "False").lower() == "true":
                    pool_files = pool_files[:1]
                for file in tqdm(pool_files, desc=f"Loading POOL metadata ({emb})"):
                    ep = self._LOAD_PROPRIO_FN(file)
                    ep_num = _episode_num_from_path(file)
                    self.pool_data[(emb, ep_num)] = dict(
                        file_path=file,
                        proprio=ep["proprio"],
                        actions=ep["actions"],
                        num_steps=ep["num_steps"],
                    )

        # Eager preload: decode every QUERY + POOL episode's JPEGs upfront so
        # __getitem__ never decodes. Threaded so init time stays bounded.
        # Pages are read-only after this returns → DataLoader worker forks
        # share them via Linux COW (no per-worker duplication).
        if self.eager_pool_load:
            import time
            from concurrent.futures import ThreadPoolExecutor, as_completed
            n_threads = int(os.environ.get("EAGER_PRELOAD_THREADS", "32"))
            all_paths = [ep["file_path"] for ep in self.data.values()] + \
                        [pe["file_path"] for pe in self.pool_data.values()]
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                futs = {
                    ex.submit(self._LOAD_IMAGES_FN, fp, self.final_image_size): fp
                    for fp in all_paths
                }
                for fut in as_completed(futs):
                    self._eager_image_cache[futs[fut]] = fut.result()
            print(
                f"[RoboTwinRetDataset] eager_pool_load decoded "
                f"{len(self._eager_image_cache)} episodes "
                f"(query={len(self.data)} + pool={len(self.pool_data)}) "
                f"in {time.time()-t0:.1f}s",
                flush=True,
            )

        # LRU image cache for QUERY-side hdf5 decode (pool side comes from
        # pair h5 in pair mode and never touches this cache). Size from
        # LAZY_IMAGE_CACHE_SIZE env (default 32).
        self._image_cache: OrderedDict = OrderedDict()
        self._image_cache_max = _LAZY_IMAGE_CACHE_SIZE

        # -- Step mapping, stats, normalization -------------------------------
        self._build_step_index_mapping()

        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as fp:
                self.t5_text_embeddings = pickle.load(fp)
        else:
            self.t5_text_embeddings = None

        # Retrieval lookup must be built BEFORE delta-stats computation (which
        # walks gt vs. ret action chunks) and BEFORE rescale_data() rewrites
        # actions/proprio in place — delta stats are computed in raw EE space.
        # In pair-h5 mode the retrieval info is already baked into the per-
        # query-ep pair h5 files, so we skip the npz entirely.
        self._retrieval_lookup: dict = {}
        if not self._use_pair_h5:
            assert retrieval_npz_path and os.path.isfile(retrieval_npz_path), (
                f"retrieval_npz_path must point to an existing file: {retrieval_npz_path!r}"
            )
            self._build_retrieval_lookup(retrieval_npz_path, expected_pools=set(self.pool_embodiments))

        self.dataset_stats: Optional[dict] = None
        self.dataset_stats_post_norm: Optional[dict] = None
        self.delta_stats: Optional[dict] = None

        if not defer_finalization:
            self.dataset_stats = load_or_compute_dataset_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )
            if self.use_residual_actions:
                self.delta_stats = self._load_or_compute_delta_statistics()
            if self.normalize_actions or self.normalize_proprio:
                self._apply_normalization_inplace()
                self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                    data_dir=self.data_dir,
                    data=self.data,
                    calculate_dataset_statistics_func=calculate_dataset_statistics,
                )

        # -- Epoch structure (demo-only) --------------------------------------
        self._rollout_success_total_steps = 0
        self._rollout_failure_total_steps = 0
        self._rollout_total_steps = 0
        self.rollout_episode_metadata: dict = {}
        self.demonstration_sampling_prob = 1.0
        self.success_rollout_sampling_prob = 0.0

        result = calculate_epoch_structure(
            num_steps=self.num_steps,
            rollout_success_total_steps=0,
            rollout_failure_total_steps=0,
            demonstration_sampling_prob=1.0,
            success_rollout_sampling_prob=0.0,
        )
        self.adjusted_demo_count = result["adjusted_demo_count"]
        self.adjusted_success_rollout_count = result["adjusted_success_rollout_count"]
        self.adjusted_failure_rollout_count = result["adjusted_failure_rollout_count"]
        self.epoch_length = result["epoch_length"]

        if self._use_pair_h5:
            n_pair_h5 = sum(
                1 for ep in self.data.values()
                if (self.pair_h5_root / f"episode{ep['ep_num']}.hdf5").is_file()
            )
            print(
                f"[RoboTwinRetDataset] {self.num_episodes} query episodes / "
                f"{self.num_steps} steps | pair_h5_root={self.pair_h5_root} "
                f"({n_pair_h5}/{self.num_episodes} files present) "
                f"eager_decode={self._pair_h5_eager}"
            )
            # Always open pair-h5 handles upfront (cheap with bounded chunk
            # cache). In eager mode we then walk them all and decode every
            # (k, t) JPEG into the in-memory cache so __getitem__ is a pure
            # dict lookup — requires num_workers=0 to keep RAM bounded.
            self._preload_pair_h5_handles()
            if self._pair_h5_eager:
                self._preload_decoded_images()
        else:
            per_split_counts = ", ".join(
                f"{e}:{sum(1 for k in self.pool_data if k[0] == e)}" for e in self.pool_embodiments
            )
            print(
                f"[RoboTwinRetDataset] {self.num_episodes} query episodes / "
                f"{self.num_steps} steps | pool={len(self.pool_data)} episodes "
                f"({per_split_counts}) | retrieval_entries={len(self._retrieval_lookup)}"
            )

    # ------------------------------------------------------------------ utils
    def _get_images(self, file_path: str) -> dict:
        """Return decoded `{images, left_wrist_images, right_wrist_images}` for
        a QUERY (or, in legacy non-pair-h5 mode, POOL) episode. Cache lookup
        order:
          1. ``self._eager_image_cache`` — populated at __init__ when
             ``eager_pool_load=True`` (legacy clean-clean fast path; covers
             both query and pool sides keyed by file_path).
          2. ``self._query_imgs_cache`` — populated by pair-h5 eager preload
             for the query side only.
          3. LRU JPEG decode (lazy fallback).
        """
        cached_eager = self._eager_image_cache.get(file_path)
        if cached_eager is not None:
            return cached_eager
        if self._use_pair_h5 and self._query_imgs_cache:
            # Reverse-resolve ep_num from file_path. Cheap because self.data
            # is small (~50 eps per task instance).
            for ep_data in self.data.values():
                if ep_data.get("file_path") == file_path:
                    cached = self._query_imgs_cache.get(int(ep_data["ep_num"]))
                    if cached is not None:
                        return cached
                    break
        cache = self._image_cache
        cached = cache.get(file_path)
        if cached is not None:
            cache.move_to_end(file_path)
            return cached
        imgs = self._LOAD_IMAGES_FN(file_path, self.final_image_size)
        cache[file_path] = imgs
        if len(cache) > self._image_cache_max:
            cache.popitem(last=False)
        return imgs

    def _build_step_index_mapping(self) -> None:
        result = build_demo_step_index_mapping(self.data)
        self._step_to_episode_map = result["_step_to_episode_map"]
        self._total_steps = result["_total_steps"]

    def _build_retrieval_lookup(self, npz_path: str, expected_pools: set[str]) -> None:
        data = np.load(npz_path, allow_pickle=True)
        query_ids = data["query_ids"]
        match_ids = data["match_ids"]
        match_sims = data["match_sims"].astype(np.float32)

        mismatched_query = 0
        mismatched_pool = 0
        for qi, qid in enumerate(query_ids):
            q_emb, q_ep, q_frame = _parse_retrieval_id(str(qid))
            if q_emb != self.query_embodiment:
                mismatched_query += 1
                continue
            matches: list[tuple[str, int, int, float]] = []
            for ki in range(match_ids.shape[1]):
                m_emb, m_ep, m_frame = _parse_retrieval_id(str(match_ids[qi, ki]))
                if m_emb not in expected_pools:
                    mismatched_pool += 1
                    continue
                matches.append((m_emb, m_ep, m_frame, float(match_sims[qi, ki])))
            if matches:
                self._retrieval_lookup[(q_ep, q_frame)] = matches

        assert self._retrieval_lookup, (
            f"retrieval npz did not produce any usable entries for "
            f"query={self.query_embodiment!r} pool={sorted(expected_pools)!r}: {npz_path}"
        )
        if mismatched_query or mismatched_pool:
            print(
                f"[RoboTwinRetDataset] retrieval filter dropped "
                f"{mismatched_query} query rows (wrong embodiment) and "
                f"{mismatched_pool} match cells (wrong pool embodiment; "
                f"expected one of {sorted(expected_pools)!r})"
            )

    def _apply_normalization_inplace(self) -> None:
        """Rescale self.data['actions']/['proprio'] using self.dataset_stats."""
        assert self.dataset_stats is not None, (
            "_apply_normalization_inplace() requires dataset_stats to be set"
        )
        if self.normalize_actions:
            self.data = rescale_data(self.data, self.dataset_stats, "actions")
        if self.normalize_proprio:
            self.data = rescale_data(self.data, self.dataset_stats, "proprio")

    def compute_raw_deltas(self) -> np.ndarray:
        """Pool (gt_action - ret_action) deltas across this instance's episodes
        using top-1 retrieval. Returns array of shape ``(N, action_dim)`` of
        RAW EE deltas. Must be called before ``rescale_data()`` mutates
        ``self.data['actions']`` in place.

        In pair-h5 mode the pool side comes from the per-query-ep pair h5
        file (k=0 slot = top-1) — its endpose chunks already contain the
        17-frame window we need, so we build ``ret_chunk`` directly from
        those instead of reaching into ``self.pool_data``.
        """
        all_deltas: list[np.ndarray] = []
        for episode_data in self.data.values():
            q_ep = int(episode_data["ep_num"])
            raw_actions = episode_data["actions"]
            T = int(episode_data["num_steps"])

            if self._use_pair_h5:
                h = self._open_pair_h5(q_ep)
                if h is None:
                    continue
                T_pair = int(h.attrs.get("T", 0))
                # Pre-read top-1 slabs once per ep to avoid 4× per-step h5 calls.
                le = h["endpose/left_endpose"][0]          # (T_pair, 17, 7)
                re = h["endpose/right_endpose"][0]         # (T_pair, 17, 7)
                lg = h["endpose/left_gripper"][0]          # (T_pair, 17)
                rg = h["endpose/right_gripper"][0]         # (T_pair, 17)
                for step_idx in range(min(T, T_pair)):
                    gt_chunk = get_action_chunk_with_padding(
                        actions=raw_actions,
                        relative_step_idx=step_idx,
                        chunk_size=self.chunk_size,
                        num_steps=T,
                    ).astype(np.float32)
                    ee = _build_ee_action_16d(
                        le[step_idx], re[step_idx], lg[step_idx], rg[step_idx],
                    ).astype(np.float32)                    # (17, 16)
                    # actions = ee[1:1+chunk_size]; pad with last if short.
                    ret_chunk = ee[1:1 + self.chunk_size]
                    if ret_chunk.shape[0] < self.chunk_size:
                        pad = np.repeat(
                            ret_chunk[-1:], self.chunk_size - ret_chunk.shape[0], axis=0,
                        )
                        ret_chunk = np.concatenate([ret_chunk, pad], axis=0)
                    all_deltas.append(gt_chunk - ret_chunk)
            else:
                for step_idx in range(T):
                    matches = self._lookup_matches(q_ep, step_idx)
                    if matches is None:
                        continue
                    m_emb, m_ep, m_start, _ = matches[0]  # top-1
                    pool_entry = self.pool_data.get((m_emb, m_ep))
                    if pool_entry is None:
                        continue
                    gt_chunk = get_action_chunk_with_padding(
                        actions=raw_actions,
                        relative_step_idx=step_idx,
                        chunk_size=self.chunk_size,
                        num_steps=T,
                    ).astype(np.float32)
                    pT = int(pool_entry["num_steps"])
                    ret_chunk = get_action_chunk_with_padding(
                        actions=pool_entry["actions"],
                        relative_step_idx=min(int(m_start), pT - 1),
                        chunk_size=self.chunk_size,
                        num_steps=pT,
                    ).astype(np.float32)
                    all_deltas.append(gt_chunk - ret_chunk)
        if not all_deltas:
            return np.empty((0, 0), dtype=np.float32)
        return np.concatenate(all_deltas, axis=0)  # (N*chunk, action_dim)

    def _load_or_compute_delta_statistics(self) -> dict:
        """Compute per-component min/max of (gt_action - ret_action) deltas
        across the training set with top-1 retrieval. Cached as
        ``delta_dataset_statistics__pool-<pool_cache_key>.json`` next to
        ``dataset_statistics.json``.

        Pool-keyed filename: deltas depend on the retrieved trajectory pool,
        so pools with different embodiments / splits / retrieval-scoring must
        not share the cache. Key encodes pool_embodiment + sha1(npz basename).

        Operates on RAW (un-normalized) EE values — must run before
        ``rescale_data()`` mutates ``self.data["actions"]`` in place.
        """
        delta_stats_path = os.path.join(
            self.data_dir,
            f"delta_dataset_statistics__pool-{self._pool_cache_key}.json",
        )
        if os.path.exists(delta_stats_path):
            with open(delta_stats_path, "r") as f:
                js = json.load(f)
            print(f"[RoboTwinRetDataset] loaded delta statistics from: {delta_stats_path}")
            return {k: np.array(v, dtype=np.float32) for k, v in js.items()}

        print(
            "[RoboTwinRetDataset] computing delta (gt - ret) statistics over the "
            "dataset (top-1 retrieval) ..."
        )
        all_deltas_arr = self.compute_raw_deltas()
        if all_deltas_arr.size == 0:
            raise RuntimeError(
                "delta-stats computation produced no samples — retrieval lookup "
                "is empty. Check retrieval_npz_path."
            )
        delta_stats = {
            "delta_actions_min": all_deltas_arr.min(axis=0).astype(np.float32),
            "delta_actions_max": all_deltas_arr.max(axis=0).astype(np.float32),
        }
        with open(delta_stats_path, "w") as f:
            json.dump({k: v.tolist() for k, v in delta_stats.items()}, f, indent=4)
        print(f"[RoboTwinRetDataset] delta statistics saved to: {delta_stats_path}")
        print(
            f"  delta_min={delta_stats['delta_actions_min']}\n"
            f"  delta_max={delta_stats['delta_actions_max']}"
        )
        return delta_stats

    def finalize(
        self,
        dataset_stats: dict,
        delta_stats: Optional[dict] = None,
    ) -> None:
        """Apply pre-computed (e.g. multi-task pooled) stats and rescale.

        Used after ``defer_finalization=True`` instances are built, when an
        external caller (the joint-RAG dataset builder) computes combined
        stats across multiple instances.

        Post-norm stats are NOT computed here — call ``set_post_norm_stats``
        after combined post-normalization stats are pooled across instances.
        """
        assert self.dataset_stats is None, (
            "finalize() called twice — instance already finalized"
        )
        self.dataset_stats = dataset_stats
        if self.use_residual_actions:
            assert delta_stats is not None, (
                "use_residual_actions=True requires delta_stats in finalize()"
            )
            self.delta_stats = delta_stats
        if self.normalize_actions or self.normalize_proprio:
            self._apply_normalization_inplace()

    def set_post_norm_stats(self, dataset_stats_post_norm: dict) -> None:
        """Inject pre-computed post-normalization stats. Pairs with
        ``defer_finalization=True`` + ``finalize()`` flow."""
        self.dataset_stats_post_norm = dataset_stats_post_norm

    def _lookup_matches(self, q_ep: int, q_frame: int):
        entry = self._retrieval_lookup.get((q_ep, q_frame))
        if entry is not None:
            return entry
        for offset in (-1, 1, -2, 2, -3, 3, -4, 4):
            entry = self._retrieval_lookup.get((q_ep, q_frame + offset))
            if entry is not None:
                return entry
        return None

    # ─────────────────────── pair-h5 helpers ────────────────────────────────
    def _pair_h5_path(self, q_ep: int) -> Optional[Path]:
        if not self._use_pair_h5:
            return None
        p = self.pair_h5_root / f"episode{q_ep}.hdf5"
        return p if p.is_file() else None

    def _open_pair_h5(self, q_ep: int) -> Optional[h5py.File]:
        """Per-(worker, q_ep) handle with LRU eviction. The handle dict is
        capped at ``self._pair_h5_files_max`` (default 16) so the libhdf5
        chunk-cache footprint can't grow unbounded — the original OOM
        cause was each worker eventually opening all ~50 eps and pinning
        ~12 datasets × default 1 MiB chunk cache = ~600 MB per worker.
        """
        cached = self._pair_h5_files.get(q_ep)
        if cached is not None:
            self._pair_h5_files.move_to_end(q_ep)
            return cached
        path = self._pair_h5_path(q_ep)
        if path is None:
            return None
        h = h5py.File(path, "r", rdcc_nbytes=self._pair_h5_rdcc_nbytes)
        self._pair_h5_files[q_ep] = h
        while len(self._pair_h5_files) > self._pair_h5_files_max:
            oldest, oldest_h = self._pair_h5_files.popitem(last=False)
            try:
                oldest_h.close()
            except Exception:
                pass
        return h

    def _preload_pair_h5_handles(self) -> None:
        """Open every per-query-ep pair h5 file upfront so per-worker
        memory is bounded and predictable. With libhdf5 chunk cache
        shrunk to ``self._pair_h5_rdcc_nbytes`` (default 128 KiB) the
        per-file footprint is ~12 datasets × 128 KiB ≈ 1.5 MiB, so 50
        query eps come in at ~75 MiB per worker.
        """
        if not self._use_pair_h5:
            return
        for ep_data in self.data.values():
            q_ep = int(ep_data["ep_num"])
            self._open_pair_h5(q_ep)
        print(
            f"[RoboTwinRetDataset] pair-h5 handles preloaded: "
            f"{len(self._pair_h5_files)}/{len(self.data)} eps "
            f"(rdcc_nbytes={self._pair_h5_rdcc_nbytes} per dataset)",
            flush=True,
        )

    def _preload_decoded_images(self) -> None:
        """Decode every JPEG in every pair h5 and every QUERY hdf5 into
        uint8 numpy arrays held by ``self._pool_imgs_cache`` /
        ``self._query_imgs_cache``. Eliminates per-step JPEG decode. Use
        with ``dataloader_train.num_workers=0`` — the cached arrays must
        not be duplicated across DataLoader worker forks.
        """
        if not self._use_pair_h5:
            return
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        n_threads = int(os.environ.get("DECODE_PRELOAD_THREADS", "32"))
        t0 = time.time()

        # ---- Query side --------------------------------------------------
        def _decode_query(ep_num: int, fp: str) -> tuple[int, dict]:
            imgs = self._LOAD_IMAGES_FN(fp, self.final_image_size)
            return ep_num, imgs

        q_pairs = [(int(ep_data["ep_num"]), ep_data["file_path"])
                   for ep_data in self.data.values()]
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futs = [ex.submit(_decode_query, ep, fp) for ep, fp in q_pairs]
            for fut in as_completed(futs):
                ep_num, imgs = fut.result()
                self._query_imgs_cache[ep_num] = imgs
        t_q = time.time() - t0

        # ---- Pool side: per (q_ep, k, t) ---------------------------------
        H = self.final_image_size

        def _decode_one_kt(q_ep: int, k: int, t: int,
                            head_b, left_b, right_b) -> tuple[tuple, dict]:
            def _dec(bytes_arr):
                out = []
                for b in bytes_arr:
                    raw = np.frombuffer(np.asarray(b, dtype=np.uint8), dtype=np.uint8)
                    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError("cv2.imdecode returned None")
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if rgb.shape[0] != H or rgb.shape[1] != H:
                        rgb = cv2.resize(rgb, (H, H), interpolation=cv2.INTER_AREA)
                    out.append(rgb)
                return np.stack(out, axis=0)
            return (q_ep, k, t), {
                "head": _dec(head_b),
                "left": _dec(left_b),
                "right": _dec(right_b),
            }

        # Build full work list across all (q_ep, k, t)
        n_total = 0
        for ep_data in self.data.values():
            q_ep = int(ep_data["ep_num"])
            h = self._pair_h5_files.get(q_ep)
            if h is None:
                continue
            K = int(h.attrs["K"])
            T = int(h.attrs["T"])
            n_total += K * T
        print(
            f"[RoboTwinRetDataset] preloading decoded images: "
            f"query={len(self._query_imgs_cache)} eps ({t_q:.1f}s), "
            f"pool entries=(K,T)→{n_total} from {len(self._pair_h5_files)} eps "
            f"(threads={n_threads})...",
            flush=True,
        )

        t1 = time.time()
        # Submit + collect in batches per query ep to bound peak memory of
        # the in-flight future result dicts.
        done = 0
        for ep_data in self.data.values():
            q_ep = int(ep_data["ep_num"])
            h = self._pair_h5_files.get(q_ep)
            if h is None:
                continue
            K = int(h.attrs["K"])
            T = int(h.attrs["T"])
            # Pre-read all JPEG bytes for this ep (single h5 read each).
            head_all = h["observation/head_camera/rgb"][:]   # (K, T, IMG_NUM)
            left_all = h["observation/left_camera/rgb"][:]
            right_all = h["observation/right_camera/rgb"][:]
            with ThreadPoolExecutor(max_workers=n_threads) as ex:
                futs = []
                for k in range(K):
                    for t in range(T):
                        futs.append(ex.submit(
                            _decode_one_kt, q_ep, k, t,
                            head_all[k, t], left_all[k, t], right_all[k, t],
                        ))
                for fut in as_completed(futs):
                    key, imgs = fut.result()
                    self._pool_imgs_cache[key] = imgs
                    done += 1
                    if done % 2000 == 0:
                        print(f"  pool preload {done}/{n_total}  "
                              f"({time.time()-t1:.1f}s)", flush=True)
        print(
            f"[RoboTwinRetDataset] decoded-image preload done — "
            f"query={len(self._query_imgs_cache)} eps + "
            f"pool={len(self._pool_imgs_cache)} (k,t) entries in "
            f"{time.time()-t0:.1f}s",
            flush=True,
        )

    def _read_pair_at(
        self, q_ep: int, q_frame: int, k: int,
    ) -> Optional[dict]:
        """Read the (k, q_frame) slot from this query ep's pair h5 and
        unpack it into the same dict shape ``_get_retrieved`` returns
        (ret_head / ret_left_wrist / ret_right_wrist + ret_actions{,_raw}
        + ret_proprio). Returns None when the file or slot is missing.
        """
        h = self._open_pair_h5(q_ep)
        if h is None:
            return None
        T_pair = int(h.attrs.get("T", 0))
        K_pair = int(h.attrs.get("K", 0))
        if T_pair == 0 or K_pair == 0:
            return None
        t = int(min(max(q_frame, 0), T_pair - 1))
        k = int(min(max(k, 0), K_pair - 1))

        # Images: prefer the eager-decoded cache when available; fall back
        # to on-the-fly JPEG decode if the entry is missing (covers the
        # legacy code path before _preload_decoded_images ran).
        cached_imgs = self._pool_imgs_cache.get((q_ep, k, t))
        if cached_imgs is not None:
            head, left, right = (
                cached_imgs["head"], cached_imgs["left"], cached_imgs["right"],
            )
        else:
            head_bytes = h["observation/head_camera/rgb"][k, t]
            left_bytes = h["observation/left_camera/rgb"][k, t]
            right_bytes = h["observation/right_camera/rgb"][k, t]
            head = self._decode_jpegs_then_resize(head_bytes)
            left = self._decode_jpegs_then_resize(left_bytes)
            right = self._decode_jpegs_then_resize(right_bytes)

        # 17-frame EE chunk → proprio (frame 0) + 16 future actions.
        le = h["endpose/left_endpose"][k, t]         # (17, 7)
        re = h["endpose/right_endpose"][k, t]        # (17, 7)
        lg = h["endpose/left_gripper"][k, t]         # (17,)
        rg = h["endpose/right_gripper"][k, t]        # (17,)
        ee_at_t = _build_ee_action_16d(le, re, lg, rg).astype(np.float32)  # (17, 16)
        proprio = ee_at_t[0].copy()                                          # (16,)
        actions = ee_at_t[1:1 + self.chunk_size].astype(np.float32, copy=True)
        if actions.shape[0] < self.chunk_size:
            pad = np.repeat(actions[-1:], self.chunk_size - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)

        return {
            "ret_head_raw": head,
            "ret_left_raw": left,
            "ret_right_raw": right,
            "ret_proprio_raw": proprio,
            "ret_actions_raw": actions,
        }

    def _decode_jpegs_then_resize(self, jpeg_bytes_arr) -> np.ndarray:
        """``jpeg_bytes_arr`` is an object-dtype array of IMG_NUM vlen<uint8>
        rows (each a JPEG byte buffer). Returns (IMG_NUM, H, H, 3) uint8."""
        out = []
        H = self.final_image_size
        for b in jpeg_bytes_arr:
            raw = np.frombuffer(np.asarray(b, dtype=np.uint8), dtype=np.uint8)
            bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError("cv2.imdecode returned None on pair-h5 JPEG")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[0] != H or rgb.shape[1] != H:
                rgb = cv2.resize(rgb, (H, H), interpolation=cv2.INTER_AREA)
            out.append(rgb)
        return np.stack(out, axis=0)

    def _get_retrieved(self, q_ep: int, q_frame: int) -> dict:
        """Returns retrieved conditioning arrays. Falls back to zeros on miss.

        Two backends:
          * pair-h5 mode (``self._use_pair_h5``): reads pre-baked pool
            images + endpose chunks from the per-query-ep h5 file at
            ``self.pair_h5_root/episode<q_ep>.hdf5``. Index ``[k, q_frame]``
            within that file selects the top-k pool match for query frame
            q_frame; pool side never touches original hdf5.
          * legacy mode: looks up retrieval npz → pool_data + lazy image
            decode (the pre-pair-h5 behavior).
        """
        action_dim = self._ACTION_DIM
        proprio_dim = self._ACTION_DIM
        n_ret_latent_frames = self.num_duplicates_per_image  # 1 latent per camera
        H = self.final_image_size
        zero_frames = np.zeros((n_ret_latent_frames, H, H, 3), dtype=np.uint8)
        zero_action_chunk = np.zeros((self.chunk_size, action_dim), dtype=np.float32)
        zero_proprio = np.zeros(proprio_dim, dtype=np.float32)
        miss = {
            "ret_head": zero_frames.copy(),
            "ret_left_wrist": zero_frames.copy(),
            "ret_right_wrist": zero_frames.copy(),
            "ret_actions": zero_action_chunk.copy(),
            "ret_actions_raw": zero_action_chunk.copy(),
            "ret_proprio": zero_proprio.copy(),
            "has_ret_data": 0,
        }

        if self._use_pair_h5:
            # Random sample k from top-K stored in the pair h5. The stored
            # K may already be <= retrieval_top_k_choice (build script's
            # --max-k) so clamp to what's actually on disk.
            h = self._open_pair_h5(q_ep)
            if h is None:
                return miss
            K_pair = int(h.attrs.get("K", 0))
            if K_pair <= 0:
                return miss
            top_k = max(1, min(self.retrieval_top_k_choice, K_pair))
            k = random.randrange(top_k) if top_k > 1 else 0
            unpacked = self._read_pair_at(q_ep, q_frame, k)
            if unpacked is None:
                return miss
            ret_head = unpacked["ret_head_raw"]
            ret_left = unpacked["ret_left_raw"]
            ret_right = unpacked["ret_right_raw"]
            ret_actions_raw = unpacked["ret_actions_raw"]
            ret_proprio = unpacked["ret_proprio_raw"]
        else:
            matches = self._lookup_matches(q_ep, q_frame)
            if matches is None:
                return miss

            top_k = min(self.retrieval_top_k_choice, len(matches))
            m_emb, m_ep, m_start, _ = random.choice(matches[:top_k]) if top_k > 1 else matches[0]
            pool_entry = self.pool_data.get((m_emb, m_ep))
            if pool_entry is None:
                return miss

            T = pool_entry["num_steps"]
            sampled_idx = [
                max(0, min(m_start + i * self.ret_image_subsample, T - 1))
                for i in range(n_ret_latent_frames)
            ]
            pool_imgs = self._get_images(pool_entry["file_path"])
            ret_head = np.stack([pool_imgs["images"][i] for i in sampled_idx], axis=0)
            ret_left = np.stack([pool_imgs["left_wrist_images"][i] for i in sampled_idx], axis=0)
            ret_right = np.stack([pool_imgs["right_wrist_images"][i] for i in sampled_idx], axis=0)

            ret_actions_raw = get_action_chunk_with_padding(
                actions=pool_entry["actions"],
                relative_step_idx=min(m_start, T - 1),
                chunk_size=self.chunk_size,
                num_steps=T,
            ).astype(np.float32)
            ret_proprio = pool_entry["proprio"][min(m_start, T - 1)].astype(np.float32).copy()

        # Keep both raw + normalized — raw is needed for residual delta target
        # (gt - ret) at full precision; normalized is what gets injected into
        # the model's ret-action latent slot during training/inference.
        ret_actions = ret_actions_raw.copy()
        if self.normalize_actions:
            a_min = self.dataset_stats["actions_min"]
            a_max = self.dataset_stats["actions_max"]
            ret_actions = 2.0 * ((ret_actions - a_min) / (a_max - a_min + 1e-8)) - 1.0
        if self.normalize_proprio:
            p_min = self.dataset_stats["proprio_min"]
            p_max = self.dataset_stats["proprio_max"]
            ret_proprio = 2.0 * ((ret_proprio - p_min) / (p_max - p_min + 1e-8)) - 1.0

        return {
            "ret_head": ret_head,
            "ret_left_wrist": ret_left,
            "ret_right_wrist": ret_right,
            "ret_actions": ret_actions.astype(np.float32),
            "ret_actions_raw": ret_actions_raw.astype(np.float32),
            "ret_proprio": ret_proprio.astype(np.float32),
            "has_ret_data": 1,
        }

    # --------------------------------------------------------------------- API
    def __len__(self) -> int:
        if self.debug:
            return 1
        return self.epoch_length if self.epoch_length > 0 else self.num_steps

    def __getitem__(self, idx: int) -> dict:
        if self.debug:
            idx = 0

        sample_type = determine_sample_type(
            idx, self.adjusted_demo_count, self.adjusted_success_rollout_count
        )
        assert sample_type == "demo", (
            f"RoboTwinRetDataset currently only supports demo sampling, got {sample_type}"
        )

        global_step_idx = idx % self.num_steps
        episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
        episode_data = self.data[episode_idx]
        q_ep = episode_data["ep_num"]

        future_frame_idx = min(
            relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1
        )

        # -- Retrieval --------------------------------------------------------
        ret = self._get_retrieved(q_ep, relative_step_idx)
        has_ret_data = ret["has_ret_data"]
        # has_ret_image tracks whether retrieval frames are meaningful (i.e. not
        # zeroed by miss, dropout, or ablation). Mirror has_ret_data on miss so
        # downstream gates don't see "image flag on, data flag off".
        has_ret_image = has_ret_data
        if has_ret_data and self.retrieval_dropout_prob > 0 and random.random() < self.retrieval_dropout_prob:
            has_ret_data = 0
            has_ret_image = 0
        if (
            has_ret_data
            and self.retrieval_image_only_dropout_prob > 0
            and random.random() < self.retrieval_image_only_dropout_prob
        ):
            has_ret_image = 0

        ret_head = ret["ret_head"]
        ret_left = ret["ret_left_wrist"]
        ret_right = ret["ret_right_wrist"]
        ret_actions = ret["ret_actions"]
        ret_proprio = ret["ret_proprio"]
        if self.force_zero_ret_image or not has_ret_image:
            ret_head = np.zeros_like(ret_head)
            ret_left = np.zeros_like(ret_left)
            ret_right = np.zeros_like(ret_right)
        if self.force_zero_ret_state or not has_ret_data:
            ret_proprio = np.zeros_like(ret_proprio)
        if not has_ret_data:
            ret_actions = np.zeros_like(ret_actions)

        # -- Build latent slot list (see module docstring) --------------------
        # Mix of unique-frame entries (repeat=1) for ret camera latents and
        # duplicated-frame entries (repeat=num_duplicates_per_image) for the
        # rest.  For ret camera latents we append exactly
        # num_duplicates_per_image DISTINCT frames, so after the later
        # repeat_interleave they still pack into exactly one latent.
        # Decode images lazily from the episode's hdf5 (LRU-cached).
        query_imgs = self._get_images(episode_data["file_path"])
        frames: list[np.ndarray] = []
        repeats: list[int] = []
        segment_idx = 0
        ref_image = query_imgs["images"][relative_step_idx]
        blank = np.zeros_like(ref_image)

        # slot 0 - blank sentinel (1 frame only)
        frames.append(blank)
        repeats.append(1)
        segment_idx += 1

        # slot 1 - ret proprio placeholder
        retrieved_state_latent_idx = segment_idx
        frames.append(blank)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # slot 2 - ret L wrist (4 distinct stride-sampled frames)
        retrieved_left_wrist_latent_idx = segment_idx
        for fr in ret_left:
            frames.append(fr)
            repeats.append(1)
        segment_idx += 1

        # slot 3 - ret R wrist
        retrieved_right_wrist_latent_idx = segment_idx
        for fr in ret_right:
            frames.append(fr)
            repeats.append(1)
        segment_idx += 1

        # slot 4 - ret head
        retrieved_head_latent_idx = segment_idx
        for fr in ret_head:
            frames.append(fr)
            repeats.append(1)
        segment_idx += 1

        # slot 5 - ret action placeholder
        retrieved_action_latent_idx = segment_idx
        frames.append(blank)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # slot 6 - current proprio placeholder
        current_proprio_latent_idx = segment_idx
        frames.append(blank)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1
        proprio = (
            episode_data["proprio"][relative_step_idx]
            if self.use_proprio
            else np.zeros_like(episode_data["proprio"][relative_step_idx])
        )

        # slot 7 - current L wrist
        current_wrist_image_latent_idx = segment_idx
        frames.append(query_imgs["left_wrist_images"][relative_step_idx])
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # slot 8 - current R wrist
        current_wrist_image2_latent_idx = segment_idx
        frames.append(query_imgs["right_wrist_images"][relative_step_idx])
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # slot 9 - current head
        current_image_latent_idx = segment_idx
        frames.append(query_imgs["images"][relative_step_idx])
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # slot 10 - action placeholder
        action_latent_idx = segment_idx
        frames.append(blank)
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # slots 11-14 - future block (proprio + L wrist + R wrist + head).
        # Skipped entirely when predict_future_states=False — sequence ends
        # at the action slot. Latent indices set to -1 so the trainer skips
        # both future_state and future_image losses for this batch.
        if self.predict_future_states:
            # slot 11 - future proprio placeholder
            future_proprio_latent_idx = segment_idx
            frames.append(blank)
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
            future_proprio = (
                episode_data["proprio"][future_frame_idx]
                if self.use_proprio
                else np.zeros_like(episode_data["proprio"][future_frame_idx])
            )

            # slot 12 - future L wrist
            future_wrist_image_latent_idx = segment_idx
            frames.append(query_imgs["left_wrist_images"][future_frame_idx])
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

            # slot 13 - future R wrist
            future_wrist_image2_latent_idx = segment_idx
            frames.append(query_imgs["right_wrist_images"][future_frame_idx])
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

            # slot 14 - future head
            future_image_latent_idx = segment_idx
            frames.append(query_imgs["images"][future_frame_idx])
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
        else:
            future_proprio_latent_idx = -1
            future_wrist_image_latent_idx = -1
            future_wrist_image2_latent_idx = -1
            future_image_latent_idx = -1
            future_proprio = np.zeros_like(episode_data["proprio"][future_frame_idx])

        # -- Preprocess + temporal expand -------------------------------------
        all_unique_images = np.stack(frames, axis=0)
        all_unique_images = preprocess_image(
            all_unique_images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        lengths = torch.as_tensor(repeats, dtype=torch.long, device=all_unique_images.device)
        all_images = torch.repeat_interleave(all_unique_images, lengths, dim=1)

        # -- Action chunks + returns ------------------------------------------
        action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )
        next_relative_step_idx = min(
            relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1
        )
        next_action_chunk = get_action_chunk_with_padding(
            actions=episode_data["actions"],
            relative_step_idx=next_relative_step_idx,
            chunk_size=self.chunk_size,
            num_steps=episode_data["num_steps"],
        )

        # -- Residual learning target ----------------------------------------
        # Replace the absolute-action target with a delta normalized by
        # delta_stats. Slot 5 (ret_action) still gets the absolute (normalized)
        # ret_actions as conditioning — the model learns to add a small
        # correction on top.  Skipped on retrieval miss / dropout (no ret_data).
        if self.use_residual_actions and self.delta_stats is not None and has_ret_data:
            # action_chunk arrives in normalized [-1,1] (rescale_data did this
            # in __init__). Reverse to raw, compute delta, then re-normalize
            # with delta-specific min/max.
            if self.normalize_actions:
                a_min = self.dataset_stats["actions_min"]
                a_max = self.dataset_stats["actions_max"]
                raw_action_chunk = (
                    0.5 * (action_chunk + 1.0) * (a_max - a_min) + a_min
                ).astype(np.float32)
            else:
                raw_action_chunk = np.asarray(action_chunk, dtype=np.float32)
            raw_delta = raw_action_chunk - ret["ret_actions_raw"][: self.chunk_size]
            d_min = self.delta_stats["delta_actions_min"]
            d_max = self.delta_stats["delta_actions_max"]
            action_chunk = (
                2.0 * ((raw_delta - d_min) / (d_max - d_min + 1e-8)) - 1.0
            ).astype(np.float32)

        if self.return_value_function_returns:
            value_function_return = episode_data["returns"][future_frame_idx]
            next_future_frame_idx = min(
                next_relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1
            )
            next_value_function_return = episode_data["returns"][next_future_frame_idx]
        else:
            value_function_return = float("-100")
            next_value_function_return = float("-100")

        if self.t5_text_embeddings is None:
            raise RuntimeError(
                "t5_text_embeddings_path was not provided. Run "
                "cosmos_policy.datasets.save_robotwin_t5_text_embeddings first."
            )
        t5_emb = torch.squeeze(self.t5_text_embeddings[episode_data["command"]])

        return {
            "video": all_images,
            "command": episode_data["command"],
            "actions": action_chunk,
            "t5_text_embeddings": t5_emb,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": proprio,
            "future_proprio": future_proprio,
            "__key__": idx,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            # Base latent indices (mirror RoboTwinDataset names)
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": -1,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_wrist_image2_latent_idx": current_wrist_image2_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_wrist_image2_latent_idx": future_wrist_image2_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
            # Retrieval-specific
            "retrieved_head_latent_idx": retrieved_head_latent_idx,
            "retrieved_left_wrist_latent_idx": retrieved_left_wrist_latent_idx,
            "retrieved_right_wrist_latent_idx": retrieved_right_wrist_latent_idx,
            "retrieved_state_latent_idx": retrieved_state_latent_idx,
            "retrieved_action_latent_idx": retrieved_action_latent_idx,
            "retrieved_actions": ret_actions,
            "retrieved_proprio": ret_proprio,
            "has_ret_data": has_ret_data,
            "has_ret_image": has_ret_image,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/mnt/ddn/dataset/RoboTwin2.0/dataset")
    parser.add_argument("--task", type=str, default="place_cans_plasticbox")
    parser.add_argument("--query_embodiment", type=str, default=DEFAULT_EMBODIMENT)
    parser.add_argument("--pool_embodiment", type=str, default="")
    parser.add_argument(
        "--pool_embodiments", type=str, nargs="+", default=["arx-x5_clean_50"],
        help="one or more pool splits (defaults to a single arx-x5_clean_50)",
    )
    parser.add_argument(
        "--retrieval_npz",
        type=str,
        default=(
            "/mnt/ddn/robotwin_rag_tools/retrieval_results/"
            "retrieval_weighted_place_cans_plasticbox__q-aloha-agilex_clean_50__p-arx-x5_clean_50.npz"
        ),
    )
    parser.add_argument("--chunk_size", type=int, default=16)
    args = parser.parse_args()

    ds = RoboTwinRetDataset(
        dataset_root=args.dataset_root,
        task=args.task,
        query_embodiment=args.query_embodiment,
        pool_embodiment=args.pool_embodiment,
        pool_embodiments=args.pool_embodiments,
        retrieval_npz_path=args.retrieval_npz,
        chunk_size=args.chunk_size,
        t5_text_embeddings_path="",
        normalize_actions=False,
        normalize_proprio=False,
        num_duplicates_per_image=4,
        ret_image_subsample=4,
    )
    print(f"len={len(ds)}  episodes={ds.num_episodes}  pool={len(ds.pool_data)}")
    # Inject a stub T5 mapping so __getitem__ runs without loading real embeddings.
    ds.t5_text_embeddings = {ds.data[0]["command"]: torch.zeros(1, 512, 1024)}
    sample = ds[0]
    for k, v in sample.items():
        if k == "video":
            continue
        shape = tuple(v.shape) if hasattr(v, "shape") else v
        print(f"  {k}: {shape}")
    print(f"video: {tuple(sample['video'].shape)} {sample['video'].dtype}")


if __name__ == "__main__":
    _main()
