# SPDX-License-Identifier: Apache-2.0
"""
RoboTwin 2.0 simulation benchmark dataloader for Cosmos Policy training.

Reads the *original* RoboTwin HDF5 layout in-place (no regeneration step).
Emits absolute end-effector poses in their native representation — per arm
[xyz(3), quat_wxyz(4), gripper(1)] = 8 — and concatenates left+right for a
16-dim action/proprio. Quaternion is kept in scalar-first (w,x,y,z) order with
sign canonicalized (w >= 0) to remove the double-cover discontinuity.

Language instruction is fixed per task via humanized folder name
(e.g., "pick_diverse_bottles" -> "pick diverse bottles").

Two supported entrypoints:

(A) Multi-task root (preferred). Point at the dataset root and pick tasks:
    <dataset_root>/<task>/<embodiment>/data/episode{N}.hdf5
    dataset_root="/mnt/ddn/dataset/RoboTwin2.0/dataset"
    tasks=["pick_diverse_bottles"]        # or a .txt/.yaml file
    embodiment="aloha-agilex_clean_50"    # exact match, so ur5_* never leaks in

(B) Legacy single-task dir (back-compat). `data_dir` points at one
    embodiment dir that already contains a `data/` subdir.

Expected HDF5 schema inside each episode file:
    /observation/head_camera/rgb      (T,)    JPEG bytes as |S<maxlen>
    /observation/left_camera/rgb      (T,)    JPEG bytes
    /observation/right_camera/rgb     (T,)    JPEG bytes
    /endpose/left_endpose             (T, 7)  [xyz, quat(w,x,y,z)]
    /endpose/right_endpose            (T, 7)
    /endpose/left_gripper             (T,)    normalized 0~1
    /endpose/right_gripper            (T,)

Run a quick sanity check (prints one sample dict):
    python -m cosmos_policy.datasets.robotwin_dataset \
        --dataset_root /mnt/ddn/dataset/RoboTwin2.0/dataset \
        --tasks pick_diverse_bottles
"""

from __future__ import annotations

import glob
import hashlib
import os
import pickle
from typing import Iterable, Optional, Sequence

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

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
    get_hdf5_files,
    preprocess_image,
    rescale_data,
    resize_images,
)

DEFAULT_EMBODIMENT = "aloha-agilex_clean_50"

np.set_printoptions(precision=3, linewidth=np.inf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_task_list(tasks: Optional[Sequence[str] | str]) -> Optional[list[str]]:
    """Normalize the `tasks` argument to a concrete list of folder names.

    Accepts: None (= all tasks), a list/tuple of folder names, or a path to a
    text file with one task per line (`#` starts a comment, blank lines ignored).
    """
    if tasks is None:
        return None
    if isinstance(tasks, str):
        # Treat as a file path if it exists; else as a single task name.
        if os.path.isfile(tasks):
            resolved: list[str] = []
            with open(tasks, "r") as fp:
                for line in fp:
                    line = line.split("#", 1)[0].strip()
                    if line:
                        resolved.append(line)
            assert resolved, f"task list file {tasks} has no non-empty entries"
            return resolved
        return [tasks]
    resolved = [t.strip() for t in tasks if t and t.strip()]
    assert resolved, "tasks must contain at least one entry"
    return resolved


def _collect_robotwin_files(
    dataset_root: str,
    tasks: Optional[list[str]],
    embodiment: str,
) -> list[str]:
    """Discover HDF5 episode files under the multi-task root layout.

    Layout: <dataset_root>/<task>/<embodiment>/data/*.hdf5
    """
    assert os.path.isdir(dataset_root), f"dataset_root not a directory: {dataset_root}"

    if tasks is None:
        tasks = sorted(
            name for name in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, name, embodiment, "data"))
        )
        assert tasks, (
            f"no tasks under {dataset_root} have an '{embodiment}/data/' subdir — "
            f"check dataset_root and embodiment"
        )

    files: list[str] = []
    for task in tasks:
        task_dir = os.path.join(dataset_root, task, embodiment, "data")
        assert os.path.isdir(task_dir), (
            f"missing data dir for task={task!r} embodiment={embodiment!r}: {task_dir}"
        )
        task_files = sorted(glob.glob(os.path.join(task_dir, "*.hdf5")))
        assert task_files, f"no hdf5 files in {task_dir}"
        files.extend(task_files)
    return files


def _split_cache_dir(
    dataset_root: str,
    tasks: list[str],
    embodiment: str,
) -> str:
    """Per-split cache directory for dataset stats + T5 pkl.

    Keyed on (embodiment, sorted tasks) so different splits don't overwrite
    each other's `dataset_statistics.json` / `t5_embeddings.pkl`.
    """
    tasks_key = ",".join(sorted(tasks))
    digest = hashlib.sha1(tasks_key.encode()).hexdigest()[:8]
    name = f"{embodiment}__{len(tasks)}tasks_{digest}"
    path = os.path.join(dataset_root, "_cosmos_cache", name)
    os.makedirs(path, exist_ok=True)
    return path


def _humanize_task_name(task_name: str) -> str:
    """`pick_diverse_bottles` -> `pick diverse bottles`. Fixed per task.

    Chosen deliberately so language instruction is deterministic across eval
    runs and matches ALOHADataset's own convention (aloha_dataset.py:325).
    """
    return task_name.replace("_", " ")


def _task_name_from_file(file_path: str) -> str:
    """Extract task_name from the RoboTwin directory structure.

    Path shape: <root>/<task_name>/<embodiment-config>/data/episode{N}.hdf5
    so index -4 from the leaf gives the task directory.
    """
    return file_path.split(os.sep)[-4]


def _decode_jpeg_sequence(jpeg_array: np.ndarray) -> np.ndarray:
    """Decode an (T,) array of JPEG byte strings to (T, H, W, 3) uint8 RGB.

    RoboTwin stores frames as fixed-width |S<maxlen> byte strings encoded via
    cv2.imencode (which round-trips through cv2.imdecode without channel
    re-ordering). The bytes RoboTwin produced from a real RGB array therefore
    decode straight back to that RGB array — NO BGR->RGB conversion is
    required. The previous redundant cvtColor turned RGB into BGR and is the
    root cause of the long-standing color swap visible in retrieved.mp4 /
    generated.mp4 (and a hidden inference-time channel mismatch with the
    runtime's ``cams["head_camera"]["rgb"]`` obs).
    """
    num_frames = len(jpeg_array)
    decoded: list[np.ndarray] = []
    for i in range(num_frames):
        buf = jpeg_array[i]
        # Trailing null padding from the fixed-width storage — harmless to JPEG
        # decoder but we strip to be explicit.
        buf = buf.rstrip(b"\x00") if isinstance(buf, (bytes, bytearray)) else bytes(buf).rstrip(b"\x00")
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imdecode failed at frame {i} (buf_len={len(buf)})")
        decoded.append(img)
    return np.stack(decoded, axis=0)


def _strip_jpeg_buffers(jpeg_array: np.ndarray) -> list[bytes]:
    """Strip the fixed-width null padding off each JPEG buffer and return a list.

    Storing a Python list[bytes] instead of the (T, maxlen) byte block avoids
    paying for per-frame padding overhead and shrinks the lazy retention
    footprint by the average compression ratio the fixed-width pre-allocates.
    """
    out: list[bytes] = []
    for i in range(len(jpeg_array)):
        buf = jpeg_array[i]
        buf = buf.rstrip(b"\x00") if isinstance(buf, (bytes, bytearray)) else bytes(buf).rstrip(b"\x00")
        out.append(bytes(buf))
    return out


def _decode_jpeg_single(buf: bytes, final_image_size: int) -> np.ndarray:
    """Decode + resize a single pre-stripped JPEG buffer."""
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"cv2.imdecode failed (buf_len={len(buf)})")
    if img.shape[:2] != (final_image_size, final_image_size):
        img = np.array(Image.fromarray(img).resize((final_image_size, final_image_size)))
    return img


def _build_ee_action_16d(
    left_endpose: np.ndarray,   # (T, 7) [xyz, quat wxyz]
    right_endpose: np.ndarray,  # (T, 7)
    left_gripper: np.ndarray,   # (T,)
    right_gripper: np.ndarray,  # (T,)
) -> np.ndarray:
    """Pack [L_xyz(3), L_quat_wxyz(4), L_grip(1), R_xyz(3), R_quat_wxyz(4), R_grip(1)] = (T, 16).

    Native RoboTwin absolute-EE representation — no rotation reparameterization.
    Deploy side hands this straight to TASK_ENV.take_action(action_type='ee').
    """
    out = np.concatenate(
        [
            left_endpose[:, :3].astype(np.float32),
            left_endpose[:, 3:7].astype(np.float32),
            left_gripper.astype(np.float32)[:, None],
            right_endpose[:, :3].astype(np.float32),
            right_endpose[:, 3:7].astype(np.float32),
            right_gripper.astype(np.float32)[:, None],
        ],
        axis=-1,
    )
    assert out.shape[-1] == 16, f"expected 16-dim action, got {out.shape}"
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RoboTwinDataset(Dataset):
    """RoboTwin 2.0 dataset loader with native 16-dim EE pose signal.

    Action/proprio are emitted as [xyz(3) + quat_wxyz(4) + grip(1)] * 2 arms = 16.
    Output dict per sample is structurally the same as ALOHADataset.__getitem__
    (keys/indices) so the tokenizer/trainer stack is reused; only the last dim
    differs.

    Rollout/value-function training is intentionally *not* supported in this
    first pass — if you need best-of-N training later, extend along
    ALOHADataset's rollout_data_dir path.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        chunk_size: int = 16,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = False,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 8,
        return_value_function_returns: bool = False,
        gamma: float = 0.998,
        debug: bool = False,
        # Multi-task root entrypoint (preferred):
        dataset_root: Optional[str] = None,
        tasks: Optional[Sequence[str] | str] = None,
        embodiment: str = DEFAULT_EMBODIMENT,
        stats_dir: Optional[str] = None,
        # Kept for signature compatibility with ALOHADataset/LIBERODataset:
        use_third_person_images: bool = True,
        use_wrist_images: bool = True,
        is_train: Optional[bool] = None,
        # When True, skip per-instance dataset_statistics computation +
        # rescale + post-norm-stats so an external builder can compute joint
        # stats across multiple instances and inject them via finalize() and
        # set_post_norm_stats(). Mirrors RoboTwinRetDataset's defer flow.
        defer_finalization: bool = False,
        # When True, retain only the raw JPEG byte buffers in RAM and decode
        # individual frames on demand inside __getitem__. Cuts the resident
        # set size by ~7x at the cost of a few-ms decode per sample —
        # absorbed by the dataloader workers in practice. Required for the
        # 10-task UR pipeline (5750 episodes would otherwise need ~360 GB).
        lazy_video_decompression: bool = False,
        # When False, drop the 4 future slots (future_proprio + future
        # head/L/R wrist) from the latent sequence and set their
        # ``*_latent_idx``s to -1 → trainer skips both future_state and
        # future_image losses. Cuts state_t from 10 → 6 and chunk_duration
        # from 37 → 21. Pair with the matching ``_ROBOTWIN_MODEL_CONFIG_NO_FUTURE``
        # model config.
        predict_future_states: bool = True,
        # Accepted-but-unused: injected by LazyConfig deep-merge when this
        # experiment inherits from a LIBERO/ALOHA base config. RoboTwin v1
        # has no rollout path (see class docstring), so these are silently
        # absorbed rather than forcing every caller to override them.
        rollout_data_dir: str = "",
        demonstration_sampling_prob: float = 1.0,
        success_rollout_sampling_prob: float = 0.0,
        treat_success_rollouts_as_demos: bool = False,
        **kwargs,
    ):
        assert (data_dir is not None) or (dataset_root is not None), (
            "must pass either data_dir (legacy single-task dir) or dataset_root "
            "(multi-task root, preferred)"
        )

        self.chunk_size = chunk_size
        self.final_image_size = final_image_size
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.predict_future_states = bool(predict_future_states)
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = num_duplicates_per_image
        self.return_value_function_returns = return_value_function_returns
        self.gamma = gamma
        self.debug = debug
        self.embodiment = embodiment
        self.lazy_video_decompression = lazy_video_decompression

        if dataset_root is not None:
            self.dataset_root = dataset_root
            self.tasks = _resolve_task_list(tasks)
            hdf5_files = _collect_robotwin_files(
                dataset_root=dataset_root,
                tasks=self.tasks,
                embodiment=embodiment,
            )
            # After resolution we know the concrete task set; fill it in if None.
            if self.tasks is None:
                self.tasks = sorted({_task_name_from_file(p) for p in hdf5_files})
            self.data_dir = stats_dir or _split_cache_dir(
                dataset_root=dataset_root,
                tasks=self.tasks,
                embodiment=embodiment,
            )
        else:
            # Legacy: data_dir points directly at a dir with episode hdf5s below it.
            self.dataset_root = None
            self.tasks = None
            self.data_dir = stats_dir or data_dir
            hdf5_files = get_hdf5_files(data_dir, is_train=is_train)
        assert len(hdf5_files) > 0, (
            f"No HDF5 files found (data_dir={data_dir!r}, dataset_root={dataset_root!r}, "
            f"tasks={tasks!r}, embodiment={embodiment!r})"
        )

        if os.environ.get("DEBUGGING", "False").lower() == "true":
            hdf5_files = hdf5_files[:1]

        self.data: dict = {}
        self.num_episodes = 0
        self.num_steps = 0
        self.unique_commands: set = set()

        for file in tqdm(hdf5_files, desc="Loading RoboTwin episodes"):
            with h5py.File(file, "r") as f:
                # ---- actions / proprio: absolute EE -> native 16-dim ----
                # endpose[t] is the current EE pose at step t (= proprio).
                # In an absolute-EE control regime, the action issued at step t
                # is the target pose the policy should command so the robot
                # transitions into endpose[t+1] — i.e. action_t == endpose[t+1].
                # So we left-shift by one and pad the terminal step with the
                # final pose (hold last). get_action_chunk_with_padding then
                # naturally returns endpose[t+1 : t+1+chunk_size] for each t.
                left_endpose = f["/endpose/left_endpose"][()]
                right_endpose = f["/endpose/right_endpose"][()]
                left_gripper = f["/endpose/left_gripper"][()]
                right_gripper = f["/endpose/right_gripper"][()]
                ee_at_t = _build_ee_action_16d(
                    left_endpose, right_endpose, left_gripper, right_gripper
                )
                proprio = ee_at_t  # proprio[t] = endpose[t]
                actions = np.concatenate([ee_at_t[1:], ee_at_t[-1:]], axis=0)

                # ---- images: either eager-decode now or stash JPEG bytes ----
                head_raw = f["/observation/head_camera/rgb"][()]
                left_raw = f["/observation/left_camera/rgb"][()]
                right_raw = f["/observation/right_camera/rgb"][()]

                if self.lazy_video_decompression:
                    images_jpeg = _strip_jpeg_buffers(head_raw)
                    left_wrist_images_jpeg = _strip_jpeg_buffers(left_raw)
                    right_wrist_images_jpeg = _strip_jpeg_buffers(right_raw)
                    images = left_wrist_images = right_wrist_images = None
                    episode_num_steps = len(images_jpeg)
                else:
                    images = resize_images(_decode_jpeg_sequence(head_raw), final_image_size)
                    left_wrist_images = resize_images(_decode_jpeg_sequence(left_raw), final_image_size)
                    right_wrist_images = resize_images(_decode_jpeg_sequence(right_raw), final_image_size)
                    images_jpeg = left_wrist_images_jpeg = right_wrist_images_jpeg = None
                    episode_num_steps = len(images)

                # Defensive: frames / actions should agree.
                assert episode_num_steps == len(actions), (
                    f"Frame/action length mismatch in {file}: "
                    f"{episode_num_steps} frames vs {len(actions)} actions"
                )

            # ---- command: fixed per-task via humanize ----
            task_name = _task_name_from_file(file)
            command = _humanize_task_name(task_name)
            self.unique_commands.add(command)

            if self.return_value_function_returns:
                returns = compute_monte_carlo_returns(
                    episode_num_steps, terminal_reward=1.0, gamma=self.gamma
                )
            else:
                returns = None

            episode_entry: dict = dict(
                file_path=file,
                task_name=task_name,
                proprio=proprio,
                actions=actions,
                command=command,
                num_steps=episode_num_steps,
                returns=returns.copy() if returns is not None else None,
                success=True,
                is_lazy_video=self.lazy_video_decompression,
            )
            if self.lazy_video_decompression:
                episode_entry["images_jpeg"] = images_jpeg
                episode_entry["left_wrist_images_jpeg"] = left_wrist_images_jpeg
                episode_entry["right_wrist_images_jpeg"] = right_wrist_images_jpeg
            else:
                episode_entry["images"] = images
                episode_entry["left_wrist_images"] = left_wrist_images
                episode_entry["right_wrist_images"] = right_wrist_images
            self.data[self.num_episodes] = episode_entry
            self.num_episodes += 1
            self.num_steps += episode_num_steps

        # ---- index map, stats, normalization ----
        self._build_step_index_mapping()

        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as fp:
                self.t5_text_embeddings = pickle.load(fp)
        else:
            self.t5_text_embeddings = None  # will error in __getitem__ if used

        self._finalized = False
        if defer_finalization:
            # External builder will call finalize() / set_post_norm_stats()
            # with joint stats. Leave self.data un-rescaled and
            # dataset_stats / dataset_stats_post_norm as None for now.
            self.dataset_stats = None
            self.dataset_stats_post_norm = None
        else:
            self.dataset_stats = load_or_compute_dataset_statistics(
                data_dir=self.data_dir,
                data=self.data,
                calculate_dataset_statistics_func=calculate_dataset_statistics,
            )

            if self.normalize_actions or self.normalize_proprio:
                if self.normalize_actions:
                    self.data = rescale_data(self.data, self.dataset_stats, "actions")
                if self.normalize_proprio:
                    self.data = rescale_data(self.data, self.dataset_stats, "proprio")
                self.dataset_stats_post_norm = load_or_compute_post_normalization_statistics(
                    data_dir=self.data_dir,
                    data=self.data,
                    calculate_dataset_statistics_func=calculate_dataset_statistics,
                )
            self._finalized = True

        # No rollout path in v1 — zero out counts so epoch math reduces to num_steps.
        self._rollout_success_total_steps = 0
        self._rollout_failure_total_steps = 0
        self._rollout_total_steps = 0
        self.rollout_episode_metadata: dict = {}
        self.rollout_data: dict = {}
        self.rollout_num_episodes = 0
        self.rollout_num_steps = 0
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

    def finalize(self, dataset_stats: dict) -> None:
        """Inject externally-computed stats and rescale ``self.data`` in place.

        Used after ``defer_finalization=True`` instances are built, when a
        joint-stats helper has merged stats across multiple instances. After
        this call, the instance behaves identically to one constructed with
        ``defer_finalization=False`` (modulo post_norm_stats which is set
        separately via ``set_post_norm_stats``).
        """
        assert not self._finalized, "finalize() called twice — already finalized"
        self.dataset_stats = dataset_stats
        if self.normalize_actions:
            self.data = rescale_data(self.data, self.dataset_stats, "actions")
        if self.normalize_proprio:
            self.data = rescale_data(self.data, self.dataset_stats, "proprio")
        self._finalized = True

    def set_post_norm_stats(self, dataset_stats_post_norm: dict) -> None:
        """Inject post-normalization stats computed externally."""
        self.dataset_stats_post_norm = dataset_stats_post_norm

    # ------------------------------------------------------------------ utils
    def _build_step_index_mapping(self):
        result = build_demo_step_index_mapping(self.data)
        self._step_to_episode_map = result["_step_to_episode_map"]
        self._total_steps = result["_total_steps"]

    def _get_frame(self, episode_data: dict, cam_key: str, step_idx: int) -> np.ndarray:
        """Return one decoded (final_image_size, final_image_size, 3) uint8 frame.

        Eager path: index the pre-decoded array directly.
        Lazy path: decode the JPEG byte buffer for `step_idx` on demand.
        """
        if episode_data.get("is_lazy_video", False):
            buf = episode_data[f"{cam_key}_jpeg"][step_idx]
            return _decode_jpeg_single(buf, self.final_image_size)
        return episode_data[cam_key][step_idx]

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
        # v1 has no rollout data, so the sampler should always land on demos.
        assert sample_type == "demo", (
            f"RoboTwinDataset currently only supports demo sampling, got {sample_type}"
        )

        global_step_idx = idx % self.num_steps
        episode_idx, relative_step_idx = self._step_to_episode_map[global_step_idx]
        episode_data = self.data[episode_idx]

        future_frame_idx = min(
            relative_step_idx + self.chunk_size, episode_data["num_steps"] - 1
        )

        # -------- segment list for tokenizer (mirror ALOHA layout) --------
        frames: list[np.ndarray] = []
        repeats: list[int] = []
        segment_idx = 0

        # Indices to be filled as we append
        action_latent_idx = -1
        value_latent_idx = -1
        current_proprio_latent_idx = -1
        current_wrist_image_latent_idx = -1
        current_wrist_image2_latent_idx = -1
        current_image_latent_idx = -1
        future_proprio_latent_idx = -1
        future_wrist_image_latent_idx = -1
        future_wrist_image2_latent_idx = -1
        future_image_latent_idx = -1

        ref_image = self._get_frame(episode_data, "images", relative_step_idx)

        # blank first frame
        frames.append(np.zeros_like(ref_image))
        repeats.append(1)
        segment_idx += 1

        # current proprio (blank image placeholder; real proprio injected later)
        if self.use_proprio:
            current_proprio_latent_idx = segment_idx
            frames.append(np.zeros_like(ref_image))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
            proprio = episode_data["proprio"][relative_step_idx]
        else:
            proprio = np.zeros_like(episode_data["proprio"][relative_step_idx])

        # current left wrist
        current_wrist_image_latent_idx = segment_idx
        frames.append(self._get_frame(episode_data, "left_wrist_images", relative_step_idx))
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # current right wrist
        current_wrist_image2_latent_idx = segment_idx
        frames.append(self._get_frame(episode_data, "right_wrist_images", relative_step_idx))
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # current primary (head)
        current_image_latent_idx = segment_idx
        frames.append(self._get_frame(episode_data, "images", relative_step_idx))
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # action (blank image placeholder)
        action_latent_idx = segment_idx
        frames.append(np.zeros_like(ref_image))
        repeats.append(self.num_duplicates_per_image)
        segment_idx += 1

        # future block (proprio + head/L/R wrist).
        # Skipped entirely when predict_future_states=False — sequence ends
        # after the action slot, indices stay -1 so the trainer skips both
        # future_state and future_image losses. Cuts state_t 10 → 6 and
        # chunk_duration 37 → 21 (pair with _ROBOTWIN_MODEL_CONFIG_NO_FUTURE).
        if self.predict_future_states:
            # future proprio
            if self.use_proprio:
                future_proprio_latent_idx = segment_idx
                frames.append(np.zeros_like(ref_image))
                repeats.append(self.num_duplicates_per_image)
                segment_idx += 1
                future_proprio = episode_data["proprio"][future_frame_idx]
            else:
                future_proprio = np.zeros_like(episode_data["proprio"][future_frame_idx])

            # future left wrist
            future_wrist_image_latent_idx = segment_idx
            frames.append(self._get_frame(episode_data, "left_wrist_images", future_frame_idx))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

            # future right wrist
            future_wrist_image2_latent_idx = segment_idx
            frames.append(self._get_frame(episode_data, "right_wrist_images", future_frame_idx))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

            # future primary
            future_image_latent_idx = segment_idx
            frames.append(self._get_frame(episode_data, "images", future_frame_idx))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1
        else:
            future_proprio = np.zeros_like(episode_data["proprio"][future_frame_idx])

        # value placeholder
        if self.return_value_function_returns:
            value_latent_idx = segment_idx
            frames.append(np.zeros_like(ref_image))
            repeats.append(self.num_duplicates_per_image)
            segment_idx += 1

        # -------- preprocess + temporal expand --------
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

        # -------- action chunks + returns --------
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
            "rollout_data_mask": 0,  # demo only
            "rollout_data_success_mask": 0,
            # Demo samples predict actions, not future state. Setting this to 1
            # makes policy_video2world_model treat the action slot as a clean
            # conditioning frame (see line ~152), so model_pred equals GT at the
            # action slot and demo_sample_action_l1_loss collapses to 0.
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": value_latent_idx,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_wrist_image2_latent_idx": current_wrist_image2_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_wrist_image2_latent_idx": future_wrist_image2_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=None,
                        help="Multi-task root, e.g. /mnt/ddn/dataset/RoboTwin2.0/dataset")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                        help="One or more task folders, or a single path to a task-list .txt")
    parser.add_argument("--embodiment", type=str, default=DEFAULT_EMBODIMENT)
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Legacy single-task dir (e.g. .../pick_diverse_bottles/aloha-agilex_clean_50)",
    )
    parser.add_argument("--chunk_size", type=int, default=16)
    args = parser.parse_args()

    if args.dataset_root is None and args.data_dir is None:
        args.dataset_root = "/mnt/ddn/dataset/RoboTwin2.0/dataset"
        args.tasks = args.tasks or ["pick_diverse_bottles"]

    tasks_arg: Optional[Sequence[str] | str] = args.tasks
    if tasks_arg and len(tasks_arg) == 1 and os.path.isfile(tasks_arg[0]):
        tasks_arg = tasks_arg[0]

    ds = RoboTwinDataset(
        data_dir=args.data_dir,
        dataset_root=args.dataset_root,
        tasks=tasks_arg,
        embodiment=args.embodiment,
        chunk_size=args.chunk_size,
        t5_text_embeddings_path="",  # no embedding needed for shape check
        normalize_actions=False,
        normalize_proprio=False,
    )
    print(f"episodes={ds.num_episodes}  total_steps={ds.num_steps}")
    print(f"unique_commands={ds.unique_commands}")
    ep0 = ds.data[0]
    print(
        f"ep0: images={ep0['images'].shape} actions={ep0['actions'].shape} "
        f"proprio={ep0['proprio'].shape} command={ep0['command']!r}"
    )
    print("actions[0]:", ep0["actions"][0])
    print("proprio[0]:", ep0["proprio"][0])
    # Sanity: under the one-step left-shift convention, action[t] should
    # equal proprio[t+1] for all t < T-1 (and action[T-1] == proprio[T-1]).
    import numpy as _np
    assert _np.allclose(ep0["actions"][:-1], ep0["proprio"][1:]), (
        "actions should be proprio shifted by +1 step"
    )
    assert _np.allclose(ep0["actions"][-1], ep0["proprio"][-1]), (
        "terminal action should equal terminal proprio (hold-last padding)"
    )
    print("shift invariant: actions[:-1] == proprio[1:]  ✓")


if __name__ == "__main__":
    _main()
