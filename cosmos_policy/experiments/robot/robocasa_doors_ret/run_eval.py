# -----------------------------------------------------------------------------
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# -----------------------------------------------------------------------------

"""RoboCasa door retrieval-augmented policy eval.

Structure mirrors ``experiments/robot/robocasa/run_robocasa_eval.py`` for
env + rollout machinery, and ``experiments/robot/pusht_ret/run_eval.py``
for retrieval flow.

Usage (illustrative)::

    uv run --extra cu128 --group robocasa --python 3.10 python \\
      -m cosmos_policy.experiments.robot.robocasa_doors_ret.run_eval \\
      --config cosmos_predict2p5_2b_480p_robocasa_open_door_ret__inference \\
      --ckpt_path  /path/to/checkpoint \\
      --config_file cosmos_policy/config/config.py \\
      --task_names OpenSingleDoor OpenDoubleDoor \\
      --pool_features_root /mnt/ddn/dataset/robocasa-retrieval/kitchen_doors \\
      --pool_source_root   /mnt/ddn/dataset/robocasa-MG_3000_hdf5/kitchen_doors \\
      --dataset_stats_path /path/to/dataset_statistics.json \\
      --t5_text_embeddings_path /path/to/reason1_embeddings.pkl \\
      --chunk_size 32 --num_open_loop_steps 16 \\
      --num_trials_per_task 10 \\
      --local_log_dir cosmos_policy/experiments/robot/robocasa_doors_ret/logs/

.. warning::
    The final model-inference call (building a ``data_batch`` from the
    assembled 15-latent layout and running the RF denoise loop) is
    delegated to the stub ``get_action_robocasa_ret`` below. The stub
    documents the required steps; fill it in by adapting the relevant
    sections of ``cosmos_policy/experiments/robot/cosmos_utils.get_action``
    (retrieval-layout branch) to this project's 15-latent layout.
"""

from __future__ import annotations

import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import draccus
import imageio
import numpy as np
import torch

from cosmos_policy.constants import ACTION_DIM
from cosmos_policy.experiments.robot.cosmos_utils import (
    COSMOS_IMAGE_SIZE,
    COSMOS_TEMPORAL_COMPRESSION_FACTOR,
    extract_action_chunk_from_latent_sequence,
    get_future_images_from_generated_samples,
    get_model,
    get_t5_embedding_from_cache,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
    prepare_images_for_model,
    rescale_proprio,
    unnormalize_actions,
)
from PIL import Image, ImageDraw, ImageFont
from cosmos_policy.experiments.robot.robot_utils import (
    log_message,
    setup_logging,
)
from cosmos_policy.utils.utils import set_seed_everywhere

from .latent_layout import (
    LATENT_BLANK,
    LATENT_CUR_LEFT,
    LATENT_CUR_PROPRIO,
    LATENT_FUTURE_PROPRIO,
    LATENT_PRED_ACTION,
    LATENT_RET_ACTION,
    LATENT_RET_PROPRIO,
    NUM_CONDITIONAL_LATENTS,
    STATE_T,
    build_image_sequence,
)

# Non-image (inject-only) latent slots. These positions are reset to the
# pre-injection clean latent before VAE-decoding the generated sample,
# otherwise the decoded image at those slots would contain garbage from
# the proprio/action injections. See cosmos_utils.undo_latent_injection.
INDICES_TO_REPLACE = [
    LATENT_BLANK,
    LATENT_RET_PROPRIO,
    LATENT_RET_ACTION,
    LATENT_CUR_PROPRIO,
    LATENT_PRED_ACTION,
    LATENT_FUTURE_PROPRIO,
]
from .retrieval import RoboCasaDoorsRetrieval

# RoboCasa sim imports are heavy — import lazily inside run().
_ROBOCASA_ENV_BUILDER = None


# ---------------------------------------------------------------------------
# Eval config
# ---------------------------------------------------------------------------

@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "robocasa_doors_ret"

    # ── Cosmos Policy model ──
    model_family: str = "cosmos"
    config: str = ""
    ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"

    use_third_person_image: bool = True
    num_third_person_images: int = 2          # left + right agent-view
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = False
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    shift: float = 5.0
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 32
    num_open_loop_steps: int = 16

    # ── Seeding / parallel ──
    deterministic: bool = True
    randomize_seed: bool = False
    seed: int = 0

    # ── RoboCasa sim ──
    task_names: list = field(default_factory=lambda: ["OpenSingleDoor", "OpenDoubleDoor"])
    num_trials_per_task: int = 10
    env_img_res: int = 128

    # ── Retrieval ──
    pool_features_root: str = "/mnt/ddn/dataset/robocasa-retrieval/kitchen_doors"
    pool_source_root:   str = "/mnt/ddn/dataset/robocasa-MG_3000_hdf5/kitchen_doors"
    pool_stride: int = 4
    pre_top_m: int = 200
    retrieval_top_k_choice: int = 1
    use_residual_actions: bool = False

    # ── Logging ──
    local_log_dir: str = "./experiments/logs"
    run_id_note: Optional[str] = None
    use_wandb: bool = False
    wandb_entity: str = "YOUR_ENTITY"
    wandb_project: str = "YOUR_PROJECT"
    # fmt: on


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _lazy_env_builder():
    """Import the baseline robocasa env utilities lazily (heavy deps)."""
    global _ROBOCASA_ENV_BUILDER
    if _ROBOCASA_ENV_BUILDER is not None:
        return _ROBOCASA_ENV_BUILDER
    sys.path.insert(
        0, os.path.join(os.path.dirname(__file__), "..", "robocasa")
    )
    from robocasa_utils import make_robocasa_env  # type: ignore
    _ROBOCASA_ENV_BUILDER = make_robocasa_env
    return _ROBOCASA_ENV_BUILDER


def prepare_observation(raw_obs: dict, flip_images: bool) -> dict:
    """Extract {wrist, left, right, proprio} from the env observation."""
    wrist = raw_obs["robot0_eye_in_hand_image"]
    left  = raw_obs["robot0_agentview_left_image"]
    right = raw_obs["robot0_agentview_right_image"]
    if flip_images:
        wrist = np.flipud(wrist).copy()
        left  = np.flipud(left).copy()
        right = np.flipud(right).copy()
    # Robosuite proprio: robot0_proprio-state expected; fall back to concat
    # of gripper_qpos + eef_pos + eef_quat if needed.
    if "robot0_proprio-state" in raw_obs:
        proprio = raw_obs["robot0_proprio-state"].astype(np.float32)
    else:
        proprio = np.concatenate([
            raw_obs["robot0_gripper_qpos"],
            raw_obs["robot0_eef_pos"],
            raw_obs["robot0_eef_quat"],
        ]).astype(np.float32)
    return {
        "primary_image": left,
        "secondary_image": right,
        "wrist_image": wrist,
        "proprio": proprio,
    }


# ---------------------------------------------------------------------------
# Model inference (STUB)
# ---------------------------------------------------------------------------

def _normalize_action_chunk(raw_actions: np.ndarray, dataset_stats: dict) -> np.ndarray:
    """Min/max normalize a (H, A) action chunk into [-1, 1] with the same stats
    used during training."""
    a_min = dataset_stats["actions_min"]
    a_max = dataset_stats["actions_max"]
    return (2.0 * (raw_actions - a_min) / (a_max - a_min + 1e-8) - 1.0).astype(np.float32)


def get_action_robocasa_ret(
    cfg: PolicyEvalConfig,
    model: torch.nn.Module,
    dataset_stats: dict,
    observation: dict,
    ret_data: dict,
    task_label_or_embedding,
    seed: int,
    batch_size: int = 1,
) -> dict:
    """Run the retrieval-augmented RF policy for one chunk on the 15-latent layout.

    Returns
    -------
    dict
        ``actions`` : list of ``chunk_size`` numpy (action_dim,) arrays
                      (raw un-normalized; if residual mode, already recombined
                      with retrieved action).
        ``latent_indices`` : dict of indices used this call (for downstream
                             visualizations).
    """
    import secrets

    if cfg.randomize_seed:
        seed = secrets.randbits(32) % 256

    with torch.inference_mode():
        # ── Text embedding ──────────────────────────────────────────────
        if isinstance(task_label_or_embedding, str):
            text_embedding = get_t5_embedding_from_cache(task_label_or_embedding)
        else:
            text_embedding = torch.tensor(task_label_or_embedding, dtype=torch.bfloat16).cuda()

        # ── Image preprocessing (train-consistent: resize→jpeg→aug) ─────
        # Order matches RoboCasaDoorsRetDataset latent order: wrist, left, right.
        cur_cams = prepare_images_for_model(
            [observation["wrist_image"], observation["primary_image"], observation["secondary_image"]],
            cfg,
        )
        ret_cams = prepare_images_for_model(
            [ret_data["ret_wrist"], ret_data["ret_left"], ret_data["ret_right"]],
            cfg,
        )

        # ── Proprio normalization ───────────────────────────────────────
        raw_proprio = observation["proprio"].astype(np.float32)
        if cfg.normalize_proprio:
            norm_proprio = rescale_proprio(
                raw_proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0
            )
        else:
            norm_proprio = raw_proprio

        raw_ret_proprio = ret_data["ret_proprio"].astype(np.float32)
        if cfg.normalize_proprio:
            norm_ret_proprio = rescale_proprio(
                raw_ret_proprio, dataset_stats, non_negative_only=False, scale_multiplier=1.0
            )
        else:
            norm_ret_proprio = raw_ret_proprio

        # ── Ret action chunk normalization ──────────────────────────────
        raw_ret_actions = ret_data["ret_actions"].astype(np.float32)     # (H, 7)
        norm_ret_actions = _normalize_action_chunk(raw_ret_actions, dataset_stats)

        # ── 15-latent image sequence ────────────────────────────────────
        images, latent_indices = build_image_sequence(
            cur_wrist=cur_cams[0],
            cur_left=cur_cams[1],
            cur_right=cur_cams[2],
            ret_wrist=ret_cams[0],
            ret_left=ret_cams[1],
            ret_right=ret_cams[2],
            num_duplicates=COSMOS_TEMPORAL_COMPRESSION_FACTOR,
        )
        video = np.transpose(np.expand_dims(images, 0), (0, 4, 1, 2, 3))   # (1, C, T, H, W)
        if batch_size > 1:
            video = np.tile(video, (batch_size, 1, 1, 1, 1))
        video = torch.from_numpy(video).to(dtype=torch.uint8).cuda()

        # ── Tensors for injection ───────────────────────────────────────
        proprio_t = (
            torch.from_numpy(norm_proprio).reshape(1, -1)
            .repeat(batch_size, 1).to(dtype=torch.bfloat16).cuda()
        )
        ret_proprio_t = (
            torch.from_numpy(norm_ret_proprio).reshape(1, -1)
            .repeat(batch_size, 1).to(dtype=torch.bfloat16).cuda()
        )
        ret_actions_t = (
            torch.from_numpy(norm_ret_actions).unsqueeze(0)
            .repeat(batch_size, 1, 1).to(dtype=torch.bfloat16).cuda()
        )
        actions_placeholder = torch.zeros(
            (batch_size, cfg.chunk_size, ACTION_DIM), dtype=torch.bfloat16, device="cuda"
        )
        future_proprio_placeholder = torch.zeros(
            (batch_size, norm_proprio.shape[0]), dtype=torch.bfloat16, device="cuda"
        )

        # ── Assemble data_batch (mirrors RoboCasaDoorsRetDataset output) ─
        def _idx_t(k: str) -> torch.Tensor:
            return torch.tensor([latent_indices[k]] * batch_size, dtype=torch.int64, device="cuda")

        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "t5_text_embeddings": text_embedding.repeat(batch_size, 1, 1).to(dtype=torch.bfloat16).cuda(),
            "fps": torch.tensor([16] * batch_size, dtype=torch.bfloat16).cuda(),
            "padding_mask": torch.zeros(
                (batch_size, 1, COSMOS_IMAGE_SIZE, COSMOS_IMAGE_SIZE), dtype=torch.bfloat16
            ).cuda(),
            "num_conditional_frames": model.config.min_num_conditional_frames,
            # Injection values.
            "proprio": proprio_t,
            "future_proprio": future_proprio_placeholder,
            "actions": actions_placeholder,
            "ret_proprio": ret_proprio_t,
            "ret_actions": ret_actions_t,
            # Latent index fields.
            "current_proprio_latent_idx":     _idx_t("current_proprio_latent_idx"),
            "current_wrist_image_latent_idx": _idx_t("current_wrist_image_latent_idx"),
            "current_image_latent_idx":       _idx_t("current_image_latent_idx"),
            "current_image2_latent_idx":      _idx_t("current_image2_latent_idx"),
            "action_latent_idx":              _idx_t("action_latent_idx"),
            "future_proprio_latent_idx":      _idx_t("future_proprio_latent_idx"),
            "future_wrist_image_latent_idx":  _idx_t("future_wrist_image_latent_idx"),
            "future_image_latent_idx":        _idx_t("future_image_latent_idx"),
            "future_image2_latent_idx":       _idx_t("future_image2_latent_idx"),
            "value_latent_idx":               _idx_t("value_latent_idx"),
            "ret_wrist_image_latent_idx":     _idx_t("ret_wrist_image_latent_idx"),
            "ret_left_image_latent_idx":      _idx_t("ret_left_image_latent_idx"),
            "ret_right_image_latent_idx":     _idx_t("ret_right_image_latent_idx"),
            "ret_proprio_latent_idx":         _idx_t("ret_proprio_latent_idx"),
            "ret_action_latent_idx":          _idx_t("ret_action_latent_idx"),
            # Loss/mask fields (kept for train-path compatibility; inference-time no-ops).
            "rollout_data_mask":          torch.zeros(batch_size, dtype=torch.float32, device="cuda"),
            "rollout_data_success_mask":  torch.zeros(batch_size, dtype=torch.float32, device="cuda"),
            "world_model_sample_mask":    torch.ones(batch_size, dtype=torch.float32, device="cuda"),
            "value_function_sample_mask": torch.zeros(batch_size, dtype=torch.float32, device="cuda"),
            "value_function_return":      torch.full((batch_size,), -100.0, dtype=torch.float32, device="cuda"),
        }

        # ── RF denoise → generated latents ──────────────────────────────
        generated_latent, _orig_clean = model.generate_samples_from_batch(
            data_batch,
            n_sample=batch_size,
            num_steps=cfg.num_denoising_steps_action,
            seed=seed,
            is_negative_prompt=False,
            use_variance_scale=cfg.use_variance_scale,
            return_orig_clean_latent_frames=True,
            shift=cfg.shift,
        )

        # ── Extract action chunk from the generated latent ──────────────
        action_latent_idx = latent_indices["action_latent_idx"]
        _raw = getattr(model, "_generated_action", None)
        if _raw is not None:
            actions_norm = (
                _raw.reshape(batch_size, cfg.chunk_size, ACTION_DIM)
                .to(torch.float32).cpu().numpy()
            )
            model._generated_action = None
        else:
            action_indices = torch.full(
                (batch_size,), action_latent_idx, dtype=torch.int64,
                device=generated_latent.device,
            )
            actions_norm = (
                extract_action_chunk_from_latent_sequence(
                    generated_latent,
                    action_shape=(cfg.chunk_size, ACTION_DIM),
                    action_indices=action_indices,
                    decoder=getattr(model, "action_decoder", None),
                )
                .to(torch.float32).cpu().numpy()
            )

        # Pick sample 0 for single-env rollout.
        actions_norm = actions_norm[0]                         # (H, A) normalized

        # ── Future image predictions (for visualization) ────────────────
        future_image_predictions = get_future_images_from_generated_samples(
            model,
            generated_latent.clone(),
            cfg,
            _orig_clean,
            INDICES_TO_REPLACE,
            latent_indices["future_wrist_image_latent_idx"],
            -1,                                                # no 2nd wrist
            latent_indices["future_image_latent_idx"],         # left
            latent_indices["future_image2_latent_idx"],        # right
            temporal_compression_factor=COSMOS_TEMPORAL_COMPRESSION_FACTOR,
        )
        # Single-env: pull out the batch-0 slice.
        future_image_predictions = {
            k: (v[0] if v is not None else None)
            for k, v in future_image_predictions.items()
        }

        # ── Un-normalize & (if residual) recombine with retrieved action ─
        if cfg.use_residual_actions:
            # Training target was delta_norm = gt_norm − ret_norm. Un-normalize the
            # delta with the chunked formula:  raw_delta = delta_norm * (max-min)/2
            a_min = dataset_stats["actions_min"]
            a_max = dataset_stats["actions_max"]
            raw_delta = actions_norm * (a_max - a_min) / 2.0
            actions_raw = raw_ret_actions + raw_delta          # (H, A)
        elif cfg.unnormalize_actions:
            actions_raw = unnormalize_actions(actions_norm, dataset_stats)
        else:
            actions_raw = actions_norm

        return {
            "actions": [actions_raw[i] for i in range(actions_raw.shape[0])],
            "latent_indices": latent_indices,
            "future_image_predictions": future_image_predictions,
        }


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_episode(
    cfg: PolicyEvalConfig,
    env,
    model,
    dataset_stats: dict,
    retrieval: RoboCasaDoorsRetrieval,
    task_name: str,
    task_description: str,
    episode_idx: int,
    max_steps: int,
    log_file,
) -> tuple[bool, dict]:
    """Run one rollout and collect enough data to produce both a plain
    replay video and a 3-panel (real | future-predicted | retrieved) video.

    Returns (success, payload_dict) where payload_dict has:
        replay        : list[{wrist, left, right}] for every env step
        chunk_future  : list[future_image_predictions dict] per chunk
        chunk_ret     : list[{wrist, left, right, meta}] per chunk
    """
    obs, _ = env.reset()
    retrieval.reset()

    action_queue: deque = deque()
    replay: list = []                  # one entry per env step
    chunk_future: list = []             # one entry per chunk
    chunk_ret: list = []                # one entry per chunk
    success = False

    for t in range(max_steps):
        observation = prepare_observation(obs, cfg.flip_images)
        replay.append({
            "wrist": observation["wrist_image"].copy(),
            "left":  observation["primary_image"].copy(),
            "right": observation["secondary_image"].copy(),
        })

        retrieval.push_observation(
            left=observation["primary_image"],
            right=observation["secondary_image"],
            proprio=observation["proprio"],
        )

        if t == 0:
            retrieval.initialize_episode(
                left_t0=observation["primary_image"],
                right_t0=observation["secondary_image"],
            )

        if len(action_queue) == 0:
            ret_data = retrieval.retrieve()

            try:
                action_return_dict = get_action_robocasa_ret(
                    cfg,
                    model,
                    dataset_stats,
                    observation,
                    ret_data,
                    task_label_or_embedding=task_description,
                    seed=cfg.seed + episode_idx,
                )
            except Exception:
                log_message(f"Inference error:\n{traceback.format_exc()}", log_file)
                break

            chunk_future.append(action_return_dict.get("future_image_predictions"))
            chunk_ret.append({
                "wrist": ret_data["ret_wrist"],
                "left":  ret_data["ret_left"],
                "right": ret_data["ret_right"],
                "meta":  (ret_data["match_folder"], ret_data["match_demo"], ret_data["match_t"]),
            })

            # Residual recombination is handled inside get_action_robocasa_ret,
            # so action_return_dict["actions"] is already the raw final action.
            for a in action_return_dict["actions"][: cfg.num_open_loop_steps]:
                action_queue.append(np.asarray(a, dtype=np.float32))

        action = action_queue.popleft()
        obs, _reward, terminated, _truncated, info = env.step(action)
        log_message(
            f"  t={t:3d}  action={np.round(action, 3)}", log_file,
        )
        if terminated or info.get("success", False):
            success = True
            break

    return success, dict(replay=replay, chunk_future=chunk_future, chunk_ret=chunk_ret)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_video(frames: list, filepath: str, fps: int = 10) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not frames:
        return
    with imageio.get_writer(filepath, fps=fps) as writer:
        for f in frames:
            writer.append_data(f)


def _label_bar(width: int, height: int, labels: list[str], colors=None) -> np.ndarray:
    """Single RGB (H, W*N, 3) label strip across N panels."""
    if colors is None:
        colors = [(255, 255, 255)] * len(labels)
    bar = Image.new("RGB", (width * len(labels), height), (40, 40, 40))
    draw = ImageDraw.Draw(bar)
    try:
        font = ImageFont.truetype("DejaVuSans", 13)
    except IOError:
        font = ImageFont.load_default()
    for i, (lb, c) in enumerate(zip(labels, colors)):
        tw = draw.textlength(lb, font=font)
        draw.text((i * width + width // 2 - tw // 2, max(height // 4, 4)), lb, font=font, fill=c)
    return np.asarray(bar)


def save_combined_video(
    replay: list,                    # [{wrist, left, right}, …]  length = T_env
    chunk_future: list,              # [future_image_predictions | None, …]  length = T_env/num_open_loop
    chunk_ret: list,                 # [{wrist, left, right, meta} | None, …]
    num_open_loop_steps: int,
    filepath: str,
    fps: int = 10,
) -> None:
    """3 rows (wrist / left / right) × 3 cols (real | generated | retrieved)."""
    if not replay:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    ref = replay[0]["left"]
    H, W = ref.shape[:2]
    text_h = 22

    def _resize(img):
        if img.shape[:2] == (H, W):
            return img
        return np.asarray(Image.fromarray(img).resize((W, H), Image.BILINEAR))

    row_labels = ["wrist", "left", "right"]
    col_labels = ["real", "generated", "retrieved"]
    col_bar = _label_bar(W, text_h, col_labels)                     # (text_h, W*3, 3)
    row_bar = _label_bar(H, text_h, row_labels)                     # (text_h, H*3, 3)
    # row label bar needs to be rotated to run vertically
    row_bar = np.rot90(row_bar, k=1)                                # (H*3, text_h, 3)
    blank = np.zeros((H, W, 3), dtype=np.uint8)

    with imageio.get_writer(filepath, fps=fps) as writer:
        for t, frame in enumerate(replay):
            chunk_idx = t // num_open_loop_steps
            fut = chunk_future[chunk_idx] if chunk_idx < len(chunk_future) else None
            ret = chunk_ret[chunk_idx] if chunk_idx < len(chunk_ret) else None

            rows = []
            for cam in ("wrist", "left", "right"):
                real_im = _resize(frame[cam])

                # generated:
                gen_im = blank
                if fut is not None:
                    key = {
                        "wrist": "future_wrist_image",
                        "left":  "future_image",
                        "right": "future_image2",
                    }[cam]
                    if fut.get(key) is not None:
                        gen_im = _resize(fut[key])

                # retrieved:
                ret_im = blank if ret is None else _resize(ret[cam])

                rows.append(np.concatenate([real_im, gen_im, ret_im], axis=1))   # (H, W*3, 3)

            grid = np.concatenate(rows, axis=0)                                  # (H*3, W*3, 3)
            # Attach label bars.
            top = np.concatenate([np.zeros((text_h, text_h, 3), dtype=np.uint8), col_bar], axis=1)
            body = np.concatenate([row_bar, grid], axis=1)
            composite = np.concatenate([top, body], axis=0)
            writer.append_data(composite)


@draccus.wrap()
def eval_robocasa_doors_ret(cfg: PolicyEvalConfig) -> None:
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)

    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    model, cosmos_config = get_model(cfg)

    log_file, _, run_id = setup_logging(
        cfg=cfg,
        task_identifier="robocasa_doors_ret",
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Eval config: {cfg}", log_file)

    retrieval = RoboCasaDoorsRetrieval(
        pool_features_root=cfg.pool_features_root,
        pool_source_root=cfg.pool_source_root,
        task_split=list(cfg.task_names),
        pool_stride=cfg.pool_stride,
        pre_top_m=cfg.pre_top_m,
        retrieval_top_k_choice=cfg.retrieval_top_k_choice,
    )

    make_env = _lazy_env_builder()
    video_dir = os.path.join(cfg.local_log_dir, run_id, "videos")
    totals = {name: {"success": 0, "n": 0} for name in cfg.task_names}

    for task_name in cfg.task_names:
        for trial in range(cfg.num_trials_per_task):
            env, task_description, max_steps = make_env(
                task_name=task_name, seed=cfg.seed + trial
            )
            t0 = time.time()
            try:
                success, payload = run_episode(
                    cfg=cfg,
                    env=env,
                    model=model,
                    dataset_stats=dataset_stats,
                    retrieval=retrieval,
                    task_name=task_name,
                    task_description=task_description,
                    episode_idx=trial,
                    max_steps=max_steps,
                    log_file=log_file,
                )
            finally:
                env.close()
            elapsed = time.time() - t0
            totals[task_name]["n"] += 1
            totals[task_name]["success"] += int(success)
            rate = totals[task_name]["success"] / totals[task_name]["n"]
            log_message(
                f"[{task_name}] trial {trial + 1}/{cfg.num_trials_per_task}: "
                f"{'SUCCESS' if success else 'FAIL'} ({elapsed:.1f}s)  "
                f"running rate: {rate * 100:.1f}%",
                log_file,
            )
            stem = f"{task_name}__ep{trial + 1:03d}--{'success' if success else 'fail'}"
            # Plain replay (primary = left) — legacy single-panel mp4.
            save_video(
                [step["left"] for step in payload["replay"]],
                os.path.join(video_dir, f"{stem}.mp4"),
            )
            # 3×3 grid: rows (wrist/left/right) × cols (real/generated/retrieved).
            save_combined_video(
                replay=payload["replay"],
                chunk_future=payload["chunk_future"],
                chunk_ret=payload["chunk_ret"],
                num_open_loop_steps=cfg.num_open_loop_steps,
                filepath=os.path.join(video_dir, f"{stem}--combined.mp4"),
                fps=10,
            )

    retrieval.close()
    log_message("\n=== FINAL RESULTS ===", log_file)
    for name, d in totals.items():
        rate = (d["success"] / d["n"]) if d["n"] else 0.0
        log_message(f"  {name}: {d['success']}/{d['n']}  ({rate * 100:.1f}%)", log_file)


if __name__ == "__main__":
    eval_robocasa_doors_ret()
