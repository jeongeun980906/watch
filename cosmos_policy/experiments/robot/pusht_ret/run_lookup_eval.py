"""
cosmos_policy/experiments/robot/pusht_ret/run_lookup_eval.py

Lookup-table baseline for PushT: skips model inference entirely.
At each chunk boundary, retrieves the closest demo subframe by GT pose
(via PushTRetrieval) and applies its raw actions directly for
``num_open_loop_steps``.

Pool resolution and retrieval feature are identical to ``run_eval.py``.

Usage:
    uv run --extra cu128 --group pusht \\
        -m cosmos_policy.experiments.robot.pusht_ret.run_lookup_eval \\
        --visual_config tri_default \\
        --retrieval_pool_split base_0,base_1,goal_flipped_0 \\
        --num_trials 50 \\
        --chunk_size 8 \\
        --num_open_loop_steps 8 \\
        --local_log_dir .../logs/ \\
        --seed 42
"""

import os
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Optional

# Register local gym_pusht
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pusht"))

import draccus
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gymnasium as gym  # noqa: F401

from cosmos_policy.experiments.robot.robot_utils import log_message, setup_logging
from cosmos_policy.utils.utils import set_seed_everywhere

from .retrieval import PushTRetrieval, resolve_retrieval_split

TASK_DESCRIPTION = "push t shaped block to the location"
MAX_STEPS = 300
SUCCESS_THRESHOLD = 0.85

VISUAL_CONFIGS = {
    "base": {},
    "goal_flipped": {"goal_flipped": True},
    "color": {
        "block_color": "salmon",
    },
    "tri_color": {
        "agent_shape": "triangle",
        "block_color": "salmon",
    },
    "tri_default": {
        "agent_shape": "triangle",
    },
    "tri_goal_flipped": {
        "agent_shape": "triangle",
        "goal_flipped": True,
    },
    "tri_rot60": {
        "agent_shape": "triangle",
        "goal_angle": 60,
    },
    "tri_rot0": {
        "agent_shape": "triangle",
        "goal_angle": 0,
    },
    "tri_rot-60": {
        "agent_shape": "triangle",
        "goal_angle": -60,
    },
    "tri_rot30": {
        "agent_shape": "triangle",
        "goal_angle": 30,
    },
    "tri_rot-30": {
        "agent_shape": "triangle",
        "goal_angle": -30,
    },
    "tri_rot15": {
        "agent_shape": "triangle",
        "goal_angle": 15,
    },
    "tri_rot-15": {
        "agent_shape": "triangle",
        "goal_angle": -15,
    },
}

# Same fallback map as run_eval.py
RETRIEVAL_SPLIT_MAP = {
    "color":            [f"color_{i}"        for i in range(5)],
    "tri_color":        [f"color_{i}"        for i in range(5)],
    "tri_rot60":        [f"rot60_{i}"        for i in range(5)],
    "tri_rot0":         [f"rot0_{i}"         for i in range(5)],
    "tri_rot-60":       [f"rot-60_{i}"       for i in range(5)],
    "tri_rot30":        [f"rot30_{i}"        for i in range(5)],
    "tri_rot-30":       [f"rot-30_{i}"       for i in range(5)],
    "tri_rot15":        [f"rot15_{i}"        for i in range(5)],
    "tri_rot-15":       [f"rot-15_{i}"       for i in range(5)],
    "tri_default":      [f"base_{i}"         for i in range(5)],
    "tri_goal_flipped": [f"goal_flipped_{i}" for i in range(5)],
}


@dataclass
class LookupEvalConfig:
    # fmt: off
    suite: str = "pusht_lookup"
    model_family: str = "lookup"  # used only by setup_logging for run_id

    ##########################################################################
    # PushT environment
    ##########################################################################
    visual_config: str = "tri_default"
    goal_angle: Optional[float] = None
    num_trials: int = 50
    env_img_res: int = 128
    chunk_size: int = 8
    num_open_loop_steps: int = 8

    deterministic: bool = True
    randomize_seed: bool = False
    seed: int = 42

    ##########################################################################
    # Retrieval (matches run_eval.py knobs)
    ##########################################################################
    retrieval_data_dir: str = "/mnt/ddn/dataset/PushT-Cosmos-Policy/success_only"
    block_rel: bool = False
    ret_context_multiplier: int = 1
    ret_image_subsample: int = 1
    retrieval_pool_split: Optional[str] = None
    retrieval_pool_k: int = 2
    retrieval_pool_pattern: str = "plain"

    ##########################################################################
    # Logging
    ##########################################################################
    local_log_dir: str = "./experiments/logs"
    run_id_note: Optional[str] = None
    use_wandb: bool = False
    wandb_entity: str = "YOUR_ENTITY"
    wandb_project: str = "YOUR_PROJECT"
    # fmt: on


def validate_config(cfg: LookupEvalConfig) -> None:
    assert cfg.visual_config in VISUAL_CONFIGS, (
        f"Unknown visual_config '{cfg.visual_config}'. "
        f"Choose from: {list(VISUAL_CONFIGS.keys())}"
    )
    if not cfg.retrieval_data_dir:
        raise ValueError("retrieval_data_dir is required.")


def create_pusht_env(cfg: LookupEvalConfig):
    visual_opts = VISUAL_CONFIGS[cfg.visual_config]
    if cfg.goal_angle is not None:
        visual_opts["goal_angle"] = cfg.goal_angle
    import gym_pusht  # noqa: F401
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        observation_width=cfg.env_img_res,
        observation_height=cfg.env_img_res,
    )
    env.success_threshold = SUCCESS_THRESHOLD
    return env, visual_opts


def save_video(frames: list, filepath: str, fps: int = 10) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with imageio.get_writer(filepath, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def save_combined_video(
    replay_frames: list,
    chunk_ret_frames: list,
    num_open_loop_steps: int,
    filepath: str,
    fps: int = 10,
) -> None:
    """Save 2-panel video: real | retrieved (side by side)."""
    if not replay_frames:
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    H, W, C = replay_frames[0].shape
    text_height = 28
    labels = ["real", "retrieved"]

    def resize_to(img):
        if img.shape[:2] == (H, W):
            return img
        return np.array(Image.fromarray(img).resize((W, H), Image.LANCZOS))

    bar_arr = np.ones((text_height, W * 2, C), dtype=np.uint8) * 40
    bar_pil = Image.fromarray(bar_arr)
    draw = ImageDraw.Draw(bar_pil)
    try:
        font = ImageFont.truetype("DejaVuSans", 13)
    except IOError:
        font = ImageFont.load_default()
    for i, label in enumerate(labels):
        x = i * W + W // 2
        tw = draw.textlength(label, font=font)
        draw.text((x - tw // 2, 7), label, font=font, fill=(255, 255, 255))
    label_bar = np.array(bar_pil)

    blank = np.zeros((H, W, C), dtype=np.uint8)

    with imageio.get_writer(filepath, fps=fps) as writer:
        for t, real in enumerate(replay_frames):
            real = resize_to(real)
            chunk_idx = t // num_open_loop_steps
            frame_in_chunk = t % num_open_loop_steps

            ret = blank
            if chunk_idx < len(chunk_ret_frames) and chunk_ret_frames[chunk_idx] is not None:
                rf = chunk_ret_frames[chunk_idx]
                ret = resize_to(rf[min(frame_in_chunk, len(rf) - 1)])

            combined = np.concatenate([real, ret], axis=1)
            writer.append_data(np.vstack([label_bar, combined]))


def run_episode(
    cfg: LookupEvalConfig,
    env,
    visual_opts: dict,
    retrieval: PushTRetrieval,
    episode_idx: int,
    log_file=None,
) -> tuple:
    """Run one lookup-table episode. Returns (success, coverage, replay_frames, chunk_ret_frames)."""
    obs, _ = env.reset(seed=cfg.seed + episode_idx, options=visual_opts)

    action_queue = deque()
    replay_frames: list = []
    chunk_ret_frames: list = []
    success = False
    coverage = 0.0

    block_pos_history = deque(maxlen=retrieval.WINDOW_SIZE)
    agent_pos_history = deque(maxlen=retrieval.WINDOW_SIZE)

    try:
        for t in range(MAX_STEPS):
            frame = obs["pixels"]
            replay_frames.append(frame.copy())

            agent_pos = obs["agent_pos"].astype(np.float32)
            unwrapped = env.unwrapped
            block_pos = np.array(unwrapped.block.position, dtype=np.float32)
            block_pos_history.append(block_pos)
            agent_pos_history.append(agent_pos)

            if len(action_queue) == 0:
                block_angle = float(unwrapped.block.angle)
                bph = np.array(block_pos_history) if len(block_pos_history) >= 2 else None
                aph = np.array(agent_pos_history) if len(agent_pos_history) >= 2 else None

                ret_frames, ret_actions, _ = retrieval.get_retrieved_data(
                    agent_pos=agent_pos,
                    block_pos=block_pos,
                    block_angle=block_angle,
                    block_pos_history=bph,
                    agent_pos_history=aph,
                )
                chunk_ret_frames.append(ret_frames)

                # Apply retrieved actions directly (lookup-table policy)
                for a in ret_actions[: cfg.num_open_loop_steps]:
                    action_queue.append(a)

            action = np.clip(action_queue.popleft(), 0.0, 512.0)
            obs, _reward, terminated, _truncated, info = env.step(action)
            coverage = info.get("coverage", 0.0)
            log_message(
                f"  t={t:3d}  coverage={coverage:.3f}  action={np.round(action, 1)}",
                log_file,
            )
            if terminated or info.get("is_success", False) or coverage >= SUCCESS_THRESHOLD:
                success = True
                break
    except Exception as e:
        log_message(f"Episode error: {e}\n{traceback.format_exc()}", log_file)

    return success, coverage, replay_frames, chunk_ret_frames


@draccus.wrap()
def eval_pusht_lookup(cfg: LookupEvalConfig) -> None:
    validate_config(cfg)

    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)

    log_file, _, run_id = setup_logging(
        cfg=cfg,
        task_identifier=f"pusht_{cfg.visual_config}",
        log_dir=cfg.local_log_dir,
        run_id_note=cfg.run_id_note,
        use_wandb=cfg.use_wandb,
        wandb_entity=cfg.wandb_entity,
        wandb_project=cfg.wandb_project,
    )
    log_message(f"Lookup eval config: {cfg}", log_file)

    env, visual_opts = create_pusht_env(cfg)

    retrieval_split = resolve_retrieval_split(
        data_dir=cfg.retrieval_data_dir,
        visual_config=cfg.visual_config,
        goal_angle=cfg.goal_angle,
        explicit_split=cfg.retrieval_pool_split,
        k=cfg.retrieval_pool_k,
        fallback_map=RETRIEVAL_SPLIT_MAP,
        pool_pattern=cfg.retrieval_pool_pattern,
    )
    log_message(f"Retrieval split: {retrieval_split}", log_file)

    retrieval = PushTRetrieval(
        data_dir=cfg.retrieval_data_dir,
        chunk_size=cfg.chunk_size,
        split=retrieval_split,
        block_rel=cfg.block_rel,
        ret_context_multiplier=cfg.ret_context_multiplier,
        ret_image_subsample=cfg.ret_image_subsample,
    )

    video_dir = os.path.join(cfg.local_log_dir, run_id, "videos")
    total_successes = 0
    total_coverage = 0.0

    for episode_idx in range(cfg.num_trials):
        log_message(f"\n--- Episode {episode_idx + 1}/{cfg.num_trials} ---", log_file)
        t0 = time.time()
        success, coverage, replay_frames, chunk_ret_frames = run_episode(
            cfg, env, visual_opts, retrieval, episode_idx, log_file,
        )
        elapsed = time.time() - t0

        if success:
            total_successes += 1
        total_coverage += coverage
        success_rate = total_successes / (episode_idx + 1)
        avg_coverage = total_coverage / (episode_idx + 1)
        log_message(
            f"Episode {episode_idx + 1}: {'SUCCESS' if success else 'FAIL'} ({elapsed:.1f}s)  "
            f"Success rate: {success_rate * 100:.1f}% ({total_successes}/{episode_idx + 1})  "
            f"Avg coverage: {avg_coverage * 100:.1f}%",
            log_file,
        )

        ep_suffix = f"ep{episode_idx + 1:03d}--{'success' if success else 'fail'}"
        save_video(replay_frames, os.path.join(video_dir, f"{ep_suffix}.mp4"))
        save_combined_video(
            replay_frames=replay_frames,
            chunk_ret_frames=chunk_ret_frames,
            num_open_loop_steps=cfg.num_open_loop_steps,
            filepath=os.path.join(video_dir, f"{ep_suffix}--combined.mp4"),
            fps=10,
        )

    final_sr = total_successes / cfg.num_trials
    final_cov = total_coverage / cfg.num_trials
    log_message(
        f"\n=== FINAL RESULTS ===\n"
        f"Visual config   : {cfg.visual_config}\n"
        f"Trials          : {cfg.num_trials}\n"
        f"Successes       : {total_successes}\n"
        f"Success rate    : {final_sr  * 100:.1f}%\n"
        f"Average coverage: {final_cov * 100:.1f}%",
        log_file,
    )
    env.close()


if __name__ == "__main__":
    eval_pusht_lookup()
