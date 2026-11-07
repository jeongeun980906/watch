"""Cosmos policy inference server for the FFW BG2 bimanual real-robot setup.

This script is the partner of
``real_robot_code/safe_constraint_patching/77_fk_ffw_bg2_cosmos_control.ipynb``.
It runs as a separate process, owns the policy model + GPU, and exposes a
ZMQ pull/push interface:
uv run --extra cu128 --group robotwin -m cosmos_policy.experiments.robot.real_ffw.deploy_realrobot --task "place bottle in plastic box"
  *  Local pyrealsense2 capture (color stream, 640×480 @ 30 fps by default).
       We apply the canonical training crop (``CROP_BOX = (0, 115, 640, 480)``
       → 640×365, see ``dataset/real_robot/CROP_RULE.md``) before handing the
       image to ``get_action``. The cosmos preprocessing pipeline then
       resizes to 224×224, matching training. The camera must be plugged
       into THIS machine; nothing else (notebook, worker) can open the
       device at the same time.

  *  SUB ``--request-port``      (77 control notebook → here)
       Inference trigger. Each message is
       ``{"type": "request", "ts": ..., "eef_l": 4x4, "eef_r": 4x4,
         "grip_l": float, "grip_r": float}``.
       The control side has already run FK in MuJoCo and packaged the EEF
       poses in arm-base coordinates, so this process can stay pure — no
       MuJoCo, no ``ri_motion_v5_package``. We assemble the 20-D
       ``[L_xyz(3), L_rot6d(6), L_grip(1), R_xyz(3), R_rot6d(6), R_grip(1)]``
       proprio vector here from those fields; it matches the layout
       ``RealRobotDataset`` produced at training time.

  *  PUB ``--chunk-port``        (here → 77 control notebook)
       New action chunk. Message format:
           ``{"type": "chunk", "actions": np.ndarray(16, 20), "req_ts": ...,
              "pub_ts": ...}``
       ``actions`` is already un-normalized (back to the original dataset
       scale), so the control side just consumes it row by row.

The design is **single-threaded with non-blocking polling**: between each
iteration of the main loop we drain image messages (conflate=1 keeps only
the latest) and check for an inference request. When a request arrives we
build the proprio from the request payload, then run ``get_action``
synchronously; downstream sockets buffer messages while the GPU is busy and
we pick up the newest values on the next poll. This avoids GIL/thread-sync
complexity at the cost of inference holding up image ingestion for ~200-500
ms — acceptable because the chunk is conditioned on the *moment we received
the request*, not on whatever the robot is doing during the denoising
steps.

Usage example::

    python -m cosmos_policy.experiments.robot.real_ffw.deploy_realrobot \
        --config cosmos_predict2p5_2b_480p_realrobot_bg2_open_cabinet_r10__inference_only \
        --ckpt-path /mnt/ddn/tmp/cosmos_policy/cosmos_v2_finetune/cosmos_predict2_2b_480p_realrobot_bg2/checkpoints/iter_000007000 \
      --dataset-stats-path /mnt/ddn/dataset/real_robot/h5/_cosmos_cache/bg2_bimanual__1tasks_c389b1fc_r10/dataset_statistics.json \
      --task "open cabinet"
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (locked to the training config + CROP_RULE.md). Changing any of
# these silently breaks inference vs. training pixel/proprio alignment, so
# they live as module-level constants rather than CLI flags.
# ---------------------------------------------------------------------------

# CROP_RULE.md canonical crop.
CROP_LEFT, CROP_TOP, CROP_RIGHT, CROP_BOTTOM = 0, 115, 640, 480
CROP_HEIGHT_W = (CROP_RIGHT - CROP_LEFT, CROP_BOTTOM - CROP_TOP)  # (640, 365) (W, H)
EXPECTED_RAW_SIZE_WH = (640, 480)  # Native RealSense color stream we crop from.

# 20-D state/action layout (per timestep) — must match
# convert_real_robot_npz_to_h5.py STATE_ACTION_LAYOUT.
ACTION_DIM = 20
PROPRIO_DIM = 20

# Model + training config defaults — paths into the on-disk training output.
DEFAULT_CONFIG_NAME = "cosmos_predict2_2b_480p_realrobot_bg2__inference_only"
DEFAULT_CKPT_PATH = (
    "/home/rilab/cosmos_policy/"
    "ckpt/baseline/model_000007000.pt"
)
DEFAULT_DATASET_STATS_PATH = (
    "/home/rilab/cosmos_policy/ckpt/baseline/"
    "dataset_statistics.json"
)
DEFAULT_T5_EMBEDDINGS_PATH = "/home/rilab/cosmos_policy/ckpt/rag/t5_embeddings.pkl"
DEFAULT_CONFIG_FILE = "cosmos_policy/config/config.py"

# Default network endpoints — chosen to leave 5556 / 5560-5561 alone since
# those are used by the BG2 bridge.
DEFAULT_REQUEST_IP = "192.168.0.175"
DEFAULT_REQUEST_PORT = 5563
DEFAULT_CHUNK_PORT = 5562

# RealSense color-stream defaults — must match the 76 recording notebook so
# inference observations are pixel-aligned with training data.
DEFAULT_REALSENSE_W = 640
DEFAULT_REALSENSE_H = 480
DEFAULT_REALSENSE_FPS = 30

# Match the training config of cosmos_predict2_2b_480p_realrobot_bg2.
DEFAULT_CHUNK_SIZE = 16


# ---------------------------------------------------------------------------
# DeployConfig — the bag of flags ``get_action`` / ``get_model`` expect.
# Mirrors the relevant subset of the aloha ``DeployConfig`` so we can call
# the existing inference utilities unchanged.
# ---------------------------------------------------------------------------


@dataclass
class DeployConfig:
    """Subset of fields ``cosmos_utils.get_action`` and ``get_model`` read.

    Mirrors the names used in the aloha/pusht inference scripts so we can
    pass this dataclass straight into the existing helpers.
    """

    suite: str = "real_robot"
    model_family: str = "cosmos"
    config: str = DEFAULT_CONFIG_NAME
    ckpt_path: str = DEFAULT_CKPT_PATH
    config_file: str = DEFAULT_CONFIG_FILE
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""

    # Camera setup — single third-eye, no wrist. Pairs with state_t=6 model.
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = False
    num_wrist_images: int = 0
    use_proprio: bool = True
    flip_images: bool = False
    use_variance_scale: bool = False
    use_jpeg_compression: bool = False

    # Autoregressive / value heads are OFF — the realrobot model has no value
    # slot and we only consume actions.
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    predict_future_states: bool = True
    predict_values: bool = False

    # Denoising / chunking.
    # 5 steps roughly halves inference latency vs the 10-step default the
    # aloha deploy uses. PushT also runs at 5 steps with minimal quality loss.
    # Important for our chunk_size=16 @ 10 Hz pipeline because the LOOKAHEAD=5
    # action buffer is only ~0.5 s — at 10 steps the inference latency would
    # exhaust the buffer between chunks.
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    chunk_size: int = DEFAULT_CHUNK_SIZE
    num_open_loop_steps: int = DEFAULT_CHUNK_SIZE  # full chunk; control side
    # rerequests via its own deque/lookahead logic, not via this constant.

    # Normalization toggles — matches realrobot_bg2 dataset settings.
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    trained_with_image_aug: bool = True

    # Planning / best-of-N off.
    use_ensemble_future_state_predictions: bool = False
    num_future_state_predictions_in_ensemble: int = 1
    future_state_ensemble_aggregation_scheme: str = "average"
    use_ensemble_value_predictions: bool = False
    num_value_predictions_in_ensemble: int = 1
    value_ensemble_aggregation_scheme: str = "average"
    search_depth: int = 1
    search_depth_value_aggregation_scheme: str = "use_last_value"
    mask_current_state_action_for_value_prediction: bool = False
    mask_future_state_for_qvalue_prediction: bool = False
    num_queries_best_of_n: int = 1
    use_parallel_inference: bool = False
    available_gpus: str = "0"
    parallel_timeout: int = 15

    # Rectified-flow scheduler shift (ignored for EDM models).
    shift: float = 5.0

    # Determinism.
    seed: int = 195
    randomize_seed: bool = False
    deterministic: bool = True

    # Filled in at runtime by get_image_resize_size (for parity with deploy.py).
    resize_size: int = 224

    # Runtime endpoints — populated from CLI args.
    task: str = "place bottle in plastic box"
    # Local RealSense capture parameters (color stream).
    realsense_width: int = DEFAULT_REALSENSE_W
    realsense_height: int = DEFAULT_REALSENSE_H
    realsense_fps: int = DEFAULT_REALSENSE_FPS
    # Control PC IP — where the 77 notebook publishes inference requests
    # (carrying eef_l/eef_r/grip_l/grip_r). The notebook binds its PUB to
    # this port, so deploy must connect via the Control PC's IP.
    cpc_ip: str = DEFAULT_REQUEST_IP
    request_port: int = DEFAULT_REQUEST_PORT
    chunk_port: int = DEFAULT_CHUNK_PORT
    dataset_stats_path: str = DEFAULT_DATASET_STATS_PATH
    t5_text_embeddings_path: str = DEFAULT_T5_EMBEDDINGS_PATH
    # R-arm-only mode for OC single-task R10 (action_dim=10). Slice 20-D
    # proprio to [10:20] before sending to the model, pad model's 10-D
    # output back to 20-D by holding cur_proprio[L] constant.
    r_arm_only: bool = False


# ---------------------------------------------------------------------------
# FK helpers — replicate the 76 recording pipeline's pose extraction so the
# 20-D proprio matches what the model saw at training time.
# ---------------------------------------------------------------------------


def rotmat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Continuous 6D rotation representation (Zhou et al., 2019).

    Identical to ``convert_real_robot_npz_to_h5.py:rotmat_to_rot6d`` — picks
    the first two columns of R and stacks them. Keep these in sync.
    """
    assert R.shape[-2:] == (3, 3), R.shape
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1).astype(np.float32)


def build_20d_proprio(
    T_arm_base_eef_l: np.ndarray,
    T_arm_base_eef_r: np.ndarray,
    grip_l: float,
    grip_r: float,
) -> np.ndarray:
    """Pack one 20-D proprio vector from EEF poses + gripper master joints.

    Identical layout to ``convert_real_robot_npz_to_h5.py:build_20d_state``
    so the inference proprio is bit-compatible with the training format.
    """
    assert T_arm_base_eef_l.shape == (4, 4)
    assert T_arm_base_eef_r.shape == (4, 4)

    l_xyz = T_arm_base_eef_l[:3, 3].astype(np.float32)
    l_rot6d = rotmat_to_rot6d(T_arm_base_eef_l[:3, :3])
    r_xyz = T_arm_base_eef_r[:3, 3].astype(np.float32)
    r_rot6d = rotmat_to_rot6d(T_arm_base_eef_r[:3, :3])
    grip_lr = np.array([grip_l], dtype=np.float32)
    grip_rr = np.array([grip_r], dtype=np.float32)

    out = np.concatenate([l_xyz, l_rot6d, grip_lr, r_xyz, r_rot6d, grip_rr])
    assert out.shape == (PROPRIO_DIM,), out.shape
    return out


# ---------------------------------------------------------------------------
# Image preprocessing — apply CROP_RULE.md crop (no resize; cosmos pipeline
# does the 224 resize internally to match training augmentations).
# ---------------------------------------------------------------------------


def crop_third_eye_bgr(bgr: np.ndarray) -> np.ndarray:
    """Raw BGR 640×480 → 640×365 RGB ``(H=365, W=640, 3) uint8``.

    Matches ``CROP_RULE.md::crop_third_eye_jpeg`` minus the JPEG round-trip:
    pyrealsense2 gives us raw BGR, so we flip channels and crop in place.
    """
    h, w = bgr.shape[:2]
    if (w, h) != EXPECTED_RAW_SIZE_WH:
        raise ValueError(
            f"unexpected third_eye size: {(w, h)} (want {EXPECTED_RAW_SIZE_WH})"
        )
    cropped = bgr[CROP_TOP:CROP_BOTTOM, CROP_LEFT:CROP_RIGHT, ::-1]  # BGR→RGB
    expected_shape = (CROP_BOTTOM - CROP_TOP, CROP_RIGHT - CROP_LEFT, 3)
    assert cropped.shape == expected_shape, (cropped.shape, expected_shape)
    return np.ascontiguousarray(cropped)


# ---------------------------------------------------------------------------
# Observation source — owns the third-eye image stream and assembles the
# 20-D proprio from the inference request payload sent by the 77 notebook.
# All FK / qpos handling lives on the control side (where MuJoCo + the
# ri_motion_v5_package helpers already exist for IK), so this process stays
# pure: image in, action chunk out.
# ---------------------------------------------------------------------------


@dataclass
class _Observation:
    """Snapshot of the policy's input at one tick."""

    primary_image: np.ndarray  # (365, 640, 3) uint8 — post-crop third-eye RGB.
    proprio: np.ndarray  # (20,) float32 — see build_20d_proprio.
    proprio_ts: float
    image_ts: float


class _ObservationSource:
    """Owns the local RealSense color stream and per-request proprio.

    The camera pipeline is polled non-blocking so we always carry the most
    recent frame — no backlog while the GPU is busy. Proprio is supplied
    per-request via ``update_proprio_from_request``; we never buffer it
    independently because each request payload pins the observation moment
    the chunk should be conditioned on.
    """

    def __init__(self, cfg: DeployConfig):
        import pyrealsense2 as rsense

        self._cfg = cfg
        self._rsense = rsense

        self._pipe = rsense.pipeline()
        rs_cfg = rsense.config()
        rs_cfg.enable_stream(
            rsense.stream.color,
            cfg.realsense_width,
            cfg.realsense_height,
            rsense.format.bgr8,
            cfg.realsense_fps,
        )
        profile = self._pipe.start(rs_cfg)
        color_profile = profile.get_stream(rsense.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        LOGGER.info(
            "RealSense color stream open: %dx%d @ %d fps (fx=%.2f fy=%.2f)",
            intr.width, intr.height, cfg.realsense_fps, intr.fx, intr.fy,
        )

        self._latest_proprio: Optional[np.ndarray] = None
        self._latest_image: Optional[np.ndarray] = None
        self._proprio_ts: float = 0.0
        self._image_ts: float = 0.0

    # ─────────────────── polling ───────────────────

    def poll_once(self) -> None:
        """Drain one camera frame (non-blocking)."""
        frames = self._pipe.poll_for_frames()
        if not frames:
            return
        color = frames.get_color_frame()
        if not color:
            return
        try:
            bgr = np.asanyarray(color.get_data())
            self._latest_image = crop_third_eye_bgr(bgr)
            self._image_ts = time.time()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("RealSense frame decode failed: %s", exc)

    def update_proprio_from_request(self, req: dict) -> bool:
        """Build the 20-D proprio from a request payload.

        Request contract (matches the 77 notebook):
            ``eef_l`` / ``eef_r`` : 4x4 nested list (np.float32-convertible),
                EEF pose in arm-base coordinates.
            ``grip_l`` / ``grip_r`` : float — master gripper joint qpos.

        Returns True on success, False if the payload is malformed (we log
        and skip; the main loop then waits for the next request).
        """
        try:
            T_l = np.asarray(req["eef_l"], dtype=np.float32)
            T_r = np.asarray(req["eef_r"], dtype=np.float32)
            grip_l = float(req["grip_l"])
            grip_r = float(req["grip_r"])
        except (KeyError, TypeError, ValueError) as exc:
            LOGGER.warning("Malformed request payload: %s", exc)
            return False

        if T_l.shape != (4, 4) or T_r.shape != (4, 4):
            LOGGER.warning(
                "Bad EEF pose shapes: eef_l=%s eef_r=%s", T_l.shape, T_r.shape
            )
            return False

        self._latest_proprio = build_20d_proprio(T_l, T_r, grip_l, grip_r)
        self._proprio_ts = float(req.get("ts", time.time()))
        return True

    # ─────────────────── snapshot ───────────────────

    def snapshot(self) -> Optional[_Observation]:
        """Return a fresh ``_Observation`` if both modalities are available."""
        if self._latest_proprio is None or self._latest_image is None:
            return None
        return _Observation(
            primary_image=self._latest_image.copy(),
            proprio=self._latest_proprio.copy(),
            proprio_ts=self._proprio_ts,
            image_ts=self._image_ts,
        )

    def close(self) -> None:
        try:
            self._pipe.stop()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("RealSense pipe.stop() raised: %s", exc)


# ---------------------------------------------------------------------------
# Main inference loop.
# ---------------------------------------------------------------------------


def _build_observation_dict(obs: _Observation) -> dict:
    """Convert ``_Observation`` to the dict ``get_action`` expects.

    For the ``real_robot`` suite we only need ``primary_image`` and
    ``proprio`` — see the get_action branch in cosmos_utils.py.
    """
    return {
        "primary_image": obs.primary_image,
        "proprio": obs.proprio,
    }


def run(cfg: DeployConfig) -> None:
    """Main entry point — load model, wire up ZMQ, serve requests."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Late imports: pulling cosmos_policy triggers heavy CUDA / model init.
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_action,
        get_model,
        get_t5_embedding_from_cache,
        init_t5_text_embeddings_cache,
        load_dataset_stats,
    )
    from cosmos_policy.experiments.robot.real_ffw.zmq_pubsub import ZMQPubSub

    if cfg.deterministic:
        os.environ.setdefault("DETERMINISTIC", "True")

    LOGGER.info("Loading T5 text embeddings from %s", cfg.t5_text_embeddings_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    # Warm the cache with the configured task; get_action will look it up
    # again via the string. We pass the string instead of the bfloat16 tensor
    # because get_action's ndarray path does a .cpu().numpy() that bfloat16
    # cannot do, and the str path hits the cache anyway.
    _ = get_t5_embedding_from_cache(cfg.task)
    LOGGER.info("Warmed T5 cache for task=%r", cfg.task)

    LOGGER.info("Loading dataset stats from %s", cfg.dataset_stats_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    LOGGER.info("Loading model checkpoint from %s", cfg.ckpt_path)
    model, model_cfg = get_model(cfg)
    assert cfg.chunk_size == model_cfg.dataloader_train.dataset.chunk_size, (
        f"Train/test chunk_size mismatch: train={model_cfg.dataloader_train.dataset.chunk_size}, "
        f"test={cfg.chunk_size}"
    )
    LOGGER.info("Model ready. action_dim=%d, proprio_dim=%d, chunk_size=%d",
                model.config.action_dim, model.config.proprio_dim, cfg.chunk_size)

    obs_source = _ObservationSource(cfg=cfg)

    sub_request = ZMQPubSub(
        role="sub", ip=cfg.cpc_ip, port=cfg.request_port,
        hwm=1, conflate=True, verbose=True,
    )
    pub_chunk = ZMQPubSub(role="pub", port=cfg.chunk_port, hwm=1, verbose=True)

    LOGGER.info("Deploy server ready — waiting for inference requests.")

    n_chunks = 0
    last_stats_t = time.time()

    try:
        while True:
            obs_source.poll_once()

            req = sub_request.recv(nonblock=True)
            if req is None:
                # No request pending — short sleep to avoid pegging a CPU
                # core. 5 ms is well below typical inference latency.
                time.sleep(0.005)
                continue

            if not isinstance(req, dict) or req.get("type") != "request":
                LOGGER.warning("Ignoring malformed request: %r", req)
                continue

            if not obs_source.update_proprio_from_request(req):
                # Bad payload — already logged. Wait for the next request.
                continue

            snapshot = obs_source.snapshot()
            if snapshot is None:
                LOGGER.warning(
                    "Request received but image not yet available "
                    "(image=%s)",
                    obs_source._latest_image is not None,
                )
                continue

            req_ts = float(req.get("ts", time.time()))
            obs_dict = _build_observation_dict(snapshot)

            # R-arm-only model: slice the 20-D proprio to R[10:20] before
            # handing it to get_action — the model emits 10-D actions and the
            # 10-D stats live at cfg.dataset_stats_path.
            if cfg.r_arm_only:
                obs_dict = dict(obs_dict)
                obs_dict["proprio"] = obs_dict["proprio"][10:20]

            t0 = time.time()
            action_return = get_action(
                cfg,
                model,
                dataset_stats,
                obs_dict,
                cfg.task,
                seed=cfg.seed,
                randomize_seed=cfg.randomize_seed,
                num_denoising_steps_action=cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=False,
            )
            dt = time.time() - t0

            actions = np.asarray(action_return["actions"])  # (chunk_size, action_dim) or (B, ...)
            if actions.ndim == 3:
                # get_action returns (B=1, chunk_size, action_dim); strip batch.
                actions = actions[0]
            expected_dim = 10 if cfg.r_arm_only else ACTION_DIM
            assert actions.shape == (cfg.chunk_size, expected_dim), (
                f"unexpected action chunk shape {actions.shape}, expected "
                f"({cfg.chunk_size}, {expected_dim})"
            )
            # Pad R-arm-only output back to 20-D by holding the chunk-start
            # L-arm proprio constant (idle hand).
            if cfg.r_arm_only:
                eef20 = np.zeros((cfg.chunk_size, ACTION_DIM), dtype=np.float32)
                eef20[:, 10:20] = actions
                eef20[:, 0:10] = snapshot.proprio[0:10][None, :]
                actions = eef20

            pub_ts = time.time()
            pub_chunk.send(
                {
                    "type": "chunk",
                    "actions": actions.astype(np.float32),
                    "req_ts": req_ts,
                    "pub_ts": pub_ts,
                    "obs_proprio_ts": snapshot.proprio_ts,
                    "obs_image_ts": snapshot.image_ts,
                    "inference_sec": dt,
                }
            )
            n_chunks += 1
            LOGGER.info(
                "chunk #%d published — inference=%.3fs, req_age=%.3fs, image_age=%.3fs",
                n_chunks,
                dt,
                pub_ts - req_ts,
                pub_ts - snapshot.image_ts,
            )

            if pub_ts - last_stats_t >= 30.0:
                LOGGER.info("stats: %d chunks published since last report", n_chunks)
                last_stats_t = pub_ts

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        sub_request.close()
        pub_chunk.close()
        obs_source.close()
        LOGGER.info("Deploy server shutdown complete. Total chunks: %d", n_chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list] = None) -> DeployConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG_NAME, help="Cosmos experiment name (must match training)")
    p.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH)
    p.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    p.add_argument("--dataset-stats-path", default=DEFAULT_DATASET_STATS_PATH)
    p.add_argument("--t5-embeddings-path", default=DEFAULT_T5_EMBEDDINGS_PATH)
    p.add_argument("--task", default="place bottle in plastic box",
                   help="T5 text-embedding lookup key (must exist in t5_embeddings.pkl)")
    p.add_argument("--cpc-ip", default=DEFAULT_REQUEST_IP,
                   help="Control PC IP — where the 77 notebook publishes inference "
                        "requests (proprio + trigger). Defaults to 127.0.0.1 (deploy "
                        "co-located with notebook).")
    p.add_argument("--realsense-width", type=int, default=DEFAULT_REALSENSE_W)
    p.add_argument("--realsense-height", type=int, default=DEFAULT_REALSENSE_H)
    p.add_argument("--realsense-fps", type=int, default=DEFAULT_REALSENSE_FPS)
    p.add_argument("--request-port", type=int, default=DEFAULT_REQUEST_PORT)
    p.add_argument("--chunk-port", type=int, default=DEFAULT_CHUNK_PORT)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    p.add_argument("--num-denoising-steps-action", type=int, default=5)
    p.add_argument("--seed", type=int, default=195)
    p.add_argument("--r-arm-only", action="store_true",
                   help="Use a 10-D R-arm-only model. Slices the 20-D proprio "
                        "to [10:20] before inference, pads the 10-D model "
                        "output back to 20-D using the L-arm proprio.")
    args = p.parse_args(argv)

    cfg = DeployConfig(
        config=args.config,
        ckpt_path=args.ckpt_path,
        config_file=args.config_file,
        dataset_stats_path=args.dataset_stats_path,
        t5_text_embeddings_path=args.t5_embeddings_path,
        task=args.task,
        realsense_width=args.realsense_width,
        realsense_height=args.realsense_height,
        realsense_fps=args.realsense_fps,
        cpc_ip=args.cpc_ip,
        request_port=args.request_port,
        chunk_port=args.chunk_port,
        chunk_size=args.chunk_size,
        num_denoising_steps_action=args.num_denoising_steps_action,
        num_open_loop_steps=args.chunk_size,
        seed=args.seed,
        r_arm_only=args.r_arm_only,
    )
    return cfg


if __name__ == "__main__":
    run(_parse_args())
