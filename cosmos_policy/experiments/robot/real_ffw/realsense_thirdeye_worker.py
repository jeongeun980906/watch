"""RealSense third-eye-view ZMQ publisher worker.

Standalone process that opens the RealSense color stream (matching the
``project/realsense_test/03`` configuration that the 76 dataset recording
notebook uses) and re-publishes JPEG-encoded frames over a single ZMQ PUB
socket. Both the cosmos policy server (``deploy_realrobot.py``) and the
control notebook (``77_fk_ffw_bg2_cosmos_control.ipynb``) subscribe to this
stream so they see byte-identical frames per tick.

Why a separate worker (vs. polling RealSense directly inside the consumer)?

- ``pyrealsense2.pipeline`` cannot be opened by two processes at once on the
  same device.
- Keeping camera I/O in its own process means the consumers never block on
  USB jitter — they always see the most recent frame (``CONFLATE=1``).
- The 76 recording notebook ran the pipe in-process; for online inference we
  want the same frames available to both producer (deploy) and visualizer
  (control) without round-trips.

Message format (Python pickle via ``ZMQPubSub.send``):

    {
        "msg_type":   "realsense_thirdeye",
        "ts":         float (time.time()),
        "jpeg":       bytes (full 640x480 RGB → JPEG, quality 80),
        "size":       (640, 480),   # (W, H), pre-crop
        "K":          (3, 3) float64 intrinsic from RealSense color profile,
        "fps_ema":    float,
    }

Crop is intentionally NOT applied here — both consumers apply
``CROP_BOX = (0, 115, 640, 480)`` themselves so the saved frames stay
inspectable at native resolution.

Usage:

    python -m cosmos_policy.experiments.robot.real_ffw.realsense_thirdeye_worker \
        --port 5564 --width 640 --height 480 --fps 30
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rsense

from cosmos_policy.experiments.robot.real_ffw.zmq_pubsub import ZMQPubSub

LOGGER = logging.getLogger(__name__)

# Defaults match the 76 recording notebook so inference observations are
# pixel-aligned with training data.
DEFAULT_W = 640
DEFAULT_H = 480
DEFAULT_FPS = 30
DEFAULT_JPEG_QUALITY = 80
DEFAULT_PORT = 5564
MSG_TYPE = "realsense_thirdeye"


@dataclass(frozen=True)
class WorkerConfig:
    port: int = DEFAULT_PORT
    width: int = DEFAULT_W
    height: int = DEFAULT_H
    fps: int = DEFAULT_FPS
    jpeg_quality: int = DEFAULT_JPEG_QUALITY
    stats_every_sec: float = 5.0


def _open_realsense_pipe(width: int, height: int, fps: int):
    """Open a color-only RealSense pipeline and return (pipe, K, (W, H))."""
    pipe = rsense.pipeline()
    cfg = rsense.config()
    cfg.enable_stream(rsense.stream.color, width, height, rsense.format.bgr8, fps)
    profile = pipe.start(cfg)
    color_profile = profile.get_stream(rsense.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    K = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return pipe, K, (intr.width, intr.height)


def run(cfg: WorkerConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pub = ZMQPubSub(role="pub", port=cfg.port, hwm=1, verbose=True)
    pipe, K, (w, h) = _open_realsense_pipe(cfg.width, cfg.height, cfg.fps)
    LOGGER.info("RealSense color stream open: %dx%d @ %d fps", w, h, cfg.fps)
    LOGGER.info("Publishing on port %d (msg_type=%s)", cfg.port, MSG_TYPE)

    n_published = 0
    n_no_frame = 0
    last_stats_t = time.time()
    fps_ema = 0.0
    t_prev: float | None = None

    try:
        while True:
            frames = pipe.poll_for_frames()
            if not frames:
                n_no_frame += 1
                # Tiny sleep to avoid a tight spin on USB stalls. 1 ms is much
                # smaller than the 33 ms frame interval at 30 fps so latency is
                # not affected.
                time.sleep(0.001)
                continue

            color = frames.get_color_frame()
            if not color:
                n_no_frame += 1
                continue

            bgr = np.asanyarray(color.get_data())
            ok, jbuf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), cfg.jpeg_quality])
            if not ok:
                LOGGER.warning("JPEG encode failed for frame; skipping")
                continue
            jpeg_bytes = jbuf.tobytes()

            t_now = time.time()
            if t_prev is not None:
                dt = t_now - t_prev
                if dt > 1e-6:
                    inst = 1.0 / dt
                    fps_ema = 0.9 * fps_ema + 0.1 * inst if fps_ema > 0 else inst
            t_prev = t_now

            pub.send(
                {
                    "msg_type": MSG_TYPE,
                    "ts": t_now,
                    "jpeg": jpeg_bytes,
                    "size": (int(bgr.shape[1]), int(bgr.shape[0])),
                    "K": K,
                    "fps_ema": float(fps_ema),
                }
            )
            n_published += 1

            if t_now - last_stats_t >= cfg.stats_every_sec:
                LOGGER.info(
                    "published=%d no_frame=%d fps_ema=%.2f",
                    n_published,
                    n_no_frame,
                    fps_ema,
                )
                last_stats_t = t_now

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        try:
            pipe.stop()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("pipe.stop() raised: %s", exc)
        pub.close()
        LOGGER.info("RealSense worker stopped. total_published=%d", n_published)


def _parse_args() -> WorkerConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--width", type=int, default=DEFAULT_W)
    p.add_argument("--height", type=int, default=DEFAULT_H)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    p.add_argument("--stats-every-sec", type=float, default=5.0)
    args = p.parse_args()
    return WorkerConfig(
        port=args.port,
        width=args.width,
        height=args.height,
        fps=args.fps,
        jpeg_quality=args.jpeg_quality,
        stats_every_sec=args.stats_every_sec,
    )


if __name__ == "__main__":
    run(_parse_args())
