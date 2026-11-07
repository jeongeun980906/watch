"""
robotwin_zmq_server.py

Serves a trained Cosmos Policy as a ZMQ REP endpoint so a RoboTwin client
(running in a separate conda env, where importing cosmos_policy is blocked
by transformer_engine and friends) can request actions over localhost.

Modeled after ``run_pusht_eval.py`` / ``run_aloha_eval.py``: same
``@dataclass PolicyEvalConfig`` + ``@draccus.wrap()`` shape, same flag
names. The only RoboTwin-specific bits are the camera slot mapping
(handled inside ``cosmos_utils.get_action`` when ``cfg.suite == 'robotwin'``)
and the ZMQ transport layer at the bottom.

Usage example:
    uv run --extra cu128 \\
        -m cosmos_policy.experiments.robot.robotwin_zmq_server \\
        --config cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__inference_only \\
        --ckpt_path /mnt/ddn/tmp/.../model_000005000.pt \\
        --config_file cosmos_policy/config/config.py \\
        --dataset_stats_path /mnt/ddn/.../dataset_statistics.json \\
        --t5_text_embeddings_path /mnt/ddn/dataset/RoboTwin2.0/dataset/t5_embeddings.pkl \\
        --predict_future_states True

Protocol (pickle over ZMQ REQ/REP):
    request  = {"op": "ping"}
        reply = {"ok": True, "suite": "robotwin", "action_dim": 16}
    request  = {"op": "reset"}
        reply = {"ok": True}
    request  = {"op": "act",
               "obs": {"primary_image": np.ndarray(H,W,3) uint8,
                       "left_wrist_image": ..., "right_wrist_image": ...,
                       "proprio": np.ndarray(16,) float32},
               "instruction": str,
               "seed": int}
        reply = {"ok": True,
                 "actions": np.ndarray(chunk, 16) float32,
                 "future_image_predictions": {"future_image": uint8(H,W,3), ...}}
    any error → {"ok": False, "error": str}
"""

import pickle
import sys
import traceback
from dataclasses import dataclass
from typing import Optional
import time
from cosmos_policy.experiments.robot.robot_utils import DATE_TIME, log_message, setup_logging
# Ensure cosmos_policy.constants picks ROBOTWIN_CONSTANTS (16/16/50)
# instead of the LIBERO default. The library probes sys.argv at import time.
if not any("robotwin" in arg.lower() for arg in sys.argv):
    sys.argv.append("--cosmos-embodiment=robotwin")

import draccus  # noqa: E402
import numpy as np  # noqa: E402
import zmq  # noqa: E402

from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)


@dataclass
class PolicyEvalConfig:
    # fmt: off
    suite: str = "robotwin"                                     # cosmos_utils branches on this

    ##########################################################################
    # Cosmos Policy model parameters (mirror run_pusht_eval.PolicyEvalConfig)
    ##########################################################################
    model_family: str = "cosmos"
    config: str = ""                                            # Inference experiment name
    ckpt_path: str = ""                                         # Trained checkpoint .pt
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"

    use_third_person_image: bool = True                         # head_camera → primary
    num_third_person_images: int = 1
    use_wrist_image: bool = True                                # left_camera + right_camera as wrists
    num_wrist_images: int = 2
    use_proprio: bool = True
    flip_images: bool = False
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    ar_future_prediction: bool = False
    ar_value_prediction: bool = False
    ar_qvalue_prediction: bool = False
    num_denoising_steps_action: int = 5
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    shift: float = 5.0                                          # rectified flow (predict2.5); ignored by EDM
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 16
    num_open_loop_steps: int = 50

    deterministic: bool = True
    randomize_seed: bool = False
    seed: int = 0
    num_queries_best_of_n: int = 1
    use_parallel_inference: bool = False
    available_gpus: str = "0"
    parallel_timeout: int = 600

    ##########################################################################
    # Future state / value heads (off by default — turn on to record videos)
    ##########################################################################
    predict_future_states: bool = True
    predict_values: bool = False
    generate_future_state_and_value_in_parallel: bool = True
    use_ensemble_future_state_predictions: bool = False
    use_ensemble_value_predictions: bool = False
    num_future_state_predictions_in_ensemble: int = 1
    num_value_predictions_in_ensemble: int = 1
    future_state_ensemble_aggregation_scheme: str = "mean"
    value_ensemble_aggregation_scheme: str = "mean"
    mask_current_state_action_for_value_prediction: bool = False
    mask_future_state_for_qvalue_prediction: bool = False
    search_depth: int = 1
    t: int = 0

    ##########################################################################
    # ZMQ transport
    ##########################################################################
    endpoint: str = "tcp://127.0.0.1:5555"

    ##########################################################################
    # Logging
    ##########################################################################
    local_log_dir: str = "./experiments/logs"
    run_id_note: Optional[str] = None
    # fmt: on


def _to_np_action(a) -> np.ndarray:
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float32).reshape(-1)


def _encode_image_png(arr: np.ndarray) -> bytes:
    """Pack an HxWx3 uint8 image as PNG bytes (lossless, numpy-agnostic).

    Client side reverses with ``PIL.Image.open(BytesIO(b))`` then
    ``np.asarray(...)``. PNG is exact for uint8 RGB, and stays small
    enough to fit comfortably in a single ZMQ frame.
    """
    from io import BytesIO

    from PIL import Image

    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"expected HxWx3 uint8 image, got shape={arr.shape} dtype={arr.dtype}")
    buf = BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _serialize_future_images(future_image_predictions) -> dict:
    """Convert torch/np images to PNG bytes for transport.

    Uses PIL/PNG instead of pickling raw ndarrays because the cosmos env
    runs numpy 2.x while the RoboTwin env runs numpy 1.26 — numpy's
    pickle reducer references private modules that differ across major
    versions, so the receiver fails on ``numpy._core.numeric``.
    """
    out = {}
    if not isinstance(future_image_predictions, dict):
        return out
    for k, v in future_image_predictions.items():
        if v is None:
            continue
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        v = np.asarray(v)
        if v.dtype != np.uint8:
            v = np.clip(v, 0, 255).astype(np.uint8)
        out[k] = {"__png__": True, "data": _encode_image_png(v)}
    return out


def _handle_act(cfg: PolicyEvalConfig, model, dataset_stats: dict, req: dict) -> dict:
    
    obs = req["obs"]
    instruction = req["instruction"]
    seed = int(req.get("seed", cfg.seed) or 0)
    start_time = time.time()
    # print("PARSED OBS")
    ret = get_action(
        cfg,
        model,
        dataset_stats,
        obs,
        instruction,
        seed=seed,
        randomize_seed=cfg.randomize_seed,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=cfg.generate_future_state_and_value_in_parallel,
    )
    query_time = time.time() - start_time
    log_message(
        f"time spent: {query_time}",
    )
    actions = ret["actions"]
    if isinstance(actions, list):
        actions = np.stack([_to_np_action(a) for a in actions], axis=0)
    else:
        actions = _to_np_action(actions)

    # Ship actions as a plain Python list so the wire format doesn't depend
    # on numpy's pickle reducer (cosmos numpy 2.x vs RoboTwin numpy 1.26).
    reply = {"ok": True, "actions": actions.astype(np.float32).tolist()}
    fips = ret.get("future_image_predictions")
    serialized = _serialize_future_images(fips) if fips else {}
    if serialized:
        reply["future_image_predictions"] = serialized
    return reply


@draccus.wrap()
def serve_robotwin(cfg: PolicyEvalConfig) -> None:
    """Main server entry point — mirrors run_pusht_eval / run_aloha_eval shape."""
    # Initialize T5 embedding cache (skipped when path is empty → on-the-fly).
    if cfg.t5_text_embeddings_path:
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    # Load dataset stats for action un-normalization / proprio normalization.
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    # Load model.
    print(f"[robotwin_zmq_server] loading model from {cfg.ckpt_path}", flush=True)
    model, _cosmos_config = get_model(cfg)
    print("[robotwin_zmq_server] model loaded.", flush=True)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(cfg.endpoint)
    print(f"[robotwin_zmq_server] listening on {cfg.endpoint}", flush=True)

    try:
        while True:
            raw = sock.recv()
            print("RECIEVED")
            try:
                req = pickle.loads(raw)
                # print(req)
                op = req.get("op")
                if op == "ping":
                    reply = {"ok": True, "suite": cfg.suite, "action_dim": 16}
                elif op == "reset":
                    reply = {"ok": True}
                elif op == "act":
                    reply = _handle_act(cfg, model, dataset_stats, req)
                else:
                    reply = {"ok": False, "error": f"unknown op: {op!r}"}
            except Exception as exc:
                traceback.print_exc()
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            sock.send(pickle.dumps(reply, protocol=pickle.HIGHEST_PROTOCOL))
    except KeyboardInterrupt:
        print("\n[robotwin_zmq_server] shutting down.", flush=True)
    finally:
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    serve_robotwin()
