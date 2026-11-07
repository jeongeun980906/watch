"""
robotwin_qpos_zmq_server.py

Joint-space (qpos, 14d) variant of ``robotwin_zmq_server.py``. Serves a
trained Cosmos Policy that was finetuned with ``RoboTwinJointDataset``
(reads /joint_action/vector, action layout
``[L_arm(6) + L_grip(1) + R_arm(6) + R_grip(1)]``).

Same ZMQ REP shape as the EE server, the RoboTwin client only changes:
  - obs.proprio:         16d EE pose -> 14d joint state
  - take_action call:    action_type='ee' -> action_type='qpos'

On the model side the only thing that differs is ``ACTION_DIM`` /
``PROPRIO_DIM`` (14 vs 16). That comes from ``cosmos_policy.constants``
which reads ``sys.argv`` *at import time*, so the embodiment sentinel
must be injected before any cosmos_policy module is imported.

Camera slots and the get_action segment ordering (INDICES_TO_REPLACE)
are identical to the EE variant — that branch in
``cosmos_utils.get_action`` is keyed on ``cfg.suite == 'robotwin'`` and
is action-dim agnostic — so we keep ``cfg.suite='robotwin'`` and just
let the constants module pick the 14d block.

Usage example:
    uv run --extra cu128 \\
        -m cosmos_policy.experiments.robot.robotwin_qpos_zmq_server \\
        --config cosmos_predict2_2b_480p_robotwin_debug_place_cans_plasticbox__qpos__inference_only \\
        --ckpt_path /mnt/ddn/tmp/.../__qpos/checkpoints/model_000005000.pt \\
        --config_file cosmos_policy/config/config.py \\
        --dataset_stats_path /mnt/ddn/.../_cosmos_cache/aloha-agilex_clean_50__1tasks_df89a568__joint/dataset_statistics.json \\
        --t5_text_embeddings_path /mnt/ddn/dataset/RoboTwin2.0/dataset/t5_embeddings.pkl \\
        --predict_future_states True

Protocol (pickle over ZMQ REQ/REP):
    request  = {"op": "ping"}
        reply = {"ok": True, "suite": "robotwin", "action_dim": 14}
    request  = {"op": "reset"}
        reply = {"ok": True}
    request  = {"op": "act",
               "obs": {"primary_image": np.ndarray(H,W,3) uint8,
                       "left_wrist_image": ..., "right_wrist_image": ...,
                       "proprio": np.ndarray(14,) float32},
               "instruction": str,
               "seed": int}
        reply = {"ok": True,
                 "actions": np.ndarray(chunk, 14) float32,
                 "future_image_predictions": {"future_image": uint8(H,W,3), ...}}
    any error -> {"ok": False, "error": str}
"""

import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

# IMPORTANT: inject the embodiment sentinel BEFORE importing anything from
# cosmos_policy. cosmos_policy/constants.py probes sys.argv at import time
# and freezes ACTION_DIM / PROPRIO_DIM. We want the qpos block (14/14),
# not the EE block (16/16) that the plain "robotwin" sentinel would pick.
if not any("robotwin_qpos" in arg.lower() for arg in sys.argv):
    sys.argv.append("--cosmos-embodiment=robotwin_qpos")

from cosmos_policy.experiments.robot.robot_utils import DATE_TIME, log_message, setup_logging  # noqa: E402

import draccus  # noqa: E402
import numpy as np  # noqa: E402
import zmq  # noqa: E402

from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)

ACTION_DIM_QPOS = 14


@dataclass
class PolicyEvalConfig:
    # fmt: off
    # cosmos_utils.get_action() branches on cfg.suite. The "robotwin" branch's
    # camera slots / INDICES_TO_REPLACE match qpos exactly — only ACTION_DIM
    # at module scope differs (handled by the sentinel above). So keep this
    # as "robotwin", do not invent a new suite name.
    suite: str = "robotwin"

    ##########################################################################
    # Cosmos Policy model parameters (mirror robotwin_zmq_server)
    ##########################################################################
    model_family: str = "cosmos"
    config: str = ""                                            # Inference experiment name
    ckpt_path: str = ""                                         # Trained checkpoint .pt
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"

    use_third_person_image: bool = True                         # head_camera -> primary
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
    from io import BytesIO

    from PIL import Image

    if arr.ndim != 3 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"expected HxWx3 uint8 image, got shape={arr.shape} dtype={arr.dtype}")
    buf = BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _serialize_future_images(future_image_predictions) -> dict:
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
    log_message(f"time spent: {query_time}")
    actions = ret["actions"]
    if isinstance(actions, list):
        actions = np.stack([_to_np_action(a) for a in actions], axis=0)
    else:
        actions = _to_np_action(actions)

    if actions.ndim == 2 and actions.shape[1] != ACTION_DIM_QPOS:
        raise ValueError(
            f"qpos server received {actions.shape[1]}-dim actions; expected "
            f"{ACTION_DIM_QPOS}. Did the embodiment sentinel get clobbered "
            f"or was the model trained with EE actions?"
        )

    reply = {"ok": True, "actions": actions.astype(np.float32).tolist()}
    fips = ret.get("future_image_predictions")
    serialized = _serialize_future_images(fips) if fips else {}
    if serialized:
        reply["future_image_predictions"] = serialized
    return reply


@draccus.wrap()
def serve_robotwin_qpos(cfg: PolicyEvalConfig) -> None:
    if cfg.t5_text_embeddings_path:
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    print(f"[robotwin_qpos_zmq_server] loading model from {cfg.ckpt_path}", flush=True)
    model, _cosmos_config = get_model(cfg)
    print("[robotwin_qpos_zmq_server] model loaded.", flush=True)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(cfg.endpoint)
    print(f"[robotwin_qpos_zmq_server] listening on {cfg.endpoint}", flush=True)

    try:
        while True:
            raw = sock.recv()
            print("RECEIVED")
            try:
                req = pickle.loads(raw)
                op = req.get("op")
                if op == "ping":
                    reply = {"ok": True, "suite": cfg.suite, "action_dim": ACTION_DIM_QPOS}
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
        print("\n[robotwin_qpos_zmq_server] shutting down.", flush=True)
    finally:
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    serve_robotwin_qpos()
