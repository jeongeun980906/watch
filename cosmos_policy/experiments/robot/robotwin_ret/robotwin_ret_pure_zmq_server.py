"""
robotwin_ret_pure_zmq_server.py

Retrieval-only RoboTwin server: skips the Cosmos diffusion policy entirely
and just returns the raw retrieved EEF action chunk as the model's action
output. Useful as a sanity check / lower-bound baseline showing what the
pool's nearest match alone can drive in the simulator.

Same ZMQ REP protocol as robotwin_obj_ret_zmq_server.py (ping/reset/act),
same retrieval class + scoring knobs (SAM3 obj filter + DINO + proprio +
chunk-delta). Differences:

  * No Cosmos model is loaded (no ckpt_path required).
  * No residual delta math, no clipping (set RESIDUAL_DELTA_CLIP_SCALE
    has no effect here — the pool action chunk is returned verbatim).
  * Per-act reply: actions = ret["retrieved_actions"][:chunk_size]
                  retrieval_meta = ret["retrieval_meta"]

Usage:
    uv run --extra cu128 --group robotwin --python 3.10 \\
      -m cosmos_policy.experiments.robot.robotwin_ret.robotwin_ret_pure_zmq_server \\
        --config_file cosmos_policy/config/config.py \\
        --dataset_stats_path /mnt/ddn/.../_cosmos_cache/.../dataset_statistics.json \\
        --pool_dataset_root /mnt/ddn/dataset/RoboTwin2.0/dataset \\
        --pool_task place_cans_plasticbox \\
        --pool_splits '["ur5_clean_50"]' \\
        --dino_cache_root /mnt/ddn/robotwin_rag_tools/cache \\
        --obj_init_root /mnt/ddn/robotwin_rag_tools/cache/object_init \\
        --obj_init_config_path /mnt/ddn/robotwin_rag_tools/robotwin_rag/configs/object_init_tasks.json \\
        --chunk_size 16 --endpoint tcp://127.0.0.1:5554

The config / cosmos sentinel injection mirrors the model-backed servers so
shared imports (constants, dataset stats loader, retrieval class) behave
identically.
"""

import json
import pickle
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

# Inject embodiment sentinel before cosmos_policy imports.
if not any("robotwin" in arg.lower() for arg in sys.argv):
    sys.argv.append("--cosmos-embodiment=robotwin")

import draccus  # noqa: E402
import numpy as np  # noqa: E402
import zmq  # noqa: E402

from cosmos_policy.experiments.robot.cosmos_utils import (  # noqa: E402
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.robotwin_ret.robotwin_ret_zmq_server import (  # noqa: E402
    PolicyEvalConfig as _BasePolicyEvalConfig,
    _RetrievalLogger,
    _to_np_action,
    _serialize_future_images,
    log_message,
)
from cosmos_policy.experiments.robot.robotwin_ret.robotwin_obj_ret_zmq_server import (  # noqa: E402
    PolicyEvalConfig as _ObjPolicyEvalConfig,
)
from cosmos_policy.experiments.robot.robotwin_ret.retrieval_object_filter import (  # noqa: E402
    RoboTwinObjectAwareRetrieval,
)


@dataclass
class PolicyEvalConfig(_ObjPolicyEvalConfig):
    """Retrieval-only server config: inherits the SAM3 object-aware fields so
    pool/retrieval setup matches the model-backed obj_ret server. Cosmos
    model fields (ckpt_path etc.) are inherited but unused at runtime."""
    pass


def _handle_act_pure(
    cfg: PolicyEvalConfig,
    dataset_stats: dict,
    retrieval: RoboTwinObjectAwareRetrieval,
    req: dict,
    logger: Optional[_RetrievalLogger] = None,
) -> dict:
    """Pure retrieval: return the matched pool action chunk verbatim."""
    obs = req["obs"]
    query_proprio_history = obs.get("proprio_history")

    t0 = time.time()
    ret = retrieval.get_retrieved(
        query_head_image=obs["primary_image"],
        query_proprio=obs["proprio"],
        dataset_stats=dataset_stats,
        query_proprio_history=query_proprio_history,
    )
    t_ret = time.time() - t0

    # Trim to chunk_size like the model-backed path does (get_action returns
    # exactly chunk_size actions).
    raw_actions = np.asarray(ret["retrieved_actions"], dtype=np.float32)
    raw_actions = raw_actions[: int(cfg.chunk_size)]

    log_message(
        f"retrieval={t_ret:.3f}s  policy=SKIPPED  "
        f"match={ret['retrieval_meta']['split']}/ep{ret['retrieval_meta']['ep_num']}/"
        f"f{ret['retrieval_meta']['frame']} score={ret['retrieval_meta']['score']:.3f}"
    )

    reply = {
        "ok": True,
        "actions": raw_actions.astype(np.float32).tolist(),
        "retrieval_meta": ret["retrieval_meta"],
    }
    if logger is not None:
        # No model future-image predictions in pure mode; log retrieval only.
        logger.log_step(ret, None)
    return reply


@draccus.wrap()
def serve_robotwin_ret_pure(cfg: PolicyEvalConfig) -> None:
    if cfg.t5_text_embeddings_path:
        # Harmless: cache loader is shared across server variants and never
        # touched here at request time (no model forward).
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    print(
        f"[robotwin_ret_pure_zmq_server] building retrieval pool: task={cfg.pool_task} "
        f"splits={cfg.pool_splits}  (model NOT loaded)",
        flush=True,
    )
    retrieval = RoboTwinObjectAwareRetrieval(
        dataset_root=cfg.pool_dataset_root,
        task=cfg.pool_task,
        pool_splits=cfg.pool_splits,
        dino_cache_root=cfg.dino_cache_root,
        chunk_size=cfg.chunk_size,
        ret_image_subsample=cfg.ret_image_subsample,
        proprio_history_window=cfg.proprio_history_window,
        w_dino=cfg.w_dino,
        w_prop=cfg.w_prop,
        w_prop_curr=cfg.w_prop_curr,
        w_grip_curr=cfg.w_grip_curr,
        w_pos_curr=cfg.w_pos_curr,
        proprio_repr=cfg.proprio_repr,
        top_k=cfg.retrieval_top_k,
        n_initial_filter=cfg.n_initial_filter,
        phase_alignment_weight=cfg.phase_alignment_weight,
        debug_dump_dir=cfg.debug_dump_dir if cfg.debug_dump_dir else None,
        debug_dump_top_k=cfg.debug_dump_top_k,
        home_pose_json=cfg.home_pose_json if cfg.home_pose_json else None,
        home_pose_auto_dir=cfg.home_pose_auto_dir if cfg.home_pose_auto_dir else None,
        home_transform_beta=cfg.home_transform_beta,
        home_transform_p=cfg.home_transform_p,
        forward_bias_weight=cfg.forward_bias_weight,
        force_initial_pool_fr0_n_acts=cfg.force_initial_pool_fr0_n_acts,
        advance_brake_weight=cfg.advance_brake_weight,
        advance_brake_max_step=cfg.advance_brake_max_step,
        disable_dino=cfg.disable_dino,
        # SAM3 object-aware fields:
        obj_init_root=cfg.obj_init_root,
        obj_init_config_path=cfg.obj_init_config_path,
        w_obj=cfg.w_obj,
        w_obj_shape=cfg.w_obj_shape,
        aspect_threshold=cfg.aspect_threshold,
        w_obj_size_objects=tuple(
            x for x in cfg.w_obj_size_objects.split(",") if x.strip()
        ),
        w_obj_wh_l1=cfg.w_obj_wh_l1,
        w_obj_wh_objects=tuple(
            x for x in cfg.w_obj_wh_objects.split(",") if x.strip()
        ),
        w_obj_quad_sig=cfg.w_obj_quad_sig,
        w_obj_quad_sig_objects=tuple(
            x for x in cfg.w_obj_quad_sig_objects.split(",") if x.strip()
        ),
        obj_xy_objects=tuple(
            x for x in cfg.obj_xy_objects.split(",") if x.strip()
        ),
        obj_xy_weights=cfg.obj_xy_weights,
        sam3_daemon_python=cfg.sam3_daemon_python,
        sam3_daemon_script=cfg.sam3_daemon_script,
        sam3_daemon_pythonpath=cfg.sam3_daemon_pythonpath,
    )
    print("[robotwin_ret_pure_zmq_server] retrieval pool ready.", flush=True)

    logger: Optional[_RetrievalLogger] = None
    if cfg.retrieval_log_dir:
        logger = _RetrievalLogger(
            root_dir=cfg.retrieval_log_dir,
            video_fps=cfg.video_fps,
            save_generated=False,  # no model forward → nothing to save
        )
        # Hand the logger to the retrieval class so each rollout's SAM3
        # preview lands in the same per-episode dir as retrieved.mp4.
        # (Was missing — preview was never written in pure mode.)
        retrieval.attach_logger(logger)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(cfg.endpoint)
    print(f"[robotwin_ret_pure_zmq_server] listening on {cfg.endpoint}", flush=True)

    try:
        while True:
            raw = sock.recv()
            try:
                req = pickle.loads(raw)
                op = req.get("op")
                if op == "ping":
                    reply = {"ok": True, "suite": cfg.suite, "action_dim": 16}
                elif op == "reset":
                    if logger is not None:
                        logger.flush()
                    retrieval.reset()
                    reply = {"ok": True}
                elif op == "act":
                    reply = _handle_act_pure(
                        cfg, dataset_stats, retrieval, req, logger
                    )
                else:
                    reply = {"ok": False, "error": f"unknown op: {op}"}
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[robotwin_ret_pure_zmq_server] op error: {exc}\n{tb}", flush=True)
                reply = {"ok": False, "error": str(exc), "traceback": tb}
            sock.send(pickle.dumps(reply))
    finally:
        if logger is not None:
            logger.flush()


if __name__ == "__main__":
    serve_robotwin_ret_pure()
