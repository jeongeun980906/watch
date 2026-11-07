"""
robotwin_obj_ret_zmq_server.py

Object-aware variant of ``robotwin_ret_zmq_server.py``: same ZMQ REP protocol,
same model + dataset_stats + logging, only the retrieval class differs.

Adds an SAM3-detected object-position L2 term to the per-rollout initial-scene
filter (cans + plasticbox positions, normalized to [0, 1] via image_size).
SAM3 runs ONCE per rollout (on the first ``act`` after ``reset``) so the
extra cost is bounded — per-step retrieval scoring is unchanged.

Pool object positions come from the offline cache produced by
``robotwin_rag_tools/scripts/run_object_extract.py``:
    {obj_init_root}/{task}/{split}/episode{N}.npz

Usage:
    uv run --extra cu128 \\
        -m cosmos_policy.experiments.robot.robotwin_ret.robotwin_obj_ret_zmq_server \\
        --config cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_inference_only \\
        --ckpt_path /mnt/ddn/tmp/.../model_000010000.pt \\
        --config_file cosmos_policy/config/config.py \\
        --dataset_stats_path /mnt/ddn/.../_cosmos_cache/.../dataset_statistics.json \\
        --t5_text_embeddings_path /mnt/ddn/dataset/RoboTwin2.0/dataset/t5_embeddings.pkl \\
        --pool_dataset_root /mnt/ddn/dataset/RoboTwin2.0/dataset \\
        --pool_task place_cans_plasticbox \\
        --pool_splits arx-x5_clean_50 arx-x5_randomized_500 \\
        --n_initial_filter 10 --w_obj 1.0
"""

import json
import pickle
import sys
import traceback
from dataclasses import dataclass
from typing import Optional

# Inject the embodiment sentinel so cosmos_policy.constants picks ROBOTWIN
# (16/16/50). Must run before any cosmos_policy import.
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
from cosmos_policy.experiments.robot.robotwin_ret.robotwin_ret_zmq_server import (  # noqa: E402
    PolicyEvalConfig as _BasePolicyEvalConfig,
    _RetrievalLogger,
    _handle_act,
)
from cosmos_policy.experiments.robot.robotwin_ret.retrieval_object_filter import (  # noqa: E402
    RoboTwinObjectAwareRetrieval,
)


@dataclass
class PolicyEvalConfig(_BasePolicyEvalConfig):
    """Extends the base RAG server config with object-prefilter knobs."""
    # fmt: off

    # Path to the offline object_init cache produced by robotwin_rag_tools.
    obj_init_root: str = "/mnt/ddn/robotwin_rag_tools/cache/object_init"

    # Path to the JSON config that names objects + prompts + IOU rules per
    # task. Same file consumed by robotwin_rag_tools/scripts/run_object_extract.py
    # so offline pool extraction and online query detection stay in lockstep.
    obj_init_config_path: str = (
        "/mnt/ddn/robotwin_rag_tools/robotwin_rag/configs/object_init_tasks.json"
    )

    # Penalty weight for object-landmark L2 in the initial-scene filter.
    # Normalized [0, 1] coords → distances are tiny (~0.01–0.1), so values
    # around 1.0 keep this term comparable to the DINO+proprio terms.
    w_obj: float = 1.0
    # Categorical shape-class step penalty in the prefilter distance:
    #   aspect = max(w, h) / min(w, h)
    #   class = "long" if aspect > aspect_threshold else "square"
    #   d_pair = ||xy_q - xy_p||₂ + w_obj_shape * (class_q != class_p)
    # 0 = legacy xy-only; >0 pushes shape-mismatched candidates down so
    # within the same shape class candidates rank purely by centroid xy.
    w_obj_shape: float = 0.0
    aspect_threshold: float = 1.5
    # Restrict the shape term (above) to a subset of object names. Empty =
    # apply to all (legacy). Comma-separated list e.g. ``"burger"`` or
    # ``"burger,fries"`` makes the shape penalty count only for those classes;
    # other landmarks still contribute their xy-only distance.
    w_obj_size_objects: str = ""
    # Continuous (w, h) L1 penalty:
    #   d_pair += w_obj_wh_l1 * (|Δw_norm| + |Δh_norm|)
    # Captures orientation (horizontal vs vertical roller share aspect but
    # differ in (w, h)). 0 = legacy. ``w_obj_wh_objects`` (comma-separated)
    # restricts the term to specific landmarks; empty applies to all.
    w_obj_wh_l1: float = 0.0
    w_obj_wh_objects: str = ""
    # 4-quadrant bbox-RGB signature L2 added to per-pair distance. Captures
    # bbox-internal asymmetry (handle vs body, 180° flip) that (w, h) misses.
    # Cached field name: ``<obj>_quad_sig`` (computed by run_object_extract.py
    # ≥ 2026-05-19). ``w_obj_quad_sig_objects`` (comma-separated) restricts
    # to specific landmarks; empty applies to all. 0 = disabled.
    w_obj_quad_sig: float = 0.0
    w_obj_quad_sig_objects: str = ""
    # Restrict the xy-distance ranking term to a subset of object names.
    # Empty = average over all detected objects (legacy). Comma-separated
    # e.g. ``"burger,fries"`` makes the prefilter ignore tray's centroid
    # (still requires it to exist) and focus ranking on the manipulands —
    # avoids diluting the picky burger/fries position match through a
    # stable scene anchor like the tray.
    obj_xy_objects: str = ""
    # Per-object xy weight inside the prefilter average. Comma-separated
    # ``name:weight`` pairs, e.g. ``"burger:1.0,fries:3.0"``. Names not
    # listed default to 1.0. Use to keep a low-variance class (fries: cx
    # std ~3 in clean) from being drowned out by a high-variance one
    # (burger: cx std ~25 in randomized) in the mean.
    obj_xy_weights: str = ""

    # SAM3 lives in a sidecar subprocess so the cosmos_policy uv env (pinned
    # to transformers==4.57.1, no Sam3Model) doesn't have to know about it.
    # The daemon runs under a Python that has transformers>=5.x installed.
    sam3_daemon_python: str = "/mnt/ddn/robotwin_rag_tools/.venv/bin/python"
    sam3_daemon_script: str = "/mnt/ddn/robotwin_rag_tools/scripts/sam3_daemon.py"
    sam3_daemon_pythonpath: str = "/mnt/ddn/robotwin_rag_tools"

    # ── Progression-prefilter knobs (multi-task pool) ──────────────────
    # Comma-separated list of pool task names. When empty, falls back to a
    # single-task pool of ``pool_task``. When set (e.g. for progression eval
    # over pool_1/2/3), the prefilter scores every (task, ep) entry against
    # the query using DINO + T5 + greedy SAM3 with exact-lang bonus.
    pool_tasks: str = ""
    # Progression prefilter weights — defaults match
    # ``render_progression_runtime_video.py`` (the renderer used to validate
    # progression retrieval offline). EXACT-MATCH cases auto-fall-back to the
    # legacy DINO+proprio prefilter inside _compute_initial_filter.
    w_dino_prefilter: float = 5.0
    w_t5_prefilter: float = 0.5
    w_sam3_prefilter: float = 5.0
    sam3_tau: float = 0.4
    # Prefilter hard constraints (renderer parity).
    # ``max_per_task``: cap eps per pool task in the top-N.
    # ``exact_task_only_when_available``: drop non-exact candidates when an
    # exact-language match exists in the pool.
    # ``diversify_when_no_exact``: legacy best-2-tasks split; disabled now.
    max_per_task: int = 3
    exact_task_only_when_available: bool = True
    diversify_when_no_exact: bool = False
    # Chunk-0 (first action chunk) rescore weights: xy-emphasized greedy SAM3
    # to anchor the pool ep with the best object image-position match.
    chunk0_w_sam3_xy: float = 5.0
    chunk0_sam3_w_xy: float = 2.0
    chunk0_sam3_w_size: float = 0.2
    chunk0_sam3_w_shape: float = 0.2
    # Per-chunk (k>=1) ep-level biases added on top of DINO + proprio.
    # No-op when the prefilter detects an exact-task pool.
    w_t5: float = 5.0
    w_sam3: float = 2.0
    # Query task name for the T5 exact-lang bonus. When empty, defaults to
    # ``pool_task`` (legacy single-task pool path).
    query_task_name: str = ""
    # Path to the T5 embeddings pickle (task-name keyed). Required for the
    # T5 fuzzy term to fire; empty disables it gracefully.
    t5_path: str = "/mnt/ddn/dataset/RoboTwin2.0/dataset/t5_embeddings.pkl"
    # fmt: on


@draccus.wrap()
def serve_robotwin_obj_ret(cfg: PolicyEvalConfig) -> None:
    """ZMQ server entry point — mirrors ``serve_robotwin_ret`` but constructs
    :class:`RoboTwinObjectAwareRetrieval` with the extra object-prefilter
    knobs. Per-step request handling is the same path."""
    if cfg.t5_text_embeddings_path:
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    print(f"[robotwin_obj_ret_zmq_server] loading model from {cfg.ckpt_path}", flush=True)
    model, _cosmos_config = get_model(cfg)
    print("[robotwin_obj_ret_zmq_server] model loaded.", flush=True)

    print(
        f"[robotwin_obj_ret_zmq_server] building retrieval pool: task={cfg.pool_task} "
        f"splits={cfg.pool_splits}",
        flush=True,
    )
    # Parse multi-task pool list. Empty CLI value → single-task pool (legacy).
    _pool_tasks_list = [
        t.strip() for t in cfg.pool_tasks.split(",") if t.strip()
    ] if cfg.pool_tasks else None
    if _pool_tasks_list:
        print(
            f"[robotwin_obj_ret_zmq_server] MULTI-TASK pool: query={cfg.pool_task}  "
            f"pool_tasks={_pool_tasks_list}",
            flush=True,
        )
    retrieval = RoboTwinObjectAwareRetrieval(
        dataset_root=cfg.pool_dataset_root,
        task=cfg.pool_task,
        pool_splits=cfg.pool_splits,
        pool_tasks=_pool_tasks_list,
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
        # Object-prefilter knobs (prompts/counts come from the JSON config).
        obj_init_root=cfg.obj_init_root,
        obj_init_config_path=cfg.obj_init_config_path,
        w_obj=cfg.w_obj,
        w_obj_shape=cfg.w_obj_shape,
        aspect_threshold=cfg.aspect_threshold,
        w_obj_size_objects=tuple(
            n.strip() for n in cfg.w_obj_size_objects.split(",") if n.strip()
        ),
        w_obj_wh_l1=cfg.w_obj_wh_l1,
        w_obj_quad_sig=cfg.w_obj_quad_sig,
        w_obj_quad_sig_objects=tuple(
            n.strip() for n in cfg.w_obj_quad_sig_objects.split(",") if n.strip()
        ),
        w_obj_wh_objects=tuple(
            n.strip() for n in cfg.w_obj_wh_objects.split(",") if n.strip()
        ),
        obj_xy_objects=tuple(
            n.strip() for n in cfg.obj_xy_objects.split(",") if n.strip()
        ),
        obj_xy_weights={
            kv.split(":")[0].strip(): float(kv.split(":")[1])
            for kv in cfg.obj_xy_weights.split(",")
            if ":" in kv and kv.strip()
        } or None,
        sam3_daemon_python=cfg.sam3_daemon_python,
        sam3_daemon_script=cfg.sam3_daemon_script,
        sam3_daemon_pythonpath=cfg.sam3_daemon_pythonpath,
        # Progression-prefilter (DINO + T5 + greedy SAM3).
        w_dino_prefilter=cfg.w_dino_prefilter,
        w_t5_prefilter=cfg.w_t5_prefilter,
        w_sam3_prefilter=cfg.w_sam3_prefilter,
        sam3_tau=cfg.sam3_tau,
        max_per_task=cfg.max_per_task,
        exact_task_only_when_available=cfg.exact_task_only_when_available,
        diversify_when_no_exact=cfg.diversify_when_no_exact,
        chunk0_w_sam3_xy=cfg.chunk0_w_sam3_xy,
        chunk0_sam3_w_xy=cfg.chunk0_sam3_w_xy,
        chunk0_sam3_w_size=cfg.chunk0_sam3_w_size,
        chunk0_sam3_w_shape=cfg.chunk0_sam3_w_shape,
        w_t5=cfg.w_t5,
        w_sam3=cfg.w_sam3,
        t5_path=cfg.t5_path,
        query_task_name=cfg.query_task_name if cfg.query_task_name else None,
    )
    print("[robotwin_obj_ret_zmq_server] retrieval pool ready.", flush=True)

    delta_stats: Optional[dict] = None
    if cfg.use_residual_actions:
        if not cfg.delta_stats_path:
            raise RuntimeError(
                "use_residual_actions=True but --delta_stats_path was not given."
            )
        with open(cfg.delta_stats_path, "r") as f:
            js = json.load(f)
        delta_stats = {k: np.asarray(v, dtype=np.float32) for k, v in js.items()}
        print(
            f"[robotwin_obj_ret_zmq_server] residual mode ON; loaded delta stats "
            f"from {cfg.delta_stats_path}",
            flush=True,
        )

    logger: Optional[_RetrievalLogger] = None
    if cfg.retrieval_log_dir:
        logger = _RetrievalLogger(
            root_dir=cfg.retrieval_log_dir,
            video_fps=cfg.video_fps,
            save_generated=cfg.save_generated_videos,
        )
        # Hand the logger to the retrieval class so each rollout's SAM3
        # preview lands in the same per-episode dir as retrieved.mp4.
        retrieval.attach_logger(logger)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(cfg.endpoint)
    print(f"[robotwin_obj_ret_zmq_server] listening on {cfg.endpoint}", flush=True)

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
                    reply = _handle_act(
                        cfg, model, dataset_stats, retrieval, req, logger, delta_stats
                    )
                else:
                    reply = {"ok": False, "error": f"unknown op: {op!r}"}
            except Exception as exc:
                traceback.print_exc()
                reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            sock.send(pickle.dumps(reply, protocol=pickle.HIGHEST_PROTOCOL))
    except KeyboardInterrupt:
        print("\n[robotwin_obj_ret_zmq_server] shutting down.", flush=True)
    finally:
        if logger is not None:
            try:
                logger.flush()
            except Exception:
                traceback.print_exc()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    serve_robotwin_obj_ret()
