"""
robotwin_ret_zmq_server.py

RAG (cross-embodiment retrieval-augmented) variant of
``robotwin_zmq_server.py``. Same ZMQ REP protocol so the existing RoboTwin
client (``policy/cosmos_policy/deploy_policy.py``) talks to it without any
change — only difference is that this server runs DINO + proprio retrieval
against a pool of pool-embodiment trajectories on every ``act`` and injects
the retrieved (head, L wrist, R wrist, proprio, action chunk) into the obs
that ``cosmos_utils.get_action`` consumes.

cfg.suite is set to ``"robotwin_ret"`` so the get_action branch picks the
RAG slot layout (mirrors RoboTwinRetDataset.__getitem__):
    0:  blank
    1:  ret proprio                    (placeholder, value injected as
                                        retrieved_proprio_tensor)
    2-4: ret L wrist / R wrist / head  (1 latent each, stride-sampled)
    5:  ret action                     (placeholder, value injected as
                                        retrieved_actions_tensor)
    6:  cur proprio (placeholder)
    7-9: cur L wrist / R wrist / head
    10: action       (placeholder; the model fills this)
    11: future proprio (placeholder)
    12-14: future L wrist / R wrist / head

Usage example:
    uv run --extra cu128 \\
        -m cosmos_policy.experiments.robot.robotwin_ret.robotwin_ret_zmq_server \\
        --config cosmos_predict2_2b_480p_robotwin_rag_debug_place_cans_plasticbox_inference_only \\
        --ckpt_path /mnt/ddn/tmp/.../model_000010000.pt \\
        --config_file cosmos_policy/config/config.py \\
        --dataset_stats_path /mnt/ddn/.../_cosmos_cache/aloha-agilex_clean_50__1tasks_<digest>/dataset_statistics.json \\
        --t5_text_embeddings_path /mnt/ddn/dataset/RoboTwin2.0/dataset/t5_embeddings.pkl \\
        --pool_dataset_root /mnt/ddn/dataset/RoboTwin2.0/dataset \\
        --pool_task place_cans_plasticbox \\
        --pool_splits arx-x5_clean_50 arx-x5_randomized_500

Protocol (pickle over ZMQ REQ/REP — identical to robotwin_zmq_server.py):
    request  = {"op": "ping"}
        reply = {"ok": True, "suite": "robotwin_ret", "action_dim": 16}
    request  = {"op": "reset"}
        reply = {"ok": True}                 # also clears proprio history buffer
    request  = {"op": "act",
               "obs": {primary_image, left_wrist_image, right_wrist_image,
                       proprio(16,)},
               "instruction": str, "seed": int}
        reply = {"ok": True,
                 "actions": (chunk, 16),
                 "future_image_predictions": {...},
                 "retrieval_meta": {split, ep_num, frame, score, ...}}
"""

import json
import os
import pathlib
import pickle
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

# Inject the embodiment sentinel so cosmos_policy.constants picks ROBOTWIN
# (16/16/50). Must run before any cosmos_policy import.
if not any("robotwin" in arg.lower() for arg in sys.argv):
    sys.argv.append("--cosmos-embodiment=robotwin")

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
from cosmos_policy.experiments.robot.robotwin_ret.retrieval import (  # noqa: E402
    RoboTwinDinoProprioRetrieval,
)


@dataclass
class PolicyEvalConfig:
    # fmt: off
    # cfg.suite drives the RAG slot layout in cosmos_utils.get_action.
    suite: str = "robotwin_ret"

    ##########################################################################
    # Cosmos Policy model parameters (mirror robotwin_zmq_server)
    ##########################################################################
    model_family: str = "cosmos"
    config: str = ""                                            # Inference experiment name (RAG variant)
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
    shift: float = 5.0
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 16
    num_open_loop_steps: int = 16

    deterministic: bool = True
    randomize_seed: bool = False
    seed: int = 0
    num_queries_best_of_n: int = 1
    use_parallel_inference: bool = False
    available_gpus: str = "0"
    parallel_timeout: int = 600

    ##########################################################################
    # Future state / value heads
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

    # Retrieval-side ablation knobs (mirror RoboTwinRetDataset training flags).
    # Server passes these through to cosmos_utils.get_action via cfg.use_ret_*.
    use_ret_image: bool = True
    use_ret_state: bool = True

    # Residual learning at inference: when True, the model's output is treated
    # as a normalized delta. The server unnormalizes the delta with delta_stats
    # and adds raw ret_actions to recover the absolute action chunk. Pair with
    # a checkpoint trained from the *_residual config + the delta stats file
    # written next to dataset_statistics.json on first training run.
    use_residual_actions: bool = False
    delta_stats_path: str = ""

    ##########################################################################
    # Pool / retrieval config (NEW vs robotwin_zmq_server)
    ##########################################################################
    pool_dataset_root: str = "/mnt/ddn/dataset/RoboTwin2.0/dataset"
    pool_task: str = "place_cans_plasticbox"
    pool_splits: list = field(default_factory=lambda: [
        "arx-x5_clean_50", "arx-x5_randomized_500",
    ])
    dino_cache_root: str = "/mnt/ddn/robotwin_rag_tools/cache"
    ret_image_subsample: int = 4         # 16-frame window -> 4 frames -> exactly one ret latent
    # K proprio frames (current EE + K-1 history) are differenced into K-1
    # consecutive deltas; matching is uniform-mean L2 over those deltas, so
    # retrieval is driven by direction-of-motion rather than absolute pose.
    proprio_history_window: int = 4
    w_dino: float = 1.0
    # Direction-of-motion (proprio delta) weight. Deltas in normalized space
    # have small magnitude (~0.01–0.1) so this typically wants to be ~5–10×
    # the legacy absolute-proprio weight.
    w_prop: float = 0.5
    # Current absolute proprio weight (||q_curr − p_curr|| in normalized
    # space). Set to 0 to disable absolute matching and use only deltas.
    w_prop_curr: float = 0.5
    # Per-dim emphasis on gripper channels (idx 7=L_grip, 15=R_grip) inside
    # the proprio L2. 1.0 = uniform; 3–5 disambiguates pre-grasp / post-grasp
    # / passive-arm-home cases where grip is the only discriminative dim.
    w_grip_curr: float = 1.0
    # Per-dim emphasis on xyz channels (only used when proprio_repr=='20d_rot6').
    # 6-d rot6 dominates raw L2 by default since it has more dims and ~unit
    # magnitude per dim. w_pos_curr=4 multiplies xyz squared contribution ×4
    # to recover xyz signal. 1.0 = no-op. Matches training-side --pos-weight.
    w_pos_curr: float = 1.0
    # Proprio representation for scoring:
    #   '16d_quat' (default): legacy, normalize via dataset_stats, raw L2 in
    #                          [-1, 1] space.
    #   '20d_rot6': convert proprio to rot6 (sign-flip immune for cross-
    #               embodiment), skip dataset_stats normalization, raw L2.
    # Stored retrieved_proprio output stays 16-d quat either way.
    proprio_repr: str = "16d_quat"
    retrieval_top_k: int = 1
    # Initial-scene filter: at the first act of each episode, score every pool
    # episode by its start-frame DINO + start-frame proprio similarity to the
    # query and keep only the top-N for the rest of the rollout. Cuts the
    # per-step scoring cost ~13× when N=40 and pool=550 episodes. Set to 0 to
    # disable the filter and score the whole pool every step.
    n_initial_filter: int = 40
    # Phase-alignment penalty weight. Score gets −w · |Q_cum(t) − P_cum(E,k)|
    # where Q_cum is the query's cumulative proprio path length since reset
    # and P_cum is the pool's cumulative path length up to frame k (both in
    # query-normalized space). 0 = legacy (no phase term).
    phase_alignment_weight: float = 0.0
    # Per-step diagnostic dump dir. Empty = disabled. When set, every retrieval
    # call appends a JSONL record (top-K breakdown + per-dim diff) and saves
    # query/pool head images. Use to diagnose backstep / wrong-frame matches.
    debug_dump_dir: str = ""
    debug_dump_top_k: int = 10
    # Cross-embodiment home-pose transform. Path to JSON with
    # {aloha_home_proprio, ur5_home_proprio, ...} produced offline. Empty =
    # disabled. β controls the per-arm Gaussian decay; smaller β makes the
    # transform vanish faster as the arm leaves home.
    home_pose_json: str = ""
    home_transform_beta: float = 0.7
    home_transform_p: float = 4.0
    # Soft forward-progress bias: small per-episode bonus when a candidate is
    # ahead of the last pick in that episode, penalty when behind. Tanh-bounded
    # to ±forward_bias_weight; 0 = disabled.
    forward_bias_weight: float = 0.0
    # Force pool_fr=0 for the first N act() calls after reset (mirrors
    # training-NPZ force-fr=0 mechanism). 0 = disabled (default).
    force_initial_pool_fr0_n_acts: int = 0
    # Per-chunk advancement brake: linear penalty when candidate's local_idx
    # exceeds (last_picked + advance_brake_max_step) within the same episode.
    # 0 = disabled (default).
    advance_brake_weight: float = 0.0
    advance_brake_max_step: int = 10
    # Skip per-step DINO ViT-B/14 embed() and use a zero query vector. Trades
    # the DINO scoring term for ~10–30 ms/step savings on Cosmos-2B. Object_init
    # (SAM3) + proprio + chunk-delta still drive retrieval. Default False.
    disable_dino: bool = False
    # When ``home_pose_json`` is empty, retrieval auto-resolves a JSON inside
    # this directory by (task, pool_family). Set to "" to disable auto-resolve.
    home_pose_auto_dir: str = "/mnt/ddn/cosmos_policy/home_poses"

    ##########################################################################
    # ZMQ transport
    ##########################################################################
    endpoint: str = "tcp://127.0.0.1:5555"

    ##########################################################################
    # Logging
    ##########################################################################
    local_log_dir: str = "./experiments/logs"
    run_id_note: Optional[str] = None

    # Retrieval-specific logging.
    # When ``retrieval_log_dir`` is set, the server writes a per-episode folder
    #     {retrieval_log_dir}/{run_tag}/episode_NNNN/
    #         retrieved.mp4         3-panel head|L|R retrieved frames per step
    #         retrieval_meta.jsonl  per-step (split, ep_num, frame, scores)
    # Episode boundary = each ``reset`` op. Flush happens at reset (or process exit).
    retrieval_log_dir: Optional[str] = None
    save_generated_videos: bool = False  # also write generated.mp4 (model future frames)
    video_fps: int = 10
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


def _to_uint8_image(v) -> Optional[np.ndarray]:
    """Coerce torch/np tensor to (H, W, 3) uint8. Returns None on missing."""
    if v is None:
        return None
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    v = np.asarray(v)
    if v.dtype != np.uint8:
        v = np.clip(v, 0, 255).astype(np.uint8)
    return v


def _serialize_future_images(future_image_predictions) -> dict:
    out = {}
    if not isinstance(future_image_predictions, dict):
        return out
    for k, v in future_image_predictions.items():
        arr = _to_uint8_image(v)
        if arr is None:
            continue
        out[k] = {"__png__": True, "data": _encode_image_png(arr)}
    return out


# ---------------------------------------------------------------------------
# Retrieval / generated-video logger (per-episode, flush at reset)
# ---------------------------------------------------------------------------


class _RetrievalLogger:
    """Buffers retrieved + generated frames for the in-progress episode.

    Episode boundary is the next ``reset`` op (or process shutdown). Flush
    writes one MP4 per signal type plus a JSONL of retrieval metadata into
    ``{root_dir}/{run_tag}/episode_NNNN/``.
    """

    def __init__(self, root_dir: str, video_fps: int = 10, save_generated: bool = False) -> None:
        self.root_dir = pathlib.Path(root_dir)
        self.run_tag = time.strftime("%Y_%m_%d-%H_%M_%S")
        self.video_fps = int(video_fps)
        self.save_generated = bool(save_generated)
        self.episode_idx = 0
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._reset_buffers()
        print(
            f"[retrieval-log] enabled — output: {self.root_dir / self.run_tag} "
            f"(save_generated={self.save_generated}, fps={self.video_fps})",
            flush=True,
        )

    def _reset_buffers(self) -> None:
        self._ret_head: list[np.ndarray] = []
        self._ret_left: list[np.ndarray] = []
        self._ret_right: list[np.ndarray] = []
        self._gen_head: list[np.ndarray] = []
        self._gen_left: list[np.ndarray] = []
        self._gen_right: list[np.ndarray] = []
        self._meta: list[dict] = []

    def log_step(self, ret_dict: dict, future_image_predictions: Optional[dict] = None) -> None:
        # Append the FULL chunk_size-frame retrieved window (one mp4 frame per
        # action timestep). Falls back to the stride-sampled 4-frame view if
        # the retrieval module didn't emit `*_full` (older code paths).
        head_full = ret_dict.get("retrieved_head_full", ret_dict["retrieved_head"])
        left_full = ret_dict.get("retrieved_left_wrist_full", ret_dict["retrieved_left_wrist"])
        right_full = ret_dict.get("retrieved_right_wrist_full", ret_dict["retrieved_right_wrist"])
        for f in head_full:
            self._ret_head.append(np.asarray(f, dtype=np.uint8))
        for f in left_full:
            self._ret_left.append(np.asarray(f, dtype=np.uint8))
        for f in right_full:
            self._ret_right.append(np.asarray(f, dtype=np.uint8))
        self._meta.append(dict(ret_dict.get("retrieval_meta", {})))

        if self.save_generated and future_image_predictions:
            head = _to_uint8_image(future_image_predictions.get("future_image"))
            left = _to_uint8_image(future_image_predictions.get("future_wrist_image"))
            right = _to_uint8_image(future_image_predictions.get("future_wrist_image2"))
            if head is not None:
                self._gen_head.append(head)
            if left is not None:
                self._gen_left.append(left)
            if right is not None:
                self._gen_right.append(right)

    def flush(self) -> None:
        if not self._meta:
            return
        ep_dir = self.root_dir / self.run_tag / f"episode_{self.episode_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        self._write_3panel(ep_dir / "retrieved.mp4", self._ret_head, self._ret_left, self._ret_right)
        if self.save_generated:
            self._write_3panel(ep_dir / "generated.mp4", self._gen_head, self._gen_left, self._gen_right)

        with open(ep_dir / "retrieval_meta.jsonl", "w") as f:
            for m in self._meta:
                f.write(json.dumps(m, default=str) + "\n")
        print(
            f"[retrieval-log] flushed episode {self.episode_idx:04d}: "
            f"{len(self._meta)} steps -> {ep_dir}",
            flush=True,
        )
        self.episode_idx += 1
        self._reset_buffers()

    def _write_3panel(
        self,
        out_path: pathlib.Path,
        head_frames: list[np.ndarray],
        left_frames: list[np.ndarray],
        right_frames: list[np.ndarray],
    ) -> None:
        n = min(len(head_frames), len(left_frames), len(right_frames))
        if n == 0:
            return
        try:
            import imageio
        except ImportError:
            print(f"[retrieval-log] imageio not installed — skipping {out_path.name}", flush=True)
            return

        # Resize all panels to a common height/width so concat works even if
        # ret/gen come in at different scales.
        h, w = head_frames[0].shape[:2]

        def _ensure(a: np.ndarray) -> np.ndarray:
            if a.shape[:2] == (h, w):
                return a
            from PIL import Image as _Image
            return np.array(_Image.fromarray(a).resize((w, h)))

        with imageio.get_writer(str(out_path), fps=self.video_fps) as writer:
            for i in range(n):
                row = np.concatenate(
                    [_ensure(head_frames[i]), _ensure(left_frames[i]), _ensure(right_frames[i])],
                    axis=1,
                )
                writer.append_data(row)
        print(f"[retrieval-log] saved {out_path} ({n} frames)", flush=True)


def _handle_act(
    cfg: PolicyEvalConfig,
    model,
    dataset_stats: dict,
    retrieval: RoboTwinDinoProprioRetrieval,
    req: dict,
    logger: Optional[_RetrievalLogger] = None,
    delta_stats: Optional[dict] = None,
) -> dict:
    obs = req["obs"]
    instruction = req["instruction"]
    seed = int(req.get("seed", cfg.seed) or 0)

    t0 = time.time()
    # Sim-step proprio history (oldest → most-recent) is optional. When the
    # client sends it (deploy_policy.py accumulates per env-step), retrieval
    # uses sim-rate K-window matching that aligns with the pool's sample
    # rate. Otherwise it falls back to chunk-spaced history (one entry per
    # act call) which is mostly meaningless for K-step alignment.
    query_proprio_history = obs.get("proprio_history")
    ret = retrieval.get_retrieved(
        query_head_image=obs["primary_image"],
        query_proprio=obs["proprio"],
        dataset_stats=dataset_stats,
        query_proprio_history=query_proprio_history,
    )
    obs.update({
        "retrieved_head": ret["retrieved_head"],
        "retrieved_left_wrist": ret["retrieved_left_wrist"],
        "retrieved_right_wrist": ret["retrieved_right_wrist"],
        "retrieved_proprio": ret["retrieved_proprio"],
        "retrieved_actions": ret["retrieved_actions"],
    })
    t_ret = time.time() - t0

    # In residual mode the model output is a normalized delta — disable the
    # default action unnormalize inside get_action and undo the delta + add
    # raw ret_actions ourselves below.
    orig_unnormalize_actions = cfg.unnormalize_actions
    if cfg.use_residual_actions:
        cfg.unnormalize_actions = False

    t1 = time.time()
    out = get_action(
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
    t_pol = time.time() - t1

    if cfg.use_residual_actions:
        cfg.unnormalize_actions = orig_unnormalize_actions
        if delta_stats is None:
            raise RuntimeError(
                "use_residual_actions=True but delta_stats was not loaded; pass --delta_stats_path."
            )
        # Stack model output (list of per-step actions) into (chunk, action_dim).
        norm_delta = np.stack([_to_np_action(a) for a in out["actions"]], axis=0)
        d_min = np.asarray(delta_stats["delta_actions_min"], dtype=np.float32)
        d_max = np.asarray(delta_stats["delta_actions_max"], dtype=np.float32)
        raw_delta = 0.5 * (norm_delta + 1.0) * (d_max - d_min) + d_min
        # Optional residual delta clipping. Default scale 1.0 = clip to the
        # training delta range (d_min/d_max from delta_stats). <1.0 tightens
        # to a fraction of training range — useful when the model
        # over-extrapolates at edge cases (e.g., placing) and produces
        # saturated outputs that overshoot when added to ret_acts_raw.
        # Set RESIDUAL_DELTA_CLIP_SCALE=0 to disable.
        try:
            _clip_scale = float(os.environ.get("RESIDUAL_DELTA_CLIP_SCALE", "1.0"))
        except ValueError:
            _clip_scale = 1.0
        if _clip_scale > 0:
            _lo = (_clip_scale * d_min).astype(np.float32)
            _hi = (_clip_scale * d_max).astype(np.float32)
            raw_delta_pre = raw_delta.copy()
            raw_delta = np.clip(raw_delta, _lo, _hi).astype(np.float32)
            _n_clipped = int(np.sum(raw_delta != raw_delta_pre))
            if _n_clipped > 0:
                log_message(f"raw_delta clipped: {_n_clipped} entries (scale={_clip_scale})")
        ret_acts_raw = ret["retrieved_actions"][: len(raw_delta)]
        raw_actions = (raw_delta + ret_acts_raw).astype(np.float32)
        out["actions"] = [raw_actions[i] for i in range(len(raw_actions))]
        # Per-call dump for residual diagnostics. Set RESIDUAL_DUMP_DIR=<dir>
        # to enable; one .npz per request with all 3 action streams. Lets us
        # trace whether overshoot comes from raw_delta (model artifact) vs
        # ret_acts_raw (chunk-boundary jump in pool) vs their sum.
        _dump_dir = os.environ.get("RESIDUAL_DUMP_DIR", "")
        if _dump_dir:
            os.makedirs(_dump_dir, exist_ok=True)
            _meta = ret["retrieval_meta"]
            _idx = len([f for f in os.listdir(_dump_dir) if f.endswith(".npz")])
            np.savez(
                os.path.join(_dump_dir, f"call_{_idx:05d}.npz"),
                norm_delta=norm_delta.astype(np.float32),
                raw_delta=raw_delta.astype(np.float32),
                ret_acts_raw=ret_acts_raw.astype(np.float32),
                raw_actions=raw_actions,
                pool_split=_meta.get("split", ""),
                pool_ep=int(_meta.get("ep_num", -1)),
                pool_frame=int(_meta.get("frame", -1)),
                pool_score=float(_meta.get("score", 0.0)),
            )

    log_message(
        f"retrieval={t_ret:.3f}s policy={t_pol:.3f}s  "
        f"match={ret['retrieval_meta']['split']}/ep{ret['retrieval_meta']['ep_num']}/"
        f"f{ret['retrieval_meta']['frame']} score={ret['retrieval_meta']['score']:.3f}"
    )

    actions = out["actions"]
    if isinstance(actions, list):
        actions = np.stack([_to_np_action(a) for a in actions], axis=0)
    else:
        actions = _to_np_action(actions)

    reply = {
        "ok": True,
        "actions": actions.astype(np.float32).tolist(),
        "retrieval_meta": ret["retrieval_meta"],
    }
    fips = out.get("future_image_predictions")
    if logger is not None:
        # Log on the raw (un-serialized) tensors before they're encoded for ZMQ.
        logger.log_step(ret, fips if isinstance(fips, dict) else None)
    serialized = _serialize_future_images(fips) if fips else {}
    if serialized:
        reply["future_image_predictions"] = serialized
    return reply


@draccus.wrap()
def serve_robotwin_ret(cfg: PolicyEvalConfig) -> None:
    if cfg.t5_text_embeddings_path:
        init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)

    print(f"[robotwin_ret_zmq_server] loading model from {cfg.ckpt_path}", flush=True)
    model, _cosmos_config = get_model(cfg)
    print("[robotwin_ret_zmq_server] model loaded.", flush=True)

    print(
        f"[robotwin_ret_zmq_server] building retrieval pool: task={cfg.pool_task} "
        f"splits={cfg.pool_splits}",
        flush=True,
    )
    retrieval = RoboTwinDinoProprioRetrieval(
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
    )
    print("[robotwin_ret_zmq_server] retrieval pool ready.", flush=True)

    delta_stats: Optional[dict] = None
    if cfg.use_residual_actions:
        if not cfg.delta_stats_path:
            raise RuntimeError(
                "use_residual_actions=True but --delta_stats_path was not given. "
                "Point it at delta_dataset_statistics.json (sibling of dataset_statistics.json)."
            )
        with open(cfg.delta_stats_path, "r") as f:
            js = json.load(f)
        delta_stats = {k: np.asarray(v, dtype=np.float32) for k, v in js.items()}
        print(
            f"[robotwin_ret_zmq_server] residual mode ON; loaded delta stats "
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

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(cfg.endpoint)
    print(f"[robotwin_ret_zmq_server] listening on {cfg.endpoint}", flush=True)

    try:
        while True:
            raw = sock.recv()
            try:
                req = pickle.loads(raw)
                op = req.get("op")
                if op == "ping":
                    reply = {"ok": True, "suite": cfg.suite, "action_dim": 16}
                elif op == "reset":
                    # Flush in-progress episode (if any) before clearing buffers.
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
        print("\n[robotwin_ret_zmq_server] shutting down.", flush=True)
    finally:
        # Catch any episode that ended without a follow-up reset (e.g., user
        # ctrl-C'd in the middle of an episode).
        if logger is not None:
            try:
                logger.flush()
            except Exception:
                traceback.print_exc()
        sock.close(0)
        ctx.term()


if __name__ == "__main__":
    serve_robotwin_ret()
