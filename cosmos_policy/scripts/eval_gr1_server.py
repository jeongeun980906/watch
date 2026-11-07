"""HTTP inference server for the cosmos_policy 2.5 GR1 multimodal conditioning-policy
(closed-loop online eval). Port of cosmos-framework's action_policy_server_gr1.py.

Speaks the SAME HTTP schema (/reset /push /predict) as the Cosmos3 server, so the SAME
client (cosmos_framework/scripts/eval_gr1_client.py) plugs in unchanged — just point its
--server here. The env↔state57 and action57(delta)→env-action geometry, and the online
retrieval (t=0 taskdemo → t>0 play), are copied verbatim; ONLY the model block is the 2.5
multimodal path: it builds the data_batch the training dataset emits, runs the 2.5 RF
model, and extracts the predicted action57 chunk (min-max de-normalized, then integrated
to env actions). Three layout modes, auto-detected from --experiment:
  * "gr1_ret_v2" (DEFAULT)  -> _build_sample_v2: mid-v2 length-general compact layout
    (state_t=20, k=g=8, slot_roles + t_positions, action slot 11) mirroring
    EgoVerseGr1RetV2Dataset — use for the v2 post-train checkpoints.
  * "aligned"               -> _build_sample_aligned (state_t=50, OBSOLETE).
  * otherwise               -> _build_sample (36-latent EgoVerseGr1RetDataset layout).
("gr1_ret_v3*" experiments, e.g. the 448 post-train, use the SAME v2 grid layout —
pass --resolution 448 and the S//2 tiling makes each quadrant a native 224 decode.)

STATE-CHUNK mode (experiments containing "_state", auto-detected): the model predicts
ABSOLUTE state57 chunks s[t+1..t+32] instead of action57 deltas. The v2 sample builder
then (a) fills slot 0 with the PAST state chunk s[t-31..t] built from the retriever's
proprio buffer (left-padded with the FIRST buffered frame — dataset convention s[0]),
(b) fills the ret slot with the demo's FUTURE state57 chunk s[w0+1..w0+32], and the
OUTPUT path converts the predicted absolute states DIRECTLY to env actions (pos +
6D->R->axis-angle from the abs9 blocks + _CTRL_OFF offsets; grips via hand15
inverse-lerp -> cmd=2g-1) — NO delta integration. Normalization mirrors the state
dataset exactly (resolve_state_stats: ROBUST mean+/-4std stats). The delta path is
untouched for non-state experiments.

EXPLICIT-DEMO mode (L2/L3 one-shot eval, docs/gr1_data_eval_redesign.md §2): /reset
accepts {"oneshot_demo": {"ep": <egoverse zarr episode>, "w0": int}} which PINS that
demo for every chunk of the episode (taskdemo + play retrieval bypassed); chunk c uses
w0 + c*chunk_size (clamped) so the demo plays forward in sync with the robot.

Launch (cosmos_policy uv env, GPU):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
      uv run --extra cu128 --group egoverse --python 3.10 \\
      python -m cosmos_policy.scripts.eval_gr1_server \\
        --checkpoint <2.5 GR1 DCP ckpt dir> --port 8000 \\
        --pool /mnt/ddn/dataset/egoverse_pairs_v2/taskdemo_pool_v4.npz --pool-kind taskdemo \\
        --play-pool /mnt/ddn/dataset/egoverse_pairs_v2/dth_pool_1x_v4.npz   # 20-d v4 pools (defaults)
Client (mimicdroid-robocasa venv, unchanged):
    MUJOCO_GL=egl python cosmos_framework/scripts/eval_gr1_client.py --server http://127.0.0.1:8000 --task PnPSinkToCabinet
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.spatial.transform import Rotation as _R

from cosmos_policy._src.imaginaire.config import load_config
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init, model_init
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent
from cosmos_policy.datasets.dataset_utils import preprocess_image
from cosmos_policy.datasets.egoverse_zread import CENTERTOP, EgoZarrEpisode, FRONT, SIDELEFT, SIDERIGHT
from cosmos_policy.scripts.gr1_online_retrieval import OnlineRetriever
from cosmos_policy.scripts import gr1_online_retrieval as _retrieval
from cosmos_policy.utils.utils import duplicate_array
from cosmos_policy.datasets.egoverse_gr1_ret_aligned_dataset import (
    DOWNSAMPLE_IDX as ALIGNED_DOWN,
    NUM_FRAMES as ALIGNED_NF,        # 92
    R as ALIGNED_R,                  # 23
    UPSAMPLE_IDX as ALIGNED_UP,
)
# v2 (mid-v2 length-general) layout constants — mirror EgoVerseGr1RetV2Dataset EXACTLY
from cosmos_policy.datasets.egoverse_gr1_ret_v2_dataset import (
    DT as V2_DT,                     # 0.2 s per latent
    K as V2_K,                       # 8 latents per 32f video block
)
from cosmos_policy.datasets.egoverse_gr1_ret_v2_state_dataset import resolve_state_stats
from cosmos_policy.datasets.egoverse_ret_v2_dataset import (
    NOW_S,
    ROLE_ACTION,
    ROLE_CUR_IMAGE,
    ROLE_CUR_STATE,
    ROLE_FUTURE,
    ROLE_RET_ACTION,
    ROLE_RET_VIDEO,
)

HUM_ROOT = "/mnt/ddn/dataset/egoverse"
DEMO_EMB = "mimicdroid"
CHUNK = 32
NUM_DUP = 4
CAMS = (FRONT, SIDELEFT, SIDERIGHT)
CAMS4 = (FRONT, SIDELEFT, SIDERIGHT, CENTERTOP)   # --grid4 (2026-07-16): 4-view grid, BR quadrant = center_top

# ---------------------------------------------------------------------------
# Geometry / convention helpers — COPIED VERBATIM from the Cosmos3 server
# (env↔state57 and action57-delta↔env-action). The 2.5 model outputs the SAME
# action57 DELTA space (non-residual), so these are reused unchanged.
# ---------------------------------------------------------------------------
_HEAD9 = np.array([-0.60, 0.0, 1.17, 0.033, -0.999, -0.003, 0.818, 0.028, -0.574], float)
_OPEN15 = np.array([0.0005, 0.0958, -0.034, 0.0285, 0.0222, -0.1263, 0.0085, 0.0217, -0.1285,
                    -0.0116, 0.0222, -0.1282, -0.0314, 0.0229, -0.1259], float)
_CLOSED15 = np.array([0.0052, 0.0775, -0.0701, 0.0266, 0.044, -0.0916, 0.0085, 0.0441, -0.1016,
                      -0.0097, 0.044, -0.0937, -0.0278, 0.044, -0.0919], float)


def _abs9_one(pos, quat_xyzw) -> np.ndarray:
    q = np.asarray(quat_xyzw, float)
    n = np.linalg.norm(q)
    q = q / n if (n > 1e-6 and np.isfinite(q).all()) else np.array([0.0, 0.0, 0.0, 1.0])
    Rm = _R.from_quat(q).as_matrix()
    return np.concatenate([np.asarray(pos, float), Rm[:, 0], Rm[:, 1]])


def _state57_from_proprio(p: dict) -> np.ndarray:
    """{pL,pR,quatL,quatR,gripL,gripR} (current env step) -> absolute state57."""
    def hand15(g):
        g = float(np.clip(g, 0.0, 1.0))
        return (1.0 - g) * _OPEN15 + g * _CLOSED15
    return np.concatenate([
        _HEAD9,
        _abs9_one(p["pL"], p["quatL"]), hand15(p["gripL"]),
        _abs9_one(p["pR"], p["quatR"]), hand15(p["gripR"]),
    ]).astype(np.float32)


def _quat_mat(quat_xyzw) -> np.ndarray:
    q = np.asarray(quat_xyzw, float)
    n = np.linalg.norm(q)
    q = q / n if (n > 1e-6 and np.isfinite(q).all()) else np.array([0.0, 0.0, 0.0, 1.0])
    return _R.from_quat(q).as_matrix()


def _sixd_to_R(c0, c1) -> np.ndarray:
    a = c0 / (np.linalg.norm(c0) + 1e-9)
    b = c1 - (a @ c1) * a
    b = b / (np.linalg.norm(b) + 1e-9)
    return np.stack([a, b, np.cross(a, b)], axis=1)


def _grip_from_hand15(h15: np.ndarray) -> float:
    axis = _CLOSED15 - _OPEN15
    return float(np.clip(((np.asarray(h15, float) - _OPEN15) @ axis) / (axis @ axis), 0.0, 1.0))


def _integrate_arm(p0, R0, eef9) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p0, float).copy()
    Rm = np.asarray(R0, float).copy()
    pos, aa = [], []
    for d in np.asarray(eef9, float):
        p = p + Rm @ d[:3]
        Rm = Rm @ _sixd_to_R(d[3:6], d[6:9])
        pos.append(p.copy())
        aa.append(_R.from_matrix(Rm).as_rotvec())
    return np.asarray(pos), np.asarray(aa)


_CTRL_OFF = {
    "R": (np.array([0.001236, 0.002752, -0.999984, -0.004839]), np.array([0.00104, 0.01354, 0.00726])),
    "L": (np.array([0.707104, 0.004116, 0.707094, 0.002261]), np.array([0.00012, -0.01272, 0.00862])),
}


def _apply_ctrl_offset(pos, aa, arm: str) -> tuple[np.ndarray, np.ndarray]:
    roff_q, poff = _CTRL_OFF[arm]
    Roff = _R.from_quat(roff_q).as_matrix()
    out_p, out_a = [], []
    for p, a in zip(np.asarray(pos), np.asarray(aa)):
        Rb = _R.from_rotvec(a).as_matrix()
        out_p.append(p + Rb @ poff)
        out_a.append(_R.from_matrix(Rb @ Roff).as_rotvec())
    return np.asarray(out_p), np.asarray(out_a)


def _decode_image(img, size: int) -> np.ndarray:
    """{list HxWx3 uint8 | base64 png} -> (size,size,3) uint8."""
    if isinstance(img, str):
        raw = base64.b64decode(img.split(",")[-1])
        a = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    else:
        a = np.asarray(img, dtype=np.uint8)
    if a.shape[:2] != (size, size):
        a = np.asarray(Image.fromarray(a).resize((size, size), Image.BILINEAR), np.uint8)
    return a


# ---------------------------------------------------------------------------
# Model block — the ONLY part rewritten for the cosmos_policy 2.5 model.
# ---------------------------------------------------------------------------
class GR1Policy:
    def __init__(self, args) -> None:
        # Proven inference load (mirrors experiments/robot get_model): load WITHOUT FSDP-wrapping so
        # the consolidated flat .pt goes in cleanly — avoids the DCP checkpointer's training-state
        # resume (optim/scheduler=None → crash) AND the FSDP-wrap-then-load state_dict mismatch.
        # --checkpoint = a .pt produced by convert_dcp_to_pt.py (NOT the raw DCP iter dir).
        with distributed_init():
            distributed.init()
        self.model, config = load_model_from_checkpoint(
            experiment_name=args.experiment, s3_checkpoint_dir=args.checkpoint,
            config_file=args.config, enable_fsdp=False, load_ema_to_reg=False,
            experiment_opts=[f"dataloader_train.dataset.manifest_dir={args.pairs_dir}"])
        self.model = self.model.cuda().eval()
        self.tk = self.model.tensor_kwargs
        self.config = config
        self.chunk_size = int(getattr(config.model.config, "chunk_size", CHUNK))
        self.S = int(args.resolution)

        # V2 layout mode (mid-v2 length-general: compact state_t=20, k=g=8 native
        # 32f@20Hz, slot roles + continuous-time rope, action slot 11 — see
        # egoverse_gr1_ret_v2_dataset.py). DEFAULT for the v2 post-train checkpoints.
        # gr1_ret_v3* (e.g. _v3_448) keeps the SAME grid layout — only final_image_size
        # differs, which self.S / _grid3(S//2) already parameterize end-to-end.
        # v2 compact role/timestamp layout: EVERY gr1_ret_v2+ experiment (v2/v3/v5/v6...)
        # uses it — substring test on v2/v3 alone silently fell back to the V1 layout for
        # v5/v6 (found 2026-07-14 in the v6 one-shot audit: banner showed v2=False).
        import re as _re2
        self.v2 = bool(_re2.search(r"gr1_ret_v[2-9]", args.experiment))
        # STATE-CHUNK mode (see module doc): predicted chunk = ABSOLUTE state57,
        # slot 0 = past state chunk, ret slot = demo FUTURE state chunk.
        self.state_mode = "_state" in args.experiment
        # MID-ALIGNED layout mode (state_t=50, action slot 49, grid-3cam, 92f upsample —
        # see egoverse_gr1_ret_aligned_dataset.py). The model generates a (92,57) chunk;
        # predict() downsamples it back to the native 32 steps via ALIGNED_DOWN. OBSOLETE.
        self.aligned = (not self.v2) and ("aligned" in args.experiment)
        # "pad": native 32f + zero-pad (default); "upsample": 32f repeated to 92f
        self.aligned_layout = "upsample" if "_aligned_up_" in args.experiment else "pad"
        # real per-class t5 embeddings (training conditions on these; zeros only as fallback)
        self._t5 = {}
        _t5p = os.path.join(args.pairs_dir, "t5_embeddings.pkl")
        if os.path.exists(_t5p):
            import pickle as _pkl
            with open(_t5p, "rb") as _fp:
                self._t5 = _pkl.load(_fp)

        # PER-EMBODIMENT min-max stats. GR1 online: query = robot "gr1", retrieved demo = "mimicdroid".
        def _emb_lohi(js, emb):
            d = json.load(open(js))
            e = d if ("lo" in d and "hi" in d) else d.get(emb, d.get("mimicdroid", d.get("_global")))
            return np.asarray(e["lo"], np.float32), np.asarray(e["hi"], np.float32)
        _af = os.path.join(args.pairs_dir, "action57_stats.json")
        _sf = os.path.join(args.pairs_dir, "state57_stats.json")
        if not os.path.exists(_sf):        # v6 manifest ships only the pctl file
            _sf = os.path.join(args.pairs_dir, "state57_stats_pctl.json")
        self._lo, self._hi = _emb_lohi(_af, "gr1")            # QUERY action space (pred-action de-norm)
        self._mlo, self._mhi = _emb_lohi(_af, "mimicdroid")   # retrieved-DEMO action space
        self._slo, self._shi = _emb_lohi(_sf, "gr1")          # QUERY state space (cur_state)
        self._mslo, self._mshi = _emb_lohi(_sf, "mimicdroid") # retrieved-DEMO state space
        if self.state_mode:
            # STATE mode: normalization must mirror the TRAINING dataset choice.
            # v6 config trains with the EXPLICIT percentile-hybrid file
            # (state_stats_path=state57_stats_pctl.json, robust_state_stats=False)
            # -> prefer that file when the manifest ships it; older manifests fall
            # back to resolve_state_stats (ROBUST mean+/-4std, v5 convention).
            _pctl = os.path.join(args.pairs_dir, "state57_stats_pctl.json")
            if os.path.exists(_pctl):
                from cosmos_policy.datasets.egoverse_ret_v3_state_dataset import load_stats_json
                _sp, _sstats = _pctl, load_stats_json(_pctl)
            else:
                _sp, _sstats = resolve_state_stats(args.pairs_dir)
            def _pick(emb):
                e = _sstats.get(emb, _sstats["_global"])
                return e["lo"], e["hi"]
            self._slo, self._shi = _pick("gr1")
            self._mslo, self._mshi = _pick("mimicdroid")
            print(f"[GR1Policy-2.5] STATE mode: state stats <- {_sp}", flush=True)
        self.pred_target = getattr(args, "pred_target", "state57")
        self.pin_lock_ep = bool(getattr(args, "pin_lock_ep", False))
        self.mono_floor = int(getattr(args, "mono_floor", 8))
        # robustness eval: fixed-sigma Gaussian noise on the normalized ret_action
        # conditioning (see _build_sample_v2). Mirrors dataset ret_action_noise_std.
        self._ret_action_noise = float(getattr(args, "ret_action_noise", 0.0))
        # robustness eval: Gaussian blur on the RETRIEVED-DEMO video only (the
        # reference the ICL arm is supposed to read). sigma is in pixels at the
        # per-camera tile scale (S//2), applied per tile BEFORE gridding so the
        # blur never bleeds across camera boundaries. Query/current obs untouched.
        self._ret_video_blur = float(getattr(args, "ret_video_blur", 0.0))
        # generated-video capture: when set, decode the predicted FUTURE-video latents
        # each inference and save frames to this dir (chunkNNNN_fKK.png), for the
        # rollout|generated|retrieval viewer. Off by default (adds a VAE decode/step).
        self._gen_video_dir = getattr(args, "gen_video_dir", None)
        self._gen_ep = -1              # per-episode gen subdir index (bumped in reset())
        self._gen_ep_dir = None
        self.grid4 = bool(getattr(args, "grid4", False))
        self.plate_state_delta = bool(getattr(args, "plate_state_delta", False))
        # SHORT (no-video) layout mode: post-trains from the short mids strip the
        # demo-video and future-video blocks (state_t 20 -> 4/3). Auto-detected from
        # the experiment name (..._short_state / ..._short_cdh); with_proprio mirrors
        # the training datasets (state variant keeps the cur_state slot, delta drops it).
        self.short = "_short" in args.experiment
        # PLAIN (non-retrieval BC) layout mode: the post-train arm that drops the demo
        # entirely (state_t 20 -> 11; see egoverse_gr1_plain_dataset.py). Experiment
        # names are "..._gr1_plain_*", which does NOT match the gr1_ret_v[2-9] regex
        # above, so self.v2 is False for them — set it here, since the plain layout IS
        # a v2-family (role/continuous-time) layout and needs the S//2 grid decode path.
        self.plain = "_plain" in args.experiment
        if self.plain:
            self.v2 = True
        self._mono_floor_default = self.mono_floor
        if self.pred_target == "abs_action57":
            _tf = os.path.join(args.pairs_dir, "abs_action57_stats_pctl.json")
            self._tlo, self._thi = _emb_lohi(_tf, "gr1")
            print(f"[GR1Policy-2.5] ABS-target mode: pred de-norm <- {_tf}", flush=True)
        if self.pred_target == "cmd_delta57":
            _cf = os.path.join(args.pairs_dir, "cmd_delta57_stats.json")
            self._cdlo, self._cdhi = _emb_lohi(_cf, "gr1")           # pred de-norm (query cmd-delta)
            self._mcdlo, self._mcdhi = _emb_lohi(_cf, "mimicdroid")  # demo ret_action cond
            self._cmd_anchor = None                                  # (L9,R9) command-space anchor
            print(f"[GR1Policy-2.5] CMD-DELTA mode: stats <- {_cf} (command-anchor integration)", flush=True)

        self.retriever = OnlineRetriever(args.pool, kind=args.pool_kind)
        self.play_retriever = (
            OnlineRetriever(args.play_pool, kind=getattr(args, "play_pool_kind", "dth"))
            if getattr(args, "play_pool", None) else None
        )
        self._t0_taskdemo = args.pool_kind == "taskdemo"
        self._chunk = 0
        self._pinned = None            # (ep, w0) explicit one-shot demo; set via /reset
        self._zcache: dict[str, EgoZarrEpisode] = {}
        self.pinned_advance = getattr(args, "pinned_advance", "match")
        self.num_steps = args.num_steps
        self.guidance = args.guidance
        # match the TRAINING flow shift unless explicitly overridden (model.config.shift=5;
        # the old hardcoded 3.0 made every online eval sample off the trained schedule)
        self.shift = args.shift if args.shift is not None else float(
            getattr(self.model.config, "shift", 3.0))
        self.seed = args.seed
        print(f"[GR1Policy-2.5] chunk={self.chunk_size} S={self.S} v2={self.v2} aligned={self.aligned} "
              f"state_mode={self.state_mode} short={self.short} t0_taskdemo={self._t0_taskdemo} "
              f"play_pool={self.play_retriever is not None} ret_action_noise={self._ret_action_noise} "
              f"ret_video_blur={self._ret_video_blur}", flush=True)

    def _grp(self, ep: str) -> EgoZarrEpisode:
        g = self._zcache.get(ep)
        if g is None:
            g = EgoZarrEpisode(os.path.join(HUM_ROOT, ep))
            self._zcache[ep] = g
        return g

    def _norm(self, a, lo, hi):
        return np.clip(2.0 * (a - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0).astype(np.float32)

    def reset(self, req: dict | None = None) -> None:
        self._chunk = 0
        # Always-on episode counter. _gen_ep only advances when --gen-video-dir is set,
        # so the attention dump needs its own or every episode would overwrite ep0.
        self._ep_seq = getattr(self, "_ep_seq", -1) + 1
        if self._gen_video_dir:        # new per-episode gen-frame subdir (ep00000, ep00001, ...)
            self._gen_ep += 1
            self._gen_ep_dir = os.path.join(self._gen_video_dir, f"ep{self._gen_ep:05d}")
        self.retriever.reset()
        if self.play_retriever is not None:
            self.play_retriever.reset()
        # per-episode mono-floor override (task rule 2026-07-16: TurnOnFaucetL3 runs
        # floor 16, everything else the server default) — client sends it in /reset.
        self.mono_floor = int((req or {}).get("mono_floor", self._mono_floor_default))
        # EXPLICIT-DEMO mode (L2/L3 one-shot protocol, gr1_data_eval_redesign.md §2):
        # /reset {"oneshot_demo": {"ep": <egoverse zarr episode>, "w0": int}} PINS that
        # demo for ALL chunks of the episode — both taskdemo and play retrieval are
        # bypassed in predict() until the next /reset.
        self._pinned = None
        self._pinned_windows = None
        self._pinned_last_w0 = 0
        self._pinned_lock_ep = None
        self._cmd_anchor = None
        self._cmd_anchor = None
        od = (req or {}).get("oneshot_demo")
        # v6: {"oneshot_task": "<task[변형]>"} pins ALL 3 taskdemos of that task —
        # within-episode matching runs over their COMBINED windows (train-매칭과 동일
        # 26-d space); covers seen/L2/L3 tiers by task-variant name.
        ot = (req or {}).get("oneshot_task")
        if ot and not od:
            eps3 = [f"taskdemo_{ot}_{ot}_D_demo_{i}" for i in (1, 2, 3)]
            eps3 = [e for e in eps3 if os.path.isdir(os.path.join(HUM_ROOT, e))]
            if not eps3:
                raise FileNotFoundError(f"oneshot_task has no taskdemos: {ot}")
            od = {"ep": eps3[0], "w0": 0, "_all": eps3}
        if od:
            ep, w0 = str(od["ep"]), int(od.get("w0", 0))
            if not os.path.isdir(os.path.join(HUM_ROOT, ep)):
                raise FileNotFoundError(f"oneshot_demo episode not found: {HUM_ROOT}/{ep}")
            self._pinned = (ep, w0)
            # WITHIN-EPISODE matching (not blind lockstep): precompute the pinned
            # episode's past-window descriptors; chunk 0 conditions on the opening
            # (w0 as sent), chunks >=1 re-match the robot's strict past against these
            # windows so the demo stays phase-aligned even if the robot runs
            # slower/faster than the demo.
            _eps = od.get("_all", [ep]) if isinstance(od, dict) else [ep]
            _w0s, _Ts, _eids = [], [], []
            for _e in _eps:
                st = self._grp(_e).state("state57").astype(np.float64)
                w0s_e, T_e = _retrieval.episode_windows(st)
                _w0s.append(np.asarray(w0s_e)); _Ts.append(np.asarray(T_e))
                _eids += [_e] * len(w0s_e)
            self._pinned_windows = (np.concatenate(_w0s), np.concatenate(_Ts))
            self._pinned_eps = np.array(_eids)
            print(f"[GR1Policy-2.5] one-shot pinned: {len(_eps)} demo(s) "
                  f"({len(self._pinned_eps)} windows for within-episode match)", flush=True)

    def push(self, req: dict) -> dict:
        props = req.get("proprios") or [req["proprio"]]
        for p in props:
            self.retriever.push(p["pL"], p["pR"], p["quatL"], p["quatR"], p["gripL"], p["gripR"])
            if self.play_retriever is not None:
                self.play_retriever.push(p["pL"], p["pR"], p["quatL"], p["quatR"], p["gripL"], p["gripR"])
        return {"ok": True}

    # --- build the 36-latent EgoVerseGr1RetDataset-format sample for inference ---
    def _build_sample(self, imgs3: list[np.ndarray], proprio: dict, demo_ep, demo_w0, has_ret: int) -> dict:
        S, C = self.S, self.chunk_size
        zero = np.zeros((S, S, 3), np.uint8)
        image_list: list[np.ndarray] = []
        cur = 0

        def push_image(arr):
            nonlocal cur
            image_list.append(duplicate_array(arr, total_num_copies=NUM_DUP))
            i = cur; cur += 1; return i

        def push_blank():
            return push_image(zero)

        def push_video(frames):
            nonlocal cur
            image_list.append(frames)
            start = cur; cur += frames.shape[0] // NUM_DUP
            return start, cur

        # demo clip reader (3-cam, C frames)
        gm = self._grp(demo_ep) if has_ret else None

        def demo_clip(cam):
            if not has_ret:
                return np.broadcast_to(zero, (C, S, S, 3)).copy()
            return np.stack([_decode_demo(gm, demo_w0 + k, cam, S) for k in range(C)])

        # latent 0: blank sentinel
        image_list.append(np.expand_dims(zero, 0)); cur += 1
        # ret videos (3 cams) — CONDITION
        ret_spans = {}
        for name, cam in (("front", FRONT), ("sideleft", SIDELEFT), ("sideright", SIDERIGHT)):
            ret_spans[name] = push_video(demo_clip(cam))
        retrieved_video_start = ret_spans["front"][0]
        retrieved_video_end = ret_spans["sideright"][1]
        # ret state / ret action (injected) — CONDITION
        retrieved_state_latent_idx = push_blank()
        ret_action_latent_idx = push_blank()
        # current images (3 cams, single obs frame) — CONDITION
        current_image_latent_idx = push_image(imgs3[0])
        current_image2_latent_idx = push_image(imgs3[1])
        current_wrist_image_latent_idx = push_image(imgs3[2])
        current_proprio_latent_idx = push_blank()
        # pred action + future images/state — GENERATED (blank; model fills via denoising)
        action_latent_idx = push_blank()
        future_image_latent_idx = push_blank()
        future_image2_latent_idx = push_blank()
        future_wrist_image_latent_idx = push_blank()
        future_proprio_latent_idx = push_blank()

        images = np.concatenate(image_list, axis=0)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)

        # ---- injected values (normalized) ----
        cur_state = self._norm(_state57_from_proprio(proprio)[None], self._slo, self._shi)[0]
        if has_ret:
            m_state = gm.state("state57"); m_act = gm.action("action57")
            ret_state = self._norm(np.asarray(m_state[min(max(demo_w0, 0), len(m_state) - 1)], np.float32)[None],
                                   self._mslo, self._mshi)[0]           # DEMO (mimicdroid) state space
            ret_act = np.stack([m_act[min(max(demo_w0 + k, 0), len(m_act) - 1)] for k in range(C)]).astype(np.float32)
            ret_act = self._norm(ret_act, self._mlo, self._mhi)         # DEMO (mimicdroid) action space
        else:
            ret_state = np.zeros(57, np.float32)
            ret_act = np.zeros((C, 57), np.float32)

        return {
            "video": images,
            "command": "gr1",
            "actions": np.zeros((C, 57), np.float32),   # placeholder (generated at inference)
            "t5_text_embeddings": torch.zeros(512, 1024),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "proprio": cur_state.astype(np.float32),
            "future_proprio": np.zeros(57, np.float32),
            "__key__": 0,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 1, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "action_latent_idx": action_latent_idx,
            "value_latent_idx": -1,
            "current_proprio_latent_idx": current_proprio_latent_idx,
            "current_image_latent_idx": current_image_latent_idx,
            "current_image2_latent_idx": current_image2_latent_idx,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx,
            "current_wrist_image2_latent_idx": -1,
            "future_proprio_latent_idx": future_proprio_latent_idx,
            "future_image_latent_idx": future_image_latent_idx,
            "future_image2_latent_idx": future_image2_latent_idx,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx,
            "future_wrist_image2_latent_idx": -1,
            "retrieved_video_start_latent_idx": retrieved_video_start,
            "retrieved_video_end_latent_idx": retrieved_video_end,
            "ret_video_front_start_latent_idx": ret_spans["front"][0],
            "ret_video_front_end_latent_idx": ret_spans["front"][1],
            "ret_video_sideleft_start_latent_idx": ret_spans["sideleft"][0],
            "ret_video_sideleft_end_latent_idx": ret_spans["sideleft"][1],
            "ret_video_sideright_start_latent_idx": ret_spans["sideright"][0],
            "ret_video_sideright_end_latent_idx": ret_spans["sideright"][1],
            "retrieved_action_latent_idx": ret_action_latent_idx,
            "retrieved_state_latent_idx": retrieved_state_latent_idx,
            "retrieved_actions": ret_act.astype(np.float32),
            "retrieved_proprio": ret_state.astype(np.float32),
            "ret_proprio": ret_state.astype(np.float32),
            "has_ret_data": has_ret, "has_ret_image": has_ret, "ret_mask": has_ret,
        }

    # --- MID-ALIGNED sample: exact egoverse_ret layout (see egoverse_gr1_ret_aligned_dataset) ---
    def _grid3(self, imgs3: list[np.ndarray]) -> np.ndarray:
        """cams -> one SxS 2x2 grid (front TL, sideleft TR, sideright BL, center_top BR
        when 4 images are given / --grid4; black BR with 3)."""
        S = self.S; h = S // 2
        canvas = np.zeros((S, S, 3), np.uint8)
        for (r0, c0), im in zip(((0, 0), (0, h), (h, 0), (h, h)), imgs3):
            if im.shape[:2] != (h, h):
                im = np.asarray(Image.fromarray(im).resize((h, h), Image.BILINEAR), np.uint8)
            canvas[r0:r0 + h, c0:c0 + h] = im
        return canvas

    def _demo_tile(self, gm: EgoZarrEpisode, i: int, cam: str, size: int) -> np.ndarray:
        """One retrieved-demo camera tile, optionally Gaussian-blurred (--ret-video-blur).

        Blur is applied per tile at the native tile scale, so a given sigma means the
        same thing whether the grid is 3-cam or 4-cam and whatever --resolution is.
        Only the DEMO pixels are touched -- the current observation goes through
        _grid3() untouched, so the arm keeps a sharp view of its own scene.
        """
        t = _decode_demo(gm, i, cam, size)
        if self._ret_video_blur <= 0.0:
            return t
        return np.asarray(
            Image.fromarray(t).filter(ImageFilter.GaussianBlur(radius=self._ret_video_blur)),
            np.uint8)

    def _grid_demo_clip92(self, gm: EgoZarrEpisode, w0: int) -> np.ndarray:
        """(92,S,S,3) grid clip of the retrieved demo. pad: native 32f + black; upsample: repeat."""
        h = self.S // 2
        native = {}
        for k in range(CHUNK):
            i = min(max(w0 + int(k), 0), gm.n_frames - 1)
            native[int(k)] = self._grid3([self._demo_tile(gm, i, cam, h) for cam in (CAMS4 if self.grid4 else CAMS)])
        if self.aligned_layout == "pad":
            clip = np.zeros((ALIGNED_NF, self.S, self.S, 3), np.uint8)
            for k in range(CHUNK):
                clip[k] = native[k]
            return clip
        return np.stack([native[int(k)] for k in ALIGNED_UP])

    def _span92(self, a32: np.ndarray) -> np.ndarray:
        if self.aligned_layout == "pad":
            out = np.zeros((ALIGNED_NF, a32.shape[1]), np.float32)
            out[:CHUNK] = a32
            return out
        return a32[ALIGNED_UP].astype(np.float32)

    def _build_sample_aligned(self, imgs3: list[np.ndarray], proprio: dict,
                              demo_ep, demo_w0, has_ret: int, task: str) -> dict:
        S = self.S
        zero = np.zeros((S, S, 3), np.uint8)
        gm = self._grp(demo_ep) if has_ret else None
        m_clip = self._grid_demo_clip92(gm, demo_w0) if has_ret \
            else np.zeros((ALIGNED_NF, S, S, 3), np.uint8)
        cur_grid = self._grid3(imgs3)

        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)
        # frame packing: blank | ret92 | ret_act blank4 | cur4 | future zeros 92 (GEN) | act blank4
        images = np.concatenate([
            blank1, m_clip, blank4,
            np.repeat(cur_grid[None], NUM_DUP, axis=0),
            np.zeros((ALIGNED_NF, S, S, 3), np.uint8), blank4,
        ], axis=0)                                                # (197,S,S,3)
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)

        # retrieved action: native 32 -> normalized (DEMO space) -> 92 span (pad/upsample)
        if has_ret:
            m_act = gm.action("action57")
            ret32 = np.stack([m_act[min(max(demo_w0 + k, 0), len(m_act) - 1)]
                              for k in range(CHUNK)]).astype(np.float32)
            ret_act = self._span92(self._norm(ret32, self._mlo, self._mhi))
        else:
            ret_act = np.zeros((ALIGNED_NF, 57), np.float32)

        # current state: normalized state57 (QUERY/gr1 space), tiled into slot 0
        cur_state = self._norm(_state57_from_proprio(proprio)[None], self._slo, self._shi)[0]

        # t5: same key convention as training (raw class); zeros fallback
        cls = _base_task(task)
        t5 = (torch.squeeze(torch.as_tensor(self._t5[cls]))
              if cls in self._t5 else torch.zeros(512, 1024))

        return {
            "video": images,
            "command": cls,
            "actions": np.zeros((ALIGNED_NF, 57), np.float32),    # placeholder (generated)
            "action_mask": np.ones(57, np.float32),
            "retrieved_actions": ret_act.astype(np.float32),
            "retrieved_action_mask": np.ones(57, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "embodiment": "gr1", "retrieved_embodiment": DEMO_EMB,
            "num_conditional_frames": 26,                         # prefix 0..25 (mid value)
            # latent indices — IDENTICAL to mid-training/aligned dataset
            "retrieved_video_start_latent_idx": 1,
            "retrieved_video_end_latent_idx": 1 + ALIGNED_R,      # 24
            "retrieved_action_latent_idx": 1 + ALIGNED_R,         # 24
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 2 + ALIGNED_R,            # 25
            "future_video_start_latent_idx": 3 + ALIGNED_R,       # 26
            "future_video_end_latent_idx": 3 + 2 * ALIGNED_R,     # 49 (exclusive)
            "action_latent_idx": 3 + 2 * ALIGNED_R,               # 49
            "proprio": cur_state.astype(np.float32),              # slot-0 injection
            "future_proprio": np.zeros(57, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": 3 + ALIGNED_R, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": has_ret, "has_ret_image": has_ret, "has_current_image": 1,
            # POLICY sample: action latent must be GENERATED (world_model_sample_mask=0 —
            # 1 would condition on the zero placeholder actions and return garbage).
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
        }

    # --- STATE mode helpers ---
    def _past_state_chunk(self, proprio: dict) -> np.ndarray:
        """(CHUNK,57) PAST absolute state57 chunk s[t-31 .. t] from the retriever's
        proprio buffer (OnlineRetriever buffers WIN+1 frames; the NEWEST buffered
        frame is the current boundary state t). Fewer than CHUNK buffered frames ->
        LEFT-PAD by repeating the FIRST buffered frame (dataset convention: the
        episode's first frame s[0])."""
        buf = list(self.retriever._buf)
        if not buf:
            buf = [(np.asarray(proprio["pL"], float), np.asarray(proprio["pR"], float),
                    np.asarray(proprio["quatL"], float), np.asarray(proprio["quatR"], float),
                    float(proprio["gripL"]), float(proprio["gripR"]))]
        rows = buf[-CHUNK:]
        rows = [rows[0]] * (CHUNK - len(rows)) + rows
        return np.stack([
            _state57_from_proprio({"pL": b[0], "pR": b[1], "quatL": b[2], "quatR": b[3],
                                   "gripL": b[4], "gripR": b[5]})
            for b in rows
        ])

    def _abs_to_env_action(self, ab: np.ndarray) -> np.ndarray:
        """ABS COMMAND chunk (C,57; abs_action57, SITE frame) -> env actions DIRECTLY.
        No _CTRL_OFF: these ARE the recorded env-action targets (v6, row t = act[t])."""
        ab = np.asarray(ab, float)

        def _arm(block):
            pos = block[:, :3]
            aa = np.stack([_R.from_matrix(_sixd_to_R(b[3:6], b[6:9])).as_rotvec() for b in block])
            return pos, aa

        Lpos, Laa = _arm(ab[:, 9:18])
        Rpos, Raa = _arm(ab[:, 33:42])
        gL = np.array([_grip_from_hand15(h) for h in ab[:, 18:33]])
        gR = np.array([_grip_from_hand15(h) for h in ab[:, 42:57]])
        return np.concatenate(
            [Rpos, Raa, Lpos, Laa, (2 * gL - 1)[:, None], (2 * gR - 1)[:, None]], axis=1
        ).astype(np.float32)

    def _state_to_env_action(self, st: np.ndarray) -> np.ndarray:
        """ABSOLUTE state57 chunk (C,57) -> env actions DIRECTLY (no delta integration):
        per frame [Rpos3,Raa3,Lpos3,Laa3,gL,gR] from the state's abs9 blocks (pos +
        6D->rotation matrix->axis-angle, then the SAME _CTRL_OFF controller offsets the
        integration path applies at its end) and hand15 inverse-lerp grips (cmd=2g-1)."""
        st = np.asarray(st, float)

        def _arm(block):
            pos = block[:, :3]
            aa = np.stack([_R.from_matrix(_sixd_to_R(b[3:6], b[6:9])).as_rotvec() for b in block])
            return pos, aa

        Lpos, Laa = _arm(st[:, 9:18])
        Rpos, Raa = _arm(st[:, 33:42])
        Lpos, Laa = _apply_ctrl_offset(Lpos, Laa, "L")
        Rpos, Raa = _apply_ctrl_offset(Rpos, Raa, "R")
        gL = np.array([_grip_from_hand15(h) for h in st[:, 18:33]])
        gR = np.array([_grip_from_hand15(h) for h in st[:, 42:57]])
        return np.concatenate(
            [Rpos, Raa, Lpos, Laa, (2 * gL - 1)[:, None], (2 * gR - 1)[:, None]], axis=1
        ).astype(np.float32)

    # --- V2 sample: exact EgoVerseGr1RetV2Dataset layout (mid-v2 length-general) ---
    def _grid_demo_clip32(self, gm: EgoZarrEpisode, w0: int) -> np.ndarray:
        """(32,S,S,3) native grid-3cam clip of the retrieved demo (frame-clamped)."""
        h = self.S // 2
        return np.stack([
            self._grid3([self._demo_tile(gm, w0 + k, cam, h) for cam in (CAMS4 if self.grid4 else CAMS)])
            for k in range(CHUNK)
        ])

    def _build_sample_v2(self, imgs3: list[np.ndarray], proprio: dict,
                         demo_ep, demo_w0, has_ret: int, task: str) -> dict:
        """Inference twin of EgoVerseGr1RetV2Dataset.__getitem__ (state_t=20, k=g=8):

            0      cur_state  (gr1-normalized state57, injected)   cond  t=NOW
            1      ret_action (32x57 demo chunk, MD-normalized)    cond  t=NOW
            2..9   ret_video  (native 32f demo, grid-3cam)         cond  t=NOW-8dt..NOW-dt
            10     cur_image  (grid-3cam env obs)                  cond  t=NOW
            -- num_conditional_frames = 11 --
            11     action     (32x57, GENERATED -> zeros here)     GEN   t=NOW
            12..19 future     (GENERATED -> zeros here)            GEN   t=NOW+dt..NOW+8dt
        """
        S = self.S
        T = 2 * V2_K + 4                                          # state_t = 20
        gm = self._grp(demo_ep) if has_ret else None
        m_clip = self._grid_demo_clip32(gm, demo_w0) if has_ret \
            else np.zeros((CHUNK, S, S, 3), np.uint8)
        cur_grid = self._grid3(imgs3)

        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)
        # pixel packing: [state 1f | ret_act 4f | demo 32f | cur 4f | act 4f | future 32f] = 77f
        images = np.concatenate([
            blank1, blank4, m_clip,
            np.repeat(cur_grid[None], NUM_DUP, axis=0),
            blank4, np.zeros((CHUNK, S, S, 3), np.uint8),         # future GENERATED -> zeros
        ], axis=0)
        assert images.shape[0] == 1 + (T - 1) * NUM_DUP           # 77
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)

        # retrieved slot: native 32, normalized in the DEMO (mimicdroid) space.
        # NOTE: normalize the RAW-dtype rows (float64) and cast after, exactly like the
        # dataset's _chunk->_norm path — pre-casting to float32 shifts values by ~1e-4.
        # STATE mode: the demo's FUTURE state57 chunk s[w0+1 .. w0+32] (dataset:
        # _chunk(m_state, mw0+1)) instead of its action chunk.
        if has_ret and self.state_mode:
            m_state = gm.state("state57")
            ret32 = np.stack([m_state[min(max(demo_w0 + 1 + k, 0), len(m_state) - 1)]
                              for k in range(CHUNK)])
            ret_act = self._norm(ret32, self._mslo, self._mshi)
        elif has_ret and self.pred_target == "cmd_delta57" and not (
                self.plate_state_delta and task.startswith("PnPSinkToRightCounterPlate")):
            from cosmos_policy.datasets.egoverse_gr1_ret_v2_cmddelta_dataset import cmd_delta57 as _cd57
            m_abs = gm.action("abs_action57")
            rows = np.stack([m_abs[min(max(demo_w0 + k, 0), len(m_abs) - 1)]
                             for k in range(CHUNK + 1)])
            ret_act = self._norm(_cd57(rows), self._mcdlo, self._mcdhi)
        elif has_ret:
            m_act = gm.action("action57")
            ret32 = np.stack([m_act[min(max(demo_w0 + k, 0), len(m_act) - 1)]
                              for k in range(CHUNK)])
            ret_act = self._norm(ret32, self._mlo, self._mhi)
        else:
            ret_act = np.zeros((CHUNK, 57), np.float32)

        # robustness eval: perturb the normalized demo-action prior with FIXED-sigma
        # Gaussian noise (matches the dataset's train-time ret_action_noise_std aug).
        if has_ret and self._ret_action_noise > 0.0:
            ret_act = (ret_act + np.random.normal(
                0.0, self._ret_action_noise, ret_act.shape)).astype(np.float32)

        # slot 0 injection (QUERY/gr1 space):
        #   delta mode: the SINGLE current state57;
        #   STATE mode: the PAST state chunk s[t-31 .. t] from the proprio buffer,
        #   flattened (32*57,) so replace_latent_with_proprio tiles the whole chunk
        #   (same mechanism as the action chunk; dataset convention).
        if self.state_mode:
            cur_state = self._norm(self._past_state_chunk(proprio), self._slo, self._shi).reshape(-1)
        else:
            cur_state = self._norm(_state57_from_proprio(proprio)[None], self._slo, self._shi)[0]

        # per-token slot roles + continuous-time rope positions — EXACT dataset recipe
        roles = np.empty(T, np.int64); tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        roles[1], tpos[1] = ROLE_RET_ACTION, NOW_S
        roles[2:2 + V2_K] = ROLE_RET_VIDEO
        tpos[2:2 + V2_K] = NOW_S - V2_DT * np.arange(V2_K, 0, -1)
        roles[2 + V2_K], tpos[2 + V2_K] = ROLE_CUR_IMAGE, NOW_S
        roles[3 + V2_K], tpos[3 + V2_K] = ROLE_ACTION, NOW_S
        roles[4 + V2_K:], tpos[4 + V2_K:] = ROLE_FUTURE, NOW_S + V2_DT * np.arange(1, V2_K + 1)

        # t5: same key convention as training (raw class); zeros fallback
        cls = _base_task(task)
        t5 = (torch.squeeze(torch.as_tensor(self._t5[cls]))
              if cls in self._t5 else torch.zeros(512, 1024))

        zeros_act = np.zeros((CHUNK, 57), np.float32)
        return {
            "video": images,
            "command": str(cls),
            "actions": zeros_act,                                 # placeholder (generated)
            "action_mask": np.ones(57, np.float32),
            "retrieved_actions": ret_act.astype(np.float32),
            "retrieved_action_mask": np.ones(57, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "embodiment": "gr1", "retrieved_embodiment": DEMO_EMB,
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": V2_K + 3,                   # 11
            "retrieved_video_start_latent_idx": 2,
            "retrieved_video_end_latent_idx": 2 + V2_K,           # 10
            "retrieved_action_latent_idx": 1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 2 + V2_K,                 # 10
            "future_video_start_latent_idx": 4 + V2_K,            # 12
            "future_video_end_latent_idx": T,                     # 20
            "action_latent_idx": 3 + V2_K,                        # 11
            "proprio": cur_state.astype(np.float32),
            "future_proprio": np.zeros(57, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": 4 + V2_K, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": has_ret, "has_ret_image": has_ret, "has_current_image": 1,
            # POLICY sample: action latent must be GENERATED (world_model_sample_mask=0).
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": zeros_act.copy(),
        }

    # --- PLAIN sample: exact EgoVerseGr1PlainStateDataset layout (no retrieval) ---
    def _build_sample_plain(self, imgs3: list[np.ndarray], proprio: dict, task: str) -> dict:
        """Inference twin of EgoVerseGr1PlainStateDataset.build_sample (state_t=11):

            0      cur_state  (past state57 chunk, gr1-normalized, injected)  cond  t=NOW
            1      cur_image  (grid-3cam env obs)                             cond  t=NOW
            -- num_conditional_frames = 2 --
            2      action     (32x57, GENERATED -> zeros here)                GEN   t=NOW
            3..10  future     (GENERATED -> zeros here)                       GEN   t=NOW+dt..

        No demo is read, so the caller must NOT gate this on retrieval hits.
        """
        S = self.S
        T = V2_K + 3                                              # state_t = 11
        cur_grid = self._grid3(imgs3)

        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)
        # pixel packing: [cur_state 1f | cur_image 4f | action 4f | future 32f] = 41f
        images = np.concatenate([
            blank1,
            np.repeat(cur_grid[None], NUM_DUP, axis=0),
            blank4,
            np.zeros((CHUNK, S, S, 3), np.uint8),                 # future GENERATED -> zeros
        ], axis=0)
        assert images.shape[0] == 1 + (T - 1) * NUM_DUP           # 41
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)

        # slot 0 injection (QUERY/gr1 space) — same convention as _build_sample_v2.
        if self.state_mode:
            cur_state = self._norm(self._past_state_chunk(proprio), self._slo, self._shi).reshape(-1)
        else:
            cur_state = self._norm(_state57_from_proprio(proprio)[None], self._slo, self._shi)[0]

        roles = np.empty(T, np.int64); tpos = np.empty(T, np.float32)
        roles[0], tpos[0] = ROLE_CUR_STATE, NOW_S
        roles[1], tpos[1] = ROLE_CUR_IMAGE, NOW_S
        roles[2], tpos[2] = ROLE_ACTION, NOW_S
        roles[3:], tpos[3:] = ROLE_FUTURE, NOW_S + V2_DT * np.arange(1, V2_K + 1)

        cls = _base_task(task)
        t5 = (torch.squeeze(torch.as_tensor(self._t5[cls]))
              if cls in self._t5 else torch.zeros(512, 1024))

        zeros_act = np.zeros((CHUNK, 57), np.float32)
        return {
            "video": images,
            "command": str(cls),
            "actions": zeros_act,                                 # placeholder (generated)
            "action_mask": np.ones(57, np.float32),
            # NO retrieved_actions — matches the training dataset.
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "embodiment": "gr1", "retrieved_embodiment": "gr1",
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": 2,
            "retrieved_video_start_latent_idx": -1,
            "retrieved_video_end_latent_idx": -1,
            "retrieved_action_latent_idx": -1,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": 1,
            "future_video_start_latent_idx": 3,
            "future_video_end_latent_idx": T,                     # 11
            "action_latent_idx": 2,
            "proprio": cur_state.astype(np.float32),
            "future_proprio": np.zeros(57, np.float32),
            "current_proprio_latent_idx": 0,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": 3, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": 0, "has_ret_image": 0, "has_current_image": 1,
            # POLICY sample: action latent must be GENERATED (world_model_sample_mask=0).
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": zeros_act.copy(),
        }

    def _ret_slot_chunk(self, gm, demo_w0: int, task: str, has_ret: int) -> np.ndarray:
        """Retrieved-demo conditioning chunk — SAME branch logic as _build_sample_v2
        (state / cmd-delta / hybrid-Plate / legacy delta), factored for the short builder."""
        if has_ret and self.state_mode:
            m_state = gm.state("state57")
            ret32 = np.stack([m_state[min(max(demo_w0 + 1 + k, 0), len(m_state) - 1)]
                              for k in range(CHUNK)])
            return self._norm(ret32, self._mslo, self._mshi)
        if has_ret and self.pred_target == "cmd_delta57" and not (
                self.plate_state_delta and task.startswith("PnPSinkToRightCounterPlate")):
            from cosmos_policy.datasets.egoverse_gr1_ret_v2_cmddelta_dataset import cmd_delta57 as _cd57
            m_abs = gm.action("abs_action57")
            rows = np.stack([m_abs[min(max(demo_w0 + k, 0), len(m_abs) - 1)]
                             for k in range(CHUNK + 1)])
            return self._norm(_cd57(rows), self._mcdlo, self._mcdhi)
        if has_ret:
            m_act = gm.action("action57")
            ret32 = np.stack([m_act[min(max(demo_w0 + k, 0), len(m_act) - 1)]
                              for k in range(CHUNK)])
            return self._norm(ret32, self._mlo, self._mhi)
        return np.zeros((CHUNK, 57), np.float32)

    def _build_sample_short(self, imgs3: list[np.ndarray], proprio: dict,
                            demo_ep, demo_w0, has_ret: int, task: str) -> dict:
        """Inference twin of the GR1 SHORT datasets (egoverse_gr1_ret_short_dataset.py):
        NO demo video / NO future video.

            state variant  (state_t=4): [cur_state | ret_action | cur_image | action], n_cond=3
            delta variants (state_t=3): [ret_action | cur_image | action],             n_cond=2
        """
        S = self.S
        gm = self._grp(demo_ep) if has_ret else None
        cur_grid = self._grid3(imgs3)
        ret_act = self._ret_slot_chunk(gm, demo_w0, task, has_ret)

        blank1 = np.zeros((1, S, S, 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)
        cur4 = np.repeat(cur_grid[None], NUM_DUP, axis=0)
        # 2026-07-20: EVERY GR1 post-train feeds current state — both state and delta
        # short carry the PAST state chunk in slot 0 (state_t=4). (Was state_mode-only.)
        with_proprio = True
        if with_proprio:
            T = 4
            images = np.concatenate([blank1, blank4, cur4, blank4], axis=0)
            roles = np.array([ROLE_CUR_STATE, ROLE_RET_ACTION, ROLE_CUR_IMAGE, ROLE_ACTION], np.int64)
            ret_idx, img_idx, act_idx, prop_idx, n_cond = 1, 2, 3, 0, 3
            cur_state = self._norm(self._past_state_chunk(proprio), self._slo, self._shi).reshape(-1)
        else:
            T = 3
            images = np.concatenate([blank1, cur4, blank4], axis=0)
            roles = np.array([ROLE_RET_ACTION, ROLE_CUR_IMAGE, ROLE_ACTION], np.int64)
            ret_idx, img_idx, act_idx, prop_idx, n_cond = 0, 1, 2, -1, 2
            cur_state = np.zeros(57, np.float32)
        assert images.shape[0] == 1 + (T - 1) * NUM_DUP
        images = preprocess_image(images, final_image_size=S, normalize_images=False,
                                  use_image_aug=False, stronger_image_aug=False)
        tpos = np.full(T, NOW_S, np.float32)

        cls = _base_task(task)
        t5 = (torch.squeeze(torch.as_tensor(self._t5[cls]))
              if cls in self._t5 else torch.zeros(512, 1024))
        zeros_act = np.zeros((CHUNK, 57), np.float32)
        return {
            "video": images,
            "command": str(cls),
            "actions": zeros_act,
            "action_mask": np.ones(57, np.float32),
            "retrieved_actions": ret_act.astype(np.float32),
            "retrieved_action_mask": np.ones(57, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "embodiment": "gr1", "retrieved_embodiment": DEMO_EMB,
            "t_positions": tpos, "slot_roles": roles,
            "num_conditional_frames": n_cond,
            "retrieved_video_start_latent_idx": -1,
            "retrieved_video_end_latent_idx": -1,
            "retrieved_action_latent_idx": ret_idx,
            "retrieved_state_latent_idx": -1,
            "current_image_latent_idx": img_idx,
            "future_video_start_latent_idx": T,
            "future_video_end_latent_idx": T,
            "action_latent_idx": act_idx,
            "proprio": cur_state.astype(np.float32),
            "future_proprio": np.zeros(57, np.float32),
            "current_proprio_latent_idx": prop_idx,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1, "future_proprio_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image_latent_idx": -1, "future_image2_latent_idx": -1,
            "value_latent_idx": -1,
            "has_ret_data": has_ret, "has_ret_image": 0, "has_current_image": 1,
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1, "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": zeros_act.copy(),
        }

    def _to_env_action(self, act: np.ndarray, p: dict) -> np.ndarray:
        Lpos, Laa = _integrate_arm(p["pL"], _quat_mat(p["quatL"]), act[:, 9:18])
        Rpos, Raa = _integrate_arm(p["pR"], _quat_mat(p["quatR"]), act[:, 33:42])
        Lpos, Laa = _apply_ctrl_offset(Lpos, Laa, "L")
        Rpos, Raa = _apply_ctrl_offset(Rpos, Raa, "R")
        gL = np.array([_grip_from_hand15(h) for h in act[:, 18:33]])
        gR = np.array([_grip_from_hand15(h) for h in act[:, 42:57]])
        return np.concatenate(
            [Rpos, Raa, Lpos, Laa, (2 * gL - 1)[:, None], (2 * gR - 1)[:, None]], axis=1
        ).astype(np.float32)

    def _integrate_cmd_delta(self, d: np.ndarray, p: dict, anchor):
        """(C,57) cmd-deltas -> (C,57) abs command rows, chained on the command anchor.
        anchor=None (episode start): ctrl-offset-mapped measured pose (command==pose at rest;
        proprio quat is BODY-frame — map to the SITE/command frame like _hold_action)."""
        from cosmos_policy.datasets.egoverse_gr1_ret_v2_cmddelta_dataset import _sixd_to_R
        C = len(d)
        out = np.empty((C, 57), np.float64)
        out[:, 0:9] = _HEAD9
        def blk_from_pose(pos, quat, side):
            aa = _R.from_matrix(_quat_mat(quat)).as_rotvec()[None]
            cpos, caa = _apply_ctrl_offset(np.asarray(pos, float)[None], aa, side)
            Rm = _R.from_rotvec(caa[0]).as_matrix()
            return np.concatenate([cpos[0], Rm[:, 0], Rm[:, 1]])
        if anchor is None:
            anchor = (blk_from_pose(p["pL"], p["quatL"], "L"),
                      blk_from_pose(p["pR"], p["quatR"], "R"))
        for side, b, a0 in (("L", 9, anchor[0]), ("R", 33, anchor[1])):
            pos = np.asarray(a0[:3], float).copy()
            Rm = _sixd_to_R(np.asarray(a0[3:6])[None], np.asarray(a0[6:9])[None])[0]
            for k in range(C):
                dp = d[k, b:b + 3]
                dR = _sixd_to_R(d[k, b + 3:b + 6][None], d[k, b + 6:b + 9][None])[0]
                pos = Rm @ dp + pos
                Rm = Rm @ dR
                out[k, b:b + 3] = pos
                out[k, b + 3:b + 6] = Rm[:, 0]
                out[k, b + 6:b + 9] = Rm[:, 1]
            out[:, b + 9:b + 24] = d[:, b + 9:b + 24]
        new_anchor = (out[-1, 9:18].copy(), out[-1, 33:42].copy())
        return out, new_anchor

    def _hold_action(self, p: dict) -> np.ndarray:
        pL = np.asarray(p["pL"], float)[None]; pR = np.asarray(p["pR"], float)[None]
        aaL = _R.from_matrix(_quat_mat(p["quatL"])).as_rotvec()[None]
        aaR = _R.from_matrix(_quat_mat(p["quatR"])).as_rotvec()[None]
        pL, aaL = _apply_ctrl_offset(pL, aaL, "L")
        pR, aaR = _apply_ctrl_offset(pR, aaR, "R")
        one = np.concatenate([pR[0], aaR[0], pL[0], aaL[0],
                              [2 * p["gripL"] - 1, 2 * p["gripR"] - 1]]).astype(np.float32)
        return np.tile(one, (self.chunk_size, 1))

    @torch.no_grad()
    def predict(self, req: dict) -> dict:
        retr_task = req["task"]
        p = req["proprio"]
        if self.plain:
            # PLAIN arm: no demo is used, so skip retrieval entirely — including the
            # `if not hits: return zeros` gate below, which would otherwise stall the
            # policy whenever the (irrelevant) retriever came back empty.
            self._chunk += 1
            imgs3 = [_decode_image(im, self.S // 2)
                     for im in req["images"][:4 if self.grid4 else 3]]
            sample = self._build_sample_plain(imgs3, p, task=retr_task)
            return self._run_model(sample, p, retr_task, None)
        if not req.get("skip_push"):
            self.retriever.push(p["pL"], p["pR"], p["quatL"], p["quatR"], p["gripL"], p["gripR"])
            if self.play_retriever is not None:
                self.play_retriever.push(p["pL"], p["pR"], p["quatL"], p["quatR"], p["gripL"], p["gripR"])
        if self._pinned is not None:
            # PINNED one-shot demo: no retriever calls at all. The demo plays FORWARD
            # in sync with the robot: chunk c reads the demo's own timeline at
            # w0 + c*chunk_size (both sides consume chunk_size env steps per /predict),
            # clamped so the last window is the demo's final full chunk — simple,
            # deterministic, and matches how the t0-taskdemo path unrolls a demo.
            ep, w0_pin = self._pinned
            if self._chunk == 0 or self._pinned_windows is None:
                w0 = w0_pin                    # opening chunk: the demo's start (t=0 convention)
            else:
                w0s, T = self._pinned_windows
                w0_max = int(w0s[-1]) if len(w0s) else w0_pin
                last = getattr(self, "_pinned_last_w0", w0_pin)
                if self.pinned_advance == "lockstep":
                    # A: demo timeline advances exactly with execution (training one-shot
                    # rows are near-aligned, so this mirrors the training coupling).
                    # _pointer_step defaults to chunk_size (1x); the 2x server sets it to
                    # chunk_size//2 so a half-length 2x demo advances at the matching pace.
                    w0 = min(last + getattr(self, "_pointer_step", self.chunk_size), w0_max)
                else:
                    # WITHIN-EPISODE re-match: strict-past query vs the pinned episode's
                    # windows — phase-aligned instead of blind lockstep.
                    src = self.play_retriever if self.play_retriever is not None else self.retriever
                    q = src.query()
                    # --pin-lock-ep: TRAINING semantics — the one-shot slot conditions on ONE
                    # consistent demo per episode (global-DTW path over a single taskdemo).
                    # The 3-demo union still picks the best episode on the FIRST re-match;
                    # afterwards candidates are restricted to that episode (per-chunk demo
                    # flapping never appears in training).
                    eps_c = getattr(self, "_pinned_eps", None)
                    w0s_c, T_c = w0s, T
                    _lock = getattr(self, "_pinned_lock_ep", None)
                    if self.pin_lock_ep and eps_c is not None and _lock is not None:
                        _mk = eps_c == _lock
                        if _mk.any():
                            w0s_c, T_c, eps_c = w0s_c[_mk], T_c[_mk], eps_c[_mk]
                    w0_max_c = int(w0s_c[-1]) if len(w0s_c) else w0_max
                    if q is None:
                        w0 = w0_pin
                    elif self.pinned_advance == "mono":
                        # C: monotonic band + minimum advance — no phase-alias jumps
                        # (Cabinet t=32 -> demo@112) and no stall self-loop (Door stuck
                        # at demo@32). Floor is --mono-floor (default 8; was hardcoded 16):
                        # +8 lets the matcher track a robot that legitimately runs SLOWER
                        # than the demo (10s demo ~ 12s live) — fixed Plate/Cabinet
                        # premature-release (corn dropped beside the plate). +16 forbids
                        # sub-16 lingering — suspected better for FAST tasks (FaucetL3
                        # was 2/2 under 16, 1/6 under 8; A/B 2026-07-16 in flight).
                        lo, hi = last + self.mono_floor, last + getattr(self, "_mono_band", 48)
                        m = (w0s_c >= lo) & (w0s_c <= hi)
                        if m.any():
                            _j = _retrieval.match_within_idx(q[0], w0s_c[m], T_c[m])
                            w0 = int(w0s_c[m][_j])
                            if eps_c is not None:
                                ep = str(eps_c[np.where(m)[0][_j]])
                        else:
                            # band beyond the demo's last window (end saturation):
                            # clamp — and STAY on the locked episode (this fallback
                            # used to leak back to the pinned default demo, flipping
                            # the conditioning demo mid-episode, e.g. demo_2 -> demo_1
                            # at t=288 in the v6.1-6K Plate eval).
                            w0 = min(lo, w0_max_c)
                            if _lock is not None:
                                ep = str(_lock)
                    else:                       # "match": unconstrained — v6 3-demo union
                        _j = _retrieval.match_within_idx(q[0], w0s_c, T_c)
                        w0 = int(w0s_c[_j])
                        if eps_c is not None:
                            ep = str(eps_c[_j])
                    if self.pin_lock_ep and _lock is None:
                        self._pinned_lock_ep = ep   # first re-match fixes the episode
                w0 = int(min(w0, w0_max))
            self._pinned_last_w0 = w0
            hits = [{"ep": ep, "w0": int(w0), "dist": 0.0}]
        elif self.play_retriever is not None and self._chunk > 0:
            hits = self.play_retriever.retrieve(_base_task(retr_task), topk=1)
        else:
            hits = self.retriever.retrieve(retr_task, topk=1)
            if hits and self._chunk == 0 and self._t0_taskdemo:
                hits[0]["w0"] = 0
        self._chunk += 1
        if not hits:
            return {"action57": np.zeros((self.chunk_size, 57), np.float32).tolist(),
                    "env_action": self._hold_action(p).tolist(), "demo": None}
        demo = hits[0]
        # v2 grid tiles are S//2: decode straight to tile size (ONE resample from the
        # client-native image, exactly like the dataset's _decode(b, S//2) path).
        imgs3 = [_decode_image(im, self.S // 2 if self.v2 else self.S)
                 for im in req["images"][:4 if self.grid4 else 3]]
        if self.v2 and self.short:
            sample = self._build_sample_short(imgs3, p, demo["ep"], int(demo["w0"]),
                                              has_ret=1, task=retr_task)
        elif self.v2:
            sample = self._build_sample_v2(imgs3, p, demo["ep"], int(demo["w0"]),
                                           has_ret=1, task=retr_task)
        elif self.aligned:
            sample = self._build_sample_aligned(imgs3, p, demo["ep"], int(demo["w0"]),
                                                has_ret=1, task=retr_task)
        else:
            sample = self._build_sample(imgs3, p, demo["ep"], int(demo["w0"]), has_ret=1)
        return self._run_model(sample, p, retr_task, demo)

    def _dump_gen_video(self, latents, data_batch):
        """Decode the predicted FUTURE-video latent block -> RGB frames -> PNGs.
        Best-effort (never break inference): chunkNNNN_fKK.png in self._gen_video_dir."""
        def _iv(k):
            v = data_batch[k]
            return int(v[0]) if torch.is_tensor(v) else int(v)
        try:
            fvs, fve = _iv("future_video_start_latent_idx"), _iv("future_video_end_latent_idx")
            if fve <= fvs:
                return
            dec = self.model.decode(latents[:, :, fvs:fve])          # (B,3,T,H,W) in ~[-1,1]
            vid = ((dec[0].float().clamp(-1, 1) + 1) / 2 * 255).byte()
            vid = vid.permute(1, 2, 3, 0).cpu().numpy()              # (T,H,W,3)
            d = self._gen_ep_dir or self._gen_video_dir
            os.makedirs(d, exist_ok=True)
            for k in range(vid.shape[0]):
                Image.fromarray(vid[k]).save(
                    os.path.join(d, f"chunk{self._chunk:04d}_f{k:02d}.png"))
        except Exception as e:
            print(f"[gen-video] decode failed (chunk {self._chunk}): {e}", flush=True)

    def _save_attn(self, out_dir: str, cap: dict, sample: dict,
                   retr_task: str, demo: dict | None) -> None:
        """Write one npz per /predict call: the action-query attention maps plus the
        RGB frames they attend to, so the renderer needs nothing but this file.

        Pixel<->latent packing is the causal-VAE contract both layouts use: latent 0
        owns pixel frame 0, latent t>0 owns frames [1+(t-1)*4, 1+t*4). We store one
        representative frame per image-bearing slot (cur_image and, for ICL, each of
        the 8 ret_video latents)."""
        ep_dir = os.path.join(out_dir, f"ep{getattr(self, '_ep_seq', 0):04d}")
        os.makedirs(ep_dir, exist_ok=True)

        roles = np.asarray(sample["slot_roles"]).astype(int)
        video = sample["video"]                       # (C, F, S, S) uint8
        video = video.numpy() if torch.is_tensor(video) else np.asarray(video)

        def frame_of(slot: int) -> np.ndarray:
            f = 0 if slot == 0 else 1 + (slot - 1) * NUM_DUP
            f = min(f, video.shape[1] - 1)
            return np.transpose(video[:, f], (1, 2, 0))          # (S,S,3) uint8

        img_slots = [t for t in range(len(roles)) if int(roles[t]) in (ROLE_CUR_IMAGE, ROLE_RET_VIDEO)]
        frames = np.stack([frame_of(t) for t in img_slots]) if img_slots else np.zeros((0, 1, 1, 3), np.uint8)

        np.savez_compressed(
            os.path.join(ep_dir, f"chunk{self._chunk:04d}.npz"),
            attn_mean=cap["mean"].numpy().astype(np.float16),
            attn_last=cap["last"].numpy().astype(np.float16),
            layers=np.asarray(cap["layers"]),
            query_slot=np.asarray(cap["query_slot"]),
            denoise_steps=np.asarray(cap["steps"]),
            slot_roles=roles,
            t_positions=np.asarray(sample["t_positions"], np.float32),
            img_slot_idx=np.asarray(img_slots),
            frames=frames,
            task=np.asarray(retr_task),
            chunk=np.asarray(self._chunk),
            demo_ep=np.asarray("" if demo is None else str(demo["ep"])),
            demo_w0=np.asarray(-1 if demo is None else int(demo["w0"])),
        )

    def _run_model(self, sample: dict, p: dict, retr_task: str, demo: dict | None) -> dict:
        """Shared tail of predict(): run the RF model on an already-built sample, extract
        the action chunk, de-normalize into the configured target space and format the
        response. ``demo`` is None for the PLAIN (non-retrieval) arm."""
        data_batch = _to_cuda(sample, self.tk)

        # optional live-batch capture for format debugging (GR1_DUMP_DIR=<dir>)
        _dump = os.environ.get("GR1_DUMP_DIR")
        if _dump and self._chunk <= 12:
            os.makedirs(_dump, exist_ok=True)
            _np = {k: (v.float().cpu().numpy() if torch.is_tensor(v) else np.asarray(v))
                   for k, v in sample.items() if not isinstance(v, str)}
            if demo is not None:
                _np["demo_ep"] = np.asarray(demo["ep"]); _np["demo_w0"] = np.asarray(int(demo["w0"]))
            _np["task"] = np.asarray(retr_task)
            _np["raw_proprio"] = np.asarray(json.dumps(p))
            np.savez_compressed(os.path.join(_dump, f"chunk{self._chunk:02d}.npz"), **_np)

        _dtype = self.tk.get("dtype", torch.bfloat16)
        # Action-query self-attention dump (GR1_ATTN_DIR=<dir>). Off by default and
        # additive: when unset the sampler call below is byte-for-byte the original one.
        _attn_dir = os.environ.get("GR1_ATTN_DIR")
        with torch.autocast("cuda", dtype=_dtype):
            _raw, x0, _cond = self.model.get_data_and_condition(data_batch)
            # hoisted above the sampler: the capture needs the action slot index to know
            # which query rows to take, and reading it from the batch (not a literal)
            # keeps ICL (11) and plain/BC (2) correct without a branch.
            a_idx = int(data_batch["action_latent_idx"][0]) if torch.is_tensor(data_batch["action_latent_idx"]) \
                else int(data_batch["action_latent_idx"])
            _sample_fn = lambda: self.model.generate_samples_from_batch(
                data_batch, guidance=self.guidance, state_shape=x0.shape[1:],
                n_sample=x0.shape[0], num_steps=self.num_steps, shift=self.shift,
            )
            if _attn_dir:
                from analysis_tools.gr1_attn_capture import Gr1AttnCapture
                _lay = os.environ.get("GR1_ATTN_LAYERS")
                with Gr1AttnCapture(
                    self.model.net, query_slot=a_idx,
                    layers=[int(x) for x in _lay.split(",")] if _lay else None,
                    head_reduce=os.environ.get("GR1_ATTN_HEADS", "mean"),
                    step_stride=int(os.environ.get("GR1_ATTN_STEP_STRIDE", "4")),
                ) as _cap:
                    latents = _sample_fn()
                self._save_attn(_attn_dir, _cap.result(), sample, retr_task, demo)
            else:
                latents = _sample_fn()
            if self._gen_video_dir:
                self._dump_gen_video(latents, data_batch)
            if self.aligned:
                # (92,57) span chunk -> native 32 steps (pad: first rows / upsample: DOWN idx)
                act92 = extract_action_chunk_from_latent(
                    latents[:, :, a_idx], (ALIGNED_NF, 57),
                    decoder=getattr(self.model, "action_decoder", None))[0].float().cpu().numpy()
                act_norm = act92[:CHUNK] if self.aligned_layout == "pad" else act92[ALIGNED_DOWN]
            else:
                act_norm = extract_action_chunk_from_latent(
                    latents[:, :, a_idx], (self.chunk_size, 57),
                    decoder=getattr(self.model, "action_decoder", None))[0].float().cpu().numpy()
        if self.state_mode and getattr(self, "pred_target", "state57") == "abs_action57":
            # v6: predicted chunk = ABS COMMANDED targets — de-norm with abs stats,
            # env actions = the commands themselves (site frame, no offsets).
            act = (act_norm + 1.0) / 2.0 * (self._thi - self._tlo) + self._tlo
            env_action = self._abs_to_env_action(act)
        elif self.state_mode:
            # legacy: predicted chunk = ABSOLUTE state57
            act = (act_norm + 1.0) / 2.0 * (self._shi - self._slo) + self._slo
            env_action = self._state_to_env_action(act)
        elif self.pred_target == "cmd_delta57" and self.plate_state_delta \
                and retr_task.startswith("PnPSinkToRightCounterPlate"):
            # hybrid: Plate*는 LEGACY state-delta (측정 리앵커 자기보정)
            act = (act_norm + 1.0) / 2.0 * (self._hi - self._lo) + self._lo
            env_action = self._to_env_action(act, p)
        elif self.pred_target == "cmd_delta57":
            d = (act_norm + 1.0) / 2.0 * (self._cdhi - self._cdlo) + self._cdlo
            act, self._cmd_anchor = self._integrate_cmd_delta(d, p, self._cmd_anchor)
            env_action = self._abs_to_env_action(act)
        else:
            act = (act_norm + 1.0) / 2.0 * (self._hi - self._lo) + self._lo   # min-max de-normalize
            env_action = self._to_env_action(act, p)
        if _dump and self._chunk <= 12:
            np.savez_compressed(os.path.join(_dump, f"chunk{self._chunk - 0:02d}_out.npz"),
                                act_norm=act_norm, act=act, env_action=env_action)
        return {"action57": act.astype(np.float32).tolist(),
                "env_action": env_action.tolist(),
                "demo": (None if demo is None else
                         {"ep": demo["ep"], "w0": int(demo["w0"]),
                          "dist": float(demo.get("dist", 0.0))})}


def _decode_demo(g: EgoZarrEpisode, i: int, cam: str, size: int) -> np.ndarray:
    """Demo zarr frame -> (size,size,3) uint8."""
    i = min(max(i, 0), g.n_frames - 1)
    b = g.frame(i, cam)
    if not b:
        return np.zeros((size, size, 3), np.uint8)
    img = Image.open(io.BytesIO(b)).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, np.uint8)


import re as _re
_LVL_RE = _re.compile(r"L\d+$")

# eval-only tasks have no training language class (t5 pkl only holds the 4 seen tasks);
# without an alias the t5 lookup silently fell back to ZEROS for these two. Map to the
# nearest trained class (user 2026-07-27): MicrowaveTop = place-onto-surface -> Plate;
# RightDoor = door-closing -> LeftDoor.
_T5_ALIAS = {
    "CloseRightCabinetDoor": "CloseLeftCabinetDoor",
    "PnPSinkToMicrowaveTop": "PnPSinkToRightCounterPlate",
}


def _base_task(task: str) -> str:
    base = _LVL_RE.sub("", task)
    return _T5_ALIAS.get(base, base)


def _to_cuda(sample, tk):
    from torch.utils.data import default_collate
    batch = default_collate([sample])
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(**tk) if v.dtype.is_floating_point else v.cuda()
        else:
            out[k] = v
    return out


def make_handler(policy: GR1Policy):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            try:
                if self.path.rstrip("/") == "/reset":
                    policy.reset(req); self._send({"ok": True})
                elif self.path.rstrip("/") == "/push":
                    self._send(policy.push(req))
                elif self.path.rstrip("/") == "/predict":
                    self._send(policy.predict(req))
                else:
                    self._send({"error": f"unknown path {self.path}"}, 404)
            except Exception as e:  # noqa: BLE001
                import traceback
                self._send({"error": str(e), "trace": traceback.format_exc()}, 500)

        def log_message(self, *a):
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="cosmos_policy/config/config.py")
    ap.add_argument("--experiment", default="cosmos_predict2p5_2b_egoverse_gr1_ret_v6_448_state")
    ap.add_argument("--checkpoint", required=True, help="2.5 GR1 DCP checkpoint dir")
    # v4 defaults: 20-d grip-augmented pools (81_future_pairs_dth_grip). t=0 conditions on the
    # taskdemo opening (w0=0 warmup gate), t>0 re-matches against the DTH pool — mirroring the
    # egoverse_pairs_gr1_v4 training pairs. NOTE: serve-time same-scene exclusion is a NO-OP —
    # eval scenes are held-out seeds that never appear in the DTH pool (see DthDemoPool.match).
    ap.add_argument("--pool", default="/mnt/ddn/dataset/egoverse_pairs_v2/taskdemo_pool_v6.npz",
                    help="t=0 pool (taskdemo_pool_v4.npz, 20-d)")
    ap.add_argument("--pool-kind", default="taskdemo", choices=["taskdemo", "play", "dth"])
    ap.add_argument("--play-pool", default="/mnt/ddn/dataset/egoverse_pairs_v2/dth_pool_1x_v6.npz",
                    help="t>0 pool: t=0 conditions on --pool taskdemo opening, t>0 uses this "
                         "(dth_pool_1x_v6.npz, 20-d; pass an empty string to disable)")
    ap.add_argument("--play-pool-kind", default="dth", choices=["play", "dth"],
                    help="matcher class for --play-pool (v5 = dth)")
    ap.add_argument("--pairs-dir", default="/mnt/ddn/dataset/egoverse_pairs_gr1_v6")
    ap.add_argument("--data-root", default="/mnt/ddn/dataset")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--shift", type=float, default=None,
                    help="flow sampling shift; default None = read model.config.shift "
                         "(training value, 5 for these runs). 3.0 was a silent train/eval "
                         "mismatch in every earlier online eval.")
    ap.add_argument("--pred-target", default="abs_action57", choices=["abs_action57", "state57", "cmd_delta57"],
                    help="v6: model predicts ABSOLUTE COMMANDED targets (abs_action57, site frame "
                         "-> env actions directly, no ctrl offset); state57 = legacy state-chunk path")
    ap.add_argument("--plate-state-delta", action="store_true",
                    help="hybrid(cdh) models: PnPSinkToRightCounterPlate* decodes via LEGACY state-delta (action57)")
    ap.add_argument("--grid4", action="store_true",
                    help="4-view grid model: BR quadrant = center_top (client sends 4 images; 2026-07-16+)")
    ap.add_argument("--mono-floor", type=int, default=8,
                    help="mono band minimum advance per re-match (frames): 8 = wait for a slow robot "
                         "(PnP placement fix), 16 = legacy pace (suspected better for fast tasks)")
    ap.add_argument("--ret-action-noise", type=float, default=0.0,
                    help="ROBUSTNESS eval: fixed Gaussian sigma added to the normalized "
                         "ret_action conditioning (matches dataset ret_action_noise_std). "
                         "0 = off; sweep e.g. 0 / 0.1 / 0.2")
    ap.add_argument("--ret-video-blur", type=float, default=0.0,
                    help="Gaussian blur sigma (px, at the S//2 camera-tile scale) applied to the "
                         "RETRIEVED-DEMO video only. 0 = off.")
    ap.add_argument("--gen-video-dir", default=None,
                    help="if set, decode the predicted FUTURE-video latents each inference "
                         "and dump frames (chunkNNNN_fKK.png) here for the generated-video viewer")
    ap.add_argument("--pin-lock-ep", action="store_true",
                    help="v6: after the first re-match, lock the pinned one-shot demo to that "
                         "EPISODE (training conditions on one consistent demo per episode)")
    ap.add_argument("--pinned-advance", default="match", choices=["match", "lockstep", "mono"],
                    help="pinned one-shot demo pointer: match=unconstrained re-match (legacy), "
                         "lockstep=+chunk_size per chunk, mono=monotonic band [last+16,last+48]")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--save-video-dir", default=None)
    args = ap.parse_args()

    # SDPA robustness (2026-07-20): the VIDEO models (e.g. state_dmid) run the demo-video
    # VAE whose attention has head_dim>128 -> flash/mem-efficient/cudnn are all disabled;
    # if the math backend is off too, SDPA raises "No available kernel" -> intermittent
    # 500 on /predict. Force-enable the math fallback so a kernel always exists.
    try:
        # Force the MATH backend only: flash/mem-efficient/cuDNN get runtime-disabled for
        # the VAE's head_dim>128 attention, and leaving them "enabled" makes SDPA raise
        # "No available kernel" instead of falling back. Math is always available.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)
        except Exception:
            pass
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    policy = GR1Policy(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    print(f"cosmos_policy 2.5 GR1 conditioning-policy server ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
