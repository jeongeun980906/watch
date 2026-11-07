"""Policy server for RoboCasa GR1 tabletop online eval.

Why a server. The tabletop environment lives in robocasa-gr1-tabletop-tasks, which installs a
package literally named `robocasa` — the same name as robocasa-cosmos-policy, which this venv
already has for the RoboCasa kitchen eval. The two cannot coexist in one environment, so the
policy runs here (cosmos_policy venv) and the environment runs in its own venv and talks to
this process over HTTP. This mirrors cosmos-gr1-fulleval-*-8gpu.yaml, which splits the GR1
eval the same way for the same reason.

Protocol (JSON over HTTP, one server per GPU):
    POST /reset    {"task": "PnPBottleToCabinetClose"}      -> {"ok": true}
    POST /predict  {"image": <b64 jpeg>, "state": [44 floats]}
                   -> {"actions": [[44 floats] x num_open_loop_steps]}
    POST /health   {}                                        -> {"ok": true, "task": ...}

The latent layout is built by cosmos_utils.get_action with use_wrist_image=False and
num_third_person_images=1, which yields blank | proprio | ego_view | action | future proprio |
future ego_view — the same order GR1TabletopDataset writes during training. Do not set
use_retrieval-style flags here; that branch orders the future slots differently.

Usage:
    .venv/bin/python -m cosmos_policy.scripts.eval_gr1_tabletop_server \
        --checkpoint /mnt/ddn/tmp/gr1_tabletop_base/.../model_000040000.pt --port 8900
"""

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from PIL import Image

from cosmos_policy.experiments.robot import cosmos_utils
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)

ACTION_DIM = 44
NUM_DUP = 4          # frames per duplicated latent slot
CHUNK_SIZE = 32


def _make_cfg(args) -> "PolicyEvalConfig":
    """Build the eval config get_action() expects.

    This subclasses run_robocasa_eval's PolicyEvalConfig rather than duck-typing it. get_action
    and its helpers read 40 different cfg attributes across ensemble, planning-model and
    parallel-inference branches; hand-listing the ones the GR1 path happens to touch means any
    unlisted attribute becomes a 500 at inference time (that is exactly how `cfg.suite` was
    found — after the model had loaded and the environment had reset). Inheriting gives every
    field its upstream default, and only the GR1-specific ones are overridden below.
    """
    from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import PolicyEvalConfig

    cfg = PolicyEvalConfig()
    # "real_robot", not a new suite name: get_action dispatches the camera layout on cfg.suite,
    # and that branch is "single primary (third-person) image, no wrist images ... pairs with
    # state_t=6 model" — exactly the GR1 tabletop layout. Adding a gr1_tabletop branch would
    # duplicate it. The suite name does not affect action dimensionality; that comes from
    # constants.py, which does have a GR1_TABLETOP entry (ACTION_DIM=44).
    cfg.suite = "real_robot"
    cfg.config = args.experiment
    cfg.ckpt_path = args.checkpoint
    cfg.config_file = args.config
    # One ego camera: third-person slot count 1, no wrist. This is what makes the latent
    # layout match GR1TabletopDataset.
    cfg.use_third_person_image = True
    cfg.num_third_person_images = 1
    cfg.use_wrist_image = False
    cfg.num_wrist_images = 0
    cfg.use_proprio = True
    cfg.predict_future_states = True
    cfg.predict_values = False
    cfg.flip_images = False  # the gym env returns upright frames
    cfg.use_variance_scale = False
    # False: GR1TabletopDataset decodes mp4 frames straight into preprocess_image with no JPEG
    # step, so compressing here would show the policy an artifact distribution it never trained
    # on. (RoboCasa sets True because its HDF5 stores JPEG-encoded frames.)
    cfg.use_jpeg_compression = False
    cfg.trained_with_image_aug = True
    # Must match how the checkpoint was TRAINED. The warm-start arms enable slot-role /
    # continuous-time PE (dataloader_train.dataset.emit_slot_pe=True); the base arm does not.
    # Getting this backwards feeds the policy a positional regime it never saw.
    cfg.emit_slot_pe = getattr(args, "slot_pe", False)  # other callers build args without it
    cfg.normalize_proprio = True
    cfg.unnormalize_actions = True
    cfg.chunk_size = args.chunk_size
    cfg.num_open_loop_steps = args.num_open_loop_steps
    cfg.num_denoising_steps_action = args.num_denoising_steps_action
    cfg.num_denoising_steps_future_state = 1
    cfg.num_denoising_steps_value = 1
    cfg.ar_future_prediction = False
    cfg.ar_value_prediction = False
    cfg.ar_qvalue_prediction = False
    cfg.num_queries_best_of_n = 1
    cfg.use_parallel_inference = False
    cfg.shift = args.shift
    cfg.seed = args.seed
    cfg.randomize_seed = False
    cfg.deterministic = True
    cfg.dataset_stats_path = args.dataset_stats_path
    cfg.t5_text_embeddings_path = args.t5_embeddings
    return cfg


def _to_cuda(sample: dict, dtype=torch.bfloat16) -> dict:
    from torch.utils.data import default_collate

    batch = default_collate([{k: v for k, v in sample.items() if not isinstance(v, str)}])
    out = {}
    for k, v in batch.items():
        out[k] = (v.to(device="cuda", dtype=dtype) if v.dtype.is_floating_point else v.cuda()) \
            if torch.is_tensor(v) else v
    for k, v in sample.items():
        if isinstance(v, str):
            out[k] = v
    return out


class BridgeSampleBuilder:
    """Inference twin of GR1TabletopDataset's bridge layout.

    get_action() cannot build this: its real_robot branch emits
    blank | proprio | image | action | future proprio | future image and has no way to place the
    zero-pixel retrieval prefix. The reference eval server has the same split (see
    eval_gr1_server.py::_build_sample_plain), for the same reason.

        0  cur_state   proprio injected, blank pixels        cond   t=NOW
        1  ret_action  blank                                 cond   t=NOW
        2  ret_video   blank                                 cond   t=NOW-DT   (past)
        3  cur_image   the env frame                         cond   t=NOW
        -- num_conditional_frames = 4 --
        4  action      GENERATED                             GEN    t=NOW
        5  future      GENERATED                             GEN    t=NOW+0.8

    Roles and timestamps come from the training loader's build_slot_pe, so the two cannot drift.
    """

    def __init__(self, image_size: int, state_t: int, future_offset_s: float = 16 / 20.0):
        from cosmos_policy.datasets.gr1_tabletop_dataset import build_slot_pe

        self.S = image_size
        self.state_t = state_t
        self.tpos, self.roles = build_slot_pe(
            state_t=state_t,
            proprio_idx=0,
            image_idx=3,
            action_idx=4,
            future_proprio_idx=-1,
            future_image_idx=5,
            future_offset_s=future_offset_s,
            ret_action_idx=1,
            ret_video_start=2,
            ret_video_end=3,
        )

    def build(self, image: np.ndarray, proprio: np.ndarray, t5: torch.Tensor, command: str) -> dict:
        from cosmos_policy.datasets.dataset_utils import preprocess_image

        # Blanks are built at the INCOMING frame size, not the final one: the training loader
        # does `np.zeros_like(current_image)` on the raw decoded frame and lets preprocess_image
        # resize the whole stack at once. Building them at S instead breaks the concat whenever
        # the env frame is not already S (it is 256 here).
        S = self.S
        blank1 = np.zeros((1, *image.shape[:2], 3), np.uint8)
        blank4 = np.repeat(blank1, NUM_DUP, axis=0)
        cur = np.repeat(image[None], NUM_DUP, axis=0)
        images = np.concatenate([blank1, blank4, blank4, cur, blank4, blank4], axis=0)
        assert images.shape[0] == 1 + (self.state_t - 1) * NUM_DUP, images.shape
        images = preprocess_image(
            images, final_image_size=S, normalize_images=False,
            use_image_aug=False, stronger_image_aug=False,
        )
        zeros_act = np.zeros((CHUNK_SIZE, ACTION_DIM), np.float32)
        return {
            "video": images,
            "command": command,
            "actions": zeros_act,
            "action_mask": np.ones(ACTION_DIM, np.float32),
            "t5_text_embeddings": t5,
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, S, S),
            "image_size": S * torch.ones(4),
            "__key__": 0,
            "t_positions": self.tpos,
            "slot_roles": self.roles,
            "num_conditional_frames": 4,
            "retrieved_video_start_latent_idx": 2,
            "retrieved_video_end_latent_idx": 3,
            "retrieved_action_latent_idx": 1,
            "retrieved_state_latent_idx": -1,
            "current_proprio_latent_idx": 0,
            "current_image_latent_idx": 3,
            "action_latent_idx": 4,
            "future_image_latent_idx": 5,
            "future_video_start_latent_idx": 5,
            "future_video_end_latent_idx": self.state_t,
            "future_proprio_latent_idx": -1,
            "current_wrist_image_latent_idx": -1, "current_wrist_image2_latent_idx": -1,
            "current_image2_latent_idx": -1,
            "future_wrist_image_latent_idx": -1, "future_wrist_image2_latent_idx": -1,
            "future_image2_latent_idx": -1, "value_latent_idx": -1,
            "proprio": proprio.astype(np.float32),
            "future_proprio": np.zeros(ACTION_DIM, np.float32),
            # POLICY sample: the action latent is GENERATED, not reconstructed.
            "rollout_data_mask": 0, "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0, "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "value_function_return": float("-100"),
            "next_value_function_return": float("-100"),
            "next_action_chunk": zeros_act.copy(),
        }


class TabletopPolicy:
    def __init__(self, args):
        self.cfg = _make_cfg(args)
        init_t5_text_embeddings_cache(self.cfg.t5_text_embeddings_path)
        if not cosmos_utils.t5_text_embeddings_cache:
            raise RuntimeError(f"No text embeddings loaded from {self.cfg.t5_text_embeddings_path}")

        # The instruction set is closed (186 sentences drawn from the dataset's own `remarks`),
        # so a cache miss means the client sent something this policy never trained on — not
        # something to paper over by encoding online with a T5 that is not even loaded here.
        def _no_online(task_label: str):
            raise RuntimeError(
                f"Task '{task_label}' has no precomputed embedding in {self.cfg.t5_text_embeddings_path}. "
                f"Known tasks: {sorted(cosmos_utils.t5_text_embeddings_cache)[:3]}..."
            )

        cosmos_utils.get_text_embedding = _no_online
        cosmos_utils.save_t5_text_embeddings_cache = lambda: None

        self.model, self.model_config = get_model(self.cfg)
        assert self.cfg.chunk_size == self.model_config.dataloader_train.dataset.chunk_size, (
            f"chunk_size mismatch: server {self.cfg.chunk_size} vs "
            f"training {self.model_config.dataloader_train.dataset.chunk_size}"
        )
        # Reuse the training normalisation. Passing no stats path means the loader pooled
        # meta/stats.json at train time, so the server has to be given the same numbers
        # explicitly — otherwise actions are unnormalised against the wrong range.
        self.dataset_stats = load_dataset_stats(self.cfg.dataset_stats_path)
        self.task = None
        # The bridge contract cannot go through get_action (see BridgeSampleBuilder); detect it
        # from the experiment name so the server cannot be pointed at a bridge checkpoint with
        # the wrong builder.
        self.bridge = "_bridge" in args.experiment
        self.builder = (
            BridgeSampleBuilder(
                # Read the size the loader was configured with rather than assuming: a mismatch
                # here reshapes every frame the policy sees.
                image_size=int(self.model_config.dataloader_train.dataset.final_image_size),
                state_t=self.model.config.state_t,
            )
            if self.bridge
            else None
        )
        if self.bridge:
            print(f"bridge layout: state_t={self.model.config.state_t} roles={self.builder.roles}", flush=True)
        print(f"tabletop policy ready | {len(cosmos_utils.t5_text_embeddings_cache)} tasks", flush=True)

    def reset(self, req: dict) -> None:
        # The embedding cache is keyed by the per-episode instruction (186 sentences), not by task
        # id — the object is randomised per episode and the sentence names it. The client reads
        # that sentence off the env and sends it here; `task` remains only as a fallback for a
        # cache built the old way.
        label = req.get("language") or req.get("task")
        if label not in cosmos_utils.t5_text_embeddings_cache:
            known = sorted(cosmos_utils.t5_text_embeddings_cache)[:2]
            raise ValueError(
                f"no embedding for '{label}'. The cache holds "
                f"{len(cosmos_utils.t5_text_embeddings_cache)} keys, e.g. {known}"
            )
        self.task = label

    def predict(self, req: dict) -> dict:
        if self.task is None:
            raise RuntimeError("/predict called before /reset")
        image = np.array(Image.open(io.BytesIO(base64.b64decode(req["image"]))).convert("RGB"))
        state = np.asarray(req["state"], dtype=np.float32)
        if state.shape != (ACTION_DIM,):
            raise ValueError(f"state must be {ACTION_DIM}-d, got {state.shape}")

        if self.bridge:
            return {"actions": self._predict_bridge(image, state)}

        # get_action's real_robot branch reads obs["primary_image"], and proprio from
        # obs["proprio"] (not "state").
        observation = {"primary_image": image, "proprio": state}
        with torch.no_grad():
            out = get_action(
                self.cfg,
                self.model,
                self.dataset_stats,
                observation,
                self.task,
                seed=self.cfg.seed,
                randomize_seed=False,
                num_denoising_steps_action=self.cfg.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=True,
            )
        # get_action already unnormalises internally when cfg.unnormalize_actions is set, so
        # these come back in dataset scale. Calling unnormalize_actions again here would apply
        # the transform twice and hand the robot silently wrong targets.
        actions = np.asarray(out["actions"], dtype=np.float64).reshape(-1, ACTION_DIM)
        return {"actions": actions[: self.cfg.num_open_loop_steps].tolist()}


def _predict_bridge_impl(self, image: np.ndarray, state: np.ndarray) -> list:
    """Run the bridge layout end to end: build -> generate -> extract -> unnormalise."""
    from cosmos_policy.experiments.robot.cosmos_utils import rescale_proprio, unnormalize_actions
    from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent

    proprio = rescale_proprio(state, self.dataset_stats, non_negative_only=False, scale_multiplier=1.0)
    sample = self.builder.build(
        image=image,
        proprio=np.asarray(proprio, dtype=np.float32),
        t5=torch.squeeze(cosmos_utils.t5_text_embeddings_cache[self.task]),
        command=self.task,
    )
    data_batch = _to_cuda(sample)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _raw, x0, _cond = self.model.get_data_and_condition(data_batch)
        latents = self.model.generate_samples_from_batch(
            data_batch,
            guidance=self.cfg.guidance if hasattr(self.cfg, "guidance") else 1.5,
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            num_steps=self.cfg.num_denoising_steps_action,
            shift=self.cfg.shift,
        )
        a_idx = int(data_batch["action_latent_idx"][0])
        act_norm = extract_action_chunk_from_latent(
            latents[:, :, a_idx],
            (self.cfg.chunk_size, ACTION_DIM),
            decoder=getattr(self.model, "action_decoder", None),
        )[0].float().cpu().numpy()
    actions = unnormalize_actions(act_norm[None], self.dataset_stats)[0]
    return np.asarray(actions, dtype=np.float64)[: self.cfg.num_open_loop_steps].tolist()


TabletopPolicy._predict_bridge = _predict_bridge_impl


def make_handler(policy: TabletopPolicy):
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
            path = self.path.rstrip("/")
            try:
                if path == "/reset":
                    policy.reset(req)
                    self._send({"ok": True})
                elif path == "/predict":
                    self._send(policy.predict(req))
                elif path == "/health":
                    self._send({"ok": True, "task": policy.task})
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
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--experiment", default="cosmos_predict2p5_2b_480p_gr1_tabletop__inference")
    ap.add_argument("--config", default="cosmos_policy/config/config.py")
    ap.add_argument("--t5-embeddings", default="/mnt/ddn/dataset/gr1_tabletop_teleop/t5_embeddings_perep.pkl")
    ap.add_argument(
        "--dataset-stats-path",
        default="/mnt/ddn/dataset/gr1_tabletop_teleop/dataset_statistics_v2.json",
        help="Pooled stats written by export_gr1_tabletop_stats.py — must be the same numbers "
        "GR1TabletopDataset used during training.",
    )
    ap.add_argument("--chunk-size", type=int, default=32)
    ap.add_argument("--num-open-loop-steps", type=int, default=16)
    ap.add_argument("--num-denoising-steps-action", type=int, default=5)
    ap.add_argument(
        "--slot-pe",
        action="store_true",
        help="emit slot_roles/t_positions — set this for checkpoints trained with "
        "dataloader_train.dataset.emit_slot_pe=True (the EgoVerse warm-start arms)",
    )
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=195)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8900)
    args = ap.parse_args()

    policy = TabletopPolicy(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(policy))
    print(f"GR1 tabletop policy server ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
