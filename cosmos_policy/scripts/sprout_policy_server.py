# SPDX-License-Identifier: Apache-2.0
"""Inference server for the Sprout real-robot post-train (``cosmos_predict2p5_2b_sprout_3task_pv0_*``).

It speaks the GR00T ``PolicyServer`` protocol (zmq REP + msgpack-numpy), the one the robot client
``sprout_real/policy.py`` already uses, so the client needs no change::

    {"endpoint": "ping"}                                   -> {"status": "ok", "chunk": C, ...}
    {"endpoint": "get_action", "data": {"observation": {
        "video":    {"front": (1, 1, 224, 224, 3) uint8 RGB, or JPEG bytes},
        "state":    {"base": (1, 1, 9), "left_eef": (1, 1, 9), "left_gripper": (1, 1, 1),
                     "right_eef": (1, 1, 9), "right_gripper": (1, 1, 1)},     float32, the 29-D state
        "language": {"annotation.human.task_description": [["<instruction>"]]}}}}
                                                           -> [{part: (1, C, d)}, {"ms": ...}]

**The sample is built by the training dataset's code**, not by a copy of it. The server opens the
store (``SPROUT_ROOT``: manifest + h5, ``hf download Jeongeun/sprout-real-3task``), takes the first
window of an episode of the requested task, and swaps that episode for a one-frame live episode
(the robot's image and state). ``SproutPlainEndDataset._plain_build`` then normalizes the state,
lays out the latent slots, roles, time positions and T5 exactly as in training. The image goes the
training way as well: RGB -> JPEG at quality 92 with cv2 (``to_sprout29.py``) -> the dataset's decoder.
The only differences from a training sample are the generated slots: the action target is zero and
the future image is black, as in ``tabletop_sim_eval_server.py``. ``--selftest`` checks this
against real training samples.

Launch (GPU; the experiment name gives the chunk size and must match the checkpoint)::

    CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=12370 \\
    PYTHONPATH=. uv run --extra cu128 --group robocasa --python 3.10 \\
      python -m cosmos_policy.scripts.sprout_policy_server \\
        --experiment cosmos_predict2p5_2b_sprout_3task_pv0_c32 \\
        --checkpoint <.../sprout_pv0_c32/model_000004000.pt> --port 5555

    # no GPU: the sample builder against training samples
    python -m cosmos_policy.scripts.sprout_policy_server --experiment ..._c32 --selftest
"""
from __future__ import annotations

import argparse
import os
import re
import time

import cv2
import numpy as np

from cosmos_policy.datasets.sprout_dataset import DIM29, SPROUT_ROOT, make

PARTS = ("base", "left_eef", "left_gripper", "right_eef", "right_gripper")
SLICES = dict(zip(PARTS, (slice(0, 9), slice(9, 18), slice(18, 19), slice(19, 28), slice(28, 29))))
LANGUAGE = "annotation.human.task_description"
JPEG_QUALITY = 92  # to_sprout29.py --jpeg-quality
SIZE = 224


def jpeg(rgb: np.ndarray) -> bytes:
    """RGB -> the JPEG the store holds: cv2 encodes BGR at quality 92."""
    rgb = np.asarray(rgb, np.uint8)
    if rgb.shape != (SIZE, SIZE, 3):
        raise ValueError(f"image must be {SIZE}x{SIZE}x3 RGB, got {rgb.shape}")
    ok, buf = cv2.imencode(".jpg", rgb[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


_BLACK = None


class LiveEpisode:
    """One observation in the shape of a ``TabletopEpisode``: frame 0 is the robot's image, every later
    frame black (the future slot is generated, its pixels unused), the state is the robot's for every
    frame, the action zero."""

    def __init__(self, image_jpeg: bytes, state29: np.ndarray, n_frames: int):
        global _BLACK
        if _BLACK is None:
            _BLACK = jpeg(np.zeros((SIZE, SIZE, 3), np.uint8))
        self._img, self._n = image_jpeg, n_frames
        self._state = np.tile(np.asarray(state29, np.float64).reshape(1, DIM29), (n_frames, 1))

    @property
    def n_frames(self) -> int:
        return self._n

    def frame(self, i: int, cam: str = "") -> bytes:
        return self._img if int(i) <= 0 else _BLACK

    def state(self, group: str) -> np.ndarray:
        return self._state

    def action(self, group: str) -> np.ndarray:
        return np.zeros_like(self._state)

    def has_state(self, group: str) -> bool:
        return True


class SampleBuilder:
    """Robot observation -> training-format sample, through ``SproutPlainEndDataset``."""

    def __init__(self, root: str, chunk: int):
        self.ds = make(root, chunk_size=chunk)
        self.chunk = chunk
        w0 = np.asarray(self.ds.ww0)
        self.slot = {}  # instruction -> a window at frame 0 of one of its episodes
        for q in np.flatnonzero(w0 == 0):
            self.slot.setdefault(str(self.ds.wcls[q]), int(q))
        emb = str(self.ds.wemb[next(iter(self.slot.values()))])
        tgt = self.ds.target_stats if self.ds.target_stats else self.ds.state_stats
        s = tgt.get(emb, tgt["_global"])
        self.lo, self.hi = np.asarray(s["lo"], np.float64), np.asarray(s["hi"], np.float64)

    def instructions(self) -> list[str]:
        return sorted(self.slot)

    def build(self, image, state29: np.ndarray, instruction: str) -> dict:
        if instruction not in self.slot:
            raise KeyError(f"unknown instruction {instruction!r}; trained on {self.instructions()}")
        q = self.slot[instruction]
        img = bytes(image) if isinstance(image, (bytes, bytearray)) else jpeg(image)
        ds = self.ds
        ds._zcache[str(ds.wep[q])] = LiveEpisode(img, state29, self.chunk + 1)
        s = ds._plain_build(q, 0, False, None)
        s["actions"] = np.zeros_like(s["actions"])
        s["next_action_chunk"] = np.zeros_like(s["next_action_chunk"])
        return s

    def unnormalize(self, a: np.ndarray) -> np.ndarray:
        return (np.asarray(a, np.float64) + 1.0) / 2.0 * (self.hi - self.lo + 1e-8) + self.lo


def parse(obs: dict):
    """GR00T-style observation -> (image: RGB array or JPEG bytes, state29, instruction)."""
    image = obs["video"]["front"]
    if not isinstance(image, (bytes, bytearray)):
        image = np.asarray(image).reshape(SIZE, SIZE, 3)
    state = np.concatenate([np.asarray(obs["state"][k], np.float64).reshape(-1) for k in PARTS])
    if state.shape != (DIM29,) or not np.isfinite(state).all():
        raise ValueError(f"bad state {state.shape}")
    lang = obs["language"][LANGUAGE]
    return image, state, str(lang[0][0] if isinstance(lang[0], (list, tuple)) else lang[0])


def split(act: np.ndarray) -> dict:
    """(C, 29) -> {part: (1, C, d)}, the GR00T reply."""
    act = np.asarray(act, np.float32)
    return {k: act[None, :, SLICES[k]] for k in PARTS}


def selftest(b: SampleBuilder, n: int = 6) -> None:
    """Training sample vs. the server's sample for the same frame: every field but the generated slots."""
    from cosmos_policy.datasets.egoverse_ret_dataset import _decode

    import torch

    ds = b.ds
    rng = np.random.default_rng(0)
    T, dup = ds.state_t, 4
    n_cond = None
    for q in rng.choice(np.arange(ds.N), n, replace=False):
        q = int(ds.query_idx[q])
        g = ds._grp(q)
        w0 = int(ds.ww0[q])
        state = g.state(ds.state_group)[w0]
        raw = g.frame(w0, "images.front_1")
        ref = ds._plain_build(q, 0, False, None)
        rgb = _decode(raw, SIZE)
        for image, exact in ((raw, True), (rgb, False)):  # the stored JPEG; the robot's RGB, re-encoded
            got = b.build(image, state, str(ds.wcls[q]))
            del ds._zcache[str(ds.wep[b.slot[str(ds.wcls[q])]])]
            n_cond = int(got["num_conditional_frames"])
            for k, v in ref.items():
                if k in ("actions", "next_action_chunk", "__key__"):
                    continue
                w = got[k]
                if k == "video":
                    v, w = torch.as_tensor(v).float(), torch.as_tensor(w).float()
                    cond = 1 + (n_cond - 1) * dup  # frames of the conditioning slots
                    d = (v[:, :cond] - w[:, :cond]).abs().mean().item()
                    assert (d == 0) if exact else d < 2.0, (k, d)
                    assert (w[:, cond + dup:] == 0).all(), "future slot not black"
                elif torch.is_tensor(v) or isinstance(v, np.ndarray):
                    assert torch.equal(torch.as_tensor(v).float(), torch.as_tensor(w).float()), k
                else:
                    assert v == w, (k, v, w)
        print(f"  window {q} ({ds.wcls[q]}, frame {w0}): equal to the training sample; "
              f"re-encoded image {d:.2f}/255 mean abs")
    print(f"selftest ok: state_t {T}, chunk {b.chunk}, n_cond {n_cond}, instructions {b.instructions()}")


class Policy:
    def __init__(self, a):
        import torch

        from cosmos_policy._src.imaginaire.utils import distributed
        from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init
        from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint

        with distributed_init():
            distributed.init()
        self.model, config = load_model_from_checkpoint(
            experiment_name=a.experiment, s3_checkpoint_dir=a.checkpoint, config_file=a.config,
            enable_fsdp=False, load_ema_to_reg=False)
        self.model = self.model.cuda().eval()
        self.tk = self.model.tensor_kwargs
        mc = config.model.config
        self.chunk = int(mc.chunk_size)
        if int(mc.action_dim) != DIM29:
            raise SystemExit(f"{a.experiment}: action_dim {mc.action_dim}, not a Sprout config")
        self.b = SampleBuilder(a.root, self.chunk)
        if int(mc.state_t) != self.b.ds.state_t:
            raise SystemExit(f"state_t: config {mc.state_t}, dataset {self.b.ds.state_t}")
        self.shift = a.shift if a.shift is not None else float(mc.shift)
        self.guidance, self.num_steps, self.seed = a.guidance, a.num_steps, a.seed
        self.torch = torch
        print(f"[sprout] {a.experiment} chunk {self.chunk} state_t {mc.state_t} shift {self.shift} "
              f"guidance {self.guidance} steps {self.num_steps} checkpoint {a.checkpoint}", flush=True)

    def info(self) -> dict:
        return {"status": "ok", "chunk": self.chunk, "dim": DIM29, "instructions": self.b.instructions()}

    def act(self, obs: dict, options: dict | None = None) -> tuple[dict, dict]:
        """options: {"num_steps": n, "guidance": g} override the server's flags for this request (latency checks)."""
        torch = self.torch
        from torch.utils.data import default_collate

        from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent

        t = time.monotonic()
        sample = self.b.build(*parse(obs))
        batch = default_collate([sample])
        batch = {k: (v.to(**self.tk) if torch.is_tensor(v) and v.dtype.is_floating_point
                     else v.cuda() if torch.is_tensor(v) else v) for k, v in batch.items()}
        if self.seed is not None:
            torch.manual_seed(self.seed)
        with torch.no_grad(), torch.autocast("cuda", dtype=self.tk.get("dtype", torch.bfloat16)):
            _, x0, _ = self.model.get_data_and_condition(batch)
            a_idx = int(sample["action_latent_idx"])
            lat = self.model.generate_samples_from_batch(
                batch, guidance=float((options or {}).get("guidance") or self.guidance), state_shape=x0.shape[1:], n_sample=x0.shape[0],
                num_steps=int((options or {}).get("num_steps") or self.num_steps), shift=self.shift)
            a = extract_action_chunk_from_latent(lat[:, :, a_idx], (self.chunk, DIM29),
                                                 decoder=getattr(self.model, "action_decoder", None))
        act = self.b.unnormalize(a[0].float().cpu().numpy())
        return split(act), {"ms": (time.monotonic() - t) * 1000}

    def reset(self) -> dict:
        return {"status": "ok"}


def serve(p, host: str, port: int) -> None:
    """p: info() -> dict, act(observation) -> (reply, info), reset() -> dict."""
    import msgpack
    import msgpack_numpy
    import zmq

    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[sprout] policy server ready on tcp://{host}:{port}", flush=True)
    while True:
        msg = msgpack.unpackb(sock.recv(), object_hook=msgpack_numpy.decode, raw=False)
        try:
            ep = msg.get("endpoint", "get_action")
            if ep == "ping":
                reply = p.info()
            elif ep == "reset":
                reply = p.reset()
            elif ep == "get_action":
                reply = list(p.act(msg["data"]["observation"], msg["data"].get("options")))
            else:
                reply = {"error": f"unknown endpoint {ep!r}"}
        except Exception as e:  # the client stops the robot on an error reply
            import traceback

            traceback.print_exc()
            reply = {"error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(reply, default=msgpack_numpy.encode))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, help="cosmos_predict2p5_2b_sprout_3task_pv0_c{32,16}[_within|_base]")
    ap.add_argument("--checkpoint", help=".pt (convert_dcp_to_pt.py), e.g. sprout_pv0_c32/model_000004000.pt")
    ap.add_argument("--config", default="cosmos_policy/config/config.py")
    ap.add_argument("--root", default=SPROUT_ROOT, help="the training store (manifest + h5)")
    ap.add_argument("--num-steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--shift", type=float, default=None, help="default: the training value")
    ap.add_argument("--seed", type=int, default=None, help="fixed sampling noise per request (offline checks)")
    ap.add_argument("--sdpa", choices=("math", "auto"), default="math",
                    help="math: PyTorch's reference attention kernel only, as tabletop_sim_eval_server (always available); "
                         "auto: let PyTorch pick flash / memory-efficient kernels (faster where they exist)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--selftest", action="store_true", help="no GPU: the sample builder against training samples")
    a = ap.parse_args()
    m = re.search(r"_pv0_c(\d+)", a.experiment)
    if not m:
        ap.error("--experiment must be a ..._sprout_3task_pv0_c<chunk> config")
    if a.selftest:
        selftest(SampleBuilder(a.root, int(m.group(1))))
        return
    if not a.checkpoint:
        ap.error("--checkpoint is required")
    import torch

    if a.sdpa == "math":  # as tabletop_sim_eval_server: a kernel always exists
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    p = Policy(a)
    if int(m.group(1)) != p.chunk:
        raise SystemExit(f"name says chunk {m.group(1)}, config {p.chunk}")
    serve(p, a.host, a.port)


if __name__ == "__main__":
    main()
