"""HaWoR hand pose estimation over whole EgoVerse episodes, as one worker of an elastic job.

Any number of workers on any number of nodes can run at the same time (4, 8 or 16 nodes; nodes may join or
leave mid-run). They share one work queue on the shared filesystem:
  * an episode is done when <OUT>/<ep>.npz exists;
  * a worker claims an episode by atomically creating the directory <OUT>/_claims/<ep> (os.mkdir); it touches
    the claim every segment, and a claim older than --stale seconds (dead worker) can be taken over;
  * a failed episode gets <OUT>/_failed/<ep>.txt and is skipped (delete the file to retry it).
So there is no rank/world size: start or stop workers freely, and re-running the launcher resumes.

Per episode: frames of images.front_1 (every --stride-th frame) are written as JPEGs to a node-local temp dir in
segments of --seg frames, then HaWoR's camera-space stage runs on each segment (WiLoR detection + tracking,
left/right tracklets, HaWoR inference per chunk, MANO forward -> 21 joints in the camera frame, OpenPose order:
0 wrist, 4/8/12/16/20 tips). SLAM / infiller are not run: WATCH's state is in the head frame and the camera is
fixed to the head, so the per-source camera -> head transform fitted in fit_cam.py replaces SLAM.
Same model logic as responses/yw5w/scripts/hawor_run_cam.py (the 1,618-window RGB-pairing check).

Output <OUT>/<ep>.npz: frames (n,) int32 = source frame indices, joints_L / joints_R (n,21,3) float32 NaN where
the hand has no estimate, trans_L / trans_R (n,3), F0 (nominal focal), H, W, stride.

usage (HaWoR venv, cwd = HaWoR repo root, watch repo on PYTHONPATH):
  python hawor_worker.py --episodes <list.txt> --video-root $EGOVERSE_HUM_ROOT --out $HAWOR_OUT [--stride 1]
"""
import argparse, os, shutil, socket, sys, tempfile, time, traceback

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from lib.pipeline.tools import detect_track, parse_chunks                         # noqa: E402
from hawor.utils.process import run_mano, run_mano_left                           # noqa: E402
from hawor.utils.rotation import rotation_matrix_to_angle_axis                    # noqa: E402
from lib.models.hawor import HAWOR                                                # noqa: E402
from cosmos_policy.datasets.egoverse_zread import EgoZarrEpisode                  # noqa: E402

F0 = 600.0


def load_hawor(checkpoint_path):
    from pathlib import Path
    from hawor.configs import get_config
    cfg = get_config(str(Path(checkpoint_path).parent.parent / "model_config.yaml"), update_cachedir=True)
    if (cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in cfg.MODEL):
        cfg.defrost(); cfg.MODEL.BBOX_SHAPE = [192, 256]; cfg.freeze()
    import collections, typing
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode
    allow = [DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode, typing.Any, dict, list, int, collections.defaultdict]
    with torch.serialization.safe_globals(allow):          # torch>=2.6 loads weights_only
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = HAWOR(cfg=cfg); model.load_state_dict(ck["state_dict"], strict=False)
    return model.cuda().eval()


def run_segment(model, imgfiles, W, H):
    """HaWoR camera-space joints for one list of consecutive frames (logic of hawor_run_cam.py)."""
    T = len(imgfiles); imgfiles = np.array(imgfiles)
    out = {f"joints_{s}": np.full((T, 21, 3), np.nan, np.float32) for s in "LR"}
    out.update({f"trans_{s}": np.full((T, 3), np.nan, np.float32) for s in "LR"})
    _, tracks = detect_track(imgfiles, thresh=0.2)
    tracks = tracks.item() if hasattr(tracks, "item") else tracks
    left, right = [], []
    for trk in tracks.values():
        is_r = np.concatenate([t["det_handedness"] for t in trk])
        (right if is_r.mean() >= 0.5 else left).extend(trk)
    for side, trk in (("L", left), ("R", right)):
        if len(trk) < 2: continue
        best = {}
        for t in trk:                                        # one detection per frame: the most confident
            if t["frame"] not in best or t["det_box"][0, 4] > best[t["frame"]]["det_box"][0, 4]: best[t["frame"]] = t
        trk = [best[k] for k in sorted(best)]
        boxes = np.concatenate([t["det_box"] for t in trk]); frame = np.array([t["frame"] for t in trk])
        for fck, bck in zip(*parse_chunks(frame, boxes, min_len=1)):
            flip = side == "L"
            res = model.inference(imgfiles[fck], bck, img_focal=F0, img_center=[W / 2, H / 2], do_flip=flip)
            root = rotation_matrix_to_angle_axis(res["pred_rotmat"][None, :, 0])
            pose = rotation_matrix_to_angle_axis(res["pred_rotmat"][None, :, 1:])
            if flip:
                root[..., 1] *= -1; root[..., 2] *= -1; pose[..., 1] *= -1; pose[..., 2] *= -1
            trans = res["pred_trans"][None, :, 0]
            o = (run_mano_left if flip else run_mano)(trans, root, pose, betas=res["pred_shape"][None, :])
            out[f"joints_{side}"][fck] = o["joints"][0].detach().cpu().numpy()
            out[f"trans_{side}"][fck] = trans[0].detach().cpu().numpy()
    return out


def claim(a, ep):
    c = f"{a.out}/_claims/{ep}"
    try:
        os.mkdir(c)
    except FileExistsError:
        if time.time() - os.path.getmtime(c) < a.stale: return None
        try:                                                 # stale claim of a dead worker: take it over
            os.rmdir(c); os.mkdir(c)
        except OSError:
            return None
    open(f"{c}/owner", "w").write(f"{socket.gethostname()} pid {os.getpid()} gpu {os.environ.get('CUDA_VISIBLE_DEVICES')}\n")
    return c


def process(a, model, ep, cpath, tmp):
    g = EgoZarrEpisode(f"{a.video_root}/{ep}")
    n = g.n_frames; frames = np.arange(0, n, a.stride, dtype=np.int32)
    if a.max_frames: frames = frames[:a.max_frames]
    import cv2
    first = cv2.imdecode(np.frombuffer(g.frame(int(frames[0])), np.uint8), cv2.IMREAD_COLOR); H, W = first.shape[:2]
    acc = {k: [] for k in ("joints_L", "joints_R", "trans_L", "trans_R")}
    for s0 in range(0, len(frames), a.seg):
        seg = frames[s0:s0 + a.seg]; d = tempfile.mkdtemp(dir=tmp); files = []
        for k, fi in enumerate(seg):
            p = f"{d}/{k:05d}.jpg"; b = g.frame(int(fi))
            if not b: b = g.frame(int(seg[k - 1])) if k else g.frame(int(fi))   # forward-fill an empty frame
            open(p, "wb").write(b); files.append(p)
        r = run_segment(model, files, W, H)
        for k in acc: acc[k].append(r[k])
        shutil.rmtree(d, ignore_errors=True); os.utime(cpath)                 # heartbeat
    out = {k: np.concatenate(v) for k, v in acc.items()}
    tmpf = f"{a.out}/.{ep}.{os.getpid()}.npz"
    np.savez(tmpf, frames=frames, F0=F0, H=H, W=W, stride=a.stride, **out)
    os.replace(tmpf, f"{a.out}/{ep}.npz")                                    # atomic: done marker
    return {s: float(np.isfinite(out[f"joints_{s}"][:, 0, 0]).mean()) for s in "LR"}, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True, help="text file, one episode id per line (shared order)")
    ap.add_argument("--video-root", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="./weights/hawor/checkpoints/hawor.ckpt")
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--seg", type=int, default=600)
    ap.add_argument("--stale", type=float, default=3600, help="seconds without heartbeat before a claim is reclaimed")
    ap.add_argument("--max-frames", type=int, default=0, help="smoke test: only the first N (strided) frames")
    ap.add_argument("--tmp", default=os.environ.get("TMPDIR", "/tmp"))
    a = ap.parse_args()
    for d in ("_claims", "_failed", "_logs"): os.makedirs(f"{a.out}/{d}", exist_ok=True)
    eps = [l.strip() for l in open(a.episodes) if l.strip()]
    rot = (hash(socket.gethostname()) + os.getpid()) % len(eps)            # start points spread over workers
    order = eps[rot:] + eps[:rot]
    model = load_hawor(a.ckpt); tag = f"{socket.gethostname()}:{os.environ.get('CUDA_VISIBLE_DEVICES')}:{os.getpid()}"
    print(f"[{tag}] model loaded, {len(eps)} episodes in the list", flush=True)
    tmp = tempfile.mkdtemp(prefix="hawor_", dir=a.tmp); done_here = 0
    for ep in order:
        if os.path.exists(f"{a.out}/{ep}.npz") or os.path.exists(f"{a.out}/_failed/{ep}.txt"): continue
        c = claim(a, ep)
        if c is None: continue
        if os.path.exists(f"{a.out}/{ep}.npz"):                             # finished between check and claim
            shutil.rmtree(c, ignore_errors=True); continue
        t0 = time.time()
        try:
            cov, nf = process(a, model, ep, c, tmp); done_here += 1
            print(f"[{tag}] {ep} {nf} frames {time.time() - t0:.0f}s ({(time.time() - t0) / max(nf, 1):.3f}s/f) "
                  f"cov L {cov['L']:.2f} R {cov['R']:.2f}", flush=True)
        except Exception:                                                    # noqa: BLE001
            open(f"{a.out}/_failed/{ep}.txt", "w").write(f"{tag}\n{traceback.format_exc()}")
            print(f"[{tag}] {ep} FAILED (see _failed/{ep}.txt)", flush=True)
            torch.cuda.empty_cache()
        shutil.rmtree(c, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[{tag}] queue empty, {done_here} episodes done by this worker", flush=True)


if __name__ == "__main__":
    main()
