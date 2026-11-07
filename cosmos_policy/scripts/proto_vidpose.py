"""PROTOTYPE — validate (1) image similarity (LPIPS/SSIM/PSNR) gen-vs-GT query video and
(2) MediaPipe hand detection on the egocentric frames (go/no-go for a pose-action metric).
Decodes a few v10test samples with a trained ckpt, runs the metrics, and saves annotated
frames (mediapipe landmarks drawn) for eyeballing. Run like run_v10test_eval_viz.sh (torchrun 1 GPU)."""
import os, sys, argparse, numpy as np, torch, cv2

sys.path.insert(0, "/mnt/ddn/cosmos_policy")
from cosmos_policy.scripts.eval_egoverse_ret import (
    _fk_skeleton, _vid_uint8, _load_stats, _to_cuda, ACTION_DIM,
)
from cosmos_policy._src.imaginaire.lazy_config import instantiate
from cosmos_policy._src.imaginaire.utils import distributed
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint
from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent

import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
from skimage.metrics import structural_similarity as ssim_fn, peak_signal_noise_ratio as psnr_fn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

_MPMODEL = "/mnt/ddn/ckpt/mediapipe/hand_landmarker.task"


_MP_CONF = 0.1        # low thresholds — egocentric + generated frames are hard; recall > precision
_MP_UPSCALE = 3       # upscale small (224) frames before detection (MP likes ≥ ~400px)


def _mp_landmarker(video_mode=True):
    """VIDEO running mode = temporal tracking (keeps a hand across frames once found), GPU delegate,
    low confidence thresholds. Falls back to CPU / IMAGE mode if unavailable."""
    mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
    for dele in (mpp.BaseOptions.Delegate.GPU, mpp.BaseOptions.Delegate.CPU):
        try:
            base = mpp.BaseOptions(model_asset_path=_MPMODEL, delegate=dele)
            opts = vision.HandLandmarkerOptions(
                base_options=base, num_hands=2, running_mode=mode,
                min_hand_detection_confidence=_MP_CONF, min_hand_presence_confidence=_MP_CONF,
                min_tracking_confidence=_MP_CONF)
            return vision.HandLandmarker.create_from_options(opts)
        except Exception:
            continue
    raise RuntimeError("could not create HandLandmarker")


def _detect(lm, frame_rgb, t=None):
    """Upscale then detect. VIDEO mode if t given. Returns LIST of wrist (x,y) per detected hand
    (up to 2), IGNORING MediaPipe's Left/Right label — it flips frame-to-frame on egocentric
    views, so we track hands by spatial continuity instead (see _track2)."""
    f = frame_rgb
    if _MP_UPSCALE > 1:
        f = cv2.resize(frame_rgb, (frame_rgb.shape[1] * _MP_UPSCALE, frame_rgb.shape[0] * _MP_UPSCALE),
                       interpolation=cv2.INTER_CUBIC)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(f))
    r = lm.detect_for_video(img, int(t * 33)) if t is not None else lm.detect(img)
    return [np.array([lms[0].x, lms[0].y], np.float32) for lms in r.hand_landmarks]   # landmark 0 = wrist


def _track2(det_list, T, gate=0.15):
    """Assign detected wrists to 2 continuity tracks by nearest-neighbor to the last known
    position (label-free). Returns (2,T,2) with NaN where a track is unassigned that frame."""
    tr = np.full((2, T, 2), np.nan, np.float32)
    last = [None, None]
    for t in range(T):
        dets = list(det_list[t])
        used_tr = set()
        # greedily match each track to its nearest detection within gate
        pairs = []
        for ti in range(2):
            if last[ti] is None:
                continue
            for di, d in enumerate(dets):
                pairs.append((np.linalg.norm(d - last[ti]), ti, di))
        pairs.sort()
        used_det = set()
        for dist, ti, di in pairs:
            if ti in used_tr or di in used_det or dist > gate:
                continue
            tr[ti, t] = dets[di]; last[ti] = dets[di]; used_tr.add(ti); used_det.add(di)
        # unmatched detections seed empty tracks (spatially: keep left-most in slot 0)
        for di, d in enumerate(dets):
            if di in used_det:
                continue
            free = [ti for ti in range(2) if ti not in used_tr]
            if not free:
                break
            ti = free[0] if last[free[0]] is None else min(free, key=lambda k: (last[k] is not None, d[0]))
            tr[ti, t] = d; last[ti] = d; used_tr.add(ti); used_det.add(di)
    return tr


def _headframe_wrists(fk):
    """FK world wrists → HEAD-frame (camera-ish) per-hand (T+1,3): R_head^T @ (P - head_p)."""
    HP, HR = fk["head_p"], fk["head_R"]; out = {}
    for s in ("L", "R"):
        out[s] = np.einsum("tji,tj->ti", HR, fk[s + "_P"] - HP)   # HR^T @ (P-HP)
    return out


def _traj_from_det(det_list, T, label):
    """Per-frame wrist (x,y) image trajectory for a handedness label; NaN where undetected."""
    P = np.full((T, 2), np.nan, np.float32)
    for t, d in enumerate(det_list):
        if label in d:
            P[t] = d[label][0, :2]            # landmark 0 = wrist (x,y normalized)
    return P


def _speed(P):
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    return d


def _valid_corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) < 1e-6 or np.std(b[m]) < 1e-6:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _pose_action_consistency(det_gen, gen_a_norm, lo, hi, state0, T):
    """Consistency of GEN video hand motion (MediaPipe) vs GEN action (FK wrist).
    Wrist SPEED correlation (axis/scale-invariant) per hand, handedness matched by best
    L/R assignment. Returns dict with per-hand speed-corr, coverage, matched assignment."""
    fk = _fk_skeleton(gen_a_norm, lo, hi, state0)
    hw = _headframe_wrists(fk)                                   # head-frame wrist (T+1,3)
    fk_speed = {s: _speed(hw[s][:T + 1])[:T - 1] for s in ("L", "R")}   # (T-1,)
    # mediapipe image-space wrist speed per handedness label
    mp_traj = {lab: _traj_from_det(det_gen, T, lab) for lab in ("Left", "Right")}
    mp_speed = {lab: _speed(mp_traj[lab])[:T - 1] for lab in ("Left", "Right")}
    cov = {lab: float(np.isfinite(mp_traj[lab][:, 0]).mean()) for lab in ("Left", "Right")}
    # try both handedness assignments (egocentric mirror ambiguity), take the one with higher mean corr
    best = None
    for assign in [{"Left": "L", "Right": "R"}, {"Left": "R", "Right": "L"}]:
        cs = [_valid_corr(mp_speed[lab], fk_speed[assign[lab]]) for lab in ("Left", "Right")]
        cs = [c for c in cs if np.isfinite(c)]
        score = np.mean(cs) if cs else -np.inf
        if best is None or score > best[0]:
            best = (score, assign, cs)
    return dict(speed_corr=best[0] if np.isfinite(best[0]) else np.nan,
                assign=best[1], n_hands_scored=len(best[2]),
                cov_L=cov["Left"], cov_R=cov["Right"])


def _norm01(a):
    a = np.asarray(a, np.float32); m = np.isfinite(a)
    if m.sum() < 2: return a
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / max(hi - lo, 1e-6)


def _debug_video(gen_q, det_gen, gen_a_norm, lo, hi, state0, pac, T, path, fps=30):
    """Per-frame [gen frame + MP wrist trail | MP-speed vs FK-speed curves] mp4 (30fps=3s)."""
    import imageio.v2 as imageio
    fk = _fk_skeleton(gen_a_norm, lo, hi, state0); hw = _headframe_wrists(fk)
    assign = pac["assign"]                                   # {"Left":"L"/"R", ...}
    mp_tr = {lab: _traj_from_det(det_gen, T, lab) for lab in ("Left", "Right")}
    mp_sp = {lab: _norm01(_speed(mp_tr[lab])[:T - 1]) for lab in ("Left", "Right")}
    fk_sp = {s: _norm01(_speed(hw[s][:T + 1])[:T - 1]) for s in ("L", "R")}
    Hc, Wc = gen_q.shape[1], 260                            # curve panel size
    trail = {lab: [] for lab in ("Left", "Right")}
    frames = []
    for t in range(T):
        # panel A: gen frame + MP wrist dots/trail
        A = cv2.cvtColor(gen_q[t], cv2.COLOR_RGB2BGR).copy(); H, W = A.shape[:2]
        for lab, col in (("Right", (0, 255, 0)), ("Left", (0, 160, 255))):
            w = mp_tr[lab][t]
            if np.isfinite(w[0]):
                trail[lab].append((int(w[0] * W), int(w[1] * H)))
            for k in range(1, len(trail[lab])):
                cv2.line(A, trail[lab][k - 1], trail[lab][k], col, 1)
            if np.isfinite(w[0]):
                cv2.circle(A, (int(w[0] * W), int(w[1] * H)), 5, (0, 0, 255), -1)
                cv2.circle(A, (int(w[0] * W), int(w[1] * H)), 5, col, 2)
        # panel C: speed curves (MP=solid, FK=dashed) for the matched hands + current line
        C = np.full((Hc, Wc, 3), 30, np.uint8)
        cv2.putText(C, f"speed_corr {pac['speed_corr']:+.2f}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
        def plot(curve, color, y0, dash=False):
            n = len(curve); pts = []
            for i in range(n):
                if not np.isfinite(curve[i]): continue
                x = 6 + int(i / max(n - 1, 1) * (Wc - 12)); y = y0 - int(curve[i] * 60)
                pts.append((x, y))
            for k in range(1, len(pts)):
                if dash and k % 2 == 0: continue
                cv2.line(C, pts[k - 1], pts[k], color, 1)
        for lab, yc, cc in (("Right", 90, (0, 255, 0)), ("Left", 165, (0, 160, 255))):
            cv2.putText(C, f"{lab}->{assign[lab]}", (6, yc - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.35, cc, 1)
            plot(mp_sp[lab], cc, yc, dash=False)             # MP speed (solid)
            plot(fk_sp[assign[lab]], (200, 200, 200), yc, dash=True)  # FK action speed (dashed grey)
        xc = 6 + int(min(t, T - 2) / max(T - 2, 1) * (Wc - 12))
        cv2.line(C, (xc, 24), (xc, Hc - 6), (80, 80, 80), 1)
        frames.append(np.concatenate([A, cv2.resize(C, (Wc, H))], 1)[:, :, ::-1])   # BGR->RGB
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights_pt", required=True)
    ap.add_argument("--act_stats", required=True)
    ap.add_argument("--out_dir", default="/mnt/ddn/tmp/proto_vidpose")
    ap.add_argument("--idxs", default="0 2000 4000 8000")
    ap.add_argument("--num_steps", type=int, default=35)
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    with distributed_init():
        distributed.init()
    _opts = [o for o in a.opts if o != "--"]
    _exp = next((o.split("=", 1)[1] for o in _opts if o.startswith("experiment=")), None)
    _exp_opts = [o for o in _opts if not o.startswith("experiment=")]
    model, config = load_model_from_checkpoint(
        experiment_name=_exp, s3_checkpoint_dir=a.weights_pt, config_file=a.config,
        enable_fsdp=False, load_ema_to_reg=False, experiment_opts=_exp_opts)
    model = model.cuda().eval(); tk = model.tensor_kwargs
    dataset = instantiate(config.dataloader_train.dataset)
    R = dataset.R; nf = dataset.num_frames
    stats = _load_stats(a.act_stats)
    lpips = LPIPS(net_type="alex", normalize=True).cuda()

    qry_px = slice(4 * R + 9, 8 * R + 9)
    print(f"R={R} nf={nf} qry latent slice={qry_px}", flush=True)

    for idx in [int(x) for x in a.idxs.split()]:
        sample = dataset[idx]; data_batch = _to_cuda(sample, tk)
        emb = sample["embodiment"]; ret = sample["retrieved_embodiment"]
        _dtype = tk.get("dtype", torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=_dtype):
            raw_data, x0, _ = model.get_data_and_condition(data_batch)
            latents = model.generate_samples_from_batch(
                data_batch, guidance=1.5, state_shape=x0.shape[1:], n_sample=x0.shape[0],
                num_steps=a.num_steps, shift=getattr(model.config, "shift", 1.0))
            gen_video = model.decode(latents)
            a_idx = int(data_batch["action_latent_idx"][0]) if torch.is_tensor(data_batch["action_latent_idx"]) else int(data_batch["action_latent_idx"])
            gen_a_norm = extract_action_chunk_from_latent(latents[:, :, a_idx], (nf, ACTION_DIM), decoder=getattr(model, "action_decoder", None))[0].float().cpu().numpy()
        gt_a_norm = sample["actions"]
        gt_v = _vid_uint8(raw_data[0]); gen_v = _vid_uint8(gen_video[0])
        gt_q = gt_v[qry_px]; gen_q = gen_v[qry_px]
        T = min(len(gt_q), len(gen_q))
        gt_q, gen_q = gt_q[:T], gen_q[:T]

        # ---- image metrics (paired gen vs GT) ----
        ssims = [ssim_fn(gt_q[t], gen_q[t], channel_axis=2, data_range=255) for t in range(T)]
        psnrs = [psnr_fn(gt_q[t], gen_q[t], data_range=255) for t in range(T)]
        g = torch.from_numpy(gt_q).permute(0, 3, 1, 2).float().cuda() / 255.0
        p = torch.from_numpy(gen_q).permute(0, 3, 1, 2).float().cuda() / 255.0
        lp = float(lpips(p, g).item())

        # ---- mediapipe on GEN video only (pose metric is gen-vs-gen-action) ----
        lm = _mp_landmarker(video_mode=True)                 # fresh per sample (VIDEO tracking state)
        det_gen = [ _detect(lm, gen_q[t], t) for t in range(T) ]
        lm.close()
        rate_gen = np.mean([len(d) > 0 for d in det_gen]); nhands_gen = np.mean([len(d) for d in det_gen])

        # ---- pose-action consistency (GEN video pose vs GEN action) ----
        _q = int(dataset.query_idx[idx % dataset.N])
        state0 = np.asarray(dataset._grp(_q).state("state57")[int(dataset.ww0[_q])], np.float32)
        lo, hi = stats.get(str(emb), stats["_global"])
        pac = _pose_action_consistency(det_gen, gen_a_norm, lo, hi, state0, T)

        print(f"\nidx={idx} {ret}2{emb} T={T} | SSIM={np.mean(ssims):.3f} PSNR={np.mean(psnrs):.1f} LPIPS={lp:.3f}"
              f" | MP detect GEN={rate_gen:.2f}({nhands_gen:.1f}h)"
              f" | POSE-ACTION speed_corr={pac['speed_corr']:+.3f} (cov L={pac['cov_L']:.2f} R={pac['cov_R']:.2f}, {pac['n_hands_scored']}h)", flush=True)

        # ---- save annotated frames (3 timepoints) for eyeball: GT | GEN with landmarks ----
        def annot(frame, det):
            o = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy(); H, W = o.shape[:2]
            for label, kp in det.items():
                for (x, y, _z) in kp:
                    cv2.circle(o, (int(x * W), int(y * H)), 2, (0, 255, 0) if label == "Right" else (0, 128, 255), -1)
                w = kp[0]; cv2.circle(o, (int(w[0] * W), int(w[1] * H)), 4, (0, 0, 255), -1)  # wrist
            return o
        rows = [annot(gen_q[t], det_gen[t]) for t in (int(T * 0.2), int(T * 0.5), int(T * 0.8))]
        cv2.imwrite(f"{a.out_dir}/idx{idx}_{ret}2{emb}_mp.png", np.concatenate(rows, 1))
        _debug_video(gen_q, det_gen, gen_a_norm, lo, hi, state0, pac, T,
                     f"{a.out_dir}/idx{idx}_{ret}2{emb}_debug.mp4")

    print("\nDONE ->", a.out_dir, flush=True)


if __name__ == "__main__":
    main()
