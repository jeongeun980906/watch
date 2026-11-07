"""Report-video visualization for the mid-v2 model (compact layout, roles + t_positions).

Renders EXACTLY the old eval_egoverse_ret.py report (header | [COND demo | GT query |
GEN query] | GT/GEN 3D skeleton panels) + the actions npz, with the SAME filename
convention  eval_<exp>_idx<N>_<retemb>2<emb>_report.mp4 , for a v2 checkpoint. Only the
pixel/latent geometry is v2 (per-sample k/g; query clip = pixels [13+4k, 13+4k+4g),
demo clip = pixels [5, 5+4k), action latent = 3+k, native action length 4g).

v10test (in-dist test split):
  CUDA_VISIBLE_DEVICES=0 ... torchrun --nproc_per_node=1 -m cosmos_policy.scripts.viz_v2 \
    --config=cosmos_policy/config/config.py --weights_pt=<model_NNN.pt> \
    --act_stats=/mnt/ddn/dataset/egoverse_pairs_v10/action57_stats.json \
    --state_stats=/mnt/ddn/dataset/egoverse_pairs_v10/state57_stats.json \
    --out_dir=... --sample_idx=0 -- experiment=cosmos_predict2p5_2b_egoverse_ret_v2 \
    checkpoint.load_training_state=False \
    dataloader_train.dataset.manifest_dir=/mnt/ddn/dataset/egoverse_pairs_v10_test

egodex (held-out embodiment): add --egodex and swap
  --act_stats=/mnt/ddn/dataset/egodex_action57_stats.json (no manifest/state overrides).
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402  (.venv cv2 is broken -> PIL for text/resize)

from cosmos_policy._src.imaginaire.lazy_config import instantiate  # noqa: E402
from cosmos_policy._src.imaginaire.utils import distributed  # noqa: E402
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init  # noqa: E402
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint  # noqa: E402
from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent  # noqa: E402
from cosmos_policy.scripts.eval_egoverse_ret import (  # noqa: E402  (reuse old renderer's helpers)
    _fk_skeleton, _load_stats, _to_cuda, _vid_uint8)
from cosmos_policy.datasets.state_action_convert import (  # noqa: E402  (shared with decode_dump_v2/batch eval)
    derived_action57_from_states, fk_skeleton_state as _fk_skeleton_state)
from cosmos_policy.scripts.decode_dump_v2 import sample_geometry  # noqa: E402  (also installs the no-aug patch)
from cosmos_policy.datasets.egoverse_ret_dataset import (  # noqa: E402
    SL_LEEF, SL_REEF, DEFAULT_ACT_STATS, CLASS_TO_COMMAND, ACTION_DIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="/mnt/ddn/dataset/_viz_samples")
    ap.add_argument("--sample_idx", type=str, default="0", help="single idx or comma list (model loads ONCE)")
    ap.add_argument("--num_steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--fps", type=int, default=None,
                    help="report fps; default = query-native (30 human/egodex, 20 mimicdroid/gr1), "
                         "so a native clip plays real-time (92f/30 = 3.07 s)")
    ap.add_argument("--weights_pt", default=None)
    ap.add_argument("--act_stats", default=DEFAULT_ACT_STATS)
    ap.add_argument("--egodex", action="store_true", help="EgoDexRetV2Dataset instead of the config dataset")
    ap.add_argument("--demo_emb", default=None,
                    help="override the row's locked demo embodiment (cross-emb viz)")
    ap.add_argument("--state_stats", default=None,
                    help="state57 stats json override (TEST-manifest eval must keep TRAIN normalization)")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    with distributed_init():
        distributed.init()
    _opts = [o for o in args.opts if o != "--"]
    _exp = next((o.split("=", 1)[1] for o in _opts if o.startswith("experiment=")), None)
    _exp_opts = [o for o in _opts if not o.startswith("experiment=")]
    model, config = load_model_from_checkpoint(
        experiment_name=_exp, s3_checkpoint_dir=args.weights_pt, config_file=args.config,
        enable_fsdp=False, load_ema_to_reg=False, experiment_opts=_exp_opts)
    model = model.cuda().eval()
    tk = model.tensor_kwargs

    if args.egodex:
        from cosmos_policy.datasets.egodex_ret_v2_dataset import EgoDexRetV2Dataset
        dataset = EgoDexRetV2Dataset()
        exp_name = f"{config.job.name}_egodex"
    else:
        dataset = instantiate(config.dataloader_train.dataset)
        exp_name = config.job.name
    if args.state_stats:
        dataset.state_stats = dataset._load_or_compute_state_stats(args.state_stats)
    _indices = [int(x) for x in str(args.sample_idx).split(",") if x != ""]
    import time as _time
    for _sample_idx in _indices:   # model loads ONCE; per-idx flow identical to a single run
      args.sample_idx = _sample_idx          # downstream naming/seed/indexing all read this
      _t0 = _time.time(); print(f"[hb] idx {_sample_idx}: dataset fetch...", flush=True)
      if args.demo_emb:                      # cross-emb: re-lock the row's demo embodiment
          dataset._m_emb[args.sample_idx % dataset.N] = args.demo_emb
      np.random.seed((1000003 * (args.sample_idx + 1)) % (2**31 - 1))   # same demo pick as decode_dump_v2
      sample = dataset[args.sample_idx]
      data_batch = _to_cuda(sample, tk)
      emb = sample["embodiment"]; ret_emb = sample["retrieved_embodiment"]
      k, g, qry_px, a_idx, nf = sample_geometry(sample)
      os.makedirs(args.out_dir, exist_ok=True)
      tag = f"{exp_name}_idx{args.sample_idx}_{ret_emb}2{emb}"

      _dtype = tk.get("dtype", torch.bfloat16)
      with torch.no_grad(), torch.autocast("cuda", dtype=_dtype):
          raw_data, x0, condition = model.get_data_and_condition(data_batch)
          latents = model.generate_samples_from_batch(
              data_batch, guidance=args.guidance, state_shape=x0.shape[1:],
              n_sample=x0.shape[0], num_steps=args.num_steps,
              shift=getattr(model.config, "shift", 1.0),
          )
          gen_video = model.decode(latents)                # (1,C,Tpix,H,W) in [-1,1]
          print(f"[hb] idx {_sample_idx}: inference done {_time.time()-_t0:.1f}s", flush=True)
          gen_a_norm = extract_action_chunk_from_latent(
              latents[:, :, a_idx], (nf, ACTION_DIM), decoder=getattr(model, "action_decoder", None))[0].float().cpu().numpy()
      gt_a_norm = sample["actions"]                        # (nf,57) normalized
      mask = sample["action_mask"]                          # (57,)

      # ---- un-normalize actions (PER-EMBODIMENT: use the QUERY emb's stats) ----
      _stats = _load_stats(args.act_stats)
      lo, hi = _stats.get(str(emb), _stats["_global"])
      unnorm = lambda a: (a + 1) / 2 * (hi - lo) + lo
      gt_a = unnorm(gt_a_norm) * mask; gen_a = unnorm(gen_a_norm) * mask
      _q = int(dataset.query_idx[args.sample_idx % dataset.N])
      state0 = np.asarray(dataset._grp(_q).state("state57")[int(dataset.ww0[_q])], np.float32)
      # STATE-chunk experiments (name contains "_state"): the chunks are ABSOLUTE state57
      # (and --act_stats must be the state stats). Keep the npz keys gt/gen = DERIVED delta
      # action57 (exact integrator inverse, anchored at state0) so downstream action metrics
      # read the same convention as delta runs; the states go into gt_state/gen_state.
      _state_mode = "_state" in exp_name
      _extra = {}
      if _state_mode:
          _extra = dict(gt_state=gt_a, gen_state=gen_a)
          gt_a = derived_action57_from_states(gt_a, state0).astype(np.float32)
          gen_a = derived_action57_from_states(gen_a, state0).astype(np.float32)
      np.savez(os.path.join(args.out_dir, f"eval_{exp_name}_idx{args.sample_idx}_actions.npz"),
               gt=gt_a, gen=gen_a, gt_norm=gt_a_norm, gen_norm=gen_a_norm, mask=mask, state0=state0,
               **_extra)
      # ---- full 57-D skeleton FK (GT + gen), world cm — for the report's two 3D panels ----
      # STATE-chunk runs: frames are ABSOLUTE state57 — compose the skeleton per frame
      # directly instead of delta-integrating.
      _fk = _fk_skeleton_state if _state_mode else _fk_skeleton
      St = _fk(gt_a_norm, lo, hi, state0)
      Sg = _fk(gen_a_norm, lo, hi, state0)
      _allp = np.concatenate([S[kk] for S in (St, Sg) for kk in ("L_P", "R_P", "head_p")]
                             + [S[kk].reshape(-1, 3) for S in (St, Sg) for kk in ("L_F", "R_F")], axis=0)
      _ctr = (_allp.min(0) + _allp.max(0)) / 2.0
      _rad = max(float((_allp.max(0) - _allp.min(0)).max()) / 2.0, 3.0) * 1.1

      # ---- task/command label ----
      cls = dataset.wcls[_q]
      command = f"{CLASS_TO_COMMAND.get(cls, CLASS_TO_COMMAND['other'])} [{emb}]"
      # header err: state runs report |abs state| error (matching the "state |abs|" label)
      _egt, _egen = (_extra["gt_state"], _extra["gen_state"]) if _state_mode else (gt_a, gen_a)
      errL = float(np.abs((_egt - _egen)[:, SL_LEEF]).mean())
      errR = float(np.abs((_egt - _egen)[:, SL_REEF]).mean())

      # ---- video panels (v2 pixel packing [1 | 4 | 4k | 4 | 4 | 4g]) ----
      cond_px = slice(5, 5 + 4 * k)                          # ret demo pixels
      gt_v = _vid_uint8(raw_data[0]); gen_v = _vid_uint8(gen_video[0])

      def _is_grid_emb(e):        # 3-cam embodiments use the full 2x2 grid canvas
          e = str(e).lower()
          return ("mimicdroid" in e) or ("gr1" in e)

      # v3 datasets pack the primary cam FULL-FRAME for every embodiment (no 2x2 grid),
      # so the TL-quadrant crop below must be skipped entirely.
      from cosmos_policy.datasets.egoverse_ret_v3_dataset import EgoVerseRetV3Dataset
      _full_frame = isinstance(dataset, EgoVerseRetV3Dataset)

      def _panel_clip(clip, e):
          """Single-cam embodiments (aria/scale/MECKA/egodex) fill only the TL quadrant of the
          2x2 grid canvas -> crop TL and upscale to panel size so panels look full-frame."""
          if _full_frame or _is_grid_emb(e):
              return clip
          Sq = clip.shape[1]; h = Sq // 2
          return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(f)).resize((Sq, Sq), Image.BILINEAR))
                           for f in clip[:, :h, :h]])

      cond = _panel_clip(gt_v[cond_px], ret_emb)
      gt_q = _panel_clip(gt_v[qry_px], emb)
      gen_q = _panel_clip(gen_v[qry_px], emb)
      # full query length; cond (4k frames) freezes on its last frame if k < g (cross bucket)
      T = min(len(gt_q), len(gen_q)); S = gt_v.shape[1]
      _FONT = ImageFont.truetype(fm.findfont("DejaVu Sans"), 12)

      def _text(im, xy, txt, color, outline=True):
          d = ImageDraw.Draw(im)
          if outline:
              for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, -1), (0, 1), (-1, 0), (1, 0)):
                  d.text((xy[0] + dx, xy[1] + dy), txt, fill=(0, 0, 0), font=_FONT)
          d.text(xy, txt, fill=tuple(color), font=_FONT)

      def lab(img, txt, color):
          im = Image.fromarray(np.ascontiguousarray(img))
          _text(im, (6, 8), txt, color)
          return np.asarray(im)

      Wv = 3 * S

      _HANDC = {"L": "tab:blue", "R": "tab:orange"}; _AXC = ("r", "g", "b")

      def _draw_pose(ax, p, Rm, length=4.0):                     # RGB orientation axes at a joint
          for kx in range(3):
              v = Rm[:, kx] * length
              ax.plot([p[0], p[0]+v[0]], [p[1], p[1]+v[1]], [p[2], p[2]+v[2]], _AXC[kx], lw=1.5)

      def render3d(t):
          Tskel = St["L_P"].shape[0]; ff = min(t, Tskel - 1); fk = min(t, St["L_F"].shape[0] - 1)
          f = plt.figure(figsize=(Wv / 100, 3.7))
          for col, (name, Sc) in enumerate([("GT skeleton", St), ("GEN skeleton", Sg)]):
              ax = f.add_subplot(1, 2, col + 1, projection="3d")
              hp, hR = Sc["head_p"][ff], Sc["head_R"][ff]
              ax.plot(*Sc["head_p"][:ff+1].T, "-", color="k", lw=1.0, alpha=0.55)
              ax.scatter(*hp, color="k", s=26); _draw_pose(ax, hp, hR)
              for side in ("L", "R"):
                  P, Rr, Fg = Sc[side+"_P"], Sc[side+"_R"], Sc[side+"_F"]; c = _HANDC[side]
                  ax.plot(*P[:ff+1].T, "-", color=c, lw=1.4, alpha=0.85)
                  ax.scatter(*P[ff], color=c, s=34); _draw_pose(ax, P[ff], Rr[ff])
                  tips = Fg[fk]
                  ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], color=c, s=12)
                  for pt in tips:
                      ax.plot([P[ff][0], pt[0]], [P[ff][1], pt[1]], [P[ff][2], pt[2]], color=c, lw=0.6, alpha=0.6)
              ax.set_xlim(_ctr[0]-_rad, _ctr[0]+_rad); ax.set_ylim(_ctr[1]-_rad, _ctr[1]+_rad)
              ax.set_zlim(_ctr[2]-_rad, _ctr[2]+_rad)
              ax.set_title(f"{name} [{emb}]  head=blk L=blu R=org", fontsize=8)
          f.tight_layout(pad=0.4); f.canvas.draw()
          arr = np.asarray(f.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(f)
          return np.asarray(Image.fromarray(arr).resize(
              (Wv, int(arr.shape[0] * Wv / arr.shape[1])), Image.BILINEAR))

      HB = 70

      def header(t):
          im = Image.fromarray(np.full((HB, Wv, 3), 28, np.uint8))
          _text(im, (8, 6), f"task: {command}", (0, 255, 255), outline=False)
          _text(im, (8, 30), f"retrieval: [{ret_emb}]  ->  query: [{emb}]   sample#{args.sample_idx}   "
                             f"frame {t+1}/{T}  (v2 k={k} g={g})", (255, 255, 255), outline=False)
          _lbl = "state |abs|" if _state_mode else "action |delta|"
          _unit = "m" if _state_mode else "m/step"
          _text(im, (8, 50), f"{_lbl} GT-vs-GEN  L={errL:.4f}  R={errR:.4f}  ({_unit})",
                (0, 255, 0), outline=False)
          return np.asarray(im)

      rep_fp = os.path.join(args.out_dir, f"eval_{tag}_report.mp4")
      fps = args.fps if args.fps else (20 if _is_grid_emb(emb) else 30)   # query-native playback
      w = imageio.get_writer(rep_fp, fps=fps, macro_block_size=16, codec="libx264", quality=9)
      for t in range(T):
          vrow = np.concatenate([
              lab(cond[min(t, len(cond) - 1)], f"COND  retrieved demo [{ret_emb}]", (0, 255, 255)),
              lab(gt_q[t], f"GT  query [{emb}]", (255, 255, 255)),
              lab(gen_q[t], f"GENERATED  query [{emb}]", (0, 255, 0)),
          ], axis=1)
          frame = np.concatenate([header(t), vrow, render3d(t)], axis=0)
          if frame.shape[1] % 2: frame = frame[:, :-1]
          if frame.shape[0] % 2: frame = frame[:-1, :]
          w.append_data(frame)
      w.close()

      print(f"[viz_v2] emb={emb} ret={ret_emb} k={k} g={g} task='{command}' action |D| L={errL:.4f} R={errR:.4f}")
      print(f"[viz_v2] saved:\n  {rep_fp}")


if __name__ == "__main__":
    main()
