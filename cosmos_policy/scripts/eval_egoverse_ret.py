"""Offline eval for the EgoVerse retrieval policy.

Pulls ONE dataset sample, generates the query video + query action from the
trained checkpoint, and saves:
  * a 3-way comparison video  [retrieved demo (cond) | GT query | generated query]
  * a 3D skeleton plot (GT vs generated: head + wrists + fingers + orientation axes)

Loads model weights only (load_training_state=False → no optimizer needed), so it
runs standalone on 1 GPU. Point it at the trained job's output via
IMAGINAIRE_OUTPUT_ROOT (auto-resumes the latest checkpoint) or pass
checkpoint.load_path=...; for a quick mechanics check it falls back to the base ckpt.

Run (1 GPU) — OFFLINE EVAL ON THE HELD-OUT v9 TEST SPLIT (override manifest_dir to
egoverse_pairs_v9_test; t5/action-stats stay at the v9 train paths from the config):
  CUDA_VISIBLE_DEVICES=0 BASE_DATASETS_DIR=/mnt/ddn/dataset \
  IMAGINAIRE_OUTPUT_ROOT=/mnt/ddn/tmp/egoverse_ret_4node HF_HOME=/mnt/ddn/hf_cache \
  HF_HUB_OFFLINE=1 PYTHONPATH=/mnt/ddn/cosmos_policy \
  uv run --extra cu128 --group egoverse --python 3.10 \
  torchrun --nproc_per_node=1 --master_port=12394 \
    -m cosmos_policy.scripts.eval_egoverse_ret \
    --config=cosmos_policy/config/config.py --out_dir=/mnt/ddn/dataset/_viz_samples \
    --sample_idx=0 --num_steps=35 --guidance=1.5 -- \
    experiment=cosmos_predict2p5_2b_egoverse_ret_5s checkpoint.load_training_state=False \
    dataloader_train.dataset.manifest_dir=/mnt/ddn/dataset/egoverse_pairs_v9_test

EgoDex eval (held-out embodiment): swap the experiment + egodex-native de-norm stats
(the EgoDexRetDataset reads its own egodex_pairs_top3 table, so NO manifest_dir override):
  ... uv run … torchrun … -m cosmos_policy.scripts.eval_egoverse_ret \
    --config=cosmos_policy/config/config.py --out_dir=/mnt/ddn/tmp/egodex_eval_viz \
    --act_stats=/mnt/ddn/dataset/egodex_action57_stats.json \
    --sample_idx=0 --num_steps=35 --guidance=1.5 -- \
    experiment=cosmos_predict2p5_2b_egodex_ret_5s_eval
  (convenience: PT=<.pt> bash scripts/run_egodex_eval_viz.sh 6)
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import cv2  # noqa: E402
from einops import rearrange  # noqa: E402

from torch.utils.data import default_collate  # noqa: E402

from cosmos_policy._src.imaginaire.config import load_config  # noqa: E402
from cosmos_policy._src.imaginaire.lazy_config import instantiate  # noqa: E402
from cosmos_policy._src.imaginaire.utils import distributed  # noqa: E402
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init, model_init  # noqa: E402,F401
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint  # noqa: E402
from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent  # noqa: E402
from cosmos_policy.datasets.egoverse_ret_dataset import (  # noqa: E402
    SL_LEEF, SL_REEF, DEFAULT_ACT_STATS, CLASS_TO_COMMAND, ACTION_DIM)

NUM_DUP = 4

# ---- skeleton FK (ported from cosmos-framework tools/eval_egoverse_generate.py) ----
# action57 layout (hand_pose): head9 | Leef9 | Lhand15 | Reef9 | Rhand15.
_SEG = {"head": (slice(0, 3), slice(3, 9)),
        "L": (slice(9, 12), slice(12, 18), slice(18, 33)),
        "R": (slice(33, 36), slice(36, 42), slice(42, 57))}


def _rot6d_to_mat(d6):
    """6-D rot [col0(3), col1(3)] -> 3x3 (columns) via Gram-Schmidt."""
    c0, c1 = d6[..., :3], d6[..., 3:6]
    b0 = c0 / (np.linalg.norm(c0, axis=-1, keepdims=True) + 1e-8)
    c1p = c1 - (b0 * c1).sum(-1, keepdims=True) * b0
    b1 = c1p / (np.linalg.norm(c1p, axis=-1, keepdims=True) + 1e-8)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=-1)


def _fk_skeleton(action, lo, hi, state0):
    """De-normalize action57 (min-max) and forward-integrate the full skeleton to WORLD (cm):
    integrate head9 world pose + each wrist (head-frame) from state0, place wrist-local hand15
    fingertips, then map head-frame skeleton into world by the per-frame head pose. Returns
    per-hand wrist path P (T+1,3), orientation R (T+1,3,3), fingers F (T,5,3) + head pose."""
    denom = np.clip(hi - lo, 1e-8, None)
    a = (action[:, :57] + 1.0) / 2.0 * denom + lo
    T = a.shape[0]; M = 100.0
    hps, hrs = _SEG["head"]
    HP = np.zeros((T + 1, 3)); HR = np.zeros((T + 1, 3, 3))
    HP[0] = state0[hps] * M; HR[0] = _rot6d_to_mat(state0[hrs])
    hd = a[:, 0:9]
    for t in range(T):
        HP[t + 1] = HP[t] + HR[t] @ (hd[t, :3] * M)
        HR[t + 1] = HR[t] @ _rot6d_to_mat(hd[t, 3:9])
    out = {"head_p": HP, "head_R": HR}
    for side in ("L", "R"):
        ps, rs, hs = _SEG[side]
        P = np.zeros((T + 1, 3)); R = np.zeros((T + 1, 3, 3)); F = np.zeros((T, 5, 3))
        P[0] = state0[ps] * M; R[0] = _rot6d_to_mat(state0[rs])
        eef = a[:, ps.start:ps.start + 9]; hand = a[:, hs].reshape(T, 5, 3) * M
        for t in range(T):
            P[t + 1] = P[t] + R[t] @ (eef[t, :3] * M)
            R[t + 1] = R[t] @ _rot6d_to_mat(eef[t, 3:9])
            F[t] = P[t][None] + (R[t] @ hand[t].T).T
        out[side + "_P"] = HP + np.einsum("tij,tj->ti", HR, P)
        out[side + "_R"] = np.einsum("tij,tjk->tik", HR, R)
        out[side + "_F"] = HP[:-1, None] + np.einsum("tij,tnj->tni", HR[:-1], F)
    return out


def _load_stats(path):
    """Per-embodiment min-max stats {emb:(lo,hi), "_global":(lo,hi)}. Back-compat: an OLD flat
    {"lo":[57],"hi":[57]} file is treated as "_global" for all embodiments."""
    d = json.load(open(path))
    out = {}
    if "lo" in d and "hi" in d:                                  # OLD flat (global) format
        out["_global"] = (np.asarray(d["lo"], np.float32), np.asarray(d["hi"], np.float32))
        return out
    for k, v in d.items():
        out[k] = (np.asarray(v["lo"], np.float32), np.asarray(v["hi"], np.float32))
    if "_global" not in out and out:
        los = np.stack([v[0] for v in out.values()]); his = np.stack([v[1] for v in out.values()])
        out["_global"] = (los.min(0), his.max(0))
    return out


def _to_cuda(sample, tk):
    """Batch one sample exactly like the training DataLoader (default_collate) then to GPU."""
    batch = default_collate([sample])
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(**tk) if v.dtype.is_floating_point else v.cuda()
        else:
            out[k] = v          # strings -> list[str], etc.
    return out


def _vid_uint8(x_c_t_h_w):
    """(C,T,H,W) in [-1,1] -> (T,H,W,3) uint8."""
    v = ((x_c_t_h_w.float().clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
    return np.transpose(v, (1, 2, 3, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="/mnt/ddn/dataset/_viz_samples")
    ap.add_argument("--sample_idx", type=str, default="0", help="single idx or comma list (model loads ONCE)")
    ap.add_argument("--num_steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--fps", type=int, default=30,
                    help="report playback fps — 30 = EgoDex/egoverse native rate so a "
                         "92-frame window plays at its true ~3s (was 16 -> stretched to 5.75s)")
    ap.add_argument("--weights_pt", default=None,
                    help="load model weights from a consolidated .pt (bypasses the DCP checkpointer's "
                         "training-state resume, which fails in eval); make it via dcp_to_torch_save on the ckpt's model/ dir")
    ap.add_argument("--act_stats", default=DEFAULT_ACT_STATS,
                    help="per-emb action57 min-max stats for de-norm (egodex eval: egodex_action57_stats.json)")
    ap.add_argument("opts", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    with distributed_init():
        distributed.init()
    # Proven inference load (mirrors experiments/robot get_model): load WITHOUT FSDP-wrapping so a
    # consolidated flat .pt goes in cleanly — avoids the DCP checkpointer's training-state resume
    # (optim/scheduler=None → crash) AND the FSDP-wrap-then-load state_dict mismatch.
    _opts = [o for o in args.opts if o != "--"]
    _exp = next((o.split("=", 1)[1] for o in _opts if o.startswith("experiment=")), None)
    _exp_opts = [o for o in _opts if not o.startswith("experiment=")]
    model, config = load_model_from_checkpoint(
        experiment_name=_exp, s3_checkpoint_dir=args.weights_pt, config_file=args.config,
        enable_fsdp=False, load_ema_to_reg=False, experiment_opts=_exp_opts)
    model = model.cuda().eval()
    tk = model.tensor_kwargs

    dataset = instantiate(config.dataloader_train.dataset)
    _indices = [int(x) for x in str(args.sample_idx).split(",") if x != ""]
    import time as _time
    for _sample_idx in _indices:
      args.sample_idx = _sample_idx   # downstream tag/npz naming uses this
      _t0 = _time.time(); print(f"[hb] idx {_sample_idx}: dataset fetch...", flush=True)
      sample = dataset[args.sample_idx]
      print(f"[hb] idx {_sample_idx}: fetched {_time.time()-_t0:.1f}s -> inference...", flush=True)
      data_batch = _to_cuda(sample, tk)
      emb = sample["embodiment"]; ret_emb = sample["retrieved_embodiment"]
      R = dataset.R; nf = dataset.num_frames
      os.makedirs(args.out_dir, exist_ok=True)
      tag = f"{config.job.name}_idx{args.sample_idx}_{ret_emb}2{emb}"

      _dtype = tk.get("dtype", torch.bfloat16)
      with torch.no_grad(), torch.autocast("cuda", dtype=_dtype):
          raw_data, x0, condition = model.get_data_and_condition(data_batch)
          latents = model.generate_samples_from_batch(
              data_batch, guidance=args.guidance, state_shape=x0.shape[1:],
              n_sample=x0.shape[0], num_steps=args.num_steps,
              shift=getattr(model.config, "shift", 1.0),
          )
          a_idx = int(data_batch["action_latent_idx"][0]) if torch.is_tensor(data_batch["action_latent_idx"]) \
              else int(data_batch["action_latent_idx"])
          # 액션 latent 를 GT(검은 프레임) latent 로 교체한 뒤 영상 디코드. causal 토크나이저가
          # future 블록 앞 latent 를 컨텍스트로 쓰는데 그 자리는 이미지가 아니라 액션 latent 라
          # 그대로 두면 future 첫 ~8 프레임이 무너진다 (eval_egoverse_ret_batch.py 의 같은 주석 참조).
          _lat_v = latents.clone(); _lat_v[:, :, a_idx] = x0[:, :, a_idx].to(_lat_v.dtype)
          gen_video = model.decode(_lat_v)                 # (1,C,Tpix,H,W) in [-1,1]
          print(f"[hb] idx {args.sample_idx}: inference done {_time.time()-_t0:.1f}s", flush=True)
          gen_a_norm = extract_action_chunk_from_latent(
              latents[:, :, a_idx], (nf, ACTION_DIM), decoder=getattr(model, "action_decoder", None))[0].float().cpu().numpy()
      gt_a_norm = sample["actions"]                        # (nf,57) normalized
      mask = sample["action_mask"]                          # (57,)

      # ---- un-normalize actions (PER-EMBODIMENT: use the QUERY emb's stats) ----
      _stats = _load_stats(args.act_stats)
      lo, hi = _stats.get(str(emb), _stats["_global"])             # lo/hi flow into unnorm + _fk_skeleton
      unnorm = lambda a: (a + 1) / 2 * (hi - lo) + lo
      gt_a = unnorm(gt_a_norm) * mask; gen_a = unnorm(gen_a_norm) * mask
      # query initial ABSOLUTE state57 (anchors the skeleton forward-integration)
      _q = int(dataset.query_idx[args.sample_idx % dataset.N])
      state0 = np.asarray(dataset._grp(_q).state("state57")[int(dataset.ww0[_q])], np.float32)
      np.savez(os.path.join(args.out_dir, f"eval_{config.job.name}_idx{args.sample_idx}_actions.npz"),
               gt=gt_a, gen=gen_a, gt_norm=gt_a_norm, gen_norm=gen_a_norm, mask=mask, state0=state0)
      # ---- full 57-D skeleton FK (GT + gen), world cm — for the report's two 3D panels ----
      St = _fk_skeleton(gt_a_norm, lo, hi, state0)
      Sg = _fk_skeleton(gen_a_norm, lo, hi, state0)
      _allp = np.concatenate([S[k] for S in (St, Sg) for k in ("L_P", "R_P", "head_p")]
                             + [S[k].reshape(-1, 3) for S in (St, Sg) for k in ("L_F", "R_F")], axis=0)
      _ctr = (_allp.min(0) + _allp.max(0)) / 2.0
      _rad = max(float((_allp.max(0) - _allp.min(0)).max()) / 2.0, 3.0) * 1.1

      # ---- task/command label ----
      q = int(dataset.query_idx[args.sample_idx % dataset.N])
      cls = dataset.wcls[q]
      command = f"{CLASS_TO_COMMAND.get(cls, CLASS_TO_COMMAND['other'])} [{emb}]"
      errL = float(np.abs((gt_a - gen_a)[:, SL_LEEF]).mean())
      errR = float(np.abs((gt_a - gen_a)[:, SL_REEF]).mean())

      # ---- video panels ----
      cond_px = slice(1, 4 * R + 1)                          # ret_video (latents 1..R)
      qry_px = slice(4 * R + 9, 8 * R + 9)                   # query_video (latents R+3..2R+2)
      gt_v = _vid_uint8(raw_data[0]); gen_v = _vid_uint8(gen_video[0])
      cond = gt_v[cond_px]; gt_q = gt_v[qry_px]; gen_q = gen_v[qry_px]
      T = min(len(cond), len(gt_q), len(gen_q), nf); S = gt_v.shape[1]
      font = cv2.FONT_HERSHEY_SIMPLEX

      def lab(img, txt, color):
          o = img.copy()
          cv2.putText(o, txt, (6, 20), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
          cv2.putText(o, txt, (6, 20), font, 0.55, color, 1, cv2.LINE_AA)
          return o

      Wv = 3 * S

      _HANDC = {"L": "tab:blue", "R": "tab:orange"}; _AXC = ("r", "g", "b")

      def _draw_pose(ax, p, Rm, length=4.0):                     # RGB orientation axes at a joint
          for k in range(3):
              v = Rm[:, k] * length
              ax.plot([p[0], p[0]+v[0]], [p[1], p[1]+v[1]], [p[2], p[2]+v[2]], _AXC[k], lw=1.5)

      def render3d(t):
          # TWO 3D panels = GT skeleton | GEN skeleton (full 57-D: head + both wrists w/ RGB
          # orientation axes + 5 wrist-local fingers + growing trajectory), world cm.
          Tskel = St["L_P"].shape[0]; ff = min(t, Tskel - 1); fk = min(t, St["L_F"].shape[0] - 1)
          f = plt.figure(figsize=(Wv / 100, 3.7))
          for col, (name, Sc) in enumerate([("GT skeleton", St), ("GEN skeleton", Sg)]):
              ax = f.add_subplot(1, 2, col + 1, projection="3d")
              hp, hR = Sc["head_p"][ff], Sc["head_R"][ff]
              ax.plot(*Sc["head_p"][:ff+1].T, "-", color="k", lw=1.0, alpha=0.55)   # head trajectory
              ax.scatter(*hp, color="k", s=26); _draw_pose(ax, hp, hR)
              for side in ("L", "R"):
                  P, Rr, Fg = Sc[side+"_P"], Sc[side+"_R"], Sc[side+"_F"]; c = _HANDC[side]
                  ax.plot(*P[:ff+1].T, "-", color=c, lw=1.4, alpha=0.85)            # wrist trajectory
                  ax.scatter(*P[ff], color=c, s=34); _draw_pose(ax, P[ff], Rr[ff])
                  tips = Fg[fk]                                                     # (5,3) fingertips
                  ax.scatter(tips[:, 0], tips[:, 1], tips[:, 2], color=c, s=12)
                  for pt in tips:
                      ax.plot([P[ff][0], pt[0]], [P[ff][1], pt[1]], [P[ff][2], pt[2]], color=c, lw=0.6, alpha=0.6)
              ax.set_xlim(_ctr[0]-_rad, _ctr[0]+_rad); ax.set_ylim(_ctr[1]-_rad, _ctr[1]+_rad)
              ax.set_zlim(_ctr[2]-_rad, _ctr[2]+_rad)
              ax.set_title(f"{name} [{emb}]  head=blk L=blu R=org", fontsize=8)
          f.tight_layout(pad=0.4); f.canvas.draw()
          arr = np.asarray(f.canvas.buffer_rgba())[:, :, :3].copy(); plt.close(f)
          return cv2.resize(arr, (Wv, int(arr.shape[0] * Wv / arr.shape[1])))

      # header bar
      HB = 70

      def header(t):
          h = np.full((HB, Wv, 3), 28, np.uint8)
          cv2.putText(h, f"task: {command}", (8, 24), font, 0.62, (0, 255, 255), 1, cv2.LINE_AA)
          cv2.putText(h, f"retrieval: [{ret_emb}]  ->  query: [{emb}]   sample#{args.sample_idx}   "
                         f"frame {t+1}/{T}", (8, 46), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
          cv2.putText(h, f"action |delta| GT-vs-GEN  L={errL:.4f}  R={errR:.4f}  (m/step)",
                      (8, 64), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
          return h

      rep_fp = os.path.join(args.out_dir, f"eval_{tag}_report.mp4")
      w = imageio.get_writer(rep_fp, fps=args.fps, macro_block_size=16, codec="libx264", quality=9)
      for t in range(T):
          vrow = np.concatenate([
              lab(cond[t], f"COND  retrieved demo [{ret_emb}]", (0, 255, 255)),
              lab(gt_q[t], f"GT  query [{emb}]", (255, 255, 255)),
              lab(gen_q[t], f"GENERATED  query [{emb}]", (0, 255, 0)),
          ], axis=1)
          frame = np.concatenate([header(t), vrow, render3d(t)], axis=0)
          if frame.shape[1] % 2: frame = frame[:, :-1]
          if frame.shape[0] % 2: frame = frame[:-1, :]
          w.append_data(frame)
      w.close()

      print(f"[eval] emb={emb} ret={ret_emb} task='{command}' action |Δ| L={errL:.4f} R={errR:.4f}")
      print(f"[eval] saved:\n  {rep_fp}")


if __name__ == "__main__":
    main()
