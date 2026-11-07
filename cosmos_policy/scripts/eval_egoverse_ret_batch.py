"""Batch offline eval for the EgoVerse retrieval policy — MOTION-SHAPE metrics (not just MSE)
over the whole v9 TEST split. Loads the model ONCE, iterates one query per held-out episode
(~150), generates the action57 chunk (metric-only: video decode skipped), and scores gen-vs-GT
end-effector motion SHAPE. Metrics split by scale-sensitivity — for an UNSEEN embodiment
(EgoDex) the model may emit a different ABSOLUTE motion scale than GT, which must NOT count
as a shape error, so the PRIMARY metrics are SCALE-INVARIANT:
  * dtw_shape — DTW after re-scaling each trajectory to unit total path length (pure shape)
  * cos       — per-step Δpos direction cosine (magnitude-free)
  * corr      — Pearson corr of the integrated xyz trajectory (affine-invariant)
SCALE-DEPENDENT (reference, trust only in-distribution): dtw_norm, mse; plus scale_ratio
(gen/gt |path|) as a DIAGNOSTIC of how far the absolute magnitude drifts. Per hand (L/R) +
head, aggregated to mean/median.

Writes eval_batch_<jobname>_metrics.csv (per sample) + eval_batch_<jobname>_summary.json.

VIDEO METRICS (env VIDEO_METRICS=1, default 0 = legacy metric-only run): each scored pair
ALSO decodes gen+GT pixel video of the query clip and adds img_l1/img_psnr/img_ssim CSV
columns + episode-macro summary entries (same/cross split), and every WILOR_EVERY-th
(default 4) query window dumps a stage2-compatible npz into <out_dir>/wilor_dumps/ for
the WiLoR+FID stage2 (tools/wilor_metric.py, run by run_batch_8gpu.sh after merge).

Run (1 GPU) on the v9 TEST split:
  CUDA_VISIBLE_DEVICES=0 BASE_DATASETS_DIR=/mnt/ddn/dataset HF_HOME=/mnt/ddn/hf_cache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/mnt/ddn/cosmos_policy \
  uv run --extra cu128 --group egoverse --python 3.10 \
  torchrun --nproc_per_node=1 --master_port=12610 \
    -m cosmos_policy.scripts.eval_egoverse_ret_batch \
    --config=cosmos_policy/config/config.py --out_dir=/mnt/ddn/dataset/_viz_v9_test \
    --weights_pt=/mnt/ddn/tmp/egoverse_ret_iter18000_model.pt --episodes 0 -- \
    experiment=cosmos_predict2p5_2b_egoverse_ret_5s \
    dataloader_train.dataset.manifest_dir=/mnt/ddn/dataset/egoverse_pairs_v9_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cosmos_policy._src.imaginaire.lazy_config import instantiate  # noqa: E402
from cosmos_policy._src.imaginaire.utils import distributed  # noqa: E402
from cosmos_policy._src.imaginaire.utils.context_managers import distributed_init  # noqa: E402
from cosmos_policy._src.predict2.utils.model_loader import load_model_from_checkpoint  # noqa: E402
from cosmos_policy.models.policy_text2world_model import extract_action_chunk_from_latent  # noqa: E402
from cosmos_policy.datasets.egoverse_ret_dataset import DEFAULT_ACT_STATS, ACTION_DIM  # noqa: E402
from cosmos_policy.datasets.egoverse_ret_v3_state_dataset import load_stats_json  # noqa: E402
from cosmos_policy.datasets.state_action_convert import (  # noqa: E402
    derived_action57_from_states, fk_skeleton_state as _fk_skeleton_state)
from cosmos_policy.scripts.eval_egoverse_ret import _to_cuda, _vid_uint8, _fk_skeleton  # noqa: E402

# action57 Δpos blocks: head=0:3, L-eef=9:12, R-eef=33:36
_BLOCKS = {"head": slice(0, 3), "L": slice(9, 12), "R": slice(33, 36)}
_EEF_DIMS = np.r_[9:12, 33:36]          # wrist-position dims (interpretable subset)
_TAUS = (0.5, 0.1)                      # accuracy thresholds in NORMALIZED action units


def _norm_space_metrics(gt_n: np.ndarray, gen_n: np.ndarray, mask: np.ndarray) -> dict:
    """MSE / L1 / accuracy@tau on the model's OWN [-1,1] action space.

    This is the convention used to compare policies across papers: errors averaged over
    action dimensions AND the chunk horizon, plus accuracy@tau = the fraction of action
    dimensions within tau of GT (tau=0.5 ~ coarse motion intent, tau=0.1 ~ precision).
    Unlike the physical-unit metrics it is scale-free per dimension, so position, rotation
    and finger dims contribute comparably and a magnitude blow-up in one block cannot
    swamp the rest.

    Only dims the sample actually carries are scored (``action_mask``) — eva rows have
    hand dims masked out, and counting those as exact matches would inflate accuracy.
    """
    keep = np.asarray(mask, np.float32) > 0
    if not keep.any():
        return {}
    d = np.abs(np.asarray(gen_n, np.float64)[:, keep] - np.asarray(gt_n, np.float64)[:, keep])
    out = {"msen_all": float((d ** 2).mean()), "l1n_all": float(d.mean())}
    for t in _TAUS:
        out[f"acc{t}"] = float((d <= t).mean())
    ek = [i for i in _EEF_DIMS if keep[i]]
    if ek:
        de = np.abs(np.asarray(gen_n, np.float64)[:, ek] - np.asarray(gt_n, np.float64)[:, ek])
        out.update({"msen_eef": float((de ** 2).mean()), "l1n_eef": float(de.mean())})
        for t in _TAUS:
            out[f"acc{t}_eef"] = float((de <= t).mean())
    return out

# ---------------------------------------------------------------------------
# OPTIONAL video-quality metrics (env VIDEO_METRICS=1; default OFF = exact legacy
# behavior/cost). When ON, each scored (query, demo) pair ALSO decodes the GEN and
# GT pixel video of the query clip and reports per-sample L1/PSNR/SSIM columns
# (img_l1/img_psnr/img_ssim) in the CSV + episode-macro summary, and every
# WILOR_EVERY-th query window dumps a stage2-compatible npz (decode_dump contract:
# gen/gt JPEG bytes, fkL/fkR head-frame wrists, emb/ret/idx) into
# <out_dir>/wilor_dumps/ for the WiLoR+FID stage2 (tools/wilor_metric.py).
# ---------------------------------------------------------------------------
VIDEO_METRIC_COLS = ["img_l1", "img_psnr", "img_ssim"]
# normalized-space columns emitted by _norm_space_metrics (all rows, no env flag)
_NORM_COLS = (["msen_all", "l1n_all"] + [f"acc{t}" for t in (0.5, 0.1)]
              + ["msen_eef", "l1n_eef"] + [f"acc{t}_eef" for t in (0.5, 0.1)])


def _ssim_np(a: np.ndarray, b: np.ndarray, win: int = 8) -> float:
    """Small windowed SSIM fallback (uint8 HxWx3, grayscale, non-overlapping win x win
    blocks) — only used if skimage is unavailable in the eval env."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    x = a.mean(2).astype(np.float64); y = b.mean(2).astype(np.float64)
    H, W = x.shape; H8, W8 = H - H % win, W - W % win
    if H8 < win or W8 < win:
        return float("nan")
    x = x[:H8, :W8].reshape(H8 // win, win, W8 // win, win).transpose(0, 2, 1, 3).reshape(-1, win * win)
    y = y[:H8, :W8].reshape(H8 // win, win, W8 // win, win).transpose(0, 2, 1, 3).reshape(-1, win * win)
    mx, my = x.mean(1), y.mean(1)
    vx, vy = x.var(1), y.var(1)
    cxy = ((x - mx[:, None]) * (y - my[:, None])).mean(1)
    s = ((2 * mx * my + C1) * (2 * cxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (vx + vy + C2))
    return float(s.mean())


try:
    from skimage.metrics import structural_similarity as _sk_ssim  # noqa: E402
    def _ssim_frame(a, b): return float(_sk_ssim(a, b, channel_axis=2, data_range=255))
except ImportError:
    _ssim_frame = _ssim_np


def _frame_metrics(gt: np.ndarray, gen: np.ndarray) -> dict:
    """gt/gen: (T,H,W,3) uint8 query clips -> per-sample video-quality scalars
    (frame-averaged). L1 is on the [0,1] scale; PSNR/SSIM use data_range=255
    (same conventions as the stage2 wilor_metric.py image metrics)."""
    T = min(len(gt), len(gen))
    gt = gt[:T].astype(np.float64); gen_f = gen[:T].astype(np.float64)
    l1 = float(np.abs(gt - gen_f).mean() / 255.0)
    mse_t = ((gt - gen_f) ** 2).reshape(T, -1).mean(1)
    psnr = float(np.mean(20.0 * np.log10(255.0 / np.sqrt(np.maximum(mse_t, 1e-10)))))
    ssim = float(np.mean([_ssim_frame(gt[t].astype(np.uint8), gen[t]) for t in range(T)]))
    return {"img_l1": l1, "img_psnr": psnr, "img_ssim": ssim}


def _tl_crop(v: np.ndarray) -> np.ndarray:
    """2x2-grid samples: single-cam embodiments carry content only in the TL quadrant —
    crop + resize back so metrics/WiLoR exclude the black quadrants (decode_dump_v2)."""
    import torch.nn.functional as _F
    h, w = v.shape[1] // 2, v.shape[2] // 2
    x = torch.from_numpy(v[:, :h, :w].copy()).permute(0, 3, 1, 2).float()
    x = _F.interpolate(x, size=(v.shape[1], v.shape[2]), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).clamp(0, 255).byte().numpy()


def _banded_dtw(a: np.ndarray, b: np.ndarray, band: int = 8) -> float:
    """Sakoe-Chiba banded DTW between two 3-D point sequences (Euclidean local cost)."""
    n, m = len(a), len(b)
    INF = 1e18
    D = np.full((n + 1, m + 1), INF, np.float64)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        jlo, jhi = max(1, i - band), min(m, i + band)
        ai = a[i - 1]
        for j in range(jlo, jhi + 1):
            c = float(np.linalg.norm(ai - b[j - 1]))
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def _shape_metrics(dgt: np.ndarray, dgen: np.ndarray, band: int = 8) -> dict:
    """dgt/dgen: (T,3) per-step Δpos of one eef/head. Returns shape-similarity scalars.

    SCALE-INVARIANT (use these for UNSEEN embodiments like EgoDex, where the model may
    emit a different ABSOLUTE motion scale than GT — a scale mismatch must NOT be counted
    as a shape error):
    - dtw_shape: DTW on each trajectory RE-SCALED to unit total path length (pure shape,
                 fully scale-invariant; lower=more similar). PRIMARY for egodex.
    - cos:       mean per-step cos(Δgt, Δgen) over non-zero steps (direction only; higher→1)
    - corr:      mean Pearson corr over xyz of the integrated trajectory (affine-invariant)

    SCALE-DEPENDENT (only meaningful when absolute scale is trusted, e.g. in-distribution):
    - dtw_norm:  path-normalized DTW (penalizes scale mismatch — NOT for egodex)
    - mse:       per-step Δpos MSE (absolute; reference only)
    - scale_ratio: DIAGNOSTIC, gen/gt total path length (1.0=same scale; reports HOW
                 different the absolute magnitude is, so a scale gap is visible not hidden)
    """
    T = len(dgt)
    band = max(band, T // 4)                                  # TIMING-robust: warp up to ±25% of length
    tgt = np.cumsum(dgt, 0)
    tgen = np.cumsum(dgen, 0)
    plen_gt = float(np.linalg.norm(dgt, axis=1).sum())
    plen_gen = float(np.linalg.norm(dgen, axis=1).sum())
    dtw = _banded_dtw(tgt, tgen, band=band)
    dtw_norm = dtw / (plen_gt + plen_gen + 1e-6)
    # scale-invariant DTW: normalize each trajectory to unit total path length first, so
    # only the SHAPE (relative bends/turns) is compared, not the absolute magnitude.
    sgt = np.cumsum(dgt / (plen_gt + 1e-9), 0)
    sgen = np.cumsum(dgen / (plen_gen + 1e-9), 0)
    dtw_shape = _banded_dtw(sgt, sgen, band=band)
    scale_ratio = float(plen_gen / (plen_gt + 1e-9))
    ng = np.linalg.norm(dgt, axis=1)
    nn = np.linalg.norm(dgen, axis=1)
    msk = (ng > 1e-6) & (nn > 1e-6)
    cos = float(((dgt[msk] * dgen[msk]).sum(1) / (ng[msk] * nn[msk])).mean()) if msk.any() else float("nan")
    # TIMING-robust cos: slide gen ±L over GT, take the BEST-aligned per-step direction cos
    # (handles a constant time offset — right motion, just started early/late).
    L = max(1, T // 4)
    best = -2.0
    for lag in range(-L, L + 1):
        a, b = (dgt[lag:], dgen[:T - lag]) if lag >= 0 else (dgt[:T + lag], dgen[-lag:])
        na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
        m = (na > 1e-6) & (nb > 1e-6)
        if m.sum() >= 3:
            best = max(best, float(((a[m] * b[m]).sum(1) / (na[m] * nb[m])).mean()))
    cos_lag = best if best > -2 else float("nan")
    # spectral (|FFT|) cosine — compares motion frequency content, PHASE(=timing)-invariant.
    sc = []
    for d in range(3):
        fa = np.abs(np.fft.rfft(dgt[:, d])); fb = np.abs(np.fft.rfft(dgen[:, d]))
        na_, nb_ = np.linalg.norm(fa), np.linalg.norm(fb)
        if na_ > 1e-9 and nb_ > 1e-9:
            sc.append(float((fa * fb).sum() / (na_ * nb_)))
    spec_cos = float(np.mean(sc)) if sc else float("nan")
    corrs = []
    for d in range(3):
        x, y = tgt[:, d], tgen[:, d]
        if x.std() > 1e-9 and y.std() > 1e-9:
            corrs.append(float(np.corrcoef(x, y)[0, 1]))
    corr = float(np.mean(corrs)) if corrs else float("nan")
    mse = float(((dgt - dgen) ** 2).mean())
    return {"dtw_shape": dtw_shape, "cos": cos, "cos_lag": cos_lag, "spec_cos": spec_cos, "corr": corr,
            "dtw_norm": dtw_norm, "mse": mse, "scale_ratio": scale_ratio}


def _agg(vals):
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], np.float64)
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "n": int(a.size)}


def _agg_by_episode(rows, col, kind=None):
    """EPISODE-LEVEL macro-average (2 levels):
      1. per episode: mean over ALL its evals (every window × every demo, equal weight)
         → one score per episode
      2. overall: mean/median over the per-episode scores (each episode weighted equally)
    `kind` filters rows to "same"/"cross" for a split view. MUST run on the full merged row
    set (an episode's evals can be split across shards)."""
    from collections import defaultdict
    by_ep = defaultdict(list)                                    # episode -> [all its eval vals]
    for r in rows:
        if kind is not None and r.get("kind") != kind:
            continue
        try:
            v = float(r.get(col))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            by_ep[r.get("episode", "?")].append(v)
    ep_means = np.asarray([float(np.mean(vs)) for vs in by_ep.values() if vs], np.float64)
    if ep_means.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "n_ep": 0, "n_eval": 0}
    return {"mean": float(ep_means.mean()), "median": float(np.median(ep_means)),
            "n_ep": int(ep_means.size), "n_eval": int(sum(len(v) for v in by_ep.values()))}


def _load_stats(path):
    """Load per-embodiment min-max stats: {emb: (lo,hi), "_global": (lo,hi)}.
    Back-compat: an OLD flat {"lo":[57],"hi":[57]} file is treated as "_global" for all embs."""
    d = json.load(open(path))
    out = {}
    if "lo" in d and "hi" in d:                                  # OLD flat (global) format
        out["_global"] = (np.asarray(d["lo"], np.float32), np.asarray(d["hi"], np.float32))
        return out
    for k, v in d.items():
        out[k] = (np.asarray(v["lo"], np.float32), np.asarray(v["hi"], np.float32))
    if "_global" not in out and out:                            # synthesize a fallback if absent
        los = np.stack([v[0] for v in out.values()]); his = np.stack([v[1] for v in out.values()])
        out["_global"] = (los.min(0), his.max(0))
    return out


def _denorm(a_norm, emb, stats):
    """De-normalize action57 using THIS sample's embodiment stats (fallback _global)."""
    lo, hi = stats.get(str(emb), stats["_global"])
    return (a_norm + 1.0) / 2.0 * (hi - lo) + lo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cosmos_policy/config/config.py")
    ap.add_argument("--out_dir", default="/mnt/ddn/tmp/egoverse_v9_eval")
    ap.add_argument("--weights_pt", required=True)
    ap.add_argument("--num_steps", type=int, default=35)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--episodes", type=int, default=0, help="max held-out episodes (0=all ~150)")
    ap.add_argument("--per-episode", dest="per_episode", type=int, default=0,
                    help="windows/episode: 0=ALL (every ~1s window, whole episode); K>0=K strided")
    ap.add_argument("--topk", type=int, default=3,
                    help="demos/query: same-top-k + cross-top-k (default 3+3); 0=ALL stored matches")
    ap.add_argument("--abc", action="store_true",
                    help="A/B/C cell mode (v10 test): score each query against THREE demos — "
                         "A = best match, SAME emb; B = best match, DIFFERENT emb (same motion "
                         "reference); C = SAME emb, DIFFERENT task, random non-match. Cells come "
                         "straight from the manifest's pairs.npz (match_idx/match_dist); `kind` "
                         "becomes A/B/C in the CSV and the summary splits on it. Overrides --topk.")
    ap.add_argument("--all-rows", dest="all_rows", action="store_true",
                    help="iterate EVERY query row (~13k, slow) instead of one-per-episode")
    ap.add_argument("--dtw-band", dest="dtw_band", type=int, default=8)
    ap.add_argument("--subset", default=None,
                    help="183_fixed_eval_subset.py 가 만든 eval_fixed200.json. 쿼리 행과 데모를 "
                         "파일이 정하므로 --per-episode/--episodes/--topk 는 무시된다. "
                         "cosmos3 edge 와 **같은 윈도우·같은 데모**로 평가하기 위한 것 "
                         "(EVAL_PROTOCOL.md 1절). HE/EgoDex 비교는 반드시 이걸 쓴다.")
    ap.add_argument("--shard", type=int, default=0, help="this shard index (0..num_shards-1)")
    ap.add_argument("--num-shards", dest="num_shards", type=int, default=1,
                    help="split samples across N GPUs (strided). Each shard writes its own "
                         "_metrics_shard{i}of{N}.csv; merge with merge_batch_metrics.py.")
    ap.add_argument("--act_stats", default=DEFAULT_ACT_STATS,
                    help="per-emb action57 min-max stats for de-norm (egodex eval: egodex_action57_stats.json)")
    ap.add_argument("--state_stats", default=None,
                    help="STATE-chunk runs (experiment contains '_state'): state57 stats json that "
                         "OVERRIDES the dataset's (a TEST manifest recomputes its own robust stats, "
                         "but eval must keep the TRAIN normalization — e.g. "
                         "egoverse_pairs_v10/state57_stats_staterobust.json for the mid eval)")
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
    _dtype = tk.get("dtype", torch.bfloat16)

    dataset = instantiate(config.dataloader_train.dataset)
    N = dataset.N
    nf = dataset.num_frames

    # ---- A/B/C cell mode ----------------------------------------------------
    # Motion similarity is NOT redefined here: A and B are the manifest's own matches
    # (pairs.npz match_idx, ranked by match_dist), so A and B have the SAME motion
    # reference and differ ONLY in the demo's embodiment. C keeps the embodiment and
    # breaks the motion instead (different task, random, never a stored match).
    KINDS = ("same", "cross")
    if args.abc:
        KINDS = ("A", "B", "C")
        _rng = np.random.default_rng(0)
        _qi, _mi = dataset.query_idx, dataset.match_idx
        _emb, _cls, _ep = dataset.wemb, dataset.wcls, dataset.wep
        _md = np.load(os.path.join(dataset.manifest_dir, "pairs.npz"),
                      allow_pickle=True)["match_dist"].astype(np.float32)
        _by_emb = {e: np.where(_emb == e)[0] for e in sorted(set(_emb.tolist()))}

        def _abc_demos(r: int, topk: int = 1):
            q = int(_qi[r])
            ms = [(int(m), float(_md[r, k])) for k, m in enumerate(_mi[r]) if m >= 0]
            same = [t for t in ms if _emb[t[0]] == _emb[q]]
            diff = [t for t in ms if _emb[t[0]] != _emb[q]]
            out = []
            if same:
                out.append((min(same, key=lambda t: t[1])[0], "A"))
            if diff:
                out.append((min(diff, key=lambda t: t[1])[0], "B"))
            mset = {m for m, _ in ms}
            pool = _by_emb[str(_emb[q])]                 # C: same emb, DIFFERENT task, random
            for _ in range(200):
                w = int(pool[_rng.integers(len(pool))])
                if w not in mset and _cls[w] != _cls[q] and _ep[w] != _ep[q]:
                    out.append((w, "C"))
                    break
            return q, (out or [(q, "self")])

        dataset.eval_demos = _abc_demos
        print(f"[batch] A/B/C mode: A=match/same-emb, B=match/CROSS-emb, "
              f"C=same-emb/DIFFERENT-task random non-match (pool by emb: "
              f"{ {k: len(v) for k, v in _by_emb.items()} })", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)

    stats = _load_stats(args.act_stats)                          # per-embodiment min-max

    # ---- optional video metrics (env; default OFF = legacy behavior/cost) ----
    video_metrics = os.environ.get("VIDEO_METRICS", "0") == "1"
    wilor_every = max(1, int(os.environ.get("WILOR_EVERY", "4")))
    wilor_dir = os.path.join(args.out_dir, "wilor_dumps")
    _crop_tl_ds = False                                          # 2x2-grid dataset (v2, NOT v3 full-frame)?
    if video_metrics:
        os.makedirs(wilor_dir, exist_ok=True)
        try:
            from cosmos_policy.datasets.egoverse_ret_v2_dataset import EgoVerseRetV2Dataset as _V2
            from cosmos_policy.datasets.egoverse_ret_v3_dataset import EgoVerseRetV3Dataset as _V3
            _crop_tl_ds = isinstance(dataset, _V2) and not isinstance(dataset, _V3)
        except ImportError:
            pass
        print(f"[batch] VIDEO_METRICS=1: decoding gen+GT query clips (L1/PSNR/SSIM per sample); "
              f"stage2 npz dump every {wilor_every} windows -> {wilor_dir} (grid_crop={_crop_tl_ds})", flush=True)

    # ---- STATE-chunk mode (experiments whose name contains "_state") ----
    # Those models predict ABSOLUTE state57 chunks; samples carry state chunks in the
    # "actions" keys. De-normalize gen AND GT with the (train) STATE stats and convert
    # both to delta action57 via the EXACT integrator inverse, so every metric below is
    # computed on the SAME quantities as the delta-action rounds (directly comparable).
    state_mode = "_state" in config.job.name
    sstats = None
    if state_mode:
        if args.state_stats:                     # keep TRAIN normalization on TEST manifests
            dataset.state_stats = load_stats_json(args.state_stats)
        sstats = {k: (np.asarray(v["lo"], np.float32), np.asarray(v["hi"], np.float32))
                  for k, v in dataset.state_stats.items()}
        print(f"[batch] STATE mode: metrics on DERIVED delta action57; state stats = "
              f"{args.state_stats or '<dataset robust stats>'}", flush=True)

    # ---- sample selection ----
    # An episode holds MANY query windows on a ~1 s stride (v9 test: windows every 30 frames
    # = 1 s; median ~75 windows/ep). DEFAULT = eval EVERY window (the whole episode at data
    # resolution, "one per second"). `--per-episode K` (K>0) sub-samples K evenly-strided
    # windows/episode instead; `--per-episode 0` or `--all-rows` = all windows.
    ep_of_row = dataset.wep[dataset.query_idx]                 # (N,) episode per query row
    fixed_demo = {}
    if args.subset:
        # 고정 부분집합: 쿼리 윈도우 id -> 행으로 되돌리고, 데모도 파일이 정한 것 하나만 쓴다.
        # eval_demos() 의 top-k 대신 단일 데모라 edge 와 조건이 정확히 일치한다.
        _sub = json.load(open(args.subset))
        # 윈도우 번호는 파이프라인마다 다르다 (cosmos_policy EgoDex 는 egodex_pairs_top3_3s.npz,
        # edge 는 egodex_pairs_c3). 그래서 **(에피소드, 시작프레임)** 으로 매칭한다 —
        # 이건 테이블이 달라도 같은 물리적 윈도우를 가리킨다. 쿼리 id 는 보조 수단.
        _wep, _w0 = np.asarray(dataset.wep).astype(str), np.asarray(dataset.ww0)
        _key2w = {(_wep[i], int(_w0[i])): i for i in range(len(_wep))}
        _w2r = {int(q): r for r, q in enumerate(dataset.query_idx)}
        rows = []
        _miss = 0
        for _ep, _s0, _qi, _di in zip(_sub["ep"], _sub["w0"], _sub["query"], _sub["demo"]):
            _w = _key2w.get((str(_ep), int(_s0)))
            _r = _w2r.get(_w if _w is not None else int(_qi))
            if _r is None:
                _miss += 1
                continue
            rows.append(_r); fixed_demo[_r] = int(_di)
        print(f"[batch] 고정 부분집합 {args.subset}: {len(_sub['query'])}개 중 "
              f"{len(rows)}개 매칭 (누락 {_miss})", flush=True)
        if _miss:
            # 조용히 넘어가면 두 모델이 다른 표본을 평가하게 된다 — 그게 원래 문제였다.
            raise SystemExit(
                f"ERROR: 고정 부분집합 {_miss}개가 이 데이터셋의 윈도우 테이블에 없다. "
                f"subset 은 (episode, w0) 로 매칭한다 — pairs 테이블이 다르면 "
                f"(예: EgoDex 는 cosmos_policy=egodex_pairs_top3_3s.npz vs edge=egodex_pairs_c3) "
                f"양쪽 테이블 모두에 존재하는 윈도우로 subset 을 다시 만들어야 한다.")
    elif args.all_rows or int(args.per_episode) <= 0:
        rows = list(range(N))                                  # every ~1 s window, whole episodes
    else:
        from collections import OrderedDict
        by_ep: "OrderedDict[str, list]" = OrderedDict()
        for r in range(N):
            by_ep.setdefault(str(ep_of_row[r]), []).append(r)
        K = max(1, int(args.per_episode))
        rows = []
        for ridx in by_ep.values():                            # ridx = this episode's windows (time-ordered)
            if K >= len(ridx):
                rows.extend(ridx)
            else:
                st = len(ridx) / float(K)
                rows.extend(ridx[int(j * st)] for j in range(K))
    if not args.subset and args.episodes and args.episodes > 0 and args.episodes < len(rows):
        # EVEN stride cap over the selected rows (egodex eps are task-sorted, so rows[:N]
        # would be one task; striding spreads the cap across the whole range).
        step = len(rows) / float(args.episodes)
        rows = [rows[int(i * step)] for i in range(args.episodes)]
    _total = len(rows)
    if args.num_shards > 1:                                     # strided sharding across GPUs
        rows = rows[args.shard::args.num_shards]
    print(f"[batch] N_rows={N} unique_eps={len(set(ep_of_row))} total_samples={_total} "
          f"shard={args.shard}/{args.num_shards} -> this shard evaluates {len(rows)} "
          f"(all_rows={args.all_rows})", flush=True)

    jobname = config.job.name
    per_sample = []
    run_dtw, run_cos = [], []
    t0 = time.time()

    def _score(q_win: int, m_win: int, want_video: bool = False):
        """Generate query action for the (query q_win, demo m_win) pair; return de-normalized
        (sample, gt_a, gen_a, gt_n, gen_n, vid). `vid` is None unless want_video: then it
        carries the decoded GT/GEN uint8 query clips + the raw normalized chunks/state0
        needed by the stage2 wilor dump (video failures degrade to vid=None, never kill
        the action metrics)."""
        sample = dataset.build_sample(q_win, m_win)
        # state datasets build NATIVE-length chunks (92/60/32); delta path keeps num_frames
        nf_s = int(np.asarray(sample["actions"]).shape[0]) if state_mode else nf
        data_batch = _to_cuda(sample, tk)
        vid = None
        with torch.no_grad(), torch.autocast("cuda", dtype=_dtype):
            _raw, x0, _cond = model.get_data_and_condition(data_batch)
            latents = model.generate_samples_from_batch(
                data_batch, guidance=args.guidance, state_shape=x0.shape[1:],
                n_sample=x0.shape[0], num_steps=args.num_steps,
                shift=getattr(model.config, "shift", 1.0))
            a_idx = int(data_batch["action_latent_idx"][0]) if torch.is_tensor(data_batch["action_latent_idx"]) \
                else int(data_batch["action_latent_idx"])
            gen_a_norm = extract_action_chunk_from_latent(
                latents[:, :, a_idx], (nf_s, ACTION_DIM),
                decoder=getattr(model, "action_decoder", None))[0].float().cpu().numpy()
            if want_video:
                try:
                    # query clip pixel region from the sample's own latent geometry:
                    # causal tokenizer => latent i>=1 covers pixel frames [4i-3, 4i].
                    fs = int(sample["future_video_start_latent_idx"])
                    fe = int(sample["future_video_end_latent_idx"])
                    px = slice(4 * fs - 3, 4 * fe - 3)
                    # 액션 latent 를 GT(검은 프레임) latent 로 갈아끼운 뒤 디코드한다.
                    # 토크나이저 디코더는 causal 이라 future 블록 바로 앞 latent 를 컨텍스트로
                    # 쓰는데, 그 자리(action_latent_idx)에 있는 건 이미지가 아니라 액션 청크가
                    # 실린 latent 다. 그대로 두면 future 첫 ~8 프레임이 무너진다 (실측 PSNR
                    # t0..7 이 15/24/18/17/20/21/21/22 -> 32/31/30/28/28/28/28/28, 클립 평균
                    # +0.9 dB). 레이아웃상 그 자리의 픽셀은 원래 검은 프레임이므로 x0 의 같은
                    # 슬롯을 넣는 것이 학습 때 디코더가 보던 것과 일치한다. 액션은 위에서 이미
                    # 원본 latents 로 뽑았으므로 영향 없다.
                    _lat_v = latents.clone()
                    _lat_v[:, :, a_idx] = x0[:, :, a_idx].to(_lat_v.dtype)
                    gen_v = model.decode(_lat_v)
                    gt_q = _vid_uint8(_raw[0])[px]
                    gen_q = _vid_uint8(gen_v[0])[px]
                    if _crop_tl_ds and not any(t in str(sample.get("embodiment", ""))
                                               for t in ("mimicdroid", "gr1")):
                        gt_q = _tl_crop(gt_q); gen_q = _tl_crop(gen_q)
                    vid = {"gt_q": gt_q, "gen_q": gen_q}
                except Exception as ex:  # noqa: BLE001 — video is auxiliary, keep action metrics
                    print(f"  [video-skip q={q_win}] {type(ex).__name__}: {ex}", flush=True)
                    vid = None
        gt_a_norm = np.asarray(sample["actions"], np.float32)
        mask = np.asarray(sample["action_mask"], np.float32)
        # normalized-space MSE/L1/accuracy@tau on the RAW chunks (NOT the per-step diffs the
        # state branch returns as gt_n/gen_n) — this is the model's actual prediction target.
        nrm = _norm_space_metrics(gt_a_norm, gen_a_norm, mask)
        qemb = str(sample.get("embodiment", ""))                 # per-emb de-norm by query emb
        if vid is not None:                                      # raw chunks for the stage2 FK dump
            vid["gen_a_norm"] = gen_a_norm.astype(np.float32)
            vid["gt_a_norm"] = gt_a_norm.copy()
        if state_mode:
            # chunks are normalized ABSOLUTE state57: de-norm with the TRAIN state stats,
            # then EXACT-inverse both to delta action57 anchored at the query boundary
            # state s[w0] (same state0 viz_v2 uses) — metrics below run UNCHANGED.
            state0 = np.asarray(dataset._grp(q_win).state("state57")[int(dataset.ww0[q_win])],
                                np.float64)
            gt_a = derived_action57_from_states(_denorm(gt_a_norm, qemb, sstats), state0)
            gen_a = derived_action57_from_states(_denorm(gen_a_norm, qemb, sstats), state0)
            if vid is not None:
                vid["state0"] = state0.astype(np.float32)
            # normalized-space diagnostics need per-step deltas -> diff the normalized
            # state chunks (stats-free direction/shape analogue for state runs)
            return (sample, gt_a.astype(np.float32) * mask, gen_a.astype(np.float32) * mask,
                    np.diff(gt_a_norm, axis=0) * mask, np.diff(gen_a_norm, axis=0) * mask,
                    vid, nrm)
        # return BOTH the de-normalized (physical, meters — true geometry) and the raw
        # normalized ([-1,1] — the model's training/loss space, de-norm-stats-independent)
        # (gt, gen) so shape metrics can be checked in each space.
        if vid is not None:
            try:
                vid["state0"] = np.asarray(
                    dataset._grp(q_win).state("state57")[int(dataset.ww0[q_win])], np.float32)
            except Exception:  # noqa: BLE001 — FK dump becomes best-effort without state0
                vid["state0"] = None
        return (sample, _denorm(gt_a_norm, qemb, stats) * mask, _denorm(gen_a_norm, qemb, stats) * mask,
                gt_a_norm * mask, gen_a_norm * mask, vid, nrm)

    def _dump_wilor(fp: str, vid: dict, gt_a: np.ndarray, gen_a: np.ndarray,
                    emb: str, ret: str, r: int, kind: str, q_win: int = -1, m_win: int = -1):
        """One stage2-compatible npz (decode_dump contract: gen/gt JPEG bytes, h/w,
        fkL/fkR head-frame gen wrists (+ _gt twins), emb/ret/idx) for tools/wilor_metric.py."""
        from cosmos_policy.scripts.decode_dump import jpg
        gen_q, gt_q = vid["gen_q"], vid["gt_q"]
        extra = {}
        if vid.get("state0") is not None:
            state0 = vid["state0"]
            if state_mode:
                slo, shi = sstats.get(str(emb), sstats["_global"])
                fk = _fk_skeleton_state(vid["gen_a_norm"], slo, shi, state0)
                fk_gt = _fk_skeleton_state(vid["gt_a_norm"], slo, shi, state0)
            else:
                lo, hi = stats.get(str(emb), stats["_global"])
                fk = _fk_skeleton(vid["gen_a_norm"], lo, hi, state0)
                fk_gt = _fk_skeleton(vid["gt_a_norm"], lo, hi, state0)
            for tag, f in (("", fk), ("_gt", fk_gt)):            # head-frame wrists, L+R
                HP, HR = f["head_p"], f["head_R"]
                extra[f"fkL{tag}"] = np.einsum("tji,tj->ti", HR, f["L_P"] - HP).astype(np.float32)
                extra[f"fkR{tag}"] = np.einsum("tji,tj->ti", HR, f["R_P"] - HP).astype(np.float32)
        if "fkL" not in extra:                                   # no state0 -> NaN wrists so stage2's
            nanw = np.full((len(vid["gen_a_norm"]) + 1, 3), np.nan, np.float32)  # pose_corr degrades to None
            extra = {"fkL": nanw, "fkR": nanw, "fkL_gt": nanw, "fkR_gt": nanw}
        np.savez(fp, gen=jpg(gen_q), gt=jpg(gt_q), h=gen_q.shape[1], w=gen_q.shape[2],
                 emb=str(emb), ret=str(ret), idx=int(r), kind=str(kind),
                 q_win=int(q_win), m_win=int(m_win),   # window ids -> demo clip is recoverable
                 gen_a=np.asarray(gen_a, np.float32), gt_a=np.asarray(gt_a, np.float32), **extra)

    for i, r in enumerate(rows):
        try:
            # DETERMINISTIC: score this query against its same-emb top-k + cross-emb top-k demos.
            q, demos = dataset.eval_demos(r, topk=args.topk)
            if r in fixed_demo:
                demos = [(fixed_demo[r], "fixed")]   # 고정 부분집합: 데모까지 파일이 정한다
            print(f"  [{i+1}/{len(rows)}] row={r} q={q} demos={len(demos)} ({time.time()-t0:.0f}s) ...", flush=True)
            want_dump = video_metrics and (i % wilor_every == 0)
            for di, (m_win, kind) in enumerate(demos):
                sample, gt_a, gen_a, gt_n, gen_n, vid, nrm = _score(q, m_win, want_video=video_metrics)
                # row per (query window, demo). idx=r groups a window's demos; kind = same/cross.
                row = {"idx": r, "episode": str(dataset.wep[q]), "emb": str(sample.get("embodiment", "")),
                       "ret_emb": str(sample.get("retrieved_embodiment", "")), "kind": kind,
                       "cls": str(dataset.wcls[q]), **nrm}
                if vid is not None:
                    row.update(_frame_metrics(vid["gt_q"], vid["gen_q"]))
                    if want_dump:
                        try:
                            _dump_wilor(os.path.join(wilor_dir, f"s{args.shard}_{i:06d}_{di}.npz"),
                                        vid, gt_a, gen_a, row["emb"], row["ret_emb"], r, kind,
                                        q_win=q, m_win=m_win)
                        except Exception as ex:  # noqa: BLE001 — dump is auxiliary
                            print(f"  [wilor-dump-skip idx {r}] {type(ex).__name__}: {ex}", flush=True)
                for key in ("head", "L", "R"):
                    sl = _BLOCKS[key]
                    mm = _shape_metrics(gt_a[:, sl], gen_a[:, sl], band=args.dtw_band)   # PHYSICAL (meters)
                    row[f"dtws_{key}"] = mm["dtw_shape"]; row[f"cos_{key}"] = mm["cos"]; row[f"corr_{key}"] = mm["corr"]
                    row[f"cosl_{key}"] = mm["cos_lag"]; row[f"spec_{key}"] = mm["spec_cos"]   # timing-robust
                    row[f"dtw_{key}"] = mm["dtw_norm"]; row[f"mse_{key}"] = mm["mse"]; row[f"sr_{key}"] = mm["scale_ratio"]
                    nn = _shape_metrics(gt_n[:, sl], gen_n[:, sl], band=args.dtw_band)    # NORMALIZED ([-1,1], stats-free)
                    row[f"dtwsn_{key}"] = nn["dtw_shape"]; row[f"cosn_{key}"] = nn["cos"]; row[f"corrn_{key}"] = nn["corr"]
                for m in ("dtws", "cos", "cosl", "spec", "dtw", "sr", "dtwsn", "cosn"):
                    row[f"{m}_LR"] = float(np.nanmean([row[f"{m}_L"], row[f"{m}_R"]]))
                per_sample.append(row)
                run_dtw.append(row["dtws_LR"]); run_cos.append(row["cos_LR"])
        except Exception as ex:  # noqa: BLE001 — one bad query shouldn't kill the sweep
            print(f"  [skip idx {r}] {type(ex).__name__}: {ex}", flush=True)
            continue
        if (i + 1) % 5 == 0 or (i + 1) == len(rows):
            el = time.time() - t0
            print(f"  {i+1}/{len(rows)} windows, {len(per_sample)} evals  dtws_LR={np.nanmean(run_dtw):.3f}  "
                  f"cos_LR={np.nanmean(run_cos):+.3f}  ({el:.0f}s)", flush=True)

    # ---- aggregate + write ----
    _suffix = f"_shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    csv_fp = os.path.join(args.out_dir, f"eval_batch_{jobname}_metrics{_suffix}.csv")
    cols = ["idx", "episode", "emb", "ret_emb", "kind", "cls",
            # scale-invariant (primary for unseen/egodex)
            "dtws_head", "dtws_L", "dtws_R", "dtws_LR", "cos_head", "cos_L", "cos_R", "cos_LR",
            "corr_head", "corr_L", "corr_R",
            # TIMING-robust: best-lag direction cosine + spectral(|FFT|) cosine
            "cosl_head", "cosl_L", "cosl_R", "cosl_LR", "spec_head", "spec_L", "spec_R", "spec_LR",
            # NORMALIZED-space ([-1,1], de-norm-stats-independent) shape/dir/corr
            "dtwsn_head", "dtwsn_L", "dtwsn_R", "dtwsn_LR", "cosn_head", "cosn_L", "cosn_R", "cosn_LR",
            "corrn_head", "corrn_L", "corrn_R",
            # scale-dependent reference + scale diagnostic
            "dtw_head", "dtw_L", "dtw_R", "dtw_LR", "mse_head", "mse_L", "mse_R",
            "sr_head", "sr_L", "sr_R", "sr_LR",
            # NORMALIZED-space chunk errors + accuracy@tau (cross-paper convention;
            # averaged over action dims AND horizon, masked to the dims the row carries)
            *_NORM_COLS]
    if video_metrics:                     # VIDEO_METRICS=1 only: keep the =0 CSV byte-identical
        cols += VIDEO_METRIC_COLS
    with open(csv_fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in per_sample:
            w.writerow(row)

    if args.num_shards > 1:                # sharded: write only this shard's CSV; merge aggregates
        print(f"[shard {args.shard}/{args.num_shards}] {len(per_sample)} samples -> {csv_fp}", flush=True)
        return

    summary = {"n_samples": len(per_sample), "jobname": jobname, "checkpoint": args.weights_pt,
               "num_steps": args.num_steps, "guidance": args.guidance}
    # HIERARCHICAL macro-average (demo → window → episode). Also split by demo kind
    # (same-emb vs cross-emb retrieval) for a diagnostic view.
    for metric in ("dtws", "dtw", "cos", "cosl", "spec", "corr", "mse", "sr", "dtwsn", "cosn", "corrn"):
        for key in ("head", "L", "R"):
            summary[f"{metric}_{key}"] = _agg_by_episode(per_sample, f"{metric}_{key}")
            for kd in KINDS:
                summary[f"{metric}_{key}__{kd}"] = _agg_by_episode(per_sample, f"{metric}_{key}", kind=kd)
    for m in ("dtws", "dtw", "cos", "cosl", "spec", "sr", "dtwsn", "cosn"):
        summary[f"{m}_LR"] = _agg_by_episode(per_sample, f"{m}_LR")
    for c in _NORM_COLS:                  # normalized-space errors + accuracy@tau
        summary[c] = _agg_by_episode(per_sample, c)
        for kd in KINDS:
            summary[f"{c}__{kd}"] = _agg_by_episode(per_sample, c, kind=kd)
    if video_metrics:                     # video-quality columns, same episode-macro + same/cross split
        for c in VIDEO_METRIC_COLS:
            summary[c] = _agg_by_episode(per_sample, c)
            for kd in KINDS:
                summary[f"{c}__{kd}"] = _agg_by_episode(per_sample, c, kind=kd)
    _ne = summary["dtws_L"]["n_ep"]; _nd = summary["dtws_L"]["n_eval"]
    json_fp = os.path.join(args.out_dir, f"eval_batch_{jobname}_summary.json")
    json.dump(summary, open(json_fp, "w"), indent=2)

    print("\n==== SHAPE-METRIC SUMMARY (demo→window→episode macro) ====", flush=True)
    print(f"  episodes: {_ne}  demo-evals: {_nd}  "
          f"({'A/B/C cells' if args.abc else f'same-top{args.topk} + cross-top{args.topk}'})")
    print("  [split] " + "  ".join(
        f"{kd}: cos_L={summary.get(f'cos_L__{kd}', {}).get('mean', float('nan')):+.3f} "
        f"mse_L={summary.get(f'mse_L__{kd}', {}).get('mean', float('nan')):.5f} "
        f"mse_R={summary.get(f'mse_R__{kd}', {}).get('mean', float('nan')):.5f} "
        f"(n_ep={summary.get(f'cos_L__{kd}', {}).get('n_ep', 0)})" for kd in KINDS))
    print("  -- SCALE-INVARIANT (primary; use for unseen egodex) --")
    print(f"  DTW_shape (unit-path shape dist, LOWER=better): head={summary['dtws_head']['mean']:.4f}  "
          f"L={summary['dtws_L']['mean']:.4f}  R={summary['dtws_R']['mean']:.4f}  LR={summary['dtws_LR']['mean']:.4f}")
    print(f"  Δpos cosine (dir sim, HIGHER=better): head={summary['cos_head']['mean']:.3f}  "
          f"L={summary['cos_L']['mean']:.3f}  R={summary['cos_R']['mean']:.3f}  LR={summary['cos_LR']['mean']:.3f}")
    print(f"  Pearson corr (HIGHER=better): head={summary['corr_head']['mean']:.3f}  "
          f"L={summary['corr_L']['mean']:.3f}  R={summary['corr_R']['mean']:.3f}")
    print("  -- SCALE-DEPENDENT (reference; trust only in-distribution) --")
    print(f"  DTW_norm (LOWER=better): head={summary['dtw_head']['mean']:.4f}  "
          f"L={summary['dtw_L']['mean']:.4f}  R={summary['dtw_R']['mean']:.4f}  LR={summary['dtw_LR']['mean']:.4f}")
    print(f"  MSE Δpos (ref): L={summary['mse_L']['mean']:.5f}  R={summary['mse_R']['mean']:.5f}")
    print(f"  scale_ratio (gen/gt |path|, 1.0=match): head={summary['sr_head']['mean']:.2f}  "
          f"L={summary['sr_L']['mean']:.2f}  R={summary['sr_R']['mean']:.2f}  LR={summary['sr_LR']['mean']:.2f}")
    print(f"  MSE Δpos (ref): L={summary['mse_L']['mean']:.5f}  R={summary['mse_R']['mean']:.5f}")
    if video_metrics:
        print("  -- VIDEO QUALITY (gen vs GT query clip; SSIM/PSNR HIGHER, L1 LOWER = better) --")
        print(f"  SSIM={summary['img_ssim']['mean']:.4f}  PSNR={summary['img_psnr']['mean']:.2f}dB  "
              f"L1={summary['img_l1']['mean']:.4f}   (stage2 WiLoR/FID dumps -> {wilor_dir})")
    print(f"  -> {csv_fp}\n  -> {json_fp}", flush=True)


if __name__ == "__main__":
    main()
