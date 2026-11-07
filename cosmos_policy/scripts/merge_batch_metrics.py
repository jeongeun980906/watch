"""Merge the per-shard CSVs from eval_egoverse_ret_batch.py (8-GPU sharded run) into one
combined CSV + aggregate summary.json, and print the shape-metric summary. Pure csv/numpy
(no GPU/torch). Run AFTER all shards finish:
    python -m cosmos_policy.scripts.merge_batch_metrics --out_dir /mnt/ddn/tmp/egoverse_v9_eval \
        --jobname cosmos_predict2p5_2b_egoverse_ret_5s
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

_METRIC_COLS = ["dtws_head", "dtws_L", "dtws_R", "dtws_LR",           # scale-invariant DTW (primary)
                "cos_head", "cos_L", "cos_R", "cos_LR",
                "corr_head", "corr_L", "corr_R",
                "cosl_head", "cosl_L", "cosl_R", "cosl_LR",           # TIMING-robust: best-lag cosine
                "spec_head", "spec_L", "spec_R", "spec_LR",           # TIMING-robust: spectral cosine
                "dtwsn_head", "dtwsn_L", "dtwsn_R", "dtwsn_LR",       # NORMALIZED-space (stats-free)
                "cosn_head", "cosn_L", "cosn_R", "cosn_LR", "corrn_head", "corrn_L", "corrn_R",
                "dtw_head", "dtw_L", "dtw_R", "dtw_LR",               # scale-dependent (ref)
                "mse_head", "mse_L", "mse_R",
                # NORMALIZED-space chunk error + accuracy@tau (cross-paper convention)
                "msen_all", "l1n_all", "acc0.5", "acc0.1",
                "msen_eef", "l1n_eef", "acc0.5_eef", "acc0.1_eef",
                "sr_head", "sr_L", "sr_R", "sr_LR"]                   # scale diagnostic


def _agg(vals):
    a = np.asarray([v for v in vals if v == v], np.float64)      # drop NaN
    if a.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "median": float(np.median(a)), "n": int(a.size)}


def _agg_by_episode(rows, col, kind=None):
    """EPISODE-LEVEL macro-average over the FULL merged rows: mean over each episode's
    evals, then mean/median over episodes (equal weight per episode). `kind` filters rows to
    "same"/"cross" for the retrieval split. Correct place to do this is here — after all
    shards merge — since one episode's evals can be split across shards."""
    from collections import defaultdict
    by_ep = defaultdict(list)
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
        return {"mean": float("nan"), "median": float("nan"), "n_ep": 0, "n_win": 0}
    return {"mean": float(ep_means.mean()), "median": float(np.median(ep_means)),
            "n_ep": int(ep_means.size), "n_win": int(sum(len(v) for v in by_ep.values()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/mnt/ddn/tmp/egoverse_v9_eval")
    ap.add_argument("--jobname", default="cosmos_predict2p5_2b_egoverse_ret_5s")
    ap.add_argument("--num-shards", dest="num_shards", type=int, default=0,
                    help="if >0, only merge shard*of<N>.csv (ignores stale runs with a different N)")
    a = ap.parse_args()

    _of = f"of{a.num_shards}" if a.num_shards > 0 else "of*"
    pat = os.path.join(a.out_dir, f"eval_batch_{a.jobname}_metrics_shard*{_of}.csv")
    shard_csvs = sorted(glob.glob(pat))
    if not shard_csvs:
        raise SystemExit(f"no shard CSVs matched {pat}")
    rows = []
    for fp in shard_csvs:
        with open(fp) as f:
            rows.extend(list(csv.DictReader(f)))
    print(f"[merge] {len(shard_csvs)} shards -> {len(rows)} samples", flush=True)

    # combined CSV
    comb_fp = os.path.join(a.out_dir, f"eval_batch_{a.jobname}_metrics.csv")
    with open(comb_fp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # video-quality columns (present only when the shards ran with VIDEO_METRICS=1)
    vid_cols = [c for c in ("img_l1", "img_psnr", "img_ssim") if c in rows[0]]

    summary = {"n_samples": len(rows), "jobname": a.jobname, "n_shards": len(shard_csvs)}
    for c in _METRIC_COLS + vid_cols:                             # EPISODE-LEVEL macro-average
        summary[c] = _agg_by_episode(rows, c)
        for kd in sorted({str(r.get("kind", "")) for r in rows} - {""}):   # same/cross or A/B/C
            summary[f"{c}__{kd}"] = _agg_by_episode(rows, c, kind=kd)
    json_fp = os.path.join(a.out_dir, f"eval_batch_{a.jobname}_summary.json")
    json.dump(summary, open(json_fp, "w"), indent=2)

    if "msen_all" in rows[0]:
        print("  -- NORMALIZED action space ([-1,1], per-dim comparable) --")
        print(f"  MSE={summary['msen_all']['mean']:.4f}  L1={summary['l1n_all']['mean']:.4f}  "
              f"acc@0.5={summary['acc0.5']['mean']:.4f}  acc@0.1={summary['acc0.1']['mean']:.4f}"
              f"   | wrist-only: acc@0.5={summary['acc0.5_eef']['mean']:.4f} "
              f"acc@0.1={summary['acc0.1_eef']['mean']:.4f}")
    _ep = summary["dtws_L"]["n_ep"]; _wn = summary["dtws_L"]["n_win"]
    print("\n==== SHAPE-METRIC SUMMARY (merged, episode-macro-avg) ====", flush=True)
    print(f"  episodes: {_ep}  windows: {_wn}  (shards: {len(shard_csvs)}; per-episode mean → over episodes)")
    print("  -- SCALE-INVARIANT (primary; use for unseen egodex) --")
    print(f"  DTW_shape (unit-path shape dist, LOWER=better): head={summary['dtws_head']['mean']:.4f}  "
          f"L={summary['dtws_L']['mean']:.4f}  R={summary['dtws_R']['mean']:.4f}  LR={summary['dtws_LR']['mean']:.4f}")
    print(f"  Δpos cosine (dir sim, HIGHER=better): head={summary['cos_head']['mean']:.3f}  "
          f"L={summary['cos_L']['mean']:.3f}  R={summary['cos_R']['mean']:.3f}  LR={summary['cos_LR']['mean']:.3f}")
    print(f"  Pearson corr (HIGHER=better): head={summary['corr_head']['mean']:.3f}  "
          f"L={summary['corr_L']['mean']:.3f}  R={summary['corr_R']['mean']:.3f}")
    print("  -- TIMING-ROBUST (best even if the motion is time-shifted/warped) --")
    print(f"  best-lag cosine (↑): head={summary['cosl_head']['mean']:.3f}  "
          f"L={summary['cosl_L']['mean']:.3f}  R={summary['cosl_R']['mean']:.3f}  LR={summary['cosl_LR']['mean']:.3f}")
    print(f"  spectral |FFT| cosine (↑): head={summary['spec_head']['mean']:.3f}  "
          f"L={summary['spec_L']['mean']:.3f}  R={summary['spec_R']['mean']:.3f}  LR={summary['spec_LR']['mean']:.3f}")
    print("  -- NORMALIZED space ([-1,1], de-norm-stats-independent) --")
    print(f"  DTW_shape(norm)↓: L={summary['dtwsn_L']['mean']:.3f} R={summary['dtwsn_R']['mean']:.3f}  "
          f"cos(norm)↑: L={summary['cosn_L']['mean']:.3f} R={summary['cosn_R']['mean']:.3f}  "
          f"corr(norm)↑: L={summary['corrn_L']['mean']:.3f} R={summary['corrn_R']['mean']:.3f}")
    print("  -- SCALE-DEPENDENT (reference; trust only in-distribution) --")
    print(f"  DTW_norm (LOWER=better): head={summary['dtw_head']['mean']:.4f}  "
          f"L={summary['dtw_L']['mean']:.4f}  R={summary['dtw_R']['mean']:.4f}  LR={summary['dtw_LR']['mean']:.4f}")
    print(f"  MSE Δpos (ref): L={summary['mse_L']['mean']:.5f}  R={summary['mse_R']['mean']:.5f}")
    print(f"  scale_ratio (gen/gt |path|, 1.0=match): head={summary['sr_head']['mean']:.2f}  "
          f"L={summary['sr_L']['mean']:.2f}  R={summary['sr_R']['mean']:.2f}  LR={summary['sr_LR']['mean']:.2f}")
    if vid_cols:
        print("  -- VIDEO QUALITY (gen vs GT query clip; SSIM/PSNR HIGHER, L1 LOWER = better) --")
        print(f"  SSIM={summary['img_ssim']['mean']:.4f}  PSNR={summary['img_psnr']['mean']:.2f}dB  "
              f"L1={summary['img_l1']['mean']:.4f}  "
              f"(same: SSIM={summary['img_ssim__same']['mean']:.4f} / cross: {summary['img_ssim__cross']['mean']:.4f})")
    print(f"  -> {comb_fp}\n  -> {json_fp}", flush=True)


if __name__ == "__main__":
    main()
