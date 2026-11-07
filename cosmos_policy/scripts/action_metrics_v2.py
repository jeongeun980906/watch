"""Stage-2b: ACTION metrics from decode_dump_v2 npz dumps (numpy-only, CPU).

Per sample (generated action vs GT action, both normalized [-1,1] action57):
  action_l1 / action_mse   — normalized action space (offline analogue of training's action fit)
  wristL/R_rmse_cm         — head-relative FK wrist trajectory RMSE (cm), gen vs GT
  wrist_speed_corr         — Pearson corr of wrist speed profiles (both hands mean)
Usage: python action_metrics_v2.py <dump_dir> <out_json>
"""
import glob
import json
import sys

import numpy as np

# IDENTICAL trajectory metrics as the old batch eval (dtw_shape / corr / dtw_norm / mse):
# reuse its functions verbatim so numbers are definitionally comparable.
from cosmos_policy.scripts.eval_egoverse_ret_batch import _shape_metrics  # noqa: E402

# action57 per-step eef Δpos columns (L: 9:12, R: 33:36) — same input the old eval used
SL_L, SL_R = slice(9, 12), slice(33, 36)

dump, out_json = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(f"{dump}/*.npz")):
    d = np.load(f, allow_pickle=True)
    if "gen_a" not in d.files:
        continue
    ga, ta = d["gen_a"], d["gt_a"]
    n = min(len(ga), len(ta))
    ga, ta = ga[:n], ta[:n]
    # STATE-chunk runs (decode_dump_v2 "_state" mode): gen_a/gt_a are DERIVED delta
    # action57 (physical) so the shape metrics below stay convention-identical; the
    # model-space L1/MSE then comes from the added normalized-chunk keys.
    gn, tn = (d["gen_norm"], d["gt_norm"]) if "gen_norm" in d.files else (ga, ta)
    nn_ = min(len(gn), len(tn))
    l1 = float(np.abs(gn[:nn_] - tn[:nn_]).mean())
    mse = float(((gn[:nn_] - tn[:nn_]) ** 2).mean())
    def rmse(a, b):
        m = min(len(a), len(b))
        return float(np.sqrt(((a[:m] - b[:m]) ** 2).sum(-1).mean()))
    wl = rmse(d["fkL"], d["fkL_gt"]); wr = rmse(d["fkR"], d["fkR_gt"])
    def spd(w):
        return np.linalg.norm(np.diff(w, axis=0), axis=1)
    cs = []
    for gw, tw in ((d["fkL"], d["fkL_gt"]), (d["fkR"], d["fkR_gt"])):
        a, b = spd(gw), spd(tw)
        m = min(len(a), len(b)); a, b = a[:m], b[:m]
        if np.std(a) > 1e-6 and np.std(b) > 1e-6:
            cs.append(float(np.corrcoef(a, b)[0, 1]))
    # OLD-EVAL-IDENTICAL shape metrics on per-step eef Δpos (both hands, averaged)
    sm = {}
    for hand, sl in (("L", SL_L), ("R", SL_R)):
        m = _shape_metrics(ta[:, sl].astype(np.float64), ga[:, sl].astype(np.float64))
        for kk, vv in m.items():
            sm.setdefault(kk, []).append(float(vv))
    shape = {kk: float(np.mean(vv)) for kk, vv in sm.items()}
    rows.append(dict(idx=int(d["idx"]), emb=str(d["emb"]), ret=str(d["ret"]),
                     action_l1=l1, action_mse=mse, wristL_rmse_cm=wl, wristR_rmse_cm=wr,
                     wrist_speed_corr=(float(np.mean(cs)) if cs else None), **shape))

wc = [r["wrist_speed_corr"] for r in rows if r["wrist_speed_corr"] is not None]
summary = dict(
    n=len(rows),
    action_l1=float(np.mean([r["action_l1"] for r in rows])) if rows else None,
    action_mse=float(np.mean([r["action_mse"] for r in rows])) if rows else None,
    wristL_rmse_cm=float(np.mean([r["wristL_rmse_cm"] for r in rows])) if rows else None,
    wristR_rmse_cm=float(np.mean([r["wristR_rmse_cm"] for r in rows])) if rows else None,
    wrist_speed_corr=float(np.mean(wc)) if wc else None,
)
# aggregate old-eval shape metrics (dtw_shape / corr / dtw_norm / mse / scale_ratio ...)
for kk in rows[0].keys() if rows else []:
    if kk in summary or kk in ("idx", "emb", "ret"):
        continue
    vals = [r[kk] for r in rows if isinstance(r.get(kk), (int, float)) and np.isfinite(r[kk])]
    if vals:
        summary[kk] = float(np.mean(vals))
json.dump(dict(summary=summary, rows=rows), open(out_json, "w"), indent=2)
print("ACTION SUMMARY", json.dumps(summary), flush=True)
