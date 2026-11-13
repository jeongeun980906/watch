"""v10 human matcher (71_3s.py, the matcher behind WATCH's egoverse_pairs_v10) on descriptors from build_desc.py.

match() and the shortlist/bucket tables are verbatim from 71_3s.py:
  same source : episodes whose task-text embedding has cosine >= 0.80 with the query's (t5_text.py) -> their
                windows -> L2 on [M | 2.5 E | 0.3 G] (per-source whitened) -> 40 nearest (one per episode)
                -> DTW (band 6) on the 40x66 sequence + handedness penalty -> top 8
  cross source: per other source, the query's fine task class bucket (fallback: coarse super-group) -> 40
                nearest on whitened M (faiss, exact L2) -> same DTW score -> keep score <= 1.6 x best, cap 7
  merged by score, top 20.
Every human window is a query (train and held-out, as in v9_human_pairs.npz); finalize_manifest.py splits them.
Shardable: --shard i/n runs the i-th of n contiguous slices of the query list (e.g. one slice per node).

usage (python, numpy, faiss-cpu; watch repo on PYTHONPATH for lang_taxonomy):
  match.py --desc <build_desc out> --temb temb.npz --manifest <v10 manifest> --out pairs_shard_<i>.npz [--shard i/n] [--procs 48]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import argparse, sys, time
from multiprocessing import Pool
import numpy as np, faiss
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lang_taxonomy as LT  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--desc", required=True); ap.add_argument("--temb", required=True); ap.add_argument("--manifest", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--shard", default="0/1"); ap.add_argument("--procs", type=int, default=48)
a = ap.parse_args(); faiss.omp_set_num_threads(1); t0 = time.time()
NPE = 40; SAME_FLOOR = 0.80; SAME_N = 8; CROSS_MAX = 7; SIM_RATIO = 1.6
DTW_BAND = 6; MINK = 5; MAXK = 20; HAND_W = 3.0; OPP_THRESH = 0.25

W = np.load(f"{a.manifest}/windows.npz", allow_pickle=True)
ep = W["ep"].astype(str); wemb = W["emb"].astype(str); wfine = W["cls"].astype(str)
DS = a.desc
M = np.asarray(np.load(f"{DS}/desc_M.npy", mmap_mode="r"), np.float32)
E = np.load(f"{DS}/desc_E.npy").astype(np.float32); G = np.asarray(np.load(f"{DS}/desc_G.npy", mmap_mode="r"), np.float32)
V9 = np.load(f"{DS}/desc_V9.npy", mmap_mode="r"); LF = np.load(f"{DS}/desc_LF.npy", mmap_mode="r")
VALID = np.load(f"{DS}/desc_valid.npy")
T = np.load(a.temb, allow_pickle=True); tkey = {k: i for i, k in enumerate(T["ep"].astype(str))}
TE = T["emb"].astype(np.float32)
wkey = np.array(["H|" + e for e in ep])
HUMAN = np.isin(wemb, ["aria", "scale", "MECKA"])
wi = np.array([tkey.get(e, -1) for e in ep])
keep = W["keep"].astype(bool) & VALID & HUMAN & (wi >= 0)
wact = np.zeros((len(ep), TE.shape[1]), np.float32); wact[keep] = TE[wi[keep]]
wact /= (np.linalg.norm(wact, axis=1, keepdims=True) + 1e-9); wact = np.ascontiguousarray(wact)
wcoarse = np.array([LT.super_of(c) for c in wfine])


def whiten(Xa):
    Y = Xa.copy()
    for e in set(wemb[keep]):
        m = (wemb == e) & keep; Y[m] = (Xa[m] - Xa[m].mean(0)) / (Xa[m].std(0) + 1e-6)
    return Y
Mw = whiten(M)
Xw = np.ascontiguousarray(Mw.astype(np.float32))
X = np.ascontiguousarray(np.concatenate([Mw, 2.5 * whiten(E), 0.3 * whiten(G)], 1).astype(np.float32))
del M, Mw
EMBS = ["aria", "scale", "MECKA"]; idxall = np.where(keep)[0]
gi = {e: idxall[wemb[idxall] == e] for e in EMBS}; gi = {e: g for e, g in gi.items() if len(g)}
print(f"loaded N={len(ep)} kept={keep.sum()} ({time.time()-t0:.0f}s)", flush=True)

# ---------------- verbatim from 71_3s.py ----------------
ep_act = {}; ep_win = {}
for e, g in gi.items():
    eks = wkey[g]; uniq, inv = np.unique(eks, return_inverse=True)
    ea = np.stack([wact[g[np.where(inv == u)[0][0]]] for u in range(len(uniq))]).astype(np.float32)
    ep_act[e] = np.ascontiguousarray(ea); ep_win[e] = [g[np.where(inv == u)[0]] for u in range(len(uniq))]
fb_ids = {}; fb_idx = {}; cb_ids = {}; cb_idx = {}
for e, g in gi.items():
    for c in set(wfine[g]):
        if c == "other": continue
        ids = g[wfine[g] == c]; ix = faiss.IndexFlatL2(Xw.shape[1]); ix.add(Xw[ids]); fb_idx[(e, c)] = ix; fb_ids[(e, c)] = ids
    for c in set(wcoarse[g]):
        if c == "other": continue
        ids = g[wcoarse[g] == c]; ix = faiss.IndexFlatL2(Xw.shape[1]); ix.add(Xw[ids]); cb_idx[(e, c)] = ix; cb_ids[(e, c)] = ids
print(f"episode tables + {len(fb_idx)} fine + {len(cb_idx)} coarse buckets ({time.time()-t0:.0f}s)", flush=True)


def _dtw(qs, cs, band=DTW_BAND):
    n = qs.shape[0]; B = len(cs)
    Da = np.linalg.norm(qs[None, :, None, :] - cs[:, None, :, :], axis=3).astype(np.float32)
    acc = np.full((B, n + 1, n + 1), 1e18, np.float32); acc[:, 0, 0] = 0.0
    for i in range(1, n + 1):
        jlo, jhi = max(1, i - band), min(n, i + band)
        for j in range(jlo, jhi + 1):
            acc[:, i, j] = Da[:, i - 1, j - 1] + np.minimum(np.minimum(acc[:, i - 1, j], acc[:, i, j - 1]), acc[:, i - 1, j - 1])
    return acc[:, n, n] / (2 * n)


def _v9(idx): return np.asarray(V9[idx], np.float32)


def match(q):
    qe = wemb[q]; qf = wfine[q]; qc = wcoarse[q]; qseq = np.asarray(V9[q], np.float32); qlf = float(LF[q])
    def score(cand):
        if not len(cand): return np.array([]), np.array([])
        dd = _dtw(qseq, _v9(cand))
        sc = dd + HAND_W * np.maximum(0.0, np.abs(qlf - LF[np.asarray(cand)]) - OPP_THRESH)
        return dd, sc
    sims = ep_act[qe] @ wact[q]; order = np.argsort(-sims)
    eps = order[sims[order] >= SAME_FLOOR]
    if len(eps) < MINK: eps = order[:20]
    selparts = []; tot = 0
    for u in eps:
        ws = ep_win[qe][u]; selparts.append(ws); tot += len(ws)
        if tot >= 400: break
    sel = np.concatenate(selparts) if selparts else np.array([], int)
    same = []
    if len(sel):
        md = np.linalg.norm(X[sel] - X[q], axis=1); seen = {wkey[q]}; cand = []
        for j in np.argsort(md):
            gg = int(sel[j]); gk = wkey[gg]
            if gk in seen: continue
            seen.add(gk); cand.append(gg)
            if len(cand) >= NPE: break
        if cand:
            _, sc = score(cand); o = np.argsort(sc); same = [(int(cand[i]), float(sc[i])) for i in o[:SAME_N]]
    cross = []
    ccand = []
    for e2 in EMBS:
        if e2 == qe: continue
        if (e2, qf) in fb_idx:
            k = min(NPE, len(fb_ids[(e2, qf)])); _, I2 = fb_idx[(e2, qf)].search(Xw[q:q + 1], k)
            ccand += [int(fb_ids[(e2, qf)][i]) for i in I2[0] if i >= 0]
        elif (e2, qc) in cb_idx:
            k = min(NPE, len(cb_ids[(e2, qc)])); _, I2 = cb_idx[(e2, qc)].search(Xw[q:q + 1], k)
            ccand += [int(cb_ids[(e2, qc)][i]) for i in I2[0] if i >= 0]
    seen2 = set(); cc = [x for x in ccand if not (wkey[x] in seen2 or seen2.add(wkey[x]))]
    if cc:
        _, sc = score(cc); o = np.argsort(sc); best = sc[o[0]]
        ncr = int(np.clip(int((sc <= best * SIM_RATIO).sum()), 0, CROSS_MAX))
        cross = [(int(cc[i]), float(sc[i])) for i in o[:ncr]]
    return sorted(same + cross, key=lambda x: x[1])[:MAXK]
# --------------------------------------------------------


def run(qs):
    MI = np.full((len(qs), MAXK), -1, np.int64); MD = np.zeros((len(qs), MAXK), np.float32)
    for r, q in enumerate(qs):
        for j, (mi, dd) in enumerate(match(int(q))): MI[r, j] = mi; MD[r, j] = dd
    return MI, MD


if __name__ == "__main__":
    i, n = (int(x) for x in a.shard.split("/"))
    qall = np.array_split(idxall, n)[i]
    print(f"shard {i}/{n}: {len(qall)} queries, {a.procs} procs ({time.time()-t0:.0f}s)", flush=True)
    t1 = time.time()
    with Pool(a.procs) as pool:
        res = pool.map(run, np.array_split(qall, a.procs * 8), chunksize=1)
    MI = np.concatenate([r[0] for r in res]); MD = np.concatenate([r[1] for r in res])
    ME = np.where(MI >= 0, wemb[np.maximum(MI, 0)], "")
    np.savez(a.out, query_idx=qall, match_idx=MI, match_dist=MD, match_emb=ME, desc=os.path.abspath(DS))
    print(f"matched {len(qall)} in {time.time()-t1:.0f}s -> {a.out}", flush=True)
