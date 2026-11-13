"""Merge match.py shards into a training manifest directory.

Like 73_v10.py: the held-out episodes (manifest/holdout_eps.txt) are removed as queries only (they stay
available as references), so training sees the same query episodes as WATCH. The output directory holds
symlinks to every file of the base manifest (windows.npz, t5_embeddings.pkl, *_stats.json, ...) plus
query_eps_nomd.json, and the new pairs file pairs_<name>.npz. It prints how the new references compare with the
released WATCH pairs (pairs.npz), for the record.

usage: finalize_manifest.py --shards 'pairs_shard_*.npz' --base <v10 manifest> --query-set <query_eps_nomd.json>
                            --name hawor --out <manifest dir>
"""
import argparse, glob, json, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--query-set", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fs = sorted(glob.glob(a.shards)); assert fs, a.shards
    S = [np.load(f, allow_pickle=True) for f in fs]
    qi = np.concatenate([s["query_idx"] for s in S]); mi = np.concatenate([s["match_idx"] for s in S])
    md = np.concatenate([s["match_dist"] for s in S]); me = np.concatenate([s["match_emb"] for s in S])
    o = np.argsort(qi, kind="stable"); qi, mi, md, me = qi[o], mi[o], md[o], me[o]
    assert len(np.unique(qi)) == len(qi), "overlapping shards"
    W = np.load(f"{a.base}/windows.npz", allow_pickle=True); ep = W["ep"].astype(str); emb = W["emb"].astype(str)
    hold = {l.split("\t")[0].strip() for l in open(f"{a.base}/holdout_eps.txt") if l.strip()}
    tr = ~np.isin(ep[qi], list(hold))
    os.makedirs(a.out, exist_ok=True)
    for f in os.listdir(a.base):                      # base manifest files, linked
        d = f"{a.out}/{f}"
        if not os.path.lexists(d) and not f.startswith("pairs"): os.symlink(os.path.abspath(f"{a.base}/{f}"), d)
    qs = f"{a.out}/query_eps_nomd.json"
    if not os.path.lexists(qs): os.symlink(os.path.abspath(a.query_set), qs)
    out = f"{a.out}/pairs_{a.name}.npz"
    np.savez(out, query_idx=qi[tr], match_idx=mi[tr], match_dist=md[tr], match_emb=me[tr],
             note=np.array(f"hawor_rgb/match.py ({a.name}); held-out episodes removed as queries"))
    want = set(json.load(open(a.query_set))["episodes"]); have = set(ep[qi[tr]].tolist())
    miss = want - have
    n = mi[tr] >= 0; cross = n & (np.where(n, emb[np.maximum(mi[tr], 0)], "") != emb[qi[tr]][:, None])
    print(f"{out}: {tr.sum():,} train query rows ({(~tr).sum():,} held-out rows dropped), refs/query {n.sum(1).mean():.2f}, "
          f"cross refs/query {cross.sum(1).mean():.2f}, rows with a cross ref {100 * cross.any(1).mean():.1f}%")
    print(f"query set {os.path.basename(a.query_set)}: {len(want - miss)}/{len(want)} episodes present"
          + (f"  MISSING {sorted(miss)[:3]} -> training will refuse this manifest" if miss else ""))
    if os.path.exists(f"{a.base}/pairs.npz"):        # agreement with the released WATCH references
        P = np.load(f"{a.base}/pairs.npz", allow_pickle=True); row = {int(q): k for k, q in enumerate(P["query_idx"])}
        rng = np.random.default_rng(0); sel = rng.choice(np.where(tr)[0], min(20000, tr.sum()), replace=False)
        jac, top1 = [], []
        for k in sel:
            r = row.get(int(qi[k]))
            if r is None: continue
            A = {x for x in mi[k] if x >= 0}; B = {x for x in P["match_idx"][r] if x >= 0}
            if A | B: jac.append(len(A & B) / len(A | B)); top1.append(mi[k][0] == P["match_idx"][r][0])
        print(f"vs released pairs.npz ({len(jac)} sampled rows): Jaccard {np.mean(jac):.3f}, same top-1 {np.mean(top1):.3f}")


if __name__ == "__main__":
    main()
