"""Merge match.py shards into a training manifest directory.

Like 73_v10.py: the held-out episodes (manifest/holdout_eps.txt) are removed as queries only (they stay
available as references), so training sees the same query episodes as WATCH. The output directory holds
symlinks to every file of the base manifest (windows.npz, t5_embeddings.pkl, *_stats.json, ...) plus
query_eps_nomd.json, and the new pairs file pairs_<name>.npz. It prints how the new references compare with the
released WATCH pairs (pairs.npz), for the record.

With --md-from <released pairs.npz> it also appends the MimicDroid play query rows (windows with source "md")
of the released pairs verbatim, as 73b_v10.py brought them into v10 (match.py matches human queries only), and
writes query_eps_md.json = the query set + the MimicDroid episodes. Output pairs_<name>.npz then holds both.

usage: finalize_manifest.py --shards 'pairs_shard_*.npz' --base <v10 manifest> --query-set <query_eps_nomd.json>
                            --name hawor --out <manifest dir> [--md-from <v10 manifest>/pairs.npz]
"""
import argparse, glob, json, os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--query-set", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--md-from", default=None, help="released pairs.npz: append its MimicDroid query rows verbatim")
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
    qset = a.query_set; note = f"hawor_rgb/match.py ({a.name}); held-out episodes removed as queries"
    qi, mi, md, me = qi[tr], mi[tr], md[tr], me[tr]; nho = int((~tr).sum())
    if a.md_from:                                     # MimicDroid play rows of the released pairs, verbatim
        src = W["source"].astype(str)
        assert not (src[qi] == "md").any(), "shards already contain MimicDroid queries"
        P = np.load(a.md_from, allow_pickle=True); pq, pm, pd = P["query_idx"], P["match_idx"], P["match_dist"]
        k = src[pq] == "md"; K = max(mi.shape[1], pm.shape[1])
        pad = lambda x, w, f: np.pad(x, ((0, 0), (0, w - x.shape[1])), constant_values=f)
        mdq, mdm, mdd = pq[k], pad(pm[k], K, -1), pad(pd[k].astype(md.dtype), K, 0)
        mde = np.where(mdm >= 0, emb[np.maximum(mdm, 0)], "")
        mi, md, me = pad(mi, K, -1), pad(md, K, 0), pad(me, K, "")
        qi = np.concatenate([qi, mdq]); mi = np.concatenate([mi, mdm]); md = np.concatenate([md, mdd])
        me = np.concatenate([me.astype(str), mde.astype(str)])
        o = np.argsort(qi, kind="stable"); qi, mi, md, me = qi[o], mi[o], md[o], me[o]
        mdeps = sorted(set(ep[mdq].tolist()))
        Q = json.load(open(a.query_set)); Q2 = dict(Q)
        Q2["episodes"] = sorted(set(Q["episodes"]) | set(mdeps)); Q2["n_episodes"] = len(Q2["episodes"])
        Q2["rule"] = Q.get("rule", "") + " + the MimicDroid play episodes (finalize_manifest.py --md-from)"
        Q2["dropped_episodes"] = []; Q2["n_rows"] = int(len(qi))
        qset = f"{a.out}/query_eps_md.json"
        if os.path.lexists(qset): os.remove(qset)
        json.dump(Q2, open(qset, "w"), indent=1)
        note += f"; + {len(mdq):,} MimicDroid query rows ({len(mdeps)} episodes) of {a.md_from} verbatim"
        print(f"MimicDroid: {len(mdq):,} query rows from {len(mdeps)} episodes appended; -> {qset} ({Q2['n_episodes']} episodes)")
    np.savez(out, query_idx=qi, match_idx=mi, match_dist=md, match_emb=me, note=np.array(note))
    tr = np.ones(len(qi), bool)
    want = set(json.load(open(qset))["episodes"]); have = set(ep[qi[tr]].tolist())
    miss = want - have
    n = mi[tr] >= 0; cross = n & (np.where(n, emb[np.maximum(mi[tr], 0)], "") != emb[qi[tr]][:, None])
    print(f"{out}: {tr.sum():,} train query rows ({nho:,} held-out rows dropped), refs/query {n.sum(1).mean():.2f}, "
          f"cross refs/query {cross.sum(1).mean():.2f}, rows with a cross ref {100 * cross.any(1).mean():.1f}%")
    print(f"query set {os.path.basename(qset)}: {len(want - miss)}/{len(want)} episodes present"
          + (f"  MISSING {sorted(miss)[:3]} -> training will refuse this manifest" if miss else ""))
    if os.path.exists(f"{a.base}/pairs.npz"):        # agreement with the released WATCH references
        P = np.load(f"{a.base}/pairs.npz", allow_pickle=True); PM = P["match_idx"]      # load once (NpzFile re-reads per access)
        row = {int(q): k for k, q in enumerate(P["query_idx"])}
        hum = np.where(W["source"].astype(str)[qi] != "md")[0]       # MimicDroid rows are copies of the released ones
        rng = np.random.default_rng(0); sel = rng.choice(hum, min(20000, len(hum)), replace=False)
        jac, top1 = [], []
        for k in sel:
            r = row.get(int(qi[k]))
            if r is None: continue
            A = {x for x in mi[k] if x >= 0}; B = {x for x in PM[r] if x >= 0}
            if A | B: jac.append(len(A & B) / len(A | B)); top1.append(mi[k][0] == PM[r][0])
        print(f"vs released pairs.npz ({len(jac)} sampled rows): Jaccard {np.mean(jac):.3f}, same top-1 {np.mean(top1):.3f}")


if __name__ == "__main__":
    main()
