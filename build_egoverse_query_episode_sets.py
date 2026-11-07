"""Pin the query-episode sets for the egoverse mid-training DATA-SCALING sweep.

Decides ONCE which episodes each arm trains on and writes the lists to disk, so an arm
is defined by a file you can read, diff and hand-edit — not by a seed that has to be
replayed to know what was in it. Run this once; the training configs then just point at
the JSON via ``dataloader_train.dataset.query_episode_set``.

Why EPISODE and not row: pairs.npz enumerates ~1 s-strided windows, so v10 has a median
of 74 query rows per episode. A random 25 % of ROWS still touches essentially all 13,315
episodes (~18 rows each) — the training set's diversity would be unchanged and the sweep
would measure nothing but step count. Dropping whole episodes is what "we collected less
data" actually means.

The draw is random at episode level but STRATIFIED per embodiment (each arm keeps the same
fraction of MECKA / scale / aria / mimicdroid episodes) so the embodiment mix — which spans
6,639 down to 20 episodes in v10 — does not drift between arms; ``--uniform`` opts out and
draws from one flat pool. Sets are NESTED (q25 subset of q50 subset of q75) because each
frac takes a prefix of the same permutation, which makes the sweep a ladder rather than
four unrelated draws.

Only the QUERY side is cut. match_idx is never touched, so every arm keeps the full
retrieval corpus and the only variable is how much query data exists.

Usage:
    python build_egoverse_query_episode_sets.py            # writes query_eps_q{25,50,75}.json
    python build_egoverse_query_episode_sets.py --dry_run  # report composition only
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np


def _nested_subsets(eps: np.ndarray, embs: np.ndarray, fracs: list[float],
                    seed: int, uniform: bool) -> dict[float, list[str]]:
    """{frac: sorted episode ids}. Prefixes of one permutation per group => nested."""
    groups: dict[str, np.ndarray] = {}
    key = np.full(len(eps), "_all") if uniform else embs
    for g in np.unique(key):
        groups[str(g)] = eps[key == g]          # eps arrives sorted => order is seed-driven

    out: dict[float, set[str]] = {f: set() for f in fracs}
    for gi, g in enumerate(sorted(groups)):
        perm = np.random.default_rng([seed, gi]).permutation(groups[g])
        for f in fracs:
            # ceil so a tiny embodiment (v10 mimicdroid: 20 episodes) never rounds to zero
            out[f].update(perm[:int(np.ceil(f * len(perm)))].tolist())
    return {f: sorted(v) for f, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_dir", default="/mnt/ddn/dataset/egoverse_pairs_v10")
    ap.add_argument("--pairs_name", default="pairs.npz")
    ap.add_argument("--fracs", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--uniform", action="store_true",
                    help="draw episodes from one flat pool instead of per-embodiment")
    ap.add_argument("--dry_run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    W = np.load(os.path.join(args.manifest_dir, "windows.npz"), allow_pickle=True)
    P = np.load(os.path.join(args.manifest_dir, args.pairs_name), allow_pickle=True)
    qi = P["query_idx"].astype(np.int64)
    qep, qemb = W["ep"].astype(str)[qi], W["emb"].astype(str)[qi]

    eps, first = np.unique(qep, return_index=True)      # sorted, one entry per query episode
    embs = qemb[first]
    print(f"[src] {args.manifest_dir}/{args.pairs_name}: {len(qi):,} query rows over "
          f"{len(eps):,} query episodes  (stratified={not args.uniform}, seed={args.seed})")

    subsets = _nested_subsets(eps, embs, args.fracs, args.seed, args.uniform)

    prev: set[str] = set()
    for f in sorted(args.fracs):
        keep = subsets[f]
        assert prev <= set(keep), f"subsets not nested at frac={f}"
        prev = set(keep)

        mask = np.isin(qep, np.asarray(keep))
        per_emb = collections.Counter(qemb[mask].tolist())
        eps_per_emb = collections.Counter(embs[np.isin(eps, np.asarray(keep))].tolist())
        tag = f"q{int(round(f * 100))}"
        print(f"[{tag}] rows {int(mask.sum()):>9,} ({mask.mean():6.1%})  "
              f"episodes {len(keep):>6,} ({len(keep) / len(eps):6.1%})  "
              + "  ".join(f"{m}:{per_emb[m]:,}r/{eps_per_emb[m]}ep" for m in sorted(eps_per_emb)))
        if args.dry_run:
            continue
        dst = os.path.join(args.manifest_dir, f"query_eps_{tag}.json")
        with open(dst, "w") as fp:
            json.dump({
                "source": os.path.join(args.manifest_dir, args.pairs_name),
                "frac": f,
                "seed": args.seed,
                "stratified": not args.uniform,
                "nested": "prefix of one per-group permutation; q25 < q50 < q75 < full",
                "n_episodes": len(keep),
                "n_rows": int(mask.sum()),
                "episodes_per_embodiment": dict(sorted(eps_per_emb.items())),
                "rows_per_embodiment": dict(sorted(per_emb.items())),
                "episodes": keep,
            }, fp, indent=1)
        print(f"       -> {dst}")


if __name__ == "__main__":
    main()
