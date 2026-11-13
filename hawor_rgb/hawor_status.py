"""Progress of the HaWoR job: done / running / failed episodes, frames done, throughput and ETA.

usage: hawor_status.py --episodes episodes.txt --out $HAWOR_OUT [--video-root $EGOVERSE_HUM_ROOT]
"""
import argparse, glob, json, os, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--video-root", default=os.environ.get("EGOVERSE_HUM_ROOT"))
    a = ap.parse_args()
    eps = [l.strip() for l in open(a.episodes) if l.strip()]
    done = {os.path.basename(p)[:-4] for p in glob.glob(f"{a.out}/*.npz")}
    failed = {os.path.basename(p)[:-4] for p in glob.glob(f"{a.out}/_failed/*.txt")}
    now = time.time()
    claims = {os.path.basename(p): now - os.path.getmtime(p) for p in glob.glob(f"{a.out}/_claims/*")}
    nf = {}
    if a.video_root:
        for e in eps:
            m = f"{a.video_root}/{e}/images.front_1/zarr.json"
            if os.path.exists(m): nf[e] = int(json.load(open(m))["shape"][0])
    tot = sum(nf.values()) or 1; fdone = sum(nf.get(e, 0) for e in done)
    print(f"episodes {len(done & set(eps))}/{len(eps)} done, {len(claims)} running, {len(failed)} failed")
    if nf: print(f"frames   {fdone:,}/{tot:,} ({100 * fdone / tot:.1f}%)")
    mt = sorted(os.path.getmtime(f"{a.out}/{e}.npz") for e in done)
    if len(mt) > 20 and nf:
        recent = [e for e in done if os.path.getmtime(f"{a.out}/{e}.npz") > now - 3600]
        rate = sum(nf.get(e, 0) for e in recent) / 3600
        if rate > 0: print(f"rate     {rate:,.0f} frames/s over the last hour -> ETA {(tot - fdone) / rate / 3600:.1f} h")
    stale = [e for e, age in claims.items() if age > 3600]
    if stale: print(f"stale claims (>1 h without heartbeat; reclaimed automatically): {stale[:5]}")
    if failed: print(f"failed: {sorted(failed)[:10]}  (reason in _failed/<ep>.txt; delete the file to retry)")


if __name__ == "__main__":
    main()
