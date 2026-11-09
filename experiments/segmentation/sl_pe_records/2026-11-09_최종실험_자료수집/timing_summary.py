"""(2026-11-09 12:3x KST dev-4a) Run-time draft for plan E6 / Tab. S11 from EXISTING E1a Spatial LI runs (exclusive GPU lock,
1 job). Boundary = client request -> returned chunk (t_client_ms, includes policy inference). Server split (timing_ms):
infer (pi0.5 forward + attention read-out), ag3s (perception / ESDF / constraints), trajopt. AG3S stage split from the chunk
records (summary_json.profile_ms). First call of every run excluded (JIT). STATUS: DERIVED from existing runs, not the
E6 protocol run (no fixed 2,000-request block, pre-optimisation prototype). usage: python timing_summary.py"""
import glob
import json

import numpy as np

B = "/mnt/dev/AG3S_VLA-storage/results/benchmark/"
OUT = "/mnt/dev/AG3S_VLA/outputs/2026-11-09_최종실험_자료수집/"
ARMS = [("pi0.5 only", "2*qe-vla-e1a-spatial-LI-t*"), ("AEGIS port (detector)", "*_e1a-aegis-spatial-LI-*"),
        ("multi-link CBF (detector)", "*_e1a-multilink_cbf-spatial-LI-*"), ("KNOWS port (detector)", "*_e1a-knows-spatial-LI-*"),
        ("AG3S B5 (GT)", "2*qe-gt23t100-e1a-spatial-LI-t*"), ("AG3S B5-tro (GT)", "2*qe-gt23tro-e1a-spatial-LI-t*"),
        ("AG3S noGT pe123458", "2*qe-pe123458-e1a-spatial-LI-t*")]


def pct(v):
    v = np.asarray([x for x in v if x is not None], float)
    return None if not len(v) else dict(n=int(len(v)), p50=round(float(np.percentile(v, 50)), 1), p95=round(float(np.percentile(v, 95)), 1))


def main():
    res = {}
    for name, pat in ARMS:
        cols = {"t_client": [], "infer": [], "ag3s": [], "trajopt": [], "total": []}
        for d in sorted(glob.glob(B + pat)):
            if "INVALID" in d:
                continue
            first = True
            for f in sorted(glob.glob(d + "/episodes/*_calls.jsonl")):
                for l in open(f):
                    c = json.loads(l)
                    if first:
                        first = False
                        continue
                    cols["t_client"].append(c.get("t_client_ms"))
                    for k in ("infer", "ag3s", "trajopt", "total"):
                        cols[k].append((c.get("timing_ms") or {}).get(k))
        res[name] = {k: pct(v) for k, v in cols.items()}
    stages = {}
    for d in sorted(glob.glob(B + "2*qe-pe123458-e1a-spatial-LI-t*")):
        for c in sorted(glob.glob(d + "/server_constraints/run_0000/chunk_*.npz"))[1:]:
            pm = json.loads(str(np.load(c)["summary_json"])).get("profile_ms") or {}
            for k, v in pm.items():
                stages.setdefault(k, []).append(v)
    res["AG3S noGT pe123458 stages (profile_ms)"] = {k: pct(v) for k, v in stages.items()}
    json.dump(res, open(OUT + "timing_summary.json", "w"), indent=1)
    for k, v in res.items():
        print(k)
        for kk, vv in v.items():
            if vv:
                print(f"   {kk:22s} n={vv['n']:5d} p50={vv['p50']:8.1f} p95={vv['p95']:8.1f}")


if __name__ == "__main__":
    main()
