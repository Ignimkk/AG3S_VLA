"""Summarize tp_timing JSONs: per setting, median/p90 of wall and per-stage exclusive ms, call counts."""
import json, sys, glob
import numpy as np
files = sorted(sum([glob.glob(a) for a in sys.argv[1:]], []))
acc = {}
for f in files:
    d = json.load(open(f))
    for ch in d["chunks"]:
        for v, r in ch.get("variants", {}).items():
            a = acc.setdefault(v, {"wall": [], "cpu": [], "st": {}, "calls": {}, "rec": []})
            a["wall"].append(np.median(r["wall_ms"])); a["cpu"].append(np.median(r["cpu_ms"]))
            for k in r["stages"][0]:
                a["st"].setdefault(k, []).append(np.median([s.get(k, 0) for s in r["stages"]]))
            for k, c in r["calls"][0].items():
                a["calls"].setdefault(k, []).append(c)
q = lambda x, p: float(np.quantile(x, p))
for v, a in acc.items():
    print(f"{v:14s} n={len(a['wall'])} wall med {np.median(a['wall']):6.1f} p90 {q(a['wall'],.9):6.1f} | cpu med {np.median(a['cpu']):6.1f}")
    print("   stages med: " + " ".join(f"{k}={np.median(x):.1f}" for k, x in sorted(a['st'].items())))
    print("   stages mean: " + " ".join(f"{k}={np.mean(x):.1f}" for k, x in sorted(a['st'].items())))
    print("   calls med: " + " ".join(f"{k}={np.median(x):.0f}" for k, x in sorted(a['calls'].items())))
