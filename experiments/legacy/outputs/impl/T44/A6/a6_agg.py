"""T44c A6 smoke aggregation (implementer). Same definitions as outputs/verify/T44/E2c/agg_e2c.py (gt_label, demo).
cond1 T neg = value < 0 ; cond2 D clearance>=margin = value - r_finger >= 10 mm (frames with any) ;
cond3 TB (z in [plane-30mm, plane)) negatives, optimistic flips (off<0 & v>=0), TA (0-10 mm above plane, outside hulls) v > off ;
cond4 q cross = (off - r >= m) & (v - r < m), demo_last2 = 3rd+4th demo frame per episode ;
cond5 v > off + 1e-6 or v < min(off, d_main) - 1e-6 over D, T, q, TB, TA.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import collections, glob, json, pathlib, sys
import numpy as np

D = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else _ag3s_workspace_path('outputs/impl/T44/A6/raw_demo'))
MS, VS = ("ball", "bo_d3"), ("off", "a5", "a", "b", "ab")
M, EPS = 0.010, 1e-6
recs = []
for f in sorted(glob.glob(str(D / "*.json"))):
    r = json.load(open(f)); r["_z"] = np.load(f[:-5] + ".npz"); recs.append(r)
by = collections.defaultdict(list)
for r in recs:
    by[r["source"]["episode_index"]].append(r["seq"])
for r in recs:
    fr = sorted(by[r["source"]["episode_index"]]); r["_rank"] = len(fr) - 1 - fr.index(r["seq"])
out = {"n_frames": len(recs), "definitions": __doc__, "methods": {}}
for m in MS:
    o = {}
    for v in VS:
        a = collections.Counter()
        for r in recs:
            z = r["_z"]
            if f"T_{m}_{v}" not in z.files:
                continue
            rf = float(z["r_finger"]); ra = z["auth_radii"].astype(np.float64)
            T = z[f"T_{m}_{v}"].astype(np.float64); h = z["T_height"].astype(np.float64)
            a["T_neg"] += int((T < 0).sum()); a["T_neg_frames"] += int((T < 0).any())
            a["T_neg_h_lt_20mm"] += int(((T < 0) & (h < 0.02)).sum())
            a["T_neg_h_lt_10mm"] += int(((T < 0) & (h < 0.01)).sum())
            a["T_n"] += len(T)
            Dv = z[f"D_{m}_{v}"].astype(np.float64)
            c = Dv - rf >= M
            a["D_clr_ge_m"] += int(c.sum()); a["D_clr_ge_m_frames"] += int(c.any())
            a["D_pos"] += int((Dv > 0).sum())
            for s in ("TB", "TA"):
                if f"{s}_{m}_{v}" in z.files:
                    x = z[f"{s}_{m}_{v}"].astype(np.float64); xo = z[f"{s}_{m}_off"].astype(np.float64)
                    xa5 = z[f"{s}_{m}_a5"].astype(np.float64)
                    a[f"{s}_n"] += len(x); a[f"{s}_neg"] += int((x < 0).sum())
                    a[f"{s}_optimistic_vs_off"] += int(((xo < 0) & (x >= 0)).sum()) if s == "TB" else int((x > xo + EPS).sum())
                    a[f"{s}_gt_a5"] += int((x > xa5 + EPS).sum())
                    a[f"{s}_sum_mm"] += float(x.sum() * 1e3)
            q = z[f"q_{m}_{v}"].astype(np.float64); qo = z[f"q_{m}_off"].astype(np.float64)
            cr = (qo - ra >= M) & (q - ra < M)
            a["q_cross"] += int(cr.sum()); a["q_cross_frames"] += int(cr.any())
            a["q_below_m"] += int((q - ra < M).sum())
            if r["_rank"] in (0, 1):
                a["last2_q_cross"] += int(cr.sum()); a["last2_q_cross_frames"] += int(cr.any())
                a["last2_frames"] += 1
            for s in ("D", "T", "q", "TB", "TA"):
                if f"{s}_{m}_{v}" not in z.files:
                    continue
                x = z[f"{s}_{m}_{v}"].astype(np.float64); xo = z[f"{s}_{m}_off"].astype(np.float64)
                dm = z[f"{s}_dmain"].astype(np.float64)
                a["p5_gt_off"] += int((x > xo + EPS).sum())
                a["p5_lt_min_off_main"] += int((x < np.minimum(xo, dm) - EPS).sum())
                a["p5_n"] += len(x)
            if v == "a":
                for s in ("D", "T"):
                    k = f"{s}_{m}_a_answer_diff"
                    if k in z.files:
                        a["answer_vs_value_maxdiff_um"] = max(a["answer_vs_value_maxdiff_um"], float(z[k].max()) * 1e6)
        a["frames"] = len(recs)
        for s in ("TB", "TA"):
            if a[f"{s}_n"]:
                a[f"{s}_mean_mm"] = round(a[f"{s}_sum_mm"] / a[f"{s}_n"], 3)
            a.pop(f"{s}_sum_mm", None)
        o[v] = dict(a)
    # coarse counts
    for fl in ("guard", "plane"):
        cs = [r["coarse"][f"{m}|{fl}"] for r in recs if "coarse" in r]
        pb = collections.Counter()
        for c in cs:
            pb.update(c.get("plane_band", {}))
        ms_ = [r["coarse_build_ms"][f"{m}|{fl}"] for r in recs if "coarse_build_ms" in r]
        o[f"coarse_{fl}"] = {"n_sign_forced_sum": int(sum(c["n_sign_forced"] for c in cs)),
                            "n_removed_sum": int(sum(c["n_removed"] for c in cs)),
                            "plane_band_sum": dict(pb),
                            "build_ms_once_median": round(float(np.median(ms_)), 3) if ms_ else None,
                            "build_ms_once_p90": round(float(np.percentile(ms_, 90)), 3) if ms_ else None}
    out["methods"][m] = o
json.dump(out, open(D.parent / "a6_agg.json", "w"), indent=1, ensure_ascii=False)
for m in MS:
    print(m)
    for v in VS:
        a = out["methods"][m][v]
        print(f"  {v:3s} Tneg {a['T_neg']:6d} ({a['T_neg_frames']}f, h<10 {a['T_neg_h_lt_10mm']}) | Dclr>=m {a['D_clr_ge_m']:5d} {a['D_clr_ge_m_frames']}f "
              f"| TBneg {a.get('TB_neg')} opt {a.get('TB_optimistic_vs_off')} | TA mean {a.get('TA_mean_mm')} opt {a.get('TA_optimistic_vs_off')} "
              f"| qcross {a['q_cross']}/{a['q_cross_frames']}f last2 {a['last2_q_cross']}/{a['last2_q_cross_frames']}f | p5 {a['p5_gt_off']}/{a['p5_lt_min_off_main']}")
    print("  ", out["methods"][m]["coarse_guard"], out["methods"][m]["coarse_plane"])
