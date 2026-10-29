"""T43 Z5 aggregation: Y2 release metrics (Y/p/y_metrics.run unchanged) for E0 no-obstacle runs (Z5 14 + T39 48) next to V4 / V5 of Y (y_metrics.json) on the same ep:seed.
usage: z5_agg.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, re, statistics as st
Z = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Z5')); Y = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Y/raw/y_metrics.json'))
HURDLE = [tuple(x.split(":")) for x in (Z / "p/lists/e0.txt").read_text().split()]
yz = json.load(open(Z / "ym_z5.json"))["runs"]; yt = json.load(open(Z / "ym_t39.json"))["runs"]; yv = json.load(open(Y))["runs"]
M = ["release_apple_floor_gap_mm", "release_v_vert_mm_s", "release_v_horiz_mm_s", "post_release_horiz_travel_mm", "in_hand_slip_at_release_mm", "tcp_speed_at_open_mm_s"]
KEEP = M + ["success", "grasp", "place", "grasp_t", "place_t", "release_t", "release_mode", "release_with_cmd", "release_open_norm", "applied_open_t", "attach_t",
            "in_hand_slip_max_mm", "post_release_max_speed_mm_s", "open_minus_release_rows", "tcp_height_above_floor_at_open_mm", "apple_floor_gap_at_open_mm"]
def slim(o): return {k: o.get(k) for k in KEEP}
def med(v):
    v = [x for x in v if x is not None]; return (float(st.median(v)) if v else None, len(v))
def summ(runs):
    rel = [r for r in runs if r["release_t"] is not None]
    d = {"n_runs": len(runs), "success": sum(bool(r["success"]) for r in runs), "grasp": sum(bool(r["grasp"]) for r in runs), "place": sum(bool(r["place"]) for r in runs),
         "n_release": len(rel), "n_no_release": len(runs) - len(rel),
         "release_mode": {m: sum(r["release_mode"] == m for r in rel) for m in ("closed_cmd", "open_cmd")}}
    d["median_all_release"] = {m: dict(zip(("median", "n"), med([r[m] for r in rel]))) for m in M}
    d["by_mode"] = {}
    for mode in ("closed_cmd", "open_cmd"):
        rr = [r for r in rel if r["release_mode"] == mode]
        d["by_mode"][mode] = {"n": len(rr), **{m: dict(zip(("median", "n"), med([r[m] for r in rr]))) for m in M}}
    # success-only subset (both place and release exist)
    ss = [r for r in rel if r["success"]]
    d["n_release_among_success"] = len(ss); d["release_mode_among_success"] = {m: sum(r["release_mode"] == m for r in ss) for m in ("closed_cmd", "open_cmd")}
    return d
G = {}
G["E0_Z5_14"] = [dict(slim(yz[f"E0/B/E0_ep{e}_s{s}"]), pair=f"{e}:{s}") for e, s in HURDLE]
G["E0_T39_48"] = [dict(slim(o), pair=k.split("E0_ep")[1].replace("_s", ":")) for k, o in sorted(yt.items())]
for arm in ("V4", "V5"):
    G[f"{arm}_H_14"] = [dict(slim(yv[f"{arm}/H/{arm}_ep{e}_s{s}"]), pair=f"{e}:{s}") for e, s in HURDLE]
    G[f"{arm}_B_5"] = [dict(slim(yv[k]), pair=k.split("_ep")[1].replace("_s", ":")) for k in sorted(yv) if k.startswith(f"{arm}/B/")]
# E0 T39 on the same 14 ep:seed (those that exist in T39)
t39p = {r["pair"]: r for r in G["E0_T39_48"]}
G["E0_T39_same_pairs"] = [t39p[f"{e}:{s}"] for e, s in HURDLE if f"{e}:{s}" in t39p]
res = {"groups": {k: {"summary": summ(v), "runs": v} for k, v in G.items()}, "metrics": M}
# per-pair side by side (14 pairs)
res["per_pair"] = []
for i, (e, s) in enumerate(HURDLE):
    row = {"pair": f"{e}:{s}"}
    for g in ("E0_Z5_14", "V4_H_14", "V5_H_14"):
        r = G[g][i]; row[g] = {k: r[k] for k in ("success", "release_mode", "release_with_cmd") + tuple(M)}
    res["per_pair"].append(row)
pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
for g, v in res["groups"].items():
    s = v["summary"]; print(g, "n", s["n_runs"], "S/G/P", s["success"], s["grasp"], s["place"], "rel", s["n_release"], s["release_mode"], "gap_med", s["median_all_release"]["release_apple_floor_gap_mm"])
