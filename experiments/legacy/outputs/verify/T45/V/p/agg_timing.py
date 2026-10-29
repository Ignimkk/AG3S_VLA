"""Aggregate raw/timing/*.timing.json -> agg_timing.json (numbers only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, sys
import numpy as np

D = _ag3s_workspace_path('outputs/verify/T45/V')
RAW = sys.argv[1] if len(sys.argv) > 1 else "timing"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
files = sorted(glob.glob(f"{D}/raw/{RAW}/*.timing.json"))
V = ("base", "PC")
rows = {v: [] for v in V}
per = {}
rec_to = []
for f in files:
    d = json.load(open(f))
    first = True
    for c in d["chunks"]:
        if "skipped" in c:
            first = True
            continue
        if first:            # first chunk of each run excluded (OSQP setup / warm-up), as tp_timing
            first = False
            continue
        rec_to.append(c["recorded_to_ms"])
        for v in V:
            r = c["variants"][v]
            rows[v].append(dict(run=d["name"], seq=c["seq"], held=c["held"], state=c["grasp_state"], load1=c["load1"],
                                wall=r["wall_ms"], cpu=r["cpu_ms"], sqp=r["sqp_iterations"], qp=r["qp_iterations"],
                                status=r["status"], budget_hit=bool(r["time_budget_hit"]),
                                core_rows=r["counts"].get("core_rows", 0), core_calls=r["counts"].get("core_calls", 0),
                                field_points=r["counts"].get("field_points", 0),
                                stages=r["stages"], nsph=d["n_spheres"][v]))
    per[d["name"]] = {"n_spheres": d["n_spheres"], "load_before": d["load_before"].get("loadavg"),
                      "load_after": d["load_after"].get("loadavg"), "gpu_before": d["gpu_before"], "gpu_after": d["gpu_after"]}

def q(a, p):
    return float(np.percentile(np.asarray(a, float), p))

def summ(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "median": float(np.median(x)), "mean": float(x.mean()), "p90": q(x, 90), "p95": q(x, 95),
            "p99": q(x, 99), "max": float(x.max())}

out = {"n_runs": len(files), "definitions": "first chunk of every run excluded; wall = server-equivalent (tp_timing: harness label time removed, server label read charged); "
       "variants solved in alternating order per chunk, server mode (one optimizer per variant per run), --field cuda, recorded sqp.time_budget_ms (50 ms); "
       "core_rows = sum of H*Q over computed _esdf_clearance_core calls per solve (memo hits not counted); field_points = points passed to CuroboEsdfField reads per solve (outermost call only)."}
for v in V:
    R = rows[v]
    out[v] = {"wall_ms": summ([r["wall"] for r in R]), "cpu_ms": summ([r["cpu"] for r in R]),
              "sqp_iterations": summ([r["sqp"] for r in R]), "qp_iterations": summ([r["qp"] for r in R]),
              "core_rows": summ([r["core_rows"] for r in R]), "core_calls": summ([r["core_calls"] for r in R]),
              "field_points": summ([r["field_points"] for r in R]),
              "rows_per_eval_S_x_H16": int(R[0]["nsph"] * 16),
              "n_spheres": int(R[0]["nsph"]),
              "time_budget_hit": int(sum(r["budget_hit"] for r in R)),
              "status_counts": {s: int(sum(1 for r in R if r["status"] == s)) for s in sorted({r["status"] for r in R})},
              "stage_median_ms": {k: float(np.median([r["stages"].get(k, 0.0) for r in R])) for k in ("fk", "esdf", "clearance", "linearize", "qp_build", "osqp", "objective", "other")},
              "stage_mean_ms": {k: float(np.mean([r["stages"].get(k, 0.0) for r in R])) for k in ("fk", "esdf", "clearance", "linearize", "qp_build", "osqp", "objective", "other")}}
out["recorded_C1a_to_solve_ms_excl_first"] = summ([x for x in rec_to if x is not None])
dw = np.asarray([b["wall"] - a["wall"] for a, b in zip(rows["base"], rows["PC"])])
dc = np.asarray([b["cpu"] - a["cpu"] for a, b in zip(rows["base"], rows["PC"])])
out["paired"] = {"n": int(len(dw)), "wall_diff_ms": summ(dw), "cpu_diff_ms": summ(dc),
                 "frac_PC_slower_wall": float((dw > 0).mean()),
                 "p90_diff_ms": out["PC"]["wall_ms"]["p90"] - out["base"]["wall_ms"]["p90"],
                 "median_diff_ms": out["PC"]["wall_ms"]["median"] - out["base"]["wall_ms"]["median"],
                 "cpu_p90_diff_ms": out["PC"]["cpu_ms"]["p90"] - out["base"]["cpu_ms"]["p90"],
                 "ratio_p90": out["PC"]["wall_ms"]["p90"] / out["base"]["wall_ms"]["p90"],
                 "ratio_median": out["PC"]["wall_ms"]["median"] / out["base"]["wall_ms"]["median"],
                 "ratio_core_rows_mean": out["PC"]["core_rows"]["mean"] / out["base"]["core_rows"]["mean"],
                 "ratio_field_points_mean": out["PC"]["field_points"]["mean"] / out["base"]["field_points"]["mean"]}
out["reference"] = {"V11AB_TP_gpu_median_ms": 38.9, "V11AB_TP_gpu_p90_ms": 69.2, "src": "T43TP.impl.md section 1 (load1 median 32)"}
out["PC_p90_minus_TP_p90_ms"] = out["PC"]["wall_ms"]["p90"] - 69.2
out["base_p90_minus_TP_p90_ms"] = out["base"]["wall_ms"]["p90"] - 69.2
out["PC_p90_minus_recorded_p90_ms"] = out["PC"]["wall_ms"]["p90"] - out["recorded_C1a_to_solve_ms_excl_first"]["p90"]
out["base_p90_minus_recorded_p90_ms"] = out["base"]["wall_ms"]["p90"] - out["recorded_C1a_to_solve_ms_excl_first"]["p90"]
L = np.asarray([r["load1"] for r in rows["base"]])
out["load1_during"] = summ(L)
# subsets: pre-grasp 8 chunks handled in agg_cmp (needs close_seq); here per-state
bys = {}
for st in sorted({r["state"] for r in rows["base"] if r["state"]}):
    idx = [i for i, r in enumerate(rows["base"]) if r["state"] == st]
    bys[st] = {v: {"n": len(idx), "median": float(np.median([rows[v][i]["wall"] for i in idx])), "p90": q([rows[v][i]["wall"] for i in idx], 90)} for v in V}
out["by_grasp_state"] = bys
# held vs not
for tag, flag in (("held", True), ("free", False)):
    idx = [i for i, r in enumerate(rows["base"]) if r["held"] == flag]
    out[f"held_{tag}"] = {v: {"n": len(idx), "median": float(np.median([rows[v][i]["wall"] for i in idx])), "p90": q([rows[v][i]["wall"] for i in idx], 90)} for v in V}
out["per_run"] = per
json.dump(out, open(f"{D}/agg_timing{TAG}.json", "w"), indent=1)
json.dump(rows, open(f"{D}/agg_timing_rows{TAG}.json", "w"))
print(json.dumps({k: out[k] for k in ("n_runs", "base", "PC", "paired", "recorded_C1a_to_solve_ms_excl_first", "load1_during", "PC_p90_minus_TP_p90_ms", "PC_p90_minus_recorded_p90_ms") if k in out}, indent=1)[:6000])
