"""T43 Z3V extra aggregation: obstacle-stop events + hurdle contact rows (analysis hurdle.contact_rows: per set Af fingers / Ae palm / B apple / C other robot, rows with
hurdle distance <= 0, first_t) per run V5 vs V5D, relative to detach_t (aggregate.json). usage: aggregate_extra.py (updates aggregate.json key 'stops_and_contacts')"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Z3V'))
agg = json.load(open(T / "aggregate.json")); st = json.load(open(T / "raw/stops.json"))
A5 = json.load(open(_ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')))["runs"]; AD = json.load(open(T / "analysis/analysis.json"))["runs"]
out = {}
for n, v in agg["per_run"].items():
    stg, rest = n.split(":"); ep, seed = rest.split("_s")
    row = {}
    for arm, A in (("V5", A5), ("V5D", AD)):
        k = next(k for k in A if k.startswith(arm + "/") and k.split("/")[1] == stg and f"_ep{ep}_s{seed}" in k)
        h = A[k].get("hurdle") or {}
        s = st[f"{arm}/{stg}:{ep}_s{seed}"]
        row[arm] = {"stop": s["stop"], "stop_clearance_m": s["clearance_m"], "n_planning": s["n_planning"], "n_control": s["n_control"], "detach_t": v[arm]["detach_t"],
                    "hurdle_contact_rows": h.get("contact_rows"), "n_hold": v[arm]["n_hold"], "hold_per_planning_row": v[arm]["n_hold"] / max(1, s["n_planning"])}
    out[n] = row
agg["stops_and_contacts"] = out
agg["stops_summary"] = {arm: {"n_runs_stop": sum(1 for r in out.values() if r[arm]["stop"]), "n_runs_stop_H": sum(1 for k, r in out.items() if r[arm]["stop"] and k.startswith("H:")),
                              "sum_planning_rows": sum(r[arm]["n_planning"] for r in out.values()), "sum_hold": sum(r[arm]["n_hold"] for r in out.values())} for arm in ("V5", "V5D")}
(T / "aggregate.json").write_text(json.dumps(agg, indent=1, default=float))
print(json.dumps(agg["stops_summary"]))
for n, r in out.items():
    print(n, {a: (r[a]["stop"], r[a]["n_control"], r[a]["detach_t"], {k: (x["n_rows_le_0"], x["first_t"]) for k, x in (r[a]["hurdle_contact_rows"] or {}).items() if x["n_rows_le_0"]}) for a in ("V5", "V5D")})
