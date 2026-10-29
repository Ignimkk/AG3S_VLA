"""T39 S2-0 dry check: does p/analyze_t39.py reproduce the recorded T37 E3b (and T28 E0_rec) numbers?
references: T37/p/E3b_summary.json (summarize_e3.py), T37/geom/a/settle.json (contact z), T28/p2r/E0_rec_summary.json.
usage: dry_check.py -> dry/dry_check.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
V = pathlib.Path(_ag3s_workspace_path('outputs/verify'))
D = json.loads((V / "T39/dry/dry_T37E3b_T28E0.json").read_text())
ref3 = json.loads((V / "T37/p/E3b_summary.json").read_text())["runs"]
ref0 = json.loads((V / "T28/p2r/E0_rec_summary.json").read_text())["runs"]
st = json.loads((V / "T37/geom/a/settle.json").read_text())["runs"]
rows, ok = [], True
for name, r in D["runs"].items():
    run = name.split("/")[1]
    ref = ref3.get(run) if r["cond"] == "E3b" else ref0.get(run)
    e = {"run": name, "success": [r["success"], ref["success"]], "grasp_t": [r["grasp_t"], ref["grasp_t"]],
         "place_t": [r["place_t"], ref["place_t"]], "apple_dz_max_mm": [r["apple_dz_max_mm"], ref["apple_dz_max_mm"]]}
    if r["cond"] == "E3b":
        e["n_hold_chunks"] = [r["n_hold_chunks"], ref["n_hold_chunks"]]
        s = st.get(name)
        e["contact_mid_z_mm"] = [(r.get("contact") or {}).get("contact_mid_z_mm"), s["contact_mid_rel_apple_mm"][2] if s else None]
        e["contact_lost_t"] = [(r.get("contact") or {}).get("contact_lost_t"), s["contact_lost_t"] if s else None]
        e["server_segment_n"] = (r["server_segment"] or {}).get("n"); e["server_refined_mismatch"] = (r["server_segment"] or {}).get("n_refined_mismatch_gt_1e-5")
        e["server_time_median_excl_first"] = {k: [(r["time_ms_excl_first"]["server_" + k] or {}).get("median"),
                                                  ((ref["server_time_ms_excl_first"] or {}).get(k) or {}).get("median")] for k in ("infer", "ag3s", "trajopt", "total")}
    eq = all((v[0] == v[1]) if not isinstance(v[0], float) else abs(v[0] - v[1]) < 1e-9
             for k, v in e.items() if isinstance(v, list) and len(v) == 2 and v[1] is not None)
    tm = e.get("server_time_median_excl_first", {})
    eq = eq and all(abs(a - b) < 1e-9 for a, b in tm.values() if a is not None and b is not None)
    e["all_equal"] = eq; ok = ok and eq; rows.append(e)
out = {"definitions": __doc__, "rows": rows, "all_equal": ok,
       "E3b": {"n_success": D["condition_summary"]["E3b"]["n_success"], "n_runs": D["condition_summary"]["E3b"]["n_runs"],
               "n_grasp": D["condition_summary"]["E3b"]["n_grasp"]},
       "E0": {"n_success": D["condition_summary"]["E0"]["n_success"], "n_runs": D["condition_summary"]["E0"]["n_runs"],
              "n_grasp": D["condition_summary"]["E0"]["n_grasp"]},
       "T37_E3b_reference": {"n_success": sum(v["success"] for v in ref3.values()), "n_runs": len(ref3), "n_grasp": sum(v["grasp"] for v in ref3.values())},
       "paired_2x2_success_codepath_only": D["paired_2x2"]["success"]}
(V / "T39/dry/dry_check.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "rows" and k != "definitions"}, indent=1))
for e in rows:
    print(e["run"], e["all_equal"], e["success"], e["grasp_t"], e["place_t"], e.get("contact_mid_z_mm"), e.get("server_segment_n"), e.get("server_refined_mismatch"))
