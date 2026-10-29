"""T43W final aggregation (verifier): numbers from final/numbers_main.json (analysis/analysis.json) and final/numbers_blind.json (analysis_blind/analysis.json).
 - arms table (hurdle 14 success / grasp / place; no-obstacle B for V3, V4, V5)
 - pairs V5 vs V4, V5B vs V5S (2x2 + McNemar exact two-sided), V5S vs V5 (counts only; the two arms differ in code snapshot)
 - V5 TO median / p90 (pooled, first chunk of every run excluded), HOLD chunk counts (incl. trailing), 1834_s18341 held-row violation chunks
 - focus (closing chunks, pooled over H runs): carve-out ball radius (mm), outward_mm (> 0 = pushed away from apple), n chunks
 - failure stage per failed hurdle run. Definitions (server grasp.state per chunk + analyze_e0 grasp / place):
     'before_pick' : grasp False and no 'closing' chunk;  'grasp_closing_only' : grasp False, closing chunks, no attached chunk;
     'grasp_attached_not_lifted' : grasp False, attached chunks > 0 (apple never >= initial + 50 mm);
     'carry' : grasp True, place False, state 'placed' never reached;  'place' : grasp True, place False, state 'placed' reached.
usage: final_agg.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, math
import numpy as np
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/W'))
sys.path.insert(0, str(V / "p"))
from partial_w import pairs, st  # noqa: E402
M = json.loads((V / "final/numbers_main.json").read_text()); B = json.loads((V / "final/numbers_blind.json").read_text())
AM = json.loads((V / "analysis/analysis.json").read_text())["runs"]; AB = json.loads((V / "analysis_blind/analysis.json").read_text())["runs"]
runs = dict(M["per_run"]); runs.update(B["per_run"]); AR = dict(AM); AR.update(AB)
S = dict(M["summary_per_arm"]); S.update(B["summary_per_arm"])
out = {"arms": {}, "pairs": {}, "focus": {}, "fail_stage": {}, "V5_detail": {}}
for arm in ("E0", "V1", "V3", "V4", "V5", "V5S", "V5B"):
    s = S[f"{arm}/H"]; out["arms"][arm] = {"H": {k: s[k] for k in ("n", "success", "grasp", "place")}}
    if f"{arm}/B" in S:
        sb = S[f"{arm}/B"]; out["arms"][arm]["B_no_obstacle"] = {k: sb[k] for k in ("n", "success", "grasp", "place")}
for a, b in (("V5", "V4"), ("V5B", "V5S"), ("V5S", "V5"), ("V5B", "V5")):
    out["pairs"][f"{a}_vs_{b}"] = pairs(runs, a, b)
def fl(arm, stage="H"):
    return [(k, v) for k, v in runs.items() if v["arm"] == arm and v["stage"] == stage]
for arm in ("V4", "V5", "V5S", "V5B"):
    cr, ow, nclo = [], [], 0
    for k, v in fl(arm):
        for c in AR[k]["server_chunks"]:
            if c["grasp_state"] == "closing" and not c["attached"]:
                if c.get("carve_ball_radius_mm") is not None: cr.append(c["carve_ball_radius_mm"])
                if c.get("closing_tip"): ow.append(c["closing_tip"]["outward_mm"])
    out["focus"][arm] = {"closing_chunks_with_carve_radius": len(cr), "carve_radius_mm": st(cr), "closing_chunks_with_outward": len(ow), "outward_mm": st(ow),
                         "n_outward_gt_2mm": int(sum(1 for x in ow if x > 2.0)), "n_outward_lt_0": int(sum(1 for x in ow if x < 0)),
                         "per_run_carve_median_mm": {f'{v["episode"]}_s{v["seed"]}': ((v.get("closing") or {}).get("carve_ball_radius_mm") or {}).get("median") for k, v in fl(arm)},
                         "per_run_outward_max_mm": {f'{v["episode"]}_s{v["seed"]}': (v.get("closing") or {}).get("outward_mm_max") for k, v in fl(arm)}}
for arm in ("V4", "V5", "V5S", "V5B"):
    rs = fl(arm); st_ = S[f"{arm}/H"]
    out["V5_detail"][arm] = {"to_solve_ms_median": st_["to_solve_ms_pooled"]["median"], "to_solve_ms_p90": st_["to_solve_ms_pooled"]["p90"], "to_n_chunks": st_["to_solve_ms_pooled"]["n"],
        "hold_chunks": st_["hold_chunks"], "runs_with_hold": st_["runs_with_hold"], "trailing_hold_chunks_per_run": st_["trailing_hold_chunks_per_run"],
        "runs_with_trailing_hold": sum(1 for x in st_["trailing_hold_chunks_per_run"].values() if x > 0),
        "trailing_hold_chunks_total": sum(st_["trailing_hold_chunks_per_run"].values()),
        "hold_kind_counts": st_["hold_kind_counts"], "held_row_violation_chunks": st_["held_row_violation_chunks"], "runs_with_held_row_violation": st_["runs_with_held_row_violation"],
        "held_row_violation_chunks_per_run": {f'{v["episode"]}_s{v["seed"]}': len(v.get("held_row_violation_chunks") or []) for k, v in rs},
        "held_row_violation_min_clearance_mm_per_run": {f'{v["episode"]}_s{v["seed"]}': min([c["clearance_mm"] for c in (v.get("held_row_violation_chunks") or [])], default=None) for k, v in rs},
        "runs_client_robot_collision": st_["runs_client_robot_collision"], "runs_client_payload_collision": st_["runs_client_payload_collision"], "runs_obstacle_stop": st_["runs_obstacle_stop"],
        "round_trip_ms_median": st_["round_trip_ms_pooled"]["median"], "time_budget_hit_count": st_["time_budget_hit_count"]}
    if arm in ("V4", "V5"):
        sb = S.get(f"{arm}/B")
        out["V5_detail"][arm]["B"] = {"to_solve_ms_median": sb["to_solve_ms_pooled"]["median"], "to_solve_ms_p90": sb["to_solve_ms_pooled"]["p90"], "hold_chunks": sb["hold_chunks"],
                                      "runs_with_hold": sb["runs_with_hold"], "trailing_hold_chunks_per_run": sb["trailing_hold_chunks_per_run"]}
    ev = {"1834_s18341": [dict(seq=c["seq"], clearance_mm=c["clearance_mm"]) for c in (runs.get(f"{arm}/H/{arm}_ep1834_s18341", {}).get("held_row_violation_chunks") or [])]}
    out["V5_detail"][arm]["viol_1834_s18341_chunk_count"] = len(ev["1834_s18341"])
for arm in ("V5", "V5S", "V5B", "V4"):
    d = {"before_pick": [], "grasp_closing_only": [], "grasp_attached_not_lifted": [], "carry": [], "place": []}
    for k, v in fl(arm):
        if v["success"]: continue
        gs = v["grasp_state_counts"] or {}
        if not v["grasp"]:
            key = "before_pick" if not gs.get("closing") else ("grasp_attached_not_lifted" if v["attached_chunks"] else "grasp_closing_only")
        else:
            key = "place" if gs.get("placed") else "carry"
        d[key].append(f'{v["episode"]}_s{v["seed"]}')
    out["fail_stage"][arm] = {"counts": {k: len(x) for k, x in d.items()}, "runs": d, "n_failed": sum(len(x) for x in d.values())}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
