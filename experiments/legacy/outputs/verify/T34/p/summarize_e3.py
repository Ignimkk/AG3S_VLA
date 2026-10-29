"""Condition-level numbers from p3/<COND>_analysis.json (analyze_e3.py). usage: summarize_e3.py COND
Counts (definitions in analyze_e3.py docstring):
  hold chunk (E3b)        : planning row whose chunk was not executed (n_executed < n_control) — kinds from gate.hold_kinds
                            (fallback: gate.kinds) ; comms = ipc in {timeout, stale, error}
  would_hold chunk (E3a)  : gate.would_hold True — kinds from gate.would_hold_kinds
  manip != apple pre-grasp: phase pre_grasp and (manipulated None or its nearest object centre != apple)
  worst inside apple      : worst.inside_apple True
Timing: medians over chunks excluding each run's first chunk (JIT)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np

COND = sys.argv[1]
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T34/p'))
R = json.loads((V / f"{COND}_analysis.json").read_text())


def med(v):
    v = [x for x in v if x is not None]
    return None if not v else {"n": len(v), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(max(v))}


def cnt(items):
    out = {}
    for x in items:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


out = {"condition": COND, "definitions": __doc__, "runs": {}}
allc = []
for name, r in R["runs"].items():
    s = r["summary"]; ch = r["chunks"]
    allc += [(name, c) for c in ch]
    hold = [c for c in ch if c["n_executed"] < c["n_control"]]
    wh = [c for c in ch if (c.get("gate") or {}).get("would_hold")]
    def kinds(c, key):
        g = c.get("gate") or {}
        k = g.get(key) or g.get("kinds") or []
        if c.get("ipc") in ("timeout", "stale", "error"):
            k = ["comms:" + c["ipc"]]
        return k
    pre = [c for c in ch if c["phase"] == "pre_grasp"]
    attach_t = [c["t"] for c in ch if (c.get("grasp_latch") or {}).get("attach")]
    detach_t = [c["t"] for c in ch if (c.get("grasp_latch") or {}).get("detach")]
    latch_states = [(c["t"], (c.get("grasp_latch") or {}).get("state")) for c in ch]
    first_state = {}
    for t, st in latch_states:
        first_state.setdefault(str(st), t)
    fj = [c["finger_joints"]["abs_diff"] for c in ch if c.get("finger_joints") and c["finger_joints"]["abs_diff"] is not None]
    st = lambda key: [((c.get("server_timing_ms") or {}).get(key)) for c in ch[1:]]
    esdf = [((c.get("server_profile_ms") or {}).get("esdf")) for c in ch[1:]]
    rr = [c.get("refined_minus_reference_deg") for c in ch]
    dest = next(((c["t"], c["destination"]) for c in ch if c.get("destination")), None)
    run = {
        "episode": s["episode_index"], "mp4": r.get("mp4"), "success": s["success"], "grasp": s["grasp"], "place": s["place"],
        "grasp_t": s["grasp_t"], "grasp_t_control_rows": s["grasp_t_control_rows"], "place_t": s["place_t"],
        "apple_dz_max_mm": s["apple_dz_max_mm"], "t_apple_dz_max": s["t_apple_dz_max"],
        "t_first_close_cmd": s["t_first_close_cmd"], "min_grip_l_applied": s["min_grip_l_applied"],
        "min_grip_l_measured_open_frac": s["min_grip_l_measured_open_frac"], "n_close_cycles_applied_lt_0p5": s["n_close_cycles_applied_lt_0p5"],
        "min_finger_to_apple_surface_mm": s["min_finger_to_apple_surface_mm"],
        "hold_control_steps": s["hold_control_steps"], "comms_hold_from_log": s["client_log"].get("comms_hold_from_log"),
        "comms_ipc_planning_rows": s["comms_ipc_planning_rows"], "ipc_counts": s["ipc_counts"],
        "hold_summary_line": s["client_log"].get("hold_summary_line"),
        "n_chunks": len(ch), "n_paired_server": sum(1 for c in ch if "server_chunk" in c),
        "hold_chunks": [{"seq": c["seq"], "t": c["t"], "phase": c["phase"], "latch": (c.get("grasp_latch") or {}).get("state"),
                         "kinds": kinds(c, "hold_kinds"), "n_executed": c["n_executed"], "worst_link": c["worst"].get("link"),
                         "worst_nearest": c["worst"].get("nearest_body"), "worst_inside_apple": c["worst"].get("inside_apple"),
                         "worst_clearance_mm": None if c["worst"].get("clearance_m") is None else 1000 * c["worst"]["clearance_m"],
                         "grip_l_cmd_reference_min": c.get("grip_l_cmd_reference_min_rows0_7"), "hold_reason": c.get("hold_reason")} for c in hold],
        "n_hold_chunks": len(hold), "hold_kind_counts": cnt(k for c in hold for k in kinds(c, "hold_kinds")),
        "hold_phase_counts": cnt(c["phase"] for c in hold), "hold_latch_counts": cnt((c.get("grasp_latch") or {}).get("state") for c in hold),
        "n_would_hold_chunks": len(wh), "would_hold_kind_counts": cnt(k for c in wh for k in kinds(c, "would_hold_kinds")),
        "would_hold_phase_counts": cnt(c["phase"] for c in wh),
        "n_manip_not_apple_pre_grasp": sum(1 for c in pre if "server_chunk" in c and (c.get("manipulated") is None or c["manipulated"]["nearest_centre"] != "apple")),
        "manip_not_apple_pre_grasp_t": [c["t"] for c in pre if "server_chunk" in c and (c.get("manipulated") is None or c["manipulated"]["nearest_centre"] != "apple")],
        "n_pre_grasp_chunks": len(pre),
        "manipulated_ids": sorted({c["manipulated"]["id"] for c in ch if c.get("manipulated")}),
        "manipulated_to_apple_mm_pre_grasp": med([c["manipulated"]["to_apple_mm"] for c in pre if c.get("manipulated")]),
        "manipulated_narrowest_extent_mm": med([c["manipulated"]["narrowest_extent_mm"] for c in ch if c.get("manipulated")]),
        "n_worst_inside_apple": sum(1 for c in ch if c["worst"].get("inside_apple")),
        "worst_apple_surface_mm_when_nearest_apple": med([c["worst"].get("apple_surface_mm") for c in ch if c["worst"].get("nearest_body") == "apple"]),
        "worst_apple_surface_mm_min": min((c["worst"].get("apple_surface_mm") for c in ch if c["worst"].get("apple_surface_mm") is not None), default=None),
        "manipulated_label_breakdown": cnt(None if c.get("manipulated") is None else f'id{c["manipulated"]["id"]}:{c["manipulated"]["state"]}:{c["manipulated"]["nearest_centre"]}' for c in ch if "server_chunk" in c),
        "n_manip_to_apple_gt_40mm_pre_grasp": sum(1 for c in pre if c.get("manipulated") and c["manipulated"]["to_apple_mm"] > 40),
        "n_worst_inside_apple_pre_grasp": sum(1 for c in pre if c["worst"].get("inside_apple")),
        "worst_nearest_counts": cnt(c["worst"].get("nearest_body") for c in ch),
        "worst_link_counts": cnt(c["worst"].get("link") for c in ch), "worst_tier_counts": cnt(c["worst"].get("tier") for c in ch),
        "destination_first": dest, "exclusion_source_counts": cnt((c.get("exclusion") or {}).get("source") for c in ch),
        "n_invariant_violation": sum(1 for c in ch if c.get("invariant_violation")),
        "ball_radius_mm": med([c["ball"]["radius_mm"] for c in ch if c.get("ball")]),
        "ball_centre_nearest_counts": cnt(c["ball"]["centre_nearest"] for c in ch if c.get("ball")),
        "finger_joints_model_minus_measured_abs_max": max(fj) if fj else None,
        "finger_joints_source_counts": cnt((c.get("finger_joints") or {}).get("source") for c in ch),
        "attach_t": attach_t, "detach_t": detach_t, "latch_first_t_by_state": first_state,
        "refined_minus_reference_deg": {"max": max(x for x in rr if x is not None), "median": float(np.median([x for x in rr if x is not None])),
                                        "max_pre_grasp": max((c.get("refined_minus_reference_deg") or 0) for c in pre) if pre else None,
                                        "max_closing_latch": max(((c.get("refined_minus_reference_deg") or 0) for c in ch if (c.get("grasp_latch") or {}).get("state") == "closing"), default=None)},
        "server_time_ms_excl_first": {"infer": med(st("infer")), "ag3s": med(st("ag3s")), "esdf_in_ag3s": med(esdf), "trajopt": med(st("trajopt")), "total": med(st("total"))},
        "client_policy_infer_ms_excl_first": med([c.get("client_policy_infer_ms") for c in ch[1:]]),
        "n_time_budget_hit": sum(1 for c in ch if (c.get("to") or {}).get("time_budget_hit")),
    }
    out["runs"][name] = run
    print(name, {k: run[k] for k in ("success", "grasp_t", "place_t", "apple_dz_max_mm", "t_first_close_cmd", "min_grip_l_measured_open_frac",
                                     "n_hold_chunks", "hold_kind_counts", "n_would_hold_chunks", "would_hold_kind_counts", "n_manip_not_apple_pre_grasp",
                                     "n_worst_inside_apple", "attach_t", "detach_t", "comms_hold_from_log")}, flush=True)
runs = out["runs"].values()
out["condition_summary"] = {
    "n_runs": len(out["runs"]), "n_success": sum(r["success"] for r in runs), "n_grasp": sum(r["grasp"] for r in runs),
    "n_hold_chunks_total": sum(r["n_hold_chunks"] for r in runs), "n_would_hold_chunks_total": sum(r["n_would_hold_chunks"] for r in runs),
    "n_manip_not_apple_pre_grasp_total": sum(r["n_manip_not_apple_pre_grasp"] for r in runs),
    "n_worst_inside_apple_total": sum(r["n_worst_inside_apple"] for r in runs),
    "n_invariant_violation_total": sum(r["n_invariant_violation"] for r in runs),
    "comms_hold_total_from_log": sum((r["comms_hold_from_log"] or 0) for r in runs),
    "comms_ipc_planning_rows_total": sum(r["comms_ipc_planning_rows"] for r in runs),
    "hold_kind_counts_total": cnt(k for r in runs for h in r["hold_chunks"] for k in h["kinds"]),
    "hold_phase_counts_total": cnt(h["phase"] for r in runs for h in r["hold_chunks"]),
    "would_hold_kind_counts_total": {k: sum(r["would_hold_kind_counts"].get(k, 0) for r in runs) for k in {k for r in runs for k in r["would_hold_kind_counts"]}},
    "server_time_ms_pooled_excl_first": {key: med([((c.get("server_timing_ms") or {}).get(key)) for name, r in R["runs"].items() for c in r["chunks"][1:]])
                                         for key in ("infer", "ag3s", "trajopt", "total")},
    "esdf_ms_pooled_excl_first": med([((c.get("server_profile_ms") or {}).get("esdf")) for name, r in R["runs"].items() for c in r["chunks"][1:]]),
    "n_server_chunks": R["n_server_chunks"], "n_server_chunks_paired": R["n_server_chunks_paired"],
}
print(json.dumps(out["condition_summary"], indent=1, default=float))
(V / f"{COND}_summary.json").write_text(json.dumps(out, indent=1, default=float))
