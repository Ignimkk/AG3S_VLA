"""Build phase3.<COND> fragment: condition summary, per-run summary, compact per-chunk table, mp4 list."""
import json, sys
c = sys.argv[1]
A = json.load(open(f"${AG3S_PROJECT_ROOT}/outputs/verify/T28/p3/{c}_analysis.json"))
S = json.load(open(f"${AG3S_PROJECT_ROOT}/outputs/verify/T28/p3/{c}_summary.json"))
cols = ["t", "seq", "phase", "executed", "gate_kinds", "manip_id", "manip_state", "manip_nearest", "manip_to_apple_mm", "manip_narrowest_mm",
        "manip_admissible_reason", "dest_id", "exclusion_source", "ball_radius_mm", "ball_centre_nearest", "invariant_violation",
        "finger_model_left", "finger_measured_left", "worst_link", "worst_tier", "worst_obstacle_label", "worst_clearance_mm", "worst_nearest_body",
        "worst_apple_surface_mm", "worst_inside_apple", "refined_minus_reference_deg", "latch_state", "attach", "detach",
        "grip_cmd_refined_min", "grip_cmd_reference_min", "grip_applied_min", "grip_measured_min",
        "srv_infer_ms", "srv_ag3s_ms", "srv_esdf_ms", "srv_trajopt_ms", "srv_total_ms", "client_round_trip_ms", "ipc"]
runs = {}
for name, r in A["runs"].items():
    rows = []
    for e in r["chunks"]:
        g = e.get("gate") or {}
        ks = (g.get("would_hold_kinds") if c == "E3a" else (g.get("hold_kinds") or g.get("kinds"))) or []
        m = e.get("manipulated") or {}; w = e["worst"]; st = e.get("server_timing_ms") or {}; b = e.get("ball") or {}
        rows.append([e["t"], e["seq"], e["phase"], e["n_executed"] == e["n_control"], ks, m.get("id"), m.get("state"), m.get("nearest_centre"),
                     m.get("to_apple_mm"), m.get("narrowest_extent_mm"), m.get("admissible_reason"), (e.get("destination") or {}).get("id"),
                     (e.get("exclusion") or {}).get("source"), b.get("radius_mm"), b.get("centre_nearest"), e.get("invariant_violation"),
                     (e.get("finger_joints") or {}).get("model_left_norm"), (e.get("finger_joints") or {}).get("measured_left_norm_planning_row"),
                     w.get("link"), w.get("tier"), w.get("obstacle"), None if w.get("clearance_m") is None else 1000 * w["clearance_m"], w.get("nearest_body"),
                     w.get("apple_surface_mm"), w.get("inside_apple"), e.get("refined_minus_reference_deg"), (e.get("grasp_latch") or {}).get("state"),
                     (e.get("grasp_latch") or {}).get("attach"), (e.get("grasp_latch") or {}).get("detach"),
                     e.get("grip_l_cmd_refined_min_rows0_7"), e.get("grip_l_cmd_reference_min_rows0_7"), e.get("grip_l_applied_min"), e.get("grip_l_measured_min"),
                     st.get("infer"), st.get("ag3s"), (e.get("server_profile_ms") or {}).get("esdf"), st.get("trajopt"), st.get("total"),
                     e.get("client_policy_infer_ms"), e.get("ipc")])
    runs[name] = {"summary": S["runs"][name], "per_chunk_columns": cols, "per_chunk": rows}
frag = {"scripts": ["outputs/verify/T28/p3/run_chain_e3.sh", "start_server_e3.sh", "run_client_e3.sh", "analyze_e3.py", "summarize_e3.py", "build_frag_e3.py"],
        "raw": {"analysis": f"outputs/verify/T28/p3/{c}_analysis.json", "summary": f"outputs/verify/T28/p3/{c}_summary.json", "runs_dir": f"outputs/verify/T28/{c}_rec/"},
        "definitions": {"analyze": A["definitions"], "summarize": S["definitions"]},
        "condition_summary": S["condition_summary"], "runs": runs,
        "mp4": sorted(r["mp4"] for r in S["runs"].values())}
json.dump(frag, open(f"${AG3S_PROJECT_ROOT}/outputs/verify/T28/p3/frag_{c}.json", "w"), default=float)
print(c, "ok", len(runs))
