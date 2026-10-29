"""T37 per-condition fragment for T37.verify.partial.json (numbers only). usage: python3 build_frag.py COND -> p/frag_<COND>.json
Sources: p/<COND>_summary.json (summarize_e3.py), p/<COND>_t37_extra.json, p/<COND>_tipdev.json, p/<COND>_revoke_detail.json,
geom/a/settle.json + geom/a/a_runs.json (T35-diag metric code), <COND>_rec/chain_<port>.log, concurrency_launch.txt."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections, re, ast
COND = sys.argv[1]
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T37'))
SU = json.loads((V / "p" / f"{COND}_summary.json").read_text())
EX = json.loads((V / "p" / f"{COND}_t37_extra.json").read_text())["runs"]
TD = json.loads((V / "p" / f"{COND}_tipdev.json").read_text())["runs"]
ST = json.loads((V / "geom/a/settle.json").read_text())["runs"]
AR = json.loads((V / "geom/a/a_runs.json").read_text())["runs"]
port = {"E3a": 8204, "E3b": 8206}[COND]
out = {"server_port": port, "chain_log": f"outputs/verify/T37/{COND}_rec/chain_{port}.log",
       "chain": (V / f"{COND}_rec" / f"chain_{port}.log").read_text().splitlines(),
       "concurrency": (V / f"{COND}_rec" / "concurrency_launch.txt").read_text().splitlines(),
       "python": {"server_client": ".venv-openpi-live", "analysis": "scratchpad venv-ag3s-local shim (analyze_e3_base, summarize_e3, t37_extra, revoke_detail, t37_tipdev, geom/a_runs, geom/a/settle)"},
       "raw": {k: f"outputs/verify/T37/{v}" for k, v in {"records": f"{COND}_rec/", "server_log": f"{COND}_rec/serve_safe_{port}.log",
               "analysis": f"p/{COND}_analysis.json", "summary": f"p/{COND}_summary.json", "extra": f"p/{COND}_t37_extra.json",
               "tipdev": f"p/{COND}_tipdev.json", "revoke_detail": f"p/{COND}_revoke_detail.json", "settle": "geom/a/settle.json",
               "a_runs": "geom/a/a_runs.json"}.items()},
       "runs": {}}
cs = SU["condition_summary"]
out["condition"] = {"n_runs": cs["n_runs"], "n_success": cs["n_success"], "n_grasp": cs["n_grasp"],
                    "n_hold_chunks_total": cs["n_hold_chunks_total"], "n_would_hold_chunks_total": cs["n_would_hold_chunks_total"],
                    "hold_kind_counts_total": cs["hold_kind_counts_total"], "would_hold_kind_counts_total": cs["would_hold_kind_counts_total"],
                    "comms_hold_total_from_log": cs["comms_hold_total_from_log"], "comms_ipc_planning_rows_total": cs["comms_ipc_planning_rows_total"],
                    "n_server_chunks": cs["n_server_chunks"], "n_server_chunks_paired": cs["n_server_chunks_paired"],
                    "server_time_ms_pooled_excl_first": cs["server_time_ms_pooled_excl_first"],
                    "n_gate_on_chunks_total": sum(TD[r]["n_gate_on"] for r in TD),
                    "n_closing_chunks_total": sum(TD[r]["n_closing"] for r in TD), "n_held_chunks_total": sum(TD[r]["n_held"] for r in TD),
                    "n_gate_on_not_closing_held": sum(1 for r in TD for c in TD[r]["chunks"] if c["continuity_off_by"] and c["state"] not in ("closing", "held")),
                    "n_closing_held_without_gate": sum(1 for r in TD for c in TD[r]["chunks"] if not c["continuity_off_by"] and c["state"] in ("closing", "held")),
                    "gate_on_reason_counts": dict(collections.Counter(c["continuity_off_by"] for r in TD for c in TD[r]["chunks"] if c["continuity_off_by"])),
                    "continuity_reason_counts_all_chunks": dict(collections.Counter(c["continuity_reason"] for r in TD for c in TD[r]["chunks"])),
                    "n_revoke": sum(len(EX[r]["revoke_t"]) for r in EX)}
gl = [l for l in (V / f"{COND}_rec" / f"serve_safe_{port}.log").read_text(errors="replace").splitlines() if "continuity term" in l]
out["condition"]["server_log_gate_lines"] = {"n_off_lines": sum("OFF" in l for l in gl), "n_back_on_lines": sum("back ON" in l for l in gl),
                                             "distinct": dict(collections.Counter(l.strip() for l in gl))}
for name in sorted(SU["runs"]):
    s = SU["runs"][name]; x = EX[name]; td = TD[name]; st = ST.get(f"{COND}_rec/{name}"); ar = AR.get(f"{COND}_rec/{name}") or {}
    lt = (V / f"{COND}_rec" / f"{name}.client.log").read_text(errors="replace")
    m1 = re.search(r"client_ipc_stats: (\{.*\})", lt); m2 = re.search(r"\[safe\] HOLD \(fixed\): (\{.*\})", lt); m3 = re.search(r"comms HOLD (\d+)", lt)
    ipc = ast.literal_eval(m1.group(1)) if m1 else None
    clog = {"client_ipc_stats": ipc, "hold_fixed_summary": ast.literal_eval(m2.group(1)) if m2 else None, "comms_hold_line": int(m3.group(1)) if m3 else None,
            "comms_timeout_stale_error": None if ipc is None else ipc.get("timeout", 0) + ipc.get("stale", 0) + ipc.get("error", 0),
            "source": f"outputs/verify/T37/{COND}_rec/{name}.client.log"}
    out["runs"][name] = {"client_log": clog,
        "success": s["success"], "grasp": s["grasp"], "grasp_t": s["grasp_t"], "place_t": s["place_t"], "release_t": x["release_t"],
        "apple_dz_max_mm": s["apple_dz_max_mm"], "t_first_close_cmd": s["t_first_close_cmd"], "n_close_cycles_applied_lt_0p5": s["n_close_cycles_applied_lt_0p5"],
        "min_grip_l_measured_open_frac": s["min_grip_l_measured_open_frac"], "mp4": s["mp4"],
        "gate": {"n_gate_on": td["n_gate_on"], "gate_on_t": td["gate_on_t"], "n_closing": td["n_closing"], "n_held": td["n_held"],
                 "gate_on_by_state": dict(collections.Counter(c["state"] for c in td["chunks"] if c["continuity_off_by"]))},
        "fingertip_dev_mm": {"closing_held_max": td["closing_held_tip_mm_max"], "closing_held_median": td["closing_held_tip_mm_median"],
                             "closing_max": td["closing_tip_mm_max"], "held_max": td["held_tip_mm_max"], "other_state_max": td["other_tip_mm_max"],
                             "n_closing_held_gt_1mm": td["n_closing_held_tip_gt_1mm"],
                             "closing_held_chunks_gt_1mm": [[c["t"], c["state"], round(c["fingertip_disp_mm_max"], 2), c["returned"]] for c in td["chunks"]
                                                           if c["state"] in ("closing", "held") and c["fingertip_disp_mm_max"] > 1.0]},
        "contact_at_settle": None if st is None else {"t_close": st["t_close"], "t_settle": st["t_settle"], "contact_mid_rel_apple_mm": st["contact_mid_rel_apple_mm"],
                                                      "contact_mid_z_mm": st["contact_mid_rel_apple_mm"][2], "open_settle": st["open_settle"],
                                                      "contact_lost_t": st["contact_lost_t"], "first_grasp_held": st["contact_lost_t"] is None,
                                                      "hold_min_open": st["hold_min_open"], "refined_minus_ref_body_mid_norm_mm": st["refined_minus_ref_body_mid_norm_mm"],
                                                      "ref_minus_measured_body_mid_norm_mm": st["ref_minus_measured_body_mid_norm_mm"], "n_close_cycles": ar.get("n_close_cycles")},
        "hold": {"n_hold_chunks": s["n_hold_chunks"], "hold_kind_counts": s["hold_kind_counts"], "n_would_hold_chunks": s["n_would_hold_chunks"],
                 "would_hold_kind_counts": s["would_hold_kind_counts"], "timeline": x["hold_timeline"], "hold_held_table": len(x["hold_held_table"]),
                 "comms_hold_from_log": s["comms_hold_from_log"], "comms_ipc_planning_rows": s["comms_ipc_planning_rows"], "ipc_counts": s["ipc_counts"],
                 "hold_control_steps": s["hold_control_steps"]},
        "latch": {"attach_t": x["attach_t"], "detach_t": x["detach_t"], "revoke_t": x["revoke_t"], "revoke_notes": x["revoke_notes"],
                  "first_t_by_state": s["latch_first_t_by_state"]},
        "to_returned_counts": x["returned_counts"], "refined_minus_reference_deg": x["refined_minus_reference_summary"],
        "n_manip_not_apple_pre_grasp": s["n_manip_not_apple_pre_grasp"], "n_worst_inside_apple": s["n_worst_inside_apple"],
        "n_invariant_violation": s["n_invariant_violation"],
    }
(V / "p" / f"frag_{COND}.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
out["condition"]["comms_timeout_stale_error_total"] = sum((r["client_log"]["comms_timeout_stale_error"] or 0) for r in out["runs"].values())
out["condition"]["hold_fixed_chunks_total"] = sum(((r["client_log"]["hold_fixed_summary"] or {}).get("hold_chunks") or 0) for r in out["runs"].values())
(V / "p" / f"frag_{COND}.json").write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False))
print(json.dumps(out["condition"], indent=1, default=float, ensure_ascii=False))
