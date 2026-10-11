"""Builds AG3S/ag3s/docs/handoff/T43V7g.verify.json from V7g/agg.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, glob, collections
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V7g')); a = json.load(open(V / "agg.json")); an = json.load(open(V / "analysis/analysis.json"))["runs"]
gpu = {}
for d in sorted(glob.glob(str(V / "runs/V7g/server_*"))):
    d = pathlib.Path(d); pk = max(int(l.split()[1]) for l in open(d / "gpu_samples.txt") if len(l.split()) == 2)
    gpu[d.name] = {"gpu_total_used_peak_mib": pk, "gpu_used_before_mib": int(open(d / "gpu_used_before_mib").read())}
stops = {k.split("/")[-1]: r["n_control"] for k, r in an.items() if r["n_control"] < 600}
ra = {x: {k: a["attach_radii"][x][k] for k in ("n_attach_runs", "body_parent_max_radius_mm", "body_parent_all_radii_mm", "body_cover_modes")} for x in ("V5", "V5D", "V7g")}
cap = [{"run": r["run"], "body_parent_radii_mm": r["body_parent_radii_mm"], "fit_radius_mm": r["fit_radius_mm"], "cap_applied": (r["cap"] or {}).get("body_cover_cap", {}).get("applied"),
        "radius_cap_mm": (r["cap"] or {}).get("body_cover_cap", {}).get("radius_cap_mm"), "n_radius_capped": (r["cap"] or {}).get("body_cover_cap", {}).get("n_radius_capped"),
        "n_query_before": (r["cap"] or {}).get("body_cover_cap", {}).get("n_before"), "n_query_after": (r["cap"] or {}).get("body_cover_cap", {}).get("n_after"),
        "record_world_radii_mm_pre_cap": r["record_world_radii_mm_pre_cap"]} for r in a["attach_radii"]["V7g"]["per_run"] if r.get("t") is not None]
v5r = [{"run": r["run"], "body_parent_radii_mm": r["body_parent_radii_mm"]} for r in a["attach_radii"]["V5"]["per_run"] if r.get("t") is not None]
hold = {x: {k: a["hold"][x][k] for k in ("total", "trailing", "runs_with_hold", "runs_with_trailing", "by_category", "by_target", "by_phase", "cells", "by_stage_total", "hold_kinds", "robot_total", "n_none_hurdle_le_50mm")} for x in ("V5", "V5D", "V7g")}
ft = a["follow_to"]
refused = [{"run": k, "refused": v["refused"]} for k, v in ft["per_run"].items() if v["refused"]]
slip = {x: {"n_detach": a["slip"][x]["n_detach"], "n_slip_flag_true": a["slip"][x]["n_slip_flag_true"],
            "detach_minus_release_rows": [r["detach_minus_release"] for r in a["slip"][x]["per_run"] if r["detach_minus_release"] is not None],
            "slip_runs": [r["run"] for r in a["slip"][x]["per_run"] if r["slip_flag"]]} for x in ("V5", "V5D", "V7g")}
pc = {x: {k: a["place_closed"][x][k] for k in ("n_release", "release_mode", "release_mode_in_success", "release_apple_floor_gap_mm", "release_apple_centre_above_floor_mm",
                                               "release_apple_floor_gap_mm_open_cmd", "release_apple_floor_gap_mm_closed_cmd", "release_v_vert_mm_s", "n_runs_apple_crate_contact", "n_runs_hand_crate_contact")} for x in ("V5", "V5D", "V7g")}
st = a["subtask"]
vk = collections.Counter()
for r in an.values():
    for k, v in r["verdict_kind_counts"].items(): vk[k] += v
out = {"_comment": "ag3s-verifier T43 V7g. Numbers and paths only; per-run tables in outputs/verify/T43/V7g/agg.json.", "step": "T43V7g", "date": "2026-10-06",
       "runs": [{"id": "V7g", "command": "outputs/verify/T43/V7g/p/launch_v7g.sh (2 chains, ports 8286 / 8287, EST_GB 28)", "venv": ".venv-openpi-live",
                 "server_flags": "V5 flags of T43/W/p5/start_server.sh minus --no-subtask-gate plus --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3",
                 "runs_dir": "outputs/verify/T43/V7g/runs/V7g", "n_runs": 19, "n_done": 19, "n_retries": 0, "chain_rc": {"E_H": 0, "E_B": 0, "F_H": 0}, "refusals": 0,
                 "raw_output": "outputs/verify/T43/V7g/runs/V7g/chain_8286.log, chain_8287.log, runs/chainE.out, runs/chainF.out, runs/launch.out", "gpu": gpu,
                 "gate_on_check": {"server_log_line": "subtask label: probe kv_L4_v1 (shipped asset) · gate ON", "cmdline_has_no_subtask_gate": False,
                                   "identity_line": "closing_keep_geometry=True, held_cover_cap=0.01, held_slip_detach=True, hold_follow_to=3, latched_identity_hold=True, supported_max_bottom=0.06",
                                   "enforced_in": "p/run_chain.sh (REFUSED otherwise)", "per_chunk_subtask_gate_true": st["gate"]}}],
       "code_state": {"repo": "benchmark", "head": "f2f335b", "snapshot": "${AG3S_PROJECT_ROOT}-t43v7 (outputs/verify/T43/V7g/snapshot.txt)", "code_md5_baseline": "outputs/verify/T43/V7g/p/code_md5_v7g.txt (405 lines, == at every chain preflight)",
                      "diff_vs_work_t43w3": ["ag3s/config.py", "ag3s/robot_models/held_object.py", "ag3s/runtime/pipeline.py", "trajopt/grasp_latch.py", "trajopt/safe_policy.py", "trajopt/serve_safe.py", "trajopt/wire.py"],
                      "with_overrides_collision_backend": "esdf", "new_flags_roundtrip": "outputs/verify/T43/V7g/with_overrides_check.txt (held_slip_detach True, held_cover_cap 0.01, hold_follow_to 3, to_dict equal)", "root_tests_commit": "0811598"},
       "baseline_check": {"matched": None, "note": "regression-baseline (.venv-ag3s esdf_rollout) not run: .venv-ag3s is excluded for this job. Guards run instead: code md5 == baseline at every preflight, import check under snapshot, autotune off, esdf_stats present after every run (no REFUSED lines), server log gate ON + held_cover_cap=0.01 + hold_follow_to=3 + held_slip_detach=True."},
       "numbers": {"runs": {"n_done": 19, "n_retries": 0, "n_full_600": 19 - len(stops), "n_obstacle_stop": len(stops), "obstacle_stop_n_control": stops},
                   "sgp": {"arms": a["sgp"]["arms"], "pairs": a["sgp"]["pairs"], "flips_old_vs_new_rule": a["sgp"]["flips"]},
                   "hold": hold, "follow_to": {"decision_counts": ft["decision_counts"], "refused_counts": ft["refused_counts"], "n_executed": ft["n_executed"], "n_runs_with_executed": ft["n_runs_with_executed"],
                                               "executed_chunks": ft["executed"], "refused_by_run": refused, "limit": 3},
                   "attach_radii": ra, "attach_cap_V7g": cap, "attach_query_radii_V5": v5r, "slip_detach": slip,
                   "subtask": {"n_chunks": st["n_chunks"], "label": st["label"], "gate": st["gate"], "blocked": st["blocked"], "would_block": st["would_block"], "decision_reason_V7g": st["decision_reason"],
                               "decision_reason_V5": st["decision_reason_V5"], "decision_reason_V5D": st["decision_reason_V5D"], "released_true_chunks": st["released_true_chunks"], "verdict_kind_counts_V7g": dict(vk)},
                   "place_closed_release": pc, "place_t": a["place_t"], "wall_s_median": a["run_counts"]},
       "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43v7g-scene.png", "kind": "scene"}, {"path": "AG3S/ag3s/docs/figures/t43/t43v7g-hold-breakdown.png", "kind": "graph"},
                   {"path": "AG3S/ag3s/docs/figures/t43/t43v7g-pairs-table.png", "kind": "table"}],
       "scripts": ["p/launch_v7g.sh", "p/run_chain.sh", "p/start_server.sh", "p/analyze_v7g.py", "p/z4_v7g.py", "p/y2_v7g.py", "p/agg_v7g.py", "p/fig_v7g.py", "p/build_verify_v7g.py", "p/with_overrides_check.py"],
       "raw": {"agg": "outputs/verify/T43/V7g/agg.json", "z4": "outputs/verify/T43/V7g/z4.json", "analysis": "outputs/verify/T43/V7g/analysis/analysis.json", "replay_V7g": "outputs/verify/T43/V7g/replay",
               "replay_V5D": "outputs/verify/T43/V7g/replay_V5D", "y2": "outputs/verify/T43/V7g/y2_{V5,V5D,V7g}.json", "snapshot": "outputs/verify/T43/V7g/snapshot.txt"},
       "verdict": "19/19 runs done, 0 retries, chain rc 0, gate ON in server log; guards matched. (facts only; no interpretation)",
       "not_measured": ["regression-baseline esdf_rollout (.venv-ag3s excluded)", "true crate / hurdle distance of executed follow_to chunks: no follow_to chunk was executed (n_executed 0)", "V5 / V5D / V7g come from different server sessions; first-chunk JIT time not cited"]}
p = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V7g.verify.json')); p.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False)); print(p, len(p.read_text()), gpu, stops)
