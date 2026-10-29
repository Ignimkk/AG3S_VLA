"""Builds AG3S/ag3s/docs/handoff/T43V8.verify.json from V8/agg.json, ct.json, extra.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V8')); a = json.load(open(V / "agg.json")); ex = json.load(open(V / "extra.json")); ct = json.load(open(V / "ct.json"))
hold = {x: {k: a["hold"][x][k] for k in ("total", "trailing", "runs_with_hold", "by_category", "by_target", "by_phase", "by_binding_link", "cells", "hold_kinds", "by_stage_total")} for x in ("V7g", "V8")}
hold["per_run_chunks_V8"] = {k: v["chunks"] for k, v in a["hold"]["V8"]["per_run"].items() if v["n_hold"]}
hold["per_run_chunks_V7g"] = {k: v["chunks"] for k, v in a["hold"]["V7g"]["per_run"].items() if v["n_hold"]}
hp = a["hold_pairs"]
ft = {x: {k: a["follow_to"][x][k] for k in ("decision_counts", "refused_counts", "n_executed", "n_runs_with_executed", "executed")} for x in ("V7g", "V8")}
ft["refused_by_run"] = {x: {k: v["refused"] for k, v in a["follow_to"][x]["per_run"].items() if v["refused"]} for x in ("V7g", "V8")}
tm = {x: {s: {k: a["timing"][x][s][k] for k in ("n_chunks_excl_first", "to_solve_ms", "verdict_trajopt_ms", "verdict_total_ms", "round_trip_ms", "round_trip_gt_533_3_n", "round_trip_n", "round_trip_gt_533_3_share", "time_budget_hit_count", "runs_with_time_budget_hit", "linearize_ms", "qp_ms", "check_ms")} for s in ("ALL", "H", "B")} for x in ("V7g", "V8")}
tm["per_run_to_solve_median"] = ex["per_run_to_solve_median"]; tm["loadavg_during_V8"] = ex["loadavg_during_V8"]
srv = ex["servers"]
out = {"_comment": "ag3s-verifier T43 V8. Numbers and paths only; tables in outputs/verify/T43/V8/{agg,ct,extra}.json.", "step": "T43V8", "date": "2026-10-06",
       "runs": [{"id": "V8", "command": "outputs/verify/T43/V8/p/launch_v8.sh (2 chains, ports 8286 / 8287, EST_GB 28)", "venv": ".venv-openpi-live",
                 "server_flags": "V7g flags (T43/V7g/p/start_server.sh) + --finger-cover both (overflow default 2.5 mm)", "runs_dir": "outputs/verify/T43/V8/runs/V8", "n_runs": 19, "n_done": 19,
                 "n_retries": sum(c["n_attempt2"] for c in ex["chains"].values()), "chain_rc": ex["chain_rc"], "chain_rc_lines": ex["chain_rc_lines"], "refusals": sum(len(c["refused"]) for c in ex["chains"].values()), "failed_dirs": ex["failed_dirs"],
                 "raw_output": "outputs/verify/T43/V8/runs/V8/chain_8286.log, chain_8287.log, runs/chainE.out, chainF.out, launch.out", "servers": srv, "chains": ex["chains"],
                 "load_samples": "outputs/verify/T43/V8/raw/load_samples.txt",
                 "gate_on_and_finger_cover_check": {"enforced_in": "p/run_chain.sh (REFUSED otherwise)", "gate_on_log_lines": {k: v["log_gate_on"] for k, v in srv.items()}, "finger_cover_log_lines": {k: v["log_finger_cover_line"] for k, v in srv.items()},
                                                    "n_constraint_spheres_in_records": {k: v["n_constraint_spheres_in_records"] for k, v in srv.items()}}}],
       "code_state": {"repo": "benchmark", "head": "786cf1b", "snapshot": "${AG3S_PROJECT_ROOT}-t43v8 (outputs/verify/T43/V8/snapshot.txt)", "code_md5_baseline": "outputs/verify/T43/V8/p/code_md5_v8.txt (405 lines, == at every chain preflight)",
                      "diff_vs_work_t43v7_py": ["ag3s/experiments/reports/grounding_report.py", "ag3s/experiments/sources/gripper_cover.py", "ag3s/robot_models/urdf_sphere_chain.py", "trajopt/serve_safe.py"], "pi05_trees": "diff -rq vs ${AG3S_PROJECT_ROOT}-t43w3/pi05_TO_hybrid empty",
                      "with_overrides_collision_backend": "esdf (outputs/verify/T43/V8/with_overrides_check.txt)"},
       "baseline_check": {"matched": None, "note": "regression-baseline (.venv-ag3s esdf_rollout) not run (.venv-ag3s excluded). Guards run: code md5 == baseline, import check under snapshot, autotune off, esdf_stats after every run, with_overrides collision_backend esdf, server log gate ON + finger cover 122 -> 270."},
       "numbers": {"runs": {"n_done": 19, "H": 14, "B": 5, "n_control_lt_600": ex["n_control_lt_600"], "wall_s_median": ex["wall_s_median"]},
                   "sgp_old_rule_by_stage": a["sgp_old_rule_by_stage"], "pairs_old_rule": a["pairs_old_rule"], "H_totals_by_rule": ex["H_totals"], "H_pairs_z4_and_ct": ex["H_pairs_z4_rule"],
                   "ct_arms": ct["arms"], "ct_pairs": ct["pairs"], "ct_crosscheck_counts": a["ct"]["crosscheck_counts"],
                   "finger_hurdle_contacts_by_phase": a["finger_contacts"], "ct_contact_table": ct["contact_table"], "obstacle_stop": a["obstacle_stop"],
                   "hold": hold, "hold_only_in_one_arm": {k: v for k, v in hp.items()}, "follow_to": ft, "timing": tm,
                   "verdict_kind_counts": {"V8": ex["verdict_kind_counts_V8"], "V7g": ex["verdict_kind_counts_V7g"]}, "hurdle_min_mm_H_per_run": a["hurdle_min_mm"]},
       "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43v8-finger-spheres-scene.png", "kind": "scene"}, {"path": "AG3S/ag3s/docs/figures/t43/t43v8-timing.png", "kind": "graph"},
                   {"path": "AG3S/ag3s/docs/figures/t43/t43v8-pairs-table.png", "kind": "table"}],
       "scripts": ["p/launch_v8.sh", "p/run_chain.sh", "p/start_server.sh", "p/run_client.sh", "p/analyze_v8.py", "p/run_replays_v8.sh", "p/agg_ct_v8.py", "p/z4_base_v8.py", "p/agg_v8.py", "p/extra_v8.py", "p/scene_probe.py", "p/fig_v8_scene.py", "p/fig_v8_graphs.py", "p/build_verify_v8.py"],
       "raw": {"agg": "outputs/verify/T43/V8/agg.json", "ct": "outputs/verify/T43/V8/ct.json", "extra": "outputs/verify/T43/V8/extra.json", "analysis": "outputs/verify/T43/V8/analysis/analysis.json", "replay": "outputs/verify/T43/V8/replay", "snapshot": "outputs/verify/T43/V8/snapshot.txt", "V7g_analysis": "outputs/verify/T43/V7g/analysis/analysis.json", "V7g_replay": "outputs/verify/T43/CT/replay"},
       "verdict": "19/19 runs done, 0 retries, chain rc 0, gate ON and finger cover 122 -> 270 in server log; guards matched. (facts only)",
       "not_measured": ["regression-baseline esdf_rollout (.venv-ag3s excluded)", "V7g and V8 come from different server sessions; first-chunk JIT time not cited; SQP iteration counts not cited (not a baseline)",
                        "Z4 / CT rules and hurdle contacts are H-stage only (stage B has no hurdle; no B replays)", "scene figure pose = planned row 0 of the chunk (record sphere_centres), not the recorded qpos at the contact row",
                        "pod working set peak during the V8 round %.1f GiB (other sessions included; per-run guard checked working set + 4 GiB <= 64 GiB before each client)" % ex["loadavg_during_V8"]["working_set_gib_max"]]}
p = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V8.verify.json')); assert not p.exists(); p.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False)); print(p, len(p.read_text()))
