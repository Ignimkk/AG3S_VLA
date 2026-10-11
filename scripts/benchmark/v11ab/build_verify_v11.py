"""Builds AG3S/ag3s/docs/handoff/T43<ARM>.verify.json (numbers and paths only) from V11/{agg,ct,extra,sharp,xm_summary}_<ARM>.json, baseline/base_curobo.json.
usage: V11_ARM=V11A build_verify_v11.py FIG_PNG_SCENE FIG_PNG_GRAPH FIG_PNG_TABLE"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, os, sys
ARM = os.environ.get("V11_ARM", "V11A"); V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11')); T = V.parent
a = json.load(open(V / f"agg_{ARM}.json")); ex = json.load(open(V / f"extra_{ARM}.json")); ct = json.load(open(V / f"ct_{ARM}.json"))
sh = json.load(open(V / f"sharp_{ARM}.json")); xs = json.load(open(V / f"xm_summary_{ARM}.json")); base = json.load(open(V / "baseline/base_curobo.json"))
fr = base["frames"]; scene_png, graph_png, table_png = sys.argv[1:4]
hold = {x: {k: a["hold"][x][k] for k in ("total", "trailing", "runs_with_hold", "by_category", "by_target", "by_phase", "by_binding_link", "cells", "hold_kinds", "by_stage_total")} for x in ("V8", ARM)}
hold["per_run_chunks_V8"] = {k: v["chunks"] for k, v in a["hold"]["V8"]["per_run"].items() if v["n_hold"]}
hold[f"per_run_chunks_{ARM}"] = {k: v["chunks"] for k, v in a["hold"][ARM]["per_run"].items() if v["n_hold"]}
hp = a["hold_pairs"]
ft = {x: {k: a["follow_to"][x][k] for k in ("decision_counts", "refused_counts", "n_executed", "n_runs_with_executed", "executed")} for x in ("V8", ARM)}
ft["refused_by_run"] = {x: {k: v["refused"] for k, v in a["follow_to"][x]["per_run"].items() if v["refused"]} for x in ("V8", ARM)}
tm = {x: {s: {k: a["timing"][x][s][k] for k in ("n_chunks_excl_first", "to_solve_ms", "verdict_trajopt_ms", "verdict_total_ms", "round_trip_ms", "round_trip_gt_533_3_n", "round_trip_n", "round_trip_gt_533_3_share",
                                                 "time_budget_hit_count", "runs_with_time_budget_hit", "linearize_ms", "qp_ms", "check_ms")} for s in ("ALL", "H", "B")} for x in ("V8", ARM)}
tm["per_run_to_solve_median"] = ex["per_run_to_solve_median"]; tm["raw_to_from_server_records"] = ex["raw_to"]
tm["loadavg_during_round"] = ex.get("loadavg_during_round"); tm["loadavg_during_V8_round"] = ex["loadavg_during_V8_round"]
cross = {arm: {g: {k: xs["arms"][arm][g].get(k) for k in ("n_chunks", "n_runs", "palm_D_max_mm", "n_deflected_gt5", "deflected_components_at_kstar_mm", "deflected_share", "deflected_dominant_axis",
                                                   "deflected_component_fraction_median", "executed_rows", "initial_source", "n_ref_violating", "hold_chunks", "binding_rows", "sqp", "max_step", "continuity_palm_mm")}
               for g in ("carry_apple", "carry_empty", "return")} for arm in ("V8", ARM)}
crossing = {arm: {"crossing_stats": xs["arms"][arm]["crossing"], "binding_at_palm_crossing": xs["arms"][arm]["binding_at_palm_crossing"],
                  "per_run": {k: {kk: v.get(kk) for kk in ("apple_cross", "tip_cross", "binding_at_palm_cross", "success", "grasp")} for k, v in xs["per_run"].items() if k.startswith(arm + "/")}} for arm in ("V8", ARM)}
srv = ex["servers"]
sharp = {arm: {s: {k: sh["arms"][arm][s][k] for k in ("n_runs", "n_boundaries_clean", "n_boundaries_excluded_hold", "acc_boundary_rad", "acc_inner_chunk_max_rad", "vel_boundary_ratio", "n_vel_boundary_gt1",
                                                       "palm_turn_deg", "palm_reversal_share", "palm_reversal_n", "palm_acc_boundary_mm")} for s in ("H", "B", "ALL")} for arm in ("V8", ARM)}
name = f"T43{ARM}"
out = {"_comment": f"ag3s-verifier T43 {ARM} (first online round of A -> A+B -> A+B+C). Numbers and paths only; tables in outputs/verify/T43/V11/{{agg,ct,extra,sharp,xm_summary}}_{ARM}.json.", "step": name, "date": "2026-10-07",
       "runs": [{"id": ARM, "command": "outputs/verify/T43/V11/p/launch_v11.sh " + ARM + " (2 chains, ports 8286 / 8287, EST_GB 28)", "venv": ".venv-openpi-live",
                 "server_flags": "exactly the V8 flags (T43/V8/p/start_server.sh; no --held-obstacle-margin, no --servo-model) + " + {"V11A": "--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4", "V11AB": "V11A + --w-deflection-rate 20", "V11ABC": "V11AB + --servo-accel relaxed --servo-accel-tolerance-deg 1 --servo-accel-weight 0.01"}[ARM],
                 "runs_dir": f"outputs/verify/T43/V11/runs/{ARM}", "n_runs": 19, "n_done": a["run_counts"][ARM]["n_runs"], "n_retries": sum(c["n_attempt2"] for c in ex["chains"].values()), "chain_rc_lines": ex["chain_rc_lines"],
                 "refusals": sum(len(c["refused"]) for c in ex["chains"].values()), "failed_dirs": ex["failed_dirs"],
                 "raw_output": f"outputs/verify/T43/V11/runs/{ARM}/chain_8286.log, chain_8287.log, runs/chainE_{ARM}.out, chainF_{ARM}.out, launch_{ARM}.out", "servers": srv, "chains": ex["chains"],
                 "load_samples": "outputs/verify/T43/V11/raw/load_samples.txt"}],
       "code_state": {"repo": "benchmark", "head": "3ca1b21", "snapshot": "${AG3S_PROJECT_ROOT}-t43v11 (outputs/verify/T43/V11/snapshot.txt)", "code_md5_baseline": "outputs/verify/T43/V11/p/code_md5_v11.txt (407 lines, == at every chain preflight)",
                      "diff_vs_work_t43v8_py": "12 changed + 2 new .py files == git diff 786cf1b 3ca1b21 (p/code_md5_diff_vs_v8.txt)", "pi05_trees": "diff -rq vs ${AG3S_PROJECT_ROOT}-t43w3/pi05_TO_hybrid empty",
                      "with_overrides_and_TA_round_trip": "outputs/verify/T43/V11/with_overrides_check.txt", "dirty_files": []},
       "baseline_check": {"matched": True, "command": "esdf_rollout --records ${AG3S_PROJECT_ROOT}/run_16d_ep1800 --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005 (.venv-openpi-live, snapshot code)",
                          "raw_output": "outputs/verify/T43/V11/baseline/base_curobo.json", "violated_start": sum(f["clearance_before_mm"] < 0 for f in fr), "has_target": sum(bool(f["target"]) for f in fr), "frame0_clearance_before_mm": fr[0]["clearance_before_mm"],
                          "note": "cuRobo variant of the regression-baseline skill: 10/15, has_target 15/15, frame0 -9.171877401271193 mm; .venv-ag3s legacy variant not run"},
       "numbers": {"runs": {"n_done": a["run_counts"][ARM]["n_runs"], "H": a["run_counts"][ARM]["H"], "B": a["run_counts"][ARM]["B"], "n_control_lt_600": ex["n_control_lt_600"], "wall_s_median": ex["wall_s_median"]},
                   "sgp_old_rule_by_stage": a["sgp_old_rule_by_stage"], "pairs_old_rule": a["pairs_old_rule"], "H_totals_by_rule": ex["H_totals"], "H_pairs_z4_rule": ex["H_pairs_z4_rule"],
                   "ct_arms": ct["arms"], "ct_pairs": ct["pairs"], "ct_crosscheck_counts": a["ct"]["crosscheck_counts"],
                   "finger_hurdle_contacts_by_phase_and_body": a["finger_contacts"], "ct_contact_table": ct["contact_table"], "obstacle_stop": a["obstacle_stop"],
                   "crossing_and_deflection": cross, "crossing_per_run": crossing, "crossing_pairs_commanded_vs_measured": json.load(open(V / f"crossing_pairs_{ARM}.json")), "sharpness_executed_command": sharp,
                   "hold": hold, "hold_only_in_one_arm": hp, "follow_to": ft, "timing": tm, "lookahead_gate_open_rate": ex["lookahead_gate_open_rate"],
                   "verdict_kind_counts": {"V8": ex["verdict_kind_counts_V8"], ARM: ex[f"verdict_kind_counts_{ARM}"]}, "hurdle_min_mm_H_per_run": a["hurdle_min_mm"]},
       "figures": [{"path": scene_png, "kind": "scene"}, {"path": graph_png, "kind": "graph"}, {"path": table_png, "kind": "table"}],
       "scripts": ["p/launch_v11.sh", "p/run_chain.sh", "p/start_server.sh", "p/run_client.sh", "p/env.sh", "p/analyze_v11.py", "p/run_replays_v11.sh", "p/run_analysis_v11.sh", "p/agg_ct_v11.py", "p/z4_base_v11.py", "p/agg_v11.py",
                   "p/extra_v11.py", "p/sharp_v11.py", "p/xm_cross_v11.py", "p/crossing_pairs_v11.py", "p/xm_agg_v11.py", "p/fig_scene_v11.py", "p/fig_graph_v11.py", "p/fig_table_v11.py", "p/build_verify_v11.py", "p/with_overrides_check.py"],
       "raw": {"agg": f"outputs/verify/T43/V11/agg_{ARM}.json", "ct": f"outputs/verify/T43/V11/ct_{ARM}.json", "extra": f"outputs/verify/T43/V11/extra_{ARM}.json", "sharp": f"outputs/verify/T43/V11/sharp_{ARM}.json",
               "xm_summary": f"outputs/verify/T43/V11/xm_summary_{ARM}.json", "xm_cross": f"outputs/verify/T43/V11/raw/cross_{ARM}.json", "analysis": f"outputs/verify/T43/V11/analysis/analysis_{ARM}.json",
               "replay": "outputs/verify/T43/V11/replay", "snapshot": "outputs/verify/T43/V11/snapshot.txt", "V8_analysis": "outputs/verify/T43/V8/analysis/analysis.json", "V8_replay": "outputs/verify/T43/V8/replay",
               "V8_xm_cross": "outputs/verify/T43/XM/raw/cross_V8.json"},
       "verdict": f"{a['run_counts'][ARM]['n_runs']}/19 runs done, {sum(c['n_attempt2'] for c in ex['chains'].values())} retries, server-log guards matched. (facts only)",
       "left_finger_name_note": "MJCF ee_finger_l2 (geom 1086) = URDF ee_finger_l1; MJCF ee_finger_l1 (geom 1079) = URDF ee_finger_l2; right hand same. CT replays / ct_contact_table use MJCF names; server records / HOLD binding_link use URDF names.",
                        "not_measured": ["regression-baseline .venv-ag3s legacy esdf_rollout (excluded); cuRobo variant run on the snapshot code",
                        "V8 and V11A come from different server sessions; first-chunk JIT time not cited; SQP iteration counts and per-frame clearances not cited (not a baseline)",
                        "Z4 / CT rules, hurdle contacts and crossing metrics are H-stage only (stage B has no hurdle, no B replays)",
                        "palm path for sharpness is the FK of the commanded joints with every other joint fixed at the run's control row 0 qpos; boundaries next to a HOLD chunk excluded",
                        "pod working set / loadavg are pod totals (other sessions included)"]}
p = pathlib.Path(f"{_ag3s_workspace_path('')}/AG3S/ag3s/docs/handoff/{name}.verify.json"); assert not p.exists(); p.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False)); print(p, len(p.read_text()))
