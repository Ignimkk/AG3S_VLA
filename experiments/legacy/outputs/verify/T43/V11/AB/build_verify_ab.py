"""Builds AG3S/ag3s/docs/handoff/T43V11AB.verify.json (numbers and paths only) from V11/AB/compare_AB.json, V11/{agg,ct,extra}_V11AB.json, V11/AB/baseline. usage: build_verify_ab.py"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, glob, re
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11')); AB = V / "AB"
c = json.load(open(AB / "compare_AB.json")); ex = json.load(open(V / "extra_V11AB.json")); ag = json.load(open(V / "agg_V11AB.json"))
fr = json.load(open(AB / "baseline/base_curobo.json"))["frames"]
F = _ag3s_workspace_path('AG3S/ag3s/docs/figures/t43/')
failed = sorted({re.sub(r"\.(check\.json|client\.log|start|end|done)\.", ".", pathlib.Path(p).name) for p in ex["failed_dirs"]})
failed_att = []
for p in sorted(glob.glob(str(V / "runs/V11AB/*/failed/*.client.log.*"))):
    t = open(p, errors="replace").read(); m = re.search(r"\[safe\] (\d+) chunks: safe (\d+), unsafe (\d+), timeout (\d+), stale (\d+), error (\d+)", t)
    first = next((l[:200] for l in t.splitlines() if "[safe] t=0 HOLD" in l), None)
    failed_att.append({"file": p, "safe_chunk_counts": None if not m else dict(zip(("chunks", "safe", "unsafe", "timeout", "stale", "error"), map(int, m.groups()))), "first_hold_line": first})
cpu0 = open(AB / "cpu_stat_mid_start.txt").read(); cpu1 = open(AB / "cpu_stat_end.txt").read()
fad = [l.strip() for l in open(AB / "fadvise_log.txt")]
out = {"_comment": "ag3s-verifier T43 V11AB (second of three online rounds A -> A+B -> A+B+C). Numbers and paths only; sources outputs/verify/T43/V11/AB/compare_AB.json (three-way) and V11/{agg,ct,extra,sharp,xm_summary}_V11AB.json (V11AB vs V8, same scripts as V11A).",
       "step": "T43V11AB", "date": "2026-10-07",
       "runs": [{"id": "V11AB", "command": "outputs/verify/T43/V11/p/launch_v11.sh V11AB (chains E 8286 / F 8287, EST_GB 28) at 18:37Z, then AB/relaunch_ab.sh, AB/redo_bad.sh, AB/missing_ab.sh for runs whose server connection dropped", "venv": ".venv-openpi-live",
                 "server_flags": "V8 flags + --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 (every server: cmdline contains --w-deflection-rate 20, log line 'TO deflection rate (T43 TA B): w_deflection_rate=20' checked by run_chain.sh before the first client; no --servo-accel, no C log line)",
                 "runs_dir": "outputs/verify/T43/V11/runs/V11AB", "n_runs": 19, "n_done_valid": 19, "n_failed_attempts_moved_to_failed": len(failed_att), "failed_attempt_names": failed, "failed_attempts": failed_att,
                 "chain_rc_lines": ex["chain_rc_lines"], "chain_refusals": {k: v["refused"] for k, v in ex["chains"].items()}, "servers": ex["servers"], "chains": ex["chains"],
                 "launch_outputs": ["runs/launch_V11AB.out", "runs/chainE_V11AB.out", "runs/chainF_V11AB.out", "runs/chainE2_V11AB.out", "runs/chainF2_V11AB.out", "runs/redo_V11AB.out", "runs/missing_V11AB.out", "runs/relaunch_V11AB.out"],
                 "samplers": {"loadavg": "outputs/verify/T43/V11/AB/load_samples_V11AB.txt (10 s, pod totals; no samples between the end of the redo pass and the start of the last single run, ~20:47-20:59Z)", "cpu_stat": "outputs/verify/T43/V11/AB/cpu_samples_V11AB.txt", "cpu_stat_mid_start": cpu0, "cpu_stat_end": cpu1},
                 "guards": {"ports": "8286 / 8287 only, free before each start, never 8123", "max_servers": 2, "gpu_peak_mib_per_server": {k: v["gpu_total_used_peak_mib"] for k, v in ex["servers"].items()}, "working_set_rule": "working set + 4 GiB (client) / 8 GiB (server start) <= 64 GiB, enforced by run_chain.sh",
                            "memory_guard_waits": "chain F idled its server ~35 min and chain E ~12 min on the 'mem wait' guard (pod working set 60-63 GiB, active_file page cache); posix_fadvise(DONTNEED) on the verifier's own output dirs (V11, V8) lowered the working set by 8.5 GB at 19:17Z and every 5 min after (AB/fadvise_log.txt)", "fadvise_log": fad}}],
       "code_state": {"repo": "benchmark", "head": "3ca1b21", "snapshot": "${AG3S_PROJECT_ROOT}-t43v11 (outputs/verify/T43/V11/snapshot.txt)", "code_md5_baseline": "outputs/verify/T43/V11/p/code_md5_v11.txt (407 lines; cmp == at the start of AB and at every chain preflight)",
                      "check_run_py": "unchanged during the round (an edit to make it fail all-comms runs was not applied); the all-comms runs were found by the client-log line '[safe] N chunks: ... error N' (AB/redo_bad.sh)", "dirty_files": []},
       "baseline_check": {"matched": True, "command": "esdf_rollout --records run_16d_ep1800 --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005 (.venv-openpi-live, snapshot code, cwd ${AG3S_PROJECT_ROOT})",
                          "raw_output": "outputs/verify/T43/V11/AB/baseline/base_curobo.json", "violated_start": sum(f["clearance_before_mm"] < 0 for f in fr), "has_target": sum(bool(f["target"]) for f in fr), "frame0_clearance_before_mm": fr[0]["clearance_before_mm"], "note": ".venv-ag3s legacy variant not run"},
       "numbers": {"totals_old_rule": c["totals_old_rule"], "pairs_old_rule": c["pairs_old_rule"], "H_totals_by_rule": c["H_totals_by_rule"], "H_pairs_z4_ct_rule": c["H_pairs_z4_ct_rule"],
                   "contacts_arms": c["contacts_arms"], "contact_table_by_body_phase": c["contact_table_by_body_phase"], "finger_hurdle_contacts_by_phase": c["finger_hurdle_contacts_by_phase"], "obstacle_stop": c["obstacle_stop"],
                   "ct_contact_fail_among_old_success": c["ct_contact_fail_among_old_success"],
                   "crossing_clearance": c["crossing3"], "deflection_direction": c["deflection_direction"], "crossing_stats_xm": c["crossing_stats_xm"], "sharpness": c["sharpness"], "sharpness_paired": c["sharpness_paired"],
                   "hold": c["hold3"], "follow_to": c["follow_to3"], "verdict_kind_counts": c["verdict_kind_counts"],
                   "grasp_phase": c["grasp_phase"], "grasp_phase_pairs": c["grasp_phase_pairs"],
                   "fail_stage": {"definition": "dropped := grasp true, place false, and (first grasp_detach/slip chunk with apple not over the crate, or Z4 release_t with the apple outside the crate xy); contact := grasp true, place false, not dropped, run ended by obstacle stop (n_control < 600); place_false := otherwise; no_grasp := grasp false",
                                  "runs_V8_success_not_in_V11A_or_V11AB": c["fail_stage_runs_V8_success_not_in_V11A_or_V11AB"], "counts": c["fail_stage_counts"], "stage_all_runs": c["stage_all_runs"]},
                   "timing": {"timing3": c["timing3"], "raw_to_from_server_records": c["raw_to_from_server_records"], "loadavg_per_round": c["loadavg_per_round"], "load1_during_runs": c["load1_during_runs"], "to_by_load_bin": c["to_by_load_bin"],
                              "per_run_load_vs_to": c["per_run_load_vs_to"], "spearman_load_vs_to_median": c["spearman_load_vs_to_median"], "per_run_to_median_pairs": c["per_run_to_median_pairs"], "cpu_throttle_during_AB_round": c.get("cpu_throttle_during_AB_round")},
                   "ta_server_log_lines": c["ta_server_log_lines"], "deflection_rate_records": c["deflection_rate_records"], "lookahead_gate": c["lookahead_gate"],
                   "runs": {"n_control_lt_600": ex["n_control_lt_600"], "wall_s_median": ex["wall_s_median"]}},
       "figures": [{"path": F + "t43v11ab-crossing-scene.png", "kind": "scene"}, {"path": F + "t43v11ab-graph.png", "kind": "graph"}, {"path": F + "t43v11ab-pairs-table.png", "kind": "table"}],
       "scripts": ["V11/p/launch_v11.sh", "V11/p/run_chain.sh", "V11/p/start_server.sh", "V11/p/run_client.sh", "V11/p/env.sh", "V11/p/analyze_v11.py", "V11/p/run_replays_v11.sh", "V11/p/run_analysis_v11.sh", "V11/p/agg_ct_v11.py", "V11/p/agg_v11.py", "V11/p/extra_v11.py",
                   "V11/p/sharp_v11.py", "V11/p/xm_cross_v11.py", "V11/p/crossing_pairs_v11.py", "V11/p/xm_agg_v11.py", "V11/AB/compare_ab.py", "V11/AB/fig_scene_ab.py", "V11/AB/fig_graph_ab.py", "V11/AB/fig_table_ab.py", "V11/AB/build_verify_ab.py",
                   "V11/AB/relaunch_ab.sh", "V11/AB/redo_bad.sh", "V11/AB/redo_after_f2.sh", "V11/AB/missing_ab.sh", "V11/AB/load_sampler_ab.sh", "V11/AB/cpu_sampler_ab.sh", "V11/AB/fadvise_drop.py", "V11/AB/fadvise_loop.sh"],
       "raw": {"compare": "outputs/verify/T43/V11/AB/compare_AB.json", "agg": "outputs/verify/T43/V11/agg_V11AB.json", "ct": "outputs/verify/T43/V11/ct_V11AB.json", "extra": "outputs/verify/T43/V11/extra_V11AB.json", "sharp": "outputs/verify/T43/V11/sharp_V11AB.json",
               "xm_summary": "outputs/verify/T43/V11/xm_summary_V11AB.json", "xm_cross": "outputs/verify/T43/V11/raw/cross_V11AB.json", "crossing_pairs": "outputs/verify/T43/V11/crossing_pairs_V11AB.json", "analysis": "outputs/verify/T43/V11/analysis/analysis_V11AB.json",
               "replay": "outputs/verify/T43/V11/replay (V11AB_H_*)", "V11A_files": "outputs/verify/T43/V11/*_V11A.json (not modified)", "V8_analysis": "outputs/verify/T43/V8/analysis/analysis.json"},
       "verdict": "19/19 valid runs done after 6 failed attempts (all-chunk server connection errors) were moved to failed/ and rerun; server-log guards matched (incl. B line, no C line). (facts only)",
       "left_finger_name_note": "MJCF ee_finger_l2 (geom 1086) = URDF ee_finger_l1; MJCF ee_finger_l1 (geom 1079) = URDF ee_finger_l2; right hand same. CT replays / contact_table_by_body_phase use MJCF names; server records / HOLD binding_link use URDF names.",
       "not_measured": ["regression-baseline .venv-ag3s legacy esdf_rollout (excluded); cuRobo variant run on the snapshot code",
                        "V8, V11A and V11AB come from different server sessions; first-chunk JIT time not cited; SQP iteration counts and per-frame clearances not cited (not a baseline)",
                        "Z4 / CT rules, hurdle contacts and crossing metrics are H-stage only (stage B has no hurdle, no B replays)",
                        "closing outward is analyze_t43w focus (row-7 fingertip midpoint of the active arm, refined vs reference, distance to the apple centre); 'first closing' = first server chunk with grasp.state closing and not attached; the pre_release variants cut closing chunks at the Z4 release_t (B runs: all chunks)",
                        "pod working set / loadavg / cpu.stat are pod totals (other sessions included); loadavg_per_round spans the whole elapsed round including waits, load1_during_runs is the mean over each run's own interval",
                        "chunk-level TO timing in V11AB rows includes only the 19 valid runs; failed attempts have no server chunk records"]}
p = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V11AB.verify.json')); assert not p.exists(); p.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False)); print(p, p.stat().st_size)
