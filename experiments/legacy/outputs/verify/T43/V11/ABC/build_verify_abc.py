"""Builds AG3S/ag3s/docs/handoff/T43V11ABC.verify.json (numbers and paths only) from V11/ABC/compare_ABC.json, V11/{agg,ct,extra}_V11ABC.json, V11/ABC/c_records_V11ABC.json, V11/ABC/baseline. usage: build_verify_abc.py"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, glob, re, numpy as np
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11')); ABC = V / "ABC"; R = V / "runs/V11ABC"
c = json.load(open(ABC / "compare_ABC.json")); ex = json.load(open(V / "extra_V11ABC.json"))
cr = json.load(open(ABC / "c_records_V11ABC.json"))
fr = json.load(open(ABC / "baseline/base_curobo.json"))["frames"]
F = _ag3s_workspace_path('AG3S/ag3s/docs/figures/t43/')
SAFE = r"\[safe\] (\d+) chunks: safe (\d+), unsafe (\d+), timeout (\d+), stale (\d+), error (\d+)"
inv = []
for p in sorted(glob.glob(str(R / "*/failed/*.client.log.*"))):
    t = open(p, errors="replace").read(); m = re.search(SAFE, t)
    first = next((l[:220] for l in t.splitlines() if "[safe] t=0 HOLD" in l), None)
    st = p.replace(".client.log.", ".start.")
    sl = [l.strip() for l in open(st)] if pathlib.Path(st).exists() else []
    inv.append({"file": p, "safe_chunk_counts": None if not m else dict(zip(("chunks", "safe", "unsafe", "timeout", "stale", "error"), map(int, m.groups()))), "first_hold_line": first,
                "start_file_lines": sl, "end_file_lines": [l.strip() for l in open(p.replace(".client.log.", ".end."))] if pathlib.Path(p.replace(".client.log.", ".end.")).exists() else []})
# per-run loadavg from the run's .start/.end lines
runload = {}
for f in sorted(glob.glob(str(R / "*/V11ABC_ep*.start"))):
    f = pathlib.Path(f); nm = f.name[:-6]
    s = {l.split("=")[0]: l.split("=", 1)[1].strip() for l in open(f) if "=" in l}
    e = {l.split("=")[0]: l.split("=", 1)[1].strip() for l in open(f.with_suffix(".end")) if "=" in l}
    sa = open(f.with_suffix(".safe_audit.txt")).read().strip() if f.with_suffix(".safe_audit.txt").exists() else None
    runload[f"{f.parent.name}/{nm}"] = {"loadavg_start": s.get("loadavg_start"), "loadavg_end": e.get("loadavg_end"), "port": s.get("port"), "safe_audit": sa}
# cpu throttle AB vs ABC
def thr(p):
    cs = []
    for l in open(p):
        x = l.split()
        if len(x) == 5: cs.append([int(x[0])] + [int(y.split("=")[1]) for y in x[1:]])
    cs = np.array(cs)
    return {"n_samples": len(cs), "elapsed_s": int(cs[-1, 0] - cs[0, 0]), "cpu_cores_used_mean": float((cs[-1, 1] - cs[0, 1]) / 1e6 / max(1, cs[-1, 0] - cs[0, 0])), "nr_periods": int(cs[-1, 2] - cs[0, 2]),
            "nr_throttled": int(cs[-1, 3] - cs[0, 3]), "throttled_share_of_periods": float((cs[-1, 3] - cs[0, 3]) / max(1, cs[-1, 2] - cs[0, 2]))}
thr_cmp = {"V11AB": thr(V / "AB/cpu_samples_V11AB.txt"), "V11ABC": thr(ABC / "cpu_samples_V11ABC.txt")}
fad = [l.strip() for l in open(ABC / "fadvise_log.txt")]
srv_logs = {k: {"cmdline": v["cmdline"], "log_ta_lines": v["log_ta_lines"], "log_gate_on": v["log_gate_on"], "log_autotune_off": v["log_autotune_off"], "gpu_total_used_peak_mib": v["gpu_total_used_peak_mib"], "gpu_used_before_mib": v["gpu_used_before_mib"]} for k, v in ex["servers"].items()}
chain_guard_lines = {k: v["server_log_line"] for k, v in ex["chains"].items()}
out = {"_comment": "ag3s-verifier T43 V11ABC (third of three online rounds A -> A+B -> A+B+C). Numbers and paths only; sources outputs/verify/T43/V11/ABC/compare_ABC.json (four-way V8 / V11A / V11AB / V11ABC) and V11/{agg,ct,extra,sharp,xm_summary}_V11ABC.json (same scripts as V11A / V11AB).",
       "step": "T43V11ABC", "date": "2026-10-07",
       "runs": [{"id": "V11ABC", "command": "outputs/verify/T43/V11/ABC/launch_abc.sh (chains E 8286 / F 8287, EST_GB 28; run_chain_abc.sh = p/run_chain.sh with run_client_abc.sh) at 21:17Z, then ABC/redo_bad_abc.sh (PORT=8287 STAGES=H) for the two H runs of ep1834:18341 / 18343 after the 8287 server stopped answering", "venv": ".venv-openpi-live",
                 "server_flags": "V8 flags + --plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4 --w-deflection-rate 20 --servo-accel relaxed --servo-accel-tolerance-deg 1 --servo-accel-weight 0.01 (no --servo-model, no --held-obstacle-margin); run_chain_abc.sh refuses to run clients unless the server cmdline has all flags and the log has the A, B, QP and C lines",
                 "runs_dir": "outputs/verify/T43/V11/runs/V11ABC", "n_runs": 19, "n_done_valid": 19, "n_invalid_attempts": len(inv) and len({re.sub(r'\.client\.log\.', '.', pathlib.Path(i['file']).name) for i in inv}), "invalid_attempts": inv,
                 "invalid_attempt_definition": "run_client_abc.sh (copy of p/run_client.sh): after check_run.py, the client log line '[safe] N chunks: ... timeout T, stale S, error E' must end 'timeout 0, stale 0, error 0' (and exist) or the run gets no .done; the chain retries once on the same server; ABC/redo_bad_abc.sh audits all 19 runs again",
                 "final_audit": "outputs/verify/T43/V11/runs/final_audit_V11ABC.out (no bad / missing runs); safe lines of the 19 valid runs: timeout 0, stale 0, error 0 each",
                 "stuck_server_diagnostic": "outputs/verify/T43/V11/ABC/notes_V11ABC.txt: server pid 972363 (port 8287) wrote no chunk after 21:54Z, CPU 0.7 %, main thread in ep_poll, clients got 'received 1011 keepalive ping timeout' on every request; killed by the verifier; the chain stopped with rc=7 ('server died')",
                 "chain_rc_lines": ex["chain_rc_lines"], "redo_output": "runs/redo_F_V11ABC.out", "chain_refusals": {k: v["refused"] for k, v in ex["chains"].items()}, "servers": srv_logs, "chain_server_log_guard_lines": chain_guard_lines,
                 "launch_outputs": ["runs/launch_V11ABC.out", "runs/chainE_V11ABC.out", "runs/chainF_V11ABC.out", "runs/redo_F_V11ABC.out", "runs/final_audit_V11ABC.out"],
                 "per_run_loadavg_start_end": runload,
                 "samplers": {"loadavg": "outputs/verify/T43/V11/ABC/load_samples_V11ABC.txt (10 s, pod totals incl. other sessions)", "cpu_stat": "outputs/verify/T43/V11/ABC/cpu_samples_V11ABC.txt", "cpu_throttle_AB_vs_ABC": thr_cmp},
                 "guards": {"ports": "8286 / 8287 only, free before each start (checked before launch: 8280-8289 all free), never 8123", "max_servers": 2, "gpu_peak_mib_per_server": {k: v["gpu_total_used_peak_mib"] for k, v in ex["servers"].items()},
                            "working_set_rule": "working set + 4 GiB (client) / 8 GiB (server start) <= 64 GiB, enforced by run_chain_abc.sh; no 'mem wait' line in either chain log", "gpu_rule": "GPU used + 28 GB estimate <= 130 GB at each server start (chain log 'gpu used_mib' lines)",
                            "fadvise_log": fad}}],
       "code_state": {"repo": "benchmark", "head": "3ca1b21", "snapshot": "${AG3S_PROJECT_ROOT}-t43v11 (outputs/verify/T43/V11/snapshot.txt)", "code_md5_baseline": "outputs/verify/T43/V11/p/code_md5_v11.txt (407 lines; cmp == before the launch and at every chain preflight)", "dirty_files": []},
       "baseline_check": {"matched": True, "command": "esdf_rollout --records run_16d_ep1800 --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005 (.venv-openpi-live, PYTHONPATH snapshot, cwd ${AG3S_PROJECT_ROOT})",
                          "raw_output": "outputs/verify/T43/V11/ABC/baseline/base_curobo.json", "violated_start": sum(f["clearance_before_mm"] < 0 for f in fr), "has_target": sum(bool(f["target"]) for f in fr), "frame0_clearance_before_mm": fr[0]["clearance_before_mm"], "note": ".venv-ag3s legacy variant not run"},
       "numbers": {"totals_old_rule": c["totals_old_rule"], "pairs_old_rule": c["pairs_old_rule"], "H_totals_by_rule": c["H_totals_by_rule"], "H_pairs_z4_ct_rule": c["H_pairs_z4_ct_rule"],
                   "contacts_arms": c["contacts_arms"], "contact_table_by_body_phase": c["contact_table_by_body_phase"], "finger_hurdle_contacts_by_phase": c["finger_hurdle_contacts_by_phase"], "obstacle_stop": c["obstacle_stop"],
                   "ct_contact_fail_among_old_success": c["ct_contact_fail_among_old_success"],
                   "crossing_clearance": c["crossing3"], "deflection_direction": c["deflection_direction"], "crossing_stats_xm": c["crossing_stats_xm"], "sharpness": c["sharpness"], "sharpness_paired": c["sharpness_paired"],
                   "c_server_records": cr, "hold": c["hold3"], "follow_to": c["follow_to3"], "verdict_kind_counts": c["verdict_kind_counts"],
                   "grasp_phase": c["grasp_phase"], "grasp_phase_pairs": c["grasp_phase_pairs"],
                   "fail_stage": {"definition": "dropped := grasp true, place false, and (first grasp_detach/slip chunk with apple not over the crate, or Z4 release_t with the apple outside the crate xy); contact := grasp true, place false, not dropped, run ended by obstacle stop (n_control < 600); place_false := otherwise; no_grasp := grasp false. Rows: runs with S_old true in some arm and false in some V11 arm.",
                                  "runs": c["fail_stage_runs_fail_in_some_V11_arm_success_in_another_arm"], "counts_over_listed_runs": c["fail_stage_counts"], "stage_all_runs": c["stage_all_runs"]},
                   "timing": {"timing3": c["timing3"], "raw_to_from_server_records": c["raw_to_from_server_records"], "loadavg_per_round": c["loadavg_per_round"], "load1_during_runs": c["load1_during_runs"], "to_by_load_bin": c["to_by_load_bin"],
                              "per_run_load_vs_to": c["per_run_load_vs_to"], "spearman_load_vs_to_median": c["spearman_load_vs_to_median"], "per_run_to_median_pairs": c["per_run_to_median_pairs"], "cpu_throttle_during_ABC_round": c.get("cpu_throttle_during_ABC_round")},
                   "ta_server_log_lines": c["ta_server_log_lines"], "deflection_rate_records": c["deflection_rate_records"], "servo_accel_records": c["servo_accel_records"], "lookahead_gate": c["lookahead_gate"],
                   "runs": {"n_control_lt_600": ex["n_control_lt_600"], "wall_s_median": ex["wall_s_median"]}},
       "figures": [{"path": F + "t43v11abc-crossing-scene.png", "kind": "scene", "sidecar": F + "t43v11abc-crossing-scene.json"}, {"path": F + "t43v11abc-graph.png", "kind": "graph", "sidecar": F + "t43v11abc-graph.json"}, {"path": F + "t43v11abc-pairs-table.png", "kind": "table", "sidecar": F + "t43v11abc-pairs-table.json"}],
       "scripts": ["V11/ABC/launch_abc.sh", "V11/ABC/run_chain_abc.sh", "V11/ABC/run_client_abc.sh", "V11/ABC/redo_bad_abc.sh", "V11/ABC/load_sampler_abc.sh", "V11/ABC/cpu_sampler_abc.sh", "V11/ABC/fadvise_loop_abc.sh", "V11/ABC/compare_abc.py", "V11/ABC/c_records.py", "V11/ABC/fig_scene_abc.py", "V11/ABC/fig_graph_abc.py", "V11/ABC/fig_table_abc.py", "V11/ABC/build_verify_abc.py",
                   "V11/p/start_server.sh", "V11/p/env.sh", "V11/p/analyze_v11.py", "V11/p/run_replays_v11.sh", "V11/p/run_analysis_v11.sh", "V11/p/agg_ct_v11.py", "V11/p/agg_v11.py", "V11/p/extra_v11.py", "V11/p/sharp_v11.py", "V11/p/xm_cross_v11.py", "V11/p/crossing_pairs_v11.py", "V11/p/xm_agg_v11.py"],
       "raw": {"compare": "outputs/verify/T43/V11/ABC/compare_ABC.json", "agg": "outputs/verify/T43/V11/agg_V11ABC.json", "ct": "outputs/verify/T43/V11/ct_V11ABC.json", "extra": "outputs/verify/T43/V11/extra_V11ABC.json", "sharp": "outputs/verify/T43/V11/sharp_V11ABC.json",
               "xm_summary": "outputs/verify/T43/V11/xm_summary_V11ABC.json", "xm_cross": "outputs/verify/T43/V11/raw/cross_V11ABC.json", "crossing_pairs": "outputs/verify/T43/V11/crossing_pairs_V11ABC.json", "analysis": "outputs/verify/T43/V11/analysis/analysis_V11ABC.json",
               "replay": "outputs/verify/T43/V11/replay (V11ABC_H_*)", "c_records": "outputs/verify/T43/V11/ABC/c_records_V11ABC.json", "earlier_arm_files": "outputs/verify/T43/V11/*_V11A.json, *_V11AB.json (not modified)", "V8_analysis": "outputs/verify/T43/V8/analysis/analysis.json"},
       "verdict": "baseline matched (10/15 violated at start, frame0 clearance_before -9.171877401271193 mm); 19/19 valid runs done after 2 invalid attempts (all 75 chunks error 75, same run ep1834:18341, server on 8287 stopped answering) were moved to failed/ and rerun; server-log guards matched (A, B, QP and C lines). (facts only)",
       "left_finger_name_note": "MJCF ee_finger_l2 (geom 1086) = URDF ee_finger_l1; MJCF ee_finger_l1 (geom 1079) = URDF ee_finger_l2; right hand same. CT replays / contact_table_by_body_phase use MJCF names; server records / HOLD binding_link use URDF names.",
       "not_measured": ["regression-baseline .venv-ag3s legacy esdf_rollout (excluded); cuRobo variant run on the snapshot code",
                        "V8, V11A, V11AB and V11ABC come from different server sessions; first-chunk JIT time not cited; SQP iteration counts and per-frame clearances not cited (not a baseline)",
                        "Z4 / CT rules, hurdle contacts and crossing metrics are H-stage only (stage B has no hurdle, no B replays)",
                        "pod loadavg / working set / cpu.stat are pod totals (other sessions included); loadavg_per_round spans the whole elapsed round, load1_during_runs is the mean over each run's own interval; pod load in V8 / V11A rounds had no cpu.stat sampler",
                        "ep1834 s18341 / s18343 H runs ended by an obstacle stop at 265 / 252 control rows in V11ABC (V8 s18341: 250)",
                        "failed attempts have no server chunk records in the aggregates"]}
p = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V11ABC.verify.json')); assert not p.exists(); p.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False)); print(p, p.stat().st_size)
print(out["runs"][0]["n_invalid_attempts"], thr_cmp)
