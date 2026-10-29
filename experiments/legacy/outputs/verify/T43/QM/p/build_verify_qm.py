"""T43 QM -> AG3S/ag3s/docs/handoff/T43QM.verify.json (numbers and paths only). usage: build_verify_qm.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
Q = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/QM')); D = json.load(open(Q / "qm_metrics.json")); M = D["paired"]; P = D["per_run"]
V11AB_VER = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V11AB.verify.json')))["numbers"]


def slim(v):
    if "V8_true" in v:
        return {k: v[k] for k in ("n_pairs", "V8_true", "V11AB_true", "V8_only", "V11AB_only", "both", "neither", "test", "p", "V8_only_keys", "V11AB_only_keys")}
    keep = ("n_pairs", "n_runs", "V8_paired_median", "V11AB_paired_median", "n_V11AB_higher", "n_V11AB_lower", "n_tied", "n_nonzero", "W", "p", "test")
    o = {k: v.get(k) for k in keep}
    o["V8"] = {k: v["V8"].get(k) for k in ("n", "median", "p25", "p75", "p90", "min", "max")}
    o["V11AB"] = {k: v["V11AB"].get(k) for k in ("n", "median", "p25", "p75", "p90", "min", "max")}
    o["diff_V11AB_minus_V8_median"] = v["diff_V11AB_minus_V8"].get("median")
    return o


def pick(prefixes):
    return {k: (({s: slim(x[s]) for s in ("ALL", "H", "B")}) if "ALL" in x else slim(x)) for k, x in M.items() if k != "retained_given_grasp" and any(k.startswith(p) for p in prefixes)}


H = [k for k in D["keys"] if k.startswith("H")]
per_run_lead = {k: {a: {f: P[k][a].get(f) for f in ("crossing_t", "crossing_t_C", "first_inhand_t", "deflection_onset_t", "deflection_onset_rows", "deflection_onset_rows_from_t0",
                                                     "deflection_onset_rows_contiguous", "palm_deflection_at_crossing_mm", "lift_hold_rows", "lift_hold_rows_before_crossing",
                                                     "lift_hold_rows_after_crossing", "lift_hold_span", "lifted_rows_in_window", "lift_dz_at_crossing_mm", "lift_dz_max_in_window_mm",
                                                     "apple_bottom_minus_bar_top_at_crossing_C_mm", "apple_bottom_minus_bar_top_at_crossing_M_mm",
                                                     "acc_M_window_in_carry_p90", "acc_C_window_in_carry_p90")} for a in ("V8", "V11AB")} for k in H}
per_run_core = {k: {a: {f: P[k][a].get(f) for f in ("n_control", "grasp", "success_old", "success_z4", "success_ct", "retained", "held_to_end", "obstacle_stop", "grasp_t", "attach_t", "release_t",
                                                     "place_t", "contact_robot_carry", "contact_apple_carry", "contact_robot_return", "minclr_fingers_mm", "minclr_palm_mm",
                                                     "minclr_apple_carry_mm", "viol_M_window_row_share", "viol_C_row_share", "slip_fm_pre_open_end_mm", "slip_fm_max_mm", "path_apple_mm",
                                                     "dur_grasp_to_release_s", "to_server_median_ms", "to_server_p90_ms", "load1_mean_during_run", "to_tp_cuda_median_ms")}
                    for a in ("V8", "V11AB")} for k in D["keys"]}
numbers = {
    "_comment": "numbers only; d = V11AB - V8 on ep:seed pairs; continuous = Wilcoxon signed-rank two-sided on per-run values, binary = exact McNemar. definitions -> definitions.",
    "n_pairs": {"ALL": len(D["keys"]), "H": len(H), "B": len(D["keys"]) - len(H)},
    "success": pick(("success_", "grasp")),
    "collision_H": pick(("contact_",)),
    "min_clearance_M_H": pick(("minclr_",)),
    "safety_violation": pick(("viol_", "violated_chunk_share", "hold_chunk_share")),
    "grasp_retention": {**pick(("retained", "held_to_end")), "retained_given_grasp": M["retained_given_grasp"]},
    "slip": pick(("slip_",)),
    "path_length": pick(("path_",)),
    "execution_time": pick(("t_", "dur_")),
    "crossing_clearance_H": pick(("apple_bottom_minus_bar_top_at_crossing", "apple_bar_geo_at_crossing")),
    "deflection_onset_H": pick(("deflection_onset", "palm_deflection_at_crossing")),
    "lift_hold_H": pick(("lift_", "lifted_rows")),
    "acc_jerk_tcp": pick(("acc_", "jerk_")),
    "joint_acc_boundary_sharp_V11AB": pick(("joint_acc_boundary",)),
    "latency_server_recorded": {**pick(("to_server_", "load1_mean")), "pooled_chunks_ms": D["latency_server_pooled"], "load1_during_runs": D["load1_during_runs"]},
    "latency_controlled_T43TP": {**pick(("to_tp_",)), "pooled_from_agg": D["latency_tp_controlled_pooled"]},
    "window_rows_H": slim(M["n_window_rows"]),
    "deflection_direction_reused_T43V11AB": {a: {k: V11AB_VER["deflection_direction"][a]["carry_apple"].get(k) for k in ("n_chunks", "n_deflected_gt5", "deflected_share", "deflected_dominant_axis", "palm_D_max_mm")}
                                             for a in ("V8", "V11AB")},
    "per_run_lead_metrics_H": per_run_lead,
    "per_run_core": per_run_core,
    "excluded_metrics_lead": {
        "Goal Pose Error": "the goal is 'anywhere inside the crate', so there is no defined goal pose; whether the apple is in the crate is part of the success rules (T43.task.md QM section, lead)",
        "Recovery / Replanning Success": "obstacles are fixed in this scene, so there is no change in the environment; HOLD exit counts are already in the HOLD breakdown (T43.task.md QM section, lead)"},
}
out = {
    "_comment": "ag3s-verifier. numbers and paths only — no interpretation.",
    "step": "T43QM", "date": "2026-10-09",
    "runs": [
        {"id": "replays_B", "command": "outputs/verify/T43/QM/p/run_replays_b.sh (CT/p/replay_ct.py unchanged, CPU, 1 worker nice 19; V8 snapshot work-t43v8, V11AB work-t43v11)",
         "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/QM/replay", "stdout_log": ["outputs/verify/T43/QM/raw/replay_b_V8.log", "outputs/verify/T43/QM/raw/replay_b_V11AB.log"],
         "qpos_mismatch_rows": 0, "n_runs": 10},
        {"id": "rows", "command": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa .venv-openpi-live/bin/python outputs/verify/T43/QM/p/qm_rows.py outputs/verify/T43/QM/raw/rows",
         "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/QM/raw/rows/*.npz", "stdout_log": "outputs/verify/T43/QM/raw/qm_rows.log",
         "crosscheck": {"palm_deflection_vs_xm_cross_V11AB_ep1807_s18071_max_abs_diff_mm": 0.0, "palm_M_vs_xm_cross_max_abs_diff_m": 0.0, "tcp_M_vs_replay_tcp_max_abs_diff_m": 0.00066, "tcp_M_vs_replay_tcp_median_abs_diff_m": 1.2e-05}},
        {"id": "metrics", "command": "same env; outputs/verify/T43/QM/p/qm_metrics.py outputs/verify/T43/QM/qm_metrics.json", "venv": ".venv-openpi-live",
         "raw_output": "outputs/verify/T43/QM/qm_metrics.json", "stdout_log": "outputs/verify/T43/QM/raw/qm_metrics.log"}],
    "code_state": {"repo": "${AG3S_PROJECT_ROOT} (HEAD 9bf5bcd), benchmark (a1ad391)", "dirty_files": [], "note": "no code executed from benchmark/ except figstyle; client snapshots work-t43v8 / work-t43v11 for replays"},
    "baseline_check": {"matched": None, "note": "regression baseline not run: it requires .venv-ag3s, which the lead excluded for this step; recorded data + CPU replays only, no new closed-loop runs"},
    "inputs": {"runs": ["outputs/verify/T43/V8/runs/V8/{H,B}", "outputs/verify/T43/V11/runs/V11AB/{H,B} (failed/ excluded)"],
               "analysis": ["outputs/verify/T43/V8/analysis/analysis.json", "outputs/verify/T43/V11/analysis/analysis_V11AB.json"],
               "ct_z4_H": "outputs/verify/T43/V11/ct_V11AB.json", "replays_H": ["outputs/verify/T43/V8/replay", "outputs/verify/T43/V11/replay"],
               "cross": ["outputs/verify/T43/XM/raw/cross_V8.json", "outputs/verify/T43/V11/raw/cross_V11AB.json"], "sharp": "outputs/verify/T43/V11/sharp_V11AB.json",
               "latency_controlled": ["outputs/impl/T43TP/timing/{cuda,cpu}/V8_V8_*.json", "outputs/impl/T43TP/agg/timing_cuda.json", "outputs/impl/T43TP/agg/timing_cpu.json"],
               "load_samples": ["outputs/verify/T43/V8/raw/load_samples.txt", "outputs/verify/T43/V11/AB/load_samples_V11AB.txt"]},
    "definitions": D["definitions"],
    "numbers": numbers,
    "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43qm-trajectory.png", "kind": "graph", "sidecar": "AG3S/ag3s/docs/figures/t43/t43qm-trajectory.json"},
                {"path": "AG3S/ag3s/docs/figures/t43/t43qm-table.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/t43/t43qm-table.json"},
                {"path": "AG3S/ag3s/docs/figures/t43/t43qm-distributions.png", "kind": "graph", "sidecar": "AG3S/ag3s/docs/figures/t43/t43qm-distributions.json"},
                {"path": "AG3S/ag3s/docs/figures/t43/t43qm-scene.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/t43/t43qm-scene.json", "ep_seed": "1807_s18073"}],
    "scripts": ["outputs/verify/T43/QM/p/run_replays_b.sh", "outputs/verify/T43/QM/p/qm_rows.py", "outputs/verify/T43/QM/p/qm_metrics.py", "outputs/verify/T43/QM/p/fig_qm.py",
                "outputs/verify/T43/QM/p/fig_qm_scene.py", "outputs/verify/T43/QM/p/build_verify_qm.py"],
    "raw": {"metrics": "outputs/verify/T43/QM/qm_metrics.json", "rows": "outputs/verify/T43/QM/raw/rows", "replay_B": "outputs/verify/T43/QM/replay", "scripts_md5": "outputs/verify/T43/QM/raw/scripts_md5.txt"},
    "facts": {
        "V8_runs_ended_before_600_rows_H": sorted(k for k in H if P[k]["V8"]["n_control"] < 600),
        "V11AB_runs_ended_before_600_rows_H": sorted(k for k in H if P[k]["V11AB"]["n_control"] < 600),
        "runs_without_apple_crossing_H": {a: sorted(k for k in H if P[k][a].get("crossing_t") is None) for a in ("V8", "V11AB")},
        "runs_without_load_samples": {a: sorted(k for k in D["keys"] if P[k][a].get("load1_mean_during_run") is None) for a in ("V8", "V11AB")},
        "slip_Y2_last_state_contains_gripper_opening": "on open_cmd releases the Y2 window's last state is after the open command row; slip_fm_pre_open_* ends one state earlier",
        "B_V8_1982_s19822": {k: P["B:1982_s19822"]["V8"].get(k) for k in ("success_old", "success_z4", "retained", "release_t", "z4_B")},
        "replays_overlapped_with": "T43 LB GPU episode screening (other verifier) was running during the 10 B replays (~90 s CPU, nice 19, 1 worker)",
        "left_finger_naming": "fingers used as a pair everywhere (min over both finger geoms, midpoint of both finger bodies); MJCF/URDF l1/l2 swap does not enter",
    },
    "verdict": "facts only; no judgement",
    "not_measured": ["regression baseline (.venv-ag3s excluded by lead)", "B-stage hurdle metrics (no obstacle)", "plan-C row violations outside crossing-window chunks (TO-row reconstruction exists only for window chunks)"],
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float)); print(sys.argv[1], len(json.dumps(out)))
