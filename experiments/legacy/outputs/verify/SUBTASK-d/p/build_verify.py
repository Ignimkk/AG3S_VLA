"""SUBTASK-d — build AG3S/ag3s/docs/handoff/SUBTASK-d.verify.json from the raw files (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, datetime, pathlib
D = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-d')); H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
A = json.load(open(D / "v2/analysis.json")); X = json.load(open(D / "v2/summary_extra.json")); V0 = json.load(open(D / "v0/v0_summary.json"))
B = json.load(open(D / "baseline/judge.json")); PART = json.load(open(H / "SUBTASK-d.verify.partial.json"))
S = A["summary"]; so, sn = S["off"], S["on"]
keep = ("n_runs", "n_success", "n_grasp", "n_place", "n_placed_latch", "n_placed_runs_with_switch_after", "n_switch_after_placed_total",
        "after_placed_chunks_total", "after_placed_excl_active_total", "after_placed_excl_source_counts", "after_placed_carved_nearest_counts",
        "after_placed_carved_to_apple_mm", "after_placed_carved_to_apple_initial_mm", "after_placed_carved_to_nearest_mm",
        "n_runs_released", "n_released_chunks_total", "placed_to_release_chunks", "placed_to_first_switch_chunks", "release_chunk_minus_place_chunk",
        "n_switch_before_placed_total", "n_switch_total", "blocked_counts_all", "n_subtask_gated_status", "first_adoption_t",
        "label_x_latch", "argmax_x_latch", "delay_pick_to_place_chunks", "delay_place_to_home_chunks", "n_label_home_before_placed",
        "n_hold_chunks_total", "n_runs_with_hold", "n_hold_after_placed_total", "n_hold_from_release_total", "n_chunks_from_release_total",
        "hold_kind_from_release", "hold_kind_counts", "hold_latch_counts", "gate_action_counts", "reason_kind_counts",
        "clearance_min_mm", "clearance_min_mm_overall", "n_chunks_clearance_lt0", "n_budget_hit", "to_status_counts",
        "time_ms_pooled_excl_first", "n_refined_mismatch", "n_runs_segment_found", "n_runs_with_subtask_record")
runs_on = {k: v for k, v in PART["runs"].items() if v["stage"] == "runs"}
gpu = []
for f in sorted((D / "runs/E3bG").glob("server_*/gpu_samples.txt")):
    vals = [int(l.split()[1]) for l in f.read_text().splitlines() if len(l.split()) == 2 and l.split()[1].isdigit()]
    gpu.append({"file": str(f), "max_total_used_mib": max(vals) if vals else None})
out = {
 "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
 "step": "SUBTASK-d", "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
 "runs": [
  {"id": "baseline", "command": "cwd ${AG3S_PROJECT_ROOT}-sd: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-sd .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records ${AG3S_PROJECT_ROOT}/run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms [--esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005]",
   "venv": ".venv-openpi-live (.venv-ag3s not used: NFS stall)", "raw_output": ["outputs/verify/SUBTASK-d/baseline/legacy.json", "outputs/verify/SUBTASK-d/baseline/curobo.json"],
   "stdout_log": ["outputs/verify/SUBTASK-d/baseline/legacy.log", "outputs/verify/SUBTASK-d/baseline/curobo.log"], "judge": "outputs/verify/SUBTASK-d/baseline/judge.json"},
  {"id": "V0", "command": "nohup setsid outputs/verify/SUBTASK-d/p/launch_v0.sh (gate off E3b, port 8230, ep1807 s18071 + ep1800 s18001, then p/compare_v0.py)",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/SUBTASK-d/v0/", "stdout_log": "outputs/verify/SUBTASK-d/v0/launch.out"},
  {"id": "V1", "command": "nohup setsid outputs/verify/SUBTASK-d/p/launch_v1.sh 4 (gate on E3bG = E3b server flags + --subtask-gate, ports 8231-8234, lists p/lists/E3bG_1..4.txt)",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/SUBTASK-d/runs/E3bG/", "stdout_log": "outputs/verify/SUBTASK-d/runs/launch.out"},
  {"id": "V2", "command": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa .venv-openpi-live/bin/python p/analyze_v2.py v2/analysis.json --off outputs/verify/T39/runs/E3b --on runs/E3bG --extra-off v0/E3b; python3 p/summarize_v2.py v2/analysis.json v2/summary_extra.json",
   "venv": ".venv-openpi-live (CPU)", "raw_output": ["outputs/verify/SUBTASK-d/v2/analysis.json", "outputs/verify/SUBTASK-d/v2/summary_extra.json"], "stdout_log": "outputs/verify/SUBTASK-d/v2/analysis.log"},
  {"id": "figures", "command": "p/fig_v2.py v2/analysis.json AG3S/ag3s/docs/figures/subtask-d 1800 18001 v0/v0_summary.json; same with 1925 19251 --only=scene", "stdout_log": "outputs/verify/SUBTASK-d/v2/fig.log"}],
 "code_state": {"repo": "benchmark", "head": "73dcb37", "snapshot": PART["snapshot"], "dirty_files": [],
                "client": "pi05_TO_hybrid 22d8dae rby1_bringup (archive), same as T39; rby1_bringup diff vs T39 snapshot: only __pycache__",
                "t39_gate_off_code": "o1-eval 9fab500 (T39.verify.partial.json s2_0.snapshot)", "scripts": "outputs/verify/SUBTASK-d/p/ (grep work-o1|work-o4|o1-eval: 0 lines)"},
 "setup": {"server_flags_E3b": "--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --record-constraints",
           "server_flags_E3bG": "E3b + --subtask-gate", "xla_flags": "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0",
           "client_flags": "--max-steps 600 --headless --warmup-steps 1000 --policy-seed 10*ep+rep --record-frames --record third_person_front.mp4 --view front --trajectory-out traj.npz --safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left --safe-gate reasons --safe-hold-mode fixed",
           "episodes": 24, "seeds_per_episode": 2, "parallel_servers_V1": 4, "parallel_servers_T39_S2_2": "4 (2 E3b + 2 E0)",
           "gpu_peak_total_used_mib_per_V1_server_file": gpu},
 "numbers": {
  "baseline": {b: {k: B[b][k] for k in ("violated_start", "has_target", "frame0_clearance_before_mm", "matched")} for b in B},
  "V0": {"runs": {n: {k: v[k] for k in ("run_t39", "run_new", "executed_actions_equal", "planning_actions_equal", "traj_npz_keys_differing", "traj_npz_differ_other",
                                         "frames_rows", "frames_other_leaf_diff", "manifest_other_leaf_diff", "server_n_chunks", "server_array_diff",
                                         "server_summary_other_leaf_diff", "server_summary_subtask_leaf_diff", "server_summary_control_vary_leaf_diff",
                                         "named_per_chunk", "other_values")} for n, v in V0["runs"].items()},
         "control_set_file": "outputs/verify/SUBTASK-d/v0/control_T39_s21.json (T39 S2-1 E3b_a vs E3b_b, same code, two server processes)",
         "leaves_outside_control_and_subtask": {"manifest scene.model_xml": "path string: ${AG3S_PROJECT_ROOT}-o1/... vs ${AG3S_PROJECT_ROOT}-sd/... (both snapshot entries symlink to ${AG3S_PROJECT_ROOT}/robots/rby1/description)",
                                                "server chunk_index (ep1800 s18001 only)": "T39 0..74 (1st run on server 8226) vs V0 75..149 (2nd run on server 8230); t_step 1..75 on both"},
         "lead_decision_2026-10-02": "both leaves treated as bookkeeping; V1 continued; V2 pairs chunks by (episode, seed, t_step)",
         "summary_file": "outputs/verify/SUBTASK-d/v0/v0_summary.json"},
  "V1": {"n_runs_done": sum(1 for r in runs_on.values() if r["status"] == "done"), "n_runs_expected": 48,
         "n_failed_attempts": sum(r["n_failed_attempts"] for r in runs_on.values()),
         "wall_s": {"min": min(r["wall_s"] for r in runs_on.values()), "max": max(r["wall_s"] for r in runs_on.values()),
                    "mean": sum(r["wall_s"] for r in runs_on.values()) / len(runs_on)},
         "start": "2026-10-02T19:00:00+00:00", "end": "2026-10-02T20:11:40+00:00"},
  "V2_paired_2x2_off_vs_on": A["paired_2x2"],
  "V2_summary_gate_off_T39": {k: so[k] for k in keep},
  "V2_summary_gate_on": {k: sn[k] for k in keep},
  "V2_v0_gate_off_rerun_with_subtask_record": {k: S["off_extra"][k] for k in ("n_runs", "label_x_latch", "argmax_x_latch", "delay_pick_to_place_chunks", "delay_place_to_home_chunks", "n_placed_runs_with_switch_after", "after_placed_excl_active_total", "after_placed_chunks_total")},
  "V2_divergence": X["divergence"],
  "V2_gate_blocked_before_placed": X["gate_blocked_before_placed"],
  "V2_first_adoption": {"t": X["first_adoption_t"], "same_t_all_pairs": X["first_adoption_same_t_all_pairs"], "same_manip_id_all_pairs": X["first_adoption_same_manip_id_all_pairs"]},
  "V2_pairs_decisions_identical_before_first_held": X["pairs_decisions_identical_before_first_held"],
  "V2_discordant_pairs": [p for p in X["pair_table"] if p["off"]["success"] != p["on"]["success"] or p["off"]["grasp"] != p["on"]["grasp"]],
  "V2_pair_table": X["pair_table"],
  "V2_cross_check_T39_analysis": {"T39_s23_E3b": {"n_success": 29, "n_grasp": 35, "n_place": 29, "n_hold_chunks_total": 294, "gate_action": {"execute": 3306, "hold": 294}},
                                  "this_analysis_gate_off": {"n_success": so["n_success"], "n_grasp": so["n_grasp"], "n_place": so["n_place"], "n_hold_chunks_total": so["n_hold_chunks_total"], "gate_action": so["gate_action_counts"]},
                                  "file": "outputs/verify/T39/s23/analysis.json"},
  "definitions": "outputs/verify/SUBTASK-d/p/analyze_v2.py docstring (also in v2/analysis.json 'definitions')"},
 "figures": [
  {"path": "AG3S/ag3s/docs/figures/subtask-d/subtask-d-scene-1800_s18001.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/subtask-d/subtask-d-scene-1800_s18001.json"},
  {"path": "AG3S/ag3s/docs/figures/subtask-d/subtask-d-scene-1925_s19251.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/subtask-d/subtask-d-scene-1925_s19251.json"},
  {"path": "AG3S/ag3s/docs/figures/subtask-d/subtask-d-timeline.png", "kind": "graph", "sidecar": "AG3S/ag3s/docs/figures/subtask-d/subtask-d-timeline.json"},
  {"path": "AG3S/ag3s/docs/figures/subtask-d/subtask-d-table.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/subtask-d/subtask-d-table.json"}],
 "baseline_check": {"matched": all(B[b]["matched"] for b in B), "note": "legacy 14/15 · 15/15 · -29.031048280806342; cuRobo 10/15 · 15/15 · -9.171877401271193; 15 clearance_before values equal T36 for both"},
 "verdict": "baseline matched. V0: executed actions, planning actions, traj.npz arrays (except inference_ms), latch / manipulated id / exclusion per chunk and server arrays (except chunk_index) equal for both runs. V1 48/48 runs done.",
 "not_measured": [
  "T39 gate-off server records have no 'subtask' key (o1-eval code): label x latch, transition delays and would_block exist only for gate-on runs and the 2 V0 gate-off reruns",
  "chunk time comparison: V1 ran 4 E3bG servers at once, T39 S2-2 ran 2 E3b + 2 E0 servers at once (different load); time numbers are listed, not paired",
  "ESDF slice in the scene figure is the 5 mm main tier only (coarse tier not drawn)",
  "SendUserFile not available to this agent: figures are reported by path"]}
(H / "SUBTASK-d.verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
PART["v1"] = {"done": out["numbers"]["V1"]["n_runs_done"], "end": out["numbers"]["V1"]["end"]}
PART["v2"] = {"analysis": str(D / "v2/analysis.json"), "summary_extra": str(D / "v2/summary_extra.json"), "verify_json": str(H / "SUBTASK-d.verify.json")}
PART["updated"] = out["date"]
(H / "SUBTASK-d.verify.partial.json").write_text(json.dumps(PART, indent=1, ensure_ascii=False))
print(json.dumps(out["numbers"]["V1"]), out["baseline_check"])
