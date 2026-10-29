"""SUBTASK-e — build AG3S/ag3s/docs/handoff/SUBTASK-e.verify.json from the raw files (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, datetime, pathlib
E = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-e')); H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
A = json.load(open(E / "v2/analysis.json")); X = json.load(open(E / "v2/pairs_extra.json")); V0 = json.load(open(E / "v0/v0_summary.json"))
B = json.load(open(E / "baseline/judge.json")); PART = json.load(open(H / "SUBTASK-e.verify.partial.json"))
S = A["summary"]
runs_on = {k: v for k, v in PART["runs"].items() if v["stage"] == "runs"}
gpu = []
for f in sorted((E / "runs/E3bG").glob("server_*/gpu_samples.txt")):
    vals = [int(l.split()[1]) for l in f.read_text().splitlines() if len(l.split()) == 2 and l.split()[1].isdigit()]
    gpu.append({"file": str(f), "max_total_used_mib": max(vals) if vals else None})
v0d = {}
for n in ("E3b_ep1807_s18071", "E3b_ep1800_s18001"):
    r = json.load(open(E / f"v0/compare_vsSUBTASKd_{n}.json"))
    v0d[n] = {"run_subtask_d": r["run_t39"], "run_new": r["run_new"], "executed_actions_equal": r["executed_actions_equal"], "planning_actions_equal": r["planning_actions_equal"],
              "traj_npz_keys_differing": sorted(r["traj_npz_differ"]), "frames_other_leaf_diff": r["classified"]["frames_leaf_diff"]["other"],
              "manifest_other_leaf_diff": r["classified"]["manifest_leaf_diff"]["other"], "server_summary_other_leaf_diff": r["classified"]["server_summary_leaf_diff"]["other"],
              "server_summary_subtask_leaf_diff": r["classified"]["server_summary_leaf_diff"]["subtask"], "server_array_diff_other": r["server_array_diff_other"],
              "named_per_chunk": r["named_per_chunk"], "compare_json": str(E / f"v0/compare_vsSUBTASKd_{n}.json")}
keep = ("n_runs", "n_success", "n_grasp", "n_place", "n_placed_latch", "n_runs_released", "n_hold_chunks_total", "n_runs_with_hold", "hold_kind_counts",
        "gate_action_counts", "reason_kind_counts", "regions", "snt", "blocked_counts_all", "clearance_min_mm_overall", "n_chunks_clearance_lt0",
        "from_release_clearance_min_mm_per_run", "pose", "contacts_from_ref", "contacts_all", "n_budget_hit", "to_status_counts", "n_refined_mismatch", "n_runs_segment_found")
P = "AG3S/ag3s/docs/figures/subtask-e/"
out = {
 "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
 "step": "SUBTASK-e", "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
 "runs": [
  {"id": "baseline", "command": "cwd ${AG3S_PROJECT_ROOT}-se: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-se XLA_PYTHON_CLIENT_PREALLOCATE=false .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records ${AG3S_PROJECT_ROOT}/run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms [--esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005] (serial, nothing else running; outputs/verify/SUBTASK-e/baseline/run_baseline.sh)",
   "venv": ".venv-openpi-live (.venv-ag3s not used, lead instruction)", "raw_output": ["outputs/verify/SUBTASK-e/baseline/legacy.json", "outputs/verify/SUBTASK-e/baseline/curobo.json"],
   "stdout_log": ["outputs/verify/SUBTASK-e/baseline/legacy.log", "outputs/verify/SUBTASK-e/baseline/curobo.log"], "judge": "outputs/verify/SUBTASK-e/baseline/judge.json"},
  {"id": "V0", "command": "outputs/verify/SUBTASK-e/p/launch_v0.sh (gate off E3b, port 8230, ep1807 s18071 + ep1800 s18001; p/compare_v0.py vs T39 and vs SUBTASK-d V0), started by p/launch_all.sh (setsid nohup)",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/SUBTASK-e/v0/", "stdout_log": "outputs/verify/SUBTASK-e/v0/launch.out"},
  {"id": "V1", "command": "outputs/verify/SUBTASK-e/p/launch_v1.sh 4 (gate on E3bG = E3b server flags + --subtask-gate, ports 8231-8234, lists p/lists/E3bG_1..4.txt = SUBTASK-d lists), started by p/launch_all.sh after V0 actions equal T39",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/SUBTASK-e/runs/E3bG/", "stdout_log": ["outputs/verify/SUBTASK-e/runs/launch.out", "outputs/verify/SUBTASK-e/launch_all.out"]},
  {"id": "V2", "command": "JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa .venv-openpi-live/bin/python p/analyze_e.py v2/analysis.json (off = T39/runs/E3b, on_d = SUBTASK-d/runs/E3bG, on_e = SUBTASK-e/runs/E3bG); p/pairs_extra.py v2/analysis.json v2/pairs_extra.json",
   "venv": ".venv-openpi-live (CPU)", "raw_output": ["outputs/verify/SUBTASK-e/v2/analysis.json", "outputs/verify/SUBTASK-e/v2/pairs_extra.json"], "stdout_log": "outputs/verify/SUBTASK-e/v2/analysis.log"},
  {"id": "pytest", "command": "cwd ${AG3S_PROJECT_ROOT}: JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} third_party/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q -p no:cacheprovider",
   "venv": "third_party/openpi/.venv (CPU)", "stdout_log": "outputs/verify/SUBTASK-e/pytest/pytest.log"},
  {"id": "figures", "command": "p/fig_e.py v2/analysis.json AG3S/ag3s/docs/figures/subtask-e 1800 18001 v0/v0_summary.json baseline/judge.json; same with 1828 18282 --only=scene --ts=31,32,35,75", "stdout_log": "outputs/verify/SUBTASK-e/v2/fig.log"}],
 "code_state": {"repo": "benchmark", "head": "7098e69", "code_change_commit": "a7994c0", "snapshot": PART["snapshot"], "dirty_files": [],
                "client": "pi05_TO_hybrid 22d8dae rby1_bringup (archive); client PYTHONPATH ${AG3S_PROJECT_ROOT}-se (wire.py with reason kind subtask_no_target)",
                "import_check_first_server": str(sorted((E / "v0/E3b").glob("preflight_*.import_server.json"))[0]),
                "import_check_first_client": str(sorted((E / "v0/E3b").glob("preflight_*.import_client.json"))[0]),
                "code_md5_checked_before_every_server": "p/run_chain.sh: preflight_*.code_md5.txt == p/code_md5_se.txt",
                "scripts": "outputs/verify/SUBTASK-e/p/ (copied from SUBTASK-d/p, paths work-sd -> work-se, SUBTASK-d -> SUBTASK-e)"},
 "setup": {"server_flags_E3b": "--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --record-constraints",
           "server_flags_E3bG": "E3b + --subtask-gate", "xla_flags": "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0", "xla_python_client_preallocate": "false",
           "client_flags": "--max-steps 600 --headless --warmup-steps 1000 --policy-seed 10*ep+rep --record-frames --record third_person_front.mp4 --view front --trajectory-out traj.npz --safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left --safe-gate reasons --safe-hold-mode fixed",
           "episodes": 24, "seeds_per_episode": 2, "parallel_servers_V1": 4, "parallel_servers_SUBTASK_d_V1": 4,
           "gpu_before_start_mib": 0, "gpu_peak_total_used_mib_per_V1_server_file": gpu},
 "numbers": {
  "pytest": {"passed": 1958, "skipped": 2, "failed": 0, "wall_s": 305.75, "log": "outputs/verify/SUBTASK-e/pytest/pytest.log"},
  "baseline": {b: {k: B[b][k] for k in ("violated_start", "has_target", "frame0_clearance_before_mm", "matched", "sha1")} for b in B},
  "V0_vs_T39": {"runs": {n: {k: v[k] for k in ("run_t39", "run_new", "executed_actions_equal", "planning_actions_equal", "traj_npz_keys_differing", "traj_npz_differ_other",
                                                "frames_rows", "frames_other_leaf_diff", "manifest_other_leaf_diff", "server_n_chunks", "server_array_diff",
                                                "server_summary_other_leaf_diff", "server_summary_subtask_leaf_diff", "named_per_chunk", "other_values")} for n, v in V0["runs"].items()},
                "summary_file": "outputs/verify/SUBTASK-e/v0/v0_summary.json"},
  "V0_vs_SUBTASK_d_V0": v0d,
  "V1": {"n_runs_done": sum(1 for r in runs_on.values() if r["status"] == "done"), "n_runs_expected": 48,
         "n_failed_attempts": sum(r["n_failed_attempts"] for r in runs_on.values()),
         "wall_s": {"min": min(r["wall_s"] for r in runs_on.values()), "max": max(r["wall_s"] for r in runs_on.values()),
                    "mean": sum(r["wall_s"] for r in runs_on.values()) / len(runs_on)},
         "start": min(r["start"] for r in runs_on.values()), "end": max(r["end"] for r in runs_on.values())},
  "paired_2x2": A["paired_2x2"],
  "summary_off_T39": {k: S["off"][k] for k in keep},
  "summary_on_d_SUBTASK_d": {k: S["on_d"][k] for k in keep},
  "summary_on_e_SUBTASK_e": {k: S["on_e"][k] for k in keep},
  "first_blocked_executed_on_e": {"n": len(S["on_e"]["first_blocked"]), "by_region": X["first_blocked_by_region"]["on_e"]["by_region"],
                                  "before_placed": X["first_blocked_by_region"]["on_e"]["before_placed"]},
  "first_blocked_by_region": X["first_blocked_by_region"],
  "identity_on_e_vs_on_d": {"summary": A["identity_summary"], "per_pair": A["identity_on_e_vs_on_d"]},
  "decision_wo_to_status_on_e_vs_on_d": X["decision_wo_to_status_on_e_vs_on_d"],
  "off_vs_on_e_exec": X["off_vs_on_e_exec"],
  "no_admissible_after_release_on_e": X["no_admissible_after_release_on_e"],
  "released_table": X["released_table"],
  "definitions": "outputs/verify/SUBTASK-e/p/analyze_e.py and p/pairs_extra.py docstrings (also in v2/analysis.json and v2/pairs_extra.json 'definitions')"},
 "figures": [
  {"path": P + "subtask-e-scene-1800_s18001.png", "kind": "scene", "sidecar": P + "subtask-e-scene-1800_s18001.json"},
  {"path": P + "subtask-e-scene-1828_s18282.png", "kind": "scene", "sidecar": P + "subtask-e-scene-1828_s18282.json"},
  {"path": P + "subtask-e-timeline.png", "kind": "graph", "sidecar": P + "subtask-e-timeline.json"},
  {"path": P + "subtask-e-table.png", "kind": "table", "sidecar": P + "subtask-e-table.json"}],
 "baseline_check": {"matched": all(B[b]["matched"] for b in B), "note": "legacy 14/15 · 15/15 · -29.031048280806342; cuRobo 10/15 · 15/15 · -9.171877401271193; sha1 of the 15 clearance_before values equal SUBTASK-d judge.json for both"},
 "verdict": "baseline matched. V0: executed and planning actions equal T39 and equal SUBTASK-d V0 for both runs. V1 48/48 runs done. on_e vs on_d: 48/48 pairs identical before the first on_d HOLD with subtask_gated grounding.",
 "not_measured": [
  "ready pose = 16-D measured joints at control step 0 (after 1000 warm-up steps); no separate ready-pose definition exists in the code",
  "collision = MuJoCo contacts with dist < 0 on the recorded control-row qpos (kinematic replay, mj_forward), plus server clearance_min_mm; no contact-force measurement",
  "chunk timings not compared: V1 ran 4 servers at once, T39 S2-2 ran 2 E3b + 2 E0",
  "ESDF slice in the scene figures is the 5 mm main tier only; area outside that tier window is blank",
  "SendUserFile not available to this agent: figures are reported by path",
  "pytest ran on the main checkout ${AG3S_PROJECT_ROOT} (benchmark HEAD 7098e69), not on the snapshot (tests/ is outside benchmark)"]}
(H / "SUBTASK-e.verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print("wrote", H / "SUBTASK-e.verify.json")
