"""SUBTASK-f — build AG3S/ag3s/docs/handoff/SUBTASK-f.verify.json from the raw files (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, datetime, pathlib, re
E = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-f')); H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
A = json.load(open(E / "v2/analysis.json")); V0 = json.load(open(E / "v0/v0_summary.json"))
B = json.load(open(E / "baseline/judge.json")); PART = json.load(open(H / "SUBTASK-f.verify.partial.json"))
S = A["summary"]
runs_on = {k: v for k, v in PART["runs"].items() if v["stage"] == "runs"}
gpu, mem = [], []
for f in sorted((E / "runs/E3bG").glob("server_*/gpu_samples.txt")):
    vals = [int(l.split()[1]) for l in f.read_text().splitlines() if len(l.split()) == 2 and l.split()[1].isdigit()]
    gpu.append({"file": str(f), "max_total_used_mib": max(vals) if vals else None})
for f in sorted((E / "runs/E3bG").glob("server_*/mem_samples.txt")):
    rows = [l.split() for l in f.read_text().splitlines() if len(l.split()) == 3]
    mem.append({"file": str(f), "max_cgroup_current_gib": max(int(r[1]) for r in rows) / 2**30 if rows else None,
                "max_server_rss_gib": max(int(r[2]) for r in rows) / 2**20 if rows else None})
pt = (E / "pytest/pytest.log").read_text()
m = re.search(r"(\d+) passed, (\d+) skipped.* in ([\d.]+)s", pt)
keep = ("n_runs", "n_success", "n_grasp", "n_place", "n_placed_latch", "n_runs_released", "n_hold_chunks_total", "n_runs_with_hold", "hold_kind_counts",
        "gate_action_counts", "reason_kind_counts", "regions", "after_placed", "trailing_hold", "snt_gs_counts", "new", "old_subtask_gated",
        "clearance_min_mm_overall", "n_chunks_clearance_lt0", "pose", "contacts_from_ref", "contacts_all", "n_budget_hit", "to_status_counts",
        "n_refined_mismatch", "n_runs_segment_found")
P = "AG3S/ag3s/docs/figures/subtask-f/"
pairs_rows = {}
for k, v in A["identity_on_f_vs_on_e"].items():
    if v["gate_first_t"] is not None:
        pairs_rows[k] = {f: v[f] for f in ("t_na_e", "gate_first_t", "decision_first_t", "decision_fields_at_first", "reason_first_t", "exec_first_chunk", "exec_first_step",
                                            "planning_actions_first_t", "refined_chunk_first_t", "rows_before_gate_first", "rows_at_gate_first",
                                            "on_e_at_gate_first_is_no_admissible_hold", "on_e_at_gate_first_home_placed_seen", "on_f_at_gate_first_new_chunk")}
out = {
 "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
 "step": "SUBTASK-f", "date": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
 "runs": [
  {"id": "pytest", "command": "cwd ${AG3S_PROJECT_ROOT}: JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} third_party/openpi/.venv/bin/python -m pytest tests/trajopt tests/ag3s -q -p no:cacheprovider",
   "venv": "third_party/openpi/.venv (CPU)", "stdout_log": "outputs/verify/SUBTASK-f/pytest/pytest.log"},
  {"id": "baseline", "command": "outputs/verify/SUBTASK-f/baseline/run_baseline.sh (cwd ${AG3S_PROJECT_ROOT}-sf, PYTHONPATH=${AG3S_PROJECT_ROOT}-sf, .venv-openpi-live, esdf_rollout --records ${AG3S_PROJECT_ROOT}/run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms [--esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005]); legacy then cuRobo, serial",
   "venv": ".venv-openpi-live (.venv-ag3s not used)", "raw_output": ["outputs/verify/SUBTASK-f/baseline/legacy.json", "outputs/verify/SUBTASK-f/baseline/curobo.json"],
   "stdout_log": ["outputs/verify/SUBTASK-f/baseline/legacy.log", "outputs/verify/SUBTASK-f/baseline/curobo.log"], "judge": "outputs/verify/SUBTASK-f/baseline/judge.json",
   "concurrent_processes_at_start": "outputs/verify/SUBTASK-f/baseline/concurrent_at_start.txt"},
  {"id": "V0", "command": "outputs/verify/SUBTASK-f/p/launch_v0.sh (gate off E3b, port 8230, ep1807 s18071 + ep1800 s18001; p/compare_v0.py vs T39 and vs SUBTASK-e V0)", "venv": ".venv-openpi-live",
   "raw_output": "outputs/verify/SUBTASK-f/v0/", "stdout_log": "outputs/verify/SUBTASK-f/v0/launch.out"},
  {"id": "V1", "command": "outputs/verify/SUBTASK-f/p/launch_v1.sh 4 (gate on E3bG = E3b + --subtask-gate, ports 8231-8234, lists p/lists/E3bG_1..4.txt = SUBTASK-e lists); first attempt 12:06-12:21 (3 runs done, then all servers OOM-killed by the pod memory cgroup), resumed by p/launch_v1_resume.sh 12:34-13:45",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/SUBTASK-f/runs/E3bG/", "stdout_log": ["outputs/verify/SUBTASK-f/runs/launch.out", "outputs/verify/SUBTASK-f/runs/launch_resume.out", "outputs/verify/SUBTASK-f/resume.out", "outputs/verify/SUBTASK-f/chain.out"]},
  {"id": "V2", "command": "JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa .venv-openpi-live/bin/python p/analyze_f.py v2/analysis.json (off = T39/runs/E3b, on_e = SUBTASK-e/runs/E3bG, on_f = SUBTASK-f/runs/E3bG)",
   "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/SUBTASK-f/v2/analysis.json", "stdout_log": "outputs/verify/SUBTASK-f/v2/analysis.log"},
  {"id": "figures", "command": "p/fig_f.py v2/analysis.json AG3S/ag3s/docs/figures/subtask-f 1828 18282 v0/v0_summary.json baseline/judge.json; p/fig_f.py ... 1976 19761 --only=scene --ts=35,36,47,75", "stdout_log": "outputs/verify/SUBTASK-f/v2/fig.log"}],
 "code_state": {"repo": "benchmark", "head": "af168a8", "code_change_commit": "ce92fd1",
   "snapshot": {"root": "${AG3S_PROJECT_ROOT}-sf", "benchmark": "git -C ${AG3S_PROJECT_ROOT}/benchmark archive af168a856b06ef5886c67e37f40d1eff34edb242 (symlink AG3S/ag3s/asset/data -> ${AG3S_PROJECT_ROOT}/AG3S/ag3s/asset/data)",
                "pi05_TO_hybrid": "git -C ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid archive 22d8dae4f2cf5632b3f68b505cf18880033e6ada rby1_bringup rby1_manipulation rby1_description; other 15 top-level entries symlinked to ${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/* as in ${AG3S_PROJECT_ROOT}-se",
                "difference_vs_work_se_layout": "rby1_manipulation and rby1_description are archives here (symlinks in work-se); main checkout pi05_TO_hybrid has uncommitted T40 edits in rby1_bringup/pi05_infer.py, rby1_manipulation/.../pick_place_obstacles.py, rby1_description/.../scenes/pick_place_obstacles.xml",
                "extra_symlink": {"${AG3S_PROJECT_ROOT}-sf/src": "${AG3S_PROJECT_ROOT}-sf/pi05_TO_hybrid"},
                "code_md5_file": "outputs/verify/SUBTASK-f/p/code_md5_sf.txt", "code_md5_n_files": 395, "code_md5_file_md5": "92e4fb04ab4aa557822db05844013dae",
                "code_md5_diff_vs_work_se_same_file_set": "outputs/verify/SUBTASK-f/s0/code_md5_se_vs_sf.diff",
                "files_differing_vs_work_se": ["AG3S/ag3s/types.py", "AG3S/trajopt/safe_policy.py", "AG3S/trajopt/wire.py"],
                "diff_vs_work_se": "outputs/verify/SUBTASK-f/s0/diff_benchmark_se_vs_sf.txt",
                "pick_place_obstacles_py_md5_equal_work_se_record": "8550a1455ebded4c10b006a76a847716"},
   "import_check": {"server": "outputs/verify/SUBTASK-f/s0/import_server.json", "client": "outputs/verify/SUBTASK-f/s0/import_client.json",
                    "all_benchmark_under_work_sf": True, "client_rby1_manipulation_under_work_sf": True,
                    "openpi_client_path": _ag3s_workspace_path('third_party/openpi/packages/openpi-client/src (symlink tree, same as work-se)')},
   "code_md5_and_import_checked_before_every_server": "p/run_chain.sh preflight_*.code_md5.txt == p/code_md5_sf.txt; preflight_*.import_{server,client}.json",
   "main_checkout_git_state_changed": False, "scripts": "outputs/verify/SUBTASK-f/p/ (copied from SUBTASK-e/p, paths work-se -> work-sf)"},
 "setup": {"server_flags_E3b": "--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --record-constraints",
           "server_flags_E3bG": "E3b + --subtask-gate", "xla_flags": "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0", "xla_python_client_preallocate": "false",
           "client_flags": "--max-steps 600 --headless --warmup-steps 1000 --policy-seed 10*ep+rep --record-frames --record third_person_front.mp4 --view front --trajectory-out traj.npz --safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left --safe-gate reasons --safe-hold-mode fixed",
           "episodes": 24, "seeds_per_episode": 2, "parallel_servers_V1": 4, "ports_used": [8230, 8231, 8232, 8233, 8234],
           "gpu_before_start_mib": 0, "gpu_peak_total_used_mib_per_V1_server_file": gpu, "host_memory_per_V1_server_file": mem,
           "pod_memory_cgroup_max_bytes": 85899345920,
           "oom_incident": {"time": "2026-10-03T12:21", "what": "memory cgroup OOM kill (dmesg 'Memory cgroup out of memory'); servers 8231-8233 killed mid-run, 8234 died during start",
                            "concurrent": "51 processes outputs/verify/T40/F/p/search.py (CPU, ~1.45-1.52 GB RSS each), cgroup memory.current 79 GiB at 12:27",
                            "killed_runs_redone": ["E3bG_ep1863_s18631", "E3bG_ep1828_s18281", "E3bG_ep1807_s18071"], "killed_run_dirs": "outputs/verify/SUBTASK-f/runs/E3bG/failed/",
                            "truncated_chunk_files_skipped_in_analysis": A["skipped_unreadable_chunk_files"],
                            "n_server_segments_for_seed_not_1": A["n_server_segments_for_seed"]}},
 "numbers": {
  "pytest": {"passed": int(m.group(1)), "skipped": int(m.group(2)), "failed": 0, "wall_s": float(m.group(3)), "log": "outputs/verify/SUBTASK-f/pytest/pytest.log"},
  "baseline": {b: {k: B[b][k] for k in ("violated_start", "has_target", "frame0_clearance_before_mm", "matched", "sha1")} for b in B},
  "V0_vs_T39": {"runs": {n: {k: v[k] for k in ("run_t39", "run_new", "executed_actions_equal", "planning_actions_equal", "traj_npz_keys_differing", "traj_npz_differ_other",
                                                "frames_rows", "frames_other_leaf_diff", "manifest_other_leaf_diff", "server_n_chunks", "server_array_diff",
                                                "server_summary_other_leaf_diff", "server_summary_subtask_leaf_diff", "named_per_chunk", "other_values")} for n, v in V0["runs"].items()},
                "summary_file": "outputs/verify/SUBTASK-f/v0/v0_summary.json"},
  "V0_vs_SUBTASK_e_V0": V0.get("vs_SUBTASK_e_v0"),
  "V1": {"n_runs_done": sum(1 for r in runs_on.values() if r["status"] == "done"), "n_runs_expected": 48,
         "n_failed_attempts_moved_to_failed": sum(r["n_failed_attempts"] for r in runs_on.values()),
         "wall_s": {"min": min(r["wall_s"] for r in runs_on.values()), "max": max(r["wall_s"] for r in runs_on.values()),
                    "mean": sum(r["wall_s"] for r in runs_on.values()) / len(runs_on)},
         "start": min(r["start"] for r in runs_on.values()), "end": max(r["end"] for r in runs_on.values())},
  "paired_2x2": A["paired_2x2"],
  "summary_off_T39": {k: S["off"][k] for k in keep},
  "summary_on_e_SUBTASK_e": {k: S["on_e"][k] for k in keep},
  "summary_on_f_SUBTASK_f": {k: S["on_f"][k] for k in keep},
  "identity_on_f_vs_on_e": {"summary": A["identity_summary"], "pairs_with_difference": pairs_rows,
                            "n_pairs_fully_identical": A["identity_summary"]["n_fully_identical"]},
  "off_vs_on_f_exec": {"n_identical": A["off_vs_on_f_exec"]["n_identical"], "first_chunk_minus_t_rel_f": A["off_vs_on_f_exec"]["first_chunk_minus_t_rel_f"]},
  "changed_runs_table": A["changed_runs_table"],
  "definitions": "outputs/verify/SUBTASK-f/p/analyze_f.py docstring (also v2/analysis.json 'definitions'); per-run base numbers p/analyze_v2.py + p/analyze_e.py (copied unchanged from SUBTASK-e)"},
 "figures": [
  {"path": P + "subtask-f-scene-1828_s18282.png", "kind": "scene", "sidecar": P + "subtask-f-scene-1828_s18282.json"},
  {"path": P + "subtask-f-scene-1976_s19761.png", "kind": "scene", "sidecar": P + "subtask-f-scene-1976_s19761.json"},
  {"path": P + "subtask-f-timeline.png", "kind": "graph", "sidecar": P + "subtask-f-timeline.json"},
  {"path": P + "subtask-f-table.png", "kind": "table", "sidecar": P + "subtask-f-table.json"}],
 "baseline_check": {"matched": all(B[b]["matched"] for b in B), "note": "legacy 14/15 · 15/15 · -29.031048280806342; cuRobo 10/15 · 15/15 · -9.171877401271193; sha1 of the 15 clearance_before values equal SUBTASK-e judge.json for both"},
 "verdict": "baseline matched. V0: executed and planning actions equal T39 and equal SUBTASK-e V0 for both runs. V1 48/48 runs done. on_f vs on_e: 38/48 pairs fully identical; 10/10 differing pairs identical before t_na_e and first differ at t_na_e.",
 "not_measured": [
  "ready pose = 16-D measured joints at control step 0 (after 1000 warm-up steps); no separate ready-pose definition exists in the code",
  "collision = MuJoCo contacts with dist < 0 on the recorded control-row qpos (kinematic replay, mj_forward), plus server clearance_min_mm; no contact-force measurement",
  "chunk timings not compared: V1 ran 4 servers at once and other sessions' CPU jobs ran during part of V1",
  "'held to the end' is defined here as trailing HOLD streak reaching t=75 with every streak chunk grounding no_admissible; lead's figure 4 (SUBTASK-e) is not reproduced by a definition in the SUBTASK-e scripts",
  "ESDF slice in the scene figures is the 5 mm main tier only; area outside that tier window is blank",
  "SendUserFile not available to this agent: figures are reported by path",
  "pytest ran on the main checkout ${AG3S_PROJECT_ROOT} (benchmark HEAD af168a8, .py files identical to the snapshot), not on the snapshot"]}
(H / "SUBTASK-f.verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print("wrote", H / "SUBTASK-f.verify.json")
