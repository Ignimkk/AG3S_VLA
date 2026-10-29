from common.workspace import workspace_path as _ag3s_workspace_path
import json
from pathlib import Path
O = Path(_ag3s_workspace_path('outputs/verify/T20'))
H = Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
base = json.load(open(O / "baseline.json")); fr = base["frames"]
summ = json.load(open(O / "t14_sweep_summary.json"))
fid = json.load(open(O / "t14_fidelity.json"))
syn = json.load(open(O / "synth_summary.json"))
py = open(O / "pytest_ag3s.log").read().strip().splitlines()[-1]
r05 = json.load(open(O / "t14_replay_min0.05_lostNone.json"))
V = "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} third_party/openpi/.venv/bin/python "
d = {
 "step": "T20", "date": "2026-09-28",
 "runs": [
  {"id": "baseline", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --out-json outputs/verify/T20/baseline.json",
   "venv": ".venv-ag3s", "raw_output": "outputs/verify/T20/baseline.json", "stdout_log": "outputs/verify/T20/baseline.log",
   "concurrent_load": "another session's AG3S.trajopt.experiments.fixed_input_ablation (pid 4141313, ~350% CPU) was running during this baseline; feasible/violated counts are therefore not comparable"},
  {"id": "pytest_ag3s", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python -m pytest -q -p no:cacheprovider tests/ag3s", "venv": ".venv-ag3s", "stdout_log": "outputs/verify/T20/pytest_ag3s.log"},
  {"id": "t14_fidelity", "command": V + "outputs/verify/T20/replay_t14.py fidelity", "venv": "third_party/openpi/.venv", "raw_output": "outputs/verify/T20/t14_fidelity.json", "stdout_log": "outputs/verify/T20/t14_fidelity.log",
   "what": "ground_target without confirm on every chunk; seeds / status / n_clusters / best score compared with the chunk's recorded seed_indices and summary_json"},
  {"id": "t14_replay_sweep", "command": V + "outputs/verify/T20/replay_t14.py sweep <min_score> <lost_frames>  (12 settings)", "venv": "third_party/openpi/.venv",
   "raw_output": "outputs/verify/T20/t14_replay_min{m}_lost{l}.json; summary outputs/verify/T20/t14_sweep_summary.json (summarize_t14.py); per-frame lines in handoff/T20.verify.partial.json (kind=t14_replay)",
   "stdout_log": "outputs/verify/T20/t14_sweep_lost{None,10,20}.log"},
  {"id": "synthetic", "command": V + "outputs/verify/T20/synth_t20.py run", "venv": "third_party/openpi/.venv",
   "raw_output": "outputs/verify/T20/synth_rows.json; summary outputs/verify/T20/synth_summary.json (summarize_synth.py); calibration outputs/verify/T20/synth_calib.log; per-frame lines in handoff/T20.verify.partial.json (kind=synthetic)"},
  {"id": "figures", "command": V + "outputs/verify/T20/fig_t20.py", "venv": "third_party/openpi/.venv"}],
 "code_state": {"repo": "benchmark", "head": "368d723", "dirty_files": ["ag3s/config.py", "ag3s/configs/default.yaml", "ag3s/configs/rby1_three_camera.yaml", "ag3s/constraints/to_adapter.py", "ag3s/experiments/sources/constraint_record.py", "ag3s/runtime/frame_record.py", "ag3s/runtime/multiview.py", "ag3s/runtime/pipeline.py", "ag3s/stages/robot_filter.py", "ag3s/stages/target_grounding.py", "ag3s/types.py", "trajopt/README.md", "trajopt/client.py", "trajopt/config.py", "trajopt/problem.py", "trajopt/refiner.py", "trajopt/safe_policy.py", "trajopt/serve_safe.py", "trajopt/sqp.py", "trajopt/wire.py"],
   "workspace_untracked_tests": ["tests/ag3s/test_manipulated_identity.py", "tests/ag3s/test_self_filter_target_guard.py"]},
 "numbers": {
  "baseline_working_tree": {"frames": len(fr), "violated_start": sum(1 for f in fr if f["clearance_before_mm"] < 0), "has_target": sum(bool(f["target"]) for f in fr),
     "frame0_clearance_before_mm": fr[0]["clearance_before_mm"], "n_filter_spheres": base["ag3s"]["n_filter_spheres"], "planned_horizon": base["trajopt"]["planned_horizon"],
     "status_counts_not_baseline": {"feasible": sum(f["status"] == "feasible" for f in fr), "violated": sum(f["status"] == "violated" for f in fr)}},
  "baseline_reference_HEAD_from_T19": {"violated_start": 13, "has_target": 15, "frame0_clearance_before_mm": 0.18597810857728936},
  "pytest_tests_ag3s": py,
  "t14_replay_inputs": {
     "source": "outputs/live_test/20260928_t14/server_constraints/run_0000/chunk_{k:05d}.npz, seq = k+1, k = 0..30 replayed in order",
     "attention_cloud": "attention_cloud (float32 -> float64); equal to filtered_cloud in 31/31 chunks",
     "attention": "attention_values", "raw_attention": "attention_raw", "exclude_mask": "support_mask",
     "config": "AG3SConfig() defaults: clustering (dbscan eps 0.03 min_points 20, target_confirm_frames 3, tolerance 0.06), attention.seed_percentile 95 / seed_threshold None, geometry.min_radius 0.005 (serve_safe builds AG3SConfig.from_dict without clustering/attention/geometry sections)",
     "timestamp": "chunk t_step",
     "differences_from_live": ["point coordinates recorded as float32 (live used float64)", "camera provenance (_with_camera_provenance) not applied", "live ran the pre-T20 TargetConfirm; replay runs the working-tree one", "cloud is the old self-filter output (apple erased for seq 12-21, T14.audit)"]},
  "t14_replay_fidelity": {"seeds_equal": sum(r["seeds_equal"] for r in fid), "status_equal": sum(r["status_replay"] == r["status_live"] for r in fid),
     "n_clusters_equal": sum(r["n_clusters_replay"] == r["n_clusters_live"] for r in fid), "frames": len(fid),
     "best_score_max_abs_diff": max(abs(r["best_score_replay"] - r["best_score_live"]) for r in fid)},
  "t14_leader_scores_seq12_21": {str(r["seq"]): [r["clusters_replay"][0]["nearest"][0], r["clusters_replay"][0]["score"]] for r in fid if 12 <= r["seq"] <= 21},
  "t14_leader_scores_seq23_25": {str(r["seq"]): [r["clusters_replay"][0]["nearest"][0], r["clusters_replay"][0]["score"]] for r in fid if 23 <= r["seq"] <= 25},
  "t14_live_target_nearest_seq10_22": {str(r["seq"]): r["live_target_nearest"] for r in fid if 10 <= r["seq"] <= 22},
  "t14_sweep": summ,
  "t14_min0.05_lostNone_occluded_target": {"target_to_apple_mm_seq12_21": sorted({round(r["target_to_apple_mm"], 6) for r in r05 if 12 <= r["seq"] <= 21}),
     "manipulated_occluded_metric_seq12_21": sorted({r["target_occluded_flag"] for r in r05 if 12 <= r["seq"] <= 21}),
     "last_geometry_n_points": r05[11]["manipulated"]["n_points"], "last_seen_frame": r05[11]["manipulated"]["last_seen_frame"],
     "streak_seq23_25": [r["streak"] for r in r05 if 23 <= r["seq"] <= 25]},
  "t14_smallest_min_score_in_sweep_with_seq12_21_id0_occluded": min(v["min_score"] for v in summ.values() if v["seq12_21_all_id0"] and v["seq12_21_all_occluded"]),
  "t14_max_occluded_age": {k: v["max_occluded_age_all"] for k, v in summ.items()},
  "synthetic_fixture": "blobs = tests/ag3s/test_manipulated_identity._lattice (27 pts, 8 mm), CFG = ClusteringConfig(min_points=10), seed_threshold 0.005, plus 2 support-excluded points at z=2 m carrying the raw-attention peak; APPLE (0.5,0,0.8), CRATE (0.5,0.5,0.8) from the test module",
  "synthetic_A_B": [dict({k: v for k, v in x.items() if k not in ("states", "ids", "ages", "modes")}, sequence=s) for s in "AB" for x in syn[s]],
  "synthetic_C": [{k: v for k, v in x.items() if k != "modes"} for x in syn["C"]]},
 "figures": [
  {"path": "AG3S/ag3s/docs/figures/t20/t20-t14-topview.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/t20/t20-t14-topview.json"},
  {"path": "AG3S/ag3s/docs/figures/t20/t20-t14-replay-state.png", "kind": "graph", "sidecar": "AG3S/ag3s/docs/figures/t20/t20-t14-replay-state.json"},
  {"path": "AG3S/ag3s/docs/figures/t20/t20-switch-rule-table.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/t20/t20-switch-rule-table.json"}],
 "baseline_check": {"matched": False, "matched_HEAD": True,
   "note": "13/15 · has_target 15/15 · frame0 +0.18597810857728936 mm — equal to the HEAD value in T19.verify.json; skill reference (14/15 · +0.15718632962849477) not reproduced, as in T19"},
 "verdict": "기준선: HEAD 값과 일치 (skill 기준값과는 어긋남, T19 와 같음). tests/ag3s: " + py,
 "not_measured": ["pipeline-level replay (AG3S.step on T14 depth: fine window / target_field_exclude / guard centre with the manipulated object) — only ground_target + TargetConfirm were replayed",
   "T14 replay with the T19 self-filter (apple kept in cloud) — cloud is the recorded old-filter one",
   "tests/trajopt (not requested)"],
 "process_note": "verifier read T20.impl.md in full, including the section '내가 기대하는 결과', before measuring"}
json.dump(d, open(O / "T20.verify.programmatic.json", "w"), indent=1, ensure_ascii=False)
print(d["numbers"]["baseline_working_tree"], d["numbers"]["t14_replay_fidelity"], d["numbers"]["t14_smallest_min_score_in_sweep_with_seq12_21_id0_occluded"])
print(d["numbers"]["t14_leader_scores_seq12_21"])
