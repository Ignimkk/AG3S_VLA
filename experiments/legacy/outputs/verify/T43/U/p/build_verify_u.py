"""T43 U — write AG3S/ag3s/docs/handoff/T43U.verify.json from raw/summary_u.json, raw/v_closing.json, baseline/judge.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
U = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/U'))
S = json.loads((U / "raw/summary_u.json").read_text()); V = json.loads((U / "raw/v_closing.json").read_text())
B = json.loads((U / "baseline/judge.json").read_text()); A = json.loads((U / "raw/t41_a_vs_c.json").read_text())
H = S["headline"]
grp_keys = ("n", "outcomes", "switch_away_before_close", "switch_away_before_attach", "n_with_close", "fruit_in_field_at_close",
            "a_on_fruit_rows_at_close", "defl_at_close_median_mm", "radial_at_close_median_mm", "pre_close_contact_runs",
            "pre_close_fruit_disp_median_mm", "fid_rows", "fid_same_row", "fid_max_centre0_mm", "fid_max_record_clearance_mm", "fid_errors")
out = {
 "_comment": "ag3s-verifier. numbers and paths only.",
 "step": "T43U", "date": "2026-10-05",
 "runs": [
  {"id": "baseline (legacy, CPU)", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= nice -n 19 .venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms",
   "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T43/U/baseline/legacy.json, judge.json", "stdout_log": "outputs/verify/T43/U/baseline/legacy.log"},
  {"id": "chunk index T43 V (V0-V3) + T42-2", "command": "p/scan_index.py raw/chunk_index_t43v_t42_2.json <roots> (copy of T43/S/p/scan_t42.py)", "venv": ".venv-openpi-live (CPU)",
   "raw_output": "outputs/verify/T43/U/raw/chunk_index_t43v_t42_2.json (+ T43/S/raw/chunk_index_t40t41.json, chunk_index_t42.json)", "stdout_log": "outputs/verify/T43/U/raw/scan_index.log"},
  {"id": "run list", "command": "p/runs_u.py", "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/U/raw/runs_u.json", "stdout_log": None},
  {"id": "per-run measurement", "command": "nice -n 19 p/measure_u.py 4 (T42-2: U_SNAP=${AG3S_PROJECT_ROOT}-t42c p/measure_u.py 4 T42/2/); p/fix_points.py 4", "venv": ".venv-openpi-live (CPU, 4 workers, OMP 1)",
   "raw_output": "outputs/verify/T43/U/raw/runs/*.json, raw/t41_a_vs_c.json", "stdout_log": "outputs/verify/T43/U/raw/{measure_u.log, measure_u_t42_2.log, fix_points.log}"},
  {"id": "aggregation", "command": "p/summarize_u.py; p/v_closing.py", "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/U/raw/summary_u.json, raw/v_closing.json", "stdout_log": "outputs/verify/T43/U/raw/summarize_u.log"},
  {"id": "figures", "command": "p/fig_t43u.py <timeline runs> <scene specs>", "venv": ".venv-openpi-live", "raw_output": "AG3S/ag3s/docs/figures/t43/t43u-*.json", "stdout_log": "outputs/verify/T43/U/raw/fig.log"}],
 "code_state": {"benchmark_head": "f8f8913", "root_head": "7672078", "dirty_files": ["ag3s/docs/handoff/T41a.verify.partial.json", "ag3s/docs/handoff/T43V.verify.partial.json"],
                "measured_code": "records of snapshots work-t40/t40h (T40), work-t41a/t41c (T41), work-t42/t42c (T42), work-t43v (T43 V); TO-row rebuild with main-tree fixed_input_ablation helpers; GT obstacles from work-t43v (T40/T41/T43 V configs) and work-t42c (T42-2 configs) PickPlaceObstacleManager",
                "note": "no server, no GPU, no .venv-ag3s, no commits"},
 "numbers": {
  "_comment": "Definitions: outputs/verify/T43/U/p/measure_u.py and summarize_u.py docstrings. fail = miss (closed, fruit never >= z0 + 50 mm) or slip_at_pick (lifted, back below z0 + 20 mm within 64 rows). switch away = manipulated id differs from the run's first id AND its centroid's nearest GT surface is not the fruit. fruit in field at close = at the chunk issuing the first close or the next one, the target-free layer the hand rows read holds the fruit (manipulated not the fruit, or >= 25 % of the fruit's upper vertices far (> 25 mm) from every other GT surface incl. both hands read <= 10 mm). Headline counts exclude T41 c (identical to T41 a through first close + 64 rows in 14 / 15 runs).",
  "n_runs": {"server_runs": sum(1 for k, v in S["runs"].items() if v.get("cond") != "E0"), "E0_runs": sum(1 for v in S["runs"].values() if v.get("cond") == "E0"),
             "headline_obstacle_with_close": H["obstacle_n"], "headline_clear_with_close": H["clear_n"]},
  "outcomes": S["outcomes"],
  "table_switch_away_before_attach_x_outcome": H["switch_away_before_attach"],
  "table_switch_away_before_first_close_x_outcome": H["switch_away_before_close"],
  "table_fruit_in_field_at_close_x_outcome": H["fruit_in_field_at_close"],
  "table_a_margin_on_fruit_nearest_hand_rows_at_close_x_outcome": H["a_on_fruit_rows_at_close"],
  "switch_away_runs": H["switch_away_runs"],
  "switch_away_first_after_first_close": H["switch_away_first_after_first_close"],
  "switch_away_first_targets": H["switch_away_first_targets"],
  "switch_away_first_dominant_camera": H["switch_away_first_dominant_camera"],
  "switch_away_first_left_wrist_argmax_row": H["switch_away_first_left_wrist_row"],
  "defl_at_close_mm": H["defl_at_close_mm"], "radial_at_close_mm": H["radial_at_close_mm"],
  "pre_close_contact_x_outcome": H["pre_close_contact"], "pre_close_fruit_disp_mm": H["pre_close_fruit_disp_mm"],
  "pairs_obstacle_vs_clear": S["pairs_counts"], "pairs_by_family": S["pairs_by_family"],
  "groups": {g: {k: v[k] for k in grp_keys} for g, v in S["groups"].items()},
  "t43v_closing_side_by_side": {k: v for k, v in V.items() if k != "doc"},
  "t41_a_vs_c_identical_through_close_plus_64": sum(1 for v in A.values() if v["identical_through_close_plus_64"]),
  "t41_a_vs_c_pairs": len(A),
  "rebuild_fidelity": {"hand_row_chunks": sum(v["fid_rows"] for v in S["groups"].values()), "same_worst_row_as_recorded": sum(v["fid_same_row"] for v in S["groups"].values()),
                       "max_sphere_centre_diff_mm": max(v["fid_max_centre0_mm"] for v in S["groups"].values()),
                       "max_plain_clearance_diff_mm": max(v["fid_max_record_clearance_mm"] for v in S["groups"].values()),
                       "note_not_same_row": "172 chunks differ: 170 in T43 V V2 (gripper-cover rows not rebuilt: 139 hurdle, 31 clear) and 1 each in T41 a / T41 c (same chunk)"},
  "grasp_tip_alignment_field_present": 0,
 },
 "figures": [
  {"path": "AG3S/ag3s/docs/figures/t43/t43u-scene.png", "kind": "scene"},
  {"path": "AG3S/ag3s/docs/figures/t43/t43u-timeline.png", "kind": "graph"},
  {"path": "AG3S/ag3s/docs/figures/t43/t43u-table.png", "kind": "table"}],
 "baseline_check": {"matched": B["matched"], "note": f"legacy (CPU, .venv-openpi-live) {B['violated_start']} / {B['has_target']} / {B['frame0_clearance_before_mm']}; sha1 equal to T41c / T43V: {B['sha1_equals_T41c_T43V']}. cuRobo baseline not run (no GPU in this lane)."},
 "verdict": "legacy baseline matches. Facts only; judgement by lead and user.",
 "not_measured": [
  "cuRobo baseline (GPU not allowed in lane U)",
  "T43 Q path rows and T43 R gripper-cover rows are not rebuilt (recorded worst rows only); hand-row facts are the 122-sphere waypoint rows",
  "grasp.tip_alignment: absent in every record (all runs predate T43 T); fingertip-to-fruit computed from MuJoCo instead",
  "per-camera 3D lift of each camera's own argmax cell (records hold one merged attention cloud); the global peak point is lifted",
  "T42-1 pair for T42-2 ep1215 (none) and ep1925 (T42-1 has s19253, T42-2 s19251)"]}
p = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43U.verify.json'))
p.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=str))
print("wrote", p)
