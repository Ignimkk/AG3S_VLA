from common.workspace import workspace_path as _ag3s_workspace_path
import json, subprocess, sys
H = _ag3s_workspace_path('AG3S/ag3s/docs/handoff'); V = _ag3s_workspace_path('outputs/verify/T19')
rows = [json.loads(l) for l in open(f"{H}/T19.verify.partial.json")]
base = json.load(open(f"{V}/baseline.json")); head = json.load(open(f"{V}/baseline_HEAD.json"))
def bsum(d):
    f = d["frames"]
    return {"frames": len(f), "violated_start": sum(x["clearance_before_mm"] < 0 for x in f),
            "has_target": sum(bool(x["target"]) for x in f), "frame0_clearance_before_mm": f[0]["clearance_before_mm"],
            "n_filter_spheres": d["ag3s"]["n_filter_spheres"], "planned_horizon": d["trajopt"]["planned_horizon"],
            "status_counts_not_baseline": {s: sum(x["status"] == s for x in f) for s in {x["status"] for x in f}}}
table = {}
for r in rows:
    table.setdefault(r["setting"], {}).setdefault(f"seq{r['seq']}_t{r['t']}", {})[r["camera"]] = {
        k: r[k] for k in ("apple_px_visible", "apple_px_passed", "robot_px_visible", "robot_px_passed",
                          "robot_px_passed_in_guard", "robot_by_body", "cloud_passed_within_50mm_of_apple",
                          "cloud_passed_within_50mm_by_body", "cloud_total", "cloud_passed",
                          "guard_centre", "guard_source", "guard_centre_to_apple_mm", "guard_chunk_target_n_points")} | {
        "n_guard_protected": r["filter_stats"]["n_guard_protected"], "guard_active": r["filter_stats"]["guard_active"]}
body_tot = {}
for r in rows:
    for b, v in r["robot_by_body"].items():
        t = body_tot.setdefault(r["setting"], {}).setdefault(b, [0, 0]); t[0] += v["visible"]; t[1] += v["passed"]
tests = json.load(open(f"{V}/pytest_ag3s.json")) if __import__("os").path.exists(f"{V}/pytest_ag3s.json") else None
out = {
 "step": "T19", "date": "2026-09-28",
 "runs": [
  {"id": "baseline", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms",
   "venv": ".venv-ag3s", "raw_output": "outputs/verify/T19/baseline.json", "stdout_log": "outputs/verify/T19/baseline.log", "code": "working tree (dirty)"},
  {"id": "baseline_HEAD", "command": "same, PYTHONPATH=<scratchpad>/head_ws:${AG3S_PROJECT_ROOT}, cwd=<scratchpad>/head_ws, benchmark = `git archive HEAD` (no uncommitted changes)",
   "venv": ".venv-ag3s", "raw_output": "outputs/verify/T19/baseline_HEAD.json", "stdout_log": "outputs/verify/T19/baseline_HEAD.log", "code": "benchmark HEAD 368d723"},
  {"id": "selffilter_replay", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} third_party/openpi/.venv/bin/python outputs/verify/T19/measure_t19.py",
   "venv": "third_party/openpi/.venv", "raw_output": "AG3S/ag3s/docs/handoff/T19.verify.partial.json (1 line per setting x seq x camera)", "stdout_log": "outputs/verify/T19/measure_t19.log",
   "code_path": "AG3S.ag3s.stages.robot_filter.filter_robot_points -> self_filter_mask -> robot_sphere_mask; inflation from resolve_self_filter_inflation(PointCloudConfig)",
   "robot_model": "load_rby1(extra_capsules=gap_filling_capsules(model_transport), fixed_joint_values={head_0:0, head_1:0}) — 218 spheres",
   "scene": "MuJoCo model_transport.xml, qpos from outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl planning seq 10-14; ground truth = TransportScene.capture(cam).body_ids; every valid pixel back-projected (no downsampling)"},
  {"id": "attribution", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} third_party/openpi/.venv/bin/python outputs/verify/T19/attribute_t19.py",
   "venv": "third_party/openpi/.venv", "raw_output": "outputs/verify/T19/attribution.json", "stdout_log": "outputs/verify/T19/attribute_t19.log",
   "what": "setting B: apple px removed per link (robot_sphere_mask called on that link's spheres only) and apple px inside a bare (inflation 0) sphere"},
  {"id": "figures", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} third_party/openpi/.venv/bin/python outputs/verify/T19/fig_t19.py", "venv": "third_party/openpi/.venv"},
 ],
 "code_state": {"repo": "benchmark", "head": subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                "dirty_files": [l[3:] for l in subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "status", "--short"], capture_output=True, text=True).stdout.splitlines() if l.startswith(" M")]},
 "numbers": {
  "settings": {
   "A": {"self_filter_inflation_by_link": {}, "self_filter_target_guard_radius": 0.0},
   "B": {"self_filter_inflation_by_link": {"gripper": 0.010}, "self_filter_target_guard_radius": 0.0, "other_links": 0.05},
   "C": {"self_filter_inflation_by_link": {"gripper": 0.010}, "self_filter_target_guard_radius": 0.06, "guard_centre": "chunk_{seq-1}.npz target_centroid (server record)"},
   "C2": {"self_filter_inflation_by_link": {"gripper": 0.010}, "self_filter_target_guard_radius": 0.06, "guard_centre": "chunk_{seq-2}.npz target_centroid (supplementary, not requested)"},
   "D": {"self_filter_inflation_by_link": {"gripper": 0.010}, "self_filter_target_guard_radius": 0.06, "guard_centre": "object_poses.apple.pos"}},
  "gripper_group_links": ["ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2"],
  "poses": {"seq10": 72, "seq11": 80, "seq12": 88, "seq13": 96, "seq14": 104},
  "apple_pos_m": rows[0] and None,
  "table": table,
  "robot_px_by_body_total_over_poses_and_cameras_visible_passed": body_tot,
  "attribution_setting_B": json.load(open(f"{V}/attribution.json")),
  "inflation_tables": json.load(open(f"{V}/inflation_tables.json")),
  "baseline_working_tree": bsum(base),
  "baseline_HEAD_368d723": bsum(head),
  "baseline_skill_reference": {"violated_start": 14, "has_target": 15, "frame0_clearance_before_mm": 0.15718632962849477,
                               "note": "T7a.verify.json: this value reproduces at 5c5291e^ (planned_horizon 32); HEAD since 8d70cdd gives 13/15 · 0.18597810857728936 (planned_horizon 8)"},
  "pytest_tests_ag3s": tests,
 },
 "figures": [
  {"path": "AG3S/ag3s/docs/figures/t19/t19-selffilter-scene-t96.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/t19/t19-selffilter-scene-t96.json"},
  {"path": "AG3S/ag3s/docs/figures/t19/t19-selffilter-per-pose.png", "kind": "graph", "sidecar": "AG3S/ag3s/docs/figures/t19/t19-selffilter-per-pose.json"},
  {"path": "AG3S/ag3s/docs/figures/t19/t19-selffilter-settings-table.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/t19/t19-selffilter-settings-table.json"}],
 "baseline_check": {"matched": False, "matched_HEAD": True,
   "note": "working tree = HEAD 368d723 (git archive, no uncommitted changes): 13/15 · has_target 15/15 · frame0 +0.18597810857728936 mm — identical to each other and to T7a's HEAD value; skill reference 14/15 · +0.15718632962849477 mm not reproduced by either"},
 "verdict": "baseline: working tree == HEAD (pre-SQP numbers identical to the last digit); != skill reference value (same as T7a). Setting A reproduces T14.audit t=96 counts (wrist_cam_l 2497 visible / 0 passed, zed_left 378 / 0).",
 "not_measured": ["T1 gripper leak reference value (task #2 comparison) — not looked up",
                  "depth-noise / real-sensor case: MuJoCo depth is exact",
                  "server downsampling / range cut: every pixel back-projected",
                  "latency of self_filter with per-sphere inflation"],
}
fr = [json.loads(l) for l in open(_ag3s_workspace_path('outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl'))]
out["numbers"]["apple_pos_m"] = [x for x in fr if x.get("kind") == "planning" and x["seq"] == 13][0]["object_poses"]["apple"]["pos"]
out["not_measured"].insert(0, "base body: 0 px visible in zed_left / wrist_cam_l / wrist_cam_r at seq 10-14 (not in robot_by_body) -> base leak not measurable at these poses")
json.dump(out, open(f"{H}/T19.verify.json", "w"), indent=1, ensure_ascii=False)
print("wrote", len(rows), "rows;", body_tot["A"])
