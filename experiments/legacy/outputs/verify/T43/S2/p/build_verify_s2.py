"""T43 S2: write AG3S/ag3s/docs/handoff/T43S2.verify.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, hashlib
S2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S2')); H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43S2.verify.json'))
A = json.loads((S2 / "raw/s2_analysis.json").read_text())
judge = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P2/baseline/judge.json')).read_text())["legacy"]
md5 = lambda p: hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()
eps = {}
for ep, e in A["episodes"].items():
    rep = json.loads((S2 / f"raw/replay/DEMO_ep{ep}.replay.json").read_text())
    d = e["demo_replay"]
    eps[ep] = {"arm": e["arm"], "fruit": e["fruit"], "split": e["split"], "dataset_success": e["dataset_success"], "dataset_events": e["dataset_events"],
               "n_dataset_frames": rep["n_dataset_frames"], "held_steps_after_last_frame": rep["held_steps_after_last_frame"],
               "demo_replay": {k: d[k] for k in ("run_dir", "success", "grasp", "place", "grasp_t", "place_t", "close_cmd_t", "n_close_cycles", "lift_t",
                                                 "regrasp", "robust_grasp", "robust_grasp_strict", "release_t")} | {"geom": d["geom"]},
               "demo_tracking": e["demo_tracking"], "dataset_geom": e["dataset_geom"],
               "policy_runs": [{k: r[k] for k in ("cond", "run_dir", "success", "grasp", "place", "close_cmd_t", "lift_t", "robust_grasp")} | {"geom": r["geom"]}
                               for r in e["policy_runs"]]}
cnt = {arm: {"episodes": sum(1 for e in eps.values() if e["arm"] == arm), "demo_success": sum(1 for e in eps.values() if e["arm"] == arm and e["demo_replay"]["success"]),
             "demo_robust_grasp": sum(1 for e in eps.values() if e["arm"] == arm and e["demo_replay"]["robust_grasp"]),
             "policy_runs": {c: [sum(1 for e in eps.values() if e["arm"] == arm for r in e["policy_runs"] if r["cond"] == c),
                                 sum(1 for e in eps.values() if e["arm"] == arm for r in e["policy_runs"] if r["cond"] == c and r["success"])] for c in ("E0", "E3b")}}
       for arm in ("right", "left")}
out = {"_comment": "ag3s-verifier. numbers and paths only; no interpretation.", "step": "T43S2", "date": "2026-10-05",
       "runs": [{"id": "baseline legacy (CPU, shared with T43 P2)", "command": "outputs/verify/T43/P2/baseline/run_baseline.sh; judge.py", "venv": ".venv-openpi-live (CPU)",
                 "raw_output": "outputs/verify/T43/P2/baseline/judge.json", "stdout_log": "outputs/verify/T43/P2/baseline/legacy.log"},
                {"id": "demo replay", "command": "outputs/verify/T43/S2/p/run_replays.sh N EP... -> replay_demo.py EP OUTDIR", "venv": ".venv-openpi-live (CPU, nice 19, <= 3 replay workers)",
                 "raw_output": "outputs/verify/T43/S2/raw/replay/DEMO_ep<EP>/ (frames.jsonl, manifest.json, traj.npz, third_person_front.mp4), DEMO_ep<EP>.replay.json",
                 "stdout_log": "outputs/verify/T43/S2/raw/replay/DEMO_ep<EP>.client.log"},
                {"id": "analysis", "command": "p/analyze_s2.py EP...", "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T43/S2/raw/s2_analysis.json", "stdout_log": None},
                {"id": "figures", "command": "p/render_s2.py EP...; p/fig_s2.py", "venv": ".venv-openpi-live (CPU, osmesa)", "raw_output": "outputs/verify/T43/S2/render/", "stdout_log": None}],
       "code_state": {"client": "${AG3S_PROJECT_ROOT}-t42 snapshot (pi05_TO_hybrid 98e07d6 = T42-1 rollout client), pi05_infer.py md5 " + md5("${AG3S_PROJECT_ROOT}-t42/integration/rby1/bringup/pi05_infer.py"),
                      "benchmark_head": subprocess.run(["git", "-C", _ag3s_workspace_path('legacy/benchmark'), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
                      "analysis_code": "outputs/verify/T42/p/grasp_robust.py + analyze_t42.py (T42-1 rule), outputs/verify/T43/S/p/measure_pick.geom_at (T43 S geometry)",
                      "dirty_files": []},
       "numbers": {"_comment": "mm / deg unless noted. outcome rule = T42-1 analyze_t42 (T39 rule generalized to the target fruit). settled row = first close command + 16 control rows.",
                   "definitions": {"replay": json.loads('"' + "x" + '"') and pathlib.Path(S2 / "p/replay_demo.py").read_text().split('"""')[1],
                                   "analysis": A["definitions"],
                                   "action_mapping": "dataset 'action' (16) = [left_arm_0..6 absolute joint targets, left_gripper opening 0..1, right_arm_0..6, right_gripper]; pi05_infer.apply_action('rby1_16d'): d.ctrl[arm actuators] = targets, d.ctrl[gripper] = clip(opening, 0, 1) * RBY1_GRIPPER_OPEN; one action per control step (15 Hz), sim steps per action = round(1 / (15 * timestep))"},
                   "counts": cnt, "episodes": eps,
                   "selection": {"right": ["1834", "1977", "1623", "1215", "1959", "1984"], "left": ["1925", "1828", "1830", "983", "1957", "1992"],
                                 "rule": "T42-1 1834/1623 (right) and 1925/1830 (left) as asked, plus T42-1 episodes covering pear/banana and E3b failures (1959 miss, 1984 grasp-only, 1957 no grasp, 1992 s2 HOLD)"}},
       "figures": [{"path": f"AG3S/ag3s/docs/figures/t43/{n}.png", "kind": k, "sidecar": f"AG3S/ag3s/docs/figures/t43/{n}.json"}
                   for n, k in (("t43s2-stills-right", "scene"), ("t43s2-stills-left", "scene"), ("t43s2-graph", "graph"), ("t43s2-table", "table"))],
       "baseline_check": {"matched": judge["matched"], "note": f"legacy CPU (run once for P2 and S2): violated_start {judge['violated_start']}, has_target {judge['has_target']}, frame0 {judge['frame0_clearance_before_mm']} mm; cuRobo not run"},
       "verdict": "기준선 일치 (legacy, CPU). cuRobo 기준선은 돌리지 않음.",
       "not_measured": ["policy rollouts were not re-run (T42-1 / T39 records); E0 runs do not exist for 1959, 1984, 1957, 1992",
                        "replay of the dataset with the collector's own controller (sim-rate waypoint interpolation); the replay uses the dataset's 15 Hz logged actions through the client",
                        "videos of the dataset itself were not compared frame by frame (only observation.state and final_fruit_poses)"]}
H.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False)); print(H, json.dumps(cnt))
