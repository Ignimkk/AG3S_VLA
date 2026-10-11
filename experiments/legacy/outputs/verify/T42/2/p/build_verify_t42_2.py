"""T42-2: write AG3S/ag3s/docs/handoff/T42-2.verify.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/2')); H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
J = lambda p: json.loads(pathlib.Path(p).read_text())
A = J(T / "analysis_place1.json"); SEL = J(T / "raw/select.json"); bl = J(T / "baseline/judge.json")
placements = {}
for ep, r in SEL["episodes"].items():
    ch = r["place"]["chosen"]; d = list(r["dryrun"].values())[0]
    s = J(T / f"raw/search_{ep}_{ch['slot']}.json")
    placements[ep] = {"seed": r["seed"], "slot": ch["slot"], "assigned_slot": r["place"]["assigned"], "fallback": r["place"]["fallback"],
                      "entry": {k: ch[k] for k in ("slot", "position", "yaw", "height", "radius", "bar_height", "length") if k in ch},
                      "source": ch["path_run"], "grid_u_v": [ch.get("grid_u"), ch.get("grid_v")], "tier": ch["tier"], "margin_m": ch["margin_m"],
                      "room_m": ch.get("room_m"), "reset_robot_m": ch["reset_robot"], "reset_object_m": ch["reset_object"], "mins_recorded_m": ch["mins"],
                      "search_counts": s["counts"], "n_feasible": s["n_feasible"], "tried": r["place"]["tried"],
                      "dryrun_client": {k: d.get(k) for k in ("outcome", "robot_clearance_m", "object_clearance_m", "active_slots")},
                      "config": r["config"]}
runs = {}
for k, r in A["runs"].items():
    o = r["obst"]; cs = o.get("client_obstacle_safety") or {}
    runs[k] = {"cond": r["cond"], "episode": r["episode"], "seed": r["seed"], "target_fruit": r["target_fruit"], "used_arm": r["used_arm"],
               "slots": o["slots"], "success": r["success"], "grasp": r["grasp"], "place": r["place"], "grasp_t": r["grasp_t"], "place_t": r["place_t"],
               "robust_grasp_B": r["robust"]["robust_grasp"], "failure_mode_B": None, "n_control": r["n_control"],
               "holds": r["holds"], "gate_action_counts": r["gate_action_counts"],
               "obstacle_stop": o["obstacle_stop"], "stop_t": o["stop_t"], "stop_phase": o["stop_phase"], "lift_t": o["lift_t"],
               "client_contact": {k2: cs.get(k2) for k2 in ("robot_collision", "payload_collision", "contact_steps", "min_robot_clearance_m", "contact_pairs")},
               "min_distance_m_by_phase": {ph: {g: v["min_m"] for g, v in d.items() if v} for ph, d in o["phases"].items()},
               "min_distance_pair_all": {g: v.get("pair") for g, v in o["phases"]["all"].items() if v},
               "ag3s": None if r["cond"] == "E0" else {k2: o.get(k2) for k2 in ("ag3s_n_chunks", "ag3s_n_target_nearest_obstacle", "ag3s_n_manipulated_nearest_obstacle",
                                                                                 "ag3s_n_chunks_with_target_points_on_obstacle", "ag3s_min_target_to_obstacle_m")},
               "n_server_segments_matching": r.get("n_server_segments_matching"),
               "divergence_from_no_obstacle_run": r["divergence"], "run_dir": o["run_dir"], "config": o["config"], "profile": o["profile"]}
    import sys; sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p')); import grasp_robust as GR
    runs[k]["failure_mode_B"] = GR.failure_mode(r["robust"])
counts = {c: {"n": sum(1 for v in runs.values() if v["cond"] == c), "success": sum(v["success"] for v in runs.values() if v["cond"] == c),
              "grasp": sum(v["grasp"] for v in runs.values() if v["cond"] == c), "place": sum(v["place"] for v in runs.values() if v["cond"] == c),
              "obstacle_stop": sum(v["obstacle_stop"] for v in runs.values() if v["cond"] == c),
              "stop_phase": {p: sum(1 for v in runs.values() if v["cond"] == c and v["stop_phase"] == p) for p in ("before_lift", "carry", "after")},
              "diverge_at_step_le_1": sum(1 for v in runs.values() if v["cond"] == c and (v["divergence_from_no_obstacle_run"]["first_step_diff_gt_1e-3"] or 0) <= 1)}
          for c in ("E0", "E3b")}
fig = lambda n, k: {"path": f"AG3S/ag3s/docs/figures/t42/{n}.png", "kind": k, "sidecar": f"AG3S/ag3s/docs/figures/t42/{n}.json"}
out = {
 "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
 "step": "T42-2", "date": "2026-10-05",
 "runs": [
  {"id": "regression baseline on the snapshot", "command": "outputs/verify/T42/2/baseline/run_baseline.sh; judge.py", "venv": ".venv-openpi-live",
   "raw_output": "outputs/verify/T42/2/baseline/", "stdout_log": "outputs/verify/T42/2/baseline/{legacy,curobo}.log"},
  {"id": "placement search (place phase)", "command": "p/capture.py; p/run_searches.sh -> p/search.py (plib.py); p/select.py (place only) -> p/dryrun.py",
   "venv": ".venv-openpi-live (CPU, nice 19, <= 4 workers)", "raw_output": "outputs/verify/T42/2/raw/{search_*,select,dryrun_*}.json, configs/t42_2_ep*.json",
   "stdout_log": "outputs/verify/T42/2/raw/logs/"},
  {"id": "place-phase obstacle runs E0 / E3b", "command": "p/launch_t42_2.sh place1 (E3b 8251, E0 8250) -> run_chain.sh / start_server.sh / run_client.sh / check_run.py",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T42/2/runs/{E0,E3b}/place/", "stdout_log": "outputs/verify/T42/2/runs/launch_place1.out, runs/{E0,E3b}/chain_825*.log"},
  {"id": "analysis", "command": "p/analyze_t42_2.py (obst_metrics.py, analyze_t42.py, grasp_robust.py); p/divergence.py; p/fig_t42_2.py; p/build_verify_t42_2.py",
   "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T42/2/analysis_place1.json"}],
 "code_state": {"snapshot": "${AG3S_PROJECT_ROOT}-t42c", "benchmark_head": "61f10aa (git archive + ag3s/asset/data symlink)", "pi05_TO_hybrid_head": "fce4af8 (git archive rby1_bringup, rby1_description/models, rby1_manipulation)",
                "code_md5": "outputs/verify/T42/2/p/code_md5_t42c.txt (402 files)", "import_check": ["outputs/verify/T42/2/import_server.json", "outputs/verify/T42/2/import_client.json"],
                "superseded_snapshot": "${AG3S_PROJECT_ROOT}-t42b (c7c1cc9, before the arrow commit; no runs)", "dirty_files": []},
 "numbers": {
  "_comment": "seed = 10*ep + 1 (rep 1). Distances: snapshot manager.pair_distance on recorded control rows. Groups: GRIP = EE_BODY_<arm> + fingers, TARGET = target fruit, ROBOT_OTHER = other robot geoms, FRUITS = other fruits. Stop phases relative to lift_t / place_t of the run.",
  "settings": {"E3b_server": "T34 settings + --no-subtask-gate (as T42-1)", "E0_server": "--no-safe --no-subtask-gate",
               "client": "as T42-1 + --obstacle-config CFG --obstacle-profile PROF --obstacle-stop-distance 0; E3b + --viz-avoidance-arrow (defaults 5 mm, x8)",
               "extra_arm_obstacle_margin_hold_measured": "not run"},
  "regression_baseline": bl,
  "placement_criteria": {"a": "client reset check robot > 0 and object > 0", "a_prime": "every robot geom >= 0.030 m at reset, every 50 of 1000 warmup steps (replay, obstacle parked), and recorded t0 rows of E0 and E3b",
                         "b": "t0 rows: robot, target, other fruits, crate > 0", "d_f": "whole recorded pick window (0 <= t < lift_t) of E0 and E3b: robot, target, other fruits >= 0.030 m",
                         "c": "recorded carry window (lift_t <= t < place_t): min(robot, target) <= 0 in every run with a carry window",
                         "candidates": "crate-frame grid (target side) + recorded target carry path (search.py docstring)", "selection": "select.py tiers (margin, room)"},
  "placements": placements, "slots_used": SEL["slots_used"],
  "counts": counts, "runs": runs},
 "figures": [fig(f"t42-2-scene-{ep}-place", "scene") for ep in placements] + [fig("t42-2-distance-place", "graph"), fig("t42-2-table-place", "table")],
 "baseline_check": {"matched": bool(bl["legacy"]["matched"] and bl["curobo"]["matched"]),
                    "note": f'legacy {bl["legacy"]["violated_start"]} / {bl["legacy"]["has_target"]} / {bl["legacy"]["frame0_clearance_before_mm"]}; cuRobo {bl["curobo"]["violated_start"]} / {bl["curobo"]["has_target"]} / {bl["curobo"]["frame0_clearance_before_mm"]}; snapshot ${AG3S_PROJECT_ROOT}-t42c'},
 "verdict": "baseline matched; 12/12 place-phase runs complete (check_run); executed actions differ from the no-obstacle run of the same seed from control step 0-1 in 12/12",
 "not_measured": ["pick-phase obstacles (post_0, fence_0, overbar_0) runs", "combined pick + place runs", "seeds rep 2-3 with obstacles",
                  "E3b with --obstacle-margin 0.010 --safe-hold-mode measured", "source of the step-0 divergence (camera image vs model XML vs physics) not separated"],
}
(H / "T42-2.verify.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(counts, indent=1))
for k, v in runs.items():
    print(k, v["slots"], "S" if v["success"] else "-", v["failure_mode_B"], v["stop_phase"], v["stop_t"], v["holds"]["n_hold"], v["divergence_from_no_obstacle_run"]["first_step_diff_gt_1e-3"])
