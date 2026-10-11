"""Assemble handoff/T28.verify.partial.json from raw outputs (numbers + paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, datetime
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T28.verify.partial.json'))


def git(repo):
    h = subprocess.run(["git", "-C", repo, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", repo, "status", "--short"], capture_output=True, text=True).stdout.split("\n")
    return {"head": h, "dirty_files": [x for x in dirty if x.strip()]}


doc = json.loads(OUT.read_text()) if OUT.exists() else {}
doc.setdefault("_comment", "ag3s-verifier. 숫자와 경로만. 해석 없음. Phase 1 only.")
doc["step"] = "T28"; doc["date"] = "2026-09-28"; doc["phase"] = "1"
doc.setdefault("runs", []); doc.setdefault("numbers", {}); doc.setdefault("figures", []); doc.setdefault("code_state", {})
doc["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")

def add_run(r):
    doc["runs"] = [x for x in doc["runs"] if x["id"] != r["id"]] + [r]

# ---- P1-1
e0 = V / "E0/e0_summary.json"
if e0.exists():
    S = json.loads(e0.read_text())
    add_run({"id": "P1-1_E0_server", "command": "outputs/verify/T28/E0/start_server.sh (serve_safe --no-safe --port 8201)",
             "venv": ".venv-openpi-live (mujoco 3.11.0)", "stdout_log": "outputs/verify/T28/E0/serve_safe.log"})
    add_run({"id": "P1-1_E0_clients", "command": "outputs/verify/T28/E0/run_client.sh EP K (pi05_infer.py --remote localhost:8201 --max-steps 600 --headless --record-frames, no --safe-remote)",
             "raw_output": "outputs/verify/T28/E0/E0_ep<EP>_r<K>/frames.jsonl", "stdout_log": "outputs/verify/T28/E0/E0_ep<EP>_r<K>.client.log",
             "analysis": "outputs/verify/T28/E0/analyze_e0.py, e0_detail.py -> outputs/verify/T28/E0/e0_summary.json"})
    runs = {}
    for k, r in S["runs"].items():
        dt = r.get("detail", {})
        runs[k] = {"episode": r["episode_index"], "success": r["success"], "grasp": r["grasp"], "place": r["place"],
                   "grasp_t": r["grasp_t"], "grasp_t_control_rows": r["grasp_t_control_rows"], "place_t": r["place_t"],
                   "apple_dz_max_mm": round(r["apple_dz_max_mm"], 2), "apple_dz_final_mm": round(r["apple_dz_final_mm"], 2),
                   "apple_final_in_crate_frame_m": [round(x, 4) for x in r["apple_final_in_crate_frame"]],
                   "infer_ms": {k2: (round(v, 1) if isinstance(v, float) else v) for k2, v in r["infer_ms"].items()},
                   "n_control": r["n_control"], "n_planning": r["n_planning"],
                   "n_not_executed_control": r["n_not_executed_control"], "n_timeout": 0, "n_stale": 0,
                   "ipc_counts": "in-process 75/75 (non-safe client path)",
                   "t_first_applied_grip_l_lt_0p5": dt.get("t_first_applied_grip_l_lt_0p5"),
                   "finger_to_apple_mm_at_first_close": dt.get("finger_to_apple_mm_at_first_close"),
                   "t_first_finger_apple_contact_le_0mm": dt.get("t_first_finger_apple_contact_le_0mm"),
                   "min_grip_l_measured_open_frac": dt.get("min_grip_l_measured"),
                   "n_close_cycles_applied_lt_0p5": dt.get("n_close_open_cycles_applied_lt_0p5")}
    det = S.get("determinism_ep1807", {})
    doc["numbers"]["P1-1_E0"] = {
        "definitions": {
            "grasp": "planning-row object_poses apple z >= initial (first planning row) + 50 mm; grasp_t = first such planning t_step (grasp_t_control_rows = same test on per-step control qpos[40])",
            "place": "last planning row: apple centre in crate-local frame |x|<0.077, |y|<0.137, -0.052<z<0.075 m (rby1_manipulation/evaluation/transport.py:object_in_crate, margin 5 mm; CRATE_INTERIOR_HALF=[0.082,0.142,0.056] transport_scene.py:82 = crate wall geoms scene_transport.xml:122-131); place_t = first t of final uninterrupted in-crate streak",
            "success": "grasp and place",
            "infer_ms": "planning row timing_ms.policy_infer (client round trip); first = warm-up/JIT, not quoted",
            "finger_to_apple": "min mj_geomDistance between ee_finger_l1/l2 collision geoms and apple_geom (MuJoCo 3.11 collision meshes), witness-checked",
            "grip_l": "applied_ctrl.gripper[0] (1=open, ctrl/-0.045); measured = |qpos[gripper_finger_l1]|/0.045"},
        "server": {"port": 8201, "flags": "--no-safe", "mujoco": "3.11.0", "started": "2026-09-28T17:36", "killed": "2026-09-28T17:44"},
        "episode_selection": {
            "rule": "randomized_episodes.jsonl (load_randomized_episode meta), split test 1800-1999, target_fruit=apple, used_arm=left -> 24 episodes (all place_one, no preload, dataset success, recovery none, 1 grasp attempt). Picked the two lowest episode_index other than 1807.",
            "candidates": [1800, 1807, 1808, 1818, 1819, 1824, 1828, 1836, 1863, 1878, 1887, 1888, 1896, 1914, 1925, 1927, 1939, 1965, 1967, 1968, 1976, 1982, 1983, 1995],
            "picked": [1800, 1808],
            "apple_initial_xy_m": {"1807": [0.564, 0.320], "1800": [0.452, 0.324], "1808": [0.540, 0.322]},
            "layout_index": {"1807": 15, "1800": 8, "1808": 0}},
        "determinism_ep1807_r1_vs_r2": {
            "step0_applied_ctrl_max_abs_diff": det.get("step0_applied_ctrl_diff"),
            "first_t_with_nonzero_applied_ctrl_diff": det.get("first_t_with_nonzero_applied_ctrl_diff"),
            "max_abs_applied_ctrl_diff": det.get("max_abs_applied_ctrl_diff"),
            "max_abs_qpos_diff": det.get("max_abs_qpos_diff"),
            "deterministic": False, "consequence_per_protocol": "N=2 per episode"},
        "runs": runs,
        "success_grid": {ep: [runs[f"E0_ep{ep}_r{k}"]["success"] for k in (1, 2)] for ep in (1807, 1800, 1808)},
        "n_success": sum(r["success"] for r in runs.values()), "n_runs": len(runs),
    }

# ---- P1-2
p12 = V / "P1-2/p12_records.json"
if p12.exists():
    R = json.loads(p12.read_text())
    add_run({"id": "P1-2_records", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-openpi-live/bin/python outputs/verify/T28/P1-2/p12_records.py",
             "venv": ".venv-openpi-live (mujoco 3.11.0)", "raw_output": "outputs/verify/T28/P1-2/p12_records.json",
             "stdout_log": "outputs/verify/T28/P1-2/p12_records.log",
             "inputs": ["outputs/live_test/20260928_t16 (server_constraints/run_0000 + execute_ep1807)", "outputs/live_test/20260928_t14 (same)"]})
    out = {"definitions": R["definitions"], "n_invalid_geomdistance_pairs_skipped": R.get("n_invalid_geomdistance_pairs_skipped")}
    for name, rr in R["runs"].items():
        s = dict(rr["summary"]); s.pop("argv", None)
        rows = []
        for e in rr["chunks"]:
            if not e.get("has_local_row"):
                rows.append({"chunk": e["chunk"], "t": e["t_step"], "local_row": False}); continue
            wr = e["worst_row"]
            rows.append({"chunk": e["chunk"], "t": e["t_step"], "executed": e["executed"], "hold_kind": e["hold_kind"],
                         "hold_reason": e["hold_reason"], "ipc": e["ipc"],
                         "server_infer_ms": round(e["server_timing_ms"]["infer"], 1), "server_total_ms": round(e["server_timing_ms"]["total"], 1),
                         "client_policy_infer_ms": round(e["client_policy_infer_ms"], 1),
                         "worst_link": wr.get("link"), "worst_clearance_mm": round(1000 * wr["clearance_m"], 2) if wr.get("clearance_m") is not None else None,
                         "worst_step": wr.get("step"), "worst_query": wr.get("query"),
                         "worst_nearest_object": e.get("worst_nearest_body"), "worst_nearest_category": e.get("worst_nearest_category"),
                         "worst_point_to_nearest_surface_mm": round(e["worst_nearest_surface_mm"], 1) if e.get("worst_nearest_surface_mm") is not None else None,
                         "worst_point_to_apple_centre_mm": round(e["worst_point_to_apple_centre_mm"], 1) if e.get("worst_point_to_apple_centre_mm") is not None else None,
                         "target_is_apple": e.get("target_is_apple"), "target_nearest_body_centre": e.get("target_nearest_body_centre"),
                         "target_to_apple_centre_mm": round(e["target_to_apple_centre_mm"], 1) if e.get("target_to_apple_centre_mm") is not None else None,
                         "target_n_points": e.get("target_n_points"), "grounding_best_score": e.get("grounding_best_score"),
                         "finger_to_apple_surface_mm": round(e["finger_to_apple_surface_mm"], 1), "apple_dz_mm": round(e["apple_dz_mm"], 2),
                         "policy_grip_l_cmd_row0": round(e["policy_grip_l_cmd_rows0_7"][0], 3), "policy_grip_l_cmd_min_rows0_7": round(e["policy_grip_l_cmd_min_rows0_7"], 3),
                         "grip_l_measured_min": round(min(e["grip_l_measured_open_frac"]), 3),
                         "refined_minus_reference_exec_max_deg": round(e["refined_minus_reference_exec_max_deg"], 3)})
        out[name] = {"summary": s, "chunks": rows}
    doc["numbers"]["P1-2"] = out

# ---- P1-3
for key, path in (("P1-3a", V / "P1-3/p13a_summary.json"), ("P1-3b", V / "P1-3/p13b_selffilter_summary.json"), ("P1-3c", V / "P1-3/p13c_opening.json")):
    if path.exists():
        doc["numbers"][key] = json.loads(path.read_text())

doc["code_state"] = {"benchmark": git(_ag3s_workspace_path('legacy/benchmark')), "pi05_TO_hybrid": git(_ag3s_workspace_path('legacy/pi05_TO_hybrid')),
                     "note": "Phase 1 uses only --no-safe server / non-safe client / offline tools; T26/T27 edits may appear as dirty files"}
doc["figures"] = [
    {"path": "AG3S/ag3s/docs/figures/t28/t28-p1-scene.png", "kind": "scene", "sidecar": "t28-p1-scene.json", "script": "outputs/verify/T28/fig_scene.py"},
    {"path": "AG3S/ag3s/docs/figures/t28/t28-e0-timeline.png", "kind": "graph", "sidecar": "t28-e0-timeline.json", "script": "outputs/verify/T28/fig_graphs.py"},
    {"path": "AG3S/ag3s/docs/figures/t28/t28-p12-chunk-diagnosis.png", "kind": "graph", "sidecar": "t28-p12-chunk-diagnosis.json", "script": "outputs/verify/T28/fig_graphs.py"},
    {"path": "AG3S/ag3s/docs/figures/t28/t28-p13b-selffilter.png", "kind": "graph", "sidecar": "t28-p13b-selffilter.json", "script": "outputs/verify/T28/fig_graphs.py"},
    {"path": "AG3S/ag3s/docs/figures/t28/t28-p1-tables.png", "kind": "table", "sidecar": "t28-p1-tables.json", "script": "outputs/verify/T28/fig_graphs.py"}]
add_run({"id": "P1-3", "command": "MUJOCO_GL=osmesa PYTHONPATH=<scratchpad>/head_ws .venv-ag3s/bin/python outputs/verify/T28/P1-3/{p13a_spheres,p13a_record_check,p13a_summary,p13b_selffilter}.py; .venv-openpi-live/bin/python outputs/verify/T28/P1-3/p13c_opening.py; python3 p13b_summary.py",
         "code": "benchmark HEAD cee8f84 via git archive into scratchpad head_ws (isolated from T26/T27 working-tree edits)",
         "raw_output": ["outputs/verify/T28/P1-3/p13a_spheres_raw.json", "p13a_record_check.json", "p13a_summary.json", "p13b_raw.jsonl", "p13b_selffilter_summary.json", "p13c_opening.json"],
         "stdout_log": ["outputs/verify/T28/P1-3/p13a_spheres.log", "p13a_record_check.log", "p13b_selffilter.log", "p13b_summary.log", "p13c_opening.log"]})
doc["not_measured"] = ["regression baseline (task: not in Phase 1)", "E1-E3b (Phase 2/3)",
                       "P1-1: only N=2 per episode (protocol); per-step ctrl of the policy chunk (non-safe client rows carry no `actions`)",
                       "P1-2: T16/T14 record the worst row only for the executed/refined chunk at step level; nearest object uses GT meshes (convex hull), 1 invalid geomDistance pair skipped",
                       "P1-3a: live E3 flags not run; sphere comparison at one arm pose (E0 ep1807 t=0) + 5 recorded chunks per run"]
doc["code_state"]["at_E0_runs"] = {
    "pi05_TO_hybrid": "dc5e660, clean (git status empty at 17:35 check)", "benchmark": "cee8f84, only handoff docs dirty at 17:35 check",
    "file_mtimes_fs_clock": {"pi05_infer.py": "2026-09-28 17:46:55", "serve_safe.py": "2026-09-28 17:46:01",
                             "last_E0_output(E0_ep1808_r2.client.log)": "2026-09-28 17:43:14"},
    "note": "both files modified (T27) after the last E0 client finished and the server was killed; no E0 crash/retry occurred"}
doc["verdict"] = "사실만. Phase 1 raw numbers; baseline not run in this phase (task)."
OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
print("wrote", OUT, "sections:", list(doc["numbers"].keys()))
