"""T40 I -> AG3S/ag3s/docs/handoff/T40I.verify.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, subprocess, datetime
import numpy as np
RAW = _ag3s_workspace_path('outputs/verify/T40/I/raw')
TAGS = ["C1807", "C1967", "C1982", "C1968", "D1807"]
S = json.load(open(f"{RAW}/settings.json"))
FG = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T40FG.verify.json')))["numbers"]
def r(x, n=2):
    if isinstance(x, float): return round(x, n)
    if isinstance(x, list): return [r(v, n) for v in x]
    if isinstance(x, dict): return {k: r(v, n) for k, v in x.items()}
    return x
def git(p): return subprocess.run(["git", "-C", p, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

cases = {}
for t in TAGS:
    c = json.load(open(f"{RAW}/case_{t}.json")); e = c["event"]; ln = e["closest_link"]
    cov = e["sphere_coverage_at_event_row"]
    ev = {k: e[k] for k in ("kind", "t", "seq", "definition", "closest_left_geom", "closest_link", "event_geom_source",
                            "event_geom_to_bar_mm_at_row", "per_link_min_geom_to_bar_mm_at_row", "client_contact_pairs")}
    ev["sphere_coverage_at_event_row"] = {
        L: {"mujoco_body": v["mujoco_body"], "n_spheres": v["n_spheres"], "sphere_radius_mm": v["sphere_radius_mm"],
            "link_spheres_to_bar_true_min_mm": v["sphere_to_bar_true_min_mm"],
            "geoms_within_30mm_of_bar": {g: {k: gv[k] for k in ("n_vertices", "frac_vertices_inside_a_sphere", "vertex_gap_to_own_link_spheres_mm",
                                                                 "geom_to_bar_mm", "gap_of_vertices_within_10mm_of_bar_mm", "witness_on_geom_gap_to_spheres_mm")}
                                         for g, gv in v["geoms"].items() if gv["geom_to_bar_mm"] <= 30}}
        for L, v in cov.items()}
    chunks = []
    for ch in c["chunks"]:
        ex = ch["executed"]
        with np.load(ch["npz"]) as zz:
            sm = json.loads(str(zz["summary_json"]))
        tv = ch["fidelity"].get("to_view", {}).get("abs_diff_mm")
        chunks.append({
            "grasp_state": (sm.get("grasp") or {}).get("state"), "exclusion_source": (sm.get("exclusion") or {}).get("source"),
            "to_view_reproduced_within_1mm": (tv is not None and tv <= 1.0),
            "seq": ch["seq"], "t_step": ch["t_step"], "npz": ch["npz"],
            "pairing_refined_vs_client_actions_max": ch["pairing_refined_vs_actions_max"],
            "gate_action": ch["gate"]["action"], "hold_kinds": ch["gate"].get("hold_kinds"), "trajopt_status": ch["trajopt_status"],
            "verdict_reasons": ch["verdict_reasons"], "TO_worst_row": ch["TO_worst_row"], "TO_clearance_m": ch["TO_clearance_m"],
            "TO_returned": ch["TO_returned"], "ag3s_status": ch["ag3s_status"], "has_target": ch["has_target"],
            "exclusion_active": ch["exclusion_active"], "attached_points": ch["attached_points"], "finger_opening_norm_left": ch["finger_opening_norm_left"],
            "fidelity": {"record_clearance_max_abs_diff_mm": ch["fidelity"].get("record_clearance", {}).get("max_abs_diff_mm"),
                         "sphere_centres_step0_max_diff_mm": ch["fidelity"].get("sphere_centres_step0_max_diff_mm"),
                         "to_view_abs_diff_mm": ch["fidelity"].get("to_view", {}).get("abs_diff_mm"),
                         "to_view_same_row": ch["fidelity"].get("to_view", {}).get("same_row")},
            "fine_window": ch["fine_window"], "target_free_window": ch["target_free_window"],
            "q1_bar_in_field": ch["bar_in_field"],
            "q2_per_link": ch["per_link"],
            "q3_executed": ex,
            "q4_policy_pull": ch["policy_pull"],
        })
    cases[t] = {"run_dir": c["run_dir"], "hurdle": c["hurdle"], "bar_geom": c["bar_geom"], "event": ev,
                "window_seqs": c["window_seqs"], "chunks": chunks, "raw": f"{RAW}/case_{t}.json"}

out = {
    "_comment": "ag3s-verifier. numbers and paths only; no interpretation, no cause verdict.",
    "step": "T40I", "date": datetime.date.today().isoformat(),
    "runs": [{"id": "T40 I measurement (records only)",
              "command": "cd ${AG3S_PROJECT_ROOT}-t40 && CUDA_VISIBLE_DEVICES= PYTHONPATH=${AG3S_PROJECT_ROOT}-t40 nice -n 19 ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python ${AG3S_PROJECT_ROOT}/outputs/verify/T40/I/p/measure_i.py",
              "venv": ".venv-openpi-live (CPU only, no server, no GPU)",
              "raw_output": "outputs/verify/T40/I/raw/{case_*.json, section_*.npy, settings.json, chunk_index.json}",
              "stdout_log": "outputs/verify/T40/I/raw/measure_i.log"},
             {"id": "chunk index", "command": "outputs/verify/T40/I/p/scan_chunks.py", "venv": ".venv-openpi-live",
              "raw_output": "outputs/verify/T40/I/raw/chunk_index.json", "stdout_log": None},
             {"id": "figures", "command": "cd ${AG3S_PROJECT_ROOT} && PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-openpi-live/bin/python outputs/verify/T40/I/p/fig_i.py",
              "venv": ".venv-openpi-live", "raw_output": "AG3S/ag3s/docs/figures/t40/t40i-*.json", "stdout_log": None}],
    "code_state": {"snapshot": "${AG3S_PROJECT_ROOT}-t40", "snapshot_benchmark_head": git("${AG3S_PROJECT_ROOT}-t40/benchmark") or "150c28f (T40FG record)",
                   "snapshot_pi05_TO_hybrid_head": git("${AG3S_PROJECT_ROOT}-t40/pi05_TO_hybrid") or "98e07d6 (T40FG record)",
                   "benchmark_head": git(_ag3s_workspace_path('legacy/benchmark')), "dirty_files": [],
                   "note": "measurement scripts live in outputs/verify/T40/I/p (not in benchmark/)"},
    "numbers": {
        "_comment": "every number of T40 I. mm unless the key says m/rad/deg. 'TO view' = the trajopt clearance d - r - esdf_margin(0) with the target-free layer for authorized links when present (fixed_input_ablation.row_details); 'true mesh' = MuJoCo collision geom of the hand (mj_geomDistance) to the hurdle bar capsule.",
        "definitions": {
            "cases": "contact = E3b runs with a hand-to-bar contact during the approach (T40FG C_E3b_failure_classes); pass = successful runs, at the carry-phase minimum hand-to-hurdle row",
            "event_row": "contact: last control row (client obstacle stop row; MuJoCo contact reported in the 33 sim steps after it). pass: argmin of gripper (EE_BODY_L, fingers) to hurdle over the carry phase (analysis_all)",
            "window": "event chunk and the 4 planning chunks before it (5 chunks)",
            "pairing": "server npz t_step == client planning seq; refined_chunk == planning 'actions' (max abs diff reported, 0.0 in all)",
            "q1_bar_in_field": "per tier (coarse 20 mm, fine 5 mm, target-free 5 mm layers as recorded; TO_plain_composite = min(coarse, fine) as CuroboEsdfField.distance; TO_target_free_composite = target_free_distance): field at 49 points on the true bar axis (true value -12 mm), at 49x16 points on the true bar surface (true 0 mm), exit radius of the field <= 0 region along 16 rays per axis point (true 12 mm), value at the axis point nearest the event contact point. Bar from the run's obstacle config placed with the snapshot client code, world -> robot base with MuJoCo body 'base' at the chunk's planning qpos",
            "q2": "URDF query spheres (constraint model rebuilt with the server's sphere options; finger path as SafePolicy: FingerJointMap.path(measured opening, reference gripper columns)); coverage at the event row with MuJoCo finger joints; vertex gap = min over spheres of |v - c| - r (<= 0 means the mesh vertex is inside a sphere)",
            "q3": "applied_ctrl arm targets of the chunk's control rows compared with refined / reference / q_hold; measured row k (pose when row k is commanded) vs plan target k-1 (row 0 vs q_now); refined/reference plans through the true MuJoCo mesh with the finger joints of the planning row",
            "q4": "link centroid = mean of the event link's query spheres; bar inward normal at q_now from the centroid to the closest bar-axis point; displacement over the 8-step plan window; + = toward the bar"},
        "body_link_mapping": S["body_link_mapping"],
        "settings_rebuilt": {k: S["settings"][k] for k in ("links", "sphere_options", "esdf_margin", "target_field_policy", "phase", "manipulators", "plan_horizon", "rows_per_step")},
        "n_spheres": S["n_spheres"],
        "server_log_sphere_coverage_warning": "serve_safe.log: '18 of 18 capsule(s) are thinner than the URDF says, worst by 27.3 mm'; effective radius ee_finger_* 19x3.25 mm, ee_left 23x6.22 mm",
        "other_E3b_approach_contacts_in_T40FG": {
            "contact_runs": ["C_E3b_ep1807", "C_E3b_ep1967", "C_E3b_ep1982"],
            "no_contact_runs_with_obstacle_stop": {k: {"client_robot_collision": FG["outcome_table"][k.split("_")[0][0]]["E3b_" + k.split("_")[-1]]["client_robot_collision"],
                                                        "client_min_robot_clearance_m": FG["outcome_table"][k.split("_")[0][0]]["E3b_" + k.split("_")[-1]]["client_min_robot_clearance_m"]}
                                                    for k in ("C_E3b_ep1995", "D_E3b_ep1967", "D_E3b_ep1982", "D_E3b_ep1995")},
            "successful_runs": ["C_E3b_ep1968", "D_E3b_ep1807"]},
        "cases": r(cases, 3),
    },
    "figures": [{"path": "AG3S/ag3s/docs/figures/t40/t40i-scene.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/t40/t40i-scene.json"},
                {"path": "AG3S/ag3s/docs/figures/t40/t40i-clearance.png", "kind": "graph", "sidecar": "AG3S/ag3s/docs/figures/t40/t40i-clearance.json"},
                {"path": "AG3S/ag3s/docs/figures/t40/t40i-table.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/t40/t40i-table.json"}],
    "baseline_check": {"matched": None, "note": "not run: task forbids GPU/servers (records only). Reconstruction fidelity instead: recorded clearance (8x122) re-evaluated within 0.0001 mm in 25/25 chunks; TO worst row (with contact policy) within 0.0001 mm in 20/25; not reproduced (4.8-17.7 mm) in 5 chunks with grasp state held and exclusion source 'attached' (C1968 seq 17,18,19,21; D1807 seq 21): held-object query points and the held-state contact policy are not in the record. q2_per_link TO_view values of those 5 chunks are the reconstruction, not the TO's own view (flag to_view_reproduced_within_1mm)"},
    "verdict": "facts only; baseline not run (records-only task); recorded clearance reproduced in 25/25 chunks, TO worst row in 20/25",
    "not_measured": ["exact sim-substep pose at contact (control rows are recorded once per 33 sim steps)",
                     "held-apple (attached) query points and held-state contact policy (not recorded) -> TO view not reproduced in C1968 seq 17/18/19/21, D1807 seq 21",
                     "server label layer (obstacle names of worst rows)"],
}
json.dump(out, open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T40I.verify.json'), "w"), indent=1, ensure_ascii=False)
print("ok")
