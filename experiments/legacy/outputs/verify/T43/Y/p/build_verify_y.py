"""T43 Y — assemble AG3S/ag3s/docs/handoff/T43Y.verify.json from raw/*.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess
from collections import Counter
Y = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/Y')); OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43Y.verify.json'))
R = json.load(open(Y / "raw/y_metrics.json"))["runs"]; S = json.load(open(Y / "raw/server_chunks.json"))["runs"]
E = json.load(open(Y / "raw/esdf_held.json"))["runs"]; H = json.load(open(Y / "raw/hold_rows.json"))["runs"]
D = json.load(open(Y / "raw/divergence.json"))["pairs"]; ST = json.load(open(Y / "raw/y2_stats.json"))
A = json.load(open(_ag3s_workspace_path('outputs/verify/T43/W/analysis/analysis.json')))["runs"]
rp = {p.stem: json.load(open(p)) for p in (Y / "replay").glob("*.json")}
TL = ("grasp_t", "lift_t", "attach_t", "over_crate_t", "in_crate_t", "place_t", "hold_intervals", "n_hold_chunks", "trailing_hold_chunks", "policy_open_t", "applied_open_t",
      "measured_open_t", "last_apple_hand_contact_t", "release_t", "release_mode", "release_with_cmd", "release_open_norm", "release_in_hold", "apple_left_hand_t",
      "first_apple-crate_contact_t", "n_apple-crate_contact_rows", "first_hand-crate_contact_t", "n_hand-crate_contact_rows", "first_robot_other-crate_contact_t",
      "n_robot_other-crate_contact_rows", "server_detach_t", "server_states_after_attach", "apple_in_crate_end_mm", "finger_open_end", "apple_finger_mid_end_mm")
Y2 = ("release_apple_floor_gap_mm", "release_apple_centre_above_floor_mm", "release_v_horiz_mm_s", "release_v_vert_mm_s", "post_release_max_speed_mm_s",
      "post_release_horiz_travel_mm", "post_release_drop_mm", "in_hand_slip_max_mm", "in_hand_slip_at_release_mm", "pre_release_apple_crate_substeps",
      "pre_release_hand_crate_substeps", "apple_obstacle_substeps_10rows_before_release", "tcp_speed_at_open_mm_s", "tcp_speed_4rows_before_open_mm_s",
      "tcp_height_above_floor_at_open_mm", "apple_floor_gap_at_open_mm", "open_minus_release_rows", "place_window_deflection_mm", "release_mode")
focus = ["V4/H/V4_ep1925_s19253", "V5/H/V5_ep1925_s19253", "V5/H/V5_ep1807_s18071", "V5/H/V5_ep1807_s18073", "V5/H/V5_ep1834_s18343", "V5/B/V5_ep1807_s18071", "V4/H/V4_ep1834_s18341"]


def hold_summary(k):
    h = H.get(k, []); rel = R[k]["release_t"]
    g = {}
    for x in h:
        key = f"{x['kind']}|{x['link'].split('[')[0] if x['kind'] == 'held' else x['link']}|{x['obstacle']}|{'before' if rel is None or x['t'] < rel else 'after'}_release"
        g.setdefault(key, []).append(x)
    mm = lambda v, f: [round(min(y[f] for y in v), 2), round(max(y[f] for y in v), 2)]
    return {key: {"n_chunks": len(v), "to_clearance_mm": mm(v, "to_clearance_mm"), "esdf_minus_r_mm": mm(v, "esdf_gap_mm"),
                  "implied_margin_mm": sorted({round(y["implied_margin_mm"], 1) for y in v}), "true_crate_sd_minus_r_mm": mm(v, "crate_sd_mm"),
                  "apple_surface_mm_approx": mm(v, "apple_surface_mm"), "table_mm": mm(v, "table_mm"), "radius_mm": sorted({round(y["radius_mm"], 2) for y in v})} for key, v in g.items()}


def attach_radii(k):
    a = next((c for c in S[k]["chunks"] if c.get("attach")), None); return ((a or {}).get("held_query") or {}).get("radii_mm")


pairs = [(k, k.replace("V4", "V5")) for k in sorted(R) if k.startswith("V4/") and k.replace("V4", "V5") in R and R[k]["success"] and R[k.replace("V4", "V5")]["success"]]
div_rows = {}
for p, r in D.items():
    st, nm = p.split("/"); k5 = f"V5/{st}/V5_{nm}"; k4 = f"V4/{st}/V4_{nm}"
    kept = [c["t"] for c in A[k5].get("server_chunks", []) if c.get("decision_reason") == "closing_kept_geometry"]
    div_rows[p] = {"first_step": r.get("first_step"), "chunk_t": r.get("chunk_t"), "reference_rows0_7_max_abs_diff_rad": r.get("reference_rows0_7_max_abs_diff_rad"),
                   "refined_rows0_7_max_abs_diff_rad": r.get("refined_rows0_7_max_abs_diff_rad"), "V4_grasp_state": (r.get("V4") or {}).get("grasp_state"),
                   "V5_decision_reason": (r.get("V5") or {}).get("decision_reason"), "carve_ball_radius_mm_V4_V5": [(r.get("V4") or {}).get("carve_ball_radius_mm"), (r.get("V5") or {}).get("carve_ball_radius_mm")],
                   "V5_first_closing_kept_geometry_t": kept[0] if kept else None, "attach_held_query_radii_mm_V4": attach_radii(k4), "attach_held_query_radii_mm_V5": attach_radii(k5),
                   "V5_latched_hold_true_chunks": sum(1 for c in A[k5].get("server_chunks", []) if c.get("latched_hold")),
                   "V5_not_supported_chunks_at_or_before_div": sum(1 for c in A[k5].get("server_chunks", []) if c.get("n_not_supported", 0) > 0 and r.get("chunk_t") is not None and c["t"] <= r["chunk_t"]),
                   "V5_not_supported_chunks_total": sum(1 for c in A[k5].get("server_chunks", []) if c.get("n_not_supported", 0) > 0),
                   "success_V4_V5": [R[k4]["success"], R[k5]["success"]]}
dv = [v for p, v in div_rows.items() if v["first_step"] not in (None, 0)]
num = {
 "replay": {"n_runs": len(rp), "qpos_mismatch_rows_total": sum(v["qpos_mismatch_rows"] for v in rp.values()), "sub_steps_per_row": 33},
 "Y1": {
  "timeline": {k: {f: R[k].get(f) for f in TL} for k in focus},
  "hold_binding_rows": {k: hold_summary(k) for k in focus if H.get(k)},
  "attach_held_query_radii_mm": {k: attach_radii(k) for k in focus},
  "V5_1925_s19253": {
    "hold_chunks": R["V5/H/V5_ep1925_s19253"]["n_hold_chunks"],
    "hold_chunks_worst_row_held_sphere0_vs_destination": sum(1 for x in E["V5/H/V5_ep1925_s19253"] if x["hold"] and x["sphere"] == 0 and x["obstacle"] == "destination"),
    "hold_to_clearance_mm_min_max": [min(x["to_clearance_mm"] for x in E["V5/H/V5_ep1925_s19253"] if x["hold"]), max(x["to_clearance_mm"] for x in E["V5/H/V5_ep1925_s19253"] if x["hold"])],
    "hold_esdf_minus_r_mm_min_max": [min(x["esdf_gap_mm"] for x in E["V5/H/V5_ep1925_s19253"] if x["hold"]), max(x["esdf_gap_mm"] for x in E["V5/H/V5_ep1925_s19253"] if x["hold"])],
    "hold_implied_margin_mm": sorted({round(x["implied_margin_mm"], 2) for x in E["V5/H/V5_ep1925_s19253"] if x["hold"]}),
    "hold_true_crate_sd_minus_r_mm_binding_sphere_min_max": [min(x["true_sd_mm"] for x in E["V5/H/V5_ep1925_s19253"] if x["hold"]), max(x["true_sd_mm"] for x in E["V5/H/V5_ep1925_s19253"] if x["hold"])],
    "hold_true_crate_sd_minus_r_mm_all_spheres_min": min(x["true_sd_min_all_spheres_mm"] for x in E["V5/H/V5_ep1925_s19253"] if x["hold"]),
    "apple_z_crate_frame_mm_t248_to_t367": [R["V5/H/V5_ep1925_s19253"]["series"]["apple_in_crate_z_mm"][248], R["V5/H/V5_ep1925_s19253"]["series"]["apple_in_crate_z_mm"][367]],
    "in_hand_slip_mm_t248_to_t367": [R["V5/H/V5_ep1925_s19253"]["series"]["apple_tcp_slip_mm"][248], R["V5/H/V5_ep1925_s19253"]["series"]["apple_tcp_slip_mm"][367]],
    "release_apple_floor_gap_mm": R["V5/H/V5_ep1925_s19253"]["release_apple_floor_gap_mm"]},
 },
 "Y2": {"pairs_both_success": [p[0].split("/", 1)[1].replace("V4_", "") for p in pairs], "n_pairs": len(pairs),
        "per_pair": {p[0].split("/", 1)[1].replace("V4_", ""): {"V4": {f: R[p[0]].get(f) for f in Y2}, "V5": {f: R[p[1]].get(f) for f in Y2}} for p in pairs},
        "paired_stats": ST["metrics"], "release_mode_counts": ST["release_mode"],
        "divergence": div_rows,
        "divergence_summary": {"n_pairs_with_divergence_after_t0": len(dv),
                               "n_div_chunk_eq_first_closing_kept_or_plus8": sum(1 for v in dv if v["V5_first_closing_kept_geometry_t"] is not None and v["chunk_t"] - v["V5_first_closing_kept_geometry_t"] in (0, 8)),
                               "n_div_with_reference_diff_0": sum(1 for v in dv if v["reference_rows0_7_max_abs_diff_rad"] == 0.0),
                               "n_div_at_V4_closing": sum(1 for v in dv if v["V4_grasp_state"] == "closing"), "n_div_at_V4_held": sum(1 for v in dv if v["V4_grasp_state"] == "held"),
                               "V5_latched_hold_true_chunks_in_both_success_pairs": sum(div_rows[p[0].split('/', 1)[1].replace('V4_', '')]["V5_latched_hold_true_chunks"] for p in pairs),
                               "V5_runs_with_not_supported_chunk_at_or_before_div": sum(1 for v in dv if v["V5_not_supported_chunks_at_or_before_div"] > 0)}},
}
doc = {"_comment": "ag3s-verifier (T43 Y, 2026-10-06). 숫자와 경로만 — 해석 없음. 새 closed-loop run 없음 (CPU 재생 · 기존 산출).",
 "step": "T43Y", "date": "2026-10-06",
 "runs": [{"id": "physics_replay", "command": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa .venv-openpi-live/bin/python outputs/verify/T43/Y/p/replay_y.py outputs/verify/T43/Y/replay <38 run dirs>", "raw_output": "outputs/verify/T43/Y/replay/*.npz|*.json", "stdout_log": "outputs/verify/T43/Y/raw/replay_0*.log"},
          {"id": "server_chunks", "command": "p/server_y.py raw/server_chunks.json <V4/V5 keys>", "raw_output": "outputs/verify/T43/Y/raw/server_chunks.json"},
          {"id": "esdf_held", "command": "p/esdf_held_y.py raw/esdf_held.json <7 keys>", "raw_output": "outputs/verify/T43/Y/raw/esdf_held.json"},
          {"id": "hold_rows", "command": "p/hold_rows_y.py raw/hold_rows.json <6 keys>", "raw_output": "outputs/verify/T43/Y/raw/hold_rows.json"},
          {"id": "metrics", "command": "p/y_metrics.py raw/y_metrics.json", "raw_output": "outputs/verify/T43/Y/raw/y_metrics.json", "stdout_log": "outputs/verify/T43/Y/raw/y_metrics.log"},
          {"id": "divergence", "command": "p/divergence_y.py raw/divergence.json", "raw_output": "outputs/verify/T43/Y/raw/divergence.json"}],
 "code_state": {"inputs": "outputs/verify/T43/W/runs/{V4,V5}/{H,B} (V4 snapshot ${AG3S_PROJECT_ROOT}-t43w, V5 snapshot ${AG3S_PROJECT_ROOT}-t43w3) + T43/W/analysis/analysis.json",
                "replay_client_snapshot": "${AG3S_PROJECT_ROOT}-t43w/pi05_TO_hybrid (pi05 trees of work-t43w3 == work-t43w, snapshot3.txt)",
                "root_head": subprocess.run(["git", "-C", _ag3s_workspace_path(''), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()},
 "settings": {"V4_flags": "E3b --obstacle-margin 0.010 --held-body-cover", "V5_flags": "V4 + --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60",
              "esdf_margin_mm": 0.0, "obstacle_margin_mm": 10.0, "destination_margin_mm_config": 20.0, "definitions": ["outputs/verify/T43/Y/p/y_metrics.py", "p/hold_rows_y.py", "p/esdf_held_y.py", "p/server_y.py", "p/divergence_y.py", "p/replay_y.py"]},
 "numbers": num,
 "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43y-y1-scene.png", "kind": "scene"}, {"path": "AG3S/ag3s/docs/figures/t43/t43y-y1-timeline.png", "kind": "graph"},
             {"path": "AG3S/ag3s/docs/figures/t43/t43y-y1-table.png", "kind": "table"}, {"path": "AG3S/ag3s/docs/figures/t43/t43y-y2-scene.png", "kind": "scene"},
             {"path": "AG3S/ag3s/docs/figures/t43/t43y-y2-pairs.png", "kind": "graph"}, {"path": "AG3S/ag3s/docs/figures/t43/t43y-y2-table.png", "kind": "table"}],
 "baseline_check": {"matched": None, "note": "regression baseline not run: it needs .venv-ag3s (legacy) or the GPU curobo path, both excluded by the lead for this step (CPU replay of existing outputs only)."},
 "verdict": "physics replay fidelity: qpos mismatch 0 rows in 38 / 38 runs. 해석 · 판정 없음.",
 "anomalies": ["mj_geomDistance(apple mesh, crate box) returned 0.0 and 63.4 mm at the same pose on consecutive chunks; replaced by mesh-vertex vs box signed distance (y_metrics.py)",
               "V4 H ep1807 s18071: first chunk ipc timeout (238434 ms > 120000 ms, comms HOLD) -> executed actions differ from V5 from step 0; excluded from divergence summary",
               "third_person_front.mp4 frames carry the client's avoidance-arrow overlay (green / red arrow and text)",
               "place rule (analyze_e0) and success are true for V5 ep1925 s19253 / ep1807 s18071 / s18073 although the server grasp state stays 'held' (attached) to the run end"],
 "not_measured": ["contact forces between fingers and apple as a grip-force series (only per-row max normal force per contact class is stored)",
                  "V5 flag effects one at a time (no single-flag runs exist; new closed-loop runs excluded)",
                  "ESDF / label content at the HOLD row points beyond the trilinear sample (no full field dump)"]}
OUT.write_text(json.dumps(doc, indent=1, default=float, ensure_ascii=False)); print(OUT)
print(json.dumps(num["Y2"]["divergence_summary"]), json.dumps(num["Y1"]["V5_1925_s19253"], default=float))
