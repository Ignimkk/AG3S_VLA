"""T43 SV — assemble AG3S/ag3s/docs/handoff/T43SV.verify.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
SV = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/SV/raw'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43SV.verify.json'))
assert not OUT.exists(), OUT
gt = json.load(open(SV / "grasp_table.json"))["pairs"]; sag = json.load(open(SV / "sag_table.json"))["arms"]; alt = json.load(open(SV / "alt_table.json"))


def fidelity(pair):
    g = json.load(open(SV / f"grasp_{pair}.json")); out = {}
    for arm in (g["A"], g["B"]):
        for kind in ("wp", "path"):
            v = [((c.get("recon") or {}).get(kind) or {}).get("fidelity_mm") for r in g["runs"].values() for c in r["arms"][arm]["chunks"]]
            v = [x for x in v if x is not None]
            out[f"{arm}_{kind}"] = {"n": len(v), "n_le_1mm": sum(x <= 1.0 for x in v), "max_mm": max(v, default=None)}
        cl = [((c.get("recon") or {}).get("wp") or {}) for r in g["runs"].values() for c in r["arms"][arm]["chunks"]]
        cl = [x for x in cl if x.get("class") and x.get("recorded_class")]
        out[f"{arm}_wp_class_agree"] = {"n": len(cl), "agree": sum(x["class"] == x["recorded_class"] for x in cl)}
    return out


num = {"code_refs_snapshot_work_t43v8": {
    "servo_path_configs_row_end_states": "AG3S/trajopt/sweep.py:222-332 (servo X rows 238-247)",
    "path_rows_linearized_through_waypoint_clearance": "AG3S/trajopt/sweep.py:359-398 -> linearize.py:1290-1352 (row_floor) -> linearize.py:652-785 (_esdf_clearance)",
    "path_violation_check": "AG3S/trajopt/sweep.py:400-425 -> linearize.py:1277-1288 (flat_clearances -> _esdf_clearance)",
    "path_target_rows_dropped": "AG3S/trajopt/sweep.py:427-460 (applied 375, 418)",
    "path_floor_min0_hqnow": "AG3S/trajopt/sweep.py:187-204",
    "target_free_layer_authorized_rows": "AG3S/trajopt/linearize.py:714-728, 1044-1052",
    "T30_F1_exempt_rows": "AG3S/trajopt/linearize.py:731-737, 1054-1072",
    "E1_manipulated_link_margin": "AG3S/trajopt/linearize.py:760-778",
    "obstacle_margin_classes": "AG3S/trajopt/linearize.py:779-785, 804-849",
    "stay_start_when_reference_violates": "AG3S/trajopt/sqp.py:241-253, 437-461",
    "path_violation_in_merit": "AG3S/trajopt/sqp.py:518",
    "sqp_max_iterations_time_budget": "AG3S/trajopt/config.py:442-443"}}
for pair in ("V9V8", "V3V4"):
    P = gt[pair]
    num[f"grasp_{pair}"] = {"A": P["A"], "B": P["B"], "summary": P["summary"], "reconstruction_fidelity": fidelity(pair),
                            "failure_rows": [{"key": r["key"], "failure_type": r["failure_type"], "div10_t": r["div10_t"], "div10_state_A": r["div10_state_A"],
                                              "div10_state_B": r["div10_state_B"], "div30_t": r["div30_t"], "div30_state_A": r["div30_state_A"],
                                              "first_reference_diff": r["first_reference_diff"],
                                              "first_closing_t_A_B": [r["A"]["first_closing_t"], r["B"]["first_closing_t"]],
                                              "tip_apple_mm_first_closing_A_B": [r["A"].get("tip_apple_mm_first_closing"), r["B"].get("tip_apple_mm_first_closing")],
                                              "tip_alignment_first_closing_A": r["A"].get("tip_alignment_first_closing"),
                                              "tip_alignment_first_closing_B": r["B"].get("tip_alignment_first_closing"),
                                              "window_t_A": r["A"]["window_t"], "window_n_outward_gt2_A_B": [r["A"]["window_n_outward_gt2"], r["B"]["window_n_outward_gt2"]],
                                              "window_outward_mm_A": r["A"]["window_outward_mm"], "window_outward_mm_B": r["B"]["window_outward_mm"],
                                              "closing_outward_mm_A": r["A"]["closing_outward_mm"], "closing_outward_mm_B": r["B"]["closing_outward_mm"],
                                              "window_stay_starts_A_B": [r["A"]["window_stay_starts"], r["B"]["window_stay_starts"]],
                                              "window_binding_counts_A": r["A"]["window_binding_counts"], "window_binding_counts_B": r["B"]["window_binding_counts"],
                                              "window_path_binding_exemptions_A": r["A"]["window_path_binding_exemptions"],
                                              "first_push_chunk_A": r["A"]["first_push_chunk"], "attach_revoked_A": r["A"]["attach_revoked"]}
                                             for r in P["rows"] if r["kind"] == "failure"]}
num["finger_table_last3_chunks_before_closing"] = sag
num["alternatives"] = {"contact_tables": alt["contact_tables"], "hold_upper_bounds": alt["hold_upper_bounds"],
                       "a_servo_return_activation": alt["a_servo_return_activation"], "n_contacts": alt["n_contacts"],
                       "per_contact": alt["per_contact"]}
V = {"_comment": "ag3s-verifier T43 SV. numbers and paths only. CPU, recorded data only, no new runs.",
     "step": "T43SV", "date": "2026-10-07",
     "runs": [
         {"id": "sv_grasp", "command": "bash outputs/verify/T43/SV/p/run_grasp.sh (V9V8: T43W_SNAP=work-t43v8; V3V4: T43W_SNAP=work-t43w; PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8; JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa)",
          "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/SV/raw/grasp_{V9V8,V3V4}.json", "stdout_log": "outputs/verify/T43/SV/raw/grasp_{V9V8,V3V4}.log"},
         {"id": "sv_rows", "command": "bash outputs/verify/T43/SV/p/run_rows.sh (V7g work-t43v7, V8 / V9 work-t43v8) ; bash outputs/verify/T43/SV/p/run_rows_w.sh (V3 / V4 work-t43w)",
          "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/SV/raw/rows_{V7g,V8,V9,V3,V4}.json", "stdout_log": "outputs/verify/T43/SV/raw/rows_{V7g,V8,V9,V3,V4}.log"},
         {"id": "aggregation", "command": "python3 outputs/verify/T43/SV/p/sv_grasp_agg.py raw/grasp_table.json ; sv_sag_agg.py raw/sag_table.json ; sv_alt_agg.py raw/alt_table.json",
          "venv": "system python3", "raw_output": "outputs/verify/T43/SV/raw/{grasp_table,sag_table,alt_table}.json", "stdout_log": "outputs/verify/T43/SV/raw/{grasp_agg,sag_agg,alt_agg}.log"},
         {"id": "figures", "command": "fig_scene_sv.py (H ep1968_s19681) · fig_graph_sv.py · fig_table_sv.py", "venv": ".venv-openpi-live",
          "raw_output": "AG3S/ag3s/docs/figures/t43/t43sv-*.json", "stdout_log": None}],
     "scripts": [f"outputs/verify/T43/SV/p/{s}" for s in ("sv_grasp.py", "run_grasp.sh", "sv_grasp_agg.py", "sv_rows.py", "run_rows.sh", "run_rows_w.sh",
                                                          "sv_sag_agg.py", "sv_alt_agg.py", "fig_scene_sv.py", "fig_graph_sv.py", "fig_table_sv.py", "build_verify_sv.py")],
     "code_state": {"repo": "benchmark", "head": "not used; server code read from snapshot ${AG3S_PROJECT_ROOT}-t43v8 (benchmark git archive 786cf1b); client snapshots work-t43v7 / work-t43v8 / work-t43w", "dirty_files": []},
     "numbers": num,
     "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43sv-scene-closing.png", "kind": "scene"},
                 {"path": "AG3S/ag3s/docs/figures/t43/t43sv-graph.png", "kind": "graph"},
                 {"path": "AG3S/ag3s/docs/figures/t43/t43sv-table.png", "kind": "table"}],
     "baseline_check": {"matched": None, "note": "regression-baseline not run: it needs .venv-ag3s, prohibited for this step. Reconstruction fidelity in numbers.grasp_*.reconstruction_fidelity; current-margin new-HOLD upper bound reproduces 0 of 917 return and 0 of 431 carry/crate chunks (recorded feasible chunks)"},
     "verdict": "facts only; interpretation by lead and user",
     "not_measured": ["TO re-solve with any alternative (offline row checks on recorded plans only; seen = recorded plan / servo-predicted rows below the margin)",
                      "destination label of held rows (labels not recorded; crate proximity by MuJoCo truth label)",
                      "own-apple residual rule of HM (proxy: truth label apple excluded, 'unexplained' reported separately)",
                      "sweep-only arm (no closed-loop record with --sweep-check and without --servo-model)"]}
OUT.write_text(json.dumps(V, indent=1, default=float))
print("wrote", OUT)
