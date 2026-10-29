"""T43 HX — assemble AG3S/ag3s/docs/handoff/T43HX.verify.json from raw/hx_summary.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
S = json.load(open(_ag3s_workspace_path('outputs/verify/T43/HX/raw/hx_summary.json')))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43HX.verify.json'))
assert not OUT.exists(), OUT
num = {"window_definition_mm": 100.0, "margin_threshold_mm": 10.0, "max_step_deg_config_doc": 10.8, "arms": {}}
for arm, A in S["arms"].items():
    ph = {}
    for p, P in A["phase"].items():
        ph[p] = {"n_window_rows": P["n_window_rows"], "n_window_rows_exec": P["n_window_rows_exec"], "n_window_rows_hold": P["n_window_rows_hold"], "groups": {}}
        for g, G in P["groups"].items():
            ph[p]["groups"][g] = {k: G[k] for k in ("min_P_geo_mm", "min_M_geo_mm", "min_TO_clear_mm", "min_TO_d_minus_r_mm", "esdf_minus_true_mm",
                                                    "n_moments_exec", "n_moments_exec_contact", "episodes_exec", "moments_exec", "moments_exec_contact",
                                                    "moments_hold", "moments_hold_contact", "P_moments", "servo_flag_geo", "servo_flag_esdf",
                                                    "servo_flag_any", "n_with_S", "servo_flag_geo_contact", "n_contact_with_S")}
    num["arms"][arm] = {"n_runs": A["n_runs"], "n_runs_grasp": A["n_runs_grasp"], "phase": ph, "servo": A["servo"], "fidelity": A["fidelity"],
                        "window_chunks": A["window_chunks"], "n_ii_moments": len(A["ii_events"]), "to_neg_exec_rows": A["to_neg_exec_rows"],
                        "apple_P_lt10_rows": A["apple_P_lt10_rows"]}
num["contacts22_summary"] = S["contacts22_summary"]
num["contacts22"] = [{k: r.get(k) for k in ("arm", "key", "group", "phase_ct", "t", "k", "geo_C_mm", "geo_M_mm", "geo_S_mm", "servo_err_mm", "cmd_err_mm",
                                            "to_clear_mm", "to_true_sphere_mm", "to_at_S_clear_mm", "flag_geo_contact", "flag_geo_margin", "flag_esdf", "M_src")}
                     for r in S["contacts22"]]
num["contacts_other_n"] = len(S["contacts_other"])
V = {"_comment": "ag3s-verifier T43 HX. numbers and paths only. CPU, recorded data only, no new runs.",
     "step": "T43HX", "date": "2026-10-07",
     "runs": [
         {"id": "hx_measure", "command": "bash outputs/verify/T43/HX/p/run_hx.sh (one process per arm: T43W_SNAP=work-t43v7 V7g | work-t43v8 V8 | work-t43w3 V5; PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8; JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=osmesa)",
          "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/HX/raw/hx_{V7g,V8,V5}.json", "stdout_log": "outputs/verify/T43/HX/raw/hx_{V7g,V8,V5}.log"},
         {"id": "hx_agg", "command": "python3 outputs/verify/T43/HX/p/hx_agg.py", "venv": "system python3", "raw_output": "outputs/verify/T43/HX/raw/hx_summary.json", "stdout_log": "outputs/verify/T43/HX/raw/hx_agg.log"},
         {"id": "figures", "command": "fig_scene_hx.py (T43W_SNAP=work-t43v8; V8 V8/H/V8_ep1807_s18073 174 270) · fig_timeline_hx.py · fig_table_hx.py", "venv": ".venv-openpi-live",
          "raw_output": "AG3S/ag3s/docs/figures/t43/t43hx-*.json", "stdout_log": None}],
     "scripts": ["outputs/verify/T43/HX/p/hx_measure.py", "outputs/verify/T43/HX/p/run_hx.sh", "outputs/verify/T43/HX/p/hx_agg.py",
                 "outputs/verify/T43/HX/p/fig_scene_hx.py", "outputs/verify/T43/HX/p/fig_timeline_hx.py", "outputs/verify/T43/HX/p/fig_table_hx.py",
                 "outputs/verify/T43/HX/p/build_verify_hx.py"],
     "code_state": {"repo": "benchmark", "head": "c908895 (not used); server code read from snapshot ${AG3S_PROJECT_ROOT}-t43v8 (benchmark git archive 786cf1b): linearize.py, problem.py, servo.py, sqp.py, fixed_input_ablation.build_field; client snapshots work-t43v7 / work-t43v8 / work-t43w3", "dirty_files": []},
     "numbers": num,
     "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43hx-scene.png", "kind": "scene"},
                 {"path": "AG3S/ag3s/docs/figures/t43/t43hx-timeline.png", "kind": "graph"},
                 {"path": "AG3S/ag3s/docs/figures/t43/t43hx-cause-table.png", "kind": "table"}],
     "baseline_check": {"matched": None, "note": "regression-baseline not run: it needs .venv-ag3s, which the lead prohibited for this step. Reconstruction fidelity instead: TO worst rows with true bar distance < 50 mm reproduced within 1 mm in V7g 81/81, V8 161/161, V5 69/70"},
     "verdict": "facts only; interpretation by lead and user",
     "not_measured": ["destination margin of held rows (field labels not recorded; held rows reconstructed with margin 0)",
                      "per-joint URDF velocity limits (max_step taken as 10.8 deg from config.py:261 doc; recorded steps up to 0.32 rad exceed it)",
                      "TO re-solve with the servo model (offline prediction only; flags = predicted state below margin, not a solve)"]}
OUT.write_text(json.dumps(V, indent=1, default=float))
print("wrote", OUT)
