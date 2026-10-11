"""T42-1: write AG3S/ag3s/docs/handoff/T42-1.verify.json from the summary and the check files (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, collections
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42')); H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
S = json.loads((T / "1/summary.json").read_text())
J = lambda p: json.loads(pathlib.Path(p).read_text())
cand = J(T / "1/candidates.json"); ext = J(T / "1/candidates_ext.json")
bl = J(T / "1/baseline/judge.json")
part = J(H / "T42-1.verify.partial.json")
runs_here = {k: v for k, v in S["runs"].items() if not k.startswith("T39/")}
cnt = collections.Counter((v["cond"], v["source"]) for v in S["runs"].values())
compact = {}
for e, t in S["table"].items():
    compact[e] = {"split": t["split"], "fruit": t["fruit"], "arm": t["arm"], "crowd_score_m": t["score_m"],
                  **{c: {r: (None if v is None else {k: v[k] for k in ("seed", "source", "success", "grasp", "place", "robust_grasp", "robust_grasp_strict",
                                                                         "failure_mode", "regrasp", "slip_B", "loss_exempt_as_place", "drift_mm_max",
                                                                         "n_rows_no_finger_within_3mm", "first_slip_t", "first_slip_xy_from_reset_m",
                                                                         "first_slip_z_above_table_m", "n_hold_chunks", "non_target_in_crate_final")})
                         for r, v in t[c].items()} for c in ("E3b", "E0")},
                  "E3b_n_done": t["E3b_n_done"], "E3b_n_success": t["E3b_n_success"], "E3b_n_success_robust_B": t["E3b_n_success_robust"],
                  "E3b_n_success_robust_strict_A": t["E3b_n_success_robust_strict"],
                  "E0_n_done": t["E0_n_done"], "E0_n_success": t["E0_n_success"], "E0_n_success_robust_B": t["E0_n_success_robust"],
                  "E0_n_success_robust_strict_A": t["E0_n_success_robust_strict"]}
sel = S["selected"]
selected = []
for f, eps in sel.items():
    for e in eps:
        t = compact[str(e)]
        selected.append({"episode": e, "fruit": f, "arm": t["arm"], "split": t["split"], "crowd_score_m": t["crowd_score_m"],
                         "E3b_success_robust_B": f'{t["E3b_n_success_robust_B"]}/{t["E3b_n_done"]}',
                         "E3b_success_robust_strict_A": f'{t["E3b_n_success_robust_strict_A"]}/{t["E3b_n_done"]}',
                         "E0_success": f'{t["E0_n_success"]}/{t["E0_n_done"]}', "E0_success_robust_B": f'{t["E0_n_success_robust_B"]}/{t["E0_n_done"]}',
                         "E0_failure_modes": [v["failure_mode"] for v in t["E0"].values() if v], "seeds": [10 * e + r for r in (1, 2, 3)],
                         "seed_sources": {r: (v or {}).get("source") for r, v in t["E3b"].items()}})
fig = lambda n, k: {"path": f"AG3S/ag3s/docs/figures/t42/{n}.png", "kind": k, "sidecar": f"AG3S/ag3s/docs/figures/t42/{n}.json"}
out = {
 "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
 "step": "T42-1", "date": "2026-10-05",
 "runs": [
  {"id": "regression baseline (legacy + cuRobo) on the snapshot", "command": "outputs/verify/T42/1/baseline/run_baseline.sh; judge.py",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T42/1/baseline/{legacy,curobo,judge}.json", "stdout_log": "outputs/verify/T42/1/baseline/{legacy,curobo}.log"},
  {"id": "candidates / crate-rule / address checks", "command": "p/candidates.py; p/candidates_ext.py; p/address_check.py (test + ext)",
   "venv": "python3 / .venv-openpi-live (CPU)", "raw_output": "outputs/verify/T42/1/{candidates,candidates_ext,address_check,address_check_ext}.json"},
  {"id": "E3b / E0 no-obstacle runs", "command": "p/launch_t42.sh (v1: p1, p2a), p/launch_t42b.sh (v2: x1, x2, x3), p/launch_t42c.sh (v3: x2 x3 x2 x3 on 8250; e0 on 8252/8251), p/par_chain.sh, p/par_e0.sh -> p/run_chain.sh -> p/start_server.sh / p/run_client.sh / p/check_run.py",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T42/1/runs/{E3b,E0}/noobs/", "stdout_log": "outputs/verify/T42/1/runs/{launch*,par_*,resume_*}.out, runs/{E3b,E0}/chain_825*.log"},
  {"id": "analysis", "command": "p/summarize_t42.py (analyze_t42.py, grasp_robust.py, plan.py); p/fig_t42_1.py; p/build_verify_t42_1.py",
   "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T42/1/summary.json", "stdout_log": "outputs/verify/T42/1/summary.log"}],
 "code_state": {"snapshot": "${AG3S_PROJECT_ROOT}-t42", "benchmark": "git archive e5debe5 + ag3s/asset/data symlink", "pi05_TO_hybrid": "git archive 98e07d6 (rby1_bringup, rby1_description/models, rby1_manipulation); other entries symlinked",
                "code_md5": "outputs/verify/T42/p/code_md5_t42.txt (399 files)", "code_md5_diff_vs_T40_snapshot": "outputs/verify/T42/1/code_md5_diff_vs_t40.txt (AG3S/ag3s/config.py, AG3S/trajopt/serve_safe.py)",
                "import_check": ["outputs/verify/T42/1/import_server.json", "outputs/verify/T42/1/import_client.json"], "dirty_files": []},
 "numbers": {
  "_comment": "seed = 10*ep + rep. success / grasp / place = T28 rule generalized to the episode's target fruit (p/analyze_t42.py). robust grasp: p/grasp_robust.py; B (primary, lead decision 2026-10-05) exempts a first slip event over the crate interior footprint (crate-local |x|<0.077, |y|<0.137, z>-0.052) in a run that ends placed; A (strict) has no exemption. library = E3b success with robust grasp (B) on 3/3 seeds; apple + left rep 1-2 are T39 runs (snapshot work-o1).",
  "settings": {"server_E3b": "--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 --no-subtask-gate",
               "server_E0": "--no-safe --no-subtask-gate", "xla_flags": "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0",
               "client": "--warmup-steps 1000 --policy-seed SEED --max-steps 600 --record-frames --record third_person_front.mp4 --view front --trajectory-out traj.npz --obstacle-stop-distance 0; E3b: --safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators <episode used_arm> --safe-gate reasons --safe-hold-mode fixed",
               "ports": [8250, 8251, 8252], "server_proc_flags": S["servers"]},
  "regression_baseline": bl,
  "subtask_gate_off_in_server_record": "summary subtask.gate false (server_8250_20261005T070710 chunk record); server log 'gate off'",
  "reproduction_ep1828_s18281_vs_T39_and_T40_G_B": S["reproduction_ep1828_s18281"],
  "address_check": {"qposadr": J(T / "1/address_check.json")["qposadr"], "test_episodes_all_consistent": J(T / "1/address_check.json")["all_consistent"],
                    "ext_episodes_pre_settle_equal_and_xpos_equal": all(all(v["pre_settle_qpos_equals_record"].values()) and v["settled_target_body_xpos_minus_qpos_max_abs"] < 1e-9 for v in J(T / "1/address_check_ext.json")["episodes"].values()),
                    "ext_all_consistent_flag_note": "address_check_ext.json all_consistent=false only because its test requires split == test"},
  "crate_rule_check_demo_records": {k: v for k, v in cand["crate_rule_check_demo_records"].items() if k != "target_final_outside"},
  "candidate_pools": {"test_split": {k: {"pool_size": v["pool_size"], "selected": v["selected"]} for k, v in cand["candidates"].items()},
                      "train_val_ext": {k: {"pool_size": v.get("pool_size"), "selected": v["selected"]} for k, v in ext["candidates"].items()},
                      "ext_filter": "train/validation, place_one, record success, grasp_attempts 1, grasp_retries 0"},
  "run_counts": {"T42_runs_E3b": cnt[("E3b", "T42")], "T42_runs_E0": cnt[("E0", "T42")], "T39_runs_reused_E3b": cnt[("E3b", "T39")], "T39_runs_reused_E0": cnt[("E0", "T39")],
                 "partial_run_counts": part.get("run_counts"), "n_failed_attempts": 0},
  "library_E3b_3of3_robust_B": S["library_E3b_3of3"], "library_E3b_3of3_robust_strict_A": S["library_E3b_3of3_strict_A"],
  "selected": selected,
  "E3b_failure_modes_per_fruit_arm_all_screened_runs": S["E3b_failure_modes_per_fruit_arm"],
  "per_episode": compact},
 "figures": [fig("t42-1-table", "table"), fig("t42-1-screening", "graph")] + [fig(f"t42-1-scene-{s['episode']}", "scene") for s in selected],
 "baseline_check": {"matched": bool(bl["legacy"]["matched"] and bl["curobo"]["matched"]),
                    "note": f'legacy {bl["legacy"]["violated_start"]} / {bl["legacy"]["has_target"]} / {bl["legacy"]["frame0_clearance_before_mm"]}; cuRobo {bl["curobo"]["violated_start"]} / {bl["curobo"]["has_target"]} / {bl["curobo"]["frame0_clearance_before_mm"]}; snapshot ${AG3S_PROJECT_ROOT}-t42'},
 "verdict": "baseline matched (legacy and cuRobo); E3b and E0 ep1828 s18281 bit-identical to T39 and T40 G B",
 "not_measured": ["banana: no episode with E3b robust (B) 3/3 among 20 screened (test 6, train/val 14); counts in numbers.per_episode",
                  "E0 rep 1-2 of apple + left episodes on this snapshot (T39 runs used)",
                  "E0 of the non-selected library episode 1854 (not run); E0 rep 1-2 of the apple + left episodes are T39 runs",
                  "grasp robustness thresholds (15 mm drift, 3 mm finger gap) were not varied"],
 "incidents": ["p2b of plan v1 superseded by the robust-grasp instruction (plan.py exits for p2b; launcher v1 stopped there)",
               "session restart ~11:00-11:30 UTC: detached chains continued; one duplicate E0 chain on 8251 (launch_t42c e0 after_pid) was stopped at preflight before its server started (no run affected)",
               "memory rule changed 07:2x UTC (working set); the first chain-1 waiting under the old rule was killed and restarted (resume_p1.sh, resume_p1b.sh)"],
}
(H / "T42-1.verify.json").write_text(json.dumps(out, indent=1, default=float))
print("written", len(json.dumps(out)))
print(json.dumps(selected, indent=0)[:2500])
