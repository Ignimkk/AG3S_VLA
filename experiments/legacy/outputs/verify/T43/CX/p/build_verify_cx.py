"""T43 CX verify.json builder (numbers and paths only). usage: build_verify_cx.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
X = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/CX')); S = json.load(open(X / "raw/cx_summary.json"))
GP = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V11ABC.verify.json')))["numbers"]["grasp_phase"]
cases = {}
for ck, E in S["cases"].items():
    cases[ck] = {"divergence": E["divergence"]}
    for arm in ("V11AB", "V11ABC"):
        r = dict(E[arm]); cases[ck][arm] = r
n = {
 "cases": cases,
 "counts": {
   "n_cases": 7, "first_divergent_refined_seq2_t8": sum(1 for E in S["cases"].values() if E["divergence"]["first_refined_seq_t"] == [2, 8]),
   "first_divergent_refined_seq3_t16": sum(1 for E in S["cases"].values() if E["divergence"]["first_refined_seq_t"] == [3, 16]),
   "first_divergent_grasp_state": {s: sum(1 for E in S["cases"].values() if E["divergence"]["grasp_state_at_first"]["V11ABC"] == s) for s in ("searching", "latched")},
   "abc_separation_within_20_rows_of_grasp_row": sum(1 for E in S["cases"].values() if (E["V11ABC"].get("rows_in_hand") or 99) <= 20),
   "abc_separation_after_hold_stretch": sum(1 for E in S["cases"].values() if (E["V11ABC"].get("hold_rows_in_hand") or 0) > 0),
   "abc_hold_rows_in_hand_nonzero": {ck: E["V11ABC"]["hold_rows_in_hand"] for ck, E in S["cases"].items() if E["V11ABC"].get("hold_rows_in_hand")},
   "abc_chunks_with_server_n_over_gt0_grasp-8_to_sep": {ck: [E["V11ABC"]["c_rows_server"]["n_chunks_n_over_gt0"], E["V11ABC"]["c_rows_server"]["n_chunks"]] for ck, E in S["cases"].items()},
   "abc_boundary_velocity_ratio_max_grasp-8_to_sep": {ck: E["V11ABC"]["c_rows_server"]["boundary_velocity_ratio_max"] for ck, E in S["cases"].items()},
   "abc_lift_meas_acc_gt_ab": sum(1 for E in S["cases"].values() if E["V11ABC"]["lift_window"]["acc_meas_max"] > E["V11AB"]["lift_window"]["acc_meas_max"]),
   "abc_first_closing_outward_gt_2mm": sum(1 for E in S["cases"].values() if (E["V11ABC"]["first_closing"]["outward_mm"] or 0) > 2),
   "ab_first_closing_outward_gt_2mm": sum(1 for E in S["cases"].values() if (E["V11AB"]["first_closing"]["outward_mm"] or 0) > 2),
   "external_contacts_apple_or_hand_vs_obstacle_before_sep_abc": "0 rows in all 7 cases (cx_cases window apple_obstacle / hand_obstacle_contact_rows_before_tsep)",
 },
 "grasp_phase_pooled_from_V11ABC_verify": {a: {"first_closing_measured_tip_apple_mm_at_chunk_start": GP[a]["first_closing_measured_tip_apple_mm_at_chunk_start"],
                                              "closing_chunks_pooled_pre_release_n": GP[a]["closing_chunks_pooled_pre_release"]["n"],
                                              "closing_chunks_pooled_pre_release_n_outward_gt_2mm": GP[a]["closing_chunks_pooled_pre_release"]["n_outward_gt_2mm"]} for a in ("V11AB", "V11ABC")},
 "hold": S["hold"],
 "offline_c_toggle": S["offline_c_toggle"],
 "reversal": {"share_ALL": {"V11AB": [228, 1355, 0.16826568265682657], "V11ABC": [267, 1298, 0.20570107858243453]},
              "split_at_end_of_task_row": {a: S["reversal_split"][a]["agg_n_boundaries_n_reversal"] for a in S["reversal_split"]}},
 "a_c_rad_per_row2": json.load(open(X / "raw/cx_cases.json"))["a_c_rad_per_row2"], "max_step_change_rad_per_row2": 0.04,
 "grip_population": json.load(open(X / "raw/cx_grip_pop.json"))["runs"],
}
rev = json.load(open(X / "raw/cx_reversal.json"))["arms"]
n["reversal"]["by_phase"] = {a: {ph: {k: rev[a]["summary"][ph][k] for k in ("n", "n_reversal", "share", "rev_vmin_mm", "rev_vmax_mm", "rev_turn_deg")} for ph in ("approach", "grasp", "carry", "after")} for a in rev}
n["reversal"]["rev_vmin_lt5mm"] = {a: [rev[a]["summary"]["rev_vmin_lt5mm"], rev[a]["summary"]["n_reversal"]] for a in rev}
n["reversal"]["per_run_V11AB_vs_V11ABC"] = {k.split("/", 1)[1].split("_", 1)[1]: [rev["V11AB"]["summary"]["per_run"][k]["n_reversal"], rev["V11AB"]["summary"]["per_run"][k]["n"],
                                              rev["V11ABC"]["summary"]["per_run"][k.replace("V11AB", "V11ABC")]["n_reversal"], rev["V11ABC"]["summary"]["per_run"][k.replace("V11AB", "V11ABC")]["n"]]
                                          for k in rev["V11AB"]["summary"]["per_run"]}
code_facts = {
 "C_rows": "AG3S/trajopt/problem.py:135-164 servo_accel_rows: X = [u-2, u-1, Q0..], relaxed bound = max(a_C, |row on reference incl. history|) (:156), capped by max_step_change (:163)",
 "C_slack_weight": "problem.py:330-332 w_slack * servo_accel_weight (0.01) per rad/row^2; merit term problem.py:499",
 "C_constraint_rows": "problem.py:416-417 servo_accel / servo_accel_lower; boundary velocity |Q0 - u-1| <= max_step problem.py:425-426, weight 10 w_slack",
 "a_C": "AG3S/trajopt/servo.py:125 accel_bound = eps dt^2 / kappa",
 "history": "AG3S/trajopt/safe_policy.py:543 command_history = exec_feedback.applied_arm[-2:] (HOLD rows included)",
 "record": "AG3S/trajopt/sqp.py:943-965 to.metrics.servo_accel (n_over, overshoot_max_rad, boundary_velocity_ratio)",
 "returned": "AG3S/trajopt/sqp.py:495 / :973-982 returned = initial when no candidate is accepted",
 "snapshot": "${AG3S_PROJECT_ROOT}-t43v11 (benchmark 3ca1b21); code md5 of 407 files == V11 p/code_md5_v11.txt (raw/code_md5_cx.txt)"}
V = {"_comment": "ag3s-verifier. numbers and paths only.", "step": "T43CX", "date": "2026-10-09",
 "runs": [
  {"id": "replay_B", "command": "outputs/verify/T43/CX/p/run_replays_cx.sh (CT/p/replay_ct.py unchanged, CPU, 2 workers)", "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/CX/replay/", "stdout_log": "outputs/verify/T43/CX/raw/replay_job*.log", "qpos_mismatch_rows": 0},
  {"id": "extract", "command": "outputs/verify/T43/CX/p/cx_extract.py raw/cx_cases.json raw/tl", "raw_output": "outputs/verify/T43/CX/raw/cx_cases.json", "stdout_log": "outputs/verify/T43/CX/raw/cx_extract.log"},
  {"id": "hold", "command": "outputs/verify/T43/CX/p/cx_hold.py raw/cx_hold.json", "raw_output": "outputs/verify/T43/CX/raw/cx_hold.json", "stdout_log": "outputs/verify/T43/CX/raw/cx_hold.log"},
  {"id": "inhand", "command": "outputs/verify/T43/CX/p/cx_inhand.py raw/cx_inhand.json raw/tl", "raw_output": "outputs/verify/T43/CX/raw/cx_inhand.json", "stdout_log": "outputs/verify/T43/CX/raw/cx_inhand.log"},
  {"id": "offline_c_toggle", "command": "outputs/verify/T43/CX/p/run_offline_cx.sh (cx_offline.py wraps AG3S.trajopt.experiments.ta_offline; variants base/noC on V11ABC, base/addC on V11AB; 1807 s18071 / s18073 chunks 16-31)", "raw_output": "outputs/verify/T43/CX/offline/", "stdout_log": "outputs/verify/T43/CX/offline/*.log"},
  {"id": "reversal", "command": "outputs/verify/T43/CX/p/cx_reversal.py, cx_reversal_split.py", "raw_output": "outputs/verify/T43/CX/raw/cx_reversal.json, raw/cx_reversal_split.json", "stdout_log": "outputs/verify/T43/CX/raw/cx_reversal.log"},
  {"id": "grip_population", "command": "outputs/verify/T43/CX/p/cx_grip_pop.py", "raw_output": "outputs/verify/T43/CX/raw/cx_grip_pop.json"},
  {"id": "summary", "command": "outputs/verify/T43/CX/p/cx_summary.py", "raw_output": "outputs/verify/T43/CX/raw/cx_summary.json"}],
 "code_state": {"repo": "benchmark", "head": "3ca1b21 (snapshot ${AG3S_PROJECT_ROOT}-t43v11)", "dirty_files": [], "code_md5": "outputs/verify/T43/CX/raw/code_md5_cx.txt == outputs/verify/T43/V11/p/code_md5_v11.txt"},
 "numbers": n, "code_facts": code_facts,
 "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43cx-1-drop-scene.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/t43/t43cx-1-drop-scene.json"},
             {"path": "AG3S/ag3s/docs/figures/t43/t43cx-2-timeline.png", "kind": "graph", "sidecar": "AG3S/ag3s/docs/figures/t43/t43cx-2-timeline.json"},
             {"path": "AG3S/ag3s/docs/figures/t43/t43cx-3-cause-table.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/t43/t43cx-3-cause-table.json"}],
 "baseline_check": {"matched": None, "note": "not run in CX (lead: CPU replays and recorded data only, no .venv-ag3s, no GPU). Snapshot code md5 (407 files) identical to the V11 round, whose baseline matched (T43V11ABC.verify.json baseline_check: 10/15, -9.171877401271193 mm)."},
 "verdict": "facts only: snapshot code identical to V11 round; replays qpos_mismatch_rows 0 (4 new B replays); offline base re-solve of HOLD chunks equals record (vs_recorded 0.00 deg).",
 "left_finger_name_note": "finger bodies in cx_inhand / replays are MJCF names (ee_finger_l1 = geom 1079 side, l2 = geom 1086 side); server link names (URDF) are not paired by name here.",
 "anomalies": [
   "lead text: 17 held HOLDs in 1807 s18071/s18073; measured: 16 there (9 + 7) + 1 in H 1967 s19673 (t208, -0.2 mm, destination crate_above); s18073 also has 2 finger HOLDs (t144, t160)",
   "H 1967 s19672 is dropped in V11AB too (t_sep 227); V11ABC separation at 167",
   "the two 'grasp failures' (1982 H/B) are attach -> separation 6 rows later (H: apple lifted 77 mm, t 128 -> 134; B: t 120 -> 126); fail_stage rule labels them no_grasp",
   "offline ta_offline base re-solve of V11AB s18071 seq23 is violated (41.7 mm) while the record executed (vs_recorded 4.37 deg): open-loop fidelity limit",
   "V11A (no B, no C) H also has 4 drops (1967 s19673, 1982 s19822/3, 1995 s19953; cx_grip_pop)"],
 "not_measured": ["new closed-loop runs (none launched)", "per-finger contact forces (replays store aggregates only)", "regression baseline (not run, see baseline_check)"]}
pathlib.Path(sys.argv[1]).write_text(json.dumps(V, indent=1, default=float)); print(sys.argv[1])
