"""Builds AG3S/ag3s/docs/handoff/T43V10.verify.json from V10/{agg,ct,extra,hold_join,hold_cause}.json (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, collections
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V10')); T = V.parent
a = json.load(open(V / "agg.json")); ex = json.load(open(V / "extra.json")); ct = json.load(open(V / "ct.json")); hj = json.load(open(V / "hold_join.json"))
an = json.load(open(V / "analysis/analysis.json"))["runs"]; an8 = json.load(open(T / "V8/analysis/analysis.json"))["runs"]; ld8 = json.load(open(T / "V8/extra.json"))["loadavg_during_V8"]
hold = {x: {k: a["hold"][x][k] for k in ("total", "trailing", "runs_with_hold", "by_category", "by_target", "by_phase", "by_binding_link", "cells", "hold_kinds", "by_stage_total")} for x in ("V8", "V10")}
hold["per_run_chunks_V10"] = {k: v["chunks"] for k, v in a["hold"]["V10"]["per_run"].items() if v["n_hold"]}
hold["per_run_chunks_V8"] = {k: v["chunks"] for k, v in a["hold"]["V8"]["per_run"].items() if v["n_hold"]}
ft = {x: {k: a["follow_to"][x][k] for k in ("decision_counts", "refused_counts", "n_executed", "n_runs_with_executed", "executed")} for x in ("V8", "V10")}
tm = {x: {s: {k: a["timing"][x][s][k] for k in ("n_chunks_excl_first", "to_solve_ms", "verdict_trajopt_ms", "verdict_total_ms", "round_trip_ms", "round_trip_gt_533_3_n", "round_trip_n", "round_trip_gt_533_3_share", "time_budget_hit_count", "runs_with_time_budget_hit", "linearize_ms", "qp_ms", "check_ms")} for s in ("ALL", "H", "B")} for x in ("V8", "V10")}
tm["per_run_to_solve_median"] = ex["per_run_to_solve_median"]; tm["loadavg_during_V10"] = ex["loadavg_during_V10"]; tm["loadavg_during_V8"] = ld8
mi = {x: sum(1 for r in an.values() for c in (r.get("server_chunks") or []) if (c.get("to_status") or "") != "" and c.get("sqp_iterations") is not None and False) for x in ()}
# max_iterations_hit: counted from the server chunk records' to summary
import glob, numpy as np
mih = 0; nch = 0; keys_seen = collections.Counter()
for f in glob.glob(str(V / "runs/V10/server_*/server_constraints/run_*/chunk_*.npz")):
    s = json.loads(str(np.load(f, allow_pickle=True)["summary_json"])); to = s.get("to") or {}; nch += 1
    for k in to:
        if "max_iter" in k: keys_seen[k] += 1
    mih += bool(to.get("max_iterations_hit"))
mih8 = 0; nch8 = 0
for f in glob.glob(str(T / "V8/runs/V8/server_*/server_constraints/run_*/chunk_*.npz")):
    s = json.loads(str(np.load(f, allow_pickle=True)["summary_json"])); to = s.get("to") or {}; nch8 += 1; mih8 += bool(to.get("max_iterations_hit"))
tm["max_iterations_hit"] = {"V10": mih, "V8": mih8, "n_chunk_records_V10": nch, "n_chunk_records_V8": nch8, "to_keys_containing_max_iter_V10": dict(keys_seen)}
# HM held-row records over attached V10 chunks
wcls = collections.Counter(); hr = collections.Counter(); mins = {}; n_att = 0
for r in an.values():
    for c in r["server_chunks"]:
        if c["attached"]:
            n_att += 1
            if c["wp"]["category"] == "held": wcls[c["wp"]["class"]] += 1
            for cl, v in (c.get("held_rows") or {}).items():
                hr[cl] += v["n"]
                if v["min_clearance_m"] is not None: mins[cl] = min(mins.get(cl, 1e9), 1e3 * v["min_clearance_m"])
hm = {"attached_server_chunks": n_att, "worst_row_is_held_row_by_class": dict(wcls), "held_rows_n_by_class_summed_over_chunks": dict(hr), "held_rows_min_clearance_mm_by_class": mins}
pairs_per_run = []
for k in sorted(ct["runs"]):
    pass
per = {}
for k, o in ct["runs"].items():
    per.setdefault(f"{o['episode']}_s{o['seed']}", {})[o["arm"]] = {"success": o["success"], "grasp": o["grasp"], "place": o["place"], "success_rel": o["success_rel"], "success_ct": o["success_ct"], "obstacle_stop": o["obstacle_stop"],
        "contacts": [(c["phase"], c["body"], c["kind"], c["first_t"], round(c["max_pen_mm"], 2)) for c in o["contacts"]], "hurdle_B_min_mm": 1e3 * ({**an, **an8})[k]["hurdle"]["phases"]["all"]["B"]["min_m"]}
srv = ex["servers"]
out = {"_comment": "ag3s-verifier T43 V10. Numbers and paths only; tables in outputs/verify/T43/V10/{agg,ct,extra,hold_join,hold_cause}.json.", "step": "T43V10", "date": "2026-10-07",
       "runs": [{"id": "V10", "command": "outputs/verify/T43/V10/p/launch_v10.sh (2 chains, ports 8286 / 8287, EST_GB 28)", "venv": ".venv-openpi-live",
                 "server_flags": "V8 flags exactly (T43/V8/p/start_server.sh: --obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 --held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both, gate ON, no --servo-model) + --held-obstacle-margin",
                 "runs_dir": "outputs/verify/T43/V10/runs/V10", "n_runs": 19, "n_done": 19, "n_retries": sum(c["n_attempt2"] for c in ex["chains"].values()), "chain_rc": ex["chain_rc"], "chain_rc_lines": ex["chain_rc_lines"],
                 "refusals": sum(len(c["refused"]) for c in ex["chains"].values()), "failed_dirs": ex["failed_dirs"],
                 "raw_output": "outputs/verify/T43/V10/runs/V10/chain_8286.log, chain_8287.log, runs/chainE.out, chainF.out, launch.out", "servers": srv, "chains": ex["chains"], "load_samples": "outputs/verify/T43/V10/raw/load_samples.txt",
                 "server_log_checks": {"gate_on": {k: v["log_gate_on"] for k, v in srv.items()}, "n_constraint_spheres_in_records": {k: v["n_constraint_spheres_in_records"] for k, v in srv.items()},
                                       "held_obstacle_margin_on_line": {k: v["log_held_obstacle_margin_on_line"] for k, v in srv.items()}, "cmdline_has_servo_model": {k: v["cmdline_has_servo_model"] for k, v in srv.items()},
                                       "autotune_off": {k: v["log_autotune_off"] for k, v in srv.items()}, "enforced_in": "p/run_chain.sh (REFUSED otherwise)"}}],
       "code_state": {"repo": "benchmark", "head": "308ddfd", "snapshot": "${AG3S_PROJECT_ROOT}-t43v10 (outputs/verify/T43/V10/snapshot.txt)", "code_md5_baseline": "outputs/verify/T43/V10/p/code_md5_v10.txt (406 lines, == at every chain preflight)",
                      "pi05_trees": "diff -rq vs ${AG3S_PROJECT_ROOT}-t43w3/pi05_TO_hybrid empty", "with_overrides_collision_backend": "esdf (outputs/verify/T43/V10/with_overrides_check.txt)",
                      "trajoptconfig_round_trip_held_obstacle_margin": "True, to_dict equal; default False; held_obstacle_margin without obstacle_margin > 0 -> ValueError (with_overrides_check.txt)"},
       "baseline_check": {"matched": True, "command": "esdf_rollout --esdf-backend curobo (SKILL cuRobo variant), snapshot, .venv-openpi-live", "raw": "outputs/verify/T43/V10/baseline/base_curobo.json, base_curobo.log",
                          "violated_start": "10/15", "has_target": "15/15", "frame0_clearance_before_mm": -9.171877401271193, "skill_reference": "10/15, 15/15, -9.171877"},
       "numbers": {"runs": {"n_done": 19, "H": 14, "B": 5, "n_control_lt_600": ex["n_control_lt_600"], "wall_s_median": ex["wall_s_median"]},
                   "sgp_old_rule_by_stage": a["sgp_old_rule_by_stage"], "pairs_old_rule": a["pairs_old_rule"], "H_totals_by_rule": ex["H_totals"], "H_pairs_z4_and_ct": ex["H_pairs_z4_rule"],
                   "ct_arms": ct["arms"], "ct_pairs": ct["pairs"], "ct_crosscheck_counts": a["ct"]["crosscheck_counts"], "per_pair": per,
                   "finger_hurdle_contacts_by_phase_MJCF_names": a["finger_contacts"], "ct_contact_table": ct["contact_table"], "obstacle_stop": a["obstacle_stop"],
                   "left_finger_name_note": "MJCF ee_finger_l2 (geom 1086) = URDF ee_finger_l1; MJCF ee_finger_l1 (geom 1079) = URDF ee_finger_l2. ct / finger_hurdle_contacts use MJCF names; hold binding_link / hold_join binding_link use URDF names (server record).",
                   "hold": hold, "hold_only_in_one_arm": dict(a["hold_pairs"]), "hold_cause_join": {"summary": hj["summary"], "chunks": hj["chunks"], "held_row_binding_chunks": hj["held_row_binding_chunks"]},
                   "hm_records": hm, "follow_to": ft, "timing": tm, "verdict_kind_counts": {"V10": ex["verdict_kind_counts_V10"], "V8": ex["verdict_kind_counts_V8"]}, "hurdle_min_mm_H_per_run": a["hurdle_min_mm"]},
       "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43v10-hurdle-scene.png", "kind": "scene"}, {"path": "AG3S/ag3s/docs/figures/t43/t43v10-graphs.png", "kind": "graph"},
                   {"path": "AG3S/ag3s/docs/figures/t43/t43v10-pairs-table.png", "kind": "table"}],
       "scripts": ["p/launch_v10.sh", "p/run_chain.sh", "p/start_server.sh", "p/run_client.sh", "p/analyze_v10.py", "p/run_replays_v10.sh", "p/agg_ct_v10.py", "p/z4_base_v10.py", "p/agg_v10.py", "p/extra_v10.py", "p/hm_survey_v10.py (not run)",
                   "p/hold_cause_v10.py", "p/hold_join_v10.py", "p/fig_v10_scene.py", "p/fig_v10_graphs.py", "p/build_verify_v10.py", "p/with_overrides_check.py"],
       "raw": {"agg": "outputs/verify/T43/V10/agg.json", "ct": "outputs/verify/T43/V10/ct.json", "extra": "outputs/verify/T43/V10/extra.json", "hold_cause": "outputs/verify/T43/V10/hold_cause.json", "hold_join": "outputs/verify/T43/V10/hold_join.json",
               "analysis": "outputs/verify/T43/V10/analysis/analysis.json", "replay": "outputs/verify/T43/V10/replay", "snapshot": "outputs/verify/T43/V10/snapshot.txt", "V8_analysis": "outputs/verify/T43/V8/analysis/analysis.json", "V8_replay": "outputs/verify/T43/V8/replay"},
       "verdict": "19/19 runs done, 0 retries, chain rc 0, guards matched, baseline matched (facts only)",
       "not_measured": ["servo OFF in both arms; V10 and V8 come from different server sessions; first-chunk JIT time not cited; SQP iteration counts not cited (not a baseline)",
                        "Z4 / CT rules, hurdle contacts and hold_cause are H-stage replays; B runs have no replay (hold_cause covers B HOLD chunks from the server record only)",
                        "nearest-surface identity = GT geom group of the field surface point p - d*grad/|grad| at the planning-row observation state, within 20 mm else unidentified; self_cap uses attach fit spheres only (runtime volume also has the pre-grasp place)",
                        "pod load1 median during V10 %.1f vs V8 %.1f (other sessions included); TO timing is wall-clock budget sensitive" % (ex["loadavg_during_V10"]["load1_median"], ld8["load1_median"])]}
p = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V10.verify.json')); assert not p.exists(); p.write_text(json.dumps(out, indent=1, default=float, ensure_ascii=False)); print(p, len(p.read_text()))
print(tm["max_iterations_hit"]); print(hm)
