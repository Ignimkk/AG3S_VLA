"""T43 CX — per-case summary (numbers only) from raw/cx_cases.json, raw/cx_inhand.json, raw/cx_hold.json, offline/*.json, raw/cx_reversal_split.json,
T43V11ABC.verify.json grasp_phase. usage: cx_summary.py OUT_JSON

Per case and arm:
  stage, grasp_t_used (+source), t_sep, rows_in_hand, hold_rows_in_hand;
  divergence: first seq whose refined rows 0-7 differ by > 1e-4 rad (and its t, grasp.state of that chunk in both arms);
  C rows (V11ABC server records, to.metrics.servo_accel) on the chunks with t in [grasp_t - 8, t_sep]: n chunks, n with n_over > 0, max overshoot_max_rad,
     max boundary_velocity_ratio; recomputed (cx_extract c_rows, window rows j = 0..7): total n_bind, n_over, max slack;
  hand acceleration: ABC window t_sep-24 .. t_sep+4 (cx_extract) commanded / measured max |a| (m/s^2), max |jerk| (m/s^3);
     lift window aligned on grasp_t: rows grasp_t .. grasp_t + 20 (or to t_sep if earlier), both arms, from raw/tl/tl_<arm>_<case>.npz;
  slip: slip_mm at t_sep - 8 and at t_sep - 1; grip_cmd mean over t_sep-8 .. t_sep-1; apple-hand contact sub-steps at t_sep-1;
  in-hand geometry at grasp_t (cx_inhand): finger gap, apple offset along approach axis / perpendicular;
  closing: first closing chunk outward_mm, measured tip-apple distance at its start (T43V11ABC.verify.json grasp_phase per_run first_closing)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, pathlib, sys
import numpy as np
X = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/CX'))
C = json.load(open(X / "raw/cx_cases.json"))["cases"]; IH = json.load(open(X / "raw/cx_inhand.json"))["cases"]
GP = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V11ABC.verify.json')))["numbers"]["grasp_phase"]


def lift(arm, ck, gt, ts):
    z = np.load(X / f"raw/tl/tl_{arm}_{ck.replace(':', '_')}.npz")
    e = min(gt + 20, ts) if ts is not None else gt + 20
    sl = slice(gt - 1, max(e - 1, gt))   # acc index k ~ row k+1
    jl = slice(gt - 1, max(e - 2, gt))
    return {"rows": [gt, e], "acc_cmd_max": float(z["acc_cmd"][sl].max()), "acc_meas_max": float(z["acc_meas"][sl].max()),
            "jerk_cmd_max": float(z["jerk_cmd"][jl].max()), "jerk_meas_max": float(z["jerk_meas"][jl].max())}


out = {"definitions": __doc__, "cases": {}}
for ck, c in C.items():
    E = {}
    dv = c["divergence"]; fd = dv["first_refined_gt_1e-4"]
    st = {a: next((e["grasp_state"] for e in c["arms"][a]["chunks"] if fd and e["seq"] == fd[0]), None) for a in ("V11AB", "V11ABC")}
    E["divergence"] = {"first_refined_seq_t": fd, "first_reference_seq_t": dv["first_ref_gt_1e-4"], "first_qpos_seq_t": dv["first_qpos_gt_1e-6"], "grasp_state_at_first": st}
    for arm in ("V11AB", "V11ABC"):
        A = c["arms"][arm]; tl = A["timeline"]; gt = tl.get("grasp_t_used"); ts = tl.get("t_sep")
        r = {"stage": c["fail_stage"][arm]["stage"], "grasp_t_used": gt, "grasp_t_source": tl.get("grasp_t_source"), "t_sep": ts,
             "rows_in_hand": tl.get("rows_in_hand"), "hold_rows_in_hand": tl.get("hold_rows_in_hand")}
        if gt is not None and ts is not None:
            ch = [e for e in A["chunks"] if gt - 8 <= e["t"] <= ts]
            sa = [e["servo_accel_server"] for e in ch if e.get("servo_accel_server")]
            r["c_rows_server"] = None if not sa else {"n_chunks": len(sa), "n_chunks_n_over_gt0": sum(1 for s in sa if (s.get("n_over") or 0) > 0),
                                                      "overshoot_max_rad": max(s.get("overshoot_max_rad") or 0 for s in sa),
                                                      "boundary_velocity_ratio_max": max(s.get("boundary_velocity_ratio") or 0 for s in sa),
                                                      "boundary_velocity_overshoot_max_rad": max(s.get("boundary_velocity_overshoot_rad") or 0 for s in sa)}
            r["c_rows_recomputed_window"] = {"n_chunks": len(ch), "n_bind": sum(e["c_rows"]["window"]["n_bind"] for e in ch),
                                             "n_over": sum(e["c_rows"]["window"]["n_over"] for e in ch), "n_rows": sum(e["c_rows"]["window"]["n_rows"] for e in ch),
                                             "slack_max_rad": max(e["c_rows"]["window"]["slack_max_rad"] for e in ch),
                                             "boundary_n_over": sum(e["c_rows"]["boundary"]["n_over"] for e in ch)}
            r["refine_dev_rad_max"] = max(e["refine_dev_rad"] for e in ch); r["tcp_dev_mm_max"] = max(e["tcp_dev_mm"] for e in ch)
            r["n_hold_chunks"] = sum(1 for e in ch if e["gate_action"] == "hold")
            r["lift_window"] = lift(arm, ck, gt, ts)
            w = tl.get("window") or {}
            r["drop_window"] = {k: w.get(k) for k in ("rows", "acc_cmd_max", "acc_meas_max", "jerk_cmd_max", "jerk_meas_max", "acc_rel_max")}
            z = np.load(X / f"raw/tl/tl_{arm}_{ck.replace(':', '_')}.npz")
            r["slip_mm_tsep_m8"] = float(z["slip_mm"][ts - 8 + 1]) if ts - 8 >= gt else None
            r["slip_mm_tsep_m1"] = float(z["slip_mm"][ts - 1 + 1])
            r["grip_cmd_mean_tsep_m8"] = float(z["grip_cmd"][max(ts - 8, gt):ts].mean()) if ts > gt else float(z["grip_cmd"][ts])
            r["finger_open_tsep_m1"] = float(z["finger_open"][ts]); r["finger_open_at_grasp"] = tl.get("finger_open_at_grasp")
            r["apple_hand_contact_substeps_tsep_m1"] = int(z["cnt_apple_hand"][ts - 1]); r["apple_hand_force_N_tsep_m1"] = float(z["f_apple_hand"][ts - 1])
            r["apple_hand_force_N_at_grasp"] = tl.get("apple_hand_force_at_grasp_N")
        ih = IH.get(ck, {}).get(arm) or {}
        g = ih.get("at_grasp") or {}
        r["inhand_at_grasp"] = {k: g.get(k) for k in ("gap_mm", "off_axis_mm", "off_app_mm", "off_perp_mm")}
        e2 = ih.get("at_tsep_m2") or {}
        r["inhand_at_tsep_m2"] = {k: e2.get(k) for k in ("off_app_mm", "off_perp_mm", "rot_deg", "tcp_vz", "tcp_az")}
        pr = (GP.get(arm) or {}).get("per_run", {}).get(ck) or {}
        fc = pr.get("first_closing") or {}
        r["first_closing"] = {"seq": fc.get("seq"), "t": fc.get("t"), "outward_mm": fc.get("outward_mm"), "measured_tip_apple_mm": fc.get("measured_tip_apple_mm_at_chunk_start"),
                              "n_closing_chunks_pre_release": (pr.get("pre_release") or {}).get("n_closing_chunks"),
                              "outward_mm_max_pre_release": (pr.get("pre_release") or {}).get("outward_mm_max")}
        E[arm] = r
    out["cases"][ck] = E
# HOLD (cx_hold) and offline toggle
H = json.load(open(X / "raw/cx_hold.json"))["arms"]
hold = {}
for arm, L in H.items():
    held = [x for x in L if x["kind"] == "held"]
    hold[arm] = {"n_hold": len(L), "n_held": len(held), "held_by_run": {k: sum(1 for x in held if x["key"] == k) for k in sorted({x["key"] for x in held})},
                 "held_links": {k: sum(1 for x in held if x["link"] == k) for k in sorted({x["link"] for x in held})},
                 "held_step_counts": {str(k): sum(1 for x in held if x["step"] == k) for k in sorted({x["step"] for x in held})},
                 "held_clearance_mm_range": [min(x["to_clearance_mm"] for x in held), max(x["to_clearance_mm"] for x in held)] if held else None,
                 "held_bar_gap_mm_range": [min(x["bar_gap_mm"] for x in held), max(x["bar_gap_mm"] for x in held)] if held else None,
                 "held_cloud_inside_n_range": [min(x["cloud_inside_n"] for x in held), max(x["cloud_inside_n"] for x in held)] if held else None,
                 "held_cloud_nearest_bar_total": sum(x["cloud_inside_by_nearest"]["bar"] for x in held), "held_cloud_inside_total": sum(x["cloud_inside_n"] for x in held),
                 "held_sqp_returned": {k: sum(1 for x in held if x["sqp"]["returned"] == k) for k in sorted({str(x["sqp"]["returned"]) for x in held})},
                 "held_servo_n_over_range": None if arm == "V11AB" or not held else [min(x["sqp"]["servo_accel"]["n_over"] for x in held), max(x["sqp"]["servo_accel"]["n_over"] for x in held)],
                 "held_servo_overshoot_max_rad_range": None if arm == "V11AB" or not held else [min(x["sqp"]["servo_accel"]["overshoot_max_rad"] for x in held), max(x["sqp"]["servo_accel"]["overshoot_max_rad"] for x in held)]}
out["hold"] = hold
off = {}
for f in sorted(glob.glob(str(X / "offline/*.json"))):
    R = json.load(open(f)); vs = [v for v in R["variants"] if v != "base"]
    rows = []
    for c in R["chunks"]:
        if "skipped" in c:
            continue
        b = c["variants"]["base"]; o = c["variants"][vs[0]]
        rows.append({"seq": c["seq"], "base": [b["status"], round(b["window_violation_mm"], 2), b["n_collision_rows"], b["returned"], round(b["vs_recorded_joint_deg_max"], 2)],
                     vs[0]: [o["status"], round(o["window_violation_mm"], 2), o["n_collision_rows"], o["returned"]]})
    off[pathlib.Path(f).stem] = {"variant": vs[0], "chunks": rows}
out["offline_c_toggle"] = off
out["reversal_split"] = json.load(open(X / "raw/cx_reversal_split.json"))["arms"]
for a in out["reversal_split"]:
    out["reversal_split"][a].pop("per_run", None)
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
for ck, E in out["cases"].items():
    for arm in ("V11AB", "V11ABC"):
        r = E[arm]; print(ck, arm, r["stage"], r["t_sep"], r.get("rows_in_hand"), r.get("hold_rows_in_hand"), "lift", {k: round(v, 2) for k, v in (r.get("lift_window") or {}).items() if k != "rows"},
              "C", r.get("c_rows_server"), "slip", r.get("slip_mm_tsep_m8"), r.get("slip_mm_tsep_m1"), "grip", r.get("grip_cmd_mean_tsep_m8"))
print(json.dumps(hold, indent=0)[:3000])
