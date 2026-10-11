"""T35-diag A3 — grasp settle and hold metrics per run (a/a_runs.json).
t_settle = last control row of the FIRST close cycle (applied left gripper cmd < 0.5); hold window = the next 24 control rows.
At t_settle: open_meas, d_l1/d_l2 (GT finger-apple, mm), contact midpoint (witness mid) - apple centre (world mm),
  body midpoint (finger body origins) - apple centre for measured / reference-commanded / refined-commanded arm pose,
  and differences ref - measured, refined - ref (mm).
Hold window: cmd median, min open_meas and open drop from t_settle, first row with min(d_l1, d_l2) > 1 mm ('contact_lost_t'),
  apple dz max (mm) and dz at contact loss, witness-mid z change (max - value at settle, mm), apple xy displacement max.
usage: settle.py -> a/settle.json"""
import json, numpy as np
R = json.load(open("a/a_runs.json"))["runs"]
out = {"definitions": __doc__, "runs": {}}
for n, r in R.items():
    tl = r["timeline"]
    i0 = next((i for i, e in enumerate(tl) if e["grip_cmd"] < 0.5), None)
    if i0 is None:
        continue
    j = i0
    while j + 1 < len(tl) and tl[j + 1]["grip_cmd"] < 0.5:
        j += 1
    e = tl[j]; w = tl[j + 1:j + 25]
    lost = next((x for x in w if min(x["d_l1_mm"], x["d_l2_mm"]) > 1.0), None)
    bm, bmr, bmf = map(np.array, (e["body_mid_rel_apple_mm"], e["body_mid_rel_apple_ref_mm"], e["body_mid_rel_apple_refined_mm"]))
    out["runs"][n] = {"success": r["success"], "t_close": r["t_close"], "t_settle": e["t"], "n_close_rows": j - i0 + 1,
                      "open_settle": e["open_meas"], "d_l1_settle_mm": e["d_l1_mm"], "d_l2_settle_mm": e["d_l2_mm"],
                      "contact_mid_rel_apple_mm": e["mid_rel_apple_mm"],
                      "body_mid_rel_apple_mm": bm.tolist(), "body_mid_rel_apple_ref_mm": bmr.tolist(), "body_mid_rel_apple_refined_mm": bmf.tolist(),
                      "ref_minus_measured_body_mid_mm": (bmr - bm).tolist(), "ref_minus_measured_body_mid_norm_mm": float(np.linalg.norm(bmr - bm)),
                      "refined_minus_ref_body_mid_mm": (bmf - bmr).tolist(), "refined_minus_ref_body_mid_norm_mm": float(np.linalg.norm(bmf - bmr)),
                      "hold_cmd_median": float(np.median([x["grip_cmd"] for x in w])) if w else None,
                      "hold_min_open": min(x["open_meas"] for x in w) if w else None,
                      "hold_open_drop": (e["open_meas"] - min(x["open_meas"] for x in w)) if w else None,
                      "contact_lost_t": lost["t"] if lost else None, "dz_at_contact_lost_mm": lost["dz_mm"] if lost else None,
                      "hold_dz_max_mm": max(x["dz_mm"] for x in w) if w else None,
                      "hold_mid_z_rise_mm": (max(x["mid_rel_apple_mm"][2] for x in w if min(x["d_l1_mm"], x["d_l2_mm"]) <= 1.0) - e["mid_rel_apple_mm"][2])
                      if any(min(x["d_l1_mm"], x["d_l2_mm"]) <= 1.0 for x in w) else None,
                      "hold_xy_max_mm": max(x["disp_xy_mm"] for x in w) if w else None}
json.dump(out, open("a/settle.json", "w"), indent=1)
print("%-24s %s %4s %4s %5s %6s %6s %-22s %-22s %6s %6s %5s %5s %5s %6s %6s" % ("run", "S", "tcl", "tst", "open", "l1", "l2", "contact_mid", "body_ref-meas", "|r-m|", "|f-r|", "hcmd", "hmin", "lost", "dzmax", "zrise"))
for n, v in out["runs"].items():
    print("%-24s %s %4d %4d %5.2f %6.1f %6.1f %-22s %-22s %6.1f %6.1f %5.2f %5.2f %5s %6.1f %6s" % (n, "S" if v["success"] else "F", v["t_close"], v["t_settle"], v["open_settle"], v["d_l1_settle_mm"], v["d_l2_settle_mm"],
          str([round(x, 1) for x in v["contact_mid_rel_apple_mm"]]), str([round(x, 1) for x in v["ref_minus_measured_body_mid_mm"]]), v["ref_minus_measured_body_mid_norm_mm"], v["refined_minus_ref_body_mid_norm_mm"],
          v["hold_cmd_median"], v["hold_min_open"], v["contact_lost_t"], v["hold_dz_max_mm"], None if v["hold_mid_z_rise_mm"] is None else round(v["hold_mid_z_rise_mm"], 1)))
