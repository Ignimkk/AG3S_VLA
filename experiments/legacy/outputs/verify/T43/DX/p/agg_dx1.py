"""T43 DX-1 aggregate: one row per CT apple-hurdle carry event (value at t* = row of the deepest CT contact) + counts.
usage: agg_dx1.py OUT_JSON   (reads raw/dx1_t43*.json)
Definitions of the per-row quantities: dx1_apple.py docstring. Count rules (numbers only):
  not_covering            : contact_pt_sd_mm > 0 (apple contact point outside the union of the held query spheres, actual pose)
  slip_gt_pad             : slip_mm > body_cover slip pad 5 mm
  held_obstacle_margin    : rows whose held query rows get obstacle_margin — 0 by linearize._obstacle_rows (class 'held' for
                            every query >= n_spheres); recorded check = worst-row class of chunks whose worst row is held.
  bar_absent_esdf         : composed field at the bar surface point > voxel of the answering tier (fine 5 mm, coarse 20 mm)
  field_sees_overlap      : min over held spheres (actual pose) of composed d - r < 0
  to_held_row_k_ge0       : recomputed TO held-row clearance at the contact row's refined row k >= 0
  held_planned_true_gt0   : held spheres at the planned (TO) pose, true hurdle distance - r > 0
  apple_planned_true_gt0  : apple mesh moved with the hand to the planned pose, true hurdle distance > 0
Tracking baseline: ee_plan_vs_actual_post_mm of every carry-chunk row of these runs (rows of contact events excluded)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, sys
import numpy as np

R = _ag3s_workspace_path('outputs/verify/T43/DX/raw/')
ev, runs = [], {}
for f in sorted(glob.glob(R + "dx1_t43*.json")):
    d = json.load(open(f)); ev += d["events"]; runs.update(d["runs"])
rows = []
for e in ev:
    r = next(x for x in e["rows"] if x["t"] == e["t_star"]); c = r["chunk"]; es = r["esdf"]
    tier = es["bar_surface"]["tier"]; vox = 5.0 if tier == "fine" else 20.0
    rows.append({"key": e["key"], "first_t": e["first_t"], "last_t": e["last_t"], "t_star": e["t_star"], "ct_max_pen_mm": e["ct_max_pen_mm"],
                 "ct_min_dist_t_star_mm": r["ct_min_dist_mm"], "state": r["state"], "apple_true_bar_mm": r["apple_true_bar_mm"],
                 "attach_t": runs[e["key"]]["attach_t"], "slip_mm": r.get("slip_mm"), "slip_deg": r.get("slip_deg"),
                 "held_set": r.get("held_set"), "near_support": r.get("near_support"), "n_held": r.get("n_held"), "held_radii_mm": r.get("held_radii_mm"),
                 "cover_frac": r.get("cover_frac"), "cover_max_out_mm": r.get("cover_max_out_mm"), "contact_pt_sd_mm": r.get("contact_pt_sd_mm"),
                 "held_true_bar_mm": r.get("held_true_bar_mm"), "held_planned_true_bar_mm": r.get("held_planned_true_bar_mm"),
                 "apple_planned_bar_mm": r.get("apple_planned_bar_mm"), "ee_plan_vs_actual_mm": r.get("ee_plan_vs_actual_mm"),
                 "ee_actual_minus_plan_world_mm": r.get("ee_actual_minus_plan_world_mm"), "arm_q_actual_minus_cmd_deg": r.get("arm_q_actual_minus_cmd_deg"),
                 "chunk_seq": r["chunk_seq"], "chunk_t": c["t"], "k": r["step_in_chunk"], "executed": r["executed"],
                 "to_held_row_k_mm": r.get("held_clear_row_k_mm"), "to_held_min_mm": r.get("held_clear_min_mm"),
                 "rec_to_clearance_mm": c["to_clearance_mm"], "rec_worst_link": c["worst"]["link"], "rec_worst_class": c["worst"]["class"],
                 "rec_worst_clearance_mm": c["worst"]["clearance_mm"], "rec_worst_step": c["worst"]["step"],
                 "to_status": c["to_status"], "geometry_certified": c["geometry_certified"], "verdict": c["verdict"], "n_executed": c["n_executed"], "n_rows": c["n_rows"],
                 "label_names": c["label_names"], "target_free_layers": c["target_free_layers"],
                 "esdf_bar_surface_composed_mm": es["bar_surface"]["composed_mm"], "esdf_bar_surface_tier": tier,
                 "esdf_bar_surface_fine_mm": es["bar_surface"]["fine_mm"], "esdf_bar_surface_coarse_mm": es["bar_surface"]["coarse_mm"],
                 "esdf_bar_axis_fine_mm": es["bar_axis"]["fine_mm"], "esdf_bar_axis_coarse_mm": es["bar_axis"]["coarse_mm"],
                 "esdf_apple_contact_composed_mm": es["apple_contact"]["composed_mm"],
                 "held_nearest_bar_d_minus_r_composed_mm": (r.get("held_actual_esdf") or {}).get("nearest_bar_d_minus_r_composed_mm"),
                 "held_min_d_minus_r_composed_mm": None if not r.get("held_actual_esdf") else min(r["held_actual_esdf"]["d_minus_r_composed_mm"]),
                 "bar_absent_esdf": bool(es["bar_surface"]["composed_mm"] > vox), "field_age_ms": r["field_age_ms"],
                 "esdf_margin_mm": runs[e["key"]]["esdf_margin_mm"], "obstacle_margin_mm": runs[e["key"]]["obstacle_margin_mm_log"]})
n = len(rows)
cnt = lambda f: int(sum(1 for x in rows if f(x)))
fid = [f for r_ in runs.values() for f in r_["fidelity"]]
base = []
contact_rows = {(e["key"], x["t"]) for e in ev for x in e["rows"]}
for k, r_ in runs.items():
    for c in r_["carry_chunks"]:
        for t, kk, v in c.get("ee_plan_vs_actual_post_mm", []):
            if (k, t) not in contact_rows:
                base.append(v)
cov_imm = {}
for k, r_ in runs.items():
    if r_.get("attach_cover"):
        cov_imm[k] = r_["attach_cover"]
q = lambda a, p: float(np.percentile(a, p)) if len(a) else None
out = {"definitions": __doc__, "n_events": n, "rows": rows,
       "counts": {"not_covering": cnt(lambda x: (x["contact_pt_sd_mm"] or -1) > 0), "slip_gt_pad_5mm": cnt(lambda x: (x["slip_mm"] or 0) > 5.0),
                  "held_obstacle_margin_rule": 0,
                  "worst_row_held_class_held": cnt(lambda x: str(x["rec_worst_link"]).startswith("attached:") and x["rec_worst_class"] == "held"),
                  "worst_row_is_held": cnt(lambda x: str(x["rec_worst_link"]).startswith("attached:")),
                  "bar_absent_esdf": cnt(lambda x: x["bar_absent_esdf"]), "field_sees_overlap_actual": cnt(lambda x: (x["held_min_d_minus_r_composed_mm"] or 1) < 0),
                  "to_held_row_k_ge0": cnt(lambda x: (x["to_held_row_k_mm"] or -1) >= 0), "to_held_min_ge0": cnt(lambda x: (x["to_held_min_mm"] or -1) >= 0),
                  "held_planned_true_gt0": cnt(lambda x: (x["held_planned_true_bar_mm"] or -1) > 0), "apple_planned_true_gt0": cnt(lambda x: (x["apple_planned_bar_mm"] or -1) > 0),
                  "held_actual_true_lt0": cnt(lambda x: (x["held_true_bar_mm"] or 1) < 0),
                  "to_feasible": cnt(lambda x: x["to_status"] == "feasible"), "certified": cnt(lambda x: x["geometry_certified"] is True),
                  "verdict_execute_only": cnt(lambda x: all(a == "execute" for _, a in x["verdict"])), "chunk_all_rows_executed": cnt(lambda x: x["n_executed"] == x["n_rows"]),
                  "row_executed": cnt(lambda x: x["executed"]), "held_set_lifted": cnt(lambda x: x["held_set"] == "lifted_parent_frame"),
                  "near_support_true": cnt(lambda x: x["near_support"] is True)},
       "ranges": {k: [min(x[k] for x in rows if x[k] is not None), float(np.median([x[k] for x in rows if x[k] is not None])), max(x[k] for x in rows if x[k] is not None)]
                  for k in ("ct_max_pen_mm", "slip_mm", "slip_deg", "cover_frac", "contact_pt_sd_mm", "held_true_bar_mm", "held_planned_true_bar_mm", "apple_planned_bar_mm",
                            "ee_plan_vs_actual_mm", "to_held_row_k_mm", "to_held_min_mm", "esdf_bar_surface_composed_mm", "held_min_d_minus_r_composed_mm")},
       "tracking_baseline_carry_rows_mm": {"n": len(base), "median": q(base, 50), "p10": q(base, 10), "p90": q(base, 90), "max": max(base) if base else None},
       "fidelity_held_worst_chunks": {"n": len(fid), "max_centre_vs_point_mm": max((f["centre_vs_point_mm"] for f in fid), default=None),
                                      "implied_margin_mm_values": sorted({round(f["implied_margin_mm"], 1) for f in fid}),
                                      "n_implied_0": sum(1 for f in fid if abs(f["implied_margin_mm"]) < 0.05), "n_implied_20": sum(1 for f in fid if abs(f["implied_margin_mm"] - 20) < 0.05),
                                      "classes": sorted({f["class"] for f in fid})},
       "attach_cover": cov_imm}
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ("n_events", "counts", "ranges", "tracking_baseline_carry_rows_mm", "fidelity_held_worst_chunks")}, indent=1, default=float))
for x in rows:
    print(x["key"].split("/")[2], x["first_t"], x["t_star"], "pen %.2f" % x["ct_max_pen_mm"], "slip %.1f/%.0f" % (x["slip_mm"], x["slip_deg"]), "cov %.2f cpt %.1f" % (x["cover_frac"], x["contact_pt_sd_mm"]),
          "heldTrue act %.1f plan %.1f appPlan %.1f" % (x["held_true_bar_mm"], x["held_planned_true_bar_mm"], x["apple_planned_bar_mm"]), "dEE %.1f" % x["ee_plan_vs_actual_mm"],
          "TOk %.1f min %.1f" % (x["to_held_row_k_mm"], x["to_held_min_mm"]), "rec", x["rec_worst_link"], x["rec_worst_class"], "%.1f" % x["rec_worst_clearance_mm"],
          x["to_status"], x["geometry_certified"], "bar %.1f %s" % (x["esdf_bar_surface_composed_mm"], x["esdf_bar_surface_tier"]), "heldESDF %.1f" % x["held_min_d_minus_r_composed_mm"])
