"""T43 DX-2 aggregate: one row per FC (a) no-bar event + counts. usage: agg_dx2.py OUT_JSON (reads raw/dx2_t43*.json).
Count rules (numbers only; definitions of the quantities: dx2_finger.py docstring):
  points_absent_raw      : n_raw_bar_near == 0          removed_by_selffilter_cloud : raw_eq_filtered false and n_filtered_bar_near < n_raw_bar_near
  bar_px_selffiltered    : sum over obs cameras of n_bar_px_near_selffiltered > 0
  not_visible_any_cam    : sum over obs cameras of n_bar_px_near == 0
  outside_fine_grid      : pb outside the fine grid (esdf_origin / esdf_shape)      outside_coarse_grid likewise
  bar_in_full_layer      : composed full-layer distance at pb <= 5 mm (fine voxel)
  bar_absent_target_free : composed target-free distance at pb > 5 mm
  pb_inside_ball         : |pb - target-free ball centre| < ball radius
  target_is_bar          : >= 90 % of the chunk's target_points within 20 mm of the hurdle"""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, sys
import numpy as np

R = _ag3s_workspace_path('outputs/verify/T43/DX/raw/')
FC = {}
for f in glob.glob(_ag3s_workspace_path('outputs/impl/T43FC/offline/fc_*.json')):
    for k, r in json.load(open(f))["runs"].items():
        for e in r["contacts"]:
            FC[(k, e["first_t"], e["mjcf_body"])] = e
ev = []
for f in sorted(glob.glob(R + "dx2_t43*.json")):
    ev += json.load(open(f))["events"]
rows = []
for e in ev:
    es = e["esdf"]; cl = e["cloud"]; fc = FC.get((e["key"], e["t"], e["mjcf_body"])) or {}
    obs = {k.split(":")[1]: v for k, v in e["cameras"].items() if k.startswith("obs:")}
    con = {k.split(":")[1]: v for k, v in e["cameras"].items() if k.startswith("contact:")}
    tp = es.get("target_points") or {}
    rows.append({"key": e["key"], "t": e["t"], "mjcf_body": e["mjcf_body"], "urdf_link": e["urdf_link"], "replay_qpos_mismatch_rows": e["replay_qpos_mismatch_rows"],
                 "contact_state": e["state"], "finger_bar_mesh_mm": e["finger_bar_mesh_mm"], "fc_esdf_old_mm": fc.get("esdf_old_mm_min"), "fc_esdf_new_mm": fc.get("esdf_new_mm_min"),
                 "fc_mask_new": sorted({x.get("mask_new") for x in fc.get("rows", []) if "mask_new" in x}), "exec_seq": e["exec_seq"], "fc_seq": e["fc_seq"],
                 "exec_t_step": e["exec_t_step"], "k": e["step_in_chunk"], "grasp_state": es["grasp_state"], "subtask": es.get("subtask"),
                 "exclusion_source": (es.get("exclusion") or {}).get("source"), "exclusion_mechanism": (es.get("exclusion") or {}).get("mechanism"),
                 "manipulated_id": es["manipulated"]["id"], "manipulated_state": es["manipulated"]["state"], "manipulated_extents_mm": es["manipulated"]["extents_mm"],
                 "n_raw": cl["n_raw"], "raw_eq_filtered": cl["raw_eq_filtered"], "n_raw_r10": cl["n_raw_r10"], "n_raw_r30": cl["n_raw_r30"], "n_raw_bar_near": cl["n_raw_bar_near"],
                 "n_filtered_bar_near": cl["n_filtered_bar_near"], "n_raw_bar_all": cl["n_raw_bar_all"], "min_dist_to_pb_raw_mm": cl["min_dist_to_pb_raw_mm"], "n_support_r30": cl["n_support_r30"],
                 "cam_obs": {c: {k: v.get(k) for k in ("in_image", "occluded", "body_at_px", "n_bar_px_near", "n_bar_px_near_selffiltered", "n_masked_px_repro", "n_masked_px_record")} for c, v in obs.items()},
                 "cam_contact": {c: {k: v.get(k) for k in ("in_image", "occluded", "body_at_px", "n_bar_px_near")} for c, v in con.items()},
                 "fine_voxel_m": es["fine"]["voxel_m"], "coarse_voxel_m": es["coarse"]["voxel_m"], "tsdf_truncation": es["tsdf_truncation"],
                 "pb_inside_fine": es["fine"]["pb_inside"], "pb_inside_coarse": es["coarse"]["pb_inside"], "pb_inside_free": (es.get("free") or {}).get("pb_inside"),
                 "pb_fine_index": es["fine"]["pb_index"], "fine_origin_m": es["fine"]["origin_m"], "fine_upper_m": es["fine"]["upper_m"],
                 "d_full_fine_pb_mm": es["fine"]["d_pb_mm"], "d_full_coarse_pb_mm": es["coarse"]["d_pb_mm"], "d_free_fine_pb_mm": (es.get("free") or {}).get("d_pb_mm"),
                 "composed_full_pb_mm": es["composed_full_pb_mm"], "composed_full_pb_tier": es["composed_full_pb_tier"],
                 "composed_free_pb_mm": es.get("composed_free_pb_mm"), "composed_free_pb_tier": es.get("composed_free_pb_tier"),
                 "composed_full_pf_mm": es["composed_full_pf_mm"], "composed_free_pf_mm": es.get("composed_free_pf_mm"),
                 "tf_mode": (es.get("target_free") or {}).get("mode"), "tf_label": (es.get("target_free") or {}).get("label"),
                 "n_ball_seeds_excluded": (es.get("target_free") or {}).get("n_ball_seeds_excluded"),
                 "ball_radius_mm": (es.get("ball") or {}).get("radius_mm"), "ball_fit_reason": (es.get("ball") or {}).get("fit_reason"),
                 "pb_dist_minus_ball_radius_mm": (es.get("ball") or {}).get("pb_dist_minus_radius_mm"),
                 "n_target_points": tp.get("n"), "n_target_within_20mm_hurdle": tp.get("n_within_20mm_of_hurdle"), "target_min_hurdle_sd_mm": tp.get("min_hurdle_sd_mm"),
                 "target_extent_mm": tp.get("extent_mm"),
                 "history_first_seq_target_is_free_ball_with_pb": next((h["seq"] for h in e["history"] if h.get("pb_in_ball")), None),
                 "history_n_chunks_pb_in_ball": sum(1 for h in e["history"] if h.get("pb_in_ball")), "history_n_chunks": len(e["history"])})
n = len(rows); cnt = lambda f: int(sum(1 for x in rows if f(x)))
counts = {"points_absent_raw": cnt(lambda x: x["n_raw_bar_near"] == 0),
          "removed_by_selffilter_cloud": cnt(lambda x: (not x["raw_eq_filtered"]) and x["n_filtered_bar_near"] < x["n_raw_bar_near"]),
          "bar_px_selffiltered": cnt(lambda x: sum(v["n_bar_px_near_selffiltered"] for v in x["cam_obs"].values()) > 0),
          "not_visible_any_cam": cnt(lambda x: sum(v["n_bar_px_near"] for v in x["cam_obs"].values()) == 0),
          "head_pb_in_image": cnt(lambda x: x["cam_obs"]["zed_left"]["in_image"]), "head_pb_occluded": cnt(lambda x: bool(x["cam_obs"]["zed_left"].get("occluded"))),
          "wrist_l_bar_px_near_gt0": cnt(lambda x: x["cam_obs"]["wrist_cam_l"]["n_bar_px_near"] > 0),
          "outside_fine_grid": cnt(lambda x: not x["pb_inside_fine"]), "outside_coarse_grid": cnt(lambda x: not x["pb_inside_coarse"]),
          "bar_in_full_layer": cnt(lambda x: x["composed_full_pb_mm"] <= 5.0), "bar_absent_target_free": cnt(lambda x: (x["composed_free_pb_mm"] or 0) > 5.0),
          "query_target_free_layer_fc": cnt(lambda x: x["fc_mask_new"] == [True]), "pb_inside_ball": cnt(lambda x: (x["pb_dist_minus_ball_radius_mm"] or 1) < 0),
          "target_is_bar": cnt(lambda x: x["n_target_points"] and x["n_target_within_20mm_hurdle"] >= 0.9 * x["n_target_points"]),
          "exec_eq_fc_chunk": cnt(lambda x: x["exec_seq"] == x["fc_seq"]), "replay_exact": cnt(lambda x: x["replay_qpos_mismatch_rows"] == 0)}
sf = [(v["n_masked_px_repro"], v["n_masked_px_record"]) for x in rows for v in x["cam_obs"].values() if v.get("n_masked_px_record") is not None]
out = {"definitions": __doc__, "n_events": n, "rows": rows, "counts": counts,
       "selffilter_repro_fidelity": {"n_cam_frames": len(sf), "max_abs_px_diff": max(abs(a - b) for a, b in sf), "max_rel_diff": max(abs(a - b) / max(b, 1) for a, b in sf)}}
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ("n_events", "counts", "selffilter_repro_fidelity")}, indent=1))
for x in rows:
    print(x["key"].split("/")[2], x["t"], "mesh %.2f" % x["finger_bar_mesh_mm"], "fc %.1f" % x["fc_esdf_new_mm"], x["grasp_state"], x["exclusion_source"], "manip", x["manipulated_id"], x["manipulated_extents_mm"],
          "raw near", x["n_raw_bar_near"], "r30", x["n_raw_r30"], "head", x["cam_obs"]["zed_left"]["in_image"], x["cam_obs"]["zed_left"]["occluded"], x["cam_obs"]["zed_left"]["body_at_px"],
          "px", {c: v["n_bar_px_near"] for c, v in x["cam_obs"].items()}, "sf", {c: v["n_bar_px_near_selffiltered"] for c, v in x["cam_obs"].items()},
          "full %.1f(%s) coarse %.1f free %.1f" % (x["composed_full_pb_mm"], x["composed_full_pb_tier"], x["d_full_coarse_pb_mm"], x["composed_free_pb_mm"]),
          "ball r %.0f in by %.1f" % (x["ball_radius_mm"], -x["pb_dist_minus_ball_radius_mm"]), "tp %d/%d" % (x["n_target_within_20mm_hurdle"], x["n_target_points"]), x["target_extent_mm"],
          "hist", x["history_n_chunks_pb_in_ball"], "/", x["history_n_chunks"], "first", x["history_first_seq_target_is_free_ball_with_pb"], "mism", x["replay_qpos_mismatch_rows"])
