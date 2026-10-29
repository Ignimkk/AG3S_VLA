"""T43 OC aggregate: raw/oc_{V7g,V8a,V8b}.json + TK contacts -> raw/oc_summary.json (numbers only).
Per contact (exec chunk = control row t chunk_seq; obs = planning-row state of that chunk):
  near_px[cam] / near_vis_px[cam] / near_robot_filtered[cam] / near_held_filtered[cam] (oc_contact.py definitions); centre_bin_vis[cam]
  last_seen_seq = last chunk seq <= exec_seq with vis_near_total >= 1; chunks_since_seen = exec_seq - last_seen_seq.
  first_unseen_streak = number of consecutive chunks ending at exec_seq with vis_near_total == 0.
  in_raw = n_raw_near >= 1; in_filtered = n_filtered_near >= 1; in_full_layer = fine_pb_mm <= 5.0 (one fine voxel);
  in_target_free_layer: None when the chunk has no target-free layer, else free_pb_mm <= 5.0.
  fine_pb_max_after_place = max fine_pb_mm over chunks with t > place_t up to exec; held_chunks_bar_in_heldfree = chunks with n_S_in_heldfree > 0.
  post-place target identity: chunks t > place_t: subtask labels, manip ids, switched_from not None, has_target, n chunks with
  >= 1 target point within 20 mm of the hurdle, tf modes, n chunks pb inside ball.
  TK link check: tk_clear_row_k_link_mm (TK value, link name = MJCF body), rec_clear_k[MJCF name] and rec_clear_k[URDF link]
  (OC exec record), bar_P_mm (TK); esdf_minus_geomP_tk = rec[MJCF name] - bar_P; esdf_minus_geomP_urdf = rec[URDF] - bar_P."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
import numpy as np
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/OC/raw'))
tk = {(c["key"], c["t"]): c for c in json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43TK.verify.json')))["numbers"]["contacts"]}
evs = []
for f in ("oc_V7g.json", "oc_V8a.json", "oc_V8b.json"):
    evs += json.load(open(R / f))["events"]
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
rows = []
for e in evs:
    H = {h["seq"]: h for h in e["history"]}; sx = e["exec_seq"]; hx = H[sx]; pt = e["place_t"]; x = e["exec"]
    seen = [s for s in sorted(H) if s <= sx and H[s]["vis_near_total"] >= 1]
    streak = 0
    for s in sorted([s for s in H if s <= sx], reverse=True):
        if H[s]["vis_near_total"] == 0: streak += 1
        else: break
    post = [H[s] for s in sorted(H) if H[s]["t"] > pt and s <= sx]
    held = [h for h in e["history"] if h.get("n_S_in_heldfree") is not None]
    c = tk.get((e["key"], e["t"]), {})
    urdf = x.get("urdf_link"); body = e["mjcf_body"]
    r = {"key": e["key"], "t": e["t"], "body_mjcf": body, "link_urdf": urdf, "exec_seq": sx, "step_in_chunk": e["step_in_chunk"],
         "contact_state": e["state"], "replay_mismatch_rows": e["replay_qpos_mismatch_rows"], "mesh_pen_mm": e["finger_bar_mesh_mm"],
         "a_pb_mm": e["bar"]["a_pb_mm"],
         "near_px": {k: hx["cams"][k]["n_near"] for k in CAMS}, "near_vis_px": {k: hx["cams"][k]["n_near_vis"] for k in CAMS},
         "near_robot_filtered": {k: hx["cams"][k]["n_near_robot_filtered"] for k in CAMS},
         "near_held_filtered": {k: hx["cams"][k]["n_near_held_filtered"] for k in CAMS},
         "centre_bin_vis": {k: hx["cams"][k]["bins_vis"][2] for k in CAMS}, "bins_vis": {k: hx["cams"][k]["bins_vis"] for k in CAMS},
         "pb_occluded": {k: hx["cams"][k].get("pb_occluded") for k in CAMS}, "pb_body_at_px": {k: hx["cams"][k].get("pb_body_at_px") for k in CAMS},
         "last_seen_seq": seen[-1] if seen else None, "chunks_since_seen": (sx - seen[-1]) if seen else None, "unseen_streak": streak,
         "n_raw_near": hx["n_raw_near"], "n_filtered_near": hx["n_filtered_near"], "raw_eq_filtered": hx["raw_eq_filtered"],
         "fine_pb_mm": hx["fine_pb_mm"], "fine_S_min_mm": hx.get("fine_S_min_mm"), "coarse_pb_mm": hx["coarse_pb_mm"], "free_pb_mm": hx.get("free_pb_mm"),
         "fine_pb_inside": hx["fine_pb_inside"], "coarse_pb_inside": hx["coarse_pb_inside"],
         "composed_full_pb_mm": x["composed_full_pb_mm"], "composed_full_pf_mm": x["composed_full_pf_mm"], "composed_full_pf_tier": x["composed_full_pf_tier"],
         "n_target_free_layers": x["n_target_free_layers"], "tf_mode": hx.get("tf_mode"), "ball_radius_mm": hx.get("ball_radius_mm"),
         "pb_minus_ball_mm": hx.get("pb_minus_ball_mm"), "manip_id": hx["manip_id"], "manip_state": hx["manip_state"], "subtask": hx["subtask"],
         "grasp_state": hx["grasp_state"], "has_target": hx["has_target"], "n_target_points": hx["n_target_points"],
         "in_raw": hx["n_raw_near"] >= 1, "in_filtered": hx["n_filtered_near"] >= 1, "in_full_layer": hx["fine_pb_mm"] <= 5.0,
         "in_target_free_layer": None if not x["n_target_free_layers"] else (hx.get("free_pb_mm") is not None and hx["free_pb_mm"] <= 5.0),
         "fine_pb_max_after_place_mm": max(h["fine_pb_mm"] for h in post) if post else None,
         "fine_pb_max_all_mm": max(h["fine_pb_mm"] for h in e["history"] if h["seq"] <= sx),
         "n_post_place_chunks": len(post),
         "post_subtask": sorted({str(h["subtask"]) for h in post}), "post_manip_ids": sorted({str(h["manip_id"]) for h in post}),
         "post_switched": sum(h["manip_switched_from"] is not None for h in post), "post_blocked": sorted({str(h["blocked"]) for h in post}),
         "post_tf_modes": sorted({str(h["tf_mode"]) for h in post}), "post_n_pb_in_ball": sum((h.get("pb_minus_ball_mm") or 1) < 0 for h in post),
         "post_n_target_near_hurdle_chunks": sum((h.get("n_target_within_20mm_hurdle") or 0) > 0 for h in post),
         "held_chunks": len(held), "held_chunks_bar_in_heldfree": sum(h["n_S_in_heldfree"] > 0 for h in held),
         "heldfree_S_sd_min_mm": min((h["heldfree_S_sd_min_mm"] for h in held), default=None),
         "to_status": x.get("to_status"), "to_clearance_mm": x.get("to_clearance_mm"),
         "rec_clear_k_mjcfname_mm": x.get(f"rec_clear_k_{body}_mm"), "rec_clear_k_urdf_mm": x.get(f"rec_clear_k_{urdf}_mm"),
         "tk_clear_row_k_link_mm": c.get("clear_row_k_link_mm"), "tk_bar_P_mm": c.get("bar_P_mm"), "tk_esdf_minus_geomP_mm": c.get("esdf_link_minus_geom_P_mm")}
    if r["tk_bar_P_mm"] is not None:
        r["esdf_minus_geomP_urdf_mm"] = None if r["rec_clear_k_urdf_mm"] is None else r["rec_clear_k_urdf_mm"] - r["tk_bar_P_mm"]
    r["field_minus_mesh_at_contact_mm"] = r["composed_full_pf_mm"] - r["mesh_pen_mm"]
    rows.append(r)
def st(v):
    v = [x for x in v if x is not None]
    return {"n": len(v), "median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v))} if v else {"n": 0}
grp = {}
for r in rows:
    g = r["key"].split("/")[0] + ("|hand" if r["body_mjcf"].startswith("ee_finger") else "|arm"); grp.setdefault(g, []).append(r)
counts = {}
for g, v in grp.items():
    counts[g] = {"n": len(v),
                 "bar_vis_near_any_cam_exec": sum(sum(r["near_vis_px"].values()) > 0 for r in v),
                 "bar_vis_near_head_exec": sum(r["near_vis_px"]["zed_left"] > 0 for r in v),
                 "bar_vis_near_wrist_exec": sum((r["near_vis_px"]["wrist_cam_l"] + r["near_vis_px"]["wrist_cam_r"]) > 0 for r in v),
                 "pb_occluded_head": sum(bool(r["pb_occluded"]["zed_left"]) for r in v),
                 "chunks_since_seen_0": sum(r["chunks_since_seen"] == 0 for r in v),
                 "in_raw": sum(r["in_raw"] for r in v), "in_filtered": sum(r["in_filtered"] for r in v), "raw_eq_filtered": sum(r["raw_eq_filtered"] for r in v),
                 "in_full_layer": sum(r["in_full_layer"] for r in v), "target_free_layer_present": sum(bool(r["n_target_free_layers"]) for r in v),
                 "pb_in_ball_exec": sum((r["pb_minus_ball_mm"] or 1) < 0 for r in v), "fine_pb_inside": sum(r["fine_pb_inside"] for r in v),
                 "coarse_pb_inside": sum(r["coarse_pb_inside"] for r in v),
                 "robot_filtered_near_px_gt0": sum(sum(r["near_robot_filtered"].values()) > 0 for r in v),
                 "post_place_target_switch": sum(r["post_switched"] > 0 for r in v),
                 "post_place_pb_in_ball_any": sum(r["post_n_pb_in_ball"] > 0 for r in v),
                 "held_free_overlapped_bar_any_chunk": sum(r["held_chunks_bar_in_heldfree"] > 0 for r in v),
                 "fine_pb_max_after_place_gt5mm": sum((r["fine_pb_max_after_place_mm"] or 0) > 5 for r in v),
                 "fine_pb_mm": st([r["fine_pb_mm"] for r in v]), "composed_full_pf_mm": st([r["composed_full_pf_mm"] for r in v]),
                 "mesh_pen_mm": st([r["mesh_pen_mm"] for r in v]), "field_minus_mesh_at_contact_mm": st([r["field_minus_mesh_at_contact_mm"] for r in v]),
                 "fine_pb_max_after_place_mm": st([r["fine_pb_max_after_place_mm"] for r in v]),
                 "tk_esdf_minus_geomP_mm": st([r["tk_esdf_minus_geomP_mm"] for r in v]),
                 "esdf_minus_geomP_urdf_mm": st([r.get("esdf_minus_geomP_urdf_mm") for r in v]),
                 "tk_clear_link_equals_rec_mjcfname": sum(r["tk_clear_row_k_link_mm"] is not None and r["rec_clear_k_mjcfname_mm"] is not None
                                                          and abs(r["tk_clear_row_k_link_mm"] - r["rec_clear_k_mjcfname_mm"]) < 1e-3 for r in v)}
out = {"definitions": __doc__, "n_contacts": len(rows), "counts": counts, "contacts": rows}
(R / "oc_summary.json").write_text(json.dumps(out, indent=1, default=float))
for g, c in counts.items():
    print(g, json.dumps({k: v for k, v in c.items()}, default=float))
for r in rows:
    print(r["key"].split("/")[-1], r["t"], r["body_mjcf"], "vis", list(r["near_vis_px"].values()), "rf", list(r["near_robot_filtered"].values()), "since", r["chunks_since_seen"],
          "raw", r["n_raw_near"], "fine %.1f pf %.1f mesh %.1f" % (r["fine_pb_mm"], r["composed_full_pf_mm"], r["mesh_pen_mm"]), "tf", r["n_target_free_layers"], r["tf_mode"], r["ball_radius_mm"], r["pb_minus_ball_mm"],
          "maxAP %.1f" % (r["fine_pb_max_after_place_mm"] or -1), "hf", r["held_chunks_bar_in_heldfree"], "tk", r["tk_esdf_minus_geomP_mm"], "urdf", r.get("esdf_minus_geomP_urdf_mm"), "sw", r["post_switched"], r["post_subtask"], r["post_manip_ids"])
