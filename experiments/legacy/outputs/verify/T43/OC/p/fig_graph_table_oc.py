"""T43 OC graph + table figures from raw/oc_{V7g,V8a,V8b}.json, raw/oc_summary.json, raw/m09_V8_18073_seq22.json (system python ok).
t43oc-history.png: (a) recorded full-layer fine ESDF at pb per chunk, x = chunk seq - exec seq (all 16 contacts; V7g slot 0, V8 slot 1;
  dashed = arm contacts); (b) bar pixels near pb that reach the TSDF (kept by the self-filter), all cameras, per chunk (symlog);
  (c) per contact: TK esdf_link_minus_geom_P (link name = MJCF body name) vs the same with the URDF link of that MJCF body.
t43oc-table.png: per-contact table (exec chunk) + the V8_ep1807_s18073 seq 22 (-0.9 mm) decomposition."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/OC/raw')); FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
OH, OT = FIG / "t43oc-history.png", FIG / "t43oc-table.png"
for p in (OH, OT):
    assert not p.exists() and not p.with_suffix(".json").exists(), p
figstyle.use_korean(); K = figstyle.CATEGORICAL
evs = []
for f in ("oc_V7g.json", "oc_V8a.json", "oc_V8b.json"):
    evs += json.load(open(R / f))["events"]
S = json.load(open(R / "oc_summary.json")); M = json.load(open(R / "m09_V8_18073_seq22.json"))
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6)); side = {"script": __file__, "definitions": __doc__, "series": []}
for e in evs:
    arm = e["key"].split("/")[0]; col = K[0] if arm == "V7g" else K[1]; ls = "-" if e["mjcf_body"].startswith("ee_finger") else "--"
    H = e["history"]; x = [h["seq"] - e["exec_seq"] for h in H]
    ax[0].plot(x, [h["fine_pb_mm"] for h in H], ls, color=col, lw=0.9, alpha=0.8)
    ax[1].plot(x, [h["vis_near_total"] for h in H], ls, color=col, lw=0.9, alpha=0.8)
    pl = [h["seq"] - e["exec_seq"] for h in H if h["t"] > e["place_t"]]
    side["series"].append({"key": e["key"], "t": e["t"], "x": x, "fine_pb_mm": [h["fine_pb_mm"] for h in H], "vis_near_total": [h["vis_near_total"] for h in H],
                           "first_post_place_x": pl[0] if pl else None})
ax[0].axhline(5, color=figstyle.INK_2, lw=0.8, ls=":"); ax[0].text(-38, 6, "5 mm = one fine voxel", fontsize=7, color=figstyle.INK_2)
ax[0].set_xlabel("chunk seq - executing chunk seq"); ax[0].set_ylabel("fine ESDF at pb (mm)"); ax[0].set_title("(a) bar contact point in the full fine layer, per chunk", fontsize=9)
ax[1].set_yscale("symlog", linthresh=1); ax[1].set_xlabel("chunk seq - executing chunk seq"); ax[1].set_ylabel("bar px within 30 mm of pb, kept (3 cameras)")
ax[1].set_title("(b) bar pixels near pb that reach the TSDF", fontsize=9)
for a_, lab, c in ((ax[0], "V7g", K[0]), (ax[0], "V8", K[1])):
    a_.plot([], [], color=c, label=lab)
ax[0].plot([], [], "--", color=figstyle.INK_2, label="arm contact (link_right_arm_4)"); ax[0].legend(fontsize=7)
C_ = [c for c in S["contacts"] if c.get("tk_esdf_minus_geomP_mm") is not None]
lab = [f"{c['key'].split('/')[0]} {c['key'].split('_')[-1]} t{c['t']}" for c in C_]; y = np.arange(len(C_))
ax[2].barh(y - 0.2, [c["tk_esdf_minus_geomP_mm"] for c in C_], 0.4, color=K[3], label="TK: spheres of link named like the MJCF body")
ax[2].barh(y + 0.2, [c["esdf_minus_geomP_urdf_mm"] for c in C_], 0.4, color=K[2], label="spheres of the URDF link of that MJCF body")
ax[2].set_yticks(y); ax[2].set_yticklabels(lab, fontsize=6); ax[2].axvline(0, color=figstyle.INK_2, lw=0.8)
ax[2].set_xlabel("record clearance at row k minus geometric bar distance at P (mm)"); ax[2].set_title("(c) TK esdf_link_minus_geom_P, link name check (14 hand contacts)", fontsize=9)
ax[2].legend(fontsize=6, loc="lower right")
side["panel_c"] = [{"label": l, "tk_mm": c["tk_esdf_minus_geomP_mm"], "urdf_mm": c["esdf_minus_geomP_urdf_mm"], "tk_bar_P_mm": c["tk_bar_P_mm"],
                    "rec_clear_k_mjcfname_mm": c["rec_clear_k_mjcfname_mm"], "rec_clear_k_urdf_mm": c["rec_clear_k_urdf_mm"]} for l, c in zip(lab, C_)]
figstyle.style_axes(fig, ax); fig.suptitle("T43 OC: post-release bar contacts, chunk history (V7g 6, V8 10)", fontsize=10); fig.tight_layout(); fig.savefig(OH, dpi=130)
OH.with_suffix(".json").write_text(json.dumps(side, indent=1, default=float)); print(OH)
# table
cols = ["run", "t", "body (MJCF)", "kept px near pb\nhead / wristL / wristR", "chunks\nsince seen", "raw / filt\npts near pb", "fine d(pb)\nmm",
        "field at pf\nvs mesh mm", "target-free\nlayer", "ball R / pb-R\nmm", "robot-filt px\nnear pb", "held-free over bar\n(held chunks)", "post-place\ntarget switch"]
rows = []
for c in S["contacts"]:
    nv = c["near_vis_px"]; rf = c["near_robot_filtered"]
    rows.append([c["key"].split("/")[-1], c["t"], c["body_mjcf"], f"{nv['zed_left']} / {nv['wrist_cam_l']} / {nv['wrist_cam_r']}", c["chunks_since_seen"],
                 f"{c['n_raw_near']} / {c['n_filtered_near']}", f"{c['fine_pb_mm']:.1f}", f"{c['composed_full_pf_mm']:.1f} / {c['mesh_pen_mm']:.1f}",
                 "none" if not c["n_target_free_layers"] else f"{c['tf_mode']}", "-" if c["ball_radius_mm"] is None else f"{c['ball_radius_mm']:.0f} / {c['pb_minus_ball_mm']:.0f}",
                 sum(rf.values()), f"{c['held_chunks_bar_in_heldfree']} ({c['held_chunks']})", c["post_switched"]])
m = M
rows2 = [["TO clearance (to.metrics)", f"{m['to_clearance_mm']:.2f} mm"], ["worst row", f"step 0, query {m['query']}, URDF {m['link']} (= MJCF ee_finger_l2), r {m['radius_mm']:.2f} mm"],
         ["d full fine at sphere centre", f"{m['composed_d_mm']:.2f} mm (record clearance d-r {m['rec_clearance_step0_mm']:.2f} mm)"],
         ["implied margin d - r - clearance", f"{m['implied_margin_mm']:.2f} mm (obstacle_margin_class 'target')"],
         ["d_obj (held spheres, record)", f"{m['d_obj_record_mm']:.2f} mm; |d - d_obj| = {m['abs_drow_minus_dobj_mm']:.2f} mm <= voxel {m['voxel_mm']:.0f} mm"],
         ["geometry at obs state", f"sphere surface to bar {m['hurdle_sd_minus_r_mm']:.1f} mm; MJCF finger to apple {m['mjcf_ee_finger_l2_to_apple_mm_obs']:.2f} mm; apple to bar {m['apple_to_hurdle_mm_obs']:.1f} mm"],
         ["classification / gate", "allowed_contact (tier fine, evidence distance) + budget_only -> execute; tolerance 1e-4 m"]]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(19, 8.6), gridspec_kw={"height_ratios": [3.3, 1.4]})
for a_ in (a1, a2):
    a_.axis("off")
tb = a1.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 1.45)
for (r, c), cell in tb.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if r == 0:
        cell.set_facecolor(figstyle.SEQ_BLUE[0]); cell.set_height(cell.get_height() * 1.8)
a1.set_title("Per post-release contact (executing chunk, observation state). kept = bar px within 30 mm of pb not removed by the self-filter", fontsize=9)
tb2 = a2.table(cellText=rows2, colLabels=["V8_ep1807_s18073 chunk 22 (rows 168-175)", "value"], loc="center", cellLoc="left", colWidths=[0.25, 0.7])
tb2.auto_set_font_size(False); tb2.set_fontsize(7.5); tb2.scale(1, 1.35)
for (r, c), cell in tb2.get_celld().items():
    cell.set_edgecolor(figstyle.GRID_INK)
    if r == 0:
        cell.set_facecolor(figstyle.SEQ_BLUE[0])
fig.patch.set_facecolor(figstyle.SURFACE); fig.tight_layout(); fig.savefig(OT, dpi=130)
OT.with_suffix(".json").write_text(json.dumps({"script": __file__, "columns": cols, "rows": rows, "m09_rows": rows2, "counts": S["counts"]}, indent=1, default=float)); print(OT)
