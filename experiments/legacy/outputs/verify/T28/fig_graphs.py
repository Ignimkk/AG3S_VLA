"""T28 Phase 1 figures: graphs + tables (numbers from the raw jsons; sidecar json next to each png)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs
KO = fs.use_korean()
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28'))
F = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t28'))
C = fs.CATEGORICAL
E0 = json.loads((V / "E0/e0_summary.json").read_text())
P12 = json.loads((V / "P1-2/p12_records.json").read_text())
P13A = json.loads((V / "P1-3/p13a_summary.json").read_text())
P13B = json.loads((V / "P1-3/p13b_selffilter_summary.json").read_text())
P13C = json.loads((V / "P1-3/p13c_opening.json").read_text())

# ---------------- 1. E0 graph: apple dz + gripper + finger->apple
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
side = {}
order = ["E0_ep1807_r1", "E0_ep1807_r2", "E0_ep1800_r1", "E0_ep1800_r2", "E0_ep1808_r1", "E0_ep1808_r2"]
for i, k in enumerate(order):
    r = E0["runs"][k]; dt = r["detail"]
    col = C[[1807, 1800, 1808].index(r["episode_index"])]; ls = "-" if k.endswith("r1") else "--"
    t = [e["t"] for e in r["timeline_control"]]; dz = [e["apple_dz_mm"] for e in r["timeline_control"]]
    lab = f"ep{r['episode_index']} r{k[-1]}  {'success' if r['success'] else 'fail'}"
    axes[0].plot(t, dz, ls, color=col, lw=1.6, label=lab)
    axes[1].plot(dt["t"], dt["grip_l_applied"], ls, color=col, lw=1.2)
    axes[2].plot(dt["t"], np.clip(dt["finger_to_apple_surface_mm"], -5, 200), ls, color=col, lw=1.2)
    side[k] = {"success": r["success"], "grasp_t": r["grasp_t"], "place_t": r["place_t"], "apple_dz_max_mm": r["apple_dz_max_mm"],
               "t_first_close": dt["t_first_applied_grip_l_lt_0p5"], "min_finger_apple_mm": dt["min_finger_to_apple_surface_mm"]}
axes[0].axhline(50, color=fs.INK_2, lw=0.8, ls=":")
axes[0].set_ylabel("apple z - z0 (mm)" if not KO else "사과 z - z0 (mm)")
axes[1].set_ylabel("left gripper cmd (1=open)" if not KO else "왼 gripper 명령 (1=열림)")
axes[2].set_ylabel("finger->apple (mm)" if not KO else "손가락->사과 표면 (mm)")
axes[2].set_xlabel("t_step (15 Hz)")
axes[0].legend(ncol=3, fontsize=8, frameon=False)
axes[0].set_title("T28 P1-1 E0 (VLA only, --no-safe, port 8201, MuJoCo 3.11): apple z, gripper, finger distance")
fs.style_axes(fig, axes); fig.tight_layout()
fig.savefig(F / "t28-e0-timeline.png", dpi=140); plt.close(fig)
(F / "t28-e0-timeline.json").write_text(json.dumps({"source": "outputs/verify/T28/E0/e0_summary.json", "runs": side}, indent=1))

# ---------------- 2. P1-2 per-chunk band graph (T16, T14)
CAT = {"apple": C[1], "crate": C[0], "table": C[3], "other_fruit": C[4], "other": C[6], None: "#bbbbbb"}
fig, axes = plt.subplots(5, 2, figsize=(15, 11), sharex="col", gridspec_kw={"height_ratios": [0.6, 0.6, 1, 1, 1]})
side = {}
for j, run in enumerate(("t16", "t14")):
    ch = [e for e in P12["runs"][run]["chunks"] if e.get("has_local_row")]
    t = np.array([e["t_step"] for e in ch])
    ax = axes[0, j]
    for e in ch:
        ax.bar(e["t_step"], 1, width=7.5, color=(C[2] if e["executed"] else C[7]), align="edge")
        if not e["executed"]:
            ax.text(e["t_step"] + 3.7, 0.5, {"violated": "V", "uncertified": "U"}.get(e["hold_kind"], "?"), ha="center", va="center", fontsize=7, color="white")
    ax.set_yticks([]); ax.set_ylabel("exec / HOLD", rotation=0, ha="right", va="center")
    ax.set_title(f"{run.upper()} ep1807 — green executed, red HOLD (V violated, U uncertified)")
    ax = axes[1, j]
    for e in ch:
        ax.bar(e["t_step"], 0.5, bottom=0.5, width=7.5, color=CAT.get(e.get("target_nearest_body_centre") if e.get("target_nearest_body_centre") in ("apple", "crate") else ("other_fruit" if e.get("target_nearest_body_centre") else None)), align="edge")
        ax.bar(e["t_step"], 0.5, bottom=0.0, width=7.5, color=CAT.get(e.get("worst_nearest_category")), align="edge")
    ax.set_yticks([0.25, 0.75]); ax.set_yticklabels(["worst-row obj", "grounding tgt"], fontsize=8)
    ax = axes[2, j]
    ax.plot(t, [e["finger_to_apple_surface_mm"] for e in ch], "o-", color=C[1], ms=3, label="finger->apple surface (GT meshes)")
    ax.plot(t, [e["worst_row"]["clearance_m"] * 1000 for e in ch], "s-", color=C[0], ms=3, label="worst-row clearance (record)")
    ax.plot(t, [e.get("target_to_apple_centre_mm") or np.nan for e in ch], "^-", color=C[3], ms=3, label="target centroid->apple centre")
    ax.axhline(0, color=fs.INK_2, lw=0.8); ax.set_ylabel("mm"); ax.set_ylim(-45, 400); ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax = axes[3, j]
    ax.plot(t, [e["policy_grip_l_cmd_min_rows0_7"] for e in ch], "o-", color=C[6], ms=3, label="policy grip cmd min rows0-7")
    ax.plot(t, [min(e["grip_l_measured_open_frac"]) for e in ch], "s-", color=C[2], ms=3, label="measured opening min")
    ax.plot(t, [e["apple_dz_mm"] / 10 for e in ch], "-", color=C[1], lw=1, label="apple dz /10 mm")
    ax.set_ylabel("0..1"); ax.legend(fontsize=7, frameon=False)
    ax = axes[4, j]
    ax.plot(t[1:], [e["server_timing_ms"]["infer"] for e in ch][1:], "o-", color=C[0], ms=3, label="server infer ms")
    ax.plot(t[1:], [e["server_timing_ms"]["total"] for e in ch][1:], "s-", color=C[3], ms=3, label="server total ms")
    ax.plot(t[1:], [e["client_policy_infer_ms"] for e in ch][1:], "^-", color=C[6], ms=3, label="client round trip ms")
    ax.set_ylabel("ms (chunk 0 = JIT, omitted)"); ax.set_xlabel("t_step"); ax.legend(fontsize=7, frameon=False)
    s = P12["runs"][run]["summary"]
    side[run] = {k: s[k] for k in ("n_executed", "n_hold", "hold_kinds", "n_timeout", "n_stale", "grasp_attempt_t_first_policy_close_cmd_lt_0p5",
                                   "target_first_crate_t", "worst_nearest_category_counts_hold_only", "server_infer_ms", "client_policy_infer_ms")}
import matplotlib.patches as mp
axes[1, 1].legend(handles=[mp.Patch(color=CAT[k], label=k) for k in ("apple", "crate", "table", "other_fruit")], fontsize=7, ncol=4, frameon=False, loc="upper right", bbox_to_anchor=(1, 1.9))
fs.style_axes(fig, axes); fig.tight_layout()
fig.savefig(F / "t28-p12-chunk-diagnosis.png", dpi=130); plt.close(fig)
(F / "t28-p12-chunk-diagnosis.json").write_text(json.dumps({"source": "outputs/verify/T28/P1-2/p12_records.json", "summary": side}, indent=1))

# ---------------- 3. P1-3b apple retention / robot leak graph
fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col")
side = {}
for j, run in enumerate(("t14", "t17")):
    for s, col in (("inflation_0.0", C[0]), ("inflation_0.05", C[1])):
        ps = P13B[run][s]["per_seq"]
        t = [p["t"] for p in ps]
        axes[0, j].plot(t, [p["apple_passed"] for p in ps], "o-", color=col, ms=3, label=f"{s} apple passed")
        axes[1, j].plot(t, [p["robot_passed"] for p in ps], "o-", color=col, ms=3, label=f"{s} robot GT px leaked")
        side.setdefault(run, {})[s] = {k: P13B[run][s][k] for k in ("apple_px_visible_total", "apple_px_passed_total", "robot_px_passed_total", "n_frames_apple_visible_but_all_removed")}
    ps = P13B[run]["inflation_0.0"]["per_seq"]
    axes[0, j].plot([p["t"] for p in ps], [p["apple_visible"] for p in ps], ":", color=fs.INK_2, label="apple visible (3 cams)")
    axes[0, j].set_yscale("symlog", linthresh=10); axes[0, j].set_title(f"{run.upper()} ep1807: self-filter inflation 0 vs 0.05 (3 cameras summed)")
    axes[0, j].legend(fontsize=7, frameon=False); axes[1, j].legend(fontsize=7, frameon=False)
    axes[0, j].set_ylabel("apple px"); axes[1, j].set_ylabel("robot px leaked"); axes[1, j].set_xlabel("t_step")
fs.style_axes(fig, axes); fig.tight_layout()
fig.savefig(F / "t28-p13b-selffilter.png", dpi=130); plt.close(fig)
(F / "t28-p13b-selffilter.json").write_text(json.dumps({"source": "outputs/verify/T28/P1-3/p13b_selffilter_summary.json", "totals": side}, indent=1))

# ---------------- 4. Tables: E0 grid, P1-3a sphere table, P1-3b leak by body, P1-3c opening
fig, axes = plt.subplots(5, 1, figsize=(16, 17), gridspec_kw={"height_ratios": [1.0, 1.5, 0.8, 0.8, 0.9]})
for ax in axes: ax.axis("off")
def table(ax, cols, rows, title, fs_=8, widths=None):
    tb = ax.table(cellText=rows, colLabels=cols, loc="upper center", cellLoc="center", colWidths=widths)
    tb.auto_set_font_size(False); tb.set_fontsize(fs_); tb.scale(1, 1.35)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(fs.GRID_INK)
        if r == 0: cell.set_facecolor("#eeeeea")
    ax.set_title(title, fontsize=10, loc="left")
rows = []
for ep in (1807, 1800, 1808):
    for k in (1, 2):
        r = E0["runs"][f"E0_ep{ep}_r{k}"]; dt = r["detail"]
        rows.append([ep, k, "yes" if r["success"] else "no", r["grasp_t"], r["place_t"], f"{r['apple_dz_max_mm']:.1f}", f"{r['apple_dz_final_mm']:.1f}",
                     dt["t_first_applied_grip_l_lt_0p5"], f"{dt['finger_to_apple_mm_at_first_close']:.1f}", f"{dt['min_grip_l_measured']:.2f}",
                     f"{r['infer_ms']['median_excl_first']:.0f} / {r['infer_ms']['max_excl_first']:.0f}", 0])
table(axes[0], ["ep", "run", "success", "grasp t", "place t", "dz max mm", "dz final mm", "1st close t", "finger->apple @close mm", "min grip meas", "infer ms med/max", "timeout+stale"], rows,
      "P1-1 E0 success grid (1807 r1 vs r2 applied ctrl differ from step 0 by 0.0043 -> not deterministic -> N=2 per episode)")
rows = []
for n, c in P13A["configs"].items():
    fm = c["finger_l1_vs_mesh_closed"]
    rows.append([n.replace("_max", " max").replace("(default)", " default"), c["n_spheres_gripper_links_both_hands"], f"{c['n_spheres']['ee_finger_l1']} / {c['n_spheres']['ee_left']}",
                 "/".join(str(x) for x in c["radii_mm"]["ee_finger_l1"]), "/".join(str(x) for x in c["radii_mm"]["ee_left"]),
                 f"{c['max_within_capsule_gap_mm']:.2f}", f"{fm['frac_mesh_axis_len_spanned']:.3f}", f"{fm['uncovered_at_min_end_mm']:.1f} / {fm['uncovered_at_max_end_mm']:.1f}",
                 f"{fm['frac_vertices_inside_union']:.3f}", f"{fm['vertex_outside_mm_median']:.1f}", f"{c['sphere_model_finger_inner_gap_mm']:.1f}",
                 f"{c['finger_l1_sphere_centroid_to_mesh_centroid_mm']['mujoco_open_-0.045']:.1f} / {c['finger_l1_sphere_centroid_to_mesh_centroid_mm']['mujoco_closed_0']:.1f}"])
table(axes[1], ["config (--links gripper)", "n sph", "finger/palm n", "finger r mm", "palm r mm", "max gap in capsule mm", "finger axis span", "uncov ends mm",
                "mesh vtx inside", "vtx out med mm", "l1-l2 sph gap mm", "sph->mesh ctr mm open/closed"], rows,
      "P1-3a gripper constraint spheres vs MuJoCo ee_finger_l1 collision mesh (finger joints pinned 0 in model; comparison at MuJoCo closed pose)", 7,
      [0.16] + [0.076] * 11)
rows = []
for run in ("t14", "t17"):
    for s in ("inflation_0.0", "inflation_0.05"):
        x = P13B[run][s]; lk = x["leak_by_body_[visible_px, passed_px, n_camera_frames_with_leak]"]
        rows.append([run, s.split("_")[1], x["n_frames"], f"{x['apple_px_passed_total']} / {x['apple_px_visible_total']} ({x['apple_retention']:.3f})",
                     x["n_frames_apple_visible_but_all_removed"], f"{x['robot_px_passed_total']} / {x['robot_px_visible_total']}", x["n_frames_with_robot_leak"]])
table(axes[2], ["run", "inflation m", "frames", "apple px passed / visible", "frames apple all removed", "robot px leaked / visible", "frames w/ leak"], rows,
      "P1-3b self-filter inflation 0 vs 0.05 — all planning rows, 3 cameras, every pixel", 8)
bodies = ["link_left_arm_2", "link_left_arm_3", "link_left_arm_4", "wrist_bracket_r", "wrist_bracket_l", "d435i_body_r", "d435i_body_l", "ee_finger_r2", "ee_finger_l1", "ee_finger_l2"]
rows = []
for run in ("t14", "t17"):
    lk = P13B[run]["inflation_0.0"]["leak_by_body_[visible_px, passed_px, n_camera_frames_with_leak]"]
    rows.append([run + " infl 0"] + [str(lk.get(b, [0, 0])[1]) for b in bodies])
    rows.append([run + " infl 0.05"] + ["0"] * len(bodies))
table(axes[3], ["leaked px by GT body"] + bodies, rows, "P1-3b robot GT pixels leaked into the cloud, per MuJoCo body (sum over frames x cameras)", 7)
rows = [[side_, q, f"{e['gap_geom_collision_mm']:.2f}", f"{e['gap_geom_visual_mm']:.2f}", f"{e['finger_body_origin_separation_mm']:.1f}"]
        for side_ in ("left",) for q, e in P13C["left"].items()]
rows.append(["apple", "extents mm", "/".join(str(x) for x in P13C["fruit_collision_geom_extents"]["apple"]["principal_extents_mm_sorted"]), "", ""])
table(axes[4], ["gripper", "finger qpos", "inner gap collision mm", "inner gap visual mm", "finger origin sep mm"], rows,
      "P1-3c inner finger opening from MuJoCo model (joint limit -0.05; dataset open cmd -0.045)")
fig.savefig(F / "t28-p1-tables.png", dpi=130, bbox_inches="tight"); plt.close(fig)
(F / "t28-p1-tables.json").write_text(json.dumps({"sources": ["outputs/verify/T28/E0/e0_summary.json", "outputs/verify/T28/P1-3/p13a_summary.json",
                                                             "outputs/verify/T28/P1-3/p13b_selffilter_summary.json", "outputs/verify/T28/P1-3/p13c_opening.json"]}, indent=1))
print("ok")
