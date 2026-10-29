"""T40 I figures (verifier). Reads outputs/verify/T40/I/raw/{case_*.json, section_*.npy}; writes
AG3S/ag3s/docs/figures/t40/t40i-{scene,clearance,table}.png + .json sidecars."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle as fs

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/I/raw'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t40'))
TAGS = ["C1807", "C1967", "C1982", "C1968", "D1807"]
LABEL = {"C1807": "C1807 (contact)", "C1967": "C1967 (contact)", "C1982": "C1982 (contact)",
         "C1968": "C1968 (success, pass)", "D1807": "D1807 (success, pass)"}
LINKS = ("ee_left", "ee_finger_l1", "ee_finger_l2")
LCOL = {"ee_left": fs.CATEGORICAL[2], "ee_finger_l1": fs.CATEGORICAL[3], "ee_finger_l2": fs.CATEGORICAL[4]}
BAR_C = fs.CATEGORICAL[1]
fs.use_korean()
C = {t: json.load(open(RAW / f"case_{t}.json")) for t in TAGS}
S = {t: np.load(RAW / f"section_{t}.npy", allow_pickle=True).item() for t in TAGS}
MAP = json.load(open(RAW / "settings.json"))["body_link_mapping"]["urdf_link_to_mujoco_body"]


def r1(x): return None if x is None else round(float(x), 1)


# ------------------------------------------------------------------ 1. scene: ESDF section through the bar
def scene():
    cols = ["locator", "coarse_20mm", "fine_5mm", "TO_link"]
    fig, axes = plt.subplots(len(TAGS), 4, figsize=(17, 4.1 * len(TAGS)))
    fs.style_axes(fig, axes.ravel().tolist())
    side = {}
    cmap = fs.sequential_cmap()
    for i, t in enumerate(TAGS):
        c = C[t]; s = S[t]; ev = s["event"]; seq = ev["ev_seq"]; sec = s[seq]
        link = c["event"]["closest_link"]
        ch = [x for x in c["chunks"] if x["seq"] == seq][0]
        reads_tf = ch["per_link"][link]["reads_target_free_layer"]
        g = sec["grid_m"] * 1000
        # locator (world top view)
        ax = axes[i, 0]; loc = c["locator"]
        for p in loc["parts"]:
            ctr = np.array(p["centre"]); axv = np.array(p["axis"])
            if abs(axv[2]) > 0.9:  # vertical post / foot -> circle
                ax.add_patch(Circle((ctr[0], ctr[1]), p["radius"], fill=False, ec=fs.INK_2, lw=1))
        a = np.array(loc["bar_a_world"]); b = np.array(loc["bar_b_world"])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=BAR_C, lw=5, solid_capstyle="round", label="bar (r 12 mm)")
        w = np.array(loc["witness_world"])
        u = (b - a)[:2] / np.linalg.norm((b - a)[:2]); n = np.array([-u[1], u[0]])
        ax.plot([w[0] - 0.08 * n[0], w[0] + 0.08 * n[0]], [w[1] - 0.08 * n[1], w[1] + 0.08 * n[1]], "--", color=fs.INK, lw=1, label="section plane (normal = bar axis)")
        cw = np.array(c["event"]["spheres_world_at_event_row"]["centres"]); ln = np.array(c["event"]["spheres_world_at_event_row"]["links"])
        for L in LINKS:
            k = ln == L
            ax.scatter(cw[k, 0], cw[k, 1], s=6, color=LCOL[L], label=f"{L} spheres (URDF)")
        ax.plot(w[0], w[1], "x", color=fs.INK, ms=8, label="closest point on hand geom")
        ax.set_aspect("equal"); ax.set_xlim(w[0] - 0.2, w[0] + 0.2); ax.set_ylim(w[1] - 0.2, w[1] + 0.2)
        ax.set_title(f"{LABEL[t]}\nworld top view, event t={c['event']['t']}", fontsize=9)
        ax.set_xlabel("x world (m)"); ax.set_ylabel("y world (m)")
        if i == 0: ax.legend(fontsize=6, loc="lower left")
        names = [("coarse_20mm", "coarse 20 mm"), ("fine_5mm", "fine 5 mm"),
                 (("TO_target_free" if reads_tf else "TO_plain"), ("TO view for this link: target-free composite" if reads_tf else "TO view for this link: min(coarse, fine)"))]
        side[t] = {"event": {k: c["event"][k] for k in ("t", "seq", "closest_left_geom", "closest_link", "event_geom_to_bar_mm_at_row")},
                   "section": "plane through the bar-axis point nearest the event contact point, normal = bar axis; field of the chunk planned at t=%d" % ch["t_step"],
                   "panels": {}}
        for j, (key, title) in enumerate(names, start=1):
            ax = axes[i, j]
            F = sec["fields"][key] * 1000
            im = ax.imshow(np.clip(F, -20, 60), origin="lower", extent=[g[0], g[-1], g[0], g[-1]], cmap=cmap, vmin=-20, vmax=60)
            if np.isfinite(F).any() and np.nanmin(F) < 0 < np.nanmax(F):
                ax.contour(g, g, F, levels=[0], colors=[fs.INK], linewidths=1.2)
            ax.add_patch(Circle((0, 0), ev["bar_radius_m"] * 1000, fill=False, ec=BAR_C, lw=2))
            V = ev["geom_vertices_uvax_link"]
            for li, L in enumerate(LINKS):
                k = (V[:, 3] == li) & (np.abs(V[:, 2]) < 0.03)
                ax.scatter(V[k, 0] * 1000, V[k, 1] * 1000, s=1, color=LCOL[L], alpha=0.35)
            Sp = ev["spheres_event_uvaxr"]; sl = np.array(ev["sphere_links"])
            for (uu, vv, aa, rr), L in zip(Sp, sl):
                if abs(aa) < 0.03:
                    ax.add_patch(Circle((uu * 1000, vv * 1000), rr * 1000, fill=False, ec=LCOL[L], lw=1.0))
            wx = ev["witness_uvax"]; ax.plot(wx[0] * 1000, wx[1] * 1000, "x", color=fs.INK, ms=7)
            ax.set_xlim(-100, 100); ax.set_ylim(-100, 100); ax.set_aspect("equal")
            ax.set_title(title, fontsize=9); ax.set_xlabel("u (mm)"); ax.set_ylabel("w (mm)")
            ff = ch["bar_in_field"].get({"TO_target_free": "TO_target_free_composite", "TO_plain": "TO_plain_composite"}.get(key, key))
            if ff:
                txt = f"field at bar axis (focus) {r1(ff['field_at_focus_axis_point_mm'])} mm (true -12)\nfield at true surface median {r1((ff['field_at_true_surface_mm'] or {}).get('median'))} mm (true 0)"
                ax.text(-97, 97, txt, fontsize=7, va="top", color=fs.INK, bbox=dict(fc=fs.SURFACE, ec="none", alpha=0.8))
                side[t]["panels"][key] = ff
        fig.colorbar(im, ax=axes[i, 3], fraction=0.046, label="field (mm), clipped -20..60")
    fig.suptitle("T40 I - ESDF section through the hurdle bar at the event chunk. orange circle = true bar (r 12 mm), black line = field zero level,\n"
                 "outlined circles = URDF query spheres at the event row (|offset along bar| < 30 mm), dots = true MuJoCo hand-mesh vertices (same window), x = closest hand point to the bar",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "t40i-scene.png", dpi=110); plt.close(fig)
    json.dump(side, open(FIG / "t40i-scene.json", "w"), indent=1)


# ------------------------------------------------------------------ 2. per-chunk clearance graph
def graph():
    fig, axes = plt.subplots(len(TAGS), 1, figsize=(13, 3.3 * len(TAGS)))
    fs.style_axes(fig, list(axes))
    side = {}
    for ax, t in zip(axes, TAGS):
        c = C[t]; link = c["event"]["closest_link"]; side[t] = {"link_urdf": link, "mujoco_body": MAP[link], "chunks": []}
        mt, mg = [], []
        for ch in c["chunks"]:
            t0 = ch["t_step"]; ex = ch["executed"]
            ks = np.arange(8)
            if ch["gate"]["action"] == "hold":
                ax.axvspan(t0 - 0.5, t0 + 7.5, color=fs.GRID_INK, alpha=0.5, lw=0)
            tov = ch["per_link"][link]["TO_view_clearance_min_per_step_mm_refined"]
            fin_g = [x[link] for x in ex["refined_plan_true_geom_to_bar_mm_per_step"]]
            ref_g = [x[link] for x in ex["reference_plan_true_geom_to_bar_mm_per_step"]]
            ax.plot(t0 + ks, tov, "-o", ms=3, color=fs.CATEGORICAL[0], label="TO view, refined plan (query spheres, recorded field)" if t0 == c["chunks"][0]["t_step"] else None)
            ax.plot(t0 + ks, fin_g, "-s", ms=3, color=fs.CATEGORICAL[2], label="refined plan, true MuJoCo mesh to bar" if t0 == c["chunks"][0]["t_step"] else None)
            ax.plot(t0 + ks, ref_g, ":", color=fs.CATEGORICAL[1], label="policy reference plan, true mesh to bar" if t0 == c["chunks"][0]["t_step"] else None)
            for tt, x in zip(ex["rows"], ex["measured_rows_true_geom_to_bar_mm"]):
                mt.append(tt); mg.append(x[link])
            side[t]["chunks"].append({"seq": ch["seq"], "t": t0, "gate": ch["gate"]["action"], "TO_view_mm": tov,
                                      "refined_plan_true_mesh_mm": fin_g, "reference_plan_true_mesh_mm": ref_g})
        o = np.argsort(mt)
        ax.plot(np.array(mt)[o], np.array(mg)[o], "-", lw=2.2, color=fs.INK, label="measured pose, true mesh to bar (recorded control rows)")
        side[t]["measured"] = {"t": np.array(mt)[o].tolist(), "mm": np.array(mg)[o].tolist()}
        ax.axhline(0, color=fs.INK_2, lw=0.8)
        if c["event"]["kind"] == "contact":
            ax.axvline(c["event"]["t"], color=fs.CATEGORICAL[7], lw=1.2, ls="--", label="obstacle stop row (contact in the following sim steps)")
        else:
            ax.axvline(c["event"]["t"], color=fs.CATEGORICAL[6], lw=1.2, ls="--", label="carry-phase minimum hand-to-hurdle row")
        ax.set_ylim(-30, 100); ax.set_ylabel("mm")
        ax.set_title(f"{LABEL[t]} - link {MAP[link]} (MuJoCo) = {link} (URDF spheres); grey = HOLD chunk", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
    axes[-1].set_xlabel("control step t (plan step k of a chunk drawn at t_chunk + k)")
    fig.tight_layout()
    fig.savefig(FIG / "t40i-clearance.png", dpi=110); plt.close(fig)
    json.dump(side, open(FIG / "t40i-clearance.json", "w"), indent=1)


# ------------------------------------------------------------------ 3. table
def table():
    rows = []
    def add(name, f): rows.append([name] + [f(t) for t in TAGS])
    ev = lambda t: C[t]["event"]
    evch = lambda t: [x for x in C[t]["chunks"] if x["seq"] == ev(t)["seq"]][0]
    link = lambda t: ev(t)["closest_link"]
    add("event row t / chunk seq", lambda t: f"{ev(t)['t']} / {ev(t)['seq']}")
    add("hand geom (MuJoCo body) / URDF link", lambda t: f"{ev(t)['closest_left_geom']} {MAP[link(t)]} / {link(t)}")
    add("hand geom to bar at row (mm)", lambda t: r1(ev(t)["event_geom_to_bar_mm_at_row"]))
    add("(1) coarse 20 mm: field at bar axis (true -12)", lambda t: r1(evch(t)["bar_in_field"]["coarse_20mm"]["field_at_focus_axis_point_mm"]))
    add("(1) fine 5 mm: field at bar axis", lambda t: r1(evch(t)["bar_in_field"]["fine_5mm"]["field_at_focus_axis_point_mm"]))
    add("(1) target-free 5 mm: field at bar axis", lambda t: r1((evch(t)["bar_in_field"].get("target_free_5mm") or {}).get("field_at_focus_axis_point_mm")) if evch(t)["bar_in_field"].get("target_free_5mm") else "no layer")
    add("(1) fine: field at true surface, median (true 0)", lambda t: r1(evch(t)["bar_in_field"]["fine_5mm"]["field_at_true_surface_mm"]["median"]))
    add("(1) coarse: field at true surface, median", lambda t: r1(evch(t)["bar_in_field"]["coarse_20mm"]["field_at_true_surface_mm"]["median"]))
    def cov(t, k):
        g = ev(t)["sphere_coverage_at_event_row"][link(t)]["geoms"][ev(t)["closest_left_geom"]]; return g[k]
    add("(2) mesh vertices inside a query sphere (fraction)", lambda t: round(cov(t, "frac_vertices_inside_a_sphere"), 3))
    add("(2) mesh vertex gap to link spheres, median (mm)", lambda t: r1(cov(t, "vertex_gap_to_own_link_spheres_mm")["median"]))
    add("(2) closest-to-bar mesh point: gap to spheres (mm)", lambda t: r1(cov(t, "witness_on_geom_gap_to_spheres_mm")))
    add("(2) link spheres to bar at row, true (mm)", lambda t: r1(ev(t)["sphere_coverage_at_event_row"][link(t)]["sphere_to_bar_true_min_mm"]))
    add("(2) event chunk: gate / TO status", lambda t: f"{evch(t)['gate']['action']} / {evch(t)['trajopt_status']}")
    add("(2) event chunk: worst row (link, tier, mm)", lambda t: (lambda w: f"{w.get('link')}, {w.get('tier')}, {r1(1000 * w.get('clearance_m', 0))}")(evch(t)["TO_worst_row"] or {}))
    add("(2) event chunk: TO view, this link, refined (mm)", lambda t: f'{r1(evch(t)["per_link"][link(t)]["TO_view_clearance_min_mm_refined"])}'
        + ("" if evch(t)["fidelity"].get("to_view", {}).get("abs_diff_mm", 0) <= 1.0 else " (reconstruction; held-object rows not recorded)"))
    add("(2) event chunk: refined plan, true mesh, min (mm)", lambda t: r1(min(x[link(t)] for x in evch(t)["executed"]["refined_plan_true_geom_to_bar_mm_per_step"])))
    add("(3) event chunk executed as", lambda t: evch(t)["executed"]["source_by_value"])
    add("(3) applied - refined, max |rad|", lambda t: f"{evch(t)['executed']['applied_vs_refined_max_abs_rad']:.3g}")
    add("(3) plan step 0 jump from q_now (deg)", lambda t: r1(evch(t)["executed"]["plan_step0_jump_from_q_now_max_deg"]))
    add("(3) first chunk executed after a collision HOLD", lambda t: {"C1807": "seq 28 (t 216) = event chunk", "C1967": "seq 26 (t 200) = event chunk",
                                                                   "C1982": "none: contact inside HOLD seq 23", "C1968": "no HOLD in run", "D1807": "no HOLD in run"}[t])
    add("(3) measured row k vs plan target k-1, max (deg)", lambda t: r1(max(evch(t)["executed"]["measured_vs_TO_plan_prev_step_max_deg_per_row"])))
    add("(3) same, link centroid, max (mm)", lambda t: r1(max(evch(t)["executed"]["measured_vs_TO_plan_link_centroid_mm_per_row"])))
    add("(3) HOLD: q_hold pose, true mesh to bar (mm)", lambda t: r1(evch(t)["executed"]["q_hold_pose_true_geom_to_bar_mm"][link(t)]) if "q_hold_pose_true_geom_to_bar_mm" in evch(t)["executed"] else "-")
    add("(3) HOLD: q_hold vs measured at entry (deg)", lambda t: r1(evch(t)["executed"]["q_hold_vs_measured_first_row_max_deg"]) if "q_hold_vs_measured_first_row_max_deg" in evch(t)["executed"] else "-")
    add("(4) |reference - refined| joint max (deg)", lambda t: r1(evch(t)["policy_pull"]["joint_abs_ref_minus_refined_max_deg"]))
    add("(4) reference - refined toward bar, max (mm)", lambda t: r1(max(evch(t)["policy_pull"]["link_centroid_ref_minus_refined_toward_bar_mm_per_step"])))
    add("(4) policy displacement toward bar (mm)", lambda t: r1(evch(t)["policy_pull"]["policy_displacement_toward_bar_mm"]))
    add("(4) policy direction vs bar inward normal (deg)", lambda t: r1(evch(t)["policy_pull"]["policy_direction_angle_to_bar_inward_normal_deg"]))
    add("(4) reference plan, true mesh, min (mm)", lambda t: r1(min(x[link(t)] for x in evch(t)["executed"]["reference_plan_true_geom_to_bar_mm_per_step"])))
    fig, ax = plt.subplots(figsize=(17, 0.36 * len(rows) + 1.2))
    fs.style_axes(fig, [ax]); ax.axis("off")
    tb = ax.table(cellText=[[str(v) for v in r] for r in rows], colLabels=["quantity (event chunk)"] + [LABEL[t] for t in TAGS],
                  loc="center", cellLoc="center", colLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.35)
    for (r, cc), cell in tb.get_celld().items():
        cell.set_edgecolor(fs.GRID_INK)
        if cc == 0: cell.set_text_props(ha="left"); cell._loc = "left"
        if r == 0: cell.set_text_props(color=fs.INK, weight="bold")
    tb.auto_set_column_width(list(range(len(TAGS) + 1)))
    ax.set_title("T40 I - event chunk per run: (1) ESDF at the bar, (2) query-sphere coverage and TO verdict, (3) what was executed, (4) policy pull", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "t40i-table.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    json.dump({"columns": TAGS, "rows": rows}, open(FIG / "t40i-table.json", "w"), indent=1)


if __name__ == "__main__":
    scene(); graph(); table()
