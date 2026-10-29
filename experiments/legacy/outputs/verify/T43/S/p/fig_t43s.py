"""T43 S figures (verifier). Reads raw/*.json written by measure_pick / measure_demo / pick_ref_target / summarize_pick /
measure_held / contact_detail. Renders MuJoCo scenes (osmesa) from the recorded qpos. Run with cwd ${AG3S_PROJECT_ROOT}-t41c.
Writes AG3S/ag3s/docs/figures/t43/t43s-*.png + same-name .json sidecars."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import measure_held as MH   # puts ${AG3S_PROJECT_ROOT}-t41c first on sys.path
import measure_pick as MP
from AG3S.ag3s.experiments.common import figstyle as FS   # byte-identical in work-t41c and main

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43')); FIG.mkdir(parents=True, exist_ok=True)
FS.use_korean()
C = FS.CATEGORICAL


def side(name, data):
    (FIG / f"{name}.json").write_text(json.dumps(MP.jl(data), indent=1))


def render(m, d, lookat, dist, azim, elev, spheres=(), w=480, h=400):
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = azim; cam.elevation = elev
    mujoco.mj_forward(m, d)
    r.update_scene(d, cam)
    scn = r.scene
    for pos, rad, rgba in spheres:
        if scn.ngeom >= scn.maxgeom: break
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([rad, 0, 0], float),
                            np.asarray(pos, float), np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
        scn.ngeom += 1
    img = r.render(); r.close()
    return img


def hexrgba(hx, a):
    hx = hx.lstrip("#"); return [int(hx[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a]


# ------------------------------------------------------------------ (4) pick scene
def fig_pick_scene():
    R = json.loads((RAW / "pick_runs.json").read_text())["runs"]
    sel = [("E0/E0_ep1828_s18281", "E0 left apple (lifted)"), ("E0/E0_ep1834_s18341", "E0 right apple (lifted)"),
           ("E3b/E3b_ep1834_s18341", "E3b right apple (lifted)"), ("E3b/E3b_ep1959_s19591", "E3b right orange (not lifted)"),
           ("E3b/E3b_ep1503_s15031", "E3b right banana (not lifted)"), ("E3b/E3b_ep1764_s17641", "E3b right banana (not lifted)")]
    M = MP.model(); m, d = M["m"], M["d"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.8)); FS.style_axes(fig, axes)
    side_rows = []
    for ax, (key, title) in zip(axes.ravel(), sel):
        r = R[key]; t = r["rows"]["settled"]["t"]
        rows = [json.loads(l) for l in pathlib.Path(MP.RUNS[key]["run_dir"]).joinpath("frames.jsonl").read_text().splitlines() if l.strip()]
        q = np.asarray(next(x for x in rows if x["kind"] == "control" and x["t_step"] == t)["qpos"], float)
        g = MP.geom_at(q, r["arm"], r["fruit"])
        d.qpos[:] = q
        azim = 160 if r["arm"] == "left" else 200
        rgba0 = m.geom_rgba.copy()
        for bn, a in (("crate", 0.12), (r["fruit"], 0.45)):
            bid = M["B"](bn)
            for gg in range(m.ngeom):
                if int(m.geom_bodyid[gg]) == bid: m.geom_rgba[gg, 3] = min(m.geom_rgba[gg, 3], a)
        img = render(m, d, g["fruit_pos"], 0.34, azim, -40,
                     spheres=[(g["tip_mid"], 0.005, hexrgba(C[0], 1.0)), (g["fruit_pos"], 0.005, hexrgba(C[1], 1.0))])
        m.geom_rgba[:] = rgba0
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        s = r["rows"]["settled"]
        ax.set_title(f"{title}\nep{r['episode']} s{r['seed']} t={t} (close+16)", fontsize=9, color=FS.INK)
        ax.text(0.02, 0.03, f"tip dz {s['tip_dz_mm']:+.0f} mm  dxy {s['tip_dxy_mm']:.0f} mm  tilt {s['tilt_deg']:.0f} deg\n"
                f"contacts F1/F2/palm {list(s['n_contacts_by_body'].values())[1]}/{list(s['n_contacts_by_body'].values())[2]}/{list(s['n_contacts_by_body'].values())[0]}",
                transform=ax.transAxes, fontsize=8, color="white", bbox=dict(facecolor="black", alpha=0.55, lw=0))
        side_rows.append({"run": key, "t": t, "camera": {"lookat": g["fruit_pos"], "distance": 0.34, "azimuth": azim, "elevation": -40}, "render_note": "crate alpha 0.12, target fruit alpha 0.45 (visual only)",
                          **{k: s[k] for k in ("tip_dz_mm", "tip_dxy_mm", "tilt_deg", "closing_yaw_deg_mirrored_mod180", "n_contacts_by_body", "finger_gap_mm", "opening_meas")}})
    fig.suptitle("T43 S (4) grasp pose at close+16 rows: blue dot = fingertip midpoint, orange dot = fruit body centre (MuJoCo, recorded qpos)",
                 fontsize=10, color=FS.INK)
    fig.tight_layout(h_pad=2.2); fig.savefig(FIG / "t43s-pick-scene.png", dpi=130); plt.close(fig)
    side("t43s-pick-scene", {"panels": side_rows, "source": "raw/pick_runs.json", "model": MP.XML})


# ------------------------------------------------------------------ (4) pick graph
def fig_pick_graph():
    S = json.loads((RAW / "pick_summary.json").read_text())
    D = json.loads((RAW / "demo_grasp.json").read_text())["episodes"]
    R = json.loads((RAW / "pick_runs.json").read_text())["runs"]
    P = json.loads((RAW / "pick_ref_target.json").read_text())["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8)); FS.style_axes(fig, axes)
    ax = axes[0]
    for arm, col in (("left", C[0]), ("right", C[1])):
        e = [x["rows"]["settled"] for x in D if x["arm"] == arm and "settled" in x["rows"]]
        ax.scatter([x["tip_dxy_mm"] for x in e], [x["tip_dz_mm"] for x in e], s=4, color=col, alpha=0.15, lw=0)
        for cond, mk in (("E0", "s"), ("E3b", "o")):
            for gr in (True, False):
                rs = [r["rows"]["settled"] for r in R.values() if r["cond"] == cond and r["arm"] == arm and r["grasp"] == gr and "settled" in r["rows"]]
                if not rs: continue
                ax.scatter([x["tip_dxy_mm"] for x in rs], [x["tip_dz_mm"] for x in rs], s=26, marker=mk,
                           facecolor=col if gr else "none", edgecolor=col, lw=1.1,
                           label=f"{cond} {arm} {'lifted' if gr else 'not lifted'} (n={len(rs)})")
    ax.scatter([], [], s=8, color=FS.INK_2, alpha=0.4, label="training demos (faint, n=1000 per arm)")
    ax.set_xlabel("fingertip midpoint to fruit centre, horizontal (mm)"); ax.set_ylabel("fingertip midpoint minus fruit centre, z (mm)")
    ax.set_xlim(0, 90); ax.set_ylim(-30, 70); ax.axhline(0, color=FS.GRID_INK, lw=0.8)
    allr = [r["rows"]["settled"] for r in R.values() if "settled" in r["rows"]]
    nclip = sum(1 for x in allr if x["tip_dxy_mm"] > 90 or x["tip_dz_mm"] > 70 or x["tip_dz_mm"] < -30)
    ax.text(0.98, 0.02, f"{nclip} rollout point(s) outside the axes", transform=ax.transAxes, ha="right", fontsize=7, color=FS.INK_2)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left"); ax.set_title("grasp geometry at close+16 (demo: close start+16)", fontsize=9)
    ax = axes[1]
    labels = []; vals = []; cols = []
    for k in ("E0/left", "E0/right", "E3b/left", "E3b/right"):
        o = S["outcomes"][k]; labels.append(f"{k}\n{o['grasp']}/{o['n']}"); vals.append(o["grasp"] / o["n"]); cols.append(C[0] if "left" in k else C[1])
    ax.bar(range(4), vals, color=cols, width=0.6); ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("lifted >= 50 mm (fraction of runs)"); ax.set_title("T42-1 runs: lifted / runs (frozen 11:41 UTC)", fontsize=9)
    ax = axes[2]
    for i, (arm, col) in enumerate((("left", C[0]), ("right", C[1]))):
        for j, gr in enumerate((True, False)):
            v = [x["close"]["ref_minus_refined_tip_mm"] for k, x in P.items() if "close" in x and R[k]["arm"] == arm and R[k]["grasp"] == gr]
            v8 = [x["close_m8"]["ref_minus_refined_tip_mm"] for k, x in P.items() if "close_m8" in x and R[k]["arm"] == arm and R[k]["grasp"] == gr]
            xpos = i * 2 + j
            ax.scatter(np.full(len(v), xpos - 0.12) + np.random.default_rng(0).uniform(-0.05, 0.05, len(v)), v, s=14, color=col,
                       marker="o" if gr else "x")
            ax.scatter(np.full(len(v8), xpos + 0.12) + np.random.default_rng(1).uniform(-0.05, 0.05, len(v8)), v8, s=14, color=col, alpha=0.45,
                       marker="o" if gr else "x")
    ax.set_xticks(range(4)); ax.set_xticklabels(["left lifted", "left not", "right lifted", "right not"], fontsize=8)
    ax.set_ylabel("|policy target - TO target| at fingertips (mm)")
    ax.set_title("E3b: policy (reference) vs TO (refined) target\nleft of each pair: close row, right (faint): close-8", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "t43s-pick-graph.png", dpi=130); plt.close(fig)
    side("t43s-pick-graph", {"outcomes": S["outcomes"], "geometry_settled": S["geometry_settled"], "policy_vs_TO_target": S["policy_vs_TO_target"],
                             "source": ["raw/pick_summary.json", "raw/demo_grasp.json", "raw/pick_runs.json", "raw/pick_ref_target.json"]})


# ------------------------------------------------------------------ (5) held scene
CONTACT_SEL = [_ag3s_workspace_path('outputs/verify/T41/a/runs/E3b/C/E3b_ep1968_s19681'), _ag3s_workspace_path('outputs/verify/T40/J/runs/E3b/D/E3b_ep1995_s19952'),
               _ag3s_workspace_path('outputs/verify/T41/a/runs/E3b/C/E3b_ep1967_s19672')]


def fig_held_scene():
    H = json.loads((RAW / "held_runs_all.json").read_text())["runs"]
    CD = json.loads((RAW / "contact_detail.json").read_text())["runs"]
    SEC = np.load(RAW / "held_sections_all.npy", allow_pickle=True).item()
    fig, axes = plt.subplots(2, 3, figsize=(14, 9.2)); FS.style_axes(fig, axes)
    sd = []
    for col, key in enumerate(CONTACT_SEL):
        r = H[key]; cd = CD[key]; t_c = cd["contact_t"]
        ix = MH.IDX[key]; rd = pathlib.Path(key); man = json.loads((rd / "manifest.json").read_text()); argv = man["extra"]["argv"]
        G = MH.setup(pathlib.Path(ix["server"]) / "serve_safe.log", man)
        m, d = MH.mj()
        mg = MH.PPO.PickPlaceObstacleManager(m, d, MH.PPO.load_pick_place_obstacle_config(argv[argv.index("--obstacle-config") + 1]))
        mg.activate(argv[argv.index("--obstacle-profile") + 1])
        rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
        q = np.asarray(next(x for x in rows if x["kind"] == "control" and x["t_step"] == t_c)["qpos"], float)
        d.qpos[:] = q
        rad = np.asarray(cd["held_radii_mm"]) / 1000
        sph = [(c, rr, hexrgba(C[0], 0.35)) for c, rr in zip(cd["held_centre_world"], rad)]
        sph += [(cd["contact_point_world"], 0.004, hexrgba(C[7], 1.0))]
        bar = np.asarray(r["bar"]["b_world"]) - np.asarray(r["bar"]["a_world"])
        azim = float(np.degrees(np.arctan2(bar[1], bar[0])))
        imgs = [render(m, d, cd["apple_centre_world"], 0.42, a, -8, spheres=sph) for a in (azim, azim + 180.0)]
        dark = [float(np.mean(im.mean(axis=2) < 45)) for im in imgs]
        img = imgs[int(np.argmin(dark))]; azim = (azim, azim + 180.0)[int(np.argmin(dark))]
        ax = axes[0, col]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        tag = key.split("verify/")[1].replace("/runs/E3b/", " ").replace("E3b_ep", "ep")
        ax.set_title(f"{tag}\nfirst apple-bar contact row t={t_c} (free camera, azimuth = bar axis or its opposite)", fontsize=9)
        ax.text(0.02, 0.03, f"apple-bar {cd['apple_bar_mm']:+.1f} mm   hand-bar {cd['hand_to_bar_min_mm']:.1f} mm\n"
                f"contact point {cd['contact_point_height_above_apple_centre_mm']:+.0f} mm from apple centre (half height {cd['apple_half_height_mm']:.0f})\n"
                f"contact point outside held spheres by {cd['contact_point_gap_to_held_spheres_mm']:.1f} mm",
                transform=ax.transAxes, fontsize=7.5, color="white", bbox=dict(facecolor="black", alpha=0.55, lw=0))
        # section at the last held chunk before contact
        seq = cd["held_chunk_seq"]; s = SEC[f"{key}#{seq}"]
        g = np.asarray(s["grid_m"]) * 1000
        ax = axes[1, col]
        comp = np.asarray(s["composite"], float)[::-1, :] * 1000   # rows = w; plot with up = -w
        g_up = -g[::-1]
        im = ax.contourf(g, g_up, comp, levels=np.arange(-30, 61, 5), cmap=FS.sequential_cmap(), extend="both")
        ax.contour(g, g_up, comp, levels=[0], colors=[C[3]], linewidths=1.4)
        ax.add_patch(Circle((0, 0), r["bar"]["radius_mm"], fill=False, ec=FS.INK, lw=1.5, ls="--"))
        av = np.asarray(s["apple_vertices_uvw"]) * 1000
        ax.scatter(av[:, 0], -av[:, 1], s=5, color=C[1], label="apple mesh vertices (projected)")
        hv = np.asarray(s["hand_vertices_uvw"]) * 1000; sl = np.abs(hv[:, 2]) < 15
        ax.scatter(hv[sl, 0], -hv[sl, 1], s=0.6, color=FS.INK_2, alpha=0.5, label="hand mesh vertices |w|<15 mm")
        for i, (u, v, wv, rr) in enumerate(np.asarray(s["held_uvwr"]) * 1000):
            ax.add_patch(Circle((u, -v), rr, fill=False, ec=C[0], lw=1.6, label="held query sphere (TO row), projected outline" if i == 0 else None))
        if s.get("filter_uvwr") is not None:
            for i, (u, v, wv, rr) in enumerate(np.asarray(s["filter_uvwr"]) * 1000):
                ax.add_patch(Circle((u, -v), rr, fill=False, ec=C[2], lw=1.0, ls=":", label="self-filter sphere (fit + 20 mm), projected" if i == 0 else None))
        ax.set_aspect("equal"); ax.set_xlim(-120, 120); ax.set_ylim(-120, 120)
        ax.set_xlabel("u (mm, horizontal, normal to bar axis)"); ax.set_ylabel("up (mm, -w; w = axis x u)")
        ch = [c for c in r["held_chunks"] if c["seq"] == seq][0]
        ax.set_title(f"section normal = bar axis, through the bar point nearest the apple\nchunk seq {seq} (t {ch['t_step']}), recorded TO field (composite)", fontsize=8.5)
        if col == 0: ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        if col == 2: fig.colorbar(im, ax=ax, fraction=0.046, label="field distance (mm); amber = 0 level; dashed = true bar")
        sd.append({"run": key, "contact": cd, "section_chunk_seq": seq, "held_radii_mm": ch["held_radii_mm"], "filter_radii_mm": ch["filter_radii_mm"],
                   "bar_in_field": ch.get("bar_in_field", {}).get("composite_at_axis_point_nearest_apple_mm"),
                   "coverage": ch.get("coverage_at_planning_row"), "camera": {"lookat": cd["apple_centre_world"], "distance": 0.42, "azimuth": azim, "elevation": -8}})
    fig.suptitle("T43 S (5) held apple vs hurdle bar: top = MuJoCo at the first contact row, blue = TO held query spheres placed by parent-link FK, "
                 "red dot = contact point on the apple; bottom = section of the TO's field", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "t43s-held-scene.png", dpi=125); plt.close(fig)
    side("t43s-held-scene", {"panels": sd, "source": ["raw/held_runs_all.json", "raw/contact_detail.json", "raw/held_sections_all.npy"]})


def fig_held_graph():
    H = json.loads((RAW / "held_runs_all.json").read_text())["runs"]
    keys = [k for k, r in H.items() if r["apple_bar_contact_t"] is not None]
    # T41a and T41c contact runs are identical up to the contact; show T40G C1968, T40J D1995, T41a C1967 / C1968 / D1807
    keys = [k for k in keys if "/T41/c/" not in k]
    fig, axes = plt.subplots(1, len(keys), figsize=(4.0 * len(keys), 4.2), sharey=True); FS.style_axes(fig, axes)
    sd = {}
    for ax, k in zip(axes, keys):
        r = H[k]; t = []; tr = []; to = []; an = []
        for c in r["held_chunks"]:
            e = c["executed"]; t += e["t"]; tr += e["true_apple_bar_mm"]; to += e["TO_view_held_mm"]; an += e["analytic_held_bar_mm"]
        ax.plot(t, tr, color=C[1], lw=1.6, label="true apple mesh to bar (MuJoCo)")
        ax.plot(t, an, color=C[0], lw=1.2, ls="--", label="held query spheres to true bar (analytic)")
        ax.plot(t, to, color=C[2], lw=1.2, label="TO view: field(held centre) - r (all obstacles)")
        ax.axhline(0, color=FS.INK, lw=0.7); ax.axvline(r["apple_bar_contact_t"], color=C[7], lw=0.8, ls=":")
        ax.set_ylim(-10, 80); ax.set_xlabel("control row t (held chunks up to contact)")
        ax.set_title(k.split("verify/")[1].replace("/runs/E3b/", " ").replace("E3b_ep", "ep"), fontsize=8.5)
        sd[k] = {"t": t, "true_apple_bar_mm": tr, "TO_view_held_mm": to, "analytic_held_bar_mm": an, "contact_t": r["apple_bar_contact_t"]}
    axes[0].set_ylabel("distance / clearance (mm)"); axes[0].legend(fontsize=6.5, frameon=False, loc="upper right")
    fig.suptitle("T43 S (5) executed rows of the held chunks before the apple-bar contact (dotted = first contact row)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "t43s-held-graph.png", dpi=130); plt.close(fig)
    side("t43s-held-graph", {"runs": sd, "note": "T41 c contact runs are identical to T41 a up to the contact row (same values) and are omitted from the plot"})


def fig_table():
    S = json.loads((RAW / "pick_summary.json").read_text()); G = S["geometry_settled"]
    CD = json.loads((RAW / "contact_detail.json").read_text())["runs"]; UL = json.loads((RAW / "contact_vs_unlifted.json").read_text())
    H = json.loads((RAW / "held_runs_all.json").read_text())["runs"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.6), gridspec_kw={"height_ratios": [1, 1.15]}); FS.style_axes(fig, axes)
    for ax in axes: ax.axis("off")
    def m(k, f): v = G[k][f]; return "-" if v is None else f"{v['median']:.1f} [{v['q25']:.1f}, {v['q75']:.1f}]"
    rows = []
    for k, lab in (("demo/left/all", "demo left (1000)"), ("demo/right/all", "demo right (1000)"), ("E0/left/all/all", "E0 left"), ("E0/right/all/all", "E0 right"),
                   ("E3b/left/grasp/all", "E3b left lifted"), ("E3b/left/no_grasp/all", "E3b left not lifted"),
                   ("E3b/right/grasp/all", "E3b right lifted"), ("E3b/right/no_grasp/all", "E3b right not lifted")):
        o = S["outcomes"].get("/".join(k.split("/")[:2]))
        rows.append([lab, str(G[k]["tilt_deg"]["n"]), m(k, "tip_dz_mm"), m(k, "tip_dxy_mm"), m(k, "tilt_deg"),
                     "-" if (o is None or "demo" in k) else f"{o['grasp']}/{o['n']}"])
    pick_rows = rows
    tb = axes[0].table(cellText=rows, colLabels=["group", "n", "tip dz mm (median [q25,q75])", "tip dxy mm", "tilt deg", "lifted/runs (cond x arm)"],
                       loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.35)
    axes[0].set_title("(4) grasp geometry at close+16 (fingertip midpoint vs fruit body centre) and outcomes, T42-1 + training demos", fontsize=9.5)
    rows = []
    for k, cd in CD.items():
        r = H[k]; c = [h for h in r["held_chunks"] if h["seq"] == cd["held_chunk_seq"]][0]; cv = c.get("coverage_at_planning_row") or {}; b = c.get("bar_in_field") or {}
        ex = c["executed"]; i = ex["t"].index(cd["contact_t"]) if cd["contact_t"] in ex["t"] else None
        rows.append([k.split("verify/")[1].replace("/runs/E3b/", " ").replace("E3b_ep", "").split("_s")[0], str(cd["contact_t"]), str(len(cd["held_radii_mm"])),
                     "/".join(f"{x:.0f}" for x in cd["held_radii_mm"]), "/".join(f"{x:.0f}" for x in UL[k]["lift_mm"]),
                     f"{cv.get('frac_inside_held_spheres', float('nan')):.2f}", f"{cv.get('max_outside_mm', float('nan')):.1f}",
                     f"{cd['contact_point_height_above_apple_centre_mm']:+.0f}", f"{cd['contact_point_gap_to_held_spheres_mm']:.1f}",
                     f"{UL[k]['contact_point_gap_to_unlifted_fit_sphere_mm']:+.1f}",
                     f"{b.get('composite_at_axis_point_nearest_apple_mm', float('nan')):.1f}", str(b.get("n_axis_points_inside_held_free_volume")),
                     f"{cd['apple_bar_mm']:+.1f}", "-" if i is None else f"{ex['TO_view_held_mm'][i]:+.1f}", "-" if i is None else f"{ex['analytic_held_bar_mm'][i]:+.1f}",
                     f"{cd['hand_to_bar_min_mm']:.1f}"])
    tb = axes[1].table(cellText=rows, colLabels=["run", "t\ncontact", "held\nrows", "radius\nmm", "J1 lift\nmm", "apple vtx\ninside", "max\noutside mm",
                                                   "contact z\nvs centre", "contact\noutside held", "vs unlifted\nfit sphere", "field at bar\n(true -12)",
                                                   "bar pts in\nheld_free", "true\napple-bar", "TO view\nheld row", "held sphere\n-true bar", "hand-bar\nmm"],
                       loc="center", cellLoc="center"); tb.auto_set_font_size(False); tb.set_fontsize(7); tb.scale(1, 1.9)
    axes[1].text(0.5, -0.02, "mm unless noted. coverage / field columns: last held planning chunk before contact (planning-row qpos); distances: first contact control row. "
                 "T41 c rows equal T41 a up to the contact.", transform=axes[1].transAxes, ha="center", fontsize=7.5, color=FS.INK_2)
    axes[1].set_title("(5) apple-bar contact runs: held query spheres (parent ee_left, FK per plan step), coverage, field, distances at the first contact row", fontsize=9.5)
    fig.tight_layout(); fig.savefig(FIG / "t43s-table.png", dpi=130); plt.close(fig)
    side("t43s-table", {"pick_rows": pick_rows, "contact_rows": rows, "geometry": G, "outcomes": S["outcomes"], "contact": CD, "unlifted": UL})


if __name__ == "__main__":
    which = sys.argv[1:] or ["pick_scene", "pick_graph", "held_scene", "held_graph", "table"]
    for w in which: globals()[f"fig_{w}"]()
    print("ok", which)
