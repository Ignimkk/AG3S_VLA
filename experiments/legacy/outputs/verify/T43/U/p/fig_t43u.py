"""T43 U figures (verifier). Reads raw/summary_u.json + raw/runs/*.json (measure_u / summarize_u). CPU, osmesa.
Writes AG3S/ag3s/docs/figures/t43/t43u-{timeline,scene,table}.png + same-name .json sidecars.
usage: fig_t43u.py RUNKEY[,RUNKEY...] (timeline runs) SCENE_SPEC[,...] (RUNKEY@seq for the scene panels)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(pathlib.Path(__file__).parent)); sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import measure_u as MU
from AG3S.ag3s.experiments.common import figstyle as FS

U = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/U'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43')); FIG.mkdir(parents=True, exist_ok=True)
KO = FS.use_korean()
C = FS.CATEGORICAL
SUM = json.loads((U / "raw/summary_u.json").read_text())
LABEL_COL = {"fruit": C[0], "obstacle": C[1], "crate": C[2], "other_fruit": C[3], "table": C[4], "hand": C[6], "none": FS.GRID_INK, None: FS.GRID_INK}
STATE_COL = {"searching": FS.SEQ_BLUE[0], "latched": FS.SEQ_BLUE[3], "closing": FS.SEQ_BLUE[9], "held": FS.SEQ_BLUE[-1], "placed": FS.GRID_INK, "released": FS.GRID_INK}  # ordinal states -> one blue ramp


def side(name, data):
    (FIG / f"{name}.json").write_text(json.dumps(MU.jl(data), indent=1, default=float))


def load(key):
    return json.loads((U / "raw/runs" / (key.replace("/", "__") + ".json")).read_text())


def series(key):
    """per control row: min finger-fruit distance (mm), fruit dz (mm), contacts."""
    RU = json.loads((U / "raw/runs_u.json").read_text())
    R = RU["server_runs"].get(key) or RU["e0_runs"][key]
    MU.place_obstacle(R["obstacle"])
    rows = [json.loads(l) for l in (pathlib.Path(R["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [r for r in rows if r["kind"] == "control"]; plan = [r for r in rows if r["kind"] == "planning"]
    fa = MU.QADR[R["fruit"]]; z0 = plan[0]["object_poses"][R["fruit"]]["pos"][2]
    t, dmin, dz, nc = [], [], [], []
    for c in ctrl[::2]:
        q = np.asarray(c["qpos"], float); h = MU.hand_fruit(q, R["arm"], R["fruit"])
        t.append(c["t_step"]); dmin.append(min(h["finger1_mm"], h["finger2_mm"])); dz.append(1e3 * (q[fa + 2] - z0))
        nc.append(h["finger1_contacts"] + h["finger2_contacts"] + h["palm_contacts"])
    return {"t": t, "finger_fruit_mm": dmin, "fruit_dz_mm": dz, "contacts": nc}


def fig_timeline(keys):
    n = len(keys)
    fig, axes = plt.subplots(4, n, figsize=(4.6 * n, 9.2), sharex="col", gridspec_kw={"height_ratios": [0.9, 1.4, 1.4, 1.4]})
    axes = np.asarray(axes).reshape(4, n)
    FS.style_axes(fig, axes.ravel().tolist())
    data = {}
    for j, key in enumerate(keys):
        r = load(key); S = SUM["runs"][key]; se = series(key)
        tl = r["timeline"]; rows = r.get("rows", [])
        data[key] = {"outcome": r["outcome"], "close_ts": r["close_ts"], "lift_t": r["lift_t"], "series": se,
                     "chunks": [{"seq": e["seq"], "t": e["t"], "state": e["grasp_state"], "man_id": e["man_id"],
                                 "man_nearest": (e["man_centroid"] or {}).get("nearest")} for e in tl],
                     "rows": [{"seq": d["seq"], "t": d["t"], "defl_mm": (d.get("deflection") or {}).get("defl_mm"),
                               "radial_mm": (d.get("deflection") or {}).get("radial_mm"),
                               "fruit_rows_obstacle": d.get("fruit_rows_obstacle_margin_applied"),
                               "remnant": (d.get("carve") or {}).get("remnant_frac_free"), "carved": (d.get("carve") or {}).get("carved")}
                              for d in rows if "error" not in d]}
        a = axes[0, j]
        for e in tl:
            a.barh(1, 8, left=e["t"], color=STATE_COL.get(e["grasp_state"], FS.GRID_INK), height=0.8, linewidth=0)
            a.barh(0, 8, left=e["t"], color=LABEL_COL.get((e["man_centroid"] or {}).get("nearest")), height=0.8, linewidth=0)
        a.set_yticks([0, 1]); a.set_yticklabels(["manipulated\n(nearest GT)", "grasp state"], fontsize=7)
        a.set_title(f"{key.replace('/E3b_', ' ').replace('/V', ' V')}\noutcome {r['outcome']}", fontsize=8)
        a2 = axes[1, j]
        a2.plot(se["t"], np.clip(se["finger_fruit_mm"], -5, 50), color=C[0], lw=1, label="finger-fruit min (mm, cap 50)")
        a2.plot(se["t"], np.clip(se["fruit_dz_mm"], -20, 250) / 5, color=C[2], lw=1, label="fruit dz / 5 (mm)")
        tc = [t for t, c in zip(se["t"], se["contacts"]) if c > 0]
        a2.scatter(tc, [-3] * len(tc), s=4, color=C[7], label="finger/palm-fruit contact")
        a2.set_ylim(-6, 52)
        a3 = axes[2, j]
        rr = data[key]["rows"]
        a3.scatter([x["t"] for x in rr], [x["defl_mm"] for x in rr], s=8, color=C[0], label="TO deflection max |ref-refined| tip (mm)")
        a3.scatter([x["t"] for x in rr], [x["radial_mm"] for x in rr], s=8, color=C[1], marker="x", label="radial (+ = away from fruit)")
        a3.axhline(0, color=FS.GRID_INK, lw=0.8); a3.set_ylim(-60, 80)
        a4 = axes[3, j]
        a4.bar([x["t"] for x in rr], [x["fruit_rows_obstacle"] or 0 for x in rr], width=6, color=C[1], label="hand rows nearest=fruit classed obstacle (a)")
        a4b = a4.twinx()
        a4b.plot([x["t"] for x in rr], [x["remnant"] if x["remnant"] is not None else np.nan for x in rr], "o", ms=3, color=C[6],
                 label="fruit remnant in target-free layer (frac)")
        a4b.set_ylim(0, 1.05)
        for ax in axes[:, j]:
            for ct in r["close_ts"][:4]:
                ax.axvline(ct, color=C[7], lw=0.8, ls="--")
            if r["lift_t"] is not None:
                ax.axvline(r["lift_t"], color=C[2], lw=0.8, ls=":")
            sw = S.get("switch_away_first")
            if sw:
                ax.axvline(sw["t"], color=C[1], lw=1.2)
        axes[3, j].set_xlabel("control step t")
        if j == 0:
            a2.legend(fontsize=6, loc="upper right"); a3.legend(fontsize=6, loc="upper right")
            h1, l1 = a4.get_legend_handles_labels(); h2, l2 = a4b.get_legend_handles_labels(); a4.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper left")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=v, label=f"state {k}") for k, v in STATE_COL.items()] +
               [Patch(color=LABEL_COL[k], label=f"manip. near {k}") for k in ("fruit", "obstacle", "crate", "other_fruit", "table", "hand")],
               loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.suptitle("T43 U - pick timeline (dashed = close command, dotted = lift, solid orange = identity switch away from the fruit)", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(FIG / "t43u-timeline.png", dpi=130); plt.close(fig)
    side("t43u-timeline", data)


def render(m, d, lookat, dist, azim, elev, markers=(), w=520, h=430):
    mj = MU.gt()["mj"]
    r = mj.Renderer(m, h, w)
    cam = mj.MjvCamera(); cam.type = mj.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.distance = dist; cam.azimuth = azim; cam.elevation = elev
    mj.mj_forward(m, d)
    r.update_scene(d, cam); scn = r.scene
    for pos, rad, rgba in markers:
        if scn.ngeom >= scn.maxgeom: break
        mj.mjv_initGeom(scn.geoms[scn.ngeom], int(mj.mjtGeom.mjGEOM_SPHERE), np.array([rad, 0, 0], float), np.asarray(pos, float),
                        np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
        scn.ngeom += 1
    img = r.render(); r.close(); return img


def rgba(hx, a):
    hx = hx.lstrip("#"); return [int(hx[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a]


def scene_panel(spec):
    key, seq = spec.split("@"); seq = int(seq)
    RU = json.loads((U / "raw/runs_u.json").read_text()); R = RU["server_runs"][key]
    G = MU.gt(); MU.place_obstacle(R["obstacle"])
    rows = [json.loads(l) for l in (pathlib.Path(R["run_dir"]) / "frames.jsonl").read_text().splitlines() if l.strip()]
    p = next(r for r in rows if r["kind"] == "planning" and r["seq"] == seq)
    run = load(key); det = next(d for d in run["rows"] if d["seq"] == seq)
    surf = MU.surfaces(R["fruit"], sorted(G["mg"][tuple(R["obstacle"])].collision_geom_ids) if R["obstacle"] else [])
    TS = MU.to_setup(str(pathlib.Path(R["server"]) / "serve_safe.log"), str(pathlib.Path(R["run_dir"]) / "manifest.json"))
    q = np.asarray(p["qpos"], float)
    with np.load(R["chunks"][str(seq)]["npz"]) as z:
        s = json.loads(str(z["summary_json"]))
        import dataclasses
        FIA, lin, layout = TS["FIA"], TS["lin"], TS["layout"]
        qn = FIA.q_now_from_qpos(q, TS["qadr"]); refc = np.asarray(z["refined_chunk"], float)
        meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
        path, _ = TS["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, 8, default=TS["robot"].param_vector())
        lin.set_joint_parameter_path(path); lin.set_attached(None, None)
        traj = layout.chunk_to_trajectory(refc)[:, :8]
        field = FIA.build_field(z, 0.5)
        has_t = bool(s.get("has_target")) and z["target_points"].shape[0] > 0
        margin, mask, _, _, _ = FIA.contact_policy(TS["st"], TS["robot"].sphere_link_names, has_target=has_t, has_free=True, attached=False)
        tf = MU.lit((s.get("esdf_stats") or {}).get("target_free")) or {}; ball = tf.get("ball") or {}
        scene = FIA.make_scene(field, lin.robot_radii, TS["st"].esdf_margin, target_points=z["target_points"] if has_t else None, margin=margin, mask=mask)
        scene = dataclasses.replace(scene, obstacle_margin=R["obstacle_margin_mm"] / 1e3, support_planes=np.array([[0, 0, 1.0, ball.get("support_plane_z_m", 0.82)]]))
        cent = lin.sphere_states(traj, qn)[0]; cl = lin.esdf_clearance(traj, qn, scene)[..., 0]
        flat = cent.reshape(-1, 3); d = np.asarray(field.distance(flat), float).reshape(cent.shape[:2])
        mk = np.asarray(mask, bool).reshape(-1)[:cent.shape[1]]
        d = np.where(mk[None, :], np.asarray(field.target_free_distance(flat), float).reshape(cent.shape[:2]), d)
        d_obj = lin._manipulated_distance(flat, scene); d_obj = None if d_obj is None else d_obj.reshape(cent.shape[:2])
        cls = lin._obstacle_rows(d, flat, scene, d_obj)
    o, Rb = MU.base_T(q)
    names = TS["names"]; hidx = [i for i, n in enumerate(names) if n in MU.hand_links(R["arm"])]
    marks = []; counts = {"fruit_obstacle_class": 0, "fruit_target_class": 0, "fruit_support_class": 0}
    k = 0  # planned step 0 rows (pose at the planning qpos + first refined row)
    for i in hidx:
        if cl[k, i] > 0.030: continue
        w = o + Rb @ cent[k, i]; lab, _ = MU.nearest_label(w, surf)
        if lab != "fruit": continue
        c_ = int(cls[k, i]); col = {0: C[7], 1: C[0], 2: C[4]}.get(c_, C[6])
        counts[{0: "fruit_obstacle_class", 1: "fruit_target_class", 2: "fruit_support_class"}.get(c_, "other")] = counts.get(
            {0: "fruit_obstacle_class", 1: "fruit_target_class", 2: "fruit_support_class"}.get(c_, "other"), 0) + 1
        marks.append((w, float(lin.robot_radii[i]) + 0.001, rgba(col, 0.95)))
    fc = q[MU.QADR[R["fruit"]]:MU.QADR[R["fruit"]] + 3]
    infield = (det["carve"].get("carved") is False) or ((det["carve"].get("remnant_frac_free") or 0) >= 0.25)
    marks.append((fc + np.array([0, 0, 0.045]), 0.008, rgba(C[7] if infield else C[2], 1.0)))  # flag above the fruit
    if ball.get("centre_m"):
        bc = o + Rb @ np.asarray(ball["centre_m"], float)
        marks.append((bc, ball["radius_mm"] / 1e3, rgba(C[2], 0.18)))
    # colour the fruit geom itself
    m = G["m"]; old = {}
    for g in G["fruit"][R["fruit"]] + [x for x in range(m.ngeom) if m.geom_bodyid[x] == G["B"](R["fruit"])]:
        old[g] = (m.geom_rgba[g].copy(), int(m.geom_matid[g])); m.geom_rgba[g] = rgba(C[7] if infield else C[2], 1.0); m.geom_matid[g] = -1
    MU.set_q(q)
    img = render(m, G["d"], fc + np.array([0, 0, 0.02]), 0.42, 150 if R["arm"] == "left" else 210, -30, marks)
    for g, v in old.items(): m.geom_rgba[g] = v[0]; m.geom_matid[g] = v[1]
    info = {"key": key, "seq": seq, "t": p["t_step"], "grasp_state": s["grasp"]["state"], "man_id": (s.get("manipulated") or {}).get("id"),
            "carve": det["carve"], "fruit_in_field": infield, "fruit_rows_counts_step0": counts, "deflection": det["deflection"],
            "worst_hand_row": det["worst_hand_row"], "ball_centre_world": None if not ball.get("centre_m") else (o + Rb @ np.asarray(ball["centre_m"])).tolist(),
            "ball_radius_mm": ball.get("radius_mm")}
    return img, info


def fig_scene(specs):
    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.4)); axes = np.atleast_1d(axes)
    FS.style_axes(fig, axes.tolist()); infos = []
    for a, sp in zip(axes, specs):
        img, info = scene_panel(sp); infos.append(info)
        a.imshow(img); a.set_xticks([]); a.set_yticks([])
        c = info["carve"]
        a.set_title(f"{info['key'].replace('/E3b_', ' ')} seq {info['seq']} ({info['grasp_state']})\n"
                    f"fruit {'IN target-free layer' if info['fruit_in_field'] else 'carved out'} - remnant {c.get('remnant_frac_free')}"
                    f" ball r {c.get('ball_radius_mm')} mm\nhand rows near fruit: obstacle-class {info['fruit_rows_counts_step0']['fruit_obstacle_class']}"
                    f" / target {info['fruit_rows_counts_step0']['fruit_target_class']} / support {info['fruit_rows_counts_step0']['fruit_support_class']}",
                    fontsize=7)
    fig.suptitle("T43 U - close moment: fruit red = still in the layer the hand rows read, green = carved; green shell = target-free ball;"
                 " dots = hand rows (step 0, clearance <= 30 mm, GT nearest = fruit): red obstacle-class (gets a), blue target, pink support",
                 fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "t43u-scene.png", dpi=130); plt.close(fig)
    side("t43u-scene", infos)


def fig_table():
    G = SUM["groups"]
    order = [g for g in G]
    cols = ["n", "outcomes", "switch_away_before_close", "switch_away_before_attach", "n_with_close", "fruit_in_field_at_close",
            "a_on_fruit_rows_at_close", "defl_at_close_median_mm", "radial_at_close_median_mm", "pre_close_contact_runs"]
    hdr = ["group", "n", "outcomes", "switch away\nbefore close", "switch away\nbefore attach", "runs\nwith close", "fruit in field\nat close",
           "a on fruit rows\nat close", "TO defl at close\nmedian mm", "radial at close\nmedian mm", "pre-close\ncontact runs"]
    cell = []
    for g in order:
        v = G[g]
        cell.append([g] + [(", ".join(f"{k} {x}" for k, x in sorted(v[c].items())) if isinstance(v[c], dict) else
                            ("-" if v[c] is None else (f"{v[c]:.1f}" if isinstance(v[c], float) else str(v[c])))) for c in cols])
    # cross tabs
    def xt(name):
        t = SUM[name]["obstacle"]["table"]
        return "; ".join(f"{k}: " + ", ".join(f"{a} {b}" for a, b in sorted(v.items())) for k, v in sorted(t.items()))
    vg = [g for g in order if g.startswith("T43/V/V") and g.endswith("obst")]
    cols2 = ["closing_chunks_all", "closing_chunks", "closing_man_not_fruit", "closing_not_carved", "closing_remnant_ge25",
             "closing_fruit_rows_obstacle_class_chunks", "closing_binding", "closing_binding_hand_point_nearest", "closing_defl_median_mm",
             "closing_radial_median_mm", "closing_path_target_rows_excluded"]
    fig = plt.figure(figsize=(18, 9.5))
    ax = fig.add_axes([0.01, 0.43, 0.98, 0.52]); ax.axis("off")
    w1 = [0.10, 0.03, 0.22, 0.07, 0.07, 0.06, 0.07, 0.08, 0.08, 0.08, 0.07]
    tb = ax.table(cellText=cell, colLabels=hdr, loc="upper center", cellLoc="center", colWidths=w1); tb.auto_set_font_size(False); tb.set_fontsize(6.5)
    tb.scale(1, 1.5)
    ax2 = fig.add_axes([0.01, 0.17, 0.98, 0.22]); ax2.axis("off")
    hdr2 = ["T43 V hurdle (P)", "closing chunks\nall", "closing chunks\ntip <= 60 mm", "manip. not fruit", "not carved", "remnant >= 25 %",
            "chunks with a on\nfruit rows", "binding row\n(rows:category)", "binding hand row\npoint nearest GT", "defl median mm", "radial median mm",
            "path rows skipped\nas target"]
    cell2 = []
    for g in vg:
        v = G[g]
        cell2.append([g] + [(", ".join(f"{k} {x}" for k, x in sorted(v[c].items())) if isinstance(v[c], dict) else
                             ("-" if v[c] is None else (f"{v[c]:.1f}" if isinstance(v[c], float) else str(v[c])))) for c in cols2])
    w2 = [0.08, 0.05, 0.06, 0.05, 0.05, 0.06, 0.07, 0.22, 0.10, 0.06, 0.06, 0.07]
    tb2 = ax2.table(cellText=cell2, colLabels=hdr2, loc="upper center", cellLoc="center", colWidths=w2); tb2.auto_set_font_size(False); tb2.set_fontsize(6.5)
    tb2.scale(1, 1.5)
    H = SUM["headline"]
    fig.text(0.01, 0.13, f"headline (T41 c excluded: identical to T41 a through close + 64 rows in 14 / 15 runs): obstacle runs with a close command {H['obstacle_n']}, clear {H['clear_n']}; "
             f"switch away from the fruit before the first close: obstacle {sum(v.get('fail', 0) + v.get('success', 0) for k, v in H['switch_away_before_close']['obstacle'].items() if k == 'True')} runs; "
             f"first switch-away after the first close in {H['switch_away_first_after_first_close']} of {len(H['switch_away_runs'])} switch-away runs (targets {H['switch_away_first_targets']}, dominant camera {H['switch_away_first_dominant_camera']})", fontsize=7)
    fig.text(0.01, 0.10, "E0 (policy only, no AG3S / TO) outcomes: obstacle " + ", ".join(f"{k} {v}" for k, v in sorted(SUM["outcomes"]["E0_obstacle"].items()))
             + "; clear " + ", ".join(f"{k} {v}" for k, v in sorted(SUM["outcomes"]["E0_clear"].items())), fontsize=7)
    fig.text(0.01, 0.07, "obstacle runs - switch away before attach x outcome: " + xt("switch_away_vs_outcome"), fontsize=7)
    fig.text(0.01, 0.045, "obstacle runs - fruit in target-free layer at close x outcome: " + xt("infield_vs_outcome"), fontsize=7)
    pc = SUM["pairs_counts"]
    fig.text(0.01, 0.02, f"same episode / seed pairs (obstacle vs clear) n {pc['n']}: switch away {pc['obst_switch']} vs {pc['clear_switch']}, fruit in field at close "
             f"{pc['obst_infield']} vs {pc['clear_infield']}, miss/slip {pc['obst_fail']} vs {pc['clear_fail']}, TO deflection at close median "
             f"{pc['obst_defl_median']:.1f} vs {pc['clear_defl_median']:.1f} mm", fontsize=7)
    fig.suptitle("T43 U - identity switch, fruit in the obstacle field, TO deflection vs pick outcome (records only)", fontsize=9)
    fig.savefig(FIG / "t43u-table.png", dpi=130); plt.close(fig)
    side("t43u-table", {"groups": G, "switch_away_vs_outcome": SUM["switch_away_vs_outcome"], "infield_vs_outcome": SUM["infield_vs_outcome"],
                        "pairs_counts": pc, "headline": SUM["headline"], "outcomes": SUM["outcomes"]})


if __name__ == "__main__":
    fig_table()
    fig_timeline(sys.argv[1].split(","))
    fig_scene(sys.argv[2].split(","))
    print("ok")
