"""T43 DX real-scene figures (verifier, CPU only). One process per snapshot.
usage: T43W_SNAP=<snap> fig_scene_dx.py apple  KEY T OUT_PNG      (problem 1: event row t of raw/dx1_t43*.json)
       T43W_SNAP=<snap> fig_scene_dx.py finger KEY T OUT_PNG      (problem 2: event of raw/dx2_t43*.json)
apple : state = the row's reported state (dx1 'state': pre = row t qpos, post = row t+1 qpos). Panels: (a) MuJoCo render (free camera
        at the contact) with held query spheres at the actual EE pose (blue) and at the TO-planned pose of refined row k (orange);
        (b) location map (top view x-y: hurdle geoms, apple, slice line); (c) recorded fine ESDF (executing chunk) on the vertical plane
        through the bar contact point perpendicular to the bar axis, with the d = 0 contour, true bar circle, held-sphere / plane
        intersections (actual, planned) and the apple mesh vertices within 3 mm of the plane.
finger: state = physics replay sub-step of the contact (dx2 method). Panels: (a) head camera (zed_left) render at the observation state
        (planning row of the executing chunk) with pb marked; (b) location map with the target-free carve-out ball circle (its z-plane
        section at pb height), hurdle, target points; (c) full-layer fine ESDF and (d) target-free fine ESDF on the plane through pb
        perpendicular to the bar axis, d = 0 contour, true bar circle, finger contact point, ball section."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys, glob
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p')); sys.path.insert(0, _ag3s_workspace_path(''))
import numpy as np, mujoco  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
import hold_w as HW  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

mode, key, t, out_png = sys.argv[1], sys.argv[2], int(sys.argv[3]), pathlib.Path(sys.argv[4])
assert not out_png.exists(), f"{out_png} exists"
R = _ag3s_workspace_path('outputs/verify/T43/DX/raw/')
cl = HW.load_client(); a = C.run_entry(key); rd = pathlib.Path(a["run_dir"]); side = a["side"]
ctrl, plan = C.frames(rd); chunks = C.segment_chunks(a["server_segment"])
man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
seed = int(np.load(rd / "traj.npz")["policy_seed"]); stt = rd.parent / f"{rd.name}.start"
cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; hurdle = sorted(L["obstacle_manager"].collision_geom_ids)
SD = "L" if side == "left" else "R"; ee_name = f"EE_BODY_{SD}"; qi, c0 = (C.LEFT_Q, 0) if side == "left" else (C.RIGHT_Q, 8)
figstyle.use_korean()
side_json = {"mode": mode, "key": key, "t": t, "snapshot": str(HW.HY), "script": __file__}


def setq(q):
    d.qpos[:] = q; mujoco.mj_forward(m, d)


def add_sphere(scn, c, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([r, 0, 0]), np.asarray(c, float), np.eye(3).reshape(-1), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def render_free(lookat, az, el, dist, spheres):
    rr = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat; cam.azimuth = az; cam.elevation = el; cam.distance = dist
    rr.update_scene(d, camera=cam)
    for c, r, rgba in spheres:
        add_sphere(rr.scene, c, r, rgba)
    img = rr.render(); rr.close(); return img


def plane_basis(h, p):
    Rg = d.geom_xmat[h].reshape(3, 3); ax = Rg[:, 2]
    w = np.array([0, 0, 1.0]); u = np.cross(w, ax); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0])
    w = np.cross(ax, u); return ax, u, w


def slice_grid(z, pre, Tb, p0, u, w, half=0.12, n=241):
    s = np.linspace(-half, half, n); U, W = np.meshgrid(s, s)
    P = p0 + U[..., None] * u + W[..., None] * w
    vals = C.samp(z[f"{pre}_distance"], z[f"{pre}_origin"], z[f"{pre}_voxel_size"], C.to_base(Tb, P.reshape(-1, 3))).reshape(U.shape)
    return s, vals


def draw_slice(ax, s, vals, title, circles, pts, vmax=0.08):
    im = ax.imshow(1e3 * vals, origin="lower", extent=[1e3 * s[0], 1e3 * s[-1], 1e3 * s[0], 1e3 * s[-1]], cmap=figstyle.sequential_cmap(), vmin=-20, vmax=1e3 * vmax)
    ax.contour(1e3 * s, 1e3 * s, 1e3 * vals, levels=[0.0], colors=[figstyle.INK], linewidths=1.2)
    for (cu, cw, r, col, ls, lab) in circles:
        ax.add_patch(plt.Circle((1e3 * cu, 1e3 * cw), 1e3 * r, fill=False, ec=col, ls=ls, lw=1.6, label=lab))
    for (pu, pw, col, mk, lab) in pts:
        ax.plot(1e3 * pu, 1e3 * pw, mk, color=col, ms=7, label=lab)
    ax.set_xlabel("u (mm, horizontal, perpendicular to bar)"); ax.set_ylabel("z (mm)"); ax.set_title(title, fontsize=9)
    ax.set_xlim(1e3 * s[0], 1e3 * s[-1]); ax.set_ylim(1e3 * s[0], 1e3 * s[-1]); ax.set_aspect("equal")
    return im


def loc_map(ax, p0, ax_dir, extra):
    for h in hurdle:
        c = d.geom_xpos[h]; Rg = d.geom_xmat[h].reshape(3, 3); hl = m.geom_size[h][1]
        a0, a1 = c - hl * Rg[:, 2], c + hl * Rg[:, 2]
        ax.plot([1e3 * a0[0], 1e3 * a1[0]], [1e3 * a0[1], 1e3 * a1[1]], "-", color=figstyle.CATEGORICAL[3], lw=4, label="hurdle (bar / legs)" if h == hurdle[0] else None)
    perp = np.cross([0, 0, 1.0], ax_dir); perp = perp / np.linalg.norm(perp)
    q0, q1 = p0 - 0.12 * perp, p0 + 0.12 * perp
    ax.plot([1e3 * q0[0], 1e3 * q1[0]], [1e3 * q0[1], 1e3 * q1[1]], "--", color=figstyle.INK, lw=1.5, label="slice plane (vertical)")
    ax.plot(1e3 * p0[0], 1e3 * p0[1], "x", color=figstyle.CATEGORICAL[7], ms=9, mew=2, label="contact (bar side)")
    for f in extra:
        f(ax)
    ax.set_aspect("equal"); ax.set_xlabel("world x (mm)"); ax.set_ylabel("world y (mm)"); ax.set_title("location map (top view)", fontsize=9)
    ax.legend(fontsize=7, loc="best")


if mode == "apple":
    E = next(e for f in glob.glob(R + "dx1_t43*.json") for e in json.load(open(f))["events"] if e["key"] == key and any(r["t"] == t for r in e["rows"]))
    row = next(r for r in E["rows"] if r["t"] == t)
    q = np.asarray(ctrl[t]["qpos"] if row["state"] == "pre" else ctrl[t + 1]["qpos"], float); setq(q)
    seq = row["chunk_seq"]; k = row["step_in_chunk"]; z = np.load(chunks[seq], allow_pickle=True); s_ = json.loads(str(z["summary_json"]))
    used, sets, fl, bc = C.held_sets(s_); Te = C.body_T(m, d, ee_name); Ca, ra = C.place(Te, sets[used])
    qp = np.asarray(plan[seq]["qpos"], float).copy(); qp[qi] = np.asarray(z["refined_chunk"], float)[k, c0:c0 + 7]
    setq(qp); Tp = C.body_T(m, d, ee_name); Cp, rp = C.place(Tp, sets[used]); setq(q)
    h = next(g for g in hurdle if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == row["hurdle_geom"]) if row["hurdle_geom"] else hurdle[0]
    pb = np.asarray(row["world_points"]["bar_surface"]); pa = np.asarray(row["world_points"]["apple_contact"]); Tb = C.T_world_base(m, d)
    axd, u, w = plane_basis(h, pb); pax = np.asarray(row["world_points"]["bar_axis"])
    blue = list(np.r_[np.array(matplotlib.colors.to_rgb(figstyle.CATEGORICAL[0])), 0.35]); orange = list(np.r_[np.array(matplotlib.colors.to_rgb(figstyle.CATEGORICAL[1])), 0.35])
    az = float(np.degrees(np.arctan2(axd[1], axd[0])))
    img = render_free(pb, az + 90.0, -30, 0.75, [(c, r, blue) for c, r in zip(Ca, ra)] + [(c, r, orange) for c, r in zip(Cp, rp)])
    fig = plt.figure(figsize=(15, 5.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.9, 1.0])
    a0 = fig.add_subplot(gs[0]); a0.imshow(img); a0.axis("off")
    a0.set_title(f"{key.split('/')[2]} t={t}: held spheres actual (blue) vs TO-planned row k={k} (orange)", fontsize=8.5)
    a1 = fig.add_subplot(gs[1])
    ga = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom"); mid = int(m.geom_dataid[ga])
    av = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float); V = d.geom_xpos[ga] + av @ d.geom_xmat[ga].reshape(3, 3).T
    loc_map(a1, pb, axd, [lambda ax: ax.plot(1e3 * V[:, 0], 1e3 * V[:, 1], ".", ms=1.5, color=figstyle.CATEGORICAL[7], label="apple mesh"),
                          lambda ax: ax.plot(1e3 * Te[0, 3], 1e3 * Te[1, 3], "o", color=figstyle.CATEGORICAL[0], label="EE actual"),
                          lambda ax: ax.plot(1e3 * Tp[0, 3], 1e3 * Tp[1, 3], "o", color=figstyle.CATEGORICAL[1], label="EE planned (row k)")])
    a2 = fig.add_subplot(gs[2]); sgrid, vals = slice_grid(z, "esdf", Tb, pb, u, w)
    circ = [((pax - pb) @ u, (pax - pb) @ w, float(m.geom_size[h][0]), figstyle.CATEGORICAL[3], "-", "true bar section")]
    for (CC, rr_, col, lab) in ((Ca, ra, figstyle.CATEGORICAL[0], "held actual"), (Cp, rp, figstyle.CATEGORICAL[1], "held planned")):
        first = True
        for c, r in zip(CC, rr_):
            dn = (c - pb) @ axd
            if abs(dn) < r:
                circ.append(((c - pb) @ u, (c - pb) @ w, float(np.sqrt(r * r - dn * dn)), col, "-" if lab == "held actual" else "--", lab if first else None)); first = False
    near = np.abs((V - pb) @ axd) < 0.003
    pts = [((pa - pb) @ u, (pa - pb) @ w, figstyle.CATEGORICAL[7], "*", "apple contact point")]
    im = draw_slice(a2, sgrid, vals, f"recorded fine ESDF (chunk seq {seq}, 5 mm), plane through bar contact", circ, pts)
    a2.plot(1e3 * ((V[near] - pb) @ u), 1e3 * ((V[near] - pb) @ w), ".", ms=2, color=figstyle.CATEGORICAL[7])
    a2.legend(fontsize=6.5, loc="lower left"); fig.colorbar(im, ax=a2, fraction=0.046, label="d (mm)")
    figstyle.style_axes(fig, [a1, a2]); fig.tight_layout()
    side_json.update({"row": {k_: row[k_] for k_ in ("state", "chunk_seq", "step_in_chunk", "slip_mm", "slip_deg", "cover_frac", "contact_pt_sd_mm", "held_true_bar_mm", "held_planned_true_bar_mm",
                                                      "apple_planned_bar_mm", "ee_plan_vs_actual_mm", "held_clear_row_k_mm", "held_clear_min_mm", "held_set")},
                      "held_actual_world_m": Ca.tolist(), "held_planned_world_m": Cp.tolist(), "held_radii_m": ra.tolist(), "bar_surface_world_m": pb.tolist(),
                      "slice": {"origin_world_m": pb.tolist(), "u": u.tolist(), "w": w.tolist(), "bar_axis": axd.tolist(), "half_m": 0.12},
                      "esdf_bar_surface": row["esdf"]["bar_surface"], "held_actual_esdf": row.get("held_actual_esdf")})
else:
    E = next(e for f in glob.glob(R + "dx2_t43*.json") for e in json.load(open(f))["events"] if e["key"] == key and e["t"] == t)
    pb = np.asarray(E["pb_world"]); pf = np.asarray(E["pf_world"]); seq = E["exec_seq"]
    z = np.load(chunks[seq], allow_pickle=True)
    setq(np.asarray(plan[seq]["qpos"], float)); Tb = C.T_world_base(m, d)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    sc = TransportScene.attach(m, d); rr = sc._renderer_for(); rr.update_scene(d, camera="zed_left"); rgb = rr.render().copy()
    fr = sc.capture("zed_left"); K = fr.camera_intrinsics; Tc = np.linalg.inv(fr.T_base_cam); pbb = C.to_base(Tb, pb)[0]; pc_ = Tc[:3, :3] @ pbb + Tc[:3, 3]
    uu, vv = K[0, 0] * pc_[0] / pc_[2] + K[0, 2], K[1, 1] * pc_[1] / pc_[2] + K[1, 2]
    hname = E["hurdle_geom"]; h = next(g for g in hurdle if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "") == hname)
    axd, u, w = plane_basis(h, pb); pax = C.geom_axis_point(m, d, h, pb)
    es = E["esdf"]; ball = es.get("ball"); tp = np.asarray(z["target_points"], float); tpw = tp @ Tb[:3, :3].T + Tb[:3, 3]
    fig = plt.figure(figsize=(17, 5.0)); gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 0.9, 1, 1])
    a0 = fig.add_subplot(gs[0]); a0.imshow(rgb); a0.plot(uu, vv, "x", color=figstyle.CATEGORICAL[7], ms=12, mew=2.5); a0.axis("off")
    a0.set_title(f"{key.split('/')[2]} t={t}: head cam at chunk {seq} observation; x = contact", fontsize=8.5)
    a1 = fig.add_subplot(gs[1])
    extra = [lambda ax: ax.plot(1e3 * tpw[:, 0], 1e3 * tpw[:, 1], ".", ms=2, color=figstyle.CATEGORICAL[4], label="AG3S target points")]
    if ball:
        bc = np.asarray(ball["centre_m"]) @ Tb[:3, :3].T + Tb[:3, 3]; br = 1e-3 * ball["radius_mm"]
        hz = abs(pb[2] - bc[2]); rs = np.sqrt(max(br * br - hz * hz, 0))
        extra.append(lambda ax: ax.add_patch(plt.Circle((1e3 * bc[0], 1e3 * bc[1]), 1e3 * rs, fill=False, ec=figstyle.CATEGORICAL[6], lw=1.6, label="target-free carve-out ball (section at contact z)")))
    loc_map(a1, pb, axd, extra)
    circ = [((pax - pb) @ u, (pax - pb) @ w, float(m.geom_size[h][0]), figstyle.CATEGORICAL[3], "-", "true bar section")]
    if ball:
        dn = (bc - pb) @ axd
        if abs(dn) < br:
            circ.append(((bc - pb) @ u, (bc - pb) @ w, float(np.sqrt(br * br - dn * dn)), figstyle.CATEGORICAL[6], "--", "carve-out ball section"))
    pts = [((pf - pb) @ u, (pf - pb) @ w, figstyle.CATEGORICAL[0], "*", "finger contact point")]
    for j, (pre, ttl) in enumerate((("esdf", "full layer"), ("esdf_free", "target-free layer (asked by authorized finger rows)"))):
        axj = fig.add_subplot(gs[2 + j]); sgrid, vals = slice_grid(z, pre, Tb, pb, u, w)
        im = draw_slice(axj, sgrid, vals, f"recorded fine ESDF, {ttl}", circ, pts)
        axj.legend(fontsize=6.5, loc="lower left")
    fig.colorbar(im, ax=axj, fraction=0.046, label="d (mm)")
    figstyle.style_axes(fig, [a1]); fig.tight_layout()
    side_json.update({"exec_seq": seq, "pb_world_m": pb.tolist(), "pf_world_m": pf.tolist(), "head_pixel_uv": [float(uu), float(vv)], "esdf": {k_: es[k_] for k_ in es if k_ in ("fine", "free", "coarse", "composed_full_pb_mm", "composed_free_pb_mm", "ball", "target_points", "target_free", "manipulated")},
                      "cameras": E["cameras"], "cloud": E["cloud"], "slice": {"origin_world_m": pb.tolist(), "u": u.tolist(), "w": w.tolist(), "bar_axis": axd.tolist(), "half_m": 0.12}})
fig.savefig(out_png, dpi=130, facecolor=figstyle.SURFACE)
json.dump(side_json, open(out_png.with_suffix(".json"), "w"), indent=1, default=float)
print("wrote", out_png)
