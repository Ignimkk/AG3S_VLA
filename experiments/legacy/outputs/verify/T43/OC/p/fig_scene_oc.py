"""T43 OC real-scene figures (verifier, CPU only, recorded data). One process, V8 snapshot.
usage: T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v8 PYTHONPATH=${AG3S_PROJECT_ROOT}-t43v8 fig_scene_oc.py RAW_JSON KEY:T [KEY:T ...]
State = observation state of the executing chunk (planning-row qpos of exec_seq; the depth the server integrated for that chunk).
t43oc-scene.png    : per contact: free-camera render at pb (pb = bar point at the contact, physics replay; red marker) + head / left wrist /
                     right wrist RGB renders with the bar pixel classes overlaid.
t43oc-depthmask.png: per contact: the three depth images (grey) with the same bar pixel classes.
  bar pixel classes (oc_contact.py rules): bar px = segmentation body is a hurdle body and back-projected point within 10 mm of the hurdle;
    near = |P - pb| < 30 mm; robot-filtered = self_filter_mask (build_robot_model(scene), code-default inflation) -> depth 0 in the TSDF input;
    slot 0 = bar, near, kept; slot 2 = bar, elsewhere, kept; slot 1 = bar, robot-filtered.
t43oc-esdf.png     : per contact: (a) location map, base-frame top view (x-y): fine-window footprint, bar capsule, slice line x = pb_x, pb;
                     (b) recorded full-layer fine ESDF (5 mm, executing chunk) on the plane x = pb_x (y-z), d = 0 contour, true bar circle,
                     pb, finger / arm contact point pf; (c) coarse ESDF (20 mm) on the same plane."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/DX/p'))
import numpy as np, mujoco  # noqa: E402
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import hold_w as HW  # noqa: E402
import dx_common as C  # noqa: E402
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, set_finger_joints_from_scene  # noqa: E402
from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model  # noqa: E402
from AG3S.ag3s.stages.robot_filter import self_filter_mask  # noqa: E402
from AG3S.ag3s.stages.reconstruction import backproject  # noqa: E402
from AG3S.ag3s.config import PointCloudConfig  # noqa: E402
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

C.ANALYSES["V8"] = C.T43 / "V8/analysis/analysis.json"; C.ARM_SRC["V8"] = "V8"
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t43'))
OUT = {n: FIG / f"t43oc-{n}.png" for n in ("scene", "depthmask", "esdf")}
for p in OUT.values():
    assert not p.exists() and not p.with_suffix(".json").exists(), f"{p} exists"
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r"); CAMLAB = {"zed_left": "head (zed_left)", "wrist_cam_l": "left wrist", "wrist_cam_r": "right wrist"}
PCFG = PointCloudConfig(range_max=2.0); K_ = figstyle.CATEGORICAL
raw = {}
for f in sys.argv[1].split(","):
    for e in json.load(open(f))["events"]:
        raw[(e["key"], e["t"])] = e
specs = [(s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in sys.argv[2:]]
figstyle.use_korean()
cl = HW.load_client(); n = len(specs)
fS, aS = plt.subplots(n, 4, figsize=(17, 3.4 * n)); fD, aD = plt.subplots(n, 3, figsize=(13, 3.4 * n)); fE, aE = plt.subplots(n, 3, figsize=(15, 4.2 * n))
aS, aD, aE = np.atleast_2d(aS), np.atleast_2d(aD), np.atleast_2d(aE)
side = {"script": __file__, "definitions": __doc__, "contacts": []}
for i, (key, t) in enumerate(specs):
    e = raw[(key, t)]; a = C.run_entry(key); rd = pathlib.Path(a["run_dir"]); ctrl, plan = C.frames(rd); chunks = C.segment_chunks(a["server_segment"])
    man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile"); ep = man["scene"]["episode_index"]
    seed = int(np.load(rd / "traj.npz")["policy_seed"]); stt = rd.parent / f"{rd.name}.start"
    cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in stt.read_text().splitlines() if l.startswith("obstacle=") and l != "obstacle=:"), None)
    L = HW.reset(cl, ep, seed, prof, cfg); m, d = L["m"], L["d"]; hurdle = sorted(L["obstacle_manager"].collision_geom_ids); hb = {int(m.geom_bodyid[g]) for g in hurdle}
    sx = e["exec_seq"]; d.qpos[:] = np.asarray(plan[sx]["qpos"], float); mujoco.mj_forward(m, d)
    pb = np.asarray(e["pb_world"]); pf = np.asarray(e["pf_world"]); Tb = C.T_world_base(m, d); pbb = C.to_base(Tb, pb)[0]; pfb = C.to_base(Tb, pf)[0]
    lab = f"{key.split('/')[-1]} t={t} ({e['mjcf_body']}), chunk {sx}"
    rec = {"key": key, "t": t, "exec_seq": sx, "pb_world": pb.tolist(), "pf_world": pf.tolist(), "pb_base": pbb.tolist(), "cams": {}}
    # (a) free camera
    rr = mujoco.Renderer(m, 480, 640); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = pb; cam.azimuth = 200; cam.elevation = -25; cam.distance = 0.9; rr.update_scene(d, camera=cam)
    scn = rr.scene; mujoco.mjv_initGeom(scn.geoms[scn.ngeom], int(mujoco.mjtGeom.mjGEOM_SPHERE), np.array([0.012, 0, 0]), pb, np.eye(3).reshape(-1), np.array([0.89, 0.29, 0.28, 1], np.float32)); scn.ngeom += 1
    aS[i, 0].imshow(rr.render()); aS[i, 0].set_title(f"{lab}\nfree camera at pb (red), observation state", fontsize=8); aS[i, 0].axis("off")
    scene = TransportScene.attach(m, d); rm = build_robot_model(scene); set_finger_joints_from_scene(scene, rm); rs = scene.robot_state()
    for j, cn in enumerate(CAMS):
        fr = scene.capture(cn); Kc = fr.camera_intrinsics; T = fr.T_base_cam; cloud = backproject(fr.depth, Kc, T, PCFG); uv = cloud.uv; P = cloud.points
        isbar = np.isin(fr.body_ids[uv[:, 1], uv[:, 0]], list(hb)); Pw = P @ Tb[:3, :3].T + Tb[:3, 3]
        hs = np.full(len(P), 1.0); hs[isbar] = C.hurdle_sd(m, d, hurdle, Pw[isbar]) if isbar.any() else hs[isbar]
        isbar &= hs < 0.010; near = isbar & (np.linalg.norm(Pw - pb, axis=1) < 0.030)
        rf = np.zeros(len(P), bool)
        if isbar.any():
            rf[isbar] = self_filter_mask(P[isbar], rm, rs, PCFG)[0]
        H, W = fr.depth.shape; cls = np.zeros((H, W), int)
        cls[uv[isbar & ~near & ~rf, 1], uv[isbar & ~near & ~rf, 0]] = 2; cls[uv[isbar & near & ~rf, 1], uv[isbar & near & ~rf, 0]] = 1; cls[uv[isbar & rf, 1], uv[isbar & rf, 0]] = 3
        cmap = ListedColormap([(0, 0, 0, 0), K_[0], K_[2], K_[1]]); masked = np.ma.masked_where(cls == 0, cls)
        r2 = mujoco.Renderer(m, H, W); r2.update_scene(d, camera=cn); rgb = r2.render(); r2.close()
        Tc = np.linalg.inv(T); pc_ = Tc[:3, :3] @ pbb + Tc[:3, 3]; uvp = None
        if pc_[2] > 0:
            u = Kc[0, 0] * pc_[0] / pc_[2] + Kc[0, 2]; v = Kc[1, 1] * pc_[1] / pc_[2] + Kc[1, 2]
            if 0 <= u < W and 0 <= v < H:
                uvp = (float(u), float(v)); occl = bool(fr.depth[int(v), int(u)] < pc_[2] - 0.005)
        cnt = {"near_kept": int((isbar & near & ~rf).sum()), "elsewhere_kept": int((isbar & ~near & ~rf).sum()), "robot_filtered": int((isbar & rf).sum()),
               "pb_px": uvp, "pb_occluded": None if uvp is None else occl}
        rec["cams"][cn] = cnt
        for ax, img, title in ((aS[i, j + 1], rgb, "RGB"), (aD[i, j], fr.depth, "depth")):
            if title == "depth":
                ax.imshow(np.where(fr.depth > 0, fr.depth, np.nan), cmap="Greys", vmin=0.2, vmax=2.0)
            else:
                ax.imshow(img)
            ax.imshow(masked, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
            if uvp is not None:
                ax.plot(*uvp, marker="o", ms=9, mfc="none" if occl else K_[7], mec=K_[7], mew=1.5)
            ax.set_title(f"{CAMLAB[cn]} {title}: bar near pb kept {cnt['near_kept']} px, elsewhere {cnt['elsewhere_kept']}, robot-filtered {cnt['robot_filtered']}\n"
                         f"pb {'outside image' if uvp is None else ('occluded at pixel (hollow)' if occl else 'visible (filled)')}", fontsize=7)
            ax.axis("off")
    scene.close(); rr.close()
    # ESDF slice (x = pb_x, base frame)
    with np.load(chunks[sx], allow_pickle=True) as z:
        for j, (nm, pre) in enumerate((("fine", "esdf"), ("coarse", "esdf_coarse"))):
            o = np.asarray(z[f"{pre}_origin"], float); vx = float(z[f"{pre}_voxel_size"]); sh = np.asarray(z[f"{pre}_shape"])
            ys = np.arange(pbb[1] - 0.15, pbb[1] + 0.15, vx / 2); zs = np.arange(pbb[2] - 0.15, pbb[2] + 0.15, vx / 2)
            YY, ZZ = np.meshgrid(ys, zs); Pq = np.c_[np.full(YY.size, pbb[0]), YY.ravel(), ZZ.ravel()]
            D = 1e3 * C.samp(z[f"{pre}_distance"], o, vx, Pq).reshape(YY.shape)
            ax = aE[i, j + 1]; im = ax.imshow(D, origin="lower", extent=[1e3 * ys[0], 1e3 * ys[-1], 1e3 * zs[0], 1e3 * zs[-1]], cmap=figstyle.sequential_cmap(), vmin=-30, vmax=150)
            if np.nanmin(D) < 0 < np.nanmax(D):
                ax.contour(1e3 * YY, 1e3 * ZZ, D, levels=[0], colors=[figstyle.INK], linewidths=1.0)
            cbw = C.to_base(Tb, np.asarray(e["bar"]["centre_w"]))[0]; th = np.linspace(0, 2 * np.pi, 100); rb = e["bar"]["radius_mm"]
            ax.plot(1e3 * cbw[1] + rb * np.cos(th), 1e3 * cbw[2] + rb * np.sin(th), color=K_[1], lw=1.5, label="true bar section")
            ax.plot(1e3 * pbb[1], 1e3 * pbb[2], "o", color=K_[7], ms=5, label="pb (bar contact point)"); ax.plot(1e3 * pfb[1], 1e3 * pfb[2], "x", color=K_[3], ms=7, mew=2, label="pf (robot contact point)")
            dpb = float(C.samp(z[f"{pre}_distance"], o, vx, pbb)[0]) * 1e3
            ax.set_title(f"{nm} ESDF ({vx * 1e3:.0f} mm) plane x = {pbb[0] * 1e3:.0f} mm, chunk {sx}; d(pb) = {dpb:.1f} mm; black = d 0 contour", fontsize=8)
            ax.set_xlabel("y base (mm)"); ax.set_ylabel("z base (mm)"); plt.colorbar(im, ax=ax, fraction=0.04, label="d (mm)")
            if j == 0:
                ax.legend(fontsize=6, loc="lower left")
            rec[f"{nm}_d_pb_mm"] = dpb; rec[f"{nm}_origin"] = o.tolist(); rec[f"{nm}_voxel"] = vx; rec[f"{nm}_upper"] = (o + vx * (sh - 1)).tolist()
            rec[f"{nm}_slice_min_mm"] = float(np.nanmin(D)); rec[f"{nm}_slice_max_mm"] = float(np.nanmax(D))
        ax = aE[i, 0]; o = rec["fine_origin"]; u_ = rec["fine_upper"]
        ax.add_patch(plt.Rectangle((1e3 * o[0], 1e3 * o[1]), 1e3 * (u_[0] - o[0]), 1e3 * (u_[1] - o[1]), fill=False, ec=K_[0], lw=1.2, label="fine window (5 mm)"))
        cbw = C.to_base(Tb, np.asarray(e["bar"]["centre_w"]))[0]; axw = np.asarray(e["bar"]["axis_w"]); hl = e["bar"]["half_len_mm"]
        ax.plot([1e3 * cbw[0] - hl * axw[0], 1e3 * cbw[0] + hl * axw[0]], [1e3 * cbw[1] - hl * axw[1], 1e3 * cbw[1] + hl * axw[1]], color=K_[1], lw=4, solid_capstyle="round", label="bar (capsule axis)")
        ax.axvline(1e3 * pbb[0], color=figstyle.INK_2, ls="--", lw=1, label="slice plane x = pb_x (viewed along +x)")
        ax.plot(1e3 * pbb[0], 1e3 * pbb[1], "o", color=K_[7], ms=5, label="pb"); ax.set_aspect("equal")
        ax.set_xlabel("x base (mm)"); ax.set_ylabel("y base (mm)"); ax.set_title(f"{lab}\nlocation map (top view)", fontsize=8); ax.legend(fontsize=6, loc="upper left")
    side["contacts"].append(rec)
for f_, a_ in ((fS, aS), (fD, aD), (fE, aE)):
    figstyle.style_axes(f_, a_)
fS.suptitle("T43 OC: observation state of the executing chunk, bar pixel classes (blue = bar near pb kept, green = bar elsewhere kept, orange = robot-filtered)", fontsize=9)
fD.suptitle("T43 OC: per-camera depth with bar pixel classes (blue = near pb kept, green = elsewhere kept, orange = robot-filtered; red ring = pb)", fontsize=9)
fE.suptitle("T43 OC: recorded ESDF on the plane through pb perpendicular to the bar (executing chunk)", fontsize=9)
for nm, f_ in (("scene", fS), ("depthmask", fD), ("esdf", fE)):
    f_.tight_layout(); f_.savefig(OUT[nm], dpi=130); OUT[nm].with_suffix(".json").write_text(json.dumps({**side, "figure": nm}, indent=1, default=float))
    print(OUT[nm])
