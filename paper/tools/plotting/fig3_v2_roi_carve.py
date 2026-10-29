"""Figure 3 v2 — visible free-space carving in the region of interest (dynamic map, serving default since 2026-10-24). CPU, read only.

  cd /mnt/dev/AG3S_VLA-development; MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python \
      paper/tools/plotting/fig3_v2_roi_carve.py --request 47 --az 10 --el -55 --dist 0.38 --rays 5 --outdir paper/figures/paperbanana/figure3/assets_v2

One recorded closed-loop run with carving off (AG3S-TASK-DYNAMIC-MAP configuration B). At one request before the grasp
closes (the palm has pushed the apple), the recorded main fine field (`esdf_distance`) is drawn as surface voxels
(|value| <= half a voxel) on a MuJoCo render of that request's qpos. The "on" panel applies the serving judgement
`curobo_builder.seen_through` with the serving-default footprint settings (footprint, pad 0 px, robot pixels ignored, margin =
fine truncation + voxel as recorded by the carving-on run) to the same voxels and the same recorded depth of all three cameras;
voxels it frees are dropped. Serving applies it to the TSDF voxels inside the region of interest every frame; here it is applied
offline to the recorded surface voxels (same rule, same input). Robot pixels: MuJoCo segmentation of the robot geoms at the same
qpos (serving uses its FK-sphere depth mask).
Voxel labels follow `task_dynamic_map_eval.run_episode` exactly: stale = > 15 mm from every current GT surface and robot surface
sample; red = stale and within 15 mm of a pose the apple has left by > 10 mm. Region of interest = the evaluation's region.
The carving-on panel also draws depth rays of one camera through the apple's earlier pose (recorded depth, K, T_base_cam).

`--paired` (used for the figure): instead of the offline judgement, the "on" panel is the recorded field of the carving-on run
(B_ROI_FPR = the serving default --roi-carve --roi-carve-footprint --roi-carve-footprint-pad-px 0
--roi-carve-footprint-robot ignore) of the same scene and seed at the same request, chosen where both runs are in the same grasp
state after the apple was displaced. One frame alone frees few voxels where the hand hides the old site; serving carves every frame.
"""
import argparse, json, pathlib, sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261024T034606+0900_8da37119afd3_31d9b16e3c97")
KITS = {"B": RUN / "verify/AG3S-TASK-DYNAMIC-MAP", "B_ROI_FPR": RUN / "verify/AG3S-TASK-DYNAMIC-MAP-E4"}
MARGIN_FINE = 0.035   # recorded esdf_stats.roi_carve.fine_tsdf.margin_m of the carving-on run (5 mm voxel + 30 mm truncation)
XML = RUN / "snapshot_tdm3/robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml"
RED, BLUE, RAY = (0.80, 0.10, 0.10, 1.0), (0.18, 0.46, 0.71, 0.55), (1.0, 0.80, 0.10, 0.9)


def labelled_voxels(conf, ep, k):
    from scipy.spatial import cKDTree
    from AG3S.ag3s.experiments.studies import task_dynamic_map_eval as E
    pdir, chunks = E.episodes_of(KITS[conf], conf)[ep]
    gt = E.scene_truth(str(XML), E.obstacle_profile(pdir.parent.parent))
    robot = E.RobotSurface(gt)
    t_idx = [i for i, (_, n, _) in enumerate(gt.kind) if n == E.TARGET]
    crate_idx = [i for i, (kd, _, _) in enumerate(gt.kind) if kd == "crate"]
    hist, roi0, first_t = [], None, None
    for j, cf, state in chunks:
        step = np.load(pdir / f"step_{j:05d}.npz")
        samples, planes = gt.at(np.asarray(step["qpos"], np.float64))
        tgt = np.vstack([samples[i] for i in t_idx])
        if roi0 is None:
            allp = np.vstack([tgt] + [samples[i] for i in crate_idx])
            roi0 = (allp.min(0) - E.ROI_PAD, allp.max(0) + E.ROI_PAD); first_t = tgt.copy()
        c = tgt.mean(0)
        if j == k:
            boxes = [roi0, (tgt.min(0) - E.ROI_PAD, tgt.max(0) + E.ROI_PAD)]

            def in_roi(p, grow=0.0):
                m = np.zeros(len(p), bool)
                for lo, hi in boxes:
                    m |= np.all((p >= lo - grow) & (p <= hi + grow), axis=1)
                return m
            gt_all = np.vstack([s[in_roi(s, 0.05)] for s in samples if len(s)])
            rob = robot.at(); rob = rob[in_roi(rob, 0.05)]
            with np.load(cf) as z:
                grid = E._grid(z, "esdf"); summary = json.loads(str(z["summary_json"]))
            occ = E._surface(grid); occ = occ[in_roi(occ)]
            d_gt = cKDTree(gt_all).query(occ)[0]
            for n, off in planes:
                d_gt = np.minimum(d_gt, np.abs(occ @ n - off))
            res = (d_gt > E.STALE_MM / 1000) & (cKDTree(rob).query(occ)[0] > E.STALE_MM / 1000)
            old = [p for cc, p in hist if np.linalg.norm(cc - c) > E.MOVED_MM / 1000]
            t_old = np.zeros(len(occ), bool)
            if old and res.any():
                t_old[res] = cKDTree(np.vstack(old)).query(occ[res])[0] <= E.STALE_MM / 1000
            return dict(gt=gt, step=step, occ=occ, res=res, t_old=t_old, roi=roi0, tgt=tgt, first_t=first_t, state=state,
                        voxel=float(grid[2]), roi_carve=(summary.get("esdf_stats") or {}).get("roi_carve"), chunk=str(cf),
                        shift_mm=float(np.linalg.norm(c - first_t.mean(0)) * 1000))
        if all(np.linalg.norm(cc - c) > 0.005 for cc, _ in hist):
            hist.append((c, tgt))
    raise SystemExit(f"request {k} not found for {conf} ep{ep}")


def rays_through(step, cam, pts, n=5):
    """Depth rays of `cam` through `pts` (base frame), ending at the recorded depth: (origin, end) pairs."""
    d = np.asarray(step[f"depth_{cam}"], np.float64) / 1000.0
    K = np.asarray(step[f"K_{cam}"], np.float64); T = np.asarray(step[f"T_base_cam_{cam}"], np.float64)
    o = T[:3, 3]; out = []
    sel = pts[np.linspace(0, len(pts) - 1, min(n, len(pts))).astype(int)] if len(pts) else pts
    for p in sel:
        pc = (p - o) @ T[:3, :3]
        u, v = int(round(K[0, 0] * pc[0] / pc[2] + K[0, 2])), int(round(K[1, 1] * pc[1] / pc[2] + K[1, 2]))
        if not (0 <= v < d.shape[0] and 0 <= u < d.shape[1]) or d[v, u] <= 0:
            continue
        ray = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0]) * d[v, u]
        out.append((o, T[:3, :3] @ ray + o, float(d[v, u] - pc[2])))
    return out


def render(info, cam, extras, W=1600, H=1200):
    import mujoco
    gt = info["gt"]; m, dd = gt.m, gt.d
    m.vis.global_.offwidth, m.vis.global_.offheight = max(W, 1600), max(H, 1200)
    R, o = gt.base_pose
    w = lambda p: R @ np.asarray(p) + o  # noqa: E731  base -> world
    r = mujoco.Renderer(m, H, W, max_geom=40000)
    c = mujoco.MjvCamera(); c.lookat[:], c.distance, c.azimuth, c.elevation = w(cam["lookat"]), cam["dist"], cam["az"], cam["el"]
    r.update_scene(dd, c); sc = r.scene
    for kind, a, b, rgba in extras:
        if sc.ngeom >= sc.maxgeom:
            break
        g = sc.geoms[sc.ngeom]
        if kind == "box":
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.array([b, b, b]), w(a), np.eye(3).ravel(), np.asarray(rgba, np.float32))
        else:
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, b, w(a[0]), w(a[1]))
        sc.ngeom += 1
    img = r.render().copy(); r.close()
    return img


def box_edges(lo, hi):
    c = [np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return [(c[i], c[j]) for i in range(8) for j in range(i + 1, 8) if np.sum(c[i] != c[j]) == 1]


class Cam:
    def __init__(self, name, step, robot_mask):
        self.name = name
        self.depth = np.asarray(step[f"depth_{name}"], np.float64) / 1000.0
        self.camera_intrinsics = np.asarray(step[f"K_{name}"], np.float64)
        self.T_base_cam = np.asarray(step[f"T_base_cam_{name}"], np.float64)
        self.robot_mask = robot_mask


def robot_masks(info):
    """Robot pixels of each policy camera at this qpos: MuJoCo segmentation, geoms under body 'base'."""
    import mujoco
    gt = info["gt"]; m, d = gt.m, gt.d
    in_robot = np.zeros(m.ngeom, bool)
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        while b > 0 and b != gt.base:
            b = int(m.body_parentid[b])
        in_robot[g] = b == gt.base
    out = {}
    for cam in ("zed_left", "wrist_cam_l", "wrist_cam_r"):
        h, w = info["step"][f"depth_{cam}"].shape
        r = mujoco.Renderer(m, h, w); r.enable_segmentation_rendering(); r.update_scene(d, camera=cam)
        seg = r.render(); r.close()
        gid, typ = seg[..., 0], seg[..., 1]
        mask = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
        mask[mask] = in_robot[gid[mask]]
        out[cam] = mask
    return out


def paired(a):
    from PIL import Image
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    infos = {conf: labelled_voxels(conf, a.ep, a.request) for conf in KITS}
    centre = infos["B"]["first_t"].mean(0)
    cam = {"lookat": centre + np.array([0, 0, 0.03]), "dist": a.dist, "az": a.az, "el": a.el}
    side = {"mode": "paired closed-loop runs", "episode": a.ep, "request": a.request, "kits": {k: str(v) for k, v in KITS.items()},
            "model_xml": str(XML), "script": "paper/tools/plotting/fig3_v2_roi_carve.py", "near_m": a.near,
            "camera": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cam.items()}, "panels": {}}
    names = {"B": "f3v2_roi_carve_off.png", "B_ROI_FPR": "f3v2_roi_carve_on.png"}
    for conf, info in infos.items():
        occ, t_old = info["occ"], info["t_old"]
        keep = np.linalg.norm(occ - centre, axis=1) <= a.near
        ex = [("box", q, info["voxel"] / 2 * 0.9, RED if s else BLUE) for q, s in zip(occ[keep], t_old[keep])]
        ex += [("line", e, 0.0015, (0.35, 0.35, 0.35, 0.9)) for e in box_edges(*info["roi"])]
        rays = []
        if conf == "B_ROI_FPR":
            ft = info["first_t"]; top = ft[ft[:, 2] >= np.quantile(ft[:, 2], 0.5)]
            rays = [r for r in rays_through(info["step"], a.ray_cam, top, n=40) if r[2] > MARGIN_FINE][:a.rays]
            ex += [("line", (o, e), 0.0012, RAY) for o, e, _ in rays]
        Image.fromarray(render(info, cam, ex)).save(out / names[conf])
        rc = info["roi_carve"] if isinstance(info["roi_carve"], dict) else None
        side["panels"][conf] = {"asset": str(out / names[conf]), "chunk": info["chunk"], "state": info["state"],
                                "apple_shift_from_first_pose_mm": round(info["shift_mm"], 1), "surface_voxels_in_roi": int(len(occ)),
                                "stale": int(info["res"].sum()), "stale_at_old_apple_pose": int(t_old.sum()),
                                "drawn": int(keep.sum()), "drawn_red": int((t_old & keep).sum()),
                                "roi_carve_record_fine": (rc or {}).get("fine_tsdf"),
                                "rays": [{"camera": a.ray_cam, "origin": o.round(4).tolist(), "end": e.round(4).tolist(),
                                          "depth_beyond_point_m": round(dz, 4)} for o, e, dz in rays]}
        print(conf, info["state"], "shift", round(info["shift_mm"], 1), "stale_old", int(t_old.sum()), "drawn_red", int((t_old & keep).sum()), "rays", len(rays))
    side["encoding"] = {"blue cubes": "recorded main fine field surface voxels within near_m of the apple's first pose",
                        "red cubes": "stale voxels at a pose the apple has left (task_dynamic_map_eval definition)",
                        "grey lines": "region of interest box (evaluation definition)",
                        "yellow lines": "depth rays of one camera through the apple's first pose that see more than the carving margin "
                                        "past it, ending at the recorded depth (on panel)"}
    side["is_measured_result"] = False
    side["note"] = "Two closed-loop runs of one scene and seed at the same request and grasp state; robot and apple poses differ slightly."
    (out / "f3v2_roi_carve.json").write_text(json.dumps(side, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=211); ap.add_argument("--request", type=int, default=20)
    ap.add_argument("--outdir", required=True); ap.add_argument("--near", type=float, default=0.12)
    ap.add_argument("--ray-cam", default="zed_left"); ap.add_argument("--az", type=float, default=135); ap.add_argument("--el", type=float, default=-30)
    ap.add_argument("--dist", type=float, default=0.55); ap.add_argument("--rays", type=int, default=4)
    ap.add_argument("--paired", action="store_true"); ap.add_argument("--ray-len", type=float, default=0.30)
    ap.add_argument("--look", type=float, nargs=3, default=None, help="look-at point (base frame); default the apple's first pose")
    a = ap.parse_args()
    if a.paired:
        return paired(a)
    from PIL import Image
    from AG3S.ag3s.fields.curobo_builder import seen_through
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    KITS.pop("B_ROI_FPR")
    info = labelled_voxels("B", a.ep, a.request)
    masks = robot_masks(info)
    cams = [Cam(n, info["step"], masks[n]) for n in masks]
    occ, t_old = info["occ"], info["t_old"]
    freed, per = seen_through(occ, cams, MARGIN_FINE, depth_min=0.05, depth_max=3.0, footprint_voxel=info["voxel"],
                              footprint_pad_px=0, footprint_robot="ignore")
    centre = info["first_t"].mean(0)
    cam = {"lookat": (np.asarray(a.look) if a.look else centre + np.array([0, 0, 0.03])), "dist": a.dist, "az": a.az, "el": a.el}
    keep = np.linalg.norm(occ - centre, axis=1) <= a.near
    hv = info["voxel"] / 2
    roi = [("line", e, 0.0008, (0.55, 0.55, 0.55, 0.7)) for e in box_edges(*info["roi"])]
    off = [("box", p, hv * 0.9, RED if s else BLUE) for p, s in zip(occ[keep], t_old[keep])] + roi
    k_on = keep & ~freed
    rays = rays_through(info["step"], a.ray_cam, occ[t_old & freed], n=a.rays)
    on = [("box", p, hv * 0.9, RED if s else BLUE) for p, s in zip(occ[k_on], t_old[k_on])] + roi
    on += [("line", (e + (o - e) / np.linalg.norm(o - e) * a.ray_len, e), 0.0006, RAY) for o, e, _ in rays]   # last ray_len m drawn
    Image.fromarray(render(info, cam, off)).save(out / "f3v2_roi_carve_off.png")
    Image.fromarray(render(info, cam, on)).save(out / "f3v2_roi_carve_on.png")
    side = {"episode": a.ep, "request": a.request, "kit": str(KITS["B"]), "configuration": "B (carving off, recorded)",
            "model_xml": str(XML), "chunk": info["chunk"], "state": info["state"], "script": "paper/tools/plotting/fig3_v2_roi_carve.py",
            "camera": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in cam.items()}, "near_m": a.near,
            "apple_shift_from_first_pose_mm": round(info["shift_mm"], 1),
            "judgement": {"function": "AG3S.ag3s.fields.curobo_builder.seen_through", "margin_m": MARGIN_FINE, "footprint_voxel_m": info["voxel"],
                          "footprint_pad_px": 0, "footprint_robot": "ignore", "depth_range_m": [0.05, 3.0],
                          "robot_mask": "MuJoCo segmentation of robot geoms (serving: FK-sphere depth mask)"},
            "counts": {"surface_voxels_in_roi": int(len(occ)), "stale": int(info["res"].sum()), "stale_at_old_apple_pose": int(t_old.sum()),
                       "freed_by_rule": int(freed.sum()), "stale_old_freed": int((t_old & freed).sum()),
                       "non_stale_freed": int((freed & ~info["res"]).sum()), "per_camera": per,
                       "drawn_off": int(keep.sum()), "drawn_on": int(k_on.sum())},
            "rays": [{"camera": a.ray_cam, "origin": o.round(4).tolist(), "end": e.round(4).tolist(), "depth_beyond_point_m": round(dz, 4)}
                     for o, e, dz in rays],
            "encoding": {"blue cubes": "recorded main fine field surface voxels within near_m of the apple's first pose",
                         "red cubes": "stale voxels at a pose the apple has left (task_dynamic_map_eval definition)",
                         "grey lines": "region of interest box (evaluation definition)",
                         "yellow lines": "depth rays of one camera through freed red voxels, ending at the recorded depth (on panel)"},
            "is_measured_result": False,
            "note": "Same recorded input for both panels; the on panel re-applies the serving judgement offline. Illustration."}
    (out / "f3v2_roi_carve.json").write_text(json.dumps(side, indent=1, default=float))
    print(json.dumps(side["counts"]), "rays", len(rays), "shift", side["apple_shift_from_first_pose_mm"])


if __name__ == "__main__":
    main()
