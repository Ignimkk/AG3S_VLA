"""AG3S-DYNAMIC-FIELD — offline field replay: does the TSDF follow moving obstacles? (task `AG3S-DYNAMIC-FIELD.task.md` §3)

The recorded observations of a closed-loop run (policy record: per request the three depth images, K, T_base_cam
and qpos) are fed, request by request, into the server's own AG3S (`serve_safe.ag3s_from_args` on the recorded
server argv) — only the field stage: this frame's self-filter, the depth robot mask, the support planes, and
`CuroboFieldBuilder.update` without a target. The same input for every configuration; nothing is closed-loop.
Ground truth: the MuJoCo state of the same request (non-robot collision geoms, surface samples).

  MUJOCO_GL=osmesa .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.dynamic_field_replay \
      --kit <run>/verify/IFG-APPLE-NBC-TARGETSURFACE --configuration NBCTS --out <dir> [--episodes 211 954 …]

Per request and TSDF (the main 5 mm TSDF; the fine TSDF too when the builder has one), on the surface-band voxels:
  stale     band voxels > STALE_MM from every GT surface sample and from the robot spheres;
            `memory` = those within STALE_MM of a movable body's surface at an earlier request
  missed    GT surface samples (fruits, crate, obstacles, the table top near the fruits, inside the TSDF extent)
            whose nearest band voxel is > MISSED_MM away; split: visible now / occluded now but seen before /
            never seen
Fidelity (configuration `off` only): the replay's coarse ESDF vs the recorded `esdf_coarse_distance` of the same
request (same origin and shape), before the first closing request (the server masks the held object after attach).
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import re
import shlex
import sys
import time

import numpy as np

STALE_MM = 15.0
MISSED_MM = 10.0
VISIBLE_MM = 10.0
WS_RADIUS = 0.25   # static geometry counted in the missed metric: within this of the fruits (the workspace)
SAMPLE_SPACING = 0.005
VARIANTS = {
    "off": [],
    "D1_carve": ["--free-space-carve"],
    "D2_robot_free": ["--robot-occupied-free"],
    "D1_D2": ["--free-space-carve", "--robot-occupied-free"],
    "D0_curobo_decay": ["--tsdf-time-decay", "0.99", "--tsdf-frustum-decay", "0.8"],
}
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
FRUITS = ("apple", "banana", "orange", "pear")


# ----------------------------------------------------------------------------------------------- GT
class GroundTruth:
    """Non-robot collision geoms of the MuJoCo model as surface samples in each geom's frame."""

    def __init__(self, model_xml: str):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_path(model_xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.base = base

        def in_robot(b):
            while b > 0:
                if b == base:
                    return True
                b = int(m.body_parentid[b])
            return False

        self.geoms, self.local, self.kind, self.planes = [], [], [], []
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            b = int(m.geom_bodyid[g])
            if in_robot(b):
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                self.planes.append(g)   # the floor: an analytic distance, not samples
                continue
            name = m.body(b).name
            kind = ("fruit" if name in FRUITS else "crate" if name == "crate" else
                    "table" if "table" in name or b == 0 else "obstacle")
            pts = self._sample_local(g)
            if len(pts):
                self.geoms.append(g)
                self.local.append(pts)
                self.kind.append((kind, name, b))
        self.movable = [i for i, (k, n, b) in enumerate(self.kind) if self.m.body_jntnum[b] > 0]
        self.local_full = [p.copy() for p in self.local]

    def reset_crop(self):
        self.local = [p.copy() for p in self.local_full]

    def _sample_local(self, g):
        m, mj = self.m, self.mj
        t, s, h = int(m.geom_type[g]), m.geom_size[g], SAMPLE_SPACING
        if t == mj.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
            f = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
            a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
            n = np.maximum(1, np.ceil(area / (h * h)).astype(int))
            rng = np.random.default_rng(g)
            idx = np.repeat(np.arange(len(f)), n)
            r1, r2 = rng.random(len(idx)), rng.random(len(idx))
            sq = np.sqrt(r1)
            return np.vstack([v, (1 - sq)[:, None] * a[idx] + (sq * (1 - r2))[:, None] * b[idx] + (sq * r2)[:, None] * c[idx]])
        if t == mj.mjtGeom.mjGEOM_BOX:
            out = []
            for ax in range(3):
                o = [i for i in range(3) if i != ax]
                u = np.arange(-s[o[0]], s[o[0]] + 1e-9, h)
                w = np.arange(-s[o[1]], s[o[1]] + 1e-9, h)
                U, W = np.meshgrid(u, w, indexing="ij")
                for sign in (-1.0, 1.0):
                    p = np.zeros((U.size, 3))
                    p[:, ax] = sign * s[ax]
                    p[:, o[0]], p[:, o[1]] = U.ravel(), W.ravel()
                    out.append(p)
            return np.vstack(out)
        if t in (mj.mjtGeom.mjGEOM_CYLINDER, mj.mjtGeom.mjGEOM_CAPSULE):
            r, hz = s[0], s[1]
            th = np.arange(0, 2 * np.pi, h / max(r, 1e-3))
            z = np.arange(-hz, hz + 1e-9, h)
            T, Z = np.meshgrid(th, z, indexing="ij")
            side = np.stack([r * np.cos(T).ravel(), r * np.sin(T).ravel(), Z.ravel()], 1)
            rr = np.arange(0, r + 1e-9, h)
            R, T2 = np.meshgrid(rr, th, indexing="ij")
            disk = np.stack([(R * np.cos(T2)).ravel(), (R * np.sin(T2)).ravel()], 1)
            caps = [np.c_[disk, np.full(len(disk), sz * hz)] for sz in (-1.0, 1.0)]
            return np.vstack([side, *caps])
        if t == mj.mjtGeom.mjGEOM_SPHERE:
            n = max(20, int(4 * np.pi * s[0] ** 2 / (h * h)))
            rng = np.random.default_rng(g)
            v = rng.normal(size=(n, 3))
            return s[0] * v / np.linalg.norm(v, axis=1, keepdims=True)
        return np.zeros((0, 3))

    def at(self, qpos, only=None):
        """Surface samples (base frame) per kept geom at this qpos, and the floor planes `(n, offset)` (base frame).

        `only`: indices to transform (the others come back empty) — static geoms are transformed once per episode."""
        mj = self.mj
        self.d.qpos[:] = qpos
        mj.mj_forward(self.m, self.d)
        R, o = self.d.xmat[self.base].reshape(3, 3), self.d.xpos[self.base]
        self.base_pose = (R.copy(), o.copy())
        want = None if only is None else set(only)
        out = []
        for i, (g, p) in enumerate(zip(self.geoms, self.local)):
            if want is not None and i not in want:
                out.append(np.zeros((0, 3)))
                continue
            Rg, og = self.d.geom_xmat[g].reshape(3, 3), self.d.geom_xpos[g]
            out.append(((p @ Rg.T + og) - o) @ R)
        planes = []
        for g in self.planes:
            n = R.T @ self.d.geom_xmat[g].reshape(3, 3)[:, 2]
            p0 = R.T @ (self.d.geom_xpos[g] - o)
            planes.append((n, float(n @ p0)))
        return out, planes

    def crop_static(self, qpos, lo, hi, pad=0.05):
        """Keep only the samples of static geoms inside `[lo − pad, hi + pad]` (base frame) — once per run."""
        samples, _ = self.at(qpos)
        for i, (p, (kind, name, b)) in enumerate(zip(samples, self.kind)):
            if self.m.body_jntnum[b] == 0:
                keep = np.all((p >= lo - pad) & (p <= hi + pad), axis=1)
                self.local[i] = self.local[i][keep]


# ----------------------------------------------------------------------------------------------- replay
def robot_state_from_qpos(m, qpos):
    import mujoco
    from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in DEFAULT_RBY1_JOINTS]
    return np.asarray(qpos, np.float64)[adr]


def observations(step, robot_state, j):
    from AG3S.ag3s.experiments.sources.mujoco_source import CAMERA_MOUNTS
    from AG3S.ag3s.types import CameraID, CameraObservation
    out = []
    for cam in CAMS:
        if f"depth_{cam}" not in step.files:
            continue
        out.append(CameraObservation(
            camera_id=CameraID.parse(CAMERA_MOUNTS[cam][1]),
            depth=np.asarray(step[f"depth_{cam}"], np.float64) / 1000.0,
            camera_intrinsics=np.asarray(step[f"K_{cam}"], np.float64),
            T_base_cam=np.asarray(step[f"T_base_cam_{cam}"], np.float64),
            robot_state=robot_state, timestamp=float(j)))
    return out


def episodes_of(kit: pathlib.Path, conf: str):
    """{episode: (policy-record dir, [(request index, chunk file), …])} — server chunks split by policy seed."""
    runs = {}
    for r in sorted(glob.glob(str(kit / "runs" / conf / f"{conf}_ep*_s*"))):
        mo = re.search(r"_ep(\d+)_s(\d+)$", r)
        if mo and pathlib.Path(r).is_dir():
            runs[int(mo.group(2))] = (int(mo.group(1)), pathlib.Path(r) / "ag3s" / "run_0000")
    out = {}
    for sdir in sorted(glob.glob(str(kit / "runs" / "server_*" / "server_constraints" / "run_0000"))):
        seg, j = None, -1
        for f in sorted(glob.glob(sdir + "/chunk_*.npz")):
            with np.load(f) as z:
                s = json.loads(str(z["summary_json"]))
            seed = (s.get("policy_seed") or {}).get("episode")
            j = j + 1 if seed == seg else 0
            seg = seed
            if seed in runs:
                ep, pdir = runs[seed]
                out.setdefault(ep, (pdir, []))[1].append((j, f, (s.get("grasp") or {}).get("state")))
    return out


def run_episode(argv, variant_flags, pdir, chunks, gt: GroundTruth, capture=None):
    """`capture`: an optional `{request index: None}` dict, filled with that request's band voxels and labels (figures)."""
    from scipy.spatial import cKDTree
    from AG3S.ag3s.stages.reconstruction import backproject
    from AG3S.ag3s.stages.support_surface import fit_support_surfaces
    from AG3S.ag3s.types import PointCloud
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(argv + variant_flags)
    ag3s = serve_safe.ag3s_from_args(args, plan_horizon_steps=None, rows_per_step=None)
    cfg = ag3s.config
    if ag3s._esdf_builder is None:   # the pipeline makes it on its first field frame (`_build_esdf`)
        from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder
        ag3s._esdf_builder = CuroboFieldBuilder(cfg.esdf)
    builder = ag3s._esdf_builder
    fmap = None   # the measured finger opening, as SafePolicy sets it (`_apply_finger_state`, T29)
    for model in (ag3s.constraint_robot_model, ag3s.robot_model):
        if model is not None and getattr(model, "param_joint_names", ()):
            from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
            fmap = FingerJointMap(model)
            break
    rows = []
    gt.reset_crop()
    seen_flags = None   # made after the static geoms are cropped to the TSDF extent (first frame)
    static_cache, static_tree, ws, mem_keys = {}, None, None, np.zeros((0, 3), np.int64)
    first_close = next((j for j, _, st in chunks if st in ("closing", "held")), None)
    for j, chunk_file, state in chunks:
        sp = pdir / f"step_{j:05d}.npz"
        if not sp.exists():
            continue
        step = np.load(sp)
        qpos = np.asarray(step["qpos"], np.float64)
        rs = robot_state_from_qpos(gt.m, qpos)
        obs = observations(step, rs, j)
        if fmap is not None:   # 16D state gripper columns 7 · 15 (client convention, 1 = open)
            st = np.asarray(step["state"], np.float64).reshape(-1)
            ag3s.set_finger_joints(fmap.joint_values({"left": float(st[7]), "right": float(st[15])}))
        ag3s._frame_guard_radius = 0.0
        ag3s._frame_guard_centre = None
        ag3s._frame_filter_config = ag3s._config_with_guard_radius(0.0)
        ag3s._mask_guard_px, ag3s._frame_depth_masks = [], []
        ag3s._prepare_frame_filter()
        cams = ag3s._depth_cameras_from(obs, None, None, None, rs)
        pts = []
        for c in cams:   # the cloud the support planes are fitted on: robot pixels removed
            bp = backproject(c.depth, c.camera_intrinsics, c.T_base_cam, cfg.pointcloud)
            keep = (np.ones(len(bp), bool) if c.robot_mask is None or bp.uv is None
                    else ~np.asarray(c.robot_mask, bool)[bp.uv[:, 1], bp.uv[:, 0]])
            pts.append(np.asarray(bp.points, np.float64)[keep])
        cloud = np.vstack(pts) if pts else np.zeros((0, 3))
        surfaces, _ = fit_support_surfaces(PointCloud(points=cloud), cfg.support_surface, timestamp=0.0)
        rfree = (ag3s._robot_free_spheres(rs, surfaces, None)
                 if bool(getattr(cfg.esdf, "robot_occupied_free", False)) else None)
        t0 = time.perf_counter()
        field = builder.update(cams, target_points=None, exclude_target=False, frame_index=j,
                               **({} if rfree is None else {"robot_free_spheres": rfree}))
        dt = (time.perf_counter() - t0) * 1000.0
        # --- band voxels of each TSDF
        tsdfs = {"main": builder._mapper.integrator}
        if getattr(builder, "_fine_mapper", None) is not None:
            tsdfs["fine"] = builder._fine_mapper.integrator

        def extent(itg):
            vs = float(itg.voxel_size)
            half = 0.5 * np.asarray(itg.grid_shape, np.float64)[::-1] * vs   # grid_shape (D, H, W) → x, y, z
            c = np.asarray(itg.origin.detach().cpu(), np.float64).reshape(-1)[:3]
            return c - half, c + half

        if seen_flags is None:
            # once per episode: static geoms cropped to the TSDF extent, one tree; the workspace samples the
            # missed metric reads (movable geoms whole; static ones within WS_RADIUS of the fruits at this frame)
            gt.crop_static(qpos, *extent(tsdfs["main"]))
            samples0, _ = gt.at(qpos)
            static_ids = [i for i, (_, _, b) in enumerate(gt.kind) if gt.m.body_jntnum[b] == 0]
            st_pts = [samples0[i] for i in static_ids if len(samples0[i])]
            static_tree = cKDTree(np.vstack(st_pts)) if st_pts else None
            fr0 = [samples0[i] for i, (k, _, _) in enumerate(gt.kind) if k == "fruit" and len(samples0[i])]
            fr0_tree = cKDTree(np.vstack(fr0)) if fr0 else None
            ws = []
            for i, p in enumerate(samples0):
                if i in static_ids:
                    ws.append((fr0_tree.query(p)[0] <= WS_RADIUS) if (fr0_tree is not None and len(p)) else np.zeros(len(p), bool))
                else:
                    ws.append(np.ones(len(p), bool))
            seen_flags = [np.zeros(int(m.sum()), bool) for m in ws]
            static_cache = {i: samples0[i] for i in static_ids}
            base0 = gt.base_pose
            mem_keys = np.zeros((0, 3), np.int64)
        samples, planes = gt.at(qpos, only=[i for i in range(len(gt.kind)) if i not in static_cache])
        for i, p in static_cache.items():
            samples[i] = p
        dyn = [samples[i] for i in range(len(samples)) if i not in static_cache and len(samples[i])]
        dyn_tree = cKDTree(np.vstack(dyn)) if dyn else None
        c_robot, r_robot = ag3s._frame_filter_model.sphere_centers_numeric(rs)
        c_robot, r_robot = np.asarray(c_robot).reshape(-1, 3), np.asarray(r_robot).reshape(-1)
        robot_tree = cKDTree(c_robot)
        ws_samples = [p[m] for p, m in zip(samples, ws)]
        vis_now = []   # visibility of each workspace sample now
        for p in ws_samples:
            v = np.zeros(len(p), bool)
            for c in cams:
                if not len(p):
                    break
                T = np.asarray(c.T_base_cam)
                pc = (p - T[:3, 3]) @ T[:3, :3]
                z = pc[:, 2]
                K = np.asarray(c.camera_intrinsics)
                ok = z > 1e-6
                u = np.full(len(p), -1)
                vv = np.full(len(p), -1)
                u[ok] = np.rint(K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]).astype(int)
                vv[ok] = np.rint(K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]).astype(int)
                h, w = c.depth.shape
                ins = ok & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
                ii = np.flatnonzero(ins)
                dd = c.depth[vv[ii], u[ii]]
                good = (dd > 0) & (np.abs(dd - z[ii]) <= VISIBLE_MM / 1000.0)
                if c.robot_mask is not None:
                    good &= ~np.asarray(c.robot_mask, bool)[vv[ii], u[ii]]
                v[ii[good]] = True
            vis_now.append(v)
        mem_tree = cKDTree((mem_keys + 0.5) * SAMPLE_SPACING) if len(mem_keys) else None
        # the static cache assumes the base does not move in the world (checked, recorded)
        base_shift_mm = 1000.0 * float(np.linalg.norm(gt.base_pose[1] - base0[1]))
        row = {"request": j, "state": state, "ms": round(dt, 2), "base_shift_mm": round(base_shift_mm, 3),
               "stats_dynamic": (field.stats or {}).get("dynamic")}
        for name, itg in tsdfs.items():
            _, _, band = builder.occupied_voxels(itg, surface_only=True)
            lo, hi = extent(itg)
            r = {"n_band": int(len(band))}
            if len(band):
                d_gt = np.full(len(band), np.inf)
                if static_tree is not None:
                    d_gt = np.minimum(d_gt, static_tree.query(band)[0])
                if dyn_tree is not None:
                    d_gt = np.minimum(d_gt, dyn_tree.query(band)[0])
                for n, off in planes:
                    d_gt = np.minimum(d_gt, np.abs(band @ n - off))
                kk = min(16, len(c_robot))
                dd, ii = robot_tree.query(band, k=kk)
                dd, ii = dd.reshape(len(band), -1), ii.reshape(len(band), -1)
                d_rob = np.min(dd - r_robot[ii], axis=1)
                stale = (d_gt > STALE_MM / 1000.0) & (d_rob > STALE_MM / 1000.0)
                r["near_robot"] = int((d_rob <= STALE_MM / 1000.0).sum())
                r["stale"] = int(stale.sum())
                if mem_tree is not None and stale.any():
                    r["stale_memory"] = int((mem_tree.query(band[stale])[0] <= STALE_MM / 1000.0).sum())
                else:
                    r["stale_memory"] = 0
                tree = cKDTree(band)
            missed = {"visible": [0, 0], "occluded_seen_before": [0, 0], "never_seen": [0, 0]}
            for i, p in enumerate(ws_samples):
                if not len(p):
                    continue
                inside = np.all((p >= lo) & (p <= hi), axis=1)
                if not inside.any():
                    continue
                far = (tree.query(p[inside])[0] > MISSED_MM / 1000.0) if len(band) else np.ones(int(inside.sum()), bool)
                vn = vis_now[i][inside]
                sb = seen_flags[i][inside]
                for key, sel in (("visible", vn), ("occluded_seen_before", ~vn & sb), ("never_seen", ~vn & ~sb)):
                    missed[key][0] += int((far & sel).sum())
                    missed[key][1] += int(sel.sum())
            r["missed"] = missed
            row[name] = r
            if capture is not None and j in capture and name == "main" and len(band):
                mem = np.zeros(len(band), bool)
                if mem_tree is not None and stale.any():
                    mem[np.flatnonzero(stale)] = mem_tree.query(band[stale])[0] <= STALE_MM / 1000.0
                capture[j] = {"band": band, "stale": stale, "memory": mem,
                              "gt": [(gt.kind[i][0], gt.kind[i][1], samples[i]) for i in range(len(samples))],
                              "robot": (c_robot, r_robot), "state": state}
        if chunk_file is not None and (first_close is None or j < first_close):
            with np.load(chunk_file) as z:
                if "esdf_coarse_distance" in z.files:
                    rec = np.asarray(z["esdf_coarse_distance"], np.float32)
                    layer = field.layers[0]
                    vals = getattr(layer, "values_device", None)
                    vals = (vals.detach().float().cpu().numpy() if vals is not None
                            else np.asarray(layer.distance_grid, np.float32))
                    same = np.allclose(np.asarray(z["esdf_coarse_origin"]), np.asarray(layer.grid.origin)) and \
                        tuple(rec.shape) == tuple(vals.shape)
                    if same:
                        dif = np.abs(vals.astype(np.float16).astype(np.float32) - rec)
                        row["fidelity_coarse"] = {"mean_mm": round(1000 * float(dif.mean()), 4),
                                                  "frac_over_5mm": round(float((dif > 0.005).mean()), 6),
                                                  "max_mm": round(1000 * float(dif.max()), 3)}
                    else:
                        row["fidelity_coarse"] = {"grid_mismatch": True}
        for i in range(len(ws_samples)):
            seen_flags[i] |= vis_now[i]
        mv = [samples[i] for i in gt.movable if len(samples[i])]
        if mv:   # where movable bodies have been, on a SAMPLE_SPACING grid (bounded growth)
            mem_keys = np.unique(np.vstack([mem_keys, np.floor(np.vstack(mv) / SAMPLE_SPACING).astype(np.int64)]), axis=0)
        rows.append(row)
    return rows


def summarise(res):
    out = {}
    for var, eps in res.items():
        agg = {"stale": 0, "stale_memory": 0, "band": 0, "ms": []}
        miss = {k: [0, 0] for k in ("visible", "occluded_seen_before", "never_seen")}
        fid = []
        for rows in eps.values():
            for r in rows:
                agg["ms"].append(r["ms"])
                m = r.get("main") or {}
                agg["stale"] += m.get("stale", 0)
                agg["stale_memory"] += m.get("stale_memory", 0)
                agg["band"] += m.get("n_band", 0)
                for k in miss:
                    miss[k][0] += m.get("missed", {}).get(k, [0, 0])[0]
                    miss[k][1] += m.get("missed", {}).get(k, [0, 0])[1]
                if "mean_mm" in (r.get("fidelity_coarse") or {}):
                    fid.append(r["fidelity_coarse"])
        ms = np.asarray(agg.pop("ms"))
        out[var] = {**agg, "ms_median": round(float(np.median(ms)), 2) if len(ms) else None,
                    "ms_p95": round(float(np.percentile(ms, 95)), 2) if len(ms) else None,
                    "missed": {k: {"n_far": v[0], "n": v[1], "frac": round(v[0] / v[1], 4) if v[1] else None}
                               for k, v in miss.items()},
                    "fidelity_coarse": ({"n": len(fid), "mean_mm_median": round(float(np.median([f["mean_mm"] for f in fid])), 4),
                                         "frac_over_5mm_max": max(f["frac_over_5mm"] for f in fid)} if fid else None)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kit", required=True)
    ap.add_argument("--configuration", required=True)
    ap.add_argument("--argv-file", default=None, help="recorded server argv (default <kit>/pre/argv_<configuration>.txt)")
    ap.add_argument("--episodes", nargs="*", type=int, default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kit = pathlib.Path(a.kit)
    argv = shlex.split(pathlib.Path(a.argv_file or kit / "pre" / f"argv_{a.configuration}.txt").read_text())
    argv = [t for i, t in enumerate(argv) if t not in ("--run-dir", "--record-constraints")
            and (i == 0 or argv[i - 1] not in ("--run-dir", "--record-constraints"))]
    model_xml = argv[argv.index("--model-xml") + 1]
    gt = GroundTruth(model_xml)
    eps = episodes_of(kit, a.configuration)
    sel = sorted(eps) if a.episodes is None else [e for e in a.episodes if e in eps]
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {}
    for var in a.variants:
        res[var] = {}
        for ep in sel:
            pdir, chunks = eps[ep]
            t_ep = time.perf_counter()
            rows = run_episode(argv, VARIANTS[var], pdir, chunks, gt)
            res[var][str(ep)] = rows
            m = [r.get("main", {}) for r in rows]
            print(var, ep, "requests", len(rows), "stale", sum(x.get("stale", 0) for x in m),
                  "memory", sum(x.get("stale_memory", 0) for x in m), "wall_s", round(time.perf_counter() - t_ep, 1), flush=True)
        (out / f"rows_{var}.json").write_text(json.dumps(res[var]))
    summ = summarise(res)
    (out / "summary.json").write_text(json.dumps({"stale_mm": STALE_MM, "missed_mm": MISSED_MM, "visible_mm": VISIBLE_MM,
                                                  "sample_spacing_m": SAMPLE_SPACING, "variants": VARIANTS,
                                                  "episodes": sel, "summary": summ}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
