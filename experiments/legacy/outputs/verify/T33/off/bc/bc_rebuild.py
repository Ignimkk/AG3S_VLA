"""T33 offline (b)+(c) — copy of T32a scratch h3_trace3.py (h3_trace3_orig.py), extended with crate-wall measurements.
Variants (current tree CuroboFieldBuilder, fresh builder each): T32 (H1 held mask + H3 free, fine truncation 30 mm),
T32_nofree (same without H3 free), T32_trunc60 (T32 with fine_truncation_m None = the 60 mm TSDF for both tiers).
Fine window: centred on the RECORDED fine window of the paired server chunk (esdf_origin -> centre, target_centroid placement
with that single point) so the window matches the T30 server / T32 H4 real-chunk numbers.
Added per seq and variant: crate wall d<0 widths (0.5 mm samples, trilinear EsdfField.distance, same as T32 H4 band) on two
lines through the GT crate centre at z = centre + 4 mm: along base x (split at centre: lo = crate_wall_nx, hi = crate_wall_px)
and along base y (lo = crate_wall_ny, hi = crate_wall_py), for the fine and coarse tiers; fine occupied voxels classified
'crate'; slices (fine + coarse, xz through crate centre y and xy at crate centre z + 4 mm) saved to <out>.slices.npz
for the listed seqs.
--- original docstring ---
T32a H3 — after H1, do old traces of the held apple remain in the TSDF?

usage: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python h3_trace.py RUN N_SEQ OUT.json
(RUN e.g. E3b_ep1807_r1 of T30; seqs 1..N_SEQ).

Input: every planning seq of the recorded run re-rendered from its `qpos` (object poses included) with
`TransportScene.attach(...).capture(cam)` for the manifest cameras (the verifier's `T32/h4/render_real.py` recipe),
depth wire-quantised to mm. Robot pixels = GT segmentation of robot bodies (isolates the held-object effect from the
robot sphere filter). Attach = the recorded attach chunk; held geometry = the previous chunk's `target_points`
(T31-diag D3), fitted exactly like `AG3S._held_primitives` (trim → fit_surface_sphere(r ≤ 35.7 mm) → covering_spheres
pad 5 mm), snapshotted in the `ee_left` frame at the attach seq's q.

Variants (one fresh `CuroboFieldBuilder` each, server esdf config: voxel 20 / fine 5 / tsdf 5 mm, max_distance 0.4):
  A60  pre-T32: truncation 60 mm for both tiers (fine_truncation_m None), no held mask, seed exclusion at the snapshot
       points (voxel 10 mm, as `attach_from_target`) — the T31-diag configuration.
  A    = A60 with the H4-fix fine truncation (30 mm)
  B    T32: held spheres in the depth mask (H1) + seed exclusion = held sphere volume above the table (H1)
  C    = B without seed exclusion (is the exclusion still needed?)
Fine window: `FineWindowRequest(manipulated_points=[GT apple centre], placement="target_centroid")`.

Per seq ≥ attach: fine-layer occupied voxels (d ≤ voxel/2) classified by nearest GT collision geom within 30 mm
(T31-diag D4 `classify`): `no_gt_surface_30mm` = phantom (trail); `apple` = at the real apple.
Also: `held_min_mm` = min over held spheres of field(centre) − r (what the optimizer's held rows read, before margin);
`table_spot` = fine/coarse occupied voxels within 40 mm of the apple's seq-1 centre and ≥ 10 mm above the table top.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections, time
import numpy as np
import mujoco

T30 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T30'))
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
FRUIT = {"banana", "orange", "pear"}


class D4:
    """Copies of T31-diag `d4/d4_voxels.py` setup_probe / classify / occupied (same definitions)."""

    @staticmethod
    def setup_probe(_):
        spec = mujoco.MjSpec.from_file(XML)
        b = spec.worldbody.add_body(name="probe", mocap=True)
        b.add_geom(name="probe_g", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-4, 0, 0], contype=0, conaffinity=0)
        m = spec.compile(); d = mujoco.MjData(m)
        pg = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "probe_g")
        mid = int(m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "probe")])
        geoms = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g]) and g != pg]
        root = {g: m.body(m.body_rootid[m.geom_bodyid[g]]).name for g in geoms}
        body = {g: m.body(m.geom_bodyid[g]).name for g in geoms}
        return m, d, pg, mid, geoms, root, body

    @staticmethod
    def classify(P, pts):
        m, d, pg, mid, geoms, root, body = P
        gx = d.geom_xpos[geoms]; rb = m.geom_rbound[geoms]
        out = []; dist = []
        ft = np.zeros(6)
        for p in pts:
            near = np.linalg.norm(gx - p, axis=1) - rb < 0.06
            best, bg = np.inf, None
            d.mocap_pos[mid] = p; mujoco.mj_kinematics(m, d)
            for gi in np.flatnonzero(near):
                g = geoms[gi]
                v = float(mujoco.mj_geomDistance(m, d, pg, g, 0.06, ft))
                if abs(abs(v) - np.linalg.norm(ft[:3] - ft[3:])) > 1e-3 and v < 0.06:
                    continue
                if v < best:
                    best, bg = v, g
            dist.append(best)
            if bg is None or best > 0.03:
                out.append("no_gt_surface_30mm")
            else:
                b = body[bg]
                out.append(b if b in ("crate", "table", "apple") else "other_fruit" if b in FRUIT else
                           "other" if root[bg] in ("office", "shelf", "world") or b in ("office", "shelf", "world") else "robot")
        return out, np.array(dist)

    @staticmethod
    def occupied(layer):
        V = np.asarray(layer.distance_grid, np.float32); vs = layer.grid.voxel_size
        idx = np.argwhere(V <= vs / 2)
        pts = np.asarray(layer.grid.origin) + idx * vs
        return pts, V[tuple(idx.T)]


def main():
    run, nseq, out_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    import torch  # noqa: F401
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder, FineWindowRequest
    from AG3S.ag3s.fields.esdf import CameraDepth
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, is_robot_body
    from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS, RBY1_URDF, UrdfSphereChain, parse_urdf
    from AG3S.ag3s.robot_models.held_object import (covering_spheres, fit_surface_sphere,
                                                         sphere_volume_points)
    from AG3S.ag3s.constraints.attached import trim_outliers, voxel_unique
    from AG3S.ag3s.stages.robot_filter import robot_sphere_mask

    cond = run.split("_")[0]
    rd = T30 / f"{cond}_rec" / run
    man = json.loads((rd / "manifest.json").read_text())
    cams = man["cameras"]
    rows = {int(x["seq"]): x for x in (json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines()
                                       if l.strip()) if x.get("kind") == "planning"}
    A = json.loads((T30 / "p" / f"{cond}_analysis.json").read_text())["runs"][run]
    chunks = [c for c in A["chunks"] if c.get("server_chunk")]
    summ = [json.loads(str(np.load(T30 / f"{cond}_rec" / c["server_chunk"], allow_pickle=False)["summary_json"]))
            for c in chunks]
    ia = next(i for i, s in enumerate(summ) if (s.get("grasp") or {}).get("attach"))
    seq_a = int(chunks[ia]["seq"])
    tp = np.asarray(np.load(T30 / f"{cond}_rec" / chunks[ia - 1]["server_chunk"])["target_points"], np.float64)

    m = mujoco.MjModel.from_xml_path(man["scene"]["model_xml"]); d = mujoco.MjData(m)
    sc = TransportScene.attach(m, d)
    qadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i]) for i in range(m.njnt)}
    robot = UrdfSphereChain(parse_urdf(RBY1_URDF), link_filter=["ee_left"])
    base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    apple_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    crate_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")
    SLICE_SEQS = set(int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else []))
    slices = {}

    def band(layer, p0, u, split, half=0.15, step=0.0005):
        s_ = np.arange(-half, half, step); pts_ = p0[None] + s_[:, None] * u[None]
        g = layer.grid; last = np.asarray(g.origin) + (np.asarray(g.shape) - 1) * g.voxel_size
        ins = np.all((pts_ >= g.origin) & (pts_ <= last), axis=1)
        if not ins.all():
            return {"covered": False, "covered_fraction": float(ins.mean())}
        v = np.asarray(layer.distance(pts_), np.float64)
        a = s_ < split
        return {"covered": True, "width_lt0_mm": float(1000 * step * (v < 0).sum()),
                "width_lt0_mm_lo": float(1000 * step * (v[a] < 0).sum()), "width_lt0_mm_hi": float(1000 * step * (v[~a] < 0).sum()),
                "min_mm_lo": float(1000 * v[a].min()), "min_mm_hi": float(1000 * v[~a].min())}

    def slice_of(layer, cc, plane, half=0.2, step=0.0025):
        ax = np.arange(-half, half + 1e-9, step)
        if plane == "xz":
            A, B = np.meshgrid(ax, ax, indexing="ij"); pts_ = np.stack([cc[0] + A, np.full_like(A, cc[1]), cc[2] + 0.004 + B], -1)
        else:
            A, B = np.meshgrid(ax, ax, indexing="ij"); pts_ = np.stack([cc[0] + A, cc[1] + B, np.full_like(A, cc[2] + 0.004)], -1)
        g = layer.grid; last = np.asarray(g.origin) + (np.asarray(g.shape) - 1) * g.voxel_size
        flat = pts_.reshape(-1, 3); ins = np.all((flat >= g.origin) & (flat <= last), axis=1)
        out = np.full(len(flat), np.nan); out[ins] = np.asarray(layer.distance(flat[ins]), np.float64)
        return out.reshape(A.shape).astype(np.float32)

    def q_of(row):
        return np.asarray([row["qpos"][qadr[j]] for j in DEFAULT_RBY1_JOINTS], np.float64)

    def set_row(row):
        d.qpos[:] = row["qpos"]; mujoco.mj_forward(m, d)
        Rb = d.xmat[base].reshape(3, 3).copy(); pb = d.xpos[base].copy()
        return Rb, pb

    # held spheres (as AG3S._held_primitives) in the ee_left frame
    keep = trim_outliers(tp, 0.020, 3)
    pts = tp[keep] if keep.sum() >= 4 else tp
    fit = fit_surface_sphere(pts, max_radius=0.07146 / 2)
    # snapshot with q at the geometry's capture (the previous request) — `AG3S.attach` does this since
    # T32 (`geometry_state`); the pre-T32 server used q at the attach request (T31-diag: 12–46 mm off)
    qa = q_of(rows[int(chunks[ia - 1]["seq"])])
    Ta = robot.link_pose(qa, "ee_left")
    locals_by_pad, rec = {}, {}
    for pad in (0.005, 0.025):
        spheres, rec[pad] = covering_spheres(pts, fit, pad=pad, max_spheres=4)
        locals_by_pad[pad] = [((c - Ta[:3, 3]) @ Ta[:3, :3], r) for c, r in spheres]
    snap_local = (voxel_unique(pts, 0.010) - Ta[:3, 3]) @ Ta[:3, :3]

    def held_at(q, pad):
        T = robot.link_pose(q, "ee_left")
        return [(T[:3, :3] @ c + T[:3, 3], r) for c, r in locals_by_pad[pad]], snap_local @ T[:3, :3].T + T[:3, 3]

    esdf = {"voxel_size": 0.020, "max_distance": 0.4, "exclude_support_surfaces": False, "backend": "curobo",
            "fine_voxel_size": 0.005, "tsdf_voxel_size": 0.005}
    # mask_pad: held-sphere pad in the depth mask (None = no held mask); excl_pad: pad of the spheres whose
    # volume is seed-excluded (None = points / none per `excl`). T32 = mask fit+25 (query 5 + slip 20),
    # exclusion = the query spheres (fit+5).
    table_top = 0.820
    variants = {"T32": dict(fine_truncation_m=0.030, mask_pad=0.025, excl="none", excl_pad=0.005, free=True),
                "T32_nofree": dict(fine_truncation_m=0.030, mask_pad=0.025, excl="volume", excl_pad=0.005, free=False),
                "T32_trunc60": dict(fine_truncation_m=None, mask_pad=0.025, excl="none", excl_pad=0.005, free=True)}
    # T32a §3.2: with cuRobo + free the pipeline passes NO attached_points (seed exclusion retired); without free it
    # uses the held sphere volume (_held_seed_points) -> T32 excl none, T32_nofree excl volume.
    # pre-grasp fit = last chunk before the closing started (exclusion.grasp_active False), as the pipeline
    kpre = max(k for k in range(ia) if not (summ[k].get("exclusion") or {}).get("grasp_active"))
    tpre = np.asarray(np.load(T30 / f"{cond}_rec" / chunks[kpre]["server_chunk"])["target_points"], np.float64)
    fpre = fit_surface_sphere(tpre, max_radius=0.07146 / 2)
    first_extra = [(c.copy(), r) for c, r in held_at(qa, 0.005)[0]]
    if fpre.accepted:
        first_extra.append((fpre.centre.copy(), fpre.radius + 0.005))
    # destination points = GT crate collision boxes' surfaces (5 mm), base frame (the pipeline's registered crate)
    set_row(rows[1])
    Rb0, pb0 = d.xmat[base].reshape(3, 3).copy(), d.xpos[base].copy()
    crate_pts = []
    for g in range(m.ngeom):
        if m.body(m.geom_bodyid[g]).name != "crate" or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
            continue
        hsz = m.geom_size[g]; R = d.geom_xmat[g].reshape(3, 3); p = d.geom_xpos[g]
        ax = [np.arange(-hsz[i], hsz[i] + 1e-9, 0.005) for i in range(3)]
        G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
        onface = np.any(np.abs(np.abs(G) - hsz) < 0.0026, axis=1)
        crate_pts.append(((G[onface] @ R.T + p) - pb0) @ Rb0)
    crate_pts = np.vstack(crate_pts)
    from scipy.spatial import cKDTree
    crate_tree = cKDTree(crate_pts)
    first_left = [2]

    def free_points(q):
        hs, _ = held_at(q, 0.005)
        sph = list(hs)
        if first_left[0] > 0:
            sph += first_extra
        pts = sphere_volume_points(sph, 0.0025, z_min=table_top + 0.005)
        dd, _ = crate_tree.query(pts, distance_upper_bound=0.010)
        return pts[~np.isfinite(dd)]
    builders = {}
    for name, v in variants.items():
        cfg = AG3SConfig.from_dict({"collision_backend": "esdf",
                                    "esdf": {**esdf, "fine_truncation_m": v["fine_truncation_m"]}}).esdf
        builders[name] = CuroboFieldBuilder(cfg)

    P = D4.setup_probe(None)
    Rb, pb = set_row(rows[1])
    spot = (d.xpos[apple_b] - pb) @ Rb
    res = {"doc": __doc__, "run": run, "attach_seq": seq_a, "attach_t": chunks[ia]["t"], "pregrasp_fit": fpre.record(), "pregrasp_t": chunks[kpre]["t"], "held": {str(k): v for k, v in rec.items()},
           "table_spot_base_m": spot.tolist(), "variants": {k: [] for k in variants}}
    t0 = time.time()
    path_centres = []          # GT apple centres from 2 seqs before attach (the held path, incl. now)
    name_first = True
    keep_pts = {seq_a, seq_a + 2, seq_a + 6, seq_a + 12, seq_a + 20}
    for seq in range(1, nseq + 1):
        row = rows[seq]
        Rb, pb = set_row(row)
        q = q_of(row)
        apple = (d.xpos[apple_b] - pb) @ Rb
        held = seq >= seq_a
        zc = np.load(T30 / f"{cond}_rec" / chunks[seq - 1]["server_chunk"], allow_pickle=False)
        assert int(chunks[seq - 1]["seq"]) == seq
        win_centre = np.asarray(zc["esdf_origin"], float) + 0.5 * np.asarray(zc["esdf_shape"], float) * float(zc["esdf_voxel_size"]) - 0.5 * float(zc["esdf_voxel_size"])
        cc = (d.xpos[crate_b] - pb) @ Rb
        if seq >= seq_a - 2 and name_first:
            path_centres.append(apple.copy())
        frames = []
        for cam in cams:
            fr = sc.capture(cam)
            depth = np.clip(np.rint(np.asarray(fr.depth, np.float64) * 1000.0), 0, 65535) / 1000.0
            ids = [int(b) for b in np.unique(fr.body_ids) if b >= 0 and is_robot_body(fr.body_names.get(int(b)))]
            rmask = np.isin(fr.body_ids, ids)
            K = np.asarray(fr.camera_intrinsics, np.float64); T = np.asarray(fr.T_base_cam, np.float64)
            hmasks = {}
            if held:
                vv, uu = np.nonzero(depth > 0)
                z = depth[vv, uu]
                cp = np.stack([(uu + 0.5 - K[0, 2]) / K[0, 0] * z, (vv + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
                bp = cp @ T[:3, :3].T + T[:3, 3]
                for pad in locals_by_pad:
                    hs_p, _ = held_at(q, pad)
                    inside = robot_sphere_mask(bp, np.array([c for c, _ in hs_p]), np.array([r for _, r in hs_p]), 0.0)
                    hm = np.zeros_like(rmask); hm[vv[inside], uu[inside]] = True
                    hmasks[pad] = hm
            apple_px = np.isin(fr.body_ids, [apple_b])
            frames.append((cam, depth, K, T, rmask, hmasks, apple_px))
        for name, v in variants.items():
            hs, snap = held_at(q, v["excl_pad"]) if held else (None, None)
            cams_d = [CameraDepth(name=c, depth=dep, camera_intrinsics=K, T_base_cam=T,
                                  robot_mask=(rm | hms[v["mask_pad"]]) if (v["mask_pad"] and held) else rm)
                      for c, dep, K, T, rm, hms, _ in frames]
            att = None
            if held and v["excl"] == "points":
                att = snap
            elif held and v["excl"] == "volume":
                att = sphere_volume_points(hs, 0.005, z_min=table_top + 0.005)
            hf = free_points(q) if (held and v["free"]) else None
            field = builders[name].update(cams_d, attached_points=att, held_free_points=hf,
                                          fine_window=FineWindowRequest(manipulated_points=win_centre[None],
                                                                        placement="target_centroid"))
            if seq < seq_a - 1 and seq not in SLICE_SEQS and seq not in (17, 21, 31, 37):
                continue
            fine = sorted(field.layers, key=lambda L: L.grid.voxel_size)[0]
            coarse = sorted(field.layers, key=lambda L: L.grid.voxel_size)[-1]
            fp, _ = D4.occupied(fine)
            P[1].qpos[:] = row["qpos"]; mujoco.mj_kinematics(P[0], P[1])
            cls, dist = D4.classify(P, (fp @ Rb.T + pb) if len(fp) else fp)   # base -> world for the probe
            cls_a = np.asarray(cls)
            ph = fp[cls_a == "no_gt_surface_30mm"] if len(fp) else fp
            past = np.asarray(path_centres) if path_centres else np.zeros((0, 3))
            near_past = (np.linalg.norm(ph[:, None] - past[None], axis=2).min(axis=1) <= 0.045
                         if len(ph) and len(past) else np.zeros(len(ph), bool))
            near_now = (np.linalg.norm(ph - apple, axis=1) <= 0.045) if len(ph) else np.zeros(0, bool)
            e_ph = {"n": int(len(ph)), "near_past_apple_45mm": int((near_past & ~near_now).sum()),
                    "near_current_apple_45mm": int(near_now.sum()),
                    "elsewhere": int((~near_past & ~near_now).sum()),
                    "z_q10_50_90": (np.quantile(ph[:, 2], [.1, .5, .9]).round(3).tolist() if len(ph) else None)}
            if seq in keep_pts:
                res.setdefault("points", {}).setdefault(name, {})[seq] = {
                    "phantom": ph.round(4).tolist(), "apple": apple.tolist(),
                    "held": [[c.tolist(), r] for c, r in hs] if held else None,
                    "past": past.round(4).tolist()}
            e = {"seq": seq, "t": int(row["t_step"]), "held": held, "fine_n_occupied": len(fp), "phantom": e_ph,
                 "held_free": (field.stats.get("held_free") if held else None),
                 "fine_class": dict(collections.Counter(cls)),
                 "apple_moved_mm": round(1000 * float(np.linalg.norm(apple - spot)), 1)}
            if held:
                vals = [float(field.distance(c[None])[0]) - r for c, r in hs]
                e["held_min_mm"] = round(1000 * min(vals), 2)
                e["held_masked_px"] = int(sum(int(hms[v["mask_pad"]].sum()) for *_, hms, _ in frames)) if v["mask_pad"] else 0
                e["apple_px"] = int(sum(int(ap.sum()) for *_, ap in frames))
                e["apple_px_unmasked"] = int(sum(int((ap & ~((rm | hms[v["mask_pad"]]) if v["mask_pad"] else rm)).sum())
                                             for _, _, _, _, rm, hms, ap in frames))
            # table spot: occupied voxels (fine if inside window, else coarse) near the apple's seq-1 place
            for L, key in ((fine, "fine"), (coarse, "coarse")):
                g = L.grid
                vs = g.voxel_size
                ax = np.arange(-0.04, 0.0401, vs)
                grid = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3) + spot
                grid = grid[(np.linalg.norm(grid - spot, axis=1) <= 0.04) & (grid[:, 2] >= table_top + 0.010)]
                last = np.asarray(g.origin) + (np.asarray(g.shape) - 1) * vs
                ins = np.all((grid >= g.origin) & (grid <= last), axis=1)
                if ins.any():
                    dv = np.asarray(L.distance(grid[ins]))
                    e[f"table_spot_{key}"] = {"n": int(ins.sum()), "n_occupied": int((dv <= vs / 2).sum()),
                                              "min_mm": round(1000 * float(dv.min()), 2)}
            for tier, L in (("fine", fine), ("coarse", coarse)):
                e[f"wall_x_{tier}"] = band(L, cc + np.array([0, 0, 0.004]), np.array([1.0, 0, 0]), 0.0)
                e[f"wall_y_{tier}"] = band(L, cc + np.array([0, 0, 0.004]), np.array([0, 1.0, 0]), 0.0)
                if seq in SLICE_SEQS:
                    for pl in ("xz", "xy"):
                        slices[f"{name}/seq{seq:03d}/{tier}/{pl}"] = slice_of(L, cc, pl)
            e["fine_window_centre_base_m"] = win_centre.tolist(); e["crate_centre_base_m"] = cc.tolist()
            e["fine_crate_occupied"] = int(sum(1 for c_ in cls if c_ == "crate"))
            res["variants"][name].append(e)
        if held:
            first_left[0] -= 1
        if seq % 5 == 0:
            print(seq, round(time.time() - t0), {k: (v[-1]["phantom"],
                                                      v[-1].get("held_min_mm")) for k, v in res["variants"].items() if v},
                  flush=True)
    pathlib.Path(out_path).write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(out_path.replace(".json", ".slices.npz"), **slices)


if __name__ == "__main__":
    main()
