"""IFG-AUDIT-LTE — how close is the AG3S distance field to the true geometry, and how precise is the target removal? CPU only, read only.

  cd /mnt/dev/AG3S_VLA-development; MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_audit_lte --run <RUN> --fig AG3S/ag3s/docs/figures/ifg-audit [--every 3] [--cache <npz>]

LTE = Link-specific Target Exemption: the cuRobo TSDF/ESDF distance field (main = 20 mm coarse -> 5 mm fine `min` composition) and the
target-free fine layer that only the authorised links (grasping hand fingers + palm) query.

Inputs (all read only)
  kit       : <RUN>/verify/<kit> (default IFG-APPLE-SPLIT), arm <arm> (default SPL): 24 apple scenes
  client    : runs/<arm>/<arm>_ep<E>_s<S>/frames.jsonl planning rows (one per request: seq, t_step, MuJoCo qpos, object_poses);
              manifest.json scene (episode_target_fruit, episode_used_arm)
  server    : replay/runs_<arm>_<arm>_ep<E>_*.json server_chunk_alignment.block -> request k = chunk_dir/chunk_{k+offset:05d}.npz
  field     : `fixed_input_ablation.build_field(z, outside_distance=1.0)` (the offline harness) -> `.distance` (main, coarse->fine) and
              `.target_free_distance` (window formula min(f, max(d, b))); `.answer_tier` names the layer that answered
  GT        : the run's MuJoCo model (snapshot_split model_transport.xml), d.qpos = planning-row qpos, mj_forward. Field frame 'base' =
              MuJoCo world (base body at the origin, measured offset < 0.1 mm). Candidate geoms = non-robot collision geoms (contype or
              conaffinity != 0, not under body 'base'): floor, office walls, table, crate, fruits, shelf, obstacles.
              Signed distance: boxes and planes analytic (exact); every other geom (fruit meshes) `mj_geomDistance` against a 0.1 mm
              probe sphere (contype=conaffinity=0, group 5) whose geom_xpos is written directly; +0.1 mm added back.
              Mesh distance is to MuJoCo's convex hull of the collision mesh (fruits are split into convex parts).
Definitions
  GT_main   : min signed distance over all candidate geoms; in the 'held' phase the held target is left out (the server frees the
              held object from the field on purpose and models it with attached spheres)
  GT_free   : min over candidate geoms except the target fruit
  error     : field - GT (m); > 0 = the field says farther than reality (dangerous), < 0 = nearer (conservative)
  layer in practice: main for every sphere whose `target_permission.target_field_exclude` row is False (unauthorised), target-free for
              authorised rows when the request carries a target-free layer; probes are evaluated on both layers
  query sets: (a) recorded `sphere_centres` (the server robot at refined_chunk[0], i.e. the commanded first plan step; checked 0.0 mm
              on ep119 requests 10 and 16); (b) probes: 48 Fibonacci directions x surface offsets {3,10,17 | 25,35,45 | 60,80 | 150,250} mm
              from the GT surface of the target fruit and of its nearest neighbour fruit (GT surface found by Newton steps on the
              signed distance along each direction from the body's mesh AABB centre)
  bands     : by GT distance: <0 (inside), 0-2, 2-5, 5-10, 10-30, >=30 cm
  visible   : the GT nearest surface point of that layer lies in the frustum of at least one of zed_left (head), wrist_cam_l,
              wrist_cam_r at the planning row (fovy from the model, 4:3 aspect assumed) and an mj_multiRay over geom groups 0-2 from 3 cm
              in front of the lens reaches it (first hit >= length - 8 mm). Current frame only; the TSDF integrates earlier frames too.
  scene type: 'neighbour' when the nearest non-target fruit's collision geometry is within 10 cm of the target's at that request
  phase     : summary grasp.state (searching / latched / closing / held / placed); 'approach' in the exemption = latched
Exemption (latched and closing requests that carry a removal trace)
  removed   : target_free_removed_ijk -> world = trace_origin + ijk * trace_voxel_size
  precision : fraction of removed voxels with GT signed distance to the target <= 5 mm (one voxel; inside counts); the rest are
              labelled by the nearest non-target geom class within 10 mm, else 'target 5-10 mm' or 'no surface within 10 mm'
  surface voxel (main): fine main grid voxel with |D| <= voxel/2 (the ESDF is discrete; seeds are exactly 0)
  target surface voxel : a surface voxel within 6 cm of the target AABB whose GT target distance <= 5 mm and <= every other geom's
  recall    : fraction of target surface voxels whose target-free grid value F > voxel/2 (no longer a surface or interior)
  residual  : target-free grid voxels with F <= voxel/2 (surface or still-negative interior) that are target voxels by the same GT rule
  residual at GT surface : target-free composed distance at GT target surface points (48 dirs) whose GT_free > 15 mm; count < 5 mm
  stale     : the removal ball's surface-fit centre (esdf_stats.target_free.ball.fit) is > 30 mm from the GT target centre (the target
              moved or fell after its geometry was frozen); precision / recall are also given without stale requests
  neighbour lost : at visible GT neighbour surface points where the main layer < 5 mm, fraction whose target-free value >= 5 mm
3D geometry
  extents   : manipulated.admissibility.extents_mm vs `principal_extents` of the GT collision-mesh vertices (same function)
  radius    : manipulated.radius, target_free ball fit radius vs GT mean / max radial surface distance from the AABB centre
  held      : grasp.attached.held.body_cover.body_parent_frame[0] (and spheres_parent_frame) carried by server FK of the parent link
              at the planning-row qpos (server FK = MuJoCo EE_BODY to < 0.1 mm) vs GT target AABB centre; coverage = fraction of GT
              target surface points inside the union of spheres_parent_frame spheres; held voxels = fine main voxels with D <= voxel/2
              that are target voxels, and coarse voxels < 0 within 10 mm of the GT target surface
"""
import argparse, ast, contextlib, datetime, glob, io, json, pathlib, sys, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
SCRIPT = "AG3S/ag3s/experiments/studies/ifg_audit_lte.py"

PHASES = ("searching", "latched", "closing", "held", "placed")
BANDS = ((-np.inf, 0.0, "<0"), (0.0, 0.02, "0-2cm"), (0.02, 0.05, "2-5cm"), (0.05, 0.10, "5-10cm"), (0.10, 0.30, "10-30cm"),
         (0.30, np.inf, ">=30cm"))
OFFSETS = (0.003, 0.010, 0.017, 0.025, 0.035, 0.045, 0.060, 0.080, 0.150, 0.250)
N_DIR = 48
CLASSES = ("target", "neighbour", "fruit_other", "table", "crate", "obstacle", "other")
VOX = 0.005
SETS = ("sphere_unauth", "sphere_auth", "probe_target", "probe_neighbour")


def P(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return x
    return x


def fib_dirs(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


class Scene:
    """MuJoCo model + probe sphere; GT signed distances and camera visibility."""

    def __init__(self, xml):
        import mujoco
        self.mj = mujoco
        spec = mujoco.MjSpec.from_file(str(xml))
        b = spec.worldbody.add_body(name="audit_probe_body", mocap=True)
        b.add_geom(name="audit_probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-4, 0, 0], contype=0, conaffinity=0, group=5)
        self.m = m = spec.compile(); self.d = mujoco.MjData(m)
        self.pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "audit_probe_geom")
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        robot = np.zeros(m.ngeom, bool)
        for g in range(m.ngeom):
            bb = m.geom_bodyid[g]
            while bb not in (0, base):
                bb = m.body_parentid[bb]
            robot[g] = bb == base
        self.robot = robot
        coll = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        self.cand = np.array([g for g in range(m.ngeom) if coll[g] and not robot[g] and g != self.pid], int)
        self.gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom#{g}" for g in range(m.ngeom)]
        self.bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) for g in range(m.ngeom)]
        self.gtype = m.geom_type[self.cand]
        self.is_box = self.gtype == mujoco.mjtGeom.mjGEOM_BOX
        self.is_plane = self.gtype == mujoco.mjtGeom.mjGEOM_PLANE
        self.is_call = ~(self.is_box | self.is_plane)
        self.cams = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c in ("zed_left", "wrist_cam_l", "wrist_cam_r")]
        self.ft = np.zeros(6)

    def set_q(self, qpos):
        q = np.asarray(qpos, np.float64)
        self.d.qpos[:len(q)] = q
        self.mj.mj_forward(self.m, self.d)

    def gd(self, p, g, dm=2.0):
        self.d.geom_xpos[self.pid] = p
        return self.mj.mj_geomDistance(self.m, self.d, self.pid, int(g), dm, self.ft) + 1e-4

    def mesh_world(self, g):
        m = self.m; mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return v @ self.d.geom_xmat[g].reshape(3, 3).T + self.d.geom_xpos[g]

    def body_centre(self, geoms):
        v = np.concatenate([self.mesh_world(g) if self.m.geom_type[g] == self.mj.mjtGeom.mjGEOM_MESH else self.d.geom_xpos[g][None]
                            for g in geoms])
        return 0.5 * (v.min(0) + v.max(0)), v

    def query(self, pts, force=()):
        """(N,3) -> D (N,G) signed distance to every candidate geom (inf where pruned), NP (N,G,3) nearest points (nan where not set).
        A geom is evaluated when its bounding-sphere lower bound is below the smallest upper bound over all geoms, or when forced."""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        N, G = len(pts), len(self.cand)
        D = np.full((N, G), np.inf); NP = np.full((N, G, 3), np.nan)
        xp = self.d.geom_xpos[self.cand]; xm = self.d.geom_xmat[self.cand].reshape(-1, 3, 3)
        size = self.m.geom_size[self.cand]; rb = self.m.geom_rbound[self.cand]
        for j in np.flatnonzero(self.is_box):
            loc = (pts - xp[j]) @ xm[j]
            q = np.abs(loc) - size[j]
            out = np.linalg.norm(np.maximum(q, 0), axis=1); ins = np.minimum(q.max(1), 0)
            D[:, j] = out + ins
            cl = np.clip(loc, -size[j], size[j])
            inside = q.max(1) < 0
            if inside.any():  # push to the nearest face
                k = np.argmax(q[inside], 1); c2 = cl[inside].copy()
                c2[np.arange(len(k)), k] = np.sign(loc[inside][np.arange(len(k)), k]) * size[j][k]
                cl[inside] = c2
            NP[:, j] = cl @ xm[j].T + xp[j]
        for j in np.flatnonzero(self.is_plane):
            n = xm[j][:, 2]; s = (pts - xp[j]) @ n
            D[:, j] = s; NP[:, j] = pts - s[:, None] * n
        dc = np.linalg.norm(pts[:, None, :] - xp[None], axis=2)
        lb = dc - rb[None]; ub = np.where(self.is_call[None], dc + rb[None], D)
        ubmin = ub.min(1)
        forced = np.isin(self.cand, np.asarray(list(force), int))
        need = self.is_call[None] & ((lb <= ubmin[:, None]) | forced[None])
        for i, j in zip(*np.nonzero(need)):
            D[i, j] = self.gd(pts[i], self.cand[j]); NP[i, j] = self.ft[3:6]
        return D, NP

    def visible(self, pts):
        """(N,3) -> bool: in at least one camera frustum (4:3 assumed) and unoccluded (groups 0-2, first hit >= length - 8 mm)."""
        mj = self.mj
        pts = np.asarray(pts, np.float64).reshape(-1, 3); N = len(pts)
        vis = np.zeros(N, bool)
        ok = np.all(np.isfinite(pts), 1)
        if not ok.any():
            return vis
        grp = np.array([1, 1, 1, 0, 0, 0], np.uint8)
        for c in self.cams:
            R = self.d.cam_xmat[c].reshape(3, 3); o = self.d.cam_xpos[c]
            fwd = -R[:, 2]
            loc = (pts - o) @ R
            zc = -loc[:, 2]
            th = np.tan(np.radians(self.m.cam_fovy[c]) / 2)
            inf = ok & (zc > 0.05) & (np.abs(loc[:, 1] / np.maximum(zc, 1e-6)) <= th) & (np.abs(loc[:, 0] / np.maximum(zc, 1e-6)) <= th * 4 / 3)
            idx = np.flatnonzero(inf & ~vis)
            if not len(idx):
                continue
            o2 = o + 0.03 * fwd
            v = pts[idx] - o2; L = np.linalg.norm(v, axis=1); v = v / L[:, None]
            gid = np.zeros(len(idx), np.int32); dist = np.zeros(len(idx))
            mj.mj_multiRay(self.m, self.d, o2, v.ravel(), grp, 1, -1, gid, dist, None, len(idx), 10.0)
            vis[idx] = (dist < 0) | (dist >= L - 0.008)
        return vis


def classes_for(scene, target_g, neigh_g, fruit_g, obst_g):
    cl = []
    for g in scene.cand:
        b = scene.bname[g]
        if g in target_g: cl.append(0)
        elif g in neigh_g: cl.append(1)
        elif g in fruit_g: cl.append(2)
        elif b == "table": cl.append(3)
        elif b == "crate": cl.append(4)
        elif g in obst_g: cl.append(5)
        else: cl.append(6)
    return np.array(cl)


def surface_points(scene, geoms, centre, dirs):
    r = np.full(len(dirs), 0.03)
    for _ in range(8):
        g = np.array([min(scene.gd(centre + u * ri, gg) for gg in geoms) for u, ri in zip(dirs, r)])
        r = np.maximum(r - g, 0.004)
    g = np.array([min(scene.gd(centre + u * ri, gg) for gg in geoms) for u, ri in zip(dirs, r)])
    return centre + dirs * r[:, None], r, np.abs(g) < 1e-3


def stats(err, n_min=1):
    err = np.asarray(err, np.float64); err = err[np.isfinite(err)]
    if len(err) < n_min:
        return {"n": int(len(err))}
    mm = err * 1000
    return {"n": int(len(err)), "median_mm": round(float(np.median(mm)), 2), "p5_mm": round(float(np.percentile(mm, 5)), 2),
            "p95_mm": round(float(np.percentile(mm, 95)), 2), "rms_mm": round(float(np.sqrt(np.mean(mm ** 2))), 2),
            "frac_over_5mm": round(float(np.mean(mm > 5)), 4), "frac_over_10mm": round(float(np.mean(mm > 10)), 4),
            "frac_under_10mm": round(float(np.mean(mm < -10)), 4)}


def band_of(gt):
    out = np.full(len(gt), -1, int)
    for i, (lo, hi, _) in enumerate(BANDS):
        out[(gt >= lo) & (gt < hi)] = i
    return out


def build_robot_quiet(chunk_dir, cache):
    from AG3S.trajopt.experiments.ta_offline import build_robot, server_args
    key = str(pathlib.Path(chunk_dir).parent.parent)
    if key not in cache:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            sa, _ = server_args(pathlib.Path(chunk_dir)); cache[key] = build_robot(sa)
    return cache[key]


# --------------------------------------------------------------------------------------------- per request

def process(scene, z, s, row, ep, k, js, man, dirs, robots, field_builder, want_slice=None):
    from AG3S.ag3s.fields.curobo_field import ANSWER_TIERS
    from AG3S.ag3s.stages.admissibility import principal_extents
    mj = scene.mj; m = scene.m
    gs = js["geom_sets"]
    gid = lambda n: (int(n[5:]) if n.startswith("geom#") else mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, n))  # noqa: E731
    target_g = [gid(n) for n in gs["target"]]
    fruit_g = [gid(n) for n in gs["fruit_other"]]
    obst_g = [gid(n) for n in gs.get("obstacle", [])]
    state = str((s.get("grasp") or {}).get("state"))
    ph = PHASES.index(state) if state in PHASES else -1
    scene.set_q(row["qpos"])
    # nearest neighbour fruit body
    fb = {}
    for g in fruit_g:
        fb.setdefault(scene.bname[g], []).append(g)
    gap = {}
    for b, gg in fb.items():
        gap[b] = min(mj.mj_geomDistance(m, scene.d, t, g, 1.0, scene.ft) for t in target_g for g in gg)
    nb = min(gap, key=gap.get); neigh_g = fb[nb]; ngap = float(gap[nb])
    scene_type = int(ngap <= 0.10)
    cls = classes_for(scene, set(target_g), set(neigh_g), set(fruit_g), set(obst_g))
    is_t = cls == 0
    field = field_builder(z, outside_distance=1.0)
    has_free = field.has_target_free
    hand_g = [gid(n) for n in gs.get("hand", [])]
    hand_gap = None
    if state == "held" and hand_g:
        hand_gap = float(min(mj.mj_geomDistance(m, scene.d, t, g, 0.5, scene.ft) for t in target_g for g in hand_g))
    in_hand = hand_gap is not None and hand_gap <= 0.005
    held = state == "held" and in_hand
    tp = P(s.get("target_permission")) or {}
    excl = tp.get("target_field_exclude")
    auth = np.zeros(len(z["sphere_centres"]), bool) if excl is None else np.asarray(excl, bool)
    tid = {n: i for i, n in enumerate(ANSWER_TIERS)}

    def gt_eval(pts, force=()):
        D, NP = scene.query(pts, force=force)
        Dnt = np.where(is_t[None], np.inf, D)
        a_all = np.argmin(D, 1); a_nt = np.argmin(Dnt, 1)
        r = np.arange(len(pts))
        g_all = D[r, a_all]; g_nt = Dnt[r, a_nt]
        g_main = g_nt if held else g_all; a_main = a_nt if held else a_all
        np_main = NP[r, a_main]; np_free = NP[r, a_nt]
        return D, g_main, g_nt, cls[a_main], cls[a_nt], np_main, np_free

    def field_eval(pts, free_mask):
        dm = field.distance(pts); tm = np.array([tid[t] for t in field.answer_tier(pts)])
        if has_free:
            df = field.target_free_distance(pts); tf = np.array([tid[t] for t in field.answer_tier(pts, target_free=True)])
        else:
            df = np.full(len(pts), np.nan); tf = np.full(len(pts), -1)
        return dm, tm, df, tf

    rows = []
    # (a) sphere centres
    C = z["sphere_centres"].astype(np.float64)
    _, gm, gf, cm, cf, npm, npf = gt_eval(C)
    dm, tm, df, tf = field_eval(C, auth)
    vm = scene.visible(npm); vf = scene.visible(npf)
    setid = np.where(auth & has_free, 1, 0)
    rows.append(dict(set=setid, shell=np.full(len(C), -1), d_main=dm, gt_main=gm, t_main=tm, d_free=df, gt_free=gf, t_free=tf,
                     vis_main=vm, vis_free=vf, cls_main=cm, cls_free=cf))
    # (b) probes around target and neighbour
    ct, vt = scene.body_centre(target_g); cn, _ = scene.body_centre(neigh_g)
    St, rt, okt = surface_points(scene, target_g, ct, dirs)
    Sn, rn, okn = surface_points(scene, neigh_g, cn, dirs)
    for si, (cc, rr, ok) in enumerate(((ct, rt, okt), (cn, rn, okn))):
        Pp = np.concatenate([cc + dirs[ok] * (rr[ok] + o)[:, None] for o in OFFSETS])
        sh = np.concatenate([np.full(int(ok.sum()), i) for i in range(len(OFFSETS))])
        _, gm, gf, cm, cf, npm, npf = gt_eval(Pp)
        dm, tm, df, tf = field_eval(Pp, None)
        rows.append(dict(set=np.full(len(Pp), 2 + si), shell=sh, d_main=dm, gt_main=gm, t_main=tm, d_free=df, gt_free=gf, t_free=tf,
                         vis_main=scene.visible(npm), vis_free=scene.visible(npf), cls_main=cm, cls_free=cf))
    pts = {kk: np.concatenate([r_[kk] for r_ in rows]) for kk in rows[0]}
    n = len(pts["set"])
    pts.update(ep=np.full(n, ep), k=np.full(n, k), phase=np.full(n, ph), scene=np.full(n, scene_type))

    req = {"ep": ep, "k": k, "phase": state, "scene_type": "neighbour" if scene_type else "clear", "neighbour": nb,
           "neighbour_gap_mm": round(ngap * 1000, 1), "has_free": bool(has_free), "n_auth_rows": int(auth.sum()),
           "apple_hand_gap_mm": None if hand_gap is None else round(hand_gap * 1000, 2), "in_hand": bool(in_hand)}
    # GT target geometry
    gt_ext = principal_extents(vt)
    req["gt_target"] = {"extents_mm": [round(float(v) * 1000, 1) for v in np.sort(gt_ext)],
                        "radial_mean_mm": round(float(np.mean(rt[okt])) * 1000, 2), "radial_max_mm": round(float(np.max(rt[okt])) * 1000, 2),
                        "centre": [round(float(v), 5) for v in ct]}
    man_ = P(s.get("manipulated"))
    if isinstance(man_, dict):
        adm = P(man_.get("admissibility")) or {}
        req["manip"] = {"extents_mm": sorted(adm.get("extents_mm") or []) or None, "radius_mm": None if man_.get("radius") is None else round(float(man_["radius"]) * 1000, 2),
                        "state": man_.get("state"), "centroid_err_mm": None if man_.get("centroid") is None else
                        round(float(np.linalg.norm(np.asarray(man_["centroid"]) - ct)) * 1000, 2)}
    es = P(s.get("esdf_stats")) or {}
    req["tsdf_truncation"] = P(es.get("tsdf_truncation")); req["esdf_max_distance_m"] = s.get("esdf_max_distance")
    req["fine_window_overflow"] = (P(es.get("window")) or {}).get("overflow") if isinstance(P(es.get("window")), dict) else None
    tfs = P(es.get("target_free")) or {}
    ball = P(tfs.get("ball")) if isinstance(tfs, dict) else None
    if isinstance(ball, dict):
        fit = ball.get("fit") or {}
        req["ball"] = {"radius_mm": ball.get("radius_mm"), "fit_radius_mm": fit.get("radius_mm"),
                       "fit_centre_err_mm": None if fit.get("centre_m") is None else round(float(np.linalg.norm(np.asarray(fit["centre_m"]) - ct)) * 1000, 2)}

    # fine grids (raw) for voxel-level metrics
    Dg = z["esdf_distance"]; og = np.asarray(z["esdf_origin"], np.float64); vs = float(z["esdf_voxel_size"]); shp = np.array(Dg.shape)
    lo = np.floor((ct - 0.06 - og) / vs).astype(int); hi = np.ceil((ct + 0.06 - og) / vs).astype(int)
    lo = np.clip(lo, 0, shp - 1); hi = np.clip(hi, 0, shp - 1)
    box_ok = np.all(hi > lo)

    def voxel_target_rule(ijk):
        W = og + ijk * vs
        D, _ = scene.query(W, force=target_g)
        dt = D[:, is_t].min(1); dnt = np.where(is_t[None], np.inf, D).min(1)
        return W, dt, dnt, (dt <= VOX) & (dt <= dnt)

    if box_ok:
        sub = Dg[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].astype(np.float32)
        Fsub = z["esdf_free_distance"][lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].astype(np.float32) if "esdf_free_distance" in z.files else None
        cand = np.argwhere((sub <= vs / 2) | ((Fsub <= vs / 2) if Fsub is not None else False)) + lo
        if len(cand):
            _, dt_c, dnt_c, tgt_c = voxel_target_rule(cand)
            Dv = Dg[cand[:, 0], cand[:, 1], cand[:, 2]].astype(np.float32)
            surf = np.abs(Dv) <= vs / 2
            req["main_target_voxels_le_half"] = int(((Dv <= vs / 2) & tgt_c).sum())
            req["main_target_surface_voxels"] = int((surf & tgt_c).sum())
            if Fsub is not None:
                Fv = z["esdf_free_distance"][cand[:, 0], cand[:, 1], cand[:, 2]].astype(np.float32)
                ts = surf & tgt_c
                req["recall"] = None if ts.sum() == 0 else round(float(np.mean(Fv[ts] > vs / 2)), 4)
                req["recall_n"] = int(ts.sum())
                req["free_residual_target_voxels"] = int(((Fv <= vs / 2) & tgt_c).sum())
                req["free_residual_target_voxels_negative"] = int(((Fv < -vs / 2) & tgt_c).sum())
            # held / coarse occupied near the target
        if held:
            req["held_fine_target_voxels"] = req.get("main_target_voxels_le_half")
    # coarse voxels < 0 within 10 mm of the GT target surface
    Cg = z["esdf_coarse_distance"]; oc = np.asarray(z["esdf_coarse_origin"], np.float64); vc = float(z["esdf_coarse_voxel_size"])
    lo2 = np.clip(np.floor((ct - 0.07 - oc) / vc).astype(int), 0, np.array(Cg.shape) - 1)
    hi2 = np.clip(np.ceil((ct + 0.07 - oc) / vc).astype(int), 0, np.array(Cg.shape) - 1)
    neg = np.argwhere(Cg[lo2[0]:hi2[0] + 1, lo2[1]:hi2[1] + 1, lo2[2]:hi2[2] + 1].astype(np.float32) < 0) + lo2
    if len(neg):
        Dn, _ = scene.query(oc + neg * vc, force=target_g)
        req["coarse_neg_near_target"] = int((Dn[:, is_t].min(1) <= 0.010).sum())
    else:
        req["coarse_neg_near_target"] = 0

    # target surface points: main / free distance, residual
    keep = okt
    Dts, _ = scene.query(St[keep])
    gnt_ts = np.where(is_t[None], np.inf, Dts).min(1)
    vis_ts = scene.visible(St[keep])
    fm_ts = field.distance(St[keep])
    far = gnt_ts > 0.015
    req["target_surface"] = {"n": int(keep.sum()), "n_far_from_other": int(far.sum()), "n_visible": int(vis_ts.sum()),
                             "main_lt5": int((fm_ts[far] < 0.005).sum()), "main_lt5_visible": int((fm_ts[far & vis_ts] < 0.005).sum()),
                             "main_median_mm": round(float(np.median(fm_ts[far])) * 1000, 2) if far.any() else None}
    if has_free:
        ff_ts = field.target_free_distance(St[keep])
        req["target_surface"].update({"free_lt5": int((ff_ts[far] < 0.005).sum()), "free_lt5_visible": int((ff_ts[far & vis_ts] < 0.005).sum()),
                                      "free_lt5_occluded": int((ff_ts[far & ~vis_ts] < 0.005).sum()),
                                      "n_far_visible": int((far & vis_ts).sum()),
                                      "free_median_mm": round(float(np.median(ff_ts[far])) * 1000, 2) if far.any() else None})
        # neighbour surface: lost?
        Dns, _ = scene.query(Sn[okn])
        vis_ns = scene.visible(Sn[okn])
        mm_ = field.distance(Sn[okn]); ff_ = field.target_free_distance(Sn[okn])
        inwin = np.array([t in ("target_free", "window_edge", "target_free_coarse") for t in field.answer_tier(Sn[okn], target_free=True)])
        seen = vis_ns & (mm_ < 0.005)
        req["neighbour_surface"] = {"n": int(okn.sum()), "n_visible": int(vis_ns.sum()), "n_visible_main_lt5": int(seen.sum()),
                                    "n_lost": int((seen & (ff_ >= 0.005)).sum()), "n_in_free_window": int(inwin.sum()),
                                    "median_free_minus_main_mm_visible": round(float(np.median(ff_[vis_ns] - mm_[vis_ns])) * 1000, 2) if vis_ns.any() else None}

    # exemption trace
    if "target_free_removed_ijk" in z.files and state in ("latched", "closing", "searching", "placed"):
        tro = np.asarray(z["target_free_trace_origin"], np.float64); trv = float(z["target_free_trace_voxel_size"])
        ex = {"n_removed": int(len(z["target_free_removed_ijk"]))}
        for key, nm in (("target_free_removed_ijk", "removed"), ("target_free_nontarget_kept_ijk", "nontarget_kept"),
                        ("target_free_ambiguous_ijk", "ambiguous"), ("target_free_unowned_kept_ijk", "unowned_kept")):
            ijk = np.asarray(z[key], np.int64)
            if not len(ijk):
                ex[nm] = {"n": 0}; continue
            W = tro + ijk * trv
            D, _ = scene.query(W, force=target_g)
            dt = D[:, is_t].min(1); Dnt = np.where(is_t[None], np.inf, D); a = np.argmin(Dnt, 1); dnt = Dnt[np.arange(len(W)), a]
            lab = {}
            ok5 = dt <= VOX
            lab["target<=5mm"] = int(ok5.sum())
            lab["target<=5mm_and_other<=5mm"] = int((ok5 & (dnt <= VOX)).sum())
            rest = ~ok5
            lab["target5-10mm"] = int((rest & (dt <= 0.010)).sum())
            rest2 = rest & (dt > 0.010)
            for ci, cname in enumerate(CLASSES):
                if ci == 0:
                    continue
                lab[cname] = int((rest2 & (dnt <= 0.010) & (cls[a] == ci)).sum())
            lab["no_surface<=10mm"] = int((rest2 & (dnt > 0.010)).sum())
            ex[nm] = {"n": int(len(W)), "frac_target_le5mm": round(float(ok5.mean()), 4), "frac_target_le7.5mm": round(float((dt <= 0.0075).mean()), 4),
                      "labels": lab}
            if nm == "removed":
                Fv = z["esdf_free_distance"][ijk[:, 0], ijk[:, 1], ijk[:, 2]].astype(np.float32) if "esdf_free_distance" in z.files and np.allclose(tro, og) else None
                if Fv is not None:
                    ex["removed_free_value_le_half_voxel"] = int((Fv <= vs / 2).sum())
                    ex["removed_free_value_negative"] = int((Fv < -vs / 2).sum())
                if want_slice is not None:
                    want_slice["removed_W"] = W
        req["exemption"] = ex

    # held geometry
    att = (s.get("grasp") or {}).get("attached")
    att = P(att)
    if state == "held" and isinstance(att, dict):
        from AG3S.trajopt.experiments.fixed_input_ablation import q_now_from_qpos
        from AG3S.ag3s.constraints.attached import _link_pose_numeric
        robot, qadr = build_robot_quiet(js["server_chunk_alignment"]["block"]["chunk_dir"], robots)
        q = q_now_from_qpos(row["qpos"], qadr)
        Tm = np.asarray(_link_pose_numeric(robot, q, att["parent_link"]))
        hd = P(att.get("held")) or {}
        bc = P(hd.get("body_cover")) or {}
        h = {}
        bpf = bc.get("body_parent_frame") or []
        if bpf:
            c0 = Tm[:3, :3] @ np.asarray(bpf[0]["centre_m"]) + Tm[:3, 3]
            h["body_centre_err_mm"] = round(float(np.linalg.norm(c0 - ct)) * 1000, 2)
            h["body_centre_err_xyz_mm"] = [round(float(v) * 1000, 2) for v in (c0 - ct)]
            h["body_radius_mm"] = bpf[0].get("radius_mm")
        spf = hd.get("spheres_parent_frame") or []
        if spf:
            cs = np.array([Tm[:3, :3] @ np.asarray(x["centre_m"]) + Tm[:3, 3] for x in spf]); rs = np.array([x["radius_mm"] for x in spf]) / 1000
            inside = (np.linalg.norm(St[okt][:, None] - cs[None], axis=2) <= rs[None]).any(1)
            gapo = (np.linalg.norm(St[okt][:, None] - cs[None], axis=2) - rs[None]).min(1)
            h["n_constraint_spheres"] = len(spf)
            h["coverage_frac"] = round(float(inside.mean()), 4)
            h["max_outside_mm"] = round(float(max(gapo.max(), 0)) * 1000, 2)
            h["nearest_sphere_centre_err_mm"] = round(float(np.linalg.norm(cs - ct, axis=1).min()) * 1000, 2)
        h["fit_radius_mm"] = (hd.get("fit") or {}).get("radius_mm")
        h["hand_shift_since_capture_mm"] = hd.get("hand_shift_since_capture_mm")
        req["held"] = h
    if want_slice is not None:
        want_slice.update(field=field, ct=ct, cn=cn, target_g=target_g, neigh_g=neigh_g, cls=cls, C=C, auth=auth & has_free,
                          radii=z["sphere_radii"].astype(np.float64), state=state)
    return pts, req


# --------------------------------------------------------------------------------------------- figures

def save(fig, path, data):
    path = pathlib.Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise SystemExit(f"refusing to overwrite {path}")
    fig.savefig(path, dpi=170, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps(data, indent=1, default=float))
    print("wrote", path)


def fig_section(a, scene, sl, ep, k, row):
    import mujoco
    ct, cn = sl["ct"], sl["cn"]
    e1 = cn - ct; e1[2] = 0; L = float(np.linalg.norm(e1)); e1 /= L
    e2 = np.array([0, 0, 1.0]); nrm = np.cross(e1, e2)
    us = np.arange(-0.10, L + 0.10, 0.0015); zs = np.arange(-0.05, 0.20, 0.0015)
    U, Z = np.meshgrid(us, zs)
    G = ct[None] + U.ravel()[:, None] * e1 + Z.ravel()[:, None] * e2
    D, _ = scene.query(G, force=sl["target_g"] + sl["neigh_g"])
    cls = sl["cls"]
    fm = sl["field"].distance(G).reshape(U.shape)
    ff = sl["field"].target_free_distance(G).reshape(U.shape) if sl["field"].has_target_free else None
    # render head camera
    rnd = mujoco.Renderer(scene.m, 480, 640)
    rnd.update_scene(scene.d, camera="zed_left"); img = rnd.render(); rnd.close()
    c = scene.cams[0]; R = scene.d.cam_xmat[c].reshape(3, 3); o = scene.d.cam_xpos[c]
    f = 0.5 * 480 / np.tan(np.radians(scene.m.cam_fovy[c]) / 2)

    def proj(p):
        loc = (np.atleast_2d(p) - o) @ R
        return np.stack([320 + f * loc[:, 0] / -loc[:, 2], 240 - f * loc[:, 1] / -loc[:, 2]], 1)

    corners = [(us[0], zs[0]), (us[-1], zs[0]), (us[-1], zs[-1]), (us[0], zs[-1]), (us[0], zs[0])]
    poly = []
    for (u0, z0), (u1, z1) in zip(corners[:-1], corners[1:]):
        for t in np.linspace(0, 1, 30):
            poly.append(ct + ((1 - t) * u0 + t * u1) * e1 + ((1 - t) * z0 + t * z1) * e2)
    pp = proj(np.array(poly))
    fig, axs = plt.subplots(1, 4, figsize=(21, 5.4), gridspec_kw={"width_ratios": [1.25, 1, 1, 1], "wspace": 0.42}); FS.style_axes(fig, axs)
    axs[0].imshow(img); axs[0].plot(pp[:, 0], pp[:, 1], color=FS.CATEGORICAL[3], lw=1.6)
    pc = proj(np.array([ct, cn]))
    axs[0].plot(pc[:, 0], pc[:, 1], "o", color=FS.CATEGORICAL[3], ms=4)
    axs[0].set_xlim(0, 640); axs[0].set_ylim(480, 0); axs[0].set_xticks([]); axs[0].set_yticks([])
    axs[0].set_title(T(f"단면 위치 (head 카메라 zed_left): target→이웃 중심을 지나는 수직면", "slice location (head camera): vertical plane through target and neighbour"), fontsize=8.5, loc="left")
    cmap = FS.sequential_cmap(); cmap.set_under(FS.CATEGORICAL[7]); cmap.set_over("#0d366b")
    ext = [us[0] * 1000, us[-1] * 1000, zs[0] * 1000, zs[-1] * 1000]
    cls_cols = {0: FS.CATEGORICAL[1], 1: FS.CATEGORICAL[2], 2: FS.CATEGORICAL[4], 3: FS.CATEGORICAL[3], 4: FS.CATEGORICAL[6], 5: FS.CATEGORICAL[5], 6: FS.INK_2}
    cls_name = {0: T("target 사과", "target apple"), 1: T("이웃 과일", "neighbour fruit"), 2: T("다른 과일", "other fruit"), 3: T("테이블", "table"),
                4: T("상자", "crate"), 5: T("장애물", "obstacle"), 6: T("기타", "other")}
    rem = sl.get("removed_W")
    hv = None if rem is None else (rem - ct) @ nrm
    near = None if rem is None else np.abs(hv) <= VOX / 2
    sph_h = (sl["C"] - ct) @ nrm
    cut = np.abs(sph_h) < sl["radii"]
    data = {"episode": ep, "request": k, "phase": sl["state"], "plane": {"origin_target_centre": ct.round(5).tolist(), "e1_toward_neighbour": e1.round(5).tolist(),
            "e2": [0, 0, 1], "u_range_mm": [round(us[0] * 1000, 1), round(us[-1] * 1000, 1)], "z_range_mm": [round(zs[0] * 1000, 1), round(zs[-1] * 1000, 1)],
            "pixel_mm": 1.5}, "n_removed_in_slice": int(near.sum()) if near is not None else 0, "n_spheres_cut": int(cut.sum()),
            "n_auth_spheres_cut": int((cut & sl["auth"]).sum())}
    for ax, F, name in ((axs[1], fm, T("main 계층 (target 포함)", "main layer (target kept)")), (axs[2], ff, T("target-free 계층 (권한 link 전용)", "target-free layer (authorised links)"))):
        if F is None:
            ax.text(0.5, 0.5, "no target-free layer", transform=ax.transAxes, ha="center"); continue
        im = ax.imshow(F * 1000, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=60, aspect="equal")
        ax.contour(U * 1000, Z * 1000, F * 1000, levels=[0, 5], colors=[FS.INK, FS.INK], linewidths=[0.9, 0.6], linestyles=["-", "--"])
        for ci in sorted(set(cls.tolist())):
            Dc = np.where((cls == ci)[None], D, np.inf).min(1).reshape(U.shape)
            if np.nanmin(Dc) < 0:
                ax.contour(U * 1000, Z * 1000, Dc, levels=[0], colors=[cls_cols[ci]], linewidths=1.6)
                ax.plot([], [], color=cls_cols[ci], lw=1.6, label=T("GT 윤곽: ", "GT: ") + cls_name[ci])
        if near is not None and near.any():
            pr = rem[near] - ct
            ax.scatter(pr @ e1 * 1000, pr @ e2 * 1000, s=5, marker="s", color=FS.CATEGORICAL[0], label=T("제거된 voxel (|h|≤2.5mm)", "removed voxels"), zorder=4)
        for i in np.flatnonzero(cut):
            rr = np.sqrt(sl["radii"][i] ** 2 - sph_h[i] ** 2) * 1000; p = sl["C"][i] - ct
            ax.add_patch(plt.Circle(((p @ e1) * 1000, (p @ e2) * 1000), rr, fill=False, lw=0.6,
                                    color=FS.CATEGORICAL[5] if sl["auth"][i] else FS.INK_2))
        ax.plot([], [], color=FS.CATEGORICAL[5], lw=0.8, label=T("손 구 (권한 link)", "sphere (authorised)"))
        ax.plot([], [], color=FS.INK_2, lw=0.8, label=T("손 구 (기타)", "sphere (other)"))
        ax.plot([], [], color=FS.INK, lw=0.9, label=T("field 0 / 5 mm 등고선", "field 0 / 5 mm"))
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel(T("target→이웃 방향 (mm, target 중심 0)", "along target→neighbour (mm)"), fontsize=8)
        ax.set_ylabel(T("높이 (mm, target 중심 0)", "height (mm)"), fontsize=8)
        ax.set_title(name, fontsize=9, loc="left")
    axs[2].legend(fontsize=6.8, frameon=False, loc="upper center", bbox_to_anchor=(-0.2, -0.17), ncol=4)
    cb = fig.colorbar(im, ax=axs[2], fraction=0.04, pad=0.02, extend="both"); cb.set_label(T("field 거리 (mm), 빨강 = 음수", "field distance (mm; red < 0)"), fontsize=7.5)
    # GT vs field along the target->neighbour line at target-centre height
    zi = int(np.argmin(np.abs(zs)))
    Dall = D.min(1).reshape(U.shape); Dnt = np.where((cls == 0)[None], np.inf, D).min(1).reshape(U.shape)
    ax = axs[3]
    ax.plot(us * 1000, Dall[zi] * 1000, color=FS.INK_2, lw=1.2, ls="--", label=T("GT (전부)", "GT all"))
    ax.plot(us * 1000, fm[zi] * 1000, color=FS.CATEGORICAL[0], lw=1.4, label="main")
    if ff is not None:
        ax.plot(us * 1000, Dnt[zi] * 1000, color=FS.INK, lw=1.0, ls=":", label=T("GT (target 제외)", "GT w/o target"))
        ax.plot(us * 1000, ff[zi] * 1000, color=FS.CATEGORICAL[1], lw=1.4, label="target-free")
    ax.axhline(0, color=FS.GRID_INK, lw=0.6); ax.set_ylim(-40, 80)
    ax.set_xlabel(T("target→이웃 방향 (mm), target 중심 높이", "along target→neighbour at centre height (mm)"), fontsize=8)
    ax.set_ylabel(T("거리 (mm)", "distance (mm)"), fontsize=8); ax.legend(fontsize=7, frameon=False)
    ax.set_title(T("중심 높이 선 위 field vs GT", "field vs GT on the centre line"), fontsize=9, loc="left")
    line = {"u_mm": (us * 1000).round(1).tolist(), "gt_all_mm": (Dall[zi] * 1000).round(2).tolist(), "main_mm": (fm[zi] * 1000).round(2).tolist()}
    if ff is not None:
        line.update(gt_free_mm=(Dnt[zi] * 1000).round(2).tolist(), free_mm=(ff[zi] * 1000).round(2).tolist())
    data["centre_line"] = line
    fig.suptitle(T(f"IFG-AUDIT-LTE 실제 장면 단면 — ep{ep} 요청 {k} ({sl['state']})", f"IFG-AUDIT-LTE section — ep{ep} request {k} ({sl['state']})"), fontsize=10.5, x=0.01, ha="left")
    save(fig, a.fig / "ifg-audit-lte-section.png", data | {"script": SCRIPT})


def band_table(err, gt, mask):
    b = band_of(gt)
    return {BANDS[i][2]: stats(err[mask & (b == i)]) for i in range(len(BANDS))}


def summarise_points(X):
    err_m = X["d_main"] - X["gt_main"]; err_f = X["d_free"] - X["gt_free"]
    s = X["set"]; out = {}
    layer_defs = {
        "main@sphere_unauth": (err_m, X["gt_main"], s == 0, X["vis_main"]),
        "free@sphere_auth": (err_f, X["gt_free"], s == 1, X["vis_free"]),
        "main@probes": (err_m, X["gt_main"], s >= 2, X["vis_main"]),
        "free@probes": (err_f, X["gt_free"], (s >= 2) & np.isfinite(X["d_free"]), X["vis_free"]),
        "main@sphere_auth(reference)": (err_m, X["gt_main"], s == 1, X["vis_main"]),
    }
    for name, (e, g, msk, vis) in layer_defs.items():
        near = msk & (g < 0.10) & (g >= 0)
        out[name] = {"by_band": band_table(e, g, msk), "all": stats(e[msk]), "gt_0_10cm": stats(e[near]),
                     "gt_0_10cm_visible": stats(e[near & vis]), "gt_0_10cm_not_visible": stats(e[near & ~vis]),
                     "gt_0_10cm_by_phase": {p: stats(e[near & (X["phase"] == i)]) for i, p in enumerate(PHASES)},
                     "gt_0_10cm_by_scene": {"clear": stats(e[near & (X["scene"] == 0)]), "neighbour": stats(e[near & (X["scene"] == 1)])},
                     "gt_0_10cm_by_tier": {}}
        from AG3S.ag3s.fields.curobo_field import ANSWER_TIERS
        tier = X["t_free"] if name.startswith("free") else X["t_main"]
        for ti, tn in enumerate(ANSWER_TIERS):
            st = stats(e[near & (tier == ti)])
            if st["n"]:
                out[name]["gt_0_10cm_by_tier"][tn] = st
        cl = X["cls_free"] if name.startswith("free") else X["cls_main"]
        out[name]["gt_0_10cm_by_nearest_class"] = {c: stats(e[near & (cl == ci)]) for ci, c in enumerate(CLASSES) if (near & (cl == ci)).any()}
    return out


def fig_bands(a, summ):
    layers = [("main@sphere_unauth", T("main · 비권한 구", "main · unauth spheres"), FS.CATEGORICAL[0]),
              ("free@sphere_auth", T("target-free · 권한 구", "target-free · auth spheres"), FS.CATEGORICAL[1]),
              ("main@probes", T("main · 껍질 probe", "main · shell probes"), FS.CATEGORICAL[2]),
              ("free@probes", T("target-free · 껍질 probe", "target-free · shell probes"), FS.CATEGORICAL[6])]
    bl = [b[2] for b in BANDS]
    fig, axs = plt.subplots(2, 1, figsize=(12, 8.2), sharex=True); FS.style_axes(fig, axs)
    w = 0.19
    for li, (key, lab, col) in enumerate(layers):
        x = np.arange(len(bl)) + (li - 1.5) * w
        for bi, b in enumerate(bl):
            st = summ[key]["by_band"][b]
            if st["n"] < 20:
                continue
            axs[0].plot([x[bi]] * 2, [st["p5_mm"], st["p95_mm"]], color=col, lw=1.2)
            axs[0].plot(x[bi], st["median_mm"], "o", color=col, ms=5)
            axs[0].text(x[bi], st["p95_mm"] + 1.5, f"n={st['n']}", fontsize=5.5, ha="center", color=FS.INK_2, rotation=90, va="bottom")
            for j, (fk, hatch) in enumerate((("frac_over_5mm", None), ("frac_over_10mm", "///"), ("frac_under_10mm", "..."))):
                pass
        axs[0].plot([], [], "o-", color=col, label=lab)
        big = lambda b_: summ[key]["by_band"][b_]["n"] >= 20  # noqa: E731
        o5 = [summ[key]["by_band"][b].get("frac_over_5mm", np.nan) if big(b) else np.nan for b in bl]
        o10 = [summ[key]["by_band"][b].get("frac_over_10mm", np.nan) if big(b) else np.nan for b in bl]
        u10 = [summ[key]["by_band"][b].get("frac_under_10mm", np.nan) if big(b) else np.nan for b in bl]
        axs[1].bar(x, np.array(o5) * 100, w * 0.9, color=col, alpha=0.45)
        axs[1].bar(x, np.array(o10) * 100, w * 0.9, color=col)
        axs[1].bar(x, -np.array(u10) * 100, w * 0.9, color=col, alpha=0.25, hatch="///", edgecolor=col)
    axs[0].axhline(0, color=FS.INK_2, lw=0.7)
    for yv in (5, 10, -10):
        axs[0].axhline(yv, color=FS.GRID_INK, lw=0.6, ls="--")
    axs[0].set_ylim(-60, 80)
    axs[0].set_ylabel(T("field − GT (mm): 중앙값 · p5–p95", "field − GT (mm): median, p5–p95"), fontsize=8.5)
    axs[0].legend(fontsize=7.5, frameon=False, ncol=4, loc="upper left")
    axs[0].set_title(T("IFG-AUDIT-LTE: GT 거리 구간별 field 오차 (양수 = 실제보다 멀다고 봄 = 위험)",
                       "IFG-AUDIT-LTE: field error by GT distance band (positive = thinks farther = dangerous)"), fontsize=10, loc="left")
    axs[1].axhline(0, color=FS.INK_2, lw=0.7)
    axs[1].set_ylabel(T("% 점: 위 = 과대 >5mm(옅음)·>10mm(진함), 아래 = 과소 >10mm", "% points: up over>5 (light), >10 (dark); down under>10"), fontsize=8)
    axs[1].set_xticks(np.arange(len(bl))); axs[1].set_xticklabels([T("내부 (<0)", "inside (<0)")] + bl[1:])
    axs[1].text(0.99, 0.02, T("n < 20 인 구간은 그리지 않음", "bins with n < 20 not drawn"), transform=axs[1].transAxes, ha="right", fontsize=7, color=FS.INK_2)
    axs[1].set_xlabel(T("GT 거리 구간 (그 계층의 GT: main = 전부, target-free = target 제외)", "GT distance band"), fontsize=8.5)
    fig.tight_layout()
    save(fig, a.fig / "ifg-audit-lte-error-bands.png", {"layers": {k: summ[k]["by_band"] for k, _, _ in layers}, "definitions": "error = field - GT; see verify.json",
                                                         "script": SCRIPT})


def fig_strata(a, summ):
    layers = [("main@sphere_unauth", T("main·비권한 구", "main·unauth"), FS.CATEGORICAL[0]), ("free@sphere_auth", T("free·권한 구", "free·auth"), FS.CATEGORICAL[1]),
              ("main@probes", T("main·probe", "main·probes"), FS.CATEGORICAL[2]), ("free@probes", T("free·probe", "free·probes"), FS.CATEGORICAL[6])]
    groups = [("phase", list(PHASES), lambda k: summ[k]["gt_0_10cm_by_phase"]),
              ("scene", ["clear", "neighbour"], lambda k: summ[k]["gt_0_10cm_by_scene"]),
              ("visibility", ["visible", "not_visible"], lambda k: {"visible": summ[k]["gt_0_10cm_visible"], "not_visible": summ[k]["gt_0_10cm_not_visible"]})]
    fig, axs = plt.subplots(2, 3, figsize=(15, 7.4), gridspec_kw={"width_ratios": [5, 2, 2]}); FS.style_axes(fig, axs)
    data = {}
    w = 0.2
    for gi, (gname, cats, getter) in enumerate(groups):
        data[gname] = {}
        for li, (key, lab, col) in enumerate(layers):
            d = getter(key); data[gname][key] = d
            x = np.arange(len(cats)) + (li - 1.5) * w
            med = [d[c].get("median_mm", np.nan) for c in cats]; p5 = [d[c].get("p5_mm", np.nan) for c in cats]; p95 = [d[c].get("p95_mm", np.nan) for c in cats]
            for xi, a5, a95, mdn in zip(x, p5, p95, med):
                axs[0, gi].plot([xi, xi], [a5, a95], color=col, lw=1.1); axs[0, gi].plot(xi, mdn, "o", color=col, ms=4)
            axs[1, gi].bar(x, [100 * d[c].get("frac_over_5mm", np.nan) for c in cats], w * 0.9, color=col, label=lab)
        for r in (0, 1):
            axs[r, gi].set_xticks(np.arange(len(cats))); axs[r, gi].set_xticklabels(cats, fontsize=7.5)
        axs[0, gi].axhline(0, color=FS.INK_2, lw=0.6); axs[0, gi].set_ylim(-40, 40)
    axs[0, 0].set_ylabel(T("field − GT (mm), GT 0–10 cm", "field − GT (mm), GT 0–10 cm"), fontsize=8.5)
    axs[1, 0].set_ylabel(T("% 과대 >5 mm (위험)", "% over-estimate > 5 mm"), fontsize=8.5)
    axs[1, 0].legend(fontsize=7, frameon=False)
    axs[0, 0].set_title(T("latch 단계별", "by latch phase"), fontsize=9, loc="left"); axs[0, 1].set_title(T("장면 유형", "scene type"), fontsize=9, loc="left")
    axs[0, 2].set_title(T("최근접 GT 표면이 현재 카메라에 보이는가", "nearest GT surface visible now"), fontsize=9, loc="left")
    fig.suptitle(T("IFG-AUDIT-LTE: GT 0–10 cm 점의 오차 — 단계 · 장면 · 가시성", "IFG-AUDIT-LTE: error for GT 0–10 cm by phase, scene, visibility"), fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, a.fig / "ifg-audit-lte-error-strata.png", {"groups": data, "script": SCRIPT})


def fig_exemption(a, reqs):
    ph = ("latched", "closing")
    sel = {p: [r for r in reqs if r["phase"] == p and "exemption" in r and r["exemption"]["n_removed"] > 0] for p in ph}
    fig, axs = plt.subplots(2, 3, figsize=(15, 8)); FS.style_axes(fig, axs)
    data = {}
    rng = np.random.default_rng(0)

    def strip(ax, vals_by_phase, ylabel, title, key):
        data[key] = {}
        for i, p in enumerate(ph):
            v = np.array([x for x in vals_by_phase[p] if x is not None], float)
            data[key][p] = {"n": int(len(v)), "median": None if not len(v) else round(float(np.median(v)), 4),
                            "p10": None if not len(v) else round(float(np.percentile(v, 10)), 4), "p90": None if not len(v) else round(float(np.percentile(v, 90)), 4),
                            "min": None if not len(v) else round(float(v.min()), 4), "max": None if not len(v) else round(float(v.max()), 4)}
            st_mask = np.array([is_stale(r) for r, x in zip(sel[p], vals_by_phase[p]) if x is not None], bool)
            data[key][p]["n_stale"] = int(st_mask.sum())
            vn = v[~st_mask] if len(v) else v
            data[key][p]["median_nonstale"] = None if not len(vn) else round(float(np.median(vn)), 4)
            if len(v):
                jit = i + rng.uniform(-0.18, 0.18, len(v))
                ax.scatter(jit[~st_mask], v[~st_mask], s=7, color=FS.CATEGORICAL[i], alpha=0.6)
                ax.scatter(jit[st_mask], v[st_mask], s=12, marker="x", color=FS.INK_2, lw=0.7,
                           label=T("stale (target 이동 >30mm)", "stale (target moved >30 mm)") if i == 0 and key == "precision" else None)
                ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=FS.INK, lw=1.6)
                ax.text(i + 0.3, np.median(v), f"{np.median(v):.3g}", fontsize=7, va="center")
        ax.set_xticks([0, 1]); ax.set_xticklabels([T("latched (접근)", "latched (approach)"), "closing"], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8); ax.set_title(title, fontsize=9, loc="left")

    strip(axs[0, 0], {p: [r["exemption"]["removed"].get("frac_target_le5mm") for r in sel[p]] for p in ph},
          T("정밀도 (제거 voxel 중 GT target ≤5 mm)", "precision"), T("제거 정밀도 — 요청별", "removal precision per request"), "precision")
    axs[0, 0].legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(0.25, 0.3))
    strip(axs[0, 1], {p: [r.get("recall") for r in sel[p]] for p in ph},
          T("재현율 (main target 표면 voxel 중 free 에서 사라진 비율)", "recall"), T("제거 재현율 — 요청별", "removal recall per request"), "recall")
    strip(axs[0, 2], {p: [r.get("free_residual_target_voxels") for r in sel[p]] for p in ph},
          T("free 계층에 남은 target voxel 수 (≤ voxel/2)", "residual target voxels in free layer"), T("남은 target voxel", "residual target voxels"), "residual_voxels")
    strip(axs[1, 0], {p: [r["target_surface"].get("free_lt5") for r in sel[p]] for p in ph},
          T("GT target 표면점 중 free < 5 mm 개수 (/48, 다른 물체 >15mm)", "target surface points with free < 5 mm"), T("free 계층의 target 잔존 (GT 표면점)", "residual at GT target surface"), "residual_surface_points")
    strip(axs[1, 1], {p: [None if r["neighbour_surface"]["n_visible_main_lt5"] == 0 else r["neighbour_surface"]["n_lost"] / r["neighbour_surface"]["n_visible_main_lt5"] for r in sel[p]] for p in ph},
          T("이웃 과일 보이는 표면점 중 free 에서 사라진 비율", "fraction of visible neighbour surface lost"), T("이웃 과일 오제거", "neighbour wrongly removed"), "neighbour_lost_frac")
    # composition
    ax = axs[1, 2]
    labs = ["target<=5mm", "target5-10mm", "neighbour", "fruit_other", "table", "crate", "obstacle", "other", "no_surface<=10mm"]
    comp = {}
    for i, p in enumerate(ph):
        tot = {l_: sum(r["exemption"]["removed"]["labels"].get(l_, 0) for r in sel[p]) for l_ in labs}
        n = sum(tot.values()); comp[p] = {"n_voxels": n, **{l_: round(v / n, 4) if n else None for l_, v in tot.items()}}
        bottom = 0
        for j, l_ in enumerate(labs):
            v = tot[l_] / n * 100 if n else 0
            ax.bar(i, v, 0.55, bottom=bottom, color=(FS.CATEGORICAL + [FS.INK_2])[j], label=l_ if i == 0 else None); bottom += v
    data["removed_composition"] = comp
    ax.set_xticks([0, 1]); ax.set_xticklabels(["latched", "closing"]); ax.set_ylabel(T("제거 voxel 구성 (%)", "removed voxel composition (%)"), fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right"); ax.set_title(T("제거된 voxel 의 GT 정체 (합계)", "GT identity of removed voxels"), fontsize=9, loc="left")
    fig.suptitle(T("IFG-AUDIT-LTE: target 제외 (removal) 정밀도 · 재현율 — latched vs closing", "IFG-AUDIT-LTE: target exemption precision / recall"), fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, a.fig / "ifg-audit-lte-exemption.png", data | {"n_requests": {p: len(sel[p]) for p in ph}, "script": SCRIPT})
    return data


def fig_geometry(a, reqs):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6)); FS.style_axes(fig, axs)
    data = {}
    # extents
    rs = [r for r in reqs if r.get("manip") and r["manip"].get("extents_mm") and r["phase"] in ("latched", "closing")]
    if rs:
        E = np.array([r["manip"]["extents_mm"] for r in rs]); G = np.array([r["gt_target"]["extents_mm"] for r in rs])
        for i in range(3):
            axs[0].scatter(G[:, i], E[:, i], s=8, color=FS.CATEGORICAL[i], label=T(f"주축 {i + 1} (작은→큰)", f"axis {i + 1}"))
        lo_, hi_ = 30, 110
        axs[0].plot([lo_, hi_], [lo_, hi_], color=FS.INK_2, lw=0.7, ls="--")
        d = E - G
        data["extents"] = {"n_requests": len(rs), "err_mm_median_by_axis": np.median(d, 0).round(2).tolist(), "err_mm_p5_by_axis": np.percentile(d, 5, 0).round(2).tolist(),
                           "err_mm_p95_by_axis": np.percentile(d, 95, 0).round(2).tolist(), "gt_mm_median": np.median(G, 0).round(2).tolist(),
                           "est_mm_median": np.median(E, 0).round(2).tolist()}
        rad = np.array([r["manip"]["radius_mm"] for r in rs if r["manip"].get("radius_mm") is not None])
        gr = np.array([r["gt_target"]["radial_mean_mm"] for r in rs]); gmx = np.array([r["gt_target"]["radial_max_mm"] for r in rs])
        data["radius"] = {"manipulated_radius_mm_median": round(float(np.median(rad)), 2), "gt_radial_mean_mm_median": round(float(np.median(gr)), 2),
                          "gt_radial_max_mm_median": round(float(np.median(gmx)), 2)}
        bl = [r["ball"] for r in rs if r.get("ball") and r["ball"].get("fit_radius_mm") is not None]
        if bl:
            data["ball"] = {"n": len(bl), "fit_radius_mm_median": round(float(np.median([b["fit_radius_mm"] for b in bl])), 2),
                            "ball_radius_mm_median": round(float(np.median([b["radius_mm"] for b in bl])), 2),
                            "fit_centre_err_mm_median": round(float(np.median([b["fit_centre_err_mm"] for b in bl])), 2),
                            "fit_centre_err_mm_p95": round(float(np.percentile([b["fit_centre_err_mm"] for b in bl], 95)), 2)}
        ce = [r["manip"]["centroid_err_mm"] for r in rs if r["manip"].get("centroid_err_mm") is not None]
        cn_ = [r["manip"]["centroid_err_mm"] for r in rs if r["manip"].get("centroid_err_mm") is not None and not is_stale(r)]
        fn_ = [r["ball"]["fit_centre_err_mm"] for r in rs if r.get("ball") and r["ball"].get("fit_centre_err_mm") is not None and not is_stale(r)]
        data["manipulated_centroid_err_mm"] = {"median": round(float(np.median(ce)), 2), "p95": round(float(np.percentile(ce, 95)), 2),
                                               "nonstale_median": round(float(np.median(cn_)), 2), "nonstale_p95": round(float(np.percentile(cn_, 95)), 2)}
        data["ball_fit_centre_err_mm_nonstale"] = {"n": len(fn_), "median": round(float(np.median(fn_)), 2), "p95": round(float(np.percentile(fn_, 95)), 2)}
    axs[0].set_xlabel(T("GT 주축 길이 (mm, principal_extents)", "GT principal extent (mm)"), fontsize=8)
    axs[0].set_ylabel(T("AG3S extents_mm (mm)", "AG3S extents_mm"), fontsize=8); axs[0].legend(fontsize=7, frameon=False)
    axs[0].set_title(T("target 크기: 추정 vs GT (latched·closing 요청)", "target size: estimate vs GT"), fontsize=9, loc="left")
    # held centre error
    hs_all = [r for r in reqs if r.get("held") and r["held"].get("body_centre_err_mm") is not None]
    hs = [r for r in hs_all if r["in_hand"]]
    lag = [r for r in hs_all if not r["in_hand"]]
    data["held_latch_without_contact"] = {"n_held_requests": len([r for r in reqs if r["phase"] == "held"]), "n_not_in_hand": len(lag),
                                          "episodes": sorted({r["ep"] for r in lag}),
                                          "body_centre_err_mm_median_not_in_hand": round(float(np.median([r["held"]["body_centre_err_mm"] for r in lag])), 2) if lag else None,
                                          "apple_hand_gap_mm_median_not_in_hand": round(float(np.median([r["apple_hand_gap_mm"] for r in lag])), 2) if lag else None}
    if hs:
        eps = sorted({r["ep"] for r in hs})
        for i, e in enumerate(eps):
            rr = [r for r in hs if r["ep"] == e]
            axs[1].plot(range(len(rr)), [r["held"]["body_centre_err_mm"] for r in rr], "-", lw=0.8, color=FS.CATEGORICAL[i % 8], alpha=0.8)
        v = np.array([r["held"]["body_centre_err_mm"] for r in hs])
        cov = np.array([r["held"].get("coverage_frac", np.nan) for r in hs]); mo = np.array([r["held"].get("max_outside_mm", np.nan) for r in hs])
        data["held"] = {"n_requests": len(hs), "n_episodes": len(eps), "body_centre_err_mm": {"median": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
                        "max": round(float(v.max()), 2)}, "coverage_frac_median": round(float(np.nanmedian(cov)), 4),
                        "coverage_frac_p5": round(float(np.nanpercentile(cov, 5)), 4), "max_outside_mm_median": round(float(np.nanmedian(mo)), 2),
                        "max_outside_mm_p95": round(float(np.nanpercentile(mo, 95)), 2),
                        "body_radius_mm_median": round(float(np.median([r["held"]["body_radius_mm"] for r in hs])), 2)}
    axs[1].set_xlabel(T("held 요청 순번 (episode 별 선)", "held request index (one line per episode)"), fontsize=8)
    axs[1].set_ylabel(T("붙인 body 구 중심 − GT 사과 중심 (mm)", "attached body centre − GT centre (mm)"), fontsize=8)
    axs[1].set_title(T(f"쥔 사과 모델의 위치 오차 (held · 손가락 접촉 중, 접촉 없는 held {len(lag)}건 제외)", "held object centre error (in contact)"), fontsize=9, loc="left")
    # held voxels remaining
    hv = [r for r in reqs if r["phase"] == "held" and r["in_hand"]]
    if hv:
        f5 = np.array([r["target_surface"]["main_lt5"] for r in hv]); fine = np.array([r.get("held_fine_target_voxels") or 0 for r in hv])
        coarse = np.array([r.get("coarse_neg_near_target", 0) for r in hv])
        axs[2].hist([fine, coarse * 1.0], bins=20, color=[FS.CATEGORICAL[0], FS.CATEGORICAL[1]],
                    label=[T("fine main voxel ≤2.5mm (target)", "fine main voxels (target)"), T("coarse voxel <0 (target 10mm 내)", "coarse voxels <0 near target")])
        data["held_voxels"] = {"n_requests": len(hv), "fine_target_voxels": {"median": float(np.median(fine)), "max": int(fine.max()), "frac_requests_gt0": round(float(np.mean(fine > 0)), 4)},
                               "coarse_neg_near_target": {"median": float(np.median(coarse)), "max": int(coarse.max()), "frac_requests_gt0": round(float(np.mean(coarse > 0)), 4)},
                               "gt_surface_points_main_lt5": {"median": float(np.median(f5)), "max": int(f5.max()), "frac_requests_gt0": round(float(np.mean(f5 > 0)), 4)}}
        axs[2].legend(fontsize=7, frameon=False)
    axs[2].set_xlabel(T("요청당 voxel 수", "voxels per request"), fontsize=8); axs[2].set_ylabel(T("held 요청 수", "held requests"), fontsize=8)
    axs[2].set_title(T("held 중 field 에 남은 쥔 사과 voxel", "held apple voxels left in the field"), fontsize=9, loc="left")
    fig.suptitle(T("IFG-AUDIT-LTE: 3D 기하 사용 — target 크기, 쥔 물체 위치", "IFG-AUDIT-LTE: 3D geometry use"), fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, a.fig / "ifg-audit-lte-geometry.png", data | {"script": SCRIPT})
    return data


# --------------------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--fig", type=pathlib.Path, required=True)
    ap.add_argument("--every", type=int, default=3, help="keep every N-th 'placed' request; searching/latched/closing/held are all kept")
    ap.add_argument("--kit", default="IFG-APPLE-SPLIT"); ap.add_argument("--arm", default="SPL")
    ap.add_argument("--max-eps", type=int, default=0); ap.add_argument("--cache", type=pathlib.Path, default=None)
    ap.add_argument("--scene-ep", type=int, default=119); ap.add_argument("--verify", type=pathlib.Path, default=None)
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field
    K = a.run / "verify" / a.kit
    xml = a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
    scene = Scene(xml)
    dirs = fib_dirs(N_DIR)
    jfs = sorted(glob.glob(str(K / "replay" / f"runs_{a.arm}_{a.arm}_ep*.json")))
    jfs = [f for f in jfs if not f.endswith("npz")]
    if a.max_eps:
        jfs = jfs[:a.max_eps]
    robots = {}
    t0 = time.time()
    if a.cache and a.cache.exists():
        cz = np.load(a.cache, allow_pickle=True)
        X = {k: cz[k] for k in cz.files if k not in ("reqs", "meta")}; reqs = json.loads(str(cz["reqs"])); meta = json.loads(str(cz["meta"]))
    else:
        allp, reqs = [], []
        meta = {"episodes": [], "n_selected": 0, "n_total": 0}
        for jf in jfs:
            js = json.load(open(jf)); ep = int(js["episode"]); b = js["server_chunk_alignment"]["block"]
            rd = pathlib.Path(js["run_dir"])
            man = json.load(open(rd / "manifest.json"))["scene"]
            pl = [json.loads(x) for x in open(rd / "frames.jsonl")]
            pl = [r for r in pl if r.get("kind") == "planning"]
            te = time.time(); n_sel = 0
            for k in range(b["block_len"]):
                p = pathlib.Path(b["chunk_dir"]) / f"chunk_{k + b['offset']:05d}.npz"
                z = np.load(p, allow_pickle=True)
                s = json.loads(str(z["summary_json"]))
                st = str((s.get("grasp") or {}).get("state"))
                meta["n_total"] += 1
                if st == "placed" and k % a.every:
                    continue
                sl = None
                pts, req = process(scene, z, s, pl[k], ep, k, js, man, dirs, robots, build_field, want_slice=sl)
                allp.append(pts); reqs.append(req); n_sel += 1
                del z
            meta["episodes"].append({"ep": ep, "side": js["side"], "profile": js["profile"], "target_fruit": man.get("episode_target_fruit"),
                                     "used_arm": man.get("episode_used_arm"), "n_selected": n_sel, "seconds": round(time.time() - te, 1)})
            meta["n_selected"] += n_sel
            print(f"ep{ep}: {n_sel} requests, {time.time() - te:.1f}s (total {time.time() - t0:.0f}s)", flush=True)
        X = {k: np.concatenate([p_[k] for p_ in allp]) for k in allp[0]}
        if a.cache:
            np.savez_compressed(a.cache, **X, reqs=json.dumps(reqs), meta=json.dumps(meta))
    # section figure
    js = json.load(open(next(f for f in jfs if f"_ep{a.scene_ep}_" in f))); b = js["server_chunk_alignment"]["block"]
    rd = pathlib.Path(js["run_dir"]); man = json.load(open(rd / "manifest.json"))["scene"]
    pl = [r for r in (json.loads(x) for x in open(rd / "frames.jsonl")) if r.get("kind") == "planning"]
    ks = None
    for k in range(b["block_len"]):
        z = np.load(pathlib.Path(b["chunk_dir"]) / f"chunk_{k + b['offset']:05d}.npz", allow_pickle=True)
        if str((json.loads(str(z["summary_json"])).get("grasp") or {}).get("state")) == "closing":
            ks = k; break
    z = np.load(pathlib.Path(b["chunk_dir"]) / f"chunk_{ks + b['offset']:05d}.npz", allow_pickle=True)
    sl = {}
    process(scene, z, json.loads(str(z["summary_json"])), pl[ks], a.scene_ep, ks, js, man, dirs, robots, build_field, want_slice=sl)
    fig_section(a, scene, sl, a.scene_ep, ks, pl[ks])
    del z, sl
    summ = summarise_points(X)
    fig_bands(a, summ); fig_strata(a, summ)
    ex = fig_exemption(a, reqs); geo = fig_geometry(a, reqs)
    # exemption tables by phase
    exn = {}
    for p in ("latched", "closing"):
        rr = [r for r in reqs if r["phase"] == p and "exemption" in r]
        rr1 = [r for r in rr if r["exemption"]["n_removed"] > 0]
        rem_vals = lambda key: [r["exemption"].get(key) for r in rr1 if r["exemption"].get(key) is not None]  # noqa: E731
        exn[p] = {"n_requests_with_trace": len(rr), "n_requests_removed_gt0": len(rr1),
                  "n_removed_median": float(np.median([r["exemption"]["n_removed"] for r in rr1])) if rr1 else None,
                  "removed_free_value_le_half_voxel_median": float(np.median(rem_vals("removed_free_value_le_half_voxel"))) if rem_vals("removed_free_value_le_half_voxel") else None,
                  "removed_free_value_negative_median": float(np.median(rem_vals("removed_free_value_negative"))) if rem_vals("removed_free_value_negative") else None,
                  "main_target_surface_voxels_median": float(np.median([r.get("main_target_surface_voxels", 0) for r in rr1])) if rr1 else None,
                  "free_residual_target_voxels": stats_counts([r.get("free_residual_target_voxels") for r in rr1]),
                  "free_residual_target_voxels_negative": stats_counts([r.get("free_residual_target_voxels_negative") for r in rr1]),
                  "target_surface_free_lt5": stats_counts([r["target_surface"].get("free_lt5") for r in rr1]),
                  "target_surface_free_lt5_visible": stats_counts([r["target_surface"].get("free_lt5_visible") for r in rr1]),
                  "target_surface_free_lt5_occluded": stats_counts([r["target_surface"].get("free_lt5_occluded") for r in rr1]),
                  "target_surface_n_far": stats_counts([r["target_surface"].get("n_far_from_other") for r in rr1]),
                  "target_surface_n_far_visible": stats_counts([r["target_surface"].get("n_far_visible") for r in rr1]),
                  "requests_with_any_free_lt5": sum(1 for r in rr1 if (r["target_surface"].get("free_lt5") or 0) > 0),
                  "neighbour_lost_total": int(sum(r["neighbour_surface"]["n_lost"] for r in rr1)),
                  "neighbour_visible_main_lt5_total": int(sum(r["neighbour_surface"]["n_visible_main_lt5"] for r in rr1)),
                  "requests_with_neighbour_lost": sum(1 for r in rr1 if r["neighbour_surface"]["n_lost"] > 0),
                  "kept_nontarget_frac_target": frac_pool(rr1, "nontarget_kept"), "ambiguous_frac_target": frac_pool(rr1, "ambiguous"),
                  "n_stale": sum(1 for r in rr1 if is_stale(r)), "stale_episodes": sorted({r["ep"] for r in rr1 if is_stale(r)}),
                  "nonstale": {"n": sum(1 for r in rr1 if not is_stale(r)),
                               "precision": stats_counts([r["exemption"]["removed"].get("frac_target_le5mm") for r in rr1 if not is_stale(r)]),
                               "recall": stats_counts([r.get("recall") for r in rr1 if not is_stale(r)]),
                               "residual_target_voxels": stats_counts([r.get("free_residual_target_voxels") for r in rr1 if not is_stale(r)]),
                               "residual_target_voxels_negative": stats_counts([r.get("free_residual_target_voxels_negative") for r in rr1 if not is_stale(r)]),
                               "removed_free_value_negative": stats_counts([r["exemption"].get("removed_free_value_negative") for r in rr1 if not is_stale(r)]),
                               "target_surface_free_lt5": stats_counts([r["target_surface"].get("free_lt5") for r in rr1 if not is_stale(r)]),
                               "composition": composition([r for r in rr1 if not is_stale(r)])},
                  "by_scene": {st_: {"n": len([r for r in rr1 if r["scene_type"] == st_]),
                                     "precision_median": med([r["exemption"]["removed"].get("frac_target_le5mm") for r in rr1 if r["scene_type"] == st_]),
                                     "recall_median": med([r.get("recall") for r in rr1 if r["scene_type"] == st_])} for st_ in ("clear", "neighbour")}}
    kst = datetime.timezone(datetime.timedelta(hours=9))
    verify = {
        "step": "IFG-AUDIT-LTE",
        "written": datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST"),
        "data": {"run": str(a.run), "kit": a.kit, "arm": a.arm, "n_episodes": len(meta["episodes"]), "n_requests_total": meta["n_total"],
                 "n_requests_measured": meta["n_selected"], "sampling": f"all searching/latched/closing/held requests + every {a.every}th placed request (k % {a.every} == 0)",
                 "n_points": int(len(X["set"])), "n_points_by_set": {SETS[i]: int((X["set"] == i).sum()) for i in range(4)}, "episodes": meta["episodes"],
                 "scene_types": {"clear": len({r["ep"] for r in reqs if r["scene_type"] == "clear"}), "neighbour_any_request": len({r["ep"] for r in reqs if r["scene_type"] == "neighbour"})},
                 "requests_by_phase": {p: sum(1 for r in reqs if r["phase"] == p) for p in PHASES}},
        "definitions": __doc__,
        "scripts": [SCRIPT],
        "figures": [str(a.fig / n) for n in ("ifg-audit-lte-section.png", "ifg-audit-lte-error-bands.png", "ifg-audit-lte-error-strata.png",
                                             "ifg-audit-lte-exemption.png", "ifg-audit-lte-geometry.png")],
        "numbers": {"field_vs_gt": summ, "exemption_per_request": ex, "exemption_summary": exn, "geometry": geo},
        "notes": NOTES + [
            "esdf_max_distance (main fine layer) over measured requests: median %.3f m, min %.3f m" % (
                float(np.median([r["esdf_max_distance_m"] for r in reqs if r.get("esdf_max_distance_m") is not None])),
                float(np.min([r["esdf_max_distance_m"] for r in reqs if r.get("esdf_max_distance_m") is not None]))),
            "tsdf_truncation values seen: " + json.dumps(sorted({json.dumps(r.get("tsdf_truncation"), sort_keys=True) for r in reqs})),
            "fine window overflow non-null in %d of %d measured requests" % (sum(1 for r in reqs if r.get("fine_window_overflow")), len(reqs))],
    }
    if a.verify:
        if a.verify.exists():
            raise SystemExit(f"refusing to overwrite {a.verify}")
        a.verify.write_text(json.dumps(verify, indent=1, default=float))
        print("wrote", a.verify)
    print(f"done in {time.time() - t0:.0f}s")


NOTES = [
    "query set (a) = recorded sphere_centres = server robot FK at refined_chunk[0] (commanded first plan step), checked 0.0 mm max on ep119 "
    "requests 10 and 16; the planning-row (measured) pose differs by up to ~40 mm while the arm moves. GT objects are taken at the planning row.",
    "field frame 'base' = MuJoCo world: base body at (-2.6e-5, -7.7e-6, 0) m; object_poses equal MuJoCo body positions; server FK ee_left = MuJoCo "
    "EE_BODY_L to 0.1 mm; finger sphere centres lie 3.9 mm (median) inside the MuJoCo finger geoms (ep119 request 10).",
    "GT probe: 0.1 mm sphere, +0.1 mm added back; apple mesh check: +0.467 m at 0.5 m from the centre (expected 0.467), -22.2 mm at 10 mm from the "
    "centre (expected -22.6); a probe exactly at a mesh centre returns 0 (degenerate) -- deep-inside points are rare in these sets.",
    "GT uses collision geoms; the depth cameras render the visual geoms (apple_vis half sizes 31/35/34 mm vs apple_geom 31/33/33 mm, MuJoCo geom_size), "
    "so up to ~2 mm of the error on fruit surfaces is visual-vs-collision geometry, not field error.",
    "visibility = current frame only (the TSDF also integrates earlier frames), 4:3 image aspect assumed (fovy 90 deg from the model), "
    "ray from 3 cm in front of the lens (the camera housing is hit within 8 mm otherwise).",
    "held phase: GT for the main layer leaves the held target out only when the apple touches the active hand's geoms (gap <= 5 mm, MuJoCo); "
    "'held' requests without contact (latch lag after release) keep the target in GT and are reported separately.",
    "probes may fall inside the table or a neighbour fruit (band <0); 'dangerous' fractions are reported per band so these do not mix with free space.",
]


def is_stale(r):
    b = r.get("ball") or {}
    return b.get("fit_centre_err_mm") is not None and b["fit_centre_err_mm"] > 30.0


def med(v):
    v = [x for x in v if x is not None]
    return None if not v else round(float(np.median(v)), 4)


def stats_counts(v):
    v = np.array([x for x in v if x is not None], float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max()), "frac_gt0": round(float(np.mean(v > 0)), 4)}


def composition(rr):
    labs = ["target<=5mm", "target<=5mm_and_other<=5mm", "target5-10mm", "neighbour", "fruit_other", "table", "crate", "obstacle", "other", "no_surface<=10mm"]
    tot = {l_: sum(r["exemption"]["removed"]["labels"].get(l_, 0) for r in rr) for l_ in labs}
    n = sum(v for k_, v in tot.items() if k_ != "target<=5mm_and_other<=5mm")
    return {"n_voxels": n, **{l_: round(v / n, 4) if n else None for l_, v in tot.items()}}


def frac_pool(rr, key):
    n = sum(r["exemption"].get(key, {}).get("n", 0) for r in rr)
    t = sum(r["exemption"].get(key, {}).get("labels", {}).get("target<=5mm", 0) for r in rr)
    return {"n_voxels": int(n), "frac_target_le5mm": round(t / n, 4) if n else None}


if __name__ == "__main__":
    main()
