"""IFG-AUDIT-ATS — how well ATS (Attention-based Target Segmentation) picks and segments the instructed fruit in current serving,
measured against MuJoCo ground truth on the recorded closed-loop runs. CPU only, read only.

  cd /mnt/dev/AG3S_VLA-development; MUJOCO_GL=osmesa PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m \
      AG3S.ag3s.experiments.studies.ifg_audit_ats --run <run> --fig AG3S/ag3s/docs/figures/ifg-audit

Data: $RUN/verify/IFG-APPLE-SPLIT, arm SPL (current serving + target split), 24 apple scenes x 75 requests. Request k of a scene =
planning row k of frames.jsonl (full MuJoCo qpos) = server chunk `chunk_dir/chunk_{k+offset:05d}.npz` (replay json
server_chunk_alignment.block); the chunk's t_step must equal the row's seq (checked, counted).

GT scene: model_transport.xml (run snapshot_split) + one probe sphere (r 0.1 mm, mocap, no contact) added with MjSpec; d.qpos = planning
row qpos, mj_forward. Obstacle (8 scenes, slot crate_post_0) = vertical cylinder from the run's obstacle config
(verify/IFG-APPLE/obst/ep<E>_place.json, centre = position + [0, 0, height/2], radius, half-length height/2 — pick_place_obstacles.slot_layout).
GT object classes (replay_contacts.py classification, collidable geoms): target fruit (manifest episode_target_fruit; apple in all 24),
other fruits banana / orange / pear (each its own class), crate (crate body), obstacle, table (table body), room (floor, office, shelf),
robot (collidable geoms of the subtree of body 'base', both hands included).
Point distance to a class = min over its geoms of the signed distance (negative inside): box / plane / cylinder exactly in closed form,
mesh geoms (fruits, robot) with mujoco.mj_geomDistance(geom, probe) + probe radius (MuJoCo uses the convex hull of a mesh; the fruit
collision meshes are convex pieces, banana = 5 pieces). The camera renders the visual fruit meshes (apple_vis 61.0 x 67.6 x 68.9 mm vs collision
61.0 x 66.6 x 66.6 mm, plus stem and leaf), so apple / orange / pear also include their visual mesh geoms (vis, stem, leaf, calyx);
banana keeps only its 5 convex collision pieces (the convex hull of its curved visual mesh would over-cover). A first pass with
collision geoms only left ~7 % of visible-state target points unlabelled; all of those (168 / 168 checked) were within 10 mm of the
apple visual geoms.
Point label = the class with the smallest distance among those <= 10 mm, else 'none'. For filtered_cloud, robot distance is evaluated
only for points that a fruit / crate / obstacle claimed (the server already removed the robot); target_points get every class.

Per request (server chunk):
  phase           = summary_json.grasp.state (searching / latched / closing / held / placed)
  has target      = summary has_target and target_points non-empty; 'no target' otherwise
  identity        = GT class nearest to target_centroid (signed distance to the class surface, all classes); correct = target fruit.
                    The same for manipulated.centroid (tracked target). Identity switches = changes of that class between
                    consecutive requests with a target in one scene; tracker switches = requests with manipulated.switched_from set.
  precision       = fraction of target_points labelled target fruit; contamination = fractions labelled each other class
  G               = filtered_cloud points labelled target fruit (visible surface); S = filtered_cloud points within 5 mm of a target point
  recall          = |G within 5 mm of a target point| / |G| (only when |G| > 0); IoU = |S n G| / |S u G| (requests with a target)
  centroid error  = |target_centroid - GT centre| and |manipulated.centroid - GT centre|, GT centre = world position of the target
                    fruit's collision geom (first geom of the class) (mesh centre; 0.6 mm from the apple body origin)
  extent          = manipulated.admissibility.extents_mm (sorted) - GT extents of the target collision mesh (sorted, vertices in the geom frame)
  attention share = sum of attention_values over points labelled each GT object (target, banana, orange, pear, crate, obstacle),
                    normalised over those objects; attention_top = the object with the largest share. Table / room / none / robot
                    shares are reported against the total attention mass (context only).
  memory          = among requests with attention_top != target fruit: 'held' = manipulated exists and its centroid's nearest class is the
                    target fruit, 'wrong' = manipulated exists and the nearest class is another class, 'lost' = no manipulated.
Scene type: 'neighbour' = another fruit's body centre within 100 mm of the target body centre at request 0 (planning row object_poses),
else 'clear'.
"""
import argparse, ast, glob, json, pathlib, sys, time
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.spatial import ConvexHull, cKDTree  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))
from AG3S.ag3s.experiments.common import figstyle as FS  # noqa: E402

KO = FS.use_korean()
T = (lambda ko, en: ko if KO else en)
P = (lambda x: ast.literal_eval(x) if isinstance(x, str) else x)   # noqa: E731  (some nested values are Python reprs)
FRUITS = ("apple", "banana", "orange", "pear")
PHASES = ("searching", "latched", "closing", "held", "placed")
PH_KO = {"searching": "탐색", "latched": "접근(latched)", "closing": "닫는 중", "held": "쥠(운반)", "placed": "놓음"}
TOL, NN_TOL, NEIGH = 0.010, 0.005, 0.100
SCRIPT = "AG3S/ag3s/experiments/studies/ifg_audit_ats.py"


# ---------------------------------------------------------------- GT geometry
class Scene:
    def __init__(self, xml: str):
        import mujoco
        self.mj = mujoco
        sp = mujoco.MjSpec.from_file(xml)
        b = sp.worldbody.add_body(name="audit_probe", mocap=True)
        b.add_geom(name="audit_probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[1e-4, 0, 0], contype=0, conaffinity=0)
        m = self.m = sp.compile(); self.d = mujoco.MjData(m)
        self.probe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "audit_probe_geom")
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        coll = np.array([bool(m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
        coll[self.probe] = False

        def subtree(root):
            out = set()
            for b_ in range(m.nbody):
                x = b_
                while x > 0 and x != root:
                    x = int(m.body_parentid[x])
                if x == root:
                    out.add(b_)
            return out
        gb = lambda bodies: [g for g in range(m.ngeom) if coll[g] and int(m.geom_bodyid[g]) in bodies]
        self.cls = {f: gb({B(f)}) for f in FRUITS}
        # the camera renders the visual fruit meshes (apple_vis / stem / leaf ...): add them, except banana's visual meshes, whose convex
        # hulls would fill the curve (banana keeps its 5 convex collision pieces)
        self.vis_added = {}
        for f in FRUITS:
            if f == "banana":
                continue
            vg = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == B(f) and not coll[g] and m.geom_type[g] == 7]
            self.cls[f] += vg; self.vis_added[f] = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in vg]
        self.cls["crate"] = gb({B("crate")}); self.cls["table"] = gb({B("table")})
        robot = gb(subtree(B("base")))
        used = set(sum(self.cls.values(), [])) | set(robot)
        self.cls["room"] = [g for g in range(m.ngeom) if coll[g] and g not in used]
        self.cls["robot"] = robot
        self.body = {f: B(f) for f in (*FRUITS, "crate")}
        self.obstacle = None   # (centre, radius, half-length), vertical cylinder
        self.ft = np.zeros(6)
        for c, gs in self.cls.items():
            bad = [g for g in gs if m.geom_type[g] not in (0, 5, 6, 7)]
            assert not bad, (c, bad)

    def set_qpos(self, q):
        self.d.qpos[:] = q; self.mj.mj_forward(self.m, self.d)

    def _geom_dist(self, g, X, prefilter):
        m, d = self.m, self.d
        c = d.geom_xpos[g]; R = d.geom_xmat[g].reshape(3, 3); t = int(m.geom_type[g]); s = m.geom_size[g]
        out = np.full(len(X), np.inf)
        if t == 0:      # plane: z axis of the geom frame
            return (X - c) @ R[:, 2]
        if t == 6:      # box
            q = np.abs((X - c) @ R) - s[:3]
            return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
        if t == 5:      # cylinder
            q = (X - c) @ R
            a = np.stack([np.hypot(q[:, 0], q[:, 1]) - s[0], np.abs(q[:, 2]) - s[1]], 1)
            return np.linalg.norm(np.maximum(a, 0), axis=1) + np.minimum(a.max(1), 0)
        # mesh: exact convex-hull distance through the probe sphere
        idx = np.flatnonzero(np.linalg.norm(X - c, axis=1) <= m.geom_rbound[g] + prefilter)
        d.geom_xmat[self.probe] = np.eye(3).ravel()
        for i in idx:
            d.geom_xpos[self.probe] = X[i]
            out[i] = self.mj.mj_geomDistance(m, d, int(g), self.probe, prefilter + 0.01, self.ft) + 1e-4
        return out

    def class_dist(self, c, X, prefilter=TOL):
        X = np.asarray(X, float).reshape(-1, 3)
        if c == "obstacle":
            if self.obstacle is None:
                return np.full(len(X), np.inf)
            cen, r, h = self.obstacle
            q = X - cen
            a = np.stack([np.hypot(q[:, 0], q[:, 1]) - r, np.abs(q[:, 2]) - h], 1)
            return np.linalg.norm(np.maximum(a, 0), axis=1) + np.minimum(a.max(1), 0)
        out = np.full(len(X), np.inf)
        for g in self.cls[c]:
            if self.m.geom_type[g] == 7 and len(X) > 1:
                # broad phase for many-geom classes: skip geoms whose bounding sphere is far from every point
                c0 = self.d.geom_xpos[g]
                if np.min(np.linalg.norm(X - c0, axis=1)) > self.m.geom_rbound[g] + prefilter:
                    continue
            out = np.minimum(out, self._geom_dist(g, X, prefilter))
        return out

    def label(self, X, classes, robot_for=None):
        """Per point: (label index into classes + ['none'], distance to that class)."""
        X = np.asarray(X, float).reshape(-1, 3)
        D = np.stack([self.class_dist(c, X) for c in classes], 1)
        lab = np.where(D.min(1) <= TOL, D.argmin(1), len(classes))
        if robot_for is not None:   # robot only for points another listed class claimed
            ci = [classes.index(c) for c in robot_for]
            sel = np.flatnonzero(np.isin(lab, ci))
            if len(sel):
                dr = self.class_dist("robot", X[sel])
                take = (dr <= TOL) & (dr < D[sel, lab[sel]])
                lab = lab.copy(); lab[sel[take]] = len(classes) + 1   # index of 'robot' in the extended name list
        return lab, D

    def nearest_class(self, x, classes):
        x = np.asarray(x, float).reshape(1, 3)
        dd = {c: float(self.class_dist(c, x, prefilter=0.3 if c == "robot" else 2.0)[0]) for c in classes}
        c = min(dd, key=dd.get)
        return c, dd

    def mesh_world_vertices(self, g):
        m, d = self.m, self.d
        mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]

    def mesh_extents_mm(self, g):
        m = self.m; mid = int(m.geom_dataid[g])
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        return sorted(float(x) for x in np.round((v.max(0) - v.min(0)) * 1000, 1))


# ---------------------------------------------------------------- per-request measurement
def scene_runs(K: pathlib.Path):
    for jf in sorted(glob.glob(str(K / "replay" / "runs_SPL_SPL_ep*.json")), key=lambda p: int(p.split("_ep")[1].split("_")[0])):
        j = json.load(open(jf))
        yield j


def obstacle_of(run: pathlib.Path, rd: pathlib.Path):
    st = rd.parent / f"{rd.name}.start"
    cfg = next((ln.split("=", 1)[1] for ln in st.read_text().splitlines() if ln.startswith("obstacle=") and ln != "obstacle=:"), None)
    if not cfg:
        return None, None
    path, prof = cfg.rsplit(":", 1)
    ob = json.load(open(path))["profiles"][prof]["obstacles"]
    assert len(ob) == 1 and ob[0]["slot"] == "crate_post_0", ob
    o = ob[0]; cen = np.array(o["position"], float) + [0, 0, o["height"] / 2]
    return (cen, float(o["radius"]), float(o["height"]) / 2), {"config": path, "profile": prof, **o}


def measure_request(S: Scene, c, tfruit, objs):
    """One server chunk at the GT state already set in S."""
    s = json.loads(str(c["summary_json"]))
    g = P(s.get("grasp")) or {}; mp = P(s.get("manipulated"))
    F = c["filtered_cloud"].astype(float); A = c["attention_values"].astype(float)
    TPs = c["target_points"].astype(float) if "target_points" in c.files else np.zeros((0, 3))
    tc = np.asarray(c["target_centroid"], float) if "target_centroid" in c.files else None
    cloud_cls = list(objs) + ["table", "room"]
    names = cloud_cls + ["none", "robot"]
    lab, _ = S.label(F, cloud_cls, robot_for=list(objs))
    lab_n = np.array(names, dtype=object)[lab]
    ti = names.index(tfruit)
    G = lab == ti
    r = {"phase": g.get("state") or "none", "status": s.get("status"),
         "has_target": bool(s.get("has_target")) and len(TPs) > 0, "n_target_points": int(len(TPs)),
         "n_cloud": int(len(F)), "n_gt_visible": int(G.sum()), "n_clusters": s.get("n_clusters"),
         "mstate": (mp or {}).get("state"), "mid": (mp or {}).get("id"), "switched_from": (mp or {}).get("switched_from"),
         "age_frames": (mp or {}).get("age_frames"), "frame": (mp or {}).get("frame"), "last_seen_frame": (mp or {}).get("last_seen_frame"),
         "noun": (P(s.get("attention_source")) or {}).get("noun")}
    # attention shares over GT objects
    mass = {o: float(A[lab_n == o].sum()) for o in objs}
    tot_obj = sum(mass.values()); tot = float(A.sum())
    r["att_share"] = {o: (mass[o] / tot_obj if tot_obj > 0 else None) for o in objs}
    r["att_top"] = max(mass, key=mass.get) if tot_obj > 0 else None
    r["att_ctx"] = {k: (float(A[lab_n == k].sum()) / tot if tot > 0 else None) for k in ("table", "room", "none", "robot")}
    r["att_ctx"]["objects"] = tot_obj / tot if tot > 0 else None
    gt_c = S.d.geom_xpos[S.cls[tfruit][0]].copy()
    r["gt_centre"] = gt_c.tolist()
    all_cls = list(objs) + ["table", "room", "robot"]
    if r["has_target"]:
        tl, _ = S.label(TPs, all_cls)
        tn = np.array(all_cls + ["none"], dtype=object)[tl]
        r["precision"] = float(np.mean(tn == tfruit))
        r["contam"] = {k: float(np.mean(tn == k)) for k in all_cls + ["none"] if k != tfruit}
        tree = cKDTree(TPs)
        dS, _ = tree.query(F, distance_upper_bound=NN_TOL)
        Sm = np.isfinite(dS)
        r["n_S"] = int(Sm.sum())
        r["recall"] = float((Sm & G).sum() / G.sum()) if G.sum() else None
        u = (Sm | G).sum()
        r["iou"] = float((Sm & G).sum() / u) if u else None
        r["tc_err_mm"] = float(np.linalg.norm(tc - gt_c) * 1000); r["tc"] = tc.tolist()
        r["id_tc"], dd = S.nearest_class(tc, all_cls)
        r["tc_to_target_surface_mm"] = dd[tfruit] * 1000
    if mp:
        mc = np.asarray(mp["centroid"], float)
        r["mc_err_mm"] = float(np.linalg.norm(mc - gt_c) * 1000)
        r["id_mc"], dd = S.nearest_class(mc, all_cls)
        r["mc_to_target_surface_mm"] = dd[tfruit] * 1000
        adm = P(mp.get("admissibility")) or {}
        if adm.get("extents_mm"):
            r["extents_mm"] = sorted(float(x) for x in adm["extents_mm"])
    # QA: how well the cloud sits on the target surface (median |signed distance| of points labelled target)
    if G.sum():
        r["qa_target_absdist_mm"] = float(np.median(np.abs(S.class_dist(tfruit, F[G]))) * 1000)
    return r


def collect(run: pathlib.Path, K: pathlib.Path, S: Scene, step: int, log):
    scenes = []
    for j in scene_runs(K):
        t0 = time.time()
        rd = pathlib.Path(j["run_dir"]); man = json.load(open(rd / "manifest.json"))["scene"]
        tfruit = man["episode_target_fruit"]; al = j["server_chunk_alignment"]["block"]
        assert al["found"], j["episode"]
        obst, ocfg = obstacle_of(run, rd)
        S.obstacle = obst
        objs = list(FRUITS) + ["crate"] + (["obstacle"] if obst is not None else [])
        objs = [tfruit] + [o for o in objs if o != tfruit]
        pl = [json.loads(ln) for ln in open(rd / "frames.jsonl") if ln.strip()]
        pl = [x for x in pl if x["kind"] == "planning"]
        op0 = P(pl[0]["object_poses"])
        dneigh = {f: float(np.linalg.norm(np.subtract(op0[f]["pos"], op0[tfruit]["pos"])) * 1000) for f in FRUITS if f != tfruit}
        S.set_qpos(pl[0]["qpos"])
        gap0 = {f: float(min(S.mj.mj_geomDistance(S.m, S.d, a, b, 1.0, S.ft) for a in S.cls[tfruit] for b in S.cls[f]) * 1000)
                for f in FRUITS if f != tfruit}
        sc = {"episode": int(j["episode"]), "seed": int(j["seed"]), "side": j["side"], "target": tfruit, "obstacle": ocfg,
              "neighbour_centre_dist_mm": dneigh, "neighbour_surface_gap_mm": gap0,
              "type": "neighbour" if min(dneigh.values()) <= NEIGH * 1000 else "clear",
              "chunk_dir": al["chunk_dir"], "offset": al["offset"], "n_requests": len(pl), "seq_mismatch": 0,
              "gt_extents_mm": S.mesh_extents_mm(S.cls[tfruit][0]), "gt_extents_visual_mm": S.mesh_extents_mm(S.cls[tfruit][1]) if len(S.cls[tfruit]) > 1 else None,
              "class_geoms_target": [S.mj.mj_id2name(S.m, S.mj.mjtObj.mjOBJ_GEOM, g) for g in S.cls[tfruit]], "rows": []}
        for k in range(0, len(pl), step):
            S.set_qpos(pl[k]["qpos"])
            with np.load(f"{al['chunk_dir']}/chunk_{k + al['offset']:05d}.npz", allow_pickle=True) as c:
                if int(c["t_step"]) != int(pl[k]["seq"]):
                    sc["seq_mismatch"] += 1
                r = measure_request(S, c, tfruit, objs)
            r["k"] = k; r["t_step"] = int(pl[k]["t_step"])
            sc["rows"].append(r)
        scenes.append(sc)
        log(f"ep{sc['episode']} {sc['type']} rows {len(sc['rows'])} {time.time() - t0:.1f}s")
    return scenes


# ---------------------------------------------------------------- aggregation
def q(v):
    v = [x for x in v if x is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(np.median(a)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "min": float(a.min()), "max": float(a.max())}


def runs_of(seq, val):
    out, n = [], 0
    for x in seq + [object()]:
        if x == val:
            n += 1
        elif n:
            out.append(n); n = 0
    return out


def mem_cat(r, tf):
    """held = tracked centroid nearest the target fruit; stale = same tracker id as the scene's first, centroid nearest another class;
    switched = a different tracker id (or switched_from set); lost = no manipulated."""
    if r["mid"] is None:
        return "lost"
    if r["id_mc"] == tf:
        return "held"
    return "switched" if (r["switched_from"] is not None or r["mid"] != r.get("mid0")) else "stale"


def aggregate(scenes):
    for s in scenes:
        m0 = next((r["mid"] for r in s["rows"] if r["mid"] is not None), None)
        byk = {r["k"]: r for r in s["rows"]}
        for r in s["rows"]:
            r["mid0"] = m0
            # occluded memory vs the GT pose when the tracker last saw the target (pipeline frame index = request index k)
            if r["mstate"] == "occluded" and r.get("tc") is not None and r["last_seen_frame"] in byk:
                g0 = np.asarray(byk[r["last_seen_frame"]]["gt_centre"]); g1 = np.asarray(r["gt_centre"])
                r["tc_err_vs_lastseen_gt_mm"] = float(np.linalg.norm(np.asarray(r["tc"]) - g0) * 1000)
                r["gt_moved_since_lastseen_mm"] = float(np.linalg.norm(g1 - g0) * 1000)
    N = {}
    types = Counter(s["type"] for s in scenes)
    N["scenes"] = {"n": len(scenes), "by_type": dict(types),
                   "neighbour_episodes": [s["episode"] for s in scenes if s["type"] == "neighbour"],
                   "clear_episodes": [s["episode"] for s in scenes if s["type"] == "clear"],
                   "obstacle_episodes": [s["episode"] for s in scenes if s["obstacle"]],
                   "reached_placed_episodes": [s["episode"] for s in scenes if any(r["phase"] == "placed" for r in s["rows"])],
                   "seq_mismatch_total": sum(s["seq_mismatch"] for s in scenes)}
    rows = [(s, r) for s in scenes for r in s["rows"]]
    N["requests"] = {"n": len(rows), "by_phase": {p: sum(r["phase"] == p for _, r in rows) for p in PHASES + ("none",)},
                     "mstate": {str(k): v for k, v in Counter((r["phase"], r["mstate"]) for _, r in rows).items()}}
    groups = {"all": lambda s: True, "clear": lambda s: s["type"] == "clear", "neighbour": lambda s: s["type"] == "neighbour"}
    tf = scenes[0]["target"]
    seg = {}
    for gname, gf in groups.items():
        seg[gname] = {}
        for p in PHASES + ("all",):
            R = [r for s, r in rows if gf(s) and (p == "all" or r["phase"] == p)]
            T_ = [r for r in R if r["has_target"]]
            vis = [r for r in T_ if r["mstate"] == "visible"]
            occ = [r for r in T_ if r["mstate"] == "occluded"]
            e = {"n_requests": len(R), "n_with_target": len(T_), "n_no_target": len(R) - len(T_),
                 "identity_rate": (sum(r["id_tc"] == tf for r in T_) / len(T_)) if T_ else None,
                 "identity_nearest_counts": dict(Counter(r["id_tc"] for r in T_)),
                 "precision": q([r["precision"] for r in T_]), "recall": q([r["recall"] for r in T_]),
                 "n_recall_undefined_target_not_visible": sum(r["recall"] is None for r in T_),
                 "iou": q([r["iou"] for r in T_]),
                 "precision_pooled": (sum(r["precision"] * r["n_target_points"] for r in T_) / sum(r["n_target_points"] for r in T_)) if T_ else None,
                 "contamination_mean": {k: float(np.mean([r["contam"].get(k, 0.0) for r in T_])) for k in sorted({k for r in T_ for k in r["contam"]})},
                 "precision_visible_state": q([r["precision"] for r in vis]), "precision_occluded_state": q([r["precision"] for r in occ]),
                 "iou_visible_state": q([r["iou"] for r in vis]), "iou_occluded_state": q([r["iou"] for r in occ]),
                 "target_centroid_err_mm": q([r["tc_err_mm"] for r in T_]),
                 "manipulated_centroid_err_mm": q([r.get("mc_err_mm") for r in R if r.get("mc_err_mm") is not None]),
                 "target_centroid_to_target_surface_mm": q([r["tc_to_target_surface_mm"] for r in T_]),
                 "n_gt_visible_points": q([r["n_gt_visible"] for r in R]),
                 "n_target_points": q([r["n_target_points"] for r in T_])}
            ex = [r["extents_mm"] for r in R if r.get("extents_mm")]
            e["extents_mm"] = {"n": len(ex), **({f"axis{i}_sorted": q([x[i] for x in ex]) for i in range(3)} if ex else {})}
            # attention vs memory
            AT = [r for r in R if r["att_top"] is not None]
            off = [r for r in AT if r["att_top"] != tf]
            mem = Counter(mem_cat(r, tf) for r in off)
            e["attention"] = {"n_defined": len(AT), "share_on_target": q([r["att_share"][tf] for r in AT]),
                              "share_mean_by_object": {o: float(np.mean([r["att_share"].get(o) or 0 for r in AT])) for o in sorted({o for r in AT for o in r["att_share"]})},
                              "top_counts": dict(Counter(r["att_top"] for r in AT)), "n_top_not_target": len(off),
                              "memory_when_top_not_target": dict(mem),
                              "memory_wrong_nearest": dict(Counter(r["id_mc"] for r in off if mem_cat(r, tf) in ("stale", "switched"))),
                              "memory_by_state": {str(k_): dict(Counter(mem_cat(r, tf) for r in off if r["mstate"] == k_)) for k_ in ("visible", "occluded", None)},
                              "ctx_share_of_total": {k: float(np.mean([r["att_ctx"][k] or 0 for r in AT])) for k in ("objects", "table", "room", "none", "robot")} if AT else {}}
            e["memory_all_requests"] = dict(Counter(mem_cat(r, tf) for r in R))
            e["tracked_identity_rate"] = (sum(r["id_mc"] == tf for r in R if r["mid"] is not None) / max(1, sum(r["mid"] is not None for r in R))) if R else None
            seg[gname][p] = e
    N["by_group_phase"] = seg
    # switches and state durations
    sw = {}
    for s in scenes:
        idseq = [r["id_tc"] for r in s["rows"] if r["has_target"]]
        mids = [r["mid"] for r in s["rows"] if r["mid"] is not None]
        sw[str(s["episode"])] = {"identity_switches": int(sum(a != b for a, b in zip(idseq, idseq[1:]))),
                                 "identity_sequence_classes": dict(Counter(idseq)),
                                 "tracker_switched_from_set": int(sum(r["switched_from"] is not None for r in s["rows"])),
                                 "tracker_id_changes": int(sum(a != b for a, b in zip(mids, mids[1:]))),
                                 "n_no_target": int(sum(not r["has_target"] for r in s["rows"])),
                                 "n_no_target_before_placed": int(sum(not r["has_target"] and r["phase"] != "placed" for r in s["rows"]))}
    N["switches"] = {"per_scene": sw, "identity_switches_total": sum(v["identity_switches"] for v in sw.values()),
                     "scenes_with_identity_switch": [e for e, v in sw.items() if v["identity_switches"]],
                     "tracker_switched_from_total": sum(v["tracker_switched_from_set"] for v in sw.values()),
                     "tracker_id_changes_total": sum(v["tracker_id_changes"] for v in sw.values()),
                     "no_target_total": sum(v["n_no_target"] for v in sw.values()),
                     "no_target_before_placed_total": sum(v["n_no_target_before_placed"] for v in sw.values()),
                     "no_target_by_phase": dict(Counter(r["phase"] for _, r in rows if not r["has_target"]))}
    st = {}
    for state in ("visible", "occluded", "kept", None):
        L = []
        for s in scenes:
            L += runs_of([r["mstate"] for r in s["rows"]], state)
        st[str(state)] = {"n_requests": sum(L), "n_runs": len(L), "run_length": q(L)}
    st["by_phase"] = {p: dict(Counter(str(r["mstate"]) for _, r in rows if r["phase"] == p)) for p in PHASES}
    for p in PHASES:
        O = [r for _, r in rows if r["phase"] == p and r.get("tc_err_vs_lastseen_gt_mm") is not None]
        st.setdefault("occluded_memory_vs_gt", {})[p] = {"n": len(O), "centroid_err_vs_gt_now_mm": q([r["tc_err_mm"] for r in O]),
                                                          "centroid_err_vs_gt_at_last_seen_mm": q([r["tc_err_vs_lastseen_gt_mm"] for r in O]),
                                                          "gt_target_moved_since_last_seen_mm": q([r["gt_moved_since_lastseen_mm"] for r in O]),
                                                          "precision": q([r["precision"] for r in O])}
    st["occluded_staleness_frames"] = q([r["frame"] - r["last_seen_frame"] for _, r in rows
                                         if r["mstate"] == "occluded" and r["frame"] is not None and r["last_seen_frame"] is not None])
    N["manipulated_state"] = st
    N["qa_target_absdist_mm"] = {p: q([r.get("qa_target_absdist_mm") for _, r in rows if r["phase"] == p]) for p in PHASES}
    N["gt_extents_mm_collision"] = scenes[0]["gt_extents_mm"]; N["gt_extents_mm_visual"] = scenes[0].get("gt_extents_visual_mm")
    N["target_class_geoms"] = scenes[0].get("class_geoms_target")
    return N


# ---------------------------------------------------------------- figures
def save(fig, path: pathlib.Path, payload: dict, force: bool):
    for p in (path, path.with_suffix(".json")):
        if p.exists() and not force:
            raise SystemExit(f"refusing to overwrite {p}")
    fig.savefig(path, dpi=175, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps(payload | {"script": SCRIPT}, indent=1, default=float))
    print(path)


def fig_panel(S: Scene, sc, k, path, force):
    import mujoco
    al_dir, off = sc["chunk_dir"], sc["offset"]
    rd = None
    for j in scene_runs(pathlib.Path(al_dir).parents[3]):
        if j["episode"] == sc["episode"]:
            rd = pathlib.Path(j["run_dir"])
    pl = [json.loads(ln) for ln in open(rd / "frames.jsonl") if ln.strip()]
    pl = [x for x in pl if x["kind"] == "planning"]
    S.set_qpos(pl[k]["qpos"])
    with np.load(f"{al_dir}/chunk_{k + off:05d}.npz", allow_pickle=True) as c:
        F = c["filtered_cloud"].astype(float); A = c["attention_values"].astype(float)
        TPs = c["target_points"].astype(float); tc = np.asarray(c["target_centroid"], float)
    r = next(x for x in sc["rows"] if x["k"] == k)
    ctr = np.asarray(r["gt_centre"])
    box = np.array([[ctr[0] - 0.16, ctr[0] + 0.16], [ctr[1] - 0.16, ctr[1] + 0.16], [0.80, ctr[2] + 0.12]])
    w = np.all((F >= box[:, 0]) & (F <= box[:, 1]), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [1.0, 1.0]}); FS.style_axes(fig, axes)
    cmap = FS.sequential_cmap()
    colors = {"apple": FS.CATEGORICAL[7], "banana": FS.CATEGORICAL[3], "orange": FS.CATEGORICAL[1], "pear": FS.CATEGORICAL[2],
              "obstacle": FS.CATEGORICAL[6]}
    outlines = {}
    for ax, (i, j_, nm) in zip(axes, ((0, 1, T("위에서 (x–y)", "top view (x-y)")), (0, 2, T("옆에서 (x–z)", "side view (x-z)")))):
        o = np.argsort(A[w])
        sca = ax.scatter(F[w][o, i], F[w][o, j_], c=A[w][o], s=4, cmap=cmap, vmin=0, vmax=1, lw=0, zorder=1)
        ax.scatter(TPs[:, i], TPs[:, j_], s=16, facecolor="none", edgecolor=FS.INK, lw=0.6, zorder=3,
                   label=T(f"ATS 선택 cluster (target_points, {len(TPs)} 점)", f"selected cluster ({len(TPs)} pts)"))
        for f in FRUITS:
            V = np.concatenate([S.mesh_world_vertices(g) for g in S.cls[f]])
            if not np.any(np.all((V >= box[:, 0] - 0.03) & (V <= box[:, 1] + 0.03), axis=1)):
                continue
            pts = []
            for g in S.cls[f]:   # banana = 5 convex pieces: outline each piece
                v2 = S.mesh_world_vertices(g)[:, [i, j_]]; h = ConvexHull(v2); pts.append(v2[h.vertices])
                ax.fill(*np.vstack([v2[h.vertices], v2[h.vertices][:1]]).T, fill=False, ec=colors[f], lw=1.6, zorder=4)
            cc = np.concatenate(pts).mean(0)
            ax.text(cc[0], np.concatenate(pts)[:, 1].max() + 0.008, T({"apple": "사과 (지시 대상)", "banana": "바나나", "orange": "오렌지", "pear": "배"}[f], f),
                    color=colors[f], fontsize=8.5, ha="center", zorder=5)
            outlines[f] = [p.tolist() for p in pts] if i == 0 and j_ == 1 else outlines.get(f)
        ax.plot(tc[i], tc[j_], "x", color=FS.INK, ms=9, mew=2, zorder=6, label=T("target_centroid", "target_centroid"))
        ax.plot(ctr[i], ctr[j_], "+", color=FS.CATEGORICAL[7], ms=12, mew=2, zorder=6, label=T("GT 사과 중심", "GT apple centre"))
        ax.set_xlim(*box[i]); ax.set_ylim(*box[j_]); ax.set_aspect("equal")
        ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel(("y" if j_ == 1 else "z") + " (m)", fontsize=8)
        ax.set_title(nm, fontsize=9, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    cb = fig.colorbar(sca, ax=axes, fraction=0.02, pad=0.01); cb.set_label(T("attention (L10H7, 명사 'apple')", "attention"), fontsize=8)
    c_ = r["contam"]
    fig.suptitle(T(f"ep{sc['episode']} 요청 {k} ({PH_KO[r['phase']]}, 상태 {r['mstate']}): 이웃 과일 중심 거리 "
                   f"{min(sc['neighbour_centre_dist_mm'].values()):.0f} mm — 선택 cluster precision {r['precision']:.2f}, recall {r['recall']:.2f}, "
                   f"IoU {r['iou']:.2f}, 중심 오차 {r['tc_err_mm']:.1f} mm",
                   f"ep{sc['episode']} request {k}: precision {r['precision']:.2f}, recall {r['recall']:.2f}, IoU {r['iou']:.2f}"),
                 fontsize=10, x=0.01, ha="left")
    save(fig, path, {"episode": sc["episode"], "request": k, "t_step": r["t_step"], "phase": r["phase"], "manipulated_state": r["mstate"],
                     "scene_type": sc["type"], "neighbour_centre_dist_mm": sc["neighbour_centre_dist_mm"],
                     "neighbour_surface_gap_mm": sc["neighbour_surface_gap_mm"],
                     "precision": r["precision"], "recall": r["recall"], "iou": r["iou"], "contamination": c_,
                     "target_centroid": tc.tolist(), "gt_centre": r["gt_centre"], "target_centroid_err_mm": r["tc_err_mm"],
                     "n_target_points": r["n_target_points"], "n_gt_visible": r["n_gt_visible"], "n_cloud_points_in_window": int(w.sum()),
                     "window_m": box.tolist(), "attention_share": r["att_share"], "fruit_outlines_xy_m": outlines}, force)


def fig_phase(N, path, force):
    G = N["by_group_phase"]
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.9)); FS.style_axes(fig, axes)
    grp = (("clear", T("이웃 없음", "clear"), FS.CATEGORICAL[0]), ("neighbour", T("이웃 과일 ≤100 mm", "neighbour"), FS.CATEGORICAL[1]))
    x = np.arange(len(PHASES)); wd = 0.38; shown = {}
    for ai, (key, title) in enumerate((("precision", "precision"), ("recall", T("visible-surface recall", "visible-surface recall")),
                                       ("iou", "IoU"), ("identity_rate", T("identity (nearest GT = 사과)", "identity rate")))):
        ax = axes[ai]; shown[key] = {}
        for gi, (g, lab, col) in enumerate(grp):
            vals, lo, hi, ns = [], [], [], []
            for p in PHASES:
                e = G[g][p]
                if key == "identity_rate":
                    v = e[key]; vals.append(np.nan if v is None else v); lo.append(0); hi.append(0); ns.append(e["n_with_target"])
                else:
                    st = e[key]
                    vals.append(st.get("median", np.nan)); lo.append(st.get("median", 0) - st.get("p25", 0) if st["n"] else 0)
                    hi.append(st.get("p75", 0) - st.get("median", 0) if st["n"] else 0); ns.append(st["n"])
            xx = x + (gi - 0.5) * wd
            ax.bar(xx, np.nan_to_num(vals), wd * 0.92, color=col, label=lab, yerr=None if key == "identity_rate" else [lo, hi],
                   error_kw=dict(ecolor=FS.INK_2, lw=0.8, capsize=2))
            for xi, v, n in zip(xx, vals, ns):
                ax.text(xi, (0 if np.isnan(v) else v) + 0.02, f"{'–' if np.isnan(v) else f'{v:.2f}'}\nn={n}", ha="center", fontsize=6.2, color=FS.INK_2)
            shown[key][g] = {p: {"value": None if np.isnan(v) else float(v), "n": n} for p, v, n in zip(PHASES, vals, ns)}
        ax.set_xticks(x); ax.set_xticklabels([PH_KO[p] if KO else p for p in PHASES], fontsize=7.5, rotation=20)
        ax.set_ylim(0, 1.18); ax.set_title(title + (T("  (중앙값, 막대 = IQR)", "  (median, IQR)") if key != "identity_rate" else ""), fontsize=9, loc="left")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    # contamination composition of target_points by phase (all scenes)
    ax = axes[4]; comp = ["apple", "fruit", "table", "crate", "robot", "room", "none", "obstacle"]
    cc = {"apple": FS.CATEGORICAL[7], "fruit": FS.CATEGORICAL[1], "table": FS.GRID_INK, "crate": FS.CATEGORICAL[3], "robot": FS.CATEGORICAL[6],
          "room": FS.INK_2, "none": "#9a9893", "obstacle": FS.CATEGORICAL[5]}
    lab_ko = {"apple": "사과", "fruit": "이웃 과일", "table": "table", "crate": "crate", "robot": "robot", "room": "바닥/방", "none": "미분류", "obstacle": "장애물"}
    bottom = np.zeros(len(PHASES)); shown["composition"] = {}
    for cpart in comp:
        v = []
        for p in PHASES:
            e = G["all"][p]
            if not e["n_with_target"]:
                v.append(0.0); continue
            if cpart == "apple":
                v.append(e["precision"]["mean"])
            elif cpart == "fruit":
                v.append(sum(e["contamination_mean"].get(f, 0) for f in FRUITS))
            else:
                v.append(e["contamination_mean"].get(cpart, 0.0))
        ax.bar(x, v, 0.6, bottom=bottom, color=cc[cpart], label=T(lab_ko[cpart], cpart)); bottom += np.array(v)
        shown["composition"][cpart] = dict(zip(PHASES, v))
    ax.set_xticks(x); ax.set_xticklabels([PH_KO[p] if KO else p for p in PHASES], fontsize=7.5, rotation=20); ax.set_ylim(0, 1.05)
    ax.set_title(T("선택 cluster 점의 GT 구성 (평균, 전체 24 장면)", "composition of target_points"), fontsize=9, loc="left")
    ax.legend(fontsize=6.6, frameon=False, loc="lower left", ncol=2)
    fig.suptitle(T("ATS 선택 cluster vs MuJoCo GT, latch phase 별 (요청 단위; target 이 있는 요청만)",
                   "ATS selected cluster vs MuJoCo GT by latch phase"), fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, path, {"values": shown, "requests_by_phase": N["requests"]["by_phase"], "scenes_by_type": N["scenes"]["by_type"]}, force)


def fig_traces(scenes, eps, tf, path, force):
    sel = [s for e in eps for s in scenes if s["episode"] == e]
    fig, axes = plt.subplots(len(sel), 1, figsize=(15, 3.3 * len(sel)), sharex=True); FS.style_axes(fig, axes)
    axes = np.atleast_1d(axes); shown = {}
    pc = {"searching": "#efeeea", "latched": "#e3eefb", "closing": "#fde8dc", "held": "#dff3ea", "placed": "#f3f2ee"}
    for ax, s in zip(axes, sel):
        R = s["rows"]; k = np.array([r["k"] for r in R])
        for r in R:
            ax.axvspan(r["k"] - 0.5, r["k"] + 0.5, color=pc.get(r["phase"], "#ffffff"), lw=0, zorder=0)
        share = lambda o: np.array([np.nan if r["att_share"].get(o) is None else r["att_share"][o] for r in R])
        ax.plot(k, share(tf), "-", color=FS.CATEGORICAL[7], lw=1.8, label=T("attention 비율: 사과", "share: apple"))
        ax.plot(k, share("crate"), "-", color=FS.CATEGORICAL[3], lw=1.4, label=T("attention 비율: crate", "share: crate"))
        oth = np.nansum(np.stack([share(f) for f in FRUITS if f != tf]), 0)
        ax.plot(k, oth, "-", color=FS.CATEGORICAL[1], lw=1.2, label=T("attention 비율: 다른 과일 합", "share: other fruits"))
        if s["obstacle"]:
            ax.plot(k, share("obstacle"), "-", color=FS.CATEGORICAL[5], lw=1.2, label=T("attention 비율: 장애물", "share: obstacle"))
        # tracked-target correctness strip
        ok = [r["k"] for r in R if r["mid"] is not None and r["id_mc"] == tf]
        bad = [r["k"] for r in R if mem_cat(r, tf) == "stale"]
        swk = [r["k"] for r in R if mem_cat(r, tf) == "switched"]
        none = [r["k"] for r in R if r["mid"] is None]
        ax.scatter(ok, [1.09] * len(ok), marker="o", s=14, color=FS.CATEGORICAL[2], label=T("추적 target = 사과", "tracked = apple"), zorder=4)
        ax.scatter(bad, [1.09] * len(bad), marker="x", s=22, color=FS.CATEGORICAL[7], label=T("추적 target 같은 id, 기억 위치가 사과 밖 (stale)", "stale memory"), zorder=4)
        ax.scatter(swk, [1.09] * len(swk), marker="D", s=16, color=FS.CATEGORICAL[4], label=T("추적 target 다른 객체로 전환", "switched"), zorder=4)
        ax.scatter(none, [1.09] * len(none), marker="_", s=22, color=FS.INK_2, label=T("추적 target 없음", "no tracked target"), zorder=4)
        occ = [r["k"] for r in R if r["mstate"] == "occluded"]
        ax.scatter(occ, [1.17] * len(occ), marker="s", s=8, color=FS.INK_2, label=T("state = occluded", "occluded"), zorder=4)
        ax.set_ylim(-0.03, 1.24); ax.set_ylabel(T("attention 비율", "attention share"), fontsize=8)
        nf = sum(1 for r in R if r["att_top"] is not None and r["att_top"] != tf)
        ax.set_title(T(f"ep{s['episode']} ({'이웃' if s['type'] == 'neighbour' else '이웃 없음'}{', 장애물' if s['obstacle'] else ''}; "
                       f"placed 도달 {'예' if any(r['phase'] == 'placed' for r in R) else '아니오'}): attention 최대 객체 ≠ 사과 {nf}/{len(R)} 요청",
                       f"ep{s['episode']}"), fontsize=9, loc="left")
        shown[str(s["episode"])] = {"k": k.tolist(), "phase": [r["phase"] for r in R], "share_target": share(tf).tolist(),
                                    "share_crate": share("crate").tolist(), "share_other_fruits": oth.tolist(),
                                    "tracked_ok": ok, "tracked_stale": bad, "tracked_switched": swk, "tracked_none": none, "occluded": occ,
                                    "n_top_not_target": nf, "type": s["type"]}
    from matplotlib.patches import Patch
    h, l_ = [], []
    for ax in axes:   # union over panels (the obstacle line exists only in obstacle scenes)
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l_:
                h.append(hh); l_.append(ll)
    h += [Patch(color=pc[p]) for p in PHASES]; l_ += [PH_KO[p] if KO else p for p in PHASES]
    axes[0].legend(h, l_, fontsize=6.8, frameon=False, ncol=6, loc="lower left", bbox_to_anchor=(0, 1.13))
    axes[-1].set_xlabel(T("서버 요청 k (chunk)", "request k"), fontsize=8)
    fig.suptitle(T("요청별 GT 객체의 attention 비율과 추적 target(manipulated) 의 정답 여부 (위 띠: 추적 target 이 가장 가까운 GT 가 사과인가)",
                   "attention share per GT object vs tracked-target correctness"), fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, path, {"episodes": shown}, force)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--fig", type=pathlib.Path, required=True)
    ap.add_argument("--kit", default="IFG-APPLE-SPLIT")
    ap.add_argument("--step", type=int, default=1, help="use every N-th request (1 = all)")
    ap.add_argument("--cache", type=pathlib.Path, default=None, help="per-request records json (read if present, else written)")
    ap.add_argument("--verify", type=pathlib.Path, default=pathlib.Path("AG3S/ag3s/docs/handoff/IFG-AUDIT-ATS.verify.json"))
    ap.add_argument("--panel", default=None, metavar="EP:K", help="scene and request of figure a (default: rule in the notes)")
    ap.add_argument("--traces", default=None, metavar="EP,EP,EP", help="scenes of figure c (default: rule in the notes)")
    ap.add_argument("--force", action="store_true", help="overwrite this study's own outputs")
    a = ap.parse_args(); a.fig.mkdir(parents=True, exist_ok=True)
    K = a.run / "verify" / a.kit
    xml = a.run / "snapshot_split/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
    S = Scene(str(xml))
    if a.cache and a.cache.exists():
        scenes = json.load(open(a.cache))
    else:
        t0 = time.time()
        scenes = collect(a.run, K, S, a.step, lambda x: print(x, flush=True))
        print(f"collected in {time.time() - t0:.0f}s", flush=True)
        if a.cache:
            a.cache.write_text(json.dumps(scenes, default=float))
    tf = scenes[0]["target"]
    N = aggregate(scenes)
    # figure a: neighbour scene with the closest neighbour fruit; request = middle latched+visible request with a target
    if a.panel:
        pe, pk = (int(v) for v in a.panel.split(":"))
    else:
        nb = sorted((s for s in scenes if s["type"] == "neighbour"), key=lambda s: min(s["neighbour_centre_dist_mm"].values()))
        pe = nb[0]["episode"]
        ks = [r["k"] for r in nb[0]["rows"] if r["phase"] == "latched" and r["mstate"] == "visible" and r["has_target"] and r["recall"] is not None]
        pk = ks[len(ks) // 2]
    sc_a = next(s for s in scenes if s["episode"] == pe)
    fa = a.fig / "ifg-audit-ats-scene.png"; fig_panel(S, sc_a, pk, fa, a.force)
    fb = a.fig / "ifg-audit-ats-phase.png"; fig_phase(N, fb, a.force)
    # figure c: panel-a scene, the reached-placed scene with most attention-top != target requests, one scene that never reached placed
    if a.traces:
        eps = [int(v) for v in a.traces.split(",")]
    else:
        nf = lambda s: sum(1 for r in s["rows"] if r["att_top"] is not None and r["att_top"] != tf)
        placed = [s for s in scenes if any(r["phase"] == "placed" for r in s["rows"]) and s["episode"] != pe]
        failed = [s for s in scenes if not any(r["phase"] == "placed" for r in s["rows"]) and s["episode"] != pe]
        eps = [pe] + [max(placed, key=nf)["episode"]] + ([max(failed, key=nf)["episode"]] if failed else [])
    fc = a.fig / "ifg-audit-ats-attention-memory.png"; fig_traces(scenes, eps, tf, fc, a.force)
    if a.verify.exists() and not a.force:
        raise SystemExit(f"refusing to overwrite {a.verify}")
    import datetime
    kst = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    a.verify.write_text(json.dumps({
        "step": "IFG-AUDIT-ATS",
        "written": kst.strftime("%Y-%m-%d %H:%M KST"),
        "data": {"run": str(a.run), "kit": str(K), "arm": "SPL (current serving: noun_target:L10H7 attention + attention-split partition, split supersedes anchor)",
                 "model_xml": str(xml), "scenes": [s["episode"] for s in scenes], "request_step": a.step,
                 "requests_measured": N["requests"]["n"], "subsampled": a.step > 1},
        "definitions": __doc__,
        "scripts": [SCRIPT],
        "figures": {"a_scene_panel": str(fa), "b_phase_metrics": str(fb), "c_attention_memory": str(fc),
                    "panel_choice": {"episode": pe, "request": pk}, "trace_episodes": eps},
        "numbers": N,
        "per_scene": [{k: s[k] for k in ("episode", "seed", "side", "type", "obstacle", "neighbour_centre_dist_mm", "neighbour_surface_gap_mm",
                                         "seq_mismatch", "n_requests")} for s in scenes],
        "notes": [
            "figure a rule: neighbour scene with the smallest neighbour centre distance; request = middle latched + visible request with a target and a visible GT target",
            "figure c rule: the figure-a scene; the scene that reached placed with the most requests whose attention_top != target; the never-placed scene with the most such requests",
            "manipulated.state 'kept' did not occur in these records (only visible / occluded / no manipulated)",
            "occluded-state target_points are the tracker's stored geometry (observed at last_seen_frame), compared with the GT at the request",
        ]}, indent=1, default=float))
    print(a.verify)


if __name__ == "__main__":
    main()
