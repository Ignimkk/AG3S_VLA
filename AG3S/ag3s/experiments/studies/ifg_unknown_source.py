"""IFG-UNKNOWN — which observation made the surface voxel behind every 'unknown' active TrajOpt row of IFG-PRECHECK-A. CPU only, read only.

  cd /mnt/dev/AG3S_VLA-development; MUJOCO_GL=osmesa JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
      PYTHONPATH=.:integration/rby1/manipulation/src .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_unknown_source \
      collect --run <RUN> --precheck <PRECHECK raw_v2 dir> --eps 320,1313 --out $AG3S_RUN_DIR/verify/IFG-UNKNOWN/raw
  ... report --raw <dir> --precheck <dir> --fig AG3S/ag3s/docs/figures/ifg-unknown --verify AG3S/ag3s/docs/handoff/IFG-UNKNOWN.verify.json

Every rule (unknown row, surface point, observations, server self-filter mask, point label, moved test, row source, U2 distance and
visibility, U3, U4) is fixed in `AG3S/ag3s/docs/handoff/IFG-UNKNOWN.verify.json` `definitions`, written before this script classified
anything; this script follows it.

Inputs (read only): IFG-PRECHECK-A raw_v2 rows (`ifg_precheck_a.py` collect), the IFG-APPLE-SPLIT record (server chunk npz with ESDF
grids + summary, client step npz with the three depth images, K, T_base_cam, qpos), the run's MuJoCo snapshot.
"""
from __future__ import annotations

import argparse, ast, contextlib, datetime, glob, io, json, pathlib, sys, time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4]))

SCRIPT = "AG3S/ag3s/experiments/studies/ifg_unknown_source.py"
KIT = "IFG-APPLE-SPLIT"
CAMS = (("zed_left", "head"), ("wrist_cam_l", "left_wrist"), ("wrist_cam_r", "right_wrist"))
H_IMG, W_IMG = 480, 640
DEPTH_MIN, DEPTH_MAX = 0.05, 3.0
R_SUP = {"fine": 0.010, "target_free": 0.010, "coarse": 0.020}
TOL_LABEL = 0.005           # rendered vs recorded depth, and geometric label fallback
TOL_MOVED = 0.005
VIS_TOL = 0.010
SHARE_MIN = 0.25
NEAR_MAX = 0.100
MOVED_TEST = "body_pose"     # v2 (2026-10-22 02:55 KST); v1 = "gt_distance" (see verify.json definitions.moved_test_revision)
CROP = 0.12
FAIL = (211, 320, 526, 954, 979, 1313)
CTRL = (250, 126, 572)
PRE = ("latched", "closing")
GT_CLASSES = ("target", "banana", "orange", "pear", "crate", "post", "table", "room", "robot")
POINT_LABELS = ("robot_hand", "robot_inactive_hand", "robot_body", "target", "neighbour", "crate", "post", "table", "room", "none",
                "unlabelled")
SOURCES = ("robot_self", "target_at_request_gt", "neighbour_fruit", "support_surface", "memory_stale", "field_artefact", "other",
           "ambiguous")


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return x
    return x


# ------------------------------------------------------------------------------------------------ GT geometry (request state)
class GTk:
    """ifg_precheck_a.GT (ifg_audit_ats.Scene + analytic post) with an 'all robot' class (every collidable robot geom)."""

    def __init__(self, xml, geom_sets, obstacle):
        from AG3S.ag3s.experiments.studies.ifg_precheck_a import GT
        self.g = GT(xml, geom_sets)
        self.g.S.obstacle = obstacle
        self.S = self.g.S

    def set_qpos(self, q):
        self.S.set_qpos(np.asarray(q, float))

    def dist(self, c, X):
        X = np.asarray(X, float).reshape(-1, 3)
        if not len(X):
            return np.zeros(0)
        if c == "robot":
            S = self.S; out = np.full(len(X), np.inf)
            for g in S.cls["robot"]:
                if np.min(np.linalg.norm(X - S.d.geom_xpos[g], axis=1)) > S.m.geom_rbound[g] + 0.12:
                    continue
                out = np.minimum(out, S._geom_dist(g, X, 0.12))
            return out
        return self.g.class_dist(c, X)

    def all(self, X):
        return {c: self.dist(c, X) for c in GT_CLASSES}


# ------------------------------------------------------------------------------------------------ client scene for labels
class Labeller:
    """Segmentation + depth renders of the client scene (obstacle placed) with the recorded camera poses."""

    def __init__(self, xml_obst, meta, geom_sets):
        import mujoco
        from rby1_manipulation.simulation.pick_place_obstacles import PickPlaceObstacleManager
        self.mj = mujoco
        m = self.m = mujoco.MjModel.from_xml_path(str(xml_obst)); d = self.d = mujoco.MjData(m)
        a = list(meta["argv"])
        self.obstacle_profile = None
        if "--obstacle-config" in a and "--obstacle-profile" in a:
            cfg = json.load(open(a[a.index("--obstacle-config") + 1])); prof = a[a.index("--obstacle-profile") + 1]
            self.mgr = PickPlaceObstacleManager(m, d, cfg); self.mgr.activate(prof); self.obstacle_profile = prof
        self.mocap_pos = d.mocap_pos.copy(); self.mocap_quat = d.mocap_quat.copy()
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        base = B("base")
        bname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]

        def under_base(b):
            while b > 0 and b != base:
                b = int(m.body_parentid[b])
            return b == base
        gid = lambda n: int(n[5:]) if n.startswith("geom#") else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        hand_b = {int(m.geom_bodyid[gid(n)]) for n in geom_sets.get("hand", [])}
        inact_b = {int(m.geom_bodyid[gid(n)]) for n in geom_sets.get("inactive_hand", [])}
        lut = np.full(m.ngeom, POINT_LABELS.index("room"), np.int16)
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g]); n = bname[b]
            if n == "apple":
                lut[g] = POINT_LABELS.index("target")
            elif n in ("banana", "orange", "pear"):
                lut[g] = POINT_LABELS.index("neighbour")
            elif n == "crate":
                lut[g] = POINT_LABELS.index("crate")
            elif n.startswith("pick_obstacle_"):
                lut[g] = POINT_LABELS.index("post")
            elif n == "table":
                lut[g] = POINT_LABELS.index("table")
            elif under_base(b):
                lut[g] = POINT_LABELS.index("robot_hand" if b in hand_b else ("robot_inactive_hand" if b in inact_b else "robot_body"))
        self.lut = lut
        self.hand_bodies = sorted(bname[b] for b in hand_b); self.inactive_bodies = sorted(bname[b] for b in inact_b)
        self.R = mujoco.Renderer(m, H_IMG, W_IMG)
        self.cam = {c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, c) for c, _ in CAMS}

    def render(self, qpos, cam, T):
        mj, m, d, R = self.mj, self.m, self.d, self.R
        d.qpos[:] = qpos; d.mocap_pos[:] = self.mocap_pos; d.mocap_quat[:] = self.mocap_quat
        mj.mj_forward(m, d)
        c = self.cam[cam]
        d.cam_xpos[c] = T[:3, 3]; d.cam_xmat[c] = (T[:3, :3] @ np.diag([1.0, -1.0, -1.0])).ravel()
        R.enable_depth_rendering(); R.update_scene(d, camera=cam); D = R.render().copy(); R.disable_depth_rendering()
        R.enable_segmentation_rendering(); R.update_scene(d, camera=cam); S = R.render().copy(); R.disable_segmentation_rendering()
        g = S[..., 0].astype(np.int64); ok = (S[..., 1] == int(mj.mjtObj.mjOBJ_GEOM)) & (g >= 0)
        lab = np.full(g.shape, POINT_LABELS.index("none"), np.int16)
        lab[ok] = self.lut[g[ok]]
        body = np.full(g.shape, -1, np.int32)
        body[ok] = m.geom_bodyid[g[ok]]
        return D, lab, body

    def close(self):
        self.R.close()


# ------------------------------------------------------------------------------------------------ helpers
def backproject_full(dep, K):
    v, u = np.nonzero((dep >= DEPTH_MIN) & (dep <= DEPTH_MAX) & np.isfinite(dep))
    z = dep[v, u]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], 1)
    return pc, u, v


def grid_of(z, tier):
    pre = {"target_free": "esdf_free", "fine": "esdf", "coarse": "esdf_coarse"}.get(tier)
    if pre is None or f"{pre}_origin" not in z.files:
        return None
    return pre, np.asarray(z[f"{pre}_origin"], np.float64), float(z[f"{pre}_voxel_size"])


def project(P, K, T):
    """world points -> (u, v, z_cam) in the recorded camera."""
    pc = (P - T[:3, 3]) @ T[:3, :3]
    zc = pc[:, 2]
    u = K[0, 0] * pc[:, 0] / np.maximum(zc, 1e-9) + K[0, 2]
    v = K[1, 1] * pc[:, 1] / np.maximum(zc, 1e-9) + K[1, 2]
    return u, v, zc


# ------------------------------------------------------------------------------------------------ collect
def collect_scene(run_root, pre_dir, ep, out_dir, log, fig_keys):
    import mujoco
    from scipy.spatial import cKDTree
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field
    from AG3S.ag3s.experiments.studies.ifg_audit_ats import obstacle_of
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, set_finger_joints_from_scene
    from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
    from AG3S.ag3s.config import PointCloudConfig
    from AG3S.ag3s.stages.reconstruction import backproject
    from AG3S.ag3s.stages.robot_filter import depth_robot_mask
    t0 = time.time()
    pj = json.load(open(pre_dir / f"{KIT}_ep{ep}.json"))
    pre_k = {q["k"]: q for q in pj["requests"] if q["pre_grasp"]}
    U = [r for r in pj["rows"] if r["traj"] == "ref" and r["binding"] and r["k"] in pre_k and r["src"] == "unknown"]
    K = run_root / "verify" / KIT
    js = json.load(open(next(iter(glob.glob(str(K / "replay" / f"runs_SPL_SPL_ep{ep}_s*.json"))))))
    blk = js["server_chunk_alignment"]["block"]
    cdir = pathlib.Path(blk["chunk_dir"]); off = int(blk["offset"])
    rd = pathlib.Path(js["run_dir"]); sdir = rd / "ag3s" / "run_0000"
    meta = json.load(open(sdir / "meta.json"))
    xml = pathlib.Path(pj["meta"]["xml"]); xml_obst = xml.with_name("model_transport_pick_place_obstacles.xml")
    pl = [json.loads(x) for x in open(rd / "frames.jsonl") if x.strip()]
    pl = [r for r in pl if r.get("kind") == "planning"]
    fid = {"n_unknown_rows": len(U)}
    out = {"meta": {"ep": ep, "kit": KIT, "run_dir": str(rd), "chunk_dir": str(cdir), "offset": off, "xml": str(xml),
                    "xml_render": str(xml_obst), "n_requests": len(pl), "obstacle_profile": None},
           "rows": [], "fidelity": fid, "requests": {}}
    if not U:
        out["meta"]["seconds"] = round(time.time() - t0, 1)
        (out_dir / f"ep{ep}.json").write_text(json.dumps(out, default=float)); log(f"ep{ep}: no unknown rows"); return
    ks = sorted({r["k"] for r in U}); kmax = max(ks)
    # ---------------- per request: field, surface points
    S_all = []
    per_k = {}
    fd = []
    frames_ok = []
    for k in ks:
        with np.load(cdir / f"chunk_{off + k:05d}.npz", allow_pickle=True) as z0:
            z = {kk: z0[kk] for kk in z0.files}
        zf = type("Z", (), {"files": list(z.keys()), "__getitem__": lambda s, kk: z[kk]})()
        s = json.loads(str(z["summary_json"]))
        frames_ok.append(int((s.get("esdf_stats") or {}).get("frames", -1)) == k + 1)
        with contextlib.redirect_stdout(io.StringIO()):
            field = build_field(zf, outside_distance=1.0)
        rr = [r for r in U if r["k"] == k]
        P = np.array([r["p"] for r in rr], np.float64)
        free = np.array([r["layer"] == "target_free" for r in rr])
        d = np.asarray(field.distance(P), np.float64).copy(); g = np.asarray(field.gradient(P), np.float64).reshape(-1, 3).copy()
        tier = np.asarray(field.answer_tier(P), dtype=object)
        if free.any():
            d[free] = np.asarray(field.target_free_distance(P[free]), np.float64)
            gf, tf = field.target_free_gradient(P[free], return_tier=True)
            g[free] = np.asarray(gf, np.float64).reshape(-1, 3); tier[free] = np.asarray(tf, dtype=object)
        n = np.linalg.norm(g, axis=1, keepdims=True)
        Ssurf = P - d[:, None] * np.where(n > 1e-12, g / np.maximum(n, 1e-12), 0.0)
        ds = np.asarray(field.distance(Ssurf), np.float64).copy()
        if free.any():
            ds[free] = np.asarray(field.target_free_distance(Ssurf[free]), np.float64)
        fd += list(np.abs(d * 1e3 - np.array([r["d_mm"] for r in rr])))
        vox = []
        for i, r in enumerate(rr):
            gi = grid_of(zf, r["tier"])
            vox.append(None if gi is None else (gi[0], *[int(x) for x in np.round((Ssurf[i] - gi[1]) / gi[2])]))
        per_k[k] = dict(rows=rr, P=P, d=d, s=Ssurf, ds=ds, tier_re=tier.astype(str), vox=vox, gnorm=n[:, 0], summary=s)
        S_all.append(Ssurf)
    fid["field_abs_diff_mm_max"] = round(float(np.max(fd)), 5)
    fid["field_abs_diff_mm_n_over_0.05"] = int(np.sum(np.array(fd) > 0.05))
    fid["frames_eq_k_plus_1"] = f"{sum(frames_ok)}/{len(frames_ok)}"
    fid["tier_recomputed_equal"] = int(sum(int(t == r["tier"]) for k in ks for t, r in zip(per_k[k]["tier_re"], per_k[k]["rows"])))
    S_all = np.concatenate(S_all)
    lo = S_all.min(0) - CROP; hi = S_all.max(0) + CROP
    # ---------------- observations 0..kmax with the server self-filter mask
    mm = mujoco.MjModel.from_xml_path(str(xml)); dd = mujoco.MjData(mm)
    tscene = TransportScene.attach(mm, dd)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rob = build_robot_model(tscene)
    obs = defaultdict(list)
    body_ids = {n: mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("apple", "banana", "orange", "pear", "crate")}
    body_ids = {n: i for n, i in body_ids.items() if i >= 0}
    pose = {}
    mask_eq, mask_diff, mask_bad = 0, [], []
    align_bad = 0
    prev_man = None
    for f in range(kmax + 1):
        st = np.load(sdir / f"step_{f:05d}.npz")
        if int(st["t_step"]) != int(pl[f]["t_step"]) or np.abs(st["qpos"] - np.asarray(pl[f]["qpos"])).max() > 1e-9:
            align_bad += 1
        with np.load(cdir / f"chunk_{off + f:05d}.npz", allow_pickle=True) as zc:
            sf = json.loads(str(zc["summary_json"]))
        pcam = lit((sf.get("esdf_stats") or {}).get("per_camera")) or {}
        dd.qpos[:] = st["qpos"]; mujoco.mj_forward(mm, dd)
        pose[f] = {bn: (dd.xpos[bi].copy(), dd.xmat[bi].reshape(3, 3).copy()) for bn, bi in body_ids.items()}
        set_finger_joints_from_scene(tscene, rob); q = tscene.robot_state()
        gc, gr = None, 0.0
        if prev_man:
            gc = np.asarray(prev_man["centroid"], np.float64); gr = float(prev_man["radius"]) + 0.005
        for ci, (cam, cname) in enumerate(CAMS):
            dep = st[f"depth_{cam}"].astype(np.float64) / 1000.0
            Kc = np.asarray(st[f"K_{cam}"], np.float64); T = np.asarray(st[f"T_base_cam_{cam}"], np.float64)
            cfg = PointCloudConfig(range_max=2.0, self_filter_target_guard_radius=gr if gc is not None else 0.0)
            full = backproject(dep, Kc, T, cfg)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                mk = depth_robot_mask(full, dep.shape, rob, q, cfg, guard_centre=gc, device="cpu").mask
            rec = (pcam.get(cname) or {}).get("n_masked_px")
            if rec is not None:
                mask_eq += int(int(mk.sum()) == int(rec)); mask_diff.append(abs(int(mk.sum()) - int(rec)))
                if int(mk.sum()) != int(rec):
                    mask_bad.append({"frame": f, "camera": cname, "reconstructed_px": int(mk.sum()), "recorded_px": int(rec)})
            pc, u, v = backproject_full(dep, Kc)
            W = pc @ T[:3, :3].T + T[:3, 3]
            sel = np.all((W >= lo) & (W <= hi), 1)
            obs["P"].append(W[sel]); obs["f"].append(np.full(int(sel.sum()), f, np.int16)); obs["c"].append(np.full(int(sel.sum()), ci, np.int8))
            obs["u"].append(u[sel].astype(np.int16)); obs["v"].append(v[sel].astype(np.int16)); obs["m"].append(mk[v[sel], u[sel]])
        man = lit(sf.get("manipulated"))
        prev_man = man if isinstance(man, dict) and man.get("centroid") is not None and man.get("radius") is not None else None
    fid["mask_exact_match"] = f"{mask_eq}/{len(mask_diff)}"; fid["mask_abs_diff_px_max"] = int(max(mask_diff) if mask_diff else -1)
    fid["step_alignment_bad"] = align_bad; fid["mask_mismatches"] = mask_bad; fid["moved_test"] = MOVED_TEST
    O = {kk: np.concatenate(v_) for kk, v_ in obs.items()}
    fid["n_obs_points_cropped"] = int(len(O["P"])); fid["n_obs_points_masked_cropped"] = int(O["m"].sum())
    # supporting + nearest per row: one tree per request over the frames it integrated (0..k), cropped to its rows
    need = set()
    row_sup = {}
    for k in ks:
        pk = per_k[k]
        lo_k = pk["s"].min(0) - NEAR_MAX - 0.01; hi_k = pk["s"].max(0) + NEAR_MAX + 0.01
        box = (O["f"] <= k) & np.all((O["P"] >= lo_k) & (O["P"] <= hi_k), 1)
        idx_u = np.flatnonzero(box & ~O["m"]); idx_m = np.flatnonzero(box & O["m"])
        tree_u = cKDTree(O["P"][idx_u]) if len(idx_u) else None
        tree_m = cKDTree(O["P"][idx_m]) if len(idx_m) else None
        for i, r in enumerate(pk["rows"]):
            rs = R_SUP.get(r["tier"], 0.010)
            sup = np.zeros(0, np.int64); near, ndist = -1, None
            if tree_u is not None:
                sup = idx_u[np.asarray(tree_u.query_ball_point(pk["s"][i], rs), np.int64)]
                dn, jn = tree_u.query(pk["s"][i], distance_upper_bound=NEAR_MAX)
                if np.isfinite(dn):
                    near, ndist = int(idx_u[jn]), float(dn)
            nm = 0 if tree_m is None else int(len(tree_m.query_ball_point(pk["s"][i], rs)))
            row_sup[(k, i)] = (sup, near, ndist, nm)
            need.update(sup.tolist())
            if near >= 0:
                need.add(near)
    need = np.array(sorted(need), np.int64)
    # ---------------- labels of the needed points (render per frame / camera)
    lab = np.full(len(O["P"]), -1, np.int16); lab_how = np.zeros(len(O["P"]), np.int8)  # 1 render, 2 geometric, 3 unlabelled
    lab_body = {}
    L = Labeller(xml_obst, meta, js["geom_sets"]); out["meta"]["obstacle_profile"] = L.obstacle_profile
    out["meta"]["hand_bodies"] = L.hand_bodies; out["meta"]["inactive_hand_bodies"] = L.inactive_bodies
    obst, ocfg = obstacle_of(run_root, rd)
    gtk = GTk(xml, js["geom_sets"], obst)
    by_fc = defaultdict(list)
    for i in need:
        by_fc[(int(O["f"][i]), int(O["c"][i]))].append(i)
    rend_err = []
    for (f, ci), ii in sorted(by_fc.items()):
        ii = np.array(ii)
        st = np.load(sdir / f"step_{f:05d}.npz")
        cam = CAMS[ci][0]
        D, LB, BD = L.render(st["qpos"], cam, np.asarray(st[f"T_base_cam_{cam}"], np.float64))
        dep = st[f"depth_{cam}"].astype(np.float64) / 1000.0
        u, v = O["u"][ii].astype(int), O["v"][ii].astype(int)
        err = np.abs(dep[v, u] - D[v, u]); rend_err += list(err)
        ok = err <= TOL_LABEL
        lab[ii[ok]] = LB[v[ok], u[ok]]; lab_how[ii[ok]] = 1
        bnames = [L.mj.mj_id2name(L.m, L.mj.mjtObj.mjOBJ_BODY, int(b)) if b >= 0 else None for b in BD[v[ok], u[ok]]]
        for x, bn in zip(ii[ok], bnames):
            lab_body[int(x)] = bn
        bad = ii[~ok]
        if len(bad):
            gtk.set_qpos(st["qpos"])
            A = gtk.all(O["P"][bad])
            M = np.stack([A[c] for c in GT_CLASSES], 1)
            a = np.argmin(M, 1); v_ = M[np.arange(len(bad)), a]
            for j, b in enumerate(bad):
                if v_[j] <= TOL_LABEL:
                    c = GT_CLASSES[a[j]]
                    lab[b] = POINT_LABELS.index({"banana": "neighbour", "orange": "neighbour", "pear": "neighbour", "robot": "robot_body"}.get(c, c))
                    lab_how[b] = 2
                    lab_body[int(b)] = {"target": "apple"}.get(c, c)
                else:
                    lab[b] = POINT_LABELS.index("unlabelled"); lab_how[b] = 3
    L.close()
    rend_err = np.array(rend_err)
    fid["label_points"] = int(len(need)); fid["label_by_render"] = int((lab_how[need] == 1).sum())
    fid["label_by_geometry"] = int((lab_how[need] == 2).sum()); fid["unlabelled"] = int((lab_how[need] == 3).sum())
    fid["render_depth_abs_err_mm_median"] = round(float(np.median(rend_err)) * 1e3, 3) if len(rend_err) else None
    # ---------------- moved test + U2 + visibility, per request k
    for k in ks:
        pk = per_k[k]
        st = np.load(sdir / f"step_{k:05d}.npz")
        gtk.set_qpos(st["qpos"])
        # moved test for supporting points of earlier frames, labelled target / neighbour / crate
        cand = set()
        for i in range(len(pk["rows"])):
            sup = row_sup[(k, i)][0]
            cand.update(int(x) for x in sup if O["f"][x] < k and POINT_LABELS[lab[x]] in ("target", "neighbour", "crate"))
        cand = np.array(sorted(cand), np.int64)
        moved, moved_disp = {}, {}
        if len(cand) and MOVED_TEST == "gt_distance":
            X = O["P"][cand]; lb = np.array([POINT_LABELS[lab[x]] for x in cand])
            dt = gtk.dist("target", X); dc = gtk.dist("crate", X)
            dn = np.minimum.reduce([gtk.dist(c, X) for c in ("banana", "orange", "pear")])
            dsel = np.where(lb == "target", dt, np.where(lb == "crate", dc, dn))
            moved = {int(x): bool(dsel[j] > TOL_MOVED) for j, x in enumerate(cand)}
        elif len(cand):
            # v2: carry the point with its body from frame f to request k; moved when it is displaced by > TOL_MOVED
            for x in cand:
                x = int(x); bn = lab_body.get(x)
                if bn not in pose[k]:
                    moved[x] = False; continue
                xf, Rf = pose[int(O["f"][x])][bn]; xk, Rk = pose[k][bn]
                o = O["P"][x]; ok_ = Rk @ (Rf.T @ (o - xf)) + xk
                moved_disp[x] = float(np.linalg.norm(ok_ - o)); moved[x] = moved_disp[x] > TOL_MOVED
        Ds = gtk.all(pk["s"])
        vis = np.zeros(len(pk["s"]), bool); obsk = np.zeros(len(pk["s"]), bool)
        for cam, _ in CAMS:
            dep = st[f"depth_{cam}"].astype(np.float64) / 1000.0
            Kc = np.asarray(st[f"K_{cam}"], np.float64); T = np.asarray(st[f"T_base_cam_{cam}"], np.float64)
            u, v, zc = project(pk["s"], Kc, T)
            ui, vi = np.round(u).astype(int), np.round(v).astype(int)
            inside = (zc >= 0.05) & (ui >= 0) & (ui < W_IMG) & (vi >= 0) & (vi < H_IMG)
            dv = np.full(len(u), np.nan); dv[inside] = dep[vi[inside], ui[inside]]
            dv = np.where(dv > 0, dv, np.inf)   # no return = nothing in front
            vis |= inside & (dv >= zc - VIS_TOL)
            obsk |= inside & (np.abs(dv - zc) <= VIS_TOL)
        for i, r in enumerate(pk["rows"]):
            sup, near, ndist, nm = row_sup[(k, i)]
            labs = [POINT_LABELS[lab[x]] for x in sup]
            psrc = []
            for x, lb in zip(sup, labs):
                x = int(x)
                if lb.startswith("robot"):
                    psrc.append("robot_self")
                elif lb in ("target", "neighbour", "crate") and moved.get(x, False):
                    psrc.append("memory_stale")
                elif lb == "target":
                    psrc.append("target_at_request_gt")
                elif lb == "neighbour":
                    psrc.append("neighbour_fruit")
                elif lb in ("crate", "post", "table"):
                    psrc.append("support_surface")
                else:
                    psrc.append("other")
            cnt = Counter(psrc); n_s = len(psrc)
            shares = {c: round(v_ / n_s, 4) for c, v_ in cnt.items()} if n_s else {}
            matched = sorted(c for c, v_ in shares.items() if v_ >= SHARE_MIN)
            src = "field_artefact" if n_s == 0 else (matched[0] if len(matched) == 1 else "ambiguous")
            fr = O["f"][sup] if n_s else np.zeros(0, int)
            moved_cls = sorted({lb for x, lb in zip(sup, labs) if moved.get(int(x), False)})
            mdisp = [moved_disp[int(x)] for x in sup if int(x) in moved_disp]
            out["rows"].append({
                "ep": ep, "k": k, "state": r["state"], "step": r["step"], "q": r["q"], "link": r["link"], "layer": r["layer"], "tier": r["tier"],
                "tier_recomputed": str(pk["tier_re"][i]), "h_mm": r["h_mm"], "d_mm": r["d_mm"], "d_recomputed_mm": round(float(pk["d"][i]) * 1e3, 4),
                "r_mm": r["r_mm"], "best_any": r["best_any"], "err_mm_r1": r["err_mm"], "D_mm_p": r["D_mm"],
                "p": r["p"], "s": [round(float(x), 5) for x in pk["s"][i]], "d_at_s_mm": round(float(pk["ds"][i]) * 1e3, 3),
                "grad_norm": round(float(pk["gnorm"][i]), 4), "voxel": pk["vox"][i],
                "n_support": n_s, "n_support_masked": nm, "support_labels": dict(Counter(labs)), "support_point_sources": dict(cnt),
                "shares": shares, "matched": matched, "source": src,
                "support_frame_k": int((fr == k).sum()), "support_frame_min": int(fr.min()) if n_s else None,
                "moved_classes": moved_cls, "moved_disp_mm_median": (round(float(np.median(mdisp)) * 1e3, 2) if mdisp else None),
                "nearest_obs_mm": None if ndist is None else round(ndist * 1e3, 2),
                "nearest_obs_label": None if near < 0 else POINT_LABELS[lab[near]], "nearest_obs_frame": None if near < 0 else int(O["f"][near]),
                "D_s_k_mm": {c: (round(float(Ds[c][i]) * 1e3, 2) if np.isfinite(Ds[c][i]) else None) for c in GT_CLASSES},
                "visible_k": bool(vis[i]), "observed_k": bool(obsk[i])})
        q = pre_k[k]
        out["requests"][str(k)] = {"state": q["state"], "n_rows_ref": q.get("n_rows_ref"), "correction_mm": q.get("correction_mm"),
                                   "hold": q.get("hold"), "penetration_to_mm": q.get("penetration_to_mm")}
        if (ep, k) in fig_keys:
            # supporting points of this request (for the scene figure): all unmasked points of frames <= k within r_sup of any s
            allsup = np.unique(np.concatenate([row_sup[(k, i)][0] for i in range(len(pk["rows"]))] + [np.zeros(0, np.int64)]))
            np.savez_compressed(out_dir / f"fig_ep{ep}_k{k}.npz", P=O["P"][allsup].astype(np.float32), f=O["f"][allsup],
                                lab=lab[allsup], qpos=st["qpos"])
    out["meta"]["seconds"] = round(time.time() - t0, 1)
    out["meta"]["obstacle"] = ocfg
    (out_dir / f"ep{ep}.json").write_text(json.dumps(out, default=float))
    log(f"ep{ep}: {len(out['rows'])} rows, {len(ks)} requests, fidelity {fid}, {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ report
def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return None if not v else round(float(np.median(v)), 2)


def pct(v, q):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return None if not v else round(float(np.percentile(v, q)), 2)


def src_distance(r):
    D = r["D_s_k_mm"]
    g = lambda cs: min([D[c] for c in cs if D.get(c) is not None] or [None], key=lambda x: (x is None, abs(x) if x is not None else 0))  # noqa: E731
    s = r["source"]
    if s == "robot_self":
        v = g(["robot"])
    elif s == "target_at_request_gt":
        v = g(["target"])
    elif s == "neighbour_fruit":
        v = g(["banana", "orange", "pear"])
    elif s == "support_surface":
        v = g(["table", "crate", "post"])
    elif s == "memory_stale":
        mc = {"target": ["target"], "neighbour": ["banana", "orange", "pear"], "crate": ["crate"]}
        v = g(sum([mc[c] for c in r["moved_classes"] if c in mc], []) or list(GT_CLASSES))
    else:
        v = g(list(GT_CLASSES))
    return None if v is None else abs(v)


def summarise(rows):
    n = len(rows)
    c = Counter(r["source"] for r in rows)
    vox = {}
    for r in rows:
        key = (r["ep"], r["k"], tuple(r["voxel"]) if r["voxel"] else ("row", r["step"], r["q"]))
        if key not in vox or r["h_mm"] < vox[key]["h_mm"]:
            vox[key] = r
    cv = Counter(r["source"] for r in vox.values())
    nv = len(vox)
    out = {"n_rows": n, "n_voxels": nv,
           "row_counts": {s: c.get(s, 0) for s in SOURCES}, "row_fractions": {s: (round(c.get(s, 0) / n, 4) if n else None) for s in SOURCES},
           "voxel_counts": {s: cv.get(s, 0) for s in SOURCES}, "voxel_fractions": {s: (round(cv.get(s, 0) / nv, 4) if nv else None) for s in SOURCES},
           "n_requests_with_source": {s: len({(r["ep"], r["k"]) for r in rows if r["source"] == s}) for s in SOURCES}}
    rs = [r for r in rows if r["source"] == "robot_self"]
    out["robot_self_detail"] = {"by_point_label": dict(sum((Counter({k_: v_ for k_, v_ in r["support_labels"].items() if k_.startswith("robot")}) for r in rs), Counter())),
                                "rows_with_support_from_frame_k": sum(1 for r in rs if r["support_frame_k"] > 0),
                                "rows_support_only_earlier_frames": sum(1 for r in rs if r["support_frame_k"] == 0)}
    ss = [r for r in rows if r["source"] == "support_surface"]
    out["support_surface_detail"] = dict(sum((Counter({k_: v_ for k_, v_ in r["support_labels"].items() if k_ in ("table", "crate", "post")}) for r in ss), Counter()))
    out["ambiguous_combinations"] = dict(Counter("+".join(r["matched"]) for r in rows if r["source"] == "ambiguous"))
    fa = [r for r in rows if r["source"] == "field_artefact"]
    out["field_artefact_detail"] = {"nearest_obs_mm_median": med([r["nearest_obs_mm"] for r in fa]),
                                    "nearest_obs_mm_p95": pct([r["nearest_obs_mm"] for r in fa], 95),
                                    "n_no_obs_within_100mm": sum(1 for r in fa if r["nearest_obs_mm"] is None),
                                    "nearest_obs_label": dict(Counter(r["nearest_obs_label"] for r in fa)),
                                    "rows_with_masked_points_within_r_sup": sum(1 for r in fa if r["n_support_masked"] > 0)}
    out["rows_with_masked_points_within_r_sup"] = sum(1 for r in rows if r["n_support_masked"] > 0)
    # U2
    u2 = {}
    for s in SOURCES:
        rr = [r for r in rows if r["source"] == s]
        if not rr:
            continue
        dist = [src_distance(r) for r in rr]
        u2[s] = {"n_rows": len(rr), "gt_distance_mm_median": med(dist), "gt_distance_mm_p95": pct(dist, 95),
                 "visible_k_fraction": round(float(np.mean([r["visible_k"] for r in rr])), 4),
                 "observed_k_fraction": round(float(np.mean([r["observed_k"] for r in rr])), 4),
                 "d_at_s_mm_median": med([r["d_at_s_mm"] for r in rr])}
    out["U2"] = u2
    # field nearer / farther than GT at p (sign of best class error at the sphere centre)
    sg = []
    for r in rows:
        Dp = [v for v in r["D_mm_p"].values() if v is not None]
        if Dp:
            b = min(Dp, key=lambda x: abs(x - r["d_mm"]))
            sg.append(b - r["d_mm"])
    out["p_best_gt_minus_field_mm"] = {"n": len(sg), "n_gt_farther": int(sum(1 for x in sg if x > 0)), "n_gt_nearer": int(sum(1 for x in sg if x < 0)),
                                       "median": med(sg)}
    return out


def report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from AG3S.ag3s.experiments.common import figstyle as FS
    from AG3S.ag3s.experiments.studies.ifg_precheck_a import per_request, load_raw
    KO = FS.use_korean()
    T = (lambda ko, en: ko if KO else en)  # noqa: E731
    data = {int(json.load(open(f))["meta"]["ep"]): json.load(open(f)) for f in sorted(a.raw.glob("ep*.json"))}
    pre = load_raw(a.precheck)
    N, per_scene = {}, {}
    fails = [e for e in FAIL if e in data]; ctrls = [e for e in CTRL if e in data]
    rows_f = [r for e in fails for r in data[e]["rows"]]; rows_c = [r for e in ctrls for r in data[e]["rows"]]
    N["U1_U2_fail6_pre_grasp"] = summarise(rows_f)
    N["U1_U2_fail6_by_phase"] = {ph: summarise([r for r in rows_f if r["state"] == ph]) for ph in PRE}
    N["U1_U2_fail6_by_layer"] = {L_: summarise([r for r in rows_f if r["layer"] == L_]) for L_ in ("target_free", "main")}
    N["U4_control3_pre_grasp"] = summarise(rows_c)
    if a.raw_v1 is not None:
        d1 = {int(json.load(open(f))["meta"]["ep"]): json.load(open(f)) for f in sorted(a.raw_v1.glob("ep*.json"))}
        r1f = [r for e in FAIL if e in d1 for r in d1[e]["rows"]]; r1c = [r for e in CTRL if e in d1 for r in d1[e]["rows"]]
        N["v1_moved_test_gt_distance"] = {"raw": str(a.raw_v1),
                                         "fail6": {k_: summarise(r1f)[k_] for k_ in ("n_rows", "row_counts", "row_fractions", "voxel_fractions")},
                                         "control3": {k_: summarise(r1c)[k_] for k_ in ("n_rows", "row_counts", "row_fractions")},
                                         "per_scene_row_counts": {str(e): summarise(d1[e]["rows"])["row_counts"] for e in sorted(d1) if d1[e]["rows"]}}
    for e in sorted(data):
        rr = data[e]["rows"]
        per_scene[str(e)] = {"role": "fail" if e in FAIL else "control", "fidelity": data[e]["fidelity"], "seconds": data[e]["meta"]["seconds"],
                             "obstacle_profile": data[e]["meta"].get("obstacle_profile"),
                             "all": summarise(rr) if rr else {"n_rows": 0},
                             "by_phase": {ph: summarise([r for r in rr if r["state"] == ph]) for ph in PRE if any(r["state"] == ph for r in rr)}}
    # ---------------- U3
    u3 = []
    for e in fails:
        for q in per_request(pre[(KIT, e)]):
            if not q.get("pre_grasp") or "skipped" in q or q.get("dominant_group") != "unknown":
                continue
            rr = [r for r in data[e]["rows"] if r["k"] == q["k"]]
            w = min(rr, key=lambda r: r["h_mm"]) if rr else None
            u3.append({"scene": e, "k": q["k"], "state": q["state"], "n_active_ref": q.get("n_rows_ref"), "n_unknown": len(rr),
                       "unknown_source_counts": dict(Counter(r["source"] for r in rr)),
                       "worst_row_source": w["source"] if w else None, "worst_row_link": w["link"] if w else None,
                       "worst_row_h_mm": w["h_mm"] if w else None,
                       "correction_mm": round(q["correction_mm"], 2), "hold": q["hold"], "penetration_to_mm": round(q["penetration_to_mm"] or 0.0, 3),
                       "out_worst_group": q.get("out_worst_group")})
    N["U3_requests_dominant_unknown"] = {"n_requests": len(u3), "n_hold": sum(1 for x in u3 if x["hold"]),
                                         "worst_row_source_counts": dict(Counter(x["worst_row_source"] for x in u3)),
                                         "correction_mm_median_by_worst_row_source": {s: med([x["correction_mm"] for x in u3 if x["worst_row_source"] == s])
                                                                                      for s in sorted({x["worst_row_source"] for x in u3})},
                                         "rows": u3}
    N["fidelity_pooled"] = {
        "field_abs_diff_mm_max": max(d_["fidelity"].get("field_abs_diff_mm_max", 0) for d_ in data.values()),
        "mask_exact_match": [d_["fidelity"].get("mask_exact_match") for d_ in data.values()],
        "mask_abs_diff_px_max": max(d_["fidelity"].get("mask_abs_diff_px_max", 0) for d_ in data.values()),
        "step_alignment_bad": sum(d_["fidelity"].get("step_alignment_bad", 0) for d_ in data.values()),
        "label_points": sum(d_["fidelity"].get("label_points", 0) for d_ in data.values()),
        "label_by_render": sum(d_["fidelity"].get("label_by_render", 0) for d_ in data.values()),
        "label_by_geometry": sum(d_["fidelity"].get("label_by_geometry", 0) for d_ in data.values()),
        "unlabelled": sum(d_["fidelity"].get("unlabelled", 0) for d_ in data.values())}
    a.fig.mkdir(parents=True, exist_ok=True)
    figs = [fig_scene(a, data, FS, T, plt), fig_graph(a, data, N, per_scene, FS, T, plt), fig_table(a, N, FS, T, plt)]
    v = json.load(open(a.verify))
    kst = datetime.timezone(datetime.timedelta(hours=9))
    v["written"] = v["written"].split(" (")[0] + " (definitions) / " + datetime.datetime.now(kst).strftime("%Y-%m-%d %H:%M KST") + " (numbers)"
    v["status"] = "measured"
    v["runs"] = [{"id": f"ep{e}", "raw_output": str(a.raw / f"ep{e}.json"), "seconds": data[e]["meta"]["seconds"]} for e in sorted(data)]
    v["numbers"] = N; v["per_scene"] = per_scene; v["figures"] = figs; v["scripts"] = [SCRIPT]
    a.verify.write_text(json.dumps(v, indent=1, ensure_ascii=False, default=float))
    print("wrote", a.verify)


def save(fig, path, data, FS, plt):
    path = pathlib.Path(path)
    if path.exists() or path.with_suffix(".json").exists():
        raise SystemExit(f"refusing to overwrite {path}")
    fig.savefig(path, dpi=170, facecolor=FS.SURFACE, bbox_inches="tight"); plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps({**data, "script": SCRIPT}, indent=1, ensure_ascii=False, default=float))
    print("wrote", path)
    return {"path": str(path), "sidecar": str(path.with_suffix(".json"))}


S_LABEL = {"robot_self": ("로봇 자기 몸", "robot self"), "target_at_request_gt": ("target 사과 (요청 GT 위치)", "target apple (request GT)"),
           "neighbour_fruit": ("이웃 과일", "neighbour fruit"), "support_surface": ("탁자 · 바구니 · 기둥", "table / crate / post"),
           "memory_stale": ("기억 (옮겨진 물체의 옛 관측)", "memory (stale observation)"), "field_artefact": ("거리장 artefact (관측 점 없음)", "field artefact (no obs.)"),
           "other": ("기타", "other"), "ambiguous": ("ambiguous (복수 출처)", "ambiguous")}


def scol(FS):
    C = FS.CATEGORICAL
    return {"robot_self": C[0], "target_at_request_gt": C[1], "neighbour_fruit": C[3], "support_surface": C[6], "memory_stale": C[5],
            "field_artefact": FS.INK, "other": C[4], "ambiguous": FS.GRID_INK}


def pick_fig_requests(data):
    out = []
    for e in (320, 1313):
        if e in data and data[e]["rows"]:
            c = Counter(r["k"] for r in data[e]["rows"])
            out.append((e, c.most_common(1)[0][0]))
    return out


def fig_scene(a, data, FS, T, plt):
    import mujoco
    col = scol(FS)
    picks = pick_fig_requests(data)
    fig, axs = plt.subplots(len(picks), 2, figsize=(14, 5.8 * len(picks)), gridspec_kw={"width_ratios": [1.35, 1]}, squeeze=False)
    FS.style_axes(fig, axs)
    side = {}
    W, H = 720, 540
    for ii, (e, k) in enumerate(picks):
        j = data[e]
        rows = [r for r in j["rows"] if r["k"] == k]
        m = mujoco.MjModel.from_xml_path(j["meta"]["xml_render"]); d = mujoco.MjData(m)
        meta = json.load(open(pathlib.Path(j["meta"]["run_dir"]) / "ag3s" / "run_0000" / "meta.json"))
        if j["meta"].get("obstacle_profile"):
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "integration/rby1/manipulation/src"))
            from rby1_manipulation.simulation.pick_place_obstacles import PickPlaceObstacleManager
            av = meta["argv"]
            PickPlaceObstacleManager(m, d, json.load(open(av[av.index("--obstacle-config") + 1]))).activate(j["meta"]["obstacle_profile"])
        fz = np.load(pathlib.Path(a.raw) / f"fig_ep{e}_k{k}.npz")
        d.qpos[:] = fz["qpos"]; mujoco.mj_forward(m, d)
        S = np.array([r["s"] for r in rows]); cen = np.median(S, 0)
        rnd = mujoco.Renderer(m, H, W)
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = cen; cam.distance = 0.6; cam.azimuth = 180.0; cam.elevation = -30.0
        rnd.update_scene(d, camera=cam); img = rnd.render()
        gc = rnd.scene.camera[0]
        pos = 0.5 * (np.array(rnd.scene.camera[0].pos) + np.array(rnd.scene.camera[1].pos)); fwd = np.array(gc.forward); up = np.array(gc.up); right = np.cross(fwd, up)
        near = gc.frustum_near; ht = gc.frustum_top; hw = ht * W / H
        rnd.close()

        def proj(p):
            v = np.atleast_2d(p) - pos; zc = v @ fwd
            return ((v @ right) * near / zc + hw) / (2 * hw) * W, (ht - (v @ up) * near / zc) / (2 * ht) * H
        ax = axs[ii, 0]; ax.imshow(img)
        cnt = Counter(r["source"] for r in rows)
        for s in SOURCES:
            rr = [r for r in rows if r["source"] == s]
            if not rr:
                continue
            u, vv = proj(np.array([r["s"] for r in rr]))
            ax.scatter(u, vv, s=9, color=col[s], edgecolor="white", linewidth=0.3, label=f"{T(*S_LABEL[s])} ({cnt[s]})", zorder=3)
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=7.5, frameon=True, loc="lower left", title=T("미상 행의 표면 voxel 출처 (U1)", "surface-voxel source of unknown rows (U1)"), title_fontsize=7.5)
        st = rows[0]["state"]
        ax.set_title(T(f"SPLIT ep{e} 요청 {k} ({st}) — 실제 씬 (MuJoCo, 요청 시점), 점 = 미상 행 {len(rows)}개의 표면점 s",
                       f"SPLIT ep{e} request {k} ({st}): real scene at the request, dots = surface points s of {len(rows)} unknown rows"), fontsize=8.5, loc="left")
        # side view (x-z): supporting observation points by label, surface points by source
        bx = axs[ii, 1]
        P = fz["P"]; lb = fz["lab"]
        lab_col = {"robot_hand": FS.CATEGORICAL[0], "robot_inactive_hand": FS.CATEGORICAL[0], "robot_body": FS.CATEGORICAL[0], "target": FS.CATEGORICAL[1],
                   "neighbour": FS.CATEGORICAL[3], "crate": FS.CATEGORICAL[6], "post": FS.CATEGORICAL[6], "table": FS.CATEGORICAL[2], "room": FS.CATEGORICAL[4],
                   "none": FS.GRID_INK, "unlabelled": FS.GRID_INK}
        for li, ln in enumerate(POINT_LABELS):
            sel = lb == li
            if sel.any():
                bx.scatter(P[sel, 0] * 1000, P[sel, 2] * 1000, s=1.5, color=lab_col[ln], alpha=0.35, zorder=1)
        for s in SOURCES:
            rr = [r for r in rows if r["source"] == s]
            if rr:
                Q = np.array([r["s"] for r in rr]) * 1000
                bx.scatter(Q[:, 0], Q[:, 2], s=10, color=col[s], edgecolor="white", linewidth=0.3, zorder=3)
        bx.set_xlim(cen[0] * 1000 - 120, cen[0] * 1000 + 120); bx.set_ylim(cen[2] * 1000 - 100, cen[2] * 1000 + 100); bx.set_aspect("equal")
        bx.set_xlabel("x (mm, base)", fontsize=8); bx.set_ylabel("z (mm, base)", fontsize=8)
        bx.set_title(T("옆에서 본 x–z: 작은 점 = 지지 관측 점 (요청 이하 프레임, mask 밖, 색 = 그 픽셀의 GT 물체), 큰 점 = 표면점 s",
                       "side view x-z: small = supporting observation points (frames <= k, unmasked; colour = GT object of the pixel), large = s"),
                     fontsize=7.5, loc="left")
        side[f"ep{e}_k{k}"] = {"source_counts": dict(cnt), "camera": {"lookat": cen.round(4).tolist(), "distance": 0.6, "azimuth": 180, "elevation": -30},
                               "n_support_points_drawn": int(len(P)), "support_point_label_counts": {POINT_LABELS[i]: int((lb == i).sum()) for i in np.unique(lb)},
                               "rows": [{"s": r["s"], "source": r["source"], "link": r["link"], "step": r["step"], "h_mm": r["h_mm"]} for r in rows]}
    fig.suptitle(T("IFG-UNKNOWN 실제 씬 — 미상 활성 제약 행의 표면 voxel 과 그것을 만든 관측 점의 출처",
                   "IFG-UNKNOWN real scene: surface voxels behind unknown active rows and the source of their observation points"), fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, a.fig / "ifg-unknown-scene.png", {"panels": side}, FS, plt)


def fig_graph(a, data, N, per_scene, FS, T, plt):
    col = scol(FS)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.4), gridspec_kw={"width_ratios": [1.3, 1.0, 1.0], "wspace": 0.38})
    FS.style_axes(fig, axs)
    out = {}
    # (a) row fractions per scene x phase
    ax = axs[0]
    labels, M, tot = [], [], []
    for e in [x for x in FAIL + CTRL if str(x) in per_scene]:
        for ph in PRE:
            b = per_scene[str(e)]["by_phase"].get(ph)
            if not b or not b.get("n_rows"):
                continue
            labels.append(f"ep{e} {ph[:5]} ({T('실패', 'fail') if e in FAIL else T('대조', 'ctrl')})")
            M.append([b["row_counts"][s] for s in SOURCES]); tot.append(b["n_rows"])
    M = np.array(M, float); F = M / np.maximum(M.sum(1, keepdims=True), 1)
    y = np.arange(len(labels))[::-1]; left = np.zeros(len(labels))
    for si, s in enumerate(SOURCES):
        ax.barh(y, F[:, si], left=left, color=col[s], height=0.7, label=T(*S_LABEL[s])); left += F[:, si]
    for yi, t_ in zip(y, tot):
        ax.text(1.02, yi, str(t_), va="center", fontsize=7, color=FS.INK_2)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5); ax.set_xlim(0, 1.12)
    ax.set_xlabel(T("미상 행 비율 (오른쪽 = 행 수)", "fraction of unknown rows (right = rows)"), fontsize=8)
    ax.set_title(T("U1/U4: 장면 · 단계별 미상 행의 출처", "U1/U4: source of unknown rows by scene and phase"), fontsize=8.5, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.9, -0.08), ncol=4)
    out["scene_phase"] = {"labels": labels, "counts": M.tolist(), "sources": list(SOURCES)}
    # (b) U2 distance per source (fail 6)
    bx = axs[1]
    rows = [r for e in FAIL if e in data for r in data[e]["rows"]]
    cats = [s for s in SOURCES if any(r["source"] == s for r in rows)]
    rng = np.random.default_rng(0); pts = {}
    for ci, s in enumerate(cats):
        dv = np.array([x for x in (src_distance(r) for r in rows if r["source"] == s) if x is not None])
        if not len(dv):
            continue
        x = ci + rng.uniform(-0.28, 0.28, len(dv))
        bx.scatter(x, np.maximum(dv, 0.1), s=4, color=col[s], alpha=0.35)
        bx.hlines(np.median(dv), ci - 0.35, ci + 0.35, color=FS.INK, lw=1.3)
        bx.hlines(np.percentile(dv, 95), ci - 0.2, ci + 0.2, color=FS.INK_2, lw=1.0, linestyle="--")
        pts[s] = {"median_mm": round(float(np.median(dv)), 2), "p95_mm": round(float(np.percentile(dv, 95)), 2), "n": int(len(dv))}
    bx.set_yscale("log"); bx.set_xticks(range(len(cats))); bx.set_xticklabels([T(*S_LABEL[s]) for s in cats], rotation=30, ha="right", fontsize=7)
    bx.set_ylabel(T("표면점 s 와 출처 GT 표면의 거리 (mm, 요청 시점)", "distance of s to the source's GT surface (mm, at request)"), fontsize=8)
    bx.set_title(T("U2: 출처별 GT 거리 (실선 중앙값, 점선 p95; 실패 6 장면)", "U2: GT distance by source (solid median, dashed p95)"), fontsize=8.5, loc="left")
    out["u2_distance"] = pts
    # (c) visibility per source
    cx = axs[2]
    u2 = N["U1_U2_fail6_pre_grasp"]["U2"]
    cs = [s for s in SOURCES if s in u2]
    xx = np.arange(len(cs))
    cx.bar(xx - 0.18, [u2[s]["visible_k_fraction"] for s in cs], width=0.36, color=FS.SEQ_BLUE[2], label=T("시선이 막히지 않음", "line of sight clear"))
    cx.bar(xx + 0.18, [u2[s]["observed_k_fraction"] for s in cs], width=0.36, color=FS.SEQ_BLUE[4], label=T("그 요청 depth 가 s 에 닿음 (10 mm)", "depth at s (10 mm)"))
    for xi, s in zip(xx, cs):
        cx.text(xi, 1.02, f"n={u2[s]['n_rows']}", ha="center", fontsize=6.5, color=FS.INK_2)
    cx.set_xticks(xx); cx.set_xticklabels([T(*S_LABEL[s]) for s in cs], rotation=30, ha="right", fontsize=7); cx.set_ylim(0, 1.4)
    cx.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cx.set_ylabel(T("행 비율", "fraction of rows"), fontsize=8); cx.legend(fontsize=7, frameon=False, loc="upper center", ncol=1)
    cx.set_title(T("U2: 요청 시점 카메라에서 보이는 면인가 (실패 6 장면)", "U2: visible from a camera at the request"), fontsize=8.5, loc="left")
    out["u2_visibility"] = {s: {k_: u2[s][k_] for k_ in ("visible_k_fraction", "observed_k_fraction", "n_rows")} for s in cs}
    fig.suptitle(T("IFG-UNKNOWN: 미상 활성 제약 행의 출처 · GT 거리 · 보임 (쥐기 전)", "IFG-UNKNOWN: source, GT distance and visibility of unknown active rows"),
                 fontsize=10.5, x=0.01, ha="left")
    return save(fig, a.fig / "ifg-unknown-sources.png", out, FS, plt)


def fig_table(a, N, FS, T, plt):
    u3 = N["U3_requests_dominant_unknown"]["rows"]
    ab = {"robot_self": "robot", "target_at_request_gt": "target", "neighbour_fruit": "neigh", "support_surface": "surf", "memory_stale": "memory",
          "field_artefact": "artefact", "other": "other", "ambiguous": "ambig"}
    cols = [T("장면", "scene"), "k", T("단계", "state"), T("활성 행", "active"), T("미상 행", "unknown")] + [ab[s] for s in SOURCES] + \
           [T("최다 침범 행\n출처", "worst row\nsource"), T("최다 침범\nlink", "worst\nlink"), T("수정 mm", "corr mm"), "HOLD", T("침범 mm", "pen mm"), T("출력 창 최악\n(R1)", "out worst\n(R1)")]
    body = []
    for x in u3:
        c = x["unknown_source_counts"]
        body.append([str(x["scene"]), str(x["k"]), x["state"], str(x["n_active_ref"]), str(x["n_unknown"])] + [str(c.get(s, 0)) for s in SOURCES] +
                    [ab.get(x["worst_row_source"], str(x["worst_row_source"])), str(x["worst_row_link"]), f"{x['correction_mm']:.1f}", "HOLD" if x["hold"] else "-",
                     f"{x['penetration_to_mm']:.1f}", str(x["out_worst_group"])])
    fig, ax = plt.subplots(figsize=(19, 0.32 * (len(body) + 3)))
    FS.style_axes(fig, ax); ax.axis("off")
    tb = ax.table(cellText=body, colLabels=cols, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(7.3); tb.scale(1, 1.25)
    for (ri, ci), cell in tb.get_celld().items():
        cell.set_edgecolor(FS.GRID_INK)
        if ri == 0:
            cell.set_facecolor(FS.SEQ_BLUE[1]); cell.set_text_props(fontweight="bold"); cell.set_height(cell.get_height() * 1.9)
    ax.set_title(T(f"U3: 최다 침범 활성 행이 미상인 요청 {len(body)}개 — 미상 행의 U1 출처 수, 수정량 (손끝, 실행 창 최대), HOLD (실패 6 장면, 쥐기 전)",
                   f"U3: {len(body)} requests whose most-violated active row is unknown: U1 source counts, correction, HOLD"), fontsize=8.5, loc="left")
    return save(fig, a.fig / "ifg-unknown-u3-table.png", {"columns": cols, "rows": body, "U3": u3}, FS, plt)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--run", type=pathlib.Path, required=True); c.add_argument("--precheck", type=pathlib.Path, required=True)
    c.add_argument("--eps", required=True); c.add_argument("--out", type=pathlib.Path, required=True)
    c.add_argument("--fig-requests", default="", help="ep:k,... whose supporting points are saved for the scene figure")
    c.add_argument("--moved-test", choices=("body_pose", "gt_distance"), default="body_pose")
    r = sub.add_parser("report")
    r.add_argument("--raw", type=pathlib.Path, required=True); r.add_argument("--precheck", type=pathlib.Path, required=True)
    r.add_argument("--fig", type=pathlib.Path, required=True); r.add_argument("--verify", type=pathlib.Path, required=True)
    r.add_argument("--raw-v1", type=pathlib.Path, default=None, help="raw dir of the v1 (gt_distance moved test) collection")
    a = ap.parse_args()
    if a.cmd == "collect":
        global MOVED_TEST
        MOVED_TEST = a.moved_test
        a.out.mkdir(parents=True, exist_ok=True)
        fk = {tuple(int(v) for v in x.split(":")) for x in a.fig_requests.split(",") if x}
        logf = open(a.out / f"collect_{a.eps.replace(',', '_')}.log", "a")

        def log(msg):
            print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
        for ep in [int(x) for x in a.eps.split(",")]:
            collect_scene(a.run, a.precheck, ep, a.out, log, fk)
    else:
        report(a)


if __name__ == "__main__":
    main()
