"""T43 FR offline (a) · (b) — T43 FC's fc_offline.py generalised to every finger_cover option in fr_options.OPTIONS
(old = axial chain; one "new" model per option). Same rows, same composition; the (b) attribution step is dropped.

Original FC docstring follows.

T43 FC offline (a) · (b) — what the per-finger body spheres (`--finger-cover`, overflow 2.5 mm) would have
seen in recorded runs (implementer, CPU only, no new runs).

usage: fc_offline.py OUT_JSON SNAPSHOT KEY [KEY ...]
       KEY = <ARM>/<STAGE>/<name> (analysis.json key);  SNAPSHOT = the run's snapshot root (manifest model_xml), used for the
       client reset (hurdle mocap pose) only.  One process per snapshot (hold_w reads T43W_SNAP at import).

Models (both built from the current tree's model_transport.xml, same robot as every snapshot):
  old = the run's constraint model (E3b sphere options from the server log; E0 has no server -> V5 options)
  new = old + finger_cover {hands both, overflow 2.5 mm}: each finger's chain replaced by body spheres.
Finger joints: from the row's MuJoCo qpos, the client's way (openings_from_mujoco -> FingerJointMap), as SafePolicy does.
Pose: robot FK in the robot base frame (= field frame 'base'); world = MuJoCo body 'base' pose at that qpos.

(a) CT finger-hurdle contact events (CT/raw/ct.json runs[key].contacts, kind hand, body ee_finger_*; MJCF names, URDF link
    via MJCF_BODY_ALIASES: MJCF ee_finger_l2 == URDF ee_finger_l1). For every row t in [first_t, last_t] and both its pre-state
    (control row t qpos) and post-state (row t+1 qpos):
      geo   = min over that link's spheres of (distance from the centre to the hurdle collision geoms (bar capsule, leg
              cylinders, mocap pose from the client reset) - r)   [true geometry]
      mesh  = mj_geomDistance(finger collision geoms, hurdle geoms) min (MuJoCo, the real finger)
      esdf  = min over the link's spheres of d - r, d = recorded field of the chunk covering t (latest planning t_step <= t)
              composed like the TO (target-free layer for authorized links when target_field_policy != relax, not
              attached, target present: FIA.contact_policy = to_adapter's rule). sees = esdf < 0.
              (The record's `clearance` array is a FULL-field diagnostic, not the TO's rows; fidelity is checked instead on
              the TO's recorded worst row to.metrics.max_violation_pair (step, query, clearance) with the old model.)
    Event value = min over its rows/states. Only arms with server records have esdf (E0 has none).
(b) normal grasp moments: server chunks with grasp.state_before or state 'closing', or grasp.attach true. Rows = finger spheres
    at q_now (planning-row qpos, the chunk's finger opening) and along the refined plan rows 0..7 (TO window) with the same
    opening (q_now: measured; plan: the server's envelope, SafePolicy._set_finger_joints). gap = d - r (TO composition).
    Counted: finger rows of the active hand with gap < esdf_margin (= violated rows), old vs new, and among them the rows the
    verdict would NOT allow (safe_policy._classify_esdf_rows: target-free answer -> collision; full layer -> allowed only if
    |d_row - d(target points)| <= voxel and tier fine) = HOLD-causing rows.
    Attribution of each NEW plan row with gap < margin + 10 mm (offline only, approximate): the true object whose signed distance
    from the sphere centre matches the row's field distance best (|d_obj - d_field| <= 10 mm, else 'unexplained'; the apple is
    not a candidate for a row answered by the target-free layer; the field sees the table top ~6-9 mm high = support band): table (world
    boxes), crate (13 boxes), apple / other fruits (convex hull of the collision mesh), hurdle (if present).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, pathlib, sys, time, functools
os.environ.setdefault("MUJOCO_GL", "osmesa")
SNAP = sys.argv[2] if len(sys.argv) > 2 else "${AG3S_PROJECT_ROOT}-t43w3"
os.environ["T43W_SNAP"] = SNAP
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import numpy as np, mujoco  # noqa: E402
import hold_w as HW  # noqa: E402
sys.path.insert(0, _ag3s_workspace_path('outputs/impl/T43FR'))
from fr_options import OPTIONS  # noqa: E402
from AG3S.trajopt.experiments import fixed_input_ablation as FIA  # noqa: E402
from AG3S.trajopt.types import ChunkLayout  # noqa: E402
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, MJCF_BODY_ALIASES  # noqa: E402
from AG3S.ag3s.experiments.reports.grounding_report import (  # noqa: E402
    GRIPPER_LINKS, FINGER_LINKS, build_constraint_robot_model)
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap, openings_from_mujoco  # noqa: E402

T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
ANALYSES = [T43 / "W/analysis/analysis.json", T43 / "W/analysis_blind/analysis.json",
            T43 / "Z3V/analysis/analysis.json", T43 / "V7g/analysis/analysis.json"]
CT = T43 / "CT/raw/ct.json"
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
E3B = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
OVERFLOW = 0.0025
FRUIT_R = 0.033
H = 8
URDF_OF_MJCF = {v: k for k, v in MJCF_BODY_ALIASES.items()}


class _Args:
    links = None; sphere_spacing = None; max_spheres_per_capsule = None; capsule_radius_scale = None
    max_sphere_radius = None; esdf_margin = None; target_field_policy = None; phase = None
    manipulators = None; model_xml = None; hand = None; allow_uncertified = False; rows_per_step = None


@functools.lru_cache(maxsize=4)
def models(sphere_key):
    opts = dict(sphere_key)
    m = mujoco.MjModel.from_xml_path(XML)
    scene = TransportScene.attach(m, mujoco.MjData(m))
    old = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts)
    news = {k: build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts, finger_cover=dict(v))
            for k, v in OPTIONS.items()}
    return old, news, FingerJointMap(old)


def all_runs():
    out = {}
    for f in ANALYSES:
        if f.exists():
            out.update(json.load(open(f))["runs"])
    return out


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-18), 0.0, 1.0)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def geom_point_dist(m, d, g, P):
    """signed distance (m) from points P (world) to geom g: capsule (3), cylinder (5), box (6), plane (0)."""
    R = d.geom_xmat[g].reshape(3, 3); c = d.geom_xpos[g]; s = m.geom_size[g]
    L = (P - c) @ R                                   # geom-local
    t = int(m.geom_type[g])
    if t == 3:
        return seg_dist(L, np.array([0, 0, -s[1]]), np.array([0, 0, s[1]])) - s[0]
    if t == 5:
        rho = np.linalg.norm(L[:, :2], axis=1) - s[0]; h = np.abs(L[:, 2]) - s[1]
        out = np.linalg.norm(np.maximum(np.stack([rho, h], 1), 0), axis=1)
        return out + np.minimum(np.maximum(rho, h), 0)
    if t == 6:
        q = np.abs(L) - s[:3]
        return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
    if t == 0:
        return L[:, 2]
    return np.full(P.shape[0], np.inf)


class Scene:
    """client reset (hurdle) + per-row MuJoCo state."""

    def __init__(self, cl, rd):
        man = json.load(open(rd / "manifest.json")); prof = man["scene"].get("obstacle_profile")
        ep = man["scene"]["episode_index"]; seed = int(np.load(rd / "traj.npz")["policy_seed"])
        start = rd.parent / f"{rd.name}.start"
        cfg = None
        if start.exists():
            cfg = next((l.split("=", 1)[1].split(":", 1)[0] for l in start.read_text().splitlines()
                        if l.startswith("obstacle=") and l != "obstacle=:"), None)
        L = HW.reset(cl, ep, seed, prof, cfg)
        self.m, self.d = L["m"], L["d"]; mg = L.get("obstacle_manager")
        self.hurdle = sorted(mg.collision_geom_ids) if mg is not None else []
        m = self.m
        self.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.qadr = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i]) for i in range(m.njnt)}
        crate = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")
        self.crate = [g for g in range(m.ngeom) if self._under(g, crate) and self._coll(g)]
        # static scene ('table'): collision geoms whose root body is not the robot, the crate, a fruit or the hurdle
        roots = {self.base, crate, *[b for b in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
                                                for n in ("apple", "banana", "orange", "pear")) if b >= 0]}
        hurdle_b = {int(m.geom_bodyid[h]) for h in self.hurdle}
        self.table = [g for g in range(m.ngeom) if self._coll(g) and int(m.geom_bodyid[g]) not in hurdle_b
                      and not any(self._under(g, r) for r in roots) and int(m.geom_type[g]) in (0, 6, 7)]
        self.fruit = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("apple", "banana", "orange", "pear")}
        self.fruit_geoms = {n: [g for g in range(m.ngeom) if b >= 0 and m.geom_bodyid[g] == b and self._coll(g)
                                and int(m.geom_type[g]) == 7] for n, b in self.fruit.items()}
        self.finger_geoms = {}
        for body in ("ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2"):
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
            self.finger_geoms[body] = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and self._coll(g)]

    def _coll(self, g):
        return self.m.geom_contype[g] != 0 or self.m.geom_conaffinity[g] != 0

    def _under(self, g, root):
        b = int(self.m.geom_bodyid[g])
        while b > 0:
            if b == root:
                return True
            b = int(self.m.body_parentid[b])
        return False

    def set(self, qpos):
        self.d.qpos[:] = qpos; mujoco.mj_kinematics(self.m, self.d)
        T = np.eye(4); T[:3, :3] = self.d.xmat[self.base].reshape(3, 3); T[:3, 3] = self.d.xpos[self.base]
        return T

    def site_pose(self, side):
        sid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_ee")
        T = np.eye(4); T[:3, :3] = self.d.site_xmat[sid].reshape(3, 3); T[:3, 3] = self.d.site_xpos[sid]
        return T

    def state(self, row, rz, t, which, fmap, side):
        """(T_world_base, M, finger params, q20). pre = control row t qpos (M = I). post = the replay post-state of row t
        (CT replay index t + 1: site pose tcp/tcp_R and finger opening): M = T_site(post) inv(T_site(pre)), the rigid hand
        motion over the row (base and torso do not move)."""
        qpos = np.asarray(row["qpos"], float)
        Tb = self.set(qpos)
        q20 = FIA.q_now_from_qpos(qpos, self.qadr)
        if which == "pre":
            params = fmap.joint_values(openings_from_mujoco(self.m, self.d, fmap.convention))
            return Tb, np.eye(4), params, q20
        i = int(np.flatnonzero(rz["t"] == t)[0]) + 1 if "t" in rz.files else t + 1
        if i >= rz["tcp"].shape[0]:
            return None
        Tp = np.eye(4); Tp[:3, :3] = rz["tcp_R"][i]; Tp[:3, 3] = rz["tcp"][i]
        M = Tp @ np.linalg.inv(self.site_pose(side))
        params = fmap.joint_values({side: float(rz["finger_open_norm"][i])})
        return Tb, M, params, q20

    def mesh_hurdle(self, body, M=None, params=None):
        """mj_geomDistance(finger collision geoms, hurdle), with the hand moved by M (done by moving the hurdle mocap
        bodies by inv(M)) and the finger joints at `params`."""
        m, d = self.m, self.d
        saved_q = d.qpos.copy(); saved_mp = d.mocap_pos.copy(); saved_mq = d.mocap_quat.copy()
        if params:
            for j, v in params.items():
                jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                if jid >= 0:
                    d.qpos[m.jnt_qposadr[jid]] = v
        if M is not None:
            Mi = np.linalg.inv(M)
            for h in self.hurdle:
                mid = int(m.body_mocapid[m.geom_bodyid[h]])
                if mid < 0:
                    continue
                R = np.zeros(9); mujoco.mju_quat2Mat(R, saved_mq[mid]); R = R.reshape(3, 3)
                d.mocap_pos[mid] = Mi[:3, :3] @ saved_mp[mid] + Mi[:3, 3]
                q = np.zeros(4); mujoco.mju_mat2Quat(q, (Mi[:3, :3] @ R).reshape(-1)); d.mocap_quat[mid] = q
        mujoco.mj_kinematics(m, d)
        best = np.inf; ft = np.zeros(6)
        for g in self.finger_geoms[body]:
            for h in self.hurdle:
                best = min(best, mujoco.mj_geomDistance(m, d, g, h, 0.2, ft))
        d.qpos[:] = saved_q; d.mocap_pos[:] = saved_mp; d.mocap_quat[:] = saved_mq
        mujoco.mj_kinematics(m, d)
        return best

    def explain(self, P, d_field, exclude=(), tol=0.010):
        """per point: the true object whose signed distance from the point best matches the field distance d_field
        (|d_obj - d_field| minimal, like the verdict's 'distance' evidence); 'unexplained' if none is within tol."""
        cand = {"table": [self.gdist(g, P) for g in self.table],
                "crate": [self.gdist(g, P) for g in self.crate],
                "hurdle": [geom_point_dist(self.m, self.d, g, P) for g in self.hurdle]}
        for n, b in self.fruit.items():
            if b >= 0:
                cand[n] = [self.hull_dist(g, P) for g in self.fruit_geoms[n]] or \
                    [np.linalg.norm(P - self.d.xpos[b], axis=1) - FRUIT_R]
        lab = [k for k, v in cand.items() if v and k not in exclude]
        D = np.stack([np.min(np.stack(cand[k]), 0) for k in lab], 1)
        E = np.abs(D - np.asarray(d_field)[:, None])
        j = E.argmin(1)
        if os.environ.get("FC_DEBUG"):
            for r in range(min(3, len(j))):
                print("  explain d_field %.1f mm |" % (1e3 * d_field[r]), {lab[i]: round(1e3 * D[r, i], 1) for i in range(len(lab))},
                      "P", np.round(P[r], 3))
        return ([lab[i] if E[r, i] <= tol else "unexplained" for r, i in enumerate(j)],
                D[np.arange(len(j)), j] - np.asarray(d_field))

    def gdist(self, g, P):
        return self.hull_dist(g, P) if int(self.m.geom_type[g]) == 7 else geom_point_dist(self.m, self.d, g, P)

    def hull_dist(self, g, P):
        """signed distance to the convex hull of mesh geom g (MuJoCo collides its hull): exact inside, face-plane
        distance outside (exact in front of a face, a lower bound near edges/corners)."""
        from scipy.spatial import ConvexHull
        m, d = self.m, self.d
        mid = int(m.geom_dataid[g]); v0, nv = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        V = np.asarray(m.mesh_vert[v0:v0 + nv], float).reshape(-1, 3)
        R = d.geom_xmat[g].reshape(3, 3); W = V @ R.T + d.geom_xpos[g]
        eq = ConvexHull(W).equations
        return np.max(P @ eq[:, :3].T + eq[:, 3], axis=1)

    def nearest(self, P):
        """per point: (label, signed distance m) of the nearest true object."""
        cand = {"table": [geom_point_dist(self.m, self.d, g, P) for g in self.table],
                "crate": [geom_point_dist(self.m, self.d, g, P) for g in self.crate],
                "hurdle": [geom_point_dist(self.m, self.d, g, P) for g in self.hurdle]}
        for n, b in self.fruit.items():
            if b >= 0:
                cand[n] = [np.linalg.norm(P - self.d.xpos[b], axis=1) - FRUIT_R]
        lab = [k for k, v in cand.items() if v]
        D = np.stack([np.min(np.stack(cand[k]), 0) for k in lab], 1)
        j = D.argmin(1)
        return [lab[i] for i in j], D[np.arange(len(j)), j]




def main():
    out_path, keys = sys.argv[1], sys.argv[3:]
    runs = all_runs(); ct = json.load(open(CT))["runs"]
    cl = HW.load_client()
    res = {"definitions": __doc__, "snapshot": SNAP, "options": OPTIONS, "runs": {}}
    for key in keys:
        t0 = time.time()
        a = runs[key]; rd = pathlib.Path(a["run_dir"])
        frames = [json.loads(l) for l in open(rd / "frames.jsonl") if l.strip()]
        ctrl = {r["t_step"]: r for r in frames if r.get("kind") == "control"}
        plan = {r["seq"]: r for r in frames if r.get("kind") == "planning"}
        seg = a.get("server_segment")
        st = None
        if seg:
            first = pathlib.Path(seg["first"]); i0 = int(first.stem.split("_")[1])
            st = FIA.resolve_settings(_Args, FIA.parse_server_log(str(first.parent.parent.parent / "serve_safe.log")),
                                      json.load(open(rd / "manifest.json")))
            sph = tuple(sorted(st.sphere_options.items()))
        else:
            sph = tuple(sorted(E3B.items()))
        old, news, fmap = models(sph)
        allm = {"old": old, **news}
        sc = Scene(cl, rd)
        layout = ChunkLayout.rby1(old.joint_names)
        qadr = {mujoco.mj_id2name(sc.m, mujoco.mjtObj.mjOBJ_JOINT, i): int(sc.m.jnt_qposadr[i]) for i in range(sc.m.njnt)}
        outside = None
        if st is not None:
            _, _, _, maxd, unk = FIA.contact_policy(st, old.sphere_link_names, has_target=False, has_free=False, attached=False)
            outside = maxd if unk == "free" else -maxd
        # chunk index by planning t_step
        chunks = []; seq_of = {}
        if seg:
            for n in range(int(seg["n"])):
                p = first.parent / f"chunk_{i0 + n:05d}.npz"
                seq_ = n + 1                    # analysis pairing: the n-th chunk of the segment = planning seq n + 1
                if seq_ in plan:
                    chunks.append((int(plan[seq_]["t_step"]), p))
                    seq_of[str(p)] = seq_
        cache = {}

        def chunk_field(path):
            """(z, summary, field, masks, Q, row_params, fidelity_mm, layer by link) of one server chunk. Q = [q_now, refined
            plan rows 0..H-1]; finger openings = measured at q_now, the server's envelope along the plan."""
            if path not in cache:
                z = np.load(path, allow_pickle=True)
                s = json.loads(str(z["summary_json"]))
                field = FIA.build_field(z, outside)
                seq = seq_of[str(path)]
                qpos = np.asarray(plan[seq]["qpos"], float)
                fj = s.get("finger_joints") or {}
                params = dict(fj.get("joints") or {}) if isinstance(fj, dict) else {}
                if not params:
                    params = params_of(qpos)
                meas = (fj.get("opening_norm") or {}) if isinstance(fj, dict) else {}
                refr = np.asarray(z["reference_chunk"], float)
                P_, _ = fmap.path({h: v for h, v in meas.items() if v is not None},
                                  {"left": refr[:, 7], "right": refr[:, 15]}, H, default=old.param_vector())
                plan_params = [dict(zip(fmap.param_joint_names, map(float, P_[:, k]))) for k in range(H)]
                qn = FIA.q_now_from_qpos(qpos, qadr)
                refc = np.asarray(z["refined_chunk"], float)
                Q = np.vstack([qn[None], layout.full_q(layout.chunk_to_trajectory(refc)[:, :H], qn).T])
                row_params = [params] + plan_params
                # TO composition (what the optimizer and the verdict use): FIA.contact_policy = to_adapter's rule — authorized
                # links query the target-free layer while a target is present and nothing is attached (exclude_authorized).
                # NOTE the record's `clearance` array (8 x n) is a full-field diagnostic, NOT the TO's rows.
                tp = z["target_points"] if "target_points" in z.files else None
                attached = int((s.get("esdf_stats") or {}).get("n_attached_points", 0) or 0) > 0
                has_t = bool(s.get("has_target")) and tp is not None
                masks = {}
                for nm, rob in allm.items():
                    _, mk, _, _, _ = FIA.contact_policy(st, rob.sphere_link_names, has_target=has_t,
                                                       has_free=bool(field.target_free_layers), attached=attached)
                    masks[nm] = np.zeros(rob.n_spheres, bool) if mk is None else np.asarray(mk, bool)
                use_free = {n: bool(masks["old"][list(old.sphere_link_names).index(n)]) for n in set(old.sphere_link_names)}
                # fidelity: the TO's recorded worst row (step, query) rebuilt with the old model and this composition
                fid = float("nan")
                mv = ((s.get("to") or {}).get("metrics") or {}).get("max_violation_pair")
                try:
                    import ast
                    mv = ast.literal_eval(mv) if isinstance(mv, str) else mv
                except Exception:
                    mv = None
                if mv and mv.get("block") == "esdf" and mv.get("query") is not None and int(mv["query"]) < old.n_spheres:
                    k, qi = int(mv["step"]), int(mv["query"])
                    old.set_joint_parameters(plan_params[k])
                    C, _ = old.sphere_centers_numeric(Q[1 + k]); c = np.asarray(C)[qi:qi + 1]
                    dd = (field.target_free_distance(c) if masks["old"][qi] else field.distance(c))
                    fid = abs(float(np.asarray(dd)[0]) - old.radii[qi] - float(st.esdf_margin) - float(mv["clearance_m"]))
                cache.clear(); cache[path] = (z, s, field, masks, Q, row_params, 1e3 * fid, use_free)
            return cache[path]

        def gaps(rob, mask, field, Q, params, links):
            """params: one dict for every row, or a list (one per row of Q)."""
            plist = params if isinstance(params, list) else [params] * len(Q)
            names = np.asarray(rob.sphere_link_names); sel = np.isin(names, links)
            r = rob.radii[sel]
            out = []
            for q, pr in zip(Q, plist):
                rob.set_joint_parameters(pr)
                C, _ = rob.sphere_centers_numeric(q)
                P = np.asarray(C)[sel]
                d = np.asarray(field.distance(P), float)
                if mask is not None and np.asarray(mask)[sel].any():
                    df = np.asarray(field.target_free_distance(P), float)
                    d = np.where(np.asarray(mask)[sel], df, d)
                out.append(d - r)
            return np.asarray(out), sel

        def params_of(qpos):
            sc.set(qpos)
            return fmap.joint_values(openings_from_mujoco(sc.m, sc.d, fmap.convention))

        rec = {"arm": a["arm"], "stage": a["stage"], "has_server": bool(seg),
               "esdf_margin_m": None if st is None else st.esdf_margin,
               "target_field_policy": None if st is None else st.target_field_policy,
               "n_spheres": {nm: rob.n_spheres for nm, rob in allm.items()}, "contacts": [], "grasp_chunks": []}
        side = "right" if int(json.load(open(rd / "manifest.json"))["scene"]["episode_index"]) == 1834 else "left"
        rzp = T43 / "CT/replay" / (key.replace("/", "_") + ".npz")
        rz = np.load(rzp) if rzp.exists() else None
        # ---------------------------------------------------------------- (a)
        for c in ((ct.get(key) or {}).get("contacts", []) if rz is not None else []):
            if c.get("kind") != "hand" or not str(c.get("body", "")).startswith("ee_finger"):
                continue
            body = c["body"]; link = URDF_OF_MJCF.get(body, body)
            ev = {"mjcf_body": body, "urdf_link": link, "phase": c["phase"], "first_t": c["first_t"],
                  "last_t": c["last_t"], "max_pen_mm": c["max_pen_mm"], "rows": []}
            for t in range(int(c["first_t"]), int(c["last_t"]) + 1):
                if t not in ctrl:
                    continue
                for which in ("pre", "post"):
                    st_ = sc.state(ctrl[t], rz, t, which, fmap, side)
                    if st_ is None:
                        continue
                    Tb, M, params, q20 = st_
                    row = {"t": t, "state": which, "mesh_mm": 1e3 * sc.mesh_hurdle(body, M, params)}
                    if which == "post" and (t + 1) in ctrl:      # fidelity: replay post-state vs next row qpos
                        Tn = sc.set(np.asarray(ctrl[t + 1]["qpos"], float))
                        Sn = sc.site_pose(side)
                        sc.set(np.asarray(ctrl[t]["qpos"], float))
                        row["post_fidelity_mm"] = 1e3 * float(np.abs(Sn[:3, 3] - (M @ sc.site_pose(side))[:3, 3]).max())
                    Wfix = {}
                    for nm, rob in allm.items():
                        rob.set_joint_parameters(params)
                        C, R = rob.sphere_centers_numeric(q20)
                        sel = np.asarray(rob.sphere_link_names) == link
                        W = (np.asarray(C)[sel] @ Tb[:3, :3].T + Tb[:3, 3]) @ M[:3, :3].T + M[:3, 3]
                        g = np.min([geom_point_dist(sc.m, sc.d, h, W) for h in sc.hurdle], 0) - np.asarray(R)[sel]
                        row[f"geo_{nm}_mm"] = 1e3 * float(g.min())
                        Wfix[nm] = (W - Tb[:3, 3]) @ Tb[:3, :3], np.asarray(R)[sel], sel
                    if chunks:
                        cands = [p for ts, p in chunks if ts <= t]
                        if cands:
                            z, s_, field, masks, _, _, fid_, use_free_ = chunk_field(cands[-1])
                            row["chunk_fidelity_mm"] = round(fid_, 3)
                            row["chunk_seq"] = seq_of[str(cands[-1])]
                            for nm in allm:
                                Pb, R, sel = Wfix[nm]
                                dd = np.asarray(field.distance(Pb), float)
                                mk = masks[nm]
                                if mk is not None and np.asarray(mk)[sel].any():
                                    dd = np.where(np.asarray(mk)[sel], np.asarray(field.target_free_distance(Pb), float), dd)
                                row[f"esdf_{nm}_mm"] = 1e3 * float((dd - R).min())
                                row[f"mask_{nm}"] = None if mk is None else bool(np.asarray(mk)[sel].any())
                    ev["rows"].append(row)
            for k in ["mesh_mm"] + [f"{p}_{nm}_mm" for nm in allm for p in ("geo", "esdf")]:
                v = [r[k] for r in ev["rows"] if k in r]
                ev[k + "_min"] = None if not v else round(min(v), 2)
            rec["contacts"].append(ev)
            print(key, "contact", body, link, c["phase"], c["first_t"], "pen %.2f" % c["max_pen_mm"], "mesh", ev["mesh_mm_min"],
                  "geo old/fc25", ev["geo_old_mm_min"], ev["geo_fc25_mm_min"], "esdf old/fc25", ev["esdf_old_mm_min"], ev["esdf_fc25_mm_min"], flush=True)
        # ---------------------------------------------------------------- (b)
        hand_fingers = [l for l in FINGER_LINKS if l[10] == side[0]]
        for ts, p in chunks:
            with np.load(p, allow_pickle=True) as zz:
                s = json.loads(str(zz["summary_json"]))
            g = s.get("grasp") or {}
            if not (g.get("attach") or "closing" in (g.get("state"), g.get("state_before"))):
                continue
            seq = seq_of[str(p)]
            z, s, field, masks, Q, row_params, fid_, use_free_ = chunk_field(p)
            params = row_params[0]
            Tb = sc.set(np.asarray(plan[seq]["qpos"], float))
            margin = float(st.esdf_margin)
            verdict = s.get("verdict") or {}
            cls = verdict.get("classification") or {}
            row = {"t": ts, "state": g.get("state"), "state_before": g.get("state_before"), "attach": bool(g.get("attach")),
                   "server_n_collision": cls.get("n_collision"), "server_n_allowed": cls.get("n_allowed"),
                   "server_collision_by_why": cls.get("collision_by_why"),
                   "held_by_server": any(x.get("action") == "hold" for x in (verdict.get("reasons") or [])),
                   "finger_params": {k: round(float(v), 4) for k, v in params.items()}}
            for nm, rob in allm.items():
                gp, sel = gaps(rob, masks[nm], field, Q, row_params, hand_fingers)
                row[f"{nm}_min_gap_now_mm"] = round(1e3 * float(gp[0].min()), 2)
                row[f"{nm}_min_gap_plan_mm"] = round(1e3 * float(gp[1:].min()), 2)
                row[f"{nm}_viol_rows_now"] = int((gp[0] < margin).sum())
                row[f"{nm}_viol_rows_plan"] = int((gp[1:] < margin).sum())
                # label-dependent margins (obstacle 10 mm on robot rows after attach, destination 20 mm) are not recorded
                # (no label layer): bracket with the plan rows below 10 and 20 mm.
                row[f"{nm}_rows_plan_lt10"] = int((gp[1:] < margin + 0.010).sum())
                row[f"{nm}_rows_plan_lt20"] = int((gp[1:] < margin + 0.020).sum())
                row[f"{nm}_n_spheres"] = int(sel.sum())
                row[f"{nm}_mask"] = None if masks[nm] is None else bool(np.asarray(masks[nm])[sel].any())
                if nm == "old":
                    row["old_fidelity_mm"] = round(fid_, 3)
                    row["finger_layer"] = {l: ("target_free" if use_free_[l] else "full") for l in hand_fingers}
                # server classification of the violated finger rows (safe_policy._classify_esdf_rows, 'distance' evidence):
                # allowed iff the row used the full layer, |d_row - d(target points)| <= voxel and the answering tier is fine.
                n_coll = [0, 0]   # (now row, plan rows)
                tp = z["target_points"] if "target_points" in z.files else None
                if (gp < margin).any():
                    from scipy.spatial import cKDTree
                    tree = cKDTree(np.asarray(tp, float)) if tp is not None and len(tp) else None
                    voxel = float(getattr(getattr(field, "grid", None), "voxel_size", 0.0) or 0.0)
                    flags = np.asarray(masks[nm], bool)[sel]
                    for k in range(Q.shape[0]):
                        bad = np.flatnonzero(gp[k] < margin)
                        if not bad.size:
                            continue
                        rob.set_joint_parameters(row_params[k])
                        C, _ = rob.sphere_centers_numeric(Q[k])
                        Pb = np.asarray(C)[sel][bad]
                        fl = flags[bad]
                        tiers = np.asarray(field.answer_tier(Pb, target_free=fl)).reshape(-1)
                        d_row = gp[k][bad] + rob.radii[sel][bad]
                        ok = ~fl & np.isin(tiers.astype(str), ["fine", "None"])
                        if tree is not None and voxel > 0:
                            ok &= np.abs(d_row - tree.query(Pb)[0]) <= voxel
                        else:
                            ok &= False
                        n_coll[0 if k == 0 else 1] += int((~ok).sum())
                row[f"{nm}_nonallowed_rows_now"], row[f"{nm}_nonallowed_rows_plan"] = n_coll
                if False:  # T43 FR: attribution dropped (FC has it for the default)
                    labs = {}
                    for k in range(1, Q.shape[0]):     # plan rows below margin + 10 mm (the bracket's lower step)
                        bad = np.flatnonzero(gp[k] < margin + 0.010)
                        if not bad.size:
                            continue
                        rob.set_joint_parameters(row_params[k])
                        C, _ = rob.sphere_centers_numeric(Q[k])
                        P = np.asarray(C)[sel][bad] @ Tb[:3, :3].T + Tb[:3, 3]
                        d_f = gp[k][bad] + rob.radii[sel][bad]
                        tf = np.asarray(masks[nm], bool)[sel][bad]
                        for flag in (False, True):          # a target-free answer cannot come from the target (apple)
                            sub = tf == flag
                            if not sub.any():
                                continue
                            lab, resid = sc.explain(P[sub], d_f[sub], exclude=("apple",) if flag else ())
                            for l_, g_, r_ in zip(lab, gp[k][bad][sub], resid):
                                e = labs.setdefault(l_, [0, 0, np.inf, []]); e[0] += 1; e[1] += int(g_ < margin)
                                e[2] = min(e[2], 1e3 * float(g_)); e[3].append(1e3 * float(r_))
                    row[f"{nm}_viol_nearest"] = {k: {"rows_lt10": v[0], "rows_lt_margin": v[1], "min_gap_mm": round(v[2], 2),
                                                     "median_dobj_minus_dfield_mm": round(float(np.median(v[3])), 2)}
                                               for k, v in labs.items()}
            rec["grasp_chunks"].append(row)
            print(key, "grasp", ts, row["state_before"], "->", row["state"], "att" if row["attach"] else "", "hold" if row["held_by_server"] else "",
                  "plan min old/fc25 %.1f/%.1f" % (row["old_min_gap_plan_mm"], row["fc25_min_gap_plan_mm"]),
                  "non-allowed plan", {nm: row[f"{nm}_nonallowed_rows_plan"] for nm in allm}, flush=True)
        rec["sec"] = round(time.time() - t0, 1)
        res["runs"][key] = rec
        json.dump(res, open(out_path, "w"), indent=1, default=float)
    json.dump(res, open(out_path, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
