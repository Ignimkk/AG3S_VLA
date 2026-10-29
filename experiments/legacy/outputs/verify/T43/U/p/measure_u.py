"""T43 U — user hypothesis test from records only (verifier, CPU, no server, no GPU). 2026-10-05.

Hypothesis (user): after the manipulated identity switches away from the fruit (e.g. onto the hurdle bar), the fruit is treated
as an obstacle at pick time; the closing fingers collide slightly with it -> miss (lift empty) or grasp-then-slip.

Inputs (read-only): raw/runs_u.json (runs_u.py), client frames.jsonl / manifest / .start, server chunk npz + serve_safe.log.
GT geometry: MuJoCo model_transport_pick_place_obstacles.xml of the ${AG3S_PROJECT_ROOT}-t43v snapshot + a probe sphere (r 1 mm,
contype 0) added by MjSpec; obstacles placed by that snapshot's PickPlaceObstacleManager(config).activate(profile)
(the T43V hurdle_metrics path). Runs without obstacle: no obstacle geoms.
TO rows: rebuilt with AG3S.trajopt.experiments.fixed_input_ablation helpers from the main tree (settings from the run's
serve_safe.log, robot sphere model 'gripper' links, 122 spheres), CollisionLinearizer.esdf_clearance on the recorded refined
chunk, scene = make_scene(field incl. target-free layer, target_points, contact-policy margin, target-free mask)
+ obstacle_margin (server log) + support plane z = esdf_stats.target_free.ball.support_plane_z_m (else 0.82).
Fidelity per chunk: sphere centres step 0 vs npz sphere_centres; plain clearance vs npz clearance; reconstructed worst_row vs
recorded to.metrics.clearance_m (waypoint rows only; T43 R gripper-cover rows and T43 Q path rows are not rebuilt).

Definitions
  fruit           = manifest episode_target_fruit; arm = episode_used_arm; hand rows = spheres of ee_finger_<s>1/2 + ee_<arm>
                    (the authorized links of that hand), 61 spheres x 8 planned steps = 488 rows.
  close_t         = first control row with applied gripper[arm] < 0.9 after a row >= 0.9; close attempts = number of such
                    transitions.
  lifted          = fruit z >= z0 + 50 mm at some row (z0 = first planning row).
  in_crate_final  = last row fruit centre inside the crate floor box (|dx| <= 0.09, |dy| <= 0.15, z < crate z + 0.08).
  outcome         = no_close (never closed) | miss (closed, never lifted) | slip (lifted, final not in crate) | success
                    (final in crate). slip_t = first row after lift_t with fruit z < z0 + 20 mm; slip_kind = 'at_pick' if
                    slip_t - lift_t <= 64 rows else 'carry'.
  contact window  = control rows [close_t - 24, close_t + 24] of each close attempt: finger (ee_finger_<s>1/2 collision geoms) /
                    palm (EE_BODY_<S>) to fruit collision-geom distance (mj_geomDistance, cap 50 mm) and MuJoCo contacts at the
                    recorded qpos; fruit displacement = |fruit pos - fruit pos at window start| (mm).
  chunk timeline  = every chunk: grasp.state / attach, manipulated id / state / switched_from / centroid, target_centroid,
                    exclusion, per-camera 16x16 attention argmax (row, col, value), global peak point
                    attention_cloud[attention_peak_index]; each 3D point -> world (base pose at the planning qpos) -> nearest GT
                    surface among fruit / other fruits / crate / table / obstacle (probe distance).
  switch          = a chunk whose manipulated.id differs from the first chunk's manipulated id (or switched_from != None);
                    'before attach' = before the first chunk with grasp.attach true (all chunks if never attached).
  carve check     = fruit collision-mesh vertices above the fruit centre (world, planning qpos) -> field.target_free_distance
                    (what the hand rows read) and field.distance (main composite); carved := median target-free distance
                    of those vertices >= 5 mm; also ball centre / radius of the target-free layer and the fruit centre's
                    distance to that ball centre.
  hand-row facts  = per chunk in grasp states latched / closing (not attached): reconstructed clearance d - r - margin of
                    the 488 hand rows; per row GT nearest surface (sphere-surface distance, probe) and the code class of
                    _obstacle_rows (target / support / obstacle); rows with GT nearest = fruit and class = obstacle get the
                    obstacle margin when it is on (a).
  deflection      = per chunk, planned steps k = 0..7: fingertip midpoint (per finger, collision-mesh vertex farthest along
                    the approach axis) with the arm columns of reference_chunk[k] vs refined_chunk[k] substituted into the
                    planning qpos; max |tip_ref - tip_refined| (mm) and, at that k, radial = |tip_refined - fruit| -
                    |tip_ref - fruit| (mm, > 0 = refined further from the fruit centre).
usage: measure_u.py N_WORKERS [KEY_SUBSTRING]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, math, ast, pathlib, dataclasses, functools, re
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from multiprocessing import Pool

U = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/U'))
SNAP = pathlib.Path(os.environ.get("U_SNAP", "${AG3S_PROJECT_ROOT}-t43v") + "/pi05_TO_hybrid")  # T42-2 runs: U_SNAP=${AG3S_PROJECT_ROOT}-t42c (crate_post slots)
XML = str(SNAP / "rby1_description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml")
QADR = {"apple": 38, "banana": 45, "orange": 52, "pear": 59, "crate": 31}
FRUITS = ("apple", "banana", "orange", "pear")
ARM_COLS = {"left": list(range(0, 7)), "right": list(range(8, 15))}
GRIP_COL = {"left": 7, "right": 15}
CAP = 0.05
_G = {}


def jl(o):
    if isinstance(o, dict): return {str(k): jl(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [jl(v) for v in o]
    if isinstance(o, np.ndarray): return jl(o.tolist())
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, float) and not math.isfinite(o): return None
    return o


def lit(x):
    if isinstance(x, str):
        try: return ast.literal_eval(x)
        except Exception: return x
    return x


# ---------------------------------------------------------------- GT model -----------------------------------------------
def gt():
    if "m" in _G:
        return _G
    import mujoco
    sys.path.insert(0, str(SNAP / "rby1_manipulation" / "src"))
    from rby1_manipulation.simulation import pick_place_obstacles as PPO
    spec = mujoco.MjSpec.from_file(XML)
    b = spec.worldbody.add_body(name="u_probe", mocap=True, pos=[5.0, 5.0, 5.0])
    b.add_geom(name="u_probe_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.001, 0, 0], contype=0, conaffinity=0)
    m = spec.compile(); d = mujoco.MjData(m)
    B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    coll = lambda bn: [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == B(bn) and (m.geom_contype[g] or m.geom_conaffinity[g])]
    J = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i): int(m.jnt_qposadr[i]) for i in range(m.njnt)}
    pb = B("u_probe")
    _G.update(mj=mujoco, PPO=PPO, m=m, d=d, B=B, coll=coll, J=J, probe_mocap=int(m.body_mocapid[pb]),
              probe_geom=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "u_probe_geom"),
              table=[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")], crate=coll("crate"),
              fruit={f: coll(f) for f in FRUITS}, mg={},
              hands=sum((coll(b) for b in ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2", "EE_BODY_R", "ee_finger_r1", "ee_finger_r2")), []))
    return _G


def place_obstacle(ob):
    """activate the run's obstacle profile; returns its collision geom ids (empty for None)."""
    G = gt()
    key = None if ob is None else tuple(ob)
    prev = G.get("active")
    if prev is not None and prev != key:
        G["mg"][prev].park_all()   # restore slot sizes before another manager / profile touches the model
    if key is None:
        if "park" not in G:
            G["park"] = G["PPO"].PickPlaceObstacleManager(G["m"], G["d"], G["PPO"].load_pick_place_obstacle_config())
        G["park"].park_all()
        G["active"] = None
        return []
    if key not in G["mg"]:
        cfg = G["PPO"].load_pick_place_obstacle_config(key[0])
        G["mg"][key] = G["PPO"].PickPlaceObstacleManager(G["m"], G["d"], cfg)
    mg = G["mg"][key]
    mg.activate(key[1])
    G["active"] = key
    return sorted(int(g) for g in mg.collision_geom_ids)


def set_q(q):
    G = gt(); m, d, mj = G["m"], G["d"], G["mj"]
    d.qpos[:] = q; d.qvel[:] = 0
    mj.mj_kinematics(m, d)


def probe_dist(p, geoms):
    """distance (m) from point p to the union of geoms (probe sphere r 1 mm, + 1 mm); cap CAP. Assumes kinematics done."""
    G = gt(); m, d, mj = G["m"], G["d"], G["mj"]
    if not len(geoms):
        return CAP
    d.mocap_pos[G["probe_mocap"]] = p
    pg = G["probe_geom"]
    d.geom_xpos[pg] = p  # kinematics of the mocap body without recomputing everything
    best = CAP; ft = np.zeros(6)
    for g in geoms:
        if np.linalg.norm(d.geom_xpos[g] - p) - m.geom_rbound[g] > best:
            continue
        v = mj.mj_geomDistance(m, d, pg, int(g), best, ft) + 0.001
        best = min(best, v)
    return float(best)


def surfaces(fruit, obst, hands=False):
    G = gt()
    s = {"fruit": G["fruit"][fruit], "other_fruit": sum((G["fruit"][f] for f in FRUITS if f != fruit), []),
         "crate": G["crate"], "table": G["table"]}
    if obst:
        s["obstacle"] = obst
    if hands:
        s["hand"] = G["hands"]
    return s


def nearest_label(p, surf):
    dd = {k: probe_dist(p, g) for k, g in surf.items()}
    k = min(dd, key=dd.get)
    return (k if dd[k] < CAP else "none"), dd


def base_T(q):
    G = gt(); set_q(q); b = G["B"]("base")
    return G["d"].xpos[b].copy(), G["d"].xmat[b].reshape(3, 3).copy()


def tips_at(q, arm):
    """finger mid, fingertip mid (mesh vertex farthest along approach) at qpos q (kinematics done inside)."""
    G = gt(); m, d = G["m"], G["d"]; s = "L" if arm == "left" else "R"; sl = s.lower()
    set_q(q)
    ee = G["B"](f"EE_BODY_{s}"); f1 = G["B"](f"ee_finger_{sl}1"); f2 = G["B"](f"ee_finger_{sl}2")
    mid = 0.5 * (d.xpos[f1] + d.xpos[f2]); a = mid - d.xpos[ee]; a = a / (np.linalg.norm(a) + 1e-12)
    tips = []
    for bn in (f"ee_finger_{sl}1", f"ee_finger_{sl}2"):
        V = []
        for g in G["coll"](bn):
            did = int(m.geom_dataid[g])
            if did < 0:
                V.append(d.geom_xpos[g][None]); continue
            va, nv = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
            V.append(d.geom_xpos[g] + m.mesh_vert[va:va + nv].astype(float) @ d.geom_xmat[g].reshape(3, 3).T)
        V = np.concatenate(V); tips.append(V[np.argmax(V @ a)])
    return mid.copy(), 0.5 * (tips[0] + tips[1])


def q_with_row(q, row, arm):
    G = gt(); q = np.array(q, float).copy()
    for i, col in enumerate(ARM_COLS[arm]):
        q[G["J"][f"{arm}_arm_{i}"]] = row[col]
    return q


def hand_fruit(q, arm, fruit):
    """finger / palm to fruit distance (mm, cap 50) and contacts at q."""
    G = gt(); m, d, mj = G["m"], G["d"], G["mj"]; s = "L" if arm == "left" else "R"; sl = s.lower()
    d.qpos[:] = q; d.qvel[:] = 0
    mj.mj_kinematics(m, d); mj.mj_comPos(m, d); mj.mj_collision(m, d)
    fg = set(G["fruit"][fruit]); ft = np.zeros(6)
    out = {}
    for nm, bn in (("finger1", f"ee_finger_{sl}1"), ("finger2", f"ee_finger_{sl}2"), ("palm", f"EE_BODY_{s}")):
        gs = G["coll"](bn)
        out[nm + "_mm"] = 1e3 * min(mj.mj_geomDistance(m, d, a, b, CAP, ft) for a in fg for b in gs)
        out[nm + "_contacts"] = sum(1 for i in range(d.ncon) if (int(d.contact[i].geom1) in fg and int(d.contact[i].geom2) in gs)
                                    or (int(d.contact[i].geom2) in fg and int(d.contact[i].geom1) in gs))
    return out


# ---------------------------------------------------------------- TO rebuild ---------------------------------------------
class _Args:
    links = None; sphere_spacing = None; max_spheres_per_capsule = None; capsule_radius_scale = None
    max_sphere_radius = None; esdf_margin = None; target_field_policy = None; phase = None
    manipulators = None; model_xml = None; hand = None; allow_uncertified = False; rows_per_step = None


@functools.lru_cache(maxsize=8)
def to_setup(server_log, manifest_path):
    from AG3S.trajopt.experiments import fixed_input_ablation as FIA
    from AG3S.trajopt.types import ChunkLayout
    from AG3S.trajopt.linearize import CollisionLinearizer
    from AG3S.ag3s.robot_models.gripper_state import FingerJointMap
    man = json.loads(pathlib.Path(manifest_path).read_text())
    man["scene"]["model_xml"] = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
    st = FIA.resolve_settings(_Args, FIA.parse_server_log(server_log), man)
    robot, qadr, _ = FIA.build_robot(st)
    layout = ChunkLayout.rby1(robot.joint_names)
    lin = CollisionLinearizer(robot, layout, 8)
    return dict(FIA=FIA, st=st, robot=robot, qadr=qadr, layout=layout, lin=lin, fmap=FingerJointMap(robot),
                names=[str(n) for n in robot.sphere_link_names])


def hand_links(arm):
    sl = "l" if arm == "left" else "r"
    return {f"ee_finger_{sl}1", f"ee_finger_{sl}2", f"ee_{arm}"}


# ---------------------------------------------------------------- per run ------------------------------------------------
def outcome(ctrl, plan0, arm, fruit, obst):
    t = np.array([c["t_step"] for c in ctrl]); Q = np.array([c["qpos"] for c in ctrl], float)
    g = np.array([(c.get("applied_ctrl") or {}).get("gripper", [1.0, 1.0])[0 if arm == "left" else 1] for c in ctrl], float)
    fa = QADR[fruit]; P = Q[:, fa:fa + 3]; z0 = float(plan0["object_poses"][fruit]["pos"][2])
    closes = [int(t[i]) for i in range(1, len(t)) if g[i] < 0.9 and g[i - 1] >= 0.9]
    if len(t) and g[0] < 0.9:
        closes = [int(t[0])] + closes
    lift_i = next((i for i in range(len(t)) if P[i, 2] >= z0 + 0.05), None)
    cr = Q[-1, QADR["crate"]:QADR["crate"] + 3]
    fin = P[-1]
    in_crate = bool(abs(fin[0] - cr[0]) <= 0.09 and abs(fin[1] - cr[1]) <= 0.15 and fin[2] < cr[2] + 0.08)
    slip_t = None; slip_kind = None
    if lift_i is not None:
        j = next((i for i in range(lift_i, len(t)) if P[i, 2] < z0 + 0.02), None)
        if j is not None:
            slip_t = int(t[j]); slip_kind = "at_pick" if slip_t - int(t[lift_i]) <= 64 else "carry"
    oc = "success" if in_crate else ("no_close" if not closes else ("miss" if lift_i is None else "slip"))
    return {"close_ts": closes, "n_close_attempts": len(closes), "lift_t": None if lift_i is None else int(t[lift_i]),
            "fruit_dz_max_mm": float(1e3 * (P[:, 2].max() - z0)), "in_crate_final": in_crate, "slip_t": slip_t,
            "slip_kind": slip_kind, "outcome": oc, "n_control": int(len(t)), "fruit_z0": z0}


def contact_window(ctrl_by_t, close_t, arm, fruit):
    rows = []; p0 = None
    for tt in range(close_t - 24, close_t + 25):
        c = ctrl_by_t.get(tt)
        if c is None:
            continue
        q = np.asarray(c["qpos"], float); fa = QADR[fruit]; pf = q[fa:fa + 3]
        if p0 is None:
            p0 = pf.copy()
        h = hand_fruit(q, arm, fruit)
        rows.append({"t": tt, "dt": tt - close_t, **h, "fruit_disp_mm": float(1e3 * np.linalg.norm(pf - p0)),
                     "grip_cmd": (c.get("applied_ctrl") or {}).get("gripper", [None, None])[0 if arm == "left" else 1],
                     "grip_open_meas": float(abs(q[gt()["J"][f"gripper_finger_{'l' if arm == 'left' else 'r'}1"]]) / 0.045)})
    pre = [r for r in rows if r["dt"] < 0]; post = [r for r in rows if r["dt"] >= 0]
    first_c = next((r["dt"] for r in rows if r["finger1_contacts"] + r["finger2_contacts"] + r["palm_contacts"] > 0), None)
    return {"close_t": close_t, "rows": rows,
            "first_contact_dt": first_c,
            "pre_close_contact_rows": sum(1 for r in pre if r["finger1_contacts"] + r["finger2_contacts"] + r["palm_contacts"] > 0),
            "pre_close_fruit_disp_mm": max((r["fruit_disp_mm"] for r in pre), default=None),
            "post_close_fruit_disp_mm": max((r["fruit_disp_mm"] for r in post), default=None),
            "min_finger_mm_pre": min((min(r["finger1_mm"], r["finger2_mm"]) for r in pre), default=None),
            "both_fingers_contact_rows_post": sum(1 for r in post if r["finger1_contacts"] > 0 and r["finger2_contacts"] > 0)}


def point_info(q, pb, surf):
    o, R = base_T(q)
    w = o + R @ np.asarray(pb, float)
    lab, dd = nearest_label(w, surf)
    return {"world": w, "nearest": lab, "dist_mm": {k: round(1e3 * v, 1) for k, v in dd.items()}}


def chunk_timeline(z, s, plan_row, surf):
    q = np.asarray(plan_row["qpos"], float)
    man = s.get("manipulated") or {}; g = s.get("grasp") or {}; ex = s.get("exclusion") or {}
    e = {"seq": None, "t": plan_row["t_step"], "grasp_state": g.get("state"), "attach": bool(g.get("attach")),
         "attached": bool(g.get("attached")), "detach": bool(g.get("detach")),
         "man_id": man.get("id"), "man_state": man.get("state"), "switched_from": man.get("switched_from"),
         "decision": ((s.get("admissibility") or {}).get("decision") or {}).get("mode"),
         "excl_active": ex.get("active"), "excl_id": ex.get("manipulated_id"), "has_target": s.get("has_target"),
         "status": s.get("status"), "gate": (plan_row.get("gate") or {}).get("action")}
    e["man_centroid"] = None if man.get("centroid") is None else point_info(q, man["centroid"], surf)
    tc = z["target_centroid"] if "target_centroid" in z.files else None
    e["target_centroid"] = None if tc is None or np.asarray(tc).size != 3 or not np.isfinite(tc).all() else point_info(q, tc, surf)
    att = {}
    for cam in ("head", "left_wrist", "right_wrist"):
        k = f"attention_{cam}"
        if k in z.files:
            a = np.asarray(z[k], float); i = int(np.nanargmax(a)) if np.isfinite(a).any() else 0
            att[cam] = {"row": i // a.shape[1], "col": i % a.shape[1], "max": float(np.nanmax(a)), "sum": float(np.nansum(a))}
    e["attention"] = att
    pk = s.get("attention_peak_index")
    if pk is not None and "attention_cloud" in z.files and 0 <= int(pk) < z["attention_cloud"].shape[0]:
        e["attention_peak"] = point_info(q, z["attention_cloud"][int(pk)], surf)
    else:
        e["attention_peak"] = None
    to = s.get("to") or {}; m = to.get("metrics") or {}
    mv = lit(m.get("max_violation_pair")) or {}; pc = lit(m.get("path_check")) or {}
    n_robot = int(z["sphere_radii"].shape[0]) if "sphere_radii" in z.files else None
    cat = lambda qq: None if qq is None else ("base" if int(qq) < 122 else ("gripper_cover" if n_robot is not None and int(qq) < n_robot else "held"))
    e["to"] = {"status": to.get("status"), "clearance_mm": None if m.get("clearance_m") is None else 1e3 * m["clearance_m"],
               "wp": {"link": mv.get("link"), "query": mv.get("query"), "cat": cat(mv.get("query")) if mv.get("block") == "esdf" else mv.get("block"),
                      "tier": mv.get("tier"), "class": mv.get("obstacle_margin_class"), "obstacle": mv.get("obstacle"),
                      "clearance_mm": None if mv.get("clearance_m") is None else 1e3 * mv["clearance_m"], "point": mv.get("point_m")},
               "path": None}
    if isinstance(pc, dict) and "clearance_m" in pc:
        w = pc.get("worst") or {}
        e["to"]["path"] = {"clearance_mm": None if pc.get("clearance_m") is None else 1e3 * pc["clearance_m"], "link": w.get("link"),
                           "query": w.get("query"), "cat": cat(w.get("query")), "at_kind": w.get("at_kind"), "point": w.get("point_m"),
                           "target_rows_excluded": pc.get("target_rows_excluded"), "floor_rows": pc.get("floor_rows")}
    for nm in ("wp", "path"):
        r = e["to"].get(nm)
        if r and r.get("point") is not None:
            r["point_nearest"] = point_info(q, r["point"], surf)["nearest"]
    wc = e["to"]["wp"]["clearance_mm"]; pcm = (e["to"]["path"] or {}).get("clearance_mm")
    e["to"]["binding"] = "path" if (pcm is not None and (wc is None or pcm < wc)) else "wp"
    e["verdict_hold"] = bool((plan_row.get("gate") or {}).get("action") == "hold")
    e["tip_alignment"] = g.get("tip_alignment")
    return e


def chunk_rows(z, s, plan_row, R, arm, fruit, surf, om):
    """hand-row rebuild + carve check + deflection for one pre-attach chunk."""
    TS = to_setup(str(pathlib.Path(R["server"]) / "serve_safe.log"), str(pathlib.Path(R["run_dir"]) / "manifest.json"))
    FIA, lin, layout = TS["FIA"], TS["lin"], TS["layout"]
    q = np.asarray(plan_row["qpos"], float)
    qn = FIA.q_now_from_qpos(q, TS["qadr"])
    refc = np.asarray(z["refined_chunk"], float); refr = np.asarray(z["reference_chunk"], float)
    meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
    path, _ = TS["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, 8, default=TS["robot"].param_vector())
    lin.set_joint_parameter_path(path); lin.set_attached(None, None)
    traj = layout.chunk_to_trajectory(refc)[:, :8]
    field = FIA.build_field(z, 0.5)
    has_t = bool(s.get("has_target")) and "target_points" in z.files and z["target_points"].shape[0] > 0
    margin, mask, authorized, _, _ = FIA.contact_policy(TS["st"], TS["robot"].sphere_link_names, has_target=has_t,
                                                        has_free=bool(field.target_free_layers), attached=False)
    tf = lit((s.get("esdf_stats") or {}).get("target_free")) or {}
    ball = (tf.get("ball") or {}) if isinstance(tf, dict) else {}
    zpl = ball.get("support_plane_z_m", 0.82)
    scene = FIA.make_scene(field, lin.robot_radii, TS["st"].esdf_margin, target_points=z["target_points"] if has_t else None,
                           margin=margin, mask=mask)
    scene = dataclasses.replace(scene, obstacle_margin=om / 1e3, support_planes=np.array([[0.0, 0.0, 1.0, zpl]]))
    cent = lin.sphere_states(traj, qn)[0]                      # (8, 122, 3) base frame
    cl = lin.esdf_clearance(traj, qn, scene)[..., 0]           # (8, 122)
    plain = FIA.make_scene(field, lin.robot_radii, TS["st"].esdf_margin)
    fid = {"centre0_mm": float(1e3 * np.abs(cent[0] - np.asarray(z["sphere_centres"], float)[:cent.shape[1]]).max())}
    if "clearance" in z.files:
        rc = np.asarray(z["clearance"], float); mine = lin.esdf_clearance(traj, qn, plain)[..., 0]
        fid["record_clearance_mm"] = float(1e3 * np.abs(mine[:rc.shape[0], :rc.shape[1]] - rc[:, :mine.shape[1]]).max())
    m_rec = ((s.get("to") or {}).get("metrics") or {})
    try:
        c_w, pair = lin.worst_row(traj, qn, scene)
        fid["to_view_rebuilt_mm"] = float(1e3 * c_w); fid["to_view_recorded_mm"] = None if m_rec.get("clearance_m") is None else 1e3 * m_rec["clearance_m"]
        rp = lit(m_rec.get("max_violation_pair")) or {}
        fid["same_row"] = bool(pair is not None and rp and int(rp.get("step", -1)) == int(pair["step"]) and int(rp.get("query", -1)) == int(pair["query"]))
    except Exception as ex:
        fid["to_view_error"] = repr(ex)
    # class (code rule) per row
    flat = cent.reshape(-1, 3)
    d = np.asarray(field.distance(flat), float).reshape(cent.shape[:2])
    if mask is not None and np.asarray(mask).any():
        mk = np.asarray(mask, bool).reshape(-1)[:cent.shape[1]]
        dfree = np.asarray(field.target_free_distance(flat), float).reshape(cent.shape[:2])
        d = np.where(mk[None, :], dfree, d)
    d_obj = lin._manipulated_distance(flat, scene)
    d_obj = None if d_obj is None else d_obj.reshape(cent.shape[:2])
    cls = lin._obstacle_rows(d, flat, scene, d_obj)             # 0.. per OBSTACLE_CLASSES
    from AG3S.trajopt.linearize import OBSTACLE_CLASSES
    names = TS["names"]; hl = hand_links(arm)
    hidx = [i for i, n in enumerate(names) if n in hl]
    o, Rb = base_T(q)
    near = []
    for k in range(cent.shape[0]):
        for i in hidx:
            if cl[k, i] > 0.030:
                continue
            w = o + Rb @ cent[k, i]
            lab, dd = nearest_label(w, surf)
            r = float(lin.robot_radii[i])
            near.append({"k": k, "i": i, "link": names[i], "cl_mm": float(1e3 * cl[k, i]), "d_seen_mm": float(1e3 * d[k, i]),
                         "class": OBSTACLE_CLASSES[int(cls[k, i])], "gt_nearest": lab,
                         "gt_fruit_mm": float(1e3 * (dd["fruit"] - r)), "gt_obstacle_mm": None if "obstacle" not in dd else float(1e3 * (dd["obstacle"] - r)),
                         "gt_table_mm": float(1e3 * (dd["table"] - r)), "tier": None})
    hcl = cl[:, hidx]
    kmin, imin = np.unravel_index(int(np.argmin(hcl)), hcl.shape)
    worst = {"cl_mm": float(1e3 * hcl[kmin, imin]), "link": names[hidx[imin]], "k": int(kmin)}
    w = o + Rb @ cent[kmin, hidx[imin]]
    lab, dd = nearest_label(w, surf)
    worst["gt_nearest"] = lab; worst["class"] = OBSTACLE_CLASSES[int(cls[kmin, hidx[imin]])]
    worst["gt_dist_mm"] = {k2: round(1e3 * (v - float(lin.robot_radii[hidx[imin]])), 1) for k2, v in dd.items()}
    # carve check
    G = gt(); set_q(q)
    fv = []
    m, dd_ = G["m"], G["d"]
    for g in G["fruit"][fruit]:
        did = int(m.geom_dataid[g]); va, nv = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
        fv.append(dd_.geom_xpos[g] + m.mesh_vert[va:va + nv].astype(float) @ dd_.geom_xmat[g].reshape(3, 3).T)
    fv = np.concatenate(fv); fc = q[QADR[fruit]:QADR[fruit] + 3]
    up = fv[fv[:, 2] > fc[2]]
    upb = (up - o) @ Rb                                         # world -> base
    d_free_up = np.asarray(field.target_free_distance(upb), float) if field.target_free_layers else None
    d_main_up = np.asarray(field.distance(upb), float)
    fcb = Rb.T @ (fc - o)
    carve = {"d_free_up_median_mm": None if d_free_up is None else float(1e3 * np.median(d_free_up)),
             "d_main_up_median_mm": float(1e3 * np.median(d_main_up)),
             "frac_up_free_ge5mm": None if d_free_up is None else float((d_free_up >= 0.005).mean()),
             "ball_centre_to_fruit_mm": None if not ball.get("centre_m") else float(1e3 * np.linalg.norm(np.asarray(ball["centre_m"]) - fcb)),
             "ball_radius_mm": ball.get("radius_mm"), "mode": tf.get("mode") if isinstance(tf, dict) else None}
    carve["carved"] = None if d_free_up is None else bool(np.median(d_free_up) >= 0.005)
    # remnant: upper fruit vertices (<= 64, evenly subsampled) whose target-free distance is <= 10 mm while every GT surface
    # other than the fruit is > 25 mm away -> something fruit-like is still in the layer the hand rows read
    if d_free_up is not None and len(up):
        sel = np.linspace(0, len(up) - 1, min(64, len(up))).astype(int)
        other = {k2: v for k2, v in surf.items() if k2 != "fruit"}; other["hand"] = gt()["hands"]
        g_other = np.array([min(probe_dist(up[j], gg) for gg in other.values()) for j in sel])
        rem = (d_free_up[sel] <= 0.010) & (g_other > 0.025)
        rem_main = (d_main_up[sel] <= 0.010) & (g_other > 0.025)
        carve["n_up_sampled"] = int(len(sel)); carve["n_up_far_from_others"] = int((g_other > 0.025).sum())
        carve["remnant_frac_free"] = float(rem.sum() / max(1, (g_other > 0.025).sum()))
        carve["remnant_frac_main"] = float(rem_main.sum() / max(1, (g_other > 0.025).sum()))
    # deflection
    best = (-1.0, None)
    for k in range(8):
        _, tr = tips_at(q_with_row(q, refr[k], arm), arm)
        _, tf_ = tips_at(q_with_row(q, refc[k], arm), arm)
        dl = float(1e3 * np.linalg.norm(tr - tf_))
        if dl > best[0]:
            best = (dl, {"k": k, "defl_mm": dl, "radial_mm": float(1e3 * (np.linalg.norm(tf_ - fc) - np.linalg.norm(tr - fc))),
                         "tip_ref_to_fruit_mm": float(1e3 * np.linalg.norm(tr - fc)), "tip_fin_to_fruit_mm": float(1e3 * np.linalg.norm(tf_ - fc))})
    _, tip_now = tips_at(q, arm)
    from collections import Counter
    fr_rows = [r for r in near if r["gt_nearest"] == "fruit"]
    return {"fid": fid, "n_hand_rows": int(hcl.size), "n_near_rows_le30": len(near),
            "near_gt": dict(Counter(r["gt_nearest"] for r in near)),
            "near_class": dict(Counter(r["class"] for r in near)),
            "fruit_rows": len(fr_rows), "fruit_rows_class": dict(Counter(r["class"] for r in fr_rows)),
            "fruit_rows_min_cl_mm": min((r["cl_mm"] for r in fr_rows), default=None),
            "fruit_rows_obstacle_margin_applied": sum(1 for r in fr_rows if r["class"] == "obstacle") if om > 0 else 0,
            "worst_hand_row": worst, "carve": carve, "deflection": best[1],
            "tip_now_to_fruit_mm": float(1e3 * np.linalg.norm(tip_now - fc)), "tip_now_dz_mm": float(1e3 * (tip_now[2] - fc[2])),
            "authorized": authorized, "obstacle_margin_mm": om, "support_plane_z": zpl}


def one(key):
    try:
        return key, run_one(key)
    except Exception as ex:
        import traceback
        return key, {"error": repr(ex), "tb": traceback.format_exc()}


def run_one(key):
    RU = json.loads((U / "raw/runs_u.json").read_text())
    R = RU["server_runs"].get(key) or RU["e0_runs"][key]
    rd = pathlib.Path(R["run_dir"]); arm, fruit = R["arm"], R["fruit"]
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [r for r in rows if r["kind"] == "control"]; plan = [r for r in rows if r["kind"] == "planning"]
    ctrl_by_t = {c["t_step"]: c for c in ctrl}
    obst = place_obstacle(R["obstacle"])
    surf = surfaces(fruit, obst)
    surf_h = surfaces(fruit, obst, hands=True)
    out = {"key": key, **{k: R[k] for k in ("campaign", "cond", "stage", "episode", "seed", "arm", "fruit", "obstacle")}}
    out.update(outcome(ctrl, plan[0], arm, fruit, obst))
    out["contact"] = [contact_window(ctrl_by_t, ct, arm, fruit) for ct in out["close_ts"][:4]]
    if R["cond"] == "E0" or "chunks" not in R:
        return jl(out)
    for k in ("obstacle_margin_mm", "path_check", "gripper_cover", "held_body_cover"):
        out[k] = R[k]
    by_seq = {p["seq"]: p for p in plan}
    tl = []; detail = []
    for seq in sorted(R["chunks"], key=int):
        p = by_seq.get(int(seq))
        if p is None:
            continue
        with np.load(R["chunks"][seq]["npz"], allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
            e = chunk_timeline(z, s, p, surf_h); e["seq"] = int(seq)
            tl.append(e)
            if e["grasp_state"] in ("latched", "closing") and not e["attached"]:
                try:
                    r = chunk_rows(z, s, p, R, arm, fruit, surf, R["obstacle_margin_mm"])
                except Exception as ex:
                    r = {"error": repr(ex)}
                r["seq"] = int(seq); r["t"] = p["t_step"]; r["grasp_state"] = e["grasp_state"]; r["man_id"] = e["man_id"]; r["gate"] = e["gate"]
                detail.append(r)
    out["timeline"] = tl; out["rows"] = detail
    return jl(out)


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    sub = sys.argv[2] if len(sys.argv) > 2 else None
    RU = json.loads((U / "raw/runs_u.json").read_text())
    keys = sorted(RU["server_runs"]) + sorted(RU["e0_runs"])
    if sub:
        keys = [k for k in keys if sub in k]
    outdir = U / "raw/runs"; outdir.mkdir(parents=True, exist_ok=True)
    todo = [k for k in keys if not (outdir / (k.replace("/", "__") + ".json")).exists()]
    print("todo", len(todo), "of", len(keys), flush=True)
    with Pool(nw, maxtasksperchild=4) as pool:
        for k, r in pool.imap_unordered(one, todo, chunksize=1):
            (outdir / (k.replace("/", "__") + ".json")).write_text(json.dumps(r, default=float))
            print(k, r.get("outcome"), r.get("error"), len(r.get("rows") or []), flush=True)
