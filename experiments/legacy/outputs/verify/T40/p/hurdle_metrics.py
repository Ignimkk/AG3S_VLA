"""T40 G — per-run hurdle metrics (verifier, CPU only, MUJOCO_GL=osmesa). Client code from the snapshot ${AG3S_PROJECT_ROOT}-t40.

model      : snapshot model_transport_pick_place_obstacles.xml (same qpos layout as model_transport.xml, nq 66).
hurdle     : PickPlaceObstacleManager(config).activate(profile) of the snapshot (the client's placement code). For a run
             WITHOUT obstacle (stage B) the hurdle of the same episode's C config is placed virtually (no physics; distance
             only) so B and C can be drawn on one axis.
distance   : manager.pair_distance (= the client's obstacle distance: mj_geomDistance, capsule-box exact), capped at 0.30 m,
             same bounding-sphere prefilter as the client; per control row on the recorded qpos (frames.jsonl control row
             'qpos' = pose when that step's action was commanded, before mj_step). Sets: A gripper = EE_BODY_L + ee_finger_l1/l2;
             B apple = apple geoms; C robot = every other collision geom under body 'base'; F = banana/orange/pear.
lift_t     : first control row after the last row <= grasp_t (planning t, analyze_e0 rule) whose apple centre z < z0 + 2 mm
             (T40 A definition); None without grasp.
in hand    : apple centre to finger midpoint (ee_finger_l1/l2 body xpos) distance per control row. apple_left_hand_t = first
             control row t > grasp_t where that distance > (its value at grasp_t) + 30 mm. None if it never happens.
client     : traj.npz obstacle_safety_json (robot_collision, payload_collision, contact_steps, min_robot_clearance_m,
             contact_pairs; contacts are MuJoCo contacts seen every sim step) and the '[SAFETY] obstacle stop' log line.
AG3S choice: per E3b server chunk (paired by time) target_centroid and manipulated.centroid (robot base frame -> world with the
             base pose at that planning row's qpos); distance of that point to the hurdle surface (analytic: capsule bar,
             cylinders), to the apple / banana / orange / pear / crate body centres (planning-row object_poses); nearest of
             these; number of target_points within 15 mm of the hurdle surface vs within 45 mm of the apple centre."""
import json, math, pathlib, sys, re, datetime, os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco

SNAP = pathlib.Path("${AG3S_PROJECT_ROOT}-t40/pi05_TO_hybrid")
sys.path.insert(0, str(SNAP / "rby1_manipulation" / "src"))
from rby1_manipulation.simulation import pick_place_obstacles as PPO  # noqa: E402
assert PPO.__file__.startswith(str(SNAP)), PPO.__file__
XML = str(SNAP / "rby1_description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml")
CAP = 0.30
_M = {}


def model():
    if "m" not in _M:
        m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
        coll = np.array([bool(m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])
        B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        gb = lambda names: np.array([g for g in range(m.ngeom) if coll[g] and int(m.geom_bodyid[g]) in {B(n) for n in names}], int)

        def desc(b, root):
            while b > 0:
                if b == root:
                    return True
                b = int(m.body_parentid[b])
            return False
        A = gb(["EE_BODY_L", "ee_finger_l1", "ee_finger_l2"])
        robot = np.array([g for g in range(m.ngeom) if coll[g] and desc(int(m.geom_bodyid[g]), B("base"))], int)
        sA = set(A.tolist())
        _M.update(m=m, d=d, A=A, B=gb(["apple"]), C=np.array([g for g in robot if g not in sA], int), F=gb(["banana", "orange", "pear"]),
                  bid={n: B(n) for n in ("apple", "ee_finger_l1", "ee_finger_l2", "base", "crate")})
    return _M


def manager(config_path, profile):
    M = model()
    cfg = PPO.load_pick_place_obstacle_config(config_path)
    mg = PPO.PickPlaceObstacleManager(M["m"], M["d"], cfg)
    mg.activate(profile)
    return mg


def hurdle_shapes(mg):
    """analytic hurdle primitives in world: list of (type, centre, axis, half_len, radius) from the active geoms."""
    M = model(); m, d = M["m"], M["d"]
    out = []
    for g in sorted(mg.collision_geom_ids):
        R = d.geom_xmat[g].reshape(3, 3); t = int(m.geom_type[g]); s = m.geom_size[g]
        out.append((t, d.geom_xpos[g].copy(), R[:, 2].copy(), float(s[1]), float(s[0])))
    return out


def point_to_hurdle(p, shapes):
    best = np.inf
    for t, c, ax, hl, r in shapes:
        v = np.asarray(p) - c; a = float(v @ ax)
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            dd = np.linalg.norm(v - np.clip(a, -hl, hl) * ax) - r
        else:  # cylinder
            rad = np.linalg.norm(v - a * ax); dr, dz = rad - r, abs(a) - hl
            dd = math.hypot(max(dr, 0), max(dz, 0)) + min(max(dr, dz), 0)
        best = min(best, dd)
    return float(best)


def series(mg, Q, sets=("A", "B", "C", "F")):
    M = model(); m, d = M["m"], M["d"]
    H = np.array(sorted(mg.collision_geom_ids), int)
    out = {k: np.full(len(Q), CAP) for k in sets}; pair = {k: [None] * len(Q) for k in sets}
    ft = np.zeros(6)
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"
    for i, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        for k in sets:
            G = M[k]
            for o in H:
                gap = np.linalg.norm(d.geom_xpos[G] - d.geom_xpos[o], axis=1) - m.geom_rbound[G] - m.geom_rbound[o]
                for g in G[gap <= min(CAP, out[k][i])]:
                    v = mg.pair_distance(int(o), int(g), CAP, ft)
                    if v < out[k][i]:
                        out[k][i] = v; pair[k][i] = (gname(o), gname(g))
    return out, pair


def in_hand(Q):
    M = model(); m, d = M["m"], M["d"]; b = M["bid"]
    r = np.zeros(len(Q))
    for i, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        r[i] = np.linalg.norm(d.xpos[b["apple"]] - 0.5 * (d.xpos[b["ee_finger_l1"]] + d.xpos[b["ee_finger_l2"]]))
    return r


def base_to_world(q, p):
    M = model(); m, d = M["m"], M["d"]
    d.qpos[:] = q; mujoco.mj_kinematics(m, d)
    b = M["bid"]["base"]
    return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(p, float)


def phase_min(t, arr, mask, pairs=None):
    if not mask.any():
        return None
    i = int(np.argmin(np.where(mask, arr, np.inf)))
    e = {"min_m": float(arr[i]), "t": int(t[i]), "n_rows_le_0": int(((arr <= 0) & mask).sum())}
    if pairs is not None:
        e["pair"] = pairs[i]
    return e


def run_hurdle(rd, config_path, profile, grasp_t, place_t, physical, segments=None):
    """all hurdle facts of one run (see module doc)."""
    rd = pathlib.Path(rd)
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [r for r in rows if r["kind"] == "control"]; plan = [r for r in rows if r["kind"] == "planning"]
    t = np.array([c["t_step"] for c in ctrl]); Q = np.array([c["qpos"] for c in ctrl])
    mg = manager(config_path, profile)
    sh = hurdle_shapes(mg)
    s, pr = series(mg, Q)
    z = Q[:, 40]; z0 = plan[0]["object_poses"]["apple"]["pos"][2]
    lift_t = None
    if grasp_t is not None:
        k = int(np.searchsorted(t, grasp_t)); k = min(k, len(t) - 1)
        while k > 0 and z[k - 1] >= z0 + 0.002:
            k -= 1
        lift_t = int(t[k])
    hand = in_hand(Q)
    left_t = None
    if grasp_t is not None and grasp_t in set(t.tolist()):
        gi = int(np.where(t == grasp_t)[0][0]); ref = hand[gi]
        j = next((i for i in range(gi + 1, len(t)) if hand[i] > ref + 0.030), None)
        left_t = None if j is None else int(t[j])
    end_carry = place_t if place_t is not None else (left_t if left_t is not None else int(t[-1]))
    ph = {"before_lift": (t < lift_t) if lift_t is not None else np.ones(len(t), bool),
          "carry_lift_to_place_or_end": ((t >= lift_t) & (t <= end_carry)) if lift_t is not None else np.zeros(len(t), bool),
          "after": (t > end_carry) if lift_t is not None else np.zeros(len(t), bool), "all": np.ones(len(t), bool)}
    out = {"run_dir": str(rd), "physical_hurdle": physical, "config": str(config_path), "profile": profile,
           "hurdle": mg.config["profiles"][profile]["obstacles"][0], "lift_t": lift_t, "grasp_t": grasp_t, "place_t": place_t,
           "apple_left_hand_t": left_t, "n_control": len(t), "phases": {}}
    for pn, mk in ph.items():
        out["phases"][pn] = {k: phase_min(t, s[k], mk, pr[k]) for k in s}
    out["series"] = {"t": t.tolist(), **{k: [float(x) for x in v] for k, v in s.items()}, "apple_finger_mid_m": hand.tolist(),
                     "apple_z_above_table_m": (z - 0.82).tolist()}
    if left_t is not None:
        mk = (t >= (grasp_t or 0)) & (t <= left_t)
        out["apple_hurdle_min_grasp_to_left_hand"] = phase_min(t, s["B"], mk, pr["B"])
        out["gripper_hurdle_min_grasp_to_left_hand"] = phase_min(t, s["A"], mk, pr["A"])
    # client side
    try:
        zz = np.load(rd / "traj.npz")
        out["client_obstacle_profile"] = str(zz["obstacle_profile"])
        out["client_obstacle_safety"] = json.loads(str(zz["obstacle_safety_json"]))
    except Exception as e:
        out["client_obstacle_safety"] = {"error": repr(e)}
    log = rd.parent / f"{rd.name}.client.log"
    txt = log.read_text(errors="replace") if log.exists() else ""
    out["client_obstacle_stop_lines"] = [l for l in txt.splitlines() if "[SAFETY] obstacle stop" in l]
    out["client_obstacle_summary_lines"] = [l for l in txt.splitlines() if "obstacle safety summary" in l or l.strip().startswith("obstacles  :")]
    # AG3S choice (E3b)
    if segments:
        by_seq = {p["seq"]: p for p in plan}
        ch = []
        for f in segments:
            zc = np.load(f, allow_pickle=True); sm = json.loads(str(zc["summary_json"])); seq = int(zc["t_step"])
            p = by_seq.get(seq)
            if p is None:
                continue
            q = np.array(p["qpos"]); op = p["object_poses"]
            e = {"seq": seq, "t": p["t_step"], "has_target": sm.get("has_target")}
            for key, pt in (("target", zc["target_centroid"] if "target_centroid" in zc.files else None),
                            ("manipulated", (sm.get("manipulated") or {}).get("centroid"))):
                if pt is None or np.asarray(pt).size != 3 or not np.isfinite(np.asarray(pt, float)).all():
                    e[key] = None; continue
                w = base_to_world(q, pt)
                dd = {b: float(np.linalg.norm(w - np.asarray(op[b]["pos"]))) for b in ("apple", "banana", "orange", "pear", "crate") if b in op}
                dd["hurdle_surface"] = point_to_hurdle(w, sh)
                e[key] = {"world": w.tolist(), "dist_m": dd, "nearest": min(dd, key=dd.get),
                          "id": (sm.get("manipulated") or {}).get("id") if key == "manipulated" else None,
                          "state": (sm.get("manipulated") or {}).get("state") if key == "manipulated" else None}
            tp = zc["target_points"] if "target_points" in zc.files else None
            if tp is not None and np.asarray(tp).ndim == 2 and len(tp):
                W = np.array([base_to_world(q, x) for x in np.asarray(tp, float)[:: max(1, len(tp) // 400)]])
                e["target_points_sampled"] = int(len(W))
                e["target_points_within_15mm_of_hurdle"] = int(sum(point_to_hurdle(x, sh) <= 0.015 for x in W))
                e["target_points_within_45mm_of_apple_centre"] = int((np.linalg.norm(W - np.asarray(op["apple"]["pos"]), axis=1) <= 0.045).sum())
            ch.append(e)
        out["ag3s_chunks"] = ch
        out["ag3s_n_chunks"] = len(ch)
        out["ag3s_n_target_nearest_hurdle"] = sum(1 for e in ch if (e.get("target") or {}).get("nearest") == "hurdle_surface")
        out["ag3s_n_manipulated_nearest_hurdle"] = sum(1 for e in ch if (e.get("manipulated") or {}).get("nearest") == "hurdle_surface")
        out["ag3s_n_chunks_with_target_points_on_hurdle"] = sum(1 for e in ch if (e.get("target_points_within_15mm_of_hurdle") or 0) > 0)
        out["ag3s_min_target_to_hurdle_m"] = min([e["target"]["dist_m"]["hurdle_surface"] for e in ch if e.get("target")], default=None)
        out["ag3s_min_manipulated_to_hurdle_m"] = min([e["manipulated"]["dist_m"]["hurdle_surface"] for e in ch if e.get("manipulated")], default=None)
    return out
