"""T42-2 per-run obstacle metrics (verifier; generalized from T40/p/hurdle_metrics.py to any slot set, target fruit and arm).
CPU only, MUJOCO_GL=osmesa. Client code from the snapshot ${AG3S_PROJECT_ROOT}-t42c (pi05_TO_hybrid fce4af8).

model      : snapshot model_transport_pick_place_obstacles.xml; obstacles placed by the snapshot PickPlaceObstacleManager
             (config, profile).activate() = the client's placement and sizing code.
distance   : manager.pair_distance (certified, T40 H), capped 0.30 m, per control row on the recorded qpos (control row 'qpos' = the
             pose when that step's action was commanded). Groups: GRIP = EE_BODY_<arm> + ee_finger_<arm>1/2; TARGET = the target
             fruit; ROBOT_OTHER = every other collision geom under body 'base'; FRUITS = the other three fruits.
phases     : lift_t (T40 A rule) / place_t / grasp_t from analyze_t42; before_lift = t < lift_t (all rows when no lift);
             carry = lift_t <= t <= place_t (or the last row); after = t > place_t.
client     : traj.npz obstacle_safety_json (robot_collision, payload_collision, contact_steps, min_robot_clearance_m,
             contact_pairs = MuJoCo contacts every sim step) and the '[SAFETY] obstacle stop' log line; stop_t = last control row.
AG3S       : per E3b server chunk paired to the run (by content), target_centroid and manipulated.centroid (base -> world at that
             planning row's qpos): distance to the obstacle surface (analytic capsule / cylinder) and to the fruit / crate body
             centres; nearest; target_points within 15 mm of the obstacle surface vs within 45 mm of the target centre.
usage: import; run_obst(rd, config, profile, grasp_t, place_t, segment)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, sys, os
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, mujoco

SNAP = pathlib.Path("${AG3S_PROJECT_ROOT}-t42c/pi05_TO_hybrid")
sys.path.insert(0, str(SNAP / "rby1_manipulation" / "src"))
from rby1_manipulation.simulation import pick_place_obstacles as PPO  # noqa: E402
assert PPO.__file__.startswith(str(SNAP)), PPO.__file__
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p'))
import analyze_t42 as AN  # noqa: E402
XML = str(SNAP / "rby1_description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml")
CAP = 0.30
FRUITS = ("apple", "banana", "orange", "pear")
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
        robot = np.array([g for g in range(m.ngeom) if coll[g] and desc(int(m.geom_bodyid[g]), B("base"))], int)
        _M.update(m=m, d=d, gb=gb, robot=robot, B=B)
    return _M


def groups(fruit, arm):
    M = model(); s = "L" if arm == "left" else "R"
    G = M["gb"]([f"EE_BODY_{s}", f"ee_finger_{s.lower()}1", f"ee_finger_{s.lower()}2"])
    sG = set(G.tolist())
    return {"GRIP": G, "TARGET": M["gb"]([fruit]), "ROBOT_OTHER": np.array([g for g in M["robot"] if g not in sG], int),
            "FRUITS": M["gb"]([f for f in FRUITS if f != fruit])}


def manager(config_path, profile):
    M = model()
    if M.get("mgr") is not None:
        M["mgr"].park_all()   # one manager per model (the snapshot refuses a model left resized)
    mg = PPO.PickPlaceObstacleManager(M["m"], M["d"], PPO.load_pick_place_obstacle_config(config_path))
    mg.activate(profile); M["mgr"] = mg
    return mg


def shapes(mg):
    M = model(); m, d = M["m"], M["d"]; out = []
    for g in sorted(mg.collision_geom_ids):
        R = d.geom_xmat[g].reshape(3, 3)
        out.append((int(m.geom_type[g]), d.geom_xpos[g].copy(), R[:, 2].copy(), float(m.geom_size[g][1]), float(m.geom_size[g][0])))
    return out


def point_to_obst(p, sh):
    best = np.inf
    for t, c, ax, hl, r in sh:
        v = np.asarray(p) - c; a = float(v @ ax)
        if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            dd = np.linalg.norm(v - np.clip(a, -hl, hl) * ax) - r
        else:
            rad = np.linalg.norm(v - a * ax); dr, dz = rad - r, abs(a) - hl
            dd = math.hypot(max(dr, 0), max(dz, 0)) + min(max(dr, dz), 0)
        best = min(best, dd)
    return float(best)


def series(mg, Q, grp, slots):
    """per group and per slot: min distance per row."""
    M = model(); m, d = M["m"], M["d"]
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom#{int(g)}"
    out = {}; pair = {}
    for slot in slots:
        H = np.array(sorted(mg._collision_geoms[slot]), int)
        for k, G in grp.items():
            key = f"{slot}:{k}"; out[key] = np.full(len(Q), CAP); pair[key] = [None] * len(Q)
    ft = np.zeros(6)
    for i, q in enumerate(Q):
        d.qpos[:] = q; mujoco.mj_kinematics(m, d)
        for slot in slots:
            H = np.array(sorted(mg._collision_geoms[slot]), int)
            for k, G in grp.items():
                key = f"{slot}:{k}"
                for o in H:
                    gap = np.linalg.norm(d.geom_xpos[G] - d.geom_xpos[o], axis=1) - m.geom_rbound[G] - m.geom_rbound[o]
                    for g in G[gap <= min(CAP, out[key][i])]:
                        v = mg.pair_distance(int(o), int(g), max(min(CAP, out[key][i]), 1e-6), ft)
                        if v < out[key][i]:
                            out[key][i] = v; pair[key][i] = (gname(o), gname(g))
    return out, pair


def base_to_world(q, p):
    M = model(); m, d = M["m"], M["d"]
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); b = M["B"]("base")
    return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.asarray(p, float)


def phase_min(t, arr, mask, pairs=None):
    if not mask.any():
        return None
    i = int(np.argmin(np.where(mask, arr, np.inf)))
    e = {"min_m": float(arr[i]), "t": int(t[i]), "n_rows_le_0": int(((arr <= 0) & mask).sum())}
    if pairs is not None:
        e["pair"] = pairs[i]
    return e


def run_obst(rd, config_path, profile, segment=None):
    rd = pathlib.Path(rd); a = AN.analyze(rd); fruit, arm = a["target_fruit"], a["used_arm"]
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [r for r in rows if r["kind"] == "control"]; plan = [r for r in rows if r["kind"] == "planning"]
    t = np.array([c["t_step"] for c in ctrl]); Q = np.array([c["qpos"] for c in ctrl])
    mg = manager(config_path, profile); slots = list(mg.active_slots); sh = shapes(mg)
    grp = groups(fruit, arm)
    s, pr = series(mg, Q, grp, slots)
    adr = AN.QADR[fruit]; z0 = plan[0]["object_poses"][fruit]["pos"][2]; z = Q[:, adr + 2]
    lift_t = None
    if a["grasp_t"] is not None:
        k = np.where((t <= a["grasp_t"]) & (z < z0 + 0.002))[0]
        lift_t = int(t[k[-1] + 1]) if len(k) and k[-1] + 1 < len(t) else None
    end = a["place_t"] if a["place_t"] is not None else int(t[-1])
    ph = {"before_lift": (t < lift_t) if lift_t is not None else np.ones(len(t), bool),
          "carry": ((t >= lift_t) & (t <= end)) if lift_t is not None else np.zeros(len(t), bool),
          "after": (t > end) if lift_t is not None else np.zeros(len(t), bool), "all": np.ones(len(t), bool)}
    out = {"run_dir": str(rd), "config": str(config_path), "profile": profile, "slots": slots,
           "obstacles": mg.config["profiles"][profile]["obstacles"], "target_fruit": fruit, "used_arm": arm,
           "lift_t": lift_t, "grasp_t": a["grasp_t"], "place_t": a["place_t"], "n_control": len(t), "phases": {}}
    for pn, mk in ph.items():
        out["phases"][pn] = {k: phase_min(t, s[k], mk, pr[k]) for k in s}
    out["series"] = {"t": t.tolist(), **{k: [float(x) for x in v] for k, v in s.items()}}
    try:
        zz = np.load(rd / "traj.npz")
        out["client_obstacle_profile"] = str(zz["obstacle_profile"])
        out["client_obstacle_safety"] = json.loads(str(zz["obstacle_safety_json"]))
    except Exception as e:
        out["client_obstacle_safety"] = {"error": repr(e)}
    log = rd.parent / f"{rd.name}.client.log"
    txt = log.read_text(errors="replace") if log.exists() else ""
    out["client_obstacle_stop_lines"] = [l for l in txt.splitlines() if "[SAFETY] obstacle stop" in l]
    out["obstacle_stop"] = bool(out["client_obstacle_stop_lines"])
    out["stop_t"] = int(t[-1]) if out["obstacle_stop"] else None
    out["stop_phase"] = None if not out["obstacle_stop"] else ("before_lift" if (lift_t is None or t[-1] < lift_t) else ("carry" if t[-1] <= end else "after"))
    if segment:
        by_seq = {p["seq"]: p for p in plan}; ch = []
        for f in segment:
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
                dd = {b: float(np.linalg.norm(w - np.asarray(op[b]["pos"]))) for b in (*FRUITS, "crate") if b in op}
                dd["obstacle_surface"] = point_to_obst(w, sh)
                e[key] = {"world": w.tolist(), "dist_m": dd, "nearest": min(dd, key=dd.get)}
            tp = zc["target_points"] if "target_points" in zc.files else None
            if tp is not None and np.asarray(tp).ndim == 2 and len(tp):
                W = np.array([base_to_world(q, x) for x in np.asarray(tp, float)[:: max(1, len(tp) // 400)]])
                e["target_points_sampled"] = int(len(W))
                e["target_points_within_15mm_of_obstacle"] = int(sum(point_to_obst(x, sh) <= 0.015 for x in W))
                e["target_points_within_45mm_of_target_centre"] = int((np.linalg.norm(W - np.asarray(op[fruit]["pos"]), axis=1) <= 0.045).sum())
            ch.append(e)
        out["ag3s_chunks"] = ch; out["ag3s_n_chunks"] = len(ch)
        out["ag3s_n_target_nearest_obstacle"] = sum(1 for e in ch if (e.get("target") or {}).get("nearest") == "obstacle_surface")
        out["ag3s_n_manipulated_nearest_obstacle"] = sum(1 for e in ch if (e.get("manipulated") or {}).get("nearest") == "obstacle_surface")
        out["ag3s_n_chunks_with_target_points_on_obstacle"] = sum(1 for e in ch if (e.get("target_points_within_15mm_of_obstacle") or 0) > 0)
        out["ag3s_min_target_to_obstacle_m"] = min([e["target"]["dist_m"]["obstacle_surface"] for e in ch if e.get("target")], default=None)
    return out
