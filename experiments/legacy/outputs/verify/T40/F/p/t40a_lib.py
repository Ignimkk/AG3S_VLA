"""T40 A shared helpers (verifier). MuJoCo replay of recorded T39 qpos on model_transport_pick_place_obstacles.xml.
Obstacle placement = PickPlaceObstacleManager.activate semantics: mocap_pos = profile position (body origin on the table
surface, z = table_surface_z), mocap_quat = (cos(yaw/2), 0, 0, sin(yaw/2)).
Geom sets (collision-enabled geoms only, contype|conaffinity != 0):
  OB[slot]   : obstacle collision geoms of that slot (bollard: post cylinder r27 z[10,287] mm + foot r45 z[0,20] mm;
               divider: box 208x36x80 mm + 2 feet 60x80x12 mm)
  A_gripper  : bodies EE_BODY_L, ee_finger_l1, ee_finger_l2 (left gripper palm + 2 fingers; T39 used_arm = left)
  B_apple    : apple_geom
  C_robot    : every other robot geom = descendants of body 'base' minus A_gripper (torso, head, both arms incl.
               FT_SENSOR_L, right gripper, wheels)
  F_fruits   : banana/orange/pear geoms ; CR_crate : crate geoms
Distance = gjk.signed_dist (GJK separation; penetration depth = min_n h_A(n)+h_B(-n) when overlapping, < 0), capped at
CAP m; pairs whose bounding spheres are farther than CAP are skipped (reported as CAP). mujoco.mj_geomDistance is NOT used
for the numbers: in MuJoCo 3.11.0 it returned wrong values for box geoms (raw/gjk_validation.log); it is kept as a
cross-check counter only (MJ_CHECK)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib
import numpy as np, mujoco
import gjk

X = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml')
CFG = _ag3s_workspace_path('integration/rby1/manipulation/integration/rby1/manipulation/config/pick_place_obstacles.json')
TABLE_Z = json.load(open(CFG))["table_surface_z"]
CAP = 0.30
m = mujoco.MjModel.from_xml_path(X)
d = mujoco.MjData(m)
BODY = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
GEOM = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom#{g}"
COLL = np.array([bool(m.geom_contype[g] or m.geom_conaffinity[g]) for g in range(m.ngeom)])


def _desc(b, root):
    while b > 0:
        if b == root:
            return True
        b = int(m.body_parentid[b])
    return False


def geoms_of_bodies(names):
    ids = {BODY(n) for n in names}
    return np.array([g for g in range(m.ngeom) if COLL[g] and int(m.geom_bodyid[g]) in ids], int)


OB = {s: geoms_of_bodies([f"pick_obstacle_{s}"]) for s in ("bollard_0", "bollard_1", "divider_0")}
A_GRIP = geoms_of_bodies(["EE_BODY_L", "ee_finger_l1", "ee_finger_l2"])
B_APPLE = geoms_of_bodies(["apple"])
_base = BODY("base")
ROBOT_ALL = np.array([g for g in range(m.ngeom) if COLL[g] and _desc(int(m.geom_bodyid[g]), _base)], int)
C_ROBOT = np.array([g for g in ROBOT_ALL if g not in set(A_GRIP.tolist())], int)
F_FRUITS = {n: geoms_of_bodies([n]) for n in ("banana", "orange", "pear")}
CR_CRATE = geoms_of_bodies(["crate"])
MOCAP = {s: int(m.body_mocapid[BODY(f"pick_obstacle_{s}")]) for s in OB}
PARK = {"bollard_0": (8.0, 8.0, 2.0), "bollard_1": (9.0, 8.0, 2.0), "divider_0": (10.0, 8.0, 2.0)}
SITE_LEE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")


def place(slot, xy, yaw, z=None):
    for s, p in PARK.items():
        d.mocap_pos[MOCAP[s]] = p; d.mocap_quat[MOCAP[s]] = (1, 0, 0, 0)
    d.mocap_pos[MOCAP[slot]] = (xy[0], xy[1], TABLE_Z if z is None else z)
    d.mocap_quat[MOCAP[slot]] = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))


def set_q(q):
    d.qpos[:] = q
    mujoco.mj_kinematics(m, d)


_ft = np.zeros(6)


MJ_CHECK = {"n": 0, "n_disagree_gt_0p1mm": 0, "max_abs_diff_m": 0.0}


def min_dist(ob_geoms, others, cap=CAP, want_pair=False, mj_check=False):
    """min signed distance obstacle geoms <-> other geoms at current d (kinematics done)."""
    best, pair = cap, None
    others = np.asarray(others, int)
    for o in ob_geoms:
        lb = np.linalg.norm(d.geom_xpos[others] - d.geom_xpos[o], axis=1) - m.geom_rbound[others] - m.geom_rbound[o]
        for g in others[lb < min(cap, best) + 1e-9]:
            dist = gjk.signed_dist(m, d, int(o), int(g))
            if mj_check:
                mj = float(mujoco.mj_geomDistance(m, d, int(o), int(g), cap, _ft))
                e = abs(min(mj, cap) - min(dist, cap)); MJ_CHECK["n"] += 1
                MJ_CHECK["max_abs_diff_m"] = max(MJ_CHECK["max_abs_diff_m"], e); MJ_CHECK["n_disagree_gt_0p1mm"] += int(e > 1e-4)
            if dist < best:
                best, pair = dist, (gname(int(o)), gname(int(g)))
    return (best, pair) if want_pair else best


def load_run(rd):
    rd = pathlib.Path(rd)
    rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = [r for r in rows if r["kind"] == "control"]
    plan = [r for r in rows if r["kind"] == "planning"]
    return {"t": np.array([c["t_step"] for c in ctrl]), "q": np.array([c["qpos"] for c in ctrl]),
            "q0": np.array(plan[0]["qpos"]), "object_poses_t0": plan[0]["object_poses"]}
