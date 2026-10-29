from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, time; sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/A/p'))
sys.path.insert(0, _ag3s_workspace_path('integration/rby1/manipulation/src'))
import numpy as np, mujoco
from gjk import make_support, gjk_dist
from rby1_manipulation.simulation.pick_place_obstacles import PickPlaceObstacleManager
M = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml')
m = mujoco.MjModel.from_xml_path(M); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
cfg = {"version": 1, "profiles": {"clear": {"obstacles": []}}}
mgr = PickPlaceObstacleManager(m, d, cfg)
hg = sorted(mgr._collision_geoms["hurdle_0"])
boxes = [g for g in mgr.payload_geom_ids if m.geom_type[g] == 6]
crate_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "crate_free")]
rng = np.random.default_rng(int(sys.argv[1])); N = int(sys.argv[2])
TN = {3: "capsule", 5: "cylinder", 6: "box"}; st = {}; bad = []; ft = np.zeros(6)
for k in range(N):
    th = rng.uniform(-np.pi, np.pi); d.qpos[crate_adr + 3:crate_adr + 7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
    tilt = rng.normal(0, 0.05, 3); q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, tilt / (np.linalg.norm(tilt) + 1e-12), np.linalg.norm(tilt))
    q2 = np.zeros(4); mujoco.mju_mulQuat(q2, q, d.qpos[crate_adr + 3:crate_adr + 7].copy()); d.qpos[crate_adr + 3:crate_adr + 7] = q2
    mujoco.mj_forward(m, d)
    part = int(rng.choice(hg)); box = int(rng.choice(boxes))
    yaw = rng.uniform(-np.pi, np.pi); bh = rng.uniform(0.20, 0.32)
    mgr._place("hurdle_0", [0, 0, 0], yaw, bh); mujoco.mj_forward(m, d)
    off = d.geom_xpos[part] - d.mocap_pos[mgr._slot_mocap_ids["hurdle_0"][0]]
    mgr._place("hurdle_0", d.geom_xpos[box] - off + rng.normal(0, 0.04, 3), yaw, bh); mujoco.mj_forward(m, d)
    for g in hg:
        for o in boxes:
            if np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[o]) - m.geom_rbound[g] - m.geom_rbound[o] > 0.25: continue
            mj = float(mujoco.mj_geomDistance(m, d, g, o, 0.25, ft))
            gj = gjk_dist(make_support(m, d, g), make_support(m, d, o), d.geom_xpos[o] - d.geom_xpos[g] + 1e-9)
            key = f"{TN[int(m.geom_type[g])]}-box"; s = st.setdefault(key, {"n": 0, "overlap": 0, "bad_overlap_reported_positive": 0, "bad_sep": 0, "maxerr_sep_mm": 0.0})
            if gj >= 0.25: continue
            s["n"] += 1
            if gj == 0.0:
                s["overlap"] += 1
                if mj > 1e-6: s["bad_overlap_reported_positive"] += 1; bad.append((key, mj, gj))
            else:
                e = abs(mj - gj) * 1e3; s["maxerr_sep_mm"] = max(s["maxerr_sep_mm"], e)
                if e > 0.1: s["bad_sep"] += 1; bad.append((key, mj, gj))
print(json.dumps(st, indent=1)); print(sorted(bad, key=lambda b: -abs(b[1] - b[2]))[:8])
