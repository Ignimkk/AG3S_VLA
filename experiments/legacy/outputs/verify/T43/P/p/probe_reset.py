"""T43 P probe: reset-scene positions of the gripper bodies + the fruit/crate per episode."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import flib as F
res = {}
for tag in sys.argv[1:]:
    c = F.Ctx(tag); m, d = c.m, c.d; c.restore()
    bp = lambda n: d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)].round(4).tolist()
    names = ["EE_BODY_L", "ee_finger_l1", "ee_finger_l2", "EE_BODY_R", "ee_finger_r1", "link_left_arm_6", "link_right_arm_6", "apple", "orange", "banana", "pear", "crate"]
    r = {n: bp(n) for n in names if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0}
    # robot AABB of collision geoms (rbound spheres) in the table band z < 1.3
    P = d.geom_xpos[c.ROBOT_ALL]; rb = m.geom_rbound[c.ROBOT_ALL]
    low = P[:, 2] - rb < F.TABLE_Z + 0.40
    r["robot_low_geoms"] = [(c.gname(g), np.round(d.geom_xpos[g], 3).tolist(), round(float(m.geom_rbound[g]), 3)) for g in c.ROBOT_ALL[low] if d.geom_xpos[g][0] > 0.25]
    res[tag] = r
    print(tag, {k: v for k, v in r.items() if k != "robot_low_geoms"})
json.dump(res, open(_ag3s_workspace_path('outputs/verify/T43/P/raw/probe_reset.json'), "w"), indent=1)
