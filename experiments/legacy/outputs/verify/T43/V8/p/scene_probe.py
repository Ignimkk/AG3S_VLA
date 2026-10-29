"""V8 scene probe/helper: finger spheres (270, chunk record) of a V8 chunk placed on the hand at control row q (link-following rigid transform from the chunk's planning pose)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa"); os.environ["T43W_SNAP"] = "${AG3S_PROJECT_ROOT}-t43v8"
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/Y/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import render_y as RY, hurdle_metrics as HM  # noqa: E402
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V8'))
MJ = {"ee_left": "EE_BODY_L", "ee_finger_l1": "ee_finger_l2", "ee_finger_l2": "ee_finger_l1", "ee_right": "EE_BODY_R", "ee_finger_r1": "ee_finger_r2", "ee_finger_r2": "ee_finger_r1"}


def chunk_spheres(npz_path):
    z = np.load(npz_path, allow_pickle=True); s = json.loads(str(z["summary_json"]))
    return np.asarray(z["sphere_centres"], float), np.asarray(z["sphere_radii"], float), list(s["sphere_link_names"])


def world_at(M, q_plan, q_now, centres_base, links):
    """base-frame centres at the planning pose -> world at q_now (each sphere follows its link's rigid motion)."""
    m, d = M["m"], M["d"]; B = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    d.qpos[:] = q_plan; mujoco.mj_kinematics(m, d)
    b = B("base"); P = d.xpos[b] + centres_base @ d.xmat[b].reshape(3, 3).T
    pose0 = {n: (d.xpos[B(n)].copy(), d.xmat[B(n)].reshape(3, 3).copy()) for n in set(MJ.values())}
    d.qpos[:] = q_now; mujoco.mj_kinematics(m, d)
    out = P.copy()
    for i, ln in enumerate(links):
        if ln in MJ:
            n = MJ[ln]; x0, R0 = pose0[n]; x1, R1 = d.xpos[B(n)], d.xmat[B(n)].reshape(3, 3)
            out[i] = x1 + R1 @ R0.T @ (P[i] - x0)
        else:
            pass
    return out
