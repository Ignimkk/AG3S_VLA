from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json; sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/A/p'))
sys.path.insert(0, _ag3s_workspace_path('integration/rby1/manipulation/src'))
import numpy as np, mujoco, time
from gjk import make_support, gjk_dist
from rby1_manipulation.simulation.pick_place_obstacles import PickPlaceObstacleManager, capsule_box_distance
M = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml')
m = mujoco.MjModel.from_xml_path(M); d = mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m, d, 0)
mgr = PickPlaceObstacleManager(m, d, {"version": 1, "profiles": {"clear": {"obstacles": []}}})
bar = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pick_obstacle_hurdle_0_collision_bar")
boxes = [g for g in mgr.payload_geom_ids if m.geom_type[g] == 6]
crate_adr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "crate_free")]
rng = np.random.default_rng(5); ft = np.zeros(6)
n = 0; nov = 0; maxe = 0; sign_bad = 0; mj_bad = 0; t_cb = 0
for k in range(6000):
    th = rng.uniform(-np.pi, np.pi); tilt = rng.normal(0, 0.05, 3)
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, tilt / np.linalg.norm(tilt), np.linalg.norm(tilt)); q2 = np.zeros(4)
    mujoco.mju_mulQuat(q2, q, np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])); d.qpos[crate_adr + 3:crate_adr + 7] = q2
    mujoco.mj_forward(m, d)
    box = int(rng.choice(boxes)); yaw = rng.uniform(-np.pi, np.pi); bh = rng.uniform(0.2, 0.32)
    mgr._place("hurdle_0", [0, 0, 0], yaw, bh); mujoco.mj_forward(m, d)
    off = d.geom_xpos[bar] - d.mocap_pos[mgr._slot_mocap_ids["hurdle_0"][0]]
    mgr._place("hurdle_0", d.geom_xpos[box] - off + rng.normal(0, 0.03, 3), yaw, bh); mujoco.mj_forward(m, d)
    gj = gjk_dist(make_support(m, d, bar), make_support(m, d, box), d.geom_xpos[box] - d.geom_xpos[bar] + 1e-9)
    t = time.perf_counter(); cb = capsule_box_distance(m, d, bar, box); t_cb += time.perf_counter() - t
    pd = mgr.pair_distance(bar, box, 0.25)
    mj = float(mujoco.mj_geomDistance(m, d, bar, box, 0.25, ft))
    n += 1
    if gj == 0.0:
        nov += 1; sign_bad += cb > 0; mj_bad += mj > 0
    else:
        maxe = max(maxe, abs(cb - gj)); sign_bad += cb <= 0
    assert pd == min(cb, 0.25)
print(dict(n=n, overlap=nov, cb_sign_wrong=int(sign_bad), mj_overlap_reported_positive=int(mj_bad), cb_max_err_sep_mm=maxe * 1e3, cb_mean_us=t_cb / n * 1e6))
