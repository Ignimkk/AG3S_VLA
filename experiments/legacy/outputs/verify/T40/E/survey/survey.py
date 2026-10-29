from common.workspace import workspace_path as _ag3s_workspace_path
import sys, time, json
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/A/p'))
sys.path.insert(0, _ag3s_workspace_path('integration/rby1/manipulation/src'))
import numpy as np, mujoco
from gjk import make_support, gjk_dist, signed_dist
from rby1_manipulation.simulation.pick_place_obstacles import PickPlaceObstacleManager
M = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml')
m = mujoco.MjModel.from_xml_path(M); d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, 0); mujoco.mj_forward(m, d)
cfg = {"version": 1, "profiles": {"clear": {"obstacles": []}, "h": {"scene": "fruit", "obstacles": [
    {"slot": "hurdle_0", "position": [0.55, 0.2, 0.82], "yaw": 0.0, "bar_height": 0.26}]}}}
mgr = PickPlaceObstacleManager(m, d, cfg); mgr.activate("h")
hg = sorted(mgr._collision_geoms["hurdle_0"])
others = list(mgr.robot_geom_ids) + list(mgr.payload_geom_ids)
TN = {0:"plane",2:"sphere",3:"capsule",4:"ellipsoid",5:"cylinder",6:"box",7:"mesh"}
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
arm_j = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{s}_arm_{i}") for s in ("left","right") for i in range(7)]
stats = {}; worst = []
ft = np.zeros(6)
NT = int(sys.argv[2]) if len(sys.argv) > 2 else 60
t0 = time.time()
for trial in range(NT):
    mujoco.mj_resetDataKeyframe(m, d, 0)
    for j in arm_j:
        lo, hi = m.jnt_range[j]; d.qpos[m.jnt_qposadr[j]] = rng.uniform(lo, hi) * 0.6 + 0.4 * d.qpos[m.jnt_qposadr[j]]
    mujoco.mj_forward(m, d)
    # pick a target: robot geom or payload geom, put a random hurdle part near it
    tgt = int(rng.choice(list(mgr.payload_geom_ids) if trial % 2 == 0 else others)); tp = d.geom_xpos[tgt].copy()
    part = int(rng.choice(hg))
    yaw = rng.uniform(-np.pi, np.pi); bh = rng.uniform(0.20, 0.32)
    mgr._place("hurdle_0", [0.55, 0.2, 0.82], yaw, bh); mujoco.mj_forward(m, d)
    off = d.geom_xpos[part] - d.mocap_pos[mgr._slot_mocap_ids["hurdle_0"][0]]
    base = tp - off + rng.normal(0, 0.03, 3)
    mgr._place("hurdle_0", base, yaw, bh); mujoco.mj_forward(m, d)
    for g in hg:
        for o in others:
            r = np.linalg.norm(d.geom_xpos[g] - d.geom_xpos[o]) - m.geom_rbound[g] - m.geom_rbound[o]
            if r > 0.25: continue
            mj = float(mujoco.mj_geomDistance(m, d, g, o, 0.25, ft))
            sa, sb = make_support(m, d, g), make_support(m, d, o)
            gj = gjk_dist(sa, sb, d.geom_xpos[o] - d.geom_xpos[g] + 1e-9)
            key = f"{TN[int(m.geom_type[g])]}-{TN[int(m.geom_type[o])]}"
            st = stats.setdefault(key, {"n": 0, "n_far": 0, "n_overlap": 0, "bad": 0, "maxerr_mm": 0.0})
            if gj >= 0.25:
                st["n_far"] += 1
                if mj < 0.25 - 1e-6 and mj < gj - 1e-4: st["bad"] += 1; worst.append((key, mj, gj))
                continue
            st["n"] += 1
            if gj == 0.0:
                st["n_overlap"] += 1
                if mj > 1e-4:
                    st["bad"] += 1; worst.append((key, mj, gj))
                    np.savez(f"{_ag3s_workspace_path()}/outputs/verify/T40/E/survey/bad_{len(worst)}.npz", g=g, o=o, qpos=d.qpos, mpos=d.mocap_pos, mquat=d.mocap_quat)
                continue
            err = abs(mj - gj) * 1e3; st["maxerr_mm"] = max(st["maxerr_mm"], err)
            if err > 0.1: st["bad"] += 1; worst.append((key, mj, gj, mujoco.mj_id2name(m, 1, g), mujoco.mj_id2name(m, 1, o)))
print(json.dumps(stats, indent=1)); print("worst", sorted(worst, key=lambda w: -abs(w[1]-w[2]))[:10]); print("time", time.time()-t0)
