"""T40 A: PickPlaceObstacleManager.object_clearance / robot_clearance (mj_geomDistance, distmax 0.25) vs GJK (gjk.py) at the
t = 0 scene (recorded planning row 0 qpos) for each written config; lists every obstacle <-> payload pair where
mj_geomDistance < 0.25 together with the GJK separation of the same pair."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, glob, pathlib
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('integration/rby1/manipulation/src'))
from rby1_manipulation.simulation.pick_place_obstacles import PickPlaceObstacleManager, load_pick_place_obstacle_config
import t40a_lib as L
allp = {f"{p['episode']}_{p['seed']}": p for p in json.load(open(_ag3s_workspace_path('outputs/verify/T40/A/raw/carry_paths_22pairs.json')))["pairs"]}
out = {}
for f in sorted(glob.glob(_ag3s_workspace_path('outputs/verify/T40/A/configs/*.json'))):
    b = pathlib.Path(f).stem; ep, seed, kind = b.split("_"); tag = f"{ep}_{seed}"
    cfg = load_pick_place_obstacle_config(f); prof = [p for p in cfg["profiles"] if p != "clear"][0]
    R = L.load_run(allp[tag]["runs"]["E3b"]["run_dir"])
    d = mujoco.MjData(L.m); d.qpos[:] = R["q0"]; mujoco.mj_forward(L.m, d)
    mgr = PickPlaceObstacleManager(L.m, d, cfg); mgr.activate(prof)
    rec = {"manager_object_clearance_m": mgr.object_clearance(), "manager_robot_clearance_m": mgr.robot_clearance(), "pairs": []}
    ft = np.zeros(6)
    for o in mgr.collision_geom_ids:
        for g in mgr.payload_geom_ids:
            mj = float(mujoco.mj_geomDistance(L.m, d, int(o), int(g), 0.25, ft))
            if mj < 0.25:
                sa, sb = L.gjk.make_support(L.m, d, int(o)), L.gjk.make_support(L.m, d, int(g))
                gd = L.gjk.gjk_dist(sa, sb, d.geom_xpos[g] - d.geom_xpos[o] + 1e-9)
                rec["pairs"].append({"obstacle_geom": L.gname(int(o)), "payload_geom": L.gname(int(g)), "mj_geomDistance_m": mj, "gjk_m": gd})
    out[b] = rec
    print(b, round(rec["manager_object_clearance_m"], 4), round(rec["manager_robot_clearance_m"], 4),
          [(p["obstacle_geom"][-12:], p["payload_geom"], round(p["mj_geomDistance_m"], 4), round(p["gjk_m"], 4)) for p in rec["pairs"] if p["mj_geomDistance_m"] < 0.03 or p["gjk_m"] < 0.03])
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T40/A/dry/manager_vs_gjk.json'), "w"), indent=1)
