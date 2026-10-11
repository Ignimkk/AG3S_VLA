from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np, mujoco
S=_ag3s_workspace_path('outputs/verify/T40/H/')
sys.path.insert(0,_ag3s_workspace_path('integration/rby1/manipulation/src')); sys.path.insert(0,_ag3s_workspace_path(''))
from rby1_manipulation.simulation import pick_place_obstacles as PPO
from tests.sim import _gjk as gjk
XML=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml')
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
runs={"E3b/C/E3b_ep1995_s19952":("t40f_ep1995_s19952_r1",1995,19952),"E3b/D/E3b_ep1967_s19672":("t40f_ep1967_s19672_r2",1967,19672),
"E3b/D/E3b_ep1982_s19822":("t40f_ep1982_s19822_r2",1982,19822),"E3b/D/E3b_ep1995_s19952":("t40f_ep1995_s19952_r2",1995,19952),"E0/D/E0_ep1967_s19672":("t40f_ep1967_s19672_r2",1967,19672)}
nm=lambda g: mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g)
bn=lambda g: mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))
out={"_comment":"T40 H: the 5 T40 G obstacle stops with client clearance exactly 0.0 and no contact, reproduced bit-exactly by replaying "
 "pi05_infer main() reset (--warmup-steps 1000) + the recorded applied_ctrl (frames.jsonl) sub-step by sub-step. qpos_kin = d.qpos before the "
 "mj_step whose stop check fired (geom_xpos after mj_step belong to it): mj_kinematics(qpos_kin) gives the stop check's geom_xpos bit for bit. "
 "config = outputs/verify/T40/F/configs/<profile>.json. Source: implementer T40 H, 2026-10-05.", "runs":{}}
for run,(prof,ep,seed) in runs.items():
    k=run.replace("/","_"); z=np.load(S+f"stop_{k}.npz")
    cfgp=f"{_ag3s_workspace_path()}/outputs/verify/T40/F/configs/{prof}.json"; cfg=json.load(open(cfgp))
    mg=PPO.PickPlaceObstacleManager(m,d,PPO.load_pick_place_obstacle_config(cfgp)); mg.activate(prof)
    d.qpos[:]=z["qpos_kin"]; mujoco.mj_kinematics(m,d); assert np.array_equal(d.geom_xpos,z["geom_xpos"])
    pairs=[]
    for o in sorted(mg.collision_geom_ids):
        for g in mg.robot_geom_ids:
            v=mujoco.mj_geomDistance(m,d,o,g,0.25,np.zeros(6))
            if v<=0: pairs.append({"obstacle_geom":nm(o),"robot_geom_id":int(g),"robot_geom_name":nm(g),"robot_body":bn(g),"mj_geomDistance":v,
                  "gjk":gjk.gjk_dist(gjk.make_support(m,d,o),gjk.make_support(m,d,g),d.geom_xpos[g]-d.geom_xpos[o]+1e-9)})
    out["runs"][run]={"profile":prof,"episode":ep,"policy_seed":seed,"stop_t_step":int(z["t"]),"stop_substep":int(z["s"]),
        "config":cfg,"qpos_kin":[float(x) for x in z["qpos_kin"]],"false_zero_pairs":pairs}
    print(run, int(z["t"]), int(z["s"]), pairs)
json.dump(out,open(_ag3s_workspace_path('tests/sim/data/t40h_stop_states.json'),"w"),indent=1)
