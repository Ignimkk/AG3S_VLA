from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, numpy as np, mujoco, glob
sys.path.insert(0,_ag3s_workspace_path('integration/rby1/manipulation/src')); sys.path.insert(0,_ag3s_workspace_path(''))
from rby1_manipulation.simulation import pick_place_obstacles as PPO
from tests.sim import _gjk as gjk
XML=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml')
m=mujoco.MjModel.from_xml_path(XML); d=mujoco.MjData(m)
cases={"E3b_C_E3b_ep1995_s19952":("t40f_ep1995_s19952_r1","foot_left",856),"E3b_D_E3b_ep1967_s19672":("t40f_ep1967_s19672_r2","foot_right",1085),
"E3b_D_E3b_ep1982_s19822":("t40f_ep1982_s19822_r2","foot_right",1085),"E3b_D_E3b_ep1995_s19952":("t40f_ep1995_s19952_r2","foot_right",904),"E0_D_E0_ep1967_s19672":("t40f_ep1967_s19672_r2","foot_right",1086)}
for k,(prof,foot,g) in cases.items():
    cfg=PPO.load_pick_place_obstacle_config(f"{_ag3s_workspace_path()}/outputs/verify/T40/F/configs/{prof}.json"); mg=PPO.PickPlaceObstacleManager(m,d,cfg); mg.activate(prof)
    z=np.load(f"{_ag3s_workspace_path()}/outputs/verify/T40/H/stop_{k}.npz"); d.qpos[:]=z["qpos_kin"]; mujoco.mj_kinematics(m,d); print("xpos bit-equal", np.array_equal(d.geom_xpos, z["geom_xpos"]))
    o=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,f"pick_obstacle_hurdle_0_collision_{foot}")
    ft=np.zeros(6); ref=gjk.gjk_dist(gjk.make_support(m,d,o),gjk.make_support(m,d,g),d.geom_xpos[g]-d.geom_xpos[o]+1e-9)
    a=mujoco.mj_geomDistance(m,d,o,g,0.25,ft); w=ft.copy()
    b=mujoco.mj_geomDistance(m,d,g,o,0.25,ft)
    m.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    c=mujoco.mj_geomDistance(m,d,o,g,0.25,ft); m.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    mid=m.geom_dataid[g]; R=d.geom_xmat[o].reshape(3,3)
    print(k, "gjk %.6f mj %.6f swapped %.6f libccd %.6f"%(ref,a,b,c), "cyl axis z", R[:,2].round(4), "size", m.geom_size[o][:2], "mesh nvert", m.mesh_vertnum[mid],
          "from-foot", (w[:3]-d.geom_xpos[o]).round(4), "to/2-from/2 gap", np.linalg.norm(w[3:]/2-w[:3]/2).round(5))
    for dm in (0.05, ref*0.99, ref*1.01, 0.3, 1.0): print("    distmax %.4f -> %.6f"%(dm, mujoco.mj_geomDistance(m,d,o,g,dm,ft)))
    print("    ccd_iterations", m.opt.ccd_iterations, "ccd_tolerance", m.opt.ccd_tolerance)
