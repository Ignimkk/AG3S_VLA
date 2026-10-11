from common.workspace import workspace_path as _ag3s_workspace_path
import os, sys, json, time, pathlib
os.environ["T43W_SNAP"]="${AG3S_PROJECT_ROOT}-t43w3"
sys.path.insert(0,_ag3s_workspace_path('outputs/verify/T43/W/p'))
import numpy as np, mujoco
import hold_w as HW
rd=pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/W/runs/V5/H/V5_ep1925_s19251'))
man=json.load(open(rd/"manifest.json")); prof=man["scene"]["obstacle_profile"]; ep=man["scene"]["episode_index"]
seed=int(np.load(rd/"traj.npz")["policy_seed"])
cfg=next((l.split("=",1)[1].split(":",1)[0] for l in (rd.parent/f"{rd.name}.start").read_text().splitlines() if l.startswith("obstacle=") and l!="obstacle=:"),None)
print("cfg",cfg)
t=time.time(); cl=HW.load_client(); L=HW.reset(cl,ep,seed,prof,cfg); print("reset %.1fs"%(time.time()-t))
m,d,mg=L["m"],L["d"],L.get("obstacle_manager")
print("nq",m.nq, "obst", mg.collision_geom_ids)
for g in mg.collision_geom_ids:
    b=m.geom_bodyid[g]; print(g, HW.gname(m,g), "type",m.geom_type[g],"size",m.geom_size[g],"body",mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b),"mocap",m.body_mocapid[b],"jnt",m.body_jntnum[b], "xpos",d.geom_xpos[g])
rows=[json.loads(l) for l in open(rd/"frames.jsonl") if l.strip()]; rows=[r for r in rows if r["kind"]=="control"]
d.qpos[:]=rows[-1]["qpos"]; mujoco.mj_kinematics(m,d)
for g in mg.collision_geom_ids: print("after qpos", g, d.geom_xpos[g])
