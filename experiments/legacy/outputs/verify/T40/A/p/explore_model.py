from common.workspace import workspace_path as _ag3s_workspace_path
import mujoco, numpy as np
X=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport_pick_place_obstacles.xml')
m=mujoco.MjModel.from_xml_path(X); d=mujoco.MjData(m)
print("nq",m.nq,"ngeom",m.ngeom,"nbody",m.nbody)
def path(b):
    s=[]
    while b>0:
        s.append(mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,b)); b=m.body_parentid[b]
    return s
for g in range(m.ngeom):
    if m.geom_contype[g] or m.geom_conaffinity[g]:
        b=m.geom_bodyid[g]
        print(g, mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g), m.geom_type[g], "body", path(b)[:3], "ct/ca",m.geom_contype[g],m.geom_conaffinity[g], np.round(m.geom_size[g],4))
for j in range(m.njnt):
    print("jnt",j,mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j),m.jnt_qposadr[j])
for c in range(m.ncam): print("cam",mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_CAMERA,c))
for s in range(m.nsite): print("site",mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,s))
