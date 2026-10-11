import os
import mujoco, numpy as np
XML="${AG3S_PROJECT_ROOT}-t44cl/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
m=mujoco.MjModel.from_xml_path(XML)
N=lambda t,i: mujoco.mj_id2name(m,t,i)
B=mujoco.mjtObj.mjOBJ_BODY
for i in range(m.nbody):
    n=N(B,i) or ""
    if "finger" in n or "EE" in n or "arm_6" in n:
        print(i,n,"parent",N(B,m.body_parentid[i]),"pos",np.round(m.body_pos[i],4).tolist(),"quat",np.round(m.body_quat[i],4).tolist(),
              "jnt",[(N(mujoco.mjtObj.mjOBJ_JOINT,j),int(m.jnt_type[j]),np.round(m.jnt_range[j],4).tolist(),np.round(m.jnt_axis[j],3).tolist(), int(m.jnt_qposadr[j])) for j in range(m.body_jntadr[i],m.body_jntadr[i]+m.body_jntnum[i])])
        for g in range(m.ngeom):
            if m.geom_bodyid[g]==i and (m.geom_contype[g]|m.geom_conaffinity[g]):
                print("   geom",g,"type",m.geom_type[g],"mesh",N(mujoco.mjtObj.mjOBJ_MESH,m.geom_dataid[g]) if m.geom_type[g]==7 else "", "size",np.round(m.geom_size[g],4).tolist(),"pos",np.round(m.geom_pos[g],4).tolist())
for a in range(m.nu):
    print("act",a,N(mujoco.mjtObj.mjOBJ_ACTUATOR,a), m.actuator_trntype[a], N(mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[a,0]) if m.actuator_trntype[a]==0 else m.actuator_trnid[a].tolist()) if "grip" in (N(mujoco.mjtObj.mjOBJ_ACTUATOR,a) or "") or "finger" in (N(mujoco.mjtObj.mjOBJ_ACTUATOR,a) or "") else None
print([ (k,N(mujoco.mjtObj.mjOBJ_EQUALITY,k), m.eq_type[k], m.eq_obj1id[k], m.eq_obj2id[k], m.eq_data[k][:5].tolist()) for k in range(m.neq)])
