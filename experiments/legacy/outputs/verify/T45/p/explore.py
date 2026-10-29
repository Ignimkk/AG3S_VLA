import os,sys
os.environ.setdefault("MUJOCO_GL","osmesa")
import mujoco, numpy as np
XML="${AG3S_PROJECT_ROOT}-t44cl/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
m=mujoco.MjModel.from_xml_path(XML)
def name(t,i): return mujoco.mj_id2name(m,t,i)
for bn in ["EE_BODY_R","EE_BODY_L"]:
    b=m.body(bn).id
    sub=[i for i in range(m.nbody) if (lambda j: any(True for _ in [0]))(i)]
    # subtree
    ids=[]
    for i in range(m.nbody):
        j=i
        while j!=0 and j!=b: j=m.body_parentid[j]
        if j==b: ids.append(i)
    for i in ids:
        print(bn, i, name(mujoco.mjtObj.mjOBJ_BODY,i), "parent", name(mujoco.mjtObj.mjOBJ_BODY,m.body_parentid[i]), "jnt", [ (name(mujoco.mjtObj.mjOBJ_JOINT,j), m.jnt_type[j], m.jnt_range[j].tolist()) for j in range(m.body_jntadr[i], m.body_jntadr[i]+m.body_jntnum[i])])
        for g in range(m.ngeom):
            if m.geom_bodyid[g]==i:
                coll = (m.geom_contype[g]|m.geom_conaffinity[g])!=0
                gt=m.geom_type[g]
                mesh = name(mujoco.mjtObj.mjOBJ_MESH, m.geom_dataid[g]) if gt==mujoco.mjtGeom.mjGEOM_MESH else ""
                print("   geom",g,name(mujoco.mjtObj.mjOBJ_GEOM,g),"type",gt,"coll",coll,"size",np.round(m.geom_size[g],4).tolist(),mesh, "group", m.geom_group[g], "margin", m.geom_margin[g])
print(m.nq, m.opt.timestep)
