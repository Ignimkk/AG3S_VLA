"""T43 FC prototype — delta-inflated inscribed spheres covering the whole finger collision mesh."""
from common.workspace import workspace_path as _ag3s_workspace_path
import dataclasses, json, time, numpy as np, mujoco
from scipy.spatial import cKDTree
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, build_constraint_robot_model
from AG3S.ag3s.experiments.sources.gripper_cover import link_surface, hull_depth, outer_cover_spheres, _gap, body_collision_surface
MODEL=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m=mujoco.MjModel.from_xml_path(MODEL); scene=TransportScene.attach(m, mujoco.MjData(m))
rob=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05))
link="ee_finger_l1"
s=link_surface(m, rob, link)
full=dataclasses.replace(s, grasp_face=np.zeros_like(s.grasp_face), hidden=np.zeros_like(s.hidden))
fine=link_surface(m, rob, link, spacing=0.0005)
tree=cKDTree(fine.points)
def overflow(C,R,n=400):
    rng=np.random.default_rng(0); out=0.0
    for c,r in zip(C,R):
        v=rng.normal(size=(n,3)); v/=np.linalg.norm(v,axis=1,keepdims=True)
        P=c+r*v; d=hull_depth(P,s.hulls); o=P[d<0]
        if o.size: out=max(out, float(tree.query(o)[0].max()))
    return out
c0,r0=rob.local_spheres(link)
base={"n":int(r0.size),"gap_max_mm":float(_gap(fine.points,c0,r0).max()*1e3),"frac_in":float(np.mean(_gap(fine.points,c0,r0)<=0)),"overflow_mm":overflow(c0,r0)}
print("current", base, flush=True)
rows=[]
dmax=float(hull_depth(s.points - 0*s.points, s.hulls).max())
for delta in (0.0010,0.0015,0.0020,0.0025,0.0030):
    t=time.time()
    fr=np.array([0.20,0.35,0.5,0.65,0.8,0.9,0.985])
    C,R,rep=outer_cover_spheres(full,(np.zeros((0,3)),np.zeros(0)),r_max=0.0048,max_gap=delta,fractions=fr)
    Ri=R+delta
    g=_gap(fine.points,C,Ri)
    row={"delta_mm":delta*1e3,"n":int(R.size),"r_mm":[float(Ri.min()*1e3),float(Ri.max()*1e3)],"uncovered_1mm":rep["n_uncovered"],"n_targets":rep["n_targets"],
         "fine_frac_in":float(np.mean(g<=0)),"fine_gap_max_mm":float(max(g.max(),0)*1e3),"overflow_sampled_mm":overflow(C,Ri),
         "inner_reach_x_mm":float((C[:,0]-Ri).min()*1e3),"outer_reach_x_mm":float((C[:,0]+Ri).max()*1e3),"tip_z_mm":float((C[:,2]-Ri).min()*1e3),"y_mm":float((np.abs(C[:,1])+Ri).max()*1e3),"sec":time.time()-t}
    print(row, flush=True); rows.append(row)
json.dump({"current":base,"rows":rows,"mesh":{"x":[float(s.vertices[:,0].min()*1e3),float(s.vertices[:,0].max()*1e3)],"z_min":float(s.vertices[:,2].min()*1e3),"y":float(np.abs(s.vertices[:,1]).max()*1e3)}},open(_ag3s_workspace_path('outputs/impl/T43FC/proto_cover.json'),"w"),indent=1)
