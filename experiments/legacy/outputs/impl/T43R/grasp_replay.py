from common.workspace import workspace_path as _ag3s_workspace_path
import time, json, logging, mujoco, numpy as np, sys
logging.disable(logging.WARNING)
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model, GRIPPER_LINKS
from AG3S.trajopt.linearize import CollisionLinearizer, SceneSnapshot, OBSTACLE_CLASSES
from AG3S.trajopt.experiments.fixed_input_ablation import build_field
M=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m=mujoco.MjModel.from_xml_path(M); d=mujoco.MjData(m); scene=TransportScene.attach(m,d)
opts=dict(sphere_spacing=0.4,max_spheres_per_capsule=32,capsule_radius_scale=0.05)
bare=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts)
cov=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts, gripper_cover=True)
R=_ag3s_workspace_path('outputs/verify/T40/G/runs/E3b/server_8240_20261003T135313/server_constraints/run_0000')
AUTH=("ee_left","ee_finger_l1","ee_finger_l2")
def kabsch(A,B):
    ca,cb=A.mean(0),B.mean(0); H=(A-ca).T@(B-cb); U,S,Vt=np.linalg.svd(H); D=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]); Rm=Vt.T@D@U.T; return Rm, cb-Rm@ca
def centres_for(model, z):
    names=np.array(json.loads(str(z["summary_json"]))["sphere_link_names"]); W=z["sphere_centres"].astype(float)
    out=np.zeros((model.n_spheres,3)); mn=np.array(model.sphere_link_names)
    for L in sorted(set(mn)):
        c0,_=bare.local_spheres(L); Rm,t=kabsch(c0,W[names==L]); cl,_=model.local_spheres(L); out[mn==L]=cl@Rm.T+t
    return out
def planes(z):
    cloud=z["filtered_cloud"].astype(np.float64); sup=cloud[z["support_mask"]]; sup=sup[sup[:,2]>0.4]
    mu=sup.mean(0); n=np.linalg.svd(sup-mu,full_matrices=False)[2][2]; n=n*np.sign(n[2]); return np.array([[*n,float(n@mu)]])
def rows(model, z, policy):
    names=np.array(model.sphere_link_names); radii=model.radii
    C=centres_for(model,z)[None]; field=build_field(z,outside_distance=1.0); auth=np.isin(names,AUTH)
    lin=object.__new__(CollisionLinearizer); lin.robot_radii=radii; lin.query_radii=radii; lin.n_spheres=radii.size; lin._attached_local=None
    sc=SceneSnapshot(candidate_pos=np.zeros((0,3)),candidate_radius=np.zeros(0),candidate_active=np.zeros(0,bool),d_safe=np.zeros((radii.size,0)),
        plane_normal=np.zeros((0,3)),plane_offset=np.zeros(0),plane_active=np.zeros(0,bool),robot_radii=radii,candidate_ids=np.zeros(0,np.int64),
        esdf=field,esdf_margin=0.0,manipulated_points=z["target_points"].astype(float),manipulated_link_margin=np.where(auth,0.02,0.05),
        target_free_mask=(auth if (policy=="exclude_authorized" and "esdf_free_distance" in z.files) else None), obstacle_margin=1e-9, support_planes=planes(z))
    cl=lin._esdf_clearance(C,sc)[0,:,0]; cls=lin._last_obstacle_class[0]
    return names,cl,cls
nb=bare.n_spheres
for policy in ("exclude_authorized","relax"):
  tot=dict(base_t=0,new_t=0,new_s=0,new_o=0)
  for k in list(range(8,32))+list(range(138,162)):
    z=np.load(f"{R}/chunk_{k:05d}.npz",allow_pickle=True); s=json.loads(str(z["summary_json"]))
    n1,c1,k1=rows(bare,z,policy); n2,c2,k2=rows(cov,z,policy)
    assert np.allclose(c1,c2[:nb]) , "base differs"
    new=np.zeros(len(c2),bool); new[nb:]=True
    left=np.isin(n2,AUTH)
    def cnt(cl,cls,sel,c): return int(((cl<0)&(cls==c)&sel).sum())
    bt=cnt(c1,k1,np.ones(nb,bool),1); nt=cnt(c2,k2,new,1); ns=cnt(c2,k2,new,2); no=cnt(c2,k2,new,0)
    tot['base_t']+=bt; tot['new_t']+=nt; tot['new_s']+=ns; tot['new_o']+=no
    mins=lambda cl,cls,sel,c: (round(float(cl[sel&(cls==c)].min()*1e3),1) if (sel&(cls==c)).any() else None)
    print(f"{policy[:5]} chunk {k} t {s['t_step']} grasp {(s.get('grasp') or {}).get('state')}: target-viol base {bt} new {nt} | min target base {mins(c1,k1,np.ones(nb,bool)&left[:nb],1)} new {mins(c2,k2,new&left,1)} | min support base {mins(c1,k1,left[:nb],2)} new {mins(c2,k2,new&left,2)} | new viol support {ns} obstacle {no}")
  print(policy, tot)
