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
t=time.time(); cov=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts, gripper_cover=True); print("build",time.time()-t, cov.n_spheres)
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
def rows(model, z, policy="exclude_authorized"):
    names=np.array(model.sphere_link_names); radii=model.radii
    C=centres_for(model,z)[None]
    field=build_field(z,outside_distance=1.0)
    auth=np.isin(names,AUTH)
    lin=object.__new__(CollisionLinearizer); lin.robot_radii=radii; lin.query_radii=radii; lin.n_spheres=radii.size; lin._attached_local=None
    sc=SceneSnapshot(candidate_pos=np.zeros((0,3)),candidate_radius=np.zeros(0),candidate_active=np.zeros(0,bool),d_safe=np.zeros((radii.size,0)),
        plane_normal=np.zeros((0,3)),plane_offset=np.zeros(0),plane_active=np.zeros(0,bool),robot_radii=radii,candidate_ids=np.zeros(0,np.int64),
        esdf=field,esdf_margin=0.0,manipulated_points=z["target_points"].astype(float),manipulated_link_margin=np.where(auth,0.02,0.05),
        target_free_mask=(auth if (policy=="exclude_authorized" and "esdf_free_distance" in z.files) else None), obstacle_margin=1e-9)
    cl=lin._esdf_clearance(C,sc)[0,:,0]
    cls=lin._last_obstacle_class[0] if hasattr(lin,"_last_obstacle_class") else None
    return names,cl,cls
for k in [int(a) for a in sys.argv[1:]]:
    z=np.load(f"{R}/chunk_{k:05d}.npz",allow_pickle=True)
    s=json.loads(str(z["summary_json"]))
    rec=z["clearance"][0]
    nb,cb,clb=rows(bare,z); nc,cc,clc=rows(cov,z)
    print(f"chunk {k} t_step {s['t_step']} has_target {s['has_target']} attached {s['esdf_stats'].get('n_attached_points') if isinstance(s.get('esdf_stats'),dict) else None}")
    for L in AUTH:
        sb=nb==L; sc_=nc==L; new=sc_.copy(); new[:bare.n_spheres]=False
        print(f"   {L}: bare min {cb[sb].min()*1e3:7.2f}  cover-new min {cc[new].min()*1e3:7.2f}  n_viol bare {(cb[sb]<0).sum()} cover {(cc[sc_]<0).sum()}  classes new-viol {np.bincount(clc[new & (cc<0)],minlength=4) if clc is not None else ''}")
    print("   base rows identical:", np.array_equal(cb, cc[:bare.n_spheres]))
