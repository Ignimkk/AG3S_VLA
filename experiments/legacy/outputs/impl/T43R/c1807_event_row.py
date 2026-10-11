from common.workspace import workspace_path as _ag3s_workspace_path
import time, json, logging, mujoco, numpy as np
logging.disable(logging.WARNING)
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model, GRIPPER_LINKS
M=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m=mujoco.MjModel.from_xml_path(M); d=mujoco.MjData(m); scene=TransportScene.attach(m,d)
opts=dict(sphere_spacing=0.4,max_spheres_per_capsule=32,capsule_radius_scale=0.05)
bare=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts)
cov=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts, gripper_cover=True)
case=json.load(open(_ag3s_workspace_path('outputs/verify/T40/I/raw/case_C1807.json')))
ev=case['event']; sw=ev['spheres_world_at_event_row']
W=np.array(sw['centres']); links=np.array(sw['links']); radii=np.array(sw['radii'])
assert list(links)==list(bare.sphere_link_names), "order"
print("radii match", np.abs(radii-bare.radii).max())
def kabsch(A,B):
    ca,cb=A.mean(0),B.mean(0); H=(A-ca).T@(B-cb); U,S,Vt=np.linalg.svd(H); D=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]); R=Vt.T@D@U.T; return R, cb-R@ca
bar=case['bar_geom']; bc=np.array(bar['centre_world']); ba=np.array(bar['axis_world']); ba/=np.linalg.norm(ba); bh=bar['half_length_mm']/1e3; br=bar['radius_mm']/1e3
def dbar(P):
    t=np.clip((P-bc)@ba,-bh,bh); return np.linalg.norm(P-(bc+t[:,None]*ba),axis=1)-br
names_c=np.array(cov.sphere_link_names)
for L in ["ee_left","ee_finger_l1","ee_finger_l2"]:
    c0,r0=bare.local_spheres(L); sel=links==L
    R,t=kabsch(c0,W[sel]); res=np.abs(c0@R.T+t-W[sel]).max()
    cc,rc=cov.local_spheres(L); Pc=cc@R.T+t
    nb=len(r0)
    db=dbar(W[sel])-r0; dc=dbar(Pc)-rc
    print(L,"kabsch resid mm",round(res*1e3,4),"bare min",round(db.min()*1e3,3),"cover(all) min",round(dc.min()*1e3,3),"new-only min",round(dc[nb:].min()*1e3,3),"mesh",round(ev['per_link_min_geom_to_bar_mm_at_row'][L],3))
