from common.workspace import workspace_path as _ag3s_workspace_path
import time, json, logging, mujoco, numpy as np
logging.disable(logging.WARNING)
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, gap_filling_capsules, HEAD_JOINTS
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS
from AG3S.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf, DEFAULT_RBY1_JOINTS
from AG3S.ag3s.robot_models.urdf_sphere_chain import UrdfCapsule, EXACT_SPHERE_ROLE
from AG3S.trajopt.linearize import CollisionLinearizer, SceneSnapshot
from AG3S.trajopt.types import ChunkLayout
from AG3S.trajopt.config import TrajOptConfig
from AG3S.trajopt.experiments.fixed_input_ablation import build_field
M=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m=mujoco.MjModel.from_xml_path(M); d=mujoco.MjData(m)
urdf=parse_urdf(RBY1_URDF); gap=gap_filling_capsules(m)
opts=dict(sphere_spacing=0.4,max_spheres_per_capsule=32,capsule_radius_scale=0.05)
R=_ag3s_workspace_path('outputs/verify/T40/G/runs/E3b/server_8240_20261003T135313/server_constraints/run_0000')
z=np.load(R+'/chunk_00130.npz',allow_pickle=True)
field=build_field(z,outside_distance=1.0)
rng=np.random.default_rng(0)
layout=ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
cfg=TrajOptConfig()
q=np.zeros((layout.nq_opt,8)); qn=np.zeros(20)
for N in (0,300,900,1800):
    extra=[]
    for i in range(N):
        L=GRIPPER_LINKS[i%6]; o=np.eye(4); o[:3,3]=rng.normal(0,0.02,3)
        extra.append(UrdfCapsule(link=L,origin=o,radius=0.004,length=0.0,role=EXACT_SPHERE_ROLE))
    t=time.time(); rob=UrdfSphereChain(urdf,extra_capsules=list(gap)+extra,link_filter=GRIPPER_LINKS,**opts); tb=time.time()-t
    t=time.time(); lin=CollisionLinearizer(rob,layout,8); tl=time.time()-t
    radii=lin.robot_radii
    scene=SceneSnapshot(candidate_pos=np.zeros((0,3)),candidate_radius=np.zeros(0),candidate_active=np.zeros(0,bool),d_safe=np.zeros((radii.size,0)),
        plane_normal=np.zeros((0,3)),plane_offset=np.zeros(0),plane_active=np.zeros(0,bool),robot_radii=radii,candidate_ids=np.zeros(0,np.int64),
        esdf=field,esdf_margin=0.0,manipulated_points=None,manipulated_link_margin=None,target_free_mask=None)
    for _ in range(2): lin.linearize(q, qn, scene, cfg)
    ts=[]; 
    for _ in range(10):
        t=time.perf_counter(); st=lin.sphere_states(q,qn); t1=time.perf_counter(); lin._esdf_clearance(st[0],scene); t2=time.perf_counter(); lin.linearize(q,qn,scene,cfg,states=st); t3=time.perf_counter()
        ts.append((t1-t,t2-t1,t3-t2))
    ts=np.median(np.array(ts),0)*1e3
    print(f"N_extra {N} spheres {rob.n_spheres}: build model {tb:.2f}s linearizer init {tl:.2f}s | fk_jac {ts[0]:.2f} ms esdf {ts[1]:.2f} ms linearize(states given) {ts[2]:.2f} ms", flush=True)
