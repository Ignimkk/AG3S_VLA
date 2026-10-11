from common.workspace import workspace_path as _ag3s_workspace_path
import numpy as np, mujoco
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, build_constraint_robot_model
from AG3S.ag3s.experiments.sources.gripper_cover import link_surface, hull_depth
MODEL=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m=mujoco.MjModel.from_xml_path(MODEL); scene=TransportScene.attach(m, mujoco.MjData(m))
rob=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05))
s=link_surface(m, rob, "ee_finger_l1")
from AG3S.ag3s.experiments.sources.gripper_cover import _geom_pieces
for i,(A,b) in enumerate(s.hulls):
    # piece vertices: recover via surface points with depth ~0 in this piece
    d=-np.max(s.points@A.T+b,axis=1)
    on=np.abs(d)<1e-4
    P=s.points[on] if on.any() else None
    # interior max depth by grid
    lo=s.vertices.min(0); hi=s.vertices.max(0)
    g=np.stack(np.meshgrid(*[np.arange(l,h,0.0005) for l,h in zip(lo,hi)],indexing='ij'),-1).reshape(-1,3)
    dd=-np.max(g@A.T+b,axis=1); inside=dd>0
    print(i, "faces",A.shape[0], "bbox", np.round(g[inside].min(0)*1e3,1), np.round(g[inside].max(0)*1e3,1), "max inscribed r %.2f mm"%(dd.max()*1e3), "at", np.round(g[dd.argmax()]*1e3,1))
# thickness profile along z: x extent of union at y=0
g=np.stack(np.meshgrid(np.arange(-0.005,0.016,0.0002),[0.0],np.arange(-0.062,0.003,0.002),indexing='ij'),-1).reshape(-1,3)
dep=hull_depth(g,s.hulls)
for z in np.unique(np.round(g[:,2],4)):
    sel=(np.abs(g[:,2]-z)<1e-6)&(dep>0)
    if sel.any(): print("z %.0f mm x[%.1f, %.1f] maxdepth %.2f"%(z*1e3, g[sel,0].min()*1e3, g[sel,0].max()*1e3, dep[sel].max()*1e3))
# y extent
g=np.stack(np.meshgrid([0.005],np.arange(-0.018,0.018,0.0002),np.arange(-0.062,0.003,0.004),indexing='ij'),-1).reshape(-1,3)
dep=hull_depth(g,s.hulls)
for z in np.unique(np.round(g[:,2],4)):
    sel=(np.abs(g[:,2]-z)<1e-6)&(dep>0)
    if sel.any(): print("z %.0f mm y[%.1f, %.1f]"%(z*1e3, g[sel,1].min()*1e3, g[sel,1].max()*1e3))
