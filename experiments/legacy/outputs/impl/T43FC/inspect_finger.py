"""T43 FC — finger collision mesh vs gap-filling capsules (link frame)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import numpy as np, mujoco
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene, gap_filling_capsules, MJCF_BODY_ALIASES, static_urdf_ancestor
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, FINGER_LINKS, build_constraint_robot_model
from AG3S.ag3s.experiments.sources.gripper_cover import link_surface, finger_inward, grasp_limits
MODEL=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m=mujoco.MjModel.from_xml_path(MODEL); scene=TransportScene.attach(m, mujoco.MjData(m))
E3B=dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
rob=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B)
print("n_spheres", rob.n_spheres)
for cap in gap_filling_capsules(m):
    if cap.link in FINGER_LINKS:
        print(cap.link, "r=%.2f mm L=%.2f mm"%(cap.radius*1e3, cap.length*1e3), "p", np.round(cap.origin[:3,3]*1e3,1), "axis", np.round(cap.origin[:3,2],3))
for link in FINGER_LINKS:
    s=link_surface(m, rob, link)
    body=MJCF_BODY_ALIASES.get(link,link)
    bid=mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    geoms=[g for g in range(m.ngeom) if m.geom_bodyid[g]==bid]
    print(link, body, "geoms", [(g, int(m.geom_type[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in geoms])
    v=s.vertices; print("  coll verts", v.shape, "min", np.round(v.min(0)*1e3,1), "max", np.round(v.max(0)*1e3,1), "pieces", len(s.hulls))
    print("  surf pts", s.points.shape[0], "outer", int(s.outer.sum()), "grasp", int(s.grasp_face.sum()), "hidden", int(s.hidden.sum()))
    c,r=rob.local_spheres(link); print("  spheres", r.size, "r %.2f-%.2f mm"%(r.min()*1e3, r.max()*1e3))
    print("  u_in", np.round(finger_inward(rob, link),3), "inner", grasp_limits(rob, link)[0])
