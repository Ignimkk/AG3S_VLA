from common.workspace import workspace_path as _ag3s_workspace_path
import numpy as np, mujoco
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
from AG3S.trajopt.serve_safe import constraint_link_filter
from AG3S.ag3s.experiments.sources.gripper_cover import link_surface, finger_body_targets, _gap
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m = mujoco.MjModel.from_xml_path(XML); sc = TransportScene.attach(m, mujoco.MjData(m))
S = dict(capsule_radius_scale=0.05, sphere_spacing=0.4, max_spheres_per_capsule=32)
pc = build_constraint_robot_model(sc, link_filter=constraint_link_filter("gripper"), sphere_options=S, finger_cover={"hands": "both"}, palm_cover=True)
c, r = pc.local_spheres("ee_right")
for sp in (0.001, 0.0005):
    s = link_surface(m, pc, "ee_right", spacing=sp)
    t = s.points[finger_body_targets(s)]
    g = _gap(t, c, r); gv = _gap(s.vertices, c, r)
    print(sp, t.shape[0], "frac out", (g > 1e-9).mean(), "max gap mm", g.max()*1e3, "vertices out", (gv>1e-9).sum(), gv.shape[0], "max", gv.max()*1e3)
