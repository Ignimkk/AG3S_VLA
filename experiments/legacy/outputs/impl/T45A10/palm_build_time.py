"""T45 A10 — 손바닥 몸통 구 풀이 시간 (CPU, 프로세스당 1 회; 왼손은 캐시)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import time, mujoco
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
from AG3S.ag3s.experiments.sources.gripper_cover import palm_body_capsules
from AG3S.trajopt.serve_safe import constraint_link_filter
XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m = mujoco.MjModel.from_xml_path(XML); sc = TransportScene.attach(m, mujoco.MjData(m))
S = dict(capsule_radius_scale=0.05, sphere_spacing=0.4, max_spheres_per_capsule=32)
t = time.perf_counter(); base = build_constraint_robot_model(sc, link_filter=constraint_link_filter("gripper"), sphere_options=S, finger_cover={"hands": "both"}); t1 = time.perf_counter()
caps, rep = palm_body_capsules(m, base, ["ee_left", "ee_right"]); t2 = time.perf_counter()
pc = build_constraint_robot_model(sc, link_filter=constraint_link_filter("gripper"), sphere_options=S, finger_cover={"hands": "both"}, palm_cover=True); t3 = time.perf_counter()
print(f"FC model {t1-t:.1f}s · palm cover solve (both palms, cold) {t2-t1:.1f}s · full FC+PC model (warm cache) {t3-t2:.1f}s · n {pc.n_spheres}")
