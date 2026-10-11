"""T43 FR — solve the finger body ILP for preset (overflow, gap) pairs and print the presets module body.

usage: make_presets.py OUT_JSON TIME_LIMIT_S OVERFLOW_MM:GAP_MM [...]
One pair per process is fine (run in parallel, <= 4). Uses the module's own candidate grid / ILP (use_presets=False).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, logging
import numpy as np, mujoco
logging.disable(logging.WARNING)
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, build_constraint_robot_model
from AG3S.ag3s.experiments.sources import gripper_cover as GC

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
E3B = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
m = mujoco.MjModel.from_xml_path(XML); scene = TransportScene.attach(m, mujoco.MjData(m))
bare = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B)
surf = GC.link_surface(m, bare, "ee_finger_l1", spacing=GC.FINGER_COVER_SPACING)
target = surf.points[GC.finger_body_targets(surf)]
G, depth, ijk = GC._finger_candidates(surf, target, GC.FINGER_COVER_SPACING, GC.FINGER_COVER_UNION_EPS)
lookup = {tuple(np.round(g, 12)): tuple(int(v) for v in k) for g, k in zip(G, ijk)}
out = {}
for pair in sys.argv[3:]:
    ov, gp = (float(x) * 1e-3 for x in pair.split(":"))
    C, R, rep = GC.finger_body_spheres(surf, overflow=ov, gap=gp, solver="ilp", use_presets=False,
                                       ilp_time_limit=float(sys.argv[2]))
    cells = sorted(lookup[tuple(np.round(c, 12))] for c in C)
    info = rep.get("ilp", {})
    key = f"{int(round(ov * 1e6))}:{int(round(gp * 1e6))}"
    out[key] = {"ijk": cells, "n": int(R.size), "greedy_n": rep["greedy_n_spheres"], "optimal": bool(info.get("optimal")),
                "dual_bound": info.get("dual_bound"), "sec": info.get("sec"), "source": rep.get("source"),
                "n_uncovered": rep["n_uncovered"], "max_gap_mm": rep["max_gap_mm"],
                "overflow_sampled_mm": rep["overflow_sampled_mm"]}
    print(key, {k: v for k, v in out[key].items() if k != "ijk"}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
