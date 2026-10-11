"""T43 FR — build every finger_cover option and report per-finger cover geometry.

Per option: spheres per finger / total, radius range, uncovered outer-surface samples (count, max depth = distance to the
union surface), mesh vertices outside, sampled overflow beyond the mesh, inner-face x (closing side), T26 max opening,
build seconds, ilp source.  Also the two no-model rows from the prototype (min radius, containment).
usage: fr_build.py OUT_JSON
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, time, logging
import numpy as np, mujoco
logging.disable(logging.WARNING)
sys.path.insert(0, _ag3s_workspace_path('outputs/impl/T43FR'))
from fr_options import OPTIONS  # noqa: E402
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene  # noqa: E402
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, FINGER_LINKS, build_constraint_robot_model  # noqa: E402
from AG3S.ag3s.experiments.sources import gripper_cover as GC  # noqa: E402

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
E3B = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
m = mujoco.MjModel.from_xml_path(XML); scene = TransportScene.attach(m, mujoco.MjData(m))
bare = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B)
surf = {l: GC.link_surface(m, bare, l, spacing=0.0005) for l in FINGER_LINKS}


def opening(rob):
    return {o["parent_link"]: round(1e3 * float(o["inner_gap_m"]), 2) for o in rob.gripper_openings()}


out = {"bare": {"n_spheres": bare.n_spheres, "max_opening_mm": opening(bare)}}
print("bare", out["bare"], flush=True)
for name, opt in OPTIONS.items():
    t = time.time()
    rob = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B, finger_cover=dict(opt))
    dt = time.time() - t
    links = {}
    for l in FINGER_LINKS:
        C, R = (np.asarray(a) for a in rob.local_spheres(l))
        s = surf[l]
        tg = s.points[GC.finger_body_targets(s)]
        g = GC._gap(tg, C, R)
        gv = GC._gap(s.vertices, C, R)
        links[l] = {"n": int(R.size), "r_mm": [round(1e3 * R.min(), 2), round(1e3 * R.max(), 2)],
                    "n_targets": int(tg.shape[0]), "uncovered": int((g > 1e-9).sum()),
                    "max_uncovered_mm": round(1e3 * max(g.max(), 0.0), 3),
                    "uncovered_gt1mm": int((g > 1e-3).sum()), "vertices_outside": int((gv > 1e-9).sum()),
                    "overflow_mm": round(1e3 * GC._sampled_overflow(C, R, s.hulls, s.points), 3)}
    rep = rob.finger_cover_report["links"]["ee_finger_l1"]
    rec = {"options": opt, "n_spheres": rob.n_spheres, "per_finger": links["ee_finger_l1"]["n"], "links": links,
           "same_four": len({json.dumps(v, sort_keys=True) for v in links.values()}) == 1,
           "max_opening_mm": opening(rob), "build_sec": round(dt, 1),
           "source": rep.get("source"), "ilp": rep.get("ilp"), "greedy_n": rep.get("greedy_n_spheres")}
    out[name] = rec
    l1 = links["ee_finger_l1"]
    print(name, "total", rob.n_spheres, "finger", l1["n"], l1["r_mm"], "unc", l1["uncovered"], "max", l1["max_uncovered_mm"],
          "vert out", l1["vertices_outside"], "ovf", l1["overflow_mm"], "open", rec["max_opening_mm"], "same4", rec["same_four"],
          "build %.1fs" % dt, rec["source"] or "", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=str)
