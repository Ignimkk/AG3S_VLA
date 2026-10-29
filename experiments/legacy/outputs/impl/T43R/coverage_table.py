"""T43 R — 덮임 지표 (bare vs 덮개 변형). CPU 만. 출력: outputs/impl/T43R/coverage_table.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, logging, time, mujoco, numpy as np
logging.disable(logging.WARNING)
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model, GRIPPER_LINKS
from AG3S.ag3s.experiments.sources import gripper_cover as gc
M = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m = mujoco.MjModel.from_xml_path(M); scene = TransportScene.attach(m, mujoco.MjData(m))
E3B = dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
variants = {"bare": None, "cover_2.5mm": True, "cover_3.0mm": {"max_gap": 0.003},
            "cover_left_only": {"hands": "left"},
            "cover_budget_250_50": {"max_spheres_by_link": {"ee_left": 250, "ee_right": 250, **{f: 50 for f in GRIPPER_LINKS[2:]}}}}
fv = [{"gripper_finger_l1": -x, "gripper_finger_l2": x, "gripper_finger_r1": -x, "gripper_finger_r2": x}
      for x in (0.0, 0.0125, 0.025, 0.0375, 0.05)]
out = {}
surf = None
for name, opt in variants.items():
    t = time.time()
    rob = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B, gripper_cover=opt)
    tb = time.time() - t
    if surf is None:
        surf = {L: gc.link_surface(m, rob, L) for L in GRIPPER_LINKS}
    met = gc.coverage_metrics(m, rob, GRIPPER_LINKS, finger_values=fv, surfaces=surf)
    rep = getattr(rob, "gripper_cover_report", None)
    if rep:
        for L in rep["links"]:
            rep["links"][L].pop("covered_after", None)
    out[name] = {"n_spheres": rob.n_spheres, "build_s": tb, "options": opt, "cover_report": rep, "metrics": met,
                 "gripper_openings_inner_gap_m": [g["inner_gap_m"] for g in rob.gripper_openings()]}
    print(name, rob.n_spheres, round(tb, 1), flush=True)
    for L, r in met.items():
        o = r["own"]
        print(f"  {L:13s} n {r['n_spheres']:4d} | verts inside {o['vertices']['frac_inside']:.3f} max {o['vertices']['max_gap_mm']:.2f} | "
              f"outer<=2.5 {o['outer']['frac_within_max_gap']:.4f} <=3 {np.nan} max {o['outer']['max_gap_mm']:.2f} p95 {o['outer']['p95_gap_mm']:.2f} | "
              f"grasp max {o['grasp_face'].get('max_gap_mm', float('nan')):.2f} | union max {[round(u['outer']['max_gap_mm'], 2) for u in r['gripper_union']]}", flush=True)
json.dump(out, open(_ag3s_workspace_path('outputs/impl/T43R/coverage_table.json'), "w"), indent=1, default=float)
