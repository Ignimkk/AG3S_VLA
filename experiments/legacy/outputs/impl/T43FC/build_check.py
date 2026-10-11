"""T43 FC — build the constraint model with --finger-cover, report per-finger cover (overflow sweep)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, time, numpy as np, mujoco
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, FINGER_LINKS, build_constraint_robot_model
MODEL=_ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
m=mujoco.MjModel.from_xml_path(MODEL); scene=TransportScene.attach(m, mujoco.MjData(m))
E3B=dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)
bare=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B)
out={"bare":{"n_spheres":bare.n_spheres,"openings":bare.gripper_openings()}}
for ov in [float(x) for x in (sys.argv[1:] or ["2.5"])]:
    t=time.time()
    rob=build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=E3B, finger_cover={"overflow":ov*1e-3})
    dt=time.time()-t
    rep=rob.finger_cover_report
    same_palm=all(np.array_equal(bare.local_spheres(l)[0],rob.local_spheres(l)[0]) and np.array_equal(bare.local_spheres(l)[1],rob.local_spheres(l)[1]) for l in ("ee_left","ee_right"))
    out[f"overflow_{ov}"]={"build_sec":dt,"n_spheres":rob.n_spheres,"palm_identical":same_palm,"openings":rob.gripper_openings(),
        "links":{k:v for k,v in rep["links"].items()},"coverage_shortfall":[s.link for s in rob.coverage_shortfall]}
    l1=rep["links"]["ee_finger_l1"]
    print(ov, "n",rob.n_spheres,"build %.1fs"%dt, "finger", l1["n_spheres"], l1["radius_mm"], "unc",l1["n_uncovered"],"/",l1["n_targets"], "gap",l1["max_gap_mm"],"ovf",round(l1["overflow_sampled_mm"],3), l1["spheres_bbox_mm"], "palm same",same_palm, flush=True)
    print("  openings", [(o.get("parent_link"), round(o.get("max_opening",o.get("opening",0))*1e3 if isinstance(o.get("max_opening",0),float) else 0,2)) for o in rob.gripper_openings()])
json.dump(out,open(_ag3s_workspace_path('outputs/impl/T43FC/build_check.json'),"w"),indent=1,default=str)
