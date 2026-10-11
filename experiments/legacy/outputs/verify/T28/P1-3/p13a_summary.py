"""Compact P1-3a numbers -> p13a_summary.json (+ within-capsule sphere gaps, sphere-model finger inner gap)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, build_constraint_robot_model
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-3'))
from p13a_spheres import CONFIGS, XML
D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/P1-3'))
raw = json.loads((D / "p13a_spheres_raw.json").read_text())
rec = json.loads((D / "p13a_record_check.json").read_text())
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m); scene = TransportScene.attach(m, d)
row0 = next(json.loads(l) for l in open(_ag3s_workspace_path('outputs/verify/T28/E0/E0_ep1807_r1/frames.jsonl')) if '"control"' in l)
d.qpos[:] = row0["qpos"]; mujoco.mj_forward(m, d); q20 = scene.robot_state()
out = {"code": raw["code"], "pose": raw["pose"],
       "definitions": {
           "frac_vertices_inside_union": "fraction of MuJoCo collision-mesh vertices of that body inside the union of that link's constraint spheres",
           "vertex_outside_mm_median/max": "distance of mesh vertices outside the nearest sphere surface",
           "span": "along the body's principal axis (collision-mesh PCA): fraction of mesh length covered by [min(c.a-r), max(c.a+r)] of the spheres; uncovered_*_end = mesh beyond sphere extent at each end",
           "comparison_pose": "MuJoCo gripper closed (qpos 0) = the finger position the constraint model uses (finger joints pinned at 0); open-pose numbers in p13a_spheres_raw.json",
           "within_capsule_gap": "max over consecutive spheres of the same capsule of |c_i-c_j|-r_i-r_j (negative = overlap)",
           "sphere_model_finger_inner_gap": "left finger spheres, pinned q: min over l1/l2 sphere pairs |c1-c2|-r1-r2"},
       "configs": {}}
for name, opts in CONFIGS:
    robot = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts)
    wc = []
    for cap in robot.capsules:
        cs, r, _ = robot._capsule_sphere_centres(cap)
        for a, b in zip(cs[:-1], cs[1:]):
            wc.append(float(1000 * (np.linalg.norm(np.asarray(a) - np.asarray(b)) - 2 * r)))
    C, R = robot.sphere_centers_numeric(q20); names = np.array(robot.sphere_link_names)
    A, B = names == "ee_finger_l1", names == "ee_finger_l2"
    ig = float(1000 * (np.linalg.norm(C[A][:, None] - C[B][None], axis=2) - R[A][:, None] - R[B][None]).min())
    c = raw["configs"][name]; vc = c["vs_mesh"]["mujoco_gripper_closed_0"]; vo = c["vs_mesh"]["mujoco_gripper_open_-0.045"]
    out["configs"][name] = {
        "sphere_options": opts, "n_spheres_gripper_links_both_hands": c["n_spheres_gripper_links_total"],
        "n_spheres": {k: c["n_spheres_by_link"][k] for k in ("ee_left", "ee_finger_l1")},
        "radii_mm": {k: c["radii_mm_by_link"][k] for k in ("ee_left", "ee_finger_l1")},
        "max_within_capsule_gap_mm": max(wc),
        "sphere_model_finger_inner_gap_mm": ig,
        "finger_l1_vs_mesh_closed": {k: round(vc["ee_finger_l1"]["collision_mesh"][k], 3) for k in ("frac_vertices_inside_union", "vertex_outside_mm_median", "vertex_outside_mm_max", "frac_mesh_axis_len_spanned", "uncovered_at_min_end_mm", "uncovered_at_max_end_mm", "mesh_len_along_axis_mm")},
        "palm_ee_left_vs_mesh": {k: round(vc["ee_left"]["collision_mesh"][k], 3) for k in ("frac_vertices_inside_union", "vertex_outside_mm_median", "frac_mesh_axis_len_spanned")},
        "finger_l1_sphere_centroid_to_mesh_centroid_mm": {"mujoco_open_-0.045": round(vo["ee_finger_l1"]["sphere_centroid_to_collision_mesh_centroid_mm"], 1),
                                                           "mujoco_closed_0": round(vc["ee_finger_l1"]["sphere_centroid_to_collision_mesh_centroid_mm"], 1)},
    }
    print(name, out["configs"][name]["max_within_capsule_gap_mm"], ig)
out["urdf_fk_pinned_finger_vs_mujoco_finger_body_mm"] = {g: {L: round(v["urdf_fk_vs_mujoco_body_pos_mm"], 2) for L, v in x.items()} for g, x in raw["finger_link_pose_urdf_pinned_vs_mujoco"].items()}
out["note_pinned"] = raw["note_pinned"]
out["live_record_check"] = {run: [{"t": e["t"], "grip_l_qpos": round(e["grip_l_qpos"], 4),
                                    "recorded_l1_l2_sphere_centroid_sep_mm": round(e["recorded_l1_l2_sphere_centroid_separation_mm"], 2),
                                    "mujoco_l1_l2_mesh_centroid_sep_mm": round(e["mujoco_l1_l2_collision_mesh_centroid_separation_mm"], 2),
                                    "l1_sphere_to_mesh_centroid_mm": round(e["ee_finger_l1"]["recorded_sphere_centroid_to_mujoco_collision_mesh_centroid_mm"], 1),
                                    "l2_sphere_to_mesh_centroid_mm": round(e["ee_finger_l2"]["recorded_sphere_centroid_to_mujoco_collision_mesh_centroid_mm"], 1)} for e in es]
                            for run, es in rec.items()}
out["live_record_check_note"] = "recorded sphere_centres = chunk step-0 FK (server), paired planning-row qpos; T16/T14 run flags: --links gripper, sphere_spacing 0.4, max_spheres_per_capsule 32, capsule_radius_scale 0.2 (serve_safe.log)"
(D / "p13a_summary.json").write_text(json.dumps(out, indent=1))
