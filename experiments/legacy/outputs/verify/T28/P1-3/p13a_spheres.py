"""T28 P1-3 (a) gripper constraint spheres vs MuJoCo finger/palm meshes, and (c) max inner finger opening.

Code: benchmark HEAD snapshot (git archive cee8f84) on PYTHONPATH — independent of T26/T27 working-tree edits.
Constraint model = grounding_report.build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS,
sphere_options=...) — the path serve_safe/fixed_input_ablation use for `--links gripper`.
Pose: E0 ep1807 r1 control row t=0 qpos (gripper open, qpos[27]=-0.045); plus the same arm pose with the
MuJoCo gripper closed (qpos[27]=0) to separate the pinned-finger effect.
All geometry compared in MuJoCo world: sphere centres (URDF FK, base frame) -> world by MuJoCo base pose.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, build_constraint_robot_model
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
import AG3S.ag3s.robot_models.urdf_sphere_chain as usc

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/P1-3'))
E0ROW = _ag3s_workspace_path('outputs/verify/T28/E0/E0_ep1807_r1/frames.jsonl')

CONFIGS = [
    ("urdf_scale1.0_sp1.0_max8(default)", dict()),
    ("live_scale0.2_sp0.4_max32", dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.2)),
    ("scale0.1_sp0.4_max32", dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.1)),
    ("scale0.05_sp0.4_max32", dict(sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)),
    ("scale0.2_sp0.2_max64", dict(sphere_spacing=0.2, max_spheres_per_capsule=64, capsule_radius_scale=0.2)),
    ("scale0.1_sp0.2_max64", dict(sphere_spacing=0.2, max_spheres_per_capsule=64, capsule_radius_scale=0.1)),
    ("scale0.05_sp0.2_max64", dict(sphere_spacing=0.2, max_spheres_per_capsule=64, capsule_radius_scale=0.05)),
    ("scale0.1_sp0.1_max64", dict(sphere_spacing=0.1, max_spheres_per_capsule=64, capsule_radius_scale=0.1)),
    ("scale0.05_sp0.1_max64", dict(sphere_spacing=0.1, max_spheres_per_capsule=64, capsule_radius_scale=0.05)),
]
LINKS = ["ee_left", "ee_finger_l1", "ee_finger_l2"]
MJ_BODY = {"ee_left": "EE_BODY_L", "ee_finger_l1": "ee_finger_l1", "ee_finger_l2": "ee_finger_l2"}


def body_mesh_world(m, d, body, collision_only):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        col = bool(m.geom_contype[g] or m.geom_conaffinity[g])
        if collision_only and not col:
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        out.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.vstack(out)


def mesh_vs_spheres(V, C, R, axis):
    dist = np.linalg.norm(V[:, None, :] - C[None, :, :], axis=2) - R[None, :]
    dmin = dist.min(1)   # >0 : vertex outside every sphere by that much
    pv = V @ axis; pc = C @ axis
    s_lo, s_hi = float((pc - R).min()), float((pc + R).max())
    return {"n_vertices": int(len(V)), "frac_vertices_inside_union": float((dmin <= 0).mean()),
            "vertex_outside_mm_median": float(1000 * np.median(np.maximum(dmin, 0))),
            "vertex_outside_mm_max": float(1000 * dmin.max()),
            "mesh_axis_extent_mm": [float(1000 * pv.min()), float(1000 * pv.max())],
            "spheres_axis_extent_mm": [1000 * s_lo, 1000 * s_hi],
            "mesh_len_along_axis_mm": float(1000 * (pv.max() - pv.min())),
            "uncovered_at_min_end_mm": float(1000 * max(0.0, s_lo - pv.min())),
            "uncovered_at_max_end_mm": float(1000 * max(0.0, pv.max() - s_hi)),
            "frac_mesh_axis_len_spanned": float(max(0.0, min(s_hi, pv.max()) - max(s_lo, pv.min())) / (pv.max() - pv.min()))}


def main():
    m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)
    row0 = next(json.loads(l) for l in open(E0ROW) if '"control"' in l)
    qpos_open = np.asarray(row0["qpos"], float)
    res = {"code": "benchmark HEAD cee8f84 snapshot", "pose": "E0_ep1807_r1 control t=0", "configs": {}}
    for gstate, gq in (("mujoco_gripper_open_-0.045", -0.045), ("mujoco_gripper_closed_0", 0.0)):
        q = qpos_open.copy(); q[27] = gq; q[28] = -gq
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        Tb = scene.body_pose("base")
        q20 = scene.robot_state()
        # finger principal axis from collision mesh (world)
        for name, opts in CONFIGS:
            robot = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=opts)
            C, R = robot.sphere_centers_numeric(q20)
            Cw = C @ Tb[:3, :3].T + Tb[:3, 3]
            names = np.array(robot.sphere_link_names)
            entry = res["configs"].setdefault(name, {"sphere_options": opts, "n_spheres_gripper_links_total": int(robot.n_spheres),
                                                     "n_spheres_by_link": {}, "radii_mm_by_link": {}, "capsules": [], "vs_mesh": {}})
            if gstate.startswith("mujoco_gripper_open"):
                for L in sorted(set(names)):
                    entry["n_spheres_by_link"][L] = int((names == L).sum())
                    entry["radii_mm_by_link"][L] = sorted({round(1000 * r, 2) for r in R[names == L]})
                for cap in robot.capsules:
                    entry["capsules"].append({"link": cap.link, "urdf_radius_mm": round(1000 * cap.radius, 2), "length_mm": round(1000 * cap.length, 2)})
                entry["coverage_shortfall"] = [{"link": s.link, "n": s.n_spheres, "sphere_r_mm": round(1000 * s.sphere_radius, 2),
                                                "required_r_mm": round(1000 * s.required_radius, 2)} for s in robot.coverage_shortfall]
                # chain gaps (consecutive spheres of the same capsule order)
                gaps = []
                for L in LINKS:
                    idx = np.where(names == L)[0]
                    for a, b in zip(idx[:-1], idx[1:]):
                        gaps.append(float(1000 * (np.linalg.norm(C[a] - C[b]) - R[a] - R[b])))
                entry["max_consecutive_sphere_gap_mm_left_gripper"] = max(gaps)
            vs = {}
            for L in LINKS:
                Vc = body_mesh_world(m, d, MJ_BODY[L], True)
                Vv = body_mesh_world(m, d, MJ_BODY[L], False)
                cen = Vc.mean(0); w, U = np.linalg.eigh(np.cov((Vc - cen).T)); axis = U[:, 2]
                sel = names == L
                vs[L] = {"collision_mesh": mesh_vs_spheres(Vc, Cw[sel], R[sel], axis),
                         "all_mesh(visual+collision)": mesh_vs_spheres(Vv, Cw[sel], R[sel], axis),
                         "sphere_centroid_to_collision_mesh_centroid_mm": float(1000 * np.linalg.norm(Cw[sel].mean(0) - cen))}
            entry["vs_mesh"][gstate] = vs
            print(gstate, name, robot.n_spheres, {L: (vs[L]["collision_mesh"]["frac_vertices_inside_union"],
                                                    round(vs[L]["collision_mesh"]["vertex_outside_mm_max"], 1),
                                                    round(vs[L]["sphere_centroid_to_collision_mesh_centroid_mm"], 1),
                                                    round(vs[L]["collision_mesh"]["uncovered_at_min_end_mm"], 1),
                                                    round(vs[L]["collision_mesh"]["uncovered_at_max_end_mm"], 1)) for L in LINKS}, flush=True)
    # finger URDF link pose vs MuJoCo body pose
    fk = {}
    for gstate, gq in (("open_-0.045", -0.045), ("closed_0", 0.0), ("limit_-0.05", -0.05)):
        q = qpos_open.copy(); q[27] = gq; q[28] = -gq
        d.qpos[:] = q; mujoco.mj_forward(m, d)
        Tb = scene.body_pose("base"); q20 = scene.robot_state()
        robot = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options=CONFIGS[1][1])
        for L in LINKS:
            Tu = Tb @ robot.link_pose(q20, L)
            Tm = scene.body_pose(MJ_BODY[L])
            fk.setdefault(gstate, {})[L] = {"urdf_fk_vs_mujoco_body_pos_mm": float(1000 * np.linalg.norm(Tu[:3, 3] - Tm[:3, 3])),
                                            "urdf_fk_minus_mujoco_mm": (1000 * (Tu[:3, 3] - Tm[:3, 3])).round(2).tolist()}
    res["finger_link_pose_urdf_pinned_vs_mujoco"] = fk
    res["note_pinned"] = ("constraint model q = DEFAULT_RBY1_JOINTS (20: torso+arms); gripper_finger_* not in q -> pinned at "
                          "fixed_joint_values.get(name, 0.0); build_constraint_robot_model passes only head joints")
    (OUT / "p13a_spheres_raw.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
