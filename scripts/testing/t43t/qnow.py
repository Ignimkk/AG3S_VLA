"""T43 T — rebuild the TO rows of the recorded held chunks (record inputs only) and classify them
(a) on the recorded refined plan — fidelity vs `clearance_min_per_step_mm` — and
(b) on the stationary plan Q = q_now (is q_now itself feasible / certifiable?).
Usage: qnow.py OUT.json RUNKEY SEQ0 SEQ1"""
from common.workspace import rebase_legacy_paths, workspace_path as _ag3s_workspace_path
import json, sys, pathlib, dataclasses, functools
import numpy as np
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
from AG3S.trajopt.types import ChunkLayout
from AG3S.trajopt.linearize import CollisionLinearizer
from AG3S.trajopt.safe_policy import classify_violations
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap

S = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify/T43/S/raw'))
IX = rebase_legacy_paths({**json.load(open(S / "chunk_index_t42.json")), **json.load(open(S / "chunk_index_t40t41.json"))})
RUNS = {
 "T42_R1908": _ag3s_workspace_path('legacy/outputs/verify/T42/1/runs/E3b/noobs/E3b_ep1908_s19082'),
 "T42_L1992": _ag3s_workspace_path('legacy/outputs/verify/T42/1/runs/E3b/noobs/E3b_ep1992_s19922'),
}
H = 8

class _Args:
    links = None; sphere_spacing = None; max_spheres_per_capsule = None; capsule_radius_scale = None
    max_sphere_radius = None; esdf_margin = None; target_field_policy = None; phase = None
    manipulators = None; model_xml = None; hand = None; allow_uncertified = False; rows_per_step = None

@functools.lru_cache(maxsize=4)
def setup(server_log, manifest_path):
    man = json.loads(pathlib.Path(manifest_path).read_text())
    man["scene"]["model_xml"] = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
    st = FIA.resolve_settings(_Args, FIA.parse_server_log(server_log), man)
    robot, qadr, _ = FIA.build_robot(st)
    layout = ChunkLayout.rby1(robot.joint_names)
    _, _, _, maxd, unk = FIA.contact_policy(st, robot.sphere_link_names, has_target=False, has_free=False, attached=False)
    margin, _, _, _, _ = FIA.contact_policy(st, robot.sphere_link_names, has_target=True, has_free=False, attached=False)
    lin = CollisionLinearizer(robot, layout, H)
    return dict(st=st, robot=robot, qadr=qadr, layout=layout, outside=maxd if unk == "free" else -maxd,
                lin=lin, margin=margin, fmap=FingerJointMap(robot))

def load(rd):
    ix = IX[rd]
    rows = [json.loads(l) for l in (pathlib.Path(rd) / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = {r["seq"]: r for r in rows if r.get("kind") == "planning"}
    return ix, plan

def scene_for(G, z, s, qn):
    att = (s["grasp"].get("attached") or {})
    held = att.get("held") or {}
    sp = held.get("spheres_parent_frame") or []
    field = FIA.build_field(z, G["outside"])
    scene = FIA.make_scene(field, G["robot"].radii, G["st"].esdf_margin)
    if not sp:
        return scene, field, None
    local = np.array([c["centre_m"] for c in sp], float); radii = np.array([c["radius_mm"] / 1e3 for c in sp])
    parent = att["parent_link"]
    T = np.asarray(G["robot"].link_pose(qn, parent), float)
    world = local @ T[:3, :3].T + T[:3, 3]
    scene = dataclasses.replace(scene, manipulated_spheres=world, manipulated_sphere_radii=radii,
                                manipulated_link_margin=G["margin"], attached_points=local,
                                attached_parent_link=parent, attached_radii=radii)
    return scene, field, (local, parent, radii, world)

def plane_of(s):
    pl = (((s["grasp"].get("attached") or {}).get("held") or {}).get("support") or {}).get("plane")
    return None if pl is None else (np.asarray(pl["normal"], float), float(pl["offset_m"]))

def rows_summary(cls, field, held, plane):
    out = []
    for r in sorted(cls["rows"], key=lambda r: r["clearance_m"])[:400]:
        p = np.asarray(r["point_m"])
        e = {k: r[k] for k in ("step", "link", "class", "why", "tier")}
        e["cl_mm"] = round(1e3 * r["clearance_m"], 2)
        e["d_field_mm"] = round(1e3 * float(field.distance(p[None])[0]), 2)
        e["d_obj_mm"] = None if r.get("object_distance_m") is None else round(1e3 * r["object_distance_m"], 2)
        if plane is not None:
            e["d_plane_mm"] = round(1e3 * (float(plane[0] @ p) - plane[1]), 2)
        e["p"] = [round(v, 4) for v in p]
        out.append(e)
    return out

def main():
    out_path, key, s0, s1 = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    rd = RUNS[key]; ix, plan = load(rd)
    G = setup(str(pathlib.Path(ix["server"]) / "serve_safe.log"), str(pathlib.Path(rd) / "manifest.json"))
    lin, layout = G["lin"], G["layout"]
    res = {}
    for seq in range(s0, s1 + 1):
        f = ix["chunks"][str(seq)]["npz"]
        with np.load(f, allow_pickle=False) as z:
            s = json.loads(str(z["summary_json"]))
            refc = np.asarray(z["refined_chunk"], float)
            qn = FIA.q_now_from_qpos(np.asarray(plan[seq]["qpos"], float), G["qadr"])
            scene, field, held = scene_for(G, z, s, qn)
        meas = (s.get("finger_joints") or {}).get("opening_norm") or {}
        path, _ = G["fmap"].path(meas, {"left": refc[:, 7], "right": refc[:, 15]}, H, default=G["robot"].param_vector())
        lin.set_joint_parameter_path(path)
        if held is not None:
            lin.set_attached(held[0], held[1], held[2])
        else:
            lin.set_attached(None, None)
        auth = ((s.get("verdict") or {}).get("classification") or {}).get("authorized_links") or []
        man = s.get("manipulated") or {}
        traj = layout.chunk_to_trajectory(refc)[:, :H]
        stay = np.tile(qn[layout.q_indices].reshape(-1, 1), (1, H))
        rec = {"t": s["t_step"], "grasp": s["grasp"].get("state"), "recorded_step_min_mm": s.get("clearance_min_per_step_mm")}
        for name, tr in (("refined", traj), ("stay", stay)):
            cl = lin.esdf_clearance(tr, qn, scene)[..., 0]
            cls = classify_violations(lin, tr, qn, scene, authorized_links=auth, tolerance=1e-4,
                                      manipulated=man, pad_m=0.01, keep_rows=True)
            rec[name] = {"step_min_mm": [round(1e3 * float(v), 2) for v in cl.min(axis=1)],
                         "robot_min_mm": round(1e3 * float(cl[:, :lin.n_spheres].min()), 2),
                         "held_min_mm": None if held is None else round(1e3 * float(cl[:, lin.n_spheres:].min()), 2),
                         "n_allowed": cls["n_allowed"], "n_unresolved": cls["n_unresolved"], "n_collision": cls["n_collision"],
                         "collision_by_why": cls["collision_by_why"],
                         "residual_mm": round(1e3 * cls["residual_violation_m"], 2),
                         "rows": rows_summary(cls, field, held, plane_of(s)) if name == "stay" else None}
        if held is not None:
            rec["held_world"] = [[round(v, 4) for v in c] for c in held[3]]; rec["held_r_mm"] = [round(1e3 * r, 1) for r in held[2]]
            rec["plane"] = None if plane_of(s) is None else [list(plane_of(s)[0]), plane_of(s)[1]]
        res[seq] = rec
        print(key, seq, "fid", rec["recorded_step_min_mm"] and [round(a - b, 2) for a, b in zip(rec["refined"]["step_min_mm"], rec["recorded_step_min_mm"])],
              "| stay", rec["stay"]["step_min_mm"][0], "robot", rec["stay"]["robot_min_mm"], "held", rec["stay"]["held_min_mm"],
              "coll", rec["stay"]["n_collision"], rec["stay"]["collision_by_why"], "unres", rec["stay"]["n_unresolved"], "resid", rec["stay"]["residual_mm"], flush=True)
    lin.set_attached(None, None)
    pathlib.Path(out_path).write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main()
