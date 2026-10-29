"""T28 Phase 3 offline 0c — T29 finger joints: model finger spheres vs the real (MuJoCo) finger collision meshes,
every planning row of T14 (30), T16 (34), T17 (75), ep 1807.

Models (working tree T26+T29), built once from model_transport.xml on a fresh MjData (as the server):
  constraint_e3   = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS, sphere_options={sphere_spacing 0.4,
                    max_spheres_per_capsule 32, capsule_radius_scale 0.05})   (E3 server flags)
  constraint_urdf = build_constraint_robot_model(scene, link_filter=GRIPPER_LINKS)  (URDF radii, scale 1.0)
  self_filter     = build_robot_model(scene)  (T29 461-sphere model; finger links include the self_filter_cover spheres)
Per row: MuJoCo re-posed with qpos; q = scene.robot_state(); finger params:
  new  = FingerJointMap(model).joint_values(openings_from_mujoco(m, d))  (the row's measured opening, client convention)
  old  = model defaults (all finger joints 0 = closed; what every model had before T29)
Geometry in MuJoCo world (sphere centres base frame -> world by the MuJoCo base body pose).
  centre offset  = |mean(sphere centres of URDF finger link L) - mean(collision-mesh vertices of MJCF body B)|, B = the MJCF finger
                   body nearest to L's sphere mean at the NEW params (mapping reported per hand).
  pair gap       = |mean(spheres L_a) - mean(spheres L_b)| (model) vs |mean(mesh B_a) - mean(mesh B_b)| (real), same hand.
  inner gap      = along u = unit(mesh centroid B_b - B_a): real = min(u.v over B_b verts) - max(u.v over B_a verts);
                   model = min(u.c - r over L_b spheres) - max(u.c + r over L_a spheres).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
from AG3S.ag3s.experiments.reports.grounding_report import GRIPPER_LINKS, build_constraint_robot_model, build_robot_model
from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
from AG3S.ag3s.robot_models.gripper_state import FingerJointMap, openings_from_mujoco

XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
RUNS = {"t14": _ag3s_workspace_path('outputs/live_test/20260928_t14/execute_ep1807/frames.jsonl'),
        "t16": _ag3s_workspace_path('outputs/live_test/20260928_t16/execute_ep1807/frames.jsonl'),
        "t17": _ag3s_workspace_path('outputs/live_test/20260928_t17/shadow_ep1807/frames.jsonl')}
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28/p3'))
HANDS = {"left": (("ee_finger_l1", "ee_finger_l2"), ("ee_finger_l1", "ee_finger_l2")),
         "right": (("ee_finger_r1", "ee_finger_r2"), ("ee_finger_r1", "ee_finger_r2"))}

m = mujoco.MjModel.from_xml_path(XML)
s0 = TransportScene.attach(m, mujoco.MjData(m))
MODELS = {"constraint_e3": build_constraint_robot_model(s0, link_filter=GRIPPER_LINKS, sphere_options=dict(
              sphere_spacing=0.4, max_spheres_per_capsule=32, capsule_radius_scale=0.05)),
          "constraint_urdf": build_constraint_robot_model(s0, link_filter=GRIPPER_LINKS),
          "self_filter": build_robot_model(s0)}
FMAP = {k: FingerJointMap(v) for k, v in MODELS.items()}
DEF = {k: {n: 0.0 for n in v.param_joint_names} for k, v in MODELS.items()}
d = mujoco.MjData(m); scene = TransportScene.attach(m, d)


def mesh_world(body):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    out = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
        out.append(v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])
    return np.vstack(out)


def spheres(name, q, params):
    mdl = MODELS[name]
    mdl.set_joint_parameters(params)
    C, R = mdl.sphere_centers_numeric(q)
    Tb = scene.body_pose("base")
    Cw = np.asarray(C) @ Tb[:3, :3].T + Tb[:3, 3]
    return Cw, np.asarray(R), np.array(mdl.sphere_link_names)


res = {"definitions": __doc__, "n_spheres": {k: int(v.n_spheres) for k, v in MODELS.items()},
       "finger_sphere_count_by_link": {k: {L: int((np.array(v.sphere_link_names) == L).sum()) for L in GRIPPER_LINKS} for k, v in MODELS.items()},
       "rows": []}
for run, path in RUNS.items():
    plan = [json.loads(l) for l in open(path) if '"planning"' in l]
    for fr in plan:
        d.qpos[:] = np.asarray(fr["qpos"], float); mujoco.mj_forward(m, d)
        q = scene.robot_state()
        opn = openings_from_mujoco(m, d)
        mesh = {b: mesh_world(b) for h in HANDS.values() for b in h[1]}
        row = {"run": run, "seq": fr["seq"], "t": fr["t_step"], "opening_norm": opn}
        for name in MODELS:
            newp = FMAP[name].joint_values(opn)
            for tag, params in (("new", newp), ("old", DEF[name])):
                Cw, R, names = spheres(name, q, params)
                ent = {"params": params}
                for hand, (links, bodies) in HANDS.items():
                    cen = {L: Cw[names == L].mean(0) for L in links}
                    mc = {b: mesh[b].mean(0) for b in bodies}
                    # mapping from NEW params geometry: computed per (row, model) on the new pass and reused for old
                    if tag == "new":
                        dmat = {(L, b): np.linalg.norm(cen[L] - mc[b]) for L in links for b in bodies}
                        straight = dmat[(links[0], bodies[0])] + dmat[(links[1], bodies[1])]
                        swapped = dmat[(links[0], bodies[1])] + dmat[(links[1], bodies[0])]
                        mp = {links[0]: bodies[0], links[1]: bodies[1]} if straight <= swapped else {links[0]: bodies[1], links[1]: bodies[0]}
                        row.setdefault("mapping", {})[f"{name}:{hand}"] = mp
                    mp = row["mapping"][f"{name}:{hand}"]
                    off = {L: float(1000 * np.linalg.norm(cen[L] - mc[mp[L]])) for L in links}
                    a, b = links
                    u = mc[mp[b]] - mc[mp[a]]; u /= np.linalg.norm(u)
                    real_inner = float(1000 * ((mesh[mp[b]] @ u).min() - (mesh[mp[a]] @ u).max()))
                    sa, sb = names == a, names == b
                    model_inner = float(1000 * ((Cw[sb] @ u - R[sb]).min() - (Cw[sa] @ u + R[sa]).max()))
                    ent[hand] = {"centre_offset_mm": off,
                                 "pair_gap_model_mm": float(1000 * np.linalg.norm(cen[a] - cen[b])),
                                 "pair_gap_real_mm": float(1000 * np.linalg.norm(mc[mp[a]] - mc[mp[b]])),
                                 "inner_gap_model_mm": model_inner, "inner_gap_real_mm": real_inner}
                row[f"{name}:{tag}"] = ent
        res["rows"].append(row)
    print(run, len(plan), flush=True)


def stat(v):
    v = np.asarray(v, float)
    return {"n": int(v.size), "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


summ = {}
for run in RUNS:
    R_ = [r for r in res["rows"] if r["run"] == run]
    for name in MODELS:
        for tag in ("new", "old"):
            for hand in HANDS:
                E = [r[f"{name}:{tag}"][hand] for r in R_]
                summ[f"{run}:{name}:{tag}:{hand}"] = {
                    "centre_offset_mm": stat([x for e in E for x in e["centre_offset_mm"].values()]),
                    "pair_gap_model_minus_real_mm": stat([e["pair_gap_model_mm"] - e["pair_gap_real_mm"] for e in E]),
                    "pair_gap_real_mm": stat([e["pair_gap_real_mm"] for e in E]),
                    "pair_gap_model_mm": stat([e["pair_gap_model_mm"] for e in E]),
                    "inner_gap_real_mm": stat([e["inner_gap_real_mm"] for e in E]),
                    "inner_gap_model_mm": stat([e["inner_gap_model_mm"] for e in E]),
                    "inner_gap_real_minus_model_mm": stat([e["inner_gap_real_mm"] - e["inner_gap_model_mm"] for e in E])}
    summ[f"{run}:opening_norm_left"] = stat([r["opening_norm"]["left"] for r in R_])
    summ[f"{run}:mapping"] = sorted({json.dumps(r["mapping"], sort_keys=True) for r in R_})
res["summary"] = summ
(OUT / "off_0c.json").write_text(json.dumps(res, indent=1, default=float))
for k, v in summ.items():
    if ":left" in k and "constraint" in k or "opening" in k:
        print(k, v if not isinstance(v, dict) else {kk: (round(vv["median"], 2), round(vv["max"], 2)) if isinstance(vv, dict) else vv for kk, vv in v.items()})
