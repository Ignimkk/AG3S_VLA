"""T28 P1-3 (c) max inner finger opening from the MuJoCo model (model_transport.xml, mujoco 3.11).
gap_geom  = min witness-checked mj_geomDistance between every ee_finger_l1 geom and every ee_finger_l2 geom
            (collision geoms: contype=1 group 3; also visual group-2 meshes separately).
gap_axis  = along the closing axis u (unit vector between the two finger body origins):
            min_{v in l1} u.v - max_{w in l2} u.w  (inner-face separation of the vertex sets).
Joint: gripper_finger_l1 range [-0.05, 0] (axis -x), l2 = -l1 by equality 'left_finger';
dataset open command RBY1_GRIPPER_OPEN = -0.045 (pi05_infer.py:127). Fruit sizes from their collision geoms.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P
XML = P.XML
m = mujoco.MjModel.from_xml_path(XML); d = mujoco.MjData(m)
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def geoms(body, collision):
    b = bid(body)
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
            and bool(m.geom_contype[g] or m.geom_conaffinity[g]) == collision]


def verts(g):
    mid = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(float)
    return v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]


row0 = next(json.loads(l) for l in open(_ag3s_workspace_path('outputs/verify/T28/E0/E0_ep1807_r1/frames.jsonl')) if '"control"' in l)
out = {"definitions": __doc__, "left": {}, "right": {}}
for side, (j1, j2, b1, b2) in {"left": (27, 28, "ee_finger_l1", "ee_finger_l2"), "right": (18, 19, "ee_finger_r1", "ee_finger_r2")}.items():
    for q in (-0.05, -0.045, -0.03, 0.0):
        d.qpos[:] = row0["qpos"]; d.qpos[j1] = q; d.qpos[j2] = -q; mujoco.mj_forward(m, d)
        e = {}
        for kind, col in (("collision", True), ("visual", False)):
            G1, G2 = geoms(b1, col), geoms(b2, col)
            e[f"gap_geom_{kind}_mm"] = 1000 * min(P.gdist(m, d, a, b, 1.0) for a in G1 for b in G2)
            u = d.xpos[bid(b1)] - d.xpos[bid(b2)]; u /= np.linalg.norm(u)
            V1 = np.vstack([verts(g) for g in G1]); V2 = np.vstack([verts(g) for g in G2])
            e[f"gap_axis_{kind}_mm"] = float(1000 * ((V1 @ u).min() - (V2 @ u).max()))
        e["finger_body_origin_separation_mm"] = float(1000 * np.linalg.norm(d.xpos[bid(b1)] - d.xpos[bid(b2)]))
        out[side][f"q={q}"] = e
        print(side, q, {k: round(v, 2) for k, v in e.items()})
d.qpos[:] = row0["qpos"]; mujoco.mj_forward(m, d)
fr = {}
for f in ("apple", "orange", "pear", "banana"):
    V = np.vstack([verts(g) for g in range(m.ngeom) if m.geom_bodyid[g] == bid(f) and (m.geom_contype[g] or m.geom_conaffinity[g])])
    c = V.mean(0); w, U = np.linalg.eigh(np.cov((V - c).T))
    ext = sorted(float(1000 * ((V @ U[:, i]).max() - (V @ U[:, i]).min())) for i in range(3))
    fr[f] = {"principal_extents_mm_sorted": [round(x, 1) for x in ext]}
out["fruit_collision_geom_extents"] = fr
print(fr)
out["n_invalid_geomdistance_pairs_skipped"] = P.N_INVALID[0]
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T28/P1-3/p13c_opening.json'), "w"), indent=1)
