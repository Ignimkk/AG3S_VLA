"""Is the nearest field surface of the worst palm row the robot's own hand (self-filter residue)? Distance from that surface point to the hand's MuJoCo hull surface at q_now."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np, mujoco
from scipy.spatial import ConvexHull, cKDTree
XML = "${AG3S_PROJECT_ROOT}-t44cl/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
M = mujoco.MjModel.from_xml_path(XML); D = mujoco.MjData(M)
S = json.load(open(_ag3s_workspace_path('outputs/verify/T45/V/diag_surface.json')))
def hull_pts(V, n=4000, rng=np.random.default_rng(0)):
    h = ConvexHull(V); Sx = h.simplices; a, b, c = V[Sx[:, 0]], V[Sx[:, 1]], V[Sx[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    f = rng.choice(len(Sx), n, p=area / area.sum()); u = rng.random((n, 2)); fl = u.sum(1) > 1; u[fl] = 1 - u[fl]
    return a[f] + u[:, :1] * (b[f] - a[f]) + u[:, 1:] * (c[f] - a[f])
out = []
for rec in S:
    name, seq = rec["run"], rec["seq"]
    rows = [json.loads(l) for l in open(f"{_ag3s_workspace_path()}/outputs/verify/T44/CL/runs/C1a/R/{name}/frames.jsonl") if l.strip()]
    row = next(r for r in rows if r["kind"] == "planning" and int(r["seq"]) == seq)
    D.qpos[:] = np.asarray(row["qpos"], float); D.qvel[:] = 0; mujoco.mj_forward(M, D)
    q = np.asarray(rec["nearest_surface_pt"])
    res = {}
    for g in range(M.ngeom):
        if not (M.geom_contype[g] | M.geom_conaffinity[g]): continue
        b = int(M.geom_bodyid[g]); bn = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)
        if M.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
        mesh = int(M.geom_dataid[g]); v0, nv = int(M.mesh_vertadr[mesh]), int(M.mesh_vertnum[mesh])
        V = np.asarray(M.mesh_vert[v0:v0 + nv], float).reshape(-1, 3) @ D.geom_xmat[g].reshape(3, 3).T + D.geom_xpos[g]
        if np.linalg.norm(V.mean(0) - q) > 0.4: continue
        P = hull_pts(V)
        d = float(np.linalg.norm(P - q, axis=1).min())
        res[bn] = min(res.get(bn, 1e9), d)
    top = sorted(res.items(), key=lambda x: x[1])[:5]
    out.append({"run": name, "seq": seq, "nearest_bodies_mm_at_q_now": [(k, round(v * 1e3, 1)) for k, v in top]})
    print(name, seq, out[-1]["nearest_bodies_mm_at_q_now"], flush=True)
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T45/V/diag_self.json'), "w"), indent=1)
