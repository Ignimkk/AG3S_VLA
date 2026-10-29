"""T45 verifier common: constraint sphere model (same builder as the T44d CL server) + MuJoCo world-frame hand collision geometry.
Snapshot code ${AG3S_PROJECT_ROOT}-t44cl (benchmark f891abe; urdf_sphere_chain / grounding_report / gripper_cover / serve_safe identical to ${AG3S_PROJECT_ROOT}/benchmark).
Read-only use of project code. CPU only."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("OMP_NUM_THREADS", "2"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-t44cl")
import numpy as np, mujoco
from scipy.spatial import ConvexHull
XML = "${AG3S_PROJECT_ROOT}-t44cl/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
SERVER_SPHERES = {"capsule_radius_scale": 0.05, "sphere_spacing": 0.4, "max_spheres_per_capsule": 32}
_cache = {}

def scene():
    if "sc" not in _cache:
        from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
        _cache["sc"] = TransportScene(XML, settle_steps=0)
    return _cache["sc"]

def model(links="gripper", sphere_options=None, finger_cover={"hands": "both"}, gripper_cover=None, key=None):
    key = key or (links, repr(sphere_options), repr(finger_cover), repr(gripper_cover))
    if key in _cache:
        return _cache[key]
    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.trajopt.serve_safe import constraint_link_filter
    sc = scene()
    kw = {}
    if finger_cover: kw["finger_cover"] = finger_cover
    if gripper_cover: kw["gripper_cover"] = gripper_cover
    qm = build_constraint_robot_model(sc, link_filter=constraint_link_filter(links),
                                      sphere_options=dict(SERVER_SPHERES if sphere_options is None else sphere_options), **kw)
    _cache[key] = qm
    return qm

def q_from_qpos(sc, qm, qpos):
    return np.asarray([qpos[sc._qadr[n]] for n in qm.joint_names], np.float64)

def params_from_qpos(sc, qm, qpos):
    names = tuple(getattr(qm, "param_joint_names", ()) or ())
    return np.asarray([qpos[sc._qadr[n]] for n in names], np.float64) if names else None

def spheres_world(qm, qpos):
    sc = scene(); qpos = np.asarray(qpos, np.float64)
    c, r = qm.sphere_centers_numeric(q_from_qpos(sc, qm, qpos), params_from_qpos(sc, qm, qpos))
    return np.asarray(c), np.asarray(r)

# ---------------- MuJoCo side ----------------
M = mujoco.MjModel.from_xml_path(XML)
D = mujoco.MjData(M)
BODY = lambda n: mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, n)
BNAME = lambda i: mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, int(i))

def coll_geoms(body_id):
    return [g for g in range(M.ngeom) if M.geom_bodyid[g] == body_id and (M.geom_contype[g] | M.geom_conaffinity[g])]

def forward(qpos):
    D.qpos[:] = np.asarray(qpos, np.float64); D.qvel[:] = 0
    mujoco.mj_forward(M, D)

def geom_world_verts(g):
    mesh = int(M.geom_dataid[g]); v0, nv = int(M.mesh_vertadr[mesh]), int(M.mesh_vertnum[mesh])
    local = np.asarray(M.mesh_vert[v0:v0 + nv], np.float64).reshape(-1, 3)
    return local @ D.geom_xmat[g].reshape(3, 3).T + D.geom_xpos[g]

def hull_sample(V, n_per_area=2.0e6, rng=None):
    """area-weighted uniform samples on the convex hull surface (MuJoCo collides mesh geoms as convex hulls).
    n_per_area: samples per m^2 (2e6 -> 1 per 0.5 mm^2)."""
    rng = rng or np.random.default_rng(0)
    h = ConvexHull(V); eq = h.equations; S = h.simplices
    a, b, c = V[S[:, 0]], V[S[:, 1]], V[S[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.round(area * n_per_area).astype(int))
    f = np.repeat(np.arange(len(S)), n)
    u = rng.random((len(f), 2)); flip = u.sum(1) > 1; u[flip] = 1 - u[flip]
    P = a[f] + u[:, :1] * (b[f] - a[f]) + u[:, 1:] * (c[f] - a[f])
    P = np.vstack([P, V[np.unique(S)]])
    return P, eq, float(area.sum())

def body_hulls(body_id):
    out = []
    for g in coll_geoms(body_id):
        V = geom_world_verts(g)
        out.append((g, V, ConvexHull(V).equations))
    return out

def inside_depth(P, eqs):
    """max over hulls of depth (>0 inside)."""
    d = np.full(len(P), -np.inf)
    for eq in eqs:
        d = np.maximum(d, -np.max(P @ eq[:, :3].T + eq[:, 3], axis=1))
    return d

def gap(P, C, R, chunk=4096):
    out = np.full(len(P), np.inf)
    if len(R) == 0: return out
    for i in range(0, len(P), chunk):
        p = P[i:i + chunk]
        out[i:i + chunk] = np.min(np.linalg.norm(p[:, None] - C[None], axis=2) - R[None], axis=1)
    return out
