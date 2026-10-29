"""T44dA verifier: constraint sphere model (same build as the CL server: links gripper|arms, capsule_radius_scale 0.05,
sphere_spacing 0.4, max_spheres_per_capsule 32, finger cover both) for CPU FK of reference / refined chunks.
Snapshot code: ${AG3S_PROJECT_ROOT}-t44cl (benchmark f891abe). Read-only use."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES", ""); os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-t44cl")
import numpy as np
XML = "${AG3S_PROJECT_ROOT}-t44cl/robots/rby1/description/models/rby1a/mujoco/model_transport.xml"
_cache = {}

def model(links="gripper"):
    if links in _cache:
        return _cache[links]
    from AG3S.ag3s.experiments.reports.grounding_report import build_constraint_robot_model
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    from AG3S.trajopt.serve_safe import constraint_link_filter
    sc = TransportScene(XML, settle_steps=0)
    qm = build_constraint_robot_model(sc, link_filter=constraint_link_filter(links),
                                      sphere_options={"capsule_radius_scale": 0.05, "sphere_spacing": 0.4, "max_spheres_per_capsule": 32},
                                      finger_cover={"hands": "both"})
    _cache[links] = (sc, qm)
    return sc, qm

def q_from_qpos(sc, qm, qpos):
    qpos = np.asarray(qpos, np.float64)
    return np.asarray([qpos[sc._qadr[n]] for n in qm.joint_names], np.float64)

def params_from_qpos(sc, qm, qpos):
    names = tuple(getattr(qm, "param_joint_names", ()) or ())
    if not names:
        return None
    qpos = np.asarray(qpos, np.float64)
    return np.asarray([qpos[sc._qadr[n]] for n in names], np.float64)

def chunk_q(sc, qm, qpos, chunk):
    """full model q per chunk row: q_now with left_arm_0..6 <- chunk[:,0:7], right_arm_0..6 <- chunk[:,8:15]."""
    base = q_from_qpos(sc, qm, qpos)
    names = list(qm.joint_names)
    out = np.repeat(base[None], len(chunk), 0)
    for i in range(7):
        out[:, names.index(f"left_arm_{i}")] = chunk[:, i]
        out[:, names.index(f"right_arm_{i}")] = chunk[:, 8 + i]
    return out

def spheres(qm, q, params=None):
    c = qm.sphere_centers_numeric(q, params) if params is not None else qm.sphere_centers_numeric(q)
    c = np.asarray(c[0] if isinstance(c, tuple) else c, np.float64)
    return c
