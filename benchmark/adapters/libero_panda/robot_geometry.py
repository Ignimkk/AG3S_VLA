"""LIBERO Panda robot geometry and kinematics, read from the MuJoCo model (SL2). Python 3.8 compatible.

The Panda robot model on the AG3S side (``AG3S/adapters/panda``, Python 3.11) is a URDF sphere chain;
this module is where its ground truth comes from -- the MuJoCo model LIBERO actually simulates
(``MountedPanda`` + ``RethinkMount`` + ``PandaGripper`` from robosuite 1.4):

* :func:`export_tables` -- the robot subtree (bodies ``robot0_*``, ``gripper0_*``, ``mount0_*``):
  constant body poses, joints, cameras, sites, and every geom with its mesh **in the body frame**.
  ``robots/panda/description/README.md`` records which export the committed self-filter cover was
  fitted from; the AG3S side builds a MuJoCo model from these tables to run the V11AB
  ``self_filter_covering_capsules`` unchanged.
* :func:`fk_poses` -- pose of every robot body / site / camera in the ``robot0_base`` frame at given
  joint values (the FK fixture the 3.11 sphere chain is compared against).

Two sources, same tables:

* ``libero`` -- a real LIBERO env (``libero_spatial`` task 0, the SL0/SL1 task); the reference.
* ``robosuite`` -- the robot alone, assembled the way robosuite ``Robot.load_model`` /
  ``SingleArm.load_model`` assemble it (``create_robot`` -> ``add_mount`` -> ``add_gripper``), in an
  empty world. ~1 s instead of ~20 s; the unit test checks that its kinematic tables equal the
  committed ``libero`` export.

CLI (``.venv-libero``; ``MUJOCO_GL=osmesa``, no ``PYTHONPATH``)::

    python benchmark/adapters/libero_panda/robot_geometry.py export --source libero --out X.npz
    python benchmark/adapters/libero_panda/robot_geometry.py fk --source robosuite --out Y.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mount0_")
BASE_BODY = "robot0_base"
#: Joint qpos layout of the adapter ``robot_state`` (camera.ROBOT_STATE_LAYOUT) -- the FK fixture's q.
ROBOT_JOINTS = ("robot0_joint1", "robot0_joint2", "robot0_joint3", "robot0_joint4", "robot0_joint5",
                "robot0_joint6", "robot0_joint7", "gripper0_finger_joint1", "gripper0_finger_joint2")
#: Fixed configurations of the FK fixture: (7 arm joints, finger_joint1, finger_joint2) in MuJoCo qpos.
#: Row 0 = LIBERO MountedPanda.init_qpos with the gripper open; the others spread every joint over its range.
FK_FIXTURE_Q = (
    (0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.2267522, 0.7853981633974483, 0.04, -0.04),
    (0.6, 0.4, -0.5, -1.6, 0.8, 1.2, -1.1, 0.0, 0.0),
    (-1.9, -1.2, 2.1, -2.8, -2.2, 3.3, 2.5, 0.021, -0.013),
    (2.5, 1.5, -2.4, -0.3, 2.6, 0.2, -2.7, 0.035, -0.002),
)
_MESH = 7  # mujoco.mjtGeom.mjGEOM_MESH


def _name(m, objtype, i):
    import mujoco

    return mujoco.mj_id2name(m, objtype, int(i)) or ""


def robot_body_ids(m):
    import mujoco

    return [b for b in range(m.nbody) if _name(m, mujoco.mjtObj.mjOBJ_BODY, b).startswith(ROBOT_BODY_PREFIXES)]


def quat_to_matrix(q):
    w, x, y, z = (float(v) for v in np.asarray(q, np.float64) / np.linalg.norm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def export_tables(m):
    """Robot subtree of raw ``mujoco.MjModel`` ``m`` -> dict of arrays (npz-ready, deterministic).

    Mesh vertices are written in the **body** frame (compiled mesh vertices transformed by the geom's
    compiled ``geom_pos`` / ``geom_quat``), float64; faces are per-geom local indices.
    """
    import mujoco

    obj = mujoco.mjtObj
    bodies = robot_body_ids(m)
    index = {b: k for k, b in enumerate(bodies)}
    out = {
        "body_names": np.array([_name(m, obj.mjOBJ_BODY, b) for b in bodies]),
        "body_parent": np.array([_name(m, obj.mjOBJ_BODY, m.body_parentid[b]) for b in bodies]),
        "body_pos": np.array([m.body_pos[b] for b in bodies], np.float64),
        "body_quat": np.array([m.body_quat[b] for b in bodies], np.float64),
    }
    joints = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) in index]
    out.update({
        "joint_names": np.array([_name(m, obj.mjOBJ_JOINT, j) for j in joints]),
        "joint_body": np.array([index[int(m.jnt_bodyid[j])] for j in joints], np.int64),
        "joint_type": np.array([int(m.jnt_type[j]) for j in joints], np.int64),
        "joint_axis": np.array([m.jnt_axis[j] for j in joints], np.float64).reshape(-1, 3),
        "joint_pos": np.array([m.jnt_pos[j] for j in joints], np.float64).reshape(-1, 3),
        "joint_range": np.array([m.jnt_range[j] for j in joints], np.float64).reshape(-1, 2),
        "joint_limited": np.array([int(m.jnt_limited[j]) for j in joints], np.int64),
    })
    cams = [c for c in range(m.ncam) if int(m.cam_bodyid[c]) in index]
    out.update({
        "camera_names": np.array([_name(m, obj.mjOBJ_CAMERA, c) for c in cams]),
        "camera_body": np.array([index[int(m.cam_bodyid[c])] for c in cams], np.int64),
        "camera_pos": np.array([m.cam_pos[c] for c in cams], np.float64).reshape(-1, 3),
        "camera_quat": np.array([m.cam_quat[c] for c in cams], np.float64).reshape(-1, 4),
        "camera_fovy": np.array([m.cam_fovy[c] for c in cams], np.float64),
    })
    sites = [s for s in range(m.nsite) if int(m.site_bodyid[s]) in index]
    out.update({
        "site_names": np.array([_name(m, obj.mjOBJ_SITE, s) for s in sites]),
        "site_body": np.array([index[int(m.site_bodyid[s])] for s in sites], np.int64),
        "site_pos": np.array([m.site_pos[s] for s in sites], np.float64).reshape(-1, 3),
        "site_quat": np.array([m.site_quat[s] for s in sites], np.float64).reshape(-1, 4),
    })
    geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in index]
    verts, faces, vadr, vnum, fadr, fnum = [], [], [], [], [], []
    nv = nf = 0
    for g in geoms:
        if int(m.geom_type[g]) == _MESH:
            mesh = int(m.geom_dataid[g])
            v0, n_v = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            f0, n_f = int(m.mesh_faceadr[mesh]), int(m.mesh_facenum[mesh])
            local = np.asarray(m.mesh_vert[v0:v0 + n_v], np.float64).reshape(-1, 3)
            body = local @ quat_to_matrix(m.geom_quat[g]).T + np.asarray(m.geom_pos[g], np.float64)
            verts.append(body)
            faces.append(np.asarray(m.mesh_face[f0:f0 + n_f], np.int64).reshape(-1, 3))
            vadr.append(nv), vnum.append(n_v), fadr.append(nf), fnum.append(n_f)
            nv += n_v
            nf += n_f
        else:
            vadr.append(-1), vnum.append(0), fadr.append(-1), fnum.append(0)
    out.update({
        "geom_names": np.array([_name(m, obj.mjOBJ_GEOM, g) for g in geoms]),
        "geom_body": np.array([index[int(m.geom_bodyid[g])] for g in geoms], np.int64),
        "geom_type": np.array([int(m.geom_type[g]) for g in geoms], np.int64),
        "geom_size": np.array([m.geom_size[g] for g in geoms], np.float64).reshape(-1, 3),
        "geom_pos": np.array([m.geom_pos[g] for g in geoms], np.float64).reshape(-1, 3),
        "geom_quat": np.array([m.geom_quat[g] for g in geoms], np.float64).reshape(-1, 4),
        "geom_group": np.array([int(m.geom_group[g]) for g in geoms], np.int64),
        "geom_contype": np.array([int(m.geom_contype[g]) for g in geoms], np.int64),
        "geom_conaffinity": np.array([int(m.geom_conaffinity[g]) for g in geoms], np.int64),
        "geom_vertadr": np.array(vadr, np.int64), "geom_vertnum": np.array(vnum, np.int64),
        "geom_faceadr": np.array(fadr, np.int64), "geom_facenum": np.array(fnum, np.int64),
        "mesh_vert_body": np.vstack(verts) if verts else np.zeros((0, 3)),
        "mesh_face": np.vstack(faces) if faces else np.zeros((0, 3), np.int64),
    })
    return out


#: Keys of :func:`export_tables` that describe kinematics (compared between the two sources).
KINEMATIC_KEYS = ("body_names", "body_parent", "body_pos", "body_quat", "joint_names", "joint_body", "joint_type",
                  "joint_axis", "joint_pos", "joint_range", "camera_names", "camera_body", "camera_pos", "camera_quat",
                  "camera_fovy", "site_names", "site_body", "site_pos", "site_quat")


#: Keys of :func:`export_tables` that describe geometry (geoms and meshes in body frames).
GEOMETRY_KEYS = ("geom_names", "geom_body", "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_group",
                 "geom_contype", "geom_conaffinity", "geom_vertadr", "geom_vertnum", "geom_faceadr", "geom_facenum",
                 "mesh_vert_body", "mesh_face")


def tables_digest(tables, keys=None):
    """SHA-256 over the arrays of ``tables`` (sorted keys; dtype, shape and bytes).

    The pose of ``robot0_base`` itself in the world (where the env places the robot) is not part of
    the robot: its ``body_pos`` / ``body_quat`` row is replaced by the identity before hashing.
    """
    tables = dict(tables)
    if "body_names" in tables:
        base = [k for k, name in enumerate(np.asarray(tables["body_names"]).tolist()) if name == BASE_BODY]
        for key, identity in (("body_pos", (0.0, 0.0, 0.0)), ("body_quat", (1.0, 0.0, 0.0, 0.0))):
            if key in tables and base:
                value = np.array(tables[key], np.float64, copy=True)
                value[base[0]] = identity
                tables[key] = value
    h = hashlib.sha256()
    for key in sorted(keys or tables):
        value = np.ascontiguousarray(tables[key])
        h.update(key.encode())
        h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def pose(position, rotation):
    T = np.eye(4)
    T[:3, :3] = np.asarray(rotation, np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(position, np.float64).reshape(3)
    return T


def fk_poses(m, d, q9, robot_qpos=None):
    """Robot body / site / camera poses in the ``robot0_base`` frame at robot joint values ``q9``.

    ``q9`` is in :data:`ROBOT_JOINTS` order (MuJoCo qpos, i.e. the adapter ``robot_state``). Every other
    qpos keeps its current value. Cameras are given with OpenCV axes (``cam_xmat @ diag(1,-1,-1)``,
    as in ``camera.world_camera_pose_opencv``). ``d`` is modified (``mj_forward``).
    """
    import mujoco

    obj = mujoco.mjtObj
    if robot_qpos is None:
        robot_qpos = [int(m.jnt_qposadr[mujoco.mj_name2id(m, obj.mjOBJ_JOINT, n)]) for n in ROBOT_JOINTS]
    d.qpos[robot_qpos] = np.asarray(q9, np.float64)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    base = mujoco.mj_name2id(m, obj.mjOBJ_BODY, BASE_BODY)
    T_world_base = pose(d.xpos[base], d.xmat[base])
    T_base_world = np.linalg.inv(T_world_base)
    bodies = robot_body_ids(m)
    sites = [s for s in range(m.nsite) if int(m.site_bodyid[s]) in set(bodies)]
    cams = [c for c in range(m.ncam) if int(m.cam_bodyid[c]) in set(bodies)]
    flip = np.diag([1.0, -1.0, -1.0])
    return {
        "body": np.stack([T_base_world @ pose(d.xpos[b], d.xmat[b]) for b in bodies]),
        "site": np.stack([T_base_world @ pose(d.site_xpos[s], d.site_xmat[s]) for s in sites]),
        "camera_opencv": np.stack([T_base_world @ pose(d.cam_xpos[c], np.asarray(d.cam_xmat[c]).reshape(3, 3) @ flip)
                                   for c in cams]),
        "body_names": np.array([_name(m, obj.mjOBJ_BODY, b) for b in bodies]),
        "site_names": np.array([_name(m, obj.mjOBJ_SITE, s) for s in sites]),
        "camera_names": np.array([_name(m, obj.mjOBJ_CAMERA, c) for c in cams]),
    }


def fk_fixture(m, qs=FK_FIXTURE_Q):
    import mujoco

    d = mujoco.MjData(m)
    rows = [fk_poses(m, d, q) for q in qs]
    out = {"q": np.asarray(qs, np.float64), "q_layout": np.array(ROBOT_JOINTS)}
    for key in ("body", "site", "camera_opencv"):
        out["T_base_" + key] = np.stack([r[key] for r in rows])
    for key in ("body_names", "site_names", "camera_names"):
        out[key] = rows[0][key]
    return out


# --- Model sources -------------------------------------------------------------------------------
def robosuite_robot_model():
    """The LIBERO robot alone (no arena, no objects): raw ``mujoco.MjModel`` (+ the XML string)."""
    import mujoco
    import libero.libero.envs.robots  # noqa: F401  registers MountedPanda
    from robosuite.models import MujocoWorldBase
    from robosuite.models.grippers import gripper_factory
    from robosuite.models.mounts import mount_factory
    from robosuite.models.robots import create_robot

    robot = create_robot("MountedPanda", idn=0)
    robot.add_mount(mount=mount_factory(robot.default_mount, idn=0))
    robot.add_gripper(gripper_factory(robot.default_gripper, idn=0))
    world = MujocoWorldBase()
    world.merge(robot)
    xml = world.get_xml()
    return mujoco.MjModel.from_xml_string(xml), xml


def libero_env_model(suite="libero_spatial", task_id=0, seed=7, resolution=256):
    """The model of a real LIBERO env (the SL0/SL1 task). Returns ``(MjModel, env)``; close the env."""
    root = pathlib.Path(__file__).resolve().parents[3]
    for path in (str(root), str(root / "benchmark/runners")):
        if path not in sys.path:
            sys.path.append(path)
    import libero_pi05_client as client
    from libero.libero import benchmark

    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    env, _ = client._get_libero_env_with_depth(task, resolution, seed)
    env.reset()
    return env.sim.model._model, env


def provenance(source, m):
    import mujoco
    import robosuite

    info = {"source": source, "mujoco": mujoco.__version__, "robosuite": robosuite.__version__,
            "robosuite_path": str(pathlib.Path(robosuite.__file__).resolve().parent), "python": sys.version.split()[0],
            "numpy": np.__version__, "integrator": int(m.opt.integrator), "timestep": float(m.opt.timestep)}
    if source == "libero":
        info.update({"suite": "libero_spatial", "task_id": 0, "seed": 7})
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("export", "fk"))
    parser.add_argument("--source", choices=("libero", "robosuite"), default="robosuite")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit("refusing to overwrite {}".format(out))
    env = None
    if args.source == "libero":
        m, env = libero_env_model()
    else:
        m, _ = robosuite_robot_model()
    arrays = export_tables(m) if args.what == "export" else fk_fixture(m)
    tables = export_tables(m)
    arrays["kinematic_digest"] = np.array(tables_digest(tables, KINEMATIC_KEYS))
    arrays["geometry_digest"] = np.array(tables_digest(tables, GEOMETRY_KEYS))
    arrays["provenance_json"] = np.array(json.dumps(provenance(args.source, m), sort_keys=True))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), **arrays)
    print(json.dumps({"out": str(out), "bytes": out.stat().st_size, "kinematic_digest": str(arrays["kinematic_digest"]),
                      "geometry_digest": str(arrays["geometry_digest"]), "bodies": int(len(tables["body_names"]))}))
    sys.stdout.flush()
    if env is not None:
        env.close()
    # MuJoCo/OSMesa teardown in this interpreter can abort at exit after a successful run; the work is done.
    os._exit(0)


if __name__ == "__main__":
    main()
