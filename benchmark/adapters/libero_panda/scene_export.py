"""LIBERO scene ground truth at the image instant of recorded frames (SL4). Python 3.8 compatible.

For every episode of a geometry-capture run (``launch_libero_pi05_only.py --capture-geometry``) this
rebuilds the LIBERO env of the episode, restores the recorded **image-instant** simulator state
(``_geom.npz`` ``sim_state_image``, SL2) at the selected frames and writes the MuJoCo facts the
Python 3.11 side (``benchmark/evaluators/libero_scene_gt.py``) needs to build surface samples and exact
distances without LIBERO:

* every geom's world pose (``geom_xpos`` / ``geom_xmat``) and every body's, per frame;
* ``T_world_base`` (``robot0_base``) per frame, checked against the recorded one;
* static model tables: geom type / size / group / contype / conaffinity / mesh id / body / name, body
  names / root / parent, and the meshes of every **non-robot** geom (mount included) -- vertices in the
  geom frame, faces local to the mesh;
* MuJoCo contacts at the restored state, and ``mj_geomDistance`` between every robot collision geom and
  every non-robot collision geom closer than ``--distmax`` (MuJoCo's own number; mesh geoms are convex
  hulls there -- the 3.11 side computes exact triangle distances);
* the BDDL target / destination / object roles (``task_target.task_target``).

The run is only read. Default frames are the policy-call frames (``obs_index = chunk_t[k]``).

Env history (SL4b fix): the env of each episode is the rollout's -- one env per task block and one
``env.reset()`` + ``set_init_state`` per preceding episode of that task, in the run's recorded order
(``env_history.episode_envs``), because LIBERO re-samples the fixture placement (cabinet, stove: model
``body_pos``, not in the recorded state) on every reset. Hard check: every exported policy-call frame
re-renders RGB **bit-identical** to the recorded frame on both cameras (``checks.rgb_rerender_equal_call_frames``),
otherwise the episode raises and nothing is written.

CLI (``.venv-libero``; re-executes itself with ``client_env.client_environment`` like
``robot_geometry.py``)::

    .venv-libero/bin/python benchmark/adapters/libero_panda/scene_export.py \\
        --run <results/benchmark/...dev run> --out <dir> [--task-ids 0,1] [--frames calls|all] [--limit-frames N]

writes ``<out>/scene_<suite>_t<task>_e<episode>.npz`` (exclusive) and ``<out>/scene_export.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

if __name__ == "__main__":
    import client_env  # this script's directory is sys.path[0]

    client_env.reexec_with_client_environment()

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
ROBOT_PREFIXES = ("robot0_", "gripper0_")
MOUNT_PREFIX = "mount0_"
BASE_BODY = "robot0_base"
#: Gripper bodies (the constraint links panda_hand / fingers live here; robosuite names).
GRIPPER_BODY_PREFIXES = ("gripper0_", "robot0_right_hand")
DISTMAX = 0.40
SCHEMA = 2  # 2: env history of the run reproduced (SL4b fix), RGB equality enforced


def _sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _imports():
    for path in (str(PROJECT_ROOT), str(PROJECT_ROOT / "benchmark/runners")):
        if path not in sys.path:
            sys.path.append(path)


def episode_rows(run):
    rows = [json.loads(l) for l in (pathlib.Path(run) / "episodes.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for row in rows:
        stem = "{}_t{}_e{}".format(row["suite"], int(row["task_id"]), int(row["episode_idx"]))
        out.append({"suite": row["suite"], "task_id": int(row["task_id"]), "episode_idx": int(row["episode_idx"]),
                    "stem": stem, "npz": row["npz"], "geom_npz": "episodes/{}_geom.npz".format(stem)})
    return out


def pose(p, m):
    T = np.eye(4)
    T[:3, :3] = np.asarray(m, np.float64).reshape(3, 3)
    T[:3, 3] = p
    return T


def model_tables(m):
    """Static tables (all geoms / bodies) + the meshes of non-robot geoms."""
    bnames = [m.body(k).name for k in range(m.nbody)]
    gnames = [m.geom(k).name for k in range(m.ngeom)]
    gb = np.asarray(m.geom_bodyid).copy()
    robot_body = np.array([n.startswith(ROBOT_PREFIXES) for n in bnames])
    robot_geom = robot_body[gb]
    keep_mesh = sorted(set(int(m.geom_dataid[k]) for k in range(m.ngeom)
                           if int(m.geom_type[k]) == 7 and not robot_geom[k] and int(m.geom_dataid[k]) >= 0))
    vert_adr = np.full(m.nmesh, -1, np.int64)
    vert_num = np.zeros(m.nmesh, np.int64)
    face_adr = np.full(m.nmesh, -1, np.int64)
    face_num = np.zeros(m.nmesh, np.int64)
    verts, faces, nv, nf = [], [], 0, 0
    for mid in keep_mesh:
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
        verts.append(np.asarray(m.mesh_vert[va:va + vn], np.float64))
        faces.append(np.asarray(m.mesh_face[fa:fa + fn], np.int64))
        vert_adr[mid], vert_num[mid], face_adr[mid], face_num[mid] = nv, vn, nf, fn
        nv += vn
        nf += fn
    coll = (np.asarray(m.geom_contype) | np.asarray(m.geom_conaffinity)) != 0
    gripper_body = np.array([n.startswith(GRIPPER_BODY_PREFIXES) for n in bnames])
    return {
        "geom_names": np.array(gnames), "geom_type": np.asarray(m.geom_type).copy(),
        "geom_size": np.asarray(m.geom_size).copy(), "geom_group": np.asarray(m.geom_group).copy(),
        "geom_contype": np.asarray(m.geom_contype).copy(), "geom_conaffinity": np.asarray(m.geom_conaffinity).copy(),
        "geom_dataid": np.asarray(m.geom_dataid).copy(), "geom_bodyid": gb, "geom_rgba": np.asarray(m.geom_rgba).copy(),
        "body_names": np.array(bnames), "body_rootid": np.asarray(m.body_rootid).copy(),
        "body_parentid": np.asarray(m.body_parentid).copy(),
        # the names task_target.scene_instances reads (same as the _geom.npz tables)
        "model_geom_bodyid": gb, "model_body_names": np.array(bnames),
        "mesh_vert": (np.concatenate(verts) if verts else np.zeros((0, 3))),
        "mesh_face": (np.concatenate(faces) if faces else np.zeros((0, 3), np.int64)),
        "mesh_vertadr": vert_adr, "mesh_vertnum": vert_num, "mesh_faceadr": face_adr, "mesh_facenum": face_num,
        "robot_geoms": np.flatnonzero(robot_geom), "robot_collision_geoms": np.flatnonzero(robot_geom & coll),
        "gripper_collision_geoms": np.flatnonzero(gripper_body[gb] & coll),
        "object_collision_geoms": np.flatnonzero(~robot_geom & coll),
        "mount_geoms": np.array([k for k in range(m.ngeom) if bnames[gb[k]].startswith(MOUNT_PREFIX)], np.int64),
    }


def export_episode(run, ep, out_dir, env, history, frames="calls", limit_frames=None, distmax=DISTMAX):
    """Export one episode from ``env`` = the task's env right after the rollout's ``start_episode`` of it
    (``env_history.episode_envs``; ``history`` = its provenance note)."""
    _imports()
    import mujoco
    from benchmark.adapters.libero_panda import camera, env_history, task_target as tt

    run = pathlib.Path(run)
    target = pathlib.Path(out_dir) / "scene_{}.npz".format(ep["stem"])
    if target.exists():
        raise SystemExit("refusing to overwrite {}".format(target))
    sim = env.sim
    m, d = sim.model._model, sim.data._data
    e = np.load(str(run / ep["npz"]))
    g = np.load(str(run / ep["geom_npz"]))
    tables = model_tables(m)
    if list(map(str, g["model_body_names"])) != list(map(str, tables["body_names"])):
        raise ValueError("{}: body table differs from the record".format(ep["stem"]))
    if not np.array_equal(np.asarray(g["model_geom_bodyid"]), tables["geom_bodyid"]):
        raise ValueError("{}: geom table differs from the record".format(ep["stem"]))
    obs_index = np.asarray(g["obs_index"]).reshape(-1)
    call_set = set(int(t) for t in np.asarray(e["chunk_t"]).reshape(-1))
    if frames == "calls":
        wanted = [int(t) for t in np.asarray(e["chunk_t"]).reshape(-1)]
    else:
        wanted = [int(t) for t in obs_index]
    if limit_frames:
        wanted = wanted[:int(limit_frames)]
    row_of = {int(o): i for i, o in enumerate(obs_index)}
    rows = [row_of[t] for t in wanted]
    F = len(rows)
    base_id = m.body(BASE_BODY).id
    out = {"frames": np.array(wanted, np.int64), "geom_rows": np.array(rows, np.int64),
           "geom_xpos": np.zeros((F, m.ngeom, 3)), "geom_xmat": np.zeros((F, m.ngeom, 9)),
           "body_xpos": np.zeros((F, m.nbody, 3)), "body_xmat": np.zeros((F, m.nbody, 9)),
           "T_world_base": np.zeros((F, 4, 4)), "robot_state": np.zeros((F, 9))}
    contacts, gdist = [], []
    check = {"T_world_base_vs_record_max_abs": 0.0, "robot_qpos_vs_robot_state_max_abs": 0.0}
    rgb_equal = {source: 0 for source, _ in env_history.RGB_CHECK_CAMERAS}
    rgb_checked = 0
    qidx = np.asarray(g["robot_qpos_indexes"])
    fromto = np.zeros(6)
    rgeoms = [int(k) for k in tables["robot_collision_geoms"]]
    ogeoms = [int(k) for k in tables["object_collision_geoms"]]
    for f, i in enumerate(rows):
        sim.set_state_from_flattened(np.asarray(g["sim_state_image"][i]).copy())
        sim.forward()
        out["geom_xpos"][f] = d.geom_xpos
        out["geom_xmat"][f] = d.geom_xmat
        out["body_xpos"][f] = d.xpos
        out["body_xmat"][f] = d.xmat
        out["T_world_base"][f] = pose(d.xpos[base_id], d.xmat[base_id])
        out["robot_state"][f] = np.asarray(g["robot_state/head"][i])
        check["T_world_base_vs_record_max_abs"] = max(check["T_world_base_vs_record_max_abs"], float(
            np.abs(out["T_world_base"][f] - np.asarray(g["T_world_base"][i])).max()))
        check["robot_qpos_vs_robot_state_max_abs"] = max(check["robot_qpos_vs_robot_state_max_abs"], float(
            np.abs(d.qpos[qidx] - out["robot_state"][f]).max()))
        for k in range(d.ncon):
            c = d.contact[k]
            contacts.append((f, int(c.geom1), int(c.geom2), float(c.dist)))
        for rg in rgeoms:
            for og in ogeoms:
                dist = mujoco.mj_geomDistance(m, d, rg, og, distmax, fromto)
                if dist < distmax:
                    gdist.append((f, rg, og, float(dist)) + tuple(float(x) for x in fromto))
        # after the MuJoCo reads above: rendering never feeds them
        if wanted[f] in call_set:
            rgb_checked += 1
            for source, equal in env_history.rgb_rerender_equal(sim, e, wanted[f]).items():
                rgb_equal[source] += int(equal)
    env_history.require_rgb_equal(ep["stem"], rgb_equal, rgb_checked)
    wire = dict(camera.CAMERA_TABLE)
    check["rgb_rerender_checked_call_frames"] = rgb_checked
    check["rgb_rerender_equal_call_frames"] = {wire[k]: v for k, v in rgb_equal.items()}
    out["contacts"] = np.array(contacts, np.float64).reshape(-1, 4)
    out["geomdist"] = np.array(gdist, np.float64).reshape(-1, 10)
    out["geomdist_distmax"] = np.array(distmax)
    out.update(tables)
    info = tt.task_target(ep["suite"], ep["task_id"])
    info["instances_geom_ids"] = tt.scene_instances(tables, info)
    out["target_json"] = np.array(json.dumps(info, sort_keys=True))
    prov = {"schema": SCHEMA, "run": str(run), "run_id": run.name, "stem": ep["stem"], "frame_selection": frames,
            "limit_frames": limit_frames, "npz": ep["npz"], "npz_sha256": _sha256(run / ep["npz"]),
            "geom_npz": ep["geom_npz"], "geom_npz_sha256": _sha256(run / ep["geom_npz"]),
            "mujoco": mujoco.__version__, "python": sys.version.split()[0], "distmax": distmax,
            "state": "sim_state_image (image instant, SL2)", "checks": check, "env_history": history,
            "camera_conventions": camera.conventions() if hasattr(camera, "conventions") else None}
    out["provenance_json"] = np.array(json.dumps(prov, sort_keys=True))
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(str(target), "xb") as stream:
        np.savez_compressed(stream, **out)
    return {"file": target.name, "sha256": _sha256(target), "frames": F, "contacts": len(contacts),
            "geomdist_rows": len(gdist), "target": info["target"], "destination": info["destination"], **prov}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="geometry-capture run directory (read only)")
    parser.add_argument("--out", required=True, help="output directory (files are created exclusively)")
    parser.add_argument("--task-ids", default=None, help="comma list (default: every episode)")
    parser.add_argument("--frames", choices=("calls", "all"), default="calls")
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--distmax", type=float, default=DISTMAX)
    args = parser.parse_args(argv)
    run, out = pathlib.Path(args.run).resolve(), pathlib.Path(args.out)
    if (out / "scene_export.json").exists():
        raise SystemExit("refusing to overwrite {}".format(out / "scene_export.json"))
    _imports()
    from benchmark.adapters.libero_panda import env_history

    tasks = None if args.task_ids is None else set(int(x) for x in args.task_ids.split(",") if x.strip())
    eps = {ep["stem"]: ep for ep in episode_rows(run) if tasks is None or ep["task_id"] in tasks}
    env_history.run_history(run)  # refuse an unknown order before creating anything
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = []
    for row, env, history in env_history.episode_envs(run, set(eps)):
        ep = eps[env_history.stem_of(row)]
        res = export_episode(run, ep, out, env, history, args.frames, args.limit_frames, args.distmax)
        res["seconds"] = time.time() - t0
        results.append(res)
        line = {k: res[k] for k in ("file", "frames", "contacts", "geomdist_rows", "target", "checks", "seconds")}
        line["resets_before"] = history["resets_before"]
        print(json.dumps(line), flush=True)
    summary = {"schema": SCHEMA, "run": str(run), "argv": sys.argv, "episodes": results, "seconds": time.time() - t0}
    with open(str(out / "scene_export.json"), "x") as stream:
        stream.write(json.dumps(summary, indent=1) + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
