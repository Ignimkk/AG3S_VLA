"""Phase track + image-instant target segmentation of recorded LIBERO episodes (SL4b). Python 3.8.

For every episode of a geometry-capture run this rebuilds the LIBERO env, restores the recorded
**image-instant** simulator state (``_geom.npz`` ``sim_state_image``, SL2) at **every** recorded frame,
``mj_forward`` and writes what the phase labels need (``benchmark/evaluators/libero_phases.py``), with the
contact rule of the SL3 verifier export (``outputs/verify/SL3/scripts/export_gt_dev.py``):

* ``target_xpos (N, 3)`` -- world position of the target body ``<target>_main`` (BDDL target, ``task_target``);
* ``contact_grip_target (N,)`` / ``n_contact_grip_target`` -- a MuJoCo contact pairs a target geom with a geom
  of a ``gripper0_*`` body; ``contact_robot_target`` -- with any ``robot0_`` / ``gripper0_`` / ``mount0_`` body;
* ``frames (N,)`` (= ``obs_index``), ``call_frames`` (sorted policy-call frames), ``chunk_t``, ``target_geoms``;
* ``segP/<head|left_wrist> (C, 256, 256) int16`` -- per-pixel geom id (-1 = not a geom, OpenCV rows) re-rendered
  at the image instant of every policy-call frame (row ``c`` = ``call_frames[c]``), exactly as the SL3 export
  (``sim.render(..., segmentation=True)`` after restoring ``sim_state_image``). This is the segmentation the
  recorded RGB / depth show: the rollout's own ``segmentation_geom/<cam>`` is rendered at the *return-time*
  state (``camera.capture`` docstring; SL3: 100-300 px per frame differ). ``--no-segmentation`` skips it.

Env history (SL4b fix): the env of each episode is the rollout's -- one env per task block and one
``env.reset()`` + ``set_init_state`` per preceding episode of that task, in the run's recorded order
(``env_history.episode_envs``), because LIBERO re-samples the fixture placement on every reset. Hard check:
every policy-call frame re-renders RGB **bit-identical** to the recorded frame on both cameras
(``rgb_rerender_equal_call_frames``), otherwise the episode raises and nothing is written.

The run is only read; output files are created exclusively. Same keys as the SL3 ``gt_<stem>.npz`` for what
the phase labels, the selection metric and the oracle read (``target_xpos``, ``contact_grip_target``,
``call_frames``, ``chunk_t``, ``segP/<cam>``).

SL6 B3 (``--finger-contacts``, default off -- without it the file is the SL4b one, key for key): per recorded frame
also ``contact_left_finger_target`` / ``contact_right_finger_target`` (a target geom paired with a geom of the
Panda **left** finger bodies ``gripper0_leftfinger`` + ``gripper0_finger_joint1_tip`` / the **right** finger bodies
``gripper0_rightfinger`` + ``gripper0_finger_joint2_tip``, :data:`FINGER_BODIES`), ``n_contact_target_env`` (target
geom paired with a geom that is neither the target nor a robot body: the support surface or another object),
``destination_xy_box (N, 4)`` ``[xmin, ymin, xmax, ymax]`` (world AABB of the BDDL destination's collision geoms,
``mjModel.geom_aabb`` corners through ``geom_xpos`` / ``geom_xmat``) and ``destination_xpos`` (``<destination>_main``).

CLI (``.venv-libero``; re-executes itself with ``client_env.client_environment``)::

    .venv-libero/bin/python benchmark/adapters/libero_panda/phase_export.py --run <run dir> --out <dir> [--task-ids 0,1]

writes ``<out>/phase_<suite>_t<task>_e<episode>.npz`` and ``<out>/phase_export.json`` (about 20 s per episode on CPU).
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
GRIPPER_PREFIX = "gripper0_"
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")
SCHEMA = 3  # 3: env history of the run reproduced (SL4b fix), RGB equality enforced
#: (robosuite camera, wire name) -- ``camera.CAMERA_TABLE``; 256 x 256 = the rollout resolution.
SEG_CAMERAS = (("agentview", "head"), ("robot0_eye_in_hand", "left_wrist"))
SEG_RESOLUTION = 256
#: SL6 B3 -- robosuite ``PandaGripper`` finger bodies (``_geom.npz`` ``model_body_names``): left = finger 1, right = 2.
FINGER_BODIES = {"left": ("gripper0_leftfinger", "gripper0_finger_joint1_tip"),
                 "right": ("gripper0_rightfinger", "gripper0_finger_joint2_tip")}
FINGER_SCHEMA = "sl6-b3-1"


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


def contact_flags(contacts, target_geoms, gripper_geom, robot_geom):
    """SL3 rule over ``(geom1, geom2)`` pairs: (grip-target any, grip-target count, robot-target any, target-other count)."""
    grip, n_grip, robot, n_any = False, 0, False, 0
    for g1, g2 in contacts:
        for a, b in ((g1, g2), (g2, g1)):
            if a in target_geoms and b not in target_geoms:
                n_any += 1
                if gripper_geom[b]:
                    grip = True
                    n_grip += 1
                if robot_geom[b]:
                    robot = True
    return grip, n_grip, robot, n_any


def finger_flags(contacts, target_geoms, finger_geom, robot_geom):
    """SL6 B3 over ``(geom1, geom2)`` pairs: (left finger-target any, right finger-target any, target-env count).
    ``finger_geom``: per geom ``"left"`` / ``"right"`` / ``""``; env = neither target nor robot."""
    left, right, n_env = False, False, 0
    for g1, g2 in contacts:
        for a, b in ((g1, g2), (g2, g1)):
            if a in target_geoms and b not in target_geoms:
                if finger_geom[b] == "left":
                    left = True
                elif finger_geom[b] == "right":
                    right = True
                if not robot_geom[b]:
                    n_env += 1
    return left, right, n_env


def geoms_xy_box(m, d, geoms):
    """``[xmin, ymin, xmax, ymax]`` of the world AABB of ``geoms`` (``geom_aabb`` box corners in world)."""
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
    for g in geoms:
        box = np.asarray(m.geom_aabb[g], np.float64)
        corners = box[:3] + signs * box[3:]
        world = np.asarray(d.geom_xpos[g]) + corners @ np.asarray(d.geom_xmat[g]).reshape(3, 3).T
        lo = np.minimum(lo, world[:, :2].min(0))
        hi = np.maximum(hi, world[:, :2].max(0))
    return np.concatenate([lo, hi])


def render_segmentation(sim, source, camera):
    """Geom-id segmentation of ``source`` at the current sim state (OpenCV rows, -1 = not a geom); the SL3 rule."""
    s = camera.opengl_rows_to_opencv(sim.render(camera_name=source, height=SEG_RESOLUTION, width=SEG_RESOLUTION,
                                                segmentation=True))
    return np.where(s[..., 0] == camera.MJOBJ_GEOM, s[..., 1], -1).astype(np.int16)


def export_episode(run, ep, out_dir, env, history, segmentation=True, finger_contacts=False):
    """Export one episode from ``env`` = the task's env right after the rollout's ``start_episode`` of it
    (``env_history.episode_envs``; ``history`` = its provenance note)."""
    _imports()
    import mujoco
    from benchmark.adapters.libero_panda import camera, env_history, task_target as tt

    run = pathlib.Path(run)
    target_path = pathlib.Path(out_dir) / "phase_{}.npz".format(ep["stem"])
    if target_path.exists():
        raise SystemExit("refusing to overwrite {}".format(target_path))
    info = tt.task_target(ep["suite"], ep["task_id"])
    sim = env.sim
    m, d = sim.model._model, sim.data._data
    e = np.load(str(run / ep["npz"]))
    g = np.load(str(run / ep["geom_npz"]))
    bnames = [m.body(k).name for k in range(m.nbody)]
    if list(map(str, g["model_body_names"])) != bnames:
        raise ValueError("{}: body table differs from the record".format(ep["stem"]))
    gb = np.asarray(m.geom_bodyid)
    if not np.array_equal(np.asarray(g["model_geom_bodyid"]), gb):
        raise ValueError("{}: geom table differs from the record".format(ep["stem"]))
    target_geoms = set(int(x) for x in tt.instance_geom_ids(info["target"], gb, bnames))
    gripper_geom = np.array([bnames[b].startswith(GRIPPER_PREFIX) for b in gb])
    robot_geom = np.array([bnames[b].startswith(ROBOT_PREFIXES) for b in gb])
    tbody = m.body(info["target"] + "_main").id
    frames = np.asarray(g["obs_index"]).astype(np.int64).reshape(-1)
    N = len(frames)
    chunk_t = np.asarray(e["chunk_t"]).astype(np.int64).reshape(-1)
    call_frames = sorted(set(int(x) for x in chunk_t))
    call_row = {f: c for c, f in enumerate(call_frames)}
    out = {"frames": frames, "target_xpos": np.zeros((N, 3)), "contact_grip_target": np.zeros(N, bool),
           "n_contact_grip_target": np.zeros(N, np.int64), "contact_robot_target": np.zeros(N, bool),
           "contact_target_any": np.zeros(N, np.int64)}
    if finger_contacts:  # SL6 B3 -- extra keys only with the flag (the SL4b file otherwise)
        finger_of_body = {}
        for side, names in FINGER_BODIES.items():
            for name in names:
                finger_of_body[bnames.index(name)] = side
        finger_geom = np.array([finger_of_body.get(int(b), "") for b in gb])
        dest_name = info.get("destination")
        if not dest_name:
            raise ValueError("{}: the BDDL goal has no destination".format(ep["stem"]))
        dest_all = [int(x) for x in tt.instance_geom_ids(dest_name, gb, bnames)]
        dest_coll = [g_ for g_ in dest_all if int(m.geom_contype[g_]) or int(m.geom_conaffinity[g_])]
        dest_geoms = dest_coll or dest_all
        dbody = m.body(dest_name + "_main").id
        out.update({"contact_left_finger_target": np.zeros(N, bool), "contact_right_finger_target": np.zeros(N, bool),
                    "n_contact_target_env": np.zeros(N, np.int64), "destination_xy_box": np.zeros((N, 4)),
                    "destination_xpos": np.zeros((N, 3))})
    missing = sorted(set(call_frames) - set(int(x) for x in frames))
    if missing:
        raise ValueError("{}: call frames {} are not recorded frames".format(ep["stem"], missing[:5]))
    wire_of = dict(SEG_CAMERAS)
    rgb_equal = {wire: 0 for _, wire in SEG_CAMERAS}
    seg_vs_recorded = {}
    if segmentation:
        for _, wire in SEG_CAMERAS:
            out["segP/" + wire] = np.full((len(call_frames), SEG_RESOLUTION, SEG_RESOLUTION), -1, np.int16)
            seg_vs_recorded[wire] = []
    for i in range(N):
        sim.set_state_from_flattened(np.asarray(g["sim_state_image"][i]).copy())
        sim.forward()
        out["target_xpos"][i] = d.xpos[tbody]
        pairs = [(int(d.contact[k].geom1), int(d.contact[k].geom2)) for k in range(d.ncon)]
        grip, n_grip, robot, n_any = contact_flags(pairs, target_geoms, gripper_geom, robot_geom)
        out["contact_grip_target"][i], out["n_contact_grip_target"][i] = grip, n_grip
        out["contact_robot_target"][i], out["contact_target_any"][i] = robot, n_any
        if finger_contacts:
            left, right, n_env = finger_flags(pairs, target_geoms, finger_geom, robot_geom)
            out["contact_left_finger_target"][i], out["contact_right_finger_target"][i] = left, right
            out["n_contact_target_env"][i] = n_env
            out["destination_xy_box"][i] = geoms_xy_box(m, d, dest_geoms)
            out["destination_xpos"][i] = d.xpos[dbody]
        f = int(frames[i])
        if f in call_row:
            for source, equal in env_history.rgb_rerender_equal(sim, e, f, SEG_RESOLUTION).items():
                rgb_equal[wire_of[source]] += int(equal)
        if segmentation and f in call_row:
            for source, wire in SEG_CAMERAS:
                seg = render_segmentation(sim, source, camera)
                out["segP/" + wire][call_row[f]] = seg
                key = "segmentation_geom/" + wire
                if key in g.files:
                    seg_vs_recorded[wire].append(int((seg != g[key][i]).sum()))
    env_history.require_rgb_equal(ep["stem"], rgb_equal, len(call_frames))
    out["chunk_t"] = chunk_t
    out["call_frames"] = np.array(call_frames, np.int64)
    out["target_geoms"] = np.array(sorted(target_geoms), np.int64)
    prov = {"schema": SCHEMA, "run": str(run), "run_id": run.name, "stem": ep["stem"], "target": info["target"],
            "npz": ep["npz"], "npz_sha256": _sha256(run / ep["npz"]), "geom_npz": ep["geom_npz"],
            "geom_npz_sha256": _sha256(run / ep["geom_npz"]), "mujoco": mujoco.__version__,
            "python": sys.version.split()[0], "state": "sim_state_image (image instant, SL2), every recorded frame",
            "contact_rule": "SL3 export_gt_dev: target geom paired with a gripper0_* body geom",
            "env_history": history,
            "rgb_rerender_equal_call_frames": rgb_equal,
            "segmentation": (None if not segmentation else
                             "segP/<cam>: geom ids re-rendered at sim_state_image of each call frame (row = index in "
                             "call_frames), OpenCV rows, -1 = not a geom (SL3 export_gt_dev rule)")}
    if finger_contacts:
        out["destination_geoms"] = np.array(dest_geoms, np.int64)
        prov["finger_contacts"] = {
            "schema": FINGER_SCHEMA, "finger_bodies": {k: list(v) for k, v in FINGER_BODIES.items()},
            "destination": dest_name, "destination_geoms": dest_geoms,
            "destination_geoms_kind": "collision (contype or conaffinity != 0)" if dest_coll else "all (no collision geom)",
            "rule": "left/right: target geom paired with a geom of that finger's bodies; n_contact_target_env: target "
                    "geom paired with a geom that is neither target nor robot0_/gripper0_/mount0_; destination_xy_box: "
                    "world AABB (geom_aabb corners) of destination_geoms"}
    out["provenance_json"] = np.array(json.dumps(prov, sort_keys=True))
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(target_path), "xb") as stream:
        np.savez_compressed(stream, **out)
    checks = {"call_frames": len(call_frames), "rgb_rerender_equal_call_frames": rgb_equal}
    if segmentation:
        checks["segP_vs_recorded_diff_px"] = {w: {"sum": int(sum(v)), "frames_nonzero": int(sum(x > 0 for x in v)),
                                                  "max": int(max(v) if v else 0)} for w, v in seg_vs_recorded.items()}
    return {"file": target_path.name, "sha256": _sha256(target_path), "frames": N, "checks": checks,
            "frames_grip_target_contact": int(out["contact_grip_target"].sum()),
            "target_z_initial": float(out["target_xpos"][0, 2]), "target_z_max": float(out["target_xpos"][:, 2].max()),
            **prov}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", required=True, help="geometry-capture run directory (read only)")
    parser.add_argument("--out", required=True, help="output directory (files are created exclusively)")
    parser.add_argument("--task-ids", default=None, help="comma list (default: every episode)")
    parser.add_argument("--no-segmentation", action="store_true",
                        help="skip the image-instant segmentation of the call frames (phases only)")
    parser.add_argument("--finger-contacts", action="store_true",
                        help="SL6 B3: also per-finger target contact, target-env contact count, destination xy box")
    args = parser.parse_args(argv)
    run, out = pathlib.Path(args.run).resolve(), pathlib.Path(args.out)
    if (out / "phase_export.json").exists():
        raise SystemExit("refusing to overwrite {}".format(out / "phase_export.json"))
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
        res = export_episode(run, ep, out, env, history, segmentation=not args.no_segmentation,
                             finger_contacts=args.finger_contacts)
        res["seconds"] = time.time() - t0
        results.append(res)
        line = {k: res[k] for k in ("file", "frames", "frames_grip_target_contact", "target", "checks", "seconds")}
        line["resets_before"] = history["resets_before"]
        print(json.dumps(line), flush=True)
    summary = {"schema": SCHEMA, "run": str(run), "argv": sys.argv, "episodes": results, "seconds": time.time() - t0}
    if args.finger_contacts:
        summary["finger_schema"] = FINGER_SCHEMA
    with open(str(out / "phase_export.json"), "x") as stream:
        stream.write(json.dumps(summary, indent=1) + "\n")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
