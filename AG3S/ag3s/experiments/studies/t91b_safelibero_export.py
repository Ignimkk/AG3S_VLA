"""T91b — SafeLIBERO ground-truth phase export (physics re-render of the recorded VLA-only rollout). **Python 3.8.**

Needs robosuite / LIBERO, which only the benchmark checkout's ``.venv-libero`` has. It reads the benchmark checkout
(its ``libero_pi05_client`` / ``libero_safe_client`` / ``task_target`` / ``client_env`` modules and the SafeLIBERO fork)
**read only** and writes nothing there: the output is one ``phase_<stem>.npz`` per episode under ``OUT`` (a new directory).
The structure follows the benchmark's own ``experiments/segmentation/capture_safelibero_dataset.py`` (env history: one env
per task, ``start_episode`` per episode in order, because LIBERO re-samples fixture placement on every reset) and its
``benchmark/adapters/libero_panda/phase_export.py --finger-contacts`` (finger / target contact rule).

For every recorded frame it restores ``sim_state_image`` (the state the recorded RGB / depth show), ``forward()``s and stores

  target_xpos (N,3)                  world position of the BDDL target body ``<target>_main``
  contact_left_finger_target (N,)    a target geom is paired with a geom of the Panda left finger bodies
  contact_right_finger_target (N,)   ... right finger bodies
  destination_xy_box (N,4)           world AABB [xmin, ymin, xmax, ymax] of the destination's collision geoms
  eef_pos (N,3)                      ``robot0_eef_pos`` of the record (world)
  call_frames (C,)                   policy-call frames (``chunk_t``)

RGB of both cameras is checked bit-identical at every policy-call frame (the env history is the rollout's), otherwise the
episode raises and nothing is written. The phase rule that turns these into 5 stages lives in `t91b_safelibero.py`.

usage (the benchmark checkout's interpreter; BENCH defaults to /mnt/dev/AG3S_VLA-benchmark)::

    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 nice -n 10 $BENCH/.venv-libero/bin/python \\
        AG3S/ag3s/experiments/studies/t91b_safelibero_export.py OUT [task ids...] [--run RUN] [--bench BENCH]
"""
from __future__ import print_function

import json
import os
import pathlib
import sys
import time
import traceback

BENCH = pathlib.Path(os.environ.get("T91B_BENCH", "/mnt/dev/AG3S_VLA-benchmark"))
RUN = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark/"
                   "20261017T151126_38722284e2fa_sl5f-safelibero-LI-dev-LI-vla_only")
CAMS = (("head", "agentview", "obs_agentview_image"), ("left_wrist", "robot0_eye_in_hand", "obs_robot0_eye_in_hand_image"))
FINGER_BODIES = {"left": ("gripper0_leftfinger", "gripper0_finger_joint1_tip"),
                 "right": ("gripper0_rightfinger", "gripper0_finger_joint2_tip")}
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")


def _argv():
    args = sys.argv[1:]
    run, bench = RUN, BENCH
    for flag in ("--run", "--bench"):
        if flag in args:
            i = args.index(flag)
            val = pathlib.Path(args[i + 1])
            run, bench = (val, bench) if flag == "--run" else (run, val)
            del args[i:i + 2]
    return args, run, bench


def geoms_xy_box(m, d, geoms):
    """World AABB [xmin, ymin, xmax, ymax] of geoms (`geom_aabb` corners through geom_xpos / geom_xmat)."""
    import numpy as np

    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
    for g in geoms:
        box = np.asarray(m.geom_aabb[g], np.float64)
        corners = box[:3] + signs * box[3:]
        world = np.asarray(d.geom_xpos[g]) + corners.dot(np.asarray(d.geom_xmat[g]).reshape(3, 3).T)
        lo = np.minimum(lo, world[:, :2].min(0))
        hi = np.maximum(hi, world[:, :2].max(0))
    return np.concatenate([lo, hi])


def main():
    args, run, bench = _argv()
    if not args:
        raise SystemExit("usage: t91b_safelibero_export.py OUT [task ids...] [--run RUN] [--bench BENCH]")
    out = pathlib.Path(args[0])
    tasks = [int(x) for x in args[1:]] or [0, 1, 2, 3]
    sys.path.insert(0, str(bench))
    sys.path.insert(0, str(bench / "benchmark/runners"))
    from benchmark.adapters.libero_panda import client_env

    if client_env.needs_reexec():
        client_env.reexec_with_client_environment()
    import numpy as np
    import libero_pi05_client as base
    import libero_safe_client as safe

    safe.activate_safelibero()
    from libero.libero import benchmark
    from benchmark.adapters.libero_panda import task_target as tt

    out.mkdir(parents=True, exist_ok=False)
    suite = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="I")
    rows = [json.loads(l) for l in (run / "episodes.jsonl").read_text().splitlines() if l.strip()]
    report = []
    for tid in tasks:
        env = None
        try:
            task = suite.get_task(tid)
            env, _ = base.make_task_env(task, 7, False)
            inits = suite.get_task_init_states(tid)
            bddl = safe.SAFELIBERO_ROOT / "libero/libero/bddl_files" / task.problem_folder / task.bddl_file
            target, destination = tt.goal_roles(tt.parse_bddl(bddl))
            for row in sorted((r for r in rows if r["task_id"] == tid), key=lambda r: r["episode_idx"]):
                ei = int(row["episode_idx"])
                stem = "safelibero_spatial_t{}_e{}".format(tid, ei)
                t0 = time.time()
                base.start_episode(env, inits, ei)
                sim = env.sim
                model = sim.model
                m, d = model._model, sim.data._data
                bnames = [model.body_id2name(i) or "" for i in range(model.nbody)]
                with np.load(str(run / "episodes" / (stem + ".npz")), allow_pickle=False) as z:
                    raw = {c: z[k] for c, _, k in CAMS}
                    chunk_t = z["chunk_t"].astype(np.int64)
                    eef = z["obs_robot0_eef_pos"]
                with np.load(str(run / "episodes" / (stem + "_geom.npz")), allow_pickle=False) as g:
                    states = g["sim_state_image"]
                    if bnames != [str(n) for n in g["model_body_names"]] or not np.array_equal(
                            model.geom_bodyid, g["model_geom_bodyid"]):
                        raise ValueError("body / geom table mismatch")
                gb = np.asarray(model.geom_bodyid)
                target_geoms = set(int(x) for x in tt.instance_geom_ids(target, gb, bnames))
                finger_of_body = {}
                for side, names in FINGER_BODIES.items():
                    for name in names:
                        finger_of_body[bnames.index(name)] = side
                finger_geom = [finger_of_body.get(int(b), "") for b in gb]
                dest_all = [int(x) for x in tt.instance_geom_ids(destination, gb, bnames)]
                dest_coll = [x for x in dest_all if int(m.geom_contype[x]) or int(m.geom_conaffinity[x])]
                dest_geoms = dest_coll or dest_all
                tbody, dbody = m.body(target + "_main").id, m.body(destination + "_main").id
                n = len(states)
                if n != len(eef) or n != len(raw["head"]):
                    raise ValueError("frame count mismatch")
                res = {"target_xpos": np.zeros((n, 3)), "destination_xpos": np.zeros((n, 3)),
                       "contact_left_finger_target": np.zeros(n, bool), "contact_right_finger_target": np.zeros(n, bool),
                       "contact_robot_target": np.zeros(n, bool), "destination_xy_box": np.zeros((n, 4))}
                call_set = set(int(x) for x in chunk_t)
                rgb_ok = {c: 0 for c, _, _ in CAMS}
                for i in range(n):
                    sim.set_state_from_flattened(np.asarray(states[i]).copy())
                    sim.forward()
                    res["target_xpos"][i] = d.xpos[tbody]
                    res["destination_xpos"][i] = d.xpos[dbody]
                    res["destination_xy_box"][i] = geoms_xy_box(m, d, dest_geoms)
                    left = right = robot = False
                    for k in range(d.ncon):
                        g1, g2 = int(d.contact[k].geom1), int(d.contact[k].geom2)
                        for a, b in ((g1, g2), (g2, g1)):
                            if a in target_geoms and b not in target_geoms:
                                left = left or finger_geom[b] == "left"
                                right = right or finger_geom[b] == "right"
                                robot = robot or bnames[int(gb[b])].startswith(ROBOT_PREFIXES)
                    res["contact_left_finger_target"][i], res["contact_right_finger_target"][i] = left, right
                    res["contact_robot_target"][i] = robot
                    if i in call_set:
                        for c, cam, _ in CAMS:
                            rgb = sim.render(camera_name=cam, height=256, width=256)
                            rgb_ok[c] += int(np.array_equal(rgb, raw[c][i]))
                if any(v != len(call_set) for v in rgb_ok.values()):
                    raise ValueError("re-rendered RGB differs from the record at call frames: {}".format(rgb_ok))
                np.savez_compressed(str(out / ("phase_" + stem + ".npz")), eef_pos=eef, call_frames=chunk_t,
                                    target=target, destination=destination, **res)
                report.append({"stem": stem, "status": "ok", "frames": n, "calls": len(call_set), "target": target,
                               "destination": destination, "rgb_equal_call_frames": rgb_ok,
                               "frames_both_fingers_touch": int((res["contact_left_finger_target"]
                                                                 & res["contact_right_finger_target"]).sum()),
                               "target_z_rise_max_mm": float((res["target_xpos"][:, 2].max()
                                                              - res["target_xpos"][int(chunk_t[0]), 2]) * 1e3),
                               "seconds": time.time() - t0})
                print(json.dumps(report[-1]), flush=True)
        except Exception as exc:  # noqa: BLE001
            report.append({"task": tid, "status": "error", "error": repr(exc), "traceback": traceback.format_exc()})
            print("task", tid, "ERROR", repr(exc), flush=True)
        finally:
            if env is not None:
                env.close()
        (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    sys.stdout.flush()
    os._exit(1 if any(r["status"] != "ok" for r in report) else 0)


if __name__ == "__main__":
    main()
