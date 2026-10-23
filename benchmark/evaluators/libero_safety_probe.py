"""LIBERO-Safety CPU probe (BM-SETUP-B; ``.venv-libero-safety``, no policy, no GPU).

Per (suite, level, task): env as the client builds it (``libero_pi05_client.make_task_env`` with geometry capture)
-> reset + init state -> wait steps (dummy action) -> observation through the V11AB camera adapter -> a scripted
arm motion that drives the gripper into the first ``CheckRobotContact`` object of the BDDL constraints (or, with
none, the first constraint object) -> per step: official ``info['cost']``, per-constraint evaluation and geom-id
robot contact (``libero_safety_client.ConstraintMonitor``).

    source scripts/setup/libero_safety_env.sh
    .venv-libero-safety/bin/python -m benchmark.evaluators.libero_safety_probe --out outputs/verify/BM-SETUP-B/probe \
        --suites affordance,human_safety --levels 0,1,2 --task-ids 0

Writes ``<out>/<suite>_L<level>_t<task>.json`` and ``..._frames.npz`` (agentview/wrist raw RGB at init and at first
violation, depth, segmentation of the init frame).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/runners"))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import libero_safety_client as lsc  # noqa: E402

OSC_POS_PER_UNIT = lsc.OSC_OUTPUT_MAX["libero_safety_fork"]["pos"]


def _jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def segmentation_names(env, seg):
    """geom id image -> {geom id: (geom name, body name)} for the ids present."""
    m = lsc._inner(env).sim.model
    out = {}
    for gid in np.unique(seg):
        gid = int(gid)
        if gid < 0 or gid >= m.ngeom:
            continue
        out[gid] = (m.geom_id2name(gid), m.body_id2name(int(m.geom_bodyid[gid])))
    return out


def target_object(inner):
    for c in inner.parsed_problem.get("constraints", []):
        if c[0] == "checkrobotcontact":
            return c[1], "checkrobotcontact"
    for c in inner.parsed_problem.get("constraints", []):
        if len(c) >= 3:
            return c[2], c[0]
    return None, None


def object_position(inner, name):
    if name in getattr(inner, "obj_body_id", {}):
        return np.array(inner.sim.data.body_xpos[inner.obj_body_id[name]], np.float64)
    obj = inner.get_object(name)
    bid = inner.sim.model.body_name2id(obj.root_body)
    return np.array(inner.sim.data.body_xpos[bid], np.float64)


def probe_task(suite_obj, suite, level, task_id, out, args):
    from benchmark.adapters.libero_panda import camera

    task = suite_obj.get_task(task_id)
    init = suite_obj.get_task_init_states(task_id)
    res = {"suite": suite, "level": level, "task_id": task_id, "task": task.name, "language": task.language,
           "bddl": task.problem_folder + "/" + task.bddl_file, "n_init_states": int(len(init)),
           "init_state_dim": int(np.asarray(init).shape[1])}
    t0 = time.time()
    env, desc = lsc.base.make_task_env(task, args.seed, True)
    res["env_create_s"] = time.time() - t0
    inner = lsc._inner(env)
    res.update({"problem_class": type(inner).__name__, "robot": type(inner.robots[0].robot_model).__name__,
                "controller": dict(inner.robots[0].controller_config),
                "control_freq": inner.control_freq, "horizon": inner.horizon,
                "camera_names": list(inner.camera_names), "camera_hw": [inner.camera_heights, inner.camera_widths],
                "robot_init_id": getattr(inner, "robot_init_id", None), "noise": getattr(inner, "noise", None),
                "obj_of_interest": list(inner.obj_of_interest),
                "dynamic_objects": _jsonable(getattr(inner, "dynamic_objects", {})),
                "objects": sorted(inner.objects_dict), "fixtures": sorted(inner.fixtures_dict),
                "goal": _jsonable(inner.parsed_problem["goal_state"]),
                "constraints": _jsonable(inner.parsed_problem.get("constraints", []))})
    obs = lsc.base.start_episode(env, init, 0)
    mon = lsc.ConstraintMonitor(env, lsc.NUM_STEPS_WAIT)
    res["robot_geom_ids"] = len(mon.robot_ids)
    res["checkrobotcontact_object_geom_ids"] = {k: len(v) for k, v in mon.obj_ids.items()}
    frames0 = camera.capture(env, obs, segmentation=True)
    for _ in range(lsc.NUM_STEPS_WAIT):
        obs, r, done, info = env.step(lsc.base.LIBERO_DUMMY_ACTION)
        mon.step(info, True)
    frames = camera.capture(env, obs, segmentation=True)
    seg = frames["head"]["segmentation_geom"]
    names = segmentation_names(env, seg)
    seg_w = frames["left_wrist"]["segmentation_geom"]
    names.update(segmentation_names(env, seg_w))
    d = frames["head"]["depth_m"]
    res["adapter"] = {"K_head": frames["head"]["K"].tolist(), "T_base_cam_head": frames["head"]["T_base_cam"].tolist(),
                      "T_world_base": frames["head"]["T_world_base"].tolist(),
                      "depth_head_m": [float(np.min(d)), float(np.median(d)), float(np.max(d))],
                      "robot_state": frames["head"]["robot_state"].tolist(),
                      "fovy_deg": {k: v["fovy_deg"] for k, v in frames.items()},
                      "seg_geoms_visible": len(names)}
    # human / hand / constraint objects as the adapter sees them (segmentation geom -> body names)
    keys = set(n.lower() for n in res["objects"] + res["fixtures"])
    vis = {}
    for gid, (gname, bname) in names.items():
        owner = next((k for k in keys if bname and bname.lower().startswith(k)), None)
        vis.setdefault(owner or ("robot" if bname and bname.startswith(("robot0", "gripper0")) else str(bname)),
                       []).append([gid, gname, bname])
    res["segmentation_by_object"] = {k: {"n_geoms": len(v), "example": v[:3]} for k, v in sorted(vis.items())}
    res["wait_official_violations"] = mon.official_wait
    res["wait_full_eval"] = mon.evaluate_full()
    res["wait_geom_contact"] = mon.evaluate_geom()

    # controller step response (fork OSC): +x / +z constant action, then zero; realised EE displacement per step
    resp = []
    for k in range(30):
        a = np.zeros(7)
        a[6] = -1.0
        if k < 10:
            a[0] = args.response_action
        elif k < 20:
            a[2] = args.response_action
        before = np.array(obs["robot0_eef_pos"], np.float64)
        obs, r, done, info = env.step(a.tolist())
        mon.step(info, True)
        resp.append({"a": a[:3].tolist(), "dp": (np.array(obs["robot0_eef_pos"]) - before).round(6).tolist()})
    res["controller_step_response"] = resp

    # scripted violation: hover above the object, then descend into it
    tgt_name, tgt_pred = target_object(inner)
    res["drive_target"] = [tgt_name, tgt_pred]
    trace = []
    first = {"official": None, "full": None, "geom": None, "done": None}
    hold_until = None
    vframe = None
    if tgt_name is not None:
        for k in range(args.drive_steps):
            ee = np.array(obs["robot0_eef_pos"], np.float64)
            p = object_position(inner, tgt_name)
            goal = p + np.array([0.0, 0.0, 0.12 if k < args.drive_steps // 3 else -0.02])
            a = np.zeros(7)
            a[:3] = np.clip((goal - ee) / OSC_POS_PER_UNIT * 0.3, -0.006, 0.006)
            a[6] = -1.0
            obs, r, done, info = env.step(a.tolist())
            official = mon.step(info, False)
            full = mon.evaluate_full()
            geom = mon.evaluate_geom()
            row = {"k": k, "ee": ee.round(4).tolist(), "obj": p.round(4).tolist(),
                   "cost": {str(a_): int(b_) for a_, b_ in (info.get("cost") or {}).items()},
                   "full": full, "geom": geom, "done": bool(done)}
            trace.append(row)
            for key, val in (("official", official), ("full", any(v is True for v in full.values())),
                             ("geom", any(geom.values())), ("done", bool(done))):
                if val and first[key] is None:
                    first[key] = k
            if vframe is None and (first["full"] is not None or first["geom"] is not None or first["official"] is not None):
                vframe = camera.capture(env, obs, segmentation=False)
                hold_until = k + 5
            if hold_until is not None and k >= hold_until:
                break
    res["drive_first_detection_step"] = first
    res["drive_min_distance_m"] = (min(float(np.linalg.norm(np.array(r_["ee"]) - np.array(r_["obj"]))) for r_ in trace)
                                   if trace else None)
    res["drive_trace_tail"] = trace[-8:]
    res["monitor_record"] = mon.record(False, False)
    np.savez_compressed(str(out / "{}_L{}_t{}_frames.npz".format(suite, level, task_id)),
                        agentview_raw_init=obs_raw(frames0, "head"), wrist_raw_init=obs_raw(frames0, "left_wrist"),
                        agentview_rgb_cv=frames["head"]["rgb"], wrist_rgb_cv=frames["left_wrist"]["rgb"],
                        depth_head_m=frames["head"]["depth_m"].astype(np.float32),
                        seg_head=frames["head"]["segmentation_geom"].astype(np.int32),
                        agentview_rgb_cv_violation=(vframe["head"]["rgb"] if vframe else np.zeros((0,))),
                        wrist_rgb_cv_violation=(vframe["left_wrist"]["rgb"] if vframe else np.zeros((0,))))
    env.close()
    return res


def obs_raw(frames, cam):
    return frames[cam]["rgb"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default=",".join(lsc.SUITES))
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--task-ids", default="0")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--drive-steps", type=int, default=150)
    ap.add_argument("--response-action", type=float, default=0.005)
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env_info = lsc.activate_libero_safety()
    from libero.libero import benchmark

    (out / "env.json").write_text(json.dumps(env_info, indent=1))
    rc = 0
    for suite in args.suites.split(","):
        for level in [int(x) for x in args.levels.split(",")]:
            so = lsc.LevelSuite(benchmark.get_benchmark_dict()[suite](), level)
            for tid in [int(x) for x in args.task_ids.split(",")]:
                path = out / "{}_L{}_t{}.json".format(suite, level, tid)
                t0 = time.time()
                try:
                    np.random.seed(args.seed)
                    res = probe_task(so, suite, level, tid, out, args)
                    res["ok"] = True
                except Exception as exc:  # noqa: BLE001
                    res = {"suite": suite, "level": level, "task_id": tid, "ok": False,
                           "error": "{}: {}".format(type(exc).__name__, exc), "traceback": traceback.format_exc()}
                    rc = 1
                res["wall_s"] = time.time() - t0
                path.write_text(json.dumps(_jsonable(res), indent=1))
                print(suite, level, tid, "ok" if res["ok"] else res["error"], "{:.0f}s".format(res["wall_s"]),
                      res.get("drive_first_detection_step"), flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
