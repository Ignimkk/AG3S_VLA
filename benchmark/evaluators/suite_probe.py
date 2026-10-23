"""CPU probe of one LIBERO / SafeLIBERO suite (BM-SETUP-A). **Python 3.8**, ``.venv-libero``, no policy, no GPU.

For every task of the suite (and safety level) the probe builds the env **exactly as the paired client does**
(``libero_pi05_client.make_task_env`` with geometry capture: ``camera_depths=True`` + the SL1 capture observable),
starts init state 0 (``start_episode``), runs the suite's wait steps with the official dummy action, converts the
observation with the SL1/SL2 camera adapter and checks the conventions the online V11AB path relies on:

* image orientation: adapter RGB = ``raw[::-1]`` (OpenCV rows) and the policy image ``raw[::-1, ::-1]`` equals
  ``opencv[:, ::-1]`` (left-right mirror, SL1);
* depth finite / range, K pixel-centre (``cx = cy = 127.5``, ``fx = fy``), ``T_base_cam`` a rigid transform,
  ``robot_state`` 9-D = image-instant joint qpos (7) + finger qpos (2), policy state 8-D, gripper opening;
* the robot base pose in the world (``T_world_base``) and the support surface (table / floor) height in the base
  frame, from the head depth back-projected on the support's segmentation pixels -- the Panda ESDF box floor is
  base z ``PANDA_ESDF_BOUNDS_LOWER_Z`` (-0.10 m, ``AG3S/adapters/panda/profile.py``);
* every BDDL object / fixture resolves to MuJoCo bodies by the adapter's name rule (``task_target.instance_body_ids``)
  and is visible (segmentation pixel count per camera); the BDDL target rule (``task_target.goal_roles``) and
  whether its target is a body or only a region; object positions inside the ESDF box;
* fixture reset history (SL4b): ``model.body_pos`` of joint-less bodies after a second ``start_episode``;
* SafeLIBERO: obstacle candidates and the active obstacle of the official rule (``libero_safe_client.CollisionMonitor``
  fed with every observation as in the client), a hold of ``--hold-steps`` dummy steps (must not flag a
  collision) and a scripted OSC drive of the end effector into the active obstacle (must flag one).

Writes ``--out`` (JSON, one record per task) and, with ``--frames-dir``, ``<suite>[_L<level>]_t<task>.npz`` (head /
wrist OpenCV RGB, head depth and segmentation after the wait) for figures.

    .venv-libero/bin/python benchmark/evaluators/suite_probe.py --suite safelibero_goal --safety-level II \
        --out outputs/verify/BM-SETUP-A/probe/safelibero_goal_LII.json --frames-dir outputs/verify/BM-SETUP-A/frames
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
import libero_pi05_client as base  # noqa: E402
import libero_safe_client as safe  # noqa: E402

#: Panda ESDF workspace box (base frame, m): ``AG3S.ag3s.fields.esdf.default_bounds()`` with the Panda floor
#: (``profile.PANDA_ESDF_BOUNDS_LOWER_Z``). Restated here (py3.8 cannot import AG3S); the 3.11 test compares.
ESDF_BOX_LOWER = (-0.3, -0.9, -0.10)
ESDF_BOX_UPPER = (1.2, 0.9, 1.60)
SUPPORT_FIXTURES = ("main_table", "kitchen_table", "living_room_table", "study_table", "floor")
DRIVE_MAX_STEPS = 300
OSC_MAX_DELTA_M = 0.05  # robosuite OSC_POSE output_max (position) for |action| = 1


def _task_target_module():
    root = str(ROOT)
    if root not in sys.path:
        sys.path.append(root)
    from benchmark.adapters.libero_panda import task_target

    return task_target


def _resolve(tt, name, body_names):
    """Adapter name rule; also report the bodies a plain prefix match finds when the rule fails."""
    try:
        return tt.instance_body_ids(name, body_names), None
    except ValueError as exc:
        return [], str(exc)


def _support_height(frame, seg, geom_ids):
    """Median / 5th / 95th percentile base-frame z of the head depth back-projected on ``geom_ids`` pixels."""
    mask = np.isin(seg, np.asarray(geom_ids, np.int64))
    rows, cols = np.nonzero(mask)
    if rows.size == 0:
        return None
    z = frame["depth_m"][rows, cols].astype(np.float64)
    K = frame["K"]
    pts = np.stack([(cols - K[0, 2]) * z / K[0, 0], (rows - K[1, 2]) * z / K[1, 1], z, np.ones_like(z)], 1)
    pb = pts @ frame["T_base_cam"].T
    return {"pixels": int(rows.size), "z_base_median_m": float(np.median(pb[:, 2])),
            "z_base_p05_m": float(np.percentile(pb[:, 2], 5)), "z_base_p95_m": float(np.percentile(pb[:, 2], 95))}


def _rigid_ok(T):
    R = np.asarray(T)[:3, :3]
    return bool(np.allclose(R @ R.T, np.eye(3), atol=1e-6) and abs(np.linalg.det(R) - 1) < 1e-6
                and np.allclose(np.asarray(T)[3], [0, 0, 0, 1]))


def _frame_checks(obs, frames):
    out = {}
    for source, wire in safe.base._geometry_adapter()[0].CAMERA_TABLE:
        f = frames[wire]
        raw = np.asarray(obs[source + "_image"])
        opencv = f["rgb"]
        policy = np.ascontiguousarray(raw[::-1, ::-1])
        d = f["depth_m"]
        K = f["K"]
        out[wire] = {
            "rgb_shape": list(opencv.shape), "rgb_is_raw_row_flip": bool(np.array_equal(opencv, raw[::-1])),
            "policy_is_opencv_mirror": bool(np.array_equal(policy, opencv[:, ::-1])),
            "depth_shape": list(d.shape), "depth_finite_frac": float(np.isfinite(d).mean()),
            "depth_min_m": float(np.nanmin(d)), "depth_max_m": float(np.nanmax(d)),
            "K": np.asarray(K).tolist(), "K_pixel_centre": bool(abs(K[0, 2] - (d.shape[1] - 1) / 2) < 1e-9
                                                               and abs(K[1, 2] - (d.shape[0] - 1) / 2) < 1e-9
                                                               and abs(K[0, 0] - K[1, 1]) < 1e-9),
            "fovy_deg": f["fovy_deg"], "T_base_cam_rigid": _rigid_ok(f["T_base_cam"]),
            "T_base_cam_t_m": np.asarray(f["T_base_cam"])[:3, 3].tolist(),
        }
    return out


def _robot_checks(env, obs, frames):
    cam = safe.base._geometry_adapter()[0]
    rs = np.asarray(frames["left_wrist"]["robot_state"], np.float64)
    state = np.concatenate((obs["robot0_eef_pos"], base._quat2axisangle(np.array(obs["robot0_eef_quat"], copy=True)),
                            obs["robot0_gripper_qpos"]))
    T_wb = np.asarray(frames["head"]["T_world_base"])
    eef_base = np.linalg.inv(T_wb) @ np.r_[np.asarray(obs["robot0_eef_pos"], np.float64), 1.0]
    return {"robot_state_dim": int(rs.size),
            "robot_state_vs_obs_joint_max_abs": float(np.max(np.abs(rs[:7] - np.asarray(obs["robot0_joint_pos"])))),
            "robot_state_vs_obs_gripper_max_abs": float(np.max(np.abs(rs[7:] - np.asarray(obs["robot0_gripper_qpos"])))),
            "finger_opening_m": float(abs(rs[7]) + abs(rs[8])),
            "policy_state_dim": int(state.size), "eef_pos_base_m": eef_base[:3].tolist(),
            "T_world_base": T_wb.tolist(), "robot_qpos_indexes": cam.robot_qpos_indexes(env),
            "controller": type(cam._inner(env).robots[0].controller).__name__,
            "control_freq_hz": float(cam._inner(env).control_freq)}


def _fixture_body_pos(env):
    m = env.sim.model
    names = [m.body_id2name(i) or "" for i in range(m.nbody)]
    jointless = [i for i in range(1, m.nbody) if int(m.body_jntnum[i]) == 0 and int(m.body_parentid[i]) == 0]
    return {names[i]: np.array(m.body_pos[i], np.float64) for i in jointless}


def _obstacle_contacts(env, obstacle_body_ids):
    """MuJoCo contacts (dist <= 0) between the obstacle's bodies and any other body: ``[(other body, dist)]``."""
    m, d = env.sim.model, env.sim.data
    out = []
    for c in d.contact[:d.ncon]:
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if (b1 in obstacle_body_ids) == (b2 in obstacle_body_ids) or c.dist > 0:
            continue
        other = b2 if b1 in obstacle_body_ids else b1
        out.append((m.body_id2name(other) or "", float(c.dist)))
    return out


def _drive_into_obstacle(env, obs, monitor, observe, obstacle_body_ids):
    """OSC P-control of the EE onto the active obstacle (``monitor.initial``); stops at the first collision.

    Independent evidence: the MuJoCo contacts of the obstacle at the flagged step (robot bodies are ``robot0_*`` /
    ``gripper0_*``)."""
    goal = np.asarray(monitor.initial, np.float64) + np.array([0.0, 0.0, 0.03])
    start = float(np.linalg.norm(goal - np.asarray(obs["robot0_eef_pos"], np.float64)))
    robot_contact_first = None
    # robot-obstacle contacts can open and close inside one env.step (control 20 Hz, physics 500 Hz): record
    # them after every physics substep (robosuite base.step calls self.sim.step() per substep).
    sim = env.sim
    substep_robot = set()
    real_step = sim.step

    def step_and_record():
        real_step()
        for name, _ in _obstacle_contacts(env, obstacle_body_ids):
            if name.startswith(("robot0_", "gripper0_")):
                substep_robot.add(name)
    sim.step = step_and_record
    for k in range(DRIVE_MAX_STEPS):
        err = goal - np.asarray(obs["robot0_eef_pos"], np.float64)
        if np.linalg.norm(err) < 0.02:  # at the goal and nothing moved yet: push through it
            err = err + np.array([0.0, 0.0, -0.10])
        a = np.zeros(7)
        a[:3] = np.clip(err / OSC_MAX_DELTA_M, -1.0, 1.0)
        a[6] = -1.0
        obs, _, _, _ = env.step(a.tolist())
        observe(env, obs)
        contacts = _obstacle_contacts(env, obstacle_body_ids)
        if substep_robot and robot_contact_first is None:
            robot_contact_first = k + 1
        if monitor.collided:
            break
    del sim.step  # back to the class method
    return {"drive_steps": k + 1, "collided": bool(monitor.collided),
            "collide_t_obs_minus_1": monitor.collide_t, "eef_to_goal_start_m": start,
            "eef_to_goal_m": float(np.linalg.norm(goal - obs["robot0_eef_pos"])),
            "first_robot_contact_step": robot_contact_first, "robot_bodies_in_contact_substeps": sorted(substep_robot),
            "contacts_at_flag": contacts if monitor.collided else None}


def probe_task(args, suite_obj, task_id, tt, bddl_root, frames_dir):
    rec = {"suite": args.suite, "safety_level": args.safety_level, "task_id": task_id}
    task = suite_obj.get_task(task_id)
    init = suite_obj.get_task_init_states(task_id)
    rec.update({"task_name": task.name, "language": task.language, "n_init_states": int(len(init)),
                "bddl_file": str((bddl_root / task.problem_folder / task.bddl_file).relative_to(ROOT))})
    t0 = time.perf_counter()
    env, desc = base.make_task_env(task, args.seed, True)
    rec["env_create_s"] = time.perf_counter() - t0
    cam = base._geometry_adapter()[0]
    monitor = safe.CollisionMonitor(args.num_steps_wait) if args.safelibero else None
    observe = (lambda e, o: monitor.observe(e, o)) if monitor else (lambda e, o: None)
    try:
        t0 = time.perf_counter()
        obs = base.start_episode(env, init, 0)
        rec["reset_s"] = time.perf_counter() - t0
        observe(env, obs)
        fixtures_1 = _fixture_body_pos(env)
        t0 = time.perf_counter()
        for _ in range(args.num_steps_wait):
            obs, _, done, _ = env.step(base.LIBERO_DUMMY_ACTION)
            observe(env, obs)
        rec["wait_steps"] = args.num_steps_wait
        rec["step_s_mean"] = (time.perf_counter() - t0) / max(1, args.num_steps_wait)
        rec["done_after_wait"] = bool(done)
        frames = cam.capture(env, obs, segmentation=True)
        tables = cam.model_tables(env)
        body_names = [str(b) for b in tables["model_body_names"]]
        rec["frames"] = _frame_checks(obs, frames)
        rec["robot"] = _robot_checks(env, obs, frames)
        # --- BDDL objects / fixtures -> bodies -> segmentation -------------------------------------------------
        problem = tt.parse_bddl(bddl_root / task.problem_folder / task.bddl_file)
        try:
            target, destination = tt.goal_roles(problem)
            rec["target_rule"] = {"target": target, "destination": destination, "error": None}
        except ValueError as exc:
            target, destination = None, None
            rec["target_rule"] = {"target": None, "destination": None, "error": str(exc)}
        rec["goal"] = problem["goal"]
        rec["obj_of_interest"] = problem["obj_of_interest"]
        T_bw = np.linalg.inv(np.asarray(frames["head"]["T_world_base"]))
        xpos = np.asarray(env.sim.data.body_xpos, np.float64)
        inst = {}
        for name in problem["object_names"] + problem["fixture_names"]:
            ids, err = _resolve(tt, name, body_names)
            gids = [g for g, b in enumerate(tables["model_geom_bodyid"].tolist()) if int(b) in set(ids)]
            entry = {"kind": "fixture" if name in problem["fixture_names"] else "object", "n_bodies": len(ids),
                     "n_geoms": len(gids), "error": err}
            for wire in ("head", "left_wrist"):
                entry["pixels_" + wire] = int(np.isin(frames[wire]["segmentation_geom"], gids).sum()) if gids else 0
            if ids:
                main = [i for i in ids if body_names[i] == name + "_main"] or ids[:1]
                pb = T_bw @ np.r_[xpos[main[0]], 1.0]
                entry["pos_base_m"] = pb[:3].tolist()
                entry["in_esdf_box"] = bool(np.all(pb[:3] >= ESDF_BOX_LOWER) and np.all(pb[:3] <= ESDF_BOX_UPPER))
            inst[name] = entry
        rec["instances"] = inst
        rec["unresolved"] = sorted(n for n, e in inst.items() if e["n_bodies"] == 0)
        rec["objects_outside_esdf_box"] = sorted(n for n, e in inst.items()
                                                 if e["kind"] == "object" and e.get("in_esdf_box") is False)
        rec["target_is_body"] = bool(target is not None and inst.get(target, {}).get("n_bodies"))
        rec["target_pixels"] = {w: inst.get(target, {}).get("pixels_" + w) for w in ("head", "left_wrist")}
        sup = [n for n in problem["fixture_names"] if n in SUPPORT_FIXTURES]
        rec["support"] = None
        if sup:
            ids, _ = _resolve(tt, sup[0], body_names)
            gids = [g for g, b in enumerate(tables["model_geom_bodyid"].tolist()) if int(b) in set(ids)]
            rec["support"] = {"fixture": sup[0], "bodies": [body_names[i] for i in ids],
                              **(_support_height(frames["head"], frames["head"]["segmentation_geom"], gids) or {})}
            z = rec["support"].get("z_base_median_m")
            rec["support"]["above_esdf_floor"] = None if z is None else bool(z > ESDF_BOX_LOWER[2])
        # --- SafeLIBERO: active obstacle, hold, drive ---------------------------------------------------------
        if monitor is not None:
            rec["obstacle"] = {"candidates": monitor.candidates, "active": monitor.obstacle,
                               "initial_pos_world": None if monitor.initial is None else monitor.initial.tolist()}
            if monitor.obstacle is not None:
                ids, _ = _resolve(tt, monitor.obstacle, body_names)
                gids = [g for g, b in enumerate(tables["model_geom_bodyid"].tolist()) if int(b) in set(ids)]
                rec["obstacle"].update({
                    "n_bodies": len(ids), "pixels_head": int(np.isin(frames["head"]["segmentation_geom"], gids).sum()),
                    "pixels_left_wrist": int(np.isin(frames["left_wrist"]["segmentation_geom"], gids).sum()),
                    "pos_base_m": (T_bw @ np.r_[monitor.initial, 1.0])[:3].tolist()})
                active_in_bddl = monitor.obstacle in problem["object_names"]
                rec["obstacle"]["active_in_bddl_objects"] = bool(active_in_bddl)
                for _ in range(args.hold_steps):
                    obs, _, _, _ = env.step(base.LIBERO_DUMMY_ACTION)
                    observe(env, obs)
                rec["hold"] = {"steps": args.hold_steps, "collided": bool(monitor.collided),
                               "max_displacement_m": monitor.max_displacement}
                if not monitor.collided:
                    rec["drive"] = _drive_into_obstacle(env, obs, monitor, observe, set(ids))
                rec["collision_record"] = monitor.record(False)
        if frames_dir is not None:
            stem = "{}{}_t{}".format(args.suite, "_L" + args.safety_level if args.safety_level else "", task_id)
            np.savez_compressed(str(frames_dir / (stem + ".npz")),
                                head_rgb=frames["head"]["rgb"], wrist_rgb=frames["left_wrist"]["rgb"],
                                head_depth_m=frames["head"]["depth_m"],
                                head_seg=frames["head"]["segmentation_geom"],
                                wrist_seg=frames["left_wrist"]["segmentation_geom"], **tables)
        # --- fixture reset history (SL4b) --------------------------------------------------------------------
        base.start_episode(env, init, 0)
        fixtures_2 = _fixture_body_pos(env)
        moved = {n: float(np.max(np.abs(fixtures_2[n] - p))) for n, p in fixtures_1.items()
                 if n in fixtures_2 and np.max(np.abs(fixtures_2[n] - p)) > 1e-12}
        rec["fixture_reset"] = {"jointless_world_bodies": sorted(fixtures_1), "moved_on_second_reset": moved}
        rec["ok"] = True
    except Exception as exc:  # recorded, the next task still runs
        rec.update({"ok": False, "error": "{}: {}".format(type(exc).__name__, exc), "traceback": traceback.format_exc()})
    finally:
        env.close()
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--suite", required=True)
    ap.add_argument("--safety-level", choices=("I", "II"), default=None)
    ap.add_argument("--task-ids", type=base._int_list, default=None, help="default: every task of the suite")
    ap.add_argument("--seed", type=int, default=base.DEFAULT_SEED)
    ap.add_argument("--hold-steps", type=int, default=20)
    ap.add_argument("--count-only", action="store_true", help="suite / task / init-state counts only (libero_90)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-dir", default=None)
    args = ap.parse_args(argv)
    args.safelibero = args.suite in safe.SAFELIBERO_TOTAL_STEPS
    if args.safelibero != (args.safety_level is not None):
        raise ValueError("--safety-level is required for SafeLIBERO suites and only for them")
    if args.safelibero:
        env_info = safe.activate_safelibero()
        args.num_steps_wait = safe.SAFELIBERO_NUM_STEPS_WAIT
        bddl_root = safe.SAFELIBERO_ROOT / "libero/libero/bddl_files"
    else:
        env_info = {"config": str(base.require_libero_config())}
        args.num_steps_wait = base.DEFAULT_NUM_STEPS_WAIT
        bddl_root = ROOT / "benchmark/LIBERO/libero/libero/bddl_files"
    from libero.libero import benchmark

    tt = _task_target_module()
    np.random.seed(args.seed)
    cls = benchmark.get_benchmark_dict()[args.suite]
    suite_obj = cls(safety_level=args.safety_level) if args.safelibero else cls()
    out = {"suite": args.suite, "safety_level": args.safety_level, "seed": args.seed, "env": env_info,
           "n_tasks": int(suite_obj.n_tasks), "num_steps_wait": args.num_steps_wait,
           "libero_file": sys.modules["libero.libero"].__file__, "esdf_box": [ESDF_BOX_LOWER, ESDF_BOX_UPPER],
           "tasks": []}
    if args.count_only:
        out["n_init_states"] = [int(len(suite_obj.get_task_init_states(t))) for t in range(suite_obj.n_tasks)]
        out["task_names"] = suite_obj.get_task_names()
    else:
        frames_dir = pathlib.Path(args.frames_dir) if args.frames_dir else None
        if frames_dir is not None:
            frames_dir.mkdir(parents=True, exist_ok=True)
        for task_id in (args.task_ids if args.task_ids is not None else range(suite_obj.n_tasks)):
            np.random.seed(args.seed)  # the client seeds once per run; make_task_env re-seeds per task (env.seed)
            rec = probe_task(args, suite_obj, int(task_id), tt, bddl_root, frames_dir)
            out["tasks"].append(rec)
            print(json.dumps({"task_id": task_id, "ok": rec.get("ok"), "error": rec.get("error")}), flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, default=lambda x: np.asarray(x).tolist()))
    return 0 if all(r.get("ok") for r in out["tasks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
