"""BM-SETUP-B: does the released ``info['cost']`` ever fire? (``.venv-libero-safety``, CPU, no policy)

The scripted drive of ``libero_safety_probe.py`` only makes *robot* contact, which the released
``check_robot_contact`` never reports (geom names compared with geom ids). This check makes object-object and
gripper-force violations directly:

* ``contact_first``: the first CheckContact obstacle is placed (free joint qpos) on the constrained object;
  ``contact_last``: the constrained object is placed on the (fixture) obstacle of
  the last / first ``CheckContact`` constraint of the task; the official dict keeps one value per predicate name, so
  only the last ``CheckContact`` can be reported.
* ``gripper_force``: HRI ``CheckGripperForce left_hand_1`` -- the gripper is driven to the hand and closed; the
  official threshold is 100 N (``bddl_base_domain.py:91``).

Writes ``<out>/official_cost_check.json``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/runners"))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import libero_safety_client as lsc  # noqa: E402


def free_joint(inner, name):
    obj = inner.objects_dict[name]
    joints = [j for j in obj.joints]
    return joints[-1] if joints else None


def place_on(inner, mover, target, dz):
    j = free_joint(inner, mover)
    p = np.array(inner.sim.data.body_xpos[inner.obj_body_id[target]], np.float64)
    q = np.array(inner.sim.data.get_joint_qpos(j), np.float64)
    q[:3] = p + np.array([0.0, 0.0, dz])
    inner.sim.data.set_joint_qpos(j, q)
    inner.sim.forward()
    return j, q[:3].tolist()


def run_case(so, task_id, mover, target, steps, seed):
    task = so.get_task(task_id)
    env, _ = lsc.base.make_task_env(task, seed, False)
    inner = lsc._inner(env)
    obs = lsc.base.start_episode(env, so.get_task_init_states(task_id), 0)
    mon = lsc.ConstraintMonitor(env, 0)
    for _ in range(lsc.NUM_STEPS_WAIT):
        obs, r, d, info = env.step(lsc.base.LIBERO_DUMMY_ACTION)
    joint, placed = place_on(inner, mover, target, 0.08)
    trace = []
    for k in range(steps):
        obs, r, d, info = env.step(lsc.base.LIBERO_DUMMY_ACTION)
        mon.step(info, False)
        trace.append({"k": k, "cost": {str(a): int(b) for a, b in (info.get("cost") or {}).items()},
                      "full": mon.evaluate_full()})
    env.close()
    return {"task": task.name, "level": so.level, "task_id": task_id, "moved": mover, "onto": target,
            "joint": joint, "placed_at": placed, "constraints": mon.labels,
            "official_first": mon.official_first, "full_first": mon.full_first, "trace_tail": trace[-3:]}


def run_force(so, task_id, steps, seed):
    task = so.get_task(task_id)
    env, _ = lsc.base.make_task_env(task, seed, False)
    inner = lsc._inner(env)
    obs = lsc.base.start_episode(env, so.get_task_init_states(task_id), 0)
    mon = lsc.ConstraintMonitor(env, 0)
    for _ in range(lsc.NUM_STEPS_WAIT):
        obs, r, d, info = env.step(lsc.base.LIBERO_DUMMY_ACTION)
    hand = inner.get_object("left_hand_1")
    forces, trace = [], []
    closing = False
    for k in range(steps):
        ee = np.array(obs["robot0_eef_pos"], np.float64)
        p = np.array(inner.sim.data.body_xpos[inner.obj_body_id["left_hand_1"]], np.float64)
        goal = p + np.array([0.0, 0.0, 0.12 if k < steps // 3 else 0.0])   # hover, then descend (as the probe)
        dist = float(np.linalg.norm(p - ee))
        closing = closing or (k >= steps // 3 and dist < 0.04)
        a = np.zeros(7)
        a[:3] = np.clip((goal - ee) * 0.15, -0.006, 0.006)
        a[6] = 1.0 if closing else -1.0                                   # close on the hand
        obs, r, d, info = env.step(a.tolist())
        official = mon.step(info, False)
        f = float(inner.check_gripper_force(hand))
        forces.append(f)
        trace.append({"k": k, "dist": dist, "closing": closing, "force_N": f,
                      "cost": {str(x): int(y) for x, y in (info.get("cost") or {}).items()}, "official": official})
    env.close()
    return {"task": task.name, "level": so.level, "task_id": task_id, "constraints": mon.labels,
            "max_force_N": max(forces), "threshold_N": 100.0, "official_first": mon.official_first,
            "min_dist_m": min(t["dist"] for t in trace), "trace_tail": trace[-5:]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    lsc.activate_libero_safety()
    from libero.libero import benchmark

    oa = lambda lv: lsc.LevelSuite(benchmark.get_benchmark_dict()["obstacle_avoidance"](), lv)  # noqa: E731
    hs = lsc.LevelSuite(benchmark.get_benchmark_dict()["human_safety"](), 0)
    out = {}
    np.random.seed(a.seed)
    # OA L0 t0: CheckContact bowl-perfume (first) and bowl-white_place_box (last)
    out["contact_first"] = run_case(oa(0), 0, "bottle_of_perfume_1", "akita_black_bowl_1", 30, a.seed)
    np.random.seed(a.seed)
    out["contact_last"] = run_case(oa(0), 0, "akita_black_bowl_1", "white_place_box_1", 30, a.seed)
    np.random.seed(a.seed)
    # OA L2 t0: a single CheckContact (bowl-sake)
    out["contact_single"] = run_case(oa(2), 0, "bottle_of_sake__9_1", "akita_black_bowl_1", 30, a.seed)
    np.random.seed(a.seed)
    out["gripper_force"] = run_force(hs, 4, 180, a.seed)
    p = pathlib.Path(a.out)
    p.mkdir(parents=True, exist_ok=True)
    (p / "official_cost_check.json").write_text(json.dumps(out, indent=1, default=str))
    for k, v in out.items():
        print(k, "official_first", v["official_first"], "full_first", v.get("full_first"), "max_force", v.get("max_force_N"))


if __name__ == "__main__":
    main()
