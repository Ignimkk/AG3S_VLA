"""T43 Q helper — execute a plan in MuJoCo from a bit-exact replayed state and measure the true mesh.

`_hold_replay` replays a recorded run bit for bit up to row `t0` (every recorded `qpos` asserted). From
there `execute` runs a chunk's first `n` rows through the client's own `apply_action` (`steps_per_action`
sim steps per row), on a copy of the state, and reports

* `contact` — the obstacle manager's robot–obstacle contact flag (the client's own detector, every sim
  step), with the first (row, sub-step);
* `min_true_mm` — the smallest GJK distance (`_gjk`, independent of `mj_geomDistance`) between the hand's
  collision geoms (`EE_BODY_L` + both fingers) and the run's obstacle collision geoms, sampled at every
  sub-step (`every` = 1) or every few;
* `joint_trace` — the measured arm joints (14, TO order) at every sub-step, for the servo-model test.

Used by `tests/sim/test_t43q_servo_sim.py` and `outputs/impl/T43Q/p/*`.
"""

from __future__ import annotations

import copy
import pathlib
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _hold_replay as H  # noqa: E402
from _gjk import gjk_dist, make_support  # noqa: E402

HAND = ("EE_BODY_L", "ee_finger_l1", "ee_finger_l2")


def load(run_dir: pathlib.Path, t0: int):
    client = H.load_client()
    run = H.load_run(pathlib.Path(run_dir))
    local = H.reset(client, run)
    H.replay_to(client, local, run["control"], t0)
    return client, local, run


def obstacle_geoms(m) -> list[int]:
    out = []
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith(("pick_obstacle", "place_obstacle")) and "collision" in name:
            out.append(g)
    return out


def hand_geoms(m) -> list[int]:
    ids = [H._body(m, b) for b in HAND]
    return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in ids
            and (m.geom_contype[g] or m.geom_conaffinity[g])]


def true_distance_mm(m, d, hand, obstacles) -> float:
    best = np.inf
    for o in obstacles:
        so = make_support(m, d, o)
        for g in hand:
            best = min(best, gjk_dist(make_support(m, d, g), so, d.geom_xpos[g] - d.geom_xpos[o]))
    return float(best * 1e3)


def arm_joints(local) -> np.ndarray:
    d, idx = local["d"], local["idx"]
    return np.asarray([d.qpos[i] for i in idx["left_q"][:H.N]] + [d.qpos[i] for i in idx["right_q"][:H.N]])


def execute(client, local, chunk, n: int = 8, *, every: int = 3) -> dict:
    """Run `chunk[:n]` from a copy of the current state. `chunk` rows are 16-D client actions."""
    m, d0, act = local["m"], local["d"], local["act"]
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, d0)
    sub = dict(local, d=d)
    mgr = local.get("obstacle_manager")
    if mgr is not None:
        mgr = copy.copy(mgr)
        mgr.data = d
        mgr.robot_collision = mgr.payload_collision = False
        mgr.contact_pairs, mgr.contact_steps, mgr.min_robot_clearance = set(), 0, float("inf")
    hand, obstacles = hand_geoms(m), obstacle_geoms(m)
    mujoco.mj_kinematics(m, d)
    true0 = true_distance_mm(m, d, hand, obstacles)
    trace, true_rows, first = [arm_joints(sub)], [], None
    minimum = true0
    for k in range(n):
        client.apply_action("rby1_16d", np.asarray(chunk[k], np.float64), d, local["idx"], act)
        row_min = np.inf
        for s in range(local["steps_per_action"]):
            mujoco.mj_step(m, d)
            trace.append(arm_joints(sub))
            if mgr is not None:
                mgr.observe_contacts()
                if mgr.robot_collision and first is None:
                    first = (k, s)
            if (s + 1) % every == 0 or s + 1 == local["steps_per_action"]:
                row_min = min(row_min, true_distance_mm(m, d, hand, obstacles))
        true_rows.append(row_min)
        minimum = min(minimum, row_min)
    return {"contact": None if mgr is None else bool(mgr.robot_collision),
            "first_contact_row_substep": first,
            "contact_pairs": None if mgr is None else sorted(mgr.contact_pairs),
            "true_mm_q_now": round(true0, 3),
            "true_mm_min_per_row": [round(v, 3) for v in true_rows],
            "true_mm_min": round(minimum, 3),
            "joint_trace": np.asarray(trace)}
