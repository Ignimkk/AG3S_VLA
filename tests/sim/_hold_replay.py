"""T41 c helper — replay a recorded closed-loop run bit for bit, then HOLD from a chosen row.

The replay is the T40 H one (``tests/sim/test_t40h_false_zero.py``): ``pi05_infer.main()`` up to
``load_remote_policy`` (episode reset, obstacle manager), ``--warmup-steps 1000``, then every recorded
control row's ``applied_ctrl`` for ``steps_per_action`` sub-steps. Every row's recorded ``qpos`` is
checked against ``d.qpos`` before it is applied, so the state the HOLD starts from is the run's own
state (velocities, contacts and solver warm start included), not a pose rebuilt from ``qpos``.

From row ``t0`` the loop is the client's HOLD branch: ``HoldController.hold_action(last_command =
d.ctrl as applied_ctrl(), measured = rby1_state())`` -> ``apply_action`` -> ``steps_per_action`` x
``mj_step`` (+ the obstacle manager's contact / clearance check every 5th sub-step, as the client).

Used by ``tests/sim/test_t41c_hold_drift.py`` and ``outputs/impl/T41c/p/measure_drift.py``.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import inspect
import io
import json
import pathlib
import sys

import mujoco
import numpy as np

HYBRID = pathlib.Path("/mnt/dev/work/pi05_TO_hybrid")
PI05_INFER = HYBRID / "rby1_bringup" / "pi05_infer.py"
RUNS = pathlib.Path("/mnt/dev/work/outputs/verify/T40/G/runs/E3b")
N = 7  # rby1_16d arm joints per side


class _Reached(Exception):
    pass


def load_client():
    if str(HYBRID / "rby1_manipulation" / "src") not in sys.path:
        sys.path.insert(0, str(HYBRID / "rby1_manipulation" / "src"))
    spec = importlib.util.spec_from_file_location("_t41c_pi05_infer", PI05_INFER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _argv_value(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def load_run(run_dir: pathlib.Path) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "frames.jsonl").read_text().splitlines() if line.strip()]
    control = sorted((r for r in rows if r["kind"] == "control"), key=lambda r: int(r["t_step"]))
    return {"manifest": manifest, "control": control, "argv": manifest["extra"]["argv"]}


def reset(client, run: dict) -> dict:
    """``main()`` up to ``load_remote_policy`` with the run's episode / seed / obstacle flags."""
    argv = run["argv"]
    captured: dict = {}

    def stub(remote):
        captured.update(inspect.currentframe().f_back.f_locals)
        raise _Reached

    new_argv = ["pi05_infer.py", "--model", _argv_value(argv, "--model"), "--remote", "localhost:1",
                "--episode-index", _argv_value(argv, "--episode-index"), "--headless",
                "--warmup-steps", _argv_value(argv, "--warmup-steps", "1000"),
                "--policy-seed", _argv_value(argv, "--policy-seed"),
                "--obstacle-stop-distance", _argv_value(argv, "--obstacle-stop-distance", "0")]
    if "--obstacle-profile" in argv:
        new_argv += ["--obstacle-config", _argv_value(argv, "--obstacle-config"),
                     "--obstacle-profile", _argv_value(argv, "--obstacle-profile")]
    original, old_argv = client.load_remote_policy, sys.argv
    client.load_remote_policy = stub
    sys.argv = new_argv
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            client.main()
    except _Reached:
        pass
    except SystemExit as exc:
        raise AssertionError(f"main() exited ({exc.code}):\n{out.getvalue()[-2000:]}") from exc
    finally:
        client.load_remote_policy, sys.argv = original, old_argv
    assert captured, "main() never reached load_remote_policy"
    m, d = captured["m"], captured["d"]
    for _ in range(int(_argv_value(argv, "--warmup-steps", "1000"))):
        mujoco.mj_step(m, d)
    captured["steps_per_action"] = max(1, int(round(1.0 / (client.CTRL_HZ * m.opt.timestep))))
    return captured


def apply_row(client, local, row) -> None:
    d, act = local["d"], local["act"]
    applied = row["applied_ctrl"]
    for i in range(N):
        d.ctrl[act["left_a"][i]] = applied["arm"][i]
        d.ctrl[act["right_a"][i]] = applied["arm"][N + i]
    d.ctrl[act["left_grip_a"]] = applied["gripper"][0] * client.RBY1_GRIPPER_OPEN
    d.ctrl[act["right_grip_a"]] = applied["gripper"][1] * client.RBY1_GRIPPER_OPEN


def replay_to(client, local, control, t0) -> None:
    """Replay rows ``0 .. t0-1``; assert every recorded qpos (rows ``0 .. t0``) bit for bit."""
    m, d = local["m"], local["d"]
    for row in control[:t0]:
        assert np.array_equal(np.asarray(row["qpos"]), d.qpos), f"t={row['t_step']}: replay diverged"
        apply_row(client, local, row)
        for _ in range(local["steps_per_action"]):
            mujoco.mj_step(m, d)
    assert np.array_equal(np.asarray(control[t0]["qpos"]), d.qpos), f"t={t0}: replay diverged"


def rby1_state(client, local) -> np.ndarray:
    """``pi05_infer.main.rby1_state`` (16-D action layout)."""
    d, idx = local["d"], local["idx"]
    left = [d.qpos[i] for i in idx["left_q"][:N]]
    right = [d.qpos[i] for i in idx["right_q"][:N]]
    lg = abs(d.qpos[idx["left_grip_q"]]) / abs(client.RBY1_GRIPPER_OPEN)
    rg = abs(d.qpos[idx["right_grip_q"]]) / abs(client.RBY1_GRIPPER_OPEN)
    return np.asarray(left + [lg] + right + [rg], dtype=np.float64)


def applied_ctrl(client, local) -> dict:
    """``pi05_infer.main.applied_ctrl``."""
    d, act = local["d"], local["act"]
    arm = np.asarray([d.ctrl[a] for a in act["left_a"][:N]] + [d.ctrl[a] for a in act["right_a"][:N]])
    grip = np.asarray([d.ctrl[act["left_grip_a"]], d.ctrl[act["right_grip_a"]]]) / client.RBY1_GRIPPER_OPEN
    return {"arm": arm, "gripper": grip}


def _body(m, name):
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    assert i >= 0, name
    return i


def hold(client, local, mode: str, n_steps: int, *, fruit_body: str = "apple") -> dict:
    """HOLD for ``n_steps`` control steps from the current state; per-step metrics (copies ``d``)."""
    from benchmark.trajopt.client import HoldController

    m, d0, act = local["m"], local["d"], local["act"]
    d = mujoco.MjData(m)
    mujoco.mj_copyData(d, m, d0)
    sub = dict(local, d=d)
    mgr = local.get("obstacle_manager")
    if mgr is not None:
        # a per-branch copy: the manager reads `data` and its collision flags are sticky
        mgr = copy.copy(mgr)
        mgr.data = d
        mgr.robot_collision = mgr.payload_collision = False
        mgr.contact_pairs, mgr.contact_steps, mgr.min_robot_clearance = set(), 0, float("inf")
    ctl = HoldController(mode=mode)
    ctl.begin_chunk(executed=False, reason="T41c sim", kinds=["collision"])
    ee, fruit = _body(m, "EE_BODY_L"), _body(m, fruit_body)
    mujoco.mj_kinematics(m, d)
    ee0 = d.xpos[ee].copy()
    rel0 = d.xmat[ee].reshape(3, 3).T @ (d.xpos[fruit] - d.xpos[ee])
    q_meas0 = rby1_state(client, sub)
    last0 = applied_ctrl(client, sub)
    arm0 = np.concatenate([q_meas0[:N], q_meas0[N + 1:2 * N + 1]])
    rows, ctrl_rows, recs = [], [], []
    collided, min_clear, first_contact = False, np.inf, None
    for k in range(n_steps):
        if k % 8 == 0 and k:
            ctl.begin_chunk(executed=False, reason="T41c sim", kinds=["collision"])  # consecutive HOLD chunk
        action = ctl.hold_action(last_command=applied_ctrl(client, sub), measured=rby1_state(client, sub))
        client.apply_action("rby1_16d", action, d, local["idx"], act)
        ctrl_rows.append(action.copy())
        recs.append(ctl.record(measured=rby1_state(client, sub)))
        row_clear = np.inf
        for s in range(local["steps_per_action"]):
            mujoco.mj_step(m, d)
            if mgr is not None:
                mgr.observe_contacts()
                if mgr.robot_collision and first_contact is None:
                    first_contact = (k, s)
                collided = collided or bool(mgr.robot_collision)
                if s % 5 == 0:
                    row_clear = min(row_clear, float(mgr.robot_clearance()))
        min_clear = min(min_clear, row_clear)
        q = rby1_state(client, sub)
        arm = np.concatenate([q[:N], q[N + 1:2 * N + 1]])
        rel = d.xmat[ee].reshape(3, 3).T @ (d.xpos[fruit] - d.xpos[ee])
        rows.append({"arm_dev_from_entry_rad": float(np.max(np.abs(arm - arm0))),
                     "arm_dev_from_q_hold_rad": float(np.max(np.abs(arm - ctl.q_hold))),
                     "ee_disp_mm": float(np.linalg.norm(d.xpos[ee] - ee0) * 1e3),
                     "fruit_in_hand_shift_mm": float(np.linalg.norm(rel - rel0) * 1e3),
                     "fruit_z_m": float(d.xpos[fruit][2]),
                     "grip_meas_norm": float(q[N]), "grip_ctrl_norm": float(action[N]),
                     "robot_clearance_min_m": (None if mgr is None else float(row_clear))})
    return {"mode": mode, "rows": rows, "ctrl_rows": np.asarray(ctrl_rows), "records": recs,
            "q_meas_entry": q_meas0, "last_command_entry": last0, "arm_entry": arm0,
            "collided": collided, "first_contact_step_substep": first_contact,
            "contact_pairs": (None if mgr is None else sorted(mgr.contact_pairs)),
            "min_robot_clearance_m": (None if mgr is None else float(min_clear))}
