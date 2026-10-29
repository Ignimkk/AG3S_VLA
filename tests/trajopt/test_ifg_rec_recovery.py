"""IFG-REC — retreat-and-retry recovery (`safety.recovery_retreat` / `serve_safe --recovery-retreat`), default off.

Failures it answers (IFG 2026-10-21): ep211 held for ~50 chunks (HOLD -> the same policy chunk -> HOLD); ep954 closed on
nothing and the policy carried the empty hand to the crate. Training-free retreat-and-retry (Sheng et al. 2610.06921 style).

What this file pins:
1. Off: no history, no trigger, the chunk untouched.
2. Deadlock trigger: `recovery_hold_streak` consecutive whole-chunk collision HOLDs pre-grasp arm a retreat to the latest
   pose whose palm was >= `recovery_clear_m` from the target.
3. Empty-grasp trigger: `recovery_empty_streak` closing requests with settled & not-blocked fingers.
4. The retreat chunk: arms move toward the target at <= `recovery_step_rad` per row, grippers open, padding kept; ends when
   reached; at most `recovery_max` per episode; no trigger after the grasp (held / placed).
5. Config / CLI.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.grasp_latch import GraspPhase

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _policy(on=True, **safety):
    from AG3S.trajopt import wire
    from AG3S.trajopt.safe_policy import SafePolicy
    from tests.trajopt.test_safe_policy import _robot_model

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"safety": {"recovery_retreat": on, **safety}})
    p._latch = types.SimpleNamespace(phase=GraspPhase.LATCHED)
    model = _robot_model()
    p.constraint_robot_model = model
    p.grasp_links = {"left": ("ee_finger_l1", ()), "right": ("ee_finger_r1", ())}
    p.hold_links = {"left": "ee_left", "right": "ee_right"}
    p.gripper_columns = wire.gripper_columns(7)
    p.ag3s = types.SimpleNamespace(manipulated=None, frame_index=0)
    p._recovery = None
    p._recovery_count = 0
    p._recovery_hist = []
    p._rec_streaks = {"hold": 0, "empty": 0}
    p._last_recovery = {}
    p._last_grasp_record = {}
    return p, model


def _palm(model, q):
    from AG3S.ag3s.constraints.attached import _link_pose_numeric

    return np.asarray(_link_pose_numeric(model, q, "ee_left")[:3, 3], np.float64)


HELD = {"available": True, "n_hold": 8, "n_steps": 8, "hold_reason": "[collision] TO says violated, 21.0 mm"}
FREE = {"available": True, "n_hold": 0, "n_steps": 8, "hold_reason": None}


def _step(p, q, state16, fb=FREE, checks=None, frame=0):
    p.ag3s.frame_index = frame
    p._pending = {"state": state16, "exec_feedback": fb}
    p._last_grasp_record = {"evidence": {"checks": checks or {}}}
    p._last_recovery = {}
    p._recovery_observe(q, ["left"])


def test_off_does_nothing():
    p, model = _policy(on=False)
    q = np.zeros(len(model.joint_names))
    for k in range(5):
        _step(p, q, np.zeros(16), fb=HELD, frame=k)
    assert p._recovery is None and p._recovery_hist == [] and p._last_recovery == {}


def test_deadlock_arms_a_retreat_to_the_latest_clear_pose():
    p, model = _policy()
    q = np.zeros(len(model.joint_names))
    target = _palm(model, q)
    far = target + np.array([0.0, 0.0, 0.20])
    near = target + np.array([0.0, 0.0, 0.05])
    # frames 0-1: palm 20 cm away (clear), frames 2-5: 5 cm away and held
    for k in range(6):
        p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=(far if k < 2 else near) - 0.0 * target)
        p.ag3s.manipulated.centroid = (target - (far - target)) if k < 2 else (target - (near - target))
        _step(p, q, np.full(16, 0.1 * k), fb=HELD if k >= 3 else FREE, frame=k)
    assert p._recovery is not None and p._recovery["reason"] == "deadlock"
    assert p._recovery["target_frame"] == 1                                   # latest pose >= 12 cm from the target
    np.testing.assert_allclose(p._recovery["target"], np.full(16, 0.1))
    assert p._recovery_count == 1 and p._last_recovery["armed"] == "deadlock"


def test_empty_grasp_arms_on_settled_unblocked_closing():
    p, model = _policy()
    q = np.zeros(len(model.joint_names))
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=_palm(model, q) + np.array([0.0, 0.0, -0.3]))
    _step(p, q, np.zeros(16), frame=0)
    p._latch.phase = GraspPhase.CLOSING
    _step(p, q, np.ones(16), checks={"settled": True, "blocked": True}, frame=1)    # blocked: something between
    assert p._recovery is None
    _step(p, q, np.ones(16), checks={"settled": True, "blocked": False}, frame=2)
    assert p._recovery is None and p._rec_streaks["empty"] == 1
    _step(p, q, np.ones(16), checks={"settled": True, "blocked": False}, frame=3)
    assert p._recovery is not None and p._recovery["reason"] == "empty_grasp"


def test_retreat_chunk_steps_opens_and_ends():
    p, model = _policy(recovery_step_rad=0.05)
    p._recovery = {"reason": "deadlock", "target": np.r_[np.full(7, 0.3), 0.0, np.full(7, -0.2), 0.0],
                   "target_frame": 1, "requests": 0}
    chunk = np.zeros((10, 32)); chunk[:, 16:] = 7.0
    out = p._recovery_chunk(chunk, np.zeros(16))
    grip = list(p.gripper_columns)
    arm = [i for i in range(16) if i not in grip]
    steps = np.abs(np.diff(np.vstack([np.zeros(16), out[:, :16]])[:, arm], axis=0))
    assert steps.max() <= 0.05 + 1e-12 and np.allclose(out[:, grip], 1.0) and np.allclose(out[:, 16:], 7.0)
    np.testing.assert_allclose(out[-1, :7], 0.3) and None
    assert p._recovery is not None and p._last_recovery["active"]
    done = p._recovery_chunk(chunk, p._recovery["target"])                    # at the target: ends, chunk untouched
    assert p._recovery is None and p._last_recovery["ended"] == "reached" and np.array_equal(done, chunk)


def test_limits_and_no_trigger_after_the_grasp():
    p, model = _policy(recovery_max=1)
    q = np.zeros(len(model.joint_names))
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=_palm(model, q) + np.array([0.0, 0.0, -0.3]))
    for k in range(4):
        _step(p, q, np.zeros(16), fb=HELD if k else FREE, frame=k)
    assert p._recovery is not None
    p._recovery = None                                                          # retreat done
    for k in range(4, 9):
        _step(p, q, np.zeros(16), fb=HELD, frame=k)
    assert p._recovery is None and "recovery_max" in p._last_recovery.get("skipped", "")
    p2, _ = _policy()
    p2._latch.phase = GraspPhase.HELD
    p2.ag3s.manipulated = p.ag3s.manipulated
    for k in range(5):
        _step(p2, q, np.zeros(16), fb=HELD, frame=k)
    assert p2._recovery is None and p2._recovery_hist == []


def test_config_and_cli(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    assert TrajOptConfig().safety.recovery_retreat is False
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"safety": {"recovery_hold_streak": 0}})
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"safety": {"recovery_step_rad": 0.0}})
    args = serve_safe.build_parser().parse_args([*BASE_ARGV, "--recovery-retreat"])
    assert serve_safe.trajopt_config_from_args(args).safety.recovery_retreat is True
    assert serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args(BASE_ARGV)).safety.recovery_retreat is False
