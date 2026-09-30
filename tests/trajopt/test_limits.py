"""Joint bounds: parsed from the URDF, converted to step units, and never invented."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmark.ag3s.robot_models import DEFAULT_RBY1_JOINTS, load_rby1
from benchmark.trajopt.config import LimitsConfig
from benchmark.trajopt.limits import build_limits, clamp_to_limits, limit_report
from benchmark.trajopt.types import ChunkLayout, JointLimits

DT = 1.0 / 15.0


@pytest.fixture(scope="module")
def robot():
    return load_rby1()


@pytest.fixture(scope="module")
def layout():
    return ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)


def test_the_urdf_supplies_every_limit_this_robot_needs(robot):
    """No joint falls back to a default, so no default is silently enforced on hardware."""
    velocity = robot.velocity_limits()
    acceleration = robot.acceleration_limits()
    assert np.all(np.isfinite(velocity)) and np.all(velocity > 0)
    assert np.all(np.isfinite(acceleration)) and np.all(acceleration > 0)
    lower, upper = robot.joint_limits()
    assert np.all(lower < upper)


def test_a_missing_limit_raises_rather_than_defaulting(robot):
    """`None` means "the URDF does not say", and inventing a number for a real robot is worse than
    stopping to ask."""

    class Incomplete:
        nq = 2
        joint_names = ("a", "b")

        def joint_limits(self):
            return np.array([-1.0, -1.0]), np.array([1.0, 1.0])

    layout = ChunkLayout(
        action_dim=4, n_valid=4, action_to_q=((0, 0), (1, 1)), fixed_q=(), passthrough=(2, 3),
        nq_model=2,
    )
    # T31: position 범위의 기본 출처는 MJCF 인데 이 가짜 로봇(관절 a·b)은 어느 MJCF 에도 없다.
    # 이 시험의 요점은 **속도** 기본값이므로 position 은 모델 자신(`source="urdf"`)에서 읽는다.
    with pytest.raises(ValueError, match="no velocity_limits"):
        build_limits(Incomplete(), layout, dt=DT, config=LimitsConfig(source="urdf"))
    limits = build_limits(
        Incomplete(), layout, dt=DT,
        config=LimitsConfig(source="urdf", default_velocity=1.0, default_acceleration=2.0),
    )
    assert limits.max_step[0] == pytest.approx(1.0 * 0.9 * DT)


def test_limits_are_converted_into_step_units_once(robot, layout):
    """`v_max * dt` and `a_max * dt^2`, with the conservative scale applied — checked against the
    URDF rather than against another copy of the same arithmetic."""
    config = LimitsConfig()
    limits = build_limits(robot, layout, dt=DT, config=config)
    index = layout.q_indices
    assert limits.nq == layout.nq_opt
    assert np.allclose(
        limits.max_step, robot.velocity_limits()[index] * config.velocity_scale * DT
    )
    assert np.allclose(
        limits.max_step_change,
        robot.acceleration_limits()[index] * config.acceleration_scale * DT * DT,
    )


@pytest.mark.parametrize("source", ["model_xml", "urdf"])
def test_the_position_margin_keeps_the_optimizer_off_the_hard_stop(robot, layout, source):
    """margin 은 **설정된 출처의** 범위에서 잰다 (T31 G1).

    T31 전에는 `robot.joint_limits()` (URDF) 와 비교했다. 이제 기본 출처는 제어 대상 MJCF 이고
    — MJCF 의 `arm_1` 하한(−0.05)은 URDF(−0.0175)보다 넓다 — URDF 와 비교하면 margin 이 아니라
    두 파일의 차이를 재게 된다. 출처마다 그 출처의 범위를 기준으로 같은 성질을 확인한다.
    """
    from benchmark.trajopt.limits import resolve_position_ranges

    config = LimitsConfig(source=source, position_margin=0.05)
    lower_all, upper_all, _ = resolve_position_ranges(robot, config, index=layout.q_indices)
    index = layout.q_indices
    tight = build_limits(robot, layout, dt=DT, config=config)
    assert np.all(tight.lower > lower_all[index])
    assert np.all(tight.upper < upper_all[index])
    assert np.allclose(tight.lower - lower_all[index], 0.05)
    assert np.allclose(upper_all[index] - tight.upper, 0.05)


def test_an_absurd_margin_is_refused_rather_than_producing_an_empty_box(robot, layout):
    with pytest.raises(ValueError, match="closes the range"):
        build_limits(robot, layout, dt=DT, config=LimitsConfig(position_margin=5.0))


def test_disabling_a_limit_widens_it_instead_of_removing_the_constraint(robot, layout):
    """The ablation must change one number, not the structure of the problem — otherwise the
    comparison is between two different optimizers."""
    off = build_limits(
        robot, layout, dt=DT,
        config=LimitsConfig(enforce_velocity=False, enforce_acceleration=False),
    )
    assert np.all(np.isinf(off.max_step))
    assert np.all(np.isinf(off.max_step_change))


def test_the_report_measures_overshoot_rather_than_flagging_it(robot, layout):
    """A micron from the QP's tolerance and a joint driven through its stop are the same boolean and
    very different problems."""
    limits = build_limits(robot, layout, dt=DT)
    inside = np.tile(
        (0.5 * (limits.lower + limits.upper))[:, None], (1, 6)
    )
    assert max(limit_report(inside, limits).values()) == pytest.approx(0.0)

    jumped = inside.copy()
    jumped[0, 3] += 10.0 * limits.max_step[0]
    report = limit_report(jumped, limits)
    assert report["velocity"] > 0.0
    assert report["acceleration"] > 0.0
    assert report["position"] == pytest.approx(0.0), "the jump stayed inside the box"


def test_clamping_touches_position_only(robot, layout):
    """Clamping velocity after the fact would satisfy the limit while breaking the collision
    constraints that were solved around the trajectory's shape."""
    limits = build_limits(robot, layout, dt=DT)
    wild = np.tile(limits.upper[:, None] + 1.0, (1, 5))
    clamped = clamp_to_limits(wild, limits)
    assert np.all(clamped <= limits.upper[:, None] + 1e-12)
    assert limit_report(clamped, limits)["position"] == pytest.approx(0.0)


def test_inconsistent_limits_are_rejected_at_construction():
    with pytest.raises(ValueError, match="lower > upper"):
        JointLimits(np.ones(2), np.zeros(2), np.ones(2), np.ones(2), DT)
    with pytest.raises(ValueError, match="must be positive"):
        JointLimits(np.zeros(2), np.ones(2), np.zeros(2), np.ones(2), DT)
