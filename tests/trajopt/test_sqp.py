"""The optimizer's contract: it makes the chunk safe, or it says it did not.

The assertions here are weighted toward what happens when things go wrong, because that is where a
safety layer earns its place. A optimizer that works on a clean scene and raises on a dirty one is
worse than useless — it fails at exactly the moment the controller has nothing else to fall back on.

Every "is it safe" assertion goes through `full_violation`, which evaluates **all** of AG3S's rows at
full resolution. None of them reads the reduced set the QP actually saw.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.trajopt.limits import limit_report
from benchmark.trajopt.linearize import SceneSnapshot
from benchmark.trajopt.sqp import TrajectoryOptimizer
from benchmark.trajopt.types import TrajOptStatus
from tests.trajopt.fixtures import Q_HOME, empty_scene, make_scenario

pytest.importorskip("casadi")
pytest.importorskip("osqp")


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(horizon=10, penetration=0.03)


def _optimizer(scenario, **overrides):
    config = scenario.config.with_overrides(overrides) if overrides else scenario.config
    return TrajectoryOptimizer(scenario.robot, scenario.layout, scenario.limits, config)


# --------------------------------------------------------------- the happy path


def test_a_colliding_reference_comes_back_collision_free(scenario):
    """The whole point, measured at full resolution rather than against the QP's own rows."""
    optimizer = _optimizer(scenario)
    result = optimizer.solve(scenario.reference, scenario.q_now, scenario.scene)

    assert scenario.reference_violation > 0.02, "the fixture stopped posing a problem"
    assert result.max_violation <= scenario.config.safety.violation_tolerance
    assert result.safe and result.improved
    verified = optimizer.linearizer.full_violation(
        result.trajectory, scenario.q_now, scenario.scene
    )
    assert verified >= -scenario.config.safety.violation_tolerance


def test_the_solution_stays_close_to_the_reference(scenario):
    """Safety is not the only requirement: a trajectory that abandons the task is also a failure, and
    no collision metric would notice."""
    result = _optimizer(scenario).solve(scenario.reference, scenario.q_now, scenario.scene)
    assert result.reference_deviation < 1.0, "the optimizer wandered off the policy's intent"
    assert result.reference_deviation > 1e-6, "nothing moved; the constraint never bound"


def test_the_solution_respects_the_robot_own_limits(scenario):
    result = _optimizer(scenario).solve(scenario.reference, scenario.q_now, scenario.scene)
    report = limit_report(result.trajectory, scenario.limits)
    assert report["position"] < 1e-6
    # Velocity and acceleration are softened so the QP always returns; any residual must be tiny.
    assert report["velocity"] < 1e-3 and report["acceleration"] < 1e-3


def test_a_clear_reference_is_returned_essentially_unchanged(scenario):
    """No obstacle, no correction. Otherwise the optimizer is a source of motion of its own."""
    optimizer = _optimizer(scenario)
    scene = empty_scene(optimizer.linearizer)
    result = optimizer.solve(scenario.reference, scenario.q_now, scene)
    assert result.status is TrajOptStatus.UNCONSTRAINED
    assert np.allclose(result.trajectory, scenario.reference)


def test_an_unconstrained_frame_is_not_called_safe(scenario):
    """"There was nothing to avoid" and "I checked and found nothing" are different claims. An empty
    constraint set can mean the cameras failed."""
    optimizer = _optimizer(scenario)
    result = optimizer.solve(scenario.reference, scenario.q_now, empty_scene(optimizer.linearizer))
    assert not result.status.safe
    assert TrajOptStatus.UNCONSTRAINED.safe is False


# --------------------------------------------------------------- degradation


def test_a_budget_that_stops_the_loop_still_returns_the_best_iterate(scenario):
    """The real-time iteration scheme's central promise: a deadline yields an answer, not an
    exception and not a half-written trajectory."""
    optimizer = _optimizer(scenario, sqp={"max_iterations": 8, "time_budget_ms": 0.001})
    result = optimizer.solve(scenario.reference, scenario.q_now, scenario.scene)
    assert result.iterations >= 1, "the budget must not prevent even one step"
    assert np.all(np.isfinite(result.trajectory))
    assert any("budget" in note for note in result.notes)


def test_an_unfixable_scene_reports_violation_instead_of_raising(scenario):
    """The robot already deep inside an obstacle. There is no safe answer, and the caller still needs
    one — plus the number that says how bad it is."""
    optimizer = _optimizer(scenario)
    centres, _ = optimizer.linearizer.sphere_states(scenario.reference, scenario.q_now)
    engulfing = SceneSnapshot(
        candidate_pos=np.tile(centres[0, 0], (4, 1)),
        candidate_radius=np.full(4, 0.8),  # swallows the whole robot
        candidate_active=np.ones(4, bool),
        d_safe=np.full((optimizer.linearizer.n_spheres, 4), 0.05),
        plane_normal=np.tile(np.array([0.0, 0.0, 1.0]), (2, 1)),
        plane_offset=np.zeros(2), plane_active=np.zeros(2, bool),
        robot_radii=optimizer.linearizer.robot_radii, candidate_ids=np.arange(4),
    )
    result = optimizer.solve(scenario.reference, scenario.q_now, engulfing)
    assert result.status is TrajOptStatus.VIOLATED
    assert not result.safe
    assert result.max_violation > 0.0
    assert np.all(np.isfinite(result.trajectory)), "a best effort is still a usable trajectory"
    assert any("best effort" in note for note in result.notes)


def test_uncertified_geometry_cannot_be_called_safe(scenario):
    """The trajectory clears everything AG3S described; AG3S said it could not describe everything.
    "Clear" is then a statement about the model, not about the world."""
    optimizer = _optimizer(scenario)
    certified = optimizer.solve(scenario.reference, scenario.q_now, scenario.scene)
    assert certified.safe

    optimizer.reset()
    uncertified = optimizer.solve(
        scenario.reference, scenario.q_now, scenario.scene, geometry_certified=False
    )
    assert uncertified.max_violation <= scenario.config.safety.violation_tolerance
    assert uncertified.status is TrajOptStatus.VIOLATED
    assert any("could not certify" in note for note in uncertified.notes)


def test_the_certification_check_can_be_switched_off_for_an_ablation(scenario):
    optimizer = _optimizer(scenario, safety={"require_certified_geometry": False})
    result = optimizer.solve(
        scenario.reference, scenario.q_now, scenario.scene, geometry_certified=False
    )
    assert result.safe


# --------------------------------------------------------------- the reductions, end to end


def test_pruning_reaches_the_same_place_as_solving_every_row(scenario):
    """The reduction is allowed to cost convergence. It is not allowed to change the answer."""
    full = _optimizer(
        scenario, reduction={"enabled": False, "rows_per_step": 4096, "activation_band": 1e6}
    ).solve(scenario.reference, scenario.q_now, scenario.scene)
    pruned = _optimizer(scenario).solve(scenario.reference, scenario.q_now, scenario.scene)

    assert full.safe and pruned.safe
    assert np.abs(full.trajectory - pruned.trajectory).max() < 0.05, (
        "pruning moved the solution; the activation band is too tight"
    )


def test_the_backoff_is_what_closes_the_last_millimetre(scenario):
    """Without it the SQP converges with the *linearized* rows satisfied and real penetration left —
    the distance to a sphere is concave along a straight line in joint space, so the truth sits below
    its own tangent."""
    without = _optimizer(scenario, reduction={"linearization_backoff": 0.0}).solve(
        scenario.reference, scenario.q_now, scenario.scene
    )
    with_backoff = _optimizer(scenario).solve(scenario.reference, scenario.q_now, scenario.scene)
    assert with_backoff.max_violation <= without.max_violation


# --------------------------------------------------------------- cross-chunk state


def test_the_warm_start_carries_across_chunks_without_changing_the_answer(scenario):
    optimizer = _optimizer(scenario)
    first = optimizer.solve(scenario.reference, scenario.q_now, scenario.scene)
    second = optimizer.solve(scenario.reference, scenario.q_now, scenario.scene)
    assert second.safe
    assert np.abs(first.trajectory - second.trajectory).max() < 0.05

    optimizer.reset()
    assert optimizer._previous is None
    assert optimizer.frame_index == 0


def test_every_frame_reports_where_its_milliseconds_went(scenario):
    """Where the time goes is the single most useful thing to know about a real-time optimizer, and
    it changes with the scene."""
    result = _optimizer(scenario).solve(scenario.reference, scenario.q_now, scenario.scene)
    timing = result.metrics["timing_ms"]
    assert set(timing) == {"linearize", "assemble", "qp", "check"}
    assert sum(timing.values()) <= result.solve_time_ms + 1e-6
    assert result.metrics["qp_iterations"] > 0
    assert result.metrics["qp_solver"] == scenario.config.qp.solver


def test_the_chunk_is_built_only_from_columns_the_layout_owns(scenario):
    template = np.full((scenario.horizon, scenario.layout.action_dim), 9.0)
    result = _optimizer(scenario).solve(
        scenario.reference, scenario.q_now, scenario.scene, template=template
    )
    untouched = list(scenario.layout.passthrough)
    assert np.array_equal(result.chunk[:, untouched], template[:, untouched])
