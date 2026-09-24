"""The quadratic subproblem: does it encode the cost and the physics it claims to?

These tests deliberately run with **no collision rows**, because that is the one case where the right
answer is known exactly and independently: with nothing to avoid, the minimiser of a tracking cost is
the reference. Anything else here is a bug in the assembly rather than a disagreement about geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.trajopt.config import TrajOptConfig
from benchmark.trajopt.problem import build_problem, difference_operator, objective
from benchmark.trajopt.types import JointLimits

osqp = pytest.importorskip("osqp", reason="the subproblem is a QP")
import scipy.sparse as sp  # noqa: E402

NQ, HORIZON = 4, 8


@pytest.fixture
def limits():
    return JointLimits(
        lower=np.full(NQ, -3.0), upper=np.full(NQ, 3.0),
        max_step=np.full(NQ, 0.2), max_step_change=np.full(NQ, 0.08), dt=1.0 / 15.0,
    )


@pytest.fixture
def config():
    return TrajOptConfig.from_dict(
        {"horizon": {"horizon": HORIZON, "execution_length": 4, "plan_horizon": None}}
    )


def _solve(problem, eps=1e-9):
    solver = osqp.OSQP()
    solver.setup(
        P=sp.triu(problem.P, format="csc"), q=problem.q, A=problem.A,
        l=np.clip(problem.l, -1e20, 1e20), u=np.clip(problem.u, -1e20, 1e20),
        verbose=False, eps_abs=eps, eps_rel=eps, polish=True, max_iter=20000,
    )
    result = solver.solve()
    assert result.info.status in ("solved", "solved inaccurate"), result.info.status
    return problem.split(result.x)[0]


@pytest.mark.parametrize("order", [1, 2])
def test_the_difference_operator_matches_numpy(order):
    """Built from diagonals for speed; checked against the definition."""
    Q = np.random.default_rng(0).normal(0, 1, (NQ, HORIZON))
    D = difference_operator(NQ, HORIZON, order)
    got = (D @ Q.T.reshape(-1)).reshape(HORIZON - order, NQ).T
    assert np.allclose(got, np.diff(Q, n=order, axis=1))


def test_a_degenerate_horizon_yields_an_empty_operator():
    assert difference_operator(NQ, 2, 2).shape == (0, NQ * 2)


def _feasible_reference(limits, horizon=HORIZON, rng=None):
    """A reference the robot could actually track: a ramp well inside every kinematic bound.

    A random trajectory is not a valid input to this test. Its second differences are far outside the
    acceleration limit, so the optimizer correctly moves away from it, and "did the optimizer return
    the reference" becomes unanswerable.
    """
    slope = 0.2 * limits.max_step[:, None]
    return slope * np.arange(horizon)[None, :]


def test_with_nothing_to_avoid_the_minimiser_is_the_reference(limits, config):
    """The load-bearing baseline. If this drifts, every collision result is measured against a
    trajectory the optimizer was already pulling away from."""
    reference = _feasible_reference(limits)
    plain = config.with_overrides({"cost": {"w_smooth": 0.0, "w_continuity": 0.0}})
    solved = _solve(build_problem(reference, reference, limits, plain, n_slack=0))
    assert np.abs(solved - reference).max() < 1e-6


def test_a_kinematically_impossible_reference_still_solves(limits, config):
    """A policy chunk can ask for motion the robot cannot perform. Softening the velocity and
    acceleration rows means the QP answers with the closest thing it *can* do, instead of returning
    "primal infeasible" and leaving the controller with nothing.
    """
    reference = np.random.default_rng(1).normal(0, 0.2, (NQ, HORIZON))
    assert np.abs(np.diff(reference, n=2, axis=1)).max() > limits.max_step_change.max(), (
        "this reference is within the limits; the test would prove nothing"
    )
    problem = build_problem(reference, reference, limits, config, n_slack=0, trust_radius=10.0)
    solved = _solve(problem)
    assert np.all(np.isfinite(solved))
    # The limits are soft, so a little overshoot is possible — but the solution must be far closer to
    # feasible than the reference it started from.
    from benchmark.trajopt.limits import limit_report

    assert limit_report(solved, limits)["acceleration"] < limit_report(reference, limits)["acceleration"]


def test_the_round_trip_between_the_vector_and_the_trajectory_is_exact(limits, config):
    problem = build_problem(
        np.zeros((NQ, HORIZON)), np.zeros((NQ, HORIZON)), limits, config, n_slack=5
    )
    Q = np.random.default_rng(2).normal(0, 1, (NQ, HORIZON))
    slack = np.arange(5, dtype=float)
    back_Q, back_s = problem.split(problem.flatten(Q, slack))
    assert np.allclose(back_Q, Q) and np.allclose(back_s, slack)


def test_the_velocity_limit_actually_binds(limits, config):
    """A reference that steps faster than the robot may move must come back slowed, not copied."""
    reference = np.zeros((NQ, HORIZON))
    reference[0] = np.arange(HORIZON) * (limits.max_step[0] * 3.0)
    plain = config.with_overrides({"cost": {"w_smooth": 0.0, "w_continuity": 0.0}})
    solved = _solve(build_problem(reference, reference, limits, plain, n_slack=0,
                                  trust_radius=10.0))
    assert np.abs(np.diff(solved, axis=1)).max() <= limits.max_step.max() + 1e-6
    assert np.abs(solved - reference).max() > 1e-3, "the limit did not bind; the test proves nothing"


def test_the_acceleration_limit_actually_binds(limits, config):
    reference = np.zeros((NQ, HORIZON))
    reference[0] = np.cumsum(np.arange(HORIZON) * limits.max_step_change[0] * 3.0)
    plain = config.with_overrides({"cost": {"w_smooth": 0.0, "w_continuity": 0.0}})
    solved = _solve(build_problem(reference, reference, limits, plain, n_slack=0,
                                  trust_radius=10.0))
    assert np.abs(np.diff(solved, n=2, axis=1)).max() <= limits.max_step_change.max() + 1e-6


def test_the_joint_box_binds(limits, config):
    reference = np.full((NQ, HORIZON), 10.0)
    solved = _solve(build_problem(reference, reference, limits, config, n_slack=0,
                                  trust_radius=100.0))
    assert np.all(solved <= limits.upper[:, None] + 1e-6)


def test_the_first_step_is_anchored_to_where_the_robot_is_not_pinned_to_it(limits, config):
    """`q_now` bounds the first step by one step's travel. Pinning it would forbid the robot from
    moving during the very control period the chunk exists to fill."""
    q_now = np.zeros(NQ)
    reference = np.full((NQ, HORIZON), 5.0)
    plain = config.with_overrides({"cost": {"w_smooth": 0.0, "w_continuity": 0.0}})
    solved = _solve(build_problem(reference, reference, limits, plain, q_now=q_now,
                                  n_slack=0, trust_radius=100.0))
    assert np.abs(solved[:, 0] - q_now).max() <= limits.max_step.max() + 1e-6
    assert np.abs(solved[:, 0] - q_now).max() > 1e-6, "anchoring became pinning"


def test_the_trust_region_confines_the_step(limits, config):
    reference = np.full((NQ, HORIZON), 2.0)
    iterate = np.zeros((NQ, HORIZON))
    solved = _solve(build_problem(reference, iterate, limits, config, n_slack=0,
                                  trust_radius=0.05))
    assert np.abs(solved - iterate).max() <= 0.05 + 1e-6


def test_a_robot_starting_outside_its_limits_still_produces_a_solvable_problem(limits, config):
    """After a fault or a bad hand-off the box and the anchor can have no intersection. Failing there
    would leave the caller with nothing at the moment it most needs something."""
    outside = limits.upper + 1.0
    problem = build_problem(
        np.zeros((NQ, HORIZON)), np.zeros((NQ, HORIZON)), limits, config,
        q_now=outside, n_slack=0,
    )
    assert np.all(problem.l <= problem.u + 1e-12)
    _solve(problem)


def test_the_continuity_term_pulls_toward_the_previous_chunk(limits, config):
    """The mitigation for SEAM never seeing the correction. Without it the two layers fight."""
    reference = np.zeros((NQ, HORIZON))
    previous = np.full((NQ, HORIZON), 0.4)
    weighted = config.with_overrides({"cost": {"w_smooth": 0.0, "w_continuity": 4.0}})
    solved = _solve(
        build_problem(reference, reference, limits, weighted, previous_chunk=previous,
                      n_slack=0, trust_radius=10.0)
    )
    assert np.all(solved > 0.1), "the continuity term had no effect"
    assert np.all(solved < 0.4), "the continuity term overwhelmed tracking"


def test_the_objective_agrees_with_what_the_matrices_encode(limits, config):
    """`objective` is the SQP's independent witness; if it were derived from the same assembly it
    would agree with the QP even when both are wrong."""
    rng = np.random.default_rng(3)
    reference = rng.normal(0, 0.3, (NQ, HORIZON))
    Q = reference + rng.normal(0, 0.05, (NQ, HORIZON))
    problem = build_problem(reference, Q, limits, config, n_slack=0)
    z = problem.flatten(Q)
    quadratic = 0.5 * z @ (problem.P @ z) + problem.q @ z
    # The matrix form drops the constant term of ||Q - ref||^2, so compare differences.
    reference_z = problem.flatten(reference)
    baseline = 0.5 * reference_z @ (problem.P @ reference_z) + problem.q @ reference_z
    assert quadratic - baseline == pytest.approx(
        objective(Q, reference, config) - objective(reference, reference, config), rel=1e-9
    )


def test_slack_is_penalised_linearly_and_bounded_below(limits, config):
    problem = build_problem(
        np.zeros((NQ, HORIZON)), np.zeros((NQ, HORIZON)), limits, config, n_slack=6
    )
    collision = slice(problem.n_q, problem.n_q + problem.n_slack)
    kinematic = slice(problem.n_q + problem.n_slack, problem.n_var)
    assert np.allclose(problem.q[collision], config.cost.w_slack)
    # Breaking the robot is worse than grazing an obstacle, and unlike a collision it is certain
    # rather than predicted — so the kinematic slack costs an order of magnitude more.
    assert np.all(problem.q[kinematic] > problem.q[collision].max())
    everything = slice(problem.n_q, problem.n_var)
    assert problem.P[everything, everything].nnz == 0, "slack must not be penalised quadratically"
    lo, hi = problem.row_blocks["slack"]
    assert np.allclose(problem.l[lo:hi], 0.0) and np.all(np.isinf(problem.u[lo:hi]))
