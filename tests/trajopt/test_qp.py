"""The QP backends: do they agree, and does the fixed pattern actually buy anything?

The interesting question is not which is fastest — `python -m benchmark.trajopt.qp --benchmark`
answers that on the machine that will run the robot — but whether swapping one for another changes
the answer. A solver choice that moved the solution would make every measurement in the README a
measurement of the solver rather than of the method.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from benchmark.trajopt.config import QpConfig
from benchmark.trajopt.problem import build_problem
from benchmark.trajopt.qp import QpSolver, _signature, available_solvers
from benchmark.trajopt.types import JointLimits
from tests.trajopt.fixtures import small_config

NQ, HORIZON = 4, 8


@pytest.fixture
def limits():
    return JointLimits(
        lower=np.full(NQ, -2.0), upper=np.full(NQ, 2.0),
        max_step=np.full(NQ, 0.3), max_step_change=np.full(NQ, 0.15), dt=1.0 / 15.0,
    )


@pytest.fixture
def problem(limits):
    config = small_config(HORIZON)
    reference = 0.1 * np.arange(NQ)[:, None] * np.ones(HORIZON)[None, :]
    return build_problem(reference, reference, limits, config, n_slack=4)


def test_at_least_one_backend_is_available():
    assert available_solvers(), "no QP backend at all; the optimizer cannot run"
    assert "osqp" in available_solvers()


def test_qpoases_is_not_offered():
    """Excluded on structure, not preference: a dense online active-set method's cost grows
    superlinearly with the constraint count, and this problem has thousands of rows."""
    assert "qpoases" not in QpConfig.SOLVERS
    with pytest.raises(Exception):
        QpConfig(solver="qpoases").validate()


@pytest.mark.parametrize("name", [s for s in ("osqp", "proxqp", "qrqp") if s in available_solvers()])
def test_the_backends_agree_on_the_same_problem(problem, name):
    """Loose tolerance on purpose: these are different algorithms at 1e-3, and what must match is the
    trajectory, not the last digit."""
    reference = QpSolver(QpConfig(solver="osqp")).solve(problem)
    other = QpSolver(QpConfig(solver=name)).solve(problem)
    assert other.success, other.status
    a, _ = problem.split(reference.x)
    b, _ = problem.split(other.x)
    assert np.abs(a - b).max() < 5e-3, f"{name} disagreed with osqp by {np.abs(a - b).max():.2e}"


def test_a_solved_problem_reports_a_usable_solution(problem):
    solution = QpSolver().solve(problem)
    assert solution.success and np.all(np.isfinite(solution.x))
    assert solution.n_var == problem.n_var
    assert solution.solve_time_ms > 0.0


def test_the_signature_ignores_values_and_notices_structure(problem, limits):
    """What a solver may reuse its factorization across. Getting this wrong either throws the
    factorization away every iteration or reuses one that no longer fits."""
    config = small_config(HORIZON)
    same_shape = build_problem(
        np.ones((NQ, HORIZON)), np.ones((NQ, HORIZON)), limits, config, n_slack=4
    )
    assert _signature(problem) == _signature(same_shape)

    wider = build_problem(
        np.zeros((NQ, HORIZON)), np.zeros((NQ, HORIZON)), limits, config, n_slack=9
    )
    assert _signature(problem) != _signature(wider)


def test_a_reused_solver_returns_the_same_answer_as_a_fresh_one(problem, limits):
    """Warm-starting is an optimization, not a change of problem. If it moved the answer, every
    result would depend on what was solved before it."""
    config = small_config(HORIZON)
    second = build_problem(
        np.full((NQ, HORIZON), 0.4), np.full((NQ, HORIZON), 0.4), limits, config, n_slack=4
    )
    warm = QpSolver()
    warm.solve(problem)
    reused = warm.solve(second)
    fresh = QpSolver().solve(second)
    assert np.abs(problem.split(reused.x)[0] - problem.split(fresh.x)[0]).max() < 5e-3


def test_reset_drops_the_cached_factorization(problem):
    solver = QpSolver()
    solver.solve(problem)
    assert solver._backend is not None
    solver.reset()
    assert solver._backend is None and solver._signature is None


def test_an_infeasible_problem_is_reported_rather_than_raised():
    """The caller needs an answer, even a bad one. Raising from inside the control loop is how a
    safety layer gets switched off."""
    n = 3
    contradiction = build_problem(
        np.zeros((n, 2)), np.zeros((n, 2)),
        JointLimits(np.zeros(n), np.zeros(n), np.full(n, 1e-9), np.full(n, 1e-9), 1.0),
        small_config(2),
        q_now=np.full(n, 100.0), n_slack=0,
    )
    solution = QpSolver().solve(contradiction)
    assert isinstance(solution.success, bool)
    assert solution.x.shape[0] == contradiction.n_var
