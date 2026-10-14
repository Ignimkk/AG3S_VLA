"""V11AB-pb2 — minimal intervention gate (`SqpConfig.minimal_intervention`): a reference that already clears every
certified row by the threshold is returned as the plan, certified exactly as a solve result."""
from __future__ import annotations

import numpy as np
import pytest

from AG3S.trajopt.sqp import TrajectoryOptimizer
from AG3S.trajopt.types import TrajOptStatus
from tests.trajopt.fixtures import make_scenario

pytest.importorskip("casadi")
pytest.importorskip("osqp")


def _opt(sc, mi=None):
    cfg = sc.config if mi is None else sc.config.with_overrides({"sqp": {"minimal_intervention": mi}})
    return TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)


def _ref_clearance(sc):
    return float(sc.linearizer.step_clearances(sc.reference, sc.q_now, sc.scene).min())


def test_clear_reference_is_returned_bit_for_bit_and_certified():
    sc = make_scenario(horizon=10, penetration=-0.05)
    clear = _ref_clearance(sc)
    assert clear > 0.015
    res = _opt(sc, mi=0.01).solve(sc.reference, sc.q_now, sc.scene)
    assert np.array_equal(res.trajectory, sc.reference)
    assert res.status is TrajOptStatus.OPTIMAL and res.status.safe and res.iterations == 0
    rec = res.metrics["minimal_intervention"]
    assert rec["applied"] and rec["threshold_m"] == 0.01 and rec["reference_clearance_m"] == pytest.approx(clear)


def test_v11ab_default_has_no_gate():
    sc = make_scenario(horizon=10, penetration=-0.05)
    res = _opt(sc).solve(sc.reference, sc.q_now, sc.scene)
    assert "minimal_intervention" not in res.metrics


def test_a_reference_within_the_threshold_is_still_optimized():
    sc = make_scenario(horizon=10, penetration=-0.005)
    clear = _ref_clearance(sc)
    res = _opt(sc, mi=clear + 0.005).solve(sc.reference, sc.q_now, sc.scene)
    assert res.metrics["minimal_intervention"]["applied"] is False
    assert res.iterations >= 1


def test_a_colliding_reference_is_optimized_and_made_safe():
    sc = make_scenario(horizon=10, penetration=0.03)
    res = _opt(sc, mi=0.01).solve(sc.reference, sc.q_now, sc.scene)
    assert res.metrics["minimal_intervention"]["applied"] is False
    assert res.safe and res.max_violation <= sc.config.safety.violation_tolerance
