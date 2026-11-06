"""T94q — CBF-QP filter on AG3S rows: shape, no-op far from obstacles, deflection near them."""
import numpy as np
import pytest

from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS, load_rby1
from AG3S.trajopt.cbf_filter import CbfSafetyFilter
from AG3S.trajopt.config import TrajOptConfig
from AG3S.trajopt.limits import build_limits
from AG3S.trajopt.sqp import TrajectoryOptimizer
from AG3S.trajopt.types import ChunkLayout, TrajOptStatus
from tests.trajopt.fixtures import Q_HOME, near_miss_scene, sweep_reference


@pytest.fixture(scope="module")
def setup():
    config = TrajOptConfig.from_dict({"horizon": {"horizon": 32, "execution_length": 8, "plan_horizon": None}})
    robot = load_rby1()
    layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
    limits = build_limits(robot, layout, dt=config.horizon.dt, config=config.limits)
    cbf = CbfSafetyFilter(robot, layout, limits, config)
    to = TrajectoryOptimizer(robot, layout, limits, config)
    ref = sweep_reference(layout, 32)
    scene = near_miss_scene(cbf.linearizer, ref, Q_HOME)
    return cbf, to, ref, scene, layout


def test_cbf_improves_near_miss(setup):
    cbf, to, ref, scene, layout = setup
    res = cbf.solve(ref, Q_HOME, scene)
    assert res.trajectory.shape == ref.shape
    assert "cbf" in res.metrics and len(res.metrics["cbf"]["steps"]) == 32
    import time
    t = time.perf_counter(); res = cbf.solve(ref, Q_HOME, scene); print("warm ms", (time.perf_counter() - t) * 1e3)
    win = cbf.lookahead_from
    c_ref, _ = cbf.linearizer.worst_row(ref, Q_HOME, scene)
    c_cbf, _ = cbf.linearizer.worst_row(res.trajectory, Q_HOME, scene)
    assert res.status.safe
    print("ref clearance", c_ref, "cbf clearance", c_cbf, res.status, res.solve_time_ms)
    assert c_cbf >= c_ref - 1e-9
    to_res = to.solve(ref, Q_HOME, scene)
    t = time.perf_counter(); to_res = to.solve(ref, Q_HOME, scene); print("TO warm ms", (time.perf_counter() - t) * 1e3)
    print("TO", to_res.status, to_res.max_violation, "dev cbf", res.reference_deviation, "dev to", to_res.reference_deviation)


def test_cbf_unconstrained_passthrough(setup):
    cbf, _, ref, _, _ = setup
    res = cbf.solve(ref, Q_HOME, None)
    assert res.status == TrajOptStatus.UNCONSTRAINED
    np.testing.assert_array_equal(res.trajectory, ref)
