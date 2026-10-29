"""IFG-TOA — grasp-goal anchor: avoid by changing the pose, not by moving the grasp point.

`cost.w_grasp_tcp` / `cost.w_grasp_goal` / `serve_safe --grasp-anchor W`, default off.
IFG ep954: TrajOpt kept the 5 mm neighbour clearance by translating the hand 6-19 mm (rotation <= 1.7 deg) and the
policy's grasp point followed the pushed hand. User (2026-10-21): the grasp point must stay fixed; every joint may
move within its limits.

What this file pins:
1. `anchor_qp_terms` is the exact quadratic model of `anchor_cost` for a linear point map (P, q, constant).
2. `LinkPointFK` matches the numeric link pose and a finite-difference Jacobian.
3. Optimizer: without an anchor the call is as before (no metrics key); with one, the anchored link's point moves less
   off the reference path while the collision is still resolved, and the goal rows pull the point to the goal.
4. `SafePolicy._grasp_anchor_gate`: freezes the goal on the first pre-grasp chunk that plans a close, keeps it while the
   policy's own close point drifts, clears it outside latched / closing; the anchor is off far from the goal.
5. Config / CLI.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.grasp_anchor import GraspAnchor, anchor_cost, anchor_qp_terms, anchor_targets

pytest.importorskip("casadi")
pytest.importorskip("osqp")

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


# ------------------------------------------------------------------------------------------------ 1. QP model
def test_qp_terms_are_the_exact_quadratic_of_a_linear_point_map():
    rng = np.random.default_rng(3)
    nq, H = 4, 5
    A = rng.normal(0, 0.2, (H, 3, nq)); b = rng.normal(0, 0.5, (H, 3))
    pmap = lambda Q: np.stack([A[k] @ Q[:, k] + b[k] for k in range(H)])   # noqa: E731
    R, X, Q = (rng.normal(0, 0.3, (nq, H)) for _ in range(3))
    anchor = GraspAnchor(link="l", goal=np.array([0.1, -0.2, 0.3]), close_row=2, w_tcp=7.0, w_goal=3.0)
    w, tg = anchor_targets(anchor, pmap(R))
    P, q = anchor_qp_terms(X, pmap(X), A, w, tg)
    z = Q.T.reshape(-1)
    quad = lambda zz: 0.5 * zz @ (P @ zz) + q @ zz   # noqa: E731
    # equal up to a constant: compare differences between two points
    Q2 = Q + rng.normal(0, 0.1, Q.shape); z2 = Q2.T.reshape(-1)
    assert quad(z2) - quad(z) == pytest.approx(anchor_cost(pmap(Q2), w, tg) - anchor_cost(pmap(Q), w, tg), rel=1e-9)
    tr, gl = anchor.masks(H)
    assert tr.all() and list(gl) == [False, False, True, True, True]
    assert not GraspAnchor("l", None, 2, 1.0, 1.0).masks(H)[1].any()          # no goal -> no goal rows
    assert not GraspAnchor("l", np.zeros(3), 9, 1.0, 1.0).masks(H)[1].any()   # close beyond the horizon
    assert not GraspAnchor("l", np.zeros(3), 0, 1.0, 1.0).masks(H)[1].any()   # starts closed: the rows lift, no goal
    assert list(np.flatnonzero(GraspAnchor("l", np.zeros(3), 3, 1.0, 1.0).masks(16)[1])) == [3, 4, 5]   # 3 rows only
    assert list(np.flatnonzero(GraspAnchor("l", np.zeros(3), 3, 1.0, 1.0, goal_rows=1).masks(16)[1])) == [3]


# ------------------------------------------------------------------------------------------------ 2. FK
def test_link_point_fk_matches_numeric_pose_and_finite_differences():
    from AG3S.trajopt.grasp_anchor import LinkPointFK
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=6, penetration=0.03)
    link = "link_right_arm_5"
    fk = LinkPointFK(sc.robot, sc.layout, 6, link)
    p, J = fk(sc.reference, sc.q_now)
    full = sc.layout.full_q(sc.reference, sc.q_now)
    for k in range(6):
        np.testing.assert_allclose(p[k], np.asarray(sc.robot.link_pose(full[:, k], link))[:3, 3], atol=1e-9)
    eps, k, j = 1e-6, 3, int(np.argmax(np.abs(J[3]).sum(axis=0)))
    Rp = sc.reference.copy(); Rp[j, k] += eps
    np.testing.assert_allclose((fk(Rp, sc.q_now)[0][k] - p[k]) / eps, J[k][:, j], atol=1e-5)


# ------------------------------------------------------------------------------------------------ 3. optimizer
def _solve(sc, anchor=None, **over):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    cfg = sc.config.with_overrides(over) if over else sc.config
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    kw = {} if anchor is None else {"grasp_anchor": anchor}
    return opt.solve(sc.reference, sc.q_now, sc.scene, **kw), opt


def test_anchor_off_is_as_before_and_on_moves_the_point_less():
    """Measured 2026-10-21: the anchored point leaves the policy path by 21.5 mm without the anchor, 3.9 mm with
    W = 2000, the collision resolved in both. (W = 20000 lets the anchor out-price the linear collision slack.)"""
    from AG3S.trajopt.grasp_anchor import LinkPointFK
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    H = sc.reference.shape[1]
    # the fixture's obstacle clips a sphere of link_right_arm_5; anchor a link proximal to it (as the palm is to the
    # fingers in IFG ep954) so the remaining joints can turn the clipped part away while the anchored point stays
    link = "link_right_arm_3"
    fk = LinkPointFK(sc.robot, sc.layout, H, link)
    p_ref = fk(sc.reference, sc.q_now)[0]
    base, _ = _solve(sc)
    assert "grasp_anchor" not in base.metrics
    same, _ = _solve(sc, GraspAnchor(link, None, None, 0.0, 0.0))            # weights 0 -> no anchor
    np.testing.assert_allclose(same.trajectory, base.trajectory, atol=1e-12)
    on, _ = _solve(sc, GraspAnchor(link, None, None, 2000.0, 0.0))
    dev = lambda r: float(np.max(np.linalg.norm(fk(np.asarray(r.trajectory)[:, :H], sc.q_now)[0] - p_ref, axis=1)))   # noqa: E731
    assert dev(base) > 0.005                                                   # the fixture really pushes the point
    assert dev(on) < 0.5 * dev(base)
    assert on.max_violation <= base.max_violation + 1e-3                      # still resolved as well as before
    rec = on.metrics["grasp_anchor"]
    assert rec["n_track_rows"] == H and rec["palm_dev_from_reference_max_mm"] == pytest.approx(dev(on) * 1000, abs=0.05)


def test_an_anchored_violation_falls_back_to_the_plain_solve():
    """IFG-APPLE-TOA2 ep211: the anchor priced the translation an obstacle needs, every SQP candidate was rejected and
    the chunk held for ~50 requests. Safety first: an anchored result that leaves a violation is re-solved without the
    anchor and the smaller violation wins. Fixture: anchoring the clipped link itself at W = 20000 leaves 12.4 mm."""
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    base, _ = _solve(sc)
    tol = float(sc.config.safety.violation_tolerance)
    assert base.max_violation <= tol
    on, _ = _solve(sc, GraspAnchor("link_right_arm_5", None, None, 20000.0, 0.0))
    fb = on.metrics["grasp_anchor"]["fallback"]
    assert fb["anchored_violation_mm"] > 1.0 and fb["used"] == "plain"
    assert on.max_violation <= tol
    # (not bit-identical to a fresh plain solve: the same optimizer re-solves warm-started from the anchored QP)
    ok, _ = _solve(sc, GraspAnchor("link_right_arm_3", None, None, 2000.0, 0.0))
    assert "fallback" not in ok.metrics["grasp_anchor"]                         # resolved anchored: no second solve


def test_goal_rows_pull_the_point_to_the_goal():
    from AG3S.trajopt.grasp_anchor import LinkPointFK
    from tests.trajopt.fixtures import empty_scene, make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    H = sc.reference.shape[1]
    link = "link_right_arm_5"
    p_ref = LinkPointFK(sc.robot, sc.layout, H, link)(sc.reference, sc.q_now)[0]
    goal = p_ref[-1] + np.array([0.0, 0.0, 0.01])                              # 10 mm above the policy's last point
    on, _ = _solve(sc, GraspAnchor(link, goal, H - 2, 0.0, 5000.0))
    rec = on.metrics["grasp_anchor"]
    assert rec["n_goal_rows"] == 2 and rec["goal_err_result_mm"] < rec["goal_err_reference_mm"]


# ------------------------------------------------------------------------------------------------ 4. SafePolicy gate
def _gate_policy(w=2000.0):
    from AG3S.trajopt.grasp_latch import GraspPhase
    from AG3S.trajopt.safe_policy import SafePolicy
    from tests.trajopt.test_safe_policy import _robot_model

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"cost": {"w_grasp_tcp": w, "w_grasp_goal": w}})
    p._latch = types.SimpleNamespace(phase=GraspPhase.LATCHED, config=types.SimpleNamespace(gripper_closed_below=0.85))
    p._grasp_goal = None
    p._last_grasp_anchor = {}
    p._last_grasp_record = {}
    model = _robot_model()
    p.constraint_robot_model = model
    from AG3S.trajopt.types import ChunkLayout
    p.layout = ChunkLayout.rby1(model.joint_names)
    p.grasp_links = {"left": ("ee_finger_l1", ()), "right": ("ee_finger_r1", ())}
    p.hold_links = {"left": "ee_left", "right": "ee_right"}
    from AG3S.trajopt import wire
    p.gripper_columns = wire.gripper_columns(p.layout.nq_opt // 2)
    p.ag3s = types.SimpleNamespace(manipulated=None, frame_index=7)
    return p, model


def _chunk(p, close_row=None, shift=0.0):
    ch = np.zeros((50, p.layout.action_dim))
    col = p.gripper_columns[0]
    ch[:, col] = 1.0
    if close_row is not None:
        ch[close_row:, col] = 0.0
    ch[:, p.layout.action_columns[0]] = shift                                    # left_arm_0 offset moves the palm
    return ch


def test_gate_freezes_holds_and_clears_the_goal():
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.trajopt.grasp_latch import GraspPhase

    p, model = _gate_policy()
    q = np.zeros(len(model.joint_names))
    palm = np.asarray(_link_pose_numeric(model, q, "ee_left")[:3, 3])
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=palm + np.array([0.0, 0.0, -0.03]))
    p._pending = {"chunk": _chunk(p)}
    assert p._grasp_anchor_gate(q, ["left"]) is None and p._last_grasp_anchor["reason"] == "no close planned yet"
    p._pending = {"chunk": _chunk(p, close_row=20)}
    a = p._grasp_anchor_gate(q, ["left"])
    assert a is not None and a.link == "ee_left" and a.close_row == 20 and p._last_grasp_anchor["frozen"]
    goal = a.goal.copy()
    np.testing.assert_allclose(goal, palm, atol=1e-9)                           # chunk at q (shift 0)
    # the policy's close point drifts: the goal stays where it was frozen
    p._pending = {"chunk": _chunk(p, close_row=10, shift=0.05)}
    a2 = p._grasp_anchor_gate(q, ["left"])
    np.testing.assert_allclose(a2.goal, goal) and None
    assert a2.close_row == 10
    # far from the goal -> no anchor, goal kept
    far = q.copy(); far[[i for i, n in enumerate(model.joint_names) if n == "left_arm_1"][0]] = 0.8
    assert p._grasp_anchor_gate(far, ["left"]) is None and p._last_grasp_anchor["reason"] == "palm outside the anchor radius"
    assert p._grasp_goal is not None
    # held -> cleared
    p._latch.phase = GraspPhase.HELD
    assert p._grasp_anchor_gate(q, ["left"]) is None and p._grasp_goal is None
    # off -> nothing at all
    p0, _ = _gate_policy(w=0.0)
    p0._pending = {"chunk": _chunk(p0, close_row=5)}
    assert p0._grasp_anchor_gate(q, ["left"]) is None and p0._last_grasp_anchor == {}


def test_gate_refuses_a_goal_far_from_the_target():
    p, model = _gate_policy()
    q = np.zeros(len(model.joint_names))
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=np.array([5.0, 5.0, 5.0]))
    p._pending = {"chunk": _chunk(p, close_row=3)}
    assert p._grasp_anchor_gate(q, ["left"]) is None and p._grasp_goal is None
    assert p._last_grasp_anchor["reason"] == "goal too far from the target centroid"


# ------------------------------------------------------------------------------------------------ 5. config / CLI
def test_config_and_cli(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    c = TrajOptConfig()
    assert c.cost.w_grasp_tcp == 0.0 and c.cost.w_grasp_goal == 0.0
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"cost": {"w_grasp_tcp": -1.0}})
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"cost": {"grasp_anchor_radius": 0.0}})
    args = serve_safe.build_parser().parse_args([*BASE_ARGV, "--grasp-anchor", "2000", "--grasp-anchor-radius-mm", "120"])
    over = serve_safe.cost_overrides(args)
    assert over == {"w_grasp_tcp": 2000.0, "w_grasp_goal": 2000.0, "grasp_anchor_radius": 0.12}
    assert TrajOptConfig.from_dict({"cost": over}).cost.w_grasp_goal == 2000.0
    for extra, needle in ((["--grasp-anchor-radius-mm", "120"], "--grasp-anchor"),
                          (["--grasp-anchor", "0"], "--grasp-anchor"),
                          (["--no-safe", "--grasp-anchor", "2000"], "--w-*")):
        monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *extra])
        with pytest.raises(SystemExit) as e:
            serve_safe.main()
        assert e.value.code == 2
        assert needle in capsys.readouterr().err
