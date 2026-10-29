"""IFG-TOW — wrist-weighted tracking: near the target, avoid with the wrist (orientation) before the shoulder (position).

`cost.wrist_track_scale` / `serve_safe --wrist-track-scale S`, default off (1.0).
IFG ep954 per-joint analysis (2026-10-21): TCP translation per degree is 12.6 mm at arm_0/1, 3.5-7 at arm_2, 7.6 at
arm_3, 3.5 at arm_4, 4.4 at arm_5, 0 at arm_6; the target-split run avoided the banana mainly with arm_0 and arm_3, so
the hand translated and the grasp point moved. User (2026-10-21): joints 1-3 place the hand, 4-6 orient it; more edit
on the distal joints should change position less and orientation more -- but hurdles and whole-path edits need every
joint, so the scale is gated to pre-grasp chunks near the target.

What this file pins:
1. `build_problem` / `objective`: no scale = as before; a scale multiplies each joint's tracking weight, and the QP's
   quadratic matches the merit.
2. Optimizer: scaling the clipped arm's distal joints moves the edit from the proximal to the distal joints while the
   collision is still resolved; scale 1 = the plain solve.
3. `SafePolicy._wrist_track_gate`: off by default; on only latched / closing with the palm within the radius; only the
   grasping arm's arm_4/5/6 get the scale.
4. Config / CLI.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

pytest.importorskip("casadi")
pytest.importorskip("osqp")

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


# ------------------------------------------------------------------------------------------------ 1. QP / merit
def test_problem_and_objective_scale_each_joint():
    from AG3S.trajopt.problem import build_problem, objective
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=6, penetration=0.03)
    nq, H = sc.reference.shape
    kw = dict(q_now=sc.q_now[sc.layout.q_indices], n_slack=0)
    plain = build_problem(sc.reference, sc.reference, sc.limits, sc.config, **kw)
    ones = build_problem(sc.reference, sc.reference, sc.limits, sc.config, joint_track_scale=np.ones(nq), **kw)
    np.testing.assert_allclose(ones.P.toarray(), plain.P.toarray())
    np.testing.assert_allclose(ones.q, plain.q)
    scale = np.linspace(0.1, 1.0, nq)
    on = build_problem(sc.reference, sc.reference, sc.limits, sc.config, joint_track_scale=scale, **kw)
    rng = np.random.default_rng(0)
    Q1, Q2 = (sc.reference + rng.normal(0, 0.05, (nq, H)) for _ in range(2))
    quad = lambda pr, Q: 0.5 * Q.T.reshape(-1) @ (pr.P[:nq * H, :nq * H] @ Q.T.reshape(-1)) + pr.q[:nq * H] @ Q.T.reshape(-1)   # noqa: E731
    obj = lambda Q, s: objective(Q, sc.reference, sc.config, joint_track_scale=s)   # noqa: E731
    assert obj(Q1, None) == pytest.approx(obj(Q1, np.ones(nq)), rel=1e-12)
    # tracking and smoothing are both quadratic: the QP difference equals the merit difference
    assert quad(on, Q2) - quad(on, Q1) == pytest.approx(obj(Q2, scale) - obj(Q1, scale), rel=1e-7)
    assert obj(Q1, scale) < obj(Q1, None)


# ------------------------------------------------------------------------------------------------ 2. optimizer
def _solve(sc, scale=None):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    return opt.solve(sc.reference, sc.q_now, sc.scene, **({} if scale is None else {"joint_track_scale": scale}))


def test_distal_scale_moves_the_edit_to_the_distal_joints():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    H = sc.reference.shape[1]
    names = [str(sc.layout.joint_names[i]) for i in sc.layout.q_indices]
    distal = np.array([n in ("right_arm_4", "right_arm_5", "right_arm_6") for n in names])
    proximal = np.array([n in ("right_arm_0", "right_arm_1", "right_arm_2", "right_arm_3") for n in names])
    tol = float(sc.config.safety.violation_tolerance)
    base = _solve(sc)
    same = _solve(sc, np.ones(len(names)))
    np.testing.assert_allclose(same.trajectory, base.trajectory, atol=1e-9)
    on = _solve(sc, np.where(distal, 0.1, 1.0))
    edit = lambda r: np.abs(np.asarray(r.trajectory)[:, :H] - sc.reference)   # noqa: E731
    share = lambda r: float(edit(r)[distal].sum() / edit(r)[distal | proximal].sum())   # noqa: E731
    assert base.max_violation <= tol and on.max_violation <= tol
    assert share(on) > share(base)
    assert edit(on)[proximal].max() < edit(base)[proximal].max()


# ------------------------------------------------------------------------------------------------ 3. SafePolicy gate
def _gate_policy(scale=0.1):
    from AG3S.trajopt.grasp_latch import GraspPhase
    from AG3S.trajopt.safe_policy import SafePolicy
    from AG3S.trajopt.types import ChunkLayout
    from tests.trajopt.test_safe_policy import _robot_model

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"cost": {"wrist_track_scale": scale}})
    p._latch = types.SimpleNamespace(phase=GraspPhase.LATCHED)
    p._last_grasp_record = {}
    p._last_wrist_track = {}
    model = _robot_model()
    p.constraint_robot_model = model
    p.layout = ChunkLayout.rby1(model.joint_names)
    p.grasp_links = {"left": ("ee_finger_l1", ()), "right": ("ee_finger_r1", ())}
    p.hold_links = {"left": "ee_left", "right": "ee_right"}
    p.ag3s = types.SimpleNamespace(manipulated=None)
    return p, model


def test_gate_scales_the_grasping_wrist_near_the_target_only():
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.trajopt.grasp_latch import GraspPhase

    p, model = _gate_policy()
    q = np.zeros(len(model.joint_names))
    palm = np.asarray(_link_pose_numeric(model, q, "ee_left")[:3, 3])
    assert p._wrist_track_gate(q, ["left"]) is None and p._last_wrist_track["reason"] == "no target"
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=palm + np.array([0.0, 0.0, -0.05]))
    s = p._wrist_track_gate(q, ["left"])
    names = [str(p.layout.joint_names[i]) for i in p.layout.q_indices]
    assert s is not None and s.shape == (len(names),)
    assert {n for n, v in zip(names, s) if v != 1.0} == {"left_arm_4", "left_arm_5", "left_arm_6"}
    assert np.allclose(s[s != 1.0], 0.1) and p._last_wrist_track["active"]
    s_r = p._wrist_track_gate(q, ["right"])                                        # right hand: far from the target
    assert s_r is None and p._last_wrist_track["reason"] == "palm outside the radius"
    p.ag3s.manipulated.centroid = palm + np.array([0.0, 0.0, -0.30])               # 30 cm: hurdle / approach
    assert p._wrist_track_gate(q, ["left"]) is None
    p.ag3s.manipulated.centroid = palm
    for phase in (GraspPhase.SEARCHING, GraspPhase.HELD, GraspPhase.PLACED):        # carry keeps uniform weights
        p._latch.phase = phase
        assert p._wrist_track_gate(q, ["left"]) is None
    p._latch.phase = GraspPhase.CLOSING
    assert p._wrist_track_gate(q, ["left"]) is not None
    off, _ = _gate_policy(scale=1.0)
    off.ag3s.manipulated = p.ag3s.manipulated
    assert off._wrist_track_gate(q, ["left"]) is None and off._last_wrist_track == {}


# ------------------------------------------------------------------------------------------------ 4. config / CLI
def test_config_and_cli():
    from AG3S.trajopt import serve_safe

    assert TrajOptConfig().cost.wrist_track_scale == 1.0
    for bad in ({"wrist_track_scale": 0.0}, {"wrist_track_scale": 1.5}, {"wrist_track_joints": (7,)},
                {"wrist_track_radius": 0.0}):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"cost": bad})
    args = serve_safe.build_parser().parse_args([*BASE_ARGV, "--wrist-track-scale", "0.1"])
    assert serve_safe.trajopt_config_from_args(args).cost.wrist_track_scale == pytest.approx(0.1)
    assert serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args(BASE_ARGV)).cost.wrist_track_scale == 1.0
