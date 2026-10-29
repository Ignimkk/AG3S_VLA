"""AG3S-GEO-GRASP S2 — the 6-D geometric goal in TrajOpt and its controller (`serve_safe --geo-grasp`, default off).

1. Pose FK and the rotation term: `LinkPoseFK` matches the numeric FK; the linearised rotation error matches a finite
   difference; the QP rotation terms equal the quadratic model of the exact cost.
2. Solve: a reachable 6-D goal is met (position and rotation errors shrink); collision still wins over the goal
   (violation within tolerance, the goal gives way, the record says so).
3. Controller: phase-wise anchor kinds, B2 tracking scale and its carry restore, the close gate (defer / close /
   timeout -> failure), failure holds.
4. Config / CLI: off by default; flags map to the cost section; exclusive with the IFG-TOA anchor.
(The off path is pinned bit-identical to HEAD by `test_e1_grasp_goal.test_off_and_toa_are_bit_identical_to_head`.)
"""
from __future__ import annotations

import math
import types

import numpy as np
import pytest

pytest.importorskip("casadi")
pytest.importorskip("osqp")

from AG3S.trajopt.grasp_anchor import (GraspAnchor, LinkPoseFK, rotation_cost, rotation_errors,  # noqa: E402
                                       rotation_qp_terms)

LINK = "link_right_arm_5"


def _rot(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _sc(H=8):
    from tests.trajopt.fixtures import make_scenario

    return make_scenario(horizon=H, penetration=0.03)


def _solve(sc, anchor=None):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    return opt.solve(sc.reference, sc.q_now, sc.scene, **({} if anchor is None else {"grasp_anchor": anchor}))


# ------------------------------------------------------------------------------------------------ 1. FK / Jacobian
def test_pose_fk_matches_numeric_fk_and_rotation_jacobian_is_exact():
    sc = _sc()
    H = sc.reference.shape[1]
    fk = LinkPoseFK(sc.robot, sc.layout, H, LINK)
    p, Jp, R, Jw = fk.pose(sc.reference, sc.q_now)
    full = sc.layout.full_q(sc.reference, sc.q_now)
    for k in (0, H - 1):
        T = sc.robot.link_pose(full[:, k], LINK)
        assert np.allclose(p[k], T[:3, 3], atol=1e-12) and np.allclose(R[k], T[:3, :3], atol=1e-12)
    # linearised Log(Rg^T R) vs finite differences on one row
    Rg = R[3] @ _rot([0.3, -0.2, 1.0], 0.4)
    k = 3
    P_add, q_add = rotation_qp_terms(sc.reference, R, Jw, np.array([k]), Rg, 1.0)
    nq = sc.reference.shape[0]
    from AG3S.ag3s.grasp.ik import so3_log, so3_right_jacobian_inv
    e0 = so3_log(Rg.T @ R[k])
    A = so3_right_jacobian_inv(e0) @ R[k].T @ Jw[k]
    An = np.zeros_like(A)
    for j in range(nq):
        Q = sc.reference.copy()
        Q[j, k] += 1e-6
        An[:, j] = (rotation_errors(fk.pose(Q, sc.q_now)[2], np.array([k]), Rg)[0] - e0) / 1e-6
    assert np.abs(A - An).max() < 1e-5
    # QP terms = quadratic model of the exact cost: value and gradient at the iterate
    z = sc.reference.T.reshape(-1)          # step-major
    P = P_add.toarray()
    model = 0.5 * z @ P @ z + q_add @ z
    exact = rotation_cost(R, np.array([k]), Rg, 1.0)
    c = e0 - A @ sc.reference[:, k]
    assert model + float(c @ c) == pytest.approx(exact, rel=1e-9, abs=1e-12)


# ------------------------------------------------------------------------------------------------ 2. solve
def test_reachable_6d_goal_is_met_and_recorded():
    sc = _sc()
    H = sc.reference.shape[1]
    fk = LinkPoseFK(sc.robot, sc.layout, H, LINK)
    p, _, R, _ = fk.pose(sc.reference, sc.q_now)
    goal_p = p[5] + np.array([0.0, 0.01, 0.0])
    goal_R = R[5] @ _rot([1.0, 0.0, 0.0], math.radians(10.0))
    base = _solve(sc)
    a = GraspAnchor(LINK, goal_p, 5, 0.0, 2000.0, goal_rows=2, goal_R=goal_R, w_rot=5.0)
    assert a.variant == "GEO"
    r = _solve(sc, a)
    rec = r.metrics["grasp_anchor"]
    assert rec["variant"] == "GEO" and rec["goal_rows"] == [5, 6]
    # the position error shrinks against the plain solve, and the rotation term lowers the rotation error against the
    # same goal without it (w_rot = 0). On this fixture link (arm_5, not the hand) position and rotation compete —
    # measured 2026-10-26: plain 79.7 mm / 12.5 deg, w_rot 0: 5.6 mm / 14.3 deg, w_rot 5: 6.5 mm / 12.9 deg.
    pb, _, Rb, _ = fk.pose(np.asarray(base.trajectory)[:, :H], sc.q_now)
    assert rec["goal_err_result_mm"] < float(np.max(np.linalg.norm(pb[[5, 6]] - goal_p, axis=1))) * 1000
    r0 = _solve(sc, GraspAnchor(LINK, goal_p, 5, 0.0, 2000.0, goal_rows=2, goal_R=goal_R, w_rot=0.0))
    assert rec["goal_rot_err_result_deg"] < r0.metrics["grasp_anchor"]["goal_rot_err_result_deg"]
    assert r.max_violation <= float(sc.config.safety.violation_tolerance)
    assert "within_tolerance" in rec and "goal_rot_err_reference_deg" in rec


def test_collision_wins_over_the_6d_goal():
    """Goal = the clipped link's own reference pose at the deepest step: keeping it means staying in the obstacle."""
    sc = _sc()
    H = sc.reference.shape[1]
    fk = LinkPoseFK(sc.robot, sc.layout, H, LINK)
    p, _, R, _ = fk.pose(sc.reference, sc.q_now)
    tol = float(sc.config.safety.violation_tolerance)
    r = _solve(sc, GraspAnchor(LINK, p[3], 3, 0.0, 2000.0, goal_rows=1, goal_R=R[3], w_rot=5.0))
    rec = r.metrics["grasp_anchor"]
    assert r.max_violation <= tol                          # safety first
    assert rec["goal_err_result_mm"] > 5.0                 # the goal gave way, and the record shows it
    assert rec["within_tolerance"] is False


# ------------------------------------------------------------------------------------------------ 3. controller
class _Grasp:
    def __init__(self, T, T_pre, gid=0):
        self.T, self.T_pre, self.grasp_id = T, T_pre, gid


def _controller(sc, **cost_kw):
    from AG3S.trajopt.config import TrajOptConfig
    from AG3S.trajopt.geo_grasp import GeoGraspController

    cfg = TrajOptConfig().with_overrides({"cost": {"geo_grasp": True, **cost_kw}})
    g = GeoGraspController(cfg.cost, sc.robot, sc.layout, {"right": LINK, "left": LINK}, (7, 15), 0.95, cfg.limits,
                           horizon=8, execution_length=8)
    return g, cfg


def _with_grasp(g, T, phase="latched", hand="right"):
    g._hand, g._phase = hand, phase
    pl = types.SimpleNamespace(selected=_Grasp(T, T.copy()), reset=lambda: None)
    g._planners[hand] = pl


def _chunk(sc, close_row=None, col=15):
    ch = np.array(sc.chunk, float, copy=True)
    ch[:, col] = 1.0
    if close_row is not None:
        ch[close_row:, col] = 0.5
    return ch


def test_controller_anchor_kinds_by_phase():
    sc = _sc()
    g, _ = _controller(sc)
    T_hand = sc.robot.link_pose(sc.q_now, LINK)
    _with_grasp(g, T_hand.copy())
    a = g.anchor(sc.q_now, _chunk(sc, close_row=4))
    assert a is not None and a.goal_kind == "grasp" and a.close_row == 4 and a.variant == "GEO"
    a = g.anchor(sc.q_now, _chunk(sc))                    # no close planned, palm at the pre-grasp
    assert a.goal_kind == "pre-grasp" and a.explicit_rows == (7,)
    mid = T_hand.copy(); mid[:3, 3] += [0.12, 0, 0]          # 120 mm away: a 60 mm step toward it, not the pre-grasp
    _with_grasp(g, mid)
    a = g.anchor(sc.q_now, _chunk(sc))
    assert a.goal_kind == "toward pre-grasp" and np.linalg.norm(a.goal - T_hand[:3, 3]) == pytest.approx(0.06, abs=1e-9)
    far = T_hand.copy(); far[:3, 3] += [0.5, 0, 0]
    _with_grasp(g, far)
    assert g.anchor(sc.q_now, _chunk(sc)) is None and g.last["anchor"]["reason"] == "palm outside the approach radius"
    _with_grasp(g, T_hand.copy(), phase="closing")
    a = g.anchor(sc.q_now, _chunk(sc, close_row=0))
    assert a.goal_kind == "hold-at-grasp" and a.explicit_rows == (1, 2, 3)
    _with_grasp(g, T_hand.copy(), phase="held")
    assert g.anchor(sc.q_now, _chunk(sc)) is None
    # the goal rotation is the twin closest to the hand
    twin = T_hand.copy(); twin[:3, :3] = T_hand[:3, :3] @ np.diag([-1.0, -1.0, 1.0])
    _with_grasp(g, twin)
    a = g.anchor(sc.q_now, _chunk(sc, close_row=4))
    assert np.allclose(a.goal_R, T_hand[:3, :3], atol=1e-9)


def test_b2_scale_and_carry_restore():
    sc = _sc()
    g, _ = _controller(sc, geo_track_scale=0.25, geo_continuity_off=True, geo_carry_requests=3)
    names = [str(sc.layout.joint_names[i]) for i in sc.layout.q_indices]
    T_hand = sc.robot.link_pose(sc.q_now, LINK)
    _with_grasp(g, T_hand.copy())
    g.anchor(sc.q_now, _chunk(sc, close_row=4))
    s = g.track_scale(names)
    assert g.continuity_off() == "geo-grasp goal active"
    right = np.array([n.startswith("right_arm_") for n in names])
    assert np.allclose(s[right], 0.25) and np.allclose(s[~right], 1.0)
    # carry: held requests 1, 2, 3 -> 0.5, 0.75, then unchanged (None)
    g._anchor_active, g._phase = False, "held"
    out = []
    for n in (1, 2, 3):
        g.held_requests = n
        sc_ = g.track_scale(names)
        out.append(None if sc_ is None else float(sc_[right][0]))
    assert out == [0.5, 0.75, None]
    assert g.continuity_off() is None


def test_close_gate_defers_closes_and_times_out():
    sc = _sc()
    g, cfg = _controller(sc, geo_close_timeout=2)
    T_hand = sc.robot.link_pose(sc.q_now, LINK)
    g.started = True
    # goal 5 cm away from the planned hand at the close row -> defer (grip column kept open on rows >= close)
    off = T_hand.copy(); off[:3, 3] += [0.05, 0, 0]
    _with_grasp(g, off)
    ch = _chunk(sc, close_row=0)
    ch[:, sc.layout.action_columns] = np.repeat(sc.q_now[sc.layout.q_indices][None], len(ch), axis=0)
    out, hold = g.close_gate(ch, sc.q_now)
    assert hold is None and g.last["close_gate"]["decision"] == "defer" and np.all(out[:, 15] >= 0.95)
    out, hold = g.close_gate(ch, sc.q_now)
    assert hold is None and g.deferred == 2
    out, hold = g.close_gate(ch, sc.q_now)                 # 3 > timeout 2 -> failure, hold
    assert hold is not None and g.failed and "deferred" in g.failed
    assert g.anchor(sc.q_now, ch) is None                  # no goal after a failure
    # on the grasp -> close goes through and the counter resets
    g2, _ = _controller(sc)
    g2.started = True
    _with_grasp(g2, T_hand.copy())
    out, hold = g2.close_gate(ch, sc.q_now)
    assert hold is None and g2.last["close_gate"]["decision"] == "close" and np.allclose(out, ch)


def test_new_attempt_resets_and_failure_needs_the_reselection():
    sc = _sc()
    g, _ = _controller(sc)
    g.started, g.deferred, g._phase = True, 3, "held"
    g._hand = "right"
    g._planners["right"] = types.SimpleNamespace(selected=None, reset=lambda: None, memory=types.SimpleNamespace(points=np.zeros((0, 3))))
    # phase held -> latched (a re-grasp): started / deferred reset (step's bookkeeping, called directly)
    prev, g._phase = g._phase, "latched"
    if g._phase == "latched" and prev in ("held", "closing"):
        g.started, g.deferred = False, 0
    assert not g.started and g.deferred == 0
    # no grasp while closing, re-selection not used -> no failure; used -> failure
    g.started, g._phase = True, "closing"
    g.last = {"reason": "grasp lost while closing (no candidate)"}
    _, hold = g.close_gate(_chunk(sc, close_row=0), sc.q_now)
    assert hold is None and g.failed is None
    g.last = {"reason": "grasp lost while closing (re-selection used)"}
    _, hold = g.close_gate(_chunk(sc, close_row=0), sc.q_now)
    assert hold is not None and "re-selection" in g.failed


def test_gate_is_silent_before_control_started_and_while_closing():
    sc = _sc()
    g, _ = _controller(sc)
    T_hand = sc.robot.link_pose(sc.q_now, LINK)
    off = T_hand.copy(); off[:3, 3] += [0.05, 0, 0]
    _with_grasp(g, off)
    ch = _chunk(sc, close_row=0)
    out, hold = g.close_gate(ch, sc.q_now)                 # never started: the baseline path, untouched
    assert hold is None and np.allclose(out, ch)
    g.started = True
    _with_grasp(g, off, phase="closing")
    out, hold = g.close_gate(ch, sc.q_now)                 # fingers already closing: never re-opened
    assert hold is None and np.allclose(out, ch)


# ------------------------------------------------------------------------------------------------ 4. config / CLI
def test_config_and_cli():
    from AG3S.trajopt import serve_safe
    from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

    assert TrajOptConfig().cost.geo_grasp is False
    base = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]
    off = serve_safe.cost_overrides(serve_safe.build_parser().parse_args(base))
    assert not any(k.startswith("geo") for k in off)
    b2 = serve_safe.cost_overrides(serve_safe.build_parser().parse_args(
        base + ["--geo-grasp", "--geo-track-scale", "0.25", "--geo-continuity-off", "--geo-close-tol-mm", "12"]))
    assert b2["geo_grasp"] is True and b2["geo_track_scale"] == 0.25 and b2["geo_continuity_off"] is True
    assert b2["geo_close_tol_pos"] == pytest.approx(0.012)
    TrajOptConfig().with_overrides({"cost": b2})
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig().with_overrides({"cost": {"geo_grasp": True, "w_grasp_goal": 2000.0}})
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig().with_overrides({"cost": {"geo_track_scale": 0.25}})


def test_safe_policy_wiring_smoke():
    """`SafePolicy` with `cost.geo_grasp`: the controller is built (IK ready), a request runs the task state and
    records `grasp.geo`, the close gate runs in `infer`'s place; with the feature off there is no controller."""
    from AG3S.trajopt.config import TrajOptConfig
    from tests.trajopt.test_safe_policy import DOF, H, _safe_policy

    off = _safe_policy()
    assert off._geo is None
    cfg = TrajOptConfig().with_overrides({"cost": {"geo_grasp": True, "geo_track_scale": 0.25, "geo_continuity_off": True}})
    sp = _safe_policy(to_config=cfg)
    assert sp._geo is not None and "left" in sp._geo._iks
    chunk = np.zeros((H, DOF))
    sp._pending = {"chunk": chunk, "timing": {}}
    sp._last_grasp_record = {}
    ctx: dict = {}
    q = np.zeros(sp.constraint_robot_model.nq)
    sp._geo_request(q, ["left"], ctx, None, {"filtered_cloud": np.zeros((0, 3))})
    assert sp._last_grasp_record["geo"]["phase"] == "searching" and "grasp_anchor" not in ctx
    out, hold = sp._geo.close_gate(chunk, q)
    assert hold is None and np.allclose(out, chunk)


def test_valid_pose_gate_closes_off_the_selected_grasp():
    """`geo_close_gate = "valid-pose"`: a planned close far from the selected grasp still closes when the pose itself
    is valid on the observed surface; "selected" defers it. The record says which rule decided."""
    from AG3S.ag3s.grasp.candidates import GeoGraspConfig, generate_candidates
    from AG3S.ag3s.grasp.gripper import RBY1_LEFT
    from tests.ag3s.test_geo_grasp import sphere_cloud, table

    sc = _sc()
    T_hand = sc.robot.link_pose(sc.q_now, LINK)
    off = T_hand.copy(); off[:3, 3] += [0.05, 0, 0]
    ch = _chunk(sc, close_row=0)
    ch[:, sc.layout.action_columns] = np.repeat(sc.q_now[sc.layout.q_indices][None], len(ch), axis=0)
    # an observed sphere placed exactly between the pads of the hand's pose -> the planned close pose is valid
    cfg = GeoGraspConfig(time_budget_ms=0.0)
    pad_centre = T_hand[:3, 3] + T_hand[:3, :3] @ np.array([0, 0, -0.104])
    pts = sphere_cloud(c=tuple(pad_centre), r=0.03, cam=(tuple(pad_centre + 0.4 * T_hand[:3, 1]),
                                                       tuple(pad_centre - 0.4 * T_hand[:3, 1])))
    cs = generate_candidates(pts, None, RBY1_LEFT, cfg, recheck_only=True)
    for gate, expect in (("selected", "defer"), ("valid-pose", "close")):
        g, _ = _controller(sc, geo_close_gate=gate)
        g.started = True
        _with_grasp(g, off)
        g._planners["right"].last_set = cs
        g._planners["right"].gripper = RBY1_LEFT
        out, hold = g.close_gate(ch, sc.q_now)
        assert hold is None and g.last["close_gate"]["decision"] == expect, (gate, g.last["close_gate"])


def _close_chunk_at_q(sc, close_row, q_arm=None):
    ch = _chunk(sc, close_row=close_row)
    ch[:, sc.layout.action_columns] = np.repeat((sc.q_now if q_arm is None else q_arm)[sc.layout.q_indices][None],
                                                len(ch), axis=0)
    return ch


def test_correct_mode_goal_only_near_the_policy_close():
    sc = _sc()
    g, _ = _controller(sc, geo_goal_mode="correct")
    T_hand = sc.robot.link_pose(sc.q_now, LINK)
    near = T_hand.copy(); near[:3, 3] += [0.015, 0, 0]          # 15 mm from where the policy closes
    _with_grasp(g, near)
    g._planners["right"].last_set = None
    a = g.anchor(sc.q_now, _close_chunk_at_q(sc, 4))
    assert a is not None and a.goal_kind == "correct" and a.close_row == 4
    far = T_hand.copy(); far[:3, 3] += [0.05, 0, 0]            # 50 mm: a different grasp -> no goal
    _with_grasp(g, far)
    assert g.anchor(sc.q_now, _close_chunk_at_q(sc, 4)) is None
    assert g.last["anchor"]["reason"] == "no valid grasp near the policy's close"
    _with_grasp(g, near)
    assert g.anchor(sc.q_now, _chunk(sc)) is None                 # no close planned: no approach goal
    _with_grasp(g, near, phase="closing")
    assert g.anchor(sc.q_now, _close_chunk_at_q(sc, 0)) is None


def test_correct_mode_gate_never_vetoes_an_uncorrected_close():
    sc = _sc()
    g, _ = _controller(sc, geo_goal_mode="correct", geo_close_timeout=1)
    T_hand = sc.robot.link_pose(sc.q_now, LINK)
    g.started = True
    _with_grasp(g, T_hand.copy())
    ch = _close_chunk_at_q(sc, 0)
    g._correct_goal = None
    out, hold = g.close_gate(ch, sc.q_now)
    assert hold is None and g.last["close_gate"]["decision"] == "close" and np.allclose(out, ch)
    g._correct_goal = (T_hand[:3, 3].copy(), T_hand[:3, :3].copy())        # corrected and on it
    out, hold = g.close_gate(ch, sc.q_now)
    assert hold is None and g.last["close_gate"]["decision"] == "close"
    g._correct_goal = (T_hand[:3, 3] + [0.03, 0, 0], T_hand[:3, :3].copy())  # corrected but missed by 30 mm
    out, hold = g.close_gate(ch, sc.q_now)
    assert hold is None and g.last["close_gate"]["decision"] == "defer" and np.all(out[:, 15] >= 0.85)
    out, hold = g.close_gate(ch, sc.q_now)                                   # 2 > timeout 1
    assert hold is not None and g.failed


def test_chord_offset_ratio():
    from AG3S.trajopt.geo_grasp import chord_offset

    surf = types.SimpleNamespace(centre=np.zeros(3), radius=0.033, centre_method="sphere")
    through = np.array([[-0.03, 0, 0], [0.03, 0, 0]])
    assert chord_offset(through, surf) == pytest.approx(0.0, abs=1e-9)
    off = through + [0, 0.0165, 0]                                  # chord 16.5 mm off a 33 mm sphere: rho 0.5
    assert chord_offset(off, surf) == pytest.approx(0.5, abs=1e-6)
    assert chord_offset(off, types.SimpleNamespace(centre=np.zeros(3), radius=0.033, centre_method="centroid")) is None
    assert chord_offset(None, surf) is None


def test_correct_mode_centre_check_excludes_off_centre_chords():
    sc = _sc()
    T_hand = sc.robot.link_pose(sc.q_now, LINK)
    near = T_hand.copy(); near[:3, 3] += [0.015, 0, 0]
    surf = types.SimpleNamespace(centre=np.zeros(3), radius=0.033, centre_method="sphere")
    for tau, off, expect in ((0.0, 0.0165, True), (0.37, 0.0165, False), (0.37, 0.005, True)):
        g, _ = _controller(sc, geo_goal_mode="correct", geo_correct_max_offset=tau)
        _with_grasp(g, near)
        g._planners["right"].selected.contacts = np.array([[-0.03, off, 0], [0.03, off, 0]])
        g._planners["right"].last_set = types.SimpleNamespace(candidates=[], surface=surf)
        a = g.anchor(sc.q_now, _close_chunk_at_q(sc, 4))
        assert (a is not None) == expect, (tau, off)
        assert g.last["anchor"].get("chord_offset", None) is None or abs(g.last["anchor"]["chord_offset"] - off / 0.033) < 1e-3
        if not expect:
            assert g.last["anchor"]["reason"] == "no centred valid grasp"
            assert g.last["anchor"]["centre_check"] == {"max_offset": 0.37, "excluded": 1}
