"""T72 — collision-aware grasp set as a reach constraint in TrajOpt (`cost.grasp_set`, `serve_safe --grasp-set-to`,
default off). Task: `AG3S/ag3s/docs/handoff/T72.task.md` §2 (fixed before results).

What this file pins:
1. Formulas (task §2b-c): Q, J (three parts), eligibility, "the same grasp" and the finger swap, the hysteresis.
2. The TrajOpt row check (task §2a-2): `field_check` calls the solver's own functions (`config_states`,
   `_clearances_from`, `_esdf_clearance`) on the solver's scene, gives the same numbers, restores the linearizer's
   record attributes; `generate_candidates` drops a candidate the check rejects and counts it.
3. The controller: window (latched with 0 < c < H / closing, the A6 window), frozen intent while closing, the reach
   anchor (box ±5 mm on c..c+2, slack at w_slack, rotation nearest the hand, closing rows 1..3 within 20 mm), the
   records, `reach_err_mm`.
4. TrajOpt: the box slack price (1 x w_slack for the reach box, E1-B unchanged), the multi-seed selection counts the
   reach-box excess as a violation (GPU-seed option b), absent for every other anchor.
5. SafePolicy: built only with the flag, HSR6 ④ does not withdraw it, the record keys.
6. Config / CLI: off by default with no keys in the dump, the T72 argv, start-up refusal with A4 / E1-B / G1.
"""
from __future__ import annotations

import dataclasses
import json
import math
import shlex
import types

import numpy as np
import pytest

from AG3S.trajopt import grasp_set as GS
from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

COST = TrajOptConfig.from_dict({"cost": {"grasp_set": True}}).cost
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _T(p=(0.0, 0.0, 0.0), R=None):
    T = np.eye(4)
    T[:3, 3] = p
    if R is not None:
        T[:3, :3] = R
    return T


# ------------------------------------------------------------------------------------------------ 1. formulas
def test_quality_formula():
    c = math.cos(math.atan(0.8))
    q = GS.quality(1.0, 0.0, (8, 8), mu=0.8, min_support=4)
    assert q["q_fric"] == 1.0 and q["q_centre"] == 1.0 and q["q_support"] == 1.0 and q["Q"] == 1.0
    q = GS.quality(c, None, (4, 12), mu=0.8, min_support=4)            # cone edge, no sphere fit, one weak pad
    assert q["q_fric"] == pytest.approx(0.0, abs=1e-12) and q["q_centre"] == 0.5 and q["q_support"] == 0.5
    assert q["Q"] == pytest.approx(1.0 / 3.0)
    mid = (1.0 + c) / 2.0
    q = GS.quality(mid, 0.3, (2, 6), mu=0.8, min_support=4)
    assert q["q_fric"] == pytest.approx(0.5) and q["q_centre"] == pytest.approx(0.7)
    assert q["q_support"] == pytest.approx(0.25) and q["Q"] == pytest.approx((0.5 + 0.7 + 0.25) / 3.0)
    assert GS.quality(c - 0.1, 2.0, (0, 9), mu=0.8, min_support=4)["Q"] == 0.0     # every part clipped at 0


def test_objective_formula_and_its_three_parts():
    dq = np.zeros(14)
    dq[3] = 0.1                                                         # 0.1 rad on one joint: J_track = 1
    W = np.ones(14)
    j = GS.objective(dq, W, 0.020, math.radians(15.0), 0.5, COST)       # 20 mm and 15 deg: J_intent = 2
    assert j["J_track"] == pytest.approx(1.0) and j["J_intent"] == pytest.approx(2.0)
    assert j["J_quality"] == pytest.approx(0.5) and j["J"] == pytest.approx(3.5)
    W[3] = 0.1                                                          # the wrist 0.1 x tracking weight
    assert GS.objective(dq, W, 0.0, 0.0, 1.0, COST)["J"] == pytest.approx(0.1)
    assert COST.grasp_set_sigma_q == 0.1 and COST.grasp_set_sigma_p == 0.020 and COST.grasp_set_sigma_theta_deg == 15.0
    assert COST.grasp_set_lambda_intent == 1.0 and COST.grasp_set_lambda_quality == 1.0


@pytest.mark.parametrize("dp,dth,ok", [(0.0599, 29.9, True), (0.060, 30.0, True), (0.0601, 10.0, False),
                                       (0.01, 30.1, False)])
def test_eligibility_60_mm_30_deg(dp, dth, ok):
    assert GS.eligible(dp, math.radians(dth), COST) is ok


def test_same_grasp_and_finger_swap():
    from AG3S.ag3s.grasp.candidates import grasp_rot_angle, nearest_symmetric

    a = _T((0.1, 0.2, 0.3), _rz(0.2))
    swapped = a.copy()
    swapped[:3, :3] = a[:3, :3] @ np.diag([-1.0, -1.0, 1.0])            # 180 deg about the approach axis
    assert GS.same_grasp(a, swapped, COST)                              # the closing axis sign does not matter
    assert grasp_rot_angle(a[:3, :3], swapped[:3, :3]) == pytest.approx(0.0, abs=1e-9)
    assert not GS.same_grasp(a, _T((0.1, 0.2, 0.3061), _rz(0.2)), COST)  # 6.1 mm apart
    assert GS.same_grasp(a, _T((0.1, 0.2, 0.3059), _rz(0.2)), COST)
    assert GS.same_grasp(a, _T((0.1, 0.2, 0.3), _rz(0.2 + math.radians(14.9))), COST)
    assert not GS.same_grasp(a, _T((0.1, 0.2, 0.3), _rz(0.2 + math.radians(15.1))), COST)
    # the rotation goal: the twin nearest the hand now
    hand = swapped[:3, :3] @ _rz(0.05)
    np.testing.assert_allclose(nearest_symmetric(a[:3, :3], hand), swapped[:3, :3])


def test_hysteresis_keeps_the_previous_grasp_unless_0_8_x():
    prev = _T((0.0, 0.0, 0.0))
    Ts = [_T((0.0, 0.0, 0.002)), _T((0.05, 0.0, 0.0)), _T((0.0, 0.05, 0.0))]
    ok = [True, True, True]
    i, rec = GS.choose([1.0, 0.85, 2.0], Ts, ok, prev, COST)
    assert i == 0 and rec["previous"] == "kept" and rec["changed"] is False and rec["argmin_id"] == 1
    i, rec = GS.choose([1.0, 0.79, 2.0], Ts, ok, prev, COST)
    assert i == 1 and rec["previous"] == "switched" and rec["changed"] is True and rec["kept_id"] == 0
    i, rec = GS.choose([1.0, 0.79, 2.0], Ts, [False, True, True], prev, COST)    # the previous one is not eligible
    assert i == 1 and rec["previous"] == "lost"
    i, rec = GS.choose([1.0, 0.79, 2.0], Ts, ok, None, COST)
    assert i == 1 and rec["previous"] == "none" and rec["changed"] is False
    assert GS.choose([1.0], Ts[:1], [False], None, COST)[0] is None
    i, _ = GS.choose([0.5, 0.5], Ts[1:], [True, True], None, COST)                 # ties: the earlier index
    assert i == 0


# ------------------------------------------------------------------------------------------------ 2. TrajOpt rows
class _Field:
    """An analytic field: a ball obstacle; `grid.voxel_size` as the curobo field has."""

    has_labels = False

    def __init__(self, centre, radius):
        self.c, self.r = np.asarray(centre, np.float64), float(radius)
        self.grid = types.SimpleNamespace(voxel_size=0.005)

    def distance(self, pts):
        return np.linalg.norm(np.asarray(pts, np.float64).reshape(-1, 3) - self.c, axis=1) - self.r


def _field_rig(margin=0.010):
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    lin = sc.linearizer
    q = sc.q_now.copy()
    centres = lin.config_states(q.reshape(-1, 1), None, with_jac=False)[0][0]
    field = _Field(centres[0] + np.array([0.0, 0.0, float(lin.robot_radii[0]) + 0.05]), 0.02)
    scene = dataclasses.replace(sc.scene, esdf=field, esdf_margin=0.0, obstacle_margin=float(margin))
    return sc, lin, q, scene


def test_field_check_reads_the_solvers_rows_on_the_solvers_scene(monkeypatch):
    sc, lin, q, scene = _field_rig()
    q2 = q.copy()
    q2[sc.layout.q_indices[0]] += 0.3
    calls = {"esdf": [], "planes": [], "states": 0}
    real_e, real_p, real_s = lin._esdf_clearance, lin._clearances_from, lin.config_states

    def spy_e(c, s):
        calls["esdf"].append(s)
        return real_e(c, s)

    def spy_p(c, s):
        calls["planes"].append(s)
        return real_p(c, s)

    def spy_s(*a, **k):
        calls["states"] += 1
        return real_s(*a, **k)

    monkeypatch.setattr(lin, "_esdf_clearance", spy_e)
    monkeypatch.setattr(lin, "_clearances_from", spy_p)
    monkeypatch.setattr(lin, "config_states", spy_s)
    rows = np.ones(lin.n_spheres, bool)
    lin.__dict__["_last_obstacle_class"] = "sentinel"
    ok, info = GS.field_check(lin, scene, [q, q2], None, rows)
    assert calls["states"] == 1 and calls["esdf"] == [scene] and calls["planes"] == [scene]
    assert lin.__dict__["_last_obstacle_class"] == "sentinel"            # the solve's record attributes untouched
    # the same numbers as the solver's own call on the same configurations
    monkeypatch.undo()
    centres = lin.config_states(np.stack([q, q2], axis=1), None, with_jac=False)[0]
    flat = lin.flat_clearances(centres, scene)
    want = [round(float(flat[k].min()) * 1000.0, 3) for k in range(2)]
    assert info["min_clearance_mm"] == want and ok is (min(want) >= 0.0)
    assert info["n_rows"] == lin.n_spheres and info["worst"]["clearance_mm"] == min(want)


def test_field_check_mask_and_margin():
    sc, lin, q, scene = _field_rig(margin=0.010)
    n = scene.candidate_pos.shape[0]
    scene = dataclasses.replace(scene, candidate_active=np.zeros(n, bool))   # the field rows only
    rows = np.zeros(lin.n_spheres, bool)
    rows[0] = True
    _, info = GS.field_check(lin, scene, [q], None, rows)
    centres = lin.config_states(q.reshape(-1, 1), None, with_jac=False)[0]
    d = float(scene.esdf.distance(centres[0, :1])[0]) - float(lin.robot_radii[0])
    assert d > 0.010 and info["worst"]["row"] == 0 and info["worst"]["block"] == "field" and info["n_rows"] == 1
    # an obstacle row clears max(esdf_margin, obstacle_margin) = 10 mm, as in the solver
    assert info["min_clearance_mm"][0] == pytest.approx((d - 0.010) * 1000.0, abs=1e-3)
    _, info0 = GS.field_check(lin, dataclasses.replace(scene, obstacle_margin=0.0), [q], None, rows)
    assert info0["min_clearance_mm"][0] == pytest.approx(d * 1000.0, abs=1e-3)
    # a ball onto the sphere: negative -> rejected
    hit = dataclasses.replace(scene, esdf=_Field(centres[0, 0], 0.001))
    ok, infoh = GS.field_check(lin, hit, [q], None, rows)
    assert ok is False and infoh["min_clearance_mm"][0] < 0.0


def test_generator_drops_and_counts_candidates_the_config_check_rejects():
    from AG3S.ag3s.grasp.candidates import GeoGraspConfig, generate_candidates
    from AG3S.ag3s.grasp.gripper import spec_for

    rng = np.random.default_rng(0)
    v = rng.normal(size=(1500, 3))
    pts = np.array([0.5, 0.0, 0.8]) + 0.035 * v / np.linalg.norm(v, axis=1, keepdims=True)
    pts = pts[pts[:, 0] < 0.51]                                          # the camera side only
    ik = lambda T, seed: (np.zeros(7), {"converged": True})             # noqa: E731
    cfg = GeoGraspConfig(time_budget_ms=0.0, ik_accept=8)
    base = generate_candidates(pts, None, spec_for("left"), cfg, ik_fn=ik)
    assert base.candidates and "n_config_rejected" not in base.stats
    seen = []

    def check(g, q, qp):
        seen.append(g.pair)
        return len(seen) % 2 == 0, {"n": len(seen)}

    out = generate_candidates(pts, None, spec_for("left"), cfg, ik_fn=ik, config_check_fn=check)
    assert out.stats["n_config_rejected"] == sum(1 for k in range(1, len(seen) + 1) if k % 2 == 1)
    assert all(g.checks["config"]["n"] % 2 == 0 for g in out.candidates)
    none = generate_candidates(pts, None, spec_for("left"), cfg, ik_fn=ik, config_check_fn=lambda g, q, qp: (False, {}))
    assert none.candidates == [] and none.stats["reason"] == "every IK solution violates the configuration check"


# ------------------------------------------------------------------------------------------------ 3. controller
@pytest.fixture(scope="module")
def ctrl_parts():
    pytest.importorskip("casadi")
    from AG3S.trajopt import wire
    from AG3S.trajopt.types import ChunkLayout
    from tests.trajopt.test_safe_policy import _robot_model

    model = _robot_model()
    layout = ChunkLayout.rby1(model.joint_names)
    cols = wire.gripper_columns(layout.nq_opt // 2)
    cfg = TrajOptConfig.from_dict({"cost": {"grasp_set": True}})
    ctrl = GS.GraspSetController(cfg.cost, model, layout, {"left": "ee_left", "right": "ee_right"}, cols, 0.85,
                                 cfg.limits, horizon=16)
    return model, layout, cols, ctrl


def _chunk(layout, cols, q, close_row=None, n=50):
    ch = np.zeros((n, layout.action_dim))
    ch[:, layout.action_columns] = np.asarray(q)[layout.q_indices][None]
    ch[:, list(cols)] = 1.0
    if close_row is not None:
        ch[close_row:, cols[0]] = 0.0
    return ch


def _manip(centre):
    rng = np.random.default_rng(1)
    v = rng.normal(size=(300, 3))
    return types.SimpleNamespace(id=0, centroid=np.asarray(centre), state="visible",
                                 geometry=types.SimpleNamespace(points=np.asarray(centre) + 0.035 * v / np.linalg.norm(
                                     v, axis=1, keepdims=True)))


def _fake_generator(monkeypatch, offsets, calls=None):
    """`generate_candidates` replaced: candidates at the VLA pose + `offsets` (m, hold-link position), IK = the seed,
    the config check called as the real generator calls it."""
    from AG3S.ag3s.grasp import candidates as C

    def gen(pts, scene, gripper, cfg, *, vla_pose, ik_fn, ik_seed, config_check_fn, **k):
        out = []
        for i, off in enumerate(offsets):
            T = np.asarray(vla_pose, np.float64).copy()
            T[:3, 3] += np.asarray(off, np.float64)
            g = C.GraspCandidate(index=i, contacts=np.stack([T[:3, 3] - [0.03, 0, 0], T[:3, 3] + [0.03, 0, 0]]),
                                 normals=np.zeros((2, 3)), width=0.06, open_width=0.07, depth=0.1, roll=0.0, T=T,
                                 T_pre=T.copy(), alignment=1.0, support=(8, 8), pair=i)
            ok, info = config_check_fn(g, np.asarray(ik_seed), np.asarray(ik_seed))
            g.checks["config"] = info
            if not ok:
                continue
            g.q_grasp, g.q_pre = np.asarray(ik_seed).copy(), np.asarray(ik_seed).copy()
            g.checks["valid"] = True
            out.append(g)
        if calls is not None:
            calls.append(vla_pose.copy())
        return C.CandidateSet(out, None, {"rejected": {"scene_collision": 2, "target_collision": 1,
                                                       "approach_collision": 0},
                                          "n_ik_failed": 1, "n_config_rejected": len(offsets) - len(out)})

    monkeypatch.setattr(C, "generate_candidates", gen)


def _field_ok(monkeypatch, fail=()):
    calls = []

    def fc(lin, scene, q_full, params, rows):
        calls.append((lin, scene, len(q_full), params))
        bad = len(calls) - 1 in fail
        return (not bad), {"ok": not bad, "min_clearance_mm": [(-1.0 if bad else 3.0), 4.0], "n_rows": int(rows.sum())}

    monkeypatch.setattr(GS, "field_check", fc)
    return calls


def _req(ctrl, model, layout, cols, phase, chunk, q, window=True, scene="SCENE", lin=None, manip=None):
    from AG3S.trajopt.safe_policy import hold_site_position

    site = hold_site_position(model, q, "ee_left")
    lin = lin or types.SimpleNamespace(robot_model=model, param_names=())
    return ctrl.request(phase=phase, hand="left", grasp_link="ee_finger_l1",
                        manip=manip if manip is not None else _manip(site + [0.0, 0.0, -0.02]),
                        cloud_points=None, q_now=q, chunk=chunk, window=("ee_left", site) if window else None,
                        window_rec={"site_to_target_mm": 20.0} if window else {"reason": "palm outside the gate"},
                        scene=scene, linearizer=lin)


def test_reach_anchor_on_a_latched_close(monkeypatch, ctrl_parts):
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.trajopt.grasp_anchor import GraspAnchor

    model, layout, cols, ctrl = ctrl_parts
    ctrl.reset()
    _fake_generator(monkeypatch, [(0.004, 0.0, 0.0), (0.03, 0.0, 0.0), (0.07, 0.0, 0.0)])
    fc = _field_ok(monkeypatch)
    q = np.zeros(len(model.joint_names))
    a = _req(ctrl, model, layout, cols, "latched", _chunk(layout, cols, q, 5), q, scene="THE-SCENE")
    rec = ctrl.last
    assert isinstance(a, GraspAnchor) and rec["active"] is True and rec["kind"] == "grasp-set"
    assert a.reach_constraint is True and a.goal_box_slack_scale == 1.0 and a.goal_box == 0.005
    assert a.w_goal == 0.0 and a.w_tcp == 0.0 and a.w_rot == 5.0 and a.close_row == 5
    assert rec["rows"] == [5, 6, 7] and list(np.flatnonzero(a.goal_mask(16))) == [5, 6, 7]
    assert fc and all(c[1] == "THE-SCENE" and c[2] == 2 for c in fc)          # grasp + pre-grasp, the TO's scene
    assert rec["n_candidates"] == 3 and rec["n_eligible"] == 2 and rec["n_rejected"]["eligibility"] == 1
    assert rec["n_rejected"]["point_check"] == 3 and rec["n_rejected"]["ik"] == 1 and rec["n_rejected"]["field_check"] == 0
    cands = rec["candidates"]
    assert [c["eligible"] for c in cands] == [True, True, False]
    assert all({"Q", "J", "J_track", "J_intent", "J_quality", "dp_mm", "dtheta_deg", "dq_norm"} <= c.keys() for c in cands)
    assert cands[0]["J_intent"] == pytest.approx((4.0 / 20.0) ** 2, abs=1e-3)     # 4 mm off the intent
    sel = rec["selected"]
    assert sel["id"] == 0 and sel["previous"] == "none" and sel["dp_mm"] == pytest.approx(4.0, abs=0.01)
    T_hand = np.asarray(_link_pose_numeric(model, q, "ee_left"))
    from AG3S.ag3s.grasp.candidates import rot_angle
    assert rot_angle(a.goal_R, T_hand[:3, :3]) == pytest.approx(0.0, abs=1e-6)    # nearest twin to the hand
    np.testing.assert_allclose(a.goal, T_hand[:3, 3] + [0.004, 0.0, 0.0], atol=1e-9)
    assert rec["intent"]["close_row"] == 5 and rec["intent"]["frozen"] is False and "gen_ms" in rec
    # after the solve: the returned chunk's hold link at the first reach row vs g
    ctrl.reach_record(_chunk(layout, cols, q, 5), 42.0)
    assert rec["reach_err_mm"] == pytest.approx(4.0, abs=0.01) and rec["reach_err_rows_mm"] == [4.0, 4.0, 4.0]
    assert rec["reach_box_excess_max_mm"] == pytest.approx(0.0) and rec["solve_ms"] == 42.0


@pytest.mark.parametrize("close_row,reason", [(None, "no close planned in this horizon"),
                                              (16, "no close planned in this horizon"),
                                              (20, "no close planned in this horizon"),
                                              (0, "chunk starts closed")])
def test_no_reach_without_a_close_inside_the_horizon(monkeypatch, ctrl_parts, close_row, reason):
    model, layout, cols, ctrl = ctrl_parts
    ctrl.reset()
    _fake_generator(monkeypatch, [(0.0, 0.0, 0.0)])
    _field_ok(monkeypatch)
    q = np.zeros(len(model.joint_names))
    assert _req(ctrl, model, layout, cols, "latched", _chunk(layout, cols, q, close_row), q) is None
    assert ctrl.last["active"] is False and ctrl.last["reason"] == reason


def test_window_and_phase_gate(monkeypatch, ctrl_parts):
    model, layout, cols, ctrl = ctrl_parts
    ctrl.reset()
    _fake_generator(monkeypatch, [(0.0, 0.0, 0.0)])
    _field_ok(monkeypatch)
    q = np.zeros(len(model.joint_names))
    ch = _chunk(layout, cols, q, 5)
    assert _req(ctrl, model, layout, cols, "latched", ch, q, window=False) is None
    assert ctrl.last["reason"] == "grasp-moment window closed: palm outside the gate"
    for phase in ("searching", "held", "placed"):
        assert _req(ctrl, model, layout, cols, phase, ch, q) is None and ctrl.last["reason"] == f"phase {phase}"
    assert _req(ctrl, model, layout, cols, "latched", ch, q, scene=None) is None
    assert ctrl.last["reason"] == "no TrajOpt scene this request"


def test_closing_freezes_the_intent_and_holds_rows_1_to_3_within_20_mm(monkeypatch, ctrl_parts):
    model, layout, cols, ctrl = ctrl_parts
    ctrl.reset()
    seen = []
    _fake_generator(monkeypatch, [(0.004, 0.0, 0.0)], calls=seen)
    _field_ok(monkeypatch)
    q = np.zeros(len(model.joint_names))
    assert _req(ctrl, model, layout, cols, "latched", _chunk(layout, cols, q, 5), q) is not None
    q_moved = q.copy()
    q_moved[layout.q_indices[0]] += 0.2                                  # the policy now plans elsewhere
    a = _req(ctrl, model, layout, cols, "closing", _chunk(layout, cols, q_moved, 0), q)
    assert a is not None and ctrl.last["kind"] == "grasp-set hold" and ctrl.last["rows"] == [1, 2, 3]
    assert a.explicit_rows == (1, 2, 3) and ctrl.last["intent"]["frozen"] is True
    np.testing.assert_allclose(seen[1], seen[0])                         # T_v frozen at the close start
    assert ctrl.last["intent"]["source"] == "latched close row"
    # the hand far from the grasp while closing: no reach (never drag the hand mid-close)
    _fake_generator(monkeypatch, [(0.025, 0.0, 0.0)])
    ctrl._prev_T = None
    assert _req(ctrl, model, layout, cols, "closing", _chunk(layout, cols, q, 0), q) is None
    assert ctrl.last["reason"] == "closing away from the grasp" and ctrl.last["palm_to_grasp_mm"] == pytest.approx(25.0, abs=0.1)
    # a new attempt (latched after closing) drops the frozen intent
    _req(ctrl, model, layout, cols, "latched", _chunk(layout, cols, q, 5), q)
    assert ctrl.last["intent"]["frozen"] is False


def test_closing_without_a_latched_intent_freezes_its_own(monkeypatch, ctrl_parts):
    model, layout, cols, ctrl = ctrl_parts
    ctrl.reset()
    _fake_generator(monkeypatch, [(0.004, 0.0, 0.0)])
    _field_ok(monkeypatch)
    q = np.zeros(len(model.joint_names))
    assert _req(ctrl, model, layout, cols, "closing", _chunk(layout, cols, q, 0), q) is not None
    assert ctrl.last["intent"]["source"] == "closing start (no latched intent)" and ctrl.last["intent"]["close_row"] == 0


def test_field_rejected_candidates_are_recorded_and_none_left_is_inactive(monkeypatch, ctrl_parts):
    model, layout, cols, ctrl = ctrl_parts
    ctrl.reset()
    _fake_generator(monkeypatch, [(0.004, 0.0, 0.0), (0.01, 0.0, 0.0)])
    _field_ok(monkeypatch, fail=(0, 1))
    q = np.zeros(len(model.joint_names))
    assert _req(ctrl, model, layout, cols, "latched", _chunk(layout, cols, q, 5), q) is None
    rec = ctrl.last
    assert rec["reason"] == "no candidate" and rec["n_rejected"]["field_check"] == 2 and len(rec["field_rejected"]) == 2


def test_hysteresis_across_requests(monkeypatch, ctrl_parts):
    """J_quality is the same for every fake candidate (Q = (1 + 0.5 + 1) / 3); J_intent = (dp / 20 mm)^2."""
    model, layout, cols, ctrl = ctrl_parts
    ctrl.reset()
    _field_ok(monkeypatch)
    q = np.zeros(len(model.joint_names))
    ch = _chunk(layout, cols, q, 5)
    jq = 1.0 - (1.0 + 0.5 + 1.0) / 3.0
    _fake_generator(monkeypatch, [(0.010, 0.0, 0.0), (0.0, 0.012, 0.0)])
    _req(ctrl, model, layout, cols, "latched", ch, q)
    assert ctrl.last["selected"]["id"] == 0 and ctrl.last["candidates"][0]["J"] == pytest.approx(0.25 + jq, abs=1e-3)
    _fake_generator(monkeypatch, [(0.011, 0.0, 0.0), (0.0, 0.010, 0.0)])     # 0.25 + jq vs 0.8 x (0.3025 + jq)
    _req(ctrl, model, layout, cols, "latched", ch, q)
    s = ctrl.last["selected"]
    assert s["id"] == 0 and s["previous"] == "kept" and s["argmin_id"] == 1
    _fake_generator(monkeypatch, [(0.012, 0.0, 0.0), (0.0, 0.002, 0.0)])     # 0.01 + jq < 0.8 x (0.36 + jq)
    _req(ctrl, model, layout, cols, "latched", ch, q)
    s = ctrl.last["selected"]
    assert s["id"] == 1 and s["previous"] == "switched" and s["changed"] is True


# ------------------------------------------------------------------------------------------------ 4. TrajOpt
def test_box_slack_price_reach_vs_e1b():
    from AG3S.trajopt.grasp_anchor import GOAL_BOX_SLACK_SCALE, GraspAnchor, anchor_record
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    g = np.array([0.3, -0.2, 0.9])
    e1b = GraspAnchor("ee_right", g, 3, 0.0, 0.0, goal_box=0.005)
    assert e1b.goal_box_slack_scale is None and e1b.reach_constraint is False    # the module constant at solve time
    st = opt._anchor_setup(e1b, sc.reference, sc.q_now)
    assert st["box"]["weight"] == float(sc.config.cost.w_slack) * GOAL_BOX_SLACK_SCALE
    reach = dataclasses.replace(e1b, goal_box_slack_scale=1.0, reach_constraint=True)
    assert opt._anchor_setup(reach, sc.reference, sc.q_now)["box"]["weight"] == float(sc.config.cost.w_slack)
    p = np.zeros((8, 3))
    assert "reach_constraint" not in anchor_record(e1b, p, p)
    r = anchor_record(reach, p, p)
    assert r["reach_constraint"] is True and r["goal_box_slack_scale"] == 1.0


def test_multi_seed_selection_counts_the_reach_excess_only_for_the_reach_box():
    """`_run_seeds` on E2's stuck standard result (18.8 mm), with an E1-B box anchor and the same box as T72's reach
    constraint: only the latter adds `reach_excess_mm` / `selection_violation_mm` and selects on max(collision, reach)."""
    from AG3S.trajopt.grasp_anchor import GraspAnchor, LinkPointFK
    from AG3S.trajopt.multi_seed import select_moving_first
    from tests.trajopt.test_e2_multi_seed import _opt, _stuck

    sc, scene = _stuck()
    std = _opt(sc).solve(sc.reference, sc.q_now, scene)
    assert std.max_violation > 0.015
    p = LinkPointFK(sc.robot, sc.layout, 8, "ee_right")(sc.reference, sc.q_now)[0]
    e1b = GraspAnchor("ee_right", p[5] + np.array([0.0, 0.0, 0.02]), 5, 0.0, 0.0, goal_box=0.005)
    reach = dataclasses.replace(e1b, goal_box_slack_scale=1.0, reach_constraint=True)

    def seeds(anchor):
        opt = _opt(sc, 16)
        res = opt._run_seeds(std, 0, sc.reference, sc.q_now, scene, {"grasp_anchor": anchor},
                             {"triggered": False, "standard_violation_mm": round(std.max_violation * 1e3, 3)})
        return res.metrics["multi_seed"]

    plain = seeds(e1b)
    assert plain["triggered"] and "reach_selection" not in plain
    assert all("selection_violation_mm" not in c and "reach_excess_mm" not in c for c in plain["candidates"])
    rec = seeds(reach)
    assert rec["triggered"] and "reach_selection" in rec
    tol = float(sc.config.safety.violation_tolerance) * 1000.0
    cands = rec["candidates"]
    for c in cands:
        assert c["selection_violation_mm"] == pytest.approx(max(c["max_violation_mm"], c["reach_excess_mm"]), abs=2e-3)
    assert any(c["reach_excess_mm"] > 0.0 for c in cands)                # the box is 20 mm off the reference: misses
    resolved = [i for i, c in enumerate(cands) if c["eligible"] and c["kind"] != "stay"
                and c["selection_violation_mm"] <= tol]
    assert rec["n_resolved"] == sum(1 for c in cands if c["eligible"] and c["selection_violation_mm"] <= tol)
    if resolved:
        assert rec["chosen"] in resolved
        want = select_moving_first([c["kind"] for c in cands], [(0,) if c["eligible"] else None for c in cands],
                                   [c["selection_violation_mm"] for c in cands], tol,
                                   [c["goal_err_mm"] for c in cands], [c["reference_deviation"] for c in cands])
        assert rec["chosen"] == want
    else:
        assert cands[rec["chosen"]]["selection_violation_mm"] == min(
            c["selection_violation_mm"] for c in cands if c["eligible"])


def test_reach_probe_is_none_for_every_other_anchor():
    from AG3S.trajopt.grasp_anchor import GraspAnchor
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    goal = {"rows": np.array([3]), "goal": np.zeros((1, 3)), "fk": None}
    g = np.zeros(3)
    assert TrajectoryOptimizer._reach_probe(None, goal) is None
    assert TrajectoryOptimizer._reach_probe(GraspAnchor("ee_right", g, 3, 0.0, 2000.0), goal) is None
    assert TrajectoryOptimizer._reach_probe(GraspAnchor("ee_right", g, 3, 0.0, 0.0, goal_box=0.005), goal) is None
    r = GraspAnchor("ee_right", g, 3, 0.0, 0.0, goal_box=0.005, reach_constraint=True)
    assert TrajectoryOptimizer._reach_probe(r, None) is None and TrajectoryOptimizer._reach_probe(r, goal)["r"] == 0.005


# ------------------------------------------------------------------------------------------------ 5. SafePolicy
def test_safe_policy_builds_it_only_with_the_flag():
    from AG3S.trajopt.safe_policy import SafePolicy
    from tests.trajopt.test_safe_policy import _safe_policy

    assert SafePolicy._grasp_set is None
    assert _safe_policy()._grasp_set is None
    on = _safe_policy(to_config=TrajOptConfig.from_dict({"cost": {"grasp_set": True}}))
    assert isinstance(on._grasp_set, GS.GraspSetController)


def test_safe_policy_request_is_not_withdrawn_by_hsr6_objective(monkeypatch, ctrl_parts):
    from AG3S.trajopt import hsr6 as H6
    from tests.trajopt.test_core_eval_a6 import _gate_policy, _site, _target
    from AG3S.trajopt.grasp_latch import GraspPhase

    model, layout, cols, ctrl = ctrl_parts
    ctrl.reset()
    _fake_generator(monkeypatch, [(0.004, 0.0, 0.0)])
    _field_ok(monkeypatch)
    p, m = _gate_policy(hsr6=H6.Hsr6Flags(objective=True))
    p.to_config = TrajOptConfig.from_dict({"cost": {"grasp_set": True}})
    p._hsr6_start({"p": {c: (0.9 if c == "approach" else 0.02) for c in H6.PHASES6}, "argmax": "approach"})
    p._latch.phase = GraspPhase.LATCHED
    q = np.zeros(len(m.joint_names))
    _target(p, _site(m, q) + np.array([0.0, 0.0, -0.03]))
    p.ag3s.manipulated = _manip(p.ag3s.manipulated.centroid)
    p._grasp_set = GS.GraspSetController(p.to_config.cost, model, layout, {"left": "ee_left", "right": "ee_right"},
                                         cols, 0.85, p.to_config.limits, horizon=16)
    p.refiner = types.SimpleNamespace(optimizer=types.SimpleNamespace(
        linearizer=types.SimpleNamespace(robot_model=model, param_names=())),
        last_result=types.SimpleNamespace(solve_time_ms=12.5))
    p._pending = {"chunk": _chunk(layout, cols, q, 5), "timing": {}}
    ctx = {}
    assert not p._hsr6_objective_ok(q, ["left"])                         # ④ would withdraw A4 / G1 here
    p._grasp_set_request(q, ["left"], ctx, {"filtered_cloud": None}, "SCENE")
    assert ctx["grasp_anchor"].reach_constraint is True
    rec = p._last_grasp_record["grasp_set"]
    assert rec["active"] is True and rec["window"]["site_to_target_mm"] == pytest.approx(30.0, abs=0.1)
    assert "grasp_set" in p._pending["timing"]
    p._grasp_set_result(_chunk(layout, cols, q, 5))
    assert rec["solve_ms"] == 12.5 and rec["reach_err_mm"] is not None
    # outside the 60 mm window: nothing in the context
    _target(p, _site(m, q) + np.array([0.0, 0.07, 0.0]))
    p.ag3s.manipulated = _manip(p.ag3s.manipulated.centroid)
    ctx = {}
    p._grasp_set_request(q, ["left"], ctx, {}, "SCENE")
    assert "grasp_anchor" not in ctx and p._last_grasp_record["grasp_set"]["reason"].startswith("grasp-moment window")


# ------------------------------------------------------------------------------------------------ 6. config / CLI
def test_config_off_by_default_no_keys_and_validation():
    c = TrajOptConfig()
    assert c.cost.grasp_set is False
    assert not any(k.startswith("grasp_set") for k in c.to_dict()["cost"])
    on = TrajOptConfig.from_dict({"cost": {"grasp_set": True}})
    assert on.to_dict()["cost"] == {**c.to_dict()["cost"], "grasp_set": True}
    assert TrajOptConfig.from_dict(on.to_dict()).cost.grasp_set is True
    for bad in ({"w_grasp_goal": 2000.0}, {"w_grasp_tcp": 1.0}, {"grasp_goal_box": 0.005}, {"geo_grasp": True},
                {"grasp_set_switch_ratio": 0.0}, {"grasp_set_rows": 0}, {"grasp_set_sigma_p": 0.0}):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"cost": {"grasp_set": True, **bad}})


def _t72_argv():
    from tests.trajopt.test_core_eval_a6 import FINAL

    a4 = "--grasp-anchor 2000 --grasp-anchor-path-weight 0 "
    g1 = "--geo-grasp --geo-goal-mode correct --geo-track-scale 0.25 --geo-continuity-off "
    assert a4 in FINAL and g1 in FINAL and "--w-deflection-rate 20" in FINAL
    return (FINAL.replace(a4, "").replace(g1, "").replace("--w-deflection-rate 20", "--w-deflection-rate 0")
            + " --w-continuity 0 --grasp-set-to")


def test_t72_argv_parses_to_the_fixed_values(monkeypatch):
    from AG3S.trajopt import gpu_seeds, serve_safe

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, "patched for a CPU test"))
    p = serve_safe.build_parser()
    a = p.parse_args(BASE_ARGV + shlex.split(_t72_argv()))
    serve_safe.reject_bad_flag_combinations(p, a)
    c = serve_safe.trajopt_config_from_args(a)
    k = c.cost
    assert k.grasp_set is True and k.w_deflection_rate == 0.0 and k.w_continuity == 0.0
    assert k.w_grasp_goal == 0.0 and k.w_grasp_tcp == 0.0 and k.geo_grasp is False and k.w_track == 1.0
    assert k.w_smooth == TrajOptConfig().cost.w_smooth and k.wrist_track_scale == pytest.approx(0.1)
    assert (k.grasp_set_mu, k.grasp_set_max_candidates, k.grasp_set_time_budget_ms) == (0.8, 8, 150.0)
    assert (k.grasp_set_box, k.grasp_set_rows, k.grasp_set_w_rot, k.grasp_set_gate) == (0.005, 3, 5.0, 0.060)
    assert c.collision.neighbour_palm_contact_depth == pytest.approx(0.010)
    assert c.collision.neighbour_contact_depth == pytest.approx(0.020)
    assert serve_safe.hsr6_flags_from_args(a).names() == ["attach_assist", "held_gate", "target_exempt", "objective"]
    assert c.sqp.gpu_seeds == "curobo" and c.sqp.multi_seed == 16


@pytest.mark.parametrize("extra", ["--grasp-anchor 2000", "--grasp-goal-box 5", "--geo-grasp"])
def test_start_refused_with_a4_e1b_or_g1(monkeypatch, extra):
    from AG3S.trajopt import gpu_seeds, serve_safe

    monkeypatch.setattr(gpu_seeds, "available", lambda: (True, "patched for a CPU test"))
    p = serve_safe.build_parser()
    a = p.parse_args(BASE_ARGV + shlex.split(_t72_argv() + " " + extra))
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(p, a)


def test_cli_off_has_no_key():
    from AG3S.trajopt import serve_safe

    p = serve_safe.build_parser()
    assert "grasp_set" not in serve_safe.cost_overrides(p.parse_args(BASE_ARGV))
    assert serve_safe.cost_overrides(p.parse_args(BASE_ARGV + ["--grasp-set-to"]))["grasp_set"] is True


# ------------------------------------------------------------------------------------------------ 7. step 1: tracking scale
def test_track_scale_on_the_grasping_arm_only_while_the_reach_box_is_on(monkeypatch, ctrl_parts):
    from AG3S.trajopt import hsr6 as H6
    from tests.trajopt.test_core_eval_a6 import _gate_policy, _site, _target
    from AG3S.trajopt.grasp_latch import GraspPhase

    model, layout, cols, ctrl = ctrl_parts
    _fake_generator(monkeypatch, [(0.004, 0.0, 0.0)])
    _field_ok(monkeypatch)
    p, m = _gate_policy(hsr6=H6.Hsr6Flags(objective=True))
    p.to_config = TrajOptConfig.from_dict({"cost": {"grasp_set": True, "grasp_set_track_scale": 0.25}})
    p.layout = layout
    p._latch.phase = GraspPhase.LATCHED
    q = np.zeros(len(m.joint_names))
    _target(p, _site(m, q) + np.array([0.0, 0.0, -0.03]))
    p.ag3s.manipulated = _manip(p.ag3s.manipulated.centroid)
    p._grasp_set = GS.GraspSetController(p.to_config.cost, model, layout, {"left": "ee_left", "right": "ee_right"},
                                         cols, 0.85, p.to_config.limits, horizon=16)
    p.refiner = types.SimpleNamespace(optimizer=types.SimpleNamespace(
        linearizer=types.SimpleNamespace(robot_model=model, param_names=())))
    p._pending = {"chunk": _chunk(layout, cols, q, 5), "timing": {}}
    names = [str(layout.joint_names[i]) for i in layout.q_indices]
    base = np.full(len(names), 0.5)
    ctx = {"joint_track_scale": base.copy()}
    p._grasp_set_request(q, ["left"], ctx, {"filtered_cloud": None}, "SCENE")
    assert ctx["grasp_anchor"].reach_constraint is True
    want = np.array([0.125 if n.startswith("left_arm_") else 0.5 for n in names])
    assert any(n.startswith("left_arm_") for n in names)
    np.testing.assert_allclose(ctx["joint_track_scale"], want)          # multiplies the IFG-TOW scale, left arm only
    assert p._last_grasp_record["grasp_set"]["track_scale"] == 0.25
    # outside the window: no reach box, the tracking scale is left as it was
    _target(p, _site(m, q) + np.array([0.0, 0.07, 0.0]))
    p.ag3s.manipulated = _manip(p.ag3s.manipulated.centroid)
    ctx = {"joint_track_scale": base.copy()}
    p._grasp_set_request(q, ["left"], ctx, {}, "SCENE")
    assert "grasp_anchor" not in ctx
    np.testing.assert_array_equal(ctx["joint_track_scale"], base)


def test_track_scale_config_and_cli():
    from AG3S.trajopt import serve_safe
    from AG3S.trajopt.config import TrajOptConfigError

    assert TrajOptConfig().cost.grasp_set_track_scale == 1.0
    assert "grasp_set_track_scale" not in json.dumps(TrajOptConfig.from_dict({"cost": {"grasp_set": True}}).to_dict())
    for bad in (0.0, 1.5):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"cost": {"grasp_set": True, "grasp_set_track_scale": bad}})
    p = serve_safe.build_parser()
    a = p.parse_args(BASE_ARGV + ["--grasp-set-to", "--grasp-set-track-scale", "0.25"])
    assert serve_safe.cost_overrides(a)["grasp_set_track_scale"] == 0.25
    assert "grasp_set_track_scale" not in serve_safe.cost_overrides(p.parse_args(BASE_ARGV + ["--grasp-set-to"]))
    for argv in (["--grasp-set-track-scale", "0.25"], ["--grasp-set-to", "--grasp-set-track-scale", "0"]):
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(p, p.parse_args(BASE_ARGV + argv))


# ------------------------------------------------------------------------------------------------ 8. T72b
def test_t72b_full_horizon_opens_the_gate_only_for_a_reach_box():
    """Look-ahead (plan 10, execution 4), gate always closed: a reach box opens it only with the flag; a plain anchor
    or the flag off keeps the window solve (bit identical to before)."""
    from AG3S.trajopt.grasp_anchor import GraspAnchor
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from AG3S.trajopt.grasp_anchor import LinkPointFK
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    goal = LinkPointFK(sc.robot, sc.layout, 10, "ee_right")(sc.reference, sc.q_now)[0][6] + np.array([0.0, 0.0, 0.01])

    def solve(cost, reach):
        cfg = sc.config.with_overrides({"horizon": {"execution_length": 4, "lookahead": "obstacles",
                                                    "lookahead_gate": -1.0}, "collision": {"backend": "esdf"},
                                        "cost": cost})
        a = GraspAnchor("ee_right", goal, 6, 0.0, 0.0, goal_rows=3, goal_box=0.005, w_rot=0.0,
                        tol_pos=0.005, goal_box_slack_scale=1.0, reach_constraint=reach)
        opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
        r = opt.solve(sc.reference, sc.q_now, sc.scene, grasp_anchor=a)
        return r, opt._gate_record

    on = {"grasp_set": True, "grasp_set_full_horizon": True}
    r_off, g_off = solve({"grasp_set": True}, True)
    r_on, g_on = solve(on, True)
    r_plain, g_plain = solve(on, False)
    assert r_off.metrics["lookahead"]["gate"] == "closed" and "forced" not in g_off
    assert r_plain.metrics["lookahead"]["gate"] == "closed"
    assert np.array_equal(r_plain.trajectory, r_off.trajectory)
    assert r_on.metrics["lookahead"]["gate"] != "closed" and g_on["forced"] == "grasp-set reach box"
    assert not np.array_equal(r_on.trajectory[:, 4:], sc.reference[:, 4:])   # the tail is solved, not handed back


def test_t72b_field_only_skips_the_generator_scene_points(monkeypatch, ctrl_parts):
    model, layout, cols, ctrl = ctrl_parts
    seen = []
    from AG3S.ag3s.grasp import candidates as C

    _fake_generator(monkeypatch, [(0.004, 0.0, 0.0)])
    fake = C.generate_candidates

    def spy(pts, scene, *a, **k):
        seen.append(scene)
        return fake(pts, scene, *a, **k)

    monkeypatch.setattr(C, "generate_candidates", spy)
    _field_ok(monkeypatch)
    from AG3S.trajopt.safe_policy import hold_site_position

    cloud = np.random.default_rng(0).normal(size=(50, 3)) + 5.0      # far from the target: not dropped as target points
    q = np.zeros(len(model.joint_names))
    site = hold_site_position(model, q, "ee_left")
    for field_only in (False, True):
        cost = TrajOptConfig.from_dict({"cost": {"grasp_set": True, "grasp_set_field_only": field_only}}).cost
        c = GS.GraspSetController(cost, model, layout, {"left": "ee_left", "right": "ee_right"}, cols, 0.85,
                                  TrajOptConfig().limits, horizon=16)
        a = c.request(phase="latched", hand="left", grasp_link="ee_finger_l1", manip=_manip(site + [0.0, 0.0, -0.02]),
                      cloud_points=cloud, q_now=q, chunk=_chunk(layout, cols, q, 5), window=("ee_left", site),
                      window_rec={"site_to_target_mm": 20.0}, scene="SCENE",
                      linearizer=types.SimpleNamespace(robot_model=model, param_names=()))
        assert a is not None
    assert seen[0] is not None and len(seen[0]) and seen[1] is None


def test_t72b_config_and_cli():
    from AG3S.trajopt import serve_safe

    d = json.dumps(TrajOptConfig.from_dict({"cost": {"grasp_set": True}}).to_dict())
    assert "grasp_set_full_horizon" not in d and "grasp_set_field_only" not in d
    p = serve_safe.build_parser()
    a = p.parse_args(BASE_ARGV + shlex.split("--grasp-set-to --grasp-set-full-horizon --grasp-set-field-only "
                                             "--grasp-set-gate-mm 90 --grasp-set-budget-ms 300"))
    serve_safe.reject_bad_flag_combinations(p, a)
    o = serve_safe.cost_overrides(a)
    assert o["grasp_set_full_horizon"] is True and o["grasp_set_field_only"] is True
    assert o["grasp_set_gate"] == pytest.approx(0.090) and o["grasp_set_time_budget_ms"] == 300.0
    for argv in ("--grasp-set-full-horizon", "--grasp-set-gate-mm 90", "--grasp-set-to --grasp-set-budget-ms 0"):
        with pytest.raises(SystemExit):
            serve_safe.reject_bad_flag_combinations(p, p.parse_args(BASE_ARGV + shlex.split(argv)))
