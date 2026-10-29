"""AG3S-PRE-CORE-TO E1 — goal-keeping TrajOpt: E1-C (goal term without the palm path term) and E1-B (goal box).

`serve_safe --grasp-anchor W --grasp-anchor-path-weight 0` (E1-C) · `--grasp-goal-box MM` (E1-B), both default off.
Task: `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO.task.md` §6 E1.

What this file pins:
1. Off is bit-identical to HEAD 35b909cf (golden SHA-1 of fixture solves measured on a `git archive HEAD` copy), and
   the IFG-TOA path (no `--grasp-anchor-path-weight`) is unchanged; `QpProblem` without a goal box is as before.
2. E1-C: `w_tcp = 0` zeroes the path term (masks, weights, QP terms) and the solve records variant E1-C.
3. E1-B QP model: the box rows hold exactly when the slack equals the per-axis excess (`goal_box_excess`).
4. E1-B optimizer: no collision -> goal error <= r; collision vs goal -> collision resolved, slack recorded; an
   anchored violation (box priced above collision) -> the IFG-TOA3 safety-first re-solve; no goal rows -> plain solve
   with a record saying so.
5. Gate (`SafePolicy._grasp_anchor_gate`) hands out the box; the offline gate (`e1_goal.RecordedGraspGate`) is the
   server gate on a record.
6. Config / CLI.
"""

from __future__ import annotations

import hashlib
import sys
import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.grasp_anchor import GraspAnchor, anchor_qp_terms, anchor_targets

pytest.importorskip("casadi")
pytest.importorskip("osqp")

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _sha(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()[:16]


def _solve(sc, anchor=None, opt=None):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = opt or TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    kw = {} if anchor is None else {"grasp_anchor": anchor}
    return opt.solve(sc.reference, sc.q_now, sc.scene, **kw)


def _p_ref(sc, link, H=8):
    from AG3S.trajopt.grasp_anchor import LinkPointFK

    return LinkPointFK(sc.robot, sc.layout, H, link)(sc.reference, sc.q_now)[0]


# ------------------------------------------------------------------------------------------------ 1. off / TOA
#: SHA-1[:16] of the first solve's trajectory, `make_scenario(horizon=8, penetration=0.03)`, measured 2026-10-25 on
#: `git archive HEAD` (35b909cf) with .venv-ag3s (same values with and without OPENBLAS_NUM_THREADS=1).
HEAD_SHA = {
    "off": "2a2b43ded1875604",
    "toa_arm3_2000": "af92a699bf27d372",
    "toa_arm5_20000_fallback": "293650ea669714aa",
    "toa_goal": "8818ec4f303f720a",
    "toa_both": "fe1c25bb847f7dba",
}


def test_off_and_toa_are_bit_identical_to_head():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    goal = _p_ref(sc, "link_right_arm_5")[-1] + np.array([0.0, 0.0, 0.01])
    cases = {"off": None,
             "toa_arm3_2000": GraspAnchor("link_right_arm_3", None, None, 2000.0, 0.0),
             "toa_arm5_20000_fallback": GraspAnchor("link_right_arm_5", None, None, 20000.0, 0.0),
             "toa_goal": GraspAnchor("link_right_arm_5", goal, 6, 0.0, 5000.0),
             "toa_both": GraspAnchor("link_right_arm_5", goal, 5, 2000.0, 2000.0)}
    for name, anchor in cases.items():
        r = _solve(sc, anchor)
        assert _sha(r.trajectory) == HEAD_SHA[name], name
    off = _solve(sc)
    assert "grasp_anchor" not in off.metrics
    # IFG-TOA keeps its record shape (no E1 keys besides `variant` / `goal_rows`)
    toa = _solve(sc, cases["toa_arm5_20000_fallback"]).metrics["grasp_anchor"]
    assert toa["variant"] == "TOA" and "fallback_used" not in toa and "goal_box_mm" not in toa


def test_qp_problem_without_goal_box_is_unchanged():
    from AG3S.trajopt.problem import build_problem
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=6, penetration=0.03)
    pb = build_problem(sc.reference, sc.reference, sc.limits, sc.config, q_now=sc.q_now[sc.layout.q_indices],
                       n_slack=4)
    assert pb.n_goal_slack == 0 and pb.n_var == pb.n_q + pb.n_slack + pb.n_kinematic_slack
    z = pb.flatten(sc.reference)
    assert z.shape == (pb.n_var,)
    np.testing.assert_array_equal(pb.kinematic_slack(z), z[pb.n_q + pb.n_slack:])
    assert pb.goal_slack(z).size == 0


# ------------------------------------------------------------------------------------------------ 2. E1-C
def test_e1c_path_weight_zero_drops_the_path_term():
    rng = np.random.default_rng(5)
    nq, H = 4, 6
    A = rng.normal(0, 0.2, (H, 3, nq)); b = rng.normal(0, 0.5, (H, 3))
    pmap = lambda Q: np.stack([A[k] @ Q[:, k] + b[k] for k in range(H)])   # noqa: E731
    R, X = rng.normal(0, 0.3, (nq, H)), rng.normal(0, 0.3, (nq, H))
    goal = np.array([0.1, -0.2, 0.3])
    e1c = GraspAnchor("l", goal, 2, 0.0, 2000.0)
    assert e1c.variant == "E1-C"
    track, gl = e1c.masks(H)
    assert not track.any() and list(np.flatnonzero(gl)) == [2, 3, 4]
    w, tg = anchor_targets(e1c, pmap(R))
    assert np.all(w[:, 0] == 0.0) and np.all(w[[0, 1, 5], 1] == 0.0)
    P, q = anchor_qp_terms(X, pmap(X), A, w, tg)
    for k in (0, 1, 5):                                                       # rows without the goal: nothing at all
        sl = slice(k * nq, (k + 1) * nq)
        assert P[sl, sl].nnz == 0 or np.allclose(P[sl, sl].toarray(), 0.0)
        assert np.allclose(q[sl], 0.0)
    # the same QP terms as the goal term of the TOA anchor alone
    toa = GraspAnchor("l", goal, 2, 7.0, 2000.0)
    w2, tg2 = anchor_targets(toa, pmap(R))
    w2[:, 0] = 0.0
    P2, q2 = anchor_qp_terms(X, pmap(X), A, w2, tg2)
    np.testing.assert_allclose(P.toarray(), P2.toarray()); np.testing.assert_allclose(q, q2)
    assert GraspAnchor("l", goal, 2, 2000.0, 2000.0).variant == "TOA"


def test_e1c_solve_records_the_variant_and_no_path_rows():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=-0.05)                          # no collision
    link = "link_right_arm_5"
    goal = _p_ref(sc, link)[5] + np.array([0.02, 0.0, 0.0])
    r = _solve(sc, GraspAnchor(link, goal, 5, 0.0, 2000.0))
    rec = r.metrics["grasp_anchor"]
    assert rec["variant"] == "E1-C" and rec["n_track_rows"] == 0 and rec["goal_rows"] == [5, 6, 7]
    assert rec["goal_err_result_mm"] < rec["goal_err_reference_mm"] and rec["fallback_used"] is False


# ------------------------------------------------------------------------------------------------ 3. E1-B QP model
def test_goal_box_rows_hold_exactly_at_the_excess():
    import scipy.sparse as sp

    from AG3S.trajopt.grasp_anchor import add_goal_box, goal_box_excess
    from AG3S.trajopt.problem import QpProblem

    rng = np.random.default_rng(11)
    nq, H = 4, 5
    A_lin = rng.normal(0, 0.2, (H, 3, nq)); b = rng.normal(0, 0.5, (H, 3))
    pmap = lambda Q: np.stack([A_lin[k] @ Q[:, k] + b[k] for k in range(H)])   # noqa: E731
    X = rng.normal(0, 0.3, (nq, H))
    n_q = nq * H
    base = QpProblem(P=sp.identity(n_q + 2, format="csc"), q=np.zeros(n_q + 2), A=sp.csc_matrix((0, n_q + 2)),
                     l=np.zeros(0), u=np.zeros(0), n_q=n_q, n_slack=2, nq=nq, horizon=H)
    rows, goal, r, w = np.array([2, 3]), np.array([0.05, -0.1, 0.2]), 0.01, 7.0
    pb = add_goal_box(base, X, pmap(X), A_lin, rows, goal, r, w)
    m = 3 * rows.size
    assert pb.n_goal_slack == m and pb.n_var == n_q + 2 + m and pb.P.shape == (pb.n_var, pb.n_var)
    np.testing.assert_array_equal(pb.q[-m:], np.full(m, w))
    for _ in range(5):
        Q = X + rng.normal(0, 0.3, X.shape)
        ex = goal_box_excess(pmap(Q), rows, goal, r).reshape(-1)              # row-major (row, axis) = slack order
        z = np.concatenate([Q.T.reshape(-1), np.zeros(2), ex])
        Az = pb.A @ z
        assert np.all(Az >= pb.l - 1e-12) and np.all(Az <= pb.u + 1e-12)
        if ex.max() > 1e-6:                                                   # less slack on a violated axis: infeasible
            z2 = z.copy(); z2[n_q + 2 + int(np.argmax(ex))] -= 1e-6
            Az2 = pb.A @ z2
            assert np.any(Az2 < pb.l - 1e-12) or np.any(Az2 > pb.u + 1e-12)
        np.testing.assert_array_equal(pb.goal_slack(z), ex)
    assert set(pb.row_blocks) >= {"goal_box", "goal_box_lower", "goal_box_slack"}


# ------------------------------------------------------------------------------------------------ 4. E1-B optimizer
@pytest.mark.parametrize("r_mm", [10.0, 20.0])
def test_e1b_without_collision_meets_the_box(r_mm):
    """Measured 2026-10-25: policy palm 162 mm (axis) from the goal on the goal rows, result 10.01 / 20.01 mm."""
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=-0.05)                          # 50 mm clear: nothing to avoid
    link = "link_right_arm_3"
    goal = _p_ref(sc, link)[5] + np.array([0.02, 0.0, 0.0])
    r = _solve(sc, GraspAnchor(link, goal, 5, 0.0, 0.0, goal_box=r_mm / 1000.0))
    rec = r.metrics["grasp_anchor"]
    assert rec["variant"] == "E1-B" and rec["goal_box_mm"] == r_mm and rec["goal_rows"] == [5, 6, 7]
    assert rec["n_track_rows"] == 0 and rec["w_tcp"] == 0.0 and rec["w_goal"] == 0.0
    assert rec["goal_err_axis_max_reference_mm"] > r_mm + 20.0                # the policy is really off the goal
    assert rec["goal_err_axis_max_result_mm"] <= r_mm + 0.5                   # linearisation tolerance
    assert rec["goal_box_excess_max_mm"] <= 0.5 and r.max_violation <= float(sc.config.safety.violation_tolerance)
    assert rec["fallback_used"] is False and "fallback" not in rec


def test_e1b_collision_wins_over_the_goal_and_the_slack_is_recorded():
    """Goal = the clipped link's own reference position at the deepest step (one row, ±2 mm): keeping it means staying
    in the obstacle. Measured 2026-10-25: violation 0, goal error 33.7 mm, slack 39.1 mm, no fallback."""
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    tol = float(sc.config.safety.violation_tolerance)
    link = "link_right_arm_5"
    p_ref = _p_ref(sc, link)
    base = _solve(sc)
    assert base.max_violation <= tol
    assert np.linalg.norm(np.asarray(sc.reference)) > 0                      # (fixture sanity)
    r = _solve(sc, GraspAnchor(link, p_ref[3], 3, 0.0, 0.0, goal_rows=1, goal_box=0.002))
    rec = r.metrics["grasp_anchor"]
    assert r.max_violation <= tol                                            # collision resolved
    assert rec["goal_rows"] == [3]
    assert rec["goal_err_axis_max_result_mm"] > 10.0                          # the goal gave way ...
    assert rec["goal_box_slack_sum_mm"] > 10.0 and rec["goal_box_excess_max_mm"] > 5.0   # ... and the record says so
    assert rec["fallback_used"] is False                                     # resolved by the boxed solve itself


def test_e1b_anchored_violation_falls_back_to_the_plain_solve(monkeypatch):
    """With the box priced above collision (scale 100 instead of 0.1) the boxed solve keeps the goal and stays in the
    obstacle; the IFG-TOA3 safety-first re-solve without the anchor wins. Measured: anchored 39.7 mm, plain 0."""
    import AG3S.trajopt.grasp_anchor as GA
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    tol = float(sc.config.safety.violation_tolerance)
    link = "link_right_arm_5"
    p_ref = _p_ref(sc, link)
    monkeypatch.setattr(GA, "GOAL_BOX_SLACK_SCALE", 100.0)
    r = _solve(sc, GraspAnchor(link, p_ref[3], 3, 0.0, 0.0, goal_rows=1, goal_box=0.002))
    rec = r.metrics["grasp_anchor"]
    assert rec["fallback"]["used"] == "plain" and rec["fallback"]["anchored_violation_mm"] > 1.0
    assert rec["fallback_used"] is True and rec["variant"] == "E1-B"
    assert r.max_violation <= tol


def test_e1b_without_goal_rows_is_the_plain_solve_with_a_record():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    base = _solve(sc)
    for close in (None, 0, 12):                                              # no close / starts closed / beyond H
        r = _solve(sc, GraspAnchor("link_right_arm_5", np.zeros(3), close, 0.0, 0.0, goal_box=0.01))
        np.testing.assert_array_equal(r.trajectory, base.trajectory)
        rec = r.metrics["grasp_anchor"]
        assert rec["applied"] is False and rec["n_goal_rows"] == 0 and rec["variant"] == "E1-B"
        assert set(r.metrics) - {"grasp_anchor"} == set(base.metrics)


# ------------------------------------------------------------------------------------------------ 5. gates
def test_safe_policy_gate_hands_out_the_box():
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from tests.trajopt.test_ifg_toa_grasp_anchor import _chunk, _gate_policy

    p, model = _gate_policy()
    p.to_config = TrajOptConfig.from_dict({"cost": {"grasp_goal_box": 0.02}})
    q = np.zeros(len(model.joint_names))
    palm = np.asarray(_link_pose_numeric(model, q, "ee_left")[:3, 3])
    p.ag3s.manipulated = types.SimpleNamespace(id=0, centroid=palm + np.array([0.0, 0.0, -0.03]))
    p._pending = {"chunk": _chunk(p, close_row=12)}
    a = p._grasp_anchor_gate(q, ["left"])
    assert a is not None and a.variant == "E1-B" and a.goal_box == 0.02 and a.w_tcp == 0.0 and a.w_goal == 0.0
    assert a.close_row == 12 and p._last_grasp_anchor["frozen"]
    np.testing.assert_allclose(a.goal, palm, atol=1e-9)
    # E1-C: same gate, goal term only
    p2, _ = _gate_policy()
    p2.to_config = TrajOptConfig.from_dict({"cost": {"w_grasp_tcp": 0.0, "w_grasp_goal": 2000.0}})
    p2.ag3s.manipulated = p.ag3s.manipulated
    p2._pending = {"chunk": _chunk(p2, close_row=12)}
    a2 = p2._grasp_anchor_gate(q, ["left"])
    assert a2.variant == "E1-C" and a2.goal_box == 0.0
    np.testing.assert_allclose(a2.goal, a.goal)


def test_recorded_gate_is_the_server_gate_on_a_record():
    """`e1_goal.RecordedGraspGate`: freeze at the first latched request that plans a close (not before), keep it while
    the policy's close point drifts, no anchor far from the goal, clear on `held`."""
    from AG3S.ag3s.constraints.attached import _link_pose_numeric
    from AG3S.trajopt.experiments.e1_goal import RecordedGraspGate, goal_eval
    from AG3S.trajopt.types import ChunkLayout
    from tests.trajopt.test_ifg_toa_grasp_anchor import _chunk, _gate_policy

    p, model = _gate_policy()                                                # only for its chunk helper
    run = types.SimpleNamespace(robot=model, layout=ChunkLayout.rby1(model.joint_names), hand="left")
    cost = TrajOptConfig.from_dict({"cost": {"grasp_goal_box": 0.01}}).cost
    gate = RecordedGraspGate(run, cost)
    q = np.zeros(len(model.joint_names))
    palm = np.asarray(_link_pose_numeric(model, q, "ee_left")[:3, 3])
    manip = {"id": 0, "centroid": (palm + np.array([0.0, 0.0, -0.03])).tolist()}

    def summ(state, **g):
        return {"grasp": {"state": state, "hand": "left", "hold_link": "ee_left", "parent_link": "ee_finger_l1", **g},
                "manipulated": manip}

    a, rec = gate.step(1, summ("searching"), _chunk(p, close_row=3), q)
    assert a is None and rec["reason"] == "phase searching"
    a, rec = gate.step(2, summ("latched"), _chunk(p), q)
    assert a is None and rec["reason"] == "no close planned yet"
    a, rec = gate.step(3, summ("latched"), _chunk(p, close_row=6), q)
    assert a is not None and rec["frozen"] and a.close_row == 6 and a.variant == "E1-B"
    goal = a.goal.copy()
    a, rec = gate.step(4, summ("closing"), _chunk(p, close_row=2, shift=0.05), q)
    np.testing.assert_allclose(a.goal, goal) and None
    assert a.close_row == 2 and "frozen" not in rec
    ev = goal_eval(run, a, run.layout.chunk_to_trajectory(_chunk(p))[:, :8],
                   run.layout.chunk_to_trajectory(_chunk(p, shift=0.05)), q)
    assert ev["goal_rows"] == [2, 3, 4] and ev["result_err_mm"] == pytest.approx(0.0, abs=1e-6)
    assert ev["reference_err_mm"] > 1.0
    far = q.copy(); far[[i for i, n in enumerate(model.joint_names) if n == "left_arm_1"][0]] = 0.8
    a, rec = gate.step(5, summ("closing"), _chunk(p, close_row=2), far)
    assert a is None and rec["reason"] == "palm outside the anchor radius"
    a, rec = gate.step(6, summ("held"), _chunk(p, close_row=0), q)
    assert a is None and rec["cleared"] == "phase held"
    a, rec = gate.step(7, summ("latched"), _chunk(p, close_row=4), q)          # re-frozen after the clear
    assert a is not None and rec["frozen"]
    a, rec = gate.step(8, summ("closing", attach_revoked={"reasons": ["x"]}), _chunk(p, close_row=0), q)
    assert a is None and rec["cleared"] == "revoked"


# ------------------------------------------------------------------------------------------------ 6. config / CLI
def test_config_and_cli(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    assert TrajOptConfig().cost.grasp_goal_box == 0.0
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"cost": {"grasp_goal_box": -0.01}})
    with pytest.raises(TrajOptConfigError):
        TrajOptConfig.from_dict({"cost": {"grasp_goal_box": 0.01, "w_grasp_goal": 2000.0}})
    parse = lambda *a: serve_safe.build_parser().parse_args([*BASE_ARGV, *a])   # noqa: E731
    assert serve_safe.cost_overrides(parse()) == {}                           # off: no key at all
    # default path weight = today's IFG-TOA
    assert serve_safe.cost_overrides(parse("--grasp-anchor", "2000")) == {"w_grasp_tcp": 2000.0, "w_grasp_goal": 2000.0}
    assert serve_safe.cost_overrides(parse("--grasp-anchor", "2000", "--grasp-anchor-path-weight", "0")) == {
        "w_grasp_tcp": 0.0, "w_grasp_goal": 2000.0}
    over = serve_safe.cost_overrides(parse("--grasp-goal-box", "10", "--grasp-anchor-radius-mm", "120"))
    assert over == {"grasp_goal_box": 0.01, "grasp_anchor_radius": 0.12}
    assert TrajOptConfig.from_dict({"cost": over}).cost.grasp_goal_box == 0.01
    for extra, needle in ((["--grasp-anchor-path-weight", "0"], "--grasp-anchor"),
                          (["--grasp-anchor", "2000", "--grasp-anchor-path-weight", "-1"], "--grasp-anchor-path-weight"),
                          (["--grasp-goal-box", "0"], "--grasp-goal-box"),
                          (["--grasp-goal-box", "10", "--grasp-anchor", "2000"], "--grasp-goal-box"),
                          (["--no-safe", "--grasp-goal-box", "10"], "--w-*")):
        monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *extra])
        with pytest.raises(SystemExit) as e:
            serve_safe.main()
        assert e.value.code == 2
        assert needle in capsys.readouterr().err


def test_offline_settings_are_registered_and_valid():
    from AG3S.trajopt.experiments import e1_goal

    assert set(e1_goal.SETTINGS) >= {"E0", "E1C", "E1B10", "E1B20"}
    base = TrajOptConfig()
    for name, over in e1_goal.SETTINGS.items():
        cfg = base.with_overrides(over) if over else base
        assert e1_goal.grasp_on(cfg.cost) == (name != "E0")
    assert e1_goal.probe_cost(base.cost).grasp_goal_box > 0.0


def test_summarize_counts_goal_chunks_and_all_chunks(tmp_path):
    import json

    from AG3S.trajopt.experiments.e1_goal import summarize

    def var(viol, err=None, fb=False):
        e = {"max_violation_mm": viol, "solve_ms": 30.0, "status": "feasible"}
        if err is not None:
            e |= {"goal_eval": {"result_err_axis_max_mm": err, "reference_err_axis_max_mm": 12.0},
                  "grasp_anchor": {"fallback_used": fb}}
        return e

    chunks = [{"seq": 1, "variants": {"E0": var(0.0), "E1B10": var(0.0)}},
              {"seq": 2, "variants": {"E0": var(2.0, 12.0), "E1B10": var(0.0, 9.5, fb=True)}}]
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"chunks": chunks}))
    t = summarize([str(path)])
    assert t["E0"]["all_chunks"]["n"] == 2 and t["E0"]["all_chunks"]["n_violation_gt_0p1mm"] == 1
    assert t["E0"]["n_goal_chunks"] == 1 and t["E0"]["goal_err_axis_mm_median"] == 12.0
    assert t["E1B10"]["n_fallback_used"] == 1 and t["E1B10"]["goal_err_axis_mm_median"] == 9.5
