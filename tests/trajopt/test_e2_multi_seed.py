"""AG3S-PRE-CORE-TO E2 — multi-seed TrajOpt (`serve_safe --to-multi-seed N`, `sqp.multi_seed`, default 0 = off).

Task: `AG3S/ag3s/docs/handoff/AG3S-PRE-CORE-TO.task.md` §6 "E2 구현 명세".

What this file pins:
1. Off is bit-identical to HEAD b3b69311 (golden SHA-1 of fixture solves measured on a `git archive HEAD` copy): the
   optimizer (standard fixture, stuck fixture twice in a row, E1-C anchor) and the refiner (two chunks with `t_step`).
2. On, but the standard solve resolves: not triggered — the same trajectory, the same next chunk, a record saying so.
3. Stuck fixture (an obstacle head-on across the swept path, one SQP iteration): the standard solve stays at 18.8 mm,
   a detour seed resolves it, the returned trajectory is a resolved seed; the main QP solver and the optimizer state
   (`_previous`, `frame_index`) end as if only the chosen trajectory had been solved.
4. Selection order: violation → goal error → path length → accelerations, ties to the standard; the recorded choice is
   the rule applied to the recorded candidates; a goal (zero-weight anchor) moves the choice to the seed nearest it.
5. Determinism: same input and chunk index → same result and record; another index changes only the random seeds.
6. Seed construction, config / CLI, refiner wiring, offline settings and the summary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

pytest.importorskip("casadi")
pytest.importorskip("osqp")

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]

#: SHA-1[:16] of the fixture solves below, measured 2026-10-26 on `git archive HEAD` (b3b69311) with .venv-ag3s
#: (`OPENBLAS_NUM_THREADS=1`); the same values in this checkout with `sqp.multi_seed` 0.
HEAD_SHA = {
    "std_off": "2a2b43ded1875604",
    "stuck_off_1": "6766eb5f80166ac0",
    "stuck_off_2": "d2ccad9345c46a69",
    "stuck_e1c_off": "770c42b74840a39b",
    "refiner_off_1": "c8a1a52ec7a483cf",
    "refiner_off_2": "ac8432ef4cf6843e",
}


def _sha(a) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(a, np.float64)).tobytes()).hexdigest()[:16]


def _head_on(sc, radius=0.05, d_safe=0.03, offset=0.002, step=None):
    """One candidate sphere centred (2 mm off) on the most-moving robot sphere's path at the middle step: the reference
    runs straight through it (159 mm), and one SQP iteration cannot leave it (18.8 mm left)."""
    lin = sc.linearizer
    centres, _ = lin.sphere_states(sc.reference, sc.q_now)
    H = centres.shape[0]
    mid = H // 2 if step is None else step
    s = int(np.argmax(np.linalg.norm(centres[mid] - centres[0], axis=1)))
    d = centres[min(mid + 1, H - 1), s] - centres[max(mid - 1, 0), s]
    d /= np.linalg.norm(d)
    perp = np.cross(d, [0.0, 0.0, 1.0])
    perp /= np.linalg.norm(perp)
    n = sc.scene.candidate_pos.shape[0]
    act = np.zeros(n, bool)
    act[0] = True
    return dataclasses.replace(sc.scene, candidate_pos=np.tile(centres[mid, s] + perp * offset, (n, 1)),
                               candidate_radius=np.full(n, radius), candidate_active=act,
                               d_safe=np.full((lin.n_spheres, n), d_safe))


def _stuck():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03, sqp={"max_iterations": 1, "min_iterations": 1})
    return sc, _head_on(sc)


def _opt(sc, n=0, **over):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    cfg = sc.config.with_overrides({"sqp": {"multi_seed": n}, **over}) if (n or over) else sc.config
    return TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)


def _strip_ms(rec):
    """The record without wall-clock fields (for determinism)."""
    out = {k: v for k, v in rec.items() if k not in ("total_ms", "standard_ms")}
    if "candidates" in out:
        out["candidates"] = [{k: v for k, v in c.items() if k != "solve_ms"} for c in out["candidates"]]
    return out


# ------------------------------------------------------------------------------------------------ 1. off
def test_off_is_bit_identical_to_head():
    from AG3S.trajopt.grasp_anchor import GraspAnchor, LinkPointFK
    from AG3S.trajopt.refiner import TrajOptChunkRefiner
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    assert TrajOptConfig().sqp.multi_seed == 0
    sc = make_scenario(horizon=8, penetration=0.03)
    r = _opt(sc).solve(sc.reference, sc.q_now, sc.scene)
    assert _sha(r.trajectory) == HEAD_SHA["std_off"] and "multi_seed" not in r.metrics
    st, scene = _stuck()
    opt = _opt(st)
    r1, r2 = opt.solve(st.reference, st.q_now, scene), opt.solve(st.reference, st.q_now, scene)
    assert (_sha(r1.trajectory), _sha(r2.trajectory)) == (HEAD_SHA["stuck_off_1"], HEAD_SHA["stuck_off_2"])
    assert r1.max_violation * 1e3 == pytest.approx(18.82, abs=0.01) and "multi_seed" not in r1.metrics
    p = LinkPointFK(st.robot, st.layout, 8, "ee_right")(st.reference, st.q_now)[0]
    anc = GraspAnchor("ee_right", p[5] + np.array([0.0, 0.0, 0.01]), 5, 0.0, 2000.0)
    r = TrajectoryOptimizer(st.robot, st.layout, st.limits, st.config).solve(st.reference, st.q_now, scene,
                                                                              grasp_anchor=anc)
    assert _sha(r.trajectory) == HEAD_SHA["stuck_e1c_off"] and "multi_seed" not in r.metrics
    # an index handed to an optimizer with multi-seed off is ignored
    r = _opt(st).solve(st.reference, st.q_now, scene, multi_seed_index=7)
    assert _sha(r.trajectory) == HEAD_SHA["stuck_off_1"]
    ref = TrajOptChunkRefiner(st.robot, st.layout, lambda c: (scene, st.q_now, True),
                              st.config.with_overrides({"horizon": {"plan_horizon": None}}))
    c1 = ref.refine(st.chunk, {"t_step": 3, "previous_physical_chunk": None})
    c2 = ref.refine(st.chunk, {"t_step": 4, "previous_physical_chunk": c1})
    assert (_sha(c1), _sha(c2)) == (HEAD_SHA["refiner_off_1"], HEAD_SHA["refiner_off_2"])
    assert "multi_seed" not in ref.last_result.metrics


# ------------------------------------------------------------------------------------------------ 2. not triggered
def test_not_triggered_when_the_standard_solve_resolves():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=8, penetration=0.03)
    off, on = _opt(sc), _opt(sc, 16)
    a, b = off.solve(sc.reference, sc.q_now, sc.scene), on.solve(sc.reference, sc.q_now, sc.scene)
    assert _sha(b.trajectory) == _sha(a.trajectory) == HEAD_SHA["std_off"]
    assert b.max_violation <= float(sc.config.safety.violation_tolerance)
    rec = b.metrics["multi_seed"]
    assert rec["triggered"] is False and rec["n_seeds"] == 16 and "candidates" not in rec
    assert rec["reason"] == "standard solve within tolerance"
    assert {k: v for k, v in b.metrics.items() if k != "multi_seed"}.keys() == a.metrics.keys()
    assert "_ms_solver" not in on.__dict__                     # no seed solver was even built
    # the next chunk is the same too (no state was touched)
    assert _sha(on.solve(sc.reference, sc.q_now, sc.scene).trajectory) == _sha(
        off.solve(sc.reference, sc.q_now, sc.scene).trajectory)


# ------------------------------------------------------------------------------------------------ 3. stuck fixture
def test_stuck_standard_solve_is_resolved_by_a_detour_seed():
    sc, scene = _stuck()
    tol = float(sc.config.safety.violation_tolerance)
    off = _opt(sc)
    std = off.solve(sc.reference, sc.q_now, scene)
    assert std.max_violation > 0.015                                           # the standard solve stays inside
    on = _opt(sc, 16)
    res = on.solve(sc.reference, sc.q_now, scene, template=None)
    rec = res.metrics["multi_seed"]
    assert rec["triggered"] is True and rec["n_seeds"] == 16 and len(rec["candidates"]) == 17
    cands = rec["candidates"]
    assert cands[0]["kind"] == "standard" and cands[0]["max_violation_mm"] == pytest.approx(std.max_violation * 1e3,
                                                                                            abs=1e-3)
    assert [c["kind"] for c in cands[1:3]] == ["policy", "stay"]
    detours = [c for c in cands if c["kind"] == "detour"]
    assert len(detours) == 10 and rec["geometry"]["palm"] == "ee_right" and rec["geometry"]["away"] is not None
    assert any(c["eligible"] and c["max_violation_mm"] <= tol * 1e3 for c in detours)    # a detour resolves it
    # what is returned is a resolved candidate, and the record says which
    assert res.max_violation <= tol and res.status.safe
    assert rec["chosen"] > 0 and cands[rec["chosen"]]["max_violation_mm"] == pytest.approx(res.max_violation * 1e3,
                                                                                          abs=1e-3)
    assert rec["n_resolved"] >= 2 and rec["total_ms"] > 0.0
    assert res.solve_time_ms == pytest.approx(rec["standard_ms"] + rec["total_ms"], rel=1e-3, abs=0.05)
    assert any("multi-seed (E2)" in n for n in res.notes)
    # state: as if only the chosen trajectory had been solved; the main QP solver is the standard one's
    assert on.frame_index == off.frame_index == 1 and res.metrics["frame_index"] == 1
    np.testing.assert_array_equal(on._previous, res.trajectory)
    np.testing.assert_array_equal(on.solver._last_x, off.solver._last_x)
    assert on.__dict__["_ms_solver"] is not on.solver


def test_e1c_on_top_runs_the_seeds_once_with_the_anchor():
    """E1-C + E2: the anchored standard solve (with its IFG-TOA3 safety-first re-solve) is candidate 0, the seeds are
    solved with the same anchor terms, `_run_seeds` runs once (not again inside the nested re-solve), and the goal
    error enters the selection."""
    from AG3S.trajopt.grasp_anchor import GraspAnchor, LinkPointFK

    sc, _ = _stuck()
    scene = _head_on(sc, radius=0.07, step=4)       # measured: E1-C anchored 58.7 mm, its plain re-solve 11.6 mm
    p = LinkPointFK(sc.robot, sc.layout, 8, "ee_right")(sc.reference, sc.q_now)[0]
    anchor = GraspAnchor("ee_right", p[5] + np.array([0.0, 0.0, 0.01]), 5, 0.0, 2000.0)
    std = _opt(sc).solve(sc.reference, sc.q_now, scene, grasp_anchor=anchor)
    assert std.metrics["grasp_anchor"]["variant"] == "E1-C" and std.metrics["grasp_anchor"]["fallback_used"] is True
    assert std.max_violation > 0.01
    opt = _opt(sc, 16)
    n = {"runs": 0}
    real = opt._run_seeds

    def counted(*a, **k):
        n["runs"] += 1
        return real(*a, **k)

    opt._run_seeds = counted
    res = opt.solve(sc.reference, sc.q_now, scene, grasp_anchor=anchor)
    rec = res.metrics["multi_seed"]
    assert n["runs"] == 1 and rec["triggered"] is True and rec["goal_rows"] == [5, 6, 7]
    assert rec["candidates"][0]["max_violation_mm"] == pytest.approx(std.max_violation * 1e3, abs=1e-3)
    assert all(c["goal_err_mm"] is not None for c in rec["candidates"])
    assert res.max_violation <= float(sc.config.safety.violation_tolerance)
    ga = res.metrics["grasp_anchor"]
    assert ga["variant"] == "E1-C" and "multi_seed" in ga and ga["goal_err_result_mm"] == pytest.approx(
        rec["candidates"][rec["chosen"]]["goal_err_mm"], abs=0.02)


# ------------------------------------------------------------------------------------------------ 4. selection
def test_moving_first_rule():
    """User rule 2026-10-26: stay only when no moving candidate resolves."""
    from AG3S.trajopt.multi_seed import select_moving_first, selection_key

    tol = 1e-4
    k = lambda v: selection_key(v, tol, None, 1.0, 0.0)                        # noqa: E731
    kinds = ["standard", "policy", "stay", "detour", "noise"]
    viol = [3.0, 3.0, 0.0, 0.0, 0.0]
    keys = [k(v / 1e3) for v in viol]
    # stay and two moving candidates resolve: the moving one nearest the reference wins
    assert select_moving_first(kinds, keys, viol, 0.1, [None] * 5, [0.0, 0.0, 0.2, 0.5, 0.3]) == 4
    # a goal ranks before the reference deviation
    assert select_moving_first(kinds, keys, viol, 0.1, [None, None, None, 2.0, 9.0], [0, 0, 0.2, 0.5, 0.3]) == 3
    # an ineligible moving candidate does not count
    assert select_moving_first(kinds, [keys[0], keys[1], keys[2], None, None], viol, 0.1, [None] * 5,
                               [0.0, 0.0, 0.2, 0.5, 0.3]) == 2
    # nothing moving resolves: the original key over all (stay resolves -> stay)
    viol2 = [3.0, 3.0, 0.0, 1.0, 2.0]
    assert select_moving_first(kinds, [k(v / 1e3) for v in viol2], viol2, 0.1, [None] * 5, [0, 0, 0.2, 0.5, 0.3]) == 2


def test_selection_key_order_and_ties():
    from AG3S.trajopt.multi_seed import select, selection_key

    tol = 1e-4
    k = lambda v, g, L, a: selection_key(v, tol, g, L, a)                     # noqa: E731
    # violation first (a resolved candidate beats any goal / length), within tolerance = 0
    assert select([k(0.002, 1.0, 0.1, 0.0), k(0.0, 150.0, 5.0, 9.0)]) == 1
    assert select([k(0.00005, 9.0, 1.0, 1.0), k(0.0, 9.0, 1.0, 1.0)]) == 0    # both "resolved": tie -> standard
    # then the goal error, then the path length, then the accelerations
    assert select([k(0.0, 12.0, 0.1, 0.0), k(0.0, 3.0, 2.0, 5.0)]) == 1
    assert select([k(0.0, 3.0, 2.0, 0.0), k(0.0, 3.0, 1.0, 5.0)]) == 1
    assert select([k(0.0, None, 1.0, 0.5), k(0.0, None, 1.0, 0.2)]) == 1
    # an exact tie stays with the standard; an ineligible seed (None) never wins
    assert select([k(0.001, None, 1.0, 0.5), k(0.001, None, 1.0, 0.5)]) == 0
    assert select([k(0.003, None, 1.0, 0.5), None, k(0.002, None, 9.0, 9.0)]) == 2
    # violations count in 0.1 mm steps: 0.04 mm less is the same step, the next key decides
    assert select([k(0.00200, None, 1.0, 0.5), k(0.00196, None, 1.0, 0.5)]) == 0
    assert select([k(0.00200, None, 1.0, 0.5), k(0.00196, None, 0.5, 0.5)]) == 1


def test_recorded_choice_is_the_rule_on_the_recorded_candidates_and_a_goal_moves_it():
    from AG3S.trajopt.grasp_anchor import GraspAnchor, LinkPointFK
    from AG3S.trajopt.multi_seed import select_moving_first, selection_key

    sc, scene = _stuck()
    tol = float(sc.config.safety.violation_tolerance)

    def rule(rec):
        c = rec["candidates"]
        keys = [selection_key(x["max_violation_mm"] / 1e3, tol, x["goal_err_mm"], x["path_length_rad"],
                              x["accel_sq"]) if x["eligible"] else None for x in c]
        return select_moving_first([x["kind"] for x in c], keys, [x["max_violation_mm"] for x in c], tol * 1e3,
                                   [x["goal_err_mm"] for x in c], [x["reference_deviation"] for x in c])

    plain = _opt(sc, 16).solve(sc.reference, sc.q_now, scene).metrics["multi_seed"]
    assert plain["goal_rows"] is None and all(c["goal_err_mm"] is None for c in plain["candidates"])
    assert rule(plain) == plain["chosen"]
    # stay is the last resort (user, 2026-10-26): a moving candidate resolves here, so the resolved moving candidate
    # nearest the policy reference wins, not stay (which the path-length key would pick: zero joint path)
    assert plain["chosen_kind"] != "stay"
    moving = [c for c in plain["candidates"] if c["eligible"] and c["kind"] != "stay"
              and c["max_violation_mm"] <= tol * 1e3]
    assert plain["candidates"][plain["chosen"]]["reference_deviation"] == min(c["reference_deviation"] for c in moving)
    # a goal handed by a zero-weight anchor (no palm cost: the solves are the same) enters the selection:
    # put it at the palm of the policy plan's rows 5-7 -> the resolved seed nearest the goal wins, not stay
    p = LinkPointFK(sc.robot, sc.layout, 8, "ee_right")(sc.reference, sc.q_now)[0]
    anchor = GraspAnchor("ee_right", p[6], 5, 0.0, 0.0)
    rec = _opt(sc, 16).solve(sc.reference, sc.q_now, scene, grasp_anchor=anchor).metrics["multi_seed"]
    assert rec["goal_rows"] == [5, 6, 7] and rule(rec) == rec["chosen"]
    assert [c["max_violation_mm"] for c in rec["candidates"]] == [c["max_violation_mm"] for c in plain["candidates"]]
    resolved = [c for c in rec["candidates"] if c["eligible"] and c["max_violation_mm"] <= tol * 1e3]
    assert rec["candidates"][rec["chosen"]]["goal_err_mm"] == min(c["goal_err_mm"] for c in resolved)
    assert rec["chosen_kind"] != "stay"


# ------------------------------------------------------------------------------------------------ 5. determinism
def test_same_input_same_result_and_the_index_drives_only_the_random_seeds():
    from AG3S.trajopt.multi_seed import seed_specs

    sc, scene = _stuck()
    a = _opt(sc, 16).solve(sc.reference, sc.q_now, scene, multi_seed_index=211)
    b = _opt(sc, 16).solve(sc.reference, sc.q_now, scene, multi_seed_index=211)
    assert _sha(a.trajectory) == _sha(b.trajectory)
    assert _strip_ms(a.metrics["multi_seed"]) == _strip_ms(b.metrics["multi_seed"])
    assert a.metrics["multi_seed"]["index"] == 211
    # the seeds themselves: another index changes the noise and the random directions only
    opt = _opt(sc, 64)
    geo = opt._seed_geometry(_opt(sc).solve(sc.reference, sc.q_now, scene), sc.reference, sc.q_now, scene, None)
    q0 = sc.q_now[sc.layout.q_indices]
    s1 = opt._build_seeds(seed_specs(64), 211, sc.reference, q0, geo)
    s2 = opt._build_seeds(seed_specs(64), 212, sc.reference, q0, geo)
    s1b = opt._build_seeds(seed_specs(64), 211, sc.reference, q0, geo)
    assert len(s1) == 64 and all(np.array_equal(x[2], y[2]) for x, y in zip(s1, s1b))
    for (k1, d1, Q1), (k2, d2, Q2), spec in zip(s1, s2, seed_specs(64)):
        random = spec.kind == "noise" or (spec.kind == "detour" and spec.weights is None)
        assert np.array_equal(Q1, Q2) != random, (spec, k1)
    # without an index the optimizer counts its own solves (0, 1, ...)
    opt = _opt(sc, 16)
    assert opt.solve(sc.reference, sc.q_now, scene).metrics["multi_seed"]["index"] == 0
    assert opt.solve(sc.reference, sc.q_now, scene).metrics["multi_seed"]["index"] == 1


# ------------------------------------------------------------------------------------------------ 6. seeds
def test_seed_specs_and_construction():
    from AG3S.trajopt import multi_seed as MS
    from AG3S.trajopt.grasp_anchor import LinkPointFK

    kinds = [s.kind for s in MS.seed_specs(16)]
    assert kinds[:2] == ["policy", "stay"] and kinds[2:8] == ["detour"] * 6 and kinds[8:10] == ["noise"] * 2
    assert [(s.direction, s.magnitude) for s in MS.seed_specs(16)[2:8]] == [
        ("away", 0.05), ("away", 0.10), ("e1", 0.05), ("e1", 0.10), ("e2", 0.05), ("e2", 0.10)]
    assert [s.sigma for s in MS.seed_specs(16)[8:10]] == [0.02, 0.05]
    assert [s.direction for s in MS.seed_specs(16)[10:12]] == ["-e1", "-e1"]
    for n in (0, 1, 5, 16, 32, 64):
        assert len(MS.seed_specs(n)) == n
    assert MS.seed_specs(16) == MS.seed_specs(32)[:16] == MS.seed_specs(64)[:16]
    assert sum(s.direction is not None and s.direction.startswith("random") for s in MS.seed_specs(64)) > 0
    # frame
    away = np.array([0.3, -0.8, 0.2]) / np.linalg.norm([0.3, -0.8, 0.2])
    e1, e2 = MS.perpendicular_frame(away)
    M = np.stack([away, e1, e2])
    np.testing.assert_allclose(M @ M.T, np.eye(3), atol=1e-12)
    assert abs(e1[2]) < 1e-12 and e2[2] >= 0.0
    e1, e2 = MS.perpendicular_frame(np.array([0.0, 0.0, 1.0]))                # vertical: still a frame
    np.testing.assert_allclose(np.stack([[0, 0, 1.0], e1, e2]) @ np.stack([[0, 0, 1.0], e1, e2]).T, np.eye(3),
                               atol=1e-12)
    # bump: 0 at both ends, 1 at the peak, smooth
    b = MS.bump(8, 4)
    assert b[0] == 0.0 and b[-1] == 0.0 and b[4] == pytest.approx(1.0) and np.all((b >= 0) & (b <= 1))
    assert MS.bump(8, 0)[1] == pytest.approx(1.0) and MS.bump(8, 7)[6] == pytest.approx(1.0)
    # the detour moves the palm by ~ the magnitude along the direction at the peak row; row 0 / H-1 untouched
    sc, scene = _stuck()
    fk = LinkPointFK(sc.robot, sc.layout, 8, "ee_right")
    p_ref, J = fk(sc.reference, sc.q_now)
    joints = np.flatnonzero(np.abs(J).sum(axis=(0, 1)) > 1e-12)
    d = np.array([0.0, 0.0, 1.0])
    Q = MS.detour_seed(sc.reference, J, d, 0.05, b, joints)
    p = fk(Q, sc.q_now)[0]
    np.testing.assert_array_equal(Q[:, 0], sc.reference[:, 0])
    np.testing.assert_array_equal(Q[:, -1], sc.reference[:, -1])
    moved = p[4] - p_ref[4]
    assert 0.04 < moved @ d < 0.06 and np.linalg.norm(moved - (moved @ d) * d) < 0.01
    others = np.setdiff1d(np.arange(Q.shape[0]), joints)
    np.testing.assert_array_equal(Q[others], sc.reference[others])
    # noise: row 0 fixed, only the chosen joints, the given sigma
    rng = MS.chunk_rng(3, 9)
    Qn = MS.noise_seed(sc.reference, 0.05, rng, joints)
    np.testing.assert_array_equal(Qn[:, 0], sc.reference[:, 0])
    np.testing.assert_array_equal(Qn[others], sc.reference[others])
    assert 0.02 < np.std((Qn - sc.reference)[joints, 1:]) < 0.08
    # the optimizer's seeds: stay = q_now on every row, all inside the position box
    opt = _opt(sc, 16)
    std = _opt(sc).solve(sc.reference, sc.q_now, scene)
    geo = opt._seed_geometry(std, sc.reference, sc.q_now, scene, None)
    assert geo["palm"] == "ee_right" and geo["peak"] == std.metrics["max_violation_pair"]["step"]
    seeds = opt._build_seeds(MS.seed_specs(16), 0, sc.reference, sc.q_now[sc.layout.q_indices], geo)
    box = lambda Q: np.clip(Q, sc.limits.lower[:, None], sc.limits.upper[:, None])   # noqa: E731
    np.testing.assert_array_equal(seeds[0][2], box(sc.reference))
    # (this fixture's q_now is 0.01 rad above one joint's upper bound: the seeds are clipped like the reference)
    np.testing.assert_array_equal(seeds[1][2], box(np.tile(sc.q_now[sc.layout.q_indices][:, None], (1, 8))))
    for _, _, Q0 in seeds:
        assert np.all(Q0 >= sc.limits.lower[:, None]) and np.all(Q0 <= sc.limits.upper[:, None])
    # no direction (nothing violated): the detour slots become noise seeds of the same slot
    geo_none = {**geo, "away": None, "e1": None, "e2": None}
    kinds = [(k, d.get("no_direction")) for k, d, _ in
             opt._build_seeds(MS.seed_specs(8), 0, sc.reference, sc.q_now[sc.layout.q_indices], geo_none)]
    assert kinds[:2] == [("policy", None), ("stay", None)] and kinds[2] == ("noise", "away")


# ------------------------------------------------------------------------------------------------ 7. wiring
def test_config_and_cli(monkeypatch, capsys):
    from AG3S.trajopt import serve_safe

    assert TrajOptConfig().sqp.multi_seed == 0
    assert TrajOptConfig.from_dict({"sqp": {"multi_seed": 32}}).sqp.multi_seed == 32
    for bad in (-1, True, 1.5):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"sqp": {"multi_seed": bad}})
    parse = lambda *a: serve_safe.build_parser().parse_args([*BASE_ARGV, *a])   # noqa: E731
    assert parse().to_multi_seed == 0
    assert "multi_seed" not in serve_safe.sqp_overrides(parse())               # off: no key at all
    assert "multi_seed" not in serve_safe.sqp_overrides(parse("--to-multi-seed", "0"))
    assert serve_safe.sqp_overrides(parse("--to-multi-seed", "16")) == {"multi_seed": 16}
    for extra, needle in ((["--to-multi-seed", "-2"], "--to-multi-seed"),
                          (["--no-safe", "--to-multi-seed", "16"], "--sqp-*")):
        monkeypatch.setattr(sys, "argv", ["serve_safe", *BASE_ARGV, *extra])
        with pytest.raises(SystemExit) as e:
            serve_safe.main()
        assert e.value.code == 2
        assert needle in capsys.readouterr().err


def test_refiner_hands_the_request_index_only_when_on():
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    sc, scene = _stuck()
    for n in (0, 16):
        cfg = sc.config.with_overrides({"horizon": {"plan_horizon": None}, "sqp": {"multi_seed": n}})
        ref = TrajOptChunkRefiner(sc.robot, sc.layout, lambda c: (scene, sc.q_now, True), cfg)
        seen = {}
        real = ref.optimizer.solve

        def spy(*a, **k):
            seen.update(k)
            return real(*a, **k)

        ref.optimizer.solve = spy
        ref.refine(sc.chunk, {"t_step": 42, "previous_physical_chunk": None})
        assert ("multi_seed_index" in seen) == (n > 0)
        if n:
            assert seen["multi_seed_index"] == 42
            assert ref.last_result.metrics["multi_seed"]["index"] == 42
            assert ref.last_result.metrics["multi_seed"]["triggered"] is True


def test_lookahead_window_runs_its_own_multi_seed_once():
    """With look-ahead and a closed gate the window optimizer's solve is the standard one: its multi-seed runs there
    (with the request's index), once; an open gate runs the seeds over the planned horizon."""
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03, sqp={"max_iterations": 1, "min_iterations": 1})
    seen = set()
    for step, gate in ((1, 0.0), (3, 0.0)):        # measured: gate closed (13.4 mm left) / open (20.7 mm left)
        scene = _head_on(sc, step=step)
        cfg = sc.config.with_overrides({"horizon": {"execution_length": 4, "lookahead": "obstacles",
                                                    "lookahead_gate": gate},
                                        "collision": {"backend": "esdf"}, "sqp": {"multi_seed": 8}})
        from AG3S.trajopt.sqp import TrajectoryOptimizer

        opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
        assert opt.window_optimizer is not None and opt.window_optimizer._ms_is_window
        calls = {"outer": 0, "inner": 0}
        for name, o in (("outer", opt), ("inner", opt.window_optimizer)):
            real = o._run_seeds

            def counted(*a, _real=real, _name=name, **k):
                calls[_name] += 1
                return _real(*a, **k)

            o._run_seeds = counted
        res = opt.solve(sc.reference, sc.q_now, scene, multi_seed_index=5)
        rec = res.metrics["multi_seed"]
        g = res.metrics["lookahead"]["gate"]
        assert rec["index"] == 5 and rec["triggered"] is True
        assert res.max_violation <= float(cfg.safety.violation_tolerance)
        assert np.asarray(res.trajectory).shape[1] == 10
        assert calls == ({"outer": 0, "inner": 1} if g == "closed" else {"outer": 1, "inner": 0})
        seen.add(g)
        if g == "closed":
            # an E1 anchor with nothing to apply (no close planned): the standard solve is nested
            # (`_solve_plain_recorded`) — the window optimizer still gets the index and runs its seeds, once
            from AG3S.trajopt.grasp_anchor import GraspAnchor

            calls.update(outer=0, inner=0)
            anchor = GraspAnchor("ee_right", np.zeros(3), None, 0.0, 2000.0)
            res = opt.solve(sc.reference, sc.q_now, scene, multi_seed_index=6, grasp_anchor=anchor)
            assert calls == {"outer": 0, "inner": 1} and res.metrics["multi_seed"]["index"] == 6
            assert res.metrics["multi_seed"]["triggered"] is True
            assert res.metrics["grasp_anchor"]["applied"] is False
    assert seen == {"closed", "open"}


# ------------------------------------------------------------------------------------------------ 8. offline
def test_offline_settings_are_registered_and_valid():
    from AG3S.trajopt.experiments import e1_goal, e2_multi_seed

    S = e2_multi_seed.SETTINGS
    assert set(S) >= {"E0", "E1C", "E2_16", "E2_32", "E2_64", "E1C_E2_16", "E1C_E2_32", "E1C_E2_64"}
    base = TrajOptConfig()
    for n in (16, 32, 64):
        e2 = base.with_overrides(S[f"E2_{n}"])
        e1c = base.with_overrides(S[f"E1C_E2_{n}"])
        assert e2.sqp.multi_seed == n and not e1_goal.grasp_on(e2.cost)
        assert e1c.sqp.multi_seed == n and e1c.cost == base.with_overrides(S["E1C"]).cost
    assert e1_goal.SETTINGS["E1C"] == {"cost": {"w_grasp_tcp": 0.0, "w_grasp_goal": e1_goal.W_E1C}}   # untouched


def test_summarize_counts_the_e0_violated_chunks(tmp_path):
    from AG3S.trajopt.experiments.e2_multi_seed import summarize

    def var(viol, err=None, kind=None, ms=None):
        e = {"max_violation_mm": viol, "solve_ms": 30.0 if ms is None else 30.0 + ms, "status": "feasible"}
        if err is not None:
            e["goal_eval"] = {"result_err_axis_max_mm": err, "result_err_mm": err * 1.5,
                              "reference_err_axis_max_mm": 12.0}
        if kind is not None:
            e["multi_seed"] = {"triggered": True, "chosen_kind": kind, "total_ms": ms}
        return e

    chunks = [{"seq": 1, "variants": {"E0": var(0.0), "E2_16": var(0.0)}},                 # E0 fine: not counted
              {"seq": 2, "variants": {"E0": var(5.0), "E2_16": var(0.0, kind="detour", ms=400.0)}},
              {"seq": 3, "variants": {"E0": var(3.0, 4.0), "E2_16": var(0.0, 8.0, kind="stay", ms=500.0)}},
              {"seq": 4, "variants": {"E0": var(2.0, 4.0), "E2_16": var(0.0, 12.0, kind="detour", ms=600.0)}},
              {"seq": 5, "variants": {"E0": var(2.0), "E2_16": var(1.0, kind="standard", ms=700.0)}}]
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"key": "K", "chunks": chunks}))
    t = summarize([str(path)])
    e0, e2 = t["per_setting"]["E0"], t["per_setting"]["E2_16"]
    assert e0["n_base_violated"] == e2["n_base_violated"] == 4 and e0["n_resolved"] == 0
    assert e2["n_resolved"] == 3 and e2["n_resolved_goal_le_10mm"] == 2 and e2["share_resolved_goal_le_10mm"] == 0.5
    assert e2["n_resolved_goal_le_10mm_euclid"] == 1                               # 8 mm axis = 12 mm Euclid
    assert e2["n_with_goal"] == 2 and e2["n_with_goal_resolved_goal_le_10mm"] == 1
    assert e2["n_triggered"] == 4 and e2["multi_seed_ms_median"] == 550.0
    assert e2["chosen_kind"] == {"detour": 2, "stay": 1, "standard": 1}
    assert e2["chosen_kind_when_resolved"] == {"detour": 2, "stay": 1}
    assert t["per_run"]["K"]["E2_16"] == [4, 2] and t["all_chunks"]["E2_16"]["all_chunks"]["n"] == 5
