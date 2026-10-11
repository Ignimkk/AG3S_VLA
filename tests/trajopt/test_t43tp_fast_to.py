"""T43 TP — TO compute time: every option a flag, default off, off ⇒ bit-identical.

User (2026-10-09): the TO compute time is too long to be usable; keep V11AB's behaviour, bring the TO
p90 to the V8 level. What this file pins:

1. Config / CLI: defaults off, round trip, rejects, the flags reach the config, absent flags change
   nothing.
2. `sqp.fast_eval` is bit-identical — FK memo, ESDF-clearance memo (side effects replayed), the
   CasADi buffer FK, the fused distance + label read, the QP built once per solve — on a scene with
   every row class (field layers with labels, held points, destination, obstacle margin, look-ahead
   gate, B, C), over consecutive solves of one optimizer.
3. The memo lives exactly as long as one solve; cached arrays are read-only.
4. `rebox_problem` equals `build_problem`; `distance_and_label` equals the two separate reads; the
   buffered FK equals `Function.__call__`.
5. QP options: the first-QP warm start at the iterate, the eps schedule, accepting the iterate at
   the iteration cap — each only when asked.
6. `horizon.lookahead_stride`: skipped tail steps carry no rows, the window and the kept steps are
   unchanged.
7. `sqp.warm_start_lookahead`: seeds only from the previous chunk's own look-ahead plan.
8. The offline harness' TP variants are valid configurations.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError

osqp = pytest.importorskip("osqp")

TP_KEYS = {"sqp": {"fast_eval": True, "warm_start_lookahead": True},
           "horizon": {"lookahead_stride": 2},
           "qp": {"warm_start_iterate": True, "eps_after_first": 1e-3, "max_iter": 1000,
                  "accept_max_iter": True}}


# =============================================================================================
# helpers
# =============================================================================================


def _analytic(centre, radius, voxel, half_extent, label=True):
    from AG3S.ag3s.fields.esdf import EsdfField, VoxelGrid

    centre = np.asarray(centre, float)
    grid = VoxelGrid.from_bounds(centre - half_extent, centre + half_extent, voxel)
    d = np.linalg.norm(grid.centres() - centre, axis=-1) - float(radius)
    f = EsdfField(grid=grid, distance_grid=np.clip(d, -0.6, 0.6).astype(np.float32),
                  max_distance=0.6, stats={"unknown_fraction": 0.0})
    if label:
        f.label_grid = np.where(d < 0.06, 0, -1).astype(np.int32)
        f.label_names = ("crate",)
    return f


def _field(sc):
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField

    centre = sc.scene.candidate_pos[0]
    radius = float(sc.scene.candidate_radius[0])
    return CuroboEsdfField((_analytic(centre, radius, 0.04, 1.6), _analytic(centre, radius, 0.02, 0.4)),
                           outside_distance=0.6)


def _rich_scene(sc, held=True):
    """Field-only rows of every class the clearance code distinguishes."""
    n = sc.linearizer.n_spheres
    kw = dict(candidate_active=np.zeros_like(sc.scene.candidate_active),
              esdf=_field(sc), esdf_margin=0.0, destination_label="crate",
              destination_margin=0.002, obstacle_margin=0.01,
              support_planes=np.array([[0.0, 0.0, 1.0, 0.0]]))
    if held:
        pts = np.array([[0.0, 0.0, 0.03], [0.01, 0.0, 0.05], [0.0, 0.01, 0.07]])
        T = np.asarray(sc.robot.link_pose(sc.q_now, "ee_finger_l1"))
        kw.update(attached_points=pts, attached_parent_link="ee_finger_l1",
                  attached_radii=np.array([0.01, 0.012, 0.01]),
                  manipulated_spheres=pts @ T[:3, :3].T + T[:3, 3],
                  manipulated_sphere_radii=np.array([0.01, 0.012, 0.01]),
                  manipulated_link_margin=np.full(n, 0.005),
                  held_capture_spheres=np.array([[*T[:3, 3], 0.03]]))
    return dataclasses.replace(sc.scene, **kw)


def _a_over(gate=0.5, **extra):
    over = {"horizon": {"execution_length": 4, "lookahead": "obstacles", "lookahead_gate": gate},
            "collision": {"backend": "esdf"},
            "cost": {"w_deflection_rate": 20.0},
            "limits": {"servo_accel": "relaxed"},
            "qp": {"eps_abs": 1e-4, "eps_rel": 1e-4}}
    for k, v in extra.items():
        over[k] = {**over.get(k, {}), **v}
    return over


def _results_equal(a, b):
    assert a.trajectory.tobytes() == b.trajectory.tobytes()
    assert a.chunk.tobytes() == b.chunk.tobytes()
    assert a.status is b.status and a.max_violation == b.max_violation
    assert a.cost == b.cost and a.iterations == b.iterations
    for key in ("clearance_m", "max_violation_pair", "candidates", "returned", "qp_iterations",
                "lookahead", "deflection_rate", "servo_accel", "held_rows"):
        assert a.metrics.get(key) == b.metrics.get(key), key


def _run(sc, cfg, scene, n=3):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    out = []
    prev = None
    hist = np.stack([sc.reference[:, 0] - 0.02, sc.reference[:, 0] - 0.01], axis=1)
    for i in range(n):
        ref = sc.reference + 0.004 * i
        out.append(opt.solve(ref, sc.q_now, scene, template=sc.chunk, previous_chunk=prev,
                             deflection_history=np.full(sc.layout.nq_opt, 0.003 * i),
                             command_history=hist))
        prev = out[-1].trajectory
    return opt, out


# =============================================================================================
# 1. config · CLI
# =============================================================================================


def test_defaults_are_off():
    c = TrajOptConfig()
    assert c.sqp.fast_eval is False and c.sqp.warm_start_lookahead is False
    assert c.horizon.lookahead_stride == 1
    assert c.qp.warm_start_iterate is False and c.qp.eps_after_first is None
    assert c.qp.accept_max_iter is False and c.qp.max_iter == 4000


def test_config_round_trip_with_every_new_key():
    base = TrajOptConfig.from_dict(_a_over(horizon={"horizon": 10, "plan_horizon": None}))
    c = base.with_overrides(TP_KEYS)
    assert TrajOptConfig.from_dict(c.to_dict()) == c and c.with_overrides({}) == c
    back = c.with_overrides({"sqp": {"fast_eval": False, "warm_start_lookahead": False},
                             "horizon": {"lookahead_stride": 1},
                             "qp": {"warm_start_iterate": False, "eps_after_first": None,
                                    "max_iter": 4000, "accept_max_iter": False}})
    assert back == base


@pytest.mark.parametrize("bad, match", [
    ({"sqp": {"fast_eval": 1}}, "fast_eval"),
    ({"sqp": {"warm_start_lookahead": "yes"}}, "warm_start_lookahead"),
    ({"horizon": {"lookahead_stride": 0}}, "lookahead_stride"),
    ({"horizon": {"lookahead_stride": 2.0}}, "lookahead_stride"),
    ({"horizon": {"lookahead_stride": True}}, "lookahead_stride"),
    ({"qp": {"warm_start_iterate": 1}}, "warm_start_iterate"),
    ({"qp": {"accept_max_iter": None}}, "accept_max_iter"),
    ({"qp": {"eps_after_first": 0.0}}, "eps_after_first"),
    ({"qp": {"eps_after_first": float("nan")}}, "eps_after_first"),
])
def test_config_rejects(bad, match):
    with pytest.raises((TrajOptConfigError, ValueError), match=match):
        TrajOptConfig.from_dict(bad)


def test_cli_flags_reach_the_config_and_absent_flags_change_nothing():
    from AG3S.trajopt import serve_safe

    base = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    ta = ["--plan-horizon", "16", "--lookahead", "obstacles", "--qp-eps", "1e-4",
          "--w-deflection-rate", "20"]
    plain_args = serve_safe.build_parser().parse_args(base + ta)
    assert serve_safe.tp_overrides(plain_args) == ({}, {}, {})
    plain = serve_safe.trajopt_config_from_args(plain_args)
    tp = serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args(
        base + ta + ["--to-fast-eval", "--qp-warm-iterate", "--qp-eps-after-first", "1e-3",
                     "--qp-max-iter", "1000", "--qp-accept-max-iter", "--lookahead-stride", "2",
                     "--sqp-warm-lookahead"]))
    assert tp == plain.with_overrides(TP_KEYS)
    assert tp.qp.eps_abs == 1e-4                     # --qp-eps is still the first QP's
    for wrong in (["--lookahead-stride", "2"], ["--sqp-warm-lookahead"],
                  ta + ["--lookahead-stride", "0"], ["--qp-eps-after-first", "0"],
                  ["--qp-max-iter", "0"]):
        with pytest.raises(SystemExit):
            serve_safe.tp_overrides(serve_safe.build_parser().parse_args(base + wrong))


# =============================================================================================
# 2-3. fast_eval: bit-identical, memo scope
# =============================================================================================


def test_fast_eval_is_bit_identical_on_the_primitive_scenario():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    _, a = _run(sc, sc.config, sc.scene)
    _, b = _run(sc, sc.config.with_overrides({"sqp": {"fast_eval": True}}), sc.scene)
    for x, y in zip(a, b):
        _results_equal(x, y)


@pytest.mark.parametrize("gate", [0.5, -1.0, None])
@pytest.mark.parametrize("held", [True, False])
def test_fast_eval_is_bit_identical_with_every_row_class(gate, held):
    """Look-ahead with the gate open / closed / absent, B and C on, held points, destination label,
    obstacle margin — three consecutive solves of one optimizer (warm starts carry over)."""
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    scene = _rich_scene(sc, held=held)
    cfg = sc.config.with_overrides(_a_over(gate))
    _, a = _run(sc, cfg, scene)
    _, b = _run(sc, cfg.with_overrides({"sqp": {"fast_eval": True}}), scene)
    for x, y in zip(a, b):
        _results_equal(x, y)
    assert a[0].metrics["lookahead"]["gate"] == {0.5: "open", -1.0: "closed", None: None}[gate]


def test_fast_eval_without_look_ahead_on_the_field_scene():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    scene = _rich_scene(sc)
    cfg = sc.config.with_overrides({"collision": {"backend": "esdf"}})
    _, a = _run(sc, cfg, scene)
    _, b = _run(sc, cfg.with_overrides({"sqp": {"fast_eval": True}}), scene)
    for x, y in zip(a, b):
        _results_equal(x, y)


def test_the_memo_lives_one_solve_and_its_arrays_are_read_only():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    scene = _rich_scene(sc)
    cfg = sc.config.with_overrides(_a_over(None, sqp={"fast_eval": True}))
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    lin = opt.linearizer
    seen = {}
    orig = lin._esdf_clearance_impl

    def spy(centres, s):
        seen["memo"] = lin._memo is not None
        seen["writeable"] = centres.flags.writeable
        return orig(centres, s)

    lin._esdf_clearance_impl = spy
    opt.solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    assert seen == {"memo": True, "writeable": False}
    assert lin._memo is None
    # outside a solve nothing is cached: fresh, writeable arrays every call
    a = lin.sphere_states(sc.reference, sc.q_now)
    b = lin.sphere_states(sc.reference, sc.q_now)
    assert a[0] is not b[0] and a[0].flags.writeable
    # a solve that raises still clears the memo
    def boom(*a, **k):
        assert lin._memo is not None
        raise RuntimeError("boom")

    opt._solve = boom
    with pytest.raises(RuntimeError):
        opt.solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    assert lin._memo is None


def test_memo_hits_replay_the_side_effects():
    """A hit must leave the attributes the original call left (class, look-ahead record, radii)."""
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    scene = _rich_scene(sc)
    lin = sc.linearizer
    lin.lookahead_from = 4
    try:
        lin.set_attached(scene.attached_points, scene.attached_parent_link)
        lin.fast_eval = True
        lin.begin_memo()
        states = lin.sphere_states(sc.reference, sc.q_now)
        assert lin.sphere_states(sc.reference.copy(), sc.q_now) is states       # by value
        first = lin._esdf_clearance(states[0], scene)
        cls = lin._last_obstacle_class
        look = lin._last_lookahead
        radii = lin.query_radii
        lin._last_obstacle_class = None
        lin._last_lookahead = None
        lin.set_attached(scene.attached_points, scene.attached_parent_link)   # resets the radii
        again = lin._esdf_clearance(states[0], scene)
        assert again is first and lin._last_obstacle_class is cls
        assert lin._last_lookahead is look and lin.query_radii is radii
        lin.end_memo()
        lin.fast_eval = False
        plain = lin._esdf_clearance(np.array(states[0]), scene)
        assert plain.tobytes() == first.tobytes()
        assert np.array_equal(lin._last_obstacle_class, cls)
    finally:
        lin.end_memo()
        lin.fast_eval = False
        lin.lookahead_from = None
        lin.set_attached(None, None)


# =============================================================================================
# 4. the pieces
# =============================================================================================


def test_buffered_fk_equals_the_casadi_call():
    from AG3S.trajopt.linearize import CollisionLinearizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    plain = CollisionLinearizer(sc.robot, sc.layout, 10)
    fast = CollisionLinearizer(sc.robot, sc.layout, 10)
    fast.fast_eval = True
    pts = np.array([[0.0, 0.0, 0.03], [0.01, 0.0, 0.05]])
    for lin in (plain, fast):
        lin.set_attached(pts, "ee_finger_l1")
    for k in range(3):
        traj = sc.reference + 0.01 * k
        a, b = plain.sphere_states(traj, sc.q_now), fast.sphere_states(traj, sc.q_now)
        assert a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()


def test_distance_and_label_equals_the_two_reads():
    from AG3S.ag3s.fields.esdf import StaticBox
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    field = _field(sc)
    rng = np.random.default_rng(0)
    pts = sc.scene.candidate_pos[0] + rng.uniform(-0.5, 0.5, (2000, 3))
    d, lab = field.distance_and_label(pts, "crate")
    assert d.tobytes() == field.distance(pts).tobytes()
    assert np.array_equal(lab, field.is_label(pts, "crate")) and lab.any() and not lab.all()
    d2, none = field.distance_and_label(pts, "nothing")
    assert d2.tobytes() == d.tobytes() and not none.any()
    shaped = _field(sc)
    shaped.static_shapes = (StaticBox(center=sc.scene.candidate_pos[0] + 0.3,
                                      half_extents=np.full(3, 0.05)),)
    d, lab = shaped.distance_and_label(pts, "crate")
    assert d.tobytes() == shaped.distance(pts).tobytes()
    assert np.array_equal(lab, shaped.is_label(pts, "crate"))


def test_rebox_problem_equals_build_problem():
    from AG3S.trajopt.problem import build_problem, rebox_problem
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    cfg = sc.config.with_overrides({"cost": {"w_deflection_rate": 20.0},
                                    "limits": {"servo_accel": "relaxed"}})
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    q0 = sc.q_now[sc.layout.q_indices]
    hist = dict(deflection_history=np.full(sc.layout.nq_opt, 0.01),
                command_history=np.stack([q0 - 0.02, q0 - 0.01], axis=1),
                servo_accel_bound=opt.servo_accel_bound)
    env = np.full((sc.layout.nq_opt, 10), 0.05)
    for envelope in (None, env):
        kw = {"anchor_envelope": envelope} if envelope is not None else {}
        first = build_problem(sc.reference, sc.reference, sc.limits, cfg, q_now=q0, n_slack=12,
                              trust_radius=0.15, previous_chunk=sc.reference + 0.01, **kw, **hist)
        it = sc.reference + 0.03
        want = build_problem(sc.reference, it, sc.limits, cfg, q_now=q0, n_slack=12,
                             trust_radius=0.3, previous_chunk=sc.reference + 0.01, **kw, **hist)
        got = rebox_problem(first, sc.limits, it, 0.3, q_now=q0, anchor_envelope=envelope)
        for name in ("P", "A"):
            assert (getattr(got, name) != getattr(want, name)).nnz == 0
        for name in ("q", "l", "u"):
            assert getattr(got, name).tobytes() == getattr(want, name).tobytes()
        assert got.row_blocks == want.row_blocks and got.n_var == want.n_var


# =============================================================================================
# 5. QP options
# =============================================================================================


def _qp_problem():
    from AG3S.trajopt.problem import QpProblem
    import scipy.sparse as sp

    n = 6
    rng = np.random.default_rng(1)
    M = rng.normal(size=(n, n))
    P = sp.csc_matrix(M @ M.T + n * np.eye(n))
    A = sp.csc_matrix(np.vstack([np.eye(n), rng.normal(size=(4, n))]))
    return QpProblem(P=P, q=rng.normal(size=n), A=A, l=np.full(n + 4, -1.0),
                     u=np.full(n + 4, 1.0), n_q=n, n_slack=0, nq=n, horizon=1)


def test_qp_defaults_do_not_touch_the_backend_settings():
    from AG3S.trajopt.config import QpConfig
    from AG3S.trajopt.qp import QpSolver

    s = QpSolver(QpConfig())
    p = _qp_problem()
    calls = []
    s.solve(p)
    real = s._backend.update_settings
    s._backend.update_settings = lambda **kw: (calls.append(kw), real(**kw))
    s.solve(p)
    assert calls == []


def test_qp_eps_override_and_its_return():
    from AG3S.trajopt.config import QpConfig
    from AG3S.trajopt.qp import QpSolver

    s = QpSolver(QpConfig(eps_abs=1e-4, eps_rel=1e-4))
    p = _qp_problem()
    s.solve(p)
    assert s._backend.settings.eps_abs == pytest.approx(1e-4)
    s.solve(p, eps=1e-3)
    assert s._backend.settings.eps_abs == pytest.approx(1e-3)
    s.solve(p)
    assert s._backend.settings.eps_abs == pytest.approx(1e-4)


def test_qp_x0_replaces_the_primal_warm_start():
    from AG3S.trajopt.config import QpConfig
    from AG3S.trajopt.qp import QpSolver

    s = QpSolver(QpConfig())
    p = _qp_problem()
    s.solve(p)
    seen = []
    real = s._backend.warm_start
    s._backend.warm_start = lambda **kw: (seen.append(kw), real(**kw))
    x0 = np.full(p.n_var, 0.25)
    s.solve(p, x0=x0)
    assert np.array_equal(seen[-1]["x"], x0) and seen[-1]["y"] is not None
    s.solve(p)
    assert not np.array_equal(seen[-1]["x"], x0)


def test_qp_accept_max_iter_only_when_asked():
    from AG3S.trajopt.config import QpConfig
    from AG3S.trajopt.qp import QpSolver

    p = _qp_problem()
    strict = QpSolver(QpConfig(max_iter=1, eps_abs=1e-9, eps_rel=1e-9)).solve(p)
    loose = QpSolver(QpConfig(max_iter=1, eps_abs=1e-9, eps_rel=1e-9,
                              accept_max_iter=True)).solve(p)
    assert not strict.success and "maximum iterations" in strict.status
    assert loose.success and loose.status.endswith("(accepted)") and np.isfinite(loose.x).all()


def test_sqp_passes_x0_on_the_first_qp_and_eps_on_the_later_ones():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    for over, want in (({}, [{}, {}, {}]),
                       ({"qp": {"warm_start_iterate": True, "eps_after_first": 1e-3}},
                        [{"x0"}, {"eps"}, {"eps"}])):
        cfg = sc.config.with_overrides({"sqp": {"max_iterations": 3, "min_iterations": 3,
                                                "step_tolerance": 0.0}, **over})
        opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
        calls = []
        real = opt.solver.solve

        def spy(problem, warm_start=True, **kw):
            calls.append(set(kw))
            return real(problem, warm_start=warm_start, **kw)

        opt.solver.solve = spy
        opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
        assert calls == [set(w) for w in want]


# =============================================================================================
# 6. lookahead stride
# =============================================================================================


def test_stride_skips_tail_steps_and_keeps_window_and_kept_steps():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=12, penetration=0.03)
    scene = _rich_scene(sc)
    lin = sc.linearizer
    lin.set_attached(scene.attached_points, scene.attached_parent_link)
    lin.lookahead_from = 4
    try:
        states = lin.sphere_states(sc.reference, sc.q_now)
        full = lin._esdf_clearance(states[0], scene)
        lin.lookahead_stride = 2
        thin = lin._esdf_clearance(states[0], scene)
        kept = [0, 1, 2, 3, 5, 7, 9, 11]
        assert thin[kept].tobytes() == full[kept].tobytes()
        assert np.isinf(thin[[4, 6, 8, 10]]).all()
        assert lin._last_lookahead["tail_steps"] == [5, 7, 9, 11]
        assert lin._last_obstacle_class.shape == full.shape[:2]
        lin.lookahead_stride = 3                       # tail 8 steps: 6, 9 (the last is not on it)
        assert np.isfinite(lin._esdf_clearance(states[0], scene)[[6, 9]]).any()
    finally:
        lin.lookahead_stride = 1
        lin.lookahead_from = None
        lin.set_attached(None, None)


def test_stride_on_the_optimizer_certifies_the_whole_window():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    scene = _rich_scene(sc)
    cfg = sc.config.with_overrides(_a_over(None, horizon={"lookahead_stride": 2}))
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    assert opt.linearizer.lookahead_stride == 2
    res = opt.solve(sc.reference, sc.q_now, scene, template=sc.chunk)
    plain = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config.with_overrides(
        _a_over(None)))
    per = plain.linearizer.step_clearances(res.trajectory, sc.q_now, scene)
    assert res.max_violation == pytest.approx(max(0.0, -per[:4].min()))
    off = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    assert off.linearizer.lookahead_stride == 1


# =============================================================================================
# 7. look-ahead warm start
# =============================================================================================


def test_warm_start_lookahead_seeds_only_from_the_aligned_previous_plan():
    """The seed is the last plan shifted by K, used only when the continuity reference the caller
    hands over starts with exactly that (the previous chunk was this optimizer's own look-ahead
    plan and K steps of it executed)."""
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    scene = _rich_scene(sc)
    k = 4

    def first_iterates(cfg, plan):
        opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
        real = opt._merit_terms
        firsts = []

        def spy(trajectory, *a, **kw):
            if firsts[-1] is None:
                firsts[-1] = np.array(trajectory)
            return real(trajectory, *a, **kw)

        opt._merit_terms = spy
        outs = []
        for gate, cont in plan:
            opt.lookahead_gate = gate
            firsts.append(None)
            prev = cont(outs) if cont is not None else None
            outs.append(opt.solve(sc.reference, sc.q_now, scene, template=sc.chunk,
                                  previous_chunk=prev))
        return opt, outs, firsts

    aligned = lambda outs: outs[-1].trajectory[:, k:]                       # noqa: E731
    stale = lambda outs: outs[-1].trajectory[:, k:] + 1e-3                   # noqa: E731
    on = sc.config.with_overrides(_a_over(0.5, sqp={"warm_start_lookahead": True}))
    opt, outs, firsts = first_iterates(on, [(0.5, None), (0.5, aligned), (0.5, stale),
                                            (-1.0, aligned), (0.5, aligned)])
    assert firsts[0].tobytes() == sc.reference.tobytes()                     # nothing before
    prev = outs[0].trajectory
    shifted = np.concatenate([prev[:, k:], np.tile(prev[:, -1:], (1, k))], axis=1)
    assert np.allclose(firsts[1], 0.5 * (sc.reference + shifted))           # seeded
    assert firsts[2].tobytes() == sc.reference.tobytes()                     # not the same plan
    assert outs[3].metrics["lookahead"]["gate"] == "closed"
    # after a closed gate the window optimizer planned: the outer last plan is stale
    assert firsts[4].tobytes() == sc.reference.tobytes()
    off = sc.config.with_overrides(_a_over(0.5))
    _, _, firsts_off = first_iterates(off, [(0.5, None), (0.5, aligned)])
    assert firsts_off[1].tobytes() == sc.reference.tobytes()                 # default: never


# =============================================================================================
# 8. the offline harness
# =============================================================================================


def test_tp_variants_are_valid_configs():
    from AG3S.trajopt.experiments import ta_offline as TA

    base = TrajOptConfig.from_dict({"collision": {"backend": "esdf"}})
    names = [v for v in TA.VARIANTS if v.startswith("TP")]
    assert {"TPfe", "TPwi", "TPe3n", "TPcap1k", "TPs2", "TPh12", "TPwl2", "TPilp"} <= set(names)
    for v in names:
        cfg = TA.variant_config(base, v, False)
        assert cfg.horizon.lookahead == "obstacles" and cfg.cost.w_deflection_rate == 20.0
        assert cfg.qp.eps_abs == 1e-4 and cfg.sqp.fast_eval is True
    assert TA.variant_server_args("TPilp") == ("--finger-cover-solver", "ilp")
    assert TA.variant_server_args("A16qB20") == ()
    # V11AB = A16qB20, and TPfe differs from it by fast_eval alone
    v11 = TA.variant_config(base, "A16qB20", False)
    assert TA.variant_config(base, "TPfe", False) == v11.with_overrides(
        {"sqp": {"fast_eval": True}})


def test_tp_timing_knows_every_setting():
    from AG3S.trajopt.experiments import tp_timing as TP

    assert TP.SETTINGS["V8"] == {}
    assert TP.SETTINGS["V11AB"]["horizon"]["plan_horizon"] == 16
    assert TP.SETTINGS["V11ABfe"]["sqp"] == {"fast_eval": True}
    with pytest.raises(SystemExit):
        TP.main(["--index", "x", "--key", "y", "--out", "z", "--repeat", "2"])


# =============================================================================================
# 9. GPU field host mirror (`collision.esdf_host_mirror`) — CPU tensors stand in for the GPU
# =============================================================================================


def _device_field(sc, device="cpu"):
    torch = pytest.importorskip("torch")
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField
    from AG3S.ag3s.fields.device_field import layer_from_device

    host = _field(sc)
    layers = tuple(layer_from_device(torch.as_tensor(f.distance_grid, device=device),
                                     f.grid.origin, f.grid.voxel_size,
                                     labels=torch.as_tensor(f.label_grid, device=device),
                                     label_names=f.label_names, outside_distance=f.outside_distance)
                   for f in host.layers)
    return CuroboEsdfField(layers, outside_distance=host.outside_distance), host


@pytest.mark.parametrize("n_points", [200, 3000])
def test_host_mirror_reads_the_same_values(n_points):
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    dev, host = _device_field(sc)
    rng = np.random.default_rng(2)
    pts = sc.scene.candidate_pos[0] + rng.uniform(-0.5, 0.5, (n_points, 3))
    before = (dev.distance(pts), dev.gradient(pts), dev.label(pts))
    assert dev.mirror_to_host() == 2 and all(layer.mirrored for layer in dev.layers)
    after = (dev.distance(pts), dev.gradient(pts), dev.label(pts))
    for a, b in zip(before, after):
        assert a.tobytes() == b.tobytes()
    assert before[0].tobytes() == host.distance(pts).tobytes()
    assert dev.layers[0].n_host_materialisations == 0          # not the whole-grid host path
    dev.layers[0].label_grid = np.zeros(dev.layers[0].grid.shape, np.int32)
    assert dev.layers[0]._mirror_labels is None                 # the mirror follows the grid
    dev.layers[0].drop_mirror()
    assert not dev.layers[0].mirrored
    assert host.mirror_to_host() == 0                           # host layers are left alone


def test_host_mirror_on_the_optimizer_is_bit_identical():
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    dev_a, _ = _device_field(sc)
    dev_b, _ = _device_field(sc)
    scene_a = dataclasses.replace(_rich_scene(sc), esdf=dev_a)
    scene_b = dataclasses.replace(_rich_scene(sc), esdf=dev_b)
    cfg = sc.config.with_overrides(_a_over(0.5, sqp={"fast_eval": True}))
    _, a = _run(sc, cfg, scene_a)
    _, b = _run(sc, cfg.with_overrides({"collision": {"esdf_host_mirror": True}}), scene_b)
    for x, y in zip(a, b):
        _results_equal(x, y)
    assert not any(layer.mirrored for layer in dev_a.layers)
    assert all(layer.mirrored for layer in dev_b.layers)
    from AG3S.trajopt import serve_safe

    base = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    on = serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args(
        base + ["--esdf-host-mirror"]))
    off = serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args(base))
    assert on.collision.esdf_host_mirror and not off.collision.esdf_host_mirror
    assert on.with_overrides({"collision": {"esdf_host_mirror": False}}) == off
