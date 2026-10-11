"""T43 TA — TO path shape: (A) look-ahead horizon, (B) deflection-rate cost, (C) servo acceleration.

User (2026-10-07): the TO should give smooth paths the robot can follow and lift over obstacles
earlier and more gently; every change applies to all scenarios (no pick-and-place phase logic).

What this file pins:

1. **Off is bit-identical** — default config, explicit-off keys and history passed with the flags off
   all give the same QP and the same solve.
2. Config: defaults, validation, `to_dict`/`from_dict` round trip, CLI (`serve_safe`).
3. B: the QP encodes `w (Σ‖ΔD‖² + ‖D0 − D_prev‖²)` exactly (objective agreement); the boundary term
   carries a deflection across the chunk start; a needed deflection is ramped.
4. C: the rows are the second difference of `[u−2, u−1, Q]`; strict / relaxed bounds; the bound binds
   in a solve (incl. the boundary) and is priced in the merit; `a_C` derivation checked against the
   discrete servo model (`ServoModel`, the same Euler scheme as MuJoCo).
5. A: the tail keeps obstacle rows only (target · support · held remnant dropped); only the executed
   window is certified / classified; the tail is recorded; the merit sums window and tail.
6. Wiring: refiner context keys, `SafePolicy._ta_history`, `classify_violations(steps=)`.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.config import TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.problem import (
    build_problem,
    deflection_rate_cost,
    objective,
    servo_accel_overshoot,
    servo_accel_rows,
)
from AG3S.trajopt.types import JointLimits

osqp = pytest.importorskip("osqp")
import scipy.sparse as sp  # noqa: E402

NQ, H = 4, 8


@pytest.fixture
def limits():
    return JointLimits(lower=np.full(NQ, -3.0), upper=np.full(NQ, 3.0),
                       max_step=np.full(NQ, 0.2), max_step_change=np.full(NQ, 0.08),
                       dt=1.0 / 15.0)


def _cfg(**over):
    base = {"horizon": {"horizon": H, "execution_length": 4, "plan_horizon": None}}
    for k, v in over.items():
        base[k] = {**base.get(k, {}), **v}
    return TrajOptConfig.from_dict(base)


def _solve(problem, eps=1e-9):
    solver = osqp.OSQP()
    solver.setup(P=sp.triu(problem.P, format="csc"), q=problem.q, A=problem.A,
                 l=np.clip(problem.l, -1e20, 1e20), u=np.clip(problem.u, -1e20, 1e20),
                 verbose=False, eps_abs=eps, eps_rel=eps, polish=True, max_iter=40000)
    res = solver.solve()
    assert res.info.status in ("solved", "solved inaccurate"), res.info.status
    return problem.split(res.x)[0]


def _ramp(limits):
    return 0.2 * limits.max_step[:, None] * np.arange(H)[None, :]


def _same(a, b):
    for name in ("P", "A"):
        x, y = getattr(a, name), getattr(b, name)
        assert x.shape == y.shape and (x != y).nnz == 0, name
    for name in ("q", "l", "u"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name


# =============================================================================================
# 1. off
# =============================================================================================


def test_defaults_are_off():
    c = TrajOptConfig()
    assert c.limits.servo_accel_weight == 0.01 and c.horizon.lookahead_gate == 0.02
    assert c.horizon.lookahead == "off" and c.horizon.lookahead_from is None
    assert c.cost.w_deflection_rate == 0.0
    assert c.limits.servo_accel == "off"


def test_off_problem_is_bit_identical_even_with_history(limits):
    ref = _ramp(limits)
    it = ref + 0.01
    plain = build_problem(ref, it, limits, _cfg(), q_now=ref[:, 0], n_slack=3)
    explicit = build_problem(
        ref, it, limits,
        _cfg(cost={"w_deflection_rate": 0.0}, limits={"servo_accel": "off"}),
        q_now=ref[:, 0], n_slack=3,
        deflection_history=np.full(NQ, 0.05), command_history=np.zeros((NQ, 2)))
    _same(plain, explicit)
    assert objective(it, ref, _cfg()) == objective(
        it, ref, _cfg(), deflection_history=np.full(NQ, 0.05))


def test_off_solve_is_bit_identical_on_the_rby1_scenario():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    explicit = sc.config.with_overrides({"horizon": {"lookahead": "off"},
                                         "cost": {"w_deflection_rate": 0.0},
                                         "limits": {"servo_accel": "off"}})
    a = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config).solve(
        sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    b = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, explicit).solve(
        sc.reference, sc.q_now, sc.scene, template=sc.chunk,
        deflection_history=np.zeros(sc.layout.nq_opt),
        command_history=np.zeros((sc.layout.nq_opt, 2)))
    assert a.trajectory.tobytes() == b.trajectory.tobytes()
    assert a.max_violation == b.max_violation
    for key in ("lookahead", "servo_accel", "deflection_rate"):
        assert key not in a.metrics and key not in b.metrics


# =============================================================================================
# 2. config · CLI
# =============================================================================================


def test_config_round_trip_with_every_new_key():
    data = {"horizon": {"plan_horizon": 16, "lookahead": "obstacles", "lookahead_gate": 0.03},
            "collision": {"backend": "esdf"},
            "cost": {"w_deflection_rate": 20.0},
            "limits": {"servo_accel": "relaxed", "servo_accel_tolerance": 0.0349,
                       "servo_accel_weight": 2.0}}
    c = TrajOptConfig.from_dict(data)
    back = TrajOptConfig.from_dict(c.to_dict())
    assert back == c and back.to_dict() == c.to_dict()
    assert c.horizon.lookahead_from == 8 and c.horizon.planned == 16
    assert c.horizon.lookahead_gate == 0.03
    assert TrajOptConfig().with_overrides(data) == c
    always = c.with_overrides({"horizon": {"lookahead_gate": None}})
    assert TrajOptConfig.from_dict(always.to_dict()) == always
    assert always.horizon.lookahead_gate is None
    assert TrajOptConfig().horizon.lookahead_gate == 0.02      # read only with lookahead on


@pytest.mark.parametrize("bad, match", [
    ({"horizon": {"lookahead": "obstacles"}}, "needs a tail"),
    ({"horizon": {"plan_horizon": 16, "lookahead": "tail"}}, "lookahead must be one of"),
    ({"cost": {"w_deflection_rate": -1.0}}, "w_deflection_rate"),
    ({"limits": {"servo_accel": "hard"}}, "servo_accel must be one of"),
    ({"limits": {"servo_accel": "strict", "servo_accel_tolerance": 0.0}}, "servo_accel_tolerance"),
    ({"limits": {"servo_accel_weight": -2.0}}, "servo_accel_weight"),
    ({"horizon": {"plan_horizon": 16, "lookahead": "obstacles"},
      "collision": {"backend": "esdf"}, "limits": {"servo_model": True}}, "not combined"),
    ({"horizon": {"plan_horizon": 16, "lookahead": "obstacles"}}, "needs collision.backend"),
    ({"horizon": {"plan_horizon": 16, "lookahead": "obstacles"},
      "collision": {"backend": "both"}}, "needs collision.backend"),
    ({"horizon": {"lookahead_gate": float("nan")}}, "lookahead_gate"),
    ({"horizon": {"lookahead_gate": "20"}}, "lookahead_gate"),
])
def test_config_rejects(bad, match):
    with pytest.raises((TrajOptConfigError, ValueError), match=match):
        TrajOptConfig.from_dict(bad)


def test_cli_flags_reach_the_config():
    from AG3S.trajopt import serve_safe

    base = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    args = serve_safe.build_parser().parse_args(base)
    assert serve_safe.ta_overrides(args) == ({}, {}, {})
    plain = serve_safe.trajopt_config_from_args(args)
    assert plain.horizon.lookahead == "off" and plain.cost.w_deflection_rate == 0.0
    args = serve_safe.build_parser().parse_args(
        base + ["--plan-horizon", "16", "--lookahead", "obstacles", "--lookahead-gate-mm", "30",
                "--w-deflection-rate", "20",
                "--servo-accel", "relaxed", "--servo-accel-tolerance-deg", "2",
                "--servo-accel-weight", "1.5"])
    c = serve_safe.trajopt_config_from_args(args)
    assert c.horizon.planned == 16 and c.horizon.lookahead == "obstacles"
    assert c.horizon.lookahead_gate == pytest.approx(0.03)
    always = serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args(
        base + ["--plan-horizon", "16", "--lookahead", "obstacles", "--lookahead-gate-mm",
                "always"]))
    assert always.horizon.lookahead_gate is None
    gated = serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args(
        base + ["--plan-horizon", "16", "--lookahead", "obstacles"]))
    assert gated.horizon.lookahead_gate == 0.02
    assert plain.qp.eps_abs == 1e-3 and serve_safe.qp_overrides(args) == {}
    tight = serve_safe.trajopt_config_from_args(serve_safe.build_parser().parse_args(
        base + ["--qp-eps", "1e-4"]))
    assert tight.qp.eps_abs == 1e-4 and tight.qp.eps_rel == 1e-4
    assert tight.with_overrides({"qp": {"eps_abs": 1e-3, "eps_rel": 1e-3}}) == plain
    with pytest.raises(SystemExit):
        serve_safe.qp_overrides(serve_safe.build_parser().parse_args(base + ["--qp-eps", "0"]))
    for wrong in (["--lookahead-gate-mm", "30"],
                  ["--plan-horizon", "16", "--lookahead", "obstacles", "--lookahead-gate-mm", "x"]):
        with pytest.raises(SystemExit):
            serve_safe.ta_overrides(serve_safe.build_parser().parse_args(base + wrong))
    assert c.cost.w_deflection_rate == 20.0
    assert c.limits.servo_accel == "relaxed" and c.limits.servo_accel_weight == 1.5
    assert c.limits.servo_accel_tolerance == pytest.approx(np.radians(2.0))
    bad = serve_safe.build_parser().parse_args(base + ["--servo-accel-tolerance-deg", "2"])
    with pytest.raises(SystemExit):
        serve_safe.ta_overrides(bad)


# =============================================================================================
# 3. B — deflection rate
# =============================================================================================


def test_b_qp_encodes_the_deflection_rate_cost(limits):
    rng = np.random.default_rng(7)
    cfg = _cfg(cost={"w_deflection_rate": 13.0})
    ref = rng.normal(0, 0.3, (NQ, H))
    Q = ref + rng.normal(0, 0.05, (NQ, H))
    dprev = rng.normal(0, 0.02, NQ)
    for hist in (None, dprev):
        p = build_problem(ref, Q, limits, cfg, deflection_history=hist)
        z, z0 = p.flatten(Q), p.flatten(ref)
        quad = lambda v: 0.5 * v @ (p.P @ v) + p.q @ v  # noqa: E731
        assert quad(z) - quad(z0) == pytest.approx(
            objective(Q, ref, cfg, deflection_history=hist)
            - objective(ref, ref, cfg, deflection_history=hist), rel=1e-9)
    D = Q - ref
    assert deflection_rate_cost(Q, ref, 2.0, dprev) == pytest.approx(
        2.0 * (np.sum(np.diff(D, axis=1) ** 2) + np.sum((D[:, 0] - dprev) ** 2)))


def test_b_boundary_term_carries_the_deflection_across_the_chunk_start(limits):
    ref = _ramp(limits)
    dprev = np.array([0.05, -0.03, 0.0, 0.02])
    plain = _solve(build_problem(ref, ref, limits, _cfg(), q_now=ref[:, 0]))
    assert np.abs(plain - ref).max() < 1e-5        # nothing to avoid: the reference
    cfg = _cfg(cost={"w_deflection_rate": 20.0, "w_smooth": 0.0})
    Q = _solve(build_problem(ref, ref, limits, cfg, q_now=ref[:, 0] + dprev,
                             deflection_history=dprev))
    D = Q - ref
    # starts near the executed deflection and decays monotonically toward the policy
    assert np.all(np.abs(D[:, 0] - dprev) < 0.25 * np.abs(dprev) + 1e-6)
    mag = np.abs(D[0])
    assert np.all(np.diff(mag) <= 1e-7) and mag[-1] < mag[0]


# =============================================================================================
# 4. C — servo acceleration
# =============================================================================================


def test_c_rows_are_second_differences_including_the_executed_history():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 0.1, (NQ, H))
    Q = rng.normal(0, 0.1, (NQ, H))
    hist = rng.normal(0, 0.1, (NQ, 2))
    a_c = np.full(NQ, 0.01)
    A, c, b = servo_accel_rows(ref, a_c, "strict", None, hist)
    vals = (A @ Q.T.reshape(-1) + c).reshape(-1, NQ).T              # (NQ, H)
    ext = np.concatenate([hist, Q], axis=1)
    assert np.allclose(vals, np.diff(ext, 2, axis=1))
    assert np.allclose(b, 0.01)
    A2, c2, _ = servo_accel_rows(ref, a_c, "strict", None, None)
    assert np.allclose((A2 @ Q.T.reshape(-1) + c2).reshape(-1, NQ).T, np.diff(Q, 2, axis=1))
    _, _, br = servo_accel_rows(ref, a_c, "relaxed", np.full(NQ, 0.08), hist)
    own = np.abs(np.diff(np.concatenate([hist, ref], axis=1), 2, axis=1)).T.reshape(-1)
    assert np.allclose(br, np.minimum(np.maximum(0.01, own), 0.08))


def test_c_strict_bound_binds_at_the_boundary_and_relaxed_leaves_the_policy(limits):
    ref = _ramp(limits)
    v = ref[:, 1] - ref[:, 0]
    # executed history on the same slope but 12 mrad below: the policy's row 0 jumps (Δ² = 0.012)
    hist = np.stack([ref[:, 0] - 2 * v - 0.012, ref[:, 0] - v - 0.012], axis=1)
    a_c = np.full(NQ, 0.004)
    strict = _cfg(limits={"servo_accel": "strict"})
    p = build_problem(ref, ref, limits, strict, q_now=ref[:, 0], command_history=hist,
                      servo_accel_bound=a_c)
    assert "servo_accel" in p.row_blocks and p.n_kinematic_slack > 0
    Q = _solve(p, eps=1e-7)
    acc = np.abs(np.diff(np.concatenate([hist, Q], axis=1), 2, axis=1))
    assert acc.max() <= 0.004 + 5e-4
    assert np.abs(np.diff(np.concatenate([hist, ref], axis=1), 2, axis=1)).max() > 0.011
    relaxed = _cfg(limits={"servo_accel": "relaxed"})
    Qr = _solve(build_problem(ref, ref, limits, relaxed, q_now=ref[:, 0], command_history=hist,
                              servo_accel_bound=a_c), eps=1e-7)
    assert np.abs(Qr - ref).max() < 1e-4             # never sharper than the policy itself


def test_c_overshoot_is_priced_in_the_merit(limits):
    ref = _ramp(limits)
    hist = np.stack([ref[:, 0] + 0.08, ref[:, 0] + 0.04], axis=1)
    cfg = _cfg(limits={"servo_accel": "strict"})
    a_c = np.full(NQ, 0.004)
    over = servo_accel_overshoot(ref, ref, a_c, "strict", limits.max_step_change, hist)
    base = objective(ref, ref, cfg)
    priced = objective(ref, ref, cfg, servo_accel_bound=a_c, command_history=hist,
                       max_step_change=limits.max_step_change)
    assert priced - base == pytest.approx(
        cfg.cost.w_slack * cfg.limits.servo_accel_weight * over.sum())
    assert over.sum() > 0


def test_c_accel_bound_matches_the_discrete_servo():
    """`a_C = ε Δt²/κ`: a parabola command with that second difference leaves the servo's path
    `ε` off the delayed command (fitted delay), in the discrete Euler model the TO evaluates."""
    from AG3S.trajopt.config import DEFAULT_LIMITS_MODEL_XML
    from AG3S.trajopt.servo import ServoModel, ServoParams, servo_params_from_xml

    names = [f"left_arm_{j}" for j in range(7)]
    p = servo_params_from_xml(DEFAULT_LIMITS_MODEL_XML, names, control_hz=15.0)
    kappa = p.shape_coefficient()
    zeta = (p.kv + p.damping) / (2 * np.sqrt(p.kp * p.inertia))
    assert np.allclose(kappa, (2 * zeta ** 2 - 1) / (p.kp / p.inertia))
    eps = 0.0175
    a = p.accel_bound(eps)
    assert np.all((a > 0.005) & (a < 0.02))
    m = ServoModel(p)
    k = np.arange(80)
    U = 0.5 * a[:, None] * k[None, :] ** 2
    ro = m.rollout(np.zeros(7), np.zeros(7), U, samples=list(range(1, p.substeps + 1)))
    t = (k[:, None] + np.arange(1, p.substeps + 1)[None, :] / p.substeps).reshape(-1)
    x = ro["q_samples"].reshape(7, -1)
    sel = t > 20
    for j in range(7):
        y = x[j, sel] - 0.5 * a[j] * t[sel] ** 2
        c1, c0 = np.polyfit(t[sel], y, 1)
        t0 = -c1 / a[j]
        offset = c0 - 0.5 * a[j] * t0 ** 2
        assert offset == pytest.approx(eps, rel=0.10)
    under = ServoParams(kp=np.array([100.0]), kv=np.array([1.0]), damping=np.array([0.0]),
                        inertia=np.array([1.0]), timestep=0.002, substeps=33)
    with pytest.raises(ValueError, match="shape coefficient"):
        under.accel_bound(0.01)


def test_c_on_the_rby1_optimizer_records_and_respects_the_bound():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import empty_scene, make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    cfg = sc.config.with_overrides({"limits": {"servo_accel": "strict",
                                               "servo_accel_tolerance": 0.0175,
                                               "servo_accel_weight": 1.0}})
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    assert opt.servo_accel_bound is not None and opt.servo_accel_bound.shape == (sc.layout.nq_opt,)
    hist = np.stack([sc.reference[:, 0] - 0.06, sc.reference[:, 0] - 0.06], axis=1)
    scene = empty_scene(sc.linearizer)
    scene.plane_active[0] = True        # satisfied floor: not empty, nothing to avoid
    scene.plane_offset[0] = -2.0
    res = opt.solve(sc.reference, sc.q_now, scene, template=sc.chunk, command_history=hist)
    rec = res.metrics["servo_accel"]
    assert rec["mode"] == "strict" and rec["boundary"] is True
    acc = np.abs(np.diff(np.concatenate([hist, res.trajectory], axis=1), 2, axis=1))
    before = np.abs(np.diff(np.concatenate([hist, sc.reference], axis=1), 2, axis=1))
    assert acc.max() < before.max()
    assert rec["overshoot_max_rad"] < before.max() - opt.servo_accel_bound.max()


# =============================================================================================
# 5. A — look-ahead
# =============================================================================================


class _Field:
    """z = 0 table, a bar (sphere) at x = 0.3, a target (sphere) at x = −0.3. Analytic."""

    def __init__(self):
        self.grid = types.SimpleNamespace(voxel_size=0.005)
        self.coverage_grid = types.SimpleNamespace(voxel_size=0.02)
        self.bar = np.array([0.3, 0.0, 0.3])
        self.target = np.array([-0.3, 0.0, 0.05])

    def _parts(self, p):
        return np.stack([p[:, 2], np.linalg.norm(p - self.bar, axis=1) - 0.012,
                         np.linalg.norm(p - self.target, axis=1) - 0.04], axis=1)

    def distance(self, pts):
        return self._parts(np.asarray(pts, np.float64).reshape(-1, 3)).min(axis=1)

    def gradient(self, pts):
        p = np.asarray(pts, np.float64).reshape(-1, 3)
        e = 1e-6
        return np.stack([(self.distance(p + e * ax) - self.distance(p - e * ax)) / (2 * e)
                         for ax in np.eye(3)], axis=1)


def _lin(start):
    from AG3S.trajopt.linearize import CollisionLinearizer

    lin = object.__new__(CollisionLinearizer)
    lin.robot_radii = np.array([0.004])
    lin.query_radii = np.array([0.004, 0.02])
    lin.n_spheres = 1
    lin._attached_local = None
    lin._attached_link = None
    lin.robot_model = types.SimpleNamespace(sphere_link_names=["ee_left"])
    lin.lookahead_from = start
    return lin


def _ascene(f, **extra):
    from AG3S.trajopt.linearize import SceneSnapshot

    target_pts = f.target + 0.04 * np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [-1.0, 0, 0]])
    return SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((1, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=np.array([0.004]), candidate_ids=np.zeros(0, np.int64), esdf=f,
        esdf_margin=0.0, manipulated_points=target_pts,
        support_planes=np.array([[0.0, 0.0, 1.0, 0.0]]), **extra)


def test_a_tail_keeps_obstacle_rows_only_and_the_window_is_untouched():
    f = _Field()
    # rows: step 0 (window) and steps 1–3 (tail); query 0 = robot, 1 = held
    near_bar = f.bar - np.array([0.0, 0.0, 0.012 + 0.004 + 0.003])
    near_table = np.array([0.0, 0.3, 0.004 + 0.002])
    near_target = f.target + np.array([0.04 + 0.004 + 0.003, 0.0, 0.0])
    held_bar = f.bar + np.array([0.0, 0.012 + 0.02 + 0.004, 0.0])
    held_table = np.array([0.0, -0.3, 0.02 + 0.003])
    centres = np.stack([
        [near_target, held_table],          # window: everything stays
        [near_bar, held_bar],               # tail: obstacles stay
        [near_table, held_table],           # tail: support dropped (robot and held)
        [near_target, held_bar],            # tail: target dropped, held obstacle stays
    ])
    scene = _ascene(f)
    off = _lin(None)._esdf_clearance(centres, scene)[..., 0]
    on = _lin(1)._esdf_clearance(centres, scene)[..., 0]
    assert np.all(np.isfinite(off))
    assert on[0].tobytes() == off[0].tobytes()
    assert np.isfinite(on[1]).all() and on[1].tobytes() == off[1].tobytes()
    assert np.isinf(on[2]).all()
    assert np.isinf(on[3, 0]) and on[3, 1] == off[3, 1]


def test_a_tail_drops_the_held_remnant():
    f = _Field()
    remnant_pt = f.bar + np.array([0.0, 0.0, 0.012 + 0.02 + 0.003])
    centres = np.stack([[remnant_pt * 0 + [0, 0.5, 0.3], remnant_pt]] * 2)
    capture = np.array([[*f.bar, 0.02]])           # the bar sits inside the capture volume
    kept = _lin(1)._esdf_clearance(centres, _ascene(f))[..., 0]
    dropped = _lin(1)._esdf_clearance(centres, _ascene(f, held_capture_spheres=capture))[..., 0]
    assert np.isfinite(kept[1, 1]) and np.isinf(dropped[1, 1])


def test_a_worst_row_and_step_clearances_respect_the_window():
    f = _Field()
    centres = np.stack([[np.array([0.0, 0.5, 0.3]), np.array([0.0, -0.5, 0.3])],
                        [f.bar - [0.0, 0.0, 0.010], np.array([0.0, -0.5, 0.3])]])
    lin = _lin(1)
    lin.sphere_states = lambda traj, q: (centres, None)
    scene = _ascene(f)
    per = lin.step_clearances(np.zeros((1, 2)), None, scene)
    assert per[1] < 0 < per[0]
    window, pair = lin.worst_row(np.zeros((1, 2)), None, scene, steps=1)
    full, _ = lin.worst_row(np.zeros((1, 2)), None, scene)
    assert window == pytest.approx(per[0]) and full == pytest.approx(per[1])
    assert pair is None or pair["step"] == 0


def test_a_on_the_rby1_optimizer_certifies_the_window_and_records_the_tail():
    """The obstacle is hit only in the tail: the plan is not VIOLATED for it (look-ahead), the
    record says where; hit in the window it still is."""
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from AG3S.trajopt.types import TrajOptStatus
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    cfg = sc.config.with_overrides({"horizon": {"execution_length": 4, "lookahead": "obstacles",
                                                "lookahead_gate": None},
                                    "collision": {"backend": "esdf"},
                                    "sqp": {"max_iterations": 1, "min_iterations": 1}})
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    assert opt.lookahead_from == 4 and opt.linearizer.lookahead_from == 4
    res = opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    look = res.metrics["lookahead"]
    assert look["from"] == 4 and look["planned"] == 10
    per = opt.linearizer.step_clearances(res.trajectory, sc.q_now, sc.scene)
    assert res.max_violation == pytest.approx(max(0.0, -per[:4].min()))
    assert look["tail_violation_m"] == pytest.approx(max(0.0, -per[4:].min()))
    if per[:4].min() >= -cfg.safety.violation_tolerance:
        assert res.status is not TrajOptStatus.VIOLATED
    # the default (no look-ahead) certifies every planned step
    full = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config.with_overrides(
        {"sqp": {"max_iterations": 1, "min_iterations": 1}})).solve(
        sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    assert "lookahead" not in full.metrics


def test_a_merit_sums_window_and_tail():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    cfg = sc.config.with_overrides({"horizon": {"execution_length": 4, "lookahead": "obstacles"},
                                    "collision": {"backend": "esdf"}})
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)
    per = opt.linearizer.step_clearances(sc.reference, sc.q_now, sc.scene)
    _, violation = opt._merit_terms(sc.reference, sc.reference, None, sc.q_now, sc.scene)
    expect = max(0.0, -per[:4].min()) + max(0.0, -per[4:].min())
    assert violation == pytest.approx(expect)


def test_a_classify_violations_reads_the_window_only():
    from AG3S.trajopt.safe_policy import classify_violations

    f = _Field()
    centres = np.stack([[np.array([0.0, 0.5, 0.3]), np.array([0.0, -0.5, 0.3])],
                        [f.bar - [0.0, 0.0, 0.010], np.array([0.0, -0.5, 0.3])]])
    lin = _lin(None)
    lin.sphere_states = lambda traj, q: (centres, None)
    lin.clearances = lambda traj, q, scene, states=None: (np.zeros((2, 1, 0)),
                                                          np.zeros((2, 1, 0)), None)
    lin.esdf_clearance = lambda traj, q, scene, states=None: lin._esdf_clearance(centres, scene)
    lin._query_name = lambda q: "ee_left" if q == 0 else "held"
    scene = _ascene(f)
    full = classify_violations(lin, np.zeros((1, 2)), None, scene, authorized_links=(),
                               tolerance=1e-4)
    window = classify_violations(lin, np.zeros((1, 2)), None, scene, authorized_links=(),
                                 tolerance=1e-4, steps=1)
    assert full["n_collision"] >= 1 and window["n_collision"] == 0


def test_a_scene_carries_support_planes_for_the_tail():
    from AG3S.trajopt.linearize import scene_from_constraint_set

    plane = types.SimpleNamespace(normal=np.array([0.0, 0.0, 1.0]), offset=0.82)
    cs = types.SimpleNamespace(
        constraints=None, esdf=_Field(), manipulated=None, manipulated_link_margin=None,
        attached=None, destination_label=None, destination_margin=None,
        target_field_exclude=None, target_field_policy="relax", support_surfaces=[plane],
        validity=None)
    cfg_off = TrajOptConfig.from_dict({"collision": {"backend": "esdf",
                                                     "use_support_planes": False}})
    cfg_on = cfg_off.with_overrides({"horizon": {"plan_horizon": 16, "lookahead": "obstacles"}})
    try:
        off = scene_from_constraint_set(cs, np.array([0.004]), cfg_off)
        on = scene_from_constraint_set(cs, np.array([0.004]), cfg_on)
    except (AttributeError, ValueError) as exc:          # pragma: no cover - adapter contract
        pytest.skip(f"constraint-set stub incomplete: {exc}")
    assert off.support_planes is None
    assert on.support_planes is not None and on.support_planes.shape == (1, 4)


# =============================================================================================
# 6. wiring
# =============================================================================================


def test_refiner_passes_the_history_only_when_given():
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    seen = []

    class _Opt:
        def solve(self, reference, q_now, scene, **kw):
            seen.append(sorted(kw))
            from AG3S.trajopt.types import TrajOptResult, TrajOptStatus

            return TrajOptResult(chunk=np.zeros((2, 2)), trajectory=reference,
                                 status=TrajOptStatus.FEASIBLE, iterations=0, solve_time_ms=0.0,
                                 max_violation=0.0, reference_violation=0.0, slack_norm=0.0,
                                 cost=0.0, reference_deviation=0.0, metrics={})

    r = object.__new__(TrajOptChunkRefiner)
    r.config = TrajOptConfig()
    r.layout = types.SimpleNamespace(chunk_to_trajectory=lambda c: np.zeros((2, c.shape[0])),
                                     n_valid=2)
    r.scene_fn = lambda ctx: (object(), np.zeros(2), True)
    r.on_result = None
    r.optimizer = _Opt()
    r._failure_count = 0
    r._previous = None
    r.refine(np.zeros((8, 2)), {"t_step": 1})
    r.refine(np.zeros((8, 2)), {"t_step": 2, "command_history": np.zeros((2, 2)),
                                "deflection_history": np.zeros(2)})
    assert "command_history" not in seen[0] and "deflection_history" not in seen[0]
    assert "command_history" in seen[1] and "deflection_history" in seen[1]


def _policy(cost=0.0, accel="off"):
    from AG3S.trajopt.safe_policy import SafePolicy

    p = object.__new__(SafePolicy)
    p.to_config = TrajOptConfig.from_dict({"cost": {"w_deflection_rate": cost},
                                           "limits": {"servo_accel": accel}})
    p.layout = types.SimpleNamespace(
        nq_opt=2, chunk_to_trajectory=lambda c: np.asarray(c, np.float64)[:, :2].T)
    p._previous_plan = {"seq": 4, "reference": np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
                        "refined": None}
    p._last_ta_history = {}
    return p


def test_safe_policy_history_from_the_execution_feedback():
    fb = {"available": True, "seq": 4, "executed": [True, True, True],
          "applied_arm": [[0.0, 0.0], [0.31, 0.42], [0.55, 0.61]]}
    assert _policy()._ta_history(fb) == {}
    out = _policy(cost=10.0, accel="strict")._ta_history(fb)
    assert np.allclose(out["command_history"], [[0.31, 0.55], [0.42, 0.61]])
    assert np.allclose(out["deflection_history"], [0.05, 0.01])
    hold = dict(fb, executed=[True, True, False])
    p = _policy(cost=10.0)
    assert "deflection_history" not in p._ta_history(hold)
    assert p._last_ta_history["deflection_note"] == "last row was a HOLD"
    other = dict(fb, seq=3)
    assert "deflection_history" not in _policy(cost=10.0)._ta_history(other)
    assert _policy(accel="relaxed")._ta_history({"available": False}) == {}


# =============================================================================================
# 7. C — velocity across the chunk start (lead, 2026-10-07)
# =============================================================================================


def test_c_adds_the_boundary_velocity_row_and_prices_it(limits):
    from AG3S.trajopt.problem import boundary_velocity_overshoot

    ref = _ramp(limits)
    u_last = ref[:, 0] - 0.35                       # the new chunk starts 0.35 rad from the last command
    hist = np.stack([u_last - 0.01, u_last], axis=1)
    cfg = _cfg(limits={"servo_accel": "relaxed"})
    a_c = np.full(NQ, 0.08)
    p = build_problem(ref, ref, limits, cfg, q_now=ref[:, 0], command_history=hist,
                      servo_accel_bound=a_c)
    assert "boundary_velocity" in p.row_blocks
    no_hist = build_problem(ref, ref, limits, cfg, q_now=ref[:, 0], servo_accel_bound=a_c)
    assert "boundary_velocity" not in no_hist.row_blocks
    Q = _solve(p, eps=1e-7)
    assert np.abs(Q[:, 0] - u_last).max() <= limits.max_step.max() + 1e-3
    over = boundary_velocity_overshoot(ref, hist, limits.max_step)
    assert over == pytest.approx(np.full(NQ, 0.35 - 0.2))
    priced = objective(ref, ref, cfg, servo_accel_bound=a_c, command_history=hist,
                       max_step_change=limits.max_step_change, max_step=limits.max_step)
    unpriced = objective(ref, ref, cfg, servo_accel_bound=a_c, command_history=hist,
                         max_step_change=limits.max_step_change)
    assert priced - unpriced == pytest.approx(cfg.cost.w_slack * 10.0 * over.sum())


def test_a_continuity_reads_the_executed_window_only():
    """With look-ahead the continuity term covers rows 0..K-1, as without it: the previous chunk's
    columns past K (a stale prediction of the tail) do not move the plan."""
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    cfg = sc.config.with_overrides({"horizon": {"execution_length": 4, "lookahead": "obstacles",
                                                "lookahead_gate": None},
                                    "collision": {"backend": "esdf"}})
    prev = sc.reference + 0.02
    other = prev.copy()
    other[:, 4:] += 0.2
    a = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg).solve(
        sc.reference, sc.q_now, sc.scene, template=sc.chunk, previous_chunk=prev)
    b = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg).solve(
        sc.reference, sc.q_now, sc.scene, template=sc.chunk, previous_chunk=other)
    assert a.trajectory.tobytes() == b.trajectory.tobytes()


# =============================================================================================
# 8. A — the gate (TA2)
# =============================================================================================


def _a_cfg(sc, gate, **extra):
    over = {"horizon": {"execution_length": 4, "lookahead": "obstacles", "lookahead_gate": gate},
            "collision": {"backend": "esdf"}}
    for k, v in extra.items():
        over[k] = {**over.get(k, {}), **v}
    return sc.config.with_overrides(over)


def _window_cfg(sc, **extra):
    over = {"horizon": {"horizon": 4, "execution_length": 4, "plan_horizon": None}}
    for k, v in extra.items():
        over[k] = {**over.get(k, {}), **v}
    return sc.config.with_overrides(over)


def test_a_gate_closed_is_the_window_solve_bit_identical():
    """Nothing within the gate in the tail: the chunk is the `execution_length` solve, byte for
    byte (incl. B's and C's history), and the reference's tail comes back untouched."""
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    on = {"cost": {"w_deflection_rate": 20.0}, "limits": {"servo_accel": "relaxed"}}
    a_opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, _a_cfg(sc, -1.0, **on))
    w_opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, _window_cfg(sc, **on))
    assert a_opt.window_optimizer is not None and a_opt.window_optimizer.horizon == 4
    prev = sc.reference + 0.01
    hist = np.stack([sc.reference[:, 0] - 0.02, sc.reference[:, 0] - 0.01], axis=1)
    kw = dict(deflection_history=np.full(sc.layout.nq_opt, 0.01), command_history=hist)
    a = a_opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk, previous_chunk=prev, **kw)
    w = w_opt.solve(sc.reference[:, :4], sc.q_now, sc.scene, template=sc.chunk,
                    previous_chunk=prev[:, :4], **kw)
    assert a.trajectory.shape == sc.reference.shape
    assert a.trajectory[:, :4].tobytes() == w.trajectory.tobytes()
    assert a.trajectory[:, 4:].tobytes() == sc.reference[:, 4:].tobytes()
    assert a.chunk.tobytes() == w.chunk.tobytes()
    assert a.max_violation == w.max_violation and a.status is w.status
    look = a.metrics["lookahead"]
    assert look["gate"] == "closed" and look["gate_m"] == -1.0 and look["from"] == 4
    assert look["tail_reference_clearance_m"] is not None
    assert "servo_accel" in a.metrics and "deflection_rate" in a.metrics


def test_a_gate_opens_on_an_obstacle_in_the_tail():
    """The near-miss sits at step 5 (tail when K = 4): a gate above its clearance opens."""
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, _a_cfg(sc, 0.02))
    gate = opt._lookahead_gate(sc.reference, sc.q_now, sc.scene)
    assert gate["open"] and gate["tail_reference_clearance_m"] < 0.0
    assert 4 <= gate["tail_reference_worst_step"] < 10
    res = opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    look = res.metrics["lookahead"]
    assert look["gate"] == "open" and look["gate_m"] == 0.02
    assert look["tail_reference_clearance_m"] == pytest.approx(gate["tail_reference_clearance_m"])
    assert res.trajectory.shape == sc.reference.shape
    assert np.abs(res.trajectory[:, 4:] - sc.reference[:, 4:]).max() > 0.0   # the tail is planned
    closed = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, _a_cfg(sc, -1.0))
    assert not closed._lookahead_gate(sc.reference, sc.q_now, sc.scene)["open"]
    ungated = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, _a_cfg(sc, None))
    assert ungated.window_optimizer is None
    assert ungated.solve(sc.reference, sc.q_now, sc.scene,
                         template=sc.chunk).metrics["lookahead"]["gate"] is None


def test_a_tail_drops_support_plane_rows():
    from AG3S.trajopt.sqp import TrajectoryOptimizer
    from tests.trajopt.fixtures import make_scenario

    sc = make_scenario(horizon=10, penetration=0.03)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, _a_cfg(sc, None))
    centres, _ = opt.linearizer.sphere_states(sc.reference, sc.q_now)
    assert sc.scene.plane_active.any()
    _, plane_on, _ = opt.linearizer._clearances_from(centres, sc.scene)
    opt.linearizer.lookahead_from = None
    _, plane_off, _ = opt.linearizer._clearances_from(centres, sc.scene)
    assert plane_on[:4].tobytes() == plane_off[:4].tobytes()
    assert np.isfinite(plane_off[4:][..., sc.scene.plane_active]).all()
    assert np.isinf(plane_on[4:]).all()


def test_a_safe_policy_hands_the_capture_volume_to_the_tail():
    from AG3S.trajopt.safe_policy import SafePolicy

    p = object.__new__(SafePolicy)
    p.ag3s = types.SimpleNamespace(held_capture_spheres=np.ones((1, 4)))
    p.to_config = TrajOptConfig()
    assert p._held_capture_for_scene() is None
    p.to_config = TrajOptConfig.from_dict({"horizon": {"plan_horizon": 16,
                                                       "lookahead": "obstacles"},
                                           "collision": {"backend": "esdf"}})
    assert p._held_capture_for_scene() is not None
