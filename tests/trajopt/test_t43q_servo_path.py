"""T43 Q — servo model (`limits.servo_model`) and continuous collision rows (`reduction.sweep_check`).

T41 b found the TO checks only the commanded waypoints while the position servo lags: after a HOLD the
plan's step 0 sat at the anchor bound (10.8°) and the joint moved 2.0–2.6° in that row. This file fixes:

1. **The servo model is MuJoCo's own servo.** Constants are read from the compiled MJCF (`kp`, `kv` from
   `dampratio`, joint damping, armature + inertia diagonal); on a single hinge the discrete model equals
   `mj_step` to round-off; the affine path map equals the rollout exactly.
2. **The reachable set is the servo's, not a clip.** From rest the first row moves ≈ 0.22 × the command
   step — 2.1–2.5° for a 10.8° command (≈ the `a_max·dt²` T41 b saw), **4.1–5.0° for 21.6°** (the escape the
   rest-start envelope blocked is reachable). The observer recovers velocity and constant torque from
   the applied rows exactly when the model is exact.
3. **Off is unchanged** — no path checker, no keys, `append_collision_rows` as before; config / CLI add
   nothing unless asked.
4. **The rows** — interpolation and servo configurations, the sweep bound (`≤ sweep_max_m` between checked
   configurations, measured), the floor (`min(0, h(q_now))`), gradients against finite differences, a fixed
   QP sparsity, and that each mode sees what the waypoints miss (a thin obstacle between two waypoints;
   an obstacle only the robot's momentum reaches).

Records (T40 G / T41 a E3b) are in `test_t43q_replay.py`; MuJoCo servo steps from replayed states in
`tests/sim/test_t43q_servo_sim.py`.
"""

from __future__ import annotations

import dataclasses
import types

import mujoco
import numpy as np
import pytest

from AG3S.trajopt.config import DEFAULT_LIMITS_MODEL_XML, TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.servo import (ServoModel, ServoObserver, ServoParams, ServoState,
                                     servo_params_from_xml)
from tests.trajopt.fixtures import Q_HOME, make_scenario, small_config

DEG = np.pi / 180.0
ARM = [f"left_arm_{i}" for i in range(7)] + [f"right_arm_{i}" for i in range(7)]

HINGE_XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body>
      <joint name="j" type="hinge" axis="0 0 1" armature="10" damping="30"/>
      <geom type="sphere" size="0.05" mass="1" pos="0.3 0 0"/>
    </body>
  </worldbody>
  <actuator><position name="a" joint="j" kp="2000" dampratio="1"/></actuator>
</mujoco>
"""


@pytest.fixture(scope="module")
def rby1_servo():
    return ServoModel(servo_params_from_xml(DEFAULT_LIMITS_MODEL_XML, ARM, control_hz=15.0))


# =============================================================================================
# 1. the model is MuJoCo's servo
# =============================================================================================


def test_rby1_servo_constants_come_from_the_compiled_mjcf(rby1_servo):
    p = rby1_servo.params
    assert p.timestep == 0.002 and p.substeps == 33                      # pi05_infer: round(1/(15·0.002))
    assert set(np.round(p.kp)) <= {1500.0, 2000.0}                        # `med` 2000, default 1500
    assert np.all((p.kv > 240) & (p.kv < 310))                            # dampratio=1 resolved at compile
    np.testing.assert_allclose(p.damping, 30.0)
    assert np.all((p.inertia > 10.0) & (p.inertia < 12.0))                # armature 10 dominates
    s = p.summary()
    assert all(1.05 < z < 1.15 for z in s["damping_ratio"])
    assert all(12.0 < w < 14.5 for w in s["natural_frequency_rad_s"])
    assert s["source"]["derived_from"] == "compiled MuJoCo model"


def test_a_model_that_is_not_a_position_servo_is_refused():
    xml = HINGE_XML.replace('<position name="a" joint="j" kp="2000" dampratio="1"/>',
                            '<motor name="a" joint="j"/>')
    with pytest.raises(ValueError, match="not a position servo"):
        ServoParams.from_mj_model(mujoco.MjModel.from_xml_string(xml), ["j"], control_hz=15.0)
    with pytest.raises(ValueError, match="not in the MuJoCo model"):
        ServoParams.from_mj_model(mujoco.MjModel.from_xml_string(HINGE_XML), ["nope"],
                                  control_hz=15.0)


def test_on_a_single_hinge_the_model_is_mj_step_to_round_off():
    m = mujoco.MjModel.from_xml_string(HINGE_XML)
    d = mujoco.MjData(m)
    model = ServoModel(ServoParams.from_mj_model(m, ["j"], control_hz=15.0))
    d.qpos[0], d.qvel[0] = 0.1, -0.4
    commands = np.array([[0.5, 0.5, -0.2, 0.0]])
    pred = model.rollout([0.1], [-0.4], commands, samples=list(range(1, 34)))
    got = []
    for k in range(commands.shape[1]):
        d.ctrl[0] = commands[0, k]
        row = []
        for _ in range(33):
            mujoco.mj_step(m, d)
            row.append(d.qpos[0])
        got.append(row)
    np.testing.assert_allclose(pred["q_samples"][0], np.asarray(got), atol=1e-10)
    np.testing.assert_allclose(pred["v"][0, -1], d.qvel[0], atol=1e-9)


def test_the_affine_path_is_the_rollout_exactly(rby1_servo):
    rng = np.random.default_rng(0)
    S = rby1_servo.substeps
    W, Fq, Eq = rby1_servo.affine_path(8, list(range(1, S + 1)))
    for _ in range(3):
        q0, v0, tau = rng.normal(size=14) * 0.3, rng.normal(size=14) * 0.5, rng.normal(size=14) * 10
        Q = q0[:, None] + rng.normal(size=(14, 8)) * 0.2
        X = (np.einsum("ncj,nj->nc", Fq, np.stack([q0, v0], 1)) + Eq * tau[:, None]
             + np.einsum("nci,ni->nc", W, Q))
        roll = rby1_servo.rollout(q0, v0, Q, tau, samples=list(range(1, S + 1)))["q_samples"]
        np.testing.assert_allclose(X.reshape(14, 8, S), roll, atol=1e-12)
    # causal: a row's state never depends on a later command
    for k in range(8):
        assert np.all(W[:, k * S:(k + 1) * S, k + 1:] == 0.0)


# =============================================================================================
# 2. the reachable set is the servo's
# =============================================================================================


def test_reach_from_rest_scales_with_the_command_not_a_blanket_clip(rby1_servo):
    q0 = np.zeros(14)
    first = {}
    for step_deg in (10.8, 21.6):
        Q = np.full((14, 8), step_deg * DEG)
        first[step_deg] = np.degrees(rby1_servo.rollout(q0, np.zeros(14), Q)["q"][:, 0])
    # T41 b measured 2.03–2.59° in the first row after a 10.8° command (C1807 row 217)
    assert np.all((first[10.8] > 2.0) & (first[10.8] < 2.6))
    # the 21.6° escape (T41 b t=216, margin 10 mm) reaches twice as far — beyond the rest-start
    # envelope's first step a_max·dt² = 2.29°, which would have clipped it
    np.testing.assert_allclose(first[21.6], 2.0 * first[10.8], rtol=1e-12)
    assert np.all(first[21.6] > 4.0) and np.all(first[21.6] > 1.75 * 2.292)


def test_moving_robot_keeps_going_the_way_it_was_going(rby1_servo):
    """Commanding the current position to a moving arm does not stop it at q_now: it overshoots."""
    v0 = np.full(14, 0.8)
    X = rby1_servo.rollout(np.zeros(14), v0, np.zeros((14, 8)))["q"]
    assert np.all(X[:, 0] > 0.01)                          # > 0.57° past q_now after one row
    assert np.all(np.abs(X[:, -1]) < np.abs(X[:, 0]))      # and comes back


def test_observer_recovers_velocity_and_torque_from_the_applied_rows(rby1_servo):
    rng = np.random.default_rng(1)
    q_prev, v_prev = rng.normal(size=14) * 0.2, np.zeros(14)
    tau = rng.normal(size=14) * 15.0
    applied = q_prev[:, None] + np.cumsum(rng.normal(size=(14, 8)) * 0.02, axis=1)
    truth = rby1_servo.rollout(q_prev, v_prev, applied, tau)
    obs = ServoObserver(rby1_servo)
    first = obs.update(q_prev, {"available": False, "reason": "first chunk"}, 1)
    assert first.source.startswith("rest") and np.all(first.v == 0)
    fb = {"available": True, "seq": 1, "applied_arm": applied.T}
    st = obs.update(truth["q"][:, -1], fb, 2)
    assert st.source.startswith("propagated")
    np.testing.assert_allclose(st.v, truth["v"][:, -1], atol=1e-9)
    np.testing.assert_allclose(st.tau, tau, atol=1e-6)


@pytest.mark.parametrize("fb, why", [
    ({"available": False, "reason": "client predates T18"}, "no exec_feedback"),
    ({"available": True, "seq": 7, "applied_arm": np.zeros((8, 14))}, "is for seq 7"),
    ({"available": True, "seq": 1, "applied_arm": np.zeros((8, 3))}, "applied_arm has shape"),
])
def test_observer_without_usable_facts_says_rest_and_why(rby1_servo, fb, why):
    obs = ServoObserver(rby1_servo)
    obs.update(np.zeros(14), None, 1)
    st = obs.update(np.ones(14) * 0.1, fb, 2)
    assert st.source.startswith("rest") and why in st.source and np.all(st.v == 0)
    assert st.record()["source"] == st.source


# =============================================================================================
# 3. off is unchanged
# =============================================================================================


def test_config_defaults_are_off_and_validated():
    cfg = TrajOptConfig()
    assert cfg.limits.servo_model is False and cfg.reduction.sweep_check is False
    assert cfg.reduction.sweep_max_m == 0.012 and cfg.reduction.sweep_max_samples == 16
    for bad in ({"sweep_max_m": 0.0}, {"sweep_max_m": float("inf")}, {"sweep_max_samples": 0},
                {"sweep_rows_per_step": 0}):
        with pytest.raises(TrajOptConfigError):
            TrajOptConfig.from_dict({"reduction": bad})


def test_off_optimizer_has_no_path_rows_and_no_new_keys():
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    sc = make_scenario(horizon=6)
    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    assert opt.path is None and opt.servo is None and opt.observer is None
    a = opt.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk)
    opt2 = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    b = opt2.solve(sc.reference, sc.q_now, sc.scene, template=sc.chunk,
                   servo_feedback={"feedback": {"available": False}, "seq": 1})
    assert a.chunk.tobytes() == b.chunk.tobytes()           # the argument is ignored when off
    for r in (a, b):
        assert "path_check" not in r.metrics and "initial_source" not in r.metrics
        assert "path" not in r.metrics["timing_ms"]


def test_append_collision_rows_default_is_unchanged():
    from AG3S.trajopt.linearize import append_collision_rows
    from AG3S.trajopt.problem import build_problem

    sc = make_scenario(horizon=6)
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    opt = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    rows = opt.linearizer.linearize(sc.reference, sc.q_now, sc.scene, sc.config)
    p = build_problem(sc.reference, sc.reference, sc.limits, sc.config,
                      q_now=sc.q_now[sc.layout.q_indices], n_slack=opt.block.n_rows)
    a = append_collision_rows(p, rows, sc.reference, opt.block, 5e-3)
    b = append_collision_rows(p, rows, sc.reference, opt.block, 5e-3, extra_slack=0)
    assert (a.A != b.A).nnz == 0 and a.l.tobytes() == b.l.tobytes()
    with pytest.raises(ValueError):
        append_collision_rows(p, rows, sc.reference, opt.block, 5e-3, extra_slack=3)


def test_linearize_row_floor_none_is_unchanged_and_a_floor_shifts_values():
    sc = make_scenario(horizon=6)
    lin = sc.linearizer
    a = lin.linearize(sc.reference, sc.q_now, sc.scene, sc.config)
    b = lin.linearize(sc.reference, sc.q_now, sc.scene, sc.config, row_floor=None)
    assert a.value.tobytes() == b.value.tobytes() and a.gradient.tobytes() == b.gradient.tobytes()
    width = lin.flat_clearances(lin.sphere_states(sc.reference, sc.q_now)[0], sc.scene).shape[1]
    c = lin.linearize(sc.reference, sc.q_now, sc.scene, sc.config, row_floor=np.full(width, -0.01))
    np.testing.assert_allclose(c.value[c.used], a.value[a.used] + 0.01)


BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def test_cli_default_adds_no_key_and_the_flags_set_the_config():
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert serve_safe.path_overrides(args) == ({}, {})
    cfg = serve_safe.trajopt_config_from_args(args)
    assert cfg.limits.servo_model is False and cfg.reduction.sweep_check is False
    assert cfg.reduction == TrajOptConfig().reduction
    assert "servo_model" not in serve_safe.joint_limits_meta(cfg.limits)
    args = serve_safe.build_parser().parse_args(
        BASE_ARGV + ["--servo-model", "--sweep-check", "--sweep-max-mm", "8",
                     "--sweep-max-samples", "10", "--sweep-rows-per-step", "16"])
    cfg = serve_safe.trajopt_config_from_args(args)
    assert cfg.limits.servo_model is True and cfg.reduction.sweep_check is True
    assert cfg.reduction.sweep_max_m == pytest.approx(0.008)
    assert cfg.reduction.sweep_max_samples == 10 and cfg.reduction.sweep_rows_per_step == 16
    assert serve_safe.joint_limits_meta(cfg.limits)["servo_model"] is True


@pytest.mark.parametrize("extra", [["--sweep-max-mm", "8"], ["--sweep-max-samples", "4"],
                                   ["--sweep-rows-per-step", "8"],
                                   ["--sweep-check", "--sweep-max-mm", "0"]])
def test_cli_rejects_meaningless_combinations(extra):
    from AG3S.trajopt import serve_safe

    args = serve_safe.build_parser().parse_args(BASE_ARGV + extra)
    with pytest.raises(SystemExit):
        serve_safe.trajopt_config_from_args(args)


class _SpyOptimizer:
    def __init__(self, inner):
        self.inner, self.kwargs = inner, []

    def solve(self, *args, **kwargs):
        self.kwargs.append(dict(kwargs))
        return self.inner.solve(*args, **kwargs)

    def reset(self):
        self.inner.reset()


def test_the_refiner_passes_servo_feedback_only_when_the_context_has_it():
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    sc = make_scenario(horizon=6)
    ref = TrajOptChunkRefiner(sc.robot, sc.layout, lambda _c: (sc.scene, sc.q_now, True), sc.config)
    spy = ref.optimizer = _SpyOptimizer(ref.optimizer)
    ref.refine(sc.chunk, {"t_step": 1})
    ref.refine(sc.chunk, {"t_step": 2, "servo_feedback": {"feedback": {}, "seq": 2}})
    assert "servo_feedback" not in spy.kwargs[0]
    assert spy.kwargs[1]["servo_feedback"] == {"feedback": {}, "seq": 2}


def test_safe_policy_puts_servo_feedback_in_the_context_only_when_on():
    from tests.trajopt.test_exec_feedback import _feedback, _obs, _safe_policy, _spy

    H, DOF = 50, 16
    on = _safe_policy(np.full((H, DOF), 0.2),
                      to_config=TrajOptConfig.from_dict({"limits": {"servo_model": True}}))
    seen = _spy(on)
    on.infer(_obs(1))
    on.infer(_obs(2, _feedback(seq=1)))
    assert seen[0]["servo_feedback"]["seq"] == 1
    assert seen[1]["servo_feedback"]["feedback"]["seq"] == 1
    assert seen[1]["servo_feedback"]["seq"] == 2
    off = _safe_policy(np.full((H, DOF), 0.2))
    seen = _spy(off)
    off.infer(_obs(1))
    off.infer(_obs(2, _feedback(seq=1)))
    assert all("servo_feedback" not in c for c in seen)


# =============================================================================================
# 4. the rows
# =============================================================================================


def _opt(sc, **sections):
    from AG3S.trajopt.sqp import TrajectoryOptimizer

    return TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config.with_overrides(sections))


def test_interpolation_configs_cut_each_row_to_the_sweep_bound():
    sc = make_scenario(horizon=6)
    opt = _opt(sc, reduction={"sweep_check": True})
    assert opt.path.mode == "interp" and opt.servo is None
    opt.path.begin(sc.q_now, sc.scene)
    cfgs = opt.path.configs(sc.reference)
    assert np.all(cfgs.pieces == np.maximum(1, np.ceil(cfgs.path_m / 0.012 - 1e-9)))
    for c in range(cfgs.n):
        k, tau = int(cfgs.segment[c]), float(cfgs.where[c])
        assert 0.0 < tau < 1.0
        if k:
            np.testing.assert_allclose(cfgs.W[:, c, k - 1], 1.0 - tau)
        np.testing.assert_allclose(cfgs.W[:, c, k], tau)
        assert np.count_nonzero(cfgs.W[:, c, :]) == (2 if k else 1) * cfgs.W.shape[0]
    worst, record = opt.path.violation(sc.reference, sc.scene, record=True)
    assert record["max_sweep_mm"] <= 12.0                         # measured on the checked chain
    assert record["cap_bound_steps"] == []


def test_servo_sweep_keeps_the_measured_spacing_within_the_bound():
    sc = make_scenario(horizon=6)
    opt = _opt(sc, limits={"servo_model": True}, reduction={"sweep_check": True})
    st = opt.observer.update(sc.q_now[sc.layout.q_indices], None, 1)
    opt.path.begin(sc.q_now, sc.scene, st)
    cfgs = opt.path.configs(sc.reference)
    assert cfgs.pieces.max() > 1 and np.all(cfgs.where[cfgs.where > 0] <= opt.servo.substeps)
    _, record = opt.path.violation(sc.reference, sc.scene, record=True)
    # equal path-length cuts, rounded to the 2 ms sub-step: at most one sub-step's motion over
    assert record["max_sweep_mm"] <= 12.0 + 3.0 and record["cap_bound_steps"] == []


def test_the_cap_binds_and_says_so():
    sc = make_scenario(horizon=6)
    opt = _opt(sc, reduction={"sweep_check": True, "sweep_max_samples": 2})
    opt.path.begin(sc.q_now, sc.scene)
    _, record = opt.path.violation(sc.reference, sc.scene, record=True)
    assert record["cap_bound_steps"] and max(record["pieces_per_step"]) == 2
    assert record["max_sweep_mm"] > 12.0


def test_servo_configs_are_the_predicted_states_and_touch_every_earlier_command():
    sc = make_scenario(horizon=6)
    opt = _opt(sc, limits={"servo_model": True})
    st = opt.observer.update(sc.q_now[sc.layout.q_indices], None, 1)
    opt.path.begin(sc.q_now, sc.scene, st)
    cfgs = opt.path.configs(sc.reference)
    assert cfgs.n == 6 and np.all(cfgs.where == opt.servo.substeps)      # row ends only (no sweep)
    X = opt.servo.rollout(sc.q_now[sc.layout.q_indices], st.v, sc.reference, st.tau)["q"]
    np.testing.assert_allclose(cfgs.full[sc.layout.q_indices], X, atol=1e-12)
    for c in range(cfgs.n):
        k = int(cfgs.segment[c])
        assert np.all(cfgs.W[:, c, : k + 1] > 0) and np.all(cfgs.W[:, c, k + 1:] == 0)
    assert [opt.path.block.pattern(k) for k in range(3)] == [[0], [0, 1], [0, 1, 2]]


def test_path_row_gradients_match_finite_differences():
    sc = make_scenario(horizon=6)
    for sections in ({"reduction": {"sweep_check": True}}, {"limits": {"servo_model": True}}):
        opt = _opt(sc, **sections)
        st = None if opt.observer is None else opt.observer.update(
            sc.q_now[sc.layout.q_indices], None, 1)
        opt.path.begin(sc.q_now, sc.scene, st)
        rows = opt.path.linearize(sc.reference, sc.scene)
        assert rows.used.any()
        cfgs = opt.path.configs(sc.reference)
        k, r = np.argwhere(rows.used)[0]
        c, eps = int(rows.config[k, r]), 1e-6
        rng = np.random.default_rng(3)
        dQ = rng.normal(size=sc.reference.shape) * eps
        moved = opt.path.configs(sc.reference + dQ)
        assert moved.n == cfgs.n
        cen0, _ = opt.path._states(cfgs.full[:, c:c + 1], None if cfgs.params is None
                                   else cfgs.params[:, c:c + 1], with_jac=False)
        cen1, _ = opt.path._states(moved.full[:, c:c + 1], None if moved.params is None
                                   else moved.params[:, c:c + 1], with_jac=False)
        h0 = opt.linearizer.flat_clearances(cen0, sc.scene)[0] - opt.path._floor
        h1 = opt.linearizer.flat_clearances(cen1, sc.scene)[0] - opt.path._floor
        col = int(np.argmin(np.abs(h0 - rows.value[k, r])))
        predicted = np.einsum("in,ni->", rows.gradient[k, r], dQ)
        assert abs((h1[col] - h0[col]) - predicted) <= 1e-3 * eps + 1e-12


def test_the_path_block_sparsity_does_not_depend_on_the_scene():
    from AG3S.trajopt.sweep import PathBlock, PathRows

    H, R, nq = 4, 3, 2
    for mode in ("interp", "servo"):
        block = PathBlock(H, R, nq, mode)
        used = np.zeros((H, R), bool)
        empty = PathRows(np.full((H, R), np.inf), np.zeros((H, R, H, nq)), used, used.astype(int),
                         used.astype(int), 0)
        full = dataclasses.replace(empty, used=np.ones((H, R), bool),
                                   gradient=np.random.default_rng(0).normal(size=(H, R, H, nq)))
        a, b = block.matrix(empty, 100, 5), block.matrix(full, 100, 5)
        assert np.array_equal(a.indices, b.indices) and np.array_equal(a.indptr, b.indptr)


def _rest_q(sc, behind_rad):
    """A robot at rest `behind_rad` behind the reference start on the sweeping joint."""
    reduced = [list(sc.layout.joint_names)[q] for _, q in sc.layout.action_to_q]
    axis = reduced.index("right_arm_0")
    q = sc.q_now.copy()
    q[sc.layout.q_indices[axis]] -= behind_rad
    return q, axis


def _ball_scene(sc, centre, radius):
    """One candidate ball, no safety distance, no active plane."""
    from AG3S.trajopt.linearize import SceneSnapshot

    lin, n_slots = sc.linearizer, sc.scene.n_slots
    active = np.zeros(n_slots, bool)
    active[0] = True
    return SceneSnapshot(
        candidate_pos=np.tile(np.asarray(centre, float), (n_slots, 1)),
        candidate_radius=np.full(n_slots, float(radius)), candidate_active=active,
        d_safe=np.zeros((lin.n_spheres, n_slots)), plane_normal=sc.scene.plane_normal,
        plane_offset=sc.scene.plane_offset, plane_active=np.zeros(2, bool),
        robot_radii=lin.robot_radii, candidate_ids=np.arange(n_slots))


def _centres(sc, q_opt):
    full_h = np.tile(np.asarray(q_opt, float)[:, None], (1, sc.linearizer.horizon))
    return sc.linearizer.sphere_states(full_h, sc.q_now)[0][0]


def test_the_floor_admits_the_start_and_forbids_going_deeper():
    sc = make_scenario(horizon=6)
    qi = sc.layout.q_indices
    stay = np.tile(sc.q_now[qi][:, None], (1, 6))
    c_now = _centres(sc, sc.q_now[qi])
    radii = sc.linearizer.robot_radii
    c_end = _centres(sc, sc.reference[:, -1])
    s = int(np.argmax(np.linalg.norm(c_end - c_now, axis=1)))        # far out on the sweeping arm
    # a ball 4 mm inside sphere s's clearance at q_now, off to its side
    side = np.array([0.0, 1.0, 0.0])
    ball = c_now[s] + side * (radii[s] + 0.01 - 0.004)
    scene = _ball_scene(sc, ball, 0.01)
    opt = _opt(sc, limits={"servo_model": True}, reduction={"sweep_check": True})
    st = opt.observer.update(sc.q_now[qi], None, 1)
    opt.path.begin(sc.q_now, scene, st)
    assert opt.path.floor_min_m < -0.003 and opt.path.floor_rows >= 1   # q_now starts inside
    assert opt.path.violation(stay, scene)[0] >= -1e-9                # at rest: every row at its floor
    # step against the worst row's gradient: the arm moves deeper into the ball
    rows = sc.linearizer.linearize(stay, sc.q_now, scene, sc.config)
    k, r = np.unravel_index(np.argmin(np.where(rows.used, rows.value, np.inf)), rows.value.shape)
    g = rows.gradient[k, r]
    deeper = stay - 0.02 * (g / np.linalg.norm(g))[:, None]
    assert sc.linearizer.full_violation(deeper, sc.q_now, scene) < -0.004 - 1e-4
    assert opt.path.violation(deeper, scene)[0] < -1e-4               # deeper than the floor: seen


def _hidden_ball(sc, inside_q, clear_qs, r_ball=0.002, depth=0.005, margin=0.003):
    """A small ball that the arm at `inside_q` penetrates by `depth` while every sphere at every
    configuration in `clear_qs` stays `margin` clear of it. Searched over directions around each
    sphere of `inside_q` (the spheres overlap one another, so a point beside one sphere is usually
    inside its neighbour too). Returns the ball centre with the most clearance at `clear_qs`."""
    radii = sc.linearizer.robot_radii
    inside = _centres(sc, inside_q)
    clear = [_centres(sc, q) for q in clear_qs]
    i = np.arange(400) + 0.5
    phi, theta = np.arccos(1 - 2 * i / 400), np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    best, best_gap = None, margin
    for sph in range(len(radii)):
        pts = inside[sph] + dirs * (radii[sph] + r_ball - depth)
        # no other sphere at `inside_q` may hold the point deeper than `depth` (it must be grazing)
        gaps = [np.min(np.linalg.norm(pts[:, None] - c[None], axis=2) - radii[None] - r_ball, axis=1)
                for c in clear]
        gap = np.min(gaps, axis=0)
        j = int(np.argmax(gap))
        if gap[j] > best_gap:
            best, best_gap = pts[j], gap[j]
    assert best is not None, "no hidden ball exists for these configurations"
    return best


def test_sweep_sees_a_thin_obstacle_between_two_clear_waypoints():
    """A 2 mm ball beside the middle of a fast row: both waypoints clear it, the swept arm does not.
    Waypoint-only calls its plan safe; the sweep rows see the ball in that plan and change it."""
    sc = make_scenario(horizon=10)
    q0, ref = sc.q_now.copy(), sc.reference.copy()
    ca, cb = _centres(sc, ref[:, 4]), _centres(sc, ref[:, 5])
    assert np.linalg.norm(cb - ca, axis=1).max() > 0.06               # a fast row
    others = [ref[:, k] for k in range(ref.shape[1])] + [q0[sc.layout.q_indices]]
    scene = _ball_scene(sc, _hidden_ball(sc, 0.5 * (ref[:, 4] + ref[:, 5]), others), 0.002)
    assert sc.linearizer.full_violation(ref, q0, scene) > 0.0         # every waypoint is clear
    off = _opt(sc).solve(ref, q0, scene, template=sc.chunk)
    assert off.status.safe                                            # waypoint-only: "safe"
    on = _opt(sc, reduction={"sweep_check": True})
    on.path.begin(q0, scene)
    worst, record = on.path.violation(off.trajectory, scene, record=True)
    assert worst < -0.002                                             # the sweep rows see the ball
    res = _opt(sc, reduction={"sweep_check": True}).solve(ref, q0, scene, template=sc.chunk)
    assert res.status.safe and res.metrics["path_check"]["clearance_m"] >= -1e-4


def test_servo_rows_see_where_only_the_robots_momentum_goes():
    """The arm moves fast on one joint and the plan holds it at q_now. Commands and interpolation
    never leave q_now; the servo carries the arm past it — into a ball placed there."""
    sc = make_scenario(horizon=4)
    q0 = sc.q_now.copy()
    qi = sc.layout.q_indices
    _, axis = _rest_q(sc, 0.0)
    stay = np.tile(q0[qi][:, None], (1, 4))
    opt = _opt(sc, limits={"servo_model": True})
    v = np.zeros(14)
    v[axis] = 3.0                                                     # rad/s
    X = opt.servo.rollout(q0[qi], v, stay, np.zeros(14))["q"]
    far = int(np.argmax(np.abs(X[axis] - q0[qi][axis])))
    c_now, c_over = _centres(sc, q0[qi]), _centres(sc, X[:, far])
    assert np.linalg.norm(c_over - c_now, axis=1).max() > 0.025       # momentum carries it > 25 mm
    scene = _ball_scene(sc, _hidden_ball(sc, X[:, far], [q0[qi]]), 0.002)
    assert sc.linearizer.full_violation(stay, q0, scene) > 0.0        # the commands are clear
    interp = _opt(sc, reduction={"sweep_check": True})
    interp.path.begin(q0, scene)
    assert interp.path.violation(stay, scene)[0] >= 0.0               # interpolation: nothing moves
    opt.path.begin(q0, scene, ServoState(q0[qi], v, np.zeros(14), "test"))
    assert opt.path.violation(stay, scene)[0] < -0.002                # servo: the arm runs into it
