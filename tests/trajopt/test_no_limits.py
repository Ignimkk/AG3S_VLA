"""T27 — `--no-limits`: TO 에서 joint position box · 속도 · 가속도 행을 뺀다.

사용자 사다리 실험 E1 ("TO only") 의 조건이다: 추적 항 하나, 충돌 행 0, joint limit 행 0.
**남기는 것**: SQP trust region (`sqp.trust_radius`, box 행 안) 과 첫 스텝 anchor
(``|Q[:, 0] - q_now| <= v_max·dt``). anchor 는 `max_step` 을 **유한하게 둔 채** 속도 행만 끄는
`RelaxedJointLimits` 로 남긴다 — `sqp._limit_overshoot` 가 초기 iterate 의 적격성을 같은 값으로 재므로
(T24), 값을 `inf` 로 만들면 QP 와 적격성 검사가 갈라진다.

## 판정표 (T15 표의 네 번째 줄)

| 충돌 | `w_smooth`·`w_continuity` | limit | 최적해 |
|---|---|---|---|
| 없음 | 0 | 켬 | reference — **limit 안일 때만** (T15: box 밖 2.547 mm 는 움직인다) |
| 없음 | 0 | **끔** | **reference 그 자체** — anchor 를 넘지 않는 한 |
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from benchmark.trajopt import serve_safe
from benchmark.trajopt.config import NO_LIMITS, LimitsConfig, TrajOptConfig, TrajOptConfigError
from benchmark.trajopt.limits import RelaxedJointLimits, build_limits, limit_report, velocity_rows
from benchmark.trajopt.problem import build_problem
from benchmark.trajopt.sqp import TrajectoryOptimizer
from benchmark.trajopt.types import JointLimits

from tests.trajopt.fixtures import make_scenario, near_miss_scene

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]

#: E1 의 목적함수 — 추적 항 하나. 충돌도 끈다 (`--no-perception` 이 그렇게 한다).
TRACK_ONLY = {"collision": {"enabled": False},
              "cost": {"w_smooth": 0.0, "w_continuity": 0.0}}


@pytest.fixture(scope="module")
def sc():
    return make_scenario(horizon=8, penetration=0.05)


@pytest.fixture(scope="module")
def scene(sc):
    return near_miss_scene(sc.linearizer, sc.reference, sc.q_now)


def _solve(sc, scene, reference, *, no_limits: bool, over=TRACK_ONLY):
    cfg = sc.config.with_overrides({**over, **({"limits": NO_LIMITS} if no_limits else {})})
    limits = build_limits(sc.robot, sc.layout, dt=cfg.horizon.dt, config=cfg.limits)
    return TrajectoryOptimizer(sc.robot, sc.layout, limits, cfg).solve(
        reference, sc.q_now, scene), limits


# --- 1. 기본값은 그대로 ----------------------------------------------------------------------


def test_limits_are_on_by_default_everywhere():
    cfg = LimitsConfig()
    assert cfg.enforce_position and cfg.enforce_velocity and cfg.enforce_acceleration
    assert cfg.keep_anchor is False and cfg.enforces_nothing is False
    assert TrajOptConfig().limits == cfg
    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert args.no_limits is False
    # T31: 서버는 position 범위를 **자기가 로드한 MJCF** (`--model-xml`) 에서 읽는다 — 그 경로만
    # `LimitsConfig()` 와 다르다. 나머지(켜진 스위치 셋)는 그대로다.
    assert serve_safe.trajopt_config_from_args(args).limits == LimitsConfig(
        model_xml=args.model_xml)


def test_the_old_velocity_switch_still_drops_the_anchor_too(sc):
    """`enforce_velocity=False` 의 예전 뜻(= `max_step` inf, anchor 도 없음)은 바뀌지 않았다."""
    off = build_limits(sc.robot, sc.layout, dt=1 / 15,
                       config=LimitsConfig(enforce_velocity=False))
    assert type(off) is JointLimits
    assert np.all(np.isinf(off.max_step)) and not velocity_rows(off)


# --- 2. 무엇이 빠지고 무엇이 남나 --------------------------------------------------------


def test_no_limits_is_one_dict_and_says_what_it_keeps():
    assert NO_LIMITS == {"enforce_position": False, "enforce_velocity": False,
                         "enforce_acceleration": False, "keep_anchor": True}
    assert LimitsConfig(**NO_LIMITS).enforces_nothing


def test_no_limits_opens_the_box_and_keeps_a_finite_anchor_step(sc):
    lim = build_limits(sc.robot, sc.layout, dt=1 / 15, config=LimitsConfig(**NO_LIMITS))
    assert isinstance(lim, RelaxedJointLimits) and lim.velocity_rows is False
    assert np.all(np.isneginf(lim.lower)) and np.all(np.isposinf(lim.upper))
    assert np.all(np.isinf(lim.max_step_change))
    # anchor 의 근거는 남는다 — 로봇 자신의 v_max·dt 그대로.
    on = build_limits(sc.robot, sc.layout, dt=1 / 15)
    assert np.all(np.isfinite(lim.max_step))
    assert np.array_equal(lim.max_step, on.max_step)
    assert not velocity_rows(lim) and velocity_rows(on)


def test_the_qp_has_no_velocity_or_acceleration_rows_and_the_box_is_trust_region_plus_anchor(sc):
    cfg = sc.config.with_overrides({"limits": NO_LIMITS})
    lim = build_limits(sc.robot, sc.layout, dt=cfg.horizon.dt, config=cfg.limits)
    q0 = sc.q_now[sc.layout.q_indices]
    problem = build_problem(sc.reference, sc.reference, lim, cfg, q_now=q0)
    assert "velocity" not in problem.row_blocks and "acceleration" not in problem.row_blocks
    assert problem.n_kinematic_slack == 0
    lo, hi = problem.row_blocks["box"]
    nq, radius = sc.layout.nq_opt, cfg.sqp.trust_radius
    flat = sc.reference.T.reshape(-1)
    # 첫 스텝 뒤는 trust region **만** (joint box 없음).
    assert np.allclose(problem.l[lo + nq:hi], flat[nq:] - radius)
    assert np.allclose(problem.u[lo + nq:hi], flat[nq:] + radius)
    # 첫 스텝은 trust region ∩ anchor.
    assert np.allclose(problem.l[lo:lo + nq], np.maximum(flat[:nq] - radius, q0 - lim.max_step))
    assert np.allclose(problem.u[lo:lo + nq], np.minimum(flat[:nq] + radius, q0 + lim.max_step))

    # 대조군: 기본 한계면 속도·가속도 행이 있다.
    default = build_problem(sc.reference, sc.reference, sc.limits, sc.config, q_now=q0)
    assert "velocity" in default.row_blocks and "acceleration" in default.row_blocks


def test_limit_report_does_not_count_a_limit_that_was_not_imposed(sc):
    lim = build_limits(sc.robot, sc.layout, dt=1 / 15, config=LimitsConfig(**NO_LIMITS))
    jumped = sc.reference.copy()
    jumped[0, 3:] += 10.0 * lim.max_step[0]          # 스텝 사이 속도 초과
    jumped[1, :] += 50.0                              # joint box 한참 밖
    assert limit_report(jumped, lim) == {"position": 0.0, "velocity": 0.0, "acceleration": 0.0}
    assert max(limit_report(jumped, sc.limits).values()) > 0.0


# --- 3. 핵심: 추적 항만 있고 limit 이 없으면 refined ≡ reference ----------------------------


def test_track_only_without_limits_returns_the_reference_exactly(sc, scene):
    """**E1 의 판정 기준 #1** (T27 task). 이 fixture 의 reference 는 joint box 밖 2.547 mm 다
    (T15) — limit 을 켜면 그만큼 움직이고, 끄면 **한 비트도** 안 움직인다."""
    assert limit_report(sc.reference, sc.limits)["position"] > 0.0
    with_limits, _ = _solve(sc, scene, sc.reference, no_limits=False)
    without, _ = _solve(sc, scene, sc.reference, no_limits=True)

    assert float(np.abs(with_limits.trajectory - sc.reference).max()) == pytest.approx(
        limit_report(sc.reference, sc.limits)["position"], rel=0.02)
    assert np.array_equal(without.trajectory, sc.reference)
    assert without.metrics["returned"] == "initial"
    assert without.metrics["initial_eligible"] is True
    assert without.reference_deviation == 0.0
    # 충돌이 꺼져 있어도 측정은 참값이다 (T15) — 이 씬의 관통은 그대로 보고된다.
    assert without.max_violation > 0.0


def test_a_velocity_violating_reference_passes_through_untouched(sc, scene):
    ref = sc.reference.copy()
    ref[0, 4:] += 3.0 * sc.limits.max_step[0]         # 스텝 3→4 에서 속도 초과
    assert limit_report(ref, sc.limits)["velocity"] > 0.0
    with_limits, _ = _solve(sc, scene, ref, no_limits=False)
    without, _ = _solve(sc, scene, ref, no_limits=True)
    assert np.abs(with_limits.trajectory - ref).max() > 1e-3
    assert np.array_equal(without.trajectory, ref)


def test_the_anchor_is_what_remains_and_it_moves_only_the_first_step(sc, scene):
    """**남긴 것이 실제로 남았다.** 첫 스텝이 q_now 에서 v_max·dt 보다 멀면, QP 는 그 초과량을 당긴다.

    **반환값이 anchor 를 완전히 지키지는 않는다** — 이 STEP 이 만든 것이 아니라 `sqp` 후보 선택(T24)
    의 성질이고, limit 을 켠 판에서도 똑같이 나온다 (실측: 0.020 rad 초과 중 0.015 rad 만 당김). 초기 iterate
    가 부적격이면 모든 QP 후보가 초기 merit 보다 나빠 **거절**되고, trust region 이 0.01 까지 줄면
    box(trust region ∩ anchor)가 교차해 중점으로 접히며, 그 후보가 가장 작은 merit 으로 `best_unaccepted`
    가 된다. 그래서 여기서는 **QP 가 anchor 를 거는가** (첫 후보가 정확히 초과량만큼 움직인다) 와
    **첫 스텝 밖은 움직이지 않는가** 를 박고, 반환된 양은 범위로만 본다.
    """
    lim = build_limits(sc.robot, sc.layout, dt=sc.config.horizon.dt,
                       config=LimitsConfig(**NO_LIMITS))
    q0 = sc.q_now[sc.layout.q_indices]
    ref = np.tile(q0[:, None], (1, sc.horizon))       # 제자리 — 속도 0
    joint = 2
    over = 0.02
    ref[joint, :] = q0[joint] + lim.max_step[joint] + over   # 모든 스텝이 anchor 밖으로 over
    r, _ = _solve(sc, scene, ref, no_limits=True)

    assert r.metrics["initial_eligible"] is False
    assert r.metrics["initial_limit_overshoot"]["anchor"] == pytest.approx(over, abs=1e-9)
    # QP 는 anchor 를 hard 로 건다: 첫 후보(trust radius 0.15 ≫ over)가 정확히 초과량만큼 움직인다.
    assert r.metrics["candidates"][0]["step_size_rad"] == pytest.approx(over, abs=5e-4)
    moved = np.abs(r.trajectory - ref)
    # 반환값: 첫 스텝의 그 관절만 당겨졌다 (위 docstring — 후보 선택 때문에 초과량 전부는 아니다).
    assert 0.5 * over <= moved[joint, 0] <= over + 5e-4
    moved[joint, 0] = 0.0
    assert moved.max() < 5e-4


# --- 4. 서버 flag -----------------------------------------------------------------------


def test_the_server_flag_reaches_the_config():
    args = serve_safe.build_parser().parse_args(BASE_ARGV + ["--no-limits"])
    cfg = serve_safe.trajopt_config_from_args(args)
    # T31: `--model-xml` 이 limits 절에 함께 실린다 (position 범위의 출처). `NO_LIMITS` 는 그대로.
    assert cfg.limits == LimitsConfig(**NO_LIMITS, model_xml=args.model_xml)
    # 목적함수·충돌은 이 flag 가 건드리지 않는다.
    assert cfg.cost == TrajOptConfig().cost and cfg.collision.enabled is True


def test_no_limits_with_no_safe_is_refused():
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--no-limits", "--no-safe"])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_w_track_zero_is_still_refused():
    """T15 의 거절은 이 STEP 에서 한 줄도 느슨해지지 않았다."""
    args = serve_safe.build_parser().parse_args(
        BASE_ARGV + ["--no-limits", "--no-perception", "--w-track", "0"])
    with pytest.raises(TrajOptConfigError, match="w_track"):
        serve_safe.trajopt_config_from_args(args)


def test_the_startup_log_says_what_is_off_and_what_remains(caplog):
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_limits_switch(LimitsConfig(**NO_LIMITS))
    text = caplog.text
    assert "JOINT LIMIT ROWS ARE OFF" in text
    assert "position, velocity, acceleration" in text
    assert "trust region" in text and "anchor" in text and "남음" in text


def test_the_startup_log_is_silent_by_default(caplog):
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_limits_switch(LimitsConfig())
    assert caplog.text == ""
