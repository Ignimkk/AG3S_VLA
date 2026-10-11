"""T31 G2 — SQP 예산은 **반복 수 우선**이고, limit 만 어긴 reference 의 대비는 **최소 투영**이다.

(i) E3 에서 새 청크 필드의 첫 linearize 가 ~100 ms 라 예산 50 ms 가 첫 반복 뒤에 걸렸다 —
    1124/1125 청크가 1 반복. `sqp.min_iterations` 만큼은 예산과 무관하게 돈다.
(ii) 초기 iterate 가 limit **만** 어기고 (충돌 위반 0) 받아들여진 후보가 없으면, 거절된 QP 후보
    (`best_unaccepted`, E3b chunk 310 에서 손목 8.66° + 다른 관절 0.4–0.9°) 대신 reference 를
    position box ∩ 첫 스텝 anchor 로 자른 궤적이 나간다 — 그것이 limit 을 전부 지키고 충돌
    검사를 통과할 때만. T24 규칙(적격한 초기 iterate 가 기준)은 그대로다.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.trajopt.config import SqpConfig, TrajOptConfig, TrajOptConfigError
from AG3S.trajopt.limits import limit_report, project_to_limits
from AG3S.trajopt.sqp import TrajectoryOptimizer
from tests.trajopt.fixtures import make_scenario, near_miss_scene
from tests.trajopt.test_candidate_selection import _ScriptedSolver

pytest.importorskip("casadi")
pytest.importorskip("osqp")

NO_SMOOTH = {"cost": {"w_smooth": 0.0, "w_continuity": 0.0}}


@pytest.fixture(scope="module")
def sc():
    return make_scenario(horizon=8, penetration=0.03)


@pytest.fixture(scope="module")
def far(sc):
    """충돌 위반이 **없는** 씬 — 장애물이 가장 가까운 구에서 20 mm 떨어져 있다 (바닥 평면은
    활성·만족). 더 멀리 두면 수직 방향으로 밀린 장애물이 토르소에 닿는다 (−0.30 에서 −185 mm)."""
    scene = near_miss_scene(sc.linearizer, sc.reference, sc.q_now, penetration=-0.05)
    assert sc.linearizer.full_violation(sc.reference, sc.q_now, scene) > 0.01
    return scene


def _opt(sc, **sections):
    cfg = sc.config.with_overrides(sections) if sections else sc.config
    return TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg)


def _row(sc, name):
    names = [sc.layout.joint_names[i] for i in sc.layout.q_indices]
    return names.index(name)


def _wrist_over(sc):
    """E3 의 모양 — 손목이 box 안(2.94)에서 출발해 `left_arm_6` 을 box(MJCF 2.967 − 0.02 = 2.947)
    밖으로 천천히 (스텝당 7 mrad, 한도 126 mrad) 명령한다. `(reference, q_now)`."""
    row = _row(sc, "left_arm_6")
    q_now = sc.q_now.copy()
    q_now[sc.layout.q_indices[row]] = 2.94
    ref = sc.reference.copy()
    ref[row] = np.linspace(2.94, 2.99, ref.shape[1])
    return ref, q_now


# ------------------------------------------------------------------ (ii) 최소 투영


@pytest.mark.parametrize("max_iterations", [1, 3])
def test_a_limit_only_reference_returns_its_minimal_projection(sc, far, max_iterations):
    """**합성 limit-only 사례.** 반환은 `clip(reference, box ∩ anchor)` 와 비트 단위로 같다."""
    ref, q_now = _wrist_over(sc)
    over = limit_report(ref, sc.limits)
    assert over["position"] > 0.04, "전제: reference 가 box 밖이다 (손목 + fixture 의 left_arm_3)"
    assert over["velocity"] == 0.0 and over["acceleration"] == 0.0
    opt = _opt(sc, sqp={"max_iterations": max_iterations, "time_budget_ms": 1e6})
    result = opt.solve(ref, q_now, far)

    m = result.metrics
    assert m["initial_eligible"] is False and m["initial_violation_m"] == 0.0
    assert m["initial_limit_overshoot"]["anchor"] == 0.0
    assert m["n_accepted"] == 0, "전제: 받아들여진 후보가 없다 (추적 비용만큼 나쁘다)"
    assert m["returned"] == "projection" and m["returned_index"] == -2
    expected = project_to_limits(ref, sc.limits, q_now[sc.layout.q_indices])
    np.testing.assert_array_equal(result.trajectory, expected)
    # 최소 투영 — box 안의 원소는 한 비트도 바뀌지 않는다.
    inside = (ref >= sc.limits.lower[:, None]) & (ref <= sc.limits.upper[:, None])
    np.testing.assert_array_equal(result.trajectory[inside], ref[inside])
    assert limit_report(result.trajectory, sc.limits)["position"] == 0.0
    p = m["projection"]
    assert p["used"] is True and p["reason"] == "limit_only"
    assert p["max_change_rad"] == pytest.approx(over["position"])
    assert p["violation_m"] == 0.0
    assert any("minimal projection" in n for n in result.notes)
    assert result.max_violation == 0.0 and result.status.safe


def test_the_projection_replaces_the_rejected_step_that_moved_every_joint(sc, far):
    """대조군 — 스위치를 끄면 T24 규칙대로 거절된 QP 후보가 나가고, 그것은 투영이 아니다."""
    ref, q_now = _wrist_over(sc)
    old = _opt(sc, sqp={"max_iterations": 1, "time_budget_ms": 1e6,
                        "limit_projection_fallback": False}).solve(ref, q_now, far)
    assert old.metrics["returned"] == "best_unaccepted"
    assert old.metrics["projection"] == {"tried": False, "used": False, "reason": "disabled",
                                         "max_change_rad": None, "violation_m": None,
                                         "limit_overshoot": None}
    expected = project_to_limits(ref, sc.limits, q_now[sc.layout.q_indices])
    assert np.abs(old.trajectory - expected).max() > 1e-4


def test_an_eligible_initial_iterate_is_still_the_t24_fallback(sc, far):
    """T24 규칙은 그대로 — 한계 안의 reference 는 투영을 시도하지도 않는다."""
    inside = project_to_limits(sc.reference, sc.limits, sc.q_now[sc.layout.q_indices])
    result = _opt(sc, sqp={"time_budget_ms": 1e6}).solve(inside, sc.q_now, far)
    assert result.metrics["initial_eligible"] is True
    assert result.metrics["returned"] in ("initial", "accepted")
    assert result.metrics["projection"] is None


def test_a_reference_that_also_collides_is_not_projected(sc):
    """충돌 위반이 있으면 limit **만** 어긴 것이 아니다 — 투영은 시도되지 않는다."""
    ref, q_now = _wrist_over(sc)
    start_inside = project_to_limits(ref, sc.limits, q_now[sc.layout.q_indices])
    opt = _opt(sc, sqp={"max_iterations": 1, "time_budget_ms": 1e6})
    opt.solver = _ScriptedSolver([start_inside + 0.03])  # 더 나쁘다 → 거절
    result = opt.solve(ref, q_now, sc.scene)
    m = result.metrics
    assert m["initial_violation_m"] > 0.0 and m["n_accepted"] == 0
    assert m["returned"] == "best_unaccepted"
    assert m["projection"]["tried"] is False
    assert m["projection"]["reason"] == "initial_collision_violation"


def test_a_projection_that_collides_is_not_returned(sc, far, monkeypatch):
    ref, q_now = _wrist_over(sc)
    expected = project_to_limits(ref, sc.limits, q_now[sc.layout.q_indices])
    opt = _opt(sc, sqp={"max_iterations": 1, "time_budget_ms": 1e6})
    original = opt.linearizer.full_violation

    def fake(trajectory, q_now, scene, states=None):
        if np.array_equal(trajectory, expected):
            return -0.01  # 투영만 1 cm 관통한다고 답한다
        return original(trajectory, q_now, scene, states)

    monkeypatch.setattr(opt.linearizer, "full_violation", fake)
    result = opt.solve(ref, q_now, far)
    p = result.metrics["projection"]
    assert p["tried"] is True and p["used"] is False and p["reason"] == "projection_collides"
    assert p["violation_m"] == pytest.approx(0.01)
    assert result.metrics["returned"] == "best_unaccepted"


def test_a_projection_that_cannot_fix_a_velocity_violation_is_not_returned(sc, far):
    """원소별 clip 은 속도를 고치지 않는다 — 그 투영은 적격하지 않으므로 나가지 않는다."""
    ref, q_now = _wrist_over(sc)
    i = _row(sc, "left_arm_0")
    ref[i, 4:] += 0.5  # 스텝 3→4 에서 0.5 rad 점프 (한 스텝 한도 ≈ 0.19 rad)
    opt = _opt(sc, sqp={"max_iterations": 1, "time_budget_ms": 1e6}, **NO_SMOOTH)
    result = opt.solve(ref, q_now, far)
    m = result.metrics
    assert m["initial_limit_overshoot"]["velocity"] > 0.0
    assert m["n_accepted"] == 0, "전제: 받아들여진 후보가 없다"
    assert m["projection"]["reason"] == "projection_outside_limits"
    assert m["projection"]["limit_overshoot"]["velocity"] > 0.0
    assert m["projection"]["violation_m"] is None, "적격하지 않으면 충돌 검사까지 가지 않는다"
    assert m["returned"] == "best_unaccepted"


def test_the_projection_includes_the_step_zero_anchor(sc, far):
    """anchor(``|Q0 − q_now| ≤ v_max·dt``)도 QP 가 hard 로 지키는 box 다 — 투영에 들어간다."""
    q0 = sc.q_now[sc.layout.q_indices]
    ref = np.repeat(q0[:, None], sc.reference.shape[1], axis=1)
    i = _row(sc, "right_arm_0")
    ref[i, :] = q0[i] + 0.2  # 첫 스텝이 anchor(≈ 0.1885 rad)를 0.0115 넘는다
    opt = _opt(sc, sqp={"max_iterations": 1, "time_budget_ms": 1e6}, **NO_SMOOTH)
    result = opt.solve(ref, sc.q_now, far)
    m = result.metrics
    assert m["initial_limit_overshoot"]["anchor"] > 0.01
    assert m["returned"] == "projection"
    assert result.trajectory[i, 0] == pytest.approx(q0[i] + sc.limits.max_step[i])
    np.testing.assert_array_equal(result.trajectory[i, 1:], ref[i, 1:])
    assert m["projection"]["limit_overshoot"]["anchor"] == 0.0


def test_with_collisions_off_the_projection_skips_the_collision_check(sc):
    """충돌이 꺼진 판 (T15) — merit 도 충돌을 보지 않으므로 투영도 보지 않는다."""
    ref, q_now = _wrist_over(sc)
    opt = _opt(sc, sqp={"max_iterations": 1, "time_budget_ms": 1e6},
               collision={"enabled": False})
    result = opt.solve(ref, q_now, sc.scene)  # 관통하는 씬
    m = result.metrics
    assert m["initial_violation_m"] is None
    assert m["returned"] == "projection" and m["projection"]["violation_m"] is None
    assert result.max_violation > 0.0, "판정은 그대로 참값이다 (관통)"
    assert any("not enforced" in n for n in result.notes)


def test_project_to_limits_returns_none_when_box_and_anchor_do_not_meet(sc):
    q_far = sc.q_now[sc.layout.q_indices].copy()
    q_far[_row(sc, "left_arm_6")] = 3.5  # 로봇이 자기 box 밖에 max_step 이상
    assert project_to_limits(sc.reference, sc.limits, q_far) is None


# ------------------------------------------------------------------ (i) 반복 수 우선


def test_the_default_budget_runs_every_designed_iteration():
    cfg = SqpConfig()
    assert cfg.min_iterations == 3 == cfg.max_iterations
    assert cfg.time_budget_ms == 50.0
    assert cfg.limit_projection_fallback is True
    with pytest.raises(TrajOptConfigError, match="min_iterations"):
        TrajOptConfig.from_dict({"sqp": {"min_iterations": 0}})


def _budget_run(sc, **sqp):
    start = project_to_limits(sc.reference, sc.limits, sc.q_now[sc.layout.q_indices])
    opt = _opt(sc, sqp={"time_budget_ms": 1e-6, "step_tolerance": 0.0, **sqp})
    # 대본: 매번 조금씩 다른(나쁜) 후보 — QP 실패·수렴 없이 반복 수만 센다.
    opt.solver = _ScriptedSolver([start + 0.01 * (k + 1) for k in range(6)])
    return opt.solve(start, sc.q_now, sc.scene).metrics


@pytest.mark.parametrize("min_iterations, max_iterations, expected, hit", [
    (1, 3, 1, True),    # T31 전 동작 — 첫 반복 뒤 곧바로 예산
    (2, 3, 2, True),    # 둘은 반드시, 그 뒤 예산
    (3, 3, 3, False),   # 기본 — 설계대로 전부 (끝난 이유는 상한)
    (3, 1, 1, False),   # 상한이 이긴다
])
def test_the_wall_clock_budget_applies_only_after_min_iterations(
        sc, min_iterations, max_iterations, expected, hit):
    m = _budget_run(sc, min_iterations=min_iterations, max_iterations=max_iterations)
    assert m["sqp_iterations"] == expected
    assert m["time_budget_hit"] is hit
    assert m["min_iterations"] == min(min_iterations, max_iterations)
