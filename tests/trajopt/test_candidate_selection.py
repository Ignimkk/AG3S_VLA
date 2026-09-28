"""T24 — `sqp.solve` 가 **무엇을 반환하는가**: 초기 iterate 도 후보이고, 그보다 나쁜 후보는 나가지 않는다.

예전에는 `best_merit = np.inf` 로 시작했다. 그래서 첫 QP 후보가 초기 궤적보다 merit 이 나빠 반복점으로
**거절**되어도 반환용 best 로는 저장됐고, 예산으로 끝나면 (T14 30/30 · T17 75/75 chunk 가 그렇게 끝났다)
그 거절된 후보가 나갔다 (지침 §7.3).

QP 를 **대본대로 답하는 가짜**로 바꿔 후보의 품질을 시험이 정한다. 진짜 OSQP 로는 "첫 후보가 나쁜"
경우를 결정적으로 만들 수 없다 — trust region 과 선형화가 대개 개선 방향을 주기 때문이다.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.trajopt.qp import QpSolution
from benchmark.trajopt.sqp import TrajectoryOptimizer
from tests.trajopt.fixtures import make_scenario

pytest.importorskip("casadi")
pytest.importorskip("osqp")

SELECTION_KEYS = ("returned", "returned_index", "initial_merit", "initial_violation_m",
                  "candidates", "n_candidates", "n_accepted", "qp_failures")


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(horizon=10, penetration=0.03)


def _optimizer(scenario, **sections):
    config = scenario.config.with_overrides(sections) if sections else scenario.config
    return TrajectoryOptimizer(scenario.robot, scenario.layout, scenario.limits, config)


class _ScriptedSolver:
    """`QpSolver` 자리에 앉아 **정해 둔 궤적**을 차례로 답한다. 목록이 끝나면 마지막 것을 반복한다."""

    def __init__(self, trajectories):
        self.trajectories = [np.asarray(t, np.float64) for t in trajectories]
        self.calls = 0

    def solve(self, problem, warm_start=True):
        Q = self.trajectories[min(self.calls, len(self.trajectories) - 1)]
        self.calls += 1
        return QpSolution(x=problem.flatten(Q), y=None, success=True, status="scripted",
                          iterations=1, solve_time_ms=0.0)

    def reset(self):
        pass


def _feasible_start(scenario):
    """**한계 안의** 시작 궤적. fixture 의 reference 는 joint box 를 2.5 mm 넘는다
    (`test_no_collision_ablation` 8 절) — 그대로 쓰면 초기 iterate 가 부적격이라 이 파일의 1 절이
    시험하려는 경로(적격한 초기 vs 나쁜 후보)가 열리지 않는다. 관통은 그대로 남는다."""
    start = np.clip(scenario.reference, scenario.limits.lower[:, None] + 1e-3,
                    scenario.limits.upper[:, None] - 1e-3)
    return start


def _scripted(scenario, trajectories, **sqp):
    optimizer = _optimizer(scenario, sqp={"time_budget_ms": 1e6, **sqp})
    optimizer.solver = _ScriptedSolver(trajectories)
    return optimizer


# ------------------------------------------------------------------ 1. 핵심 — 나쁜 첫 후보


def test_a_first_candidate_worse_than_the_start_is_not_returned(scenario):
    """**T24 의 수정 그 자체.** 한 번만 돌고 (예산 도달과 같은 모양) 그 후보가 나쁘면 초기 궤적이 나간다."""
    start = _feasible_start(scenario)
    worse = start + 0.02  # 추적 비용만으로도 merit 이 오른다 — 관통은 그대로 남는다
    optimizer = _scripted(scenario, [worse], max_iterations=1)
    result = optimizer.solve(start, scenario.q_now, scenario.scene)

    m = result.metrics
    assert m["initial_eligible"] is True, "시험의 전제: 초기 궤적은 한계 안이다"
    assert m["n_candidates"] == 1
    cand = m["candidates"][0]
    assert cand["merit"] > m["initial_merit"], "시험의 전제: 첫 후보가 초기 궤적보다 나쁘다"
    assert cand["accepted"] is False
    # 예전 규칙(`best_merit = inf`)이었다면 이 후보가 반환됐다. 지금은 초기 iterate 다.
    assert m["returned"] == "initial"
    assert m["returned_index"] == -1
    np.testing.assert_array_equal(result.trajectory, start)


def test_the_returned_initial_trajectory_is_still_checked_at_full_resolution(scenario):
    """초기 궤적이 나가도 `_finish` 의 최종 검증은 그대로다 — 관통하는 reference 는 `violated` 다."""
    start = _feasible_start(scenario)
    optimizer = _scripted(scenario, [start + 0.02], max_iterations=1)
    result = optimizer.solve(start, scenario.q_now, scenario.scene)
    assert result.metrics["returned"] == "initial"
    assert result.max_violation > 0.0, "fixture 의 reference 는 관통한다"
    assert result.max_violation == pytest.approx(result.metrics["initial_violation_m"], abs=1e-9)
    assert not result.status.safe


def test_a_better_candidate_is_returned_and_marked_accepted(scenario):
    """좋은 후보는 그대로 나간다 — 수정이 개선을 막지 않는다. 진짜 OSQP 로 돈다."""
    result = _optimizer(scenario, sqp={"time_budget_ms": 1e6}).solve(
        scenario.reference, scenario.q_now, scenario.scene)
    m = result.metrics
    assert m["returned"] == "accepted"
    chosen = m["candidates"][m["returned_index"]]
    assert chosen["accepted"] is True
    assert chosen["merit"] < m["initial_merit"]
    assert m["initial_violation_m"] > 0.0 and result.max_violation < m["initial_violation_m"]


def test_the_returned_merit_is_the_minimum_over_start_and_accepted_candidates(scenario):
    """나쁜 후보 · 좋은 후보 · 더 나쁜 후보가 섞여도 **반환은 수락된 것 중 최선 또는 초기**다."""
    start = _feasible_start(scenario)
    script = [start + 0.02, start + 0.02, start - 0.0005, start + 0.03]
    optimizer = _scripted(scenario, script, max_iterations=len(script),
                          step_tolerance=0.0)
    result = optimizer.solve(start, scenario.q_now, scenario.scene)
    m = result.metrics
    assert m["returned"] != "best_unaccepted"
    merits = [m["initial_merit"]] + [c["merit"] for c in m["candidates"] if c["accepted"]]
    returned_merit = (m["initial_merit"] if m["returned_index"] < 0
                      else m["candidates"][m["returned_index"]]["merit"])
    assert returned_merit == pytest.approx(min(merits))
    assert m["initial_eligible"] is True
    # 누가 이겼는지는 merit 이 정하고 (대본의 어느 후보가 관통을 줄이는지는 기하가 정한다),
    # 이 시험은 "수락된 것 중 최선 또는 초기" 라는 규칙만 본다.


# ------------------------------------------------------------------ 2. 기록의 모양


def test_every_candidate_carries_merit_violation_and_acceptance(scenario):
    result = _optimizer(scenario, sqp={"time_budget_ms": 1e6}).solve(
        scenario.reference, scenario.q_now, scenario.scene)
    for c in result.metrics["candidates"]:
        assert set(c) >= {"iteration", "merit", "violation_m", "accepted"}
        assert isinstance(c["accepted"], bool)
        assert c["violation_m"] is not None and c["violation_m"] >= 0.0
    assert result.metrics["n_candidates"] == len(result.metrics["candidates"])
    assert result.metrics["n_accepted"] == sum(c["accepted"] for c in result.metrics["candidates"])


def test_with_collision_off_the_violation_is_not_evaluated_rather_than_zero(scenario):
    """꺼진 판의 merit 은 위반을 재지 않는다 (T15). 기록은 `0` 이 아니라 `None` — "재지 않음" 이다."""
    result = _optimizer(scenario, sqp={"time_budget_ms": 1e6},
                        collision={"enabled": False}).solve(
        scenario.reference, scenario.q_now, scenario.scene)
    assert result.metrics["initial_violation_m"] is None
    assert all(c["violation_m"] is None for c in result.metrics["candidates"])
    # 반환된 궤적의 위반은 여전히 참값이다 (`_finish`).
    assert result.max_violation > 0.0


def test_passthrough_carries_the_same_selection_keys(scenario):
    passthrough = _optimizer(scenario).solve(scenario.reference, scenario.q_now, None)
    full = _optimizer(scenario, sqp={"time_budget_ms": 1e6}).solve(
        scenario.reference, scenario.q_now, scenario.scene)
    for key in SELECTION_KEYS:
        assert key in passthrough.metrics, key
        assert key in full.metrics, key
    assert passthrough.metrics["returned"] == "initial"
    assert passthrough.metrics["candidates"] == []


def test_merit_is_unchanged_by_the_split(scenario):
    """`_merit` 은 `_merit_terms` 의 첫 값이다 — 수락 기준이 바뀌지 않았다."""
    optimizer = _optimizer(scenario)
    optimizer.linearizer.set_attached(None, None)
    merit = optimizer._merit(scenario.reference, scenario.reference, None, scenario.q_now,
                             scenario.scene)
    terms = optimizer._merit_terms(scenario.reference, scenario.reference, None, scenario.q_now,
                                   scenario.scene)
    assert merit == terms[0]
    violation = max(0.0, -optimizer.linearizer.full_violation(
        scenario.reference, scenario.q_now, scenario.scene))
    assert terms[1] == pytest.approx(violation)


# ------------------------------------------------------------------ 3. 적격성 — 한계 밖의 초기 궤적


def _outside_box(scenario, by=0.003):
    """reference 한 스텝을 joint box 밖으로 `by` rad 밀어낸다 (anchor 에 안 걸리게 가운데 스텝)."""
    ref = _feasible_start(scenario).copy()
    k = ref.shape[1] // 2
    ref[0, k] = scenario.limits.upper[0] + by
    return ref


def test_an_initial_trajectory_outside_the_joint_box_is_not_eligible(scenario):
    """**merit 은 관절 한계를 보지 않는다.** box 밖의 초기 궤적은 box 안으로 들어온 후보보다 추적
    비용이 낮아 merit 으로는 언제나 이긴다 — 그것을 반환하면 한계를 넘는 궤적이 나간다.

    그래서 한계 밖의 초기 iterate 는 비교에서 빠진다 (예전 규칙 그대로). 이것이
    `test_no_collision_ablation::test_with_no_collision_and_no_jerk_the_only_motion_left_is_the_limit`
    가 초기화 한 줄만 바꿨을 때 깨진 이유다.
    """
    start = _outside_box(scenario)
    inside = np.clip(start, scenario.limits.lower[:, None], scenario.limits.upper[:, None])
    # 충돌과 평활을 끈다 — merit 이 추적 비용뿐이어야 "box 안으로 들어온 후보가 merit 으로 진다" 가
    # 기하와 무관하게 성립한다 (충돌이 켜져 있으면 3 mrad 이 관통을 줄이고, 평활이 켜져 있으면
    # 튀어나온 한 스텝을 깎아 후보가 이길 수 있다).
    optimizer = _optimizer(scenario, sqp={"time_budget_ms": 1e6, "max_iterations": 1},
                           collision={"enabled": False},
                           cost={"w_smooth": 0.0, "w_continuity": 0.0})
    optimizer.solver = _ScriptedSolver([inside])
    result = optimizer.solve(start, scenario.q_now, scenario.scene)
    m = result.metrics
    assert m["initial_eligible"] is False
    assert m["initial_limit_overshoot"]["position"] == pytest.approx(0.003, abs=1e-9)
    cand = m["candidates"][0]
    assert cand["merit"] > m["initial_merit"] and cand["accepted"] is False, (
        "시험의 전제: box 안으로 들어온 후보는 merit 으로 진다")
    assert m["returned"] == "best_unaccepted", "초기가 부적격이면 거절된 후보라도 그것이 낫다"
    np.testing.assert_allclose(result.trajectory, inside)
    assert result.metrics["limit_overshoot"]["position"] <= 1e-6


def test_the_anchor_counts_as_a_limit_for_eligibility(scenario):
    """첫 스텝이 `q_now` 에서 한 스텝 거리보다 멀면 QP 의 hard 행을 넘는 것이다."""
    optimizer = _optimizer(scenario)
    start = scenario.reference.copy()
    q0 = scenario.q_now[scenario.layout.q_indices]
    start[:, 0] = q0
    start[0, 0] = q0[0] + scenario.limits.max_step[0] + 0.01
    report = optimizer._limit_overshoot(start, scenario.q_now)
    assert report["anchor"] == pytest.approx(0.01, abs=1e-9)
    start[0, 0] = q0[0]
    assert optimizer._limit_overshoot(start, scenario.q_now)["anchor"] == 0.0


def test_a_feasible_initial_is_eligible(scenario):
    start = _feasible_start(scenario)
    optimizer = _scripted(scenario, [start + 0.02], max_iterations=1)
    result = optimizer.solve(start, scenario.q_now, scenario.scene)
    assert result.metrics["initial_eligible"] is True
    assert max(result.metrics["initial_limit_overshoot"].values()) <= 1e-6
