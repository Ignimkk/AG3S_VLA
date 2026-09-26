"""TO 는 **실행되는 창만** 다듬는다 (T6f).

왜 이것이 계약인지: T6d 실측에서 refined chunk 는 실행되는 앞 8 스텝 동안 손끝을 사과에서
+89.86 mm 떼어 놓고, 버려지는 뒷부분에서만 −17.61 mm 로 접근했다. 회피는 매 chunk 실행되고
접근은 매 chunk 버려져서 사과가 네 실행 내내 0.0 mm 움직였다 — TO 가 틀린 것이 아니라
**미룰 자리가 있었던 것**이다.

그래서 이 파일이 지키는 것은 최적화 품질이 아니라 **창의 경계**다:

* 기본 설정이면 다듬는 스텝 수 = `execution_length`. 8 을 박지 않고 그 값을 따라간다.
* 실행 창 밖의 chunk 행은 **글자 그대로** 정책이 낸 값이다.
* 되돌릴 수 있다 — `plan_horizon` 에 숫자나 `full` 을 주면 예전 동작이다.
* 겹치는 부분이 없으면 warm start 가 지난 chunk 의 자세를 끌고 오지 않는다.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.trajopt import serve_safe
from benchmark.trajopt.config import (
    PLAN_EXECUTION_WINDOW,
    HorizonConfig,
    TrajOptConfig,
    TrajOptConfigError,
)
from benchmark.trajopt.refiner import TrajOptChunkRefiner
from benchmark.trajopt.sqp import TrajectoryOptimizer
from tests.trajopt.fixtures import make_scenario

pytest.importorskip("casadi")
pytest.importorskip("osqp")

#: 필수 인자만. 여기에 아무것도 더하지 않은 것이 "지금과 같은 서버" 다.
BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


# --- 기본값이 실행 창이다 ------------------------------------------------------------


def test_the_default_plans_exactly_the_executed_window():
    horizon = TrajOptConfig().horizon
    assert horizon.plan_horizon == PLAN_EXECUTION_WINDOW
    assert horizon.planned == horizon.execution_length == 8
    assert horizon.plans_execution_window_only


def test_the_sentinel_follows_execution_length_rather_than_the_number_eight():
    """실행 길이는 로컬의 `OPEN_LOOP_HORIZON` 에서 오고, 이 패키지에서는
    `horizon.execution_length` 하나가 그것을 들고 있다. 두 번째 자리에 8 을 박으면 한쪽만
    바뀌는 날이 온다."""
    cfg = TrajOptConfig.from_dict({"horizon": {"execution_length": 5}})
    assert cfg.horizon.planned == 5
    cfg = TrajOptConfig.from_dict({"horizon": {"horizon": 12, "execution_length": 12}})
    assert cfg.horizon.planned == 12


def test_an_integer_or_none_restores_the_older_behaviour():
    assert TrajOptConfig.from_dict({"horizon": {"plan_horizon": 32}}).horizon.planned == 32
    assert TrajOptConfig.from_dict({"horizon": {"plan_horizon": None}}).horizon.planned == 50
    assert not TrajOptConfig.from_dict(
        {"horizon": {"plan_horizon": 32}}).horizon.plans_execution_window_only


def test_a_misspelled_sentinel_stops_instead_of_planning_everything():
    """조용히 '전체 계획' 으로 떨어지면 T6d 의 실패가 그대로 돌아온다."""
    with pytest.raises(TrajOptConfigError, match="plan_horizon"):
        HorizonConfig(plan_horizon="excecution").validate()


def test_the_planned_window_is_never_shorter_than_what_executes():
    with pytest.raises(TrajOptConfigError, match="unplanned"):
        HorizonConfig(execution_length=8, plan_horizon=4).validate()


# --- refiner: 창 밖은 정책의 답 그대로 -----------------------------------------------


@pytest.fixture(scope="module")
def scenario():
    """실행 창(4)보다 긴 chunk(12). fixture 의 `execution_length` 는 min(4, H) 다."""
    return make_scenario(horizon=12, penetration=0.03)


def _refiner(scenario, **overrides):
    config = scenario.config.with_overrides(overrides) if overrides else scenario.config
    return TrajOptChunkRefiner(
        scenario.robot, scenario.layout, lambda ctx: (scenario.scene, scenario.q_now, True), config
    )


def test_only_the_executed_window_is_touched(scenario):
    k = scenario.config.horizon.execution_length
    refiner = _refiner(scenario, horizon={"plan_horizon": PLAN_EXECUTION_WINDOW})
    assert refiner.config.horizon.planned == k

    out = refiner.refine(scenario.chunk)
    # 창 밖은 **글자 그대로** 정책의 답이다. 근사 비교가 아니라 동일성이어야 한다 —
    # `trajectory_to_chunk` 가 template 을 복사해 앞부분만 덮어쓰기 때문이다.
    assert np.array_equal(out[k:], scenario.chunk[k:])
    # 그리고 창 안에서는 실제로 무언가 했다 (안 했으면 위 단정이 공허하다).
    assert not np.array_equal(out[:k], scenario.chunk[:k])


def test_a_longer_plan_window_writes_past_the_executed_window(scenario):
    """되돌린 설정에서는 실행되지 않는 스텝도 바뀐다 — 바로 그것이 T6d 가 잡은 문제다."""
    k = scenario.config.horizon.execution_length
    planned = scenario.horizon  # 12 — chunk 전체
    out = _refiner(scenario, horizon={"plan_horizon": planned}).refine(scenario.chunk)
    assert not np.array_equal(out[k:], scenario.chunk[k:])


def test_the_verdict_is_measured_over_the_window_that_executes(scenario):
    """`max_violation` 이 다듬는 창을 따라간다 — 실행되지 않는 스텝의 위반으로 HOLD 되면
    로봇은 실행할 수 있는 궤적을 들고 멈춘다."""
    k = scenario.config.horizon.execution_length
    refiner = _refiner(scenario, horizon={"plan_horizon": PLAN_EXECUTION_WINDOW})
    refiner.refine(scenario.chunk)
    assert refiner.last_result.trajectory.shape[1] == k
    assert refiner.last_result.metrics["planned_steps"] == k


# --- warm start: 겹치는 것이 없으면 물려받지 않는다 ----------------------------------


def _one_step_optimizer(scenario, execution_length):
    """SQP 한 걸음만 허용한다 — 그러면 결과가 **첫 iterate 근처**에 머물러, 무엇으로
    시작했는지가 관측 가능해진다. 걸음마다 신뢰 영역만큼만 움직일 수 있기 때문이다."""
    config = scenario.config.with_overrides({
        "horizon": {"horizon": scenario.horizon, "execution_length": execution_length,
                    "plan_horizon": None},
        "sqp": {"max_iterations": 1},
    })
    return TrajectoryOptimizer(scenario.robot, scenario.layout, scenario.limits, config), config


def _solve_with_previous(scenario, execution_length, offset):
    optimizer, config = _one_step_optimizer(scenario, execution_length)
    # 지난 chunk 의 궤적을 **눈에 띄게 멀리** 둔다. 물려받으면 결과가 그쪽으로 끌려간다.
    optimizer._previous = scenario.reference + offset  # noqa: SLF001
    result = optimizer.solve(scenario.reference, scenario.q_now, scenario.scene)
    moved = float(np.abs(result.trajectory - scenario.reference).max())
    return moved, config.sqp.trust_radius


def test_no_overlap_means_no_inherited_pose(scenario):
    """실행 길이가 계획 지평과 같으면 지난 계획과 겹치는 스텝이 **하나도 없다.** 그때
    shift 한 warm start 는 실행 창보다 앞선 자세를 섞어 넣는다 — 그것을 안 한다."""
    moved, trust = _solve_with_previous(scenario, scenario.horizon, offset=1.0)
    assert moved <= trust + 1e-6


def test_with_overlap_the_warm_start_is_still_used(scenario):
    """겹치는 스텝이 있으면 예전처럼 섞는다 — 위 테스트가 무엇을 재는지 이것이 보인다."""
    moved, trust = _solve_with_previous(scenario, 1, offset=1.0)
    assert moved > trust + 1e-6


# --- 서버 flag -----------------------------------------------------------------------


def test_the_server_defaults_to_the_execution_window():
    args = serve_safe.build_parser().parse_args(BASE_ARGV)
    assert args.plan_horizon == PLAN_EXECUTION_WINDOW
    assert serve_safe.resolve_plan_horizon(args.plan_horizon) == PLAN_EXECUTION_WINDOW


@pytest.mark.parametrize("given,expected", [
    ("execution", PLAN_EXECUTION_WINDOW),
    ("window", PLAN_EXECUTION_WINDOW),
    ("full", None),
    ("none", None),
    ("32", 32),
])
def test_the_flag_resolves_to_a_config_value(given, expected):
    assert serve_safe.resolve_plan_horizon(given) == expected


def test_a_nonsense_plan_horizon_stops_the_server_before_the_gpu_is_touched():
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--plan-horizon", "soonish"])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_the_flag_reaches_the_config_that_the_optimizer_reads():
    """flag → `TrajOptConfig` 까지 이어지는지. 여기가 끊기면 로그에는 8 이 찍히고 최적화기는
    50 을 계획한다."""
    resolved = serve_safe.resolve_plan_horizon("32")
    cfg = TrajOptConfig.from_dict({"horizon": {"plan_horizon": resolved}})
    assert cfg.horizon.planned == 32
