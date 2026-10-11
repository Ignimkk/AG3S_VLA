"""T15 — `--no-collision`: 충돌 제약 **행만** 끄고 TO 는 그대로 돌린다.

사용자가 제안한 ablation 이다: *"TO 에 가장 기본적인 jerk 목적함수만 포함하고 나머지 제약을 모두
제거해 봐. 충돌 제약을 제거한 후에도 사과를 못 집으면 이건 궤적최적화의 문제로 볼 수 있지 않나?"*

## 판정표 — 그리고 이 파일이 세 번째 줄을 더한다

| `w_smooth` | 충돌 | 최적해가 무엇이어야 하나 |
|---|---|---|
| 0 | 없음 | **reference 그 자체** — 목적이 추적뿐이므로 |
| 0.05 | 없음 | jerk 가 미는 만큼만 다름 |
| 0 | 없음 | **reference 가 limit 밖이면 그만큼은 여전히 움직인다** ← 실측으로 나왔다 |

세 번째 줄이 중요하다. 이 파일의 마지막 절이 그것을 수치로 박는다 — limit 밖의 reference 를
주면 해가 `position` 초과량과 **정확히 같은 만큼** 움직인다. 그러면 "충돌을 껐는데도 청크가
바뀌었다" 를 목적함수 탓으로 읽으면 안 된다.

## 규율

* 기본값은 **켠 상태**. `collision.enabled = True`.
* `--no-safe` 와의 조합은 **시작 전에 거절** (TO 가 안 도는데 TO 의 제약을 끄는 것은 뜻이 없다).
* 행을 **지우지 않는다** — `LinearizedRows.all_unused` 로 같은 모양, 전부 비활성. 이 클래스의
  고정 희소성이 QP 인수분해 재사용의 전제이고 (193 ms → 25 ms), 그것을 깨지 않는다.
* **판정은 거짓말하지 않는다.** 측정(`worst_row`)은 이 스위치와 무관하므로 `max_violation` 은
  참값이고, 충돌이 꺼졌다는 사실이 `notes` 와 `metrics` 양쪽에 실린다.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from AG3S.trajopt import serve_safe, wire
from AG3S.trajopt.config import TrajOptConfig
from AG3S.trajopt.limits import limit_report
from AG3S.trajopt.linearize import LinearizedRows
from AG3S.trajopt.sqp import TrajectoryOptimizer

from tests.trajopt.fixtures import make_scenario, near_miss_scene

BASE_ARGV = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


@pytest.fixture(scope="module")
def sc():
    return make_scenario(horizon=8, penetration=0.05)


@pytest.fixture(scope="module")
def scene(sc):
    return near_miss_scene(sc.linearizer, sc.reference, sc.q_now)


def _solve(sc, scene, over=None):
    cfg = sc.config.with_overrides(over) if over else sc.config
    return TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg).solve(
        sc.reference, sc.q_now, scene)


OFF = {"collision": {"enabled": False}}
OFF_NO_JERK = {"collision": {"enabled": False},
               "cost": {"w_smooth": 0.0, "w_continuity": 0.0}}


# --- 1. 기본값 --------------------------------------------------------------------------


def test_collisions_are_on_by_default():
    assert TrajOptConfig().collision.enabled is True
    assert TrajOptConfig.from_dict({}).collision.enabled is True
    assert serve_safe.build_parser().parse_args(BASE_ARGV).no_collision is False


def test_the_default_path_is_untouched(sc, scene):
    """flag 를 안 주면 행이 예전처럼 나온다 — 이 STEP 이 기본 동작을 건드리지 않았다는 증거."""
    rows = sc.linearizer.linearize(sc.reference, sc.q_now, scene, sc.config)
    assert rows.n_rows > 0
    assert TrajOptConfig().collision.enabled


# --- 2. 행을 지우지 않는다 — 같은 모양, 전부 비활성 -------------------------------------


def test_disabling_keeps_the_row_shape_exactly(sc, scene):
    """**희소성을 깨지 않는다.** 그것이 QP 인수분해 재사용의 전제다 (193 ms → 25 ms)."""
    on = sc.linearizer.linearize(sc.reference, sc.q_now, scene, sc.config)
    off = sc.linearizer.linearize(sc.reference, sc.q_now, scene,
                                  sc.config.with_overrides(OFF))
    assert off.value.shape == on.value.shape
    assert off.gradient.shape == on.gradient.shape
    assert off.used.shape == on.used.shape
    assert off.capacity == on.capacity
    # 비활성 행의 내용은 이 클래스가 이미 정의한 padding 상태다.
    assert off.n_rows == 0 and on.n_rows > 0
    assert np.all(np.isinf(off.value)) and np.all(off.gradient == 0.0)
    assert np.all(off.slot == -999)
    assert off.n_considered == 0 and off.budget_bound_steps == ()


def test_all_unused_is_the_same_array_as_a_frame_with_nothing_nearby(sc):
    """"근접 제약이 하나도 없는 프레임" 과 **글자 그대로 같은 배열**이어야 한다."""
    rows = LinearizedRows.all_unused(8, sc.config.reduction.rows_per_step,
                                     sc.linearizer.nq_opt)
    assert rows.n_rows == 0
    assert rows.value.shape == (8, sc.config.reduction.rows_per_step)
    assert np.all(np.isinf(rows.value))
    assert rows.used.dtype == bool and not rows.used.any()


# --- 3. 최적화기가 충돌을 보지 않는다 ---------------------------------------------------


def test_with_collisions_off_the_solution_stops_avoiding(sc, scene):
    """켠 판은 관통을 고치고, 끈 판은 안 고친다 — 그것이 이 flag 의 내용 전부다.

    **끈 판의 움직임은 limit 이 요구하는 만큼을 뺀 뒤에 잰다** (T31). T31 부터 position 범위가
    MJCF 에서 오고 (`left_arm_3` 상한 0.01 − margin 0.02 = −0.01), 이 fixture 의 reference 는 그
    box 를 10 mrad 넘는다. 끈 판에서는 받아들여지는 후보가 없어 **reference 의 최소 투영**이
    나가므로 (G2-ii) 끈 판은 정확히 그 10 mrad 만 움직인다. 그것은 충돌이 새어 든 것이 아니라
    limit 이다 — 8 절 (`..._the_only_motion_left_is_the_limit`) 이 그 성질을 따로 박는다.
    """
    from AG3S.trajopt.limits import project_to_limits

    on, off = _solve(sc, scene), _solve(sc, scene, OFF)
    required = project_to_limits(sc.reference, sc.limits, sc.q_now[sc.layout.q_indices])
    moved_on = float(np.abs(on.trajectory - sc.reference).max())
    moved_off = float(np.abs(off.trajectory - required).max())
    assert moved_on > 10.0 * moved_off, (
        f"켠 판 {moved_on * 1000:.1f} mm vs 끈 판 {moved_off * 1000:.1f} mm — "
        "끈 판이 여전히 크게 움직이면 충돌이 다른 길로 새어 들어온다")
    assert on.max_violation < off.max_violation, "켠 판이 관통을 줄여야 한다"


def test_the_penalty_is_off_too_not_just_the_rows(sc, scene):
    """행만 비우고 merit 에 위반을 남기면 충돌이 **step 수락 기준**을 통해 계속 해를 민다.

    merit 을 직접 잰다. 궤적을 비교하는 것으로는 이것을 가릴 수 없다 — `w_slack` 은 충돌 slack
    **과 kinematic slack 을 함께** 스케일하므로 (`problem.py:202-205`), 그 값을 바꾸면 충돌이
    꺼져 있어도 해가 움직인다. 실측 0.23 mm. 그래서 궤적이 아니라 merit 이 답할 질문이다.
    """
    from AG3S.trajopt.problem import objective

    penetrating = sc.reference  # 이 fixture 의 reference 는 실제로 관통한다
    cfg_off = sc.config.with_overrides(OFF)
    off = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg_off)
    on = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config)
    off.linearizer.set_attached(None, None)
    on.linearizer.set_attached(None, None)

    cost = objective(penetrating, sc.reference, cfg_off, previous_chunk=None)
    merit_off = off._merit(penetrating, sc.reference, None, sc.q_now, scene)
    merit_on = on._merit(penetrating, sc.reference, None, sc.q_now, scene)

    assert merit_off == pytest.approx(cost), "꺼진 판의 merit 은 **비용 그 자체**여야 한다"
    assert merit_on > merit_off, "켠 판은 관통에 벌점을 물어야 한다"
    # 그 차이가 곧 `w_slack × 위반` 이다 — 꺼진 판에서 사라진 항이 그것임을 확인한다.
    violation = max(0.0, -off.linearizer.full_violation(penetrating, sc.q_now, scene))
    assert violation > 0.0, "이 씬의 reference 가 관통해야 시험이 뜻을 갖는다"
    assert merit_on - merit_off == pytest.approx(sc.config.cost.w_slack * violation, rel=1e-6)


# --- 4. 판정은 거짓말하지 않는다 ---------------------------------------------------------


def test_the_violation_is_still_measured_truthfully(sc, scene):
    """**이 STEP 에서 가장 중요한 부분.** 이 설정의 기록이 "위반 0" 으로 읽히면 안 된다."""
    off = _solve(sc, scene, OFF)
    assert off.max_violation > 0.01, (
        "충돌을 껐으니 이 씬에서는 관통이 남아야 하고, 그 값이 기록에 참값으로 실려야 한다")
    assert off.status.value == "violated", "판정도 그대로 나와야 한다"
    assert off.metrics["max_violation_pair"] is not None, "최악 행의 신원도 살아 있다"


def test_the_fact_is_in_the_notes_and_in_the_metrics(sc, scene):
    """사람이 읽는 쪽(`notes`)과 기계가 세는 쪽(`metrics`) **둘 다** 있어야 한다."""
    off = _solve(sc, scene, OFF)
    shout = [n for n in off.notes if "COLLISION CONSTRAINTS WERE OFF" in n]
    assert len(shout) == 1
    assert "still measured" in shout[0], "측정은 살아 있다는 것을 문장이 말해야 한다"
    assert off.metrics["collision_enabled"] is False

    on = _solve(sc, scene)
    assert not [n for n in on.notes if "COLLISION CONSTRAINTS WERE OFF" in n]
    assert on.metrics["collision_enabled"] is True


# --- 5. 구조화된 SQP 수치 (셀 수 있어야 한다) -------------------------------------------


def test_the_iteration_count_and_the_budget_are_structured_values(sc, scene):
    """예전에는 `notes` 의 문장뿐이라 **셀 수 없었다.**"""
    r = _solve(sc, scene)
    m = r.metrics
    assert m["sqp_iterations"] == r.iterations
    assert m["max_iterations"] == sc.config.sqp.max_iterations
    assert m["time_budget_ms"] == pytest.approx(sc.config.sqp.time_budget_ms)
    assert m["time_budget_hit"] is False, "이 fixture 의 예산은 넉넉하다"
    assert isinstance(m["max_iterations_hit"], bool)


def test_a_tight_budget_is_reported_as_a_value_not_only_as_a_sentence(sc, scene):
    r = _solve(sc, scene, {"sqp": {"time_budget_ms": 1e-6}})
    assert r.metrics["time_budget_hit"] is True
    assert r.metrics["sqp_iterations"] >= 1, "예산이 0 이어도 한 번은 돈다 (최소 단위)"
    # 문장도 그대로 남는다 — 사람이 읽는 쪽을 없애지 않았다.
    assert any("time budget" in n for n in r.notes)


def test_hitting_the_iteration_cap_is_reported(sc, scene):
    r = _solve(sc, scene, {"sqp": {"max_iterations": 1, "step_tolerance": 0.0}})
    assert r.metrics["max_iterations_hit"] is True
    assert r.metrics["sqp_iterations"] == 1


def test_the_passthrough_result_carries_the_same_keys(sc):
    """읽는 쪽이 `.get` 두 갈래로 갈리지 않아야 한다 — 키 집합이 같다."""
    full = _solve(sc, near_miss_scene(sc.linearizer, sc.reference, sc.q_now))
    passthrough = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, sc.config).solve(
        sc.reference, sc.q_now, None)
    for key in ("sqp_iterations", "max_iterations", "time_budget_ms", "time_budget_hit",
                "max_iterations_hit", "collision_enabled"):
        assert key in passthrough.metrics, key
        assert key in full.metrics, key


# --- 6. 와이어와 클라이언트 -------------------------------------------------------------


def test_the_to_block_rides_on_the_wire():
    verdict = wire.SafetyVerdict(ag3s_status="ok", geometry_certified=True,
                                 trajopt_status="optimal", max_violation_m=0.0, safe=True)
    block = {"iterations": 3, "time_budget_hit": False, "collision_enabled": False}
    out = wire.pack_response(np.zeros((4, wire.ACTION_WIDTH)), verdict, seq=1, timing_ms={},
                             to=block)
    assert wire.unpack_to(out) == block
    # **없으면 키가 안 실린다** (`ag3s` 와 같은 규약) — 옛 서버와 구별된다.
    plain = wire.pack_response(np.zeros((4, wire.ACTION_WIDTH)), verdict, seq=1, timing_ms={})
    assert wire.TO_BLOCK not in plain
    assert wire.unpack_to(plain) == {}


def test_the_safe_policy_fills_the_block_from_the_optimizer_result():
    """배선을 본다 — `refiner.last_result` 의 값이 응답 블록으로 그대로 나간다.

    합성 씬(카메라 0 대)에서는 refiner 가 결과를 안 남기므로(아래 시험), 결과를 하나 놓고
    블록을 만든다. 이 시험이 묻는 것은 최적화기의 동작이 아니라 **값이 와이어까지 가는가** 다.
    """
    import types as _types

    from tests.trajopt.test_safe_policy import _safe_policy

    sp = _safe_policy()
    sp.refiner.last_result = _types.SimpleNamespace(
        status=_types.SimpleNamespace(value="feasible"), iterations=4,
        solve_time_ms=12.5, reference_deviation=0.031,
        metrics={"sqp_iterations": 4, "qp_iterations": 17, "max_iterations": 6,
                 "time_budget_ms": 50.0, "time_budget_hit": True,
                 "max_iterations_hit": False, "collision_enabled": False})
    block = sp._to_block()
    assert block["iterations"] == 4 and block["sqp_iterations"] == 4
    assert block["qp_iterations"] == 17
    assert block["time_budget_hit"] is True and block["max_iterations_hit"] is False
    assert block["collision_enabled"] is False
    assert block["solve_ms"] == pytest.approx(12.5)
    assert block["reference_deviation"] == pytest.approx(0.031)


def test_no_optimizer_result_means_no_block():
    """**없음을 지어내지 않는다.** refiner 가 아직 안 돌았으면 키가 아예 없다 (`ag3s` 와 같은 규약)."""
    from tests.trajopt.test_safe_policy import _shadow_policy

    sp = _shadow_policy(np.zeros((5, wire.ACTION_WIDTH)))  # refine 을 lambda 로 바꿔 둔 판
    out = sp.infer({"state": np.zeros(wire.ACTION_WIDTH), wire.PREFIX + "cameras": []})
    assert wire.TO_BLOCK not in out
    assert wire.unpack_to(out) == {}


def test_the_client_keeps_the_block_and_clears_it_on_a_hold():
    from tests.trajopt.test_closed_loop_reference import _client, _infer, _response

    out = _response(shadow=False)
    out[wire.TO_BLOCK] = {"iterations": 2, "collision_enabled": False}
    client = _client(out, local_shadow=False)
    _infer(client)
    assert client.last_to == {"iterations": 2, "collision_enabled": False}


# --- 7. 서버 flag -----------------------------------------------------------------------


def test_the_flag_reaches_the_config():
    args = serve_safe.build_parser().parse_args(BASE_ARGV + ["--no-collision"])
    assert args.no_collision is True
    cfg = TrajOptConfig.from_dict({"collision": {"enabled": not args.no_collision}})
    assert cfg.collision.enabled is False


def test_the_flag_is_shouted_at_startup(caplog):
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_collision_switch(False)
    assert "COLLISION CONSTRAINTS ARE OFF" in caplog.text
    assert "--no-safe" in caplog.text, "두 flag 의 차이를 말해야 한다"
    assert "max_violation_m" in caplog.text, "측정은 살아 있다는 것을 말해야 한다"
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=serve_safe.__name__):
        serve_safe.announce_collision_switch(True)
    assert caplog.text == "", "켜져 있으면 조용하다"


def test_the_flag_is_refused_together_with_no_safe():
    ap = serve_safe.build_parser()
    args = ap.parse_args(BASE_ARGV + ["--no-collision", "--no-safe"])
    with pytest.raises(SystemExit):
        serve_safe.reject_bad_flag_combinations(ap, args)


def test_the_help_says_it_is_not_no_safe():
    text = serve_safe.build_parser().format_help()
    assert "--no-collision" in text


# --- 8. 판정표의 세 번째 줄 — reference 가 limit 밖이면 그만큼은 움직인다 ---------------


def test_with_no_collision_and_no_jerk_the_only_motion_left_is_the_limit(sc, scene):
    """**ablation 을 읽는 사람이 꼭 알아야 할 것.**

    충돌도 jerk 도 연속성도 없으면 남는 목적은 추적뿐이므로 해가 reference 여야 한다 — 그런데
    이 fixture 에서는 **움직인다.** 이유가 목적함수가 아니다: reference 자체가 joint box 밖에 있고
    해는 그 초과를 0 으로 만든다. URDF 범위(T31 전)에서는 2.547 mrad (`left_arm_1` 하한), T31 의
    MJCF 범위에서는 10 mrad (`left_arm_3` 상한 0.01 − margin 0.02). T31 부터 이 해는 QP 후보가
    아니라 **reference 의 최소 투영**이다 (G2-ii) — 초과한 원소만, 초과한 만큼만 움직인다.

    즉 *"충돌을 껐는데도 청크가 바뀌었다"* 를 곧바로 목적함수 탓으로 읽으면 안 된다. 먼저
    reference 의 limit 초과량을 봐야 한다.
    """
    over_ref = limit_report(sc.reference, sc.limits)
    r = _solve(sc, scene, OFF_NO_JERK)
    over_sol = limit_report(r.trajectory, sc.limits)

    assert over_ref["position"] > 0.0, "이 fixture 의 reference 가 joint box 밖이어야 한다"
    # QP 는 box 를 soft row 로 다루므로 수십 nm 의 잔차가 남는다. 기준은 1 µm 다.
    assert over_sol["position"] == pytest.approx(0.0, abs=1e-6), "해는 box 안으로 들어온다"
    moved = float(np.abs(r.trajectory - sc.reference).max())
    # 움직인 양이 초과량과 **같다** — 그것이 "남은 것은 limit 뿐" 의 수치적 내용이다.
    assert moved == pytest.approx(over_ref["position"], rel=0.02)


def test_a_feasible_reference_comes_back_untouched(sc, scene):
    """세 번째 줄의 대조군. limit 안의 reference 는 **그대로** 돌아온다."""
    # `JointLimits` 는 이미 최적화 공간(nq_opt)이다 — 모델 인덱스로 다시 자르면 안 된다.
    inside = np.clip(sc.reference, sc.limits.lower[:, None] + 1e-3,
                     sc.limits.upper[:, None] - 1e-3)
    assert limit_report(inside, sc.limits)["position"] == pytest.approx(0.0, abs=1e-9)
    cfg = sc.config.with_overrides(OFF_NO_JERK)
    r = TrajectoryOptimizer(sc.robot, sc.layout, sc.limits, cfg).solve(inside, sc.q_now, scene)
    assert np.abs(r.trajectory - inside).max() < 1e-3, (
        "충돌도 jerk 도 없고 reference 가 실행 가능하면 최적해는 reference 그 자체다")
