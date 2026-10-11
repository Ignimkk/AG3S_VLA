"""T22 — attach 는 "닫으라고 했다" 가 아니라 "잡았다" 에 붙는다.

`_run_latch()` 는 계획 청크의 gripper 값이 문턱 아래면 `attach()` 를 불렀다. 실제로 닫혔는지,
손이 대상 근처인지 보지 않았다 — T14 seq 26 은 손에서 636 mm 떨어진 cluster 를 `ee_finger_l1` 에
붙였고, T16 은 HOLD 로 취소된 닫힘에 attach 됐다. 이 파일이 고정하는 것:

1. **잠금 (`GraspLatch(evidence=True)`)** — 대상 확정 → 닫힘 시도 → 파지 확인 → 놓임 확인. 파지 확인은
   실행된 닫힘 명령 · 정착 · 개도 감소 · 명령보다 위에서 막힘 · 조작 대상 · reach 가 **모두** 맞을 때.
   T17 의 실제 개도 수열, T16 (취소된 닫힘), 636 mm, 빈손, 파지 후 HOLD, 놓임.
   **T30 F4:** 정착(`settled`)은 "닫힘 명령 스텝 누적" 이 아니라 **측정 개도가 멈췄다** — 연속 두 요청의
   측정 개도 차 < `settle_epsilon`. 그래서 개도가 한 요청 만에 1.0 → 0.715 로 뛴 요청에서는 attach 하지
   않고, 같은 개도를 한 번 더 본 요청에서 attach 한다 (아래 테스트 대부분이 그 한 요청을 더 준다).
2. **마지막 관문 (`AG3S.attach(reach=)`)** — 잠금이 틀려도 멀리 있는 물체는 붙지 않는다.
3. **SafePolicy** — 파지 상태 갱신이 `process_multi_debug` **앞**에서 일어나고, 기록에
   `grasp: {state, evidence, state_version, constraint_version}` 이 실린다. 옛 클라이언트는 attach 없음.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt import wire
from AG3S.trajopt.grasp_latch import (
    GraspLatch,
    GraspPhase,
    GraspSignal,
    LatchConfig,
    grasp_signal_from_feedback,
)

N_ARM = 7


def _fb(seq, left_applied, measured_left, *, executed=None, right_measured=1.0):
    """8 스텝 청크 하나의 실행 피드백 (왼손 적용값 `left_applied`, 촬영 시점 측정 `measured_left`)."""
    applied = np.asarray(left_applied, np.float64).reshape(-1)
    n = applied.shape[0]
    flags = [True] * n if executed is None else list(executed)
    held = not all(flags)
    blob = wire.make_exec_feedback(
        seq=seq, t_step_start=(seq - 1) * n, executed=flags,
        executed_chunk="none" if not any(flags) else "reference",
        ipc="unsafe" if held else "ok", hold_kind="unsafe" if held else None,
        hold_reason="TO says violated" if held else None,
        planned_gripper=np.column_stack([applied, np.ones(n)]),
        applied_gripper=np.column_stack([applied, np.ones(n)]),
        applied_arm=np.zeros((n, 2 * N_ARM)),
        measured_gripper=(measured_left, right_measured))
    return wire.unpack_exec_feedback({wire.EXEC_FEEDBACK: blob})


def _signal(*args, **kw) -> GraspSignal:
    return grasp_signal_from_feedback(_fb(*args, **kw), 0, 0.85)


def _latched(**cfg) -> GraspLatch:
    latch = GraspLatch(LatchConfig(**cfg), evidence=True)
    for _ in range(3):
        latch.update(label="apple", score=0.9, runner_up=0.1, gripper=1.0)
    assert latch.phase is GraspPhase.LATCHED
    return latch


def _observe(latch, signal, reach=0.055, mid=0, state="visible", placed=False):
    return latch.observe_grasp(signal, reach_m=reach, manipulated_id=mid,
                               manipulated_state=state, placed=placed)


#: T17 (성공 파지) 실측 — `outputs/live_test/20260928_t17/shadow_ep1807/frames.jsonl`.
#: (요청 t_step, 직전 청크의 왼손 적용 명령 8 개 = 실행된 reference 행, 촬영 시점 측정 개도 qpos[27]/−0.045)
T17 = [
    (104, [1.0] * 8, 1.012),
    (112, [1.0] * 8, 1.009),
    (120, [1.0, 0.99, 0.99, 0.0, -0.0, 0.0, 0.01, 0.01], 0.715),
    (128, [0.0, -0.0, -0.0, 0.0, 0.02, 0.56, 0.58, 0.58], 0.720),
    (136, [0.57] * 8, 0.720),
    (144, [0.58] * 8, 0.721),
]


# --- 1. 잠금 ------------------------------------------------------------------------------


def test_t17_attaches_once_when_the_measured_opening_stops_on_the_apple():
    """T30 F4: attach 는 t=120 (개도가 1.009 → 0.715 로 **움직인** 요청) 이 아니라 t=128 (0.715 → 0.720,
    멈춘 뒤) 이다. T22 에서는 t=120 이었다 — 닫힘 명령 스텝 수(5 ≥ 3)만 봤기 때문이다."""
    latch = _latched()
    events = [(t, _observe(latch, _signal(i + 1, applied, measured)))
              for i, (t, applied, measured) in enumerate(T17)]
    assert [t for t, ev in events if ev.attach] == [128]
    moving = dict(events)[120]
    assert not moving.attach and moving.phase is GraspPhase.CLOSING
    assert moving.evidence["checks"]["settled"] is False
    assert [k for k, v in moving.evidence["checks"].items() if not v] == ["settled"]
    assert moving.evidence["opening_delta"] == pytest.approx(0.294)
    assert "아직 움직인다" in moving.note
    ev = dict(events)[128]
    assert ev.phase is GraspPhase.HELD
    assert ev.evidence["opening_before"] == pytest.approx(1.009)
    assert ev.evidence["opening_after"] == pytest.approx(0.720)
    assert ev.evidence["opening_delta"] == pytest.approx(0.005)
    assert ev.evidence["settle_epsilon"] == pytest.approx(0.02)
    assert ev.evidence["reach_mm"] == pytest.approx(55.0)
    assert ev.evidence["feedback_seq"] == 4
    assert all(ev.evidence["checks"].values())
    assert not any(e.detach for _, e in events)


def test_t16_a_closure_cancelled_by_hold_is_not_evidence():
    """닫힘이 제안됐지만 HOLD — 적용값은 현재 상태이고, 빈손 HOLD 의 측정 0.82 도 문턱 아래다."""
    latch = _latched()
    for seq in range(1, 6):
        ev = _observe(latch, _signal(seq, [0.82] * 8, 0.82, executed=[False] * 8))
        assert not ev.attach
        assert latch.phase is GraspPhase.LATCHED        # 닫힘 시도조차 아니다
        assert ev.evidence["closing_steps"] == 0


def test_the_636_mm_phantom_is_refused_by_reach_whatever_the_opening_says():
    latch = _latched()
    _observe(latch, _signal(1, [1.0] * 8, 1.0), reach=0.636)
    _observe(latch, _signal(2, [0.0] * 8, 0.715), reach=0.636)
    ev = _observe(latch, _signal(3, [0.0] * 8, 0.715), reach=0.636)      # 멈춘 뒤
    assert not ev.attach and latch.phase is GraspPhase.CLOSING
    assert ev.evidence["checks"]["reach"] is False
    assert [k for k, v in ev.evidence["checks"].items() if not v] == ["reach"]
    assert "reach" in ev.note
    assert ev.evidence["reach_mm"] == pytest.approx(636.0)


@pytest.mark.parametrize("command, measured", [(0.0, 0.004), (0.57, 0.575), (0.0, 0.03)])
def test_an_empty_hand_closes_to_the_command_and_is_not_attached(command, measured):
    latch = _latched()
    _observe(latch, _signal(1, [1.0] * 8, 1.0))
    for seq in range(2, 6):
        ev = _observe(latch, _signal(seq, [command] * 8, measured))
        assert not ev.attach
    assert latch.phase is GraspPhase.CLOSING
    assert ev.evidence["checks"]["blocked"] is False
    assert "빈손" in ev.note


def test_a_closure_at_the_end_of_a_chunk_waits_until_it_has_settled():
    latch = _latched()
    _observe(latch, _signal(1, [1.0] * 8, 1.0))
    ev = _observe(latch, _signal(2, [1.0] * 7 + [0.0], 0.80))     # 1 스텝만 닫힘 명령
    assert not ev.attach and ev.evidence["checks"]["settled"] is False
    ev = _observe(latch, _signal(3, [0.0] * 8, 0.715))             # 아직 움직였다 (0.80 → 0.715)
    assert not ev.attach and ev.evidence["checks"]["settled"] is False
    ev = _observe(latch, _signal(4, [0.0] * 8, 0.715))             # 멈췄다
    assert ev.attach
    assert ev.evidence["opening_before"] == pytest.approx(1.0)   # 시도 직전 개도


def test_after_the_grasp_a_hold_keeps_it_and_only_an_opening_releases():
    latch = _latched()
    _observe(latch, _signal(1, [1.0] * 8, 1.0))
    assert not _observe(latch, _signal(2, [0.0] * 8, 0.715)).attach
    assert _observe(latch, _signal(3, [0.0] * 8, 0.715)).attach
    for seq in range(4, 8):                                        # 파지 후 HOLD
        ev = _observe(latch, _signal(seq, [0.72] * 8, 0.72, executed=[False] * 8))
        assert not ev.detach and latch.phase is GraspPhase.HELD
    ev = _observe(latch, _signal(8, [1.0] * 8, 0.98))              # 열림 1 프레임
    assert not ev.detach
    ev = _observe(latch, _signal(9, [1.0] * 8, 1.0))               # 열림 2 프레임
    assert ev.detach and latch.phase is GraspPhase.PLACED


def test_placed_releases_only_once_the_measured_opening_is_open():
    latch = _latched()
    _observe(latch, _signal(1, [1.0] * 8, 1.0))
    _observe(latch, _signal(2, [0.0] * 8, 0.715))
    assert _observe(latch, _signal(3, [0.0] * 8, 0.715)).attach
    assert not _observe(latch, _signal(4, [0.57] * 8, 0.72), placed=True).detach
    assert _observe(latch, _signal(5, [1.0] * 8, 0.95), placed=True).detach


def test_without_feedback_nothing_attaches_and_the_note_says_why():
    latch = _latched()
    for _ in range(4):
        ev = _observe(latch, grasp_signal_from_feedback(
            wire.unpack_exec_feedback({}), 0, 0.85))
        assert not ev.attach
    assert latch.phase is GraspPhase.LATCHED
    assert "실행 피드백 없음" in ev.note and "predates T18" in ev.note
    assert "계획값" in ev.note


def test_a_reopened_gripper_abandons_the_attempt_and_a_new_one_can_succeed():
    latch = _latched()
    _observe(latch, _signal(1, [1.0] * 8, 1.0))
    _observe(latch, _signal(2, [0.0] * 8, 0.01))                   # 빈손 닫힘
    assert latch.phase is GraspPhase.CLOSING
    ev = _observe(latch, _signal(3, [1.0] * 8, 1.0))                # 다시 연다
    assert latch.phase is GraspPhase.LATCHED and "포기" in ev.note
    assert not _observe(latch, _signal(4, [0.0] * 8, 0.715)).attach   # 1.0 → 0.715: 움직이는 중
    assert _observe(latch, _signal(5, [0.0] * 8, 0.715)).attach


def test_no_confirmed_target_or_a_lost_object_attaches_nothing():
    searching = GraspLatch(LatchConfig(), evidence=True)
    _observe(searching, _signal(1, [1.0] * 8, 1.0))
    ev = _observe(searching, _signal(2, [0.0] * 8, 0.715))
    assert not ev.attach and searching.phase is GraspPhase.SEARCHING

    for mid, state in ((None, None), (0, "lost")):
        latch = _latched()
        _observe(latch, _signal(1, [1.0] * 8, 1.0))
        ev = _observe(latch, _signal(2, [0.0] * 8, 0.715), mid=mid, state=state)
        assert not ev.attach and ev.evidence["checks"]["manipulated"] is False


def test_revert_attach_returns_to_closing_and_keeps_the_reason():
    latch = _latched()
    _observe(latch, _signal(1, [1.0] * 8, 1.0))
    _observe(latch, _signal(2, [0.0] * 8, 0.715))
    assert _observe(latch, _signal(3, [0.0] * 8, 0.715)).attach
    ev = latch.revert_attach("reach: 300 mm")
    assert latch.phase is GraspPhase.CLOSING and ev.evidence["rejected"] == "reach: 300 mm"
    assert _observe(latch, _signal(4, [0.0] * 8, 0.715)).attach   # 다음 프레임에 다시 판정


def test_in_evidence_mode_the_gripper_value_alone_never_attaches():
    latch = _latched()
    ev = latch.update(label="apple", score=0.9, runner_up=0.1, gripper=0.02)
    assert not ev.attach and latch.phase is GraspPhase.LATCHED
    with pytest.raises(RuntimeError):
        GraspLatch().observe_grasp(GraspSignal(available=False))


def test_hold_steps_are_not_commands():
    """HOLD 스텝의 적용값은 `rby1_state()` (현재 개도) 다 — 쥔 채 HOLD 면 0.72 라 닫힘처럼 보인다."""
    s = _signal(1, [1.0, 1.0, 0.72, 0.72, 0.72, 0.72, 0.72, 0.72], 0.72,
                executed=[True, True] + [False] * 6)
    assert s.n_exec == 2 and s.closing_steps == 0 and s.last_command == pytest.approx(1.0)
    right = grasp_signal_from_feedback(_fb(1, [0.0] * 8, 0.7, right_measured=0.9), 1, 0.85)
    assert right.opening == pytest.approx(0.9) and right.closing_steps == 0


def test_the_latch_and_the_pipeline_gate_share_the_reach_default():
    from AG3S.ag3s.constraints.attached import DEFAULT_ATTACH_REACH_M

    cfg = LatchConfig()
    assert cfg.reach == DEFAULT_ATTACH_REACH_M
    assert cfg.reach < 0.636
    assert (cfg.gripper_closed_below, cfg.confirm_frames) == (0.85, 3)   # 기본 문턱 유지
    assert cfg.legacy_gripper_attach is False


# --- 1b. T30 F4: `settled` = 측정 개도가 멈췄다 ----------------------------------------------

#: T28 E3b ep1808 r1 (`outputs/verify/T28/E3b_rec/E3b_ep1808_r1/frames.jsonl`) — 옛 규칙이 t=120 에
#: attach 했고 사과는 들리지 않았다 (t=240 detach). (요청 t, 직전 청크 왼손 적용 명령, 촬영 시점 측정 개도).
#: 닫힘 명령 0.006 이 t=114–119 에 적용되는 동안 개도는 0.997 → 0.364 로 **움직이는 중**이었고, 그 뒤
#: 정책 명령 ≈ 0.43 을 개도가 그대로 따라갔다 (위치 제어 — 막힌 것이 없다).
E3B_1808_R1 = [
    (112, [0.997, 0.996, 0.996, 0.998, 0.996, 0.998, 0.998, 0.997], 0.997),
    (120, [1.0, 0.982, 0.006, 0.009, 0.008, 0.006, 0.012, 0.006], 0.3641),
    (128, [0.434, 0.434, 0.431, 0.433, 0.434, 0.434, 0.432, 0.43], 0.4309),
    (136, [0.432, 0.437, 0.439, 0.441, 0.441, 0.439, 0.437, 0.44], 0.4389),
    (144, [0.432, 0.43, 0.434, 0.43, 0.434, 0.434, 0.433, 0.436], 0.4334),
    (152, [0.439, 0.437, 0.438, 0.44, 0.441, 0.443, 0.44, 0.438], 0.44),
    (160, [0.437, 0.436, 0.437, 0.437, 0.438, 0.437, 0.437, 0.435], 0.4369),
    (168, [0.428, 0.431, 0.434, 0.436, 0.438, 0.438, 0.44, 0.442], 0.4378),
    (176, [0.43, 0.435, 0.434, 0.434, 0.437, 0.437, 0.439, 0.439], 0.4372),
    (184, [0.415, 0.416, 0.416, 0.418, 0.421, 0.42, 0.418, 0.415], 0.4186),
    (192, [0.439, 0.439, 0.438, 0.442, 0.445, 0.447, 0.445, 0.443], 0.4433),
    (200, [0.414, 0.419, 0.422, 0.422, 0.425, 0.421, 0.417, 0.417], 0.4205),
    (208, [0.435, 0.436, 0.447, 0.439, 0.445, 0.446, 0.444, 0.442], 0.4429),
    (216, [0.442] * 8, 0.4421, [False] * 8),
    (224, [0.442, 0.441, 0.443, 0.441, 0.462, 0.989, 0.993, 0.997], 0.7495),
    (232, [0.591, 0.577, 0.578, 0.991, 0.998, 0.998, 0.999, 0.998], 0.9381),
    (240, [1.0, 1.0, 1.0, 0.998, 0.999, 0.998, 0.997, 0.997], 0.9987),
]

#: T22 verify #4e — MuJoCo 빈손 닫힘 sweep 중 **옛 규칙이 attach 한 10 건** (`outputs/verify/T22/
#: empty_sweep_t22.json`). (닫힘 목표, 청크 1 안에서 닫힘이 시작된 스텝, 요청 경계의 측정 개도 4 개).
#: 옛 규칙은 요청 1 (손가락이 청크 경계에서 아직 움직이는 중, 0.16–0.76) 에서 attach 했다.
EMPTY_SWEEP_OLD_ATTACH = [
    (0.0, 3, [1.0, 0.163, 0.0001, 0.0]),
    (0.0, 4, [1.0, 0.2801, 0.0001, 0.0]),
    (0.0, 5, [1.0, 0.4472, 0.0001, 0.0]),
    (0.3, 2, [1.0, 0.3614, 0.2989, 0.3]),
    (0.3, 3, [1.0, 0.4141, 0.2985, 0.3]),
    (0.3, 4, [1.0, 0.4961, 0.2982, 0.3]),
    (0.3, 5, [1.0, 0.613, 0.2983, 0.2999]),
    (0.57, 3, [1.0, 0.6401, 0.5691, 0.57]),
    (0.57, 4, [1.0, 0.6905, 0.5689, 0.57]),
    (0.57, 5, [1.0, 0.7623, 0.569, 0.57]),
]


def test_settled_is_the_measured_opening_standing_still_not_a_command_count():
    cfg = LatchConfig()
    assert cfg.settle_epsilon == pytest.approx(0.02)
    assert not hasattr(cfg, "settle_steps")                        # 명령 스텝 수는 더 이상 문턱이 아니다
    latch = _latched()
    _observe(latch, _signal(1, [1.0] * 8, 1.0))
    # 닫힘 명령 8 스텝 전부 실행 — 옛 규칙이면 정착. 그러나 개도는 1.0 → 0.60 → 0.70 로 움직였다.
    ev = _observe(latch, _signal(2, [0.0] * 8, 0.60))
    assert not ev.attach and ev.evidence["checks"]["settled"] is False
    assert ev.evidence["settled_steps"] == 8                       # 기록에는 남는다 (판정에는 안 쓴다)
    ev = _observe(latch, _signal(3, [0.0] * 8, 0.70))
    assert not ev.attach and ev.evidence["opening_delta"] == pytest.approx(0.10)
    ev = _observe(latch, _signal(4, [0.0] * 8, 0.715))             # 차 0.015 < 0.02
    assert ev.attach and ev.evidence["checks"]["settled"] is True


def test_settled_is_false_when_there_is_no_previous_measurement():
    latch = _latched()
    ev = _observe(latch, _signal(1, [0.0] * 8, 0.715))              # 첫 피드백 — 비교할 직전 개도가 없다
    assert latch.phase is GraspPhase.CLOSING and not ev.attach
    assert ev.evidence["opening_delta"] is None and ev.evidence["checks"]["settled"] is False
    ev = _observe(latch, _signal(2, [0.0] * 8, 0.715))
    assert ev.evidence["checks"]["settled"] is True                # 이제 비교할 직전 개도가 있다
    assert ev.evidence["checks"]["dropped"] is False and not ev.attach   # 시도 직전 개도가 없다 (T22, fail-closed)


def test_e3b_1808_r1_moving_fingers_and_a_tracked_partial_command_never_attach():
    """옛 규칙의 거짓 attach (t=120). 새 규칙: t=120·128 은 움직이는 중, t=136 부터는 멈췄지만
    개도 ≈ 명령이라 `blocked` 가 거절한다 (빈손/쥔 손 구별은 T22 그대로)."""
    latch = _latched()
    events = {}
    for i, row in enumerate(E3B_1808_R1):
        t, applied, measured = row[:3]
        executed = row[3] if len(row) > 3 else None
        events[t] = _observe(latch, _signal(i + 1, applied, measured, executed=executed))
    assert [t for t, ev in events.items() if ev.attach] == []
    assert events[120].evidence["checks"]["settled"] is False
    assert events[120].evidence["opening_delta"] == pytest.approx(0.6329)
    assert events[128].evidence["checks"]["settled"] is False     # 0.364 → 0.431 (명령을 따라 다시 벌어짐)
    assert events[136].evidence["checks"]["settled"] is True      # 0.431 → 0.439
    assert events[136].evidence["checks"]["blocked"] is False     # 개도 0.439 − 명령 0.44 < 0.05
    assert "빈손" in events[136].note
    assert latch.phase is GraspPhase.LATCHED                        # t=232 에 다시 열려 시도 포기
    assert "포기" in events[232].note


@pytest.mark.parametrize("target, start, openings", EMPTY_SWEEP_OLD_ATTACH)
def test_the_t22_empty_hand_sweep_cases_that_used_to_attach_do_not(target, start, openings):
    latch = _latched()
    chunks = [[1.0] * 8, [1.0] * start + [target] * (8 - start), [target] * 8, [target] * 8]
    evs = [_observe(latch, _signal(i + 1, cmd, meas))
           for i, (cmd, meas) in enumerate(zip(chunks, openings))]
    assert not any(ev.attach for ev in evs)
    assert evs[1].evidence["checks"]["settled"] is False           # 옛 규칙이 attach 한 요청
    assert evs[1].evidence["opening_delta"] >= 0.2
    # 멈춘 뒤에는 T22 의 빈손 판정(명령까지 닫힘 / 끝까지 닫힘)이 거절한다.
    assert evs[3].evidence["checks"]["settled"] is True
    failed = [k for k, v in evs[3].evidence["checks"].items() if not v]
    assert failed and set(failed) <= {"blocked", "not_empty"}


# --- 2. AG3S.attach 의 reach 관문 ----------------------------------------------------------


@pytest.fixture(scope="module")
def arm():
    pytest.importorskip("casadi")
    from tests.ag3s.fixtures import PlanarTwoLinkArm

    return PlanarTwoLinkArm()


def _target_at(centre, radius=0.03):
    from AG3S.ag3s.stages.geometry import fit_sphere
    from AG3S.ag3s.types import TargetGeometry

    points = np.asarray(centre, np.float64) + np.random.default_rng(0).normal(
        0.0, radius / 3.0, (60, 3))
    return TargetGeometry(
        id=7, points=points, point_indices=np.arange(60), centroid=points.mean(axis=0),
        bounding_geometry=fit_sphere(points), attention_score=0.9, confidence=0.9)


def _ag3s(arm):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from tests.ag3s.fixtures import FINGERTIP_LINK

    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 3,
                                                      "max_attached_primitives": 2}})
    return AG3S(cfg, robot_model=arm, attached_parent_links=(FINGERTIP_LINK,))


def test_attach_refuses_a_target_beyond_reach_and_changes_nothing(arm):
    from AG3S.ag3s.constraints.attached import AttachRejected
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, FINGERTIP_LINK

    ag3s = _ag3s(arm)
    tip = arm.end_effector_numeric(DEFAULT_ARM_Q)
    far = _target_at(tip + np.array([0.636, 0.0, 0.0]))
    with pytest.raises(AttachRejected) as info:
        ag3s.attach(far, robot_state=DEFAULT_ARM_Q, parent_link=FINGERTIP_LINK, reach=0.12)
    assert info.value.reason.startswith("reach")
    assert info.value.reach_m == pytest.approx(0.636, abs=0.01)
    assert isinstance(info.value, ValueError)                     # 옛 호출자도 잡는다
    assert info.value.record()["limit_mm"] == pytest.approx(120.0)
    assert ag3s.attached is None

    near = ag3s.attach(_target_at(tip + np.array([0.03, 0.0, 0.0])),
                       robot_state=DEFAULT_ARM_Q, parent_link=FINGERTIP_LINK, reach=0.12)
    assert ag3s.attached is near
    # 직접 호출자의 기본(reach=None)은 예전 그대로 — 관문은 SafePolicy 가 값을 넘길 때 선다.
    ag3s.detach()
    assert ag3s.attach(far, robot_state=DEFAULT_ARM_Q,
                       parent_link=FINGERTIP_LINK) is ag3s.attached


def test_attach_refuses_prebuilt_geometry_far_from_its_parent(arm):
    from AG3S.ag3s.constraints.attached import AttachRejected, attached_reach
    from AG3S.ag3s.types import AttachedCollisionGeometry
    from tests.ag3s.fixtures import FINGERTIP_LINK

    ag3s = _ag3s(arm)
    far = AttachedCollisionGeometry(points=np.full((4, 3), 0.4), parent_link=FINGERTIP_LINK,
                                    T_parent_object=np.eye(4), primitives=[],
                                    allowed_contact_links=frozenset())
    assert attached_reach(far) == pytest.approx(np.sqrt(3) * 0.4)
    with pytest.raises(AttachRejected):
        ag3s.attach(far, reach=0.12)
    assert ag3s.attached is None


# --- 3. SafePolicy ------------------------------------------------------------------------


def _policy(**kw):
    from tests.trajopt.test_safe_policy import _StubPolicy, _safe_policy

    return _safe_policy(_StubPolicy(), **kw)


def _manipulated(centroid, geometry_centroid=None, state="visible", mid=0):
    from AG3S.ag3s.stages.target_grounding import ManipulatedIdentity

    geometry = _target_at(centroid if geometry_centroid is None else geometry_centroid)
    return ManipulatedIdentity(id=mid, state=state, geometry=geometry,
                               centroid=np.asarray(centroid, np.float64), age_frames=0,
                               last_seen_frame=4, last_seen_time=1.0, switched_from=None,
                               frame=4)


def _finger_origin(sp, q):
    return np.asarray(sp.constraint_robot_model.link_pose(q, "ee_finger_l1"))[:3, 3]


def _grasp_frames(sp, q, feedbacks):
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q)]
    for fb in feedbacks:
        sp._pending = {"scene": {}, "exec_feedback": fb}               # noqa: SLF001
        sp._advance_grasp(obs, ["left"])                               # noqa: SLF001
        sp._grasp_constraint_version = sp._grasp_state_version         # noqa: SLF001
        sp._finish_grasp_record(types.SimpleNamespace(attached=sp.ag3s.attached))  # noqa: SLF001
        yield dict(sp._last_grasp_record)                              # noqa: SLF001


def _lock(sp):
    for _ in range(3):
        sp._latch.update(label="obj0", score=0.9, runner_up=0.1, gripper=1.0)  # noqa: SLF001


def test_safe_policy_defaults_to_evidence_and_legacy_is_opt_in():
    assert _policy()._latch.evidence is True                                   # noqa: SLF001
    legacy = _policy(latch=LatchConfig(legacy_gripper_attach=True))
    assert legacy._latch.evidence is False                                     # noqa: SLF001


def rec_hold_link(sp):
    return sp._last_grasp_record.get("hold_link")                          # noqa: SLF001


def test_safe_policy_attaches_the_manipulated_object_on_evidence_and_records_it():
    sp = _policy()
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    apple = _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0])
    sp.ag3s._manipulated = _manipulated(apple)                                 # noqa: SLF001
    _lock(sp)
    recs = list(_grasp_frames(sp, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715)]))

    assert sp.ag3s.attached is not None
    # T32 H2: the held object rides the palm (the finger's parent), not the moving finger; the
    # reach below is still measured from the finger (`grasp_links`).
    assert sp.ag3s.attached.parent_link == "ee_left"
    assert rec_hold_link(sp) == "ee_left"
    rec = recs[-1]
    assert rec["state"] == "held" and rec["attach"] and rec["transition"]
    assert rec["evidence"]["reach_mm"] == pytest.approx(50.0, abs=0.5)
    assert rec["evidence"]["opening_before"] == pytest.approx(1.0)
    assert rec["evidence"]["opening_after"] == pytest.approx(0.715)
    assert rec["evidence"]["feedback_seq"] == 3
    assert rec["state_version"] == rec["constraint_version"] == 1
    assert rec["constraints_match_state"] and rec["attached_in_constraints"]
    assert rec["alignment"]["geometry_frame"] == 4
    assert any(n.startswith("grasp: 파지 확인") for n in sp._feedback_notes())  # noqa: SLF001


def test_safe_policy_does_not_attach_a_far_object_or_without_feedback():
    sp = _policy()
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(                                      # noqa: SLF001
        _finger_origin(sp, q) + np.array([0.636, 0.0, 0.0]))
    _lock(sp)
    recs = list(_grasp_frames(sp, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715)]))
    assert sp.ag3s.attached is None
    assert recs[-1]["evidence"]["checks"]["reach"] is False

    old = _policy()
    old.ag3s._manipulated = _manipulated(_finger_origin(old, q))              # noqa: SLF001
    _lock(old)
    recs = list(_grasp_frames(old, q, [wire.unpack_exec_feedback({})] * 3))
    assert old.ag3s.attached is None
    assert "predates T18" in recs[-1]["note"] and recs[-1]["state"] == "latched"


def test_the_pipeline_gate_overrules_the_latch_and_the_latch_steps_back():
    """잠금은 manipulated.centroid (가까움) 를 봤지만 붙일 기하의 centroid 가 멀다 — 마지막 관문이 막는다."""
    sp = _policy()
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    near = _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0])
    sp.ag3s._manipulated = _manipulated(near, geometry_centroid=near + [0.5, 0.0, 0.0])  # noqa: SLF001
    _lock(sp)
    recs = list(_grasp_frames(sp, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715)]))
    assert sp.ag3s.attached is None
    assert sp._latch.phase is GraspPhase.CLOSING                                # noqa: SLF001
    assert recs[-1]["rejected"]["reason"].startswith("reach")
    assert recs[-1]["state_version"] == 0


def test_release_on_measured_opening_detaches_before_the_constraints():
    sp = _policy()
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(                                      # noqa: SLF001
        _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    fbs = [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715),
           _fb(4, [0.72] * 8, 0.72, executed=[False] * 8),                  # 파지 후 HOLD
           _fb(5, [1.0] * 8, 1.0), _fb(6, [1.0] * 8, 1.0)]
    recs = list(_grasp_frames(sp, q, fbs))
    assert [r["attach"] for r in recs] == [False, False, True, False, False, False]
    assert [r["detach"] for r in recs] == [False, False, False, False, False, True]
    assert sp.ag3s.attached is None and recs[-1]["state_version"] == 2


def test_the_grasp_update_runs_before_the_constraints_are_built(monkeypatch):
    """순서 (a): 이번 프레임의 attach 가 이번 프레임의 `process_multi_debug` 에 이미 보인다."""
    sp = _policy()
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(                                      # noqa: SLF001
        _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q, camera_id="zed_left")]
    monkeypatch.setattr(wire, "unpack_camera_observations", lambda scene, **kw: obs)
    seen = []

    class _Stop(Exception):
        pass

    def spy(*args, **kwargs):
        seen.append(sp.ag3s.attached)
        raise _Stop

    monkeypatch.setattr(sp.ag3s, "process_multi_debug", spy)
    for fb in (_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715)):
        sp._pending = {"scene": {}, "exec_feedback": fb}                      # noqa: SLF001
        with pytest.raises(_Stop):
            sp._scene_fn(None)                                                 # noqa: SLF001
    assert seen[0] is None and seen[1] is None and seen[2] is not None
    assert sp._grasp_constraint_version == sp._grasp_state_version == 1       # noqa: SLF001
    # 같은 요청으로 두 번 불려도 (`would_violate`) 닫힘 스텝을 두 번 세지 않는다.
    before = dict(sp._last_grasp_record)                                       # noqa: SLF001
    with pytest.raises(_Stop):
        sp._scene_fn(None)                                                     # noqa: SLF001
    assert sp._last_grasp_record == before                                     # noqa: SLF001


def test_the_chunk_summary_carries_the_grasp_block(tmp_path):
    import json

    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    sp = _policy()
    sp.recorder = ConstraintRecordWriter(tmp_path, esdf_mode="none")
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(                                      # noqa: SLF001
        _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    list(_grasp_frames(sp, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715)]))
    sp._record(2, np.zeros((4, wire.ACTION_WIDTH)), np.zeros((4, wire.ACTION_WIDTH)))  # noqa: SLF001
    path = sorted(sp.recorder.run_dir.glob("chunk_*.npz"))[-1]
    summary = json.loads(str(np.load(path)["summary_json"]))
    grasp = summary["grasp"]
    assert grasp["state"] == "held" and grasp["mode"] == "evidence"
    for key in ("opening_before", "opening_after", "reach_mm", "feedback_seq"):
        assert key in grasp["evidence"]
    assert grasp["state_version"] == grasp["constraint_version"] == 1


def test_legacy_mode_still_goes_through_the_reach_gate():
    """`legacy_gripper_attach=True` (옛 기록 재생) 도 마지막 관문은 지난다 — 거절되면 LATCHED 로."""
    latch = GraspLatch(LatchConfig(confirm_frames=1))
    latch.update(label="apple", score=0.9, runner_up=0.1, gripper=1.0)
    assert latch.update(label="apple", score=0.9, runner_up=0.1, gripper=0.2).attach
    ev = latch.revert_attach("reach: 636 mm")
    assert latch.phase is GraspPhase.LATCHED and "reach" in ev.note
