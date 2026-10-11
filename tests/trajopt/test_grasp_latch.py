"""조작 대상의 잠금 — 걸기 · 유지 · 풀기 (F17).

`target_grounding` 은 무상태라 프레임마다 "지금 attention 이 가리키는 것" 을 새로 고른다. 그런데
실측에서 **정책의 attention 은 파지에 착수하는 순간 목적지로 옮겨간다** (F11). 잠금이 그 사이를
메운다.

아래 `RUN_0004` 는 **실제 기록에서 그대로 옮긴 수열**이다 — `step-04-grounding_004.json` 의
프레임별 선택·점수와 `run_0004` 의 왼손 그리퍼 값. 이 파일이 고정하는 것은 그 수열에서 잠금이
내는 답이다.
"""

import numpy as np
import pytest

from AG3S.trajopt.grasp_latch import CentroidIdentity, GraspLatch, GraspPhase, LatchConfig

#: (프레임, grounding 이 고른 것, 1위 점수, 2위 점수, 왼손 그리퍼). run_0004 실측.
#: 그리퍼는 열림 ≈ 1.00, 닫힘 ≈ 0.72 — 프레임 10 에 닫히고 19 에 열린다.
RUN_0004 = [
    (0, "apple", 0.89, 0.46, 1.000), (1, "apple", 0.89, 0.47, 0.998),
    (2, "apple", 0.71, 0.02, 0.999), (3, "apple", 0.89, 0.49, 1.000),
    (4, "apple", 0.89, 0.49, 0.998), (5, "apple", 0.90, 0.51, 0.999),
    (6, "apple", 0.88, 0.52, 1.000), (7, "crate", 0.35, 0.00, 0.999),
    (8, "crate", 0.37, 0.04, 1.000), (9, "crate", 0.36, 0.04, 0.999),
    (10, "crate", 0.36, 0.05, 0.732), (11, "crate", 0.34, 0.03, 0.739),
    (12, "crate", 0.33, 0.04, 0.738), (13, "crate", 0.34, 0.04, 0.735),
    (14, "crate", 0.31, 0.04, 0.732), (15, "crate", 0.32, 0.04, 0.726),
    (16, "crate", 0.36, 0.05, 0.725), (17, "crate", 0.47, 0.35, 0.722),
    (18, "orange", 0.45, 0.44, 0.722), (19, "crate", 0.65, 0.55, 0.916),
    (20, "crate", 0.68, 0.56, 0.997),
]
PLACED_AT = 19          # 성공 판정 (사과가 바구니 안 + 손에서 이탈), 실측


def _replay(latch, rows, placed_at=PLACED_AT):
    out = []
    for frame, label, score, runner_up, gripper in rows:
        out.append((frame, latch.update(label=label, score=score, runner_up=runner_up,
                                        gripper=gripper, placed=frame >= placed_at)))
    return out


def test_manipulated_object_survives_attention_moving_to_the_destination():
    """이 파일의 이유 — 프레임 7 부터 grounding 은 crate 를 내지만 조작 대상은 apple 로 남는다."""
    latch = GraspLatch()
    log = _replay(latch, RUN_0004)
    during_grasp = [ev.manipulated for frame, ev in log if 7 <= frame <= 18]
    assert set(during_grasp) == {"apple"}


def test_attach_and_detach_fire_once_at_the_measured_frames():
    latch = GraspLatch()
    log = _replay(latch, RUN_0004)
    assert [f for f, ev in log if ev.attach] == [10]      # 그리퍼가 닫히는 프레임
    assert [f for f, ev in log if ev.detach] == [19]      # 성공 판정이 서는 프레임


def test_destination_is_learned_from_attention_after_the_latch():
    """F11 을 뒤집어 쓴다 — 잠금이 걸린 뒤 grounding 이 내는 이름이 목적지다.

    실측: 파지~놓기 구간 14 프레임 중 13 이 `crate` 이고 프롬프트는 `put the apple in the
    basket` 이다. 한 프레임(18)이 `orange` 로 튀지만 잠금이 그것을 먹지 않는다.
    """
    latch = GraspLatch()
    _replay(latch, RUN_0004)
    assert latch.destination == "crate"
    assert latch.manipulated == "apple"


def test_a_single_confident_frame_does_not_latch():
    """걸기에 문턱이 필요한 이유 — 두 기록 모두 프레임 2 에서 purity 가 0.62 로 떨어진다.

    잠금에는 회복 지점이 없으므로, 한 프레임의 확신으로 잠그면 안 된다.
    """
    latch = GraspLatch(LatchConfig(confirm_frames=3))
    ev = latch.update(label="apple", score=0.9, runner_up=0.1, gripper=1.0)
    assert ev.manipulated is None and ev.phase is GraspPhase.SEARCHING
    latch.update(label="banana", score=0.9, runner_up=0.1, gripper=1.0)
    assert latch.manipulated is None


def test_an_ambiguous_frame_neither_latches_nor_breaks_the_streak():
    """1·2위 격차가 없는 프레임은 세지도 않고 연속을 끊지도 않는다."""
    latch = GraspLatch(LatchConfig(confirm_frames=3, score_ratio=1.3))
    latch.update(label="apple", score=0.9, runner_up=0.1)
    latch.update(label="apple", score=0.50, runner_up=0.49)   # 애매 — 무시
    assert latch.manipulated is None
    latch.update(label="apple", score=0.9, runner_up=0.1)
    latch.update(label="apple", score=0.9, runner_up=0.1)
    assert latch.manipulated == "apple"


def test_a_one_frame_gripper_blip_does_not_release():
    """release_frames 가 있는 이유 — 한 프레임 튐으로 쥔 물체를 잃으면 안 된다."""
    latch = GraspLatch(LatchConfig(confirm_frames=1, release_frames=2))
    for _ in range(2):
        latch.update(label="apple", score=0.9, runner_up=0.1, gripper=1.0)
    latch.update(label="apple", score=0.9, runner_up=0.1, gripper=0.7)   # 쥠
    assert latch.holding
    ev = latch.update(label="apple", score=0.9, runner_up=0.1, gripper=1.0)  # 한 프레임 튐
    assert not ev.detach and latch.holding
    ev = latch.update(label="apple", score=0.9, runner_up=0.1, gripper=1.0)
    assert ev.detach and latch.phase is GraspPhase.PLACED


def test_reset_clears_the_episode():
    latch = GraspLatch()
    _replay(latch, RUN_0004)
    latch.reset()
    assert latch.manipulated is None and latch.destination is None
    assert latch.phase is GraspPhase.SEARCHING


def test_centroid_identity_separates_objects_that_jump():
    """grounding 은 이름표를 주지 않는다. target 이 바뀌면 무게중심이 307~597 mm 뛴다 (실측)."""
    identity = CentroidIdentity(tolerance=0.06)
    a = identity.label([0.45, 0.30, 0.85])
    assert identity.label([0.46, 0.31, 0.85]) == a      # 같은 물체의 흔들림
    b = identity.label([0.75, 0.30, 0.85])              # 307 mm 점프
    assert b != a
    assert identity.label([0.45, 0.30, 0.85]) == a      # 돌아오면 같은 이름
    assert identity.label(None) is None
