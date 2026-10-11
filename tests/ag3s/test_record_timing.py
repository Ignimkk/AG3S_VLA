"""기록이 스스로 말하는 것들 — 시간 단위, 단계 경계, 출처 도장.

이 셋은 전부 **상수로 박혀 있다가 틀렸던 것**이라 회귀 테스트가 따로 있다 (2026-09-25):

* `step7_state_lag.STEP_MS = 16.0` — sim timestep 2 ms × 8 이었는데, 8 이 곱해질 상대는
  제어 주기 66.7 ms 였다. 실제 값은 533 ms 로 **33 배** 차이다.
* `_phase(t, b=(24, 56, 72))` — `run_0004` 에서 나온 값인데 50 장짜리 긴 기록에 쓰면
  `t_step >= 72` 가 전부 `grasp` 이라 호출 9~49 가 잘못 라벨된다.
* 출처 도장 없음 — 자세만 새 기록이고 depth·필드는 옛 기록인 혼합 실행이 조용히 성립했다.

여기 테스트는 **기록이 있어야만** 뜻이 있으므로 없으면 skip 한다.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.sources.policy_record import (
    RunRecord, StepRecord, gripper_columns_for_state, grasp_onset,
    phase_boundaries_for_path, phase_boundaries_from_record, phase_for,
    provenance_arrays, read_provenance, record_fingerprint, record_step_interval_ms,
    require_same_record)

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_14D = ROOT / "run_0004"
RUN_16D = ROOT / "outputs/live_test/20260924_long16d/run_0000"


def _synthetic(width: int, n: int, *, stride: int = 8, close_at: int | None = None,
               ctrl_hz: int = 15) -> RunRecord:
    """기록 없이도 도는 최소 `RunRecord`. 그리퍼 열만 뜻이 있다."""
    left, _right = gripper_columns_for_state(width)
    steps = []
    for i in range(n):
        state = np.ones(width, np.float64)
        if close_at is not None and i >= close_at:
            state[left] = 0.44
        steps.append(StepRecord(
            t_step=i * stride, qpos=np.full(3, float(i)), qvel=np.zeros(3),
            state=state, actions=np.zeros((1, width), np.float32), infer_ms=0.0,
            images={}))
    meta = {"ctrl_hz": ctrl_hz, "open_loop_horizon": stride, "prompt": "p",
            "policy_model": "m", "nq": 3}
    return RunRecord(path=pathlib.Path("/synthetic"), meta=meta, steps=steps)


# ----------------------------------------------------------------- 그리퍼 열


@pytest.mark.parametrize("width, expected", [(14, (6, 13)), (16, (7, 15))])
def test_gripper_columns_follow_width(width, expected):
    """상수로 박으면 차원을 바꿀 때 엉뚱한 열을 그리퍼로 읽는다 — 증상이 안 보이는 실패다."""
    assert gripper_columns_for_state(width) == expected


@pytest.mark.parametrize("bad", [0, 3, 15, 7])
def test_gripper_columns_reject_odd_width(bad):
    with pytest.raises(ValueError):
        gripper_columns_for_state(bad)


# ----------------------------------------------------------------- 시간 단위


def test_record_step_interval_is_derived_not_16ms():
    """기록 한 장은 `open_loop_horizon / ctrl_hz` 다. 옛 상수 16 ms 가 아니다."""
    run = _synthetic(16, 5, stride=8, ctrl_hz=15)
    assert record_step_interval_ms(run) == pytest.approx(533.333, abs=1e-2)
    # 33 배 차이라는 것을 테스트가 직접 못박는다.
    assert record_step_interval_ms(run) > 30 * 16.0


def test_record_step_interval_follows_ctrl_hz():
    assert record_step_interval_ms(
        _synthetic(16, 5, stride=8, ctrl_hz=30)) == pytest.approx(266.667, abs=1e-2)
    assert record_step_interval_ms(
        _synthetic(16, 5, stride=4, ctrl_hz=15)) == pytest.approx(266.667, abs=1e-2)


def test_record_step_interval_rejects_disagreeing_meta():
    """`meta` 의 `open_loop_horizon` 과 실제 `t_step` 간격이 다르면 멈춘다 — 어느 쪽이
    참인지 모르는 채로 ms 를 지어내면 안 된다."""
    run = _synthetic(16, 5, stride=8)
    run.meta["open_loop_horizon"] = 4
    with pytest.raises(ValueError, match="open_loop_horizon"):
        record_step_interval_ms(run)


def test_record_step_interval_rejects_missing_ctrl_hz():
    run = _synthetic(16, 5)
    run.meta["ctrl_hz"] = 0
    with pytest.raises(ValueError, match="ctrl_hz"):
        record_step_interval_ms(run)


# ----------------------------------------------------------------- 단계 경계


def test_phase_boundaries_from_gripper():
    """파지는 그리퍼가 벗어난 t_step 에서 **한 칸 앞당긴** 자리다 (지령이 한 호출 앞의
    청크에 있었으므로)."""
    run = _synthetic(16, 40, stride=8, close_at=33)   # t_step 264 에서 닫힌다
    boundaries, evidence = phase_boundaries_from_record(run)
    assert evidence["source"] == "gripper"
    assert evidence["grasp_onset"]["t_step"] == 264
    assert boundaries == (208, 240, 256)              # 264 - 8, 그 앞에 16 · 32


def test_phase_boundaries_fall_back_without_grasp():
    """그리퍼가 끝까지 열려 있으면 경계를 지어내지 않는다."""
    boundaries, evidence = phase_boundaries_from_record(_synthetic(16, 15, close_at=None))
    assert evidence["source"] == "fallback"
    assert boundaries == (24, 56, 72)


def test_phase_boundaries_fall_back_when_grasp_too_early():
    """파지가 너무 일러 앞 구간이 안 들어가면 순서가 깨진 경계를 내놓지 않는다."""
    boundaries, evidence = phase_boundaries_from_record(_synthetic(16, 10, close_at=1))
    assert evidence["source"] == "fallback"
    assert boundaries == (24, 56, 72)


def test_phase_for_orders_phases():
    b = (208, 240, 256)
    assert phase_for(0, b) == "transit"
    assert phase_for(207, b) == "transit"
    assert phase_for(208, b) == "approach"
    assert phase_for(239, b) == "approach"
    assert phase_for(240, b) == "pre_grasp"
    assert phase_for(255, b) == "pre_grasp"
    assert phase_for(256, b) == "grasp"
    assert phase_for(392, b) == "grasp"


# ----------------------------------------------------------------- 출처 도장


def test_fingerprint_is_independent_of_how_many_frames_were_loaded():
    """**이것이 깨지면 짝 검사가 짝이 맞는 짝을 거부한다.** 중간 산출물은 보통 앞 몇 장만
    굽고 소비자는 `limit` 으로 읽으므로, 지문이 프레임 수를 타면 안 된다."""
    full = _synthetic(16, 40, close_at=33)
    part = RunRecord(path=full.path, meta=full.meta, steps=full.steps[:3])
    assert record_fingerprint(full) == record_fingerprint(part)


def test_fingerprint_separates_different_records():
    a = _synthetic(16, 10)
    b = _synthetic(14, 10)
    assert record_fingerprint(a) != record_fingerprint(b)
    c = _synthetic(16, 10)
    c.meta["policy_model"] = "other"
    assert record_fingerprint(a) != record_fingerprint(c)


def test_require_same_record_accepts_matching_stamp(tmp_path):
    run = _synthetic(16, 10)
    path = tmp_path / "frames.npz"
    np.savez(path, **provenance_arrays(run, 3), payload=np.zeros(2))
    with np.load(path) as blob:
        stamp = read_provenance(blob, where="frames")
    assert stamp["stamped"] and stamp["frames"] == 3
    require_same_record(run, stamp)          # 예외가 안 나야 한다


def test_require_same_record_rejects_other_record(tmp_path):
    mine, other = _synthetic(16, 10), _synthetic(14, 10)
    path = tmp_path / "frames.npz"
    np.savez(path, **provenance_arrays(other, 3))
    with np.load(path) as blob:
        stamp = read_provenance(blob, where="frames")
    with pytest.raises(SystemExit, match="다른 기록에서 나왔다"):
        require_same_record(mine, stamp)


def test_require_same_record_rejects_unstamped(tmp_path):
    """도장이 없는 옛 형식을 **통과시키지 않는다**. "모른다" 를 "맞다" 로 읽는 것이
    2026-09-25 에 난 일이다."""
    run = _synthetic(16, 10)
    path = tmp_path / "old.npz"
    np.savez(path, payload=np.zeros(2))
    with np.load(path) as blob:
        stamp = read_provenance(blob, where="old")
    assert not stamp["stamped"]
    with pytest.raises(SystemExit, match="출처 도장이 없다"):
        require_same_record(run, stamp)


# ------------------------------------------------- 실제 기록이 있을 때만 (회귀)


@pytest.mark.skipif(not RUN_14D.exists(), reason="run_0004 가 없다")
def test_derivation_reproduces_the_old_hardcoded_constants():
    """**이 테스트가 이 규칙의 근거다.** 그리퍼에서 뽑은 경계가 14D 시절 하드코딩되어
    있던 `(24, 56, 72)` 와 세 값 모두 일치한다 — 새로 지어낸 규칙이 아니라 옛 상수를
    만든 규칙이라는 뜻이다."""
    boundaries, evidence = phase_boundaries_for_path(RUN_14D)
    assert evidence["source"] == "gripper"
    assert evidence["grasp_onset"]["t_step"] == 80
    assert boundaries == (24, 56, 72)


@pytest.mark.skipif(not RUN_16D.exists(), reason="16D 긴 기록이 없다")
def test_long_16d_record_needs_different_boundaries():
    """같은 규칙이 긴 기록에서는 다른 답을 낸다. 옛 상수를 그대로 쓰면 호출 9~49 가
    전부 `grasp` 이 된다."""
    boundaries, evidence = phase_boundaries_for_path(RUN_16D)
    assert evidence["source"] == "gripper"
    assert boundaries == (208, 240, 256)
    assert boundaries != (24, 56, 72)
    # 옛 상수였다면 t_step 72 (호출 9) 부터 grasp 이었을 것이다.
    assert phase_for(72, (24, 56, 72)) == "grasp"
    assert phase_for(72, boundaries) == "transit"


@pytest.mark.skipif(not RUN_16D.exists(), reason="16D 긴 기록이 없다")
def test_long_16d_record_step_is_533ms_not_16ms():
    meta = json.loads((RUN_16D / "meta.json").read_text())
    assert meta["ctrl_hz"] == 15 and meta["open_loop_horizon"] == 8
    from AG3S.ag3s.experiments.sources.policy_record import load_run
    assert record_step_interval_ms(load_run(RUN_16D, limit=4)) == pytest.approx(533.33, abs=0.1)


@pytest.mark.skipif(not RUN_16D.exists(), reason="16D 긴 기록이 없다")
def test_grasp_onset_matches_the_recorded_gripper():
    from AG3S.ag3s.experiments.sources.policy_record import load_run
    index, t_step, hand, value = grasp_onset(load_run(RUN_16D))
    assert (index, t_step, hand) == (33, 264, "left")
    assert value < 0.7
