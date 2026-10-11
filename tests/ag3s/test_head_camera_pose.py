"""머리 카메라가 테이블을 보는 자세로 남아 있는가 — 되돌아가지 못하게 막는다.

2026-09-25 에 난 일: `teleop` 키프레임은 `head_1 = 0.7 rad`(테이블을 내려다봄)를 담는데
**`ctrl` 은 안 담는다.** `mj_resetDataKeyframe` 는 `qpos` 와 `ctrl` 을 둘 다 적용하므로
`head_1_act`(position actuator)가 0 을 지령한 채로 남고, 안정화 1.5 초 동안 머리를 0 으로
끌어내린다. 그래서 `cam_high` 가 테이블 대신 **벽**을 찍었다.

팔은 두 리셋 경로가 `ctrl` 을 명시적으로 덮어써서 살아남았고 머리만 아무도 안 세웠다.
수집 경로(`transport_scene.py`)는 *모든* 구동 관절을 키프레임 qpos 로 붙들어 처음부터
문제가 없었다 — 두 경로가 갈라져 있던 것이다.

2026-09-25 이전의 16D 기록(`run_16d_ep1800`, `20260924_long16d`)은 전부 이 상태로 떠졌고,
정책도 그 프레임을 먹었다. 그 위에서 잰 grounding 수치는 다시 재야 한다.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import pathlib
import re

import mujoco
import numpy as np
import pytest

MODEL = pathlib.Path(
    _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
)
INFER = pathlib.Path(_ag3s_workspace_path('integration/rby1/bringup/pi05_infer.py'))

#: 테이블을 내려다보는 각도. `teleop` 키프레임이 담은 값이고, 수집 데이터셋이 이 자세로 찍혔다.
HEAD_TILT_RAD = 0.7


def _model():
    if not MODEL.exists():
        pytest.skip(f"model not present: {MODEL}")
    return mujoco.MjModel.from_xml_path(str(MODEL))


def _qadr(m, name):
    joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    assert joint >= 0, f"joint {name!r} missing"
    return int(m.jnt_qposadr[joint])


def _hold_keyframe_pose(m, d):
    """`pi05_infer.hold_keyframe_pose` 와 같은 동작 (여기서 import 하기엔 의존이 무겁다)."""
    for actuator in range(m.nu):
        if m.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        d.ctrl[actuator] = d.qpos[m.jnt_qposadr[m.actuator_trnid[actuator, 0]]]


def test_teleop_keyframe_still_tilts_the_head():
    """키프레임의 의도가 남아 있는가. 이 값이 바뀌면 카메라가 보는 곳이 바뀐다."""
    m = _model()
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    assert key >= 0, "model has no 'teleop' keyframe"
    assert float(m.key_qpos[key][_qadr(m, "head_1")]) == pytest.approx(HEAD_TILT_RAD)


def test_head_survives_the_settle_when_the_keyframe_pose_is_held():
    """**이것이 진짜 불변식이다.** 리셋 뒤 머리가 숙인 채로 남아야 한다."""
    m = _model()
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    head = _qadr(m, "head_1")

    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    _hold_keyframe_pose(m, d)
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)

    assert float(d.qpos[head]) == pytest.approx(HEAD_TILT_RAD, abs=1e-2), (
        "안정화 뒤 머리가 숙인 자세를 잃었다 — cam_high 가 테이블 대신 벽을 찍는다"
    )


def test_bare_keyframe_reset_still_drops_the_head():
    """왜 붙들어야 하는지를 못박는다. 이 테스트가 깨지면 모델이 스스로 고쳐진 것이니,
    그때는 `hold_keyframe_pose` 가 불필요해졌는지 확인하고 이 테스트를 지운다."""
    m = _model()
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "teleop")
    head = _qadr(m, "head_1")

    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)  # ctrl 을 안 붙든다
    for _ in range(int(1.5 / m.opt.timestep)):
        mujoco.mj_step(m, d)

    assert abs(float(d.qpos[head])) < 1e-2, (
        "맨 리셋이 더는 머리를 떨어뜨리지 않는다 — 키프레임에 ctrl 이 생겼을 수 있다"
    )


def test_every_keyframe_reset_in_pi05_infer_holds_the_pose():
    """구조적 가드 — `mj_resetDataKeyframe` 뒤에는 반드시 붙들어야 한다.

    동작 테스트만으로는 *새로 추가된* 리셋 경로를 못 잡는다. 이 검사는 호출부를 센다.
    """
    if not INFER.exists():
        pytest.skip(f"not present: {INFER}")
    text = INFER.read_text()
    resets = [mm.start() for mm in re.finditer(r"mj_resetDataKeyframe\(", text)]
    assert resets, "pi05_infer 에서 키프레임 리셋을 못 찾았다 — 이 가드가 낡았다"
    for pos in resets:
        window = text[pos : pos + 400]
        assert "hold_keyframe_pose(" in window, (
            f"{INFER.name} 의 오프셋 {pos} 근처 mj_resetDataKeyframe 뒤에 "
            "hold_keyframe_pose() 가 없다 — 머리가 다시 0 으로 떨어진다"
        )
