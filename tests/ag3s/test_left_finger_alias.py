"""T11 — 왼손가락 이름이 URDF 와 MJCF 에서 서로 바뀌어 있다.

## 무엇이었나

`link_pose_error` 가 왼손가락 둘에서 **6.000 mm** 를 냈다. 오른손은 0.0005 mm 였다. 기하가 아니라
**이름**이다 — 교차로 비교하면 0 이다:

    URDF ee_finger_l1  vs  MJCF ee_finger_l1 :  6.0000 mm | 180.000 deg
    URDF ee_finger_l1  vs  MJCF ee_finger_l2 :  0.0005 mm |   0.000 deg   <- 정상 잔차

`link_*_arm_6` 기준으로 URDF 는 양팔 모두 `finger1` 을 `x = +3 mm` · 회전 0° 에 두는데 MJCF 는
**왼손만** 거울이다. 6.000 mm 는 `|+3 − (−3)|` 다.

**어느 파일도 틀리지 않았다.** 관절 축·리밋까지 양쪽이 각자 일관되고 frame 과 축이 함께 뒤집혀
있어 물리적 개폐 동작이 같다. 그래서 고친 것은 `MJCF_BODY_ALIASES` 두 줄이고, URDF 기하는
한 줄도 안 건드렸다.

## 이 파일이 박는 것

* 짝이 맞는다 — 네 손가락과 손바닥 둘 전부 `link_pose_error` 가 정상 잔차를 낸다.
* **구는 하나도 안 움직인다.** 네 손가락의 mesh 가 각자의 body frame 에서 **완전히 같기**
  때문이다 (실측: capsule 세 개가 글자 그대로 같다). 그래서 이 수정은 진단의 짝만 고치고
  충돌 모델은 건드리지 않는다 — 기대가 아니라 측정이 그렇게 말한다.
* `link_pose_error` 가 **조용히 건너뛰지 않는다.** 예전에는 alias 를 안 써서 `ee_left`/`ee_right`
  를 `KeyError` 로 건너뛰었고, 그 함수가 "frame 이 맞는가" 를 묻는 유일한 도구인데 **손바닥 둘을
  한 번도 안 재고 있었다.**
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import pytest

MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml"
)

FINGERS = ("ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2")
PALMS = ("ee_left", "ee_right")
#: `link_pose_error` 가 정상 link 에서 내는 잔차. 두 모델이 다른 파일에서 오므로 0 은 아니다.
NORMAL_RESIDUAL_MM = 0.01


@pytest.fixture(scope="module")
def scene():
    mujoco = pytest.importorskip("mujoco")
    if not MODEL.exists():  # pragma: no cover - asset-dependent
        pytest.skip(f"model not present: {MODEL}")
    from benchmark.ag3s.experiments.sources.mujoco_source import TransportScene

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    out = TransportScene.attach(model, data)
    mujoco.mj_forward(model, data)
    return out


@pytest.fixture(scope="module")
def robot(scene):
    from benchmark.ag3s.experiments.reports.grounding_report import build_robot_model

    return build_robot_model(scene)


# --- 1. 짝이 맞는다 ---------------------------------------------------------------------


def test_every_finger_and_palm_frame_agrees_now(scene, robot):
    """**이 시험이 회귀다.** 예전에는 왼손가락 둘이 6.000 mm 였다."""
    from benchmark.ag3s.experiments.sources.mujoco_source import link_pose_error

    got = link_pose_error(scene, robot, FINGERS + PALMS + ("link_left_arm_6",))
    assert set(got) >= set(FINGERS + PALMS), "손바닥도 재야 한다 — 예전에는 건너뛰었다"
    for link, error in got.items():
        assert error * 1000.0 < NORMAL_RESIDUAL_MM, f"{link}: {error * 1000:.4f} mm"


def test_the_alias_table_says_the_two_left_fingers_swap():
    from benchmark.ag3s.experiments.sources.mujoco_source import MJCF_BODY_ALIASES

    assert MJCF_BODY_ALIASES["ee_finger_l1"] == "ee_finger_l2"
    assert MJCF_BODY_ALIASES["ee_finger_l2"] == "ee_finger_l1"
    # 오른손은 이름이 같으므로 **항목이 없어야 한다** — 넣으면 자기 자신을 가리키는 줄이 되고,
    # 읽는 사람이 "오른손도 뭔가 어긋났다" 고 읽는다.
    assert "ee_finger_r1" not in MJCF_BODY_ALIASES
    assert "ee_finger_r2" not in MJCF_BODY_ALIASES


def test_the_swap_is_an_involution():
    """두 줄이 서로를 가리킨다. 한쪽만 있으면 다른 쪽 손가락이 같은 mesh 를 두 번 받는다."""
    from benchmark.ag3s.experiments.sources.mujoco_source import MJCF_BODY_ALIASES

    for link in ("ee_finger_l1", "ee_finger_l2"):
        assert MJCF_BODY_ALIASES[MJCF_BODY_ALIASES[link]] == link


def test_the_urdf_and_the_mjcf_are_each_internally_consistent(scene):
    """**어느 파일도 틀리지 않았다**는 근거. 고칠 기하가 없다는 뜻이다.

    양쪽 다 `finger1` = axis `−x` · 리밋 `[−0.05, 0]`, `finger2` = axis `+x` · `[0, 0.05]` 다.
    frame 과 축이 함께 뒤집혀 있어 **물리적 개폐 동작이 같다.**
    """
    import mujoco

    from benchmark.ag3s.robot_models import RBY1_URDF, parse_urdf

    urdf = parse_urdf(RBY1_URDF)
    for joint in urdf.joints:
        if not joint.name.startswith("gripper_finger"):
            continue
        first = joint.name.endswith("1")
        assert joint.axis[0] == (-1.0 if first else 1.0), joint.name
        assert (joint.lower, joint.upper) == ((-0.05, 0.0) if first else (0.0, 0.05)), joint.name

    model = scene.model
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name or not name.startswith("gripper_finger"):
            continue
        first = name.endswith("1")
        assert model.jnt_axis[j][0] == (-1.0 if first else 1.0), name
        lo, hi = model.jnt_range[j]
        assert (lo, hi) == pytest.approx((-0.05, 0.0) if first else (0.0, 0.05)), name


# --- 2. 구는 안 움직인다 (기대가 아니라 측정) -------------------------------------------


def test_all_four_finger_meshes_are_identical_in_their_own_body_frame(scene):
    """**이것이 "구가 안 움직인다" 의 이유다.** mesh 가 같으니 local capsule 이 같다."""
    from benchmark.ag3s.experiments.sources.mujoco_source import bounding_capsules

    sets = []
    for body in FINGERS:
        caps = bounding_capsules(scene.model, body, link="x")
        assert len(caps) == 3, body
        sets.append([(round(c.radius, 9), round(c.length, 9),
                      np.round(c.origin[:3, 3], 9).tolist()) for c in caps])
    assert all(s == sets[0] for s in sets), "네 손가락의 local capsule 이 같아야 한다"


def test_swapping_the_alias_moves_no_sphere_in_either_model(scene, monkeypatch):
    """alias 를 되돌려도 구가 그대로다 — 이 수정은 충돌 모델을 건드리지 않는다.

    lead 의 예상은 "self-filter capsule 이 6 mm 움직인다" 였는데 **측정은 0.000 mm** 다. 이유가
    위 시험이다: 네 손가락 mesh 가 각자의 frame 에서 같으므로, 어느 body 에서 재도 local capsule
    이 같고 배치는 URDF FK 가 정한다. 6 mm 는 구에 있던 적이 없고 **이름의 짝**에만 있었다.
    """
    from benchmark.ag3s.experiments.reports.grounding_report import (
        build_constraint_robot_model, build_robot_model)
    from benchmark.ag3s.experiments.sources import mujoco_source

    q = scene.robot_state()

    def centres_of(builder):
        model = builder(scene)
        pos, _ = model.sphere_centers_numeric(q)
        return {link: pos[[i for i, n in enumerate(model.sphere_link_names) if n == link]]
                for link in FINGERS}

    fixed = {"filter": centres_of(build_robot_model),
             "constraint": centres_of(build_constraint_robot_model)}
    # alias 두 줄을 지운 상태 = 고치기 전.
    for link in ("ee_finger_l1", "ee_finger_l2"):
        monkeypatch.delitem(mujoco_source.MJCF_BODY_ALIASES, link)
    before = {"filter": centres_of(build_robot_model),
              "constraint": centres_of(build_constraint_robot_model)}

    for label in fixed:
        for link in FINGERS:
            shift = np.linalg.norm(fixed[label][link] - before[label][link], axis=1)
            assert shift.max() * 1000.0 < 0.01, f"{label}/{link}: {shift.max() * 1000:.4f} mm"


# --- 3. 조용히 건너뛰지 않는다 ----------------------------------------------------------


def test_an_unmeasurable_link_is_named_instead_of_skipped_in_silence(scene, robot, caplog):
    """키가 **없는 것**과 0 mm 는 완전히 다른 뜻이다. 읽는 쪽이 둘을 구별할 근거가 있어야 한다."""
    from benchmark.ag3s.experiments.sources import mujoco_source

    with caplog.at_level(logging.WARNING, logger=mujoco_source.__name__):
        got = mujoco_source.link_pose_error(scene, robot, ("link_left_arm_6", "no_such_link"))
    assert "no_such_link" not in got, "못 잰 link 은 키가 없어야 한다 (0 mm 가 아니다)"
    assert "no_such_link" in caplog.text
    assert "MJCF_BODY_ALIASES" in caplog.text, "무엇을 고치면 되는지 말해야 한다"
    assert "link_left_arm_6" in got


def test_the_palms_were_the_links_that_used_to_be_skipped(scene, robot, monkeypatch):
    """alias 없이는 `ee_left` 가 MJCF 에 없다 — 그것이 조용히 건너뛰어진 이유다."""
    from benchmark.ag3s.experiments.sources import mujoco_source

    monkeypatch.delitem(mujoco_source.MJCF_BODY_ALIASES, "ee_left")
    got = mujoco_source.link_pose_error(scene, robot, ("ee_left", "ee_right"))
    assert "ee_left" not in got and "ee_right" in got
