"""`UNCOVERED_LINKS` 의 이름이 두 로봇 파일에서 **동시에** 맞는가.

2026-09-25 에 난 일: 손목 카메라가 자기 그리퍼 뭉치를 늘 보는데 자기 필터가 그것을 못 지웠다.
참값 segmentation 기준으로 `EE_BODY_L` 90,690 px · `EE_BODY_R` 90,688 px 가 샜고, 전부
`wrist_cam_l`/`wrist_cam_r` 로만 갔다. 원인은 필터 값이 아니라 **빠진 이름**이었다 —
목록에 손가락(`ee_finger_*`)은 있고 손바닥이 없었다.

빠진 채로 있을 수 있었던 이유가 이 파일의 존재 이유다. `gap_filling_capsules` 는
`bounding_capsules` 의 `KeyError` 를 삼킨다 (시뮬레이터가 따로 모델링하지 않는 링크를 위해서다).
그래서 MuJoCo 이름을 틀리면 **조용히 캡슐 0 개**가 나오고, 자기 필터에는 아무도 모르는 구멍이
남는다. URDF 는 `ee_left`, MJCF 는 `EE_BODY_L` 로 같은 부품을 다르게 부른다.

여기서 고정하는 것은 세 가지다: 목록의 **모든** 이름이 실제로 캡슐을 내놓는가 ·
그 캡슐이 `UrdfSphereChain` 에 붙는 URDF 링크 이름을 다는가 · 치수가 손으로 고른 값이 아니라
MJCF 메시에서 나왔는가(= 메시 정점을 전부 담는가).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

MODEL = pathlib.Path(
    "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml"
)

mujoco = pytest.importorskip("mujoco")

from benchmark.ag3s.experiments.sources.mujoco_source import (  # noqa: E402
    MJCF_BODY_ALIASES,
    UNCOVERED_LINKS,
    _body_vertices,
    bounding_capsules,
    gap_filling_capsules,
)


@pytest.fixture(scope="module")
def model():
    if not MODEL.exists():
        pytest.skip(f"model not present: {MODEL}")
    return mujoco.MjModel.from_xml_path(str(MODEL))


@pytest.fixture(scope="module")
def urdf():
    from benchmark.ag3s.robot_models import RBY1_URDF, parse_urdf

    try:
        return parse_urdf(RBY1_URDF)
    except FileNotFoundError:  # pragma: no cover - asset-dependent
        pytest.skip("RB-Y1 URDF not present")


def _body_id(model, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def test_every_uncovered_link_actually_produces_capsules(model):
    """조용한 0 개를 막는다. 이것이 손바닥이 빠진 채 남아 있던 실패 모드다."""
    empty = []
    for link in UNCOVERED_LINKS:
        body = MJCF_BODY_ALIASES.get(link, link)
        if _body_id(model, body) < 0:
            empty.append(f"{link!r}: no MuJoCo body {body!r} (alias needed?)")
            continue
        if not bounding_capsules(model, body, link=link):
            empty.append(f"{link!r}: body {body!r} yielded no capsule")
    assert not empty, "\n".join(empty)


def test_the_gripper_palm_is_covered_on_both_sides(model):
    """`ee_left`/`ee_right` 는 손목 카메라가 매 프레임 보는 것이다. 목록에 있고 캡슐이 난다."""
    assert {"ee_left", "ee_right"} <= set(UNCOVERED_LINKS)
    links = {c.link for c in gap_filling_capsules(model)}
    assert {"ee_left", "ee_right"} <= links


def test_capsules_carry_urdf_link_names_not_mujoco_ones(model, urdf):
    """`UrdfSphereChain` 이 URDF 에 없는 링크를 단 `extra_capsules` 를 거부한다."""
    from benchmark.ag3s.robot_models import UrdfSphereChain

    extra = gap_filling_capsules(model)
    unknown = {c.link for c in extra} - set(urdf.links)
    assert not unknown, f"capsules name links the URDF does not have: {sorted(unknown)}"
    robot = UrdfSphereChain(urdf, extra_capsules=extra)  # raises if a name is wrong
    assert {"ee_left", "ee_right"} <= set(robot.sphere_link_names)


@pytest.mark.parametrize(
    "link, parent", [("ee_left", "link_left_arm_6"), ("ee_right", "link_right_arm_6")]
)
def test_the_aliased_body_sits_in_the_same_frame_as_its_urdf_link(model, urdf, link, parent):
    """별칭이 성립하는 근거. **프레임이 같아야** MJCF 정점을 그대로 URDF 링크에 붙일 수 있다.

    MJCF 는 `link_*_arm_6` 밑에 `EE_BODY_*` 를 (0, 0, -0.1548) 로 매달고, URDF 는 같은 자리를
    `tool_*`(-0.1087) + `FT_Sensor_END_*`(-0.0461) 두 fixed joint 로 간다. 합이 같다는 것이
    `_body_vertices` 의 body-frame 좌표를 변환 없이 쓸 수 있는 이유다.
    """
    from benchmark.ag3s.robot_models import UrdfSphereChain

    body = _body_id(model, MJCF_BODY_ALIASES[link])
    assert body >= 0
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body])) == parent
    mj_offset = np.asarray(model.body_pos[body], float)
    mj_rot = np.zeros(9)
    mujoco.mju_quat2Mat(mj_rot, model.body_quat[body])

    robot = UrdfSphereChain(urdf, extra_capsules=gap_filling_capsules(model))
    q = np.zeros(robot.nq)
    T = np.linalg.inv(robot.link_pose(q, parent)) @ robot.link_pose(q, link)
    assert np.allclose(T[:3, 3], mj_offset, atol=1e-9), f"{T[:3, 3]} vs {mj_offset}"
    assert np.allclose(T[:3, :3], mj_rot.reshape(3, 3), atol=1e-9)


@pytest.mark.parametrize("link", ["ee_left", "ee_right"])
def test_palm_capsule_dimensions_come_from_the_mesh_and_contain_it(model, link):
    """치수를 눈대중으로 정하지 않았다는 것을 수치로 고정한다.

    `fit_capsule` 의 보장(반지름 = 선분까지의 **최대** 거리)이 슬래브별로 성립하고, 슬래브가
    정점을 남김없이 나누므로 합집합은 몸체를 통째로 담는다. 그리고 그 위로 크지 않다 —
    반지름이 몸체의 최대 폭을 넘지 않는다.
    """
    body = _body_id(model, MJCF_BODY_ALIASES[link])
    points = _body_vertices(model, body)
    assert points.shape[0] > 1000, "EE_BODY should have a real mesh, not a placeholder"
    capsules = bounding_capsules(model, MJCF_BODY_ALIASES[link], link=link)
    assert len(capsules) == 3  # segments=3, the default used for every other link

    inside = np.zeros(points.shape[0], bool)
    half_width = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)) / 2.0)
    for cap in capsules:
        axis = cap.origin[:3, :3][:, 2]
        local = points - cap.origin[:3, 3]
        t = np.clip(local @ axis, -cap.length / 2.0, cap.length / 2.0)
        inside |= np.linalg.norm(local - t[:, None] * axis, axis=1) <= cap.radius + 1e-9
        assert 0.0 < cap.radius <= half_width, f"{cap.radius} out of range for {link}"
    assert inside.all(), f"{(~inside).sum()} of {points.shape[0]} mesh vertices uncovered"
