"""`ag3s.static_scene` — 정적 도형을 누가 주는가 (A1).

고정하는 것은 셋이다:

1. **추출 규칙** — 움직이는 것(free joint) · 로봇 · 시각 전용은 안 뽑는다. 특히 시각 전용:
   실제 씬의 `left_ee_target` 마커가 로봇 작업공간 한가운데 있어서, 이 필터가 빠지면
   **없는 장애물**이 제약에 들어온다.
2. **JSON 왕복** — 실기에는 MuJoCo 가 없으므로 JSON 이 계약이다. 왕복해도 거리가 안 바뀌어야 한다.
3. **좌표계** — base 가 아닌 좌표계는 **거절**한다. 틀린 좌표계의 도형은 조용히 엉뚱한 자리를
   막고, 증상은 "최적화기가 왜 여기서 멈추지" 로만 보인다.
"""

import numpy as np
import pytest

from benchmark.ag3s.fields.esdf import StaticBox, StaticPlane, analytic_distance
from benchmark.ag3s.fields.static_scene import (
    dump, from_jsonable, from_mujoco, load, to_jsonable,
)

MJCF = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1"/>
    <body name="table" pos="0.6 0 0.8">
      <geom name="table_top" type="box" size="0.25 0.5 0.02"/>
    </body>
    <body name="marker" pos="0.5 0 0.5">
      <geom name="ee_target" type="box" size="0.04 0.04 0.04" contype="0" conaffinity="0"/>
    </body>
    <body name="apple" pos="0.45 -0.3 0.85">
      <freejoint name="apple_free"/>
      <geom name="apple_geom" type="sphere" size="0.03"/>
    </body>
    <body name="link_arm" pos="0.2 0 1.0">
      <geom name="link_arm_geom" type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="pillar" pos="1.5 1.5 1.0">
      <geom name="pillar_geom" type="cylinder" size="0.1 1.0"/>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture
def model():
    mujoco = pytest.importorskip("mujoco")
    return mujoco.MjModel.from_xml_string(MJCF)


def test_extraction_keeps_only_static_collidable_boxes_and_planes(model):
    shapes, stats = from_mujoco(model)
    labels = {s.label.split("/")[-1] for s in shapes}
    assert labels == {"floor", "table_top"}
    # 버린 것을 **센다** — 말없이 빠지면 안 되는 것들이다.
    assert stats.n_free == 1        # apple
    assert stats.n_robot == 1       # link_arm (ROBOT_BODY_PREFIXES 의 "link_")
    assert stats.n_visual == 1      # ee_target 마커
    assert [n.split("/")[-1] for n, _ in stats.unsupported] == ["pillar_geom"]  # 실린더


def test_visual_marker_would_otherwise_become_a_phantom_obstacle(model):
    """`require_collidable=False` 로 두면 작업공간 한가운데 없는 장애물이 생긴다."""
    lax, _ = from_mujoco(model, require_collidable=False)
    probe = np.array([[0.5, 0.0, 0.5]])          # 마커의 중심
    strict, _ = from_mujoco(model)
    assert analytic_distance(probe, lax)[0] < 0.0        # 마커 안쪽이라고 답한다
    # 걸러내면 가장 가까운 것은 테이블 상판 아랫면 (0.78 − 0.50 = 0.28 m) 이다.
    assert analytic_distance(probe, strict)[0] == pytest.approx(0.28)


def test_plane_normal_points_into_free_space(model):
    shapes, _ = from_mujoco(model)
    floor = next(s for s in shapes if s.label.endswith("floor"))
    assert np.allclose(floor.normal, [0, 0, 1])
    # 바닥 위 1 m 는 +1 m, 아래는 음수. 부호 규약(음수 = 물체 안쪽)이 뒤집히면 최적화기가
    # 바닥을 통과하는 쪽을 자유공간으로 본다.
    assert analytic_distance(np.array([[0.0, 0.0, 1.0]]), [floor])[0] == pytest.approx(1.0)
    assert analytic_distance(np.array([[0.0, 0.0, -0.2]]), [floor])[0] < 0.0


def test_base_frame_transform_moves_the_shapes(model):
    """`T_world_base` 가 주어지면 도형이 base 좌표계로 온다."""
    T = np.eye(4)
    T[:3, 3] = [1.0, 0.0, 0.0]                   # base 가 월드 x=+1 m 에 있다
    shapes, _ = from_mujoco(model, T_world_base=T)
    table = next(s for s in shapes if s.label.endswith("table_top"))
    assert np.allclose(table.center, [-0.4, 0.0, 0.8])


def test_json_roundtrip_preserves_distances(model, tmp_path):
    shapes, _ = from_mujoco(model)
    path = dump(tmp_path / "static.json", shapes, note="test")
    back = load(path)
    pts = np.array([[0.6, 0.0, 0.9], [0.0, 0.0, 0.3], [2.0, 2.0, 1.0], [0.6, 0.0, 0.81]])
    assert np.allclose(analytic_distance(pts, shapes), analytic_distance(pts, back))
    assert [s.label for s in back] == [s.label for s in shapes]


def test_rotated_box_survives_the_roundtrip():
    """회전을 떨어뜨리면 축이 돌아간 선반이 조용히 축정렬 상자가 된다."""
    c, s = np.cos(0.7), np.sin(0.7)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    box = StaticBox(center=np.array([1.0, 0.0, 0.5]),
                    half_extents=np.array([0.4, 0.05, 0.5]), rotation=rot, label="tilted")
    back = from_jsonable(to_jsonable([box]))
    pts = np.array([[1.3, 0.3, 0.5], [1.0, 0.0, 0.5], [0.5, -0.4, 0.5]])
    assert np.allclose(analytic_distance(pts, [box]), analytic_distance(pts, back))
    assert not np.allclose(rot, np.eye(3))       # 회전이 실제로 자명하지 않은 경우를 쟀다


def test_wrong_frame_is_refused():
    payload = to_jsonable([StaticPlane(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))])
    payload["frame"] = "world"
    with pytest.raises(ValueError, match="base"):
        from_jsonable(payload)


def test_unknown_shape_type_is_refused():
    with pytest.raises(ValueError, match="box | plane"):
        from_jsonable({"frame": "robot_base", "shapes": [{"type": "capsule"}]})
