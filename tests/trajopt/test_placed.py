"""A3 — 성공 판정 주입. 기하는 `placed.py` 가, 이탈은 그리퍼가 답한다.

고정하는 것:

1. **모르면 거짓** — 쥔 것이 없거나 필드가 없거나 목적지 라벨이 없으면 판정하지 않는다.
   모르는 것을 "놓았다" 로 읽으면 쥔 채로 detach 되고, 그러면 쥔 물체가 장애물로 되돌아온다
   (E3 — 쥔 물체가 optimizer 에 도달하지 않는다 — 가 만들던 구멍).
2. **목적지 범위를 라벨 층에서 읽는다** — 따로 기하를 주입받지 않는다.
3. **테두리 아래여야 한다** — 목적지 위를 지나는 것만으로는 놓은 것이 아니다.
4. **그리퍼가 닫혀 있으면 잠금이 안 푼다** — 기하만으로 detach 되면 안 된다.
"""

import numpy as np
import pytest

from benchmark.ag3s.types import AttachedCollisionGeometry
from benchmark.trajopt.grasp_latch import GraspLatch, GraspPhase, LatchConfig
from benchmark.trajopt.placed import destination_extent, destination_placement

VOXEL = 0.02
LABEL = "destination"


class _Model:
    """`link_pose` 만 있으면 된다 — `attached_points_in_base` 가 그것만 쓴다."""

    def __init__(self, offset):
        self.offset = np.asarray(offset, float)

    def link_pose(self, q, link):          # noqa: ARG002 — 자세는 q[0] 으로만 흔든다
        T = np.eye(4)
        T[:3, 3] = self.offset + np.array([0.0, 0.0, float(q[0])])
        return T


class _Grid:
    def __init__(self, origin, voxel_size):
        self.origin = np.asarray(origin, float)
        self.voxel_size = float(voxel_size)


class _Field:
    """라벨 층만 흉내 낸다. `destination_extent` 가 쓰는 것이 그것뿐이다."""

    def __init__(self, label_grid, origin=(0.0, 0.0, 0.0), names=(LABEL,)):
        self.label_grid = np.asarray(label_grid, np.int32)
        self.grid = _Grid(origin, VOXEL)
        self.label_names = tuple(names)

    def label_id(self, name):
        return self.label_names.index(name) if name in self.label_names else -1


class _Set:
    def __init__(self, *, attached, esdf, destination_label, robot_state=(0.0,)):
        self.attached = attached
        self.esdf = esdf
        self.destination_label = destination_label
        self.robot_state = np.asarray(robot_state, float)


def _crate_field():
    """`x,y in [2..6]`, `z in [0..4]` 복셀이 목적지. 테두리 중심은 z = 4 * 0.02 = 0.08 m."""
    g = np.full((10, 10, 10), -1, np.int32)
    g[2:7, 2:7, 0:5] = 0
    return _Field(g)


def _attached(points):
    return AttachedCollisionGeometry(parent_link="hand", T_parent_object=np.eye(4),
                                     primitives=[], points=np.asarray(points, float))


def test_extent_comes_from_the_label_layer():
    lo, hi = destination_extent(_crate_field(), LABEL)
    assert np.allclose(lo, [0.04, 0.04, 0.00])
    assert np.allclose(hi, [0.12, 0.12, 0.08])
    assert destination_extent(_crate_field(), "nope") is None


def test_placed_when_inside_and_below_the_rim():
    # 손이 상자 한가운데 위에 있고, q[0] 이 손의 높이를 정한다.
    fn = destination_placement(robot_model=_Model([0.08, 0.08, 0.0]), below_rim=0.02)
    held = _attached([[0.0, 0.0, 0.0], [0.005, 0.0, 0.0]])

    # 테두리(0.08)보다 20 mm 아래 = 0.06 이하여야 한다. 손 높이 0.03 -> 안쪽.
    deep = _Set(attached=held, esdf=_crate_field(), destination_label=LABEL,
                robot_state=(0.03,))
    assert fn({}, deep) is True

    # 테두리 바로 위를 지나는 중 — 아직 놓은 것이 아니다.
    over = _Set(attached=held, esdf=_crate_field(), destination_label=LABEL,
                robot_state=(0.10,))
    assert fn({}, over) is False


def test_outside_the_horizontal_range_is_not_placed():
    fn = destination_placement(robot_model=_Model([0.30, 0.08, 0.0]), below_rim=0.02)
    held = _attached([[0.0, 0.0, 0.0]])
    cs = _Set(attached=held, esdf=_crate_field(), destination_label=LABEL,
              robot_state=(0.03,))
    assert fn({}, cs) is False


def test_unknown_is_false_not_placed():
    """모르면 **거짓**이다 — fail-closed."""
    fn = destination_placement(robot_model=_Model([0.08, 0.08, 0.0]))
    held = _attached([[0.0, 0.0, 0.0]])
    field = _crate_field()
    assert fn({}, _Set(attached=None, esdf=field, destination_label=LABEL)) is False
    assert fn({}, _Set(attached=held, esdf=None, destination_label=LABEL)) is False
    assert fn({}, _Set(attached=held, esdf=field, destination_label=None)) is False
    # 라벨 이름이 필드에 없을 때도 마찬가지.
    assert fn({}, _Set(attached=held, esdf=field, destination_label="other")) is False


def test_a_few_stray_points_do_not_flip_the_verdict():
    fn = destination_placement(robot_model=_Model([0.08, 0.08, 0.0]), below_rim=0.02,
                               min_points_inside=0.5)
    # 절반 미만만 안쪽 -> 아직 아니다.
    held = _attached([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0]])
    cs = _Set(attached=held, esdf=_crate_field(), destination_label=LABEL,
              robot_state=(0.03,))
    assert fn({}, cs) is False


def test_latch_does_not_release_while_the_gripper_is_closed():
    """기하만으로 풀면 **쥔 채로** detach 된다 — 잠금이 그리퍼와 AND 로 묶는다."""
    cfg = LatchConfig(confirm_frames=1, release_frames=2)
    latch = GraspLatch(cfg)
    for _ in range(cfg.confirm_frames):
        latch.update(label="apple", score=1.0, runner_up=0.0, gripper=1.0)
    e = latch.update(label="apple", score=1.0, runner_up=0.0, gripper=0.0)
    assert e.attach and latch.phase is GraspPhase.HELD

    # 목적지 위를 지나는 중: 기하는 참인데 그리퍼는 닫혀 있다 -> 안 푼다.
    e = latch.update(label="crate", score=1.0, runner_up=0.0, gripper=0.0, placed=True)
    assert not e.detach and latch.phase is GraspPhase.HELD

    # 그리퍼가 열리면 **그 프레임에** 푼다 — 폴백의 2 프레임을 기다리지 않는다.
    e = latch.update(label="crate", score=1.0, runner_up=0.0, gripper=1.0, placed=True)
    assert e.detach and latch.phase is GraspPhase.PLACED
    assert "성공 판정" in e.note


def test_gripper_fallback_still_works_without_a_placed_fn():
    cfg = LatchConfig(confirm_frames=1, release_frames=2)
    latch = GraspLatch(cfg)
    latch.update(label="apple", score=1.0, runner_up=0.0, gripper=1.0)
    latch.update(label="apple", score=1.0, runner_up=0.0, gripper=0.0)
    assert latch.phase is GraspPhase.HELD
    assert not latch.update(label="crate", score=1.0, runner_up=0.0, gripper=1.0).detach
    e = latch.update(label="crate", score=1.0, runner_up=0.0, gripper=1.0)
    assert e.detach and "그리퍼" in e.note


@pytest.mark.parametrize("below_rim, expected", [(0.02, True), (0.07, False)])
def test_below_rim_threshold_is_the_knob(below_rim, expected):
    fn = destination_placement(robot_model=_Model([0.08, 0.08, 0.0]), below_rim=below_rim)
    cs = _Set(attached=_attached([[0.0, 0.0, 0.0]]), esdf=_crate_field(),
              destination_label=LABEL, robot_state=(0.03,))
    assert fn({}, cs) is expected
