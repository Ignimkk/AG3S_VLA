"""해석적 채널 — 아는 기하는 관측을 기다리지 않는다 (N2 · E4).

거리장은 depth 가 본 것만 담는다. 그래서 카메라가 한 번도 보지 않은 곳은 `unknown_policy=free`
에 따라 **자유**로 답한다 (E4 — 격자 밖·미관측은 무조건 자유). 실측(`run_0004`, 3 카메라,
20 mm): 격자의 66.3 % 가 미관측, 로봇 구 질의의 44.1 % 가 그런 복셀에 떨어지고, 그중 237 회는
거리가 500 mm 로 **포화**된다. 그 자리의 참 거리 대비 필드는 최대 **+219.5 mm 낙관적**이었다.

벽·선반·테이블은 아는 기하다. 여기 실어 두면 그 낙관이 **0.0 mm** 가 된다.

**찍기(`stamp_static`)로는 안 됐다.** 중요한 기하가 격자 밖이라(선반 y=-1550, 벽 ±3 m; 격자는
y ∈ [-900, 900]) 찍힐 자리가 없었고, 낙관이 0.0 mm 줄었다. 그래서 복셀이 아니라 **따로 잰다.**
"""

import numpy as np

from AG3S.ag3s.fields.esdf import (
    FREE, OCCUPIED, EsdfField, StaticBox, StaticPlane, VoxelGrid, analytic_distance,
    analytic_gradient, esdf_from_occupancy, stamp_static,
)

VOXEL = 0.05
BOX = StaticBox(center=np.zeros(3), half_extents=np.array([0.10, 0.10, 0.10]))
FLOOR = StaticPlane(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))


def _empty_field(shapes=()):
    """관측이 **하나도 없는** 필드 — 미관측이 자유로 나가는 바로 그 상태."""
    grid = VoxelGrid.from_bounds([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3], VOXEL)
    occupancy = np.full(grid.shape, FREE, np.int8)
    distance, _, stats = esdf_from_occupancy(
        occupancy, grid, max_distance=0.5, unknown_policy="free")
    return EsdfField(grid=grid, distance_grid=distance, max_distance=0.5, stats=stats,
                     static_shapes=tuple(shapes))


def test_box_and_plane_distances_are_exact():
    probe = np.array([[0.30, 0.0, 0.0], [0.10, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(analytic_distance(probe, [BOX]), [0.20, 0.0, -0.10])
    assert np.allclose(analytic_distance(probe, [FLOOR]), [0.0, 0.0, 0.0])
    # 도형이 없으면 무한대 — `min` 에서 아무 일도 하지 않는다.
    assert np.all(np.isinf(analytic_distance(probe, [])))


def test_gradients_match_finite_differences():
    """거리만 바꾸고 기울기를 안 바꾸면 더 나쁘다 — 값과 방향이 어긋난다."""
    rng = np.random.default_rng(0)
    probe = rng.uniform(-0.4, 0.4, size=(40, 3))
    shapes = [BOX, StaticPlane(point=np.array([0.0, 0.0, -0.35]), normal=np.array([0, 0, 1.0]))]
    analytic = analytic_gradient(probe, shapes)
    eps = 1e-6
    for axis in range(3):
        step = np.zeros(3); step[axis] = eps
        finite = (analytic_distance(probe + step, shapes)
                  - analytic_distance(probe - step, shapes)) / (2 * eps)
        assert np.allclose(finite, analytic[:, axis], atol=2e-4)


def test_an_unobserved_field_says_free_until_the_channel_is_added():
    """이 파일이 있는 이유 — 관측이 없으면 필드는 '멀리 떨어져 안전' 이라고 답한다."""
    probe = np.array([[0.12, 0.0, 0.0]])          # 상자 표면에서 20 mm
    blind = _empty_field()
    assert blind.distance(probe)[0] >= 0.49       # 포화 — 아무것도 안 보인다
    informed = _empty_field([BOX])
    assert np.isclose(informed.distance(probe)[0], 0.02, atol=1e-9)


def test_the_channel_only_ever_tightens():
    """`min(복셀, 해석적)` 이므로 답이 커질 수 없다 — 틀리는 방향이 안전한 쪽이다."""
    rng = np.random.default_rng(1)
    probe = rng.uniform(-0.28, 0.28, size=(60, 3))
    blind, informed = _empty_field(), _empty_field([BOX, FLOOR])
    assert np.all(informed.distance(probe) <= blind.distance(probe) + 1e-12)


def test_the_channel_works_outside_the_grid():
    """격자 밖이 이 채널이 가장 필요한 자리다 — 실측에서 선반과 벽이 거기 있었다."""
    far = np.array([[5.0, 0.0, 0.0]])             # 격자 밖
    informed = _empty_field([StaticBox(center=np.array([5.2, 0.0, 0.0]),
                                       half_extents=np.full(3, 0.05))])
    assert np.isclose(informed.distance(far)[0], 0.15, atol=1e-9)


def test_stamping_is_still_available_for_geometry_inside_the_grid():
    grid = VoxelGrid.from_bounds([-0.3, -0.3, -0.3], [0.3, 0.3, 0.3], VOXEL)
    occupancy = np.full(grid.shape, FREE, np.int8)
    changed, stats = stamp_static(occupancy, grid, [BOX])
    assert changed > 0 and stats["n_static_shapes"] == 1
    assert (occupancy == OCCUPIED).sum() == stats["n_static_voxels"]
    # 빈 목록은 아무 일도 하지 않는다.
    assert stamp_static(occupancy, grid, [])[0] == 0
