"""라벨 층 — 거리장이 "가장 가까운 것이 **무엇인지**" 를 함께 답한다.

거리장은 원래 익명이다. "가장 가까운 표면까지 얼마"만 답하고 그것이 무엇인지는 말하지 않으므로,
"목적지에는 얇은 마진, 나머지는 전부 50 mm" 같은 정책을 걸 수 없다 (F18 — 전역 50 mm 가 담기
동작을 구조적으로 막는 문제). 라벨 층이 그 익명을 푼다.

비용이 거의 없다는 것이 이 설계의 핵심이다. **거리 변환은 최근접 표면 복셀을 이미 계산한다** —
`distance_transform_edt(return_indices=True)` 가 그것을 돌려주므로 라벨은 새 알고리즘이 아니라
조회 한 번이다. 아래 `test_labels_do_not_change_the_distance_field` 가 그 "거의 없음" 을 고정한다.
"""

import numpy as np

from AG3S.ag3s.fields.esdf import FREE, OCCUPIED, EsdfField, VoxelGrid, esdf_from_occupancy

VOXEL = 0.02


def _two_spheres(gap: float = 0.30, radius: float = 0.06):
    """x 축 위에 라벨이 다른 구 둘. 경계가 어디여야 하는지 답을 알고 있는 씬."""
    grid = VoxelGrid.from_bounds([-0.5, -0.2, -0.2], [0.5, 0.2, 0.2], VOXEL)
    occupancy = np.full(grid.shape, FREE, np.int8)
    labels = np.full(grid.shape, -1, np.int32)
    index = np.stack(np.meshgrid(*[np.arange(n) for n in grid.shape], indexing="ij"), -1)
    centres = grid.origin + index * VOXEL
    for label, sign in ((0, -1.0), (1, +1.0)):
        inside = np.linalg.norm(centres - np.array([sign * gap, 0.0, 0.0]), axis=-1) <= radius
        occupancy[inside] = OCCUPIED
        labels[inside] = label
    return grid, occupancy, labels


def _field(grid, occupancy, labels):
    distance, label_grid, stats = esdf_from_occupancy(
        occupancy, grid, max_distance=0.5, unknown_policy="free", labels=labels
    )
    return EsdfField(grid=grid, distance_grid=distance, max_distance=0.5, stats=stats,
                     label_grid=label_grid, label_names=("A", "B"))


def test_nearest_surface_decides_the_label():
    """라벨은 **질의점이 속한 복셀**의 것이 아니라 그 거리를 만든 표면의 것이다."""
    field = _field(*_two_spheres())
    left = np.stack([np.linspace(-0.45, -0.05, 9), np.zeros(9), np.zeros(9)], axis=1)
    right = np.stack([np.linspace(0.05, 0.45, 9), np.zeros(9), np.zeros(9)], axis=1)
    assert np.all(field.is_label(left, "A"))
    assert np.all(field.is_label(right, "B"))
    # 빈 공간의 복셀은 자기 라벨이 없다 — 그래도 답이 나온다는 것이 요점.
    assert np.all(field.label(left) == 0)


def test_labels_do_not_change_the_distance_field():
    """라벨 층은 거리에 손대지 않는다. 같은 씬에서 거리 격자가 **바이트 단위로** 같아야 한다."""
    grid, occupancy, labels = _two_spheres()
    with_labels, label_grid, _ = esdf_from_occupancy(
        occupancy, grid, max_distance=0.5, unknown_policy="free", labels=labels)
    without, none_grid, _ = esdf_from_occupancy(
        occupancy, grid, max_distance=0.5, unknown_policy="free")
    assert np.array_equal(with_labels, without)
    assert none_grid is None
    assert label_grid is not None


def test_unknown_name_fails_closed():
    """모르는 이름은 예외가 아니라 `-1` 이고, 어떤 점과도 일치하지 않는다.

    완화는 **일치할 때만** 일어나므로, 모르는 이름이 완화를 부르는 일은 없다 — 닫히는 방향이다.
    """
    field = _field(*_two_spheres())
    probe = np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]])
    assert field.label_id("destination") == -1
    assert not field.is_label(probe, "destination").any()


def test_label_is_absent_when_none_were_given():
    grid, occupancy, _ = _two_spheres()
    distance, label_grid, _ = esdf_from_occupancy(
        occupancy, grid, max_distance=0.5, unknown_policy="free")
    field = EsdfField(grid=grid, distance_grid=distance, max_distance=0.5)
    assert not field.has_labels
    assert label_grid is None
    probe = np.array([[0.0, 0.0, 0.0]])
    assert field.label(probe)[0] == -1
    assert not field.is_label(probe, "A")[0]


def test_ties_are_arbitrary_and_that_is_recorded():
    """두 물체에서 **같은 거리**인 점의 라벨은 임의로 정해진다 — 알고 있는 한계다.

    거리 변환은 동점에서 어느 site 를 고를지 보장하지 않는다. 마진을 라벨로 고르는 소비 쪽은
    그래서 동점 근처에서 **얇은 쪽을 고를 수 있다** — 위험한 방향이다. 지금 씬에서는 목적지
    안쪽 표면이 전부 같은 물체라 발동하지 않지만, 마진이 다른 두 물체가 한 복셀 안에서 만나는
    씬이 생기면 소비 쪽에서 보수적으로 풀어야 한다.

    **전환 신호**: 마진이 서로 다른 두 물체가 복셀 하나 안에서 맞닿는 씬이 생기는 것.
    """
    field = _field(*_two_spheres())
    midpoint = np.array([[0.0, 0.0, 0.0]])
    assert field.label(midpoint)[0] in (0, 1)
    # 그 지점의 거리는 양쪽에서 같다 — 그것이 동점이라는 뜻이다.
    a = field.distance(np.array([[-0.001, 0.0, 0.0]]))[0]
    b = field.distance(np.array([[+0.001, 0.0, 0.0]]))[0]
    assert abs(a - b) < 1e-6
