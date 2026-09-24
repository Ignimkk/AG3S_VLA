"""쥔 물체가 optimizer 의 질의점으로 도달하는가 (E3 — 쥔 물체가 optimizer 에 도달하지 않던 문제).

`sphere_states` 는 이제 로봇 구 뒤에 **쥔 물체의 점**을 반지름 0 으로 붙여 내놓는다. 이 파일이
고정하는 것은 넷이다.

1. **배치가 정확한가** — 네 프로브(링크 원점과 세 단위축)의 아핀 결합으로 임의의 점을 놓는다.
   강체 변환은 아핀이므로 근사가 아니라 **항등식**이고, 오차는 부동소수점 수준이어야 한다.
2. **기울기가 맞는가** — 유한차분과 일치해야 한다. 여기가 틀리면 optimizer 는 쥔 물체를
   엉뚱한 방향으로 밀어낸다.
3. **ESDF 행이 실제로 늘어나는가** — 늘지 않으면 쥔 물체는 여전히 보이지 않는다.
4. **놓으면 되돌아오는가** — 파지는 청크 사이에 시작되고 끝난다.

**QP 의 희소 패턴은 질의점 수와 무관하다** (행 수는 `reduction.rows_per_step`). 그래서 파지가
롤아웃 도중 시작돼도 factorization 이 무효가 되지 않는다 — 그것도 아래에서 확인한다.
"""

import numpy as np
import pytest

from benchmark.ag3s.constraints.attached import base_frame_points, voxel_unique
from benchmark.ag3s.types import AttachedCollisionGeometry
from benchmark.trajopt.linearize import SceneSnapshot

from .fixtures import make_scenario

LINK = "ee_finger_l1"


class _FlatField:
    """모든 점에서 같은 거리를 답하는 가짜 거리장. 여유거리 산술만 검사한다."""

    class grid:
        voxel_size = 0.02

    def __init__(self, distance: float = 0.4):
        self._d = float(distance)

    def distance(self, points):
        return np.full(len(points), self._d)

    def gradient(self, points):
        g = np.zeros((len(points), 3))
        g[:, 2] = 1.0
        return g


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(horizon=5)


@pytest.fixture
def held(scenario):
    rng = np.random.default_rng(0)
    return rng.normal(scale=0.03, size=(23, 3))


def test_placement_is_exact(scenario, held):
    """네 프로브의 아핀 결합이 AG3S 의 수치 FK 와 같은 답을 낸다."""
    lin = scenario.linearizer
    lin.set_attached(held, LINK)
    try:
        centres, _ = lin.sphere_states(scenario.reference, scenario.q_now)
        attached = AttachedCollisionGeometry(
            parent_link=LINK, T_parent_object=np.eye(4), primitives=[], points=held
        )
        full = scenario.layout.full_q(scenario.reference, scenario.q_now)
        for h in range(centres.shape[0]):
            truth = base_frame_points(attached, scenario.robot, np.asarray(full[:, h]).reshape(-1))
            assert np.allclose(centres[h, lin.n_spheres:], truth, atol=1e-12)
    finally:
        lin.set_attached(None, None)


def test_gradient_matches_finite_differences(scenario, held):
    """기울기가 틀리면 optimizer 는 쥔 물체를 엉뚱한 방향으로 민다."""
    lin = scenario.linearizer
    lin.set_attached(held, LINK)
    try:
        traj, q_now = scenario.reference, scenario.q_now
        centres, jac = lin.sphere_states(traj, q_now)
        eps = 1e-6
        for step in (0, centres.shape[0] - 1):
            for column in range(scenario.layout.nq_opt):
                bumped = traj.copy()
                bumped[column, step] += eps
                moved, _ = lin.sphere_states(bumped, q_now)
                finite = (moved[step, lin.n_spheres:] - centres[step, lin.n_spheres:]) / eps
                assert np.allclose(finite, jac[step, lin.n_spheres:, :, column], atol=1e-5)
    finally:
        lin.set_attached(None, None)


def test_held_object_produces_its_own_esdf_rows(scenario, held):
    """쥔 물체의 점마다 행이 하나씩 늘고, 반지름 0 으로 여유거리가 계산된다."""
    lin = scenario.linearizer
    scene = SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((lin.n_spheres, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=lin.robot_radii, candidate_ids=np.zeros(0, np.int64),
        esdf=_FlatField(0.4), esdf_margin=0.05,
    )
    before = lin.esdf_clearance(scenario.reference, scenario.q_now, scene)
    assert before.shape[1] == lin.n_spheres

    lin.set_attached(held, LINK)
    try:
        after = lin.esdf_clearance(scenario.reference, scenario.q_now, scene)
        assert after.shape[1] == lin.n_spheres + len(held)
        # 반지름 0 · 마진 0.05 · 거리 0.4 이므로 정확히 0.35 여야 한다.
        assert np.allclose(after[:, lin.n_spheres:, 0], 0.35)
        # 로봇 구 쪽은 건드려지지 않는다.
        assert np.allclose(after[:, : lin.n_spheres], before)
    finally:
        lin.set_attached(None, None)


def test_detach_restores_the_robot_only_block(scenario, held):
    lin = scenario.linearizer
    plain, _ = lin.sphere_states(scenario.reference, scenario.q_now)
    lin.set_attached(held, LINK)
    lin.set_attached(None, None)
    again, _ = lin.sphere_states(scenario.reference, scenario.q_now)
    assert again.shape == plain.shape
    assert np.allclose(again, plain)
    assert lin.query_radii.shape[0] == lin.n_spheres


def test_qp_sparsity_does_not_depend_on_the_held_object(scenario, held):
    """행 수는 `reduction.rows_per_step` 이 정한다 — 파지가 factorization 을 무효화하면 안 된다."""
    lin = scenario.linearizer
    scene = SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((lin.n_spheres, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=lin.robot_radii, candidate_ids=np.zeros(0, np.int64),
        esdf=_FlatField(0.4), esdf_margin=0.05,
    )
    args = (scenario.reference, scenario.q_now, scene, scenario.config)
    plain = lin.linearize(*args)
    lin.set_attached(held, LINK)
    try:
        with_object = lin.linearize(*args)
    finally:
        lin.set_attached(None, None)
    assert with_object.value.shape == plain.value.shape
    assert with_object.gradient.shape == plain.gradient.shape


def test_voxel_unique_is_deterministic_and_shrinks():
    rng = np.random.default_rng(1)
    points = rng.normal(scale=0.03, size=(4000, 3))
    coarse = voxel_unique(points, 0.010)
    assert len(coarse) < len(points)
    assert np.array_equal(coarse, voxel_unique(points, 0.010))
    # 모든 대표점은 원본 안에 있다 — 무게중심이 아니라 첫 점을 남기기 때문.
    assert len(voxel_unique(points, 0.020)) < len(coarse)
    assert voxel_unique(np.zeros((0, 3)), 0.01).shape == (0, 3)


class _LabelledField(_FlatField):
    """`x > 0` 쪽 표면만 목적지라고 답하는 가짜 필드."""

    has_labels = True

    def is_label(self, points, name):
        points = np.asarray(points).reshape(-1, 3)
        if name != "destination":
            return np.zeros(len(points), bool)
        return points[:, 0] > 0.0


def _scene(lin, **extra):
    from benchmark.trajopt.linearize import SceneSnapshot

    base = dict(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((lin.n_spheres, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=lin.robot_radii, candidate_ids=np.zeros(0, np.int64),
        esdf=_LabelledField(0.4), esdf_margin=0.05,
    )
    base.update(extra)
    return SceneSnapshot(**base)


def test_destination_margin_applies_only_to_the_held_object(scenario, held):
    """목적지 마진은 **쥔 물체에만** 붙는다 — 로봇 팔·몸통은 전역 마진을 유지한다.

    처음 구현에서 "가장 가까운 것이 목적지인 모든 질의점" 에 걸었더니 로봇 구의 여유거리까지
    함께 완화됐다 (실측: 프레임 18 에서 -77.8 -> -47.8 mm). E1 — 조작 대상을 필드에서 파내면
    손끝뿐 아니라 전신에게 사라진다 — 과 같은 종류의 실수다. 바구니 안으로 들어가야 하는 것은
    쥔 물체이지 팔꿈치가 아니다.
    """
    lin = scenario.linearizer
    plain = _scene(lin)
    dest = _scene(lin, destination_label="destination", destination_margin=0.02)

    lin.set_attached(held, LINK)
    try:
        without = lin.esdf_clearance(scenario.reference, scenario.q_now, plain)
        with_dest = lin.esdf_clearance(scenario.reference, scenario.q_now, dest)
        centres, _ = lin.sphere_states(scenario.reference, scenario.q_now)
    finally:
        lin.set_attached(None, None)

    # 로봇 구 블록은 **완전히** 같아야 한다.
    assert np.array_equal(with_dest[:, : lin.n_spheres], without[:, : lin.n_spheres])

    # 쥔 물체 중 목적지 쪽(x > 0)에 있는 점만 30 mm 만큼 완화된다 (0.05 -> 0.02).
    held_block = slice(lin.n_spheres, None)
    on_dest = centres[:, held_block, 0] > 0.0
    delta = (with_dest[:, held_block, 0] - without[:, held_block, 0])
    assert np.allclose(delta[on_dest], 0.03)
    assert np.allclose(delta[~on_dest], 0.0)


def test_no_destination_changes_nothing(scenario, held):
    lin = scenario.linearizer
    lin.set_attached(held, LINK)
    try:
        a = lin.esdf_clearance(scenario.reference, scenario.q_now, _scene(lin))
        b = lin.esdf_clearance(scenario.reference, scenario.q_now,
                               _scene(lin, destination_label=None, destination_margin=0.02))
    finally:
        lin.set_attached(None, None)
    assert np.array_equal(a, b)


def test_unknown_destination_name_fails_closed(scenario, held):
    """이름이 라벨 층에 없으면 완화가 일어나지 않는다 — 닫히는 방향."""
    lin = scenario.linearizer
    lin.set_attached(held, LINK)
    try:
        a = lin.esdf_clearance(scenario.reference, scenario.q_now, _scene(lin))
        b = lin.esdf_clearance(scenario.reference, scenario.q_now,
                               _scene(lin, destination_label="nowhere", destination_margin=0.02))
    finally:
        lin.set_attached(None, None)
    assert np.array_equal(a, b)
