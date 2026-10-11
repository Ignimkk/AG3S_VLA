"""`max_violation_pair` — **위반을 만든 제약의 신원.**

`T5f` 가 여기서 막혔다: 판정에는 `max_violation_m` 한 숫자만 실려 있고, 그 숫자를 만든 행이
어느 link 대 어느 obstacle 인지는 최적화기 안에만 있었다. 숫자만 보고는 고칠 수 없다 —
팔꿈치가 탁자를 스친 것과 손끝이 사과를 파고든 것은 전혀 다른 일이다.

이 파일이 고정하는 것 셋:

1. `worst_row` 의 **값이 `full_violation` 과 같다** — 신원이 다른 계산에서 나오면 그 신원은
   보고된 숫자의 신원이 아니다.
2. 신원이 **해석 가능하다** — 구 인덱스가 아니라 link 이름이다.
3. 활성 제약이 없으면 **`None`** 이다 — 없는 신원을 지어내지 않는다.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.trajopt import wire
from AG3S.trajopt.sqp import TrajectoryOptimizer
from tests.trajopt.fixtures import empty_scene, make_scenario


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(horizon=8, penetration=0.03)


# --- 1. 값이 갈라지지 않는다 -------------------------------------------------------------
def test_worst_row_agrees_with_full_violation(scenario):
    s = scenario
    worst, pair = s.linearizer.worst_row(s.reference, s.q_now, s.scene)

    assert pair is not None
    assert worst == pytest.approx(
        s.linearizer.full_violation(s.reference, s.q_now, s.scene))
    # 신원이 실어 온 값이 곧 그 숫자여야 한다.
    assert pair["clearance_m"] == pytest.approx(worst)


def test_the_identity_points_at_a_real_row(scenario):
    """`step`·`query` 를 다시 재면 같은 값이 나온다 — 인덱스가 뜻을 갖는다."""
    s = scenario
    worst, pair = s.linearizer.worst_row(s.reference, s.q_now, s.scene)
    candidate, plane, _ = s.linearizer._clearances_from(
        s.linearizer.sphere_states(s.reference, s.q_now)[0], s.scene)

    block = {"candidate": candidate, "plane": plane}[pair["block"]]
    assert block[pair["step"], pair["query"], pair["slot"]] == pytest.approx(worst)


# --- 2. 신원이 해석 가능하다 -------------------------------------------------------------
def test_the_identity_names_a_link_not_a_sphere_index(scenario):
    s = scenario
    _, pair = s.linearizer.worst_row(s.reference, s.q_now, s.scene)

    assert pair["link"] in s.robot.sphere_link_names
    assert not pair["link"].startswith("sphere[")


def test_the_identity_carries_the_candidate_and_the_point(scenario):
    """fixture 의 장애물은 slot 0 하나뿐이고 `candidate_ids` 는 `arange` 다."""
    s = scenario
    _, pair = s.linearizer.worst_row(s.reference, s.q_now, s.scene)

    assert pair["block"] == "candidate"
    assert pair["slot"] == 0
    assert pair["candidate_id"] == 0
    # ESDF 가 아니므로 이름 붙일 라벨이 없다 — 모르는 것을 지어내지 않는다.
    assert pair["obstacle"] is None
    # 그림에 바로 찍을 수 있는 좌표.
    centres = s.linearizer.sphere_states(s.reference, s.q_now)[0]
    np.testing.assert_allclose(pair["point_m"],
                               centres[pair["step"], pair["query"]])


def test_the_key_set_is_fixed():
    """A2 가 프레임을 세로로 세므로 키가 프레임마다 달라지면 파서가 갈린다."""
    s = make_scenario(horizon=6)
    _, pair = s.linearizer.worst_row(s.reference, s.q_now, s.scene)

    # T21 이 `tier`·`authorized_rows` 를 더했다 — 어느 계층이 답했는가 / 권한 행의 검증 가능 여부.
    # 두 키는 **모든 행에 항상** 실린다 (값이 없으면 None) — 키 집합이 프레임마다 고정이라는 취지는 그대로다.
    assert set(pair) == {"block", "clearance_m", "step", "query", "link", "slot",
                         "candidate_id", "obstacle", "point_m", "tier", "authorized_rows"}


# --- 3. 없으면 없다고 한다 ---------------------------------------------------------------
def test_an_empty_scene_has_no_worst_row(scenario):
    """활성 제약이 하나도 없으면 모든 행이 `+inf` 다 — argmin 에 뜻이 없다."""
    s = scenario
    scene = empty_scene(s.linearizer)
    worst, pair = s.linearizer.worst_row(s.reference, s.q_now, scene)

    assert pair is None
    assert not np.isfinite(worst)
    assert worst == pytest.approx(
        s.linearizer.full_violation(s.reference, s.q_now, scene))


def test_a_cleared_plane_is_not_an_absent_plane(scenario):
    """평면이 만족돼 있어도 활성이면 행이 있다 — 그때 신원은 `plane` 일 수 있다."""
    s = scenario
    scene = empty_scene(s.linearizer)
    scene = type(scene)(**{**{f.name: getattr(scene, f.name)
                              for f in scene.__dataclass_fields__.values()},
                           "plane_active": np.array([True, False])})
    _, pair = s.linearizer.worst_row(s.reference, s.q_now, scene)

    assert pair is not None
    assert pair["block"] == "plane"
    assert pair["slot"] == 0
    assert pair["candidate_id"] is None


# --- 4. SQP 가 그것을 metrics 로 내보낸다 -----------------------------------------------
def test_the_optimizer_reports_the_pair(scenario):
    s = scenario
    opt = TrajectoryOptimizer(s.robot, s.layout, s.limits, s.config)
    result = opt.solve(s.reference, s.q_now, s.scene, template=s.chunk)

    pair = result.metrics["max_violation_pair"]
    assert pair is not None
    # 보고된 숫자와 같은 행이어야 한다 — `violation = max(0, -clearance)`.
    assert pair["clearance_m"] == pytest.approx(result.metrics["clearance_m"])
    assert result.max_violation == pytest.approx(max(0.0, -pair["clearance_m"]))


def test_an_unconstrained_solve_still_has_the_key(scenario):
    """제약이 없으면 `None` 이지만 **키는 있다** — 없음과 키 부재는 다른 뜻이다."""
    s = scenario
    opt = TrajectoryOptimizer(s.robot, s.layout, s.limits, s.config)
    result = opt.solve(s.reference, s.q_now, None, template=s.chunk)

    assert "max_violation_pair" in result.metrics
    assert result.metrics["max_violation_pair"] is None


# --- 5. 와이어 왕복 ---------------------------------------------------------------------
_PAIR = {"block": "esdf", "clearance_m": -0.031, "step": 12, "query": 5,
         "link": "ee_finger_r1", "slot": None, "candidate_id": None,
         "obstacle": "apple", "point_m": [0.51, 0.09, 0.84]}


def _verdict(**kw):
    base = dict(ag3s_status="ok", geometry_certified=True, trajopt_status="violated",
                max_violation_m=0.031, safe=False)
    return wire.SafetyVerdict(**{**base, **kw})


def test_the_pair_rides_the_response_and_comes_back():
    response = wire.pack_response(np.zeros((4, wire.ACTION_WIDTH)),
                                 _verdict(max_violation_pair=_PAIR),
                                 seq=3, timing_ms={"total": 1.0})
    assert response[wire.VIOLATION_PAIR] == _PAIR
    assert wire.unpack_violation_pair(response) == _PAIR


def test_a_frame_without_a_pair_carries_no_key():
    """정상 프레임의 응답을 T0 때와 같게 둔다 — `ag3s` 블록과 같은 규약이다."""
    response = wire.pack_response(np.zeros((4, wire.ACTION_WIDTH)), _verdict(),
                                 seq=3, timing_ms={})
    assert wire.VIOLATION_PAIR not in response
    assert wire.unpack_violation_pair(response) is None


def test_the_rest_of_the_verdict_is_untouched():
    with_pair = _verdict(max_violation_pair=_PAIR).to_dict()
    without = _verdict().to_dict()
    assert set(with_pair) - set(without) == {wire.VIOLATION_PAIR}
    assert {k: v for k, v in with_pair.items() if k != wire.VIOLATION_PAIR} == without


# --- 6. ESDF 행 — 라벨 층이 obstacle 에 **이름**을 준다 ----------------------------------
def _labelled_field(centre, radius, *, name: str, voxel=0.02, half_extent=0.6):
    """한 구의 정확한 부호 거리 + 전 복셀에 그 구의 라벨. 해석적으로 짓는다.

    필드를 depth 에서 적분해 만들면 이 테스트가 지각의 정확도를 재게 된다. 여기서 재려는 것은
    *"신원이 label 층을 읽어 오는가"* 하나다.
    """
    from AG3S.ag3s.fields.esdf import EsdfField, VoxelGrid

    centre = np.asarray(centre, float)
    grid = VoxelGrid.from_bounds(centre - half_extent, centre + half_extent, voxel)
    d = np.linalg.norm(grid.centres() - centre, axis=-1) - float(radius)
    return EsdfField(
        grid=grid, distance_grid=np.clip(d, -0.6, 0.6).astype(np.float32),
        max_distance=0.6, stats={"unknown_fraction": 0.0}, outside_distance=0.6,
        label_grid=np.zeros(grid.shape, np.int32), label_names=(name,))


def test_an_esdf_row_names_the_obstacle_it_hit(scenario):
    """거리장은 원래 익명이다. 라벨 층이 있으면 신원이 그 이름을 실어야 한다."""
    import dataclasses

    s = scenario
    centres = s.linearizer.sphere_states(s.reference, s.q_now)[0]
    # 궤적 위의 한 점을 그대로 덮는 필드 — 반드시 ESDF 행이 최악이 된다.
    middle = centres.shape[0] // 2
    field = _labelled_field(centres[middle, 0], 0.30, name="apple")
    scene = dataclasses.replace(empty_scene(s.linearizer), esdf=field, esdf_margin=0.05)

    worst, pair = s.linearizer.worst_row(s.reference, s.q_now, scene)
    assert pair["block"] == "esdf"
    assert pair["obstacle"] == "apple"
    # ESDF 는 slot 을 열거하지 않는다 — 그것이 이 backend 가 싼 이유다.
    assert pair["slot"] is None
    assert pair["candidate_id"] is None
    assert worst == pytest.approx(
        s.linearizer.full_violation(s.reference, s.q_now, scene))


def test_an_unlabelled_field_says_it_does_not_know(scenario):
    """라벨 없는 backend 에서는 `None` 이다 — "이름을 모른다" 이고 "물체가 없다" 가 아니다."""
    import dataclasses

    s = scenario
    centres = s.linearizer.sphere_states(s.reference, s.q_now)[0]
    field = _labelled_field(centres[centres.shape[0] // 2, 0], 0.30, name="apple")
    field = dataclasses.replace(field, label_grid=None, label_names=())
    scene = dataclasses.replace(empty_scene(s.linearizer), esdf=field, esdf_margin=0.05)

    _, pair = s.linearizer.worst_row(s.reference, s.q_now, scene)
    assert pair["block"] == "esdf"
    assert pair["obstacle"] is None
