"""T21 — `worst_row` 신원이 **어느 계층이 답했는지**와 **권한 행의 검증 가능 여부**를 싣는다.

T24 가 막힌 곳: T17 접근 구간의 QP 행이 전부 `ee_finger_l1/l2` 의 coarse 20 mm 행이었는데, 그 사실은
오프라인 재생으로만 알 수 있었다. 이제 신원에 두 키가 있다.

* `tier` — ESDF 행이면 그 답을 낸 계층 (`coarse`·`fine`·`target_free`·`window_edge`·`outside`·`static`).
* `authorized_rows` — 권한 link 행 **전부**(모든 step)의 계층별 수와 `verifiable`. 창 밖(coarse·outside)
  답은 자유 공간도 접촉 허용도 아니다 — `verifiable=False` 와 `note` 로 **검증 불가를 명시**한다 (지침 §5.3-4).

값(`clearance_m`)은 바뀌지 않는다 — 이름표만 붙는다.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmark.ag3s.fields.curobo_field import ANSWER_TIERS, CuroboEsdfField, layer_from_arrays
from tests.trajopt.fixtures import empty_scene, make_scenario

AUTH = slice(57, 61)          # 팔 끝 4 구를 "권한 link" 로 둔다 (이 fixture 에는 손가락이 없다)


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(horizon=8, penetration=0.03)


def _grid(origin, shape, vs):
    ax = [origin[i] + np.arange(shape[i]) * vs for i in range(3)]
    return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)


def _field(obstacle, radius, *, fine_lo, fine_hi, fine_vs=0.01):
    """한 구 장애물. 거친 40 mm 는 작업공간, 미세 10 mm 는 `[fine_lo, fine_hi]` 창, target 없는 층도 같은 창.

    target 없는 층에서는 장애물을 **뺀다** (그것이 사과라는 뜻) — 그래서 권한 행은 그 층에서 멀어진다.
    """
    obstacle = np.asarray(obstacle, float)
    c_origin, c_shape, c_vs = obstacle - 1.2, (61, 61, 61), 0.04
    dist = lambda p: np.linalg.norm(p - obstacle, axis=-1) - radius  # noqa: E731
    coarse = layer_from_arrays(dist(_grid(c_origin, c_shape, c_vs)) + 0.004, c_origin, c_vs)
    fine_lo = np.asarray(fine_lo, float)
    shape = tuple(int(v) for v in np.ceil((np.asarray(fine_hi) - fine_lo) / fine_vs) + 1)
    pts = _grid(fine_lo, shape, fine_vs)
    fine = layer_from_arrays(dist(pts), fine_lo, fine_vs)
    free = layer_from_arrays(np.full(shape, 0.5, np.float32), fine_lo, fine_vs)
    return CuroboEsdfField((coarse, fine), outside_distance=0.5, target_free_layers=(free,),
                           stats={})


def _scene(s, field, *, mask=True, **kw):
    n = s.linearizer.n_spheres
    target_free = None
    if mask:
        target_free = np.zeros(n, bool)
        target_free[AUTH] = True
    return dataclasses.replace(empty_scene(s.linearizer), esdf=field, esdf_margin=0.05,
                               target_free_mask=target_free, **kw)


def _obstacle(s):
    centres = s.linearizer.sphere_states(s.reference, s.q_now)[0]
    return centres, centres[centres.shape[0] // 2, 0]


def test_an_esdf_row_says_which_tier_answered_and_the_value_does_not_move(scenario):
    s = scenario
    centres, obstacle = _obstacle(s)
    field = _field(obstacle, 0.30, fine_lo=obstacle - 0.3, fine_hi=obstacle + 0.3)
    scene = _scene(s, field, mask=False)
    worst, pair = s.linearizer.worst_row(s.reference, s.q_now, scene)
    assert pair["block"] == "esdf" and pair["tier"] in ANSWER_TIERS
    expected = field.answer_tier(np.asarray(pair["point_m"])[None])[0]
    assert pair["tier"] == expected
    assert worst == pytest.approx(s.linearizer.full_violation(s.reference, s.q_now, scene))
    assert pair["authorized_rows"] is None          # 권한 행이 없다


def test_a_window_away_from_the_authorized_rows_is_declared_unverifiable(scenario):
    """T14 seq 12–21 의 축소판: 창이 crate 쪽에 있으면 권한 행은 coarse 가 답한다."""
    s = scenario
    centres, obstacle = _obstacle(s)
    far = obstacle + np.array([2.0, 2.0, 0.0])
    field = _field(obstacle, 0.30, fine_lo=far - 0.2, fine_hi=far + 0.2)
    scene = _scene(s, field)
    worst, pair = s.linearizer.worst_row(s.reference, s.q_now, scene)
    rows = pair["authorized_rows"]
    n = centres.shape[0] * (AUTH.stop - AUTH.start)
    assert rows["n"] == n and rows["reads_target_free"] is True
    assert rows["by_tier"]["coarse"] + rows["by_tier"]["outside"] == n
    assert rows["verifiable"] is False and rows["n_unresolved"] == n
    assert "unverified" in rows["note"] and "not a contact permission" in rows["note"]
    assert worst == pytest.approx(s.linearizer.full_violation(s.reference, s.q_now, scene))


def test_a_window_over_the_authorized_rows_is_verifiable(scenario):
    s = scenario
    centres, obstacle = _obstacle(s)
    auth = centres[:, AUTH].reshape(-1, 3)
    lo, hi = auth.min(axis=0) - 0.6, auth.max(axis=0) + 0.6    # 여유 > 0.5 (free 층의 값)
    field = _field(obstacle, 0.30, fine_lo=lo, fine_hi=hi, fine_vs=0.02)
    scene = _scene(s, field)
    _, pair = s.linearizer.worst_row(s.reference, s.q_now, scene)
    rows = pair["authorized_rows"]
    assert rows["verifiable"] is True and rows["note"] is None
    assert rows["by_tier"]["target_free"] == rows["n"]


def test_under_relax_the_authorized_rows_are_the_relaxed_ones(scenario):
    """정책 `relax` 에는 마스크가 없다 — 접촉 허용이 마진을 내린 link 이 권한 행이다."""
    s = scenario
    centres, obstacle = _obstacle(s)
    far = obstacle + np.array([2.0, 2.0, 0.0])
    field = _field(obstacle, 0.30, fine_lo=far - 0.2, fine_hi=far + 0.2)
    margin = np.full(s.linearizer.n_spheres, 0.05)
    margin[AUTH] = 0.0
    scene = _scene(s, field, mask=False, manipulated_link_margin=margin)
    _, pair = s.linearizer.worst_row(s.reference, s.q_now, scene)
    rows = pair["authorized_rows"]
    assert rows["reads_target_free"] is False
    assert rows["n"] == centres.shape[0] * (AUTH.stop - AUTH.start)


def test_a_field_that_cannot_name_its_tiers_says_none(scenario):
    """legacy `EsdfField` 에는 계층이 하나뿐이다 — 이름을 지어내지 않는다."""
    from benchmark.ag3s.fields.esdf import EsdfField, VoxelGrid

    s = scenario
    _, obstacle = _obstacle(s)
    grid = VoxelGrid.from_bounds(obstacle - 0.6, obstacle + 0.6, 0.02)
    d = np.linalg.norm(grid.centres() - obstacle, axis=-1) - 0.30
    field = EsdfField(grid=grid, distance_grid=d.astype(np.float32), max_distance=0.6,
                      stats={}, outside_distance=0.6)
    scene = dataclasses.replace(empty_scene(s.linearizer), esdf=field, esdf_margin=0.05)
    _, pair = s.linearizer.worst_row(s.reference, s.q_now, scene)
    assert pair["block"] == "esdf" and pair["tier"] is None and pair["authorized_rows"] is None


def test_non_esdf_rows_carry_the_keys_as_none(scenario):
    s = scenario
    scene = empty_scene(s.linearizer)
    scene = dataclasses.replace(scene, plane_active=np.array([True, False]))
    _, pair = s.linearizer.worst_row(s.reference, s.q_now, scene)
    assert pair["block"] == "plane"
    assert pair["tier"] is None and pair["authorized_rows"] is None
