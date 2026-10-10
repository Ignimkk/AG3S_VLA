"""T44c A5 — trajopt hot path (`_esdf_clearance` 값 · `_esdf_directions` 방향) 가 target-free 의 새 합성을
**같은 branch** 로 읽는가.

linearize 는 합성을 다시 구현하지 않는다 — 권한 행 값은 `field.target_free_distance`, 방향은
`field.target_free_gradient` 에서 온다. 그래서 이 파일이 박는 것은 그 두 호출이 새 계층 (`c_free`) 이 답한
행에서 **같은 계층의 값과 기울기**를 주고, 권한 없는 행은 한 값도 안 바뀐다는 것이다.

| # | 성질 |
|---|---|
| 1 | 권한 행 clearance = `c − r − margin` (c 가 답한 곳), 방향 = c 계층의 기울기, `last_target_free_tiers` 에 새 이름 |
| 2 | 유한차분: clearance 의 공간 미분 = 방향 (c 가 선형인 장면) |
| 3 | 권한 없는 행 · off 필드 → 값 · 방향 비트 동일 |
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays
from benchmark.trajopt.linearize import CollisionLinearizer, SceneSnapshot

VC, VF = 0.02, 0.005
OC, NC = np.zeros(3), (20, 20, 20)
OF, NF = np.array([0.10, 0.10, 0.10]), (33, 33, 33)
N = np.array([0.3, -0.2, 0.9327379]) / np.linalg.norm([0.3, -0.2, 0.9327379])
R = np.array([0.004, 0.004, 0.02])


def _grid(origin, shape, vs):
    ax = [origin[i] + np.arange(shape[i]) * vs for i in range(3)]
    return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)


def _lin(p):
    return (p - np.array([0.18, 0.18, 0.18])) @ N


def _field(coarse=True):
    gc = _grid(OC, NC, VC)
    layers = (layer_from_arrays(_lin(gc) - 0.01, OC, VC),
              layer_from_arrays(np.full(NF, 0.5), OF, VF))
    free = (layer_from_arrays(np.full(NF, 0.05), OF, VF),)
    cl = layer_from_arrays(_lin(gc), OC, VC) if coarse else None
    return CuroboEsdfField(layers, outside_distance=0.5, target_free_layers=free,
                           target_free_coarse_layer=cl)


def _rig(field):
    lin = object.__new__(CollisionLinearizer)
    lin.query_radii = R.copy()
    lin.robot_radii = R.copy()
    lin.n_spheres = 3
    lin.fast_eval = False
    lin.horizon = 1
    scene = SceneSnapshot(
        candidate_pos=np.zeros((0, 3)), candidate_radius=np.zeros(0),
        candidate_active=np.zeros(0, bool), d_safe=np.zeros((3, 0)),
        plane_normal=np.zeros((0, 3)), plane_offset=np.zeros(0), plane_active=np.zeros(0, bool),
        robot_radii=R.copy(), candidate_ids=np.zeros(0, np.int64),
        esdf=field, esdf_margin=0.003,
        target_free_mask=np.array([True, True, False]))
    return lin, scene


# two fingers inside the window where c < 0.05 (c answers), one forearm sphere (not authorized)
P = np.array([[0.16, 0.17, 0.17], [0.19, 0.20, 0.16], [0.16, 0.17, 0.17]])


def test_authorized_rows_read_the_coarse_free_value_and_its_gradient():
    field = _field()
    lin, scene = _rig(field)
    cl = lin._esdf_clearance(P[None], scene)[0, :, 0]
    np.testing.assert_allclose(cl[:2], _lin(P[:2]) - R[:2] - 0.003, atol=1e-9)
    np.testing.assert_allclose(cl[2], field.distance(P[2:])[0] - R[2] - 0.003, atol=1e-12)
    g = lin._esdf_directions(P, np.arange(3), scene, 3)
    np.testing.assert_allclose(g[:2], np.broadcast_to(N, (2, 3)), atol=1e-5)
    np.testing.assert_array_equal(g[2], field.gradient(P[2:])[0])
    assert lin.last_target_free_tiers == {"target_free_coarse": 2}


def test_clearance_finite_difference_equals_the_direction():
    field = _field()
    lin, scene = _rig(field)
    g = lin._esdf_directions(P, np.arange(3), scene, 3)
    h = 1e-4
    for k in range(2):
        fd = []
        for e in np.eye(3):
            plus, minus = P.copy(), P.copy()
            plus[k] += h * e
            minus[k] -= h * e
            fd.append((lin._esdf_clearance(plus[None], scene)[0, k, 0]
                       - lin._esdf_clearance(minus[None], scene)[0, k, 0]) / (2 * h))
        np.testing.assert_allclose(fd, g[k], atol=1e-5)


def test_off_field_and_unauthorized_rows_are_unchanged():
    on, off = _field(), _field(coarse=False)
    lin_on, s_on = _rig(on)
    lin_off, s_off = _rig(off)
    c_on = lin_on._esdf_clearance(P[None], s_on)[0, :, 0]
    c_off = lin_off._esdf_clearance(P[None], s_off)[0, :, 0]
    assert c_on[2] == c_off[2]
    # off: the fine target-free layer (0.05) answers the authorized rows as before T44c.
    np.testing.assert_allclose(c_off[:2], 0.05 - R[:2] - 0.003, atol=1e-12)
    g_off = lin_off._esdf_directions(P, np.arange(3), s_off, 3)
    np.testing.assert_array_equal(g_off[:2], 0.0)
    assert lin_off.last_target_free_tiers == {"target_free": 2}
