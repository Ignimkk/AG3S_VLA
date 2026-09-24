"""A2 — 쥔 물체는 질의점이면서 동시에 장애물이면 안 된다.

파지가 닫히면 그 물체는 **로봇 쪽 질의점**이 된다 (E3 — 쥔 물체가 optimizer 에 도달하지
않는다 / F19 — 단일 primitive 가 아니라 점 기반). 그러면 장애물 쪽에서는 빠져야 한다.
양쪽에 다 있으면 물체가 자기 자신에게 부딪히고, 그 행은 **어떤 해로도 못 푼다** — 손에
강체로 붙어 있어 관절로는 자기 복셀에서 못 벗어난다.

고정하는 것:

1. **파낸다** — `attached_points` 를 주면 그 자리의 점유가 사라지고 질의 거리가 양수가 된다.
2. **`attach` 가 없으면 아무 일도 안 한다** — 기본 동작이 안 바뀐다.
3. **파내는 자리는 쥔 물체뿐** — 옆에 있는 다른 물체는 그대로 남는다 (E1 — 조작 대상을
   파내면 손끝뿐 아니라 전신에게 사라진다 — 의 교훈이 여기서도 유효하다).
4. **좌표 변환이 옳다** — `attached_points_in_base` 가 최적화기가 질의하는 점과 같은 점을 낸다.
5. **팽창이 단조롭게 듣는다** — 필드의 표면을 정하는 것은 TSDF 의 **영교차**라 원본 점이
   떨어진 복셀과 꼭 같지 않다(라벨 층이 겪은 것과 같은 문제). 얼마나 팽창해야 하는지는
   씬에 달려 있어 여기서 값을 고정하지 않고, **더 팽창하면 더 파이고 더 멀어진다**는
   방향만 고정한다. 실제 값은 `experiments/a2_field_exception.py` 가 실측으로 정한다.
"""

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig, AG3SConfigError
from benchmark.ag3s.fields.esdf import CameraDepth, EsdfBuilder

VOXEL = 0.02


def _config(**esdf):
    base = {"voxel_size": VOXEL, "max_distance": 0.4, "bounds_lower": [-0.5, -0.5, -0.1],
            "bounds_upper": [0.9, 0.5, 0.7]}
    base.update(esdf)
    return AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": base}).esdf


def _plate_scene(plates, *, hw=(60, 80), f=60.0):
    """`plates` 를 정면에서 본 depth 한 장. 카메라는 base 원점에서 +x 를 본다.

    점 몇 개를 찍는 대신 **판**을 그린다 — 성긴 점은 TSDF 가중치를 못 모아 점유가 안 생기고,
    그러면 테스트가 파내기가 아니라 "애초에 아무것도 없었다" 를 재게 된다 (실측으로 잡았다).
    """
    h, w = hw
    K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
    # 카메라 +z 가 base +x 를 보게.
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]).T
    T = np.eye(4)
    T[:3, :3] = R
    depth = np.full((h, w), 5.0)          # 배경은 멀리
    for v in range(h):
        for u in range(w):
            d = np.array([(u - K[0, 2]) / f, (v - K[1, 2]) / f, 1.0])
            dw = R @ d
            if dw[0] <= 1e-6:
                continue
            for (x, cy, cz, half) in plates:
                t = x / dw[0]
                pt = t * dw
                if abs(pt[1] - cy) < half and abs(pt[2] - cz) < half:
                    depth[v, u] = min(depth[v, u], float(t * np.linalg.norm(d)))
    return CameraDepth("cam", depth, K, T)


def _plate_points(x, cy, cz, half=0.05, n=4):
    g = np.linspace(-half * 0.8, half * 0.8, n)
    yy, zz = np.meshgrid(g, g)
    return np.column_stack([np.full(yy.size, x), (cy + yy).ravel(), (cz + zz).ravel()])


def test_carving_removes_the_held_object_from_the_field():
    cam = _plate_scene([(0.40, 0.0, 0.0, 0.06)])
    held = _plate_points(0.40, 0.0, 0.0)

    plain = EsdfBuilder(_config()).update([cam])
    carved = EsdfBuilder(_config()).update([cam], attached_points=held)

    # 파내기 전에는 쥔 물체의 점이 자기 표면 위에 있으므로 거리가 한 복셀 안쪽이다.
    assert float(plain.distance(held).min()) < VOXEL
    # 파낸 뒤에는 자기 자신이 사라져 거리가 커진다. 이것이 없으면 여유거리가
    # `0 - 0 - margin` 이 되어 영원히 못 푸는 행이 된다.
    assert float(carved.distance(held).min()) > float(plain.distance(held).min())
    assert carved.stats["n_attached_voxels_carved"] > 0
    assert plain.stats["n_attached_voxels_carved"] == 0


def test_no_attachment_changes_nothing():
    cam = _plate_scene([(0.40, 0.0, 0.0, 0.06)])
    held = _plate_points(0.40, 0.0, 0.0)
    a = EsdfBuilder(_config()).update([cam])
    b = EsdfBuilder(_config()).update([cam], attached_points=None)
    assert np.array_equal(a.distance_grid, b.distance_grid)
    assert b.stats["n_attached_voxels_carved"] == 0


def test_carving_spares_a_neighbouring_object():
    """E1 의 교훈 — 파내기는 **모두에게** 지운다. 쥔 것만 지워야 한다."""
    cam = _plate_scene([(0.40, -0.10, 0.0, 0.04), (0.40, 0.10, 0.0, 0.04)])
    held = _plate_points(0.40, -0.10, 0.0, half=0.03)
    other = _plate_points(0.40, 0.10, 0.0, half=0.03)

    plain = EsdfBuilder(_config()).update([cam])
    carved = EsdfBuilder(_config()).update([cam], attached_points=held)

    # 쥔 것은 멀어지고, 옆 물체는 그대로다.
    assert float(carved.distance(held).min()) > float(plain.distance(held).min())
    assert float(carved.distance(other).min()) == pytest.approx(
        float(plain.distance(other).min()), abs=1e-9)


def test_points_in_base_matches_the_optimizer_query_points():
    """필드가 파내는 점과 최적화기가 묻는 점은 **같아야** 한다."""
    from benchmark.ag3s.constraints.attached import attached_points_in_base
    from benchmark.ag3s.types import AttachedCollisionGeometry

    class _Model:
        def link_pose(self, q, link):
            assert link == "hand"
            T = np.eye(4)
            T[:3, 3] = [float(q[0]), 0.2, 0.3]
            return T

    local = np.array([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    att = AttachedCollisionGeometry(parent_link="hand", T_parent_object=np.eye(4),
                                    primitives=[], points=local)
    out = attached_points_in_base(att, robot_model=_Model(), robot_state=np.array([0.5]))
    assert np.allclose(out, local + np.array([0.5, 0.2, 0.3]))

    # 붙은 것이 없으면 None — 호출부가 조건문 없이 그대로 넘길 수 있어야 한다.
    assert attached_points_in_base(None, robot_model=_Model(),
                                   robot_state=np.array([0.5])) is None


def test_dilation_is_monotone():
    """팽창을 키우면 더 파이고 질의 거리가 더 멀어진다.

    **몇 복셀이 필요한지는 씬이 정한다.** 필드의 표면은 TSDF 의 영교차가 정하므로 관측 점이
    떨어진 복셀과 꼭 같지 않고(라벨 층이 겪은 것과 같은 문제), 그 어긋남은 물체의 크기·
    카메라 각도·복셀 크기에 따라 다르다. 여기서는 **방향만** 고정하고 값은 실측이 정한다.
    """
    cam = _plate_scene([(0.40, 0.0, 0.0, 0.06)])
    held = _plate_points(0.40, 0.0, 0.0)
    got = {}
    for dilate in (0, 1, 2):
        f = EsdfBuilder(_config(attached_dilate_voxels=dilate)).update(
            [cam], attached_points=held)
        got[dilate] = (f.stats["n_attached_voxels_carved"], float(f.distance(held).min()))
    assert got[2][0] > got[1][0] > got[0][0]    # 많이 팽창할수록 많이 파인다
    assert got[2][1] > got[1][1] >= got[0][1]   # 그리고 더 멀어진다


def test_negative_dilation_is_refused():
    with pytest.raises(AG3SConfigError, match="attached_dilate_voxels"):
        _config(attached_dilate_voxels=-1)
