"""ESDF backend — 기하가 도형으로 축약되지 않는다는 것이 이 파일이 지키는 것.

primitive backend 를 대체하는 것이 아니라 나란히 두는 것이므로, 여기서 재는 것은 "ESDF 가 더
낫다"가 아니라 **"ESDF 가 말한 대로 동작한다"** 이다: 거리가 참값과 복셀 반 칸 안에서 맞고,
틀리는 방향이 보수적이고, 국소 갱신이 전역과 같은 값을 내고, target 제외가 후보를 지우지 않는다.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig, AG3SConfigError, EsdfConfig
from benchmark.ag3s.fields.esdf import (
    FREE, OCCUPIED, UNKNOWN, CameraDepth, EsdfBuilder, EsdfField, TsdfVolume, VoxelGrid,
    build_once, carve, esdf_from_occupancy, occupied_mask,
)

SPHERE_R = 0.20


def _sphere_cameras(positions=((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)), h=160, w=200):
    """원점의 반지름 `SPHERE_R` 구를 보는 합성 깊이 카메라들. 해석적 정답이 있는 유일한 씬이다."""
    K = np.array([[220.0, 0, w / 2], [0, 220.0, h / 2], [0, 0, 1.0]])
    cams = []
    for pos in positions:
        p = np.asarray(pos, float)
        f = -p / np.linalg.norm(p)
        r = np.cross(np.array([0.0, 0, 1.0]), f)
        r = np.array([1.0, 0, 0]) if np.linalg.norm(r) < 1e-6 else r / np.linalg.norm(r)
        up = np.cross(f, r)
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = r, -up, f, p
        v, u = np.mgrid[0:h, 0:w]
        d = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1)
        dn = d / np.linalg.norm(d, axis=-1, keepdims=True)
        b = 2.0 * ((dn @ T[:3, :3].T) @ p)
        c = p @ p - SPHERE_R ** 2
        disc = b * b - 4 * c
        with np.errstate(invalid="ignore"):
            t = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0.0))) / 2.0, np.nan)
        depth = np.where(np.isfinite(t) & (t > 0), t * dn[..., 2], 0.0)
        cams.append(CameraDepth(str(pos), depth, K, T))
    return cams


def _bounds():
    return (np.full(3, -0.5), np.full(3, 0.5))


# --------------------------------------------------------------------------- 격자와 좌표


def test_grid_covers_the_requested_bounds():
    g = VoxelGrid.from_bounds([-0.5, -0.5, 0.0], [0.5, 0.5, 1.0], 0.01)
    assert g.shape == (100, 100, 100)
    # 원점은 첫 복셀의 **중심**이므로 하한보다 반 칸 안쪽이다.
    assert np.allclose(g.origin, [-0.495, -0.495, 0.005])
    idx, inside = g.to_index(np.array([[0.0, 0.0, 0.5], [9.0, 0.0, 0.0]]))
    assert inside.tolist() == [True, False]


def test_grid_rejects_a_degenerate_box():
    with pytest.raises(ValueError):
        VoxelGrid.from_bounds([0, 0, 0], [0, 1, 1], 0.01)


# --------------------------------------------------------------------------- 정확도


@pytest.mark.parametrize("voxel", [0.010, 0.020])
def test_distance_matches_the_analytic_sphere_within_half_a_voxel(voxel):
    """이 backend 의 핵심 주장. 오차 한계가 복셀 반 칸이고, **부호가 보수적**이다."""
    field = build_once(_sphere_cameras(), EsdfConfig(voxel_size=voxel, max_distance=0.4),
                       bounds=_bounds())
    # 카메라 축 방향의 질의점만 쓴다. 그 방향의 구 표면은 세 카메라 중 하나가 정면으로 보므로
    # **관측된** 기하이고, 그래야 이산화 오차만 남는다. 관측되지 않은 방향은 아래 별도 검사.
    q = np.array([[0, 0, 0.35], [0.40, 0, 0], [0, 0.35, 0], [0, 0, 0.25]])
    truth = np.linalg.norm(q, axis=1) - SPHERE_R
    error = field.distance(q) - truth
    assert np.all(np.abs(error) <= voxel / 2 + 1e-9), error
    # 부호가 보수적이다. 표면을 복셀 **중심**에 찍으므로 거리가 참값보다 작게 나오고, 제약이
    # `d - r - margin >= 0` 이므로 작게 나오는 것은 여유를 더 요구하는 쪽이다.
    assert np.all(error <= 1e-9), error


def test_unobserved_surface_makes_the_field_optimistic_not_conservative():
    """`unknown_policy` 의 대가를 숫자로 못박는다. 이 성질을 모르면 ESDF 를 과신하게 된다.

    세 카메라가 +x, +y, +z 를 보므로 그 사이 대각선 방향의 구 표면은 어느 카메라에도 정면으로
    잡히지 않는다. 그 표면이 `UNKNOWN` 이면 점유가 아니고, 거리장은 **더 먼** 점유 복셀까지를
    재므로 참값보다 **크게** 나온다 — 이산화 편향과 반대 방향이고, 여유를 덜 요구하는 쪽이다.

    이산화 편향은 복셀을 줄이면 사라지지만 이쪽은 그렇지 않다. 줄일 수 있는 것은 관측뿐이다.
    """
    field = build_once(_sphere_cameras(), EsdfConfig(voxel_size=0.010, max_distance=0.4),
                       bounds=_bounds())
    diagonal = np.array([[0.2, 0.2, 0.2]])
    truth = float(np.linalg.norm(diagonal[0]) - SPHERE_R)
    got = float(field.distance(diagonal)[0])
    assert got > truth, "대각선 방향은 관측이 얇아 낙관적으로 나오는 것이 이 설정의 성질이다"
    assert field.unknown_fraction > 0.0


def test_gradient_is_unit_length_in_free_space():
    """거리장은 아이코널 방정식을 만족하므로 자유 공간에서 |grad| = 1 이다.

    이것이 깨지면 TO 의 선형화가 잘못된 크기의 스텝을 내고, 그건 값이 맞아도 수렴을 망친다.
    """
    field = build_once(_sphere_cameras(), EsdfConfig(voxel_size=0.010, max_distance=0.4),
                       bounds=_bounds())
    q = np.array([[0, 0, 0.35], [0.35, 0, 0], [0.25, 0.25, 0]])
    g = field.gradient(q)
    assert np.allclose(np.linalg.norm(g, axis=1), 1.0, atol=0.05)
    # 방향은 표면에서 멀어지는 쪽, 즉 구 중심의 반대다.
    outward = q / np.linalg.norm(q, axis=1, keepdims=True)
    assert np.all(np.einsum("ij,ij->i", g, outward) > 0.9)


def test_outside_the_grid_returns_the_configured_default():
    field = build_once(_sphere_cameras(), EsdfConfig(voxel_size=0.02, max_distance=0.4),
                       bounds=_bounds())
    far = np.array([[9.0, 9.0, 9.0]])
    assert field.distance(far)[0] == pytest.approx(field.outside_distance)
    assert np.allclose(field.gradient(far), 0.0)


# --------------------------------------------------------------------------- 세 상태 점유


def test_unobserved_space_is_unknown_not_free():
    """이 backend 에서 가장 중요한 구분. 자유와 미관측을 섞으면 못 본 곳을 봤다고 말하게 된다."""
    cams = _sphere_cameras(positions=((1.0, 0, 0),))  # 한 방향만
    g = VoxelGrid.from_bounds(*_bounds(), 0.02)
    vol = TsdfVolume(g, truncation=0.06)
    vol.integrate(cams[0].depth, cams[0].camera_intrinsics, cams[0].T_base_cam)
    occ = vol.occupancy()
    assert (occ == UNKNOWN).any(), "한 카메라로는 볼 수 없는 곳이 반드시 있다"
    assert (occ == OCCUPIED).any()
    assert (occ == FREE).any()
    # 구의 뒷면(-x)은 이 카메라가 볼 수 없으므로 점유로 표시되어서는 안 된다.
    back = np.array([[-SPHERE_R * 0.9, 0.0, 0.0]])
    idx, inside = g.to_index(back)
    assert inside[0] and occ[tuple(idx[0])] != OCCUPIED


def test_unknown_policy_changes_only_what_it_says_it_changes():
    occ = np.array([[[FREE, UNKNOWN, OCCUPIED]]], np.int8)
    assert occupied_mask(occ, "free").tolist() == [[[False, False, True]]]
    assert occupied_mask(occ, "occupied").tolist() == [[[False, True, True]]]


def test_unknown_fraction_is_reported():
    field = build_once(_sphere_cameras(), EsdfConfig(voxel_size=0.02, max_distance=0.4),
                       bounds=_bounds())
    assert 0.0 < field.unknown_fraction < 1.0
    assert field.stats["unknown_policy"] == "free"
    assert field.stats["n_occupied"] > 0


# --------------------------------------------------------------------------- 국소 갱신


def test_local_update_reproduces_the_global_transform_exactly():
    """국소 갱신은 최적화이지 근사가 아니다. 값이 달라지면 그건 버그다."""
    cams = _sphere_cameras()
    cfg = EsdfConfig(voxel_size=0.02, max_distance=0.3, incremental=True, block_voxels=8)
    local = EsdfBuilder(cfg, bounds=_bounds())
    glob = EsdfBuilder(dataclasses.replace(cfg, incremental=False), bounds=_bounds())
    for _ in range(3):
        a = local.update(cams)
        b = glob.update(cams)
    assert np.allclose(a.distance_grid, b.distance_grid, atol=1e-6)


def test_a_static_scene_recomputes_nothing_after_the_first_frame():
    cams = _sphere_cameras()
    b = EsdfBuilder(EsdfConfig(voxel_size=0.02, max_distance=0.3, incremental=True),
                    bounds=_bounds())
    b.update(cams)
    second = b.update(cams)
    assert second.stats["n_occupancy_changed"] == 0
    assert second.stats["n_blocks"] == 0


def test_local_update_falls_back_to_global_when_blocks_would_cost_more():
    """되돌림이 있어야 국소 갱신이 최악의 경우에 더 느려지지 않는다."""
    cfg = EsdfConfig(voxel_size=0.02, max_distance=0.4, incremental=True, block_voxels=2)
    b = EsdfBuilder(cfg, bounds=_bounds())
    changed = np.arange(b.grid.n_voxels)  # 전부 바뀐 극단
    assert b._dirty_blocks(changed, cfg) is None


# --------------------------------------------------------------------------- target 제외


def test_carving_the_target_frees_its_voxels_only():
    cams = _sphere_cameras()
    g = VoxelGrid.from_bounds(*_bounds(), 0.02)
    vol = TsdfVolume(g, truncation=0.06)
    for c in cams:
        vol.integrate(c.depth, c.camera_intrinsics, c.T_base_cam)
    occ = vol.occupancy()
    before = int((occ == OCCUPIED).sum())
    # 구의 +z 쪽 표면 근처 점만 판다.
    pts = np.array([[0.0, 0.0, SPHERE_R], [0.0, 0.02, SPHERE_R]])
    n = carve(occ, g, pts)
    after = int((occ == OCCUPIED).sum())
    assert n > 0 and after == before - n
    assert (occ == OCCUPIED).any(), "파낸 것은 target 뿐이고 나머지 기하는 남아야 한다"


def test_exclude_target_is_visible_in_the_stats():
    cams = _sphere_cameras()
    pts = np.array([[0.0, 0.0, SPHERE_R]])
    field = build_once(cams, EsdfConfig(voxel_size=0.02, max_distance=0.3), bounds=_bounds(),
                       target_points=pts, exclude_target=True)
    assert field.stats["n_target_voxels_carved"] > 0


# --------------------------------------------------------------------------- 설정


def test_the_backend_defaults_to_primitive_so_nothing_changes_silently():
    assert AG3SConfig().collision_backend == "primitive"
    assert AG3SConfig().esdf.voxel_size == pytest.approx(0.010)
    assert AG3SConfig().esdf.unknown_policy == "free"


@pytest.mark.parametrize("bad", [
    {"collision_backend": "nope"},
    {"esdf": {"voxel_size": 0.0}},
    {"esdf": {"unknown_policy": "maybe"}},
    {"esdf": {"exclude_target": "sometimes"}},
    {"esdf": {"block_voxels": 0}},
    {"esdf": {"bounds_lower": [0, 0, 0]}},  # upper 없이 lower 만
])
def test_malformed_configuration_is_rejected(bad):
    with pytest.raises(AG3SConfigError):
        AG3SConfig.from_dict(bad)


def test_voxel_size_is_an_ablation_axis():
    """5 / 10 / 20 mm 를 바꿔 끼울 수 있어야 한다는 요구가 설정에서 성립하는지."""
    for vs in (0.005, 0.010, 0.020):
        cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {"voxel_size": vs}})
        assert cfg.esdf.voxel_size == pytest.approx(vs)
        assert cfg.esdf.truncation == pytest.approx(3.0 * vs)
