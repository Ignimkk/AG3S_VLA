"""T32 H4-fix — the fine tier's TSDF truncation is its own (30 mm), the coarse tier's is unchanged (60 mm).

Needs cuRobo + CUDA (`.venv-openpi-live`); skipped elsewhere. Same synthetic wall as the verifier's
`outputs/verify/T32/h4/synth.py` (a 16 mm box wall on a table slab, analytic ray-cast depth, mm-quantised),
same line measurement (d<0 width along the wall normal, 0.5 mm samples of the trilinear field).

Measured before the fix (`T32.h4.verify.json`, one TSDF at 60 mm): normal_1side fine band 50 mm,
head_2side 70 mm, normal_2side 80 mm. With 30 mm: 20 / 22 / 44.5 mm (the last is the two-sided bound,
see its test). At 15 mm the fine band was 0 — the floor
(`EsdfConfig.fine_truncation_floor`, 4 TSDF voxels) keeps the fine truncation above that.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("cuRobo needs CUDA", allow_module_level=True)

from AG3S.ag3s.config import AG3SConfig  # noqa: E402
from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder, FineWindowRequest  # noqa: E402
from AG3S.ag3s.fields.esdf import CameraDepth  # noqa: E402

X0, Y0, ZL = 0.603, -0.0155, 0.884
K = np.array([[240.0, 0.0, 320.0], [0.0, 240.0, 240.0], [0.0, 0.0, 1.0]])
DIST = 0.755
N_FRAMES = 8


def _boxes(thickness):
    wall = (np.array([X0 - thickness / 2, Y0 - 0.150, 0.828]),
            np.array([X0 + thickness / 2, Y0 + 0.150, 0.940]))
    table = (np.array([0.30, -0.50, 0.780]), np.array([1.00, 0.50, 0.820]))
    return [wall, table]


def _raycast(Tbc, boxes, h=480, w=640):
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    dw = dc @ Tbc[:3, :3].T
    o = Tbc[:3, 3]
    best = np.full(len(dw), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for lo, hi in boxes:
            t1, t2 = (lo - o) / dw, (hi - o) / dw
            tmin = np.nanmax(np.minimum(t1, t2), axis=1)
            tmax = np.nanmin(np.maximum(t1, t2), axis=1)
            hit = (tmax >= tmin) & (tmax > 0)
            t = np.where(tmin > 0, tmin, tmax)
            best = np.where(hit & (t < best), t, best)
    depth = np.where(np.isfinite(best), best * (dw @ Tbc[:3, 2]), 0.0).reshape(h, w)
    return np.clip(np.rint(depth * 1000.0), 0, 65535) / 1000.0


def _look_at(pos, target, up=(0.0, 0.0, 1.0)):
    z = np.asarray(target, float) - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = pos
    return T


def _views(two_sided):
    wc = np.array([X0, Y0, ZL])
    Ts = [_look_at(np.array([X0 - DIST, Y0, ZL]), wc)]
    if two_sided:
        Ts.append(_look_at(np.array([X0 + DIST, Y0, ZL]), wc))
    return Ts


def _build(fine_truncation_m="default", *, thickness=0.016, two_sided=False):
    esdf = {"voxel_size": 0.020, "max_distance": 0.4, "exclude_support_surfaces": False,
            "backend": "curobo", "fine_voxel_size": 0.005, "tsdf_voxel_size": 0.005,
            "bounds_lower": [0.0, -0.8, 0.5], "bounds_upper": [1.4, 0.8, 1.6]}
    if fine_truncation_m != "default":
        esdf["fine_truncation_m"] = fine_truncation_m
    cfg = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": esdf}).esdf
    builder = CuroboFieldBuilder(cfg)
    boxes = _boxes(thickness)
    cams = [CameraDepth(name=f"c{i}", depth=_raycast(T, boxes), camera_intrinsics=K, T_base_cam=T)
            for i, T in enumerate(_views(two_sided))]
    window = FineWindowRequest(manipulated_points=np.array([[X0, Y0, ZL]]),
                               placement="target_centroid")
    field = None
    for _ in range(N_FRAMES):
        field = builder.update(cams, fine_window=window)
    return builder, field


def _band_mm(layer):
    s = np.arange(-0.10, 0.10, 0.0005)
    pts = np.column_stack([X0 + s, np.full_like(s, Y0), np.full_like(s, ZL)])
    v = np.asarray(layer.distance(pts), np.float64)
    return float(0.5 * (v < 0).sum()), float(1000 * v.min())


def _layers(field):
    by_vs = sorted(field.layers, key=lambda layer: layer.grid.voxel_size)
    return by_vs[0], by_vs[-1]          # fine, coarse


def test_one_sided_16mm_wall_has_a_band_close_to_its_thickness_and_never_zero():
    builder, field = _build()
    assert builder.separate_fine_tsdf
    assert field.stats["tsdf_truncation"]["fine_m"] == pytest.approx(0.030)
    fine, _ = _layers(field)
    band, vmin = _band_mm(fine)
    assert 8.0 <= band <= 26.0, (band, vmin)       # measured (h4 synth, normal_1side): 20 mm


def test_two_sided_16mm_wall_band_is_bounded_by_the_fine_truncation():
    """Seen square-on from **both** sides, the voxel just outside face B is 'occluded, behind face A'
    for camera A (negative) and 'free' for camera B (small positive); the weighted mean is negative
    until A's ray passes the truncation — so the band is thickness + 2 × (truncation − thickness)
    at most. 30 mm: ≤ 44 mm (measured 44.5); the 60 mm TSDF gave 80 mm (h4 synth normal_2side). The
    oblique head-camera pair measured 22 mm at 30 mm (h4 synth head_2side)."""
    _, field = _build(two_sided=True)
    band, _ = _band_mm(_layers(field)[0])
    _, old = _build(None, two_sided=True)
    band_old, _ = _band_mm(_layers(old)[0])
    assert 16.0 <= band <= 16.0 + 2 * (30.0 - 16.0) + 5.0, band
    assert band < band_old - 20.0, (band, band_old)


def test_the_old_single_tsdf_is_what_made_the_wide_band_and_the_coarse_tier_is_unchanged():
    _, new = _build()
    old_builder, old = _build(None)
    assert not old_builder.separate_fine_tsdf
    band_old, _ = _band_mm(_layers(old)[0])
    band_new, _ = _band_mm(_layers(new)[0])
    assert band_old >= 40.0 and band_new < band_old  # 60 mm TSDF: 50 mm band (h4 synth)
    c_new, c_old = _layers(new)[1], _layers(old)[1]
    assert np.array_equal(np.asarray(c_new.distance_grid), np.asarray(c_old.distance_grid))


def test_the_floor_keeps_a_too_small_truncation_from_erasing_the_wall():
    builder, field = _build(0.010)                   # below 4 × 5 mm → clamped to 20 mm
    assert builder.fine_truncation == pytest.approx(0.020)
    band, _ = _band_mm(_layers(field)[0])
    assert band > 0.0, band


# ------------------------------------------------------------------ T32 H3 — held traces → free


def _sphere_depth(Tbc, centre, r, boxes, h=480, w=640):
    """Depth of the boxes plus one sphere (ray cast), and the sphere's pixel mask."""
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    dc = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1).reshape(-1, 3)
    dw = dc @ Tbc[:3, :3].T
    o = Tbc[:3, 3]
    oc = o - centre
    b = 2.0 * (dw @ oc)
    a = (dw * dw).sum(1)
    disc = b * b - 4 * a * (oc @ oc - r * r)
    ts = np.where(disc >= 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    ts = np.where(ts > 0, ts, np.inf)
    base = _raycast(Tbc, boxes, h, w).reshape(-1)
    sphere = np.isfinite(ts) & ((base == 0) | (ts * (dw @ Tbc[:3, 2]) < base))
    depth = np.where(sphere, np.rint(ts * (dw @ Tbc[:3, 2]) * 1000.0) / 1000.0, base)
    return depth.reshape(h, w), sphere.reshape(h, w)


@pytest.mark.parametrize("free", [True, False])
def test_held_traces_are_set_free_in_the_tsdf_and_the_table_stays(free):
    """An apple integrated on the table, then masked (H1, it is held): without freeing its old
    surface stays an obstacle for ever (the control); with `held_free_points` its voxels are free
    and the table under it is untouched (the z cut)."""
    from AG3S.ag3s.robot_models.held_object import sphere_volume_points

    table = (np.array([0.30, -0.50, 0.780]), np.array([1.00, 0.50, 0.820]))
    apple = np.array([0.60, 0.0, 0.820 + 0.034])
    esdf = {"voxel_size": 0.020, "max_distance": 0.4, "exclude_support_surfaces": False,
            "backend": "curobo", "fine_voxel_size": 0.005, "tsdf_voxel_size": 0.005,
            "bounds_lower": [0.0, -0.8, 0.5], "bounds_upper": [1.4, 0.8, 1.6]}
    builder = CuroboFieldBuilder(AG3SConfig.from_dict({"collision_backend": "esdf",
                                                       "esdf": esdf}).esdf)
    T = _look_at(np.array([0.10, 0.0, 1.30]), apple)
    depth, sphere_px = _sphere_depth(T, apple, 0.034, [table])
    window = FineWindowRequest(manipulated_points=apple[None], placement="target_centroid")
    for _ in range(N_FRAMES):
        builder.update([CameraDepth("c", depth, K, T)], fine_window=window)
    held = sphere_volume_points([(apple, 0.039)], 0.0025, z_min=0.820 + 0.005)
    field = None
    for _ in range(3):
        field = builder.update([CameraDepth("c", depth, K, T, robot_mask=sphere_px)],
                               fine_window=window, held_free_points=held if free else None)
    fine = _layers(field)[0]
    probe_front = apple + (T[:3, 3] - apple) / np.linalg.norm(T[:3, 3] - apple) * 0.030
    d_front = float(fine.distance(probe_front[None])[0])
    d_table = float(fine.distance(np.array([[0.60, 0.08, 0.815]]))[0])
    if free:
        assert field.stats["held_free"]["n_voxels_freed"]["fine_tsdf"] > 100
        assert d_front > 0.0, d_front                   # the old apple surface is gone
    else:
        assert d_front < 0.005, d_front                 # the trace the control keeps
    assert d_table < 0.0, d_table                       # the table is never erased
