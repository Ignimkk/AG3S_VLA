"""T38 B3 — the ESDF tiers stay on the GPU and answer at the query points, with the same bits.

What is pinned here:

* **Same definition, same bits.** `DeviceEsdfField` (grids on CUDA) and `EsdfField` (grids in numpy) give
  bit-identical `distance` · `gradient` · `label` · `is_label` on the same grid — including points on
  lattice planes and one ulp either side, the last voxel centre, points outside the grid, and a 2-voxel axis
  — on **both** arithmetic paths (`gpu`: float64 on the device; `host`: one `index_select` of the query
  cells, then the numpy code), forced through `config.ESDF_QUERY_HOST_ARITH_MAX_POINTS`.
* **The gradient is `np.gradient(edge_order=1)` at the voxel, rounded to float32** — the per-voxel stencil
  evaluated on the GPU equals the full-grid `np.gradient(...).astype(float32)` cache at every voxel.
* **The composite** (`CuroboEsdfField`: coarse + fine + target-free + labels + static shapes) answers
  `distance` · `gradient` · `target_free_distance` · `target_free_gradient` (+ tier) · `answer_tier` · `label`
  identically with device layers and numpy layers.
* **No full-grid D2H on the query path.** Queries never materialise the host grids; `distance_grid` /
  `label_grid` do, lazily, read-only.
* **Builder pieces:** sign correction · site unpacking · labels on the GPU equal the numpy code; the whole
  cuRobo `update` equals a pre-B3 reference builder (host copies + numpy) on a synthetic scene; depth is
  uploaded once and the two mappers' TSDFs equal two separate uploads.
* **Recorder:** a device field records the same npz bytes as the numpy field, and the grid is pulled on the
  writer thread (background) without touching the field.
"""

from __future__ import annotations

import threading
import types

import numpy as np
import pytest

DF = pytest.importorskip("benchmark.ag3s.fields.device_field")

from benchmark.ag3s.fields.curobo_builder import (  # noqa: E402
    CuroboFieldBuilder,
    _SeedIndex,
    _Tier,
    unpack_site_linear,
)
from benchmark.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays  # noqa: E402
from benchmark.ag3s.fields.esdf import EsdfField, StaticBox, StaticPlane, VoxelGrid  # noqa: E402


@pytest.fixture(params=["gpu", "host", "auto"])
def arith(request, monkeypatch):
    """Force the query arithmetic path: all on the GPU, all on the host, or the size rule."""
    from benchmark.ag3s import config
    if request.param == "gpu":
        monkeypatch.setattr(config, "ESDF_QUERY_HOST_ARITH_MAX_POINTS", 0)
    elif request.param == "host":
        monkeypatch.setattr(config, "ESDF_QUERY_HOST_ARITH_MAX_POINTS", 10**9)
    return request.param


def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch


def _bits(a):
    a = np.ascontiguousarray(a)
    return a.view(np.uint8).tobytes(), a.dtype, a.shape


def _same_bits(a, b):
    return _bits(np.asarray(a)) == _bits(np.asarray(b))


def _probe_points(grid: VoxelGrid, rng, n=4000):
    """Random points (incl. 20 % outside), lattice points ± 1 ulp, the first and last voxel centres."""
    ext = np.asarray(grid.shape) * grid.voxel_size
    p = grid.origin - 0.2 * ext + rng.random((n, 3)) * 1.4 * ext
    ii = rng.integers(0, np.asarray(grid.shape), (600, 3))
    lp = grid.origin + ii * grid.voxel_size
    last = grid.origin + (np.asarray(grid.shape) - 1) * grid.voxel_size
    half = grid.origin + (ii + 0.5) * grid.voxel_size
    return np.vstack([p, lp, np.nextafter(lp, np.inf), np.nextafter(lp, -np.inf), half,
                      grid.origin[None], last[None], np.nextafter(last, np.inf)[None]])


GRIDS = [((17, 23, 11), 0.02, (0.1, -0.3, 0.7)),
         ((40, 9, 31), 0.005, (0.55, -0.12, 0.79)),
         ((2, 3, 2), 0.01, (0.0, 0.0, 0.0))]


@pytest.mark.parametrize("shape,vs,origin", GRIDS)
def test_device_layer_answers_exactly_like_the_numpy_layer(shape, vs, origin, arith):
    torch = _cuda()
    rng = np.random.default_rng(sum(shape))
    d = (rng.standard_normal(shape) * 0.1).astype(np.float32)
    d[tuple(rng.integers(0, s) for s in shape)] = -0.0
    labels = rng.integers(-1, 4, shape).astype(np.int32)
    grid = VoxelGrid(origin=np.asarray(origin, float), shape=shape, voxel_size=vs)
    names = ("a", "b", "c", "d")
    ref = EsdfField(grid=grid, distance_grid=d.copy(), max_distance=float(np.abs(d).max()),
                    label_grid=labels.copy(), label_names=names, outside_distance=0.4)
    dev = DF.DeviceEsdfField(grid, torch.as_tensor(d, device="cuda"), labels=labels,
                             label_names=names, outside_distance=0.4)
    pts = _probe_points(grid, rng)
    assert _same_bits(ref.distance(pts), dev.distance(pts))
    assert _same_bits(ref.gradient(pts), dev.gradient(pts))
    assert _same_bits(ref.label(pts), dev.label(pts))
    assert _same_bits(ref.is_label(pts, "c"), dev.is_label(pts, "c"))
    assert (ref.queried_point_count, ref.outside_query_count) == \
        (dev.queried_point_count, dev.outside_query_count)
    assert ref.outside_query_fraction == dev.outside_query_fraction
    assert dev.max_distance == float(np.max(np.abs(d)))
    # queries never pulled the grids
    assert dev.n_host_materialisations == 0 and dev._host_values is None and dev._host_labels is None
    # empty input keeps the numpy shapes / dtypes
    e = np.zeros((0, 3))
    for f in ("distance", "gradient", "label"):
        a, b = getattr(ref, f)(e), getattr(dev, f)(e)
        assert a.shape == b.shape and a.dtype == b.dtype, f


def test_device_layer_static_shapes_follow_the_numpy_rule(arith):
    torch = _cuda()
    rng = np.random.default_rng(3)
    shape, vs = (20, 20, 20), 0.02
    grid = VoxelGrid(origin=np.zeros(3), shape=shape, voxel_size=vs)
    d = (rng.standard_normal(shape) * 0.05 + 0.1).astype(np.float32)
    shapes = (StaticBox(center=np.array([0.2, 0.2, 0.1]), half_extents=np.array([0.05, 0.3, 0.02])),
              StaticPlane(point=np.array([0.0, 0.0, 0.03]), normal=np.array([0.0, 0.0, 1.0])))
    ref = EsdfField(grid=grid, distance_grid=d.copy(), max_distance=0.4, static_shapes=shapes)
    dev = DF.DeviceEsdfField(grid, torch.as_tensor(d, device="cuda"))
    dev.static_shapes = shapes
    pts = _probe_points(grid, rng, 2000)
    assert _same_bits(ref.distance(pts), dev.distance(pts))
    assert _same_bits(ref.gradient(pts), dev.gradient(pts))


def test_gradient_is_np_gradient_at_the_voxel_rounded_to_float32():
    """The definition, voxel by voxel: `_corner_gradients` = `np.gradient(f64, vs, edge_order=1)` → float32."""
    torch = _cuda()
    rng = np.random.default_rng(11)
    for shape, vs in (((9, 5, 7), 0.02), ((2, 2, 3), 0.005), ((33, 4, 2), 0.013)):
        d = (rng.standard_normal(shape) * 0.2).astype(np.float32)
        grid = VoxelGrid(origin=np.zeros(3), shape=shape, voxel_size=vs)
        dev = DF.DeviceEsdfField(grid, torch.as_tensor(d, device="cuda"))
        gx, gy, gz = np.gradient(d.astype(np.float64), vs, edge_order=1)
        ref = np.stack([gx, gy, gz], axis=-1).astype(np.float32).reshape(-1, 3)
        idx = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
        got = dev._corner_gradients(torch.as_tensor(idx, device="cuda")[None])[0].cpu().numpy()
        assert got.dtype == np.float64
        assert _same_bits(got.astype(np.float32), ref)
        assert _same_bits(got, ref.astype(np.float64))  # float32 values carried exactly in float64
        host = dev._host_corner_gradients(idx[None])[0]  # the host-arithmetic path, same definition
        assert host.dtype == np.float32 and _same_bits(host, ref)


def test_a_one_voxel_axis_fails_like_np_gradient(arith):
    torch = _cuda()
    grid = VoxelGrid(origin=np.zeros(3), shape=(4, 1, 4), voxel_size=0.01)
    d = np.zeros((4, 1, 4), np.float32)
    ref = EsdfField(grid=grid, distance_grid=d, max_distance=0.1)
    dev = DF.DeviceEsdfField(grid, torch.as_tensor(d, device="cuda"))
    with pytest.raises(ValueError, match="too small"):
        ref.gradient(np.zeros((1, 3)))
    with pytest.raises(ValueError, match="too small"):
        dev.gradient(np.zeros((1, 3)))


def test_the_size_rule_picks_the_path(monkeypatch):
    torch = _cuda()
    from benchmark.ag3s import config
    grid = VoxelGrid(origin=np.zeros(3), shape=(8, 8, 8), voxel_size=0.01)
    dev = DF.DeviceEsdfField(grid, torch.zeros((8, 8, 8), device="cuda"))
    calls = []
    monkeypatch.setattr(dev, "_host_distance", lambda *a: calls.append("host") or np.zeros(len(a[0])))
    monkeypatch.setattr(dev, "_device_distance", lambda *a: calls.append("gpu") or np.zeros(len(a[0])))
    monkeypatch.setattr(config, "ESDF_QUERY_HOST_ARITH_MAX_POINTS", 5)
    dev.distance(np.zeros((4, 3)))
    dev.distance(np.zeros((5, 3)))
    assert calls == ["host", "gpu"]
    assert config.ESDF_QUERY_HOST_ARITH_MAX_POINTS == 5


def test_host_grids_are_lazy_read_only_and_equal():
    torch = _cuda()
    rng = np.random.default_rng(5)
    d = rng.standard_normal((6, 7, 8)).astype(np.float32)
    lab = rng.integers(-1, 3, (6, 7, 8)).astype(np.int32)
    grid = VoxelGrid(origin=np.zeros(3), shape=(6, 7, 8), voxel_size=0.02)
    dev = DF.DeviceEsdfField(grid, torch.as_tensor(d, device="cuda"), labels=lab, label_names=("x",))
    assert dev.n_host_materialisations == 0
    worker_copy = dev.host_distance_grid(cache=False)          # recorder path: no state change
    assert np.array_equal(worker_copy, d) and dev.n_host_materialisations == 0
    assert dev._host_values is None
    a = dev.distance_grid
    assert a.dtype == np.float32 and np.array_equal(a, d) and not a.flags.writeable
    assert dev.distance_grid is a and dev.n_host_materialisations == 1
    assert np.array_equal(dev.label_grid, lab) and not dev.label_grid.flags.writeable
    with pytest.raises(ValueError):
        a[0, 0, 0] = 1.0
    # setting labels from host uploads them (the GPU copy is the truth)
    dev.label_grid = np.zeros((6, 7, 8), np.int32)
    assert dev.has_labels and int(dev.labels_device.abs().sum()) == 0
    dev.label_grid = None
    assert not dev.has_labels and dev.label_grid is None
    with pytest.raises(ValueError):
        dev.label_grid = np.zeros((2, 2, 2), np.int32)
    assert "DeviceEsdfField" in repr(dev)
    with pytest.raises(TypeError):
        DF.DeviceEsdfField(grid, torch.as_tensor(d.astype(np.float64), device="cuda"))
    with pytest.raises(ValueError):
        DF.DeviceEsdfField(grid, torch.zeros((2, 2, 2), device="cuda"))


def test_max_distance_matches_layer_from_arrays():
    torch = _cuda()
    d = np.array([[[0.1, -0.7], [np.inf, np.nan]], [[-np.inf, 0.3], [0.2, -0.0]]], np.float32)
    ref = layer_from_arrays(d, np.zeros(3), 0.01)
    dev = DF.layer_from_device(torch.as_tensor(d, device="cuda"), np.zeros(3), 0.01)
    assert dev.max_distance == ref.max_distance == float(np.float32(0.7))
    none = np.full((2, 2, 2), np.inf, np.float32)
    assert DF.layer_from_device(torch.as_tensor(none, device="cuda"), np.zeros(3), 0.01).max_distance \
        == layer_from_arrays(none, np.zeros(3), 0.01).max_distance == 0.0


# ----------------------------------------------------------------------------- the composite
def _composite(make, rng):
    def layer(lower, upper, vs, salt, labels=False):
        g = VoxelGrid.from_bounds(np.asarray(lower, float), np.asarray(upper, float), vs)
        c = g.centres()
        vals = (np.linalg.norm(c - np.array([0.5, 0.0, 0.1]), axis=-1) - 0.05 + 0.002 * salt
                + 0.004 * rng.standard_normal(g.shape)).astype(np.float32)
        lab = rng.integers(-1, 3, g.shape).astype(np.int32) if labels else None
        return make(vals, g.origin, vs, lab)

    coarse = layer((0.0, -0.4, -0.1), (1.0, 0.4, 0.5), 0.02, 0, True)
    fine = layer((0.40, -0.10, 0.0), (0.62, 0.10, 0.20), 0.005, 1, True)
    free = layer((0.40, -0.10, 0.0), (0.62, 0.10, 0.20), 0.005, 9)
    f = CuroboEsdfField((coarse, fine), outside_distance=0.5, target_free_layers=(free,),
                        target_label="apple")
    f.static_shapes = (StaticPlane(point=np.array([0.0, 0.0, 0.02]), normal=np.array([0.0, 0.0, 1.0])),)
    return f


def test_composite_field_with_device_layers_is_bit_identical(arith):
    torch = _cuda()

    def host(vals, origin, vs, lab):
        layer = layer_from_arrays(vals, origin, vs)
        layer.label_grid = lab
        layer.label_names = ("apple", "table", "crate") if lab is not None else ()
        return layer

    def device(vals, origin, vs, lab):
        return DF.layer_from_device(torch.as_tensor(vals, device="cuda"), origin, vs, labels=lab,
                                    label_names=("apple", "table", "crate") if lab is not None else ())

    a = _composite(host, np.random.default_rng(7))
    b = _composite(device, np.random.default_rng(7))
    rng = np.random.default_rng(8)
    pts = np.vstack([rng.uniform([-0.1, -0.5, -0.2], [1.1, 0.5, 0.6], (3000, 3)),
                     rng.uniform([0.38, -0.12, -0.02], [0.64, 0.12, 0.22], (3000, 3))])
    flags = rng.random(len(pts)) < 0.5
    assert _same_bits(a.distance(pts), b.distance(pts))
    assert _same_bits(a.gradient(pts), b.gradient(pts))
    assert _same_bits(a.target_free_distance(pts), b.target_free_distance(pts))
    ga, ta = a.target_free_gradient(pts, return_tier=True)
    gb, tb = b.target_free_gradient(pts, return_tier=True)
    assert _same_bits(ga, gb) and np.array_equal(ta, tb)
    assert len(set(ta.tolist())) >= 4                      # the fixture exercises the branches
    for tf in (False, True, flags):
        assert np.array_equal(a.answer_tier(pts, target_free=tf), b.answer_tier(pts, target_free=tf))
    assert _same_bits(a.label(pts), b.label(pts))
    assert _same_bits(a.is_label(pts, "table"), b.is_label(pts, "table"))
    assert a.has_labels and b.has_labels and a.label_names == b.label_names
    assert a.outside_query_fraction == b.outside_query_fraction
    assert all(l.n_host_materialisations == 0 for l in b.layers + b.target_free_layers)


# ----------------------------------------------------------------------------- builder pieces
def test_force_sign_on_the_gpu_equals_numpy_at_the_threshold():
    torch = _cuda()
    vs, k = 0.005, 1.5
    thr = k * vs
    t32 = np.float32(thr)
    edge = [t32, np.nextafter(t32, np.float32(1)), np.nextafter(t32, np.float32(0)),
            -t32, -np.nextafter(t32, np.float32(1)), -np.nextafter(t32, np.float32(0)),
            np.float32(-0.02), np.float32(0.0), np.float32(-0.0), np.float32(0.03)]
    rng = np.random.default_rng(2)
    vals = rng.normal(0, 0.01, (12, 12, 12)).astype(np.float32)
    idx = np.unique(rng.integers(0, 12, (300, 3)), axis=0)
    for n, v in enumerate(edge):
        vals[tuple(idx[n])] = v
    host = vals.copy()
    dev = torch.as_tensor(vals, device="cuda")
    s_host = CuroboFieldBuilder._force_sign_on_pure_voxels(host, idx, vs, threshold_voxels=k)
    s_dev = CuroboFieldBuilder._force_sign_on_pure_voxels(dev, idx, vs, threshold_voxels=k)
    assert s_host == s_dev
    assert _same_bits(host, dev.cpu().numpy())
    assert CuroboFieldBuilder._force_sign_on_pure_voxels(dev, None, vs) == \
        CuroboFieldBuilder._force_sign_on_pure_voxels(host, None, vs)


def test_unpack_site_linear_on_the_gpu_equals_numpy():
    torch = _cuda()
    rng = np.random.default_rng(4)
    shape = (16, 12, 20)
    x, y, z = (rng.integers(0, s, (5000,)) for s in shape)
    packed = ((z << 20) | (y << 10) | x).astype(np.int32)
    packed[rng.random(5000) < 0.3] = -1
    packed = packed.reshape(10, 25, 20)
    got = unpack_site_linear(torch.as_tensor(packed, device="cuda"), shape)
    assert got.dtype == torch.int64
    assert np.array_equal(got.cpu().numpy(), unpack_site_linear(packed, shape))


def _synthetic_tier(rng, shape, vs, as_torch):
    n = int(np.prod(shape))
    sites = rng.choice(n, 900, replace=False)
    site_lin = np.where(rng.random(n) < 0.6, sites[rng.integers(0, 900, n)], -1).astype(np.int64)
    values = rng.standard_normal(shape).astype(np.float32)
    origin = np.array([0.3, -0.2, 0.7])
    if as_torch:
        import torch
        return _Tier("t", torch.as_tensor(values, device="cuda"),
                     torch.as_tensor(site_lin.reshape(shape), device="cuda"), origin, vs)
    return _Tier("t", values, site_lin.reshape(shape), origin, vs)


def test_seed_box_prefilter_answers_like_querying_every_site():
    """`_SeedIndex.snap` skips sites outside the seed AABB ± snap·(1+1e-6); hit/owner must not change —
    including sites exactly `snap` from a box face (on an axis) and one ulp either side. CPU only."""
    rng = np.random.default_rng(12)
    seeds = {"a": rng.normal([0.5, 0.0, 0.85], 0.02, (120, 3)), "b": rng.normal([0.6, 0.1, 0.9], 0.01, (60, 3))}
    names = tuple(seeds)
    index = _SeedIndex.build(seeds, names)
    snap = 2.0 * 0.005
    pts = np.concatenate([seeds[n] for n in names])
    lo, hi = pts.min(0), pts.max(0)
    # sites exactly on the axis through the extreme seeds, `snap` away from them, ± 1 ulp
    ext = []
    for a in range(3):
        for sign, src in ((-1, pts[np.argmin(pts[:, a])]), (1, pts[np.argmax(pts[:, a])])):
            c = src.copy()
            c[a] = src[a] + sign * snap
            ext += [c, np.nextafter(c, c + sign), np.nextafter(c, c - sign)]
    centres = np.vstack([rng.uniform(lo - 0.3, hi + 0.3, (20000, 3)),
                         rng.uniform(lo - 2 * snap, hi + 2 * snap, (20000, 3)), np.array(ext)])
    h0, o0 = index.snap(centres, snap, box_filter=False)
    h1, o1 = index.snap(centres, snap)
    assert np.array_equal(h0, h1) and np.array_equal(o0, o1) and o1.dtype == np.int32
    assert h0.sum() > 100 and h0[-len(ext):].any() and not h0[-len(ext):].all()
    far = np.full((5, 3), 10.0)
    assert not index.snap(far, snap)[0].any() and index.snap(far, snap)[1].shape == (0,)


def test_labels_from_sites_on_the_gpu_equal_the_numpy_labels():
    _cuda()
    shape, vs = (30, 22, 26), 0.02
    origin = np.array([0.3, -0.2, 0.7])
    rng = np.random.default_rng(6)
    centres = origin + rng.integers(0, 22, (400, 3)) * vs
    seeds = {"apple": centres[:150] + rng.normal(0, 0.01, (150, 3)),
             "table": centres[150:] + rng.normal(0, 0.02, (250, 3))}
    names = tuple(seeds)
    host = _synthetic_tier(np.random.default_rng(1), shape, vs, False)
    dev = _synthetic_tier(np.random.default_rng(1), shape, vs, True)
    grid = VoxelGrid(origin=origin, shape=shape, voxel_size=vs)
    ref = CuroboFieldBuilder._labels_from_sites(host, grid, seeds, names)
    index = _SeedIndex.build(seeds, names)
    # the reference with every site queried (the pre-B3 KD-tree call)
    unfiltered = _SeedIndex.build(seeds, names)
    unfiltered.snap = lambda c, r: _SeedIndex.snap(unfiltered, c, r, box_filter=False)
    assert np.array_equal(CuroboFieldBuilder._labels_from_sites(host, grid, seeds, names,
                                                                 seed_index=unfiltered), ref)
    got = CuroboFieldBuilder._labels_from_sites(dev, grid, seeds, names, seed_index=index)
    again = CuroboFieldBuilder._labels_from_sites(dev, grid, seeds, names, seed_index=index)
    assert ref.dtype == np.int32 and (ref >= 0).sum() > 100 and (ref == 1).sum() > 0
    assert np.array_equal(got.cpu().numpy(), ref) and np.array_equal(again.cpu().numpy(), ref)
    # nothing used / no seeds -> None, as before
    empty = _Tier("e", dev.values, dev.site_linear.clone().fill_(-1), origin, vs)
    assert CuroboFieldBuilder._labels_from_sites(empty, grid, seeds, names) is None
    assert CuroboFieldBuilder._labels_from_sites(dev, grid, {}, ()) is None


# ----------------------------------------------------------------------------- cuRobo end to end
def _curobo():
    torch = _cuda()
    pytest.importorskip("curobo")
    return torch


def _scene_cameras():
    from benchmark.ag3s.fields.esdf import CameraDepth
    T32 = pytest.importorskip("tests.ag3s.test_t32_fine_truncation_curobo")
    boxes = T32._boxes(0.016) + [(np.array([0.48, -0.06, 0.82]), np.array([0.54, 0.0, 0.88]))]
    cams = []
    for i, T in enumerate(T32._views(True)):
        cams.append(CameraDepth(name=f"c{i}", depth=T32._raycast(T, boxes).astype(np.float32),
                                camera_intrinsics=T32.K, T_base_cam=T))
    return cams


def _config(**over):
    from benchmark.ag3s.config import AG3SConfig
    esdf = {"voxel_size": 0.020, "max_distance": 0.4, "exclude_support_surfaces": False,
            "backend": "curobo", "fine_voxel_size": 0.005, "tsdf_voxel_size": 0.005,
            "bounds_lower": [0.2, -0.6, 0.6], "bounds_upper": [1.0, 0.6, 1.2], **over}
    return AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": esdf}).esdf


def _pre_b3_update(builder, cameras, **kw):
    """Run `update` but convert every tier to host numpy right after it is built (the pre-B3 data path)
    and build the layers with `layer_from_arrays` + numpy labels."""
    import benchmark.ag3s.fields.curobo_builder as CB

    real_build = CB.CuroboFieldBuilder._build_tiers
    real_force = CB.CuroboFieldBuilder._force_sign_on_pure_voxels

    def build(self, *a, **k):
        # numpy sign correction on host copies, exactly as before B3
        def force(values, att_idx, vs, *, threshold_voxels=1.5):
            host = values.detach().cpu().numpy().copy()
            out = real_force(host, att_idx, vs, threshold_voxels=threshold_voxels)
            values.copy_(values.new_tensor(host))
            return out
        CB.CuroboFieldBuilder._force_sign_on_pure_voxels = staticmethod(force)
        try:
            tiers = real_build(self, *a, **k)
        finally:
            CB.CuroboFieldBuilder._force_sign_on_pure_voxels = staticmethod(real_force)
        for t in tiers:
            t.values = t.values.detach().cpu().numpy().copy()
            t.site_linear = t.site_linear.detach().cpu().numpy().copy()
        return tiers

    real_lfd = CB.layer_from_device

    def lfa(values, origin, vs, *, labels=None, label_names=()):
        layer = layer_from_arrays(values, origin, vs)
        layer.label_grid = labels
        layer.label_names = label_names
        return layer

    CB.CuroboFieldBuilder._build_tiers = build
    CB.layer_from_device = lfa
    try:
        return builder.update(cameras, **kw)
    finally:
        CB.CuroboFieldBuilder._build_tiers = real_build
        CB.layer_from_device = real_lfd


def test_curobo_update_equals_the_pre_b3_host_path():
    _curobo()
    from benchmark.ag3s.fields.curobo_builder import FineWindowRequest, TargetBall

    cams = _scene_cameras()
    apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (300, 3))
    table = np.random.default_rng(1).uniform([0.4, -0.3, 0.818], [0.9, 0.3, 0.822], (500, 3))
    held = np.random.default_rng(2).normal([0.51, -0.03, 0.86], 0.008, (120, 3))
    kw = dict(target_points=apple, target_free_points=apple,
              target_free_ball=TargetBall(centre=[0.51, -0.03, 0.85], radius=0.04, z_min=0.825),
              attached_points=held, labelled_points={"apple": apple, "table": table},
              fine_window=FineWindowRequest(manipulated_points=apple, placement="target_centroid"))
    fields = {}
    for name in ("new", "old"):
        builder = CuroboFieldBuilder(_config(attached_sign_threshold_voxels=1.5,
                                             fine_truncation_m=0.03))
        for _ in range(2):   # two frames: accumulated TSDF
            f = (builder.update(cams, **kw) if name == "new" else _pre_b3_update(builder, cams, **kw))
        fields[name] = f
    new, old = fields["new"], fields["old"]
    assert all(DF.is_device_layer(l) for l in new.layers + new.target_free_layers)
    assert not any(DF.is_device_layer(l) for l in old.layers + old.target_free_layers)
    assert len(new.layers) == 2 and len(new.target_free_layers) == 1
    for a, b in zip(new.layers + new.target_free_layers, old.layers + old.target_free_layers):
        assert a.n_host_materialisations == 0
        assert _same_bits(a.distance_grid, b.distance_grid)
        assert (a.label_grid is None) == (b.label_grid is None)
        if b.label_grid is not None:
            assert _same_bits(a.label_grid, b.label_grid)
        assert a.max_distance == b.max_distance
        assert _same_bits(a.grid.origin, b.grid.origin) and a.grid.shape == b.grid.shape \
            and a.grid.voxel_size == b.grid.voxel_size
    assert old.layers[0].label_grid is not None and (old.layers[0].label_grid >= 0).any()
    sc_new = {k: v for k, v in new.stats.items() if k not in ("target_free", "per_camera")}
    sc_old = {k: v for k, v in old.stats.items() if k not in ("target_free", "per_camera")}
    assert sc_new == sc_old
    assert new.stats["attached_sign_correction"][0]["n_attached_voxels"] > 0
    rng = np.random.default_rng(3)
    pts = rng.uniform([0.35, -0.2, 0.75], [0.7, 0.15, 0.95], (5000, 3))
    for m in ("distance", "gradient", "target_free_distance", "label"):
        assert _same_bits(getattr(new, m)(pts), getattr(old, m)(pts)), m
    assert np.array_equal(new.answer_tier(pts, target_free=True), old.answer_tier(pts, target_free=True))


def test_depth_is_uploaded_once_and_the_two_tsdfs_match_separate_uploads():
    torch = _curobo()
    cams = _scene_cameras()
    builder = CuroboFieldBuilder(_config(fine_truncation_m=0.03))
    assert builder.separate_fine_tsdf
    builder._ensure_mapper(cams)
    prepared = builder._observations(cams, "cuda:0")
    before = [o.depth_image.clone() for _, _, o in prepared]
    seen = []
    for mapper in (builder._mapper, builder._fine_mapper):
        real = mapper.integrate
        mapper.integrate = (lambda real: lambda obs: (seen.append(id(obs)), real(obs)))(real)
    per_cam = builder._integrate(builder._mapper, prepared)
    builder._integrate(builder._fine_mapper, prepared, contribution=False)
    assert len(seen) == 2 * len(cams) and seen[:len(cams)] == seen[len(cams):]   # same objects
    for b, (_, _, o) in zip(before, prepared):             # cuRobo did not write into them
        assert torch.equal(b, o.depth_image)
    assert sorted(per_cam) == sorted(c.name for c in cams)

    # reference: separate uploads per mapper (the pre-B3 `_integrate`)
    ref = CuroboFieldBuilder(_config(fine_truncation_m=0.03))
    ref._ensure_mapper(cams)
    for mapper in (ref._mapper, ref._fine_mapper):
        ref._integrate(mapper, ref._observations(cams, "cuda:0"))

    def tsdf(m):
        data = m.integrator.tsdf.data
        n = int(data.num_allocated.item())
        c = data.block_coords.reshape(-1, 3)[:n].to(torch.int64)   # stored flat (see `_free_tsdf_points`)
        order = torch.argsort(c[:, 0] * 1_000_003 ** 2 + c[:, 1] * 1_000_003 + c[:, 2])
        return c[order].cpu().numpy(), data.block_data[:n][order].float().cpu().numpy()

    for m_new, m_ref in ((builder._mapper, ref._mapper), (builder._fine_mapper, ref._fine_mapper)):
        (ka, va), (kb, vb) = tsdf(m_new), tsdf(m_ref)
        assert len(ka) > 0 and np.array_equal(ka, kb) and _same_bits(va, vb)


def test_update_does_not_pull_the_grids_to_the_host():
    torch = _curobo()
    cams = _scene_cameras()
    builder = CuroboFieldBuilder(_config())
    apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (200, 3))
    big = []
    real_cpu = torch.Tensor.cpu

    def spy(self, *a, **k):
        if self.numel() >= 64 ** 3:
            big.append(tuple(self.shape))
        return real_cpu(self, *a, **k)

    torch.Tensor.cpu = spy
    try:
        f = builder.update(cams, target_points=apple, labelled_points={"apple": apple})
        f.distance(apple)
        f.gradient(apple)
        f.label(apple)
    finally:
        torch.Tensor.cpu = real_cpu
    assert big == []


# ----------------------------------------------------------------------------- recorder
def _record_inputs(field):
    return dict(t_step=1, chunk_index=2,
                constraint_set=types.SimpleNamespace(
                    esdf=field, target=None, candidates=[], support_surfaces=[], status=None,
                    validity=None, notes=[], metrics={}, profile={}, has_target=False),
                debug={}, clearance=np.zeros((2, 3), np.float32))


@pytest.mark.parametrize("background", [False, True])
def test_recorder_writes_device_grids_with_the_same_bytes(tmp_path, background):
    torch = _cuda()
    from benchmark.ag3s.experiments.sources import constraint_record as CR

    def host(vals, origin, vs, lab):
        layer = layer_from_arrays(vals, origin, vs)
        layer.label_grid = lab
        layer.label_names = ("apple", "table", "crate") if lab is not None else ()
        return layer

    def device(vals, origin, vs, lab):
        return DF.layer_from_device(torch.as_tensor(vals, device="cuda"), origin, vs, labels=lab,
                                    label_names=("apple", "table", "crate") if lab is not None else ())

    a = _composite(host, np.random.default_rng(7))
    b = _composite(device, np.random.default_rng(7))
    a.stats = b.stats = {"backend": "curobo"}
    threads = []
    real = CR._DeferredGrid.resolve

    def resolve(self):
        threads.append(threading.current_thread().name)
        return real(self)

    CR._DeferredGrid.resolve = resolve
    try:
        wa = CR.ConstraintRecordWriter(tmp_path / "a", background=background)
        wb = CR.ConstraintRecordWriter(tmp_path / "b", background=background)
        pa = wa.record(**_record_inputs(a))
        pb = wb.record(**_record_inputs(b))
        wa.close()
        wb.close()
    finally:
        CR._DeferredGrid.resolve = real
    with np.load(pa) as za, np.load(pb) as zb:
        assert sorted(za.files) == sorted(zb.files)
        assert {"esdf_distance", "esdf_coarse_distance", "esdf_free_distance"} <= set(za.files)
        for k in za.files:
            assert za[k].dtype == zb[k].dtype and za[k].tobytes() == zb[k].tobytes(), k
    assert len(threads) == 3
    if background:
        assert set(threads) == {"constraint-record-writer"}
    assert all(l.n_host_materialisations == 0 and l._host_values is None
               for l in b.layers + b.target_free_layers)
