"""T38 B4 — scene reconstruction and the support RANSAC scoring on the GPU, with numpy's exact bits.

What is pinned here:

* **The BLAS fact the GPU path relies on is re-measured.** numpy's ``(N, 3) @ (3, m)`` (OpenBLAS dgemm, N ≥ 2,
  m ≥ 2) equals the k-ordered FMA chain ``fma(z, c, fma(y, b, x·a))`` the jiterator kernels compute — every
  width 2…32 and N from 2 to 170 k. If the BLAS under numpy changes, this fails loudly instead of the GPU
  path drifting silently.
* **Back-projection · voxel downsample · cap** on the device equal `reconstruction.backproject` /
  `downsample_and_cap` to the bit: points, pixels, order, stats — with invalid depth (0 · NaN · ±inf · out of
  range), `depth_scale`, `range_max`, both cap strategies (incl. `covered=False`), odd voxel sizes; fewer
  than two points → None (numpy path).
* **The representative is the first point of its voxel in input order** (`np.unique(return_index)`).
* **The depth robot mask on device points** equals the numpy-point mask (mask · protected · guard counts ·
  held count, tail columns and separate held spheres) and goes through the one seam `robot_sphere_mask`.
* **One camera front end ≡ the numpy branch** of `process_observation` (cloud · pixels · attention · stats ·
  pixel masks), the fused cloud of three cameras is unchanged, sphere FK runs once per `q` per frame, and the
  TSDF's device depth equals ``np.where(mask, 0, float32(depth))``; the cuRobo builder uses it without
  another upload.
* **Attention lift** reading only the cloud's pixels equals reading the full resampled map, every pixel.
* **RANSAC:** GPU hypothesis counts equal numpy's (including near-threshold points and the 1-wide trailing
  block numpy computes with gemv); fitted planes and inlier sets are bit-identical.
* **grow_region**'s C-level flattening returns what the generator did.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

DR = pytest.importorskip("AG3S.ag3s.stages.device_recon")

from AG3S.ag3s import config  # noqa: E402
from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, PointCloudConfig, SupportSurfaceConfig  # noqa: E402
from AG3S.ag3s.runtime import multiview as MV  # noqa: E402
from AG3S.ag3s.stages import attention_lifting as AL  # noqa: E402
from AG3S.ag3s.stages import robot_filter as RF  # noqa: E402
from AG3S.ag3s.stages import support_surface as SS  # noqa: E402
from AG3S.ag3s.stages.reconstruction import backproject, downsample_and_cap  # noqa: E402
from AG3S.ag3s.types import PointCloud  # noqa: E402
from tests.ag3s.fixtures import (  # noqa: E402
    DEFAULT_ARM_Q,
    FINGERTIP_LINK,
    PlanarTwoLinkArm,
    make_scene,
    make_three_camera_scene,
)


def _cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch


def _bits(a):
    a = np.ascontiguousarray(np.asarray(a))
    return a.dtype.str, a.shape, a.tobytes()


def _same(a, b):
    return _bits(a) == _bits(b)


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _depth_image(rng, hw=(120, 160), *, invalid=True, scale=1.0):
    h, w = hw
    d = rng.uniform(0.2, 2.8, size=hw)
    d[rng.random(hw) < 0.05] = rng.uniform(2.9, 4.0)          # beyond depth_max
    if invalid:
        d[rng.random(hw) < 0.03] = 0.0
        d[rng.random(hw) < 0.01] = np.nan
        d[rng.random(hw) < 0.01] = np.inf
        d[rng.random(hw) < 0.01] = -np.inf
        d[rng.random(hw) < 0.01] = 0.03                     # below depth_min
    return d / scale


def _pose(rng):
    T = np.eye(4)
    T[:3, :3] = _rotation(rng)
    T[:3, 3] = rng.normal(size=3)
    return T


K = np.array([[151.3, 0.0, 79.7], [0.0, 149.9, 60.2], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------------------- BLAS fact


@pytest.mark.parametrize("n", [2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 33, 63, 64, 65, 1000, 33_333, 170_001])
def test_numpy_dgemm_is_the_fma_chain_the_kernels_compute(n):
    torch = _cuda()
    rng = np.random.default_rng(n)
    P = rng.normal(size=(n, 3)) * np.array([0.5, 0.4, 1.2]) + np.array([0.1, -0.2, 1.0])
    Pt = torch.as_tensor(P, device="cuda")
    dot3 = DR.kernel("dot3")
    for m in range(2, 33):
        full = rng.normal(size=(m + 3, 3))
        blk = full[1:1 + m]                               # a row-slice view, like `normals[block]`
        ref = P @ blk.T
        Nt = torch.as_tensor(blk, device="cuda")
        got = dot3(Pt[:, 0:1], Pt[:, 1:2], Pt[:, 2:3], Nt[:, 0][None], Nt[:, 1][None], Nt[:, 2][None])
        assert _same(got.cpu().numpy(), ref), (n, m)
    T = _pose(rng)
    ref = P @ T[:3, :3].T + T[:3, 3]
    rot = DR.kernel("rot_row")
    got = torch.stack([rot(Pt[:, 0], Pt[:, 1], Pt[:, 2], a=float(T[j, 0]), b=float(T[j, 1]),
                           c=float(T[j, 2]), t=float(T[j, 3])) for j in range(3)], 1)
    assert _same(got.cpu().numpy(), ref)


# ---------------------------------------------------------------------- back-projection · voxels

CONFIGS = [
    {},
    {"depth_scale": 0.001},
    {"range_max": 1.7},
    {"voxel_size": 0.0123},
    {"voxel_size": 0.0},
    {"max_points": 1500},                                     # coverage cap, a few growth steps
    {"max_points": 50, "cap_max_iterations": 2},              # cap runs out → covered=False
    {"max_points": 1500, "cap_strategy": "stride"},
    {"voxel_size": 0.03, "depth_min": 0.5, "depth_max": 2.0},
]


@pytest.mark.parametrize("over", CONFIGS, ids=[str(c) for c in CONFIGS])
@pytest.mark.parametrize("seed", [0, 1])
def test_device_backproject_downsample_cap_equal_numpy(over, seed):
    torch = _cuda()
    rng = np.random.default_rng(seed)
    cfg = dataclasses.replace(PointCloudConfig(voxel_size=0.02), **over)
    depth = _depth_image(rng, scale=cfg.depth_scale)
    T = _pose(rng)
    ref = backproject(depth, K, T, cfg)
    ref_cloud, ref_stats = downsample_and_cap(ref, cfg)

    full = DR.backproject_device(DR.upload_depth(depth, torch.device("cuda")), K, T, cfg)
    assert _same(full.points.cpu().numpy(), ref.points)
    uv = np.stack([full.cols.cpu().numpy(), full.rows.cpu().numpy()], 1).astype(np.int32)
    assert _same(uv, ref.uv)
    assert _same(full.depth_m.cpu().numpy(), np.asarray(depth, np.float64) * cfg.depth_scale)
    idx, stats = DR.downsample_and_cap_device(full.points, cfg)
    i = idx.cpu().numpy()
    assert stats == ref_stats
    assert _same(ref.points[i], ref_cloud.points) and _same(ref.uv[i], ref_cloud.uv)


def test_backproject_device_hands_tiny_clouds_back_to_numpy():
    torch = _cuda()
    cfg = PointCloudConfig()
    for n_valid in (0, 1):
        depth = np.zeros((8, 9))
        depth.flat[:n_valid] = 1.0
        assert DR.backproject_device(DR.upload_depth(depth, torch.device("cuda")), K, np.eye(4), cfg) is None
    depth = np.zeros((8, 9))
    depth.flat[:2] = 1.0
    full = DR.backproject_device(DR.upload_depth(depth, torch.device("cuda")), K, np.eye(4), cfg)
    assert len(full) == 2 and _same(full.points.cpu().numpy(), backproject(depth, K, np.eye(4), cfg).points)


def test_the_representative_is_the_first_point_of_its_voxel_in_input_order():
    torch = _cuda()
    rng = np.random.default_rng(3)
    cells = rng.integers(-20, 20, size=(300, 3))
    pts = (cells[rng.integers(0, 300, 5000)] + rng.random((5000, 3))) * 0.01   # many points per voxel
    idx = DR.voxel_first_device(torch.as_tensor(pts, device="cuda"), 0.01).cpu().numpy()
    keys = np.floor(pts / 0.01).astype(np.int64)
    first = {}
    for i, k in enumerate(map(tuple, keys)):
        first.setdefault(k, i)
    assert idx.tolist() == sorted(first.values())


def test_voxel_overflow_raises_like_numpy():
    torch = _cuda()
    pts = np.array([[0.0, 0.0, 0.0], [1e9, 1e9, 1e9]])
    with pytest.raises(ValueError, match="overflows int64"):
        DR.voxel_first_device(torch.as_tensor(pts, device="cuda"), 1e-6)
    with pytest.raises(ValueError, match="overflows int64"):
        downsample_and_cap(PointCloud(pts), PointCloudConfig(voxel_size=1e-6))


def test_device_choice():
    for bad in ("gpu", "tpu"):
        with pytest.raises(AG3SConfigError):
            DR.torch_device(bad, what="RECON")
    assert DR.torch_device("cpu", what="RECON") is None
    assert config.RECON_DEVICE == "auto" and config.SUPPORT_RANSAC_DEVICE == "auto"


# ------------------------------------------------------------------------------- robot mask


def _mask_inputs():
    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    cfg = AG3SConfig.from_dict({"pointcloud": {"self_filter_inflation": 0.03,
                                               "self_filter_inflation_by_link": {FINGERTIP_LINK: 0.0},
                                               "self_filter_target_guard_radius": 0.08}}).pointcloud
    infl = RF.resolve_self_filter_inflation(arm.sphere_link_names, cfg)
    centers, radii = arm.sphere_centers_numeric(DEFAULT_ARM_Q)
    guard = centers[-1] + np.array([0.0, 0.0, radii[-1] + 0.01])
    return arm, scene, cfg, infl, guard


class _HeldArm:
    """`PlanarTwoLinkArm` + a held sphere, either appended to the model (tail columns) or not."""

    def __init__(self, arm, appended: bool):
        self.arm, self.appended = arm, appended
        self.nq = arm.nq
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.arm, name)

    def _held(self, q):
        c, r = self.arm.sphere_centers_numeric(q)
        return np.asarray(c)[-1:] + np.array([[0.0, 0.0, 0.05]]), np.array([0.04])

    def sphere_centers_numeric(self, q):
        self.calls += 1
        c, r = self.arm.sphere_centers_numeric(q)
        if self.appended:
            hc, hr = self._held(q)
            return np.vstack([c, hc]), np.concatenate([r, hr])
        return c, r

    def held_spheres(self, q):
        return self._held(q)


@pytest.mark.parametrize("variant", ["plain", "held-tail", "held-extra", "no-guard"])
def test_device_depth_mask_equals_the_numpy_point_mask(variant, monkeypatch):
    torch = _cuda()
    arm, scene, cfg, infl, guard = _mask_inputs()
    model = arm
    if variant.startswith("held"):
        model = _HeldArm(arm, appended=(variant == "held-tail"))
        if variant == "held-tail":
            infl = np.concatenate([infl, [0.0]])
    if variant == "no-guard":
        guard = None
    ref_full = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam, cfg)
    ref = RF.depth_robot_mask(ref_full, scene.depth.shape, model, DEFAULT_ARM_Q, cfg,
                              inflation=infl, guard_centre=guard, device="cpu")
    seen = []
    real = RF.robot_sphere_mask
    monkeypatch.setattr(RF, "robot_sphere_mask", lambda p, *a, **k: (seen.append(type(p).__module__), real(p, *a, **k))[1])
    full = DR.backproject_device(DR.upload_depth(scene.depth, torch.device("cuda")),
                                 scene.camera_intrinsics, scene.T_base_cam, cfg)
    got = RF.depth_robot_mask_device(full, model, DEFAULT_ARM_Q, cfg, inflation=infl, guard_centre=guard)
    assert seen and all(m.startswith("torch") for m in seen)        # through the one seam, on device
    assert got.protected is None
    assert _same(got.mask, ref.mask) and _same(got.protected_mask(), ref.protected)
    assert got.n_held_px == ref.n_held_px and got.guard == ref.guard
    assert ref.mask.any()
    if variant.startswith("held"):
        assert got.n_held_px is not None and got.n_held_px > 0
    if variant == "plain":
        assert got.n_guard_protected_px > 0
    d32 = np.where(ref.mask, np.float32(0.0), np.asarray(scene.depth * cfg.depth_scale, np.float32))
    assert _same(got.masked_depth_f32_device().cpu().numpy(), d32)


def test_fk_cache_runs_the_sphere_fk_once_per_q():
    arm, scene, cfg, infl, guard = _mask_inputs()
    model = _HeldArm(arm, appended=False)
    full = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam, cfg)
    cache = {}
    a = RF.depth_robot_mask(full, scene.depth.shape, model, DEFAULT_ARM_Q, cfg, inflation=infl,
                            guard_centre=guard, fk_cache=cache)
    b = RF.depth_robot_mask(full, scene.depth.shape, model, DEFAULT_ARM_Q.copy(), cfg, inflation=infl,
                            guard_centre=guard, fk_cache=cache)
    assert model.calls == 1
    RF.depth_robot_mask(full, scene.depth.shape, model, DEFAULT_ARM_Q + 1e-3, cfg, inflation=infl,
                        guard_centre=guard, fk_cache=cache)
    assert model.calls == 2
    c = RF.depth_robot_mask(full, scene.depth.shape, model, DEFAULT_ARM_Q, cfg, inflation=infl,
                            guard_centre=guard)
    for x in (a, b):
        assert _same(x.mask, c.mask) and x.n_held_px == c.n_held_px and x.guard == c.guard


# ------------------------------------------------------------------------------ camera front end


def _cfg3():
    return AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "esdf": {"voxel_size": 0.02, "fine_voxel_size": None},
        "pointcloud": {"self_filter_inflation": 0.02,
                       "self_filter_inflation_by_link": {FINGERTIP_LINK: 0.0},
                       "self_filter_target_guard_radius": 0.08}})


@pytest.mark.parametrize("max_points", [None, 3000])
def test_the_gpu_front_end_equals_the_numpy_branch(max_points, monkeypatch):
    _cuda()
    arm = PlanarTwoLinkArm()
    scene3 = make_three_camera_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    cfg = _cfg3()
    if max_points is not None:
        cfg = cfg.with_overrides({"pointcloud": {"max_points": max_points}})
    infl = RF.resolve_self_filter_inflation(arm.sphere_link_names, cfg.pointcloud)
    centers, _ = arm.sphere_centers_numeric(DEFAULT_ARM_Q)
    guard = centers[-1]
    out = {}
    for dev in ("cpu", "auto"):
        monkeypatch.setattr(config, "RECON_DEVICE", dev)
        out[dev] = MV.fuse_observations(scene3.observations, cfg, robot_model=arm,
                                        self_filter_inflation=infl, self_filter_guard_centre=guard,
                                        with_depth_mask=True)
    a, b = out["cpu"], out["auto"]
    for ra, rb in zip(a.per_camera, b.per_camera):
        assert ra.points_device is None and rb.points_device is not None
        assert rb.depth_mask.device is not None and ra.depth_mask.device is None
        assert _same(rb.cloud.points, ra.cloud.points) and _same(rb.cloud.uv, ra.cloud.uv)
        assert _same(rb.points_device.cpu().numpy(), ra.cloud.points)
        assert _same(rb.attention, ra.attention) and _same(rb.raw_attention, ra.raw_attention)
        assert rb.stats == ra.stats
        assert _same(rb.depth_mask.mask, ra.depth_mask.mask)
        assert rb.depth_mask.protected is None and ra.depth_mask.protected is not None
        assert _same(rb.depth_mask.protected_mask(), ra.depth_mask.protected)
        assert rb.depth_mask.n_held_px == ra.depth_mask.n_held_px and rb.depth_mask.guard == ra.depth_mask.guard
    for k in ("points", "obs_offset", "obs_camera", "obs_uv", "obs_time", "obs_attention", "representative"):
        assert _same(getattr(b.cloud, k), getattr(a.cloud, k)), k
    assert _same(b.attention, a.attention) and _same(b.raw_attention, a.raw_attention)
    assert b.metrics == a.metrics and b.validity == a.validity and b.notes == a.notes


def test_the_fused_esdf_path_feeds_the_tsdf_the_device_depth(monkeypatch):
    """`AG3S` hands `CameraDepth.device_depth` (masked float32, from the front end's upload) to the
    builder; its values are the host masked depth's."""
    _cuda()
    from AG3S.ag3s.runtime.pipeline import AG3S

    arm = PlanarTwoLinkArm()
    scene3 = make_three_camera_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    ag3s = AG3S(_cfg3(), robot_model=arm)
    got = {}
    real_build = AG3S._build_esdf

    def build(self, cameras, *a, **kw):
        got["cameras"] = list(cameras)
        return real_build(self, cameras, *a, **kw)

    monkeypatch.setattr(AG3S, "_build_esdf", build)
    ag3s.process_multi(scene3.observations, phase="approach")
    assert len(got["cameras"]) == 3
    for cam in got["cameras"]:
        assert cam.device_depth is not None
        host = np.where(np.asarray(cam.robot_mask, bool), np.float32(0.0), np.asarray(cam.depth, np.float32))
        assert _same(cam.device_depth.cpu().numpy(), host)


def test_curobo_observations_use_the_device_depth_without_uploading():
    torch = _cuda()
    pytest.importorskip("curobo")
    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder
    from AG3S.ag3s.fields.esdf import CameraDepth

    rng = np.random.default_rng(0)
    depth = rng.uniform(0.3, 1.5, (48, 64))
    mask = rng.random((48, 64)) < 0.2
    d32 = np.where(mask, np.float32(0.0), depth.astype(np.float32))
    dev_depth = torch.as_tensor(d32, device="cuda")
    cams = [CameraDepth("a", depth, K, np.eye(4), robot_mask=mask, device_depth=dev_depth),
            CameraDepth("b", depth, K, np.eye(4), robot_mask=mask)]
    esdf = AG3SConfig.from_dict({"collision_backend": "esdf", "esdf": {
        "voxel_size": 0.02, "backend": "curobo", "fine_voxel_size": 0.005, "tsdf_voxel_size": 0.005,
        "bounds_lower": [0.2, -0.6, 0.6], "bounds_upper": [1.0, 0.6, 1.2]}}).esdf
    builder = CuroboFieldBuilder(esdf)
    prepared = builder._observations(cams, "cuda:0")                       # noqa: SLF001
    (_, ha, oa), (_, hb, ob) = prepared
    assert oa.depth_image.data_ptr() == dev_depth.data_ptr()               # no second upload
    assert torch.equal(oa.depth_image, ob.depth_image)                      # = the host path's values
    assert _same(ha, hb)


# --------------------------------------------------------------------------------- attention lift


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("src_hw,out_hw", [((16, 16), (480, 640)), ((7, 11), (37, 53)), ((30, 40), (30, 40)),
                                           ((64, 48), (20, 15))])
def test_resample_at_reads_the_full_map_exactly(mode, src_hw, out_hw):
    rng = np.random.default_rng(7)
    src = rng.random(src_hw).astype(np.float32) * rng.choice([1e-3, 1.0, 37.0])
    full = AL._resample(src, out_hw, mode)
    rows, cols = np.mgrid[0:out_hw[0], 0:out_hw[1]]
    got = AL._resample_at(src, out_hw, mode, rows.ravel(), cols.ravel())
    assert _same(got, full[rows.ravel(), cols.ravel()].astype(np.float32))


def test_lift_with_pixel_sampling_equals_the_full_map_path():
    rng = np.random.default_rng(5)
    hw = (480, 640)
    uv = np.stack([rng.integers(-3, hw[1] + 3, 40_000), rng.integers(-3, hw[0] + 3, 40_000)], 1).astype(np.int32)
    cloud = PointCloud(rng.normal(size=(40_000, 3)), uv)
    grid = rng.random(256).astype(np.float32)
    cfg = AG3SConfig().attention

    class Plain:                                         # an adapter without `sample_pixels`
        def __init__(self, inner):
            self.inner = inner

        def to_pixel_map(self, raw, out_hw):
            return self.inner.to_pixel_map(raw, out_hw)

    for adapter in (AL.GridAttentionAdapter.from_config(cfg), AL.DenseAttentionAdapter("bilinear"),
                    AL.GridAttentionAdapter(interpolation="nearest")):
        raw = grid if isinstance(adapter, AL.GridAttentionAdapter) else rng.random((120, 160)).astype(np.float32)
        a = AL.lift(cloud, raw, cfg, adapter=adapter, image_hw=hw)
        b = AL.lift(cloud, raw, cfg, adapter=Plain(adapter), image_hw=hw)
        assert _same(a.raw_attention, b.raw_attention) and _same(a.attention, b.attention)


# ------------------------------------------------------------------------------------- RANSAC


def _plane_cloud(rng, n, shares=(0.6, 0.25)):
    """A table (60 %), a floor (25 %) and clutter — the two planes the default config fits."""
    pts = []
    for k, share in enumerate(shares):
        m = int(n * share)
        xy = rng.uniform(-0.5, 0.5, (m, 2))
        pts.append(np.c_[xy, 0.8 - 0.3 * k + rng.normal(0, 0.003, m)])
    pts.append(rng.uniform([-0.5, -0.5, 0.3], [0.5, 0.5, 1.1], (n - sum(map(len, pts)), 3)))
    return np.vstack(pts)


@pytest.mark.parametrize("m", [1, 2, 31, 32, 33, 64, 65, 97, 200])
def test_gpu_hypothesis_counts_equal_numpy(m, monkeypatch):
    _cuda()
    rng = np.random.default_rng(m)
    pts = _plane_cloud(rng, 20_000)
    normals = rng.normal(size=(m, 3))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    offsets = rng.uniform(-0.2, 1.0, m)
    thr = 0.008
    # put points exactly on (and an ulp around) each hypothesis' threshold in numpy's own arithmetic
    d = pts @ normals.T
    offsets = d[rng.integers(0, len(pts), m), np.arange(m)] - thr
    monkeypatch.setattr(config, "SUPPORT_RANSAC_DEVICE", "cpu")
    ref = SS._hypothesis_counts(pts, normals, offsets, thr)
    monkeypatch.setattr(config, "SUPPORT_RANSAC_DEVICE", "auto")
    got = SS._hypothesis_counts(pts, normals, offsets, thr)
    assert got.dtype == np.int64 and np.array_equal(got, ref)
    # the old block loop, written out here
    old = []
    for start in range(0, m, 32):
        blk = slice(start, start + 32)
        old.extend((np.abs(pts @ normals[blk].T - offsets[blk]) <= thr).sum(axis=0).tolist())
    assert got.tolist() == old


@pytest.mark.parametrize("iters", [200, 65, 33, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fitted_planes_and_inliers_are_bit_identical(iters, seed, monkeypatch):
    _cuda()
    rng = np.random.default_rng(seed)
    pts = _plane_cloud(rng, 30_000)
    cfg = dataclasses.replace(SupportSurfaceConfig(), max_iterations=iters, seed=seed)
    out = {}
    for dev in ("cpu", "auto"):
        monkeypatch.setattr(config, "SUPPORT_RANSAC_DEVICE", dev)
        out[dev] = SS.fit_support_surfaces(PointCloud(pts), cfg, timestamp=0.0)
    (sa, ma), (sb, mb) = out["cpu"], out["auto"]
    assert _same(ma, mb) and len(sa) == len(sb)
    for a, b in zip(sa, sb):
        assert _same(a.normal, b.normal) and a.offset.hex() == b.offset.hex()
        assert _same(a.point_indices, b.point_indices) and a.rms_error.hex() == b.rms_error.hex()
    if iters >= 33:
        assert len(sa) == 2


def test_the_selected_hypothesis_is_the_first_with_the_most_inliers():
    counts = np.array([3, 9, 9, 1, 9])
    assert int(np.argmax(counts)) == 1


# ------------------------------------------------------------------------------- grow_region


def test_grow_region_flattening_is_the_generators():
    from scipy.spatial import cKDTree

    from AG3S.ag3s.stages.target_grounding import grow_region

    rng = np.random.default_rng(11)
    pts = rng.normal(size=(4000, 3)) * 0.1
    seeds = rng.integers(0, 4000, 30)

    def old(points, seed_indices, radius, max_points=None):
        visited = np.zeros(len(points), bool)
        visited[seed_indices] = True
        tree = cKDTree(points)
        frontier = np.asarray(seed_indices, np.int64)
        while frontier.size:
            nb = tree.query_ball_point(points[frontier], radius, workers=1)
            flat = np.fromiter((j for g in nb for j in g), np.int64, count=sum(len(g) for g in nb))
            if flat.size == 0:
                break
            cand = np.unique(flat)
            frontier = cand[~visited[cand]]
            visited[frontier] = True
            if max_points is not None and int(visited.sum()) >= max_points:
                break
        return np.nonzero(visited)[0]

    for r, mp in ((0.02, None), (0.035, None), (0.03, 500)):
        assert np.array_equal(grow_region(pts, seeds, r, max_points=mp), old(pts, seeds, r, mp))
