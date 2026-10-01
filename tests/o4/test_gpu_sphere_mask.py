"""T38 B2 — the robot self-filter decided once per camera, on the GPU, with the CPU's exact answer.

What is pinned here:

* **One decision rule, two backends, the same bits.** `sphere_mask_parts` on CUDA (torch float64,
  point axis chunked) and on CPU (the pre-B2 KD-tree loop) agree on every point, including points
  placed *on* an inflated or bare sphere surface and one ulp either side of it.
* **The CPU backend is the pre-B2 code.** A copy of the old `robot_sphere_mask` body, written out
  here independently, gives the same masks.
* **One pass per camera.** In the fused ESDF path each camera's self-filter is evaluated once; the
  TSDF depth mask and the voxel cloud both come from it and equal what the two separate calls gave.
* **The held-object count comes out of the same pass** (T32 H1 semantics: bare radius, no guard).
* **`AG3S.reset` frees the dropped field builder** even when it sits in a reference cycle.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import pytest

config = pytest.importorskip("benchmark.ag3s.config")
if not hasattr(config, "SELF_FILTER_DEVICE"):
    pytest.skip("SELF_FILTER_DEVICE absent — this is not the O4 worktree (T38 B2)",
                allow_module_level=True)

from benchmark.ag3s.config import AG3SConfig, AG3SConfigError  # noqa: E402
from benchmark.ag3s.runtime import multiview as MV  # noqa: E402
from benchmark.ag3s.runtime.pipeline import AG3S  # noqa: E402
from benchmark.ag3s.stages import robot_filter as RF  # noqa: E402
from benchmark.ag3s.stages.reconstruction import (  # noqa: E402
    backproject,
    downsample_and_cap,
    reconstruct,
)
from benchmark.ag3s.types import PointCloud  # noqa: E402
from tests.ag3s.fixtures import (  # noqa: E402
    DEFAULT_ARM_Q,
    FINGERTIP_LINK,
    PlanarTwoLinkArm,
    make_scene,
    make_three_camera_scene,
)


def _cuda_or_skip():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch


def _pre_b2_mask(points, centers, radii, inflation, guard_centre=None, guard_radius=0.0):
    """The pre-B2 `robot_sphere_mask` body (T19), written out by the test, not imported."""
    from scipy.spatial import cKDTree

    points = np.asarray(points, np.float64).reshape(-1, 3)
    inside = np.zeros(points.shape[0], bool)
    guarded = None
    if guard_centre is not None and guard_radius > 0.0:
        guarded = np.linalg.norm(points - np.asarray(guard_centre, np.float64), axis=1) <= guard_radius
        if not guarded.any():
            guarded = None
    touched = np.zeros(points.shape[0], bool) if guarded is not None else None
    tree = cKDTree(points)
    for centre, bare, radius in zip(centers, radii, radii + inflation):
        hits = tree.query_ball_point(centre, float(radius))
        if not hits:
            continue
        if guarded is None:
            inside[hits] = True
            continue
        hits = np.asarray(hits, np.intp)
        touched[hits] = True
        g = hits[guarded[hits]]
        keep = np.linalg.norm(points[g] - centre, axis=1) <= float(bare)
        inside[np.concatenate([hits[~guarded[hits]], g[keep]])] = True
    protected = np.zeros_like(inside) if touched is None else (touched & ~inside)
    return inside, protected


def _spheres_and_points(seed=0, n=40_000, s=120):
    """Random spheres + random points + points exactly on (and one ulp off) the spheres."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-0.4, 0.4, (s, 3))
    radii = rng.uniform(0.01, 0.06, s)
    inflation = rng.uniform(0.0, 0.02, s)
    pts = [rng.uniform(-0.5, 0.5, (n, 3))]
    k = rng.integers(0, s, 3000)
    u = rng.normal(size=(3000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    for rad in (radii[k] + inflation[k], radii[k]):
        on = centers[k] + u * rad[:, None]
        pts += [on, np.nextafter(on, centers[k]), np.nextafter(on, 2 * on - centers[k])]
    return np.vstack(pts), centers, radii, inflation


GUARDS = [(None, 0.0), (np.array([0.05, -0.02, 0.01]), 0.2), (np.array([5.0, 5.0, 5.0]), 0.1)]


@pytest.mark.parametrize("guard", GUARDS, ids=["no-guard", "guard", "guard-empty"])
def test_the_cpu_backend_is_the_pre_b2_code(guard):
    pts, c, r, infl = _spheres_and_points(1)
    parts = RF.sphere_mask_parts(pts, c, r, infl, guard_centre=guard[0], guard_radius=guard[1],
                                 device="cpu")
    inside, protected = _pre_b2_mask(pts, c, r, infl, *guard)
    assert parts.backend == "cpu"
    np.testing.assert_array_equal(parts.inside, inside)
    np.testing.assert_array_equal(parts.protected, protected)
    assert parts.held is None


@pytest.mark.parametrize("guard", GUARDS, ids=["no-guard", "guard", "guard-empty"])
@pytest.mark.parametrize("scalar_inflation", [False, True])
def test_gpu_equals_cpu_to_the_bit_on_and_around_every_surface(guard, scalar_inflation):
    _cuda_or_skip()
    pts, c, r, infl = _spheres_and_points(2)
    infl = 0.013 if scalar_inflation else infl
    held = np.arange(len(r) - 7, len(r))
    kw = dict(guard_centre=guard[0], guard_radius=guard[1], held_columns=held)
    cpu = RF.sphere_mask_parts(pts, c, r, infl, device="cpu", **kw)
    gpu = RF.sphere_mask_parts(pts, c, r, infl, device="cuda", **kw)
    assert (cpu.backend, gpu.backend) == ("cpu", "cuda")
    np.testing.assert_array_equal(gpu.inside, cpu.inside)
    np.testing.assert_array_equal(gpu.protected, cpu.protected)
    np.testing.assert_array_equal(gpu.held, cpu.held)
    assert cpu.inside.any() and (~cpu.inside).any() and cpu.held.any()
    if guard[0] is not None and guard[1] == 0.2:
        assert cpu.protected.any()            # the guard did something, so the test can see it


def test_gpu_chunking_does_not_change_the_answer(monkeypatch):
    _cuda_or_skip()
    pts, c, r, infl = _spheres_and_points(3, n=5000, s=50)
    g = np.array([0.0, 0.0, 0.0])
    ref = RF.sphere_mask_parts(pts, c, r, infl, guard_centre=g, guard_radius=0.15, device="cuda")
    for nbytes in (8 * 50, 8 * 50 * 777, 1):        # one row per chunk, odd size, < one row
        monkeypatch.setattr(config, "SELF_FILTER_GPU_CHUNK_BYTES", nbytes)
        out = RF.sphere_mask_parts(pts, c, r, infl, guard_centre=g, guard_radius=0.15,
                                   device="cuda")
        np.testing.assert_array_equal(out.inside, ref.inside)
        np.testing.assert_array_equal(out.protected, ref.protected)


def test_the_guard_membership_uses_numpys_norm_arithmetic():
    """The GPU guard test must be `np.linalg.norm(p - g) <= rho` to the bit: points placed on the
    guard sphere (and one ulp either side) are where a different summation order would show."""
    _cuda_or_skip()
    rng = np.random.default_rng(4)
    g = np.array([0.11, -0.07, 0.31])
    u = rng.normal(size=(20_000, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    on = g + 0.123 * u
    pts = np.vstack([on, np.nextafter(on, g), np.nextafter(on, 2 * on - g)])
    # one sphere covering everything, inflated, so every point is touched and only the guard
    # decides whether its bare test applies
    c, r, infl = np.array([[0.11, -0.07, 0.31]]), np.array([0.1]), np.array([0.5])
    cpu = RF.sphere_mask_parts(pts, c, r, infl, guard_centre=g, guard_radius=0.123, device="cpu")
    gpu = RF.sphere_mask_parts(pts, c, r, infl, guard_centre=g, guard_radius=0.123, device="cuda")
    np.testing.assert_array_equal(gpu.inside, cpu.inside)
    np.testing.assert_array_equal(gpu.protected, cpu.protected)
    assert cpu.protected.any() and (~cpu.protected).any()


def test_device_selection():
    pts, c, r, infl = _spheres_and_points(5, n=100, s=5)
    assert RF.sphere_mask_parts(pts, c, r, infl, device="cpu").backend == "cpu"
    with pytest.raises(AG3SConfigError):
        RF.sphere_mask_parts(pts, c, r, infl, device="gpu0")
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False
    auto = RF.sphere_mask_parts(pts, c, r, infl, device="auto").backend
    assert auto == ("cuda" if has_cuda else "cpu")
    if not has_cuda:
        with pytest.raises(RuntimeError):
            RF.sphere_mask_parts(pts, c, r, infl, device="cuda")


def test_empty_inputs_keep_their_old_shapes():
    for dev in ("cpu", "auto"):
        m = RF.robot_sphere_mask(np.zeros((0, 3)), np.zeros((3, 3)), np.ones(3), device=dev)
        assert m.shape == (0,) and m.dtype == bool
        m, p = RF.robot_sphere_mask(np.ones((4, 3)), np.zeros((0, 3)), np.zeros(0),
                                    return_protected=True, device=dev)
        assert m.shape == p.shape == (4,) and not m.any()
    with pytest.raises(ValueError):
        RF.robot_sphere_mask(np.ones((4, 3)), np.zeros((3, 3)), np.ones(2))


# ------------------------------------------------------------------ one pass per camera


def test_backproject_once_then_downsample_equals_reconstruct():
    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    cfg = AG3SConfig().pointcloud
    a, sa = reconstruct(depth=scene.depth, camera_intrinsics=scene.camera_intrinsics,
                        T_base_cam=scene.T_base_cam, config=cfg)
    full = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam, cfg)
    b, sb = downsample_and_cap(full, cfg)
    assert sa == sb
    np.testing.assert_array_equal(a.points, b.points)
    np.testing.assert_array_equal(a.uv, b.uv)


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_the_depth_mask_and_the_voxel_cloud_come_from_one_decision(device, monkeypatch):
    """`depth_robot_mask` once ≡ the two pre-B2 calls (cloud filter on the voxel cloud, and the
    depth mask on the full back-projection), with the guard on and a per-link override."""
    monkeypatch.setattr(config, "SELF_FILTER_DEVICE", device)
    arm = PlanarTwoLinkArm()
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    cfg = AG3SConfig.from_dict({"pointcloud": {"self_filter_inflation": 0.03,
                                               "self_filter_inflation_by_link": {FINGERTIP_LINK: 0.0},
                                               "self_filter_target_guard_radius": 0.08}}).pointcloud
    infl = RF.resolve_self_filter_inflation(arm.sphere_link_names, cfg)
    full = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam, cfg)
    centers, radii = arm.sphere_centers_numeric(DEFAULT_ARM_Q)
    # a guard centre where the guard actually protects something: on the fingertip's surface
    guard = centers[-1] + np.array([0.0, 0.0, radii[-1] + 0.01])

    dm = RF.depth_robot_mask(full, scene.depth.shape, arm, DEFAULT_ARM_Q, cfg,
                             inflation=infl, guard_centre=guard)
    # pre-B2 depth mask
    inside, protected = _pre_b2_mask(full.points, centers, radii, infl, guard, 0.08)
    expect = np.zeros(scene.depth.shape, bool)
    expect[full.uv[inside, 1], full.uv[inside, 0]] = True
    np.testing.assert_array_equal(dm.mask, expect)
    assert dm.n_guard_protected_px == int(protected.sum())
    assert dm.n_held_px is None                       # the arm holds nothing
    assert dm.mask.any() and (~dm.mask).any()

    # pre-B2 cloud filter on the voxel cloud cut from the same back-projection
    voxel, _ = downsample_and_cap(full, cfg)
    kept_old, stats_old = RF.filter_robot_points(voxel, arm, DEFAULT_ARM_Q, cfg, inflation=infl,
                                                 guard_centre=guard)
    kept_new, stats_new = dm.filter_cloud(voxel)
    np.testing.assert_array_equal(kept_new.points, kept_old.points)
    np.testing.assert_array_equal(kept_new.uv, kept_old.uv)
    assert stats_new == stats_old
    assert stats_new["n_removed"] > 0


def _ag3s(arm, backend="esdf"):
    cfg = AG3SConfig.from_dict({
        "collision_backend": backend,
        "esdf": {"voxel_size": 0.02, "fine_voxel_size": None},
        "pointcloud": {"self_filter_inflation": 0.02,
                       "self_filter_inflation_by_link": {FINGERTIP_LINK: 0.0},
                       "self_filter_target_guard_radius": 0.08}})
    return AG3S(cfg, robot_model=arm)


def test_the_fused_esdf_path_decides_each_camera_once(monkeypatch):
    arm = PlanarTwoLinkArm()
    scene3 = make_three_camera_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    ag3s = _ag3s(arm)
    ag3s.process_multi(scene3.observations, phase="approach")       # frame 1: confirms a target

    calls, got = [], {}
    real_mask, real_build = RF.robot_sphere_mask, AG3S._build_esdf

    def spy(points, *a, **kw):
        calls.append(len(points))
        return real_mask(points, *a, **kw)

    def build(self, cameras, *a, **kw):
        got["cameras"] = list(cameras)
        return real_build(self, cameras, *a, **kw)

    monkeypatch.setattr(RF, "robot_sphere_mask", spy)
    monkeypatch.setattr(AG3S, "_build_esdf", build)
    cs = ag3s.process_multi(scene3.observations, phase="approach")  # frame 2: guard (if any) on
    n = len(scene3.observations)
    n_full = [len(backproject(o.depth, o.camera_intrinsics, o.resolve_T_base_cam(arm),
                              ag3s.config.pointcloud)) for o in scene3.observations]
    assert calls == n_full, "exactly one full-resolution decision per camera, nothing else"
    assert ag3s._frame_depth_masks == []                             # noqa: SLF001 — consumed

    # the masks the TSDF got = what the pre-B2 per-camera call computes (frame state unchanged)
    monkeypatch.setattr(RF, "robot_sphere_mask", real_mask)
    guard_px = []
    for cam, obs in zip(got["cameras"], scene3.observations):
        before = len(ag3s._mask_guard_px)                            # noqa: SLF001
        expect = ag3s._robot_mask_for(obs.depth, obs.camera_intrinsics,  # noqa: SLF001
                                      obs.resolve_T_base_cam(arm), obs.robot_state)
        guard_px.append(ag3s._mask_guard_px[before])                 # noqa: SLF001
        np.testing.assert_array_equal(cam.robot_mask, expect)
    sf = cs.metrics["self_filter"]
    assert sf["n_depth_masks"] == n
    assert sf["n_guard_protected_depth_px"] == sum(guard_px)


def test_the_fused_cloud_is_unchanged_by_the_one_pass_path():
    arm = PlanarTwoLinkArm()
    scene3 = make_three_camera_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    cfg = _ag3s(arm).config
    infl = RF.resolve_self_filter_inflation(arm.sphere_link_names, cfg.pointcloud)
    centers, _ = arm.sphere_centers_numeric(DEFAULT_ARM_Q)
    guard = centers[-1]
    for obs in scene3.observations:
        old = MV.process_observation(obs, cfg, robot_model=arm, self_filter_inflation=infl,
                                     self_filter_guard_centre=guard)
        new = MV.process_observation(obs, cfg, robot_model=arm, self_filter_inflation=infl,
                                     self_filter_guard_centre=guard, with_depth_mask=True)
        assert old.depth_mask is None and new.depth_mask is not None
        np.testing.assert_array_equal(new.cloud.points, old.cloud.points)
        np.testing.assert_array_equal(new.cloud.uv, old.cloud.uv)
        np.testing.assert_array_equal(new.attention, old.attention)
        assert new.stats == old.stats


def test_the_primitive_backend_does_not_build_depth_masks(monkeypatch):
    arm = PlanarTwoLinkArm()
    scene3 = make_three_camera_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    ag3s = _ag3s(arm, backend="primitive")
    calls = []
    real = RF.robot_sphere_mask
    monkeypatch.setattr(RF, "robot_sphere_mask",
                        lambda p, *a, **k: (calls.append(len(p)), real(p, *a, **k))[1])
    ag3s.process_multi(scene3.observations, phase="approach")
    voxel = [len(downsample_and_cap(backproject(o.depth, o.camera_intrinsics,
                                                o.resolve_T_base_cam(arm), ag3s.config.pointcloud),
                                    ag3s.config.pointcloud)[0]) for o in scene3.observations]
    assert calls == voxel        # the voxel clouds only, as before B2


def test_the_held_count_comes_out_of_the_same_pass(monkeypatch):
    """While attached the filter model appends the held spheres; their pixel count (T32 H1: bare
    radius, no guard) is read from those columns of the one pass and equals the separate call."""
    from tests.ag3s.test_t32_held_object import _held

    arm = PlanarTwoLinkArm()
    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 2, "max_attached_primitives": 4},
                                       "collision_backend": "esdf",
                                       "esdf": {"voxel_size": 0.02, "fine_voxel_size": None}})
    ag3s = AG3S(cfg, robot_model=arm, attached_parent_links=(FINGERTIP_LINK, "link_1"))
    held = _held(arm, DEFAULT_ARM_Q)
    ag3s.attach(held)
    ag3s._prepare_frame_filter()                                         # noqa: SLF001
    model = ag3s._frame_filter_model                                     # noqa: SLF001
    assert model.n_held >= 1
    hc, hr = model.held_spheres(DEFAULT_ARM_Q)
    # a camera looking at the held sphere from 0.5 m
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    calls = []
    real = RF.robot_sphere_mask
    monkeypatch.setattr(RF, "robot_sphere_mask",
                        lambda p, *a, **k: (calls.append(k.get("held_columns")), real(p, *a, **k))[1])
    mask = ag3s._robot_mask_for(scene.depth, scene.camera_intrinsics,  # noqa: SLF001
                                scene.T_base_cam, DEFAULT_ARM_Q)
    assert mask is not None and len(calls) == 1 and calls[0] is not None   # one pass, held columns
    full = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam, cfg.pointcloud)
    expect = int(real(full.points, hc, hr, 0.0, device="cpu").sum())
    assert ag3s._mask_held_px == [expect]                                # noqa: SLF001


# ------------------------------------------------------------------ reset frees the GPU state


class _CyclicBuilder:
    """Stands in for `CuroboFieldBuilder`: its state sits in a reference cycle, like cuRobo's
    `GraphExecutor(capture_fn=self._compute_esdf_impl)`."""

    def __init__(self):
        self.executor = self._impl          # bound method -> self: a cycle
        self.payload = bytearray(1 << 20)

    def _impl(self):
        return None


def test_reset_collects_a_cyclic_field_builder(monkeypatch):
    arm = PlanarTwoLinkArm()
    ag3s = _ag3s(arm)
    gc.disable()
    try:
        builder = _CyclicBuilder()
        ref = weakref.ref(builder)
        ag3s._esdf_builder = builder                                     # noqa: SLF001
        del builder
        ag3s.reset()
        assert ref() is None, "the cyclic builder outlived reset()"
    finally:
        gc.enable()


def test_reset_without_a_builder_does_not_collect(monkeypatch):
    arm = PlanarTwoLinkArm()
    ag3s = _ag3s(arm)
    n = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: n.append(1) or 0)
    ag3s.reset()
    assert n == []
