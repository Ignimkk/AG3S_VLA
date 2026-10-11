"""Stage-3 tests: adapters, resolution mismatch, normalization, and the no-drop invariant.

The invariant tested hardest is the boring one — `len(lift(cloud, ...)) == len(cloud)`, always. It is
the type-level expression of "geometry decides what can collide", and it is the assertion that would
catch a well-meaning future optimization that skips points below some attention floor.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.ag3s.stages.attention_lifting import (
    DenseAttentionAdapter,
    GridAttentionAdapter,
    lift,
    make_adapter,
    normalize_attention,
)
from AG3S.ag3s.config import AG3SConfig, AttentionConfig
from AG3S.ag3s.stages.reconstruction import backproject, reconstruct
from AG3S.ag3s.types import AttentionAdapter, PointCloud
from tests.ag3s.fixtures import LABEL_BOX, LABEL_SPHERE, make_scene


@pytest.fixture(scope="module")
def scene():
    return make_scene()


@pytest.fixture(scope="module")
def cloud(scene):
    out, _ = reconstruct(
        depth=scene.depth,
        camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam,
        config=AG3SConfig().pointcloud,
    )
    return out


# ---------------------------------------------------------------------------------- adapters


def test_adapters_satisfy_the_protocol():
    assert isinstance(GridAttentionAdapter(), AttentionAdapter)
    assert isinstance(DenseAttentionAdapter(), AttentionAdapter)


def test_grid_adapter_accepts_a_flat_patch_vector():
    """π0.5 hands back 256 agent-view patches; sqrt gives the 16x16 grid."""
    raw = np.arange(256, dtype=np.float32)
    grid = GridAttentionAdapter().to_grid(raw)
    assert grid.shape == (16, 16)
    assert np.array_equal(grid, raw.reshape(16, 16))


def test_grid_adapter_selects_layer_and_head():
    """The `[L, heads, tokens, patches]` block is exactly probe_p0b's shape."""
    raw = np.zeros((18, 8, 4, 256), np.float32)
    raw[12, 3] = 7.0  # the paper's cell
    grid = GridAttentionAdapter(layer=12, head=3).to_grid(raw)
    assert np.allclose(grid, 7.0)
    # A different cell must give a different answer, or the selection is not happening.
    assert np.allclose(GridAttentionAdapter(layer=0, head=0).to_grid(raw), 0.0)


def test_grid_adapter_averages_the_token_axis():
    raw = np.zeros((2, 2, 4, 4), np.float32)
    raw[0, 0, :, 0] = [0.0, 1.0, 2.0, 3.0]
    grid = GridAttentionAdapter(layer=0, head=0).to_grid(raw)
    assert grid[0, 0] == pytest.approx(1.5)


def test_grid_adapter_refuses_to_guess_a_layer_head():
    """Silently defaulting to (0, 0) would report one head's numbers under another's name."""
    with pytest.raises(ValueError, match="attention.layer and attention.head"):
        GridAttentionAdapter().to_grid(np.zeros((18, 8, 256), np.float32))


def test_grid_adapter_rejects_a_non_square_patch_count():
    with pytest.raises(ValueError, match="not a square grid"):
        GridAttentionAdapter().to_grid(np.zeros(200, np.float32))


def test_make_adapter_routes_by_shape(scene):
    assert isinstance(make_adapter(scene.attention_grid, AttentionConfig()), GridAttentionAdapter)
    assert isinstance(make_adapter(np.zeros((480, 640)), AttentionConfig()), DenseAttentionAdapter)


# ------------------------------------------------------------------------ resolution mismatch


@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_upsampling_a_16x16_grid_to_the_depth_resolution(scene, mode):
    """The realistic case: a 16x16 patch grid stretched 30-40x onto a 480x640 depth image."""
    adapter = GridAttentionAdapter(interpolation=mode)
    pixel_map = adapter.to_pixel_map(scene.attention_grid, scene.hw)
    assert pixel_map.shape == scene.hw
    assert pixel_map.dtype == np.float32
    assert pixel_map.min() >= scene.attention_grid.min() - 1e-6
    assert pixel_map.max() <= scene.attention_grid.max() + 1e-6


def test_bilinear_and_nearest_agree_on_where_the_peak_is(scene):
    """Interpolation is an ablation axis; it may change values but not which object is peaked on."""
    maps = {
        m: GridAttentionAdapter(interpolation=m).to_pixel_map(scene.attention_grid, scene.hw)
        for m in ("bilinear", "nearest")
    }
    peaks = {m: np.unravel_index(np.argmax(a), a.shape) for m, a in maps.items()}
    v0, u0 = peaks["bilinear"]
    v1, u1 = peaks["nearest"]
    # Within one patch of each other (a patch is 640/16 = 40 px wide).
    assert abs(int(v0) - int(v1)) <= 40 and abs(int(u0) - int(u1)) <= 40


def test_bilinear_is_smooth_and_nearest_is_blocky(scene):
    """If these came out the same, the interpolation switch would be doing nothing."""
    bil = GridAttentionAdapter(interpolation="bilinear").to_pixel_map(scene.attention_grid, scene.hw)
    nea = GridAttentionAdapter(interpolation="nearest").to_pixel_map(scene.attention_grid, scene.hw)
    assert np.abs(np.diff(bil, axis=1)).max() < np.abs(np.diff(nea, axis=1)).max()


def test_identity_resample_when_resolutions_already_match():
    a = np.random.default_rng(0).random((8, 8)).astype(np.float32)
    assert np.array_equal(DenseAttentionAdapter().to_pixel_map(a, (8, 8)), a)


def test_dense_adapter_downsamples_too():
    a = np.random.default_rng(0).random((480, 640)).astype(np.float32)
    assert DenseAttentionAdapter().to_pixel_map(a, (120, 160)).shape == (120, 160)


# ------------------------------------------------------------------------------ normalization


@pytest.mark.parametrize("mode", ["percentile", "minmax", "softmax", "none"])
def test_every_normalization_preserves_the_ranking(mode):
    """The property that lets `normalization` and `seed_percentile` be chosen independently."""
    rng = np.random.default_rng(0)
    values = rng.random(500).astype(np.float32)
    out = normalize_attention(values, AttentionConfig(normalization=mode))
    order_in = np.argsort(values, kind="stable")
    assert np.all(np.diff(out[order_in]) >= -1e-6)


@pytest.mark.parametrize("mode", ["percentile", "minmax", "softmax"])
def test_normalizations_land_in_the_unit_interval(mode):
    rng = np.random.default_rng(1)
    out = normalize_attention(rng.random(500).astype(np.float32) * 37.0, AttentionConfig(normalization=mode))
    assert out.min() >= 0.0 and out.max() == pytest.approx(1.0, abs=1e-5)


def test_percentile_normalization_is_robust_to_a_single_hot_patch():
    """The reason it is the default: one outlier must not crush the rest of the map to zero."""
    values = np.concatenate([np.linspace(0.0, 1.0, 100), [1000.0]]).astype(np.float32)
    minmax = normalize_attention(values, AttentionConfig(normalization="minmax"))
    pct = normalize_attention(values, AttentionConfig(normalization="percentile"))
    assert minmax[:100].max() < 0.01  # everything real is squashed
    assert pct[:100].max() > 0.9  # the real range survives


def test_softmax_temperature_controls_contrast():
    values = np.linspace(0.0, 1.0, 64).astype(np.float32)
    sharp = normalize_attention(values, AttentionConfig(normalization="softmax", temperature=0.1))
    soft = normalize_attention(values, AttentionConfig(normalization="softmax", temperature=10.0))
    assert sharp.mean() < soft.mean()


def test_degenerate_attention_normalizes_to_zeros():
    """A flat map must produce zeros so grounding can report NO_ATTENTION rather than inventing one."""
    for mode in ("percentile", "minmax"):
        out = normalize_attention(np.full(100, 0.5, np.float32), AttentionConfig(normalization=mode))
        assert np.all(out == 0.0)


def test_unknown_normalization_is_rejected():
    cfg = dataclasses.replace(AttentionConfig(), normalization="percentile")
    object.__setattr__(cfg, "normalization", "bogus")  # bypass config validation on purpose
    with pytest.raises(ValueError, match="unknown attention.normalization"):
        normalize_attention(np.zeros(3, np.float32), cfg)


def test_empty_input_is_handled():
    assert normalize_attention(np.zeros(0, np.float32), AttentionConfig()).size == 0


# ----------------------------------------------------------------------------------- lifting


def test_lift_never_drops_a_point(scene, cloud):
    """The core invariant. Low attention must not remove geometry."""
    apc = lift(cloud, scene.attention_grid, AG3SConfig().attention, image_hw=scene.hw)
    assert len(apc) == len(cloud)
    assert apc.cloud is cloud
    assert np.array_equal(apc.points, cloud.points)


def test_lift_keeps_near_zero_attention_points(scene, cloud):
    """Explicitly: the far cylinder gets almost no attention and must still be present."""
    apc = lift(cloud, scene.attention_grid, AG3SConfig().attention, image_hw=scene.hw)
    assert float(apc.attention.min()) < 0.05
    assert int((apc.attention < 0.05).sum()) > 0.5 * len(apc)


def test_lift_puts_the_highest_attention_on_the_target(scene, cloud):
    """The whole point of the stage: the peak lands on the object the fixture aimed it at."""
    apc = lift(cloud, scene.attention_grid, AG3SConfig().attention, image_hw=scene.hw)
    u, v = cloud.uv[:, 0], cloud.uv[:, 1]
    labels = scene.labels[v, u]
    assert labels[int(np.argmax(apc.peak_signal))] == LABEL_SPHERE
    sphere_mean = apc.attention[labels == LABEL_SPHERE].mean()
    box_mean = apc.attention[labels == LABEL_BOX].mean()
    assert sphere_mean > box_mean


def test_normalized_argmax_is_ambiguous_but_the_peak_signal_is_not(scene, cloud):
    """The reason `AttentionPointCloud` carries two arrays.

    Percentile normalization clips its top percentile, so `attention` has hundreds of exact ties at
    1.0 spanning more than one surface and `argmax` is decided by raster order. `peak_signal` keeps
    the unclipped ranking, so the peak is a single well-defined point on the target.
    """
    apc = lift(cloud, scene.attention_grid, AG3SConfig().attention, image_hw=scene.hw)
    u, v = cloud.uv[:, 0], cloud.uv[:, 1]
    labels = scene.labels[v, u]

    tied = np.nonzero(apc.attention >= apc.attention.max() - 1e-6)[0]
    assert tied.size > 100, "fixture should exercise the clipping regime"
    assert len(np.unique(labels[tied])) > 1, "ties should span more than one surface"

    peak_tied = np.nonzero(apc.peak_signal >= apc.peak_signal.max() - 1e-9)[0]
    assert peak_tied.size < 10
    assert set(labels[peak_tied].tolist()) == {LABEL_SPHERE}


def test_raw_attention_is_carried_and_survives_selection(scene, cloud):
    apc = lift(cloud, scene.attention_grid, AG3SConfig().attention, image_hw=scene.hw)
    assert apc.raw_attention is not None and apc.raw_attention.shape == apc.attention.shape
    subset = apc.select(np.arange(0, len(apc), 7))
    assert subset.raw_attention is not None
    assert np.array_equal(subset.raw_attention, apc.raw_attention[::7])


def test_lift_leaks_onto_the_neighbour_as_the_fixture_intends(scene, cloud):
    """Sanity check on the fixture, not on AG3S.

    If the synthetic attention did not actually smear onto the box, the 3D-vs-2D comparison in
    `test_target_grounding.py` would be measuring nothing.
    """
    apc = lift(cloud, scene.attention_grid, AG3SConfig().attention, image_hw=scene.hw)
    u, v = cloud.uv[:, 0], cloud.uv[:, 1]
    labels = scene.labels[v, u]
    assert apc.attention[labels == LABEL_BOX].max() > 0.5


def test_lift_uses_the_stored_uv_exactly(scene, cloud):
    """Integer uv is an exact lookup, so the lifted value equals the map at that pixel."""
    adapter = GridAttentionAdapter()
    pixel_map = adapter.to_pixel_map(scene.attention_grid, scene.hw)
    cfg = dataclasses.replace(AG3SConfig().attention, normalization="none")
    apc = lift(cloud, scene.attention_grid, cfg, adapter=adapter, image_hw=scene.hw)
    u, v = cloud.uv[:, 0], cloud.uv[:, 1]
    assert np.abs(apc.attention - pixel_map[v, u]).max() < 1e-6


def test_lift_falls_back_to_projection_without_uv(scene):
    """A sensor that delivers 3D with no image behind it still has to work."""
    with_uv = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    without_uv = PointCloud(with_uv.points)
    cfg = dataclasses.replace(AG3SConfig().attention, normalization="none")

    a = lift(with_uv, scene.attention_grid, cfg, image_hw=scene.hw)
    b = lift(
        without_uv,
        scene.attention_grid,
        cfg,
        image_hw=scene.hw,
        camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam,
    )
    # Projection reconstructs the same correspondence to sub-pixel accuracy.
    assert np.abs(a.attention - b.attention).max() < 1e-3


def test_lift_without_uv_and_without_intrinsics_is_an_error():
    cloud = PointCloud(np.zeros((4, 3)))
    with pytest.raises(ValueError, match="camera_intrinsics"):
        lift(cloud, np.zeros((16, 16), np.float32), AttentionConfig())


def test_lift_of_an_empty_cloud_is_empty():
    apc = lift(PointCloud.empty(), np.zeros((16, 16), np.float32), AttentionConfig())
    assert len(apc) == 0


def test_lift_is_deterministic(scene, cloud):
    cfg = AG3SConfig().attention
    a = lift(cloud, scene.attention_grid, cfg, image_hw=scene.hw)
    b = lift(cloud, scene.attention_grid, cfg, image_hw=scene.hw)
    assert np.array_equal(a.attention, b.attention)


def test_attention_point_cloud_rejects_a_length_mismatch():
    from AG3S.ag3s.types import AttentionPointCloud

    with pytest.raises(ValueError, match="attention has 3 values for 4 points"):
        AttentionPointCloud(PointCloud(np.zeros((4, 3))), np.zeros(3, np.float32))
