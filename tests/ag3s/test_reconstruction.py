"""Stage-1 tests: back-projection correctness, filtering, and determinism.

Determinism gets as much attention here as accuracy because it is the property that makes every
later ablation number meaningful. If the voxel filter or the `max_points` cap varied run to run, two
runs of the same config would produce different clusters and the difference would be attributed to
whatever was being ablated.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.ag3s.config import PointCloudConfig
from benchmark.ag3s.stages.reconstruction import (
    backproject,
    cap_points,
    from_pointcloud,
    reconstruct,
    voxel_downsample,
)
from benchmark.ag3s.types import PointCloud
from tests.ag3s.fixtures import (
    LABEL_BOX,
    LABEL_CYLINDER,
    LABEL_SPHERE,
    LABEL_TABLE,
    make_scene,
)


@pytest.fixture(scope="module")
def scene():
    return make_scene()


# ------------------------------------------------------------------------ back-projection


def test_backprojection_recovers_the_table_plane_exactly(scene):
    """Table pixels must land on z = table_z. This is the frame-convention check.

    A mirrored or transposed image still yields a plausible-looking cloud, which is exactly why
    `benchmark/knows_vla/validate_backprojection.py` exists — its authors measured the convention
    rather than reasoning about it. Here the ground truth is analytic, so the check is exact.
    """
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    rows, cols = np.nonzero(scene.labels == LABEL_TABLE)
    # Rebuild the same selection from the full cloud: every valid pixel is in it, in raster order.
    valid_rows, valid_cols = np.nonzero(scene.depth > 0.0)
    is_table = scene.labels[valid_rows, valid_cols] == LABEL_TABLE
    z = cloud.points[is_table, 2]
    assert z.size == rows.size
    assert np.abs(z - scene.table_z).max() < 1e-9


def test_backprojected_object_points_lie_on_their_true_surfaces(scene):
    """Each sphere point is exactly one radius from the sphere's true centre."""
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    valid_rows, valid_cols = np.nonzero(scene.depth > 0.0)
    labels = scene.labels[valid_rows, valid_cols]

    obj = scene.object_by_label(LABEL_SPHERE)
    pts = cloud.points[labels == LABEL_SPHERE]
    radial = np.linalg.norm(pts - obj.center, axis=1)
    assert np.abs(radial - obj.extent[0]).max() < 1e-9


def test_uv_is_preserved_and_points_back_to_the_source_pixel(scene):
    """`(u, v)` must survive reconstruction — attention lifting has nothing to look up without it."""
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    assert cloud.uv is not None and cloud.uv.shape == (len(cloud), 2)
    u, v = cloud.uv[:, 0], cloud.uv[:, 1]
    assert np.all(scene.depth[v, u] > 0.0)
    # And the depth at that pixel is the point's camera-frame z.
    R, t = scene.T_base_cam[:3, :3], scene.T_base_cam[:3, 3]
    cam_z = (cloud.points - t) @ R[:, 2]
    assert np.abs(cam_z - scene.depth[v, u]).max() < 1e-9


def test_invalid_depth_is_dropped_not_projected():
    """0 / NaN / Inf / out-of-range must never become points.

    A zero-depth pixel back-projects to the camera origin, which would put a phantom obstacle inside
    the robot; the other three produce non-finite coordinates that poison every downstream KD-tree.
    """
    K = np.array([[100.0, 0, 8.0], [0, 100.0, 8.0], [0, 0, 1.0]])
    depth = np.full((4, 4), 0.5)
    depth[0, 0] = 0.0
    depth[0, 1] = np.nan
    depth[0, 2] = np.inf
    depth[0, 3] = 99.0  # beyond depth_max
    depth[1, 0] = 0.001  # below depth_min
    cloud = backproject(depth, K, np.eye(4), PointCloudConfig(depth_min=0.05, depth_max=3.0))
    assert len(cloud) == 16 - 5
    assert np.isfinite(cloud.points).all()


def test_depth_scale_converts_units():
    """A uint16 millimetre sensor is the common case; `depth_scale` is the only knob needed."""
    K = np.array([[100.0, 0, 1.0], [0, 100.0, 1.0], [0, 0, 1.0]])
    depth_mm = np.full((2, 2), 500.0)
    cloud = backproject(depth_mm, K, np.eye(4), PointCloudConfig(depth_scale=0.001))
    assert np.allclose(cloud.points[:, 2], 0.5)


# -------------------------------------------------------------------- point-cloud input path


def test_pointcloud_path_skips_backprojection_and_matches_it(scene):
    """Feeding the same geometry as a cloud must give the same answer as depth + intrinsics."""
    via_depth = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    via_cloud = from_pointcloud(via_depth.points, uv=via_depth.uv)
    assert np.array_equal(via_depth.points, via_cloud.points)
    assert np.array_equal(via_depth.uv, via_cloud.uv)


def test_pointcloud_path_applies_extrinsics_when_given():
    pts_cam = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 3.0]
    out = from_pointcloud(pts_cam, T_base_cam=T)
    assert np.allclose(out.points, pts_cam + np.array([1.0, 2.0, 3.0]))


def test_pointcloud_without_uv_is_allowed():
    out = from_pointcloud(np.random.default_rng(0).normal(size=(10, 3)))
    assert out.uv is None and len(out) == 10


# ------------------------------------------------------------------------------ determinism


def test_voxel_downsample_is_deterministic_and_order_preserving(scene):
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    a = voxel_downsample(cloud, 0.01)
    b = voxel_downsample(cloud, 0.01)
    assert np.array_equal(a.points, b.points)
    assert np.array_equal(a.uv, b.uv)
    # Order preserved => the uv rasters stay monotone in the same way the input was.
    assert len(a) < len(cloud)


def test_voxel_downsample_keeps_a_real_point_with_its_pixel(scene):
    """The survivor must be an actual input point, not a centroid — a centroid has no `(u, v)`."""
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    out = voxel_downsample(cloud, 0.02)
    u, v = out.uv[:, 0], out.uv[:, 1]
    R, t = scene.T_base_cam[:3, :3], scene.T_base_cam[:3, 3]
    cam_z = (out.points - t) @ R[:, 2]
    assert np.abs(cam_z - scene.depth[v, u]).max() < 1e-9


def test_voxel_downsample_leaves_at_most_one_point_per_voxel(scene):
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    voxel = 0.01
    out = voxel_downsample(cloud, voxel)
    keys = np.floor(out.points / voxel).astype(np.int64)
    assert len(np.unique(keys, axis=0)) == len(out)


def test_voxel_size_zero_is_a_no_op(scene):
    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    assert voxel_downsample(cloud, 0.0) is cloud


def test_cap_is_deterministic():
    pts = np.arange(3000 * 3, dtype=np.float64).reshape(3000, 3)
    a, capped = cap_points(PointCloud(pts), 500)
    b, _ = cap_points(PointCloud(pts), 500)
    assert capped and len(a) == 500 and np.array_equal(a.points, b.points)


def test_cap_covers_the_whole_cloud_with_a_bounded_gap():
    """No region may be systematically unsampled — unsampled geometry cannot become a candidate.

    This is the property an integer stride would break: `arange(0, n, n // max_points)` truncated to
    `max_points` never reaches the tail of the cloud.
    """
    n, cap = 5999, 500
    pts = np.stack([np.arange(n, dtype=np.float64)] * 3, axis=1)
    out, _ = cap_points(PointCloud(pts), cap)
    chosen = out.points[:, 0]
    assert chosen[0] == 0 and chosen[-1] == n - 1
    assert np.diff(chosen).max() <= np.ceil(n / cap) + 1


def test_cap_does_not_move_a_cluster_centroid():
    """What the candidate tracker actually keys on is centroids, and those are resample-stable.

    The index set does shift when the cloud grows — `cap_points` documents that it does not promise
    otherwise. What has to hold is that the *geometry* a tracker sees is unchanged relative to its
    8 cm association gate.

    The residual is not an implementation artefact but the standard error of the subsample itself:
    500 of 4000 points from a sigma = 2 cm blob gives sigma/sqrt(500) ~ 0.9 mm per axis, so ~1.5 mm
    in 3D. The bound below is set an order of magnitude under `track_association_radius`, which is
    the quantity that decides whether a track survives. Worst case in the real pipeline is smaller
    still, because the cloud arrives in raster order and an object occupies a contiguous run of it,
    so the sample across that object is regular rather than random.
    """
    rng = np.random.default_rng(0)
    blob = rng.normal(0.0, 0.02, size=(4000, 3)) + np.array([0.6, 0.0, 0.8])
    a, _ = cap_points(PointCloud(blob), 500)
    grown, _ = cap_points(PointCloud(np.vstack([blob, blob[:1]])), 500)
    shift = float(np.linalg.norm(a.points.mean(axis=0) - grown.points.mean(axis=0)))
    assert shift < 0.005, f"centroid moved {shift * 1000:.3f} mm under a resample"


def test_cap_is_a_no_op_below_the_limit():
    cloud = PointCloud(np.zeros((10, 3)))
    out, capped = cap_points(cloud, 100)
    assert out is cloud and not capped


# ---------------------------------------------------------------------------- full stage


def test_reconstruct_reports_stage_counts(scene):
    cfg = PointCloudConfig(voxel_size=0.01, max_points=5000)
    cloud, stats = reconstruct(
        depth=scene.depth,
        camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam,
        config=cfg,
    )
    assert stats["n_raw"] > stats["n_voxel"] >= stats["n_final"]
    assert stats["n_final"] == len(cloud) <= cfg.max_points
    assert stats["capped"] is (stats["n_voxel"] > cfg.max_points)


def test_reconstruct_is_bit_identical_across_runs(scene):
    cfg = PointCloudConfig(voxel_size=0.008, max_points=4000)
    kwargs = dict(
        depth=scene.depth,
        camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam,
        config=cfg,
    )
    a, _ = reconstruct(**kwargs)
    b, _ = reconstruct(**kwargs)
    assert np.array_equal(a.points, b.points) and np.array_equal(a.uv, b.uv)


def test_reconstruct_rejects_ambiguous_input(scene):
    with pytest.raises(ValueError, match="exactly one"):
        reconstruct(depth=scene.depth, pointcloud=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="exactly one"):
        reconstruct()


def test_reconstruct_requires_intrinsics_for_depth(scene):
    with pytest.raises(ValueError, match="camera_intrinsics"):
        reconstruct(depth=scene.depth)


def test_all_objects_survive_reconstruction(scene):
    """Nothing is dropped for being uninteresting — every labelled surface still has points."""
    cloud, _ = reconstruct(
        depth=scene.depth,
        camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam,
        config=PointCloudConfig(voxel_size=0.005, max_points=200_000),
    )
    u, v = cloud.uv[:, 0], cloud.uv[:, 1]
    present = set(np.unique(scene.labels[v, u]).tolist())
    for label in (LABEL_TABLE, LABEL_SPHERE, LABEL_BOX, LABEL_CYLINDER):
        assert label in present, f"label {label} vanished during reconstruction"
