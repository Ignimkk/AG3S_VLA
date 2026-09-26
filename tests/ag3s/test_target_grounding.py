"""Stage-4 tests: numeric grounding accuracy, the 2D-vs-3D claim, and every failure mode.

The fixture's ground truth is analytic, so "grounding works" is a number here rather than a
judgement. The two that matter most:

* **IoU against the GT target mask** — does the cluster contain the object and nothing else.
* **3D connectivity vs 2D attention alone** — the design claim of the whole module. The fixture
  deliberately leaks attention onto a neighbouring object, so a 2D threshold merges them; if the 3D
  path did not separate them, this file would say so.

Failure modes get equal weight. `target=None` with a reason is a correct answer, and each of the
five `GroundingStatus` values is driven from a real configuration rather than constructed by hand.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmark.ag3s.stages.attention_lifting import lift
from benchmark.ag3s.config import AG3SConfig, ClusteringConfig
from benchmark.ag3s.stages.reconstruction import reconstruct
from benchmark.ag3s.stages.support_surface import fit_support_surfaces
from benchmark.ag3s.stages.target_grounding import (
    ClusterInfo,
    connected_components_3d,
    dbscan,
    extract_seeds,
    ground_target,
    grow_region,
)
from benchmark.ag3s.types import (
    AttentionPointCloud,
    GroundingStatus,
    PointCloud,
    PrimitiveType,
)
from tests.ag3s.fixtures import (
    LABEL_BOX,
    LABEL_CYLINDER,
    LABEL_SPHERE,
    make_scene,
)

CFG = AG3SConfig()


@pytest.fixture(scope="module")
def scene():
    return make_scene()


@pytest.fixture(scope="module")
def grounded(scene):
    """Reconstruct -> lift -> label lookup, shared by the accuracy tests."""
    cloud, _ = reconstruct(
        depth=scene.depth,
        camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam,
        config=CFG.pointcloud,
    )
    apc = lift(cloud, scene.attention_grid, CFG.attention, image_hw=scene.hw)
    labels = scene.labels[cloud.uv[:, 1], cloud.uv[:, 0]]
    # The real RANSAC support-surface fit. Grounding needs the plane gone before connectivity runs
    # (see the module docstring), so this is the actual pipeline order, not a stand-in.
    _, table_mask = fit_support_surfaces(cloud, CFG.support_surface)
    return cloud, apc, labels, table_mask


def _iou(indices: np.ndarray, labels: np.ndarray, target_label: int) -> tuple[float, float, float]:
    selected = labels[indices]
    tp = int((selected == target_label).sum())
    gt = int((labels == target_label).sum())
    union = indices.size + gt - tp
    return tp / union, tp / max(indices.size, 1), tp / max(gt, 1)


# --------------------------------------------------------------------------- numeric accuracy


def test_grounding_recovers_the_target_exactly(scene, grounded):
    """IoU, precision and recall against the analytic GT mask."""
    cloud, apc, labels, table = grounded
    result = ground_target(apc, CFG.clustering, seed_percentile=95.0, exclude_mask=table)

    assert result.status is GroundingStatus.OK and result.target is not None
    iou, precision, recall = _iou(result.target.point_indices, labels, LABEL_SPHERE)
    assert iou > 0.95, f"IoU {iou:.4f}"
    assert precision > 0.95 and recall > 0.95


def test_target_centroid_matches_the_visible_surface(scene, grounded):
    """Compared against the *visible* centroid, not the true centre.

    A single view sees the near hemisphere only, so the observed centroid sits ~2.8 cm toward the
    camera. That bias is the camera's, not AG3S's; asserting against the true centre would be
    demanding that grounding hallucinate the far side.
    """
    _, apc, _, table = grounded
    result = ground_target(apc, CFG.clustering, seed_percentile=95.0, exclude_mask=table)
    error = float(np.linalg.norm(result.target.centroid - scene.visible_centroid(LABEL_SPHERE)))
    assert error < 0.01, f"centroid off by {error * 100:.2f} cm"

    true_centre_gap = float(np.linalg.norm(result.target.centroid - scene.target.center))
    assert true_centre_gap > error, "single-view bias should be the larger of the two"


def test_target_bounding_sphere_contains_its_points(grounded):
    _, apc, _, table = grounded
    target = ground_target(apc, CFG.clustering, seed_percentile=95.0, exclude_mask=table).target
    sphere = target.bounding_geometry
    assert sphere.type is PrimitiveType.SPHERE
    radii = np.linalg.norm(target.points - sphere.center, axis=1)
    assert radii.max() <= sphere.dimensions[0] + 1e-9


def test_all_six_cluster_metrics_are_populated(grounded):
    _, apc, _, table = grounded
    result = ground_target(apc, CFG.clustering, seed_percentile=95.0, exclude_mask=table)
    for cluster in result.clusters:
        assert isinstance(cluster, ClusterInfo)
        assert cluster.point_count > 0
        assert 0.0 <= cluster.mean_attention <= 1.0
        assert cluster.max_attention >= cluster.mean_attention
        assert cluster.attention_mass > 0.0
        assert 0.0 < cluster.spatial_compactness <= 1.0
        assert cluster.distance_from_attention_peak >= 0.0
    assert set(result.target.metrics) >= {
        "mean_attention", "max_attention", "attention_mass",
        "point_count", "spatial_compactness", "distance_from_attention_peak",
    }


def test_the_attention_peak_lands_on_the_target(scene, grounded):
    cloud, apc, labels, table = grounded
    result = ground_target(apc, CFG.clustering, seed_percentile=95.0, exclude_mask=table)
    assert labels[result.attention_peak_index] == LABEL_SPHERE
    # And the winning cluster is much closer to it than the runner-up.
    assert result.clusters[0].distance_from_attention_peak < result.clusters[1].distance_from_attention_peak


# ------------------------------------------------ the design claim: 3D connectivity vs 2D alone


def test_3d_connectivity_separates_what_2d_attention_merges(scene, grounded):
    """The headline comparison.

    The fixture leaks 55% of the target's attention onto a box 13.5 cm away, whose surface is 5 cm
    from the target's — wider than the 3 cm `eps`. A 2D threshold cannot tell them apart; 3D
    connectivity does, and the numbers below are the margin.
    """
    _, apc, labels, table = grounded
    kw = dict(seed_percentile=95.0, exclude_mask=table)

    three_d = ground_target(apc, CFG.clustering, **kw)
    two_d = ground_target(apc, dataclasses.replace(CFG.clustering, use_3d_connectivity=False), **kw)

    iou_3d, precision_3d, _ = _iou(three_d.target.point_indices, labels, LABEL_SPHERE)
    iou_2d, precision_2d, _ = _iou(two_d.target.point_indices, labels, LABEL_SPHERE)

    assert iou_3d > 0.95 and iou_2d < 0.6
    assert iou_3d > iou_2d + 0.4, f"3D {iou_3d:.4f} vs 2D {iou_2d:.4f}"

    # The 2D target is contaminated by the neighbour; the 3D one is not.
    assert int((labels[two_d.target.point_indices] == LABEL_BOX).sum()) > 100
    assert int((labels[three_d.target.point_indices] == LABEL_BOX).sum()) == 0

    err_3d = np.linalg.norm(three_d.target.centroid - scene.visible_centroid(LABEL_SPHERE))
    err_2d = np.linalg.norm(two_d.target.centroid - scene.visible_centroid(LABEL_SPHERE))
    assert err_3d < 0.01 < err_2d, f"centroid error 3D {err_3d:.4f} m vs 2D {err_2d:.4f} m"


def test_grounding_ignores_objects_attention_never_pointed_at(grounded):
    """The far cylinder must not become the target — and must not be clustered here at all.

    It is not lost: nothing was deleted, and stage 5 will pick it up as a collision candidate from
    the residual cloud. Grounding's job is only to name the target.
    """
    _, apc, labels, table = grounded
    result = ground_target(apc, CFG.clustering, seed_percentile=95.0, exclude_mask=table)
    clustered = np.concatenate([c.point_indices for c in result.clusters])
    assert int((labels[clustered] == LABEL_CYLINDER).sum()) == 0
    assert int((labels == LABEL_CYLINDER).sum()) > 0  # still in the cloud


# ------------------------------------------------------------------------- clustering methods


def test_dbscan_finds_known_clusters_and_flags_noise():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.005, (100, 3))
    b = rng.normal(0, 0.005, (100, 3)) + np.array([1.0, 0.0, 0.0])
    outlier = np.array([[5.0, 5.0, 5.0]])
    labels = dbscan(np.vstack([a, b, outlier]), eps=0.03, min_points=10)
    assert len(set(labels[:200].tolist())) == 2
    assert labels[200] == -1
    assert len(set(labels[:100].tolist())) == 1 and len(set(labels[100:200].tolist())) == 1


def test_dbscan_prunes_a_sparse_bridge_where_plain_connectivity_does_not():
    """Why both methods exist, in one test.

    Two dense blobs joined by a thin line of points spaced 2 cm apart. Plain Euclidean connectivity
    chains straight through the bridge and merges them; DBSCAN's `min_points` test refuses to make
    the bridge points core, so the merge never propagates.
    """
    rng = np.random.default_rng(1)
    a = rng.normal(0, 0.008, (200, 3))
    b = rng.normal(0, 0.008, (200, 3)) + np.array([0.30, 0.0, 0.0])
    bridge = np.stack([np.arange(0.04, 0.28, 0.02), np.zeros(12), np.zeros(12)], axis=1)
    points = np.vstack([a, b, bridge])

    assert len(set(connected_components_3d(points, 0.03).tolist())) == 1

    labels = dbscan(points, eps=0.03, min_points=20)
    dense_labels = set(labels[:400].tolist()) - {-1}
    assert len(dense_labels) == 2, f"DBSCAN merged the blobs: {dense_labels}"


def test_clustering_method_is_an_ablation_axis(grounded):
    _, apc, labels, table = grounded
    kw = dict(seed_percentile=95.0, exclude_mask=table)
    for method in ("dbscan", "region_growing"):
        result = ground_target(apc, dataclasses.replace(CFG.clustering, method=method), **kw)
        iou, _, _ = _iou(result.target.point_indices, labels, LABEL_SPHERE)
        assert iou > 0.95, f"{method}: IoU {iou:.4f}"


def test_grounding_is_stable_across_attention_normalizations(scene, grounded):
    """Normalization changes values, not ranking, so the target must not move."""
    cloud, _, labels, table = grounded
    for mode in ("percentile", "minmax", "softmax", "none"):
        apc = lift(
            cloud,
            scene.attention_grid,
            dataclasses.replace(CFG.attention, normalization=mode),
            image_hw=scene.hw,
        )
        result = ground_target(apc, CFG.clustering, seed_percentile=95.0, exclude_mask=table)
        iou, _, _ = _iou(result.target.point_indices, labels, LABEL_SPHERE)
        assert iou > 0.95, f"{mode}: IoU {iou:.4f}"


# ---------------------------------------------------------------------------- support surface


def test_without_the_support_surface_removed_the_flood_becomes_the_target(grounded):
    """The reason the pipeline fits planes before grounding — and **the only** reason since T5e.

    The target stands on the table, so connectivity floods from it across the whole tabletop and out
    to every other object. Until T5e this test asserted `LOW_SCORE`: the flood was so un-compact that
    the score gate rejected it, which read as a second line of defence. It was never a reliable one —
    the same gate was rejecting 35 of 51 correctly-identified chunks on ep1807 — and with the gate off
    by default the flood is now *named*. `exclude_mask` is what prevents this, not the score.
    """
    _, apc, labels, _ = grounded
    result = ground_target(apc, CFG.clustering, seed_percentile=95.0)
    assert result.status is GroundingStatus.OK
    flooded = result.clusters[0]
    assert flooded.point_count > 10_000
    assert flooded.spatial_compactness < 0.01
    # It is the flood that was named, not a plausible object that happened to survive it.
    assert result.target.point_indices.size == flooded.point_count
    # The gate still exists for a caller that wants it: above the flood's score it rejects as before.
    gated = ground_target(
        apc, dataclasses.replace(CFG.clustering, target_score_threshold=0.25),
        seed_percentile=95.0,
    )
    assert gated.target is None and gated.status is GroundingStatus.LOW_SCORE


# ------------------------------------------------------------------------------ failure modes


def test_empty_cloud_reports_no_geometry():
    empty = AttentionPointCloud(PointCloud.empty(), np.zeros(0, np.float32))
    result = ground_target(empty, CFG.clustering)
    assert result.target is None and result.status is GroundingStatus.NO_GEOMETRY


def test_flat_attention_reports_no_attention(scene, grounded):
    """A uniform map says nothing about which object is the target, so no target is named."""
    cloud, _, _, table = grounded
    flat = lift(cloud, np.ones((16, 16), np.float32), CFG.attention, image_hw=scene.hw)
    result = ground_target(flat, CFG.clustering, seed_percentile=95.0, exclude_mask=table)
    assert result.target is None and result.status is GroundingStatus.NO_ATTENTION


def test_unreachable_seed_threshold_reports_no_seed(grounded):
    _, apc, _, table = grounded
    result = ground_target(apc, CFG.clustering, seed_threshold=2.0, exclude_mask=table)
    assert result.target is None and result.status is GroundingStatus.NO_SEED


def test_impossible_min_points_reports_no_cluster(grounded):
    _, apc, _, table = grounded
    result = ground_target(
        apc, dataclasses.replace(CFG.clustering, min_points=100_000),
        seed_percentile=95.0, exclude_mask=table,
    )
    assert result.target is None and result.status is GroundingStatus.NO_CLUSTER


def test_unreachable_score_threshold_reports_low_score_and_still_shows_its_work(grounded):
    """A rejected run must still report the clusters, so tuning and perception failures differ."""
    _, apc, _, table = grounded
    result = ground_target(
        apc, dataclasses.replace(CFG.clustering, target_score_threshold=0.99),
        seed_percentile=95.0, exclude_mask=table,
    )
    assert result.target is None and result.status is GroundingStatus.LOW_SCORE
    assert len(result.clusters) > 0 and 0.0 < result.best_score < 0.99


def test_failure_never_fabricates_a_target(grounded):
    """Across every failure mode: `target` is None, full stop."""
    _, apc, _, table = grounded
    variants = [
        dict(seed_threshold=2.0),
        dict(seed_percentile=95.0, config_override={"min_points": 100_000}),
        dict(seed_percentile=95.0, config_override={"target_score_threshold": 0.99}),
    ]
    for variant in variants:
        override = variant.pop("config_override", {})
        cfg = dataclasses.replace(CFG.clustering, **override)
        result = ground_target(apc, cfg, exclude_mask=table, **variant)
        assert result.target is None
        assert not result.status.ok


# ---------------------------------------------------------------------------- building blocks


def test_extract_seeds_by_percentile_takes_the_top_fraction():
    attention = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
    seeds = extract_seeds(attention, ClusteringConfig(), seed_percentile=95.0)
    assert 45 <= seeds.size <= 55
    assert attention[seeds].min() >= 0.94


def test_extract_seeds_by_absolute_threshold_can_return_nothing():
    """Which is what makes `NO_SEED` reachable — a percentile always returns something."""
    attention = np.linspace(0.0, 1.0, 100, dtype=np.float32)
    assert extract_seeds(attention, ClusteringConfig(), 95.0, seed_threshold=2.0).size == 0
    assert extract_seeds(attention, ClusteringConfig(), 95.0, seed_threshold=0.5).size == 50


def test_extract_seeds_cap_keeps_the_strongest_and_is_deterministic():
    rng = np.random.default_rng(0)
    attention = rng.random(10_000).astype(np.float32)
    cfg = dataclasses.replace(ClusteringConfig(), max_seed_points=100)
    a = extract_seeds(attention, cfg, seed_percentile=50.0)
    b = extract_seeds(attention, cfg, seed_percentile=50.0)
    assert a.size == 100 and np.array_equal(a, b)
    assert attention[a].min() > np.percentile(attention, 98.0)
    assert np.array_equal(a, np.sort(a))  # index order, not score order


def test_grow_region_reaches_a_connected_body_and_stops_at_a_gap():
    body = np.stack([np.arange(0, 0.20, 0.01), np.zeros(20), np.zeros(20)], axis=1)
    far = np.array([[1.0, 0.0, 0.0]])
    points = np.vstack([body, far])
    grown = grow_region(points, np.array([0]), radius=0.02)
    assert grown.size == 20 and 20 not in grown.tolist()


def test_grow_region_from_no_seeds_is_empty():
    assert grow_region(np.zeros((10, 3)), np.zeros(0, np.int64), 0.1).size == 0


def test_grow_region_respects_max_points():
    body = np.stack([np.arange(0, 1.0, 0.005), np.zeros(200), np.zeros(200)], axis=1)
    grown = grow_region(body, np.array([0]), radius=0.01, max_points=50)
    assert 0 < grown.size < 200


# ------------------------------------------------------------------------------- determinism


def test_grounding_is_bit_identical_across_runs(grounded):
    _, apc, _, table = grounded
    kw = dict(seed_percentile=95.0, exclude_mask=table)
    a = ground_target(apc, CFG.clustering, **kw)
    b = ground_target(apc, CFG.clustering, **kw)
    assert np.array_equal(a.target.point_indices, b.target.point_indices)
    assert a.best_score == b.best_score
    assert [c.target_score for c in a.clusters] == [c.target_score for c in b.clusters]


def test_clusters_are_returned_in_descending_score_order(grounded):
    _, apc, _, table = grounded
    scores = [c.target_score for c in
              ground_target(apc, CFG.clustering, seed_percentile=95.0, exclude_mask=table).clusters]
    assert scores == sorted(scores, reverse=True)


def test_score_weights_shift_the_ranking(grounded):
    """`w_attention` / `w_geometry` must actually do something, or they are not an ablation axis."""
    _, apc, _, table = grounded
    kw = dict(seed_percentile=95.0, exclude_mask=table)
    attention_only = ground_target(
        apc, dataclasses.replace(CFG.clustering, w_attention=1.0, w_geometry=0.0), **kw)
    geometry_only = ground_target(
        apc, dataclasses.replace(CFG.clustering, w_attention=0.0, w_geometry=1.0), **kw)
    assert attention_only.best_score != geometry_only.best_score
    assert attention_only.best_score == pytest.approx(attention_only.clusters[0].mean_attention)


# --------------------------------------------------------------- operating window and noise


def _scene_with_gap(box_y: float):
    """Default scene with the neighbouring box moved to a chosen surface gap."""
    from tests.ag3s.fixtures import DEFAULT_OBJECTS, GTObject, TABLE_Z

    sphere = DEFAULT_OBJECTS[0]
    box = GTObject(
        LABEL_BOX, "box", "box",
        np.array([0.62, box_y, TABLE_Z + 0.050]), np.array([0.045, 0.045, 0.050]),
    )
    gap = float(np.linalg.norm(sphere.center - box.center)) - sphere.extent[0] - box.extent[1]
    scene = make_scene(objects=(sphere, box, DEFAULT_OBJECTS[2]))
    cloud, _ = reconstruct(
        depth=scene.depth, camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam, config=CFG.pointcloud,
    )
    labels = scene.labels[cloud.uv[:, 1], cloud.uv[:, 0]]
    _, table = fit_support_surfaces(cloud, CFG.support_surface)
    apc = lift(cloud, scene.attention_grid, CFG.attention, image_hw=scene.hw)
    return gap, apc, labels, table


def test_the_two_methods_agree_when_run_at_the_same_radius():
    """`clustering.method` must isolate the density rule, not the neighbourhood radius.

    `region_growing_radius` defaults to None precisely so both methods use `eps`. With a number
    there instead, a "method comparison" silently becomes a radius comparison — which it did, and
    which this test exists to prevent recurring.
    """
    _, apc, labels, table = _scene_with_gap(0.105)
    kw = dict(seed_percentile=95.0, exclude_mask=table)
    results = {
        method: ground_target(apc, dataclasses.replace(CFG.clustering, method=method), **kw)
        for method in ("dbscan", "region_growing")
    }
    a, b = results["dbscan"], results["region_growing"]
    assert np.array_equal(a.target.point_indices, b.target.point_indices)


@pytest.mark.parametrize(
    "eps, separates",
    [(0.030, False), (0.020, True), (0.015, True)],
)
def test_eps_must_be_smaller_than_the_gap_it_has_to_resolve(eps, separates):
    """The real operating window, measured: `voxel_size < eps < smallest inter-object gap`.

    On a 2.0 cm gap, the default 3 cm `eps` bridges the objects and grounding returns both as one
    target. This is a genuine limitation of Euclidean connectivity, not a bug — and it is why `eps`
    is a config field rather than a constant.
    """
    gap, apc, labels, table = _scene_with_gap(0.105)
    assert 0.015 < gap < 0.025, f"fixture drifted: gap is {gap * 100:.1f} cm"
    result = ground_target(
        apc, dataclasses.replace(CFG.clustering, eps=eps), seed_percentile=95.0, exclude_mask=table
    )
    iou, _, _ = _iou(result.target.point_indices, labels, LABEL_SPHERE)
    assert (iou > 0.95) is separates, f"eps={eps}: IoU {iou:.4f}"


def test_eps_below_the_point_spacing_fragments_the_object():
    """The lower end of the window: below the voxel pitch, an object stops being connected to itself."""
    _, apc, labels, table = _scene_with_gap(0.105)
    result = ground_target(
        apc, dataclasses.replace(CFG.clustering, eps=0.008),
        seed_percentile=95.0, exclude_mask=table,
    )
    assert result.target is None and result.status is GroundingStatus.NO_CLUSTER


@pytest.mark.parametrize("noise_std", [0.001, 0.003, 0.005])
def test_grounding_degrades_gracefully_under_depth_noise(noise_std):
    """A RealSense D435 at 0.8 m is good to about 2-3 mm; this covers that and beyond."""
    scene = make_scene(noise_std=noise_std, seed=0)
    cloud, _ = reconstruct(
        depth=scene.depth, camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam, config=CFG.pointcloud,
    )
    labels = scene.labels[cloud.uv[:, 1], cloud.uv[:, 0]]
    # Noise widens the plane's own spread, so the inlier band has to widen with it.
    _, table = fit_support_surfaces(
        cloud,
        dataclasses.replace(
            CFG.support_surface,
            distance_threshold=max(CFG.support_surface.distance_threshold, 3.0 * noise_std),
        ),
    )
    apc = lift(cloud, scene.attention_grid, CFG.attention, image_hw=scene.hw)
    result = ground_target(apc, CFG.clustering, seed_percentile=95.0, exclude_mask=table)

    iou, _, _ = _iou(result.target.point_indices, labels, LABEL_SPHERE)
    error = float(np.linalg.norm(result.target.centroid - scene.visible_centroid(LABEL_SPHERE)))
    assert iou > 0.97, f"noise {noise_std * 1000:.0f} mm: IoU {iou:.4f}"
    assert error < 0.01, f"noise {noise_std * 1000:.0f} mm: centroid off {error * 100:.2f} cm"


# ------------------------------------------- the runner-up score the grasp latch reads (2026-09-25)
#
# `trajopt/safe_policy.py` passes `metrics["runner_up_score"]` into `GraspLatch.update`, which locks
# a target only on frames where `score >= score_ratio * max(runner_up, 1e-9)`. The key was missing
# from this stage's `metrics`, so `.get(..., 0.0)` handed the latch 0.0 on every frame: the threshold
# became 1.3e-9 and *every* grounded frame read as unambiguous. These tests hold the key in place and
# fix what its two branches mean.


def test_metrics_carry_the_runner_up_score_the_latch_reads(grounded):
    """The key exists, and it is the second-best candidate's score — not a placeholder."""
    _, apc, _, table = grounded
    result = ground_target(apc, CFG.clustering, seed_percentile=95.0, exclude_mask=table)

    assert len(result.clusters) > 1, "this fixture must have a runner-up for the test to mean anything"
    runner_up = result.target.metrics["runner_up_score"]
    assert runner_up == pytest.approx(result.clusters[1].target_score)
    # The vacuous-threshold regression, stated as the inequality that failed: a real second candidate
    # must not arrive at the latch as "no competitor".
    assert runner_up > 1e-9
    assert result.target.confidence == result.clusters[0].target_score > runner_up

    # The other side of the change: a genuinely dominant target must still read as confident, or the
    # fix would trade a useless test for a latch that never locks.
    from benchmark.trajopt.grasp_latch import LatchConfig

    assert result.target.confidence >= LatchConfig().score_ratio * runner_up


def test_a_single_candidate_reports_no_runner_up(grounded):
    """One cluster has no second score. 0.0 is the value, and it is unambiguous.

    A scored cluster can never *reach* 0.0 — `_score_cluster` mixes mean attention with two strictly
    positive `exp(-d/scale)` terms — so a reader can tell "there was no competitor" from "the
    competitor scored nothing". The 2D ablation is the cheapest real path with exactly one candidate:
    the seed set is taken whole, so there is nothing to come second.
    """
    _, apc, _, table = grounded
    result = ground_target(
        apc, dataclasses.replace(CFG.clustering, use_3d_connectivity=False),
        seed_percentile=95.0, exclude_mask=table,
    )
    assert result.status is GroundingStatus.OK and len(result.clusters) == 1
    assert result.target.metrics["runner_up_score"] == 0.0


def test_two_equally_good_candidates_are_reported_as_ambiguous():
    """What the fix buys: a contested frame now says so.

    Two identical blobs sit symmetrically about the attention peak, which is a lone point midway
    between them (too far from either to join a cluster, but still the peak). Same size, same
    attention, same distance — so their scores are within a few percent and neither dominates. Before
    the fix this frame reached the latch as `runner_up=0.0`, i.e. perfectly confident.
    """
    rng = np.random.default_rng(4)
    blob = rng.normal(0.0, 0.008, (200, 3))
    points = np.vstack([blob - [0.10, 0, 0], blob + [0.10, 0, 0], [[0.0, 0.0, 0.0]]])
    attention = np.concatenate([np.full(400, 0.9, np.float32), [1.0]])
    apc = AttentionPointCloud(PointCloud(points), attention)

    result = ground_target(apc, CFG.clustering, seed_threshold=0.5)
    assert result.status is GroundingStatus.OK and len(result.clusters) == 2

    score = result.target.confidence
    runner_up = result.target.metrics["runner_up_score"]
    assert runner_up > 0.0
    # Deliberately no hard-coded margin: the claim is that the two are close enough that the latch's
    # ratio test rejects the frame, which is exactly the comparison `_Confirm.update` makes.
    from benchmark.trajopt.grasp_latch import LatchConfig

    assert score < LatchConfig().score_ratio * runner_up, f"{score:.4f} vs {runner_up:.4f}"
