"""Where geometry used to disappear, and what happens there now.

AG3S had three paths that silently deleted physical geometry: clusters beyond `max_clusters`,
residue below `unknown_min_points`, and constraint spheres beyond `max_candidates`. Each removed
mass from the scene while the pipeline reported a normal frame, and a trajectory optimizer consuming
that frame would plan straight through what had been removed.

The contract now is that every valid non-support geometry element ends up as exactly one of:

1. a normal `CollisionCandidate`,
2. a conservative `OVERFLOW` aggregate that **contains** every point it stands for, or
3. an explicit `GEOMETRY_INCOMPLETE` / `ConstraintValidity.INCOMPLETE` frame.

The last one is the reason `ConstraintValidity` is a field on the output rather than a log line. An
empty `VALID` set means "AG3S looked and there is nothing there". An `INCOMPLETE` set means "AG3S
cannot tell you what is there". A caller that cannot distinguish those two will eventually treat the
second as the first, which is how a robot drives into something nobody modelled.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.ag3s.stages.collision_candidates import generate_candidates
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.constraints.constraint_builder import ConstraintBuilder
from AG3S.ag3s.stages.geometry import containment_report, contains, fit_primitive, inflate, to_spheres
from AG3S.ag3s.runtime.pipeline import AG3S
from AG3S.ag3s.stages.reconstruction import coverage_preserving_cap, voxel_downsample
from AG3S.ag3s.types import (
    ConstraintValidity,
    PipelineStatus,
    PointCloud,
    PrimitiveType,
    SourceType,
)
from tests.ag3s.fixtures import DEFAULT_ARM_Q, PlanarTwoLinkArm, make_scene

CFG = AG3SConfig()


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


@pytest.fixture(scope="module")
def scene(arm):
    return make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)


def _inputs(scene, **overrides):
    base = dict(
        attention_map=scene.attention_grid,
        depth=scene.depth,
        camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam,
        robot_state=DEFAULT_ARM_Q,
        phase="approach",
        image_hw=scene.hw,
    )
    base.update(overrides)
    return base


def _clutter(n_blobs=8, points=40, seed=0):
    rng = np.random.default_rng(seed)
    return PointCloud(np.vstack([
        rng.normal(0, 0.006, (points, 3)) + np.array([0.35 + 0.18 * i, 0.02 * i, 0.90])
        for i in range(n_blobs)
    ]))


# ------------------------------------------------------------------- the validity contract


def test_a_clean_frame_is_certified(scene, arm):
    constraint_set = AG3S(CFG, robot_model=arm).process(**_inputs(scene))
    assert constraint_set.validity is ConstraintValidity.VALID
    assert constraint_set.geometry_certified is True


def test_an_incomplete_frame_is_never_reported_as_a_clear_scene(scene, arm):
    """The single most important assertion in this file.

    `overflow_groups: 0` reproduces the old delete-the-residue behaviour. The frame still produces
    constraints and still looks perfectly ordinary from the outside — which is exactly why the
    validity field has to disagree, and why `status` must not be `OK`.
    """
    # `max_clusters: 1` guarantees residue on a scene with three objects, and `overflow_groups: 0`
    # leaves nothing to catch it. Both are configured deliberately — this is the shape of the bug the
    # validity field exists to expose, reproduced on purpose.
    config = CFG.with_overrides({"collision_candidate": {
        "max_clusters": 1, "overflow_groups": 0,
    }})
    constraint_set = AG3S(config, robot_model=arm).process(**_inputs(scene))
    assert constraint_set.metrics["n_unassigned_points"] > 0
    assert constraint_set.constraints is not None, (
        "the frame still produces constraints, which is exactly why validity has to disagree"
    )
    assert constraint_set.validity is ConstraintValidity.INCOMPLETE
    assert constraint_set.status is PipelineStatus.GEOMETRY_INCOMPLETE
    assert constraint_set.geometry_certified is False
    assert any("discarded" in note for note in constraint_set.notes)


def test_validity_only_ever_gets_worse_as_it_travels(scene, arm):
    """`ConstraintValidity.worst` is how the flag survives a pipeline of independent stages.

    Each stage may downgrade and none may upgrade. Without that, a late stage reporting "my part was
    fine" would erase an earlier stage's warning, and the erasure would be invisible.
    """
    assert ConstraintValidity.worst(
        ConstraintValidity.VALID, ConstraintValidity.INCOMPLETE, ConstraintValidity.DEGRADED
    ) is ConstraintValidity.INCOMPLETE
    assert ConstraintValidity.worst(ConstraintValidity.VALID) is ConstraintValidity.VALID
    assert ConstraintValidity.worst() is ConstraintValidity.VALID
    assert ConstraintValidity.DEGRADED.certified is False


# ------------------------------------------------------------------------ cluster overflow


def test_residue_below_unknown_min_points_is_aggregated_not_counted_away():
    """§26.7's first case. Three points are not a shape; they are still mass."""
    speck = np.array([[0.30, 0.30, 0.90], [0.305, 0.30, 0.90], [0.31, 0.30, 0.90]])
    blob = np.random.default_rng(0).normal(0, 0.01, (200, 3)) + np.array([0.6, 0.0, 0.9])
    cloud = PointCloud(np.vstack([blob, speck]))

    candidates, stats = generate_candidates(
        cloud, phase="transit",
        config=dataclasses.replace(CFG.collision_candidate, unknown_min_points=50),
        geometry_config=CFG.geometry, contact_config=CFG.contact,
    )
    assert stats["n_unassigned_points"] == 0
    assert stats["n_overflow"] >= 1
    assert any(
        contains(c.geometry[0], speck).all()
        for c in candidates
        if c.source_type is SourceType.OVERFLOW
    )


def test_every_point_survives_an_aggressive_max_clusters_cap():
    """The invariant stated as a measurement over the whole cloud rather than a spot check."""
    cloud = _clutter(n_blobs=8)
    candidates, stats = generate_candidates(
        cloud, phase="transit",
        config=dataclasses.replace(CFG.collision_candidate, max_clusters=2),
        geometry_config=CFG.geometry, contact_config=CFG.contact,
    )
    inside = np.zeros(len(cloud), bool)
    for candidate in candidates:
        if candidate.geometry:
            inside |= contains(candidate.geometry[0], cloud.points)
    assert inside.all(), f"{int((~inside).sum())} of {len(cloud)} points are unrepresented"
    assert stats["n_unassigned_points"] == 0


def test_overflow_aggregates_report_their_own_conservatism():
    """Containment first, excess volume second — and both visible.

    A containing shape that is ten times the volume of what it contains is safe and nearly useless.
    Recording the ratio is what lets that be noticed instead of merely survived.
    """
    cloud = _clutter(n_blobs=8)
    _, stats = generate_candidates(
        cloud, phase="transit",
        config=dataclasses.replace(CFG.collision_candidate, max_clusters=2),
        geometry_config=CFG.geometry, contact_config=CFG.contact,
    )
    assert stats["overflow_containment_rate"] == pytest.approx(1.0)
    assert stats["overflow_excess_volume_ratio"] > 0.0


def test_more_overflow_groups_mean_tighter_aggregates():
    """The knob does what it claims: splitting the residue reduces the excess volume.

    This is the trade the reserved slots exist to make. One aggregate is maximally conservative and
    blocks the workspace; several are still conservative and leave room to move.
    """
    cloud = _clutter(n_blobs=8)
    ratios = {}
    for groups in (1, 4):
        _, stats = generate_candidates(
            cloud, phase="transit",
            config=dataclasses.replace(
                CFG.collision_candidate, max_clusters=1, overflow_groups=groups
            ),
            geometry_config=CFG.geometry, contact_config=CFG.contact,
        )
        ratios[groups] = stats["overflow_excess_volume_ratio"]
    assert ratios[4] < ratios[1], f"splitting did not tighten the aggregate: {ratios}"


# ----------------------------------------------------------------------- constraint slots


def test_constraint_slot_overflow_contains_what_it_replaces(arm):
    """Slot overflow is the last place geometry could vanish, and the hardest to notice.

    A dropped sphere leaves the candidate list unchanged — the object is still listed, it simply has
    no row. So the check is against the parameter vector, comparing the same scene solved with
    enough slots against the same scene squeezed.
    """
    candidates, _ = generate_candidates(
        _clutter(n_blobs=10), phase="transit",
        config=dataclasses.replace(CFG.collision_candidate, max_clusters=64),
        geometry_config=CFG.geometry, contact_config=CFG.contact,
    )

    def _slots(max_candidates):
        cfg = CFG.with_overrides({"geometry": {"max_candidates": max_candidates}})
        spec = ConstraintBuilder(arm, cfg.constraint, cfg.geometry).build(candidates)
        lo_pos, _ = spec.layout["pos"]
        lo_r, _ = spec.layout["radius"]
        return spec, [
            (spec.parameter_values[lo_pos + 3 * j : lo_pos + 3 * j + 3],
             spec.parameter_values[lo_r + j])
            for j in range(spec.max_candidates)
            if spec.active_mask[j] > 0
        ]

    roomy_spec, roomy = _slots(64)
    tight_spec, tight = _slots(4)
    assert roomy_spec.layout["n_overflow_spheres"] == 0
    assert tight_spec.layout["n_overflow_spheres"] > 0

    for centre, radius in roomy:
        assert any(
            np.linalg.norm(centre - c) + radius <= r + 1e-9 for c, r in tight
        ), f"sphere at {centre} r={radius:.4f} lost its constraint when slots ran short"


def test_an_unfittable_support_surface_is_incomplete_not_merely_degraded(scene, arm):
    """A plane that does not fit is a plane the robot can drive through.

    There is no conservative aggregate for a half-space — you cannot coarsen a table into something
    that still bounds the workspace — so this is a genuine hole rather than a coarsening, and it is
    the one overflow path that reports `INCOMPLETE`.
    """
    from AG3S.ag3s.constraints.to_adapter import build_constraint_set
    from AG3S.ag3s.types import SupportSurface

    cfg = CFG.with_overrides({"geometry": {"max_candidates": 8}})
    builder = ConstraintBuilder(arm, cfg.constraint, cfg.geometry, max_support_surfaces=1)
    planes = [
        SupportSurface(id=i, normal=np.array([0.0, 0.0, 1.0]), offset=0.75 + 0.1 * i,
                       point_indices=np.arange(10), point_count=10, rms_error=1e-4)
        for i in range(3)
    ]
    constraint_set = build_constraint_set(
        builder, candidates=[], support_surfaces=planes, target=None,
        robot_state=DEFAULT_ARM_Q, phase="transit",
    )
    assert constraint_set.validity is ConstraintValidity.INCOMPLETE
    assert constraint_set.status is PipelineStatus.GEOMETRY_INCOMPLETE
    assert any("NOT constrained" in note for note in constraint_set.notes)


# ------------------------------------------------------------------------- the max_points cap


def test_the_default_cap_coarsens_rather_than_selecting():
    """`cap_strategy: "voxel"` keeps a representative in every occupied cell.

    Index selection samples the whole index range, which is a statement about the array and none at
    all about space — raster adjacency is not spatial adjacency, and a thin object seen edge-on can
    fall entirely between two selected indices. The measurable difference is coverage: after the
    voxel cap, no original point is further than the final voxel diagonal from a surviving point.
    """
    rng = np.random.default_rng(0)
    cloud = PointCloud(rng.uniform(0.0, 1.0, (20_000, 3)))
    config = dataclasses.replace(
        CFG.pointcloud, max_points=2_000, voxel_size=0.005, cap_strategy="voxel"
    )
    capped, stats = coverage_preserving_cap(cloud, config)

    assert stats["capped"] is True and stats["covered"] is True
    assert len(capped) <= config.max_points
    assert stats["voxel_size"] > config.voxel_size

    from scipy.spatial import cKDTree

    worst = cKDTree(capped.points).query(cloud.points)[0].max()
    diagonal = float(np.sqrt(3.0) * stats["voxel_size"])
    assert worst <= diagonal + 1e-9, (
        f"a point sits {worst:.4f} m from anything kept, beyond the {diagonal:.4f} m voxel diagonal"
    )


def test_the_stride_ablation_admits_it_makes_no_spatial_promise(scene, arm):
    """Kept for comparison, and honest about what it costs.

    The old selection is still reachable, but a cap that actually bound leaves geometry unaccounted
    for, so the frame is reported `INCOMPLETE` rather than normal. That is the difference between an
    ablation and a silent regression.
    """
    config = CFG.with_overrides({"pointcloud": {"cap_strategy": "stride", "max_points": 500}})
    constraint_set = AG3S(config, robot_model=arm).process(**_inputs(scene))
    assert constraint_set.validity is ConstraintValidity.INCOMPLETE
    assert constraint_set.status is PipelineStatus.GEOMETRY_INCOMPLETE
    assert any("spatial coverage" in note for note in constraint_set.notes)


def test_a_voxel_cap_that_binds_is_degraded_but_still_certified_geometry(scene, arm):
    """Coarser is not incomplete. The distinction is worth keeping sharp.

    Growing the voxel loses resolution, not regions: every occupied cell still contributes a point.
    So the frame is `DEGRADED` — usable, with a recorded loss of detail — rather than `INCOMPLETE`.
    """
    config = CFG.with_overrides({"pointcloud": {"max_points": 500}})
    constraint_set = AG3S(config, robot_model=arm).process(**_inputs(scene))
    assert constraint_set.validity is ConstraintValidity.DEGRADED
    assert constraint_set.status is PipelineStatus.DEGRADED
    assert constraint_set.metrics["final_voxel_size_m"] > CFG.pointcloud.voxel_size


# ------------------------------------------------------------- the uncertainty budget, once


@pytest.mark.parametrize("kind", list(PrimitiveType))
def test_inflation_grows_every_primitive_type_outward(kind):
    """One configured number has to mean the same physical thing whichever shape a cluster fits."""
    rng = np.random.default_rng(2)
    points = rng.normal(0, 0.02, (80, 3)) + np.array([0.5, 0.0, 0.9])
    tight = fit_primitive(points, kind)
    grown = inflate(tight, 0.03)
    assert grown.type is tight.type
    assert np.allclose(grown.center, tight.center)
    assert contains(grown, points).all()
    # Every surface moves out by the full amount; a capsule's half-length is not also extended,
    # since its caps are already hemispheres of the grown radius.
    assert grown.bounding_radius > tight.bounding_radius


def test_the_uncertainty_budget_is_applied_to_the_primitive_and_not_to_d_safe(arm):
    """§19's "do not apply the same uncertainty twice", checked where it would show up.

    The budget grows the fitted shape. If it also reached `d_safe`, the same centimetre would be
    counted in both terms of `r_robot + r_cand + d_safe` and every detour would silently double.
    """
    cloud = _clutter(n_blobs=3)
    margins = {}
    radii = {}
    for uncertainty in (0.0, 0.04):
        cfg = CFG.with_overrides({"geometry": {"perception_uncertainty": uncertainty}})
        candidates, _ = generate_candidates(
            cloud, phase="transit", config=cfg.collision_candidate,
            geometry_config=cfg.geometry, contact_config=cfg.contact,
        )
        spec = ConstraintBuilder(arm, cfg.constraint, cfg.geometry).build(candidates)
        lo_d, _ = spec.layout["d_safe"]
        lo_r, _ = spec.layout["radius"]
        margins[uncertainty] = spec.parameter_values[
            lo_d : lo_d + spec.n_robot_spheres * spec.max_candidates
        ].max()
        radii[uncertainty] = spec.parameter_values[lo_r : lo_r + spec.max_candidates].max()

    assert margins[0.04] == pytest.approx(margins[0.0]), "uncertainty leaked into d_safe"
    assert radii[0.04] == pytest.approx(radii[0.0] + 0.04, abs=1e-9), (
        "uncertainty did not reach the primitive radius"
    )


def test_containment_report_notices_a_shape_that_excludes_its_own_points():
    """The diagnostic has to be able to fail, or it is not measuring anything."""
    points = np.random.default_rng(3).normal(0, 0.05, (200, 3))
    honest = fit_primitive(points, PrimitiveType.SPHERE)
    assert containment_report(honest, points)["containment_rate"] == pytest.approx(1.0)

    shrunk = dataclasses.replace(honest, dimensions=honest.dimensions * 0.5)
    report = containment_report(shrunk, points)
    assert report["containment_rate"] < 1.0
    assert report["n_outside"] > 0


def test_a_capsule_sphere_chain_covers_the_capsule_it_replaces():
    """The other containment invariant, restated here because the two are easy to confuse.

    `to_spheres` is what the constraint form actually sees. Centres one radius apart do *not* cover
    the capsule — the surface point midway between two of them is sqrt(r^2 + (s/2)^2) away — so each
    sphere is inflated to exactly that. Under-covering the obstacle is the error direction that
    causes a collision rather than a detour.
    """
    from AG3S.ag3s.types import Primitive

    capsule = Primitive(
        type=PrimitiveType.CAPSULE,
        center=np.array([0.5, 0.0, 0.9]),
        dimensions=np.array([0.04, 0.30, 0.04]),
    )
    spheres = to_spheres(capsule, max_spheres=4)  # forced coarse, so the spacing genuinely binds
    rng = np.random.default_rng(4)
    t = rng.uniform(-0.30, 0.30, 2000)
    phi = rng.uniform(0, 2 * np.pi, 2000)
    surface = np.stack([
        np.full_like(t, 0.5) + 0.04 * np.cos(phi),
        0.04 * np.sin(phi),
        0.9 + t,
    ], axis=1)
    covered = np.zeros(surface.shape[0], bool)
    for centre, radius in spheres:
        covered |= np.linalg.norm(surface - centre, axis=1) <= radius + 1e-9
    assert covered.all(), f"{int((~covered).sum())} of {covered.size} surface points uncovered"
