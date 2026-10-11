"""Stage-5 tests: support surfaces, candidate generation, phase rules, and cross-frame identity.

The assertions that matter most are the ones about what is *kept*:

* the target is still in the candidate set, in every phase;
* points that classified as nothing still become `UNKNOWN_GEOMETRY`;
* an object's safety margin does not move when the phase changes — only the target's does.

Those three are the difference between this design and a filter that deletes what it is reaching
for, and each of them is a place a well-meaning refactor could quietly do the wrong thing.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from AG3S.ag3s.stages.attention_lifting import lift
from AG3S.ag3s.stages.collision_candidates import CandidateTracker, generate_candidates
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.stages.geometry import contains
from AG3S.ag3s.stages.reconstruction import reconstruct
from AG3S.ag3s.stages.support_surface import fit_plane_ransac, fit_support_surfaces
from AG3S.ag3s.stages.target_grounding import ground_target
from AG3S.ag3s.types import Phase, PointCloud, SourceType
from tests.ag3s.fixtures import (
    DEFAULT_OBJECTS,
    LABEL_BOX,
    LABEL_CYLINDER,
    LABEL_SPHERE,
    LABEL_TABLE,
    TABLE_Z,
    GTObject,
    make_scene,
)

CFG = AG3SConfig()


def _pipeline(scene, phase="approach", tracker=None, config=CFG):
    """Reconstruct -> planes -> lift -> ground -> candidates. The M3 stack, end to end."""
    cloud, _ = reconstruct(
        depth=scene.depth, camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam, config=config.pointcloud,
    )
    surfaces, support_mask = fit_support_surfaces(cloud, config.support_surface)
    apc = lift(cloud, scene.attention_grid, config.attention, image_hw=scene.hw)
    grounding = ground_target(
        apc, config.clustering,
        seed_percentile=config.attention.seed_percentile, exclude_mask=support_mask,
    )
    candidates, stats = generate_candidates(
        cloud, phase=phase, target=grounding.target,
        support_surfaces=surfaces, support_mask=support_mask,
        config=config.collision_candidate, geometry_config=config.geometry,
        contact_config=config.contact, tracker=tracker,
    )
    labels = scene.labels[cloud.uv[:, 1], cloud.uv[:, 0]]
    return cloud, surfaces, support_mask, grounding, candidates, stats, labels


@pytest.fixture(scope="module")
def scene():
    return make_scene()


@pytest.fixture(scope="module")
def run(scene):
    return _pipeline(scene)


def _by_type(candidates, source_type):
    return [c for c in candidates if c.source_type is source_type]


# ------------------------------------------------------------------------- support surfaces


def test_plane_fit_recovers_the_table_to_micrometres(scene, run):
    """Ground truth is a plane at exactly z = 0.75, so the fit has a number to be wrong by."""
    _, surfaces, _, _, _, _, _ = run
    assert len(surfaces) == 1
    plane = surfaces[0]
    assert float(np.arccos(min(1.0, abs(plane.normal @ [0, 0, 1])))) < 1e-4
    assert abs(plane.offset - scene.table_z) < 1e-4
    assert plane.rms_error < 1e-3


def test_plane_mask_covers_the_table_without_eating_the_objects(run):
    _, _, support_mask, _, _, _, labels = run
    table = labels == LABEL_TABLE
    true_positive = int((support_mask & table).sum())
    false_positive = int((support_mask & ~table).sum())
    assert true_positive / int(table.sum()) > 0.99  # recall
    assert true_positive / (true_positive + false_positive) > 0.99  # precision
    # The few false positives are object rims genuinely within `distance_threshold` of the plane.
    assert false_positive < 200


def test_every_object_survives_plane_removal(run):
    _, _, support_mask, _, _, _, labels = run
    residual = labels[~support_mask]
    for label in (LABEL_SPHERE, LABEL_BOX, LABEL_CYLINDER):
        kept = int((residual == label).sum())
        total = int((labels == label).sum())
        assert kept / total > 0.85, f"label {label}: only {kept}/{total} survived"


def test_the_size_gate_is_measured_against_the_original_cloud(scene):
    """Otherwise the second plane slot accepts a sliver.

    The table claims ~96% of the cloud, so a `min_inlier_ratio` scored against the *remainder*
    collapses from ~2,066 points to ~82 and a 661-point plane across the objects' tops qualifies as
    a "support surface" — removing real object geometry from clustering.
    """
    cloud, _ = reconstruct(
        depth=scene.depth, camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam, config=CFG.pointcloud,
    )
    surfaces, _ = fit_support_surfaces(
        cloud, dataclasses.replace(CFG.support_surface, max_planes=4)
    )
    assert len(surfaces) == 1, f"spurious planes: {[s.point_count for s in surfaces]}"


def test_half_space_conversion_matches_the_constraint_convention(run):
    """`n . p >= d + margin`, with the free side above the table."""
    _, surfaces, _, _, _, _, _ = run
    plane = surfaces[0]
    normal, offset = plane.to_halfspace()
    assert offset == pytest.approx(plane.offset + plane.safety_margin)
    assert np.linalg.norm(normal) == pytest.approx(1.0)
    above = plane.normal * (plane.offset + 0.5)
    assert plane.signed_distance(above[None, :])[0] > 0
    below = plane.normal * (plane.offset - 0.5)
    assert plane.signed_distance(below[None, :])[0] < 0


def test_plane_fit_is_deterministic(scene):
    cloud, _ = reconstruct(
        depth=scene.depth, camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam, config=CFG.pointcloud,
    )
    a = fit_plane_ransac(cloud.points, CFG.support_surface)
    b = fit_plane_ransac(cloud.points, CFG.support_surface)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1] and np.array_equal(a[2], b[2])


def test_plane_fitting_can_be_disabled(scene):
    cloud, _ = reconstruct(
        depth=scene.depth, camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam, config=CFG.pointcloud,
    )
    surfaces, mask = fit_support_surfaces(
        cloud, dataclasses.replace(CFG.support_surface, enabled=False)
    )
    assert surfaces == [] and not mask.any()


def test_a_normal_far_from_the_reference_is_rejected():
    """A wall must not be mistaken for a table."""
    rng = np.random.default_rng(0)
    wall = np.stack([np.zeros(3000), rng.uniform(-1, 1, 3000), rng.uniform(0, 1, 3000)], axis=1)
    normal, _, _ = fit_plane_ransac(wall, CFG.support_surface)
    assert normal is None


# --------------------------------------------------------------------- residual clustering


def test_residual_clustering_recovers_exactly_the_non_target_objects(run):
    """Two objects, each pure, plus the target kept separately."""
    _, _, _, _, candidates, stats, labels = run
    objects = _by_type(candidates, SourceType.OBJECT)
    assert len(objects) == 2
    for candidate in objects:
        composition = labels[candidate.point_indices]
        dominant = np.bincount(composition).argmax()
        purity = float((composition == dominant).mean())
        assert dominant in (LABEL_BOX, LABEL_CYLINDER)
        assert purity > 0.99, f"candidate {candidate.id} purity {purity:.4f}"
    assert {int(np.bincount(labels[c.point_indices]).argmax()) for c in objects} == {LABEL_BOX, LABEL_CYLINDER}
    assert stats["n_object"] == 2 and stats["n_target"] == 1 and stats["n_support"] == 1


def test_nothing_is_silently_discarded(run):
    """The conservative-mode promise, as a number."""
    _, _, _, _, _, stats, _ = run
    assert stats["n_unassigned_points"] == 0


def test_preserve_unknown_keeps_geometry_that_clusters_too_small():
    """A sparse body no object cluster claims must still become a candidate."""
    rng = np.random.default_rng(0)
    blob = rng.normal(0, 0.01, (200, 3)) + np.array([0.6, 0.0, 0.9])
    speck = np.array([[0.3, 0.3, 0.9], [0.305, 0.3, 0.9], [0.31, 0.3, 0.9]])
    cloud = PointCloud(np.vstack([blob, speck]))

    kept, stats_kept = generate_candidates(
        cloud, phase="transit", config=CFG.collision_candidate,
        geometry_config=CFG.geometry, contact_config=CFG.contact,
    )
    dropped, stats_dropped = generate_candidates(
        cloud, phase="transit",
        config=dataclasses.replace(CFG.collision_candidate, preserve_unknown=False),
        geometry_config=CFG.geometry, contact_config=CFG.contact,
    )
    assert stats_kept["n_unknown"] == 1 and stats_kept["n_unassigned_points"] == 0

    # With `preserve_unknown` off the speck no longer gets its own UNKNOWN_GEOMETRY candidate, but
    # it is still mass: it lands in a conservative overflow aggregate rather than being deleted.
    assert stats_dropped["n_unknown"] == 0
    assert stats_dropped["n_overflow"] >= 1
    assert stats_dropped["n_unassigned_points"] == 0, "geometry may be coarsened, never dropped"
    assert stats_dropped["n_overflow_points"] == 3
    covered = any(
        contains(c.geometry[0], speck).all() for c in dropped if c.source_type is SourceType.OVERFLOW
    )
    assert covered, "the overflow aggregate must contain every point it stands for"


def test_only_the_explicit_ablation_actually_discards_points():
    """`overflow_groups: 0` is the one setting that loses geometry, and it says so.

    It exists to reproduce the old behaviour for comparison. What makes it acceptable is that the
    loss is counted and surfaces as `GEOMETRY_INCOMPLETE` upstream, rather than being invisible.
    """
    speck = np.array([[0.3, 0.3, 0.9], [0.305, 0.3, 0.9], [0.31, 0.3, 0.9]])
    blob = np.random.default_rng(0).normal(0, 0.01, (200, 3)) + np.array([0.6, 0.0, 0.9])
    cloud = PointCloud(np.vstack([blob, speck]))
    _, stats = generate_candidates(
        cloud, phase="transit",
        config=dataclasses.replace(
            CFG.collision_candidate, preserve_unknown=False, overflow_groups=0
        ),
        geometry_config=CFG.geometry, contact_config=CFG.contact,
    )
    assert stats["n_overflow"] == 0
    assert stats["n_unassigned_points"] == 3


def test_unknown_candidates_get_a_usable_primitive():
    """A three-point speck still needs a radius a constraint can push against."""
    speck = np.array([[0.3, 0.3, 0.9], [0.302, 0.3, 0.9], [0.304, 0.3, 0.9]])
    candidates, _ = generate_candidates(
        PointCloud(speck), phase="transit", config=CFG.collision_candidate,
        geometry_config=CFG.geometry, contact_config=CFG.contact,
    )
    unknown = _by_type(candidates, SourceType.UNKNOWN_GEOMETRY)
    assert len(unknown) == 1
    assert unknown[0].geometry[0].dimensions[0] >= CFG.geometry.min_radius


def test_max_clusters_aggregates_the_smallest_rather_than_deleting_them():
    """The cap decides what gets its *own* candidate, not what exists.

    Two blobs win individual candidates because they carry the most mass; the other two are still
    represented, coarsely, by overflow aggregates. The assertion that matters is the last one — every
    original point is inside some candidate's primitive, which is what "no silent geometry loss"
    means when stated as a measurement rather than a slogan.
    """
    rng = np.random.default_rng(0)
    blobs = [rng.normal(0, 0.008, (60 + 40 * i, 3)) + np.array([0.4 + 0.3 * i, 0.0, 0.9])
             for i in range(4)]
    cloud = PointCloud(np.vstack(blobs))
    candidates, stats = generate_candidates(
        cloud, phase="transit",
        config=dataclasses.replace(CFG.collision_candidate, max_clusters=2),
        geometry_config=CFG.geometry, contact_config=CFG.contact,
    )

    objects = _by_type(candidates, SourceType.OBJECT)
    overflow = _by_type(candidates, SourceType.OVERFLOW)
    assert len(objects) == 2, "the cap still limits how many candidates get their own identity"
    assert sorted((c.point_count for c in objects), reverse=True)[0] >= 180
    assert overflow, "the clusters beyond the cap must still be represented"
    assert stats["n_unassigned_points"] == 0

    inside = np.zeros(len(cloud), bool)
    for candidate in candidates:
        if candidate.geometry:
            inside |= contains(candidate.geometry[0], cloud.points)
    assert inside.all(), f"{int((~inside).sum())} point(s) fell out of the candidate set entirely"


def test_overflow_aggregates_are_never_relaxed_and_carry_no_identity():
    """`OVERFLOW` is the least-understood geometry in the scene, so it gets the strictest treatment.

    Negative ids because there is nothing to track — an aggregate's membership changes with the
    scene, so carrying one across frames would give a TO a warm start for an object that does not
    exist. Zero confidence for the same reason: it was located, not identified.
    """
    rng = np.random.default_rng(1)
    blobs = [rng.normal(0, 0.008, (60, 3)) + np.array([0.4 + 0.3 * i, 0.0, 0.9]) for i in range(4)]
    candidates, _ = generate_candidates(
        PointCloud(np.vstack(blobs)), phase="grasp",
        config=dataclasses.replace(CFG.collision_candidate, max_clusters=1),
        geometry_config=CFG.geometry, contact_config=CFG.contact,
    )
    for candidate in _by_type(candidates, SourceType.OVERFLOW):
        assert candidate.collision_enabled is True
        assert candidate.contact_permission is False
        assert candidate.safety_margin == pytest.approx(CFG.geometry.safety_margin)
        assert candidate.id < 0 and candidate.confidence == 0.0


def test_support_surfaces_carry_no_primitives(run):
    """Their constraint is the exact half-space, not a pile of spheres."""
    _, _, _, _, candidates, _, _ = run
    planes = _by_type(candidates, SourceType.SUPPORT_SURFACE)
    assert len(planes) == 1
    assert planes[0].geometry == []
    assert planes[0].point_count > 10_000


# ------------------------------------------------------------------------------ the target


def test_the_target_is_never_removed_from_the_candidate_set(scene):
    """In every phase, including the one where it may be touched."""
    for phase in Phase:
        _, _, _, grounding, candidates, _, _ = _pipeline(scene, phase=phase)
        assert grounding.target is not None
        targets = _by_type(candidates, SourceType.TARGET)
        assert len(targets) == 1, f"phase {phase.value}: target missing from the candidate set"


@pytest.mark.parametrize(
    "phase, contact",
    [
        (Phase.TRANSIT, False),
        (Phase.APPROACH, False),
        (Phase.PRE_GRASP, False),
        (Phase.GRASP, True),
    ],
)
def test_phase_rules_follow_the_spec(scene, phase, contact):
    """`collision_enabled` is True in every phase, GRASP included.

    It used to be False at GRASP, which did not relax a margin — it deleted the target's constraint
    rows, leaving the left arm, the forearm and the torso free to pass through the object being
    grasped. Permitted contact is now a margin on named links (`clearance.ClearancePolicy`), so the
    row survives and still forbids penetration.
    """
    _, _, _, _, candidates, _, _ = _pipeline(scene, phase=phase)
    target = _by_type(candidates, SourceType.TARGET)[0]
    assert target.collision_enabled is True
    assert target.contact_permission is contact
    assert target.phase_rule == phase.value


def test_the_candidates_own_margin_is_the_conservative_one_in_every_phase(scene):
    """The scalar on the candidate is the margin for links with **no** contact authorization.

    An earlier version shrank this number through the phases, down to zero at GRASP. That reads as
    "the target is easier to approach now" and means "every part of the robot may touch the target
    now" — the torso included. Any relaxation belongs to a (robot sphere, slot) cell, never to the
    candidate as a whole, so this value must not move.
    """
    for phase in Phase:
        _, _, _, _, candidates, _, _ = _pipeline(scene, phase=phase)
        target = _by_type(candidates, SourceType.TARGET)[0]
        assert target.safety_margin == pytest.approx(CFG.geometry.safety_margin), (
            f"phase {phase.value} moved the target's conservative margin"
        )


def test_phase_never_touches_a_non_target_candidate(scene):
    """The safety property. Relaxing clearance on the target must not relax it on anything else."""
    for phase in Phase:
        _, _, _, _, candidates, _, _ = _pipeline(scene, phase=phase)
        for candidate in candidates:
            if candidate.source_type is SourceType.TARGET:
                continue
            if candidate.source_type is SourceType.SUPPORT_SURFACE:
                assert candidate.safety_margin == pytest.approx(CFG.support_surface.safety_margin)
            else:
                assert candidate.safety_margin == pytest.approx(CFG.geometry.safety_margin)
            assert candidate.collision_enabled is True
            assert candidate.contact_permission is False


def test_phase_aware_off_pins_the_target_to_full_clearance(scene):
    """The ablation: without phase awareness, grasping is impossible — which is the point."""
    config = CFG.with_overrides({"contact": {"phase_aware": False}})
    for phase in Phase:
        _, _, _, _, candidates, _, _ = _pipeline(scene, phase=phase, config=config)
        target = _by_type(candidates, SourceType.TARGET)[0]
        assert target.safety_margin == pytest.approx(CFG.geometry.safety_margin)
        assert target.collision_enabled is True and target.contact_permission is False


def test_candidates_never_see_attention():
    """A signature check, deliberately.

    `generate_candidates` takes no attention argument. Geometry decides what can collide; if this
    ever gains an attention parameter, the core principle has been broken.
    """
    import inspect

    parameters = set(inspect.signature(generate_candidates).parameters)
    assert not any("attention" in name for name in parameters), parameters


# --------------------------------------------------------------------- cross-frame identity


def _moving_scene(frame: int):
    cylinder = GTObject(
        LABEL_CYLINDER, "cyl", "cylinder",
        np.array([0.86 - 0.02 * frame, -0.19, TABLE_Z + 0.060]), np.array([0.035, 0.035, 0.060]),
    )
    return make_scene(objects=(DEFAULT_OBJECTS[0], DEFAULT_OBJECTS[1], cylinder))


def test_ids_survive_a_moving_object_and_track_age_accumulates():
    """TO warm-starting depends on slot *k* meaning the same object next frame."""
    tracker = CandidateTracker(CFG.collision_candidate)
    seen: list[set[int]] = []
    for frame in range(5):
        _, _, _, _, candidates, _, _ = _pipeline(_moving_scene(frame), tracker=tracker)
        movable = [c for c in candidates if c.source_type is not SourceType.SUPPORT_SURFACE]
        seen.append({c.id for c in movable})
        if frame > 0:
            assert seen[frame] == seen[0], f"ids changed at frame {frame}: {seen[frame]} vs {seen[0]}"
            assert all(c.track_age == frame for c in movable)


def test_a_track_survives_a_short_disappearance():
    """An object occluded for a frame must come back with its own id, not a fresh one."""
    tracker = CandidateTracker(CFG.collision_candidate)
    assert tracker.update([np.zeros(3), np.array([1.0, 0.0, 0.0])]) == [0, 1]
    assert tracker.update([np.array([0.01, 0.0, 0.0])]) == [0]
    assert tracker.update([np.array([0.02, 0.0, 0.0])]) == [0]
    assert tracker.update([np.array([0.03, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]) == [0, 1]


def test_a_track_is_retired_after_too_many_missed_frames():
    config = dataclasses.replace(CFG.collision_candidate, track_max_missed=1)
    tracker = CandidateTracker(config)
    assert tracker.update([np.zeros(3), np.array([1.0, 0.0, 0.0])]) == [0, 1]
    for _ in range(3):
        tracker.update([np.zeros(3)])
    assert tracker.update([np.zeros(3), np.array([1.0, 0.0, 0.0])]) == [0, 2]


def test_adjacent_objects_do_not_swap_ids():
    """Global assignment, not greedy nearest-first.

    Greedy would let whichever detection is processed first claim the shared nearest track, forcing
    the other onto a new id — so two adjacent objects could trade identities every frame.
    """
    tracker = CandidateTracker(CFG.collision_candidate)
    assert tracker.update([np.zeros(3), np.array([0.05, 0.0, 0.0])]) == [0, 1]
    assert tracker.update([np.array([0.05, 0.0, 0.0]), np.zeros(3)]) == [1, 0]


def test_a_jump_beyond_the_association_radius_starts_a_new_track():
    tracker = CandidateTracker(CFG.collision_candidate)
    assert tracker.update([np.zeros(3)]) == [0]
    assert tracker.update([np.array([1.0, 0.0, 0.0])]) == [1]


def test_tracker_handles_empty_frames():
    tracker = CandidateTracker(CFG.collision_candidate)
    assert tracker.update([]) == []
    assert tracker.update([np.zeros(3)]) == [0]


# ------------------------------------------------------------------------------ determinism


def test_candidate_generation_is_deterministic(scene):
    a = _pipeline(scene)[4]
    b = _pipeline(scene)[4]
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.point_count for c in a] == [c.point_count for c in b]
    for x, y in zip(a, b):
        assert np.array_equal(x.centroid, y.centroid)


def test_candidate_fields_are_all_populated(run):
    _, _, _, _, candidates, _, _ = run
    for candidate in candidates:
        assert isinstance(candidate.source_type, SourceType)
        assert candidate.semantic_role
        assert candidate.pose.shape == (4, 4)
        assert candidate.dimensions.shape == (3,)
        assert 0.0 <= candidate.confidence <= 1.0
        assert candidate.safety_margin >= 0.0
        assert isinstance(candidate.collision_enabled, bool)
        assert isinstance(candidate.contact_permission, bool)
        assert candidate.phase_rule
        assert candidate.timestamp > 0.0
        assert candidate.frame_id == "base"
        assert candidate.track_age >= 0
