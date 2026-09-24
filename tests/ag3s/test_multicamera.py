"""Three cameras, one scene — registration, provenance, freshness, and the fused attention.

The failure this replaces was not subtle in its consequences and was invisible in its symptoms:
`experiments/rby1_transport.py` ran one `AG3S` per camera, so the crate was candidate `0` in the head
view and the table was candidate `0` in a wrist view, and nothing downstream could tell. Every
individual result looked fine.

Two properties carry most of the weight here.

**Capture-time kinematics.** A wrist camera rides the arm, so its cloud belongs at the pose the arm
held when the shutter opened. Transforming all three views with one current `q` misplaces the wrist
clouds *and* misaligns the self-filter, so the arm's own points survive to cluster into a phantom
obstacle welded to the gripper. The tests below drive the three cameras from genuinely different
configurations, because with identical states the bug is unobservable.

**Provenance survives fusion.** A fused point keeps every observation that produced it — camera,
pixel, timestamp, attention. Without that, the attention `max` would be a max over an anonymous pile,
and a candidate could not be traced back to the view that produced it.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmark.ag3s.stages.attention_lifting import normalize_attention
from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.runtime.multiview import check_freshness, fuse, fuse_observations, process_observation
from benchmark.ag3s.runtime.pipeline import AG3S
from benchmark.ag3s.types import (
    CameraID,
    CameraObservation,
    ConstraintValidity,
    FusedPointCloud,
    PipelineStatus,
    SourceType,
)
from tests.ag3s.fixtures import (
    DEFAULT_ARM_Q,
    LABEL_SPHERE,
    PlanarTwoLinkArm,
    make_scene,
    make_three_camera_scene,
)

CFG = AG3SConfig().with_overrides({
    "timing": {"expected_cameras": ["head", "left_wrist", "right_wrist"]},
})

#: A second arm configuration, far enough from `DEFAULT_ARM_Q` that a wrist camera placed with the
#: wrong one lands somewhere obviously different. 0.35 rad on a 0.30 m link is over 10 cm.
MOVED_ARM_Q = np.array([-0.25, -0.85])


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


@pytest.fixture(scope="module")
def scene(arm):
    return make_three_camera_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)


# --------------------------------------------------------------------------- registration


def test_the_three_views_register_into_one_base_frame_scene(arm, scene):
    """The whole scene must land on the table, whichever camera saw it.

    Registration error would show up as the fused cloud straddling several heights — three tables at
    three offsets — so the test is a spread, not a mean. A mean would be right even if the views
    disagreed symmetrically.
    """
    fusion = fuse_observations(scene.observations, CFG, robot_model=arm)
    assert len(fusion.per_camera) == 3
    assert all(len(r.cloud) > 100 for r in fusion.per_camera)

    table_z = scene.views[CameraID.HEAD].table_z
    for result in fusion.per_camera:
        on_table = result.cloud.points[
            np.abs(result.cloud.points[:, 2] - table_z) < 0.02
        ]
        assert on_table.shape[0] > 50, f"{result.camera_id.value} did not see the table"
        assert abs(float(on_table[:, 2].mean()) - table_z) < 2e-3, (
            f"{result.camera_id.value}'s table sits at the wrong height"
        )


def test_a_wrist_camera_is_placed_by_forward_kinematics_at_its_own_capture_state(arm):
    """§26.4's core. The same hand-eye calibration must give different poses at different `q`.

    Both halves matter. The pose at the capture state has to be *exact* — that is the calibration
    round-tripping — and the pose at another state has to be *different*, or the FK route would be an
    elaborate way of storing a constant and the whole timestamp argument would be moot.
    """
    scene = make_three_camera_scene(
        robot_model=arm, robot_state=DEFAULT_ARM_Q,
        wrist_states=[DEFAULT_ARM_Q, MOVED_ARM_Q, DEFAULT_ARM_Q],
    )
    wrist = scene.observations[1]
    assert wrist.T_base_cam is None and wrist.mount_link is not None

    resolved = wrist.resolve_T_base_cam(arm)
    expected = scene.views[CameraID.LEFT_WRIST].T_base_cam
    assert np.abs(resolved - expected).max() < 1e-12

    stale = dataclasses.replace(wrist, robot_state=DEFAULT_ARM_Q)
    displacement = float(np.linalg.norm(stale.resolve_T_base_cam(arm)[:3, 3] - resolved[:3, 3]))
    assert displacement > 0.05, (
        f"the wrong robot state moved the camera only {displacement * 1000:.1f} mm; "
        "this fixture cannot detect a q_now regression"
    )


def test_one_q_now_for_every_camera_visibly_misplaces_the_wrist_clouds(arm):
    """The regression this design exists to prevent, measured rather than asserted by construction.

    Reprocessing the wrist observations with the head's state — the `q_now` shortcut — moves their
    geometry by centimetres. If a future refactor reintroduced a shared state, this is the number
    that would go to zero.
    """
    scene = make_three_camera_scene(
        robot_model=arm, robot_state=DEFAULT_ARM_Q,
        wrist_states=[DEFAULT_ARM_Q, MOVED_ARM_Q, MOVED_ARM_Q],
    )
    correct = fuse_observations(scene.observations, CFG, robot_model=arm)
    shared = fuse_observations(
        [dataclasses.replace(obs, robot_state=DEFAULT_ARM_Q) for obs in scene.observations],
        CFG, robot_model=arm,
    )

    for right, wrong in zip(correct.per_camera[1:], shared.per_camera[1:]):
        shift = float(np.linalg.norm(right.cloud.points.mean(axis=0) - wrong.cloud.points.mean(axis=0)))
        assert shift > 0.02, (
            f"{right.camera_id.value}: a shared q_now moved its cloud only {shift * 1000:.1f} mm"
        )


def test_a_mount_link_camera_without_a_robot_model_says_so(arm, scene):
    """Failing loudly beats guessing an identity transform and producing a plausible wrong scene."""
    wrist = scene.observations[1]
    with pytest.raises(ValueError, match="needs a robot model"):
        wrist.resolve_T_base_cam(None)
    with pytest.raises(ValueError, match="robot_state"):
        dataclasses.replace(wrist, robot_state=None).resolve_T_base_cam(arm)


def test_an_observation_must_carry_enough_to_be_placed():
    with pytest.raises(ValueError, match="exactly one of depth"):
        CameraObservation(camera_id=CameraID.HEAD, T_base_cam=np.eye(4))
    with pytest.raises(ValueError, match="mount_link together with T_link_cam"):
        CameraObservation(camera_id=CameraID.HEAD, depth=np.ones((4, 4)))


# ------------------------------------------------------------------------------ provenance


def test_a_fused_point_remembers_every_camera_that_saw_it(arm, scene):
    """§26.5. Two cameras in one voxel must become one geometry element with two observations."""
    fusion = fuse_observations(scene.observations, CFG, robot_model=arm)
    fused = fusion.cloud

    assert fused.n_observations == fusion.metrics["n_points_before_fusion"]
    assert len(fused) < fused.n_observations, "the views did not overlap; nothing was fused"

    multi = [i for i in range(len(fused)) if len(fused.cameras_of(i)) > 1]
    assert multi, "no fused point was seen by more than one camera"
    for i in multi[:20]:
        cameras = fused.cameras_of(i)
        assert cameras <= set(CameraID)
        assert len(cameras) >= 2


def test_pixel_and_timestamp_provenance_survive_fusion(arm, scene):
    """`(u, v)` and the capture time are what let a fused point be traced back to an image."""
    fusion = fuse_observations(scene.observations, CFG, robot_model=arm)
    fused = fusion.cloud
    stamps = {obs.camera_id: obs.timestamp for obs in scene.observations}

    assert fused.obs_uv.shape == (fused.n_observations, 2)
    assert (fused.obs_uv >= 0).all(), "a pixel index was lost during fusion"
    for i in range(0, len(fused), max(1, len(fused) // 50)):
        block = fused.observation_slice(i)
        for camera_index, time in zip(fused.obs_camera[block], fused.obs_time[block]):
            assert time == pytest.approx(stamps[fused.cameras[int(camera_index)]])


def test_the_representative_is_an_observed_point_not_a_centroid(arm, scene):
    """A synthetic centroid belongs to no pixel, so it has no attention and no camera.

    Averaging would destroy the correspondence at precisely the moment several cameras agree — the
    moment the correspondence is most informative.
    """
    fusion = fuse_observations(scene.observations, CFG, robot_model=arm)
    fused = fusion.cloud
    originals = np.vstack([r.cloud.points for r in fusion.per_camera if len(r.cloud)])

    from scipy.spatial import cKDTree

    distances = cKDTree(originals).query(fused.points)[0]
    assert distances.max() < 1e-12, "a fused point is not one of the observed points"


def test_fusion_does_not_lose_coverage(arm, scene):
    """Fusion may merge duplicates; it may not create empty space.

    Stated as a distance rather than a count, because a count would pass even if fusion had kept
    exactly the wrong points. Nothing observed may end up further than one fusion voxel diagonal
    from something kept.
    """
    fusion = fuse_observations(scene.observations, CFG, robot_model=arm)
    originals = np.vstack([r.cloud.points for r in fusion.per_camera if len(r.cloud)])

    from scipy.spatial import cKDTree

    worst = cKDTree(fusion.cloud.points).query(originals)[0].max()
    diagonal = float(np.sqrt(3.0) * CFG.timing.fusion_voxel_size)
    assert worst <= diagonal + 1e-9, f"{worst:.4f} m gap exceeds the {diagonal:.4f} m voxel diagonal"


def test_selecting_a_subset_carries_its_observations_with_it(arm, scene):
    """`select` has to renumber the CSR block, not merely slice the points.

    The representative index points into the observation table, so a naive slice would leave it
    pointing at a stranger's pixel — a provenance error that produces no exception and no visibly
    wrong geometry.
    """
    fused = fuse_observations(scene.observations, CFG, robot_model=arm).cloud
    keep = np.arange(0, len(fused), 3)
    subset = fused.select(keep)

    assert len(subset) == keep.size
    assert np.allclose(subset.points, fused.points[keep])
    for new_index, old_index in enumerate(keep[:30]):
        assert subset.cameras_of(new_index) == fused.cameras_of(int(old_index))
        assert np.array_equal(
            subset.obs_uv[subset.representative[new_index]],
            fused.obs_uv[fused.representative[int(old_index)]],
        )


# -------------------------------------------------------------------------------- freshness


def test_a_dropped_camera_is_degraded_not_normal(arm, scene):
    """§12. A partially observed scene must never look like a fully observed one."""
    fusion = fuse_observations(scene.observations[:2], CFG, robot_model=arm)
    assert fusion.validity is ConstraintValidity.DEGRADED
    assert fusion.metrics["missing_cameras"] == ["right_wrist"]
    assert any("did not report" in note for note in fusion.notes)

    # And with every camera present, nothing is flagged.
    full = fuse_observations(scene.observations, CFG, robot_model=arm)
    assert full.metrics["missing_cameras"] == []
    assert full.validity is ConstraintValidity.VALID


def test_a_stale_robot_state_is_reported_rather_than_absorbed(arm, scene):
    """A stale state puts a fresh image at the wrong pose — plausible geometry, wrong place."""
    stale = list(scene.observations)
    stale[1] = dataclasses.replace(stale[1], state_timestamp=stale[1].timestamp - 0.5)
    validity, notes, metrics = check_freshness(stale, CFG)

    assert validity is ConstraintValidity.DEGRADED
    assert metrics["n_stale_state"] == 1
    assert any("robot state is" in note for note in notes)


def test_camera_skew_beyond_the_limit_is_reported(arm, scene):
    skewed = list(scene.observations)
    skewed[2] = dataclasses.replace(skewed[2], timestamp=skewed[2].timestamp + 0.4)
    validity, notes, metrics = check_freshness(skewed, CFG)

    assert validity is ConstraintValidity.DEGRADED
    assert metrics["camera_skew_sec"] > CFG.timing.max_camera_skew_sec
    assert any("span" in note for note in notes)


def test_no_observations_at_all_is_incomplete(arm):
    validity, notes, metrics = check_freshness([], CFG)
    assert validity is ConstraintValidity.INCOMPLETE
    assert metrics["n_cameras"] == 0


def test_one_valid_camera_still_produces_a_conservative_scene(arm, scene):
    """§12's other half: degraded coverage is not a reason to emit nothing.

    With one camera left the geometry it *did* see is still real and still has to be avoided. What
    changes is the certification, not the constraints.
    """
    constraint_set = AG3S(CFG, robot_model=arm).process_multi(scene.observations[:1])
    assert constraint_set.candidates, "a surviving camera must still produce candidates"
    assert constraint_set.validity is ConstraintValidity.DEGRADED
    assert constraint_set.geometry_certified is False


# ------------------------------------------------------------- multi-view attention lifting


def test_attention_fuses_by_max_over_the_contributing_observations(arm, scene):
    """§13.1's required default, checked against the CSR block rather than re-derived.

    **F2 changed which array the max runs on** (`AG3S_REVIEW_LOG.md` Step 5). The rule is still max
    and never mean; what moved is normalization, which now happens *once after* fusion instead of
    per camera. So `FusionResult.attention` is no longer the max of the normalized provenance block
    — `FusedPointCloud.fused_attention()` still is, and stays available as provenance. Asserting the
    old identity here is what would let the pre-F2 behaviour come back unnoticed.
    """
    fusion = fuse_observations(scene.observations, CFG, robot_model=arm)
    fused = fusion.cloud
    assert fusion.attention.shape == (len(fused),)
    assert fusion.raw_attention.shape == (len(fused),)

    # The provenance view is unchanged: max, not mean, over the per-camera normalized values.
    provenance = fused.fused_attention()
    for i in range(0, len(fused), max(1, len(fused) // 200)):
        block = fused.obs_attention[fused.observation_slice(i)]
        assert provenance[i] == pytest.approx(block.max())

    # F2: what the pipeline reads is the fused *raw* signal, normalized exactly once.
    assert np.array_equal(
        fusion.attention, normalize_attention(fusion.raw_attention, CFG.attention)
    )

    # Normalization is monotone, so fusing by max cannot reorder points. This is the property that
    # makes `seed_percentile` and `attention.normalization` independent knobs.
    order = np.argsort(fusion.raw_attention, kind="stable")
    assert np.all(np.diff(fusion.attention[order]) >= -1e-6)


def test_a_camera_without_attention_keeps_all_of_its_geometry(arm):
    """§13.2. Zero attention is "no opinion", not "not there"."""
    scene = make_three_camera_scene(
        robot_model=arm, robot_state=DEFAULT_ARM_Q, with_attention=(True, False, False)
    )
    fusion = fuse_observations(scene.observations, CFG, robot_model=arm)

    for result in fusion.per_camera[1:]:
        assert len(result.cloud) > 100, "a camera lost its geometry for lacking an attention map"
        assert float(result.attention.max()) == 0.0
    assert any("no attention map" in note for note in fusion.notes)
    assert fusion.metrics["cameras_with_attention"] == ["head"]


def test_attention_lifting_does_not_reduce_geometry_coverage(arm):
    """§13.3. The point count must not depend on whether attention was supplied."""
    with_attention = make_three_camera_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    without = make_three_camera_scene(
        robot_model=arm, robot_state=DEFAULT_ARM_Q, with_attention=(False, False, False)
    )
    a = fuse_observations(with_attention.observations, CFG, robot_model=arm)
    b = fuse_observations(without.observations, CFG, robot_model=arm)
    assert len(a.cloud) == len(b.cloud)
    assert np.allclose(a.cloud.points, b.cloud.points)


def test_a_secondary_camera_can_carry_the_attention_peak(arm):
    """Max, not mean, so a close wrist view is not averaged away by two distant ones.

    Constructed rather than hoped for: only the right wrist gets an attention map, and its peak must
    survive into the fused signal at full strength.
    """
    scene = make_three_camera_scene(
        robot_model=arm, robot_state=DEFAULT_ARM_Q, with_attention=(False, False, True)
    )
    fusion = fuse_observations(scene.observations, CFG, robot_model=arm)
    wrist = fusion.per_camera[2]
    assert float(fusion.attention.max()) == pytest.approx(float(wrist.attention.max()))


def test_every_camera_uses_the_same_normalization(arm, scene):
    """§13's "do not silently use different normalization schemes per camera".

    There is one `config.attention` and `process_observation` takes no per-camera override, so this
    is a structural property. The test pins the signature so it stays one.
    """
    import inspect

    signature = inspect.signature(process_observation)
    assert "config" in signature.parameters
    assert not any(
        "attention_config" in name or "normalization" in name for name in signature.parameters
    ), "a per-camera normalization override appeared; all cameras must share one config"

    softmax = CFG.with_overrides({"attention": {"normalization": "softmax"}})
    fusion = fuse_observations(scene.observations, softmax, robot_model=arm)
    # Under softmax every camera's map sums to one over its own points, so no camera can dominate by
    # scaling alone — the property that would break if normalizations diverged.
    for result in fusion.per_camera:
        assert 0.0 <= float(result.attention.min()) <= float(result.attention.max()) <= 1.0


# ---------------------------------------------------------------------- end to end, one scene


def test_process_multi_produces_one_scene_with_one_identity_per_object(arm, scene):
    """The point of all of it: the crate is one candidate, not three.

    Running three pipelines gave three id spaces in which `0` meant a different object each time.
    Here there is one tracker over one fused cloud, so an object has one id.
    """
    ag3s = AG3S(CFG, robot_model=arm)
    constraint_set = ag3s.process_multi(scene.observations, phase="approach")

    assert constraint_set.status in (PipelineStatus.OK, PipelineStatus.NO_TARGET,
                                     PipelineStatus.DEGRADED)
    ids = [c.id for c in constraint_set.candidates]
    assert len(ids) == len(set(ids)), f"duplicate candidate ids in one fused scene: {ids}"
    assert constraint_set.metrics["n_cameras"] == 3
    assert constraint_set.metrics["n_points_fused"] > 0
    assert constraint_set.metrics["n_points_fused"] < constraint_set.metrics["n_points_before_fusion"]


def test_process_multi_takes_the_constraint_state_from_the_newest_observation(arm):
    """A TO plans from a configuration; the freshest one AG3S was told about is the honest choice."""
    scene = make_three_camera_scene(
        robot_model=arm, robot_state=DEFAULT_ARM_Q,
        timestamps=(1.00, 1.02, 1.09),
        wrist_states=[DEFAULT_ARM_Q, DEFAULT_ARM_Q, MOVED_ARM_Q],
    )
    constraint_set = AG3S(CFG, robot_model=arm).process_multi(scene.observations)
    assert np.allclose(constraint_set.robot_state, MOVED_ARM_Q)


def test_the_single_camera_api_still_works_unchanged(arm):
    """`process()` is the compatibility path and must not have acquired multi-camera requirements."""
    single = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    constraint_set = AG3S(AG3SConfig(), robot_model=arm).process(
        attention_map=single.attention_grid,
        depth=single.depth,
        camera_intrinsics=single.camera_intrinsics,
        T_base_cam=single.T_base_cam,
        robot_state=DEFAULT_ARM_Q,
        phase="approach",
        image_hw=single.hw,
    )
    assert constraint_set.candidates
    assert constraint_set.validity is ConstraintValidity.VALID


def test_fusion_of_nothing_is_empty_rather_than_an_exception():
    empty, attention, raw = fuse([], voxel_size=0.01)
    assert isinstance(empty, FusedPointCloud) and empty.is_empty
    assert attention.size == 0 and raw.size == 0
    assert empty.as_pointcloud().is_empty


def test_the_target_records_which_cameras_saw_it(arm, scene):
    """§14. Minimal provenance: which cameras contributed points, and which carried the peak.

    Kept small on purpose — the fused cloud already stores a per-point camera list, and copying it
    onto the one object most likely to be logged every frame would double its memory for nothing.
    """
    constraint_set = AG3S(CFG, robot_model=arm).process_multi(scene.observations, phase="approach")
    target = constraint_set.target
    if target is None:
        pytest.skip("grounding declined on this scene; provenance has nothing to record")

    assert target.seed_camera in set(CameraID)
    assert target.supporting_cameras, "a grounded target must name at least one contributing camera"
    assert target.seed_camera in target.supporting_cameras
    assert set(target.supporting_cameras) <= {obs.camera_id for obs in scene.observations}


def test_a_single_camera_frame_leaves_the_provenance_fields_empty(arm):
    """The fields are for the fused path. A single-camera frame has nothing to disambiguate."""
    single = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    constraint_set = AG3S(AG3SConfig(), robot_model=arm).process(
        attention_map=single.attention_grid, depth=single.depth,
        camera_intrinsics=single.camera_intrinsics, T_base_cam=single.T_base_cam,
        robot_state=DEFAULT_ARM_Q, phase="approach", image_hw=single.hw,
    )
    if constraint_set.target is not None:
        assert constraint_set.target.seed_camera is None
        assert constraint_set.target.supporting_cameras == ()
