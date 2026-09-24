"""End-to-end pipeline tests: orchestration, degradation, ablation reachability, profiling.

`pipeline.py` is supposed to contain no algorithm, so what is tested here is *composition*: that the
stages run in the order that makes them correct, that each degraded input produces the documented
status rather than an exception, and that every ablation the spec requires can be reached by
config alone and visibly changes the output.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.runtime.pipeline import AG3S
from benchmark.ag3s.runtime.profiler import STAGE_ORDER, StageProfiler
from benchmark.ag3s.constraints.to_adapter import summary
from benchmark.ag3s.types import (
    GroundingStatus,
    Phase,
    PipelineStatus,
    PrimitiveType,
    SourceType,
)
from tests.ag3s.fixtures import (
    DEFAULT_ARM_Q,
    LABEL_BOX,
    LABEL_CYLINDER,
    LABEL_SPHERE,
    PlanarTwoLinkArm,
    make_scene,
)

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


def _run(scene, arm, config=CFG, **overrides):
    ag3s = AG3S(config, robot_model=arm)
    return ag3s, ag3s.process(**_inputs(scene, **overrides))


# ------------------------------------------------------------------------------ the happy path


def test_the_full_api_shape_from_the_spec(scene, arm):
    """The §4 call signature, and every field of `CollisionConstraintSet`."""
    ag3s = AG3S(CFG, robot_model=arm)
    constraint_set = ag3s.process(
        attention_map=scene.attention_grid,
        depth=scene.depth,
        pointcloud=None,
        camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam,
        robot_state=DEFAULT_ARM_Q,
        phase="approach",
        image_hw=scene.hw,
    )
    assert constraint_set.status is PipelineStatus.OK
    assert constraint_set.grounding_status is GroundingStatus.OK
    assert constraint_set.constraints is not None
    assert constraint_set.candidates and constraint_set.support_surfaces
    assert constraint_set.target is not None
    assert constraint_set.phase is Phase.APPROACH
    assert constraint_set.timestamp > 0 and constraint_set.frame_id == "base"
    assert np.array_equal(constraint_set.robot_state, DEFAULT_ARM_Q)


def test_the_pipeline_decomposes_the_scene_correctly(scene, arm):
    """The M1-M4 numbers, reproduced through the orchestrator rather than by hand."""
    ag3s, constraint_set = _run(scene, arm)
    by_type = {}
    for candidate in constraint_set.candidates:
        by_type.setdefault(candidate.source_type, []).append(candidate)
    assert len(by_type[SourceType.OBJECT]) == 2
    assert len(by_type[SourceType.TARGET]) == 1
    assert len(by_type[SourceType.SUPPORT_SURFACE]) == 1

    _, debug = ag3s.process_debug(**_inputs(scene))
    labels = scene.labels[debug["filtered_cloud"].uv[:, 1], debug["filtered_cloud"].uv[:, 0]]
    target = constraint_set.target
    hit = int((labels[target.point_indices] == LABEL_SPHERE).sum())
    gt = int((labels == LABEL_SPHERE).sum())
    iou = hit / (target.point_indices.size + gt - hit)
    assert iou > 0.95, f"IoU {iou:.4f}"


def test_nothing_is_dropped_end_to_end(scene, arm):
    """Every reconstructed point ends up owned by some candidate."""
    ag3s = AG3S(CFG, robot_model=arm)
    constraint_set, debug = ag3s.process_debug(**_inputs(scene))
    covered = np.zeros(len(debug["filtered_cloud"]), bool)
    for candidate in constraint_set.candidates:
        covered[candidate.point_indices] = True
    assert covered.all()
    assert debug["candidate_stats"]["n_unassigned_points"] == 0


def test_the_point_cloud_input_path_agrees_with_the_depth_path(scene, arm):
    from benchmark.ag3s.stages.reconstruction import backproject

    cloud = backproject(scene.depth, scene.camera_intrinsics, scene.T_base_cam)
    _, from_depth = _run(scene, arm)
    _, from_cloud = _run(
        scene, arm, depth=None, pointcloud=cloud.points, uv=cloud.uv,
        camera_intrinsics=scene.camera_intrinsics, T_base_cam=None,
    )
    assert from_cloud.status is PipelineStatus.OK
    assert len(from_cloud.candidates) == len(from_depth.candidates)
    assert np.allclose(from_cloud.target.centroid, from_depth.target.centroid, atol=1e-9)


def test_repeated_frames_are_deterministic(scene, arm):
    ag3s = AG3S(CFG, robot_model=arm)
    first = ag3s.process(**_inputs(scene))
    second = ag3s.process(**_inputs(scene))
    assert [c.id for c in first.candidates] == [c.id for c in second.candidates]
    assert np.allclose(first.target.centroid, second.target.centroid)
    assert np.array_equal(
        first.constraints.parameter_values, second.constraints.parameter_values
    )


def test_track_age_accumulates_across_frames(scene, arm):
    ag3s = AG3S(CFG, robot_model=arm)
    for frame in range(3):
        constraint_set = ag3s.process(**_inputs(scene))
        movable = [c for c in constraint_set.candidates
                   if c.source_type is not SourceType.SUPPORT_SURFACE]
        assert all(c.track_age == frame for c in movable)
    assert ag3s.frame_index == 3
    ag3s.reset()
    assert ag3s.frame_index == 0


# ---------------------------------------------------------------------------- degradation


def test_without_attention_the_pipeline_still_produces_candidates(scene, arm):
    """A frame with no attention has no target — but it still has geometry to avoid."""
    _, constraint_set = _run(scene, arm, attention_map=None)
    assert constraint_set.target is None
    assert constraint_set.grounding_status is GroundingStatus.NO_ATTENTION
    assert constraint_set.status is PipelineStatus.NO_TARGET
    assert len(constraint_set.candidates) >= 2
    assert any("no target" in note for note in constraint_set.notes)
    # And everything is held at full clearance, since nothing was identified as the target.
    for candidate in constraint_set.candidates:
        assert candidate.source_type is not SourceType.TARGET
        assert candidate.collision_enabled


def test_an_empty_depth_frame_reports_no_geometry(arm):
    ag3s = AG3S(CFG, robot_model=arm)
    constraint_set = ag3s.process(
        depth=np.zeros((60, 80)),
        camera_intrinsics=np.array([[100.0, 0, 40.0], [0, 100.0, 30.0], [0, 0, 1.0]]),
        T_base_cam=np.eye(4),
        robot_state=DEFAULT_ARM_Q,
        phase="transit",
    )
    assert constraint_set.status is PipelineStatus.NO_GEOMETRY
    assert constraint_set.candidates == [] and constraint_set.target is None


def test_without_a_robot_model_perception_still_runs(scene):
    """Debuggable on a machine with no robot description; constraints are simply absent."""
    ag3s = AG3S(CFG)
    constraint_set = ag3s.process(**_inputs(scene))
    assert ag3s.builder is None
    assert constraint_set.constraints is None
    assert constraint_set.target is not None and len(constraint_set.candidates) >= 3
    assert any("no robot model" in note for note in constraint_set.notes)


def test_slot_overflow_is_reported_as_degraded(scene, arm):
    config = CFG.with_overrides({"geometry": {"max_candidates": 1}})
    _, constraint_set = _run(scene, arm, config=config)
    assert constraint_set.status is PipelineStatus.DEGRADED
    assert any("max_candidates" in note for note in constraint_set.notes)
    assert int(constraint_set.constraints.active_mask.sum()) == 1


def test_max_points_cap_is_reported(scene, arm):
    config = CFG.with_overrides({"pointcloud": {"max_points": 5000}})
    _, constraint_set = _run(scene, arm, config=config)
    assert any("capped at max_points" in note for note in constraint_set.notes)


# ------------------------------------------------------------------- ablation reachability


@pytest.mark.parametrize(
    "overrides, check",
    [
        ({"attention": {"normalization": "minmax"}}, lambda cs: cs.target is not None),
        ({"attention": {"seed_percentile": 99.0}}, lambda cs: cs.target is not None),
        ({"clustering": {"method": "region_growing"}}, lambda cs: cs.target is not None),
        ({"pointcloud": {"self_filter": False}}, lambda cs: cs.target is not None),
        ({"geometry": {"safety_margin": 0.12}},
         lambda cs: any(c.safety_margin == pytest.approx(0.12)
                        for c in cs.candidates if c.source_type is SourceType.OBJECT)),
        ({"constraint": {"squared_distance": False}},
         lambda cs: cs.constraints.squared_distance is False),
        ({"constraint": {"horizon": 7}}, lambda cs: cs.constraints.horizon == 7),
    ],
    ids=["normalization", "seed-threshold", "clustering-method", "self-filter",
         "safety-margin", "squared-distance", "horizon"],
)
def test_config_only_ablations_reach_the_output(scene, arm, overrides, check):
    _, constraint_set = _run(scene, arm, config=CFG.with_overrides(overrides))
    assert check(constraint_set)


@pytest.mark.parametrize("primitive", [t.value for t in PrimitiveType])
def test_the_primitive_type_ablation_reaches_the_candidates(scene, arm, primitive):
    config = CFG.with_overrides({"geometry": {"primitive": primitive}})
    _, constraint_set = _run(scene, arm, config=config)
    shaped = [c for c in constraint_set.candidates if c.geometry]
    assert shaped and all(c.geometry[0].type.value == primitive for c in shaped)


def test_the_2d_only_ablation_degrades_the_target(scene, arm):
    """The headline comparison, reachable from the top-level API."""
    ag3s_3d = AG3S(CFG, robot_model=arm)
    three_d, debug = ag3s_3d.process_debug(**_inputs(scene))
    two_d, _ = AG3S(
        CFG.with_overrides({"clustering": {"use_3d_connectivity": False}}), robot_model=arm
    ).process_debug(**_inputs(scene))

    labels = scene.labels[debug["filtered_cloud"].uv[:, 1], debug["filtered_cloud"].uv[:, 0]]

    def _iou(target):
        hit = int((labels[target.point_indices] == LABEL_SPHERE).sum())
        gt = int((labels == LABEL_SPHERE).sum())
        return hit / (target.point_indices.size + gt - hit)

    assert _iou(three_d.target) > 0.95
    assert _iou(two_d.target) < 0.6
    assert int((labels[two_d.target.point_indices] == LABEL_BOX).sum()) > 0


@pytest.mark.parametrize("phase", [p.value for p in Phase])
def test_the_phase_ablation_reaches_the_constraint_slots(scene, arm, phase):
    """`contact.phase_aware` must be reachable from config alone, and must change the margins.

    The observable moved with the contract. It used to be the candidate's own `safety_margin`, which
    shrank through the phases; that scalar is now the conservative margin and is deliberately
    constant, because shrinking it relaxed the torso along with the gripper. What varies is the
    `(robot sphere, slot)` block, so that is what this reads.
    """
    from benchmark.ag3s.types import Manipulator
    from tests.ag3s.fixtures import FINGERTIP_LINK

    config = CFG.with_overrides({"contact": {"contact_links": {"right": [FINGERTIP_LINK]}}})
    blind = config.with_overrides({"contact": {"phase_aware": False}})
    _, aware_set = _run(scene, arm, phase=phase, config=config,
                        active_manipulators=[Manipulator.RIGHT])
    _, blind_set = _run(scene, arm, phase=phase, config=blind,
                        active_manipulators=[Manipulator.RIGHT])

    def _target_margins(constraint_set):
        spec = constraint_set.constraints
        lo_d, _ = spec.layout["d_safe"]
        m = spec.max_candidates
        margins = spec.parameter_values[lo_d : lo_d + spec.n_robot_spheres * m].reshape(-1, m)
        target_id = [c.id for c in constraint_set.candidates if c.source_type is SourceType.TARGET][0]
        slots = np.flatnonzero(spec.candidate_ids == target_id)
        assert slots.size, "the target must occupy a slot in every phase"
        return margins[:, slots[0]]

    aware = _target_margins(aware_set)
    blind_margins = _target_margins(blind_set)
    fingertip = [i for i, n in enumerate(arm.sphere_link_names) if n == FINGERTIP_LINK]

    assert np.allclose(blind_margins, CFG.geometry.safety_margin), (
        "phase_aware=False must hold every link at full clearance — that is the ablation"
    )
    if phase == Phase.TRANSIT.value:
        assert np.allclose(aware, blind_margins)
    else:
        assert aware[fingertip].max() < CFG.geometry.safety_margin, (
            f"phase {phase} did not relax the authorized fingertip"
        )
    # Whatever the phase, everything that is not the fingertip stays at full clearance, and the
    # target keeps its slot. This is the property the old `collision_enabled=False` destroyed.
    others = [i for i in range(len(aware)) if i not in fingertip]
    assert np.allclose(aware[others], CFG.geometry.safety_margin), (
        f"phase {phase} relaxed a link that was never authorized"
    )
    assert int(aware_set.constraints.active_mask.sum()) == int(
        blind_set.constraints.active_mask.sum()
    ), "a phase change must not alter how many slots are occupied"


# ------------------------------------------------------------------------------- profiling


def test_every_stage_is_timed(scene, arm):
    ag3s = AG3S(CFG, robot_model=arm)
    ag3s.process(**_inputs(scene))
    measured = set(ag3s.profiler.stages)
    assert set(STAGE_ORDER) <= measured, f"untimed: {set(STAGE_ORDER) - measured}"
    for name in STAGE_ORDER:
        assert ag3s.profiler.stages[name].last_ms >= 0.0


def test_the_constraint_set_carries_the_frame_profile(scene, arm):
    _, constraint_set = _run(scene, arm)
    assert set(STAGE_ORDER) <= set(constraint_set.profile)
    assert sum(constraint_set.profile.values()) > 0.0


def test_the_latency_table_lists_every_stage(scene, arm):
    ag3s = AG3S(CFG, robot_model=arm)
    ag3s.process(**_inputs(scene))
    table = ag3s.profiler.to_table()
    for name in STAGE_ORDER:
        assert name in table
    assert "TOTAL" in table


def test_warmup_frames_are_excluded():
    profiler = StageProfiler(enabled=True, warmup_frames=1)
    for value in (100.0, 1.0, 1.0):
        with profiler.stage("x"):
            pass
        profiler.record("y", value)
        profiler.end_frame()
    assert profiler.stages["y"].count == 2
    assert profiler.stages["y"].mean_ms == pytest.approx(1.0)


def test_profiling_can_be_switched_off(scene, arm):
    config = CFG.with_overrides({"profiling": {"enabled": False}})
    ag3s, constraint_set = _run(scene, arm, config=config)
    assert ag3s.profiler.stages == {}
    assert constraint_set.profile == {}


# --------------------------------------------------------------------------- summary + viz


def test_the_summary_is_json_friendly(scene, arm):
    import json

    _, constraint_set = _run(scene, arm)
    digest = summary(constraint_set)
    json.loads(json.dumps(digest, default=str))
    assert digest["status"] == "ok" and digest["has_target"] is True
    assert digest["n_active_slots"] == 3


def test_visualization_imports_lazily_and_writes_every_panel(scene, arm, tmp_path):
    """Core must import without matplotlib, and the dumps must index the right cloud.

    The index basis matters: `point_indices` refer to the post-self-filter cloud, so plotting them
    against the raw cloud silently scatters every cluster across the scene.
    """
    import sys

    import benchmark.ag3s  # noqa: F401  - importing the package must not pull matplotlib in

    assert "benchmark.ag3s.runtime.visualization" not in sys.modules

    from benchmark.ag3s.runtime.visualization import dump_from_pipeline

    ag3s = AG3S(CFG, robot_model=arm)
    constraint_set, debug = ag3s.process_debug(**_inputs(scene))
    written = dump_from_pipeline(tmp_path, ag3s, constraint_set, debug)
    assert len(written) == 8
    for path in written:
        assert path.exists() and path.stat().st_size > 1000


def test_visualization_uses_the_indexed_cloud(scene, arm, tmp_path):
    """A regression guard for the index-basis bug, asserted numerically rather than visually."""
    from benchmark.ag3s.runtime.visualization import dump_stages

    ag3s = AG3S(CFG, robot_model=arm)
    constraint_set, debug = ag3s.process_debug(**_inputs(scene))
    raw, working = debug["raw_cloud"], debug["filtered_cloud"]
    assert len(raw) > len(working), "fixture should exercise the self-filter"

    target = constraint_set.target
    # Indices are valid against the working cloud and would land elsewhere in the raw one.
    assert np.allclose(working.points[target.point_indices], target.points)
    assert not np.allclose(raw.points[target.point_indices], target.points)

    written = dump_stages(
        tmp_path, raw_cloud=raw, working_cloud=working,
        attention_cloud=debug["attention_cloud"], seed_indices=debug["seed_indices"],
        target=target, candidates=constraint_set.candidates,
        support_surfaces=constraint_set.support_surfaces,
        constraint_spec=constraint_set.constraints,
    )
    assert len(written) == 8


def test_the_cli_smoke_run_succeeds(tmp_path):
    from benchmark.ag3s.runtime.pipeline import main

    out = tmp_path / "summary.json"
    assert main(["--frames", "2", "--json", str(out), "--profile"]) == 0
    assert out.exists()
