"""The SEAM hook: does it fit the chunk contract, and does it leave alone what it must?

Most of what can go wrong here is not about optimization at all. A refiner that returned the right
joints in the wrong columns, or that smoothed the gripper's open/close into something in between, or
that took down the policy when a camera dropped a frame, would each be a working optimizer attached
to a broken robot.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.seam_vla.refinement.base import DecodedChunkRefiner
from benchmark.trajopt import wire
from benchmark.trajopt.refiner import TrajOptChunkRefiner
from benchmark.trajopt.types import TrajOptStatus
from tests.trajopt.fixtures import empty_scene, make_scenario

pytest.importorskip("casadi")
pytest.importorskip("osqp")


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(horizon=12, penetration=0.03)


def _refiner(scenario, scene=None, **overrides):
    config = scenario.config.with_overrides(overrides) if overrides else scenario.config
    scene = scenario.scene if scene is None else scene
    return TrajOptChunkRefiner(
        scenario.robot, scenario.layout, lambda ctx: (scene, scenario.q_now, True), config
    )


def test_it_implements_the_hook_seam_reserved():
    """`seam_vla` is not modified: the class is imported and subclassed."""
    assert issubclass(TrajOptChunkRefiner, DecodedChunkRefiner)


def test_the_chunk_keeps_its_shape(scenario):
    out = _refiner(scenario).refine(scenario.chunk)
    assert out.shape == scenario.chunk.shape


def test_the_grippers_pass_through_untouched(scenario):
    """Near-binary open/close. Smoothing them would blur a discrete decision into a half-grasp."""
    out = _refiner(scenario).refine(scenario.chunk)
    cols = list(wire.GRIPPER_COLUMNS)
    assert np.array_equal(out[:, cols], scenario.chunk[:, cols])


def test_the_padding_passes_through_untouched(scenario):
    """π0.5's action vector is 32 wide and 14 of it means anything. The rest must arrive as it left."""
    padding = list(range(scenario.layout.n_valid, scenario.layout.action_dim))
    out = _refiner(scenario).refine(scenario.chunk)
    assert np.array_equal(out[:, padding], scenario.chunk[:, padding])


def test_only_the_twelve_arm_joints_move(scenario):
    out = _refiner(scenario).refine(scenario.chunk)
    changed = np.flatnonzero(np.abs(out - scenario.chunk).max(axis=0) > 1e-9)
    assert set(changed) <= set(scenario.layout.action_columns.tolist())


def test_the_executed_portion_is_made_safe(scenario):
    refiner = _refiner(scenario)
    refiner.refine(scenario.chunk)
    result = refiner.last_result
    assert result.improved
    assert result.max_violation <= scenario.config.safety.violation_tolerance


def test_a_clear_scene_returns_the_chunk_unchanged(scenario):
    refiner = _refiner(scenario, scene=None)
    empty = empty_scene(refiner.optimizer.linearizer)
    unchanged = _refiner(scenario, scene=empty).refine(scenario.chunk)
    assert np.allclose(unchanged, scenario.chunk)


def test_a_perception_failure_passes_the_chunk_through_rather_than_raising(scenario):
    """A refiner that raised here would take down the policy over one dropped camera frame — a worse
    outcome than executing the policy's own answer and saying so."""

    def broken(_context):
        raise RuntimeError("camera timeout")

    seen = []
    refiner = TrajOptChunkRefiner(
        scenario.robot, scenario.layout, broken, scenario.config, on_result=seen.append
    )
    out = refiner.refine(scenario.chunk)
    assert np.array_equal(out, scenario.chunk)
    assert refiner.last_result is None
    assert seen and seen[0].status is TrajOptStatus.SOLVER_FAILED
    assert "camera timeout" in seen[0].notes[0]


def test_only_the_planned_window_is_rewritten(scenario):
    """The executor runs `execution_length` actions and discards the rest, so planning the whole
    chunk costs time for motion that is thrown away. The tail is the policy's own answer."""
    planned = 6
    refiner = _refiner(scenario, horizon={"plan_horizon": planned})
    out = refiner.refine(scenario.chunk)
    columns = scenario.layout.action_columns
    assert np.array_equal(out[planned:, columns], scenario.chunk[planned:, columns])
    assert not np.allclose(out[:planned, columns], scenario.chunk[:planned, columns])
    assert refiner.last_result.metrics["planned_steps"] == planned
    assert "plan_join_discontinuity_rad" in refiner.last_result.metrics


def test_the_plan_window_must_cover_what_actually_executes(scenario):
    from benchmark.trajopt.config import TrajOptConfigError

    with pytest.raises(TrajOptConfigError, match="shorter than"):
        scenario.config.with_overrides(
            {"horizon": {"execution_length": 8, "plan_horizon": 4}}
        )


def test_the_continuity_reference_is_aligned_to_the_executed_offset(scenario):
    """The previous chunk's first K steps have already run, so the part that overlaps this chunk
    starts at index K — the same alignment SEAM's own prior uses."""
    refiner = _refiner(scenario)
    previous = np.zeros_like(scenario.chunk)
    previous[:, scenario.layout.action_columns] = 0.25
    aligned = refiner._continuity_reference(
        {"previous_physical_chunk": previous}, scenario.horizon
    )
    expected = scenario.horizon - scenario.config.horizon.execution_length
    assert aligned is not None and aligned.shape[1] == min(expected, scenario.horizon)
    assert np.allclose(aligned, 0.25)


def test_no_previous_chunk_disables_continuity_rather_than_failing(scenario):
    refiner = _refiner(scenario)
    assert refiner._continuity_reference(None, scenario.horizon) is None
    assert refiner._continuity_reference({"previous_physical_chunk": None}, scenario.horizon) is None


def test_would_violate_flags_the_input_not_the_output(scenario):
    """Mirrors the diagnostic helper on `seam_vla`'s own skeleton, so the offline analysis scripts
    that already call it keep working."""
    refiner = _refiner(scenario)
    flags = refiner.would_violate(scenario.chunk)
    assert flags.shape == (scenario.horizon,)
    assert flags.any(), "the fixture's chunk does violate something"

    refined = refiner.refine(scenario.chunk)
    assert not refiner.would_violate(refined).any()


def test_reset_clears_the_warm_start_and_the_continuity_reference(scenario):
    refiner = _refiner(scenario)
    refiner.refine(scenario.chunk)
    assert refiner.last_result is not None
    refiner.reset()
    assert refiner.last_result is None and refiner._previous is None


def test_the_callback_sees_every_frame(scenario):
    """Where a caller hooks its own policy for VIOLATED frames — stop, hold, re-plan. Nothing in
    this package decides that."""
    seen = []
    refiner = TrajOptChunkRefiner(
        scenario.robot, scenario.layout, lambda ctx: (scenario.scene, scenario.q_now, True),
        scenario.config, on_result=seen.append,
    )
    refiner.refine(scenario.chunk)
    refiner.refine(scenario.chunk)
    assert len(seen) == 2
    assert all(r.metrics["frame_index"] > 0 for r in seen)
