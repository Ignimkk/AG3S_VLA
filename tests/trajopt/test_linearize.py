"""Clearances, gradients, and the two reductions that decide which rows the QP sees.

The gradient checks are finite differences against the clearance function itself, not against another
derivation of the same formula. A closed-form gradient that agrees with a closed-form re-derivation
proves only that the algebra was copied consistently.

The reduction checks all rest on one property: pruning may cost convergence, but it may not hide a
collision, because `full_violation` re-evaluates every row at full resolution. Each test below either
verifies that property or verifies that a pruned solve reaches the same place as an unpruned one.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.trajopt.linearize import (
    CollisionBlock,
    CollisionLinearizer,
    SceneSnapshot,
    append_collision_rows,
)
from benchmark.trajopt.problem import build_problem
from tests.trajopt.fixtures import Q_HOME, empty_scene, make_scenario, small_config

pytest.importorskip("casadi", reason="the linearizer needs CasADi for forward kinematics")

EPS = 1e-6


@pytest.fixture(scope="module")
def scenario():
    return make_scenario(horizon=6)


def _all_rows(scenario):
    """Every row, with the reductions switched off — the reference the pruned versions are judged against."""
    config = scenario.config.with_overrides(
        {"reduction": {"activation_band": 1e6, "rows_per_step": 4096, "temporal_stride": 1}}
    )
    return scenario.linearizer.linearize(scenario.reference, scenario.q_now, scenario.scene, config)


# ------------------------------------------------------------------ kinematics


def test_forward_kinematics_matches_the_model_ag3s_uses(scenario):
    """The optimizer and the constraint set must place the robot in the same place, step by step.

    Two implementations of forward kinematics eventually disagree, and the disagreement here would be
    invisible: the optimizer would clear obstacles that the constraints describe somewhere else.
    """
    centres, _ = scenario.linearizer.sphere_states(scenario.reference, scenario.q_now)
    full = scenario.layout.full_q(scenario.reference, scenario.q_now)
    for k in range(scenario.horizon):
        expected, _ = scenario.robot.sphere_centers_numeric(full[:, k])
        assert np.abs(centres[k] - expected).max() < 1e-12


def test_the_position_jacobian_matches_finite_differences(scenario):
    centres, jac = scenario.linearizer.sphere_states(scenario.reference, scenario.q_now)
    rng = np.random.default_rng(0)
    for _ in range(4):
        k = int(rng.integers(scenario.horizon))
        s = int(rng.integers(scenario.linearizer.n_spheres))
        approximate = np.zeros((3, scenario.layout.nq_opt))
        for j in range(scenario.layout.nq_opt):
            moved = scenario.reference.copy()
            moved[j, k] += EPS
            approximate[:, j] = (
                scenario.linearizer.sphere_states(moved, scenario.q_now)[0][k, s] - centres[k, s]
            ) / EPS
        assert np.abs(jac[k, s] - approximate).max() < 1e-5


def test_the_fixed_joints_position_the_arm_rather_than_being_ignored(scenario):
    """The torso is not a decision variable, but leaving it out of kinematics would place every
    collision sphere as though the robot were standing straight."""
    tilted = scenario.q_now.copy()
    tilted[1] = 0.5  # torso_1
    upright = scenario.linearizer.sphere_states(scenario.reference, scenario.q_now)[0]
    leaning = scenario.linearizer.sphere_states(scenario.reference, tilted)[0]
    assert np.abs(leaning - upright).max() > 0.05


# ------------------------------------------------------------------ clearances


def test_the_clearance_is_the_distance_minus_everything_that_must_fit_between(scenario):
    """Checked against the definition, computed independently for one pair."""
    centres, _ = scenario.linearizer.sphere_states(scenario.reference, scenario.q_now)
    candidate, _, _ = scenario.linearizer._clearances_from(centres, scenario.scene)
    k, s, j = 2, 10, 0
    expected = (
        np.linalg.norm(centres[k, s] - scenario.scene.candidate_pos[j])
        - scenario.scene.robot_radii[s]
        - scenario.scene.candidate_radius[j]
        - scenario.scene.d_safe[s, j]
    )
    assert candidate[k, s, j] == pytest.approx(expected)


def test_an_inactive_slot_is_absent_rather_than_far_away(scenario):
    """`+inf`, not a large number: an empty slot must not flatter the violation report or win a place
    in the row budget."""
    centres, _ = scenario.linearizer.sphere_states(scenario.reference, scenario.q_now)
    candidate, plane, _ = scenario.linearizer._clearances_from(centres, scenario.scene)
    inactive = ~scenario.scene.candidate_active
    assert np.all(np.isinf(candidate[:, :, inactive]))
    assert np.all(np.isinf(plane[:, :, ~scenario.scene.plane_active]))


def test_the_row_gradients_match_finite_differences(scenario):
    rows = _all_rows(scenario)
    used = np.flatnonzero(rows.used.reshape(-1))
    value = rows.value.reshape(-1)
    gradient = rows.gradient.reshape(-1, scenario.layout.nq_opt)
    step = np.repeat(np.arange(rows.value.shape[0]), rows.value.shape[1])
    sphere = rows.sphere.reshape(-1)
    slot = rows.slot.reshape(-1)

    def clearance(trajectory, i):
        centres = scenario.linearizer.sphere_states(trajectory, scenario.q_now)[0]
        candidate, plane, _ = scenario.linearizer._clearances_from(centres, scenario.scene)
        k, s, j = int(step[i]), int(sphere[i]), int(slot[i])
        return float(candidate[k, s, j]) if j >= 0 else float(plane[k, s, -1 - j])

    rng = np.random.default_rng(1)
    worst = 0.0
    for i in rng.choice(used, min(25, used.size), replace=False):
        assert clearance(scenario.reference, i) == pytest.approx(value[i], abs=1e-12)
        for j in range(scenario.layout.nq_opt):
            moved = scenario.reference.copy()
            moved[j, int(step[i])] += EPS
            worst = max(worst, abs((clearance(moved, i) - value[i]) / EPS - gradient[i, j]))
    assert worst < 1e-5, f"gradient disagrees with finite differences by {worst:.2e}"


def test_a_row_only_touches_its_own_step(scenario):
    """The property that makes the collision block's sparsity fixed. If a row reached across steps,
    `CollisionBlock`'s pattern would be wrong and the QP would silently solve a different problem."""
    rows = _all_rows(scenario)
    block = CollisionBlock(scenario.horizon, rows.value.shape[1], scenario.layout.nq_opt)
    dense = block.matrix(rows).toarray()[:, : scenario.layout.nq_opt * scenario.horizon]
    for i in np.flatnonzero(rows.used.reshape(-1))[:20]:
        k = i // rows.value.shape[1]
        own = slice(k * scenario.layout.nq_opt, (k + 1) * scenario.layout.nq_opt)
        outside = np.delete(dense[i], np.arange(own.start, own.stop))
        assert np.abs(outside).max() == 0.0


# ------------------------------------------------------------------ the reductions


def test_the_full_violation_never_looks_at_a_subset(scenario):
    """The guarantee behind every reduction: what the caller is told comes from all the rows."""
    # The band must stay above the trust radius — the config enforces that — so both shrink
    # together. A band this tight prunes almost everything, which is the point: the report must not
    # come from what survived.
    tight = scenario.config.with_overrides(
        {"reduction": {"activation_band": 1e-3, "rows_per_step": 1},
         "sqp": {"trust_radius": 1e-4, "trust_radius_min": 1e-5}}
    )
    pruned = scenario.linearizer.linearize(scenario.reference, scenario.q_now, scenario.scene, tight)
    assert pruned.n_rows <= scenario.horizon
    reported = scenario.linearizer.full_violation(
        scenario.reference, scenario.q_now, scenario.scene
    )
    assert reported == pytest.approx(-scenario.reference_violation)
    assert reported < 0.0, "the fixture's reference must actually violate something"


def test_the_activation_band_keeps_every_row_that_could_matter(scenario):
    """A row outside the band cannot become violated within one trust-region step, so dropping it
    cannot change the QP's answer. What it must not do is drop a row that is already violated."""
    banded = scenario.config.with_overrides({"reduction": {"activation_band": 0.35}})
    rows = scenario.linearizer.linearize(
        scenario.reference, scenario.q_now, scenario.scene, banded
    )
    kept = rows.value[rows.used]
    assert kept.size > 0
    assert kept.max() < 0.35
    assert kept.min() == pytest.approx(-scenario.reference_violation, abs=1e-9), (
        "the tightest row was pruned; the band is unsafe"
    )


def test_the_per_step_budget_keeps_the_tightest_rows(scenario):
    rows = scenario.linearizer.linearize(
        scenario.reference, scenario.q_now, scenario.scene,
        scenario.config.with_overrides({"reduction": {"rows_per_step": 3}}),
    )
    assert rows.value.shape[1] == 3
    for k in range(scenario.horizon):
        kept = rows.value[k][rows.used[k]]
        if kept.size:
            assert kept.max() <= 0.35


def test_a_binding_budget_is_reported(scenario):
    """Dropping a *near-active* row is the dangerous case, so it never happens silently."""
    rows = scenario.linearizer.linearize(
        scenario.reference, scenario.q_now, scenario.scene,
        scenario.config.with_overrides({"reduction": {"rows_per_step": 1}}),
    )
    assert rows.budget_bound_steps, "the budget bound but nothing said so"
    roomy = _all_rows(scenario)
    assert roomy.budget_bound_steps == ()


def test_the_temporal_stride_leaves_the_row_shape_alone(scenario):
    """Striding changes which steps are enforced, never how many rows exist — the QP's pattern must
    not depend on it either."""
    dense = scenario.linearizer.linearize(
        scenario.reference, scenario.q_now, scenario.scene, scenario.config
    )
    strided = scenario.linearizer.linearize(
        scenario.reference, scenario.q_now, scenario.scene,
        scenario.config.with_overrides({"reduction": {"temporal_stride": 2}}),
    )
    assert strided.value.shape == dense.value.shape
    assert strided.gradient.shape == dense.gradient.shape
    assert not strided.used[1::2].any(), "an unenforced step contributed rows"
    assert strided.n_rows < dense.n_rows


def test_unused_rows_are_trivially_satisfied_rather_than_absent(scenario):
    """The same device AG3S uses for empty candidate slots, and for the same reason: the shape of the
    problem must not depend on what the scene happens to contain."""
    rows = scenario.linearizer.linearize(
        scenario.reference, scenario.q_now, scenario.scene,
        scenario.config.with_overrides(
            {"reduction": {"activation_band": 1e-3},
             "sqp": {"trust_radius": 1e-4, "trust_radius_min": 1e-5}}
        ),
    )
    block = CollisionBlock(scenario.horizon, rows.value.shape[1], scenario.layout.nq_opt)
    problem = build_problem(
        scenario.reference, scenario.reference, scenario.limits, scenario.config,
        n_slack=block.n_rows,
    )
    full = append_collision_rows(problem, rows, scenario.reference, block)
    lo, hi = full.row_blocks["collision"]
    unused = ~rows.used.reshape(-1)
    assert np.allclose(full.l[lo:hi][unused], 0.0)
    assert np.all(np.isinf(full.u[lo:hi]))


def test_the_collision_block_pattern_is_identical_across_scenes(scenario):
    """What lets a QP solver reuse its factorization. Measured at 193 ms per solve when the pattern
    moved every iteration, against a 25 ms budget for the whole optimization."""
    block = CollisionBlock(scenario.horizon, scenario.config.reduction.rows_per_step,
                           scenario.layout.nq_opt)
    first = block.matrix(
        scenario.linearizer.linearize(
            scenario.reference, scenario.q_now, scenario.scene, scenario.config
        )
    )
    elsewhere = scenario.linearizer.linearize(
        scenario.reference * 1.1, scenario.q_now,
        empty_scene(scenario.linearizer), scenario.config,
    )
    second = block.matrix(elsewhere)
    assert np.array_equal(first.indices, second.indices)
    assert np.array_equal(first.indptr, second.indptr)


def test_the_backoff_tightens_the_linearized_row_only(scenario):
    """The QP is asked for a few millimetres more than the true constraint needs, because the true
    clearance sits below its own tangent. Without it an SQP converges with real penetration left."""
    rows = _all_rows(scenario)
    block = CollisionBlock(scenario.horizon, rows.value.shape[1], scenario.layout.nq_opt)
    problem = build_problem(
        scenario.reference, scenario.reference, scenario.limits, scenario.config,
        n_slack=block.n_rows,
    )
    plain = append_collision_rows(problem, rows, scenario.reference, block, 0.0)
    backed = append_collision_rows(problem, rows, scenario.reference, block, 5e-3)
    lo, hi = plain.row_blocks["collision"]
    used = rows.used.reshape(-1)
    assert np.allclose(backed.l[lo:hi][used] - plain.l[lo:hi][used], 5e-3)
    assert np.allclose(backed.l[lo:hi][~used], plain.l[lo:hi][~used])


def test_a_snapshot_decoded_from_ag3s_carries_the_clearance_matrix(scenario):
    """`SceneSnapshot` reads AG3S's own parameter vector rather than re-deriving the policy, so a
    per-link margin set by AG3S reaches the optimizer without a second implementation."""
    assert scenario.scene.d_safe.shape == (
        scenario.linearizer.n_spheres, scenario.scene.n_slots
    )
    assert scenario.scene.robot_radii.shape == (scenario.linearizer.n_spheres,)


def test_an_empty_scene_reports_itself_as_empty(scenario):
    assert empty_scene(scenario.linearizer).is_empty
    assert not scenario.scene.is_empty
