"""Constraint-emission tests.

Two things are being pinned down here. First, that the emitted number *is* the distance it claims to
be — every row is cross-checked against an analytic sphere-to-sphere computation, because a
constraint that is subtly wrong is worse than one that is missing. Second, that the structure never
moves: the row count, the parameter length and the sparsity pattern must be identical across frames,
because that invariance is what a real-time TO warm-starts against.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.constraints.constraint_builder import ConstraintBuilder, evaluate
from AG3S.ag3s.stages.geometry import fit_capsule, fit_sphere, to_spheres
from AG3S.ag3s.types import CollisionCandidate, PrimitiveType, SourceType, SupportSurface
from tests.ag3s.fixtures import PlanarTwoLinkArm

ca = pytest.importorskip("casadi")

HORIZON = 4
CFG = AG3SConfig().with_overrides({"constraint": {"horizon": HORIZON}})


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


@pytest.fixture(scope="module")
def builder(arm):
    return ConstraintBuilder(arm, CFG.constraint, CFG.geometry)


def _candidate(centre, *, radius=0.04, candidate_id=0, margin=0.05, enabled=True, primitive=None):
    rng = np.random.default_rng(candidate_id)
    points = np.asarray(centre, np.float64) + rng.normal(0.0, radius / 3.0, (40, 3))
    geometry = [primitive] if primitive is not None else [
        fit_sphere(points, safety_margin=margin, candidate_id=candidate_id)
    ]
    return CollisionCandidate(
        id=candidate_id, source_type=SourceType.OBJECT, semantic_role="object",
        geometry=geometry, pose=np.eye(4), dimensions=np.full(3, radius), confidence=1.0,
        safety_margin=margin, collision_enabled=enabled, contact_permission=False,
        phase_rule="always_on", timestamp=1.0, centroid=points.mean(axis=0),
        point_indices=np.arange(40), point_count=40,
    )


def _rows(spec, builder):
    """Reshape the sphere block to `(step, robot_sphere, slot)` — the documented row order."""
    n = builder.horizon * builder.n_robot_spheres * builder.max_candidates
    return n


# --------------------------------------------------------------------------- correctness


def test_the_constraint_value_equals_the_analytic_distance(arm, builder):
    """`h = ||p_r - p_c||^2 - (r_r + r_c + d_safe)^2`, checked against numpy."""
    q = np.array([-0.6, -0.6])
    candidate = _candidate(arm.end_effector_numeric(q) + np.array([0.05, 0.0, 0.05]))
    spec = builder.build([candidate])
    h = evaluate(spec, np.tile(q.reshape(-1, 1), (1, builder.horizon)), builder)

    centre, radius = to_spheres(candidate.geometry[0])[0]
    robot_centres, robot_radii = arm.sphere_centers_numeric(q)
    expected = ((robot_centres - centre) ** 2).sum(axis=1) - (
        robot_radii + radius + candidate.safety_margin
    ) ** 2
    got = np.array([h[s * builder.max_candidates] for s in range(builder.n_robot_spheres)])
    assert np.abs(expected - got).max() < 1e-12


def test_the_non_squared_form_equals_the_analytic_distance(arm):
    config = CFG.with_overrides({"constraint": {"squared_distance": False}})
    builder = ConstraintBuilder(arm, config.constraint, config.geometry)
    q = np.array([-0.6, -0.6])
    candidate = _candidate(arm.end_effector_numeric(q) + np.array([0.05, 0.0, 0.05]))
    spec = builder.build([candidate])
    h = evaluate(spec, np.tile(q.reshape(-1, 1), (1, builder.horizon)), builder)

    centre, radius = to_spheres(candidate.geometry[0])[0]
    robot_centres, robot_radii = arm.sphere_centers_numeric(q)
    expected = np.linalg.norm(robot_centres - centre, axis=1) - (
        robot_radii + radius + candidate.safety_margin
    )
    got = np.array([h[s * builder.max_candidates] for s in range(builder.n_robot_spheres)])
    assert np.abs(expected - got).max() < 1e-6


def test_the_two_forms_agree_on_the_sign(arm, builder):
    """They must classify feasibility identically, or the ablation changes safety, not just scaling."""
    squared_builder = builder
    plain_builder = ConstraintBuilder(
        arm, CFG.with_overrides({"constraint": {"squared_distance": False}}).constraint, CFG.geometry
    )
    for offset in (0.0, 0.03, 0.08, 0.20):
        q = np.array([-0.6, -0.6])
        candidate = _candidate(arm.end_effector_numeric(q) + np.array([offset, 0.0, 0.0]))
        Q = np.tile(q.reshape(-1, 1), (1, HORIZON))
        a = evaluate(squared_builder.build([candidate]), Q, squared_builder)
        b = evaluate(plain_builder.build([candidate]), Q, plain_builder)
        assert np.sign(a.min()) == np.sign(b.min()), f"offset {offset}: {a.min()} vs {b.min()}"


def test_a_violated_configuration_reports_a_negative_row(arm, builder):
    q = np.array([-0.6, -0.6])
    spec = builder.build([_candidate(arm.end_effector_numeric(q))])  # obstacle on the tip
    h = evaluate(spec, np.tile(q.reshape(-1, 1), (1, builder.horizon)), builder)
    assert h.min() < 0.0


def test_the_horizon_is_expanded(arm, builder):
    """Every step gets its own rows, and a step that differs must produce different values."""
    trajectory = np.stack(
        [np.array([-0.6, -0.6]) + 0.1 * k * np.array([1.0, 0.0]) for k in range(builder.horizon)],
        axis=1,
    )
    spec = builder.build([_candidate(arm.end_effector_numeric(trajectory[:, 0]))])
    h = evaluate(spec, trajectory, builder)
    assert h.shape[0] == spec.n_constraints
    per_step = builder.n_robot_spheres * builder.max_candidates
    assert not np.allclose(h[:per_step], h[per_step : 2 * per_step])


def test_a_capsule_candidate_occupies_several_slots():
    """One candidate, many constraint spheres — that is why a slot is a sphere and not a candidate."""
    arm = PlanarTwoLinkArm()
    builder = ConstraintBuilder(arm, CFG.constraint, CFG.geometry)
    rng = np.random.default_rng(0)
    axis = np.array([1.0, 0.0, 0.0])
    rod = np.array([0.6, 0.0, 1.0]) + rng.uniform(-0.2, 0.2, (300, 1)) * axis + rng.normal(0, 0.005, (300, 3))
    capsule = fit_capsule(rod, safety_margin=0.05, candidate_id=5)
    assert capsule.type is PrimitiveType.CAPSULE

    spec = builder.build([_candidate(rod.mean(axis=0), candidate_id=5, primitive=capsule)])
    assert int(spec.active_mask.sum()) == len(to_spheres(capsule)) > 1
    assert set(spec.candidate_ids[spec.active_mask > 0].tolist()) == {5}


# ------------------------------------------------------------------------------ active mask


def test_inactive_slots_are_satisfied_and_inert(builder):
    spec = builder.build([])
    h = evaluate(spec, np.zeros((builder.nq, builder.horizon)), builder)
    assert np.allclose(h, 1.0)
    assert int(spec.active_mask.sum()) == 0
    assert np.all(spec.candidate_ids == -1)


def test_an_inactive_slot_contributes_no_gradient(builder):
    """Its Jacobian rows must be exactly zero, or the optimizer is nudged by an absent obstacle."""
    spec = builder.build([])
    Q = ca.SX.sym("Q", builder.nq, builder.horizon)
    jacobian = ca.jacobian(spec.expr_factory(Q, spec.parameter_symbol), ca.reshape(Q, -1, 1))
    fn = ca.Function("J", [Q, spec.parameter_symbol], [jacobian])
    values = np.asarray(fn(np.zeros((builder.nq, builder.horizon)), spec.parameter_values))
    assert np.abs(values).max() == 0.0


def test_adding_a_candidate_does_not_change_the_structure(builder):
    """The invariance a warm start depends on."""
    empty = builder.build([])
    one = builder.build([_candidate(np.array([0.5, 0.0, 1.0]))])
    two = builder.build([
        _candidate(np.array([0.5, 0.0, 1.0]), candidate_id=0),
        _candidate(np.array([0.8, 0.0, 1.0]), candidate_id=1),
    ])
    for spec in (one, two):
        assert spec.n_constraints == empty.n_constraints
        assert spec.parameter_values.shape == empty.parameter_values.shape
        assert spec.active_mask.shape == empty.active_mask.shape
        # Bound methods are fresh objects on each attribute access, so compare what actually
        # matters: the same underlying function on the same builder, and the identical symbol.
        assert spec.expr_factory.__func__ is empty.expr_factory.__func__
        assert spec.expr_factory.__self__ is empty.expr_factory.__self__
        assert spec.parameter_symbol is empty.parameter_symbol
    assert [int(s.active_mask.sum()) for s in (empty, one, two)] == [0, 1, 2]


def test_the_sparsity_pattern_is_invariant(builder):
    """Same nonzero structure whatever the scene, so IPOPT never re-analyses it."""
    Q = ca.SX.sym("Q", builder.nq, builder.horizon)
    jacobian = ca.jacobian(
        builder.expr_factory(Q, builder.parameter_symbol), ca.reshape(Q, -1, 1)
    )
    fn = ca.Function("J", [Q, builder.parameter_symbol], [jacobian])
    a = fn(np.zeros((builder.nq, builder.horizon)), builder.build([]).parameter_values)
    b = fn(
        np.zeros((builder.nq, builder.horizon)),
        builder.build([_candidate(np.array([0.5, 0.0, 1.0]))]).parameter_values,
    )
    assert a.sparsity() == b.sparsity()


def test_geometry_always_occupies_a_slot_whatever_collision_enabled_says(builder):
    """The contract that replaced "disable the target at GRASP".

    `collision_enabled` used to gate slot assignment, which meant a phase rule could delete a real
    object's constraint rows and let the optimizer drive straight through it. A candidate now earns
    its slot by existing physically; permitted contact is a margin, resolved per (robot sphere, slot)
    in `clearance.ClearancePolicy`, and a margin of zero still forbids penetration.
    """
    for enabled in (True, False):
        spec = builder.build([_candidate(np.array([0.5, 0.0, 1.0]), enabled=enabled)])
        assert int(spec.active_mask.sum()) == 1, (
            f"collision_enabled={enabled} must not remove the candidate's row"
        )


# ------------------------------------------------------------------------------- overflow


def _active_slots(spec):
    """`[(centre, radius)]` for every active slot — the spheres the NLP actually constrains."""
    lo_pos, _ = spec.layout["pos"]
    lo_r, _ = spec.layout["radius"]
    return [
        (spec.parameter_values[lo_pos + 3 * j : lo_pos + 3 * j + 3], spec.parameter_values[lo_r + j])
        for j in range(spec.max_candidates)
        if spec.active_mask[j] > 0
    ]


def test_slot_overflow_aggregates_conservatively_instead_of_dropping(arm):
    """Overflow geometry is coarsened, never deleted.

    The old behaviour kept the largest `max_candidates` spheres and discarded the rest, which is
    under-approximation: a small sphere is small, not absent, and a trajectory routed through where
    it used to be hits it just as hard.

    The reference is the *same scene with enough slots*, not the radii handed to `_candidate` —
    primitive fitting decides the actual constraint spheres, so comparing against the inputs would
    test the fitter rather than the aggregation.
    """
    candidates = [
        _candidate(np.array([0.4 + 0.2 * i, 0.0, 1.0]), radius=0.02 + 0.01 * i, candidate_id=i)
        for i in range(6)
    ]
    roomy = CFG.with_overrides({"geometry": {"max_candidates": 8}})
    reference = ConstraintBuilder(arm, roomy.constraint, roomy.geometry).build(candidates)
    assert reference.layout["n_overflow_spheres"] == 0

    tight = CFG.with_overrides({"geometry": {"max_candidates": 3}})
    spec = ConstraintBuilder(arm, tight.constraint, tight.geometry).build(candidates)
    assert int(spec.active_mask.sum()) == 3, "every slot is used; none is wasted by aggregating"
    assert spec.layout["n_overflow_spheres"] == 5
    assert spec.layout["n_dropped_spheres"] == 0, "nothing may be dropped any more"

    slots = _active_slots(spec)
    for centre, radius in _active_slots(reference):
        covered = any(
            np.linalg.norm(centre - agg_c) + radius <= agg_r + 1e-9 for agg_c, agg_r in slots
        )
        assert covered, f"sphere at {centre} r={radius:.4f} is not contained by any active slot"


def test_overflow_aggregates_stay_local_rather_than_swallowing_the_workspace(arm):
    """Two reserved slots, not one, because one aggregate over scattered clutter forbids everything.

    With obstacles spread over 1.2 m, a single containing sphere would have a radius over half that
    and block the whole reachable volume. Grouping along the spread's longest axis keeps each
    aggregate near its members. Conservatism is mandatory; uselessness is not.
    """
    candidates = [
        _candidate(np.array([0.4 + 0.25 * i, 0.0, 1.0]), radius=0.03, candidate_id=i)
        for i in range(6)
    ]
    config = CFG.with_overrides({"geometry": {"max_candidates": 3}})
    spec = ConstraintBuilder(arm, config.constraint, config.geometry).build(candidates)
    aggregate_radii = sorted(r for _, r in _active_slots(spec))[1:]  # the two aggregates
    span = 0.25 * 5
    assert all(r < span / 2 for r in aggregate_radii), (
        f"aggregates {aggregate_radii} are as large as the whole scene ({span} m)"
    )


def test_candidate_ids_map_slots_back_to_their_owner(builder):
    spec = builder.build([
        _candidate(np.array([0.5, 0.0, 1.0]), radius=0.06, candidate_id=11),
        _candidate(np.array([0.8, 0.0, 1.0]), radius=0.03, candidate_id=22),
    ])
    active = spec.candidate_ids[spec.active_mask > 0]
    assert set(active.tolist()) == {11, 22}
    assert np.all(spec.candidate_ids[spec.active_mask == 0] == -1)


# ------------------------------------------------------------------------ support surfaces


def test_a_plane_row_is_the_exact_half_space(arm, builder):
    """`n . p - (d + margin) - r_robot`, not an approximation of one."""
    plane = SupportSurface(
        id=0, normal=np.array([0.0, 0.0, 1.0]), offset=0.75,
        point_indices=np.arange(10), point_count=10, rms_error=0.0, safety_margin=0.01,
    )
    q = np.array([-0.6, -0.6])
    spec = builder.build([], [plane])
    h = evaluate(spec, np.tile(q.reshape(-1, 1), (1, builder.horizon)), builder)

    plane_block = h[spec.layout["n_sphere_rows"]:]
    centres, radii = arm.sphere_centers_numeric(q)
    expected = centres[:, 2] - (plane.offset + plane.safety_margin) - radii
    got = plane_block[: builder.n_robot_spheres * builder.max_support_surfaces]
    got = got.reshape(builder.n_robot_spheres, builder.max_support_surfaces)[:, 0]
    assert np.abs(expected - got).max() < 1e-12


def test_planes_do_not_consume_candidate_slots(builder):
    """A table as spheres would swallow every slot; as a half-space it costs one linear row."""
    plane = SupportSurface(
        id=0, normal=np.array([0.0, 0.0, 1.0]), offset=0.75,
        point_indices=np.arange(10), point_count=10, rms_error=0.0, safety_margin=0.01,
    )
    spec = builder.build([_candidate(np.array([0.5, 0.0, 1.0]))], [plane])
    assert int(spec.active_mask.sum()) == 1  # the object only
    assert spec.layout["n_plane_rows"] == builder.horizon * builder.n_robot_spheres * builder.max_support_surfaces


def test_an_unused_plane_slot_is_inert(builder):
    spec = builder.build([], [])
    h = evaluate(spec, np.zeros((builder.nq, builder.horizon)), builder)
    assert np.allclose(h[spec.layout["n_sphere_rows"]:], 1.0)


# ------------------------------------------------------------------------------- bookkeeping


def test_spec_reports_its_own_shape(builder):
    spec = builder.build([])
    assert spec.horizon == HORIZON
    assert spec.max_candidates == CFG.geometry.max_candidates
    assert spec.n_robot_spheres == builder.n_robot_spheres
    assert spec.squared_distance is True
    assert spec.n_constraints == (
        HORIZON * builder.n_robot_spheres * (builder.max_candidates + builder.max_support_surfaces)
    )
    assert np.all(spec.lower_bound == 0.0) and np.all(np.isinf(spec.upper_bound))
    assert spec.layout["row_order"].startswith("(step, robot_sphere, slot)")


def test_build_is_deterministic(builder):
    candidates = [_candidate(np.array([0.5, 0.0, 1.0]), candidate_id=1)]
    a, b = builder.build(candidates), builder.build(candidates)
    assert np.array_equal(a.parameter_values, b.parameter_values)
    assert np.array_equal(a.candidate_ids, b.candidate_ids)


# ------------------------------------------------------------------- the graph must not move


def _graph_shape(spec):
    """Everything a TO holding an `nlpsol` object across frames depends on staying put."""
    return (
        spec.n_constraints,
        spec.parameter_values.shape,
        spec.active_mask.shape,
        spec.candidate_ids.shape,
        tuple(spec.lower_bound.tolist()[:4]),
        spec.layout["n_params"],
        spec.horizon,
        spec.max_candidates,
        spec.n_robot_spheres,
    )


def _sparsity(builder, spec):
    """The Jacobian sparsity in `q` — the thing IPOPT detects once and reuses."""
    import casadi as ca

    Q = ca.SX.sym("Q", builder.nq, builder.horizon)
    jac = ca.jacobian(spec.expr_factory(Q, spec.parameter_symbol), Q)
    rows, cols = jac.sparsity().get_triplet()
    return tuple(rows), tuple(cols)


def test_the_graph_shape_survives_every_phase_and_manipulator(arm):
    """§26.8. Phase and contact authorization are numbers, not structure.

    This is the property that makes the clearance policy affordable at all: a TO builds its solver
    once, and a phase transition mid-trajectory must not invalidate it.
    """
    from AG3S.ag3s.types import ContactPolicyContext, Manipulator, Phase

    builder = ConstraintBuilder(arm, CFG.constraint, CFG.geometry)
    candidates = [_candidate(np.array([0.5, 0.0, 1.0]), candidate_id=3)]

    shapes, sparsities = set(), set()
    for phase in Phase:
        for manipulators in ((), (Manipulator.LEFT,), (Manipulator.RIGHT,),
                             (Manipulator.LEFT, Manipulator.RIGHT)):
            spec = builder.build(
                candidates, context=ContactPolicyContext.make(phase, manipulators)
            )
            shapes.add(_graph_shape(spec))
            sparsities.add(_sparsity(builder, spec))
    assert len(shapes) == 1, f"the graph changed shape across phases/manipulators: {shapes}"
    assert len(sparsities) == 1, "the Jacobian sparsity changed across phases/manipulators"


def test_the_graph_shape_survives_a_changing_scene(arm):
    """Objects appearing and disappearing must move parameters, never rows."""
    builder = ConstraintBuilder(arm, CFG.constraint, CFG.geometry)
    scenes = [
        [],
        [_candidate(np.array([0.5, 0.0, 1.0]), candidate_id=0)],
        [_candidate(np.array([0.4 + 0.1 * i, 0.0, 1.0]), candidate_id=i) for i in range(40)],
    ]
    shapes = {_graph_shape(builder.build(scene)) for scene in scenes}
    assert len(shapes) == 1, f"a changing scene changed the graph: {shapes}"


def test_only_the_parameter_values_differ_between_frames(arm):
    """The positive statement: something *does* change, and it is only the numbers.

    Without this the previous two tests would pass on a builder that ignored its input entirely.
    """
    import dataclasses

    from AG3S.ag3s.constraints.clearance import ClearancePolicy
    from AG3S.ag3s.types import ContactPolicyContext, Manipulator, Phase
    from tests.ag3s.fixtures import FINGERTIP_LINK

    # A TARGET candidate and a policy that names the mock's fingertip: without both, nothing is
    # eligible for relaxation and the assertion below would be vacuous.
    policy = ClearancePolicy.from_config(
        dataclasses.replace(CFG.contact, contact_links={"right": (FINGERTIP_LINK,)}), CFG.geometry
    )
    builder = ConstraintBuilder(arm, CFG.constraint, CFG.geometry, clearance_policy=policy)
    candidate = _candidate(np.array([0.5, 0.0, 1.0]), candidate_id=3)
    candidate.source_type = SourceType.TARGET
    candidates = [candidate]
    transit = builder.build(candidates, context=ContactPolicyContext.make(Phase.TRANSIT))
    grasp = builder.build(
        candidates, context=ContactPolicyContext.make(Phase.GRASP, [Manipulator.RIGHT])
    )
    assert not np.array_equal(transit.parameter_values, grasp.parameter_values), (
        "the phase reached nothing; the previous two tests would then be vacuous"
    )
    assert transit.parameter_symbol is grasp.parameter_symbol
    assert _graph_shape(transit) == _graph_shape(grasp)


def test_refresh_returns_a_vector_the_same_solver_can_consume(arm):
    """The real-time path: `to_casadi` once at setup, `refresh` every frame after."""
    from AG3S.ag3s.constraints.to_adapter import build_constraint_set, refresh

    builder = ConstraintBuilder(arm, CFG.constraint, CFG.geometry)
    first = build_constraint_set(
        builder, candidates=[_candidate(np.array([0.5, 0.0, 1.0]), candidate_id=0)],
        support_surfaces=[], target=None, robot_state=np.zeros(arm.nq), phase="transit",
    )
    values = refresh(
        first, builder, candidates=[_candidate(np.array([0.7, 0.0, 1.0]), candidate_id=1)]
    )
    assert values.shape == first.constraints.parameter_values.shape
    assert not np.array_equal(values, first.constraints.parameter_values)
