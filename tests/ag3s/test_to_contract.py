"""Minimal CasADi/IPOPT harness — does an AG3S constraint set actually solve?

**This is a verification harness, not a trajectory optimizer.** It lives in `tests/` deliberately and
must never move into `ag3s/`. Its cost is a toy (stay near a reference configuration), its model is
the two-link analytic mock, and it has no dynamics, no limits and no chunk interface. All it answers
is the one question AG3S cannot answer about itself: *can a solver consume what this module emits?*

Three properties are checked, and the third is the one that makes the design real-time rather than
merely correct:

1. IPOPT converges on a scene with active obstacles, and the returned trajectory satisfies every
   constraint AG3S emitted.
2. The constraints actually bind — the solution is pushed away from the unconstrained optimum. A
   harness where the obstacles were irrelevant would pass trivially.
3. A **second scene solves through the same solver object with only the parameter vector changed.**
   If the NLP had to be rebuilt per frame, warm-starting would be impossible and the whole
   fixed-slot design would be pointless.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.stages.collision_candidates import generate_candidates
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.constraints.constraint_builder import ConstraintBuilder, evaluate
from AG3S.ag3s.stages.geometry import fit_sphere
from AG3S.ag3s.constraints.to_adapter import build_constraint_set, refresh, to_casadi, warm_start_map
from AG3S.ag3s.types import (
    CollisionCandidate,
    GroundingStatus,
    Phase,
    PointCloud,
    SourceType,
    SupportSurface,
)
from tests.ag3s.fixtures import PlanarTwoLinkArm

ca = pytest.importorskip("casadi", reason="the TO contract is a CasADi contract")

HORIZON = 5
CFG = AG3SConfig().with_overrides({"constraint": {"horizon": HORIZON}})
IPOPT_OPTIONS = {
    "ipopt.print_level": 0,
    "print_time": False,
    "ipopt.sb": "yes",
    "ipopt.max_iter": 300,
    "ipopt.tol": 1e-8,
}


def _obstacle(centre, radius=0.05, candidate_id=0, margin=0.05, enabled=True):
    """A candidate built directly, so the harness tests the contract and not the perception stack."""
    rng = np.random.default_rng(candidate_id)
    points = np.asarray(centre, np.float64) + rng.normal(0.0, radius / 3.0, (40, 3))
    primitive = fit_sphere(points, safety_margin=margin, candidate_id=candidate_id)
    return CollisionCandidate(
        id=candidate_id,
        source_type=SourceType.OBJECT,
        semantic_role="object",
        geometry=[primitive],
        pose=np.eye(4),
        dimensions=np.full(3, radius),
        confidence=1.0,
        safety_margin=margin,
        collision_enabled=enabled,
        contact_permission=False,
        phase_rule="always_on",
        timestamp=1.0,
        centroid=points.mean(axis=0),
        point_indices=np.arange(40),
        point_count=40,
    )


class Harness:
    """`min sum ||q_k - q_ref||^2` subject to the AG3S rows. Nothing else."""

    def __init__(self, builder: ConstraintBuilder):
        self.builder = builder
        self.Q = ca.SX.sym("Q", builder.nq, builder.horizon)

    def solve(self, spec, q_ref, q_init=None):
        fragment = to_casadi(spec, self.Q)
        reference = np.tile(np.asarray(q_ref, np.float64).reshape(-1, 1), (1, self.builder.horizon))
        error = ca.reshape(self.Q, -1, 1) - ca.DM(reference.reshape(-1, 1, order="F"))
        nlp = {
            "x": ca.reshape(self.Q, -1, 1),
            "f": ca.dot(error, error),
            "g": fragment["g"],
            "p": fragment["p"],
        }
        solver = ca.nlpsol("to_contract", "ipopt", nlp, IPOPT_OPTIONS)
        return solver, self._run(solver, fragment, q_ref, q_init)

    def _run(self, solver, fragment, q_ref, q_init=None):
        start = np.tile(
            np.asarray(q_ref if q_init is None else q_init, np.float64).reshape(-1, 1),
            (1, self.builder.horizon),
        )
        result = solver(
            x0=start.reshape(-1, 1, order="F"),
            lbg=fragment["lbg"],
            ubg=fragment["ubg"],
            p=fragment["p_val"],
        )
        status = solver.stats()["return_status"]
        trajectory = np.asarray(result["x"]).reshape(self.builder.nq, self.builder.horizon, order="F")
        return status, trajectory

    def resolve(self, solver, spec, q_ref, q_init=None):
        """Re-run an existing solver with new parameter values — the real-time path."""
        fragment = {
            "lbg": spec.lower_bound, "ubg": spec.upper_bound, "p_val": spec.parameter_values
        }
        return self._run(solver, fragment, q_ref, q_init)


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


@pytest.fixture(scope="module")
def builder(arm):
    return ConstraintBuilder(arm, CFG.constraint, CFG.geometry)


# ------------------------------------------------------------------ 1. the constraints solve


def test_ipopt_converges_and_the_solution_respects_every_constraint(arm, builder, capsys):
    """The headline: AG3S output goes into IPOPT and comes back feasible."""
    q_ref = np.array([-0.60, -0.60])
    blocked = arm.end_effector_numeric(q_ref)
    spec = builder.build([_obstacle(blocked, candidate_id=0)])

    harness = Harness(builder)
    _, (status, trajectory) = harness.solve(spec, q_ref)

    residual = evaluate(spec, trajectory, builder)
    with capsys.disabled():
        print(
            f"\n  IPOPT status         : {status}"
            f"\n  rows                 : {spec.n_constraints}"
            f"\n  active slots         : {int(spec.active_mask.sum())}"
            f"\n  min h at solution    : {residual.min():+.3e}   (>= 0 required)"
            f"\n  |q* - q_ref|         : {np.linalg.norm(trajectory[:, 0] - q_ref):.4f} rad"
        )
    assert status in ("Solve_Succeeded", "Solved_To_Acceptable_Level"), status
    assert residual.min() > -1e-6, f"violated by {residual.min():.3e}"


def test_the_constraints_actually_bind(arm, builder):
    """Without this, convergence would prove nothing — the obstacle has to change the answer."""
    q_ref = np.array([-0.60, -0.60])
    blocked = arm.end_effector_numeric(q_ref)
    harness = Harness(builder)

    _, (_, constrained) = harness.solve(builder.build([_obstacle(blocked, candidate_id=0)]), q_ref)
    _, (_, unconstrained) = harness.solve(builder.build([]), q_ref)

    assert np.linalg.norm(unconstrained[:, 0] - q_ref) < 1e-6  # empty scene => reference is optimal
    assert np.linalg.norm(constrained[:, 0] - q_ref) > 1e-3, "the obstacle did not move the solution"


def test_an_empty_scene_is_feasible_and_leaves_the_reference_alone(builder):
    """Every slot inactive must mean every row satisfied, not an infeasible NLP."""
    q_ref = np.array([0.2, -0.3])
    spec = builder.build([])
    assert int(spec.active_mask.sum()) == 0
    status, trajectory = Harness(builder).solve(spec, q_ref)[1]
    assert status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    assert np.abs(trajectory - q_ref.reshape(-1, 1)).max() < 1e-6


def test_a_support_surface_half_space_also_solves(arm, builder):
    """Planes travel as exact half-spaces, not as spheres, and must solve the same way."""
    q_ref = np.array([-1.20, -0.30])
    tip = arm.end_effector_numeric(q_ref)
    plane = SupportSurface(
        id=0, normal=np.array([0.0, 0.0, 1.0]), offset=float(tip[2] + 0.10),
        point_indices=np.arange(10), point_count=10, rms_error=0.0, safety_margin=0.01,
    )
    spec = builder.build([], [plane])
    status, trajectory = Harness(builder).solve(spec, q_ref)[1]
    assert status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")

    centres, radii = arm.sphere_centers_numeric(trajectory[:, 0])
    clearance = centres[:, 2] - (plane.offset + plane.safety_margin) - radii
    assert clearance.min() > -1e-6, f"a sphere sank {clearance.min():.3e} m into the plane"


# --------------------------------------------------------- 2. the structure is not rebuilt


def test_a_new_scene_solves_through_the_same_solver_object(arm, builder, capsys):
    """The real-time claim, tested rather than asserted.

    One `nlpsol` is constructed. A second, different scene is solved by handing it a new `p` — no new
    symbolic expression, no new solver, no change to the row count or the sparsity pattern.
    """
    q_ref = np.array([-0.60, -0.60])
    first = builder.build([_obstacle(arm.end_effector_numeric(q_ref), candidate_id=0)])
    harness = Harness(builder)
    solver, (status_a, trajectory_a) = harness.solve(first, q_ref)

    second = builder.build([
        _obstacle(arm.end_effector_numeric(q_ref) + np.array([0.03, 0.0, 0.02]), candidate_id=0),
        _obstacle(arm.end_effector_numeric(q_ref) + np.array([-0.10, 0.0, 0.12]), candidate_id=1),
    ])
    assert second.n_constraints == first.n_constraints
    assert second.parameter_values.shape == first.parameter_values.shape
    assert not np.array_equal(second.parameter_values, first.parameter_values)

    status_b, trajectory_b = harness.resolve(solver, second, q_ref, q_init=trajectory_a[:, 0])
    residual = evaluate(second, trajectory_b, builder)
    with capsys.disabled():
        print(
            f"\n  frame 1: {status_a}, active slots {int(first.active_mask.sum())}"
            f"\n  frame 2: {status_b}, active slots {int(second.active_mask.sum())}"
            f"  (same solver object, parameters only)"
            f"\n  min h frame 2        : {residual.min():+.3e}"
        )
    assert status_b in ("Solve_Succeeded", "Solved_To_Acceptable_Level"), status_b
    assert residual.min() > -1e-6
    assert not np.allclose(trajectory_a, trajectory_b)


def test_refresh_produces_the_same_numbers_as_a_rebuild(builder):
    """`refresh` is the per-frame path; it must not diverge from `build`."""
    candidates = [_obstacle(np.array([0.5, 0.0, 1.0]), candidate_id=3)]
    constraint_set = build_constraint_set(
        builder, candidates=candidates, support_surfaces=[], target=None,
        robot_state=np.zeros(builder.nq), phase=Phase.TRANSIT,
        grounding_status=GroundingStatus.NO_SEED,
    )
    assert np.array_equal(
        refresh(constraint_set, builder), constraint_set.constraints.parameter_values
    )


def test_warm_start_map_flags_slot_ownership_changes(builder):
    a = builder.build([_obstacle(np.array([0.5, 0.0, 1.0]), candidate_id=1)])
    b = builder.build([_obstacle(np.array([0.51, 0.0, 1.0]), candidate_id=1)])
    c = builder.build([_obstacle(np.array([0.51, 0.0, 1.0]), candidate_id=9)])
    assert warm_start_map(a, b)[0] and not warm_start_map(a, b)[1]
    assert not warm_start_map(a, c)[0], "a slot that changed owner must not reuse a warm start"
    assert not warm_start_map(None, a).any()


# ---------------------------------------------------------------- 3. contract-level checks


def _grasp_setup(arm):
    """A builder whose contact allowlist names the mock arm's fingertip, plus a target under it.

    The shipped allowlist holds RB-Y1's `ee_finger_*` links, which the two-link mock does not have.
    Renaming them here rather than renaming the mock's links keeps the allowlist mechanism itself
    under test — the policy has to match on a name it was configured with, not on a hard-coded one.
    """
    from AG3S.ag3s.constraints.clearance import ClearancePolicy
    from AG3S.ag3s.types import TargetGeometry
    from tests.ag3s.fixtures import FINGERTIP_LINK

    cfg = CFG.with_overrides({"contact": {"contact_links": {"right": [FINGERTIP_LINK]}}})
    builder = ConstraintBuilder(
        arm, cfg.constraint, cfg.geometry,
        clearance_policy=ClearancePolicy.from_config(cfg.contact, cfg.geometry),
    )
    q_ref = np.array([-0.60, -0.60])
    tip = arm.end_effector_numeric(q_ref)
    cloud = PointCloud(tip + np.random.default_rng(0).normal(0.0, 0.01, (60, 3)))
    target = TargetGeometry(
        id=0, points=cloud.points, point_indices=np.arange(len(cloud)),
        centroid=cloud.points.mean(axis=0),
        bounding_geometry=fit_sphere(cloud.points), attention_score=0.9, confidence=0.9,
    )
    return cfg, builder, q_ref, cloud, target


def test_permitted_contact_still_forbids_penetration(arm):
    """A margin of zero is not a disabled constraint.

    The old contract expressed "the gripper may touch the cup" by dropping the target's rows, so at
    GRASP the solver could put any part of the robot anywhere inside the object. Now the row is
    present with `d_safe = 0` on the authorized link: the surfaces may meet, and the centres may not
    approach closer than `r_robot + r_target`. This test measures that the emitted rows still hold at
    the solution rather than trusting the margin arithmetic.
    """
    from AG3S.ag3s.types import ContactPolicyContext, Manipulator

    cfg, builder, q_ref, cloud, target = _grasp_setup(arm)
    candidates, _ = generate_candidates(
        cloud, phase=Phase.GRASP, target=target,
        config=cfg.collision_candidate, geometry_config=cfg.geometry, contact_config=cfg.contact,
    )
    context = ContactPolicyContext.make(Phase.GRASP, [Manipulator.RIGHT])
    spec = builder.build(candidates, context=context)

    assert int(spec.active_mask.sum()) >= 1, "the target must still occupy a slot at GRASP"
    assert (spec.candidate_ids[spec.active_mask > 0] >= 0).any()

    _, (status, trajectory) = Harness(builder).solve(spec, q_ref)
    assert status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    h = evaluate(spec, trajectory, builder)
    assert h.min() > -1e-6, f"the solution penetrates the target by {-h.min():.2e}"


def test_contact_clearance_reaches_only_the_authorized_link(arm):
    """The margin block must relax one sphere and leave its neighbours alone.

    Read straight out of the parameter vector rather than inferred from behaviour: the `(sphere,
    slot)` layout is the thing that makes link-wise contact expressible, and an off-by-one here would
    hand fingertip clearance to the forearm while every end-to-end test still passed.
    """
    from AG3S.ag3s.types import ContactPolicyContext, Manipulator
    from tests.ag3s.fixtures import FINGERTIP_LINK

    cfg, builder, q_ref, cloud, target = _grasp_setup(arm)
    candidates, _ = generate_candidates(
        cloud, phase=Phase.GRASP, target=target,
        config=cfg.collision_candidate, geometry_config=cfg.geometry, contact_config=cfg.contact,
    )
    spec = builder.build(
        candidates, context=ContactPolicyContext.make(Phase.GRASP, [Manipulator.RIGHT])
    )

    lo_d, _ = spec.layout["d_safe"]
    m = spec.max_candidates
    margins = spec.parameter_values[lo_d : lo_d + builder.n_robot_spheres * m].reshape(-1, m)
    slot = int(np.flatnonzero(spec.active_mask > 0)[0])
    links = builder.sphere_link_names

    for i, link in enumerate(links):
        expected = 0.0 if link == FINGERTIP_LINK else cfg.geometry.safety_margin
        assert margins[i, slot] == pytest.approx(expected), (
            f"sphere {i} on {link!r} got margin {margins[i, slot]}, expected {expected}"
        )


def test_an_unauthorized_manipulator_gets_no_relaxation_at_grasp(arm):
    """GRASP alone authorizes nothing. Someone has to say which hand."""
    from AG3S.ag3s.types import ContactPolicyContext, Manipulator

    cfg, builder, q_ref, cloud, target = _grasp_setup(arm)
    candidates, _ = generate_candidates(
        cloud, phase=Phase.GRASP, target=target,
        config=cfg.collision_candidate, geometry_config=cfg.geometry, contact_config=cfg.contact,
    )
    for context in (
        ContactPolicyContext.make(Phase.GRASP),                       # nobody named
        ContactPolicyContext.make(Phase.GRASP, [Manipulator.LEFT]),   # the other hand
    ):
        spec = builder.build(candidates, context=context)
        lo_d, _ = spec.layout["d_safe"]
        m = spec.max_candidates
        margins = spec.parameter_values[lo_d : lo_d + builder.n_robot_spheres * m]
        assert margins.min() == pytest.approx(cfg.geometry.safety_margin), (
            f"context {context} relaxed a margin it was never authorized to relax"
        )


def test_the_phase_rule_reaches_the_solver(arm):
    """End to end: a phase change must move the trajectory the optimizer returns.

    The old version asserted the robot could sit exactly at `q_ref` during GRASP, which was only
    true because the target's constraint had been deleted. Under the corrected contract the target
    is still there, so the honest statement is a *graded* one: the authorized fingertip may come
    much closer at GRASP than at TRANSIT, and at neither phase may anything penetrate.
    """
    from AG3S.ag3s.types import ContactPolicyContext, Manipulator

    cfg, builder, q_ref, cloud, target = _grasp_setup(arm)
    harness = Harness(builder)
    displacement, rows = {}, set()
    for phase, manipulators in ((Phase.TRANSIT, ()), (Phase.GRASP, [Manipulator.RIGHT])):
        candidates, _ = generate_candidates(
            cloud, phase=phase, target=target,
            config=cfg.collision_candidate, geometry_config=cfg.geometry,
            contact_config=cfg.contact,
        )
        spec = builder.build(candidates, context=ContactPolicyContext.make(phase, manipulators))
        _, (status, trajectory) = harness.solve(spec, q_ref)
        assert status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        assert evaluate(spec, trajectory, builder).min() > -1e-6, f"{phase.value} penetrates"
        displacement[phase] = float(np.linalg.norm(trajectory[:, 0] - q_ref))
        rows.add(spec.n_constraints)

    assert len(rows) == 1, f"a phase change altered the graph shape: {rows}"
    assert displacement[Phase.TRANSIT] > 1e-3, "the target must be avoided while transiting"
    assert displacement[Phase.GRASP] < displacement[Phase.TRANSIT], (
        "authorized contact must let the gripper get closer than transit does: "
        f"{displacement[Phase.GRASP]:.4f} vs {displacement[Phase.TRANSIT]:.4f}"
    )


def test_the_non_squared_form_also_converges(arm):
    """The ablation has to be runnable, not just configurable."""
    config = CFG.with_overrides({"constraint": {"squared_distance": False}})
    builder = ConstraintBuilder(arm, config.constraint, config.geometry)
    q_ref = np.array([-0.60, -0.60])
    spec = builder.build([_obstacle(arm.end_effector_numeric(q_ref), candidate_id=0)])
    assert spec.squared_distance is False

    status, trajectory = Harness(builder).solve(spec, q_ref)[1]
    assert status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    assert evaluate(spec, trajectory, builder).min() > -1e-6


def test_the_constraint_fragment_carries_no_ag3s_types(builder):
    """The boundary: a TO must be able to consume this without importing AG3S."""
    spec = builder.build([_obstacle(np.array([0.5, 0.0, 1.0]), candidate_id=0)])
    fragment = to_casadi(spec, ca.SX.sym("Q", builder.nq, builder.horizon))
    assert isinstance(fragment["g"], ca.SX)
    assert isinstance(fragment["p"], ca.SX)
    assert isinstance(fragment["lbg"], np.ndarray) and isinstance(fragment["ubg"], np.ndarray)
    assert isinstance(fragment["p_val"], np.ndarray)
    assert fragment["g"].shape[0] == spec.n_constraints


def test_the_constraint_is_symbolically_differentiable(builder):
    """No black-box callback: IPOPT must get an exact Jacobian, not a finite difference."""
    spec = builder.build([_obstacle(np.array([0.5, 0.0, 1.0]), candidate_id=0)])
    Q = ca.SX.sym("Q", builder.nq, builder.horizon)
    jacobian = ca.jacobian(spec.expr_factory(Q, spec.parameter_symbol), ca.reshape(Q, -1, 1))
    assert jacobian.shape == (spec.n_constraints, builder.nq * builder.horizon)
    assert jacobian.nnz() > 0
