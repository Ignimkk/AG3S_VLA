"""The held object: does it move with the hand, and is it still constrained against everything else?

After a grasp closes, the crate is no longer an obstacle in the world — it is part of the moving
robot, and the thing that must not hit the table. Checking only robot-versus-environment leaves the
object itself unconstrained, which is how a robot lifts a box cleanly and then sweeps it through a
shelf.

Four properties carry the weight here, and three of them are about what does *not* happen:

* the object's pose follows symbolic FK, so the optimizer can differentiate its motion;
* only the explicitly allowlisted links are excused from colliding with it — the forearm, the torso
  and the opposite arm are not;
* perception failure never detaches it;
* attach and detach do not change the graph.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmark.ag3s.constraints.attached import (
    attach_from_target,
    attached_spheres,
    base_frame_spheres,
    self_collision_mask,
)
from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.constraints.constraint_builder import ConstraintBuilder, evaluate
from benchmark.ag3s.stages.geometry import fit_sphere
from benchmark.ag3s.runtime.pipeline import AG3S
from benchmark.ag3s.types import (
    AttachedCollisionGeometry,
    ConstraintValidity,
    PipelineStatus,
    PointCloud,
    TargetGeometry,
)
from tests.ag3s.fixtures import DEFAULT_ARM_Q, FINGERTIP_LINK, PlanarTwoLinkArm, make_scene

ca = pytest.importorskip("casadi", reason="attached geometry is a CasADi contract")

HORIZON = 3
CFG = AG3SConfig().with_overrides(
    {"constraint": {"horizon": HORIZON, "max_attached_primitives": 2}}
)
PARENTS = (FINGERTIP_LINK,)
MOVED_Q = np.array([-0.20, -0.90])


@pytest.fixture(scope="module")
def arm():
    return PlanarTwoLinkArm()


def _target(arm, q=DEFAULT_ARM_Q, radius=0.04):
    """A small object sitting exactly at the fingertip, as a just-closed grasp would leave it."""
    tip = arm.end_effector_numeric(q)
    points = tip + np.random.default_rng(0).normal(0.0, radius / 3.0, (60, 3))
    return TargetGeometry(
        id=7, points=points, point_indices=np.arange(60), centroid=points.mean(axis=0),
        bounding_geometry=fit_sphere(points), attention_score=0.9, confidence=0.9,
    )


def _attach(arm, q=DEFAULT_ARM_Q, allowed=(FINGERTIP_LINK,)):
    return attach_from_target(
        _target(arm, q), robot_model=arm, robot_state=q,
        parent_link=FINGERTIP_LINK, allowed_contact_links=allowed, label="crate",
    )


def _builder(arm, parents=PARENTS):
    return ConstraintBuilder(arm, CFG.constraint, CFG.geometry, attached_parent_links=parents)


# ------------------------------------------------------------------------ transform direction


def test_the_attach_transform_round_trips_at_the_grasp_configuration(arm):
    """`T_parent_object = inv(T_base_parent(q_grasp)) @ T_base_object`, checked by going back.

    Getting the direction backwards produces an object that swings the wrong way and still looks
    plausible in a plot, so the convention is pinned numerically rather than by naming.
    """
    attached = _attach(arm)
    target = _target(arm)
    recovered = base_frame_spheres(attached, arm, DEFAULT_ARM_Q)
    assert len(recovered) == 1
    centre, radius = recovered[0]
    assert np.allclose(centre, target.bounding_geometry.center, atol=1e-12)
    assert radius == pytest.approx(float(target.bounding_geometry.dimensions[0]))


def test_the_object_travels_with_its_parent_link(arm):
    """Move the arm and the object must move with it — by roughly the fingertip's own displacement."""
    attached = _attach(arm)
    at_grasp = base_frame_spheres(attached, arm, DEFAULT_ARM_Q)[0][0]
    moved = base_frame_spheres(attached, arm, MOVED_Q)[0][0]

    tip_shift = np.linalg.norm(
        arm.end_effector_numeric(MOVED_Q) - arm.end_effector_numeric(DEFAULT_ARM_Q)
    )
    object_shift = np.linalg.norm(moved - at_grasp)
    assert object_shift > 0.05, "the object did not follow the hand"
    assert object_shift == pytest.approx(tip_shift, abs=0.02), (
        f"the object moved {object_shift:.3f} m while the fingertip moved {tip_shift:.3f} m"
    )


def test_the_symbolic_pose_matches_the_numeric_one_and_depends_on_q(arm):
    """The optimizer differentiates the held object's motion; a constant pose would be silent.

    A numeric pose baked in at attach time would satisfy every collision test at the configuration it
    was taken in, and be wrong at every other point on the trajectory — including all the ones the
    optimizer is choosing between.
    """
    attached = _attach(arm)
    q = ca.SX.sym("q", arm.nq)
    T = arm.link_pose_symbolic(q, FINGERTIP_LINK)
    local, _ = attached_spheres(attached)[0]
    centre = ca.mtimes(T[:3, :3], ca.DM(local.reshape(3, 1))) + T[:3, 3]
    f = ca.Function("c", [q], [centre])

    for q_val in (DEFAULT_ARM_Q, MOVED_Q):
        expected = base_frame_spheres(attached, arm, q_val)[0][0]
        assert np.abs(np.asarray(f(q_val)).ravel() - expected).max() < 1e-12

    jac = ca.Function("J", [q], [ca.jacobian(centre, q)])
    assert float(np.abs(np.asarray(jac(DEFAULT_ARM_Q))).max()) > 1e-6, (
        "the attached pose does not depend on q; it was baked in as a constant"
    )


# ----------------------------------------------------------------- the contact allowlist


def test_only_allowlisted_links_are_excused_from_colliding_with_the_object(arm):
    """§23.1. The fingers holding it are excused; nothing else is.

    A global self-collision disable would be the easy version and would also let the object pass
    through the forearm it is being carried past.
    """
    attached = _attach(arm, allowed=(FINGERTIP_LINK,))
    mask = self_collision_mask(attached, arm.sphere_link_names)

    for i, link in enumerate(arm.sphere_link_names):
        expected = 0.0 if link == FINGERTIP_LINK else 1.0
        assert mask[i] == expected, f"sphere {i} on {link!r} got mask {mask[i]}"
    assert mask.sum() == len(arm.sphere_link_names) - 1


def test_an_unknown_allowlist_entry_excuses_nothing(arm):
    """Fail closed. A misspelled link name must not quietly disable a real constraint."""
    attached = _attach(arm, allowed=("ee_finger_typo", "link_that_does_not_exist"))
    mask = self_collision_mask(attached, arm.sphere_link_names)
    assert mask.min() == 1.0, "an unrecognised link name excused a robot sphere"


def test_an_empty_allowlist_constrains_the_object_against_the_whole_robot(arm):
    attached = _attach(arm, allowed=())
    assert self_collision_mask(attached, arm.sphere_link_names).min() == 1.0


# ---------------------------------------------------------------- collisions that must hold


def _rows_at(builder, spec, q, block: str) -> np.ndarray:
    """The attached block of the constraint vector, evaluated along a constant trajectory."""
    values = evaluate(spec, np.tile(np.asarray(q, np.float64).reshape(-1, 1), (1, builder.horizon)), builder)
    total = builder.n_constraints
    attached_rows = builder.n_attached_rows
    tail = values[total - attached_rows :]
    per_step = attached_rows // builder.horizon
    m, k, s = builder.max_candidates, builder.max_support_surfaces, builder.n_robot_spheres
    a = builder.max_attached
    step0 = tail[:per_step]
    # Layout inside one (parent, slot): candidates, then planes, then robot spheres.
    width = m + k + s
    out = []
    for slot in range(a):
        chunk = step0[slot * width : (slot + 1) * width]
        out.append({"candidate": chunk[:m], "plane": chunk[m : m + k], "robot": chunk[m + k :]}[block])
    return np.concatenate(out)


def test_the_held_object_is_constrained_against_the_support_plane(arm):
    """§23's "held object vs support surface". The crate must not be driven through the table."""
    from benchmark.ag3s.types import SupportSurface

    builder = _builder(arm)
    attached = _attach(arm)
    tip_z = float(arm.end_effector_numeric(DEFAULT_ARM_Q)[2])

    # A plane just below the held object: satisfied here, violated if the object were lowered.
    low = SupportSurface(id=0, normal=np.array([0.0, 0.0, 1.0]), offset=tip_z - 0.10,
                         point_indices=np.arange(4), point_count=4, rms_error=1e-4)
    high = SupportSurface(id=0, normal=np.array([0.0, 0.0, 1.0]), offset=tip_z + 0.10,
                          point_indices=np.arange(4), point_count=4, rms_error=1e-4)

    clear = _rows_at(builder, builder.build([], [low], attached=attached), DEFAULT_ARM_Q, "plane")
    through = _rows_at(builder, builder.build([], [high], attached=attached), DEFAULT_ARM_Q, "plane")
    assert clear.min() > 0.0, "a comfortably clear plane reported a violation"
    assert through.min() < 0.0, "the object sitting below the plane reported no violation"


def test_the_held_object_is_constrained_against_an_environment_candidate(arm):
    """§23's "held object vs environment"."""
    from benchmark.ag3s.types import CollisionCandidate, SourceType

    builder = _builder(arm)
    attached = _attach(arm)
    tip = arm.end_effector_numeric(DEFAULT_ARM_Q)

    def _obstacle(centre):
        points = np.asarray(centre) + np.random.default_rng(1).normal(0.0, 0.01, (40, 3))
        return CollisionCandidate(
            id=1, source_type=SourceType.OBJECT, semantic_role="object",
            geometry=[fit_sphere(points, safety_margin=0.0)], pose=np.eye(4),
            dimensions=np.full(3, 0.03), confidence=1.0, safety_margin=0.0,
            collision_enabled=True, contact_permission=False, phase_rule="always_on",
            timestamp=1.0, centroid=points.mean(axis=0), point_indices=np.arange(40), point_count=40,
        )

    far = _rows_at(builder, builder.build([_obstacle(tip + np.array([0.5, 0.0, 0.0]))],
                                          attached=attached), DEFAULT_ARM_Q, "candidate")
    overlapping = _rows_at(builder, builder.build([_obstacle(tip)], attached=attached),
                           DEFAULT_ARM_Q, "candidate")
    assert far.min() > 0.0
    assert overlapping.min() < 0.0, "an obstacle inside the held object reported no violation"


def test_the_held_object_is_constrained_against_non_allowlisted_robot_links(arm):
    """§23's opposite arm / torso / same-arm-non-contact cases, all one mechanism.

    With the fingertip allowlisted the object's own row against that sphere is switched off, and the
    rows against every other sphere stay live. The measurement is the count of violated rows: an
    object placed on top of the arm must trip the links it overlaps and not the one holding it.
    """
    builder = _builder(arm)
    # A deliberately oversized object, big enough to swallow several of the arm's own spheres.
    target = _target(arm, radius=0.04)
    fat = dataclasses.replace(
        target.bounding_geometry, dimensions=np.full(3, 0.35)
    )
    attached = AttachedCollisionGeometry(
        parent_link=FINGERTIP_LINK,
        T_parent_object=np.eye(4),
        primitives=[dataclasses.replace(fat, center=np.zeros(3))],
        allowed_contact_links=frozenset({FINGERTIP_LINK}),
    )
    rows = _rows_at(builder, builder.build([], [], attached=attached), DEFAULT_ARM_Q, "robot")
    violated = int((rows < 0.0).sum())
    assert violated > 0, "an object engulfing the arm violated nothing"

    # The allowlisted sphere's row is inactive, so it can never be among the violations.
    fingertip_indices = [i for i, n in enumerate(arm.sphere_link_names) if n == FINGERTIP_LINK]
    per_slot = builder.n_robot_spheres
    for slot in range(builder.max_attached):
        for i in fingertip_indices:
            assert rows[slot * per_slot + i] > 0.0, (
                "the allowlisted fingertip's row was enforced anyway"
            )


# --------------------------------------------------------------- the graph must not move


def test_attach_and_detach_leave_the_graph_identical(arm):
    """§25. Attaching is a parameter change, never a rebuild.

    Row count, bounds and the parameter vector's length all have to match, because a TO holds an
    `nlpsol` object built once and a warm start indexed by row.
    """
    builder = _builder(arm)
    attached = _attach(arm)

    empty = builder.build([], [])
    held = builder.build([], [], attached=attached)
    released = builder.build([], [])

    for other in (held, released):
        assert other.n_constraints == empty.n_constraints
        assert other.parameter_values.shape == empty.parameter_values.shape
        assert np.array_equal(other.lower_bound, empty.lower_bound)
        assert np.array_equal(other.upper_bound, empty.upper_bound)
    # Bound methods compare unequal by identity even when they wrap the same function on the same
    # object, so the comparison has to go through `__func__`/`__self__`.
    for other in (held, released):
        assert other.expr_factory.__func__ is empty.expr_factory.__func__
        assert other.expr_factory.__self__ is empty.expr_factory.__self__
    assert empty.parameter_symbol is held.parameter_symbol

    lo, _ = builder.layout["attached_active"]
    n = builder.max_attached * len(builder.attached_parent_links)
    assert held.parameter_values[lo : lo + n].max() == 1.0
    assert released.parameter_values[lo : lo + n].max() == 0.0, (
        "a stale active flag survived detach; the object would still be constrained"
    )


def test_an_inactive_attached_slot_contributes_no_gradient(arm):
    """Detached rows must be constants, or the solver still pays for an object that is gone."""
    builder = _builder(arm)
    spec = builder.build([], [])
    Q = ca.SX.sym("Q", builder.nq, builder.horizon)
    g = spec.expr_factory(Q, spec.parameter_symbol)
    jac = ca.Function("J", [Q, spec.parameter_symbol], [ca.jacobian(g, Q)])
    rows = np.asarray(jac(np.tile(DEFAULT_ARM_Q.reshape(-1, 1), (1, builder.horizon)),
                          spec.parameter_values))
    tail = rows[builder.n_constraints - builder.n_attached_rows :]
    assert np.abs(tail).max() < 1e-12, "an inactive attached row still has a gradient"


def test_declaring_no_parent_links_costs_nothing(arm):
    """A caller who never attaches anything gets exactly the graph they had before."""
    plain = ConstraintBuilder(arm, CFG.constraint, CFG.geometry)
    assert plain.n_attached_rows == 0
    assert plain.max_attached == 0
    with_slots = _builder(arm)
    assert with_slots.n_constraints > plain.n_constraints


def test_attaching_to_an_undeclared_parent_link_is_refused(arm):
    """The parent is part of the symbolic expression, so it cannot be chosen at attach time."""
    builder = _builder(arm, parents=("link_0",))
    with pytest.raises(ValueError, match="reserved slots for"):
        builder.build([], [], attached=_attach(arm))

    plain = ConstraintBuilder(arm, CFG.constraint, CFG.geometry)
    with pytest.raises(ValueError, match="reserved no attached slots"):
        plain.build([], [], attached=_attach(arm))


def test_an_object_too_complex_for_its_slots_is_incomplete(arm):
    """Held geometry has no conservative fallback — an aggregate riding on the hand engulfs the hand."""
    from benchmark.ag3s.constraints.to_adapter import build_constraint_set

    builder = _builder(arm)
    base = _attach(arm)
    many = AttachedCollisionGeometry(
        parent_link=FINGERTIP_LINK, T_parent_object=np.eye(4),
        primitives=base.primitives * 5,  # ten spheres into two slots
        allowed_contact_links=base.allowed_contact_links,
    )
    constraint_set = build_constraint_set(
        builder, candidates=[], support_surfaces=[], target=None,
        robot_state=DEFAULT_ARM_Q, phase="grasp", attached=many,
    )
    assert constraint_set.validity is ConstraintValidity.INCOMPLETE
    assert constraint_set.status is PipelineStatus.GEOMETRY_INCOMPLETE
    assert any("attached-object sphere" in note for note in constraint_set.notes)


# --------------------------------------------------------- attach state is externally driven


def test_perception_failure_does_not_detach(arm):
    """§24. The robot does not put an object down because it stopped being able to see it."""
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    ag3s = AG3S(CFG, robot_model=arm, attached_parent_links=PARENTS)
    ag3s.attach(_attach(arm))
    assert ag3s.attached is not None

    # A frame with no attention at all: grounding fails, and the held object is untouched.
    constraint_set = ag3s.process(
        attention_map=None, depth=scene.depth, camera_intrinsics=scene.camera_intrinsics,
        T_base_cam=scene.T_base_cam, robot_state=DEFAULT_ARM_Q, phase="grasp",
        image_hw=scene.hw,
    )
    assert constraint_set.target is None
    assert ag3s.attached is not None
    assert constraint_set.attached is ag3s.attached

    # An empty frame — total sensor loss — likewise.
    ag3s.process(
        depth=np.zeros((40, 40)),
        camera_intrinsics=np.array([[80.0, 0, 20.0], [0, 80.0, 20.0], [0, 0, 1.0]]),
        T_base_cam=np.eye(4), robot_state=DEFAULT_ARM_Q, phase="grasp",
    )
    assert ag3s.attached is not None, "a blank frame detached the held object"


def test_only_an_explicit_detach_releases_the_object(arm):
    ag3s = AG3S(CFG, robot_model=arm, attached_parent_links=PARENTS)
    held = ag3s.attach(_attach(arm))
    assert ag3s.detach() is held
    assert ag3s.attached is None
    assert ag3s.detach() is None, "detaching twice must be harmless"


def test_ag3s_never_attaches_by_itself(arm):
    """There is no grasp detector. `Phase.GRASP` alone attaches nothing."""
    scene = make_scene(robot_model=arm, robot_state=DEFAULT_ARM_Q)
    ag3s = AG3S(CFG, robot_model=arm, attached_parent_links=PARENTS)
    ag3s.process(
        attention_map=scene.attention_grid, depth=scene.depth,
        camera_intrinsics=scene.camera_intrinsics, T_base_cam=scene.T_base_cam,
        robot_state=DEFAULT_ARM_Q, phase="grasp", image_hw=scene.hw,
    )
    assert ag3s.attached is None


def test_attaching_a_target_needs_the_configuration_the_grasp_closed_in(arm):
    """A later `q` bakes in however far the arm has moved, and the offset then travels forever."""
    ag3s = AG3S(CFG, robot_model=arm, attached_parent_links=PARENTS)
    with pytest.raises(ValueError, match="robot_state"):
        ag3s.attach(_target(arm))

    attached = ag3s.attach(
        _target(arm), robot_state=DEFAULT_ARM_Q, parent_link=FINGERTIP_LINK,
        allowed_contact_links=(FINGERTIP_LINK,), label="crate",
    )
    assert attached.parent_link == FINGERTIP_LINK
    assert attached.source_candidate_id == 7
    assert attached.permits_contact_with(FINGERTIP_LINK)
    assert not attached.permits_contact_with("link_0")


def test_reset_drops_the_attached_object(arm):
    """An episode boundary is an explicit statement that the task is over — unlike a bad frame."""
    ag3s = AG3S(CFG, robot_model=arm, attached_parent_links=PARENTS)
    ag3s.attach(_attach(arm))
    ag3s.reset()
    assert ag3s.attached is None
