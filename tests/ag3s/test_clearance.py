"""What `ClearancePolicy` is allowed to relax, and — mostly — what it is not.

The policy replaced a contract that expressed permitted contact by deleting the target's constraint
rows. Deleting rows is not a smaller margin; it is no constraint at all, and for the duration of a
grasp it let every part of the robot pass through the object being grasped. So the tests here are
weighted toward the negative direction: the great majority assert that some margin did **not** move.

The one thing that may be relaxed is a successfully grounded `TARGET`, against a link that appears in
the allowlist of a manipulator an external context has named, in a phase whose rule permits it. Break
any link of that chain and the answer is the full margin.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from benchmark.ag3s.constraints.clearance import UNKNOWN_LINK, ClearancePolicy, resolve_link_names
from benchmark.ag3s.config import AG3SConfig
from benchmark.ag3s.types import (
    ContactPolicyContext,
    Manipulator,
    Phase,
    SourceType,
)
from tests.ag3s.fixtures import FINGERTIP_LINK, PlanarTwoLinkArm

CFG = AG3SConfig()
FULL = CFG.geometry.safety_margin

#: A stand-in RB-Y1 link set, so the tests can name a forearm and a torso rather than talking about
#: sphere indices. These are the URDF's own names, checked against
#: `src/rby1_description/models/rby1a/urdf/model.urdf`.
RIGHT_FINGERTIP = "ee_finger_r1"
RIGHT_GRIPPER = "ee_right"
RIGHT_FOREARM = "link_right_arm_5"
LEFT_FINGERTIP = "ee_finger_l1"
LEFT_GRIPPER = "ee_left"
TORSO = "link_torso_3"


@pytest.fixture
def policy() -> ClearancePolicy:
    return ClearancePolicy.from_config(CFG.contact, CFG.geometry, CFG.support_surface)


def _ask(policy, phase, link, *, source=SourceType.TARGET, manipulators=(Manipulator.RIGHT,),
         grounded=True) -> float:
    return policy.lookup(
        phase=phase,
        candidate_source=source,
        robot_link=link,
        active_manipulators=manipulators,
        target_grounded=grounded,
    )


# ------------------------------------------------------------------ the one permitted relaxation


def test_a_right_grasp_relaxes_the_right_gripper_links(policy):
    """The whole point of the mechanism, stated positively once."""
    for link in (RIGHT_FINGERTIP, RIGHT_GRIPPER):
        assert _ask(policy, Phase.GRASP, link) == pytest.approx(CFG.contact.contact_margin)


def test_clearance_on_the_authorized_link_shrinks_monotonically_toward_the_grasp(policy):
    """TRANSIT -> APPROACH -> PRE_GRASP -> GRASP, on the fingertip and nowhere else.

    The graded approach is real and useful; what makes it safe is its *scope*. The companion
    assertion — that the forearm's margin is flat across the same four phases — is the reason this
    test exists in this form rather than as a bare monotonicity check.
    """
    phases = (Phase.TRANSIT, Phase.APPROACH, Phase.PRE_GRASP, Phase.GRASP)
    authorized = [_ask(policy, p, RIGHT_FINGERTIP) for p in phases]
    forearm = [_ask(policy, p, RIGHT_FOREARM) for p in phases]

    assert authorized == sorted(authorized, reverse=True), authorized
    assert authorized[0] == pytest.approx(FULL), "TRANSIT must not relax anything"
    assert authorized[-1] == pytest.approx(CFG.contact.contact_margin)
    assert forearm == [pytest.approx(FULL)] * 4, (
        f"the forearm's clearance moved with the phase: {forearm}"
    )


# ---------------------------------------------------------------- everything that must not relax


@pytest.mark.parametrize("link", [RIGHT_FOREARM, LEFT_FINGERTIP, LEFT_GRIPPER, TORSO, UNKNOWN_LINK])
@pytest.mark.parametrize("phase", list(Phase))
def test_a_right_grasp_never_relaxes_any_other_link(policy, phase, link):
    """The forearm, the opposite arm's gripper, the torso, and anything unrecognised.

    Under the old contract every one of these was unconstrained against the target at GRASP, because
    the candidate itself had been switched off. There was no link dimension to get right.
    """
    assert _ask(policy, phase, link) == pytest.approx(FULL)


@pytest.mark.parametrize(
    "source",
    [SourceType.OBJECT, SourceType.UNKNOWN_GEOMETRY, SourceType.OVERFLOW,
     SourceType.RESTRICTED_REGION],
)
@pytest.mark.parametrize("phase", list(Phase))
def test_non_target_sources_are_never_relaxed_on_any_link(policy, phase, source):
    """Only a grounded target is ever eligible. Everything else keeps full clearance always.

    `OVERFLOW` matters most here: it is geometry AG3S could not describe individually, so it is the
    last thing that should inherit a relaxation meant for an object the policy actually identified.
    """
    for link in (RIGHT_FINGERTIP, RIGHT_GRIPPER, RIGHT_FOREARM, TORSO):
        assert _ask(policy, phase, link, source=source) == pytest.approx(FULL)


def test_grounding_failure_relaxes_nothing(policy):
    """§26.2's first case. No target means no target clearance, whatever the phase claims.

    A pipeline that relaxed on `phase == GRASP` alone would hand contact permission to a frame in
    which grounding had just failed — precisely the frame where the robot least knows what is in
    front of it.
    """
    for phase in Phase:
        for link in (RIGHT_FINGERTIP, RIGHT_GRIPPER):
            assert _ask(policy, phase, link, grounded=False) == pytest.approx(FULL)


@pytest.mark.parametrize(
    "manipulators",
    [(), (Manipulator.LEFT,), ("not-an-arm",), (None,)],
    ids=["nobody", "the-other-hand", "garbage-string", "none"],
)
def test_a_missing_or_invalid_manipulator_gets_the_full_margin(policy, manipulators):
    """Fail closed. `Phase.GRASP` says a grasp is happening, not which hand is doing it."""
    assert _ask(policy, Phase.GRASP, RIGHT_FINGERTIP, manipulators=manipulators) == pytest.approx(FULL)


def test_an_unnameable_source_is_treated_as_solid_geometry(policy):
    """An unrecognised source type is a bug somewhere upstream, and bugs are not permission."""
    assert _ask(policy, Phase.GRASP, RIGHT_FINGERTIP, source="something_new") == pytest.approx(FULL)
    assert policy.full_margin("something_new") == pytest.approx(FULL)


def test_the_phase_aware_ablation_pins_every_link_in_every_phase():
    """`phase_aware=False` makes grasping impossible, which is exactly what it is for."""
    contact = dataclasses.replace(CFG.contact, phase_aware=False)
    blind = ClearancePolicy.from_config(contact, CFG.geometry, CFG.support_surface)
    for phase in Phase:
        for link in (RIGHT_FINGERTIP, RIGHT_GRIPPER, TORSO):
            assert _ask(blind, phase, link) == pytest.approx(FULL)


# ------------------------------------------------------------------------------- bimanual + misc


def test_a_bimanual_grasp_authorizes_both_allowlists_and_no_more(policy):
    """The set-valued contract, so a second arm needs no new code path."""
    both = (Manipulator.LEFT, Manipulator.RIGHT)
    for link in (RIGHT_FINGERTIP, LEFT_FINGERTIP, RIGHT_GRIPPER, LEFT_GRIPPER):
        assert _ask(policy, Phase.GRASP, link, manipulators=both) == pytest.approx(
            CFG.contact.contact_margin
        )
    for link in (RIGHT_FOREARM, TORSO, UNKNOWN_LINK):
        assert _ask(policy, Phase.GRASP, link, manipulators=both) == pytest.approx(FULL)


def test_support_surfaces_carry_their_own_margin(policy):
    """A table is fitted far more precisely than a cluster, so it gets its own number — never the
    target's relaxation."""
    assert policy.full_margin(SourceType.SUPPORT_SURFACE) == pytest.approx(
        CFG.support_surface.safety_margin
    )
    assert _ask(policy, Phase.GRASP, RIGHT_FINGERTIP, source=SourceType.SUPPORT_SURFACE) == (
        pytest.approx(CFG.support_surface.safety_margin)
    )


def test_a_zero_contact_margin_is_a_clearance_not_a_disabled_constraint(policy):
    """The value permitted at GRASP is a real number the constraint row still enforces.

    `0.0` says the surfaces may meet. It does not say the centres may coincide: the row is
    ``||p_r - p_c||^2 >= (r_r + r_c + 0)^2``, so penetration is still forbidden. The distinction is
    the entire reason this is a margin rather than a flag.
    """
    assert _ask(policy, Phase.GRASP, RIGHT_FINGERTIP) == 0.0
    assert CFG.contact.contact_margin >= 0.0


# --------------------------------------------------------------------------- the margin matrix


def test_the_margin_matrix_agrees_with_the_scalar_lookup():
    """The vectorized path is on the per-frame hot path; the scalar one is the readable definition.

    On RB-Y1 the matrix is 61 x 32 and a Python call per cell would be too slow, so there are two
    implementations. Two implementations of a safety rule is exactly the situation that needs a test
    saying they agree.
    """
    arm = PlanarTwoLinkArm()
    policy = ClearancePolicy.from_config(
        dataclasses.replace(CFG.contact, contact_links={"right": (FINGERTIP_LINK,)}),
        CFG.geometry,
        CFG.support_surface,
    )
    links = arm.sphere_link_names
    sources = [SourceType.TARGET, SourceType.OBJECT, SourceType.OVERFLOW, SourceType.TARGET]
    context = ContactPolicyContext.make(Phase.GRASP, [Manipulator.RIGHT])

    matrix = policy.margin_matrix(links, sources, context=context)
    assert matrix.shape == (len(links), len(sources))
    for i, link in enumerate(links):
        for j, source in enumerate(sources):
            expected = policy.lookup(
                phase=Phase.GRASP, candidate_source=source, robot_link=link,
                active_manipulators=[Manipulator.RIGHT],
            )
            assert matrix[i, j] == pytest.approx(expected), f"cell ({link}, {source}) disagrees"


def test_the_matrix_relaxes_exactly_one_sphere_of_the_mock_arm():
    """Only the distal sphere is named `FINGERTIP_LINK`; its three neighbours share `link_1`.

    A per-*link* implementation would relax all four and still pass an end-to-end test. Counting the
    relaxed cells is what distinguishes the two.
    """
    arm = PlanarTwoLinkArm()
    policy = ClearancePolicy.from_config(
        dataclasses.replace(CFG.contact, contact_links={"right": (FINGERTIP_LINK,)}),
        CFG.geometry,
        CFG.support_surface,
    )
    matrix = policy.margin_matrix(
        arm.sphere_link_names,
        [SourceType.TARGET],
        context=ContactPolicyContext.make(Phase.GRASP, [Manipulator.RIGHT]),
    )
    relaxed = np.flatnonzero(matrix[:, 0] < FULL)
    assert relaxed.size == 1, f"expected one relaxed sphere, got {relaxed.size}"
    assert arm.sphere_link_names[int(relaxed[0])] == FINGERTIP_LINK


# --------------------------------------------------------------------------- link-name plumbing


def test_a_model_without_link_names_degrades_to_unknown_rather_than_raising():
    """Losing grasp capability is a survivable degradation; losing obstacle avoidance is not.

    An older `RobotCollisionModel` predating `sphere_link_names` still self-filters and still avoids
    obstacles. It simply cannot express contact, because every one of its spheres reports
    `UNKNOWN_LINK`, which no allowlist contains.
    """

    class LegacyModel:
        nq = 2

        def sphere_centers_numeric(self, q):
            return np.zeros((3, 3)), np.full(3, 0.05)

        def sphere_centers_symbolic(self, q):
            return []

    names = resolve_link_names(LegacyModel(), 3)
    assert names == (UNKNOWN_LINK,) * 3
    policy = ClearancePolicy.from_config(CFG.contact, CFG.geometry)
    assert _ask(policy, Phase.GRASP, names[0]) == pytest.approx(FULL)


def test_a_short_link_name_list_is_padded_rather_than_silently_truncating_the_robot():
    """A model whose metadata drifted out of sync must not leave spheres unaddressed."""

    class DriftedModel:
        nq = 2
        sphere_link_names = ("link_0",)

        def sphere_centers_numeric(self, q):
            return np.zeros((4, 3)), np.full(4, 0.05)

        def sphere_centers_symbolic(self, q):
            return []

    names = resolve_link_names(DriftedModel(), 4)
    assert len(names) == 4
    assert names == ("link_0", UNKNOWN_LINK, UNKNOWN_LINK, UNKNOWN_LINK)


def test_describe_reports_what_a_context_actually_authorizes(policy):
    """Diagnostics must show the two numbers side by side, not just the relaxed one."""
    record = policy.describe(ContactPolicyContext.make(Phase.GRASP, [Manipulator.RIGHT]))
    assert record["contact_permission"] is True
    assert set(record["authorized_links"]) == set(CFG.contact.links_for("right"))
    assert record["target_margin_authorized"] == pytest.approx(CFG.contact.contact_margin)
    assert record["target_margin_other"] == pytest.approx(FULL)

    nobody = policy.describe(ContactPolicyContext.make(Phase.GRASP))
    assert nobody["contact_permission"] is False
    assert nobody["authorized_links"] == []
    assert nobody["target_margin_authorized"] == pytest.approx(FULL)
