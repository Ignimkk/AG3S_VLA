"""V11AB-pb4 — the contact phase from the grasp latch (`SafePolicy.latch_phase`)."""
from __future__ import annotations

import numpy as np

from AG3S.trajopt.grasp_latch import GraspPhase
from tests.trajopt.test_grasp_evidence import _policy


def test_v11ab_uses_the_requested_phase_in_every_latch_state():
    sp = _policy()
    for ph in GraspPhase:
        sp._latch.phase = ph                                               # noqa: SLF001
        assert sp._contact_phase("approach") == "approach"                 # noqa: SLF001
    assert "contact_phase" not in (sp._to_block() or {})                   # noqa: SLF001


def test_pb4_maps_the_latch_to_v11ab_phase_rules():
    sp = _policy()
    sp.latch_phase = True
    expect = {GraspPhase.LATCHED: "pre_grasp", GraspPhase.CLOSING: "grasp", GraspPhase.HELD: "grasp"}
    for ph in GraspPhase:
        sp._latch.phase = ph                                               # noqa: SLF001
        got = sp._contact_phase("approach")                                # noqa: SLF001
        assert got == expect.get(ph, "approach"), ph
        rec = sp._last_contact_phase                                       # noqa: SLF001
        assert rec["applied"] == got and rec["requested"] == "approach"
        assert rec["source"] == ("latch" if ph in expect else "request")


def test_pb4_grasp_phase_relaxes_only_authorized_links_on_the_target():
    from AG3S.ag3s.constraints.clearance import ClearancePolicy
    from AG3S.ag3s.types import SourceType

    pol = ClearancePolicy.from_config()
    left = sorted(pol.contact.links_for("left"))[0]
    m = lambda phase, src, link: pol.lookup(phase=phase, candidate_source=src, robot_link=link,
                                            active_manipulators=["left"])
    assert m("approach", SourceType.TARGET, left) > m("pre_grasp", SourceType.TARGET, left) > m("grasp", SourceType.TARGET, left)
    assert m("grasp", SourceType.TARGET, left) == pol.contact.contact_margin
    assert m("grasp", SourceType.TARGET, "some_forearm") == m("approach", SourceType.TARGET, "some_forearm")
    assert m("grasp", SourceType.SUPPORT_SURFACE, left) == m("approach", SourceType.SUPPORT_SURFACE, left)
    assert m("grasp", SourceType.OBJECT, left) == m("approach", SourceType.OBJECT, left)


def test_grasp_contact_authorized_mask_is_the_clearance_policy_set():
    """pb4/pb5 fix: the mask is the constraint builder's authorized links (palm + fingers), not the latch's grasp links."""
    import types

    sp = _policy()
    sp.held_hand_exempt = True
    cs = types.SimpleNamespace(contact_context=None)
    want = set(sp._authorized_links(cs))                                    # noqa: SLF001
    gc = sp._grasp_contact(cs)                                              # noqa: SLF001
    names = list(sp.constraint_robot_model.sphere_link_names)
    assert {n for n, a in zip(names, gc["authorized"]) if a} == {n for n in names if n in want}
    sp.held_hand_exempt = False
    assert sp._grasp_contact(cs) is None                                    # noqa: SLF001
