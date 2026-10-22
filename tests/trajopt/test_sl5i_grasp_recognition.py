"""SL5i — V11AB-pb6 (grasp recognition: sticky reach, drop from open) and V11AB-pb7 (closing hand exemption).

All three switches are off by default; off must be the pre-SL5i behaviour. Numbers in the scenarios are the SL5g pb5-C
cases the task names (LI t2e4: reach 91.1 mm on entering CLOSING, 123 mm after a later capture; LIBERO t5e0: latched with
the fingers already at 0.072, open was ~1.0).
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.grasp_latch import GraspLatch, GraspPhase, LatchConfig
from tests.trajopt.test_grasp_evidence import (
    _ag3s,
    _fb,
    _finger_origin,
    _grasp_frames,
    _latched,
    _lock,
    _manipulated,
    _observe,
    _policy,
    _signal,
    _target_at,
    arm,  # noqa: F401  (fixture)
)

SL5I_EVIDENCE_KEYS = ("reach_sticky", "drop_from_open")


# --- defaults ------------------------------------------------------------------------------------------
def test_switches_default_off_and_leave_no_record_keys():
    cfg = LatchConfig()
    assert cfg.reach_sticky is False and cfg.drop_from_open is False
    latch = _latched()
    for i, (applied, measured) in enumerate([([1.0] * 8, 1.0), ([0.0] * 8, 0.715), ([0.0] * 8, 0.715)]):
        ev = _observe(latch, _signal(i + 1, applied, measured))
        assert not any(k in ev.evidence for k in SL5I_EVIDENCE_KEYS)
    assert ev.attach
    sp = _policy()
    assert sp.closing_hand_exempt is False
    assert sp._grasp_contact(types.SimpleNamespace(contact_context=None)) is None      # noqa: SLF001


# --- (a) sticky reach --------------------------------------------------------------------------------------
def _t2e4_like(latch, reaches):
    """open -> closing (moving, reach[0]) -> stopped (reach[1]) -> stopped (reach[2])."""
    out = [_observe(latch, _signal(1, [1.0] * 8, 1.0), reach=reaches[0])]
    out.append(_observe(latch, _signal(2, [0.0] * 8, 0.715), reach=reaches[0]))   # enters CLOSING, still moving
    out.append(_observe(latch, _signal(3, [0.0] * 8, 0.715), reach=reaches[1]))   # stopped, re-measured
    out.append(_observe(latch, _signal(4, [0.0] * 8, 0.715), reach=reaches[2]))
    return out


def test_off_a_later_capture_keeps_reach_failed():
    evs = _t2e4_like(_latched(), (0.0911, 0.123, 0.123))
    assert evs[1].phase is GraspPhase.CLOSING and evs[1].evidence["checks"]["reach"] is True
    assert [k for k, v in evs[2].evidence["checks"].items() if not v] == ["reach"]
    assert not any(e.attach for e in evs)


def test_sticky_keeps_the_pass_within_the_attempt_and_records_both_values():
    latch = _latched(reach_sticky=True)
    evs = _t2e4_like(latch, (0.0911, 0.123, 0.123))
    assert [e.attach for e in evs] == [False, False, True, False]
    st = evs[2].evidence["reach_sticky"]
    assert st["used"] is True and st["now"] is False and st["now_mm"] == pytest.approx(123.0)
    assert st["passed"]["reach_mm"] == pytest.approx(91.1) and st["passed"]["feedback_seq"] == 2
    assert evs[2].evidence["reach_mm"] == pytest.approx(123.0)          # the measured value stays in the record
    assert "reach sticky 91.1 mm" in evs[2].note
    assert evs[1].evidence["reach_sticky"]["used"] is False             # passed on its own


def test_sticky_needs_a_pass_first_and_the_same_object():
    latch = _latched(reach_sticky=True)
    evs = _t2e4_like(latch, (0.123, 0.123, 0.123))
    assert not any(e.attach for e in evs) and evs[-1].evidence["reach_sticky"]["passed"] is None
    latch = _latched(reach_sticky=True)
    _observe(latch, _signal(1, [1.0] * 8, 1.0), reach=0.09)
    _observe(latch, _signal(2, [0.0] * 8, 0.715), reach=0.09, mid=0)
    ev = _observe(latch, _signal(3, [0.0] * 8, 0.715), reach=0.123, mid=5)          # another object
    assert not ev.attach and ev.evidence["checks"]["reach"] is False
    assert ev.evidence["reach_sticky"]["same_object"] is False


def test_sticky_ends_with_the_attempt():
    latch = _latched(reach_sticky=True)
    _observe(latch, _signal(1, [1.0] * 8, 1.0), reach=0.09)
    ev = _observe(latch, _signal(2, [0.0] * 8, 0.715), reach=0.09)
    assert ev.phase is GraspPhase.CLOSING and ev.evidence["reach_sticky"]["passed"] is not None
    ev = _observe(latch, _signal(3, [1.0] * 8, 1.0), reach=0.123)                    # reopened -> LATCHED
    assert ev.phase is GraspPhase.LATCHED
    ev = _observe(latch, _signal(4, [0.0] * 8, 0.715), reach=0.123)                  # new attempt
    ev = _observe(latch, _signal(5, [0.0] * 8, 0.715), reach=0.123)
    assert not ev.attach and ev.evidence["reach_sticky"]["passed"] is None
    latch.reset()
    assert latch._reach_pass is None                                                  # noqa: SLF001


def test_sticky_survives_a_rejected_attach_inside_the_attempt():
    latch = _latched(reach_sticky=True)
    _observe(latch, _signal(1, [1.0] * 8, 1.0), reach=0.0911)
    _observe(latch, _signal(2, [0.0] * 8, 0.715), reach=0.0911)
    assert _observe(latch, _signal(3, [0.0] * 8, 0.715), reach=0.123).attach
    latch.revert_attach("inadmissible (test)")
    assert latch.phase is GraspPhase.CLOSING
    ev = _observe(latch, _signal(5, [0.0] * 8, 0.715), reach=0.125)
    assert ev.attach and ev.evidence["reach_sticky"]["used"]


# --- (b) drop from open ------------------------------------------------------------------------------------
def _t5e0_like(**cfg):
    """open while SEARCHING, the fingers close before the target is locked, then the attempt starts at 0.072."""
    latch = GraspLatch(LatchConfig(**cfg), evidence=True)
    _observe(latch, _signal(1, [1.0] * 5, 1.0))
    _observe(latch, _signal(2, [1.0] * 5, 1.0))
    _observe(latch, _signal(3, [0.0] * 5, 0.30))      # closing while SEARCHING: no attempt
    _observe(latch, _signal(4, [0.0] * 5, 0.0719))
    for _ in range(3):
        latch.update(label="bowl", score=0.9, runner_up=0.1, gripper=0.07)
    assert latch.phase is GraspPhase.LATCHED
    evs = [_observe(latch, _signal(5, [0.0] * 5, 0.0594)),
           _observe(latch, _signal(6, [0.0] * 5, 0.060))]
    return latch, evs


def test_off_a_late_latch_measures_drop_from_the_closed_opening():
    _, evs = _t5e0_like(empty_below=0.05)
    assert evs[0].phase is GraspPhase.CLOSING
    assert evs[1].evidence["opening_before"] == pytest.approx(0.0719)
    assert [k for k, v in evs[1].evidence["checks"].items() if not v] == ["dropped"]
    assert not any(e.attach for e in evs)


def test_drop_from_open_uses_the_last_open_request():
    _, evs = _t5e0_like(empty_below=0.05, drop_from_open=True)
    ev = evs[0]                                                          # the first stopped request already holds
    assert ev.attach and ev.evidence["opening_before"] == pytest.approx(1.0)
    basis = ev.evidence["drop_from_open"]
    assert basis["source"] == "last_open" and basis["transition_opening"] == pytest.approx(0.0719)
    assert basis["last_open"]["feedback_seq"] == 2 and basis["last_open"]["command"] == pytest.approx(1.0)


def test_drop_from_open_needs_command_and_opening_open_and_falls_back():
    # never open: the transition value stays (fail-closed)
    latch = _latched(drop_from_open=True)
    _observe(latch, _signal(1, [1.0] * 8, 0.80))                          # opening not open
    ev = _observe(latch, _signal(2, [0.0] * 8, 0.715))
    assert ev.evidence["drop_from_open"]["source"] == "transition"
    assert ev.evidence["opening_before"] == pytest.approx(0.80)
    # an open command with a closed measured opening is not "open"
    latch = _latched(drop_from_open=True)
    _observe(latch, _signal(1, [1.0] * 8, 1.0))
    _observe(latch, _signal(2, [1.0] * 8, 0.80))
    ev = _observe(latch, _signal(3, [0.0] * 8, 0.715))
    assert ev.evidence["drop_from_open"]["last_open"]["feedback_seq"] == 1


# --- (a) at AG3S's attach gate --------------------------------------------------------------------------
def test_attach_gate_takes_the_latch_pass(arm):  # noqa: F811
    from AG3S.ag3s.constraints.attached import AttachRejected
    from tests.ag3s.fixtures import DEFAULT_ARM_Q, FINGERTIP_LINK

    ag3s = _ag3s(arm)
    tip = arm.end_effector_numeric(DEFAULT_ARM_Q)
    far = _target_at(tip + np.array([0.13, 0.0, 0.0]))
    with pytest.raises(AttachRejected):
        ag3s.attach(far, robot_state=DEFAULT_ARM_Q, parent_link=FINGERTIP_LINK, reach=0.12)
    with pytest.raises(AttachRejected):                                   # no pass record: the gate is unchanged
        ag3s.attach(far, robot_state=DEFAULT_ARM_Q, parent_link=FINGERTIP_LINK, reach=0.12,
                    reach_sticky={"passed": None, "used": False})
    sticky = {"passed": {"reach_mm": 91.1, "feedback_seq": 6, "manipulated_id": 1, "context": {"geometry_frame": 5}},
              "used": True, "same_object": True}
    got = ag3s.attach(far, robot_state=DEFAULT_ARM_Q, parent_link=FINGERTIP_LINK, reach=0.12, reach_sticky=sticky)
    assert ag3s.attached is got
    gate = ag3s._held_record["reach_gate"]                                # noqa: SLF001
    assert gate["over_limit"] and gate["decided_by"] == "latch_pass" and gate["latch_pass"]["reach_mm"] == 91.1
    assert gate["dist_mm"] > 120.0
    ag3s.detach()
    ag3s.attach(_target_at(tip + np.array([0.03, 0.0, 0.0])), robot_state=DEFAULT_ARM_Q,
                parent_link=FINGERTIP_LINK, reach=0.12)
    assert "reach_gate" not in ag3s._held_record                          # noqa: SLF001


def _sticky_policy_run(sp):
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    near = _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0])
    far = _finger_origin(sp, q) + np.array([0.13, 0.0, 0.0])
    sp.ag3s._manipulated = _manipulated(near)                                         # noqa: SLF001
    _lock(sp)
    recs = list(_grasp_frames(sp, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715)]))
    sp.ag3s._manipulated = _manipulated(far)                                          # noqa: SLF001  later capture
    recs += list(_grasp_frames(sp, q, [_fb(3, [0.0] * 8, 0.715)]))
    return recs


def test_safe_policy_off_the_later_capture_blocks_the_grasp():
    sp = _policy()
    recs = _sticky_policy_run(sp)
    assert sp.ag3s.attached is None and recs[-1]["state"] == "closing"
    assert recs[-1]["evidence"]["checks"]["reach"] is False


def test_safe_policy_sticky_latch_and_attach_gate_agree():
    sp = _policy()
    sp.set_latch_options(reach_sticky=True)
    recs = _sticky_policy_run(sp)
    assert sp.ag3s.attached is not None and recs[-1]["state"] == "held" and recs[-1]["rejected"] is None
    st = recs[-1]["evidence"]["reach_sticky"]
    assert st["used"] and st["passed"]["context"]["geometry_frame"] == 4
    assert sp.ag3s._held_record["reach_gate"]["decided_by"] == "latch_pass"           # noqa: SLF001


# --- ② closing hand exemption ---------------------------------------------------------------------------
def _closing_with(sp, checks, phase=GraspPhase.CLOSING):
    from AG3S.trajopt.grasp_latch import LatchEvent

    sp._latch.phase = phase                                                         # noqa: SLF001
    sp._latch._last_grasp = LatchEvent(phase=phase, evidence={"checks": dict(checks)})  # noqa: SLF001


ALL = {"settled": True, "closed": True, "dropped": False, "not_empty": True, "blocked": True,
       "manipulated": True, "reach": False}


def test_closing_exempt_condition():
    sp = _policy()
    sp.closing_hand_exempt = True
    _closing_with(sp, ALL)
    assert sp.closing_exempt_now()
    for k in ("closed", "not_empty", "blocked"):
        _closing_with(sp, {**ALL, k: False})
        assert not sp.closing_exempt_now(), k
    for ph in (GraspPhase.LATCHED, GraspPhase.HELD, GraspPhase.SEARCHING):
        _closing_with(sp, ALL, phase=ph)
        assert not sp.closing_exempt_now(), ph
    _closing_with(sp, ALL)
    gc = sp._grasp_contact(types.SimpleNamespace(contact_context=None))             # noqa: SLF001
    assert gc["closing_exempt"] is True and gc["latch"] == "closing" and gc["held_exempt"] is False
    sp.closing_hand_exempt = False
    assert sp._grasp_contact(types.SimpleNamespace(contact_context=None)) is None   # noqa: SLF001


def _gc(latch, **kw):
    from tests.trajopt.test_pb45_grasp_contact import _gc as base

    extra = {k: kw.pop(k) for k in list(kw) if k == "closing_exempt"}
    return {**base(latch, **kw), **extra}


def test_closing_exempt_drops_only_the_authorized_rows_on_the_manipulated_object():
    from tests.trajopt.test_pb45_grasp_contact import ROWS, _pair

    off, on, lin = _pair(_gc("closing", closing_exempt=True), mask=(False,) * 4)
    assert on[ROWS["apple"]] > off[ROWS["apple"]] + 0.9
    for name in ("bar", "table", "box", "held"):
        assert on[ROWS[name]] == off[ROWS[name]], name
    assert lin._last_grasp_contact["closing_exempt_rows"] == 1                       # noqa: SLF001
    off, on, _ = _pair(_gc("closing", closing_exempt=True, auth=(False,) * 4), mask=(False,) * 4)
    assert np.array_equal(off, on)


@pytest.mark.parametrize("latch,flag", [("closing", False), ("held", True), ("latched", True)])
def test_closing_exempt_waits_for_its_condition(latch, flag):
    from tests.trajopt.test_pb45_grasp_contact import _pair

    off, on, lin = _pair(_gc(latch, closing_exempt=flag), mask=(False,) * 4)
    assert np.array_equal(off, on)
    assert lin._last_grasp_contact["closing_exempt_rows"] == 0                       # noqa: SLF001


def test_closing_exempt_key_absent_keeps_the_pb5_record():
    from tests.trajopt.test_pb45_grasp_contact import _pair

    _, _, lin = _pair(_gc("closing", zone=0.010))
    assert "closing_exempt_rows" not in lin._last_grasp_contact                      # noqa: SLF001


# --- profile / CLI ------------------------------------------------------------------------------------------
def test_pb6_pb7_profiles_and_flags():
    from AG3S.adapters.panda import profile
    from AG3S.trajopt import serve_safe as ss

    pb5 = profile.version_server_flags("V11AB-pb5")
    pb6 = profile.version_server_flags("V11AB-pb6")
    pb7 = profile.version_server_flags("V11AB-pb7")
    assert pb6 == pb5 + ["--grasp-reach-sticky", "--grasp-drop-from-open"]
    assert pb7 == pb6 + ["--closing-hand-exempt"]

    def applied(version):
        args = profile.parse_v11ab_args(argv=profile.v11ab_server_argv() + profile.version_server_flags(version))
        sp = _policy()
        return sp, ss.apply_grasp_contact_flags(sp, args)

    sp, out = applied("V11AB-pb5")
    assert set(out) == {"latch_phase", "held_hand_exempt", "grasp_zone_allowance_m", "grasp_zone_radius_m"}
    assert sp._latch.config == LatchConfig() and not sp.closing_hand_exempt            # noqa: SLF001
    sp, out = applied("V11AB-pb6")
    assert out["latch_reach_sticky"] and out["latch_drop_from_open"] and "closing_hand_exempt" not in out
    assert sp._latch.config.reach_sticky and sp._latch.config.drop_from_open            # noqa: SLF001
    assert sp._latch._manipulated._cfg is sp._latch.config                              # noqa: SLF001
    sp, out = applied("V11AB-pb7")
    assert out["closing_hand_exempt"] and sp.closing_hand_exempt and sp._latch.config.reach_sticky  # noqa: SLF001
