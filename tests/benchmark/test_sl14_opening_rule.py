"""SL14 KS -- ``--grasp-opening-rule`` (attach on "closed command + settled + opening >= 0.03 + reach"; detach on "open command + opening
rose" or "fully closed"). CPU only.

(a) Off is unchanged: the server argv of every version registered before this one (including the two other SL14 versions) is
    bit-identical (path-independent sha256 measured at 066e4b33, before this flag); the parser / latch config / call record keep their keys.
(b) The rule on the latch: thin-rim grasp attaches; an empty close does not; a far target does not; the evidence checks
    (blocked / not_empty / dropped / manipulated) are not asked; the T34 revocation does not fire; the open command with a rising opening
    detaches to LATCHED; a re-grasp attaches again; PLACED and a locked SEARCHING may attach.
(c) The rule on the real ``SafePolicy._advance_grasp``: the held object leaves the distance field (``ag3s.attached``), the record key.
(d) The version ``-pc-openrule`` and its record dest.
"""
from __future__ import annotations

import hashlib
import json
import types

import numpy as np
import pytest

from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: F401 (fixture)
from tests.benchmark.test_sl12_oracle_panda import MI, PC, _args
from tests.benchmark.test_sl13_oracle_volume import path_independent_argv
from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _lock, _manipulated, _policy
from AG3S.trajopt.grasp_latch import GraspLatch, GraspPhase, GraspSignal, LatchConfig

OR = f"{PC}-openrule"
#: sha256 of ``path_independent_argv`` over the 65 versions registered at 066e4b33 (before ``-openrule``), measured there.
EARLIER_ARGV_SHA256 = "3b724adb1bf6251b7c0f7ceaf4501e5480c4c9dec4be9df7bd9c90072fd852a7"


# ---------------------------------------------------------------------------------------------------- (a) off is unchanged
def test_every_earlier_versions_server_argv_is_bit_identical_including_the_other_sl14_versions():
    from AG3S.adapters.panda import profile as pf

    old = {v: a for v, a in path_independent_argv(pf).items() if v != OR and v not in pf.SL15_NTM_VERSIONS and v not in pf.SL15_SFHAND_VERSIONS and v not in pf.SL16_HVC_VERSIONS}
    assert len(old) == 65
    assert hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == EARLIER_ARGV_SHA256
    assert {f"{PC}-oraclelatch", f"{PC}-rearm"} <= set(old)
    assert not any("--grasp-opening-rule" in a for a in old.values())


def test_off_by_default_in_parser_latch_config_and_record():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    B = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    assert ap.parse_args(B).grasp_opening_rule is False and ap.parse_args(B + ["--grasp-opening-rule"]).grasp_opening_rule is True
    with pytest.raises(SystemExit):      # the RB-Y1 server (default --robot) refuses it
        ss.reject_panda_only_flags(ap, ap.parse_args(B + ["--grasp-opening-rule"]))
    ss.reject_panda_only_flags(ap, ap.parse_args(B))
    assert LatchConfig().opening_rule is False
    from AG3S.adapters.panda.server import panda_latch_config

    assert panda_latch_config().opening_rule is False and panda_latch_config(opening_rule=True).opening_rule is True
    # the latch's evidence and a SafePolicy record carry no opening_rule key when off
    sp, q = _sp(latch=LatchConfig())
    recs = _frames(sp, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715)])
    assert all("opening_rule" not in r and "opening_rule" not in r["evidence"] for r in recs)
    assert recs[-1]["state"] == "held"                       # the evidence path attached as before


# ---------------------------------------------------------------------------------------------------- (b) the rule on the latch
def _sig(cmd, opening, *, closing=None, seq=1):
    steps = (8 if cmd < 0.85 else 0) if closing is None else closing
    return GraspSignal(available=True, feedback_seq=seq, n_steps=8, n_exec=8, closing_steps=steps, last_command=cmd, opening=opening)


def _latch(phase=GraspPhase.LATCHED, **cfg):
    latch = GraspLatch(LatchConfig(opening_rule=True, **cfg), evidence=True)
    for _ in range(3):
        latch.update(label="apple", score=0.9, runner_up=0.1, gripper=1.0)
    assert latch.manipulated == "apple"
    latch.phase = phase
    return latch


def _grasp(latch, opening=0.04, reach=0.05, **kw):
    """An open request, then two closed ones at ``opening`` (the first is not settled yet). Returns the last event."""
    latch.observe_grasp(_sig(1.0, 1.0), reach_m=reach, **kw)
    first = latch.observe_grasp(_sig(0.0, opening), reach_m=reach, **kw)
    assert not first.attach and latch.phase is GraspPhase.CLOSING            # not settled: the opening just moved 0.96
    return latch.observe_grasp(_sig(0.0, opening), reach_m=reach, **kw)


def test_a_thin_rim_grasp_attaches():
    latch = _latch()
    ev = _grasp(latch, opening=0.04)             # blocked (opening - command = 0.04 < 0.05) would call this an empty hand
    assert ev.attach and not ev.detach and latch.phase is GraspPhase.HELD and latch.holding
    assert ev.evidence["opening_rule"] == {"applied": True, "decision": "attach", "opening": 0.04, "last_command": 0.0, "reach_mm": 50.0}
    assert ev.evidence["checks"] == {"closed_command": True, "settled": True, "opening_min": True, "reach": True}


def test_the_evidence_checks_are_not_asked():
    """No manipulated id, state ``lost``, no drop since the open hand: none of them matters."""
    latch = _latch()
    ev = _grasp(latch, opening=0.035, manipulated_id=None, manipulated_state="lost")
    assert ev.attach and latch.phase is GraspPhase.HELD


def test_an_empty_close_does_not_attach():
    latch = _latch()
    ev = _grasp(latch, opening=0.012)
    assert not ev.attach and latch.phase is GraspPhase.CLOSING
    assert ev.evidence["checks"]["opening_min"] is False and ev.evidence["opening_rule"]["decision"] is None


def test_a_far_target_does_not_attach():
    latch = _latch()
    ev = _grasp(latch, opening=0.05, reach=0.30)
    assert not ev.attach and latch.phase is GraspPhase.CLOSING and ev.evidence["checks"]["reach"] is False
    assert ev.evidence["opening_rule"]["reach_mm"] == 300.0
    unknown = _grasp(_latch(), opening=0.05, reach=None)         # no reach = no attach (fail-closed)
    assert not unknown.attach


def test_an_open_command_or_an_unsettled_opening_does_not_attach():
    latch = _latch()
    latch.observe_grasp(_sig(1.0, 1.0), reach_m=0.05)
    ev = latch.observe_grasp(_sig(1.0, 0.04, closing=0), reach_m=0.05)       # settled-looking opening, but the command is open
    ev = latch.observe_grasp(_sig(1.0, 0.04, closing=0), reach_m=0.05)
    assert not ev.attach and ev.evidence["checks"]["closed_command"] is False and ev.evidence["checks"]["settled"] is True
    moving = _latch()
    moving.observe_grasp(_sig(0.0, 0.07, closing=8), reach_m=0.05)
    ev = moving.observe_grasp(_sig(0.0, 0.04, closing=8), reach_m=0.05)       # 0.03 apart >= 0.02
    assert not ev.attach and ev.evidence["checks"]["settled"] is False


def test_a_hold_chunk_keeps_the_last_executed_command():
    """A HOLD request executes nothing (``last_command`` None): the command of the last executed step still counts."""
    latch = _latch()
    latch.observe_grasp(_sig(1.0, 1.0), reach_m=0.05)
    latch.observe_grasp(_sig(0.0, 0.04), reach_m=0.05)
    hold = GraspSignal(available=True, feedback_seq=3, n_steps=8, n_exec=0, closing_steps=0, last_command=None, opening=0.04)
    ev = latch.observe_grasp(hold, reach_m=0.05)
    assert ev.attach and ev.evidence["opening_rule"]["last_command"] == 0.0


def test_the_attach_revocation_does_not_fire_under_the_rule():
    latch = _latch()
    _grasp(latch, opening=0.04)
    assert latch.holding
    far = {"centre_distance_m": 0.5, "held_radius_m": 0.04, "hand_rise_m": 0.1, "object_rise_m": 0.0}   # observed_elsewhere + not_following
    ev = latch.observe_grasp(_sig(0.0, 0.04, seq=4), reach_m=0.05, held_observation=far)
    assert not ev.detach and not ev.revoked and latch.holding
    assert ev.evidence["opening_rule"]["decision"] is None and "revoke" not in ev.evidence
    # the evidence placed-success and a long closed hold do not release either
    ev = latch.observe_grasp(_sig(0.0, 0.04, seq=5), reach_m=0.05, placed=True)
    assert not ev.detach and latch.holding


def test_an_open_command_with_a_rising_opening_detaches_to_latched():
    latch = _latch()
    _grasp(latch, opening=0.04)
    ev = latch.observe_grasp(_sig(1.0, 0.055, closing=0, seq=4), reach_m=0.05)   # rose by 0.015 only: still held
    assert not ev.detach and latch.holding
    ev = latch.observe_grasp(_sig(1.0, 0.07, closing=0, seq=5), reach_m=0.05)    # 0.07 > 0.04 + 0.02
    assert ev.detach and not ev.attach and not ev.revoked and not ev.slipped
    assert latch.phase is GraspPhase.LATCHED and not latch.holding and latch.manipulated == "apple"
    assert ev.evidence["opening_rule"]["decision"] == "detach"
    # the open hand does not attach again, and a closed command with a wide opening does not detach
    ev = latch.observe_grasp(_sig(1.0, 0.07, closing=0, seq=6), reach_m=0.05)
    assert not ev.attach and latch.phase is GraspPhase.LATCHED
    wide = _latch()
    _grasp(wide, opening=0.04)
    ev = wide.observe_grasp(_sig(0.0, 0.09, seq=4), reach_m=0.05)               # closed command, opening larger: no detach
    assert not ev.detach and wide.holding


def test_a_fully_closed_hand_detaches_and_a_regrasp_attaches():
    latch = _latch()
    _grasp(latch, opening=0.04)
    ev = latch.observe_grasp(_sig(0.0, 0.012, seq=4), reach_m=0.05)             # closed on nothing
    assert ev.detach and latch.phase is GraspPhase.LATCHED
    # open, then grasp the rim again: LATCHED -> CLOSING -> HELD by the same rule
    latch.observe_grasp(_sig(1.0, 1.0, closing=0, seq=5), reach_m=0.05)
    latch.observe_grasp(_sig(0.0, 0.05, seq=6), reach_m=0.05)
    ev = latch.observe_grasp(_sig(0.0, 0.05, seq=7), reach_m=0.05)
    assert ev.attach and latch.holding
    # the detach after an open release -> LATCHED -> a re-grasp, with nothing in between that needs the rearm flag
    rel = _latch()
    _grasp(rel, opening=0.04)
    assert rel.observe_grasp(_sig(1.0, 0.1, closing=0, seq=4), reach_m=0.05).detach
    assert rel.phase is GraspPhase.LATCHED
    rel.observe_grasp(_sig(0.0, 0.05, seq=5), reach_m=0.05)
    assert rel.observe_grasp(_sig(0.0, 0.05, seq=6), reach_m=0.05).attach


def test_placed_and_a_locked_searching_may_attach():
    for phase in (GraspPhase.PLACED, GraspPhase.SEARCHING):
        latch = _latch(phase)
        latch.observe_grasp(_sig(1.0, 1.0), reach_m=0.05)
        latch.observe_grasp(_sig(0.0, 0.04), reach_m=0.05)
        ev = latch.observe_grasp(_sig(0.0, 0.04), reach_m=0.05)
        assert ev.attach and latch.holding, phase
    unlocked = GraspLatch(LatchConfig(opening_rule=True), evidence=True)     # SEARCHING without any lock: nothing to attach
    unlocked.observe_grasp(_sig(1.0, 1.0), reach_m=0.05)
    unlocked.observe_grasp(_sig(0.0, 0.04), reach_m=0.05)
    assert not unlocked.observe_grasp(_sig(0.0, 0.04), reach_m=0.05).attach


def test_without_feedback_the_state_does_not_move():
    latch = _latch()
    ev = latch.observe_grasp(GraspSignal(available=False, reason="no feedback"), reach_m=0.05)
    assert not ev.attach and not ev.detach and "opening_rule" not in ev.evidence


def test_flag_off_the_old_evidence_path_still_refuses_the_thin_rim():
    """The same signals with the flag off: ``blocked`` (opening - command >= 0.05) says an empty hand."""
    latch = GraspLatch(LatchConfig(), evidence=True)
    for _ in range(3):
        latch.update(label="apple", score=0.9, runner_up=0.1, gripper=1.0)
    ev = _grasp(latch, opening=0.04, manipulated_id=1, manipulated_state="visible")
    assert not ev.attach and latch.phase is GraspPhase.CLOSING and "opening_rule" not in ev.evidence


# ---------------------------------------------------------------------------------------------------- (c) on the SafePolicy
def _sp(distance=0.05, **kw):
    sp = _policy(**kw)
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(_finger_origin(sp, q) + np.array([distance, 0.0, 0.0]))  # noqa: SLF001
    _lock(sp)
    return sp, q


def _frames(sp, q, feedbacks):
    obs = [types.SimpleNamespace(timestamp=1.0, robot_state=q)]
    out = []
    for fb in feedbacks:
        sp._pending = {"scene": {}, "exec_feedback": fb}                       # noqa: SLF001
        sp._advance_grasp(obs, ["left"])                                       # noqa: SLF001
        sp._grasp_constraint_version = sp._grasp_state_version                 # noqa: SLF001
        sp._finish_grasp_record(types.SimpleNamespace(attached=sp.ag3s.attached))  # noqa: SLF001
        out.append(dict(sp._last_grasp_record))                                # noqa: SLF001
    return out


def test_the_policy_attaches_a_thin_rim_and_records_the_decision():
    sp, q = _sp(latch=LatchConfig(opening_rule=True))
    # opening 0.04 with a 0.0 command: gap 0.04 < blocked_gap 0.05 -> the evidence path would refuse
    recs = _frames(sp, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.04), _fb(3, [0.0] * 8, 0.04)])
    assert [r["opening_rule"]["decision"] for r in recs] == [None, None, "attach"]
    assert recs[2]["state"] == "held" and recs[2]["attach"] is True and sp.ag3s.attached is not None   # the object left the field
    assert set(recs[2]["opening_rule"]) == {"applied", "decision", "opening", "last_command", "reach_mm"}
    assert recs[2]["opening_rule"]["opening"] == pytest.approx(0.04) and 40.0 <= recs[2]["opening_rule"]["reach_mm"] <= 60.0
    # open command + rising opening: detach -> the object is a scene object again, the latch is LATCHED
    rel = _frames(sp, q, [_fb(4, [1.0] * 8, 0.07)])[0]
    assert rel["opening_rule"]["decision"] == "detach" and rel["detach"] is True and rel["state"] == "latched"
    assert sp.ag3s.attached is None


def test_the_policy_does_not_attach_an_empty_close_or_a_far_target():
    empty, q = _sp(latch=LatchConfig(opening_rule=True))
    recs = _frames(empty, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.012), _fb(3, [0.0] * 8, 0.012)])
    assert recs[-1]["state"] == "closing" and empty.ag3s.attached is None and recs[-1]["opening_rule"]["decision"] is None
    far, q = _sp(distance=0.30, latch=LatchConfig(opening_rule=True))
    recs = _frames(far, q, [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.04), _fb(3, [0.0] * 8, 0.04)])
    assert recs[-1]["state"] == "closing" and far.ag3s.attached is None
    assert recs[-1]["opening_rule"]["reach_mm"] > 120.0


# ---------------------------------------------------------------------------------------------------- (d) the version
def test_the_version_is_pc_plus_one_flag(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf, server as ps

    reg = pf.AG3S_VERSIONS
    assert list(reg[OR]) == list(reg[PC]) + ["--grasp-opening-rule"] and OR in pf.SL14_VERSIONS and pf.is_final_version(OR)
    assert pf.version_server_argv(OR) == pf.version_server_argv(PC)
    assert pf.version_server_flags(OR)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]          # a probe-taking pc variant
    assert pf.version_server_flags(OR, hsr6_probe="/p.npz")[-2:] == ["--hsr6-probe", "/p.npz"]
    assert not pf.needs_oracle_grasp(OR) and not pf.needs_oracle_segmentation(OR) and f"{OR}-lsfork" not in reg
    a = _args(OR, monkeypatch)
    ps.check_unported_flags(a, None)
    assert a.grasp_opening_rule is True and a.oracle_latch is False and a.placed_rearm is False
    assert pf.match_ag3s_version(a) == OR
    base = _args(PC, monkeypatch)
    vals = pf.version_flag_values(a)
    assert vals["grasp_opening_rule"] is True
    assert {k: v for k, v in vals.items() if k not in ("grasp_opening_rule", "oracle_latch", "placed_rearm")} == pf.version_flag_values(base)
    bv = pf.version_flag_values(base)
    assert "grasp_opening_rule" not in bv and pf.match_ag3s_version(base) == PC
    for v in (MI, f"{PC}-gap", f"{PC}-rearm", f"{PC}-oraclelatch"):   # earlier versions' records gain no key
        assert "grasp_opening_rule" not in pf.version_flag_values(_args(v, monkeypatch))
