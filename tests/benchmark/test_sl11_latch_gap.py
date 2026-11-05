"""SL11 section 5 -- the grasp latch measures the opening as the two-finger gap (``--gripper-gap-opening``, Panda only). CPU.

What this file pins:

(a) Off is bit-identical. The new ``LatchConfig`` fields default to "off"; a fixed synthetic request sequence through the
    default latch gives the same events as on main 67613cf1 (golden sha256 measured there with the unchanged latch); the Panda
    latch config, the server and the request obs are untouched; no earlier version's registry entry or record key changes.
(b) The measure. ``gripper_gap_norm`` = ``clip((q1 - q2) / 0.08)`` from the MuJoCo qpos; the single finger value is unchanged.
(c) The two documented misjudgments. Fingers pushed to one side together (q1 -0.0005, q2 -0.0228 -> gap 22 mm): the old
    opening reads 0.012 and the latch calls it a slip (PLACED), the gap opening (0.279) keeps it HELD. A thin rim (gap 2 mm)
    with the object present (observed at the hand) stays HELD; a rim held at 3-5 mm attaches.
(d) A true empty close is still detected: the closed stop (gap 1.0 mm), a closure on a 2.4 mm sliver with nothing near, an HSR6
    assist with no reach evidence, and a hold that closes to the stop (revoke / slip).
(e) Versions ``-pc-gap`` / ``-pc-mi10-gap``; every other version unchanged; launchers accept the names; RB-Y1 refuses the flag.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from AG3S.trajopt.grasp_latch import GraspLatch, GraspPhase, GraspSignal, LatchConfig
from tests.benchmark.test_sl6a_panda_final import ROOT, _Stub, gpu_ok  # noqa: F401 (fixture)

PC = "AG3S-final-panda-v3-e0to-pc"
MI = f"{PC}-mi10"
GAP = f"{PC}-gap"
MIGAP = f"{PC}-mi10-gap"
NEW = (GAP, MIGAP)

#: sha256 of ``json.dumps({name: list(flags)}, sort_keys=True)`` over AG3S_VERSIONS before SL11 (51 versions); the same constant as
#: ``test_sl11_pass_versions`` (the three SL11 pass versions and the -lsfork ones are added on top of it).
OLD_REGISTRY_SHA256 = "c5f2473207811c4d71292450cf611451a02f94b2f1b8b3875cbfb9378d73479e"

#: sha256 over the events of :func:`_golden_events` through the default Panda latch, measured on main 67613cf1 with the unchanged
#: ``grasp_latch.py`` (before the SL11 section 5 fields existed).
GOLDEN_EVENTS_SHA256 = "28832a58be19caf7a0abf7f2a3d7df66c49a04ab211c7d82a9b65c479dd9013d"


# ---------------------------------------------------------------------------------------------------- helpers
def _sig(opening, command=0.0, closing=5, n_exec=5, seq=0):
    return GraspSignal(available=True, feedback_seq=seq, n_steps=5, n_exec=n_exec, closing_steps=closing,
                       last_command=command, opening=opening)


def _latch(cfg):
    latch = GraspLatch(cfg, evidence=True, external_destination=True)
    latch.phase = GraspPhase.LATCHED
    return latch


def _step(latch, opening, command=0.0, seq=0, closing=5, reach=0.05, manip_state="visible", **kw):
    return latch.observe_grasp(_sig(opening, command, closing=closing, seq=seq), reach_m=reach, manipulated_id=1,
                               manipulated_state=manip_state, **kw)


def _close_on(latch, opening, n=4, stop_on_attach=False, **kw):
    """Open (1.0), then close and settle at ``opening``: returns the events (up to the attach with ``stop_on_attach``)."""
    evs = [_step(latch, 1.0, command=1.0, seq=0, closing=0)]
    evs.append(_step(latch, 0.5, seq=1))
    for k in range(n):
        evs.append(_step(latch, opening, seq=2 + k, **kw))
        if stop_on_attach and evs[-1].attach:
            break
    return evs


def _golden_events(cfg):
    """A fixed synthetic request sequence through the single-finger latch (the pc options on): approach, close on a rim, hold,
    revoke window, slip, release, an HSR6 assist -- the paths the section 5 fields touch."""
    out = []

    def add(ev):
        d = dataclasses.asdict(ev) if dataclasses.is_dataclass(ev) else dict(ev)
        d["phase"] = ev.phase.value
        out.append(d)

    # 1. close on a 3.7 mm rim (opening 0.07), hold, observed elsewhere never; open -> released
    lt = _latch(cfg)
    for k, op in enumerate([1.0, 0.6, 0.15, 0.072, 0.07, 0.07, 0.069, 0.07, 0.07, 0.07, 0.9, 0.95]):
        add(_step(lt, op, command=1.0 if op >= 0.9 or k == 0 else 0.0, seq=k, closing=0 if op >= 0.9 or k == 0 else 5))
    # 2. attach, then the fingers close past the object: the old slip rule
    lt = _latch(cfg)
    for k, op in enumerate([1.0, 0.6, 0.15, 0.074, 0.074, 0.074, 0.074, 0.074, 0.074, 0.074, 0.012, 0.012, 0.012]):
        add(_step(lt, op, command=1.0 if k == 0 else 0.0, seq=k, closing=0 if k == 0 else 5))
    # 3. attach, then a revoke inside the window (opening dropped)
    lt = _latch(cfg)
    for k, op in enumerate([1.0, 0.6, 0.15, 0.074, 0.074, 0.01, 0.01]):
        add(_step(lt, op, command=1.0 if k == 0 else 0.0, seq=k, closing=0 if k == 0 else 5))
    # 4. a thin gap that the old blocked check refuses, with and without the HSR6 assist (reach unknown -> strong check fails)
    lt = _latch(cfg)
    for k, op in enumerate([1.0, 0.6, 0.15, 0.04, 0.04, 0.04, 0.04]):
        add(_step(lt, op, command=1.0 if k == 0 else 0.0, seq=k, closing=0 if k == 0 else 5, reach=None,
                  attach_assist={"phase": "lift"}))
    # 5. a thin gap, strong path: not_empty / blocked fail
    lt = _latch(cfg)
    for k, op in enumerate([1.0, 0.6, 0.15, 0.04, 0.04, 0.04]):
        add(_step(lt, op, command=1.0 if k == 0 else 0.0, seq=k, closing=0 if k == 0 else 5))
    return out


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _pc_cfg(**kw):
    """The pc version's latch (Panda calibration + the options pc turns on)."""
    from AG3S.adapters.panda.server import panda_latch_config

    return dataclasses.replace(panda_latch_config(), slip_detach=True, reach_sticky=True, drop_from_open=True, **kw)


def _gap_cfg(**kw):
    from AG3S.adapters.panda.server import panda_latch_config

    return dataclasses.replace(panda_latch_config(gap_opening=True), slip_detach=True, reach_sticky=True, drop_from_open=True, **kw)


# ---------------------------------------------------------------------------------------------------- (a) off
def test_new_latch_fields_default_to_off_and_the_default_events_are_main_bit_for_bit():
    d = LatchConfig()
    assert d.hold_empty_below is None and d.assist_requires_reach is False
    assert _sha(_golden_events(_pc_cfg())) == GOLDEN_EVENTS_SHA256


def test_off_latch_records_no_new_key():
    lt = _latch(_pc_cfg())
    evs = [_step(lt, op, command=1.0 if k == 0 else 0.0, seq=k, closing=0 if k == 0 else 5)
           for k, op in enumerate([1.0, 0.6, 0.15, 0.074, 0.074, 0.074])]
    blob = json.dumps([e.evidence for e in evs], default=str)
    for key in ("closed_on_nothing", "observed_at_hand"):
        assert key not in blob


def test_panda_latch_config_is_the_calibrated_one_unless_the_gap_opening_asks():
    from AG3S.adapters.panda.server import panda_latch_config

    assert panda_latch_config() == LatchConfig(empty_below=0.025325)
    assert panda_latch_config(gap_opening=False) == panda_latch_config()
    g = panda_latch_config(gap_opening=True)
    assert (g.empty_below, g.blocked_gap, g.hold_empty_below, g.assist_requires_reach) == (0.04, 0.03, 0.01678, True)
    assert dataclasses.replace(g, empty_below=0.025325, blocked_gap=0.05, hold_empty_below=None, assist_requires_reach=False) \
        == panda_latch_config()                                                       # nothing else differs


class _Capture:
    """What ``PandaSafeServer`` wraps: records the request ``SafePolicy`` would get, then stops before the conversion."""

    def __init__(self):
        self.obs = None

    def reset(self):
        pass

    def infer(self, obs, **kwargs):
        self.obs = obs
        raise _Stop


class _Stop(Exception):
    pass


def _feedback_obs(rs9):
    return {"ag3s/seq": 4, "panda/robot_state": np.asarray(rs9, np.float64),
            "ag3s/exec_feedback": {"available": True, "seq": 3, "n_steps": 5, "n_exec": 5,
                                   "executed": [True] * 5, "applied_gripper": np.zeros((5, 2), np.float32),
                                   "measured_gripper": np.array([0.0125, 1.0], np.float32)}}


def _rs(q1, q2):
    return [0.0] * 7 + [q1, q2]


def _served(on):
    from AG3S.adapters.panda import server as ps

    cap = _Capture()
    srv = ps.PandaSafeServer(cap, ps.ChunkToJointPolicy(_Stub(), None), **({"gap_opening": True} if on else {}))
    # the applied-arm rows need the conversion of the request that sent the chunk (not under test): keep the client's feedback as is
    srv._feedback = lambda obs: {"used": True, "seq": 3}                                  # noqa: SLF001
    return srv, cap


def test_server_flag_off_leaves_the_measured_opening_untouched_and_on_replaces_only_the_first_hand():
    for on in (False, True):
        srv, cap = _served(on)
        with pytest.raises(_Stop):
            srv.infer(_feedback_obs(_rs(-0.0005, -0.0228)))
        mg = np.asarray(cap.obs["ag3s/exec_feedback"]["measured_gripper"])
        if on:
            assert mg.dtype == np.float32 and mg[0] == pytest.approx((-0.0005 + 0.0228) / 0.08, abs=1e-6) and mg[1] == 1.0
        else:
            assert mg[0] == np.float32(0.0125) and srv.gap_opening is False
        # the policy-side state opening (finger models, SafePolicy state column) is the single-finger one either way
        assert cap.obs["state"][7] == pytest.approx(0.0005 / 0.04)


def test_server_gap_feedback_is_a_noop_without_feedback_or_robot_state():
    from AG3S.adapters.panda.server import PandaSafeServer

    fb_rec = {"used": False}
    assert PandaSafeServer._gap_feedback({"ag3s/exec_feedback": {"available": False}, "panda/robot_state": _rs(0, 0)}, fb_rec) is fb_rec
    assert PandaSafeServer._gap_feedback({"panda/robot_state": _rs(0, 0)}, fb_rec) is fb_rec
    obs = {"ag3s/exec_feedback": {"available": True, "measured_gripper": np.array([0.5, 1.0], np.float32)}}
    assert PandaSafeServer._gap_feedback(obs, fb_rec) is fb_rec and obs["ag3s/exec_feedback"]["measured_gripper"][0] == 0.5


def test_server_records_both_openings_only_with_the_flag():
    srv, cap = _served(True)
    obs = _feedback_obs(_rs(-0.0005, -0.0228))
    rec = srv._gap_feedback(obs, {"used": True, "seq": 3})                              # noqa: SLF001
    assert rec["used"] is True and set(rec["gripper_gap_opening"]) == {"single_finger", "gap", "q1", "q2"}
    assert rec["gripper_gap_opening"]["single_finger"] == pytest.approx(0.0125) and rec["gripper_gap_opening"]["gap"] > 0.27
    json.dumps(rec)


# ---------------------------------------------------------------------------------------------------- (b) the measure
def test_gripper_gap_norm():
    from AG3S.adapters.panda.server import gripper_gap_norm, gripper_opening_norm

    assert gripper_gap_norm(_rs(0.04, -0.04)) == pytest.approx(1.0)
    assert gripper_gap_norm(_rs(0.0005, -0.0005)) == pytest.approx(0.0125)           # the closed stop, gap 1.0 mm
    assert gripper_gap_norm(_rs(-0.0005, -0.0228)) == pytest.approx(0.278750)         # shifted together, 22.3 mm
    assert gripper_gap_norm(_rs(0.05, -0.05)) == 1.0 and gripper_gap_norm(_rs(-0.01, 0.01)) == 0.0   # clipped
    assert gripper_opening_norm(_rs(-0.0005, -0.0228)) == pytest.approx(0.0125)       # the single finger value, unchanged
    with pytest.raises(ValueError):
        gripper_gap_norm([0.0] * 8)


def test_replay_helpers_use_the_same_formula_as_the_server():
    from AG3S.adapters.panda.server import gripper_gap_norm
    from AG3S.trajopt.experiments.panda_latch_replay import gap_opening, single_opening

    for q1, q2 in ((0.04, -0.04), (0.0021, -0.0019), (-0.0005, -0.0228), (0.0, 0.0)):
        assert gap_opening(q1, q2) == pytest.approx(gripper_gap_norm(_rs(q1, q2)))
    assert single_opening(-0.0005) == pytest.approx(0.0125)


# ---------------------------------------------------------------------------------------------------- (c) the misjudgments
def _attach_rim(latch, opening=0.05):
    """Close on a rim and stop at the attach request (the revoke window then starts clean)."""
    evs = _close_on(latch, opening, stop_on_attach=True)
    assert latch.phase is GraspPhase.HELD and evs[-1].attach, [e.note for e in evs]
    return evs


def _run_out_window(latch, opening, **kw):
    for k in range(6):                                            # more than `revoke_chunks` executed chunks
        ev = _step(latch, opening, seq=10 + k, closing=5, **kw)
        assert ev.phase is GraspPhase.HELD, ev.note


def test_a_rim_held_at_4_mm_attaches_with_the_gap_opening():
    lt = _latch(_gap_cfg())
    evs = _close_on(lt, 4.0 / 80.0)
    assert lt.phase is GraspPhase.HELD and any(e.attach for e in evs)
    assert evs[-1].evidence is not None


def test_shifted_fingers_are_a_slip_for_the_single_finger_opening_and_a_hold_for_the_gap_opening():
    from AG3S.trajopt.experiments.panda_latch_replay import gap_opening, single_opening

    q_hold, q_shift = (0.0021, -0.0019), (-0.0005, -0.0228)       # attach pose; the call-40 pose of LIBERO-Safety obstacle e0
    for gap_mode, cfg in ((False, _pc_cfg()), (True, _gap_cfg())):
        op = (lambda q: gap_opening(*q)) if gap_mode else (lambda q: single_opening(q[0]))
        hold = op(q_hold) if gap_mode else 0.074                 # the single-finger opening of that call was 0.074 (recorded)
        lt = _latch(cfg)
        _attach_rim(lt, hold)
        _run_out_window(lt, hold)
        states = [_step(lt, op(q_shift), seq=30 + k, closing=5).phase for k in range(8)]
        if gap_mode:
            assert op(q_shift) == pytest.approx(0.27875) and set(states) == {GraspPhase.HELD}
        else:
            assert op(q_shift) == pytest.approx(0.0125) and states[-1] is GraspPhase.PLACED     # the misjudgment this section removes


def test_a_thin_rim_at_2_mm_with_the_object_observed_at_the_hand_stays_held():
    lt = _latch(_gap_cfg())
    _attach_rim(lt, 4.0 / 80.0)
    obs = {"centre_distance_m": 0.01, "held_radius_m": 0.04, "hand_rise_m": 0.0, "object_rise_m": 0.0}
    for k in range(4):                                             # inside the 4-chunk revoke window
        ev = _step(lt, 2.0 / 80.0, seq=20 + k, closing=5, held_observation=obs)
        assert ev.phase is GraspPhase.HELD and not ev.detach, ev.note
        assert ev.evidence["hold_checks"]["observed_at_hand"] is True
    for k in range(6):                                             # past the window nothing re-checks a settled hold
        ev = _step(lt, 2.0 / 80.0, seq=40 + k, closing=5)
        assert ev.phase is GraspPhase.HELD and not ev.detach, ev.note


def test_a_thin_rim_at_2_mm_without_any_object_evidence_in_the_window_is_revoked():
    """The limit stated in impl.md: the gap alone does not tell a 2 mm rim from a closure on nothing."""
    lt = _latch(_gap_cfg())
    _attach_rim(lt, 4.0 / 80.0)
    ev = _step(lt, 2.0 / 80.0, seq=20, closing=5)
    assert ev.revoked and "not_blocked" in ev.evidence["revoke"]["reasons"]


# ---------------------------------------------------------------------------------------------------- (d) true empties
def test_a_closure_to_the_mechanical_stop_is_empty():
    lt = _latch(_gap_cfg())
    evs = _close_on(lt, 1.0 / 80.0)                                  # gap 1.0 mm = the closed stop
    assert lt.phase is GraspPhase.CLOSING and not any(e.attach for e in evs)
    assert evs[-1].evidence["checks"]["not_empty"] is False and evs[-1].evidence["checks"]["blocked"] is False


def test_a_closure_on_a_2_4_mm_sliver_does_not_attach_either():
    lt = _latch(_gap_cfg())
    evs = _close_on(lt, 2.4 / 80.0)
    assert lt.phase is GraspPhase.CLOSING and not any(e.attach for e in evs)
    assert evs[-1].evidence["checks"]["not_empty"] is False


def test_the_assist_needs_reach_evidence_with_the_gap_opening_only():
    for cfg, expect_attach in ((_pc_cfg(), True), (_gap_cfg(), False)):
        lt = _latch(dataclasses.replace(cfg, empty_below=0.01))      # fails the strong check through `reach` only
        evs = [_step(lt, 1.0, command=1.0, seq=0, closing=0, reach=0.5, attach_assist={"phase": "lift"})]
        evs.append(_step(lt, 0.5, seq=1, reach=0.5, attach_assist={"phase": "lift"}))
        for k in range(4):
            evs.append(_step(lt, 0.08, seq=2 + k, reach=0.5, attach_assist={"phase": "lift"}))
        assert any(e.attach for e in evs) is expect_attach, [e.note for e in evs]


def test_a_hold_that_closes_to_the_stop_inside_the_window_is_revoked_for_closed_on_nothing():
    lt = _latch(_gap_cfg())
    _attach_rim(lt, 4.0 / 80.0)
    ev = _step(lt, 1.0 / 80.0, seq=20, closing=5)
    assert ev.revoked and "closed_on_nothing" in ev.evidence["revoke"]["reasons"]


def test_a_hold_that_closes_to_the_stop_after_the_window_slips_after_two_requests():
    lt = _latch(_gap_cfg())
    _attach_rim(lt, 4.0 / 80.0)
    _run_out_window(lt, 4.0 / 80.0)
    first = _step(lt, 1.0 / 80.0, seq=30, closing=5)
    second = _step(lt, 1.0 / 80.0, seq=31, closing=5)
    assert first.phase is GraspPhase.HELD and second.slipped and second.phase is GraspPhase.PLACED
    assert second.evidence["slip"]["closed_on_nothing"] is True
    # the single-finger latch is blind to it: a rim at 0.05 closing to the stop (0.0125) drops 0.0375 < slip_drop 0.05
    old = _latch(_pc_cfg())
    _attach_rim(old, 0.05)
    _run_out_window(old, 0.05)
    assert [_step(old, 0.0125, seq=30 + k, closing=5).phase for k in range(4)] == [GraspPhase.HELD] * 4


# ---------------------------------------------------------------------------------------------------- (e) versions
def test_each_new_version_is_the_base_plus_the_flag():
    from AG3S.adapters.panda import profile as pf

    assert pf.SL11_GAP_VERSIONS == NEW and pf.GAP_OPENING_FLAGS == ("--gripper-gap-opening",)
    reg = pf.AG3S_VERSIONS
    assert list(reg[GAP]) == list(reg[PC]) + ["--gripper-gap-opening"]
    assert list(reg[MIGAP]) == list(reg[MI]) + ["--gripper-gap-opening"]
    for v in NEW:
        assert pf.is_final_version(v) and v in pf.PC_LATCH_VERSIONS and v not in pf.SL11_VERSIONS
        assert pf.version_server_argv(v) == pf.version_server_argv(PC)
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]
    assert [n for n, f in reg.items() if "--gripper-gap-opening" in f] == list(NEW)


def test_every_other_version_is_unchanged():
    from AG3S.adapters.panda import profile as pf

    pass_versions = set(pf.SL11_VERSIONS)
    old = {k: list(v) for k, v in pf.AG3S_VERSIONS.items()
           if k not in pass_versions and k not in NEW and not k.endswith("-lsfork") and k not in pf.SL12_VERSIONS}   # SL12: the -oraclegt pair
    assert len(old) == 51 and hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest() == OLD_REGISTRY_SHA256
    assert not any(v.endswith("-lsfork") and "gap" in v for v in pf.AG3S_VERSIONS)       # no -lsfork twin is registered


def _args(v, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(v)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(v))


def test_parsed_flags_and_record_keys(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    base, mi = _args(PC, monkeypatch), _args(MI, monkeypatch)
    gap, migap = _args(GAP, monkeypatch), _args(MIGAP, monkeypatch)
    diff = lambda a, b: {k for k in vars(a) if vars(a)[k] != vars(b)[k]}   # noqa: E731
    assert diff(gap, base) == {"gripper_gap_opening"} and gap.gripper_gap_opening is True
    assert diff(migap, mi) == {"gripper_gap_opening"} and migap.minimal_intervention_mm == 10.0
    assert base.gripper_gap_opening is False
    # the name is read back
    assert pf.match_ag3s_version(gap) == GAP and pf.match_ag3s_version(migap) == MIGAP
    assert pf.match_ag3s_version(base) == PC and pf.match_ag3s_version(mi) == MI
    # the record: the dest joins only when set -- every other version keeps exactly its keys
    vb, vg, vmi, vmg = (pf.version_flag_values(a) for a in (base, gap, mi, migap))
    assert set(vg) - set(vb) == {"gripper_gap_opening"} and vg["gripper_gap_opening"] is True
    assert set(vmg) - set(vmi) == {"gripper_gap_opening"}
    for v in (PC, MI, f"{PC}-mi10-sdev", f"{PC}-observe", f"{PC}-lsfork", f"{PC}-surface", "AG3S-final-panda-v3-e0to", "V11AB-pb7"):
        assert "gripper_gap_opening" not in pf.version_flag_values(_args(v, monkeypatch)), v


def test_launchers_accept_the_versions(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from benchmark.runners import launch_libero_paired as lp
    from benchmark.runners import launch_libero_safety_paired as lsp

    for v in NEW:
        got = lp.server_argv("vla_v11ab", "/ck", 9, v, hsr6_probe="/p.npz")
        want = lp.server_argv("vla_v11ab", "/ck", 9, MI if v == MIGAP else PC, hsr6_probe="/p.npz")
        assert [t for t in got if t != "--gripper-gap-opening"] == want and got[-2:] == ["--hsr6-probe", "/p.npz"]
        a = lp.build_parser().parse_args(["--task-ids", "1", "--episode-ids", "1", "--run-id", "x", "--conversion", "C",
                                          "--ag3s-version", v])
        assert a.ag3s_version == v and pf.version_server_flags(v, **lp.version_options(a))
        wrapped = lsp.make_server_argv("pi05_libero_safety")("vla_v11ab", "/ck", 9, v)
        assert "--gripper-gap-opening" in wrapped and wrapped[wrapped.index("--robot") + 1] == "panda"


# ---------------------------------------------------------------------------------------------------- refusals
BASE = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]


def test_rby1_server_refuses_the_flag_and_the_panda_server_does_not():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    ss.reject_panda_only_flags(ap, ap.parse_args(BASE))                                     # off: nothing
    rby1 = ap.parse_args(BASE + ["--gripper-gap-opening"])
    assert rby1.robot == "rby1" and rby1.gripper_gap_opening is True
    with pytest.raises(SystemExit):
        ss.reject_panda_only_flags(ap, rby1)
    ss.reject_panda_only_flags(ap, ap.parse_args(BASE + ["--gripper-gap-opening", "--robot", "panda"]))


def test_the_flag_needs_the_safe_policy():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + ["--gripper-gap-opening"]))     # alone: accepted
    with pytest.raises(SystemExit):
        ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + ["--gripper-gap-opening", "--no-safe", "--robot", "panda"]))


def test_panda_server_accepts_the_flag_as_ported():
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    assert "gripper_gap_opening" not in ps.UNPORTED_FINAL_CONFIG_DESTS
    ps.check_unported_flags(ss.build_parser().parse_args(BASE + ["--robot", "panda", "--gripper-gap-opening"]), None)
