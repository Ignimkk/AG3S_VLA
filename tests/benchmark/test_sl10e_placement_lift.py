"""SL10e -- "들어 올림 때도 아래 받침면 허용": `--hsr6-placement-contact-lift` (task `AG3S/ag3s/docs/handoff/SL10e.task.md` §1). CPU.

(a) off (default): `placement_contact_ok` returns what it returned before for every (phase, latch, checks) in the table
    (checked against an independent copy of the old rule), `Hsr6Flags.names()` and the SafePolicy record keep their keys;
(b) on: phase = placement ∨ lift; lift ∧ HELD is on, lift ∧ (CLOSING ∧ closed ∧ blocked) is on, the latch condition is
    unchanged (lift ∧ CLOSING without evidence / LATCHED / SEARCHING stays off), transport · approach · grasp · retreat · None
    stay off; the record says ``phase_rule == "placement|lift"`` (and only then); the snapshot's ``placement_contact`` is the
    same spec as with the flag off (the geometry condition is not touched);
(c) the two new versions are their bases plus this one flag, take the HSR6 probe like their bases, the built Panda server
    carries the flag into SafePolicy's Hsr6Flags; the flag without ``--hsr6-placement-contact`` is refused; the RB-Y1 server
    refuses it; the old versions' records keep their keys.
"""
from __future__ import annotations

import itertools
import pathlib

import numpy as np
import pytest

from AG3S.trajopt import hsr6 as H6
from AG3S.trajopt.grasp_latch import GraspPhase
from tests.benchmark.test_sl6a_panda_final import _Stub, gpu_ok  # noqa: F401 (fixture)
from tests.trajopt.test_sl6h_placement_contact import CB, HELD_PTS, _policy, _snapshot, _with_latch

PC = "AG3S-final-panda-v3-e0to-pc"
PCS = "AG3S-final-panda-v3-e0to-pc-surface"
VERSIONS = {f"{PC}-plift": PC, f"{PCS}-plift": PCS}
FLAG = "--hsr6-placement-contact-lift"

PHASES = (*H6.PHASES6, None, "unknown")
LATCHES = ("held", "closing", "latched", "searching", "placed", None)
CHECKS = ({}, {"closed": True, "blocked": True}, {"closed": True, "blocked": False}, {"closed": False, "blocked": True},
          {"closed": False, "blocked": False}, {"closed": True}, {"blocked": True})


def _old_rule(phase, *, latch, checks):
    """The ⑤ rule as it was before SL10e (copied, not imported)."""
    if phase != "placement":
        return False
    if latch == "held":
        return True
    return bool(latch == "closing" and all(checks.get(k) is True for k in ("closed", "blocked")))


# ------------------------------------------------------------------------------------------------ (a) off = as before
def test_off_is_bit_identical_over_the_whole_table():
    n = 0
    for phase, latch, checks in itertools.product(PHASES, LATCHES, CHECKS):
        want = _old_rule(phase, latch=latch, checks=checks)
        assert H6.placement_contact_ok(phase, latch=latch, checks=checks) is want
        assert H6.placement_contact_ok(phase, latch=latch, checks=checks, lift=False) is want
        n += 1
    assert n == len(PHASES) * len(LATCHES) * len(CHECKS)


def test_flags_default_off_and_names_keep_their_order():
    assert H6.Hsr6Flags().placement_contact_lift is False and not H6.Hsr6Flags().on
    assert H6.Hsr6Flags(placement_contact=True).names() == ["placement_contact"]
    assert H6.Hsr6Flags(placement_contact=True, objective_window=True).names() == ["placement_contact", "objective_window"]
    f = H6.Hsr6Flags(placement_contact=True, placement_contact_lift=True)
    assert f.on and f.names() == ["placement_contact", "placement_contact_lift"]
    assert H6.Hsr6Flags(placement_contact_lift=True).on         # counted so that start-up refuses it (c)


# ------------------------------------------------------------------------------------------------ (b) on
@pytest.mark.parametrize("phase,latch,checks,want", [
    ("lift", "held", {}, True),
    ("lift", "held", {"closed": False, "blocked": False}, True),            # HELD needs no evidence
    ("lift", "closing", {"closed": True, "blocked": True}, True),
    ("lift", "closing", {"closed": True, "blocked": False}, False),
    ("lift", "closing", {"closed": False, "blocked": True}, False),
    ("lift", "closing", {}, False),
    ("lift", "latched", {"closed": True, "blocked": True}, False),
    ("lift", "searching", {}, False),
    ("lift", "placed", {}, False),
    ("placement", "held", {}, True),                                         # placement as before
    ("placement", "closing", {"closed": True, "blocked": True}, True),
    ("placement", "closing", {}, False),
    ("transport", "held", {}, False),                                        # not transport
    ("transport", "closing", {"closed": True, "blocked": True}, False),
    ("approach", "held", {}, False),
    ("grasp", "held", {}, False),
    ("retreat", "held", {}, False),
    (None, "held", {}, False),
    ("unknown", "held", {}, False)])
def test_on_rule(phase, latch, checks, want):
    assert H6.placement_contact_ok(phase, latch=latch, checks=checks, lift=True) is want


def test_on_differs_from_off_only_in_the_lift_rows():
    diff = [(p, l, tuple(sorted(c.items()))) for p, l, c in itertools.product(PHASES, LATCHES, CHECKS)
            if H6.placement_contact_ok(p, latch=l, checks=c, lift=True) != H6.placement_contact_ok(p, latch=l, checks=c)]
    assert diff and {p for p, _, _ in diff} == {"lift"}
    # every lift row that is on is the row placement has: the latch condition is shared
    for l, c in itertools.product(LATCHES, CHECKS):
        assert (H6.placement_contact_ok("lift", latch=l, checks=c, lift=True)
                == H6.placement_contact_ok("placement", latch=l, checks=c))


def test_phase_labels_are_the_strings_the_probe_and_hsr5_emit():
    assert H6.PLACEMENT == "placement" and "lift" in H6.PHASES6 and "transport" in H6.PHASES6
    assert H6.PLACEMENT_LIFT_PHASES == frozenset({"placement", "lift"}) and H6.PLACEMENT_LIFT_RULE == "placement|lift"
    from AG3S.trajopt.hsr5 import RULE_PHASE6

    assert set(RULE_PHASE6.values()) <= set(H6.PHASES6)       # --hsr5's mapped labels (carry -> transport) are not lift


_BASE_KEYS = {"on", "phase", "latch", "closed", "blocked", "attached", "applied", "n_held_rows", "rows_released",
              "held_rows_released"}


@pytest.mark.parametrize("hsr6_phase,latch,checks,attached,lift_on,off_applied", [
    ("lift", GraspPhase.HELD, {}, True, True, False),
    ("lift", GraspPhase.CLOSING, CB, True, True, False),
    ("lift", GraspPhase.CLOSING, {**CB, "blocked": False}, True, False, False),
    ("lift", GraspPhase.LATCHED, CB, True, False, False),
    ("lift", GraspPhase.HELD, {}, False, True, False),          # rule on, nothing attached: nothing to release
    ("transport", GraspPhase.HELD, {}, True, False, False),
    ("placement", GraspPhase.HELD, {}, True, True, True),
    ("approach", GraspPhase.HELD, {}, True, False, False)])
def test_safe_policy_gate_and_record(hsr6_phase, latch, checks, attached, lift_on, off_applied):
    snap = _snapshot(attached_points=HELD_PTS if attached else None, attached_parent_link="l" if attached else None)
    sp_off = _policy(hsr6=H6.Hsr6Flags(placement_contact=True))
    _with_latch(sp_off, latch, checks, hsr6_phase)
    out_off = sp_off._hsr6_placement_contact(snap)
    rec_off = sp_off._last_hsr6["detail"]["placement_contact"]
    assert (out_off.placement_contact is not None) is off_applied
    assert "phase_rule" not in rec_off and set(rec_off) - {"reason"} == _BASE_KEYS          # the keys it always had

    sp = _policy(hsr6=H6.Hsr6Flags(placement_contact=True, placement_contact_lift=True))
    _with_latch(sp, latch, checks, hsr6_phase)
    out = sp._hsr6_placement_contact(snap)
    rec = sp._last_hsr6["detail"]["placement_contact"]
    assert rec["on"] is lift_on and rec["applied"] is (lift_on and attached)
    assert rec["phase_rule"] == "placement|lift"
    assert set(rec) - {"reason", "phase_rule"} == _BASE_KEYS
    assert {k: v for k, v in rec.items() if k not in ("on", "applied", "phase_rule", "reason")} == \
        {k: v for k, v in rec_off.items() if k not in ("on", "applied", "reason")}
    if rec["applied"]:
        # the geometry condition is the one ⑤ already had: the same spec object content
        assert out.placement_contact == H6.placement_contact_spec() and out.placement_contact == \
            {"cos_min": pytest.approx(H6.PLACEMENT_BENEATH_COS), "footprint_tol_m": None}
    else:
        assert out is snap
    if off_applied:       # placement: identical with or without the lift flag
        assert out_off.placement_contact == out.placement_contact


def test_safe_policy_result_is_unchanged_by_the_flag():
    for lift in (False, True):
        sp = _policy(hsr6=H6.Hsr6Flags(placement_contact=True, placement_contact_lift=lift))
        _with_latch(sp, GraspPhase.HELD, {}, "lift" if lift else "placement")
        sp._hsr6_placement_contact(_snapshot(attached_points=HELD_PTS, attached_parent_link="l"))
        import types

        sp.refiner.last_result = types.SimpleNamespace(metrics={"placement_contact": {
            "rows_released": 5, "held_rows_released": 2, "min_clearance_released_m": -0.01}})
        sp._hsr6_placement_contact_result()
        rec = sp._last_hsr6["detail"]["placement_contact"]
        assert rec["rows_released"] == 5 and rec["held_rows_released"] == 2
        assert sp._last_hsr6["changed"]["placement_contact"] is True
        assert ("phase_rule" in rec) is lift


# ------------------------------------------------------------------------------------------------ (c) versions / servers
def _probe_or_skip(v):
    from AG3S.adapters.panda import profile as pf

    if not pathlib.Path(pf.PANDA_HSR6_PROBE).exists():
        pytest.skip("Panda HSR6 probe not in this storage root")


def _args(v, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    monkeypatch.setenv("AG3S_PANDA_CONVERSION", "C")
    argv = pf.version_server_argv(v)
    argv[argv.index("--config") + 1] = "pi05_libero"
    return pf.parse_v11ab_args(argv=argv + ["--robot", "panda"] + pf.version_server_flags(v))


def test_versions_are_their_bases_plus_the_one_flag(gpu_ok):
    from AG3S.adapters.panda import profile as pf

    assert set(VERSIONS) == set(pf.SL10E_VERSIONS)
    assert pf.PLACEMENT_LIFT_FLAGS == (FLAG,)
    for v, base in VERSIONS.items():
        assert list(pf.AG3S_VERSIONS[v]) == list(pf.AG3S_VERSIONS[base]) + [FLAG]
        assert "--hsr6-placement-contact" in pf.AG3S_VERSIONS[base] and FLAG not in pf.AG3S_VERSIONS[base]
        assert pf.is_final_version(v) and v in pf.PC_LATCH_VERSIONS
        assert pf.version_server_argv(v) == pf.version_server_argv(base)
        assert pf.version_server_flags(v)[-2:] == ["--hsr6-probe", pf.PANDA_HSR6_PROBE]     # the probe like the base
        assert pf.version_server_flags(v)[:-3] == pf.version_server_flags(base)[:-2]
    assert not any(FLAG in f for n, f in pf.AG3S_VERSIONS.items() if n not in VERSIONS)


@pytest.mark.parametrize("version", sorted(VERSIONS))
def test_flag_parses_reaches_safe_policy_and_only_it_differs(version, gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as ps
    from AG3S.trajopt import serve_safe as ss

    _probe_or_skip(version)
    base = VERSIONS[version]
    a, b = _args(version, monkeypatch), _args(base, monkeypatch)
    assert a.hsr6_placement_contact_lift is True and b.hsr6_placement_contact_lift is False and a.hsr6_placement_contact
    assert {k for k in vars(a) if vars(a)[k] != vars(b)[k]} == {"hsr6_placement_contact_lift"}
    assert ss.hsr6_flags_from_args(a).names()[-2:] == ["placement_contact", "placement_contact_lift"]
    assert ss.hsr6_flags_from_args(b).names()[-1] == "placement_contact"
    ps.check_unported_flags(a, None)
    ss.reject_hsr6_flags(ss.build_parser(), a)
    served, _info = ps.build_safe_server(_Stub(), a, gpu_warm_up=False)
    assert served._policy.hsr6.placement_contact_lift is True                     # noqa: SLF001
    base_served, _ = ps.build_safe_server(_Stub(), b, gpu_warm_up=False)
    assert base_served._policy.hsr6.placement_contact_lift is False               # noqa: SLF001
    # record keys: on adds its own, off keeps the old ones
    assert pf.match_ag3s_version(a) == version and pf.match_ag3s_version(b) == base
    va, vb = pf.version_flag_values(a), pf.version_flag_values(b)
    assert va["hsr6_placement_contact_lift"] is True and "hsr6_placement_contact_lift" not in vb
    # like the SL10 / SL10d versions: the earlier groups' dests join with it; on shared keys the values are the base's
    assert set(vb) <= set(va) and all(va[k] == vb[k] for k in vb)
    parser = ss.build_parser()
    assert all(va[k] == pf._jsonable(parser.get_default(k)) for k in set(va) - set(vb) - {"hsr6_placement_contact_lift"})
    # the server's constraint-record meta lists the flag only when on
    assert "placement_contact_lift" in ss.hsr6_record_meta(a)["hsr6"]["flags"]
    assert "placement_contact_lift" not in ss.hsr6_record_meta(b)["hsr6"]["flags"]


def test_old_versions_record_keys_do_not_gain_the_dest(gpu_ok, monkeypatch):
    from AG3S.adapters.panda import profile as pf

    for v in (PC, PCS, "AG3S-final-panda-v3-e0to-pc-hfq", "AG3S-final-panda-v3-e0to-pc-abox", "AG3S-final-panda-v3-e0to"):
        _probe_or_skip(v)
        assert "hsr6_placement_contact_lift" not in pf.version_flag_values(_args(v, monkeypatch)), v


RBY1_BASE = ["--config", "cfg", "--checkpoint", "/ckpt", "--model-xml", "/scene.xml"]


def _parse(extra):
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    return ap, ap.parse_args(RBY1_BASE + list(extra))


def test_flag_without_placement_contact_is_refused():
    from AG3S.trajopt import serve_safe as ss

    ap, args = _parse(["--robot", "panda", FLAG])
    assert args.hsr6_placement_contact_lift is True and args.hsr6_placement_contact is False
    with pytest.raises(SystemExit):
        ss.reject_hsr6_flags(ap, args)
    ap, args = _parse(["--robot", "panda", FLAG, "--hsr6-placement-contact"])
    ss.reject_hsr6_flags(ap, args)
    assert _parse([])[1].hsr6_placement_contact_lift is False


def test_the_rby1_server_refuses_the_flag(monkeypatch):
    import sys

    from AG3S.trajopt import serve_safe as ss

    ap, args = _parse(["--hsr6-placement-contact", FLAG])
    assert args.robot == "rby1"
    with pytest.raises(SystemExit):
        ss.reject_panda_only_flags(ap, args)
    ap, args = _parse(["--hsr6-placement-contact", FLAG, "--robot", "panda"])
    ss.reject_panda_only_flags(ap, args)                                           # the Panda server takes it
    ap, args = _parse(["--hsr6-placement-contact"])
    ss.reject_panda_only_flags(ap, args)                                           # the older flag is not refused (unchanged)
    # and it is `main` that refuses, before anything is loaded
    monkeypatch.setattr(sys, "argv", ["serve_safe"] + RBY1_BASE + ["--hsr6-placement-contact", FLAG])
    with pytest.raises(SystemExit):
        ss.main()
