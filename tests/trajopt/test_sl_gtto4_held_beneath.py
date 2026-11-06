"""SL-GTTO4 F3 — `--held-beneath-contact` (lead self-ruling G29, 2026-11-05), CPU.

SL6-H's placement-contact rows (`CollisionLinearizer._placement_contact_margin`, tested in
`tests/trajopt/test_sl6h_placement_contact.py`) switched on by the grasp latch alone:

1. `SafePolicy._held_beneath_contact`: the snapshot gets `placement_contact` iff latch HELD ∧ attached points ∧ a field;
   otherwise the same object comes back. An HSR6 spec already set is kept. Every call is recorded.
2. `_held_beneath_record`: the final check's released rows join the record; `_to_block` carries it only when on.
3. serve_safe: the flag parses, `apply_grasp_contact_flags` sets it (and reports it only when on).
4. profile: `AG3S-commonto-gt2-hbc` / `-termsoff-hbc` = their base + the flag; read back; server argv as gt2.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from AG3S.trajopt import hsr6 as H6
from AG3S.trajopt.grasp_latch import GraspPhase, LatchEvent
from tests.trajopt.test_sl6h_placement_contact import HELD_PTS, _policy, _snapshot


def _latch(sp, phase):
    sp._pending = {}
    sp._latch.phase = phase
    sp._latch._last_grasp = LatchEvent(phase=phase, evidence={"checks": {}})


@pytest.mark.parametrize("latch,attached,field,applied", [
    (GraspPhase.HELD, True, True, True),
    (GraspPhase.HELD, False, True, False),
    (GraspPhase.HELD, True, False, False),
    (GraspPhase.CLOSING, True, True, False),
    (GraspPhase.LATCHED, True, True, False),
    (GraspPhase.PLACED, True, True, False),
    (GraspPhase.SEARCHING, False, True, False)])
def test_gate(latch, attached, field, applied):
    sp = _policy()
    sp.held_beneath_contact = True
    _latch(sp, latch)
    snap = _snapshot(attached_points=HELD_PTS if attached else None, attached_parent_link="l" if attached else None)
    if not field:
        snap = dataclasses.replace(snap, esdf=None)
    out = sp._held_beneath_contact(snap)
    assert (out.placement_contact is not None) is applied
    if applied:
        assert out.placement_contact == H6.placement_contact_spec()
    else:
        assert out is snap
    rec = sp._last_held_beneath
    assert rec == {"latch": latch.value, "attached": attached, "applied": applied,
                   "n_held_rows": 3 if attached else 0}


def test_an_hsr6_spec_already_set_is_kept():
    sp = _policy()
    sp.held_beneath_contact = True
    _latch(sp, GraspPhase.HELD)
    own = {"cos_min": 0.5, "footprint_tol_m": 0.01}
    snap = _snapshot(attached_points=HELD_PTS, attached_parent_link="l", placement_contact=own)
    assert sp._held_beneath_contact(snap) is snap


def test_record_reads_the_final_check_and_to_block_only_when_on():
    sp = _policy()
    assert sp.held_beneath_contact is False
    sp.refiner.last_result = types.SimpleNamespace(metrics={}, status=None, iterations=0, solve_time_ms=0.0,
                                                   reference_deviation=0.0)
    assert "held_beneath_contact" not in sp._to_block()
    sp.held_beneath_contact = True
    _latch(sp, GraspPhase.HELD)
    sp._held_beneath_contact(_snapshot(attached_points=HELD_PTS, attached_parent_link="l"))
    sp.refiner.last_result = types.SimpleNamespace(
        metrics={"placement_contact": {"rows_released": 5, "held_rows_released": 2,
                                       "min_clearance_released_m": -0.0213}},
        status=None, iterations=1, solve_time_ms=1.0, reference_deviation=0.1)
    rec = sp._to_block()["held_beneath_contact"]
    assert rec["applied"] is True and rec["rows_released"] == 5 and rec["held_rows_released"] == 2
    assert rec["min_clearance_released_m"] == pytest.approx(-0.0213)
    # not applied: no released-row keys
    _latch(sp, GraspPhase.CLOSING)
    sp._held_beneath_contact(_snapshot(attached_points=HELD_PTS, attached_parent_link="l"))
    rec = sp._to_block()["held_beneath_contact"]
    assert rec["applied"] is False and "rows_released" not in rec


def test_serve_safe_flag():
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    act = ap._option_string_actions["--held-beneath-contact"]
    assert act.dest == "held_beneath_contact" and act.default is False and act.const is True
    sp = types.SimpleNamespace()
    off = ss.apply_grasp_contact_flags(sp, types.SimpleNamespace(held_beneath_contact=False))
    assert "held_beneath_contact" not in off and not getattr(sp, "held_beneath_contact", False)
    on = ss.apply_grasp_contact_flags(sp, types.SimpleNamespace(held_beneath_contact=True))
    assert on["held_beneath_contact"] is True and sp.held_beneath_contact is True


def test_profile_versions(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from tests.benchmark.test_sl_gtto_gt_perception import _args

    gt2, toff = "AG3S-commonto-gt2", "AG3S-commonto-gt2-termsoff"
    assert pf.GT4_VERSIONS == ("AG3S-commonto-gt2-hbc", "AG3S-commonto-gt2-termsoff-hbc",
                               "AG3S-commonto-gt2-termsoff-pass-hbc")
    assert list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-hbc"]) == (
        list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass"]) + ["--held-beneath-contact"])
    assert list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-hbc"]) == list(pf.AG3S_VERSIONS[gt2]) + ["--held-beneath-contact"]
    assert list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-hbc"]) == (list(pf.AG3S_VERSIONS[toff])
                                                                         + ["--held-beneath-contact"])
    assert "held_beneath_contact" in srv.PORTED_FINAL_CONFIG_DESTS
    assert "held_beneath_contact" not in srv.UNPORTED_FINAL_CONFIG_DESTS
    for v in pf.GT4_VERSIONS:
        assert pf.is_final_version(v) and pf.version_requires_oracle_segmentation(v)
        assert pf.version_server_argv(v) == pf.version_server_argv(gt2)
        args = _args(v)
        assert pf.match_ag3s_version(args) == v and args.held_beneath_contact is True
        assert pf.version_flag_values(args)["held_beneath_contact"] is True
    for v in pf.AG3S_VERSIONS:
        if v not in pf.GT4_VERSIONS and v not in ("AG3S-commonto-gt2-termsoff-pass-zr-oc-hbc",
                                                   "AG3S-commonto-gt2-termsoff-pass-zr-oc-hbch"):   # G42 / G44 combine F3
            assert "--held-beneath-contact" not in pf.AG3S_VERSIONS[v], v
    # gt2's record keeps its keys (the new dest joins only when set)
    assert "held_beneath_contact" not in pf.version_flag_values(_args(gt2))


from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: E402,F401 (fixture)
