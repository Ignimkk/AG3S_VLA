"""SL-GTTO7 F3b — `--held-beneath-hand` (lead self-ruling G44, 2026-11-05), CPU, on the SL6-H analytic rig.

With `placement_contact["hand_rows"]` and `grasp_contact["authorized"]`: an authorized hand row whose nearest surface is
beneath it and inside the held object's footprint is released (finger lowered onto the plate under the held bowl);
a finger beside a wall, an unauthorized row, and the spec without the key are unchanged (= SL6-H as tested there).
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt import hsr6 as H6
from tests.trajopt.test_sl6h_placement_contact import CENTRES, N_ROBOT, ROWS, _clear, _lin, _snapshot


def _gc(auth):
    return {"latch": "held", "authorized": np.asarray(auth, bool), "held_exempt": False, "zone_allowance": None,
            "zone_radius": 0.10, "centroid": None}


def _centres():
    c = CENTRES.copy()
    c[ROWS["finger_on_plate"], 2] = 0.015 - 0.002            # 2 mm inside the plate, under the held sphere's footprint
    return c


def test_hand_row_on_the_plate_beneath_is_released():
    lin = _lin()
    c = _centres()
    base = _clear(lin, _snapshot(grasp_contact=_gc([True, True])), c)
    spec = {**H6.placement_contact_spec(), "hand_rows": True}
    on = _clear(lin, _snapshot(grasp_contact=_gc([True, True]), placement_contact=spec), c)
    # margin −1 m replaces the finger row's obstacle margin (10 mm on this rig): + 1.010
    assert on[ROWS["finger_on_plate"]] == pytest.approx(base[ROWS["finger_on_plate"]] + 1.010)
    assert on[ROWS["finger_by_wall"]] == base[ROWS["finger_by_wall"]]            # beside: unchanged
    assert lin._last_placement_contact["hand_rows_released"] == 1
    assert on[ROWS["held_on_plate"]] == pytest.approx(base[ROWS["held_on_plate"]] + 1.0)   # the SL6-H part as before


def test_unauthorized_or_key_absent_changes_no_hand_row():
    lin = _lin()
    c = _centres()
    base = _clear(lin, _snapshot(grasp_contact=_gc([False, False])), c)
    spec = {**H6.placement_contact_spec(), "hand_rows": True}
    on = _clear(lin, _snapshot(grasp_contact=_gc([False, False]), placement_contact=spec), c)
    assert np.array_equal(base[:N_ROBOT], on[:N_ROBOT])
    plain = _clear(lin, _snapshot(grasp_contact=_gc([True, True]), placement_contact=H6.placement_contact_spec()), c)
    assert np.array_equal(base[:N_ROBOT], plain[:N_ROBOT]) and "hand_rows_released" not in lin._last_placement_contact


def test_policy_spec_and_flag(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.grasp_latch import GraspPhase
    from tests.benchmark.test_sl_gtto_gt_perception import _args
    from tests.trajopt.test_sl6h_placement_contact import HELD_PTS, _policy
    from tests.trajopt.test_sl_gtto4_held_beneath import _latch

    sp = _policy()
    sp.held_beneath_contact = True
    _latch(sp, GraspPhase.HELD)
    snap = _snapshot(attached_points=HELD_PTS, attached_parent_link="l")
    assert "hand_rows" not in sp._held_beneath_contact(snap).placement_contact
    sp.held_beneath_hand = True
    assert sp._held_beneath_contact(snap).placement_contact["hand_rows"] is True
    act = ss.build_parser()._option_string_actions["--held-beneath-hand"]
    assert act.dest == "held_beneath_hand" and act.default is False
    out = ss.apply_grasp_contact_flags(types.SimpleNamespace(), types.SimpleNamespace(held_beneath_hand=True))
    assert out["held_beneath_hand"] is True
    v = "AG3S-commonto-gt2-termsoff-pass-zr-oc-hbch"
    assert pf.GT7_VERSIONS == (v,)
    assert list(pf.AG3S_VERSIONS[v]) == list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass-zr-oc-hbc"]) + [
        "--held-beneath-hand"]
    args = _args(v)
    assert pf.match_ag3s_version(args) == v and pf.version_flag_values(args)["held_beneath_hand"] is True
    for name, flags in pf.AG3S_VERSIONS.items():
        if name != v:
            assert "--held-beneath-hand" not in flags, name


from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: E402,F401 (fixture)
