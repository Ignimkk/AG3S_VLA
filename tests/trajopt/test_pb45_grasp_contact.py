"""V11AB-pb4 / pb5 — grasp contact rules on ESDF rows (`SceneSnapshot.grasp_contact`), on the T41 a analytic rig."""
from __future__ import annotations

import numpy as np
import pytest

from tests.trajopt.test_t41a_obstacle_margin import M, ROWS, TABLE_PLANE, _clear, _rig, _scene


def _gc(latch, *, held=False, zone=None, radius=0.30, auth=(True,) * 4, centroid=(0.0, 0.0, 0.033)):
    return {"latch": latch, "authorized": np.asarray(auth, bool), "held_exempt": held,
            "zone_allowance": zone, "zone_radius": radius, "centroid": np.asarray(centroid)}


def _pair(gc, mask=(True,) * 4):
    lin, centres, field, points = _rig()
    base = dict(margin=M, mask=list(mask), support_planes=TABLE_PLANE)
    off = _clear(lin, centres, _scene(field, points, **base))
    on = _clear(lin, centres, _scene(field, points, grasp_contact=gc, **base))
    return off, on, lin


def test_none_is_v11ab():
    lin, centres, field, points = _rig()
    a = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE))
    b = _clear(lin, centres, _scene(field, points, margin=M, support_planes=TABLE_PLANE, grasp_contact=None))
    assert np.array_equal(a, b)


def test_pb4_held_drops_only_the_authorized_rows_on_the_held_object():
    # while HELD the attention target is the destination: the held object is in the field the hand reads
    # (no target-free row reaches past it), so the row is classified by distance to the manipulated object
    off, on, lin = _pair(_gc("held", held=True), mask=(False,) * 4)
    assert on[ROWS["apple"]] > off[ROWS["apple"]] + 0.9          # the hand on the object it holds: cannot bind
    for name in ("bar", "table", "box", "held"):
        assert on[ROWS[name]] == off[ROWS[name]], name             # obstacles, support, the held object's own rows
    assert lin._last_grasp_contact["held_exempt_rows"] == 1


@pytest.mark.parametrize("latch", ["latched", "closing", "searching"])
def test_pb4_held_rule_waits_for_held(latch):
    off, on, _ = _pair(_gc(latch, held=True), mask=(False,) * 4)
    assert np.array_equal(off, on)


def test_pb5_zone_lets_the_hand_reach_the_target_and_the_support_under_it():
    off, on, lin = _pair(_gc("closing", zone=0.010))
    for name in ("apple", "table"):
        assert on[ROWS[name]] - off[ROWS[name]] == pytest.approx(0.010 + max(0.0, off[ROWS[name]] - on[ROWS[name]]), abs=0.0105), name
        assert on[ROWS[name]] >= off[ROWS[name]] + 0.0099, name
    for name in ("bar", "box", "held"):
        assert on[ROWS[name]] == off[ROWS[name]], name             # obstacle-class rows and held rows never change
    assert lin._last_grasp_contact["zone_rows"] == 2


def test_pb5_zone_is_local_and_only_for_authorized_links_and_grasp_states():
    off, on, _ = _pair(_gc("closing", zone=0.010, radius=0.10))   # the table row is 250 mm away
    assert on[ROWS["table"]] == off[ROWS["table"]] and on[ROWS["apple"]] > off[ROWS["apple"]]
    off, on, _ = _pair(_gc("closing", zone=0.010, auth=(False,) * 4))
    assert np.array_equal(off, on)
    for latch in ("held", "searching", "placed"):
        off, on, _ = _pair(_gc(latch, zone=0.010))
        assert np.array_equal(off, on), latch
