"""SL-GTTO5 F4 — `--grasp-zone-release` (user ruling 2026-11-05 ~09:3x "best direction", lead design), CPU.

1. linearizer (`_grasp_contact_margin`, T41 a analytic rig): LATCHED / CLOSING, authorized rows whose centre lies within
   `zone_radius` (horizontal) of the manipulated centroid are released whatever their class (target, support, obstacle);
   unauthorized rows, rows outside the zone, the held object's rows and other latch states never change; key absent =
   pb4 / pb5 bit-identical.
2. SafePolicy `_grasp_contact` carries `zone_release` only when on; serve_safe flag + `apply_grasp_contact_flags`.
3. profile `AG3S-commonto-gt2-termsoff-pass-zr` = termsoff-pass + the flag; read back.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from tests.trajopt.test_pb45_grasp_contact import _gc, _pair
from tests.trajopt.test_t41a_obstacle_margin import ROWS, _rig


def _zr(latch, **kw):
    g = _gc(latch, **kw)
    g["zone_release"] = True
    return g


def _horiz():
    _, centres, _, _ = _rig()
    return np.linalg.norm(centres[0, :, :2], axis=1)       # centroid (0, 0, 0.033) of _gc


@pytest.mark.parametrize("latch", ["latched", "closing"])
def test_release_inside_the_zone_whatever_the_class(latch):
    h = _horiz()
    radius = 0.28
    off, on, lin = _pair(_zr(latch, radius=radius))
    released = 0
    for name in ("bar", "apple", "table", "box"):
        i = ROWS[name]
        if h[i] <= radius:
            assert on[i] > off[i] + 0.9, name                  # cannot bind
            released += 1
        else:
            assert on[i] == off[i], name
    assert on[ROWS["held"]] == off[ROWS["held"]]                 # the held object's rows are never authorized
    assert released >= 2 and lin._last_grasp_contact["zone_release_rows"] == released


def test_release_is_local_authorized_only_and_only_before_held():
    off, on, _ = _pair(_zr("closing", radius=0.0))
    assert np.array_equal(off, on)
    off, on, _ = _pair(_zr("closing", auth=(False,) * 4))
    assert np.array_equal(off, on)
    for latch in ("held", "searching", "placed"):
        off, on, lin = _pair(_zr(latch))
        assert np.array_equal(off, on), latch
        assert lin._last_grasp_contact["zone_release_rows"] == 0


def test_key_absent_is_pb5():
    a_off, a_on, lin = _pair(_gc("closing", zone=0.010))
    assert "zone_release_rows" not in lin._last_grasp_contact
    g = _gc("closing", zone=0.010)
    g["zone_release"] = False
    _, b_on, _ = _pair(g)
    assert np.array_equal(a_on, b_on)


def test_safe_policy_and_flag():
    from AG3S.trajopt import serve_safe as ss
    from tests.trajopt.test_sl6h_placement_contact import _policy

    sp = _policy()
    assert sp.grasp_zone_release is False and sp._grasp_contact(None) is None
    act = ss.build_parser()._option_string_actions["--grasp-zone-release"]
    assert act.dest == "grasp_zone_release" and act.default is False
    out = ss.apply_grasp_contact_flags(sp, types.SimpleNamespace(grasp_zone_release=True))
    assert out["grasp_zone_release"] is True and sp.grasp_zone_release is True
    sp.constraint_robot_model = types.SimpleNamespace(sphere_link_names=("panda_hand", "panda_link5"))
    sp._authorized_links = lambda cs: ("panda_hand",)
    gc = sp._grasp_contact(None)
    assert gc["zone_release"] is True and gc["zone_allowance"] is None
    assert list(gc["authorized"]) == [True, False]
    off = ss.apply_grasp_contact_flags(types.SimpleNamespace(), types.SimpleNamespace(grasp_zone_release=False))
    assert "grasp_zone_release" not in off


def test_profile_version(gpu_ok):
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from tests.benchmark.test_sl_gtto_gt_perception import _args

    v = "AG3S-commonto-gt2-termsoff-pass-zr"
    assert pf.GT5_VERSIONS == (v,)
    assert list(pf.AG3S_VERSIONS[v]) == list(pf.AG3S_VERSIONS["AG3S-commonto-gt2-termsoff-pass"]) + ["--grasp-zone-release"]
    assert "grasp_zone_release" in srv.PORTED_FINAL_CONFIG_DESTS
    assert pf.is_final_version(v) and pf.version_requires_oracle_segmentation(v)
    assert pf.version_server_argv(v) == pf.version_server_argv("AG3S-commonto-gt2")
    args = _args(v)
    assert pf.match_ag3s_version(args) == v and args.grasp_zone_release is True
    assert pf.version_flag_values(args)["grasp_zone_release"] is True
    assert "grasp_zone_release" not in pf.version_flag_values(_args("AG3S-commonto-gt2-termsoff-pass"))
    for name, flags in pf.AG3S_VERSIONS.items():
        if name != v and name not in getattr(pf, "GT6_VERSIONS", ()) + getattr(pf, "GT7_VERSIONS", ()) + getattr(pf, "GT8_VERSIONS", ()) + getattr(pf, "GT9_VERSIONS", ()) + getattr(pf, "GT10_VERSIONS", ()) + getattr(pf, "GT11_VERSIONS", ()) + getattr(pf, "GT12_VERSIONS", ()) + getattr(pf, "GT13_VERSIONS", ()) + getattr(pf, "GT14_VERSIONS", ()) + getattr(pf, "GT15_VERSIONS", ()):   # build on zr
            assert "--grasp-zone-release" not in flags, name


from tests.benchmark.test_sl6a_panda_final import gpu_ok  # noqa: E402,F401 (fixture)
