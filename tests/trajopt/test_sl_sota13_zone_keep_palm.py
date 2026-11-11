"""SL-SOTA13 (2026-11-12, dev-57) — `--grasp-zone-keep-palm-obstacles` (versions ``-zp``): inside the grasp zone
(SL-GTTO5 F4 releases every hand / finger row within `zone_radius` of the manipulated object's centroid) the palm's rows
whose nearest surface is an obstacle object are not released.

E1a ...-pt6-pg4-hr150-hb-qs-tro task 0 init 7 / 13 (success + collision 32 / 12 mm): the palm sphere on the storage box next
to the bowl read -5 .. -13 mm, lay 72 / 93 mm from the bowl's centroid and was released. Offline (zone_rows.py, GT labels):
in the 20 safe successes no released palm row with a predicted penetration lies on a non-target object. Off by default.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.linearize import _OBSTACLE, _SUPPORT, _TARGET, CollisionLinearizer


class _Esdf:
    """The nearest surface is the plane x = 0.2 (gradient -x for points left of it)."""

    def gradient(self, flat):
        g = np.zeros((len(flat), 3))
        g[:, 0] = -1.0
        return g


def _margin(keep, cls_row, oop=((0.2, 0.0, 0.1),), latch="latched"):
    """Four spheres 50 mm from the centroid (all inside the 100 mm zone): 0 = palm, 1 = palm, 2 = finger, 3 = arm (not
    authorized). Every row is 30 mm from the plane x = 0.2, i.e. its surface point is (0.2, y, 0.1)."""
    lin = types.SimpleNamespace(n_spheres=4, _obstacle_rows=lambda d, flat, scene, d_object: np.asarray([cls_row], np.int8),
                                _on_obstacle_object_surface=CollisionLinearizer._on_obstacle_object_surface)
    flat = np.array([[0.17, 0.0, 0.1], [0.17, 0.05, 0.1], [0.17, 0.0, 0.1], [0.17, 0.0, 0.1]])
    scene = types.SimpleNamespace(esdf=_Esdf(), obstacle_object_points=None if oop is None else np.asarray(oop, float).reshape(-1, 3))
    gc = {"latch": latch, "authorized": np.array([True, True, True, False]), "held_exempt": False, "zone_allowance": None,
          "zone_radius": 0.10, "centroid": np.array([0.12, 0.0, 0.0]), "zone_release": True}
    if keep:
        gc["zone_keep_palm_obstacles"] = {"links": np.array([True, True, False, False]), "surface_tol": 0.02}
    m = CollisionLinearizer._grasp_contact_margin(lin, gc, np.full((1, 4), 0.01), np.full((1, 4), 0.03), flat, scene,
                                                  np.full((1, 4), 0.2), (1, 4), 4)
    return m.reshape(-1).tolist(), dict(lin._last_grasp_contact)


def test_off_releases_every_authorized_row_and_adds_no_key():
    m, rec = _margin(False, [_OBSTACLE] * 4)
    assert m == [-1.0, -1.0, -1.0, 0.01] and rec["zone_release_rows"] == 3 and "zone_kept_palm_rows" not in rec


def test_the_palm_row_on_an_obstacle_object_is_kept():
    m, rec = _margin(True, [_OBSTACLE] * 4)
    # sphere 0: palm, surface point (0.2, 0, 0.1) on the set -> kept. sphere 1: palm, surface point 50 mm from the set ->
    # released. sphere 2: finger -> released. sphere 3: arm -> never released.
    assert m == [0.01, -1.0, -1.0, 0.01]
    assert rec["zone_kept_palm_rows"] == 1 and rec["zone_release_rows"] == 2


@pytest.mark.parametrize("cls", [_TARGET, _SUPPORT])
def test_the_palm_on_the_target_or_a_support_is_still_released(cls):
    m, rec = _margin(True, [cls] * 4)
    assert m[:3] == [-1.0, -1.0, -1.0] and rec["zone_kept_palm_rows"] == 0


def test_no_obstacle_set_or_an_empty_one_keeps_nothing():
    assert _margin(True, [_OBSTACLE] * 4, oop=None)[0][:3] == [-1.0, -1.0, -1.0]
    assert _margin(True, [_OBSTACLE] * 4, oop=())[0][:3] == [-1.0, -1.0, -1.0]


def test_only_while_latched_or_closing():
    m, rec = _margin(True, [_OBSTACLE] * 4, latch="held")
    assert m == [0.01] * 4 and rec["zone_release_rows"] == 0 and "zone_kept_palm_rows" not in rec


def test_flag_plumbing_and_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.safe_policy import SafePolicy
    from tests.trajopt.test_t88_passthrough_when_clear import BASE as ARGV

    assert SafePolicy.grasp_zone_keep_palm_obstacles is None
    ap = ss.build_parser()
    assert ap.parse_args(ARGV).grasp_zone_keep_palm_obstacles is False
    with pytest.raises(SystemExit):
        ss.reject_bad_flag_combinations(ap, ap.parse_args(ARGV + ["--grasp-zone-keep-palm-obstacles"]))
    safe = types.SimpleNamespace()
    out = ss.apply_grasp_contact_flags(safe, ap.parse_args(ARGV + ["--grasp-zone-release", "--grasp-zone-keep-palm-obstacles",
                                                                   "--obstacle-surface-tol-mm", "40"]))
    assert safe.grasp_zone_keep_palm_obstacles == pytest.approx(0.04) and out["grasp_zone_keep_palm_obstacles_m"] == pytest.approx(0.04)
    safe = types.SimpleNamespace()
    out = ss.apply_grasp_contact_flags(safe, ap.parse_args(ARGV + ["--grasp-zone-release"]))
    assert not hasattr(safe, "grasp_zone_keep_palm_obstacles") and "grasp_zone_keep_palm_obstacles_m" not in out
    assert "grasp_zone_keep_palm_obstacles" in srv.PORTED_FINAL_CONFIG_DESTS
    base = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150"
    for mid in ("-qs", "-hb-qs", "-hb-qs-rs"):
        name = base + mid + "-zp-tro"
        assert name in pf.SOTA_ZONEPALM_VERSIONS and name in pf.GT15_VERSIONS
        v = tuple(pf.AG3S_VERSIONS[name])
        assert v[-1] == "--grasp-zone-keep-palm-obstacles" and v[:-1] == tuple(pf.AG3S_VERSIONS[base + mid + "-tro"])
        args = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(v))
        assert args.grasp_zone_keep_palm_obstacles is True and args.grasp_zone_release is True
        assert args.grasp_zone_radius_mm == 100.0 and float(args.obstacle_surface_tol_mm) == 40.0
        assert pf.version_server_flags(name)
    for name, flags in pf.AG3S_VERSIONS.items():
        if name not in pf.SOTA_ZONEPALM_VERSIONS:
            assert "--grasp-zone-keep-palm-obstacles" not in flags, name
    assert "grasp_zone_keep_palm_obstacles" not in pf.version_flag_values(pf.parse_v11ab_args())


def test_the_policy_passes_the_hold_link_mask():
    from AG3S.trajopt.safe_policy import SafePolicy

    sp = types.SimpleNamespace(held_hand_exempt=False, grasp_zone_allowance=None, closing_hand_exempt=False,
                               grasp_zone_release=True, grasp_zone_radius=0.1, grasp_zone_keep_palm_obstacles=0.04,
                               hold_links={"left": "panda_hand"},
                               constraint_robot_model=types.SimpleNamespace(
                                   sphere_link_names=["panda_link7", "panda_hand", "panda_leftfinger"]),
                               _authorized_links=lambda cs: ["panda_hand", "panda_leftfinger"],
                               ag3s=types.SimpleNamespace(manipulated=None),
                               _latch=types.SimpleNamespace(phase=types.SimpleNamespace(value="latched")))
    gc = SafePolicy._grasp_contact(sp, None)
    assert gc["zone_release"] is True and gc["zone_keep_palm_obstacles"]["surface_tol"] == 0.04
    assert gc["zone_keep_palm_obstacles"]["links"].tolist() == [False, True, False]
    sp.grasp_zone_keep_palm_obstacles = None
    assert "zone_keep_palm_obstacles" not in SafePolicy._grasp_contact(sp, None)
