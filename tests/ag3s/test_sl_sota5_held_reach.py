"""SL-SOTA5 (2026-11-11, dev-57) — SL-PE8's reach has a floor (`clustering.scene_obstacle_held_motion_reach`).

The reach (fingertips -> a point that may be the held object) was the perceived target's extent + 20 mm. When the perception
holds a piece of the object, the rest of it stays in the scene obstacle-object set while it is carried (E1a
...-ntc-pt6-pg4-tro task 0 init 6: a 61 mm piece of a 140 mm bowl -> reach 81 mm, the hand blocked 23 calls by the bowl's own
far wall). Off by default.
"""
from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.stages.held_motion import HeldMotionTracker

from tests.ag3s.test_sl_pe8_held_motion import _T, _bowl

ON = {"scene_obstacle_objects": True, "scene_obstacle_held_motion": True}


def test_default_off_and_validated():
    assert ClusteringConfig().scene_obstacle_held_motion_reach is None
    AG3SConfig().with_overrides({"clustering": {**ON, "scene_obstacle_held_motion_reach": 0.15}})
    for bad in ({"scene_obstacle_objects": True, "scene_obstacle_held_motion_reach": 0.15},        # needs SL-PE8
                {**ON, "scene_obstacle_held_motion_reach": 0.0}, {**ON, "scene_obstacle_held_motion_reach": 0.5},
                {**ON, "scene_obstacle_held_motion_reach": True}):
        with pytest.raises(AG3SConfigError):
            AG3SConfig().with_overrides({"clustering": bad})


def _carry(tracker):
    """A 140 mm bowl held at its rim; the perception knows a small piece next to the fingers. Returns the fraction of the
    bowl's far part (100-140 mm from the fingertips) removed after three 40 mm moves, and the reach."""
    hand = np.array([0.5, 0.0, 0.30])
    tip = hand - [0, 0, 0.05]
    centre = tip + [0.07, 0.0, 0.0]                       # the bowl hangs to one side of the fingertips
    bowl = _bowl(centre, n=600, r=0.07)
    piece = bowl[np.linalg.norm(bowl - tip, axis=1) <= 0.035]
    tracker.update(bowl, _T(*hand), tip, held=False, preheld_geometry=piece)
    far = np.linalg.norm(bowl - tip, axis=1) >= 0.10
    mask = np.zeros(len(bowl), bool)
    for k in range(1, 4):
        shift = np.array([0.0, 0.04 * k, 0.0])
        mask = tracker.update(bowl + shift, _T(*(hand + shift)), tip + shift, held=True)
    return float(mask[far].mean()), tracker.last["reach_mm"]


def test_reach_floor_takes_the_rest_of_the_object():
    frac, reach = _carry(HeldMotionTracker())
    assert reach < 95.0 and frac < 0.2                      # as before: the far wall is outside the reach
    frac, reach = _carry(HeldMotionTracker(reach_min_m=0.15))
    # not 100 %: a point the previous request's cloud also explains ('at rest') is kept, as in SL-PE8
    assert reach == pytest.approx(150.0) and frac > 0.7


def test_reach_floor_does_not_take_a_static_neighbour():
    tr = HeldMotionTracker(reach_min_m=0.15)
    hand = np.array([0.5, 0.0, 0.30])
    tip = hand - [0, 0, 0.05]
    bowl = _bowl(tip + [0.07, 0.0, 0.0], n=600, r=0.07)
    pot = _bowl(tip + [0.0, 0.20, 0.0], seed=1, n=300, r=0.04)   # an obstacle the hand moves toward, never moving itself
    tr.update(np.r_[bowl, pot], _T(*hand), tip, held=False, preheld_geometry=bowl[np.linalg.norm(bowl - tip, axis=1) <= 0.06])
    for k in range(1, 4):
        shift = np.array([0.0, 0.03 * k, 0.0])
        mask = tr.update(np.r_[bowl + shift, pot], _T(*(hand + shift)), tip + shift, held=True)
        assert not mask[len(bowl):].any()                   # the pot (within 150 mm of the fingertips by now) stays


def test_flag_and_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss

    base = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-"
    name = base + "pe12345811-to0-om15p10s40-ntc-pt6-pg4-gr-hr150-tro"
    assert name in pf.SOTA_HELD_VERSIONS and set(pf.SOTA_HELD_VERSIONS) <= set(pf.GT15_VERSIONS)
    v = tuple(pf.AG3S_VERSIONS[name])
    assert v[-3:] == ("--scene-obstacle-held-reach-mm", "150", "--ee-action-qp-translation-only")
    for n in pf.SOTA_HELD_VERSIONS:
        assert pf.version_server_flags(n)
    p = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(pf.AG3S_VERSIONS[name]))
    assert ss.t43t_identity_section(p)["scene_obstacle_held_motion_reach"] == pytest.approx(0.15)
    p0 = pf.parse_v11ab_args(argv=pf.version_server_argv(pf.PE_VERSIONS[6]) + list(pf.AG3S_VERSIONS[pf.PE_VERSIONS[6]]))
    assert "scene_obstacle_held_motion_reach" not in ss.t43t_identity_section(p0)
    assert "scene_obstacle_held_reach_mm" in srv.PORTED_FINAL_CONFIG_DESTS
    assert "scene_obstacle_held_reach_mm" not in pf.version_flag_values(pf.parse_v11ab_args())
    ap = ss.build_parser()
    from tests.trajopt.test_t88_passthrough_when_clear import BASE as ARGV
    with pytest.raises(SystemExit):                         # needs --scene-obstacle-held-motion
        ss.reject_bad_flag_combinations(ap, ap.parse_args(ARGV + ["--scene-obstacle-held-reach-mm", "150"]))
