"""SL-SOTA7 (2026-11-11, dev-57) — while HELD, the grasping hand's rows over the held object's footprint are not
obstacle-object rows (`sqp.obstacle_hand_beneath`, `--obstacle-hand-beneath`, versions ``-hb``); the held object's own
rows are not either when the set point is under its bottom (+ 2 voxels).

E1a ...-ntc-pt6-pg4-hr150-tro task 2 init 5 / 6 / 13: the destination (plate) was not or only partly registered, the place
was refused for a finger row against the plate inside the obstacle margin, the QP then moved the hand off the plate
(|Da| 1.4-1.6) and the bowl was put down beside it. Off by default.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from AG3S.trajopt.linearize import CollisionLinearizer
from tests.trajopt.test_sl_gtto16_obstacle_tolerance import BASE, _PlaneField, _violating


def _stub():
    """Three robot spheres (0, 1 = the grasping hand, 2 = an arm link) and two held points (radius 20 mm) 100 mm under
    the hand, at two plan steps; the held object moves 50 mm in x between them."""
    lin = types.SimpleNamespace(n_spheres=3, query_radii=np.array([0.02, 0.02, 0.05, 0.02, 0.02]))
    centres = np.zeros((2, 5, 3))
    for k, x in enumerate((0.0, 0.05)):
        centres[k, :3] = [[x, 0.0, 0.20], [x + 0.02, 0.0, 0.20], [x, 0.0, 0.50]]
        centres[k, 3:] = [[x, 0.0, 0.10], [x + 0.04, 0.0, 0.10]]
    scene = types.SimpleNamespace(grasp_contact={"latch": "held", "authorized": np.array([True, True, False])})
    return lin, centres, scene


def _under(lin, centres, scene, nearest):
    return CollisionLinearizer._hand_rows_over_held_footprint(lin, nearest, centres, scene, 0.005)


def test_hand_rows_over_the_held_footprint():
    lin, centres, scene = _stub()
    q = np.zeros((2, 5, 3))
    q[0, :, :] = [0.04, 0.0, 0.0]            # step 0: the surface point under the held object (the plate)
    q[1, :, :] = [0.04, 0.0, 0.0]            # step 1: the same point, the footprint now covers x = 0.03 .. 0.11
    got = _under(lin, centres, scene, q)
    assert got.shape == (2, 5) and got[:, :2].all()
    assert not got[:, 2].any()                                 # the arm row is never touched
    assert got[:, 3:].all()                                    # the held rows: the point is under their bottom (0.08)


def test_outside_the_footprint_or_above_the_top_stays_an_obstacle_row():
    lin, centres, scene = _stub()
    q = np.zeros((2, 5, 3))
    q[:, :, :] = [0.04, 0.10, 0.0]           # 100 mm beside the held object: 70 mm outside its discs + 10 mm
    assert not _under(lin, centres, scene, q).any()
    q[:, :, :] = [0.04, 0.0, 0.13]           # inside the footprint, 10 mm above the held object's top (0.12)
    assert not _under(lin, centres, scene, q).any()
    q[:, :, :] = [0.04, 0.0, 0.12]           # at the top: still under for the hand, not for the held rows
    got = _under(lin, centres, scene, q)
    assert got[:, :2].all() and not got[:, 3:].any()
    q[:, :, :] = [0.04, 0.0, 0.09]           # the held object's bottom (0.08) + 2 voxels (0.01): the band's upper edge
    assert _under(lin, centres, scene, q)[:, 3:].all()
    q[:, :, :] = [0.04, 0.0, 0.095]          # an obstacle beside the held object at its own height: stays an obstacle row
    assert not _under(lin, centres, scene, q)[:, 3:].any()
    q[0, :, :] = [0.11, 0.0, 0.0]            # step 0 footprint ends at x = 0.07 (+ 0.01); step 1 reaches it
    q[1, :, :] = [0.11, 0.0, 0.0]
    got = _under(lin, centres, scene, q)
    assert not got[0].any() and got[1, :2].all()


@pytest.mark.parametrize("gc", [None, {"latch": "latched", "authorized": np.ones(3, bool)}, {"latch": "held"},
                                {"latch": "held", "authorized": np.zeros(3, bool)}])
def test_only_while_held_with_an_authorized_hand(gc):
    lin, centres, _ = _stub()
    assert _under(lin, centres, types.SimpleNamespace(grasp_contact=gc), np.zeros((2, 5, 3))) is None


def test_no_held_rows_no_rule():
    lin, centres, scene = _stub()
    assert _under(lin, centres[:, :3], scene, np.zeros((2, 3, 3))) is None


def test_mask_drops_those_rows_only_when_the_switch_is_on(monkeypatch):
    sc = _violating(0.01)
    lin = sc.linearizer
    centres = np.asarray(lin.sphere_states(sc.reference, sc.q_now)[0])
    h, s = centres.shape[:2]
    a = centres[2, s // 2]
    clear = np.full((h, s), 0.05)
    clear[2, s // 2] = -0.012                                  # nearest surface = the plane right under it, in the set
    scene = types.SimpleNamespace(esdf=_PlaneField(np.zeros(3)), target_free_mask=None, target_volume=None,
                                  support_planes=None, destination_label=None, manipulated_spheres=None,
                                  manipulated_points=None, grasp_contact={"latch": "held"},
                                  obstacle_object_points=np.array([[a[0], a[1], 0.0]]),
                                  n_slots=sc.scene.n_slots, n_planes=sc.scene.n_planes)
    cls = type(lin)
    calls = []

    def fake(self, nearest, c, sc_, voxel):
        calls.append(np.asarray(nearest).shape)
        out = np.zeros((h, s), bool)
        out[2, s // 2] = True
        return out

    monkeypatch.setattr(cls, "_esdf_clearance", lambda self, c, sc_: clear)
    monkeypatch.setattr(cls, "_hand_rows_over_held_footprint", fake)
    off = lin.obstacle_object_rows(sc.reference, sc.q_now, scene, class_tests=True)
    assert off[0] == pytest.approx(-0.012) and not calls
    on = lin.obstacle_object_rows(sc.reference, sc.q_now, scene, class_tests=True, hand_beneath=True)
    # the other rows whose nearest surface point is within the surface tolerance of the set stay (50 mm clear)
    assert calls == [(h, s, 3)] and on[0] == pytest.approx(0.05) and on[1]["n_beneath"] == 1
    assert on[1]["n_obstacle_rows"] == off[1]["n_obstacle_rows"] - 1
    floor, obst = lin.obstacle_row_floor(centres, scene, 0.015, class_tests=True, hand_beneath=True)
    assert not obst[2, s // 2]
    floor, obst = lin.obstacle_row_floor(centres, scene, 0.015, class_tests=True)
    assert obst[2, s // 2]


def test_config_flag_and_versions():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.config import SqpConfig, TrajOptConfigError
    from AG3S.trajopt.sqp import _obstacle_row_kwargs

    assert SqpConfig().obstacle_hand_beneath is False
    assert "hand_beneath" not in _obstacle_row_kwargs(SqpConfig())
    assert _obstacle_row_kwargs(types.SimpleNamespace(obstacle_hand_beneath=True))["hand_beneath"] is True
    with pytest.raises(TrajOptConfigError):
        SqpConfig(obstacle_hand_beneath=True).validate()       # needs the row classification
    ap = ss.build_parser()
    with pytest.raises(SystemExit):
        ss.reject_bad_flag_combinations(ap, ap.parse_args(BASE + ["--passthrough-when-clear", "--obstacle-hand-beneath"]))
    a = ap.parse_args(BASE + ["--passthrough-when-clear", "--passthrough-obstacle-tolerance-mm", "0", "--obstacle-hand-beneath"])
    assert ss.passthrough_overrides(a)["obstacle_hand_beneath"] is True
    assert "obstacle_hand_beneath" not in ss.passthrough_overrides(ap.parse_args(BASE + ["--passthrough-when-clear"]))
    assert ss.trajopt_config_from_args(a).sqp.obstacle_hand_beneath is True
    base = "AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-"
    name = base + "pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-tro"
    assert name in pf.SOTA_HANDB_VERSIONS and set(pf.SOTA_HANDB_VERSIONS) <= set(pf.GT15_VERSIONS)
    assert tuple(pf.AG3S_VERSIONS[name])[-2:] == ("--obstacle-hand-beneath", "--ee-action-qp-translation-only")
    assert base + "pe12345811-to0-om15p10s40-ntc-pt6-pg4-hr150-hb-qs-tro" in pf.SOTA_SCALED_VERSIONS
    for n in pf.SOTA_HANDB_VERSIONS:
        assert pf.version_server_flags(n)
    assert "obstacle_hand_beneath" in srv.PORTED_FINAL_CONFIG_DESTS
    assert "obstacle_hand_beneath" not in pf.version_flag_values(pf.parse_v11ab_args())
    args = pf.parse_v11ab_args(argv=pf.version_server_argv(name) + list(pf.AG3S_VERSIONS[name]))
    assert args.obstacle_hand_beneath is True
