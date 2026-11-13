"""SL-AUD6-8 (code audit of the frozen version, 2026-11-13; `local_docs/2026-11-13_frozen_config_code_audit_KO.md` §4): three
switches of the place phase, all default off.

* SL-AUD6 `clustering.scene_obstacle_history_refilter` / `--scene-obstacle-history-refilter` (version segment ``-ohr``): the
  two remembered requests of the scene obstacle set lose what this request excludes.
* SL-AUD7 `clustering.destination_leader_outside_hand` / `--destination-leader-outside-hand` (``-dlo``): the destination
  candidate is the score leader among the clusters away from the hand.
* SL-AUD8 `SafePolicy.progress_release_off` / `--ee-action-qp-progress-release-off` (``-pro``): no "toward the target"
  preference after the opening rule released.

Off: every result and record key below is the one before these switches.
"""
from __future__ import annotations

import dataclasses
import types

import numpy as np
import pytest

from AG3S.ag3s.config import AG3SConfig, AG3SConfigError, ClusteringConfig
from AG3S.ag3s.runtime.pipeline import AG3S


def _sphere(centre, radius, n=500, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return np.asarray(centre, float) + radius * v / np.linalg.norm(v, axis=1, keepdims=True)


def _near(out, c, r):
    return out is not None and len(out) > 0 and bool(np.any(np.linalg.norm(out - np.asarray(c), axis=1) < r))


PLATE_C, OBST_C, BOWL_C = [0.60, 0.20, 0.03], [0.40, -0.20, 0.06], [0.50, 0.00, 0.05]
PLATE = _sphere(PLATE_C, 0.06, seed=1)
OBST = _sphere(OBST_C, 0.04, seed=2)
BOWL = _sphere(BOWL_C, 0.04, seed=3)


# --- SL-AUD6: the remembered requests of the scene obstacle set ---------------------------------------------------------
def _two_calls(first, second, **kw):
    """`first` / `second` = (cloud, target_points, destination_points) of two consecutive requests."""
    me = types.SimpleNamespace(_obstacle_history=[])
    AG3S._scene_obstacle_points(me, first[0], None, first[1], first[2], **kw)
    return AG3S._scene_obstacle_points(me, second[0], None, second[1], second[2], **kw)


def test_history_refilter_defaults_off_and_is_validated():
    assert ClusteringConfig().scene_obstacle_history_refilter is False
    with pytest.raises(AG3SConfigError):                                  # needs the scene obstacle set
        AG3SConfig().with_overrides({"clustering": {"scene_obstacle_history_refilter": True}})
    AG3SConfig().with_overrides({"clustering": {"scene_obstacle_history_refilter": True, "scene_obstacle_objects": True}})
    with pytest.raises(AG3SConfigError):
        AG3SConfig().with_overrides({"clustering": {"scene_obstacle_history_refilter": 1, "scene_obstacle_objects": True}})


def test_a_destination_registered_now_leaves_the_remembered_requests_too():
    cloud = np.concatenate([PLATE, OBST])
    # off: the plate, registered at the second request, is still an obstacle object through the first request's points
    out, rec = _two_calls((cloud, None, None), (cloud, None, PLATE))
    assert _near(out, PLATE_C, 0.07) and _near(out, OBST_C, 0.05) and "history_refilter" not in rec
    # on: it is gone at the request that registers it; the obstacle is untouched
    out, rec = _two_calls((cloud, None, None), (cloud, None, PLATE), history_refilter=True)
    assert not _near(out, PLATE_C, 0.07) and _near(out, OBST_C, 0.05)
    assert rec["history_refilter"]["n_removed"] > 0 and rec["history_refilter"]["n_excluded_now"] == len(PLATE)


def test_a_target_adopted_now_leaves_the_remembered_requests_too():
    cloud = np.concatenate([BOWL, OBST])
    out, _ = _two_calls((cloud, None, None), (cloud, BOWL, None))
    assert _near(out, BOWL_C, 0.05)                                       # off: the new target's earlier points stay
    out, _ = _two_calls((cloud, None, None), (cloud, BOWL, None), history_refilter=True)
    assert not _near(out, BOWL_C, 0.05) and _near(out, OBST_C, 0.05)


def test_an_obstacle_not_seen_this_request_is_still_remembered():
    """The union's purpose (an obstacle that flickers out of one frame stays) is kept: nothing excludes it."""
    for kw in ({}, {"history_refilter": True}):
        out, _ = _two_calls((np.concatenate([PLATE, OBST]), None, None), (PLATE, None, PLATE), **kw)
        assert _near(out, OBST_C, 0.05), kw
    # and the filtered memory is what the next request starts from: the plate does not come back at a third request
    me = types.SimpleNamespace(_obstacle_history=[])
    cloud = np.concatenate([PLATE, OBST])
    AG3S._scene_obstacle_points(me, cloud, None, None, None, history_refilter=True)
    AG3S._scene_obstacle_points(me, cloud, None, None, PLATE, history_refilter=True)
    out, rec = AG3S._scene_obstacle_points(me, cloud, None, None, PLATE, history_refilter=True)
    assert not _near(out, PLATE_C, 0.07) and rec["history_refilter"]["n_removed"] == 0


def test_nothing_excluded_changes_nothing():
    cloud = np.concatenate([PLATE, OBST])
    a, _ = _two_calls((cloud, None, None), (cloud, None, None))
    b, rec = _two_calls((cloud, None, None), (cloud, None, None), history_refilter=True)
    assert np.array_equal(a, b) and rec["history_refilter"] == {"n_removed": 0, "n_excluded_now": 0}


# --- SL-AUD7: the destination candidate away from the hand ----------------------------------------------------------------
def test_leader_outside_hand_defaults_off_and_is_validated():
    assert ClusteringConfig().destination_leader_outside_hand is False
    with pytest.raises(AG3SConfigError):                                  # needs destination_held_leader
        AG3SConfig().with_overrides({"clustering": {"destination_leader_outside_hand": True}})
    AG3SConfig().with_overrides({"clustering": {"destination_after_held": True, "destination_held_leader": True,
                                                "destination_leader_outside_hand": True}})


def test_registry_takes_the_callers_leader():
    from AG3S.ag3s.stages.admissibility import DestinationRegistry
    from tests.ag3s.test_admissibility import _clusters

    piece, plate = _sphere([0.50, -0.15, 1.05], 0.03, seed=5), _sphere(PLATE_C, 0.06, seed=6)
    clusters, _cloud = _clusters((piece, 0.9), (plate, 0.5))             # score order: the piece leads
    points = [piece, plate]
    grasp = [True, False]                                                 # the piece near the hand keeps "graspable"

    def run(**kw):
        reg = DestinationRegistry(frames=3, min_score=0.1, tolerance=0.05)
        events = [reg.observe(clusters, points, grasp, frame=k, **kw) for k in range(reg.frames + 1)]
        return reg, events

    reg, events = run()
    assert reg.current is None and set(events) == {None}                  # as before: the graspable leader resets the streak
    reg, events = run(leader_index=1)
    assert "registered" in events and np.allclose(reg.current.points, plate)
    reg, events = run(leader_index=-1)                                          # the caller found no cluster away from the hand
    assert reg.current is None and reg._streak == 0
    reg, events = run(leader_index=0)                                           # the caller's leader is the score leader: as before
    assert reg.current is None


def test_a_held_piece_that_leads_the_score_does_not_block_the_plate():
    from AG3S.ag3s.stages.admissibility import HandSpheres
    from tests.ag3s.test_sl_pe_wiring import APPLE, BASE, _apc, _confirm, _grid, _step

    plate = _grid([0.30, 0.20, 0.70], [0.44, 0.34, 0.70])
    plate = np.concatenate([plate, plate + [0, 0, 0.010]])
    hand_c = np.array([0.50, -0.15, 1.10])
    hand = HandSpheres(hand_c[None, :], np.array([0.04]), ("panda_hand",))
    piece = _sphere(hand_c + [0.0, 0.0, -0.06], 0.02, n=300, seed=7)      # a piece of the held object under the palm

    def run(cfg):
        c = _confirm(cfg)
        _step(c, _apc((APPLE, 0.9)), cfg)
        for _ in range(5):
            c.note_hand(hand)
            _step(c, _apc((piece, 0.95), (plate, 0.6)), cfg, frozen=True, pre_attach=False)
        return c.destination

    leader = dataclasses.replace(BASE, destination_after_held=True, destination_held_leader=True)
    assert run(leader) is None                                            # the piece leads and is graspable: no candidate
    dest = run(dataclasses.replace(leader, destination_leader_outside_hand=True))
    assert dest is not None and abs(float(np.asarray(dest.points)[:, 2].mean()) - 0.705) < 0.01   # the plate
    # with the plate alone both register it (the switch does not change the case SL-PE5 was made for)
    def alone(cfg):
        c = _confirm(cfg)
        _step(c, _apc((APPLE, 0.9)), cfg)
        for _ in range(5):
            c.note_hand(hand)
            _step(c, _apc((plate, 0.95)), cfg, frozen=True, pre_attach=False)
        return c.destination is not None
    assert alone(leader) and alone(dataclasses.replace(leader, destination_leader_outside_hand=True))


# --- SL-AUD8: no "toward the target" after the release ------------------------------------------------------------------
def test_the_opening_rule_marks_the_release_until_the_next_closing_attempt():
    pytest.importorskip("osqp")
    from AG3S.trajopt.grasp_latch import GraspPhase
    from tests.benchmark.test_sl14_opening_rule import _latch, _sig

    latch = _latch(opening_rule_moved_min=0.05)
    assert latch._rule_released is False

    def step(cmd, opening, k, reach=0.05):
        return latch.observe_grasp(_sig(cmd, opening, seq=k), reach_m=reach, manipulated_id=0, manipulated_state="occluded",
                                   reach_context={"geometry_frame": k})

    for k, (cmd, opening) in enumerate([(1.0, 0.83), (0.0, 0.40), (0.0, 0.10), (0.0, 0.10)], start=1):
        ev = step(cmd, opening, k)
    assert ev.attach and latch.phase is GraspPhase.HELD and latch._rule_released is False
    ev = step(1.0, 0.60, 5)                                               # open command, the opening rises: released
    assert ev.detach and latch.phase is GraspPhase.LATCHED and latch._rule_released is True
    step(1.0, 0.95, 6)
    assert latch._rule_released is True                                   # still open, nothing new is being grasped
    step(0.0, 0.50, 7)                                                    # a new closing attempt
    assert latch.phase is GraspPhase.CLOSING and latch._rule_released is False
    latch._rule_released = True
    latch.reset()
    assert latch._rule_released is False


def test_safe_policy_reports_the_release_only_with_the_switch():
    pytest.importorskip("osqp")
    from AG3S.trajopt.grasp_latch import GraspPhase
    from AG3S.trajopt.safe_policy import SafePolicy

    def contact(switch, released):
        sp = SafePolicy.__new__(SafePolicy)
        sp.held_hand_exempt, sp.grasp_zone_allowance, sp.closing_hand_exempt, sp.grasp_zone_release = True, None, False, True
        sp.grasp_zone_radius = 0.1
        sp.constraint_robot_model = types.SimpleNamespace(sphere_link_names=("panda_hand", "panda_link5"))
        sp._authorized_links = lambda cs: ("panda_hand",)
        sp.ag3s = types.SimpleNamespace(manipulated=None)
        sp._latch = types.SimpleNamespace(phase=GraspPhase.LATCHED, _rule_released=released)
        if switch:
            sp.progress_release_off = True
        return sp._grasp_contact(None)

    assert "released" not in contact(False, True) and "released" not in contact(True, False)
    assert contact(True, True)["released"] is True


def test_progress_is_off_after_the_release():
    pytest.importorskip("osqp")
    from tests.trajopt.test_sl_sota1_obstacle_row_margin import PROG, VIOL, _prog_solve

    target = np.array([[0.0, 0.2, -0.1]])
    before = types.SimpleNamespace(grasp_contact={"latch": "latched"}, manipulated_points=target)
    da, rec = _prog_solve(PROG, VIOL, before)
    assert np.allclose(da[:, 1], 0.4, atol=1e-4) and rec["progress"]["mode"].startswith("before the grasp")
    after = types.SimpleNamespace(grasp_contact={"latch": "latched", "released": True}, manipulated_points=target)
    da, rec = _prog_solve(PROG, VIOL, after)
    assert np.allclose(da[:, 1:], 0.0, atol=1e-5) and rec["progress"]["mode"] == "after the release"
    assert rec["progress"]["direction"] is None
    # HELD and CLOSING are decided before the key is read
    held = types.SimpleNamespace(grasp_contact={"latch": "held", "released": True}, manipulated_points=None)
    assert _prog_solve(PROG, VIOL, held)[1]["progress"]["mode"] == "held: up"


# --- flags ------------------------------------------------------------------------------------------------------------------
def test_flags_reach_the_config_and_the_policy():
    from AG3S.adapters.panda import profile as pf
    from AG3S.adapters.panda import server as srv
    from AG3S.trajopt import serve_safe as ss

    ap = ss.build_parser()
    frozen = pf.FROZEN_20261112
    argv = pf.version_server_argv(frozen) + list(pf.version_server_flags(frozen))
    base = ap.parse_args(argv)
    keys = {"scene_obstacle_history_refilter", "destination_leader_outside_hand"}
    assert not keys & set(ss.t43t_identity_section(base))                 # the frozen version: no new key
    new = ["--scene-obstacle-history-refilter", "--destination-leader-outside-hand", "--ee-action-qp-progress-release-off"]
    args = ap.parse_args(argv + new)
    ss.reject_bad_flag_combinations(ap, args)
    srv.check_unported_flags(args, None)
    sec = ss.t43t_identity_section(args)
    assert sec["scene_obstacle_history_refilter"] is True and sec["destination_leader_outside_hand"] is True
    cfg = pf.panda_ag3s_config(pf.v11ab_ag3s_config(pf.v11ab_build_kwargs(args)))
    assert cfg.clustering.scene_obstacle_history_refilter and cfg.clustering.destination_leader_outside_hand
    off = pf.panda_ag3s_config(pf.v11ab_ag3s_config(pf.v11ab_build_kwargs(base)))
    assert not off.clustering.scene_obstacle_history_refilter and not off.clustering.destination_leader_outside_hand
    safe = types.SimpleNamespace(set_latch_options=lambda **k: None)
    assert ss.apply_grasp_contact_flags(safe, args)["progress_release_off"] is True and safe.progress_release_off is True
    safe = types.SimpleNamespace(set_latch_options=lambda **k: None)
    assert "progress_release_off" not in ss.apply_grasp_contact_flags(safe, base) and not hasattr(safe, "progress_release_off")
    bare = ["--config", "x", "--checkpoint", "y", "--esdf-margin", "0", "--model-xml", "m.xml"]
    for lone in new:
        with pytest.raises(SystemExit):
            ss.reject_bad_flag_combinations(ap, ap.parse_args(bare + [lone]))
    assert {"scene_obstacle_history_refilter", "destination_leader_outside_hand",
            "ee_action_qp_progress_release_off"} <= set(srv.PORTED_FINAL_CONFIG_DESTS)
    assert not {"scene_obstacle_history_refilter", "destination_leader_outside_hand",
                "ee_action_qp_progress_release_off"} & set(pf.version_flag_values(pf.parse_v11ab_args()))
