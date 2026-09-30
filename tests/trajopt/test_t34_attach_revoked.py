"""T34 J2 — attach 뒤 `revoke_chunks` 실행 청크 동안 "정말 쥐고 있나" 를 다시 보고, 아니면
detach + `attach_revoked` (잠금 LATCHED 로). 합성 수열로 규칙 하나하나를 고정한다; T33 기록 재생은
`test_t34_attach_revoked_replay.py`.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from benchmark.trajopt.grasp_latch import GraspPhase, LatchConfig
from tests.trajopt.test_grasp_evidence import (
    _fb,
    _finger_origin,
    _grasp_frames,
    _latched,
    _lock,
    _manipulated,
    _policy,
    _signal,
)


def _attach(latch, *, opening=0.72, command=0.57):
    """T17-shaped attach: close, settle, attach at the request that sees the same opening again."""
    ev = []
    for seq, applied, meas in ((1, [1.0] * 8, 1.0), (2, [0.0] * 8, opening),
                               (3, [command] * 8, opening)):
        ev.append(latch.observe_grasp(_signal(seq, applied, meas), reach_m=0.055,
                                      manipulated_id=0, manipulated_state="visible"))
    assert ev[-1].attach and latch.phase is GraspPhase.HELD
    return ev


def _hold(latch, seq, applied, meas, obs=None, executed=None):
    return latch.observe_grasp(_signal(seq, applied, meas, executed=executed), reach_m=0.055,
                               manipulated_id=0, manipulated_state="occluded",
                               held_observation=obs)


def test_a_true_hold_keeps_the_opening_above_the_command_and_is_never_revoked():
    """T33 true attaches: opening 0.705–0.725, command 0.56–0.59 (gap 0.13–0.16) through the window."""
    latch = _latched()
    _attach(latch)
    for k, meas in enumerate((0.72, 0.719, 0.718, 0.715, 0.71, 0.71), start=4):
        e = _hold(latch, k, [0.57] * 8, meas)
        assert not e.detach and not e.revoked and latch.phase is GraspPhase.HELD
    assert e.evidence["hold_window"]["active"] is False            # the window has closed
    assert e.evidence["hold_window"]["exec_chunks"] == 6


def test_fingers_closing_on_nothing_revoke_the_attach_and_keep_the_target_latched():
    """E3b ep1808 r1: attach at 0.684, next request 0.555 with the command 0.552 — nothing was
    between the fingers. Revoked (both opening checks), latch back to LATCHED, not PLACED."""
    latch = _latched()
    _attach(latch, opening=0.684, command=0.576)
    e = _hold(latch, 4, [0.552] * 8, 0.555)
    assert e.detach and e.revoked and not e.attach
    assert latch.phase is GraspPhase.LATCHED and latch.manipulated == "apple"
    assert e.evidence["revoke"]["reasons"] == ["opening_dropped", "not_blocked"]
    assert e.evidence["hold_checks"]["opening_dropped"] == pytest.approx(0.129, abs=1e-3)
    assert e.evidence["hold_checks"]["gap_to_command"] == pytest.approx(0.003, abs=1e-3)
    assert e.note.startswith("attach_revoked")
    # a new attempt is judged from scratch: still closed on nothing → no attach
    e = latch.observe_grasp(_signal(5, [0.44] * 8, 0.441), reach_m=0.055, manipulated_id=0,
                            manipulated_state="visible")
    assert latch.phase is GraspPhase.CLOSING and not e.attach
    # the gripper reopens, then closes on the apple again → a real attach
    latch.observe_grasp(_signal(6, [1.0] * 8, 1.0), reach_m=0.055, manipulated_id=0,
                        manipulated_state="visible")
    assert latch.phase is GraspPhase.LATCHED
    _attach(latch)


def test_each_opening_check_alone_is_enough():
    latch = _latched()
    _attach(latch)
    e = _hold(latch, 4, [0.70] * 8, 0.70)          # command loosened to ~ the opening: not judged
    assert not e.revoked and "gap_to_command" not in e.evidence["hold_checks"]
    e = _hold(latch, 5, [0.30] * 8, 0.62)          # still blocked (gap 0.32) but dropped 0.10
    assert e.revoked and e.evidence["revoke"]["reasons"] == ["opening_dropped"]

    latch = _latched(revoke_drop=0.5)
    _attach(latch)
    e = _hold(latch, 4, [0.64] * 8, 0.66)          # gap 0.02 < blocked_gap, drop 0.06 < 0.5
    assert e.revoked and e.evidence["revoke"]["reasons"] == ["not_blocked"]


def test_the_window_counts_executed_chunks_only_and_then_closes():
    """HOLD chunks (nothing executed) do not use up the window — T33 E3b ep1800 r2 held for 56."""
    latch = _latched()
    _attach(latch)
    for k in range(4, 30):                          # 26 HOLD chunks, opening steady
        e = _hold(latch, k, [0.72] * 8, 0.718, executed=[False] * 8)
        assert not e.revoked
    assert e.evidence["hold_window"]["exec_chunks"] == 0 and e.evidence["hold_window"]["active"]
    for k in range(30, 34):                         # 4 executed chunks
        e = _hold(latch, k, [0.57] * 8, 0.718)
    assert e.evidence["hold_window"]["exec_chunks"] == 4 and e.evidence["hold_window"]["active"]
    e = _hold(latch, 34, [0.55] * 8, 0.55)          # a drop on the 5th executed chunk: not revoked
    assert not e.revoked and not e.detach and e.evidence["hold_window"]["active"] is False


def test_a_release_inside_the_window_is_a_release_not_a_revoke():
    latch = _latched()
    _attach(latch)
    e = _hold(latch, 4, [1.0] * 8, 0.95)
    assert not e.revoked and not e.detach
    e = _hold(latch, 5, [1.0] * 8, 1.0)
    assert e.detach and not e.revoked and latch.phase is GraspPhase.PLACED


def _obs(distance, hand_rise=0.0, object_rise=0.0, radius=0.0386):
    return {"centre_distance_m": distance, "held_radius_m": radius, "hand_rise_m": hand_rise,
            "object_rise_m": object_rise}


def test_an_observation_away_from_the_held_sphere_revokes():
    """E3b ep1808 r1 t=152: the apple seen on the table 215 mm from the held sphere."""
    latch = _latched()
    _attach(latch)
    e = _hold(latch, 4, [0.57] * 8, 0.72, obs=_obs(0.070))    # within 38.6 + 40 mm: kept
    assert not e.revoked
    e = _hold(latch, 5, [0.57] * 8, 0.72, obs=_obs(0.215))
    assert e.revoked and e.evidence["revoke"]["reasons"] == ["observed_elsewhere"]
    assert e.evidence["hold_checks"]["observed_limit_mm"] == pytest.approx(78.6)


def test_a_lift_the_object_does_not_follow_revokes():
    latch = _latched()
    _attach(latch)
    e = _hold(latch, 4, [0.57] * 8, 0.72, obs=_obs(0.02, hand_rise=0.020, object_rise=0.0))
    assert not e.revoked                                      # 20 mm is not a lift yet
    e = _hold(latch, 5, [0.57] * 8, 0.72, obs=_obs(0.02, hand_rise=0.080, object_rise=0.075))
    assert not e.revoked                                      # followed
    e = _hold(latch, 6, [0.57] * 8, 0.72, obs=_obs(0.05, hand_rise=0.080, object_rise=0.010))
    assert e.revoked and e.evidence["revoke"]["reasons"] == ["not_following"]


# --- SafePolicy wiring ------------------------------------------------------------------------


def test_safe_policy_revokes_detaches_and_records_the_anchor():
    sp = _policy()
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    apple = _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0])
    sp.ag3s._manipulated = _manipulated(apple)                                 # noqa: SLF001
    _lock(sp)
    fbs = [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.57] * 8, 0.715),
           _fb(4, [0.55] * 8, 0.56)]
    recs = list(_grasp_frames(sp, q, fbs))
    assert [r["attach"] for r in recs] == [False, False, True, False]
    last = recs[-1]
    assert last["detach"] and last["state"] == "latched" and last["transition"]
    assert last["attach_revoked"]["reasons"] == ["opening_dropped", "not_blocked"]
    assert "anchor" in last["attach_revoked"]
    assert sp.ag3s.attached is None and last["state_version"] == 2
    assert sp._latch.phase is GraspPhase.LATCHED                              # noqa: SLF001
    assert all(r["attach_revoked"] is None for r in recs[:-1])


def test_safe_policy_hands_the_pipeline_observation_to_the_latch(monkeypatch):
    sp = _policy()
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    sp.ag3s._manipulated = _manipulated(                                      # noqa: SLF001
        _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0]))
    _lock(sp)
    seen = []

    def far():
        seen.append(True)
        return _obs(0.21, hand_rise=0.22, object_rise=0.0)

    monkeypatch.setattr(sp.ag3s, "held_observation", far)
    fbs = [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.57] * 8, 0.715),
           _fb(4, [0.57] * 8, 0.715)]
    recs = list(_grasp_frames(sp, q, fbs))
    assert len(seen) == 1                               # asked only while held
    assert recs[-1]["attach_revoked"]["reasons"] == ["observed_elsewhere", "not_following"]
    assert recs[-1]["attach_revoked"]["held_observation"]["centre_distance_m"] == 0.21


# --- AG3S.held_observation ------------------------------------------------------------------


def test_ag3s_held_observation_compares_a_fresh_observation_with_the_fit_sphere():
    from benchmark.ag3s.stages.target_grounding import ManipulatedIdentity

    sp = _policy()
    ag3s = sp.ag3s
    q = np.zeros(len(sp.constraint_robot_model.joint_names))
    apple = _finger_origin(sp, q) + np.array([0.05, 0.0, 0.0])
    ag3s._manipulated = _manipulated(apple)                                    # noqa: SLF001
    ag3s.frame_index = 10
    held = ag3s.attach(ag3s._manipulated.geometry, robot_state=q,              # noqa: SLF001
                       parent_link="ee_finger_l1", geometry_state=q)
    assert ag3s.held_observation() is None             # the last observation predates the attach

    def seen(centroid, frame, last_seen=None):
        geom = ag3s._manipulated.geometry                                      # noqa: SLF001
        return ManipulatedIdentity(id=0, state="visible", geometry=geom,
                                   centroid=np.asarray(centroid, np.float64), age_frames=0,
                                   last_seen_frame=frame if last_seen is None else last_seen,
                                   last_seen_time=1.0, switched_from=None, frame=frame)

    ref = ag3s._held_check_ref                                                  # noqa: SLF001
    ag3s._state_by_frame[10] = q                                               # noqa: SLF001
    ag3s._manipulated = seen(apple + [0.0, 0.0, 0.0], 10)                      # noqa: SLF001
    obs = ag3s.held_observation()
    assert obs["frame"] == 10 and obs["attach_frame"] == 10
    assert obs["centre_distance_m"] == pytest.approx(
        float(np.linalg.norm(apple - ref["primary_fit_at_attach"])), abs=1e-9)
    assert obs["hand_rise_m"] == pytest.approx(0.0, abs=1e-12)
    assert obs["held_radius_m"] == pytest.approx(ref["primary_fit_radius"])
    ag3s._manipulated = seen(apple, 11, last_seen=10)                          # noqa: SLF001
    assert ag3s.held_observation() is None             # not observed in its latest frame
    ag3s.detach()
    ag3s._manipulated = seen(apple, 10)                                        # noqa: SLF001
    assert ag3s.held_observation() is None             # nothing held
    assert held is not None
