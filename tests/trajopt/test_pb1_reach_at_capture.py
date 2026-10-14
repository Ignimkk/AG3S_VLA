"""V11AB-pb1 — grasp reach gate at the manipulated geometry's capture state (`AG3S.reach_at_capture`).

SL5b (Panda/LIBERO): the policy lifts right after closing, the object leaves the view (`occluded`), its centroid
stays where it was last seen, and the hand-now reach grows past 120 mm, so a real grasp is never attached. With
the flag the latch evidence and `AG3S.attach` pair that centroid with the hand pose of the same capture.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.trajopt.test_grasp_evidence import _fb, _finger_origin, _grasp_frames, _lock, _manipulated, _policy


def _lifted(sp, q, min_shift=0.15):
    """A configuration whose finger origin is > min_shift from the one at ``q`` (the lift)."""
    o = _finger_origin(sp, q)
    for j in range(len(q)):
        for d in (0.6, -0.6, 1.0, -1.0):
            q2 = q.copy(); q2[j] += d
            if np.linalg.norm(_finger_origin(sp, q2) - o) > min_shift:
                return q2
    raise AssertionError("no lifted configuration found")


def _scene(flag, keep_capture=True):
    sp = _policy()
    sp.ag3s.reach_at_capture = flag
    q_cap = np.zeros(len(sp.constraint_robot_model.joint_names))
    q_now = _lifted(sp, q_cap)
    bowl = _finger_origin(sp, q_cap) + np.array([0.05, 0.0, 0.0])
    sp.ag3s._manipulated = _manipulated(bowl, state="occluded")                 # noqa: SLF001  last_seen_frame=4
    if keep_capture:
        sp.ag3s._state_by_frame[4] = q_cap                                      # noqa: SLF001
    _lock(sp)
    return sp, q_cap, q_now


GRASP = [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.715), _fb(3, [0.0] * 8, 0.715)]


def test_v11ab_refuses_the_lifted_occluded_object():
    sp, _, q_now = _scene(False)
    recs = list(_grasp_frames(sp, q_now, GRASP))
    assert sp.ag3s.attached is None
    assert recs[-1]["evidence"]["checks"]["reach"] is False
    assert recs[-1]["evidence"]["reach_mm"] > 120.0
    assert "reach_state" not in recs[-1]["alignment"]                           # baseline record unchanged


def test_pb1_measures_reach_at_the_capture_and_attaches():
    sp, q_cap, q_now = _scene(True)
    recs = list(_grasp_frames(sp, q_now, GRASP))
    assert sp.ag3s.attached is not None
    rec = recs[-1]
    assert rec["evidence"]["reach_mm"] == pytest.approx(50.0, abs=0.5)
    assert rec["alignment"]["reach_state"] == "capture frame 4"
    assert rec["alignment"]["reach_now_mm"] > 120.0
    # the snapshot is taken at the same capture state (T32 H2), so the held object sits where it was seen
    assert rec["attach"] and rec["state"] == "held"


def test_pb1_without_a_kept_capture_state_falls_back_to_now_and_refuses():
    sp, _, q_now = _scene(True, keep_capture=False)
    recs = list(_grasp_frames(sp, q_now, GRASP))
    assert sp.ag3s.attached is None
    assert recs[-1]["alignment"]["reach_state"] == "now (no capture state kept)"


def test_pb1_does_not_attach_an_empty_closure():
    sp, _, q_now = _scene(True)
    empty = [_fb(1, [1.0] * 8, 1.0), _fb(2, [0.0] * 8, 0.0), _fb(3, [0.0] * 8, 0.0)]
    recs = list(_grasp_frames(sp, q_now, empty))
    assert sp.ag3s.attached is None
    failed = [k for k, v in recs[-1]["evidence"]["checks"].items() if not v]
    assert "not_empty" in failed and "reach" not in failed


def test_pb1_attach_gate_uses_the_capture_state():
    from AG3S.ag3s.constraints.attached import AttachRejected

    sp, q_cap, q_now = _scene(True)
    geom = sp.ag3s._manipulated.geometry                                       # noqa: SLF001
    sp.ag3s.reach_at_capture = False
    with pytest.raises(AttachRejected):
        sp.ag3s.attach(geom, robot_state=q_now, parent_link="ee_finger_l1", reach=0.12)
    sp.ag3s.reach_at_capture = True
    assert sp.ag3s.attach(geom, robot_state=q_now, parent_link="ee_finger_l1", reach=0.12) is not None
