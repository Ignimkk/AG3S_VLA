"""T56 HELD event: server chunk records and the offline grasp_latch run (synthetic)."""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t56_held as hd


def _summ(state, attach=False, detach=False, revoked=False, slip=False, hand="right"):
    return {"grasp": {"state": state, "attach": attach, "detach": detach, "hand": hand,
                      "attach_revoked": {"reasons": ["x"]} if revoked else None,
                      "held_slip": slip, "rejected": None}}


def test_server_event_is_first_attach_that_leaves_held_and_exit_reasons():
    s = [_summ("latched"), _summ("closing"), _summ("held", attach=True), _summ("held"),
         _summ("placed", detach=True, slip=True), _summ("placed")]
    ev = hd.from_summaries(s)
    assert ev.source == "server_chunk" and ev.pos == 2 and ev.hand == "right"
    assert ev.exit_pos == 4 and ev.exit_reason == "slip"
    assert ev.state_at(3) == "held" and ev.state_at(5) == "placed"
    rel = hd.from_summaries([_summ("closing"), _summ("held", attach=True), _summ("placed", detach=True)])
    assert rel.exit_reason == "release"
    rev = hd.from_summaries([_summ("held", attach=True), _summ("latched", detach=True, revoked=True)])
    assert rev.pos == 0 and rev.exit_reason == "revoked"
    # an attach the AG3S rejected (state back to closing) is not the event
    rej = hd.from_summaries([_summ("closing", attach=True), _summ("held", attach=True)])
    assert rej.pos == 1


def test_no_held_event_without_attach():
    ev = hd.from_summaries([_summ("latched"), _summ("closing"), _summ("closing")])
    assert not ev.has_event and ev.pos is None and ev.details["n_held_frames"] == 0


def _record(measured_left, cmd_left, n_f=None):
    """state (F,16) with the left gripper opening at column 7; actions (F,8,16) with the left gripper
    command (applied by the next chunk) at column 7. Right hand stays open."""
    n_f = n_f or len(measured_left)
    st = np.zeros((n_f, 16))
    st[:, 7] = measured_left
    st[:, 15] = 1.0
    act = np.zeros((n_f, 8, 16))
    act[:, :, 7] = np.asarray(cmd_left)[:, None]
    act[:, :, 15] = 1.0
    parent = np.zeros((n_f, 2, 3))
    return st, act, parent


def test_offline_latch_holds_on_a_blocked_settled_closure():
    # open, command close from chunk 1; fingers stop at 0.7 (blocked by the object), settle
    st, act, parent = _record([1.0, 1.0, 0.7, 0.7, 0.7, 0.7], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    manip = [np.array([0.0, 0.0, 0.05])] * 6
    ev = hd.offline_latch(t_step=np.arange(6) * 8, state=st, actions=act, parent_world=parent,
                          manip_world=manip, frames=list(range(10, 16)))
    assert ev.source == "offline_latch" and ev.hand == "left"
    assert ev.pos == 13  # record frame index (position 3)
    assert ev.states[:4] == ["latched", "latched", "closing", "held"]


def test_offline_latch_empty_closure_is_not_held():
    st, act, parent = _record([1.0, 1.0, 0.3, 0.02, 0.02, 0.02], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ev = hd.offline_latch(t_step=np.arange(6) * 8, state=st, actions=act, parent_world=parent,
                          manip_world=[np.zeros(3)] * 6)
    assert not ev.has_event


def test_offline_latch_reach_needs_a_manipulated_geometry():
    st, act, parent = _record([1.0, 1.0, 0.7, 0.7, 0.7, 0.7], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    far = hd.offline_latch(t_step=np.arange(6) * 8, state=st, actions=act, parent_world=parent,
                           manip_world=[np.array([1.0, 0.0, 0.0])] * 6)
    assert not far.has_event  # 1 m > reach 0.12 m
    none = hd.offline_latch(t_step=np.arange(6) * 8, state=st, actions=act, parent_world=parent,
                            manip_world=[None] * 6)
    assert not none.has_event
    nc = hd.offline_latch(t_step=np.arange(6) * 8, state=st, actions=act, parent_world=parent,
                          manip_world=[None] * 6, cfg=hd.OfflineLatchConfig(reach_source="not_checked"))
    assert nc.has_event


def test_offline_latch_exit_after_the_object_leaves_the_hand():
    # held at position 3, then the fingers close to the command (object gone) → revoke / slip exit
    st, act, parent = _record([1.0, 1.0, 0.7, 0.7, 0.01, 0.01, 0.01, 0.01],
                              [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ev = hd.offline_latch(t_step=np.arange(8) * 8, state=st, actions=act, parent_world=parent,
                          manip_world=[np.zeros(3)] * 8)
    assert ev.pos == 3 and ev.exit_pos is not None and ev.exit_pos > 3
    assert ev.exit_reason in ("revoked", "slip")


def test_manipulated_track_is_the_previous_frames_geometry():
    c = np.array([1.0, 2.0, 3.0])
    tr = hd.manipulated_track([None, c, None, None])
    assert tr[0] == (None, None) and tr[1] == (None, None)
    assert tr[2][1] == "visible" and np.allclose(tr[2][0], c)
    assert tr[3][1] == "occluded"


def test_event_roundtrip(tmp_path):
    ev = hd.from_summaries([_summ("held", attach=True)])
    hd.write_event(tmp_path / "e.json", ev)
    back = hd.read_event(tmp_path / "e.json")
    assert back.pos == 0 and back.state_at(0) == "held" and back.source == "server_chunk"


def resim_npz(holder, *, tag="E0", target="apple", arm="right", pos=3, n_f=6, window_start=-1):
    """A T55 re-simulation payload: the latch reaches HELD (attach) at `pos`; `holder` = the fruit the
    used hand physically holds then (None = empty grasp)."""
    fruits = ["apple", "banana", "pear", "orange"]
    st = np.full((n_f, 4), "table", dtype="U12")
    hd = np.full((n_f, 4), "", dtype="U8")
    if holder is not None:
        j = fruits.index(holder)
        st[pos:, j] = "held"
        hd[pos:, j] = arm
    phase = np.asarray(["latched"] * 2 + ["closing"] * (pos - 2) + ["held"] * (n_f - pos), dtype="U10")
    attach = np.zeros(n_f, bool)
    attach[pos] = True
    return {"fruits": np.asarray(fruits), "target": np.asarray(target), "used_arm": np.asarray(arm),
            "phys_state": st, "phys_hand": hd, "window_start": np.asarray(window_start, np.int64),
            "key": np.asarray("ep1804"), "tag": np.asarray(tag),
            "latch_record__phase": phase, "latch_record__attach": attach}


def test_resim_held_is_a_trigger_not_a_certification():
    ok = hd.resim_held_check(resim_npz("apple"))
    assert ok["modes"]["record"] == {"has_event": True, "pos": 3, "held_fruits": ["apple"], "verdict": "correct",
                                     "label_correct": True, "after_intervention": None}
    empty = hd.resim_held_check(resim_npz(None, tag="empty_L0O0_park", window_start=2))
    assert empty["modes"]["record"]["verdict"] == "empty" and not empty["modes"]["record"]["label_correct"]
    assert empty["modes"]["record"]["after_intervention"] is True
    wrong = hd.resim_held_check(resim_npz("banana"))
    assert wrong["modes"]["record"]["verdict"] == "wrong_object"
    # the other hand holding the target does not make the used hand's HELD correct
    other = resim_npz("apple")
    other["phys_hand"][3:, 0] = "left"
    assert hd.resim_held_check(other)["modes"]["record"]["verdict"] == "empty"
    summ = hd.resim_summary([ok, empty, wrong])
    assert summ["record"]["all"]["n_held_events"] == 3 and summ["record"]["all"]["n_label_wrong"] == 2
    assert summ["record"]["all"]["held_label_error_rate"] == pytest.approx(2 / 3)
    assert summ["record"]["empty"]["wrong"] == ["ep1804__empty_L0O0_park"]


def test_first_held_uses_the_attach_when_present():
    ph = ["closing", "held", "held", "latched", "held"]
    assert hd.first_held(ph) == 1
    assert hd.first_held(ph, [False, False, False, False, True]) == 4
    assert hd.first_held(["closing", "closing"]) is None
