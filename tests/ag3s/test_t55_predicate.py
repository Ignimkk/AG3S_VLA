"""T55 physical predicates (`t55_predicate`) — synthetic, pure numpy.

Pinned (T55.task.md 사전 고정 1 / 4 / 구현 test):
  - a different object in the active hand is P_grasp True and P_task False (wrong-object grasp), and the
    other hand holding the target is neither;
  - persistence K: a run shorter than K never counts, onset at the K-th frame (causal), a gap shorter than K
    inside a hold does not release, the same K applies on release; ``runs`` mode keeps onsets;
  - the 4 states with their precedence, P_task implies P_grasp;
  - `combine` returns the latch trace untouched (feature never alters attach / detach / phase);
  - errors / latency events (censored when undetected) / paired latency.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t55_predicate as pr


def _hb(n, **held):
    """held_by with every fruit None, then `fruit=(hand, frames)`."""
    out = {f: [None] * n for f in ("apple", "banana", "orange", "pear")}
    for f, (hand, frames) in held.items():
        for i in frames:
            out[f][i] = hand
    return out


def test_wrong_object_is_grasp_not_task():
    hb = _hb(6, banana=("right", [1, 2, 3]))
    r = pr.raw_predicates(hb, "apple", "right")
    assert r["p_grasp"].tolist() == [False, True, True, True, False, False]
    assert not r["p_task"].any()
    assert r["held_object"][2] == "banana"
    st = pr.four_state(r["p_grasp"], r["p_task"], r["placed"])
    assert st[2] == "wrong_object" and st[0] == "empty"


def test_other_hand_does_not_count_and_instruction_swap():
    hb = _hb(4, apple=("left", [0, 1, 2, 3]))
    r = pr.raw_predicates(hb, "apple", "right")
    assert not r["p_grasp"].any() and not r["p_task"].any()
    hb = _hb(4, apple=("right", [1, 2]))
    orig = pr.raw_predicates(hb, "apple", "right")
    swapped = pr.raw_predicates(hb, "pear", "right")  # instruction changed: same physics
    assert np.array_equal(orig["p_grasp"], swapped["p_grasp"])
    assert orig["p_task"].any() and not swapped["p_task"].any()


def test_placed_needs_in_crate_and_no_hand():
    hb = _hb(4, apple=("right", [0, 1]))
    r = pr.raw_predicates(hb, "apple", "right", in_crate=[True, True, True, False])
    assert r["placed"].tolist() == [False, False, True, False]
    assert pr.four_state(r["p_grasp"], r["p_task"], r["placed"]).tolist() == \
        ["target_held", "target_held", "placed", "empty"]


def test_raw_predicate_errors():
    with pytest.raises(ValueError):
        pr.raw_predicates(_hb(3), "kiwi", "right")
    with pytest.raises(ValueError):
        pr.raw_predicates(_hb(3), "apple", "up")
    with pytest.raises(ValueError):
        pr.raw_predicates(_hb(3), "apple", "right", in_crate=[True])


def test_persistence_causal():
    x = [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0]
    out = pr.persist(x, 2)
    # single frame at 1 never counts; onset at frame 4 (second frame of the run 3-5); the 1-frame gap at 6
    # does not release; release after 2 false frames (9, 10) -> False from frame 10
    assert out.astype(int).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
    assert np.array_equal(pr.persist(x, 1), np.asarray(x, bool))
    with pytest.raises(ValueError):
        pr.persist(x, 0)
    with pytest.raises(ValueError):
        pr.persist(x, 2, mode="nope")


def test_persistence_runs_mode_keeps_onsets():
    x = [0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0]
    out = pr.persist(x, 2, mode="runs")
    assert out.astype(int).tolist() == [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0]


def test_predicates_task_implies_grasp_and_K():
    hb = _hb(8, apple=("right", [2, 3, 4, 5]), banana=("right", [0]))
    p = pr.predicates(hb, "apple", "right", K=2)
    assert not p["p_grasp"][0]  # the momentary banana contact
    assert p["p_task"].astype(int).tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    assert np.all(~p["p_task"] | p["p_grasp"])
    assert p["K"] == 2 and set(p["state4"]) <= set(pr.STATES4)


def test_combine_never_alters_latch():
    phase = ("latched", "closing", "held", "held", "held", "placed")
    attach = (False, False, True, False, False, False)
    detach = (False, False, False, False, False, True)
    tr = pr.LatchTrace.from_arrays(phase, attach, detach)
    feat = [False, False, False, True, True, False]
    out = pr.combine(tr, feat, p=1)
    assert out["phase"] is tr.phase and out["attach"] is tr.attach and out["detach"] is tr.detach
    assert out["task_ready"].tolist() == [False, False, False, True, True, False]
    alone = pr.combine(tr, None)
    assert alone["task_ready"].tolist() == [False, False, True, True, True, False]
    assert pr.combine(tr, feat, p=2)["task_ready"].tolist() == [False, False, False, False, True, False]
    with pytest.raises(ValueError):
        pr.combine(tr, [True], p=1)


def test_errors_and_latency():
    truth = np.array([0, 0, 1, 1, 1, 0, 1, 1], bool)
    ready = np.array([0, 1, 0, 1, 1, 0, 0, 0], bool)
    e = pr.errors(ready, truth)
    assert (e["false_pass"], e["false_reject"], e["n_pos"], e["n_neg"]) == (1, 3, 5, 3)
    ev = pr.latencies(ready, truth)
    assert [(x["onset"], x["latency"]) for x in ev] == [(2, 1), (6, None)]
    s = pr.latency_summary(ev)
    assert s["n_undetected"] == 1 and s["latency_mean"] == 1.0
    w = np.array([0, 0, 0, 1, 1, 1, 1, 1], bool)
    ev_w = pr.latencies(ready, truth, w)
    assert [x["onset"] for x in ev_w] == [3, 6] and ev_w[0]["latency"] == 0
    ev2 = pr.latencies(truth, truth)
    assert pr.paired_latency(ev, ev2) == [-1, None]
    with pytest.raises(ValueError):
        pr.paired_latency(ev, ev_w)


def test_natural_events():
    pred = {"p_grasp": np.array([0, 1, 1, 0, 1, 1, 0, 0], bool),
            "p_task": np.array([0, 1, 1, 0, 0, 0, 0, 0], bool),
            "placed": np.array([0, 0, 0, 0, 0, 0, 0, 1], bool)}
    closed = np.array([0, 1, 1, 1, 1, 1, 0, 0], bool)
    ne = pr.natural_events(pred, closed, ["latched", "closing", "held", "closing", "held", "held", "placed", "placed"])
    assert ne["wrong_object_frames"] == 2 and ne["closed_empty_frames"] == 1
    # the false run of P_task after frame 2 ends in ``placed`` -> a set-down, not a slip
    assert ne["grasp_onsets"] == 2 and ne["closing_attempts"] == 2 and ne["slips"] == 0
    pred["placed"][:] = False
    assert pr.natural_events(pred, closed)["slips"] == 1
