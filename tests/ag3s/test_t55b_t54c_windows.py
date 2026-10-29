"""T55b (#48 HELD false-negative trace) and the T54c P · H add-on — pure helpers only (no storage access)."""

from __future__ import annotations

import numpy as np

from AG3S.ag3s.experiments.studies import t54c_windows as W
from AG3S.ag3s.experiments.studies import t55b_held_fn as B


def test_manipulated_is_labels_nearest_fruit():
    fruits = ["apple", "banana", "pear"]
    pos = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]])
    assert B.manipulated_is(np.array([0.29, 0.0, 0.0]), pos, fruits, "banana", ["apple"]) == "noun"
    assert B.manipulated_is(np.array([0.01, 0.0, 0.0]), pos, fruits, "banana", ["apple"]) == "held"
    assert B.manipulated_is(np.array([0.0, 0.29, 0.0]), pos, fruits, "banana", ["apple"]) == "other"
    assert B.manipulated_is(np.array([np.nan, 0.0, 0.0]), pos, fruits, "banana", ["apple"]) == "none"


def test_summarize_counts_causes_and_gate_rows():
    rows = [
        {"held": False, "p_task": False, "latch_phase": "closing", "failed_checks": ["reach"],
         "manipulated_is": "noun", "reach_mm": 500.0, "reach_limit_mm": 120.0, "held_fruit_mm": 55.0},
        {"held": True, "p_task": True, "latch_phase": "held", "failed_checks": [],
         "manipulated_is": "held", "reach_mm": 60.0, "reach_limit_mm": 120.0, "held_fruit_mm": 50.0},
    ]
    s = B.summarize(rows)
    assert s["n_p_grasp"] == 2 and s["n_held_false"] == 1 and s["n_held"] == 1
    assert s["held_false_by_cause"] == {"closing|reach|manipulated=noun": 1}
    assert s["held_false_with_p_task"] == 0
    assert s["reach_limit_mm"] == [120.0]


def test_redirect_sets_points_t60_tools_at_t54c(tmp_path):
    saved_sets, saved_pairs = dict(W.TS.SETS), dict(W.CL.PAIRS)
    try:
        W.redirect_sets(tmp_path)
        assert dict(W.TS.SETS) == {"T54c": tmp_path}
        assert W.CL.SETS is W.TS.SETS and W.CL.PAIRS == {"T54c": []}
    finally:
        W.TS.SETS.clear()
        W.TS.SETS.update(saved_sets)
        W.CL.SETS, W.CL.PAIRS = W.TS.SETS, saved_pairs


def test_load_masks_parses_episode_key(tmp_path):
    (tmp_path / "masks").mkdir()
    np.savez(tmp_path / "masks" / "T54c__C1a_ep1806_s18061.npz", t_P=40, t_H=-1,
             keep_frames_P=np.array([1, 1, 0]), keep_frames_H=np.array([1, 1, 1]),
             chunk_start_steps=np.array([0, 8, 16]))
    m = W.load_masks(tmp_path)
    assert list(m) == ["ep1806"]
    assert m["ep1806"]["P"].tolist() == [True, True, False] and m["ep1806"]["t_H"] == -1
    assert m["ep1806"]["chunk_start_steps"] == [0, 8, 16]
