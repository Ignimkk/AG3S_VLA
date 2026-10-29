"""T56 selection / regret / τ-b / bootstrap (synthetic, pure numpy)."""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t56_select as sm

NAMES = ["a", "b", "c"]


def _gt(identity, purity=None, coverage=None, no_target=None):
    identity = np.asarray(identity, float)
    z = np.zeros_like(identity)
    return {"identity": identity, "purity": z + 1 if purity is None else np.asarray(purity, float),
            "coverage": z + 1 if coverage is None else np.asarray(coverage, float),
            "no_target": z if no_target is None else np.asarray(no_target, float)}


def test_tau_b_all_tied_self_is_undefined_not_zero():
    assert sm.kendall_tau_b([0.5, 0.5, 0.5], [0.1, 0.9, 0.4]) is None
    assert sm.kendall_tau_b([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]) is None
    assert sm.kendall_tau_b([0.1], [0.2]) is None
    taus = sm.tau_per_episode(np.array([[1.0, 1.0, 1.0], [0.1, 0.2, 0.3]]),
                              np.array([[0.2, 0.5, 0.9], [0.2, 0.5, 0.9]]))
    assert taus[0] is None and taus[1] == pytest.approx(1.0)
    s = sm.tau_summary(taus, ["e0", "e1"])
    assert s["n_undefined"] == 1 and s["n_defined"] == 1 and s["mean_defined"] == pytest.approx(1.0)
    assert s["episodes"]["e0"] is None


def test_tau_b_matches_scipy_with_ties():
    from scipy.stats import kendalltau

    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.integers(0, 4, 9).astype(float)
        y = rng.integers(0, 3, 9).astype(float)
        mine = sm.kendall_tau_b(x, y)
        ref = kendalltau(x, y, variant="b").statistic
        if mine is None:
            assert np.isnan(ref)
        else:
            assert mine == pytest.approx(ref)


def test_tie_break_order_coverage_then_name():
    # a and c within the resolution of the max (b is far): coverage decides
    S = np.array([[0.900, 0.50, 0.905]])
    tb = np.array([[0.2, 0.9, 0.8]])
    pt = sm.evaluate(S, tb, _gt([[0.1, 0.2, 0.3]]), NAMES, resolution=0.01)
    assert NAMES[int(pt["selected"][0])] == "c" and int(pt["tie_size"][0]) == 2
    # equal coverage → lexicographic first
    tb2 = np.array([[0.8, 0.9, 0.8]])
    pt2 = sm.evaluate(S, tb2, _gt([[0.1, 0.2, 0.3]]), NAMES, resolution=0.01)
    assert NAMES[int(pt2["selected"][0])] == "a"
    # resolution 0 → no tie, the max score wins regardless of coverage
    pt3 = sm.evaluate(S, tb, _gt([[0.1, 0.2, 0.3]]), NAMES, resolution=0.0)
    assert NAMES[int(pt3["selected"][0])] == "c" and int(pt3["tie_size"][0]) == 1
    pt4 = sm.evaluate(np.array([[0.906, 0.5, 0.905]]), tb, _gt([[0.1, 0.2, 0.3]]), NAMES, resolution=0.0)
    assert NAMES[int(pt4["selected"][0])] == "a"


def test_names_must_be_sorted():
    with pytest.raises(ValueError):
        sm.evaluate(np.ones((1, 2)), np.ones((1, 2)), _gt([[0, 0]]), ["b", "a"], resolution=0.0)


def test_tie_set_mean_regret_and_selected_regret():
    S = np.array([[1.0, 1.0, 0.2], [1.0, 1.0, 0.2]])  # a, b tied; c out
    tb = np.array([[0.5, 0.1, 0.9], [0.5, 0.1, 0.9]])  # a wins the tie-break
    gt = _gt([[0.2, 0.8, 1.0], [0.4, 0.6, 1.0]])  # GT means: a .3, b .7, c 1.0
    pt = sm.evaluate(S, tb, gt, NAMES, resolution=0.0)
    assert NAMES[int(pt["selected"][0])] == "a"
    assert pt["all:identity:best"][0] == pytest.approx(1.0)
    assert pt["all:identity:regret"][0] == pytest.approx(0.7)
    assert pt["all:identity:tie_regret"][0] == pytest.approx(1.0 - 0.5)
    assert sm.tie_set(pt, NAMES, 0.0) == ["a", "b"]


def test_episodes_without_event_are_excluded_from_selection_not_chosen_by_gt():
    # episode 1 has no event (NaN row). GT in episode 1 strongly favours b; selection must ignore it.
    S = np.array([[0.9, 0.1, 0.1], [np.nan, np.nan, np.nan]])
    tb = np.zeros((2, 3))
    gt = _gt([[0.5, 0.4, 0.0], [0.0, 1.0, 0.0]])
    pt = sm.evaluate(S, tb, gt, NAMES, resolution=0.0)
    assert NAMES[int(pt["selected"][0])] == "a"
    assert pt["event_weight"][0] == 1
    # GT 'all' includes the event-less episode, 'event' does not
    assert pt["all:identity:selected"][0] == pytest.approx(0.25)
    assert pt["event:identity:selected"][0] == pytest.approx(0.5)
    assert pt["all:identity:best"][0] == pytest.approx(0.7)  # b: (0.4 + 1.0) / 2


def test_no_event_at_all_gives_no_selection():
    S = np.full((2, 3), np.nan)
    pt = sm.evaluate(S, np.zeros((2, 3)), _gt([[1, 0, 0], [1, 0, 0]]), NAMES, resolution=0.0)
    assert int(pt["selected"][0]) == -1
    assert np.isnan(pt["all:identity:regret"][0])


def test_bootstrap_draws_are_episode_multiplicities_and_paired():
    W = sm.episode_draws(5, 300, sm.SEED)
    assert W.shape == (300, 5) and np.all(W.sum(axis=1) == 5)
    assert np.array_equal(W, sm.episode_draws(5, 300, 5607))
    S = np.random.default_rng(1).random((5, 3))
    gt = _gt(np.random.default_rng(2).random((5, 3)))
    one = sm.evaluate(S, S, gt, NAMES, resolution=0.0, W=np.ones((1, 5)))
    pt = sm.evaluate(S, S, gt, NAMES, resolution=0.0)
    assert one["selected"][0] == pt["selected"][0]
    bt = sm.evaluate(S, S, gt, NAMES, resolution=0.0, W=W)
    assert bt["all:identity:regret"].shape == (300,)
    summ = sm.summarize(pt, bt, NAMES)
    assert "all:identity:regret:ci95" in summ and 0.0 <= summ["tie_rate_boot"] <= 1.0
    d = sm.paired(bt, bt, pt, pt, ["all:identity:regret"])
    assert d["all:identity:regret"]["diff"] == 0.0 and d["all:identity:regret"]["ci95"] == [0.0, 0.0]


def test_bootstrap_redoes_the_selection_per_draw():
    # episode 0 favours a, episode 1 favours b → draws with only episode 1 select b
    S = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    gt = _gt([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    W = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    bt = sm.evaluate(S, np.zeros((2, 3)), gt, NAMES, resolution=0.0, W=W)
    assert [NAMES[i] for i in bt["selected"]] == ["a", "b", "a"]
    assert list(bt["tie_size"]) == [1, 1, 2]
    assert np.allclose(bt["all:identity:regret"], [0.0, 0.0, 0.0])


def test_tau_b_reasons_and_float_residue_ties():
    assert sm.kendall_tau_b_status([0.5, 0.5, 0.5], [0.1, 0.9, 0.4]) == (None, "self_all_tied")
    assert sm.kendall_tau_b_status([0.1, 0.2, 0.3], [1.0, 1.0, 1.0]) == (None, "gt_all_tied")
    assert sm.kendall_tau_b_status([1.0, 1.0], [2.0, 2.0]) == (None, "both_all_tied")
    assert sm.kendall_tau_b_status([np.nan, 1.0], [0.0, 1.0]) == (None, "fewer_than_2")
    # equal means that differ only by floating residue are a tie, not an order
    x = [0.1 + 0.2, 0.3, 0.3]
    assert sm.kendall_tau_b_status(x, [0.0, 1.0, 0.5])[1] == "self_all_tied"
    st = sm.tau_per_episode_status(np.array([[np.nan, np.nan, np.nan], [1.0, 1.0, 1.0]]),
                                   np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]))
    assert st == [(None, "no_event"), (None, "self_all_tied")]
    s = sm.tau_summary(st, ["e0", "e1"])
    assert s["mean_defined"] is None and s["mean_status"] == "undefined"
    assert s["n_no_event"] == 1 and s["n_undefined_with_event"] == 1
    assert s["episode_status"] == {"e0": "undefined:no_event", "e1": "undefined:self_all_tied"}


def test_episode_ties_and_tie_rate_leave_no_event_draws_out():
    S = np.array([[0.9, 0.9, 0.1], [0.2, 0.8, 0.79], [np.nan, np.nan, np.nan]])
    t = sm.episode_ties(S, 0.0, ["e0", "e1", "e2"])
    assert t["per_episode"] == {"e0": 2, "e1": 1, "e2": None}
    assert t["tie_rate"] == 0.5 and t["mean_tie_size"] == 1.5 and t["n_event"] == 2
    assert sm.episode_ties(S, 0.02, ["e0", "e1", "e2"])["per_episode"]["e1"] == 2
    # draws: only the no-event episode (no selection), only e0 (tied), only e1 (not tied)
    W = np.array([[0.0, 0.0, 3.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    gt = _gt(np.zeros((3, 3)))
    pt = sm.evaluate(S, np.zeros((3, 3)), gt, NAMES, resolution=0.0)
    bt = sm.evaluate(S, np.zeros((3, 3)), gt, NAMES, resolution=0.0, W=W)
    assert list(bt["selected"]) == [-1, 0, 1]
    summ = sm.summarize(pt, bt, NAMES)
    assert summ["tie_rate_boot"] == 0.5 and summ["boot_no_event_draws"] == 1
