"""T51 protocol core (`t51_protocol`) on synthetic data — no MuJoCo, no storage.

Pinned:
  - P1 filter rules on hand-made maps: a fixed-patch sink, a broad blob, a low-mass map are excluded;
    an L8H2-like compact moving blob passes; one bad camera excludes the cell; NaN never triggers;
    strict inequalities at the thresholds;
  - 2D scores (`record_2d_scores`) agree with `t47_head_metrics.map_scores` on the kept samples;
  - ranking / preselection tie rules and forced cells;
  - bootstrap: same seed -> identical output, different seed -> different draws, exact enumeration
    probabilities sum to 1, a dominant cell gets P(rank 1) = 1;
  - nested CV: no episode leakage (the ranking never sees the test unit; perturbing the test unit's
    scores cannot change that fold's selection), `required_pairs` covers every evaluation;
  - the k simplicity rule; random cell draws are seeded.
"""

from __future__ import annotations

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t47_head_metrics as hm
from AG3S.ag3s.experiments.studies import t47b_sink as sk
from AG3S.ag3s.experiments.studies import t51_protocol as pr

G = 16


def _stats(maps):
    """maps (F, 16, 16) for one cell / camera -> filter summary scalars."""
    st = sk.map_stats(maps[:, None, None, None])  # (F, 1, 1, 1)
    summ = pr.filter_summary(st)
    return {k: v.reshape(-1)[0] for k, v in summ.items() if np.ndim(v) >= 3}, summ


def sink_maps(n=20, mass=0.3):
    m = np.full((n, G, G), mass * 0.3 / 255)
    m[:, 1, 11] = mass * 0.7  # same patch every frame, 70 % of the mass
    return m


def broad_maps(n=20, mass=0.3):
    m = np.full((n, G, G), mass * 0.4 / 256)
    m[:, :8, :8] += mass * 0.6 / 64  # one big low-peak blob, 25 % of the grid
    return m


def compact_moving(n=20, mass=0.3):
    m = np.full((n, G, G), mass * 0.5 / 256)
    for i in range(n):
        r, c = 2 + (i % 10), 3 + (i % 9)
        m[i, r:r + 2, c:c + 2] += mass * 0.5 / 4 * np.array([[1.4, 0.8], [0.9, 0.9]])
    return m


def test_filter_classifies_synthetic_maps():
    for maps, want in ((sink_maps(), "fixed_patch_sink"), (broad_maps(), "broad_map"),
                       (compact_moving(), None), (sink_maps(mass=0.005), "low_mass")):
        _, summ = _stats(maps)
        reasons = pr.filter_reasons(summ)
        fired = [r for r in pr.FILTER_REASONS if bool(reasons[r].reshape(-1)[0])]
        if want is None:
            assert fired == [], fired
        else:
            assert want in fired, (want, fired)


def test_filter_thresholds_are_strict():
    base = {"mass_median": np.array([0.01]), "modal_fraction": np.array([0.5]),
            "top1_share_mean": np.array([0.4]), "largest_share_of_grid_mean": np.array([0.15])}
    r = pr.filter_reasons(base)
    assert not r["low_mass"][0] and not r["fixed_patch_sink"][0] and not r["broad_map"][0]
    r = pr.filter_reasons({**base, "top1_share_mean": np.array([0.03])})
    assert not r["broad_map"][0]
    r = pr.filter_reasons({**base, "mass_median": np.array([0.0099]),
                           "modal_fraction": np.array([0.51]), "top1_share_mean": np.array([0.41])})
    assert r["low_mass"][0] and r["fixed_patch_sink"][0]
    r = pr.filter_reasons({**base, "top1_share_mean": np.array([0.029]),
                           "largest_share_of_grid_mean": np.array([0.151])})
    assert r["broad_map"][0]


def test_filter_nan_never_triggers_and_one_camera_excludes_cell():
    summ = {"mass_median": np.array([[0.2, 0.2, 0.2]]),
            "modal_fraction": np.array([[np.nan, 0.1, 0.9]]),
            "top1_share_mean": np.array([[np.nan, 0.2, 0.6]]),
            "largest_share_of_grid_mean": np.array([[np.nan, 0.05, 0.01]])}
    r = pr.filter_reasons(summ)
    assert r["any"].tolist() == [[False, False, True]]
    assert pr.filter_eligible(r).tolist() == [False]
    r2 = pr.filter_reasons({k: v[:, :2] for k, v in summ.items()})
    assert pr.filter_eligible(r2).tolist() == [True]


def test_record_2d_scores_match_map_scores():
    rng = np.random.default_rng(0)
    F, L, H, C = 4, 2, 3, 3
    block = rng.random((F, L, H, C, G, G))
    cov = np.zeros((F, C, 2, G, G))
    cov[:, :, 0, 4:7, 4:7] = 1.0
    cov[:, :, 1] = 1.0 - cov[:, :, 0]
    keep = np.ones((F, C), bool)
    keep[0, 1] = False
    out = pr.record_2d_scores(block, cov, keep)
    fi, ci = np.nonzero(keep)
    for l in range(L):
        for h in range(H):
            sc = hm.map_scores(block[fi, l, h, ci], cov[fi, ci])
            assert out["lift"][l, h] == pytest.approx(np.nanmean(sc["lift"]))
            assert out["peak_on_target"][l, h] == pytest.approx(sc["peak_on_target"].mean())
    assert int(out["n_kept"]) == keep.sum()
    empty = pr.record_2d_scores(block, cov, np.zeros((F, C), bool))
    assert np.isnan(empty["lift"]).all()


def test_rank_order_ties_and_nan():
    s = np.array([0.5, np.nan, 0.5, 0.9, 0.5])
    t = np.array([0.1, 9.0, 0.2, 0.0, 0.2])
    assert pr.rank_order(s, t).tolist() == [3, 2, 4, 0, 1]
    assert pr.rank_order(s, t, np.array([1, 1, 1, 0, 1], bool)).tolist() == [2, 4, 0, 1]


def test_preselect_forced_and_ineligible():
    s = np.arange(10, dtype=float)
    el = np.ones(10, bool)
    el[2] = False
    res = pr.preselect(s, np.zeros(10), el, n_top=3, forced=[9, 2, 0])
    assert res["top"] == [9, 8, 7]
    assert res["ranked_set"] == [0, 7, 8, 9]
    assert res["scored_set"] == [0, 2, 7, 8, 9]
    assert res["forced_ineligible"] == [2] and res["forced_already_in_top"] == [9]


def test_cells_and_combo_ids():
    om = pr.omega()
    assert len(om) == 3 * 16 * 8 and om[0].id == "last:L2H0"
    c = pr.parse_cell("mean:L8H2")
    assert (c.query, c.layer, c.head) == ("mean", 8, 2)
    cid = pr.combo_id(["last:L8H3", "last:L10H7"])
    assert cid == "sum[last:L10H7+last:L8H3]" == pr.combo_id(["last:L10H7", "last:L8H3"])
    assert [x.id for x in pr.combo_cells(cid)] == ["last:L10H7", "last:L8H3"]
    assert pr.combo_id(["last:L8H2"]) == "last:L8H2"
    assert pr.safe_name(cid) == "sum__last_L10H7__last_L8H3"
    with pytest.raises(ValueError):
        pr.parse_cell("L8H2")


# ----------------------------------------------------------------------------- bootstrap


def _scores(E=3, N=6, seed=1):
    rng = np.random.default_rng(seed)
    return rng.random((E, N)), rng.random((E, N))


def test_bootstrap_reproducible_by_seed():
    S, T = _scores()
    el = np.ones(S.shape[1], bool)
    a = pr.bootstrap_ranking(S, T, el, n_boot=500, seed=7)
    b = pr.bootstrap_ranking(S, T, el, n_boot=500, seed=7)
    for k in ("p_rank1", "p_top3", "ci_lo", "ci_hi"):
        np.testing.assert_array_equal(a[k], b[k])
    S5, T5 = _scores(E=6, N=8, seed=3)
    c = pr.bootstrap_ranking(S5, T5, np.ones(8, bool), n_boot=500, seed=7)
    d = pr.bootstrap_ranking(S5, T5, np.ones(8, bool), n_boot=500, seed=8)
    assert not np.array_equal(c["ci_lo"], d["ci_lo"]) or not np.array_equal(c["p_rank1"], d["p_rank1"])
    assert a["p_rank1"].sum() == pytest.approx(1.0)
    assert a["p_top3"].sum() == pytest.approx(3.0)


def test_bootstrap_exact_and_dominant_cell():
    S = np.array([[0.9, 0.1, 0.2], [0.8, 0.3, 0.1], [0.95, 0.2, 0.3]])
    bs = pr.bootstrap_ranking(S, np.zeros_like(S), np.ones(3, bool), n_boot=200, seed=1)
    assert bs["p_rank1"][0] == 1.0
    assert bs["exact"]["n_ordered_resamples"] == 27
    assert bs["exact"]["p_rank1"][0] == pytest.approx(1.0)
    assert bs["n_distinct_resamples"] == 10
    # two cells that win on different episodes: exact probabilities from the 27 resamples
    S2 = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    ex = pr.bootstrap_ranking(S2, np.zeros_like(S2), np.ones(2, bool), n_boot=200, seed=1)["exact"]
    # cell 0 wins when episode 0 is drawn more often than episodes 1+2 combined: 7 of 27 (3 ties
    # of 1:1 do not exist with 3 draws); cell 1 otherwise
    assert ex["p_rank1"][0] == pytest.approx(7 / 27)
    assert ex["p_rank1"].sum() == pytest.approx(1.0)


def test_bootstrap_ineligible_never_wins_and_nan_episode_skipped():
    S = np.array([[0.1, 0.9], [0.2, np.nan], [0.3, 0.8]])
    bs = pr.bootstrap_ranking(S, np.zeros_like(S), np.array([True, False]), n_boot=100, seed=0)
    assert bs["p_rank1"][1] == 0.0 and bs["p_rank1"][0] == 1.0
    assert bs["mean"][1] == pytest.approx(0.85)


# ------------------------------------------------------------------------------------ CV


def _toy(scores: dict):
    """rank_fn / eval_fn over per-unit single-cell scores {unit: {cell: s}}; combos score as the mean
    of their cells' per-unit scores (any deterministic rule works for the plumbing)."""
    seen = []

    def rank_fn(train):
        seen.append(tuple(train))
        cells = sorted(next(iter(scores.values())))
        m = {c: np.mean([scores[u][c] for u in train]) for c in cells}
        return sorted(cells, key=lambda c: (-m[c], c))

    def eval_fn(cid, unit):
        return float(np.mean([scores[unit][c.id] for c in pr.combo_cells(cid)]))

    return rank_fn, eval_fn, seen


def _scores_dict(units=("a", "b", "c", "d"), seed=0):
    rng = np.random.default_rng(seed)
    cells = [f"last:L{l}H0" for l in range(2, 8)]
    return {u: {c: float(rng.random()) for c in cells} for u in units}


def test_loeo_folds_partition():
    f = pr.loeo_folds(["a", "b", "c"])
    assert f == [(("b", "c"), "a"), (("a", "c"), "b"), (("a", "b"), "c")]
    with pytest.raises(ValueError):
        pr.loeo_folds(["a", "a"])


def test_cv_has_no_episode_leakage():
    sc = _scores_dict()
    rank_fn, eval_fn, seen = _toy(sc)
    res = pr.cv_scores(list(sc), (1, 2, 3), rank_fn, eval_fn)
    for train, test in pr.loeo_folds(list(sc)):
        assert test not in train
    assert all(len(t) == 3 for t in seen)
    # perturbing the held-out unit's scores cannot change the combo chosen for that fold
    for i, test in enumerate(sc):
        pert = {u: dict(v) for u, v in sc.items()}
        pert[test] = {c: 100.0 - s for c, s in sc[test].items()}
        rf2, ef2, _ = _toy(pert)
        res2 = pr.cv_scores(list(pert), (1, 2, 3), rf2, ef2)
        for k in (1, 2, 3):
            assert res2["combos"][k][i] == res["combos"][k][i]


def test_nested_cv_inner_never_sees_outer_test_and_required_pairs_cover():
    sc = _scores_dict(units=("a", "b", "c"))
    rank_fn, eval_fn, seen = _toy(sc)
    asked = []

    def spy(cid, unit):
        asked.append((cid, unit))
        return eval_fn(cid, unit)

    res = pr.nested_cv(list(sc), (1, 2, 3), rank_fn, spy, n_boot=200, seed=1)
    for o in res["nested"]:
        if o["inner"] is not None:
            for t in o["inner"]["units"]:
                assert t != o["test"]
    need = pr.required_pairs(list(sc), (1, 2, 3), rank_fn)
    assert set(asked) <= need
    assert {u for _, u in need} == set(sc)
    # every ranking call was on a strict subset of the units (never on a set containing its test)
    assert all(len(t) < len(sc) for t in seen)


def test_choose_k_simplicity_rule():
    # k=3 best but k=1 within CI (noisy, mixed-sign differences) -> 1
    r = pr.choose_k({1: [0.5, 0.9, 0.5], 2: [0.6, 0.6, 0.6], 3: [0.7, 0.7, 0.6]}, n_boot=500, seed=1)
    assert r["best_k"] == 3 and r["chosen_k"] == 1
    # k=3 better on every unit by a clear margin -> differences' CI excludes 0 -> 3
    r = pr.choose_k({1: [0.1, 0.1, 0.1], 2: [0.2, 0.2, 0.2], 3: [0.9, 0.9, 0.9]}, n_boot=500, seed=1)
    assert r["chosen_k"] == 3
    # ties on the mean -> the smaller k is best
    r = pr.choose_k({1: [0.5, 0.5], 2: [0.5, 0.5], 3: [0.5, 0.5]}, n_boot=200, seed=1)
    assert r["best_k"] == 1 and r["chosen_k"] == 1


def test_bootstrap_mean_ci_seeded():
    a = pr.bootstrap_mean_ci([0.1, 0.5, 0.9], n_boot=300, seed=3)
    b = pr.bootstrap_mean_ci([0.1, 0.5, 0.9], n_boot=300, seed=3)
    assert a == b and a[1] <= a[0] <= a[2]
    assert np.isnan(pr.bootstrap_mean_ci([np.nan, np.nan])[0])


def test_random_cell_sets_seeded():
    pool = [c.id for c in pr.omega(("last",))]
    a = pr.random_cell_sets(pool, n=20, k=2, seed=5)
    assert a == pr.random_cell_sets(pool, n=20, k=2, seed=5)
    assert a != pr.random_cell_sets(pool, n=20, k=2, seed=6)
    assert all(len(pr.combo_cells(x)) == 2 for x in a)
    assert all(len(pr.combo_cells(x)) == 1 for x in pr.random_cell_sets(pool, n=5, k=1))
    with pytest.raises(ValueError):
        pr.random_cell_sets(pool[:1], k=2)


def test_frames_needed_curve():
    rng = np.random.default_rng(0)
    X = np.zeros((200, 4))
    X[:, 0] = rng.random(200) < 0.9
    X[:, 1] = rng.random(200) < 0.6
    X[:, 2] = rng.random(200) < 0.3
    curve = pr.frames_needed_curve(X, np.ones(4, bool), [0, 1, 2], [5, 50, 200], n_boot=200, seed=1)
    assert [c["n_frames"] for c in curve] == [5, 50, 200]
    assert curve[-1]["p_same_top1"] == 1.0
    assert curve[0]["p_same_top1"] <= curve[1]["p_same_top1"] <= 1.0
    assert all(c["p_top1_in_reference"] >= c["p_same_top1"] for c in curve)
    again = pr.frames_needed_curve(X, np.ones(4, bool), [0, 1, 2], [5, 50, 200], n_boot=200, seed=1)
    assert again == curve
