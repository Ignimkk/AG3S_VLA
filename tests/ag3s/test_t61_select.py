import numpy as np

from AG3S.ag3s.experiments.studies import t61_select as S


def test_preselect_allocates_per_query_and_ignores_forced():
    n = S.N_PER_QUERY
    score = np.zeros(4 * n); tie = np.zeros(4 * n); el = np.ones(4 * n, bool)
    score[:n] = np.arange(n)            # query 0: best are the last indices
    score[n:2 * n] = 100 + np.arange(n)  # query 1 dominates globally
    el[2 * n:3 * n] = False              # query 2: nothing eligible
    el[3 * n] = True; el[3 * n + 1:] = False
    out = S.preselect(score, tie, el, n_top=16, forced=[5, 7])
    assert out["per_query"][0] == list(range(n - 1, n - 1 - S.PER_QUERY, -1))
    assert len(out["per_query"][1]) == S.PER_QUERY
    assert out["per_query"][2] == []
    assert out["per_query"][3] == [3 * n]
    assert out["forced"] == [] and out["forced_ignored"] == [5, 7]
    assert 5 not in out["ranked_set"] and 7 not in out["ranked_set"]


def test_choose_k_one_se_picks_smallest_within_one_se():
    scores = {1: [0.60, 0.62, 0.58, 0.61, 0.59], 2: [0.64, 0.66, 0.60, 0.63, 0.62], 3: [0.70, 0.40, 0.90, 0.50, 0.80]}
    out = S.choose_k(scores)
    assert out["best_k"] == 3  # mean 0.66
    se3 = np.std(scores[3], ddof=1) / np.sqrt(5)
    assert abs(out["threshold"] - (0.66 - se3)) < 1e-12
    assert out["chosen_k"] == 1  # 0.60 >= 0.66 - 0.0927


def test_choose_k_keeps_best_when_others_outside():
    scores = {1: [0.1] * 5, 2: [0.9, 0.9, 0.9, 0.9, 0.91]}
    assert S.choose_k(scores)["chosen_k"] == 2


def test_ranking_tie_breaks_purity_then_coverage_then_order(monkeypatch):
    ids = ["c0", "c1", "c2", "c3"]
    prim = {"c0": 0.5, "c1": 0.6, "c2": 0.6, "c3": 0.6}
    pur = {"c0": 0.9, "c1": 0.7, "c2": 0.8, "c3": 0.8}
    cov = {"c0": 0.1, "c1": 0.9, "c2": 0.5, "c3": 0.6}
    monkeypatch.setattr(S.st, "metric_matrix",
                        lambda root, units, sids, field: np.array([[(prim if field != "purity_mean" else pur)[s] for s in sids]]))
    monkeypatch.setattr(S, "metric_rows", lambda root, units, sids, fn: np.array([[cov[s] for s in sids]]))
    monkeypatch.setattr(S.st, "calib_mean", lambda m, units, mode: np.asarray(m)[0])
    out = S.ranking_3d(None, "gt", ["u"], ids, {}, ["u"], "scene")
    assert out == ["c3", "c2", "c1", "c0"]


def test_coverage_counts_no_output_as_zero_and_skips_unobserved():
    rows = [{"n_target_obs": 10, "chosen__target": 5, "chosen_n": 8},
            {"n_target_obs": 10, "chosen__target": 0, "chosen_n": 0},
            {"n_target_obs": 0, "chosen__target": 0, "chosen_n": 3}]
    assert S._coverage(rows) == 0.25


def test_heldout_registry_matches_rule():
    from collections import Counter
    from AG3S.ag3s.experiments.studies import t61_records as R

    assert R.selection_problems(R.load_meta()) == []
    assert not set(R.EPISODES) & set(R.USED)
    assert Counter(v[0] for v in R.EXPECTED.values()) == {"apple": 4, "banana": 4, "orange": 4, "pear": 4}
    assert Counter(v[1] for v in R.EXPECTED.values()) == {"left": 8, "right": 8}
    assert [r["port"] for r in R.seeds_table()][:2] == [8292, 8293]
