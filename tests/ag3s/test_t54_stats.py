"""T54 metrics / aggregation / bootstrap (pure numpy) and the CLI's guards.

* no output counts as 0 in identity and coverage (and is excluded from purity / neighbour);
* episode means weight episodes equally, not frames;
* the episode bootstrap resamples whole episodes (never frames), with one shared set of draws;
* the moving-block sensitivity resamples contiguous target frames inside an episode;
* the T57 entry rows are mechanical comparisons of the fixed tolerances.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t54_stats as st


def _row(frame, *, out=True, ident=True, purity=1.0, n_obs=100, cov=0.8, other=0, dest=0, table=0,
         present=True, on=True, t_step=None, ms=50.0):
    return {"frame": frame, "t_step": frame * 8 if t_step is None else t_step,
            "has_target": out, "identity_correct": ident and out,
            "chosen_cluster_purity": purity if out else float("nan"),
            "n_target_obs": n_obs, "coverage": (cov if out else 0.0) if n_obs else float("nan"),
            "chosen__other_object": other if out else 0, "chosen__destination": dest if out else 0,
            "chosen__table": table if out else 0, "target_candidate_present": present,
            "on_target": on and out, "ms_lift_ground": ms}


def _gt(n, visible=None, px=500):
    cp = np.zeros((n, 3, 7), np.int64)
    vis = np.ones(n, bool) if visible is None else np.asarray(visible, bool)
    cp[vis, 0, 0] = px
    return {"t_step": np.arange(n) * 8, "class_px": cp}


# ------------------------------------------------------------------------------- frames


def test_no_output_counts_as_zero_in_identity_and_coverage():
    v = st.frame_values(_row(0, out=False, n_obs=120))
    assert v["identity"] == 0.0 and v["coverage"] == 0.0 and v["no_target"] == 1.0
    assert math.isnan(v["purity"]) and math.isnan(v["neighbour"]) and v["neighbour_all"] == 0.0
    assert math.isnan(v["destination"]) and math.isnan(v["support"])
    # a frame whose cloud has no observed target point has no coverage, for any arm
    assert math.isnan(st.frame_values(_row(0, out=False, n_obs=0))["coverage"])
    assert math.isnan(st.frame_values(_row(0, out=True, n_obs=0))["coverage"])
    # an output frame keeps its values
    w = st.frame_values(_row(0, out=True, ident=False, purity=0.4, cov=0.5, other=7, present=True))
    assert (w["identity"], w["purity"], w["coverage"], w["neighbour"]) == (0.0, 0.4, 0.5, 7.0)
    assert w["candidate_fail"] == 0.0 and w["choice_fail"] == 1.0
    x = st.frame_values(_row(0, out=True, ident=False, present=False))
    assert x["candidate_fail"] == 1.0 and x["choice_fail"] == 0.0


def test_has_target_frames_any_camera_threshold():
    cp = np.zeros((3, 3, 7), np.int64)
    cp[0, 2, 0] = 200
    cp[1, 1, 0] = 199
    assert st.has_target_frames(cp).tolist() == [True, False, False]
    assert st.has_target_frames(cp, 1).tolist() == [True, True, False]


def test_build_frames_aligns_by_t_step_and_assigns_phase():
    rows = {("ep1", "current", "L8H2"): [_row(0), _row(1), _row(2)]}
    gt = {"ep1": _gt(3, visible=[True, False, True])}
    grasp = {"ep1": {"t_step": np.arange(3) * 8, "grasp_state": np.asarray(["pre", "held", "placed"])}}
    fr = st.build_frames(rows, gt, grasp)
    assert [f["has_target_gt"] for f in fr] == [True, False, True]
    assert [f["phase"] for f in fr] == ["pre", "held", "placed"]
    bad = {("ep1", "current", "L8H2"): [_row(0, t_step=3)]}
    with pytest.raises(ValueError):
        st.build_frames(bad, gt, None)
    with pytest.raises(ValueError):
        st.build_frames(rows, gt, {"ep1": {"t_step": np.arange(3), "grasp_state": grasp["ep1"]["grasp_state"]}})


# ------------------------------------------------------------------------- episode means


def _frames_two_episodes():
    """ep_a: 70 target frames all identity-correct; ep_b: 5 target frames, all no output."""
    rows = {("a", "current", "L8H2"): [_row(i) for i in range(70)],
            ("b", "current", "L8H2"): [_row(i, out=False) for i in range(5)],
            ("a", "current", "A"): [_row(i) for i in range(70)],
            ("b", "current", "A"): [_row(i, out=True) for i in range(5)]}
    gt = {"a": _gt(70), "b": _gt(5)}
    return st.build_frames(rows, gt, None)


def test_episode_mean_weights_episodes_not_frames():
    fr = _frames_two_episodes()
    e, n = st.episode_matrix(fr, keys=["a", "b"], arms=["L8H2", "A"], condition="current",
                             metric="identity")
    assert e[:, 0].tolist() == [1.0, 0.0] and n[:, 0].tolist() == [70, 5]
    s = st.summarize_matrix(e, n, arms=["L8H2", "A"], keys=["a", "b"],
                            draws=st.episode_draws(2, 2000), base="L8H2")
    assert s["L8H2"]["mean"] == 0.5                 # not 70/75
    assert s["A"]["diff_vs_L8H2"]["mean"] == 0.5     # (0 + 1) / 2 paired episode differences
    assert s["A"]["diff_vs_L8H2"]["episodes"] == {"a": 0.0, "b": 1.0}
    # purity: output frames only -> episode b of L8H2 has no value and is left out
    ep, np_ = st.episode_matrix(fr, keys=["a", "b"], arms=["L8H2", "A"], condition="current",
                                metric="purity")
    assert math.isnan(ep[1, 0]) and np_[1, 0] == 0 and ep[1, 1] == 1.0
    sp = st.summarize_matrix(ep, np_, arms=["L8H2", "A"], keys=["a", "b"],
                             draws=st.episode_draws(2, 200), base="L8H2")
    assert sp["L8H2"]["n_episodes"] == 1 and sp["A"]["diff_vs_L8H2"]["n_episodes"] == 1


def test_episode_bootstrap_resamples_whole_episodes():
    d = st.episode_draws(16, 10000, 5407)
    assert d.shape == (10000, 16) and d.min() >= 0 and d.max() <= 15
    assert np.array_equal(d, st.episode_draws(16, 10000, 5407))      # seeded, shared across arms
    vals = np.arange(16, dtype=float)
    b = st.boot_mean(vals, d)
    assert np.allclose(b, vals[d].mean(axis=1))
    # every resample is a mean of 16 episode values: with all episodes equal the CI is degenerate,
    # whatever their frame counts were (frames are never resampled at this level)
    e = np.full((16, 2), 0.25)
    n = np.tile(np.array([[3, 3]]), (16, 1))
    n[0] = [70, 70]
    s = st.summarize_matrix(e, n, arms=["L8H2", "A"], keys=[f"k{i}" for i in range(16)], draws=d,
                            base="L8H2")
    assert s["L8H2"]["ci95"] == [0.25, 0.25]
    assert s["A"]["diff_vs_L8H2"]["ci95_episode_bootstrap"] == [0.0, 0.0]
    # NaN episodes are left out of a resample, not counted as zero
    v = np.r_[np.full(15, 1.0), np.nan]
    assert np.allclose(st.boot_mean(v, d), 1.0)


def test_moving_block_positions_are_contiguous_blocks_within_the_episode():
    rng = np.random.default_rng(0)
    pos = st.block_positions(12, rng, 50, 5)
    assert pos.shape == (50, 12) and pos.min() >= 0 and pos.max() <= 11
    for row in pos:
        for b in range(0, 10, 5):
            blk = row[b:b + 5]
            assert np.array_equal(blk, np.arange(blk[0], blk[0] + 5))
    assert st.block_positions(3, rng, 4, 5).tolist() == [[0, 1, 2]] * 4   # L = min(block, n)


def test_moving_block_resamples_frames_of_fixed_episodes():
    fr = _frames_two_episodes()
    mb = st.moving_block(fr, keys=["a", "b"], arms=["L8H2", "A"], condition="current",
                         metric="identity", base="L8H2", n_boot=500)
    assert mb["L8H2"]["ci95"] == [0.5, 0.5]      # constant within each episode -> degenerate
    assert mb["A"]["diff_ci95"] == [0.5, 0.5]


def test_frame_mcnemar_and_timing():
    fr = _frames_two_episodes()
    m = st.frame_mcnemar(fr, condition="current", arm="A", base="L8H2")
    assert (m["n_pairs"], m["arm_only"], m["base_only"]) == (75, 5, 0)
    assert m["p_exact"] == pytest.approx(2 * 0.5 ** 5)
    t = st.timing(fr, condition="current", arms=["L8H2", "A"], extract_ms={"action": 30.0, "noun": 20.0},
                  extract_kinds={"L8H2": ["action"], "A": ["action", "noun"]}, base="L8H2")
    assert t["A"]["lift_ground_ms"]["median"] == 50.0 and t["A"]["e2e_ms"]["median"] == 100.0
    assert t["L8H2"]["e2e_ms"]["median"] == 80.0
    assert t["A"]["extra_vs_L8H2"]["e2e_ms"]["median"] == 20.0
    t2 = st.timing(fr, condition="current", arms=["L8H2"])
    assert t2["L8H2"]["e2e_ms"] is None


# --------------------------------------------------------------------------- T57 entry


def _summary(di, dc, dp, dn, hi, hc, e2e=None):
    def m(v):
        return {"A": {"diff_vs_L8H2": {"mean": v, "ci95_episode_bootstrap": [v - 0.1, v + 0.1]}}}
    return {"current": {"all": {"identity": m(di), "coverage": m(dc), "purity": m(dp),
                                "neighbour": m(dn), "no_target": m(0.0)},
                        "timing": {"A": {"lift_ground_ms": {"median": 1.0}, "e2e_ms": e2e}}},
            "hold": {"all": {"identity": m(hi), "coverage": m(hc)}}}


def test_entry_criteria_are_mechanical():
    c = st.entry_criteria(_summary(-0.03, -0.05, 0.0, 0.0, -0.02, 0.0))
    r = c["rows"]
    assert r["identity"]["pass"] and r["coverage"]["pass"]           # boundaries included
    assert not r["improvement_observed"]["pass"]                      # nothing improved
    assert not c["all_numeric_rows_pass"]
    c2 = st.entry_criteria(_summary(-0.01, -0.02, 0.0, -3.0, -0.031, 0.0),
                           budget_ms=100.0)
    assert c2["rows"]["improvement_observed"]["neighbour_down"] and not c2["rows"]["hold"]["pass"]
    assert c2["rows"]["timing"]["within_budget_median"] is None       # no e2e given
    c3 = st.entry_criteria(_summary(0.02, 0.0, 0.01, 0.0, 0.0, 0.0,
                                    e2e={"median": 90.0, "p95": 120.0}), budget_ms=100.0)
    assert c3["all_numeric_rows_pass"]
    assert c3["rows"]["timing"]["within_budget_median"] is True
    assert c3["rows"]["timing"]["within_budget_p95"] is False
    assert st.IDENTITY_TOL == -0.03 and st.COVERAGE_TOL == -0.05
    assert (st.N_BOOT, st.SEED, st.BLOCK) == (10000, 5407, 5)


def test_analyze_end_to_end_small():
    fr = _frames_two_episodes()
    out = st.analyze(fr, keys=["a", "b"], arms=["L8H2", "A"], conditions=["current"], n_boot=200,
                     phases=(), log=lambda m: None)
    cur = out["current"]
    assert cur["all"]["identity"]["A"]["diff_vs_L8H2"]["mean"] == 0.5
    assert cur["n_target_frames"] == 75 and cur["sensitivity_1px"]["identity"]["L8H2"]["mean"] == 0.5


# ------------------------------------------------------------------------------- CLI guards


def test_cli_parser_and_run_dir_guards(tmp_path):
    from AG3S.ag3s.experiments.studies import t54_study as cli

    ap = cli.build_parser()
    with pytest.raises(SystemExit):
        ap.parse_args(["--set", "dev", "--stage", "inputs"])           # --run-dir is required
    a = ap.parse_args(["--run-dir", "x", "--set", "test", "--stage", "analyze"])
    assert (a.n_boot, a.seed, a.block, a.min_target_px) == (10000, 5407, 5, 200)
    assert a.conditions == ["current", "hold"] and a.workers == 1
    with pytest.raises(SystemExit):
        cli.out_root(str(tmp_path))                                     # not a development run
    with pytest.raises(SystemExit):
        cli.out_root(cli.DEFAULTS["t53_run"], cli.DEFAULTS["t53_run"])  # never the T53 run
    with pytest.raises(SystemExit):
        cli.stage_replay(ap.parse_args(["--run-dir", "x", "--set", "dev", "--stage", "replay",
                                        "--workers", "5"]), tmp_path)


def test_cli_parity_compare_rows_is_nan_safe():
    from AG3S.ag3s.experiments.studies import t54_study as cli

    mine = [{"frame": 0, "identity_correct": True, "seed_recall": float("nan"), "n_cloud": 5},
            {"frame": 1, "identity_correct": False, "seed_recall": 0.5, "n_cloud": 6}]
    ref = [{"frame": 0, "identity_correct": True, "seed_recall": float("nan"), "n_cloud": 5},
           {"frame": 1, "identity_correct": True, "seed_recall": 0.5, "n_cloud": 6}]
    c = cli.compare_rows(mine, ref, ("identity_correct", "seed_recall", "n_cloud"))
    assert c["fields_compared"] == 6 and c["n_mismatch"] == 1
    assert c["mismatch"][0]["frame"] == 1 and c["mismatch"][0]["field"] == "identity_correct"
