"""T51 extension (`T51.task.md` 판정 보완, 2026-10-12) on synthetic data — no MuJoCo, no GPU, no storage.

Pinned:
  - scene registry: the five calibration scenes (1800: long + t2; 1807; 0; 500; 1200), held-out
    unchanged, training episodes named in the provenance;
  - scene-level (cluster) bootstrap: one record per scene == the record bootstrap with the same seed;
    runs of one scene are always drawn together; exact enumeration size; seed reproducibility;
  - leave-one-scene-out CV: the ranking never sees any run of the test scene, a test scene's score is
    the mean of its runs, `expand_pairs` covers every evaluation, `group_train_sets` lists exactly the
    tuples the ranker receives; for the real calibration keys long and t2 never split;
  - extra-run wiring: npz format check against the T47 asset format, pairing of a synthetic extra
    record (prompt / frames / episode / format mismatches refused), GT cache location, CLI refusals;
  - the GPU extract command line (pinned producer arguments) and the attention comparison;
  - LIBERO task grouping.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t51_extract as ex
from AG3S.ag3s.experiments.studies import t51_libero as lb
from AG3S.ag3s.experiments.studies import t51_protocol as pr
from AG3S.ag3s.experiments.studies import t51_records as rc
from AG3S.ag3s.experiments.studies import t51_study as st


@pytest.fixture(autouse=True)
def _isolated_extra_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "_EXTRA_DIR", tmp_path / "T51")


# ------------------------------------------------------------------------------ registry


def test_scene_registry_and_calibration_set():
    assert rc.CALIB_KEYS == ("long", "t2", "ep1807", "ep0", "ep500", "ep1200")
    assert rc.HELDOUT_KEYS == tuple(hs.HELDOUT_KEYS)
    assert st.CALIB_KEYS == rc.CALIB_KEYS and st.HELDOUT_KEYS == rc.HELDOUT_KEYS
    assert rc.scene_map(rc.CALIB_KEYS) == {"long": "scene1800", "t2": "scene1800",
                                           "ep1807": "scene1807", "ep0": "scene0",
                                           "ep500": "scene500", "ep1200": "scene1200"}
    names, members = pr.group_members([rc.scene_of(k) for k in rc.CALIB_KEYS])
    assert names == ["scene1800", "scene1807", "scene0", "scene500", "scene1200"]
    assert [m.tolist() for m in members] == [[0, 1], [2], [3], [4], [5]]
    prov = rc.provenance(rc.CALIB_KEYS)
    assert prov["n_scenes"] == 5 and prov["vla_training_episodes"] == ["ep0", "ep500", "ep1200"]
    assert rc.provenance(rc.HELDOUT_KEYS)["vla_training_episodes"] == []
    assert rc.role("ep500") == "calibration_train" and rc.role("ep1850") == "heldout"
    assert rc.scene_group("ep0") == "selection" and rc.scene_group("ep1900") == "new_scene"
    with pytest.raises(SystemExit):
        rc.resolve_keys(["ep0", "ep9"], ())
    with pytest.raises(SystemExit):
        rc.resolve_keys(["ep0", "ep0"], ())
    with pytest.raises(KeyError):
        rc.scene_of("nope")


def test_extra_paths_never_in_storage(tmp_path):
    rc.configure(tmp_path / "X")
    assert rc.npz_path("ep0") == tmp_path / "X" / "attention" / "attention_16d_ep0.npz"
    assert rc.gt_cache_path("ep1200") == tmp_path / "X" / "gt_cache_extra" / "ep1200.npz"
    assert rc.npz_path("long") == hs.npz_path("long")  # T47 keys dispatch unchanged
    with pytest.raises(SystemExit):
        rc.load_gt(tmp_path, "ep500")  # no cache yet -> clear refusal
    np.savez_compressed(rc.gt_cache_path("ep500").parent.mkdir(parents=True) or
                        rc.gt_cache_path("ep500"), class_cov=np.zeros((2, 3, 4, 16, 16)))
    assert rc.load_gt(tmp_path, "ep500")["class_cov"].shape == (2, 3, 4, 16, 16)


# --------------------------------------------------------------------- grouped bootstrap


def _xt(e, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((e, n)), rng.random((e, n))


def test_grouped_bootstrap_singletons_equal_record_bootstrap():
    x, t = _xt(4, 6)
    el = np.ones(6, bool)
    a = pr.bootstrap_ranking(x, t, el, n_boot=300, seed=11)
    b = pr.bootstrap_ranking(x, t, el, n_boot=300, seed=11, groups=["a", "b", "c", "d"])
    for k in ("p_rank1", "p_top3", "ci_lo", "ci_hi", "mean"):
        np.testing.assert_allclose(a[k], b[k], atol=1e-12)
    np.testing.assert_allclose(a["exact"]["p_rank1"], b["exact"]["p_rank1"], atol=1e-12)
    assert a["n_distinct_resamples"] == b["n_distinct_resamples"] == math.comb(7, 4)


def test_group_weights_keep_runs_together():
    members = [np.array([0, 1]), np.array([2]), np.array([3])]
    draws = np.array([[0, 0, 2], [1, 2, 2]])
    w = pr._group_weights(draws, members, 4)  # noqa: SLF001
    assert w.tolist() == [[2, 2, 0, 1], [0, 0, 1, 2]]
    # records 0 and 1 (one scene) always enter with the same multiplicity
    rng = np.random.default_rng(0)
    w = pr._group_weights(rng.integers(0, 3, (500, 3)), members, 4)  # noqa: SLF001
    assert np.array_equal(w[:, 0], w[:, 1])


def test_grouped_bootstrap_scene_structure():
    # cell 0 wins only on run "t2"; cell 1 wins on "long". Same scene -> never drawn apart, so a
    # resample containing scene A always has both runs: their mean is what counts.
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.4], [0.5, 0.4], [0.5, 0.4], [0.5, 0.4]])
    t = np.zeros_like(x)
    groups = ["A", "A", "B", "C", "D", "E"]
    bs = pr.bootstrap_ranking(x, t, np.ones(2, bool), n_boot=400, seed=3, groups=groups)
    # every resample mean is (0.5*nA + 0.5*nOther) for cell 0 vs (0.5*nA + 0.4*nOther) for cell 1:
    # cell 0 never loses (ties only when no other scene is drawn: exactly the all-A resample)
    assert bs["exact"]["n_ordered_resamples"] == 5 ** 5
    assert bs["n_distinct_resamples"] == math.comb(9, 5) and bs["n_groups"] == 5
    assert bs["exact"]["p_rank1"][0] == pytest.approx(1.0)
    assert bs["group_members"] == {"A": [0, 1], "B": [2], "C": [3], "D": [4], "E": [5]}
    # ungrouped: drawing "t2" without "long" makes cell 0 lose sometimes
    ug = pr.bootstrap_ranking(x, t, np.ones(2, bool), n_boot=400, seed=3)
    assert ug["p_rank1"][1] > 0
    again = pr.bootstrap_ranking(x, t, np.ones(2, bool), n_boot=400, seed=3, groups=groups)
    np.testing.assert_array_equal(bs["p_rank1"], again["p_rank1"])
    with pytest.raises(ValueError):
        pr.bootstrap_ranking(x, t, np.ones(2, bool), groups=["A"] * 5)


# --------------------------------------------------------------------- grouped CV


def _cv_world(records, scene_of, seed=0, n=6):
    rng = np.random.default_rng(seed)
    score = {r: rng.random(n) for r in records}
    calls = {"rank": [], "eval": []}

    def rank_fn(train):
        calls["rank"].append(tuple(train))
        m = np.mean([score[r] for r in train], axis=0)
        return [f"last:L{2 + i}H0" for i in np.argsort(-m, kind="stable")]

    def eval_fn(cid, rec):
        calls["eval"].append((cid, rec))
        cells = pr.combo_cells(cid)
        return float(np.mean([score[rec][c.layer - 2] for c in cells]))

    return score, calls, rank_fn, eval_fn


def test_grouped_cv_never_sees_test_scene_and_covers_pairs():
    recs = list(rc.CALIB_KEYS)
    gof = rc.scene_map(recs)
    _, calls, rank_fn, eval_fn = _cv_world(recs, gof)
    scenes, rank_s, eval_s, records_of = pr.grouped_cv(recs, gof, rank_fn, eval_fn)
    assert scenes == ["scene1800", "scene1807", "scene0", "scene500", "scene1200"]
    assert records_of["scene1800"] == ["long", "t2"]
    need = pr.expand_pairs(pr.required_pairs(scenes, pr.K_CHOICES, rank_s), records_of)
    calls["rank"].clear()
    calls["eval"].clear()
    res = pr.nested_cv(scenes, pr.K_CHOICES, rank_s, eval_s, n_boot=200)
    assert set(calls["eval"]) <= need  # every evaluation was planned (no surprise replay)
    for o in res["nested"]:
        test_recs = set(records_of[o["test"]])
        assert o["test"] not in o["train"]
        for inner_train, inner_test in pr.loeo_folds(o["train"]):
            assert not test_recs & set(r for s in inner_train for r in records_of[s])
    # no ranking call ever contains one run of a scene without the other
    for train in calls["rank"]:
        assert ("long" in train) == ("t2" in train)
    # the ranker receives exactly the tuples the 2D preselection was computed for
    assert set(calls["rank"]) <= set(pr.group_train_sets(recs, gof)) | {tuple(recs)}
    assert len(pr.group_train_sets(recs, gof)) == 5 + math.comb(5, 3)


def test_grouped_cv_scene_score_is_mean_of_runs_and_test_flip_is_invisible():
    recs = list(rc.CALIB_KEYS)
    gof = rc.scene_map(recs)
    score, _, rank_fn, eval_fn = _cv_world(recs, gof, seed=4)
    _, rank_s, eval_s, _ = pr.grouped_cv(recs, gof, rank_fn, eval_fn)
    assert eval_s("last:L3H0", "scene1800") == pytest.approx(np.mean([score["long"][1],
                                                                      score["t2"][1]]))
    train = ["scene1807", "scene0", "scene500", "scene1200"]
    before = list(rank_s(train))
    score["long"][:] = score["long"][::-1] * 10  # perturb only the held-out scene's runs
    score["t2"][:] = -score["t2"]
    assert list(rank_s(train)) == before


def test_train_subsets_scene_mode():
    keys = list(rc.CALIB_KEYS)
    subs = st.train_subsets(keys, rc.scene_map(keys))
    assert subs == pr.group_train_sets(keys, rc.scene_map(keys))
    assert ("t2", "ep1807", "ep0", "ep500", "ep1200") not in subs  # long never left out alone
    assert ("ep1807", "ep0", "ep500", "ep1200") in subs
    assert st.train_subsets(["a", "b", "c"])[0] == ("b", "c")  # record mode unchanged


# --------------------------------------------------------------------- npz format / pairing


def _fields(n=5, **over):
    f = {"t_step": np.arange(n, dtype=np.int64) * 8, "denoise_steps": np.asarray([0, 4, 9]),
         "aggregations": np.asarray(["mean", "first", "last"]),
         "cameras": np.asarray(list(hs.POLICY_CAMERAS)), "prompt": np.asarray("Put the banana in "
                                                                             "the basket."),
         "noise_seeds": np.asarray([0]), "records": np.asarray("/x/run_0000"),
         "checkpoint": np.asarray("/somewhere/" + rc.CHECKPOINT_TAIL)}
    f.update(over)
    return f


def test_format_problems():
    shape = (5, 3, 3, 18, 8, 3, 16, 16)
    assert rc.format_problems(_fields(), shape, "float16") == []
    bad = {"denoise_steps": rc.format_problems(_fields(denoise_steps=np.asarray([0, 9])), shape, "float16"),
           "agg": rc.format_problems(_fields(aggregations=np.asarray(["last", "mean", "first"])),
                                     shape, "float16"),
           "seed": rc.format_problems(_fields(noise_seeds=np.asarray([0, 1])), shape, "float16"),
           "ckpt": rc.format_problems(_fields(checkpoint=np.asarray("/c/pi05_rby1_lora/x/29999")),
                                      shape, "float16"),
           "dtype": rc.format_problems(_fields(), shape, "float32"),
           "shape": rc.format_problems(_fields(), (5, 1, 3, 18, 8, 3, 16, 16), "float16"),
           "frames": rc.format_problems(_fields(n=4), shape, "float16")}
    for k, v in bad.items():
        assert v, k
    f = _fields()
    del f["noise_seeds"]
    assert rc.format_problems(f, shape, "float16")


def test_npz_array_header(tmp_path):
    p = tmp_path / "a.npz"
    np.savez_compressed(p, attention=np.zeros((2, 3, 3, 18, 8, 3, 16, 16), np.float16), x=np.ones(3))
    assert rc.npz_array_header(p, "attention") == ((2, 3, 3, 18, 8, 3, 16, 16), "float16")


def _fake_extra(tmp_path, monkeypatch, *, episode=500, prompt="Put the banana in the basket.",
                n=4, npz_over=None, n_npz=None):
    rec = tmp_path / "rec" / "run_0000"
    rec.mkdir(parents=True)
    argv = ["pi05_infer.py", "--episode-index", str(episode), "--record-ag3s", "outputs/x/ep500"]
    (rec / "meta.json").write_text(json.dumps({"prompt": prompt, "argv": argv}))
    for i in range(n):
        np.savez(rec / f"step_{i:05d}.npz", t_step=np.asarray(i * 8))
    monkeypatch.setattr(rc, "record_path", lambda key: rec)
    nn = n if n_npz is None else n_npz
    f = _fields(n=nn, records=np.asarray(str(rec.resolve())))
    f.update(npz_over or {})
    p = rc.npz_path("ep500")
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, attention=np.zeros((nn, 3, 3, 18, 8, 3, 16, 16), np.float16), **f)
    return rec


def test_check_pairing_extra_ok(tmp_path, monkeypatch):
    _fake_extra(tmp_path, monkeypatch)
    info = rc.check_pairing("ep500")
    assert info["scene_id"] == "scene500" and info["vla_training_episode"] is True
    assert info["n_frames"] == 4
    att, dn, ag, cams = rc.load_attention("ep500")
    assert att.shape[0] == 4 and dn == [0, 4, 9] and ag == ["mean", "first", "last"]
    assert rc.n_frames("ep500") == 4
    assert rc.target_of("ep500") == "banana"


@pytest.mark.parametrize("kw", [
    {"episode": 501},
    {"prompt": "Put the pear in the basket."},
    {"n_npz": 3},
    {"npz_over": {"denoise_steps": np.asarray([0])}},
    {"npz_over": {"records": np.asarray("/elsewhere/run_0000")}},
])
def test_check_pairing_extra_refuses(tmp_path, monkeypatch, kw):
    _fake_extra(tmp_path, monkeypatch, **kw)
    with pytest.raises(SystemExit):
        rc.check_pairing("ep500")


def test_check_pairing_extra_missing_npz(tmp_path, monkeypatch):
    rec = tmp_path / "r"
    rec.mkdir()
    (rec / "meta.json").write_text("{}")
    monkeypatch.setattr(rc, "record_path", lambda key: rec)
    with pytest.raises(SystemExit, match="t51_extract"):
        rc.check_pairing("ep0")


def test_gt_extra_refuses_t47_keys(tmp_path):
    a = st.build_parser().parse_args(["--stage", "gt_extra", "--records", "long"])
    with pytest.raises(SystemExit, match="read-only"):
        st.stage_gt_extra(a, tmp_path)


def test_noun_query_refused_for_extra_runs(tmp_path, monkeypatch):
    _fake_extra(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="T49"):
        st.query_blocks("ep500", ["last", st.NOUN_QUERY], t49=str(tmp_path))


# --------------------------------------------------------------------- GPU extract plumbing


def test_producer_argv_pins_t47_format(tmp_path):
    argv = ex.producer_argv(pathlib.Path("/r/run_0000"), pathlib.Path("/c/29999"),
                            tmp_path / "o.npz", python="py")
    assert argv[:3] == ["py", "-m", "AG3S.ag3s.experiments.sources.pi05_attention"]
    s = " ".join(argv)
    for frag in ("--config pi05_rby1_randomized_pick_place_16d_lora", "--num-denoise 10",
                 "--denoise-steps 0 4 9", "--agg mean first last", "--store-dtype float16",
                 "--noise-seeds 0"):
        assert frag in s
    assert "--limit" not in argv
    assert ex.producer_argv(pathlib.Path("/r"), pathlib.Path("/c"), tmp_path / "o.npz",
                            limit=4)[-2:] == ["--limit", "4"]
    env = ex.producer_env("cuda")
    assert env["JAX_PLATFORMS"] == "cuda" and env["XLA_PYTHON_CLIENT_PREALLOCATE"] in ("false",)
    # the parser rejects nothing it needs and never defaults to writing T47 assets
    a = ex.build_parser().parse_args(["--stage", "extract"])
    assert a.records is None and a.parity_record == "ep1807" and not a.allow_busy_gpu


def test_extract_refuses_t47_keys_without_gpu_call(monkeypatch, tmp_path):
    monkeypatch.setattr(ex, "gpu_guard", lambda args: [])
    monkeypatch.setattr(ex, "checkpoint_path", lambda args: tmp_path)
    a = ex.build_parser().parse_args(["--stage", "extract", "--records", "long"])
    with pytest.raises(SystemExit, match="read-only"):
        ex.stage_extract(a)


def test_checkpoint_path_requires_the_t47_step(tmp_path):
    p = tmp_path / "other" / "29999"
    (p / "params").mkdir(parents=True)
    with pytest.raises(SystemExit):
        ex.checkpoint_path(argparse.Namespace(checkpoint=str(p)))
    q = tmp_path / rc.CHECKPOINT_TAIL
    (q / "params").mkdir(parents=True)
    assert ex.checkpoint_path(argparse.Namespace(checkpoint=str(q))) == q


def test_compare_attention():
    rng = np.random.default_rng(0)
    a = rng.random((2, 3, 3, 18, 8, 3, 16, 16)).astype(np.float16)
    c = ex.compare_attention(a, a)
    assert c["bitwise_equal"] and c["max_abs"] == 0 and c["argmax_patch_same_share"] == 1.0
    assert c["serving_cell_pearson_r"] == pytest.approx(1.0)
    b = a.astype(np.float32).copy()
    b[0, 0, 0, 0, 0, 0] = 0.0
    b[0, 0, 0, 0, 0, 0, 5, 5] = 9.0  # one map's argmax moves
    c2 = ex.compare_attention(b, a)
    n_maps = 2 * 3 * 3 * 18 * 8 * 3
    assert not c2["bitwise_equal"]
    assert c2["argmax_patch_same_share"] == pytest.approx(1 - 1 / n_maps)
    with pytest.raises(ValueError):
        ex.compare_attention(a[:1], a)


# ------------------------------------------------------------------------------ LIBERO


def test_libero_task_groups():
    assert lb.task_of("libero_10_task0_ep1.npz") == "libero_10_task0"
    assert lb.task_of("libero_spatial_task4_ep0") == "libero_spatial_task4"
    with pytest.raises(ValueError):
        lb.task_of("weird.npz")
    a = lb.build_parser().parse_args(["--stage", "bootstrap"])
    assert a.unit == "task" and a.init == "gaussian"
    assert lb.attention_dir(pathlib.Path("/r")) == pathlib.Path("/r/attention")
    assert lb.attention_dir(pathlib.Path("/r"), "zeros") == pathlib.Path("/r/attention_zeros")


def test_libero_bootstrap_task_unit(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    names = [f"libero_10_task{t}_ep{e}.npz" for t in range(3) for e in range(2)]
    lift = rng.random((6, 3, 16, 8))
    np.savez_compressed(tmp_path / "score2d_arrays.npz", episodes=np.asarray(names),
                        queries=np.asarray(["last", "mean", "first"]), lift=lift,
                        peak_on_target=rng.random((6, 3, 16, 8)), knows_hit=lift,
                        eligible=np.ones((3, 16, 8), bool))
    monkeypatch.setattr(lb.hs, "git_state", lambda: {})
    out = lb.stage_bootstrap(argparse.Namespace(n_boot=200, unit="task"), tmp_path)
    assert out["unit"] == "task" and out["n_units"] == 3
    ep = lb.stage_bootstrap(argparse.Namespace(n_boot=200, unit="episode"), tmp_path)
    assert ep["n_units"] == 6 and (tmp_path / "bootstrap__episode.json").is_file()
