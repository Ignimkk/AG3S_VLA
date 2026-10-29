"""T51 aggregation unit (`T51.task.md` 판정 보완 → 집계 단위, fix 2026-10-12) on synthetic data — no
MuJoCo, no GPU, no storage.

Pinned:
  - `t51_protocol.aggregate_mean` / `group_means`: record mean vs scene mean (runs of a scene first);
  - the grouped bootstrap with ``group_mean=True``: same draws as the record-level estimator, the
    resample estimate is the mean of the drawn groups' means (checked against brute-force enumeration),
    and singletons reduce to the plain record bootstrap;
  - `t51_study` ``--aggregate``: a calibration where scene1800 (long + t2) flips the winner — the
    record mean picks A, the scene mean picks B — in score2d (2D preselection), score3d (P2),
    bootstrap (P3), ksel (P4 fold rankers), annotate; ``scene`` outputs carry ``__scene`` and the
    default (record) outputs are untouched by a scene run (byte-identical);
  - default behaviour pinned: record-mean values / winner and the original file names;
  - held-out / ablation point estimate under scene (headfix + ep1800 share scene1800);
  - T51b: ``heldout_compare`` / ``summary`` read and write the ``__scene`` files;
  - LIBERO ``--aggregate task`` (a task with a NaN episode flips the winner) vs the default.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t47_head_selection as hs
from AG3S.ag3s.experiments.studies import t51_libero as lb
from AG3S.ag3s.experiments.studies import t51_protocol as pr
from AG3S.ag3s.experiments.studies import t51_records as rc
from AG3S.ag3s.experiments.studies import t51_study as st
from AG3S.ag3s.experiments.studies import t51b_study as tb

KEYS = rc.CALIB_KEYS  # long, t2 (scene1800), ep1807, ep0, ep500, ep1200
A, B = "last:L5H0", "last:L6H1"
C = "last:L8H2"  # third ranked cell (low everywhere)
#: per-record 3D identity rates (exact on 20 frames): long / t2 lift A, the other scenes favour B.
RATE = {A: {"long": 1.0, "t2": 1.0, "ep1807": 0.5, "ep0": 0.5, "ep500": 0.5, "ep1200": 0.5},
        B: {"long": 0.4, "t2": 0.4, "ep1807": 0.75, "ep0": 0.75, "ep500": 0.75, "ep1200": 0.75},
        C: {k: 0.1 for k in KEYS}}
#: record mean A = 4/6 = .667 > B = 3.8/6 = .633; scene mean A = 3/5 = .6 < B = 3.4/5 = .68
N_FRAMES = 20


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(rc, "_EXTRA_DIR", tmp_path / "T51")
    monkeypatch.setattr(hs, "git_state", lambda: {})
    yield
    tb.uninstall()


def _args(mod=st, **kw) -> argparse.Namespace:
    a = mod.build_parser().parse_args(["--stage", kw.pop("stage", "bootstrap")])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


# ------------------------------------------------------------------------------ protocol


def test_aggregate_mean_record_vs_scene():
    x = np.array([[RATE[A][k], RATE[B][k]] for k in KEYS])
    groups = [rc.scene_of(k) for k in KEYS]
    np.testing.assert_allclose(pr.aggregate_mean(x), [4 / 6, 3.8 / 6])
    np.testing.assert_allclose(pr.aggregate_mean(x, groups), [3 / 5, 3.4 / 5])
    assert pr.rank_order(pr.aggregate_mean(x))[0] == 0  # record mean: A
    assert pr.rank_order(pr.aggregate_mean(x, groups))[0] == 1  # scene mean: B
    names, gm = pr.group_means(x, groups)
    assert names == ["scene1800", "scene1807", "scene0", "scene500", "scene1200"]
    assert gm.shape == (5, 2) and gm[0].tolist() == [1.0, 0.4]
    # NaN: a NaN run is skipped inside its scene; an all-NaN scene is skipped across scenes
    y = np.array([[1.0], [np.nan], [0.0], [np.nan]])
    assert pr.aggregate_mean(y, ["a", "a", "b", "c"])[0] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        pr.group_means(y, ["a"])


def _brute_group_mean_p_rank1(x, t, groups):
    names, xg = pr.group_means(x, groups)
    _, tg = pr.group_means(t, groups)
    g = len(names)
    wins = np.zeros(x.shape[1])
    for d in itertools.product(range(g), repeat=g):
        m = pr.unit_mean(xg[list(d)])
        tt = pr.unit_mean(tg[list(d)])
        wins[pr.rank_order(m, tt)[0]] += 1
    return wins / g ** g


def test_bootstrap_group_mean_matches_enumeration_and_differs_from_record():
    x = np.array([[RATE[A][k], RATE[B][k], RATE[C][k]] for k in KEYS])
    t = np.zeros_like(x)
    groups = [rc.scene_of(k) for k in KEYS]
    el = np.ones(3, bool)
    rec = pr.bootstrap_ranking(x, t, el, n_boot=500, seed=pr.SEED_BOOTSTRAP, groups=groups)
    scn = pr.bootstrap_ranking(x, t, el, n_boot=500, seed=pr.SEED_BOOTSTRAP, groups=groups,
                               group_mean=True)
    np.testing.assert_allclose(rec["mean"], [4 / 6, 3.8 / 6, 0.1])
    np.testing.assert_allclose(scn["mean"], [3 / 5, 3.4 / 5, 0.1])
    assert scn["aggregate"] == "group_mean" and "aggregate" not in rec
    np.testing.assert_allclose(scn["exact"]["p_rank1"], _brute_group_mean_p_rank1(x, t, groups))
    assert rec["exact"]["p_rank1"][0] > rec["exact"]["p_rank1"][1]
    assert scn["exact"]["p_rank1"][1] > scn["exact"]["p_rank1"][0]
    assert scn["n_distinct_resamples"] == rec["n_distinct_resamples"] == 126
    # Monte-Carlo resample = the group means of the same draws
    rng = np.random.default_rng(pr.SEED_BOOTSTRAP)
    draws = rng.integers(0, 5, size=(500, 5))
    _, xg = pr.group_means(x, groups)
    means = np.stack([pr.unit_mean(xg[d]) for d in draws])
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    np.testing.assert_allclose(scn["ci_lo"], lo)
    np.testing.assert_allclose(scn["ci_hi"], hi)
    # one record per group: group mean == record bootstrap (same seed)
    y, ty = np.random.default_rng(0).random((4, 5)), np.random.default_rng(1).random((4, 5))
    a = pr.bootstrap_ranking(y, ty, np.ones(5, bool), n_boot=300, seed=11)
    b = pr.bootstrap_ranking(y, ty, np.ones(5, bool), n_boot=300, seed=11, groups=list("abcd"),
                             group_mean=True)
    for k in ("p_rank1", "p_top3", "ci_lo", "ci_hi", "mean"):
        np.testing.assert_allclose(a[k], b[k], atol=1e-12)
    with pytest.raises(ValueError):
        pr.bootstrap_ranking(y, ty, np.ones(5, bool), group_mean=True)


# ---------------------------------------------------------------------------- t51_study


def _write_rows(root, key, sid, rate, n=N_FRAMES):
    p = st.rows_path(root, key, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    k_ok = int(round(rate * n))
    with p.open("w") as fh:
        for i in range(n):
            ok = i < k_ok
            fh.write(json.dumps({"frame": i, "identity_correct": ok, "on_target": ok,
                                 "has_target": True, "chosen_cluster_purity": 0.9,
                                 "nontarget_seed_loss": 0, "nontarget_seed_loss__other_object": 0,
                                 "self_labeled": i < 10, "self_identity_correct": ok if i < 10 else None,
                                 "self_out_n": 1, "source": sid, "record": key}) + "\n")


def _rate(sid: str, key: str) -> float:
    """Single cells: RATE (held-out records reuse the scene-mates' pattern); a sum: the max of its cells."""
    cells = [c.id for c in pr.combo_cells(sid)]
    ref = {"headfix": "long", "ep1800": "t2", "ep1850": "ep500", "ep1900": "ep0"}.get(key, key)
    return max(RATE[c][ref] for c in cells)


def _fake_score2d(root, tag=""):
    """A score2d payload / arrays with the 2D lift equal for A and B (ties go to the 3D rate)."""
    out = root / "gt" / "score2d"
    out.mkdir(parents=True, exist_ok=True)
    ranked = [A, B, C]
    folds = {st.subset_key(t): {"ranked_set": ranked, "scored_set": ranked}
             for t in st.train_subsets(list(KEYS), rc.scene_map(KEYS))}
    js = {"records": list(KEYS), "candidates_3d": ranked,
          "global": {"ranked_set": ranked, "scored_set": ranked}, "folds": folds}
    (out / f"score2d{tag}.json").write_text(json.dumps(js))
    lift = np.ones((len(KEYS), 3, 16, 8))
    np.savez_compressed(out / f"score2d_arrays{tag}.npz", records=np.asarray(KEYS),
                        queries=np.asarray(["last", "mean", "first"]), lift=lift,
                        peak_on_target=lift, target_mass=lift, eligible=np.ones((3, 16, 8), bool))


def _world(tmp_path):
    root = tmp_path / "T51"
    _fake_score2d(root)
    _fake_score2d(root, "__scene")
    ranked = [A, B, C]
    sids = set(ranked)
    for k in (2, 3):
        sids |= {pr.combo_id([pr.parse_cell(c) for c in t]) for t in itertools.permutations(ranked, k)}
    for key in list(KEYS) + list(rc.HELDOUT_KEYS):
        for s in sids:
            _write_rows(root, key, s, _rate(s, key))
    return root


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*.json"))
            if "__scene" not in p.name}


def test_record_default_pinned_and_scene_flips_the_winner(tmp_path):
    root = _world(tmp_path)
    assert st.build_parser().parse_args(["--stage", "score3d"]).aggregate == "record"
    # ---- default (record mean): the original estimator and file names
    rec3 = st.aggregate3d(root, "gt", _args(stage="score3d"))
    assert rec3["top1"] == A and "aggregate" not in rec3
    assert [r["cell"] for r in rec3["ranking"]] == [A, B, C]
    assert rec3["ranking"][0]["gt_identity"] == pytest.approx(4 / 6)
    assert rec3["cells"][B]["gt_identity_scene_mean"] == pytest.approx(3.4 / 5)  # reported only
    rec_bs = st.stage_bootstrap(_args(label="gt", n_boot=200), root)
    assert rec_bs["full_sample_top1"] == A and "aggregate" not in rec_bs
    rec_ks = st.stage_ksel(_args(stage="ksel", label="gt", n_boot=200, no_run=True), root)
    assert rec_ks["ranking_all_calibration_top10"][:2] == [A, B]
    rec_an = st.stage_annotate(_args(stage="annotate", n_boot=200, annotate_frames=[5]), root)
    assert rec_an["reference_top3"][0] == A
    names = {"gt/score3d/score3d.json", "gt/bootstrap/bootstrap.json", "gt/ksel/ksel.json",
             "gt/annotate/annotate.json", "gt/score2d/score2d.json"}
    before = _snapshot(root)
    assert set(before) == names
    # ---- scene mean: B, written to __scene, record files untouched
    sc3 = st.aggregate3d(root, "gt", _args(stage="score3d", aggregate="scene"))
    assert sc3["top1"] == B and sc3["aggregate"] == "scene"
    assert [r["cell"] for r in sc3["ranking"]] == [B, A, C]
    assert sc3["ranking"][0]["gt_identity"] == pytest.approx(3.4 / 5)
    assert sc3["ranking"][0]["gt_identity_record_mean"] == pytest.approx(3.8 / 6)
    sc_bs = st.stage_bootstrap(_args(label="gt", n_boot=200, aggregate="scene"), root)
    assert sc_bs["full_sample_top1"] == B and sc_bs["aggregate"] == "scene"
    top = {r["cell"]: r for r in sc_bs["all"]}
    assert top[B]["mean"] == pytest.approx(3.4 / 5) and top[A]["mean"] == pytest.approx(0.6)
    assert top[B]["exact_p_rank1"] > top[A]["exact_p_rank1"]
    sc_ks = st.stage_ksel(_args(stage="ksel", label="gt", n_boot=200, no_run=True,
                                aggregate="scene"), root)
    assert sc_ks["ranking_all_calibration_top10"][:2] == [B, A]
    assert B in sc_ks["selected"]["cells"] and sc_ks["aggregate"] == "scene"
    # fold rankers: leaving scene1807 out keeps scene1800's double weight under record, not scene
    js, arrs = st.load_score2d(root, "gt", "__scene")
    fold = tuple(k for k in KEYS if k != "ep1807")
    assert st.make_rank_fn(root, "gt", js, arrs, "scene")(fold)[0] == B
    assert st.make_rank_fn(root, "gt", js, arrs, "record")(fold)[0] == A
    sc_an = st.stage_annotate(_args(stage="annotate", n_boot=200, annotate_frames=[5],
                                    aggregate="scene"), root)
    assert sc_an["reference_top3"][0] == B
    for rel in ("gt/score3d/score3d__scene.json", "gt/bootstrap/bootstrap__scene.json",
                "gt/ksel/ksel__scene.json", "gt/annotate/annotate__scene.json",
                "gt/score3d/score3d_arrays__scene.npz"):
        assert (root / rel).is_file(), rel
    assert _snapshot(root) == before  # record outputs byte-identical
    # and the record outputs, recomputed after the scene run, are the same as before
    again = st.aggregate3d(root, "gt", _args(stage="score3d"))
    assert again["ranking"] == rec3["ranking"]


def test_scene_aggregate_needs_scene_unit_and_scene_inputs(tmp_path):
    ap = st.build_parser()
    with pytest.raises(SystemExit):
        st.validate_args(ap, ap.parse_args(["--stage", "bootstrap", "--aggregate", "scene",
                                            "--unit", "record"]))
    with pytest.raises(SystemExit):
        ap.parse_args(["--stage", "score3d", "--aggregate", "task"])
    st.validate_args(ap, ap.parse_args(["--stage", "bootstrap", "--aggregate", "scene"]))
    root = tmp_path / "T51"
    _fake_score2d(root)  # record score2d only
    with pytest.raises(SystemExit, match="aggregate scene"):
        st.stage_ksel(_args(stage="ksel", label="gt", no_run=True, aggregate="scene"), root)
    with pytest.raises(SystemExit):
        st.stage_bootstrap(_args(label="gt", unit="record", aggregate="scene"), root)


def test_score2d_preselection_uses_scene_mean(tmp_path, monkeypatch):
    """2D preselection (global and the CV folds): with --preselect-top 1, A tops the record mean and B
    the scene mean of the 2D lift."""
    root = tmp_path / "T51"
    root.mkdir()
    ids = [c.id for c in pr.omega(["last"])]
    monkeypatch.setattr(st, "load_filter", lambda r: {"queries": ["last"], "eligible": ids})
    monkeypatch.setattr(st, "t47_dir", lambda a: tmp_path)
    monkeypatch.setattr(st, "t47_cells", lambda p: {"oracle3": [], "paper3": []})
    monkeypatch.setattr(st, "label_data", lambda *a, **k: (None, np.ones((1, 1), bool)))
    monkeypatch.setattr(st, "query_blocks", lambda key, qs, t49=None: {"last": np.zeros((1, 18, 8))})
    keys_iter = iter([])

    def scores(block, cov, keep):
        key = next(keys_iter)
        lift = np.zeros((16, 8))
        for cid in (A, B):
            c = pr.parse_cell(cid)
            lift[pr.LAYERS.index(c.layer), c.head] = RATE[cid][key] + 1.0
        return {"lift": lift, "peak_on_target": np.zeros((16, 8)), "target_mass": np.zeros((16, 8))}

    monkeypatch.setattr(pr, "record_2d_scores", scores)
    out = {}
    for mode in ("record", "scene"):
        keys_iter = iter(KEYS)
        out[mode] = st.stage_score2d(_args(stage="score2d", label="gt", preselect_top=1,
                                           aggregate=mode), root)
    assert out["record"]["global"]["top"] == [A] and out["scene"]["global"]["top"] == [B]
    assert out["record"]["ranking_2d_top30"][0]["lift"] == pytest.approx(1 + 4 / 6)
    assert out["scene"]["ranking_2d_top30"][0]["lift"] == pytest.approx(1 + 3.4 / 5)
    assert "aggregate" not in out["record"] and out["scene"]["aggregate"] == "scene"
    assert "record mean" in out["record"]["rule"] and "scene mean" in out["scene"]["rule"]
    # a fold without scene1807: still B under scene (scene1800 counts once)
    fold = st.subset_key(tuple(k for k in KEYS if k != "ep1807"))
    assert out["scene"]["folds"][fold]["top"] == [B] and out["record"]["folds"][fold]["top"] == [A]
    assert (root / "gt" / "score2d" / "score2d.json").is_file()
    assert (root / "gt" / "score2d" / "score2d__scene.json").is_file()
    assert (root / "gt" / "score2d" / "score2d_arrays__scene.npz").is_file()


def test_heldout_summary_scene_mean(tmp_path):
    root = tmp_path / "T51"
    rates = {"headfix": 1.0, "ep1800": 1.0, "ep1850": 0.5, "ep1900": 0.5}
    for k, r in rates.items():
        _write_rows(root, k, A, r)
    keys = list(rc.HELDOUT_KEYS)
    rec = st.heldout_summary(root, keys, [A])[A]
    sc = st.heldout_summary(root, keys, [A], "scene")[A]
    assert rec["record_mean_gt_identity"] == pytest.approx(0.75) and "scene_mean_gt_identity" not in rec
    assert sc["record_mean_gt_identity"] == pytest.approx(0.75)
    assert sc["scene_mean_gt_identity"] == pytest.approx(2 / 3)  # scene1800 once
    assert sc["by_scene_id"]["scene1800"]["records"] == ["headfix", "ep1800"]
    assert st.point_gt_identity(sc, "scene") == pytest.approx(2 / 3)
    assert st.point_gt_identity(sc, "record") == pytest.approx(0.75)


# ---------------------------------------------------------------------------------- T51b


def _ksel(path: pathlib.Path, combo):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"selected": {"k": 1, "combo": combo, "cells": [combo]},
                                "nested_mean": 0.5}))


def test_t51b_heldout_compare_and_summary_scene_files(tmp_path, monkeypatch):
    noun = st.NOUN_QUERY
    base = tmp_path / "T51b"
    cfg = tb.Config(omega=None, t49_dir=str(tmp_path / "T49"), noun_extra_dir=str(base / "ne"),
                    t51_dir=str(tmp_path / "T51"), rows_dir=str(base / "rows"))
    # record selections and scene selections differ: each run must read its own
    _ksel(tmp_path / "T51" / "gt" / "ksel" / "ksel.json", A)
    _ksel(tmp_path / "T51" / "gt" / "ksel" / "ksel__scene.json", B)
    _ksel(base / "noun" / "gt" / "ksel" / "ksel__scene.json", f"{noun}:L8H5")
    keys = list(rc.HELDOUT_KEYS)
    vals = {"headfix": 1.0, "ep1800": 1.0, "ep1850": 0.5, "ep1900": 0.5}
    with tb.installed(cfg):
        for k in keys:
            for s in (A, B, f"{noun}:L8H5", st.SERVING.id):
                _write_rows(tmp_path, k, s, vals[k])  # rows_path is rebound to T51b's cache
        monkeypatch.setattr(rc, "check_pairing", lambda k: {"key": k})
        monkeypatch.setattr(rc, "n_frames", lambda k: N_FRAMES)
        monkeypatch.setattr(st, "t47_dir", lambda args: tmp_path)
        with pytest.raises(SystemExit, match="noun_best_gt"):  # no record-mode noun selection
            tb.stage_heldout_compare(_args(tb, stage="heldout_compare", labels=["gt"]),
                                     base / "heldout_compare")
        out = tb.stage_heldout_compare(_args(tb, stage="heldout_compare", labels=["gt"],
                                             aggregate="scene"), base / "heldout_compare")
        assert out["named"]["action_best_gt"] == B and out["aggregate"] == "scene"
        assert out["scene_mean_gt_identity"]["action_best_gt"] == pytest.approx(2 / 3)
        assert out["record_mean_gt_identity"]["action_best_gt"] == pytest.approx(0.75)
        assert (base / "heldout_compare" / "heldout_compare__scene.json").is_file()
        assert not (base / "heldout_compare" / "heldout_compare.json").exists()
        summ = tb.stage_summary(_args(tb, stage="summary", labels=["gt"], aggregate="scene"),
                                base / "summary")
    assert summ["omegas"]["action_t51"]["gt"]["ksel"]["selected"]["combo"] == B
    assert summ["heldout_compare"]["scene_mean_gt_identity"]["noun_best_gt"] == pytest.approx(2 / 3)
    assert (base / "summary" / "summary__scene.json").is_file()
    assert not (base / "summary" / "summary.json").exists()
    assert tb.build_parser().parse_args(["--stage", "summary"]).aggregate == "record"


# -------------------------------------------------------------------------------- LIBERO


def test_libero_task_aggregate(tmp_path):
    names = [f"libero_10_task{t}_ep{e}.npz" for t in range(3) for e in range(2)]
    lift = np.zeros((6, 3, 16, 8))
    P, Q = (0, 0, 0), (0, 0, 1)  # (query, layer index, head): last:L2H0, last:L2H1
    lift[(slice(None), *P)] = [1.0, np.nan, 0.2, 0.2, 0.2, 0.2]  # episode .36, task .467
    lift[(slice(None), *Q)] = 0.4  # .4 both ways
    for tag in ("", "__task"):
        np.savez_compressed(tmp_path / f"score2d_arrays{tag}.npz", episodes=np.asarray(names),
                            queries=np.asarray(["last", "mean", "first"]), lift=lift,
                            peak_on_target=np.zeros_like(lift), knows_hit=lift,
                            eligible=np.ones((3, 16, 8), bool))
    a = lb.build_parser().parse_args(["--stage", "bootstrap"])
    assert a.aggregate == "episode" and a.unit == "task"
    a.n_boot = 200
    ep = lb.stage_bootstrap(a, tmp_path)
    assert ep["full_sample_top1"] == "last:L2H1" and "aggregate" not in ep
    a.aggregate = "task"
    tk = lb.stage_bootstrap(a, tmp_path)
    assert tk["full_sample_top1"] == "last:L2H0" and tk["aggregate"] == "task"
    assert tk["top10"][0]["cell"] in ("last:L2H0", "last:L2H1")
    m = {r["cell"]: r["mean"] for r in tk["top10"]}
    assert m["last:L2H0"] == pytest.approx((1.0 + 0.2 + 0.2) / 3)
    assert (tmp_path / "bootstrap.json").is_file() and (tmp_path / "bootstrap__task.json").is_file()
    a.unit = "episode"
    with pytest.raises(SystemExit):
        lb.stage_bootstrap(a, tmp_path)
