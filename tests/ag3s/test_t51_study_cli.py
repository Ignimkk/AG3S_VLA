"""T51 CLI plumbing (`t51_study`) and the LIBERO pieces (`t51_libero`), without MuJoCo or storage.

Pinned:
  - argument validation (CPU budget <= 2 workers, units, source ids) and the AG3S_RUN_DIR refusal;
  - the nested-CV training subsets the 2D preselection is computed for;
  - record-level metrics from rows (GT on all frames, GT on self-labeled frames, self label);
  - source maps: a ``sum[...]`` is the sum of its cells; the specials are T47's combine / average;
  - stages ``score3d`` aggregation, ``bootstrap`` and ``ksel`` end to end on fake rows in a tmp root
    (no replay: every required row file exists), including the bootstrap seed reproducibility;
  - the L8H2 parity check against T47b-shaped rows; held-out summary;
  - LIBERO: coverage / target set from a synthetic segmentation, the KNOWS hit, grid slicing.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pytest

from AG3S.ag3s.experiments.studies import t47_head_metrics as hm
from AG3S.ag3s.experiments.studies import t51_libero as lb
from AG3S.ag3s.experiments.studies import t51_protocol as pr
from AG3S.ag3s.experiments.studies import t51_records as rc
from AG3S.ag3s.experiments.studies import t51_study as st

KEYS = ("long", "t2", "ep1807")


def _args(**kw) -> argparse.Namespace:
    a = st.build_parser().parse_args(["--stage", kw.pop("stage", "bootstrap")])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_validate_args_rejects():
    ap = st.build_parser()
    assert ap.parse_args(["--stage", "bootstrap"]).unit == "scene"  # pre-registered (판정 보완)
    for argv in (["--stage", "score3d", "--workers", "3"], ["--stage", "bootstrap", "--unit", "frame"],
                 ["--stage", "score3d", "--cells", "L8H2"], ["--stage", "ksel", "--n-boot", "10"]):
        with pytest.raises(SystemExit):
            st.validate_args(ap, ap.parse_args(argv))
    st.validate_args(ap, ap.parse_args(["--stage", "score3d", "--cells", "last:L8H2", "paper3_s1"]))


def test_out_root_refuses_without_run_dir(monkeypatch):
    monkeypatch.delenv("AG3S_RUN_DIR", raising=False)
    with pytest.raises(SystemExit):
        st.out_root()


def test_train_subsets():
    assert st.train_subsets(["a", "b", "c"]) == [("b", "c"), ("c",), ("b",), ("a", "c"), ("a",),
                                                 ("a", "b")]
    assert st.subset_key(("a", "c")) == "a|c"


def _row(frame, ident, *, lab=False, self_ok=None, on=None, has=True):
    return {"frame": frame, "identity_correct": ident, "on_target": ident if on is None else on,
            "has_target": has, "chosen_cluster_purity": 0.8 if ident else 0.2,
            "nontarget_seed_loss": 0 if ident else 10, "nontarget_seed_loss__other_object": 0,
            "self_labeled": lab, "self_identity_correct": self_ok, "self_out_n": 1 if lab else None,
            "phase": "approach", "source": "x", "record": "long"}


def test_record_metrics():
    rows = [_row(0, True, lab=True, self_ok=True), _row(1, False, lab=True, self_ok=True),
            _row(2, False), _row(3, True)]
    m = st.record_metrics(rows)
    assert m["gt_identity"] == 0.5 and m["gt_identity_on_self_frames"] == 0.5
    assert m["self_identity"] == 1.0 and m["n_self_labeled"] == 2
    assert m["nontarget_seed_loss_per_frame"] == 5.0
    assert np.isnan(st.record_metrics([_row(0, True)])["self_identity"])


# ------------------------------------------------------------------------- fake T47 / maps


def _fake_t47(tmp: pathlib.Path) -> pathlib.Path:
    t47 = tmp / "T47"
    (t47 / "select").mkdir(parents=True)
    (t47 / "validate").mkdir(parents=True)
    (t47 / "gt_cache").mkdir(parents=True)
    sel = {"settings": {"d0_last": {"tau_source": {"cam": {"common": {"topk": {"3": [[4, 1], [2, 0], [14, 5]]}}}}}}}
    (t47 / "select" / "selection.json").write_text(json.dumps(sel))
    (t47 / "validate" / "oracle.json").write_text(json.dumps(
        {"settings": {"d0_last": {"top3": [[8, 3], [10, 7], [12, 4]]}}}))
    return t47


def test_source_maps(tmp_path):
    t47 = _fake_t47(tmp_path)
    rng = np.random.default_rng(0)
    blocks = {q: rng.random((2, 18, 8, 3, 16, 16)) for q in ("last", "mean")}
    a = st.source_maps(blocks, "last:L8H2", t47)
    np.testing.assert_array_equal(a, blocks["last"][:, 8, 2])
    s = st.source_maps(blocks, "sum[last:L8H2+mean:L3H1]", t47)
    np.testing.assert_allclose(s, blocks["last"][:, 8, 2] + blocks["mean"][:, 3, 1])
    block_c = np.moveaxis(blocks["last"], 3, 1)
    np.testing.assert_allclose(st.source_maps(blocks, "paper3_s1", t47),
                               hm.combine_maps(block_c, [(4, 1), (2, 0), (14, 5)], 1.0))
    np.testing.assert_allclose(st.source_maps(blocks, "headavg_last", t47), hm.head_average(block_c))
    cells = st.t47_cells(t47)
    assert [c.id for c in cells["oracle3"]] == ["last:L8H3", "last:L10H7", "last:L12H4"]


# --------------------------------------------------------------- stages on fake rows


def _write_rows(root, key, sid, idents, selfs):
    p = st.rows_path(root, key, sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as fh:
        for i, (g, s) in enumerate(zip(idents, selfs)):
            r = _row(i, bool(g), lab=s is not None, self_ok=None if s is None else bool(s))
            r.update(source=sid, record=key)
            fh.write(json.dumps(r) + "\n")


def _fake_score2d(root, label, cand, ranked, rng):
    out = root / label / "score2d"
    out.mkdir(parents=True)
    folds = {st.subset_key(t): {"ranked_set": ranked, "scored_set": cand}
             for t in st.train_subsets(list(KEYS), rc.scene_map(KEYS))}
    js = {"records": list(KEYS), "candidates_3d": cand,
          "global": {"ranked_set": ranked, "scored_set": cand}, "folds": folds}
    (out / "score2d.json").write_text(json.dumps(js))
    np.savez_compressed(out / "score2d_arrays.npz", records=np.asarray(KEYS),
                        queries=np.asarray(["last", "mean", "first"]),
                        lift=rng.random((3, 3, 16, 8)) + 1.0, peak_on_target=rng.random((3, 3, 16, 8)),
                        target_mass=rng.random((3, 3, 16, 8)), eligible=np.ones((3, 16, 8), bool))


def _setup_fake_run(tmp_path, label="gt"):
    rng = np.random.default_rng(1)
    root = tmp_path / "T51"
    cand = ["last:L8H2", "last:L9H1", "last:L10H7", "mean:L12H4", "first:L5H5"]
    ranked = cand[:4]
    _fake_score2d(root, label, cand, ranked, rng)
    p_cell = {"last:L8H2": 0.9, "last:L9H1": 0.6, "last:L10H7": 0.5, "mean:L12H4": 0.4,
              "first:L5H5": 0.1}
    for k in KEYS:
        n = 20
        for c, p in p_cell.items():
            g = rng.random(n) < p
            s = [bool(x) if i < 12 else None for i, x in enumerate(rng.random(n) < p)]
            _write_rows(root, k, c, g, s)
    return root, cand, ranked


def test_aggregate_bootstrap_ksel_end_to_end(tmp_path):
    root, cand, ranked = _setup_fake_run(tmp_path)
    args = _args(stage="bootstrap", label="gt", n_boot=200)
    agg = st.aggregate3d(root, "gt", args)
    assert agg["ranked_set"] == ranked and set(agg["cells"]) == set(cand)
    assert agg["top1"] in ranked and "first:L5H5" not in [r["cell"] for r in agg["ranking"]]
    b1 = st.stage_bootstrap(_args(stage="bootstrap", label="gt", n_boot=200, unit="record"), root)
    b2 = st.stage_bootstrap(_args(stage="bootstrap", label="gt", n_boot=200, unit="record"), root)
    assert b1["top10_by_p_rank1"] == b2["top10_by_p_rank1"]
    assert b1["n_units"] == 3 and b1["n_distinct_resamples"] == 10
    sc = st.stage_bootstrap(args, root)  # default unit: scene (long + t2 = scene1800)
    assert sc["unit"] == "scene" and sc["units"] == ["scene1800", "scene1807"]
    assert sc["units_records"] == {"scene1800": ["long", "t2"], "scene1807": ["ep1807"]}
    assert sc["n_distinct_resamples"] == 3 and (root / "gt" / "bootstrap" / "bootstrap.json").is_file()
    assert (root / "gt" / "bootstrap" / "bootstrap__record.json").is_file()
    assert sum(r["p_rank1"] for r in b1["all"]) == pytest.approx(1.0)
    assert sum(r["exact_p_rank1"] for r in b1["all"]) == pytest.approx(1.0)
    # ksel: write every (combo, record) row the nested CV needs, so nothing replays
    js, arrs = st.load_score2d(root, "gt")
    rank_fn = st.make_rank_fn(root, "gt", js, arrs)
    rng = np.random.default_rng(2)
    scenes, rank_s, _, records_of = pr.grouped_cv(list(KEYS), rc.scene_map(KEYS), rank_fn,
                                                  lambda c, u: 0.0)
    for cid, k in pr.expand_pairs(pr.required_pairs(scenes, pr.K_CHOICES, rank_s), records_of):
        if st.load_rows(root, k, cid) is None:
            _write_rows(root, k, cid, rng.random(20) < 0.7, [None] * 20)
    ks = st.stage_ksel(_args(stage="ksel", label="gt", n_boot=200, no_run=True), root)
    assert ks["selected"]["k"] in (1, 2, 3)
    assert ks["selected"]["combo"] == pr.topk(rank_fn, KEYS, ks["selected"]["k"])
    assert ks["cv_unit"] == "scene" and ks["units"] == ["scene1800", "scene1807"]
    for o in ks["nested"]:
        assert o["test"] not in o["train"]
    assert set(ks["cv_mean_ci95"]) == {1, 2, 3}
    assert ks["provenance"]["vla_training_episodes"] == []
    blk = st.stage_bootstrap(_args(stage="bootstrap", label="gt", n_boot=200, unit="block:5"), root)
    assert blk["n_units"] == 12  # 3 records x 20 frames / 5
    ann = st.stage_annotate(_args(stage="annotate", n_boot=200, annotate_frames=[5, 20]), root)
    assert [c["n_frames"] for c in ann["curve"]] == [5, 20, 60]
    assert ann["curve"][-1]["p_same_top1"] == 1.0
    assert ann["reference_top3"][0] == agg["top1"]


def test_ksel_refuses_missing_rows_with_no_run(tmp_path):
    root, _, _ = _setup_fake_run(tmp_path)
    with pytest.raises(SystemExit):
        st.stage_ksel(_args(stage="ksel", label="gt", n_boot=200, no_run=True), root)


def test_self_label_aggregation(tmp_path):
    root, cand, ranked = _setup_fake_run(tmp_path, label="self")
    agg = st.aggregate3d(root, "self", _args(stage="score3d"))
    assert agg["primary"] == "self_identity"
    c = agg["cells"]["last:L8H2"]
    assert np.isfinite(c["self_identity_mean"]) and np.isfinite(c["gt_identity_on_self_frames_mean"])


def test_parity_and_heldout_summary(tmp_path):
    root = tmp_path / "T51"
    t47b = tmp_path / "T47b"
    (t47b / "e2e").mkdir(parents=True)
    _write_rows(root, "ep1850", "last:L8H2", [1, 0, 1], [None] * 3)
    mine = st.load_rows(root, "ep1850", "last:L8H2")
    with (t47b / "e2e" / "rows__ep1850__s0.jsonl").open("w") as fh:
        for r in mine:
            t = dict(r, source="l8h2")
            if r["frame"] == 1:
                t["identity_correct"] = True
            fh.write(json.dumps(t) + "\n")
    par = st.parity_vs_t47b(root, t47b, ["ep1850", "ep1900"])
    assert par["ep1850"]["identical"] == 2
    assert par["ep1850"]["differing"] == [{"frame": 1, "fields": ["identity_correct"]}]
    assert par["ep1900"] == {"available": False}
    summ = st.heldout_summary(root, ["ep1850"], ["last:L8H2"])
    s = summ["last:L8H2"]
    assert s["record_mean_gt_identity"] == pytest.approx(2 / 3)
    assert s["by_scene"]["new_scene"]["records"] == ["ep1850"]


# ------------------------------------------------------------------------------- LIBERO


def _fake_episode():
    seg = np.zeros((6, 256, 256), np.int16)
    seg[:, 0:32, 0:32] = 1  # bowl (manipulated)
    seg[:, 128:192, 128:192] = 2  # plate (destination)
    seg[:, 200:256, 0:40] = 3  # robot
    action = np.zeros((6, 7))
    action[3:, 6] = 1.0  # gripper closes at raw step 3
    return {"id2name": np.asarray(["0:__background__", "1:akita_black_bowl_1", "2:plate_1",
                                   "3:PandaGripper0"]),
            "manipulated": np.asarray(["akita_black_bowl_1"]),
            "destinations": np.asarray(["plate_1_contain_region"]),
            "t": np.arange(6), "action": action, "seg_full": seg}


def test_libero_episode_gt():
    g = lb.episode_gt(_fake_episode(), stride=1)
    assert g["names"] == ["akita_black_bowl_1", "plate_1"]
    assert g["reach"] == ["akita_black_bowl_1"] and g["place"] == ["plate_1"]
    assert g["grasp_t"] == 3
    assert g["target_mask"][:3, 0].all() and g["target_mask"][3:, 1].all()
    # 256 -> 224 nearest: the bowl's 32x32 block covers 28x28 px = 4 patches of 14x14
    assert g["area"][0, 0] == 28 * 28
    assert g["target_cov"][0].sum() == pytest.approx(4.0)
    assert g["target_cov"][5].sum() == pytest.approx(56 * 56 / 196)


def test_knows_hits():
    mass = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    area = np.ones((3, 2))
    tm = np.array([[True, False], [True, False], [False, True]])
    assert lb.knows_hits(mass, area, tm, window=1).tolist() == [True, False, True]
    # window 5: step 1 density (1, 1) ties -> argmax picks candidate 0 -> hit
    assert lb.knows_hits(mass, area, tm, window=5).tolist() == [True, True, True]
    assert lb.knows_hits(mass, area, tm, window=5, perm=np.array([1, 0])).tolist() == [True, False, True]


def test_blocks_to_grids_slices_cameras():
    rng = np.random.default_rng(0)
    blk = rng.random((18, 8, 10, 768)).astype(np.float32)
    g = lb.blocks_to_grids({0: blk}, (0,), ("mean", "last"))
    assert g.shape == (1, 2, 18, 8, 2, 16, 16)
    np.testing.assert_allclose(g[0, 0, 3, 2, 1].reshape(-1), blk[3, 2].mean(axis=0)[256:512], rtol=1e-6)
    np.testing.assert_allclose(g[0, 1, 3, 2, 0].reshape(-1), blk[3, 2, -1, :256])
