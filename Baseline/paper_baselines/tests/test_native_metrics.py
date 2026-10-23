"""native_metrics (BASELINE-EVAL F): grounding-native 2D metrics against hand cases and the pinned official code.

Parity tests load the official evaluation code from the pinned clones (skipped when the clone or its deps are absent):
``external/Grounded-SAM-2/sav_dataset/utils/sav_benchmark.py`` (J / F) and
``external/TokenCut/unsupervised_saliency_detection/metric.py`` (maxFbeta / IoU / Acc).
"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from Baseline.paper_baselines.eval import native_metrics as N

EXTERNAL = pathlib.Path(__file__).resolve().parents[1] / "external"


def _load(path: pathlib.Path, name: str):
    if not path.exists():
        pytest.skip(f"pinned clone missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:  # cv2 / skimage / torch missing in this interpreter
        pytest.skip(f"{name}: {e}")
    return mod


def _rand_masks(rng, n=12, hw=(64, 80)):
    out = []
    for _ in range(n):
        m = np.zeros(hw, bool)
        for _ in range(rng.integers(0, 3)):
            y0, x0 = rng.integers(0, hw[0] - 8), rng.integers(0, hw[1] - 8)
            m[y0:y0 + rng.integers(2, 30), x0:x0 + rng.integers(2, 30)] = True
        out.append(m)
    return out


# --- boxes / CorLoc / pointing ------------------------------------------------------------------------------------
def test_mask_box_and_iou():
    m = np.zeros((10, 10), bool)
    m[2:5, 3:7] = True
    np.testing.assert_array_equal(N.mask_box(m), [3, 2, 7, 5])  # pixel edges
    assert N.mask_box(np.zeros((4, 4), bool)) is None
    assert N.box_iou([0, 0, 2, 2], [1, 1, 3, 3]) == pytest.approx(1 / 7)
    assert N.box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    assert N.best_box_iou(None, [[0, 0, 1, 1]]) == 0.0
    assert N.best_box_iou([0, 0, 1, 1], [None]) is None
    assert N.best_box_iou([0, 0, 2, 2], [[5, 5, 6, 6], [0, 0, 2, 2]]) == 1.0  # CorLoc: any GT box


def test_pointing_game_tolerance():
    g = np.zeros((40, 40), bool)
    g[10:20, 10:20] = True
    assert N.pointing_hit([15, 15], g, 0) is True
    assert N.pointing_hit([25, 15], g, 0) is False
    assert N.pointing_hit([25, 15], g, 6) is True  # 6 px right of column 19
    assert N.pointing_hit([25, 15], g, 5) is False
    assert N.pointing_hit(None, g, 15) is False  # no point -> miss
    assert N.pointing_hit([1, 1], np.zeros((4, 4), bool), 15) is None  # no GT -> not a sample


# --- saliency (TokenCut metric.py) -----------------------------------------------------------------------------------
def test_fbeta_conventions():
    g = np.zeros((8, 8), bool)
    g[:4] = True
    p = np.zeros((8, 8), bool)
    assert N.precision_recall_tokencut(p, g) == (1.0, 0.0)  # empty prediction -> P = 1
    assert N.precision_recall_tokencut(g, np.zeros_like(g)) == (0.0, 1.0)  # empty GT -> R = 1
    assert float(N.f_beta(1.0, 0.0)) == 0.0
    P, R = 0.5, 1.0
    assert float(N.f_beta(P, R)) == pytest.approx(1.3 * 0.5 / (0.3 * 0.5 + 1.0))


def test_tokencut_metric_parity():
    torch = pytest.importorskip("torch")
    off = _load(EXTERNAL / "TokenCut/unsupervised_saliency_detection/metric.py", "_tc_metric")
    rng = np.random.default_rng(0)
    gts = [m for m in _rand_masks(rng, 10) if m.any()]
    preds = [np.clip(rng.random(g.shape) * 0.6 + 0.5 * g, 0, 1).astype(np.float32) for g in gts]
    ref = off.metrics(preds, [g.astype(np.float32) for g in gts])
    P, R = zip(*[N.threshold_pr(p, g) for p, g in zip(preds, gts)])
    assert N.max_f_beta(np.array(P), np.array(R)) == pytest.approx(ref["F_max"], abs=1e-6)
    assert np.mean([N.pixel_accuracy(p > 0.5, g) for p, g in zip(preds, gts)]) == pytest.approx(ref["accuracy"], abs=1e-6)
    ious = [N.inter_union(p > 0.5, g) for p, g in zip(preds, gts)]
    assert np.mean([i / u for i, u in ious]) == pytest.approx(ref["IoU"], abs=1e-6)
    del torch


def test_binary_map_maxf_equals_mean_f():
    rng = np.random.default_rng(1)
    gts = [m for m in _rand_masks(rng, 8) if m.any()]
    preds = [g ^ (rng.random(g.shape) < 0.05) for g in gts]
    P, R = zip(*[N.threshold_pr(p.astype(float), g) for p, g in zip(preds, gts)])
    mean_f = np.mean([float(N.f_beta(*N.precision_recall_tokencut(p, g))) for p, g in zip(preds, gts)])
    assert N.max_f_beta(np.array(P), np.array(R)) == pytest.approx(mean_f)


# --- DAVIS J / F (SAM 2 sav_benchmark.py) ----------------------------------------------------------------------------
def test_davis_edge_cases():
    z = np.zeros((16, 16), bool)
    m = z.copy()
    m[4:8, 4:8] = True
    assert N.davis_j(z, z) == 1.0 and N.davis_f(z, z) == 1.0
    assert N.davis_j(m, z) == 0.0 and N.davis_f(m, z) == 0.0
    assert N.davis_j(m, m) == 1.0 and N.davis_f(m, m) == 1.0
    np.testing.assert_array_equal(N.disk(2), [[0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0]])


def test_davis_parity_with_sam2_benchmark():
    off = _load(EXTERNAL / "Grounded-SAM-2/sav_dataset/utils/sav_benchmark.py", "_sav_benchmark")
    rng = np.random.default_rng(2)
    preds, gts = _rand_masks(rng, 15, (96, 128)), _rand_masks(rng, 15, (96, 128))
    ev = off.Evaluator()
    for p, g in zip(preds, gts):
        ev.feed_frame(p.astype(np.uint8), g.astype(np.uint8))
        np.testing.assert_array_equal(N.seg2bmap(p), off._seg2bmap(p.copy()).astype(bool))
    J = [N.davis_j(p, g) for p, g in zip(preds, gts)]
    F = [N.davis_f(p, g) for p, g in zip(preds, gts)]
    np.testing.assert_allclose(J, ev.object_iou[1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(F, ev.boundary_f[1], rtol=0, atol=1e-12)


def test_db_statistics():
    v = np.array([1.0, 1.0, 0.9, 0.8, 0.4, 0.3, 0.2, 0.1])
    M, O, D = N.db_statistics(v)
    assert M == pytest.approx(v.mean())
    assert O == pytest.approx(4 / 8)
    # davis2017 bins: ids = round(linspace(1, 8, 5)) - 1 = [0, 2, 4, 5, 7]; bins [0:3], [5:8]
    assert D == pytest.approx(np.mean(v[0:3]) - np.mean(v[5:8]))


# --- aggregation ---------------------------------------------------------------------------------------------------------
def test_bootstrap_ratio_pooled():
    r = N.bootstrap_ratio({"a": 2.0, "b": 0.0}, {"a": 2.0, "b": 2.0})
    assert r["value"] == 0.5 and r["n_episodes"] == 2
    assert 0.0 <= r["ci95"][0] <= 0.5 <= r["ci95"][1] <= 1.0
    assert N.bootstrap_ratio({}, {})["value"] is None


def test_sample_row_and_summary():
    g = np.zeros((32, 32), bool)
    g[8:16, 8:16] = True
    other = np.zeros_like(g)
    other[20:28, 20:28] = True
    objs = {"target": g, "bowl_2": other}
    rows = []
    for ep, pred in (("e0", g.copy()), ("e1", other.copy())):  # e0 correct, e1 grabs the distractor
        r = N.sample_row(pred, g, objs, det_box=N.mask_box(pred), point=[12, 12] if ep == "e0" else [24, 24], pointing=True,
                         box_centre_eval=True)
        r.update(arm="x|phrase", episode_id=ep, obs_index=0, camera="head")
        rows.append(r)
    s = N.summarize_samples(rows)
    assert s["acc05_mask"]["value"] == 0.5 and s["acc05_det"]["value"] == 0.5
    assert s["corloc_any_object"]["value"] == 1.0  # the distractor is an object too (class-agnostic CorLoc)
    assert s["pointing_tol0"]["value"] == 0.5 and s["pointing_box_centre_tol0"]["value"] == 0.5
    assert s["mIoU"]["value"] == 0.5 and s["oIoU"]["value"] == pytest.approx(64 / (64 + 128))
    seq = N.summarize_sequences(rows)
    assert seq["J_mean"] == pytest.approx(0.5) and seq["n_sequences"] == 2


def test_continuous_map_row():
    g = np.zeros((16, 16), bool)
    g[4:8, 4:8] = True
    score = np.zeros((16, 16))
    score[5, 6] = 1.0
    r = N.sample_row(None, g, {"t": g}, score=score)
    assert r["point_xy"] == [6.0, 5.0] and r["point_hit_tol0"] == 1.0
    assert len(r["score_P"]) == N.FMAX_BINS


def test_conf_sweep():
    rows = [{"gt_visible": True, "det_score": 0.9, "hit05_det": 1.0, "any_candidate_hit": 1.0},
            {"gt_visible": True, "det_score": 0.4, "hit05_det": 0.0, "any_candidate_hit": 1.0},
            {"gt_visible": True, "det_score": None, "hit05_det": 0.0, "any_candidate_hit": 0.0}]
    cs = N.conf_sweep(rows, thresholds=(0.0, 0.5, 0.95))
    assert cs["precision"] == [0.5, 1.0, None] and cs["recall"] == pytest.approx([1 / 3, 1 / 3, 0.0])
    assert cs["any_candidate_recall"] == pytest.approx(2 / 3)


def test_arm_dir_and_native_point():
    assert N.arm_dir_name("grounded_sam2:per_frame|phrase") == "grounded_sam2-per_frame__phrase"
    assert N.arm_dir_name("gt_oracle|instruction") == "gt_oracle__instruction"
    p, src = N.native_point("tokencut:foreground|no_text", {"extras": {"seed": 17, "feature_grid": [16, 16]}})
    assert src == "tokencut_seed" and p == [16 + 7.5, 16 + 7.5]  # row 1, col 1 -> centre of patch (1, 1)
    p, src = N.native_point("localization_heads:threshold|no_text", {"extras": {"peak_xy": [3, 4]}})
    assert p == [3.0, 4.0] and src == "lh_peak"
    assert N.native_point("yoloe|phrase", {}) == (None, None)


def test_load_mask_npz_roundtrip(tmp_path):
    m = np.random.default_rng(3).random((5, 7)) > 0.5
    with open(tmp_path / "x.npz", "wb") as f:
        np.savez_compressed(f, head=np.packbits(m, axis=None), head__shape=np.asarray(m.shape))
    np.testing.assert_array_equal(N.load_mask_npz(tmp_path / "x.npz")["head"], m)


# --- recorded data (skipped when the SL3 grounding run is not reachable) ------------------------------------------------
SL3_GROUNDING = "20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack"


def _grounding_run():
    try:
        return N.resolve_run(SL3_GROUNDING)
    except Exception:
        pytest.skip("SL3 grounding run not reachable")


def test_recorded_episode_matches_harness_2d_iou(tmp_path, monkeypatch):
    run = _grounding_run()
    monkeypatch.setattr(N, "PROJECT_ROOT", tmp_path)  # --out-dir must be below <root>/outputs
    out = N.run(N.build_parser().parse_args([
        "--grounding-run", str(run), "--episodes", "libero_spatial_t0_e0", "--limit-frames", "2", "--no-maps",
        "--out-dir", str(tmp_path / "outputs/native")]))
    s = json.loads((out / "summary.json").read_text())
    chk = s["check_vs_harness_2d_iou"]
    assert chk["n"] > 0 and chk["max_abs_diff"] < 1e-12  # same masks + same GT -> the harness's own 2D IoU
    g = s["arms"]["gt_oracle|phrase"]["all"]
    assert g["mIoU"]["value"] == 1.0 and g["acc05_mask"]["value"] == 1.0 and g["davis"]["JF_mean"] == 1.0
