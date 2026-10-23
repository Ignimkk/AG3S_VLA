"""metrics: toy soft-IoU cases, 2D metrics, aggregation and bootstrap reproducibility."""
import numpy as np
import pytest

from Baseline.paper_baselines.eval import metrics as M

TGT, BG, OBJ = M.CLASSES.index("target"), M.CLASSES.index("background"), M.CLASSES.index("other_object")


def _grid():
    """Camera looking down +z from the origin with identity pose: pixel (u, v) at depth d lands at ((u-cx) d/f, ...)."""
    K = np.array([[1000.0, 0, 1.5], [0, 1000.0, 1.5], [0, 0, 1]])
    T = np.eye(4)
    depth = np.full((4, 4), 1.0)
    labels = np.full((4, 4), BG)
    labels[0, :] = TGT
    labels[1, 0] = OBJ
    return M.build_eval_grid([{"depth_m": depth, "K": K, "T_base_cam": T, "labels": labels}]), K, T, depth, labels


def test_mask_metrics_2d():
    gt = np.zeros((4, 4), bool); gt[0] = True
    pred = np.zeros((4, 4), bool); pred[0, :2] = True; pred[3, 3] = True
    m = M.mask_metrics_2d(pred, gt)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 2)
    assert m["iou"] == pytest.approx(2 / 5) and m["precision"] == pytest.approx(2 / 3) and m["recall"] == pytest.approx(0.5)
    e = M.mask_metrics_2d(np.zeros((4, 4), bool), np.zeros((4, 4), bool))
    assert e["iou"] is None and e["precision"] is None and e["recall"] is None


def test_soft_voxel_fractions_and_metrics():
    grid, K, T, depth, labels = _grid()
    # pixels are 1 mm apart at 1 m -> all 16 pixels in a handful of 5 mm voxels; check fractions sum to 1
    np.testing.assert_allclose(grid.class_frac.sum(1), 1.0)
    # each voxel's target fraction = share of its pixels labelled target
    vox = grid.pixel_voxels(_pts(K, T, depth))
    lab = labels.ravel()
    for v in range(len(grid.packed)):
        assert grid.target[v] == pytest.approx((lab[vox == v] == TGT).mean())
    # predict exactly the voxels containing target pixels
    from Baseline.paper_baselines.lift import lift_mask, voxel_keys

    pred = voxel_keys(lift_mask(labels == TGT, depth, K, T))
    r = M.voxel_metrics(grid, pred)
    t = grid.target
    sel = grid.lookup(pred)
    assert r["TP"] == pytest.approx(t[sel].sum()) and r["FN"] == pytest.approx(t.sum() - t[sel].sum())
    assert r["iou"] == pytest.approx(r["TP"] / (r["TP"] + r["FP"] + r["FN"]))
    assert r["recall"] == pytest.approx(1.0)
    # empty prediction -> precision None, iou 0
    e = M.voxel_metrics(grid, np.zeros((0, 3), np.int64))
    assert e["precision"] is None and e["iou"] == 0.0 and e["pred_voxels"] == 0


def _pts(K, T, depth):
    from Baseline.paper_baselines.lift import backproject_pixels

    p, u, v = backproject_pixels(depth, K, T)
    order = np.argsort(v * depth.shape[1] + u)
    return p[order]


def test_toy_two_voxel_soft_iou_by_hand():
    """Two eval voxels: A (target fraction 0.75), B (0.0, other object). Predict A and B."""
    g = M.EvalGrid(keys=np.array([[0, 0, 0], [1, 0, 0]]), class_frac=np.zeros((2, len(M.CLASSES))), classes=M.CLASSES,
                   kmin=np.zeros(3, np.int64), ext=np.array([2, 1, 1]), packed=np.array([0, 1]), voxel=0.005)
    g.class_frac[0, TGT], g.class_frac[0, BG] = 0.75, 0.25
    g.class_frac[1, OBJ] = 1.0
    r = M.voxel_metrics(g, np.array([[0, 0, 0], [1, 0, 0], [5, 5, 5]]))  # last one off-grid
    assert r["TP"] == pytest.approx(0.75) and r["FP"] == pytest.approx(0.25 + 1.0 + 1) and r["FN"] == pytest.approx(0.0)
    assert r["iou"] == pytest.approx(0.75 / 3.0) and r["pred_offgrid"] == 1
    assert r["neighbor_fp"] == pytest.approx(1.0) and r["background_fp"] == pytest.approx(0.25)
    assert r["hard_iou"] == pytest.approx(1 / 3)
    none = M.voxel_metrics(M.EvalGrid(g.keys, np.zeros_like(g.class_frac), g.classes, g.kmin, g.ext, g.packed, 0.005),
                           np.array([[0, 0, 0]]))
    assert none["iou"] is None and none["missing_reason"] == "target_not_visible"


def test_aggregation_frame_then_episode():
    rows = [{"episode_id": "a", "x": 1.0}, {"episode_id": "a", "x": 0.0}, {"episode_id": "b", "x": 1.0},
            {"episode_id": "b", "x": None}]
    assert M.episode_means(rows, "x") == {"a": 0.5, "b": 1.0}
    agg = M.aggregate(rows, "x")
    assert agg["episode_mean"] == pytest.approx(0.75) and agg["n_episodes"] == 2 and agg["n_frames"] == 3


def test_paired_bootstrap_reproducible_and_matches_dev_rule():
    rng = np.random.default_rng(1)
    a = {f"e{i}": float(x) for i, x in enumerate(rng.random(12))}
    b = {f"e{i}": float(x) for i, x in enumerate(rng.random(12))}
    r1, r2 = M.paired_bootstrap(a, b), M.paired_bootstrap(a, b)
    assert r1 == r2 and r1["n_boot"] == 2000 and r1["seed"] == 20261015
    d = np.array([a[e] - b[e] for e in sorted(a)])
    boot = d[np.random.default_rng(20261015).integers(0, len(d), (2000, len(d)))].mean(1)
    assert r1["ci95"] == [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    assert r1["diff"] == pytest.approx(d.mean())
    assert M.paired_bootstrap({"x": 1.0}, {"y": 1.0}) is None
    assert M.paired_bootstrap(a, b, seed=1) != r1


def test_latency_and_found():
    assert M.latency_stats([1.0, 2.0, 3.0, None])["median_ms"] == 2.0
    assert M.latency_stats([])["p95_ms"] is None
    assert M.found_rate([True, False, True, True])["rate"] == 0.75
