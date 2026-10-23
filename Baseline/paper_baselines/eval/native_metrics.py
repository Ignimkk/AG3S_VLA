"""Grounding-native 2D metrics of every paper baseline, computed post hoc from the masks an ``offline_grounding`` run
saved (``--save-masks``) -- BASELINE-EVAL (Implementer F). The harness is not re-run and not modified.

Each baseline paper scores its grounding with its own metric; this module computes those metrics on the same
recorded LIBERO frames, per arm (``provider[:mode]|prompt``) and camera (definitions + sources:
``Baseline/paper_baselines/docs/metrics.md``):

* **box Acc@0.5** (REC, Localization heads [14] Sec. 6.1) -- a sample is correct if the predicted box has IoU >= 0.5
  with the GT target box. Two predicted boxes: ``mask`` = tight pixel-edge box of the saved mask (every arm) and
  ``det`` = the detector's selected box from ``frames.jsonl`` (Grounded-SAM / -2 / YOLOE). GT box = tight box of the
  GT target segmentation (visible part; no amodal box exists in the recording);
* **CorLoc** (TokenCut [26] Sec. 4.1, ``main_tokencut.py:295-298``) -- the predicted (mask) box has IoU >= 0.5 with
  *any* GT object box of the image: ``corloc_any`` uses every BDDL object instance (target, destination, other
  objects; class-agnostic like VOC/COCO object discovery), ``corloc_target`` only the target (= Acc@0.5 mask box);
* **pointing game** (Zhang et al., ECCV 2016 / IJCV 2018, Sec. 4.1) -- the predicted point is a hit if it lies in the
  GT target mask grown by ``tol`` pixels (original 15 px; 0 px also reported). Points: localization heads = peak of
  the combined head map (``extras.peak_xy``), TokenCut = centre of the seed patch, ``v11ab_attention_L8H2`` = peak of
  the V11AB attention cell (recorded audit run), detector arms = centre of the selected box (**variant**: the
  original game scores a saliency peak, a box has none);
* **mIoU / oIoU (= cIoU)** (RES; [14] reports cIoU) -- mean of per-sample IoU / cumulative intersection over
  cumulative union over all samples;
* **maxFbeta, IoU, Acc** (unsupervised saliency, TokenCut [26] Sec. 4.3, ``unsupervised_saliency_detection/
  metric.py``) -- ``F = (1 + b2) P R / (b2 P + R)``, ``b2 = 0.3``; P / R per sample at 255 uniform thresholds
  ``np.arange(0, 1, 1/255)`` of a [0, 1] map, F averaged over samples, max over thresholds; empty prediction ->
  P = 1, empty GT -> R = 1 (as in ``metric.py``). For a binary mask every threshold gives the same mask, so maxFbeta
  is the mean Fbeta of the mask. Continuous maps (min-max normalised): ``lh_combined_map`` (frozen localization
  heads) and ``v11ab_attention_L8H2``;
* **J, F, J&F** (DAVIS; SAM 2 reports J&F) -- per frame ``J`` = IoU with both-empty = 1, ``F`` = boundary F-measure
  with the boundary tolerance ``ceil(0.008 * image diagonal)`` px (``sav_benchmark.py:215-303`` in the pinned
  Grounded-SAM-2 clone, from davis2017-evaluation); statistics per sequence (episode x camera) = mean, recall
  (fraction > 0.5) and decay (first-quartile mean - last-quartile mean, davis2017 ``db_statistics``), then the mean
  over sequences. Every evaluated frame of the sequence counts (also frames where the target is not visible: then
  an empty prediction scores J = F = 1, as in DAVIS). ``temporal_iou_*``: IoU between consecutive evaluated frames
  of the prediction vs of the GT (stability, not a paper metric);
* **detector confidence sweep** (YOLOE / Grounding DINO arms; stands in for the papers' AP, which needs category-
  rich annotation we do not have) -- precision / recall of the top-1 box (IoU >= 0.5) when only predictions with
  score >= tau are kept, tau = 0, 0.05, ..., 0.95 (the provider never output candidates below its own threshold),
  plus ``any_candidate_recall`` (some output box hits the target).

Samples: ``(frame, camera)`` pairs whose GT target mask is non-empty (REC / RES / saliency / pointing); a not-found
arm (empty mask) is a miss / IoU 0. Aggregation: pooled over samples (the papers' dataset-level numbers), with a
95 % percentile CI from 2,000 episode resamples (seed 20261015); per-camera and both-camera rows.

Run (``.venv-baselines``, CPU; reads the grounding run, its source run, scene GT and attention run)::

    source Baseline/paper_baselines/setup/paper_baselines_env.sh
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 .venv-baselines/bin/python -m Baseline.paper_baselines.eval.native_metrics \\
        --grounding-run 20261023T134909_bf1403d54bc6_paper-baselines-2-sl3dev-notrack --run-id baseline-eval-native-sl3dev \\
        --dump-figure-data --tokencut-dino
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
CAMERAS = ("head", "left_wrist")
ACC_IOU = 0.5
POINTING_TOLS_PX = (0, 15)
FBETA_SQ = 0.3
FMAX_BINS = 255
DAVIS_BOUND_TH = 0.008
OBJECT_CLASSES = ("target", "destination", "other_object")
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261015
#: pseudo-arms that score a continuous map instead of a saved mask
MAP_ARMS = ("lh_combined_map", "v11ab_attention_L8H2")
DETECTOR_PROVIDERS = ("grounded_sam", "grounded_sam2", "yoloe")


# =============================================================================================================
# pure metric functions
# =============================================================================================================
def mask_box(mask: np.ndarray) -> np.ndarray | None:
    """Tight pixel-edge box ``[x0, y0, x1 + 1, y1 + 1]`` of a bool mask (TokenCut ``detect_box`` convention)."""
    ys, xs = np.nonzero(np.asarray(mask, bool))
    if not len(xs):
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64)


def box_iou(a, b) -> float:
    """IoU of two ``xyxy`` boxes with continuous areas (TokenCut ``datasets.bbox_iou`` without its 1e-7 eps)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def best_box_iou(pred_box, gt_boxes: Sequence) -> float | None:
    """Max IoU of ``pred_box`` over ``gt_boxes`` (0.0 for no prediction, None for no GT box)."""
    gts = [g for g in gt_boxes if g is not None]
    if not gts:
        return None
    if pred_box is None:
        return 0.0
    return float(max(box_iou(pred_box, g) for g in gts))


def pointing_hit(point_xy, gt: np.ndarray, tol_px: float = 0.0) -> bool | None:
    """Pointing game: the point ``(x, y)`` lies in ``gt`` grown by ``tol_px`` (Euclidean). None if ``gt`` is empty;
    a missing point (``None``) is a miss."""
    g = np.asarray(gt, bool)
    if not g.any():
        return None
    if point_xy is None:
        return False
    H, W = g.shape
    x = int(np.clip(np.rint(float(point_xy[0])), 0, W - 1))
    y = int(np.clip(np.rint(float(point_xy[1])), 0, H - 1))
    if g[y, x]:
        return True
    if tol_px <= 0:
        return False
    ys, xs = np.nonzero(g)
    return bool(np.min((xs - x) ** 2 + (ys - y) ** 2) <= float(tol_px) ** 2)


def inter_union(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int]:
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    return int(np.count_nonzero(p & g)), int(np.count_nonzero(p | g))


def precision_recall_tokencut(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """``metric.precision_recall`` (TokenCut): empty prediction -> P = 1, empty GT -> R = 1."""
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    tp = float(np.count_nonzero(p & g))
    pa, ga = float(np.count_nonzero(p)), float(np.count_nonzero(g))
    return (tp / pa if pa > 0 else 1.0), (tp / ga if ga > 0 else 1.0)


def f_beta(p, r, beta_sq: float = FBETA_SQ):
    """``metric.F_score``: ``(1 + b2) P R / (b2 P + R)``, NaN -> 0 (works elementwise on arrays)."""
    p, r = np.asarray(p, np.float64), np.asarray(r, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = (1.0 + beta_sq) * p * r / (beta_sq * p + r)
    return np.where(np.isfinite(f), f, 0.0)


def threshold_pr(score: np.ndarray, gt: np.ndarray, bins: int = FMAX_BINS) -> tuple[np.ndarray, np.ndarray]:
    """P and R of ``score > split`` for ``split in np.arange(0, 1, 1/bins)`` (``metric.metrics`` F_max branch)."""
    s = np.asarray(score, np.float64)
    g = np.asarray(gt, bool)
    splits = np.arange(0.0, 1.0, 1.0 / bins)
    ps, rs = np.empty(len(splits)), np.empty(len(splits))
    for i, sp in enumerate(splits):
        ps[i], rs[i] = precision_recall_tokencut(s > sp, g)
    return ps, rs


def max_f_beta(P: np.ndarray, R: np.ndarray, beta_sq: float = FBETA_SQ) -> float:
    """``metric.F_max``: F per (sample, threshold), mean over samples, max over thresholds."""
    P, R = np.asarray(P, np.float64), np.asarray(R, np.float64)
    if not P.size:
        return float("nan")
    return float(f_beta(P, R, beta_sq).mean(0).max())


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean(np.asarray(pred, bool) == np.asarray(gt, bool)))


def minmax(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, np.float64)
    lo, hi = float(a.min()), float(a.max())
    return np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo)


# --- DAVIS J / F (port of sav_benchmark.py:156-303, Grounded-SAM-2 @ b7a9c29) ------------------------------------
def davis_j(pred: np.ndarray, gt: np.ndarray) -> float:
    """``get_iou``: IoU, 1 when both masks are empty."""
    i, u = inter_union(pred, gt)
    return 1.0 if u == 0 else i / u


def seg2bmap(seg: np.ndarray) -> np.ndarray:
    """``_seg2bmap`` for the same-size case: 1-pixel boundary, offset 1/2 pixel towards the origin."""
    seg = np.asarray(seg, bool)
    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0
    return b


def disk(radius: float) -> np.ndarray:
    """``skimage.morphology.disk``: ``x^2 + y^2 <= r^2`` on a ``(2r+1)^2`` grid."""
    r = int(radius)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= radius * radius


def _dilate(b: np.ndarray, se: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_dilation

    return binary_dilation(b, structure=se) if b.any() else np.zeros_like(b)


def davis_f(pred: np.ndarray, gt: np.ndarray, bound_th: float = DAVIS_BOUND_TH) -> float:
    """Boundary F-measure (``Evaluator.feed_frame`` boundary part / davis2017 ``f_measure``)."""
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    bound_pix = bound_th if bound_th >= 1 else np.ceil(bound_th * np.linalg.norm(p.shape))
    se = disk(bound_pix)
    pb, gb = seg2bmap(p), seg2bmap(g)
    pd, gd = _dilate(pb, se), _dilate(gb, se)
    gt_match = gb & pd
    fg_match = pb & gd
    n_fg, n_gt = int(pb.sum()), int(gb.sum())
    if n_fg == 0 and n_gt > 0:
        prec, rec = 1.0, 0.0
    elif n_fg > 0 and n_gt == 0:
        prec, rec = 0.0, 1.0
    elif n_fg == 0 and n_gt == 0:
        prec, rec = 1.0, 1.0
    else:
        prec, rec = fg_match.sum() / float(n_fg), gt_match.sum() / float(n_gt)
    return 0.0 if prec + rec == 0 else float(2 * prec * rec / (prec + rec))


def db_statistics(values: Sequence[float]) -> tuple[float, float, float]:
    """davis2017-evaluation ``utils.db_statistics``: (mean, recall = mean(v > 0.5), decay over 4 bins)."""
    v = np.asarray(values, np.float64)
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    with np.errstate(invalid="ignore"):
        M = float(np.nanmean(v))
        O = float(np.nanmean(v > 0.5))
        ids = (np.round(np.linspace(1, len(v), 5) + 1e-10) - 1).astype(np.int64)
        bins = [v[ids[i]:ids[i + 1] + 1] for i in range(4)]
        D = float(np.nanmean(bins[0]) - np.nanmean(bins[3])) if len(bins[0]) and len(bins[3]) else float("nan")
    return M, O, D


# --- aggregation -------------------------------------------------------------------------------------------------
def bootstrap_ratio(num_by_ep: Mapping[str, float], den_by_ep: Mapping[str, float], *, n: int = BOOTSTRAP_N,
                    seed: int = BOOTSTRAP_SEED) -> dict:
    """Pooled ratio ``sum num / sum den`` and its 95 % percentile CI over episode resamples."""
    eps = sorted(e for e in den_by_ep if den_by_ep[e] > 0)
    if not eps:
        return {"value": None, "ci95": None, "n_episodes": 0}
    num = np.array([num_by_ep.get(e, 0.0) for e in eps], np.float64)
    den = np.array([den_by_ep[e] for e in eps], np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(eps), (n, len(eps)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = num[idx].sum(1) / den[idx].sum(1)
    boot = boot[np.isfinite(boot)]
    return {"value": float(num.sum() / den.sum()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
            if len(boot) else None, "n_episodes": len(eps), "num": float(num.sum()), "den": float(den.sum())}


def _by_episode(rows: Iterable[Mapping], key: str, den_key: str | None = None) -> tuple[dict, dict]:
    num: dict[str, float] = {}
    den: dict[str, float] = {}
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        e = r["episode_id"]
        if den_key is None:
            num[e] = num.get(e, 0.0) + float(v)
            den[e] = den.get(e, 0.0) + 1.0
        else:
            num[e] = num.get(e, 0.0) + float(v)
            den[e] = den.get(e, 0.0) + float(r[den_key])
    return num, den


SAMPLE_RATIOS = {  # summary key -> (row key, denominator key or None = count of samples with the key set)
    "acc05_mask": ("hit05_mask", None), "acc05_det": ("hit05_det", None), "corloc_target": ("hit05_mask", None),
    "corloc_any_object": ("corloc_any", None), "mIoU": ("iou", None), "oIoU": ("inter", "union"),
    "pixel_acc": ("pixel_acc", None), "fbeta_binary": ("fbeta", None),
    **{f"pointing_tol{t}": (f"point_hit_tol{t}", None) for t in POINTING_TOLS_PX},
    **{f"pointing_box_centre_tol{t}": (f"boxc_hit_tol{t}", None) for t in POINTING_TOLS_PX},
}


def summarize_samples(rows: Sequence[Mapping]) -> dict:
    """Pooled ratios (+ episode-bootstrap CI) of one arm x camera set of sample rows (GT-visible samples)."""
    vis = [r for r in rows if r.get("gt_visible")]
    out: dict[str, Any] = {"n_samples": len(vis), "n_episodes": len({r["episode_id"] for r in vis})}
    if any(r.get("map_missing") for r in vis):
        out["n_map_missing"] = sum(bool(r.get("map_missing")) for r in vis)
    for name, (key, den) in SAMPLE_RATIOS.items():
        if not any(r.get(key) is not None for r in vis):
            continue
        out[name] = bootstrap_ratio(*_by_episode(vis, key, den))
    if any(r.get("score_P") is not None for r in vis):
        P = np.array([r["score_P"] for r in vis if r.get("score_P") is not None])
        R = np.array([r["score_R"] for r in vis if r.get("score_P") is not None])
        f = f_beta(P, R).mean(0)
        out["maxFbeta"] = {"value": float(f.max()), "threshold": float(np.arange(0, 1, 1 / FMAX_BINS)[int(np.argmax(f))]),
                           "n_samples": int(len(P))}
    return out


CONF_THRESHOLDS = tuple(np.round(np.arange(0.0, 1.0, 0.05), 2))


def conf_sweep(rows: Sequence[Mapping], thresholds=CONF_THRESHOLDS) -> dict | None:
    """Detection P / R of the top-1 box (IoU >= 0.5 with the GT target box) when only predictions with score >= tau
    are kept (YOLOE / Grounding DINO confidence sweep; scores below the provider's own threshold were never output,
    so the curve starts at that threshold)."""
    vis = [r for r in rows if r.get("gt_visible") and "det_score" in r]
    if not vis:
        return None
    s = np.array([np.nan if r["det_score"] is None else float(r["det_score"]) for r in vis])
    hit = np.array([float(r.get("hit05_det") or 0.0) for r in vis])
    out = {"thresholds": list(thresholds), "precision": [], "recall": [], "n_pred": [], "n_samples": len(vis),
           "any_candidate_recall": float(np.mean([r.get("any_candidate_hit", 0.0) for r in vis]))}
    for tau in thresholds:
        keep = np.nan_to_num(s, nan=-1.0) >= tau
        out["n_pred"].append(int(keep.sum()))
        out["precision"].append(float(hit[keep].mean()) if keep.any() else None)
        out["recall"].append(float((hit * keep).sum() / len(vis)))
    return out


def summarize_sequences(rows: Sequence[Mapping]) -> dict:
    """DAVIS J / F statistics per (episode, camera) sequence, averaged over sequences."""
    seqs: dict[tuple, list] = {}
    for r in rows:
        if r.get("J") is None:
            continue
        seqs.setdefault((r["episode_id"], r["camera"]), []).append(r)
    stats = {k: [] for k in ("J_mean", "J_recall", "J_decay", "F_mean", "F_recall", "F_decay", "JF_mean",
                             "temporal_iou_pred", "temporal_iou_gt", "temporal_iou_abs_gap")}
    for _, rs in sorted(seqs.items()):
        rs = sorted(rs, key=lambda r: r["obs_index"])
        jm, jo, jd = db_statistics([r["J"] for r in rs])
        fm, fo, fd = db_statistics([r["F"] for r in rs])
        for k, v in zip(("J_mean", "J_recall", "J_decay", "F_mean", "F_recall", "F_decay"), (jm, jo, jd, fm, fo, fd)):
            stats[k].append(v)
        stats["JF_mean"].append(0.5 * (jm + fm))
        tp = [r["temporal_iou_pred"] for r in rs if r.get("temporal_iou_pred") is not None]
        tg = [r["temporal_iou_gt"] for r in rs if r.get("temporal_iou_gt") is not None]
        if tp:
            stats["temporal_iou_pred"].append(float(np.mean(tp)))
            stats["temporal_iou_gt"].append(float(np.mean(tg)))
            stats["temporal_iou_abs_gap"].append(float(np.mean([abs(a - b) for a, b in zip(tp, tg)])))
    out = {"n_sequences": len(seqs)}
    for k, v in stats.items():
        v = [x for x in v if np.isfinite(x)]
        out[k] = float(np.mean(v)) if v else None
    return out


# =============================================================================================================
# data access
# =============================================================================================================
def resolve_run(arg: str | pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(arg)
    if p.is_dir():
        return p.resolve()
    from common.run_artifacts import results_root

    q = results_root() / str(arg)
    if not q.is_dir():
        raise FileNotFoundError(f"run not found: {arg}")
    return q.resolve()


def arm_dir_name(arm: str) -> str:
    """``grounded_sam2:per_frame|phrase`` -> ``grounded_sam2-per_frame__phrase`` (``offline_grounding.save_masks``)."""
    spec, _, pm = arm.partition("|")
    return f"{spec.replace(':', '-')}__{pm}"


def load_mask_npz(path: pathlib.Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        out = {}
        for k in z.files:
            if k.endswith("__shape"):
                continue
            H, W = (int(v) for v in z[f"{k}__shape"])
            out[k] = np.unpackbits(z[k], count=H * W).reshape(H, W).astype(bool)
    return out


def read_frame_rows(run: pathlib.Path, episodes: set[str] | None = None) -> dict[tuple[str, str, int], dict]:
    """``(arm, episode_id, obs_index) -> {camera: {found, boxes, extras}}`` from the run's ``frames.jsonl``."""
    out = {}
    with open(run / "frames.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if "cameras" not in r or (episodes is not None and r["episode_id"] not in episodes):
                continue
            cams = {c: {"found": v.get("found"), "boxes": v.get("boxes"), "extras": v.get("extras") or {},
                        "score": v.get("score"), "harness_2d_iou": v.get("2d_iou")} for c, v in r["cameras"].items()}
            out[(r["arm"], r["episode_id"], int(r["obs_index"]))] = cams
    return out


def run_settings(run: pathlib.Path) -> dict:
    p = run / "provenance.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text())
    s = dict(d.get("settings") or {})
    s["_scene_gt_abs"] = d.get("scene_gt")
    s["_attention_abs"] = (d.get("attention") or {}).get("run")
    return s


def selected_detector_box(cam: Mapping) -> np.ndarray | None:
    boxes = cam.get("boxes")
    if not cam.get("found") or not boxes:
        return None
    sel = cam.get("extras", {}).get("selected", 0)
    try:
        sel = int(sel)
    except (TypeError, ValueError):
        sel = 0
    if sel >= len(boxes):
        return None
    return np.asarray(boxes[sel], np.float64)


def native_point(arm: str, cam: Mapping) -> tuple[Any, str | None]:
    """``(point_xy, source)`` of an arm's native pointing location (OpenCV pixel frame) or ``(None, None)``."""
    provider = arm.split("|")[0].split(":")[0]
    ex = cam.get("extras", {}) or {}
    if provider == "localization_heads":
        p = ex.get("peak_xy")
        return (None if p is None else [float(p[0]), float(p[1])]), "lh_peak"
    if provider == "tokencut":
        seed, grid = ex.get("seed"), ex.get("feature_grid")
        if seed is None or grid is None:
            return None, "tokencut_seed"
        rows, cols = int(grid[0]), int(grid[1])
        r, c = divmod(int(seed), cols)
        ps = 16.0  # TokenCutConfig.patch_size (DINO ViT-S/16 on the 256 px OpenCV image)
        return [(c + 0.5) * ps - 0.5, (r + 0.5) * ps - 0.5], "tokencut_seed"
    return None, None


class EpisodeGT:
    """GT target / object-instance masks of one recorded episode (OpenCV frame), via ``offline_grounding.Episode``."""

    def __init__(self, ep_meta: dict, cameras, scene_dir: pathlib.Path | None):
        from Baseline.paper_baselines.eval import metrics as M
        from Baseline.paper_baselines.eval import offline_grounding as og

        self.ep = og.Episode(ep_meta, cameras, scene_dir)
        self.cameras = tuple(cameras)
        self._seg_labels = og.seg_labels
        self.target_idx = M.CLASSES.index("target")
        inst = self.ep.target["instances_geom_ids"]
        self.object_instances = {}
        for name, gids in inst.items():
            gids = np.asarray(gids, np.int64)
            if not len(gids):
                continue
            cls = M.CLASSES[int(self.ep.lut[gids[0]])]
            if cls in OBJECT_CLASSES:
                self.object_instances[name] = gids

    def frame(self, t: int) -> dict:
        r = self.ep.row(t)
        out = {"target": {}, "objects": {}, "rgb": {}}
        for c in self.cameras:
            seg = self.ep.seg[c][r]
            out["target"][c] = self._seg_labels(seg, self.ep.lut) == self.target_idx
            out["objects"][c] = {n: np.isin(seg, g) for n, g in self.object_instances.items()}
        return out

    def rgb(self, t: int) -> dict:
        from Baseline.paper_baselines import lift as L

        return {c: L.opencv_rgb_from_raw(self.ep.raw[c][t]) for c in self.cameras}


class SaliencyMaps:
    """Continuous maps from the recorded pi0.5 attention (audit run): frozen localization-heads combined map and the
    V11AB L8H2 cell, in the OpenCV frame (``to_image_orientation`` + bilinear upsampling, same as the provider)."""

    def __init__(self, attention_run: pathlib.Path, source_run: pathlib.Path, selection: str | None):
        from Baseline.paper_baselines.eval import offline_grounding as og
        from Baseline.paper_baselines.providers import localization_heads as LH

        self.LH = LH
        self.attn = og.AttentionRun(attention_run, source_run)
        sel = selection or str(og.DEFAULT_LH_SELECTION)
        self.provider = LH.LocalizationHeadsProvider(sel)
        self.selection_path = sel

    def maps(self, stem: str, t: int, hw: tuple[int, int]) -> dict | None:
        aux = self.attn.aux(stem, t)
        if aux is None:
            return None
        out = {"lh_combined_map": {}, "v11ab_attention_L8H2": {}, "lh_heads": {}, "lh_grid": {}, "cell_grid": {}}
        for c in aux["attention_map"]:
            M_img, reason, heads = self.provider.combined_map(aux, c)
            if M_img is not None:
                out["lh_grid"][c] = np.asarray(M_img, np.float32)
                out["lh_combined_map"][c] = self.LH.upscale_bilinear(M_img, hw)
                lo, hi = (int(v) for v in aux["attention_blocks"][c])
                P = int(aux.get("attention_grid", 16))
                out["lh_heads"][c] = {f"L{l}H{h}": self.LH.to_image_orientation(
                    np.asarray(aux["attention_pooled"][l, h, lo:hi], np.float32).reshape(P, P), aux["attention_orientation"])
                    for (l, h) in heads}
            g = self.LH.to_image_orientation(np.asarray(aux["attention_map"][c], np.float32), aux["attention_orientation"])
            out["cell_grid"][c] = g
            if float(g.max()) > float(g.min()):
                out["v11ab_attention_L8H2"][c] = self.LH.upscale_bilinear(g, hw)
        return out


# =============================================================================================================
# per-sample evaluation
# =============================================================================================================
def sample_row(pred: np.ndarray | None, gt: np.ndarray, objects: Mapping[str, np.ndarray], *, det_box=None,
               point=None, pointing: bool = False, box_centre_eval: bool = False, score: np.ndarray | None = None) -> dict:
    """All per-(frame, camera) quantities of one arm. ``pred`` None = continuous-map arm (``score`` given).

    ``pointing``: the arm has a native point (``point``; ``None`` = it produced none -> a miss)."""
    g = np.asarray(gt, bool)
    vis = bool(g.any())
    row: dict[str, Any] = {"gt_visible": vis, "gt_pixels": int(g.sum())}
    gbox = mask_box(g)
    if pred is not None:
        p = np.asarray(pred, bool)
        i, u = inter_union(p, g)
        pbox = mask_box(p)
        row.update(pred_pixels=int(p.sum()), inter=i, union=u, J=davis_j(p, g), F=davis_f(p, g))
        if vis:
            P, R = precision_recall_tokencut(p, g)
            row.update(iou=i / u if u else 0.0, pixel_acc=pixel_accuracy(p, g), fbeta=float(f_beta(P, R)),
                       precision=P, recall=R, box_iou_mask=best_box_iou(pbox, [gbox]))
            row["hit05_mask"] = float(row["box_iou_mask"] >= ACC_IOU)
            any_iou = best_box_iou(pbox, [mask_box(m) for m in objects.values()])
            row["corloc_any"] = None if any_iou is None else float(any_iou >= ACC_IOU)
            if det_box is not None or box_centre_eval:
                row["box_iou_det"] = best_box_iou(det_box, [gbox])
                row["hit05_det"] = float(row["box_iou_det"] >= ACC_IOU)
            if box_centre_eval:
                c = None if det_box is None else [0.5 * (det_box[0] + det_box[2]) - 0.5, 0.5 * (det_box[1] + det_box[3]) - 0.5]
                for t in POINTING_TOLS_PX:
                    row[f"boxc_hit_tol{t}"] = float(pointing_hit(c, g, t))
    if score is not None and vis:
        s = minmax(score)
        Ps, Rs = threshold_pr(s, g)
        row["score_P"], row["score_R"] = Ps.tolist(), Rs.tolist()
        pk = np.unravel_index(int(np.argmax(score)), np.asarray(score).shape)
        point, pointing = [float(pk[1]), float(pk[0])], True
        sb = s > 0.5
        i, u = inter_union(sb, g)
        row.update(inter=i, union=u, iou=i / u if u else 0.0, pixel_acc=pixel_accuracy(sb, g))
    if vis and pointing:
        for t in POINTING_TOLS_PX:
            row[f"point_hit_tol{t}"] = float(pointing_hit(point, g, t))
    row["point_xy"] = point
    return row


# =============================================================================================================
# main loop
# =============================================================================================================
def make_output(run_id: str | None, out_dir: str | None, settings: dict, tool: str) -> pathlib.Path:
    """``create_run`` under AG3S_VLA-storage/results/benchmark (``--run-id``), or ``--out-dir`` below ``outputs/``."""
    if out_dir:
        out = pathlib.Path(out_dir).resolve()
        base = (PROJECT_ROOT / "outputs").resolve()
        if not out.is_relative_to(base):
            raise ValueError(f"--out-dir must be below {base} (benchmark results go through create_run)")
        out.mkdir(parents=True, exist_ok=False)
        (out / "settings.json").write_text(json.dumps({"tool": tool, **settings}, indent=1, default=str) + "\n")
        return out
    from common.run_artifacts import create_run

    return create_run(run_id=run_id, settings={"tool": tool, **settings})


def _episode_list(source: pathlib.Path, wanted, limit):
    from Baseline.paper_baselines.eval import offline_grounding as og

    eps = [e for e in og.load_episodes(source) if e["has_geom"] and e["has_npz"]]
    if wanted:
        eps = [e for e in eps if e["stem"] in set(wanted)]
    if limit:
        eps = eps[:limit]
    return eps


def run(args) -> pathlib.Path:
    grun = resolve_run(args.grounding_run)
    st = run_settings(grun)
    source = resolve_run(args.source_run or st.get("source_run"))
    scene_dir = pathlib.Path(args.scene_gt or st.get("_scene_gt_abs") or st.get("scene_gt") or "")
    if args.scene_gt or st.get("_scene_gt_abs") or st.get("scene_gt"):
        scene_dir = (PROJECT_ROOT / scene_dir).resolve() if not scene_dir.is_absolute() else scene_dir
    else:
        scene_dir = None
    attn_arg = args.attention_run or st.get("_attention_abs") or st.get("attention_run")
    cameras = tuple(args.cameras)
    mask_root = grun / "masks"
    arms = sorted(d.name for d in mask_root.iterdir() if d.is_dir()) if mask_root.is_dir() else []
    if not arms:
        raise SystemExit(f"{grun}: no masks/ (run offline_grounding with --save-masks)")
    if args.arms:
        arms = [a for a in arms if a in {arm_dir_name(x) if "|" in x else x for x in args.arms}]
    episodes = _episode_list(source, args.episodes, args.limit_episodes)
    frame_rows = read_frame_rows(grun, {e["stem"] for e in episodes})
    dir_to_arm = {arm_dir_name(a): a for (a, _, _) in frame_rows}
    sal = None
    if attn_arg and not args.no_maps:
        sal = SaliencyMaps(resolve_run(attn_arg), source, args.lh_selection or st.get("lh_selection"))
    settings = {"grounding_run": str(grun), "source_run": str(source), "scene_gt": str(scene_dir) if scene_dir else None,
                "attention_run": str(sal.attn.run) if sal else None, "lh_selection": sal.selection_path if sal else None,
                "cameras": list(cameras), "arms": arms,
                "episodes": [e["stem"] for e in episodes], "argv": sys.argv,
                "constants": {"acc_iou": ACC_IOU, "pointing_tols_px": list(POINTING_TOLS_PX), "fbeta_sq": FBETA_SQ,
                              "fmax_bins": FMAX_BINS, "davis_bound_th": DAVIS_BOUND_TH, "bootstrap": [BOOTSTRAP_N, BOOTSTRAP_SEED]}}
    out = make_output(args.run_id, args.out_dir, settings, "Baseline/paper_baselines/eval/native_metrics.py")
    rows_f = (out / "native_frames.jsonl").open("x")
    all_rows: list[dict] = []
    dump_dir = out / "figure_dump"
    tc = TokenCutRecompute() if args.tokencut_dino and args.dump_figure_data else None
    harness_diff: list[float] = []
    t0 = time.time()
    for ep_meta in episodes:
        gt = EpisodeGT(ep_meta, cameras, scene_dir)
        stem = gt.ep.stem
        frames = gt.ep.eval_frames()
        if args.limit_frames:
            frames = frames[: args.limit_frames]
        dump_frames = set(pick_dump_frames(frames, args.dump_per_episode, args.dump_frames, stem)) if args.dump_figure_data else set()
        prev: dict[tuple, dict] = {}
        for t in frames:
            G = gt.frame(t)
            maps = sal.maps(stem, t, G["target"][cameras[0]].shape) if sal is not None else None
            dump = {} if t in dump_frames else None
            for adir in arms:
                arm = dir_to_arm.get(adir, adir)
                p = mask_root / adir / f"{stem}_{t:04d}.npz"
                if not p.exists():
                    continue
                masks = load_mask_npz(p)
                cams_meta = frame_rows.get((arm, stem, t), {})
                provider = arm.split("|")[0].split(":")[0]
                for c in cameras:
                    cm = cams_meta.get(c, {})
                    det = selected_detector_box(cm) if provider in DETECTOR_PROVIDERS else None
                    point, psrc = native_point(arm, cm)
                    row = sample_row(masks[c], G["target"][c], G["objects"][c], det_box=det, point=point,
                                     pointing=psrc is not None, box_centre_eval=provider in DETECTOR_PROVIDERS)
                    row.update(arm=arm, episode_id=stem, obs_index=int(t), camera=c, point_source=psrc)
                    if provider in DETECTOR_PROVIDERS and row["gt_visible"]:
                        gbox = mask_box(G["target"][c])
                        row["det_score"] = cm.get("score") if cm.get("found") else None
                        row["any_candidate_hit"] = float(any(box_iou(b, gbox) >= ACC_IOU for b in (cm.get("boxes") or [])))
                    h = cm.get("harness_2d_iou")
                    if h is not None and row.get("iou") is not None:  # same mask, same GT -> must agree
                        harness_diff.append(abs(float(h) - row["iou"]))
                    key = (arm, c)
                    if key in prev:
                        row["temporal_iou_pred"] = davis_j(prev[key]["pred"], masks[c])
                        row["temporal_iou_gt"] = davis_j(prev[key]["gt"], G["target"][c])
                    prev[key] = {"pred": masks[c], "gt": G["target"][c]}
                    rows_f.write(json.dumps(_slim(row)) + "\n")
                    all_rows.append(_slim(row, keep_pr=True))
                if dump is not None:
                    dump[f"mask/{arm}"] = masks
                    dump[f"meta/{arm}"] = cams_meta
            if maps is not None:
                for marm in MAP_ARMS:
                    for c in cameras:
                        score = maps[marm].get(c)
                        if score is None:  # constant / non-finite attention: no prediction -> a miss, IoU 0
                            row = sample_row(np.zeros_like(G["target"][c]), G["target"][c], G["objects"][c], point=None,
                                             pointing=True)
                            keep = ("gt_visible", "gt_pixels", "inter", "union", "iou", "pixel_acc", "point_xy") + tuple(
                                f"point_hit_tol{tt}" for tt in POINTING_TOLS_PX)
                            row = {k: v for k, v in row.items() if k in keep}
                            row["map_missing"] = True
                        else:
                            row = sample_row(None, G["target"][c], G["objects"][c], score=score)
                        row.update(arm=marm, episode_id=stem, obs_index=int(t), camera=c, point_source=f"{marm}_peak")
                        rows_f.write(json.dumps(_slim(row)) + "\n")
                        all_rows.append(_slim(row, keep_pr=True))
            if dump is not None:
                write_dump(dump_dir, stem, t, gt, G, dump, maps, tc)
        rows_f.flush()
        print(f"[native_metrics] {stem}: {len(frames)} frames, {time.time() - t0:.1f}s", flush=True)
    rows_f.close()
    summary = summarize(all_rows, cameras)
    summary["settings"] = settings
    summary["definitions"] = __doc__
    summary["check_vs_harness_2d_iou"] = {"n": len(harness_diff), "max_abs_diff": max(harness_diff) if harness_diff else None}
    if tc is not None:
        summary["tokencut_dino_parity"] = tc.parity
    (out / "summary.json").write_text(json.dumps(_finite(summary), indent=1) + "\n")
    print(f"[native_metrics] wrote {out}")
    return out


def _slim(row: dict, keep_pr: bool = False) -> dict:
    """Drop the per-threshold P / R lists from the jsonl (they are only needed for maxFbeta in memory)."""
    if keep_pr:
        return dict(row)
    return {k: v for k, v in row.items() if k not in ("score_P", "score_R")}


def _finite(x):
    if isinstance(x, dict):
        return {str(k): _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating,)):
        return _finite(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def summarize(rows: Sequence[dict], cameras) -> dict:
    by_arm: dict[str, list] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    out = {"arms": {}}
    for arm, rs in sorted(by_arm.items()):
        d = {}
        for scope in ("all",) + tuple(cameras):
            sub = rs if scope == "all" else [r for r in rs if r["camera"] == scope]
            d[scope] = {**summarize_samples(sub), **({} if arm in MAP_ARMS else {"davis": summarize_sequences(sub)})}
            cs = conf_sweep(sub)
            if cs is not None:
                d[scope]["conf_sweep"] = cs
        out["arms"][arm] = d
    return out


# --- figure dump ----------------------------------------------------------------------------------------------------
def pick_dump_frames(frames: Sequence[int], per_episode: int, explicit: Sequence[str] | None, stem: str) -> list[int]:
    if explicit:
        sel = [int(x.split(":")[1]) for x in explicit if x.split(":")[0] == stem]
        return [t for t in sel if t in set(frames)]
    if per_episode <= 0 or not frames:
        return []
    idx = np.unique(np.round(np.linspace(0, len(frames) - 1, per_episode + 2)[1:-1]).astype(int)) if per_episode > 1 \
        else [len(frames) // 2]
    return [frames[i] for i in idx]


class TokenCutRecompute:
    """Re-runs TokenCut (DINO ViT-S/16 on CPU, cached weights) for dumped frames: the 2nd generalized eigenvector,
    bipartition and seed for the core-function figure, plus a parity check against the saved mask."""

    def __init__(self):
        from Baseline.paper_baselines.providers import tokencut as TC

        self.TC = TC
        self.provider = TC.TokenCutProvider(target_select="foreground", device="cpu")
        self.parity: list[dict] = []

    def __call__(self, rgb: np.ndarray, saved_mask: np.ndarray | None) -> dict:
        feats, dims, ps = self.provider._features("cam", rgb, None)
        res = self.TC.ncut(feats, dims, tau=self.provider.cfg.tau, eps=self.provider.cfg.eps,
                           binary_graph=self.provider.cfg.binary_graph)
        mask = self.TC.patch_mask_to_pixels(res.mask, rgb.shape[:2], ps).astype(bool)
        same = None if saved_mask is None else bool(np.array_equal(mask, saved_mask))
        return {"eigvec": res.eigvec.astype(np.float32), "bipartition": res.bipartition, "seed": np.int64(res.seed),
                "component_mask": res.mask, "mask_px": mask, "same_as_saved": same}


def write_dump(dump_dir: pathlib.Path, stem: str, t: int, gt: EpisodeGT, G: dict, dump: dict, maps, tc) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    arr: dict[str, np.ndarray] = {}
    meta: dict[str, Any] = {"episode_id": stem, "obs_index": int(t), "target": gt.ep.target["target"],
                            "object_instances": list(G["objects"][gt.cameras[0]].keys()), "arms": {}}
    rgb = gt.rgb(t)
    r = gt.ep.row(t)
    for c in gt.cameras:
        arr[f"rgb/{c}"] = rgb[c]
        arr[f"depth_mm/{c}"] = gt.ep.depth_mm[c][r]  # wire depth (uint16 mm), same OpenCV frame: lift with lift.lift_mask
        arr[f"K/{c}"] = gt.ep.K[c][r]
        arr[f"T_base_cam/{c}"] = gt.ep.T[c][r]
        arr[f"gt_target/{c}"] = G["target"][c]
        inst = np.full(G["target"][c].shape, -1, np.int16)
        for i, (n, m) in enumerate(G["objects"][c].items()):
            inst[m] = i
        arr[f"gt_instances/{c}"] = inst
    for k, v in dump.items():
        kind, arm = k.split("/", 1)
        if kind == "mask":
            for c, m in v.items():
                arr[f"mask/{arm}/{c}"] = m
        else:
            meta["arms"][arm] = v
    if maps is not None:
        for c in gt.cameras:
            if c in maps["lh_grid"]:
                arr[f"lh_combined_grid/{c}"] = maps["lh_grid"][c]
                arr[f"lh_combined_map/{c}"] = np.asarray(maps["lh_combined_map"][c], np.float32)
                for hn, g in maps["lh_heads"][c].items():
                    arr[f"lh_head/{c}/{hn}"] = np.asarray(g, np.float32)
            if c in maps["cell_grid"]:
                arr[f"v11ab_cell_grid/{c}"] = np.asarray(maps["cell_grid"][c], np.float32)
    if tc is not None:
        tc_arm = next((a for a in meta["arms"] if a.startswith("tokencut:foreground")), None)
        for c in gt.cameras:
            saved = dump.get(f"mask/{tc_arm}", {}).get(c) if tc_arm else None
            r = tc(rgb[c], saved)
            for k, v in r.items():
                if k != "same_as_saved":
                    arr[f"tokencut/{c}/{k}"] = np.asarray(v)
            tc.parity.append({"episode_id": stem, "obs_index": int(t), "camera": c, "same_as_saved": r["same_as_saved"]})
            meta.setdefault("tokencut_same_as_saved", {})[c] = r["same_as_saved"]
    arr["meta_json"] = np.array(json.dumps(meta, default=str))
    np.savez_compressed(dump_dir / f"grounding_{stem}_{t:04d}.npz", **arr)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grounding-run", required=True, help="offline_grounding run with masks/ (--save-masks)")
    ap.add_argument("--source-run", help="default: the grounding run's provenance settings")
    ap.add_argument("--scene-gt", help="default: the grounding run's provenance")
    ap.add_argument("--attention-run", help="attention_audit run (continuous maps); default: grounding run's provenance")
    ap.add_argument("--lh-selection")
    ap.add_argument("--no-maps", action="store_true", help="skip the continuous-map arms")
    ap.add_argument("--arms", nargs="*", help="arm labels (provider[:mode]|prompt) or mask dir names")
    ap.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    ap.add_argument("--episodes", nargs="*")
    ap.add_argument("--limit-episodes", type=int)
    ap.add_argument("--limit-frames", type=int)
    ap.add_argument("--dump-figure-data", action="store_true", help="write figure_dump/grounding_<stem>_<t>.npz")
    ap.add_argument("--dump-per-episode", type=int, default=1, help="dumped frames per episode (evenly spaced)")
    ap.add_argument("--dump-frames", nargs="*", help="explicit <stem>:<obs_index> frames to dump")
    ap.add_argument("--tokencut-dino", action="store_true", help="recompute TokenCut eigenvectors (DINO, CPU) for dumps")
    ap.add_argument("--run-id")
    ap.add_argument("--out-dir", help="below outputs/ (smoke); otherwise create_run under results/benchmark")
    return ap


def main(argv=None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
