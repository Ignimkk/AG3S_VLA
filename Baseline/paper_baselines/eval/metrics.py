"""Grounding metrics: 2D mask metrics and the development branch's 5 mm **soft** voxel metrics.

Soft IoU (same definition as ``/mnt/dev/AG3S_VLA-development`` ``AG3S/ag3s/experiments/studies/ifg_graph.py``
``build_geometry`` + ``evaluate``; read, not imported):

* **eval voxels** -- every valid depth pixel of every camera, back-projected (V11AB ``backproject``,
  ``range_max`` 2 m) and binned on the global 5 mm grid ``floor(p / 0.005)``; the union over cameras;
* each eval voxel ``v`` carries the **pixel fraction** of each GT class among the pixels that fell in it;
  ``t(v)`` = target fraction (soft label);
* a prediction is a voxel set ``P``: ``TP = sum_{v in P} t(v)``, ``FP = sum_{v in P} (1 - t(v))``,
  ``FN = sum_{v not in P} t(v)``; ``IoU = TP / (TP + FP + FN)``, ``precision = TP / (TP + FP)`` (None if P is empty),
  ``recall = TP / sum t`` ; all None when ``sum t = 0`` (``missing_reason = target_not_visible``);
* ``neighbor_fp = sum_{v in P}`` of the class fractions of ``NEIGHBOR`` classes (other objects + destination),
  likewise ``*_fp`` for every class.

Additions here (not in ifg_graph): a predicted voxel that is not an eval voxel (possible only for point-based
predictions such as ATS points) counts as FP with weight 1 and is reported as ``pred_offgrid``; hard metrics
(``t >= 0.5``) are reported next to the soft ones.

Aggregation (ifg_heldout_summary.py): frame -> episode mean -> mean over episodes; paired bootstrap over episodes,
2,000 resamples, seed 20261015, 95 % percentile interval of the mean paired difference.
"""
from __future__ import annotations

import collections
import dataclasses
from typing import Iterable, Mapping, Sequence

import numpy as np

from Baseline.paper_baselines.lift import RANGE_MAX_M, VOXEL_M, backproject_pixels

CLASSES = ("background", "target", "destination", "other_object", "fixture", "support", "robot", "mount")
NEIGHBOR = ("other_object", "destination")
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20261015


# --- 2D -------------------------------------------------------------------------------------------------------
def mask_metrics_2d(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = np.asarray(pred, bool), np.asarray(gt, bool)
    if p.shape != g.shape:
        raise ValueError(f"pred {p.shape} vs gt {g.shape}")
    tp = int(np.count_nonzero(p & g))
    fp = int(np.count_nonzero(p & ~g))
    fn = int(np.count_nonzero(~p & g))
    n_gt, n_pred = tp + fn, tp + fp
    return {"tp": tp, "fp": fp, "fn": fn, "gt_pixels": n_gt, "pred_pixels": n_pred,
            "iou": tp / (tp + fp + fn) if n_gt > 0 else None,
            "precision": tp / n_pred if n_pred > 0 else None,
            "recall": tp / n_gt if n_gt > 0 else None}


# --- 3D soft voxel grid ---------------------------------------------------------------------------------------
@dataclasses.dataclass
class EvalGrid:
    keys: np.ndarray  # (V, 3) int64, sorted by packed key
    class_frac: np.ndarray  # (V, C)
    classes: tuple
    kmin: np.ndarray
    ext: np.ndarray
    packed: np.ndarray  # (V,) sorted
    voxel: float

    @property
    def target(self) -> np.ndarray:
        return self.class_frac[:, self.classes.index("target")]

    def _pack(self, keys: np.ndarray):
        k = np.asarray(keys, np.int64).reshape(-1, 3) - self.kmin
        inside = np.all((k >= 0) & (k < self.ext), axis=1)
        packed = (k[:, 0] * self.ext[1] + k[:, 1]) * self.ext[2] + k[:, 2]
        return packed, inside

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        """Eval-voxel index of each key, -1 if not an eval voxel."""
        packed, inside = self._pack(keys)
        if not len(self.packed):
            return np.full(len(packed), -1, np.int64)
        idx = np.minimum(np.searchsorted(self.packed, packed), len(self.packed) - 1)
        hit = inside & (self.packed[idx] == packed)
        return np.where(hit, idx, -1)

    def pixel_voxels(self, points: np.ndarray) -> np.ndarray:
        return self.lookup(np.floor(np.asarray(points, np.float64).reshape(-1, 3) / self.voxel).astype(np.int64))


def build_eval_grid(cameras: Sequence[Mapping], *, classes: Sequence[str] = CLASSES, voxel: float = VOXEL_M,
                    range_max: float = RANGE_MAX_M) -> EvalGrid:
    """``cameras``: ``[{"depth_m", "K", "T_base_cam", "labels" (H, W) int class index}, ...]`` (same frame as depth)."""
    P, L = [], []
    for cam in cameras:
        pts, u, v = backproject_pixels(cam["depth_m"], cam["K"], cam["T_base_cam"], range_max=range_max)
        P.append(pts)
        L.append(np.asarray(cam["labels"])[v, u].astype(np.int64))
    P = np.concatenate(P) if P else np.zeros((0, 3))
    L = np.concatenate(L) if L else np.zeros(0, np.int64)
    C = len(classes)
    if not len(P):
        z = np.zeros((0, 3), np.int64)
        return EvalGrid(z, np.zeros((0, C)), tuple(classes), np.zeros(3, np.int64), np.ones(3, np.int64), np.zeros(0, np.int64), voxel)
    keys = np.floor(P / voxel).astype(np.int64)
    kmin = keys.min(0)
    ext = keys.max(0) - kmin + 1
    packed = ((keys[:, 0] - kmin[0]) * ext[1] + (keys[:, 1] - kmin[1])) * ext[2] + (keys[:, 2] - kmin[2])
    uniq, inv = np.unique(packed, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    frac = np.zeros((len(uniq), C))
    np.add.at(frac, (inv, L), 1.0)
    frac /= cnt[:, None]
    k0, rem = np.divmod(uniq, ext[1] * ext[2])
    k1, k2 = np.divmod(rem, ext[2])
    vkeys = np.stack([k0, k1, k2], 1) + kmin
    return EvalGrid(vkeys, frac, tuple(classes), kmin, ext, uniq, voxel)


def voxel_metrics(grid: EvalGrid, pred_keys: np.ndarray, *, neighbor: Iterable[str] = NEIGHBOR) -> dict:
    """Soft (and hard) voxel metrics of a predicted voxel-key set ``(K, 3)`` on ``grid``."""
    pred_keys = np.unique(np.asarray(pred_keys, np.int64).reshape(-1, 3), axis=0)
    idx = grid.lookup(pred_keys) if len(pred_keys) else np.zeros(0, np.int64)
    offgrid = int(np.count_nonzero(idx < 0))
    sel = np.unique(idx[idx >= 0])
    t = grid.target
    tot = float(t.sum())
    tp = float(t[sel].sum())
    fp = float((1.0 - t[sel]).sum()) + offgrid
    fn = tot - tp
    hard = t >= 0.5
    htp = int(np.count_nonzero(hard[sel]))
    hfp = int(len(sel) - htp) + offgrid
    hfn = int(np.count_nonzero(hard)) - htp
    n_pred = len(sel) + offgrid
    cf = grid.class_frac[sel]
    out = {"TP": tp, "FP": fp, "FN": fn, "target_voxels": tot, "pred_voxels": int(n_pred), "pred_offgrid": offgrid,
           "iou": tp / (tp + fp + fn) if tot > 0 else None,
           "precision": tp / (tp + fp) if n_pred > 0 and tot > 0 else None,
           "recall": tp / tot if tot > 0 else None,
           "hard_iou": htp / (htp + hfp + hfn) if hard.any() else None,
           "hard_precision": htp / (htp + hfp) if n_pred > 0 and hard.any() else None,
           "hard_recall": htp / int(np.count_nonzero(hard)) if hard.any() else None,
           "neighbor_fp": float(sum(cf[:, grid.classes.index(c)].sum() for c in neighbor))}
    for c in grid.classes:
        if c != "target":
            out[f"{c}_fp"] = float(cf[:, grid.classes.index(c)].sum())
    if tot <= 0:
        out["missing_reason"] = "target_not_visible"
    return out


# --- aggregation ----------------------------------------------------------------------------------------------
def episode_means(rows: Iterable[Mapping], metric: str, *, episode_key: str = "episode_id") -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        v = r.get(metric)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            continue
        by[r[episode_key]].append(float(v))
    return {e: float(np.mean(v)) for e, v in by.items()}


def aggregate(rows: Sequence[Mapping], metric: str, *, episode_key: str = "episode_id") -> dict:
    em = episode_means(rows, metric, episode_key=episode_key)
    n_frames = sum(1 for r in rows if r.get(metric) is not None)
    return {"episode_mean": float(np.mean(list(em.values()))) if em else None, "n_episodes": len(em), "n_frames": n_frames}


def paired_bootstrap(a: Mapping, b: Mapping, *, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict | None:
    """Mean paired difference ``a - b`` over common episodes with a percentile 95 % CI (episode resampling)."""
    es = sorted(set(a) & set(b))
    if not es:
        return None
    d = np.array([a[e] - b[e] for e in es], np.float64)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return {"n_episodes": len(es), "diff": float(d.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "better": int((d > 1e-9).sum()), "worse": int((d < -1e-9).sum()), "n_boot": int(n), "seed": int(seed)}


def latency_stats(values: Iterable[float]) -> dict:
    v = np.asarray([x for x in values if x is not None], np.float64)
    if not len(v):
        return {"n": 0, "median_ms": None, "p95_ms": None}
    return {"n": int(len(v)), "median_ms": float(np.median(v)), "p95_ms": float(np.percentile(v, 95))}


def found_rate(flags: Iterable[bool]) -> dict:
    f = np.asarray(list(flags), bool)
    return {"n": int(len(f)), "found": int(f.sum()), "rate": float(f.mean()) if len(f) else None}


def pooled_sum(rows: Iterable[Mapping], key: str) -> float:
    return float(sum((r.get(key) or 0.0) for r in rows))


__all__ = ["BOOTSTRAP_N", "BOOTSTRAP_SEED", "CLASSES", "EvalGrid", "NEIGHBOR", "aggregate", "build_eval_grid", "episode_means",
           "found_rate", "latency_stats", "mask_metrics_2d", "paired_bootstrap", "pooled_sum", "voxel_metrics"]
