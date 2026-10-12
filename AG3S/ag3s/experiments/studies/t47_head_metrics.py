"""T47 — pure numpy pieces of the localization-head selection (no MuJoCo, no GT reading).

Kang et al., *"Your Large Vision-Language Model Only Needs A Few Attention Heads For Visual
Grounding"* (arXiv 2503.06287, `papers/`) pick a handful of attention heads with two label-free
criteria and sum their maps:

1. **attention sum** ``S_img`` — how much of the query's softmax mass lands on image tokens. Heads
   are kept when ``S_img >= tau``, with ``tau`` the knee ("maximum curvature") of the ascending
   sorted ``S_img`` curve (paper §4.1, Fig.3).
2. **spatial entropy** (appendix B.1 Eq.4–7) — binarize ``ReLU(A - mean(A))``, take the
   8-connected components, ``H = -sum_C P(C) log P(C)`` with ``P(C) = |C| / sum |C|`` (raw nats).
   One compact blob -> 0; scattered blobs -> large.

Per sample the ``n_lowest`` (10) eligible heads with the lowest entropy are *selected*; a head's
**selection frequency** is the share of samples that selected it. The top-k by frequency (k = 3
in the paper) is the fixed head set, combined by Gaussian smoothing + element-wise sum.

What differs for us (T47.task.md "우리에게 맞춘 차이"): the query is the *action* token, the grid is
16x16, there are three cameras, and the npz holds raw softmax mass (no renormalisation), so
``S_cam`` / ``S_img`` are plain sums of the stored maps.

Everything here is deterministic numpy so `tests/ag3s/test_t47_head_selection.py` can pin it on
synthetic data. Definitions that the paper leaves open are fixed here and listed in `T47.impl.md`.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np

GRID = 16
#: Layers the paper excludes from selection (§4: the first two layers attend broadly).
EXCLUDED_LAYERS = (0, 1)
#: Relative tolerance under which ``A - mean(A)`` counts as zero. A constant map's mean is not
#: always bit-equal to its entries (0.1 * 256 / 256 rounds), and without a tolerance a uniform map
#: would come out as hundreds of one-pixel "components" — entropy ln(256) instead of "undefined".
BINARIZE_REL_EPS = 1e-9
#: Classes the GT patch coverage is split into, in this order (index 0 is always the target).
CLASS_NAMES = ("target", "destination", "other_object", "robot_gripper", "robot", "support",
               "background")
#: step-01 convention (`attention_report.score_maps`): the hottest patch counts as "on target" when
#: the target covers more than this fraction of it.
PATCH_HIT_COV = 0.05


# ------------------------------------------------------------------------------ criterion 1


def image_attention_sum(cell_maps: np.ndarray, *, camera_axis: int = -3):
    """``(S_cam, S_img)`` of raw attention maps.

    `cell_maps` is ``(..., C, 16, 16)`` (camera axis third from the end by default). ``S_cam`` is
    the 16x16 sum per camera ``(..., C)`` and ``S_img`` the sum over the cameras ``(...)`` — the
    total softmax mass the query put on the 768 image tokens, i.e. the paper's ``S_img``.
    """
    maps = np.asarray(cell_maps, np.float64)
    if maps.ndim < 3 or maps.shape[-2:] != (GRID, GRID):
        raise ValueError(f"expected (..., C, {GRID}, {GRID}) maps, got {maps.shape}")
    s_cam = maps.sum(axis=(-1, -2))
    axis = camera_axis + 2 if camera_axis < 0 else camera_axis
    return s_cam, s_cam.sum(axis=axis)


# ------------------------------------------------------------------------------ criterion 2


def binarize_above_mean(maps: np.ndarray, *, rel_eps: float = BINARIZE_REL_EPS) -> np.ndarray:
    """``ReLU(A - mean(A)) > 0`` per 16x16 map (last two axes), with a relative zero tolerance.

    The tolerance is ``rel_eps * max|A|`` per map; differences at or below it are zero. This only
    matters for (near-)constant maps — see `BINARIZE_REL_EPS`.
    """
    a = np.asarray(maps, np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("attention maps contain NaN/inf")
    diff = a - a.mean(axis=(-1, -2), keepdims=True)
    tol = rel_eps * np.abs(a).max(axis=(-1, -2), keepdims=True)
    return diff > tol


def spatial_entropy_batch(maps: np.ndarray, *, chunk: int = 20000):
    """Spatial entropy (raw nats) and component count of every 16x16 map in `maps` (``(..., H, W)``).

    Eq.4–7: ``B = ReLU(A - mean A) > 0``; 8-connected components ``C_i`` of ``B``;
    ``P(C_i) = |C_i| / sum_j |C_j|`` (pixel counts, not attention mass); ``H = -sum P log P``.

    **No component** (the map is constant, so nothing is above its mean) -> ``H = NaN``: entropy is
    undefined and the cell is not selectable for that sample (`selection_frequency` skips NaN).

    Vectorised: maps are stacked along a leading axis and labelled at once with a 3D structuring
    element that is all-ones in the middle slice and zero across the stack, so no component can join
    two maps.
    """
    from scipy import ndimage

    a = np.asarray(maps, np.float64)
    if a.ndim < 2:
        raise ValueError(f"expected (..., H, W) maps, got {a.shape}")
    lead, hw = a.shape[:-2], a.shape[-2:]
    flat = a.reshape((-1,) + hw)
    n = flat.shape[0]
    entropy = np.full(n, np.nan, np.float64)
    count = np.zeros(n, np.int64)
    structure = np.zeros((3, 3, 3), bool)
    structure[1] = True  # 8-connectivity inside a map, nothing across maps
    for start in range(0, n, chunk):
        block = binarize_above_mean(flat[start:start + chunk])
        labels, n_labels = ndimage.label(block, structure=structure)
        if n_labels == 0:
            continue
        flat_labels = labels.ravel()
        sizes = np.bincount(flat_labels, minlength=n_labels + 1)[1:].astype(np.float64)
        # Which map each label lives in: the stack index of its first pixel.
        ids, first = np.unique(flat_labels, return_index=True)
        owner = (first[ids > 0] // (hw[0] * hw[1])).astype(np.int64)
        total = np.bincount(owner, weights=sizes, minlength=block.shape[0])
        p = sizes / total[owner]
        h = np.bincount(owner, weights=-p * np.log(p), minlength=block.shape[0])
        c = np.bincount(owner, minlength=block.shape[0])
        has = c > 0
        seg = entropy[start:start + block.shape[0]]
        seg[has] = h[has]
        count[start:start + block.shape[0]] = c
    # -0.0 from a single component (p = 1) reads oddly in JSON; make it +0.0.
    entropy = np.where(entropy == 0.0, 0.0, entropy)
    return entropy.reshape(lead), count.reshape(lead)


def spatial_entropy(map16: np.ndarray) -> float:
    """`spatial_entropy_batch` for one map. NaN when the map has no above-mean pixel."""
    a = np.asarray(map16, np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected one (H, W) map, got {a.shape}")
    return float(spatial_entropy_batch(a[None])[0][0])


# ------------------------------------------------------------------------- threshold tau


@dataclasses.dataclass(frozen=True)
class KneeThreshold:
    """The knee of the ascending sorted curve and how it was found."""

    tau: float
    index: int  # position of tau in the ascending sorted values
    method: str
    knee_found: bool  # False -> degenerate curve; tau = min (every value eligible)
    sorted_values: np.ndarray
    score: np.ndarray  # per sorted position: (x - y) for kneedle, kappa for curvature


def max_curvature_threshold(values: Sequence[float], *, method: str = "kneedle") -> KneeThreshold:
    """``tau`` = the value at the "maximum curvature" point of the ascending sorted curve (§4.1).

    Both methods first map the sorted curve into the unit square: ``x_i = i / (n - 1)``,
    ``y_i = (v_i - v_min) / (v_max - v_min)``.

    * ``kneedle`` (primary) — Satopää et al. 2011: for an increasing convex curve (flat, then
      rising — what a few image-heavy heads among many make) the knee is ``argmax (x_i - y_i)``,
      the point farthest below the chord. It is the standard discrete stand-in for maximum
      curvature and, unlike a second difference, does not lock onto noise in the flat part.
    * ``curvature`` — discrete ``kappa_i = |y''| / (1 + y'^2)^{3/2}`` with ``np.gradient`` central
      differences on the normalised curve; the two endpoints (one-sided differences) are excluded.

    Degenerate curves (fewer than 3 values, all equal, or — for kneedle — no point below the
    chord) return ``knee_found=False`` and ``tau = min(values)`` so every value is eligible; the
    caller reports that rather than silently dropping cells.
    """
    v = np.sort(np.asarray(values, np.float64).reshape(-1))
    if v.size == 0 or not np.all(np.isfinite(v)):
        raise ValueError("values must be a non-empty finite sequence")
    if method not in ("kneedle", "curvature"):
        raise ValueError(f"method must be 'kneedle' or 'curvature', got {method!r}")
    n = v.size
    span = v[-1] - v[0]
    if n < 3 or span <= 0:
        return KneeThreshold(float(v[0]), 0, method, False, v, np.zeros(n))
    x = np.linspace(0.0, 1.0, n)
    y = (v - v[0]) / span
    if method == "kneedle":
        score = x - y
        i = int(np.argmax(score))
        if score[i] <= 0:
            return KneeThreshold(float(v[0]), 0, method, False, v, score)
        return KneeThreshold(float(v[i]), i, method, True, v, score)
    dy = np.gradient(y, x)
    d2y = np.gradient(dy, x)
    kappa = np.abs(d2y) / (1.0 + dy * dy) ** 1.5
    inner = kappa.copy()
    inner[[0, -1]] = -np.inf
    i = int(np.argmax(inner))
    return KneeThreshold(float(v[i]), i, method, True, v, kappa)


def eligible_cells(s_values: np.ndarray, *, method: str = "kneedle", axis: int = -1):
    """Per sample (all axes but `axis`), ``(eligible bool, tau)``: ``S >= tau`` with tau per sample."""
    s = np.moveaxis(np.asarray(s_values, np.float64), axis, -1)
    lead = s.shape[:-1]
    flat = s.reshape(-1, s.shape[-1])
    taus = np.empty(flat.shape[0], np.float64)
    found = np.empty(flat.shape[0], bool)
    for i, row in enumerate(flat):
        k = max_curvature_threshold(row, method=method)
        taus[i], found[i] = k.tau, k.knee_found
    elig = flat >= taus[:, None]
    return (np.moveaxis(elig.reshape(s.shape), -1, axis), taus.reshape(lead), found.reshape(lead))


# ------------------------------------------------------------------- selection frequency


@dataclasses.dataclass(frozen=True)
class SelectionFrequency:
    counts: np.ndarray  # (n_cells,) samples that selected each cell
    frequency: np.ndarray  # counts / n_samples
    n_samples: int
    n_selected: np.ndarray  # (n_samples,) how many cells each sample selected (<= n_lowest)
    selected: np.ndarray  # (n_samples, n_cells) bool


def selection_frequency(entropy: np.ndarray, eligible: np.ndarray, *, n_lowest: int = 10
                        ) -> SelectionFrequency:
    """Per sample, the `n_lowest` eligible cells with the lowest entropy; per cell, the share selected.

    `entropy` and `eligible` are ``(n_samples, n_cells)``. A cell is a candidate in a sample only if
    it is eligible **and** its entropy is finite (NaN = no component). Fewer than `n_lowest`
    candidates -> all of them. Ties in entropy are broken by cell index (stable sort), so the
    result is deterministic. The denominator is every sample, including ones with no candidate.
    """
    e = np.asarray(entropy, np.float64)
    m = np.asarray(eligible, bool)
    if e.ndim != 2 or e.shape != m.shape:
        raise ValueError(f"entropy {e.shape} and eligible {m.shape} must be the same (S, cells)")
    if n_lowest < 1:
        raise ValueError("n_lowest must be >= 1")
    cand = m & np.isfinite(e)
    key = np.where(cand, e, np.inf)
    order = np.argsort(key, axis=1, kind="stable")
    n_cand = cand.sum(axis=1)
    take = np.minimum(n_cand, int(n_lowest))
    selected = np.zeros_like(cand)
    rows = np.arange(e.shape[0])
    for j in range(int(n_lowest)):
        if j >= e.shape[1]:
            break
        hit = take > j
        selected[rows[hit], order[hit, j]] = True
    counts = selected.sum(axis=0).astype(np.int64)
    n = int(e.shape[0])
    freq = counts / n if n else np.zeros(e.shape[1])
    return SelectionFrequency(counts, freq, n, take.astype(np.int64), selected)


def rank_cells(frequency: np.ndarray, *, mean_entropy: Optional[np.ndarray] = None,
               mean_sum: Optional[np.ndarray] = None) -> np.ndarray:
    """Cell indices, best first: frequency desc, then mean entropy asc, then mean S desc, then index.

    The tie-breaks only order cells with equal frequency (common among never-selected cells); they
    are reported, never used to *select* a cell that has a lower frequency.
    """
    f = np.asarray(frequency, np.float64)
    n = f.size
    me = np.zeros(n) if mean_entropy is None else np.nan_to_num(
        np.asarray(mean_entropy, np.float64), nan=np.inf)
    ms = np.zeros(n) if mean_sum is None else np.asarray(mean_sum, np.float64)
    return np.lexsort((np.arange(n), -ms, me, -f))


# ------------------------------------------------------------------------ combining maps


def gaussian_smooth(maps: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing of the last two axes, sigma in grid cells. ``sigma == 0`` -> a copy.

    Boundary: ``mode="reflect"`` (half-sample symmetric, ``d c b a | a b c d``). With the kernel
    truncated at 4 sigma inside the 16-cell grid, every pixel's weight that falls outside is folded
    back in, so the **total mass of each map is preserved** (pinned by a unit test); ``constant``
    would leak mass at the edges and ``nearest`` would add it.
    """
    from scipy import ndimage

    a = np.asarray(maps, np.float64)
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if sigma == 0:
        return a.copy()
    sig = [0.0] * (a.ndim - 2) + [float(sigma), float(sigma)]
    return ndimage.gaussian_filter(a, sigma=sig, mode="reflect", truncate=4.0)


def combine_maps(maps: np.ndarray, cells: Sequence[tuple[int, int]], sigma: float = 0.0
                 ) -> np.ndarray:
    """Smooth each selected cell's map (sigma grid cells) and sum element-wise.

    `maps` is ``(..., L, H, 16, 16)`` (layer and head axes just before the grid); `cells` is a
    sequence of ``(layer, head)``. Returns ``(..., 16, 16)``. Raw maps are summed as is — no
    per-head normalisation (paper: element-wise sum of the selected heads' maps).
    """
    a = np.asarray(maps)
    if a.ndim < 4:
        raise ValueError(f"expected (..., L, H, {GRID}, {GRID}), got {a.shape}")
    cells = [(int(l), int(h)) for l, h in cells]
    if not cells:
        raise ValueError("cells must not be empty")
    out = None
    for l, h in cells:
        m = gaussian_smooth(np.asarray(a[..., l, h, :, :], np.float64), sigma)
        out = m if out is None else out + m
    return out


def head_average(maps: np.ndarray, *, excluded_layers: Sequence[int] = EXCLUDED_LAYERS
                 ) -> np.ndarray:
    """Mean over every (layer, head) except `excluded_layers` — baseline (c)."""
    a = np.asarray(maps, np.float64)
    keep = [l for l in range(a.shape[-4]) if l not in set(excluded_layers)]
    return a[..., keep, :, :, :].mean(axis=(-4, -3))


def cell_list(n_layers: int, n_heads: int, *, excluded_layers: Sequence[int] = EXCLUDED_LAYERS
              ) -> list[tuple[int, int]]:
    """The selectable ``(layer, head)`` cells in a fixed order (layer-major)."""
    ex = set(int(l) for l in excluded_layers)
    return [(l, h) for l in range(n_layers) if l not in ex for h in range(n_heads)]


# ------------------------------------------------------------------------- 2D scoring


def map_scores(maps: np.ndarray, class_cov: np.ndarray, *, target_index: int = 0,
               iou_thresholds: Sequence[float] = (0.5, 0.25),
               patch_hit_cov: float = PATCH_HIT_COV) -> dict:
    """Score ``(N, 16, 16)`` maps against ``(N, K, 16, 16)`` GT class coverage (per-patch shares).

    Per sample:
      * ``class_mass`` (N, K) — share of the map's mass on each class (sums to 1 when the coverage
        does); ``target_mass`` = its target column.
      * ``lift`` — target mass / target area share (step-01 definition; 1.0 = what uniform does).
      * ``peak_on_target`` — target coverage of the argmax patch > `patch_hit_cov` (step-01).
        A constant map's argmax is patch (0, 0) — meaningless but deterministic, as in step-01.
      * ``iou@t`` — IoU of the mean-binarized map (`binarize_above_mean`) with the GT patch mask
        ``target coverage >= t``. NaN when the GT mask is empty (target too small to fill any
        patch to t) — counted by the caller, never scored as 0.
    Maps with non-positive total mass give NaN for the mass-based fields.
    """
    a = np.asarray(maps, np.float64)
    cov = np.asarray(class_cov, np.float64)
    if a.ndim != 3 or cov.ndim != 4 or cov.shape[0] != a.shape[0] or cov.shape[2:] != a.shape[1:]:
        raise ValueError(f"maps {a.shape} / class_cov {cov.shape} do not line up")
    total = a.sum(axis=(1, 2))
    good = total > 0
    norm = np.where(good[:, None, None], a / np.where(good, total, 1.0)[:, None, None], np.nan)
    class_mass = np.einsum("nij,nkij->nk", norm, cov)
    tcov = cov[:, target_index]
    area_share = tcov.sum(axis=(1, 2)) / float(a.shape[1] * a.shape[2])
    target_mass = class_mass[:, target_index]
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(area_share > 0, target_mass / area_share, np.nan)
    flat = a.reshape(a.shape[0], -1).argmax(axis=1)
    peak_cov = tcov.reshape(a.shape[0], -1)[np.arange(a.shape[0]), flat]
    out = {
        "class_mass": class_mass,
        "target_mass": target_mass,
        "area_share": area_share,
        "lift": lift,
        "peak_target_cov": peak_cov,
        "peak_on_target": (peak_cov > patch_hit_cov).astype(np.float64),
    }
    binary = binarize_above_mean(a)
    for t in iou_thresholds:
        gt = tcov >= float(t)
        inter = (binary & gt).sum(axis=(1, 2)).astype(np.float64)
        union = (binary | gt).sum(axis=(1, 2)).astype(np.float64)
        has_gt = gt.any(axis=(1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = np.where(has_gt, inter / np.maximum(union, 1.0), np.nan)
        out[f"iou@{t:g}"] = iou
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> dict:
    """Spearman rho and two-sided p over the pairs where both are finite."""
    from scipy import stats

    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return {"rho": float("nan"), "p": float("nan"), "n": int(ok.sum())}
    r = stats.spearmanr(x[ok], y[ok])
    return {"rho": float(r.statistic), "p": float(r.pvalue), "n": int(ok.sum())}


__all__ = [
    "BINARIZE_REL_EPS", "CLASS_NAMES", "EXCLUDED_LAYERS", "GRID", "KneeThreshold",
    "PATCH_HIT_COV", "SelectionFrequency", "binarize_above_mean", "cell_list", "combine_maps",
    "eligible_cells", "gaussian_smooth", "head_average", "image_attention_sum", "map_scores",
    "max_curvature_threshold", "rank_cells", "selection_frequency", "spatial_entropy",
    "spatial_entropy_batch", "spearman",
]
