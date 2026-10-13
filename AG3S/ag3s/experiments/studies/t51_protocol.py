"""T51 — the head-selection protocol P1–P6, pure numpy (no MuJoCo, no GT reading, no storage).

`T51.task.md` fixes the procedure *before* any result is seen; this module is that procedure, written
so every rule can be pinned by a unit test on synthetic data (`tests/ag3s/test_t51_protocol.py`).

Candidates Ω = {(query, layer, head)}: query = an action-token aggregation at denoise 0 (``last`` ·
``mean`` · ``first``; T49's noun rows can be added as further queries), layer L2–17 (L0 · L1 excluded,
Kang et al. §4), head 0–7. A cell is written ``"<query>:L<l>H<h>"`` (`Cell.id`).

P1 filter (label-free)       `filter_summary` + `filter_reasons` + `filter_eligible`
P2 calibration score         2D (`record_2d_scores`) → 2D→3D preselection (`preselect`) → 3D rates
                             (computed by the replay, ranked here with `rank_order`)
P3 stability                 `bootstrap_ranking` (B = 1000, fixed seed; and the exact enumeration of every
                             resample when there are few units). ``groups=`` resamples **scenes**: the
                             runs of one scene are drawn together (`T51.task.md` 판정 보완, 2026-10-12)
P4 k                         `cv_scores` · `choose_k` · `nested_cv` · `required_pairs` (leave-one-unit-
                             out; the smaller k wins when the difference's 95 % CI contains 0).
                             `grouped_cv` turns per-record rank / eval functions into per-scene ones
                             (leave-one-scene-out), `expand_pairs` maps its needs back to records
P6 random cells              `random_cell_sets`

Everything is deterministic for a given seed. Ties are broken by fixed, documented rules (`rank_order`).
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import re
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------ pre-registered constants

#: P1 (`T51.task.md` "Protocol"). Strict inequalities exactly as written there:
#: (i) image mass median < 0.01, (ii) modal-argmax fraction > 0.5 AND top-1 share > 0.4,
#: (iii) top-1 share < 0.03 AND largest component > 15 % of the 16x16 grid.
FILTER_THRESHOLDS = {
    "mass_median_lt": 0.01,
    "sink_modal_gt": 0.5,
    "sink_top1_gt": 0.4,
    "broad_top1_lt": 0.03,
    "broad_grid_gt": 0.15,
}
FILTER_REASONS = ("low_mass", "fixed_patch_sink", "broad_map")
#: P2 2D -> 3D preselection: the top-16 by 2D lift (plus the forced cells).
PRESELECT_TOP = 16
#: P4 candidates and P3 / P4 bootstrap size.
K_CHOICES = (1, 2, 3)
N_BOOT = 1000
#: Fixed seeds (one per use so adding a draw in one place never shifts another).
SEED_BOOTSTRAP = 5103
SEED_KSEL = 5104
SEED_RANDOM_CELLS = 5106
#: P6: number of random cell draws.
N_RANDOM = 20
#: Above this many ordered resamples the exact bootstrap enumeration is skipped (3 episodes = 27).
EXACT_MAX_RESAMPLES = 100_000

LAYERS = tuple(range(2, 18))
HEADS = tuple(range(8))
ACTION_QUERIES = ("last", "mean", "first")


# ----------------------------------------------------------------------------------- cells


@dataclasses.dataclass(frozen=True, order=True)
class Cell:
    query: str
    layer: int
    head: int

    @property
    def id(self) -> str:
        return f"{self.query}:L{self.layer}H{self.head}"

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.id


_CELL_RE = re.compile(r"^([A-Za-z0-9_]+):L(\d+)H(\d+)$")


def parse_cell(text: str) -> Cell:
    m = _CELL_RE.match(str(text).strip())
    if not m:
        raise ValueError(f"not a cell id {text!r} (expected '<query>:L<l>H<h>')")
    return Cell(m.group(1), int(m.group(2)), int(m.group(3)))


def omega(queries: Sequence[str] = ACTION_QUERIES, layers: Sequence[int] = LAYERS,
          heads: Sequence[int] = HEADS) -> list[Cell]:
    """Ω in a fixed order: query-major, then layer, then head (the tie-break index)."""
    return [Cell(q, int(l), int(h)) for q in queries for l in layers for h in heads]


def cell_positions(cells: Sequence[Cell], queries: Sequence[str],
                   layers: Sequence[int] = LAYERS, heads: Sequence[int] = HEADS
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(query index, layer index, head index)`` of each cell into a ``(Q, L, H)`` grid."""
    qi = np.asarray([list(queries).index(c.query) for c in cells], np.int64)
    li = np.asarray([list(layers).index(c.layer) for c in cells], np.int64)
    hi = np.asarray([list(heads).index(c.head) for c in cells], np.int64)
    return qi, li, hi


def combo_id(cells: Iterable) -> str:
    """Canonical id of a raw sum of cells (sorted, so the same set always has the same id)."""
    ids = sorted(c.id if isinstance(c, Cell) else str(c) for c in cells)
    if not ids:
        raise ValueError("empty combination")
    return ids[0] if len(ids) == 1 else "sum[" + "+".join(ids) + "]"


def combo_cells(cid: str) -> list[Cell]:
    if cid.startswith("sum[") and cid.endswith("]"):
        return [parse_cell(p) for p in cid[4:-1].split("+")]
    return [parse_cell(cid)]


def safe_name(source_id: str) -> str:
    """A file-name-safe form of a source id (``last:L8H2`` -> ``last_L8H2``)."""
    s = source_id.replace("sum[", "sum__").replace("]", "").replace("+", "__").replace(":", "_")
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", s):
        raise ValueError(f"cannot make a file name from {source_id!r}")
    return s


# ------------------------------------------------------------------------------- P1 filter


def filter_summary(per_frame: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per (cell, camera) summary over frames of `t47b_sink.map_stats` output (frame axis first).

    Returns ``mass_median`` (median ``S_cam``), ``modal_fraction`` (share of frames at the most
    frequent argmax patch, T47b #5 (a)), ``top1_share_mean`` (T47b #5 (b)), ``largest_share_of_grid_mean``
    (T47b #5 (c)) and the other `t47b_sink.summarize_cells` fields.
    """
    from AG3S.ag3s.experiments.studies.t47b_sink import summarize_cells

    out = dict(summarize_cells(per_frame))
    s = np.asarray(per_frame["S_cam"], np.float64)
    out["mass_median"] = np.median(s, axis=0)
    return out


def filter_reasons(summary: Mapping[str, np.ndarray],
                   thresholds: Mapping[str, float] = FILTER_THRESHOLDS) -> dict[str, np.ndarray]:
    """Boolean arrays (same shape as the summary fields) for each exclusion reason.

    NaN statistics never trigger (a comparison with NaN is False): a cell whose maps are all zero has
    a NaN top-1 share, but its zero mass already triggers ``low_mass``.
    """
    t = dict(FILTER_THRESHOLDS, **dict(thresholds))
    with np.errstate(invalid="ignore"):
        mass = np.asarray(summary["mass_median"], np.float64)
        modal = np.asarray(summary["modal_fraction"], np.float64)
        top1 = np.asarray(summary["top1_share_mean"], np.float64)
        grid = np.asarray(summary["largest_share_of_grid_mean"], np.float64)
        low = mass < t["mass_median_lt"]
        sink = (modal > t["sink_modal_gt"]) & (top1 > t["sink_top1_gt"])
        broad = (top1 < t["broad_top1_lt"]) & (grid > t["broad_grid_gt"])
    return {"low_mass": low, "fixed_patch_sink": sink, "broad_map": broad,
            "any": low | sink | broad}


def filter_eligible(reasons: Mapping[str, np.ndarray], *, camera_axis: int = -1) -> np.ndarray:
    """A cell is eligible when **no** camera triggers any reason (`T51.task.md`: "한 camera 라도
    해당하면 그 camera 에서 제외, 공통 cell 이면 cell 제외")."""
    return ~np.any(np.asarray(reasons["any"], bool), axis=camera_axis)


# ------------------------------------------------------------------------------ P2 2D scores


def record_2d_scores(block: np.ndarray, class_cov: np.ndarray, keep: np.ndarray, *,
                     target_index: int = 0) -> dict[str, np.ndarray]:
    """2D scores of every cell of one record over its kept (frame, camera) samples.

    `block` ``(F, L, H, C, 16, 16)`` raw maps; `class_cov` ``(F, C, K, 16, 16)`` per-patch class shares
    (GT cache, or ``[self, 1 - self]`` for the self label); `keep` ``(F, C)``. Returns ``(L, H)`` arrays:
    ``lift`` / ``peak_on_target`` / ``target_mass`` = means over the kept samples (NaN-skipping, the T47
    `_aggregate` rule), and ``n_kept`` (scalar). Uses `t47_head_metrics.map_scores` (T47 definitions).
    """
    from AG3S.ag3s.experiments.studies.t47_head_metrics import map_scores

    a = np.asarray(block)
    f, n_l, n_h, n_c = a.shape[:4]
    cov = np.asarray(class_cov, np.float64)
    k = np.asarray(keep, bool)
    if cov.shape[:2] != (f, n_c) or k.shape != (f, n_c):
        raise ValueError(f"block {a.shape} / class_cov {cov.shape} / keep {k.shape} do not line up")
    fi, ci = np.nonzero(k)
    out = {m: np.full((n_l, n_h), np.nan) for m in ("lift", "peak_on_target", "target_mass")}
    out["n_kept"] = np.asarray(len(fi))
    if not len(fi):
        return out
    cov_k = cov[fi, ci]  # (N, K, 16, 16)
    for l in range(n_l):
        for h in range(n_h):
            maps = np.asarray(a[fi, l, h, ci], np.float64)
            sc = map_scores(maps, cov_k, target_index=target_index)
            for m in ("lift", "peak_on_target", "target_mass"):
                v = sc[m]
                fin = np.isfinite(v)
                out[m][l, h] = float(v[fin].mean()) if fin.any() else np.nan
    return out


def unit_mean(per_unit: np.ndarray, axis: int = 0) -> np.ndarray:
    """Unweighted mean over episodes (NaN episodes skipped; all-NaN -> NaN, no warning)."""
    x = np.asarray(per_unit, np.float64)
    fin = np.isfinite(x)
    n = fin.sum(axis=axis)
    s = np.where(fin, x, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def group_means(per_unit: np.ndarray, groups: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """``(group names, (G, ...) per-group means)``: the NaN-skipping `unit_mean` of each group's rows
    (axis 0), groups in first-appearance order (`group_members`). A group whose rows are all NaN for a
    column stays NaN there (and is then skipped by the outer `unit_mean`)."""
    x = np.asarray(per_unit, np.float64)
    if len(groups) != x.shape[0]:
        raise ValueError(f"groups has {len(groups)} entries for {x.shape[0]} rows")
    names, members = group_members([str(g) for g in groups])
    return names, np.stack([unit_mean(x[m]) for m in members])


def aggregate_mean(per_unit: np.ndarray, groups: Optional[Sequence[str]] = None) -> np.ndarray:
    """The calibration point estimate over axis 0.

    ``groups is None``: `unit_mean` over the rows (records) — T51's original (record-mean) estimator.
    Otherwise the **group (scene / task) mean**: rows of a group are averaged first, then the groups
    with equal weight (`T51.task.md` 판정 보완 → 집계 단위)."""
    if groups is None:
        return unit_mean(per_unit)
    return unit_mean(group_means(per_unit, groups)[1])


# --------------------------------------------------------------------------------- ranking


def rank_order(score: np.ndarray, tiebreak: Optional[np.ndarray] = None,
               eligible: Optional[np.ndarray] = None) -> np.ndarray:
    """Indices of the eligible cells, best first.

    Order: higher `score`; then higher `tiebreak`; then the lower index (Ω order). NaN sorts below
    every number (a cell with no defined score can only win when nothing else is defined).
    """
    s = np.asarray(score, np.float64)
    n = s.size
    t = np.zeros(n) if tiebreak is None else np.asarray(tiebreak, np.float64)
    e = np.ones(n, bool) if eligible is None else np.asarray(eligible, bool)
    s_key = np.where(np.isfinite(s), s, -np.inf)
    t_key = np.where(np.isfinite(t), t, -np.inf)
    order = np.lexsort((np.arange(n), -t_key, -s_key))
    return order[e[order]]


def preselect(score2d: np.ndarray, tie2d: np.ndarray, eligible: np.ndarray, *,
              n_top: int = PRESELECT_TOP, forced: Sequence[int] = ()) -> dict:
    """The 2D -> 3D preselection: the top `n_top` eligible cells by 2D lift (tie: 2D peak, then index)
    plus the `forced` cells (L8H2 and T47's oracle top-3). Forced cells are 3D-scored even when the
    filter excluded them; they stay out of the ranking then (``forced_ineligible``)."""
    order = rank_order(score2d, tie2d, eligible)
    top = [int(i) for i in order[: int(n_top)]]
    el = np.asarray(eligible, bool)
    forced = [int(i) for i in forced]
    ranked = sorted(set(top) | {i for i in forced if el[i]})
    return {"top": top, "forced": forced,
            "forced_ineligible": [i for i in forced if not el[i]],
            "forced_already_in_top": [i for i in forced if i in top],
            "ranked_set": ranked,
            "scored_set": sorted(set(top) | set(forced))}


# ------------------------------------------------------------------------------ P3 bootstrap


def _resample_means(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """``x`` (E, N), ``idx`` (B, E) -> (B, N) NaN-skipping means over the drawn episodes."""
    d = x[idx]  # (B, E, N)
    fin = np.isfinite(d)
    n = fin.sum(axis=1)
    s = np.where(fin, d, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _weighted_means(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``x`` (E, N), ``weights`` (B, E) multiplicities -> (B, N) NaN-skipping weighted means.

    With integer multiplicities this is `_resample_means` of the corresponding index draws."""
    fin = np.isfinite(x)
    num = np.asarray(weights, np.float64) @ np.where(fin, x, 0.0)
    den = np.asarray(weights, np.float64) @ fin.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)


def group_members(groups: Sequence[str]) -> tuple[list[str], list[np.ndarray]]:
    """Group names in first-appearance order and the member (record) indices of each."""
    names: list[str] = []
    for g in groups:
        if str(g) not in names:
            names.append(str(g))
    members = [np.asarray([i for i, g in enumerate(groups) if str(g) == n], np.int64) for n in names]
    return names, members


def _group_weights(draws: np.ndarray, members: Sequence[np.ndarray], n_units: int) -> np.ndarray:
    """``draws`` (B, G) group indices -> (B, E) record multiplicities (every run of a drawn scene)."""
    m = np.zeros((len(members), n_units))
    for g, idx in enumerate(members):
        m[g, idx] = 1.0
    counts = np.zeros((draws.shape[0], len(members)))
    for g in range(len(members)):
        counts[:, g] = (draws == g).sum(axis=1)
    return counts @ m


def _rank_stats(means: np.ndarray, ties: np.ndarray, eligible: np.ndarray, top_n: int) -> dict:
    """Equal-weight resamples: integer counts divided once (exact 0 / 1 when unanimous)."""
    b, n = means.shape
    c1 = np.zeros(n, np.int64)
    ctop = np.zeros(n, np.int64)
    tied = 0
    for r in range(b):
        order = rank_order(means[r], ties[r], eligible)
        if not order.size:
            continue
        c1[order[0]] += 1
        ctop[order[:top_n]] += 1
        if order.size > 1 and np.isfinite(means[r, order[0]]) and \
                means[r, order[0]] == means[r, order[1]]:
            tied += 1
    return {"p_rank1": c1 / b, f"p_top{top_n}": ctop / b, "rank1_tied_rate": tied / b}


def bootstrap_ranking(score: np.ndarray, tiebreak: np.ndarray, eligible: np.ndarray, *,
                      n_boot: int = N_BOOT, seed: int = SEED_BOOTSTRAP, top_n: int = 3,
                      ci: float = 0.95, exact_max: int = EXACT_MAX_RESAMPLES,
                      groups: Optional[Sequence[str]] = None, group_mean: bool = False) -> dict:
    """Episode bootstrap of the P2 ranking.

    `score` / `tiebreak` are ``(E, N)`` per-episode values (primary 3D rate / 2D lift) of the N cells.
    Each resample draws E episodes with replacement (``numpy.random.default_rng(seed).integers``),
    averages each cell over the drawn episodes, and ranks with `rank_order` (tie-break on the
    resample's mean `tiebreak`, then Ω index). Returns per cell ``p_rank1``, ``p_top{top_n}``, the
    full-sample ``mean``, the percentile ``ci_lo`` / ``ci_hi`` of the resampled means, and
    ``rank1_tied_rate`` (resamples whose winner tied with the runner-up on the primary score).

    With few episodes the Monte-Carlo estimate is a draw from a small discrete distribution, so the
    exact distribution over all ``E**E`` ordered resamples is returned too (``exact``) when
    ``E**E <= exact_max``. ``n_distinct_resamples`` counts the distinct multisets.

    ``groups`` (length E, e.g. the scene id of each record): **cluster bootstrap** — each resample
    draws G groups with replacement (``integers(0, G, (B, G))``) and every record of a drawn group
    enters (a group drawn twice enters twice); the cell mean is the NaN-skipping mean over those
    records, i.e. the same record-level estimator as the full sample. Exact enumeration is over the
    ``G**G`` ordered group draws. With one record per group the draws, and so the result, equal the
    ungrouped call with the same seed.

    ``group_mean=True`` (needs ``groups``; `T51.task.md` 판정 보완 → 집계 단위): the estimator is the
    **mean of group means** — each group's records are averaged first (`group_means`), and a resample's
    cell mean is the NaN-skipping mean over the drawn groups (a group drawn twice counts twice). The
    draws are the same as with ``group_mean=False`` (same seed); ``mean`` is the full-sample group mean.
    """
    x = np.asarray(score, np.float64)
    t = np.asarray(tiebreak, np.float64)
    if x.ndim != 2 or t.shape != x.shape:
        raise ValueError(f"score {x.shape} / tiebreak {t.shape} must be the same (E, N)")
    e_n, n = x.shape
    el = np.asarray(eligible, bool)
    if el.shape != (n,):
        raise ValueError(f"eligible {el.shape} != ({n},)")
    if group_mean and groups is None:
        raise ValueError("group_mean=True needs groups")
    if groups is not None:
        return _bootstrap_grouped(x, t, el, groups, n_boot=n_boot, seed=seed, top_n=top_n, ci=ci,
                                  exact_max=exact_max, group_mean=group_mean)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, e_n, size=(int(n_boot), e_n))
    means = _resample_means(x, idx)
    ties = _resample_means(t, idx)
    stats = _rank_stats(means, ties, el, top_n)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ci_lo = np.nanpercentile(means, lo_q, axis=0)
            ci_hi = np.nanpercentile(means, hi_q, axis=0)
    out = {"n_episodes": e_n, "n_boot": int(n_boot), "seed": int(seed), "ci": ci,
           "mean": unit_mean(x), "ci_lo": ci_lo, "ci_hi": ci_hi,
           "n_distinct_resamples": math.comb(2 * e_n - 1, e_n), **stats}
    if e_n ** e_n <= exact_max:
        all_idx = np.asarray(list(itertools.product(range(e_n), repeat=e_n)), np.int64)
        em, et = _resample_means(x, all_idx), _resample_means(t, all_idx)
        ex = _rank_stats(em, et, el, top_n)
        out["exact"] = {"n_ordered_resamples": int(len(all_idx)), **ex}
    else:
        out["exact"] = None
    return out


def _percentile_ci(means: np.ndarray, ci: float) -> tuple[np.ndarray, np.ndarray]:
    import warnings

    lo_q, hi_q = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(means, lo_q, axis=0), np.nanpercentile(means, hi_q, axis=0)


def _bootstrap_grouped(x: np.ndarray, t: np.ndarray, el: np.ndarray, groups: Sequence[str], *,
                       n_boot: int, seed: int, top_n: int, ci: float, exact_max: int,
                       group_mean: bool = False) -> dict:
    e_n = x.shape[0]
    if len(groups) != e_n:
        raise ValueError(f"groups has {len(groups)} entries for {e_n} records")
    names, members = group_members([str(g) for g in groups])
    g_n = len(names)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, g_n, size=(int(n_boot), g_n))
    if group_mean:  # mean of group means: resample the (G, N) group-mean rows with the same draws
        xg, tg = group_means(x, groups)[1], group_means(t, groups)[1]

        def resampled(d):
            return _resample_means(xg, d), _resample_means(tg, d)
        full_mean = unit_mean(xg)
    else:  # record-level mean over every run of the drawn groups (T51's original estimator)
        def resampled(d):
            w = _group_weights(d, members, e_n)
            return _weighted_means(x, w), _weighted_means(t, w)
        full_mean = unit_mean(x)
    means, ties = resampled(draws)
    stats = _rank_stats(means, ties, el, top_n)
    ci_lo, ci_hi = _percentile_ci(means, ci)
    out = {"n_episodes": e_n, "n_groups": g_n, "groups": names,
           "group_members": {nm: [int(i) for i in m] for nm, m in zip(names, members)},
           "unit": "group", "n_boot": int(n_boot), "seed": int(seed), "ci": ci,
           "mean": full_mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
           "n_distinct_resamples": math.comb(2 * g_n - 1, g_n), **stats}
    if group_mean:
        out["aggregate"] = "group_mean"
    if g_n ** g_n <= exact_max:
        all_draws = np.asarray(list(itertools.product(range(g_n), repeat=g_n)), np.int64)
        ex = _rank_stats(*resampled(all_draws), el, top_n)
        out["exact"] = {"n_ordered_resamples": int(len(all_draws)), **ex}
    else:
        out["exact"] = None
    return out


# ------------------------------------------------------------------------------- P4 k / CV


def loeo_folds(units: Sequence[str]) -> list[tuple[tuple[str, ...], str]]:
    """Leave-one-episode-out folds: ``[(train units, test unit), ...]`` in `units` order."""
    u = list(units)
    if len(set(u)) != len(u):
        raise ValueError(f"duplicate units {u}")
    return [(tuple(x for x in u if x != t), t) for t in u]


RankFn = Callable[[Sequence[str]], Sequence[str]]
EvalFn = Callable[[str, str], float]


def topk(rank_fn: RankFn, train: Sequence[str], k: int) -> str:
    """The combo id of the top-`k` cells ranked on `train` only."""
    ranked = list(rank_fn(tuple(train)))
    if len(ranked) < k:
        raise ValueError(f"only {len(ranked)} ranked cells for k={k}")
    return combo_id(ranked[:k])


def cv_scores(units: Sequence[str], ks: Sequence[int], rank_fn: RankFn, eval_fn: EvalFn
              ) -> dict:
    """Leave-one-episode-out: for each held-out unit, rank on the others and evaluate top-k on it.

    Returns ``{"units": [...], "scores": {k: [score per unit]}, "combos": {k: [combo per unit]}}``.
    `rank_fn` never sees the held-out unit (the folds are built by `loeo_folds`).
    """
    folds = loeo_folds(units)
    scores = {int(k): [] for k in ks}
    combos = {int(k): [] for k in ks}
    for train, test in folds:
        for k in ks:
            cid = topk(rank_fn, train, int(k))
            combos[int(k)].append(cid)
            scores[int(k)].append(float(eval_fn(cid, test)))
    return {"units": list(units), "scores": scores, "combos": combos}


def bootstrap_mean_ci(values: Sequence[float], *, n_boot: int = N_BOOT, seed: int = SEED_KSEL,
                      ci: float = 0.95) -> tuple[float, float, float]:
    """``(mean, lo, hi)``: percentile bootstrap over units of the NaN-skipping mean."""
    v = np.asarray(values, np.float64)
    if not np.isfinite(v).any():
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, v.size, size=(int(n_boot), v.size))
    m = _resample_means(v[:, None], idx)[:, 0]
    m = m[np.isfinite(m)]
    return (float(unit_mean(v)), float(np.percentile(m, (1 - ci) / 2 * 100)),
            float(np.percentile(m, (1 + ci) / 2 * 100)))


def choose_k(scores: Mapping[int, Sequence[float]], *, n_boot: int = N_BOOT,
             seed: int = SEED_KSEL, ci: float = 0.95) -> dict:
    """The simplicity rule (`T51.task.md` P4, fixed before results).

    ``best`` = the k with the highest mean CV score (ties -> smaller k). Then, from the smallest k
    upward (k < best), the first k whose paired difference ``score[best] - score[k]`` (per unit) has a
    bootstrap 95 % CI with lower bound <= 0 is chosen ("the difference is within the CI"). Otherwise
    ``best``. The differences are paired by unit (same held-out episode).
    """
    ks = sorted(int(k) for k in scores)
    means = {k: float(unit_mean(np.asarray(scores[k], np.float64))) for k in ks}
    best = max(ks, key=lambda k: (np.nan_to_num(means[k], nan=-np.inf), -k))
    tests = []
    chosen = best
    for k in ks:
        if k >= best:
            break
        d = np.asarray(scores[best], np.float64) - np.asarray(scores[k], np.float64)
        m, lo, hi = bootstrap_mean_ci(d, n_boot=n_boot, seed=seed, ci=ci)
        within = bool(np.isfinite(lo) and lo <= 0.0) or not np.isfinite(m)
        tests.append({"k": k, "vs_best": best, "diff_mean": m, "diff_ci": [lo, hi],
                      "within_ci": within})
        if within:
            chosen = k
            break
    return {"chosen_k": chosen, "best_k": best, "means": means, "tests": tests,
            "rule": "smallest k < best whose paired difference CI (bootstrap over units) has "
                    "lower bound <= 0; else best"}


def nested_cv(units: Sequence[str], ks: Sequence[int], rank_fn: RankFn, eval_fn: EvalFn, *,
              n_boot: int = N_BOOT, seed: int = SEED_KSEL) -> dict:
    """P4 in full.

    * ``cv``: `cv_scores` over all units; ``choice`` = `choose_k` on it -> the k used by the protocol.
    * ``nested``: for each outer unit r, `cv_scores` + `choose_k` on the other units only (inner),
      then the outer score is the outer fold's top-``k_inner`` evaluated on r (= ``cv`` at that k,
      since the outer fold trains on the same units). ``nested_mean`` estimates the whole procedure.
    """
    cv = cv_scores(units, ks, rank_fn, eval_fn)
    choice = choose_k(cv["scores"], n_boot=n_boot, seed=seed)
    outer = []
    for i, (train, test) in enumerate(loeo_folds(units)):
        entry = {"test": test, "train": list(train)}
        if len(train) >= 2:
            inner = cv_scores(train, ks, rank_fn, eval_fn)
            ic = choose_k(inner["scores"], n_boot=n_boot, seed=seed)
            entry.update(inner=inner, inner_choice=ic, k_inner=ic["chosen_k"])
        else:  # one training unit: no inner validation is possible -> the smallest k
            entry.update(inner=None, inner_choice=None, k_inner=min(int(k) for k in ks),
                         note="one training unit: smallest k by the simplicity rule")
        k_in = int(entry["k_inner"])
        entry["combo"] = cv["combos"][k_in][i]
        entry["score"] = cv["scores"][k_in][i]
        outer.append(entry)
    return {"cv": cv, "choice": choice, "nested": outer,
            "nested_mean": float(unit_mean(np.asarray([o["score"] for o in outer], np.float64))),
            "k_inner_per_fold": {o["test"]: o["k_inner"] for o in outer}}


def required_pairs(units: Sequence[str], ks: Sequence[int], rank_fn: RankFn) -> set[tuple[str, str]]:
    """Every ``(combo id, unit)`` evaluation `nested_cv` will ask for — known before any combo is
    run, because the rankings depend only on the single-cell scores (outer folds of `cv_scores` and
    the inner folds of every outer training set)."""
    need: set[tuple[str, str]] = set()

    def add(us):
        for train, test in loeo_folds(us):
            for k in ks:
                need.add((topk(rank_fn, train, int(k)), test))

    add(units)
    for train, _ in loeo_folds(units):
        if len(train) >= 2:
            add(train)
    return need


def grouped_cv(records: Sequence[str], group_of: Mapping[str, str], rank_fn: RankFn,
               eval_fn: EvalFn) -> tuple[list[str], RankFn, EvalFn, dict[str, list[str]]]:
    """Leave-one-**scene**-out from per-record functions.

    Returns ``(scenes, rank_scenes, eval_scene, records_of)``: the units of `nested_cv` /
    `required_pairs` become the scenes (first-appearance order of `records`); ranking on a set of
    scenes ranks on all their records (in `records` order, so the training tuple is the same one the
    2D preselection was computed for, `group_train_sets`); the score of a test scene is the
    unweighted mean of its records' scores (NaN-skipping). A scene's runs are never split between
    train and test.
    """
    recs = list(records)
    names, members = group_members([group_of[r] for r in recs])
    records_of = {nm: [recs[i] for i in m] for nm, m in zip(names, members)}

    def rank_scenes(train_scenes):
        keep = set(train_scenes)
        return rank_fn(tuple(r for r in recs if group_of[r] in keep))

    def eval_scene(cid, scene):
        return float(unit_mean(np.asarray([eval_fn(cid, r) for r in records_of[scene]], np.float64)))

    return names, rank_scenes, eval_scene, records_of


def expand_pairs(pairs: Iterable[tuple[str, str]], records_of: Mapping[str, Sequence[str]]
                 ) -> set[tuple[str, str]]:
    """``(combo, scene)`` needs -> ``(combo, record)`` for every record of the scene."""
    return {(c, r) for c, s in pairs for r in records_of[s]}


def group_train_sets(records: Sequence[str], group_of: Mapping[str, str]) -> list[tuple[str, ...]]:
    """Every record tuple the scene-level nested CV ranks on (outer leave-one-scene-out trains and
    their inner trains), each in `records` order — the tuples `grouped_cv`'s ranker passes on."""
    recs = list(records)
    names, _ = group_members([group_of[r] for r in recs])
    out: list[tuple[str, ...]] = []

    def as_records(scenes):
        keep = set(scenes)
        return tuple(r for r in recs if group_of[r] in keep)

    for train, _ in loeo_folds(names):
        out.append(as_records(train))
        if len(train) >= 2:
            out += [as_records(t2) for t2, _ in loeo_folds(train)]
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


# ------------------------------------------------------- fallback: annotated frames needed


def frames_needed_curve(frame_scores: np.ndarray, eligible: np.ndarray, reference_top: Sequence[int],
                        ns: Sequence[int], *, n_boot: int = N_BOOT, seed: int = SEED_BOOTSTRAP,
                        tiebreak: Optional[np.ndarray] = None) -> list[dict]:
    """If a small human-annotated set replaced the full GT: for each budget n, draw n calibration
    frames without replacement (B times), rank the cells by their mean per-frame score on those frames
    (`rank_order`, tie `tiebreak` then Ω index) and report how often the winner equals the full-data
    winner ``reference_top[0]`` and how often it lies in ``reference_top`` (e.g. the full top-3).

    `frame_scores` ``(n_frames, N)`` per-frame primary score (0/1 identity) of every cell.
    """
    x = np.asarray(frame_scores, np.float64)
    n_frames = x.shape[0]
    el = np.asarray(eligible, bool)
    ref = [int(i) for i in reference_top]
    rng = np.random.default_rng(int(seed))
    out = []
    for n in ns:
        n = int(min(n, n_frames))
        same = within = 0
        for _ in range(int(n_boot)):
            idx = rng.choice(n_frames, size=n, replace=False)
            order = rank_order(unit_mean(x[idx]), tiebreak, el)
            if order.size:
                same += int(order[0] == ref[0])
                within += int(order[0] in ref)
        out.append({"n_frames": n, "p_same_top1": same / n_boot, "p_top1_in_reference": within / n_boot})
    return out


# ---------------------------------------------------------------------------- P6 random


def random_cell_sets(eligible: Sequence[str], *, n: int = N_RANDOM, k: int = 1,
                     seed: int = SEED_RANDOM_CELLS) -> list[str]:
    """`n` random draws of `k` distinct cells from the eligible cell ids (draws may repeat a set;
    within a draw no cell repeats). Returns combo ids, in draw order."""
    pool = sorted(str(c) for c in eligible)
    if len(pool) < k:
        raise ValueError(f"{len(pool)} eligible cells for k={k}")
    rng = np.random.default_rng(int(seed))
    return [combo_id([pool[i] for i in rng.choice(len(pool), size=int(k), replace=False)])
            for _ in range(int(n))]


__all__ = [
    "ACTION_QUERIES", "Cell", "EXACT_MAX_RESAMPLES", "FILTER_REASONS", "FILTER_THRESHOLDS",
    "HEADS", "K_CHOICES", "LAYERS", "N_BOOT", "N_RANDOM", "PRESELECT_TOP", "SEED_BOOTSTRAP",
    "SEED_KSEL", "SEED_RANDOM_CELLS", "bootstrap_mean_ci", "bootstrap_ranking", "cell_positions",
    "choose_k", "combo_cells", "combo_id", "cv_scores", "expand_pairs", "filter_eligible",
    "filter_reasons", "filter_summary", "frames_needed_curve", "group_members", "group_train_sets",
    "grouped_cv", "loeo_folds", "nested_cv", "omega", "parse_cell", "preselect",
    "random_cell_sets", "rank_order", "record_2d_scores", "required_pairs", "safe_name", "topk",
    "unit_mean",
]
