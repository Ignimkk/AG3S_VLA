"""T56 — selection by self score, regret against GT, Kendall τ-b, and the paired episode bootstrap.

`T56.task.md` 사전 고정 3 · 4, as code (pure numpy):

Selection (fixed before the test)
    score(c) = mean over the **event episodes** (episodes where the variant has a label) of the
    candidate's episode self score. Episodes without an event carry no self score; they are counted
    and left out of the selection — never chosen by GT.
    Ties: candidates with ``score >= max − resolution`` (dev-fixed) form the tie set. Inside it:
    (1) the larger S1 coverage component (same episode averaging; compared exactly, 1e-12), then
    (2) the lexicographically first candidate name. A 2D lift never enters.
Regret (primary)
    GT value of a candidate = mean over episodes of its episode-mean ``identity_correct`` (T54
    definition; no output = 0). ``regret = max_c GT(c) − GT(selected)``; ``tie_regret = max_c GT(c) −
    mean_{c ∈ tie set} GT(c)`` (the tie-set average). GT episodes: ``all`` (every episode of the set —
    the setting chosen is used everywhere; primary) and ``event`` (event episodes only).
    Auxiliary: the same "loss" for purity (with the selected candidate's ``no_target``) and coverage.
Kendall τ-b (diagnostic)
    Per event episode between the candidates' self episode scores and their GT episode
    ``identity_correct``; ties allowed (exact, |Δ| <= 1e-12 — the selection resolution does not enter).
    When either side is all tied (or < 2 candidates are finite) τ-b is **undefined** (``None``, written
    ``"undefined:<reason>"``, `TAU_REASONS`) — never 0. Means are over defined episodes and travel with
    the number of undefined ones (by reason; ``no_event`` separately).
Tie rate
    ``tie_size`` = size of the selection's tie set (point); ``tie_rate_boot`` = share of bootstrap draws
    whose selection was tied (T51's "rank-1 tie rate"; draws without an event episode left out);
    `episode_ties` = the same per event episode (share of episodes whose own top is tied).
Bootstrap
    Episodes resampled with replacement (B = 10000, seed 5607, percentile 95 %), the **same draws** for
    every variant, group, condition and phase (paired). Each draw redoes the selection with the drawn
    episodes' multiplicities.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import numpy as np

N_BOOT = 10000
SEED = 5607
CI = (2.5, 97.5)
TB_EPS = 1e-12
GT_METRICS = ("identity", "purity", "coverage", "no_target")
LOSS_METRICS = ("identity", "purity", "coverage")


TAU_UNDEFINED = "undefined"
#: Why a τ-b is undefined (``defined`` = a number). Never replaced by 0.
TAU_REASONS = ("defined", "no_event", "fewer_than_2", "self_all_tied", "gt_all_tied", "both_all_tied")


def _sign_tol(d: np.ndarray) -> np.ndarray:
    """sign with |d| <= TB_EPS counted as a tie (floating residue of equal means)."""
    return np.where(np.abs(d) <= TB_EPS, 0.0, np.sign(d))


def kendall_tau_b_status(x: Sequence[float], y: Sequence[float]) -> tuple[Optional[float], str]:
    """``(τ-b, reason)``. Ties allowed (|Δ| <= 1e-12 is a tie). τ-b is None — **undefined, never 0** —
    when fewer than 2 finite pairs remain or one side is all tied (`TAU_REASONS`)."""
    a = np.asarray(x, np.float64)
    b = np.asarray(y, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 2:
        return None, "fewer_than_2"
    iu = np.triu_indices(n, 1)
    sa = _sign_tol(a[:, None] - a[None, :])[iu]
    sb = _sign_tol(b[:, None] - b[None, :])[iu]
    n0 = len(sa)
    t1 = float((sa == 0).sum())
    t2 = float((sb == 0).sum())
    if t1 == n0 or t2 == n0:
        return None, ("both_all_tied" if t1 == n0 and t2 == n0
                      else "self_all_tied" if t1 == n0 else "gt_all_tied")
    s = float((sa * sb).sum())
    return s / math.sqrt((n0 - t1) * (n0 - t2)), "defined"


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """τ-b with ties; None when undefined (all tied on one side, or fewer than 2 finite pairs)."""
    return kendall_tau_b_status(x, y)[0]


def episode_draws(n_ep: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """``(B, n_ep)`` multiplicity of each episode in each draw (with replacement)."""
    idx = np.random.default_rng(seed).integers(0, n_ep, size=(n_boot, n_ep))
    W = np.zeros((n_boot, n_ep), np.float64)
    rows = np.repeat(np.arange(n_boot), n_ep)
    np.add.at(W, (rows, idx.reshape(-1)), 1.0)
    return W


def _wmean(W: np.ndarray, M: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
    """``(B, n_c)`` weighted nan-mean of `M` (n_ep, n_c) over episodes with weights `W` (B, n_ep);
    `rows` (n_ep,) bool restricts the episodes."""
    M = np.asarray(M, np.float64)
    fin = np.isfinite(M)
    if rows is not None:
        fin = fin & np.asarray(rows, bool)[:, None]
    num = W @ np.where(fin, M, 0.0)
    den = W @ fin.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _clean(x: np.ndarray) -> np.ndarray:
    """Floating-point residue of a mean (|x| < 1e-12) → exactly 0."""
    x = np.asarray(x, np.float64)
    return np.where(np.abs(x) < 1e-12, 0.0, x)


def select_rows(score: np.ndarray, tb: np.ndarray, resolution: float) -> tuple[np.ndarray, np.ndarray]:
    """Per row (draw): ``(selected index, tie mask)``. `score` / `tb` (B, n_c); candidates are in name
    order (so the first True is the lexicographic tie-break). Rows with no finite score → -1."""
    s = np.asarray(score, np.float64)
    t = np.where(np.isfinite(tb), tb, -np.inf)
    fin = np.isfinite(s)
    mx = np.max(np.where(fin, s, -np.inf), axis=1, keepdims=True)
    tie = fin & (s >= mx - float(resolution) - 1e-12)
    tmax = np.max(np.where(tie, t, -np.inf), axis=1, keepdims=True)
    best = tie & (t >= tmax - TB_EPS)
    sel = np.where(best.any(axis=1), np.argmax(best, axis=1), -1)
    return sel, tie


def evaluate(self_ep: np.ndarray, tb_ep: np.ndarray, gt: Mapping[str, np.ndarray], names: Sequence[str],
             *, resolution: float, W: Optional[np.ndarray] = None) -> dict:
    """Selection + regret for every row of `W` (default: one row of ones = the point estimate).

    `self_ep` (n_ep, n_c) self episode scores (rows all NaN = no event), `tb_ep` (n_ep, n_c) S1 coverage
    component, `gt[metric]` (n_ep, n_c) GT episode means. Returns arrays over draws."""
    names = list(names)
    if names != sorted(names):
        raise ValueError("candidates must be in name order (the final tie-break)")
    S = np.asarray(self_ep, np.float64)
    n_ep, n_c = S.shape
    if W is None:
        W = np.ones((1, n_ep))
    event = np.isfinite(S).any(axis=1)
    sc = _wmean(W, S, event)
    tb = _wmean(W, tb_ep, event)
    sel, tie = select_rows(sc, tb, resolution)
    ok = sel >= 0
    rows = np.arange(len(sel))
    out: dict[str, Any] = {"selected": sel, "tie_size": tie.sum(axis=1), "event_weight": W @ event,
                           "score": sc}
    for scope, mask in (("all", None), ("event", event)):
        for m in GT_METRICS:
            if m not in gt:
                continue
            g = _wmean(W, gt[m], mask)
            gsel = np.where(ok, g[rows, np.maximum(sel, 0)], np.nan)
            with np.errstate(invalid="ignore"):
                gtie = np.where(ok, np.nansum(np.where(tie, g, 0.0), axis=1)
                                / np.maximum((tie & np.isfinite(g)).sum(axis=1), 1), np.nan)
            out[f"{scope}:{m}:selected"] = gsel
            out[f"{scope}:{m}:tie_mean"] = gtie
            if m in LOSS_METRICS:
                best = np.nanmax(np.where(np.isfinite(g), g, -np.inf), axis=1)
                best = np.where(np.isfinite(best), best, np.nan)
                out[f"{scope}:{m}:best"] = best
                out[f"{scope}:{m}:regret"] = _clean(best - gsel)
                out[f"{scope}:{m}:tie_regret"] = _clean(best - gtie)
    return out


def tau_per_episode_status(self_ep: np.ndarray, gt_ep: np.ndarray) -> list[tuple[Optional[float], str]]:
    """``(τ-b, reason)`` per episode; an episode without an event (self row all NaN) is ``no_event``."""
    S = np.asarray(self_ep, np.float64)
    G = np.asarray(gt_ep, np.float64)
    out = []
    for e in range(S.shape[0]):
        if not np.isfinite(S[e]).any():
            out.append((None, "no_event"))
        else:
            out.append(kendall_tau_b_status(S[e], G[e]))
    return out


def tau_per_episode(self_ep: np.ndarray, gt_ep: np.ndarray) -> list[Optional[float]]:
    """τ-b per episode (None = undefined or no event)."""
    return [t for t, _ in tau_per_episode_status(self_ep, gt_ep)]


def tau_summary(taus: Sequence, keys: Sequence[str], W: Optional[np.ndarray] = None) -> dict:
    """Mean over the **defined** episodes, with the undefined ones counted by reason. `taus` = τ values
    (None = undefined) or ``(τ, reason)`` pairs. When no episode is defined the mean is None and
    ``mean_status`` = ``"undefined"`` — never 0 (`T56.task.md` 사전 고정 3)."""
    pairs = [t if isinstance(t, tuple) else (t, "defined" if t is not None else "undefined")
             for t in taus]
    v = np.asarray([np.nan if t is None else float(t) for t, _ in pairs], np.float64)
    d = np.isfinite(v)
    reasons: dict[str, int] = {}
    for (t, r) in pairs:
        if t is None:
            reasons[r] = reasons.get(r, 0) + 1
    rec = {"mean_defined": float(v[d].mean()) if d.any() else None,
           "mean_status": "defined" if d.any() else TAU_UNDEFINED,
           "n_defined": int(d.sum()), "n_undefined": int((~d).sum()),
           "n_no_event": int(reasons.get("no_event", 0)),
           "n_undefined_with_event": int((~d).sum()) - int(reasons.get("no_event", 0)),
           "undefined_reasons": reasons,
           "episodes": {k: (None if t is None else float(t)) for k, (t, _) in zip(keys, pairs)},
           "episode_status": {k: ("defined" if t is not None else f"{TAU_UNDEFINED}:{r}")
                              for k, (t, r) in zip(keys, pairs)}}
    if W is not None:
        num = W @ np.where(d, v, 0.0)
        den = W @ d.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            rec["_boot"] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return rec


def episode_ties(self_ep: np.ndarray, resolution: float, keys: Sequence[str]) -> dict:
    """Per event episode: the size of the episode-level top tie set (candidates within `resolution` of
    that episode's best self score). ``tie_rate`` = share of event episodes whose top is tied (> 1).
    (The selection itself ties on the across-episode mean — `evaluate` ``tie_size``; the bootstrap
    ``tie_rate_boot`` is T51's "rank-1 tie rate".)"""
    S = np.asarray(self_ep, np.float64)
    sizes: list[Optional[int]] = []
    for row in S:
        fin = np.isfinite(row)
        if not fin.any():
            sizes.append(None)
            continue
        mx = row[fin].max()
        sizes.append(int((fin & (row >= mx - float(resolution) - 1e-12)).sum()))
    ev = [z for z in sizes if z is not None]
    return {"n_event": len(ev),
            "tie_rate": (float(np.mean([z > 1 for z in ev])) if ev else None),
            "mean_tie_size": (float(np.mean(ev)) if ev else None),
            "all_tied_rate": (float(np.mean([z == S.shape[1] for z in ev])) if ev else None),
            "per_episode": {k: z for k, z in zip(keys, sizes)}}


def ci(x: np.ndarray) -> list:
    b = np.asarray(x, np.float64)
    b = b[np.isfinite(b)]
    if not b.size:
        return [None, None]
    return [float(np.percentile(b, CI[0])), float(np.percentile(b, CI[1]))]


def _mean_or_none(x: np.ndarray) -> Optional[float]:
    x = np.asarray(x, np.float64)
    return float(x.mean()) if x.size else None


def summarize(point: Mapping[str, np.ndarray], boot: Mapping[str, np.ndarray], names: Sequence[str]) -> dict:
    """Point values + bootstrap CIs of one (group, variant, condition, phase)."""
    s = int(point["selected"][0])
    out: dict[str, Any] = {"selected": None if s < 0 else names[s],
                           "tie_size": int(point["tie_size"][0]),
                           "tied": bool(point["tie_size"][0] > 1),
                           "tie_set": None,
                           # draws with no event episode select nothing and are left out of both
                           "tie_rate_boot": _mean_or_none(boot["tie_size"][boot["selected"] >= 0] > 1),
                           "tie_size_boot_mean": _mean_or_none(boot["tie_size"][boot["selected"] >= 0]),
                           "n_event_episodes": int(point["event_weight"][0]),
                           "boot_no_event_draws": int(np.sum(boot["event_weight"] == 0))}
    for k, v in point.items():
        if ":" in k:
            x = float(v[0])
            out[k] = None if not math.isfinite(x) else x
            out[k + ":ci95"] = ci(boot[k])
    return out


def tie_set(point: Mapping[str, np.ndarray], names: Sequence[str], resolution: float) -> list[str]:
    sc = np.asarray(point["score"][0], np.float64)
    fin = np.isfinite(sc)
    if not fin.any():
        return []
    mx = sc[fin].max()
    return [n for n, v, f in zip(names, sc, fin) if f and v >= mx - resolution - 1e-12]


def paired(boot_a: Mapping[str, np.ndarray], boot_b: Mapping[str, np.ndarray], point_a, point_b,
           keys: Sequence[str]) -> dict:
    """a − b for each key: point difference and its paired-bootstrap CI."""
    out = {}
    for k in keys:
        if k not in point_a or k not in point_b:
            continue
        d = float(point_a[k][0]) - float(point_b[k][0])
        out[k] = {"diff": None if not math.isfinite(d) else d,
                  "ci95": ci(np.asarray(boot_a[k]) - np.asarray(boot_b[k]))}
    return out


__all__ = ["CI", "GT_METRICS", "LOSS_METRICS", "N_BOOT", "SEED", "TAU_REASONS", "TAU_UNDEFINED", "ci",
           "episode_draws", "episode_ties", "evaluate", "kendall_tau_b", "kendall_tau_b_status", "paired",
           "select_rows", "summarize", "tau_per_episode", "tau_per_episode_status", "tau_summary",
           "tie_set"]
