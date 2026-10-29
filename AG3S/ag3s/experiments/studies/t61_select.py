"""T61 — pre-registered attention head selection on the calibration (P window), `handoff/T61.task.md`.

The T51b protocol (`t51b_study`, combined Ω = action last · mean · first + noun_target) on the T60 P window
(`t60_t51b` patches), with the T61 rules swapped in (parent process):
    pr.preselect   -> per-query fixed allocation: the top `PER_QUERY` P1-eligible cells of **each** query by
                      2D lift (tie 2D peak, then Ω index); **no forced cells** (L8H2 / T47 oracle get no
                      special treatment)
    st.ranking_3d  -> scene-mean GT identity_correct; exact ties (≤ 1e-9) broken by scene-mean purity, then
                      scene-mean coverage (chosen__target / n_target_obs, no output = 0, frames with no observed
                      target point excluded), then Ω order
    pr.choose_k    -> one-standard-error rule: best = highest CV mean (ties -> smaller k); choose the smallest
                      k whose CV mean ≥ mean_best − SE_best, SE = sd(units, ddof 1) / √n
Rankings stay fold-wise (t51 `grouped_cv` / `nested_cv` rank each training set from its own preselection).
Calibration = ``--records t2 ep1807 ep0 ep500 ep1200`` (``long`` excluded: input wiring, T56).

usage: t61_select.py --mask-dir DIR --base DIR -- <t51b_study args (with --omega combined)>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from AG3S.ag3s.experiments.studies import t51_protocol as pr  # noqa: E402
from AG3S.ag3s.experiments.studies import t51_study as st  # noqa: E402
from AG3S.ag3s.experiments.studies import t60_t51b as w  # noqa: E402

PER_QUERY = 6
TIE_EPS = 1e-9
CALIB = ("t2", "ep1807", "ep0", "ep500", "ep1200")
N_PER_QUERY = len(pr.LAYERS) * len(pr.HEADS)


def preselect(score2d, tie2d, eligible, *, n_top=None, forced=()):
    """Per-query fixed allocation, no forced cells (T61). Ω is query-major with N_PER_QUERY cells each."""
    s = np.asarray(score2d, np.float64); t = np.asarray(tie2d, np.float64); el = np.asarray(eligible, bool)
    n_q = len(s) // N_PER_QUERY
    assert n_q * N_PER_QUERY == len(s), len(s)
    top, per_query = [], {}
    for q in range(n_q):
        lo, hi = q * N_PER_QUERY, (q + 1) * N_PER_QUERY
        order = pr.rank_order(s[lo:hi], t[lo:hi], el[lo:hi])
        pick = [int(i) + lo for i in order[:PER_QUERY]]
        per_query[q] = pick
        top += pick
    ranked = sorted(top)
    return {"top": top, "per_query": per_query, "forced": [], "forced_ignored": [int(i) for i in forced],
            "forced_ineligible": [], "forced_already_in_top": [], "ranked_set": ranked, "scored_set": ranked}


def _coverage(rows) -> float:
    v = []
    for r in rows:
        n = r.get("n_target_obs") or 0
        if n <= 0:
            continue
        v.append(float(r.get("chosen__target") or 0) / float(n) if (r.get("chosen_n") or 0) > 0 else 0.0)
    return float(np.mean(v)) if v else float("nan")


def metric_rows(root, units, ids, fn) -> np.ndarray:
    out = np.full((len(units), len(ids)), np.nan)
    for e, u in enumerate(units):
        for j, i in enumerate(ids):
            rows = st.load_rows(root, u, i)
            if rows is None:
                raise SystemExit(f"[{u}] no rows for {i}")
            out[e, j] = fn(rows)
    return out


def ranking_3d(root, label, units, ranked_ids, arrs, keys_all, mode="record"):
    ids = list(ranked_ids)
    prim = st.calib_mean(st.metric_matrix(root, units, ids, st.PRIMARY[label]), units, mode)
    pur = st.calib_mean(st.metric_matrix(root, units, ids, "purity_mean"), units, mode)
    cov = st.calib_mean(metric_rows(root, units, ids, _coverage), units, mode)
    key = lambda a: -np.nan_to_num(np.asarray(a, np.float64), nan=-np.inf)
    # quantise the primary so that differences ≤ TIE_EPS are ties
    p = np.round(np.nan_to_num(prim, nan=-np.inf) / TIE_EPS) * TIE_EPS
    order = np.lexsort((np.arange(len(ids)), key(cov), key(pur), -p))
    return [ids[i] for i in order]


def choose_k(scores, *, n_boot=None, seed=None, ci=None):
    ks = sorted(int(k) for k in scores)
    vals = {k: np.asarray(scores[k], np.float64) for k in ks}
    means = {k: float(np.nanmean(v)) for k, v in vals.items()}
    ses = {k: float(np.nanstd(v, ddof=1) / np.sqrt(np.isfinite(v).sum())) if np.isfinite(v).sum() > 1 else float("nan")
           for k, v in vals.items()}
    best = max(ks, key=lambda k: (np.nan_to_num(means[k], nan=-np.inf), -k))
    thr = means[best] - (ses[best] if np.isfinite(ses[best]) else 0.0)
    chosen = next(k for k in ks if np.nan_to_num(means[k], nan=-np.inf) >= thr)
    return {"chosen_k": chosen, "best_k": best, "means": means, "se": ses, "threshold": thr, "tests": [],
            "rule": "one-SE: smallest k with CV mean >= mean_best - SE_best (SE = sd ddof1 / sqrt(n units))"}


def install_t61() -> None:
    pr.preselect = preselect
    pr.choose_k = choose_k
    st.ranking_3d = ranking_3d


def main() -> None:
    install_t61()
    if "--window" not in sys.argv:
        sys.argv[1:1] = ["--window", "P"]
    w.main()


if __name__ == "__main__":
    main()
