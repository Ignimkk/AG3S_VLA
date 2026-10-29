"""T48b — pure numpy metrics of the HSR counterfactual (b). No MuJoCo, no openpi.

Rows are flat (V,) arrays over all runs: `run`, `family`, `host` (record index), `gt`, `pred`.
Families: ``E0rec`` (recorded RGB + recorded state), ``E0`` (re-rendered RGB + reconstructed
state), edits ``E1 / E1s / E1z / E2 / E3`` (`t48b_scene.INTENDED`).

Aggregation (task: run unit and frame-pooled, ep1807 separately): every number comes as
`t48_hsr_core.episode_summary` — pooled k/n over frames plus the per-run ratio's median / q25 /
q75 — for each run group (`GROUPS`). Runs are not independent scenes (long / t2 / headfix /
ep1800 all start from episode 1800); the per-run table keeps that visible.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies.t48_hsr_core import CLASSES, confusion, episode_summary

EDIT_FAMILIES = ("E1", "E1s", "E1z", "E2", "E3")
FAMILIES = ("E0rec", "E0") + EDIT_FAMILIES
#: Default gate of the parity step (task #0): probe prediction agreement between recorded and
#: re-rendered images.
PARITY_AGREEMENT_MIN = 0.95


def _per_run(run: np.ndarray, hit: np.ndarray, mask: np.ndarray) -> dict[str, tuple[int, int]]:
    out = {}
    for r in np.unique(run[mask]):
        m = mask & (run == r)
        out[str(r)] = (int(hit[m].sum()), int(m.sum()))
    return out


def grouped(run: np.ndarray, hit: np.ndarray, mask: np.ndarray, groups: Mapping[str, Sequence[str]]) -> dict:
    """`{group: episode_summary + per_run}` for each run group (groups without rows omitted)."""
    run = np.asarray(run).astype(str)
    hit = np.asarray(hit, bool)
    mask = np.asarray(mask, bool)
    out = {}
    for g, members in groups.items():
        m = mask & np.isin(run, list(members))
        if not m.any():
            continue
        per = _per_run(run, hit, m)
        out[g] = {**episode_summary(per), "per_run": {k: {"k": a, "n": b, "ratio": a / b} for k, (a, b) in per.items()}}
    return out


def host_reference(run: np.ndarray, family: np.ndarray, host: np.ndarray, pred: np.ndarray,
                   ref: str) -> np.ndarray:
    """For every row, the prediction of family `ref` on the same (run, host); -1 when absent."""
    run, family = np.asarray(run).astype(str), np.asarray(family).astype(str)
    host, pred = np.asarray(host, np.int64), np.asarray(pred, np.int64)
    m = family == ref
    lut = {(r, int(h)): int(p) for r, h, p in zip(run[m], host[m], pred[m])}
    return np.asarray([lut.get((r, int(h)), -1) for r, h in zip(run, host)], np.int64)


def parity_agreement(run, family, host, pred, groups, *, a: str = "E0rec", b: str = "E0") -> dict:
    """Prediction agreement between family `a` and `b` on the same host (task #0)."""
    run, family = np.asarray(run).astype(str), np.asarray(family).astype(str)
    pred = np.asarray(pred, np.int64)
    ref = host_reference(run, family, host, pred, a)
    m = (family == b) & (ref >= 0) & (pred >= 0)
    return grouped(run, pred == ref, m, groups)


def method_metrics(run, family, host, gt, gt_phys, pred, groups) -> dict:
    """All T48b numbers of one method.

    per family:
      target_rate     pred == gt (edits: pick for E1/E1s/E1z/E2, place for E3; E0: closed_loop_gt
                      class of the host, T48's ep1807 definition)
      phys_rate       pred == gt_phys (E0 rows with an unambiguous physical state only)
      pred_counts     {group: [n pick, n place, n home]}
      changed_vs_E0   pred != the E0 (re-rendered, unedited) prediction of the same host
      confusion       gt x pred, pooled over the "all" group
    """
    run, family = np.asarray(run).astype(str), np.asarray(family).astype(str)
    gt, gt_phys, pred = (np.asarray(x, np.int64) for x in (gt, gt_phys, pred))
    valid = pred >= 0
    e0 = host_reference(run, family, host, pred, "E0")
    out: dict = {"n_rows": int(len(pred)), "n_missing": int((~valid).sum()), "families": {}}
    for fam in FAMILIES:
        m = valid & (family == fam)
        if not m.any():
            continue
        r: dict = {"n": int(m.sum()),
                   "target_rate": grouped(run, pred == gt, m, groups),
                   "pred_counts": {g: [int(((pred == c) & m & np.isin(run, list(mem))).sum()) for c in range(3)]
                                   for g, mem in groups.items() if (m & np.isin(run, list(mem))).any()},
                   "confusion_all": confusion(gt[m], pred[m])}
        mp = m & (gt_phys >= 0)
        if mp.any():
            r["phys_rate"] = grouped(run, pred == gt_phys, mp, groups)
        if fam != "E0":
            mc = m & (e0 >= 0)
            if mc.any():
                r["changed_vs_E0"] = grouped(run, pred != e0, mc, groups)
                r["E0_pred_counts_same_hosts"] = [int(((e0 == c) & mc).sum()) for c in range(3)]
        out["families"][fam] = r
    return out


def exclusion_table(run: np.ndarray, family: np.ndarray, included: np.ndarray, reason: np.ndarray) -> dict:
    """{family: {"planned", "included", "excluded": {reason: n}, "per_run": {run: [incl, planned]}}}."""
    run, family, reason = (np.asarray(x).astype(str) for x in (run, family, reason))
    included = np.asarray(included, bool)
    out = {}
    for fam in np.unique(family):
        m = family == fam
        ex = {}
        for r in reason[m & ~included]:
            ex[str(r)] = ex.get(str(r), 0) + 1
        out[str(fam)] = {"planned": int(m.sum()), "included": int((m & included).sum()), "excluded": ex,
                         "per_run": {str(r): [int((m & included & (run == r)).sum()), int((m & (run == r)).sum())]
                                     for r in np.unique(run[m])}}
    return out


def parity_pixels(run: np.ndarray, mad: np.ndarray, mx: np.ndarray, cams: Sequence[str]) -> dict:
    """Per camera and run: MAD median / max over frames and the max abs difference."""
    run = np.asarray(run).astype(str)
    mad, mx = np.asarray(mad, float), np.asarray(mx, float)
    out: dict = {"cameras": list(cams), "per_run": {}, "all": {}}
    for r in list(np.unique(run)) + [None]:
        m = np.ones(len(run), bool) if r is None else run == r
        d = {c: {"mad_median": float(np.median(mad[m, i])), "mad_max": float(mad[m, i].max()),
                 "max_abs": int(mx[m, i].max()), "n": int(m.sum())} for i, c in enumerate(cams)}
        if r is None:
            out["all"] = d
        else:
            out["per_run"][str(r)] = d
    return out


def proba_change(run, family, host, p: np.ndarray, a: str = "E0rec", b: str = "E0") -> Optional[dict]:
    """max |p_b - p_a| over classes per host pair (probe probability drift between recorded and
    re-rendered input)."""
    run, family = np.asarray(run).astype(str), np.asarray(family).astype(str)
    host = np.asarray(host, np.int64)
    p = np.asarray(p, float)
    ia = {(r, int(h)): i for i, (r, h, f) in enumerate(zip(run, host, family)) if f == a}
    d = [float(np.abs(p[i] - p[ia[(r, int(h))]]).max()) for i, (r, h, f) in enumerate(zip(run, host, family))
         if f == b and (r, int(h)) in ia and np.isfinite(p[i]).all() and np.isfinite(p[ia[(r, int(h))]]).all()]
    if not d:
        return None
    d = np.asarray(d)
    return {"n": int(len(d)), "median": float(np.median(d)), "p99": float(np.percentile(d, 99)), "max": float(d.max())}


__all__ = ["CLASSES", "EDIT_FAMILIES", "FAMILIES", "PARITY_AGREEMENT_MIN", "exclusion_table", "grouped",
           "host_reference", "method_metrics", "parity_agreement", "parity_pixels", "proba_change"]
