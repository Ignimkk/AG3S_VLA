"""T54 — frame metrics, episode aggregation, paired differences and the two bootstraps (pure numpy).

`T54.task.md` 사전 고정 3 (지표) and 4 (집계 · 불확실성), as code:

Frames
    Target frames (`has_target`) = the GT target is visible in at least one of the 3 policy cameras:
    ``class_px[frame, camera, target] >= min_target_px`` for some camera (GT cache at 480×640; default
    200 px = the T47 keep rule, the T52 "visible camera" rule). No frame is dropped to improve a metric;
    the sensitivity ``min_target_px = 1`` is reported next to it.

Metrics (one value per target frame; `METRICS`)
    ``identity``        T47 `identity_correct` (strict plurality of the chosen cluster's labelled points
                        is the target). **No output = False.** All target frames.
    ``no_target``       no chosen cluster. All target frames.
    ``purity``          chosen-cluster target share. **Output frames only** — always read with ``no_target``.
    ``coverage``        observed target points (`n_target_obs`, cloud ∖ support) inside the chosen
                        cluster. **No output = 0.** Target frames with ``n_target_obs > 0`` (a frame whose
                        cloud has no target point has no coverage for any arm; the count is reported).
    ``neighbour``       GT ``other_object`` points in the chosen cluster, output frames (the
                        ``neighbour_all`` variant counts 0 on no-output frames, all target frames).
    ``destination`` / ``support``  GT crate / table points in the chosen cluster, output frames.
    ``candidate_present``  `target_candidate_present` (some candidate just before the choice has the
                        target as strict plurality). ``candidate_fail`` = ¬present, ``choice_fail`` =
                        present ∧ ¬identity (A's two failure kinds; reported for every arm).
    ``on_target``       centroid within 10 cm — reference only, never a success criterion.

Aggregation
    Episode mean (frames of one episode averaged, every episode weighted equally; an episode with no
    eligible frame for a metric is NaN and left out, the count travels with the number). Paired
    difference arm − L8H2 per episode, and its mean.
    Episode block bootstrap: the episodes are resampled with replacement (B = 10000, seed 5407,
    percentile 95 %), the **same draws** for every arm / metric / condition (paired). Sensitivity:
    moving-block bootstrap inside each episode (block = 5 consecutive target frames, non-circular),
    episodes fixed, the same frame draws for every arm. Frame-level exact McNemar on identity —
    reference only.

T57 entry (사전 고정 5): `entry_criteria` writes each row's number and a mechanical pass flag only.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

N_BOOT = 10000
SEED = 5407
BLOCK = 5
CI = (2.5, 97.5)
MIN_TARGET_PX = 200
TARGET_CLASS = 0  # `t47_head_metrics.CLASS_NAMES[0] == "target"`
GRASP_STATES = ("pre", "held", "post", "placed", "held_other_hand")
#: T57 entry tolerances (practical, not derived statistically — `T54.task.md` 사전 고정 5).
IDENTITY_TOL = -0.03
COVERAGE_TOL = -0.05


def _nan(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


# ------------------------------------------------------------------------------- frames


def has_target_frames(class_px: np.ndarray, min_px: int = MIN_TARGET_PX) -> np.ndarray:
    """``(F,)`` bool: target pixels >= `min_px` in at least one camera (``class_px`` (F, C, K))."""
    px = np.asarray(class_px)
    return (px[:, :, TARGET_CLASS] >= int(min_px)).any(axis=1)


def frame_values(row: Mapping[str, Any]) -> dict[str, float]:
    """Per-frame metric values of one replay row (NaN where the metric is undefined for the frame;
    target-frame eligibility is applied by the caller)."""
    out = bool(row["has_target"])
    ident = bool(row["identity_correct"]) and out
    present = bool(row.get("target_candidate_present", False))
    n_obs = int(row.get("n_target_obs", 0) or 0)
    cov = _nan(row.get("coverage"))
    if n_obs > 0 and not out:
        cov = 0.0
    if n_obs == 0:
        cov = float("nan")
    return {
        "identity": float(ident),
        "no_target": float(not out),
        "purity": _nan(row.get("chosen_cluster_purity")) if out else float("nan"),
        "coverage": cov,
        "neighbour": float(row.get("chosen__other_object", 0)) if out else float("nan"),
        "neighbour_all": float(row.get("chosen__other_object", 0)) if out else 0.0,
        "destination": float(row.get("chosen__destination", 0)) if out else float("nan"),
        "support": float(row.get("chosen__table", 0)) if out else float("nan"),
        "candidate_present": float(present),
        "candidate_fail": float(not present),
        "choice_fail": float(present and not ident),
        "on_target": float(bool(row.get("on_target", False))),
        "lift_ground_ms": _nan(row.get("ms_lift_ground")),
    }


METRICS = ("identity", "no_target", "purity", "coverage", "neighbour", "neighbour_all",
           "destination", "support", "candidate_present", "candidate_fail", "choice_fail",
           "on_target")
#: Higher is better (+1) / lower is better (−1) — only for reading tables, never for a decision.
DIRECTION = {"identity": 1, "no_target": -1, "purity": 1, "coverage": 1, "neighbour": -1,
             "neighbour_all": -1, "destination": -1, "support": -1, "candidate_present": 1,
             "candidate_fail": -1, "choice_fail": -1, "on_target": 1}


def build_frames(rows: Mapping[tuple, Sequence[Mapping[str, Any]]], gt: Mapping[str, Mapping[str, Any]],
                 grasp: Optional[Mapping[str, Mapping[str, Any]]] = None,
                 min_px: int = MIN_TARGET_PX) -> list[dict]:
    """Frame table. `rows`: ``{(key, condition, arm): [row per frame]}``; `gt`: ``{key: GT cache}``
    (``class_px``, ``t_step``); `grasp`: ``{key: grasp npz}`` (``grasp_state``, ``t_step``) or None.
    Rows, GT and grasp are aligned by ``t_step`` (a mismatch raises)."""
    out = []
    for (key, cond, arm), rs in rows.items():
        g = gt[key]
        t_gt = [int(t) for t in np.asarray(g["t_step"])]
        tgt = has_target_frames(g["class_px"], min_px)
        tgt1 = has_target_frames(g["class_px"], 1)
        phase = None
        if grasp is not None and key in grasp:
            if [int(t) for t in np.asarray(grasp[key]["t_step"])] != t_gt:
                raise ValueError(f"[{key}] grasp t_step differs from the GT cache")
            phase = [str(s) for s in np.asarray(grasp[key]["grasp_state"])]
        for r in rs:
            i = int(r["frame"])
            if int(r["t_step"]) != t_gt[i]:
                raise ValueError(f"[{key}] row frame {i} t_step {r['t_step']} != GT {t_gt[i]}")
            out.append({"key": key, "condition": cond, "arm": arm, "frame": i,
                        "t_step": int(r["t_step"]), "has_target_gt": bool(tgt[i]),
                        "has_target_gt_1px": bool(tgt1[i]),
                        "phase": None if phase is None else phase[i],
                        "n_target_obs": int(r.get("n_target_obs", 0) or 0),
                        **frame_values(r)})
    return out


# ------------------------------------------------------------------------- episode means


def episode_matrix(frames: Sequence[Mapping[str, Any]], *, keys: Sequence[str], arms: Sequence[str],
                   condition: str, metric: str, target_field: str = "has_target_gt",
                   phase: Optional[str] = None) -> tuple[np.ndarray, np.ndarray]:
    """``(E (n_ep, n_arm) episode means, N (n_ep, n_arm) frame counts)`` over target frames (and
    `phase` when given); frames whose value is NaN do not count."""
    ki = {k: i for i, k in enumerate(keys)}
    ai = {a: j for j, a in enumerate(arms)}
    s = np.zeros((len(keys), len(arms)))
    n = np.zeros((len(keys), len(arms)), np.int64)
    for f in frames:
        if f["condition"] != condition or f["key"] not in ki or f["arm"] not in ai:
            continue
        if not f[target_field] or (phase is not None and f["phase"] != phase):
            continue
        v = f[metric]
        if v is None or not math.isfinite(v):
            continue
        s[ki[f["key"]], ai[f["arm"]]] += v
        n[ki[f["key"]], ai[f["arm"]]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return e, n


def episode_draws(n_ep: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """``(B, n_ep)`` episode indices drawn with replacement — the episode block bootstrap."""
    return np.random.default_rng(seed).integers(0, n_ep, size=(n_boot, n_ep))


def boot_mean(values: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """``(B,)`` nan-mean of `values` (n_ep,) over each row of episode `draws`."""
    v = np.asarray(values, np.float64)[draws]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(v, axis=1)


def ci(boot: np.ndarray) -> list[float]:
    b = np.asarray(boot, np.float64)
    b = b[np.isfinite(b)]
    if not b.size:
        return [float("nan"), float("nan")]
    return [float(np.percentile(b, CI[0])), float(np.percentile(b, CI[1]))]


def summarize_matrix(e: np.ndarray, n: np.ndarray, *, arms: Sequence[str], keys: Sequence[str],
                     draws: np.ndarray, base: str) -> dict:
    """Per arm: episode mean, episodes used, episode values, bootstrap CI; per arm ≠ base: paired
    episode differences, their mean and its episode-bootstrap CI."""
    b = list(arms).index(base)
    out: dict[str, Any] = {}
    for j, a in enumerate(arms):
        col = e[:, j]
        ok = np.isfinite(col)
        rec = {"mean": float(np.nanmean(col)) if ok.any() else float("nan"),
               "n_episodes": int(ok.sum()), "n_frames": int(n[:, j].sum()),
               "episodes": {k: (None if not np.isfinite(col[i]) else float(col[i]))
                            for i, k in enumerate(keys)},
               "ci95": ci(boot_mean(col, draws))}
        if a != base:
            d = col - e[:, b]
            okd = np.isfinite(d)
            rec["diff_vs_" + base] = {
                "mean": float(np.nanmean(d)) if okd.any() else float("nan"),
                "n_episodes": int(okd.sum()),
                "episodes": {k: (None if not np.isfinite(d[i]) else float(d[i]))
                             for i, k in enumerate(keys)},
                "ci95_episode_bootstrap": ci(boot_mean(d, draws)),
                "n_episodes_up": int((d[okd] > 0).sum()), "n_episodes_down": int((d[okd] < 0).sum()),
            }
        out[a] = rec
    return out


# ------------------------------------------------------------------- moving block (sensitivity)


def block_positions(n: int, rng: np.random.Generator, n_boot: int, block: int = BLOCK) -> np.ndarray:
    """``(B, n)`` positions into a sequence of length `n`: ceil(n / L) non-circular blocks of L
    consecutive positions, truncated to n (L = min(block, n))."""
    if n <= 0:
        return np.zeros((n_boot, 0), np.int64)
    L = min(int(block), n)
    nb = int(math.ceil(n / L))
    starts = rng.integers(0, n - L + 1, size=(n_boot, nb))
    pos = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(n_boot, nb * L)
    return pos[:, :n]


def moving_block(frames: Sequence[Mapping[str, Any]], *, keys: Sequence[str], arms: Sequence[str],
                 condition: str, metric: str, base: str, n_boot: int = N_BOOT, seed: int = SEED,
                 block: int = BLOCK, target_field: str = "has_target_gt") -> dict:
    """Within-episode moving-block bootstrap of the episode-mean metric and the paired difference.
    The block sequence of an episode is its target frames in time order (GT, the same for every arm),
    so every arm is resampled on the same frames."""
    rng = np.random.default_rng(seed)
    by = {}
    for f in frames:
        if f["condition"] == condition and f[target_field]:
            by.setdefault(f["key"], {}).setdefault(f["arm"], {})[f["frame"]] = f[metric]
    per_arm = {a: np.full((n_boot, len(keys)), np.nan) for a in arms}
    for ei, k in enumerate(keys):
        if k not in by:
            continue
        fr = sorted(set().union(*[set(v) for v in by[k].values()]))
        pos = block_positions(len(fr), rng, n_boot, block)
        for a in arms:
            vals = np.asarray([_nan(by[k].get(a, {}).get(i)) for i in fr], np.float64)
            if not len(vals):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                per_arm[a][:, ei] = np.nanmean(vals[pos], axis=1)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a in arms:
            m = np.nanmean(per_arm[a], axis=1)
            rec = {"ci95": ci(m)}
            if a != base:
                rec["diff_ci95"] = ci(np.nanmean(per_arm[a] - per_arm[base], axis=1))
            out[a] = rec
    return out


# ----------------------------------------------------------------------------- reference


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p (binomial on the discordant pairs)."""
    from scipy.stats import binom

    n = int(b) + int(c)
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(int(b), int(c)), n, 0.5)))


def frame_mcnemar(frames: Sequence[Mapping[str, Any]], *, condition: str, arm: str, base: str,
                  metric: str = "identity", target_field: str = "has_target_gt") -> dict:
    v = {}
    for f in frames:
        if f["condition"] == condition and f[target_field] and f["arm"] in (arm, base):
            v.setdefault((f["key"], f["frame"]), {})[f["arm"]] = bool(f[metric])
    pairs = [x for x in v.values() if arm in x and base in x]
    b = sum(1 for x in pairs if x[arm] and not x[base])
    c = sum(1 for x in pairs if x[base] and not x[arm])
    return {"n_pairs": len(pairs), "arm_only": b, "base_only": c, "p_exact": mcnemar_exact(b, c),
            "note": "frames are not independent — reference only"}


def timing(frames: Sequence[Mapping[str, Any]], *, condition: str, arms: Sequence[str],
           extract_ms: Optional[Mapping[str, float]] = None,
           extract_kinds: Optional[Mapping[str, Sequence[str]]] = None, base: Optional[str] = None
           ) -> dict:
    """Median / p95 of the per-frame lifting + grounding ms (all frames of the condition) and, when
    the per-kind extraction ms are given, of e2e = extraction + lifting/grounding; plus the paired
    per-frame extra cost vs `base`."""
    per = {a: {} for a in arms}
    for f in frames:
        if f["condition"] == condition and f["arm"] in per:
            per[f["arm"]][(f["key"], f["frame"])] = f["lift_ground_ms"]
    out: dict[str, Any] = {}

    def stats(x):
        x = np.asarray([v for v in x if v is not None and math.isfinite(v)], np.float64)
        if not x.size:
            return {"median": float("nan"), "p95": float("nan"), "n": 0}
        return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "n": int(x.size)}

    for a in arms:
        ext = None
        if extract_ms is not None and extract_kinds is not None:
            kinds = list(extract_kinds[a])
            if all(k in extract_ms for k in kinds):
                ext = float(sum(float(extract_ms[k]) for k in kinds))
        rec = {"lift_ground_ms": stats(per[a].values()), "extract_ms": ext,
               "extract_kinds": None if extract_kinds is None else list(extract_kinds[a])}
        rec["e2e_ms"] = (None if ext is None else stats([v + ext for v in per[a].values()]))
        if base is not None and a != base:
            common = [k for k in per[a] if k in per[base]]
            extra = [per[a][k] - per[base][k] for k in common]
            base_ext = None
            if extract_ms is not None and extract_kinds is not None:
                bk = list(extract_kinds[base])
                if all(k in extract_ms for k in bk):
                    base_ext = float(sum(float(extract_ms[k]) for k in bk))
            rec["extra_vs_" + base] = {
                "lift_ground_ms": stats(extra),
                "e2e_ms": (None if ext is None or base_ext is None
                           else stats([x + ext - base_ext for x in extra]))}
        out[a] = rec
    return out


# --------------------------------------------------------------------------- T57 entry


def entry_criteria(summary: Mapping[str, Any], *, arm: str = "A", base: str = "L8H2",
                   budget_ms: Optional[float] = None, hold_condition: str = "hold",
                   direction_positive: bool = False) -> dict:
    """Mechanical comparison of `T54.task.md` 사전 고정 5 (numbers + pass flags; no interpretation).

    `summary[condition]["all"][metric]` is `summarize_matrix` output. `hold_condition` names the
    condition of the "시간 유지" row (T54 ``hold``; T54b / T54c ``hold_attach``). `direction_positive`
    (T54c 사전 고정) adds the row ``direction`` = current identity difference > 0 and makes it part of
    ``all_numeric_rows_pass``. Defaults give the T54 table unchanged."""
    def diff(cond, metric):
        try:
            return float(summary[cond]["all"][metric][arm]["diff_vs_" + base]["mean"])
        except (KeyError, TypeError):
            return float("nan")

    def ci_of(cond, metric):
        try:
            return summary[cond]["all"][metric][arm]["diff_vs_" + base]["ci95_episode_bootstrap"]
        except (KeyError, TypeError):
            return None

    rows = {}
    di, dc = diff("current", "identity"), diff("current", "coverage")
    dp, dn = diff("current", "purity"), diff("current", "neighbour")
    dnt = diff("current", "no_target")
    rows["identity"] = {"value": di, "ci95": ci_of("current", "identity"), "threshold": IDENTITY_TOL,
                        "pass": bool(math.isfinite(di) and di >= IDENTITY_TOL)}
    rows["coverage"] = {"value": dc, "ci95": ci_of("current", "coverage"), "threshold": COVERAGE_TOL,
                        "pass": bool(math.isfinite(dc) and dc >= COVERAGE_TOL)}
    improved = {"identity_up": bool(math.isfinite(di) and di > 0),
                "purity_up": bool(math.isfinite(dp) and dp > 0),
                "neighbour_down": bool(math.isfinite(dn) and dn < 0)}
    rows["improvement_observed"] = {"identity": di, "purity": dp, "neighbour": dn,
                                    "no_target_with_purity": dnt, **improved,
                                    "pass": any(improved.values())}
    hi, hc = diff(hold_condition, "identity"), diff(hold_condition, "coverage")
    rows["hold"] = {"identity": hi, "coverage": hc,
                    "identity_ci95": ci_of(hold_condition, "identity"),
                    "coverage_ci95": ci_of(hold_condition, "coverage"),
                    "pass": bool(math.isfinite(hi) and hi >= IDENTITY_TOL
                                 and math.isfinite(hc) and hc >= COVERAGE_TOL)}
    t = (summary.get("current", {}).get("timing", {}) or {}).get(arm, {})
    e2e = t.get("e2e_ms")
    rows["timing"] = {"lift_ground_ms": t.get("lift_ground_ms"), "e2e_ms": e2e,
                      "budget_ms": budget_ms,
                      "within_budget_median": (None if budget_ms is None or not e2e
                                               else bool(e2e["median"] <= budget_ms)),
                      "within_budget_p95": (None if budget_ms is None or not e2e
                                            else bool(e2e["p95"] <= budget_ms))}
    passed = bool(rows["identity"]["pass"] and rows["coverage"]["pass"]
                  and rows["improvement_observed"]["pass"] and rows["hold"]["pass"])
    out = {"arm": arm, "base": base, "condition": "current (hold for the hold row)",
           "note": "mechanical comparison only; the tolerances are practical, not statistical",
           "rows": rows}
    if hold_condition != "hold":
        out["hold_condition"] = hold_condition
    if direction_positive:
        rows["direction"] = {"identity": di, "rule": "current identity difference > 0 (same sign as T53)",
                             "pass": bool(math.isfinite(di) and di > 0)}
        passed = passed and rows["direction"]["pass"]
    out["all_numeric_rows_pass"] = passed
    return out


# ------------------------------------------------------------------------------- driver


def analyze(frames: Sequence[Mapping[str, Any]], *, keys: Sequence[str], arms: Sequence[str],
            conditions: Sequence[str], base: str = "L8H2", n_boot: int = N_BOOT, seed: int = SEED,
            block: int = BLOCK, phases: Sequence[str] = GRASP_STATES,
            extract_ms: Optional[Mapping[str, float]] = None,
            extract_kinds: Optional[Mapping[str, Sequence[str]]] = None,
            log: Callable[[str], None] = print) -> dict:
    draws = episode_draws(len(keys), n_boot, seed)
    out: dict[str, Any] = {"keys": list(keys), "arms": list(arms), "base": base,
                           "bootstrap": {"episode": {"B": n_boot, "seed": seed, "ci": list(CI),
                                                     "unit": "episode (resampled with replacement)"},
                                         "moving_block": {"B": n_boot, "seed": seed, "block": block,
                                                          "unit": "target frames within an episode"}}}
    for cond in conditions:
        res: dict[str, Any] = {"all": {}, "phase": {}, "moving_block": {}, "sensitivity_1px": {}}
        for m in METRICS:
            e, n = episode_matrix(frames, keys=keys, arms=arms, condition=cond, metric=m)
            res["all"][m] = summarize_matrix(e, n, arms=arms, keys=keys, draws=draws, base=base)
            res["moving_block"][m] = moving_block(frames, keys=keys, arms=arms, condition=cond,
                                                  metric=m, base=base, n_boot=n_boot, seed=seed,
                                                  block=block)
            e1, n1 = episode_matrix(frames, keys=keys, arms=arms, condition=cond, metric=m,
                                    target_field="has_target_gt_1px")
            res["sensitivity_1px"][m] = summarize_matrix(e1, n1, arms=arms, keys=keys, draws=draws,
                                                         base=base)
        for ph in phases:
            res["phase"][ph] = {}
            for m in METRICS:
                e, n = episode_matrix(frames, keys=keys, arms=arms, condition=cond, metric=m, phase=ph)
                res["phase"][ph][m] = summarize_matrix(e, n, arms=arms, keys=keys, draws=draws,
                                                       base=base)
        res["frame_mcnemar_identity"] = {a: frame_mcnemar(frames, condition=cond, arm=a, base=base)
                                         for a in arms if a != base}
        res["timing"] = timing(frames, condition=cond, arms=arms, extract_ms=extract_ms,
                               extract_kinds=extract_kinds, base=base)
        res["n_target_frames"] = int(sum(1 for f in frames if f["condition"] == cond
                                         and f["arm"] == base and f["has_target_gt"]))
        res["n_target_frames_no_obs_points"] = {
            a: int(sum(1 for f in frames if f["condition"] == cond and f["arm"] == a
                       and f["has_target_gt"] and f["n_target_obs"] == 0)) for a in arms}
        out[cond] = res
        log(f"  analyzed {cond}")
    return out


__all__ = ["BLOCK", "CI", "COVERAGE_TOL", "DIRECTION", "GRASP_STATES", "IDENTITY_TOL", "METRICS",
           "MIN_TARGET_PX", "N_BOOT", "SEED", "analyze", "block_positions", "boot_mean",
           "build_frames", "ci", "entry_criteria", "episode_draws", "episode_matrix",
           "frame_mcnemar", "frame_values", "has_target_frames", "mcnemar_exact", "moving_block",
           "summarize_matrix", "timing"]
