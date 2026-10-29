"""T50 — pure numpy / scipy pieces: feature sensitivity (①), counterfactual probe (②) and controls (②′).

Question (`AG3S/ag3s/docs/handoff/T50.task.md`): the `kv_L4_v1` phase probe hardly reacts when the
prompt names a distractor instead of the target (T48a C1: place -> pick 0.040, home -> 0.000). Is the
cause (a) how the probe was trained, (b) the representation, or (c) our 16D LoRA fine-tuning?

Everything here works on the T48 feature cache format (`t48_hsr_counterfactual.stage_extract`,
one npz per unit: `feature (V, 512)` + `plan__*` rows + `plan_digest`), so the same code reads the
LoRA cache of T48 and the `pi05_base` / train-split caches that `t50_extract` writes (T50b).

Pre-registered definitions (T50.task.md "사전 고정"), fixed here:

* **Feature halves.** `kv_L4` = `[mean valid text tokens | mean 768 image tokens]`
  (`stages/subtask_probe.py:64-76`) -> text = dims 0-255, image = dims 256-511, full = 0-511.
* **① sensitivity.** Normalised L2 distance = relative L2 to the host frame's C0 feature, computed
  inside each part: `d(C0, X) = ||f_X - f_C0|| / ||f_C0||` (the T48 parity `rel_l2` convention,
  `t48_hsr_counterfactual.py:573`). Per frame: `d(C0, C1)` = mean over the 3 distractors,
  `d(C0, C1p)` = mean over the 2 other templates, `r = d(C0, C1) / d(C0, C1p)` (normalisation by
  `||f_C0||` cancels in r). Baseline: `d_adj` = relative L2 between the C0 features of the frame and
  of the next sampled frame of the same episode (stride 6 = 0.4 s; the last frame has none).
  Aggregation: per episode the median over its frames, then median / IQR over episodes; frame-pooled
  median / IQR side by side; per original class (pick / place / home) and all.
* **② counterfactual probe.** StandardScaler + multinomial L2 logistic regression (lbfgs) — the
  SUBTASK-b probe (`experiments/legacy/outputs/verify/SUBTASK-b/probe.py:66-82`, `kv_L4_v1.json`
  model), re-implemented with numpy + scipy L-BFGS-B because no venv here has sklearn
  (`fit_logreg`; sklearn 1.5 objective and solver options). Training rows = original C0 rows
  (label = dataset phase class) + **one C1 row per C0 frame** (one of its 3 distractors drawn with a
  fixed seed, label `pick`) -> original : C1 = 1 : 1. C from `C_GRID` by `GroupKFold(5)` over episodes
  (sklearn's deterministic group assignment, `group_kfold`) on the training rows, mixed accuracy.
  (i) `valcv` (T50a primary): outer `GroupKFold(5)` over the 50 val episodes; inner `GroupKFold(5)`
  over the 40 training episodes picks C; evaluation on the held-out episodes. (ii) `trainval` (T50b):
  train split (200 episodes) -> val (50 episodes).
  Metrics: C0 accuracy; **C1 instruction accuracy** = share of C1 rows whose host frame is originally
  place / home that the probe calls `pick`; plus every family's accuracy against its gt.
* **②′ controls.** Same training on the text half only and on the image half only; and
  `shuffled`: the C1 rows' labels replaced by uniform random classes (fixed seed). `standard`
  (C0 rows only, no C1) is reported as the reference probe trained like `kv_L4_v1`.
* **Decision rule.** C0 >= 0.95 and C1 instruction >= 0.80. The task does not say whether the
  frame-pooled value or the episode median is compared; both are reported (`rule_check`).
"""

from __future__ import annotations

import dataclasses
import glob
import pathlib
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t48_hsr_core as t48core
from AG3S.ag3s.experiments.studies.t48_hsr_io import stable_key

CLASSES = t48core.CLASSES
PICK, PLACE, HOME = t48core.PICK, t48core.PLACE, t48core.HOME
#: SUBTASK-b probe C grid (`probe.py:29`) and solver settings (`probe.py:66-67`, `kv_L4_v1.json` model).
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
MAX_ITER = 3000
TOL = 1e-4
N_SPLITS = 5          # GroupKFold(5), `probe.py:73`
#: Pre-registered thresholds (T50.task.md 판정 규칙).
THRESH_C0 = 0.95
THRESH_C1_INSTR = 0.80
FEATURE_DIM = 512
HALF = 256
PARTS = {"full": slice(0, FEATURE_DIM), "text": slice(0, HALF), "image": slice(HALF, FEATURE_DIM)}
#: Training mixes: counterfactual (C0 + C1->pick, 1:1), standard (C0 only), shuffled (C1 random labels).
MIXES = ("counterfactual", "standard", "shuffled")
MODES = ("valcv", "trainval")
#: RNG stream tags (with the seed and the unit's stable key): which distractor, which shuffled label.
_TAG_C1_PICK, _TAG_SHUFFLE = 1, 2


# ----------------------------------------------------------------------------- feature cache


@dataclasses.dataclass
class FeatureSet:
    """Flat rows of a T48-format feature cache directory (one set, e.g. val or train)."""

    unit: np.ndarray        # (N,) str
    variant: np.ndarray     # (N,) str
    sub: np.ndarray         # (N,) str
    host: np.ndarray        # (N,) int  sampled row of the host frame
    clock: np.ndarray       # (N,) int  frame_index of the host
    y_host: np.ndarray      # (N,) int  C0 class of the host frame
    gt: np.ndarray          # (N,) int  answer of this row (C1 = pick)
    y_image: np.ndarray     # (N,) int
    y_state: np.ndarray     # (N,) int
    X: np.ndarray           # (N, 512) float32
    digests: dict           # unit -> plan digest

    def __len__(self) -> int:
        return int(len(self.unit))

    @property
    def units(self) -> list[str]:
        return sorted(set(self.unit.tolist()))

    def take(self, mask: np.ndarray) -> "FeatureSet":
        m = np.asarray(mask)
        keep = set(self.unit[m].tolist())
        kw = {f.name: getattr(self, f.name)[m] for f in dataclasses.fields(self) if f.name != "digests"}
        return FeatureSet(**kw, digests={u: d for u, d in self.digests.items() if u in keep})


def load_feature_dir(path: str | pathlib.Path, *, max_units: Optional[int] = None,
                     families: Optional[Sequence[str]] = None, verify_digest: bool = True) -> FeatureSet:
    """Read every `<unit>.npz` of a T48-format cache dir (sorted by name; first `max_units` only).

    `verify_digest`: recompute the plan digest from the stored `plan__*` rows
    (`t48_hsr_core.VariantTable.digest`) and refuse a file whose rows do not match its own digest.
    """
    files = sorted(glob.glob(str(pathlib.Path(path) / "*.npz")))
    files = [f for f in files if not f.endswith(".tmp.npz")]
    if max_units is not None:
        files = files[:max_units]
    if not files:
        raise FileNotFoundError(f"no feature npz under {path}")
    cols: dict[str, list] = {k: [] for k in ("unit", "variant", "sub", "host", "clock", "y_host", "gt",
                                             "y_image", "y_state", "X")}
    digests = {}
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            table = t48core.VariantTable.from_arrays(z, "plan__")
            dig = str(z["plan_digest"])
            if verify_digest and table.digest() != dig:
                raise ValueError(f"{f}: stored plan rows do not match plan_digest")
            uid = str(z["unit_id"]) if "unit_id" in z.files else pathlib.Path(f).stem
            feats = np.asarray(z["feature"], np.float32)
            y_frames = np.asarray(z["y"], np.int64)
            clock = np.asarray(z["clock"], np.int64)
        if feats.shape != (len(table), FEATURE_DIM):
            raise ValueError(f"{f}: feature shape {feats.shape} for {len(table)} plan rows")
        keep = np.ones(len(table), bool) if families is None else np.isin(table.variant, list(families))
        digests[uid] = dig
        cols["unit"].append(np.full(int(keep.sum()), uid, dtype=object))
        cols["variant"].append(table.variant[keep].astype(str))
        cols["sub"].append(table.sub[keep].astype(str))
        cols["host"].append(table.host[keep])
        cols["clock"].append(clock[table.host[keep]])
        cols["y_host"].append(y_frames[table.host[keep]])
        cols["gt"].append(table.gt[keep])
        cols["y_image"].append(table.y_image[keep])
        cols["y_state"].append(table.y_state[keep])
        cols["X"].append(feats[keep])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["unit"] = out["unit"].astype(str)
    return FeatureSet(**out, digests=digests)


# ----------------------------------------------------------------------------- ① sensitivity


def rel_l2(a: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Row-wise `||a - ref|| / ||ref||` (NaN where `||ref|| == 0`)."""
    a, ref = np.asarray(a, np.float64), np.asarray(ref, np.float64)
    den = np.linalg.norm(ref, axis=-1)
    num = np.linalg.norm(a - ref, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def sensitivity_frames(fs: FeatureSet, parts: Mapping[str, slice] = PARTS) -> dict[str, np.ndarray]:
    """Per host frame: `d_c1`, `d_c1p`, `r`, `d_adj` for each part (+ unit, host, clock, y).

    Hosts without a C0 row, without C1 rows or without C1p rows are skipped (counted by the caller
    as the difference to the C0 count).
    """
    rows: dict[str, list] = {"unit": [], "host": [], "clock": [], "y": [], "n_c1": [], "n_c1p": []}
    for p in parts:
        for k in ("d_c1", "d_c1p", "r", "d_adj"):
            rows[f"{p}__{k}"] = []
    for u in fs.units:
        mu = fs.unit == u
        idx = np.flatnonzero(mu)
        var, host = fs.variant[idx], fs.host[idx]
        c0 = {int(h): int(i) for h, i in zip(host[var == "C0"], idx[var == "C0"])}
        by_host: dict[int, dict[str, list[int]]] = {}
        for fam in ("C1", "C1p"):
            for h, i in zip(host[var == fam], idx[var == fam]):
                by_host.setdefault(int(h), {"C1": [], "C1p": []})[fam].append(int(i))
        for h in sorted(c0):
            g = by_host.get(h)
            if g is None or not g["C1"] or not g["C1p"]:
                continue
            i0 = c0[h]
            rows["unit"].append(u)
            rows["host"].append(h)
            rows["clock"].append(int(fs.clock[i0]))
            rows["y"].append(int(fs.y_host[i0]))
            rows["n_c1"].append(len(g["C1"]))
            rows["n_c1p"].append(len(g["C1p"]))
            nxt = c0.get(h + 1)
            for p, sl in parts.items():
                f0 = fs.X[i0, sl]
                d1 = float(np.mean(rel_l2(fs.X[g["C1"], sl], f0[None])))
                d1p = float(np.mean(rel_l2(fs.X[g["C1p"], sl], f0[None])))
                rows[f"{p}__d_c1"].append(d1)
                rows[f"{p}__d_c1p"].append(d1p)
                rows[f"{p}__r"].append(d1 / d1p if d1p > 0 else np.nan)
                rows[f"{p}__d_adj"].append(float(rel_l2(fs.X[nxt, sl][None], f0[None])[0])
                                           if nxt is not None else np.nan)
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["unit"] = out["unit"].astype(str) if len(out["unit"]) else np.asarray([], str)
    return out


def _quartiles(x: np.ndarray) -> dict:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "median": None, "q25": None, "q75": None}
    q25, med, q75 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "median": float(med), "q25": float(q25), "q75": float(q75)}


def episode_then_pooled(units: np.ndarray, values: np.ndarray) -> dict:
    """Per-episode median of `values` -> median / IQR over episodes; plus frame-pooled quartiles."""
    units, values = np.asarray(units).astype(str), np.asarray(values, np.float64)
    per_ep = []
    for u in sorted(set(units.tolist())):
        v = values[(units == u) & np.isfinite(values)]
        if len(v):
            per_ep.append(float(np.median(v)))
    e = _quartiles(np.asarray(per_ep))
    return {"episode": {"n_episodes": e["n"], "median": e["median"], "q25": e["q25"], "q75": e["q75"]},
            "pooled_frames": _quartiles(values)}


def sensitivity_summary(frames: Mapping[str, np.ndarray], parts: Iterable[str] = PARTS) -> dict:
    """`{part: {metric: {class|all: episode_then_pooled}}}` for r, d_c1, d_c1p, d_adj."""
    y = np.asarray(frames["y"], np.int64)
    out: dict = {}
    for p in parts:
        out[p] = {}
        for k in ("r", "d_c1", "d_c1p", "d_adj"):
            v = np.asarray(frames[f"{p}__{k}"], np.float64)
            out[p][k] = {"all": episode_then_pooled(frames["unit"], v)}
            for c, name in enumerate(CLASSES):
                m = y == c
                out[p][k][name] = episode_then_pooled(frames["unit"][m], v[m])
    return out


# ----------------------------------------------------------------------------- logistic regression


@dataclasses.dataclass
class LogReg:
    """StandardScaler + multinomial logistic regression; `proba` = `SubtaskProbe.proba` formula."""

    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray        # (K, D)
    intercept: np.ndarray   # (K,)
    C: float
    n_iter: int
    converged: bool

    def proba(self, X: np.ndarray) -> np.ndarray:
        Z = ((np.asarray(X, np.float64) - self.mean) / self.scale) @ self.coef.T + self.intercept
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.proba(X).argmax(1)


def standard_scaler(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """sklearn `StandardScaler`: mean, population std; zero std -> scale 1."""
    X = np.asarray(X, np.float64)
    mean = X.mean(0)
    std = X.std(0)
    scale = np.where(std == 0, 1.0, std)
    return mean, scale


def fit_logreg(X: np.ndarray, y: np.ndarray, C: float, *, n_classes: int = 3,
               max_iter: int = MAX_ITER, tol: float = TOL) -> LogReg:
    """sklearn 1.5 `LogisticRegression(C, solver='lbfgs', max_iter, tol)` multinomial, after
    `StandardScaler` (SUBTASK-b `probe.py:66-67`).

    Objective (sklearn `LinearModelLoss`, no sample weights): mean cross-entropy over the n rows +
    `1 / (2 C n) * ||coef||^2` (intercept unpenalised), full K-class parameterisation, zero init.
    Solver: `scipy.optimize.minimize(L-BFGS-B, maxiter=max_iter, gtol=tol, ftol=64 eps, maxls=50)`
    as in sklearn `_logistic_regression_path`.
    """
    from scipy.optimize import minimize

    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.int64)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError(f"X {X.shape} / y {y.shape}")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"labels must be in [0, {n_classes})")
    mean, scale = standard_scaler(X)
    Xs = (X - mean) / scale
    n, d = Xs.shape
    K = int(n_classes)
    Y = np.zeros((n, K))
    Y[np.arange(n), y] = 1.0
    l2 = 1.0 / (float(C) * n)

    def fg(w):
        W = w.reshape(K, d + 1)
        Z = Xs @ W[:, :d].T + W[:, d]
        zmax = Z.max(axis=1, keepdims=True)
        E = np.exp(Z - zmax)
        S = E.sum(axis=1, keepdims=True)
        lse = (np.log(S) + zmax)[:, 0]
        loss = float(np.mean(lse - (Z * Y).sum(1))) + 0.5 * l2 * float((W[:, :d] ** 2).sum())
        G = (E / S - Y) / n
        grad = np.empty_like(W)
        grad[:, :d] = G.T @ Xs + l2 * W[:, :d]
        grad[:, d] = G.sum(0)
        return loss, grad.ravel()

    res = minimize(fg, np.zeros(K * (d + 1)), jac=True, method="L-BFGS-B",
                   options={"maxiter": int(max_iter), "gtol": float(tol),
                            "ftol": 64 * np.finfo(float).eps, "maxls": 50})
    W = res.x.reshape(K, d + 1)
    return LogReg(mean=mean, scale=scale, coef=W[:, :d].copy(), intercept=W[:, d].copy(), C=float(C),
                  n_iter=int(res.nit), converged=bool(res.success))


def group_kfold(groups: np.ndarray, n_splits: int = N_SPLITS) -> list[tuple[np.ndarray, np.ndarray]]:
    """sklearn 1.5 `GroupKFold(n_splits).split` (deterministic): groups sorted by size (largest
    first, `np.argsort(sizes)[::-1]`), each assigned to the currently lightest fold."""
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    if n_splits > len(uniq):
        raise ValueError(f"cannot have n_splits={n_splits} > number of groups {len(uniq)}")
    sizes = np.bincount(inv)
    order = np.argsort(sizes)[::-1]
    sizes = sizes[order]
    per_fold = np.zeros(n_splits)
    g2f = np.zeros(len(uniq), dtype=np.intp)
    for gi, w in enumerate(sizes):
        light = int(np.argmin(per_fold))
        per_fold[light] += w
        g2f[order[gi]] = light
    fold_of_row = g2f[inv]
    out = []
    for f in range(n_splits):
        te = np.flatnonzero(fold_of_row == f)
        tr = np.flatnonzero(fold_of_row != f)
        out.append((tr, te))
    return out


def choose_C(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, grid: Sequence[float] = C_GRID,
             n_splits: int = N_SPLITS) -> tuple[float, dict]:
    """SUBTASK-b `choose_C` (`probe.py:70-80`): mean held-out accuracy per C over GroupKFold; the
    first C of the grid with the highest mean wins."""
    folds = group_kfold(groups, n_splits)
    scores = {}
    for C in grid:
        acc = []
        for tr, te in folds:
            m = fit_logreg(X[tr], y[tr], C)
            acc.append(float((m.predict(X[te]) == y[te]).mean()))
        scores[float(C)] = float(np.mean(acc))
    best = max(scores, key=scores.get)
    return best, scores


# ----------------------------------------------------------------------------- ② training rows


def training_rows(fs: FeatureSet, units: Iterable[str], mix: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(row indices into `fs`, labels) of the training data of `units` for one mix.

    counterfactual: every C0 row (label = its phase class) + one C1 row of the same host (one of
    the 3 distractors, uniform, RNG `default_rng([seed, 1, stable_key(unit)])` consumed in host
    order) with label `pick` -> original : C1 = 1 : 1.
    shuffled: the same rows; the C1 rows' labels drawn uniformly from {0, 1, 2}
    (RNG `default_rng([seed, 2, stable_key(unit)])`).
    standard: C0 rows only.
    """
    if mix not in MIXES:
        raise ValueError(f"mix {mix!r} not in {MIXES}")
    idx_all, lab_all = [], []
    for u in sorted(set(units)):
        mu = fs.unit == u
        i0 = np.flatnonzero(mu & (fs.variant == "C0"))
        i0 = i0[np.argsort(fs.host[i0], kind="stable")]
        idx_all.append(i0)
        lab_all.append(fs.gt[i0].astype(np.int64))
        if mix == "standard":
            continue
        rng_pick = np.random.default_rng([int(seed), _TAG_C1_PICK, stable_key(u)])
        rng_lab = np.random.default_rng([int(seed), _TAG_SHUFFLE, stable_key(u)])
        i1 = np.flatnonzero(mu & (fs.variant == "C1"))
        by_host: dict[int, list[int]] = {}
        for i in i1:
            by_host.setdefault(int(fs.host[i]), []).append(int(i))
        chosen = []
        for h in fs.host[i0]:
            cand = sorted(by_host.get(int(h), []), key=lambda i: str(fs.sub[i]))
            if not cand:
                raise ValueError(f"{u}: host {int(h)} has a C0 row but no C1 row")
            chosen.append(cand[int(rng_pick.integers(len(cand)))])
        chosen = np.asarray(chosen, np.int64)
        idx_all.append(chosen)
        if mix == "counterfactual":
            lab_all.append(np.full(len(chosen), PICK, np.int64))
        else:
            lab_all.append(rng_lab.integers(0, 3, size=len(chosen)).astype(np.int64))
    if not idx_all:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(idx_all), np.concatenate(lab_all)


# ----------------------------------------------------------------------------- ② evaluation


def _group(units: np.ndarray, hit: np.ndarray, mask: np.ndarray) -> dict:
    per = {}
    for u in sorted(set(np.asarray(units)[mask].tolist())):
        m = mask & (units == u)
        per[str(u)] = (int(hit[m].sum()), int(m.sum()))
    return t48core.episode_summary(per)


def evaluate(fs: FeatureSet, pred: np.ndarray) -> dict:
    """Metrics of predictions on `fs` rows (`pred < 0` = not evaluated, excluded).

    accuracy[family]          pred == gt (C1 gt = pick)
    C1_instr                  C1 rows whose host is originally place / home -> pred == pick  (primary)
    C1_to_pick[orig class]    C1 rows by the host's class -> pred == pick
    C0_confusion              gt x pred on C0 rows
    All as frame-pooled k/n plus per-episode ratio median / IQR (`t48_hsr_core.episode_summary`).
    """
    pred = np.asarray(pred, np.int64)
    valid = pred >= 0
    units = fs.unit
    out: dict = {"n_rows": int(len(pred)), "n_evaluated": int(valid.sum()), "accuracy": {}}
    for fam in t48core.VARIANTS:
        m = valid & (fs.variant == fam)
        if m.any():
            out["accuracy"][fam] = _group(units, pred == fs.gt, m)
    m1 = valid & (fs.variant == "C1")
    if m1.any():
        out["C1_instr"] = _group(units, pred == PICK, m1 & np.isin(fs.y_host, [PLACE, HOME]))
        out["C1_to_pick"] = {CLASSES[c]: _group(units, pred == PICK, m1 & (fs.y_host == c)) for c in range(3)}
        out["C1_matrix_origclass_x_pred"] = t48core.confusion(fs.y_host[m1], pred[m1])
    m0 = valid & (fs.variant == "C0")
    if m0.any():
        out["C0_confusion"] = t48core.confusion(fs.gt[m0], pred[m0])
    return out


def rule_check(ev: dict) -> dict:
    """The pre-registered decision thresholds on both aggregates (the task leaves the choice open)."""
    c0 = ev.get("accuracy", {}).get("C0", {})
    c1 = ev.get("C1_instr", {})
    out = {"thresholds": {"C0": THRESH_C0, "C1_instr": THRESH_C1_INSTR},
           "note": "T50.task.md does not fix pooled vs episode median; both reported, the lead decides"}
    for agg, key in (("pooled", "pooled"), ("episode_median", "episode_median")):
        a, b = c0.get(key), c1.get(key)
        out[agg] = {"C0": a, "C1_instr": b,
                    "pass": (None if a is None or b is None
                             else bool(a >= THRESH_C0 and b >= THRESH_C1_INSTR))}
    return out


def outer_folds(fs: FeatureSet, n_splits: int = N_SPLITS) -> list[list[str]]:
    """Held-out unit lists of the outer GroupKFold over the C0 rows' units (same for every cell)."""
    m0 = fs.variant == "C0"
    groups = fs.unit[m0]
    folds = group_kfold(groups, n_splits)
    return [sorted(set(groups[te].tolist())) for _, te in folds]


def run_valcv(fs: FeatureSet, part: str, mix: str, *, seed: int = 0, grid: Sequence[float] = C_GRID,
              n_splits: int = N_SPLITS, folds: Optional[list[list[str]]] = None) -> dict:
    """② (i): nested episode CV on one set. Returns predictions (all rows), probabilities, and per
    outer fold the chosen C, inner scores, training size and solver status."""
    sl = PARTS[part]
    folds = outer_folds(fs, n_splits) if folds is None else folds
    pred = np.full(len(fs), -1, np.int64)
    prob = np.full((len(fs), 3), np.nan)
    info = []
    all_units = fs.units
    for k, held in enumerate(folds):
        train_units = [u for u in all_units if u not in set(held)]
        idx, lab = training_rows(fs, train_units, mix, seed)
        X = fs.X[idx, sl].astype(np.float64)
        C, scores = choose_C(X, lab, fs.unit[idx], grid=grid, n_splits=n_splits)
        model = fit_logreg(X, lab, C)
        te = np.flatnonzero(np.isin(fs.unit, held))
        p = model.proba(fs.X[te, sl])
        prob[te] = p
        pred[te] = p.argmax(1)
        info.append({"fold": k, "held_out_units": held, "n_train_rows": int(len(idx)),
                     "train_label_counts": np.bincount(lab, minlength=3).tolist(),
                     "C": C, "inner_cv_acc": scores, "n_iter": model.n_iter, "converged": model.converged})
    return {"pred": pred, "prob": prob, "folds": info}


def run_trainval(train: FeatureSet, val: FeatureSet, part: str, mix: str, *, seed: int = 0,
                 grid: Sequence[float] = C_GRID, n_splits: int = N_SPLITS) -> dict:
    """② (ii): train split -> val. C by GroupKFold over the train episodes."""
    sl = PARTS[part]
    idx, lab = training_rows(train, train.units, mix, seed)
    X = train.X[idx, sl].astype(np.float64)
    C, scores = choose_C(X, lab, train.unit[idx], grid=grid, n_splits=n_splits)
    model = fit_logreg(X, lab, C)
    p = model.proba(val.X[:, sl])
    return {"pred": p.argmax(1), "prob": p,
            "folds": [{"fold": "train->val", "n_train_rows": int(len(idx)),
                       "train_label_counts": np.bincount(lab, minlength=3).tolist(), "C": C,
                       "inner_cv_acc": scores, "n_iter": model.n_iter, "converged": model.converged}]}
