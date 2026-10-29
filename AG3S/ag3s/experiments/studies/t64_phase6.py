"""T64 Part A — the 6-phase (or 4-phase fallback) task-phase probe on the stored HSR features (`handoff/T64.task.md`).

Same feature and model family as the serving HSR probe `kv_L4_v1`: ``normal__kv_L4`` (512-D prefix KV, layer 4),
StandardScaler + multinomial logistic regression; C chosen by GroupKFold(5) over the train episodes (macro-F1).
Labels = the expert collector's ``phase_index`` regrouped (pre-registered table), recovery phases 20–22 -> approach,
every other index excluded (counted).

Outputs (``OUT/partA.json`` + the fitted probe ``OUT/probe_<n>.npz``): per-class precision / recall / F1, macro-F1,
confusion matrix, boundary delay (feature frames and control steps via ``frame_index``) on the 50 val episodes, the
6 → 4 decision (val macro-F1 < 0.80 -> 4 phases; decided here only, never from Part B).

usage: t64_phase6.py OUT_DIR      (.venv-ag3s, CPU, one process)
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

FEAT = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/workspace/verify/SUBTASK-b/feat")
KEY = "normal__kv_L4"
PHASES6 = ("approach", "grasp", "lift", "transport", "placement", "retreat")
PHASES4 = ("approach", "grasp", "transport", "placement")
MAP6 = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 3, 7: 4, 8: 4, 9: 4, 10: 5, 11: 5, 12: 5, 13: 5, 20: 0, 21: 0, 22: 0}
MAP4 = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 3, 8: 3, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3, 20: 0, 21: 0, 22: 0}
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
SWITCH_BELOW = 0.80
SEED = 6401


def load(split: str, mapping: dict):
    X, y, g, fi, excluded, recovery = [], [], [], [], 0, 0
    for f in sorted((FEAT / split).glob("ep_*.npz")):
        z = np.load(f, allow_pickle=True)
        ph = np.asarray(z["phase_index"], int)
        keep = np.array([p in mapping for p in ph])
        excluded += int((~keep).sum())
        recovery += int(np.isin(ph, (20, 21, 22)).sum())
        X.append(np.asarray(z[KEY], np.float32)[keep])
        y.append(np.array([mapping[p] for p in ph[keep]], int))
        g.append(np.full(int(keep.sum()), int(z["episode"])))
        fi.append(np.asarray(z["frame_index"], int)[keep])
    return (np.concatenate(X), np.concatenate(y), np.concatenate(g), fi, y,
            {"excluded_frames": excluded, "recovery_frames_mapped_to_approach": recovery})


ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))


def fit(X, y, C, K):
    """sklearn-equivalent StandardScaler + multinomial LogisticRegression (lbfgs) — `t50_core.fit_logreg`
    (no sklearn in the checkout's venvs)."""
    from AG3S.ag3s.experiments.studies.t50_core import fit_logreg

    return fit_logreg(X, y, C, n_classes=K, max_iter=3000, tol=1e-4)


def predict(model, X):
    return np.argmax(model.proba(np.asarray(X, np.float64)), axis=1)


def prf(y, p, K):
    cm = np.zeros((K, K), int)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(K), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(K), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    return prec, rec, f1, cm.sum(1), cm


def choose_C(X, y, g, K):
    from AG3S.ag3s.experiments.studies.t50_core import group_kfold

    scores = {}
    for C in C_GRID:
        s = []
        for tr, te in group_kfold(g, 5):
            s.append(float(np.mean(prf(y[te], predict(fit(X[tr], y[tr], C, K), X[te]), K)[2])))
        scores[C] = float(np.mean(s))
    best = max(C_GRID, key=lambda c: (scores[c], -c))
    return best, scores


def boundary_delays(pred_eps, true_eps, fidx_eps, n_cls):
    """Per true transition a -> b: first frame at/after it where the prediction is b (delay ≥ 0) or the last frame
    before it where the prediction already is b and stays b until the transition (lead, delay < 0)."""
    out = {}
    for p, t, fi in zip(pred_eps, true_eps, fidx_eps):
        for i in range(1, len(t)):
            if t[i] == t[i - 1]:
                continue
            a, b = int(t[i - 1]), int(t[i])
            j = i
            while j > 0 and p[j - 1] == b and t[j - 1] == a:
                j -= 1
            if j < i:
                d_f, d_s = j - i, int(fi[j] - fi[i])
            else:
                k = next((k for k in range(i, len(t)) if p[k] == b), None)
                if k is None:
                    d_f, d_s = None, None
                else:
                    d_f, d_s = k - i, int(fi[k] - fi[i])
            out.setdefault(f"{a}->{b}", []).append({"frames": d_f, "steps": d_s})
    summ = {}
    for key, v in out.items():
        fr = [x["frames"] for x in v if x["frames"] is not None]
        st = [x["steps"] for x in v if x["steps"] is not None]
        summ[key] = {"n": len(v), "missed": sum(x["frames"] is None for x in v),
                     "median_frames": float(np.median(fr)) if fr else None,
                     "median_steps": float(np.median(st)) if st else None,
                     "lead_frac": float(np.mean([x < 0 for x in fr])) if fr else None}
    return summ


def evaluate(names, mapping, out: pathlib.Path) -> dict:
    K = len(names)
    Xtr, ytr, gtr, _, _, ex_tr = load("train", mapping)
    Xva, yva, gva, fva, yva_eps, ex_va = load("val", mapping)
    C, cv = choose_C(Xtr, ytr, gtr, K)
    model = fit(Xtr, ytr, C, K)
    pva = predict(model, Xva)
    pr, rc, f1, sup, cm = prf(yva, pva, K)
    pred_eps, i0 = [], 0
    for t in yva_eps:
        pred_eps.append(pva[i0:i0 + len(t)])
        i0 += len(t)
    np.savez(out / f"probe_{K}.npz", classes=np.array(names), mean=model.mean, scale=model.scale,
             coef=model.coef, intercept=model.intercept, C=C, feature=KEY)
    return {"phases": list(names), "C": C, "cv_macro_f1": cv,
            "train_frames": int(len(ytr)), "val_frames": int(len(yva)), "train_excluded": ex_tr, "val_excluded": ex_va,
            "class_counts_train": np.bincount(ytr, minlength=len(names)).tolist(),
            "val": {"macro_f1": float(np.mean(f1)), "accuracy": float(np.mean(pva == yva)),
                    "per_class": {n: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                                      "support": int(sup[i])} for i, n in enumerate(names)},
                    "confusion": cm.tolist(),
                    "boundary_delay": boundary_delays(pred_eps, yva_eps, fva, len(names))}}


def main(out: pathlib.Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    res = {"definitions": __doc__, "feature_dir": str(FEAT), "feature": KEY, "map6": MAP6,
           "switch_rule": f"val macro-F1 < {SWITCH_BELOW} -> 4 phases"}
    res["six"] = evaluate(PHASES6, MAP6, out)
    six = res["six"]["val"]["macro_f1"]
    res["decision"] = "4 phases" if six < SWITCH_BELOW else "6 phases"
    if six < SWITCH_BELOW:
        res["four"] = evaluate(PHASES4, MAP4, out)
        res["map4"] = MAP4
    (out / "partA.json").write_text(json.dumps(res, indent=1, default=float))
    print("6-phase val macro-F1", round(six, 4), "->", res["decision"])


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
