"""SUBTASK-b probe training / evaluation (.venv-curobo, sklearn).

M1  per feature: StandardScaler + LogisticRegression (L2, multinomial), C by GroupKFold(5) on train
    episodes; train on train -> eval val, closed-loop. 3-class acc / recall / confusion / OvR AUROC.
    14-way fine probe on the best feature (by val acc, normal) -> val confusion.
M2  (a) normal-trained probe evaluated on masked features, (b) masked-trained probe on masked features.
M3  rule baseline: min(s7, s15) < 0.9 -> place; else ever closed (this episode, up to t) -> home; else pick.
M4  probe inference ms (single frame, predict_proba incl. scaler).
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import glob
import json
import pathlib
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-b'))
FEATS = ["prefix_last", "prefix_img", "kv_L0", "kv_L4", "kv_L8", "kv_L12", "kv_L17", "ae_final", "attn_L8"]
CLASSES = ["pick", "place", "home"]
GT_MAP = {"reach": 0, "held": 1, "released": 2}
C_GRID = [1e-3, 1e-2, 1e-1, 1.0]


def phase_to_class(p):
    p = np.asarray(p)
    c = np.full(p.shape, -1)
    c[(p >= 0) & (p <= 3)] = 0
    c[(p >= 4) & (p <= 8)] = 1
    c[(p >= 9) & (p <= 13)] = 2
    return c


def load_set(name):
    files = sorted(glob.glob(str(OUT / "feat" / name / "*.npz")))
    parts = [dict(np.load(f, allow_pickle=False)) for f in files]
    out = {}
    for cond in ("normal", "masked"):
        for k in FEATS:
            out[f"{cond}__{k}"] = np.concatenate([p[f"{cond}__{k}"] for p in parts])
    out["state"] = np.concatenate([p["state"] for p in parts])
    out["ever_closed"] = np.concatenate([p["ever_closed"] for p in parts]).astype(bool)
    out["t"] = np.concatenate([p["frame_index"] for p in parts])
    out["episode"] = np.concatenate([np.full(len(p["frame_index"]), str(p["episode"])) for p in parts])
    if name == "closed":
        out["y"] = np.asarray([GT_MAP[g] for p in parts for g in p["gt"]])
        out["phase"] = np.full(len(out["y"]), -1)
        out["stride2"] = np.concatenate([p["stride2"] for p in parts]).astype(bool)
    else:
        out["phase"] = np.concatenate([p["phase_index"] for p in parts])
        out["y"] = phase_to_class(out["phase"])
    out["n_episodes"] = len(parts)
    return out


def subset(d, m):
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[:1] == m.shape else v) for k, v in d.items()}


def make(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, tol=1e-4))


def choose_C(X, y, groups):
    gkf = GroupKFold(n_splits=5)
    scores = {}
    for C in C_GRID:
        acc = []
        for tr, te in gkf.split(X, y, groups):
            m = make(C).fit(X[tr], y[tr])
            acc.append(float((m.predict(X[te]) == y[te]).mean()))
        scores[C] = float(np.mean(acc))
    best = max(scores, key=scores.get)
    return best, scores


def metrics(y, prob, labels):
    pred = np.asarray(labels)[prob.argmax(1)]
    k = len(labels)
    cm = confusion_matrix(y, pred, labels=labels)
    rec = {str(labels[i]): (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None) for i in range(k)}
    out = dict(n=int(len(y)), accuracy=float((pred == y).mean()), recall=rec, confusion=cm.tolist())
    if k == 3:
        au = {}
        for i, c in enumerate(CLASSES):
            yy = (y == labels[i]).astype(int)
            au[c] = float(roc_auc_score(yy, prob[:, i])) if 0 < yy.sum() < len(yy) else None
        out["auroc_ovr"] = au
        vals = [v for v in au.values() if v is not None]
        out["auroc_macro"] = float(np.mean(vals)) if vals else None
    return out, pred


def rule(d):
    s = d["state"]
    closed = np.minimum(s[:, 7], s[:, 15]) < 0.9
    return np.where(closed, 1, np.where(d["ever_closed"], 2, 0))


def main():
    tr, va, cl = load_set("train"), load_set("val"), load_set("closed")
    counts = {}
    for nm, d in (("train", tr), ("val", va)):
        ph = d["phase"]
        counts[nm] = dict(episodes=d["n_episodes"], frames_total=int(len(ph)),
                          frames_phase_ge20=int((ph >= 20).sum()),
                          frames_used=int((d["y"] >= 0).sum()),
                          per_class={c: int((d["y"] == i).sum()) for i, c in enumerate(CLASSES)},
                          per_phase={int(p): int((ph == p).sum()) for p in np.unique(ph)})
    counts["closed"] = dict(episodes=cl["n_episodes"], frames_all=int(len(cl["y"])),
                            frames_stride2=int(cl["stride2"].sum()),
                            per_class_stride2={c: int(((cl["y"] == i) & cl["stride2"]).sum()) for i, c in enumerate(CLASSES)},
                            per_class_all={c: int((cl["y"] == i).sum()) for i, c in enumerate(CLASSES)})
    tr, va = subset(tr, tr["y"] >= 0), subset(va, va["y"] >= 0)
    evals = {"val": va, "closed246": subset(cl, cl["stride2"]), "closed414": cl}
    groups = tr["episode"]

    res = dict(counts=counts, C_grid=C_GRID, features={}, rule={})
    rows = []
    for e, d in evals.items():
        pred = rule(d)
        prob = np.eye(3)[pred]
        res["rule"][e], _ = metrics(d["y"], prob, [0, 1, 2])
        res["rule"][e]["auroc_note"] = "rule output is a hard label; AUROC computed on one-hot scores"
    timing = {}
    for f in FEATS:
        t0 = time.time()
        r = {}
        models = {}
        for cond in ("normal", "masked"):
            X = tr[f"{cond}__{f}"]
            C, cv = choose_C(X, tr["y"], groups)
            models[cond] = make(C).fit(X, tr["y"])
            r[f"train_{cond}"] = dict(C=C, cv_acc=cv, dim=int(X.shape[1]))
        for e, d in evals.items():
            for mode, (mk, xk) in {"normal": ("normal", "normal"), "mask_a": ("normal", "masked"),
                                   "mask_b": ("masked", "masked")}.items():
                prob = models[mk].predict_proba(d[f"{xk}__{f}"])
                m, pred = metrics(d["y"], prob, [0, 1, 2])
                r[f"{e}__{mode}"] = m
                for i in range(len(pred)):
                    rows.append((e, d["episode"][i], int(d["t"][i]), int(d["y"][i]), int(d["phase"][i]), f, mode,
                                 *[float(x) for x in prob[i]], int(pred[i])))
        # probe inference time (single frame)
        x1 = va[f"normal__{f}"][:1]
        for _ in range(20):
            models["normal"].predict_proba(x1)
        ts = []
        for _ in range(300):
            t1 = time.perf_counter()
            models["normal"].predict_proba(x1)
            ts.append(time.perf_counter() - t1)
        timing[f] = float(np.median(ts) * 1e3)
        res["features"][f] = r
        print(f"{f}: C={r['train_normal']['C']} val acc={r['val__normal']['accuracy']:.3f} "
              f"a={r['val__mask_a']['accuracy']:.3f} b={r['val__mask_b']['accuracy']:.3f} | "
              f"cl246 acc={r['closed246__normal']['accuracy']:.3f} a={r['closed246__mask_a']['accuracy']:.3f} "
              f"b={r['closed246__mask_b']['accuracy']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(OUT / "probe_results.json", "w"), indent=1)
    res["probe_infer_ms_median"] = timing

    # fine 14-way on best feature (val acc, normal)
    best = max(FEATS, key=lambda f: res["features"][f]["val__normal"]["accuracy"])
    yf = tr["phase"]
    C, cv = choose_C(tr[f"normal__{best}"], yf, groups)
    m14 = make(C).fit(tr[f"normal__{best}"], yf)
    labels = list(range(14))
    out = {}
    for cond in ("normal", "masked"):
        prob = m14.predict_proba(va[f"{cond}__{best}"])
        full = np.zeros((len(prob), 14))
        full[:, m14.classes_] = prob
        mm, _ = metrics(va["phase"], full, labels)
        out[f"val__{'normal' if cond == 'normal' else 'mask_a'}"] = mm
    res["fine14"] = dict(feature=best, C=C, cv_acc=cv, **out)
    res["best_feature"] = best

    json.dump(res, open(OUT / "probe_results.json", "w"), indent=1)
    import csv
    with open(OUT / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["set", "episode", "t", "label", "phase_index", "feature", "mode", "p_pick", "p_place", "p_home", "pred"])
        w.writerows(rows)
    print("best", best, "fine14 val acc", res["fine14"]["val__normal"]["accuracy"])
    print("rule", {e: v["accuracy"] for e, v in res["rule"].items()})


if __name__ == "__main__":
    main()
