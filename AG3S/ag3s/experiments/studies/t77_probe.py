"""T77 — 5-stage HSR classifier: layer-4 KV (``kv_L4``) against PaliGemma mid-layer hidden states (`handoff/T77.task.md`
§3, §4; fixed before results).

Data (all produced elsewhere, read only):
  expert   ``EXPERT_DIR/{train,val}/ep_*.npz`` — `pi05_hidden --pool summary --dataset` (SUBTASK-b frames: train =
           episodes 0–1599 every 8th, val = 1600–1799 every 4th, every 6th frame). Label = ``MAP5[phase_index]``,
           other indices excluded (counted).
  closed   ``CLOSED_DIR/<safe label>.npz`` — `pi05_hidden --pool summary --spec`, one per run; joined on ``t_step``
           with ``LABELS`` (`t77_labels labels`): ``gt.phase5`` per request; ``split`` from the spec (scene split §2).

Features (same classifier for all: StandardScaler + multinomial logistic regression, `t64_phase6.fit`):
  ``kv_L4``           512   (baseline; the serving HSR feature)
  ``h<l>_img3+last``  8192  hidden layer l: [cam_high mean | left mean | right mean | last valid text token]
  ``h<l>_last``       2048  hidden layer l: last valid text token

Training rows = expert train + closed-loop train-scene requests. C per feature from `t64_phase6.C_GRID` by
GroupKFold(5) over episodes / runs (mean fold macro-F1, `t64_phase6.choose_C` rule). The hidden layer · pooling is
chosen by the same train-CV score only (ties -> smaller feature, then lower layer); the test set is not consulted.
Every feature is still fitted and evaluated (graph data), but the verdict uses the selected one only.

Metrics on closed-loop **test** scenes (per request):
  false_carry_rate   GT ∈ {approach, grasp} (not held yet) predicted carry / place / retreat
  accuracy, macro_f1 (5 stages), confusion
  boundary delay     GT grasp -> carry ("1->2") and carry -> place ("2->3"): requests until the prediction follows
                     (median; `t64_phase6.boundary_delays`, negative = prediction led)
expert val macro-F1 (5 stages). Serving reference on the same test rows: HSR6 mapped to 5 stages, and the 3-stage
signal on the coarse false-carry (predicted place / home while not held).

Verdict (§4, "hidden is better", all three): test false_carry(hidden) <= 0.5 × false_carry(kv_L4) ∧ test
accuracy(hidden) >= accuracy(kv_L4) − 0.02 ∧ expert val macro-F1(hidden) >= macro-F1(kv_L4) − 0.02.

usage (.venv-ag3s, CPU):
  OPENBLAS_NUM_THREADS=4 PYTHONPATH=$PWD .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.t77_probe \\
      OUT_DIR --expert-dir E --closed-dir C --labels labels.json [--layers 0 2 4 ...] [--jobs 4]
Writes ``OUT_DIR/cache/`` (row table + per-layer float32 memmaps, reused), ``OUT_DIR/models/<feature>.npz``,
``OUT_DIR/t77_probe.json`` (all numbers + per-request test predictions for figures).
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time
from typing import Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies.t77_labels import MAP5, PHASES5, COARSE_OF_5

LAYERS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 17, 18)
POOLS = {"img3+last": (0, 1, 2, 4), "last": (4,)}
K = len(PHASES5)
NOT_HELD = (0, 1)              # approach, grasp
CARRY_OR_LATER = (2, 3, 4)     # carry, place, retreat
FC_FACTOR = 0.5
ACC_MARGIN = 0.02
F1_MARGIN = 0.02


# ================================================================================ metrics (tested)


def false_carry_rate(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    m = np.isin(y, NOT_HELD)
    n = int(m.sum())
    k = int(np.isin(p[m], CARRY_OR_LATER).sum())
    return {"rate": (k / n) if n else None, "n_not_held": n, "n_false_carry": k}


def closed_metrics(y, p, run_ids, t_steps) -> dict:
    from AG3S.ag3s.experiments.studies.t64_phase6 import boundary_delays, prf

    y, p = np.asarray(y, int), np.asarray(p, int)
    pr, rc, f1, sup, cm = prf(y, p, K)
    run_ids, t_steps = np.asarray(run_ids), np.asarray(t_steps)
    pe, te, fe = [], [], []
    for r in dict.fromkeys(run_ids.tolist()):
        idx = np.flatnonzero(run_ids == r)
        idx = idx[np.argsort(t_steps[idx], kind="stable")]
        pe.append(p[idx])
        te.append(y[idx])
        fe.append(t_steps[idx])
    bd = boundary_delays(pe, te, fe, K)
    return {"n": int(len(y)), "false_carry": false_carry_rate(y, p), "accuracy": float(np.mean(y == p)) if len(y) else None,
            "macro_f1": float(np.mean(f1)),
            "per_class": {n: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                              "support": int(sup[i])} for i, n in enumerate(PHASES5)},
            "confusion": cm.tolist(),
            "delay_grasp_to_carry": bd.get("1->2"), "delay_carry_to_place": bd.get("2->3"), "boundary_delay": bd}


def verdict(hid: dict, kv: dict) -> dict:
    fh, fk = hid["test"]["false_carry"]["rate"], kv["test"]["false_carry"]["rate"]
    c1 = fh is not None and fk is not None and fh <= FC_FACTOR * fk
    c2 = hid["test"]["accuracy"] >= kv["test"]["accuracy"] - ACC_MARGIN
    c3 = hid["val"]["macro_f1"] >= kv["val"]["macro_f1"] - F1_MARGIN
    return {"false_carry_half_or_less": bool(c1), "accuracy_not_2pp_lower": bool(c2),
            "val_macro_f1_not_0.02_lower": bool(c3), "hidden_better": bool(c1 and c2 and c3),
            "values": {"false_carry": [fh, fk], "accuracy": [hid["test"]["accuracy"], kv["test"]["accuracy"]],
                       "val_macro_f1": [hid["val"]["macro_f1"], kv["val"]["macro_f1"]]}}


def select_feature(cv_best: dict) -> str:
    """Highest train-CV macro-F1 among hidden features; ties -> smaller feature, then lower layer."""
    def key(name):
        l, pool = name[1:].split("_", 1)
        return (-cv_best[name], len(POOLS[pool]), int(l))
    return min((n for n in cv_best if n != "kv_L4"), key=key)


def feature_names(layers: Sequence[int], pools: Sequence[str]) -> list[str]:
    return ["kv_L4"] + [f"h{l}_{p}" for l in layers for p in pools]


# ================================================================================ data


def safe_name(label: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in str(label))


def build_cache(cache: pathlib.Path, expert_dir: pathlib.Path, closed_dir: pathlib.Path, labels: pathlib.Path,
                layers: Sequence[int]) -> dict:
    """Row table ``rows.npz`` + memmaps ``kv.npy`` [N, 512] and ``L<l>.npy`` [N, 5, 2048] (float32)."""
    cache.mkdir(parents=True, exist_ok=True)
    srcs = []   # (file, row selector, meta per row)
    meta = collections.defaultdict(list)
    excl = collections.Counter()
    for split in ("train", "val"):
        for f in sorted((expert_dir / split).glob("ep_*.npz")):
            with np.load(f) as z:
                ph = np.asarray(z["phase_index"], int)
                keep = np.flatnonzero([p in MAP5 for p in ph])
                excl[f"expert_{split}_excluded_phase"] += int(len(ph) - len(keep))
                ep = int(z["episode"])
                fi = np.asarray(z["frame_index"], int)
            srcs.append((f, keep))
            for i in keep:
                meta["source"].append("expert")
                meta["split"].append(split)
                meta["group"].append(f"ep{ep}")
                meta["y"].append(MAP5[ph[i]])
                meta["t_step"].append(int(fi[i]))
                meta["hsr6_to5"].append(-1)
                meta["coarse3"].append("")
    lab = json.loads(labels.read_text())
    for run in lab["runs"]:
        f = closed_dir / f"{safe_name(run['label'])}.npz"
        rows = [r for r in run["rows"] if "gt" in r and r["gt"].get("phase5")]
        if not f.exists():
            excl["closed_runs_without_features"] += 1
            excl["closed_rows_without_features"] += len(rows)
            continue
        with np.load(f) as z:
            ts = np.asarray(z["t_step"], int)
        pos = {int(t): i for i, t in enumerate(ts)}
        keep = []
        for r in rows:
            i = pos.get(int(r["t_step"]))
            if i is None:
                excl["closed_rows_without_feature_frame"] += 1
                continue
            keep.append(i)
            meta["source"].append("closed")
            meta["split"].append("test" if run["split"] == "test" else "train")
            meta["group"].append(f"cl:{run['label']}")
            meta["y"].append(PHASES5.index(r["gt"]["phase5"]))
            meta["t_step"].append(int(r["t_step"]))
            s5 = (r.get("serving") or {}).get("hsr6_to5")
            meta["hsr6_to5"].append(PHASES5.index(s5) if s5 in PHASES5 else -1)
            meta["coarse3"].append((r.get("serving") or {}).get("subtask_label") or "")
        srcs.append((f, np.asarray(keep, int)))
    n = len(meta["y"])
    kvm = np.lib.format.open_memmap(cache / "kv.npy", "w+", np.float32, (n, 512))
    lm = {l: np.lib.format.open_memmap(cache / f"L{l}.npy", "w+", np.float32, (n, 5, 2048)) for l in layers}
    o = 0
    for f, keep in srcs:
        if not len(keep):
            continue
        with np.load(f) as z:
            fl = [int(x) for x in z["layers"]]
            missing = [l for l in layers if l not in fl]
            if missing:
                raise SystemExit(f"{f}: layers {missing} not stored (has {fl})")
            kvm[o:o + len(keep)] = z["kv_L4"][keep]
            H = z["hidden_summary"]
            for l in layers:
                lm[l][o:o + len(keep)] = H[keep, fl.index(l)]
        o += len(keep)
    assert o == n
    kvm.flush()
    for m in lm.values():
        m.flush()
    np.savez(cache / "rows.npz", **{k: np.asarray(v) for k, v in meta.items()})
    info = {"n_rows": n, "excluded": dict(excl), "layers": list(layers),
            "rows_by_source_split": dict(collections.Counter(f"{a}/{b}" for a, b in zip(meta["source"], meta["split"])))}
    (cache / "cache.json").write_text(json.dumps(info, indent=1))
    return info


def load_X(cache: pathlib.Path, feature: str) -> np.ndarray:
    if feature == "kv_L4":
        return np.load(cache / "kv.npy", mmap_mode="r")
    l, pool = feature[1:].split("_", 1)
    A = np.load(cache / f"L{int(l)}.npy", mmap_mode="r")
    return np.asarray(A[:, list(POOLS[pool])]).reshape(len(A), -1)


# ================================================================================ fitting (one worker per feature)


def run_feature(cache: str, feature: str, c_grid: Sequence[float], models_dir: str) -> dict:
    from AG3S.ag3s.experiments.studies.t50_core import group_kfold
    from AG3S.ag3s.experiments.studies.t64_phase6 import fit, predict, prf

    cache = pathlib.Path(cache)
    t0 = time.time()
    rows = np.load(cache / "rows.npz")
    X = np.asarray(load_X(cache, feature), np.float32)
    split, y, g, src = rows["split"], rows["y"].astype(int), rows["group"], rows["source"]
    tr = np.flatnonzero(split == "train")
    Xtr, ytr, gtr = X[tr], y[tr], g[tr]
    folds = group_kfold(gtr, 5)
    cv = {}
    for C in c_grid:
        s = []
        for a, b in folds:
            s.append(float(np.mean(prf(ytr[b], predict(fit(Xtr[a], ytr[a], C, K), Xtr[b]), K)[2])))
        cv[C] = float(np.mean(s))
    best = max(c_grid, key=lambda c: (cv[c], -c))
    model = fit(Xtr, ytr, best, K)
    out = {"feature": feature, "dim": int(X.shape[1]), "C": best, "cv_macro_f1": {str(c): v for c, v in cv.items()},
           "cv_best": cv[best], "converged": bool(model.converged), "n_iter": int(model.n_iter)}
    for part in ("val", "test"):
        idx = np.flatnonzero(split == part)
        out[f"pred_{part}"] = predict(model, X[idx]).tolist() if len(idx) else []
    np.savez(pathlib.Path(models_dir) / f"{safe_name(feature)}.npz", classes=np.array(PHASES5), mean=model.mean,
             scale=model.scale, coef=model.coef, intercept=model.intercept, C=best, feature=feature)
    out["seconds"] = round(time.time() - t0, 1)
    print(f"[t77_probe] {feature}: C={best} cv={cv[best]:.4f} ({out['seconds']}s)", flush=True)
    return out


def evaluate(res: dict, rows) -> dict:
    split, y = rows["split"], rows["y"].astype(int)
    vi, ti = np.flatnonzero(split == "val"), np.flatnonzero(split == "test")
    from AG3S.ag3s.experiments.studies.t64_phase6 import prf

    pv = np.asarray(res["pred_val"], int)
    f1 = prf(y[vi], pv, K)[2] if len(vi) else np.zeros(K)
    return {"val": {"n": int(len(vi)), "macro_f1": float(np.mean(f1)), "accuracy": float(np.mean(pv == y[vi])) if len(vi) else None},
            "test": closed_metrics(y[ti], np.asarray(res["pred_test"], int), rows["group"][ti], rows["t_step"][ti])}


def serving_reference(rows) -> dict:
    ti = np.flatnonzero(rows["split"] == "test")
    y = rows["y"].astype(int)[ti]
    h5 = rows["hsr6_to5"].astype(int)[ti]
    have = h5 >= 0
    out = {"hsr6_to5": {"n_with_hsr6": int(have.sum())}}
    if have.any():
        out["hsr6_to5"].update(closed_metrics(y[have], h5[have], rows["group"][ti][have], rows["t_step"][ti][have]))
    c3 = rows["coarse3"][ti]
    have3 = c3 != ""
    nh = np.isin(y, NOT_HELD) & have3
    out["subtask3"] = {"n_with_label": int(have3.sum()), "n_not_held": int(nh.sum()),
                       "false_carry_coarse": (float(np.isin(c3[nh], ("place", "home")).mean()) if nh.any() else None),
                       "coarse_accuracy": (float(np.mean([COARSE_OF_5[PHASES5[a]] == b for a, b in zip(y[have3], c3[have3])]))
                                           if have3.any() else None)}
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--expert-dir", required=True)
    ap.add_argument("--closed-dir", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=list(LAYERS))
    ap.add_argument("--pools", nargs="+", default=list(POOLS), choices=list(POOLS))
    ap.add_argument("--c-grid", type=float, nargs="+", default=None, help="default t64_phase6.C_GRID")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (one feature each); pod limit 4")
    ap.add_argument("--rebuild-cache", action="store_true")
    args = ap.parse_args(argv)
    from AG3S.ag3s.experiments.studies.t64_phase6 import C_GRID

    if args.jobs > 4:
        raise SystemExit("--jobs > 4 (pod worker limit)")
    out = pathlib.Path(args.out)
    cache = out / "cache"
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    if args.rebuild_cache or not (cache / "rows.npz").exists():
        info = build_cache(cache, pathlib.Path(args.expert_dir), pathlib.Path(args.closed_dir),
                           pathlib.Path(args.labels), args.layers)
    else:
        info = json.loads((cache / "cache.json").read_text())
        if not set(args.layers) <= set(info["layers"]):
            raise SystemExit(f"cache has layers {info['layers']}; pass --rebuild-cache")
    print(f"[t77_probe] rows {info['rows_by_source_split']}, excluded {info['excluded']}", flush=True)
    rows = dict(np.load(cache / "rows.npz"))
    grid = tuple(args.c_grid) if args.c_grid else C_GRID
    feats = feature_names(args.layers, args.pools)
    if args.jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("spawn")) as ex:
            res = dict(zip(feats, ex.map(run_feature, [str(cache)] * len(feats), feats, [grid] * len(feats),
                                         [str(models)] * len(feats))))
    else:
        res = {f: run_feature(str(cache), f, grid, str(models)) for f in feats}
    per = {f: {**{k: v for k, v in r.items() if not k.startswith("pred_")}, **evaluate(r, rows)} for f, r in res.items()}
    sel = select_feature({f: r["cv_best"] for f, r in res.items()})
    ti = np.flatnonzero(rows["split"] == "test")
    preds = {"group": rows["group"][ti].tolist(), "t_step": rows["t_step"][ti].tolist(), "gt": rows["y"][ti].tolist(),
             "kv_L4": res["kv_L4"]["pred_test"], sel: res[sel]["pred_test"],
             "hsr6_to5": rows["hsr6_to5"][ti].tolist(), "subtask3": rows["coarse3"][ti].tolist()}
    result = {"step": "T77", "definitions": __doc__, "phases5": list(PHASES5), "c_grid": list(grid),
              "cache": info, "features": per, "selected_hidden": sel,
              "selection_rule": "max train-CV macro-F1 over hidden features; ties -> smaller dim, lower layer",
              "verdict": verdict(per[sel], per["kv_L4"]), "serving_reference": serving_reference(rows),
              "layer_curve": {p: {str(l): {"cv": per[f"h{l}_{p}"]["cv_best"],
                                           "test_false_carry": per[f"h{l}_{p}"]["test"]["false_carry"]["rate"],
                                           "test_accuracy": per[f"h{l}_{p}"]["test"]["accuracy"],
                                           "test_macro_f1": per[f"h{l}_{p}"]["test"]["macro_f1"],
                                           "val_macro_f1": per[f"h{l}_{p}"]["val"]["macro_f1"]}
                                  for l in args.layers} for p in args.pools},
              "test_predictions": preds}
    (out / "t77_probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float))
    v = result["verdict"]
    print(f"[t77_probe] selected {sel}; verdict hidden_better={v['hidden_better']} {v['values']}")


if __name__ == "__main__":
    main()
