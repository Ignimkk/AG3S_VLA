#!/usr/bin/env python3
"""P4 (제약 단계 추종, 2026-11-10 KST): which per-request signal reads the constraint stages best?

Rules fixed before results: `outputs/2026-11-10_제약단계_추종/README.md` §5 (+ A2, A3).
  labels    P2 truth per request (`p2/p2_requests.csv`, S1..S4), events (`p2/p2_episodes.json`)
  features  GRIP, ACT (records, no GPU), V-L{l} / K-L{l} (l = 0..17), H-L{l} (l = 0..18), kvL4 (P3 `p3/features`)
            hidden rows used: agentview mean, wrist mean, valid-text mean (rows 0, 1, 2)
  model     [train-fold PCA 128 for hidden sets] -> StandardScaler + multinomial logistic (`t50_core.fit_logreg`)
  split     leave-one-group-out over {LIBERO, SafeLIBERO11, E1a-t0..t3}; C (and for 'auto' the layer) chosen by
            inner episode-level 3-fold accuracy inside the training groups
  score     S3 miss (truth S3, pred != S3), false HELD (truth S1/S2, pred S3), undetected E-fail
            (no S1 prediction between the event and its horizon), error sum = the three
  adopt     error sum <= 0.7 x (GRIP+ACT) and false HELD not above GRIP+ACT

usage: .venv-ag3s/bin/python experiments/segmentation/stage_track/p4_feature_compare.py OUT_DIR [--workers 2]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path("/mnt/dev/AG3S_VLA")
sys.path.insert(0, str(ROOT))
RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
VSUFFIX = "-vla_v11ab-AG3S-commonto-att-termsoff-pass-zr-ee-hdp-loose2-obst4-pts-nocarve-fh-t100-pe123458-C"
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)
PCA_DIM, ROWS, K_CLS = 128, (0, 1, 2), 4
LATCH = ("searching", "latched", "closing", "held", "placed")
GROUPS = ("LIBERO", "SafeLIBERO11", "E1a-t0", "E1a-t1", "E1a-t2", "E1a-t3")

D: dict = {}                                            # filled once per process (fork inherits)


def load(root: pathlib.Path):
    rows = list(csv.DictReader(open(root / "p2" / "p2_requests.csv")))
    eps = json.loads((root / "p2" / "p2_episodes.json").read_text())
    key = lambda r: (r["tag"], r["stem"])
    by = collections.defaultdict(list)
    for r in rows:
        by[key(r)].append(r)
    feats = collections.defaultdict(list)
    meta = []
    for e in eps:
        rs = by[(e["tag"], e["stem"])]
        run = sorted(RESULTS.glob(f"*_{e['tag']}{VSUFFIX}"))[-1]
        F = np.load(root / "p3" / "features" / e["tag"] / f"{e['stem']}.npz")
        obs = [int(r["obs"]) for r in rs]
        if [int(x) for x in F["obs_index"]] != obs:
            raise ValueError(f"{e['tag']}/{e['stem']}: P3 requests differ from P2")
        E = np.load(run / "episodes" / f"{e['stem']}.npz")
        Cn = np.load(run / "episodes" / f"{e['stem']}_calls.npz")
        calls = [json.loads(l) for l in (run / "episodes" / f"{e['stem']}_calls.jsonl").read_text().splitlines() if l.strip()]
        q = E["obs_robot0_gripper_qpos"]
        gap = (q[:, 0] - q[:, 1]) * 1e3
        for k, o in enumerate(obs):
            g_prev = gap[obs[k - 1]] if k else gap[o]
            lat = np.zeros(len(LATCH)); s = (calls[k].get("grasp") or {}).get("state")
            if s in LATCH:
                lat[LATCH.index(s)] = 1
            meas = (calls[k].get("latch_signal") or {}).get("value")
            last_cmd = float(Cn["actions_executed"][k - 1][-1, 6]) if k else -1.0
            feats["GRIP"].append(np.r_[gap[o], g_prev, gap[o] - g_prev, meas if meas is not None else 1.0, lat, last_cmd])
            feats["ACT"].append(Cn["actions_policy"][k].reshape(-1))
        feats["H"].append(F["hidden"][:, :, ROWS, :].reshape(len(obs), F["hidden"].shape[1], -1))     # [n, 19, 3*2048] f16
        feats["V"].append(F["V"][:, :, ROWS, :].reshape(len(obs), F["V"].shape[1], -1).astype(np.float32))
        feats["K"].append(F["K"][:, :, ROWS, :].reshape(len(obs), F["K"].shape[1], -1).astype(np.float32))
        feats["kvL4"].append(F["kv_L4"].astype(np.float32))
        for k, r in enumerate(rs):
            meta.append({"group": e["group"], "ep": f"{e['tag']}/{e['stem']}", "obs": int(r["obs"]), "truth": int(r["truth"]),
                         "success": e["success"]})
    X = {"GRIP": np.asarray(feats["GRIP"], np.float64), "ACT": np.asarray(feats["ACT"], np.float64),
         "H": np.concatenate(feats["H"]), "V": np.concatenate(feats["V"]), "K": np.concatenate(feats["K"]),
         "kvL4": np.concatenate(feats["kvL4"])}
    return X, meta, eps


def pca_fit(X, dim, seed=0):
    """Randomized PCA (rank `dim`, 4 power iterations) on the training rows -> (mean, components [dim, d])."""
    mu = X.mean(0)
    Xc = X - mu
    rng = np.random.default_rng(seed)
    Q = Xc @ rng.standard_normal((X.shape[1], dim + 16))
    for _ in range(4):
        Q, _ = np.linalg.qr(Xc @ (Xc.T @ Q))
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    return mu, Vt[:dim]


def design(name, idx_fit, idx_apply):
    """Feature matrix for set `name` on rows `idx_apply`, any projection fitted on `idx_fit` only."""
    parts, fam, layer, plus = name.split("+")[0], None, None, name.endswith("+GRIP")
    out = []
    if parts in ("GRIP", "ACT"):
        out.append(D["X"][parts][idx_apply])
    elif parts == "GRIP&ACT":
        out += [D["X"]["GRIP"][idx_apply], D["X"]["ACT"][idx_apply]]
    else:
        if parts == "kvL4":
            src = D["X"]["kvL4"]
            get = lambda ix: src[ix].astype(np.float64)
        else:
            fam, layer = parts.split("-L")
            src = D["X"][fam]
            get = lambda ix: src[ix, int(layer)].astype(np.float64)
        Xf = get(idx_fit)
        if Xf.shape[1] > PCA_DIM:
            mu, comp = pca_fit(Xf, PCA_DIM)
            out.append((get(idx_apply) - mu) @ comp.T)
        else:
            out.append(get(idx_apply))
    if plus:
        out.append(D["X"]["GRIP"][idx_apply])
    return np.concatenate(out, 1)


def fit_predict(name, tr, te, C):
    from AG3S.ag3s.experiments.studies.t50_core import fit_logreg

    y = D["y"]
    Xtr, Xte = design(name, tr, tr), design(name, tr, te)
    present = np.unique(y[tr])
    remap = {c: i for i, c in enumerate(present)}
    m = fit_logreg(Xtr, np.array([remap[c] for c in y[tr]]), C, n_classes=len(present), max_iter=2000, tol=1e-4)
    return present[np.argmax(m.proba(Xte), axis=1)]


def run_set(name):
    """Outer LOGO with nested C: returns {'pred': [N], 'inner': {group: {C: acc}}, 'C': {group: C}}."""
    from AG3S.ag3s.experiments.studies.t50_core import group_kfold

    t0 = time.time()
    y, groups, eps = D["y"], D["groups"], D["eps_id"]
    pred = np.full(len(y), -1)
    inner_all, chosen = {}, {}
    for g in GROUPS:
        te = np.flatnonzero(groups == g)
        tr = np.flatnonzero(groups != g)
        if not len(te):
            continue
        scores = {}
        for C in C_GRID:
            acc = []
            for itr, ite in group_kfold(eps[tr], 3):
                acc.append(float((fit_predict(name, tr[itr], tr[ite], C) == y[tr[ite]]).mean()))
            scores[C] = float(np.mean(acc))
        best = max(C_GRID, key=lambda c: (scores[c], -c))
        pred[te] = fit_predict(name, tr, te, best)
        inner_all[g], chosen[g] = scores, best
    return name, {"pred": pred.tolist(), "inner": inner_all, "C": chosen, "seconds": time.time() - t0}


def score(pred, meta, eps, mask=None):
    pred = np.asarray(pred); y = np.array([m["truth"] for m in meta])
    sel = np.ones(len(y), bool) if mask is None else mask
    s3 = int(((y == 2) & (pred != 2) & sel).sum())
    fh = int((np.isin(y, (0, 1)) & (pred == 2) & sel).sum())
    fh2 = int(((y == 1) & (pred == 2) & sel).sum())
    by = collections.defaultdict(list)
    for i, m in enumerate(meta):
        by[m["ep"]].append(i)
    und = n_ev = 0
    for e in eps:
        ix = by[f"{e['tag']}/{e['stem']}"]
        if mask is not None and not sel[ix].any():
            continue
        for ev in e["event_detection"]:
            if ev["type"] != "fail" or ev["n_req_window"] == 0:
                continue
            win = [i for i in ix if ev["t"] <= meta[i]["obs"] < ev["horizon"]]
            n_ev += 1
            und += int(not any(pred[i] == 0 for i in win))
    acc = float((pred[sel] == y[sel]).mean())
    f1 = []
    for c in range(K_CLS):
        tp = int(((y == c) & (pred == c) & sel).sum()); fp = int(((y != c) & (pred == c) & sel).sum()); fn = int(((y == c) & (pred != c) & sel).sum())
        if tp + fp + fn:
            f1.append(2 * tp / (2 * tp + fp + fn))
    return {"S3_miss": s3, "S3_total": int(((y == 2) & sel).sum()), "false_held": fh, "false_held_in_S2": fh2,
            "E_fail_undetected": und, "E_fail": n_ev, "error_sum": s3 + fh + und, "acc4": acc, "macro_f1_4": float(np.mean(f1))}


def auto(results, family, plus):
    """Per outer group, pick the (layer, C) of `family` with the best inner accuracy, use its outer prediction."""
    names = [n for n in results if n.startswith(family + "-L") and n.endswith("+GRIP") == plus]
    pred = np.full(len(D["y"]), -1)
    pick = {}
    for g in GROUPS:
        te = np.flatnonzero(D["groups"] == g)
        best = max(names, key=lambda n: (results[n]["inner"][g][results[n]["C"][g]], -int(n.split("-L")[1].split("+")[0])))
        pred[te] = np.asarray(results[best]["pred"])[te]
        pick[g] = best
    return pred, pick


def figures(table, out, latch_score):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from AG3S.ag3s.experiments.common import figstyle

    figstyle.use_korean()
    CAT = figstyle.CATEGORICAL
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for i, (fam, plus) in enumerate([("V", False), ("K", False), ("H", False), ("V", True), ("H", True)]):
        names = sorted([n for n in table if n.startswith(fam + "-L") and n.endswith("+GRIP") == plus],
                       key=lambda n: int(n.split("-L")[1].split("+")[0]))
        ls = [int(n.split("-L")[1].split("+")[0]) for n in names]
        ax[0].plot(ls, [table[n]["error_sum"] for n in names], marker="o", ms=3, color=CAT[i % len(CAT)],
                   ls="--" if plus else "-", label=fam + (" + GRIP" if plus else ""))
    for nm, c, st in (("GRIP&ACT", "k", "-"), ("GRIP", "gray", ":")):
        ax[0].axhline(table[nm]["error_sum"], color=c, ls=st, lw=1, label=nm)
    ax[0].axhline(0.7 * table["GRIP&ACT"]["error_sum"], color="k", ls="--", lw=0.8, label="adopt line (0.7 x GRIP&ACT)")
    ax[0].axhline(latch_score["error_sum"], color="red", ls="-.", lw=1, label="current latch (pe123458)")
    ax[0].set_xlabel("layer"); ax[0].set_ylabel("error sum (S3 miss + false HELD + undetected E-fail)")
    ax[0].set_title("leave-one-group-out, per layer (descriptive)"); ax[0].legend(fontsize=7, ncol=2)
    keys = ["latch", "GRIP", "ACT", "GRIP&ACT", "kvL4", "V-auto", "K-auto", "H-auto", "V-auto+GRIP", "H-auto+GRIP"]
    vals = [latch_score if k == "latch" else table[k] for k in keys]
    x = np.arange(len(keys)); w = 0.27
    ax[1].bar(x - w, [v["S3_miss"] for v in vals], w, label="S3 miss", color=CAT[0])
    ax[1].bar(x, [v["false_held"] for v in vals], w, label="false HELD", color=CAT[1])
    ax[1].bar(x + w, [v["E_fail_undetected"] for v in vals], w, label="undetected E-fail", color=CAT[2])
    ax[1].set_xticks(x, keys, rotation=30, ha="right"); ax[1].legend(fontsize=8)
    ax[1].set_title("error counts (auto = layer chosen inside training folds)")
    figstyle.style_axes(fig, ax); fig.tight_layout(); fig.savefig(out / "그래프_P4_feature비교.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    root = pathlib.Path(args.out_root)
    out = root / "p4"
    out.mkdir(exist_ok=True)
    X, meta, eps = load(root)
    D.update({"X": X, "y": np.array([m["truth"] for m in meta]), "groups": np.array([m["group"] for m in meta]),
              "eps_id": np.array([m["ep"] for m in meta])})
    nL_kv, nL_h = X["V"].shape[1], X["H"].shape[1]
    sets = ["GRIP", "ACT", "GRIP&ACT", "kvL4"]
    sets += [f"V-L{l}" for l in range(nL_kv)] + [f"K-L{l}" for l in range(nL_kv)] + [f"H-L{l}" for l in range(nL_h)]
    sets += [f"V-L{l}+GRIP" for l in range(nL_kv)] + [f"H-L{l}+GRIP" for l in range(nL_h)]
    print(f"{len(meta)} requests, {len(sets)} feature sets, {args.workers} workers", flush=True)
    results = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for name, res in pool.map(run_set, sets):
            results[name] = res
            print(f"{time.strftime('%T')} {name}: {score(res['pred'], meta, eps)['error_sum']} ({res['seconds']:.0f} s)", flush=True)
    table = {n: score(r["pred"], meta, eps) for n, r in results.items()}
    picks = {}
    for fam, plus in (("V", False), ("K", False), ("H", False), ("V", True), ("H", True)):
        pred, pick = auto(results, fam, plus)
        nm = fam + "-auto" + ("+GRIP" if plus else "")
        table[nm], picks[nm] = score(pred, meta, eps), pick
        results[nm] = {"pred": pred.tolist()}
    lat_pred = np.array([{0: 0, 1: 0, 2: 1, 3: 2, 4: 3}.get(int(np.argmax(X["GRIP"][i, 4:9])), 0) if X["GRIP"][i, 4:9].any() else 0
                         for i in range(len(meta))])
    latch_score = score(lat_pred, meta, eps)
    base = table["GRIP&ACT"]
    verdict = {}
    for n in ("kvL4", "V-auto", "K-auto", "H-auto", "V-auto+GRIP", "H-auto+GRIP"):
        t = table[n]
        verdict[n] = {"error_sum": t["error_sum"], "vs_GRIP&ACT": t["error_sum"] / max(base["error_sum"], 1),
                      "adopt": t["error_sum"] <= 0.7 * base["error_sum"] and t["false_held"] <= base["false_held"]}
    per_group = {n: {g: score(results[n]["pred"], meta, eps, np.array([m["group"] == g for m in meta])) for g in GROUPS}
                 for n in ("GRIP&ACT", "V-auto", "H-auto", "V-auto+GRIP", "H-auto+GRIP", "kvL4")}
    rec = {"finished": time.strftime("%F %T %Z"), "n_requests": len(meta), "latch_current": latch_score, "table": table,
           "auto_picks": picks, "verdict": verdict, "per_group": per_group,
           "chosen_C": {n: r.get("C") for n, r in results.items() if "C" in r}}
    (out / "p4_summary.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    np.savez_compressed(out / "p4_predictions.npz", **{n.replace("&", "_and_").replace("+", "_plus_"): np.asarray(r["pred"])
                                                      for n, r in results.items()},
                        truth=D["y"], group=D["groups"], episode=D["eps_id"], obs=np.array([m["obs"] for m in meta]))
    figures(table, out, latch_score)
    print(json.dumps({"latch_current": latch_score, "GRIP&ACT": base, "verdict": verdict, "auto_picks": picks}, indent=1))


if __name__ == "__main__":
    main()
