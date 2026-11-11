#!/usr/bin/env python3
"""P5a (제약 단계 추종, 2026-11-10 KST): fix the server-side stage estimator offline.

Rules fixed before results: `outputs/2026-11-10_제약단계_추종/README.md` §9.
  layers    H-L10 (+ neighbours H-L8, H-L12) with GRIP, C = P4's chosen C per outer group
  post      R = argmax, H = causal HMM forward filter (transition from training-group truth), tau sweep on p(S3)
  split     the same leave-one-group-out as P4; scoring = `p4_feature_compare.score`

usage: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       .venv-ag3s/bin/python experiments/segmentation/stage_track/p5a_estimator.py OUT_DIR
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import p4_feature_compare as p4  # noqa: E402

LAYERS = (10, 8, 12)
TAUS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


def proba_logo(name, chosen_c):
    """Outer LOGO probabilities [N, 4] (classes absent from a training fold get 0)."""
    from AG3S.ag3s.experiments.studies.t50_core import fit_logreg

    y, groups = p4.D["y"], p4.D["groups"]
    P = np.zeros((len(y), p4.K_CLS))
    for g in p4.GROUPS:
        te, tr = np.flatnonzero(groups == g), np.flatnonzero(groups != g)
        Xtr, Xte = p4.design(name, tr, tr), p4.design(name, tr, te)
        present = np.unique(y[tr])
        remap = {c: i for i, c in enumerate(present)}
        m = fit_logreg(Xtr, np.array([remap[c] for c in y[tr]]), chosen_c[g], n_classes=len(present),
                       max_iter=2000, tol=1e-4)
        P[np.ix_(te, present)] = m.proba(Xte)
    return P


def episodes_index(meta):
    by = collections.OrderedDict()
    for i, m in enumerate(meta):
        by.setdefault(m["ep"], []).append(i)
    return by


def hmm_filter(P, meta):
    """Causal forward filter per episode; transition / prior from the other groups' truth sequences."""
    y, groups = p4.D["y"], p4.D["groups"]
    by = episodes_index(meta)
    out = np.zeros_like(P)
    trans = {}
    for g in p4.GROUPS:
        A = np.ones((p4.K_CLS, p4.K_CLS))
        prior = np.ones(p4.K_CLS)
        init = np.ones(p4.K_CLS)
        for ix in by.values():
            if groups[ix[0]] == g:
                continue
            s = y[ix]
            init[s[0]] += 1
            np.add.at(prior, s, 1)
            np.add.at(A, (s[:-1], s[1:]), 1)
        A /= A.sum(1, keepdims=True)
        prior /= prior.sum()
        init /= init.sum()
        trans[g] = A.round(4).tolist()
        for ix in by.values():
            if groups[ix[0]] != g:
                continue
            b = init
            for i in ix:
                like = P[i] / prior
                post = like * b
                post = post / post.sum() if post.sum() > 0 else np.full(p4.K_CLS, 1 / p4.K_CLS)
                out[i] = post
                b = post @ A
    return out, trans


def tau_pred(P, tau):
    pred = np.argmax(P, 1)
    low = (pred == 2) & (P[:, 2] < tau)
    Q = P.copy()
    Q[:, 2] = -1
    pred[low] = np.argmax(Q[low], 1)
    return pred


def figure(res, latch, out):
    import matplotlib.pyplot as plt

    from AG3S.ag3s.experiments.common import figstyle

    figstyle.use_korean()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for j, (name, r) in enumerate(res.items()):
        c = figstyle.CATEGORICAL[j % len(figstyle.CATEGORICAL)]
        for post, ls in (("R", "-"), ("H", "--")):
            pts = [r["tau"][post][str(t)] for t in TAUS]
            ax[0].plot([p["false_held"] for p in pts], [p["S3_miss"] for p in pts], ls, marker="o", color=c, ms=3,
                       label=f"{name} {'argmax' if post == 'R' else 'HMM'}")
            ax[1].plot(TAUS, [p["error_sum"] for p in pts], ls, marker="o", color=c, ms=3)
    ax[0].scatter([latch["false_held"]], [latch["S3_miss"]], color="red", marker="*", s=140, zorder=5,
                  label="current latch")
    ax[0].set_xlabel("false HELD (requests)")
    ax[0].set_ylabel("S3 miss (requests)")
    ax[0].set_title("trade-off along tau (p(S3) >= tau to call HELD)")
    ax[0].legend(fontsize=8)
    ax[1].axhline(latch["error_sum"], color="red", ls="-.", lw=1, label="current latch")
    ax[1].set_xlabel("tau")
    ax[1].set_ylabel("error sum (S3 miss + false HELD + undetected E-fail)")
    ax[1].set_title("error sum vs tau (leave-one-group-out)")
    ax[1].legend(fontsize=8)
    figstyle.style_axes(fig, ax)
    fig.suptitle("P5a stage estimator, dev split 104 ep / 3604 requests", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "그래프_P5a_tau_HMM.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    args = ap.parse_args()
    root = pathlib.Path(args.out_root)
    out = root / "p5a"
    out.mkdir(exist_ok=True)
    X, meta, eps = p4.load(root)
    p4.D.update({"X": X, "y": np.array([m["truth"] for m in meta]), "groups": np.array([m["group"] for m in meta]),
                 "eps_id": np.array([m["ep"] for m in meta])})
    s4 = json.loads((root / "p4" / "p4_summary.json").read_text())
    ref = np.load(root / "p4" / "p4_predictions.npz")
    latch = s4["latch_current"]
    res, probs = {}, {}
    for L in LAYERS:
        name = f"H-L{L}+GRIP"
        t0 = time.time()
        P = proba_logo(name, s4["chosen_C"][name])
        Ph, trans = hmm_filter(P, meta)
        r_arg = np.argmax(P, 1)
        same = float((r_arg == ref[f"H-L{L}_plus_GRIP"]).mean())
        r = {"reproduces_P4": same, "R": p4.score(r_arg, meta, eps), "H": p4.score(np.argmax(Ph, 1), meta, eps),
             "tau": {post: {str(t): p4.score(tau_pred(Q, t), meta, eps) for t in TAUS} for post, Q in (("R", P), ("H", Ph))},
             "transition": trans, "seconds": round(time.time() - t0, 1)}
        res[name], probs[name.replace("+", "_plus_")] = r, P
        print(f"{time.strftime('%T')} {name}: P4-repro {same:.3f}  R {r['R']['error_sum']} (fh {r['R']['false_held']})  "
              f"H {r['H']['error_sum']} (fh {r['H']['false_held']})  {r['seconds']} s", flush=True)
    sel = {}
    for name, r in res.items():
        base = "R" if r["R"]["error_sum"] <= r["H"]["error_sum"] else "H"
        pick = {"post": base, **r[base]}
        if r[base]["false_held"] > latch["false_held"]:
            ok = [t for t in TAUS if r["tau"][base][str(t)]["false_held"] <= latch["false_held"]
                  and r["tau"][base][str(t)]["error_sum"] < latch["error_sum"]]
            pick["tau_variant"] = ({"tau": min(ok), **r["tau"][base][str(min(ok))]} if ok else None)
        sel[name] = pick
    rec = {"finished": time.strftime("%F %T %Z"), "latch_current": latch, "results": res, "selection": sel}
    (out / "p5a_summary.json").write_text(json.dumps(rec, indent=1))
    np.savez(out / "p5a_proba.npz", **probs, truth=p4.D["y"], group=p4.D["groups"], episode=p4.D["eps_id"])
    figure(res, latch, out)
    print(json.dumps(sel, indent=1))


if __name__ == "__main__":
    main()
