#!/usr/bin/env python3
"""P5b (제약 단계 추종, 2026-11-10 KST): export the server stage-track asset and check that the server can rebuild its inputs.

  asset    H-L10 + GRIP (README §10): train-set PCA 128 on the hidden rows, StandardScaler + multinomial logistic, fitted on
           all 104 dev episodes; C = the value P4's nested CV chose most often for H-L10+GRIP.
  check 1  the asset format (`AG3S.trajopt.stage_track.StageTrackProbe`) reproduces `fit_logreg` probabilities (per LOGO fold).
  check 2  GRIP rebuilt with the server's rules (`stage_track.grip_features` on the request state, the previous request's
           state, `latch_signal.value`, `grasp.state`, `last_executed_command(actions_response[k-1], n_exec)`) vs P4's GRIP:
           column-wise max |diff| and the asset's prediction agreement.
  check 3  `stage_track.stage_hidden_feature` (numpy) on a synthetic hidden equals P3's pooling formula.

usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       .venv-ag3s/bin/python experiments/segmentation/stage_track/p5b_build_asset.py OUT_DIR ASSET.npz
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import p4_feature_compare as p4  # noqa: E402  (inserts the main checkout root; put this checkout first again)

sys.path.insert(0, str(REPO))

from AG3S.trajopt import stage_track as st  # noqa: E402

NAME, LAYER = "H-L10+GRIP", 10


def fit(tr, C):
    from AG3S.ag3s.experiments.studies.t50_core import fit_logreg

    Xh = p4.D["X"]["H"][tr, LAYER].astype(np.float64)
    mu, comp = p4.pca_fit(Xh, p4.PCA_DIM)
    Z = np.concatenate([(Xh - mu) @ comp.T, p4.D["X"]["GRIP"][tr]], 1)
    y = p4.D["y"][tr]
    m = fit_logreg(Z, y, C, n_classes=4, max_iter=2000, tol=1e-4)
    return st.StageTrackProbe(pca_mean=mu, pca_components=comp, mean=m.mean, scale=m.scale, coef=m.coef,
                              intercept=m.intercept, classes=st.STAGES, name="fold", md5="", meta={}), m


def server_grip(root, meta, eps):
    """GRIP per request with the server's rules, in `meta` order."""
    rows = []
    for e in eps:
        run = sorted(p4.RESULTS.glob(f"*_{e['tag']}{p4.VSUFFIX}"))[-1]
        stem = e["stem"]
        E = np.load(run / "episodes" / f"{stem}.npz")
        Cn = np.load(run / "episodes" / f"{stem}_calls.npz")
        calls = [json.loads(l) for l in (run / "episodes" / f"{stem}_calls.jsonl").read_text().splitlines() if l.strip()]
        prev_gap = None
        for k in range(len(E["chunk_t"])):
            gap = st.gap_mm_of_state(E["policy_state"][k])
            fb = calls[k].get("feedback_server") or {}
            last = st.last_executed_command(Cn["actions_response"][k - 1] if k else None, fb.get("n_exec"), fb.get("n_steps"))
            rows.append(st.grip_features(gap_mm=gap, prev_gap_mm=prev_gap,
                                         measured_opening=(calls[k].get("latch_signal") or {}).get("value"),
                                         latch_state=(calls[k].get("grasp") or {}).get("state"), last_command=last))
            prev_gap = gap
    G = np.asarray(rows)
    if len(G) != len(meta):
        raise ValueError(f"server GRIP has {len(G)} rows, P4 {len(meta)}")
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("asset")
    args = ap.parse_args()
    root = pathlib.Path(args.out_root)
    out = root / "p5b"
    out.mkdir(exist_ok=True)
    X, meta, eps = p4.load(root)
    p4.D.update({"X": X, "y": np.array([m["truth"] for m in meta]), "groups": np.array([m["group"] for m in meta]),
                 "eps_id": np.array([m["ep"] for m in meta])})
    s4 = json.loads((root / "p4" / "p4_summary.json").read_text())
    chosen = s4["chosen_C"][NAME]
    C = collections.Counter(chosen.values()).most_common(1)[0][0]
    ref = np.load(root / "p5a" / "p5a_proba.npz")["H-L10_plus_GRIP"]
    rec = {"started": time.strftime("%F %T %Z"), "C_per_fold": chosen, "C_final": C}

    # check 1 — asset format == fit_logreg, per LOGO fold (fold C as in P4 / P5a)
    P = np.zeros((len(meta), 4))
    err = 0.0
    for g in p4.GROUPS:
        te, tr = np.flatnonzero(p4.D["groups"] == g), np.flatnonzero(p4.D["groups"] != g)
        probe, m = fit(tr, chosen[g])
        Zte = probe.design(X["H"][te, LAYER].astype(np.float64), X["GRIP"][te])
        P[te] = probe.proba(X["H"][te, LAYER], X["GRIP"][te])
        err = max(err, float(np.abs(P[te] - m.proba(Zte)).max()))
    rec["check1"] = {"max_abs_vs_fit_logreg": err, "max_abs_vs_p5a": float(np.abs(P - ref).max()),
                     "argmax_equal_p5a": float((P.argmax(1) == ref.argmax(1)).mean())}

    # check 2 — server-rule GRIP
    G = server_grip(root, meta, eps)
    d = np.abs(G - X["GRIP"])
    rec["check2"] = {"col_max_abs_diff": d.max(0).round(6).tolist(),
                     "rows_any_diff_gt_1e-6": int((d.max(1) > 1e-6).sum()), "rows": len(G)}

    # final asset (all 104 episodes)
    allr = np.arange(len(meta))
    probe, m = fit(allr, C)
    pa = probe.proba(X["H"][:, LAYER], X["GRIP"]).argmax(1)
    ps = probe.proba(X["H"][:, LAYER], G).argmax(1)
    rec["check2"].update({"final_asset_pred_agreement_P4grip_vs_servergrip": float((pa == ps).mean()),
                          "final_asset_train_acc": float((pa == p4.D["y"]).mean()), "final_converged": bool(m.converged)})

    # check 3 — pooling formula on a synthetic hidden (P3 `pool`: camera means unmasked, text mean masked)
    rng = np.random.default_rng(0)
    T = 768 + 48
    h = rng.standard_normal((T, st.HIDDEN_DIM)).astype(np.float32)
    mask = np.r_[np.ones(768, bool), np.ones(20, bool), np.zeros(28, bool)]
    tm = mask[768:].astype(np.float32)
    p3 = np.concatenate([h[0:256].mean(0), h[256:512].mean(0), (h[768:] * tm[:, None]).sum(0) / tm.sum()])
    rec["check3"] = {"max_abs": float(np.abs(st.stage_hidden_feature(np, h, mask) - p3.astype(np.float16).astype(np.float32)).max())}

    git = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    asset = pathlib.Path(args.asset)
    asset.parent.mkdir(parents=True, exist_ok=True)
    np.savez(asset, pca_mean=probe.pca_mean, pca_components=probe.pca_components, mean=probe.mean, scale=probe.scale,
             coef=probe.coef, intercept=probe.intercept, classes=np.array(st.STAGES),
             meta_trained_on=np.array("dev split 104 ep / 3604 requests (outputs/2026-11-10_제약단계_추종 P2 truth)"),
             meta_feature=np.array(f"h_{LAYER} [agentview mean | wrist mean | valid-text mean] PCA {p4.PCA_DIM} + GRIP {st.GRIP_DIM}"),
             meta_C=np.array(float(C)), meta_git_head=np.array(git), meta_built=np.array(time.strftime("%F %T %Z")))
    loaded = st.StageTrackProbe.load(asset)
    rec["asset"] = {"path": str(asset), **loaded.record()}
    rec["finished"] = time.strftime("%F %T %Z")
    (out / "p5b_asset_record.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
    print(json.dumps(rec, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
