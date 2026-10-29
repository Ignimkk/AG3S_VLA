"""SL6 B3 + B4 — Panda physical phase labels per request and the HSR6-Panda probe (``SL6.task.md`` §3, CPU).

Inputs (read only):

* dev rollouts ``…/20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4`` (libero_spatial task 0–9 × init 1–4;
  ``episodes/<stem>.npz`` ``action`` / ``chunk_t``);
* their ``phase_export.py --finger-contacts`` tracks (``phase_<stem>.npz``, image-instant MuJoCo states);
* served ``kv_L4`` features ``…/20261013T094739_714c891415d9_sl4b-attention-audit-dev-e1-4/attention/<stem>_attn.npz``
  (``kv_feature``, ``obs_index`` — one row per policy request, the same call the server would label).

Never used (refused): LIBERO init 0 and anything that is not libero_spatial init 1–4 (task §3 / §4).

B3: `sl6_phase_labels.request_phases` per request. B4 (task §3 B4, RB-Y1 `t64_phase6` model): StandardScaler +
multinomial logistic regression (`t50_core.fit_logreg`, lbfgs, max_iter 3000, tol 1e-4), C from {0.003, 0.01, 0.03,
0.1, 0.3} by episode-grouped `t50_core.group_kfold(5)` mean macro-F1 (ties -> smaller C, as `t64_phase6.choose_C`).
Macro-F1 of a fold = mean F1 over the classes present in that fold's truth or prediction (sklearn ``f1_score(average=
"macro")``). Reported at the chosen C from the out-of-fold predictions of the same 5 folds: fold-mean macro-F1 (the
VB3 number), pooled macro-F1, per-class precision / recall / F1, confusion, and the grasp-onset delay in requests
(`t64_phase6.boundary_delays` on approach→grasp, plus first predicted grasp − first true grasp per episode). The probe
fitted on all 40 episodes is written in the ``probe_6.npz`` format (classes, mean, scale, coef, intercept, C, feature,
+ name) and read back through `AG3S.trajopt.hsr6.load_hsr6_probe` (B2).

usage (``.venv-ag3s``, CPU, one process)::

    python -m AG3S.ag3s.experiments.studies.sl6_hsr6_panda --out-run <new run dir> \\
        [--dev-run …] [--attn-run …] [--phase-dir <out-run>/phase]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
from typing import Any, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import sl6_phase_labels as SPL

STORAGE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
DEV_RUN = STORAGE / "20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4"
ATTN_RUN = STORAGE / "20261013T094739_714c891415d9_sl4b-attention-audit-dev-e1-4"
C_GRID = (0.003, 0.01, 0.03, 0.1, 0.3)
N_SPLITS = 5
FEATURE = "kv_L4"
PROBE_NAME = "hsr6_panda_kv_L4"
SUITE = "libero_spatial"
ALLOWED_INITS = (1, 2, 3, 4)
VB2_EPISODES = ("libero_spatial_t0_e1", "libero_spatial_t3_e1", "libero_spatial_t5_e1", "libero_spatial_t8_e1")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ data


def load_episodes(dev_run: pathlib.Path, attn_run: pathlib.Path, phase_dir: pathlib.Path) -> list[dict]:
    """One dict per dev episode: stem, task, init, features ``X (n, 512)``, request frames, labels, info."""
    rows = [json.loads(line) for line in (dev_run / "episodes.jsonl").read_text().splitlines() if line.strip()]
    index = {(r["task_id"], r["episode_idx"]): r for r in
             (json.loads(line) for line in (attn_run / "attention_index.jsonl").read_text().splitlines() if line.strip())}
    out = []
    for r in rows:
        stem = f"{r['suite']}_t{int(r['task_id'])}_e{int(r['episode_idx'])}"
        init = int(r["init_state_index"])
        if r["suite"] != SUITE or init not in ALLOWED_INITS:
            raise ValueError(f"{stem}: suite {r['suite']} init {init} is not a SL6 dev episode (libero_spatial init 1-4)")
        ep_path = dev_run / r["npz"]
        with np.load(ep_path) as e:
            actions, chunk_t = np.asarray(e["action"]), np.asarray(e["chunk_t"], np.int64)
        ai = index[(int(r["task_id"]), int(r["episode_idx"]))]
        if ai["npz_sha256"] != sha256(ep_path):
            raise ValueError(f"{stem}: the audit run read another episode npz")
        with np.load(attn_run / ai["attn_npz"]) as a:
            X = np.asarray(a["kv_feature"], np.float32)
            obs = np.asarray(a["obs_index"], np.int64)
            gstep = np.asarray(a["global_step"], np.int64)
        if not np.array_equal(obs, chunk_t):
            raise ValueError(f"{stem}: audit obs_index != episode chunk_t")
        track_path = phase_dir / f"phase_{stem}.npz"
        with np.load(track_path) as t:
            track = {k: t[k] for k in t.files if not k.startswith("segP/")}
        prov = json.loads(str(track["provenance_json"]))
        if prov.get("geom_npz_sha256") != ai["geom_npz_sha256"] or "finger_contacts" not in prov:
            raise ValueError(f"{stem}: phase track is not the finger-contact export of this episode's geometry npz")
        y, info = SPL.request_phases(track, actions, chunk_t)
        out.append({"stem": stem, "task_id": int(r["task_id"]), "init": init, "episode_idx": int(r["episode_idx"]),
                    "success": bool(r.get("official_success")), "X": X, "frames": chunk_t, "global_step": gstep,
                    "y": y, "info": info, "track_sha256": sha256(track_path), "attn_npz_sha256": sha256(attn_run / ai["attn_npz"])})
    return out


def label_counts(eps: Sequence[dict]) -> dict:
    K = len(SPL.PHASES5)
    total = np.zeros(K, np.int64)
    per_task: dict[str, list[int]] = {}
    for e in eps:
        c = np.bincount(e["y"], minlength=K)
        total += c
        per_task.setdefault(str(e["task_id"]), [0] * K)
        per_task[str(e["task_id"])] = [int(a + b) for a, b in zip(per_task[str(e["task_id"])], c)]
    return {"classes": list(SPL.PHASES5), "total": total.tolist(), "requests": int(total.sum()),
            "per_task": per_task,
            "per_episode": {e["stem"]: np.bincount(e["y"], minlength=K).tolist() for e in eps}}


# ------------------------------------------------------------------------------------------------ model


def fit(X, y, C, K):
    from AG3S.ag3s.experiments.studies.t50_core import fit_logreg

    return fit_logreg(X, y, C, n_classes=K, max_iter=3000, tol=1e-4)


def macro_f1_present(y: np.ndarray, p: np.ndarray) -> float:
    """sklearn ``f1_score(average="macro")``: mean F1 over the classes in ``y ∪ p``."""
    labels = sorted(set(np.asarray(y).tolist()) | set(np.asarray(p).tolist()))
    f = []
    for c in labels:
        tp = int(np.sum((y == c) & (p == c)))
        fp = int(np.sum((y != c) & (p == c)))
        fn = int(np.sum((y == c) & (p != c)))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def prf(y, p, K):
    cm = np.zeros((K, K), np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(K), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(K), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(K), where=(prec + rec) > 0)
    return prec, rec, f1, cm


def cross_validate(X, y, g, K, C_grid=C_GRID, n_splits=N_SPLITS) -> dict:
    """Episode-grouped k-fold for every C; returns per-C fold scores and out-of-fold predictions."""
    from AG3S.ag3s.experiments.studies.t50_core import group_kfold

    folds = group_kfold(g, n_splits)
    res = {}
    for C in C_grid:
        oof = np.full(len(y), -1, np.int64)
        scores, iters = [], []
        for tr, te in folds:
            m = fit(X[tr], y[tr], C, K)
            pr = np.argmax(m.proba(np.asarray(X[te], np.float64)), 1)
            oof[te] = pr
            scores.append(macro_f1_present(y[te], pr))
            iters.append({"n_iter": int(m.n_iter), "converged": bool(m.converged)})
        res[C] = {"fold_macro_f1": scores, "mean": float(np.mean(scores)), "oof": oof, "fits": iters}
    best = max(C_grid, key=lambda c: (res[c]["mean"], -c))
    return {"folds": [[int(x) for x in np.unique(g[te])] for _, te in folds], "per_C": res, "best_C": best}


def grasp_onset(eps: Sequence[dict], oof_by_stem: dict) -> dict:
    """First predicted grasp request − first true grasp request, per episode with a true grasp; and the
    `t64_phase6.boundary_delays` summary of approach→grasp (frames = requests)."""
    from AG3S.ag3s.experiments.studies.t64_phase6 import boundary_delays

    per, preds, trues, idx = {}, [], [], []
    for e in eps:
        y, p = e["y"], oof_by_stem[e["stem"]]
        preds.append(p)
        trues.append(y)
        idx.append(np.arange(len(y)))
        t0 = np.flatnonzero(y == 1)
        if not len(t0):
            per[e["stem"]] = None
            continue
        p0 = np.flatnonzero(p == 1)
        per[e["stem"]] = None if not len(p0) else int(p0[0] - t0[0])
    d = [v for v in per.values() if v is not None]
    bd = boundary_delays(preds, trues, idx, len(SPL.PHASES5))
    return {"first_pred_minus_first_true_requests": per,
            "summary": {"n": len(d), "never_predicted": sum(v is None for k, v in per.items()
                                                            if np.any(next(e for e in eps if e["stem"] == k)["y"] == 1)),
                        "median": float(np.median(d)) if d else None, "mean": float(np.mean(d)) if d else None,
                        "min": int(min(d)) if d else None, "max": int(max(d)) if d else None},
            "boundary_delays_requests": bd}


def save_probe(path: pathlib.Path, model, C: float) -> None:
    with open(path, "xb") as f:
        np.savez(f, classes=np.array(SPL.PHASES5), mean=model.mean, scale=model.scale, coef=model.coef,
                 intercept=model.intercept, C=float(C), feature=FEATURE, name=PROBE_NAME)


# ------------------------------------------------------------------------------------------------ main


def vb2_bands(eps: Sequence[dict], stems=VB2_EPISODES) -> dict:
    out = {}
    for e in eps:
        if e["stem"] not in stems:
            continue
        fl = e["info"]["frame_labels"]
        out[e["stem"]] = {
            "classes": list(SPL.PHASES5),
            "requests": [{"k": k, "frame": int(f), "global_step": int(s), "label": SPL.PHASES5[int(c)]}
                         for k, (f, s, c) in enumerate(zip(e["frames"], e["global_step"], e["y"]))],
            "frame_band": "".join("AGLTP"[int(c)] for c in fl),
            "frame_band_legend": "A approach, G grasp, L lift, T transport, P placement (one char per recorded frame)",
            "events": {k: v for k, v in e["info"].items() if k != "frame_labels"},
            "video_note": "frame = observation index of the episode npz / recorded video; request k is served on it"}
    return out


def run(out_run: pathlib.Path, dev_run: pathlib.Path, attn_run: pathlib.Path, phase_dir: pathlib.Path) -> dict:
    from AG3S.trajopt.hsr6 import load_hsr6_probe

    eps = load_episodes(dev_run, attn_run, phase_dir)
    K = len(SPL.PHASES5)
    labels_dir, probe_dir = out_run / "labels", out_run / "probe"
    labels_dir.mkdir(exist_ok=False)
    probe_dir.mkdir(exist_ok=False)
    counts = label_counts(eps)
    with open(labels_dir / "request_labels.jsonl", "x") as f:
        for e in eps:
            f.write(json.dumps({"stem": e["stem"], "task_id": e["task_id"], "init": e["init"], "success": e["success"],
                                "frames": e["frames"].tolist(), "global_step": e["global_step"].tolist(),
                                "labels": [SPL.PHASES5[int(c)] for c in e["y"]],
                                "frame_band": "".join("AGLTP"[int(c)] for c in e["info"]["frame_labels"]),
                                "info": {k: v for k, v in e["info"].items() if k != "frame_labels"},
                                "track_sha256": e["track_sha256"], "attn_npz_sha256": e["attn_npz_sha256"]}) + "\n")
    (labels_dir / "vb2_bands.json").write_text(json.dumps(vb2_bands(eps), indent=1) + "\n")
    X = np.concatenate([e["X"] for e in eps]).astype(np.float64)
    y = np.concatenate([e["y"] for e in eps])
    g = np.concatenate([np.full(len(e["y"]), i) for i, e in enumerate(eps)])
    cv = cross_validate(X, y, g, K)
    C = cv["best_C"]
    oof = cv["per_C"][C]["oof"]
    pr, rc, f1, cm = prf(y, oof, K)
    oof_by_stem, i0 = {}, 0
    for e in eps:
        oof_by_stem[e["stem"]] = oof[i0:i0 + len(e["y"])]
        i0 += len(e["y"])
    model = fit(X, y, C, K)
    probe_path = probe_dir / f"{PROBE_NAME}.npz"
    save_probe(probe_path, model, C)
    served = load_hsr6_probe(probe_path)                       # B2 path: ordered subset -> padded predict
    served_arg = np.array([SPL.PHASES5.index(served.predict(x)["argmax"]) for x in X])
    fit_arg = np.argmax(model.proba(X), 1)
    res = {
        "step": "SL6-B", "definition_b3": SPL.DEFINITION, "rule_version": SPL.RULE_VERSION,
        "inputs": {"dev_run": str(dev_run), "attn_run": str(attn_run), "phase_dir": str(phase_dir),
                   "episodes": len(eps), "inits": sorted({e["init"] for e in eps}),
                   "tasks": sorted({e["task_id"] for e in eps})},
        "label_counts": counts,
        "episode_events": {e["stem"]: {k: v for k, v in e["info"].items() if k != "frame_labels"} for e in eps},
        "model": "StandardScaler + multinomial logistic regression (t50_core.fit_logreg, lbfgs, max_iter 3000, tol 1e-4)",
        "C_grid": list(C_GRID), "cv_folds_episode_index": cv["folds"],
        "cv_by_C": {str(c): {"fold_macro_f1": v["fold_macro_f1"], "mean": v["mean"], "fits": v["fits"]}
                    for c, v in cv["per_C"].items()},
        "chosen_C": C,
        "oof_at_chosen_C": {
            "fold_mean_macro_f1": cv["per_C"][C]["mean"],
            "pooled_macro_f1": macro_f1_present(y, oof),
            "accuracy": float(np.mean(oof == y)),
            "per_class": {n: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                              "support": int(cm[i].sum())} for i, n in enumerate(SPL.PHASES5)},
            "confusion_rows_truth_cols_pred": cm.tolist(),
            "grasp_onset": grasp_onset(eps, oof_by_stem)},
        "vb3": {"rule": "fold-mean macro-F1 >= 0.70 and grasp recall >= 0.6 -> usable for serving",
                "macro_f1": cv["per_C"][C]["mean"], "grasp_recall": float(rc[1]),
                "pass": bool(cv["per_C"][C]["mean"] >= 0.70 and rc[1] >= 0.6)},
        "probe": {"path": str(probe_path), "sha256": sha256(probe_path),
                  "md5": hashlib.md5(probe_path.read_bytes()).hexdigest(), "classes": list(SPL.PHASES5),
                  "n_iter": int(model.n_iter), "converged": bool(model.converged),
                  "load_hsr6_probe": {"type": type(served).__name__, "classes": list(served.classes),
                                      "absent": list(getattr(served, "absent", ()))},
                  "train_argmax_served_equals_fit": bool(np.array_equal(served_arg, fit_arg)),
                  "train_accuracy": float(np.mean(fit_arg == y))},
        "limits": ["C chosen and macro-F1 reported on the same 5 folds (no nested CV): the reported F1 is optimistic",
                   "40 episodes, 10 tasks x 4 inits; one rollout per init; libero_spatial only",
                   "request interval 0.25 s on Panda (5 env steps at 20 Hz) vs RB-Y1 0.53 s"],
    }
    (out_run / "sl6b_hsr6_panda.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-run", required=True, type=pathlib.Path)
    ap.add_argument("--dev-run", default=DEV_RUN, type=pathlib.Path)
    ap.add_argument("--attn-run", default=ATTN_RUN, type=pathlib.Path)
    ap.add_argument("--phase-dir", default=None, type=pathlib.Path)
    a = ap.parse_args(argv)
    res = run(a.out_run, a.dev_run, a.attn_run, a.phase_dir or a.out_run / "phase")
    o = res["oof_at_chosen_C"]
    print(json.dumps({"C": res["chosen_C"], "fold_mean_macro_f1": o["fold_mean_macro_f1"],
                      "recall": {k: v["recall"] for k, v in o["per_class"].items()},
                      "counts": res["label_counts"]["total"], "vb3": res["vb3"], "probe": res["probe"]["path"],
                      "sha256": res["probe"]["sha256"]}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
