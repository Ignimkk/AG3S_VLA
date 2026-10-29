"""IFG S1 — seed source x feature prototype factorial (protocol v1.1 §5), patch-level diagnostics.

Scores (algorithm side, no GT):
  map controls      A (action L8H2 d0 last), N (noun target L10H7), P = norm(A)*norm(N), A61 (T61
                    action mean L10H7 d0), A_first8 (query diagnostic: L8H2 d0 mean of queries 0..7)
  prototypes        for seed in {A, N, P} x feature in FEATURES: top-k seeds (k=5; ties by ascending
                    global token index), mu = normalize(sum w_i fhat_i), w = seed score / sum;
                    s_j = fhat_j . mu   -> arm id  "proto:<seed>:<feature>"
Evaluation (GT patch coverage, evaluator only): top-K patches (K in 5/10/20/40) taken as wholly
positive; precision / recall / IoU integrate the target pixel fraction of each patch; class fractions
of the selection; seed purity (target fraction of the seed patches, mean and seed-weighted).
"""

from __future__ import annotations

import numpy as np

from AG3S.ag3s.experiments.studies.ifg_reader import FEATURES, normalize

SEEDS = ("A", "N", "P")
MAP_CONTROLS = ("A", "N", "P", "A61", "A_first8")
TOPK_EVAL = (5, 10, 20, 40)
SEED_K = 5


def top_k(score: np.ndarray, k: int, valid: np.ndarray | None = None) -> np.ndarray:
    """Indices of the k largest valid scores, ties by ascending index. Empty if all valid scores are 0."""
    s = np.asarray(score, np.float64)
    idx = np.arange(s.size)
    if valid is not None:
        idx = idx[np.asarray(valid, bool)]
    if idx.size == 0 or not np.any(s[idx] > 0):
        return np.zeros(0, np.int64)
    order = np.lexsort((idx, -s[idx]))
    return idx[order[:k]].astype(np.int64)


def prototype(feat: np.ndarray, seed_score: np.ndarray, k: int = SEED_K, valid=None):
    """``(cosine scores (768,), seed indices, weights)``; None scores when the seed abstains."""
    seeds = top_k(seed_score, k, valid)
    if seeds.size == 0:
        return None, seeds, None
    w = np.asarray(seed_score, np.float64)[seeds]
    w = w / w.sum()
    fh = normalize(feat)
    mu = normalize(w @ fh[seeds])
    return fh @ mu, seeds, w


def score_frame(features: dict, maps: dict, k: int = SEED_K) -> tuple[dict, dict]:
    """``(scores, seeds)``: every S1 arm's (768,) score and its seed indices/weights."""
    scores, seeds = {}, {}
    for m in MAP_CONTROLS:
        if m in maps:
            scores[f"map:{m}"] = np.asarray(maps[m], np.float64)
    for sd in SEEDS:
        for fn in FEATURES:
            if fn not in features:
                continue
            s, ix, w = prototype(features[fn], maps[sd], k)
            arm = f"proto:{sd}:{fn}"
            seeds[arm] = (ix, w)
            if s is not None:
                scores[arm] = s
    return scores, seeds


def cov_flat(class_cov: np.ndarray) -> np.ndarray:
    """(3, C, 16, 16) -> (768, C), global token order cam-major, row-major patches."""
    return np.asarray(class_cov, np.float64).transpose(0, 2, 3, 1).reshape(768, -1)


def patch_metrics(score: np.ndarray, flat: np.ndarray, classes, K: int) -> dict:
    truth = flat[:, classes.index("target")]
    tot = float(truth.sum())
    ix = top_k(score, K)
    if ix.size == 0:
        return dict(valid=False, missing_reason="abstain")
    tp = float(truth[ix].sum())
    k = ix.size
    return dict(valid=tot > 0, missing_reason=None if tot > 0 else "target_not_visible",
                precision=tp / k, recall=tp / tot if tot > 0 else None,
                iou=tp / (k + tot - tp) if tot > 0 else None, target_patch_mass=tot,
                **{f"frac_{c}": float(flat[ix, j].mean()) for j, c in enumerate(classes)})


def seed_purity(seeds_w, flat, classes) -> dict:
    ix, w = seeds_w
    if ix.size == 0:
        return dict(seed_abstain=True)
    t = flat[ix, classes.index("target")]
    out = dict(seed_abstain=False, seed_target_frac_mean=float(t.mean()), seed_target_frac_weighted=float(w @ t),
               seed_cameras=[int(i) // 256 for i in ix])
    for j, c in enumerate(classes):
        out[f"seed_frac_{c}"] = float(w @ flat[ix, j])
    return out


def oracle_patch_iou(flat, classes) -> dict:
    """GT-only 16x16 whole-patch representation ceiling (diagnostic, not an algorithm)."""
    truth = flat[:, classes.index("target")]
    if truth.sum() <= 0:
        return dict(iou=None, patch_count=0)
    o = np.argsort(-truth, kind="stable")
    tp = np.cumsum(truth[o])
    ks = np.arange(1, 769)
    iou = tp / (ks + truth.sum() - tp)
    b = int(np.argmax(iou))
    return dict(iou=float(iou[b]), patch_count=b + 1)


# --------------------------------------------------------------------- frame022 legacy layout


def load_frame022_legacy(dump_dir):
    """The 2026-10-15 full dump (`vla_dump/frame_022`): per-layer npz layout. Regression only."""
    import pathlib

    d = pathlib.Path(dump_dir) / "frame_022"
    with np.load(d / "derived.npz") as z:
        a = z["serving_L8H2_last_d0"].astype(np.float64)
        n = z["noun_target_L10H7"].astype(np.float64)
    with np.load(d / "suffix_s00.npz") as z:
        first8 = z["L_attn_probs"][8, 2, :8, :768].astype(np.float64).mean(0)
        a61 = z["L_attn_probs"][10, 7, :, :768].astype(np.float64).mean(0)
    cams = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    f = {"siglip_encoded": np.concatenate([np.load(d / f"siglip_{c}.npz")["encoded"] for c in cams])}
    with np.load(d / "embed.npz") as z:
        f["projector"] = z["image_tokens"]
    for l in (4, 8, 12):
        with np.load(d / f"prefix_L{l:02d}.npz") as z:
            f[f"prefix_L{l}"] = z["resid_out"][:768]
    mx = lambda m: m / m.max()
    return f, {"A": a, "N": n, "P": mx(a) * mx(n), "A61": a61, "A_first8": first8}
