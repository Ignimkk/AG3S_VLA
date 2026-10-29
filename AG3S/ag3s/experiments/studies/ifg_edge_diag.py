"""IFG R2 diagnostic (F3, dev only): which observable cue separates target-internal edges from the
target-boundary edges that merge the target with other surfaces in one component?

Per frame, seed N ROI graph (R0 graph). Edge class from the GT majority class of the two eval voxels:
  internal   target-target
  boundary   target-X, X in {other_object, destination, support, robot_gripper, robot, background}
Cues per edge (no GT):
  dist       centroid distance (m)
  normal     |n_i . n_j| (PCA normals; NaN if unavailable)
  convexity  (n_j - n_i) . (p_j - p_i) / |p_j - p_i| with normals oriented to the node's dominant camera
             (> 0 convex, < 0 concave; LCCP-style criterion, Stein et al. CVPR 2014)
  feat       2 - 2 cos of node features (projector, prefix_L8)
Output: per-edge samples (subsampled) + per-frame counts, and AUC of each cue (boundary vs internal).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_graph as G
from AG3S.ag3s.experiments.studies import ifg_reader as R
from AG3S.ag3s.experiments.studies import ifg_s1 as S1

FEATS = ("projector", "prefix_L8")


def oriented_normals(sub, cam_pos):
    """Orient each PCA normal toward the camera that contributed most of the node's pixels."""
    px = sub.patch_px.tocsc()
    per_cam = np.stack([np.asarray(px[:, 256 * c:256 * (c + 1)].sum(1)).ravel() for c in range(3)], 1)
    dom = per_cam.argmax(1)
    n = sub.normal.copy()
    v = cam_pos[dom] - sub.pos
    flip = np.sum(n * v, 1) < 0
    n[flip] *= -1
    return n


def edge_cues(sub, n_or, feats):
    i, j = sub.edges[:, 0], sub.edges[:, 1]
    dp = sub.pos[j] - sub.pos[i]
    d = np.linalg.norm(dp, axis=1)
    out = dict(dist=d, normal=np.abs(np.sum(sub.normal[i] * sub.normal[j], 1)),
               convexity=np.sum((n_or[j] - n_or[i]) * dp, 1) / np.maximum(d, 1e-9))
    for fn, f in feats.items():
        out[f"feat_{fn}"] = np.maximum(2.0 - 2.0 * np.einsum("ij,ij->i", f[i], f[j]), 0.0)
    return out


def diag_frame(args):
    ref, s0 = args
    cfg = G.GraphConfig()
    fin = R.load_frame(ref, features=FEATS)
    with np.load(s0 / "replay" / f"{ref.frame_id}.npz") as z:
        depth, rmask, K, T = z["depth"], z["robot_mask"], z["K"], z["T_base_cam"]
    with np.load(s0 / "labels" / f"{ref.frame_id}.npz") as z:
        class_map, classes = z["class_map"], [str(c) for c in z["classes"]]
    geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
    vp = G.valid_seed_patches(geo, cfg)
    six = S1.top_k(fin.maps["N"], S1.SEED_K, vp)
    if six.size == 0:
        return None
    sw = fin.maps["N"][six] / fin.maps["N"][six].sum()
    sub = G.subset(geo, G.roi_nodes(geo, G.seed_unary(geo, six, sw), cfg.roi_radius))
    cls = geo.eval_class_frac[sub.node_eval].argmax(1)
    ti = classes.index("target")
    a, b = cls[sub.edges[:, 0]], cls[sub.edges[:, 1]]
    internal = (a == ti) & (b == ti)
    boundary = (a == ti) ^ (b == ti)
    other = np.where(a == ti, b, a)
    n_or = oriented_normals(sub, T[:, :3, 3])
    feats = {fn: G.node_features(sub, R.normalize(fin.features[fn])) for fn in FEATS}
    cues = edge_cues(sub, n_or, feats)
    rng = np.random.default_rng(0)
    ii = np.flatnonzero(internal)
    if ii.size > 400:
        ii = rng.choice(ii, 400, replace=False)
    bb = np.flatnonzero(boundary)
    samples = []
    for k, lab in [(ii, "internal"), (bb, "boundary")]:
        for e in k:
            samples.append(dict(kind=lab, other=classes[int(other[e])] if lab == "boundary" else "target",
                                **{c: float(v[e]) for c, v in cues.items()}))
    return dict(frame_id=ref.frame_id, episode_id=ref.episode_id, moment=ref.moment,
                n_internal=int(internal.sum()),
                n_boundary={classes[c]: int((boundary & (other == c)).sum()) for c in range(len(classes)) if c != ti},
                samples=samples)


def auc(pos, neg):
    """P(pos > neg) with ties 0.5 (rank-based)."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not pos.size or not neg.size:
        return None
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind="stable"), kind="stable") + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    refs = [r for r in R.list_frames() if (a.s0 / "replay" / f"{r.frame_id}.npz").exists()]
    res = []
    with mp.get_context("spawn").Pool(4) as pool:
        for d in pool.imap_unordered(diag_frame, [(r, a.s0) for r in refs]):
            if d:
                res.append(d)
    with open(a.out / "edge_samples.jsonl", "w") as f:
        for d in res:
            f.write(json.dumps(d) + "\n")
    S = [s for d in res for s in d["samples"]]
    cues = [k for k in S[0] if k not in ("kind", "other")]
    summ = {}
    for other in ("ALL", "other_object", "destination", "support", "robot_gripper", "background"):
        bnd = [s for s in S if s["kind"] == "boundary" and (other == "ALL" or s["other"] == other)]
        itn = [s for s in S if s["kind"] == "internal"]
        if not bnd:
            continue
        summ[other] = dict(n_boundary=len(bnd), n_internal=len(itn), cues={})
        for c in cues:
            bv = np.array([s[c] for s in bnd]); iv = np.array([s[c] for s in itn])
            summ[other]["cues"][c] = dict(boundary_q=[float(np.nanpercentile(bv, q)) for q in (10, 50, 90)],
                                          internal_q=[float(np.nanpercentile(iv, q)) for q in (10, 50, 90)],
                                          auc_boundary_gt_internal=auc(bv, iv))
    summ["frames_with_target_neighbor_edge"] = int(sum(1 for d in res if d["n_boundary"].get("other_object", 0) + d["n_boundary"].get("destination", 0) > 0))
    summ["n_frames"] = len(res)
    (a.out / "edge_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1)[:6000])


if __name__ == "__main__":
    main()
