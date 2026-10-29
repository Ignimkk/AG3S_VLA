"""T44b D-merge (verifier-owned, CPU only): stage decomposition of the cluster merge + offline candidates.

Reads dump_frames.py output (`raw_gt/` or `raw_vla/`). benchmark/ is imported read-only (split_by_attention,
dbscan, connected_components_3d, principal_extents).

Definitions fixed before reading the results (2026-10-10):
  * parent cluster = the grounding cluster holding the most GT-target, non-support cloud points.
  * merged (V-E2 definition) = parent target fraction < 0.8 and >= 1 other-fruit point in it. The V-E2 34-frame set
    is read from outputs/verify/T44/E2/raw_demo/*.json (identities.gt.merge) and used as the frame grouping.
  * cloud gap = min distance between the parent's target points and its other-fruit points (both non-support);
    mesh gap = min surface distance target <-> that fruit (GT meshes, 2 mm sampling).
  * candidate output on one parent: a set of components + the target component. Target component chosen by the
    largest attention mass (the existing rule) — `sel=att` (normalised attention, what the pipeline uses) and,
    for segmentation quality alone, `sel=oracle` (the component with most GT-target points).
    precision = target points / segment points; recall = target points in segment / GT target points in parent.
  * pure = precision >= 0.95; complete = recall >= 0.90.
usage: merge_analyze.py --raw DIR --out F.json
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse, collections, json, pathlib, sys
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.stages.admissibility import principal_extents  # noqa: E402
from AG3S.ag3s.stages.attention_split import split_by_attention, triggered  # noqa: E402
from AG3S.ag3s.stages.target_grounding import connected_components_3d, dbscan  # noqa: E402

FRUITS = ("apple", "banana", "orange", "pear")
E2RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2/raw_demo'))
MINP = 20


# ------------------------------------------------------------------ helpers
def attach_stragglers(pts, labels, minp=MINP):
    ids, cnt = np.unique(labels, return_counts=True)
    big = ids[cnt >= minp]
    if big.size == 0:
        return np.zeros(len(pts), np.int64), 1
    small = ~np.isin(labels, big)
    lab = labels.copy()
    if small.any():
        tr = cKDTree(pts[~small])
        _, j = tr.query(pts[small], k=1)
        lab[np.nonzero(small)[0]] = labels[np.nonzero(~small)[0][j]]
    _, lab = np.unique(lab, return_inverse=True)
    return lab, int(lab.max() + 1)


def components_from_edges(n, edges):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if len(edges) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def normals(pts, view, k=10):
    tr = cKDTree(pts)
    _, nn = tr.query(pts, k=min(k, len(pts)))
    nrm = np.zeros_like(pts)
    for i in range(len(pts)):
        q = pts[nn[i]] - pts[nn[i]].mean(0)
        w, v = np.linalg.eigh(q.T @ q)
        nrm[i] = v[:, 0]
    flip = np.einsum("ij,ij->i", nrm, view - pts) < 0
    nrm[flip] *= -1
    return nrm


def rgb2lab(c):
    c = np.asarray(c, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.column_stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])])


# ------------------------------------------------------------------ candidates: parent -> component labels
def cand_recluster(pts, radius):
    lab, _ = attach_stragglers(pts, connected_components_3d(pts, radius))
    return lab


def cand_dbscan(pts, eps):
    lab = dbscan(pts, eps, MINP)
    if (lab < 0).all():
        return np.zeros(len(pts), np.int64)
    lab2 = lab.copy()
    lab2[lab < 0] = lab.max() + 1 + np.arange((lab < 0).sum())   # noise as singletons -> stragglers
    return attach_stragglers(pts, lab2)[0]


def cand_convexity(pts, view, radius=0.016, tau_deg=10.0, k=10):
    """LCCP-style: radius graph, an edge is concave when the normals point towards each other
    ((n_i - n_j) . (x_i - x_j)/|x_i - x_j| < -sin(tau)); concave edges are cut."""
    nrm = normals(pts, view, k)
    pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros(len(pts), np.int64)
    d = pts[pairs[:, 0]] - pts[pairs[:, 1]]
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    s = np.einsum("ij,ij->i", nrm[pairs[:, 0]] - nrm[pairs[:, 1]], d)
    keep = s >= -np.sin(np.deg2rad(tau_deg))
    return attach_stragglers(pts, components_from_edges(len(pts), pairs[keep]))[0]


def cand_colour(pts, col, has_col, radius=0.016, de=20.0):
    lab = rgb2lab(col)
    pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros(len(pts), np.int64)
    dE = np.linalg.norm(lab[pairs[:, 0]] - lab[pairs[:, 1]], axis=1)
    unknown = ~has_col[pairs[:, 0]] | ~has_col[pairs[:, 1]]
    keep = (dE <= de) | unknown
    return attach_stragglers(pts, components_from_edges(len(pts), pairs[keep]))[0]


def fit_sphere(p):
    A = np.column_stack([2 * p, np.ones(len(p))])
    b = (p ** 2).sum(1)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = x[:3]
    r = np.sqrt(max(x[3] + c @ c, 1e-12))
    return c, r


def cand_sphere(pts, rmin=0.025, rmax=0.050, tol=0.006, iters=300, seed=0):
    """Sequential RANSAC spheres with a size prior (fruit radius 25-50 mm). Points of no sphere -> nearest."""
    rng = np.random.default_rng(seed)
    lab = np.full(len(pts), -1, np.int64)
    rest = np.arange(len(pts))
    k = 0
    while len(rest) >= MINP and k < 4:
        best, bc = None, None
        P = pts[rest]
        for _ in range(iters):
            s = rng.choice(len(P), 4, replace=False)
            try:
                c, r = fit_sphere(P[s])
            except np.linalg.LinAlgError:
                continue
            if not (rmin <= r <= rmax):
                continue
            inl = np.abs(np.linalg.norm(P - c, axis=1) - r) < tol
            if best is None or inl.sum() > best.sum():
                best, bc = inl, (c, r)
        if best is None or best.sum() < MINP:
            break
        c, r = fit_sphere(P[best])
        if rmin <= r <= rmax:
            inl = np.abs(np.linalg.norm(P - c, axis=1) - r) < tol
            if inl.sum() >= best.sum():
                best = inl
        lab[rest[best]] = k
        rest = rest[~best]
        k += 1
    if k == 0:
        return np.zeros(len(pts), np.int64)
    if len(rest):
        lab[rest] = k   # the leftover is one more candidate object (e.g. a banana)
    return attach_stragglers(pts, lab)[0]


def cand_mincut(pts, att, radius=0.016, sigma=0.008, lam=1.0, q=0.2):
    """Attention-weighted s-t min-cut: unary from attention rank inside the parent, pairwise exp(-d^2/2s^2)."""
    import networkx as nx
    a = np.asarray(att, np.float64)
    if a.max() - a.min() < 1e-12:
        return np.zeros(len(pts), np.int64)
    a = (a - a.min()) / (a.max() - a.min())
    eps = 1e-3
    G = nx.DiGraph()
    src, snk = "s", "t"
    for i in range(len(pts)):
        G.add_edge(src, i, capacity=float(-np.log(1 - a[i] + eps)))   # cost of labelling i background
        G.add_edge(i, snk, capacity=float(-np.log(a[i] + eps)))       # cost of labelling i target
    pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
    for i, j in pairs:
        w = lam * float(np.exp(-np.sum((pts[i] - pts[j]) ** 2) / (2 * sigma ** 2)))
        G.add_edge(int(i), int(j), capacity=w); G.add_edge(int(j), int(i), capacity=w)
    _, (S, _) = nx.minimum_cut(G, src, snk)
    lab = np.ones(len(pts), np.int64)
    idx = [v for v in S if v != src]
    lab[np.asarray(idx, np.int64)] = 0
    if (lab == 0).sum() < MINP or (lab == 1).sum() < MINP:
        return np.zeros(len(pts), np.int64)
    return lab


# ------------------------------------------------------------------ evaluation
def evaluate(lab, is_t, att, raw, ratio_min=None, ratio_raw=False):
    """lab: component per parent point. Target component chosen by
      sel=att       largest normalised-attention mass (the existing split rule), with the existing ratio gate when
                    ratio_min is set (mean(top)/max mean(others) >= ratio_min on normalised attention, or raw when
                    ratio_raw) -> otherwise the parent stays whole;
      sel=mean_raw  largest mean raw attention (same gate, on raw);
      sel=oracle    the component with most GT-target points (segmentation quality alone)."""
    comps = np.unique(lab)
    n_t = int(is_t.sum())
    out = {"n_components": int(len(comps))}

    def seg_metrics(m):
        tp = int((m & is_t).sum())
        return {"precision": round(tp / max(1, int(m.sum())), 4), "recall": round(tp / max(1, n_t), 4),
                "n": int(m.sum())}

    def pick(score, gate_arr):
        sc = sorted(((score(c), c) for c in comps), key=lambda x: -x[0])
        top = sc[0][1]
        taken, ratio = True, None
        if len(comps) > 1 and ratio_min is not None:
            mt = float(gate_arr[lab == top].mean())
            mo = max(float(gate_arr[lab == c].mean()) for c in comps if c != top)
            ratio = np.inf if mo <= 1e-12 else mt / mo
            taken = ratio >= ratio_min
        seg = (lab == top) if (taken and len(comps) > 1) else np.ones(len(lab), bool)
        return {**seg_metrics(seg), "split_taken": bool(taken and len(comps) > 1),
                "ratio": None if ratio is None or not np.isfinite(ratio) else round(ratio, 3)}

    out["att"] = pick(lambda c: float(att[lab == c].sum()), raw if ratio_raw else att)
    out["mean_raw"] = pick(lambda c: float(raw[lab == c].mean()), raw)
    best = max(comps, key=lambda c: int(((lab == c) & is_t).sum()))
    out["oracle"] = seg_metrics(lab == best)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for fj in sorted(pathlib.Path(args.raw).glob("demo_*.json")):
        rec = json.loads(fj.read_text())
        z = np.load(fj.with_suffix(".npz"))
        key, target = rec["key"], rec["target"]
        names = {int(k): v for k, v in json.loads(str(z["body_names"])).items()}
        P = z["points"].astype(np.float64)
        bid = z["body_id"]
        nm = np.array([names.get(int(b), "none") for b in bid])
        sup = z["support"]
        att, raw, peak = z["attention"], z["raw_attention"], z["peak_signal"]
        cl = z["cluster_label"]
        is_t_all = (nm == target) & ~sup
        is_fruit_other = np.isin(nm, [f for f in FRUITS if f != target]) & ~sup
        # parent
        cands = [(int((is_t_all & (cl == c)).sum()), c) for c in range(cl.max() + 1)] if cl.max() >= 0 else []
        cands.sort(key=lambda x: -x[0])
        e2 = json.loads((E2RAW / f"{key}.json").read_text())["identities"]["gt"]["merge"]["target_cluster"]
        row = {"key": key, "episode": rec["episode"], "frame": rec["frame"], "target": target, "arm": rec["arm"],
               "bin": rec["bin"], "attention_source": rec["attention_source"],
               "attention_cameras": rec["attention_cameras"], "n_clusters": int(cl.max() + 1),
               "e2_target_fraction": e2["target_fraction"], "e2_distractor_fraction": e2["distractor_fraction"],
               "e2_merged": bool(e2["target_fraction"] < 0.8 and e2["distractor_fraction"] > 0),
               "mesh_gap_mm": rec["mesh_gap_mm"], "n_gt_target_cloud": int(is_t_all.sum()),
               "max_opening_mm": None if rec["config"]["max_opening_m"] is None else round(rec["config"]["max_opening_m"] * 1000, 2)}
        # seeds on what
        seeds = z["seeds"]
        sn = nm[seeds]
        row["seeds"] = {"n": int(len(seeds)), "target": int((sn == target).sum()),
                        "other_fruit": int(np.isin(sn, [f for f in FRUITS if f != target]).sum()),
                        "other": int(len(seeds) - (sn == target).sum() - np.isin(sn, [f for f in FRUITS if f != target]).sum())}
        if not cands or cands[0][0] == 0:
            row["parent"] = None
            rows.append(row); continue
        pc = cands[0][1]
        pm = cl == pc
        # the cluster ground_target named (rank 1 / confirm decision), and whether it is the target
        ids = [c["id"] for c in rec["clusters"]]
        if rec["chosen_cluster_id"] is not None and rec["chosen_cluster_id"] in ids:
            cm = cl == ids.index(rec["chosen_cluster_id"])
            row["chosen"] = {"n": int(cm.sum()), "target_fraction": round(float(is_t_all[cm].mean()), 4),
                             "recall": round(float(is_t_all[cm].sum() / max(1, is_t_all.sum())), 4),
                             "is_target_cluster": bool(ids.index(rec["chosen_cluster_id"]) == pc),
                             "top_label": str(collections.Counter(nm[cm]).most_common(1)[0][0])}
        else:
            row["chosen"] = None
        # pipeline outcome = the final cluster holding most GT-target points
        row["pipeline_cluster"] = {"n": int(pm.sum()), "target_fraction": round(float(is_t_all[pm].mean()), 4),
                                   "recall": round(float(is_t_all[pm].sum() / max(1, is_t_all.sum())), 4),
                                   "split": rec["clusters"][pc]["split"]}
        sp = rec["clusters"][pc]["split"]
        if sp and sp.get("role") in ("target", "obstacle"):   # pre-split parent = union of its components
            sib = [ci for ci, c in enumerate(rec["clusters"]) if c["split"] and c["split"].get("split_index") == sp["split_index"]
                   and c["split"].get("role") in ("target", "obstacle")]
            pm = np.isin(cl, sib)
        pidx = np.nonzero(pm)[0]
        pts = P[pidx]
        is_t = is_t_all[pidx]
        others = nm[pidx][is_fruit_other[pidx]]
        uo, co = np.unique(others, return_counts=True)
        row["parent"] = {"n": int(pm.sum()), "n_target": int(is_t.sum()),
                         "target_fraction": round(float(is_t.mean()), 4),
                         "share_of_gt_target": round(float(is_t.sum() / max(1, is_t_all.sum())), 4),
                         "other_fruits": {str(a): int(b) for a, b in zip(uo, co)},
                         "other_labels": {str(a): int(b) for a, b in zip(*np.unique(nm[pidx][~is_t & ~is_fruit_other[pidx]], return_counts=True))},
                         "extents_mm": [round(float(v) * 1000, 1) for v in principal_extents(pts)],
                         "is_chosen": rec["chosen_cluster_id"] is not None and
                                      rec["clusters"][[c["id"] for c in rec["clusters"]].index(rec["chosen_cluster_id"])]["n"] == int(pm.sum()) if rec["clusters"] else None}
        row["merged_mine"] = bool(row["parent"]["target_fraction"] < 0.8 and len(uo) > 0)
        # gaps
        if len(uo):
            main_o = str(uo[np.argmax(co)])
            om = nm[pidx] == main_o
            d, _ = cKDTree(pts[om]).query(pts[is_t], k=1)
            row["merge_with"] = main_o
            row["cloud_gap_mm"] = round(float(d.min()) * 1000, 2)
            row["mesh_gap_merged_mm"] = rec["mesh_gap_mm"].get(main_o)
            # attention on the two objects
            row["attention_target_vs_other"] = {
                "norm_mean_t": round(float(att[pidx][is_t].mean()), 4), "norm_mean_o": round(float(att[pidx][om].mean()), 4),
                "raw_mean_t": float(raw[pidx][is_t].mean()), "raw_mean_o": float(raw[pidx][om].mean()),
                "raw_ratio": round(float(raw[pidx][is_t].mean() / max(raw[pidx][om].mean(), 1e-12)), 3),
                "norm_ratio": round(float(att[pidx][is_t].mean() / max(att[pidx][om].mean(), 1e-12)), 3),
                "frac_norm_saturated_t": round(float((att[pidx][is_t] >= 0.999).mean()), 4),
                "frac_norm_saturated_o": round(float((att[pidx][om] >= 0.999).mean()), 4),
                "mass_share_t": round(float(att[pidx][is_t].sum() / max(att[pidx].sum(), 1e-12)), 4)}
        else:
            row["merge_with"] = None
        # nearest other fruit gap overall (non-merged frames)
        row["nearest_mesh_gap_mm"] = min(rec["mesh_gap_mm"].values())
        # existing split decision (recorded) for the parent
        sr = rec.get("split_record") or {}
        dec = [d for d in sr.get("decisions", []) if d["before"]["n_points"] == int(pm.sum())]
        parent_split = rec["clusters"][pc].get("split")
        row["split_recorded"] = {"ran": sr.get("ran"), "skipped": sr.get("skipped"),
                                 "triggered": bool(dec) or (parent_split is not None),
                                 "reason": (dec[0]["reason"] if dec else (parent_split or {}).get("reason") if parent_split else "not_triggered"),
                                 "ratio": dec[0]["ratio"] if dec else None,
                                 "n_components": len(dec[0]["components"]) if dec else None,
                                 "parent_role": None if parent_split is None else parent_split.get("role")}
        # if the split was taken the parent is a component; record the pre-split parent too
        ext = principal_extents(pts)
        mo = rec["config"]["max_opening_m"]
        row["trigger_check"] = {"narrowest_mm": round(float(np.sort(ext)[0]) * 1000, 1),
                                "largest_mm": round(float(np.sort(ext)[-1]) * 1000, 1),
                                "triggered_now": bool(triggered(ext, mo))}
        # -------- candidates on the parent
        view = np.asarray(rec.get("view_origin") or [0, 0, 0], np.float64)
        try:
            T = z["T_zed_left"]; view = T[:3, 3]
        except KeyError:
            pass
        col, has_col = z["colour"][pidx], z["colour_src"][pidx] >= 0
        a_n, a_r = att[pidx].astype(np.float64), raw[pidx].astype(np.float64)
        C = {}
        C["none (parent whole)"] = (np.zeros(len(pts), np.int64), None, None)
        C["current: recluster15 + ratio2 (norm att)"] = (cand_recluster(pts, 0.015), 2.0, None)
        C["A: recluster15 + ratio2 (raw att)"] = (cand_recluster(pts, 0.015), 2.0, True)
        C["A: recluster15 + ratio1.2"] = (cand_recluster(pts, 0.015), 1.2, None)
        C["A: recluster12 + ratio2"] = (cand_recluster(pts, 0.012), 2.0, None)
        C["A: recluster12 + ratio2 (raw att)"] = (cand_recluster(pts, 0.012), 2.0, True)
        C["A: recluster15, no ratio gate"] = (cand_recluster(pts, 0.015), None, None)
        C["A: recluster12, no ratio gate"] = (cand_recluster(pts, 0.012), None, None)
        C["A: recluster10 + ratio2"] = (cand_recluster(pts, 0.010), 2.0, None)
        C["A: dbscan eps20"] = (cand_dbscan(pts, 0.020), None, None)
        C["A: dbscan eps15"] = (cand_dbscan(pts, 0.015), None, None)
        C["B: convexity (LCCP-like)"] = (cand_convexity(pts, view), None, None)
        C["B: colour region (dE20)"] = (cand_colour(pts, col, has_col), None, None)
        C["B: sphere RANSAC size prior"] = (cand_sphere(pts), None, None)
        C["B: attention min-cut (raw)"] = (cand_mincut(pts, a_r), None, None)
        # the same new segmentations behind the existing ratio gate (2.0, on raw attention)
        C["B: convexity + raw ratio2 gate"] = (C["B: convexity (LCCP-like)"][0], 2.0, True)
        C["B: colour + raw ratio2 gate"] = (C["B: colour region (dE20)"][0], 2.0, True)
        C["B: sphere + raw ratio2 gate"] = (C["B: sphere RANSAC size prior"][0], 2.0, True)
        C["A+B: recluster12 | colour (union of cuts) + raw ratio2 gate"] = (None, 2.0, True)
        # union of cuts: a pair of points stays together only if both recluster12 and colour keep them together
        l1, l2 = C["A: recluster12 + ratio2"][0], C["B: colour region (dE20)"][0]
        _, comb = np.unique(l1 * 1000 + l2, return_inverse=True)
        C["A+B: recluster12 | colour (union of cuts) + raw ratio2 gate"] = (attach_stragglers(pts, comb)[0], 2.0, True)
        res = {}
        for name, (lab, rmin, ar) in C.items():
            res[name] = evaluate(lab, is_t, a_n, a_r, ratio_min=rmin, ratio_raw=bool(ar))
        row["candidates"] = res
        # reproduce the pipeline decision with the benchmark function on the parent (sanity)
        if mo is not None:
            dd = split_by_attention(P, att, pidx, max_opening=mo, radius=0.015, min_points=MINP, min_ratio=2.0)
            row["split_reproduced"] = None if dd is None else {"reason": dd.reason,
                                                               "ratio": None if dd.ratio is None else (None if not np.isfinite(dd.ratio) else round(dd.ratio, 3))}
        rows.append(row)
        print(key, row.get("merged_mine"), row.get("merge_with"), row.get("cloud_gap_mm"), row["split_recorded"]["reason"], flush=True)
    pathlib.Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
