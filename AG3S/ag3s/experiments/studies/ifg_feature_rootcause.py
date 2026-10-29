"""IFG root cause (dev only, GT for diagnosis): why does the VLA feature add nothing on the R7 base?

R7 base: render robot mask (R6), non-support seed eligibility (R5), seed N, ROI graph, G-Q solution.
Per frame and feature (prefix_L8, projector):
  H1 resolution  node patch purity = pixel-weighted GT share of the node's own class in its patches;
                 share of target nodes in mixed patches (target share < 0.5);
                 target-boundary edges whose two nodes share their dominant patch
  H2 redundancy  G-Q predicted regions: argmax seed mass vs argmax region prototype score q;
                 agreement, accuracy of each (region majority GT target), rescues when seed is wrong
  H3 recall loss target nodes with prototype unary < 0.5: share lying in mixed patches
  H4 seed dependence  node-level AUC(proto, target) when the seeds are on target vs off target
"""
import argparse, json, multiprocessing as mp, pathlib

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from AG3S.ag3s.experiments.studies import ifg_graph as G
from AG3S.ag3s.experiments.studies import ifg_reader as R
from AG3S.ag3s.experiments.studies import ifg_round_ns as NS
from AG3S.ag3s.experiments.studies import ifg_run as RN
from AG3S.ag3s.experiments.studies import ifg_s1 as S1
from AG3S.ag3s.experiments.studies.ifg_edge_diag import auc

FEATS = ("prefix_L8", "projector")


def frame(args):
    ref, run = args
    cfg = G.GraphConfig()
    fin = R.load_frame(ref, features=FEATS)
    with np.load(run / "s0/replay" / f"{ref.frame_id}.npz") as z:
        depth, K, T = z["depth"], z["K"], z["T_base_cam"]
    with np.load(run / "rounds/R6/robot_mask" / f"{ref.frame_id}.npz") as z:
        rmask = z["robot_mask"]
    with np.load(run / "s0/labels" / f"{ref.frame_id}.npz") as z:
        class_map, classes, cov = z["class_map"], [str(c) for c in z["classes"]], z["class_cov"]
    geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
    flat = S1.cov_flat(cov)
    ti = classes.index("target")
    vp = NS.eligible(geo, cfg, True)
    six = S1.top_k(fin.maps["N"], S1.SEED_K, vp)
    if six.size == 0:
        return None
    sw = fin.maps["N"][six] / fin.maps["N"][six].sum()
    sub = G.subset(geo, G.roi_nodes(geo, G.seed_unary(geo, six, sw), cfg.roi_radius))
    n = len(sub.pos)
    sl = G.seed_unary(sub, six, sw)
    sup = cfg.kappa_support * sub.support.astype(float)
    lam, s = RN._combine([(sl, np.ones(n)), (sup, np.zeros(n))])
    mG, _ = G.solve(sub, np.ones(n, bool), G.edge_weights(sub, cfg, None), lam, s, cfg.lam)
    cls = geo.eval_class_frac[sub.node_eval].argmax(1)
    tnode = cls == ti
    px = sub.patch_px.toarray()  # (n,768)
    pxs = px.sum(1)
    node_patch_tshare = (px @ flat[:, ti]) / np.maximum(pxs, 1)  # target share of the node's patches
    out = dict(frame_id=ref.frame_id, moment=ref.moment, n_target_nodes=int(tnode.sum()),
               seed_target_frac=float(flat[six, ti].mean()))
    # H1
    if tnode.any():
        out["H1_target_nodes_in_mixed_patches"] = float((node_patch_tshare[tnode] < 0.5).mean())
        out["H1_target_node_patch_target_share_mean"] = float(node_patch_tshare[tnode].mean())
    dom = px.argmax(1)
    a, b = sub.edges[:, 0], sub.edges[:, 1]
    bnd = tnode[a] ^ tnode[b]
    out["H1_boundary_edges"] = int(bnd.sum())
    out["H1_boundary_edges_same_dominant_patch"] = float((dom[a][bnd] == dom[b][bnd]).mean()) if bnd.any() else None
    # regions of the G prediction
    pred = mG >= 0.5
    e = sub.edges[pred[a] & pred[b]]
    _, reg = connected_components(sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)), directed=False)
    regs = np.unique(reg[pred])
    for fn in FEATS:
        uf = R.normalize(fin.features[fn])
        mu = R.normalize((sw @ uf[six])[None])[0]
        pu, _ = G.proto_unary(uf @ mu, six, vp)
        node_p = G.node_patch_mean(sub, pu)
        r = {}
        # H3
        low = tnode & (node_p < 0.5)
        r["H3_target_nodes_proto_below_0.5"] = float(low.sum() / max(tnode.sum(), 1))
        r["H3_low_proto_target_nodes_in_mixed_patches"] = float((node_patch_tshare[low] < 0.5).mean()) if low.any() else None
        r["H3_high_proto_target_nodes_in_mixed_patches"] = (float((node_patch_tshare[tnode & ~low] < 0.5).mean())
                                                            if (tnode & ~low).any() else None)
        # H4
        r["H4_node_auc_proto_target"] = auc(node_p[tnode], node_p[~tnode]) if tnode.any() and (~tnode).any() else None
        # H2
        if len(regs) >= 2:
            sm = np.array([sl[(reg == g) & pred].sum() for g in regs])
            q = np.array([(pxs[(reg == g) & pred] * node_p[(reg == g) & pred]).sum() / max(pxs[(reg == g) & pred].sum(), 1) for g in regs])
            isT = np.array([tnode[(reg == g) & pred].mean() > 0.5 for g in regs])
            r["H2"] = dict(n_regions=len(regs), seed_pick_target=bool(isT[np.argmax(sm)]), proto_pick_target=bool(isT[np.argmax(q)]),
                           agree=bool(np.argmax(sm) == np.argmax(q)), any_target=bool(isT.any()))
        out[fn] = r
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    refs = R.list_frames()
    with mp.get_context("spawn").Pool(4) as p:
        res = [d for d in p.map(frame, [(r, a.run) for r in refs]) if d]
    with open(a.out / "rootcause.jsonl", "w") as f:
        for d in res:
            f.write(json.dumps(d) + "\n")
    S = {"n_frames": len(res)}
    v = [d["H1_target_nodes_in_mixed_patches"] for d in res if "H1_target_nodes_in_mixed_patches" in d]
    S["H1_target_nodes_in_mixed_patches_mean"] = float(np.mean(v))
    v = [d["H1_boundary_edges_same_dominant_patch"] for d in res if d["H1_boundary_edges_same_dominant_patch"] is not None]
    S["H1_boundary_edges_same_dominant_patch_mean"] = float(np.mean(v)); S["H1_frames_with_boundary_edges"] = len(v)
    on = [d for d in res if d["seed_target_frac"] >= 0.3]; off = [d for d in res if d["seed_target_frac"] < 0.1]
    for fn in FEATS:
        x = [d[fn] for d in res]
        S[fn] = dict(
            H3_target_nodes_proto_below_0p5=float(np.mean([r["H3_target_nodes_proto_below_0.5"] for r in x if r["H3_target_nodes_proto_below_0.5"] is not None])),
            H3_low_proto_target_in_mixed=float(np.nanmean([r["H3_low_proto_target_nodes_in_mixed_patches"] for r in x if r["H3_low_proto_target_nodes_in_mixed_patches"] is not None])),
            H3_high_proto_target_in_mixed=float(np.nanmean([r["H3_high_proto_target_nodes_in_mixed_patches"] for r in x if r["H3_high_proto_target_nodes_in_mixed_patches"] is not None])),
            H4_auc_seeds_on_target=float(np.mean([d[fn]["H4_node_auc_proto_target"] for d in on if d[fn]["H4_node_auc_proto_target"] is not None])),
            H4_auc_seeds_off_target=float(np.mean([d[fn]["H4_node_auc_proto_target"] for d in off if d[fn]["H4_node_auc_proto_target"] is not None])),
            H4_n_on=len(on), H4_n_off=len(off))
        h2 = [r["H2"] for r in x if "H2" in r]
        S[fn]["H2"] = dict(frames_multi_region=len(h2), agree=sum(h["agree"] for h in h2),
                           seed_correct=sum(h["seed_pick_target"] for h in h2), proto_correct=sum(h["proto_pick_target"] for h in h2),
                           seed_wrong_proto_right=sum((not h["seed_pick_target"]) and h["proto_pick_target"] for h in h2),
                           seed_right_proto_wrong=sum(h["seed_pick_target"] and not h["proto_pick_target"] for h in h2),
                           no_target_region=sum(not h["any_target"] for h in h2))
    (a.out / "rootcause_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
