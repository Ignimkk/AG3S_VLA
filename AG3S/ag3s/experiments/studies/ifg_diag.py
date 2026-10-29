"""IFG R1 diagnostics (dev only, GT used for diagnosis) — where do G / FG false positives come from?

Per frame, seed N (R0 seed rule), the R0 graph (ifg_graph defaults):
  components of the ROI graph; per component: seed mass, size, support nodes, GT target / neighbor mass;
  FP / TP mass of G:N and FG:N:<feature> split by
    anchored   component carries seed mass (> 0)
    unanchored component without seed mass (only possible in FG: the prototype unary is its only evidence)
  plus "best single component" oracle: TP / FP if only the highest-seed-mass anchored component were kept
  (the analogue of serving's one-cluster choice; GT not used to choose it).

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_diag --s0 $RUN/s0 --out $RUN/rounds/R0/diag
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from AG3S.ag3s.experiments.studies import ifg_graph as G
from AG3S.ag3s.experiments.studies import ifg_reader as R
from AG3S.ag3s.experiments.studies import ifg_run as RN
from AG3S.ag3s.experiments.studies import ifg_s1 as S1

FEATS = ("projector", "siglip_encoded", "prefix_L8")


def diag_frame(args):
    ref, s0 = args
    cfg = G.GraphConfig()
    fin = R.load_frame(ref, features=FEATS)
    with np.load(s0 / "replay" / f"{ref.frame_id}.npz") as z:
        depth, rmask, K, T = z["depth"], z["robot_mask"], z["K"], z["T_base_cam"]
    with np.load(s0 / "labels" / f"{ref.frame_id}.npz") as z:
        class_map, classes, class_cov = z["class_map"], [str(c) for c in z["classes"]], z["class_cov"]
    geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
    vp = G.valid_seed_patches(geo, cfg)
    six = S1.top_k(fin.maps["N"], S1.SEED_K, vp)
    sw = fin.maps["N"][six] / fin.maps["N"][six].sum()
    roi = G.roi_nodes(geo, G.seed_unary(geo, six, sw), cfg.roi_radius)
    sub = G.subset(geo, roi)
    n = len(sub.pos)
    seed_lam = G.seed_unary(sub, six, sw)
    sup = sub.support.astype(float)
    W = sp.coo_matrix((np.ones(len(sub.edges)), (sub.edges[:, 0], sub.edges[:, 1])), shape=(n, n))
    ncomp, comp = connected_components(W, directed=False)
    t = geo.eval_target[sub.node_eval]
    ci = {c: i for i, c in enumerate(classes)}
    nb = geo.eval_class_frac[sub.node_eval][:, [ci["other_object"], ci["destination"]]].sum(1)
    cmass = np.bincount(comp, weights=seed_lam, minlength=ncomp)
    anchored = cmass[comp] > 0
    comps = []
    for c in np.flatnonzero(cmass > 0):
        k = comp == c
        comps.append(dict(seed_mass=float(cmass[c]), n=int(k.sum()), support=int(sup[k].sum()),
                          target=float(t[k].sum()), neighbor=float(nb[k].sum())))
    comps.sort(key=lambda d: -d["seed_mass"])
    lam_g, s_g = RN._combine([(seed_lam, np.ones(n)), (sup, np.zeros(n))])
    all_in = np.ones(n, bool)
    mG, _ = G.solve(sub, all_in, G.edge_weights(sub, cfg, None), lam_g, s_g, cfg.lam)
    out = dict(frame_id=ref.frame_id, episode_id=ref.episode_id, moment=ref.moment, n_roi=n,
               n_components=int(ncomp), n_anchored_components=int((cmass > 0).sum()),
               target_total=float(geo.eval_target.sum()), target_in_roi=float(t.sum()),
               target_in_anchored=float(t[anchored].sum()), components=comps[:6],
               seed_target_frac=float(S1.cov_flat(class_cov)[six, classes.index("target")].mean()))

    def split(m):
        p = np.isfinite(m) & (m >= 0.5)
        return dict(tp_anch=float(t[p & anchored].sum()), fp_anch=float((1 - t)[p & anchored].sum()),
                    tp_unanch=float(t[p & ~anchored].sum()), fp_unanch=float((1 - t)[p & ~anchored].sum()),
                    nbr_fp_anch=float(nb[p & anchored].sum()), nbr_fp_unanch=float(nb[p & ~anchored].sum()))

    out["G"] = split(mG)
    # one-cluster analogue: keep only the anchored component with the largest seed mass
    if (cmass > 0).any():
        best = int(np.argmax(cmass))
        keep = np.where(comp == best, mG, -1.0)
        out["G_top1"] = split(keep)
    for fn in FEATS:
        uf = R.normalize(fin.features[fn])
        mu = R.normalize((sw @ uf[six])[None])[0]
        pu, pinfo = G.proto_unary(uf @ mu, six, vp)
        node_p = G.node_patch_mean(sub, pu)
        lam_u, s_u = RN._combine([(seed_lam, np.ones(n)), (sup, np.zeros(n)), (np.full(n, cfg.kappa_proto), node_p)])
        mF, _ = G.solve(sub, all_in, G.edge_weights(sub, cfg, G.node_features(sub, uf)), lam_u, s_u, cfg.lam)
        out[f"FG:{fn}"] = split(mF)
        # proto score separation inside the ROI: target vs neighbor nodes (diagnostic for F4)
        tt, nn = node_p[t >= 0.5], node_p[nb >= 0.5]
        out[f"proto:{fn}"] = dict(target_mean=float(tt.mean()) if tt.size else None,
                                 neighbor_mean=float(nn.mean()) if nn.size else None,
                                 neighbor_ge_0p5=float((nn >= 0.5).mean()) if nn.size else None,
                                 target_ge_0p5=float((tt >= 0.5).mean()) if tt.size else None, **pinfo)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    refs = [r for r in R.list_frames() if (a.s0 / "replay" / f"{r.frame_id}.npz").exists()]
    with mp.get_context("spawn").Pool(min(a.workers, 4)) as pool, open(a.out / "diagnostics.jsonl", "w") as f:
        for d in pool.imap_unordered(diag_frame, [(r, a.s0) for r in refs]):
            f.write(json.dumps(d) + "\n")
    print("done", len(refs))


if __name__ == "__main__":
    main()
