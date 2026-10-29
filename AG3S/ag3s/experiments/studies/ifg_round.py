"""IFG improvement rounds (guide §15) — component-level feature gate on the geometry-only graph.

R1 (registry `rounds/R1/hypothesis.json`, failure F5): the prototype score is patch-resolution evidence, so
it is aggregated per depth-surface component of the ROI graph and used only to accept / reject a seed-
anchored component of the G solution (identity from features, extent from geometry):

  q_c = sum_{i in c} px_i * proto_i / sum_{i in c} px_i      (proto_i: R0 node prototype unary)
  FGc: m_i = m^G_i if q_{c(i)} >= tau_q else 0               (unanchored components stay unknown, as in G)

Arms per frame (seed N, R0 seed rule and graph): G:N (parent; must equal R0), FGc:N:<feature>,
FGc-shuffle:N:<feature> (proto scores permuted across valid non-seed patches, rng seed 20261015 + frame
hash), O-comp:N (diagnostic: keep anchored components whose GT target mass > half their nodes).

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_round --round R1 \
      --s0 $RUN/s0 --out $RUN/rounds/R1 --features projector prefix_L8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pathlib
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from AG3S.ag3s.experiments.studies import ifg_graph as G
from AG3S.ag3s.experiments.studies import ifg_reader as R
from AG3S.ag3s.experiments.studies import ifg_run as RN
from AG3S.ag3s.experiments.studies import ifg_s1 as S1

TAU_Q = 0.5


def components(sub):
    n = len(sub.pos)
    W = sp.coo_matrix((np.ones(len(sub.edges)), (sub.edges[:, 0], sub.edges[:, 1])), shape=(n, n))
    return connected_components(W, directed=False)


def component_gate(m, comp, seed_lam, px, proto, tau):
    """Reject anchored components whose pixel-weighted prototype score is below tau."""
    ncomp = comp.max() + 1
    mass = np.bincount(comp, weights=seed_lam, minlength=ncomp)
    q = np.bincount(comp, weights=px * proto, minlength=ncomp) / np.maximum(np.bincount(comp, weights=px, minlength=ncomp), 1e-12)
    out = m.copy()
    rej = (mass[comp] > 0) & (q[comp] < tau)
    out[rej] = 0.0
    return out, dict(n_anchored=int((mass > 0).sum()), n_rejected=int(((mass > 0) & (q < tau)).sum()),
                     q_anchored=[float(x) for x in q[mass > 0]])


def region_gate(m, sub, px, proto, tau, seed_lam=None, protect_top=False, top1_only=False):
    """R2: gate each connected region of the G prediction {m >= 0.5} by its own pixel-weighted prototype score.
    R3 (`protect_top`): the region holding the largest seed mass is always kept; `top1_only`: keep only it."""
    pred = np.isfinite(m) & (m >= 0.5)
    n = len(m)
    e = sub.edges[pred[sub.edges[:, 0]] & pred[sub.edges[:, 1]]]
    W = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, reg = connected_components(W, directed=False)
    nr = reg.max() + 1
    w = np.where(pred, px, 0.0)
    q = np.bincount(reg, weights=w * proto, minlength=nr) / np.maximum(np.bincount(reg, weights=w, minlength=nr), 1e-12)
    out = m.copy()
    has = np.bincount(reg, weights=pred.astype(float), minlength=nr) > 0
    rej_r = q < tau
    if protect_top or top1_only:
        sm = np.bincount(reg, weights=np.where(pred, seed_lam, 0.0), minlength=nr)
        sm[~has] = -1.0
        top = int(np.argmax(sm))
        rej_r = np.ones(nr, bool) if top1_only else rej_r
        rej_r[top] = False
    rej = pred & rej_r[reg]
    q = np.where(rej_r, q, np.maximum(q, 0.0))
    out[rej] = 0.0
    return out, reg, pred, dict(n_regions=int(has.sum()), n_rejected=int((has & rej_r).sum()),
                                q_regions=[float(x) for x in q[has]])


def round_frame(args):
    ref, s0, out, feats, mode = args
    dst = out / "rows" / f"{ref.frame_id}.jsonl"
    if dst.exists():
        return ref.frame_id, None
    t0 = time.perf_counter()
    cfg = G.GraphConfig()
    fin = R.load_frame(ref, features=tuple(feats))
    with np.load(s0 / "replay" / f"{ref.frame_id}.npz") as z:
        depth, rmask, K, T = z["depth"], z["robot_mask"], z["K"], z["T_base_cam"]
    with np.load(s0 / "labels" / f"{ref.frame_id}.npz") as z:
        class_map, classes = z["class_map"], [str(c) for c in z["classes"]]
    geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
    vp = G.valid_seed_patches(geo, cfg)
    six = S1.top_k(fin.maps["N"], S1.SEED_K, vp)
    base = dict(frame_id=ref.frame_id, episode_id=ref.episode_id, ds=ref.ds, moment=ref.moment, kind=ref.kind,
                split="dev", level="voxel", seed_source="N", threshold=0.5, voxel_size=cfg.voxel)
    rows = []
    if six.size == 0:
        rows.append({**base, "arm_id": "G:N", "valid": False, "missing_reason": "seed_abstain"})
    else:
        sw = fin.maps["N"][six] / fin.maps["N"][six].sum()
        roi = G.roi_nodes(geo, G.seed_unary(geo, six, sw), cfg.roi_radius)
        sub = G.subset(geo, roi)
        n = len(sub.pos)
        seed_lam = G.seed_unary(sub, six, sw)
        sup = cfg.kappa_support * sub.support.astype(float)
        lam_g, s_g = RN._combine([(seed_lam, np.ones(n)), (sup, np.zeros(n))])
        t1 = time.perf_counter()
        mG, info = G.solve(sub, np.ones(n, bool), G.edge_weights(sub, cfg, None), lam_g, s_g, cfg.lam)
        ms_g = (time.perf_counter() - t1) * 1000
        _, comp = components(sub)
        px = np.asarray(sub.patch_px.sum(1)).ravel()
        info["ms_solve"] = ms_g
        arms = {"G:N": (mG, info)}
        rng = np.random.default_rng(20261015 + int(hashlib.sha1(ref.frame_id.encode()).hexdigest()[:6], 16))
        for fn in feats:
            t2 = time.perf_counter()
            uf = R.normalize(fin.features[fn])
            mu = R.normalize((sw @ uf[six])[None])[0]
            pu, pinfo = G.proto_unary(uf @ mu, six, vp)
            node_p = G.node_patch_mean(sub, pu)
            if mode == "component":
                gate = lambda pr: component_gate(mG, comp, seed_lam, px, pr, TAU_Q)  # noqa: E731
            else:
                gate = lambda pr: region_gate(mG, sub, px, pr, TAU_Q, seed_lam, protect_top=(mode == "region_top1"))[::3]  # noqa: E731
            tag = {"component": "FGc", "region": "FGr", "region_top1": "FGr1"}[mode]
            m1, gi = gate(node_p)
            arms[f"{tag}:N:{fn}"] = (m1, dict(ms_solve=ms_g + (time.perf_counter() - t2) * 1000, **gi, proto=pinfo))
            perm_ix = np.flatnonzero(vp & ~np.isin(np.arange(768), six))
            ps = pu.copy()
            ps[perm_ix] = pu[rng.permutation(perm_ix)]
            if mode == "component" or fn == feats[0]:
                m2, gi2 = gate(G.node_patch_mean(sub, ps))
                arms[f"{tag}-shuffle:N:{fn}"] = (m2, gi2)
        # oracle component identity (diagnostic only)
        t = geo.eval_target[sub.node_eval]
        ncomp = comp.max() + 1
        tmass = np.bincount(comp, weights=t, minlength=ncomp) / np.maximum(np.bincount(comp, minlength=ncomp), 1)
        mO = mG.copy()
        if mode == "component":
            mO[(np.bincount(comp, weights=seed_lam, minlength=ncomp)[comp] > 0) & (tmass[comp] <= 0.5)] = 0.0
            arms["O-comp:N"] = (mO, dict(oracle=True))
        elif mode == "region_top1":
            arms["top1-only:N"] = (region_gate(mG, sub, px, np.zeros(n), TAU_Q, seed_lam, top1_only=True)[0], {})
        else:
            _, reg, pred, _ = region_gate(mG, sub, px, np.zeros(n), TAU_Q)
            nr = reg.max() + 1
            rt = np.bincount(reg, weights=np.where(pred, t, 0), minlength=nr) / np.maximum(np.bincount(reg, weights=pred.astype(float), minlength=nr), 1)
            mO[pred & (rt[reg] <= 0.5)] = 0.0
            arms["O-region:N"] = (mO, dict(oracle=True))
        for arm, (m, ii) in arms.items():
            ev = G.evaluate(sub, m, 0.5, cfg)
            rows.append({**base, "arm_id": arm, "method": arm.split(":")[0],
                         "feature_name": arm.split(":")[2] if arm.count(":") == 2 else None,
                         "valid": ev.get("missing_reason") is None, "missing_reason": ev.get("missing_reason"),
                         **ev, **{k: v for k, v in ii.items() if k != "q_anchored"},
                         **({"q_anchored": ii["q_anchored"]} if "q_anchored" in ii else {})})
    tmp = dst.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    os.replace(tmp, dst)
    return ref.frame_id, f"{time.perf_counter() - t0:.1f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--s0", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--features", nargs="+", default=["projector", "prefix_L8"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mode", choices=["component", "region", "region_top1"], default="component")
    a = ap.parse_args()
    (a.out / "rows").mkdir(parents=True, exist_ok=True)
    (a.out / "resolved_config.json").write_text(json.dumps(dict(round=a.round, graph=G.GraphConfig().__dict__, seed="N",
                                                                tau_q=TAU_Q, features=a.features, gate_unit=a.mode), indent=1, default=str))
    refs = [r for r in R.list_frames() if (a.s0 / "replay" / f"{r.frame_id}.npz").exists()]
    with mp.get_context("spawn").Pool(min(a.workers, 4)) as pool:
        for fid, msg in pool.imap_unordered(round_frame, [(r, a.s0, a.out, a.features, a.mode) for r in refs]):
            pass
    print("done", len(refs))


if __name__ == "__main__":
    main()
