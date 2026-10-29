"""IFG R10 (cross-frame identity memory) and R11 (confidence gate) — one pass per episode, frames in time order.

Base per frame: R6 candidate (render robot mask, non-support seed eligibility, seed N, G-Q + top1 region).
Candidates: connected regions of the full-scene surface graph after removing support and robot nodes
(ifg_round_mass.regions), kept if >= MIN_NODES nodes and bbox max side <= MAX_SIDE.
Region feature: normalise( (sum of the region's node patch-pixel rows) @ unit patch features ).
Memory: the R6 chosen region's feature at the episode's first dumped frame ("first"), or the previous
frame's final choice ("chain"). Only past frames are used.
Writes rounds/R10/rows/<frame>.jsonl (R10 arms) and rounds/R11/conf.jsonl (per-frame scores + GT outcome).
"""
import argparse, json, multiprocessing as mp, pathlib

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_graph as G
from AG3S.ag3s.experiments.studies import ifg_reader as R
from AG3S.ag3s.experiments.studies import ifg_round as RD
from AG3S.ag3s.experiments.studies import ifg_round_mass as RM
from AG3S.ag3s.experiments.studies import ifg_round_ns as NS
from AG3S.ag3s.experiments.studies import ifg_run as RN
from AG3S.ag3s.experiments.studies import ifg_s1 as S1

FEATS = ("prefix_L8", "projector")
DELTAS = (0.0, 0.05, 0.1)
MIN_NODES, MAX_SIDE = 20, 0.4


def unit_rows(w768, U):
    f = np.asarray(w768, float).ravel() @ U
    return f / max(np.linalg.norm(f), 1e-12)


def r6_choice(geo, cfg, smap):
    """R6 candidate as a boolean mask over geo nodes, plus seed info and the ROI sub-graph pieces."""
    vp = NS.eligible(geo, cfg, True)
    six = S1.top_k(smap, S1.SEED_K, vp)
    if six.size == 0:
        return None
    sw = smap[six] / smap[six].sum()
    roi = G.roi_nodes(geo, G.seed_unary(geo, six, sw), cfg.roi_radius)
    sub = G.subset(geo, roi)
    n = len(sub.pos)
    sl = G.seed_unary(sub, six, sw)
    lam, s = RN._combine([(sl, np.ones(n)), (cfg.kappa_support * sub.support.astype(float), np.zeros(n))])
    m, _ = G.solve(sub, np.ones(n, bool), G.edge_weights(sub, cfg, None), lam, s, cfg.lam)
    mt = RD.region_gate(m, sub, np.ones(n), np.zeros(n), 0.5, sl, top1_only=True)[0]
    chosen = np.zeros(len(geo.pos), bool)
    idx = np.flatnonzero(roi)
    chosen[idx[mt >= 0.5]] = True
    pred = np.flatnonzero(m >= 0.5)
    mass_all = float(sl[pred].sum()); mass_top = float(sl[mt >= 0.5].sum())
    return dict(chosen=chosen, six=six, sw=sw, vp=vp, c_mass=(mass_top / mass_all) if mass_all > 0 else None)


def episode(args):
    refs, run = args[0], args[1]
    center = bool(args[2]) if len(args) > 2 else False
    cfg = G.GraphConfig()
    out_rows, conf = {}, []
    mem = {k: {fn: None for fn in FEATS} for k in ("first", "chain")}
    mem_first_iou = None
    for ref in sorted(refs, key=lambda r: r.frame):
        fin = R.load_frame(ref, features=FEATS)
        with np.load(run / "s0/replay" / f"{ref.frame_id}.npz") as z:
            depth, K, T = z["depth"], z["K"], z["T_base_cam"]
        with np.load(run / "rounds/R6/robot_mask" / f"{ref.frame_id}.npz") as z:
            rmask = z["robot_mask"]
        with np.load(run / "s0/labels" / f"{ref.frame_id}.npz") as z:
            class_map, classes, cov = z["class_map"], [str(c) for c in z["classes"]], z["class_cov"]
        geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
        U = {fn: R.normalize(fin.features[fn]) for fn in FEATS}
        if center:  # R10b: remove the frame-mean patch feature (shared component) before normalising
            U = {fn: R.normalize(u - u.mean(0, keepdims=True)) for fn, u in U.items()}
        base = dict(frame_id=ref.frame_id, episode_id=ref.episode_id, ds=ref.ds, moment=ref.moment, kind=ref.kind,
                    split="dev", level="voxel", threshold=0.5, voxel_size=cfg.voxel, seed_source="N")
        rows = []
        c = r6_choice(geo, cfg, fin.maps["N"])
        comp = RM.regions(geo, ~geo.support)
        ok = comp >= 0
        nreg = comp.max() + 1
        size = np.bincount(comp[ok], minlength=nreg)
        lo = np.full((nreg, 3), np.inf); hi = np.full((nreg, 3), -np.inf)
        np.minimum.at(lo, comp[ok], geo.pos[ok]); np.maximum.at(hi, comp[ok], geo.pos[ok])
        cand = np.flatnonzero((size >= MIN_NODES) & ((hi - lo).max(1) <= MAX_SIDE))
        px = geo.patch_px.tocsr()
        reg_w = {int(r): np.asarray(px[comp == r].sum(0)).ravel() for r in cand}

        def ev(mask):
            return G.evaluate(geo, np.where(mask, 1.0, -1.0), 0.5, cfg)

        if c is None:
            r6_mask = np.zeros(len(geo.pos), bool)
        else:
            r6_mask = c["chosen"]
        r6_ev = ev(r6_mask)
        rows.append({**base, "arm_id": "R6", **r6_ev})
        chosen_w = np.asarray(px[r6_mask].sum(0)).ravel() if r6_mask.any() else None
        # region of the R6 choice among full-scene regions (majority)
        r6_reg = int(np.bincount(comp[r6_mask & ok]).argmax()) if (r6_mask & ok).any() else -1
        rec = dict(frame_id=ref.frame_id, episode_id=ref.episode_id, moment=ref.moment, kind=ref.kind,
                   r6_iou=r6_ev.get("iou"), r6_tp=r6_ev["TP"], r6_fp=r6_ev["FP"], target_voxels=r6_ev["target_voxels"],
                   c_mass=None if c is None else c["c_mass"], n_candidates=int(len(cand)), first_frame=mem["first"]["prefix_L8"] is None)
        for fn in FEATS:
            f_ch = unit_rows(chosen_w, U[fn]) if chosen_w is not None else None
            if c is not None and f_ch is not None:
                mu = R.normalize((c["sw"] @ U[fn][c["six"]])[None])[0]
                rec[f"q_seed:{fn}"] = float(f_ch @ mu)
            for kind in ("first", "chain"):
                M = mem[kind][fn]
                cos = {} if (M is None or f_ch is None) else {r: float(unit_rows(w, U[fn]) @ M) for r, w in reg_w.items()}
                if not cos:  # no memory yet (first frame) or nothing to compare: the R6 output stands
                    for d in DELTAS:
                        rows.append({**base, "arm_id": f"Mem-{kind}:{fn}:d{d}", "switched": False, "no_memory": True, **r6_ev})
                    rows.append({**base, "arm_id": f"Mem-only-{kind}:{fn}", "no_memory": True, **r6_ev})
                    continue
                rec[f"q_mem_{kind}:{fn}"] = float(f_ch @ M)
                best = max(cos, key=cos.get)
                cur = float(f_ch @ M)
                for d in DELTAS:
                    sw_ = best != r6_reg and cos[best] - cur >= d
                    mask = (comp == best) if sw_ else r6_mask
                    rows.append({**base, "arm_id": f"Mem-{kind}:{fn}:d{d}", "switched": bool(sw_), **ev(mask)})
                rows.append({**base, "arm_id": f"Mem-only-{kind}:{fn}", **ev(comp == best)})
        # memory update (after this frame's decisions): first = this frame's R6 choice on the first frame;
        # chain = this frame's final Mem-chain choice at delta 0.1 (falls back to R6)
        for fn in FEATS:
            f_ch = unit_rows(chosen_w, U[fn]) if chosen_w is not None else None
            if mem["first"][fn] is None and f_ch is not None:
                mem["first"][fn] = f_ch
                mem_first_iou = r6_ev.get("iou")
            sw_rows = [r for r in rows if r["arm_id"] == f"Mem-chain:{fn}:d0.1" and r.get("switched")]
            if sw_rows:
                best_mask = None  # recompute the switched region's feature
                M = mem["chain"][fn]
                cos = {r: float(unit_rows(w, U[fn]) @ M) for r, w in reg_w.items()}
                best = max(cos, key=cos.get)
                mem["chain"][fn] = unit_rows(reg_w[best], U[fn])
            elif f_ch is not None:
                mem["chain"][fn] = f_ch
        rec["memory_source_iou"] = mem_first_iou
        conf.append(rec)
        out_rows[ref.frame_id] = rows
    return out_rows, conf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--center", action="store_true"); ap.add_argument("--round", default="R10"); ap.add_argument("--conf-round", default="R11")
    a = ap.parse_args()
    (a.run / "rounds" / a.round / "rows").mkdir(parents=True, exist_ok=True)
    (a.run / "rounds" / a.conf_round).mkdir(parents=True, exist_ok=True)
    eps = {}
    for r in R.list_frames():
        eps.setdefault(r.episode_id, []).append(r)
    allconf = []
    with mp.get_context("spawn").Pool(4) as p:
        for rows, conf in p.imap_unordered(episode, [(v, a.run, a.center) for v in eps.values()]):
            for fid, rs in rows.items():
                (a.run / "rounds" / a.round / "rows" / f"{fid}.jsonl").write_text("".join(json.dumps(x, default=float) + "\n" for x in rs))
            allconf += conf
    with open(a.run / "rounds" / a.conf_round / "conf.jsonl", "w") as f:
        for c in allconf:
            f.write(json.dumps(c, default=float) + "\n")
    print("done", len(allconf))


if __name__ == "__main__":
    main()
