"""IFG R7 — attention + feature + 3D re-validated on the improved base (R5 non-support seeds, R6 render mask).

Same graph / unary / solver defaults as R0 (ifg_graph.GraphConfig). Arms per frame (seed N):
  Q (continuous Dirichlet energy, protocol §7), 2x2 feature mechanism control:
    G-Q            geometry edges, unary seed + support                         (parent = R6 G-ns+rm)
    FGe-Q:<f>      + feature affinity in the edges
    FGu-Q:<f>      + node prototype unary (kappa_proto)
    FG-Q:<f>       both
    <arm>+top1     keep only the predicted region holding the largest seed mass (R3 rule; G-Q+top1 = R6 candidate)
  R (seeded region growing, guide §9 step 2 / §10 A2-A3):
    G-R            BFS from seed nodes over depth-continuous edges between non-support nodes
    FG-R:<f>       G-R, a node joins only if its prototype score >= 0.5 (seed nodes always kept)
    <arm>+top1     keep the grown region holding the largest seed mass
  O-seed (guide A5, diagnostic): GT top-5 target patches as seeds, G-Q+top1 and FG-Q:prefix_L8+top1
  shuffle: FG-Q:<f>+top1 with prototype scores permuted across eligible non-seed patches

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_round7 --run $RUN --round R7
"""
import argparse, hashlib, json, multiprocessing as mp, os, pathlib, time
from collections import deque

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_graph as G
from AG3S.ag3s.experiments.studies import ifg_reader as R
from AG3S.ag3s.experiments.studies import ifg_round as RD
from AG3S.ag3s.experiments.studies import ifg_round_ns as NS
from AG3S.ag3s.experiments.studies import ifg_run as RN
from AG3S.ag3s.experiments.studies import ifg_s1 as S1

SHUFFLE_FEATS = ("prefix_L8", "projector")


def grow(sub, seed_nodes, allow):
    """BFS over sub.edges from seed_nodes through nodes where allow is True (seeds always included)."""
    n = len(sub.pos)
    adj = [[] for _ in range(n)]
    for a, b in sub.edges:
        adj[a].append(b); adj[b].append(a)
    on = np.zeros(n, bool)
    q = deque(int(s) for s in seed_nodes)
    on[list(q)] = True
    while q:
        i = q.popleft()
        for j in adj[i]:
            if not on[j] and allow[j]:
                on[j] = True; q.append(j)
    return on


def top1(m, sub, seed_lam):
    return RD.region_gate(m, sub, np.ones(len(m)), np.zeros(len(m)), 0.5, seed_lam, top1_only=True)[0]


def frame(args):
    ref, run, rnd, mask_dir = args
    out = run / "rounds" / rnd / "rows" / f"{ref.frame_id}.jsonl"
    if out.exists():
        return
    cfg = G.GraphConfig()
    fin = R.load_frame(ref)
    with np.load(run / "s0/replay" / f"{ref.frame_id}.npz") as z:
        depth, K, T = z["depth"], z["K"], z["T_base_cam"]
    with np.load(mask_dir / f"{ref.frame_id}.npz") as z:
        rmask = z["robot_mask"]
    with np.load(run / "s0/labels" / f"{ref.frame_id}.npz") as z:
        class_map, classes, cov = z["class_map"], [str(c) for c in z["classes"]], z["class_cov"]
    geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
    flat = S1.cov_flat(cov)
    vp = NS.eligible(geo, cfg, True)
    base = dict(frame_id=ref.frame_id, episode_id=ref.episode_id, ds=ref.ds, moment=ref.moment, kind=ref.kind,
                split="dev", level="voxel", threshold=0.5, voxel_size=cfg.voxel, seed_source="N")
    rows = []
    unit = {f: R.normalize(v) for f, v in fin.features.items()}
    rng = np.random.default_rng(20261015 + int(hashlib.sha1(ref.frame_id.encode()).hexdigest()[:6], 16))

    def run_seed(smap, tag, feats, with_r, with_shuffle):
        six = S1.top_k(smap, S1.SEED_K, vp)
        if six.size == 0:
            rows.append({**base, "arm_id": f"G-Q{tag}", "valid": False, "missing_reason": "seed_abstain"})
            return
        sw = smap[six] / smap[six].sum()
        sub = G.subset(geo, G.roi_nodes(geo, G.seed_unary(geo, six, sw), cfg.roi_radius))
        n = len(sub.pos)
        sl = G.seed_unary(sub, six, sw)
        sup = cfg.kappa_support * sub.support.astype(float)
        ones, zeros = np.ones(n), np.zeros(n)
        lam_g, s_g = RN._combine([(sl, ones), (sup, zeros)])
        w_geo = G.edge_weights(sub, cfg, None)
        arms = {}
        t0 = time.perf_counter()
        mG, _ = G.solve(sub, np.ones(n, bool), w_geo, lam_g, s_g, cfg.lam)
        arms[f"G-Q{tag}"] = (mG, (time.perf_counter() - t0) * 1000)
        seed_nodes = np.flatnonzero((sl > 0) & ~sub.support)
        if with_r:
            t0 = time.perf_counter()
            arms["G-R" + tag] = (np.where(grow(sub, seed_nodes, ~sub.support), 1.0, 0.0), (time.perf_counter() - t0) * 1000)
        for fn in feats:
            t0 = time.perf_counter()
            uf = unit[fn]
            mu = R.normalize((sw @ uf[six])[None])[0]
            pu, _ = G.proto_unary(uf @ mu, six, vp)
            node_p = G.node_patch_mean(sub, pu)
            w_f = G.edge_weights(sub, cfg, G.node_features(sub, uf))
            lam_u, s_u = RN._combine([(sl, ones), (sup, zeros), (np.full(n, cfg.kappa_proto), node_p)])
            base_ms = (time.perf_counter() - t0) * 1000
            for arm, (w, lu, su) in {"FGe-Q": (w_f, lam_g, s_g), "FGu-Q": (w_geo, lam_u, s_u), "FG-Q": (w_f, lam_u, s_u)}.items():
                t1 = time.perf_counter()
                arms[f"{arm}:{fn}{tag}"] = (G.solve(sub, np.ones(n, bool), w, lu, su, cfg.lam)[0], base_ms + (time.perf_counter() - t1) * 1000)
            if with_r:
                t1 = time.perf_counter()
                arms[f"FG-R:{fn}{tag}"] = (np.where(grow(sub, seed_nodes, ~sub.support & (node_p >= 0.5)), 1.0, 0.0),
                                           base_ms + (time.perf_counter() - t1) * 1000)
            if with_shuffle and fn in SHUFFLE_FEATS:
                perm_ix = np.flatnonzero(vp & ~np.isin(np.arange(768), six))
                ps = pu.copy(); ps[perm_ix] = pu[rng.permutation(perm_ix)]
                lam_s, s_s = RN._combine([(sl, ones), (sup, zeros), (np.full(n, cfg.kappa_proto), G.node_patch_mean(sub, ps))])
                arms[f"FG-Q-shuffle:{fn}{tag}"] = (G.solve(sub, np.ones(n, bool), w_f, lam_s, s_s, cfg.lam)[0], 0.0)
        for arm, (m, ms) in list(arms.items()):
            t1 = time.perf_counter()
            arms[arm + "+top1"] = (top1(m, sub, sl), ms + (time.perf_counter() - t1) * 1000)
        st = float(flat[six, classes.index("target")].mean())
        for arm, (m, ms) in arms.items():
            if tag == ":O" and not arm.endswith("+top1"):
                continue
            ev = G.evaluate(sub, m, 0.5, cfg)
            rows.append({**base, "arm_id": arm, "method": arm.split(":")[0], "valid": ev.get("missing_reason") is None,
                         "missing_reason": ev.get("missing_reason"), **ev, "ms_solve": ms, "seed_target_frac": st,
                         "n_roi": n, "oracle": tag == ":O"})

    run_seed(fin.maps["N"], "", tuple(R.FEATURES), True, True)
    run_seed(flat[:, classes.index("target")].copy(), ":O", ("prefix_L8",), False, False)  # GT seed: diagnostic only
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--round", default="R7")
    a = ap.parse_args()
    (a.run / "rounds" / a.round / "rows").mkdir(parents=True, exist_ok=True)
    md = a.run / "rounds/R6/robot_mask"
    refs = [r for r in R.list_frames() if (md / f"{r.frame_id}.npz").exists()]
    t0 = time.time()
    with mp.get_context("spawn").Pool(4) as p:
        p.map(frame, [(r, a.run, a.round, md) for r in refs])
    print("done", len(refs), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
