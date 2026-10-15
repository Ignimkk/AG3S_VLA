"""IFG R5 (F1): seed eligibility on non-robot, non-support pixels (serving excludes support from seeds).

Arms: G:N (R0 seed rule; parent, parity with R0), G-ns:N, top1-only:N (R3 rule), top1-only-ns:N.
  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_round_ns --run $RUN --round R5
"""
import argparse, json, multiprocessing as mp, os, pathlib, time

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_graph as G
from AG3S.ag3s.experiments.studies import ifg_reader as R
from AG3S.ag3s.experiments.studies import ifg_round as RD
from AG3S.ag3s.experiments.studies import ifg_run as RN
from AG3S.ag3s.experiments.studies import ifg_s1 as S1


def eligible(geo, cfg, nonsupport):
    if not nonsupport:
        return G.valid_seed_patches(geo, cfg)
    return np.asarray(geo.patch_px[~geo.support].sum(0)).ravel() >= cfg.min_seed_patch_px


def g_and_top1(geo, cfg, smap, vp):
    six = S1.top_k(smap, S1.SEED_K, vp)
    if six.size == 0:
        return None
    t0 = time.perf_counter()
    sw = smap[six] / smap[six].sum()
    sub = G.subset(geo, G.roi_nodes(geo, G.seed_unary(geo, six, sw), cfg.roi_radius))
    n = len(sub.pos)
    sl = G.seed_unary(sub, six, sw)
    sup = cfg.kappa_support * sub.support.astype(float)
    lam, s = RN._combine([(sl, np.ones(n)), (sup, np.zeros(n))])
    m, info = G.solve(sub, np.ones(n, bool), G.edge_weights(sub, cfg, None), lam, s, cfg.lam)
    px = np.asarray(sub.patch_px.sum(1)).ravel()
    mt = RD.region_gate(m, sub, px, np.zeros(n), 0.5, sl, top1_only=True)[0]
    info["ms_solve"] = (time.perf_counter() - t0) * 1000
    return sub, six, m, mt, info


def frame(args):
    ref, run, rnd, mask_dir = args
    out = run / "rounds" / rnd / "rows" / f"{ref.frame_id}.jsonl"
    if out.exists():
        return
    cfg = G.GraphConfig()
    fin = R.load_frame(ref, features=())
    s0 = run / "s0"
    with np.load(s0 / "replay" / f"{ref.frame_id}.npz") as z:
        depth, rmask, K, T = z["depth"], z["robot_mask"], z["K"], z["T_base_cam"]
    if mask_dir is not None:  # R6: rendered-robot-depth mask replaces the serving FK sphere mask
        with np.load(mask_dir / f"{ref.frame_id}.npz") as z:
            rmask = z["robot_mask"]
    with np.load(s0 / "labels" / f"{ref.frame_id}.npz") as z:
        class_map, classes, cov = z["class_map"], [str(c) for c in z["classes"]], z["class_cov"]
    geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
    flat = S1.cov_flat(cov)
    base = dict(frame_id=ref.frame_id, episode_id=ref.episode_id, ds=ref.ds, moment=ref.moment, kind=ref.kind,
                split="dev", level="voxel", threshold=0.5, voxel_size=cfg.voxel, seed_source="N")
    rows = []
    rm = "" if mask_dir is None else "+rm"
    if mask_dir is not None:  # parent rows (serving mask) copied from R5 for the paired comparison
        rows += [json.loads(l) for l in open(run / "rounds/R5/rows" / f"{ref.frame_id}.jsonl")]
    for tag, ns in (("", False), ("-ns", True)):
        vp = eligible(geo, cfg, ns)
        res = g_and_top1(geo, cfg, fin.maps["N"], vp)
        if res is None:
            for arm in (f"G{tag}{rm}:N", f"top1-only{tag}{rm}:N"):
                rows.append({**base, "arm_id": arm, "valid": False, "missing_reason": "seed_abstain"})
            continue
        sub, six, m, mt, info = res
        st = float(flat[six, classes.index("target")].mean())
        for arm, mm in ((f"G{tag}{rm}:N", m), (f"top1-only{tag}{rm}:N", mt)):
            ev = G.evaluate(sub, mm, 0.5, cfg)
            rows.append({**base, "arm_id": arm, "method": arm.split(":")[0], "valid": ev.get("missing_reason") is None,
                         "missing_reason": ev.get("missing_reason"), **ev, **info, "seed_patches": six.tolist(),
                         "seed_target_frac": st, "n_eligible_patches": int(vp.sum())})
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--round", default="R5")
    ap.add_argument("--mask-dir", type=pathlib.Path, default=None)
    a = ap.parse_args()
    (a.run / "rounds" / a.round / "rows").mkdir(parents=True, exist_ok=True)
    refs = [r for r in R.list_frames() if (a.run / "s0/replay" / f"{r.frame_id}.npz").exists()]
    with mp.get_context("spawn").Pool(4) as p:
        p.map(frame, [(r, a.run, a.round, a.mask_dir) for r in refs])
    print("done", len(refs))


if __name__ == "__main__":
    main()
