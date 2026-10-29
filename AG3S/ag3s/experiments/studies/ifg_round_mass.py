"""IFG R4 (F1): identity by region-level noun attention mass on the depth-surface graph.

Regions = connected components of the full-scene surface graph (ifg_graph, range 2.0 m) after removing
support (serving RANSAC) and FK-robot nodes. Node mass = sum_p map_p * px_ip / px_p (each patch's
score split over its non-robot pixels: patch mass conserved). The region with the largest summed
mass is the target; extent = all its nodes. Parent rows G:N and top1-only:N are copied from R3.

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_round_mass --run $RUN --round R4
"""
import argparse, json, multiprocessing as mp, os, pathlib, time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from AG3S.ag3s.experiments.studies import ifg_graph as G
from AG3S.ag3s.experiments.studies import ifg_reader as R


def regions(geo, keep):
    n = len(geo.pos)
    e = geo.edges[keep[geo.edges[:, 0]] & keep[geo.edges[:, 1]]]
    W = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    _, comp = connected_components(W, directed=False)
    comp = np.where(keep, comp, -1)
    return comp


def mass_top1(geo, comp, patch_score):
    node_m = geo.patch_px @ (np.asarray(patch_score, float) / np.maximum(geo.patch_alg_px, 1.0))
    ok = comp >= 0
    rm = np.bincount(comp[ok], weights=node_m[ok], minlength=comp.max() + 1)
    best = int(np.argmax(rm))
    m = np.where(comp == best, 1.0, -1.0)
    srt = np.sort(rm)[::-1]
    return m, dict(region_mass=float(rm[best]), mass_ratio_2nd=float(srt[1] / srt[0]) if len(srt) > 1 and srt[0] > 0 else None,
                   region_nodes=int((comp == best).sum()), n_regions=int(len(rm)))


def frame(args):
    ref, run, rnd = args
    out = run / "rounds" / rnd / "rows" / f"{ref.frame_id}.jsonl"
    if out.exists():
        return
    cfg = G.GraphConfig()
    fin = R.load_frame(ref, features=())
    s0 = run / "s0"
    with np.load(s0 / "replay" / f"{ref.frame_id}.npz") as z:
        depth, rmask, K, T = z["depth"], z["robot_mask"], z["K"], z["T_base_cam"]
    with np.load(s0 / "labels" / f"{ref.frame_id}.npz") as z:
        class_map, classes = z["class_map"], [str(c) for c in z["classes"]]
    geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
    t0 = time.perf_counter()
    comp = regions(geo, ~geo.support)
    ms_reg = (time.perf_counter() - t0) * 1000
    base = dict(frame_id=ref.frame_id, episode_id=ref.episode_id, ds=ref.ds, moment=ref.moment, kind=ref.kind,
                split="dev", level="voxel", threshold=0.5, voxel_size=cfg.voxel)
    rows = [json.loads(l) for l in open(run / "rounds/R3/rows" / f"{ref.frame_id}.jsonl")
            if json.loads(l)["arm_id"] in ("G:N", "top1-only:N")]
    arms = {}
    for src in ("N", "A"):
        t1 = time.perf_counter()
        m, info = mass_top1(geo, comp, fin.maps[src])
        info["ms_solve"] = ms_reg + (time.perf_counter() - t1) * 1000
        arms[f"MassTop1:{src}"] = (m, info)
    # oracle: the region holding the most GT target (diagnostic only)
    ok = comp >= 0
    tg = np.bincount(comp[ok], weights=geo.eval_target[geo.node_eval][ok], minlength=comp.max() + 1)
    arms["O-region-best"] = (np.where(comp == int(np.argmax(tg)), 1.0, -1.0), dict(oracle=True))
    for arm, (m, info) in arms.items():
        ev = G.evaluate(geo, m, 0.5, cfg)
        rows.append({**base, "arm_id": arm, "method": arm.split(":")[0], "seed_source": arm.split(":")[1] if ":" in arm else None,
                     "valid": ev.get("missing_reason") is None, "missing_reason": ev.get("missing_reason"), **ev, **info})
    tmp = out.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, default=float) + "\n" for r in rows))
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--round", default="R4")
    a = ap.parse_args()
    (a.run / "rounds" / a.round / "rows").mkdir(parents=True, exist_ok=True)
    refs = [r for r in R.list_frames() if (a.run / "rounds/R3/rows" / f"{r.frame_id}.jsonl").exists()]
    with mp.get_context("spawn").Pool(4) as p:
        p.map(frame, [(r, a.run, a.round) for r in refs])
    print("done", len(refs))


if __name__ == "__main__":
    main()
