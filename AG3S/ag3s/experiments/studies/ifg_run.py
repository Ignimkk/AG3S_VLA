"""IFG batch runner — S1 patch diagnostics + S2/S3 graph arms per dump frame (protocol v1.1 §5–7).

Inputs: dump frame (features, maps; `ifg_reader`), S0 replay (`<s0>/replay/<id>.npz`: depth, FK robot
mask, K, T — algorithm inputs) and S0 labels (`<s0>/labels/<id>.npz`: class map — evaluator only).
GT is read only after every score/membership of the frame is computed.

Output (one file per frame, written atomically; existing files are skipped):
  <out>/rows/<frame_id>.jsonl   one row per (arm, threshold) at level "patch" (S1, K) or "voxel" (S2/S3)
  <out>/masks/<frame_id>.npz    eval-voxel centres + memberships of a few arms (overlay figures)

  MUJOCO_GL=osmesa PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_run \
      --s0 $RUN/s0 --out $RUN/s3 --workers 4
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import pathlib
import time

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_graph as G
from AG3S.ag3s.experiments.studies import ifg_reader as R
from AG3S.ag3s.experiments.studies import ifg_s1 as S1

ARMS_PER_FEATURE = ("F", "FGe", "FGu", "FG")
MASK_ARMS = ("G:N", "F:N:projector", "FG:N:projector", "F:N:siglip_encoded", "FG:N:siglip_encoded",
             "G:A", "FG:A:projector")


def _combine(parts):
    """Unary terms [(lam, s)] -> (Λ, s) with s the Λ-weighted mean."""
    lam = sum(p[0] for p in parts)
    num = sum(p[0] * p[1] for p in parts)
    s = np.where(lam > 0, num / np.maximum(lam, 1e-300), 0.0)
    return lam, s


def run_frame(ref: R.FrameRef, s0: pathlib.Path, cfg: G.GraphConfig, oracle: bool = False) -> tuple[list, dict]:
    """`oracle`: diagnostic only — seed source O = GT target fraction per patch (protocol §11 oracle identity)."""
    t_all = time.perf_counter()
    fin = R.load_frame(ref)
    with np.load(s0 / "replay" / f"{ref.frame_id}.npz") as z:
        depth, rmask, K, T = z["depth"], z["robot_mask"], z["K"], z["T_base_cam"]
    with np.load(s0 / "labels" / f"{ref.frame_id}.npz") as z:  # evaluator-only content, see below
        class_map, classes, class_cov = z["class_map"], [str(c) for c in z["classes"]], z["class_cov"]
    # NOTE: class_map enters build_geometry only to attach per-voxel labels for `evaluate`;
    # no algorithm array below reads eval_* fields.
    geo = G.build_geometry(depth, K, T, rmask, class_map, classes, cfg)
    base = dict(run_cfg=dataclasses.asdict(cfg), frame_id=ref.frame_id, episode_id=ref.episode_id, ds=ref.ds,
                moment=ref.moment, kind=ref.kind, split="dev")
    rows = []
    # ------------------------------------------------------------ S1 patch level (raw-all-patch)
    t0 = time.perf_counter()
    scores, seeds = S1.score_frame(fin.features, fin.maps)
    ms_s1 = (time.perf_counter() - t0) * 1000
    flat = S1.cov_flat(class_cov)
    for arm, sc in scores.items():
        for Kk in S1.TOPK_EVAL:
            r = S1.patch_metrics(sc, flat, classes, Kk)
            rows.append({**base, "level": "patch", "arm_id": arm, "K": Kk, **r,
                         **(S1.seed_purity(seeds[arm], flat, classes) if arm in seeds else {})})
    oracle_patch = S1.oracle_patch_iou(flat, classes)
    # ------------------------------------------------------------ S2/S3 graph arms
    unit = {f: R.normalize(v) for f, v in fin.features.items()}
    valid_patch = G.valid_seed_patches(geo, cfg)
    masks = {}
    seeds_run = S1.SEEDS
    if oracle:
        fin.maps["O"] = flat[:, classes.index("target")].copy()  # GT: oracle seed, never a method input
        seeds_run = ("O",)
        rows = [r for r in rows if r["level"] != "patch"]
    for sd in seeds_run:
        seed_ix = S1.top_k(fin.maps[sd], S1.SEED_K, valid_patch)
        seed_info = dict(seed_source=sd, seed_patches=seed_ix.tolist(), n_valid_seed_patches=int(valid_patch.sum()),
                         seed_target_frac=(float(flat[seed_ix, classes.index("target")].mean()) if seed_ix.size else None))
        if seed_ix.size == 0:
            rows.append({**base, "level": "voxel", "arm_id": f"G:{sd}", "valid": False, "missing_reason": "seed_abstain",
                         **seed_info})
            continue
        sw = fin.maps[sd][seed_ix] / fin.maps[sd][seed_ix].sum()
        t1 = time.perf_counter()
        roi = G.roi_nodes(geo, G.seed_unary(geo, seed_ix, sw), cfg.roi_radius)
        sub = G.subset(geo, roi)
        ms_roi = (time.perf_counter() - t1) * 1000
        all_in = np.ones(len(sub.pos), bool)
        seed_lam = G.seed_unary(sub, seed_ix, sw)
        sup_lam = cfg.kappa_support * sub.support.astype(float)
        w_geo = G.edge_weights(sub, cfg, None)
        lam_g, s_g = _combine([(seed_lam, np.ones_like(seed_lam)), (sup_lam, np.zeros_like(sup_lam))])
        m, info = G.solve(sub, all_in, w_geo, lam_g, s_g, cfg.lam)
        info["ms_solve"] += ms_roi
        results = {f"G:{sd}": (m, info, None)}
        for fn, uf in unit.items():
            t1 = time.perf_counter()
            mu = R.normalize((sw @ uf[seed_ix])[None])[0]
            cos = uf @ mu
            pu, pinfo = G.proto_unary(cos, seed_ix, valid_patch)
            node_p = G.node_patch_mean(sub, pu)
            ms_proto = (time.perf_counter() - t1) * 1000 + ms_roi
            results[f"F:{sd}:{fn}"] = (node_p.copy(), dict(ms_solve=ms_proto, n_roi=int(len(sub.pos))), pinfo)
            t2 = time.perf_counter()
            w_f = G.edge_weights(sub, cfg, G.node_features(sub, uf))
            ms_feat = (time.perf_counter() - t2) * 1000
            lam_u, s_u = _combine([(seed_lam, np.ones_like(seed_lam)), (sup_lam, np.zeros_like(sup_lam)),
                                   (np.full(len(node_p), cfg.kappa_proto), node_p)])
            for arm, (w, lu, su, extra_ms) in {"FGe": (w_f, lam_g, s_g, ms_feat), "FGu": (w_geo, lam_u, s_u, 0.0),
                                                "FG": (w_f, lam_u, s_u, ms_feat)}.items():
                mm, ii = G.solve(sub, all_in, w, lu, su, cfg.lam)
                ii["ms_solve"] += ms_proto + extra_ms
                results[f"{arm}:{sd}:{fn}"] = (mm, ii, pinfo)
        geo_eval = sub
        for arm, (mm, ii, pinfo) in results.items():
            parts = arm.split(":")
            for tau in cfg.thresholds:
                ev = G.evaluate(geo_eval, mm, tau, cfg)
                rows.append({**base, "level": "voxel", "arm_id": arm, "method": parts[0], "seed_source": sd,
                             "feature_name": parts[2] if len(parts) > 2 else None, "threshold": tau,
                             "voxel_size": cfg.voxel, "valid": ev.get("missing_reason") is None,
                             "missing_reason": ev.get("missing_reason"), **ev, **seed_info,
                             **{k: v for k, v in ii.items()}, **({"proto": pinfo} if pinfo else {})})
            if arm in MASK_ARMS:
                full = np.full(len(geo.pos), -1.0)
                full[np.flatnonzero(roi)] = mm
                masks[arm] = full
    # eval-voxel level arrays for overlays (membership mapped back to eval voxels, max over nodes)
    V = len(geo.eval_target)
    out_masks = {"eval_target": geo.eval_target.astype(np.float16),
                 "eval_robot_removed": geo.eval_robot_removed,
                 "node_pos": geo.pos.astype(np.float32), "node_eval": geo.node_eval.astype(np.int32),
                 "support": geo.support}
    for arm, mm in masks.items():
        out_masks["m:" + arm] = mm.astype(np.float16)
    frame_info = dict(frame_id=ref.frame_id, geometry=geo.stats, oracle_patch=oracle_patch, ms_s1=ms_s1,
                      ms_total=(time.perf_counter() - t_all) * 1000, n_eval_voxels=V)
    return rows, {"info": frame_info, "masks": out_masks}


def _job(args):
    ref, s0, out, cfg, oracle = args
    path = out / "rows" / f"{ref.frame_id}.jsonl"
    if path.exists():
        return ref.frame_id, None
    try:
        rows, extra = run_frame(ref, s0, cfg, oracle)
    except Exception as e:  # recorded, never silently dropped
        with open(out / "failures.jsonl", "a") as f:
            f.write(json.dumps({"frame_id": ref.frame_id, "error": repr(e)}) + "\n")
        return ref.frame_id, repr(e)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    np.savez_compressed(out / "masks" / f"{ref.frame_id}.npz", **extra["masks"])
    (out / "frames").mkdir(exist_ok=True)
    (out / "frames" / f"{ref.frame_id}.json").write_text(json.dumps(extra["info"], default=float))
    os.replace(tmp, path)
    return ref.frame_id, f"{extra['info']['ms_total'] / 1000:.1f}s nodes={extra['info']['geometry']['n_nodes']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s0", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--frames", nargs="*", default=None)
    ap.add_argument("--roi", type=float, default=0.15, help="<=0: full scene")
    ap.add_argument("--jump-gate", type=float, default=0.010)
    ap.add_argument("--voxel", type=float, default=0.005)
    ap.add_argument("--oracle", action="store_true", help="diagnostic: GT seed (arm suffix :O)")
    a = ap.parse_args()
    cfg = G.GraphConfig(roi_radius=(a.roi if a.roi > 0 else None), jump_gate=a.jump_gate, voxel=a.voxel)
    for d in ("rows", "masks"):
        (a.out / d).mkdir(parents=True, exist_ok=True)
    (a.out / "resolved_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=1))
    refs = [r for r in R.list_frames() if (a.s0 / "replay" / f"{r.frame_id}.npz").exists()]
    if a.frames:
        refs = [r for r in refs if r.frame_id in a.frames]
    t0 = time.time()
    jobs = [(r, a.s0, a.out, cfg, a.oracle) for r in refs]
    with mp.get_context("spawn").Pool(min(a.workers, 4)) as pool:
        for fid, msg in pool.imap_unordered(_job, jobs):
            print(fid, msg, f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
