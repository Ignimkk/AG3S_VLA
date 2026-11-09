"""(offline, 2026-11-09 06:0x dev-4a) candidate SL-PE11: drop the manipulated object from the scene obstacle-object set by
POINTS (within 15 mm of its points, + the 0-30 mm band under its base footprint) instead of whole 15 mm components.
Why: pe12345789 SafeLIBERO t3 e2 / e4 -- the moment the identity became the real bowl (call 5), the book / mug touching it
left the obstacle set (123-174 voxels -> 0) because they were one component with the bowl; the hand then hit them.
Per pre-held call (recorded cloud, identity = recorded target_points): GT-scored voxels of (a) the SafeLIBERO obstacle and
(b) the target that stay in the set, component rule vs point rule. Run: SCENE_DIR=.. [SAFELIBERO_PATCH=1] python pe11_pointdrop.py '<glob>'"""
import collections
import json
import pathlib
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/mnt/dev/AG3S_VLA/outputs/2026-11-08_PE_경로끝점_identity")
import pe_signal as PS  # noqa: E402

Q = PS.Q
V = 0.015


import os
MODE = os.environ.get("PE11_MODE", "point")


def point_rule(pts, support, tgt):
    pts = pts[~support] if support is not None and len(support) == len(pts) else pts
    if not len(pts):
        return np.zeros((0, 3))
    keep = np.ones(len(pts), bool)
    if tgt is not None and len(tgt):
        if MODE == "point":
            keep &= cKDTree(tgt).query(pts)[0] > V
        elif MODE == "cut":   # identity points + the object re-cut at the identity centre (support band / walls removed)
            keep &= cKDTree(tgt).query(pts)[0] > V
            sys.path.insert(0, "/mnt/dev/AG3S_VLA")
            from AG3S.ag3s.stages.close_seed import split_at_seed
            idx, _cut = split_at_seed(pts, tgt.mean(axis=0))
            if idx is not None:
                keep[np.asarray(idx, int)] = False
        else:   # bbox / hybrid: inside the identity's axis-aligned box + 15 mm
            lo_, hi_ = tgt.min(axis=0) - V, tgt.max(axis=0) + V
            keep &= ~np.all((pts >= lo_) & (pts <= hi_), axis=1)
            if MODE == "hybrid":
                from scipy import ndimage
                lo = pts.min(axis=0)
                ijk = np.floor((pts - lo) / V).astype(np.int64)
                occ = np.zeros(tuple(int(x) for x in ijk.max(axis=0) + 1), bool)
                occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
                lab, _ = ndimage.label(occ, structure=np.ones((3, 3, 3), bool))
                comp = lab[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
                ext = float(np.ptp(tgt, axis=0).max()) + 0.04
                for ids in set(int(comp[i]) for ids_ in cKDTree(pts).query_ball_point(tgt, V) for i in ids_):
                    cm = comp == ids
                    if np.ptp(pts[cm], axis=0).max() <= ext:
                        keep &= ~cm
        zmin = float(tgt[:, 2].min())
        under = (pts[:, 2] <= zmin + 0.005) & (pts[:, 2] >= zmin - 0.03)
        base = tgt[tgt[:, 2] <= zmin + 0.01]
        if under.any():
            near = cKDTree(base[:, :2]).query(pts[under, :2])[0] <= 0.01
            idx = np.flatnonzero(under)[near]
            keep[idx] = False
    sel = pts[keep]
    if len(sel):
        key = np.floor(sel / V).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        sums = np.zeros((inv.max() + 1, 3))
        np.add.at(sums, inv, sel)
        sel = sums / np.bincount(inv)[:, None]
    return sel


def main():
    runs = [p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID") and (p / "episodes.jsonl").exists()]
    tot = collections.Counter()
    for run in runs:
        eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
        order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
        obst = {o: (e.get("safelibero") or {}).get("obstacle") for o, e in zip(order, eps)}
        ch = PS.episode_chunks(run, order)
        for stem in order:
            if not (Q.SCENE / f"{stem}_scene.xml").exists():
                continue
            info, m, rg, cache, stc = PS.scene_for(stem)
            d = mujoco.MjData(m)
            gb = np.array([m.body(b).name for b in m.geom_bodyid])
            geom = np.load(run / "episodes" / f"{stem}_geom.npz")
            grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
            calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
            line = []
            for ts, cp in ch.get(stem, []):
                c = calls.get(ts)
                if c is None or (c.get("grasp") or {}).get("state") not in ("searching", "latched", "closing"):
                    continue
                z = np.load(cp)
                tgt = z["target_points"].astype(float) if "target_points" in z.files and len(z["target_points"]) else None
                i = grow[int(c["obs_index"])]
                d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
                mujoco.mj_forward(m, d)
                S1, G1 = stc.world(d)
                S2, G2 = cache.world(d)
                S = np.r_[S1, S2]
                B = np.r_[gb[G1], gb[G2]]
                R = np.r_[rg[G1], rg[G2]]
                k = R != "robot"
                tree = cKDTree(S[k])
                Bk, Rk = B[k], R[k]
                Tw = geom["T_world_base"][i]
                out = []
                for rule in ("comp", "point"):
                    ob = (Q.scene_obstacles(z["filtered_cloud"].astype(float), z["support_mask"].astype(bool), tgt, None)
                          if rule == "comp" else point_rule(z["filtered_cloud"].astype(float), z["support_mask"].astype(bool), tgt))
                    if not len(ob):
                        out.append((0, 0))
                        continue
                    dd, jj = tree.query(ob @ Tw[:3, :3].T + Tw[:3, 3])
                    okd = dd <= 0.015
                    n_obs = int(sum(str(b).startswith(obst[stem]) for b in Bk[jj][okd])) if obst.get(stem) else 0
                    n_tgt = int((Rk[jj][okd] == "target").sum())
                    out.append((n_obs, n_tgt))
                    tot[(rule, "obs")] += n_obs
                    tot[(rule, "tgt")] += n_tgt
                tot["calls"] += 1
                if tgt is not None:
                    tot["calls_with_id"] += 1
                line.append(f"{ts}:{out[0][0]}/{out[1][0]}|{out[0][1]}/{out[1][1]}")
            print(run.name[16:52], stem[-6:], "seq:obstacle comp/point|target comp/point ->", " ".join(line), flush=True)
    print("TOTAL", dict(tot))


if __name__ == "__main__":
    main()
