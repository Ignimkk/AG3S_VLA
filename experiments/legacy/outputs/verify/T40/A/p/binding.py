"""T40 A sensitivity (bollard): which t = 0 clearance binds, per pair.
For every bollard candidate on the 5 mm grid that blocks the recorded apple path (proxy, window [lift_t, place_t]) and is
not inside the crate footprint: the t = 0 GJK separation (overlap = 0, capped 50 mm) to apple / banana / orange / pear /
crate / robot, min over the two runs' t = 0 scenes. Two blocking modes: 'both' (blocks E0 and E3b, max of the two proxy
minima <= 0) and 'either' (min <= 0). Output: counts of candidates whose smallest t = 0 clearance exceeds 20/15/10/5/0 mm,
which item is the smallest for each candidate, and the candidate with the largest smallest-clearance.
usage: binding.py [TAG ...]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
import numpy as np, mujoco
import t40a_lib as L, search as S

CAPV = 0.05


def t0_vals(xy, runs):
    vals = {}
    for cond, R in runs.items():
        ctx = S._ctx(cond, R)
        L.set_q(R["q0"]); L.place("bollard_0", xy, 0.0); mujoco.mj_kinematics(L.m, L.d)
        obs = [(L.d.geom_xpos[o].copy(), float(L.m.geom_rbound[o]), L.gjk.make_support(L.m, L.d, int(o))) for o in L.OB["bollard_0"]]
        for n, lst in ctx:
            best = vals.get(n, CAPV)
            for g, cg, rg, sg in lst:
                for co, ro, so in obs:
                    v = cg - co
                    if np.linalg.norm(v) - ro - rg > best:
                        continue
                    best = min(best, L.gjk.gjk_dist(so, sg, v + 1e-9))
            vals[n] = best
    return vals


if __name__ == "__main__":
    allp = json.load(open(_ag3s_workspace_path('outputs/verify/T40/A/raw/carry_paths_22pairs.json')))["pairs"]
    sel = sys.argv[1:]
    out = {}
    for ent in allp:
        tag = f"{ent['episode']}_{ent['seed']}"
        if sel and tag not in sel:
            continue
        runs = {c: L.load_run(ent["runs"][c]["run_dir"]) for c in ("E0", "E3b")}
        win = {c: S.window(runs[c], ent["runs"][c]["grasp_t"], ent["runs"][c]["place_t"]) for c in runs}
        P = {c: runs[c]["q"][(runs[c]["t"] >= win[c][0]) & (runs[c]["t"] <= win[c][2]), 38:41] for c in runs}
        op = runs["E0"]["object_poses_t0"]
        gx, gy = np.meshgrid(np.arange(0.38, 0.85001, 0.005), np.arange(-0.35, 0.35001, 0.005))
        grid = np.stack([gx.ravel(), gy.ravel()], 1)
        d0, d3 = S.proxy_bollard(grid, P["E0"]), S.proxy_bollard(grid, P["E3b"])
        res = {}
        cache = {}
        for mode, sc in (("both", np.maximum(d0, d3)), ("either", np.minimum(d0, d3))):
            idx = [i for i in np.where(sc <= 0)[0] if not S.in_crate_fp(grid[i], op)]
            rows = []
            for i in idx:
                if i not in cache:
                    cache[i] = t0_vals(grid[i], runs)
                rows.append((grid[i].tolist(), float(sc[i]), float(d0[i]), float(d3[i]), cache[i]))
            tmin = np.array([min(r[4].values()) for r in rows]) if rows else np.zeros(0)
            best = rows[int(np.argmax(tmin))] if rows else None
            binding = {}
            for r in rows:
                k = min(r[4], key=r[4].get); binding[k] = binding.get(k, 0) + 1
            res[mode] = {"n_blocking_not_in_crate": len(rows),
                         "n_t0_min_gt_mm": {str(c): int((tmin > c / 1000).sum()) for c in (20, 15, 10, 5, 0)},
                         "smallest_item_counts": binding,
                         "max_t0_clearance_candidate": best and {"xy": best[0], "proxy_score_m": best[1], "proxy_E0_m": best[2],
                                                                 "proxy_E3b_m": best[3], "t0_gjk_min_by_item_m": best[4]}}
        out[tag] = res
        print(tag, {m: (r["n_blocking_not_in_crate"], r["n_t0_min_gt_mm"], r["smallest_item_counts"],
                        r["max_t0_clearance_candidate"] and round(1e3 * min(r["max_t0_clearance_candidate"]["t0_gjk_min_by_item_m"].values()), 1))
                    for m, r in res.items()}, flush=True)
        json.dump(res, open(f"${AG3S_PROJECT_ROOT}/outputs/verify/T40/A/raw/binding_{tag}.json", "w"), indent=1)
