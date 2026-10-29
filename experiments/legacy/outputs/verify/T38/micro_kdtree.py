"""T38 micro-measurement: cKDTree query cost with workers=-1 (as in the code) vs workers=1, on the real inputs of
target_grounding.grow_region / dbscan / robot_sphere_mask for one recorded chunk. Measures only; the worktree
functions are called unchanged for the workers=-1 numbers, and the workers=1 numbers come from a copy of the
same loop in this file with only the `workers` argument changed.
usage: PYTHONPATH=${AG3S_PROJECT_ROOT}-o4 .venv-openpi-live/bin/python micro_kdtree.py <server_chunk_npz> <label>
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, os, sys, time, pathlib
import numpy as np
from scipy.spatial import cKDTree
from AG3S.ag3s.stages import target_grounding as TG
from AG3S.ag3s.config import AG3SConfig

npz = sys.argv[1]; label = sys.argv[2]
z = np.load(npz, allow_pickle=False)
pts = z["raw_cloud"].astype(np.float64); sup = z["support_mask"]; seeds = z["seed_indices"]
cfg = AG3SConfig.from_dict({}).clustering
cand = np.nonzero(~sup)[0]
local_seed = np.searchsorted(cand, seeds[np.isin(seeds, cand)])
P = pts[cand]


def grow(points, seed_idx, radius, workers):
    n = points.shape[0]; visited = np.zeros(n, bool); visited[seed_idx] = True
    tree = cKDTree(points); frontier = seed_idx; waves = 0
    while frontier.size:
        nb = tree.query_ball_point(points[frontier], radius, workers=workers); waves += 1
        flat = np.fromiter((j for g in nb for j in g), np.int64, count=sum(len(g) for g in nb))
        if flat.size == 0:
            break
        c = np.unique(flat); frontier = c[~visited[c]]; visited[frontier] = True
    return np.nonzero(visited)[0], waves


def t(fn, rep=5):
    out = []
    for _ in range(rep):
        a = time.perf_counter(); r = fn(); out.append((time.perf_counter() - a) * 1e3)
    return float(np.median(out)), r


res = {"npz": npz, "label": label, "cpu_count": os.cpu_count(), "loadavg": os.getloadavg(),
       "n_fused": int(len(pts)), "n_candidates": int(len(P)), "n_seeds": int(len(local_seed)), "eps": cfg.eps}
ms, g = t(lambda: TG.grow_region(P, local_seed, cfg.eps))
res["grow_region_code_workers-1_ms"] = ms; res["n_grown"] = int(len(g))
for w in (1, 8, -1):
    ms, (g2, waves) = t(lambda: grow(P, local_seed, cfg.eps, w))
    res[f"grow_copy_workers{w}_ms"] = ms; res["waves"] = waves; res["same_result"] = bool(np.array_equal(g, g2))
G = P[g]
ms, lab = t(lambda: TG.dbscan(G, cfg.eps, cfg.min_points))
res["dbscan_code_ms"] = ms; res["dbscan_n_in"] = int(len(G))
tree = cKDTree(G)
for w in (1, -1):
    res[f"dbscan_ball_count_workers{w}_ms"] = t(lambda: tree.query_ball_point(G, cfg.eps, return_length=True, workers=w))[0]
    res[f"nn_query_workers{w}_ms"] = t(lambda: tree.query(G, k=1, distance_upper_bound=cfg.eps, workers=w))[0]
# robot_sphere_mask pattern: 587 single-point queries on a ~50k cloud (synthetic centres inside the cloud bbox)
rng = np.random.default_rng(0)
C = pts[rng.choice(len(pts), 587, replace=False)]
big = cKDTree(pts)
for w in (1, -1):
    res[f"587_single_point_ball_queries_workers{w}_ms"] = t(lambda: [big.query_ball_point(c, 0.05, workers=w) for c in C], rep=3)[0]
res["587_batched_ball_query_workers1_ms"] = t(lambda: big.query_ball_point(C, 0.05, workers=1), rep=3)[0]
out = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38/micro')); out.mkdir(exist_ok=True)
(out / f"micro_kdtree_{label}.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
