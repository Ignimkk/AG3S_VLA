"""T44c S1 candidates (verifier, 2026-10-10). Fixed BEFORE any run. CPU, dataset metadata only.
Pool: episode_index < 1600 and meta split == 'train', success true, nearest OTHER fruit xy distance (initial_fruit_poses) < 0.10 m.
bin: lt_0.08 / 0.08_0.10.  cells = (target_fruit x used_arm x bin) = 16, order fruit apple,orange,pear,banana x arm left,right x bin lt_0.08,0.08_0.10.
Pass 1: the smallest episode index of every cell (16).  Pass 2: the second smallest episode of the 8 lt_0.08 cells (more crowded) (8).  Total 24.
E2-demo overlap: episodes in outputs/verify/T44/E2/e2_demo_frames.json groups.  seed = 10*ep + rep, rep = 1,2 (LB/T42 rule)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, numpy as np
DS = _ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/meta/randomized_episodes.jsonl')
R = [json.loads(l) for l in open(DS) if l.strip()]
pool = {}
for r in R:
    if r["episode_index"] >= 1600 or r["split"] != "train" or not r["success"]:
        continue
    fp = r["initial_fruit_poses"]; tf = r["target_fruit"]
    d = {k: float(np.hypot(*(np.array(fp[k][:2]) - np.array(fp[tf][:2])))) for k in fp if k != tf}
    nk = min(d, key=d.get); nn = d[nk]
    if nn < 0.10:
        pool[r["episode_index"]] = {"episode": r["episode_index"], "target_fruit": tf, "used_arm": r["used_arm"], "layout_index": r["layout_index"],
            "nn_other_fruit": nk, "nn_xy_m": nn, "bin": "lt_0.08" if nn < 0.08 else "0.08_0.10",
            "grasp_attempts": r["grasp_attempts"], "grasp_retries": r["grasp_retries"]}
e2 = json.load(open(_ag3s_workspace_path('outputs/verify/T44/E2/e2_demo_frames.json')))["groups"]
if isinstance(e2, str): e2 = eval(e2)
e2eps = {g["episode"] for g in e2}
cells = {}; order = []
for f in ("apple", "orange", "pear", "banana"):
    for a in ("left", "right"):
        for b in ("lt_0.08", "0.08_0.10"):
            k = f"{f}_{a}_{b}"; order.append(k)
            cells[k] = sorted(e for e, v in pool.items() if v["target_fruit"] == f and v["used_arm"] == a and v["bin"] == b)
sel = []
for k in order: sel.append((k, 1, cells[k][0]))
for k in order:
    if k.endswith("lt_0.08"): sel.append((k, 2, cells[k][1]))
eps = []
for k, rk, e in sel:
    v = dict(pool[e]); v.update(cell=k, pick_rank_in_cell=rk, in_e2_demo=e in e2eps, seeds=[10 * e + 1, 10 * e + 2]); eps.append(v)
out = {"definitions": __doc__, "n_pool": len(pool), "cell_pool_sizes": {k: len(v) for k, v in cells.items()}, "n_candidates": len(eps),
       "n_in_e2_demo": sum(v["in_e2_demo"] for v in eps), "episodes": eps}
pathlib.Path("candidates.json").write_text(json.dumps(out, indent=1))
for v in eps: print(v["episode"], v["cell"], v["pick_rank_in_cell"], round(v["nn_xy_m"], 4), v["nn_other_fruit"], "E2demo" if v["in_e2_demo"] else "")
print(out["n_pool"], out["n_candidates"], out["n_in_e2_demo"])
