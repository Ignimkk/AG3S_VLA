"""T43 LB candidate pools (verifier, 2026-10-09). CPU only, dataset metadata only.
Pool = TRAIN split (episode 0-1599, meta/randomized_episodes.jsonl split == 'train'), place_one, record success true,
grasp_attempts 1, grasp_retries 0 (the T42-1 candidates_ext.py filter, but train only).
score_m = crowd score of T42 (min of nn_target, gap_target_crate, gap_crate_others), larger = less crowded.
Per (fruit, arm) ranked by (-score_m, episode). usage: candidates_lb.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
DS = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/meta/randomized_episodes.jsonl'))
SPL = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/meta/randomized_splits.json'))
def local(p, crate):
    w, x, y, z = crate[3:7]
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    return R.T @ (np.asarray(p[:3]) - np.asarray(crate[:3]))
def rect_dist(l):
    return float(np.linalg.norm(np.maximum(np.abs(l[:2]) - np.array([0.090, 0.150]), 0.0)))
recs = [json.loads(l) for l in DS.read_text().splitlines() if l.strip()]
bounds = {}
for r in recs:
    b = bounds.setdefault(r["split"], [10**9, -1, 0]); b[0] = min(b[0], r["episode_index"]); b[1] = max(b[1], r["episode_index"]); b[2] += 1
pool = [r for r in recs if r.get("split") == "train" and r["task_type"] == "place_one" and r.get("success")
        and r.get("grasp_retries") == 0 and r.get("grasp_attempts") == 1]
eps = {}
for r in pool:
    e = r["episode_index"]; tf = r["target_fruit"]; crate = r["basket_pose"]; fp = r["initial_fruit_poses"]
    others = [k for k in fp if k != tf]; pt = np.asarray(fp[tf][:2])
    nn = min(float(np.linalg.norm(pt - np.asarray(fp[k][:2]))) for k in others)
    gtc = rect_dist(local(fp[tf], crate)); gco = min(rect_dist(local(fp[k], crate)) for k in others)
    eps[e] = {"split": r["split"], "target_fruit": tf, "used_arm": r["used_arm"], "layout_index": r["layout_index"], "nn_target_m": nn,
              "gap_target_crate_m": gtc, "gap_crate_others_m": gco, "score_m": min(nn, gtc, gco)}
cands = {}
for f in ("apple", "orange", "pear", "banana"):
    for a in ("left", "right"):
        ranked = sorted([e for e, v in eps.items() if v["target_fruit"] == f and v["used_arm"] == a], key=lambda e: (-eps[e]["score_m"], e))
        cands[f"{f}_{a}"] = {"pool_size": len(ranked), "ranked": ranked[:20]}
out = {"definitions": __doc__, "split_bounds_from_records": {k: {"first": v[0], "last": v[1], "n": v[2]} for k, v in bounds.items()},
       "randomized_splits_json": json.loads(SPL.read_text()), "pool_total": len(pool), "candidates": cands,
       "episodes": {str(e): eps[e] for c in cands.values() for e in c["ranked"]}}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
print(out["split_bounds_from_records"], out["pool_total"])
for k, v in cands.items():
    print(k, v["pool_size"], [(e, round(eps[e]["score_m"], 3)) for e in v["ranked"][:10]])
