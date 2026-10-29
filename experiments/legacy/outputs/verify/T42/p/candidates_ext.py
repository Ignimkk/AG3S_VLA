"""T42-1 candidate extension (verifier, 2026-10-05, after the user instruction 'any episode index; robust grasp').
Pool: train + validation episodes (0-1799) of meta/randomized_episodes.jsonl, place_one, record success true, record
grasp_retries == 0 and grasp_attempts == 1 (a clean demo grasp). Crowding score as candidates.py (score_m = min(nn_target,
gap_target_crate, gap_crate_others)); per (fruit, arm) in EXT the best N by score (ties by index).
EXT chosen from the first robust-grasp evaluation of the test-split screening (grasp_robust.py): banana left / right (no test
episode passed), orange right (none passed), plus apple + left T39 episode 1939 (E0 and E3b success on T39 rep 1-2, both E3b
robust) which needs rep 3 only.
usage: candidates_ext.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p'))


class C:  # helpers copied from candidates.py (that module runs its main code on import)
    DS = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/meta/randomized_episodes.jsonl'))

    @staticmethod
    def local(p, crate):
        w, x, y, z = crate[3:7]
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        return R.T @ (np.asarray(p[:3]) - np.asarray(crate[:3]))

    @staticmethod
    def rect_dist(l):
        return float(np.linalg.norm(np.maximum(np.abs(l[:2]) - np.array([0.090, 0.150]), 0.0)))

EXT = {("banana", "left"): 7, ("banana", "right"): 7, ("orange", "right"): 6, ("orange", "left"): 4, ("pear", "left"): 4, ("pear", "right"): 4}   # banana 4 -> 7 after 0/8 robust in x1; orange / pear added 10:35 UTC (no strict 3/3)
recs = [json.loads(l) for l in C.DS.read_text().splitlines() if l.strip()]
pool = [r for r in recs if r["episode_index"] < 1800 and r.get("split") in ("train", "validation") and r["task_type"] == "place_one"
        and r.get("success") and r.get("grasp_retries") == 0 and r.get("grasp_attempts") == 1]
eps = {}
for r in pool:
    e = r["episode_index"]; tf = r["target_fruit"]; crate = r["basket_pose"]; fp = r["initial_fruit_poses"]
    others = [k for k in fp if k != tf]; pt = np.asarray(fp[tf][:2])
    nn = min(float(np.linalg.norm(pt - np.asarray(fp[k][:2]))) for k in others)
    gtc = C.rect_dist(C.local(fp[tf], crate)); gco = min(C.rect_dist(C.local(fp[k], crate)) for k in others)
    eps[e] = {"split": r["split"], "target_fruit": tf, "used_arm": r["used_arm"], "layout_index": r["layout_index"], "nn_target_m": nn,
              "gap_target_crate_m": gtc, "gap_crate_others_m": gco, "score_m": min(nn, gtc, gco)}
cands = {}
for (f, a), n in EXT.items():
    ranked = sorted([e for e, v in eps.items() if v["target_fruit"] == f and v["used_arm"] == a], key=lambda e: (-eps[e]["score_m"], e))
    cands[f"{f}_{a}"] = {"pool_size": len(ranked), "ranked": [[e, round(eps[e]["score_m"], 4), eps[e]["split"]] for e in ranked[:12]],
                         "selected": ranked[:n]}
cands["apple_left"] = {"pool_size": None, "selected": [1939], "note": "T39 episode (test split), rep 3 only"}
out = {"definitions": __doc__, "candidates": cands, "episodes": {str(k): v for k, v in sorted(eps.items()) if any(k in c["selected"] for c in cands.values())}}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
for k, v in cands.items():
    print(k, v["pool_size"], v["selected"], v.get("ranked", [])[:5])
