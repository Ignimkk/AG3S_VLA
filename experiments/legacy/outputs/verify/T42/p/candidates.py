"""T42-1 candidate list (verifier, 2026-10-05). CPU only, dataset metadata only (no MuJoCo).

Reads meta/randomized_episodes.jsonl, test split 1800-1999, place_one, and per episode computes layout-crowding numbers
(xy plane, initial poses of the record = the client's reset poses before the 1.5 s settle):
  nn_target_m        : min distance target fruit centre -> other fruit centres
  gap_target_crate_m : distance target fruit centre -> crate OUTER rectangle (crate frame, half 0.090 x 0.150)
  gap_crate_others_m : min over non-target fruits of distance centre -> crate outer rectangle
  score_m            : min of the three (larger = less crowded)
Per (fruit, arm) the episodes are ranked by score_m (descending, ties by episode index).
Apple + left: candidates are restricted to the T39 episodes whose E0 and E3b both succeeded on both T39 seeds
(T39.verify.json numbers.matrix); those need only seed 3.
Also writes the crate-interior rule check over the demo records: for every test episode, the record's final target pose in
the frame of the record's basket_pose (initial crate pose) vs the analysis rule (|x| < 0.077, |y| < 0.137,
-0.052 < z < 0.075), and the initial non-target poses (must be outside).
usage: candidates.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np

DS = pathlib.Path(_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/meta/randomized_episodes.jsonl'))
T39 = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T39.verify.json'))
CRATE_HALF_XY = np.array([0.090, 0.150]); HALF_IN = np.array([0.082, 0.142]); MARGIN = 0.005; ZR = (-0.052, 0.075)
N_PER = {("apple", "left"): 3, ("apple", "right"): 4}   # default 3 per (fruit, arm)


def R(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def local(p, crate):
    return R(crate[3:7]).T @ (np.asarray(p[:3]) - np.asarray(crate[:3]))


def rect_dist(l):
    d = np.maximum(np.abs(l[:2]) - CRATE_HALF_XY, 0.0)
    return float(np.linalg.norm(d))


def inside(l):
    return bool(np.all(np.abs(l[:2]) < HALF_IN - MARGIN) and ZR[0] < l[2] < ZR[1])


recs = [json.loads(l) for l in DS.read_text().splitlines() if l.strip()]
test = [r for r in recs if 1800 <= r["episode_index"] <= 1999 and r.get("split") == "test"]
t39 = json.loads(T39.read_text())["numbers"]["matrix"]
both = {}
for m in t39:
    both.setdefault(m["episode"], []).append(m["E0"]["success"] and m["E3b"]["success"])
t39_ok = sorted(e for e, v in both.items() if len(v) == 2 and all(v))

eps = {}
rule = {"target_final_inside": 0, "target_final_outside": [], "nontarget_initial_inside": [], "n": 0,
        "target_final_local_z_by_fruit": {}, "target_final_local_absx_max_by_fruit": {}, "target_final_local_absy_max_by_fruit": {}}
for r in test:
    e = r["episode_index"]; tf = r["target_fruit"]; crate = r["basket_pose"]; fp = r["initial_fruit_poses"]
    others = [k for k in fp if k != tf]
    pt = np.asarray(fp[tf][:2])
    nn = min(float(np.linalg.norm(pt - np.asarray(fp[k][:2]))) for k in others)
    gtc = rect_dist(local(fp[tf], crate)); gco = min(rect_dist(local(fp[k], crate)) for k in others)
    eps[e] = {"target_fruit": tf, "used_arm": r["used_arm"], "task_type": r["task_type"], "layout_index": r["layout_index"],
              "slot_order": r["slot_order"], "nn_target_m": nn, "gap_target_crate_m": gtc, "gap_crate_others_m": gco,
              "score_m": min(nn, gtc, gco), "target_xy": pt.tolist(), "crate_xy": list(crate[:2]),
              "record_success": r.get("success")}
    lf = local(r["final_fruit_poses"][tf], crate)
    rule["n"] += 1
    if inside(lf):
        rule["target_final_inside"] += 1
    else:
        rule["target_final_outside"].append({"episode": e, "fruit": tf, "local": lf.tolist()})
    rule["target_final_local_z_by_fruit"].setdefault(tf, []).append(float(lf[2]))
    rule["target_final_local_absx_max_by_fruit"][tf] = max(rule["target_final_local_absx_max_by_fruit"].get(tf, 0), float(abs(lf[0])))
    rule["target_final_local_absy_max_by_fruit"][tf] = max(rule["target_final_local_absy_max_by_fruit"].get(tf, 0), float(abs(lf[1])))
    for k in others:
        if inside(local(fp[k], crate)):
            rule["nontarget_initial_inside"].append({"episode": e, "fruit": k})
rule["target_final_local_z_by_fruit"] = {k: {"min": min(v), "max": max(v), "n": len(v)} for k, v in rule["target_final_local_z_by_fruit"].items()}

cands = {}
for fruit in ("apple", "orange", "banana", "pear"):
    for arm in ("left", "right"):
        pool = [e for e, v in eps.items() if v["target_fruit"] == fruit and v["used_arm"] == arm and v["task_type"] == "place_one"]
        if (fruit, arm) == ("apple", "left"):
            pool = [e for e in pool if e in t39_ok]
        ranked = sorted(pool, key=lambda e: (-eps[e]["score_m"], e))
        n = N_PER.get((fruit, arm), 3)
        cands[f"{fruit}_{arm}"] = {"pool_size": len(pool), "ranked": [[e, round(eps[e]["score_m"], 4)] for e in ranked],
                                   "selected": ranked[:n]}
out = {"definitions": __doc__, "t39_apple_left_both_both": t39_ok, "candidates": cands, "crate_rule_check_demo_records": rule,
       "episodes": {str(k): v for k, v in sorted(eps.items())}}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
for k, v in cands.items():
    print(k, v["pool_size"], "selected", v["selected"], "top", v["ranked"][:8])
print("rule", {k: v for k, v in rule.items() if k not in ("target_final_outside",)}, "outside", rule["target_final_outside"][:5])
