#!/usr/bin/env python3
"""(2026-11-13 sdm-f6, SL-SOTA19 offline check) Per-frame grounding replay on the frozen version's pre-grasp chunks (no identity
memory) under four rules, scored by the GT body of the chosen cluster:
  R0 frozen             - the frozen clustering config, admissible = narrowest extent <= 78 mm
  R1 wide split         - R0 + the T31b attention split also on clusters wider than the opening in EVERY direction
                          (re-cluster at the frozen radius only, no attention partition fallback for these)
  R2 hand-sized top     - R0 + a cluster with narrowest > 78 mm and largest <= 156 mm is admissible when it has the top mean
                          attention among admissible clusters (SL-SOTA18b)
  R3 = R1 + R2
Population: every pre-grasp call; 'attention-right' = the GT body with the highest mean attention (6 cm, >= 15 points) is a task
object. Reports, per rule, the share of calls whose pick is a task object, overall and on attention-right calls, and how many
calls switch from a task object to another body (harm) relative to R0.
usage: .venv-ag3s/bin/python grounding_replay2.py OUT.json"""
import ast, glob, json, os, re, sys, collections
import numpy as np
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grounding_replay import clustering_config, goals, objs, eps_of, B, OPEN
from AG3S.ag3s.stages import target_grounding as tg, attention_split as asp
from AG3S.ag3s.types import AttentionPointCloud, PointCloud
HAND = 0.156
_orig_split, _orig_trig = asp.split_by_attention, asp.triggered


def wide_split(points, attention, members, *, max_opening, partition=False, extents=None, **kw):
    pts = np.asarray(points, np.float64)[np.asarray(members, np.int64)]
    ext = np.sort(asp.principal_extents(pts) if extents is None else np.asarray(extents, np.float64))
    if max_opening is not None and ext.size == 3 and ext[0] > max_opening:      # the new region: wide in every direction
        asp.triggered = lambda e, mo: True
        try:
            return _orig_split(points, attention, members, max_opening=max_opening, partition=False, extents=extents, **kw)
        finally:
            asp.triggered = _orig_trig
    return _orig_split(points, attention, members, max_opening=max_opening, partition=partition, extents=extents, **kw)


def pick(res, hand):
    cl = [c for c in (res.clusters or []) if c.principal_extents is not None]
    adm = [c for c in cl if float(np.min(c.principal_extents)) <= OPEN]
    if hand:
        big = [c for c in cl if float(np.min(c.principal_extents)) > OPEN and float(np.max(c.principal_extents)) <= HAND]
        top = max(adm + big, key=lambda c: c.mean_attention) if adm + big else None
        if top is not None and top in big:
            adm = adm + [top]
    return max(adm, key=lambda c: c.target_score) if adm else None


def main():
    import dataclasses
    # the server computes the max opening from the robot model (78 mm, Panda); offline the config must carry it, else the
    # T31b split is off ("max_opening_unknown") for every rule
    cc = dataclasses.replace(clustering_config().clustering, gripper_max_opening=OPEN); G = goals()
    print("split radius", cc.attention_split_radius, "ratio source", cc.attention_split_ratio_source, "partition", cc.attention_split_partition, "max opening", cc.gripper_max_opening)
    tot = {r: collections.Counter() for r in ("R0", "R1", "R2", "R3")}
    harm = collections.Counter(); gain = collections.Counter(); per_cell = collections.defaultdict(lambda: collections.Counter())
    for pat in ("2*_ms-pkgFrssdm-*", "2*_ms-tab10sdm-*-i19-*", "2*_sub-pkgFrssdm-e1a-spatial-LI-t*"):
        for D in sorted(glob.glob(B + pat)):
            if not os.path.isfile(D + "/episodes.jsonl") or os.path.getsize(D + "/episodes.jsonl") == 0:
                continue
            m = re.search(r"_ms-[^-]+-(\w+)-(LI+)-i", D)
            suite, lvl = (m.group(1), m.group(2)) if m else ("spatial", "LI")
            E = [json.loads(l) for l in open(D + "/episodes.jsonl") if l.strip()]
            chunks = eps_of(D)
            if len(chunks) != len(E):
                continue
            for e, files in zip(E, chunks):
                t, i = int(e["task_id"]), int(e["init_state_index"]); goal = G.get((suite, t))
                if goal is None:
                    continue
                cf = glob.glob(D + f"/episodes/*_t{t}_e{i}_calls.jsonl")[0]; cs = [json.loads(x) for x in open(cf)]
                g = np.load(cf.replace("_calls.jsonl", "_geom.npz"), allow_pickle=True); O = objs(g); oi = [int(x) for x in g["obs_index"]]
                T = np.asarray(g["T_world_base"], np.float64).reshape(-1, 4, 4)
                for f in files:
                    z = np.load(f, allow_pickle=True); sq = int(z["t_step"]); sm = json.loads(str(z["summary_json"]))
                    if ((sm.get("grasp") or {}).get("state")) in ("held", "closing"):
                        break
                    if sq >= len(cs) or cs[sq]["obs_index"] not in oi:
                        continue
                    k = oi.index(cs[sq]["obs_index"]); Tinv = np.linalg.inv(T[min(k, len(T) - 1)])
                    gt = {n: (Tinv @ np.r_[p[k], 1.0])[:3] for n, p in O.items() if p[k, 2] < 2 and np.linalg.norm((Tinv @ np.r_[p[k], 1.0])[:3]) < 1.0}
                    P = np.asarray(z["attention_cloud"], np.float32); A = np.asarray(z["attention_values"], np.float32)
                    AR = np.asarray(z["attention_raw"], np.float32); sup = np.asarray(z["support_mask"], bool)
                    owner = lambda pts: min(gt, key=lambda n: np.linalg.norm(gt[n] - pts.mean(0)))
                    means = {}
                    for n, q in gt.items():
                        msk = (np.linalg.norm(P - q, axis=1) < 0.06) & ~sup
                        if msk.sum() >= 15:
                            means[n] = float(A[msk].mean())
                    att_right = bool(means) and max(means, key=means.get) in goal
                    cloud = AttentionPointCloud(cloud=PointCloud(points=P), attention=A, raw_attention=AR)
                    res0 = tg.ground_target(cloud, cc, exclude_mask=sup)
                    tg.split_by_attention = wide_split
                    try:
                        res1 = tg.ground_target(cloud, cc, exclude_mask=sup)
                    finally:
                        tg.split_by_attention = _orig_split
                    out = {}
                    for r, res, hand in (("R0", res0, False), ("R1", res1, False), ("R2", res0, True), ("R3", res1, True)):
                        c = pick(res, hand)
                        v = "none" if c is None else ("task" if owner(P[np.asarray(c.point_indices)]) in goal else "other")
                        out[r] = v; tot[r][v] += 1; tot[r][("att_right_" if att_right else "att_wrong_") + v] += 1
                    for r in ("R1", "R2", "R3"):
                        if out["R0"] == "task" and out[r] != "task":
                            harm[r] += 1
                        if out["R0"] != "task" and out[r] == "task":
                            gain[r] += 1; per_cell[r][f"{suite} {lvl}"] += 1
            print(os.path.basename(D)[16:70], flush=True)
    rep = {}
    for r, v in tot.items():
        n = v["task"] + v["other"] + v["none"]; nr = v["att_right_task"] + v["att_right_other"] + v["att_right_none"]
        rep[r] = {"task_share_all": round(v["task"] / n, 3), "task_share_attention_right": round(v["att_right_task"] / max(nr, 1), 3),
                  "none": v["none"], "gain_vs_R0": gain.get(r, 0), "harm_vs_R0": harm.get(r, 0), "gain_by_cell": dict(per_cell.get(r, {}))}
        print(f"{r}: task object {v['task']}/{n} = {100 * v['task'] / n:.1f} % | on attention-right calls {100 * v['att_right_task'] / max(nr, 1):.1f} % "
              f"| none {v['none']} | gain {gain.get(r, 0)} harm {harm.get(r, 0)} | gains by cell {dict(per_cell.get(r, {}))}")
    json.dump(rep, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
