"""Summarise off2/f3_replay.json. Groups (coordinator note, T30):
  pre_grasp_nonfrozen : E3 -> T28 analysis phase 'pre_grasp' and not frozen; T14/T17 -> before the first planning row with GT apple
                        dz >= 50 mm (and before the freeze chunk), not frozen
  frozen              : frozen (E3: recorded exclusion.grasp_active; T14/T17: from the grasp-attempt chunk)
  after_switch        : any chunk (either of the above) after the first T26 'switch' decision of that run, per rule
Counts per rule (new = T30 defaults, old = T26 behaviour, E3 only). 'manip_not_apple' = manipulated centroid's nearest object
centre != apple (or no manipulated)."""
import json, collections
import numpy as np

R = json.load(open("f3_replay.json"))


def stats(v):
    v = [x for x in v if x is not None]
    return None if not v else {"n": len(v), "sum": int(sum(v)), "n_gt0": sum(1 for x in v if x > 0), "median": float(np.median(v)), "max": int(max(v))}


def group_rows(label, rows):
    if label.startswith("E3"):
        pre = lambda e: e["phase"] == "pre_grasp"
    else:
        z0 = rows[0]["apple_z"]
        first = next((i for i, e in enumerate(rows) if e["apple_z"] - z0 >= 0.05), len(rows))
        pre = lambda e, first=first: rows.index(e) < first
    return {"pre_grasp_nonfrozen": [e for e in rows if pre(e) and not e["frozen"]], "frozen": [e for e in rows if e["frozen"]]}


def count(rows, rule):
    V = [e[rule] for e in rows if rule in e]
    out = {"n": len(V),
           "n_ball": sum(1 for v in V if v["ball"]),
           "n_no_ball": sum(1 for v in V if not v["ball"]),
           "n_apple_centre_inside_ball": sum(1 for v in V if v["ball"] and v["ball"]["centre_inside"]),
           "n_whole_apple_inside_ball": sum(1 for v in V if v["ball"] and v["ball"]["whole_inside"]),
           "n_neighbour_centre_inside_ball": sum(1 for v in V if v["ball"] and v["ball"]["neighbour_centres_inside"]),
           "n_manip_not_apple": sum(1 for v in V if v.get("nearest_centre") != "apple"),
           "n_association_rejected": sum(1 for v in V if v.get("reason") == "association_rejected"),
           "n_subset_kept_anchor": sum(1 for v in V if v.get("reason") == "subset_kept_anchor"),
           "gate_problem_counts": dict(collections.Counter(str(v.get("gate_problem")) for v in V if v.get("gate_problem"))),
           "ball_radius_mm": stats([None if not v["ball"] else round(v["ball"]["radius_mm"]) for v in V]),
           "apple_centre_to_ball_centre_mm": stats([None if not v["ball"] else round(v["ball"]["apple_centre_dist_mm"]) for v in V])}
    occ = [v for v in V if "occ_fine_near_apple_outside_ball" in v]
    if occ:
        out["uncovered_chunks_occupancy"] = {
            "definition": "chunks with whole apple NOT inside the ball: recorded fine-layer voxels (5 mm) with d <= 0 within 34+15 mm of the GT apple centre; 'outside_ball' = those the target-free layer keeps",
            "n_chunks": len(occ),
            "occupied_near_apple": stats([v["occ_fine_near_apple"] for v in occ]),
            "occupied_near_apple_outside_ball": stats([v["occ_fine_near_apple_outside_ball"] for v in occ]),
            "n_chunks_with_ball_and_outside_gt0": sum(1 for v in occ if v["ball"] and v["occ_fine_near_apple_outside_ball"] > 0),
            "n_chunks_without_ball": sum(1 for v in occ if not v["ball"])}
    return out


res = {"definitions": __doc__, "runs": {}, "pooled": {}}
pool = collections.defaultdict(list)
for label, rows in R["runs"].items():
    g = group_rows(label, rows)
    rules = ["new", "old"] if label.startswith("E3") else ["new"]
    per = {}
    for gn, gr in g.items():
        per[gn] = {rule: count(gr, rule) for rule in rules}
        pool[("E3" if label.startswith("E3") else label, gn)] += gr
    for rule in rules:
        sw = [e for e in rows if e[rule]["after_switch"]]
        per.setdefault("after_switch", {})[rule] = count(sw, rule)
        pool[("E3" if label.startswith("E3") else label, "after_switch_" + rule)] += sw
    if label.startswith("E3"):
        fid = [e["old"] for e in rows if "id_state_match_record" in e["old"]]
        per["fidelity_old_rule_vs_record"] = {"n": len(fid), "id_state_match": sum(1 for v in fid if v["id_state_match_record"]),
                                              "centroid_diff_mm_max": max(v["centroid_vs_record_mm"] for v in fid)}
        val = [e["validation_recorded"] for e in rows if "validation_recorded" in e]
        per["validation_target_free_equals_fine_minus_ball"] = {
            "n": len(val), "target_free_occ_near_apple": stats([v["target_free_occ_near_apple"] for v in val]),
            "fine_occ_outside_recorded_ball_or_below_zmin": stats([v["fine_occ_outside_recorded_ball_or_below_zmin"] for v in val]),
            "n_equal": sum(1 for v in val if v["target_free_occ_near_apple"] == v["fine_occ_outside_recorded_ball_or_below_zmin"]),
            "abs_diff": stats([abs(v["target_free_occ_near_apple"] - v["fine_occ_outside_recorded_ball_or_below_zmin"]) for v in val])}
    res["runs"][label] = per
for (who, gn), gr in pool.items():
    if gn.startswith("after_switch_"):
        rule = gn.split("_")[-1]
        res["pooled"].setdefault(who, {})["after_switch"] = res["pooled"].get(who, {}).get("after_switch", {})
        res["pooled"][who]["after_switch"][rule] = count(gr, rule)
    else:
        rules = ["new", "old"] if who == "E3" else ["new"]
        res["pooled"].setdefault(who, {})[gn] = {rule: count(gr, rule) for rule in rules}
E3 = [(l, r) for l, r in res["runs"].items() if l.startswith("E3")]
res["pooled"]["E3"]["fidelity_old_rule_vs_record"] = {"n": sum(r["fidelity_old_rule_vs_record"]["n"] for _, r in E3),
                                                     "id_state_match": sum(r["fidelity_old_rule_vs_record"]["id_state_match"] for _, r in E3),
                                                     "centroid_diff_mm_max": max(r["fidelity_old_rule_vs_record"]["centroid_diff_mm_max"] for _, r in E3)}
res["pooled"]["E3"]["validation_n_equal"] = [sum(r["validation_target_free_equals_fine_minus_ball"]["n_equal"] for _, r in E3),
                                             sum(r["validation_target_free_equals_fine_minus_ball"]["n"] for _, r in E3)]
json.dump(res, open("f3_summary.json", "w"), indent=1)
print(json.dumps(res["pooled"], indent=1))
