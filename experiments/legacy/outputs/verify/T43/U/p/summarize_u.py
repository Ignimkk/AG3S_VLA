"""T43 U — aggregate raw/runs/*.json (measure_u.py) into raw/summary_u.json. Definitions (on top of measure_u's):

  physical obstacle  = run has an obstacle profile (.start obstacle=...). T42-2 'invis' = obstacle present, invisible to cameras.
  first attach       = first chunk with grasp.attach true; 'before attach' = chunks before it (all chunks if none).
  switch             = chunk whose manipulated id != the first non-null manipulated id of the run; switch_target = GT nearest
                       surface of that chunk's manipulated centroid. 'before close' = chunk plan t < first close_t.
  close chunks       = the planning chunk whose 8 rows contain the first close_t (k0) and the next one (k0 + 1).
  fruit in field     = at a close chunk: the hand rows' target-free layer still holds the fruit, i.e. carved is False
                       (manipulated is not the fruit) or remnant_frac_free >= 0.25 (>= 25 % of the fruit's upper vertices that are
                       > 25 mm from every other GT surface read <= 10 mm in the target-free layer).
  a on fruit rows    = at a close chunk: >= 1 hand row whose GT nearest surface is the fruit is classed 'obstacle' by
                       _obstacle_rows while obstacle_margin > 0 (gets max(margin, 10 mm)).
  defl at close      = max over the close chunks of deflection.defl_mm; radial at that chunk.
  outcome (refined)  = success | miss | slip_at_pick (fell below z0 + 20 mm within 64 rows of lift) | drop_carry (fell later) |
                       lifted_not_placed (lifted, never fell, not in the crate at the end) | no_close.
  failure            = miss or slip_at_pick (pick failures in the user's sense); drop_carry / lifted_not_placed reported apart.
usage: summarize_u.py"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, statistics as stt
from collections import Counter, defaultdict

U = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/U'))
RU = json.loads((U / "raw/runs_u.json").read_text())


def med(v):
    v = [x for x in v if x is not None]
    return None if not v else float(stt.median(v))


def run_summary(r):
    o = {k: r.get(k) for k in ("key", "campaign", "cond", "stage", "episode", "seed", "arm", "fruit", "outcome", "close_ts",
                               "n_close_attempts", "lift_t", "slip_t", "slip_kind", "fruit_dz_max_mm", "obstacle_margin_mm",
                               "path_check", "gripper_cover", "held_body_cover")}
    if o["outcome"] == "slip" and r.get("slip_t") is None:
        o["outcome"] = "lifted_not_placed"   # lifted >= 50 mm, never fell back below z0 + 20 mm, not in the crate at the end
    elif o["outcome"] == "slip":
        o["outcome"] = "slip_at_pick" if r.get("slip_kind") == "at_pick" else "drop_carry"
    o["obstacle"] = r.get("obstacle") is not None
    o["invis"] = bool(r.get("obstacle") and "invis" in r["obstacle"][1])
    c0 = (r.get("contact") or [None])[0]
    o["contact"] = None if c0 is None else {k: c0[k] for k in c0 if k != "rows"}
    if "timeline" not in r:
        return o
    tl = r["timeline"]; rows = {d["seq"]: d for d in r.get("rows", [])}
    att = next((e for e in tl if e["attach"]), None)
    o["attach_t"] = None if att is None else att["t"]
    first_id = next((e["man_id"] for e in tl if e["man_id"] is not None), None)
    ct = (r.get("close_ts") or [None])[0]
    sw = [e for e in tl if e["man_id"] is not None and e["man_id"] != first_id and (att is None or e["seq"] < att["seq"])]
    o["first_man_id"] = first_id
    o["switch_before_attach"] = bool(sw)
    o["switch_first"] = None if not sw else {"seq": sw[0]["seq"], "t": sw[0]["t"], "to_id": sw[0]["man_id"], "state": sw[0]["grasp_state"],
                                             "target": (sw[0]["man_centroid"] or {}).get("nearest"),
                                             "dist_mm": (sw[0]["man_centroid"] or {}).get("dist_mm"),
                                             "cam_max": {k: v["max"] for k, v in sw[0]["attention"].items()},
                                             "cam_cell": {k: [v["row"], v["col"]] for k, v in sw[0]["attention"].items()},
                                             "peak_nearest": (sw[0]["attention_peak"] or {}).get("nearest")}
    o["switch_targets"] = dict(Counter((e["man_centroid"] or {}).get("nearest") for e in sw))
    swa = [e for e in sw if (e["man_centroid"] or {}).get("nearest") not in ("fruit", None)]
    o["switch_away_before_attach"] = bool(swa)
    o["switch_away_before_close"] = bool(ct is not None and any(e["t"] < ct for e in swa))
    o["switch_away_first"] = None if not swa else {"seq": swa[0]["seq"], "t": swa[0]["t"], "state": swa[0]["grasp_state"],
                                                   "target": swa[0]["man_centroid"]["nearest"],
                                                   "cam_max": {k: v["max"] for k, v in swa[0]["attention"].items()},
                                                   "cam_cell": {k: [v["row"], v["col"]] for k, v in swa[0]["attention"].items()},
                                                   "peak_nearest": (swa[0]["attention_peak"] or {}).get("nearest"),
                                                   "after_first_close": bool(ct is not None and swa[0]["t"] >= ct)}
    o["switch_before_close"] = bool(ct is not None and any(e["t"] < ct for e in sw))
    o["switch_in_latched_closing"] = sum(1 for e in sw if e["grasp_state"] in ("latched", "closing"))
    o["n_chunks_man_not_fruit_before_attach"] = sum(1 for e in tl if (att is None or e["seq"] < att["seq"]) and e["man_centroid"]
                                                     and e["man_centroid"]["nearest"] != "fruit")
    o["states"] = dict(Counter(e["grasp_state"] for e in tl))
    # close chunks
    if ct is not None:
        k0 = next((e["seq"] for e in tl if e["t"] <= ct < e["t"] + 8), None)
        cc = [rows[s] for s in (k0, None if k0 is None else k0 + 1) if s in rows and "error" not in rows[s]]
        o["close_chunks"] = [d["seq"] for d in cc]
        if cc:
            def infield(d):
                c = d["carve"]
                return (c.get("carved") is False) or ((c.get("remnant_frac_free") or 0) >= 0.25)
            o["fruit_in_field_at_close"] = any(infield(d) for d in cc)
            o["carve_at_close"] = [{k: d["carve"].get(k) for k in ("carved", "remnant_frac_free", "d_free_up_median_mm", "ball_radius_mm",
                                                                     "ball_centre_to_fruit_mm")} for d in cc]
            o["man_at_close"] = [next((e["man_id"], (e["man_centroid"] or {}).get("nearest")) for e in tl if e["seq"] == d["seq"]) for d in cc]
            o["a_on_fruit_rows_at_close"] = any(d["fruit_rows_obstacle_margin_applied"] > 0 for d in cc)
            o["fruit_rows_at_close"] = [d["fruit_rows"] for d in cc]
            o["fruit_rows_class_at_close"] = [d["fruit_rows_class"] for d in cc]
            o["fruit_rows_min_cl_at_close_mm"] = min((d["fruit_rows_min_cl_mm"] for d in cc if d["fruit_rows_min_cl_mm"] is not None), default=None)
            best = max(cc, key=lambda d: (d["deflection"] or {}).get("defl_mm", -1))
            o["defl_at_close_mm"] = (best["deflection"] or {}).get("defl_mm"); o["radial_at_close_mm"] = (best["deflection"] or {}).get("radial_mm")
            o["worst_hand_row_at_close"] = [d["worst_hand_row"] for d in cc]
            o["fid_at_close"] = [d["fid"] for d in cc]
    # closing phase aggregates (all pre-attach closing chunks)
    cl_all = [d for d in r.get("rows", []) if "error" not in d and d["grasp_state"] == "closing"]
    for nm, cl in (("closing", [d for d in cl_all if (d.get("tip_now_to_fruit_mm") or 1e9) <= 60.0]), ("closing_all", cl_all)):
        o[nm] = closing_agg(r, tl, cl, cl_all, nm)
    o["fid_all"] = {"n": len([d for d in r.get("rows", []) if "error" not in d]),
                    "same_row": sum(1 for d in r.get("rows", []) if "error" not in d and d["fid"].get("same_row")),
                    "max_centre0_mm": max((d["fid"]["centre0_mm"] for d in r.get("rows", []) if "error" not in d), default=None),
                    "max_record_clearance_mm": max((d["fid"].get("record_clearance_mm") or 0 for d in r.get("rows", []) if "error" not in d), default=None),
                    "n_errors": sum(1 for d in r.get("rows", []) if "error" in d)}
    return o


def closing_agg(r, tl, cl, cl_all, nm):
    if True:
        near_seq = {d["seq"] for d in cl}
    tlc = [e for e in tl if e["grasp_state"] == "closing" and not e["attached"] and e["seq"] in near_seq]
    return {
        "definition": "grasp state closing, not attached" + (", fingertip midpoint within 60 mm of the fruit centre at the planning qpos" if nm == "closing" else ""),
        "n_chunks_all_closing": len(cl_all),
        "n_chunks": len(cl),
        "n_man_not_fruit": sum(1 for e in tlc if e["man_centroid"] and e["man_centroid"]["nearest"] != "fruit"),
        "n_not_carved": sum(1 for d in cl if d["carve"].get("carved") is False),
        "n_remnant_ge25": sum(1 for d in cl if (d["carve"].get("remnant_frac_free") or 0) >= 0.25),
        "n_fruit_rows_obstacle_class_chunks": sum(1 for d in cl if d["fruit_rows_obstacle_margin_applied"] > 0),
        "fruit_rows_obstacle_class_total": sum(d["fruit_rows_obstacle_margin_applied"] for d in cl),
        "defl_median_mm": med([(d["deflection"] or {}).get("defl_mm") for d in cl]),
        "radial_median_mm": med([(d["deflection"] or {}).get("radial_mm") for d in cl]),
        "worst_hand_gt": dict(Counter(d["worst_hand_row"]["gt_nearest"] for d in cl if d["worst_hand_row"]["cl_mm"] <= 30)),
        "binding": dict(Counter(f'{e["to"]["binding"]}:{(e["to"][e["to"]["binding"]] or {}).get("cat")}' for e in tlc)),
        "binding_hand_link_point_nearest": dict(Counter(
            (e["to"][e["to"]["binding"]] or {}).get("point_nearest_nohand") for e in tlc
            if (e["to"][e["to"]["binding"]] or {}).get("link") in {f"ee_finger_{'l' if r['arm'] == 'left' else 'r'}1",
                                                                    f"ee_finger_{'l' if r['arm'] == 'left' else 'r'}2", f"ee_{r['arm']}"})),
        "binding_clearance_median_mm": med([(e["to"][e["to"]["binding"]] or {}).get("clearance_mm") for e in tlc]),
        "path_target_rows_excluded_total": sum(((e["to"].get("path") or {}).get("target_rows_excluded") or 0) for e in tlc),
        "n_hold": sum(1 for e in tlc if e["gate"] == "hold"),
        "tip_now_to_fruit_median_mm": med([d.get("tip_now_to_fruit_mm") for d in cl]),
        "fid_same_row": sum(1 for d in cl if d["fid"].get("same_row")),
        "fid_max_record_clearance_mm": max((d["fid"].get("record_clearance_mm") or 0 for d in cl), default=None),
    }


def xtab(runs, rowkey, label):
    t = defaultdict(Counter)
    for o in runs:
        t[str(o.get(rowkey))][o["outcome"]] += 1
    return {"definition": label, "table": {k: dict(v) for k, v in t.items()}}


def main():
    S = {}
    for f in sorted((U / "raw/runs").glob("*.json")):
        r = json.loads(f.read_text())
        if "error" in r:
            S[r.get("key", f.stem)] = {"error": r["error"]}; continue
        S[r["key"]] = run_summary(r)
    # T41 c runs repeat T41 a bit-for-bit through first close + 64 rows in 14 / 15 runs (raw/t41_a_vs_c.json) -> headline counts drop T41 c
    srv_all = [o for o in S.values() if "error" not in o and o.get("cond") != "E0"]
    srv = [o for o in srv_all if o["campaign"] != "T41/c"]
    obs = [o for o in srv if o["obstacle"]]; clr = [o for o in srv if not o["obstacle"]]
    out = {"doc": __doc__, "headline_excludes": "T41/c (duplicates T41/a through the pick)", "n_runs": len(S), "errors": {k: v for k, v in S.items() if "error" in v}}
    out["outcomes"] = {"server_obstacle": dict(Counter(o["outcome"] for o in obs)), "server_clear": dict(Counter(o["outcome"] for o in clr)),
                       "E0_obstacle": dict(Counter(o["outcome"] for o in S.values() if o.get("cond") == "E0" and o["obstacle"])),
                       "E0_clear": dict(Counter(o["outcome"] for o in S.values() if o.get("cond") == "E0" and not o["obstacle"]))}
    out["switch_vs_outcome"] = {"obstacle": xtab(obs, "switch_before_attach", "switch before attach x outcome"),
                                "clear": xtab(clr, "switch_before_attach", "switch before attach x outcome")}
    out["switch_away_vs_outcome"] = {"obstacle": xtab(obs, "switch_away_before_attach", "switch to a non-fruit surface before attach x outcome"),
                                     "clear": xtab(clr, "switch_away_before_attach", "")}
    out["switch_away_before_close_vs_outcome"] = {"obstacle": xtab(obs, "switch_away_before_close", "switch to non-fruit before first close x outcome"),
                                                  "clear": xtab(clr, "switch_away_before_close", "")}
    out["switch_before_close_vs_outcome"] = {"obstacle": xtab(obs, "switch_before_close", "switch before first close x outcome"),
                                             "clear": xtab(clr, "switch_before_close", "")}
    out["infield_vs_outcome"] = {"obstacle": xtab(obs, "fruit_in_field_at_close", "fruit in target-free layer at close x outcome"),
                                 "clear": xtab(clr, "fruit_in_field_at_close", "")}
    out["a_fruit_rows_vs_outcome"] = {"obstacle": xtab([o for o in obs if (o.get("obstacle_margin_mm") or 0) > 0], "a_on_fruit_rows_at_close", "a on"),
                                      "clear": xtab([o for o in clr if (o.get("obstacle_margin_mm") or 0) > 0], "a_on_fruit_rows_at_close", "a on")}
    def fail(o):
        return {"miss": "fail", "slip_at_pick": "fail", "success": "success"}.get(o["outcome"], o["outcome"])
    def tab2(L, key):
        t = defaultdict(Counter)
        for o in L:
            t[str(bool(o.get(key)))][fail(o)] += 1
        return {k: dict(v) for k, v in t.items()}
    withc_o = [o for o in obs if o.get("close_ts")]; withc_c = [o for o in clr if o.get("close_ts")]
    out["headline"] = {
        "definition": "runs with >= 1 close command; fail = miss or slip_at_pick; other = lifted_not_placed / drop_carry",
        "obstacle_n": len(withc_o), "clear_n": len(withc_c),
        "switch_away_before_attach": {"obstacle": tab2(withc_o, "switch_away_before_attach"), "clear": tab2(withc_c, "switch_away_before_attach")},
        "switch_away_before_close": {"obstacle": tab2(withc_o, "switch_away_before_close"), "clear": tab2(withc_c, "switch_away_before_close")},
        "fruit_in_field_at_close": {"obstacle": tab2(withc_o, "fruit_in_field_at_close"), "clear": tab2(withc_c, "fruit_in_field_at_close")},
        "a_on_fruit_rows_at_close": {"obstacle": tab2([o for o in withc_o if (o.get("obstacle_margin_mm") or 0) > 0], "a_on_fruit_rows_at_close"),
                                     "clear": tab2([o for o in withc_c if (o.get("obstacle_margin_mm") or 0) > 0], "a_on_fruit_rows_at_close")},
        "switch_away_first_after_first_close": sum(1 for o in obs if o.get("switch_away_first") and o["switch_away_first"]["after_first_close"]),
        "switch_away_first_targets": dict(Counter(o["switch_away_first"]["target"] for o in obs if o.get("switch_away_first"))),
        "switch_away_first_dominant_camera": dict(Counter(max(o["switch_away_first"]["cam_max"], key=o["switch_away_first"]["cam_max"].get)
                                                          for o in obs if o.get("switch_away_first"))),
        "switch_away_first_left_wrist_row": [o["switch_away_first"]["cam_cell"].get("left_wrist", [None])[0] for o in obs if o.get("switch_away_first")],
        "switch_away_first_peak_nearest": dict(Counter(o["switch_away_first"]["peak_nearest"] for o in obs if o.get("switch_away_first"))),
        "switch_away_runs": [{"key": o["key"], "outcome": o["outcome"], **o["switch_away_first"], "first_close_t": (o.get("close_ts") or [None])[0]}
                             for o in obs if o.get("switch_away_first")],
        "defl_at_close_mm": {"obstacle_median": med([o.get("defl_at_close_mm") for o in withc_o]), "clear_median": med([o.get("defl_at_close_mm") for o in withc_c]),
                             "obstacle_fail_median": med([o.get("defl_at_close_mm") for o in withc_o if fail(o) == "fail"]),
                             "obstacle_success_median": med([o.get("defl_at_close_mm") for o in withc_o if fail(o) == "success"]),
                             "clear_fail_median": med([o.get("defl_at_close_mm") for o in withc_c if fail(o) == "fail"]),
                             "clear_success_median": med([o.get("defl_at_close_mm") for o in withc_c if fail(o) == "success"])},
        "radial_at_close_mm": {"obstacle_median": med([o.get("radial_at_close_mm") for o in withc_o]), "clear_median": med([o.get("radial_at_close_mm") for o in withc_c]),
                               "obstacle_fail_median": med([o.get("radial_at_close_mm") for o in withc_o if fail(o) == "fail"]),
                               "obstacle_success_median": med([o.get("radial_at_close_mm") for o in withc_o if fail(o) == "success"])},
        "pre_close_contact": {"obstacle": tab2([dict(o, pc=(o.get("contact") or {}).get("pre_close_contact_rows", 0) > 0) for o in withc_o], "pc"),
                              "clear": tab2([dict(o, pc=(o.get("contact") or {}).get("pre_close_contact_rows", 0) > 0) for o in withc_c], "pc")},
        "pre_close_fruit_disp_mm": {"obstacle_fail_median": med([(o.get("contact") or {}).get("pre_close_fruit_disp_mm") for o in withc_o if fail(o) == "fail"]),
                                    "obstacle_success_median": med([(o.get("contact") or {}).get("pre_close_fruit_disp_mm") for o in withc_o if fail(o) == "success"]),
                                    "clear_median": med([(o.get("contact") or {}).get("pre_close_fruit_disp_mm") for o in withc_c])},
    }
    out["switch_targets_obstacle"] = dict(sum((Counter(o.get("switch_targets") or {}) for o in obs), Counter()))
    out["switch_targets_clear"] = dict(sum((Counter(o.get("switch_targets") or {}) for o in clr), Counter()))
    # per group
    grp = defaultdict(list)
    for o in srv_all:
        g = f'{o["campaign"]}/{o["cond"] if o["campaign"] == "T43/V" else "E3b"}/{"obst" if o["obstacle"] else "clear"}{"-invis" if o["invis"] else ""}'
        grp[g].append(o)
    G = {}
    for g, L in sorted(grp.items()):
        withc = [o for o in L if o.get("close_ts")]
        G[g] = {"n": len(L), "outcomes": dict(Counter(o["outcome"] for o in L)),
                "switch_before_attach": sum(o.get("switch_before_attach", False) for o in L),
                "switch_before_close": sum(o.get("switch_before_close", False) for o in L),
                "switch_away_before_attach": sum(o.get("switch_away_before_attach", False) for o in L),
                "switch_away_before_close": sum(o.get("switch_away_before_close", False) for o in L),
                "closing_chunks_all": sum(o["closing"]["n_chunks_all_closing"] for o in L if "closing" in o),
                "n_with_close": len(withc),
                "fruit_in_field_at_close": sum(bool(o.get("fruit_in_field_at_close")) for o in withc),
                "a_on_fruit_rows_at_close": sum(bool(o.get("a_on_fruit_rows_at_close")) for o in withc),
                "defl_at_close_median_mm": med([o.get("defl_at_close_mm") for o in withc]),
                "radial_at_close_median_mm": med([o.get("radial_at_close_mm") for o in withc]),
                "pre_close_contact_runs": sum(1 for o in withc if (o.get("contact") or {}).get("pre_close_contact_rows", 0) > 0),
                "pre_close_fruit_disp_median_mm": med([(o.get("contact") or {}).get("pre_close_fruit_disp_mm") for o in withc]),
                "closing_chunks": sum(o["closing"]["n_chunks"] for o in L if "closing" in o),
                "closing_man_not_fruit": sum(o["closing"]["n_man_not_fruit"] for o in L if "closing" in o),
                "closing_not_carved": sum(o["closing"]["n_not_carved"] for o in L if "closing" in o),
                "closing_remnant_ge25": sum(o["closing"]["n_remnant_ge25"] for o in L if "closing" in o),
                "closing_fruit_rows_obstacle_class_chunks": sum(o["closing"]["n_fruit_rows_obstacle_class_chunks"] for o in L if "closing" in o),
                "closing_binding": dict(sum((Counter(o["closing"]["binding"]) for o in L if "closing" in o), Counter())),
                "closing_binding_hand_point_nearest": dict(sum((Counter({str(k): v for k, v in o["closing"]["binding_hand_link_point_nearest"].items()}) for o in L if "closing" in o), Counter())),
                "closing_binding_clearance_median_mm": med([o["closing"]["binding_clearance_median_mm"] for o in L if "closing" in o]),
                "closing_defl_median_mm": med([o["closing"]["defl_median_mm"] for o in L if "closing" in o]),
                "closing_radial_median_mm": med([o["closing"]["radial_median_mm"] for o in L if "closing" in o]),
                "closing_hold": sum(o["closing"]["n_hold"] for o in L if "closing" in o),
                "closing_path_target_rows_excluded": sum(o["closing"]["path_target_rows_excluded_total"] for o in L if "closing" in o),
                "closing_tip_to_fruit_median_mm": med([o["closing"]["tip_now_to_fruit_median_mm"] for o in L if "closing" in o]),
                "closing_all": {k: (dict(sum((Counter({str(a): b for a, b in o["closing_all"][k].items()}) for o in L if "closing_all" in o), Counter()))
                                    if k in ("binding", "binding_hand_link_point_nearest", "worst_hand_gt")
                                    else sum(o["closing_all"][k] for o in L if "closing_all" in o))
                                for k in ("n_chunks", "n_man_not_fruit", "n_not_carved", "n_remnant_ge25", "n_fruit_rows_obstacle_class_chunks",
                                          "binding", "binding_hand_link_point_nearest", "worst_hand_gt", "n_hold", "path_target_rows_excluded_total")},
                "fid_rows": sum(o["fid_all"]["n"] for o in L), "fid_same_row": sum(o["fid_all"]["same_row"] for o in L),
                "fid_max_centre0_mm": max((o["fid_all"]["max_centre0_mm"] or 0) for o in L),
                "fid_max_record_clearance_mm": max((o["fid_all"]["max_record_clearance_mm"] or 0) for o in L),
                "fid_errors": sum(o["fid_all"]["n_errors"] for o in L)}
    out["groups"] = G
    # pairs obstacle vs clear (same campaign family, episode, seed)
    def fam(o):
        f = {"T42/1": "T42", "T42/2": "T42"}.get(o["campaign"], o["campaign"])
        return f + "/" + o["cond"] if f == "T43/V" else f   # T43 V: same arm only (V2 H <-> V2 B)
    clear_idx = {(fam(o), o["episode"], o["seed"]): o for o in clr}
    pairs = []
    for o in obs:
        c = clear_idx.get((fam(o), o["episode"], o["seed"]))
        if c is None:
            continue
        pairs.append({"family": fam(o), "obst": o["key"], "clear": c["key"], "obst_outcome": o["outcome"], "clear_outcome": c["outcome"],
                      "obst_switch": o.get("switch_away_before_attach"), "clear_switch": c.get("switch_away_before_attach"),
                      "obst_infield": o.get("fruit_in_field_at_close"), "clear_infield": c.get("fruit_in_field_at_close"),
                      "obst_defl": o.get("defl_at_close_mm"), "clear_defl": c.get("defl_at_close_mm")})
    out["pairs"] = pairs
    pc_f = defaultdict(list)
    for p_ in pairs:
        pc_f[p_["family"]].append(p_)
    out["pairs_by_family"] = {f: {"n": len(v), "obst_switch": sum(bool(p_["obst_switch"]) for p_ in v), "clear_switch": sum(bool(p_["clear_switch"]) for p_ in v),
                                  "obst_infield": sum(bool(p_["obst_infield"]) for p_ in v), "clear_infield": sum(bool(p_["clear_infield"]) for p_ in v),
                                  "obst_outcomes": dict(Counter(p_["obst_outcome"] for p_ in v)), "clear_outcomes": dict(Counter(p_["clear_outcome"] for p_ in v)),
                                  "obst_defl_median": med([p_["obst_defl"] for p_ in v]), "clear_defl_median": med([p_["clear_defl"] for p_ in v])}
                              for f, v in sorted(pc_f.items())}
    out["pairs_counts"] = {
        "n": len(pairs),
        "obst_switch": sum(bool(p["obst_switch"]) for p in pairs), "clear_switch": sum(bool(p["clear_switch"]) for p in pairs),
        "obst_infield": sum(bool(p["obst_infield"]) for p in pairs), "clear_infield": sum(bool(p["clear_infield"]) for p in pairs),
        "obst_fail": sum(p["obst_outcome"] in ("miss", "slip_at_pick") for p in pairs), "clear_fail": sum(p["clear_outcome"] in ("miss", "slip_at_pick") for p in pairs),
        "obst_defl_median": med([p["obst_defl"] for p in pairs]), "clear_defl_median": med([p["clear_defl"] for p in pairs])}
    out["runs"] = S
    (U / "raw/summary_u.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("outcomes", "switch_vs_outcome", "switch_away_vs_outcome", "switch_away_before_close_vs_outcome", "switch_before_close_vs_outcome", "infield_vs_outcome",
                                          "a_fruit_rows_vs_outcome", "switch_targets_obstacle", "switch_targets_clear", "pairs_counts", "pairs_by_family", "headline")}, indent=1))
    for g, v in G.items():
        print(g, json.dumps(v))


if __name__ == "__main__":
    main()
