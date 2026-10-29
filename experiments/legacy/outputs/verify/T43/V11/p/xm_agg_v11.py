"""T43 XM part 1 aggregation (raw/cross_{V5,V7g,V8,V9}.json -> raw/cross_summary.json).

Chunk groups (per run, chunks with in_window = some row of the chunk within 100 mm of the hurdle, HX rule):
  carry_apple : grasp.state != 'placed', t0 >= t_from, apple in hand at the planning row (inhand_at_plan)
  carry_empty : grasp.state != 'placed', t0 >= t_from, apple not in hand at the planning row
  return      : grasp.state == 'placed'
Executed rows of a group: control rows t of those chunks with executed true; D(t) = the chunk's D(k = step_in_chunk).
Chunk-level direction: at k* = argmax_k |D_palm(k)| (palm = EE_BODY origin); 'deflected' chunk = |D_palm(k*)| > 5 mm.
  dominant axis of D(k*): argmax of |dz|, |dn|, |da| -> up/down, away/toward (sign of 'away'), along bar.
V9 classes: in-hand intervals = runs of apple_in_hand rows (gaps < 8 rows merged); drop_t = row after the last interval;
  success (analysis) | grasped_then_dropped_near_hurdle (longest interval >= 8 rows, drop row within 100 mm of the hurdle) |
  grasped_then_dropped_far_from_hurdle | no_attach (no attached server chunk) | attach_then_revoke (first non-attached chunk after the
  first attached chunk has a state other than 'placed') | other.
Reference violation v_ref = min over k and queries of the reconstructed reference-row TO clearance (incl. 10 mm obstacle margin;
  held rows margin 0); refined clearance = recorded binding clearance (min of waypoint, path).
'At margin' shares: recorded binding clearance in [0, 1] mm and in [0, 2] mm.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, collections
import numpy as np
import os
ARM = os.environ.get("V11_ARM", "V11A")
RAW = {"V8": pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/XM/raw')), ARM: pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V11/raw'))}
ARMS = ("V8", ARM)   # T43 V11: copy of T43/XM/p/xm_agg.py (method unchanged); V8 from XM/raw/cross_V8.json, ARM from V11/raw/cross_<ARM>.json


def q(x, ps=(50, 90)):
    x = [v for v in x if v is not None and np.isfinite(v)]
    if not x:
        return {"n": 0}
    return {"n": len(x), "median": float(np.median(x)), **{f"p{p}": float(np.percentile(x, p)) for p in ps if p != 50},
            "min": float(np.min(x)), "max": float(np.max(x))}


def share(xs, f):
    xs = [x for x in xs if x is not None]
    return None if not xs else round(sum(1 for x in xs if f(x)) / len(xs), 4)


def dom(dd):
    a = {"z": abs(dd["dz"]), "n": abs(dd["dn"]), "a": abs(dd["da"])}
    k = max(a, key=a.get)
    if k == "z":
        return "up" if dd["dz"] > 0 else "down"
    if k == "n":
        return "away" if dd["away"] > 0 else "toward"
    return "along_bar"


def inhand_intervals(rows, gap=8):
    """[start, end] of apple-in-hand row runs, runs separated by < gap rows merged."""
    iv = []; cur = None
    for w in rows:
        if w["inhand"] and cur is None:
            cur = [w["t"], w["t"]]
        elif w["inhand"]:
            cur[1] = w["t"]
        elif cur is not None and w["t"] - cur[1] >= gap:
            iv.append(cur); cur = None
    if cur is not None:
        iv.append(cur)
    return iv


def group_of(r, c):
    t_from = r["first_inhand_t"] if r["first_inhand_t"] is not None else (r["first_closing_t"] or 0)
    if not c["in_window"]:
        return None
    if c["grasp_state"] == "placed":
        return "return"
    if c["t0"] + 7 < t_from:
        return None
    return "carry_apple" if c["inhand_at_plan"] else "carry_empty"


def main():
    out = {"definitions": __doc__, "arms": {}, "v9_timeline": {}, "v9_classes": {}, "per_run": {}}
    for arm in ARMS:
        R = json.load(open(RAW[arm] / f"cross_{arm}.json"))["runs"]
        A = {}
        G = collections.defaultdict(list); E = collections.defaultdict(list)
        crossing = collections.defaultdict(list); bind_at_cross = collections.Counter(); bind_cl = []
        for key, r in R.items():
            if "error" in r:
                continue
            ch = {c["seq"]: c for c in r["chunks"]}
            for c in r["chunks"]:
                g = group_of(r, c)
                if g:
                    G[g].append((key, c))
            for t, w in enumerate(r["rows"]):
                c = ch.get(w["seq"])
                if c is None or not w["executed"]:
                    continue
                g = group_of(r, c)
                if g and w["in_window"] and w["k"] < len(c["deflection"]["palm"]):
                    E[g].append((key, t, c["deflection"]["palm"][w["k"]], c["deflection"]["apple"][w["k"]]))
            cr = r["crossing"]; pr = {"key": key, "grasp": r["grasp"], "success": r["success"], "first_inhand_t": r["first_inhand_t"],
                                      "drop_t": r["drop_t"], "first_closing_t": r["first_closing_t"], "first_attached_t": r["first_attached_t"],
                                      "crossing_rows": r["crossing_rows"]}
            a_ = cr.get("carry_apple_M"); tp = cr.get("carry_tip_M"); pm = cr.get("carry_palm_M")
            if a_ and a_["inhand"]:
                crossing["apple_bottom_minus_bar_top_M"].append(a_["apple_bottom_minus_bar_top_mm"])
                crossing["apple_bar_M"].append(a_["geo_bar_mm"]["apple"])
                crossing["apple_min_hurdle_pm8_M"].append(a_["min_geoH_pm8_mm"]["apple"])
                pr["apple_cross"] = {"t": a_["t"], "bottom_minus_bar_top_mm": a_["apple_bottom_minus_bar_top_mm"], "apple_bar_mm": a_["geo_bar_mm"]["apple"],
                                     "apple_min_hurdle_pm8_mm": a_["min_geoH_pm8_mm"]["apple"]}
                ac = cr.get("carry_apple_C")
                if ac and ac["inhand"]:
                    crossing["apple_bottom_minus_bar_top_C"].append(ac["apple_bottom_minus_bar_top_mm"])
                    pr["apple_cross"]["C_t"] = ac["t"]; pr["apple_cross"]["C_bottom_minus_bar_top_mm"] = ac["apple_bottom_minus_bar_top_mm"]
            if tp:
                crossing[("fingers_bar_M_apple" if (tp["inhand"]) else "fingers_bar_M_empty")].append(tp["geo_bar_mm"]["fingers"])
                crossing[("fingers_min_hurdle_pm8_M_apple" if tp["inhand"] else "fingers_min_hurdle_pm8_M_empty")].append(tp["min_geoH_pm8_mm"]["fingers"])
                pr["tip_cross"] = {"t": tp["t"], "inhand": tp["inhand"], "fingers_bar_mm": tp["geo_bar_mm"]["fingers"],
                                   "fingers_min_hurdle_pm8_mm": tp["min_geoH_pm8_mm"]["fingers"], "z_tip_minus_bar_top_mm": tp["z_minus_bar_top_mm"]["tip"]}
            if pm:
                c = ch.get(pm["seq"])
                if c:
                    b = c["binding"]
                    bind_at_cross[f"{b['rows']}:{b.get('category')}:{b.get('class', '-')}"] += 1
                    bind_cl.append(b["clearance_mm"])
                    pr["binding_at_palm_cross"] = {"t": pm["t"], "seq": c["seq"], **b, "initial_source": c["initial_source"], "to_status": c["to_status"]}
            out["per_run"][key] = pr
        for g in ("carry_apple", "carry_empty", "return"):
            L = G[g]
            if not L:
                A[g] = {"n_chunks": 0}; continue
            ks = [max(range(len(c["deflection"]["palm"])), key=lambda i: c["deflection"]["palm"][i]["mag"]) for _, c in L]
            Dp = [c["deflection"]["palm"][k] for (_, c), k in zip(L, ks)]
            Da = [c["deflection"]["apple"][k] for (_, c), k in zip(L, ks)]
            mag = [d["mag"] for d in Dp]
            defl = [d for d in Dp if d["mag"] > 5.0]
            vref = [min(x["min_mm"] for x in c["to_rfc"]) if c.get("to_rfc") else None for _, c in L]
            vrfc_rec = [c["binding"]["clearance_mm"] for _, c in L]
            vrfc_rec_c = [c["binding"]["clearance_mm"] for (_, c), v in zip(L, vref) if v is not None and v < 0]
            pairs = [(c["deflection"]["palm"][k]["mag"], -v) for (_, c), k, v in zip(L, ks, vref) if v is not None and v < 0]
            prof = [[c["deflection"]["palm"][k]["mag"] for (_, c) in L if c["deflection"]["palm"] and max(x["mag"] for x in c["deflection"]["palm"]) > 5.0 and k < len(c["deflection"]["palm"])] for k in range(8)]
            cont_ref = [x for _, c in L if c["continuity_palm_mm"] for x in c["continuity_palm_mm"]["ref"]]
            cont_rfc = [x for _, c in L if c["continuity_palm_mm"] for x in c["continuity_palm_mm"]["rfc"]]
            ex = E[g]
            A[g] = {"n_chunks": len(L), "n_runs": len({k_ for k_, _ in L}),
                    "palm_D_max_mm": q(mag), "n_deflected_gt5": len(defl),
                    "deflected_components_at_kstar_mm": {c_: q([d[c_] for d in defl]) for c_ in ("dz", "away", "du", "dn", "da")},
                    "deflected_share": {"dz_pos": share(defl, lambda d: d["dz"] > 0), "away_pos": share(defl, lambda d: d["away"] > 0),
                                        "du_neg": share(defl, lambda d: d["du"] < 0)},
                    "deflected_dominant_axis": dict(collections.Counter(dom(d) for d in defl)),
                    "deflected_component_fraction_median": {"dz": float(np.median([d["dz"] / d["mag"] for d in defl])) if defl else None,
                                                            "away": float(np.median([d["away"] / d["mag"] for d in defl])) if defl else None,
                                                            "du": float(np.median([d["du"] / d["mag"] for d in defl])) if defl else None},
                    "apple_D_max_mm": q([d["mag"] for d, (_, c) in zip(Da, L) if c["inhand_at_plan"]]),
                    "apple_deflected_components_at_kstar_mm": {c_: q([d[c_] for d, (_, c) in zip(Da, L) if c["inhand_at_plan"] and d["mag"] > 5]) for c_ in ("dz", "away", "du")},
                    "executed_rows": {"n": len(ex), "palm_D_mm": q([e[2]["mag"] for e in ex]), "share_gt5": share(ex, lambda e: e[2]["mag"] > 5),
                                      "components_rows_gt5_mm": {c_: q([e[2][c_] for e in ex if e[2]["mag"] > 5]) for c_ in ("dz", "away", "du")},
                                      "dominant_axis_rows_gt5": dict(collections.Counter(dom(e[2]) for e in ex if e[2]["mag"] > 5)),
                                      "apple_D_mm": q([e[3]["mag"] for e in ex if e[3]["mag"] > 0])},
                    "initial_source": dict(collections.Counter(str(c["initial_source"]) for _, c in L)),
                    "to_status": dict(collections.Counter(str(c["to_status"]) for _, c in L)),
                    "hold_chunks": sum(1 for _, c in L if not all(c["executed"])),
                    "binding_rows": dict(collections.Counter(f"{c['binding']['rows']}:{c['binding'].get('category')}" for _, c in L)),
                    "ref_violation_recon_mm": q(vref), "n_ref_violating": sum(1 for v in vref if v is not None and v < 0),
                    "refined_binding_clearance_rec_mm_when_ref_violates": q(vrfc_rec_c, ps=(10, 50, 90)),
                    "share_binding_in_0_1mm_when_ref_violates": share(vrfc_rec_c, lambda x: 0 <= x <= 1),
                    "share_binding_in_0_2mm_when_ref_violates": share(vrfc_rec_c, lambda x: 0 <= x <= 2),
                    "share_binding_lt0_when_ref_violates": share(vrfc_rec_c, lambda x: x < 0),
                    "refined_binding_clearance_rec_mm_all": q(vrfc_rec, ps=(10, 50, 90)),
                    "D_vs_ref_violation": {"n": len(pairs), "pearson_r": float(np.corrcoef(np.array(pairs).T)[0, 1]) if len(pairs) > 2 else None,
                                           "ratio_D_over_viol": q([p / v for p, v in pairs]),
                                           "slope_lsq": float(np.polyfit([v for _, v in pairs], [p for p, _ in pairs], 1)[0]) if len(pairs) > 2 else None,
                                           "pairs": [[round(p, 2), round(v, 2)] for p, v in pairs]},
                    "palm_D_profile_by_k_median_mm": [float(np.median(p)) if p else None for p in prof],
                    "kstar_hist": dict(collections.Counter(int(k) for k, d in zip(ks, Dp) if d["mag"] > 5)),
                    "sqp": {"iterations": dict(collections.Counter(str(c["sqp_iterations"]) for _, c in L)),
                            "max_iterations_hit_share": share([c["max_iterations_hit"] for _, c in L], lambda x: str(x) == "True"),
                            "n_accepted": dict(collections.Counter(str(c["n_accepted"]) for _, c in L)),
                            "last_step_rad": q([c["candidates"][-1]["step_rad"] for _, c in L if c["candidates"] and c["candidates"][-1]["step_rad"] is not None]),
                            "first_step_rad": q([c["candidates"][0]["step_rad"] for _, c in L if c["candidates"] and c["candidates"][0]["step_rad"] is not None]),
                            "trust_at_last": q([c["candidates"][-1]["trust"] for _, c in L if c["candidates"] and c["candidates"][-1]["trust"] is not None])},
                    "max_step": {"anchor_ratio": q([c["anchor_ratio"] for _, c in L]), "anchor_ge_0.99": share([c["anchor_ratio"] for _, c in L], lambda x: x >= 0.99),
                                 "step_ratio_max": q([max(c["step_ratio"]) for _, c in L]), "step_ge_0.99": share([max(c["step_ratio"]) for _, c in L], lambda x: x >= 0.99),
                                 "accel_ratio_max": q([max(c["accel_ratio"]) if c["accel_ratio"] else None for _, c in L]),
                                 "accel_gt_1": share([max(c["accel_ratio"]) if c["accel_ratio"] else None for _, c in L], lambda x: x > 1.0)},
                    "continuity_palm_mm": {"refined_vs_prev_tail": q(cont_ref), "reference_vs_prev_tail": q(cont_rfc)},
                    "joint_max_abs_rad_active_arm": q([c["joint_max_abs_rad_active_arm"] for _, c in L])}
            # stay vs reference start split
            for src in ("stay", "reference"):
                sel = [(c, k) for (_, c), k in zip(L, ks) if str(c["initial_source"]).startswith(src)]
                if sel:
                    A[g].setdefault("by_initial_source", {})[src] = {
                        "n": len(sel), "palm_D_max_mm": q([c["deflection"]["palm"][k]["mag"] for c, k in sel]),
                        "dz": q([c["deflection"]["palm"][k]["dz"] for c, k in sel]), "away": q([c["deflection"]["palm"][k]["away"] for c, k in sel]),
                        "du": q([c["deflection"]["palm"][k]["du"] for c, k in sel])}
        A["crossing"] = {k_: q(v) for k_, v in crossing.items()}
        A["binding_at_palm_crossing"] = dict(bind_at_cross); A["binding_clearance_at_palm_crossing_mm"] = q(bind_cl)
        out["arms"][arm] = A
        # ---------- V9 timeline and classes ----------
        if False:
            for key, r in R.items():
                ch = {c["seq"]: c for c in r["chunks"]}
                att_seqs = [c["seq"] for c in r["chunks"] if c["attached"]]
                revoke = None
                if att_seqs:
                    s0 = att_seqs[0]
                    nxt = next((c for c in r["chunks"] if c["seq"] > s0 and not c["attached"]), None)
                    if nxt is not None:
                        revoke = {"seq": nxt["seq"], "t": nxt["t0"], "state": nxt["grasp_state"], "n_attached_chunks_before": sum(1 for s in att_seqs if s < nxt["seq"])}
                tl = []
                t_lo = r["first_closing_t"] or 0
                for t, w in enumerate(r["rows"]):
                    if t < t_lo - 8:
                        continue
                    c = ch.get(w["seq"])
                    if c is None or w["k"] >= len(c["deflection"]["palm"]):
                        continue
                    dp = c["deflection"]["palm"][w["k"]]
                    tl.append({"t": t, "seq": w["seq"], "k": w["k"], "executed": w["executed"], "state": c["grasp_state"], "init": c["initial_source"],
                               "inhand": w["inhand"], "apple_dz_mm": round(w["apple_dz_mm"], 1), "s_palm_M": round(w["s_M"]["palm"], 1),
                               "z_palm_M": round(w["z_M"]["palm"], 1), "geoH_fingers_M": round(w["geoH_M"]["fingers"], 1), "geoH_apple_M": round(w["geoH_M"]["apple"], 1),
                               "D": round(dp["mag"], 1), "dz": round(dp["dz"], 1), "away": round(dp["away"], 1), "du": round(dp["du"], 1),
                               "grip_cmd": round(w["grip_cmd"], 3), "finger_open": round(w["finger_open_M"], 4)})
                out["v9_timeline"][key] = tl
                pc = r["crossing_rows"]["palm_M"]["carry"]
                iv = inhand_intervals(r["rows"])
                main_iv = max(iv, key=lambda x: x[1] - x[0]) if iv else None
                dt = (iv[-1][1] + 1) if iv and iv[-1][1] + 1 < len(r["rows"]) else None
                lift = max(r["rows"], key=lambda w: w["apple_dz_mm"])
                if r["success"] in (True, "True"):
                    cls = "success"
                elif main_iv is not None and dt is not None and (main_iv[1] - main_iv[0]) >= 8 and r["rows"][dt]["in_window"]:
                    cls = "grasped_then_dropped_near_hurdle"
                elif main_iv is not None and dt is not None and (main_iv[1] - main_iv[0]) >= 8:
                    cls = "grasped_then_dropped_far_from_hurdle"
                elif r["first_attached_t"] is None:
                    cls = "no_attach"
                elif revoke is not None and revoke["state"] != "placed":
                    cls = "attach_then_revoke"
                else:
                    cls = "other"
                out["v9_classes"][key] = {"inhand_intervals": iv,
                    "class": cls, "first_closing_t": r["first_closing_t"], "first_attached_t": r["first_attached_t"], "revoke": revoke,
                    "first_inhand_t": r["first_inhand_t"], "drop_t": dt, "max_apple_lift_mm": round(lift["apple_dz_mm"], 1), "max_apple_lift_t": lift["t"],
                    "palm_carry_crossing_t": pc, "tip_carry_crossing_t": r["crossing_rows"]["tip_M"]["carry"],
                    "drop_row": None if dt is None else {"t": dt, "s_palm_mm": r["rows"][dt]["s_M"]["palm"], "apple_hurdle_mm_prev_row": r["rows"][dt - 1]["geoH_M"]["apple"],
                                                         "fingers_hurdle_mm": r["rows"][dt]["geoH_M"]["fingers"], "in_window": r["rows"][dt]["in_window"],
                                                         "chunk_init": ch[r["rows"][dt]["seq"]]["initial_source"] if r["rows"][dt]["seq"] in ch else None,
                                                         "grip_cmd": r["rows"][dt]["grip_cmd"], "state": ch[r["rows"][dt]["seq"]]["grasp_state"] if r["rows"][dt]["seq"] in ch else None},
                    "video": {"mp4": r["mp4"], "frame_rule": "frame t = post state of control row t, 15 fps, time = t/15 s",
                              "frames": {k_: v for k_, v in (("closing", r["first_closing_t"]), ("attach", r["first_attached_t"]), ("max_lift", lift["t"]),
                                                             ("drop", dt), ("palm_cross", pc)) if v is not None}}}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
    for arm in ARMS:
        A = out["arms"][arm]
        print(arm, json.dumps({g: {k_: A[g].get(k_) for k_ in ("n_chunks", "n_runs", "n_deflected_gt5", "deflected_dominant_axis", "initial_source", "n_ref_violating", "hold_chunks")} for g in ("carry_apple", "carry_empty", "return")}))


if __name__ == "__main__":
    main()
