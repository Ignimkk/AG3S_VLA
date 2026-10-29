"""T44dA verifier: per C1a-ok / C1b-fail pair (7): first divergence + the C1b chunk records at the first material divergence (refined diff
> 0.01 rad) and at the C1b close chunk (seq_c), joined from diverge_C1a_C1b.json, chunks/, field/, close_geom.json. -> CLA/pairs_table.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA'))
dv = json.load(open(C / "diverge_C1a_C1b.json")); cg = json.load(open(C / "close_geom.json"))
L = lambda d, n: {r["seq"]: r for r in map(json.loads, open(C / d / f"{n}.jsonl")) if "seq" in r}
out = {}
for k, r in dv.items():
    A = L("chunks", "C1a_" + k); B = L("chunks", "C1b_" + k); FB = L("field", "C1b_" + k)
    def snap(X, F, s):
        x = X.get(s) or {}; f = (F or {}).get(s, {}).get("ref") or {}; fk = x.get("fk") or {}; m = x.get("mvp") or {}
        return {"seq": s, "hold": x.get("hold"), "grasp_state": x.get("grasp_state"), "to_status": x.get("to_status"),
                "verdict": [(q["kind"], q["action"], q["detail"][:120]) for q in x.get("reasons", []) if q["action"] != "execute"] or "execute",
                "worst_row": {"link": m.get("link"), "tier": m.get("tier"), "clearance_mm": None if m.get("clearance_m") is None else round(1e3 * m["clearance_m"], 2)},
                "auth_rows_now_by_tier": x.get("win_by_tier_now"),
                "push7_mm": (fk.get("push_mm") or [None] * 8)[7], "dz7_mm": None if not fk else round(1e3 * (fk["mid_fin_row7"][2] - fk["mid_ref_row7"][2]), 2),
                "disp_hand_mm": None if not fk else max(fk["disp_hand_mm"][:8]), "gapT_ref_fin_mm": None if not fk else [round(fk["gap_target_ref_mm"], 1), round(fk["gap_target_fin_mm"], 1)],
                "field_ref_min_on_off_mm": [f.get("min_on_0_7"), f.get("min_off_0_7")] if f else None, "field_cross10_0_7": f.get("cross10_0_7"),
                "field_worst_on": f.get("worst_on")}
    sc_b = (cg.get("C1b_" + k) or {}).get("seq_c"); sc_a = (cg.get("C1a_" + k) or {}).get("seq_c")
    out[k] = {"failure_mode": r["failure_mode"], "first_fin_diff_seq": r["first_fin"], "first_ref_diff_seq": r["first_ref"], "first_material_seq": r["first_mat"],
              "first_hold_diff_seq": r["first_hold_diff"], "close_seq": {"C1a": sc_a, "C1b": sc_b},
              "close_geom_c8": {"C1a": cg["C1a_" + k]["at_c8"], "C1b": cg["C1b_" + k]["at_c8"]},
              "hold_rows_after_close": {"C1a": cg["C1a_" + k]["hold_rows_c_16"], "C1b": cg["C1b_" + k]["hold_rows_c_16"]},
              "C1b_at_first_material": snap(B, FB, r["first_mat"]), "C1a_at_first_material": snap(A, None, r["first_mat"]),
              "C1b_at_close": snap(B, FB, sc_b), "C1b_before_close": [snap(B, FB, s) for s in range(sc_b - 3, sc_b)],
              "C1a_at_close": snap(A, None, sc_a)}
    o = out[k]; b = o["C1b_at_first_material"]
    print(k, o["failure_mode"]["C1b"], "fin", o["first_fin_diff_seq"], "mat", o["first_material_seq"], "holddiff", o["first_hold_diff_seq"], "close", o["close_seq"],
          "| B@mat", b["hold"], b["worst_row"], "push", b["push7_mm"], "dz", b["dz7_mm"], "f on/off", b["field_ref_min_on_off_mm"], (b["field_worst_on"] or {}).get("tier_on"),
          "| dz_c8 A/B %.1f/%.1f" % (o["close_geom_c8"]["C1a"]["dz"], o["close_geom_c8"]["C1b"]["dz"]), "holdrows A/B", o["hold_rows_after_close"])
    for s in o["C1b_before_close"]:
        w = s["field_worst_on"] or {}
        print("    pre-close", s["seq"], s["hold"], s["worst_row"]["tier"], "push", s["push7_mm"], "dz", s["dz7_mm"], "gapT", s["gapT_ref_fin_mm"], "f", s["field_ref_min_on_off_mm"] and [round(v, 1) for v in s["field_ref_min_on_off_mm"]],
              w.get("tier_on"), "h", w.get("h_bottom_mm") and round(w["h_bottom_mm"], 1), "inball", w.get("inball"), "gT", w.get("gap_target_mm") and round(w["gap_target_mm"], 1), "gO", w.get("gap_other_mm") and round(w["gap_other_mm"], 1))
json.dump(out, open(C / "pairs_table.json", "w"), indent=1)
