"""T44dA verifier: TO deflection of the active hand in the LAST 4 CHUNKS BEFORE THE FIRST CLOSE (seq_c-4 .. seq_c-1, seq_c from close_geom.json),
per arm, from CLA/chunks (FK of reference vs refined, rows 0..7 = executed window).
 dz7_mm   = z(hand_mid refined row 7) - z(hand_mid reference row 7)   (hand_mid = mean of the two active fingers' sphere centres)
 push7_mm = |hand_mid - GT target| refined - reference at row 7 ; gapT = min GT-hull gap of active finger spheres to the target, rows 0..7 (ref / refined)
 tfc_now  = authorized rows (q_now) answered by target_free_coarse (server window record)
And from CLA/field (C1b, C2): coarse ON vs OFF clearance of authorized active-hand rows along the reference rows 0..7 in the same chunks.
-> CLA/approach_stats.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, pathlib
import numpy as np
C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA'))
cg = json.load(open(C / "close_geom.json")); out = {"per_run": {}, "per_arm": {}}
def med(x): x = [v for v in x if v is not None]; return float(np.median(x)) if x else None
for arm in ("C0", "C1a", "C1b", "C2"):
    rows = []
    for f in sorted(glob.glob(str(C / "chunks" / f"{arm}_*.jsonl"))):
        name = pathlib.Path(f).stem; g = cg.get(name) or {}
        if g.get("seq_c") is None: continue
        sc = g["seq_c"]; R = {r["seq"]: r for r in map(json.loads, open(f))}
        FS = {}
        fp = C / "field" / f"{name}.jsonl"
        if fp.exists(): FS = {r["seq"]: r for r in map(json.loads, open(fp)) if "ref" in r}
        rr = []
        for s in range(sc - 4, sc):
            r = R.get(s)
            if not r or "fk" not in r: continue
            fk = r["fk"]; fs = FS.get(s, {}).get("ref") or {}
            rr.append({"seq": s, "hold": r["hold"], "dz7_mm": 1e3 * (fk["mid_fin_row7"][2] - fk["mid_ref_row7"][2]), "push7_mm": fk["push_mm"][7],
                       "disp_mm": max(fk["disp_hand_mm"][:8]), "dq": fk["dq_max_rad_0_7"], "gapT_ref": fk["gap_target_ref_mm"], "gapT_fin": fk["gap_target_fin_mm"],
                       "gapO_fin": fk["gap_other_fin_mm"], "tfc_now": (r.get("win_by_tier_now") or {}).get("target_free_coarse"), "mvp_tier": (r.get("mvp") or {}).get("tier"),
                       "fs_on": fs.get("min_on_0_7"), "fs_off": fs.get("min_off_0_7"), "fs_cross10": fs.get("cross10_0_7"), "fs_cross0": fs.get("cross0_0_7"),
                       "fs_worst_tier": (fs.get("worst_on") or {}).get("tier_on"), "fs_worst_h": (fs.get("worst_on") or {}).get("h_bottom_mm"),
                       "fs_worst_inball": (fs.get("worst_on") or {}).get("inball"), "fs_worst_gapT": (fs.get("worst_on") or {}).get("gap_target_mm"),
                       "fs_worst_gapO": (fs.get("worst_on") or {}).get("gap_other_mm"),
                       "fs_low_h": (fs.get("lower_loc") or {}).get("h_bottom_above_support_mm_median"), "fs_low_inball": (fs.get("lower_loc") or {}).get("frac_inball"),
                       "fs_low_gapT": (fs.get("lower_loc") or {}).get("gap_target_mm_median"), "fs_low_gapO": (fs.get("lower_loc") or {}).get("gap_other_mm_median"),
                       "fs_drop": (fs.get("lower_loc") or {}).get("drop_mm_median")})
        out["per_run"][name] = {"seq_c": sc, "first_grasp_ok": g.get("first_grasp_ok"), "success": g.get("success"), "dz_c8": g["at_c8"]["dz"], "chunks": rr}
        rows += [dict(x, name=name) for x in rr]
    if not rows: continue
    k = lambda key: med([x[key] for x in rows])
    fsr = [x for x in rows if x["fs_on"] is not None]
    out["per_arm"][arm] = {"n_chunks": len(rows), "n_runs": len({x['name'] for x in rows}), "dz7_mm_median": k("dz7_mm"), "push7_mm_median": k("push7_mm"),
                           "disp_mm_median": k("disp_mm"), "frac_dq_gt_0.01": float(np.mean([x["dq"] > 0.01 for x in rows])),
                           "frac_dz7_gt_3mm": float(np.mean([x["dz7_mm"] > 3 for x in rows])),
                           "gapT_ref_median": k("gapT_ref"), "gapT_fin_median": k("gapT_fin"), "tfc_now_median": k("tfc_now"),
                           "frac_mvp_tier_tfc": float(np.mean([x["mvp_tier"] == "target_free_coarse" for x in rows])),
                           "n_hold": int(sum(x["hold"] for x in rows)),
                           "field_n": len(fsr), "field_min_on_median": med([x["fs_on"] for x in fsr]), "field_min_off_median": med([x["fs_off"] for x in fsr]),
                           "field_off_minus_on_median": med([x["fs_off"] - x["fs_on"] for x in fsr]),
                           "field_frac_cross10": float(np.mean([x["fs_cross10"] > 0 for x in fsr])) if fsr else None,
                           "field_frac_cross0": float(np.mean([x["fs_cross0"] > 0 for x in fsr])) if fsr else None,
                           "field_frac_worst_tfc": float(np.mean([x["fs_worst_tier"] == "target_free_coarse" for x in fsr])) if fsr else None,
                           "field_worst_tfc_h_bottom_median": med([x["fs_worst_h"] for x in fsr if x["fs_worst_tier"] == "target_free_coarse"]),
                           "field_worst_tfc_inball_frac": float(np.mean([x["fs_worst_inball"] for x in fsr if x["fs_worst_tier"] == "target_free_coarse"] or [np.nan])),
                           "field_worst_tfc_gapT_median": med([x["fs_worst_gapT"] for x in fsr if x["fs_worst_tier"] == "target_free_coarse"]),
                           "field_worst_tfc_gapO_median": med([x["fs_worst_gapO"] for x in fsr if x["fs_worst_tier"] == "target_free_coarse"]),
                           "field_lowered_rows_h_median": med([x["fs_low_h"] for x in fsr]), "field_lowered_rows_inball": med([x["fs_low_inball"] for x in fsr]),
                           "field_lowered_rows_gapT": med([x["fs_low_gapT"] for x in fsr]), "field_lowered_rows_gapO": med([x["fs_low_gapO"] for x in fsr]),
                           "field_lowered_drop_median": med([x["fs_drop"] for x in fsr])}
    print(arm, json.dumps(out["per_arm"][arm]))
json.dump(out, open(C / "approach_stats.json", "w"), indent=1)
