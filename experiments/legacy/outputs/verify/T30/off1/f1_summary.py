"""Summarise f1_<COND>.json (f1_counterfactual.py). Counts per trajectory (reference / refined) x rule (old / new)."""
import json, collections, sys
import numpy as np
out = {}
for C in ("E3a", "E3b"):
    R = json.load(open(f"f1_{C}.json"))
    ch = [c for c in R["chunks"] if "skipped" not in c]
    s = {"n_chunks": len(R["chunks"]), "n_evaluated": len(ch), "n_skipped_attached": sum(1 for c in R["chunks"] if c.get("skipped") == "attached")}
    # recorded
    s["recorded_refined_clearance_lt0"] = sum(1 for c in R["chunks"] if (c["recorded"]["clearance_m"] or 0) < 0)
    s["recorded_refined_lt0_target_free_target"] = sum(1 for c in R["chunks"] if (c["recorded"]["clearance_m"] or 0) < 0 and c["recorded"]["tier"] == "target_free" and c["recorded"]["obstacle"] == "target")
    s["recorded_initial_violation_gt0"] = sum(1 for c in R["chunks"] if (c["recorded"]["initial_violation_m"] or 0) > 0)
    fd = [c["fidelity_refined_old_abs_diff_mm"] for c in ch if "fidelity_refined_old_abs_diff_mm" in c]
    s["fidelity_refined_old_abs_diff_mm"] = {"n": len(fd), "max": max(fd), "median": float(np.median(fd)), "n_le_1mm": sum(1 for x in fd if x <= 1.0)}
    s["fidelity_same_row"] = sum(1 for c in ch if c.get("fidelity_same_row"))
    fi = [c["fidelity_reference_old_initial_violation_abs_diff_mm"] for c in ch if "fidelity_reference_old_initial_violation_abs_diff_mm" in c]
    s["fidelity_reference_old_initial_violation_abs_diff_mm"] = {"n": len(fi), "max": max(fi), "median": float(np.median(fi)), "n_le_1mm": sum(1 for x in fi if x <= 1.0)}
    for tn in ("reference", "refined"):
        for rn in ("old", "new"):
            k = f"{tn}_{rn}"
            V = [c for c in ch if c[k]["violated"]]
            s[k] = {"n_violated_chunks": len(V),
                    "n_violated_worst_target_free_target": sum(1 for c in V if c[k].get("tier") == "target_free" and c[k].get("target_margin_row")),
                    "n_violated_worst_target_margin_row_any_tier": sum(1 for c in V if c[k].get("target_margin_row")),
                    "n_violated_chunks_with_any_neg_row_target_free_target": sum(1 for c in V if c[k]["n_neg_rows_target_free_target"] > 0),
                    "tier_counts": dict(collections.Counter(str(c[k].get("tier")) for c in V)),
                    "gt_nearest_counts": dict(collections.Counter(str(c[k].get("gt_nearest")) for c in V)),
                    "link_counts": dict(collections.Counter(str(c[k].get("link")) for c in V)),
                    "worst_clearance_mm_min": min((c[k]["worst_clearance_mm"] for c in V), default=None),
                    "by_run": dict(collections.Counter(c["run"] for c in V)),
                    "violations": [{"run": c["run"], "t": c["t"], "link": c[k].get("link"), "tier": c[k].get("tier"), "clearance_mm": round(c[k]["worst_clearance_mm"], 2),
                                    "gt": c[k].get("gt_nearest"), "gt_surface_mm": round(c[k].get("gt_nearest_surface_mm", float("nan")), 1),
                                    "apple_surface_mm": round(c[k].get("apple_surface_mm", float("nan")), 1), "target_margin_row": c[k].get("target_margin_row")} for c in V]}
    # old-rule reproduction of recorded label: recorded 'target_free'+'target' vs my target-margin flag on the same chunk (refined, old)
    agree = [(c["recorded"]["tier"] == "target_free" and c["recorded"]["obstacle"] == "target", c["refined_old"].get("tier") == "target_free" and bool(c["refined_old"].get("target_margin_row")))
             for c in ch if (c["recorded"]["clearance_m"] or 0) < 0]
    s["label_check_refined_old_vs_recorded"] = {"n": len(agree), "both": sum(1 for a, b in agree if a and b), "recorded_only": sum(1 for a, b in agree if a and not b),
                                                "reconstructed_only": sum(1 for a, b in agree if b and not a), "neither": sum(1 for a, b in agree if not a and not b)}
    out[C] = s
    print(C, json.dumps({k: (v if not isinstance(v, dict) or "violations" not in v else {kk: vv for kk, vv in v.items() if kk != "violations"}) for k, v in s.items()}, indent=1))
json.dump(out, open("f1_summary.json", "w"), indent=1)
