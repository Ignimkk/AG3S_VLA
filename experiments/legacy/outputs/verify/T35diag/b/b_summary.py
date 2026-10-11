"""T35-diag B summary — per run over the +-10 chunk window around the first close command (b/b_<run>.json).
Per chunk: recorded dev (deg, planned horizon), fingertip displacement refined vs reference (mm, max over steps),
join |ref[0]-prev refined[8]| (deg, arm max), returned, per-variant dev, 'driver' = variants whose dev <= 0.25 * replay dev
when replay dev >= 0.5 deg (term whose removal takes the deviation away), reference worst row (link, tier, clearance, GT nearest),
limits (recorded initial_limit_overshoot position > 0; reference beyond TO position limits; reference/refined beyond MJCF).
usage: b_summary.py -> b/b_summary.json"""
import json, glob, pathlib
import numpy as np
out = {"definitions": __doc__, "runs": {}}
for f in sorted(glob.glob("b/b_E3*_r?.json")):
    d = json.load(open(f)); run = d["run"]
    rows = []
    for e in d["chunks"]:
        ab = e["ablation"]; rp = ab.get("replay", {}).get("dev_max_deg", np.nan)
        drv = [k for k in ("no_collision", "no_limits", "no_continuity", "no_smooth")
               if rp >= 0.5 and ab.get(k, {}).get("dev_max_deg", np.inf) <= 0.25 * rp]
        rw = e["reference"].get("worst") or {}
        rows.append({"t": e["t"], "rel": e["rel_seq"], "held": e["held"], "returned": e["recorded"]["returned"],
                     "dev_deg": e["recorded_dev_deg"]["max"], "dev_rows0_7_deg": e["recorded_dev_deg"]["rows0_7_max"],
                     "top_joint": max(e["recorded_dev_deg"]["per_joint"].items(), key=lambda kv: kv[1])[0],
                     "per_joint_deg": e["recorded_dev_deg"]["per_joint"],
                     "tip_disp_mm": e["recorded_refined_vs_reference"]["fingertip_disp_mm_max"],
                     "join_deg": (e.get("join_ref0_vs_prev_refined8_deg") or {}).get("max_arm"),
                     "variants_dev_deg": {k: v.get("dev_max_deg") for k, v in ab.items()},
                     "variants_tip_mm": {k: v.get("fingertip_disp_mm_max") for k, v in ab.items()},
                     "variants_returned": {k: v.get("returned") for k, v in ab.items()},
                     "replay_vs_recorded_deg": ab.get("replay", {}).get("vs_recorded_refined_max_deg"),
                     "driver": drv,
                     "ref_worst": {"link": rw.get("link"), "tier": rw.get("tier"), "clearance_mm": e["reference"]["worst_clearance_mm"],
                                   "gt_nearest": rw.get("gt_nearest"), "gt_nearest_mm": rw.get("gt_nearest_mm"),
                                   "gt_nearest_excl_apple": rw.get("gt_nearest_excl_apple"), "gt_apple_mm": rw.get("gt_apple_mm")},
                     "ref_n_neg_rows": e["reference"]["n_neg_rows"], "ref_neg_by_link": e["reference"]["neg_by_link"],
                     "ref_neg_by_tier": e["reference"]["neg_by_tier"],
                     "initial_limit_overshoot_position_rad": (e["recorded"]["initial_limit_overshoot"] or {}).get("position"),
                     "ref_beyond_TO_position": {k: v["over_deg"] for k, v in e["reference"]["beyond_TO_position"].items()},
                     "ref_beyond_MJCF_deg": {k: v["over_deg"] for k, v in e["reference"]["beyond_MJCF_range"].items()},
                     "refined_beyond_MJCF_deg": {k: v["over_deg"] for k, v in e["refined"]["beyond_MJCF_range"].items()},
                     "fidelity_refined_worst_abs_diff_mm": e["fidelity"]["refined_worst_abs_diff_mm"],
                     "fidelity_same_row": e["fidelity"]["refined_same_row"]})
    pre = [r for r in rows if r["rel"] < 0]; post = [r for r in rows if r["rel"] >= 0]
    def agg(rr):
        if not rr:
            return None
        k = int(np.argmax([r["dev_deg"] for r in rr]))
        kt = int(np.argmax([r["tip_disp_mm"] for r in rr]))
        from collections import Counter
        return {"n": len(rr), "max_dev_deg": rr[k]["dev_deg"], "t_max_dev": rr[k]["t"], "joint_max_dev": rr[k]["top_joint"],
                "driver_at_max_dev": rr[k]["driver"], "max_tip_disp_mm": rr[kt]["tip_disp_mm"], "t_max_tip": rr[kt]["t"],
                "returned_counts": dict(Counter(r["returned"] for r in rr)),
                "driver_counts": dict(Counter(x for r in rr for x in r["driver"])),
                "n_ref_neg_rows_chunks": sum(1 for r in rr if r["ref_n_neg_rows"] > 0),
                "n_projection": sum(1 for r in rr if r["returned"] == "projection"),
                "n_ref_beyond_TO_position": sum(1 for r in rr if r["ref_beyond_TO_position"]),
                "n_ref_beyond_MJCF": sum(1 for r in rr if r["ref_beyond_MJCF_deg"]),
                "n_refined_beyond_MJCF": sum(1 for r in rr if r["refined_beyond_MJCF_deg"]),
                "max_join_deg": max((r["join_deg"] or 0) for r in rr),
                "fidelity_max_abs_diff_mm": max((r["fidelity_refined_worst_abs_diff_mm"] or 0) for r in rr),
                "fidelity_all_same_row": all(r["fidelity_same_row"] for r in rr)}
    out["runs"][run] = {"t_close": d["t_close"], "s_close": d["s_close"], "pre10": agg(pre), "post11": agg(post), "chunks": rows}
json.dump(out, open("b/b_summary.json", "w"), indent=1, default=float)
for run, v in out["runs"].items():
    print(run, "close", v["t_close"])
    for w in ("pre10", "post11"):
        a = v[w]; print("  ", w, {k: a[k] for k in ("max_dev_deg", "t_max_dev", "joint_max_dev", "driver_at_max_dev", "max_tip_disp_mm", "t_max_tip", "returned_counts", "driver_counts", "n_ref_neg_rows_chunks", "n_projection", "n_ref_beyond_TO_position", "n_ref_beyond_MJCF", "max_join_deg", "fidelity_max_abs_diff_mm", "fidelity_all_same_row")})
