"""T21 verifier — summary tables from replay_<run>.json (no interpretation)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
import numpy as np
from pathlib import Path
OUT = Path(_ag3s_workspace_path('outputs/verify/t21'))
KEYS = ["live_exact", "live_proxy", "a_rec", "b1_rec", "b2_rec", "a_t20", "b1_t20", "b2_t20",
        "a_default", "b1_default", "b2_default"]
COL = ["live_exact", "live_proxy", "b1_rec", "b2_rec", "a_t20", "b1_t20", "b2_t20"]


def summarize(run):
    d = json.loads((OUT / f"replay_{run}.json").read_text())
    S = {"run": run, "pads": d["pads"], "chunks": [], "per_rule": {}, "collision": {}, "fidelity": [], "m3": {}}
    ch = [c for c in d["chunks"] if not c.get("skipped_reason")]
    S["chunks"] = [c["chunk"] for c in ch]
    S["t_steps"] = [c["t_step"] for c in ch]
    S["skipped"] = [(c["chunk"], c["skipped_reason"]) for c in d["chunks"] if c.get("skipped_reason")]
    S["fk_vs_linearizer_max_mm"] = max(c["fk_vs_linearizer_max_mm"] for c in ch)
    S["live_vs_a_rec_lower_mm_max"] = max(c["live_vs_a_rec_lower_mm"] for c in ch)
    for key in KEYS:
        rows = []
        for c in ch:
            e = c["fields"].get(key)
            if e is None:
                rows.append(None); continue
            rows.append({"frac_all": e["fraction_in_window_all"], "frac_now": e["fraction_in_window_now"],
                         "frac_path": e["fraction_in_window_path"], "finger_frac_all": e["finger_fraction_in_window_all"],
                         "by_tier_all": e["by_tier_all"], "finger_by_tier_all": e["finger_by_tier_all"],
                         "finger_by_tier_now": e["finger_by_tier_now"],
                         "n_unresolved_all": e["n_unresolved_all"], "n_all": e["n_all"],
                         "basis": e.get("basis", "recorded"), "overflow": e.get("overflow"),
                         "m3": e["m3_finger"], "m3_nearest": e.get("m3_finger_nearest_kept_label"),
                         "m3_cloud": e.get("m3_cloud_in_window")})
        ok = [r for r in rows if r]
        tot = {}
        for r in ok:
            for t, v in r["by_tier_all"].items():
                tot[t] = tot.get(t, 0) + v
        ftot = {}
        for r in ok:
            for t, v in r["finger_by_tier_all"].items():
                ftot[t] = ftot.get(t, 0) + v
        S["per_rule"][key] = {
            "per_chunk": rows,
            "n_chunks": len(ok),
            "chunks_finger_all_resolved": sum(1 for r in ok if r["finger_by_tier_all"]["coarse"] + r["finger_by_tier_all"]["outside"] == 0),
            "chunks_finger_now_in_fine_or_tf": sum(1 for r in ok if r["finger_by_tier_now"]["fine"] + r["finger_by_tier_now"]["target_free"] > 0),
            "chunks_all_authorized_in_window": sum(1 for r in ok if r["frac_all"] == 1.0),
            "frac_all_mean": float(np.mean([r["frac_all"] for r in ok])) if ok else None,
            "by_tier_all_total": tot, "finger_by_tier_all_total": ftot,
            "n_unresolved_total": sum(r["n_unresolved_all"] for r in ok),
            "n_rows_total": sum(r["n_all"] for r in ok),
            "basis_counts": {b: sum(1 for r in ok if r["basis"] == b) for b in {r["basis"] for r in ok}},
            "overflow_counts": {str(o): sum(1 for r in ok if r["overflow"] == o) for o in {r["overflow"] for r in ok}},
        }
    for key in COL:
        per = [c["collision_variant"].get(key) for c in ch]
        per = [p for p in per if p]
        if not per:
            continue
        disp = np.array([p["fingertip_disp_mm_per_step"] for p in per])
        wr = [p["worst_row"] or {} for p in per]
        auth = {}
        for w in wr:
            for t, v in ((w.get("authorized_rows") or {}).get("by_tier") or {}).items():
                auth[t] = auth.get(t, 0) + v
        S["collision"][key] = {
            "per_chunk_median_mm": np.median(disp, axis=1).tolist(),
            "per_chunk_max_mm": disp.max(axis=1).tolist(),
            "pooled_median_mm": float(np.median(disp)), "pooled_max_mm": float(disp.max()),
            "returned": [p["returned"] for p in per], "sqp_iterations": [p["sqp_iterations"] for p in per],
            "worst_row_tier": [w.get("tier") for w in wr], "worst_row_link": [w.get("link") for w in wr],
            "authorized_rows_by_tier_total_returned": auth,
            "authorized_rows_verifiable": [(w.get("authorized_rows") or {}).get("verifiable") for w in wr],
            "active_rows_selected_by_tier_reference": [p["active_rows_selected_by_tier"] for p in per],
        }
    S["fidelity"] = [dict(chunk=c["chunk"], **c["proxy_fidelity"]) for c in ch]
    S["ball_removed_by_label"] = [dict(chunk=c["chunk"], **c["ball_removed_by_label"]) for c in ch]
    S["manipulated"] = [dict(chunk=c["chunk"], **c["manipulated"]) for c in ch]
    S["truth_at_query"] = [dict(chunk=c["chunk"], **c["truth_at_query"]["finger_min_mm"]) for c in ch]
    (OUT / f"summary_{run}.json").write_text(json.dumps(S, indent=1))
    return S


if __name__ == "__main__":
    for run in sys.argv[1:]:
        S = summarize(run)
        print("==", run, S["chunks"], "skipped", S["skipped"], "fk", S["fk_vs_linearizer_max_mm"], "live-a_rec", S["live_vs_a_rec_lower_mm_max"])
        for k, v in S["per_rule"].items():
            print(f"{k:11s} n={v['n_chunks']} frac={v['frac_all_mean']:.3f} fingerResolvedChunks={v['chunks_finger_all_resolved']} "
                  f"fingerNowFine={v['chunks_finger_now_in_fine_or_tf']} tiers={ {t:n for t,n in v['by_tier_all_total'].items() if n} } "
                  f"unres={v['n_unresolved_total']}/{v['n_rows_total']} basis={v['basis_counts']} ovf={v['overflow_counts']}")
        for k, v in S["collision"].items():
            print(f"COL {k:11s} med={v['pooled_median_mm']:.2f} max={v['pooled_max_mm']:.2f} per_med={np.round(v['per_chunk_median_mm'],1).tolist()} ret={v['returned']} tier={v['worst_row_tier']} auth={ {t:n for t,n in v['authorized_rows_by_tier_total_returned'].items() if n} }")
        for f in S["fidelity"]:
            print("FID", f["chunk"], round(f["tier_agreement_all"], 3), f.get("fine_query_diff_mm"), f.get("fine_voxel_diff_mm_exact_0_100mm_above_support", {}).get("median"), f.get("fine_voxel_diff_mm_exact_0_100mm_above_support", {}).get("p90"))
