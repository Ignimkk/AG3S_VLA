"""T43 SV part 1/2 aggregation — grasp-collapse cause table from raw/grasp_{V9V8,V3V4}.json (+ analysis json for attach_revoked).
usage: sv_grasp_agg.py OUT_JSON
Per pair key with grasp_A False and grasp_B True ("failure"), and as contrast keys with both grasp True:
  failure_type     : no_attach (A never attached) | attach_revoked (A attached, next chunk detached; reasons = grasp.attach_revoked.reasons)
  divergence       : div10_t / div30_t and server grasp.state of each arm at that row; first policy-output (reference) difference
  closing          : first closing t (A, B); measured fingertip-apple distance (mm) and server tip_alignment.now (d3 / dxy / dz, mm) on the
                     first closing chunk; outward_mm on closing chunks (max, n > 2 mm, A);
  approach window  : chunks of each arm with t in [first_closing_t - 48, first_closing_t) (6 chunks), state latched:
                     outward_mm median / max, n chunks with outward > 2 mm, initial_source 'stay' count, binding rows (rows:class:label),
                     tip dz median (tip_alignment.now).
  binding rows     : class = recorded obstacle_margin_class (waypoint) or reconstructed (path; fidelity reported); label = truth nearest;
                     exemptions on path binding rows: reads_target_free (authorized row on the target-free layer = T30 F1 exempt),
                     e1_margin_applies, path_target_row_dropped.
  first_push_chunk : first A chunk (t >= 64) whose binding is a path row with outward_mm > 2 mm (row, class, label, link).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, collections
import numpy as np
SV = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/SV/raw'))
T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
AN = {"V9": T43 / "V9/analysis/analysis.json", "V8": T43 / "V8/analysis/analysis.json", "V3": T43 / "W/analysis/analysis.json", "V4": T43 / "W/analysis/analysis.json"}
_an = {}


def an(arm):
    if arm not in _an:
        _an[arm] = json.load(open(AN[arm]))["runs"]
    return _an[arm]


def stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": len(a), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def bind_desc(c):
    b = c.get("binding") or {}; rc = c.get("recon") or {}
    rows = b.get("rows")
    r = (rc.get("path") if rows == "path" else rc.get("wp")) or {}
    cls = r.get("recorded_class") if rows == "waypoint" and r.get("recorded_class") else r.get("class")
    if r.get("held_row"):
        cls = "held"
    return {"t": c["t"], "rows": rows, "link": b.get("link"), "clearance_mm": b.get("clearance_mm"), "class": cls, "label": r.get("truth_nearest"),
            "reads_target_free": r.get("reads_target_free"), "e1_margin_applies": r.get("e1_margin_applies"),
            "path_target_row_dropped": r.get("path_target_row_dropped"), "fidelity_mm": r.get("fidelity_mm"),
            "recon_class": r.get("class"), "recorded_class": r.get("recorded_class")}


def window(a, t_close):
    if t_close is None:
        return []
    return [c for c in a["chunks"] if t_close - 48 <= c["t"] < t_close]


def summarize_arm(a, arm, key):
    fc = a["first_closing_t"]; ch = {c["t"]: c for c in a["chunks"]}
    out = {"first_closing_t": fc, "attach_t": a["attach_t"]}
    if fc is not None and fc in ch:
        c = ch[fc]; out["tip_apple_mm_first_closing"] = c["tip_apple_mm"]; out["tip_alignment_first_closing"] = c.get("tip_alignment_now")
    cl = [c for c in a["chunks"] if c["grasp_state"] == "closing" and not c.get("attached")]
    out["closing_outward_mm"] = stat([c["outward_mm"] for c in cl]); out["closing_n_outward_gt2"] = sum(c["outward_mm"] > 2 for c in cl)
    w = window(a, fc)
    out["window_t"] = [c["t"] for c in w]
    out["window_outward_mm"] = stat([c["outward_mm"] for c in w]); out["window_n_outward_gt2"] = sum(c["outward_mm"] > 2 for c in w)
    out["window_stay_starts"] = sum(1 for c in w if c.get("initial_source") == "stay")
    out["window_tip_dz_mm"] = stat([(c.get("tip_alignment_now") or {}).get("dz_mm") for c in w])
    out["window_tip_dxy_mm"] = stat([(c.get("tip_alignment_now") or {}).get("dxy_mm") for c in w])
    out["window_tip_apple_mm"] = stat([c["tip_apple_mm"] for c in w])
    bd = [bind_desc(c) for c in w]
    out["window_binding"] = bd
    out["window_binding_counts"] = dict(collections.Counter(f'{b["rows"]}:{b["class"]}:{b["label"]}' for b in bd))
    pb = [b for b in bd if b["rows"] == "path"]
    out["window_path_binding_exemptions"] = {"n": len(pb), "reads_target_free_F1_exempt": sum(bool(b["reads_target_free"]) for b in pb),
                                             "e1_margin_applies": sum(bool(b["e1_margin_applies"]) for b in pb),
                                             "target_row_dropped": sum(bool(b["path_target_row_dropped"]) for b in pb),
                                             "fidelity_mm_max": max([b["fidelity_mm"] for b in pb if b["fidelity_mm"] is not None], default=None)}
    fp = next((c for c in a["chunks"] if c["t"] >= 64 and (c.get("binding") or {}).get("rows") == "path" and c["outward_mm"] > 2.0), None)
    out["first_push_chunk"] = None if fp is None else dict(bind_desc(fp), outward_mm=fp["outward_mm"], grasp_state=fp["grasp_state"])
    # attach revoked
    r = an(arm)[key]; rev = [(c["t"], (c.get("attach_revoked") or {}).get("reasons")) for c in (r.get("server_chunks") or []) if c.get("attach_revoked")]
    out["attach_revoked"] = rev
    out["fidelity_all_recon_mm_max"] = max([((c.get("recon") or {}).get(x) or {}).get("fidelity_mm") or 0.0 for c in a["chunks"] for x in ("wp", "path")], default=None)
    out["wp_class_recon_vs_recorded_disagree"] = sum(1 for c in a["chunks"] if ((c.get("recon") or {}).get("wp") or {}).get("recorded_class") not in (None,)
                                                     and ((c.get("recon") or {}).get("wp") or {}).get("class") is not None
                                                     and c["recon"]["wp"]["class"] != c["recon"]["wp"]["recorded_class"])
    out["wp_class_recon_n"] = sum(1 for c in a["chunks"] if ((c.get("recon") or {}).get("wp") or {}).get("class") is not None
                                   and c["recon"]["wp"].get("recorded_class") is not None)
    return out


def main():
    res = {"definitions": __doc__, "pairs": {}}
    for pair in ("V9V8", "V3V4"):
        d = json.load(open(SV / f"grasp_{pair}.json")); A, B = d["A"], d["B"]
        rows = []
        for ka, r in sorted(d["runs"].items()):
            kind = "failure" if (not r["grasp_A"] and r["grasp_B"]) else ("both_grasp" if (r["grasp_A"] and r["grasp_B"]) else ("both_fail" if not r["grasp_A"] and not r["grasp_B"] else "A_only"))
            sa = summarize_arm(r["arms"][A], A, r["key_A"]); sb = summarize_arm(r["arms"][B], B, r["key_B"])
            ftype = None
            if not r["grasp_A"]:
                ftype = "attach_revoked" if sa["attach_t"] is not None else "no_attach"
            dv = r["divergence"]
            rows.append({"key": ka, "stage": r["stage"], "episode": r["episode"], "seed": r["seed"], "kind": kind, "failure_type": ftype,
                         "div10_t": dv.get("div10_t"), "div10_state_A": dv.get("div10_state_A"), "div10_state_B": dv.get("div10_state_B"),
                         "div30_t": dv.get("div30_t"), "div30_state_A": dv.get("div30_state_A"), "div30_state_B": dv.get("div30_state_B"),
                         "first_reference_diff": dv.get("first_reference_diff"), "seq1_refined_A_minus_B_max_abs_rad": dv.get("seq1_refined_A_minus_B_max_abs_rad"),
                         "seq1_reference_equal_max_abs_rad": dv.get("seq1_reference_equal_max_abs_rad"), "A": sa, "B": sb})
        fl = [x for x in rows if x["kind"] == "failure"]; ok = [x for x in rows if x["kind"] == "both_grasp"]

        def pool(sel, arm, f):
            return stat([f(x[arm]) for x in sel])
        summ = {"n_pairs": len(rows), "n_failure": len(fl), "n_both_grasp": len(ok), "kinds": dict(collections.Counter(x["kind"] for x in rows)),
                "failure_types": dict(collections.Counter(x["failure_type"] for x in fl)),
                "failure_stage": dict(collections.Counter(x["stage"] for x in fl)),
                "div10_state_A_failures": dict(collections.Counter(str(x["div10_state_A"]) for x in fl)),
                "div30_state_A_failures": dict(collections.Counter(str(x["div30_state_A"]) for x in fl)),
                "div10_t_failures": stat([x["div10_t"] for x in fl]), "div30_t_failures": stat([x["div30_t"] for x in fl]),
                "first_reference_diff_t_failures": stat([(x["first_reference_diff"] or {}).get("t") for x in fl])}
        for lab, sel in (("failures", fl), ("both_grasp", ok)):
            for arm in ("A", "B"):
                summ[f"{lab}_{arm}"] = {
                    "tip_apple_mm_first_closing": pool(sel, arm, lambda s: s.get("tip_apple_mm_first_closing")),
                    "tip_dz_mm_first_closing": pool(sel, arm, lambda s: (s.get("tip_alignment_first_closing") or {}).get("dz_mm")),
                    "tip_dxy_mm_first_closing": pool(sel, arm, lambda s: (s.get("tip_alignment_first_closing") or {}).get("dxy_mm")),
                    "first_closing_t": pool(sel, arm, lambda s: s.get("first_closing_t")),
                    "window_outward_median_mm": pool(sel, arm, lambda s: (s.get("window_outward_mm") or {}).get("median")),
                    "window_outward_max_mm": pool(sel, arm, lambda s: (s.get("window_outward_mm") or {}).get("max")),
                    "window_n_outward_gt2_total": sum(x[arm]["window_n_outward_gt2"] for x in sel),
                    "window_chunks_total": sum(len(x[arm]["window_t"]) for x in sel),
                    "window_stay_starts_total": sum(x[arm]["window_stay_starts"] for x in sel),
                    "window_tip_dz_median_mm": pool(sel, arm, lambda s: (s.get("window_tip_dz_mm") or {}).get("median")),
                    "closing_outward_max_mm": pool(sel, arm, lambda s: (s.get("closing_outward_mm") or {}).get("max")),
                    "closing_n_outward_gt2_total": sum(x[arm]["closing_n_outward_gt2"] for x in sel),
                    "window_binding_counts_total": dict(sum((collections.Counter(x[arm]["window_binding_counts"]) for x in sel), collections.Counter())),
                    "window_path_binding_exemptions_total": {k: sum((x[arm]["window_path_binding_exemptions"].get(k) or 0) for x in sel)
                                                             for k in ("n", "reads_target_free_F1_exempt", "e1_margin_applies", "target_row_dropped")},
                    "first_push_chunk_class_label": dict(collections.Counter(f'{(x[arm]["first_push_chunk"] or {}).get("class")}:{(x[arm]["first_push_chunk"] or {}).get("label")}' for x in sel)),
                    "recon_fidelity_max_mm": max([x[arm]["fidelity_all_recon_mm_max"] or 0 for x in sel], default=None),
                    "wp_class_recon_disagree": sum(x[arm]["wp_class_recon_vs_recorded_disagree"] for x in sel),
                    "wp_class_recon_n": sum(x[arm]["wp_class_recon_n"] for x in sel)}
        res["pairs"][pair] = {"A": A, "B": B, "summary": summ, "rows": rows}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
    for pair, v in res["pairs"].items():
        print(pair, json.dumps(v["summary"], default=float)[:6000])


if __name__ == "__main__":
    main()
