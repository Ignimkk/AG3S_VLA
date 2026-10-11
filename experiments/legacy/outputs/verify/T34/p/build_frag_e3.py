"""Build the T30 <COND> fragment: T28-style condition/run summary (summarize_e3.py) + T30 extras from <COND>_analysis_extra.json.
usage: build_frag_e3.py COND [DUP_JSON]
Extras per run:
  association_counts   : server decision reason / association outcome counts (subset_kept_anchor, association_rejected, accepted)
  ball_vs_gt_apple     : per group (pre_grasp non-frozen / frozen / after_switch): chunks with a ball, GT apple centre inside, whole (+34 mm) inside
  violating_rows       : chunks with >= 1 refined-window esdf row < 0; authorized rows by (tier, GT nearest) and unauthorized by (link, GT nearest)
  stamps               : stamp_mode counts, camera_skew_sec max, render_spread_sec median / max
  hold_timeline        : (t, kinds) of every HOLD chunk (E3b) / would_hold chunk (E3a)
Q counts (per run and pooled):
  Q4 hold / would_hold kinds; comms HOLD (ipc timeout/stale/error) ; n_time_budget_hit
  Q5 constraint spheres: from server log (122, finger 3.25 mm, palm 6.22 mm) — not per chunk
  Q6 n chunks whose worst row (local max_violation_pair) is inside the GT apple; n violating authorized rows whose GT nearest is apple
  Q7 pre-grasp chunks where manipulated nearest centre != apple (T28 definition) and where the ball does not contain the GT apple centre"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, collections
import numpy as np
c = sys.argv[1]
V = _ag3s_workspace_path('outputs/verify/T34/p')
A = json.load(open(f"{V}/{c}_analysis.json"))
S = json.load(open(f"{V}/{c}_summary.json"))
X = json.load(open(f"{V}/{c}_analysis_extra.json"))["runs"]
DUP = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
cnt = lambda it: dict(collections.Counter(str(x) for x in it))
runs = {}
for name, r in A["runs"].items():
    ch = r["chunks"]; xs = {x["seq"]: x for x in X[name]}
    s = S["runs"][name]
    pre = [e for e in ch if e["phase"] == "pre_grasp" and not xs[e["seq"]].get("frozen")]
    fro = [e for e in ch if xs[e["seq"]].get("frozen")]
    sw = [e for e in ch if xs[e["seq"]].get("after_switch")]

    def cov(G):
        B = [xs[e["seq"]].get("ball_vs_gt_apple") for e in G]
        return {"n": len(G), "n_ball": sum(1 for b in B if b), "n_centre_inside": sum(1 for b in B if b and b["centre_inside"]),
                "n_whole_inside": sum(1 for b in B if b and b["whole_inside"])}
    held = [e for e in ch if e["n_executed"] < e["n_control"]]
    wh = [e for e in ch if (e.get("gate") or {}).get("would_hold")]

    def kinds(e, key):
        g = e.get("gate") or {}
        k = g.get(key) or g.get("kinds") or []
        return ["comms:" + e["ipc"]] if e.get("ipc") in ("timeout", "stale", "error") else k
    vr = [xs[e["seq"]].get("violating_rows") or {} for e in ch]
    rows = [row for v in vr for row in v.get("rows", [])]
    st = [xs[e["seq"]].get("stamps") or {} for e in ch]
    runs[name] = {
        "summary": s,
        "association_counts": {"decision_reason": cnt((xs[e["seq"]].get("decision") or {}).get("reason") for e in ch if "server_chunk" in e),
                               "association_outcome": cnt((xs[e["seq"]].get("association") or {}).get("outcome") for e in ch if "server_chunk" in e),
                               "anchor_source": cnt((xs[e["seq"]].get("anchor") or {}).get("source") for e in ch if "server_chunk" in e),
                               "n_association_rejected": sum(1 for e in ch if (xs[e["seq"]].get("decision") or {}).get("reason") == "association_rejected"),
                               "n_subset_kept_anchor": sum(1 for e in ch if (xs[e["seq"]].get("decision") or {}).get("reason") == "subset_kept_anchor"),
                               "switch_t": [e["t"] for e in ch if (xs[e["seq"]].get("decision") or {}).get("mode") == "switch"]},
        "ball_vs_gt_apple": {"pre_grasp_nonfrozen": cov(pre), "frozen": cov(fro), "after_switch": cov(sw)},
        "violating_rows": {"n_chunks_with_neg_row": sum(1 for v in vr if v.get("n")), "n_rows": len(rows),
                           "authorized_by_tier_gt": cnt((x["tier"], x["gt_nearest"]) for x in rows if x["authorized"]),
                           "unauthorized_by_link_gt": cnt((x["link"], x["gt_nearest"]) for x in rows if not x["authorized"]),
                           "n_chunks_attached_robot_rows_only": sum(1 for v in vr if v.get("attached_robot_rows_only"))},
        "stamps": {"stamp_mode": cnt(x.get("stamp_mode") for x in st), "camera_skew_sec_max": max((x.get("camera_skew_sec") or 0) for x in st),
                   "render_spread_sec_median": float(np.median([x["render_spread_sec"] for x in st if x.get("render_spread_sec") is not None])),
                   "render_spread_sec_max": max(x["render_spread_sec"] for x in st if x.get("render_spread_sec") is not None)},
        "hold_timeline": [{"t": e["t"], "kinds": kinds(e, "hold_kinds"), "phase": e["phase"], "latch": (e.get("grasp_latch") or {}).get("state")} for e in held],
        "would_hold_timeline": [{"t": e["t"], "kinds": kinds(e, "would_hold_kinds"), "phase": e["phase"], "latch": (e.get("grasp_latch") or {}).get("state")} for e in wh],
        "Q": {"Q4_hold_kinds": s["hold_kind_counts"], "Q4_would_hold_kinds": s["would_hold_kind_counts"], "Q4_comms_ipc_rows": s["comms_ipc_planning_rows"],
              "Q4_n_time_budget_hit": s["n_time_budget_hit"],
              "Q6_n_worst_inside_apple": s["n_worst_inside_apple"], "Q6_n_worst_inside_apple_pre_grasp": s["n_worst_inside_apple_pre_grasp"],
              "Q6_n_authorized_neg_rows_gt_apple": sum(1 for x in rows if x["authorized"] and x["gt_nearest"] == "apple"),
              "Q7_n_manip_not_apple_pre_grasp": s["n_manip_not_apple_pre_grasp"], "Q7_pre_grasp_chunks": s["n_pre_grasp_chunks"],
              "Q7_pre_grasp_nonfrozen_apple_centre_not_in_ball": cov(pre)["n"] - cov(pre)["n_centre_inside"]},
        "duplicate_of": DUP.get(name),
    }
cols = ["t", "seq", "phase", "executed", "gate_kinds", "manip_id", "manip_state", "manip_nearest", "manip_to_apple_mm", "decision_reason", "association",
        "anchor_source", "frozen", "after_switch", "ball_radius_mm", "ball_apple_dist_mm", "ball_whole_apple", "exclusion_source", "finger_model_left",
        "finger_measured_left", "worst_link", "worst_tier", "worst_clearance_mm", "worst_nearest_body", "worst_inside_apple", "n_neg_rows", "n_neg_auth_rows",
        "refined_minus_reference_deg", "latch_state", "attach", "detach", "grip_cmd_reference_min", "grip_measured_min", "stamp_mode", "camera_skew_sec",
        "srv_total_ms", "ipc"]
for name, r in A["runs"].items():
    xs = {x["seq"]: x for x in X[name]}
    tab = []
    for e in r["chunks"]:
        x = xs[e["seq"]]; g = e.get("gate") or {}; m = e.get("manipulated") or {}; w = e["worst"]; b = x.get("ball_vs_gt_apple") or {}
        ks = (g.get("would_hold_kinds") if c == "E3a" else (g.get("hold_kinds") or g.get("kinds"))) or []
        v = x.get("violating_rows") or {}
        tab.append([e["t"], e["seq"], e["phase"], e["n_executed"] == e["n_control"], ks, m.get("id"), m.get("state"), m.get("nearest_centre"), m.get("to_apple_mm"),
                    (x.get("decision") or {}).get("reason"), (x.get("association") or {}).get("outcome"), (x.get("anchor") or {}).get("source"), x.get("frozen"),
                    x.get("after_switch"), b.get("radius_mm"), b.get("apple_centre_dist_mm"), b.get("whole_inside"), (e.get("exclusion") or {}).get("source"),
                    (e.get("finger_joints") or {}).get("model_left_norm"), (e.get("finger_joints") or {}).get("measured_left_norm_planning_row"),
                    w.get("link"), w.get("tier"), None if w.get("clearance_m") is None else 1000 * w["clearance_m"], w.get("nearest_body"), w.get("inside_apple"),
                    v.get("n"), v.get("n_authorized"), e.get("refined_minus_reference_deg"), (e.get("grasp_latch") or {}).get("state"),
                    (e.get("grasp_latch") or {}).get("attach"), (e.get("grasp_latch") or {}).get("detach"), e.get("grip_l_cmd_reference_min_rows0_7"),
                    e.get("grip_l_measured_min"), (x.get("stamps") or {}).get("stamp_mode"), (x.get("stamps") or {}).get("camera_skew_sec"),
                    (e.get("server_timing_ms") or {}).get("total"), e.get("ipc")])
    runs[name]["per_chunk_columns"] = cols; runs[name]["per_chunk"] = tab
distinct = [n for n in runs if not runs[n]["duplicate_of"]]
frag = {"scripts": ["outputs/verify/T34/p/run_chain_e3.sh", "start_server_e3.sh", "run_client_e3.sh", "analyze_e3_base.py", "analyze_e3_extra.py", "summarize_e3.py",
                    "build_frag_e3.py", "dup_check.py"],
        "raw": {"analysis": f"outputs/verify/T34/p/{c}_analysis.json", "extra": f"outputs/verify/T34/p/{c}_analysis_extra.json",
                "summary": f"outputs/verify/T34/p/{c}_summary.json", "runs_dir": f"outputs/verify/T34/{c}_rec/"},
        "definitions": {"analyze": A["definitions"], "summarize": S["definitions"], "extra": json.load(open(f"{V}/{c}_analysis_extra.json"))["definitions"], "build": __doc__},
        "condition_summary_all_runs": S["condition_summary"],
        "distinct_runs": distinct,
        "condition_summary_distinct": {"n_runs": len(distinct), "n_success": sum(runs[n]["summary"]["success"] for n in distinct),
                                       "n_grasp": sum(runs[n]["summary"]["grasp"] for n in distinct)},
        "runs": runs, "mp4": sorted(r["summary"]["mp4"] for r in runs.values())}
json.dump(frag, open(f"${AG3S_PROJECT_ROOT}/outputs/verify/T34/frag/{c}.json", "w"), default=float)
print(c, "ok", len(runs), frag["condition_summary_distinct"])
