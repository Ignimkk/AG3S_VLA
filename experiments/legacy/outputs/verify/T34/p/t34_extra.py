"""T34 per-run extras (copy of T33/p/t33_extra.py) from p/<COND>_analysis.json (analyze_e3_base.py copy with T33 fields). usage: t33_extra.py COND -> p/<COND>_t34_extra.json
Per run:
  returned_counts          : server to.metrics.returned over paired chunks (initial / accepted / projection / best_unaccepted)
  projection               : chunks with returned == projection: t, projection record
  refined_minus_reference  : per chunk (t, deg = max over both arms, executed rows 0-7, local), plus max/median overall and in grasp->release
  hold_timeline            : E3b: chunks not fully executed, (t, kinds, latch); E3a: would_hold chunks (t, kinds)
  attach / detach t        : server grasp.attach / detach
  held                     : held record at the first attach (n spheres, radii, fit, snapshot_state)
  worst rows place window  : chunks with grasp_t <= t <= release_t (release = first control row t >= place_t with left gripper cmd >= 0.5,
                             D1 rule) — worst row link / tier / clearance / GT nearest body (and nearest excluding apple for held rows 'attached:*')
  initial_limit_overshoot  : chunks whose recorded initial_limit_overshoot has any entry > 1e-9 (G1 on the new server)
  hand / split             : n chunks manipulated.hand.occluded_by_hand True; split n_triggered / n_split counts; release split events"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, collections, pathlib
import numpy as np
COND = sys.argv[1]
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T34/p'))
A = json.loads((V / f"{COND}_analysis.json").read_text())
out = {"condition": COND, "definitions": __doc__, "runs": {}}
for name, r in A["runs"].items():
    ch = r["chunks"]; s = r["summary"]
    rows = [json.loads(l) for l in open(pathlib.Path(s["run_dir"]) / "frames.jsonl") if l.strip()]
    ctrl = [x for x in rows if x["kind"] == "control"]
    g, p = s["grasp_t"], s["place_t"]
    rel = None if p is None else next((x["t_step"] for x in ctrl if x["t_step"] >= p and x["applied_ctrl"]["gripper"][0] >= 0.5), None)
    paired = [c for c in ch if "server_chunk" in c]
    ret = collections.Counter((c.get("to_metrics") or {}).get("returned") for c in paired)
    rr = [(c["t"], c.get("refined_minus_reference_deg")) for c in ch]
    inw = [x for t, x in rr if g is not None and x is not None and g <= t <= (rel if rel is not None else (p if p is not None else 10 ** 9))]
    if COND == "E3b":
        holds = [{"t": c["t"], "kinds": ((c.get("gate") or {}).get("hold_kinds") or (c.get("gate") or {}).get("kinds") or []),
                  "latch": (c.get("grasp_latch") or {}).get("state"), "phase": c["phase"], "n_executed": c["n_executed"], "ipc": c.get("ipc"),
                  "worst_link": c["worst"].get("link"), "worst_clearance_mm": None if c["worst"].get("clearance_m") is None else 1000 * c["worst"]["clearance_m"],
                  "worst_nearest": c["worst"].get("nearest_body"), "worst_nearest_excl_apple": c["worst"].get("nearest_body_excl_apple")}
                 for c in ch if c["n_executed"] < c["n_control"]]
    else:
        holds = [{"t": c["t"], "kinds": (c.get("gate") or {}).get("would_hold_kinds") or [], "latch": (c.get("grasp_latch") or {}).get("state"), "phase": c["phase"],
                  "worst_link": c["worst"].get("link"), "worst_clearance_mm": None if c["worst"].get("clearance_m") is None else 1000 * c["worst"]["clearance_m"],
                  "worst_nearest": c["worst"].get("nearest_body"), "worst_nearest_excl_apple": c["worst"].get("nearest_body_excl_apple")}
                 for c in ch if (c.get("gate") or {}).get("would_hold")]
    att = [c for c in paired if (c.get("grasp_latch") or {}).get("attach")]
    held0 = (att[0].get("held") or {}) if att else None
    wp = []
    if g is not None:
        hi = rel if rel is not None else (p if p is not None else 600)
        for c in ch:
            if g <= c["t"] <= hi:
                w = c["worst"]
                wp.append({"t": c["t"], "link": w.get("link"), "tier": w.get("tier"), "clearance_mm": None if w.get("clearance_m") is None else 1000 * w["clearance_m"],
                           "nearest": w.get("nearest_body"), "nearest_excl_apple": w.get("nearest_body_excl_apple"),
                           "nearest_excl_apple_mm": None if not w.get("per_body_surface_mm") else w["per_body_surface_mm"].get(w.get("nearest_body_excl_apple")),
                           "held_row": str(w.get("link") or "").startswith("attached")})
    ilo = []
    for c in paired:
        o = (c.get("to_metrics") or {}).get("initial_limit_overshoot") or {}
        if isinstance(o, dict) and any((v or 0) > 1e-9 for v in o.values()):
            ilo.append({"t": c["t"], "overshoot": o, "returned": c["to_metrics"].get("returned")})
    me = [c.get("manip_extra") or {} for c in paired]
    out["runs"][name] = {
        "success": s["success"], "grasp_t": g, "place_t": p, "release_t": rel, "apple_dz_max_mm": s["apple_dz_max_mm"],
        "returned_counts": dict(ret), "projection": [{"t": c["t"], "projection": c["to_metrics"].get("projection")} for c in paired if (c.get("to_metrics") or {}).get("returned") == "projection"],
        "refined_minus_reference_per_chunk": rr,
        "refined_minus_reference_summary": {"max": max(x for _, x in rr if x is not None), "median": float(np.median([x for _, x in rr if x is not None])),
                                            "grasp_to_release_max": max(inw) if inw else None, "grasp_to_release_median": float(np.median(inw)) if inw else None},
        "hold_timeline": holds, "n_hold_or_would_hold": len(holds), "kind_counts": dict(collections.Counter(k for h in holds for k in h["kinds"])),
        "comms_hold_from_log": s["client_log"].get("comms_hold_from_log"), "comms_ipc_planning_rows": s["comms_ipc_planning_rows"],
        "attach_t": [c["t"] for c in att], "detach_t": [c["t"] for c in paired if (c.get("grasp_latch") or {}).get("detach")],
        "held_at_first_attach": held0,
        "revoke_t": [c["t"] for c in paired if (c.get("grasp_latch") or {}).get("detach") and "attach_revoked" in str((c.get("grasp_latch") or {}).get("note"))],
        "revoke_notes": [(c.get("grasp_latch") or {}).get("note") for c in paired if (c.get("grasp_latch") or {}).get("detach") and "attach_revoked" in str((c.get("grasp_latch") or {}).get("note"))],
        "hold_held_table": [h for h in holds if str(h.get("worst_link") or "").startswith("attached") and h.get("worst_nearest_excl_apple") == "table"],
        "hold_held_rows": [h for h in holds if str(h.get("worst_link") or "").startswith("attached")],
        "worst_rows_grasp_to_release": wp, "worst_nearest_counts_grasp_to_release": dict(collections.Counter(w["nearest"] for w in wp)),
        "worst_nearest_excl_apple_counts_held_rows": dict(collections.Counter(w["nearest_excl_apple"] for w in wp if w["held_row"])),
        "worst_link_counts_grasp_to_release": dict(collections.Counter(w["link"] for w in wp)),
        "initial_limit_overshoot_chunks": ilo,
        "n_occluded_by_hand": sum(1 for x in me if (x.get("hand") or {}).get("occluded_by_hand")),
        "split_triggered_chunks": sum(1 for x in me if (x.get("split_n_triggered") or 0) > 0), "split_accepted_chunks": sum(1 for x in me if (x.get("split_n_split") or 0) > 0),
        "release_split_events": [x.get("split_release") for x in me if x.get("split_release")],
        "switch_chunks_t": [c["t"] for c, x in zip(paired, me) if x.get("switched_from") is not None],
    }
    o = out["runs"][name]
    print(name, o["success"], g, p, rel, o["returned_counts"], "rr", {k: round(v, 2) if v else v for k, v in o["refined_minus_reference_summary"].items()},
          "holds", o["n_hold_or_would_hold"], o["kind_counts"], "att", o["attach_t"], "det", o["detach_t"], "rev", o["revoke_t"], "held-table holds", len(o["hold_held_table"]), "ilo", len(ilo), flush=True)
(V / f"{COND}_t34_extra.json").write_text(json.dumps(out, indent=1, default=float))
