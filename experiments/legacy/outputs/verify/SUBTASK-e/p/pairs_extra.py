"""SUBTASK-e — tallies over v2/analysis.json (no new measurement except traj.npz reads):
  off_vs_on_e_exec  : per pair, first control step where executed actions differ (> 0), chunk = step // 8 + 1, max abs; for pairs
                      with an on_e release: first diff chunk - t_rel_e.
  decision_wo_to_status : on_e vs on_d, first t where the decision tuple minus to_status differs (fields DEC of analyze_e).
  first_blocked_by_region : gate blocked == 'first' chunks per side by region and by (gate action, hold).
  no_admissible_after_release : on_e runs with grounding_status no_admissible at t >= t_rel: n chunks, first t, n HOLD.
  released_table    : one row per (episode, seed) with an on_e release.
usage: pairs_extra.py ANALYSIS_JSON OUT_JSON"""
import json, sys, pathlib, collections
import numpy as np
d = json.loads(pathlib.Path(sys.argv[1]).read_text()); PR = d["per_run"]
DEC = ("latch", "label", "argmax", "blocked", "released", "released_id", "placed_seen", "manip_id", "mode", "excl_active", "excl_source", "grounding_status")
keys = sorted({k.split("/", 1)[1] for k in PR}, key=lambda s: (int(s.split("_s")[0]), int(s.split("_s")[1])))
C = lambda x: dict(sorted(collections.Counter(str(v) for v in x).items()))
out = {"definitions": __doc__}
ex = {}
for k in keys:
    a = np.load(PR["off/" + k]["run_dir"] + "/traj.npz")["executed_actions"]; b = np.load(PR["on_e/" + k]["run_dir"] + "/traj.npz")["executed_actions"]
    dif = np.abs(a.astype(np.float64) - b.astype(np.float64)).max(axis=1); s = int(np.argmax(dif > 0)) if (dif > 0).any() else None
    ex[k] = {"first_step": s, "first_chunk": None if s is None else s // 8 + 1, "max_abs": float(dif.max()),
             "t_rel_e": PR["on_e/" + k]["t_release"], "t_placed_off": PR["off/" + k]["t_placed"], "t_placed_on_e": PR["on_e/" + k]["t_placed"]}
out["off_vs_on_e_exec"] = {"per_pair": ex, "n_identical": sum(1 for v in ex.values() if v["first_step"] is None),
                           "first_chunk_minus_t_rel_e": C(v["first_chunk"] - v["t_rel_e"] for v in ex.values() if v["first_chunk"] is not None and v["t_rel_e"] is not None),
                           "n_pairs_with_release_e": sum(1 for v in ex.values() if v["t_rel_e"] is not None)}
dw = {}
for k in keys:
    be = {x["t"]: x for x in PR["on_e/" + k]["rows"]}; bd = {x["t"]: x for x in PR["on_d/" + k]["rows"]}
    t = next((t for t in sorted(set(be) & set(bd)) if any(json.dumps(be[t][f], default=str) != json.dumps(bd[t][f], default=str) for f in DEC)), None)
    dw[k] = {"t": t, "fields": None if t is None else [f for f in DEC if json.dumps(be[t][f], default=str) != json.dumps(bd[t][f], default=str)],
             "t_rel_d": PR["on_d/" + k]["t_release"], "t_gh_d": PR["on_d/" + k]["t_gate_hold_subtask_gated"]}
out["decision_wo_to_status_on_e_vs_on_d"] = {"per_pair": dw, "n_no_difference": sum(1 for v in dw.values() if v["t"] is None),
                                             "t_minus_t_gh_d": C(v["t"] - v["t_gh_d"] for v in dw.values() if v["t"] is not None and v["t_gh_d"] is not None),
                                             "fields_at_first": C(",".join(v["fields"]) for v in dw.values() if v["fields"])}
fb = {}
for sd in ("on_d", "on_e"):
    rows = [x for k in keys for x in PR[f"{sd}/{k}"]["rows"] if x["blocked"] == "first"]
    fb[sd] = {"n": len(rows), "by_region": C(x["region"] for x in rows), "by_action_hold": C(f'{x["gate_action"]}|hold={x["hold"]}' for x in rows),
              "by_latch_label": C(f'{x["latch"]}|{x["label"]}' for x in rows),
              "before_placed": [{"run": k, "t": x["t"], "latch": x["latch"], "label": x["label"], "gate_action": x["gate_action"]}
                                for k in keys for x in PR[f"{sd}/{k}"]["rows"] if x["blocked"] == "first" and x["region"] in ("before_first_held", "held_before_placed")]}
out["first_blocked_by_region"] = fb
na = []
for k in keys:
    r = PR["on_e/" + k]
    if r["t_release"] is None:
        continue
    rr = [x for x in r["rows"] if x["region"] == "from_release"]
    n = [x for x in rr if x["grounding_status"] == "no_admissible"]
    if n:
        na.append({"run": k, "t_rel": r["t_release"], "n_chunks_from_release": len(rr), "n_no_admissible": len(n), "first_t": n[0]["t"], "last_t": n[-1]["t"],
                   "n_hold_no_admissible": sum(x["hold"] for x in n), "hold_kinds": C(h for x in n for h in x["hold_kinds"]),
                   "modes": C(x["mode"] for x in n), "contiguous": [x["t"] for x in n] == list(range(n[0]["t"], n[-1]["t"] + 1))})
out["no_admissible_after_release_on_e"] = {"runs": na, "n_runs": len(na), "n_chunks": sum(x["n_no_admissible"] for x in na), "n_hold": sum(x["n_hold_no_admissible"] for x in na)}
tab = []
for k in keys:
    e, dd, o = PR["on_e/" + k], PR["on_d/" + k], PR["off/" + k]
    if e["t_release"] is None and dd["t_release"] is None:
        continue
    fr = e["regions"]["from_release"]
    tab.append({"run": k, "success_off_d_e": [o["success"], dd["success"], e["success"]], "t_placed_e": e["t_placed"], "t_rel_e": e["t_release"], "t_rel_d": dd["t_release"],
                "from_release_chunks": fr["n_chunks"], "hold_from_release_d": dd["regions"]["from_release"]["n_hold"], "hold_from_release_e": fr["n_hold"],
                "snt_chunks_e": e["snt"]["n_chunks"], "grounding_from_release_e": fr["grounding_status"],
                "clearance_min_from_release_mm_d_e": [dd["regions"]["from_release"]["clearance_min_mm"], fr["clearance_min_mm"]],
                "dL_rad_last_off_d_e": [o["pose"]["dL_rad_last"], dd["pose"]["dL_rad_last"], e["pose"]["dL_rad_last"]],
                "dEE_mm_last_off_d_e": [o["pose"]["dEE_mm_last"], dd["pose"]["dEE_mm_last"], e["pose"]["dEE_mm_last"]],
                "dL_rad_at_ref_e": e["pose"].get("dL_rad_at_ref"), "L_path_rad_from_ref_d_e": [dd["pose"].get("L_path_rad_from_ref"), e["pose"].get("L_path_rad_from_ref")]})
out["released_table"] = tab
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k not in ("released_table", "definitions")}, indent=1, default=float)[:7000])
