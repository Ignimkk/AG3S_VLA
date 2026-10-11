"""SUBTASK-d V2 — tallies over p/analyze_v2.py output (no new measurement): divergence chunk / latch / gate state per pair,
gate-blocked chunks before latch PLACED (by latch state, label, run, grasp outcome), per-pair table.
usage: summarize_v2.py ANALYSIS_JSON OUT_JSON"""
import json, sys, collections, pathlib
import numpy as np
d = json.loads(pathlib.Path(sys.argv[1]).read_text()); out = {}
C = collections.Counter
pairs = d["pairs"]; runs = d["runs"]
div = {"n_pairs": len(pairs), "n_no_executed_difference": 0, "first_diff_chunk": [], "latch_at_first_diff": C(), "on_state_at_first_diff": C(),
       "first_diff_chunk_minus_on_t_placed": [], "first_decision_diff_equals_first_action_diff_chunk": 0, "first_decision_diff_before": 0}
for p in pairs:
    dv = p["divergence"]
    if dv["first_step_gt0"] is None:
        div["n_no_executed_difference"] += 1; continue
    a = dv["at_gt0"]; div["first_diff_chunk"].append(a["chunk_t"])
    div["latch_at_first_diff"][f'off={a["off"]["latch"]}|on={a["on"]["latch"]}'] += 1
    div["on_state_at_first_diff"][f'blocked={a["on"]["blocked"]}|released={a["on"]["released"]}|label={a["on"]["label"]}'] += 1
    if a["on_t_placed"] is not None:
        div["first_diff_chunk_minus_on_t_placed"].append(a["chunk_t"] - a["on_t_placed"])
    fd = dv.get("first_decision_diff_t")
    if fd == a["chunk_t"]:
        div["first_decision_diff_equals_first_action_diff_chunk"] += 1
    elif fd is not None and fd < a["chunk_t"]:
        div["first_decision_diff_before"] += 1
div["latch_at_first_diff"] = dict(div["latch_at_first_diff"]); div["on_state_at_first_diff"] = dict(div["on_state_at_first_diff"])
div["first_diff_chunk_stats"] = {"min": min(div["first_diff_chunk"]), "median": float(np.median(div["first_diff_chunk"])), "max": max(div["first_diff_chunk"])}
out["divergence"] = div
on = [r for r in runs.values() if r["side"] == "on"]
bb = {"n_runs_with_block_before_placed": 0, "n_chunks": 0, "by_kind": C(), "by_latch": C(), "by_label": C(), "by_latch_label": C(),
      "runs": [], "n_runs_block_before_first_held": 0}
for r in on:
    b = r["before_placed"]["blocked"]
    if not b:
        continue
    bb["n_runs_with_block_before_placed"] += 1; bb["n_chunks"] += len(b)
    for x in b:
        bb["by_kind"][x["blocked"]] += 1; bb["by_latch"][x["latch"]] += 1; bb["by_label"][str(x["label"])] += 1
        bb["by_latch_label"][f'{x["latch"]}|{x["label"]}'] += 1
    tH = r["t_first_held"]
    pre = [x for x in b if tH is None or x["t"] < tH]
    if pre:
        bb["n_runs_block_before_first_held"] += 1
    bb["runs"].append({"run": r["run_dir"].split("/")[-1], "n_blocked": len(b), "first_t": b[0]["t"], "t_first_held": tH, "t_placed": r["t_placed"],
                       "grasp": r["grasp"], "success": r["success"], "kinds": dict(C(x["blocked"] for x in b))})
for k in ("by_kind", "by_latch", "by_label", "by_latch_label"):
    bb[k] = dict(bb[k])
out["gate_blocked_before_placed"] = bb
out["first_adoption_t"] = {s: dict(C(r["first_adoption_t"] for r in runs.values() if r["side"] == s)) for s in ("off", "on")}
out["first_adoption_same_t_all_pairs"] = all(runs[p["off_run"]]["first_adoption_t"] == runs[p["on_run"]]["first_adoption_t"] for p in pairs)
out["first_adoption_same_manip_id_all_pairs"] = all(
    next((c["manip_id"] for c in runs[p["off_run"]]["chunks"] if c["mode"] == "first"), None) == next((c["manip_id"] for c in runs[p["on_run"]]["chunks"] if c["mode"] == "first"), None) for p in pairs)
# pick stage (before first 'held'): decisions identical?
same = 0; tot = 0
for p in pairs:
    ro, rn = runs[p["off_run"]], runs[p["on_run"]]
    tH = min([t for t in (ro["t_first_held"], rn["t_first_held"]) if t is not None], default=76)
    fo = {c["t"]: c for c in ro["chunks"]}; fn = {c["t"]: c for c in rn["chunks"]}
    ts = [t for t in sorted(set(fo) & set(fn)) if t < tH]
    tot += 1
    if all((fo[t]["manip_id"], fo[t]["mode"], fo[t]["excl_active"]) == (fn[t]["manip_id"], fn[t]["mode"], fn[t]["excl_active"]) for t in ts):
        same += 1
out["pairs_decisions_identical_before_first_held"] = {"n_same": same, "n_pairs": tot,
    "definition": "manipulated id, decision mode, exclusion active equal at every chunk t < min(first 'held' chunk of the two runs; 76 if neither)"}
out["pair_table"] = [{"episode": p["episode"], "seed": p["seed"], "off": p["off"], "on": p["on"],
                      "off_t_placed": runs[p["off_run"]]["t_placed"], "on_t_placed": runs[p["on_run"]]["t_placed"],
                      "first_exec_diff_step": p["divergence"]["first_step_gt0"], "first_decision_diff_t": p["divergence"].get("first_decision_diff_t"),
                      "on_release_t": runs[p["on_run"]]["after_placed"]["first_released_t"],
                      "off_after_placed_excl": [runs[p["off_run"]]["after_placed"]["n_excl_active"], runs[p["off_run"]]["after_placed"]["n_chunks"]],
                      "on_after_placed_excl": [runs[p["on_run"]]["after_placed"]["n_excl_active"], runs[p["on_run"]]["after_placed"]["n_chunks"]],
                      "off_hold": runs[p["off_run"]]["n_hold_chunks"], "on_hold": runs[p["on_run"]]["n_hold_chunks"]} for p in pairs]
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: v for k, v in out.items() if k != "pair_table"}, indent=1, default=float)[:6000])
