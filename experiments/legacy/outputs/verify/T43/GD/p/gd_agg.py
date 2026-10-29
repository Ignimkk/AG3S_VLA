"""T43GD aggregation (verifier, CPU only, numbers only). usage: gd_agg.py OUT_JSON
Inputs: GD/analysis/analysis.json (analyze_gd.py: arms V7 = gate OFF, V7g = gate ON), GD/z4.json (gd_z4.py), GD/p/ct_runs.json if present (contact-inclusive rule from outputs/verify/T43/CT).
Pairs: (stage, episode, seed). Definitions:
  S / G / P (old)  : success / grasp / place by analyze_e0 (apple z >= z0 + 50 mm at some planning row; apple centre in the crate place box at the LAST planning row).
  S / P (Z4)       : z4 release rule (gd_z4.py docstring): released = last apple-hand contact row exists before the run's end and the apple stays > d(grasp) + 30 mm away from the finger midpoint to the end; success_rel = grasp and released and in_crate_end.
  run mode         : check_run mode: full (600 control rows) | obstacle_stop (client ended the run on '[SAFETY] obstacle stop').
  HOLD chunk       : client planning row whose control rows were not all executed; hold kinds from client gate.hold_kinds (one count per kind per HOLD chunk).
  latch / label / grounding_status / mode / excl_active : server summary per chunk (grasp.state / subtask.label (confirmed) / grounding_status / admissibility.decision.mode / exclusion.active).
  t_placed         : (control step t_step of) the first server chunk with latch 'placed'; after PLACED = t >= t_placed. t_first_held likewise with 'held'.
  subtask_no_target: client planning-row reason kind 'subtask_no_target' (chunks); grounding status of that reason from the server verdict reason evidence (gs).
  target switches  : after PLACED, chunks with admissibility mode 'switch' ('first' counted separately); carve chunks = after PLACED chunks with exclusion.active True.
  ready posture    : SUBTASK-e definition for the active arm: ready pose = traj.npz measured_qpos[0]; dJ = || q[s, arm] - q[0, arm] ||_2 rad; dEE = || FK link_<side>_arm_6 (s) - (0) || mm. Reported at the last control step, and at the control step of the first PLACED chunk (t_placed).
  label time line  : confirmed subtask label per chunk (None before confirmation)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, sys
import numpy as np
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); GD = T / "GD"


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def cnt(it):
    c = {}
    for x in it:
        c[str(x)] = c.get(str(x), 0) + 1
    return dict(sorted(c.items()))


def addc(ds):
    o = {}
    for d in ds:
        for k, v in d.items():
            o[k] = o.get(k, 0) + v
    return dict(sorted(o.items()))


def stat(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def per_run(key, r, z):
    arm, stage, name = key.split("/")
    sc = r.get("server_chunks") or []
    ch = {c["t"]: c for c in r["chunks"]}
    sv = {c["t"]: c for c in sc}
    tP = next((c["t"] for c in sc if c["latch"] == "placed"), None)
    tH = next((c["t"] for c in sc if c["latch"] == "held"), None)
    def region(t):
        if tH is None or t < tH:
            return "before_first_held"
        if tP is None or t < tP:
            return "held_before_placed"
        return "after_placed"
    hold = [c for c in r["chunks"] if c["hold"]]
    after = [c for c in sc if tP is not None and c["t"] >= tP]
    snt = [c for c in r["chunks"] if "subtask_no_target" in (c.get("verdict_reason_kinds") or [])]
    snt_gs = cnt((next((x["gs"] for x in sv[c["t"]]["verdict_reasons"] if x["kind"] == "subtask_no_target"), None)) for c in snt if c["t"] in sv)
    lab = [(c["t"], c["subtask"]["label"]) for c in sc]
    first_lab = {l: next((t for t, x in lab if x == l), None) for l in ("pick", "place", "home")}
    ser = r["pose"]; sref = tP   # t_placed is a control step (server chunk t_step, multiple of 8)
    dJ_ref = dEE_ref = None
    if sref is not None and sref // 8 < len(ser["series_dJ_rad"]):
        dJ_ref = ser["series_dJ_rad"][sref // 8]; dEE_ref = ser["series_dEE_mm"][sref // 8]
    o = {"key": key, "arm": arm, "stage": stage, "episode": int(r["episode"]), "seed": int(name.split("_s")[-1]),
         "success": bool(r["success"]), "grasp": bool(r["grasp"]), "place": bool(r["place"]),
         "success_rel": z["success_rel"], "place_rel": z["place_rel"], "released": z["released"], "release_t": z["release_t"], "release_mode": z["release_mode"],
         "mode": (r.get("check") or {}).get("mode"), "n_planning": r["n_planning"], "n_control": r["n_control"],
         "n_hold": len(hold), "hold_t": [c["t"] for c in hold], "hold_kinds": cnt(k for c in hold for k in (c["hold_kinds"] or [])),
         "hold_by_region": cnt(region(c["t"]) for c in hold if c["t"] in sv), "hold_by_latch": cnt(sv[c["t"]]["latch"] for c in hold if c["t"] in sv),
         "hold_by_label": cnt(sv[c["t"]]["subtask"]["label"] for c in hold if c["t"] in sv), "hold_by_grounding": cnt(sv[c["t"]]["grounding_status"] for c in hold if c["t"] in sv),
         "trailing_hold": r["trailing_hold_chunks"], "longest_hold_run": r["longest_hold_run_chunks"],
         "t_first_held": tH, "t_placed": tP, "first_label_t": first_lab, "label_counts": cnt(l for _, l in lab),
         "label_seq": [l for _, l in lab], "latch_seq": [c["latch"] for c in sc], "hold_seq": [bool(ch[c["t"]]["hold"]) if c["t"] in ch else None for c in sc],
         "verdict_seq": [("hold" if ch[c["t"]]["hold"] else "execute") if c["t"] in ch else None for c in sc],
         "follow_to_seq": [(c["follow_to"] or {}).get("decision") for c in sc], "t_seq": [c["t"] for c in sc],
         "grounding_seq": [c["grounding_status"] for c in sc], "snt_seq": [bool(c["t"] in ch and "subtask_no_target" in (ch[c["t"]].get("verdict_reason_kinds") or [])) for c in sc],
         "mode_seq": [c["mode"] for c in sc], "excl_seq": [bool(c["excl_active"]) for c in sc],
         "home_after_placed_delay": (first_lab["home"] - tP) if (tP is not None and first_lab["home"] is not None and first_lab["home"] >= tP) else None,
         "grounding_status_counts": cnt(c["grounding_status"] for c in sc),
         "grounding_status_after_placed": cnt(c["grounding_status"] for c in after),
         "n_released_flag": sum(bool(c["subtask"]["released"]) for c in sc), "n_blocked_first": sum(c["subtask"]["blocked"] == "first" for c in sc), "n_blocked_switch": sum(c["subtask"]["blocked"] == "switch" for c in sc),
         "n_would_block_first": sum(c["subtask"]["would_block"] == "first" for c in sc), "n_would_block_switch": sum(c["subtask"]["would_block"] == "switch" for c in sc),
         "reason_kind_counts": cnt(k for c in r["chunks"] for k in (c.get("verdict_reason_kinds") or [])),
         "n_subtask_no_target": len(snt), "subtask_no_target_gs": snt_gs, "subtask_no_target_t": [c["t"] for c in snt],
         "after_placed": {"n_chunks": len(after), "mode_counts": cnt(c["mode"] for c in after), "n_switch": sum(c["mode"] == "switch" for c in after),
                          "n_first": sum(c["mode"] == "first" for c in after), "n_carve": sum(c["excl_active"] for c in after),
                          "n_has_target": sum(bool(c["has_target"]) for c in after), "n_manip_id_not_none": sum(c["manipulated_id"] is not None for c in after),
                          "ids_manip": sorted({c["manipulated_id"] for c in after if c["manipulated_id"] is not None}),
                          "n_hold": sum(1 for c in hold if tP is not None and c["t"] >= tP)},
         "n_switch_total": sum(c["mode"] == "switch" for c in sc), "n_first_total": sum(c["mode"] == "first" for c in sc),
         "n_carve_total": sum(c["excl_active"] for c in sc),
         "pose": {"dJ_rad_last": ser["dJ_rad_last"], "dEE_mm_last": ser["dEE_mm_last"], 
                  "dJ_rad_at_placed": dJ_ref, "dEE_mm_at_placed": dEE_ref, "side": ser["side"], "n_ctrl": ser["n_ctrl"],
                  "dJ_rad_min_from_placed": (min(ser["series_dJ_rad"][sref // 8:]) if (sref is not None and sref // 8 < len(ser["series_dJ_rad"])) else None),
                  "dEE_mm_min_from_placed": (min(ser["series_dEE_mm"][sref // 8:]) if (sref is not None and sref // 8 < len(ser["series_dEE_mm"])) else None)},
         "hurdle_contact_steps": ((r.get("check") or {}).get("npz_obstacle_safety") or {}).get("contact_steps"),
         "hurdle_min_robot_clearance_m": ((r.get("check") or {}).get("npz_obstacle_safety") or {}).get("min_robot_clearance_m"),
         "n_chunks_server": len(sc), "n_budget_hit": r.get("time_budget_hit_count")}
    return o


def arm_sum(rs):
    pl = lambda f: sum(bool(r[f]) for r in rs)
    return {"n": len(rs), "success": pl("success"), "grasp": pl("grasp"), "place": pl("place"), "success_rel": pl("success_rel"), "place_rel": pl("place_rel"), "released": pl("released"),
            "success_ct": pl("success_ct"), "n_pre_release_contact": pl("pre_release_contact"), "n_post_release_contact": pl("post_release_contact"),
            "mode_counts": cnt(r["mode"] for r in rs),
            "n_hold_total": sum(r["n_hold"] for r in rs), "n_runs_with_hold": sum(1 for r in rs if r["n_hold"]),
            "hold_kinds": addc(r["hold_kinds"] for r in rs), "hold_by_region": addc(r["hold_by_region"] for r in rs), "hold_by_latch": addc(r["hold_by_latch"] for r in rs),
            "hold_by_label": addc(r["hold_by_label"] for r in rs), "hold_by_grounding": addc(r["hold_by_grounding"] for r in rs),
            "n_hold_after_placed": sum(r["after_placed"]["n_hold"] for r in rs), "n_runs_trailing_hold": sum(1 for r in rs if r["trailing_hold"]),
            "n_chunks": sum(r["n_chunks_server"] for r in rs),
            "label_counts": addc(r["label_counts"] for r in rs), "n_runs_label_place": sum(r["first_label_t"]["place"] is not None for r in rs), "n_runs_label_home": sum(r["first_label_t"]["home"] is not None for r in rs),
            "t_first_label_place": stat([r["first_label_t"]["place"] for r in rs]), "t_first_label_home": stat([r["first_label_t"]["home"] for r in rs]),
            "t_placed": stat([r["t_placed"] for r in rs]), "home_after_placed_delay": stat([r["home_after_placed_delay"] for r in rs]),
            "grounding_status_counts": addc(r["grounding_status_counts"] for r in rs), "grounding_status_after_placed": addc(r["grounding_status_after_placed"] for r in rs),
            "n_released_flag_chunks": sum(r["n_released_flag"] for r in rs), "n_runs_released_flag": sum(1 for r in rs if r["n_released_flag"]),
            "n_blocked_first": sum(r["n_blocked_first"] for r in rs), "n_blocked_switch": sum(r["n_blocked_switch"] for r in rs),
            "n_would_block_first": sum(r["n_would_block_first"] for r in rs), "n_would_block_switch": sum(r["n_would_block_switch"] for r in rs),
            "reason_kind_counts": addc(r["reason_kind_counts"] for r in rs), "n_subtask_no_target": sum(r["n_subtask_no_target"] for r in rs),
            "n_runs_subtask_no_target": sum(1 for r in rs if r["n_subtask_no_target"]), "subtask_no_target_gs": addc(r["subtask_no_target_gs"] for r in rs),
            "after_placed": {"n_runs_placed": sum(r["t_placed"] is not None for r in rs), "n_chunks": sum(r["after_placed"]["n_chunks"] for r in rs),
                             "mode_counts": addc(r["after_placed"]["mode_counts"] for r in rs), "n_switch": sum(r["after_placed"]["n_switch"] for r in rs), "n_first": sum(r["after_placed"]["n_first"] for r in rs),
                             "n_carve": sum(r["after_placed"]["n_carve"] for r in rs), "n_runs_carve": sum(1 for r in rs if r["after_placed"]["n_carve"]),
                             "n_has_target": sum(r["after_placed"]["n_has_target"] for r in rs), "n_manip_id_not_none": sum(r["after_placed"]["n_manip_id_not_none"] for r in rs)},
            "n_switch_total": sum(r["n_switch_total"] for r in rs), "n_first_total": sum(r["n_first_total"] for r in rs), "n_carve_total": sum(r["n_carve_total"] for r in rs),
            "pose": {k: stat([r["pose"][k] for r in rs]) for k in ("dJ_rad_last", "dEE_mm_last", "dJ_rad_at_placed", "dEE_mm_at_placed", "dJ_rad_min_from_placed", "dEE_mm_min_from_placed")},
            "pose_full_runs": {k: stat([r["pose"][k] for r in rs if r["mode"] == "full"]) for k in ("dJ_rad_last", "dEE_mm_last")},
            "hurdle_contact_steps": stat([r["hurdle_contact_steps"] for r in rs if r["stage"] == "H"]), "n_runs_hurdle_contact": sum(1 for r in rs if r["stage"] == "H" and (r["hurdle_contact_steps"] or 0) > 0)}


def pair(R, A, B, f, stage):
    ka = {(o["stage"], o["episode"], o["seed"]): o for o in R if o["arm"] == A and (stage == "ALL" or o["stage"] == stage)}
    kb = {(o["stage"], o["episode"], o["seed"]): o for o in R if o["arm"] == B and (stage == "ALL" or o["stage"] == stage)}
    keys = sorted(set(ka) & set(kb))
    both = sum(ka[k][f] and kb[k][f] for k in keys); ao = sum(ka[k][f] and not kb[k][f] for k in keys); bo = sum(kb[k][f] and not ka[k][f] for k in keys)
    return {"n_pairs": len(keys), f"{A}_total": both + ao, f"{B}_total": both + bo, "both": both, f"{A}_only": ao, f"{B}_only": bo, "neither": len(keys) - both - ao - bo, "mcnemar_exact_p": mcnemar(ao, bo),
            f"{A}_only_keys": [f"{k[0]}:{k[1]}_s{k[2]}" for k in keys if ka[k][f] and not kb[k][f]], f"{B}_only_keys": [f"{k[0]}:{k[1]}_s{k[2]}" for k in keys if kb[k][f] and not ka[k][f]]}


def first_diff(a, b):
    """first chunk t (index in the common t list) where (hold, label, latch, grounding_status) differ between the two runs"""
    return None


if __name__ == "__main__":
    A = json.load(open(GD / "analysis/analysis.json"))["runs"]; Z = json.load(open(GD / "z4.json"))["runs"]
    CT = json.load(open(GD / "ct_runs.json")) if (GD / "ct_runs.json").exists() else None
    runs = [per_run(k, A[k], Z[k]) for k in sorted(A) if k in Z]
    for o in runs:
        c = CT.get(o["key"]) if CT is not None else None
        o["success_ct"] = None if c is None else c["success_ct"]; o["pre_release_contact"] = None if c is None else c["pre_release_contact"]
        o["post_release_contact"] = None if c is None else c["post_release_contact"]
        o["contacts"] = None if c is None else [{k: e[k] for k in ("body", "kind", "phase", "first_t", "n_rows", "max_pen_mm")} for e in c["contacts"]]
    by = {"V7": [o for o in runs if o["arm"] == "V7"], "V7g": [o for o in runs if o["arm"] == "V7g"]}
    keyset = lambda a: {(o["stage"], o["episode"], o["seed"]) for o in by[a]}
    common = keyset("V7") & keyset("V7g")
    byc = {a: [o for o in by[a] if (o["stage"], o["episode"], o["seed"]) in common] for a in by}
    res = {"definitions": __doc__, "n_runs": {a: len(v) for a, v in by.items()}, "n_pairs": len(common),
           "arms": {a: {"ALL": arm_sum(byc[a]), "H": arm_sum([o for o in byc[a] if o["stage"] == "H"]), "B": arm_sum([o for o in byc[a] if o["stage"] == "B"])} for a in byc}}
    res["pairs"] = {st: {f: pair(runs, "V7", "V7g", f, st) for f in ("success", "grasp", "place", "success_rel", "place_rel", "released", "success_ct")} for st in ("H", "B", "ALL")}
    res["runs"] = {o["key"]: o for o in runs}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
    for a in res["arms"]:
        s = res["arms"][a]["ALL"]
        print(a, {k: s[k] for k in ("n", "success", "grasp", "place", "success_rel", "place_rel", "n_hold_total", "n_subtask_no_target", "n_released_flag_chunks")}, s["grounding_status_counts"], s["after_placed"])
