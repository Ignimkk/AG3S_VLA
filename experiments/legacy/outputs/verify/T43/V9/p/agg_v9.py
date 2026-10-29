"""T43 V9 aggregation (verifier, CPU only, numbers only). usage: agg_v8.py OUT_JSON
Adapted from V8/p/agg_v7g.py (HOLD / follow_to definitions unchanged), arms V8 (V8/analysis/analysis.json = 19 runs; H runs identical to CT/raw/analysis_v7g.json on success/grasp/place/hold/solve_ms, replays CT/replay) and V9 (V9/analysis/analysis.json, replays V9/replay).
Pairing: (stage, episode, seed).
 HOLD chunk       : client planning row with hold flag (analysis chunks[].hold).
 binding category : analysis server_chunks[].binding.category (waypoint vs path rows, the smaller clearance); here relabelled by the binding link name:
                    link starts with 'ee_finger' -> 'finger', category 'held' -> 'held', every other robot row -> 'robot_other'. (V9 has 270 robot spheres, 224 of them finger body spheres; V8 122, 76 finger.)
 target           : chunks[].worst.obstacle of the client planning row: destination | hurdle-named | None.
 phase            : pre_pick (t < attach_t) | release_after (t >= detach_t) | crate_above (attached, apple over the crate) | carry. (same as agg_v7g)
 V9-only HOLD     : a HOLD chunk of the V9 run at a planning step t at which the paired V8 run has no HOLD chunk (and vice versa).
 follow_to        : server verdict.follow_to per chunk: decision follow_to (executed) | hold with refused reason | not_needed. True distances from the replay npz (apple_crate_gap, hand_crate_gap) over control rows t..t+7.
 TO timing        : server chunk to.solve_ms and verdict.timing_ms.trajopt / total (run's first chunk excluded, analyze_t39 rule), time_budget_hit_count (to.time_budget_hit), client round trip timing_ms.policy_infer,
                    share of round trips > CHUNK_PERIOD_MS (533.3 ms); pooled over all runs of the arm and over the 14 H / 5 B runs separately; percentile = numpy linear.
 finger-hurdle contacts : CT replays obstacle_contact_rows (MuJoCo contacts of the active hand's finger bodies with the hurdle geoms), phase before_grasp | carry | after_release (agg_ct rules); MJCF body ee_finger_l2 = URDF ee_finger_l1 (T11 alias)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, math, collections
import numpy as np
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); V = T / "V9"
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
AN = {"V8": json.load(open(T / "V8/analysis/analysis.json"))["runs"], "V9": json.load(open(V / "analysis/analysis.json"))["runs"]}
REPLAY = {"V8": T / "V8/replay", "V9": V / "replay"}
ARMS = ("V8", "V9")
CHUNK_PERIOD_MS = 533.3


def keyof(k):
    arm, st, nm = k.split("/")
    return (st, int(nm.split("_ep")[1].split("_s")[0]), int(nm.split("_s")[-1]))


def runs_of(arm):
    return {keyof(k): (k, r) for k, r in AN[arm].items() if k.startswith(arm + "/")}


def attach_detach(r):
    sc = r.get("server_chunks") or []
    att = det = None
    for c in sc:
        if c["attached"] and att is None:
            att = c["t"]
        if att is not None and det is None and c["grasp_state"] == "placed" and not c["attached"]:
            det = c["t"]; break
    return att, det


def plan_rows(r):
    out = {}
    for l in open(pathlib.Path(r["run_dir"]) / "frames.jsonl"):
        if l.strip():
            x = json.loads(l)
            if x["kind"] == "planning":
                out[x["seq"]] = x
    return out


def qmat(q):
    import mujoco
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def over_crate(p):
    op = p["object_poses"]
    loc = qmat(op["crate"]["quat"]).T @ (np.asarray(op["apple"]["pos"], float) - np.asarray(op["crate"]["pos"], float))
    return bool(abs(loc[0]) < 0.077 and abs(loc[1]) < 0.137)


def hurdle_min_mm(r, t0):
    h = r.get("hurdle")
    if not h:
        return None
    ser = h["series"]; ts = np.asarray(ser["t"]); sel = (ts >= t0) & (ts <= t0 + 7)
    if not sel.any():
        return None
    return 1e3 * float(min(np.asarray(ser[k], float)[sel].min() for k in ("Af", "Ae", "B", "C")))


def cat_of(s):
    if s is None:
        return None
    b = s["binding"]; cat = b.get("category"); link = b.get("link") or ""
    if cat == "held":
        return "held"
    return "finger" if str(link).startswith("ee_finger") else "robot_other"


def phase_of(r, att, det, t, plan, seq):
    if att is None or t < att:
        return "pre_pick"
    if det is not None and t >= det:
        return "release_after"
    return "crate_above" if over_crate(plan[seq]) else "carry"


def hold_breakdown(arm):
    out = {"per_run": {}, "cells": collections.Counter(), "by_category": collections.Counter(), "by_target": collections.Counter(), "by_phase": collections.Counter(),
           "by_binding_link": collections.Counter(), "total": 0, "trailing": 0, "runs_with_hold": 0, "runs_with_trailing": 0, "n_none_hurdle_le_50mm": 0,
           "by_stage_total": collections.Counter(), "hold_kinds": collections.Counter(), "by_wp_block": collections.Counter()}
    for key, (k, r) in sorted(runs_of(arm).items()):
        att, det = attach_detach(r); plan = plan_rows(r); sc = {c["seq"]: c for c in r["server_chunks"]}
        holds = [c for c in r["chunks"] if c["hold"]]
        rr = {"n_hold": len(holds), "trailing": r["trailing_hold_chunks"], "longest_run": r["longest_hold_run_chunks"], "attach_t": att, "detach_t": det, "chunks": []}
        for c in holds:
            s = sc.get(c["seq"]); t = c["t"]
            cat = cat_of(s); link = (s["binding"]["link"] if s else None)
            tgt = (c.get("worst") or {}).get("obstacle")
            ph = phase_of(r, att, det, t, plan, c["seq"])
            hm = hurdle_min_mm(r, t)
            near = bool(tgt is None and hm is not None and hm <= 50.0)
            out["cells"][f"{cat}|{tgt}|{ph}"] += 1; out["by_category"][str(cat)] += 1; out["by_target"][str(tgt)] += 1; out["by_phase"][ph] += 1
            out["by_binding_link"][str(link)] += 1; out["by_wp_block"][str((s or {}).get("wp", {}).get("block"))] += 1
            out["n_none_hurdle_le_50mm"] += near
            for kd in c.get("hold_kinds") or []:
                out["hold_kinds"][kd] += 1
            rr["chunks"].append({"seq": c["seq"], "t": t, "category": cat, "binding_link": link, "binding_rows": (s["binding"]["rows"] if s else None), "target": tgt, "phase": ph,
                                 "worst_link": (c.get("worst") or {}).get("link"), "hurdle_min_mm": hm, "hold_kinds": c.get("hold_kinds"),
                                 "binding_clearance_mm": (s["binding"]["clearance_mm"] if s else None),
                                 "clearance_mm": None if (c.get("worst") or {}).get("clearance_m") is None else 1e3 * c["worst"]["clearance_m"]})
        out["per_run"][f"{key[0]}:{key[1]}_s{key[2]}"] = rr
        out["total"] += len(holds); out["trailing"] += r["trailing_hold_chunks"]; out["runs_with_hold"] += bool(holds); out["runs_with_trailing"] += r["trailing_hold_chunks"] > 0
        out["by_stage_total"][key[0]] += len(holds)
    for f in ("cells", "by_category", "by_target", "by_phase", "by_stage_total", "hold_kinds", "by_binding_link", "by_wp_block"):
        out[f] = dict(sorted(out[f].items()))
    return out


def pair_holds(hb):
    """HOLD chunks that exist only in one arm of a (stage, ep, seed) pair, matched on the planning step t."""
    out = {"V9_only": [], "V8_only": [], "both_n": 0, "pairs": 0}
    for run in sorted(set(hb["V8"]["per_run"]) & set(hb["V9"]["per_run"])):
        a = {c["t"]: c for c in hb["V8"]["per_run"][run]["chunks"]}; b = {c["t"]: c for c in hb["V9"]["per_run"][run]["chunks"]}
        out["pairs"] += 1; out["both_n"] += len(set(a) & set(b))
        for t in sorted(set(b) - set(a)):
            out["V9_only"].append(dict(b[t], run=run))
        for t in sorted(set(a) - set(b)):
            out["V8_only"].append(dict(a[t], run=run))
    for f in ("V9_only", "V8_only"):
        L = out[f]
        out[f + "_n"] = len(L)
        out[f + "_by_category"] = dict(collections.Counter(str(x["category"]) for x in L)); out[f + "_by_target"] = dict(collections.Counter(str(x["target"]) for x in L))
        out[f + "_by_phase"] = dict(collections.Counter(x["phase"] for x in L)); out[f + "_by_binding_link"] = dict(collections.Counter(str(x["binding_link"]) for x in L))
    return out


def follow_to(arm):
    out = {"per_run": {}, "decision_counts": collections.Counter(), "refused_counts": collections.Counter(), "executed": []}
    for key, (k, r) in sorted(runs_of(arm).items()):
        sc = r.get("server_chunks") or []
        rep = np.load(REPLAY[arm] / f"{k.split('/')[0]}_{key[0]}_{k.split('/')[2]}.npz") if (REPLAY[arm] / f"{k.split('/')[0]}_{key[0]}_{k.split('/')[2]}.npz").exists() else None
        t_arr = rep["t"] if rep is not None else []; plan = plan_rows(r); att, det = attach_detach(r)
        pr = {"decisions": collections.Counter(), "refused": collections.Counter(), "executed_t": []}
        for c in sc:
            f = c.get("follow_to") or {}
            dcs = f.get("decision"); pr["decisions"][str(dcs)] += 1; out["decision_counts"][str(dcs)] += 1
            if dcs == "hold":
                pr["refused"][str(f.get("refused"))] += 1; out["refused_counts"][str(f.get("refused"))] += 1
            if dcs != "follow_to":
                continue
            t0 = c["t"]; rows = [i for i in range(len(t_arr)) if t0 <= t_arr[i] <= t0 + 7]
            apple = [1e3 * float(rep["apple_crate_gap"][i + 1]) for i in rows] if rep is not None else []
            hand = [1e3 * float(rep["hand_crate_gap"][i + 1]) for i in rows] if rep is not None else []
            e = {"run": f"{key[0]}:{key[1]}_s{key[2]}", "seq": c["seq"], "t": t0, "phase": phase_of(r, att, det, t0, plan, c["seq"]), "streak": f.get("streak"), "limit": f.get("limit"),
                 "n_rows": f.get("n_rows"), "min_esdf_minus_r_mm": 1e3 * f["min_esdf_minus_r_m"], "plan_min_clearance_mm": 1e3 * f["plan_min_clearance_m"],
                 "now_min_clearance_mm": 1e3 * f["now_min_clearance_m"], "binding": c["binding"], "worst_obstacle": c.get("wp_obstacle"),
                 "true_apple_crate_min_mm": min(apple) if apple else None, "true_hand_crate_min_mm_post": min(hand) if hand else None,
                 "true_hurdle_min_mm": hurdle_min_mm(r, t0),
                 "contact_substeps": ({cl: int(sum(rep["cnt_" + cl][i] for i in rows)) for cl in ("hand-crate", "apple-crate", "robot_other-crate", "apple-obstacle")} if rep is not None else None)}
            out["executed"].append(e); pr["executed_t"].append(t0)
        pr["decisions"] = dict(pr["decisions"]); pr["refused"] = dict(pr["refused"])
        out["per_run"][f"{key[0]}:{key[1]}_s{key[2]}"] = pr
    out["decision_counts"] = dict(out["decision_counts"]); out["refused_counts"] = dict(out["refused_counts"])
    out["n_executed"] = len(out["executed"]); out["n_runs_with_executed"] = len({e["run"] for e in out["executed"]})
    return out


def st(v):
    v = [float(x) for x in v if x is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": len(a), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max()), "min": float(a.min()), "mean": float(a.mean())}


def mi_hit(r):
    """count of server chunks (whole run, first included) with to.max_iterations_hit true, read from the chunk records (server_segment first + n)."""
    seg = r.get("server_segment")
    if isinstance(seg, str):
        import ast
        seg = ast.literal_eval(seg)
    first = pathlib.Path(seg["first"]); i0 = int(first.stem.split("_")[1]); n = 0; tot = 0
    for j in range(int(seg["n"])):
        with np.load(first.parent / f"chunk_{i0 + j:05d}.npz", allow_pickle=True) as z:
            to = json.loads(str(z["summary_json"])).get("to") or {}
        tot += 1; n += bool(to.get("max_iterations_hit"))
    return n, tot


def timing(arm, stages=("H", "B")):
    solve, trajopt, total, rt, hit, sqp, lin, qp, chk = [], [], [], [], 0, [], [], [], []
    n_chunks = 0; n_hit_runs = 0; per_run = {}; mih = 0; mih_tot = 0
    for key, (k, r) in sorted(runs_of(arm).items()):
        if key[0] not in stages:
            continue
        sc = (r.get("server_chunks") or [])[1:]; cc = r["chunks"][1:]
        s_ = [c["solve_ms"] for c in sc if c["solve_ms"] is not None]
        tr = [(c["server_ms"] or {}).get("trajopt") for c in cc]; tt = [(c["server_ms"] or {}).get("total") for c in cc]; rtr = [c["client_round_trip_ms"] for c in cc]
        h = sum(bool(c["time_budget_hit"]) for c in (r.get("server_chunks") or []))   # whole run, first chunk included (count only)
        a_, b_ = mi_hit(r); mih += a_; mih_tot += b_
        solve += s_; trajopt += tr; total += tt; rt += [x for x in rtr if x is not None]; hit += h; n_chunks += len(sc); n_hit_runs += h > 0
        sqp += [c["sqp_iterations"] for c in sc]
        lin += [(c["timing_ms"] or {}).get("linearize") for c in sc]; qp += [(c["timing_ms"] or {}).get("qp") for c in sc]; chk += [(c["timing_ms"] or {}).get("check") for c in sc]
        per_run[f"{key[0]}:{key[1]}_s{key[2]}"] = {"to_solve_median_ms": float(np.median(s_)) if s_ else None, "n_chunks": len(sc), "time_budget_hit": h,
                                                   "round_trip_median_ms": float(np.median([x for x in rtr if x is not None])) if rtr else None}
    rtv = [x for x in rt if x is not None]
    return {"n_chunks_excl_first": n_chunks, "to_solve_ms": st(solve), "verdict_trajopt_ms": st(trajopt), "verdict_total_ms": st(total), "round_trip_ms": st(rtv),
            "round_trip_gt_533_3_n": int(sum(x > CHUNK_PERIOD_MS for x in rtv)), "round_trip_n": len(rtv),
            "round_trip_gt_533_3_share": (sum(x > CHUNK_PERIOD_MS for x in rtv) / len(rtv)) if rtv else None,
            "time_budget_hit_count": hit, "runs_with_time_budget_hit": n_hit_runs, "max_iterations_hit_count": mih, "n_chunks_all_incl_first": mih_tot, "sqp_iterations": st(sqp),
            "linearize_ms": st(lin), "qp_ms": st(qp), "check_ms": st(chk), "per_run": per_run}


def finger_contacts(ct):
    out = {}
    for arm in ARMS:
        tab = {}
        for key, o in ct["runs"].items():
            if o["arm"] != arm:
                continue
            for c in o["contacts"]:
                if not c["body"].startswith("ee_finger"):
                    continue
                e = tab.setdefault((c["body"], c["phase"]), {"body": c["body"], "phase": c["phase"], "runs": [], "n_rows": 0, "substeps": 0, "max_pen_mm": 0.0, "max_force_N": 0.0, "margin_substeps": 0})
                e["runs"].append(f"{o['episode']}_s{o['seed']}"); e["n_rows"] += c["n_rows"]; e["substeps"] += c["substeps"]
                e["max_pen_mm"] = max(e["max_pen_mm"], c["max_pen_mm"]); e["max_force_N"] = max(e["max_force_N"], c["max_force_N"]); e["margin_substeps"] += c["margin_substeps"]
        rows = []
        for (b, ph), e in sorted(tab.items()):
            e["n_runs"] = len(e["runs"]); rows.append(e)
        byph = collections.OrderedDict()
        for ph in ("before_grasp", "carry", "after_release"):
            rs = {r for e in rows if e["phase"] == ph for r in e["runs"]}
            byph[ph] = {"runs": sorted(rs), "n_runs": len(rs), "n_rows": sum(e["n_rows"] for e in rows if e["phase"] == ph), "substeps": sum(e["substeps"] for e in rows if e["phase"] == ph),
                        "max_pen_mm": max([e["max_pen_mm"] for e in rows if e["phase"] == ph] or [0.0])}
        out[arm] = {"by_body_phase": rows, "by_phase": byph}
    return out


def hurdle_dist(arm):
    rows = {}
    for key, (k, r) in sorted(runs_of(arm).items()):
        h = r.get("hurdle")
        if not h:
            continue
        ph = h["phases"]
        rows[f"{key[0]}:{key[1]}_s{key[2]}"] = {p: {s: (None if (ph.get(p) or {}).get(s) is None else 1e3 * ph[p][s]["min_m"]) for s in ("Af", "Ae", "B", "C")} for p in ph}
    return rows


def sgp_old(arm):
    out = {}
    for stage in ("H", "B"):
        L = [r for kk, (k, r) in runs_of(arm).items() if kk[0] == stage]
        out[stage] = {"n": len(L), "S": sum(r["success"] for r in L), "G": sum(r["grasp"] for r in L), "P": sum(r["place"] for r in L)}
    return out


def pairs_old(stage):
    ka = {k: r for k, (kk, r) in runs_of("V9").items() if k[0] in stage}; kb = {k: r for k, (kk, r) in runs_of("V8").items() if k[0] in stage}
    ks = sorted(set(ka) & set(kb)); res = {"n_pairs": len(ks)}
    for f in ("success", "grasp", "place"):
        both = sum(ka[k][f] and kb[k][f] for k in ks); ao = sum(ka[k][f] and not kb[k][f] for k in ks); bo = sum(kb[k][f] and not ka[k][f] for k in ks)
        n = ao + bo; kk_ = min(ao, bo)
        p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(kk_ + 1)) / 2 ** n)
        res[f] = {"V9_total": both + ao, "V8_total": both + bo, "both": both, "V9_only": ao, "V8_only": bo, "neither": len(ks) - both - ao - bo, "mcnemar_exact_p": p,
                  "V9_only_keys": [f"{k[0]}:{k[1]}_s{k[2]}" for k in ks if ka[k][f] and not kb[k][f]], "V8_only_keys": [f"{k[0]}:{k[1]}_s{k[2]}" for k in ks if kb[k][f] and not ka[k][f]]}
    return res


def main():
    ct = json.load(open(V / "ct.json"))
    res = {"definitions": __doc__, "run_counts": {a: {"n_runs": len(runs_of(a)), "H": sum(1 for k in runs_of(a) if k[0] == "H"), "B": sum(1 for k in runs_of(a) if k[0] == "B")} for a in ARMS}}
    res["sgp_old_rule_by_stage"] = {a: sgp_old(a) for a in ARMS}
    res["pairs_old_rule"] = {"H": pairs_old(("H",)), "B": pairs_old(("B",)), "ALL": pairs_old(("H", "B"))}
    res["ct"] = {"arms": ct["arms"], "pairs": ct["pairs"], "crosscheck_counts": {k: (v if k == "n" else len(v)) for k, v in ct["crosscheck"].items()}}
    res["finger_contacts"] = finger_contacts(ct)
    res["obstacle_stop"] = {a: {"runs": ct["arms"][a]["obstacle_stop_runs"], "keys": sorted(o["key"] for o in ct["runs"].values() if o["arm"] == a and o["obstacle_stop"])} for a in ARMS}
    hb = {a: hold_breakdown(a) for a in ARMS}
    res["hold"] = hb; res["hold_pairs"] = pair_holds(hb)
    res["follow_to"] = {a: follow_to(a) for a in ARMS}
    res["timing"] = {a: {"ALL": timing(a), "H": timing(a, ("H",)), "B": timing(a, ("B",))} for a in ARMS}
    res["hurdle_min_mm"] = {a: hurdle_dist(a) for a in ARMS}
    res["trailing_hold_chunks"] = {a: hb[a]["trailing"] for a in ARMS}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
    for a in ARMS:
        h = hb[a]; t = res["timing"][a]["ALL"]
        print(a, "hold", h["total"], "trail", h["trailing"], "runs", h["runs_with_hold"], h["by_category"], h["by_target"], h["by_phase"])
        print(a, "TO solve", {k: round(t["to_solve_ms"][k], 1) for k in ("median", "p90", "max")}, "budget_hit", t["time_budget_hit_count"], "max_it_hit", t["max_iterations_hit_count"], "/", t["n_chunks_all_incl_first"], "rt", {k: round(t["round_trip_ms"][k], 1) for k in ("median", "p90")}, "gt533", t["round_trip_gt_533_3_share"])
        f = res["follow_to"][a]; print(a, "follow_to", f["decision_counts"], f["refused_counts"], "executed", f["n_executed"])
    print("new HOLD V9-only", res["hold_pairs"]["V9_only_n"], res["hold_pairs"]["V9_only_by_category"], res["hold_pairs"]["V9_only_by_target"], "V8-only", res["hold_pairs"]["V8_only_n"])


if __name__ == "__main__":
    main()
