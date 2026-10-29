"""T43 V7g aggregation (verifier, CPU only, numbers only). usage: agg_v7g.py OUT_JSON
Inputs: W/analysis (V5), Z3V/analysis (V5D), V7g/analysis (V7g), V7g/z4.json (z4_v7g.py), V7g/y2_<ARM>.json (y2_v7g.py), replays (z4_v7g.REPLAY).
 HOLD chunk       : client planning row with hold flag (analysis chunks[].hold); trailing = consecutive HOLD chunks ending at the run's last planning row.
 binding category : analysis server_chunks[].binding.category (waypoint vs path rows, the smaller clearance) = base (q < 122) | gripper_cover | held; robot = base + gripper_cover.
 target           : chunks[].worst.obstacle of the client planning row (the hold reason's worst row label): destination | hurdle-named | None. n_none_hurdle_le_50mm = chunks with label None whose
                    hurdle distance (analysis hurdle series, min of Af / Ae / B / C over the chunk's control rows t .. t+7, capped 0.30 m) is <= 50 mm.
 phase            : pre_pick  t < attach_t (attach_t = first server chunk with attached true; no attach -> every chunk);
                    release_after  t >= detach_t (first server chunk with grasp_state placed after an attached chunk);
                    crate_above    attached chunk (before detach) whose planning-row apple centre has crate-local |x| < 0.077 and |y| < 0.137 (any z; the Y over_crate rule);
                    carry          attached chunk (before detach) not over the crate.
 follow_to        : server verdict.follow_to per chunk: decision follow_to (executed) | hold with refused reason | not_needed.
                    True distances of an executed chunk (replay of the run, post-state of control rows t .. t+7 of the chunk; pre-state = state before row t):
                    apple_crate_mm = y2 series apple_crate_gap_mm (apple collision-mesh vertices vs the 13 crate boxes, Y method), hand_crate_mm = replay hand_crate_gap (m, capped 0.10),
                    hurdle_mm = analysis hurdle series min(Af, Ae, B, C) (capped 300), contact sub-steps hand-crate / apple-crate / robot_other-crate / apple-obstacle over those rows.
 attach radii     : analysis attach_held per run (first attach): mode, max_radius_mm (held mode), body_cover_radii_mm, body_cover_mode, cap record (grasp.attached.held.cap / body_cover.cap).
 slip detach      : detach_t and the held_slip flag (server chunk record) vs release_t measured (z4) and its mode.
 subtask          : server chunk subtask label / gate / blocked / would_block counts (V7g only; V5 and V5D ran with the gate off and the older analysis does not carry the fields)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, math, collections
import numpy as np
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
V = T / "V7g"
AN = {"V5": json.load(open(T / "W/analysis/analysis.json"))["runs"], "V5D": json.load(open(T / "Z3V/analysis/analysis.json"))["runs"], "V7g": json.load(open(V / "analysis/analysis.json"))["runs"]}
Z4 = json.load(open(V / "z4.json"))
Y2 = {a: json.load(open(V / f"y2_{a}.json"))["runs"] for a in ("V5", "V5D", "V7g")}
REPLAY = {a: pathlib.Path(p) for a, p in Z4["replay_dirs"].items()}
ARMS = ("V5", "V5D", "V7g")


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


def hold_breakdown(arm):
    out = {"per_run": {}, "cells": collections.Counter(), "by_category": collections.Counter(), "by_target": collections.Counter(), "by_phase": collections.Counter(),
           "total": 0, "trailing": 0, "runs_with_hold": 0, "runs_with_trailing": 0, "n_none_hurdle_le_50mm": 0, "by_stage_total": collections.Counter(),
           "hold_kinds": collections.Counter()}
    for key, (k, r) in sorted(runs_of(arm).items()):
        att, det = attach_detach(r); plan = plan_rows(r); sc = {c["seq"]: c for c in r["server_chunks"]}
        holds = [c for c in r["chunks"] if c["hold"]]
        rr = {"n_hold": len(holds), "trailing": r["trailing_hold_chunks"], "longest_run": r["longest_hold_run_chunks"], "attach_t": att, "detach_t": det, "chunks": []}
        for c in holds:
            s = sc.get(c["seq"]); t = c["t"]
            cat = (s["binding"]["category"] if s else None)
            tgt = (c.get("worst") or {}).get("obstacle")
            if att is None or t < att:
                ph = "pre_pick"
            elif det is not None and t >= det:
                ph = "release_after"
            else:
                ph = "crate_above" if over_crate(plan[c["seq"]]) else "carry"
            hm = hurdle_min_mm(r, t)
            near = bool(tgt is None and hm is not None and hm <= 50.0)
            out["cells"][f"{cat}|{tgt}|{ph}"] += 1; out["by_category"][str(cat)] += 1; out["by_target"][str(tgt)] += 1; out["by_phase"][ph] += 1
            out["n_none_hurdle_le_50mm"] += near
            for kd in c.get("hold_kinds") or []:
                out["hold_kinds"][kd] += 1
            rr["chunks"].append({"seq": c["seq"], "t": t, "category": cat, "target": tgt, "phase": ph, "worst_link": (c.get("worst") or {}).get("link"), "hurdle_min_mm": hm,
                                 "clearance_mm": None if (c.get("worst") or {}).get("clearance_m") is None else 1e3 * c["worst"]["clearance_m"], "hold_kinds": c.get("hold_kinds")})
        out["per_run"][f"{key[0]}:{key[1]}_s{key[2]}"] = rr
        out["total"] += len(holds); out["trailing"] += r["trailing_hold_chunks"]; out["runs_with_hold"] += bool(holds); out["runs_with_trailing"] += r["trailing_hold_chunks"] > 0
        out["by_stage_total"][key[0]] += len(holds)
    for f in ("cells", "by_category", "by_target", "by_phase", "by_stage_total", "hold_kinds"):
        out[f] = dict(sorted(out[f].items()))
    out["robot_total"] = out["by_category"].get("base", 0) + out["by_category"].get("gripper_cover", 0)
    return out


def follow_to():
    out = {"per_run": {}, "decision_counts": collections.Counter(), "refused_counts": collections.Counter(), "executed": []}
    for key, (k, r) in sorted(runs_of("V7g").items()):
        sc = r.get("server_chunks") or []
        rep = np.load(REPLAY["V7g"] / f"{k.split('/')[0]}_{key[0]}_{k.split('/')[2]}.npz")
        y2 = Y2["V7g"].get(k) or {}; ser = y2.get("series") or {}
        t_arr = rep["t"]; plan = plan_rows(r); att, det = attach_detach(r)
        pr = {"decisions": collections.Counter(), "refused": collections.Counter(), "executed_t": []}
        for c in sc:
            f = c.get("follow_to") or {}
            dcs = f.get("decision"); pr["decisions"][str(dcs)] += 1; out["decision_counts"][str(dcs)] += 1
            if dcs == "hold":
                pr["refused"][str(f.get("refused"))] += 1; out["refused_counts"][str(f.get("refused"))] += 1
            if dcs != "follow_to":
                continue
            t0 = c["t"]; rows = [i for i in range(len(t_arr)) if t0 <= t_arr[i] <= t0 + 7]
            apple = [ser["apple_crate_gap_mm"][i] for i in rows] if ser else []
            hand = [1e3 * float(rep["hand_crate_gap"][i + 1]) for i in rows]
            pre_hand = 1e3 * float(rep["hand_crate_gap"][rows[0]]) if rows else None
            ph = "pre_pick" if (att is None or t0 < att) else ("release_after" if (det is not None and t0 >= det) else ("crate_above" if over_crate(plan[c["seq"]]) else "carry"))
            ch = next((x for x in r["chunks"] if x["seq"] == c["seq"]), {})
            e = {"run": f"{key[0]}:{key[1]}_s{key[2]}", "seq": c["seq"], "t": t0, "phase": ph, "streak": f.get("streak"), "limit": f.get("limit"), "n_rows": f.get("n_rows"),
                 "min_esdf_minus_r_mm": 1e3 * f["min_esdf_minus_r_m"], "plan_min_clearance_mm": 1e3 * f["plan_min_clearance_m"], "now_min_clearance_mm": 1e3 * f["now_min_clearance_m"],
                 "refined_minus_reference_max_rad": f.get("refined_minus_reference_max_rad"), "binding": c["binding"], "worst_obstacle": c.get("wp_obstacle"),
                 "n_executed_control_rows": ch.get("n_executed"), "true_apple_crate_min_mm": min(apple) if apple else None,
                 "true_hand_crate_min_mm_post": min(hand) if hand else None, "true_hand_crate_pre_mm": pre_hand,
                 "true_hurdle_min_mm": hurdle_min_mm(r, t0), "contact_substeps": {cl: int(sum(rep["cnt_" + cl][i] for i in rows)) for cl in ("hand-crate", "apple-crate", "robot_other-crate", "apple-obstacle")},
                 "replaced_kinds": [x.get("kind") for x in (f.get("replaced") or [])] if isinstance(f.get("replaced"), list) else None}
            out["executed"].append(e); pr["executed_t"].append(t0)
        pr["decisions"] = dict(pr["decisions"]); pr["refused"] = dict(pr["refused"])
        out["per_run"][f"{key[0]}:{key[1]}_s{key[2]}"] = pr
    out["decision_counts"] = dict(out["decision_counts"]); out["refused_counts"] = dict(out["refused_counts"])
    ex = out["executed"]
    out["n_executed"] = len(ex); out["n_runs_with_executed"] = len({e["run"] for e in ex})
    for fld in ("true_apple_crate_min_mm", "true_hand_crate_min_mm_post", "true_hurdle_min_mm"):
        v = [e[fld] for e in ex if e[fld] is not None]
        out[fld + "_min"] = min(v) if v else None; out[fld + "_median"] = float(np.median(v)) if v else None
    out["n_executed_with_contact_substeps"] = sum(1 for e in ex if any(e["contact_substeps"][c] for c in ("hand-crate", "apple-crate", "robot_other-crate")))
    return out


def attach_radii(arm):
    """effective held query spheres at attach, read from the server chunk record of the attach chunk: body_cover.body_parent_frame / lifted_parent_frame radii
    (what the TO queries; body_cover.spheres / held.spheres are the world-frame record BEFORE the cap), held.spheres_parent_frame radii, cap record (V7g)."""
    import ast
    lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    rows = []
    for key, (k, r) in sorted(runs_of(arm).items()):
        nm = f"{key[0]}:{key[1]}_s{key[2]}"; ah = (r.get("attach_held") or [])
        if not ah:
            rows.append({"run": nm, "t": None}); continue
        a = ah[0]; ch = next(x for x in r["chunks"] if x["seq"] == a["seq"])
        z = np.load(ch["server_chunk"], allow_pickle=True); sm = json.loads(str(z["summary_json"]))
        held = lit(lit((sm.get("grasp") or {})["attached"])["held"]); bc = lit(held.get("body_cover")) or {}
        rad = lambda L: [x["radius_mm"] for x in (L or [])]
        rows.append({"run": nm, "t": a["t"], "mode": a.get("mode"), "body_cover_mode": bc.get("mode"), "body_parent_radii_mm": rad(bc.get("body_parent_frame")),
                     "lifted_parent_radii_mm": rad(bc.get("lifted_parent_frame")), "held_spheres_parent_radii_mm": rad(held.get("spheres_parent_frame")),
                     "record_world_radii_mm_pre_cap": rad(bc.get("spheres")), "fit_radius_mm": a.get("fit_radius_mm"), "body_radius_mm": bc.get("body_radius_mm"),
                     "n_query": len(rad(bc.get("body_parent_frame"))), "cap": a.get("cap")})
    mx = [max(x["body_parent_radii_mm"]) for x in rows if x.get("body_parent_radii_mm")]
    allr = [q for x in rows for q in (x.get("body_parent_radii_mm") or [])]
    return {"per_run": rows, "n_attach_runs": len(mx), "body_parent_max_radius_mm": {"min": min(mx) if mx else None, "median": float(np.median(mx)) if mx else None, "max": max(mx) if mx else None},
            "body_parent_all_radii_mm": {"n": len(allr), "min": min(allr) if allr else None, "median": float(np.median(allr)) if allr else None, "max": max(allr) if allr else None},
            "body_cover_modes": dict(collections.Counter(x.get("body_cover_mode") for x in rows if x.get("body_cover_mode")))}


def slip(arm):
    z = Z4["runs"]; rows = []
    for key, (k, r) in sorted(runs_of(arm).items()):
        att, det = attach_detach(r)
        dc = next((c for c in (r.get("server_chunks") or []) if att is not None and c["t"] >= att and c["grasp_state"] == "placed" and not c["attached"]), None)
        zz = z.get(k) or {}
        rows.append({"run": f"{key[0]}:{key[1]}_s{key[2]}", "attach_t": att, "detach_t": det, "slip_flag": (dc or {}).get("held_slip"), "release_t_measured": zz.get("release_t"),
                     "release_mode": zz.get("release_mode"), "applied_open_t": zz.get("applied_open_t"),
                     "detach_minus_release": None if (det is None or zz.get("release_t") is None) else det - zz["release_t"]})
    return {"per_run": rows, "n_detach": sum(1 for x in rows if x["detach_t"] is not None), "n_slip_flag_true": sum(1 for x in rows if x["slip_flag"])}


def subtask():
    out = {"per_run": {}, "label": collections.Counter(), "gate": collections.Counter(), "blocked": collections.Counter(), "would_block": collections.Counter(), "decision_reason": collections.Counter(),
           "released_true_chunks": 0, "n_chunks": 0}
    for key, (k, r) in sorted(runs_of("V7g").items()):
        sc = r.get("server_chunks") or []
        out["per_run"][f"{key[0]}:{key[1]}_s{key[2]}"] = {"label": r.get("subtask_label_counts"), "blocked": r.get("subtask_blocked_counts"), "would_block": r.get("subtask_would_block_counts"),
                                                          "decision_reason": r.get("decision_reason_counts")}
        for c in sc:
            s = c["subtask"]; out["n_chunks"] += 1
            out["label"][str(s["label"])] += 1; out["gate"][str(s["gate"])] += 1; out["blocked"][str(s["blocked"])] += 1; out["would_block"][str(s["would_block"])] += 1
            out["released_true_chunks"] += bool(s.get("released")); out["decision_reason"][str(c["decision_reason"])] += 1
    for f in ("label", "gate", "blocked", "would_block", "decision_reason"):
        out[f] = dict(sorted(out[f].items()))
    out["decision_reason_V5"] = {}; out["decision_reason_V5D"] = {}
    for arm in ("V5", "V5D"):
        cc = collections.Counter()
        for k, r in AN[arm].items():
            if k.startswith(arm + "/"):
                for x, n in (r.get("decision_reason_counts") or {}).items():
                    cc[x] += n
        out[f"decision_reason_{arm}"] = dict(sorted(cc.items()))
    return out


def y2_table():
    out = {}
    for arm in ARMS:
        rows = []
        for k, o in sorted(Y2[arm].items()):
            rows.append({"run": "%s:%d_s%d" % keyof(k), "release_t": o["release_t"], "release_mode": o.get("release_mode"), "release_with_cmd": o.get("release_with_cmd"),
                         "release_apple_floor_gap_mm": o.get("release_apple_floor_gap_mm"), "release_apple_centre_above_floor_mm": o.get("release_apple_centre_above_floor_mm"),
                         "release_v_horiz_mm_s": o.get("release_v_horiz_mm_s"), "release_v_vert_mm_s": o.get("release_v_vert_mm_s"), "post_release_drop_mm": o.get("post_release_drop_mm"),
                         "in_hand_slip_max_mm": o.get("in_hand_slip_max_mm"), "applied_open_t": o["applied_open_t"], "open_minus_release_rows": o.get("open_minus_release_rows"),
                         "in_crate_t": o["in_crate_t"], "over_crate_t": o["over_crate_t"], "place_t": o["place_t"], "success": o["success"], "grasp": o["grasp"], "place": o["place"],
                         "first_apple_crate_contact_t": o.get("first_apple-crate_contact_t"), "first_hand_crate_contact_t": o.get("first_hand-crate_contact_t"),
                         "n_hand_crate_contact_rows": o.get("n_hand-crate_contact_rows"), "n_apple_crate_contact_rows": o.get("n_apple-crate_contact_rows")})
        def st(f, sel=lambda x: True):
            v = [x[f] for x in rows if x[f] is not None and sel(x)]
            return {"n": len(v), "median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "mean": float(np.mean(v))} if v else {"n": 0}
        out[arm] = {"per_run": rows, "n_release": sum(1 for x in rows if x["release_t"] is not None),
                    "release_mode": dict(collections.Counter(x["release_mode"] for x in rows if x["release_t"] is not None)),
                    "release_mode_in_success": dict(collections.Counter(x["release_mode"] for x in rows if x["release_t"] is not None and x["success"])),
                    "release_apple_floor_gap_mm": st("release_apple_floor_gap_mm"), "release_apple_centre_above_floor_mm": st("release_apple_centre_above_floor_mm"),
                    "release_apple_floor_gap_mm_open_cmd": st("release_apple_floor_gap_mm", lambda x: x["release_mode"] == "open_cmd"),
                    "release_apple_floor_gap_mm_closed_cmd": st("release_apple_floor_gap_mm", lambda x: x["release_mode"] == "closed_cmd"),
                    "release_v_vert_mm_s": st("release_v_vert_mm_s"), "n_runs_apple_crate_contact": sum(1 for x in rows if x["first_apple_crate_contact_t"] is not None),
                    "n_runs_hand_crate_contact": sum(1 for x in rows if x["first_hand_crate_contact_t"] is not None)}
    return out


def place_t():
    out = {}
    for arm in ARMS:
        out[arm] = {"%s:%d_s%d" % key: r["place_t"] for key, (k, r) in sorted(runs_of(arm).items())}
    return out


def main():
    res = {"definitions": __doc__, "sgp": {"arms": Z4["arms"], "pairs": Z4["pairs"], "flips": Z4["flips"]}}
    res["hold"] = {a: hold_breakdown(a) for a in ARMS}
    res["follow_to"] = follow_to()
    res["attach_radii"] = {a: attach_radii(a) for a in ARMS}
    res["slip"] = {a: slip(a) for a in ("V5", "V5D", "V7g")}
    res["subtask"] = subtask()
    res["place_closed"] = y2_table()
    res["place_t"] = place_t()
    res["run_counts"] = {a: {"n_runs": len(runs_of(a)), "wall_s_median": float(np.median([r["wall_s"] for _, (k, r) in runs_of(a).items() if r.get("wall_s")] or [0]))} for a in ARMS}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
    for a in ARMS:
        h = res["hold"][a]; print(a, "hold", h["total"], "trailing", h["trailing"], "runs_hold", h["runs_with_hold"], h["by_category"], h["by_target"], h["by_phase"])
    f = res["follow_to"]; print("follow_to", f["decision_counts"], f["refused_counts"], "executed", f["n_executed"])


if __name__ == "__main__":
    main()
