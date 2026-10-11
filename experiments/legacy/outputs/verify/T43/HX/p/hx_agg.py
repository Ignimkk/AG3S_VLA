"""T43 HX aggregate — raw/hx_{V7g,V8,V5}.json -> raw/hx_summary.json (numbers only).

Moments: rows of the crossing window (hx_measure.py definition), per group (fingers, palm, apple[carry only]).
  below-margin moment: geometric M distance < 10 mm (the obstacle margin); contact moment: M < 0.
  HOLD rows (executed false) are counted apart (their command is q_hold, not the TO plan).
Cause of an executed-row moment (first rule that holds):
  (ii)  the TO row of that group at (chunk, k) has clearance < 0 (TO saw it; status / iterations / budget / verdict recorded)
  (i)   P (= C) geometric distance >= 10 mm  (plan geometrically clear; only M enters = tracking lag)
  (iv)  margin-0 row: held (apple) group with TO d - r < 10 mm, or robot group whose binding sphere has class != obstacle
        (margin 0 / phase margin) with d - r < 10 mm
  (iii) ESDF: with the exact bar distance of the same TO spheres (true_sphere - margin) the row would be < 0, the ESDF said >= 0
  (v)   other: TO spheres clear of the bar by >= margin but the collision mesh is closer (sphere / held-sphere coverage)
P moments: executed rows with P geometric < 10 mm, classified with (ii) / (iv) / (iii) / (v) (no (i)).
Servo: rows from grasp_t with a prediction; 'stop' = last row of a run with '[SAFETY] obstacle stop' (physics broke off early).
Contacts: T43TK numbers.contacts with kind hand (after_release) or apple (carry) = 22; before_grasp 2 and robot_other 2 listed apart.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, collections
import numpy as np

R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/HX/raw'))
TK = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43TK.verify.json')))["numbers"]["contacts"]
MARG = 10.0
MAX_STEP_RAD = float(np.radians(10.8))  # v_max*dt = 10.8 deg (work-t43v8 AG3S/trajopt/config.py:261)


def st(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "min": float(x.min()), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
            "max": float(x.max())}


def episodes(ts):
    ts = sorted(ts); n = 0; prev = None
    for t in ts:
        if prev is None or t != prev + 1:
            n += 1
        prev = t
    return n


def classify(row, g):
    to = (row.get("to") or {}).get(g)
    if to is None:
        return "v_no_to_row", None
    if to["to_clear_mm"] < 0:
        return "ii", to
    if row["geo_C"][g] >= MARG:
        return "i", to
    if g == "apple" and to["d_minus_r_mm"] < MARG:
        return "iv_held", to
    if g != "apple" and to["cls"] != "obstacle" and to["d_minus_r_mm"] < MARG:
        return "iv_robot_class_" + to["cls"], to
    if to["true_sphere_mm"] - to["true_nearest_margin_mm"] < 0:
        return "iii", to
    return "v_coverage", to


def classify_P(row, g):
    to = (row.get("to") or {}).get(g)
    if to is None:
        return "v_no_to_row"
    if to["to_clear_mm"] < 0:
        return "ii"
    if g == "apple" and to["d_minus_r_mm"] < MARG:
        return "iv_held"
    if g != "apple" and to["cls"] != "obstacle" and to["d_minus_r_mm"] < MARG:
        return "iv_robot_class_" + to["cls"]
    if to["true_sphere_mm"] - to["true_nearest_margin_mm"] < 0:
        return "iii"
    return "v_coverage"


out = {"definitions": __doc__, "arms": {}, "contacts22": [], "contacts_other": []}
for arm in ("V7g", "V8", "V5"):
    D = json.load(open(R / f"hx_{arm}.json")); runs = D["runs"]
    A = {"n_runs": len(runs), "n_runs_grasp": 0, "phase": {}, "servo": {}, "fidelity": {}, "per_run": {}, "ii_events": [], "to_neg_exec_rows": [], "apple_P_lt10_rows": []}
    servo_rows = {"all_exec": [], "carry_window_exec": [], "stop": [], "hold": []}
    for ph in ("carry", "return"):
        P = {"n_window_rows": 0, "n_window_rows_exec": 0, "n_window_rows_hold": 0, "groups": {}}
        for g in (("fingers", "palm", "apple") if ph == "carry" else ("fingers", "palm")):
            P["groups"][g] = {"min_P_geo_mm": [], "min_M_geo_mm": [], "min_TO_clear_mm": [], "min_TO_d_minus_r_mm": [],
                              "moments_exec": collections.Counter(), "moments_exec_contact": collections.Counter(),
                              "moments_hold": 0, "moments_hold_contact": 0, "episodes_exec": 0, "episodes_hold": 0,
                              "P_moments": collections.Counter(), "servo_flag_geo": 0, "servo_flag_esdf": 0, "servo_flag_any": 0,
                              "servo_flag_geo_contact": 0, "n_contact_with_S": 0, "n_with_S": 0,
                              "esdf_minus_true_mm": [], "rows_ts": []}
        A["phase"][ph] = P
    for key, run in runs.items():
        rows = run["rows"]
        if run.get("grasp_t") is None:
            continue
        A["n_runs_grasp"] += 1
        last_t = run["n_rows"] - 1
        pr = {"window_rows": {"carry": 0, "return": 0}, "min": {}}
        for row in rows:
            ex = row["executed"]; ph = row["phase"]
            if "servo_err_mm" in row:
                rec = (row["servo_err_mm"], row["cmd_err_mm"], row["servo_err_joint_deg_max"], row["cmd_err_joint_deg_max"])
                if row["t"] == last_t and row["M_src"] != "row t+1 qpos":
                    servo_rows["stop"].append(rec)
                elif not ex:
                    servo_rows["hold"].append(rec)
                else:
                    servo_rows["all_exec"].append(rec)
                    if row["in_window"] and ph == "carry":
                        servo_rows["carry_window_exec"].append(rec)
            if not row["in_window"]:
                continue
            P = A["phase"][ph]; P["n_window_rows"] += 1; P["n_window_rows_exec" if ex else "n_window_rows_hold"] += 1
            pr["window_rows"][ph] += 1
            for g, G in P["groups"].items():
                if g not in row["geo_M"]:
                    continue
                to = (row.get("to") or {}).get(g)
                if ex:
                    G["min_P_geo_mm"].append(row["geo_C"][g]); G["min_M_geo_mm"].append(row["geo_M"][g])
                    if to:
                        G["min_TO_clear_mm"].append(to["to_clear_mm"]); G["min_TO_d_minus_r_mm"].append(to["d_minus_r_mm"])
                        G["esdf_minus_true_mm"].append(to["esdf_at_true_nearest_mm"] - to["true_sphere_mm"])
                    if row["geo_C"][g] < MARG:
                        G["P_moments"][classify_P(row, g)] += 1
                    if to and to["to_clear_mm"] < 0:
                        ch = run["chunks"].get(str(row["seq"])) or {}
                        A["to_neg_exec_rows"].append({"key": key, "phase": ph, "t": row["t"], "k": row["k"], "group": g, "to_clear_mm": to["to_clear_mm"],
                                                      "cls": to["cls"], "margin_mm": to["margin_mm"], "true_sphere_mm": to["true_sphere_mm"],
                                                      "P_mm": row["geo_C"][g], "M_mm": row["geo_M"][g], "status": ch.get("status"),
                                                      "chunk_clearance_mm": ch.get("clearance_mm"), "initial_violation_mm": ch.get("initial_violation_mm"),
                                                      "sqp_iterations": ch.get("sqp_iterations"), "max_iterations_hit": ch.get("max_iterations_hit"),
                                                      "time_budget_hit": ch.get("time_budget_hit"), "verdict": ch.get("verdict"), "worst": ch.get("worst"),
                                                      "max_abs_step_rad_row_k": (ch.get("max_abs_step_rad_by_row") or [None] * 8)[row["k"]] if row["k"] < 8 else None,
                                                      "to_reference_clear_mm": ((row.get("to_reference") or {}).get(g) or {}).get("to_clear_mm")})
                    if g == "apple" and row["geo_C"][g] < MARG:
                        A["apple_P_lt10_rows"].append({"key": key, "t": row["t"], "k": row["k"], "P_mm": row["geo_C"][g], "M_mm": row["geo_M"][g],
                                                       "to_clear_mm": (to or {}).get("to_clear_mm"), "d_minus_r_mm": (to or {}).get("d_minus_r_mm"),
                                                       "true_sphere_mm": (to or {}).get("true_sphere_mm")})
                m = row["geo_M"][g]
                if m < MARG:
                    if not ex:
                        G["moments_hold"] += 1; G["moments_hold_contact"] += int(m < 0)
                        continue
                    c, tov = classify(row, g)
                    G["moments_exec"][c] += 1
                    if m < 0:
                        G["moments_exec_contact"][c] += 1
                    G["rows_ts"].append((key, row["t"]))
                    if "geo_S" in row and g in row["geo_S"]:
                        G["n_with_S"] += 1
                        fg = row["geo_S"][g] < MARG
                        fe = bool((row.get("to_at_S") or {}).get(g)) and row["to_at_S"][g]["to_clear_mm"] < 0
                        G["servo_flag_geo"] += int(fg); G["servo_flag_esdf"] += int(fe); G["servo_flag_any"] += int(fg or fe)
                        if m < 0:
                            G["n_contact_with_S"] += 1; G["servo_flag_geo_contact"] += int(row["geo_S"][g] < 0)
                    if c == "ii":
                        ch = run["chunks"].get(str(row["seq"])) or {}
                        A["ii_events"].append({"key": key, "t": row["t"], "k": row["k"], "group": g, "M_mm": m, "P_mm": row["geo_C"][g],
                                               "to_clear_mm": tov["to_clear_mm"], "cls": tov["cls"], "status": ch.get("status"),
                                               "sqp_iterations": ch.get("sqp_iterations"), "max_iterations_hit": ch.get("max_iterations_hit"),
                                               "time_budget_hit": ch.get("time_budget_hit"), "solve_ms": ch.get("solve_ms"),
                                               "initial_violation_mm": ch.get("initial_violation_mm"), "chunk_clearance_mm": ch.get("clearance_mm"),
                                               "verdict": ch.get("verdict"), "notes": ch.get("notes"), "worst": ch.get("worst"),
                                               "max_abs_step_rad_row_k": (ch.get("max_abs_step_rad_by_row") or [None] * 8)[row["k"]] if row["k"] < 8 else None,
                                               "max_abs_step_rad_rows": ch.get("max_abs_step_rad_by_row"),
                                               "to_reference_clear_mm": ((row.get("to_reference") or {}).get(g) or {}).get("to_clear_mm")})
        # episodes per run/group
        A["per_run"][key] = pr
    # finalize
    for ph, P in A["phase"].items():
        for g, G in P["groups"].items():
            ts = collections.defaultdict(list)
            for k_, t_ in G.pop("rows_ts"):
                ts[k_].append(t_)
            G["episodes_exec"] = sum(episodes(v) for v in ts.values())
            for f in ("min_P_geo_mm", "min_M_geo_mm", "min_TO_clear_mm", "min_TO_d_minus_r_mm", "esdf_minus_true_mm"):
                G[f] = st(G[f])
            G["moments_exec"] = dict(G["moments_exec"]); G["moments_exec_contact"] = dict(G["moments_exec_contact"]); G["P_moments"] = dict(G["P_moments"])
            G["n_moments_exec"] = int(sum(G["moments_exec"].values())); G["n_moments_exec_contact"] = int(sum(G["moments_exec_contact"].values()))
    for nm, v in servo_rows.items():
        v = np.asarray(v, float).reshape(-1, 4)
        A["servo"][nm] = {"servo_err_mm": st(v[:, 0]), "cmd_err_mm": st(v[:, 1]), "servo_err_joint_deg_max": st(v[:, 2]), "cmd_err_joint_deg_max": st(v[:, 3])}
    # fidelity
    fa, near, far, held_far = [], [], [], []
    for key, run in runs.items():
        for sq, c in run["chunks"].items():
            if c.get("fid_record_clearance_max_abs_mm") is not None:
                fa.append(c["fid_record_clearance_max_abs_mm"])
            det = c.get("mine_at_rec_worst_detail"); w = c["worst"]
            if det and w.get("clearance_mm") is not None:
                diff = abs(c["mine_at_rec_worst_mm"] - w["clearance_mm"])
                (near if det["true_bar_mm"] < 50 else far).append(diff)
    fa, near, far = (np.asarray(x, float) for x in (fa, near, far))
    A["fidelity"] = {"n_chunks": int(len(fa)), "record_clearance_max_abs_mm": st(fa), "record_clearance_n_gt_1mm": int((fa > 1).sum()),
                     "worst_row_true_bar_lt50mm": {"n": int(len(near)), "n_within_1mm": int((near < 1).sum()), "max_abs_mm": float(near.max()) if len(near) else None},
                     "worst_row_true_bar_ge50mm": {"n": int(len(far)), "n_within_1mm": int((far < 1).sum())}}
    # chunk-level counters over window chunks
    cs = collections.Counter(); mih = 0; tbh = 0; nch = 0
    for key, run in runs.items():
        for sq, c in run["chunks"].items():
            nch += 1; cs[c["status"]] += 1; mih += int(bool(c.get("max_iterations_hit"))); tbh += int(bool(c.get("time_budget_hit")))
    steps = [v for run in runs.values() for c in run["chunks"].values() for v in (c.get("max_abs_step_rad_by_row") or [])]
    steps = np.asarray(steps, float)
    ps = collections.Counter(c.get("planned_steps") for run in runs.values() for c in run["chunks"].values())
    sqpit = collections.Counter(c.get("sqp_iterations") for run in runs.values() for c in run["chunks"].values())
    verd = collections.Counter(tuple(sorted({a for _, a in (c.get("verdict") or [])})) for run in runs.values() for c in run["chunks"].values())
    A["window_chunks"] = {"n": nch, "status": dict(cs), "max_iterations_hit": mih, "time_budget_hit": tbh,
                          "planned_steps": {str(k): v for k, v in ps.items()}, "sqp_iterations": {str(k): v for k, v in sqpit.items()},
                          "verdict_actions": {"|".join(k): v for k, v in verd.items()},
                          "max_step_rad": MAX_STEP_RAD, "n_rows": int(len(steps)), "rows_step_ge_98pct_max_step": int((steps >= 0.98 * MAX_STEP_RAD).sum()),
                          "max_abs_step_rad": st(steps)}
    out["arms"][arm] = A

# 22 contacts
H = {arm: json.load(open(R / f"hx_{arm}.json"))["runs"] for arm in ("V7g", "V8")}
for c in TK:
    grp = "apple" if c["kind"] == "apple" else ("fingers" if "finger" in c["body"] else ("palm" if "EE_BODY" in c["body"] else "other"))
    rec = {"arm": c["arm"], "key": c["key"], "kind": c["kind"], "phase_ct": c["phase_ct"], "body": c["body"], "t": c["t"], "group": grp,
           "tk_bar_M_mm": c.get("bar_M_mm"), "tk_bar_P_mm": c.get("bar_P_mm"), "tk_to_clearance_mm": c.get("to_clearance_mm")}
    run = H[c["arm"]].get(c["key"]); row = None if run is None else next((r for r in run["rows"] if r["t"] == c["t"]), None)
    if row is not None and grp in row["geo_M"]:
        rec.update({"in_window": row["in_window"], "executed": row["executed"], "k": row["k"], "M_src": row["M_src"],
                    "geo_C_mm": row["geo_C"][grp], "geo_M_mm": row["geo_M"][grp], "geo_S_mm": (row.get("geo_S") or {}).get(grp),
                    "servo_err_mm": row.get("servo_err_mm"), "cmd_err_mm": row.get("cmd_err_mm"),
                    "to_clear_mm": ((row.get("to") or {}).get(grp) or {}).get("to_clear_mm"),
                    "to_true_sphere_mm": ((row.get("to") or {}).get(grp) or {}).get("true_sphere_mm"),
                    "to_d_minus_r_mm": ((row.get("to") or {}).get(grp) or {}).get("d_minus_r_mm"),
                    "to_at_S_clear_mm": ((row.get("to_at_S") or {}).get(grp) or {}).get("to_clear_mm"),
                    "to_at_S_true_sphere_mm": ((row.get("to_at_S") or {}).get(grp) or {}).get("true_sphere_mm"),
                    "to_at_M_clear_mm": ((row.get("to_at_M") or {}).get(grp) or {}).get("to_clear_mm"),
                    "binding": row.get("binding")})
        gs = rec.get("geo_S_mm"); ts_ = rec.get("to_at_S_clear_mm")
        rec["flag_geo_contact"] = None if gs is None else bool(gs < 0)
        rec["flag_geo_margin"] = None if gs is None else bool(gs < MARG)
        rec["flag_esdf"] = None if ts_ is None else bool(ts_ < 0)
        rec["flag_esdf_true_sphere"] = None if rec.get("to_at_S_true_sphere_mm") is None else bool(rec["to_at_S_true_sphere_mm"] - MARG < 0)
    sel = (c["kind"] == "hand" and c["phase_ct"] == "after_release") or (c["kind"] == "apple" and c["phase_ct"] == "carry")
    (out["contacts22"] if sel else out["contacts_other"]).append(rec)
c22 = out["contacts22"]
out["contacts22_summary"] = {
    "n": len(c22),
    "by_group_phase": dict(collections.Counter(f"{r['group']}|{r['phase_ct']}" for r in c22)),
    "flag_geo_contact": sum(1 for r in c22 if r.get("flag_geo_contact")),
    "flag_geo_margin": sum(1 for r in c22 if r.get("flag_geo_margin")),
    "flag_esdf": sum(1 for r in c22 if r.get("flag_esdf")),
    "flag_esdf_true_sphere": sum(1 for r in c22 if r.get("flag_esdf_true_sphere")),
    "n_with_prediction": sum(1 for r in c22 if r.get("geo_S_mm") is not None),
    "geo_S_mm": st([r.get("geo_S_mm") for r in c22]), "geo_M_mm": st([r.get("geo_M_mm") for r in c22]),
    "geo_C_mm": st([r.get("geo_C_mm") for r in c22]),
    "S_minus_M_mm": st([r["geo_S_mm"] - r["geo_M_mm"] for r in c22 if r.get("geo_S_mm") is not None]),
    "M_vs_TK_abs_mm": st([abs(r["geo_M_mm"] - r["tk_bar_M_mm"]) for r in c22 if r.get("geo_M_mm") is not None and r.get("tk_bar_M_mm") is not None]),
    "to_clear_lt0": sum(1 for r in c22 if r.get("to_clear_mm") is not None and r["to_clear_mm"] < 0),
}
for sub in ("apple|carry", "fingers|after_release"):
    rr = [r for r in c22 if f"{r['group']}|{r['phase_ct']}" == sub]
    out["contacts22_summary"][sub] = {"n": len(rr), "flag_geo_contact": sum(1 for r in rr if r.get("flag_geo_contact")),
                                      "flag_geo_margin": sum(1 for r in rr if r.get("flag_geo_margin")),
                                      "flag_esdf": sum(1 for r in rr if r.get("flag_esdf")),
                                      "flag_esdf_true_sphere": sum(1 for r in rr if r.get("flag_esdf_true_sphere"))}
(R / "hx_summary.json").write_text(json.dumps(out, indent=1, default=float))
for arm, A in out["arms"].items():
    print(arm, "runs", A["n_runs"], "grasp", A["n_runs_grasp"], "window chunks", A["window_chunks"])
    for ph, P in A["phase"].items():
        print(" ", ph, "rows", P["n_window_rows"], "exec", P["n_window_rows_exec"], "hold", P["n_window_rows_hold"])
        for g, G in P["groups"].items():
            print("   ", g, "minP", round(G["min_P_geo_mm"].get("min", np.nan), 1), "minM", round(G["min_M_geo_mm"].get("min", np.nan), 1),
                  "minTO", round(G["min_TO_clear_mm"].get("min", np.nan), 1), "min d-r", round(G["min_TO_d_minus_r_mm"].get("min", np.nan), 1),
                  "| M<10 exec", G["n_moments_exec"], G["moments_exec"], "contact", G["n_moments_exec_contact"], G["moments_exec_contact"],
                  "ep", G["episodes_exec"], "hold", G["moments_hold"], G["moments_hold_contact"], "| P<10", G["P_moments"],
                  "| servo flag geo/esdf/any", G["servo_flag_geo"], G["servo_flag_esdf"], G["servo_flag_any"], "/", G["n_with_S"],
                  "contact geo", G["servo_flag_geo_contact"], "/", G["n_contact_with_S"])
    print("  servo", {k: (v["servo_err_mm"].get("median"), v["servo_err_mm"].get("p90"), v["servo_err_mm"].get("max"), v["cmd_err_mm"].get("median"), v["servo_err_mm"].get("n")) for k, v in A["servo"].items()})
    print("  fidelity", A["fidelity"])
    print("  ii", len(A["ii_events"]), "TO<0 exec window rows", len(A["to_neg_exec_rows"]))
    for e in A["to_neg_exec_rows"]:
        print("     ", e["key"].split("/")[-1], e["phase"], e["t"], e["k"], e["group"], round(e["to_clear_mm"], 2), e["cls"], "P", round(e["P_mm"], 1), "M", round(e["M_mm"], 1), e["status"], e["verdict"], round(e["chunk_clearance_mm"], 2), "ref", None if e["to_reference_clear_mm"] is None else round(e["to_reference_clear_mm"], 1), (e["worst"] or {}).get("link"), (e["worst"] or {}).get("class"))
    for ph, P in A["phase"].items():
        for g, G in P["groups"].items():
            print("   esdf-true", ph, g, {k: round(v, 1) for k, v in G["esdf_minus_true_mm"].items() if k != "n"}, G["esdf_minus_true_mm"].get("n"))
print("contacts22", json.dumps(out["contacts22_summary"], default=float))
