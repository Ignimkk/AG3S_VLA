"""T43 V11ABC four-way comparison (copy of AB/compare_ab.py extended to V8 / V11A / V11AB / V11ABC; fail-stage rows now = runs with S_old true in some arm and false in some V11 arm) -- previously: T43 V11AB three-way comparison (verifier, CPU only, numbers only). usage: T43W_SNAP=${AG3S_PROJECT_ROOT}-t43v11 compare_abc.py OUT_JSON
Reads (does not modify) V11/{agg,ct,extra,sharp,xm_summary}_{V11A,V11AB}.json, V11/crossing_pairs_{V11A,V11AB}.json, V11/analysis/analysis_{V11A,V11AB}.json, V8/analysis/analysis.json,
load samples (V8/raw, V11/raw, V11/AB). Arms: V8 / V11A / V11AB, paired by (stage, episode, seed).
Sections:
 pairs3        : S / G / P (old rule, H / B / ALL), Z4 rule and CT rule (H) totals per arm and exact McNemar for V11AB-V8, V11AB-V11A, V11A-V8.
 fail_stage    : for each run with S_old true in V8 and false in V11A or V11AB: stage in {no_grasp, dropped, place_false, success_other} + evidence (grasp_t, first detach/slip chunk t and whether the apple was over the crate then,
                 apple final in crate frame, release_t, Z4 / CT flags). dropped := grasp true, place false, and (first grasp_detach / slip chunk with the apple NOT over the crate, or Z4 release_t exists with the apple outside the crate xy); contact := grasp true, place false, not dropped, run ended by an obstacle stop (n_control < 600); place_false := grasp true, place false otherwise.
                 CT-rule failures of runs that pass the old rule are listed separately as contact.
 grasp_phase   : per run (19) and per arm: G, grasp_t, closing chunks (grasp.state closing, not attached): n, n outward > 2 mm, max / median outward_mm (analyze_t43w focus: row-7 fingertip midpoint of the active arm,
                 refined vs reference, distance to apple centre); first closing chunk: d_refined, d_reference, outward, measured fingertip midpoint-apple distance at the chunk's planning row (qpos of the row).
 crossing3     : commanded vs measured crossing clearance per run for the three arms and the V11AB - V11A paired differences.
 hold3, follow_to3, timing3 (+ loadavg per arm, per-run load vs TO median), contacts3.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, math, collections, re, datetime, os
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); V = T / "V11"; AB = V / "AB"; ABC = V / "ABC"
ARMS = ("V8", "V11A", "V11AB", "V11ABC"); NEW = ("V11A", "V11AB", "V11ABC")
PAIRS = (("V11ABC", "V8"), ("V11ABC", "V11AB"), ("V11ABC", "V11A"), ("V11AB", "V8"), ("V11A", "V8"), ("V11AB", "V11A"))


def mc(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def pair(fa, fb, keys=None):
    """fa, fb: dict key -> bool (A, B). Returns A/B totals, discordant counts and exact McNemar p."""
    ks = sorted(set(fa) & set(fb)) if keys is None else keys
    both = sum(fa[k] and fb[k] for k in ks); ao = sum(fa[k] and not fb[k] for k in ks); bo = sum(fb[k] and not fa[k] for k in ks)
    return {"n_pairs": len(ks), "A_total": both + ao, "B_total": both + bo, "both": both, "A_only": ao, "B_only": bo, "neither": len(ks) - both - ao - bo,
            "mcnemar_exact_p": mc(ao, bo), "A_only_keys": [str(k) for k in ks if fa[k] and not fb[k]], "B_only_keys": [str(k) for k in ks if fb[k] and not fa[k]]}


def st(v):
    v = [float(x) for x in v if x is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": len(a), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max()), "mean": float(a.mean())}


AN = {"V8": json.load(open(T / "V8/analysis/analysis.json"))["runs"], "V11A": json.load(open(V / "analysis/analysis_V11A.json"))["runs"], "V11AB": json.load(open(V / "analysis/analysis_V11AB.json"))["runs"], "V11ABC": json.load(open(V / "analysis/analysis_V11ABC.json"))["runs"]}
CT = {a: json.load(open(V / f"ct_{a}.json")) for a in NEW}
AGG = {a: json.load(open(V / f"agg_{a}.json")) for a in NEW}
EX = {a: json.load(open(V / f"extra_{a}.json")) for a in NEW}


def runs_of(arm):
    out = {}
    for k, r in AN[arm].items():
        if not k.startswith(arm + "/"):
            continue
        nm = k.split("/")[2]
        out[(k.split("/")[1], int(nm.split("_ep")[1].split("_s")[0]), int(nm.split("_s")[-1]))] = (k, r)
    return out


RUNS = {a: runs_of(a) for a in ARMS}
CTR = {}
for a in ARMS:
    src = CT["V11A"] if a == "V8" else CT[a]
    CTR[a] = {(o["stage"], o["episode"], o["seed"]): o for o in src["runs"].values() if o["arm"] == a}
out = {"arms": list(ARMS), "n_runs": {a: len(RUNS[a]) for a in ARMS}}

# ------------------------------------------------------------------ pairs3
p3 = {}
tot = {}
for stg in ("H", "B", "ALL"):
    tot[stg] = {}
    for a in ARMS:
        L = [r for kk, (k, r) in RUNS[a].items() if stg == "ALL" or kk[0] == stg]
        tot[stg][a] = {"n": len(L), "S": sum(bool(r["success"]) for r in L), "G": sum(bool(r["grasp"]) for r in L), "P": sum(bool(r["place"]) for r in L)}
    for A, B in PAIRS:
        for f, nm in (("success", "S"), ("grasp", "G"), ("place", "P")):
            fa = {kk: bool(r[f]) for kk, (k, r) in RUNS[A].items() if stg == "ALL" or kk[0] == stg}
            fb = {kk: bool(r[f]) for kk, (k, r) in RUNS[B].items() if stg == "ALL" or kk[0] == stg}
            p3.setdefault(f"{A}_vs_{B}", {}).setdefault(stg, {})[nm] = pair(fa, fb)
out["totals_old_rule"] = tot
out["pairs_old_rule"] = p3
ztot = {a: {f: sum(bool(o[f]) for o in CTR[a].values()) for f in ("success", "grasp", "place", "released", "place_rel", "success_rel", "success_ct")} for a in ARMS}
zp = {}
for A, B in PAIRS:
    for f in ("grasp", "released", "place_rel", "success_rel", "success_ct"):
        zp.setdefault(f"{A}_vs_{B}", {})[f] = pair({k: bool(o[f]) for k, o in CTR[A].items()}, {k: bool(o[f]) for k, o in CTR[B].items()})
out["H_totals_by_rule"] = ztot
out["H_pairs_z4_ct_rule"] = zp

# ------------------------------------------------------------------ contacts
cta = {}
for a in ARMS:
    L = list(CTR[a].values())
    cta[a] = {"n": len(L), "S_old": sum(o["success"] for o in L), "S_release": sum(o["success_rel"] for o in L), "S_contact": sum(o["success_ct"] for o in L),
              "pre_release_contact_runs": sum(o["pre_release_contact"] for o in L), "post_release_contact_runs": sum(o["post_release_contact"] for o in L),
              "obstacle_stop_runs": sum(o["obstacle_stop"] for o in L), "any_contact_runs": sum(bool(o["contacts"]) for o in L),
              "pre_release_contact_by_kind": dict(collections.Counter(k for o in L for k in o["pre_release_kinds"])),
              "pre_release_contact_keys": sorted(f"{o['episode']}_s{o['seed']}" for o in L if o["pre_release_contact"]),
              "post_release_contact_keys": sorted(f"{o['episode']}_s{o['seed']}" for o in L if o["post_release_contact"])}
out["contacts_arms"] = cta
tab = {}
for a in ARMS:
    for o in CTR[a].values():
        for c in o["contacts"]:
            e = tab.setdefault((c["body"], c["kind"], c["phase"]), {"body": c["body"], "kind": c["kind"], "phase": c["phase"], "runs": {x: [] for x in ARMS}, "n_rows": {x: 0 for x in ARMS},
                                                                      "max_pen_mm": {x: 0.0 for x in ARMS}, "max_force_N": {x: 0.0 for x in ARMS}, "first_t": {x: [] for x in ARMS}})
            e["runs"][a].append(f"{o['episode']}_s{o['seed']}"); e["n_rows"][a] += c["n_rows"]; e["max_pen_mm"][a] = max(e["max_pen_mm"][a], c["max_pen_mm"])
            e["max_force_N"][a] = max(e["max_force_N"][a], c["max_force_N"]); e["first_t"][a].append(c["first_t"])
rows = []
for e in tab.values():
    e["n_runs"] = {a: len(e["runs"][a]) for a in ARMS}; rows.append(e)
rows.sort(key=lambda e: ({"before_grasp": 0, "carry": 1, "after_release": 2}[e["phase"]], -sum(e["n_runs"].values()), e["body"]))
out["contact_table_by_body_phase"] = rows
fb_ = {}
for a in ARMS:
    byph = {}
    for ph in ("before_grasp", "carry", "after_release"):
        rs = set(); nr = 0; pen = 0.0
        for e in rows:
            if e["phase"] == ph and e["body"].startswith("ee_finger"):
                rs |= set(e["runs"][a]); nr += e["n_rows"][a]; pen = max(pen, e["max_pen_mm"][a])
        byph[ph] = {"n_runs": len(rs), "runs": sorted(rs), "n_rows": nr, "max_pen_mm": pen}
    fb_[a] = byph
out["finger_hurdle_contacts_by_phase"] = fb_
out["obstacle_stop"] = {a: sorted(o["key"] for o in CTR[a].values() if o["obstacle_stop"]) for a in ARMS}

# ------------------------------------------------------------------ failure stages
sys.argv_saved = list(sys.argv)


def qmat(q):
    import mujoco
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def over_crate(p):
    op = p["object_poses"]
    loc = qmat(op["crate"]["quat"]).T @ (np.asarray(op["apple"]["pos"], float) - np.asarray(op["crate"]["pos"], float))
    return bool(abs(loc[0]) < 0.077 and abs(loc[1]) < 0.137)


def plan_rows(r):
    out_ = {}
    for l in open(pathlib.Path(r["run_dir"]) / "frames.jsonl"):
        if l.strip():
            x = json.loads(l)
            if x["kind"] == "planning":
                out_[x["seq"]] = x
    return out_


def stage_of(arm, key):
    k, r = RUNS[arm][key]
    ev = {"success": bool(r["success"]), "grasp": bool(r["grasp"]), "place": bool(r["place"]), "grasp_t": r["grasp_t"], "place_t": r["place_t"], "apple_dz_max_mm": r["apple_dz_max_mm"],
          "apple_dz_final_mm": r["apple_dz_final_mm"], "apple_final_in_crate_frame_mm": None if r.get("apple_final_in_crate_frame") is None else [1e3 * x for x in r["apple_final_in_crate_frame"]]}
    o = CTR[arm].get(key)
    if o:
        ev.update({"release_t": o["release_t"], "released_z4": o["released"], "success_rel": o["success_rel"], "success_ct": o["success_ct"], "pre_release_contact": o["pre_release_contact"],
                   "post_release_contact": o["post_release_contact"], "in_crate_end": o["in_crate_end"]})
    plan = plan_rows(r)
    det = []
    for c in r["server_chunks"]:
        if c.get("grasp_detach") or c.get("held_slip") or c.get("attach_revoked"):
            det.append({"seq": c["seq"], "t": c["t"], "grasp_detach": bool(c.get("grasp_detach")), "held_slip": bool(c.get("held_slip")),
                        "attach_revoked_reasons": (c.get("attach_revoked") or {}).get("reasons") if isinstance(c.get("attach_revoked"), dict) else None,
                        "apple_over_crate": over_crate(plan[c["seq"]]) if c["seq"] in plan else None, "grasp_state": c.get("grasp_state")})
    ev["detach_slip_chunks"] = det
    if o:
        ev["obstacle_stop"] = bool(o["obstacle_stop"]); ev["n_control"] = o["n_control_rows"]
        ev["in_crate_xy_at_release"] = o.get("in_crate_xy_at_release"); ev["apple_in_crate_at_release_mm"] = o.get("apple_in_crate_at_release_mm")
        ev["contacts_before_or_at_release"] = [{"body": c["body"], "kind": c["kind"], "phase": c["phase"], "first_t": c["first_t"], "max_pen_mm": c["max_pen_mm"]} for c in o["contacts"] if c["phase"] != "after_release"]
    if not r["grasp"]:
        s = "no_grasp"
    elif not r["place"]:
        first = next((d for d in det if d["grasp_detach"] or d["held_slip"]), None)
        if (first is not None and first["apple_over_crate"] is False) or (o and o.get("release_t") is not None and o.get("in_crate_xy_at_release") is False):
            s = "dropped"   # apple left the hand away from the crate (an obstacle stop after that is recorded in the evidence: obstacle_stop / n_control)
        elif o and o["obstacle_stop"] and o["n_control_rows"] < 600:
            s = "contact"
        else:
            s = "place_false"
    elif not r["success"]:
        s = "success_other"
    else:
        s = "success"
    return s, ev


fs = {}
for key in sorted(set(RUNS["V8"]) & set(RUNS["V11A"]) & set(RUNS["V11AB"]) & set(RUNS["V11ABC"])):
    nm = f"{key[0]}:{key[1]}_s{key[2]}"
    row = {"V8_success": bool(RUNS["V8"][key][1]["success"])}
    for a in ARMS:
        s_, ev = stage_of(a, key)
        row[a] = {"stage": s_, **ev}
    # S_old false in some V11 arm while true in V8 or another V11 arm
    succ = [bool(RUNS[a][key][1]["success"]) for a in ARMS]
    if any(succ) and any(not RUNS[a][key][1]["success"] for a in NEW):
        fs[nm] = row
out["fail_stage_runs_fail_in_some_V11_arm_success_in_another_arm"] = fs
out["fail_stage_counts"] = {a: dict(collections.Counter(v[a]["stage"] for v in fs.values())) for a in ARMS}
# every run's stage, all arms (for the record)
allst = {}
for key in sorted(RUNS["V8"]):
    allst[f"{key[0]}:{key[1]}_s{key[2]}"] = {a: stage_of(a, key)[0] for a in ARMS if key in RUNS[a]}
out["stage_all_runs"] = allst
out["ct_contact_fail_among_old_success"] = {a: sorted(f"{o['episode']}_s{o['seed']}" for o in CTR[a].values() if o["success"] and not o["success_ct"]) for a in ARMS}
out["ct_contact_fail_V8_ct_success_but_not_in_arm"] = {a: sorted(f"{o['episode']}_s{o['seed']}" for k, o in CTR[a].items() if CTR["V8"].get(k) and CTR["V8"][k]["success_ct"] and not o["success_ct"]) for a in NEW}

# ------------------------------------------------------------------ grasp phase
import hurdle_metrics as HM  # noqa: E402
import mujoco  # noqa: E402


def measured_tip_apple(plan_row, side):
    M = HM.model(); m, d = M["m"], M["d"]
    fl = "l" if side == "left" else "r"
    b1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ee_finger_{fl}1"); b2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"ee_finger_{fl}2")
    d.qpos[:] = plan_row["qpos"]; mujoco.mj_kinematics(m, d)
    tip = 0.5 * (d.xpos[b1] + d.xpos[b2]); ap = np.asarray(plan_row["object_poses"]["apple"]["pos"], float)
    return 1e3 * float(np.linalg.norm(tip - ap))


gp = {}
for a in ARMS:
    per = {}
    for key, (k, r) in sorted(RUNS[a].items()):
        sc = r["server_chunks"]; cl = [c for c in sc if c.get("closing_tip")]
        ent = {"grasp": bool(r["grasp"]), "grasp_t": r["grasp_t"], "success": bool(r["success"]), "side": r["side"], "n_closing_chunks": len(cl),
               "outward_mm_max": max([c["closing_tip"]["outward_mm"] for c in cl], default=None),
               "outward_mm_median": float(np.median([c["closing_tip"]["outward_mm"] for c in cl])) if cl else None,
               "n_outward_gt_2mm": sum(c["closing_tip"]["outward_mm"] > 2.0 for c in cl),
               "shift_mm_median": float(np.median([c["closing_tip"]["shift_mm"] for c in cl])) if cl else None}
        o_ = CTR[a].get(key); rel_t = o_["release_t"] if o_ else None
        clp = [c for c in cl if rel_t is None or c["t"] < rel_t]   # closing chunks before the apple left the hand (Z4 release_t; B runs have no replay: all)
        ent["pre_release"] = {"release_t": rel_t, "n_closing_chunks": len(clp), "outward_mm_max": max([c["closing_tip"]["outward_mm"] for c in clp], default=None),
                              "n_outward_gt_2mm": sum(c["closing_tip"]["outward_mm"] > 2.0 for c in clp)}
        if cl:
            c0 = cl[0]; plan = plan_rows(r)
            ent["first_closing"] = {"seq": c0["seq"], "t": c0["t"], "d_refined_mm": c0["closing_tip"]["d_refined_mm"], "d_reference_mm": c0["closing_tip"]["d_reference_mm"],
                                    "outward_mm": c0["closing_tip"]["outward_mm"], "shift_mm": c0["closing_tip"]["shift_mm"],
                                    "measured_tip_apple_mm_at_chunk_start": measured_tip_apple(plan[c0["seq"]], r["side"]) if c0["seq"] in plan else None}
        per[f"{key[0]}:{key[1]}_s{key[2]}"] = ent
    allc = [c["closing_tip"]["outward_mm"] for kk, (k, r) in RUNS[a].items() for c in r["server_chunks"] if c.get("closing_tip")]
    fcd = [e["first_closing"]["d_refined_mm"] for e in per.values() if "first_closing" in e]
    fcm = [e["first_closing"]["measured_tip_apple_mm_at_chunk_start"] for e in per.values() if "first_closing" in e]
    fco = [e["first_closing"]["outward_mm"] for e in per.values() if "first_closing" in e]
    allp = [c["closing_tip"]["outward_mm"] for kk, (k, r) in RUNS[a].items() for c in r["server_chunks"] if c.get("closing_tip") and ((CTR[a].get(kk) or {}).get("release_t") is None or c["t"] < CTR[a][kk]["release_t"])]
    gp[a] = {"closing_chunks_pooled_pre_release": {"n": len(allp), "n_outward_gt_2mm": sum(x > 2.0 for x in allp), "outward_mm": st(allp)},
             "outward_mm_max_per_run_pre_release": st([e["pre_release"]["outward_mm_max"] for e in per.values()]),
             "G": sum(e["grasp"] for e in per.values()), "n_runs": len(per), "closing_chunks_pooled": {"n": len(allc), "n_outward_gt_2mm": sum(x > 2.0 for x in allc), "outward_mm": st(allc)},
             "outward_mm_max_per_run": st([e["outward_mm_max"] for e in per.values()]), "first_closing_d_refined_mm": st(fcd), "first_closing_outward_mm": st(fco),
             "first_closing_measured_tip_apple_mm_at_chunk_start": st(fcm), "per_run": per}
out["grasp_phase"] = gp
gpp = {}
for A, B in PAIRS:
    ks = sorted(set(gp[A]["per_run"]) & set(gp[B]["per_run"])); row = {}
    for f, g in (("outward_mm_max_pre_release", lambda e: e["pre_release"]["outward_mm_max"]), ("n_outward_gt_2mm_pre_release", lambda e: e["pre_release"]["n_outward_gt_2mm"]),
                 ("outward_mm_max", lambda e: e["outward_mm_max"]), ("outward_mm_median", lambda e: e["outward_mm_median"]), ("n_outward_gt_2mm", lambda e: e["n_outward_gt_2mm"]),
                 ("first_closing_d_refined_mm", lambda e: (e.get("first_closing") or {}).get("d_refined_mm")), ("first_closing_outward_mm", lambda e: (e.get("first_closing") or {}).get("outward_mm")),
                 ("first_closing_measured_tip_apple_mm", lambda e: (e.get("first_closing") or {}).get("measured_tip_apple_mm_at_chunk_start"))):
        d = [g(gp[A]["per_run"][k]) - g(gp[B]["per_run"][k]) for k in ks if g(gp[A]["per_run"][k]) is not None and g(gp[B]["per_run"][k]) is not None]
        row[f] = {"n_pairs": len(d), "diff_A_minus_B": st(d), "n_A_higher": int(sum(x > 0 for x in d)), "n_A_lower": int(sum(x < 0 for x in d))}
        if len(d) >= 5 and any(x != 0 for x in d):
            from scipy.stats import wilcoxon
            try:
                row[f]["wilcoxon_p"] = float(wilcoxon(d).pvalue)
            except Exception:
                row[f]["wilcoxon_p"] = None
    gpp[f"{A}_vs_{B}"] = row
out["grasp_phase_pairs"] = gpp

# ------------------------------------------------------------------ crossing3
cp = {a: json.load(open(V / f"crossing_pairs_{a}.json")) for a in NEW}
perrun = {}
for k in sorted(cp["V11ABC"]["per_run"]):
    perrun[k] = {"V8": cp["V11ABC"]["per_run"][k]["V8"], **{a: cp[a]["per_run"].get(k, {}).get(a) for a in NEW}}
fields = sorted({f for v in perrun.values() for a_ in v.values() if a_ for f in a_})
arms_c = {a: {f: st([v[a][f] for v in perrun.values() if v[a] and v[a].get(f) is not None]) for f in fields} for a in ARMS}
pc = {}
for A, B in PAIRS:
    pc[f"{A}_vs_{B}"] = {}
    for f in fields:
        d = [v[A][f] - v[B][f] for v in perrun.values() if v[A] and v[B] and v[A].get(f) is not None and v[B].get(f) is not None]
        pc[f"{A}_vs_{B}"][f] = {"n_pairs": len(d), "diff_A_minus_B": st(d), "n_A_higher": int(sum(x > 0 for x in d)), "n_A_lower": int(sum(x < 0 for x in d))}
out["crossing3"] = {"definitions": cp["V11ABC"]["definitions"], "per_run": perrun, "arms": arms_c, "paired": pc}
xs = {a: json.load(open(V / f"xm_summary_{a}.json")) for a in NEW}
xarm = {"V8": xs["V11ABC"]["arms"]["V8"], **{a: xs[a]["arms"][a] for a in NEW}}
keep = ("n_chunks", "n_runs", "palm_D_max_mm", "n_deflected_gt5", "deflected_components_at_kstar_mm", "deflected_share", "deflected_dominant_axis", "deflected_component_fraction_median", "executed_rows",
        "initial_source", "n_ref_violating", "hold_chunks", "binding_rows", "sqp", "max_step", "continuity_palm_mm")
out["deflection_direction"] = {a: {g: {kk: xarm[a][g].get(kk) for kk in keep} for g in ("carry_apple", "carry_empty", "return")} for a in ARMS}
out["crossing_stats_xm"] = {a: {"crossing": xarm[a]["crossing"], "binding_at_palm_crossing": xarm[a]["binding_at_palm_crossing"]} for a in ARMS}

# ------------------------------------------------------------------ sharpness
sh = {a: json.load(open(V / f"sharp_{a}.json")) for a in NEW}
shd = {"V8": sh["V11ABC"]["arms"]["V8"], **{a: sh[a]["arms"][a] for a in NEW}}
out["sharpness"] = {a: {s: {k: shd[a][s][k] for k in ("n_runs", "n_boundaries_clean", "n_boundaries_excluded_hold", "acc_boundary_rad", "acc_inner_chunk_max_rad", "vel_boundary_ratio", "n_vel_boundary_gt1",
                                                           "palm_turn_deg", "palm_reversal_share", "palm_reversal_n", "palm_acc_boundary_mm")} for s in ("H", "B", "ALL")} for a in ARMS}
# paired per-run boundary acceleration median / reversal share
pr = {}
for a in ARMS:
    pr[a] = {k.split("/")[-1]: r for k, r in shd[a]["per_run"].items()}
sp = {}
for A, B in PAIRS:
    ks = sorted(set(pr[A]) & set(pr[B]).copy()) if False else sorted({k.split("_ep")[1] for k in pr[A]} & {k.split("_ep")[1] for k in pr[B]})
    d = []
    for kk in ks:
        ra = next(v for n, v in pr[A].items() if n.endswith("_ep" + kk)); rb = next(v for n, v in pr[B].items() if n.endswith("_ep" + kk))
        if ra["acc_boundary_rad"] and rb["acc_boundary_rad"]:
            d.append(float(np.median(ra["acc_boundary_rad"])) - float(np.median(rb["acc_boundary_rad"])))
    sp[f"{A}_vs_{B}"] = {"per_run_median_boundary_acc_diff_rad": st(d), "n_A_higher": int(sum(x > 0 for x in d)), "n_A_lower": int(sum(x < 0 for x in d))}
out["sharpness_paired"] = sp

# ------------------------------------------------------------------ HOLD / follow_to
hold = {}
hb = {"V8": AGG["V11ABC"]["hold"]["V8"], **{a: AGG[a]["hold"][a] for a in NEW}}
for a in ARMS:
    h = hb[a]
    hold[a] = {k: h[k] for k in ("total", "trailing", "runs_with_hold", "by_category", "by_target", "by_phase", "by_binding_link", "cells", "hold_kinds", "by_stage_total")}
    hold[a]["per_run_chunks"] = {k: v["chunks"] for k, v in h["per_run"].items() if v["n_hold"]}


def hold_pair(A, B):
    o = {"A_only": [], "B_only": [], "both_n": 0, "pairs": 0}
    for run in sorted(set(hb[A]["per_run"]) & set(hb[B]["per_run"])):
        a_ = {c["t"]: c for c in hb[A]["per_run"][run]["chunks"]}; b_ = {c["t"]: c for c in hb[B]["per_run"][run]["chunks"]}
        o["pairs"] += 1; o["both_n"] += len(set(a_) & set(b_))
        o["A_only"] += [dict(a_[t], run=run) for t in sorted(set(a_) - set(b_))]; o["B_only"] += [dict(b_[t], run=run) for t in sorted(set(b_) - set(a_))]
    for f in ("A_only", "B_only"):
        L = o[f]; o[f + "_n"] = len(L)
        o[f + "_by_category"] = dict(collections.Counter(str(x["category"]) for x in L)); o[f + "_by_target"] = dict(collections.Counter(str(x["target"]) for x in L))
        o[f + "_by_phase"] = dict(collections.Counter(x["phase"] for x in L)); o[f + "_by_binding_link"] = dict(collections.Counter(str(x["binding_link"]) for x in L))
    return o


hold["pairs"] = {f"{A}_vs_{B}": hold_pair(A, B) for A, B in PAIRS}
out["hold3"] = hold
fo = {"V8": AGG["V11ABC"]["follow_to"]["V8"], **{a: AGG[a]["follow_to"][a] for a in NEW}}
out["follow_to3"] = {a: {k: fo[a][k] for k in ("decision_counts", "refused_counts", "n_executed", "n_runs_with_executed", "executed")} for a in ARMS}
out["follow_to3"]["refused_by_run"] = {a: {k: v["refused"] for k, v in fo[a]["per_run"].items() if v["refused"]} for a in ARMS}
out["verdict_kind_counts"] = {"V8": EX["V11ABC"]["verdict_kind_counts_V8"], **{a: EX[a][f"verdict_kind_counts_{a}"] for a in NEW}}

# ------------------------------------------------------------------ timing + load
tmg = {"V8": AGG["V11ABC"]["timing"]["V8"], **{a: AGG[a]["timing"][a] for a in NEW}}
keys_t = ("n_chunks_excl_first", "to_solve_ms", "verdict_trajopt_ms", "verdict_total_ms", "round_trip_ms", "round_trip_gt_533_3_n", "round_trip_n", "round_trip_gt_533_3_share", "time_budget_hit_count",
          "runs_with_time_budget_hit", "linearize_ms", "qp_ms", "check_ms", "sqp_iterations")
out["timing3"] = {a: {s: {k: tmg[a][s].get(k) for k in keys_t} for s in ("ALL", "H", "B")} for a in ARMS}
out["raw_to_from_server_records"] = {"V8": EX["V11ABC"]["raw_to"]["V8"], **{a: EX[a]["raw_to"][a] for a in NEW}}


def load_file(p):
    L = [l.split() for l in open(p) if len(l.split()) >= 6 and l.split()[-1].startswith("wset=")]
    return np.array([[float(x[0]), float(x[1]), float(x[2]), float(x[3]), int(x[-1].split("=")[1])] for x in L])   # epoch, load1, load5, load15, wset


LF = {"V8": T / "V8/raw/load_samples.txt", "V11A": V / "raw/load_samples.txt", "V11AB": AB / "load_samples_V11AB.txt", "V11ABC": ABC / "load_samples_V11ABC.txt"}
LD = {a: load_file(LF[a]) for a in ARMS}
out["loadavg_per_round"] = {a: {"n_samples": int(len(LD[a])), "load1_median": float(np.median(LD[a][:, 1])), "load1_p90": float(np.percentile(LD[a][:, 1], 90)), "load1_max": float(LD[a][:, 1].max()),
                                "load1_mean": float(LD[a][:, 1].mean()), "working_set_gib_max": float(LD[a][:, 4].max() / 2 ** 30), "start_epoch": int(LD[a][0, 0]), "end_epoch": int(LD[a][-1, 0]),
                                "file": str(LF[a])} for a in ARMS}
RUNDIR = {"V8": T / "V8/runs/V8", "V11A": V / "runs/V11A", "V11AB": V / "runs/V11AB", "V11ABC": V / "runs/V11ABC"}


def iso(p):
    return datetime.datetime.fromisoformat(open(p).read().splitlines()[0].strip()).timestamp()


perrun_load = {}
for a in ARMS:
    for key, (k, r) in sorted(RUNS[a].items()):
        nm = k.split("/")[2]; st_ = key[0]
        try:
            t0, t1 = iso(RUNDIR[a] / st_ / f"{nm}.start"), iso(RUNDIR[a] / st_ / f"{nm}.end")
        except Exception:
            continue
        m = (LD[a][:, 0] >= t0) & (LD[a][:, 0] <= t1)
        sv = [c["solve_ms"] for c in r["server_chunks"][1:] if c["solve_ms"] is not None]
        rt = [c["client_round_trip_ms"] for c in r["chunks"][1:] if c["client_round_trip_ms"] is not None]
        perrun_load[f"{a}/{key[0]}:{key[1]}_s{key[2]}"] = {"arm": a, "load1_mean_during_run": float(LD[a][m, 1].mean()) if m.any() else None, "load1_max_during_run": float(LD[a][m, 1].max()) if m.any() else None,
                                                          "n_load_samples": int(m.sum()), "to_solve_median_ms": float(np.median(sv)) if sv else None, "round_trip_median_ms": float(np.median(rt)) if rt else None,
                                                          "wall_s": r["wall_s"]}
out["per_run_load_vs_to"] = perrun_load
out["load1_during_runs"] = {a: st([v["load1_mean_during_run"] for v in perrun_load.values() if v["arm"] == a]) for a in ARMS}
bins = ((0, 30), (30, 60), (60, 1e9)); lb = {}
for a in ARMS:
    lb[a] = {}
    for lo, hi in bins:
        L = [v for v in perrun_load.values() if v["arm"] == a and v["load1_mean_during_run"] is not None and lo <= v["load1_mean_during_run"] < hi and v["to_solve_median_ms"]]
        lb[a][f"load1_{lo}_{hi}"] = {"n_runs": len(L), "to_solve_median_ms_of_run_medians": float(np.median([v["to_solve_median_ms"] for v in L])) if L else None,
                                     "round_trip_median_ms_of_run_medians": float(np.median([v["round_trip_median_ms"] for v in L])) if L else None}
out["to_by_load_bin"] = lb
from scipy.stats import spearmanr  # noqa: E402
corr = {}
for grp in (("V8",), ("V11A",), ("V11AB",), ("V11ABC",), ("V11A", "V11AB", "V11ABC"), ARMS):
    L = [v for v in perrun_load.values() if v["arm"] in grp and v["load1_mean_during_run"] is not None and v["to_solve_median_ms"] is not None]
    if len(L) >= 4:
        r_, p_ = spearmanr([v["load1_mean_during_run"] for v in L], [v["to_solve_median_ms"] for v in L])
        corr["+".join(grp)] = {"n_runs": len(L), "spearman_rho_load1_vs_to_median": float(r_), "p": float(p_)}
out["spearman_load_vs_to_median"] = corr
# per-run paired TO median
ptm = {}
for A, B in PAIRS:
    d = []; rr = []
    for key in sorted(set(RUNS[A]) & set(RUNS[B])):
        a_ = perrun_load.get(f"{A}/{key[0]}:{key[1]}_s{key[2]}"); b_ = perrun_load.get(f"{B}/{key[0]}:{key[1]}_s{key[2]}")
        if a_ and b_ and a_["to_solve_median_ms"] and b_["to_solve_median_ms"]:
            d.append(a_["to_solve_median_ms"] - b_["to_solve_median_ms"]); rr.append(a_["to_solve_median_ms"] / b_["to_solve_median_ms"])
    ptm[f"{A}_vs_{B}"] = {"diff_ms": st(d), "ratio": st(rr), "n_A_higher": int(sum(x > 0 for x in d)), "n_pairs": len(d)}
out["per_run_to_median_pairs"] = ptm
# cpu.stat throttling over the AB round (sampler) and at the V11A time (not sampled)
cs = []
p_ = ABC / "cpu_samples_V11ABC.txt"
if p_.exists():
    for l in open(p_):
        x = l.split()
        if len(x) == 5:
            cs.append([int(x[0])] + [int(y.split("=")[1]) for y in x[1:]])
if len(cs) >= 2:
    cs = np.array(cs)
    out["cpu_throttle_during_ABC_round"] = {"n_samples": len(cs), "start_epoch": int(cs[0, 0]), "end_epoch": int(cs[-1, 0]), "cpu_cores_used_mean": float((cs[-1, 1] - cs[0, 1]) / 1e6 / max(1, cs[-1, 0] - cs[0, 0])),
                                           "nr_periods": int(cs[-1, 2] - cs[0, 2]), "nr_throttled": int(cs[-1, 3] - cs[0, 3]), "throttled_share_of_periods": float((cs[-1, 3] - cs[0, 3]) / max(1, cs[-1, 2] - cs[0, 2])),
                                           "throttled_sec": float((cs[-1, 4] - cs[0, 4]) / 1e6), "cpu_max": "1600000 100000 (16 cores)"}
# ------------------------------------------------------------------ server-log facts for the deflection-rate weight + metrics
out["ta_server_log_lines"] = {a: sorted({l for s in EX[a]["servers"].values() for l in s["log_ta_lines"]}) for a in NEW}
out["deflection_rate_records"] = {a: {"n": EX[a]["raw_to"][a]["deflection_rate_records"], "keys": EX[a]["raw_to"][a]["deflection_rate_keys"]} for a in NEW}
out["servo_accel_records"] = {a: {"n": EX[a]["raw_to"][a]["servo_accel_records"], "keys": EX[a]["raw_to"][a]["servo_accel_keys"]} for a in NEW}
out["lookahead_gate"] = {a: EX[a]["lookahead_gate_open_rate"] for a in NEW}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
print("written", sys.argv[1])
for stg in ("H", "B", "ALL"):
    print(stg, tot[stg])
for pn, v in p3.items():
    print(pn, {s: {f: (x["A_total"], x["B_total"], x["A_only"], x["B_only"], round(x["mcnemar_exact_p"], 3)) for f, x in v[s].items()} for s in v})
print("Z4/CT", ztot)
for pn, v in zp.items():
    print(pn, {f: (x["A_total"], x["B_total"], x["A_only"], x["B_only"], round(x["mcnemar_exact_p"], 3)) for f, x in v.items()})
