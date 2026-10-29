"""T43 S (4) — summary numbers for the right-arm vs left-arm pick comparison (reads raw/pick_runs.json, raw/pick_ref_target.json,
raw/demo_grasp.json, raw/t42_runs_frozen.json, the runs' frames.jsonl). Writes raw/pick_summary.json.

Groups: cond (E0 / E3b) x arm; 'grasp' = T42-1 rule (target z >= z0 + 50 mm at a planning row; T42/p/analyze_t42.py).
Geometry: rollout row 'settled' = first close command + 16 control rows; demo row 'settled' = grasp_close_start_frame + 16.
Post-close window (frames.jsonl only): planning chunks with t_step in [close_t - 16, close_t + 64): HOLD chunks (not all 8 rows
executed), gate hold kinds, policy-vs-refined arm joint max |d| (deg, used arm, 8 executed steps) from planning 'actions' vs
'actions_reference'; target max dz over the control rows up to close_t + 64."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
import numpy as np

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw'))
R = json.loads((RAW / "pick_runs.json").read_text())["runs"]
P = json.loads((RAW / "pick_ref_target.json").read_text())["runs"]
D = json.loads((RAW / "demo_grasp.json").read_text())["episodes"]
F = json.loads((RAW / "t42_runs_frozen.json").read_text())
GEO = ("tilt_deg", "approach_yaw_deg_mirrored", "closing_yaw_deg_mirrored_mod180", "closing_elev_deg", "tip_dz_mm", "tip_dxy_mm", "grasp_dz_mm", "grasp_dxy_mm")
ARMC = {"left": list(range(0, 7)), "right": list(range(8, 15))}
QADR = {"apple": 38, "banana": 45, "orange": 52, "pear": 59}


def q3(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not v.size: return None
    return {"n": int(v.size), "q25": float(np.percentile(v, 25)), "median": float(np.median(v)), "q75": float(np.percentile(v, 75)),
            "min": float(v.min()), "max": float(v.max())}


def post_close(key):
    r = R[key]; rd = pathlib.Path(F[key]["run_dir"]); ct = r["close_t"]
    if ct is None: return None
    rows = [json.loads(l) for l in rd.joinpath("frames.jsonl").read_text().splitlines() if l.strip()]
    ctrl = {c["t_step"]: c for c in rows if c["kind"] == "control"}; plan = [p for p in rows if p["kind"] == "planning"]
    adr = QADR[r["fruit"]]; z0 = r["fruit_z0"]
    win = [p for p in plan if ct - 16 <= p["t_step"] < ct + 64]
    hold = 0; kinds = {}; dq = []
    for p in win:
        cs = [ctrl[t] for t in range(p["t_step"], p["t_step"] + 8) if t in ctrl]
        if cs and not all(c.get("executed", True) for c in cs):
            hold += 1
            for k in (p.get("gate") or {}).get("hold_kinds") or []:
                kinds[k] = kinds.get(k, 0) + 1
        if p.get("actions_reference") is not None:
            a = np.asarray(p["actions"], float)[:8, ARMC[r["arm"]]]; b = np.asarray(p["actions_reference"], float)[:8, ARMC[r["arm"]]]
            dq.append(float(np.degrees(np.abs(a - b)).max()))
    dz = [1e3 * (ctrl[t]["qpos"][adr + 2] - z0) for t in range(ct, ct + 64) if t in ctrl]
    return {"n_chunks": len(win), "n_hold": hold, "hold_kinds": kinds, "ref_vs_refined_arm_max_deg": dq,
            "ref_vs_refined_arm_max_deg_max": max(dq) if dq else None, "target_dz_max_mm_to_close_p64": max(dz) if dz else None}


def main():
    S = {"definitions": __doc__}
    # outcomes
    out = {}
    for key, r in R.items():
        g = out.setdefault(f"{r['cond']}/{r['arm']}", {"n": 0, "grasp": 0, "success": 0, "by_fruit": {}})
        g["n"] += 1; g["grasp"] += int(r["grasp"]); g["success"] += int(r["success"])
        f = g["by_fruit"].setdefault(r["fruit"], {"n": 0, "grasp": 0, "success": 0})
        f["n"] += 1; f["grasp"] += int(r["grasp"]); f["success"] += int(r["success"])
    S["outcomes"] = out
    # paired E0 / E3b (same episode, seed)
    pairs = []
    for key, r in R.items():
        if r["cond"] != "E0": continue
        k3 = f"E3b/E3b_ep{r['episode']}_s{r['seed']}"
        if k3 in R:
            pairs.append({"episode": r["episode"], "seed": r["seed"], "arm": r["arm"], "fruit": r["fruit"],
                          "E0_grasp": r["grasp"], "E3b_grasp": R[k3]["grasp"], "E0_success": r["success"], "E3b_success": R[k3]["success"],
                          "E0_settled": {k: r["rows"].get("settled", {}).get(k) for k in GEO},
                          "E3b_settled": {k: R[k3]["rows"].get("settled", {}).get(k) for k in GEO}})
    S["paired_E0_E3b"] = sorted(pairs, key=lambda x: (x["arm"], x["episode"], x["seed"]))
    # geometry at settled: rollouts by cond x arm x grasp, apple-only too; demos by arm (all, apple)
    geo = {}
    for cond in ("E0", "E3b"):
        for arm in ("left", "right"):
            for gsel in ("all", "grasp", "no_grasp"):
                for fr in ("all", "apple"):
                    rs = [r for r in R.values() if r["cond"] == cond and r["arm"] == arm and "settled" in r["rows"]
                          and (gsel == "all" or (gsel == "grasp") == bool(r["grasp"])) and (fr == "all" or r["fruit"] == fr)]
                    if rs:
                        geo[f"{cond}/{arm}/{gsel}/{fr}"] = {k: q3([x["rows"]["settled"][k] for x in rs]) for k in GEO}
    for arm in ("left", "right"):
        for fr in ("all", "apple", "orange", "pear", "banana"):
            es = [e for e in D if e["arm"] == arm and (fr == "all" or e["fruit"] == fr) and "settled" in e.get("rows", {})]
            geo[f"demo/{arm}/{fr}"] = {k: q3([e["rows"]["settled"][k] for e in es]) for k in GEO}
    S["geometry_settled"] = geo
    # rollout minus the same episode's demo (settled row)
    dm = {e["episode"]: e for e in D}
    diffs = []
    for key, r in R.items():
        e = dm.get(r["episode"])
        if e is None or "settled" not in r["rows"] or "settled" not in e["rows"]: continue
        diffs.append({"key": key, "cond": r["cond"], "arm": r["arm"], "fruit": r["fruit"], "grasp": r["grasp"],
                      **{f"d_{k}": r["rows"]["settled"][k] - e["rows"]["settled"][k] for k in ("tip_dz_mm", "tip_dxy_mm", "tilt_deg", "approach_yaw_deg_mirrored")},
                      "demo_tip_dz_mm": e["rows"]["settled"]["tip_dz_mm"], "demo_tip_dxy_mm": e["rows"]["settled"]["tip_dxy_mm"]})
    S["rollout_minus_demo_same_episode"] = {
        f"{c}/{a}": {k: q3([x[k] for x in diffs if x["cond"] == c and x["arm"] == a]) for k in ("d_tip_dz_mm", "d_tip_dxy_mm", "d_tilt_deg", "d_approach_yaw_deg_mirrored")}
        for c in ("E0", "E3b") for a in ("left", "right")}
    # policy vs TO target at the close row and +16 (E3b)
    pt = {}
    for key, v in P.items():
        r = R[key]
        for nm in ("close_m8", "close", "close_p8", "close_p16"):
            if nm not in v: continue
            g = pt.setdefault(f"{r['arm']}/{'grasp' if r['grasp'] else 'no_grasp'}/{nm}", {"ref_minus_refined_tip_mm": [], "reference_tip_dxy_mm": [],
                                                                                        "refined_tip_dxy_mm": [], "measured_tip_dxy_mm": [],
                                                                                        "reference_tip_dz_mm": [], "measured_tip_dz_mm": []})
            g["ref_minus_refined_tip_mm"].append(v[nm]["ref_minus_refined_tip_mm"])
            g["reference_tip_dxy_mm"].append(v[nm]["reference"]["tip_dxy_mm"]); g["refined_tip_dxy_mm"].append(v[nm]["refined"]["tip_dxy_mm"])
            g["measured_tip_dxy_mm"].append(v[nm]["measured"]["tip_dxy_mm"])
            g["reference_tip_dz_mm"].append(v[nm]["reference"]["tip_dz_mm"]); g["measured_tip_dz_mm"].append(v[nm]["measured"]["tip_dz_mm"])
    S["policy_vs_TO_target"] = {k: {kk: q3(vv) for kk, vv in g.items()} for k, g in pt.items()}
    # per-run table for the no-grasp E3b runs + E0 no-grasp
    per = {}
    for key, r in R.items():
        pc = post_close(key)
        s = r["rows"].get("settled") or {}
        gc = r.get("grasp_chunks") or []
        per[key] = {"cond": r["cond"], "arm": r["arm"], "fruit": r["fruit"], "grasp": r["grasp"], "success": r["success"], "close_t": r["close_t"],
                    "settled": {k: s.get(k) for k in GEO} | {"n_contacts_by_body": s.get("n_contacts_by_body"), "finger_gap_mm": s.get("finger_gap_mm"),
                                                             "opening_meas": s.get("opening_meas")},
                    "post_close": pc,
                    "grasp_window_TO": [{"seq": c["seq"], "t_step": c["t_step"], "gate": c["gate"], "trajopt_status": c["trajopt_status"],
                                         "to_returned": c["to_returned"], "worst_row": c["worst_row"],
                                         "ref_vs_refined_arm_max_deg": c["ref_vs_refined_arm_max_deg"],
                                         "ref_vs_refined_tcp_mm_max": max(c["ref_vs_refined_tcp_mm_per_step"]),
                                         "ref_vs_refined_gripper_max_abs": c["ref_vs_refined_gripper_max_abs"],
                                         "grasp_state": c["grasp"]["state"], "grasp_hand": c["grasp"]["hand"]} for c in gc],
                    "ref_target": P.get(key)}
    S["per_run"] = per
    # aggregates of the post-close window per cond x arm x grasp
    agg = {}
    for key, v in per.items():
        if v["post_close"] is None: continue
        g = agg.setdefault(f"{v['cond']}/{v['arm']}/{'grasp' if v['grasp'] else 'no_grasp'}", {"runs": 0, "runs_with_hold": 0, "hold_chunks": 0,
                                                                                            "dq_max_per_run": [], "hold_kinds": {}})
        g["runs"] += 1; g["hold_chunks"] += v["post_close"]["n_hold"]; g["runs_with_hold"] += int(v["post_close"]["n_hold"] > 0)
        if v["post_close"]["ref_vs_refined_arm_max_deg_max"] is not None: g["dq_max_per_run"].append(v["post_close"]["ref_vs_refined_arm_max_deg_max"])
        for k, n in v["post_close"]["hold_kinds"].items(): g["hold_kinds"][k] = g["hold_kinds"].get(k, 0) + n
    for g in agg.values(): g["dq_max_per_run"] = q3(g["dq_max_per_run"])
    S["post_close_window"] = agg
    # whole-run audit sums (E3b)
    au = {}
    for key, r in R.items():
        if r["cond"] != "E3b" or "audit" not in r: continue
        a = au.setdefault(r["arm"], {"runs": 0})
        a["runs"] += 1
        for k, v in r["audit"].items():
            if isinstance(v, dict):
                dd = a.setdefault(k, {})
                for kk, n in v.items(): dd[kk] = dd.get(kk, 0) + n
            else:
                a[k] = a.get(k, 0) + v
    S["audit_E3b"] = au
    (RAW / "pick_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps({k: S[k] for k in ("outcomes", "post_close_window", "rollout_minus_demo_same_episode")}, indent=1)[:6000])


if __name__ == "__main__":
    main()
