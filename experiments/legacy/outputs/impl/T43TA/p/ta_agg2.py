"""T43 TA2 — aggregate the offline re-solves (raw/*.json, AG3S.trajopt.experiments.ta_offline).

usage: ta_agg2.py OUT_JSON PREFIX_GLOB   (e.g. 'raw/V8_*.json')

Chunk sets (fixed across variants, chosen on the *base* re-solve):
  crossing = carry/crate/return chunks whose base commanded rows come within 50 mm (geometric d − r, obstacle
             class, nearest surface outside the destination box) of an obstacle in any group (palm / fingers / held).
  grasp    = chunks in grasp state `closing`, the attach chunk, and the 2 chunks before the first closing chunk.
Per variant (medians over chunks unless said):
  hurdle_cmd / hurdle_servo: per chunk min over rows 0-7 and groups of that clearance (mm), commanded rows /
             servo-predicted states (4 sub-steps per row, evaluator only).
  dev_palm:  per chunk max over rows 0-7 of |palm(variant) − palm(policy)| (mm).
  acc_bd:    max joint |u[k] − 2u[k−1] + u[k−2]| over (u−2, u−1, Q0, Q1) (rad/row²); acc_in: inside rows 0-7.
  reversal:  share of chunks whose palm direction turns > 90° at the chunk start (displacements > 2 mm).
  vel_bd>1:  chunks with |Q0 − u[−1]| > max_step on some joint.
  new_viol:  chunks with collision rows (classify_violations, rows 0-7) in the variant and none in base, per phase.
  contacts:  HX contact rows of these runs: servo-predicted clearance of that group at row k ≥ 0 / TO clearance ≥ 0.
  cpu:       per chunk median(cpu_ms variant) / median(cpu_ms base) (repeats interleaved); and Σ variant / Σ base.
  gate_open: share of chunks whose look-ahead tail was used (A with a gate).
  grasp:     vs base: chunks moved > 0.5°, palm |Δ| (mm), outward Δ (finger midpoint − target at row 7, mm; > 0 = farther
             from the apple than base), new collision chunks, gate-open chunks.
"""
import glob
import json
import sys

import numpy as np

PHASES = ("approach", "closing", "carry", "crate", "return")


def stats(a, nd=3):
    a = [x for x in a if x is not None and np.isfinite(x)]
    if not a:
        return {"n": 0}
    v = np.asarray(a, float)
    return {"n": int(v.size), "p10": round(float(np.quantile(v, .1)), nd), "median": round(float(np.median(v)), nd),
            "p90": round(float(np.quantile(v, .9)), nd), "min": round(float(v.min()), nd), "max": round(float(v.max()), nd)}


def cmin(block, key="hurdle_mm"):
    out = []
    for g in ("palm", "fingers", "held"):
        b = block.get(g)
        if b:
            out += [x for x in b[key] if x is not None]
    return min(out) if out else None


def main():
    out_path = sys.argv[1]
    files = sorted({f for g in sys.argv[2:] for f in glob.glob(g)})
    runs = [json.load(open(f)) for f in files]
    variants = [v for v in runs[0]["variants"] if all(v in r["variants"] for r in runs)]
    keys = ("hurdle_cmd", "hurdle_servo", "dev_palm_all", "dev_palm_cross", "acc_bd_all", "acc_bd_cross", "acc_in_all",
            "rev_all", "rev_cross", "vel_bd", "cpu_ratio", "cpu_ms", "tail_viol_cross", "gate_open", "gate_open_cross",
            "dev_joint_all")
    per = {v: {k: [] for k in keys} for v in variants}
    viol = {v: {p: {"n_chunks": 0, "collision": 0, "new_vs_base": 0, "fixed_vs_base": 0} for p in PHASES} for v in variants}
    grasp = {v: {"n": 0, "moved_gt_0p5deg": 0, "joint_vs_base_deg": [], "palm_vs_base_mm": [], "outward_vs_base_mm": [],
                 "outward_vs_policy_mm": [], "n_outward_gt_2mm_vs_base": 0, "new_collision": 0, "gate_open": 0,
                 "untouched_n": 0, "untouched_moved": 0, "moved_vs_q4": 0, "outward_vs_q4_mm": [],
                 "n_outward_gt_2mm_vs_q4": 0,
                 "at": {"n": 0, "moved": 0, "moved_vs_q4": 0, "out_vs_base": [], "out_vs_q4": [], "out_gt2_base": 0,
                        "out_gt2_q4": 0, "palm_vs_base": []}} for v in variants}
    contacts = {v: {"n": 0, "servo_clear": 0, "cmd_to_clear": 0, "servo_mm": [], "cmd_to_mm": [], "rows": []} for v in variants}
    cpu_sum = {v: 0.0 for v in variants}
    status = {v: {} for v in variants}
    fid, vs_rec = [], []
    n_chunks = n_cross = 0
    for r in runs:
        chunks = [c for c in r["chunks"] if c.get("variants")]
        first_closing = next((i for i, c in enumerate(chunks) if c["phase"] == "closing"), None)
        pre = set() if first_closing is None else {chunks[j]["seq"] for j in range(max(0, first_closing - 2), first_closing)}
        for c in chunks:
            n_chunks += 1
            V = c["variants"]
            base = V["base"]
            if c.get("fidelity"):
                fid.append(c["fidelity"]["abs_diff_mm"])
            vs_rec.append(base["vs_recorded_joint_deg_max"])
            hb = cmin(base["clear_cmd"])
            cross = c["phase"] in ("carry", "crate", "return") and hb is not None and hb < 50.0
            n_cross += cross
            is_grasp = c["phase"] == "closing" or c.get("attach") or c["seq"] in pre
            base_cpu = float(np.median(base["cpu_ms"]))
            for v in variants:
                x = V[v]
                P = per[v]
                status[v][x["status"]] = status[v].get(x["status"], 0) + 1
                dpm = max(x["dev_palm_mm"])
                P["dev_palm_all"].append(dpm)
                P["dev_joint_all"].append(x["dev_joint_deg_max"])
                bd = x["boundary"]
                P["acc_in_all"].append(bd.get("acc_inner_max_rad"))
                if "acc_boundary_max_rad" in bd:
                    P["acc_bd_all"].append(bd["acc_boundary_max_rad"])
                    P["rev_all"].append(1.0 if bd["palm_reversal"] else 0.0)
                    P["vel_bd"].append(bd.get("vel_boundary_ratio"))
                P["cpu_ratio"].append(float(np.median(x["cpu_ms"])) / base_cpu)
                P["cpu_ms"].append(float(np.median(x["cpu_ms"])))
                cpu_sum[v] += float(np.median(x["cpu_ms"]))
                la = x.get("lookahead") or {}
                gate_open = bool(la) and la.get("gate") != "closed"
                P["gate_open"].append(1.0 if gate_open else 0.0)
                ph = c["phase"]
                coll = x["n_collision_rows"] > 0
                bcoll = base["n_collision_rows"] > 0
                viol[v][ph]["n_chunks"] += 1
                viol[v][ph]["collision"] += coll
                viol[v][ph]["new_vs_base"] += coll and not bcoll
                viol[v][ph]["fixed_vs_base"] += (not coll) and bcoll
                if cross:
                    P["hurdle_cmd"].append(cmin(x["clear_cmd"]))
                    P["hurdle_servo"].append(cmin(x["clear_servo"]))
                    P["dev_palm_cross"].append(dpm)
                    P["gate_open_cross"].append(1.0 if gate_open else 0.0)
                    if "acc_boundary_max_rad" in bd:
                        P["acc_bd_cross"].append(bd["acc_boundary_max_rad"])
                        P["rev_cross"].append(1.0 if bd["palm_reversal"] else 0.0)
                    if la and la.get("tail_violation_m") is not None:
                        P["tail_viol_cross"].append(1000.0 * la["tail_violation_m"])
                if is_grasp:
                    G = grasp[v]
                    G["n"] += 1
                    G["moved_gt_0p5deg"] += x["vs_base_joint_deg_max"] > 0.5
                    G["joint_vs_base_deg"].append(x["vs_base_joint_deg_max"])
                    G["palm_vs_base_mm"].append(float(np.max(np.abs(np.asarray(x["dev_palm_mm"]) - np.asarray(base["dev_palm_mm"])))))
                    if x.get("outward_mm") is not None and base.get("outward_mm") is not None:
                        d = x["outward_mm"] - base["outward_mm"]
                        G["outward_vs_base_mm"].append(d)
                        G["outward_vs_policy_mm"].append(x["outward_mm"])
                        G["n_outward_gt_2mm_vs_base"] += d > 2.0
                    G["new_collision"] += coll and not bcoll
                    G["gate_open"] += gate_open
                    q4 = V.get("Q4")
                    if x.get("vs_q4_joint_deg_max") is not None:
                        G["moved_vs_q4"] += x["vs_q4_joint_deg_max"] > 0.5
                    if q4 is not None and x.get("outward_mm") is not None and q4.get("outward_mm") is not None:
                        d4 = x["outward_mm"] - q4["outward_mm"]
                        G["outward_vs_q4_mm"].append(d4)
                        G["n_outward_gt_2mm_vs_q4"] += d4 > 2.0
                    # at the apple: the policy's fingers within 60 mm of the target at row 7, or the attach chunk
                    d_pol = (x["finger_target_mm"] - x["outward_mm"]) if x.get("outward_mm") is not None else None
                    if c.get("attach") or (d_pol is not None and d_pol <= 60.0):
                        A = G["at"]
                        A["n"] += 1
                        A["moved"] += x["vs_base_joint_deg_max"] > 0.5
                        A["moved_vs_q4"] += (x.get("vs_q4_joint_deg_max") or 0.0) > 0.5
                        A["palm_vs_base"].append(G["palm_vs_base_mm"][-1])
                        if x.get("outward_mm") is not None and base.get("outward_mm") is not None:
                            A["out_vs_base"].append(x["outward_mm"] - base["outward_mm"])
                            A["out_gt2_base"] += x["outward_mm"] - base["outward_mm"] > 2.0
                        if q4 is not None and x.get("outward_mm") is not None and q4.get("outward_mm") is not None:
                            A["out_vs_q4"].append(x["outward_mm"] - q4["outward_mm"])
                            A["out_gt2_q4"] += x["outward_mm"] - q4["outward_mm"] > 2.0
                    if base.get("returned") == "initial" and base["dev_joint_deg_max"] < 1e-9:
                        G["untouched_n"] += 1
                        G["untouched_moved"] += x["vs_base_joint_deg_max"] > 0.5
            for cr in c.get("contact_rows", []):
                k, g = cr["k"], cr["group"]
                gk = "held" if g == "apple" else "fingers"
                for v in variants:
                    x = V[v]
                    sv = None
                    if x["clear_servo"].get(gk):
                        vals = [x["clear_servo"][gk]["hurdle_mm"][i] for i in range(4 * k, 4 * k + 4)]
                        vals = [a for a in vals if a is not None]
                        sv = min(vals) if vals else None
                    to = x["clear_cmd"][gk]["to_mm"][k] if x["clear_cmd"].get(gk) else None
                    C = contacts[v]
                    C["n"] += 1
                    C["servo_clear"] += sv is None or sv >= 0.0
                    C["cmd_to_clear"] += to is not None and to >= 0.0
                    C["servo_mm"].append(sv)
                    C["cmd_to_mm"].append(to)
                    C["rows"].append({"key": r["key"], "t": cr["t"], "k": k, "group": g, "servo_mm": sv, "to_mm": to})
    table = {}
    for v in variants:
        P = per[v]
        table[v] = {
            "hurdle_cmd_mm": stats(P["hurdle_cmd"]), "hurdle_servo_mm": stats(P["hurdle_servo"]),
            "n_cross_cmd_lt0": int(sum(1 for a in P["hurdle_cmd"] if a is not None and a < 0)),
            "n_cross_servo_lt0": int(sum(1 for a in P["hurdle_servo"] if a is not None and a < 0)),
            "dev_palm_all_mm": stats(P["dev_palm_all"]), "dev_palm_cross_mm": stats(P["dev_palm_cross"]),
            "dev_joint_all_deg": stats(P["dev_joint_all"]),
            "acc_boundary_all_rad": stats(P["acc_bd_all"], 4), "acc_boundary_cross_rad": stats(P["acc_bd_cross"], 4),
            "acc_inner_all_rad": stats(P["acc_in_all"], 4),
            "reversal_share_all": float(np.mean(P["rev_all"])) if P["rev_all"] else None,
            "reversal_share_cross": float(np.mean(P["rev_cross"])) if P["rev_cross"] else None,
            "n_vel_boundary_gt1": int(sum(1 for a in P["vel_bd"] if a is not None and a > 1.0 + 1e-6)),
            "vel_boundary_ratio": stats(P["vel_bd"]),
            "tail_violation_cross_mm": stats(P["tail_viol_cross"]),
            "gate_open_share": float(np.mean(P["gate_open"])), "gate_open_share_cross": float(np.mean(P["gate_open_cross"])) if P["gate_open_cross"] else None,
            "cpu_ratio": stats(P["cpu_ratio"]), "cpu_sum_ratio": round(cpu_sum[v] / cpu_sum["base"], 3),
            "cpu_ms": stats(P["cpu_ms"], 1),
            "status": status[v], "violations_by_phase": viol[v],
            "new_viol_total": int(sum(viol[v][p]["new_vs_base"] for p in PHASES)),
            "fixed_viol_total": int(sum(viol[v][p]["fixed_vs_base"] for p in PHASES)),
            "grasp": {**{k: grasp[v][k] for k in ("n", "moved_gt_0p5deg", "n_outward_gt_2mm_vs_base", "new_collision", "gate_open",
                                                   "untouched_n", "untouched_moved")},
                      "joint_vs_base_deg": stats(grasp[v]["joint_vs_base_deg"]),
                      "palm_vs_base_mm": stats(grasp[v]["palm_vs_base_mm"]),
                      "outward_vs_base_mm": stats(grasp[v]["outward_vs_base_mm"]),
                      "outward_vs_policy_mm": stats(grasp[v]["outward_vs_policy_mm"]),
                      "moved_vs_q4_gt_0p5deg": grasp[v]["moved_vs_q4"],
                      "outward_vs_q4_mm": stats(grasp[v]["outward_vs_q4_mm"]),
                      "n_outward_gt_2mm_vs_q4": grasp[v]["n_outward_gt_2mm_vs_q4"],
                      "at_apple": {"n": grasp[v]["at"]["n"], "moved_vs_base": grasp[v]["at"]["moved"],
                                   "moved_vs_q4": grasp[v]["at"]["moved_vs_q4"],
                                   "outward_vs_base_mm": stats(grasp[v]["at"]["out_vs_base"]),
                                   "outward_vs_q4_mm": stats(grasp[v]["at"]["out_vs_q4"]),
                                   "n_outward_gt_2mm_vs_base": grasp[v]["at"]["out_gt2_base"],
                                   "n_outward_gt_2mm_vs_q4": grasp[v]["at"]["out_gt2_q4"],
                                   "palm_vs_base_mm": stats(grasp[v]["at"]["palm_vs_base"])}},
            "contacts": {"n": contacts[v]["n"], "servo_clear": contacts[v]["servo_clear"], "cmd_to_clear": contacts[v]["cmd_to_clear"],
                         "servo_mm": stats(contacts[v]["servo_mm"]), "cmd_to_mm": stats(contacts[v]["cmd_to_mm"]),
                         "rows": contacts[v]["rows"]},
        }
    res = {"definitions": __doc__, "files": files, "n_runs": len(runs), "n_chunks": n_chunks, "n_cross": n_cross,
           "fidelity_abs_diff_mm": stats(fid), "fidelity_n_gt_1mm": int(sum(1 for a in fid if a > 1.0)),
           "base_vs_recorded_joint_deg": stats(vs_rec), "variants": runs[0]["variants"], "table": table}
    json.dump(res, open(out_path, "w"), indent=1)
    hdr = ("var", "cmd_p10", "cmd_med", "cmd<0", "srv_p10", "srv_med", "srv<0", "devP_med", "devP_p90", "accbd_p90",
           "accin_p90", "velbd>1", "rev", "newV", "fixV", "contact", "gate", "cpu_med", "cpu_sum")
    print(" ".join(f"{h:>9}" for h in hdr))
    for v in variants:
        t = table[v]
        print(" ".join(f"{str(x):>9}" for x in (
            v, t["hurdle_cmd_mm"].get("p10"), t["hurdle_cmd_mm"].get("median"), t["n_cross_cmd_lt0"],
            t["hurdle_servo_mm"].get("p10"), t["hurdle_servo_mm"].get("median"), t["n_cross_servo_lt0"],
            t["dev_palm_all_mm"].get("median"), t["dev_palm_all_mm"].get("p90"), t["acc_boundary_all_rad"].get("p90"),
            t["acc_inner_all_rad"].get("p90"), t["n_vel_boundary_gt1"], round(t["reversal_share_all"] or 0, 3),
            t["new_viol_total"], t["fixed_viol_total"], f"{t['contacts']['servo_clear']}/{t['contacts']['n']}",
            round(t["gate_open_share"], 3), t["cpu_ratio"].get("median"), t["cpu_sum_ratio"])))
    print("\nat the apple (policy fingers <= 60 mm from the target at row 7, or attach):")
    hdr = ("var", "n", "mv>.5 b", "mv>.5 q4", "out>2 b", "out>2 q4", "outb_p90", "outb_max", "outq_max", "palm_max")
    print(" ".join(f"{h:>9}" for h in hdr))
    for v in variants:
        a = table[v]["grasp"]["at_apple"]
        print(" ".join(f"{str(x):>9}" for x in (
            v, a["n"], a["moved_vs_base"], a["moved_vs_q4"], a["n_outward_gt_2mm_vs_base"], a["n_outward_gt_2mm_vs_q4"],
            a["outward_vs_base_mm"].get("p90"), a["outward_vs_base_mm"].get("max"), a["outward_vs_q4_mm"].get("max"),
            a["palm_vs_base_mm"].get("max"))))
    print("\ngrasp set (vs base):")
    hdr = ("var", "n", "moved>.5", "jmax", "palm_p90", "palm_max", "out>2", "out_p90", "out_max", "newColl", "gateOpen", "untouch_mv")
    print(" ".join(f"{h:>9}" for h in hdr))
    for v in variants:
        g = table[v]["grasp"]
        print(" ".join(f"{str(x):>9}" for x in (
            v, g["n"], g["moved_gt_0p5deg"], g["joint_vs_base_deg"].get("max"), g["palm_vs_base_mm"].get("p90"),
            g["palm_vs_base_mm"].get("max"), g["n_outward_gt_2mm_vs_base"], g["outward_vs_base_mm"].get("p90"),
            g["outward_vs_base_mm"].get("max"), g["new_collision"], g["gate_open"], f"{g['untouched_moved']}/{g['untouched_n']}")))


if __name__ == "__main__":
    main()
