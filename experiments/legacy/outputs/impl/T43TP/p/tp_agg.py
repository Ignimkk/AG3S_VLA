"""T43 TP — aggregate the offline quality pass (ta_offline raw JSONs) against V11AB (= A16qB20).

usage: tp_agg.py OUT_JSON REF GLOB [GLOB ...]      (REF = A16qB20)

Same chunk sets and definitions as T43 TA (`outputs/impl/T43TA/p/ta_agg2.py`):
  crossing = carry/crate/return chunks whose *base* commanded rows come within 50 mm of an obstacle.
  grasp set = closing chunks, the attach chunk, the 2 chunks before the first closing chunk;
  at the apple = grasp-set chunks whose policy fingers are within 60 mm of the target at row 7, or attach.
Per variant, against REF (V11AB) unless said:
  hurdle_cmd / hurdle_servo  per crossing chunk min clearance (mm), p10 / median, n < 0
  contacts                   HX contact rows: servo-predicted clearance >= 0 (n / 22)
  new / fixed vs REF         chunks with collision rows (classify_violations, rows 0-7) in the variant
                             and none in REF (new), or the reverse (fixed), per phase
  new vs base                as T43 TA
  grasp moved                grasp-set / at-apple chunks whose rows 0-7 move > 0.5 deg vs REF
  outward vs REF             finger midpoint - target distance at row 7, variant - REF (mm), > 2 mm count
  boundary                   acc_boundary p90, reversal share, |Q0 - u[-1]| > max_step count
  gate_open                  share of chunks whose look-ahead tail was used
  status                     counts; qp failures
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
    return {"n": int(v.size), "p10": round(float(np.quantile(v, .1)), nd),
            "median": round(float(np.median(v)), nd), "p90": round(float(np.quantile(v, .9)), nd),
            "min": round(float(v.min()), nd), "max": round(float(v.max()), nd)}


def cmin(block, key="hurdle_mm"):
    out = []
    for g in ("palm", "fingers", "held"):
        b = block.get(g)
        if b:
            out += [x for x in b[key] if x is not None]
    return min(out) if out else None


def main():
    out_path, ref = sys.argv[1], sys.argv[2]
    files = sorted({f for g in sys.argv[3:] for f in glob.glob(g)})
    runs = [json.load(open(f)) for f in files]
    variants = [v for v in runs[0]["variants"] if all(v in r["variants"] for r in runs)]
    T = {v: {"hurdle_cmd": [], "hurdle_servo": [], "acc_bd": [], "rev": [], "vel_bd_gt1": 0,
             "gate": [], "cpu_ms": [], "status": {}, "qp_failures": 0, "n_chunks": 0,
             "new_ref": {p: 0 for p in PHASES}, "fixed_ref": {p: 0 for p in PHASES},
             "new_base": {p: 0 for p in PHASES}, "fixed_base": {p: 0 for p in PHASES},
             "coll": {p: 0 for p in PHASES},
             "g_n": 0, "g_moved": 0, "g_out": [], "g_out_gt2": 0,
             "a_n": 0, "a_moved": 0, "a_out": [], "a_out_gt2": 0,
             "closing_n": 0, "closing_moved": 0, "closing_identical": 0,
             "c_n": 0, "c_servo_clear": 0, "c_cmd_clear": 0, "c_servo_mm": [],
             "joint_vs_ref": []} for v in variants}
    n_cross = 0
    phase_n = {p: 0 for p in PHASES}
    for r in runs:
        chunks = [c for c in r["chunks"] if c.get("variants")]
        first_closing = next((i for i, c in enumerate(chunks) if c["phase"] == "closing"), None)
        pre = set() if first_closing is None else {
            chunks[j]["seq"] for j in range(max(0, first_closing - 2), first_closing)}
        for c in chunks:
            V = c["variants"]
            base, R = V["base"], V[ref]
            hb = cmin(base["clear_cmd"])
            cross = c["phase"] in ("carry", "crate", "return") and hb is not None and hb < 50.0
            n_cross += cross
            ph = c["phase"]
            phase_n[ph] += 1
            is_grasp = c["phase"] == "closing" or c.get("attach") or c["seq"] in pre
            rcoll = R["n_collision_rows"] > 0
            bcoll = base["n_collision_rows"] > 0
            for v in variants:
                x = V[v]
                t = T[v]
                t["n_chunks"] += 1
                t["status"][x["status"]] = t["status"].get(x["status"], 0) + 1
                t["qp_failures"] += int(x.get("qp_failures") or 0)
                t["cpu_ms"].append(float(np.median(x["cpu_ms"])))
                la = x.get("lookahead") or {}
                t["gate"].append(1.0 if (la and la.get("gate") != "closed") else 0.0)
                bd = x["boundary"]
                if "acc_boundary_max_rad" in bd:
                    t["acc_bd"].append(bd["acc_boundary_max_rad"])
                    t["rev"].append(1.0 if bd["palm_reversal"] else 0.0)
                    t["vel_bd_gt1"] += int((bd.get("vel_boundary_ratio") or 0) > 1.0 + 1e-6)
                coll = x["n_collision_rows"] > 0
                t["coll"][ph] += coll
                t["new_ref"][ph] += coll and not rcoll
                t["fixed_ref"][ph] += (not coll) and rcoll
                t["new_base"][ph] += coll and not bcoll
                t["fixed_base"][ph] += (not coll) and bcoll
                jr = x.get("vs_ref_joint_deg_max")
                t["joint_vs_ref"].append(jr)
                if cross:
                    t["hurdle_cmd"].append(cmin(x["clear_cmd"]))
                    t["hurdle_servo"].append(cmin(x["clear_servo"]))
                if c["phase"] == "closing":
                    t["closing_n"] += 1
                    t["closing_moved"] += (jr or 0.0) > 0.5
                    t["closing_identical"] += (jr is not None and jr == 0.0)
                if is_grasp:
                    t["g_n"] += 1
                    t["g_moved"] += (jr or 0.0) > 0.5
                    d = None
                    if x.get("outward_mm") is not None and R.get("outward_mm") is not None:
                        d = x["outward_mm"] - R["outward_mm"]
                        t["g_out"].append(d)
                        t["g_out_gt2"] += d > 2.0
                    d_pol = (x["finger_target_mm"] - x["outward_mm"]) if x.get("outward_mm") is not None \
                        else None
                    if c.get("attach") or (d_pol is not None and d_pol <= 60.0):
                        t["a_n"] += 1
                        t["a_moved"] += (jr or 0.0) > 0.5
                        if d is not None:
                            t["a_out"].append(d)
                            t["a_out_gt2"] += d > 2.0
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
                    t = T[v]
                    t["c_n"] += 1
                    t["c_servo_clear"] += sv is None or sv >= 0.0
                    t["c_cmd_clear"] += to is not None and to >= 0.0
                    t["c_servo_mm"].append(sv)
    table = {}
    ref_cpu = np.asarray(T[ref]["cpu_ms"])
    for v in variants:
        t = T[v]
        cpu = np.asarray(t["cpu_ms"])
        table[v] = {
            "n_chunks": t["n_chunks"],
            "hurdle_cmd_mm": stats(t["hurdle_cmd"]), "hurdle_servo_mm": stats(t["hurdle_servo"]),
            "n_cross_cmd_lt0": int(sum(1 for a in t["hurdle_cmd"] if a is not None and a < 0)),
            "n_cross_servo_lt0": int(sum(1 for a in t["hurdle_servo"] if a is not None and a < 0)),
            "contacts": {"n": t["c_n"], "servo_clear": t["c_servo_clear"], "cmd_to_clear": t["c_cmd_clear"],
                         "servo_mm": stats(t["c_servo_mm"])},
            "violations": {"collision_chunks": t["coll"], "new_vs_ref": t["new_ref"],
                           "fixed_vs_ref": t["fixed_ref"], "new_vs_base": t["new_base"],
                           "fixed_vs_base": t["fixed_base"],
                           "new_vs_ref_total": int(sum(t["new_ref"].values())),
                           "fixed_vs_ref_total": int(sum(t["fixed_ref"].values())),
                           "new_vs_base_total": int(sum(t["new_base"].values()))},
            "grasp_set": {"n": t["g_n"], "moved_gt_0p5deg_vs_ref": t["g_moved"],
                          "outward_vs_ref_mm": stats(t["g_out"]), "n_outward_gt_2mm_vs_ref": t["g_out_gt2"]},
            "at_apple": {"n": t["a_n"], "moved_gt_0p5deg_vs_ref": t["a_moved"],
                         "outward_vs_ref_mm": stats(t["a_out"]), "n_outward_gt_2mm_vs_ref": t["a_out_gt2"]},
            "closing": {"n": t["closing_n"], "moved_gt_0p5deg_vs_ref": t["closing_moved"],
                        "identical_to_ref": t["closing_identical"]},
            "joint_vs_ref_deg": stats(t["joint_vs_ref"]),
            "n_identical_to_ref": int(sum(1 for a in t["joint_vs_ref"] if a == 0.0)),
            "acc_boundary_p90_rad": stats(t["acc_bd"], 4).get("p90"),
            "reversal_share": round(float(np.mean(t["rev"])), 4) if t["rev"] else None,
            "n_vel_boundary_gt1": t["vel_bd_gt1"],
            "gate_open_share": round(float(np.mean(t["gate"])), 4),
            "status": t["status"], "qp_failures": t["qp_failures"],
            "offline_cpu_ms": stats(t["cpu_ms"], 1),
            "offline_cpu_ratio_vs_ref": stats(cpu / ref_cpu, 3),
        }
    res = {"definitions": __doc__, "ref": ref, "files": files, "n_runs": len(runs),
           "n_cross": n_cross, "phase_chunks": phase_n, "table": table}
    json.dump(res, open(out_path, "w"), indent=1)
    hdr = ("variant", "srv_p10", "srv_med", "srv<0", "cmd_p10", "cmd<0", "contact", "newR", "fixR",
           "newB", "grasp_mv", "apple_mv", "out>2", "out_max", "close_mv", "accbd90", "rev", "gate",
           "cpu_x")
    print(f"n_runs {len(runs)} chunks {T[ref]['n_chunks']} crossing {n_cross} phases {phase_n}")
    print(" ".join(f"{h:>9}" for h in hdr))
    for v in variants:
        t = table[v]
        print(" ".join(f"{str(x):>9}" for x in (
            v[:9], t["hurdle_servo_mm"].get("p10"), t["hurdle_servo_mm"].get("median"),
            t["n_cross_servo_lt0"], t["hurdle_cmd_mm"].get("p10"), t["n_cross_cmd_lt0"],
            f"{t['contacts']['servo_clear']}/{t['contacts']['n']}",
            t["violations"]["new_vs_ref_total"], t["violations"]["fixed_vs_ref_total"],
            t["violations"]["new_vs_base_total"],
            f"{t['grasp_set']['moved_gt_0p5deg_vs_ref']}/{t['grasp_set']['n']}",
            f"{t['at_apple']['moved_gt_0p5deg_vs_ref']}/{t['at_apple']['n']}",
            t["grasp_set"]["n_outward_gt_2mm_vs_ref"], t["grasp_set"]["outward_vs_ref_mm"].get("max"),
            f"{t['closing']['moved_gt_0p5deg_vs_ref']}/{t['closing']['n']}",
            t["acc_boundary_p90_rad"], t["reversal_share"], t["gate_open_share"],
            t["offline_cpu_ratio_vs_ref"].get("median"))))
    print("\nnew / fixed vs REF by phase:")
    for v in variants:
        vv = table[v]["violations"]
        print(f"  {v:22s} " + " ".join(f"{p}={vv['new_vs_ref'][p]}/{vv['fixed_vs_ref'][p]}" for p in PHASES))


if __name__ == "__main__":
    main()
