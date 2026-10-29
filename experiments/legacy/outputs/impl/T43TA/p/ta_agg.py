"""T43 TA — aggregate the offline re-solves (raw/*.json from AG3S.trajopt.experiments.ta_offline).

usage: ta_agg.py OUT_JSON RAW_GLOB [RAW_GLOB ...]

Chunk sets (fixed across variants, chosen on the *base* re-solve):
  crossing = carry/crate/return chunks whose base commanded rows come within 50 mm (geometric d − r,
             obstacle class, nearest surface outside the destination box) of a hurdle-ish obstacle
             in any group (palm / fingers / held).
  closing  = chunks in grasp state `closing`, the attach chunk, and the 2 chunks before the first
             closing chunk (the V9 lesson: these must not change).
Metrics per variant (medians over chunks unless said):
  hurdle_cmd / hurdle_servo: per chunk min over rows 0-7 and groups of the hurdle clearance (mm),
             commanded rows / servo-predicted states (4 sub-steps per row). p10, median, n < 0.
  dev_palm:  per chunk max over rows 0-7 of |palm(variant) − palm(policy)| (mm).
  acc_bd:    command second difference across the chunk start (u−2, u−1, Q0, Q1), max joint (rad/row²).
  reversal:  share of chunks whose palm turns > 90° at the chunk start (displacements > 2 mm).
  new_viol:  chunks with collision rows (classify_violations over rows 0-7) in the variant and none
             in base, per phase.
  contacts:  HX contact rows (t, k, group) of these runs: servo-predicted hurdle clearance of that
             group at row k ≥ 0 (= predicted clear) and commanded TO clearance ≥ 0.
  cpu_ratio: per chunk median(cpu_ms variant) / median(cpu_ms base), then median / p90 over chunks.
"""
import glob
import json
import sys

import numpy as np


def q(a, f):
    a = [x for x in a if x is not None and np.isfinite(x)]
    return None if not a else float(f(np.asarray(a, float)))


def stats(a):
    a = [x for x in a if x is not None and np.isfinite(x)]
    if not a:
        return {"n": 0}
    v = np.asarray(a, float)
    return {"n": int(v.size), "p10": round(float(np.quantile(v, .1)), 3),
            "median": round(float(np.median(v)), 3), "p90": round(float(np.quantile(v, .9)), 3),
            "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}


def cmin(block, key="hurdle_mm", rows=None):
    out = []
    for g in ("palm", "fingers", "held"):
        b = block.get(g)
        if not b:
            continue
        vals = b[key] if rows is None else [b[key][r] for r in rows if r < len(b[key])]
        out += [x for x in vals if x is not None]
    return min(out) if out else None


def main():
    out_path = sys.argv[1]
    files = sorted({f for g in sys.argv[2:] for f in glob.glob(g)
                    if not f.split("/")[-1].startswith("supp_")})
    runs = []
    for f in files:
        r = json.load(open(f))
        sp = f.rsplit("/", 1)
        supp = f"{sp[0]}/supp_{sp[1]}" if len(sp) == 2 else f"supp_{f}"
        try:
            s2 = json.load(open(supp))
        except FileNotFoundError:
            s2 = None
        if s2 is not None:
            extra = {c["seq"]: c for c in s2["chunks"] if c.get("variants")}
            for v, cfg in s2["variants"].items():
                r["variants"].setdefault(v, cfg)
            for c in r["chunks"]:
                e = extra.get(c["seq"])
                if c.get("variants") and e:
                    for v, x in e["variants"].items():
                        if v not in c["variants"]:
                            # vs_base was measured against the supplement's own base re-solve —
                            # the same deterministic solve as the main file's base
                            c["variants"][v] = x
        runs.append(r)
    variants = [v for v in runs[0]["variants"]
                if all(v in r["variants"] for r in runs)]
    per = {v: {k: [] for k in ("hurdle_cmd", "hurdle_servo", "hurdle_cmd_held", "hurdle_servo_held",
                               "hurdle_cmd_fing", "hurdle_servo_fing", "dev_palm_all",
                               "dev_palm_cross", "acc_bd_all", "acc_bd_cross", "acc_in_all",
                               "rev_all", "rev_cross", "cpu_ratio", "cpu_ms", "tail_viol_cross",
                               "dev_profile_cross", "turn_bd_all", "vel_bd_all",
                               "vel_in_all")} for v in variants}
    phases = ("approach", "closing", "carry", "crate", "return")
    viol = {v: {p: {"n_chunks": 0, "collision": 0, "new_vs_base": 0, "fixed_vs_base": 0}
                for p in phases} for v in variants}
    closing = {v: {"vs_base_joint_deg": [], "dev_palm_minus_base_mm": [], "n_changed_gt_0p5deg": 0,
                   "new_collision": 0, "n": 0, "untouched_n": 0, "untouched_changed_gt_0p5deg": 0,
                   "untouched_palm_mm": [], "last_row_palm_dev_mm": [],
                   "last_row_palm_dev_minus_base_mm": []} for v in variants}
    contacts = {v: {"n": 0, "servo_clear": 0, "cmd_to_clear": 0, "servo_mm": [], "cmd_to_mm": []}
                for v in variants}
    fid = []
    status = {v: {} for v in variants}
    qpfail = {v: 0 for v in variants}
    n_chunks = 0
    for r in runs:
        chunks = [c for c in r["chunks"] if "variants" in c and c["variants"]]
        first_closing = next((i for i, c in enumerate(chunks) if c["phase"] == "closing"), None)
        pre = set()
        if first_closing is not None:
            pre = {chunks[j]["seq"] for j in range(max(0, first_closing - 2), first_closing)}
        for c in chunks:
            n_chunks += 1
            if c.get("fidelity"):
                fid.append(c["fidelity"]["abs_diff_mm"])
            V = c["variants"]
            base = V["base"]
            hb = cmin(base["clear_cmd"])
            cross = c["phase"] in ("carry", "crate", "return") and hb is not None and hb < 50.0
            is_closing = c["phase"] == "closing" or c.get("attach") or c["seq"] in pre
            base_cpu = np.median(base["cpu_ms"])
            for v in variants:
                x = V[v]
                status[v][x["status"]] = status[v].get(x["status"], 0) + 1
                qpfail[v] += int(x.get("qp_failures") or 0) > 0
                dpm = max(x["dev_palm_mm"])
                per[v]["dev_palm_all"].append(dpm)
                bd = x["boundary"]
                per[v]["acc_in_all"].append(bd.get("acc_inner_max_rad"))
                if "acc_boundary_max_rad" in bd:
                    per[v]["acc_bd_all"].append(bd["acc_boundary_max_rad"])
                    per[v]["rev_all"].append(1.0 if bd["palm_reversal"] else 0.0)
                    per[v]["vel_bd_all"].append(bd.get("vel_boundary_ratio"))
                    t = [a for a in bd["palm_turn_deg"] if a is not None]
                    if t:
                        per[v]["turn_bd_all"].append(max(t))
                per[v]["vel_in_all"].append(bd.get("vel_inner_ratio"))
                per[v]["cpu_ratio"].append(np.median(x["cpu_ms"]) / base_cpu)
                per[v]["cpu_ms"].append(np.median(x["cpu_ms"]))
                ph = c["phase"]
                viol[v][ph]["n_chunks"] += 1
                coll = x["n_collision_rows"] > 0
                viol[v][ph]["collision"] += coll
                viol[v][ph]["new_vs_base"] += coll and base["n_collision_rows"] == 0
                viol[v][ph]["fixed_vs_base"] += (not coll) and base["n_collision_rows"] > 0
                if cross:
                    per[v]["hurdle_cmd"].append(cmin(x["clear_cmd"]))
                    per[v]["hurdle_servo"].append(cmin(x["clear_servo"]))
                    for g, key in (("held", "held"), ("fingers", "fing")):
                        bc, bs = x["clear_cmd"].get(g), x["clear_servo"].get(g)
                        per[v][f"hurdle_cmd_{key}"].append(
                            None if not bc else q(bc["hurdle_mm"], np.min))
                        per[v][f"hurdle_servo_{key}"].append(
                            None if not bs else q(bs["hurdle_mm"], np.min))
                    per[v]["dev_palm_cross"].append(dpm)
                    per[v]["dev_profile_cross"].append(x["dev_palm_mm"])
                    if "acc_boundary_max_rad" in bd:
                        per[v]["acc_bd_cross"].append(bd["acc_boundary_max_rad"])
                        per[v]["rev_cross"].append(1.0 if bd["palm_reversal"] else 0.0)
                    la = x.get("lookahead")
                    if la:
                        per[v]["tail_viol_cross"].append(1000.0 * (la.get("tail_violation_m") or 0.0))
                if is_closing:
                    cl = closing[v]
                    cl["n"] += 1
                    cl["vs_base_joint_deg"].append(x["vs_base_joint_deg_max"])
                    cl["dev_palm_minus_base_mm"].append(
                        float(np.max(np.abs(np.asarray(x["dev_palm_mm"])
                                            - np.asarray(base["dev_palm_mm"])))))
                    cl["n_changed_gt_0p5deg"] += x["vs_base_joint_deg_max"] > 0.5
                    cl["new_collision"] += coll and base["n_collision_rows"] == 0
                    cl["last_row_palm_dev_mm"].append(x["dev_palm_mm"][-1])
                    cl["last_row_palm_dev_minus_base_mm"].append(
                        x["dev_palm_mm"][-1] - base["dev_palm_mm"][-1])
                    if base.get("returned") == "initial" and base["dev_joint_deg_max"] < 1e-9:
                        # the base TO left the policy chunk as it was
                        cl["untouched_n"] += 1
                        cl["untouched_changed_gt_0p5deg"] += x["vs_base_joint_deg_max"] > 0.5
                        cl["untouched_palm_mm"].append(max(x["dev_palm_mm"]))
            for cr in c.get("contact_rows", []):
                k, g = cr["k"], cr["group"]
                gk = "held" if g == "apple" else "fingers"
                for v in variants:
                    x = V[v]
                    s_rows = list(range(4 * k, 4 * k + 4))
                    sv = None
                    if x["clear_servo"].get(gk):
                        vals = [x["clear_servo"][gk]["hurdle_mm"][i] for i in s_rows]
                        vals = [a for a in vals if a is not None]
                        sv = min(vals) if vals else None
                    to = x["clear_cmd"][gk]["to_mm"][k] if x["clear_cmd"].get(gk) else None
                    cc = contacts[v]
                    cc["n"] += 1
                    cc["servo_clear"] += sv is None or sv >= 0.0
                    cc["cmd_to_clear"] += to is not None and to >= 0.0
                    cc["servo_mm"].append(sv)
                    cc["cmd_to_mm"].append(to)
    table = {}
    for v in variants:
        P = per[v]
        prof = np.asarray(P["dev_profile_cross"], float) if P["dev_profile_cross"] else None
        table[v] = {
            "hurdle_cmd_mm": stats(P["hurdle_cmd"]),
            "hurdle_servo_mm": stats(P["hurdle_servo"]),
            "hurdle_cmd_held_mm": stats(P["hurdle_cmd_held"]),
            "hurdle_servo_held_mm": stats(P["hurdle_servo_held"]),
            "hurdle_cmd_fingers_mm": stats(P["hurdle_cmd_fing"]),
            "hurdle_servo_fingers_mm": stats(P["hurdle_servo_fing"]),
            "n_cross_servo_lt0": int(sum(1 for a in P["hurdle_servo"] if a is not None and a < 0)),
            "n_cross_cmd_lt0": int(sum(1 for a in P["hurdle_cmd"] if a is not None and a < 0)),
            "dev_palm_all_mm": stats(P["dev_palm_all"]),
            "dev_palm_cross_mm": stats(P["dev_palm_cross"]),
            "dev_profile_cross_median_mm": None if prof is None else
            [round(float(x), 2) for x in np.median(prof, axis=0)],
            "acc_boundary_all_rad": stats(P["acc_bd_all"]),
            "acc_boundary_cross_rad": stats(P["acc_bd_cross"]),
            "acc_inner_all_rad": stats(P["acc_in_all"]),
            "palm_turn_boundary_all_deg": stats(P["turn_bd_all"]),
            "vel_boundary_ratio": stats(P["vel_bd_all"]),
            "n_vel_boundary_gt1": int(sum(1 for a in P["vel_bd_all"] if a is not None and a > 1.0 + 1e-6)),
            "vel_inner_ratio": stats(P["vel_in_all"]),
            "n_vel_inner_gt1": int(sum(1 for a in P["vel_in_all"] if a is not None and a > 1.0 + 1e-6)),
            "reversal_share_all": q(P["rev_all"], np.mean),
            "reversal_share_cross": q(P["rev_cross"], np.mean),
            "tail_violation_cross_mm": stats(P["tail_viol_cross"]),
            "cpu_ratio": stats(P["cpu_ratio"]),
            "cpu_ms": stats(P["cpu_ms"]),
            "status": status[v], "chunks_with_qp_failure": qpfail[v],
            "violations_by_phase": viol[v],
            "closing": {"n": closing[v]["n"],
                        "vs_base_joint_deg": stats(closing[v]["vs_base_joint_deg"]),
                        "dev_palm_minus_base_mm": stats(closing[v]["dev_palm_minus_base_mm"]),
                        "n_changed_gt_0p5deg": closing[v]["n_changed_gt_0p5deg"],
                        "new_collision": closing[v]["new_collision"],
                        "last_row_palm_dev_mm": stats(closing[v]["last_row_palm_dev_mm"]),
                        "last_row_palm_dev_minus_base_mm": stats(
                            closing[v]["last_row_palm_dev_minus_base_mm"]),
                        "untouched_n": closing[v]["untouched_n"],
                        "untouched_changed_gt_0p5deg": closing[v]["untouched_changed_gt_0p5deg"],
                        "untouched_palm_dev_mm": stats(closing[v]["untouched_palm_mm"])},
            "contacts": {k: (stats(val) if isinstance(val, list) else val)
                         for k, val in contacts[v].items()},
        }
    res = {"definitions": __doc__, "files": files, "n_runs": len(runs), "n_chunks": n_chunks,
           "n_cross": len(per["base"]["hurdle_cmd"]),
           "fidelity_abs_diff_mm": stats(fid),
           "fidelity_n_gt_1mm": int(sum(1 for a in fid if a > 1.0)),
           "variants": runs[0]["variants"], "table": table}
    json.dump(res, open(out_path, "w"), indent=1)
    hdr = ("var", "cmd_p10", "cmd_med", "srv_p10", "srv_med", "srv<0", "dev_med", "accbd_p90",
           "velbd>1", "rev", "newviol", "close_chg", "contact_srv", "cpu_x")
    print(" ".join(f"{h:>10}" for h in hdr))
    for v in variants:
        t = table[v]
        nv = sum(t["violations_by_phase"][p]["new_vs_base"] for p in phases)
        print(" ".join(f"{str(x):>10}" for x in (
            v, t["hurdle_cmd_mm"].get("p10"), t["hurdle_cmd_mm"].get("median"),
            t["hurdle_servo_mm"].get("p10"), t["hurdle_servo_mm"].get("median"),
            t["n_cross_servo_lt0"], t["dev_palm_all_mm"].get("median"),
            t["acc_boundary_all_rad"].get("p90"), t["n_vel_boundary_gt1"],
            round(t["reversal_share_all"] or 0, 3), nv,
            t["closing"]["n_changed_gt_0p5deg"],
            f"{t['contacts']['servo_clear']}/{t['contacts']['n']}",
            t["cpu_ratio"].get("median"))))


if __name__ == "__main__":
    main()
