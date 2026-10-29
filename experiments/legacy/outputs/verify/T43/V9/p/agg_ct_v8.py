"""T43 V8 (copy of T43/CT/p/agg_ct.py, arms V7g + V8) — contact-inclusive re-scoring (verifier, CPU only; aggregation of existing runs + physics replays CT/replay, no new robot runs).
usage: agg_ct.py OUT_JSON      (re-runnable: reads only runs with .done; run again when V7g B finishes — stage B has no hurdle, so it adds nothing here)

Inputs: H-stage runs of E0 V1 V3 V4 V5 (T43/W/runs), V5S V5B (W/runs_blind), V5D (Z3V/runs), V7g (V7g/runs).
  old success / grasp_t / place_t : analysis.json of each arm (V7g: CT/raw/analysis_v7g.json = V7g/p/analyze_v7g.py rerun, because V7g/analysis/analysis.json was stale)
  Z4 release rule                 : T43/CT/p/z4_base.py = score() of T43/V7g/p/z4_v7g.py (= Z/p/z4_release.py) unchanged, on CT/replay (replay_ct.py = replay_y.py + hurdle contacts)
  contacts                        : CT/replay/*.json obstacle_contact_rows = per control row, per robot / apple geom, MuJoCo contacts (d.contact) with any collision geom of the
                                    obstacle manager (hurdle: bar + 2 legs); kind: apple | hand (palm EE_BODY + fingers of the active arm) | robot_other; substeps (33 / row),
                                    min contact dist (m; penetration = max(0, -dist)), max normal force (N), substeps with dist > 0 (margin contacts).
Phase of a contact row t:  before_grasp: t < grasp_t (grasp false: every row);  carry: grasp_t <= t <= release_t (release_t none: to the run's end);  after_release: t > release_t.
Rules: pre_release_contact = >= 1 robot or apple contact row with t <= release_t (release_t none: any row);  post_release_contact = >= 1 such row with t > release_t.
  obstacle_stop = '[SAFETY] obstacle stop' in the client log (check.json obstacle_stop_in_log).
  success_ct = success (old) and success_rel (Z4: release + separation kept + in crate at end) and not pre_release_contact.
McNemar exact two-sided (min(b, c) binomial, p = min(1, 2 P)) on (episode, seed) pairs."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, re, math, collections
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/V8/p'))
import z4_base_v8 as Z  # noqa: E402

T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); CT = T / "V8"   # CT = V8 dir: replays V8/replay, V7g contacts from CT/replay (see run_contacts)
CTOLD = T / "CT"
ANA = [(T / "V7g/analysis/analysis.json", ("V7g",)), (T / "V8/analysis/analysis.json", ("V8",))]
ROOT = {"V7g": T/"V7g/runs/V7g", "V8": T/"V8/runs/V8"}
ARMS = ["V7g", "V8"]


def phase(t, grasp, grasp_t, rel_t):
    if not grasp or grasp_t is None or t < grasp_t:
        return "before_grasp"
    if rel_t is None or t <= rel_t:
        return "carry"
    return "after_release"


def run_contacts(o, key):
    arm, st, name = key.split("/")
    j = json.load(open((CTOLD if arm == "V7g" else CT) / "replay" / f"{arm}_{st}_{name}.json"))
    rel_t, gt, grasp = o["release_t"], o["grasp_t"], o["grasp"]
    rec = {}                       # (body, kind, phase) -> dict
    for r in j["obstacle_contact_rows"]:
        for g, v in r["geoms"].items():
            gname, body, kind, sub, mind, maxf, npos = v
            ph = phase(r["t"], grasp, gt, rel_t)
            e = rec.setdefault((body, kind, ph), {"body": body, "kind": kind, "phase": ph, "geoms": set(), "first_t": r["t"], "last_t": r["t"], "n_rows": 0, "substeps": 0,
                                                  "max_pen_mm": 0.0, "min_dist_mm": 1e9, "max_force_N": 0.0, "margin_substeps": 0})
            e["geoms"].add(gname); e["first_t"] = min(e["first_t"], r["t"]); e["last_t"] = max(e["last_t"], r["t"]); e["n_rows"] += 1; e["substeps"] += sub
            e["max_pen_mm"] = max(e["max_pen_mm"], max(0.0, -1e3 * mind)); e["min_dist_mm"] = min(e["min_dist_mm"], 1e3 * mind)
            e["max_force_N"] = max(e["max_force_N"], maxf); e["margin_substeps"] += npos
    for e in rec.values():
        e["geoms"] = sorted(e["geoms"])
    return sorted(rec.values(), key=lambda e: (e["first_t"], e["body"])), j


def mcnemar(b, c):
    return Z.mcnemar(b, c)


def pair(R, A, B, f):
    ka = {(o["episode"], o["seed"]): o for o in R.values() if o["arm"] == A}; kb = {(o["episode"], o["seed"]): o for o in R.values() if o["arm"] == B}
    ks = sorted(set(ka) & set(kb)); both = sum(ka[k][f] and kb[k][f] for k in ks); ao = sum(ka[k][f] and not kb[k][f] for k in ks); bo = sum(kb[k][f] and not ka[k][f] for k in ks)
    return {"n_pairs": len(ks), "both": both, "A_only": ao, "B_only": bo, "neither": len(ks) - both - ao - bo, "A_total": both + ao, "B_total": both + bo,
            "mcnemar_exact_p": mcnemar(ao, bo), "A_only_keys": [f"{e}_s{s}" for e, s in ks if ka[(e, s)][f] and not kb[(e, s)][f]],
            "B_only_keys": [f"{e}_s{s}" for e, s in ks if kb[(e, s)][f] and not ka[(e, s)][f]]}


if __name__ == "__main__":
    ana = {}
    for f, arms in ANA:
        for k, v in json.load(open(f))["runs"].items():
            if k.split("/")[0] in arms and k.split("/")[1] == "H":
                ana[k] = v
    R, xc = {}, {"n": 0, "pairs_mismatch": [], "robot_flag_mismatch": [], "payload_flag_mismatch": [], "rows_vs_steps": [], "qpos_mismatch": [], "stop_vs_rows": []}
    for arm in ARMS:
        for dn in sorted((ROOT[arm] / "H").glob(f"{arm}_ep*_s*.done")):
            name = dn.stem; key = f"{arm}/H/{name}"
            if key not in ana:
                print("no analysis for", key); continue
            o = Z.score(key, ana[key]); cons, j = run_contacts(o, key)
            chk = json.load(open(ROOT[arm] / "H" / f"{name}.check.json"))["checks"]; sf = chk["npz_obstacle_safety"]
            pre = [c for c in cons if c["phase"] != "after_release"]; post = [c for c in cons if c["phase"] == "after_release"]
            stop = bool(chk["obstacle_stop_in_log"])
            o.update({"contacts": cons, "n_control_rows": j["n_rows"], "pre_release_contact": bool(pre), "post_release_contact": bool(post), "obstacle_stop": stop,
                      "pre_release_kinds": sorted({c["kind"] for c in pre}), "post_release_kinds": sorted({c["kind"] for c in post}),
                      "success_ct": bool(o["success"] and o["success_rel"] and not pre), "check_json": {k: sf[k] for k in ("robot_collision", "payload_collision", "contact_steps", "contact_pairs")},
                      "replay_json": str((CTOLD if arm == "V7g" else CT) / "replay" / f"{arm}_H_{name}.json")})
            o["flip_ct"] = None if o["success_rel"] == o["success_ct"] else "S_rel->F"
            R[key] = o
            # cross-checks vs check.json (client-side contact record)
            xc["n"] += 1
            mine = sorted({g for c in cons for g in c["geoms"]}); theirs = sorted(p[0] for p in sf["contact_pairs"])
            if mine != theirs:
                xc["pairs_mismatch"].append([key, mine, theirs])
            if sf["robot_collision"] != any(c["kind"] != "apple" for c in cons):
                xc["robot_flag_mismatch"].append(key)
            if sf["payload_collision"] != any(c["kind"] == "apple" for c in cons):
                xc["payload_flag_mismatch"].append(key)
            rows = sum(c["n_rows"] for c in cons); sub = sum(c["substeps"] for c in cons)
            xc["rows_vs_steps"].append([key, sf["contact_steps"], rows, sub])
            if j["qpos_mismatch_rows"]:
                xc["qpos_mismatch"].append(key)
            xc["stop_vs_rows"].append([key, stop, chk["n_control"], j["n_rows"]])
    arms = {}
    for a in ARMS:
        L = [o for o in R.values() if o["arm"] == a]
        if not L:
            continue
        arms[a] = {"n": len(L), "S_old": sum(o["success"] for o in L), "S_release": sum(o["success_rel"] for o in L), "S_contact": sum(o["success_ct"] for o in L),
                   "pre_release_contact_runs": sum(o["pre_release_contact"] for o in L), "post_release_contact_runs": sum(o["post_release_contact"] for o in L),
                   "obstacle_stop_runs": sum(o["obstacle_stop"] for o in L), "any_contact_runs": sum(bool(o["contacts"]) for o in L),
                   "pre_release_contact_by_kind": dict(collections.Counter(k for o in L for k in o["pre_release_kinds"])),
                   "flip_rel_to_ct": [o["key"] for o in L if o["flip_ct"]],
                   "post_release_contact_keys": [o["key"] for o in L if o["post_release_contact"]],
                   "success_rel_with_post_release_contact": sum(o["success_rel"] and o["post_release_contact"] for o in L),
                   "success_ct_runs": [o["key"] for o in L if o["success_ct"]]}
    PAIRS = (("V8", "V7g"),)
    pairs = {}
    for A, B in PAIRS:
        if any(o["arm"] == A for o in R.values()) and any(o["arm"] == B for o in R.values()):
            pairs[f"{A}_vs_{B}"] = {f: pair(R, A, B, f) for f in ("success", "success_rel", "success_ct")}
    # contact list by body x phase
    tab = collections.OrderedDict()
    for o in R.values():
        for c in o["contacts"]:
            e = tab.setdefault((c["body"], c["kind"], c["phase"]), {"body": c["body"], "kind": c["kind"], "phase": c["phase"], "n_runs": 0, "runs_by_arm": collections.Counter(), "first_t_min": 1e9,
                                                                      "first_t_values": [], "max_pen_mm": 0.0, "max_force_N": 0.0})
            e["n_runs"] += 1; e["runs_by_arm"][o["arm"]] += 1; e["first_t_min"] = min(e["first_t_min"], c["first_t"]); e["first_t_values"].append(c["first_t"])
            e["max_pen_mm"] = max(e["max_pen_mm"], c["max_pen_mm"]); e["max_force_N"] = max(e["max_force_N"], c["max_force_N"])
    bodies = []
    for e in tab.values():
        e["runs_by_arm"] = dict(e["runs_by_arm"]); e["first_t_values"] = sorted(e["first_t_values"]); bodies.append(e)
    bodies.sort(key=lambda e: ({"before_grasp": 0, "carry": 1, "after_release": 2}[e["phase"]], -e["n_runs"], e["body"]))
    res = {"definitions": __doc__, "arms": arms, "pairs": pairs, "contact_table": bodies, "crosscheck": xc,
           "n_runs": len(R), "v8_H_done": sum(o["arm"] == "V8" for o in R.values()), "runs": R}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
    for a, v in arms.items():
        print(a, {k: v[k] for k in ("n", "S_old", "S_release", "S_contact", "pre_release_contact_runs", "post_release_contact_runs", "obstacle_stop_runs", "any_contact_runs")})
    for p, v in pairs.items():
        print(p, {f: (x["both"], x["A_only"], x["B_only"], x["neither"], round(x["mcnemar_exact_p"], 4)) for f, x in v.items()})
    for e in bodies:
        print(e["phase"], e["body"], e["kind"], e["n_runs"], e["runs_by_arm"], "first_t", e["first_t_min"], "pen_mm", round(e["max_pen_mm"], 2))
    print({k: (v if k == "n" else (v if not isinstance(v, list) else len(v))) for k, v in xc.items()})
