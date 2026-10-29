"""T43 XM part 2 aggregation — speed-proportional obstacle margin alternatives (raw/speed_{V7g,V8}_{H,B}.json -> raw/speed_table.json).
Format = T43 SV alternatives (sv_alt_agg.py): contacts seen + lead rows, new-HOLD upper bound per phase.

Variants (xm_speed.py doc): cur (robot obstacle 10 mm, held 0) | HM (robot 10, held obstacle 10) | k0.5 | k1.0
  (margin = 10 mm + k * 2 rows * per-sphere commanded displacement per row, robot obstacle rows and HM-obstacle held rows).
Cell value v(row) = min clearance (d - r - margin, mm) over the selected group|label cells of obstacle-class rows:
  hand_hurdle : finger + palm, label hurdle          held_hurdle : held, label hurdle
  all_obst    : finger + palm + robot_other + held, any label (support / target / own-remnant rows are not obstacle-class)
A chunk 'sees' a contact when min_k v(row k) < 0 (the plan violates the variant's margin).
Contacts: T43HX.verify.json numbers.contacts22 (CT replays; fingers after release -> return chunks, apple during carry -> carry and
  crate_entry chunks).  seen / lead_chunks / lead_rows as sv_alt_agg.lead (consecutive flagged chunks ending at the contact chunk).
New-HOLD upper bound of a phase = chunks with min_k v_all_obst < 0 whose recorded TO status is feasible and recorded clearance >= 0;
  by argmin cause group|label.  Phases: approach | closing | carry | crate_entry | return (xm_speed.py doc).
Displacement stats: per row max sphere displacement (mm / row) over hand spheres and held spheres, per phase.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, collections
import numpy as np
RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/XM/raw'))
HX = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43HX.verify.json'))
INF = 1e9
VARS = ("cur", "HM", "k0.5", "k1.0")
PHASES = ("approach", "closing", "carry", "crate_entry", "return")
SEL = {"hand_hurdle": (("finger", "palm"), ("hurdle",)), "held_hurdle": (("held",), ("hurdle",)),
       "all_obst": (("finger", "palm", "robot_other", "held"), None)}


def cellmin(red, sel):
    groups, labels = SEL[sel]; v = INF; arg = None
    for key, val in red.items():
        g, l = key.split("|")
        if g in groups and (labels is None or l in labels) and val[0] < v:
            v = val[0]; arg = key
    return v, arg


def chunk_min(c, var, sel):
    best = INF; arg = None
    for row in c["P"]:
        v, a = cellmin(row[0][var], sel)
        if v < best:
            best, arg = v, a
    return best, arg


def contact_chunk(chunks, t):
    for c in chunks:
        if c["t0"] <= t < c["t0"] + 8:
            return c
    return None


def lead(chunks_phase, t_c, flag):
    pre = [c for c in chunks_phase if c["t0"] <= t_c]
    if not pre:
        return {"seen": None, "reason": "no chunk of the phase before the contact"}
    cc = pre[-1]
    if not (cc["t0"] <= t_c < cc["t0"] + 8):
        return {"seen": None, "reason": "contact row not inside a chunk of the phase", "last_t0": cc["t0"]}
    if not flag(cc):
        prev = [c for c in pre[:-1] if flag(c)]
        return {"seen": False, "contact_chunk_t0": cc["t0"], "last_flagged_t0_before": prev[-1]["t0"] if prev else None}
    run = [cc]
    for c in reversed(pre[:-1]):
        if flag(c) and run[-1]["t0"] - c["t0"] == 8:
            run.append(c)
        else:
            break
    return {"seen": True, "contact_chunk_t0": cc["t0"], "lead_chunks": len(run), "lead_rows": int(t_c - run[-1]["t0"]), "first_flag_t0": run[-1]["t0"]}


def main():
    R = {}
    for arm in ("V8", "V7g"):
        R[arm] = {}
        for st in ("H", "B"):
            p = RAW / f"speed_{arm}_{st}.json"
            R[arm].update(json.load(open(p))["runs"])
    contacts = json.load(open(HX))["numbers"]["contacts22"]
    out = {"definitions": __doc__, "contacts_source": str(HX), "n_contacts": len(contacts), "errors": {f"{a}:{k}": r["error"] for a in R for k, r in R[a].items() if "error" in r}}
    per = []
    for ct in contacts:
        arm, key, grp, t_c = ct["arm"], ct["key"], ct["group"], int(ct["t"])
        r = R[arm].get(key)
        if r is None or "chunks" not in r:
            per.append({"arm": arm, "key": key, "group": grp, "t": t_c, "error": "run not reconstructed"}); continue
        ch = r["chunks"]
        e = {"arm": arm, "key": key, "group": grp, "phase_ct": ct["phase_ct"], "t": t_c, "k": ct.get("k")}
        if grp == "fingers":
            ph = [c for c in ch if c["phase"] == "return"]; sel = "hand_hurdle"
        else:
            ph = [c for c in ch if c["phase"] in ("carry", "crate_entry")]; sel = "held_hurdle"
        cc = contact_chunk(ph, t_c)
        if cc is not None:
            e["contact_chunk"] = {"t0": cc["t0"], "phase": cc["phase"], **{f"{v}_mm": chunk_min(cc, v, sel)[0] for v in VARS},
                                  "disp_hand_max_mm": max(row[1]["disp_hand_max_mm"] or 0 for row in cc["P"]),
                                  "disp_held_max_mm": max((row[1]["disp_held_max_mm"] or 0) for row in cc["P"]),
                                  "recorded_clearance_mm": cc["clearance_mm"], "to_status": cc["to_status"]}
        for v in VARS:
            e[v] = lead(ph, t_c, lambda c, v=v: chunk_min(c, v, sel)[0] < 0.0)
        per.append(e)
    out["per_contact"] = per

    def tally(group, v, arms):
        xs = [e for e in per if e["group"] == group and e["arm"] in arms and v in e]
        seen = [e for e in xs if e[v].get("seen")]
        return {"n": len(xs), "seen": len(seen), "not_seen": sum(1 for e in xs if e[v].get("seen") is False),
                "undetermined": sum(1 for e in xs if e[v].get("seen") is None),
                "lead_rows": sorted(e[v]["lead_rows"] for e in seen), "lead_chunks": sorted(e[v]["lead_chunks"] for e in seen),
                "lead_rows_median": float(np.median([e[v]["lead_rows"] for e in seen])) if seen else None}
    out["contact_tables"] = {lab: {"return_fingers": {v: tally("fingers", v, arms) for v in VARS},
                                   "carry_apple": {v: tally("apple", v, arms) for v in VARS}}
                             for lab, arms in (("V8", ("V8",)), ("V7g", ("V7g",)), ("V8+V7g", ("V8", "V7g")))}

    def hold_ub(phase, v, arms, stage=None):
        n = 0; nf = 0; nn = 0; lab = collections.Counter(); runs = set(); rows_below = 0
        for arm in arms:
            for key, r in R[arm].items():
                if "chunks" not in r or (stage and f"/{stage}/" not in key):
                    continue
                for c in r["chunks"]:
                    if c["phase"] != phase:
                        continue
                    n += 1
                    val, arg = chunk_min(c, v, "all_obst")
                    if val < 0:
                        nf += 1
                        if c["to_status"] == "feasible" and (c["clearance_mm"] is None or c["clearance_mm"] >= 0.0):
                            nn += 1; lab[arg] += 1; runs.add(f"{arm}:{key}")
                            rows_below += sum(1 for row in c["P"] if cellmin(row[0][v], "all_obst")[0] < 0)
        return {"n_chunks_phase": n, "n_chunks_below": nf, "new_hold_upper_bound_chunks": nn, "rows_below_in_those": rows_below,
                "by_cause": dict(lab), "n_runs": len(runs)}
    ub = {}
    for lab_, arms in (("V8", ("V8",)), ("V7g", ("V7g",)), ("V8+V7g", ("V8", "V7g"))):
        ub[lab_] = {ph: {v: hold_ub(ph, v, arms) for v in VARS} for ph in PHASES}
        ub[lab_ + "_H"] = {ph: {v: hold_ub(ph, v, arms, "H") for v in VARS} for ph in PHASES}
    out["hold_upper_bounds"] = ub
    disp = {}
    for ph in PHASES:
        hd = [row[1]["disp_hand_max_mm"] for a in R for r in R[a].values() for c in r.get("chunks", []) if c["phase"] == ph for row in c["P"] if row[1]["disp_hand_max_mm"] is not None]
        hl = [row[1]["disp_held_max_mm"] for a in R for r in R[a].values() for c in r.get("chunks", []) if c["phase"] == ph for row in c["P"] if row[1]["disp_held_max_mm"] is not None]
        disp[ph] = {"hand_disp_mm_per_row": {"n": len(hd), "median": float(np.median(hd)) if hd else None, "p90": float(np.percentile(hd, 90)) if hd else None, "max": float(np.max(hd)) if hd else None},
                    "held_disp_mm_per_row": {"n": len(hl), "median": float(np.median(hl)) if hl else None, "p90": float(np.percentile(hl, 90)) if hl else None, "max": float(np.max(hl)) if hl else None}}
    out["displacement_by_phase"] = disp
    held_cls = collections.Counter()
    for a in R:
        for r in R[a].values():
            for c in r.get("chunks", []):
                for row in c["P"]:
                    for k_, v_ in (row[1].get("n_held_cls") or {}).items():
                        held_cls[f"{c['phase']}:{k_}"] += v_
    out["held_row_classes"] = dict(held_cls)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
    t = out["contact_tables"]["V8+V7g"]
    for g in ("return_fingers", "carry_apple"):
        print(g, {v: (t[g][v]["seen"], t[g][v]["n"], t[g][v]["lead_rows_median"], t[g][v]["lead_rows"]) for v in VARS})
    for ph in PHASES:
        print(ph, {v: (ub["V8+V7g"][ph][v]["new_hold_upper_bound_chunks"], ub["V8+V7g"][ph][v]["n_chunks_phase"]) for v in VARS})
    print(json.dumps(disp))
    print(out["errors"])


if __name__ == "__main__":
    main()
