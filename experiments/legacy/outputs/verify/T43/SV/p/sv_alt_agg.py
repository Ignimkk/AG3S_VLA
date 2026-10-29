"""T43 SV part 3 — offline alternatives on the V8 / V7g records (raw/rows_{V8,V7g}.json from sv_rows.py).
usage: sv_alt_agg.py OUT_JSON

Contacts: T43HX.verify.json numbers.contacts22 (CT replays; V7g 5 + V8 9 finger-hurdle after release, V7g 3 + V8 5 apple-hurdle carry;
  t = first contact control row, k = its row in the executing chunk).  Run end: the client's obstacle stop ends a run at the contact.
Phases: return = chunks from the first server grasp.state 'placed' chunk; carry / crate_entry = attached chunks (sv_rows doc).
Cell value v(state, row) = min of d - r (mm, no margin) over the selected (group, class, label) cells of that row.
  hand-hurdle  : groups finger + palm, class obstacle, label hurdle      (variant 'any': label hurdle or unexplained)
  robot-obst   : groups finger + palm + robot_other, class obstacle, any label
  held-hurdle  : group held, label hurdle
  held-obst    : group held, labels hurdle | crate | fruit_other (+ unexplained in the 'incl. unexplained' variant); label table = support and
                 label apple = the held apple's own surface are excluded (HM rule: support and own residual excluded)
(a) servo after PLACED: S rows (servo prediction of each plan row), current margin 10 mm on obstacle rows -> a chunk 'sees' a contact
    when min_k v_S(hand-hurdle) < 10 mm.  Path-row activation count: return chunks with min_k v_S(robot-obst) < 10 mm, and of those the
    chunks whose plan rows were clear (min_k v_P(robot-obst) >= 10 mm).
(b) return margin M in {20, 30, 40} on P rows (the plan): a chunk sees a contact when min_k v_P(hand-hurdle) < M.
    New-HOLD upper bound = return chunks with min_k v_P(robot-obst) < M whose recorded TO status was feasible and recorded clearance >= 0
    (rows whose recorded clearance falls below the new margin); by label of the argmin cell.
(c) carry margin M in {20, 30}: held rows margin M (held-obst), robot obstacle rows max(10, M): sees a carry contact when
    min_k v_P(held-hurdle) < M.  New-HOLD upper bound split carry / crate_entry, by cause (held / robot) and label.
    Carry analogue of (a): S rows, held-hurdle < 0 (current held margin 0) and < 10 mm (HM margin 10).
Seen / lead: chunks of the phase with t0 <= t_contact, in order; seen = the contact chunk (the chunk executing row t) is flagged;
  lead_chunks = length of the run of consecutive flagged chunks ending at the contact chunk; lead_rows = t_contact - t0 of the first chunk of
  that run (rows between the first plan that flagged it and the contact).  Also: last flagged chunk before the contact if the contact chunk
  is not flagged.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, collections
import numpy as np
SV = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/SV/raw'))
HX = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43HX.verify.json'))
INF = 1e9


def cellmin(red, groups, classes, labels):
    v = INF; arg = None
    for key, val in red.items():
        g, c, l = key.split("|")
        if g in groups and c in classes and l in labels and val < v:
            v = val; arg = l
    return v, arg


SEL = {"hand_hurdle": (("finger", "palm"), ("obstacle",), ("hurdle",)),
       "hand_hurdle_any": (("finger", "palm"), ("obstacle",), ("hurdle", "unexplained")),
       "robot_obst": (("finger", "palm", "robot_other"), ("obstacle",), ("hurdle", "crate", "table", "apple", "fruit_other", "unexplained")),
       "held_hurdle": (("held",), ("held",), ("hurdle",)),
       "held_obst": (("held",), ("held",), ("hurdle", "crate", "fruit_other")),
       "held_obst_unexpl": (("held",), ("held",), ("hurdle", "crate", "fruit_other", "unexplained"))}


def chunk_min(c, state, sel):
    best = INF; lab = None
    for row in c[state]:
        if row is None:
            continue
        v, l = cellmin(row[0], *SEL[sel])
        if v < best:
            best, lab = v, l
    return best, lab


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
    R = {a: json.load(open(SV / f"rows_{a}.json"))["runs"] for a in ("V8", "V7g")}
    contacts = json.load(open(HX))["numbers"]["contacts22"]
    out = {"definitions": __doc__, "contacts_source": str(HX), "n_contacts": len(contacts)}
    # ---------------- per contact ----------------
    per = []
    for ct in contacts:
        arm, key, grp, t_c = ct["arm"], ct["key"], ct["group"], int(ct["t"])
        r = R[arm].get(key)
        if r is None or "chunks" not in r:
            per.append({"arm": arm, "key": key, "group": grp, "t": t_c, "error": "run not reconstructed"}); continue
        ch = r["chunks"]
        e = {"arm": arm, "key": key, "group": grp, "phase_ct": ct["phase_ct"], "t": t_c, "k": ct.get("k"), "placed_seq": r.get("placed_seq")}
        if grp == "fingers":
            ph = [c for c in ch if c["phase"] == "return"]
            cc = contact_chunk(ph, t_c)
            if cc is not None:
                e["contact_chunk"] = {"t0": cc["t0"], "P_hand_hurdle_mm": chunk_min(cc, "P", "hand_hurdle")[0],
                                      "S_hand_hurdle_mm": chunk_min(cc, "S", "hand_hurdle")[0], "M_hand_hurdle_mm": chunk_min(cc, "M", "hand_hurdle")[0],
                                      "P_hand_hurdle_any_mm": chunk_min(cc, "P", "hand_hurdle_any")[0],
                                      "true_hand_hurdle_P_mm": min([row[1].get("hand_true_hurdle_mm") for row in cc["P"] if row and row[1].get("hand_true_hurdle_mm") is not None], default=None),
                                      "recorded_clearance_mm": cc["clearance_mm"], "to_status": cc["to_status"]}
            e["a_servo_10"] = lead(ph, t_c, lambda c: chunk_min(c, "S", "hand_hurdle")[0] < 10.0)
            e["a_servo_10_any"] = lead(ph, t_c, lambda c: chunk_min(c, "S", "hand_hurdle_any")[0] < 10.0)
            for M in (10, 20, 30, 40):
                e[f"b_margin_{M}"] = lead(ph, t_c, lambda c, M=M: chunk_min(c, "P", "hand_hurdle")[0] < M)
                e[f"b_margin_{M}_any"] = lead(ph, t_c, lambda c, M=M: chunk_min(c, "P", "hand_hurdle_any")[0] < M)
        else:
            ph = [c for c in ch if c["phase"] in ("carry", "crate_entry")]
            cc = contact_chunk(ph, t_c)
            if cc is not None:
                e["contact_chunk"] = {"t0": cc["t0"], "phase": cc["phase"], "P_held_hurdle_mm": chunk_min(cc, "P", "held_hurdle")[0],
                                      "S_held_hurdle_mm": chunk_min(cc, "S", "held_hurdle")[0], "M_held_hurdle_mm": chunk_min(cc, "M", "held_hurdle")[0],
                                      "true_held_hurdle_P_mm": min([row[1].get("held_true_hurdle_mm") for row in cc["P"] if row and row[1].get("held_true_hurdle_mm") is not None], default=None),
                                      "recorded_clearance_mm": cc["clearance_mm"], "to_status": cc["to_status"]}
            e["a_servo_0"] = lead(ph, t_c, lambda c: chunk_min(c, "S", "held_hurdle")[0] < 0.0)
            e["a_servo_10"] = lead(ph, t_c, lambda c: chunk_min(c, "S", "held_hurdle")[0] < 10.0)
            for M in (0, 10, 20, 30):
                e[f"c_margin_{M}"] = lead(ph, t_c, lambda c, M=M: chunk_min(c, "P", "held_hurdle")[0] < M)
        per.append(e)
    out["per_contact"] = per

    def tally(sel_group, keyname, arms=("V8", "V7g")):
        xs = [e for e in per if e["group"] == sel_group and e["arm"] in arms and keyname in e]
        seen = [e for e in xs if e[keyname].get("seen")]
        return {"n": len(xs), "seen": len(seen), "not_seen": sum(1 for e in xs if e[keyname].get("seen") is False),
                "undetermined": sum(1 for e in xs if e[keyname].get("seen") is None),
                "lead_rows": sorted(e[keyname]["lead_rows"] for e in seen), "lead_chunks": sorted(e[keyname]["lead_chunks"] for e in seen),
                "lead_rows_median": float(np.median([e[keyname]["lead_rows"] for e in seen])) if seen else None}
    tab = {}
    for arms_lab, arms in (("V8", ("V8",)), ("V7g", ("V7g",)), ("V8+V7g", ("V8", "V7g"))):
        t_ = {"return_fingers": {"a_servo_10": tally("fingers", "a_servo_10", arms), "a_servo_10_any": tally("fingers", "a_servo_10_any", arms)},
              "carry_apple": {"a_servo_0": tally("apple", "a_servo_0", arms), "a_servo_10": tally("apple", "a_servo_10", arms)}}
        for M in (10, 20, 30, 40):
            t_["return_fingers"][f"b_margin_{M}"] = tally("fingers", f"b_margin_{M}", arms)
            t_["return_fingers"][f"b_margin_{M}_any"] = tally("fingers", f"b_margin_{M}_any", arms)
        for M in (0, 10, 20, 30):
            t_["carry_apple"][f"c_margin_{M}"] = tally("apple", f"c_margin_{M}", arms)
        tab[arms_lab] = t_
    out["contact_tables"] = tab

    # ---------------- new-HOLD upper bounds ----------------
    def hold_ub(phases, M_rob, M_held=None, held_sel="held_obst", arms=("V8", "V7g"), stage=None, state="P"):
        n_chunks = 0; n_flag = 0; n_flag_new = 0; by_lab = collections.Counter(); by_cause = collections.Counter(); runs = set(); n_rows = 0
        for arm in arms:
            for key, r in R[arm].items():
                if "chunks" not in r or (stage and r.get("stage") != stage):
                    continue
                for c in r["chunks"]:
                    if c["phase"] not in phases:
                        continue
                    n_chunks += 1
                    vr, lr = chunk_min(c, state, "robot_obst")
                    flag = vr < M_rob; cause = "robot" if flag else None; lab = lr if flag else None
                    if M_held is not None:
                        vh, lh = chunk_min(c, state, held_sel)
                        if vh < M_held:
                            if not flag or vh - M_held < vr - M_rob:
                                cause, lab = "held", lh
                            flag = True
                    if flag:
                        n_flag += 1
                        new = (c["to_status"] == "feasible" and (c["clearance_mm"] is None or c["clearance_mm"] >= 0.0))
                        if new:
                            n_flag_new += 1; by_lab[f"{cause}:{lab}"] += 1; by_cause[cause] += 1; runs.add(f"{arm}:{key}")
                            for row in c[state]:
                                if row is None:
                                    continue
                                v1 = cellmin(row[0], *SEL["robot_obst"])[0] < M_rob
                                v2 = (M_held is not None and cellmin(row[0], *SEL[held_sel])[0] < M_held)
                                n_rows += int(v1 or v2)
        return {"n_chunks_phase": n_chunks, "n_chunks_below": n_flag, "new_hold_upper_bound_chunks": n_flag_new, "rows_below_in_those": n_rows,
                "by_cause_label": dict(by_lab), "by_cause": dict(by_cause), "n_runs": len(runs)}
    ub = {}
    for arms_lab, arms in (("V8", ("V8",)), ("V7g", ("V7g",)), ("V8+V7g", ("V8", "V7g"))):
        u = {"return": {}, "carry": {}, "crate_entry": {}}
        for M in (10, 20, 30, 40):
            u["return"][f"M{M}"] = hold_ub(("return",), M, arms=arms)
            u["return"][f"M{M}_H"] = hold_ub(("return",), M, arms=arms, stage="H")
        u["return"]["a_servo_S_lt10"] = hold_ub(("return",), 10, arms=arms, state="S")
        for M in (20, 30):
            for ph in ("carry", "crate_entry"):
                u[ph][f"M{M}"] = hold_ub((ph,), max(10, M), M_held=M, arms=arms)
                u[ph][f"M{M}_incl_unexplained"] = hold_ub((ph,), max(10, M), M_held=M, held_sel="held_obst_unexpl", arms=arms)
        for ph in ("carry", "crate_entry"):
            u[ph]["current_M10_held0"] = hold_ub((ph,), 10, M_held=0.0, arms=arms)
            u[ph]["HM_M10_held10"] = hold_ub((ph,), 10, M_held=10.0, arms=arms)
        ub[arms_lab] = u
    out["hold_upper_bounds"] = ub
    # servo path-row activations in return where the plan rows were clear
    act = collections.Counter()
    for arm in ("V8", "V7g"):
        for key, r in R[arm].items():
            for c in r.get("chunks", []):
                if c["phase"] != "return":
                    continue
                act["chunks"] += 1
                s_, ls = chunk_min(c, "S", "robot_obst"); p_, lp = chunk_min(c, "P", "robot_obst")
                if s_ < 10.0:
                    act["S_lt10"] += 1; act[f"S_lt10_label_{ls}"] += 1
                    if p_ >= 10.0:
                        act["S_lt10_P_ge10"] += 1
    out["a_servo_return_activation"] = dict(act)
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["contact_tables"]["V8+V7g"], default=float)[:3000])
    print(json.dumps({k: {kk: (vv["new_hold_upper_bound_chunks"], vv["n_chunks_phase"]) for kk, vv in v.items()} for k, v in ub["V8+V7g"].items()}))
    print(out["a_servo_return_activation"])


if __name__ == "__main__":
    main()
