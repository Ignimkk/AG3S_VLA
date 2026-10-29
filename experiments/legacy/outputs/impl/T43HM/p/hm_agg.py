"""T43 HM — aggregate eval.json into the offline tables (a) carry contacts, (b) attach / lift, (c) crate entry (+ carry).

usage: hm_agg.py OUT_JSON      (reads outputs/impl/T43HM/raw/eval.json, CT contact tables of V7g / V8)

Clearances are the production `_esdf_clearance` values of hm_eval.py (held rows only), mm.
(a) events = CT apple-hurdle contacts in the carry phase (V7g: T43/CT/raw/ct.json, V8: T43/V8/ct.json).  Window = control rows
    first_t-2 .. last_t (tracking lag 1-3 rows, DX).  Per pose (planned = refined row k of the executing chunk; servo = the
    servo-predicted state of that row, HX pred_q14): min over the window rows and held queries of the off / on clearance;
    "seen" = < 0 (the TO has a violated row to fix).  cls / GT = class and GT group of that minimum row (on).
(b)/(c)/carry: per chunk min over rows 0..7 and held queries (planned pose).  new = on < 0 <= off;  already = off < 0.
    Without the remnant rule (counterfactual): rows of class `held` (own remnant) would take the 10 mm too.
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
from collections import Counter, defaultdict
import numpy as np

RAW = pathlib.Path(_ag3s_workspace_path('outputs/impl/T43HM/raw'))
GROUPS = ("hurdle", "table", "crate", "apple", "fruit", "shelf", "room", "robot")


def load_gt():
    """(key, seq, k, src) -> per held query the survey's GT distances (mm) to each group."""
    out = {}
    for arm in ("V7g", "V8"):
        for key, run in json.load(open(RAW / f"survey_{arm}.json"))["runs"].items():
            for ch in run.get("chunks", []):
                for rr in ch["rows"]:
                    for src in ("plan", "servo"):
                        if rr.get(src):
                            out[(key, ch["seq"], rr["k"], src)] = rr[src]["gt_all_mm"]
    return out


def cause(gall):
    """GT cause of a row's nearest surface: the nearest group within 20 mm, `a~b` when the next group is within
    5 mm of it, `unidentified` when no scene geom is within 20 mm."""
    order = sorted(range(len(GROUPS)), key=lambda j: gall[j])
    d1, d2 = gall[order[0]], gall[order[1]]
    if d1 > 20.0:
        return "unidentified"
    if d2 - d1 < 5.0:
        return f"{GROUPS[order[0]]}~{GROUPS[order[1]]}"
    return GROUPS[order[0]]
CT = {"V7g": _ag3s_workspace_path('outputs/verify/T43/CT/raw/ct.json'), "V8": _ag3s_workspace_path('outputs/verify/T43/V8/ct.json')}


def main():
    out_path = pathlib.Path(sys.argv[1])
    rows = json.load(open(RAW / "eval.json"))["rows"]
    GT = load_gt()
    by_t = defaultdict(list)
    by_chunk = defaultdict(list)
    for r in rows:
        if r["t"] is not None:
            by_t[(r["key"], r["t"], r["src"])].append(r)
        if r["src"] == "plan":
            by_chunk[(r["key"], r["seq"])].append(r)
    res = {"definitions": __doc__}

    # ---------------- (a) ----------------
    ev = []
    for arm, f in CT.items():
        for key, run in sorted(json.load(open(f))["runs"].items()):
            if not key.startswith(arm + "/"):
                continue
            for c in run.get("contacts", []):
                if c["kind"] != "apple" or c["phase"] != "carry":
                    continue
                e = {"key": key, "first_t": c["first_t"], "last_t": c["last_t"], "max_pen_mm": c["max_pen_mm"]}
                for src in ("plan", "servo"):
                    best = None
                    for t in range(c["first_t"] - 2, c["last_t"] + 1):
                        for r in by_t.get((key, t, src), []):
                            for i in range(len(r["on_mm"])):
                                cand = (r["on_mm"][i], r["off_mm"][i], r["cls"][i], r["gt"][i], t, r["seq"], r["k"],
                                        (r["true_hurdle_mm"] or [None] * 9)[i], r["d_minus_r_mm"][i])
                                if best is None or cand[0] < best[0]:
                                    best = cand
                    if best is None:
                        e[src] = None; continue
                    e[src] = {"on_min_mm": best[0], "off_min_mm": best[1], "cls": best[2], "gt": best[3], "t": best[4],
                              "seq": best[5], "k": best[6], "true_hurdle_mm": best[7], "d_minus_r_mm": best[8],
                              "seen_off": best[1] < 0, "seen_on": best[0] < 0}
                ev.append(e)
    res["a_events"] = ev
    res["a_counts"] = {src: {"n": sum(e[src] is not None for e in ev),
                             "seen_off": sum(bool(e[src] and e[src]["seen_off"]) for e in ev),
                             "seen_on": sum(bool(e[src] and e[src]["seen_on"]) for e in ev),
                             "cls_on": dict(Counter(e[src]["cls"] for e in ev if e[src]))} for src in ("plan", "servo")}

    # ---------------- (b) (c) carry: per chunk ----------------
    def chunk_stats(phase):
        out = {"n_chunks": 0, "B0": Counter(), "B1": Counter(), "new_causes_B0": Counter(), "new_causes_B1": Counter(),
               "new_gt_cause_B0": Counter(), "new_gt_cause_B1": Counter(),
               "remnant_rows_lt10": 0, "remnant_chunks_rescued": 0, "new_chunks_B0": [], "new_chunks_B1": [],
               "recorded_hold": 0}
        for (key, seq), L in sorted(by_chunk.items()):
            if L[0]["phase"] != phase:
                continue
            out["n_chunks"] += 1
            out["recorded_hold"] += int(L[0]["gate"] == "hold")
            for b, off_k, on_k in (("B0", "off_mm", "on_mm"), ("B1", "off_b1_mm", "on_b1_mm")):
                vals = [(r[on_k][i], r[off_k][i], r["cls"][i], r["gt"][i], r["k"], i) for r in L for i in range(len(r[on_k]))]
                on_min = min(vals); off_min = min(v[1] for v in vals)
                if off_min < 0:
                    out[b]["already"] += 1
                if on_min[0] < 0 <= off_min:
                    out[b]["new"] += 1
                    out[f"new_causes_{b}"][f"{on_min[2]}/{on_min[3]}"] += 1
                    gc = cause(GT[(key, seq, on_min[4], "plan")][on_min[5]])
                    out[f"new_gt_cause_{b}"][gc] += 1
                    out[f"new_chunks_{b}"].append({"key": key, "seq": seq, "t0": L[0]["t0"], "on_min_mm": round(on_min[0], 2),
                                                   "off_min_mm": round(off_min, 2), "cls": on_min[2], "gt": on_min[3], "k": on_min[4],
                                                   "gate": L[0]["gate"], "n_att": L[0]["n_att"], "gt_cause": gc})
            # remnant rule counterfactual (B0): rows of class held (own remnant) taking the 10 mm too
            rem = [(r["off_mm"][i]) for r in L for i in range(len(r["off_mm"])) if r["cls"][i] == "held"]
            out["remnant_rows_lt10"] += sum(v < 10.0 for v in rem)
            vals_cf = [r["on_mm"][i] - (10.0 if r["cls"][i] == "held" else 0.0) for r in L for i in range(len(r["on_mm"]))]
            off_min = min(r["off_mm"][i] for r in L for i in range(len(r["off_mm"])))
            on_min = min(r["on_mm"][i] for r in L for i in range(len(r["on_mm"])))
            if min(vals_cf) < 0 <= off_min and not on_min < 0:
                out["remnant_chunks_rescued"] += 1
        for k in ("B0", "B1", "new_causes_B0", "new_causes_B1", "new_gt_cause_B0", "new_gt_cause_B1"):
            out[k] = dict(out[k])
        return out

    for ph in ("lift", "carry", "crate"):
        res[f"chunks_{ph}"] = chunk_stats(ph)
    # rows by class (planned, d - r < 10 mm), for the record
    cnt = Counter()
    for r in rows:
        if r["src"] != "plan":
            continue
        for i in range(len(r["on_mm"])):
            if r["d_minus_r_mm"][i] < 10.0:
                cnt[(r["phase"], r["cls"][i], r["gt"][i])] += 1
    res["rows_lt10_by_phase_class_gt"] = {f"{a}/{b}/{c}": v for (a, b, c), v in sorted(cnt.items())}
    out_path.write_text(json.dumps(res, indent=1))
    # print tables
    print("(a)", json.dumps(res["a_counts"]))
    for e in ev:
        p, s = e["plan"], e["servo"]
        fmt = lambda x: "-" if x is None else f"off {x['off_min_mm']:6.1f} on {x['on_min_mm']:6.1f} {x['cls']}/{x['gt']} t{x['t']}"
        print(f"  {e['key']:26s} t{e['first_t']}-{e['last_t']} pen {e['max_pen_mm']:.2f} | plan {fmt(p)} | servo {fmt(s)}")
    for ph in ("lift", "carry", "crate"):
        c = res[f"chunks_{ph}"]
        print(ph, {k: c[k] for k in ("n_chunks", "recorded_hold", "B0", "B1", "new_gt_cause_B0", "new_gt_cause_B1", "remnant_rows_lt10", "remnant_chunks_rescued")})
        for x in c["new_chunks_B0"]:
            print("   B0 new", x)
        for x in c["new_chunks_B1"]:
            print("   B1 new", x)
    print(json.dumps(res["rows_lt10_by_phase_class_gt"], indent=0))


if __name__ == "__main__":
    main()
