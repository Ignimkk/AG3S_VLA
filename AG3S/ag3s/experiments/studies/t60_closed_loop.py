"""T60 step 2a — closed-loop metrics of CL2 / CL3 / T53 re-aggregated under the P and H windows.

Fixed arms, fixed records (no rollout). For every run three windows:
    full   the published definition (agg_cl.py): success = grasp ∧ target in crate at the last planning row;
           success_ct = success ∧ success_rel ∧ no disallowed contact on rows t ≤ release_t (Z4).
    P      rows t ≤ t_P (all rows when P is not reached). success_P = grasp ∧ P reached;
           success_ct_P = success_P ∧ no disallowed contact on rows ≤ t_P.
    H      rows t ≤ t_H (all rows when H is not reached). success_H = success_P ∧ H reached (user decision
           2026-10-13: a placed run that does not return is an H failure; P success is reported next to it);
           success_ct_H = success_H ∧ no disallowed contact on rows ≤ t_H (place → ready included).
Disallowed classes = the V11 CT set {hand-fruit_other, robot_other-target, robot_other-fruit_other};
``_fruit`` = the fruit-only pair. HOLD chunk = a planning chunk with a non-executed control row (analyze_cl);
it is inside a window when its start t ≤ t_end. Completion time = (t_end + 1) / control_hz.
The full window is asserted against the published agg.json before any P / H number is written.

usage: t60_closed_loop.py STEP1_DIR OUT_JSON
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402
from AG3S.ag3s.experiments.studies.t60_timestamps import SETS  # noqa: E402

PAIRS = {
    "CL2": [("C0", "C1a"), ("C1a", "C1bp"), ("C1bp", "C2p"), ("C0", "C1bp"), ("C0", "C2p")],
    "CL3": [("C1a", "C1aF1"), ("C1aF1", "C1aF1F2"), ("C1a", "C1aF1F2")],
    "T53": [],
}
WINDOWS = ("full", "P", "H")
FLAGS = ("success", "success_ct", "success_ct_fruit", "any_disallowed")


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def run_windows(set_name: str, name: str, ts: dict, agg_run: dict, an: dict) -> dict:
    base = SETS[set_name]
    arm = agg_run["arm"]
    s = T.series_from_record(base / "replay" / f"{arm}_R_{name}.npz",
                             base / "runs" / arm / "R" / name / "traj.npz", agg_run["used_arm"])
    t = np.asarray(s["t"]); n = len(t); hz = s["control_hz"]
    t_p, t_h = ts["t_P"], ts["t_H"]
    holds = an["hold"]
    out = {"arm": arm, "episode": agg_run["episode"], "seed": agg_run["seed"], "t_P": t_p, "t_H": t_h}
    rel = agg_run.get("release_t")
    ends = {"full": None, "P": t_p, "H": t_h}
    for w in WINDOWS:
        t_end = ends[w]
        keep = T.keep_rows(n, t_end)
        ct_end = (rel if rel is not None else n - 1) if w == "full" else (t_end if t_end is not None else n - 1)
        ct_rows = t <= ct_end
        rows = {c: int(((s["cnt_" + c] > 0) & ct_rows).sum()) for c in T.DISALLOWED}
        any_dis = any(rows[c] for c in T.DISALLOWED)
        any_fruit = any(rows[c] for c in T.FRUIT_DISALLOWED)
        if w == "full":
            succ = bool(agg_run["success"])
            sct = bool(succ and agg_run["success_rel"] and not any_dis)
            sct_f = bool(succ and agg_run["success_rel"] and not any_fruit)
        else:
            succ = bool(agg_run["grasp"] and t_p is not None and (w == "P" or t_h is not None))
            sct = bool(succ and not any_dis)
            sct_f = bool(succ and not any_fruit)
        hw = [h for h in holds if t_end is None or h["t"] <= t_end]
        kinds = Counter(k for h in hw for k in h["kinds"])
        out[w] = {"success": succ, "success_ct": sct, "success_ct_fruit": sct_f, "any_disallowed": any_dis,
                  "disallowed_rows": rows, "hold_chunks": len(hw), "hold_kinds": dict(kinds),
                  "rows_in_window": int(keep.sum()),
                  "completion_s": None if (w == "full" or t_end is None or not succ) else (t_end + 1) / hz}
    # HOLD position relative to the endpoints (T46b / W1 cross-table)
    pos = Counter()
    for h in holds:
        for k in h["kinds"]:
            if t_p is None:
                where = "no_P"
            elif h["t"] <= t_p:
                where = "le_tP"
            elif t_h is None:
                where = "after_tP_noH"
            elif h["t"] <= t_h:
                where = "tP_to_tH"
            else:
                where = "after_tH"
            pos[f"{k}|{where}"] += 1
    out["hold_position"] = dict(pos)
    out["pre8_close_hold_seqs"] = (agg_run.get("pre_grasp") or {}).get("hold_seqs_window", [])
    return out


def arm_summary(runs: list) -> dict:
    o = {"n_runs": len(runs)}
    for w in WINDOWS:
        rs = [r[w] for r in runs]
        comp = [x["completion_s"] for x in rs if x["completion_s"] is not None]
        kinds = Counter()
        for x in rs:
            kinds.update(x["hold_kinds"])
        o[w] = {
            "success": sum(x["success"] for x in rs), "success_ct": sum(x["success_ct"] for x in rs),
            "success_ct_fruit": sum(x["success_ct_fruit"] for x in rs),
            "runs_any_disallowed": sum(x["any_disallowed"] for x in rs),
            "disallowed_rows": {c: sum(x["disallowed_rows"][c] for x in rs) for c in T.DISALLOWED},
            "hold_chunks": sum(x["hold_chunks"] for x in rs), "hold_runs": sum(x["hold_chunks"] > 0 for x in rs),
            "hold_kinds": dict(kinds), "rows_in_window": sum(x["rows_in_window"] for x in rs),
            "completion_s_median": float(np.median(comp)) if comp else None,
            "completion_s_mean": float(np.mean(comp)) if comp else None, "n_completed": len(comp),
        }
    pos = Counter()
    for r in runs:
        pos.update(r["hold_position"])
    o["hold_position"] = dict(pos)
    return o


def pair_table(runs: dict, a: str, b: str, w: str, flag: str) -> dict:
    keys = sorted({(r["episode"], r["seed"]) for r in runs.values() if r["arm"] in (a, b)})
    idx = {(r["arm"], r["episode"], r["seed"]): r for r in runs.values()}
    both = a_only = b_only = neither = 0
    a_keys, b_keys = [], []
    for e, sd in keys:
        ra, rb = idx.get((a, e, sd)), idx.get((b, e, sd))
        if ra is None or rb is None:
            continue
        x, y = ra[w][flag], rb[w][flag]
        both += x and y; neither += (not x) and (not y)
        if x and not y:
            a_only += 1; a_keys.append(f"ep{e}_s{sd}")
        if y and not x:
            b_only += 1; b_keys.append(f"ep{e}_s{sd}")
    return {"A": a, "B": b, "both": both, "A_only": a_only, "B_only": b_only, "neither": neither,
            "A_total": both + a_only, "B_total": both + b_only, "mcnemar_exact_p": mcnemar_exact(a_only, b_only),
            "A_only_keys": a_keys, "B_only_keys": b_keys}


def main(step1: pathlib.Path, out_json: pathlib.Path) -> None:
    ts = json.loads((step1 / "timestamps.json").read_text())
    res = {"protocol": T.PROTOCOL_VERSION, "definitions": __doc__, "params": ts["primary"], "sets": {}}
    for set_name, base in SETS.items():
        agg = json.loads((base / "agg.json").read_text())
        runs = {}
        for name, ar in agg["runs"].items():
            an = json.loads((base / "analysis" / f"{name}.json").read_text())
            r = run_windows(set_name, name, ts["runs"][f"{set_name}/{name}"], ar, an)
            # reproduction of the published full-window run flags
            assert r["full"]["success"] == ar["success"], name
            assert r["full"]["success_ct"] == ar["success_ct"], (name, r["full"]["success_ct"], ar["success_ct"])
            assert r["full"]["success_ct_fruit"] == ar["success_ct_fruit"], name
            assert r["full"]["any_disallowed"] == ar["pre_release_new_contact"], name
            assert r["full"]["hold_chunks"] == ar["n_hold_chunks"], name
            runs[name] = r
        arms = sorted({r["arm"] for r in runs.values()}, key=list(agg["arms"]).index)
        summ = {a: arm_summary([r for r in runs.values() if r["arm"] == a]) for a in arms}
        for a in arms:  # arm totals against the published table
            pub = agg["arms"][a]
            assert summ[a]["full"]["success"] == pub["t1_success"]["task_success"], a
            assert summ[a]["full"]["success_ct"] == pub["t1_success"]["success_ct"], a
            assert summ[a]["full"]["success_ct_fruit"] == pub["t1_success"]["success_ct_fruit"], a
            assert summ[a]["full"]["hold_chunks"] == pub["t4_hold"]["hold_chunks_total"], a
        pairs = {f"{a}_vs_{b}": {w: {f: pair_table(runs, a, b, w, f) for f in FLAGS} for w in WINDOWS}
                 for a, b in PAIRS[set_name]}
        res["sets"][set_name] = {"source": str(base), "arms": summ, "pairs": pairs, "runs": runs,
                                 "full_reproduced": "run flags (success, success_ct, success_ct_fruit, "
                                                    "pre_release_new_contact, n_hold_chunks) and arm totals equal agg.json"}
    T.write_json(out_json, res)
    for st, v in res["sets"].items():
        for a, s in v["arms"].items():
            print(st, a, {w: (s[w]["success"], s[w]["success_ct"], s[w]["hold_chunks"], s[w]["completion_s_median"]) for w in WINDOWS})
        for p, d in v["pairs"].items():
            print("  ", p, {w: (d[w]["success_ct"]["A_total"], d[w]["success_ct"]["B_total"],
                                round(d[w]["success_ct"]["mcnemar_exact_p"], 3)) for w in WINDOWS})


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
