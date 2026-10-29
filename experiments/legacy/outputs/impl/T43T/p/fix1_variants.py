"""T43 T fix 1 — variants on the chunks where the recovery floor engaged (fix1_census.json):
A = as implemented (floor collision/unresolved + allowed rows violated at q_now, stationary second start);
B = floor collision/unresolved rows only; C = A without the stationary start; D = B without it."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, dataclasses
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import qnow as Q
import recov_replay as RR
from AG3S.trajopt import sqp as SQP
import AG3S.trajopt.safe_policy as SP

ORIG_STAY = SQP.TrajectoryOptimizer._stay_start
ORIG_FLOOR = SP.recovery_floor_rows

def floor_no_allowed(*a, **k):
    floor, rec = ORIG_FLOOR(*a, **k)
    if floor is not None:
        floor = floor.copy(); floor[rec["allowed_queries"]] = 0.0
    return floor, rec

def no_stay(self, iterate, states, merit, violation, *a, **k):
    return iterate, states, merit, violation, {"source": "reference (stay start off)"}

def main(out):
    cen = json.load(open(_ag3s_workspace_path('outputs/impl/T43T/fix1_census.json')))
    dead = {("noobs/E3b_ep1908_s19082", 16), ("noobs/E3b_ep1908_s19082", 24),
            ("noobs/E3b_ep1992_s19922", 16), ("noobs/E3b_ep1992_s19922", 24)}
    todo = [(r["run"], c["seq"], c["off_hold"]) for r in cen for c in r["chunks"] if c["engaged"]
            and (c["off_hold"] != c["on_hold"] or (r["run"].split("/E3b/")[-1], c["seq"]) in dead)]
    res = []
    for rd, seq, off_hold in todo:
        Q.RUNS["X"] = rd
        run = RR.Run("X")
        row = {"run": rd.split("/runs/")[-1], "seq": seq, "off_hold": off_hold}
        for name, (fl, st) in {"A": (ORIG_FLOOR, ORIG_STAY), "B": (floor_no_allowed, ORIG_STAY),
                               "C": (ORIG_FLOOR, no_stay), "D": (floor_no_allowed, no_stay)}.items():
            SP.recovery_floor_rows = fl
            SQP.TrajectoryOptimizer._stay_start = st
            r = run.solve(seq, recovery=True, overrides={"safety": {"recovery_floor": True}})
            row[name] = {"hold": r["hold"], "reasons": r["reasons"], "returned": r["returned"],
                         "progress": None if r["progress"] is None else r["progress"]["ok"]}
        SP.recovery_floor_rows, SQP.TrajectoryOptimizer._stay_start = ORIG_FLOOR, ORIG_STAY
        res.append(row)
        print(row["run"], seq, "off", off_hold, {k: row[k]["hold"] for k in "ABCD"}, flush=True)
    pathlib.Path(out).write_text(json.dumps(res, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
