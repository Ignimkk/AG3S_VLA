"""T43 SV part 1/2 — finger height over the table in the last chunks before closing: plan P vs servo prediction S vs measured M.
usage: sv_sag_agg.py OUT_JSON      (inputs raw/rows_{V9,V8,V3,V4}.json, raw/grasp_table.json for the failure / both-grasp key sets)
Window: approach chunks (sv_rows phase) with first_closing_t - 16 <= t0 <= first_closing_t (the last 3 planning chunks up to and including
  the first closing chunk) of each run; every row k with a measured post state.
Values per row (mm): finger_table = min over finger spheres of (z_centre - table_top_z - r) (MuJoCo truth plane, sv_rows);
  TO support row = min over finger spheres of class support of d - r (the TO's own field value, no margin).
Differences M - P (measured minus plan), S - P (servo prediction minus plan), M - S (prediction error), per row.
Key sets: 'failure' = grasp lost in the servo arm and kept in the no-servo arm; 'both_grasp' = both kept (grasp_table.json kinds).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys
import numpy as np
SV = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/SV/raw'))


def stat(v):
    v = [x for x in v if x is not None and abs(x) < 1e8]
    if not v:
        return None
    a = np.asarray(v, float)
    return {"n": len(a), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


def support_min(red):
    v = [val for k, val in red.items() if k.startswith("finger|support|")]
    return min(v) if v else None


def main():
    gt = json.load(open(SV / "grasp_table.json"))["pairs"]
    out = {"definitions": __doc__, "arms": {}}
    for pair, (A, B) in (("V9V8", ("V9", "V8")), ("V3V4", ("V3", "V4"))):
        kinds = {}
        for r in gt[pair]["rows"]:
            ep_s = r["key"].split("_ep", 1)[1]; kinds[(r["stage"], ep_s)] = r["kind"]
        for arm in (A, B):
            p = SV / f"rows_{arm}.json"
            if not p.exists():
                continue
            R = json.load(open(p))["runs"]
            res = {}
            for sel in ("failure", "both_grasp"):
                vals = {k: [] for k in ("P", "S", "M", "M_minus_P", "S_minus_P", "M_minus_S", "P_support_row", "S_support_row", "M_support_row")}
                n_runs = 0
                for key, r in R.items():
                    if "chunks" not in r:
                        continue
                    stage = key.split("/")[1]; ep_s = key.split("_ep", 1)[1]
                    if kinds.get((stage, ep_s)) != sel:
                        continue
                    fc = r.get("first_closing_t")
                    if fc is None:
                        continue
                    n_runs += 1
                    for c in r["chunks"]:
                        if c["phase"] != "approach" or not (fc - 16 <= c["t0"] <= fc):
                            continue
                        for P, S, M in zip(c["P"], c["S"], c["M"]):
                            if M is None:
                                continue
                            p_, s_, m_ = P[1]["finger_table_mm"], S[1]["finger_table_mm"], M[1]["finger_table_mm"]
                            vals["P"].append(p_); vals["S"].append(s_); vals["M"].append(m_)
                            vals["M_minus_P"].append(m_ - p_); vals["S_minus_P"].append(s_ - p_); vals["M_minus_S"].append(m_ - s_)
                            vals["P_support_row"].append(support_min(P[0])); vals["S_support_row"].append(support_min(S[0])); vals["M_support_row"].append(support_min(M[0]))
                res[sel] = {"n_runs": n_runs, **{k: stat(v) for k, v in vals.items()}}
            out["arms"][arm] = res
    pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
    for arm, v in out["arms"].items():
        for sel, x in v.items():
            print(arm, sel, x["n_runs"], {k: (None if x[k] is None else (x[k]["n"], round(x[k]["median"], 2))) for k in x if k != "n_runs"})


if __name__ == "__main__":
    main()
