"""T44dA verifier: C1a vs C1b (and C1b vs C2) divergence per (ep, seed) pair. Per seq: policy output diff (reference_chunk A vs B, rows 0..7 arm cols),
executed diff (refined_chunk A vs B), state diff (planning-row qpos arm joints A vs B), TO deflection |refined - reference| in A and in B (rows 0..7 arm cols).
first_ref    = first seq with reference diff > 1e-6 (policy saw a different state)
first_fin    = first seq with refined diff > 1e-6
first_mat    = first seq with refined diff > THR (0.01 rad)
first_defl   = first seq where B's TO deflection exceeds A's by > THR while the references agree within 1e-3 (TO, not the policy, made the difference)
-> CLA/diverge_<A>_<B>.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL')); OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA'))
ARMC = list(range(7)) + list(range(8, 15)); THR = 0.01
A, B = sys.argv[1], sys.argv[2]; keys = sys.argv[3:]

def load(arm, k):
    a = json.load(open(CL / "analysis" / f"{arm}_{k}.json"))
    rows = [json.loads(l) for l in open(pathlib.Path(_ag3s_workspace_path('')) / a["run_dir"] / "frames.jsonl")]
    P = {r["seq"]: r for r in rows if r["kind"] == "planning"}
    ch = {c["seq"]: c for c in a["server_chunks"]}
    return a, P, ch

res = {}
for k in keys:
    a, PA, CA = load(A, k); b, PB, CB = load(B, k)
    per = []
    for seq in sorted(set(CA) & set(CB)):
        za = np.load(CA[seq]["server_file"]); zb = np.load(CB[seq]["server_file"])
        ra, fa = za["reference_chunk"][:8, ARMC], za["refined_chunk"][:8, ARMC]
        rb, fb = zb["reference_chunk"][:8, ARMC], zb["refined_chunk"][:8, ARMC]
        qa = np.asarray(PA[seq]["qpos"]); qb = np.asarray(PB[seq]["qpos"])
        per.append({"seq": seq, "t": PA[seq]["t_step"], "ref_diff": float(np.abs(ra - rb).max()), "fin_diff": float(np.abs(fa - fb).max()),
                    "qpos_diff": float(np.abs(qa - qb).max()), "defl_A": float(np.abs(fa - ra).max()), "defl_B": float(np.abs(fb - rb).max()),
                    "hold_A": CA[seq]["hold"], "hold_B": CB[seq]["hold"]})
    f = lambda cond: next((p["seq"] for p in per if cond(p)), None)
    res[k] = {"first_ref": f(lambda p: p["ref_diff"] > 1e-6), "first_fin": f(lambda p: p["fin_diff"] > 1e-6), "first_mat": f(lambda p: p["fin_diff"] > THR),
              "first_defl": f(lambda p: p["ref_diff"] < 1e-3 and p["defl_B"] - p["defl_A"] > THR),
              "first_hold_diff": f(lambda p: p["hold_A"] != p["hold_B"]),
              "grasp_t": {A: a["grasp_t"], B: b["grasp_t"]}, "failure_mode": {A: a["failure_mode"], B: b["failure_mode"]}, "per_seq": per}
    r = res[k]; print(k, {x: r[x] for x in ("first_ref", "first_fin", "first_mat", "first_defl", "first_hold_diff", "failure_mode")})
json.dump(res, open(OUT / f"diverge_{A}_{B}.json", "w"), indent=0)
