"""T43 V7g (copy of T43/Z3V/p/z4_z3v.py; V5 from T43/Y/replay, V5D from T43/V7g/replay_V5D or T43/Z3V/replay, V7g from T43/V7g/replay) — re-score every T43 W run with a release requirement, next to the old rule (verifier, CPU only, no new runs).

usage: z4_release.py OUT_JSON

Inputs: physics replays (Y/p/replay_y.py, qpos_mismatch_rows 0) — V4 / V5 from T43/Y/replay, E0 / V1 / V3 / V5S / V5B from T43/Z/replay;
  old outcome from T43/W/analysis/analysis.json (E0 V1 V3 V4 V5) and T43/W/analysis_blind/analysis.json (V5S V5B).
Old rule (unchanged, analyze_e0): grasp = planning-row apple z >= z0 + 50 mm; place = apple centre inside the crate place box at the
  LAST planning row (crate-local |x| < 0.077, |y| < 0.137, -0.052 < z < 0.075); success = grasp and place.
Row convention as y_metrics.py: control row i, post-state = replay state index i + 1; d(i) = |apple centre - finger midpoint| (post-state).
  attach_t   = first analysis server chunk with attach_now (none for E0 / runs without attach); at = attach_t, else grasp_t.
  release_t  = (Y) last row with >= 1 apple-hand MuJoCo contact sub-step + 1, if that row exists before the run's end and grasp is true
               -> by construction no apple-hand contact on any row >= release_t.
  release_mode (Y) = 'open_cmd' if the first row t >= at with applied gripper command >= 0.9 is <= release_t, else 'closed_cmd'.
  release_with_cmd = applied gripper command at release_t (1 = open, 0 = close).
  sep_threshold_mm = d(grasp_t row) + 30 mm (analysis hurdle 'apple_left_hand' threshold, recomputed on the replay).
  sep_t      = first row >= release_t with d > threshold.  sep_kept = sep_t exists and d > threshold on every row from sep_t to the end.
  in_crate_end = apple centre inside the place box (old box) at the post-state of the LAST control row.
  in_crate_at_release = same box at the post-state of row release_t (apple centre z may be above the box -> also xy-only flag).
New rule: released = release_t exists and sep_kept;  place_rel = released and in_crate_end;  success_rel = grasp (old) and place_rel.
Flip lists: runs with success != success_rel. McNemar exact two-sided on discordant pairs (min(b,c) binomial, p = min(1, 2 P)),
  pairs by (episode, seed) of stage H: V5 vs V4, V5B vs V5S, V5 vs V1 (same pairings as W final_agg)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, math
import numpy as np
import mujoco

T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
LIM = (0.077, 0.137); ZR = (-0.052, 0.075)
REPLAY = {"V5": T / "Y/replay", "V5D": T / "V7g/replay_V5D", "V7g": T / "V7g/replay"}
if (T / "Z3V/replay").exists() and not (T / "V7g/replay_V5D").exists():
    REPLAY["V5D"] = T / "Z3V/replay"


def qmat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)


def local(z, i):
    return qmat(z["crate_quat"][i]).T @ (z["apple_pos"][i] - z["crate_pos"][i])


def inbox(p):
    return bool(abs(p[0]) < LIM[0] and abs(p[1]) < LIM[1] and ZR[0] < p[2] < ZR[1])


def score(key, r):
    arm, stage, name = key.split("/")
    rz = REPLAY[arm] / f"{arm}_{stage}_{name}.npz"
    z = np.load(rz); t = z["t"]; N = len(t)
    d = np.linalg.norm(z["apple_pos"] - z["finger_mid"], axis=1)          # state index
    sc = r.get("server_chunks") or []
    attach_t = next((c["t"] for c in sc if c.get("attach_now")), None)
    o = {"key": key, "arm": arm, "stage": stage, "episode": int(r["episode"]), "seed": int(name.split("_s")[-1]), "success": bool(r["success"]),
         "grasp": bool(r["grasp"]), "place": bool(r["place"]), "grasp_t": r["grasp_t"], "place_t": r["place_t"], "attach_t": attach_t, "n_rows": int(N),
         "replay": str(rz)}
    at = attach_t if attach_t is not None else (r["grasp_t"] or 0)
    ah = np.where(z["cnt_apple-hand"] > 0)[0]
    rel_i = (int(ah[-1]) + 1) if (len(ah) and ah[-1] + 1 < N and r["grasp"]) else None
    o["n_apple_hand_contact_rows"] = int(len(ah))
    o["release_t"] = int(t[rel_i]) if rel_i is not None else None
    o["applied_open_t"] = next((int(t[i]) for i in range(N) if t[i] >= at and z["grip_cmd"][i] >= 0.9), None) if r["grasp"] else None
    pe = local(z, N)
    o["apple_in_crate_end_mm"] = (1e3 * pe).tolist(); o["in_crate_end"] = inbox(pe)
    o["apple_finger_mid_end_mm"] = 1e3 * float(d[-1])
    if r["grasp"] and r["grasp_t"] is not None:
        ig = min(int(np.searchsorted(t, r["grasp_t"])), N - 1)
        thr = float(d[ig + 1]) + 0.030
        o["sep_threshold_mm"] = 1e3 * thr
    else:
        thr = None; o["sep_threshold_mm"] = None
    o["release_mode"] = o["release_with_cmd"] = o["sep_t"] = None; o["sep_kept"] = False; o["in_crate_at_release"] = None; o["in_crate_xy_at_release"] = None
    if rel_i is not None:
        o["release_mode"] = "open_cmd" if (o["applied_open_t"] is not None and o["applied_open_t"] <= o["release_t"]) else "closed_cmd"
        o["release_with_cmd"] = float(z["grip_cmd"][rel_i]); o["release_in_hold"] = bool(z["hold"][rel_i])
        pr = local(z, rel_i + 1); o["apple_in_crate_at_release_mm"] = (1e3 * pr).tolist()
        o["in_crate_at_release"] = inbox(pr); o["in_crate_xy_at_release"] = bool(abs(pr[0]) < LIM[0] and abs(pr[1]) < LIM[1])
        if thr is not None:
            idx = [i for i in range(rel_i, N) if d[i + 1] > thr]
            if idx:
                k = idx[0]; o["sep_t"] = int(t[k]); o["sep_kept"] = bool(np.all(d[k + 1:] > thr))
                o["min_d_after_sep_mm"] = 1e3 * float(d[k + 1:].min())
    o["released"] = bool(o["release_t"] is not None and o["sep_kept"])
    o["place_rel"] = bool(o["released"] and o["in_crate_end"])
    o["success_rel"] = bool(o["grasp"] and o["place_rel"])
    o["flip"] = None if o["success"] == o["success_rel"] else ("S->F" if o["success"] else "F->S")
    return o


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def pair(R, A, B, field):
    ka = {(o["episode"], o["seed"]): o for o in R.values() if o["arm"] == A and o["stage"] == "H"}
    kb = {(o["episode"], o["seed"]): o for o in R.values() if o["arm"] == B and o["stage"] == "H"}
    keys = sorted(set(ka) & set(kb))
    both = sum(ka[k][field] and kb[k][field] for k in keys); a_only = sum(ka[k][field] and not kb[k][field] for k in keys)
    b_only = sum(kb[k][field] and not ka[k][field] for k in keys); nei = len(keys) - both - a_only - b_only
    return {"n_pairs": len(keys), "both": both, f"{A}_only": a_only, f"{B}_only": b_only, "neither": nei, "mcnemar_exact_p": mcnemar(a_only, b_only),
            f"{A}_only_keys": [f"{e}_s{s}" for e, s in keys if ka[(e, s)][field] and not kb[(e, s)][field]],
            f"{B}_only_keys": [f"{e}_s{s}" for e, s in keys if kb[(e, s)][field] and not ka[(e, s)][field]]}


if __name__ == "__main__":
    R = {}
    for f, pre in ((T / "W/analysis/analysis.json", "V5/"), (T / "Z3V/analysis/analysis.json", "V5D/"), (T / "V7g/analysis/analysis.json", "V7g/")):
        A = json.load(open(f))["runs"]
        for k in sorted(A):
            if k.startswith(pre):
                R[k] = score(k, A[k])
    arms = {}
    for o in R.values():
        a = arms.setdefault(o["arm"], {}).setdefault(o["stage"], {"n": 0, "success": 0, "grasp": 0, "place": 0, "released": 0, "place_rel": 0, "success_rel": 0,
                                                              "release_open_cmd": 0, "release_closed_cmd": 0, "no_release": 0, "flip_S_to_F": 0, "flip_F_to_S": 0})
        a["n"] += 1
        for k in ("success", "grasp", "place", "released", "place_rel", "success_rel"):
            a[k] += o[k]
        a["release_open_cmd" if o["release_mode"] == "open_cmd" else ("release_closed_cmd" if o["release_mode"] == "closed_cmd" else "no_release")] += 1
        a["flip_S_to_F"] += o["flip"] == "S->F"; a["flip_F_to_S"] += o["flip"] == "F->S"
    def pair_all(A_, B_, field, stage):
        ka = {(o["stage"], o["episode"], o["seed"]): o for o in R.values() if o["arm"] == A_ and (stage == "ALL" or o["stage"] == stage)}
        kb = {(o["stage"], o["episode"], o["seed"]): o for o in R.values() if o["arm"] == B_ and (stage == "ALL" or o["stage"] == stage)}
        keys = sorted(set(ka) & set(kb))
        both = sum(ka[k][field] and kb[k][field] for k in keys); ao = sum(ka[k][field] and not kb[k][field] for k in keys); bo = sum(kb[k][field] and not ka[k][field] for k in keys)
        return {"n_pairs": len(keys), f"{A_}_total": both + ao, f"{B_}_total": both + bo, "both": both, f"{A_}_only": ao, f"{B_}_only": bo, "neither": len(keys) - both - ao - bo,
                "mcnemar_exact_p": mcnemar(ao, bo),
                f"{A_}_only_keys": [f"{k[0]}:{k[1]}_s{k[2]}" for k in keys if ka[k][field] and not kb[k][field]], f"{B_}_only_keys": [f"{k[0]}:{k[1]}_s{k[2]}" for k in keys if kb[k][field] and not ka[k][field]]}
    pairs = {}
    for A_, B_ in (("V7g", "V5"), ("V7g", "V5D"), ("V5D", "V5")):
        pairs[f"{A_}_vs_{B_}"] = {st: {f: pair_all(A_, B_, f, st) for f in ("success", "grasp", "place", "success_rel", "place_rel", "released")} for st in ("H", "B", "ALL")}
    flips = [{"key": k, "flip": o["flip"], "release_t": o["release_t"], "release_mode": o["release_mode"], "sep_kept": o["sep_kept"], "in_crate_end": o["in_crate_end"],
              "place_t": o["place_t"], "apple_finger_mid_end_mm": o["apple_finger_mid_end_mm"]} for k, o in R.items() if o["flip"]]
    res = {"definitions": __doc__, "replay_dirs": {k: str(v) for k, v in REPLAY.items()}, "arms": arms, "pairs": pairs, "flips": flips, "runs": R}
    pathlib.Path(sys.argv[1]).write_text(json.dumps(res, indent=1, default=float))
    for a in arms:
        for s_, v in arms[a].items():
            print(a, s_, v)
    for pn, pv in pairs.items():
        for st, v in pv.items():
            for f, x in v.items():
                print(pn, st, f, {k: x[k] for k in x if k in ("n_pairs", "both", "neither") or k.endswith("_only") or k.endswith("_total")}, round(x["mcnemar_exact_p"], 4))
