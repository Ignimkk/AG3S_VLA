"""T60b — why placed runs do not return to the ready pose (record-based diagnosis, no rollout).

Pre-registered in `handoff/T60.task.md` (T60b). For every run with P (CL2 · CL3 · T53), after t_P on the used arm:
    pred8_min        min over chunks starting after t_P of min over the executed part (first `execution_length`
                     predicted steps) of max|a − q0|
    pred50_min       the same over the full predicted horizon (descriptive)
    hold_after_P     HOLD chunks (analyze_cl `hold`) starting after t_P, with kinds
    exec_dev         max over executed rows after t_P of max|executed_action − recorded chunk action of that row|
                     (the recorded `predicted_chunks` are the chunks after TO, so this only checks execution)
    to_dev           max over chunks after t_P of max|TO action − policy action| over the executed part, from the
                     planning rows' `actions` / `actions_reference` (correction 2026-10-14: the pre-registered intent
                     of exec_dev — "did TO change the proposal" — needs the policy reference, not the recorded chunk)
    pred8_ref_min    pred8_min computed on the policy reference (`actions_reference`)
    tcp_move         max |TCP − TCP(t_P)| after t_P ; tcp_target_final = |TCP − target| at the last row
    q_err_final, still (max joint motion over the last 80 rows), rows_after_P
Category (priority order): (b) blocked = HOLD after P, or (exec_dev or to_dev) > 0.05 rad with
pred8_ref_min ≤ q_tol (the policy proposed the return but it was not executed as proposed);
(c) retreat elsewhere = tcp_move ≥ 0.10 m ∧ tcp_target_final ≥ 0.10 m; (a) no return proposed = otherwise.

usage: t60b_return.py STEP1_DIR OUT_JSON
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402
from AG3S.ag3s.experiments.studies.t60_timestamps import SETS  # noqa: E402

EXEC_DEV_TOL = 0.05
RETREAT_M = 0.10   # transport_atomic.RETREAT_TARGET_CLEARANCE_M
STILL_ROWS = 80


def classify(m: dict, q_tol: float) -> str:
    """Pre-registered T60b category (priority order) from the run measures."""
    dev_any = max(m["exec_dev_max"], m.get("to_dev_max") or 0.0)
    ref = m["pred8_ref_min"] if m.get("pred8_ref_min") is not None else float("inf")
    if m["hold_after_P"] > 0 or (dev_any > EXEC_DEV_TOL and ref <= q_tol):
        return "b_blocked"
    if m["tcp_move_m"] >= RETREAT_M and m["tcp_target_final_m"] >= RETREAT_M:
        return "c_retreat_elsewhere"
    return "a_no_return_proposed"


def diagnose(base: pathlib.Path, arm: str, name: str, r: dict, p: T.Params) -> dict:
    tj = np.load(base / "runs" / arm / "R" / name / "traj.npz", allow_pickle=True)
    rp = np.load(base / "replay" / f"{arm}_R_{name}.npz", allow_pickle=True)
    an = json.loads((base / "analysis" / f"{name}.json").read_text())
    sl = T.ARM_SLICE[r["used_arm"]]
    q = tj["measured_qpos"]; q0 = q[0, sl]
    ex = tj["executed_actions"]; pc = tj["predicted_chunks"]; cs = np.asarray(tj["chunk_start_steps"])
    L = int(tj["execution_length"]) if "execution_length" in tj.files else T.CHUNK
    t_p = r["t_P"]; n = len(q)
    after = [c for c in range(len(cs)) if cs[c] > t_p]
    err = lambda a: np.abs(a[..., sl] - q0).max(axis=-1)
    pred8 = [float(err(pc[c, :L]).min()) for c in after]
    pred50 = [float(err(pc[c]).min()) for c in after]
    first8_ok = next((int(cs[c]) for c, v in zip(after, pred8) if v <= p.q_tol), None)
    executed = np.asarray(rp["executed"], bool)
    dev = []
    for c in after:
        for k in range(L):
            t = int(cs[c]) + k
            if t < n and executed[t]:
                dev.append(float(np.abs(ex[t, sl] - pc[c, k, sl]).max()))
    rows = [json.loads(l) for l in (base / "runs" / arm / "R" / name / "frames.jsonl").read_text().splitlines() if l.strip()]
    plan = {int(x["t_step"]): x for x in rows if x["kind"] == "planning" and x.get("actions_reference") is not None}
    to_dev, pred8_ref = [], []
    for c in after:
        x = plan.get(int(cs[c]))
        if x is None:
            continue
        ref = np.asarray(x["actions_reference"], float)[:L]
        act = np.asarray(x["actions"], float)[:L]
        pred8_ref.append(float(err(ref).min()))
        to_dev.append(float(np.abs(act[:, sl] - ref[:, sl]).max()))
    holds = [h for h in an["hold"] if h["t"] > t_p]
    tcp = rp["tcp"][:n]; tgt = rp["target_pos"][:n]
    tcp_move = float(np.linalg.norm(tcp[t_p + 1:] - tcp[t_p], axis=1).max()) if t_p + 1 < n else 0.0
    out = {
        "rows_after_P": n - 1 - t_p, "n_chunks_after_P": len(after),
        "pred8_min": min(pred8) if pred8 else None, "pred50_min": min(pred50) if pred50 else None,
        "first_chunk_t_pred8_within_qtol": first8_ok,
        "hold_after_P": len(holds), "hold_kinds_after_P": dict(Counter(k for h in holds for k in h["kinds"])),
        "exec_dev_max": max(dev) if dev else 0.0,
        "to_dev_max": max(to_dev) if to_dev else None, "pred8_ref_min": min(pred8_ref) if pred8_ref else None,
        "n_chunks_with_reference": len(pred8_ref),
        "tcp_move_m": tcp_move, "tcp_target_final_m": float(np.linalg.norm(tcp[-1] - tgt[-1])),
        "q_err_final": float(err(q[-1:])[0]), "q_err_at_P": float(err(q[t_p:t_p + 1])[0]),
        "still_rad": float(np.abs(q[-STILL_ROWS:, sl] - q[-1, sl]).max()),
        "gripper_open_final": float(rp["finger_open_norm"][min(n, len(rp["finger_open_norm"]) - 1)]),
    }
    out["category"] = classify(out, p.q_tol)
    return out


def summary(rows: list) -> dict:
    keys = ("rows_after_P", "pred8_min", "pred8_ref_min", "pred50_min", "exec_dev_max", "to_dev_max", "tcp_move_m", "tcp_target_final_m",
            "q_err_final", "still_rad")
    o = {"n": len(rows), "category": dict(Counter(r["category"] for r in rows)),
         "hold_after_P_runs": sum(r["hold_after_P"] > 0 for r in rows),
         "pred8_within_qtol_runs": sum((r["pred8_min"] is not None and r["pred8_min"] <= 0.35) for r in rows),
         "pred8_ref_within_qtol_runs": sum((r["pred8_ref_min"] is not None and r["pred8_ref_min"] <= 0.35) for r in rows),
         "to_dev_gt_tol_runs": sum(((r["to_dev_max"] or 0) > EXEC_DEV_TOL) for r in rows),
         "pred50_within_qtol_runs": sum((r["pred50_min"] is not None and r["pred50_min"] <= 0.35) for r in rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if r[k] is not None], float)
        o[k] = {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())} if v.size else None
    return o


def main(step1: pathlib.Path, out_json: pathlib.Path) -> None:
    ts = json.loads((step1 / "timestamps.json").read_text())
    p = T.Params()
    runs = {}
    for key, r in ts["runs"].items():
        if r["t_P"] is None:
            continue
        d = diagnose(SETS[r["set"]], r["arm"], r["name"], r, p)
        d.update({"set": r["set"], "arm": r["arm"], "episode": r["episode"], "seed": r["seed"],
                  "used_arm": r["used_arm"], "t_P": r["t_P"], "t_H": r["t_H"], "returned": r["t_H"] is not None})
        runs[key] = d
    nr = [v for v in runs.values() if not v["returned"]]
    rr = [v for v in runs.values() if v["returned"]]
    by_ep = Counter((v["set"], v["episode"]) for v in nr)
    eps = {}
    for v in runs.values():
        e = eps.setdefault(str(v["episode"]), {"returned": 0, "not_returned": 0, "used_arm": v["used_arm"]})
        e["returned" if v["returned"] else "not_returned"] += 1
    res = {"definitions": __doc__, "params": p, "not_returned": summary(nr), "returned": summary(rr),
           "by_set_arm": {f"{s}/{a}": {"not_returned": sum(1 for v in nr if v["set"] == s and v["arm"] == a),
                                      "placed": sum(1 for v in runs.values() if v["set"] == s and v["arm"] == a)}
                          for s, a in sorted({(v["set"], v["arm"]) for v in runs.values()})},
           "by_episode": eps, "runs": runs}
    T.write_json(out_json, res)
    print("not returned", json.dumps(res["not_returned"], indent=0)[:1500])
    print("returned", json.dumps(res["returned"], indent=0)[:1500])
    print("by episode", {k: v for k, v in eps.items()})


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
