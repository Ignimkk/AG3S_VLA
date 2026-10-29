"""T64 Part C — can the 6-phase HSR recover the attaches the strong rule missed, without false holds? (`handoff/T64.task.md`)

HELD' = strong HELD  OR  (6-phase HSR ∈ {lift, transport, placement}  AND  the latch's finger evidence `closed` and
`blocked` are both true in that request's grasp record). Data: T55 test re-run-grounding records (`t55-legacy`):
instruction (prompt noun changed), natural, empty grasp, slip. Truth = T55 physics P_grasp (pre-registered K).

Criteria (all, fixed before results):
  1. instruction: recovered / missed-by-strong P_grasp frames ≥ 0.50
  2. empty + slip: false HELD' frames (P_grasp false) − false strong HELD frames == 0
  3. natural: false-pass rate increase (P_task) ≤ +1 %p

usage: t64_partC.py PARTA_DIR OUT_DIR      (.venv-ag3s)
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
T55 = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c/verify/T55")
HOLD_PHASES = (2, 3, 4)


def finger_evidence(q) -> np.ndarray:
    out = []
    for g in q["grasp_json"]:
        g = str(g)
        ch = (json.loads(g).get("evidence") or {}).get("checks") or {} if g else {}
        out.append(bool(ch.get("closed")) and bool(ch.get("blocked")))
    return np.array(out, bool)


def main(partA: pathlib.Path, out: pathlib.Path) -> None:
    from AG3S.ag3s.experiments.studies import t55_study as S
    from AG3S.ag3s.experiments.studies.t50_core import LogReg

    out.mkdir(parents=True, exist_ok=True)
    pz = np.load(partA / "probe_6.npz", allow_pickle=True)
    probe = LogReg(mean=pz["mean"], scale=pz["scale"], coef=pz["coef"], intercept=pz["intercept"], C=float(pz["C"]),
                   n_iter=0, converged=True)
    prereg = json.loads((T55 / "T55.prereg.json").read_text())
    K, pm = int(prereg["truth"]["K"]), prereg["truth"].get("persistence", "causal")
    traces = S.load_resim_traces(T55 / "resim/test", T55 / "feat/test", "test",
                                 perception_dir=T55 / "perception/test/t55-legacy")
    cache = S.FeatureCache()
    acc = {}
    for tr in traces:
        if tr.latch.get("reground") is None or tr.feat_path is None:
            continue
        tru = S.truth(tr, K, pm)
        held = S.held_of(tr, "reground")
        q = np.load(tr.meta["perception"], allow_pickle=False)
        ph = np.argmax(probe.proba(np.asarray(cache.get(tr)["kv_l4"], np.float64)), 1)
        held2 = held | (np.isin(ph, HOLD_PHASES) & finger_evidence(q))
        w = tr.window
        a = acc.setdefault(tr.condition, {k: 0 for k in ("pg_pos", "missed_strong", "recovered", "false_strong",
                                                         "false_or", "pt_neg", "fp_task_strong", "fp_task_or")})
        pg, pt = tru["p_grasp"] & w, tru["p_task"]
        a["pg_pos"] += int(pg.sum())
        a["missed_strong"] += int((pg & ~held).sum())
        a["recovered"] += int((pg & ~held & held2).sum())
        a["false_strong"] += int((w & ~tru["p_grasp"] & held).sum())
        a["false_or"] += int((w & ~tru["p_grasp"] & held2).sum())
        a["pt_neg"] += int((w & ~pt).sum())
        a["fp_task_strong"] += int((w & ~pt & held).sum())
        a["fp_task_or"] += int((w & ~pt & held2).sum())
    ins, nat = acc["instruction"], acc["natural"]
    es = {k: acc.get("empty", {}).get(k, 0) + acc.get("slip", {}).get(k, 0) for k in ("false_strong", "false_or")}
    c1 = ins["recovered"] / ins["missed_strong"] if ins["missed_strong"] else float("nan")
    c2 = es["false_or"] - es["false_strong"]
    c3 = (nat["fp_task_or"] - nat["fp_task_strong"]) / nat["pt_neg"] if nat["pt_neg"] else float("nan")
    res = {"definitions": __doc__, "counts": acc,
           "criteria": {"1_recovered_frac": c1, "1_pass": bool(c1 >= 0.5),
                        "2_false_hold_increase_empty_slip": c2, "2_pass": bool(c2 == 0),
                        "3_natural_false_pass_increase": c3, "3_pass": bool(c3 <= 0.01)}}
    res["all_pass"] = bool(res["criteria"]["1_pass"] and res["criteria"]["2_pass"] and res["criteria"]["3_pass"])
    (out / "partC.json").write_text(json.dumps(res, indent=1, default=float))
    print(res["criteria"], "all_pass", res["all_pass"])
    print({k: v for k, v in acc.items()})


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
