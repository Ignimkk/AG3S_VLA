#!/usr/bin/env python3
"""SL-GTTO11 기제 시험 (G54 규칙, 결과 전 고정): F9 (`gt12qs`) vs F8 (`gt11hdp`) on every finished run of the two arms.

(i-a) F7 action-QP solves (`to.ee_action_qp.status`) that stop on OSQP's iteration cap: F9 ≤ 10 %
(i-c') attaches with the fingers still open (`grasp.opening_rule.decision == attach` ∧ opening > 0.9): F9 = 0
Also: QP iterations per solve, unreachable rows per solve (F9), |Δa| of the QP calls.
usage: python3 experiments/segmentation/gtto11_mech.py --out outputs/2026-11-07_F9_스케일QP_닫힘확인/기제시험
"""
import argparse
import collections
import json
import pathlib

import numpy as np

RESULTS = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
ARMS = {"F8": "gt11hdp", "F9": "gt12qs"}


def stats(tag):
    s = {"solves": 0, "cap": 0, "iters": [], "unreach": [], "qp_calls": 0, "dmax": [], "attach": 0, "open_attach": 0,
         "runs": [], "status": collections.Counter()}
    for d in sorted(RESULTS.glob(f"*_gtto-{tag}-*")):
        if not (d / "episodes.jsonl").is_file():
            continue
        s["runs"].append(d.name)
        for f in sorted((d / "episodes").glob("*_calls.jsonl")):
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                q = (r.get("to") or {}).get("ee_action_qp") if isinstance(r.get("to"), dict) else None
                if q:
                    s["qp_calls"] += 1
                    s["dmax"].append(float(q.get("max_abs_delta_action") or 0.0))
                    for st in q.get("status") or []:
                        s["status"][st] += 1
                        s["solves"] += 1
                        s["cap"] += st == "maximum iterations reached"
                    s["iters"] += [int(x) for x in q.get("osqp_iter") or []]
                    s["unreach"] += [int(x) for x in q.get("n_rows_unreachable") or []]
                o = (r.get("grasp") or {}).get("opening_rule") or {}
                if o.get("decision") == "attach":
                    s["attach"] += 1
                    s["open_attach"] += float(o.get("opening") or 0.0) > 0.9
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    S = {k: stats(t) for k, t in ARMS.items()}
    m = {}
    for k, s in S.items():
        m[k] = {"runs": len(s["runs"]), "qp_calls": s["qp_calls"], "solves": s["solves"],
                "cap_frac": s["cap"] / s["solves"] if s["solves"] else None,
                "iter_median": float(np.median(s["iters"])) if s["iters"] else None,
                "unreachable_rows_median": float(np.median(s["unreach"])) if s["unreach"] else None,
                "dmax_median": float(np.median(s["dmax"])) if s["dmax"] else None,
                "attach": s["attach"], "open_attach": s["open_attach"], "status": dict(s["status"])}
    verdict = {"(i-a) F9 cap_frac <= 0.10": m["F9"]["cap_frac"] is not None and m["F9"]["cap_frac"] <= 0.10,
               "(i-c') F9 open attaches == 0": m["F9"]["runs"] > 0 and m["F9"]["open_attach"] == 0}
    (a.out / "mech11.json").write_text(json.dumps({"metrics": m, "verdict": verdict, "runs": {k: S[k]["runs"] for k in S}}, indent=1) + "\n")
    f = lambda x: "–" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))
    L = ["# F9 기제 시험 (G54) — F8 vs F9, 두 판의 끝난 run 전부", "", "| 지표 | F8 | F9 | 규칙 |", "|---|---|---|---|"]
    for key, lab, rule in (("runs", "run 수", "–"), ("qp_calls", "action QP 가 돈 call", "–"), ("solves", "OSQP 풀이 수", "–"),
                           ("cap_frac", "(i-a) 반복 상한 도달 비율", "F9 ≤ 0.10"), ("iter_median", "풀이당 반복 중앙값", "–"),
                           ("unreachable_rows_median", "풀이당 action 이 닿지 않는 행 (F9)", "–"),
                           ("dmax_median", "QP call 의 max |Δa| 중앙값", "–"), ("attach", "opening rule attach", "–"),
                           ("open_attach", "(i-c') 개도 > 0.9 에서 attach", "F9 = 0")):
        L.append(f"| {lab} | {f(m['F8'][key])} | {f(m['F9'][key])} | {rule} |")
    L += ["", "판정: " + " · ".join(f"{k} → {'통과' if v else '미달'}" for k, v in verdict.items())]
    (a.out / "기제시험표.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for i, k in enumerate(S):
        if S[k]["iters"]:
            ax[0].hist(np.log10(np.maximum(S[k]["iters"], 1)), bins=30, alpha=0.6, label=f"{k} (cap {f(m[k]['cap_frac'])})")
    ax[0].set_xlabel("log10 OSQP iterations per solve (cap = 20000 -> 4.3)")
    ax[0].legend()
    ax[0].set_title("Action-QP convergence")
    ax[1].bar(range(2), [m[k]["open_attach"] for k in S], tick_label=list(S), color=["#d95f02", "#1b9e77"])
    ax[1].set_title("Attaches with fingers still open (> 0.9)")
    fig.tight_layout()
    fig.savefig(a.out / "그래프_기제시험.png", dpi=120)
    print(json.dumps({"metrics": m, "verdict": verdict}))


if __name__ == "__main__":
    main()
