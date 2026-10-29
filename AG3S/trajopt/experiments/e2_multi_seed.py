"""AG3S-PRE-CORE-TO E2 — offline settings for multi-seed TrajOpt (`tp_timing`) and the summary of a pass.

Settings (`tp_timing --settings-module AG3S.trajopt.experiments.e2_multi_seed`, each on top of the run's server config;
the E1 settings `E0` · `E1C` · `E1B10` · `E1B20` · `TOA` of `e1_goal` are included, so one module serves the whole
comparison):

| name | what |
|---|---|
| `E2_16` · `E2_32` · `E2_64` | E0 + `--to-multi-seed 16 / 32 / 64` (`sqp.multi_seed`) |
| `E1C_E2_16` · `E1C_E2_32` · `E1C_E2_64` | E1-C (`--grasp-anchor 2000 --grasp-anchor-path-weight 0`) + the same |
| `SMOKE_IT1_*` | smoke only (one SQP iteration, to make the standard solve leave violations); not for the comparison |

A multi-seed setting records per chunk `multi_seed` (the optimizer's record: triggered, the candidates with kind ·
max violation · goal error · status, the chosen one, total ms); `solve_ms` includes the seeds. The goal error used by the
selection exists only where the solve carries a grasp anchor (E1-C settings); on E0 settings the selection goes
violation → path length → accelerations, and `goal_eval` (probe gate, every setting) measures the goal error after.

    python -m AG3S.trajopt.experiments.e2_multi_seed summarize OUT/*.json

Summary (task §6 E2 / §7 E2, fixed before the results): over the chunks where **E0** leaves a violation > 0.1 mm (per
run, by request seq), per setting: the number resolved (max violation <= 0.1 mm) and resolved ∧ goal error <= 10 mm
(axis max, `goal_eval.result_err_axis_max_mm`, as `e1_goal summarize`; a chunk without a goal counts on the violation
alone), the Euclidean variant, the chosen seed kinds, and the time (solve ms on those chunks, multi-seed ms where it
triggered). `all_chunks` repeats the `e1_goal` table over every solved chunk.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from typing import Any

import numpy as np

from AG3S.trajopt.experiments import e1_goal as E1

#: Seed counts of task §6 E2.
SEED_COUNTS = (16, 32, 64)
#: Violation (mm) at or below which a chunk counts as resolved (= `safety.violation_tolerance`, 0.1 mm).
RESOLVED_MM = 0.1
#: Goal error bound r (mm) of the criterion "violation resolved ∧ goal error <= r".
GOAL_MM = 10.0

SETTINGS: dict[str, dict] = dict(E1.SETTINGS)
for _n in SEED_COUNTS:
    SETTINGS[f"E2_{_n}"] = {"sqp": {"multi_seed": _n}}
    SETTINGS[f"E1C_E2_{_n}"] = {**{k: dict(v) for k, v in E1.SETTINGS["E1C"].items()}, "sqp": {"multi_seed": _n}}

#: **Smoke only, not for the comparison** — the multi-seed path on a recorded run whose standard solves all resolve
#: (S-REG ep211): one SQP iteration (`SMOKE_IT1_*`) leaves violations for the seeds to work on.
SMOKE_SETTINGS: dict[str, dict] = {
    "SMOKE_IT1_E0": {"sqp": {"max_iterations": 1, "min_iterations": 1}},
    "SMOKE_IT1_E2_16": {"sqp": {"max_iterations": 1, "min_iterations": 1, "multi_seed": 16}},
    "SMOKE_IT1_E1C_E2_16": {**{k: dict(v) for k, v in E1.SETTINGS["E1C"].items()},
                            "sqp": {"max_iterations": 1, "min_iterations": 1, "multi_seed": 16}},
}
SETTINGS.update(SMOKE_SETTINGS)


def _pct(a, q) -> float | None:
    return round(float(np.percentile(a, q)), 1) if len(a) else None


def _med(a) -> float | None:
    return round(float(np.median(a)), 1) if len(a) else None


def summarize(paths: list[str], *, base: str = "E0") -> dict[str, Any]:
    """`{"criterion", "per_setting": {v: …}, "per_run": {key: {v: [n_base_violated, n_resolved_goal_ok]}},
    "all_chunks": e1_goal.summarize(paths)}` — see the module docstring. `base`: the setting whose violated chunks
    define the set (`E0`; `SMOKE_IT1_E0` for the smoke settings, CLI `summarize --base NAME`)."""
    acc: dict[str, dict[str, Any]] = {}
    per_run: dict[str, dict[str, list[int]]] = {}
    for path in paths:
        d = json.load(open(path))
        key = d.get("key", path)
        for ch in d.get("chunks", []):
            vs = ch.get("variants") or {}
            b = vs.get(base)
            if b is None or float(b.get("max_violation_mm") or 0.0) <= RESOLVED_MM:
                continue
            for v, e in vs.items():
                a = acc.setdefault(v, {"n": 0, "resolved": 0, "ok_axis": 0, "ok_euclid": 0, "with_goal": 0,
                                       "goal_ok_axis_of_goal": 0, "viol": [], "ms": [], "ms_seed": [],
                                       "triggered": 0, "chosen": Counter(), "chosen_resolved": Counter(),
                                       "status": Counter()})
                viol = float(e.get("max_violation_mm") or 0.0)
                ge = e.get("goal_eval")
                resolved = viol <= RESOLVED_MM
                ok_axis = resolved and (ge is None or float(ge["result_err_axis_max_mm"]) <= GOAL_MM)
                ok_euclid = resolved and (ge is None or float(ge["result_err_mm"]) <= GOAL_MM)
                a["n"] += 1
                a["resolved"] += int(resolved)
                a["ok_axis"] += int(ok_axis)
                a["ok_euclid"] += int(ok_euclid)
                if ge is not None:
                    a["with_goal"] += 1
                    a["goal_ok_axis_of_goal"] += int(ok_axis)
                a["viol"].append(viol)
                a["ms"].append(float(e.get("solve_ms") or 0.0))
                a["status"][str(e.get("status"))] += 1
                ms = e.get("multi_seed")
                if isinstance(ms, dict) and ms.get("triggered"):
                    a["triggered"] += 1
                    a["ms_seed"].append(float(ms.get("total_ms") or 0.0))
                    a["chosen"][str(ms.get("chosen_kind"))] += 1
                    if resolved:
                        a["chosen_resolved"][str(ms.get("chosen_kind"))] += 1
                r = per_run.setdefault(key, {}).setdefault(v, [0, 0])
                r[0] += 1
                r[1] += int(ok_axis)
    table: dict[str, Any] = {}
    for v, a in acc.items():
        n = a["n"]
        table[v] = {
            "n_base_violated": n,
            "n_resolved": a["resolved"],
            "n_resolved_goal_le_10mm": a["ok_axis"],
            "share_resolved_goal_le_10mm": round(a["ok_axis"] / n, 4) if n else None,
            "n_resolved_goal_le_10mm_euclid": a["ok_euclid"],
            "n_with_goal": a["with_goal"],
            "n_with_goal_resolved_goal_le_10mm": a["goal_ok_axis_of_goal"],
            "violation_mm_median": round(float(np.median(a["viol"])), 3) if n else None,
            "violation_mm_max": round(float(np.max(a["viol"])), 3) if n else None,
            "solve_ms_median": _med(a["ms"]), "solve_ms_p95": _pct(a["ms"], 95),
            "solve_ms_max": round(float(np.max(a["ms"])), 1) if n else None,
            "n_triggered": a["triggered"],
            "multi_seed_ms_median": _med(a["ms_seed"]), "multi_seed_ms_p95": _pct(a["ms_seed"], 95),
            "multi_seed_ms_max": round(float(np.max(a["ms_seed"])), 1) if a["ms_seed"] else None,
            "chosen_kind": dict(a["chosen"]),
            "chosen_kind_when_resolved": dict(a["chosen_resolved"]),
            "status": dict(a["status"]),
        }
    return {"criterion": (f"over chunks where {base} leaves max violation > {RESOLVED_MM} mm: resolved (<= "
                          f"{RESOLVED_MM} mm) and goal error (axis max, goal_eval) <= {GOAL_MM} mm when a goal exists"),
            "per_setting": table, "per_run": per_run, "all_chunks": E1.summarize(paths)}


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] != "summarize":
        raise SystemExit("usage: python -m AG3S.trajopt.experiments.e2_multi_seed summarize [--base NAME] OUT.json ...")
    rest, base = argv[1:], "E0"
    if rest[:1] == ["--base"]:
        base, rest = rest[1], rest[2:]
    print(json.dumps(summarize(rest, base=base), indent=1))


if __name__ == "__main__":
    sys.exit(main())
