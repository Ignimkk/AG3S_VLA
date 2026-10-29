"""T56 — add the test-stage numbers to handoff/T56.verify.json (numbers only; the dev block is kept as is).

usage: t56_build_verify_test.py T56_RUN_DIR
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = ROOT / "AG3S/ag3s/docs/handoff/T56.verify.json"
GROUPS = ("cells", "cells_action", "cells_noun", "arms")
VARIANTS = ("S0", "S1", "S2")
KEEP = ("selected", "tie_size", "n_event_episodes", "all:identity:selected", "all:identity:selected:ci95",
        "all:identity:best", "all:identity:regret", "all:identity:regret:ci95", "all:identity:tie_regret",
        "all:identity:tie_regret:ci95", "all:purity:regret", "all:coverage:regret", "all:no_target:selected",
        "tau_b:identity")


def main(run: pathlib.Path) -> None:
    v = run / "verify/T56"
    s = json.loads((v / "test/analysis/t56_summary.json").read_text())
    status = (v / "test_status.txt").read_text().splitlines()
    sel = {}
    for cond in ("current", "hold"):
        for g in GROUPS:
            for var in VARIANTS:
                e = s["results"][cond][g]["all"][var]
                sel.setdefault(cond, {}).setdefault(g, {})[var] = {k: e.get(k) for k in KEEP}
            sel[cond][g]["paired"] = s["results"][cond][g]["all"]["paired"]
    lc = s["label_checks"]["current"]
    out = json.loads(OUT.read_text())
    out["test"] = {
        "status": "test_done — mechanical criteria only, no verdict",
        "summary": str(v / "test/analysis/t56_summary.json"),
        "prereg_sha256": s["prereg_sha256"],
        "run_history": [ln for ln in status if ln.startswith(("test run2", "inputs", "replay", "held", "score",
                                                               "TEST DONE"))],
        "worker_change": {"stopped_utc": "2026-10-14T14:33:17+00:00", "resumed_utc": "2026-10-14T14:33:47+00:00",
                          "workers": "4 -> 2 (user decision 2026-10-14, pod memory for session A)",
                          "done_before_stop": 55, "done_md5_unchanged_after_resume": True,
                          "lists": [str(v / "w2_done_before_stop.txt"), str(v / "w2_done_md5_before_stop.txt")],
                          "driver": str(v / "run_test_w2.sh")},
        "code_md5": "md5_test_start_run2 == md5_test_before_score (serving + t54/t56 files)",
        "numbers": {
            "criteria_mechanical": s["criteria_mechanical"],
            "selection": sel,
            "held_label_checks": {k: lc[k] for k in ("n_held_events", "n_label_wrong", "held_label_error_rate",
                                                     "wrong_records", "verdict_counts", "records_without_held")},
        },
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float) + "\n")
    print("->", OUT)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
