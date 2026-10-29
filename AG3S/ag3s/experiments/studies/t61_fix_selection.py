"""T61 — freeze the calibration selection before the held-out is opened → handoff/T61.selection.json.

`handoff/T61.task.md` "고정 시점": the selection result is saved with its sha256 **before** any held-out number is
read. This reads only the calibration selection outputs (`t61_select` chain) and writes the configuration, the
one-SE evidence, the report-only side results and the sha256 of every selection input file. It opens nothing of
the held-out (no rows, no replay, no rollout metrics).

e4 → B hand-over note: e4 ended without writing T61.verify.json, so there is no earlier digest to compare against.
The digests below are taken at freeze time; ``chain_select.log`` shows the chain finished before it.

usage: t61_fix_selection.py RUN61_DIR
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
OUT = ROOT / "AG3S/ag3s/docs/handoff/T61.selection.json"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(run: pathlib.Path) -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists — the selection is frozen once")
    v = run / "verify/T61"
    sel = v / "select/combined"
    ksel = json.loads((sel / "gt/ksel/ksel__scene.json").read_text())
    kself = json.loads((sel / "self/ksel/ksel__scene.json").read_text())
    boot = json.loads((sel / "gt/bootstrap/bootstrap__scene.json").read_text())
    chain = (v / "logs/chain_select.log").read_text().splitlines()
    assert any(line.endswith("chain done") for line in chain), "selection chain not finished"
    assert ksel["records"] == ["t2", "ep1807", "ep0", "ep500", "ep1200"], ksel["records"]
    assert ksel["label"] == "gt" and ksel["aggregate"] == "scene"
    files = sorted(p for p in sel.rglob("*.json")) + sorted(sel.rglob("*.npz"))
    selected = ksel["selected"]
    out = {
        "step": "T61", "frozen_kst": _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(),
        "frozen_by": "session B (ag3s-vla-development-c4) — before any held-out number was read",
        "git_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "run_dir": str(run), "prereg": "AG3S/ag3s/docs/handoff/T61.task.md (commit 95b6586c)",
        "selected": selected,
        "serving_unit": {"query": "action", "denoise": 0, "aggregation": selected["combo"].split(":")[0],
                         "cells": selected["cells"], "k": selected["k"],
                         "note": "selection unit = query · denoise · aggregation · layer · head (판정 3-4); "
                                 "serving must implement this aggregation, not substitute `last`"},
        "rule_evidence": {"k_choice": ksel["choice"], "cv_scores_per_scene": ksel["cv"]["scores"],
                          "cv_combos_per_fold": ksel["cv"]["combos"], "nested_mean": ksel["nested_mean"],
                          "cv_mean_ci95": ksel["cv_mean_ci95"], "units": ksel["units"],
                          "calibration_top10": ksel["ranking_all_calibration_top10"]},
        "report_only": {
            "self_label_selection": kself["selected"],
            "bootstrap_gt": {k: boot[k] for k in ("n_boot", "seed", "full_sample_top1", "full_sample_top1_p_rank1",
                                                   "rank1_tied_rate", "top10_by_p_rank1")},
            "bootstrap_deviation": "T61.task.md fixed the report-only scene bootstrap at B 10000 · seed 6101; the "
                                   "chain ran t51's default (B 1000 · seed 5103). Report-only, not used by the rule",
            "query_wise_best": "not run by the chain (action-only / noun-only) — reported as missing",
        },
        "selection_chain_log": chain,
        "selection_files_sha256": {str(p.relative_to(v)): sha(p) for p in files},
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("selected", selected, "->", OUT, "sha256", sha(OUT))


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
