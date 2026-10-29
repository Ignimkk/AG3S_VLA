"""Collect the numbers of T55b (#48) and the T55 · T54c P · H add-ons into their verify.json (numbers only).

usage: t55b_build_verify.py RUN_DIR
    RUN_DIR/verify/T55b/t55b_held_fn.json          -> handoff/T55b.verify.json
    RUN_DIR/verify/T55_windows/t55_windows.json    -> handoff/T55_windows.verify.json
    RUN_DIR/verify/T54c_windows/{grounding,closed_loop}.json, step1/timestamps.json
                                                   -> handoff/T54c_windows.verify.json
"""

from __future__ import annotations

import datetime as _dt
import filecmp
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
HANDOFF = ROOT / "AG3S/ag3s/docs/handoff"
T55W_FIRST = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/"
                          "20261014T193322+0900_284366d2bbec_e3469147913d/verify/T55_windows")


def _head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def _write(name: str, obj: dict) -> None:
    (HANDOFF / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float) + "\n")
    print("->", HANDOFF / name)


def _base(step: str, run: pathlib.Path, sub: str) -> dict:
    return {"_comment": "numbers and paths only; no interpretation. Rates are fractions unless *_pp.",
            "step": step, "date": _dt.date.today().isoformat(), "git_head": _head(),
            "run_dir": str(run), "verify_dir": str(run / "verify" / sub)}


def t55b(run: pathlib.Path) -> None:
    d = json.loads((run / "verify/T55b/t55b_held_fn.json").read_text())
    out = _base("T55b", run, "T55b")
    out.update({
        "status": "done",
        "inputs": d["inputs"],
        "reproduces_published": "n_p_grasp · false_reject (natural, instruction) and P_task n_pos · false_reject per "
                                "condition equal T55.verify.json test.primary_reground (asserted in t55b_held_fn)",
        "numbers": {
            "by_condition": d["by_condition"],
            "attach_manipulated_is": d["attach_manipulated_is"],
            "gate_limit_p_task_held_false": d["gate_limit"],
            "per_trace": d["per_trace"],
        },
        "not_measured": [
            "whether the attached geometry of the 5 instruction traces that attached with manipulated = noun fruit "
            "is the held fruit",
            "a latch variant that measures reach to the object between the fingers (no re-run)",
            "SL5b (occluded stale centroid) — A's T46g",
        ],
    })
    _write("T55b.verify.json", out)


def t55_windows(run: pathlib.Path) -> None:
    v = run / "verify/T55_windows"
    d = json.loads((v / "t55_windows.json").read_text())
    same = {f: filecmp.cmp(T55W_FIRST / f, v / f, shallow=False)
            for f in ("decision.csv", "side_by_side.csv", "scores_full.json", "scores_P.json", "scores_H.json")}
    out = _base("T55_windows", run, "T55_windows")
    out.update({
        "status": "done (independent re-run in a new run dir)",
        "first_run_dir": str(T55W_FIRST),
        "rerun_outputs_identical_to_first_run": same,
        "parity_full_vs_test_results": d["parity"],
        "inputs_unchanged": d["inputs_unchanged"],
        "seconds": d["seconds"],
        "numbers": {"endpoints": d["endpoints"], "frames_removed": d["frames_removed"],
                    "scored_frames": d["scored_frames"], "headline": d["headline"],
                    "natural_events": d["natural_events"]},
    })
    _write("T55_windows.verify.json", out)


def t54c_windows(run: pathlib.Path) -> None:
    v = run / "verify/T54c_windows"
    g = json.loads((v / "grounding.json").read_text())
    cl = json.loads((v / "closed_loop.json").read_text())["sets"]["T54c"]
    ts = json.loads((v / "step1/timestamps.json").read_text())
    runs = {k: {"t_P": r["t_P"], "t_H": r["t_H"], "frames_removed_P": r["frames_removed_P"],
                "frames_removed_H": r["frames_removed_H"], "success": r["recorded"]["success"]}
            for k, r in ts["runs"].items()}
    grounding = {}
    for w, wv in g["windows"].items():
        grounding[w] = {"n_frame_rows": wv["n_frame_rows"], "t57_entry_all_numeric_rows_pass":
                        {a: e["all_numeric_rows_pass"] for a, e in wv["t57_entry_by_arm"].items()},
                        "t57_entry_by_arm": wv["t57_entry_by_arm"], "summary": wv["summary"]}
    out = _base("T54c_windows", run, "T54c_windows")
    out.update({
        "status": "done",
        "full_reproduced": "closed loop: run flags and arm totals equal T54c agg.json (t60_closed_loop asserts); "
                           "grounding: current · hold_attach means of identity · no_target · purity · coverage for "
                           "all 8 arms within 1e-6 of the published t54_summary.json, T57 entry pass flags equal",
        "params": ts["primary"],
        "numbers": {
            "endpoints": {"n_runs": len(runs), "P_reached": sum(r["t_P"] is not None for r in runs.values()),
                          "H_reached": sum(r["t_H"] is not None for r in runs.values()),
                          "frames_removed_P": sum(r["frames_removed_P"] for r in runs.values()),
                          "frames_removed_H": sum(r["frames_removed_H"] for r in runs.values()),
                          "per_run": runs},
            "closed_loop_C1a": cl["arms"]["C1a"],
            "grounding": grounding,
        },
    })
    _write("T54c_windows.verify.json", out)


if __name__ == "__main__":
    r = pathlib.Path(sys.argv[1])
    t55b(r)
    t55_windows(r)
    t54c_windows(r)
