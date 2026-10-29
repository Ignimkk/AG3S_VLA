"""T55b — why the strong rule loses HELD after an instruction change (re-measure wait #48), from the T55 records.

T55 test (`T55.verify.json` ``test.primary_reground.instruction.p_grasp.strong``): with the prompt's target noun
replaced by another fruit and the serving L8H2 grounding re-run on the re-simulated observations, the strong rule
(`grasp_latch` HELD) missed 117 of 333 frames in which the used hand physically holds a fruit (P_grasp). This
script traces those frames through the recorded latch evidence; it re-runs nothing.

Per frame (test E0, ``reground`` perception = `t55_perception` output, truth = `t55_study.truth` with the
pre-registered K and persistence):
    p_grasp / p_task     physics truth (P_task = the instructed noun's fruit is the one held)
    held                 latch state == HELD (the strong rule)
    latch_phase          latch state of that request
    failed_checks        the grasp-evidence checks that were False in that request's grasp record
                         (`grasp_json.evidence.checks`; empty when the latch never evaluated evidence)
    manipulated_is       which fruit AG3S's manipulated centroid (`target_centroid`) is nearest to, in the re-sim's
                         ``fruit_pos`` of that frame: ``noun`` (the instructed fruit) · ``held`` (the fruit in the
                         hand) · ``other`` · ``none`` (no centroid)
    reach_mm             the latch's reach distance (parent link origin → manipulated centroid), as recorded
    held_fruit_mm        MuJoCo distance parent link origin → the held fruit's centre (``parent_pos``,
                         ``fruit_pos``; simulator truth, diagnostic only — not something the latch can see)

Gate limitation: ``task_ready = HELD ∧ probe`` can only veto; a frame with P_task true and HELD false stays a false
reject whatever the probe says. Those counts come from the same truth and latch arrays, per condition.

usage: t55b_held_fn.py OUT_JSON      (.venv-ag3s, CPU, one process)
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t55_study as S  # noqa: E402

T55_DIR = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/development/"
                       "20261013T171325+0900_cf38c9dbca7b_2ebfd8a5a43c/verify/T55")
CONDS = ("natural", "instruction", "empty", "slip")


def manipulated_is(centroid: np.ndarray, fruit_pos: np.ndarray, fruits: list, noun: str, held: list) -> str:
    c = np.asarray(centroid, float)
    if c.shape != (3,) or not np.all(np.isfinite(c)):
        return "none"
    near = fruits[int(np.argmin(np.linalg.norm(fruit_pos - c, axis=1)))]
    if near == noun:
        return "noun"
    return "held" if near in held else "other"


def frame_rows(tr, tru: dict) -> list:
    q = np.load(tr.meta["perception"], allow_pickle=False)
    z = np.load(tr.meta["resim"], allow_pickle=False)
    fruits = [str(f) for f in z["fruits"]]
    fp, parent = np.asarray(z["fruit_pos"], float), np.asarray(z["parent_pos"], float)
    held = S.held_of(tr, "reground")
    rows = []
    for i in np.where(tr.window & tru["p_grasp"])[0]:
        gj = str(q["grasp_json"][i])
        g = json.loads(gj) if gj else {}
        ev = g.get("evidence") or {}
        held_fruits = [f for f in fruits if tr.held_by[f][i]]
        hf_mm = (None if not held_fruits else
                 float(np.linalg.norm(fp[i, fruits.index(held_fruits[0])] - parent[i]) * 1000.0))
        rows.append({
            "frame": int(i), "held": bool(held[i]), "p_task": bool(tru["p_task"][i]),
            "latch_phase": str(q["latch_reground__phase"][i]),
            "failed_checks": sorted(k for k, v in (ev.get("checks") or {}).items() if v is False),
            "manipulated_is": manipulated_is(q["target_centroid"][i], fp[i], fruits, tr.noun, held_fruits),
            "reach_mm": ev.get("reach_mm"), "reach_limit_mm": ev.get("reach_limit_mm"),
            "held_fruit_mm": hf_mm, "held_fruit": held_fruits[0] if held_fruits else None,
        })
    return rows


def summarize(rows: list) -> dict:
    fn = [r for r in rows if not r["held"]]
    tp = [r for r in rows if r["held"]]

    def _q(xs):
        xs = [x for x in xs if x is not None]
        return None if not xs else {"n": len(xs), "min": float(np.min(xs)), "median": float(np.median(xs)),
                                    "max": float(np.max(xs))}

    by_cause = collections.Counter(
        f"{r['latch_phase']}|{'+'.join(r['failed_checks']) or '-'}|manipulated={r['manipulated_is']}" for r in fn)
    return {
        "n_p_grasp": len(rows), "n_held": len(tp), "n_held_false": len(fn),
        "held_false_by_cause": dict(by_cause.most_common()),
        "held_false_manipulated_is": dict(collections.Counter(r["manipulated_is"] for r in fn)),
        "held_true_manipulated_is": dict(collections.Counter(r["manipulated_is"] for r in tp)),
        "held_false_reach_mm": _q([r["reach_mm"] for r in fn]),
        "held_false_held_fruit_mm": _q([r["held_fruit_mm"] for r in fn]),
        "held_true_reach_mm": _q([r["reach_mm"] for r in tp]),
        "reach_limit_mm": sorted({r["reach_limit_mm"] for r in rows if r["reach_limit_mm"] is not None}),
        "held_false_with_p_task": sum(r["p_task"] for r in fn),
    }


def main(out_json: pathlib.Path) -> None:
    prereg = json.loads((T55_DIR / "T55.prereg.json").read_text())
    K = int(prereg["truth"]["K"])
    pm = prereg["truth"].get("persistence", "causal")
    traces = S.load_resim_traces(T55_DIR / "resim/test", None, "test",
                                 perception_dir=T55_DIR / "perception/test/t55-legacy")
    out = {"inputs": {"t55_dir": str(T55_DIR), "K": K, "persistence": pm, "latch_mode": "reground"},
           "definitions": __doc__, "by_condition": {}, "gate_limit": {}, "per_trace": []}
    rows_by = collections.defaultdict(list)
    for tr in traces:
        if tr.latch.get("reground") is None:
            continue
        tru = S.truth(tr, K, pm)
        held = S.held_of(tr, "reground")
        w = tr.window
        # gate limitation: P_task frames the strong rule does not call HELD (no probe can recover them)
        gl = out["gate_limit"].setdefault(tr.condition, {"n_p_task": 0, "n_p_task_held_false": 0})
        gl["n_p_task"] += int((w & tru["p_task"]).sum())
        gl["n_p_task_held_false"] += int((w & tru["p_task"] & ~held).sum())
        if tr.condition not in ("natural", "instruction"):
            continue
        rows = frame_rows(tr, tru)
        rows_by[tr.condition] += rows
        s = summarize(rows)
        # the attach decision: the manipulated object at the first HELD request (its centroid is the one the
        # reach check passed; later HELD frames carry a frozen pre-grasp centroid and are not diagnostic)
        if held.any():
            a = int(np.argmax(held))
            q = np.load(tr.meta["perception"], allow_pickle=False)
            z = np.load(tr.meta["resim"], allow_pickle=False)
            fruits = [str(f) for f in z["fruits"]]
            ever_held = [f for f in fruits if any(tr.held_by[f][a:])]
            label = manipulated_is(q["target_centroid"][a], np.asarray(z["fruit_pos"][a], float), fruits,
                                   tr.noun, ever_held)
        else:
            label = "never_held" if s["n_p_grasp"] else "no_p_grasp"
        out.setdefault("attach_manipulated_is", {}).setdefault(tr.condition, collections.Counter())[label] += 1
        out["per_trace"].append({"key": tr.key, "prompt_tag": tr.prompt_tag, "condition": tr.condition,
                                 "noun": tr.noun, "n_p_grasp": s["n_p_grasp"], "n_held": s["n_held"],
                                 "n_held_false": s["n_held_false"],
                                 "held_false_manipulated_is": s["held_false_manipulated_is"],
                                 "held_true_manipulated_is": s["held_true_manipulated_is"]})
    for c, rows in rows_by.items():
        out["by_condition"][c] = summarize(rows)
    # the published strong-rule counts must be reproduced before anything is read from this file
    pub = json.loads((ROOT / "AG3S/ag3s/docs/handoff/T55.verify.json").read_text())["test"]["primary_reground"]
    for c in ("natural", "instruction"):
        st = pub[c]["p_grasp"]["strong"]
        assert out["by_condition"][c]["n_p_grasp"] == st["n_pos"], (c, out["by_condition"][c]["n_p_grasp"], st["n_pos"])
        assert out["by_condition"][c]["n_held_false"] == st["false_reject"], c
    for c, gl in out["gate_limit"].items():
        st = pub[c]["p_task"]["strong"]
        assert gl["n_p_task"] == st["n_pos"] and gl["n_p_task_held_false"] == st["false_reject"], (c, gl, st)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1, default=float))
    for c, s in out["by_condition"].items():
        print(c, s["n_p_grasp"], s["n_held_false"], s["held_false_by_cause"], s["held_false_reach_mm"],
              s["held_false_held_fruit_mm"])
    print("gate_limit", out["gate_limit"])


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
