"""T54c P · H add-on — the T54c closed-loop record and offline grounding replay under the T60 windows.

User decision 2026-10-14: experiments fixed before T60 (T54c · T55 · T56) keep their primary verdict and report
the P (placement complete) / H (ready-pose return complete) windows next to it as an auxiliary analysis. T54c
is the T57 baseline arm, so both parts are computed here, with the existing tools and records only:

1. endpoints   `t60_timestamps.main` on the T54c replay npz (`replay_contacts.py` output) — t_P · t_H per run
               and the inference-frame masks ``keep_frames_P`` / ``keep_frames_H`` (T60 primary parameters).
2. closed loop `t60_closed_loop.main` on the T54c C1a record — success · success_ct · HOLD · completion time
               per window. Its full window is asserted against the published T54c ``agg.json``.
3. grounding   the published T54c frame rows (`t54_frames.jsonl`) re-analysed with `t54_stats.analyze` (same
               arms · conditions · bootstrap seed · block · min_target_px as the published run) on the full,
               P and H frame sets; the T57 entry table per window for N · B · A. The full window is asserted
               against the published `t54_summary.json` means (1e-6) before any P / H number is written.

Failed episodes (P or H not reached) keep every frame (T60 rule). Nothing is re-simulated or re-extracted.
The T60 modules are reused unchanged; their set table is redirected to T54c inside this process only.

usage: t54c_windows.py OUT_DIR      (.venv-ag3s, CPU, one process)
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t54_stats as stats  # noqa: E402
from AG3S.ag3s.experiments.studies import t60_closed_loop as CL  # noqa: E402
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402
from AG3S.ag3s.experiments.studies import t60_timestamps as TS  # noqa: E402

HANDOFF = ROOT / "AG3S/ag3s/docs/handoff"
SET = "T54c"
WINDOWS = ("full", "P", "H")
SUMMARY_METRICS = ("identity", "no_target", "purity", "coverage", "neighbour")
ASSERT_METRICS = ("identity", "no_target", "purity", "coverage")


def t54c_paths() -> dict:
    v = json.loads((HANDOFF / "T54c.verify.json").read_text())
    base = pathlib.Path(v["verify_dir"])
    ra = v["replay_analysis"]
    return {"verify": v, "base": base, "summary": pathlib.Path(ra["summary_file"]),
            "frames": pathlib.Path(ra["frames_file"])}


def redirect_sets(base: pathlib.Path) -> None:
    """Point the T60 set table at T54c (this process only)."""
    TS.SETS.clear()
    TS.SETS[SET] = base
    CL.SETS = TS.SETS
    CL.PAIRS = {SET: []}


def load_masks(step1: pathlib.Path) -> dict:
    import numpy as np

    masks = {}
    for p in sorted((step1 / "masks").glob(f"{SET}__C1a_*.npz")):
        key = p.stem.split("__", 1)[1].split("_")[1]  # C1a_ep1806_s18061 -> ep1806
        with np.load(p) as z:
            masks[key] = {"P": z["keep_frames_P"].astype(bool), "H": z["keep_frames_H"].astype(bool),
                          "t_P": int(z["t_P"]), "t_H": int(z["t_H"]),
                          "chunk_start_steps": z["chunk_start_steps"].tolist()}
    return masks


def grounding_windows(paths: dict, masks: dict) -> dict:
    pub = json.loads(paths["summary"].read_text())
    args = pub["args"]
    keys, arms = pub["summary"]["keys"], pub["summary"]["arms"]
    conds = list(args["conditions"])
    hold_c = pub["hold_condition"]
    frames = [json.loads(line) for line in paths["frames"].read_text().splitlines() if line.strip()]
    for k in keys:  # frame index ↔ inference step must agree with the replay mask
        steps = sorted({f["t_step"] for f in frames if f["key"] == k})
        assert steps == masks[k]["chunk_start_steps"], k
    sel = {"full": frames,
           "P": [f for f in frames if masks[f["key"]]["P"][f["frame"]]],
           "H": [f for f in frames if masks[f["key"]]["H"][f["frame"]]]}
    out = {"keys": keys, "arms": arms, "conditions": conds, "hold_condition": hold_c,
           "bootstrap": {"n_boot": args["n_boot"], "seed": args["seed"], "block": args["block"]},
           "windows": {}}
    for w in WINDOWS:
        res = stats.analyze(sel[w], keys=keys, arms=arms, conditions=conds, base="L8H2",
                            n_boot=args["n_boot"], seed=args["seed"], block=args["block"],
                            phases=stats.GRASP_STATES, extract_ms=pub.get("extract_ms"),
                            extract_kinds=pub.get("extract_kinds"), log=lambda s: None)
        if w == "full":
            for cond in conds:
                for metric in ASSERT_METRICS:
                    for a in arms:
                        exp = pub["summary"][cond]["all"][metric][a]["mean"]
                        got = res[cond]["all"][metric][a]["mean"]
                        assert (exp is None and got is None) or abs(got - exp) < 1e-6, (cond, metric, a, got, exp)
        entry = {a: stats.entry_criteria(res, arm=a, base="L8H2", budget_ms=args.get("budget_ms"),
                                         hold_condition=hold_c, direction_positive=True)
                 for a in args.get("entry_arms") or ("N", "B", "A")}
        if w == "full":
            for a, e in entry.items():
                pe = pub["t57_entry_by_arm"][a]
                assert e["all_numeric_rows_pass"] == pe["all_numeric_rows_pass"], a
                assert abs(e["rows"]["identity"]["value"] - pe["rows"]["identity"]["value"]) < 1e-6, a
        slim = {}
        for cond in conds:
            slim[cond] = {"n_target_frames": res[cond]["n_target_frames"], "metrics": {}, "phase_identity": {}}
            for metric in SUMMARY_METRICS:
                m = {}
                for a in arms:
                    cell = res[cond]["all"][metric][a]
                    row = {"mean": cell["mean"], "n_frames": cell["n_frames"]}
                    if a != "L8H2":
                        d = cell["diff_vs_L8H2"]
                        mb = res[cond]["moving_block"][metric].get(a, {})
                        row.update({"diff_vs_L8H2": d["mean"],
                                    "diff_ci95_episode_bootstrap": d["ci95_episode_bootstrap"],
                                    "diff_ci95_moving_block": mb.get("ci95"),
                                    "n_episodes_up_down": [d["n_episodes_up"], d["n_episodes_down"]]})
                    m[a] = row
                slim[cond]["metrics"][metric] = m
            for ph in stats.GRASP_STATES:
                slim[cond]["phase_identity"][ph] = {
                    a: {"mean": res[cond]["phase"][ph]["identity"][a]["mean"],
                        "n_frames": res[cond]["phase"][ph]["identity"][a]["n_frames"]} for a in arms}
        out["windows"][w] = {"n_frame_rows": len(sel[w]), "summary": slim, "t57_entry_by_arm": entry}
        print(w, {c: slim[c]["n_target_frames"] for c in conds},
              {a: round(slim["current"]["metrics"]["identity"][a]["mean"], 4) for a in ("L8H2", "N", "B", "A")},
              {a: e["all_numeric_rows_pass"] for a, e in entry.items()})
    return out


def main(out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = t54c_paths()
    redirect_sets(paths["base"])
    step1 = out_dir / "step1"
    TS.main(step1)
    CL.main(step1, out_dir / "closed_loop.json")
    masks = load_masks(step1)
    assert len(masks) == 16, len(masks)
    T.write_json(out_dir / "grounding.json", grounding_windows(paths, masks))
    print("->", out_dir)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
