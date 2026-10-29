"""T60 step 2b — T54 (current / hold, 8 fixed arms) re-evaluated on the P and H frame windows.

The T54 rows, GT cache, arms, bootstrap (B = 10000, seed 5407) and `t54_stats.analyze` / `entry_criteria`
are reused unchanged; only frames after completion are removed with the T60 masks
(`keep_frames_P` / `keep_frames_H`: inference frames with t_step ≤ t_end). Failed episodes keep every frame.
The full window must reproduce the published T54 means (1e-6) before any P / H number is written.

usage: t60_t54.py STEP1_DIR OUT_JSON
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t54_stats as stats  # noqa: E402
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402

HANDOFF = ROOT / "AG3S/ag3s/docs/handoff"
CONDS = ("current", "hold")
SUMMARY_METRICS = ("identity", "no_target", "purity", "coverage", "neighbour")


def load(step1: pathlib.Path):
    v53 = json.loads((HANDOFF / "T53.verify.json").read_text())
    v54 = json.loads((HANDOFF / "T54.verify.json").read_text())
    gt_dir = pathlib.Path(v53["extraction"]["paths"]["gt"])
    rows_dir = pathlib.Path(v54["run_dir"]) / "verify/T54/test/rows"
    keys, arms = v54["numbers"]["keys"], v54["numbers"]["arms"]
    gt, grasp, rows, masks = {}, {}, {}, {}
    for key in keys:
        with np.load(gt_dir / f"{key}.npz", allow_pickle=True) as z:
            gt[key] = {k: z[k] for k in ("t_step", "class_px")}
        with np.load(gt_dir / f"{key}.grasp.npz", allow_pickle=True) as z:
            grasp[key] = {k: z[k] for k in ("t_step", "grasp_state")}
        m = sorted((step1 / "masks").glob(f"T53__C1a_{key}_s*.npz"))
        assert len(m) == 1, (key, m)
        with np.load(m[0]) as z:
            assert z["chunk_start_steps"].tolist() == gt[key]["t_step"].tolist(), key
            masks[key] = {w: z[f"keep_frames_{w}"].astype(bool) for w in ("P", "H")}
            masks[key]["t_P"], masks[key]["t_H"] = int(z["t_P"]), int(z["t_H"])
        for cond in CONDS:
            for line in (rows_dir / f"{key}__{cond}.jsonl").read_text().splitlines():
                r = json.loads(line)
                rows.setdefault((key, cond, r["arm"]), []).append(r)
    return v54, keys, arms, gt, grasp, rows, masks


def main(step1: pathlib.Path, out_json: pathlib.Path) -> None:
    v54, keys, arms, gt, grasp, rows, masks = load(step1)
    frames = stats.build_frames(rows, gt, grasp)
    windows = {"full": frames,
               "P": [f for f in frames if masks[f["key"]]["P"][f["frame"]]],
               "H": [f for f in frames if masks[f["key"]]["H"][f["frame"]]]}
    out = {"protocol": T.PROTOCOL_VERSION, "definitions": __doc__, "keys": keys, "arms": arms,
           "t53_endpoints": {k: {"t_P": None if m["t_P"] < 0 else m["t_P"], "t_H": None if m["t_H"] < 0 else m["t_H"],
                                 "frames_kept_P": int(m["P"].sum()), "frames_kept_H": int(m["H"].sum())}
                             for k, m in masks.items()},
           "windows": {}}
    for w, fs in windows.items():
        res = stats.analyze(fs, keys=keys, arms=arms, conditions=CONDS, log=lambda s: None)
        if w == "full":
            for cond in CONDS:
                for metric in ("identity", "no_target", "purity", "coverage"):
                    for a in arms:
                        exp = v54["numbers"][cond]["metrics"][metric][a]["mean"]
                        got = res[cond]["all"][metric][a]["mean"]
                        assert abs(got - exp) < 1e-6, (cond, metric, a, got, exp)
        entry = stats.entry_criteria(res, arm="A", base="L8H2", hold_condition="hold")
        if w == "full":
            pub = v54["numbers"]["t57_entry"]["from_t54_stats_entry_criteria"]
            assert entry["all_numeric_rows_pass"] == pub["all_numeric_rows_pass"]
            assert abs(entry["rows"]["identity"]["value"] - pub["rows"]["identity"]["value"]) < 1e-6
        slim = {}
        for cond in CONDS:
            slim[cond] = {"n_target_frames": res[cond]["n_target_frames"], "metrics": {}, "phase_identity": {}}
            for metric in SUMMARY_METRICS:
                slim[cond]["metrics"][metric] = {
                    a: {"mean": res[cond]["all"][metric][a]["mean"],
                        "n_frames": res[cond]["all"][metric][a]["n_frames"],
                        **({"diff_vs_L8H2": res[cond]["all"][metric][a]["diff_vs_L8H2"]["mean"],
                            "diff_ci95_episode_bootstrap": res[cond]["all"][metric][a]["diff_vs_L8H2"]["ci95_episode_bootstrap"],
                            "diff_ci95_moving_block": res[cond]["moving_block"][metric][a]["ci95"]
                            if "ci95" in res[cond]["moving_block"][metric].get(a, {}) else None,
                            "n_episodes_up_down": [res[cond]["all"][metric][a]["diff_vs_L8H2"]["n_episodes_up"],
                                                   res[cond]["all"][metric][a]["diff_vs_L8H2"]["n_episodes_down"]]}
                           if a != "L8H2" else {})}
                    for a in arms}
            for ph in stats.GRASP_STATES:
                slim[cond]["phase_identity"][ph] = {
                    a: {"mean": res[cond]["phase"][ph]["identity"][a]["mean"],
                        "n_frames": res[cond]["phase"][ph]["identity"][a]["n_frames"]} for a in arms}
        out["windows"][w] = {"n_frames_all": len(fs) // (len(arms) * len(CONDS)), "summary": slim,
                             "t57_entry_A": entry}
        print(w, "frames/arm/cond", out["windows"][w]["n_frames_all"],
              {c: slim[c]["n_target_frames"] for c in CONDS},
              {a: round(slim["current"]["metrics"]["identity"][a]["mean"], 4) for a in ("L8H2", "N", "B", "A")},
              "entry pass", entry["all_numeric_rows_pass"],
              {k: v.get("pass") for k, v in entry["rows"].items()})
    T.write_json(out_json, out)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
