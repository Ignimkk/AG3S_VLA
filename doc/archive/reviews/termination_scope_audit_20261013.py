"""Read-only source audit; write a separate diagnostic, never replace published results.

Run from the repository root with .venv-ag3s/bin/python.
The provisional cutoff is the first recorded physical `placed` frame, NOT a
newly certified stable-placement or home-return detector. Keep failed episodes.
"""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t54_stats as stats


def read(path):
    return json.loads(Path(path).read_text())


def main():
    handoff = ROOT / "AG3S/ag3s/docs/handoff"
    v53 = read(handoff / "T53.verify.json")
    v54 = read(handoff / "T54.verify.json")
    gt_dir = Path(v53["extraction"]["paths"]["gt"])
    rows_dir = Path(v54["run_dir"]) / "verify/T54/test/rows"
    keys = v54["numbers"]["keys"]
    arms = v54["numbers"]["arms"]
    records = {}
    rows = {}
    gt, grasp = {}, {}
    for key in keys:
        with np.load(gt_dir / f"{key}.npz", allow_pickle=True) as z:
            gt[key] = {k: z[k] for k in ("t_step", "class_px")}
        with np.load(gt_dir / f"{key}.grasp.npz", allow_pickle=True) as z:
            grasp[key] = {k: z[k] for k in ("t_step", "grasp_state")}
        phases = grasp[key]["grasp_state"].astype(str)
        placed = np.flatnonzero(phases == "placed")
        cut = int(placed[0]) if len(placed) else len(phases)
        records[key] = {
            "frames": len(phases), "first_placed_frame": cut if len(placed) else None,
            "placed_frames": int((phases == "placed").sum()),
            "frames_removed_by_proxy": len(phases) - cut,
            "later_nonplaced_frames": int((phases[cut:] != "placed").sum()),
        }
        for cond in ("current", "hold"):
            source = rows_dir / f"{key}__{cond}.jsonl"
            for line in source.read_text().splitlines():
                r = json.loads(line)
                rows.setdefault((key, cond, r["arm"]), []).append(r)
    frames = stats.build_frames(rows, gt, grasp)
    before = [r for r in frames if records[r["key"]]["first_placed_frame"] is None
              or r["frame"] < records[r["key"]]["first_placed_frame"]]
    draws = stats.episode_draws(len(keys), 10000, 5407)
    out = {
        "scope": "diagnostic only; first physical placed proxy, excluding its inference frame",
        "limitations": [
            "placed means in crate and not held; no stable dwell or actual home criterion",
            "hold replay retains the original T54 attach limitation",
            "fixed existing arms; no new selection, training, policy inference or rollout",
        ],
        "sources": {"T53": v53["run_dir"], "T54": v54["run_dir"]},
        "records": records,
        "frames_total": sum(v["frames"] for v in records.values()),
        "frames_removed": sum(v["frames_removed_by_proxy"] for v in records.values()),
        "results": {},
    }
    for cond in ("current", "hold"):
        out["results"][cond] = {}
        for label, fs in (("original", frames), ("before_first_placed_proxy", before)):
            rec = {}
            for metric in ("identity", "no_target", "purity", "coverage"):
                e, counts = stats.episode_matrix(fs, keys=keys, arms=arms,
                                                condition=cond, metric=metric)
                assert np.isfinite(e).all(), (cond, label, metric)
                vals = {}
                for j, arm in enumerate(arms):
                    diff = e[:, j] - e[:, 0]
                    vals[arm] = {
                        "mean": float(e[:, j].mean()), "n_frames": int(counts[:, j].sum()),
                        "diff_vs_L8H2": float(diff.mean()),
                        "diff_ci95": np.percentile(diff[draws].mean(axis=1), [2.5, 97.5]).tolist(),
                    }
                    if label == "original":
                        expected = v54["numbers"][cond]["metrics"][metric][arm]["mean"]
                        assert abs(vals[arm]["mean"] - expected) < 1e-6, (metric, arm)
                rec[metric] = vals
            out["results"][cond][label] = rec
    p = ROOT / "doc/archive/reviews/TERMINATION_SCOPE_AUDIT_20261013.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print("original T54 means reproduced within 1e-6")
    print("frames", out["frames_total"], "removed", out["frames_removed"])
    for cond in out["results"]:
        for label, rec in out["results"][cond].items():
            print(cond, label, {a: {k: round(v, 6) if isinstance(v, float) else v
                                    for k, v in rec["identity"][a].items()}
                                for a in ("L8H2", "N", "B", "A")})
    print(p)


if __name__ == "__main__":
    main()
