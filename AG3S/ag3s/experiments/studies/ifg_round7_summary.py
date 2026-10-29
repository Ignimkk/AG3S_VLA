"""R7 summary: every arm vs the R6 parents (G-Q, G-Q+top1) and B0, paired episode bootstrap.
  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_round7_summary --run $RUN
"""
import argparse, csv, json, pathlib

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_summary as S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--round", default="R7")
    a = ap.parse_args()
    RD = a.run / "rounds" / a.round
    valid = S.s0_valid(a.run / "s0")
    rows, _ = S.load_rows(RD, (a.run / "baseline",), valid)
    rows = [r for r in rows if not r["arm_id"].startswith("T54") or r["arm_id"] == "T54region:N"]
    epm = S.episode_means(rows)
    arms = sorted({k[0] for k in epm})
    res, comps = {}, []
    for x in arms:
        e = epm[(x, 0.5, "voxel")]
        res[x] = {m: float(np.mean([v[m] for v in e.values() if m in v])) for m in ("iou", "precision", "recall", "neighbor_fp", "support_fp", "robot_fp")}
        res[x]["fp_proxy"] = float(np.mean([v["neighbor_fp"] + v.get("support_fp", 0) + v.get("robot_fp", 0) for v in e.values()]))
        for ref in ("G-Q", "G-Q+top1", "T54region:N"):
            if x != ref:
                for m in ("iou", "recall", "precision"):
                    comps.append(S.paired(epm, (x, 0.5, "voxel"), (ref, 0.5, "voxel"), m))
    (RD / "paired_summary.json").write_text(json.dumps(dict(arms=res, comparisons=comps, bootstrap=dict(replicates=S.B, seed=S.SEED, unit="episode")), indent=1))
    with open(RD / "episode_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["arm_id", "episode_id", *S.METRICS])
        for x in arms:
            for e, v in sorted(epm[(x, 0.5, "voxel")].items()):
                w.writerow([x, e, *[v.get(m) for m in S.METRICS]])
    C = {(c["a"][0], c["b"][0], c["metric"]): c for c in comps}
    for x in sorted(arms, key=lambda k: -res[k]["iou"]):
        r = res[x]
        line = f"{x:30s} IoU={r['iou']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} FPproxy={r['fp_proxy']:.0f}"
        ref = "G-Q+top1" if x.endswith("+top1") else "G-Q"
        c = C.get((x, ref, "iou"))
        if c:
            line += f" | vs {ref}: dIoU {c['diff']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] {c['episodes_a_better']}/{c['episodes_b_better']}, dR {C[(x, ref, 'recall')]['diff']:+.3f}"
        print(line)


if __name__ == "__main__":
    main()
