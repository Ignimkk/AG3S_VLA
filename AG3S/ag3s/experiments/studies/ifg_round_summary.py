"""Paired episode summary of one improvement round against the fixed reference (G:N) and B0 (T54region:N).

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_round_summary --run $RUN --round R2
Writes rounds/<R>/paired_summary.json, episode_metrics.csv; checks the parent G:N equals R0 (parity).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

from AG3S.ag3s.experiments.studies import ifg_summary as S

REF, B0 = "G:N", "T54region:N"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--round", required=True)
    a = ap.parse_args()
    RD = a.run / "rounds" / a.round
    valid = S.s0_valid(a.run / "s0")
    rr, _ = S.load_rows(RD, (), valid)
    r0, _ = S.load_rows(a.run / "s3", (a.run / "baseline",), valid)
    par = {r["frame_id"]: r for r in rr if r["arm_id"] == REF}
    ref0 = {r["frame_id"]: r for r in r0 if r["arm_id"] == REF and r.get("threshold") == 0.5}
    parity = max(abs((par[f].get("iou") or 0) - (ref0[f].get("iou") or 0)) + abs(par[f]["FP"] - ref0[f]["FP"]) for f in par)
    rows = rr + [r for r in r0 if r["arm_id"] in (B0, "T54ball:N") and r.get("threshold") == 0.5]
    epm = S.episode_means(rows)
    arms = sorted({r["arm_id"] for r in rows}, key=lambda x: (x != REF, x.startswith("T54"), x))
    res, comps = {}, []
    for x in arms:
        e = epm[(x, 0.5, "voxel")]
        res[x] = {m: float(np.mean([v[m] for v in e.values() if m in v])) for m in ("iou", "precision", "recall", "neighbor_fp", "support_fp", "robot_fp")}
        res[x]["fp_proxy"] = float(np.mean([v["neighbor_fp"] + v.get("support_fp", 0) + v.get("robot_fp", 0) for v in e.values()]))
        for ref in (REF, B0):
            if x == ref:
                continue
            for m in ("iou", "recall", "precision", "neighbor_fp"):
                comps.append(S.paired(epm, (x, 0.5, "voxel"), (ref, 0.5, "voxel"), m))
    (RD / "paired_summary.json").write_text(json.dumps(dict(parent_parity_vs_R0=parity, arms=res, comparisons=comps,
                                                             bootstrap=dict(replicates=S.B, seed=S.SEED, unit="episode")), indent=1))
    with open(RD / "episode_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm_id", "episode_id", *S.METRICS])
        for x in arms:
            for e, v in sorted(epm[(x, 0.5, "voxel")].items()):
                w.writerow([x, e, *[v.get(m) for m in S.METRICS]])
    print("parent parity vs R0:", parity)
    for x in arms:
        r = res[x]
        print(f"{x:26s} IoU={r['iou']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} nbrFP={r['neighbor_fp']:.0f} FPproxy={r['fp_proxy']:.0f}")
    for c in comps:
        if c["metric"] in ("iou", "recall"):
            print(f"  {c['a'][0]} - {c['b'][0]} [{c['metric']}] {c['diff']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] {c['episodes_a_better']}/{c['episodes_b_better']}")


if __name__ == "__main__":
    main()
