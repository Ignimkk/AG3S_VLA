"""IFG R11 — confidence gate: does feature agreement flag wrong identity choices of the R6 candidate?
Reads rounds/<conf>/conf.jsonl (ifg_memory). wrong := R6 chosen IoU < 0.1 with a visible target (evaluator only).
Scores (higher = more confident): q_seed:<f>, q_mem_first:<f>, q_mem_chain:<f>, c_mass (feature-free control).
Reports AUROC(wrong detection) on frames where the score exists, and for an abstain gate at score quantiles
{5,10,20}%: wrong frames caught, FP voxels removed, TP voxels lost (abstain = empty prediction, no exemption).
  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_confidence --run $RUN --conf R11
"""
import argparse, json, pathlib

import numpy as np

from AG3S.ag3s.experiments.studies.ifg_edge_diag import auc

SCORES = ("c_mass", "q_seed:prefix_L8", "q_seed:projector", "q_mem_first:prefix_L8", "q_mem_first:projector",
          "q_mem_chain:prefix_L8", "q_mem_chain:projector")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--conf", default="R11")
    a = ap.parse_args()
    C = [json.loads(l) for l in open(a.run / "rounds" / a.conf / "conf.jsonl")]
    C = [c for c in C if c["target_voxels"] > 0]
    out = {}
    for s in SCORES:
        D = [c for c in C if c.get(s) is not None]
        wrong = np.array([(c["r6_iou"] or 0) < 0.1 for c in D]); sc = np.array([c[s] for c in D])
        r = dict(n=len(D), n_wrong=int(wrong.sum()), auroc=auc(-sc[wrong], -sc[~wrong]) if wrong.any() and (~wrong).any() else None)
        for qq in (5, 10, 20):
            thr = np.percentile(sc, qq); ab = sc <= thr
            r[f"abstain_{qq}pct"] = dict(threshold=float(thr), frames=int(ab.sum()), wrong_caught=int((ab & wrong).sum()),
                                         fp_removed=float(sum(c["r6_fp"] for c, x in zip(D, ab) if x)), fp_total=float(sum(c["r6_fp"] for c in D)),
                                         tp_lost=float(sum(c["r6_tp"] for c, x in zip(D, ab) if x)), tp_total=float(sum(c["r6_tp"] for c in D)))
        out[s] = r
    (a.run / "rounds" / a.conf / "summary.json").write_text(json.dumps(out, indent=1))
    for s, r in out.items():
        g = r["abstain_10pct"]
        print(f"{s:24s} n={r['n']:3d} wrong={r['n_wrong']:2d} AUROC={r['auroc'] if r['auroc'] is None else round(r['auroc'], 3)} | abstain 10%: caught {g['wrong_caught']}/{r['n_wrong']}, FP removed {g['fp_removed']:.0f}/{g['fp_total']:.0f}, TP lost {g['tp_lost']:.0f}/{g['tp_total']:.0f}")


if __name__ == "__main__":
    main()
