"""Final overview figure: episode-mean IoU (95% episode bootstrap CI) of the baselines and each adopted step.
  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_final_figure --run $RUN --out AG3S/ag3s/docs/figures/ifg
"""
import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from AG3S.ag3s.experiments.studies import ifg_summary as S  # noqa: E402

ITEMS = [  # (label, rows dirs, arm id, threshold, colour slot, group)
    ("B0 serving L8H2 (as served)", ("baseline",), "T54region:L8H2", 0.5, 7, "baseline"),
    ("B0 serving noun (as served)", ("baseline",), "T54region:N", 0.5, 7, "baseline"),
    ("B0 serving noun + render mask", ("baseline_rm",), "T54region+rm:N", 0.5, 7, "baseline"),
    ("R0 FG (attention+feature+3D, projector)", ("s3",), "FG:N:projector", 0.5, 1, "feature"),
    ("R0 G (attention+3D graph)", ("s3",), "G:N", 0.5, 0, "ours"),
    ("R3 + top-1 seed-mass region", ("rounds/R3",), "top1-only:N", 0.5, 0, "ours"),
    ("R5 + non-support seed eligibility", ("rounds/R5",), "top1-only-ns:N", 0.5, 0, "ours"),
    ("R6 + render robot mask (candidate)", ("rounds/R6",), "top1-only-ns+rm:N", 0.5, 0, "ours"),
    ("R7 FG on the R6 base (prefix_L8)", ("rounds/R7",), "FG-Q:prefix_L8+top1", 0.5, 1, "feature"),
    ("R7 oracle seed (diagnostic)", ("rounds/R7",), "G-Q:O+top1", 0.5, 2, "oracle"),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    valid = S.s0_valid(a.run / "s0")
    rng = np.random.default_rng(S.SEED)
    vals = []
    for lab, dirs, arm, thr, col, grp in ITEMS:
        rows, _ = S.load_rows(a.run / dirs[0], tuple(a.run / d for d in dirs[1:]), valid)
        rows = [r for r in rows if r["arm_id"] == arm and (r.get("threshold") in (thr, None))]
        epm = S.episode_means(rows)
        e = epm[(arm, thr, "voxel")]
        v = np.array([d["iou"] for d in e.values()])
        boot = v[rng.integers(0, v.size, (S.B, v.size))].mean(1)
        vals.append(dict(label=lab, arm=arm, source=dirs[0], group=grp, mean=float(v.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                         n_episodes=int(v.size), colour=col))
    fig, ax = plt.subplots(figsize=(12, 6)); figstyle.style_axes(fig, ax)
    for i, d in enumerate(vals):
        ax.barh(i, d["mean"], color=figstyle.CATEGORICAL[d["colour"]], alpha=0.85)
        ax.errorbar(d["mean"], i, xerr=[[d["mean"] - d["ci95"][0]], [d["ci95"][1] - d["mean"]]], color=figstyle.INK, capsize=2, lw=0.8)
        ax.text(d["ci95"][1] + 0.01, i, f"{d['mean']:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(vals))); ax.set_yticklabels([d["label"] for d in vals], fontsize=8); ax.invert_yaxis()
    ax.set_xlim(0, 1.05); ax.set_xlabel("episode-mean visible 3D IoU (5 mm voxels), 95% episode bootstrap CI")
    ax.set_title("IFG dev summary: 180 frames / 32 episodes (T53+T54c, all dev; no held-out yet)\nred = serving baseline, blue = Attention+3D steps, orange = feature graph, teal = oracle (diagnostic)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(a.out / "ifg-final-overview.png", dpi=150); plt.close(fig)
    (a.out / "ifg-final-overview.json").write_text(json.dumps(dict(items=vals, bootstrap=dict(replicates=S.B, seed=S.SEED)), indent=1))
    for d in vals:
        print(f"{d['label']:45s} {d['mean']:.3f} [{d['ci95'][0]:.3f},{d['ci95'][1]:.3f}] n={d['n_episodes']}")


if __name__ == "__main__":
    main()
