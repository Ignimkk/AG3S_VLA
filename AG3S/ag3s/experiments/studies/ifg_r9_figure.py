"""R9 figure: filter-matched comparison of the serving region (B0) and the Attention + 3D candidate.
  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_r9_figure --run $RUN --out AG3S/ag3s/docs/figures/ifg
"""
import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    s = json.loads((a.run / "rounds/R9/paired_summary.json").read_text())
    A = s["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw=dict(width_ratios=[1, 1.4])); figstyle.style_axes(fig, axes)
    groups = [("sphere self-filter", "T54region:N", "top1-only-ns:N"), ("render robot mask", "T54region+rm:N", "top1-only-ns+rm:N")]
    x = np.arange(len(groups)); w = 0.35
    for j, (lab, k) in enumerate((("B0 serving region", 7), ("Attention + 3D", 0))):
        vals = [A[g[1 + j]]["iou"] for g in groups]
        axes[0].bar(x + (j - 0.5) * w, vals, w, color=figstyle.CATEGORICAL[k], label=lab)
        for xi, v in zip(x + (j - 0.5) * w, vals):
            axes[0].text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(x); axes[0].set_xticklabels([g[0] for g in groups]); axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("episode-mean visible 3D IoU"); axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    C = {(c["a"][0], c["b"][0], c["metric"]): c for c in s["comparisons"]}
    m = C[("top1-only-ns+rm:N", "T54region+rm:N", "iou")]; f = C[("T54region+rm:N", "T54region:N", "iou")]
    axes[0].set_title(f"same mask: method {m['diff']:+.3f} [{m['ci95'][0]:+.3f},{m['ci95'][1]:+.3f}]; mask on B0 {f['diff']:+.3f}", fontsize=9)
    M = s["per_moment_iou"]; mom = list(M["top1-only-ns+rm:N"])
    xm = np.arange(len(mom))
    for j, (arm, lab, k) in enumerate((("T54region:N", "B0 (sphere)", 5), ("T54region+rm:N", "B0 (render mask)", 7), ("top1-only-ns+rm:N", "Attention + 3D (render mask)", 0))):
        axes[1].bar(xm + (j - 1) * 0.27, [M[arm][q] for q in mom], 0.27, color=figstyle.CATEGORICAL[k], label=lab)
    axes[1].set_xticks(xm); axes[1].set_xticklabels(mom); axes[1].set_ylim(0, 1.05); axes[1].set_ylabel("frame-mean IoU per moment")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("R9: filter-matched comparison (dev, 180 frames / 32 episodes, current condition)", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out / "ifg-r9-filter-matched.png", dpi=150); plt.close(fig)
    (a.out / "ifg-r9-filter-matched.json").write_text(json.dumps(dict(arms=A, method_effect=m, filter_effect_on_B0=f, per_moment=M), indent=1))
    print("ok")


if __name__ == "__main__":
    main()
