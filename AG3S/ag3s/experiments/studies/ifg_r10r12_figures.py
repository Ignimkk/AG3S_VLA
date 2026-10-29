"""Figures for R10/R10b (identity memory), R11/R11b (confidence gate), R12 (phase / held). JSON sidecars.
  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_r10r12_figures --run $RUN --out AG3S/ag3s/docs/figures/ifg
"""
import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args(); R = a.run / "rounds"
    s10 = json.loads((R / "R10/paired_summary.json").read_text()); s10b = json.loads((R / "R10b/paired_summary.json").read_text())
    c11 = [json.loads(l) for l in open(R / "R11b/conf.jsonl")]
    r12 = json.loads((R / "R12/summary.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(19, 6)); figstyle.style_axes(fig, axes)
    # (a) memory switching: R10 vs R10b, prefix_L8 first-frame memory
    arms = [f"Mem-first:prefix_L8:d{d}" for d in (0.0, 0.05, 0.1)]
    side = {"memory": {}}
    for j, (lab, s, k) in enumerate((("raw cosine (R10)", s10, 1), ("centered (R10b)", s10b, 0))):
        d = [s["comparisons"][x]["iou"] for x in arms]
        xs = np.arange(len(arms)) + (j - 0.5) * 0.3
        axes[0].errorbar(xs, [c["diff"] for c in d], yerr=[[c["diff"] - c["ci95"][0] for c in d], [c["ci95"][1] - c["diff"] for c in d]],
                         fmt="o", color=figstyle.CATEGORICAL[k], capsize=3, label=lab)
        side["memory"][lab] = {x: dict(diff=c["diff"], ci95=c["ci95"], switches=s["switches"].get(x)) for x, c in zip(arms, d)}
    axes[0].axhline(0, color=figstyle.INK, lw=0.8); axes[0].axhline(0.02, color=figstyle.INK_2, lw=0.8, ls="--")
    axes[0].set_xticks(range(len(arms))); axes[0].set_xticklabels(["delta 0", "delta 0.05", "delta 0.1"])
    axes[0].set_ylabel("paired episode IoU diff vs R6 (95% CI)"); axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("R10 identity memory (prefix_L8, first-frame memory)", fontsize=9)
    # (b) confidence gate curve: fraction of FP removed vs TP lost when abstaining on the lowest scores (centered)
    D = [c for c in c11 if c["target_voxels"] > 0 and c.get("q_mem_chain:prefix_L8") is not None and c.get("c_mass") is not None]
    fp = np.array([c["r6_fp"] for c in D]); tp = np.array([c["r6_tp"] for c in D])
    side["gate"] = {}
    for s, lab, k in (("c_mass", "seed-mass share (no feature)", 5), ("q_seed:prefix_L8", "same-frame seed prototype", 3),
                      ("q_mem_chain:prefix_L8", "chain memory (centered)", 0)):
        sc = np.array([c[s] for c in D]); o = np.argsort(sc)
        fx = np.cumsum(fp[o]) / fp.sum(); tx = np.cumsum(tp[o]) / tp.sum()
        axes[1].plot(np.r_[0, tx], np.r_[0, fx], color=figstyle.CATEGORICAL[k], label=lab)
        side["gate"][s] = dict(tp_lost=np.r_[0, tx].tolist(), fp_removed=np.r_[0, fx].tolist())
    axes[1].plot([0, 1], [0, 1], color=figstyle.GRID_INK, lw=0.8)
    axes[1].set_xlim(0, 0.3); axes[1].set_xlabel("share of target TP voxels lost (abstained frames)"); axes[1].set_ylabel("share of FP voxels removed")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right"); axes[1].set_title(f"R11 confidence gate ({len(D)} frames, 6 wrong choices)", fontsize=9)
    # (c) phase / held accuracy
    ph = r12["results"]["phase"]; hd = r12["results"]["held"]
    names = ["P1 HSR argmax", "LR[hsr]", "LR[hsr+grip]", "LR[hsr+grip+geo]", "LR[hsr+grip+geo+feat]"]
    hn = ["rule grip<0.9", "HSR place>0.5", "LR[grip+geo]", "LR[hsr+grip+geo]", "LR[hsr+grip+geo+feat]"]
    x = np.arange(len(names))
    axes[2].bar(x - 0.2, [ph[n]["accuracy"] for n in names], 0.4, color=figstyle.CATEGORICAL[0], label="phase (pick/place/home)")
    axes[2].bar(x + 0.2, [hd[n]["accuracy"] for n in hn], 0.4, color=figstyle.CATEGORICAL[1], label="held (yes/no)")
    axes[2].set_xticks(x); axes[2].set_xticklabels([f"{a}\n{b}" for a, b in zip(names, hn)], fontsize=6.5); axes[2].set_ylim(0.5, 1.0)
    axes[2].set_ylabel("accuracy (LOEO CV for LR)"); axes[2].legend(frameon=False, fontsize=8, loc="upper left")
    axes[2].set_title("R12 phase / held judgment (supervised ablation for LR)", fontsize=9)
    side["phase_held"] = dict(phase={n: ph[n]["accuracy"] for n in names}, held={n: hd[n]["accuracy"] for n in hn}, paired=r12["paired"])
    fig.suptitle("Feature reuse candidates (dev, 180 frames / 32 episodes)", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out / "ifg-r10-r12-feature-reuse.png", dpi=150); plt.close(fig)
    (a.out / "ifg-r10-r12-feature-reuse.json").write_text(json.dumps(side, indent=1))
    print("ok")


if __name__ == "__main__":
    main()
