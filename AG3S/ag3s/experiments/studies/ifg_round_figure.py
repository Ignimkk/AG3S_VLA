"""IFG improvement-round figure: paired IoU / recall difference vs the fixed reference G:N per round arm,
with the fixed constraint (recall >= -0.02) and the minimum practical effect (+0.02). JSON sidecar.

  PYTHONPATH=. .venv-ag3s/bin/python -m AG3S.ag3s.experiments.studies.ifg_round_figure --run $RUN --out AG3S/ag3s/docs/figures/ifg
"""
import argparse, json, pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

ARMS = [("R0", "FG:N:projector"), ("R0", "FG:N:prefix_L8"), ("R1", "FGc:N:prefix_L8"), ("R1", "FGc-shuffle:N:prefix_L8"),
        ("R1", "O-comp:N"), ("R2", "FGr:N:prefix_L8"), ("R2", "FGr-shuffle:N:prefix_L8"), ("R2", "O-region:N"),
        ("R3", "FGr1:N:prefix_L8"), ("R3", "FGr1-shuffle:N:prefix_L8"), ("R3", "top1-only:N"), ("R3", "T54region:N")]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", type=pathlib.Path, required=True); ap.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    data = []
    for rnd, arm in ARMS:
        src = a.run / ("s3" if rnd == "R0" else f"rounds/{rnd}") / "paired_summary.json"
        comps = json.loads(src.read_text())["comparisons"]
        got = {}
        for c in comps:
            if c["a"][0] == arm and c["b"][0] == "G:N" and c["a"][1] == 0.5 and c["metric"] in ("iou", "recall"):
                got[c["metric"]] = dict(diff=c["diff"], ci95=c["ci95"], n=c["n_episodes"])
        if arm == "T54region:N":  # stored as G:N - T54region:N; flip sign
            for c in comps:
                if c["a"][0] == "G:N" and c["b"][0] == "T54region:N" and c["metric"] in ("iou", "recall"):
                    got[c["metric"]] = dict(diff=-c["diff"], ci95=[-c["ci95"][1], -c["ci95"][0]], n=c["n_episodes"])
        data.append(dict(round=rnd, arm=arm, **got))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharey=True)
    figstyle.style_axes(fig, axes)
    labels = [f"{d['round']}  {d['arm']}" for d in data]
    for ax, metric, line, lab in ((axes[0], "iou", 0.02, "min practical effect +0.02"), (axes[1], "recall", -0.02, "constraint -0.02")):
        for i, d in enumerate(data):
            m = d[metric]
            kind = 2 if ("O-" in d["arm"]) else (4 if "shuffle" in d["arm"] else (7 if d["arm"].startswith("T54") else (5 if d["arm"].startswith("top1") else 1)))
            ax.errorbar(m["diff"], i, xerr=[[m["diff"] - m["ci95"][0]], [m["ci95"][1] - m["diff"]]], fmt="o", color=figstyle.CATEGORICAL[kind], capsize=2)
        ax.axvline(0, color=figstyle.INK, lw=0.8); ax.axvline(line, color=figstyle.INK_2, lw=0.8, ls="--", label=lab)
        ax.set_xlabel(f"paired episode {metric} difference vs G:N (95% CI, 32 episodes)"); ax.legend(frameon=False, loc="lower left")
    axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels, fontsize=8); axes[0].invert_yaxis()
    fig.suptitle("IFG improvement rounds (dev). orange = feature candidate, pink = shuffle control, teal = GT oracle (diagnostic only), dark green = feature-free top1 rule, red = B0 serving region", fontsize=10)
    fig.tight_layout(); fig.savefig(a.out / "ifg-rounds.png", dpi=150); plt.close(fig)
    (a.out / "ifg-rounds.json").write_text(json.dumps(dict(reference="G:N tau 0.5", rows=data), indent=1))
    print(json.dumps(data)[:400])


if __name__ == "__main__":
    main()
