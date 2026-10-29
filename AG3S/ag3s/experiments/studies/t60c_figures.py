"""T60c figures — T51b selections per window (table) and bootstrap rank-1 probability / held-out identity (graph).

usage: t60c_figures.py FIG_DIR      (reads handoff/T60c.verify.json)
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.common import figstyle  # noqa: E402
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402

WINDOWS = ("full", "P", "H")
OMEGAS = (("action_t51", "action (T51)"), ("noun", "noun"), ("combined", "combined"))


def main(fig_dir: pathlib.Path) -> None:
    ko = figstyle.use_korean()
    n = json.loads((ROOT / "AG3S/ag3s/docs/handoff/T60c.verify.json").read_text())["numbers"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    # graph
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    ax = axes[0]
    labels = [f"{lab}\n{l}" for om, lab in OMEGAS for l in ("gt", "self")]
    x = np.arange(len(labels))
    side = {"p_rank1": {}, "heldout": {}}
    for i, w in enumerate(WINDOWS):
        v = [n[w]["omegas"][om][l]["bootstrap_p_rank1"] for om, _ in OMEGAS for l in ("gt", "self")]
        ax.bar(x + (i - 1) * 0.27, v, 0.27, color=figstyle.SEQ_BLUE[[1, 5, -1][i]], label=w)
        side["p_rank1"][w] = dict(zip(labels, v))
    ax.axhline(0.5, color=figstyle.INK_2, ls=":", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(0, 1)
    ax.set_ylabel("scene bootstrap P(rank 1) of the selected cell")
    ax.set_title("선택의 안정성 — 창별" if ko else "selection stability per window", loc="left"); ax.legend(fontsize=8)
    ax = axes[1]
    keys = ["l8h2", "action_best_gt", "action_best_self", "noun_best_gt", "noun_best_self", "combined_best_gt", "combined_best_self"]
    xk = np.arange(len(keys))
    for i, w in enumerate(WINDOWS):
        v = [n[w]["heldout_scene_mean_gt_identity"][k] for k in keys]
        ax.bar(xk + (i - 1) * 0.27, v, 0.27, color=figstyle.SEQ_BLUE[[1, 5, -1][i]], label=w)
        side["heldout"][w] = {k: [n[w]["heldout_named"][k], vv] for k, vv in zip(keys, v)}
    ax.set_xticks(xk); ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8); ax.set_ylim(0.8, 0.93)
    ax.set_ylabel("held-out GT identity (scene mean)")
    ax.set_title("held-out — 어느 창의 선택도 L8H2 (0.911) 를 넘지 않는다" if ko else "held-out identity", loc="left")
    ax.legend(fontsize=8)
    figstyle.style_axes(fig, axes)
    fig.tight_layout(); fig.savefig(fig_dir / "t60c-stability-heldout.png", dpi=130); plt.close(fig)
    T.write_json(fig_dir / "t60c-stability-heldout.json", side)
    # table
    rows = []
    for om, lab in OMEGAS:
        for l in ("gt", "self"):
            rows.append([f"{lab} · {l} label: P4 selected (P(rank1))"] +
                        [f"{n[w]['omegas'][om][l]['selected']} ({n[w]['omegas'][om][l]['bootstrap_p_rank1']:.3f})" for w in WINDOWS])
    rows.append(["held-out identity: noun best GT / combined best GT / L8H2"] +
                [f"{n[w]['heldout_scene_mean_gt_identity']['noun_best_gt']:.3f} / {n[w]['heldout_scene_mean_gt_identity']['combined_best_gt']:.3f} / {n[w]['heldout_scene_mean_gt_identity']['l8h2']:.3f}" for w in WINDOWS])
    fig, ax = plt.subplots(figsize=(17, 0.6 + 0.45 * len(rows))); ax.axis("off")
    tb = ax.table(cellText=rows, colLabels=["", "full (published)", "P", "H"], loc="center", cellLoc="center",
                  colWidths=[0.34, 0.22, 0.22, 0.22])
    tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.45)
    for (i, j), c in tb.get_celld().items():
        c.set_edgecolor(figstyle.GRID_INK)
        if i == 0:
            c.set_facecolor(figstyle.SEQ_BLUE[0])
        elif j > 1 and rows[i - 1][j].split(" (")[0] != rows[i - 1][1].split(" (")[0]:
            c.set_facecolor("#fde9dd")
    ax.set_title("T60c — T51b 선택의 창별 비교 (주황 = full 과 다른 cell)" if ko else "T60c — T51b selections per window", fontsize=12)
    figstyle.style_axes(fig, [ax])
    fig.savefig(fig_dir / "t60c-selection-table.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    T.write_json(fig_dir / "t60c-selection-table.json", {"rows": rows})
    print("ok")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]))
