"""T67 stage 1 — mechanism figures for session B (reads `t67_mech.py` output and session A's outcome table).

  graph_close_and_push.png   per failed run (final configuration's 8): palm–apple distance when closing starts and the
                             largest palm shift in the last 60 mm, final vs C1 / C2 / C3; outcome as marker
  table_t67_s1_mech.md

usage: t67_s1_figs.py MECH_JSON FINAL_AGG_JSON OUT_DIR
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "common"))
import figstyle  # noqa: E402

RUNS = ["회귀211_k0", "회귀211_k1", "회귀954_k0", "가림122_k0", "hurdle954_k0", "hurdle954_k1", "hurdle1345_k0", "hurdle211_k0"]
EN = {"회귀211_k0": "regr 211 d0", "회귀211_k1": "regr 211 d1", "회귀954_k0": "regr 954 d0", "가림122_k0": "occl 122 d0",
      "hurdle954_k0": "hurdle 954 d0", "hurdle954_k1": "hurdle 954 d1", "hurdle1345_k0": "hurdle 1345 d0", "hurdle211_k0": "hurdle 211 d0"}
CONFS = ["final", "C1", "C2", "C3"]
NAMES = {"final": "final config", "C1": "C1 window", "C2": "C2 table offset", "C3": "C3 all three"}
# session A's stage-1 outcome table (outputs/2026-10-29_T67_쥐기실패_보정실험/README.md), contact-free success = True
OUTCOME = {
    "final": dict.fromkeys(RUNS, False),
    "C1": {"회귀211_k0": True, "회귀211_k1": True, "회귀954_k0": False, "가림122_k0": False, "hurdle954_k0": False,
           "hurdle954_k1": False, "hurdle1345_k0": True, "hurdle211_k0": True},
    "C2": {"회귀211_k0": False, "회귀211_k1": True, "회귀954_k0": True, "가림122_k0": True, "hurdle954_k0": False,
           "hurdle954_k1": False, "hurdle1345_k0": True, "hurdle211_k0": False},
    "C3": {"회귀211_k0": False, "회귀211_k1": True, "회귀954_k0": False, "가림122_k0": True, "hurdle954_k0": False,
           "hurdle954_k1": False, "hurdle1345_k0": False, "hurdle211_k0": True},
}


def main(mech_json, final_json, out):
    out.mkdir(parents=True, exist_ok=True)
    mech = json.load(open(mech_json))
    fin = {x["run"]: x for x in json.load(open(final_json))}
    val = {}
    for c in CONFS:
        for r in RUNS:
            x = fin[r] if c == "final" else mech[f"{c}|{r}"]
            val[(c, r)] = {"close": x["close_dist_mm"], "push": x["max_shift_win_mm"], "ok": OUTCOME[c][r],
                           "goal": None if c == "final" else x.get("n_goal_win"), "n_win": x["n_win"]}
    cols = figstyle.CATEGORICAL
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    y = np.arange(len(RUNS))
    h = 0.2
    for k, c in enumerate(CONFS):
        for ax, key in zip(axes, ("close", "push")):
            v = [val[(c, r)][key] for r in RUNS]
            ax.barh(y + (k - 1.5) * h, v, h, color=cols[k], label=NAMES[c])
            for i, r in enumerate(RUNS):
                if val[(c, r)]["ok"]:
                    ax.text(v[i] + 1, y[i] + (k - 1.5) * h, "✓", va="center", fontsize=7, color=cols[k])
    for ax, xl in zip(axes, ("palm to apple centre when closing starts (mm)",
                             "largest palm shift by TrajOpt, last 60 mm before closing (mm)")):
        ax.set_yticks(y, [EN[r] for r in RUNS], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(xl)
    axes[0].axvspan(0, 25, color=figstyle.GRID_INK, alpha=0.4, lw=0)
    axes[0].set_title("shaded: 0-25 mm, where the grasped runs closed;  ✓ = contact-free success", fontsize=9)
    axes[1].set_title("same 8 runs the final configuration missed (same scene and seed)", fontsize=9)
    axes[1].legend(fontsize=8)
    figstyle.style_axes(fig, list(axes))
    fig.tight_layout()
    fig.savefig(out / "graph_close_and_push.png", dpi=150)
    plt.close(fig)
    (out / "graph_close_and_push.json").write_text(json.dumps({f"{c}|{r}": v for (c, r), v in val.items()},
                                                              ensure_ascii=False, indent=1))
    lines = ["| run | " + " | ".join(NAMES[c] for c in CONFS) + " |", "|---|" + "---|" * len(CONFS)]
    for r in RUNS:
        cells = []
        for c in CONFS:
            v = val[(c, r)]
            cells.append(f"{'**성공**' if v['ok'] else '실패'} · 닫기 {v['close']:.0f} mm · 밀림 {v['push']:.0f} mm"
                         + ("" if v["goal"] is None else f" · 목표 {v['goal']}/{v['n_win']}"))
        lines.append(f"| {r} | " + " | ".join(cells) + " |")
    (out / "table_t67_s1_mech.md").write_text("\n".join(lines) + "\n")
    print("->", out)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3]))
