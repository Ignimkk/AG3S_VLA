"""T60 step 2 figures — closed-loop metrics per arm × window, HOLD position vs the endpoints, T54 differences.

usage: t60_figures2.py STEP2_DIR FIG_DIR
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
WCOL = {"full": figstyle.SEQ_BLUE[1], "P": figstyle.SEQ_BLUE[5], "H": figstyle.SEQ_BLUE[-1]}


def closed_loop(cl: dict, fig_dir: pathlib.Path, ko: bool) -> dict:
    arms = [(s, a) for s, v in cl["sets"].items() for a in v["arms"]]
    labels = [f"{s}\n{a}" for s, a in arms]
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    side = {}
    panels = (("success", "success (runs)"), ("success_ct", "safe success success_ct (runs)"),
              ("hold_chunks", "HOLD chunks in window"), ("completion_s_median", "completion time median (s)"))
    x = np.arange(len(arms)); wdt = 0.27
    for ax, (key, title) in zip(axes.flat, panels):
        for i, w in enumerate(WINDOWS):
            vals = [cl["sets"][s]["arms"][a][w][key] for s, a in arms]
            vals = [np.nan if v is None else v for v in vals]
            ax.bar(x + (i - 1) * wdt, vals, wdt, color=WCOL[w], label=w)
            for xi, v in zip(x + (i - 1) * wdt, vals):
                if np.isfinite(v):
                    ax.text(xi, v, f"{v:.0f}" if key != "completion_s_median" else f"{v:.1f}",
                            ha="center", va="bottom", fontsize=6.5)
            side.setdefault(key, {})[w] = dict(zip([f"{s}/{a}" for s, a in arms], vals))
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8); ax.set_title(title, loc="left", fontsize=11)
    n_runs = {f"{s}/{a}": cl["sets"][s]["arms"][a]["n_runs"] for s, a in arms}
    axes[0, 0].legend(title="window", fontsize=8)
    figstyle.style_axes(fig, axes.flat)
    fig.suptitle("T60 — closed-loop 지표: full (발표 정의) · P (안정 placement 까지) · H (준비자세 복귀까지)" if ko else
                 "T60 — closed-loop metrics: full (published) · P · H windows", fontsize=13)
    fig.tight_layout(); fig.savefig(fig_dir / "t60-closed-loop.png", dpi=130); plt.close(fig)
    side["n_runs"] = n_runs
    T.write_json(fig_dir / "t60-closed-loop.json", side)
    return side


def hold_position(cl: dict, fig_dir: pathlib.Path, ko: bool) -> dict:
    arms = [(s, a) for s, v in cl["sets"].items() for a in v["arms"]]
    cats = [("no_P", "P 미도달 run" if ko else "run without P"), ("le_tP", "≤ t_P"),
            ("tP_to_tH", "t_P – t_H (복귀 중)" if ko else "t_P – t_H (returning)"),
            ("after_tP_noH", "> t_P, H 미도달" if ko else "> t_P, no H"),
            ("after_tH", "> t_H (복귀 완료 후)" if ko else "> t_H (after return)")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), sharey=True)
    side = {}
    for ax, kind in zip(axes, ("collision", "uncertified")):
        bottom = np.zeros(len(arms))
        side[kind] = {}
        for ci, (c, lab) in enumerate(cats):
            v = np.array([cl["sets"][s]["arms"][a]["hold_position"].get(f"{kind}|{c}", 0) for s, a in arms])
            ax.bar(np.arange(len(arms)), v, bottom=bottom, color=figstyle.CATEGORICAL[ci], label=lab)
            bottom += v
            side[kind][c] = dict(zip([f"{s}/{a}" for s, a in arms], v.tolist()))
        for xi, b in enumerate(bottom):
            ax.text(xi, b, f"{int(b)}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(arms))); ax.set_xticklabels([f"{s}\n{a}" for s, a in arms], fontsize=8)
        ax.set_title(f"HOLD kind = {kind}", loc="left")
    axes[0].set_ylabel("HOLD chunks (full horizon)")
    axes[1].legend(fontsize=8)
    figstyle.style_axes(fig, axes)
    fig.suptitle("T60 — HOLD chunk 가 종료 시점의 어디에 있었나" if ko else "T60 — where the HOLD chunks sit relative to t_P / t_H",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(fig_dir / "t60-hold-position.png", dpi=130); plt.close(fig)
    T.write_json(fig_dir / "t60-hold-position.json", side)
    return side


def t54(d: dict, fig_dir: pathlib.Path, ko: bool) -> dict:
    arms = [a for a in d["arms"] if a != "L8H2"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    side = {}
    for ax, cond in zip(axes, ("current", "hold")):
        side[cond] = {}
        for i, w in enumerate(WINDOWS):
            m = d["windows"][w]["summary"][cond]["metrics"]["identity"]
            v = np.array([m[a]["diff_vs_L8H2"] * 100 for a in arms])
            ci = np.array([[m[a]["diff_ci95_episode_bootstrap"][0] * 100, m[a]["diff_ci95_episode_bootstrap"][1] * 100] for a in arms])
            xs = np.arange(len(arms)) + (i - 1) * 0.27
            ax.bar(xs, v, 0.27, color=WCOL[w], label=f"{w} (target frames {d['windows'][w]['summary'][cond]['n_target_frames']})")
            ax.errorbar(xs, v, yerr=[v - ci[:, 0], ci[:, 1] - v], fmt="none", ecolor=figstyle.INK_2, lw=0.8, capsize=2)
            side[cond][w] = {a: {"diff_pp": float(x), "ci95_pp": c.tolist()} for a, x, c in zip(arms, v, ci)}
        ax.axhline(0, color=figstyle.INK, lw=0.8); ax.axhline(-3, color=figstyle.CATEGORICAL[1], ls="--", lw=1)
        ax.text(len(arms) - 0.6, -3.4, "−3 %p", color=figstyle.CATEGORICAL[1], fontsize=8)
        ax.set_xticks(np.arange(len(arms))); ax.set_xticklabels(arms)
        ax.set_ylabel("identity − L8H2 (%p, episode mean, 95% CI)")
        ax.set_title(f"T54 {cond}", loc="left"); ax.legend(fontsize=8)
    side["entry_A"] = {w: {"all_pass": d["windows"][w]["t57_entry_A"]["all_numeric_rows_pass"],
                           "rows": {k: v.get("pass") for k, v in d["windows"][w]["t57_entry_A"]["rows"].items()}}
                       for w in WINDOWS}
    figstyle.style_axes(fig, axes)
    fig.suptitle("T60 — T54 고정 arm 의 identity 차이: 창에 따라 어떻게 바뀌나" if ko else "T60 — T54 identity differences per window",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(fig_dir / "t60-t54-identity.png", dpi=130); plt.close(fig)
    T.write_json(fig_dir / "t60-t54-identity.json", side)
    return side


def main(step2: pathlib.Path, fig_dir: pathlib.Path) -> None:
    ko = figstyle.use_korean()
    cl = json.loads((step2 / "closed_loop.json").read_text())
    closed_loop(cl, fig_dir, ko)
    hold_position(cl, fig_dir, ko)
    t54(json.loads((step2 / "t54.json").read_text()), fig_dir, ko)
    print("ok")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
