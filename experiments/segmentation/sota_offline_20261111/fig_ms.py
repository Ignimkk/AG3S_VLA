#!/usr/bin/env python3
"""(2026-11-11 dev-57) Rule-A figure for plan §9.16: current config vs the -qs candidate on E1a 40 and the other-suite
set (56 ep) -- TSR / CAR / SSR, per-cell SSR, and the shape of the episodes that are not a safe success.
usage: OMP_NUM_THREADS=1 .venv-ag3s/bin/python outputs/2026-11-11_SOTA_개선/fig_ms.py OUT.png"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.experiments.common import figstyle

O = "/mnt/dev/AG3S_VLA/outputs/2026-11-11_SOTA_개선/offline/"
out = sys.argv[1]
figstyle.use_korean()
C = figstyle.CATEGORICAL
TOT = {"E1a 40 (Spatial L-I)": {"current": (22, 23, 15), "-qs": (22, 32, 18), "n": 40, "ref": (63.0, 67.5, 53.0), "refname": "KNOWS Spatial L-I (reported)"},
       "other suites (56 ep)": {"current": (24, 31, 17), "-qs": (24, 40, 19), "n": 56, "ref": (67.5, 76.4, 59.9), "refname": "KNOWS 4-suite L-I (reported)"}}
CELLS = ["Object L-I", "Object L-II", "Goal L-I", "Goal L-II", "Long L-I", "Long L-II", "Spatial L-II"]
CELL = {"current": [(5, 4, 4), (5, 5, 3), (3, 4, 2), (3, 5, 3), (2, 5, 2), (3, 7, 2), (3, 1, 1)],
        "-qs": [(5, 7, 5), (4, 8, 4), (4, 3, 1), (4, 3, 2), (2, 6, 2), (1, 6, 1), (4, 7, 4)]}
GROUPS = [("no grasp: QP persists before the grasp", lambda k: k.startswith("못 쥠: 쥐기 전 QP")),
          ("no grasp: few QP calls", lambda k: k.startswith("못 쥠: QP 적음")),
          ("still holding at the end", lambda k: k.startswith("쥔 채")),
          ("grasped, released, not completed", lambda k: k.startswith("쥐었다")),
          ("HOLD to the end", lambda k: k.startswith("끝까지")),
          ("success + collision", lambda k: k.startswith("성공"))]
SH = [("current, other suites", "shape_ms_pe123458.json"), ("-qs, other suites", "shape_ms_pkgFqs.json"),
      ("current, E1a 40", "shape_e1a_pe123458.json"), ("-qs, E1a 40", "shape_e1a_pkgFqs.json")]
shapes = []
for name, f in SH:
    d = json.load(open(O + f))
    shapes.append((name, d["n"], d["safe_success"], [sum(v for k, v in d["classes"].items() if g(k)) for _, g in GROUPS]))
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.15, 1.25, 1.5]})
figstyle.style_axes(fig, axes)
ax = axes[0]
x0 = 0
ticks, labels = [], []
for gi, (gname, g) in enumerate(TOT.items()):
    for m in range(3):
        for a, arm in enumerate(("current", "-qs")):
            x = x0 + m + (a - 0.5) * 0.36
            v = 100 * g[arm][m] / g["n"]
            ax.bar(x, v, 0.34, color=C[a], label=("current config (pe123458)" if a == 0 else "-qs candidate") if gi == 0 and m == 0 else None)
            ax.text(x, v + 1.2, str(g[arm][m]), ha="center", fontsize=8, color=figstyle.INK)
        ax.plot([x0 + m - 0.42, x0 + m + 0.42], [g["ref"][m]] * 2, color=figstyle.INK, lw=1.3, ls="--",
                label="KNOWS reported (Spatial L-I | 4-suite L-I)" if gi == 0 and m == 0 else None)
        ticks.append(x0 + m); labels.append(["TSR", "CAR", "SSR"][m])
    ax.text(x0 + 1, -13, gname, ha="center", fontsize=9, color=figstyle.INK)
    x0 += 3.6
ax.set_xticks(ticks); ax.set_xticklabels(labels); ax.set_ylim(0, 100)
ax.set_title("(a) totals, % (count above the bar)", fontsize=10, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax = axes[1]
xs = np.arange(len(CELLS))
for a, arm in enumerate(("current", "-qs")):
    ax.bar(xs + (a - 0.5) * 0.36, [c[2] for c in CELL[arm]], 0.34, color=C[a])
    for x, c in zip(xs, CELL[arm]):
        ax.text(x + (a - 0.5) * 0.36, c[2] + 0.12, f"{c[0]}·{c[1]}·{c[2]}", ha="center", fontsize=6.5, color=figstyle.INK, rotation=90, va="bottom")
ax.set_xticks(xs); ax.set_xticklabels(CELLS, fontsize=7.5, rotation=20); ax.set_ylim(0, 8)
ax.set_title("(b) other suites: safe successes of 8 per cell (label = TSR·CAR·SSR)", fontsize=10, loc="left")
ax = axes[2]
ys = np.arange(len(shapes))[::-1]
left = np.zeros(len(shapes))
for gi, (gname, _) in enumerate(GROUPS):
    vals = np.array([s[3][gi] for s in shapes], float)
    ax.barh(ys, vals, left=left, color=C[(gi + 2) % len(C)], label=gname, height=0.62)
    for y, l, v in zip(ys, left, vals):
        if v >= 1:
            ax.text(l + v / 2, y, str(int(v)), ha="center", va="center", fontsize=8, color="white")
    left += vals
ax.set_yticks(ys); ax.set_yticklabels([f"{s[0]}\n({s[1] - s[2]} of {s[1]} ep)" for s in shapes], fontsize=8)
ax.set_title("(c) episodes that are not a safe success, by shape of the call record", fontsize=10, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.08), ncol=2)
fig.suptitle("-qs candidate vs current config: collisions go down on every suite, task success does not go up", fontsize=12, color=figstyle.INK)
fig.tight_layout()
fig.savefig(out, dpi=160, facecolor=fig.get_facecolor())
json.dump({"totals": {k: {"n": v["n"], "current_TSR_CAR_SSR": v["current"], "qs_TSR_CAR_SSR": v["-qs"], "reference_pct": v["ref"], "reference": v["refname"]} for k, v in TOT.items()},
           "cells": {c: {"current_TSR_CAR_SSR": CELL["current"][i], "qs_TSR_CAR_SSR": CELL["-qs"][i], "n": 8} for i, c in enumerate(CELLS)},
           "shapes": [{"set": s[0], "n": s[1], "safe_success": s[2], **{g[0]: s[3][i] for i, g in enumerate(GROUPS)}} for s in shapes],
           "source": "sota_compare.py ms pe123458 pkgFqs; offline/shape.py (shape_*.json); plan §9.13 / §9.16"},
          open(out.rsplit(".", 1)[0] + ".json", "w"), ensure_ascii=False, indent=1)
print(out, shapes)
