#!/usr/bin/env python3
"""Planning figure (not for the manuscript): reported SafeLIBERO TSR vs CAR per paper, plus our dev-cell status.

Input: paper/evidence/reported_baselines_20261111/reported_safelibero_metrics.json
       (from paper/tools/analysis/reported_safelibero_metrics.py).
Output: paper/evidence/reported_baselines_20261111/reported_tsr_car.png
Panel A: Level I, four-suite mean (REPORTED / DERIVED).  Panel B: levels and suites pooled.
Our points are MEASURED on the development cell only (Spatial Level I, init 5-14, n = 40) and are
drawn hollow because they are not on the same split or suite mix as the reported values.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "evidence" / "reported_baselines_20261111"
data = json.loads((D / "reported_safelibero_metrics.json").read_text())

# colour by obstacle-information class
C_PERC, C_GT, C_BASE, C_TRAIN, C_OURS = "#1f77b4", "#d62728", "#7f7f7f", "#9467bd", "#2ca02c"

l1 = [
    ("π0.5 [VLSA]", data["vlsa_full_action"]["pi05"]["per_level_4suite_mean"]["L1"], C_BASE),
    ("AEGIS [VLSA]", data["vlsa_full_action"]["aegis"]["per_level_4suite_mean"]["L1"], C_PERC),
    ("π0.5 [KNOWS]", data["knows"]["no_cbf"]["per_level_4suite_mean"]["L1"], C_BASE),
    ("KNOWS", data["knows"]["knows"]["per_level_4suite_mean"]["L1"], C_PERC),
    ("Naive GT seg [KNOWS]", data["knows"]["naive_gt"]["per_level_4suite_mean"]["L1"], C_GT),
    ("CFM (sim GT)", data["cfm"]["cfm"]["per_level_4suite_mean"]["L1"], C_GT),
]
pooled = [
    ("π0.5 [multi-link]", data["multilink"]["pi05"]["pooled"], C_BASE),
    ("AEGIS [multi-link rerun]", data["multilink"]["aegis_authors_code_qwen"]["pooled"], C_PERC),
    ("Multi-link (contemp.)", data["multilink"]["multilink_full"]["pooled"], C_PERC),
    ("AEGIS [VLSA]", data["vlsa_full_action"]["aegis"]["pooled"], C_PERC),
    ("VLPSA, GT labels (contemp.)", data["vlpsa"]["vlpsa"]["pooled"], C_GT),
    ("VLSA* [VLPSA]", data["vlpsa"]["vlsa_star"]["pooled"], C_GT),
    ("SafeHarness GPT-5.5 (contemp.)", data["safeharness"]["gpt55"]["all"], C_TRAIN),
]
# MEASURED, development cell (Spatial L1, init 5-14, n = 40): AG3S_PLAN_SCHEDULE_20261110_KO.md §3 table
ours_dev = [("π0.5, ours\n(dev cell) SSR 15.0", 62.5, 15.0, C_BASE, (6, -4)),
            ("AG3S pe123458 (dev cell)\nSSR 37.5", 55.0, 57.5, C_OURS, (6, -16)),
            ("AG3S B5, GT (dev cell)\nSSR 32.5", 55.0, 52.5, C_GT, (-70, -24))]
# per-label offsets (points) to avoid overlap; default (4, 3)
OFFS = {"π0.5 [KNOWS]": (4, 4), "π0.5 [VLSA]": (4, -10), "AEGIS [multi-link rerun]": (-118, -6), "Multi-link (contemp.)": (6, -16),
        "AEGIS [VLSA]": (-62, 4), "VLSA* [VLPSA]": (4, 6), "π0.5 [multi-link]": (4, 2)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
for ax, rows, title, dev in ((axes[0], l1, "A. Level I, four-suite mean (reported)", True),
                             (axes[1], pooled, "B. Levels and suites pooled (reported)", False)):
    for name, v, c in rows:
        ax.scatter(v["CAR"], v["TSR"], s=46, color=c, zorder=3)
        lab = name + (f"\nSSR {v['SSR']:.1f}" if v.get("SSR") is not None else "")
        ax.annotate(lab, (v["CAR"], v["TSR"]), xytext=OFFS.get(name, (4, 3)), textcoords="offset points", fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("CAR (%)")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(40, 90)
    if not dev:
        continue
    for name, tsr, car, c, off in ours_dev:
        ax.scatter(car, tsr, s=46, facecolors="none", edgecolors=c, linewidths=1.6, zorder=3)
        ax.annotate(name, (car, tsr), xytext=off, textcoords="offset points", fontsize=7, color=c)
axes[0].set_ylabel("TSR (%)")
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in (
    (C_BASE, "π0.5 alone"), (C_PERC, "perception (no GT)"), (C_GT, "GT obstacle info"),
    (C_TRAIN, "LLM agent"), (C_OURS, "AG3S"))]
handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="hollow = our dev cell, Spatial L-I only (MEASURED, n=40)"))
fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(D / "reported_tsr_car.png", dpi=150)
print(D / "reported_tsr_car.png")
