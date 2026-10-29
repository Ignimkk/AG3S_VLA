"""T8e figure 3 — s x spacing 격자 heatmap. 남은 위반.

칸 색 = 남은 위반 행 수 (log 눈금, 단일 파랑 램프). 칸 글자 = 행 수 · chunk 수 · 최소 slack mm.
0 인 칸은 테두리로 표시한다. 위 줄 exec8, 아래 줄 full50. 열은 손바닥 세 판.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

T8E = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8e'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8e/t8e-global-grid.png'))
SIDE = OUT.with_suffix(".json")
PALMS = (("out", "palm not in the model"),
         ("in", "palm in, apple KEPT for ee_left"),
         ("in_authorized", "palm in, apple REMOVED for ee_left"))


def main() -> None:
    d = json.loads((T8E / "s2_global_scale_sweep.json").read_text())
    scales, spacings = d["scales"], d["spacings"]
    cells = {(c["sphere_spacing"], c["capsule_radius_scale"], c["palm"]): c for c in d["grid"]}

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 3, figsize=(18.6, 9.2))
    figstyle.style_axes(fig, axes)
    cmap = figstyle.sequential_cmap() if hasattr(figstyle, "sequential_cmap") else "Blues"
    side: dict = {"z_cut_m": d["condition"]["z_cut_m"], "margin_mm": d["condition"]["margin_mm"],
                  "scales": scales, "spacings": spacings, "cells": []}

    for r, w in enumerate(("exec8", "full50")):
        for c, (ptag, plab) in enumerate(PALMS):
            ax = axes[r, c]
            M = np.zeros((len(spacings), len(scales)))
            for i, sp in enumerate(spacings):
                for j, sc in enumerate(scales):
                    M[i, j] = cells[(sp, sc, ptag)]["windows"][w]["n_rows_slack_le0"]
            im = ax.imshow(np.where(M > 0, M, np.nan), cmap=cmap, aspect="auto",
                           norm=LogNorm(vmin=1, vmax=max(M.max(), 2)))
            for i, sp in enumerate(spacings):
                for j, sc in enumerate(scales):
                    cell = cells[(sp, sc, ptag)]
                    v = cell["windows"][w]
                    ok = v["feasible_at_10mm"]
                    if ok:
                        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               facecolor=figstyle.SURFACE,
                                               edgecolor=figstyle.CATEGORICAL[2], lw=2.4))
                    txt = (f"{v['n_rows_slack_le0']}\n{v['n_chunks_slack_le0']} chk\n"
                           f"{v['global_min_slack_mm']:+.2f}")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7.6,
                            family="monospace",
                            color=(figstyle.CATEGORICAL[2] if ok else
                                   (figstyle.SURFACE if M[i, j] > 0.35 * M.max()
                                    else figstyle.INK)))
                    if r == 0 and c == 0:
                        side["cells"].append({
                            "sphere_spacing": sp, "capsule_radius_scale": sc,
                            "n_spheres_model": cell["n_spheres_model"],
                            "n_spheres_used_after_cut": cell["n_spheres_used_after_cut"],
                            "radius_mm_link_left_arm_5":
                                cell["radius_mm_by_link"]["link_left_arm_5"],
                            "radius_mm_ee_finger_l1":
                                cell["radius_mm_by_link"]["ee_finger_l1"],
                            "worst_shortfall_mm": cell["worst_shortfall_mm"],
                            "n_coverage_shortfall": cell["n_coverage_shortfall"]})
            ax.set_xticks(range(len(scales)))
            ax.set_xticklabels([f"x{s:g}" for s in scales], fontsize=8.5)
            ax.set_yticks(range(len(spacings)))
            ax.set_yticklabels(
                [f"{sp:g}\n({cells[(sp, 1.0, ptag)]['n_spheres_model']} spheres)"
                 for sp in spacings], fontsize=8.0)
            ax.set_xlabel("global capsule_radius_scale", fontsize=8.5, color=figstyle.INK_2)
            if c == 0:
                ax.set_ylabel("sphere_spacing", fontsize=8.5, color=figstyle.INK_2)
            ax.set_title(f"[{w}]  {plab}", fontsize=9.6, color=figstyle.INK)
            for sp in ax.spines.values():
                sp.set_visible(False)

    fig.suptitle("T8e - global capsule_radius_scale x sphere_spacing, z_cut -100 mm: "
                 "rows still violating at margin 10 mm",
                 fontsize=12.5, color=figstyle.INK, y=0.978)
    fig.text(0.5, 0.012,
             "each cell prints rows / chunks / min margin_slack (mm) · green box = 0 violations · "
             "blue ramp is a magnitude (log rows) · apple rows removed for ee_finger_l1, "
             "ee_finger_l2, link_left_arm_5 · max_spheres_per_capsule raised to 32 so the spacing "
             "actually acts (the default 8 clips it below spacing 0.5)",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.032, 1, 0.945))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
