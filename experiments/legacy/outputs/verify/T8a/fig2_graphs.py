"""T8a figure 2 — chunk 축 × link 별. 세 계열을 겹친다.

계열 (모두 clearance = 표면거리 - 구 반지름, chunk 안의 최소):
  apple only   — 그 link 의 구가 사과 하나만 상대로
  apple removed— 사과를 뺀 나머지 body 전부 (table·crate·banana·orange·pear)
  table only   — table 하나만

가로 점선은 margin 10 mm (지금 서버), 실선 0 mm, 옅은 점선 50 mm (T7a 가 쓴 값).
위 줄 = authorized 세 link (M1 이 고른 것), 아래 줄 = 대조 link 셋.
오른쪽 끝 칸은 M3: 사과가 음수인 질의점에서 같은 점의 table 거리 분포.
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

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

T8A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8a/t8a-graphs.png'))
SIDE = OUT.with_suffix(".json")
MARGIN = 10.0
TOP = ("ee_finger_l1", "ee_finger_l2", "link_left_arm_5")
BOTTOM = ("link_left_arm_6", "link_left_arm_4", "link_left_arm_3")


def main() -> None:
    z = np.load(T8A / "m1m2m3_curves.npz", allow_pickle=True)
    links = [str(x) for x in z["link_axis"]]
    ana = json.loads((T8A / "m1m2m3_analysis.json").read_text())
    m3 = ana["m3_table_distance_at_apple_negative_points"]
    ap, na = z["apple_only_full50"], z["no_apple_full50"]
    tb, cr = z["table_only_full50"], z["crate_only_full50"]
    F = ap.shape[0]
    x = np.arange(F)

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 4, figsize=(19.2, 9.0))
    figstyle.style_axes(fig, axes)
    side: dict = {"margin_mm": MARGIN, "window": "full horizon 50 steps, min within chunk",
                  "source_curves": str(T8A / "m1m2m3_curves.npz"), "panels": []}

    for r, group in enumerate((TOP, BOTTOM)):
        for c, ln in enumerate(group):
            ax = axes[r, c]
            li = links.index(ln)
            ax.plot(x, ap[:, li], color=figstyle.CATEGORICAL[1], lw=1.9, label="apple only")
            ax.plot(x, na[:, li], color=figstyle.CATEGORICAL[0], lw=1.6,
                    label="apple removed (all other bodies)")
            ax.plot(x, tb[:, li], color=figstyle.CATEGORICAL[2], lw=1.3, ls="-.",
                    label="table only")
            ax.plot(x, cr[:, li], color=figstyle.CATEGORICAL[4], lw=1.0, ls=":",
                    label="crate only")
            ax.axhline(0.0, color=figstyle.INK, lw=1.0)
            ax.axhline(MARGIN, color=figstyle.CATEGORICAL[7], lw=1.1, ls="--")
            ax.axhline(50.0, color=figstyle.GRID_INK, lw=1.0, ls=":")
            bad = ap[:, li] <= MARGIN
            if bad.any():
                ax.fill_between(x, -200, 400, where=bad, color=figstyle.CATEGORICAL[1],
                                alpha=0.07, step=None)
            e = ana["m1_apple_only_per_link"]["full50"][ln]
            ax.set_title(
                f"{ln}   (r {'/'.join(f'{v:.1f}' for v in e['sphere_radius_mm'])} mm, "
                f"{e['n_spheres']} spheres)\n"
                f"apple-only min {e['min_clearance_mm']:+.2f} mm · chunks<=0 at 0/10/50 mm: "
                f"{e['n_chunks_slack_le0_at_0mm']}/{e['n_chunks_slack_le0_at_10mm']}/"
                f"{e['n_chunks_slack_le0_at_50mm']} of {F}",
                fontsize=8.6, color=figstyle.INK)
            ax.set_ylim(-110, 330)
            ax.set_xlim(0, F - 1)
            ax.set_xlabel("chunk (shadow, 75 planning records)", fontsize=8.5,
                          color=figstyle.INK_2)
            if c == 0:
                ax.set_ylabel("clearance  [mm]   (min within chunk)", fontsize=8.5,
                              color=figstyle.INK_2)
            if r == 0 and c == 0:
                ax.legend(fontsize=7.0, loc="lower right", framealpha=0.9)
            side["panels"].append({
                "link": ln, "apple_only_mm": [round(float(v), 4) for v in ap[:, li]],
                "apple_removed_mm": [round(float(v), 4) for v in na[:, li]],
                "table_only_mm": [round(float(v), 4) for v in tb[:, li]],
                "crate_only_mm": [round(float(v), 4) for v in cr[:, li]],
                "m1_full50": e,
            })

    # --- 오른쪽 위: M2 남은 행 · chunk 별 최소 slack -------------------------------------
    ax = axes[0, 3]
    m2 = ana["m2_apple_removed_remaining_rows_at_10mm"]["full50"]
    y = np.asarray(m2["per_chunk_min_slack_mm"], np.float64)
    ax.bar(x, y, color=np.where(y <= 0, figstyle.CATEGORICAL[7], figstyle.CATEGORICAL[0]),
           width=0.85)
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.set_title(f"M2 - apple rows removed for the 3 authorized links\n"
                 f"min slack over remaining rows, margin {MARGIN:.0f} mm\n"
                 f"{m2['n_chunks_with_any_slack_le0']}/{F} chunks <= 0 · "
                 f"{m2['n_rows_slack_le0']}/{m2['n_rows_total']} rows",
                 fontsize=8.6, color=figstyle.INK)
    ax.set_xlabel("chunk", fontsize=8.5, color=figstyle.INK_2)
    ax.set_ylabel("min margin_slack  [mm]", fontsize=8.5, color=figstyle.INK_2)
    ax.set_xlim(-0.5, F - 0.5)

    # --- 오른쪽 아래: M3 분포 ------------------------------------------------------------
    ax = axes[1, 3]
    pc = m3["per_chunk"]
    cx = [p["chunk"] for p in pc]
    ax.plot(cx, [p["min_apple_clearance_mm"] for p in pc], color=figstyle.CATEGORICAL[1],
            marker="o", ms=3, lw=1.5, label="apple (removed row)")
    ax.plot(cx, [p["min_table_clearance_mm"] for p in pc], color=figstyle.CATEGORICAL[2],
            marker="s", ms=3, lw=1.5, label="table at the same query point")
    ax.plot(cx, [p["min_crate_clearance_mm"] for p in pc], color=figstyle.CATEGORICAL[4],
            marker="^", ms=3, lw=1.2, ls=":", label="crate at the same query point")
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.axhline(MARGIN, color=figstyle.CATEGORICAL[7], lw=1.1, ls="--")
    ax.set_title(f"M3 - {m3['n_query_points']} query points where an authorized\n"
                 f"link is <= 0 vs the apple at margin {MARGIN:.0f} mm\n"
                 f"table min {m3['table']['min_clearance_mm']:+.2f} · "
                 f"median {m3['table']['median_clearance_mm']:+.2f} mm",
                 fontsize=8.6, color=figstyle.INK)
    ax.set_xlabel("chunk (only chunks with such a query point)", fontsize=8.5,
                  color=figstyle.INK_2)
    ax.set_ylabel("clearance  [mm]   (min within chunk)", fontsize=8.5, color=figstyle.INK_2)
    ax.legend(fontsize=7.4, loc="upper left", framealpha=0.9)
    side["m2_per_chunk_min_slack_mm"] = m2["per_chunk_min_slack_mm"]
    side["m3_per_chunk"] = pc

    fig.suptitle("T8a - shadow actions_reference, clearance per link across chunks: "
                 "apple only / apple removed / table only", fontsize=12.5, color=figstyle.INK,
                 y=0.985)
    fig.text(0.5, 0.010,
             "solid black line = 0 mm contact · dashed line = margin 10 mm (current server "
             "--esdf-margin 0.01) · dotted grey = 50 mm (the margin T7a used) · shaded band = "
             "chunks whose apple-only clearance is at or below the 10 mm margin",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.028, 1, 0.962))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
