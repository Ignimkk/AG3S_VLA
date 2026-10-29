"""T8a figure 3 — 표·수식.

왼쪽 위 : M1 — link 별 사과 하나만. margin 0 / 10 / 50 mm 세 값을 나란히.
왼쪽 아래: M3 — 사과가 음수인 질의점에서 같은 점의 table·crate 거리.
오른쪽 위: M2 — authorized link 에서 사과를 뺀 뒤 margin 10 mm 로 남는 (link, obstacle) 쌍.
오른쪽 아래: 행의 식과 이 측정이 쓴 숫자·경로.
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
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8a/t8a-table.png'))
SIDE = OUT.with_suffix(".json")
AUTH = ("ee_finger_l1", "ee_finger_l2", "link_left_arm_5")


def table(ax, title, cols, rows, widths, colours=None, fontsize=8.0, head=8.6, dymax=0.075):
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.2, color=figstyle.INK, loc="left", pad=10)
    x0 = np.concatenate([[0.0], np.cumsum(widths)])
    y = 0.95
    dy = min(dymax, 0.90 / max(len(rows) + 1, 1))
    for j, c in enumerate(cols):
        ax.text(x0[j], y, c, fontsize=head, color=figstyle.INK, va="top", family="monospace")
    ax.plot([0, 1], [y - dy * 0.45] * 2, color=figstyle.GRID_INK, lw=1.0)
    for i, row in enumerate(rows):
        yy = y - dy * (i + 1)
        col = (colours or {}).get(i)
        for j, cell in enumerate(row):
            ax.text(x0[j], yy, str(cell), fontsize=fontsize, family="monospace",
                    color=col or (figstyle.INK if j == 0 else figstyle.INK_2), va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main() -> None:
    ana = json.loads((T8A / "m1m2m3_analysis.json").read_text())
    meta = json.loads((T8A / "m0_clearance_tensor.meta.json").read_text())
    m1 = ana["m1_apple_only_per_link"]["full50"]
    m2 = ana["m2_apple_removed_remaining_rows_at_10mm"]
    m3 = ana["m3_table_distance_at_apple_negative_points"]
    links = ana["shape"]["link_axis"]
    F = ana["shape"]["n_chunks"]

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.4))
    figstyle.style_axes(fig, axes)

    # --- M1 ---------------------------------------------------------------------------
    rows, colours = [], {}
    order = [l for l in links if m1[l]["n_chunks_slack_le0_at_50mm"] > 0] + \
            [l for l in links if m1[l]["n_chunks_slack_le0_at_50mm"] == 0]
    for i, ln in enumerate(order):
        e = m1[ln]
        rows.append([ln, f"{e['n_spheres']:2d}",
                     "/".join(f"{v:.1f}" for v in e["sphere_radius_mm"]),
                     f"{e['min_clearance_mm']:+9.2f}",
                     f"{e['n_chunks_slack_le0_at_0mm']:3d}  {e['n_rows_slack_le0_at_0mm']:6d}",
                     f"{e['n_chunks_slack_le0_at_10mm']:3d}  {e['n_rows_slack_le0_at_10mm']:6d}",
                     f"{e['n_chunks_slack_le0_at_50mm']:3d}  {e['n_rows_slack_le0_at_50mm']:6d}"])
        if ln in AUTH:
            colours[i] = figstyle.CATEGORICAL[1]
        elif e["n_chunks_slack_le0_at_50mm"] > 0:
            colours[i] = figstyle.CATEGORICAL[3]
    table(axes[0, 0],
          f"M1  apple only - per link, shadow actions_reference, all {F} chunks x 50 steps\n"
          f"     orange = slack <= 0 at margin 10 mm (authorized set) · "
          f"amber = only at margin 50 mm",
          ["link", "nS", "radius mm", "min clr mm", "m=0: chk rows", "m=10: chk rows",
           "m=50: chk rows"],
          rows, [0.20, 0.05, 0.13, 0.13, 0.17, 0.17], colours, fontsize=7.8, dymax=0.050)

    # --- M2 ---------------------------------------------------------------------------
    rows, colours = [], {}
    for w in ("full50", "exec8"):
        d = m2[w]
        rows.append([f"[{w}] all remaining", "",
                     f"rows {d['n_rows_slack_le0']}/{d['n_rows_total']}",
                     f"chunks {d['n_chunks_with_any_slack_le0']}/{F}",
                     f"min slack {d['global_min_slack_mm']:+.2f} mm"])
        colours[len(rows) - 1] = figstyle.INK
        for p in d["violating_pairs"]:
            rows.append(["   " + p["link"], p["obstacle_body"],
                         f"rows {p['n_rows_slack_le0']:5d}/{p['n_rows_total']}",
                         f"chunks {p['n_chunks_slack_le0']:3d}",
                         f"min clr {p['min_clearance_mm']:+8.2f}  "
                         f"slack {p['min_slack_mm']:+8.2f} mm"])
            colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
        rows.append(["", "", "", "", ""])
    c = ana["m2_contrast_full50_at_10mm"]
    rows.append(["[contrast full50]", "", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for tag, d in c.items():
        rows.append(["   " + tag, "", f"rows {d['n_rows_slack_le0']}",
                     f"chunks {d['n_chunks_with_any_slack_le0']}/{F}",
                     f"min slack {d['global_min_slack_mm']:+.2f} mm"])
    table(axes[0, 1],
          "M2  apple rows removed for " + ", ".join(AUTH) + " - every remaining row at margin "
          "10 mm\n     red = (link, obstacle) pairs that still have slack <= 0",
          ["scope / link", "obstacle", "rows <= 0", "chunks <= 0", "worst"],
          rows, [0.22, 0.11, 0.20, 0.14, 0.33], colours, fontsize=7.8, dymax=0.052)

    # --- M3 ---------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["all authorized", f"{m3['n_query_points']}",
                 f"{m3['n_chunks_with_any']}/{F}",
                 f"{m3['table']['min_clearance_mm']:+8.2f}",
                 f"{m3['table']['median_clearance_mm']:+8.2f}",
                 f"{m3['table']['n_query_points_slack_le0_at_10mm']}",
                 f"{m3['table']['n_query_points_below_0mm']}"])
    colours[0] = figstyle.INK
    for ln, e in m3["per_link"].items():
        rows.append(["   " + ln, f"{e['n_query_points']}", f"{e['n_chunks']}/{F}",
                     f"{e['min_table_clearance_mm']:+8.2f}",
                     f"{e['median_table_clearance_mm']:+8.2f}", "", ""])
        colours[len(rows) - 1] = figstyle.CATEGORICAL[1]
    rows.append(["", "", "", "", "", "", ""])
    for nm in ("crate", "min_over_non_apple_bodies"):
        e = m3[nm]
        rows.append([nm + " (same points)", "", "", f"{e['min_clearance_mm']:+8.2f}",
                     f"{e['median_clearance_mm']:+8.2f}",
                     f"{e['n_query_points_slack_le0_at_10mm']}",
                     f"{e['n_query_points_below_0mm']}"])
    table(axes[1, 0],
          "M3  at the query points where an authorized link is <= 0 against the apple "
          "(margin 10 mm)\n     how far is the SAME point from the table surface",
          ["selection", "points", "chunks", "min mm", "median mm", "<=10mm", "<0mm"],
          rows, [0.26, 0.10, 0.10, 0.13, 0.14, 0.12], colours, fontsize=8.0, dymax=0.075)

    # --- 식 ----------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("row definition and what this measurement used", fontsize=10.2,
                 color=figstyle.INK, loc="left", pad=10)
    txt = (
        "one row = (chunk c, plan step s, constraint sphere i, scene body b)\n"
        "\n"
        "    clearance(c,s,i,b) = min over collision geoms g of b\n"
        "                            [ surface_distance( centre_i(q_{c,s}), g ) ]  -  r_i\n"
        "    margin_slack       = clearance - margin            margin in {0, 10, 50} mm\n"
        "    violation          <=>  margin_slack <= 0\n"
        "\n"
        "    q_{c,s} : recorded qpos of chunk c for the torso, action row s of\n"
        "              actions_reference for the 14 arm joints  (shadow, executed = reference)\n"
        "    surface_distance : analytic. box exact; mesh = min distance to vertices\n"
        "                       (upper bound). mj_geomDistance NOT used -> no distmax saturation\n"
        "    scene bodies are frozen at the pose recorded with chunk c for all 50 steps\n"
        "\n"
        "    apple removal model : row is dropped iff  b == apple  AND  link(i) in A\n"
        "    A = { " + ", ".join(AUTH) + " }\n"
        "\n"
        f"    tensor  {meta['tensor_shape']}   spheres {meta['n_spheres']}  "
        f"scene geoms {meta['n_scene_geoms']}\n"
        f"    bodies  {', '.join(meta['bodies'])}\n"
        f"    rebuild time {meta['elapsed_s']} s\n"
        "\n"
        f"    T7a reproduction check (margin 50 mm, exec8, apple excluded):\n"
        f"        max |this - T7a s1_curves| = "
        f"{ana['t7a_reproduction_check']['max_abs_diff_mm']:.2e} mm over "
        f"{ana['t7a_reproduction_check']['n_cells']} cells\n"
        "\n"
        "    raw     outputs/verify/T8a/m0_clearance_tensor.npz\n"
        "            outputs/verify/T8a/m1m2m3_analysis.json\n"
        "    record  outputs/live_test/local_shadow_from_pc/frames.jsonl  (75 planning records)"
    )
    ax.text(0.0, 0.96, txt, fontsize=8.4, family="monospace", color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("T8a - does removing the apple make the successful shadow trajectory feasible",
                 fontsize=12.5, color=figstyle.INK, y=0.985)
    fig.tight_layout(rect=(0, 0.01, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(
        {"m1_full50": m1, "m1_authorized_link_set": ana["m1_authorized_link_set"],
         "m2": {w: {k: v for k, v in m2[w].items() if k != "per_chunk_min_slack_mm"}
                for w in m2},
         "m2_contrast_full50_at_10mm": ana["m2_contrast_full50_at_10mm"],
         "m3": {k: v for k, v in m3.items() if k != "per_chunk"},
         "t7a_reproduction_check": ana["t7a_reproduction_check"],
         "tensor_meta": meta}, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
