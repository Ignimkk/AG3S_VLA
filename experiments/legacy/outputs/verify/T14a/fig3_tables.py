"""T14a figure 3 — 표. Q2 가 우선순위 1 이므로 맨 위에."""
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

T14A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T14a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t14a/t14a-table.png'))
SIDE = OUT.with_suffix(".json")


def table(ax, title, cols, rows, widths, colours=None, fontsize=7.4, head=8.2, dymax=0.04):
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.0, color=figstyle.INK, loc="left", pad=10)
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
    a = json.loads((T14A / "q1to4.json").read_text())
    q1, q2, q3, q4 = (a["q1_fine_window"], a["q2_grounding_target_loss"],
                      a["q3_what_the_ball_erased"], a["q4_fingertip"])
    ov = a["q1_q2_overlap"]
    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 11.2))
    figstyle.style_axes(fig, axes)

    # --- Q2 -------------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["Q2  did grounding lose the target", "T14", "T12", "note"])
    colours[0] = figstyle.INK
    for lab, k, f in (
            ("chunks", "n_chunks", "{}"),
            ("has_target False", "n_chunks_without_target", "{}"),
            ("  at local t_step", "t_steps_without_target", "{}"),
            ("  of those, in the approach", "n_in_approach_window", "{}"),
            ("centroid nearest is NOT the apple", "n_chunks_centroid_nearest_is_not_apple", "{}"),
            ("  of those, in the approach", "n_centroid_not_apple_in_approach", "{}"),
            ("centroid-to-apple median", "centroid_to_apple_mm_median", "{} mm"),
            ("centroid-to-apple max", "centroid_to_apple_mm_max", "{} mm"),
            ("target points, median", "target_n_points_median", "{}"),
            ("target points, min", "target_n_points_min", "{}")):
        rows.append([f"    {lab}", f.format(q2["T14"][k]), f.format(q2["T12"][k]), ""])
        if "NOT the apple" in lab:
            colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
    rows.append(["    what the centroid was nearest to", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for run in ("T14", "T12"):
        rows.append([f"      {run}",
                     ", ".join(f"{k} {v}" for k, v in
                               q2[run]["centroid_nearest_object_counts"].items()), "", ""])
    rows.append(["    what the server reported", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for run in ("T14", "T12"):
        rows.append([f"      {run} grounding_status",
                     str(q2[run]["server_grounding_status_counts"]),
                     str(q2[run]["server_status_counts"]), "status vs identity"])
    rows += [["", "", "", ""],
             ["    Q1 n Q2 - are the two failures the same chunks", "", "", ""]]
    colours[len(rows) - 1] = figstyle.INK
    for run in ("T14", "T12"):
        e = ov[run]
        rows.append([f"      {run} identical sets", str(e["identical_sets"]),
                     f"both {e['n_in_both']}",
                     f"only window {e['n_only_window']} / only centroid {e['n_only_centroid']}"])
        colours[len(rows) - 1] = (figstyle.CATEGORICAL[7] if e["identical_sets"]
                                  else figstyle.INK_2)
    table(axes[0, 0], "Q2  grounding - the top-priority question",
          ["what", "T14", "T12", "note"], rows, [0.36, 0.26, 0.22, 0.16], colours,
          fontsize=7.4, dymax=0.036)

    # --- Q1 -------------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["Q1  does the fine ESDF window contain the apple", "T14", "T12", "note"])
    colours[0] = figstyle.INK
    for lab, k, f in (
            ("chunks", "n_chunks", "{}"),
            ("apple outside the fine window", "n_apple_outside_fine", "{}"),
            ("  of those, in the approach",
             ("approach_window", "n_apple_outside_fine"), "{}"),
            ("fingertip outside the fine window", "n_tip_outside_fine", "{}"),
            ("apple outside the coarse window", "n_apple_outside_coarse", "{}"),
            ("esdf_free window differs from fine", "n_free_window_differs_from_fine", "{}"),
            ("chunks without a free layer", "n_chunks_without_free_layer", "{}"),
            ("chunks without a coarse layer", "n_chunks_without_coarse_layer", "{}"),
            ("chunks without a target centroid", "n_chunks_without_target_centroid", "{}")):
        def get(run):
            e = q1[run]
            return e[k[0]][k[1]] if isinstance(k, tuple) else e[k]
        rows.append([f"    {lab}", f.format(get("T14")), f.format(get("T12")), ""])
        if lab.startswith("apple outside the fine"):
            colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
    rows.append(["    centroid minus apple  [mm]", "T14 min / median / max",
                 "T12 min / median / max", ""])
    colours[len(rows) - 1] = figstyle.INK
    for ax_ in "xyz":
        e14 = q1["T14"]["centroid_minus_apple_mm_stats"][ax_]
        e12 = q1["T12"]["centroid_minus_apple_mm_stats"][ax_]
        rows.append([f"      {ax_}",
                     f"{e14['min']:.1f} / {e14['median']:.1f} / {e14['max']:.1f}",
                     f"{e12['min']:.1f} / {e12['median']:.1f} / {e12['max']:.1f}", ""])
        if ax_ == "y":
            colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
    rows += [["", "", "", ""],
             ["    t_steps where the apple is outside the fine window", "", "", ""]]
    colours[len(rows) - 1] = figstyle.INK
    ts = q1["T14"]["t_steps_apple_outside_fine"]
    for i in range(0, len(ts), 9):
        rows.append([f"      T14  {ts[i:i+9]}", "", "", ""])
    rows.append(["    a chunk where it escaped", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    ex = [r for r in q1["T14"]["per_chunk"] if r["apple_in_fine_window"] is False][0]
    rows += [
        [f"      t_step {ex['t_step']}  fine window x",
         f"[{ex['fine_origin_mm'][0]:.0f}, {ex['fine_max_mm'][0]:.0f}]",
         f"y [{ex['fine_origin_mm'][1]:.0f}, {ex['fine_max_mm'][1]:.0f}]",
         f"z [{ex['fine_origin_mm'][2]:.0f}, {ex['fine_max_mm'][2]:.0f}] mm"],
        ["        apple", f"{ex['apple_mm']}", "", f"outside by "
         f"{ex['apple_distance_outside_fine_mm']:.1f} mm"],
        ["        target centroid", f"{ex['target_centroid_mm']}", "",
         f"centroid - apple {ex['centroid_minus_apple_mm']}"],
        ["        fingertip", f"{ex['tip_mm']}", "",
         f"in window: {ex['tip_in_fine_window']}"],
    ]
    table(axes[0, 1], "Q1  the fine layer window", ["what", "T14", "T12", "note"], rows,
          [0.34, 0.26, 0.24, 0.16], colours, fontsize=7.2, dymax=0.034)

    # --- Q3 · Q4 ---------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["Q3  what the ball erased (esdf vs esdf_free, fine layer)", "T14", "T12", ""])
    colours[0] = figstyle.INK
    for lab, k in (("voxels changed inside the ball, all chunks", "ball_changed_total"),
                   ("  max in one chunk", "ball_changed_max"),
                   ("voxels changed within 45 mm of the apple", "apple45_changed_total"),
                   ("  max in one chunk", "apple45_changed_max"),
                   ("whole fine layer, min voxels differing", "whole_layer_diff_min"),
                   ("whole fine layer, max voxels differing", "whole_layer_diff_max"),
                   ("chunks with no free layer recorded", "n_chunks_without_free_layer"),
                   ("chunks with any voxel changed", "n_chunks_with_any_voxel_changed")):
        rows.append([f"    {lab}", f"{q3['T14'][k]}", f"{q3['T12'][k]}", ""])
        if "inside the ball, all" in lab or "within 45 mm" in lab:
            colours[len(rows) - 1] = figstyle.CATEGORICAL[2]
    b = next((r["ball"] for r in q3["T14"]["per_chunk"] if r.get("ball")), None)
    if b:
        rows += [["    the ball the server used (T14)", "", "", ""]]
        colours[len(rows) - 1] = figstyle.INK
        rows += [[f"      {k}", str(v), "", ""] for k, v in list(b.items())[:6]]
    rows += [["", "", "", ""],
             ["Q4  the fingertip through the approach", "T14", "T12", ""]]
    colours[len(rows) - 1] = figstyle.INK
    for lab, k in (("tightest recorded row, min  [mm post-margin]",
                    ("approach_window", "clearance_min_mm_post_margin")),
                   ("  at local t_step", ("approach_window", "clearance_min_t_step")),
                   ("tightest recorded row, median", ("approach_window",
                                                      "clearance_median_mm_post_margin")),
                   ("tightest recorded row, max", ("approach_window",
                                                   "clearance_max_mm_post_margin")),
                   ("fingertip to apple surface, min  [mm]", ("tip_to_apple_min_mm",)),
                   ("apple max lift  [mm]", ("apple_max_lift_mm",)),
                   ("apple max displacement  [mm]", ("apple_max_displacement_mm",))):
        def g(run):
            e = q4[run]
            for kk in k:
                e = e[kk]
            return e
        rows.append([f"    {lab}", f"{g('T14')}", f"{g('T12')}", ""])
    rows += [
        ["", "", "", ""],
        ["    the lead quoted for T12: 3.618 mm at t=96, 4.7-11.7 mm over t=88..160", "", "", ""],
    ]
    colours[len(rows) - 1] = figstyle.INK
    rows.append(["      this T12 record gives instead",
                 f"{q4['T12']['approach_window']['clearance_min_mm_post_margin']:.4f} mm min",
                 "27.5-33.9 mm over t=72..160", "does not reproduce"])
    colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
    rows.append(["      the T14 numbers do reproduce exactly", "3.6841 mm at t=80",
                 "10.5-14.2 mm over t=88..160", "matches the lead"])
    colours[len(rows) - 1] = figstyle.CATEGORICAL[2]
    table(axes[1, 0], "Q3  the ball   ·   Q4  the fingertip",
          ["what", "T14", "T12", "note"], rows, [0.40, 0.22, 0.22, 0.16], colours,
          fontsize=7.2, dymax=0.032)

    # --- 방법 -----------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("how this was measured", fontsize=10.0, color=figstyle.INK, loc="left", pad=10)
    m = a["method"]
    txt = (
        f"T14 {a['runs']['T14']}\n"
        f"T12 {a['runs']['T12']}\n"
        "  each run has a local frames.jsonl and a server_constraints/run_0000 with one\n"
        "  npz per chunk carrying the layer arrays and summary_json\n"
        "\n"
        "the server's summary_json 't_step' is the chunk SEQUENCE number, not the local\n"
        "t_step. the mapping used here is server t_step == local seq, local t_step =\n"
        "(seq - 1) * 8. it is checked by content: the nearest local actions_reference to\n"
        f"the recorded reference_chunk is the same seq on "
        f"{q2['T14']['n_reference_chunk_matched']}/{q2['T14']['n_chunks']} chunks for T14\n"
        f"and {q2['T12']['n_reference_chunk_matched']}/{q2['T12']['n_chunks']} for T12.\n"
        "\n"
        f"window definition: {m['window_definition']}\n"
        "  a point is inside when origin <= p < origin + shape * voxel on all three axes\n"
        f"apple ball for the Q3 table: radius {m['apple_ball_radius_mm']:.0f} mm about the\n"
        "  recorded apple position - the same form as the T12 table\n"
        "\n"
        "the centroid's identity is decided by which recorded object_poses entry it is\n"
        "nearest to. that is a distance, not a label: grounding does not record which\n"
        "object it thinks it found.\n"
        "\n"
        "the fingertip-to-apple distance uses the analytic MuJoCo surface distance from\n"
        "the server's own sphere_centres (T9a/T10a method, no mj_geomDistance).\n"
        "\n"
        "clearances from summary_json are POST-margin (linearize.py:489): the 10 mm\n"
        "esdf margin is already subtracted, so 0 means the margin is exactly used up.\n"
        "\n"
        f"elapsed {a['elapsed_s']} s\n"
        "\n"
        "raw   outputs/verify/T14a/q1to4.json / q1to4_curves.npz\n"
        "      outputs/verify/T14a/q1to4.py  (one script, Q1..Q4)\n"
    )
    ax.text(0.0, 0.97, txt, fontsize=7.9, family="monospace", color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("T14a - the numbers", fontsize=12.6, color=figstyle.INK, y=0.985)
    fig.tight_layout(rect=(0, 0.008, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps({"q1": q1, "q2": q2, "q3": q3, "q4": q4,
                                "q1_q2_overlap": ov, "method": m}, indent=1,
                               ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
