"""T17a figure 3 — 표. R2 가 우선순위 1 이므로 맨 위에."""
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

T17A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T17a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t17a/t17a-table.png'))
SIDE = OUT.with_suffix(".json")


def table(ax, title, cols, rows, widths, colours=None, fontsize=7.2, head=8.2, dymax=0.04):
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
    a = json.loads((T17A / "r1to4.json").read_text())
    r2, r3, r1, r4 = (a["r2_what_surface"], a["r3_if_not_the_apple"],
                      a["r1_what_to_pushed"], a["r4_gripper_close"])
    ov = a["r2_r3_overlap"]
    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 11.2))
    figstyle.style_axes(fig, axes)

    # --- R2/R3 per-chunk ------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["t_step", "link        clearance", "ESDF d | ground truth apple / crate",
                 "surface the field points at"])
    colours[0] = figstyle.INK
    for r in r3["rows"]:
        gt = r["gt_distance_by_body_mm"]
        rows.append([f"  {r['t_step']:>4d}",
                     f"{r['mvp_link']:<13s}{r['clearance_mm_post_margin']:+8.2f}",
                     f"{r['esdf_distance_mm']:7.2f} | {gt.get('apple', float('nan')):7.2f} / "
                     f"{gt.get('crate', float('nan')):7.2f}",
                     f"{r['esdf_surface_body']} at "
                     + ("n/a" if r["esdf_surface_body_distance_mm"] is None
                        else f"{r['esdf_surface_body_distance_mm']:.2f} mm")
                     + f"  [{r['esdf_gradient_tier']}]"])
        colours[len(rows) - 1] = (figstyle.CATEGORICAL[7] if r["esdf_surface_body"] == "apple"
                                  else figstyle.CATEGORICAL[0])
    rows += [["", "", "", ""],
             ["  surface counts", str(r2["esdf_surface_body_counts"]), "", ""],
             ["  query point inside the fine layer",
              f"{r2['n_point_in_fine_window']} of {r2['n_rows_with_mvp']}",
              f"inside the coarse layer {r2['n_point_in_coarse_window']}", ""],
             ["  gradient walk used which tier", str(r2["esdf_gradient_tier_counts"]), "", ""],
             ["  ESDF minus the true apple surface, on the apple rows",
              str(r2["esdf_minus_gt_apple_mm_on_apple_rows"]), "mm", ""]]
    colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
    table(axes[0, 0], "R2 / R3  what the tightest row was bound to, chunk by chunk",
          ["t_step", "link / clearance", "distances  [mm]", "surface"], rows,
          [0.10, 0.27, 0.34, 0.29], colours, fontsize=7.0, dymax=0.036)

    # --- R2 grounding ----------------------------------------------------------------------
    rows, colours = [], {}
    g = r2["grounding"]
    rows.append(["R2  was the exclusion in place during the approach", "", "", ""])
    colours[0] = figstyle.INK
    for lab, v in (("approach chunks", a["n_approach_chunks"]),
                   ("centroid's nearest object is the apple", g["n_centroid_nearest_is_apple"]),
                   ("centroid's nearest object counts",
                    str(g["centroid_nearest_object_counts"])),
                   ("centroid to apple, median mm", g["centroid_to_apple_mm_median"]),
                   ("apple inside the fine window", g["n_apple_in_fine_window"]),
                   ("exclusion ball contains the apple centre",
                    g["n_ball_contains_apple_centre"]),
                   ("ball radius, median mm", g["ball_radius_mm_median"]),
                   ("ball centre to apple, median mm", g["ball_centre_to_apple_mm_median"])):
        rows.append([f"    {lab}", str(v), "", ""])
    rows += [["", "", "", ""],
             ["    the two regimes", "t <= 64", "t >= 72", ""]]
    colours[len(rows) - 1] = figstyle.INK
    rows += [
        ["      ball radius mm",
         f"{min(ov['ball_radius_mm_when_centroid_is_apple']):.1f}"
         f"-{max(ov['ball_radius_mm_when_centroid_is_apple']):.1f}",
         f"{min(ov['ball_radius_mm_when_centroid_is_not_apple']):.1f}"
         f"-{max(ov['ball_radius_mm_when_centroid_is_not_apple']):.1f}", ""],
        ["      centroid's nearest object", "apple", "crate", ""],
        ["      apple inside the fine window", "yes", "no", ""],
        ["      ball contains the apple centre", "yes", "no", ""],
        ["      what bound the tightest row", "crate", "apple on 5 of 7, crate on 2", ""],
    ]
    for i in range(5):
        colours[len(rows) - 1 - i] = figstyle.CATEGORICAL[7]
    rows += [["", "", "", ""],
             ["    R2 n R3 - same chunks?", "", "", ""]]
    colours[len(rows) - 1] = figstyle.INK
    rows += [
        ["      surface is the apple", str(ov["t_steps_surface_is_apple"]), "", ""],
        ["      centroid is not the apple", str(ov["t_steps_centroid_not_apple"]), "", ""],
        ["      apple not in the fine window", str(ov["t_steps_apple_not_in_fine_window"]),
         "", ""],
        ["      ball misses the apple centre", str(ov["t_steps_ball_misses_apple_centre"]),
         "", ""],
        ["      apple-surface rows are a subset of centroid-not-apple",
         str(ov["surface_apple_is_subset_of_centroid_not_apple"]), "", ""],
    ]
    colours[len(rows) - 1] = figstyle.CATEGORICAL[2]
    rows += [["", "", "", ""],
             ["    esdf_free minus esdf at the query point", "", "", ""]]
    colours[len(rows) - 1] = figstyle.INK
    e = r2["esdf_free_minus_esdf_at_point_mm"]
    rows.append([f"      rows measurable / nonzero", f"{e['n_rows']} / {e['n_nonzero']}",
                 f"max |.| {e['max_abs']} mm",
                 "the free layer is fine-only (T14a)"])
    table(axes[0, 1], "R2  grounding, the exclusion ball, and the window",
          ["what", "value", "value", "note"], rows, [0.40, 0.24, 0.22, 0.14], colours,
          fontsize=7.2, dymax=0.032)

    # --- R1 ---------------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["R1  how hard TO pushed in the approach (shadow: computed, not executed)",
                 "", "", ""])
    colours[0] = figstyle.INK
    rows += [
        ["    max |TO - reference|, median / max deg",
         f"{r1['to_max_abs_deg']['median']:.5f}", f"{r1['to_max_abs_deg']['max']:.5f}",
         f"at t={r1['to_max_abs_deg']['argmax_t_step']}"],
        ["    fingertip displacement, median / max mm",
         f"{r1['tip_displacement_mm']['median']:.4f}",
         f"{r1['tip_displacement_mm']['max']:.4f}",
         f"at t={r1['tip_displacement_mm']['argmax_t_step']}"],
        ["    which joint led, by degrees", str(r1["largest_joint_counts"]), "", ""],
        ["    which joint led, by millimetres", str(r1["largest_contrib_joint_counts"]), "", ""],
        ["", "", "", ""],
        ["    per joint over the approach", "max deg", "max fingertip mm", ""],
    ]
    colours[len(rows) - 1] = figstyle.INK
    for j in r1["per_joint_max_over_approach_deg"]:
        rows.append([f"      {j}", f"{r1['per_joint_max_over_approach_deg'][j]:.5f}",
                     f"{r1['per_joint_max_contrib_mm'][j]:.4f}", ""])
    rows += [["", "", "", ""],
             ["R4  the moment the gripper closes", "", "", ""]]
    colours[len(rows) - 1] = figstyle.INK
    c = r4["at_the_closing_step"]
    rows += [
        ["    chunk / step", f"t_step {c['t_step_of_chunk']} (seq {c['seq']})",
         f"step {c['step_in_chunk']} of 0..7", f"global {c['t_step_of_the_step']}"],
        ["    gripper command", f"{c['gripper_value']:.6f}",
         f"previous step {c['gripper_value_previous_step']:.6f}", ""],
        ["    ee_finger_l1 origin to apple surface",
         f"{c['ee_finger_l1_origin_to_apple_surface_mm']:.3f} mm", "",
         "analytic, radius not subtracted"],
        ["    offset from the apple centre",
         f"x {c['offset_tip_minus_apple_centre_mm'][0]:+.2f}",
         f"y {c['offset_tip_minus_apple_centre_mm'][1]:+.2f}",
         f"z {c['offset_tip_minus_apple_centre_mm'][2]:+.2f} mm"],
        ["    from the pose stored with that chunk", "sphere surface "
         f"{r4['tip_to_apple_surface_mm']:.3f} mm",
         f"sphere centre {r4['tip_centre_to_apple_surface_mm']:.3f} mm", "122-sphere model"],
        ["    smallest over all chunks", f"{r4['min_over_all_chunks_mm']:.3f} mm",
         f"at t={r4['min_over_all_chunks_t_step']}", ""],
        ["    the earlier T7a value", f"{r4['reference_value_T7a_mm']:.3f} mm", "",
         "different sphere model and reference point"],
    ]
    colours[len(rows) - 1] = figstyle.CATEGORICAL[3]
    table(axes[1, 0], "R1  what TO pushed   ·   R4  the closing moment",
          ["what", "value", "value", "note"], rows, [0.38, 0.24, 0.24, 0.14], colours,
          fontsize=7.4, dymax=0.036)

    # --- 방법 ---------------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("how this was measured", fontsize=10.0, color=figstyle.INK, loc="left", pad=10)
    m = a["method"]
    txt = (
        f"record {a['record']}\n"
        f"server {a['server_record']}\n"
        f"  {a['n_chunks']} chunks, approach window t_step 0..{a['approach_window_t_step'][1]}"
        f" ({a['n_approach_chunks']} chunks), esdf margin {a['esdf_margin_mm']:.0f} mm\n"
        "  this is a SHADOW run: the robot executes actions_reference, and the TO output\n"
        "  is computed and recorded but not executed.\n"
        "\n"
        "naming the surface behind a row, in three independent ways:\n"
        "  1. walk the recorded distance field from the query point down its gradient by\n"
        "     the field's own value, then ask which ground-truth body that point lands on.\n"
        "     the fine layer is used when the query point is inside it, the coarse 20 mm\n"
        "     layer otherwise. this is the field's own answer.\n"
        "  2. the analytic distance from the query point to every body's surface.\n"
        "  3. the nearest point of filtered_cloud - reported but NOT relied on: that cloud\n"
        "     is one frame and the field is accumulated over many.\n"
        "  mj_geomDistance is not used anywhere.\n"
        "\n"
        "ESDF distance at a query point is reconstructed as\n"
        "  d = clearance_m + sphere radius + margin\n"
        "because the recorded clearance is POST-margin (linearize.py:489).\n"
        "\n"
        "the server's summary_json 't_step' is the chunk sequence number;\n"
        "local t_step = (seq - 1) * 8.\n"
        "\n"
        "R1 splits the deviation per joint with mujoco mj_jacBody at the reference pose:\n"
        "  contribution_j = |J[:, dof_j] * (TO_j - reference_j)| in mm\n"
        "\n"
        f"elapsed {a['elapsed_s']} s\n"
        "\n"
        "raw   outputs/verify/T17a/r1to4.json / r1to4_curves.npz\n"
        "      outputs/verify/T17a/r1to4.py  (one script, R1..R4)\n"
    )
    ax.text(0.0, 0.97, txt, fontsize=7.9, family="monospace", color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("T17a - the numbers", fontsize=12.6, color=figstyle.INK, y=0.985)
    fig.tight_layout(rect=(0, 0.008, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps({"r1": r1, "r2": r2, "r3": r3, "r4": r4,
                                "r2_r3_overlap": ov, "method": m}, indent=1,
                               ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
