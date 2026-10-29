"""T9a figure 3 — 표·도식. M1..M5 를 숫자로 나란히."""
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

T9A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T9a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t9a/t9a-table.png'))
SIDE = OUT.with_suffix(".json")


def table(ax, title, cols, rows, widths, colours=None, fontsize=7.8, head=8.4, dymax=0.055):
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
    ana = json.loads((T9A / "m1to5_analysis.json").read_text())
    meta = json.loads((T9A / "m0_state_tensor.meta.json").read_text())
    m1, m2, m3 = ana["m1_finger_to_apple"], ana["m2_gripper_close"], ana["m3_apple_ground_truth"]
    m4, m5 = ana["m4_descent_over_crate"], ana["m5_max_violation_pair"]

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 11.0))
    figstyle.style_axes(fig, axes)

    # --- M3 + M1 + M2 --------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["M3  did the apple move (ground truth)", "", "", ""])
    colours[0] = figstyle.INK
    rows += [
        ["    max lift", f"{m3['max_lift_mm']:+.4f} mm", "", "0 means never grasped"],
        ["    max total displacement", f"{m3['max_total_displacement_mm']:.4f} mm",
         f"at t={m3['argmax_total_t_step']}", ""],
        ["    rows with displacement > 1 mm", f"{m3['n_rows_with_displacement_gt_1mm']}",
         f"of {ana['n_state_rows']}", ""],
        ["    other objects max displacement",
         ", ".join(f"{k} {v:.3f}" for k, v in m3["other_objects_max_total_displacement_mm"].items()
                   if k != "apple"), "", "mm"],
        ["", "", "", ""],
    ]
    rows.append(["M1  closest left fingertip to apple surface", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    rows += [
        ["    this run (execute_ep1807)", f"{m1['left_min_mm']:+.4f} mm",
         f"t={m1['left_min_t_step']} seq={m1['left_min_chunk_seq']}", "clearance = d - r"],
        ["    previous closed-loop (T7a)",
         f"{m1['comparison_mm']['prev_closed_loop_T7a']:+.4f} mm", "", "reference"],
        ["    successful shadow (T7a)",
         f"{m1['comparison_mm']['successful_shadow_T7a']:+.4f} mm", "", "reference"],
        ["    gap closed",
         f"{m1['comparison_mm']['fraction_of_the_gap_closed']*100:.1f} %",
         f"{m1['comparison_mm']['this_minus_prev_closed_loop']:+.2f} mm vs prev",
         f"{m1['comparison_mm']['this_minus_shadow']:+.2f} mm vs shadow"],
        ["    centre-to-surface (radius removed)",
         f"{m1['left_min_centre_to_apple_surface_mm']:+.4f} mm", "",
         "this model r 3.6/3.97/4.72 mm"],
        ["    ee_left palm closest", f"{m1['left_palm_min_mm']:+.4f} mm",
         f"t={m1['left_palm_min_t_step']}", "new link this run"],
        ["    right fingertip closest", f"{m1['right_min_mm']:+.2f} mm", "", ""],
        ["", "", "", ""],
    ]
    L = m2["left"]
    rows.append(["M2  the moment the gripper closed (left, column 7)", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    rows += [
        ["    first command < 0.85", f"t_step {L['t_step']}",
         f"value {L['command_value']:.4f}",
         f"at t={L['previous_t_step']} it was {L['command_at_previous_t_step']:.4f}"],
        ["    finger -> apple surface then", f"{L['finger_to_apple_surface_mm']:+.4f} mm",
         f"centre {L['finger_centre_to_apple_surface_mm']:+.2f} mm", L["closest_sphere_link"]],
        ["    offset sphere - apple centre",
         f"x {L['offset_closest_sphere_minus_apple_centre_mm']['x']:+.2f}",
         f"y {L['offset_closest_sphere_minus_apple_centre_mm']['y']:+.2f}",
         f"z {L['offset_closest_sphere_minus_apple_centre_mm']['z']:+.2f} mm"],
        ["    offset hand centroid - apple",
         f"x {L['offset_hand_centroid_minus_apple_centre_mm']['x']:+.2f}",
         f"y {L['offset_hand_centroid_minus_apple_centre_mm']['y']:+.2f}",
         f"z {L['offset_hand_centroid_minus_apple_centre_mm']['z']:+.2f} mm"],
        ["    right gripper", "never below 0.85",
         f"min {m2['right']['min_command']:.4f}", ""],
    ]
    table(axes[0, 0], "M3 · M1 · M2", ["what", "value", "where", "note"], rows,
          [0.34, 0.21, 0.24, 0.21], colours, fontsize=7.8, dymax=0.040)

    # --- M4 ------------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["M4  descent over the crate", "", "", ""])
    colours[0] = figstyle.INK
    rows += [
        ["    hand first over the crate", f"t_step {m4['first_over_crate_t_step']}",
         f"{m4['n_rows_over_crate']} rows", m4["over_crate_rule"]],
        ["    crate rim top z", f"{m4['crate_rim_top_z_mm']:.2f} mm", "", "from crate box geoms"],
        ["    lowest EE_BODY_L z while over",
         f"{m4['hand_z_mm_min_while_over_crate']:.2f} mm",
         f"{m4['hand_above_crate_rim_mm_min']:+.2f} mm above rim", "never went lower"],
        ["    EE_BODY_L z at the last record", f"{m4['hand_z_mm_at_last_row']:.2f} mm", "", ""],
        ["", "", "", ""],
    ]
    h = m4["destination_margin_hypothesis"]
    at = h["attached_rows_in_max_violation_pair"]
    rows.append(["    destination_margin hypothesis - what the record shows", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    rows += [
        ["      tightest row is attached:*", f"{at['n_chunks_with_attached_link']}",
         f"of {at['n_chunks_total']}", "so attached points were active"],
        ["      attached labels seen", f"{len(at['attached_link_labels'])} distinct",
         ", ".join(x.replace("attached:ee_finger_l1", "l1") for x in
                   at["attached_link_labels"][:4]) + " ...", ""],
        ["      attached row clearance range",
         f"{at['attached_clearance_mm_min']:.2f} .. {at['attached_clearance_mm_max']:.2f} mm",
         "", f"esdf margin {h['esdf_margin_mm']:.0f} mm"],
        ["      obstacle values seen", ", ".join(at["obstacle_values_seen"]), "", ""],
        ["      apple max lift", f"{h['apple_max_lift_mm']:+.4f} mm", "",
         "nothing was actually held"],
        ["      left hand min clearance to crate",
         f"{h['left_hand_min_clearance_to_crate_mm_over_descent']:.2f} mm", "", "during descent"],
        ["      right hand min clearance to crate",
         f"{h['right_hand_min_clearance_to_crate_mm_over_descent']:.2f} mm", "", "during descent"],
        ["      all chunks safe", f"{h['all_chunks_safe']}",
         f"max_violation_m max {h['max_violation_m_max']}", ""],
        ["", "", "", ""],
    ]
    rows.append(["    planning rows from first-over-crate (offline clearance, mm)", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for r in m4["planning_rows_from_first_over_crate"]:
        rows.append([f"      t={r['t_step']} seq={r['chunk_seq']}",
                     f"hand z {r['hand_z_mm']:.1f}",
                     f"crate {r['min_clearance_to_crate_mm']:.2f} ({r['min_clearance_to_crate_link']})",
                     f"mvp {r['mvp_link']} {r['mvp_clearance_mm']:.2f}"])
    table(axes[0, 1], "M4  what the hand was doing over the crate",
          ["what", "value", "where", "note"], rows, [0.30, 0.18, 0.29, 0.23], colours,
          fontsize=7.2, dymax=0.036)

    # --- M5 ------------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["M5  max_violation_pair over 39 chunks (unsafe 0 - the tightest row, "
                 "not a violation)", "", "", ""])
    colours[0] = figstyle.INK
    rows += [
        ["    clearance POST-margin  min / med / max",
         f"{m5['clearance_mm_min']:.2f}", f"{m5['clearance_mm_median']:.2f}",
         f"{m5['clearance_mm_max']:.2f} mm"],
        ["    safe / unsafe", f"{m5['n_chunks_safe']} / {m5['n_chunks_unsafe']}",
         f"trajopt {list(m5['trajopt_status_counts'])}", f"block {list(m5['block_counts'])}"],
        ["", "", "", ""],
        ["    clearance_m is already margin-subtracted", "", "", ""],
    ]
    colours[len(rows) - 1] = figstyle.INK
    rv = m5["recorded_clearance_is_post_margin"]
    mis = rv["corrected_misread"]
    rows += [
        ["      definition", "d_esdf - r_robot - margin", "linearize.py:489 / :704 / :718",
         "margin already removed"],
        ["      clearance < 0 (real violation)", f"{rv['n_chunks_clearance_below_0']}",
         f"of 39", "max_violation = max(0, -clearance)"],
        ["      min post-margin / +margin",
         f"{rv['min_clearance_mm_post_margin']:.4f} mm",
         f"{rv['min_clearance_mm_pre_margin_equivalent']:.4f} mm", "+ margin 10 mm"],
        ["      verdict.max_violation_m all zero", f"{rv['verdict_max_violation_m_all_zero']}",
         f"safe=True on {rv['n_chunks_safe_true']}/39", f"consistent: {rv['consistent']}"],
        ["      CORRECTED MISREAD", f"{mis['n_chunks_clearance_below_10mm_post_margin']} chunks",
         f"t_steps {mis['t_steps_of_those_chunks']}",
         "comparing a post-margin value to the margin"],
        ["", "", "", ""],
        ["    link counts (recorded)", "", "", ""],
    ]
    colours[len(rows) - 1] = figstyle.INK
    for k, v in list(m5["link_counts"].items()):
        rows.append([f"      {k}", f"{v}", "", ""])
        if str(k).startswith("attached:"):
            colours[len(rows) - 1] = figstyle.CATEGORICAL[1]
    rows.append(["", "", "", ""])
    rows.append(["    offline clearance below is PRE-margin (surface distance - radius)",
                 "", "", ""])
    rows.append(["", "", "", ""])
    rows.append(["    offline tightest sphere over the 39 planning rows", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for k, v in m5["offline_tightest_link_counts"].items():
        rows.append([f"      {k}", f"{v}", "", ""])
    for k, v in m5["offline_tightest_body_counts"].items():
        rows.append([f"      body: {k}", f"{v}", "", ""])
    table(axes[1, 0], "M5  which row was tightest", ["what", "count / value", "where", "note"],
          rows, [0.35, 0.20, 0.25, 0.20], colours, fontsize=6.9, dymax=0.029)

    # --- 모델·방법 -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("the constraint sphere model this measurement rebuilt, and how",
                 fontsize=10.0, color=figstyle.INK, loc="left", pad=10)
    r = meta["radius_mm_by_link"]
    n = meta["n_spheres_by_link"]
    lines = [
        "server settings reproduced offline:",
        f"  sphere_spacing {meta['sphere_options']['sphere_spacing']}  ·  "
        f"capsule_radius_scale {meta['sphere_options']['capsule_radius_scale']}  ·  "
        f"max_spheres_per_capsule {meta['sphere_options']['max_spheres_per_capsule']}",
        f"  capsule_extent_by_link {meta['sphere_options']['capsule_extent_by_link']}",
        f"  -> {meta['n_spheres']} spheres, {meta['n_capsules']} capsules   "
        f"server log says {tuple(meta['server_log_says_spheres_capsules'])}   "
        f"match: {meta['matches_server']}",
        "",
        "effective radius by link  [mm]:",
    ]
    for ln in sorted(r):
        lines.append(f"  {ln:18s} n={n[ln]:3d}  r={'/'.join(f'{v:.2f}' for v in r[ln])}")
    lines += [
        "",
        "arm_5 capsule trims (link-frame z):",
    ]
    for tr in meta.get("capsule_trims", []):
        lines.append(f"  {tr['link']}: keep z >= {tr['keep_z_min']*1000:.0f} mm, "
                     f"axis [{tr['axis_z_min']*1000:.0f}, {tr['axis_z_max']*1000:.0f}], "
                     f"spheres {tr['n_spheres_before']} -> {tr['n_spheres_after']}, "
                     f"reach z {tr['reach_z_min_before']*1000:.1f} -> "
                     f"{tr['reach_z_min_after']*1000:.1f} mm")
    lines += [
        "",
        "row definition:",
        "  clearance(row, sphere, body) = min over collision geoms g of body",
        "      [ surface_distance(sphere centre, g) ] - sphere radius",
        "  surface_distance analytic; box exact, mesh = min distance to vertices (upper bound)",
        "  mj_geomDistance NOT used -> no distmax saturation",
        "  this offline clearance is PRE-margin.",
        "",
        "the recorded max_violation_pair.clearance_m is POST-margin:",
        "  clearance_m = d_esdf(p) - r_robot - margin      (linearize.py:489 _esdf_clearance)",
        "  linearize.py:704 hands that block element to _identify, :718 stores it as clearance_m",
        "  so max_violation = max(0, -clearance_m); clearance_m >= 0 is NOT a violation",
        "  add the 10 mm margin back to compare it with the offline numbers above",
        "",
        f"state rows: {meta['n_state_rows']} "
        f"(312 control + 39 planning), recorded qpos carries the object free joints",
        f"crate rim top z {meta['crate_rim_top_z_mm']:.2f} mm · "
        f"table top z {meta['table_top_z_mm']:.2f} mm",
        f"rebuild time {meta['elapsed_s']} s",
        "",
        "raw   outputs/verify/T9a/m0_state_tensor.npz / .meta.json",
        "      outputs/verify/T9a/m1to5_analysis.json / m1to5_curves.npz",
        "record outputs/live_test/20260926_t8/execute_ep1807/frames.jsonl",
    ]
    ax.text(0.0, 0.97, "\n".join(lines), fontsize=7.4, family="monospace",
            color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("T9a - execute_ep1807: the numbers", fontsize=12.6, color=figstyle.INK, y=0.985)
    fig.tight_layout(rect=(0, 0.008, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps({"m1": m1, "m2": m2, "m3": m3,
                                "m4": {k: v for k, v in m4.items()},
                                "m5": {k: v for k, v in m5.items() if k != "per_chunk"},
                                "model_meta": meta}, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
