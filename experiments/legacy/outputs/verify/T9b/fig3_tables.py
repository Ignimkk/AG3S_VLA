"""T9b figure 3 — 표. N1 이 판정이므로 맨 위에 크게."""
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

T9B = pathlib.Path(_ag3s_workspace_path('outputs/verify/T9b'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t9b/t9b-table.png'))
SIDE = OUT.with_suffix(".json")
GRIPPER = ("ee_finger_l1", "ee_finger_l2", "ee_left")


def table(ax, title, cols, rows, widths, colours=None, fontsize=7.6, head=8.4, dymax=0.05):
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
    ana = json.loads((T9B / "n1to5.json").read_text())
    n1, n2, n3 = (ana["n1_link_to_apple"], ana["n2_fingertip_divergence"],
                  ana["n3_reference_full_horizon"])
    n4, n5 = ana["n4_joint_contribution_mm"], ana["n5_gripper_columns"]

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 11.0))
    figstyle.style_axes(fig, axes)

    # --- N1 ------------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["N1  which link is nearest the apple, per chunk (min over the 8 exec steps)",
                 "", "", ""])
    colours[0] = figstyle.INK
    rows.append(["    nearest link", "policy reference", "TO output", "note"])
    colours[1] = figstyle.INK
    allk = list(dict.fromkeys(list(n1["nearest_link_counts_reference"])
                              + list(n1["nearest_link_counts_refined"])))
    for k in allk:
        a = n1["nearest_link_counts_reference"].get(k, 0)
        b = n1["nearest_link_counts_refined"].get(k, 0)
        rows.append([f"      {k}", f"{a} chunks", f"{b} chunks",
                     "gripper link" if k in GRIPPER else "ARM link"])
        colours[len(rows) - 1] = (figstyle.CATEGORICAL[2] if k in GRIPPER
                                  else figstyle.CATEGORICAL[7])
    rows += [
        ["      any arm link (link_*_arm_*)", "0 chunks", "0 chunks", "never nearest"],
        ["    chunks with a different nearest",
         f"{n1['n_chunks_different_nearest_link']} of {n1['n_chunks']}", "", "all 5 are "
         "finger<->finger or palm<->finger"],
        ["", "", "", ""],
        ["    closest chunk overall", f"chunk {n1['closest_chunk_detail']['chunk']} "
         f"(t_step {n1['closest_chunk_detail']['t_step']})", "", ""],
    ]
    colours[len(rows) - 1] = figstyle.INK
    rr = n1["closest_chunk_detail"]["ranking_reference"]
    rt = n1["closest_chunk_detail"]["ranking_refined"]
    rows.append(["      rank", "reference link / mm", "TO output link / mm", ""])
    colours[len(rows) - 1] = figstyle.INK
    for k in range(len(rr)):
        rows.append([f"        {k+1}", f"{rr[k]['link']}  {rr[k]['mm']:+.2f}",
                     f"{rt[k]['link']}  {rt[k]['mm']:+.2f}",
                     "same order" if rr[k]["link"] == rt[k]["link"] else "ORDER DIFFERS"])
        if rr[k]["link"] != rt[k]["link"]:
            colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
    rows += [
        ["", "", "", ""],
        ["    global minimum over all chunks",
         f"{n1['global_min_mm_reference']:+.4f} mm  {n1['global_min_link_reference']}",
         f"{n1['global_min_mm_refined']:+.4f} mm  {n1['global_min_link_refined']}",
         f"chunk {n1['global_min_chunk_reference']}"],
    ]
    colours[len(rows) - 1] = figstyle.INK
    rows.append(["", "", "", ""])
    rows.append(["    per-link min, all chunks", "reference mm",
                 "TO output mm", "TO - reference"])
    colours[len(rows) - 1] = figstyle.INK
    for k, v in n1["per_link_min_over_all_chunks_mm"].items():
        if "left" not in k and k not in GRIPPER:
            continue
        rows.append([f"      {k}", f"{v['reference']:+.2f}", f"{v['refined']:+.2f}",
                     f"{v['refined_minus_reference']:+.2f}"])
    table(axes[0, 0],
          "N1  the verdict - is TO bringing a different part to the apple",
          ["what", "policy reference", "TO output", "note"], rows,
          [0.33, 0.24, 0.24, 0.19], colours, fontsize=7.4, dymax=0.032)

    # --- N2 · N3 --------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["N2  fingertip: intended vs produced (ee_finger_l1 / l2 body origin)",
                 "", "", ""])
    colours[0] = figstyle.INK
    rows += [
        ["    max |reference - TO|", f"{n2['max_abs_mm']:.4f} mm",
         f"chunk {n2['max_abs_chunk']} step {n2['max_abs_step']}", ""],
        ["    median of the per-chunk max", f"{n2['median_over_chunks_of_max_mm']:.4f} mm", "",
         ""],
        ["    mean over all rows", f"{n2['mean_mm']:.4f} mm", "", ""],
        ["    component max |.|",
         f"x {n2['component_max_abs_mm']['x']:.2f}",
         f"y {n2['component_max_abs_mm']['y']:.2f}",
         f"z {n2['component_max_abs_mm']['z']:.2f} mm"],
        ["    component mean signed",
         f"x {n2['component_mean_signed_mm']['x']:+.2f}",
         f"y {n2['component_mean_signed_mm']['y']:+.2f}",
         f"z {n2['component_mean_signed_mm']['z']:+.2f} mm"],
        ["    mean gap per exec step 0..7",
         "  ".join(f"{v:.2f}" for v in n2["per_step_mean_mm"][:4]),
         "  ".join(f"{v:.2f}" for v in n2["per_step_mean_mm"][4:]), "mm"],
        ["    tip to apple, best chunk",
         f"reference {n2['tip_to_apple_mm']['reference_min']:+.4f} mm",
         f"TO {n2['tip_to_apple_mm']['refined_min']:+.4f} mm",
         f"cost {n2['tip_to_apple_mm']['refined_minus_reference_at_the_reference_min_chunk']:+.2f} mm"],
        ["", "", "", ""],
    ]
    rows.append([f"N3  follow the policy's own chunk {n3['chunk']} to step 49", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    rows += [
        ["    reference min over 50 steps", f"{n3['reference_min_mm_over_50_steps']:+.4f} mm",
         f"at step {n3['reference_argmin_step']}", "negative = inside the apple surface"],
        ["    reference min inside exec 8", f"{n3['reference_min_mm_in_exec8']:+.4f} mm", "", ""],
        ["    steps at or below 0 mm", f"{n3['reference_below_0mm_steps']} of {n3['horizon']}",
         f"TO {n3['refined_below_0mm_steps']} of {n3['horizon']}", ""],
        ["    TO min over 50 steps", f"{n3['refined_min_mm_over_50_steps']:+.4f} mm",
         f"at step {n3['refined_argmin_step']}",
         "identical: TO only rewrites the 8 exec steps"],
        ["    caveat", "object pose frozen at this chunk", "", "no future pose in the record"],
        ["", "", "", ""],
    ]
    rows.append(["N5  gripper columns, every element of 75 x 50", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for c in (7, 15):
        e = n5[f"column_{c}"]
        rows.append([f"    column {c} ({'left' if c == 7 else 'right'})",
                     f"{e['n_bitwise_identical']} / {e['n_elements']} identical",
                     f"max |diff| {e['max_abs_difference']:.1f}",
                     f"all identical: {e['all_identical']}"])
        colours[len(rows) - 1] = figstyle.CATEGORICAL[2]
    s = n5["sanity_joint_deltas_deg"]
    rows += [
        ["", "", "", ""],
        ["    re-check of the lead's degree table", "", "", ""],
    ]
    colours[len(rows) - 1] = figstyle.INK
    rows += [
        ["      left arm max |dev| L0..L6",
         "  ".join(f"{v:.2f}" for v in s["left_arm_max_abs_deg_exec8"][:4]),
         "  ".join(f"{v:.2f}" for v in s["left_arm_max_abs_deg_exec8"][4:]), "deg"],
        ["      right arm max |dev| R0..R6",
         "  ".join(f"{v:.2f}" for v in s["right_arm_max_abs_deg_exec8"][:4]),
         "  ".join(f"{v:.2f}" for v in s["right_arm_max_abs_deg_exec8"][4:]), "deg"],
        ["      median of per-chunk max",
         f"{s['per_chunk_max_abs_deg_median_exec8']:.4f} deg", "", "matches the lead's 1.06"],
    ]
    table(axes[0, 1], "N2 · N3 · N5", ["what", "value", "where", "note"], rows,
          [0.31, 0.24, 0.24, 0.21], colours, fontsize=7.4, dymax=0.036)

    # --- N4 --------------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["N4  which joint moved the fingertip - millimetres, not degrees", "", "", "", ""])
    colours[0] = figstyle.INK
    rows.append(["    joint", "mean contrib mm", "max contrib mm", "share %", "max dev deg"])
    colours[1] = figstyle.INK
    joints = n4["joints"]
    for j in joints:
        rows.append([f"      {j}",
                     f"{n4['mean_contribution_mm'][j]:.4f}",
                     f"{n4['max_contribution_mm'][j]:.4f}",
                     f"{n4['total_contribution_share'][j]*100:.2f}",
                     f"{n4['degree_vs_mm']['max_abs_deg_by_joint'][j]:.4f}"])
        if j == n4["largest_share_joint"]:
            colours[len(rows) - 1] = figstyle.CATEGORICAL[0]
        if j == "left_arm_6":
            colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
    lc = n4["linearisation_check"]
    rows += [
        ["", "", "", "", ""],
        ["    largest in millimetres", n4["largest_mean_joint"], "", "", ""],
        ["    largest in degrees", "left_arm_6", "", "",
         f"{n4['degree_vs_mm']['max_abs_deg_by_joint']['left_arm_6']:.2f} deg"],
        ["", "", "", "", ""],
        ["    Jacobian linearisation check", "", "", "", ""],
        ["      mean predicted / actual", f"{lc['mean_predicted_mm']:.4f} mm",
         f"{lc['mean_actual_mm']:.4f} mm", "", ""],
        ["      max |residual|", f"{lc['max_abs_residual_mm']:.4f} mm", "", "", ""],
    ]
    colours[len(rows) - 4] = figstyle.INK
    table(axes[1, 0], "N4  degrees rank L6 first, millimetres rank it last",
          ["what", "mean mm", "max mm", "share %", "max deg"], rows,
          [0.30, 0.18, 0.18, 0.16, 0.18], colours, fontsize=7.8, dymax=0.05)

    # --- 방법 -------------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("how this was measured", fontsize=10.0, color=figstyle.INK, loc="left", pad=10)
    m = ana["method"]
    txt = (
        f"record {ana['record']}\n"
        f"  {ana['n_chunks']} planning chunks x {ana['horizon']} steps, "
        f"execution window {ana['exec_window']}\n"
        f"  every chunk carries BOTH actions_reference (policy) and actions (TO output)\n"
        "\n"
        "for each chunk and each execution step, for BOTH chunks:\n"
        "  recorded qpos places the scene (object free joints included)\n"
        "  the action row drives the 14 arm joints; the torso comes from qpos\n"
        "  constraint sphere centres by forward kinematics, then\n"
        "  clearance = analytic surface distance to the apple geom - sphere radius\n"
        "  mj_geomDistance NOT used -> no distmax saturation\n"
        "\n"
        f"sphere model (same as the server, verified in T9a): "
        f"{m['n_spheres']} spheres, {m['n_capsules']} capsules\n"
        f"  {m['sphere_options']}\n"
        "\n"
        f"clearance is {('PRE-margin' if 'PRE-margin' in m['clearance_definition'] else '?')}: "
        "the 10 mm esdf margin is NOT subtracted here\n"
        "  (the recorded max_violation_pair.clearance_m is post-margin - see T9a)\n"
        "\n"
        "fingertip point: MuJoCo body origin of ee_finger_l1 / ee_finger_l2\n"
        "  (the link frame origin, not a constraint sphere centre)\n"
        "action columns: left_arm_0..6 -> col 0..6, left gripper col 7,\n"
        "  right_arm_0..6 -> col 8..14, right gripper col 15\n"
        "  (the lead confirmed this in types.py:193-198 / :120 / :142)\n"
        "\n"
        "N4 jacobian: mujoco.mj_jacBody translational rows at the reference pose of each step,\n"
        "  contribution_j = |J[:, dof_j] * (refined_j - reference_j)| in mm\n"
        "\n"
        f"elapsed {ana['elapsed_s']} s\n"
        "\n"
        "raw   outputs/verify/T9b/n1to5.json / n1to5_curves.npz\n"
        "      outputs/verify/T9b/n1to5.py  (one script, N1..N5)\n"
    )
    ax.text(0.0, 0.97, txt, fontsize=8.0, family="monospace", color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("T9b - the numbers", fontsize=12.6, color=figstyle.INK, y=0.985)
    fig.tight_layout(rect=(0, 0.008, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps({"n1": n1, "n2": n2, "n3": {k: v for k, v in n3.items()
                                                           if not k.startswith("series")},
                                "n4": n4, "n5": n5, "method": m}, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
