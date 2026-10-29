"""T10a figure 3 — 표. P1 이 판정이므로 맨 위에."""
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

T10A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T10a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t10a/t10a-table.png'))
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
    ana = json.loads((T10A / "p1to4.json").read_text())
    p1, p2, p3, p4 = (ana["p1_what_blocks_the_fingers"], ana["p2_margin_sweep"],
                      ana["p3_trust_region"], ana["p4_t9_vs_t10"])
    rc = ana["recheck_of_lead_numbers"]
    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 11.2))
    figstyle.style_axes(fig, axes)

    # --- P1 -------------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["P1  nearest ground-truth body to a gripper sphere, approach window (t <= 200)",
                 "", "", ""])
    colours[0] = figstyle.INK
    for tag, lab in (("refined__apple_excluded", "TO output, apple excluded (what the "
                                                 "gripper links see)"),
                     ("refined__all_bodies", "TO output, all bodies"),
                     ("reference__apple_excluded", "policy reference, apple excluded")):
        e = p1["T10"][tag]
        rows.append([f"    {lab}", "", "", ""])
        colours[len(rows) - 1] = figstyle.INK
        rows.append(["      nearest-body counts",
                     ", ".join(f"{k} {v}" for k, v in
                               e["nearest_body_counts_per_chunk"].items()),
                     f"of {e['n_chunks']}", ""])
        rows.append(["      min by body [mm]  "
                     + "  ".join(f"{k} {v:.1f}" for k, v in e["min_mm_by_body"].items()),
                     "", "", ""])
        rows.append(["      global minimum", f"{e['global_min_mm']:.4f} mm",
                     f"{e['global_min_body']}", f"t={e['global_min_chunk_t_step']}"])
        colours[len(rows) - 1] = (figstyle.CATEGORICAL[2] if e["global_min_mm"] > 10.0
                                  else figstyle.CATEGORICAL[7])
        rows.append(["", "", "", ""])
    w = p1["T10"]["recorded_worst_row_named_offline"]
    ev = w["esdf_vs_ground_truth"]
    rows.append(["    the recorded worst row, named offline", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    rows += [
        ["      rows on a robot sphere / attached point",
         f"{w['n_rows_on_a_robot_sphere']} / {w['n_rows_on_an_attached_point']}",
         f"link name matches offline: {w['n_link_name_matches']}", ""],
        ["      nearest body, all bodies",
         ", ".join(f"{k} {v}" for k, v in w["nearest_body_counts_all"].items()), "", ""],
        ["      nearest body, apple excluded",
         ", ".join(f"{k} {v}" for k, v in w["nearest_body_counts_apple_excluded"].items()),
         "", ""],
        ["      offline minus ESDF on the same sphere",
         f"min {ev['offline_minus_esdf_mm_all_bodies']['min']:+.2f}",
         f"median {ev['offline_minus_esdf_mm_all_bodies']['median']:+.2f}",
         f"max {ev['offline_minus_esdf_mm_all_bodies']['max']:+.2f} mm"],
        ["      rows where ground truth is further",
         f"{ev['n_rows_offline_further_than_esdf']} of {ev['n_rows']}", "",
         "recorded obstacle field is None on all of them"],
        ["", "", "", ""],
        ["      t_step   ESDF d-r   gt all      gt no-apple", "", "", ""],
    ]
    colours[len(rows) - 1] = figstyle.INK
    for x in w["rows"]:
        if x["is_attached_query_point"]:
            continue
        if x["t_step"] % 16 and x["t_step"] not in (96, 192):
            continue
        rows.append([f"        {x['t_step']:>4d}   {x['esdf_d_minus_r_mm']:8.2f}   "
                     f"{x['nearest_body_all']} {x['nearest_mm_all']:7.2f}   "
                     f"{x['nearest_body_apple_excluded']} {x['nearest_mm_apple_excluded']:7.2f}",
                     "", "", f"{x['sphere_link_offline']}"])
    table(axes[0, 0], "P1  what is near the fingers - the verdict",
          ["what", "value", "where", "note"], rows, [0.42, 0.22, 0.20, 0.16], colours,
          fontsize=7.0, dymax=0.030)

    # --- P2 · P4 --------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["P2  margin sweep on the POLICY REFERENCE chunk, approach window", "", "", ""])
    colours[0] = figstyle.INK
    rows.append(["    margin", "all bodies  rows/chunks",
                 "no apple  rows/chunks", "min slack, no apple"])
    colours[1] = figstyle.INK
    for mg in ("10mm", "5mm", "2mm", "1mm", "0mm"):
        a = p2["sweeps"]["all_bodies"][mg]
        b = p2["sweeps"]["apple_excluded"][mg]
        rows.append([f"      {mg}", f"{a['n_rows_slack_le0']} / {a['n_chunks_with_any']}"
                     + ("" if a["feasible"] else f"  ({list(a['rows_by_body'])})"),
                     f"{b['n_rows_slack_le0']} / {b['n_chunks_with_any']}",
                     f"{b['min_slack_mm']:+.3f} mm"])
        colours[len(rows) - 1] = (figstyle.CATEGORICAL[2] if a["feasible"]
                                  else figstyle.CATEGORICAL[7])
    for tag in ("all_bodies", "apple_excluded"):
        c = p2["sweeps"][tag]["critical_margin_mm"]
        rows.append([f"    critical margin, {tag}", f"{c['value']:.4f} mm",
                     f"{c['argmin_body']} / {c['argmin_link']}", f"t={c['argmin_t_step']}"])
        colours[len(rows) - 1] = figstyle.INK
    rows += [["", "", "", ""],
             ["P4  T9 (objective on) vs T10 (objective off)", "", "", ""]]
    colours[len(rows) - 1] = figstyle.INK
    rows.append(["    quantity", "T9", "T10", "note"])
    colours[len(rows) - 1] = figstyle.INK
    for lab, k, fmt in (("tip-apple, reference min",
                         "tip_to_apple_reference_min_mm", "{:+.4f} mm"),
                        ("tip-apple, TO output min", "tip_to_apple_refined_min_mm",
                         "{:+.4f} mm"),
                        ("TO dev median, all", "to_deviation_deg_median_all",
                         "{:.4f} deg"),
                        ("TO dev median, approach",
                         "to_deviation_deg_median_approach", "{:.4f} deg")):
        rows.append([f"      {lab}", fmt.format(p4["runs"]["T9"][k]),
                     fmt.format(p4["runs"]["T10"][k]), ""])
    rows.append(["      nearest body, approach",
                 ", ".join(f"{a} {b}" for a, b in
                           p4["runs"]["T9"]["nearest_body_counts_approach"].items()),
                 ", ".join(f"{a} {b}" for a, b in
                           p4["runs"]["T10"]["nearest_body_counts_approach"].items()), ""])
    rows += [["", "", "", ""],
             ["    re-check of the lead's numbers (T10)", "", "", ""]]
    colours[len(rows) - 1] = figstyle.INK
    rows += [
        ["      apple lift / displacement", f"{rc['apple_max_lift_mm']:.4f} mm",
         f"{rc['apple_max_total_displacement_mm']:.4f} mm", "lead: 0.0200 / 0.5431"],
        ["      safe / max_violation_m",
         f"{rc['n_chunks_safe']} of {rc['n_chunks']}", f"{rc['max_violation_m_max']}",
         "lead: 66/66, 0.0"],
        ["      TO dev median all / approach",
         f"{rc['to_deviation_median_deg_all_T10']:.4f}",
         f"{rc['to_deviation_median_deg_approach_T10']:.4f} deg",
         "lead: 0.581 / 0.945"],
    ]
    table(axes[0, 1], "P2  margin sweep   ·   P4  T9 vs T10",
          ["what", "T9 / all bodies", "T10 / apple excluded", "note"], rows,
          [0.30, 0.26, 0.26, 0.18], colours, fontsize=7.2, dymax=0.036)

    # --- P3 ---------------------------------------------------------------------------------
    rows, colours = [], {}
    o = p3["outlier_chunk_detail"]
    rows.append([f"P3  trust region (trust_radius {p3['trust_radius_rad']} rad = "
                 f"{np.degrees(p3['trust_radius_rad']):.4f} deg, {p3['source']})",
                 "", "", ""])
    colours[0] = figstyle.INK
    rows.append(["    quantity", "T9", "T10", "note"])
    colours[1] = figstyle.INK
    for lab, k, fmt in (("max |dq| over all chunks", "max_abs_dq_deg_max", "{:.4f} deg"),
                        ("ratio to trust_radius, max", "ratio_to_trust_radius_max", "{:.6f}"),
                        ("median ratio, all chunks", "median_ratio", "{:.6f}"),
                        ("chunks with ratio > 0.95", "n_chunks_ratio_gt_0.95", "{}"),
                        ("chunks with ratio > 0.50", "n_chunks_ratio_gt_0.50", "{}")):
        rows.append([f"      {lab}", fmt.format(p3["runs"]["T9"][k]),
                     fmt.format(p3["runs"]["T10"][k]), ""])
    rows.append(["    approach window only (t <= 200)", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for lab, k, fmt in (("max ratio", "max_ratio", "{:.6f}"),
                        ("median ratio", "median_ratio", "{:.6f}"),
                        ("max |dq|", "max_abs_dq_deg", "{:.4f} deg"),
                        ("chunks with ratio > 0.95", "n_chunks_ratio_gt_0.95", "{}"),
                        ("chunks with ratio > 0.50", "n_chunks_ratio_gt_0.50", "{}")):
        rows.append([f"      {lab}", fmt.format(p3["runs"]["T9"]["approach_window"][k]),
                     fmt.format(p3["runs"]["T10"]["approach_window"][k]), ""])
        if k == "max_ratio":
            colours[len(rows) - 1] = figstyle.CATEGORICAL[2]
    rows += [["", "", "", ""],
             [f"    the outlier: T10 chunk {o['chunk']} at t={o['t_step']}", "", "", ""]]
    colours[len(rows) - 1] = figstyle.INK
    rows += [
        ["      ratio to trust_radius", f"{o['ratio_to_trust_radius']:.6f}",
         f"{o['max_deg']:.4f} deg = {o['max_rad']:.5f} rad", "trust_radius 0.15 rad"],
        ["      joints within 0.01 deg of the max",
         f"{o['n_joints_within_0.01deg_of_the_max']} of 7", "", "they move together"],
        ["      per joint max |dq| deg",
         "  ".join(f"{v:.3f}" for v in list(o["per_joint_max_abs_deg"].values())[:4]),
         "  ".join(f"{v:.3f}" for v in list(o["per_joint_max_abs_deg"].values())[4:]),
         "L0..L6"],
        ["      recorded", f"safe={o['recorded']['safe']}",
         f"{o['recorded']['trajopt_status']}, "
         f"max_violation {o['recorded']['max_violation_m']}",
         f"mvp {o['recorded']['mvp_link']}"],
        ["      recorded notes", "", "", ""],
    ]
    for n in o["recorded"]["notes"][:3]:
        rows.append([f"        {n[:78]}", "", "", ""])
    rows.append(["      tightest offline rows in that chunk", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for x in o["tightest_offline_rows"][:5]:
        rows.append([f"        {x['link']} sphere {x['sphere']}",
                     f"{x['min_clearance_mm']:.2f} mm", f"vs {x['nearest_body']}", ""])
    table(axes[1, 0], "P3  was the trust region binding",
          ["what", "T9 / value", "T10 / value", "note"], rows,
          [0.36, 0.24, 0.24, 0.16], colours, fontsize=7.2, dymax=0.034)

    # --- 방법 ---------------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("how this was measured", fontsize=10.0, color=figstyle.INK, loc="left", pad=10)
    m = ana["method"]
    txt = (
        f"T10 record  {ana['runs']['T10']}\n"
        f"T9  record  {ana['runs']['T9']}\n"
        "  the two servers differ ONLY in the objective weights\n"
        "  (T10: w_track 1, w_smooth 0, w_continuity 0, w_slack 1000)\n"
        "  sphere model, target-field policy and the 10 mm margin are identical\n"
        "\n"
        "for every chunk and every execution step, for BOTH the policy chunk and the\n"
        "TO output:\n"
        "  the recorded qpos places the scene (object free joints included)\n"
        "  the action row drives the 14 arm joints, the torso comes from qpos\n"
        "  constraint sphere centres by forward kinematics, then\n"
        "  clearance = analytic surface distance to each scene body - sphere radius\n"
        "  mj_geomDistance NOT used -> no distmax saturation\n"
        "\n"
        f"sphere model: {m['n_spheres']} spheres, {m['n_capsules']} capsules\n"
        f"  {m['sphere_options']}\n"
        f"scene bodies: {', '.join(m['scene_bodies'])}   (table has no free joint)\n"
        f"approach window: {m['approach_window']}   execution window: {m['exec_window']} steps\n"
        f"gripper links: {', '.join(m['gripper_links'])}\n"
        "\n"
        "clearance here is PRE-margin: the 10 mm esdf margin is NOT subtracted.\n"
        "the recorded max_violation_pair.clearance_m is POST-margin (linearize.py:489),\n"
        "so 'ESDF d - r' in the P1 rows is the recorded value plus 10 mm.\n"
        "\n"
        "query indices >= 266 are attached points, not robot spheres - those rows are\n"
        "flagged and left unnamed (the offline sphere tensor cannot name them).\n"
        "\n"
        f"elapsed {ana['elapsed_s']} s\n"
        "\n"
        "raw   outputs/verify/T10a/p1to4.json / p1to4_curves.npz\n"
        "      outputs/verify/T10a/p1to4.py  (one script, P1..P4)\n"
    )
    ax.text(0.0, 0.97, txt, fontsize=7.8, family="monospace", color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("T10a - the numbers", fontsize=12.6, color=figstyle.INK, y=0.985)
    fig.tight_layout(rect=(0, 0.008, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps({"p1": p1, "p2": p2, "p3": p3, "p4": p4,
                                "recheck": rc, "method": m}, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
