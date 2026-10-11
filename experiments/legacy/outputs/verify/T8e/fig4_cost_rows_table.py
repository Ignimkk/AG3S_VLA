"""T8e figure 4 — 안전 대가와 행 개수. 표만.

왼쪽 위 : exec8 위반 0 인 칸 전부 — 반지름·덮개 손실·행 개수·full50 잔여.
왼쪽 아래: 가장 공격적인 칸의 **link 별 유효 반지름** (현재 값과 나란히).
오른쪽 위: 세 길 비교 — 얇게만(T8c) · 자르기+손가락(T8e s1) · 자르기+전역(T8e s2).
오른쪽 아래: 행 개수의 성질과 경로.
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

T8E = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8e'))
T8C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8c'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8e/t8e-global-cost.png'))
SIDE = OUT.with_suffix(".json")


def table(ax, title, cols, rows, widths, colours=None, fontsize=7.6, head=8.3, dymax=0.05):
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
    g = json.loads((T8E / "s2_global_scale_sweep.json").read_text())
    e1 = json.loads((T8E / "s1_cut_plus_finger.json").read_text())
    c3 = json.loads((T8C / "s3_critical_radius.json").read_text())
    thin = [c for c in c3["inverted_scale_combo_verified_on_the_real_model"]
            if c["window"] == "exec8"][0]
    cut_only = [r for r in e1["rows"] if r["key"] == "finger_0.22__zcut_100mm__palm_out"][0]
    pick = g["most_aggressive_feasible_cell"]["out__exec8"]

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.4))
    figstyle.style_axes(fig, axes)

    # --- exec8 위반 0 인 칸 전부 ---------------------------------------------------------
    rows, colours = [], {}
    for ptag in ("out", "in_authorized"):
        rows.append([f"[palm {ptag}]", "", "", "", "", ""])
        colours[len(rows) - 1] = figstyle.INK
        for c in g["grid"]:
            if c["palm"] != ptag or not c["windows"]["exec8"]["feasible_at_10mm"]:
                continue
            f50 = c["windows"]["full50"]
            rc = c["row_cost"]
            rows.append([
                f"  spacing {c['sphere_spacing']:g} · scale x{c['capsule_radius_scale']:g}",
                f"{c['n_spheres_used_after_cut']:4d} used / {c['n_spheres_model']:4d}",
                f"L5 {c['radius_mm_by_link']['link_left_arm_5'][0]:6.2f}  "
                f"fl1 {c['radius_mm_by_link']['ee_finger_l1'][-1]:5.2f}  "
                f"palm {max(c['radius_mm_by_link']['ee_left']):5.2f}",
                f"{c['n_coverage_shortfall']:2d}  {c['worst_shortfall_mm']:6.2f}",
                f"cand {rc['candidate_rows_per_chunk']:5d}  QP {rc['qp_rows_per_chunk']}  "
                f"fk {rc['fk_map_eval_ms_mean_of_5']:.3f} ms",
                f"{f50['n_rows_slack_le0']:5d} {f50['n_chunks_slack_le0']:3d} "
                f"{f50['global_min_slack_mm']:+7.2f}"])
            colours[len(rows) - 1] = (figstyle.CATEGORICAL[2]
                                      if c["key"] == pick["key"] else figstyle.INK_2)
    table(axes[0, 0],
          "every cell that reaches 0 violations in exec8 (z_cut -100 mm, margin 10 mm)\n"
          "     green = the most aggressive one (smallest radii)",
          ["cell", "spheres", "radius mm", "short n/mm", "rows per chunk",
           "full50: rows chk min"],
          rows, [0.22, 0.13, 0.23, 0.13, 0.22], colours, fontsize=7.4, dymax=0.055)

    # --- 추천 칸의 link 별 반지름 ---------------------------------------------------------
    cur_cell = [c for c in g["grid"]
                if c["sphere_spacing"] == 1.0 and c["capsule_radius_scale"] == 1.0
                and c["palm"] == "out"][0]
    rows, colours = [], {}
    for ln in sorted(pick["radius_mm_by_link"]):
        now = cur_cell["radius_mm_by_link"][ln]
        new = pick["radius_mm_by_link"][ln]
        rows.append([ln, "/".join(f"{v:.2f}" for v in now),
                     "/".join(f"{v:.2f}" for v in new),
                     f"{cur_cell['n_spheres_by_link'][ln]:3d} -> "
                     f"{[c for c in g['grid'] if c['key'] == pick['key']][0]['n_spheres_by_link'][ln]:3d}"])
        if ln in ("link_left_arm_5", "ee_finger_l1", "ee_finger_l2", "ee_left"):
            colours[len(rows) - 1] = figstyle.CATEGORICAL[1]
    table(axes[1, 0],
          f"per-link effective radius at the most aggressive feasible cell\n"
          f"     {pick['key']}  ({pick['n_spheres_used_after_cut']} spheres used after the cut, "
          f"{g['n_spheres_current_constraint_model']} in the model running today)",
          ["link", "now (spacing 1.0, x1) mm", "picked cell mm", "n spheres"],
          rows, [0.26, 0.28, 0.28], colours, fontsize=7.4, dymax=0.042)

    # --- 세 길 -----------------------------------------------------------------------------
    rows, colours = [], {}

    def blk(name, setting, e8, f50, ns, nsh, worst, extra):
        rows.append([name, "", "", ""])
        colours[len(rows) - 1] = figstyle.INK
        rows.append(["  setting", setting, "", ""])
        rows.append(["  exec8", f"rows {e8[0]}", f"chunks {e8[1]}",
                     f"min slack {e8[2]:+.4f} mm  feasible {e8[3]}"])
        colours[len(rows) - 1] = figstyle.CATEGORICAL[2] if e8[3] else figstyle.CATEGORICAL[7]
        rows.append(["  full50", f"rows {f50[0]}", f"chunks {f50[1]}",
                     f"min slack {f50[2]:+.2f} mm  feasible {f50[3]}"])
        colours[len(rows) - 1] = figstyle.CATEGORICAL[2] if f50[3] else figstyle.CATEGORICAL[7]
        rows.append(["  coverage_shortfall", f"{nsh} capsules", f"worst {worst:.4f} mm", ""])
        colours[len(rows) - 1] = figstyle.CATEGORICAL[1]
        rows.append(["  constraint spheres", ns, "", ""])
        for k, v in extra:
            rows.append([f"  {k}", v, "", ""])
        rows.append(["", "", "", ""])

    tw = thin["evaluated_windows"]
    blk("A  thin only (T8c inverted radius scales, no cut)",
        ", ".join(f"{k} x{v:g}" for k, v in sorted(thin["applied"].items())),
        (tw["exec8"]["n_rows_slack_le0"], tw["exec8"]["n_chunks_slack_le0"],
         tw["exec8"]["min_slack_mm"], tw["exec8"]["feasible_at_10mm"]),
        (tw["full50"]["n_rows_slack_le0"], tw["full50"]["n_chunks_slack_le0"],
         tw["full50"]["min_slack_mm"], tw["full50"]["feasible_at_10mm"]),
        "120 (unchanged)", thin["n_coverage_shortfall"], thin["worst_shortfall_mm"],
        [("link_left_arm_5 radius",
          f"81.2500 -> {thin['radius_mm_by_scaled_link']['link_left_arm_5'][0]:.4f} mm")])

    cw = cut_only["windows"]
    blk("B  cut + fingertips only (T8e s1)",
        "z_cut -100 mm, ee_finger_l1/l2 x0.22, everything else x1",
        (cw["exec8"]["n_rows_slack_le0"], cw["exec8"]["n_chunks_slack_le0"],
         cw["exec8"]["global_min_slack_mm"], cw["exec8"]["feasible_at_10mm"]),
        (cw["full50"]["n_rows_slack_le0"], cw["full50"]["n_chunks_slack_le0"],
         cw["full50"]["global_min_slack_mm"], cw["full50"]["feasible_at_10mm"]),
        f"{cut_only['n_arm5_spheres_kept']} of {cut_only['n_arm5_spheres_total']} arm_5 kept; "
        f"120 total", cut_only["n_coverage_shortfall"], cut_only["worst_shortfall_mm"],
        [("link_left_arm_5 radius", "81.2500 mm, unchanged")])

    pe, pf = pick["windows"]["exec8"], pick["windows"]["full50"]
    blk("C  cut + global (T8e s2, most aggressive feasible)",
        f"z_cut -100 mm, global capsule_radius_scale x{pick['capsule_radius_scale']:g}, "
        f"sphere_spacing {pick['sphere_spacing']:g}, max_spheres_per_capsule 32",
        (pe["n_rows_slack_le0"], pe["n_chunks_slack_le0"], pe["global_min_slack_mm"],
         pe["feasible_at_10mm"]),
        (pf["n_rows_slack_le0"], pf["n_chunks_slack_le0"], pf["global_min_slack_mm"],
         pf["feasible_at_10mm"]),
        f"{pick['n_spheres_used_after_cut']} used / {pick['n_spheres_model']} in the model",
        pick["n_coverage_shortfall"], pick["worst_shortfall_mm"],
        [("link_left_arm_5 radius",
          f"81.2500 -> {pick['radius_mm_by_link']['link_left_arm_5'][0]:.4f} mm"),
         ("candidate rows per chunk",
          f"{pick['row_cost']['candidate_rows_per_chunk']} "
          f"(horizon {pick['row_cost']['horizon_used']} x "
          f"{pick['row_cost']['linearizer_n_spheres']} spheres)"),
         ("QP rows per chunk",
          f"{pick['row_cost']['qp_rows_per_chunk']} - unchanged, fixed by rows_per_step"),
         ("FK map eval",
          f"{pick['row_cost']['fk_map_eval_ms_mean_of_5']:.3f} ms "
          f"(spacing 1.0: 0.509 ms)")])
    table(axes[0, 1], "three routes to 0 violations in exec8, and the bill for each",
          ["route", "", "", ""], rows, [0.26, 0.26, 0.15, 0.33], colours,
          fontsize=7.4, dymax=0.037)

    # --- 행 개수 설명 ----------------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("what grows with the sphere count, and what does not", fontsize=10.0,
                 color=figstyle.INK, loc="left", pad=10)
    lines = ["  spacing   spheres   candidate rows   QP rows   jac nnz    build s   fk ms"]
    seen = set()
    for c in g["grid"]:
        sp = c["sphere_spacing"]
        if sp in seen:
            continue
        seen.add(sp)
        rc = c["row_cost"]
        lines.append(f"  {sp:<9g} {rc['linearizer_n_spheres']:7d} {rc['candidate_rows_per_chunk']:15d} "
                     f"{rc['qp_rows_per_chunk']:9d} {rc['jacobian_structural_nnz']:9d} "
                     f"{rc['linearizer_build_s']:9.2f} {rc['fk_map_eval_ms_mean_of_5']:7.3f}")
    txt = (
        "measured on a real CollisionLinearizer (horizon 8, ChunkLayout.rby1, palm included)\n\n"
        + "\n".join(lines) + "\n\n"
        "    QP rows = horizon x reduction.rows_per_step = 8 x 24 = 192, fixed.\n"
        "    linearize.py:258 and set_attached's docstring: widening the candidate list\n"
        "    changes what competes for those rows, not how many there are.\n"
        "\n"
        "radius is not linear in the scale, and it has a floor\n"
        "    n >= 2 : r_eff(s) = sqrt( (r_urdf s)^2 + (spacing/2)^2 )\n"
        "    spacing = capsule length / (n - 1), set by the UNSCALED radius\n"
        "    link_*_arm_5 (r 75, L 250 mm): floor 31.25 mm at spacing 1.0 (n=5),\n"
        "                                   22.12 at 0.6 (n=7), 15.79 at 0.4 (n=10)\n"
        "    max_spheres_per_capsule default 8 clips n below spacing 0.5 - raised to 32 here\n"
        "\n"
        "self-filter was not touched\n"
        "    build_robot_model has no sphere_options channel and was never called here.\n"
        "    Only the constraint model (build_constraint_robot_model) was swept.\n"
        "\n"
        "the cut is an approximation\n"
        "    arm_5 spheres with capsule-frame z < -0.100 m are dropped from the rows;\n"
        "    the ones that stay keep the radius the scale gives them. A real shorter\n"
        "    capsule would re-space them and change the radius - not measured.\n"
        "\n"
        "    raw   outputs/verify/T8e/s2_global_scale_sweep.json / s2_grid.npz\n"
        "          outputs/verify/T8e/s1_cut_plus_finger.json    (route B)\n"
        "          outputs/verify/T8c/s3_critical_radius.json    (route A)\n"
        "    input outputs/live_test/local_shadow_from_pc/frames.jsonl"
    )
    ax.text(0.0, 0.96, txt, fontsize=8.0, family="monospace", color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("T8e - shrink everything: what it costs in coverage and in rows",
                 fontsize=12.5, color=figstyle.INK, y=0.985)
    fig.tight_layout(rect=(0, 0.01, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps({
        "feasible_exec8_cells": [
            {k: c[k] for k in ("key", "sphere_spacing", "capsule_radius_scale", "palm",
                               "n_spheres_model", "n_spheres_used_after_cut",
                               "n_coverage_shortfall", "worst_shortfall_mm", "row_cost")}
            | {"radius_mm_by_link": c["radius_mm_by_link"]}
            | {w: {kk: c["windows"][w][kk] for kk in
                   ("n_rows_slack_le0", "n_chunks_slack_le0", "global_min_slack_mm",
                    "feasible_at_10mm", "violating_pairs")} for w in ("exec8", "full50")}
            for c in g["grid"] if c["windows"]["exec8"]["feasible_at_10mm"]],
        "most_aggressive_feasible_cell": g["most_aggressive_feasible_cell"],
        "route_A_thin_T8c": thin,
        "route_B_cut_plus_fingers_T8e_s1": cut_only,
        "current_model_reference": {
            "n_spheres": g["n_spheres_current_constraint_model"],
            "radius_mm_by_link": cur_cell["radius_mm_by_link"],
            "n_spheres_by_link": cur_cell["n_spheres_by_link"]},
    }, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
