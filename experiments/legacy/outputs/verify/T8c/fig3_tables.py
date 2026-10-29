"""T8c figure 3 — 표·수식.

왼쪽 위 : 배율 격자 (arm5 × finger) — 남은 위반 행 / chunk / 최소 slack, full50 과 exec8.
왼쪽 아래: 가장 공격적인 조합에서 **끝까지 남는 쌍**, 그리고 역산한 배율의 실제 모델 확인.
오른쪽 위: capsule group 별 r_crit 역산표 (full50 · exec8).
오른쪽 아래: 식과 이 측정이 쓴 경로.
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

T8C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8c'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8c/t8c-table.png'))
SIDE = OUT.with_suffix(".json")


def table(ax, title, cols, rows, widths, colours=None, fontsize=7.8, head=8.4, dymax=0.06):
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
    sw = json.loads((T8C / "s2_scale_sweep.json").read_text())
    s3 = json.loads((T8C / "s3_critical_radius.json").read_text())
    s1 = json.loads((T8C / "s1_radii_by_scale.json").read_text())

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.8))
    figstyle.style_axes(fig, axes)

    # --- 격자 ---------------------------------------------------------------------------
    rows, colours = [], {}
    for c in sw["combos"]:
        f50, e8 = c["windows"]["full50"], c["windows"]["exec8"]
        rows.append([
            f"x{c['arm5_scale']:g}", f"x{c['finger_scale']:g}",
            f"{c['radius_mm_link_left_arm_5'][0]:6.2f}",
            "/".join(f"{v:.1f}" for v in c["radius_mm_ee_finger_l1"]),
            f"{f50['n_rows_slack_le0']:5d} {f50['n_chunks_slack_le0']:3d} "
            f"{f50['global_min_slack_mm']:8.2f}",
            f"{e8['n_rows_slack_le0']:4d} {e8['n_chunks_slack_le0']:3d} "
            f"{e8['global_min_slack_mm']:8.2f}",
            f"{c['n_coverage_shortfall']:2d}  {c['worst_shortfall_mm']:6.2f}"])
        if f50["feasible_at_10mm"]:
            colours[len(rows) - 1] = figstyle.CATEGORICAL[2]
        elif e8["feasible_at_10mm"]:
            colours[len(rows) - 1] = figstyle.CATEGORICAL[3]
        else:
            colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
    table(axes[0, 0],
          "scale grid - apple removed for ee_finger_l1, ee_finger_l2, link_left_arm_5; "
          "margin 10 mm\n     red = still has violations in both windows · no combo reaches 0 "
          "in full50",
          ["arm5", "fing", "L5 r mm", "finger_l1 r mm", "full50: rows chk minSlack",
           "exec8: rows chk minSlack", "shortfall: n worst mm"],
          rows, [0.07, 0.07, 0.10, 0.17, 0.28, 0.21], colours, fontsize=7.6, dymax=0.048)

    # --- 남는 쌍 + 역산 확인 -------------------------------------------------------------
    rows, colours = [], {}
    rows.append([f"most aggressive combo: {sw['most_aggressive_combo']}", "", "", ""])
    colours[0] = figstyle.INK
    for w in ("full50", "exec8"):
        ps = sw["pairs_surviving_the_most_aggressive_combo"][w]
        rows.append([f"  [{w}]", "", f"{len(ps)} pair(s) survive", ""])
        colours[len(rows) - 1] = figstyle.INK
        for p in ps:
            rows.append([f"    {p['link']}", p["obstacle_body"],
                         f"rows {p['n_rows_slack_le0']:5d}  chunks {p['n_chunks_slack_le0']:3d}",
                         f"min clr {p['min_clearance_mm']:+7.2f}  slack {p['min_slack_mm']:+7.2f}"])
            colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
        if not ps:
            rows.append(["      none", "", "", ""])
            colours[len(rows) - 1] = figstyle.CATEGORICAL[2]
    rows.append(["", "", "", ""])
    rows.append(["inverted scale applied to the real UrdfSphereChain", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for c in s3["inverted_scale_verified_on_the_real_model"]:
        rows.append([f"  [{c['window']}] link_*_arm_5 x{c['scale_applied']:g}",
                     f"r {c['link_left_arm_5_radius_mm']:.3f} mm",
                     f"L5 rows<=0 {c['link_left_arm_5_rows_slack_le0']}  "
                     f"min {c['link_left_arm_5_min_slack_mm']:+.4f}",
                     f"all links rows<=0 {c['all_links_rows_slack_le0']}  "
                     f"min {c['all_links_min_slack_mm']:+.2f}"])
    for c in s3["inverted_scale_combo_verified_on_the_real_model"]:
        if c.get("blocked_links"):
            rows.append([f"  [{c['window']}] combo blocked",
                         ", ".join(c["blocked_links"]), "r_crit <= 0", ""])
            colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
            continue
        ap = ", ".join(f"{k} x{v:g}" for k, v in sorted(c["applied"].items()))
        rows.append([f"  [{c['window']}] combo", ap, "", ""])
        colours[len(rows) - 1] = figstyle.INK
        for ev, v in c["evaluated_windows"].items():
            rows.append([f"      evaluated on {ev}", "",
                         f"rows<=0 {v['n_rows_slack_le0']}  chunks {v['n_chunks_slack_le0']}",
                         f"min slack {v['min_slack_mm']:+.4f}  feasible "
                         f"{v['feasible_at_10mm']}"])
            colours[len(rows) - 1] = (figstyle.CATEGORICAL[2] if v["feasible_at_10mm"]
                                      else figstyle.CATEGORICAL[7])
        rows.append([f"      coverage_shortfall", f"{c['n_coverage_shortfall']} capsules",
                     f"worst {c['worst_shortfall_mm']:.2f} mm", ""])
    table(axes[1, 0], "what survives, and the inverted scale checked on the real model",
          ["scope", "detail", "count", "worst"], rows,
          [0.34, 0.20, 0.25, 0.21], colours, fontsize=7.6, dymax=0.048)

    # --- r_crit 역산표 ------------------------------------------------------------------
    rows, colours = [], {}
    for w in ("full50", "exec8"):
        rows.append([f"[{w}]", "", "", "", "", "", ""])
        colours[len(rows) - 1] = figstyle.INK
        for gk, g in sorted(s3["groups"][w].items()):
            sc = g["scale_crit"]
            rows.append([
                "  " + g["link"], f"{g['n_spheres']:2d}",
                f"{g['r_at_scale_1_mm']:7.3f}", f"{g['floor_mm']:7.3f}",
                f"{g['min_surface_distance_mm']:8.3f}", f"{g['r_crit_mm']:8.3f}",
                "undefined" if sc is None else f"{sc:.6f}"])
            if sc is None:
                colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
            elif sc > 1.0:
                colours[len(rows) - 1] = figstyle.CATEGORICAL[2]
            else:
                colours[len(rows) - 1] = figstyle.CATEGORICAL[1]
        rows.append(["", "", "", "", "", "", ""])
    table(axes[0, 1],
          "r_crit inversion per capsule group - the largest radius with zero violations\n"
          "     orange = needs shrinking · green = s_crit > 1, already fine · "
          "red = r_crit <= 0, no scale can fix it",
          ["capsule group", "nS", "r@1 mm", "floor mm", "min centre-surf mm", "r_crit mm",
           "scale_crit"],
          rows, [0.22, 0.05, 0.11, 0.11, 0.19, 0.13], colours, fontsize=7.6, dymax=0.048)

    # --- 식 ----------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("how the radius sweep reuses the T8a tensor", fontsize=10.0,
                 color=figstyle.INK, loc="left", pad=10)
    ck = sw["centre_invariance_check"]
    txt = (
        "T8a tensor holds   clearance_1.0(c,s,i,b) = surface_distance(c,s,i,b) - r_1.0(i)\n"
        "sphere centres do not move with the scale, so\n"
        "\n"
        "    surface_distance = clearance_1.0 + r_1.0(i)        (exact, no new geometry)\n"
        "    clearance_s      = surface_distance - r_s(i)\n"
        "    margin_slack     = clearance_s - 10 mm\n"
        "\n"
        f"    centre drift over every scale tried : "
        f"{ck['max_abs_centre_drift_over_all_scales_m']:.1e} m  "
        f"(probe {ck['probe_q']})\n"
        "\n"
        "radius is NOT linear in the scale (urdf_sphere_chain._effective_radius):\n"
        "    n = 1 sphere  :  r_eff(s) = r_urdf * s + L/2\n"
        "    n >= 2        :  r_eff(s) = sqrt( (r_urdf * s)^2 + (spacing/2)^2 )\n"
        "    spacing comes from the UNSCALED radius -> r_eff has a floor at spacing/2\n"
        "    every r_s in this measurement was read from a real UrdfSphereChain, not computed\n"
        "\n"
        "    r_crit = min over kept rows of surface_distance  -  margin\n"
        "    scale_crit = sqrt(r_crit^2 - floor^2) / r_urdf        (n >= 2)\n"
        "    r_crit <= floor  ->  no scale reaches it\n"
        "    r_crit <= 0      ->  the query point itself is inside the margin\n"
        "\n"
        f"    links scaled : arm5 {', '.join(sw['arm5_links_scaled'])}\n"
        f"                   fingers {', '.join(sw['finger_links_scaled'])}\n"
        f"    grid         : arm5 {s1['arm5_scales']} x finger {s1['finger_scales']}\n"
        f"    max_sphere_radius : "
        f"{s1['combos'][0]['max_sphere_radius']} (no cap in these runs)\n"
        "\n"
        "    raw    outputs/verify/T8c/s1_radii_by_scale.json / .npz\n"
        "           outputs/verify/T8c/s2_scale_sweep.json / s2_scale_curves.npz\n"
        "           outputs/verify/T8c/s3_critical_radius.json\n"
        "    input  outputs/verify/T8a/m0_clearance_tensor.npz  (75 x 50 x 120 x 6)"
    )
    ax.text(0.0, 0.96, txt, fontsize=8.2, family="monospace", color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("T8c - how thin would the spheres have to be, and what does that cost",
                 fontsize=12.5, color=figstyle.INK, y=0.985)
    fig.tight_layout(rect=(0, 0.01, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps({
        "grid": [{k: c[k] for k in ("key", "arm5_scale", "finger_scale",
                                    "radius_mm_link_left_arm_5", "radius_mm_ee_finger_l1",
                                    "radius_mm_ee_finger_l2", "n_coverage_shortfall",
                                    "worst_shortfall_mm")}
                 | {w: {kk: c["windows"][w][kk] for kk in
                        ("n_rows_slack_le0", "n_chunks_slack_le0", "global_min_slack_mm",
                         "feasible_at_10mm")} for w in ("full50", "exec8")}
                 for c in sw["combos"]],
        "pairs_surviving_the_most_aggressive_combo":
            sw["pairs_surviving_the_most_aggressive_combo"],
        "r_crit_groups": s3["groups"],
        "inverted_scale_verified_on_the_real_model":
            s3["inverted_scale_verified_on_the_real_model"],
        "inverted_scale_combo_verified_on_the_real_model":
            s3["inverted_scale_combo_verified_on_the_real_model"],
        "centre_invariance_check": ck,
    }, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
