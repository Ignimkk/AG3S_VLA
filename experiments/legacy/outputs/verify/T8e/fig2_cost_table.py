"""T8e figure 2 — 두 길의 안전 대가를 나란히. 표만.

위 왼쪽 : s_finger 격자 (z_cut -100 mm, 손바닥 세 판).
위 오른쪽: 두 길 비교 — 얇게(T8c 역산 조합) 대 자르기(T8e).
아래 왼쪽: 손바닥을 넣었을 때 새로 생기는 쌍, 그리고 그 group 의 r_crit.
아래 오른쪽: 자르기가 `coverage_shortfall` 에 안 잡히는 부분 (기하로 적는다) + 경로.
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
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8e/t8e-cost-table.png'))
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
    e = json.loads((T8E / "s1_cut_plus_finger.json").read_text())
    c3 = json.loads((T8C / "s3_critical_radius.json").read_text())
    rows_all = e["rows"]

    def get(tag, f, z="100mm"):
        return [r for r in rows_all if r["key"] == f"finger_{f:g}__zcut_{z}__palm_{tag}"][0]

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.4))
    figstyle.style_axes(fig, axes)

    # --- s_finger 격자 -------------------------------------------------------------------
    rows, colours = [], {}
    for tag, lab in (("out", "palm out"), ("in", "palm in, apple KEPT for ee_left"),
                     ("in_authorized", "palm in, apple REMOVED for ee_left")):
        rows.append([f"[{lab}]", "", "", "", ""])
        colours[len(rows) - 1] = figstyle.INK
        for f in e["finger_scales"]:
            r = get(tag, f)
            e8, f50 = r["windows"]["exec8"], r["windows"]["full50"]
            rows.append([
                f"  finger x{f:g}",
                f"{r['radius_mm_ee_finger_l1'][-1]:6.2f}",
                f"{e8['n_rows_slack_le0']:5d} {e8['n_chunks_slack_le0']:3d} "
                f"{e8['global_min_slack_mm']:+8.2f}",
                f"{f50['n_rows_slack_le0']:5d} {f50['n_chunks_slack_le0']:3d} "
                f"{f50['global_min_slack_mm']:+8.2f}",
                f"{r['n_coverage_shortfall']:2d}  {r['worst_shortfall_mm']:6.2f}"])
            colours[len(rows) - 1] = (figstyle.CATEGORICAL[2] if e8["feasible_at_10mm"]
                                      else figstyle.CATEGORICAL[7])
    table(axes[0, 0],
          "s_finger grid at z_cut = -100 mm, arm_5 radius scale 1.0, margin 10 mm\n"
          "     green = exec8 reaches 0 violations · z_cut -108.7 mm gives the identical numbers",
          ["variant / scale", "distal r mm", "exec8: rows chk minSlack",
           "full50: rows chk minSlack", "shortfall: n worst"],
          rows, [0.24, 0.13, 0.25, 0.25], colours, fontsize=7.6, dymax=0.040)

    # --- 두 길 비교 -----------------------------------------------------------------------
    thin = [c for c in c3["inverted_scale_combo_verified_on_the_real_model"]
            if c["window"] == "exec8"][0]
    cut = get("out", 0.22)
    cut26 = get("out", 0.26)
    rows, colours = [], {}

    def norm(w):
        return {"n_rows_slack_le0": w["n_rows_slack_le0"],
                "n_chunks_slack_le0": w["n_chunks_slack_le0"],
                "global_min_slack_mm": w.get("global_min_slack_mm", w.get("min_slack_mm")),
                "feasible_at_10mm": w["feasible_at_10mm"]}

    def block(name, applied, e8, f50, nshort, worst, extra):
        e8, f50 = norm(e8), norm(f50)
        rows.append([name, "", "", ""])
        colours[len(rows) - 1] = figstyle.INK
        rows.append(["  setting", applied, "", ""])
        rows.append(["  exec8", f"rows {e8['n_rows_slack_le0']}", 
                     f"chunks {e8['n_chunks_slack_le0']}",
                     f"min slack {e8['global_min_slack_mm']:+.4f} mm  "
                     f"feasible {e8['feasible_at_10mm']}"])
        colours[len(rows) - 1] = (figstyle.CATEGORICAL[2] if e8["feasible_at_10mm"]
                                  else figstyle.CATEGORICAL[7])
        rows.append(["  full50", f"rows {f50['n_rows_slack_le0']}",
                     f"chunks {f50['n_chunks_slack_le0']}",
                     f"min slack {f50['global_min_slack_mm']:+.2f} mm  "
                     f"feasible {f50['feasible_at_10mm']}"])
        colours[len(rows) - 1] = (figstyle.CATEGORICAL[2] if f50["feasible_at_10mm"]
                                  else figstyle.CATEGORICAL[7])
        rows.append(["  coverage_shortfall", f"{nshort} capsules", f"worst {worst:.4f} mm", ""])
        colours[len(rows) - 1] = figstyle.CATEGORICAL[1]
        for k, v in extra:
            rows.append([f"  {k}", v, "", ""])
        rows.append(["", "", "", ""])

    block("A  thin - T8c inverted radius scales",
          ", ".join(f"{k} x{v:g}" for k, v in sorted(thin["applied"].items())),
          thin["evaluated_windows"]["exec8"], thin["evaluated_windows"]["full50"],
          thin["n_coverage_shortfall"], thin["worst_shortfall_mm"],
          [("link_left_arm_5 radius", "81.2500 -> "
            f"{thin['radius_mm_by_scaled_link']['link_left_arm_5'][0]:.4f} mm"),
           ("shortfall is reported by", "UrdfSphereChain.coverage_shortfall (URDF capsule basis)")])

    block("B  cut - T8e, arm_5 radius untouched",
          f"z_cut -100 mm on link_*_arm_5, ee_finger_l1/l2 x0.22",
          cut["windows"]["exec8"], cut["windows"]["full50"],
          cut["n_coverage_shortfall"], cut["worst_shortfall_mm"],
          [("link_left_arm_5 radius", "81.2500 mm, unchanged"),
           ("arm_5 spheres kept",
            f"{cut['n_arm5_spheres_kept']} of {cut['n_arm5_spheres_total']} (both arms)"),
           ("what the cut removes",
            "capsule axis z in [-0.2250, -0.1625] m; surface reached z = -0.300 m"),
           ("that loss is NOT in", "coverage_shortfall - the cut is applied outside the class")])

    block("B' cut with a gentler finger scale",
          "z_cut -100 mm, ee_finger_l1/l2 x0.26",
          cut26["windows"]["exec8"], cut26["windows"]["full50"],
          cut26["n_coverage_shortfall"], cut26["worst_shortfall_mm"], [])

    table(axes[0, 1], "two ways to the same exec8 result, and what each one gives up",
          ["path", "", "", ""], rows, [0.24, 0.24, 0.16, 0.36], colours,
          fontsize=7.6, dymax=0.042)

    # --- 손바닥 --------------------------------------------------------------------------
    rows, colours = [], {}
    for tag, lab in (("in", "apple KEPT for ee_left"),
                     ("in_authorized", "apple REMOVED for ee_left")):
        for f in (1.0, 0.22):
            r = get(tag, f)
            rows.append([f"[palm in · {lab} · finger x{f:g}]", "", "", ""])
            colours[len(rows) - 1] = figstyle.INK
            for w in ("exec8", "full50"):
                ps = [p for p in r["windows"][w]["violating_pairs"]
                      if p["link"] in ("ee_left", "ee_right")]
                if not ps:
                    rows.append([f"  {w}", "no palm pair", "", ""])
                    colours[len(rows) - 1] = figstyle.CATEGORICAL[2]
                for p in ps:
                    rows.append([f"  {w}  {p['link']}", p["obstacle_body"],
                                 f"rows {p['n_rows_slack_le0']:5d} chunks "
                                 f"{p['n_chunks_slack_le0']:3d}",
                                 f"min clr {p['min_clearance_mm']:+8.2f}  "
                                 f"slack {p['min_slack_mm']:+8.2f} mm"])
                    colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
    rows.append(["", "", "", ""])
    rows.append(["r_crit of the palm capsule groups (z_cut -100, apple removed for ee_left)",
                 "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    rc = e["r_crit_by_capsule_group"]["zcut_100mm__palm_in_authorized"]
    for w in ("exec8", "full50"):
        for k, v in rc[w].items():
            if v["link"] not in ("ee_left", "ee_right"):
                continue
            rows.append([f"  [{w}] {v['link']}", f"r@1 {v['r_at_scale_1_mm']:.4f} mm",
                         f"min centre-surf {v['min_surface_distance_mm']:8.3f}",
                         f"r_crit {v['r_crit_mm']:+8.3f} mm"])
            colours[len(rows) - 1] = (figstyle.CATEGORICAL[7] if v["r_crit_mm"] <= 0
                                      else figstyle.INK_2)
    table(axes[1, 0],
          "what the palm adds - ee_left / ee_right put into the constraint model\n"
          "     (they are not in ARM_LINKS today; gap_filling_capsules has 3 capsules each)",
          ["scope", "obstacle", "count", "worst"], rows, [0.30, 0.16, 0.26, 0.28],
          colours, fontsize=7.4, dymax=0.040)

    # --- 식/경로 -------------------------------------------------------------------------
    ax = axes[1, 1]
    ax.set_axis_off()
    ax.set_title("what was held fixed, and where the numbers live", fontsize=10.0,
                 color=figstyle.INK, loc="left", pad=10)
    m = e["model"]
    rep = e["t8a_m2_reproduction_no_cut_no_palm_finger1"]
    txt = (
        "row = (chunk c, plan step s, sphere i, scene body b)\n"
        "    clearance = surface_distance(centre_i(q_{c,s}), b) - r_i\n"
        "    slack     = clearance - 10 mm          violation <=> slack <= 0\n"
        "\n"
        "held fixed\n"
        "    arm_5 capsule_radius_scale            1.0   (untouched on this path)\n"
        "    apple rows removed for                ee_finger_l1, ee_finger_l2, link_left_arm_5\n"
        "                                          (+ ee_left in the third variant)\n"
        "    record   shadow actions_reference, 75 chunks x 50 steps\n"
        "\n"
        "the cut is an approximation\n"
        "    drop link_*_arm_5 spheres with capsule-frame z < z_cut\n"
        f"    arm_5 sphere z (m): {m['arm5_local_z_m']}\n"
        "    z_cut -0.100 and -0.1087 keep the same 3 of 5 -> identical numbers throughout\n"
        "    the spheres that stay keep r = 81.25 mm; a real shorter capsule would re-space\n"
        "    them and change the radius - that difference is not measured here\n"
        "\n"
        "palm\n"
        f"    {m['n_palm_spheres']} spheres on ee_left + ee_right, radii "
        f"{m['palm_radius_mm']} mm\n"
        f"    model with palm: {m['n_spheres_with_palm']} spheres "
        f"(without: {m['n_spheres_with_palm'] - m['n_palm_spheres']})\n"
        "\n"
        "cross-check against T8a M2 (no cut, no palm, finger x1)\n"
        f"    full50 rows {rep['full50']['n_rows_slack_le0']}, "
        f"chunks {rep['full50']['n_chunks_slack_le0']}, "
        f"min {rep['full50']['global_min_slack_mm']} mm   "
        f"(T8a: 1582 / 21 / -34.7846)\n"
        f"    exec8  rows {rep['exec8']['n_rows_slack_le0']}, "
        f"min {rep['exec8']['global_min_slack_mm']} mm   (T8a: 199 / -34.6128)\n"
        "\n"
        "    raw   outputs/verify/T8e/s1_cut_plus_finger.json / s1_curves.npz\n"
        "    T8c   outputs/verify/T8c/s3_critical_radius.json  (path A numbers)\n"
        "    input outputs/live_test/local_shadow_from_pc/frames.jsonl"
    )
    ax.text(0.0, 0.96, txt, fontsize=8.0, family="monospace", color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("T8e - cut vs thin: the same exec8 result, a different bill",
                 fontsize=12.5, color=figstyle.INK, y=0.985)
    fig.tight_layout(rect=(0, 0.01, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps({
        "grid": [{k: r[k] for k in ("key", "finger_scale", "z_cut_m", "palm_in_model",
                                    "apple_removed_for_ee_left", "radius_mm_ee_finger_l1",
                                    "radius_mm_ee_finger_l2", "n_coverage_shortfall",
                                    "worst_shortfall_mm", "n_arm5_spheres_kept",
                                    "n_arm5_spheres_total")}
                 | {w: {kk: r["windows"][w][kk] for kk in
                        ("n_rows_slack_le0", "n_chunks_slack_le0", "global_min_slack_mm",
                         "feasible_at_10mm")} for w in ("exec8", "full50")}
                 for r in rows_all],
        "path_A_thin_from_T8c": thin,
        "path_B_cut": {k: cut[k] for k in ("key", "finger_scale", "z_cut_m",
                                           "n_coverage_shortfall", "worst_shortfall_mm",
                                           "n_arm5_spheres_kept", "n_arm5_spheres_total")}
                      | {w: cut["windows"][w] for w in ("exec8", "full50")},
        "palm_r_crit": e["r_crit_by_capsule_group"],
        "t8a_m2_reproduction": rep,
        "model": m,
    }, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
