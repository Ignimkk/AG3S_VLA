"""T8c figure 4 — 길이 축. 자르는 것과 얇게 하는 것을 나란히.

왼쪽 위 : `link_left_arm_5` capsule 의 축 도식 — 구 5 개, URDF capsule, 해부학 landmark,
          z_cut 네 개가 어디를 자르는지.
오른쪽 위: z_cut 별 남는 위반 (full50 · exec8) 과 잃는 덮개 길이.
왼쪽 아래: 구 개수 축 — sphere_spacing 을 바꿨을 때 n 과 유효 반지름, 배율 바닥값.
오른쪽 아래: 표 — z_cut 별 남는 쌍 · 손바닥 덮개 · 배율 축과의 대조.
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
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8c/t8c-length-axis.png'))
SIDE = OUT.with_suffix(".json")


def table(ax, title, cols, rows, widths, colours=None, fontsize=7.6, head=8.2, dymax=0.06):
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.0, color=figstyle.INK, loc="left", pad=10)
    x0 = np.concatenate([[0.0], np.cumsum(widths)])
    y, dy = 0.95, min(dymax, 0.90 / max(len(rows) + 1, 1))
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
    s4 = json.loads((T8C / "s4_length_axis.json").read_text())
    s5 = json.loads((T8C / "s5_sphere_count_axis.json").read_text())
    s2 = json.loads((T8C / "s2_scale_sweep.json").read_text())
    cap = s4["capsule_link_left_arm_5"]
    lm = s4["landmark_z_in_arm5_frame"]

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 2, figsize=(19.2, 10.4))
    figstyle.style_axes(fig, axes)

    # --- 축 도식 -------------------------------------------------------------------------
    ax = axes[0, 0]
    r = cap["sphere_radius_mm"]
    for k, z in enumerate(cap["sphere_axis_z_mm"]):
        ax.add_patch(plt.Circle((z, 0.0), r, facecolor=figstyle.CATEGORICAL[1],
                                edgecolor="none", alpha=0.22, zorder=2))
        ax.add_patch(plt.Circle((z, 0.0), r, facecolor="none",
                                edgecolor=figstyle.CATEGORICAL[1], lw=0.9, zorder=3))
        ax.plot([z], [0], marker="+", ms=7, color=figstyle.CATEGORICAL[1], zorder=4)
        ax.annotate(f"{z:g}", (z, -r - 9), ha="center", fontsize=7.0,
                    color=figstyle.CATEGORICAL[1])
    a0, a1 = cap["axis_z_range_mm"]
    ax.plot([a0, a1], [0, 0], lw=2.0, color=figstyle.INK, zorder=5)
    ax.add_patch(plt.Rectangle((a0, -cap["radius_mm"]), a1 - a0, 2 * cap["radius_mm"],
                               facecolor="none", edgecolor=figstyle.INK_2, lw=1.2, ls="--",
                               zorder=4))
    ax.annotate(f"URDF capsule  r {cap['radius_mm']:.0f} · L {cap['length_mm']:.0f} mm\n"
                f"surface z {cap['surface_z_range_mm'][0]:.0f} .. "
                f"{cap['surface_z_range_mm'][1]:.0f} mm",
                (a1 + 8, cap["radius_mm"] + 4), fontsize=8.0, color=figstyle.INK_2)
    for k, (ln, v) in enumerate(lm.items()):
        z = v["z_mm"]
        ax.axvline(z, color=figstyle.CATEGORICAL[2], lw=1.0, ls=":")
        ax.annotate(f"{ln}  {z:.1f}", (z, -110 - 16 * (k % 2)), rotation=0, fontsize=7.2,
                    ha="center", color=figstyle.CATEGORICAL[2],
                    bbox=dict(boxstyle="round,pad=0.14", facecolor=figstyle.SURFACE,
                              edgecolor="none", alpha=0.85))
    for k, res in enumerate([x for x in s4["z_cut_results"] if x["z_cut_mm"] is not None]):
        ax.axvline(res["z_cut_mm"], color=figstyle.CATEGORICAL[7], lw=1.4)
        ax.annotate(f"z_cut {res['z_cut_mm']:g}", (res["z_cut_mm"], 100 + 20 * (k % 2)),
                    rotation=90, fontsize=7.2, va="bottom", ha="center",
                    color=figstyle.CATEGORICAL[7],
                    bbox=dict(boxstyle="round,pad=0.10", facecolor=figstyle.SURFACE,
                              edgecolor="none", alpha=0.85))
    ax.set_xlim(-330, 200)
    ax.set_ylim(-150, 195)
    ax.set_aspect("equal")
    ax.set_xlabel("z in the link_left_arm_5 frame  [mm]   (-z is distal)", fontsize=8.5,
                  color=figstyle.INK_2)
    ax.set_title(f"link_left_arm_5: {cap['n_spheres']} spheres of r {r:.2f} mm, "
                 f"spacing {cap['sphere_spacing_mm']:.1f} mm\n"
                 f"the capsule surface reaches z {cap['surface_z_range_mm'][0]:.0f} mm; "
                 f"the finger root sits at {lm['ee_finger_l1']['z_mm']:.1f} mm",
                 fontsize=9.4, color=figstyle.INK)

    # --- z_cut 별 위반과 덮개 -------------------------------------------------------------
    ax = axes[0, 1]
    res = s4["z_cut_results"]
    xs = np.arange(len(res))
    lab = ["none" if r_["z_cut_mm"] is None else f"{r_['z_cut_mm']:g}" for r_ in res]
    w = 0.38
    ax.bar(xs - w / 2, [r_["windows"]["full50"]["n_rows_slack_le0"] for r_ in res], w,
           color=figstyle.CATEGORICAL[0], label="rows <= 0, full50")
    ax.bar(xs + w / 2, [r_["windows"]["exec8"]["n_rows_slack_le0"] for r_ in res], w,
           color=figstyle.CATEGORICAL[1], label="rows <= 0, exec8")
    for i, r_ in enumerate(res):
        ax.annotate(f"{r_['windows']['full50']['min_slack_mm']:.1f}",
                    (i - w / 2, r_["windows"]["full50"]["n_rows_slack_le0"]), ha="center",
                    va="bottom", fontsize=7.0, color=figstyle.INK_2)
    ax2 = ax.twinx()
    ax2.plot(xs, [r_["uncovered_distal_length_mm"] for r_ in res], marker="o", ms=5,
             color=figstyle.CATEGORICAL[7], lw=1.6, label="uncovered distal length")
    ax2.set_ylabel("uncovered distal length on the axis  [mm]", fontsize=8.5,
                   color=figstyle.CATEGORICAL[7])
    ax2.tick_params(colors=figstyle.CATEGORICAL[7], labelsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(lab)
    ax.set_xlabel("z_cut  [mm]  (spheres with centre distal to the cut are removed)",
                  fontsize=8.5, color=figstyle.INK_2)
    ax.set_ylabel("remaining rows with slack <= 0", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("cutting the capsule instead of thinning it\n"
                 "radius stays 81.25 mm · apple removed for the 3 authorized links · "
                 "margin 10 mm · number above bar = min slack mm",
                 fontsize=9.4, color=figstyle.INK)
    ax.legend(fontsize=7.4, loc="upper right", framealpha=0.9)

    # --- 구 개수 축 -----------------------------------------------------------------------
    ax = axes[1, 0]
    for k, g in enumerate(s5["geometries"]):
        xs_ = [c["arm5_scale"] for c in g["combos"] if c["finger_scale"] == 1.0]
        ys_ = [c["radius_mm_link_left_arm_5"] for c in g["combos"] if c["finger_scale"] == 1.0]
        ax.plot(xs_, ys_, marker="o", ms=4, lw=1.5, color=figstyle.CATEGORICAL[k % 8],
                label=f"spacing {g['sphere_spacing']:g} · cap {g['max_spheres_per_capsule']} "
                      f"-> n={g['n_spheres_link_left_arm_5']} (S={g['n_spheres_total']})")
        ax.axhline(min(ys_), color=figstyle.CATEGORICAL[k % 8], lw=0.7, ls=":")
    ax.set_xlabel("capsule_radius_scale on link_*_arm_5", fontsize=8.5, color=figstyle.INK_2)
    ax.set_ylabel("effective sphere radius  [mm]", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("the sphere-count axis: the radius floor is spacing/2, so more spheres lower "
                 "the floor\nn = clip(ceil(L / (sphere_spacing * r)) + 1, 1, "
                 "max_spheres_per_capsule)", fontsize=9.4, color=figstyle.INK)
    ax.legend(fontsize=7.0, loc="upper left", framealpha=0.9)
    ax.invert_xaxis()

    # --- 표 ------------------------------------------------------------------------------
    rows, colours = [], {}
    rows.append(["[z_cut sweep] radius 81.25 mm kept", "", "", "", ""])
    colours[0] = figstyle.INK
    for r_ in res:
        f50, e8 = r_["windows"]["full50"], r_["windows"]["exec8"]
        pal = r_["landmark_coverage_3d"]["ee_left"]
        rows.append([
            "  z_cut " + ("  none" if r_["z_cut_mm"] is None
                        else f"{r_['z_cut_mm']:6.1f}"),
            f"{r_['n_arm5_spheres_kept_per_arm']}/5 kept",
            f"f50 {f50['n_rows_slack_le0']:5d}/{f50['n_chunks_slack_le0']:2d} "
            f"{f50['min_slack_mm']:7.2f}",
            f"e8 {e8['n_rows_slack_le0']:4d}/{e8['n_chunks_slack_le0']:2d} "
            f"{e8['min_slack_mm']:7.2f}",
            f"uncov {r_['uncovered_distal_length_mm']:6.1f} mm · palm "
            f"{pal['n_probe_rows_covered']}/{pal['n_probe_rows']}"])
        colours[len(rows) - 1] = (figstyle.CATEGORICAL[7]
                                  if pal["n_probe_rows_covered"] < pal["n_probe_rows"]
                                  else figstyle.INK_2)
    rows.append(["", "", "", "", ""])
    rows.append(["  pairs left at z_cut -100 mm", "", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for p in [x for x in res if x["z_cut_mm"] == -100.0][0]["windows"]["full50"][
            "violating_pairs"]:
        rows.append([f"    {p['link']}", p["obstacle_body"],
                     f"rows {p['n_rows_slack_le0']:5d}", f"chunks {p['n_chunks_slack_le0']:3d}",
                     f"min clr {p['min_clearance_mm']:+7.2f}  slack {p['min_slack_mm']:+7.2f}"])
        colours[len(rows) - 1] = figstyle.CATEGORICAL[7]
    rows.append(["", "", "", "", ""])
    rows.append(["[for contrast] radius axis, no cut", "", "", "", ""])
    colours[len(rows) - 1] = figstyle.INK
    for c in s2["combos"]:
        if c["finger_scale"] != 1.0 or c["arm5_scale"] not in (1.0, 0.6, 0.4):
            continue
        f50, e8 = c["windows"]["full50"], c["windows"]["exec8"]
        rows.append([f"    arm5 x{c['arm5_scale']:g}",
                     f"r {c['radius_mm_link_left_arm_5'][0]:.2f} mm",
                     f"f50 {f50['n_rows_slack_le0']:5d}/{f50['n_chunks_slack_le0']:2d} "
                     f"{f50['global_min_slack_mm']:7.2f}",
                     f"e8 {e8['n_rows_slack_le0']:4d}/{e8['n_chunks_slack_le0']:2d} "
                     f"{e8['global_min_slack_mm']:7.2f}",
                     f"shortfall {c['worst_shortfall_mm']:.1f} mm x "
                     f"{c['n_coverage_shortfall']}"])
    table(axes[1, 1],
          "cut vs thin, on the same rows\n     red = the palm (ee_left) is no longer inside any "
          "sphere of the constraint model",
          ["scope", "detail", "full50 rows/chunks min", "exec8 rows/chunks min", "cost"],
          rows, [0.21, 0.15, 0.24, 0.22, 0.18], colours, fontsize=7.4, dymax=0.048)

    fig.suptitle("T8c - the length axis: cutting link_*_arm_5 at the wrist, and the sphere-count "
                 "axis behind the radius floor", fontsize=12.5, color=figstyle.INK, y=0.985)
    fig.text(0.5, 0.010,
             "the z_cut sweep is an approximation: there is no flag to shorten a capsule, so it "
             "is done by dropping the arm_5 spheres whose centre is distal to the cut, at radius "
             "81.25 mm unchanged\n"
             "a really shortened capsule would carry fewer spheres and therefore a different "
             "radius · ee_left / ee_right are not in the constraint model's link set, so only "
             "the arm_5 lump covers the palm today",
             ha="center", fontsize=8.2, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.026, 1, 0.955))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps({
        "capsule_link_left_arm_5": cap,
        "landmark_z_in_arm5_frame": lm,
        "z_cut_results": [{k: v for k, v in r_.items() if k != "landmark_coverage_3d"}
                          | {"landmark_coverage_3d": {
                              ln: {kk: vv for kk, vv in val.items() if kk != "rows"}
                              for ln, val in r_["landmark_coverage_3d"].items()}}
                          for r_ in res],
        "sphere_count_axis": [{k: v for k, v in g.items() if k != "combos"}
                              | {"combos": [{kk: vv for kk, vv in c.items() if kk != "windows"}
                                            | {"windows": {w: {k2: v2 for k2, v2
                                                               in c["windows"][w].items()
                                                               if k2 != "violating_pairs"}
                                                           for w in c["windows"]}}
                                            for c in g["combos"]]}
                              for g in s5["geometries"]],
        "constraint_model_link_set": s4["constraint_model_link_set"],
    }, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
