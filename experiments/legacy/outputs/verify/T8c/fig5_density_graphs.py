"""T8c figure 5 — density × 배율 그래프와 표를 한 장에.

위 줄: 배율 축 × density 4 계열. (1) 유효 반지름 mm (바닥값 점선) (2) 남은 위반 행 수
       (3) 최소 slack mm. finger 계열은 x1 과 x0.35 두 가지를 실선/점선으로.
아래 줄: (4) density 별 exec8 위반 0 에 닿는 지점과 coverage_shortfall
        (5) 길이 축 근사 (arm5 구 중 z < -0.15 제외) 대 배율
        (6) 표 — density 별 요약.
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
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8c/t8c-density-graphs.png'))
SIDE = OUT.with_suffix(".json")
MARGIN = 10.0


def main() -> None:
    s4 = json.loads((T8C / "s4_density_scale_sweep.json").read_text())
    dens = s4["densities"]
    scales = s4["arm5_scales"]
    figstyle.use_korean()
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.6))
    figstyle.style_axes(fig, axes)
    side: dict = {"margin_mm": MARGIN, "arm5_scales": scales, "panels": {}}

    def get(dn, a, f):
        for c in dn["combos"]:
            if c["arm5_scale"] == a and c["finger_scale"] == f:
                return c
        raise KeyError

    # (1) 반지름
    ax = axes[0, 0]
    for k, dn in enumerate(dens):
        n = dn["arm5_n_spheres"]
        y = [get(dn, a, 1.0)["radius_mm_link_left_arm_5"] for a in scales]
        ax.plot(scales, y, marker="o", ms=4, lw=1.7, color=figstyle.CATEGORICAL[k],
                label=f"n={n}  (spacing {dn['sphere_spacing']:g}, {dn['n_spheres_total']} spheres)")
        ax.axhline(250.0 / (n - 1) / 2, color=figstyle.CATEGORICAL[k], lw=0.8, ls=":")
    ax.set_xlabel("capsule_radius_scale on link_*_arm_5", fontsize=8.5, color=figstyle.INK_2)
    ax.set_ylabel("effective radius of link_left_arm_5  [mm]", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("the radius floor is set by the sphere count\n"
                 "dotted line = spacing/2 = 250 mm /(n-1) /2, the value s -> 0 cannot pass",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.0, loc="upper left", framealpha=0.9)
    ax.invert_xaxis()
    side["panels"]["radius"] = [
        {"n": dn["arm5_n_spheres"], "floor_mm": round(250.0 / (dn["arm5_n_spheres"] - 1) / 2, 4),
         "radius_mm": [get(dn, a, 1.0)["radius_mm_link_left_arm_5"] for a in scales]}
        for dn in dens]

    # (2)(3) 위반
    for col, (key, ylab) in enumerate(
            [("n_rows_slack_le0", "remaining rows with slack <= 0"),
             ("global_min_slack_mm", "min margin_slack over remaining rows  [mm]")]):
        ax = axes[0, col + 1]
        for k, dn in enumerate(dens):
            for f, ls, mk in ((1.0, "-", "o"), (0.35, "--", "s")):
                for w, lw, alpha in (("full50", 1.7, 1.0), ("exec8", 1.0, 0.55)):
                    if w == "exec8" and f == 1.0:
                        continue
                    y = [get(dn, a, f)["windows"][w][key] for a in scales]
                    ax.plot(scales, y, ls, marker=mk, ms=3.5, lw=lw, alpha=alpha,
                            color=figstyle.CATEGORICAL[k],
                            label=(f"n={dn['arm5_n_spheres']} · finger x{f:g} · {w}"
                                   if col == 0 else None))
        if key == "global_min_slack_mm":
            ax.axhline(0.0, color=figstyle.INK, lw=1.0)
        ax.set_xlabel("capsule_radius_scale on link_*_arm_5", fontsize=8.5, color=figstyle.INK_2)
        ax.set_ylabel(ylab, fontsize=8.5, color=figstyle.INK_2)
        ax.set_title(ylab + "\napple removed for the 3 authorized links, margin 10 mm",
                     fontsize=9.0, color=figstyle.INK)
        ax.invert_xaxis()
        if col == 0:
            ax.set_yscale("symlog", linthresh=10)
            ax.legend(fontsize=6.3, loc="lower left", framealpha=0.9, ncol=2)
    side["panels"]["violations"] = [
        {"n": dn["arm5_n_spheres"], "finger_scale": f, "window": w,
         "n_rows_slack_le0": [get(dn, a, f)["windows"][w]["n_rows_slack_le0"] for a in scales],
         "n_chunks_slack_le0": [get(dn, a, f)["windows"][w]["n_chunks_slack_le0"] for a in scales],
         "global_min_slack_mm": [get(dn, a, f)["windows"][w]["global_min_slack_mm"]
                                 for a in scales],
         "feasible_at_10mm": [get(dn, a, f)["windows"][w]["feasible_at_10mm"] for a in scales]}
        for dn in dens for f in s4["finger_scales"] for w in ("full50", "exec8")]

    # (4) coverage shortfall
    ax = axes[1, 0]
    for k, dn in enumerate(dens):
        y = [get(dn, a, 0.35)["worst_shortfall_mm"] for a in scales]
        ax.plot(scales, y, marker="o", ms=4, lw=1.6, color=figstyle.CATEGORICAL[k],
                label=f"n={dn['arm5_n_spheres']}  (finger x0.35)")
        feas = [c for c in dn["combos"] if c["windows"]["exec8"]["feasible_at_10mm"]]
        if feas:
            p = max(feas, key=lambda c: c["arm5_scale"])
            ax.plot([p["arm5_scale"]], [p["worst_shortfall_mm"]], marker="*", ms=14,
                    color=figstyle.CATEGORICAL[k], zorder=9)
    ax.set_xlabel("capsule_radius_scale on link_*_arm_5", fontsize=8.5, color=figstyle.INK_2)
    ax.set_ylabel("worst coverage_shortfall  [mm]", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("coverage given up against the URDF capsule\n"
                 "star = the largest grid scale that clears exec8 at that density",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.0, loc="upper right", framealpha=0.9)
    ax.invert_xaxis()
    side["panels"]["shortfall"] = [
        {"n": dn["arm5_n_spheres"],
         "worst_shortfall_mm_finger035": [get(dn, a, 0.35)["worst_shortfall_mm"] for a in scales],
         "n_shortfall_finger035": [get(dn, a, 0.35)["n_coverage_shortfall"] for a in scales]}
        for dn in dens]

    # (5) 길이 축 근사
    ax = axes[1, 1]
    for k, dn in enumerate(dens):
        y_full = [get(dn, a, 1.0)["windows"]["full50"]["n_rows_slack_le0"] for a in scales]
        y_short = [c["windows"]["full50"]["n_rows_slack_le0"]
                   for c in dn["length_axis_combos"]]
        ax.plot(scales, y_full, "-", marker="o", ms=3.5, lw=1.6,
                color=figstyle.CATEGORICAL[k], label=f"n={dn['arm5_n_spheres']} full capsule")
        ax.plot(scales, y_short, "--", marker="s", ms=3.5, lw=1.2, alpha=0.7,
                color=figstyle.CATEGORICAL[k],
                label=f"n={dn['arm5_n_spheres']} shortened (approx)")
    ax.set_xlabel("capsule_radius_scale on link_*_arm_5", fontsize=8.5, color=figstyle.INK_2)
    ax.set_ylabel("remaining rows with slack <= 0  (full50)", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("length axis (approximation): arm_5 spheres with capsule-frame z < -0.15 m\n"
                 "dropped, finger scale 1.0 — this cut is what the shortened capsule stands for",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=6.6, loc="upper left", framealpha=0.9, ncol=2)
    ax.set_yscale("symlog", linthresh=10)
    ax.invert_xaxis()
    side["panels"]["length_axis"] = [
        {"n": dn["arm5_n_spheres"], "n_spheres_dropped": dn["length_axis_approx"]["n_spheres_dropped"],
         "dropped_local_z_m": dn["length_axis_approx"]["dropped_local_z_m"],
         "full_capsule_rows_full50": [get(dn, a, 1.0)["windows"]["full50"]["n_rows_slack_le0"]
                                      for a in scales],
         "shortened_rows_full50": [c["windows"]["full50"]["n_rows_slack_le0"]
                                   for c in dn["length_axis_combos"]],
         "shortened_min_slack_full50": [c["windows"]["full50"]["global_min_slack_mm"]
                                        for c in dn["length_axis_combos"]],
         "shortened_min_slack_exec8": [c["windows"]["exec8"]["global_min_slack_mm"]
                                       for c in dn["length_axis_combos"]]}
        for dn in dens]

    # (6) 표
    ax = axes[1, 2]
    ax.set_axis_off()
    ax.set_title("per density: what the grid reaches", fontsize=10.0, color=figstyle.INK,
                 loc="left", pad=10)
    lines = [f"{'n':>3s} {'spacing':>8s} {'S':>5s} {'r@1':>7s} {'floor':>7s}  "
             f"{'exec8 feasible at':<26s} {'full50 rows there':>18s}"]
    rows_tbl = []
    for dn in dens:
        n = dn["arm5_n_spheres"]
        feas = [c for c in dn["combos"] if c["windows"]["exec8"]["feasible_at_10mm"]]
        if feas:
            p = max(feas, key=lambda c: c["arm5_scale"])
            what = f"arm5 x{p['arm5_scale']:g} finger x{p['finger_scale']:g} r{p['radius_mm_link_left_arm_5']:.1f}"
            there = (f"{p['windows']['full50']['n_rows_slack_le0']} rows, "
                     f"min {p['windows']['full50']['global_min_slack_mm']:+.2f}")
        else:
            what, there = "none in this grid", "-"
        r1 = dn["radius_mm_by_link_at_scale_1"]["link_left_arm_5"][0]
        lines.append(f"{n:>3d} {dn['sphere_spacing']:>8g} {dn['n_spheres_total']:>5d} "
                     f"{r1:>7.2f} {250.0/(n-1)/2:>7.2f}  {what:<26s} {there:>18s}")
        rows_tbl.append({"n": n, "spacing": dn["sphere_spacing"],
                         "n_spheres_total": dn["n_spheres_total"],
                         "arm5_r_at_scale_1_mm": r1,
                         "arm5_floor_mm": round(250.0 / (n - 1) / 2, 4),
                         "exec8_feasible_at": what, "full50_there": there})
    lines += ["", "no combination in this grid reaches 0 violations on full50;",
              "the rows that stay are ee_finger_l1 / ee_finger_l2 against the table.",
              "", "length axis (approx, finger x1, any arm5 scale):"]
    for dn in dens:
        c = dn["length_axis_combos"][0]
        lines.append(f"  n={dn['arm5_n_spheres']:>2d}  drop "
                     f"{dn['length_axis_approx']['n_spheres_dropped']:>2d} arm5 spheres -> "
                     f"full50 {c['windows']['full50']['n_rows_slack_le0']:>5d} rows "
                     f"({c['windows']['full50']['n_chunks_slack_le0']} chunks, "
                     f"min {c['windows']['full50']['global_min_slack_mm']:+.2f} mm)")
    ax.text(0.0, 0.97, "\n".join(lines), fontsize=7.6, family="monospace",
            color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    side["panels"]["summary_table"] = rows_tbl

    fig.suptitle("T8c - density x radius scale: how thin link_*_arm_5 can actually get, and "
                 "what still violates", fontsize=12.5, color=figstyle.INK, y=0.982)
    fig.text(0.5, 0.010,
             "each density has its own clearance tensor (sphere centres move when the sphere "
             "count changes) - rebuilt with the same analytic point-to-surface method as T8a, "
             "no mj_geomDistance, no rollout · row counts are not comparable across densities "
             "because the number of spheres changes; chunk counts and min slack are",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.028, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
