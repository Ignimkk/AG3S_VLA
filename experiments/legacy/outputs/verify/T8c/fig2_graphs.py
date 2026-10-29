"""T8c figure 2 — 배율 sweep 그래프.

위 줄: arm5 배율 축 (x = link_*_arm_5 의 유효 반지름 mm, 위 눈금에 배율) ×
       손가락 배율 3 계열. (1) 남은 위반 행 수 (2) 위반 chunk 수 (3) 최소 slack mm.
아래 줄: (4) 배율 -> 유효 반지름 (비선형) 과 sphere-chain 바닥값
        (5) coverage_shortfall mm (안전을 얼마 내주는가)
        (6) chunk 축 최소 slack 곡선 — 배율 몇 개를 겹쳐서.
세로 점선은 s3 가 역산한 `scale_crit`.
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
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t8c/t8c-graphs.png'))
SIDE = OUT.with_suffix(".json")
WIN = "full50"


def main() -> None:
    sw = json.loads((T8C / "s2_scale_sweep.json").read_text())
    s3 = json.loads((T8C / "s3_critical_radius.json").read_text())
    s1 = json.loads((T8C / "s1_radii_by_scale.json").read_text())
    curves = np.load(T8C / "s2_scale_curves.npz", allow_pickle=True)
    arm5 = s1["arm5_scales"]
    fing = s1["finger_scales"]
    by = {(c["arm5_scale"], c["finger_scale"]): c for c in sw["combos"]}

    figstyle.use_korean()
    fig, axes = plt.subplots(2, 3, figsize=(17.4, 9.4))
    figstyle.style_axes(fig, axes)
    side: dict = {"window_primary": WIN, "arm5_scales": arm5, "finger_scales": fing,
                  "series": [], "scale_crit": {}}

    s_crit_arm5 = {w: min(v["scale_crit"] for v in s3["groups"][w].values()
                          if v["link"] == "link_left_arm_5" and v["scale_crit"] is not None)
                   for w in ("full50", "exec8")}
    side["scale_crit"]["link_left_arm_5"] = s_crit_arm5

    metrics = [("n_rows_slack_le0", "remaining rows with slack <= 0", 0),
               ("n_chunks_slack_le0", "chunks with any slack <= 0  (of 75)", 1),
               ("global_min_slack_mm", "min margin_slack over remaining rows  [mm]", 2)]
    for key, ylab, col in metrics:
        ax = axes[0, col]
        for k, f in enumerate(fing):
            for w, ls, mk in (("full50", "-", "o"), ("exec8", "--", "s")):
                y = [by[(a, f)]["windows"][w][key] for a in arm5]
                ax.plot(arm5, y, ls, marker=mk, ms=4, lw=1.6 if w == "full50" else 1.1,
                        color=figstyle.CATEGORICAL[k],
                        label=f"finger x{f:g} · {w}" if col == 0 else None)
                if col == 0:
                    side["series"].append({"metric": key, "finger_scale": f, "window": w,
                                           "arm5_scales": arm5, "values": y})
        for w, c in (("full50", figstyle.INK), ("exec8", figstyle.INK_2)):
            ax.axvline(s_crit_arm5[w], color=c, lw=1.0, ls=":")
        if key == "global_min_slack_mm":
            ax.axhline(0.0, color=figstyle.INK, lw=1.0)
        ax.set_xlabel("capsule_radius_scale on link_*_arm_5", fontsize=8.5, color=figstyle.INK_2)
        ax.set_ylabel(ylab, fontsize=8.5, color=figstyle.INK_2)
        ax.set_title(ylab + "\n(apple removed for the 3 authorized links, margin 10 mm)",
                     fontsize=9.0, color=figstyle.INK)
        ax.invert_xaxis()
        if col == 0:
            ax.legend(fontsize=6.9, loc="lower left", framealpha=0.9, ncol=1)
            ax.annotate(f"dotted vertical = scale_crit for link_left_arm_5\n"
                        f"full50 {s_crit_arm5['full50']:.6f} · exec8 {s_crit_arm5['exec8']:.6f}",
                        (0.98, 0.98), xycoords="axes fraction", ha="right", va="top",
                        fontsize=7.2, color=figstyle.INK_2)

    # --- 배율 -> 반지름 (비선형) --------------------------------------------------------
    ax = axes[1, 0]
    ss = np.linspace(0.2, 1.05, 200)
    groups = s3["groups"]["full50"]
    off: list[str] = []
    for k, (gk, g) in enumerate(sorted(groups.items())):
        ru, fl = g["r_urdf_mm"], g["floor_mm"]
        r = (np.sqrt((ru * ss) ** 2 + fl ** 2) if g["n_spheres"] > 1 else ru * ss + fl)
        ax.plot(ss, r, lw=1.5, color=figstyle.CATEGORICAL[k % 8],
                label=f"{g['link']} r{g['r_at_scale_1_mm']:.1f} (floor {fl:.1f})")
        ax.axhline(fl, color=figstyle.CATEGORICAL[k % 8], lw=0.7, ls=":")
        if g["scale_crit"] is not None and g["scale_crit"] <= 1.0:
            ax.plot([g["scale_crit"]], [g["r_crit_mm"]], marker="*", ms=13,
                    color=figstyle.CATEGORICAL[k % 8], zorder=9)
            ax.annotate(f"s_crit {g['scale_crit']:.3f}", (g["scale_crit"], g["r_crit_mm"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7.0,
                        color=figstyle.CATEGORICAL[k % 8])
        else:
            off.append(f"{g['link']} r{g['r_at_scale_1_mm']:.1f}: "
                       + ("r_crit <= 0 (no scale)" if g["scale_crit"] is None
                          else f"s_crit {g['scale_crit']:.2f} > 1 (no shrink needed)"))
    for a in arm5:
        ax.axvline(a, color=figstyle.GRID_INK, lw=0.6)
    ax.set_xlabel("capsule_radius_scale", fontsize=8.5, color=figstyle.INK_2)
    ax.set_ylabel("effective sphere radius  [mm]", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("radius is not linear in the scale\n"
                 "r_eff(s) = sqrt((r_urdf s)^2 + (spacing/2)^2); star = r_crit (full50)",
                 fontsize=9.0, color=figstyle.INK)
    ax.set_xlim(0.2, 1.05)
    ax.legend(fontsize=6.4, loc="upper left", framealpha=0.9)
    ax.annotate("\n".join(["off this chart:"] + off), (0.98, 0.03), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=6.6, color=figstyle.INK_2)
    side["groups_off_chart"] = off
    side["radius_curve_groups"] = {gk: {k: g[k] for k in
                                        ("link", "n_spheres", "r_at_scale_1_mm", "r_urdf_mm",
                                         "floor_mm", "r_crit_mm", "scale_crit")}
                                   for gk, g in groups.items()}

    # --- coverage shortfall -------------------------------------------------------------
    ax = axes[1, 1]
    for k, f in enumerate(fing):
        y = [by[(a, f)]["worst_shortfall_mm"] for a in arm5]
        n = [by[(a, f)]["n_coverage_shortfall"] for a in arm5]
        ax.plot(arm5, y, marker="o", ms=4, lw=1.6, color=figstyle.CATEGORICAL[k],
                label=f"finger x{f:g}  (worst)")
        for a, yy, nn in zip(arm5, y, n):
            ax.annotate(f"{nn}", (a, yy), fontsize=6.4, color=figstyle.INK_2,
                        ha="center", va="bottom")
    ax.axvline(s_crit_arm5["full50"], color=figstyle.INK, lw=1.0, ls=":")
    ax.set_xlabel("capsule_radius_scale on link_*_arm_5", fontsize=8.5, color=figstyle.INK_2)
    ax.set_ylabel("worst coverage_shortfall  [mm]", fontsize=8.5, color=figstyle.INK_2)
    ax.set_title("coverage lost against the URDF capsule\n"
                 "number above each marker = how many capsules fall short",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.2, loc="upper right", framealpha=0.9)
    ax.invert_xaxis()
    side["coverage_shortfall"] = [{"finger_scale": f,
                                   "arm5_scales": arm5,
                                   "worst_shortfall_mm": [by[(a, f)]["worst_shortfall_mm"]
                                                          for a in arm5],
                                   "n_shortfall": [by[(a, f)]["n_coverage_shortfall"]
                                                   for a in arm5]} for f in fing]

    # --- chunk 축 곡선 -------------------------------------------------------------------
    ax = axes[1, 2]
    show = [(1.0, 1.0), (0.6, 1.0), (0.458, None), (0.4, 0.6), (0.3, 0.35)]
    k = 0
    for a, f in show:
        key = None
        if f is not None and (a, f) in by:
            key = by[(a, f)]["key"]
        if key is None:
            continue
        y = np.asarray(curves[f"{key}__{WIN}"], np.float64)
        ax.plot(np.arange(y.size), y, lw=1.4, color=figstyle.CATEGORICAL[k % 8],
                label=f"arm5 x{a:g} · finger x{f:g}")
        side.setdefault("per_chunk_min_slack_mm", {})[key] = [round(float(v), 4) for v in y]
        k += 1
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.set_xlabel("chunk (shadow, 75 planning records)", fontsize=8.5, color=figstyle.INK_2)
    ax.set_ylabel("min margin_slack over remaining rows  [mm]", fontsize=8.5,
                  color=figstyle.INK_2)
    ax.set_title(f"per-chunk worst remaining row, {WIN}\nmargin 10 mm, apple removed for the "
                 f"3 authorized links", fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.0, loc="lower right", framealpha=0.9)

    fig.suptitle("T8c - shrinking link_*_arm_5 and the left fingertips: violations left, and "
                 "coverage given up", fontsize=12.5, color=figstyle.INK, y=0.985)
    fig.text(0.5, 0.010,
             "every point is the T8a clearance tensor re-read with a new radius "
             "(surface_distance = clearance + r) - no new distance computation, no MuJoCo, "
             "no rollout · radii read from UrdfSphereChain at each scale",
             ha="center", fontsize=8.3, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.026, 1, 0.960))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
