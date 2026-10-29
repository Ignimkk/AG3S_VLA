"""T10a figure 2 — T9 (목적함수 켠 것) 대 T10 (끈 것) 를 같은 축에.

(1) 손끝-사과 거리 (정책 원본 · TO 출력) 시계열
(2) chunk 별 TO 편차 (deg) 와 trust radius 비율
(3) 접근 구간의 최근접 body (사과 제외) 분포
(4) P2 margin sweep — 정책 원본이 건드리는 행 수
(5) P1 기록 ESDF 거리 대 오프라인 참값 거리
(6) P3 trust radius 비율 시계열
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import collections
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
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t10a/t10a-graphs.png'))
SIDE = OUT.with_suffix(".json")
APPROACH_T = 200
COL = {"T9": figstyle.CATEGORICAL[0], "T10": figstyle.CATEGORICAL[1]}


def main() -> None:
    ana = json.loads((T10A / "p1to4.json").read_text())
    p1, p2, p3, p4 = (ana["p1_what_blocks_the_fingers"], ana["p2_margin_sweep"],
                      ana["p3_trust_region"], ana["p4_t9_vs_t10"])
    figstyle.use_korean()
    fig, axes = plt.subplots(2, 3, figsize=(18.6, 9.8))
    figstyle.style_axes(fig, axes)
    side: dict = {"approach_t": APPROACH_T, "panels": {}}

    # (1)
    ax = axes[0, 0]
    for run in ("T9", "T10"):
        r = p4["runs"][run]
        t = np.asarray(r["t_step"])
        ax.plot(t, r["tip_to_apple_reference_mm"], color=COL[run], lw=1.9,
                label=f"{run} policy reference")
        ax.plot(t, r["tip_to_apple_refined_mm"], color=COL[run], lw=1.2, ls="--",
                label=f"{run} TO output")
    ax.axvline(APPROACH_T, color=figstyle.INK_2, lw=1.0, ls=":")
    ax.annotate("approach window ends", (APPROACH_T, 420), rotation=90, fontsize=7.4,
                color=figstyle.INK_2, va="top", ha="right")
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.axhline(10.0, color=figstyle.CATEGORICAL[7], lw=1.0, ls="--")
    ax.set_ylim(-20, 470)
    ax.set_xlabel("t_step", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("left fingertip to apple surface  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("P4 - fingertip to apple, T9 vs T10\n"
                 f"T9 reference min {p4['runs']['T9']['tip_to_apple_reference_min_mm']:+.2f} / "
                 f"TO {p4['runs']['T9']['tip_to_apple_refined_min_mm']:+.2f} · "
                 f"T10 reference {p4['runs']['T10']['tip_to_apple_reference_min_mm']:+.2f} / "
                 f"TO {p4['runs']['T10']['tip_to_apple_refined_min_mm']:+.2f} mm",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
    side["panels"]["tip_to_apple"] = {r: p4["runs"][r] for r in ("T9", "T10")}

    # (2)
    ax = axes[0, 1]
    for run in ("T9", "T10"):
        t = np.asarray(p4["runs"][run]["t_step"])
        ax.plot(t, p3["runs"][run]["per_chunk_max_deg"], color=COL[run], lw=1.7,
                label=f"{run}  median all {p4['runs'][run]['to_deviation_deg_median_all']:.3f}, "
                      f"approach {p4['runs'][run]['to_deviation_deg_median_approach']:.3f} deg")
    ax.axhline(np.degrees(p3["trust_radius_rad"]), color=figstyle.CATEGORICAL[7], lw=1.2,
               ls="--")
    ax.annotate(f"trust_radius {np.degrees(p3['trust_radius_rad']):.2f} deg "
                f"(config.py:286)", (0, np.degrees(p3["trust_radius_rad"])),
                textcoords="offset points", xytext=(4, 4), fontsize=7.4,
                color=figstyle.CATEGORICAL[7])
    ax.axvline(APPROACH_T, color=figstyle.INK_2, lw=1.0, ls=":")
    ax.set_xlabel("t_step", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("max |TO - reference| over the exec window  [deg]", fontsize=8.6,
                  color=figstyle.INK_2)
    ax.set_title("P3/P4 - how far TO moves the chunk\n"
                 f"T10 outlier: chunk {p3['outlier_chunk_detail']['chunk']} at t="
                 f"{p3['outlier_chunk_detail']['t_step']}, "
                 f"{p3['outlier_chunk_detail']['max_deg']:.4f} deg = "
                 f"{p3['outlier_chunk_detail']['max_rad']:.5f} rad",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
    side["panels"]["to_deviation_deg"] = {r: p3["runs"][r]["per_chunk_max_deg"]
                                          for r in ("T9", "T10")}

    # (3)
    ax = axes[0, 2]
    labels = sorted(set(list(p4["runs"]["T9"]["nearest_body_counts_approach"])
                        + list(p4["runs"]["T10"]["nearest_body_counts_approach"])))
    w = 0.38
    for k, run in enumerate(("T9", "T10")):
        vals = [p4["runs"][run]["nearest_body_counts_approach"].get(b, 0) for b in labels]
        ax.bar(np.arange(len(labels)) + (k - 0.5) * w, vals, width=w, color=COL[run], label=run)
        for j, v in enumerate(vals):
            if v:
                ax.annotate(str(v), (j + (k - 0.5) * w, v), ha="center",
                            textcoords="offset points", xytext=(0, 3), fontsize=8.0,
                            color=figstyle.INK_2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("approach chunks", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("P1/P4 - nearest ground-truth body to a gripper sphere\n"
                 "approach window, TO output, apple excluded (it is carved out)",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.6, framealpha=0.9)
    side["panels"]["nearest_body_approach"] = {
        r: p4["runs"][r]["nearest_body_counts_approach"] for r in ("T9", "T10")}

    # (4) margin sweep
    ax = axes[1, 0]
    margins = [10.0, 5.0, 2.0, 1.0, 0.0]
    for tag, c, lab in (("all_bodies", figstyle.CATEGORICAL[7], "all bodies (apple included)"),
                        ("apple_excluded", figstyle.CATEGORICAL[2],
                         "apple excluded (what the gripper links see)")):
        y = [p2["sweeps"][tag][f"{mg:g}mm"]["n_rows_slack_le0"] for mg in margins]
        ax.plot(margins, y, marker="o", ms=6, lw=1.8, color=c, label=lab)
        crit = p2["sweeps"][tag]["critical_margin_mm"]
        ax.axvline(crit["value"], color=c, lw=1.0, ls=":")
        ax.annotate(f"critical margin {crit['value']:.3f} mm\n"
                    f"({crit['argmin_body']}, {crit['argmin_link']}, t={crit['argmin_t_step']})",
                    (crit["value"], max(1, max(y)) * 0.55), fontsize=7.4, color=c,
                    ha="left" if tag == "all_bodies" else "right")
    ax.invert_xaxis()
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("esdf margin  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("rows of the POLICY REFERENCE with slack <= 0", fontsize=8.6,
                  color=figstyle.INK_2)
    ax.set_title("P2 - how far the margin has to drop for the policy's own chunk to be clear\n"
                 f"approach window, gripper spheres, {p2['sweeps']['all_bodies']['10mm']['n_rows_total']} rows",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.4, loc="upper right", framealpha=0.9)
    side["panels"]["margin_sweep"] = p2["sweeps"]

    # (5) ESDF vs ground truth
    ax = axes[1, 1]
    rows = [x for x in p1["T10"]["recorded_worst_row_named_offline"]["rows"]
            if not x["is_attached_query_point"]]
    t = [x["t_step"] for x in rows]
    ax.plot(t, [x["esdf_d_minus_r_mm"] for x in rows], marker="o", ms=4, lw=1.7,
            color=figstyle.CATEGORICAL[0], label="ESDF  d - r  (recorded clearance + margin)")
    ax.plot(t, [x["nearest_mm_all"] for x in rows], marker="s", ms=4, lw=1.7,
            color=figstyle.CATEGORICAL[7], label="ground truth nearest body (apple included)")
    ax.plot(t, [x["nearest_mm_apple_excluded"] for x in rows], marker="^", ms=4, lw=1.2,
            ls="--", color=figstyle.CATEGORICAL[2], label="ground truth, apple excluded")
    ax.axhline(10.0, color=figstyle.CATEGORICAL[7], lw=1.0, ls=":")
    ax.annotate("margin 10 mm", (max(t), 10.0), ha="right", va="bottom", fontsize=7.4,
                color=figstyle.CATEGORICAL[7])
    ax.set_xlabel("t_step (approach window only)", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("distance from the same sphere  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ev = p1["T10"]["recorded_worst_row_named_offline"]["esdf_vs_ground_truth"]
    ax.set_title("P1 - the field the TO saw vs the ground-truth scene, same sphere\n"
                 f"offline minus ESDF: min {ev['offline_minus_esdf_mm_all_bodies']['min']:+.2f}, "
                 f"median {ev['offline_minus_esdf_mm_all_bodies']['median']:+.2f}, "
                 f"max {ev['offline_minus_esdf_mm_all_bodies']['max']:+.2f} mm",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
    side["panels"]["esdf_vs_ground_truth"] = rows

    # (6) trust ratio
    ax = axes[1, 2]
    for run in ("T9", "T10"):
        t = np.asarray(p4["runs"][run]["t_step"])
        ax.plot(t, p3["runs"][run]["per_chunk_ratio"], color=COL[run], lw=1.7, label=run)
    ax.axhline(1.0, color=figstyle.CATEGORICAL[7], lw=1.2, ls="--")
    ax.annotate("the whole trust region step", (0, 1.0), textcoords="offset points",
                xytext=(4, 4), fontsize=7.4, color=figstyle.CATEGORICAL[7])
    ax.axvline(APPROACH_T, color=figstyle.INK_2, lw=1.0, ls=":")
    ax.axvspan(0, APPROACH_T, color=figstyle.CATEGORICAL[0], alpha=0.05)
    ax.set_xlabel("t_step", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("max |dq| / trust_radius", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("P3 - is the trust region binding\n"
                 f"approach max: T9 {p3['runs']['T9']['approach_window']['max_ratio']:.3f}, "
                 f"T10 {p3['runs']['T10']['approach_window']['max_ratio']:.3f} · "
                 f"overall max T10 {p3['runs']['T10']['ratio_to_trust_radius_max']:.4f}",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.6, loc="upper left", framealpha=0.9)
    side["panels"]["trust_ratio"] = {r: p3["runs"][r]["per_chunk_ratio"] for r in ("T9", "T10")}

    fig.suptitle("T10a - objective off (T10) vs objective on (T9): what changed and what did not",
                 fontsize=12.6, color=figstyle.INK, y=0.985)
    fig.text(0.5, 0.010,
             "T10 differs from T9 only in the objective weights (w_track 1, w_smooth 0, "
             "w_continuity 0, w_slack 1000); the sphere model, the target policy and the 10 mm "
             "margin are identical · all offline clearances are PRE-margin ground truth",
             ha="center", fontsize=8.2, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.024, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
