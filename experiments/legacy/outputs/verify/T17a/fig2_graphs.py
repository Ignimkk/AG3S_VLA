"""T17a figure 2 — 접근 구간 그래프. 무엇이 미는가 · 얼마나 미는가 · 그리퍼가 닫히는 순간."""
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

T17A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T17a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t17a/t17a-graphs.png'))
SIDE = OUT.with_suffix(".json")
BODY_COL = {"apple": figstyle.CATEGORICAL[7], "crate": figstyle.CATEGORICAL[0],
            "table": figstyle.CATEGORICAL[3], "None": figstyle.INK_2}


def main() -> None:
    a = json.loads((T17A / "r1to4.json").read_text())
    r2, r3, r1, r4 = (a["r2_what_surface"], a["r3_if_not_the_apple"],
                      a["r1_what_to_pushed"], a["r4_gripper_close"])
    per = a["per_chunk"]
    APPT = a["approach_window_t_step"][1]
    rows = r3["rows"]
    t = [r["t_step"] for r in rows]
    figstyle.use_korean()
    fig, axes = plt.subplots(2, 3, figsize=(18.6, 9.8))
    figstyle.style_axes(fig, axes)
    side: dict = {"approach_t": a["approach_window_t_step"], "panels": {}}

    # (1) 무엇이 묶었나 + ESDF 대 참값
    ax = axes[0, 0]
    ax.plot(t, [r["esdf_distance_mm"] for r in rows], color=figstyle.CATEGORICAL[0], lw=2.0,
            marker="o", ms=4, label="ESDF distance at the query point (d = clearance + r + margin)")
    ax.plot(t, [r["gt_apple_mm"] for r in rows], color=BODY_COL["apple"], lw=1.5, ls="--",
            marker="s", ms=3.5, label="ground truth: apple surface")
    ax.plot(t, [r["gt_distance_by_body_mm"].get("crate") for r in rows],
            color=BODY_COL["crate"], lw=1.5, ls="-.", marker="^", ms=3.5,
            label="ground truth: crate surface")
    ax.plot(t, [r["gt_distance_by_body_mm"].get("table") for r in rows],
            color=BODY_COL["table"], lw=1.1, ls=":", label="ground truth: table surface")
    for r in rows:
        ax.plot([r["t_step"]], [r["esdf_distance_mm"]], marker="*", ms=13,
                color=BODY_COL.get(str(r["esdf_surface_body"]), figstyle.INK), zorder=9)
    ax.axhline(10.0, color=figstyle.CATEGORICAL[6], lw=1.0, ls="--")
    ax.annotate("margin 10 mm", (t[-1], 10.0), ha="right", va="bottom", fontsize=7.4,
                color=figstyle.CATEGORICAL[6])
    ax.set_xlabel("t_step (approach)", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("distance from the query point  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("R2/R3 - what the tightest row was bound to\n"
                 f"star colour = the body the gradient walk lands on: "
                 f"crate {r2['esdf_surface_body_counts'].get('crate', 0)}, "
                 f"apple {r2['esdf_surface_body_counts'].get('apple', 0)} of "
                 f"{r2['n_rows_with_mvp']} chunks", fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=6.9, loc="upper left", framealpha=0.9)
    side["panels"]["r2_distances"] = rows

    # (2) 어느 link 이 가장 빡빡했나 + clearance
    ax = axes[0, 1]
    cl = [r["clearance_mm_post_margin"] for r in rows]
    cols = [BODY_COL.get(str(r["esdf_surface_body"]), figstyle.INK_2) for r in rows]
    ax.bar(t, cl, width=5.5, color=cols)
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    for r in rows:
        ax.annotate(r["mvp_link"].replace("ee_", ""), (r["t_step"], r["clearance_mm_post_margin"]),
                    rotation=90, fontsize=6.6, ha="center",
                    va="bottom" if r["clearance_mm_post_margin"] >= 0 else "top",
                    color=figstyle.INK_2)
    ax.set_xlabel("t_step (approach)", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("tightest row, POST-margin  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("R2 - the tightest row per chunk, coloured by the surface\n"
                 "0 means the 10 mm margin is exactly used up; below 0 is a violation",
                 fontsize=9.0, color=figstyle.INK)

    # (3) grounding 이 제자리였나
    ax = axes[0, 2]
    app = [p for p in per if p["t_step"] <= APPT]
    tt = [p["t_step"] for p in app]
    ax.plot(tt, [p.get("centroid_to_apple_mm") for p in app], color=figstyle.CATEGORICAL[4],
            lw=1.8, marker="o", ms=3.5, label="grounding centroid to apple")
    ax.plot(tt, [p.get("ball_centre_to_apple_mm") for p in app], color=figstyle.CATEGORICAL[2],
            lw=1.5, ls="--", marker="s", ms=3.5, label="exclusion ball centre to apple")
    ax.plot(tt, [p.get("ball_radius_mm") for p in app], color=figstyle.CATEGORICAL[3],
            lw=1.2, ls=":", label="exclusion ball radius")
    bad = [(p["t_step"], p.get("centroid_to_apple_mm")) for p in app
           if p.get("centroid_nearest_is_apple") is False]
    if bad:
        ax.scatter([x for x, _ in bad], [y for _, y in bad], s=40, facecolor="none",
                   edgecolor=figstyle.CATEGORICAL[7], lw=1.5, zorder=9,
                   label="centroid's nearest object is not the apple")
    ax.axhline(33.3, color=figstyle.CATEGORICAL[7], lw=1.0, ls="--")
    ax.annotate("apple radius 33.3 mm", (tt[0], 33.3), textcoords="offset points",
                xytext=(4, 4), fontsize=7.4, color=figstyle.CATEGORICAL[7])
    ax.set_xlabel("t_step (approach)", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("mm", fontsize=8.6, color=figstyle.INK_2)
    g = r2["grounding"]
    ax.set_title(f"R2 - was the exclusion in place\n"
                 f"centroid nearest is the apple on {g['n_centroid_nearest_is_apple']}/"
                 f"{len(app)}, ball contains the apple centre on "
                 f"{g['n_ball_contains_apple_centre']}/{len(app)}, apple in the fine window on "
                 f"{g['n_apple_in_fine_window']}/{len(app)}", fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.0, loc="upper left", framealpha=0.9)
    side["panels"]["r2_grounding"] = [{k: p.get(k) for k in
                                       ("t_step", "centroid_to_apple_mm",
                                        "centroid_nearest_object", "ball_centre_to_apple_mm",
                                        "ball_radius_mm", "ball_contains_apple_centre",
                                        "apple_in_fine_window")} for p in app]

    # (4) R1 — TO 가 얼마나 밀었나
    ax = axes[1, 0]
    ax.plot(tt, [p["to_max_abs_deg"] for p in app], color=figstyle.CATEGORICAL[0], lw=1.9,
            marker="o", ms=4, label="max |TO - reference| over the exec window")
    ax.axhline(np.degrees(0.15), color=figstyle.CATEGORICAL[7], lw=1.1, ls="--")
    ax.annotate("trust_radius 8.594 deg (config.py:286)", (tt[0], np.degrees(0.15)),
                textcoords="offset points", xytext=(4, 4), fontsize=7.4,
                color=figstyle.CATEGORICAL[7])
    ax2 = ax.twinx()
    ax2.plot(tt, [p["to_tip_displacement_mm_max"] for p in app],
             color=figstyle.CATEGORICAL[2], lw=1.5, ls="-.",
             label="fingertip displacement it would cause")
    ax2.set_ylabel("fingertip displacement  [mm]", fontsize=8.6, color=figstyle.CATEGORICAL[2])
    ax2.tick_params(labelsize=8, colors=figstyle.CATEGORICAL[2])
    ax.set_xlabel("t_step (approach)", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("joint deviation  [deg]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title(f"R1 - how hard TO pushed (shadow: it is computed, not executed)\n"
                 f"median {r1['to_max_abs_deg']['median']:.3f} deg, max "
                 f"{r1['to_max_abs_deg']['max']:.3f} deg at t="
                 f"{r1['to_max_abs_deg']['argmax_t_step']}; tip median "
                 f"{r1['tip_displacement_mm']['median']:.2f} mm, max "
                 f"{r1['tip_displacement_mm']['max']:.2f} mm", fontsize=9.0, color=figstyle.INK)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.2, loc="upper left", framealpha=0.9)
    side["panels"]["r1"] = [{k: p.get(k) for k in
                             ("t_step", "to_max_abs_deg", "to_largest_joint",
                              "to_largest_contrib_joint", "to_tip_displacement_mm_max")}
                            for p in app]

    # (5) R1 — 관절별
    ax = axes[1, 1]
    js = list(r1["per_joint_max_over_approach_deg"])
    w = 0.38
    ax.bar(np.arange(len(js)) - w / 2,
           [r1["per_joint_max_over_approach_deg"][j] for j in js], width=w,
           color=figstyle.CATEGORICAL[3], label="max joint deviation (deg)")
    ax2 = ax.twinx()
    ax2.bar(np.arange(len(js)) + w / 2,
            [r1["per_joint_max_contrib_mm"][j] for j in js], width=w,
            color=figstyle.CATEGORICAL[0], label="max fingertip contribution (mm)")
    ax2.set_ylabel("fingertip contribution  [mm]", fontsize=8.6, color=figstyle.CATEGORICAL[0])
    ax2.tick_params(labelsize=8, colors=figstyle.CATEGORICAL[0])
    ax.set_xticks(range(len(js)))
    ax.set_xticklabels([j.replace("left_arm_", "L") for j in js], fontsize=9)
    ax.set_ylabel("joint deviation  [deg]", fontsize=8.6, color=figstyle.CATEGORICAL[3])
    ax.set_title("R1 - degrees and millimetres, per joint\n"
                 f"largest in mm: {max(r1['per_joint_max_contrib_mm'], key=r1['per_joint_max_contrib_mm'].get).replace('left_arm_', 'L')}"
                 f"; the joint that led most often: "
                 f"{list(r1['largest_contrib_joint_counts'])[0].replace('left_arm_', 'L')}",
                 fontsize=9.0, color=figstyle.INK)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.2, loc="upper right", framealpha=0.9)
    side["panels"]["r1_per_joint"] = {"deg": r1["per_joint_max_over_approach_deg"],
                                      "mm": r1["per_joint_max_contrib_mm"]}

    # (6) R4
    ax = axes[1, 2]
    ax.set_axis_off()
    c = r4["at_the_closing_step"]
    lines = [
        "R4 - the moment the gripper closes in the successful shadow",
        "",
        f"  chunk t_step {c['t_step_of_chunk']} (seq {c['seq']}), step {c['step_in_chunk']}"
        f" of the execution window",
        f"  global step index                  {c['t_step_of_the_step']}",
        f"  gripper command                    {c['gripper_value']:.6f}"
        f"   (previous step {c['gripper_value_previous_step']:.6f})",
        "",
        f"  ee_finger_l1 origin to apple surface  {c['ee_finger_l1_origin_to_apple_surface_mm']:8.3f} mm",
        f"  offset from the apple centre          x {c['offset_tip_minus_apple_centre_mm'][0]:+8.2f}"
        f"  y {c['offset_tip_minus_apple_centre_mm'][1]:+8.2f}"
        f"  z {c['offset_tip_minus_apple_centre_mm'][2]:+8.2f} mm",
        "",
        "  from the pose the server stored with that chunk (122-sphere model):",
        f"    fingertip sphere surface to apple   {r4['tip_to_apple_surface_mm']:8.3f} mm",
        f"    fingertip sphere centre to apple    {r4['tip_centre_to_apple_surface_mm']:8.3f} mm",
        "",
        f"  smallest over all {a['n_chunks']} chunks      "
        f"{r4['min_over_all_chunks_mm']:8.3f} mm at t={r4['min_over_all_chunks_t_step']}",
        "",
        f"  the earlier reference value (T7a)     {r4['reference_value_T7a_mm']:8.3f} mm",
        "    that number came from a different sphere model and a different",
        "    reference point, so the two are not the same quantity - see not_measured",
    ]
    ax.text(0.0, 0.97, "\n".join(lines), fontsize=8.4, family="monospace",
            color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    side["panels"]["r4"] = r4

    fig.suptitle("T17a - the successful shadow: what pushed TO during the approach",
                 fontsize=12.6, color=figstyle.INK, y=0.985)
    fig.text(0.5, 0.010,
             "every number comes from the server's own ESDF record and the local frames.jsonl · "
             "no rollout · clearances marked POST-margin already have the 10 mm margin "
             "subtracted · the gradient walk uses the fine layer when the query point is inside "
             "it and the coarse 20 mm layer otherwise",
             ha="center", fontsize=8.2, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.024, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
