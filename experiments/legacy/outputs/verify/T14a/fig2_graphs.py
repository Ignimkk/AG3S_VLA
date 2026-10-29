"""T14a figure 2 — chunk 축 그래프. Q2 · Q1 · Q3 · Q4 를 T12 와 나란히."""
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

T14A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T14a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t14a/t14a-graphs.png'))
SIDE = OUT.with_suffix(".json")
COL = {"T12": figstyle.CATEGORICAL[0], "T14": figstyle.CATEGORICAL[1]}


def main() -> None:
    a = json.loads((T14A / "q1to4.json").read_text())
    q1, q2, q3, q4 = (a["q1_fine_window"], a["q2_grounding_target_loss"],
                      a["q3_what_the_ball_erased"], a["q4_fingertip"])
    ov = a["q1_q2_overlap"]
    AP = a["approach_window_t_step"]
    figstyle.use_korean()
    fig, axes = plt.subplots(2, 3, figsize=(18.6, 9.8))
    figstyle.style_axes(fig, axes)
    side: dict = {"approach": AP, "panels": {}}

    def tt(rows):
        return [r["t_step"] for r in rows if r["t_step"] is not None]

    def band(ax):
        ax.axvspan(AP[0], AP[1], color=figstyle.CATEGORICAL[0], alpha=0.06)
        ax.annotate("approach window", (AP[0] + 4, ax.get_ylim()[1]), fontsize=7.4,
                    color=figstyle.INK_2, va="top")

    # (1) Q2 — centroid 가 어느 물체에 가까운가
    ax = axes[0, 0]
    for run in ("T12", "T14"):
        rows = [r for r in q2[run]["per_chunk"] if r["t_step"] is not None]
        t = [r["t_step"] for r in rows]
        y = [r["centroid_to_apple_mm"] for r in rows]
        ax.plot(t, y, color=COL[run], lw=1.8, label=f"{run} centroid to apple")
        bad = [(r["t_step"], r["centroid_to_apple_mm"]) for r in rows
               if r["centroid_nearest_is_apple"] is False]
        if bad:
            ax.scatter([x for x, _ in bad], [v for _, v in bad], s=26, zorder=8,
                       facecolor="none", edgecolor=COL[run], lw=1.3,
                       label=f"{run} nearest object is NOT the apple")
    ax.axhline(33.3, color=figstyle.CATEGORICAL[7], lw=1.0, ls="--")
    ax.annotate("apple radius 33.3 mm", (0, 33.3), textcoords="offset points", xytext=(4, 4),
                fontsize=7.4, color=figstyle.CATEGORICAL[7])
    band(ax)
    ax.set_xlabel("local t_step", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("distance from target centroid to the apple  [mm]", fontsize=8.6,
                  color=figstyle.INK_2)
    ax.set_title("Q2 - what grounding actually picked\n"
                 f"T14 nearest is not the apple on {q2['T14']['n_chunks_centroid_nearest_is_not_apple']}"
                 f"/{q2['T14']['n_chunks']} chunks, T12 on "
                 f"{q2['T12']['n_chunks_centroid_nearest_is_not_apple']}/{q2['T12']['n_chunks']}",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.0, loc="upper left", framealpha=0.9)
    side["panels"]["q2_centroid_to_apple"] = {
        run: [{"t_step": r["t_step"], "mm": r["centroid_to_apple_mm"],
               "nearest": r["centroid_nearest_object"]}
              for r in q2[run]["per_chunk"] if r["t_step"] is not None] for run in ("T12", "T14")}

    # (2) Q1 — 사과가 fine 창 안인가
    ax = axes[0, 1]
    for k, run in enumerate(("T12", "T14")):
        rows = [r for r in q1[run]["per_chunk"] if r["t_step"] is not None]
        t = [r["t_step"] for r in rows]
        y = [(r["apple_distance_outside_fine_mm"] or 0.0) for r in rows]
        ax.plot(t, y, color=COL[run], lw=1.8, marker="o", ms=3,
                label=f"{run} apple distance outside the fine window")
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    band(ax)
    ax.set_xlabel("local t_step", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("how far the apple is outside the fine window  [mm]", fontsize=8.6,
                  color=figstyle.INK_2)
    ax.set_title(f"Q1 - the fine layer window vs the apple\n"
                 f"T14 outside on {q1['T14']['n_apple_outside_fine']}/{q1['T14']['n_chunks']} "
                 f"({q1['T14']['approach_window']['n_apple_outside_fine']} in the approach), "
                 f"T12 on {q1['T12']['n_apple_outside_fine']}/{q1['T12']['n_chunks']}",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.9)
    side["panels"]["q1_apple_outside"] = {
        run: [{"t_step": r["t_step"], "mm": r["apple_distance_outside_fine_mm"],
               "inside": r["apple_in_fine_window"]}
              for r in q1[run]["per_chunk"] if r["t_step"] is not None] for run in ("T12", "T14")}

    # (3) Q1 — 창 중심의 y
    ax = axes[0, 2]
    for run in ("T12", "T14"):
        rows = [r for r in q1[run]["per_chunk"] if r["t_step"] is not None]
        t = [r["t_step"] for r in rows]
        cy = [(None if r["target_centroid_mm"] is None else r["target_centroid_mm"][1])
              for r in rows]
        ay = [(None if r["apple_mm"] is None else r["apple_mm"][1]) for r in rows]
        lo = [r["fine_origin_mm"][1] for r in rows]
        hi = [r["fine_max_mm"][1] for r in rows]
        ax.fill_between(t, lo, hi, color=COL[run], alpha=0.12,
                        label=f"{run} fine window y range")
        ax.plot(t, cy, color=COL[run], lw=1.6, ls="--", label=f"{run} target centroid y")
        if run == "T14":
            ax.plot(t, ay, color=figstyle.CATEGORICAL[7], lw=2.0, label="apple y (ground truth)")
    band(ax)
    ax.set_xlabel("local t_step", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("base y  [mm]", fontsize=8.6, color=figstyle.INK_2)
    ax.set_title("Q1 - the window follows the centroid, and the centroid leaves the apple\n"
                 "shaded = the fine window's y extent, dashed = the centroid it is centred on",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.0, loc="lower left", framealpha=0.9)

    # (4) Q3 — 공 안에서 지워진 voxel
    ax = axes[1, 0]
    w = 0.38
    labels = ["voxels changed\ninside the ball", "voxels changed\nwithin 45 mm of the apple"]
    for k, run in enumerate(("T12", "T14")):
        vals = [q3[run]["ball_changed_total"], q3[run]["apple45_changed_total"]]
        ax.bar(np.arange(2) + (k - 0.5) * w, vals, width=w, color=COL[run], label=run)
        for j, v in enumerate(vals):
            ax.annotate(f"{v}", (j + (k - 0.5) * w, max(v, 0.6)), ha="center",
                        textcoords="offset points", xytext=(0, 3), fontsize=8.4,
                        color=figstyle.INK_2)
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylabel("voxels where esdf differs from esdf_free  (all chunks)", fontsize=8.6,
                  color=figstyle.INK_2)
    ax.set_title("Q3 - what the ball actually erased\n"
                 "T12 erased 0 voxels inside the ball; T14 is the first run where it moved",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.6, framealpha=0.9)
    side["panels"]["q3"] = {run: {k: q3[run][k] for k in
                                  ("ball_changed_total", "ball_changed_max",
                                   "apple45_changed_total", "apple45_changed_max",
                                   "whole_layer_diff_min", "whole_layer_diff_max",
                                   "n_chunks_without_free_layer")} for run in ("T12", "T14")}

    # (5) Q4 — 접근 구간 최빡빡 행
    ax = axes[1, 1]
    for run in ("T12", "T14"):
        rows = q4[run]["approach_window"]["per_chunk"]
        ax.plot([r["t_step"] for r in rows],
                [r["mvp_clearance_mm_post_margin"] for r in rows],
                color=COL[run], lw=2.0, marker="o", ms=4, label=f"{run}")
    ax.axhline(0.0, color=figstyle.INK, lw=1.0)
    ax.annotate("0 = the 10 mm margin is exactly used up", (AP[0] + 2, 0.6), fontsize=7.4,
                color=figstyle.INK_2)
    ax.set_xlabel("local t_step", fontsize=8.6, color=figstyle.INK_2)
    ax.set_ylabel("tightest recorded row, POST-margin  [mm]", fontsize=8.6,
                  color=figstyle.INK_2)
    ax.set_title(f"Q4 - the tightest row through the approach\n"
                 f"T14 min {q4['T14']['approach_window']['clearance_min_mm_post_margin']:.4f} mm "
                 f"at t={q4['T14']['approach_window']['clearance_min_t_step']}, "
                 f"T12 min {q4['T12']['approach_window']['clearance_min_mm_post_margin']:.4f} mm "
                 f"at t={q4['T12']['approach_window']['clearance_min_t_step']}",
                 fontsize=9.0, color=figstyle.INK)
    ax.legend(fontsize=7.6, framealpha=0.9)
    side["panels"]["q4_approach"] = {run: q4[run]["approach_window"]["per_chunk"]
                                     for run in ("T12", "T14")}

    # (6) Q1 ∩ Q2
    ax = axes[1, 2]
    ax.set_axis_off()
    lines = ["Q1 n Q2 - are the two failures the same chunks?", ""]
    for run in ("T14", "T12"):
        e = ov[run]
        lines += [
            f"[{run}]  identical sets: {e['identical_sets']}",
            f"   apple outside the fine window : {len(e['t_steps_apple_outside_fine'])} chunks",
            f"   centroid not nearest the apple: {len(e['t_steps_centroid_not_apple'])} chunks",
            f"   in both                       : {e['n_in_both']}",
            f"   only window / only centroid   : {e['n_only_window']} / {e['n_only_centroid']}",
            f"   has_target False              : {e['t_steps_has_target_false']}",
            "",
        ]
    lines += [
        "t_steps where the apple is outside the fine window:",
        f"  T14 {q1['T14']['t_steps_apple_outside_fine']}",
        "",
        "t_steps where the centroid's nearest object is not the apple:",
        f"  T14 {q2['T14']['t_steps_centroid_not_apple']}",
        "",
        "what grounding reported on those chunks:",
        f"  server grounding_status counts T14 {q2['T14']['server_grounding_status_counts']}",
        f"  server status counts          T14 {q2['T14']['server_status_counts']}",
        f"  local ag3s_status counts      T14 {q2['T14']['local_ag3s_status_counts']}",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), fontsize=8.0, family="monospace",
            color=figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    side["panels"]["q1_q2_overlap"] = ov

    fig.suptitle("T14a - T14 (the fixed ball) against T12, on the same axes",
                 fontsize=12.6, color=figstyle.INK, y=0.985)
    fig.text(0.5, 0.010,
             "every number is read from the server's own ESDF record (summary_json and the "
             "layer arrays) plus the local frames.jsonl · no rollout, no MuJoCo except the "
             "analytic apple-surface distance · clearances marked POST-margin already have the "
             "10 mm margin subtracted",
             ha="center", fontsize=8.2, color=figstyle.INK_2)
    fig.tight_layout(rect=(0, 0.024, 1, 0.958))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
