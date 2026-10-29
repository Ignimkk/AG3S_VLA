"""T7a figure 3 — 표·수식.

왼쪽 위: **관절 축** (사용자가 요구한 형태) — `left_arm_0..6` 마다 활성 제약에 걸린 chunk 수.
왼쪽 아래: **link 축** — `ee_finger_l1/l2` 포함.
오른쪽 위: 기록(ESDF)과 오프라인(ground truth) 의 대조 — 네 실행 전부.
오른쪽 아래: 제약 행의 식과 이 측정이 쓴 숫자.
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

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T7a'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t7a/t7a-table.png'))
SIDE = OUT.with_suffix(".json")
SEL = "no_target__left_spheres"


def table(ax, title, cols, rows, widths, fontsize=8.0, head=8.6):
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.2, color=figstyle.INK, loc="left", pad=10)
    x0 = np.concatenate([[0.0], np.cumsum(widths)])
    y = 0.94
    dy = min(0.10, 0.90 / max(len(rows) + 1, 1))
    for j, c in enumerate(cols):
        ax.text(x0[j], y, c, fontsize=head, color=figstyle.INK, va="top")
    ax.plot([0, 1], [y - dy * 0.42] * 2, color=figstyle.GRID_INK, lw=1.0)
    for i, row in enumerate(rows):
        yy = y - dy * (i + 1)
        for j, cell in enumerate(row):
            ax.text(x0[j], yy, str(cell), fontsize=fontsize,
                    color=figstyle.INK if j == 0 else figstyle.INK_2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def short(name: str) -> str:
    return (str(name).replace("link_left_arm_", "L").replace("link_right_arm_", "R")
            .replace("ee_finger_", "f"))


def main() -> None:
    s1 = json.loads((V / "s1_active_rows.json").read_text())
    s2 = json.loads((V / "s2_joint_axis.json").read_text())
    s3 = json.loads((V / "s3_divergence_context.json").read_text())
    cl = s2["runs"]["closed_loop_approach"][f"refined__{SEL}"]
    sh = s2["runs"]["shadow_approach"][f"reference__{SEL}"]

    figstyle.use_korean()
    fig, ax = plt.subplots(2, 2, figsize=(16.4, 10.4))
    fig.patch.set_facecolor(figstyle.SURFACE)
    for a in ax.ravel():
        a.set_facecolor(figstyle.SURFACE)
    side: dict = {"selection": SEL, "exec_window_steps": 8, "required_margin_mm": 50.0}

    # ---- 1. joint 축 ---------------------------------------------------------------
    rows = []
    for j in [f"left_arm_{i}" for i in range(7)]:
        c, s = cl["joint_axis"][j], sh["joint_axis"][j]
        rows.append([j,
                     f"{c['chunks_rank1']}/75", f"{c['chunks_min_below_0mm']}",
                     f"{c['chunks_min_below_5mm']}", f"{c['chunks_min_below_10mm']}",
                     f"{c['tightest24_slots']}",
                     f"{s['chunks_rank1']}/75", f"{s['chunks_min_below_0mm']}",
                     f"{s['tightest24_slots']}"])
    table(ax[0, 0],
          "1. joint axis — which joint the active collision rows hang on  (chunks out of 75)",
          ["joint", "rank1", "<=0mm", "<=5mm", "<=10mm", "t24 slots",
           "rank1", "<=0mm", "t24 slots"],
          rows, [0.155, 0.085, 0.085, 0.085, 0.10, 0.125, 0.085, 0.085, 0.10])
    ax[0, 0].text(0.155, 1.005, "closed-loop (failed, apple 0.0 mm)", fontsize=8.0,
                  color=figstyle.CATEGORICAL[1], transform=ax[0, 0].transAxes)
    ax[0, 0].text(0.635, 1.005, "shadow (succeeded, apple 245.2 mm)", fontsize=8.0,
                  color=figstyle.CATEGORICAL[0], transform=ax[0, 0].transAxes)
    side["joint_axis_closed_loop"] = cl["joint_axis"]
    side["joint_axis_shadow"] = sh["joint_axis"]

    # ---- 2. link 축 ----------------------------------------------------------------
    rows = []
    for ln in ["link_left_arm_5", "link_left_arm_6", "ee_finger_l1", "ee_finger_l2",
               "link_left_arm_4", "link_left_arm_3"]:
        c, s = cl["link_axis"][ln], sh["link_axis"][ln]
        rows.append([ln, f"{c['chunks_rank1']}/75", f"{c['chunks_min_below_0mm']}",
                     f"{c['chunks_min_below_10mm']}", f"{c['tightest24_slots']}",
                     f"{s['chunks_rank1']}/75", f"{s['chunks_min_below_0mm']}",
                     f"{s['tightest24_slots']}"])
    rows.append(["", "", "", "", "", "", "", ""])
    rows.append(["rank1 obstacle CL",
                 ", ".join(f"{k} {v}" for k, v in cl["rank1_obstacle_counts"].items()),
                 "", "", "", "", "", ""])
    rows.append(["rank1 obstacle SH",
                 ", ".join(f"{k} {v}" for k, v in sh["rank1_obstacle_counts"].items()),
                 "", "", "", "", "", ""])
    table(ax[1, 0],
          "2. link axis — same counts, per constraint link  (left arm + fingertips)",
          ["link", "rank1", "<=0mm", "<=10mm", "t24 slots", "rank1", "<=0mm", "t24 slots"],
          rows, [0.215, 0.095, 0.085, 0.10, 0.125, 0.095, 0.085, 0.10])
    side["link_axis_closed_loop"] = cl["link_axis"]
    side["link_axis_shadow"] = sh["link_axis"]
    side["rank1_obstacle_counts"] = {"closed_loop": cl["rank1_obstacle_counts"],
                                     "shadow": sh["rank1_obstacle_counts"]}

    # ---- 3. 기록 대 오프라인 --------------------------------------------------------
    rows = []
    rec_tbl = {}
    for run, rd in s1["runs"].items():
        recs = [c["recorded"] for c in rd["chunks"]]
        clr = [r["max_violation_pair_clearance_mm"] for r in recs]
        links = {}
        for r in recs:
            links[r["max_violation_pair_link"]] = links.get(r["max_violation_pair_link"], 0) + 1
        nviol = sum(1 for r in recs if (r["max_violation_m"] or 0.0) > 0.0)
        off = s2["runs"][run][f"{rd['executed_variant']}__{SEL}"]
        rows.append([run.replace("_", " "),
                     f"{min(clr):+.2f} / {np.median(clr):+.2f} / {max(clr):+.2f}",
                     f"{nviol}/75",
                     ", ".join(f"{short(k)} {v}"
                               for k, v in sorted(links.items(), key=lambda kv: -kv[1])),
                     f"{min(off['chunk_min_slack_mm']):+.2f}",
                     f"{sum(1 for v in off['chunk_min_slack_mm'] if v <= 0)}/75"])
        rec_tbl[run] = {"recorded_clearance_min_median_max_mm":
                        [min(clr), float(np.median(clr)), max(clr)],
                        "recorded_chunks_with_violation": nviol,
                        "recorded_max_violation_pair_link_counts": links,
                        "offline_min_slack_mm": min(off["chunk_min_slack_mm"]),
                        "offline_chunks_below_0mm":
                            sum(1 for v in off["chunk_min_slack_mm"] if v <= 0)}
    table(ax[0, 1],
          "3. what the run recorded (ESDF) vs this offline rebuild (ground truth, left arm)",
          ["run", "recorded clearance\nmin / med / max mm", "violating\nchunks",
           "recorded tightest link (count)", "offline\nmin mm", "offline\n<=0mm"],
          rows, [0.185, 0.215, 0.095, 0.28, 0.115, 0.11], fontsize=7.4, head=8.0)
    side["recorded_vs_offline"] = rec_tbl

    # ---- 4. 식과 숫자 --------------------------------------------------------------
    a = ax[1, 1]
    a.set_axis_off()
    a.set_title("4. the constraint row, and the numbers this measurement used",
                fontsize=10.2, color=figstyle.INK, loc="left", pad=10)
    cmp_ = s2["comparison"]
    txt = (
        "row (link sphere i, scene surface g, plan step s)\n"
        "    clearance      = surface_distance(centre_i(q_s), g) - r_i\n"
        "    required_margin= collision.esdf_margin = 50.00 mm        (config.py:383)\n"
        "    margin_slack   = clearance - required_margin\n"
        "        slack <  0  -> violating      slack ~ 0 -> active      slack >> 0 -> idle\n\n"
        f"constraint spheres       120 (arms), identical to the T6d set: "
        f"{s1['meta']['sphere_model_identical_to_T6d']}\n"
        f"scene surfaces           {s1['meta']['n_scene_geoms']} collision geoms "
        f"({', '.join(s1['meta']['scene_bodies'])})\n"
        "apple                    excluded from the rows shown (authorized target, carved out "
        "of the ESDF)\n"
        f"QP row budget            reduction.rows_per_step = "
        f"{s1['meta']['reduction_rows_per_step']} per step; every one of the 75 chunks in all "
        "four runs\n"
        "                         carries the 'rows_per_step bound' note\n"
        f"QP slack columns         {s1['meta']['qp_slack_rows_in_record']} in the wire record — "
        "slack != 0 rows cannot be counted from it\n\n"
        "closed-loop (failed)     tightest row is link_left_arm_5 vs crate in 75/75 chunks; "
        f"min slack {cmp_['closed_loop_min_slack_mm_over_all_chunks']:+.2f} mm, "
        f"median {cmp_['closed_loop_median_min_slack_mm']:+.2f} mm,\n"
        f"                         chunks with slack <= 0: "
        f"{cmp_['closed_loop_chunks_min_slack_below_0mm']}/75 · "
        "recorded max_violation_m = 0.0 in 75/75\n"
        "shadow (succeeded)       same rank-1 link in 75/75; min slack "
        f"{cmp_['shadow_min_slack_mm_over_all_chunks']:+.2f} mm, median "
        f"{cmp_['shadow_median_min_slack_mm']:+.2f} mm,\n"
        f"                         chunks with slack <= 0: "
        f"{cmp_['shadow_chunks_min_slack_below_0mm']}/75\n"
        f"same chunk, opposite sign {cmp_['n_chunks_shadow_negative_closed_loop_positive']}/75 "
        "chunks have shadow <= 0 < closed-loop\n"
        f"rank-1 obstacle differs   {cmp_['n_chunks_rank1_obstacle_differs']}/75 chunks "
        f"(first at chunk {cmp_['first_chunk_rank1_obstacle_differs']}); rank-1 link differs "
        f"{cmp_['n_chunks_rank1_link_differs']}/75\n"
        "\nregression baseline       HEAD: 13/15 start-violating, frame0 +0.185978 mm, "
        "planned_horizon 8\n"
        "                          5c5291e^ worktree: 14/15, frame0 +0.15718632962849477 mm, "
        "planned_horizon 32\n"
        f"fingertip to apple        shadow reaches {s3['shadow_tip_min_mm']:+.1f} mm "
        f"(first at chunk {s3['first_chunk_shadow_tip_below_0mm']}); closed-loop never closer "
        f"than {s3['closed_loop_tip_min_mm']:+.1f} mm"
    )
    a.text(0.0, 0.97, txt, fontsize=8.0, color=figstyle.INK, va="top", family="monospace",
           linespacing=1.55)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    side["comparison"] = cmp_
    side["divergence_context"] = {k: v for k, v in s3.items() if not isinstance(v, list)}

    fig.suptitle("T7a — which joint the collision constraint hangs on, counted over 75 chunks",
                 fontsize=12.5, color=figstyle.INK, y=0.985)
    fig.tight_layout(rect=(0, 0.005, 1, 0.965))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, facecolor=figstyle.SURFACE)
    SIDE.write_text(json.dumps(side, indent=1, ensure_ascii=False))
    print("wrote", OUT, SIDE)


if __name__ == "__main__":
    main()
