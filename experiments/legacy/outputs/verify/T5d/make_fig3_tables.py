"""T5d figure 3 (표·도식) — 구간 표 · 실패 grounding 해부 표 · grace window 표 · 경로 도식.

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5d/make_fig3_tables.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

AGG = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5d/aggregate_T5d.json'))
STAGES = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5d/apple_stages_ep1807.json'))
OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t5d'))
PNG = OUTDIR / "t5d-segments-failures-and-grace-table.png"
SIDECAR = OUTDIR / "t5d-segments-failures-and-grace-table.json"

NAME_SLOT = {"apple": 0, "crate": 1, "table": 2, "pear": 3, "orange": 4,
             "banana": 5, "robot": 6, "unknown": 7}


def draw_table(ax, title, header, rows, colw, row_colors=None, fs=8.2):
    ax.set_axis_off()
    ax.set_title(title, fontsize=10.5, color=figstyle.INK, loc="left", pad=8)
    x = np.concatenate([[0.0], np.cumsum(colw)])
    x = x / x[-1]
    n = len(rows) + 1
    h = 1.0 / max(n, 1)
    for j, htxt in enumerate(header):
        ax.text(x[j] + 0.006, 1.0 - h * 0.5, htxt, fontsize=fs, color=figstyle.INK,
                va="center", ha="left")
    ax.plot([0, 1], [1.0 - h, 1.0 - h], color=figstyle.INK, lw=0.9)
    for i, row in enumerate(rows):
        y = 1.0 - h * (i + 1.5)
        if row_colors and row_colors[i] is not None:
            ax.add_patch(mpatches.Rectangle((0, y - h * 0.42), 0.0085, h * 0.84,
                                            color=row_colors[i], linewidth=0))
        for j, cell in enumerate(row):
            ax.text(x[j] + 0.012, y, str(cell), fontsize=fs, color=figstyle.INK_2,
                    va="center", ha="left")
        ax.plot([0, 1], [y - h * 0.5, y - h * 0.5], color=figstyle.GRID_INK, lw=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main() -> None:
    korean = figstyle.use_korean()
    agg = json.loads(AGG.read_text())
    stages = json.loads(STAGES.read_text())
    stage_by_seq = {f["seq"]: f for f in stages["frames"]}

    fig = plt.figure(figsize=(15.0, 11.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.15, 0.95], hspace=0.30, wspace=0.14)
    ax_seg = fig.add_subplot(gs[0, :])
    ax_fail = fig.add_subplot(gs[1, 0])
    ax_grace = fig.add_subplot(gs[1, 1])
    ax_pipe = fig.add_subplot(gs[2, :])
    figstyle.style_axes(fig, np.array([ax_seg, ax_fail, ax_grace, ax_pipe]))

    # ---- 1. 구간 표 ---------------------------------------------------------------
    seg_rows, seg_colors = [], []
    for s in agg["segments_by_top_name"]:
        sc = s["top_score"]
        seg_rows.append([
            s["top_name"],
            f"{s['seq_from']}-{s['seq_to']}",
            f"{s['t_step_from']}-{s['t_step_to']}",
            s["n"],
            f"{sc['p50']:.4f}",
            f"{sc['min']:.4f} / {sc['max']:.4f}",
            " > ".join(s["task_phases"]),
        ])
        seg_colors.append(figstyle.CATEGORICAL[NAME_SLOT.get(s["top_name"], 7)])
    draw_table(
        ax_seg,
        ("구간 — 1 등 cluster 가 바뀌는 지점으로 끊었다 (seq 1-51)" if korean else
         "segments - cut where the top cluster changes (seq 1-51)"),
        ["top cluster", "seq", "t_step", "n", "score p50", "score min / max", "task phase"],
        seg_rows, [1.05, 0.72, 0.95, 0.42, 0.82, 1.25, 1.85])

    # ---- 2. 실패 해부 표 ----------------------------------------------------------
    dl, do = agg["dissect_live_failing"], agg["dissect_offline_failing"]

    def tn(d, k):
        return d["top_name_counts"].get(k, 0)

    fail_rows = [
        ["n frames", dl["n"], do["n"]],
        ["top cluster = apple", tn(dl, "apple"), tn(do, "apple")],
        ["top cluster = crate", tn(dl, "crate"), tn(do, "crate")],
        ["top cluster = pear", tn(dl, "pear"), tn(do, "pear")],
        ["apple cluster present", dl["n_apple_cluster_present"], do["n_apple_cluster_present"]],
        ["apple cluster absent", dl["n_apple_cluster_absent"], do["n_apple_cluster_absent"]],
        ["apple cloud points = 0", dl["n_apple_zero_cloud_points"], do["n_apple_zero_cloud_points"]],
        ["apple cloud points > 0", dl["n_apple_cloud_points_gt_zero"],
         do["n_apple_cloud_points_gt_zero"]],
        ["threshold gap p50", f"{dl['threshold_gap']['p50']:.4f}",
         f"{do['threshold_gap']['p50']:.4f}"],
        ["threshold gap max", f"{dl['threshold_gap']['max']:.4f}",
         f"{do['threshold_gap']['max']:.4f}"],
        ["gap <= 0.05", dl["n_gap_within_0p05"], do["n_gap_within_0p05"]],
        ["gap > 0.10", dl["n_gap_over_0p10"], do["n_gap_over_0p10"]],
        ["offline status = ok on the same seq", dl["n_offline_status_ok"], "-"],
        ["latch manipulated = obj0",
         dl["latch_manipulated_col7_counts"].get("obj0", 0),
         do["latch_manipulated_col7_counts"].get("obj0", 0)],
        ["latch manipulated = None",
         dl["latch_manipulated_col7_counts"].get("None", 0),
         do["latch_manipulated_col7_counts"].get("None", 0)],
    ]
    draw_table(
        ax_fail,
        ("실패한 grounding 은 어떻게 생겼나 (seq 1-51)" if korean else
         "what a failed grounding looks like (seq 1-51)"),
        ["", "live shadow (35)", "offline replay (30)"],
        fail_rows, [2.1, 1.0, 1.05])

    # ---- 3. grace window 표 -------------------------------------------------------
    gl, go = agg["grace_window_live"], agg["grace_window_offline"]
    cl = {c["N"]: c for c in gl["coverage_by_N"]}
    co = {c["N"]: c for c in go["coverage_by_N"]}
    grace_rows = [["runs / longest run",
                   f"{gl['n_runs']} / {gl['longest_run_n']}",
                   f"{go['n_runs']} / {go['longest_run_n']}"],
                  ["run lengths",
                   ", ".join(f"{k}x{v}" for k, v in gl["run_length_histogram"].items()),
                   ", ".join(f"{k}x{v}" for k, v in go["run_length_histogram"].items())],
                  ["runs on grasp boundary",
                   f"{gl['runs_by_boundary_class']['grasp']['n_runs']} "
                   f"({gl['runs_by_boundary_class']['grasp']['n_frames']} frames)",
                   f"{go['runs_by_boundary_class']['grasp']['n_runs']} "
                   f"({go['runs_by_boundary_class']['grasp']['n_frames']} frames)"],
                  ["runs on place boundary",
                   f"{gl['runs_by_boundary_class']['place']['n_runs']} "
                   f"({gl['runs_by_boundary_class']['place']['n_frames']} frames)",
                   f"{go['runs_by_boundary_class']['place']['n_runs']} "
                   f"({go['runs_by_boundary_class']['place']['n_frames']} frames)"],
                  ["runs elsewhere",
                   f"{gl['runs_by_boundary_class']['other']['n_runs']} "
                   f"({gl['runs_by_boundary_class']['other']['n_frames']} frames)",
                   f"{go['runs_by_boundary_class']['other']['n_runs']} "
                   f"({go['runs_by_boundary_class']['other']['n_frames']} frames)"]]
    for n in (1, 2, 3, 5, 8, 12, 13, 15, 20):
        grace_rows.append([
            f"N = {n}  covered",
            f"{cl[n]['runs_covered']}/{gl['n_runs']} runs · "
            f"{cl[n]['frames_covered_pct']:.1f} % frames",
            f"{co[n]['runs_covered']}/{go['n_runs']} runs · "
            f"{co[n]['frames_covered_pct']:.1f} % frames",
        ])
    draw_table(
        ax_grace,
        ("grace window — N frame 을 버티면 얼마가 덮이나 (경계 반경 2 seq)" if korean else
         "grace window - what an N-frame hold covers (boundary radius 2 seq)"),
        ["", "live shadow", "offline replay"],
        grace_rows, [1.35, 1.35, 1.35])

    # ---- 4. 경로 도식 --------------------------------------------------------------
    ax_pipe.set_axis_off()
    ax_pipe.set_title(
        ("apple point 이 어디서 사라지나 — 두 frame 의 같은 경로 (세 카메라 합)" if korean else
         "where the apple points go - the same path at two frames (three cameras summed)"),
        fontsize=10.5, color=figstyle.INK, loc="left", pad=8)
    boxes = [("depth pixel\n(MuJoCo seg)", "n_gt_pixel"),
             ("range <= 2.0 m", "n_in_range"),
             ("reconstruct\n(voxel 5 mm)", "n_after_recon"),
             ("robot self-filter\n(inflation 50 mm)", "n_after_filter")]
    show = [18, 25, 40]
    w, gap, x_left = 0.16, 0.03, 0.20
    for bi, (label, _key) in enumerate(boxes):
        x0 = bi * (w + gap) + x_left
        ax_pipe.add_patch(mpatches.FancyBboxPatch(
            (x0, 0.60), w, 0.30, boxstyle="round,pad=0.008",
            facecolor=figstyle.SEQ_BLUE[1], edgecolor=figstyle.SEQ_BLUE[7], lw=1.0))
        ax_pipe.text(x0 + w / 2, 0.75, label, ha="center", va="center", fontsize=9,
                     color=figstyle.INK)
        if bi < len(boxes) - 1:
            ax_pipe.annotate("", xy=(x0 + w + gap - 0.004, 0.75), xytext=(x0 + w + 0.004, 0.75),
                             arrowprops=dict(arrowstyle="->", color=figstyle.INK_2, lw=1.2))
    for si, seq in enumerate(show):
        st = stage_by_seq[seq]
        y = 0.44 - si * 0.155
        row = next(r for r in agg["frame_table_seq_1_51"] if r["seq"] == seq)
        ax_pipe.text(0.008, y - 0.055, f"seq {seq}  (t {st['t_step']}, {row['task_phase']})",
                     fontsize=8.6, color=figstyle.INK, va="center", ha="left")
        for bi, (_label, key) in enumerate(boxes):
            x0 = bi * (w + gap) + x_left
            val = sum(v[key] for v in st["per_camera"].values())
            ax_pipe.text(x0 + w / 2, y - 0.055, f"{val}", fontsize=10.5, ha="center",
                         va="center",
                         color=(figstyle.CATEGORICAL[7] if val == 0 else figstyle.INK))
    ax_pipe.text(0.02, -0.10,
                 ("게이트: pipeline.py:406 은 grounding.target is None 만 본다 — 같은 frame 의 "
                  "GraspLatch 상태(manipulated)를 보지 않는다. latch 는 seq 1-51 의 실패 35 "
                  "frame 전부에서 obj0 를 쥐고 있었다 (safe_policy 의 gripper 열 6 / wire 의 "
                  "열 7 둘 다에서 같다)"
                  if korean else
                  "gate: pipeline.py:406 reads only grounding.target is None; it does not read the "
                  "GraspLatch state of that frame. the latch held obj0 on all 35 failing frames"),
                 fontsize=8.6, color=figstyle.INK_2, va="top", ha="left", wrap=True)
    ax_pipe.set_xlim(0, 1)
    ax_pipe.set_ylim(-0.25, 1.0)

    fig.suptitle(
        ("T5d — ep1807 seq 1-51 (t_step <= 400). threshold 0.25, "
         "점수는 offline replay · 라벨은 live shadow"
         if korean else
         "T5d - ep1807 seq 1-51 (t_step <= 400). threshold 0.25, "
         "scores from offline replay, labels from live shadow"),
        fontsize=12.5, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=170, facecolor=figstyle.SURFACE)

    SIDECAR.write_text(json.dumps({
        "source": "outputs/verify/T5d/aggregate_T5d.json",
        "segments_by_top_name": agg["segments_by_top_name"],
        "failure_table": {"header": ["", "live shadow (35)", "offline replay (30)"],
                          "rows": fail_rows},
        "grace_table": {"header": ["", "live shadow", "offline replay"], "rows": grace_rows},
        "pipeline_counts": {str(s): {k: sum(v[k] for v in stage_by_seq[s]["per_camera"].values())
                                     for _l, k in [("", "n_gt_pixel"), ("", "n_in_range"),
                                                   ("", "n_after_recon"), ("", "n_after_filter")]}
                            for s in show},
        "gate_reference": "AG3S/ag3s/runtime/pipeline.py:406",
    }, indent=1, ensure_ascii=False))
    print("wrote", PNG)


if __name__ == "__main__":
    main()
