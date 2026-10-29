"""T5d figure 2 (그래프) — frame 축 × 1 등 물체 띠 그림 + 점수 + apple point + grace window.

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5d/make_fig2_band.py
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
PNG = OUTDIR / "t5d-topcluster-band-and-score.png"
SIDECAR = OUTDIR / "t5d-topcluster-band-and-score.json"

NAME_SLOT = {"apple": 0, "crate": 1, "table": 2, "pear": 3, "orange": 4,
             "banana": 5, "robot": 6, "unknown": 7}
PHASE_SLOT = {"approach": 0, "grasp": 3, "transport": 2, "place": 1, "retreat": 7}


def main() -> None:
    korean = figstyle.use_korean()
    agg = json.loads(AGG.read_text())
    rows = agg["frame_table_seq_1_51"]
    stages = {f["seq"]: f for f in json.loads(STAGES.read_text())["frames"]}
    thr = float(agg["target_score_threshold"])
    seqs = np.array([r["seq"] for r in rows])

    fig = plt.figure(figsize=(14.2, 11.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.35, 1.25, 0.95, 1.1], hspace=0.78, wspace=0.2)
    ax_band = fig.add_subplot(gs[0, :])
    ax_score = fig.add_subplot(gs[1, :], sharex=ax_band)
    ax_pts = fig.add_subplot(gs[2, :], sharex=ax_band)
    ax_cov = fig.add_subplot(gs[3, 0])
    ax_run = fig.add_subplot(gs[3, 1])
    figstyle.style_axes(fig, np.array([ax_band, ax_score, ax_pts, ax_cov, ax_run]))

    # ---------------- band ----------------------------------------------------------
    strips = [
        ("top cluster", [r["top_name"] for r in rows], NAME_SLOT),
        ("attention peak", [r["attention_peak_name"] for r in rows], NAME_SLOT),
        ("task phase", [r["task_phase"] for r in rows], PHASE_SLOT),
    ]
    for si, (label, vals, slot) in enumerate(strips):
        y = len(strips) - 1 - si
        for k, v in enumerate(vals):
            ax_band.add_patch(mpatches.Rectangle(
                (seqs[k] - 0.5, y - 0.42), 1.0, 0.84, linewidth=0,
                facecolor=figstyle.CATEGORICAL[slot.get(str(v), 7)]))
        ax_band.text(0.2, y, label, ha="right", va="center", fontsize=9, color=figstyle.INK)
        # 띠 안에 이름을 직접 적는다 — 국면 띠는 물체 띠와 색 뜻이 다르다
        run_s = 0
        for k in range(len(vals) + 1):
            if k == len(vals) or vals[k] != vals[run_s]:
                if k - run_s >= 3:
                    ax_band.text((seqs[run_s] + seqs[k - 1]) / 2.0, y, str(vals[run_s]),
                                 ha="center", va="center", fontsize=7.5, color="#ffffff")
                run_s = k
    # has_target 표시
    for k, r in enumerate(rows):
        if r["offline_has_target"]:
            ax_band.plot([r["seq"]], [3.28], marker="v", ms=4.2,
                         color=figstyle.CATEGORICAL[0], mew=0)
        if r["live_has_target"]:
            ax_band.plot([r["seq"]], [3.62], marker="v", ms=4.2,
                         color=figstyle.CATEGORICAL[6], mew=0)
    ax_band.text(0.2, 3.28, "offline has_target", ha="right", va="center", fontsize=8,
                 color=figstyle.CATEGORICAL[0])
    ax_band.text(0.2, 3.62, "live has_target", ha="right", va="center", fontsize=8,
                 color=figstyle.CATEGORICAL[6])
    gb = agg["grace_window_live"]["grasp_boundary_seq"]
    pb = agg["grace_window_live"]["place_boundary_seq"]
    for ax in (ax_band, ax_score, ax_pts):
        for x, lab in ((gb, "gripper closes"), (pb, "gripper reopens")):
            ax.axvline(x, color=figstyle.INK_2, lw=0.9, ls=(0, (4, 3)), zorder=0)
    ax_band.text(gb, 3.95, f"gripper closes (seq {gb})", fontsize=8, color=figstyle.INK_2,
                 ha="center")
    ax_band.text(pb, 3.95, f"gripper reopens (seq {pb})", fontsize=8, color=figstyle.INK_2,
                 ha="center")
    ax_band.set_ylim(-0.7, 4.35)
    ax_band.set_yticks([])
    ax_band.set_xlim(-4.5, 52.5)
    for sp in ("left",):
        ax_band.spines[sp].set_visible(False)
    handles = [mpatches.Patch(color=figstyle.CATEGORICAL[NAME_SLOT[n]], label=n)
               for n in ("apple", "crate", "table", "pear", "orange", "unknown")]
    ax_band.legend(handles=handles, ncol=6, fontsize=8, frameon=False,
                   loc="lower center", bbox_to_anchor=(0.5, -0.30))
    ax_band.set_title(
        ("1 등 cluster 의 참값 이름 · attention peak 의 참값 이름 · 과제 국면 (seq 1-51)"
         if korean else
         "ground-truth name of the top cluster / attention peak / task phase (seq 1-51)"),
        fontsize=10.5, color=figstyle.INK, pad=16)

    # ---------------- score ---------------------------------------------------------
    top = np.array([r["top_score"] for r in rows], float)
    ax_score.plot(seqs, top, lw=1.5, color=figstyle.INK, label="top cluster score")
    for nm in ("apple", "crate", "pear"):
        m = np.array([r["top_name"] == nm for r in rows])
        ax_score.scatter(seqs[m], top[m], s=26, zorder=4, linewidths=0,
                         color=figstyle.CATEGORICAL[NAME_SLOT[nm]], label=f"top = {nm}")
    ap = np.array([np.nan if r["apple_score"] is None else r["apple_score"] for r in rows], float)
    ax_score.scatter(seqs, ap, s=40, marker="x", linewidths=1.5,
                     color=figstyle.CATEGORICAL[0], label="apple cluster score", zorder=5)
    ax_score.axhline(thr, color=figstyle.CATEGORICAL[7], lw=1.1, ls=(0, (4, 3)))
    ax_score.text(52.2, thr, f" threshold {thr}", fontsize=8, va="center",
                  color=figstyle.CATEGORICAL[7])
    ax_score.set_ylabel("target_score", fontsize=9)
    ax_score.set_ylim(0, 0.95)
    ax_score.legend(fontsize=8, frameon=False, ncol=5, loc="upper center",
                bbox_to_anchor=(0.5, -0.12))
    ax_score.set_title(("target_score — apple cluster 가 없는 frame 에는 X 가 없다"
                        if korean else
                        "target_score - no X means no apple cluster in that frame"),
                       fontsize=10, color=figstyle.INK)

    # ---------------- apple points --------------------------------------------------
    recon = np.array([stages[r["seq"]]["n_after_recon_total"] for r in rows], float)
    filt = np.array([stages[r["seq"]]["n_after_filter_total"] for r in rows], float)
    ax_pts.bar(seqs, recon, width=0.82, color=figstyle.SEQ_BLUE[2], linewidth=0,
               label="apple points after reconstruct")
    ax_pts.bar(seqs, filt, width=0.82, color=figstyle.SEQ_BLUE[10], linewidth=0,
               label="apple points after robot self-filter")
    ax_pts.set_ylabel("apple points", fontsize=9)
    ax_pts.set_xlabel("seq  (t_step = (seq - 1) x 8)", fontsize=9)
    ax_pts.legend(fontsize=8, frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.30))
    ax_pts.set_title(("apple point 이 cloud 에 몇 개 남나 — 세 카메라 합"
                      if korean else
                      "apple points surviving into the cloud - three cameras summed"),
                     fontsize=10, color=figstyle.INK)
    ax_pts.set_xticks(list(range(0, 52, 5)))

    # ---------------- grace window --------------------------------------------------
    for gi, (key, lab, slot) in enumerate((
            ("grace_window_live", "live shadow path", 6),
            ("grace_window_offline", "offline replay path", 0))):
        cov = agg[key]["coverage_by_N"]
        ax_cov.plot([c["N"] for c in cov], [c["frames_covered_pct"] for c in cov],
                    lw=1.8, marker="o", ms=3.2, color=figstyle.CATEGORICAL[slot],
                    label=f"{lab} (frames)")
        ax_cov.plot([c["N"] for c in cov], [c["runs_covered_pct"] for c in cov],
                    lw=1.2, ls=(0, (3, 2)), color=figstyle.CATEGORICAL[slot],
                    label=f"{lab} (runs)")
    ax_cov.set_xlabel("N  (grace window, frames)", fontsize=9)
    ax_cov.set_ylabel("covered [%]", fontsize=9)
    ax_cov.set_xticks(list(range(0, 21, 2)))
    ax_cov.set_ylim(-4, 104)
    ax_cov.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax_cov.set_title(("N frame 을 버티면 실패 구간의 몇 %가 덮이나"
                      if korean else
                      "fraction of failure runs covered by an N-frame grace window"),
                     fontsize=10, color=figstyle.INK)

    # run bars
    ypos, labels = [], []
    y = 0
    for key, lab, slot in (("grace_window_live", "live", 6),
                           ("grace_window_offline", "offline", 0)):
        for r in agg[key]["runs"]:
            ax_run.barh(y, r["n"], left=r["seq_from"], height=0.62,
                        color=figstyle.CATEGORICAL[slot], linewidth=0)
            ax_run.text(r["seq_from"] + r["n"] + 0.6, y,
                        f"n={r['n']} · {r['boundary_class']}", va="center", fontsize=7.5,
                        color=figstyle.INK_2)
            ypos.append(y)
            labels.append(f"{lab} {r['seq_from']}-{r['seq_to']}")
            y += 1
        y += 0.6
    ax_run.set_yticks(ypos)
    ax_run.set_yticklabels(labels, fontsize=7.5)
    ax_run.invert_yaxis()
    ax_run.set_xlim(0, 70)
    ax_run.axvline(gb, color=figstyle.INK_2, lw=0.9, ls=(0, (4, 3)))
    ax_run.axvline(pb, color=figstyle.INK_2, lw=0.9, ls=(0, (4, 3)))
    ax_run.set_xlabel("seq", fontsize=9)
    ax_run.set_title(("실패 구간의 자리와 길이 (경계 반경 2 seq)"
                      if korean else
                      "where each failure run sits and how long it is (boundary radius 2 seq)"),
                     fontsize=10, color=figstyle.INK)

    fig.suptitle(
        ("T5d — ep1807, seq 1-51 (t_step <= 400). offline replay 가 점수를, live shadow 가 라벨을 낸다"
         if korean else
         "T5d - ep1807, seq 1-51 (t_step <= 400). scores from offline replay, labels from live shadow"),
        fontsize=12.5, color=figstyle.INK)
    fig.tight_layout(rect=(0.035, 0, 1, 0.965))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=170, facecolor=figstyle.SURFACE)

    SIDECAR.write_text(json.dumps({
        "source": "outputs/verify/T5d/aggregate_T5d.json",
        "threshold": thr,
        "grasp_boundary_seq": gb, "place_boundary_seq": pb,
        "band": [{"seq": r["seq"], "t_step": r["t_step"], "top_name": r["top_name"],
                  "top_score": r["top_score"], "apple_score": r["apple_score"],
                  "apple_rank": r["apple_rank"],
                  "attention_peak_name": r["attention_peak_name"],
                  "task_phase": r["task_phase"],
                  "offline_has_target": r["offline_has_target"],
                  "live_has_target": r["live_has_target"],
                  "apple_after_recon": stages[r["seq"]]["n_after_recon_total"],
                  "apple_after_self_filter": stages[r["seq"]]["n_after_filter_total"]}
                 for r in rows],
        "segments_by_top_name": agg["segments_by_top_name"],
        "grace_window_live": {k: agg["grace_window_live"][k]
                              for k in ("n_failing_frames", "n_runs", "longest_run_n",
                                        "run_length_histogram", "runs", "coverage_by_N",
                                        "runs_by_boundary_class")},
        "grace_window_offline": {k: agg["grace_window_offline"][k]
                                 for k in ("n_failing_frames", "n_runs", "longest_run_n",
                                           "run_length_histogram", "runs", "coverage_by_N",
                                           "runs_by_boundary_class")},
    }, indent=1, ensure_ascii=False))
    print("wrote", PNG)


if __name__ == "__main__":
    main()
