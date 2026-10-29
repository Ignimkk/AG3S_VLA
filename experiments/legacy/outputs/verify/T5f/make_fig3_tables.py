"""T5f figure 3 (표·도식) — 나란히 놓은 숫자.

  (a) 두 live 실행의 판정 계수 표.
  (b) target 이 된 물체별 크기·퍼짐·순도 표.
  (c) live `violated` 7 건 x 참값 clearance 표.
  (d) 회귀 기준선 표 + 무엇이 바뀐 자리인지 도식.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5f/make_fig3_tables.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5f'))
OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t5f'))
PNG = OUTDIR / "t5f-tables-before-after.png"
SIDECAR = OUTDIR / "t5f-tables-before-after.json"


def draw(ax, title, columns, rows, widths, highlight=(), fs=8.2):
    ax.axis("off")
    ax.set_title(title, fontsize=10.5, color=figstyle.INK, loc="left", pad=8)
    x = np.concatenate([[0.0], np.cumsum(widths)])
    y0, dy = 0.94, 0.078
    for j, c in enumerate(columns):
        ax.text(x[j], y0, c, fontsize=fs, color=figstyle.INK_2, ha="left", va="top",
                fontweight="normal")
    ax.plot([0, x[-1]], [y0 - 0.022, y0 - 0.022], color=figstyle.GRID_INK, lw=0.9)
    for i, r in enumerate(rows):
        yy = y0 - 0.032 - (i + 1) * dy + dy * 0.45
        col = figstyle.CATEGORICAL[7] if i in highlight else figstyle.INK
        for j, cell in enumerate(r):
            ax.text(x[j], yy, str(cell), fontsize=fs, color=col, ha="left", va="center")
    ax.set_xlim(-0.01, x[-1] + 0.01)
    ax.set_ylim(y0 - 0.032 - (len(rows) + 1) * dy, 1.0)


def main() -> None:
    korean = figstyle.use_korean()
    agg = json.loads((V / "aggregate_T5f.json").read_text())
    sbs = agg["live_counts_side_by_side"]
    cam = agg["camera_timing"]
    base = agg["baseline_check"]

    fig, axes = plt.subplots(2, 2, figsize=(14.4, 8.6))
    figstyle.style_axes(fig, axes)
    for ax in axes.ravel():
        ax.set_facecolor(figstyle.SURFACE)

    # --- (a) live 판정 계수 ---------------------------------------------------------------
    rows_a = [
        ["safe / unsafe chunk", f"{sbs['safe'][0]} / {sbs['unsafe'][0]}",
         f"{sbs['safe'][1]} / {sbs['unsafe'][1]}", "75 chunk"],
        ["ag3s no_target", sbs["ag3s_no_target"][0], sbs["ag3s_no_target"][1], ""],
        ["ag3s degraded", sbs["ag3s_degraded"][0], sbs["ag3s_degraded"][1], ""],
        ["trajopt violated", sbs["trajopt_violated"][0], sbs["trajopt_violated"][1], ""],
        ["field tiers = 2 (fine 5 mm)", sbs["two_tier_chunks"][0], sbs["two_tier_chunks"][1], ""],
        ["offline has_target (same replay)",
         agg["has_target_offline"]["before_threshold_0p25"],
         agg["has_target_offline"]["after_top1_confirm3"], "ep1807 run_0000"],
        ["camera span p50 [ms]", cam["before"]["span_ms"]["p50"],
         cam["after"]["span_ms"]["p50"], ""],
        ["camera span p95 / max [ms]",
         f"{cam['before']['span_ms']['p95']} / {cam['before']['span_ms']['max']}",
         f"{cam['after']['span_ms']['p95']} / {cam['after']['span_ms']['max']}", "limit 100"],
        ["chunks with span >= 100 ms", cam["before"]["n_ge_100ms"],
         cam["after"]["n_ge_100ms"], ""],
    ]
    draw(axes[0, 0],
         "(a) " + ("같은 ep1807 · 같은 서버 설정 — 바뀐 것은 target 선택 하나"
                   if korean else "same ep1807, same server - only target selection changed"),
         ["", "before (threshold 0.25)", "after (top-1 + confirm 3)", ""],
         rows_a, [0.40, 0.24, 0.26, 0.16])

    # --- (b) target 이 된 물체 -------------------------------------------------------------
    rows_b = []
    for nm, d in agg["target_size_by_name_after"].items():
        rows_b.append([nm, d["n"],
                       f"{d['point_count']['min']:.0f}-{d['point_count']['max']:.0f}",
                       f"{d['rms_radius_mm']['min']:.0f}-{d['rms_radius_mm']['max']:.0f}",
                       f"{d['name_purity']['min']:.2f}-{d['name_purity']['max']:.2f}",
                       f"{d['target_score']['min']:.3f}-{d['target_score']['max']:.3f}"])
    rows_b.append(["table / unknown / robot", agg["wrong_object_as_target"]["n_after"],
                   "-", "-", "-", "-"])
    rows_b.append(["flood shape (>=10,000 pts",
                   len(agg["wrong_object_as_target"]["flood_by_t5e_unit_test_shape"]),
                   "and compactness < 0.01)", "", "", ""])
    draw(axes[0, 1],
         "(b) " + ("이후가 target 으로 고른 것 — MuJoCo 참값 이름으로"
                   if korean else "what the after run chose, by MuJoCo ground-truth name"),
         ["object", "chunks", "point count", "rms radius [mm]", "name purity", "score"],
         rows_b, [0.30, 0.10, 0.18, 0.18, 0.14, 0.20])

    # --- (c) violated 7 건 ----------------------------------------------------------------
    rows_c = []
    for v in agg["live_violated_after"]:
        rows_c.append([v["seq"], v["t_step"], v["task_phase"], v["target_name_after"],
                       f"{v['violation_before_sqp_mm']:.1f}",
                       f"{v['max_violation_mm']:.2f}",
                       f"{v['gt_min_clearance_mm']:+.3f}",
                       f"{v['gt_min_pair'][0]} vs {v['gt_min_pair'][1]}"])
    draw(axes[1, 0],
         "(c) " + ("이후 실행의 violated 7 건과 그 chunk 의 MuJoCo 참값 clearance"
                   if korean
                   else "the 7 violated chunks and their MuJoCo GT clearance"),
         ["seq", "t_step", "phase", "target", "pre-SQP [mm]", "max viol [mm]",
          "GT clear [mm]", "GT closest pair"],
         rows_c, [0.06, 0.08, 0.11, 0.09, 0.13, 0.14, 0.13, 0.26], fs=7.8)

    # --- (d) 기준선 ------------------------------------------------------------------------
    rows_d = [
        ["위반으로 시작 / frames" if korean else "violated at start / frames",
         f"{base['violated_start']}/{base['frames']}", "14/15", "same"],
        ["frame0 clearance_before [mm]", f"{base['frame0_clearance_before_mm']:.17f}",
         "0.15718632962849477", "same"],
        ["has_target", f"{base['has_target']}/{base['frames']}", "9/15", "changed"],
        ["해소 / 개선" if korean else "resolved / improved",
         f"{base['resolved']} / {base['improved']}", "not a baseline", "-"],
        ["feasible / violated",
         f"{base['status_counts'].get('feasible', 0)} / "
         f"{base['status_counts'].get('violated', 0)}", "not a baseline", "-"],
        ["", "", "", ""],
        ["GT clearance, all 75 chunks",
         f"min {agg['gt_clearance_all_chunks']['min_mm']['min']:.3f} mm", "", ""],
        ["  negative GT chunks",
         f"{agg['gt_clearance_all_chunks']['n_negative']} "
         f"(seq {agg['gt_clearance_all_chunks']['negative_seqs'][0]}-"
         f"{agg['gt_clearance_all_chunks']['negative_seqs'][-1]})", "", ""],
        ["  distmax cross-check 0.02/0.05/0.10",
         "no frame disagrees", "", ""],
    ]
    draw(axes[1, 1],
         "(d) " + ("회귀 기준선 — 둘은 그대로, has_target 은 이 변경으로 바뀐다"
                   if korean
                   else "regression baseline - two hold, has_target changes by design"),
         ["", "measured now", "baseline", ""],
         rows_d, [0.38, 0.30, 0.22, 0.14], highlight=(2,))

    fig.suptitle(("T5f — 숫자를 나란히 (ep1807, 75 chunk, 새 촬영 없음)"
                  if korean else "T5f - the numbers side by side (ep1807, 75 chunks)"),
                 fontsize=13, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=165, facecolor=figstyle.SURFACE)
    SIDECAR.write_text(json.dumps({
        "source": "outputs/verify/T5f/aggregate_T5f.json",
        "panel_a": rows_a, "panel_b": rows_b, "panel_c": rows_c, "panel_d": rows_d,
    }, indent=1, ensure_ascii=False))
    print("wrote", PNG)


if __name__ == "__main__":
    main()
