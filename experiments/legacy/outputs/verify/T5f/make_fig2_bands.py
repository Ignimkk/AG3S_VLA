"""T5f figure 2 (그래프) — 이전/이후의 chunk 별 target 이름을 위아래로 나란히 둔 띠.

  (a) 띠 셋 — 위: 이전(threshold 0.25) · 가운데: rank 1 이 가리킨 물체 · 아래: 이후(1 등+confirm 3).
      `hold`/`switch`/`unobserved` 가 일한 chunk 를 아래 띠에 표시한다.
  (b) target cluster 의 점 수와 rms radius (물체 참값 크기 선과 함께).
  (c) 세 카메라 캡처 폭 — 두 실행을 겹쳐 그리고 100 ms 한도를 긋는다.
  (d) 참값 최소 clearance 와 live `violated` chunk.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5f/make_fig2_bands.py
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
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5f'))
OUTDIR = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t5f'))
PNG = OUTDIR / "t5f-target-name-bands.png"
SIDECAR = OUTDIR / "t5f-target-name-bands.json"

NAME_SLOT = {"apple": 0, "crate": 1, "table": 2, "pear": 3, "orange": 4,
             "banana": 5, "robot": 6, "unknown": 7}
NO_TARGET = "#c9c8c4"


def main() -> None:
    korean = figstyle.use_korean()
    agg = json.loads((V / "aggregate_T5f.json").read_text())
    shadow = json.loads((V / "shadow_agg_two_runs.json").read_text())
    ext = json.loads((V / "object_extents_gt.json").read_text())
    rows = agg["rows"]
    seqs = [r["seq"] for r in rows]

    fig = plt.figure(figsize=(14.6, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 1.0], hspace=0.42, wspace=0.22)
    ax_band = fig.add_subplot(gs[0, :])
    ax_size = fig.add_subplot(gs[1, 0])
    ax_rms = fig.add_subplot(gs[1, 1])
    ax_cam = fig.add_subplot(gs[2, 0])
    ax_cl = fig.add_subplot(gs[2, 1])
    figstyle.style_axes(fig, np.array([ax_band, ax_size, ax_rms, ax_cam, ax_cl], dtype=object))

    # ---------------------------------------------------------------- (a) 띠 셋
    tracks = [("before  threshold 0.25", [r["before_name"] for r in rows], 2.0),
              ("rank 1  (ranking alone)", [r["rank1_name"] for r in rows], 1.0),
              ("after  top-1 + confirm 3", [r["after_name"] for r in rows], 0.0)]
    for label, names, y in tracks:
        for r, nm in zip(rows, names):
            c = NO_TARGET if nm is None else figstyle.CATEGORICAL[NAME_SLOT[nm]]
            ax_band.add_patch(Rectangle((r["seq"] - 0.5, y), 1.0, 0.8, facecolor=c,
                                        edgecolor="none"))
        ax_band.text(-1.0, y + 0.4, label, ha="right", va="center", fontsize=9,
                     color=figstyle.INK)
    mark = {"hold": "H", "switch": "S", "unobserved": "U"}
    for r in rows:
        m = mark.get(r["after_mode"])
        if m:
            ax_band.text(r["seq"], -0.30, m, ha="center", va="center", fontsize=7,
                         color=figstyle.INK)
    ax_band.text(-1.0, -0.30, "confirm acted", ha="right", va="center", fontsize=8,
                 color=figstyle.INK_2)
    for r in rows:
        if r["seq"] in [v["seq"] for v in agg["live_violated_after"]]:
            ax_band.text(r["seq"], 3.05, "V", ha="center", va="center", fontsize=7.5,
                         color=figstyle.CATEGORICAL[7])
    ax_band.text(-1.0, 3.05, "live violated (after)", ha="right", va="center", fontsize=8,
                 color=figstyle.INK_2)
    ax_band.set_xlim(-22, 76.5); ax_band.set_ylim(-0.7, 3.4)
    ax_band.set_yticks([]); ax_band.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 75])
    ax_band.set_xlabel("chunk seq  (t_step = (seq - 1) x 8)", fontsize=9)
    ax_band.spines["left"].set_visible(False)
    handles = [Patch(facecolor=figstyle.CATEGORICAL[NAME_SLOT[n]], label=n)
               for n in ("apple", "crate", "pear")]
    handles.append(Patch(facecolor=NO_TARGET, label="no target (LOW_SCORE)"))
    ax_band.legend(handles=handles, fontsize=8, ncol=4, frameon=False,
                   loc="upper left", bbox_to_anchor=(0.0, 1.30))
    ax_band.set_title("(a) " + ("chunk 별 target 이 어느 물체였나 — 세 갈래"
                                if korean else "which object was the target, per chunk"),
                      fontsize=10.5, color=figstyle.INK, loc="left")

    # ---------------------------------------------------------------- (b) 점 수
    for nm in ("apple", "crate", "pear"):
        sel = [r for r in rows if r["after_name"] == nm]
        ax_size.plot([r["seq"] for r in sel], [r["after_point_count"] for r in sel],
                     ls="none", marker="o", ms=3.4,
                     color=figstyle.CATEGORICAL[NAME_SLOT[nm]], label=nm)
    ax_size.axhline(10000, color=figstyle.INK_2, lw=0.9, ls="--")
    ax_size.text(2, 10500, "T5e unit-test flood shape: >= 10,000 pts", fontsize=7,
                 color=figstyle.INK_2)
    ax_size.set_yscale("log"); ax_size.set_ylim(20, 30000)
    ax_size.set_xlabel("chunk seq", fontsize=9)
    ax_size.set_ylabel("target point count", fontsize=9)
    ax_size.legend(fontsize=8, frameon=False)
    ax_size.set_title("(b) " + ("target cluster 의 점 수" if korean else "target point count"),
                      fontsize=10.5, color=figstyle.INK, loc="left")

    # ---------------------------------------------------------------- (c) rms radius
    for nm in ("apple", "crate", "pear"):
        sel = [r for r in rows if r["after_name"] == nm]
        ax_rms.plot([r["seq"] for r in sel], [r["after_rms_radius_mm"] for r in sel],
                    ls="none", marker="o", ms=3.4,
                    color=figstyle.CATEGORICAL[NAME_SLOT[nm]], label=nm)
        hd = ext[nm]["half_diagonal_mm"]
        ax_rms.axhline(hd, color=figstyle.CATEGORICAL[NAME_SLOT[nm]], lw=0.9, ls=":")
        ax_rms.text(1, hd * 1.08, f"{nm} GT half-diag {hd:.0f} mm", fontsize=6.5,
                    va="bottom", color=figstyle.CATEGORICAL[NAME_SLOT[nm]])
    ax_rms.axhline(ext["table"]["half_diagonal_mm"], color=figstyle.CATEGORICAL[2], lw=0.9,
                   ls=":")
    ax_rms.text(1, ext["table"]["half_diagonal_mm"] * 1.08,
                f"table GT half-diag {ext['table']['half_diagonal_mm']:.0f} mm", fontsize=6.5,
                va="bottom", color=figstyle.CATEGORICAL[2])
    ax_rms.set_yscale("log"); ax_rms.set_xlim(-2, 78); ax_rms.set_ylim(15, 3000)
    ax_rms.set_xlabel("chunk seq", fontsize=9)
    ax_rms.set_ylabel("target rms radius [mm]", fontsize=9)
    ax_rms.set_title("(c) " + ("target 의 퍼짐 대 물체 참값 크기"
                               if korean else "target spread vs GT object size"),
                     fontsize=10.5, color=figstyle.INK, loc="left")

    # ---------------------------------------------------------------- (d) 카메라 폭
    for key, lab, slot in (("before_threshold_0p25", "before (threshold 0.25)", 0),
                           ("after_top1_confirm3", "after (top-1 + confirm 3)", 1)):
        ch = shadow["runs"][key]["chunks"]
        ax_cam.plot([c["seq"] for c in ch], [c["camera_span_ms"] for c in ch], lw=1.1,
                    color=figstyle.CATEGORICAL[slot], label=lab)
    ax_cam.axhline(100.0, color=figstyle.CATEGORICAL[7], lw=1.1, ls="--")
    ax_cam.text(2, 101.5, "limit 100 ms", fontsize=7, color=figstyle.CATEGORICAL[7])
    ax_cam.set_xlabel("chunk seq", fontsize=9)
    ax_cam.set_ylabel("three-camera capture span [ms]", fontsize=9)
    ax_cam.legend(fontsize=8, frameon=False, loc="lower right")
    ax_cam.set_title("(d) " + ("세 카메라 캡처 폭 — 두 실행"
                               if korean else "three-camera capture span, both runs"),
                     fontsize=10.5, color=figstyle.INK, loc="left")

    # ---------------------------------------------------------------- (e) 참값 clearance
    ax_cl.plot(seqs, [r["gt_min_clearance_mm"] for r in rows], lw=1.1,
               color=figstyle.CATEGORICAL[6], label="MuJoCo GT min clearance (arm vs objects)")
    ax_cl.axhline(0.0, color=figstyle.INK_2, lw=0.8)
    vio = agg["live_violated_after"]
    ax_cl.plot([v["seq"] for v in vio], [v["gt_min_clearance_mm"] for v in vio], ls="none",
               marker="v", ms=7, color=figstyle.CATEGORICAL[7],
               label="live 'violated' chunk (after)")
    off = {4: (-16, 8), 15: (-20, -12), 16: (2, 8), 32: (4, -10),
           72: (-30, 10), 74: (-4, 12), 75: (4, -12)}
    for v in vio:
        ax_cl.annotate(f"seq {v['seq']}: {v['max_violation_mm']:.1f} mm",
                       (v["seq"], v["gt_min_clearance_mm"]), textcoords="offset points",
                       xytext=off.get(v["seq"], (3, 6)), fontsize=6.5,
                       color=figstyle.CATEGORICAL[7])
    ax_cl.set_xlabel("chunk seq", fontsize=9)
    ax_cl.set_ylabel("GT min clearance [mm]", fontsize=9)
    ax_cl.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax_cl.set_title("(e) " + ("참값 최소 clearance 와 SQP 가 violated 라 한 chunk"
                              if korean
                              else "GT min clearance vs the chunks SQP called violated"),
                    fontsize=10.5, color=figstyle.INK, loc="left")

    fig.suptitle(("T5f — target 선택만 바꿨을 때 무엇이 달라졌나 (ep1807, 75 chunk)"
                  if korean else
                  "T5f - what changed when only the target selection changed (ep1807)"),
                 fontsize=13, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG, dpi=165, facecolor=figstyle.SURFACE)
    SIDECAR.write_text(json.dumps({
        "sources": ["outputs/verify/T5f/aggregate_T5f.json",
                    "outputs/verify/T5f/shadow_agg_two_runs.json",
                    "outputs/verify/T5f/object_extents_gt.json"],
        "bands": {"before_threshold_0p25": [r["before_name"] for r in rows],
                  "rank1": [r["rank1_name"] for r in rows],
                  "after_top1_confirm3": [r["after_name"] for r in rows],
                  "confirm_mode": [r["after_mode"] for r in rows],
                  "seq": seqs},
        "target_point_count": [r["after_point_count"] for r in rows],
        "target_rms_radius_mm": [r["after_rms_radius_mm"] for r in rows],
        "gt_min_clearance_mm": [r["gt_min_clearance_mm"] for r in rows],
        "camera_span_ms": {
            "before": [c["camera_span_ms"] for c in
                       shadow["runs"]["before_threshold_0p25"]["chunks"]],
            "after": [c["camera_span_ms"] for c in
                      shadow["runs"]["after_top1_confirm3"]["chunks"]]},
        "live_violated_after": agg["live_violated_after"],
        "object_gt_half_diagonal_mm": {k: v["half_diagonal_mm"] for k, v in ext.items()
                                       if isinstance(v, dict)},
    }, indent=1, ensure_ascii=False))
    print("wrote", PNG)


if __name__ == "__main__":
    main()
