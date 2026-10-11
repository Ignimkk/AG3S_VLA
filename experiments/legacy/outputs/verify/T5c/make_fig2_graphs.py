"""T5c figure 2 (그래프) — 75 청크에 걸친 판정·점수·카메라 시각·지연.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5c/make_fig2_graphs.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

SHADOW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/shadow_agg.json'))
OFFLINE = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/offline_grounding_ep1807.json'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t5c'))
NAME = "t5c-chunkwise-verdict-and-score"
BUDGET_MS = 533.3333333333334
SKEW_LIMIT_MS = 100.0


def main() -> None:
    KO = figstyle.use_korean()

    def T(ko, en):
        return ko if KO else en

    sh = json.loads(SHADOW.read_text())
    off = json.loads(OFFLINE.read_text())
    rows = sh["per_chunk"]
    thr = off["target_score_threshold"]
    t = np.array([r["t_step"] for r in rows])
    off_by_t = {f["t_step"]: f for f in off["frames"]}
    score = np.array([off_by_t[int(x)]["target_score"] for x in t])
    off_has = np.array([off_by_t[int(x)]["has_target"] for x in t])
    live_has = np.array([r["has_target"] for r in rows])
    span = np.array([c["span_ms"] for c in sh["camera_capture"]["per_frame"]])
    span_t = np.array([c["t_step"] for c in sh["camera_capture"]["per_frame"]])

    C = figstyle.CATEGORICAL
    fig, axes = plt.subplots(4, 1, figsize=(12.6, 11.2), sharex=True)
    figstyle.style_axes(fig, axes)
    a0, a1, a2, a3 = axes

    # (a) 청크별 판정 띠 — live 서버 경로
    kinds = [("has_target (grounding ok)", lambda r: r["has_target"], C[2]),
             ("no_target (grounding low_score)", lambda r: r["ag3s_status"] == "no_target", C[1]),
             ("degraded (camera timing)", lambda r: r["ag3s_status"] == "degraded", C[3])]
    for yi, (label, pred, col) in enumerate(kinds):
        xs = [r["t_step"] for r in rows if pred(r)]
        a0.bar(xs, 0.8, bottom=yi, width=7.0, color=col, align="edge",
               label=f"{label}  n={len(xs)}")
    viol = [r["t_step"] for r in rows if r["trajopt_status"] == "violated"
            and r["ag3s_status"] == "ok"]
    a0.scatter(viol, [3.4] * len(viol), marker="v", s=46, color=C[7], zorder=3,
               label=f"TO violated (ag3s ok)  n={len(viol)}")
    safe = [r["t_step"] for r in rows if r["safe"]]
    a0.scatter(safe, [-0.5] * len(safe), marker="s", s=10, color=C[0], zorder=3,
               label=f"safe=True  n={len(safe)}")
    a0.set_yticks([0.4, 1.4, 2.4])
    a0.set_yticklabels(["has_target", "no_target", "degraded"], fontsize=8)
    a0.set_ylim(-1.1, 5.6)
    a0.set_title(T("(a) live 서버 경로(shadow) 75 청크의 판정 — ep1807",
                   "(a) live server path (shadow), verdict over 75 chunks — ep1807"),
                 fontsize=10, color=figstyle.INK, loc="left")
    a0.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper center")

    # (b) 오프라인 target_score
    a1.axhline(thr, color=figstyle.INK_2, lw=1.0, ls=(0, (4, 3)))
    a1.text(597, thr, T(f" 문턱 {thr}", f" threshold {thr}"), fontsize=8,
            color=figstyle.INK_2, va="bottom", ha="right")
    a1.plot(t, score, lw=1.0, color=figstyle.GRID_INK, zorder=1)
    a1.scatter(t[off_has], score[off_has], s=26, color=C[2], zorder=3,
               label=T(f"오프라인 has_target  n={int(off_has.sum())}",
                       f"offline has_target  n={int(off_has.sum())}"))
    a1.scatter(t[~off_has], score[~off_has], s=26, color=C[1], zorder=3,
               label=T(f"오프라인 low_score  n={int((~off_has).sum())}",
                       f"offline low_score  n={int((~off_has).sum())}"))
    for x in t[live_has]:
        a1.axvspan(x - 4, x + 4, color=C[0], alpha=0.10, lw=0, zorder=0)
    a1.set_ylabel("target_score", fontsize=9, color=figstyle.INK)
    a1.set_ylim(0, 1.0)
    a1.set_title(T("(b) 같은 에피소드의 오프라인 재생 경로 target_score "
                   "(옅은 띠 = live 에서 has_target 인 청크)",
                   "(b) offline replay path target_score on the same episode "
                   "(shaded = chunks where live had a target)"),
                 fontsize=10, color=figstyle.INK, loc="left")
    a1.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=2)

    # (c) 카메라 세 대 캡처 span
    a2.axhline(SKEW_LIMIT_MS, color=figstyle.INK_2, lw=1.0, ls=(0, (4, 3)))
    a2.text(597, SKEW_LIMIT_MS, T(" 한도 100 ms", " limit 100 ms"), fontsize=8,
            color=figstyle.INK_2, va="bottom", ha="right")
    over = span >= SKEW_LIMIT_MS
    a2.plot(span_t, span, lw=1.0, color=C[0], zorder=2)
    a2.scatter(span_t[over], span[over], s=34, color=C[3], zorder=3,
               label=T(f"한도 이상 {int(over.sum())} 프레임",
                       f"at/over limit: {int(over.sum())} frames"))
    a2.set_ylabel(T("세 카메라 캡처 폭 (ms)", "3-camera capture span (ms)"),
                  fontsize=9, color=figstyle.INK)
    a2.set_title(T("(c) zed_left → wrist_cam_l → wrist_cam_r 첫·끝 시각차",
                   "(c) first-to-last camera stamp span"),
                 fontsize=10, color=figstyle.INK, loc="left")
    a2.legend(fontsize=7.5, frameon=False, loc="upper right")

    # (d) 단계별 지연
    for i, key in enumerate(("ag3s", "trajopt", "total")):
        ys = [r["timing_ms"][key] for r in rows]
        a3.plot(t, ys, lw=1.2, color=C[i], label=key)
    a3.axhline(BUDGET_MS, color=figstyle.INK_2, lw=1.0, ls=(0, (4, 3)))
    a3.text(597, BUDGET_MS, T(" 청크 예산 533 ms", " chunk budget 533 ms"), fontsize=8,
            color=figstyle.INK_2, va="bottom", ha="right")
    a3.set_yscale("log")
    a3.set_ylabel(T("서버 단계 (ms, 로그)", "server stage (ms, log)"), fontsize=9,
                  color=figstyle.INK)
    a3.set_xlabel(T("제어 스텝 t (청크마다 8 스텝)", "control step t (8 per chunk)"),
                  fontsize=9, color=figstyle.INK)
    a3.set_title(T("(d) 서버 단계별 지연 — 첫 청크는 cuRobo 커널 컴파일을 포함한다",
                   "(d) server stage latency — the first chunk includes cuRobo kernel compile"),
                 fontsize=10, color=figstyle.INK, loc="left")
    a3.legend(fontsize=7.5, frameon=False, loc="upper right", ncol=3)

    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / f"{NAME}.png"
    fig.savefig(png, dpi=150, facecolor=figstyle.SURFACE)
    (FIG / f"{NAME}.json").write_text(json.dumps({
        "sources": {"live_shadow": str(SHADOW), "offline": str(OFFLINE)},
        "target_score_threshold": thr, "chunk_budget_ms": BUDGET_MS,
        "camera_skew_limit_ms": SKEW_LIMIT_MS,
        "live": {"has_target": int(live_has.sum()),
                 "no_target": sum(1 for r in rows if r["ag3s_status"] == "no_target"),
                 "degraded": sum(1 for r in rows if r["ag3s_status"] == "degraded"),
                 "safe": sum(1 for r in rows if r["safe"]),
                 "to_violated_with_ag3s_ok": viol},
        "offline": {"has_target": int(off_has.sum()), "low_score": int((~off_has).sum())},
        "per_chunk": [{"t_step": int(x), "live_has_target": bool(h),
                       "live_ag3s_status": r["ag3s_status"],
                       "offline_target_score": float(s), "camera_span_ms": float(sp),
                       "ag3s_ms": r["timing_ms"]["ag3s"], "trajopt_ms": r["timing_ms"]["trajopt"],
                       "total_ms": r["timing_ms"]["total"]}
                      for x, h, r, s, sp in zip(t, live_has, rows, score, span)],
    }, indent=1, ensure_ascii=False))
    print("wrote", png)


if __name__ == "__main__":
    main()
