"""T5c figure 3 (표·도식) — 세 경로를 나란히 · 카메라 시각 6 건 · 지연 표.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5c/make_fig3_tables.py
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

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

SHADOW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/shadow_agg.json'))
OFFLINE = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/offline_grounding_ep1807.json'))
BASE = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/baseline_check.json'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t5c'))
NAME = "t5c-paths-and-camera-timing-table"


def _table(ax, head, rows, widths, title, KO):
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=head, cellLoc="center", colLoc="center",
                   loc="center", colWidths=widths)
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.42)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        if r == 0:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(color=figstyle.INK)
        if c == 0:
            cell.set_text_props(ha="left")
    ax.set_title(title, fontsize=10.5, color=figstyle.INK, pad=12, loc="left")


def main() -> None:
    KO = figstyle.use_korean()

    def T(ko, en):
        return ko if KO else en

    sh = json.loads(SHADOW.read_text())
    off = json.loads(OFFLINE.read_text())
    base = json.loads(BASE.read_text())
    bf = base["frames"]
    n_base_target = sum(1 for f in bf if f["target"])
    thr = off["target_score_threshold"]
    ots = off["target_score"]

    head1 = [T("경로", "path"),
             T("live shadow · ep1807", "live shadow · ep1807"),
             T("오프라인 재생 · ep1807", "offline replay · ep1807"),
             T("회귀 기준선 · ep1800", "regression baseline · ep1800")]
    rows1 = [
        [T("청크(프레임) 수", "chunks (frames)"), "75", "75", "15"],
        [T("has_target", "has_target"),
         f"16 / 75 (21.3 %)", f"{off['has_target_true']} / 75 "
         f"({off['has_target_true']/75*100:.1f} %)",
         f"{n_base_target} / 15 ({n_base_target/15*100:.1f} %)"],
        [T("grounding low_score", "grounding low_score"),
         str(sh["counts"]["grounding_status"].get("low_score", 0)),
         str(off["grounding_status_counts"].get("low_score", 0)), "not_measured"],
        [T("ag3s_status no_target", "ag3s_status no_target"),
         str(sh["counts"]["ag3s_status"].get("no_target", 0)),
         str(off["status_counts"].get("no_target", 0)), "not_measured"],
        [T("ag3s_status degraded", "ag3s_status degraded"),
         str(sh["counts"]["ag3s_status"].get("degraded", 0)), "0", "not_measured"],
        [T("safe / uncertified", "safe / uncertified"),
         f"{sh['counts']['safe']} / {sh['counts']['unsafe']}", "—",
         f"{sum(1 for f in bf if f['safe'])} / {sum(1 for f in bf if not f['safe'])}"],
        [T("target_score 를 싣는가", "target_score reported?"),
         T("아니오 (응답·로그 어디에도 없다)", "no (not in response or log)"),
         T("예 (measure 스크립트)", "yes (measure script)"),
         T("아니오", "no")],
        [T("attention 출처", "attention source"),
         T("정책 체크포인트 2 번째 사본 (live)", "2nd checkpoint copy (live)"),
         "attention_16d_ep1807.npz", "attention_16d_ep1800.npz"],
        [T("ESDF backend", "ESDF backend"), "curobo 20 mm + 5 mm", "legacy 20 mm",
         "legacy 20 mm"],
    ]

    head2 = [T("오프라인 target_score (문턱 %.2f)" % thr,
               "offline target_score (threshold %.2f)" % thr), T("값", "value")]
    rows2 = [
        [T("실패 46 프레임 최솟값", "failing 46 frames, min"), f"{ots['no_target']['min']:.4f}"],
        [T("실패 46 프레임 중앙값", "failing 46 frames, median"), f"{ots['no_target']['p50']:.4f}"],
        [T("실패 46 프레임 최댓값", "failing 46 frames, max"), f"{ots['no_target']['max']:.4f}"],
        [T("문턱 0.05 아래 이내", "within 0.05 below threshold"),
         f"{ots['n_within_0p05_below_threshold']} / 46"],
        [T("문턱 절반(0.125) 미만", "below half threshold (0.125)"),
         f"{ots['n_below_half_threshold']} / 46"],
        [T("성공 29 프레임 중앙값", "passing 29 frames, median"),
         f"{ots['has_target']['p50']:.4f}"],
    ]

    head3 = [T("degraded 청크", "degraded chunk"), "seq", "t_step",
             T("head 지연 (ms)", "head lag (ms)"), T("카메라 폭 (ms)", "camera span (ms)"),
             "grounding"]
    rows3 = []
    for i, d in enumerate(sh["degraded"]["detail"], 1):
        rows3.append([f"#{i}", str(d["seq"]), str(d["t_step"]),
                      str(d["camera_stale_ms"][0]) if d["camera_stale_ms"] else "—",
                      str(d["camera_skew_ms"][0]) if d["camera_skew_ms"] else "—",
                      d["grounding_status"]])
    cc = sh["camera_capture"]
    rows3.append([T("75 프레임 전체", "all 75 frames"), "—", "—", "—",
                  f"p50 {cc['span_ms']['p50']:.1f} · p95 {cc['span_ms']['p95']:.1f} · "
                  f"max {cc['span_ms']['max']:.1f} · min {cc['span_ms']['min']:.1f}", "—"])
    rows3.append([T("한도 100 ms 이상", "at/over the 100 ms limit"), "—", "—", "—",
                  f"{cc['n_span_ge_100ms']} / 75  (>=90 ms: {cc['n_span_ge_90ms']} / 75)", "—"])
    rows3.append([T("카메라 사이 평균 간격", "mean gap between cameras"), "—", "—", "—",
                  " · ".join(f"{k} {v:.1f}" for k, v in cc["gap_ms_mean"].items()), "—"])

    head4 = [T("서버 단계", "server stage"), T("첫 청크 (seq 1)", "first chunk (seq 1)"),
             "P50", "P95", "max"]
    tm = sh["timing_ms"]["excluding_first"]
    fc = sh["timing_ms"]["first_chunk_seq1"]
    rows4 = []
    for key, label in (("infer", "infer (policy)"), ("ag3s", "ag3s"),
                       ("trajopt", "trajopt (SQP)"), ("total", "server total")):
        rows4.append([label, f"{fc[key]:.0f}", f"{tm[key]['p50']:.0f}",
                      f"{tm[key]['p95']:.0f}", f"{tm[key]['max']:.0f}"])
    rt = tm["client_roundtrip_ms_excl_first"]
    rows4.append([T("클라이언트 왕복", "client round trip"), f"{fc['client_roundtrip_ms']:.0f}",
                  f"{rt['p50']:.0f}", f"{rt['p95']:.0f}", f"{rt['max']:.0f}"])
    rows4.append([T("청크 예산 (8 / 15 Hz)", "chunk budget (8 / 15 Hz)"), "533", "533", "533",
                  "533"])

    fig, axes = plt.subplots(4, 1, figsize=(13.0, 15.2),
                             gridspec_kw={"height_ratios": [2.5, 1.8, 2.5, 2.0]})
    figstyle.style_axes(fig, axes)
    _table(axes[0], head1, rows1, [0.26, 0.26, 0.24, 0.24],
           T("(1) 같은 파이프라인, 세 경로 — 단위는 청크(프레임) 수",
             "(1) same pipeline, three paths — counts are chunks (frames)"), KO)
    _table(axes[1], head2, rows2, [0.55, 0.45],
           T("(2) 오프라인 경로에서 실제로 잰 target_score — live 경로는 이 값을 싣지 않는다",
             "(2) target_score measured on the offline path — the live path does not report it"), KO)
    _table(axes[2], head3, rows3, [0.24, 0.08, 0.10, 0.14, 0.32, 0.12],
           T("(3) camera timing 으로 degraded 가 된 6 청크와 75 프레임 캡처 폭",
             "(3) the 6 chunks degraded by camera timing, and capture span over 75 frames"), KO)
    _table(axes[3], head4, rows4, [0.28, 0.18, 0.18, 0.18, 0.18],
           T("(4) 서버 단계별 지연 (ms) — 실시간성은 이번 판정 항목이 아니다",
             "(4) server stage latency (ms) — real-time is out of scope for this verdict"), KO)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / f"{NAME}.png"
    fig.savefig(png, dpi=150, facecolor=figstyle.SURFACE)
    (FIG / f"{NAME}.json").write_text(json.dumps({
        "sources": {"live_shadow": str(SHADOW), "offline": str(OFFLINE),
                    "regression_baseline": str(BASE)},
        "tables": {"paths": [head1] + rows1, "target_score": [head2] + rows2,
                   "camera_timing": [head3] + rows3, "latency": [head4] + rows4},
    }, indent=1, ensure_ascii=False))
    print("wrote", png)


if __name__ == "__main__":
    main()
