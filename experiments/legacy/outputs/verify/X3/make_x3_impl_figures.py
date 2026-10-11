"""X3 구현 검증 figure 3종 — max_points 를 올린 뒤 무엇이 그대로이고 무엇이 달라졌나.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/X3/make_x3_impl_figures.py

숫자 출처:
  - 기준선 재현      : outputs/verify/X3/baseline_ep1800_alone.json
  - 부하중 실행      : outputs/verify/X3/baseline_ep1800_after.json
  - 프레임별 반복 수 : outputs/verify/X3/probe_ep1800_cap60k.json
  - 상한별 점군      : 이 스크립트가 기록 depth 에서 직접 만든다
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from AG3S.ag3s.config import PointCloudConfig
from AG3S.ag3s.stages.reconstruction import reconstruct
from AG3S.ag3s.experiments.common import figstyle
from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

KO = figstyle.use_korean()
C = figstyle.CATEGORICAL
OK, BAD = C[2], C[7]
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/x3'))
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/X3'))
HEADFIX = _ag3s_workspace_path('outputs/live_test/20260925_headfix16d/records/run_0000/run_0000')
FRAME = 2  # 인증이 처음 꺼진 프레임


def T(ko: str, en: str) -> str:
    return ko if KO else en


def clouds_at_both_caps():
    """세 카메라를 **합친 뒤** 상한을 건다 — 파이프라인이 그렇게 한다.

    처음엔 카메라마다 따로 `reconstruct` 를 불렀는데 그러면 카메라당 약 26k 라 상한에
    안 걸려 두 쪽이 똑같이 나왔다. 상한은 fusion 뒤의 구름에 걸린다.
    """
    import dataclasses

    from AG3S.ag3s.stages.reconstruction import coverage_preserving_cap

    run = load_run(HEADFIX, limit=FRAME + 1)
    scene = replay_scene(run)
    pose_scene(scene, run.steps[FRAME])

    base_cfg = PointCloudConfig(voxel_size=0.005, range_max=1.6, depth_max=3.0,
                                cap_strategy="voxel", max_points=10 ** 9)
    parts = []
    for cam in ("zed_left", "wrist_cam_l", "wrist_cam_r"):
        cf = scene.capture(cam)
        cloud, _ = reconstruct(depth=cf.depth, camera_intrinsics=cf.camera_intrinsics,
                               T_base_cam=cf.T_base_cam, config=base_cfg)
        parts.append(cloud)

    fused_points = np.concatenate([np.asarray(c.points) for c in parts], 0)
    # uv 는 카메라마다 자기 픽셀을 가리키므로 합치면 뜻이 없다. 밀도만 보는 그림이라 버린다.
    fused = dataclasses.replace(parts[0], points=fused_points, uv=None)

    caps = {}
    for tag, mx in (("cap_60k", 60000), ("cap_200k", 200000)):
        capped, st = coverage_preserving_cap(fused, dataclasses.replace(base_cfg, max_points=mx))
        caps[tag] = (np.asarray(capped.points), [st])
    return caps


def scene_figure(caps) -> dict:
    """위에서 내려다본 점군 — 상한 하나가 밀도를 어떻게 바꾸나."""
    fig = plt.figure(figsize=(13.0, 5.4))
    gs = GridSpec(1, 3, figure=fig, wspace=0.22, width_ratios=[1, 1, 0.85])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    figstyle.style_axes(fig, np.array(axes))

    # 배치도 대신: 어느 방향에서 보는지 축 라벨로 못박는다 (위에서 내려다본 x-y 평면).
    lim = None
    for ax, tag, label in ((axes[0], "cap_60k", T("max_points 60000", "max_points 60000")),
                           (axes[1], "cap_200k", T("max_points 200000", "max_points 200000"))):
        p, st = caps[tag]
        vox = st[0]["voxel_size"] * 1000.0
        # 테이블 높이 근처 한 겹만 — 전부 그리면 밀도 차가 안 보인다.
        m = (p[:, 2] > 0.80) & (p[:, 2] < 0.86)
        ax.scatter(p[m, 0], p[m, 1], s=0.6, c=C[0], alpha=0.55, linewidths=0)
        ax.set_aspect("equal")
        if lim is None:
            lim = (ax.get_xlim(), ax.get_ylim())
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"{label}\n" + T(f"voxel {vox:.1f} mm · 이 층 {int(m.sum())} 점",
                                     f"voxel {vox:.1f} mm · {int(m.sum())} pts in slice"),
                     fontsize=10)

    ax = axes[2]
    ax.axis("off")
    n60 = len(caps["cap_60k"][0])
    n200 = len(caps["cap_200k"][0])
    c60 = sum(1 for s in caps["cap_60k"][1] if s["capped"])
    c200 = sum(1 for s in caps["cap_200k"][1] if s["capped"])
    v60 = caps["cap_60k"][1][0]["voxel_size"] * 1000.0
    v200 = caps["cap_200k"][1][0]["voxel_size"] * 1000.0
    lines = [
        T(f"수정 후 기록 · 프레임 {FRAME}", f"headfix record · frame {FRAME}"), "",
        T("자른 자리: z 0.80 ~ 0.86 m 한 겹", "slice: z 0.80 - 0.86 m"),
        T("보는 방향: 위에서 아래로 (x-y)", "viewed from above (x-y)"), "",
        T(f"점 수 (세 카메라 합)   {n60}  →  {n200}", f"points (3 cams)   {n60}  ->  {n200}"),
        T(f"상한이 걸렸나          {bool(c60)}  →  {bool(c200)}",
          f"did the cap bind   {bool(c60)}  ->  {bool(c200)}"),
        T(f"voxel                 {v60:.1f} mm  →  {v200:.1f} mm",
          f"voxel   {v60:.1f} mm  ->  {v200:.1f} mm"), "",
        T("사라진 영역은 없다 — 대표점이", "no region disappeared —"),
        T("성기어졌을 뿐이다.", "the representatives just got sparser."),
    ]
    ax.text(0.0, 0.95, "\n".join(lines), va="top", ha="left", fontsize=10,
            color=figstyle.INK, transform=ax.transAxes, linespacing=1.75)

    fig.suptitle(T("X3 구현 — 상한 하나가 점군 밀도를 어떻게 바꾸나",
                   "X3 implementation — what one cap does to the cloud"),
                 fontsize=12, color=figstyle.INK)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "x3-impl-scene.png", dpi=150, bbox_inches="tight",
                facecolor=figstyle.SURFACE)
    plt.close(fig)
    return {"frame": FRAME, "slice_z_m": [0.80, 0.86],
            "n_points_3cam": {"cap_60k": int(n60), "cap_200k": int(n200)},
            "cap_bound": {"cap_60k": int(c60), "cap_200k": int(c200)},
            "final_voxel_mm": {
                "cap_60k": [round(s["voxel_size"] * 1000, 1) for s in caps["cap_60k"][1]],
                "cap_200k": [round(s["voxel_size"] * 1000, 1) for s in caps["cap_200k"][1]]}}


def load_trap_figure() -> dict:
    alone = json.load(open(V / "baseline_ep1800_alone.json"))["frames"]
    conc = json.load(open(V / "baseline_ep1800_after.json"))["frames"]
    probe = json.load(open(V / "probe_ep1800_cap60k.json"))["frames"]
    x = [r["i"] for r in alone]
    ca = [r["clearance_after_mm"] for r in alone]
    cc = [r["clearance_after_mm"] for r in conc]
    ia = [r["iterations"] for r in alone]
    ic = [r["iterations"] for r in conc]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    figstyle.style_axes(fig, axes)

    ax = axes[0]
    ax.axhline(0, color=figstyle.INK, lw=1.2)
    ax.plot(x, ca, "-o", color=OK, ms=5, label=T("단독 실행", "run alone"))
    ax.plot(x, cc, "-o", color=BAD, ms=5, label=T("pytest·다른 rollout 과 동시", "run under load"))
    ax.set_xlabel(T("정책 호출", "policy call"))
    ax.set_ylabel("clearance_after (mm)")
    ax.set_title(T("같은 코드 · 같은 기록 · 같은 명령", "same code, record and command"),
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    w = 0.38
    ax.bar(np.array(x) - w / 2, ia, w, color=OK, label=T("단독", "alone"))
    ax.bar(np.array(x) + w / 2, ic, w, color=BAD, label=T("부하중", "under load"))
    ax.set_xlabel(T("정책 호출", "policy call"))
    ax.set_ylabel(T("SQP 반복 수", "SQP iterations"))
    ax.set_title(T("벽시계 마감 50 ms 가 반복을 깎는다",
                   "the 50 ms wall-clock deadline cuts iterations"), fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    ax.axis("off")
    fa = sum(1 for r in alone if r["status"] == "feasible")
    fc = sum(1 for r in conc if r["status"] == "feasible")
    flip = [i for i, (a, b) in enumerate(zip(ia, ic)) if a != b]
    lines = [
        T("하마터면 이것을 코드 탓으로 적을 뻔했다",
          "this was almost recorded as a code regression"), "",
        T(f"feasible   단독 {fa} / 15", f"feasible   alone {fa} / 15"),
        T(f"           부하중 {fc} / 15", f"           under load {fc} / 15"), "",
        T(f"반복이 줄어든 프레임  {flip}", f"frames with fewer iterations  {flip}"),
        T(f"프레임 10 clearance  {ca[10]:+.2f} → {cc[10]:+.2f} mm",
          f"frame 10 clearance  {ca[10]:+.2f} -> {cc[10]:+.2f} mm"), "",
        T("sqp.time_budget_ms 는 벽시계 마감이다", "sqp.time_budget_ms is a wall-clock deadline"),
        T("(sqp.py:159). 부하가 걸리면 같은 50 ms 에", "(sqp.py:159). under load the same 50 ms"),
        T("반복을 덜 돌고 더 나쁜 iterate 를 돌려준다.", "buys fewer iterations."), "",
        T("→ 회귀 기준선은 단독으로 돌린다.", "-> run the baseline alone."),
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9.5,
            color=figstyle.INK, transform=ax.transAxes, linespacing=1.6)

    fig.tight_layout()
    fig.savefig(OUT / "x3-impl-loadtrap.png", dpi=150, bbox_inches="tight",
                facecolor=figstyle.SURFACE)
    plt.close(fig)
    return {"clearance_alone_mm": ca, "clearance_under_load_mm": cc,
            "iterations_alone": ia, "iterations_under_load": ic,
            "feasible": {"alone": fa, "under_load": fc},
            "frames_with_fewer_iterations": flip}


def table_figure() -> dict:
    alone = json.load(open(V / "baseline_ep1800_alone.json"))["frames"]
    head = json.load(open(V / "headfix_after_maxpoints.json"))["frames"]

    def counts(r):
        cb = [x["clearance_before_mm"] for x in r]
        ca = [x["clearance_after_mm"] for x in r]
        return {"violated_start": sum(1 for v in cb if v < 0),
                "resolved": sum(1 for a, b in zip(ca, cb) if b < 0 <= a),
                "improved": sum(1 for a, b in zip(ca, cb) if a > b),
                "feasible": sum(1 for x in r if x["status"] == "feasible"),
                "violated": sum(1 for x in r if x["status"] == "violated"),
                "has_target": sum(1 for x in r if x["target"])}

    a, h = counts(alone), counts(head)
    rows = [
        [T("위반으로 시작", "violated at start"), "15/15", f"{a['violated_start']}/15",
         f"{h['violated_start']}/15"],
        [T("해소", "resolved"), "13", str(a["resolved"]), str(h["resolved"])],
        [T("개선", "improved"), "15", str(a["improved"]), str(h["improved"])],
        ["feasible", "13", str(a["feasible"]), str(h["feasible"])],
        ["violated", "2", str(a["violated"]), str(h["violated"])],
        ["has_target", "9/15", f"{a['has_target']}/15", f"{h['has_target']}/15"],
    ]
    header = [T("지표", "metric"), T("문서화된 기준선", "documented baseline"),
              T("변경 뒤 ep1800 (단독)", "after change, ep1800 (alone)"),
              T("변경 뒤 headfix 기록", "after change, headfix record")]

    fig, ax = plt.subplots(figsize=(11.0, 3.6))
    figstyle.style_axes(fig, np.array([ax]))
    ax.axis("off")
    tab = ax.table(cellText=rows, colLabels=header, cellLoc="center", loc="center",
                   colWidths=[0.24, 0.25, 0.27, 0.24])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9.5)
    tab.scale(1, 1.7)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        if r == 0:
            cell.set_facecolor("#eef2f7")
        elif c in (1, 2) and rows[r - 1][1] == rows[r - 1][2]:
            cell.set_facecolor("#eef7f0")
    ax.set_title(T("max_points 60000 → 200000 뒤 — 회귀 기준선은 한 자리도 안 움직인다\n"
                   "(기준선 기록은 상한에 애초에 안 걸린다: 점 수 53625~91268, capped 0/15)",
                   "after max_points 60000 -> 200000 — the baseline does not move\n"
                   "(that record never hits the cap: 53625-91268 points, capped 0/15)"),
                 fontsize=10.5, color=figstyle.INK, pad=14)
    fig.savefig(OUT / "x3-impl-table.png", dpi=150, bbox_inches="tight",
                facecolor=figstyle.SURFACE)
    plt.close(fig)
    return {"documented_baseline": {"violated_start": "15/15", "resolved": 13, "improved": 15,
                                    "feasible": 13, "violated": 2, "has_target": "9/15"},
            "after_change_ep1800_alone": a, "after_change_headfix": h}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    caps = clouds_at_both_caps()
    s = scene_figure(caps)
    (OUT / "x3-impl-scene.json").write_text(json.dumps(
        {"figure": "x3-impl-scene.png", "numbers": s}, indent=1, ensure_ascii=False))
    print("wrote", OUT / "x3-impl-scene.png")

    t = load_trap_figure()
    (OUT / "x3-impl-loadtrap.json").write_text(json.dumps(
        {"figure": "x3-impl-loadtrap.png",
         "sources": ["outputs/verify/X3/baseline_ep1800_alone.json",
                     "outputs/verify/X3/baseline_ep1800_after.json"],
         "numbers": t}, indent=1, ensure_ascii=False))
    print("wrote", OUT / "x3-impl-loadtrap.png")

    tb = table_figure()
    (OUT / "x3-impl-table.json").write_text(json.dumps(
        {"figure": "x3-impl-table.png", "numbers": tb}, indent=1, ensure_ascii=False))
    print("wrote", OUT / "x3-impl-table.png")
    print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
