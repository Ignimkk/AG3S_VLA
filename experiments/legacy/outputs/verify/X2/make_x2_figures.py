"""X2 재촬영 figure 3종 — head cam 을 고친 기록이 무엇을 바꿨나.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/X2/make_x2_figures.py

두 기록은 `--episode-index 1800` 으로 argv 가 출력 경로만 빼고 같다. 차이는
`model_transport.xml` 의 teleop 키프레임 하나뿐이라 나란히 놓을 수 있다.

숫자 출처:
  - zed_left 사과 픽셀 (수정 후) : outputs/verify/X2/head_pitch_headfix.json
  - zed_left 사과 픽셀 (수정 전) : outputs/verify/T1-a/visibility_ep1800.json
  - head_1 : 두 기록의 qpos 에서 직접
  - 회귀 지표 대조표 : AG3S_T0T6_LOG.md 의 "재촬영 결과" 절 (raw 미보존 — X3 가 재생성한다)
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
import mujoco

from AG3S.ag3s.experiments.common import figstyle
from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

KO = figstyle.use_korean()
C = figstyle.CATEGORICAL
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/x2'))
VER = pathlib.Path(_ag3s_workspace_path('outputs/verify/X2'))

BEFORE = _ag3s_workspace_path('legacy/run_16d_ep1800')
AFTER = _ag3s_workspace_path('outputs/live_test/20260925_headfix16d/records/run_0000/run_0000')

# 로그 "재촬영 결과" 절의 대조표. raw 가 남지 않아 여기서만 인용한다.
LOG_TABLE = [
    ("has_target — 지었나", "9/15", "15/15"),
    ("위반으로 시작", "15/15", "15/15"),
    ("해소 (feasible 이 된 수)", "13", "15"),
    ("개선 (clearance 나아진 수)", "15", "15"),
    ("feasible — 최악 clearance >= 0", "13", "2"),
    ("violated — 최악 clearance < 0", "2", "13"),
    ("clearance_after 중앙", "+4.35 mm", "+4.19 mm"),
    ("clearance_after 최소", "-3.15 mm", "+0.02 mm"),
    ("n_candidates > 0 프레임", "0", "0"),
    ("target_voxels_carved 총합", "0", "0"),
    ("AG3S 실행 중앙값", "1115 ms", "3169 ms"),
]
LOG_TABLE_EN = [
    ("has_target", "9/15", "15/15"),
    ("violated at start", "15/15", "15/15"),
    ("resolved", "13", "15"),
    ("improved", "15", "15"),
    ("feasible (worst clearance >= 0)", "13", "2"),
    ("violated (worst clearance < 0)", "2", "13"),
    ("clearance_after median", "+4.35 mm", "+4.19 mm"),
    ("clearance_after min", "-3.15 mm", "+0.02 mm"),
    ("frames with n_candidates > 0", "0", "0"),
    ("target_voxels_carved total", "0", "0"),
    ("AG3S runtime median", "1115 ms", "3169 ms"),
]


def T(ko: str, en: str) -> str:
    return ko if KO else en


def px(row: dict, cam: str) -> int:
    """두 visibility json 의 모양이 다르다 — T1-a 는 평평하고 X2 는 as_recorded 아래다."""
    if "as_recorded" in row:
        return int(row["as_recorded"][cam])
    return int(row[cam])


def head1_trace(run) -> tuple[np.ndarray, int]:
    jn = list(run.meta["joint_names"])
    qi = jn.index("head_1")
    return np.array([float(np.asarray(s.qpos)[qi]) for s in run.steps]), qi


def apple_mask(run, frame: int):
    """그 프레임 zed_left 의 depth 와 사과 마스크."""
    scene = replay_scene(run)
    name = lambda i: mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    apple_ids = [i for i in range(scene.model.nbody) if "apple" in name(i)]
    pose_scene(scene, run.steps[frame])
    cf = scene.capture("zed_left")
    return cf.depth, np.isin(np.asarray(cf.body_ids), apple_ids)


def scene_figure(before, after, px_before, px_after) -> dict:
    d_bef, m_bef = apple_mask(before, 0)
    d_aft, m_aft = apple_mask(after, 0)

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.18)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    figstyle.style_axes(fig, np.array(axes))

    # (0,0) 배치도 — 머리 관절 하나가 시선을 어디로 보내나
    ax = axes[0]
    ax.set_aspect("equal")
    ax.plot([0, 1.0], [0, 0], color=C[3], lw=2.4, ls="--")
    ax.plot([0, 0.765], [0, -0.644], color=C[0], lw=2.8)
    ax.plot([0], [0], "o", color=figstyle.INK, ms=7)
    ax.annotate(T("head_1 = 0 (기록된 값)", "head_1 = 0 (as recorded)"), (1.0, 0.0),
                xytext=(0.40, 0.14), color=C[3], fontsize=9)
    ax.annotate(T("head_1 = 0.7 (teleop 키프레임)", "head_1 = 0.7 (teleop keyframe)"),
                (0.765, -0.644), xytext=(0.20, -0.86), color=C[0], fontsize=9)
    ax.plot([0.25, 1.05], [-0.62, -0.62], color=figstyle.INK_2, lw=3)
    ax.text(1.06, -0.62, T("테이블", "table"), color=figstyle.INK_2, fontsize=9, va="center")
    ax.set_xlim(-0.15, 1.35)
    ax.set_ylim(-1.05, 0.35)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(T("머리 관절 하나 — 40.1 도", "one head joint — 40.1 deg"), fontsize=10)

    # (0,1) (0,2) 정책이 실제로 본 영상
    for ax, run, tag, px in ((axes[1], before, T("수정 전", "before"), px_before),
                             (axes[2], after, T("수정 후", "after"), px_after)):
        ax.imshow(run.steps[0].images["cam_high"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"cam_high — {tag}\n" + T(f"사과 {px} px", f"apple {px} px"), fontsize=10)

    # (1,0) 수정 후 depth
    ax = axes[3]
    im = ax.imshow(np.clip(d_aft, 0, 3.0), cmap=figstyle.sequential_cmap())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(T("zed_left depth — 수정 후 (m)", "zed_left depth — after (m)"), fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)

    # (1,1) (1,2) 사과 마스크
    for ax, mask, dep, tag in ((axes[4], m_bef, d_bef, T("수정 전", "before")),
                               (axes[5], m_aft, d_aft, T("수정 후", "after"))):
        ax.imshow(np.clip(dep, 0, 3.0), cmap="Greys", vmin=0, vmax=4.2)
        ov = np.zeros(mask.shape + (4,))
        ov[mask] = matplotlib.colors.to_rgba(C[1])
        ax.imshow(ov)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(T(f"zed_left 사과 마스크 — {tag}\n{int(mask.sum())} px",
                       f"zed_left apple mask — {tag}\n{int(mask.sum())} px"), fontsize=10)

    fig.suptitle(T("X2 — head cam 이 벽을 보고 있었다. 같은 에피소드 1800, 프레임 0",
                   "X2 — the head cam faced a wall. same episode 1800, frame 0"),
                 fontsize=12, color=figstyle.INK)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "x2-scene.png", dpi=150, bbox_inches="tight",
                facecolor=figstyle.SURFACE)
    plt.close(fig)
    return {"frame": 0,
            "zed_left_apple_px_frame0": {"before": int(m_bef.sum()), "after": int(m_aft.sum())},
            "gaze_unit_vectors": {"head_1=0.0": [1, 0, 0], "head_1=0.7": [0.765, 0, -0.644]},
            "gaze_delta_deg": 40.1}


def trend_figure(before, after, vis_before, vis_after) -> dict:
    h_bef, _ = head1_trace(before)
    h_aft, _ = head1_trace(after)
    px_bef = [px(r, "zed_left") for r in vis_before["rows"]]
    px_aft = [px(r, "zed_left") for r in vis_after["rows"]]
    n = len(px_aft)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    figstyle.style_axes(fig, axes)

    ax = axes[0]
    ax.plot(x, h_bef[:n], "-o", color=C[3], ms=4, label=T("수정 전", "before"))
    ax.plot(x, h_aft[:n], "-o", color=C[0], ms=4, label=T("수정 후", "after"))
    ax.axhline(0.7, color=figstyle.INK_2, lw=1, ls=":")
    ax.text(0.2, 0.71, T("teleop 키프레임 0.7", "teleop keyframe 0.7"),
            color=figstyle.INK_2, fontsize=8)
    ax.set_xlabel(T("정책 호출", "policy call"))
    ax.set_ylabel(T("head_1 (rad)", "head_1 (rad)"))
    ax.set_title(T("머리 관절 — 프레임마다", "head joint — per frame"), fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    ax.plot(x, px_bef, "-o", color=C[3], ms=4, label=T("수정 전", "before"))
    ax.plot(x, px_aft, "-o", color=C[0], ms=4, label=T("수정 후", "after"))
    ax.set_xlabel(T("정책 호출", "policy call"))
    ax.set_ylabel(T("사과 픽셀 수", "apple pixels"))
    ax.set_title(T("zed_left 에 사과가 몇 픽셀 있나", "apple pixels in zed_left"), fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]
    cams = ("zed_left", "wrist_cam_l", "wrist_cam_r")
    w = 0.38
    for k, (src, tag, col) in enumerate(((vis_before, T("수정 전", "before"), C[3]),
                                         (vis_after, T("수정 후", "after"), C[0]))):
        vals = [sum(1 for r in src["rows"] if px(r, c) > 0) for c in cams]
        ax.bar(np.arange(3) + (k - 0.5) * w, vals, w, color=col, label=tag)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(cams, fontsize=8)
    ax.set_ylabel(T("사과가 보이는 프레임 수 / 15", "frames with apple / 15"))
    ax.set_title(T("어느 카메라가 사과를 보나", "which camera sees the apple"), fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "x2-trend.png", dpi=150, bbox_inches="tight", facecolor=figstyle.SURFACE)
    plt.close(fig)
    return {
        "head_1": {"before": [round(v, 6) for v in h_bef[:n].tolist()],
                   "after": [round(v, 6) for v in h_aft[:n].tolist()]},
        "zed_left_apple_px": {"before": px_bef, "after": px_aft},
        "zed_left_apple_px_median": {"before": int(np.median(px_bef)),
                                     "after": int(np.median(px_aft))},
        "frames_with_apple": {
            c: {"before": sum(1 for r in vis_before["rows"] if px(r, c) > 0),
                "after": sum(1 for r in vis_after["rows"] if px(r, c) > 0)}
            for c in cams},
    }


def table_figure() -> dict:
    rows = LOG_TABLE if KO else LOG_TABLE_EN
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    figstyle.style_axes(fig, np.array([ax]))
    ax.axis("off")
    head = [T("지표", "metric"), T("수정 전 (머리 0)", "before (head 0)"),
            T("수정 후 (머리 0.7)", "after (head 0.7)")]
    tab = ax.table(cellText=[list(r) for r in rows], colLabels=head,
                   cellLoc="center", colLoc="center", loc="center",
                   colWidths=[0.48, 0.26, 0.26])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.55)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor(figstyle.GRID_INK)
        if r == 0:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(color=figstyle.INK)
        elif rows[r - 1][0].startswith(("has_target", "feasible", "violated")):
            cell.set_facecolor("#fdf3ea")
        if c == 0:
            cell.set_text_props(ha="left")
    ax.set_title(T("X2 재촬영 — 같은 명령, 키프레임 하나 차이\n"
                   "raw 가 남지 않았다. 출처는 AG3S_T0T6_LOG.md 의 재촬영 결과 절이고 X3 가 재생성한다",
                   "X2 re-capture — same command, one keyframe apart\n"
                   "raw was not saved; source is the log's re-capture section, X3 regenerates it"),
                 fontsize=11, color=figstyle.INK, pad=16)
    fig.savefig(OUT / "x2-table.png", dpi=150, bbox_inches="tight", facecolor=figstyle.SURFACE)
    plt.close(fig)
    return {"rows": [list(r) for r in LOG_TABLE],
            "source": "AG3S/ag3s/docs/AG3S_T0T6_LOG.md — '재촬영 결과 (2026-09-25)' 절",
            "raw_output": None,
            "note": "이 열한 줄만 verify.json 근거가 없다. X3 가 재실행해 raw 를 남긴다."}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    before = load_run(BEFORE, limit=15)
    after = load_run(AFTER, limit=15)
    vis_before = json.loads(
        pathlib.Path(_ag3s_workspace_path('outputs/verify/T1-a/visibility_ep1800.json')).read_text())
    vis_after = json.loads((VER / "head_pitch_headfix.json").read_text())

    px_b = px(vis_before["rows"][0], "zed_left")
    px_a = px(vis_after["rows"][0], "zed_left")

    s = scene_figure(before, after, px_b, px_a)
    (OUT / "x2-scene.json").write_text(json.dumps(
        {"figure": "x2-scene.png", "records": {"before": BEFORE, "after": AFTER},
         "numbers": s}, indent=1, ensure_ascii=False))
    print("wrote", OUT / "x2-scene.png")

    t = trend_figure(before, after, vis_before, vis_after)
    (OUT / "x2-trend.json").write_text(json.dumps(
        {"figure": "x2-trend.png",
         "sources": {"before": "outputs/verify/T1-a/visibility_ep1800.json",
                     "after": "outputs/verify/X2/head_pitch_headfix.json"},
         "numbers": t}, indent=1, ensure_ascii=False))
    print("wrote", OUT / "x2-trend.png")

    tb = table_figure()
    (OUT / "x2-table.json").write_text(json.dumps(
        {"figure": "x2-table.png", "numbers": tb}, indent=1, ensure_ascii=False))
    print("wrote", OUT / "x2-table.png")


if __name__ == "__main__":
    main()
