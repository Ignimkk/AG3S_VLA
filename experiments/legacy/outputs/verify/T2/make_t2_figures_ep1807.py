"""T2 ep1807 figure 3종 — 아홉 이벤트, 손/과일 궤적, 그리퍼-latch 타임라인.

`outputs/verify/T2/t2_events_ep1807.json` 을 읽는다. 해석 없음 — 프레임 번호와 궤적만.
ep1800 그림(`t2-*.png`)을 덮지 않는다 — 전부 `-ep1807` 접미사.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T2/make_t2_figures_ep1807.py
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

from AG3S.ag3s.experiments.common import figstyle

OUT_DIR = pathlib.Path("AG3S/ag3s/docs/figures/t2")
EVENTS_JSON = pathlib.Path("outputs/verify/T2/t2_events_ep1807.json")
RECORDS = "outputs/live_test/20260925_ep1807/run_0000"

# 값이 None 이면 이 기록에서 그 사건이 일어나지 않았다는 뜻 (좌표 플롯에서는 생략)
EVENT_FRAMES = {
    "target acquired": 0, "target latched": 2, "grasp contact": 19, "attached": 19,
    "destination attention (raw)": 28, "destination attention (locked)": None,
    "transport start (hand > 5cm)": 21, "placement": 32, "detached": 32,
    "task_state_reset": None,
}


def main() -> None:
    figstyle.use_korean()
    d = json.loads(EVENTS_JSON.read_text())
    frames = d["frames"]

    from AG3S.ag3s.experiments.sources.policy_record import load_run, replay_scene, pose_scene
    run = load_run(RECORDS)
    scene = replay_scene(run)
    hand_xy, obj_pos = [], {n: [] for n in ("apple", "banana", "orange", "pear", "crate")}
    for step in run:
        pose_scene(scene, step)
        hand_xy.append(scene.body_position("ee_finger_l1")[:2].copy())
        for n in obj_pos:
            obj_pos[n].append(scene.body_position(n).copy())
    scene.close()
    hand_xy = np.array(hand_xy)
    for n in obj_pos:
        obj_pos[n] = np.array(obj_pos[n])

    # --- 1. 실제 씬: 손 궤적 + 과일/crate 위치(고정) 위에서 본 모습 ------------------------
    fig, ax = plt.subplots(figsize=(8, 9.5))
    ax.plot(hand_xy[:, 0], hand_xy[:, 1], "-", color=figstyle.GRID_INK, lw=1, zorder=1)
    sc = ax.scatter(hand_xy[:, 0], hand_xy[:, 1], c=np.arange(len(hand_xy)),
                    cmap=figstyle.sequential_cmap(), s=18, zorder=2)
    fig.colorbar(sc, ax=ax, label="observation frame i", shrink=0.8)
    markers = {"apple": "o", "banana": "o", "orange": "o", "pear": "o", "crate": "s"}
    for ci, (n, m) in enumerate(markers.items()):
        p0 = obj_pos[n][0]
        pN = obj_pos[n][-1]
        drift = float(np.linalg.norm(pN[:2] - p0[:2]))
        ax.scatter(*p0[:2], marker=m, s=140, facecolors="none",
                  edgecolors=figstyle.CATEGORICAL[ci], linewidths=2,
                  label=f"{n} (drift {drift*1000:.1f} mm)")
    k = 0
    labels_plotted = []
    for label, i in EVENT_FRAMES.items():
        k += 1
        if i is None:
            labels_plotted.append(f"{k}={label}(없음)")
            continue
        ax.annotate(str(k), hand_xy[i], fontsize=8, color=figstyle.INK, fontweight="bold",
                   xytext=(6, 6), textcoords="offset points")
        labels_plotted.append(f"{k}={label}(f{i})")
    event_legend = "   ".join(labels_plotted)
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_title("T2 ep1807 - 왼손(ee_finger_l1) 궤적과 과일-crate 위치 (위에서)")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.set_aspect("equal")
    figstyle.style_axes(fig, [ax])
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.text(0.02, 0.02, event_legend, fontsize=6.5, wrap=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "t2-scene-hand-trajectory-ep1807.png", dpi=140)
    plt.close(fig)

    # --- 2. 그래프: 그리퍼 값 · latch phase · attach/detach 타임라인 ---------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [1.2, 0.8]})
    xs = [f["i"] for f in frames]
    grip = [f["gripper_left_state7"] for f in frames]
    ax1.plot(xs, grip, "-o", ms=3, color=figstyle.CATEGORICAL[0], label="gripper_left (state[7])")
    ax1.axhline(0.85, color=figstyle.INK_2, ls="--", lw=1, label="gripper_closed_below=0.85")
    for f in frames:
        if f["attach_this_frame"]:
            ax1.axvline(f["i"], color=figstyle.CATEGORICAL[2], lw=1.5)
            ax1.annotate("attach", (f["i"], 1.02), fontsize=8, color=figstyle.CATEGORICAL[2])
        if f["detach_this_frame"]:
            ax1.axvline(f["i"], color=figstyle.CATEGORICAL[3], lw=1.5)
            ax1.annotate("detach", (f["i"], 1.02), fontsize=8, color=figstyle.CATEGORICAL[3])
        if f["contact_fruit"]:
            ax1.scatter([f["i"]], [0.05], marker="^", s=40, c=figstyle.CATEGORICAL[6],
                       zorder=5)
    ax1.scatter([], [], marker="^", c=figstyle.CATEGORICAL[6], label="mujoco contact (finger-fruit)")
    ax1.set_ylabel("gripper_left value")
    ax1.legend(fontsize=8, loc="center right")

    phase_order = ["searching", "latched", "held", "placed"]
    phase_y = [phase_order.index(f["latch_phase"]) for f in frames]
    ax2.step(xs, phase_y, where="post", color=figstyle.CATEGORICAL[1])
    ax2.set_yticks(range(len(phase_order)))
    ax2.set_yticklabels(phase_order)
    ax2.set_xlabel("observation frame i")
    ax2.set_ylabel("GraspPhase")
    ax2.axvline(32, color=figstyle.GRID_INK, lw=1, ls=":")
    ax2.annotate("detach (frame32) 이후\nPLACED 에서 못 빠져나옴\n(2차 파지 시도 없음, 이 기록엔)",
               (33, 3.05), fontsize=7, color=figstyle.INK_2)
    fig.suptitle("T2 ep1807 - 그리퍼 값 · attach/detach · GraspPhase 타임라인 (75 프레임)")
    figstyle.style_axes(fig, [ax1, ax2])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "t2-timeline-ep1807.png", dpi=140)
    plt.close(fig)

    # --- 3. 표: 아홉 이벤트 + 물체 이동량 ------------------------------------------------
    rows = []
    for label, i in EVENT_FRAMES.items():
        if i is None:
            rows.append([label, "없음", "-"])
        else:
            rows.append([label, str(i), str(frames[i]["t_step"])])
    rows.append(["placed_ground_truth == True 프레임", "32 (전 75 프레임 중 유일)", "256"])
    rows.append(["2차 파지 시도", "없음 (그리퍼가 frame32 뒤 1.0 로 고정)", "-"])

    obj_rows = []
    for n in obj_pos:
        p0, pN = obj_pos[n][0], obj_pos[n][-1]
        obj_rows.append([n, f"{p0.round(3).tolist()}", f"{pN.round(3).tolist()}",
                        f"{np.linalg.norm(pN-p0)*1000:.1f} mm"])

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(11, 6.8),
                                  gridspec_kw={"height_ratios": [1.15, 0.6]})
    for ax in (axa, axb):
        ax.axis("off")
    t1 = axa.table(cellText=rows, colLabels=["사건", "관측 프레임 i", "t_step"],
                  loc="center", cellLoc="left")
    t1.auto_set_font_size(False)
    t1.set_fontsize(8.5)
    t1.scale(1, 1.4)
    axa.set_title("T2 ep1807 - 아홉 이벤트의 관측 프레임 (1차 파지 사이클, 유일한 사이클)", fontsize=11)

    t2 = axb.table(cellText=obj_rows, colLabels=["body", "위치 @frame0 (m)", "위치 @frame74 (m)",
                                                 "총 변위"],
                  loc="center", cellLoc="center")
    t2.auto_set_font_size(False)
    t2.set_fontsize(8.5)
    t2.scale(1, 1.7)
    axb.set_title("T2 ep1807 - 물체 위치, 관측 시작 대 끝 (75 프레임 전체)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "t2-events-table-ep1807.png", dpi=140)
    plt.close(fig)

    (OUT_DIR / "t2-events-table-ep1807.json").write_text(json.dumps(
        {"event_frames": EVENT_FRAMES, "object_start_end": {
            n: {"start": obj_pos[n][0].tolist(), "end": obj_pos[n][-1].tolist(),
               "drift_mm": float(np.linalg.norm(obj_pos[n][-1] - obj_pos[n][0]) * 1000)}
            for n in obj_pos}}, indent=1, ensure_ascii=False))
    (OUT_DIR / "t2-scene-hand-trajectory-ep1807.json").write_text(json.dumps(
        {"hand_xy": hand_xy.tolist(), "event_frames": EVENT_FRAMES}, indent=1))
    (OUT_DIR / "t2-timeline-ep1807.json").write_text(json.dumps(
        {"frames": [{k: f[k] for k in ("i", "t_step", "gripper_left_state7", "latch_phase",
                                       "attach_this_frame", "detach_this_frame", "contact_fruit")}
                   for f in frames]}, indent=1, ensure_ascii=False))
    print("wrote", OUT_DIR / "t2-scene-hand-trajectory-ep1807.png")
    print("wrote", OUT_DIR / "t2-timeline-ep1807.png")
    print("wrote", OUT_DIR / "t2-events-table-ep1807.png")


if __name__ == "__main__":
    main()
