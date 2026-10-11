"""T1-a 보강 figure — 수집 영상 대 기록 영상, 그리고 머리 관절 하나의 차이.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1-a/make_headfix_figure.py

새 이름으로만 쓴다 (t1a-headfix-*). 기존 t1a-* 는 건드리지 않는다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import subprocess
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import mujoco

from AG3S.ag3s.experiments.common import figstyle
from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

KO = figstyle.use_korean()
C = figstyle.CATEGORICAL
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t1-a'))
VER = pathlib.Path(_ag3s_workspace_path('outputs/verify/T1-a'))
MP4 = (_ag3s_workspace_path('legacy/pi05_TO_hybrid/data/rby1_randomized_pick_place_16d_v1/videos/chunk-001/observation.images.cam_high/episode_001800.mp4'))
XML_KEY = "robots/rby1/description/models/rby1a/mujoco/model_transport.xml  key 'teleop'"
FRAME = 0
TELEOP_H1 = 0.7


def T(ko: str, en: str) -> str:
    return ko if KO else en


def collection_frame(n: int) -> np.ndarray:
    """원본 mp4 에서 n 번째 프레임을 직접 뽑는다."""
    dst = VER / "collection_frames" / f"camhigh_f{n}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", MP4,
                    "-vf", f"select=eq(n\\,{n})", "-vframes", "1", str(dst)], check=True)
    return mpimg.imread(dst)


def main() -> None:
    run = load_run("run_16d_ep1800", limit=15)
    scene = replay_scene(run)
    name = lambda i: mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    apple_ids = [i for i in range(scene.model.nbody) if "apple" in name(i)]
    jn = list(run.meta["joint_names"])
    h1 = jn.index("head_1")

    step = run.steps[FRAME]
    rec_h1 = float(np.asarray(step.qpos)[h1])

    pose_scene(scene, step)
    d_rec = np.asarray(scene.capture("zed_left").depth, float)
    px_rec = int(np.isin(np.asarray(scene.capture("zed_left").body_ids), apple_ids).sum())

    scene.data.qpos[h1] = TELEOP_H1
    mujoco.mj_forward(scene.model, scene.data)
    cap = scene.capture("zed_left")
    d_fix = np.asarray(cap.depth, float)
    m_fix = np.isin(np.asarray(cap.body_ids), apple_ids)
    px_fix = int(m_fix.sum())

    coll = collection_frame(FRAME)
    rec_rgb = np.asarray(step.images["cam_high"])

    hp = json.loads((VER / "head_pitch_ep1800.json").read_text())
    hl = json.loads((VER / "head_pitch_long.json").read_text())

    fig = plt.figure(figsize=(15.0, 8.4))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.16)

    for k, (img, title, col) in enumerate([
        (coll, T("수집 영상 cam_high (원본 mp4, 프레임 0)",
                 "collection video cam_high (source mp4, frame 0)"), C[2]),
        (rec_rgb, T("기록의 image_cam_high (같은 에피소드 1800)",
                    "record image_cam_high (same episode 1800)"), C[7]),
    ]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9, color=col)

    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    lines = [
        T("같은 에피소드 1800 인데 화각이 다르다.", "same episode 1800, different framing."),
        "",
        T("머리 관절 (qpos index 30)", "head joint (qpos index 30)") + ":",
        f"    teleop keyframe   head_1 = {TELEOP_H1:+.2f} rad",
        f"    " + T("기록 전 프레임", "record, every frame") + f"   head_1 = {rec_h1:+.4f} rad",
        f"    " + T("차이", "difference")
        + f"   {np.degrees(TELEOP_H1 - rec_h1):.1f}" + T(" 도 안 숙임", " deg less pitch"),
        "",
        XML_KEY,
        "",
        T("head_0 는 양쪽 다 0 이다 — 좌우는 같고 상하만 다르다.",
          "head_0 is 0 in both - only the pitch differs."),
    ]
    ax.text(0.0, 0.98, "\n".join(lines), va="top", ha="left", fontsize=9, color=figstyle.INK,
            family=matplotlib.rcParams["font.family"][0])

    ax = fig.add_subplot(gs[0, 3])
    xi = np.arange(2)
    ax.bar(xi - 0.2, [hp["summary"]["zed_left_frames_with_apple"]["as_recorded"],
                      hl["summary"]["zed_left_frames_with_apple"]["as_recorded"]],
           0.4, color=C[7], label=T("기록대로 (head_1=0)", "as recorded (head_1=0)"))
    ax.bar(xi + 0.2, [hp["summary"]["zed_left_frames_with_apple"]["with_teleop"],
                      hl["summary"]["zed_left_frames_with_apple"]["with_teleop"]],
           0.4, color=C[2], label=T("head_1=0.7 로 되돌림", "head_1 restored to 0.7"))
    for j, (a, b, n) in enumerate([(0, hp["summary"]["zed_left_frames_with_apple"]["with_teleop"], 15),
                                   (0, hl["summary"]["zed_left_frames_with_apple"]["with_teleop"], 50)]):
        ax.annotate(f"{a}", (xi[j] - 0.2, 0.4), ha="center", fontsize=8, color=figstyle.INK)
        ax.annotate(f"{b}/{n}", (xi[j] + 0.2, b), ha="center", va="bottom", fontsize=8,
                    color=figstyle.INK)
    ax.set_xticks(xi); ax.set_xticklabels(["ep1800 (15)", "long16d (50)"], fontsize=8)
    ax.set_ylabel(T("사과가 보이는 프레임 수", "frames where the apple is visible"))
    ax.set_title(T("zed_left 가 사과를 보나", "does zed_left see the apple"),
                 fontsize=10, color=figstyle.INK)
    ax.legend(fontsize=7, frameon=False)
    figstyle.style_axes(fig, np.array([ax]))

    for k, (d, m, ttl, px) in enumerate([
        (d_rec, None, T(f"기록대로 head_1={rec_h1:+.2f} — zed_left depth",
                        f"as recorded head_1={rec_h1:+.2f} - zed_left depth"), px_rec),
        (d_fix, m_fix, T(f"head_1={TELEOP_H1:+.2f} 로 되돌린 같은 프레임",
                         f"same frame with head_1={TELEOP_H1:+.2f}"), px_fix),
    ]):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(d, cmap=figstyle.sequential_cmap(), vmin=0.2, vmax=2.0)
        if m is not None and m.any():
            ys, xs = np.nonzero(m)
            ax.scatter(xs, ys, s=1.0, color=C[7])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(ttl + T(f"   사과 {px} px", f"   apple {px} px"), fontsize=9,
                     color=C[2] if px else C[7])

    ax = fig.add_subplot(gs[1, 2:])
    rows = hp["rows"]
    ax.plot([r["frame"] for r in rows], [r["as_recorded"]["zed_left"] for r in rows],
            "-o", ms=3, color=C[7], label=T("기록대로 head_1=0", "as recorded head_1=0"))
    ax.plot([r["frame"] for r in rows], [r["with_teleop_head_1"]["zed_left"] for r in rows],
            "-o", ms=3, color=C[2], label=T("head_1=0.7", "head_1=0.7"))
    ax.set_xlabel(T("프레임 (ep1800)", "frame (ep1800)"))
    ax.set_ylabel(T("zed_left 의 사과 픽셀", "apple pixels in zed_left"))
    ax.set_title(T("머리 하나만 바꿨다 — 다른 qpos 는 그대로",
                   "only the head joint changed - every other qpos identical"),
                 fontsize=10, color=figstyle.INK)
    ax.legend(fontsize=8, frameon=False)
    figstyle.style_axes(fig, np.array([ax]))

    fig.suptitle(T("T1-a 보강 — 사과가 cam_high 에 없는 것은 씬이 아니라 기록 생성 쪽이다",
                   "T1-a addendum - the apple's absence from cam_high comes from record "
                   "generation, not the scene"),
                 fontsize=13, color=figstyle.INK)
    fig.savefig(OUT / "t1a-headfix.png", dpi=130, facecolor=figstyle.SURFACE, bbox_inches="tight")
    plt.close(fig)

    side = {
        "source_video": MP4, "video_frame": FRAME,
        "xml_keyframe": XML_KEY, "teleop_head_1": TELEOP_H1,
        "recorded_head_1": rec_h1, "delta_deg": float(np.degrees(TELEOP_H1 - rec_h1)),
        "frame0_zed_left_apple_px": {"as_recorded": px_rec, "with_teleop": px_fix},
        "ep1800": hp["summary"], "long16d": hl["summary"],
    }
    (OUT / "t1a-headfix.json").write_text(json.dumps(side, indent=2, ensure_ascii=False))
    print("wrote", OUT / "t1a-headfix.png", "+ .json")


if __name__ == "__main__":
    main()
