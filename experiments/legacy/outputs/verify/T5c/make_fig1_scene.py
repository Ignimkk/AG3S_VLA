"""T5c figure 1 (실제 씬) — grounding 이 서는 프레임과 무너지는 프레임을 나란히.

ep1807 기록(`outputs/live_test/20260925_ep1807/run_0000`, T2 가 쓴 것과 같은 기록)을 다시
세워 프레임마다 3인칭 자유 카메라 + 정책이 실제로 본 세 카메라를 그린다. 사과는 MuJoCo
segmentation 참값으로 윤곽만 겹친다 — 어느 카메라가 사과를 보는가를 눈으로 셀 수 있게.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5c/make_fig1_scene.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T1'))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from AG3S.ag3s.experiments.common import figstyle  # noqa: E402

RECORDS = "outputs/live_test/20260925_ep1807/run_0000"
OFFLINE = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/offline_grounding_ep1807.json'))
SHADOW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/shadow_agg.json'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t5c'))
NAME = "t5c-scene-grounding-frames"
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
T_STEPS = (64, 120, 424)


def main() -> None:
    import mujoco
    from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

    KO = figstyle.use_korean()

    def T(ko, en):
        return ko if KO else en

    off = json.loads(OFFLINE.read_text())
    off_by_t = {f["t_step"]: f for f in off["frames"]}
    sh = json.loads(SHADOW.read_text())
    sh_by_t = {r["t_step"]: r for r in sh["per_chunk"]}

    run = load_run(RECORDS)
    scene = replay_scene(run, height=480, width=640)
    steps = {int(s.t_step): s for s in run}
    apple_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "apple")

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    tid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "table")
    base = scene.data.xpos[tid] if tid >= 0 else np.array([0.65, 0.0, 0.0])
    cam.lookat[:] = [base[0], base[1], 0.95]
    cam.distance, cam.azimuth, cam.elevation = 2.2, 135.0, -25.0
    renderer = mujoco.Renderer(scene.model, 480, 640)

    fig, axes = plt.subplots(len(T_STEPS), 4, figsize=(13.0, 3.1 * len(T_STEPS)))
    figstyle.style_axes(fig, axes)
    side = {}
    for r, t in enumerate(T_STEPS):
        pose_scene(scene, steps[t])
        renderer.update_scene(scene.data, cam)
        third = renderer.render().copy()
        frames = {c: scene.capture(c) for c in CAMS}
        o, s = off_by_t[t], sh_by_t[t]
        side[t] = {
            "offline": {k: o[k] for k in ("target_score", "grounding_status", "has_target",
                                          "apple_pixels", "ee_apple_dist_mm", "phase",
                                          "target_center_error_mm")},
            "live_shadow": {k: s[k] for k in ("seq", "ag3s_status", "grounding_status",
                                              "trajopt_status", "safe", "reason_codes")},
        }
        ax = axes[r, 0]
        ax.imshow(third)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"t={t}  " + T("3인칭 참값", "third person (ground truth)"),
                     fontsize=9, color=figstyle.INK)
        ax.set_ylabel(T(f"오프라인 점수 {o['target_score']:.3f}\nlive {s['ag3s_status']}",
                        f"offline score {o['target_score']:.3f}\nlive {s['ag3s_status']}"),
                      fontsize=8.5, color=figstyle.INK)
        for c_i, c in enumerate(CAMS):
            ax = axes[r, c_i + 1]
            fr = frames[c]
            renderer.update_scene(scene.data, camera=c)
            ax.imshow(renderer.render().copy())
            mask = (np.asarray(fr.body_ids) == apple_id)
            if mask.any():
                ax.contour(mask.astype(float), levels=[0.5],
                           colors=[figstyle.CATEGORICAL[1]], linewidths=1.4)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{c} ({T('재현', 'rerender')})  "
                         f"{T('사과 화소', 'apple px')} {int(mask.sum())}",
                         fontsize=9, color=figstyle.INK)
    fig.suptitle(T("ep1807 — grounding 이 서는 프레임과 무너지는 프레임 (윤곽 = 사과 참값)",
                   "ep1807 — frames where grounding holds and where it collapses "
                   "(outline = apple ground truth)"),
                 fontsize=11, color=figstyle.INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    FIG.mkdir(parents=True, exist_ok=True)
    png = FIG / f"{NAME}.png"
    fig.savefig(png, dpi=150, facecolor=figstyle.SURFACE)
    (FIG / f"{NAME}.json").write_text(json.dumps(
        {"records": RECORDS, "t_steps": list(T_STEPS),
         "target_score_threshold": off["target_score_threshold"],
         "note": "왼쪽 열은 MuJoCo 자유 카메라, 나머지는 정책이 본 세 카메라. 사과 윤곽은 "
                 "MuJoCo body segmentation 참값. live_shadow 열은 서버 경로의 판정 "
                 "(target_score 를 싣지 않는다).",
         "frames": side}, indent=1, ensure_ascii=False))
    scene.close()
    print("wrote", png)
    print(json.dumps(side, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
