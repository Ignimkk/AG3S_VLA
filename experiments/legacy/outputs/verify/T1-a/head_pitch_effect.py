"""T1-a 보강 — 머리 관절을 teleop 값으로 되돌리면 cam_high 가 사과를 보나.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1-a/head_pitch_effect.py --records <run dir> --out <json>

기록의 qpos 를 그대로 놓고 **head_1 만** 바꿔 같은 프레임을 두 번 잡는다. 다른 것은 손대지
않으므로 차이는 머리 각도 하나에서만 온다. 기록 파일은 고치지 않는다 — 메모리 위에서만 바꾼다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np
import mujoco

from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")


def apple_px(scene, apple_ids, cam):
    b = np.asarray(scene.capture(cam).body_ids)
    return int(np.isin(b, apple_ids).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--target", default="apple")
    ap.add_argument("--teleop-head1", type=float, default=0.7,
                    help="model_transport.xml 의 teleop 키프레임 값")
    args = ap.parse_args()

    run = load_run(args.records, limit=args.frames)
    scene = replay_scene(run)
    name = lambda i: mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    apple_ids = [i for i in range(scene.model.nbody) if args.target in name(i)]
    jn = list(run.meta["joint_names"])
    qi = {j: jn.index(j) for j in ("head_0", "head_1")}

    rows = []
    for i in range(min(args.frames, len(run.steps))):
        step = run.steps[i]
        rec_h1 = float(np.asarray(step.qpos)[qi["head_1"]])

        pose_scene(scene, step)
        as_recorded = {c: apple_px(scene, apple_ids, c) for c in CAMS}

        # head_1 만 teleop 값으로. qpos 사본에만 쓴다.
        scene.data.qpos[qi["head_1"]] = args.teleop_head1
        mujoco.mj_forward(scene.model, scene.data)
        with_teleop = {c: apple_px(scene, apple_ids, c) for c in CAMS}

        rows.append({"frame": i, "t_step": int(step.t_step), "recorded_head_1": rec_h1,
                     "as_recorded": as_recorded, "with_teleop_head_1": with_teleop})
        print(f"  frame {i:3d}  head_1 기록 {rec_h1:+.4f} -> teleop {args.teleop_head1:+.2f}   "
              f"zed_left 사과 {as_recorded['zed_left']:6d} -> {with_teleop['zed_left']:6d} px")

    summary = {
        "recorded_head_1_unique": sorted({round(r["recorded_head_1"], 6) for r in rows}),
        "teleop_head_1": args.teleop_head1,
        "delta_rad": args.teleop_head1 - rows[0]["recorded_head_1"],
        "delta_deg": float(np.degrees(args.teleop_head1 - rows[0]["recorded_head_1"])),
        "zed_left_frames_with_apple": {
            "as_recorded": sum(1 for r in rows if r["as_recorded"]["zed_left"] > 0),
            "with_teleop": sum(1 for r in rows if r["with_teleop_head_1"]["zed_left"] > 0)},
        "zed_left_median_px": {
            "as_recorded": int(np.median([r["as_recorded"]["zed_left"] for r in rows])),
            "with_teleop": int(np.median([r["with_teleop_head_1"]["zed_left"] for r in rows]))},
        "zed_left_max_px": {
            "as_recorded": max(r["as_recorded"]["zed_left"] for r in rows),
            "with_teleop": max(r["with_teleop_head_1"]["zed_left"] for r in rows)},
    }
    out = {"records": args.records, "target": args.target, "n_frames": len(rows),
           "xml_keyframe": "robots/rby1/description/models/rby1a/mujoco/model_transport.xml "
                           "key name='teleop' (head_0 head_1 = 0 0.7)",
           "method": "기록 qpos 를 그대로 놓고 head_1 만 메모리에서 바꿔 같은 프레임을 두 번 캡처",
           "summary": summary, "rows": rows}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    print(f"  head_1 {rows[0]['recorded_head_1']:+.4f} -> {args.teleop_head1:+.2f} "
          f"({summary['delta_deg']:.1f} deg)")
    print(f"  zed_left 에 사과가 보이는 프레임  {summary['zed_left_frames_with_apple']['as_recorded']}"
          f" -> {summary['zed_left_frames_with_apple']['with_teleop']} / {len(rows)}")
    print(f"  zed_left 사과 픽셀 중앙  {summary['zed_left_median_px']['as_recorded']}"
          f" -> {summary['zed_left_median_px']['with_teleop']}")


if __name__ == "__main__":
    main()
