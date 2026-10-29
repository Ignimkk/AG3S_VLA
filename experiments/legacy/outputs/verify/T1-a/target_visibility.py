"""T1-a — target 이 어느 카메라에 몇 픽셀로 보이나. 프레임 x 카메라 전수.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1-a/target_visibility.py --records <run dir> --out <json>

grounding 은 `cam_high`(zed_left) 의 attention 을 zed_left 클라우드에 얹는다. target 이
그 영상에 없으면 어느 셀을 골라도 지을 것이 없다 — 셀 선택보다 앞서는 조건이라 따로 잰다.
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
FRUIT = ("apple", "banana", "orange", "pear", "crate")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=15)
    ap.add_argument("--target", default="apple")
    args = ap.parse_args()

    run = load_run(args.records, limit=args.frames)
    scene = replay_scene(run)
    name = lambda i: mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
    ids = {t: [i for i in range(scene.model.nbody) if t in name(i)] for t in FRUIT}
    tid = ids[args.target]

    rows = []
    for i in range(min(args.frames, len(run.steps))):
        pose_scene(scene, run.steps[i])
        row = {"frame": i, "t_step": int(run.steps[i].t_step),
               "target_pos_base": np.round(np.asarray(
                   scene.body_position_in_base(args.target), float), 4).tolist()}
        for c in CAMS:
            b = np.asarray(scene.capture(c).body_ids)
            row[c] = int(np.isin(b, tid).sum())
            row[f"{c}_total_px"] = int(b.size)
        rows.append(row)
        print(f"  frame {i:3d}  " + "  ".join(f"{c} {row[c]:6d}" for c in CAMS)
              + f"   apple@{row['target_pos_base']}")

    summary = {c: {"frames_with_any_px": sum(1 for r in rows if r[c] > 0),
                   "max_px": max(r[c] for r in rows),
                   "median_px": int(np.median([r[c] for r in rows]))} for c in CAMS}
    out = {"records": args.records, "target": args.target,
           "target_body_ids": tid, "n_frames": len(rows),
           "note": "grounding 은 cam_high=zed_left 의 attention 만 쓴다 "
                   "(esdf_rollout.py:137 ci=cameras.index('cam_high'))",
           "summary": summary, "rows": rows}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")
    for c in CAMS:
        s = summary[c]
        print(f"  {c:14} target 픽셀이 있는 프레임 {s['frames_with_any_px']}/{len(rows)}  "
              f"최대 {s['max_px']}  중앙 {s['median_px']}")


if __name__ == "__main__":
    main()
