"""T1 항목 1 재측정 — grounding 점수와 target 중심 오차의 관계.

사용자 판정: banana 중앙오차 5.9mm 인데 apple/pear 는 34.8/36.0mm — banana 는 grounding
9/15 만 통과(6 프레임 no_target/low_score)했고 apple/pear 는 15/15 전부 통과했다. 가설 둘:

  (a) 점수 문턱이 일을 한다 — banana 에서 걸러진 6 프레임이 원래 오차가 컸다
  (b) 두 에피소드에서 문턱이 헐겁다 — apple/pear 에도 오차 큰 프레임이 있는데 통과했다

`AG3S.ag3s.stages.target_grounding.GroundingResult` 는 실패(`LOW_SCORE`)해도
`best_score` 와 `clusters`(점수 내림차순, `clusters[0]` 이 문턱에 걸린 바로 그 후보)를
그대로 들고 있다 — 그래서 걸러진 프레임도 "만약 그 후보를 썼다면" 오차를 낼 수 있다.
`benchmark/**` 는 건드리지 않는다. `measure_items.py` 와 같은 진입점을 그대로 다시 부른다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1/measure_item1_score.py \
        --records <run dir> --attention <npz> --target <apple|banana|pear> \
        --out outputs/verify/T1/item1_score_<ep>.json
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T1'))

import numpy as np

import measure_items as mi  # noqa: E402


def main() -> None:
    import mujoco

    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.ag3s.experiments.reports.attention_report import target_from_prompt
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import camera_observation
    from AG3S.ag3s.experiments.sources.policy_record import (
        load_run, phase_boundaries_for_path, phase_for, pose_scene, replay_scene)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", required=True)
    ap.add_argument("--step1-json",
                    default="AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--target", default=None)
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_wall0 = time.time()
    run = load_run(args.records)
    target = args.target or target_from_prompt(run.prompt)
    boundaries, _ = phase_boundaries_for_path(args.records)

    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    attention_block = np.asarray(blob["attention"], np.float32)
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    names_in_npz = [str(c) for c in blob["cameras"]]
    attention_camera_index = {mujoco_name: names_in_npz.index(policy_name)
                              for policy_name, mujoco_name in mi.ATTENTION_ALIAS.items()
                              if policy_name in names_in_npz}

    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, target)
    if body_id < 0:
        raise SystemExit(f"no body {target!r}")

    ag_cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"},
    })
    ag = AG3S(ag_cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot)

    frames_out = []
    for i, step in enumerate(run):
        pose_scene(scene, step)
        gt_frames = {c: scene.capture(c) for c in mi.CAMERAS}
        att_by_cam = {cam: attention_block[i, di, ai, cell["layer"], cell["head"], ci]
                      for cam, ci in attention_camera_index.items()}
        observations = []
        for c in mi.CAMERAS:
            obs, _f = camera_observation(scene, c, filter_robot, timestamp=float(i),
                                         attention_map=att_by_cam.get(c))
            observations.append(obs)
        phase = phase_for(step.t_step, boundaries)
        cs, debug = ag.process_multi_debug(observations, phase=phase)
        grounding = debug["grounding"]

        gt_pts = [p for c in mi.CAMERAS
                  if (p := mi.backproject_body(gt_frames[c], body_id)) is not None]
        gt_center = (np.concatenate(gt_pts, axis=0).mean(axis=0) if gt_pts else None)

        best_cluster = grounding.clusters[0] if len(grounding.clusters) else None
        best_cluster_centroid = (np.asarray(best_cluster.centroid, np.float64).tolist()
                                 if best_cluster is not None else None)
        would_be_error_mm = (
            float(np.linalg.norm(np.asarray(best_cluster_centroid) - gt_center) * 1000.0)
            if best_cluster_centroid is not None and gt_center is not None else None)
        actual_error_mm = (
            float(np.linalg.norm(np.asarray(cs.target.centroid, np.float64) - gt_center) * 1000.0)
            if cs.target is not None and gt_center is not None else None)

        frames_out.append({
            "i": i, "t_step": int(step.t_step), "phase": phase,
            "has_target": bool(cs.has_target),
            "status": str(cs.status.value), "grounding_status": str(grounding.status.value),
            "target_score": float(grounding.best_score),
            "n_clusters": len(grounding.clusters),
            "best_cluster_target_score": (float(best_cluster.target_score)
                                          if best_cluster is not None else None),
            "actual_target_center_error_mm": actual_error_mm,
            "would_be_target_center_error_mm_from_best_cluster": would_be_error_mm,
            "gt_target_center_base": gt_center.tolist() if gt_center is not None else None,
        })
        if i % 5 == 0 or i == len(run) - 1:
            print(f"  [{i:3d}] score={grounding.best_score:.4f} status={grounding.status.value:10s} "
                  f"actual_err={actual_error_mm} would_be_err={would_be_error_mm}")

    scene.close()

    out = {"records": str(args.records), "attention": str(args.attention), "target": target,
          "wall_s": time.time() - t_wall0, "frames": frames_out}
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
