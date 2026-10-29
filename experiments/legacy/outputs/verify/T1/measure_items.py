"""T1 측정 1~7 — 실측 policy attention 으로 지각이 낸 값을 MuJoCo 참값과 잰다.

`AG3S/trajopt/experiments/esdf_rollout.py` 의 AG3S 호출부(관측 재생 -> `process_multi`)를
그대로 따라간다. **SQP 는 부르지 않는다** — T1 이 재는 것은 지각이지 궤적최적화가 아니다.
`benchmark/**` 는 건드리지 않는다. 참값은 전부 MuJoCo 자체 세그멘테이션/기하에서 온다
(합성하지 않는다).

각 프레임에서 잰다.

  1. target 중심 오차   — `cs.target.centroid` (관측된 표면점 평균, body 원점이 아니다) vs
                          MuJoCo 세그멘테이션 참값으로 target body 픽셀만 역투영한 점의 평균
  2. 지지면 높이 오차    — `cs.support_surfaces` 중 수평(법선 [0,0,1]에 가장 가까운) 평면의
                          `offset` vs MuJoCo `table_top` geom 의 실제 상면 높이 (base frame)
  3. 카메라 간 표면 정렬 — 세 카메라 각각이 참값 세그멘테이션으로 본 table 표면 점의
                          median z 를 base frame 에서 비교 (최대-최소)
  4. 로봇 마스크 위반    — `ag._robot_mask_for` 로 가린 뒤에도 남는, 참값이 로봇인 픽셀 수.
                          잔여는 `body_ids` 로 어느 body 인지 그대로 귀속된다 (참값 자체이므로
                          "최근접" 탐색이 필요 없다)
  5. 낮은 실제 장애물    — target 이 아닌 과일들의 실제 위치에서 `cs.esdf.distance()` 를 질의.
                          가공되지 않은 값을 그대로 남긴다 (판정하지 않는다)
  6. 입력 부족 처리      — `cs.status` · `cs.validity` · grounding `status` 의 분포
  7. completeness       — 처리된 프레임 수 vs 기록된 관측 프레임 수, attention npz 프레임 수

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1/measure_items.py \
        --records <run dir> --attention <npz> --out outputs/verify/T1/items_<ep>.json
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import itertools
import json
import pathlib
import sys
import time

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np


CAMERAS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
ATTENTION_ALIAS = {"cam_high": "zed_left",
                   "cam_left_wrist": "wrist_cam_l",
                   "cam_right_wrist": "wrist_cam_r"}
FRUITS = ("apple", "banana", "orange", "pear")


def backproject_body(frame, body_id: int):
    """`(N,3)` base frame, or `None` if that body is not visible in this camera."""
    ids = frame.body_ids
    d = np.asarray(frame.depth, np.float64)
    mask = (ids == body_id) & np.isfinite(d) & (d > 0.02) & (d < 3.0)
    if not mask.any():
        return None
    v, u = np.nonzero(mask)
    z = d[v, u]
    K = frame.camera_intrinsics
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u.astype(np.float64) - cx) * z / fx
    y = (v.astype(np.float64) - cy) * z / fy
    pc_cam = np.stack([x, y, z], axis=1)
    T = frame.T_base_cam
    return pc_cam @ np.asarray(T, np.float64)[:3, :3].T + np.asarray(T, np.float64)[:3, 3]


def table_top_corner_world(model, data):
    """table_top geom 의 8 꼭짓점 중 z 가 가장 큰 것 (world). `(3,)`."""
    import mujoco
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
    if g < 0:
        raise KeyError("no geom 'table_top' in model")
    c = np.asarray(data.geom_xpos[g], np.float64)
    rot = np.asarray(data.geom_xmat[g], np.float64).reshape(3, 3)
    half = np.asarray(model.geom_size[g], np.float64)
    best = None
    best_z = -np.inf
    for signs in itertools.product((-1.0, 1.0), repeat=3):
        corner = c + rot @ (half * np.asarray(signs))
        if corner[2] > best_z:
            best_z = corner[2]
            best = corner
    return best


def horizontal_surface_heights(surfaces, cloud_points, *, align_min=0.7):
    """수평(법선 [0,0,1] 에 가까운) 지지면마다 `(surface, 실제 inlier 점의 median z)`.

    평면식 `n . p = d` 에서 `z = d / n_z` 로 되짚으면 평면이 지나는 x=y=0 지점의 높이를
    돌려준다 — 로봇 베이스 원점이지 지지면이 있는 곳이 아니다 (평면에 기울기가 조금이라도
    있으면 이 자리에서 틀린다, 실측: -818 mm 로 어긋남). 그래서 그 지지면이 **실제로 낸
    inlier 점**의 z 중앙값을 쓴다 — 그 지지면을 fit 한 바로 그 cloud, 바로 그 인덱스다.
    """
    out = []
    for s in surfaces:
        n = np.asarray(s.normal, np.float64)
        align = float(n[2])
        if align < align_min:
            continue
        idx = np.asarray(s.point_indices, np.int64)
        if idx.size == 0:
            continue
        z = float(np.median(cloud_points[idx, 2]))
        out.append({"surface_id": int(s.id), "normal_z": align, "point_count": int(s.point_count),
                    "median_z_m": z})
    return out


def main() -> None:
    import mujoco

    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.ag3s.experiments.reports.attention_report import target_from_prompt
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import camera_observation, is_robot_body
    from AG3S.ag3s.experiments.sources.policy_record import (
        load_run, phase_boundaries_for_path, phase_for, pose_scene, replay_scene)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", required=True)
    ap.add_argument("--step1-json",
                    default="AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--target", default=None)
    ap.add_argument("--frames", type=int, default=0, help="0 이면 전부")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_wall0 = time.time()
    run = load_run(args.records, limit=args.frames or None)
    target = args.target or target_from_prompt(run.prompt)
    boundaries, phase_evidence = phase_boundaries_for_path(args.records)

    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    attention_block = np.asarray(blob["attention"], np.float32)
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    attention_camera_index = {}
    names_in_npz = [str(c) for c in blob["cameras"]]
    for policy_name, mujoco_name in ATTENTION_ALIAS.items():
        if policy_name in names_in_npz:
            attention_camera_index[mujoco_name] = names_in_npz.index(policy_name)
    n_attention_frames = int(attention_block.shape[0])

    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)

    body_id = {}
    for name in ("table", "crate") + FRUITS:
        bid = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            body_id[name] = bid
    body_names_all = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
                      for i in range(scene.model.nbody)}
    robot_body_ids = {i for i, n in body_names_all.items() if is_robot_body(n)}

    if target not in body_id:
        raise SystemExit(f"target {target!r} (from prompt {run.prompt!r}) has no matching body; "
                          f"have {sorted(body_id)}")
    other_fruits = [f for f in FRUITS if f in body_id and f != target]

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
        gt_frames = {c: scene.capture(c) for c in CAMERAS}

        att_by_cam = {cam: attention_block[i, di, ai, cell["layer"], cell["head"], ci]
                      for cam, ci in attention_camera_index.items()}

        observations = []
        for c in CAMERAS:
            obs, _frame = camera_observation(scene, c, filter_robot, timestamp=float(i),
                                             attention_map=att_by_cam.get(c))
            observations.append(obs)
        phase = phase_for(step.t_step, boundaries)
        cs, debug = ag.process_multi_debug(observations, phase=phase)
        grounding = debug["grounding"]

        # --- item 1: target center -------------------------------------------------------
        gt_pts = [p for c in CAMERAS
                  if (p := backproject_body(gt_frames[c], body_id[target])) is not None]
        gt_center = (np.concatenate(gt_pts, axis=0).mean(axis=0).tolist()
                    if gt_pts else None)
        pred_center = (np.asarray(cs.target.centroid, np.float64).tolist()
                       if cs.target is not None else None)
        target_center_error_mm = (
            float(np.linalg.norm(np.asarray(pred_center) - np.asarray(gt_center)) * 1000.0)
            if gt_center is not None and pred_center is not None else None)

        # --- item 2: support surface height -----------------------------------------------
        corner_world = table_top_corner_world(scene.model, scene.data)
        T_base_world = np.linalg.inv(scene.body_pose("base"))
        gt_table_top_base_z = float(
            (T_base_world @ np.array([*corner_world, 1.0]))[2])
        cloud = debug["filtered_cloud"]
        horiz = horizontal_surface_heights(cs.support_surfaces, cloud.points)
        # 참 테이블 높이에 가장 가까운 수평 지지면을 "그 테이블"로 본다 (30 cm 문턱 —
        # 그보다 멀면 이 프레임엔 테이블과 맞는 수평면이 없다는 뜻, 바닥만 잡혔을 수 있다).
        matched = min(horiz, key=lambda h: abs(h["median_z_m"] - gt_table_top_base_z),
                     default=None)
        if matched is not None and abs(matched["median_z_m"] - gt_table_top_base_z) > 0.30:
            matched = None
        pred_table_top_base_z = matched["median_z_m"] if matched is not None else None
        support_height_error_mm = (
            float((pred_table_top_base_z - gt_table_top_base_z) * 1000.0)
            if pred_table_top_base_z is not None else None)

        # --- item 3: cross-camera surface alignment -----------------------------------------
        per_cam_table_z = {}
        for c in CAMERAS:
            pts = backproject_body(gt_frames[c], body_id["table"])
            if pts is not None and len(pts) >= 20:
                per_cam_table_z[c] = float(np.median(pts[:, 2]))
        cross_camera_alignment_mm = (
            float((max(per_cam_table_z.values()) - min(per_cam_table_z.values())) * 1000.0)
            if len(per_cam_table_z) >= 2 else None)

        # --- item 4: robot mask violations ---------------------------------------------------
        violations_by_body: dict[str, int] = {}
        n_violations = 0
        for c in CAMERAS:
            frame = gt_frames[c]
            mask_removed = ag._robot_mask_for(frame.depth, frame.camera_intrinsics,
                                              frame.T_base_cam, frame.robot_state)
            if mask_removed is None:
                continue
            d = np.asarray(frame.depth, np.float64)
            valid = np.isfinite(d) & (d > 0.02) & (d < args.range_max)
            gt_robot = np.isin(frame.body_ids, list(robot_body_ids))
            leaked = gt_robot & (~mask_removed) & valid
            n_violations += int(leaked.sum())
            if leaked.any():
                ids, counts = np.unique(frame.body_ids[leaked], return_counts=True)
                for bid, cnt in zip(ids, counts):
                    nm = body_names_all.get(int(bid), f"body_{bid}")
                    violations_by_body[nm] = violations_by_body.get(nm, 0) + int(cnt)

        # --- item 5: low-attention real obstacle preservation --------------------------------
        low_obstacle_esdf_mm = {}
        if cs.esdf is not None:
            for f in other_fruits:
                p = scene.body_position_in_base(f)
                dist = cs.esdf.distance(p[None, :])
                low_obstacle_esdf_mm[f] = float(dist[0] * 1000.0)

        # --- item 6: degraded-input handling ---------------------------------------------------
        status = str(cs.status.value)
        validity = str(cs.validity.value)
        grounding_status = str(grounding.status.value)

        frames_out.append({
            "i": i, "t_step": int(step.t_step), "phase": phase,
            "has_target": bool(cs.has_target),
            "target_center_error_mm": target_center_error_mm,
            "gt_target_center_base": gt_center,
            "pred_target_center_base": pred_center,
            "support_height_error_mm": support_height_error_mm,
            "gt_table_top_base_z_m": gt_table_top_base_z,
            "pred_table_top_base_z_m": pred_table_top_base_z,
            "n_support_surfaces": len(cs.support_surfaces),
            "horizontal_surfaces": horiz,
            "cross_camera_alignment_mm": cross_camera_alignment_mm,
            "per_camera_table_median_z_m": per_cam_table_z,
            "robot_mask_violations": n_violations,
            "robot_mask_violations_by_body": violations_by_body,
            "low_obstacle_esdf_mm": low_obstacle_esdf_mm,
            "status": status, "validity": validity, "grounding_status": grounding_status,
        })
        if i % 5 == 0 or i == len(run) - 1:
            print(f"  [{i:3d}] t={step.t_step:4d} {phase:9s} status={status:9s} "
                  f"validity={validity:9s} target_err={target_center_error_mm} "
                  f"support_err={support_height_error_mm} xcam={cross_camera_alignment_mm} "
                  f"mask_viol={n_violations}")

    scene.close()

    n_processed = len(frames_out)
    n_captured = len(run.steps)
    completeness = {
        "n_captured_observation_frames": n_captured,
        "n_processed_frames": n_processed,
        "n_attention_frames_in_npz": n_attention_frames,
        "n_scene_frames_with_status": sum(1 for f in frames_out if f["status"] is not None),
        "n_scene_frames_with_grounding": sum(1 for f in frames_out
                                             if f["grounding_status"] is not None),
    }

    status_counts = {}
    validity_counts = {}
    grounding_counts = {}
    for f in frames_out:
        status_counts[f["status"]] = status_counts.get(f["status"], 0) + 1
        validity_counts[f["validity"]] = validity_counts.get(f["validity"], 0) + 1
        grounding_counts[f["grounding_status"]] = grounding_counts.get(f["grounding_status"], 0) + 1

    def agg(key):
        vals = [f[key] for f in frames_out if f[key] is not None]
        return {"n": len(vals), "mean": float(np.mean(vals)) if vals else None,
                "max_abs": float(np.max(np.abs(vals))) if vals else None,
                "median": float(np.median(vals)) if vals else None}

    out = {
        "records": str(args.records), "attention": str(args.attention),
        "prompt": run.prompt, "target": target, "other_fruits": other_fruits,
        "phase_boundaries": phase_evidence,
        "attention_wiring": sorted(attention_camera_index),
        "wall_s": time.time() - t_wall0,
        "frames": frames_out,
        "summary": {
            "item1_target_center_error_mm": agg("target_center_error_mm"),
            "item2_support_height_error_mm": agg("support_height_error_mm"),
            "item3_cross_camera_alignment_mm": agg("cross_camera_alignment_mm"),
            "item4_robot_mask_violations_total": sum(f["robot_mask_violations"] for f in frames_out),
            "item6_status_counts": status_counts,
            "item6_validity_counts": validity_counts,
            "item6_grounding_status_counts": grounding_counts,
            "item7_completeness": completeness,
        },
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {out_path}")
    print(json.dumps(out["summary"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
