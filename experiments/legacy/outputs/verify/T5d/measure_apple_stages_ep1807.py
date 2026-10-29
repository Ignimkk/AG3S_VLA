"""T5d — apple point 이 파이프라인의 어느 단계에서 사라지나. camera 별 · frame 별 전수.

`attention_flow_ep1807.json` 이 seq 16..32 에서 `apple_cloud_points = 0` 을 냈다. 그 0 이
어디서 생기는지를 단계별 개수로만 낸다 (판정하지 않는다). `process_observation` 의 앞 두
단계를 같은 config 로 그대로 다시 부른다.

  n_gt_pixel        MuJoCo segmentation 이 apple 이라고 한 pixel 수
  n_depth_valid     그중 depth 가 유효한 것 (range_min/range_max 적용 전)
  n_in_range        그중 `pointcloud.range_max` 안에 든 것
  n_after_recon     `reconstruct`(back-project + voxel downsample + cap) 뒤 남은 apple point
  n_after_filter    `filter_robot_points`(self_filter_inflation 0.05 m) 뒤 남은 apple point

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5d/measure_apple_stages_ep1807.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys
import time

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T1'))

import numpy as np  # noqa: E402

import measure_items as mi  # noqa: E402

RECORDS = "outputs/live_test/20260925_ep1807/run_0000"
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5d/apple_stages_ep1807.json'))
N_FRAMES = 51  # seq 1..51 = t_step 0..400


def main() -> None:
    import mujoco

    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.experiments.reports.grounding_report import build_robot_model
    from AG3S.ag3s.experiments.sources.mujoco_source import camera_observation
    from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene
    from AG3S.ag3s.stages.reconstruction import reconstruct
    from AG3S.ag3s.stages.robot_filter import filter_robot_points, robot_sphere_mask

    t0 = time.time()
    run = load_run(RECORDS)
    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    apple = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "apple")

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"},
    })
    pc = cfg.pointcloud

    frames = []
    for i, step in enumerate(run):
        if i >= N_FRAMES:
            break
        pose_scene(scene, step)
        per_cam = {}
        for c in mi.CAMERAS:
            obs, frame = camera_observation(scene, c, filter_robot, timestamp=float(i))
            ids = np.asarray(frame.body_ids)
            depth = np.asarray(frame.depth, np.float64)
            m_apple = ids == apple
            n_gt = int(m_apple.sum())
            valid = m_apple & np.isfinite(depth) & (depth > 0.0)
            n_valid = int(valid.sum())
            rmax = float(pc.range_max) if pc.range_max else np.inf
            in_range = valid & (depth <= rmax)
            n_range = int(in_range.sum())

            T_base_cam = obs.resolve_T_base_cam(filter_robot)
            cloud, _rs = reconstruct(depth=obs.depth, camera_intrinsics=obs.camera_intrinsics,
                                     T_base_cam=T_base_cam, config=pc, frame_id=cfg.frame_id)
            if cloud.uv is None:
                raise SystemExit("reconstruct dropped uv; cannot attribute points to bodies")
            uv = np.asarray(cloud.uv, np.int64)
            u = np.clip(uv[:, 0], 0, ids.shape[1] - 1)
            v = np.clip(uv[:, 1], 0, ids.shape[0] - 1)
            is_apple = ids[v, u] == apple
            n_recon = int(is_apple.sum())

            filtered, fstats = filter_robot_points(cloud, filter_robot, obs.robot_state, pc)
            fuv = np.asarray(filtered.uv, np.int64)
            fu = np.clip(fuv[:, 0], 0, ids.shape[1] - 1)
            fv = np.clip(fuv[:, 1], 0, ids.shape[0] - 1)
            n_filt = int((ids[fv, fu] == apple).sum())

            # 지워진 apple point 가 어느 로봇 구 안에 들어갔나 — inflation 없이도 들어가나
            centers, radii = filter_robot.sphere_centers_numeric(
                np.asarray(obs.robot_state, np.float64).reshape(-1))
            apple_pts = cloud.points[is_apple]
            n_inside_inflated = n_inside_bare = 0
            min_signed_mm = None
            if apple_pts.size:
                n_inside_inflated = int(robot_sphere_mask(
                    apple_pts, centers, radii, pc.self_filter_inflation).sum())
                n_inside_bare = int(robot_sphere_mask(apple_pts, centers, radii, 0.0).sum())
                dd = np.linalg.norm(apple_pts[:, None, :] - centers[None, :, :], axis=2) \
                    - np.asarray(radii, np.float64)[None, :]
                min_signed_mm = float(dd.min(axis=1).min() * 1000.0)

            per_cam[c] = {
                "n_gt_pixel": n_gt,
                "n_depth_valid": n_valid,
                "n_in_range": n_range,
                "n_after_recon": n_recon,
                "n_after_filter": n_filt,
                "n_removed_by_self_filter": n_recon - n_filt,
                "n_apple_inside_inflated_spheres": n_inside_inflated,
                "n_apple_inside_bare_spheres": n_inside_bare,
                "min_apple_to_sphere_surface_mm": min_signed_mm,
                "cloud_n_total": int(len(cloud)),
                "cloud_n_after_filter": int(len(filtered)),
                "self_filter_removed_total": int(fstats["n_removed"]),
            }
        frames.append({"seq": i + 1, "t_step": int(step.t_step), "per_camera": per_cam,
                       "n_after_filter_total": sum(v["n_after_filter"] for v in per_cam.values()),
                       "n_after_recon_total": sum(v["n_after_recon"] for v in per_cam.values())})
        f = frames[-1]
        print(f"  seq {f['seq']:3d} t={f['t_step']:4d} recon={f['n_after_recon_total']:6d} "
              f"filt={f['n_after_filter_total']:6d}  "
              + " ".join(f"{c}:{v['n_gt_pixel']}/{v['n_in_range']}/{v['n_after_recon']}/"
                         f"{v['n_after_filter']}" for c, v in f["per_camera"].items()), flush=True)

    scene.close()
    OUT.write_text(json.dumps({
        "records": RECORDS,
        "n_frames": len(frames),
        "cameras": list(mi.CAMERAS),
        "pointcloud_config": {"voxel_size": pc.voxel_size, "range_max": pc.range_max,
                              "range_min": getattr(pc, "range_min", None),
                              "max_points": pc.max_points,
                              "self_filter": pc.self_filter,
                              "self_filter_inflation": pc.self_filter_inflation,
                              "cap_strategy": pc.cap_strategy},
        "n_filter_spheres": int(len(filter_robot.sphere_radii())
                                if hasattr(filter_robot, "sphere_radii") else -1),
        "stages": ["n_gt_pixel", "n_depth_valid", "n_in_range", "n_after_recon", "n_after_filter"],
        "wall_s": time.time() - t0,
        "frames": frames,
    }, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
