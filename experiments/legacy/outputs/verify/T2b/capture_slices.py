"""T2-b figure 1(실제 씬)용 재료 — 선택 프레임의 ESDF 수평 단면 + 그 프레임의 실제 렌더 이미지.

`measure_t2b.py` 와 같은 배선/같은 config 를 쓴다 (대조 파이프라인만 없다). TSDF 는 누적이므로
프레임 0 부터 순서대로 돌리고, 지정한 프레임에서만 단면을 뜬다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T2b/capture_slices.py --out outputs/verify/T2b/slices.npz
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T1'))

import numpy as np

import measure_items as mi  # noqa: E402

LEFT_GRIPPER_STATE_IDX = 7


def render_rgb(scene, camera: str) -> np.ndarray:
    """`TransportScene` 에 RGB 진입점이 없어(capture 는 depth+segmentation 만) 렌더러를 직접 쓴다."""
    r = scene._renderer_for()
    r.disable_depth_rendering()
    r.update_scene(scene.data, camera=camera)
    return np.asarray(r.render(), np.uint8)
CRATE_INNER_X, CRATE_INNER_Y = 0.082, 0.142
CRATE_RIM_LOCAL_Z, CRATE_BELOW_RIM_MARGIN = 0.062, 0.010


def main() -> None:
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.ag3s.constraints.attached import attached_points_in_base
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import camera_observation
    from AG3S.ag3s.experiments.sources.policy_record import (
        load_run, phase_boundaries_for_path, phase_for, pose_scene, replay_scene)
    from AG3S.trajopt.grasp_latch import CentroidIdentity, GraspLatch
    from AG3S.trajopt.safe_policy import DEFAULT_GRASP_LINKS, grasp_parent_links

    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="outputs/live_test/20260925_ep1807/run_0000")
    ap.add_argument("--attention", default="AG3S/ag3s/asset/data/attention_16d_ep1807.npz")
    ap.add_argument("--step1-json",
                    default="AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--slice-frames", default="0,19,20,25,47,74")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    want = {int(v) for v in args.slice_frames.split(",")}

    run = load_run(args.records)
    boundaries, _ev = phase_boundaries_for_path(args.records)
    blob = np.load(args.attention, allow_pickle=False)
    cell = json.loads(pathlib.Path(args.step1_json).read_text())["best"]
    block = np.asarray(blob["attention"], np.float32)
    di = [int(x) for x in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(x) for x in blob["aggregations"]].index(str(cell["agg"]))
    names = [str(c) for c in blob["cameras"]]
    cam_index = {mj: names.index(pol) for pol, mj in mi.ATTENTION_ALIAS.items() if pol in names}

    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf", "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"}})
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot,
              attached_parent_links=grasp_parent_links(DEFAULT_GRASP_LINKS))
    identity, latch = CentroidIdentity(), GraspLatch()
    destination_points, latched_target = None, None

    first = next(iter(run))
    pose_scene(scene, first)
    probe = {"apple_frame0": np.asarray(scene.body_position_in_base("apple"), np.float64),
             "pear_static": np.asarray(scene.body_position_in_base("pear"), np.float64)}

    out: dict = {}
    for i, step in enumerate(run):
        pose_scene(scene, step)
        att_by_cam = {c: block[i, di, ai, cell["layer"], cell["head"], ci]
                      for c, ci in cam_index.items()}
        observations = []
        for c in mi.CAMERAS:
            obs, _f = camera_observation(scene, c, filter_robot, timestamp=float(i),
                                         attention_map=att_by_cam.get(c))
            observations.append(obs)
        phase = phase_for(step.t_step, boundaries)
        cs, _dbg = ag.process_multi_debug(observations, phase=phase,
                                          destination_points=destination_points)
        target = cs.target
        label = identity.label(None if target is None else target.centroid)
        if target is not None and label is not None and label == latch.manipulated:
            latched_target = target
        gripper = float(step.state[LEFT_GRIPPER_STATE_IDX])
        placed = False
        if latch.holding and ag.attached is not None:
            T = np.linalg.inv(scene.body_pose("crate")) @ scene.body_pose("apple")
            placed = bool(abs(T[0, 3]) <= CRATE_INNER_X and abs(T[1, 3]) <= CRATE_INNER_Y
                          and T[2, 3] <= CRATE_RIM_LOCAL_Z - CRATE_BELOW_RIM_MARGIN)
        score = float(getattr(target, "confidence", 0.0) or 0.0)
        event = latch.update(label=label, score=score, runner_up=0.0,
                             gripper=gripper, placed=placed)
        if event.attach and latched_target is not None:
            q = max(observations, key=lambda o: o.timestamp).robot_state
            ag.attach(latched_target, robot_state=np.asarray(q, np.float64),
                      parent_link="ee_finger_l1",
                      allowed_contact_links=("ee_finger_l1", "ee_finger_l2"), label="manipulated")
        elif event.detach:
            ag.detach()
            destination_points = None
        if event.destination is not None and label == event.destination and target is not None:
            destination_points = np.asarray(target.points, np.float64)

        if i in want and cs.esdf is not None:
            g = cs.esdf.grid
            kz = int(round((probe["apple_frame0"][2] - g.origin[0 * 0 + 2]) / g.voxel_size))
            kz = int(np.clip(kz, 0, g.shape[2] - 1))
            # x, y 창: 두 탐침을 모두 담는 범위 + 10 cm 여유
            xs = np.array([probe["apple_frame0"][0], probe["pear_static"][0]])
            ys = np.array([probe["apple_frame0"][1], probe["pear_static"][1]])
            lo = np.array([xs.min() - 0.18, ys.min() - 0.18])
            hi = np.array([xs.max() + 0.18, ys.max() + 0.18])
            ix0 = int(np.clip(np.floor((lo[0] - g.origin[0]) / g.voxel_size), 0, g.shape[0] - 1))
            ix1 = int(np.clip(np.ceil((hi[0] - g.origin[0]) / g.voxel_size), 0, g.shape[0] - 1))
            iy0 = int(np.clip(np.floor((lo[1] - g.origin[1]) / g.voxel_size), 0, g.shape[1] - 1))
            iy1 = int(np.clip(np.ceil((hi[1] - g.origin[1]) / g.voxel_size), 0, g.shape[1] - 1))
            sl = np.array(cs.esdf.distance_grid[ix0:ix1 + 1, iy0:iy1 + 1, kz], np.float32)
            out[f"slice_{i}"] = sl
            out[f"extent_{i}"] = np.array([
                g.origin[0] + ix0 * g.voxel_size - 0.5 * g.voxel_size,
                g.origin[0] + ix1 * g.voxel_size + 0.5 * g.voxel_size,
                g.origin[1] + iy0 * g.voxel_size - 0.5 * g.voxel_size,
                g.origin[1] + iy1 * g.voxel_size + 0.5 * g.voxel_size], np.float64)
            out[f"slice_z_{i}"] = np.array([g.origin[2] + kz * g.voxel_size], np.float64)
            out[f"rgb_{i}"] = render_rgb(scene, "zed_left")
            out[f"hand_{i}"] = np.asarray(scene.body_position_in_base("ee_finger_l1"), np.float64)
            out[f"apple_{i}"] = np.asarray(scene.body_position_in_base("apple"), np.float64)
            att = ag.attached
            if att is not None:
                p = attached_points_in_base(
                    att, robot_model=constraint_robot,
                    robot_state=np.asarray(max(observations, key=lambda o: o.timestamp).robot_state,
                                           np.float64))
                out[f"attached_pts_{i}"] = np.asarray(p, np.float64)
            print(f"  captured frame {i} slice {sl.shape} z={out[f'slice_z_{i}'][0]:.3f}")
        if i % 10 == 0:
            print(f"  [{i:3d}] phase={latch.phase.value}")

    scene.close()
    for k, v in probe.items():
        out[f"probe_{k}"] = v
    np.savez_compressed(args.out, **out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
