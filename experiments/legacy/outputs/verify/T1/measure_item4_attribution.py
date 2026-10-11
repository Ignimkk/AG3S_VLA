"""T1 항목 4 귀속 한 겹 더 — A0 판정 2 건.

`benchmark/**` 는 건드리지 않는다. `measure_items.py` 와 같은 진입점을 그대로 다시 부른다.

1. **카메라별로 갈라 본다.** `EE_BODY_L`/`EE_BODY_R` 누수가 자기 손목 카메라(`wrist_cam_l`/
   `wrist_cam_r`)에만 몰려 있는지, 아니면 `zed_left`·반대쪽 손목에도 퍼져 있는지.
2. **그 픽셀이 실제로 장애물이 되는지.** 누수 픽셀을 역투영해 `cs.esdf.distance()` 를 질의한다.
   음수(또는 0에 가까움)면 뒤 단계(ESDF)에서 점유로 잡힌 것 — "마스크 누수" 가 아니라
   "마스크 뒤에서 잡힌다". 양수로 크면 진짜로 빠져나간 것.

고치지 않는다. 수치만 낸다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T1/measure_item4_attribution.py \
        --records <run dir> --attention <npz> --out outputs/verify/T1/item4_attr_<ep>.json
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np

sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T1'))
import measure_items as mi  # noqa: E402


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
    body_names_all = {i: (mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, i) or "")
                      for i in range(scene.model.nbody)}
    robot_body_ids = {i for i, n in body_names_all.items() if is_robot_body(n)}

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

        per_camera = {}
        for c in mi.CAMERAS:
            frame = gt_frames[c]
            mask_removed = ag._robot_mask_for(frame.depth, frame.camera_intrinsics,
                                              frame.T_base_cam, frame.robot_state)
            d = np.asarray(frame.depth, np.float64)
            valid = np.isfinite(d) & (d > 0.02) & (d < args.range_max)
            gt_robot = np.isin(frame.body_ids, list(robot_body_ids))
            leaked = gt_robot & (~mask_removed if mask_removed is not None else True) & valid

            by_body = {}
            if leaked.any():
                ids, counts = np.unique(frame.body_ids[leaked], return_counts=True)
                for bid, cnt in zip(ids, counts):
                    nm = body_names_all.get(int(bid), f"body_{bid}")
                    by_body[nm] = int(cnt)

            # --- esdf 질의: 이 카메라에서 누수된 픽셀이 뒤에서 점유로 잡히나 -----------------
            esdf_stats_by_body = {}
            if cs.esdf is not None:
                for nm in by_body:
                    body_mask = leaked & (frame.body_ids ==
                                          next(k for k, v in body_names_all.items() if v == nm))
                    v_idx, u_idx = np.nonzero(body_mask)
                    z = d[v_idx, u_idx]
                    K = frame.camera_intrinsics
                    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                    x = (u_idx.astype(np.float64) - cx) * z / fx
                    y = (v_idx.astype(np.float64) - cy) * z / fy
                    pc_cam = np.stack([x, y, z], axis=1)
                    T = np.asarray(frame.T_base_cam, np.float64)
                    pts = pc_cam @ T[:3, :3].T + T[:3, 3]
                    dist = cs.esdf.distance(pts) if len(pts) else np.zeros(0)
                    if len(dist):
                        esdf_stats_by_body[nm] = {
                            "n_points": int(len(dist)),
                            "mean_mm": float(np.mean(dist) * 1000.0),
                            "median_mm": float(np.median(dist) * 1000.0),
                            "min_mm": float(np.min(dist) * 1000.0),
                            "max_mm": float(np.max(dist) * 1000.0),
                            "frac_occupied_le0": float(np.mean(dist <= 0.0)),
                            "frac_near_le_voxel": float(np.mean(dist <= args.voxel)),
                            "frac_clearly_free_gt_50mm": float(np.mean(dist > 0.05)),
                        }

            per_camera[c] = {"n_leaked": int(leaked.sum()), "by_body": by_body,
                            "esdf_at_leaked": esdf_stats_by_body}

        frames_out.append({"i": i, "t_step": int(step.t_step), "phase": phase,
                          "per_camera": per_camera})
        if i % 5 == 0 or i == len(run) - 1:
            print(f"  [{i:3d}] " + " | ".join(
                f"{c}: n={per_camera[c]['n_leaked']} {per_camera[c]['by_body']}"
                for c in mi.CAMERAS))

    scene.close()

    # --- 집계 ------------------------------------------------------------------------------
    agg_by_camera_body: dict[str, dict[str, int]] = {}
    agg_esdf_by_body: dict[str, list] = {}
    for f in frames_out:
        for c, pc in f["per_camera"].items():
            agg_by_camera_body.setdefault(c, {})
            for nm, cnt in pc["by_body"].items():
                agg_by_camera_body[c][nm] = agg_by_camera_body[c].get(nm, 0) + cnt
            for nm, stats in pc["esdf_at_leaked"].items():
                agg_esdf_by_body.setdefault(nm, []).append(stats)

    def pool_esdf(nm):
        entries = agg_esdf_by_body.get(nm, [])
        if not entries:
            return None
        n_total = sum(e["n_points"] for e in entries)
        # 가중 평균 (frame·camera 별 n_points 로 가중)
        w_frac_occ = sum(e["frac_occupied_le0"] * e["n_points"] for e in entries) / n_total
        w_frac_near = sum(e["frac_near_le_voxel"] * e["n_points"] for e in entries) / n_total
        w_frac_free = sum(e["frac_clearly_free_gt_50mm"] * e["n_points"] for e in entries) / n_total
        w_mean = sum(e["mean_mm"] * e["n_points"] for e in entries) / n_total
        return {"n_points_total": n_total, "mean_mm_weighted": w_mean,
               "frac_occupied_le0_weighted": w_frac_occ,
               "frac_near_le_voxel_weighted": w_frac_near,
               "frac_clearly_free_gt_50mm_weighted": w_frac_free}

    summary = {
        "leaked_px_by_camera_by_body": agg_by_camera_body,
        "esdf_at_leaked_pooled_by_body": {nm: pool_esdf(nm) for nm in agg_esdf_by_body},
    }

    out = {"records": str(args.records), "attention": str(args.attention), "target": target,
          "wall_s": time.time() - t_wall0, "frames": frames_out, "summary": summary}
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {out_path}")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
