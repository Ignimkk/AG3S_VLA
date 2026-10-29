"""T5c — ep1807 오프라인 grounding 점수. 75 프레임 전수.

live 서버 경로(shadow 실행)는 `target_score` 를 응답에도 로그에도 싣지 않는다. 같은
에피소드의 **오프라인 재생 경로**에서 같은 grounding 단계를 돌려 점수의 실제 값을 낸다.
`benchmark/**` 는 건드리지 않는다 — `outputs/verify/T1/measure_item1_score.py` 와 같은
진입점(`AG3S.process_multi_debug`)을 그대로 부르고, 프레임마다 다음을 더 남긴다:

  · 카메라별로 사과가 몇 픽셀 보이는가 (MuJoCo 참값 segmentation)
  · 왼팔 관절값, ee_finger_l1 ↔ apple 거리 (참값)

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5c/measure_offline_grounding_ep1807.py
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
ATTENTION = "AG3S/ag3s/asset/data/attention_16d_ep1807.npz"
STEP1 = "AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json"
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/offline_grounding_ep1807.json'))


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

    t0 = time.time()
    run = load_run(RECORDS)
    target = target_from_prompt(run.prompt)
    boundaries, _ = phase_boundaries_for_path(RECORDS)

    blob = np.load(ATTENTION, allow_pickle=False)
    cell = json.loads(pathlib.Path(STEP1).read_text())["best"]
    attention_block = np.asarray(blob["attention"], np.float32)
    di = [int(d) for d in blob["denoise_steps"]].index(int(cell["denoise"]))
    ai = [str(a) for a in blob["aggregations"]].index(str(cell["agg"]))
    names_in_npz = [str(c) for c in blob["cameras"]]
    cam_index = {mj: names_in_npz.index(pol) for pol, mj in mi.ATTENTION_ALIAS.items()
                 if pol in names_in_npz}
    n_steps = len(run)
    if attention_block.shape[0] < n_steps:
        raise SystemExit(f"attention has {attention_block.shape[0]} frames, run has {n_steps}")

    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    bid = {n: mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in ("apple", "crate", "ee_finger_l1")}
    if min(bid.values()) < 0:
        raise SystemExit(f"missing bodies: {bid}")

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"},
    })
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot)
    threshold = float(cfg.clustering.target_score_threshold)

    frames = []
    for i, step in enumerate(run):
        pose_scene(scene, step)
        gt = {c: scene.capture(c) for c in mi.CAMERAS}
        att = {cam: attention_block[i, di, ai, cell["layer"], cell["head"], ci]
               for cam, ci in cam_index.items()}
        observations = []
        for c in mi.CAMERAS:
            obs, _f = camera_observation(scene, c, filter_robot, timestamp=float(i),
                                         attention_map=att.get(c))
            observations.append(obs)
        phase = phase_for(step.t_step, boundaries)
        cs, debug = ag.process_multi_debug(observations, phase=phase)
        g = debug["grounding"]

        apple_px, apple_pts = {}, []
        for c in mi.CAMERAS:
            p = mi.backproject_body(gt[c], bid["apple"])
            apple_px[c] = 0 if p is None else int(len(p))
            if p is not None:
                apple_pts.append(p)
        gt_center = (np.concatenate(apple_pts, axis=0).mean(axis=0) if apple_pts else None)

        d = scene.data
        apple_w = np.asarray(d.xpos[bid["apple"]], np.float64)
        ee_w = np.asarray(d.xpos[bid["ee_finger_l1"]], np.float64)
        crate_w = np.asarray(d.xpos[bid["crate"]], np.float64)
        qpos = np.asarray(step.qpos, np.float64)

        best = g.clusters[0] if len(g.clusters) else None
        err_mm = (float(np.linalg.norm(np.asarray(cs.target.centroid, np.float64) - gt_center)
                        * 1000.0)
                  if cs.target is not None and gt_center is not None else None)
        frames.append({
            "i": i, "t_step": int(step.t_step), "phase": phase,
            "has_target": bool(cs.has_target),
            "status": str(cs.status.value),
            "grounding_status": str(g.status.value),
            "target_score": float(g.best_score),
            "n_clusters": int(len(g.clusters)),
            "cluster_scores_top3": [float(c.target_score) for c in g.clusters[:3]],
            "best_cluster_point_count": (int(best.point_count) if best is not None else None),
            "target_center_error_mm": err_mm,
            "apple_pixels": apple_px,
            "apple_pixels_total": int(sum(apple_px.values())),
            "apple_world": apple_w.tolist(),
            "ee_finger_l1_world": ee_w.tolist(),
            "ee_apple_dist_mm": float(np.linalg.norm(ee_w - apple_w) * 1000.0),
            "apple_crate_xy_dist_mm": float(np.linalg.norm((apple_w - crate_w)[:2]) * 1000.0),
            "apple_z_m": float(apple_w[2]),
            "qpos_len": int(qpos.size),
        })
        if i % 5 == 0 or i == n_steps - 1:
            f = frames[-1]
            print(f"  [{i:3d}] t={f['t_step']:4d} score={f['target_score']:.4f} "
                  f"{f['grounding_status']:10s} px={f['apple_pixels_total']:5d} "
                  f"d(ee,apple)={f['ee_apple_dist_mm']:7.1f}mm", flush=True)

    scene.close()
    scores = [f["target_score"] for f in frames]
    ht = [f for f in frames if f["has_target"]]
    nt = [f for f in frames if not f["has_target"]]

    def summ(xs):
        xs = sorted(xs)
        if not xs:
            return None
        return {"n": len(xs), "min": xs[0], "p50": xs[len(xs) // 2], "max": xs[-1],
                "mean": float(np.mean(xs))}

    out = {
        "records": RECORDS, "attention": ATTENTION, "target": target,
        "attention_cell": cell, "cameras": list(mi.CAMERAS),
        "target_score_threshold": threshold,
        "esdf_backend": "legacy", "voxel_m": 0.020,
        "wall_s": time.time() - t0,
        "n_frames": len(frames),
        "has_target_true": len(ht),
        "grounding_status_counts": {k: sum(1 for f in frames if f["grounding_status"] == k)
                                    for k in sorted({f["grounding_status"] for f in frames})},
        "status_counts": {k: sum(1 for f in frames if f["status"] == k)
                          for k in sorted({f["status"] for f in frames})},
        "target_score": {
            "all": summ(scores),
            "has_target": summ([f["target_score"] for f in ht]),
            "no_target": summ([f["target_score"] for f in nt]),
            "n_within_0p05_below_threshold": sum(
                1 for f in nt if threshold - 0.05 <= f["target_score"] < threshold),
            "n_below_half_threshold": sum(1 for f in nt
                                          if f["target_score"] < threshold / 2.0),
            "n_zero": sum(1 for f in nt if f["target_score"] == 0.0),
        },
        "ee_apple_dist_mm": {"has_target": summ([f["ee_apple_dist_mm"] for f in ht]),
                             "no_target": summ([f["ee_apple_dist_mm"] for f in nt])},
        "apple_pixels_total": {"has_target": summ([f["apple_pixels_total"] for f in ht]),
                               "no_target": summ([f["apple_pixels_total"] for f in nt])},
        "apple_pixels_per_camera_mean": {
            c: {"has_target": (float(np.mean([f["apple_pixels"][c] for f in ht])) if ht else None),
                "no_target": (float(np.mean([f["apple_pixels"][c] for f in nt])) if nt else None)}
            for c in mi.CAMERAS},
        "frames": frames,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    print("has_target", len(ht), "/", len(frames), "threshold", threshold)
    print("score summary", json.dumps(out["target_score"], ensure_ascii=False))


if __name__ == "__main__":
    main()
