"""T5d — ep1807 offline replay: frame 별 **모든 cluster** 의 score 와 MuJoCo 참값 이름.

`outputs/verify/T5c/measure_offline_grounding_ep1807.py` 를 넓힌 것이다. T5c 는 1 등 score 만
냈다. 여기서는 frame 마다

  · 모든 cluster 의 `target_score` · `point_count` · centroid
  · 각 cluster 의 **참값 이름** — cluster member point 를 fusion provenance
    (`FusedPointCloud.obs_camera` / `obs_uv` / `representative`) 로 원래 pixel 까지 되짚고
    그 pixel 의 MuJoCo segmentation body 를 읽어 다수결한다. "가장 가까운 물체" 추정이 아니라
    그 point 가 실제로 무엇의 표면인지를 그대로 쓴다. 비교용으로 centroid 최근접 body 도 남긴다.
  · apple cluster 의 rank 와 score, apple point 가 cloud 에 몇 개 남았나 (camera 별)
  · attention peak pixel 이 앉은 참값 body
  · `GraspLatch` 를 같은 입력으로 frame 순서대로 돌린 `phase` / `manipulated`
  · gripper 관측값 · apple 참값 z · ee↔apple · apple↔crate xy

를 남긴다. `benchmark/**` 는 한 줄도 고치지 않는다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5d/measure_attention_flow_ep1807.py
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
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5d/attention_flow_ep1807.json'))

#: 참값 이름으로 접는 규칙. 이 여덟 개 밖의 body 는 전부 `unknown`.
OBJECT_BODIES = ("apple", "banana", "orange", "pear", "crate", "table")


def build_body_group(model) -> dict[int, str]:
    """body id -> 여덟 이름 중 하나. 로봇 링크는 전부 `robot`."""
    import mujoco

    names = {b: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}")
             for b in range(model.nbody)}
    # 로봇은 `base` body 의 후손 전부 + IK target marker body 들.
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    robot = set()
    if base >= 0:
        robot.add(base)
        for b in range(model.nbody):
            p, seen = b, set()
            while p > 0 and p not in seen:
                seen.add(p)
                if p == base:
                    robot.add(b)
                    break
                p = int(model.body_parentid[p])
    for b, n in names.items():
        if n.endswith("_target") or n == "com_target":
            robot.add(b)
    group = {}
    for b, n in names.items():
        if n in OBJECT_BODIES:
            group[b] = n
        elif b in robot:
            group[b] = "robot"
        else:
            group[b] = "unknown"
    return group, names


def main() -> None:
    import mujoco

    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.ag3s.experiments.reports.attention_report import target_from_prompt
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import camera_observation
    from AG3S.ag3s.experiments.sources.policy_record import (
        gripper_columns_for_state, load_run, phase_boundaries_for_path, phase_for, pose_scene,
        replay_scene)
    from AG3S.trajopt.grasp_latch import CentroidIdentity, GraspLatch, LatchConfig

    t0 = time.time()
    run = load_run(RECORDS)
    target_word = target_from_prompt(run.prompt)
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
    group, body_names = build_body_group(scene.model)
    bid = {n: mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in ("apple", "banana", "orange", "pear", "crate", "table", "ee_finger_l1")}
    if min(bid.values()) < 0:
        raise SystemExit(f"missing bodies: {bid}")
    # centroid 최근접용 후보: 물체 여섯 + 로봇 링크 전부
    robot_bodies = [b for b, g in group.items() if g == "robot"]

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"},
    })
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot)
    threshold = float(cfg.clustering.target_score_threshold)

    latch_cfg = LatchConfig()
    latch = GraspLatch(latch_cfg)
    identity = CentroidIdentity()

    state_width = int(np.asarray(run.steps[0].state).reshape(-1).size)
    gcol_l, gcol_r = gripper_columns_for_state(state_width)

    frames = []
    for i, step in enumerate(run):
        pose_scene(scene, step)
        gt = {c: scene.capture(c) for c in mi.CAMERAS}
        att = {cam: attention_block[i, di, ai, cell["layer"], cell["head"], ci]
               for cam, ci in cam_index.items()}
        observations = []
        obs_cam_name = {}
        for c in mi.CAMERAS:
            obs, _f = camera_observation(scene, c, filter_robot, timestamp=float(i),
                                         attention_map=att.get(c))
            observations.append(obs)
            # `CameraObservation.camera_id` 는 CameraID enum(head/wrist_left/...) 이고
            # MuJoCo 카메라 이름과 다르다. fusion 의 cameras 튜플은 이 id 를 담으므로
            # 되짚을 때 쓸 사전을 여기서 만든다.
            obs_cam_name[obs.camera_id] = c
        phase = phase_for(step.t_step, boundaries)
        cs, debug = ag.process_multi_debug(observations, phase=phase)
        g = debug["grounding"]
        fusion = debug["fusion"]
        fc = fusion.cloud

        # ---- fused point 별 참값 이름 (provenance -> pixel -> segmentation) -------------
        cams = [obs_cam_name[cid] for cid in fc.cameras]
        if len(cams) != len(set(cams)):
            raise SystemExit(f"camera_id -> mujoco name mapping is not 1:1: {fc.cameras}")
        rep = np.asarray(fc.representative, np.int64)
        rep_cam = np.asarray(fc.obs_camera, np.int16)[rep]
        rep_uv = np.asarray(fc.obs_uv, np.int32)[rep]
        point_body = np.full(len(fc), -1, np.int64)
        for ci, cname in enumerate(cams):
            sel = np.nonzero(rep_cam == ci)[0]
            if not sel.size:
                continue
            ids = np.asarray(gt[cname].body_ids)
            u = np.clip(rep_uv[sel, 0], 0, ids.shape[1] - 1)
            v = np.clip(rep_uv[sel, 1], 0, ids.shape[0] - 1)
            point_body[sel] = ids[v, u]
        point_group = np.array([group.get(int(b), "unknown") for b in point_body], dtype=object)

        # 모든 observation(대표점만이 아닌 전부)에 대한 apple 생존 수 — camera 별
        all_cam = np.asarray(fc.obs_camera, np.int16)
        all_uv = np.asarray(fc.obs_uv, np.int32)
        apple_obs_per_cam = {}
        for ci, cname in enumerate(cams):
            sel = np.nonzero(all_cam == ci)[0]
            if not sel.size:
                apple_obs_per_cam[cname] = 0
                continue
            ids = np.asarray(gt[cname].body_ids)
            u = np.clip(all_uv[sel, 0], 0, ids.shape[1] - 1)
            v = np.clip(all_uv[sel, 1], 0, ids.shape[0] - 1)
            apple_obs_per_cam[cname] = int(np.count_nonzero(ids[v, u] == bid["apple"]))

        d = scene.data
        world = {n: np.asarray(d.xpos[bid[n]], np.float64) for n in bid}
        cand_centers = [(n, world[n]) for n in OBJECT_BODIES]
        robot_xpos = np.asarray([d.xpos[b] for b in robot_bodies], np.float64)

        def nearest_name(c: np.ndarray) -> tuple[str, float]:
            best_n, best_d = "unknown", float("inf")
            for n, p in cand_centers:
                dd = float(np.linalg.norm(c - p))
                if dd < best_d:
                    best_n, best_d = n, dd
            dd = float(np.min(np.linalg.norm(robot_xpos - c[None, :], axis=1)))
            if dd < best_d:
                best_n, best_d = "robot", dd
            return best_n, best_d

        clusters = []
        for rank, cl in enumerate(g.clusters, start=1):
            idx = np.asarray(cl.point_indices, np.int64)
            labs, cnts = np.unique(point_group[idx], return_counts=True)
            order = np.argsort(-cnts)
            major = str(labs[order[0]])
            purity = float(cnts[order[0]]) / float(idx.size)
            bodies, bcnt = np.unique(point_body[idx], return_counts=True)
            bo = np.argsort(-bcnt)
            nn, nnd = nearest_name(np.asarray(cl.centroid, np.float64))
            clusters.append({
                "rank": rank,
                "id": int(cl.id),
                "name": major,
                "name_purity": purity,
                "name_counts": {str(labs[k]): int(cnts[k]) for k in order},
                "top_body": str(body_names.get(int(bodies[bo[0]]), "?")),
                "name_nearest_gt": nn,
                "nearest_gt_dist_mm": nnd * 1000.0,
                "target_score": float(cl.target_score),
                "mean_attention": float(cl.mean_attention),
                "max_attention": float(cl.max_attention),
                "point_count": int(cl.point_count),
                "rms_radius_mm": float(cl.rms_radius) * 1000.0,
                "dist_from_peak_mm": float(cl.distance_from_attention_peak) * 1000.0,
                "centroid": [float(x) for x in np.asarray(cl.centroid, np.float64)],
                "dist_to_apple_gt_mm": float(
                    np.linalg.norm(np.asarray(cl.centroid, np.float64) - world["apple"]) * 1000.0),
            })

        apple_cl = next((c for c in clusters if c["name"] == "apple"), None)
        top = clusters[0] if clusters else None

        peak = int(g.attention_peak_index)
        peak_name = (str(point_group[peak]) if 0 <= peak < len(fc) else "not_measured")

        apple_px = {}
        for c in mi.CAMERAS:
            p = mi.backproject_body(gt[c], bid["apple"])
            apple_px[c] = 0 if p is None else int(len(p))

        # ---- latch 를 safe_policy 와 같은 입력으로 한 frame 돌린다 ---------------------
        tgt = cs.target
        label = identity.label(getattr(tgt, "centroid", None))
        metrics = getattr(tgt, "metrics", {}) or {}
        chunk = np.asarray(step.actions, np.float64)
        # safe_policy.py:411 은 left gripper 열을 6 으로 박아 두었다 (14D 배치). 16D 의
        # wire.gripper_columns() 는 7 을 준다. 둘 다 남긴다.
        g_chunk_col6 = float(chunk[0, 6]) if chunk.ndim == 2 and chunk.shape[1] > 6 else None
        g_chunk_col7 = float(chunk[0, 7]) if chunk.ndim == 2 and chunk.shape[1] > 7 else None
        event = latch.update(
            label=label,
            score=float(getattr(tgt, "confidence", 0.0) or 0.0),
            runner_up=float(metrics.get("runner_up_score", 0.0) or 0.0),
            gripper=g_chunk_col6,
            placed=False,
        )

        st = np.asarray(step.state, np.float64).reshape(-1)
        frames.append({
            "seq": i + 1, "i": i, "t_step": int(step.t_step), "phase_injected": phase,
            "has_target": bool(cs.has_target),
            "status": str(cs.status.value),
            "grounding_status": str(g.status.value),
            "best_score": float(g.best_score),
            "threshold_gap": float(threshold - g.best_score),
            "n_clusters": int(len(g.clusters)),
            "n_cloud_points": int(len(fc)),
            "attention_peak_name": peak_name,
            "attention_peak_index": peak,
            "top_name": (top["name"] if top else None),
            "top_score": (top["target_score"] if top else None),
            "top_point_count": (top["point_count"] if top else None),
            "apple_rank": (apple_cl["rank"] if apple_cl else None),
            "apple_score": (apple_cl["target_score"] if apple_cl else None),
            "apple_point_count": (apple_cl["point_count"] if apple_cl else None),
            "apple_cluster_present": bool(apple_cl is not None),
            "apple_cloud_points": int(np.count_nonzero(point_group == "apple")),
            "apple_obs_in_cloud_per_camera": apple_obs_per_cam,
            "apple_gt_pixels_per_camera": apple_px,
            "apple_gt_pixels_total": int(sum(apple_px.values())),
            "cloud_group_counts": {str(k): int(v) for k, v in
                                   zip(*np.unique(point_group, return_counts=True))},
            "clusters": clusters,
            "latch_phase": str(event.phase.value),
            "latch_manipulated": event.manipulated,
            "latch_destination": event.destination,
            "latch_attach": bool(event.attach),
            "latch_detach": bool(event.detach),
            "latch_label_this_frame": label,
            "gripper_obs_left": float(st[gcol_l]),
            "gripper_obs_right": float(st[gcol_r]),
            "gripper_chunk_col6": g_chunk_col6,
            "gripper_chunk_col7": g_chunk_col7,
            "apple_z_m": float(world["apple"][2]),
            "ee_apple_dist_mm": float(np.linalg.norm(world["ee_finger_l1"] - world["apple"])
                                      * 1000.0),
            "apple_crate_xy_dist_mm": float(
                np.linalg.norm((world["apple"] - world["crate"])[:2]) * 1000.0),
        })
        if i % 5 == 0 or i == n_steps - 1:
            f = frames[-1]
            print(f"  [seq {f['seq']:3d}] t={f['t_step']:4d} top={str(f['top_name']):8s} "
                  f"{f['top_score'] if f['top_score'] is None else round(f['top_score'], 4)!s:7s} "
                  f"apple rank={f['apple_rank']} score="
                  f"{f['apple_score'] if f['apple_score'] is None else round(f['apple_score'], 4)} "
                  f"n_cl={f['n_clusters']:3d} latch={f['latch_phase']}/{f['latch_manipulated']}",
                  flush=True)

    scene.close()

    out = {
        "records": RECORDS,
        "attention": ATTENTION,
        "target_word": target_word,
        "attention_cell": cell,
        "cameras": list(mi.CAMERAS),
        "target_score_threshold": threshold,
        "esdf_backend": "legacy",
        "voxel_m": 0.020,
        "phase_boundaries_injected": [int(x) for x in boundaries],
        "state_gripper_columns": [int(gcol_l), int(gcol_r)],
        "latch_config": {"confirm_frames": latch_cfg.confirm_frames,
                         "score_ratio": latch_cfg.score_ratio,
                         "gripper_closed_below": latch_cfg.gripper_closed_below,
                         "release_frames": latch_cfg.release_frames},
        "naming_rule": ("cluster 이름 = member point 의 MuJoCo segmentation body 다수결. point 는 "
                        "FusedPointCloud.representative -> obs_camera/obs_uv 로 원래 pixel 까지 "
                        "되짚었다. apple/banana/orange/pear/crate/table 은 그 body 이름, base "
                        "후손 링크와 *_target marker 는 robot, 그 밖은 unknown"),
        "wall_s": time.time() - t0,
        "n_frames": len(frames),
        "frames": frames,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
