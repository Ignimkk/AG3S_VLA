"""T2-b — (1) attach 뒤 쥔 물체가 robot 쪽으로 옮겨 앉는가, (2) 떠난 자리의 잔상은 몇 프레임인가.

`outputs/verify/T2/measure_t2_events.py` 의 latch 재생을 **그대로** 옮겨 쓴다 (같은 config,
같은 GraspLatch/CentroidIdentity 배선). `benchmark/**` 는 읽기만 한다.

T2 와 다른 자리 셋 (전부 여기에 적는다):
  1. `esdf.emit_candidates = True` — 장애물 cluster 개수를 세려면 primitive 경로가 돌아야 한다.
     latch 입력(label/score/gripper/placed)에는 영향이 없고, attach/detach 프레임이 T2 와
     같은지는 출력의 `t2_replay_match` 로 스스로 확인한다.
  2. **대조 파이프라인** `ag_ctrl` — 같은 관측을 먹이지만 `attach()` 를 절대 부르지 않는다.
     쥔 물체 자리의 거리를 두 필드에서 나란히 읽으면 "carve 가 무엇을 지웠나" 가 분리된다.
  3. 잔상 탐침 — apple 의 frame 0 자리와 pear 자리에 반지름 5 cm 구를 두고 프레임마다 그 안의
     점유 복셀을 센다. 필드는 base 프레임이므로 `body_position_in_base` 로 옮긴다.

측정 중에 A1 이 `stages/target_grounding.py` 에 `runner_up_score` 키를 넣었다 (md5
caae04a1... -> ee33a055...). 그 변경이 latch 에 닿는 경로는 **한 줄뿐이다** —
`safe_policy.py:341` 의 `metrics.get("runner_up_score", 0.0)`. 그래서 `--runner-up-mode zero`
(기본값) 는 latch 에 0.0 을 그대로 먹여 **변경 전과 같은 판정**을 재생하고, metrics 가 실제로
내놓는 값은 `runner_up_score_from_metrics` 로 따로 싣는다 (latch 에는 안 들어간다).

필드는 프레임마다 **즉시** 읽는다. 국소 갱신이 `previous` 격자를 제자리에서 덮으므로 이전
프레임의 `distance_grid` 를 들고 있다가 나중에 읽으면 다른 값이 나온다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T2b/measure_t2b.py \
        --records outputs/live_test/20260925_ep1807/run_0000 \
        --attention AG3S/ag3s/asset/data/attention_16d_ep1807.npz \
        --out outputs/verify/T2b/t2b_attached_residual.json
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

import measure_items as mi  # noqa: E402  (재사용: CAMERAS, ATTENTION_ALIAS)

LEFT_GRIPPER_STATE_IDX = 7
FRUITS = ("apple", "banana", "orange", "pear")
CRATE_INNER_X = 0.082
CRATE_INNER_Y = 0.142
CRATE_RIM_LOCAL_Z = 0.062
CRATE_BELOW_RIM_MARGIN = 0.010
PROBE_RADIUS = 0.050          # 잔상 탐침 구의 반지름 (task.md 지시: 5 cm)
NEAR_CLUSTER_M = 0.080        # "쥔 물체 자리에 장애물 cluster 가 있다" 로 세는 거리


def crate_local_point(scene, world_point: np.ndarray) -> np.ndarray:
    T_crate_world = np.linalg.inv(scene.body_pose("crate"))
    p = T_crate_world @ np.array([*world_point, 1.0])
    return p[:3]


def is_placed_ground_truth(scene, object_name: str) -> bool:
    local = crate_local_point(scene, scene.body_position(object_name))
    inside_xy = abs(local[0]) <= CRATE_INNER_X and abs(local[1]) <= CRATE_INNER_Y
    below_rim = local[2] <= (CRATE_RIM_LOCAL_Z - CRATE_BELOW_RIM_MARGIN)
    return bool(inside_xy and below_rim)


def nearest_fruit_name(scene, world_point: np.ndarray):
    best_name, best_d = None, float("inf")
    for f in FRUITS:
        try:
            d = float(np.linalg.norm(scene.body_position(f) - world_point))
        except KeyError:
            continue
        if d < best_d:
            best_name, best_d = f, d
    return best_name, best_d


def grasp_contact_bodies(scene, finger_body_ids: set[int]) -> list[str]:
    import mujoco
    data = scene.data
    out = []
    for i in range(int(data.ncon)):
        c = data.contact[i]
        b1 = int(scene.model.geom_bodyid[c.geom1])
        b2 = int(scene.model.geom_bodyid[c.geom2])
        if b1 in finger_body_ids and b2 not in finger_body_ids:
            out.append(mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b2) or f"body_{b2}")
        elif b2 in finger_body_ids and b1 not in finger_body_ids:
            out.append(mujoco.mj_id2name(scene.model, mujoco.mjtObj.mjOBJ_BODY, b1) or f"body_{b1}")
    return out


# --- 필드 조회 ----------------------------------------------------------------------------
def sphere_voxel_stats(field, centre: np.ndarray, radius: float) -> dict:
    """`centre` 반지름 `radius` 구 안의 복셀 통계. 점유 = ESDF 거리 <= 0 (음수가 표면 안쪽)."""
    grid = field.grid
    vs = grid.voxel_size
    lo = np.floor((centre - radius - grid.origin) / vs).astype(np.int64)
    hi = np.ceil((centre + radius - grid.origin) / vs).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(grid.shape) - 1)
    if np.any(hi < lo):
        return {"n_voxels_in_sphere": 0, "n_occupied": 0, "n_occupied_upper": 0,
                "min_distance_mm": None, "centre_distance_mm": None, "in_grid": False}
    ax = [grid.origin[i] + np.arange(lo[i], hi[i] + 1) * vs for i in range(3)]
    gx, gy, gz = np.meshgrid(*ax, indexing="ij")
    centres = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    sub = field.distance_grid[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1].reshape(-1)
    inside = np.linalg.norm(centres - centre, axis=1) <= radius
    if not inside.any():
        return {"n_voxels_in_sphere": 0, "n_occupied": 0, "n_occupied_upper": 0,
                "min_distance_mm": None, "centre_distance_mm": None, "in_grid": False}
    d = sub[inside]
    z = centres[inside][:, 2]
    occ = d <= 0.0
    return {
        "n_voxels_in_sphere": int(inside.sum()),
        "n_occupied": int(occ.sum()),
        # 위쪽 반구만 — 테이블 면이 탐침 구 아래쪽을 늘 채우므로, 물체/잔상만 보려면 잘라야 한다
        "n_occupied_upper": int((occ & (z >= centre[2])).sum()),
        "min_distance_mm": float(d.min() * 1000.0),
        "centre_distance_mm": float(field.distance(centre.reshape(1, 3))[0] * 1000.0),
        "in_grid": True,
    }


def point_distance_stats(field, points: np.ndarray) -> dict:
    """쥔 물체의 점들에서 필드가 뭐라고 답하나. 음수 = 그 점이 아직 장애물 안쪽."""
    if points is None or not len(points):
        return {"n": 0}
    d = np.asarray(field.distance(points), np.float64) * 1000.0
    idx, inside = field.grid.to_index(points)
    return {
        "n": int(len(points)),
        "n_in_grid": int(inside.sum()),
        "min_mm": float(d.min()), "median_mm": float(np.median(d)), "max_mm": float(d.max()),
        "n_nonpositive": int((d <= 0.0).sum()),
    }


def main() -> None:
    import mujoco

    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S
    from AG3S.ag3s.constraints.attached import attached_points_in_base, attached_spheres
    from AG3S.ag3s.experiments.reports.attention_report import target_from_prompt
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model, build_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import camera_observation
    from AG3S.ag3s.experiments.sources.policy_record import (
        load_run, phase_boundaries_for_path, phase_for, pose_scene, replay_scene)
    from AG3S.ag3s.types import SourceType
    from AG3S.trajopt.grasp_latch import CentroidIdentity, GraspLatch
    from AG3S.trajopt.safe_policy import DEFAULT_GRASP_LINKS, grasp_parent_links

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", required=True)
    ap.add_argument("--step1-json",
                    default="AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--no-control", action="store_true",
                    help="대조 파이프라인을 끈다 (빠른 연습용)")
    ap.add_argument("--runner-up-mode", choices=("zero", "metrics"), default="zero",
                    help="latch 에 먹이는 runner_up. zero = A1 변경 전과 같은 판정")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t_wall0 = time.time()
    run = load_run(args.records, limit=args.frames or None)
    prompt_target = target_from_prompt(run.prompt)
    boundaries, phase_evidence = phase_boundaries_for_path(args.records)

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
    finger_body_ids = {mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, n)
                       for n in ("ee_finger_l1", "ee_finger_l2")}

    cfg_dict = {
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy",
                 # T2 와 다른 유일한 config — cluster 를 세기 위해 primitive 경로를 켠다
                 "emit_candidates": True},
    }
    ag_cfg = AG3SConfig.from_dict(cfg_dict)
    ag = AG3S(ag_cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot,
              attached_parent_links=grasp_parent_links(DEFAULT_GRASP_LINKS))
    ag_ctrl = None
    if not args.no_control:
        ag_ctrl = AG3S(AG3SConfig.from_dict(cfg_dict), robot_model=filter_robot,
                       constraint_robot_model=constraint_robot,
                       attached_parent_links=grasp_parent_links(DEFAULT_GRASP_LINKS))

    identity = CentroidIdentity()
    latch = GraspLatch()
    destination_points = None
    latched_target = None

    # --- 잔상 탐침 자리 (base 프레임, frame 0 자세에서 한 번 고정) -----------------------
    first_step = next(iter(run))
    pose_scene(scene, first_step)
    probe_sites = {
        "apple_frame0": np.asarray(scene.body_position_in_base("apple"), np.float64),
        "pear_static": np.asarray(scene.body_position_in_base("pear"), np.float64),
    }
    probe_world = {
        "apple_frame0": np.asarray(scene.body_position("apple"), np.float64),
        "pear_static": np.asarray(scene.body_position("pear"), np.float64),
    }
    probe_body = {"apple_frame0": "apple", "pear_static": "pear"}
    base_pose_frame0 = np.asarray(scene.body_pose("base"), np.float64)

    frames_out = []
    for i, step in enumerate(run):
        pose_scene(scene, step)
        att_by_cam = {cam: attention_block[i, di, ai, cell["layer"], cell["head"], ci]
                      for cam, ci in attention_camera_index.items()}
        observations = []
        for c in mi.CAMERAS:
            obs, _f = camera_observation(scene, c, filter_robot, timestamp=float(i),
                                         attention_map=att_by_cam.get(c))
            observations.append(obs)
        phase = phase_for(step.t_step, boundaries)

        cs, debug = ag.process_multi_debug(observations, phase=phase,
                                           destination_points=destination_points)
        target = cs.target
        label = identity.label(None if target is None else target.centroid)
        if target is not None and label is not None and label == latch.manipulated:
            latched_target = target

        gripper_left = float(step.state[LEFT_GRIPPER_STATE_IDX])

        placed = False
        gt_name_at_attach = None
        if latch.holding and ag.attached is not None:
            gt_name_at_attach = frames_out[-1].get("manipulated_gt_name") if frames_out else None
            if gt_name_at_attach:
                placed = is_placed_ground_truth(scene, gt_name_at_attach)

        score = float(getattr(target, "confidence", 0.0) or 0.0)
        runner_up_metrics = (float((target.metrics or {}).get("runner_up_score", 0.0) or 0.0)
                             if target is not None else 0.0)
        runner_up = 0.0 if args.runner_up_mode == "zero" else runner_up_metrics
        # 두 모드에서 confident 판정이 갈리는 프레임이 있나 — 실행 없이 식으로만 (score_ratio=1.3)
        sr = 1.3
        confident_zero = score >= sr * max(0.0, 1e-9)
        confident_metrics = score >= sr * max(runner_up_metrics, 1e-9)

        event = latch.update(label=label, score=score, runner_up=runner_up,
                             gripper=gripper_left, placed=placed)

        did_attach = did_detach = False
        if event.attach and latched_target is not None:
            q = max(observations, key=lambda o: o.timestamp).robot_state
            ag.attach(latched_target, robot_state=np.asarray(q, np.float64),
                      parent_link="ee_finger_l1",
                      allowed_contact_links=("ee_finger_l1", "ee_finger_l2"), label="manipulated")
            did_attach = True
        elif event.detach:
            ag.detach()
            destination_points = None
            did_detach = True

        if event.destination is not None and label == event.destination and target is not None:
            destination_points = np.asarray(target.points, np.float64)

        # --- 대조: 같은 관측, attach 없음 -------------------------------------------------
        cs_ctrl = None
        if ag_ctrl is not None:
            cs_ctrl, debug_ctrl = ag_ctrl.process_multi_debug(
                observations, phase=phase, destination_points=destination_points)

        # --- 측정 1: 쥔 물체가 robot 쪽에 있고 obstacle 쪽에서 빠졌나 ---------------------
        robot_state = np.asarray(max(observations, key=lambda o: o.timestamp).robot_state, np.float64)
        att = ag.attached
        att_block: dict = {"is_set": att is not None}
        if att is not None:
            spheres = attached_spheres(att)
            pts_base = attached_points_in_base(att, robot_model=constraint_robot,
                                               robot_state=robot_state)
            att_block.update({
                "n_snapshot_points": (0 if att.points is None else int(len(att.points))),
                "n_spheres": len(spheres),
                "sphere_radii_mm": [round(float(r) * 1000.0, 4) for _c, r in spheres],
                "parent_link": att.parent_link,
                "label": att.label,
                "source_candidate_id": int(att.source_candidate_id),
                "allowed_contact_links": sorted(att.allowed_contact_links),
            })
            if pts_base is not None and len(pts_base):
                c = pts_base.mean(axis=0)
                att_block.update({
                    "points_in_base_centroid": [float(v) for v in c],
                    "points_in_base_extent_mm": [float(v) for v in
                                                 (pts_base.max(axis=0) - pts_base.min(axis=0)) * 1000.0],
                    # 필드가 쥔 물체 점들에 뭐라고 답하나 (음수 = 아직 장애물)
                    "field_at_points_main": (None if cs.esdf is None else
                                             point_distance_stats(cs.esdf, pts_base)),
                    "field_at_points_control": (None if cs_ctrl is None or cs_ctrl.esdf is None else
                                                point_distance_stats(cs_ctrl.esdf, pts_base)),
                })
        # 쥔 물체의 **참값** 자리 (base) — cluster 가 거기 남아 있는지 보는 기준점
        held_gt_name = gt_name_at_attach or ("apple" if att is not None else None)
        held_gt_base = (None if held_gt_name is None else
                        np.asarray(scene.body_position_in_base(held_gt_name), np.float64))

        def cluster_block(constraint_set, held_base):
            if constraint_set is None:
                return None
            by_type: dict[str, int] = {}
            near = []
            for cand in constraint_set.candidates:
                key = SourceType(cand.source_type).value
                by_type[key] = by_type.get(key, 0) + 1
                if held_base is not None:
                    d = float(np.linalg.norm(np.asarray(cand.centroid, np.float64) - held_base))
                    near.append((d, key, int(cand.point_count), int(cand.id)))
            near.sort()
            out = {"n_candidates": len(constraint_set.candidates), "by_source_type": by_type,
                   "n_within_80mm_of_held_gt": sum(1 for d, *_ in near if d <= NEAR_CLUSTER_M)}
            if near:
                d, key, npts, cid = near[0]
                out["nearest_cluster_to_held_gt"] = {
                    "dist_mm": d * 1000.0, "source_type": key, "point_count": npts, "id": cid}
            nonts = [n for n in near if n[1] != "target"]
            if nonts:
                d, key, npts, cid = nonts[0]
                out["nearest_non_target_cluster_to_held_gt"] = {
                    "dist_mm": d * 1000.0, "source_type": key, "point_count": npts, "id": cid}
            return out

        # --- 측정 2: 잔상 탐침 -------------------------------------------------------------
        residual = {}
        for name, centre in probe_sites.items():
            entry = {"main": (None if cs.esdf is None else
                              sphere_voxel_stats(cs.esdf, centre, PROBE_RADIUS))}
            if cs_ctrl is not None and cs_ctrl.esdf is not None:
                entry["control_no_attach"] = sphere_voxel_stats(cs_ctrl.esdf, centre, PROBE_RADIUS)
            body = probe_body[name]
            now_base = np.asarray(scene.body_position_in_base(body), np.float64)
            entry["body_dist_from_probe_mm"] = float(np.linalg.norm(now_base - centre) * 1000.0)
            entry["body_pos_in_base_now"] = [float(v) for v in now_base]
            residual[name] = entry

        contact_bodies = grasp_contact_bodies(scene, finger_body_ids)
        contact_fruit = [b for b in contact_bodies if b in FRUITS]
        manipulated_gt_name_for_next = gt_name_at_attach
        if did_attach and latched_target is not None:
            manipulated_gt_name_for_next, _ = nearest_fruit_name(
                scene, np.asarray(latched_target.centroid))

        base_drift_mm = float(np.linalg.norm(
            np.asarray(scene.body_pose("base"), np.float64)[:3, 3] - base_pose_frame0[:3, 3]) * 1000.0)

        frames_out.append({
            "i": i, "t_step": int(step.t_step), "phase": phase,
            "gripper_left_state7": gripper_left,
            "has_target": bool(cs.has_target),
            "label": label,
            "target_score_confidence": score,
            "runner_up_score_from_metrics": runner_up_metrics,
            "runner_up_fed_to_latch": runner_up,
            "confident_if_runner_up_zero": bool(confident_zero),
            "confident_if_runner_up_from_metrics": bool(confident_metrics),
            "latch_phase": latch.phase.value,
            "manipulated": latch.manipulated,
            "destination": latch.destination,
            "attach_this_frame": did_attach,
            "detach_this_frame": did_detach,
            "latch_note": event.note,
            "contact_fruit": contact_fruit,
            "manipulated_gt_name": manipulated_gt_name_for_next,
            "placed_ground_truth": placed,
            "attached": att_block,
            "n_attached_voxels_carved": (None if cs.esdf is None else
                                         int(cs.esdf.stats.get("n_attached_voxels_carved", 0))),
            "esdf_occupied_main": (None if cs.esdf is None else
                                   int(cs.esdf.stats.get("n_occupied", 0))),
            "esdf_occupied_control": (None if cs_ctrl is None or cs_ctrl.esdf is None else
                                      int(cs_ctrl.esdf.stats.get("n_occupied", 0))),
            "clusters_main": cluster_block(cs, held_gt_base),
            "clusters_control": cluster_block(cs_ctrl, held_gt_base),
            "held_gt_name": held_gt_name,
            "held_gt_pos_in_base": (None if held_gt_base is None else
                                    [float(v) for v in held_gt_base]),
            "residual_probes": residual,
            "base_body_drift_from_frame0_mm": base_drift_mm,
        })
        if i % 5 == 0 or 15 <= i <= 35 or i == len(run) - 1:
            a = att_block
            fm = (a.get("field_at_points_main") or {}) if a.get("is_set") else {}
            fc = (a.get("field_at_points_control") or {}) if a.get("is_set") else {}
            rp = residual["apple_frame0"]["main"] or {}
            print(f"  [{i:3d}] t={step.t_step:4d} grip={gripper_left:.3f} {latch.phase.value:8s} "
                  f"att={a['is_set']} carve={frames_out[-1]['n_attached_voxels_carved']} "
                  f"pts_main_min={fm.get('min_mm')} n<=0={fm.get('n_nonpositive')} "
                  f"ctrl_min={fc.get('min_mm')} n<=0={fc.get('n_nonpositive')} "
                  f"apple_site_occ={rp.get('n_occupied')}/{rp.get('n_occupied_upper')}")

    scene.close()

    out = {
        "records": str(args.records), "attention": str(args.attention),
        "prompt": run.prompt, "prompt_target": prompt_target,
        "phase_boundaries": phase_evidence,
        "config": cfg_dict,
        "runner_up_mode": args.runner_up_mode,
        "probe_radius_m": PROBE_RADIUS,
        "near_cluster_m": NEAR_CLUSTER_M,
        "probe_sites_base": {k: [float(v) for v in p] for k, p in probe_sites.items()},
        "probe_sites_world_frame0": {k: [float(v) for v in p] for k, p in probe_world.items()},
        "control_pipeline": ag_ctrl is not None,
        "wall_s": time.time() - t_wall0,
        "frames": frames_out,
    }
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
