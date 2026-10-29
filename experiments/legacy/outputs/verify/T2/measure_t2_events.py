"""T2 — pick-place 전 구간 상태 전이. 아홉 이벤트의 프레임과 판정 근거.

`AG3S/trajopt/grasp_latch.py`(GraspLatch·CentroidIdentity)와 `safe_policy.py._run_latch`
가 실제로 하는 것을 **그대로** 재생한다 — 새 latch 로직을 만들지 않는다. `benchmark/**` 는
읽기만 한다.

production 배선과 다른 자리 셋 (문서화):
  1. `gripper` 값은 정책이 낸 chunk 의 첫 행이 아니라 **기록된 `state[7]`**(왼손, 코디네이터가
     이미 특정)을 쓴다 — 이 기록은 재생이라 새 chunk 를 만들지 않는다.
  2. `placed_fn` 은 production 참조 구현(`trajopt/placed.py`, ESDF 라벨 층 기반)이 아니라
     **MuJoCo 참값**으로 잰다 — 조작 대상의 실제 위치가 crate 로컬 프레임에서 벽 안쪽·바닥
     쪽에 있는가. 라벨 층 배선(destination_points 를 ESDF 까지 통과시키는 것)은 그대로
     재생하지만, "놓였다" 판정 자체는 참값을 쓴다.
  3. `phase`(transit/approach/grasp)는 `phase_boundaries_for_path` 가 단일 파지를 가정하고
     추정한 것이라 **두 번째 파지에서는 못 미더울 수 있다** — 참고용으로만 싣는다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T2/measure_t2_events.py \
        --records outputs/live_test/20260925_t2/run_0000/run_0000 \
        --attention AG3S/ag3s/asset/data/attention_16d_t2.npz \
        --out outputs/verify/T2/t2_events.json
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

import measure_items as mi  # noqa: E402  (재사용: CAMERAS, ATTENTION_ALIAS, backproject_body)

LEFT_GRIPPER_STATE_IDX = 7   # wire.gripper_columns(7) == (7, 15); 이 기록은 왼손만 쥔다
FRUITS = ("apple", "banana", "orange", "pear")
CRATE_INNER_X = 0.082   # crate_wall_p/nx local x=±0.086, 벽 두께 0.004 절반을 뺀 안쪽
CRATE_INNER_Y = 0.142   # crate_wall_p/ny local y=±0.146, 마찬가지
CRATE_RIM_LOCAL_Z = 0.062   # crate_vis_rim_* local z
CRATE_BELOW_RIM_MARGIN = 0.010


def crate_local_point(scene, world_point: np.ndarray) -> np.ndarray:
    T_crate_world = np.linalg.inv(scene.body_pose("crate"))
    p = T_crate_world @ np.array([*world_point, 1.0])
    return p[:3]


def is_placed_ground_truth(scene, object_name: str) -> bool:
    """참값 놓기 판정 — production 의 `destination_placement`(ESDF 라벨층)와 다른, 단순화한 대응.

    crate 로컬 프레임에서 벽 안쪽 · 테두리보다 아래(margin 10mm)에 있으면 놓인 것으로 본다.
    """
    p_world = scene.body_position(object_name)
    local = crate_local_point(scene, p_world)
    inside_xy = abs(local[0]) <= CRATE_INNER_X and abs(local[1]) <= CRATE_INNER_Y
    below_rim = local[2] <= (CRATE_RIM_LOCAL_Z - CRATE_BELOW_RIM_MARGIN)
    return bool(inside_xy and below_rim)


def nearest_fruit_name(scene, world_point: np.ndarray) -> tuple[str, float]:
    """`world_point` 에 가장 가까운 과일 body 이름과 거리(m) — 참값으로 "무엇을 쥐었나" 읽기용."""
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
    """이번 프레임 mujoco contact 중 손가락 body 가 낀 것의 **상대 body 이름** 목록."""
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
    from AG3S.trajopt.grasp_latch import CentroidIdentity, GraspLatch
    from AG3S.trajopt.safe_policy import DEFAULT_GRASP_LINKS, grasp_parent_links

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--attention", required=True)
    ap.add_argument("--step1-json",
                    default="AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json")
    ap.add_argument("--voxel", type=float, default=0.020)
    ap.add_argument("--range-max", type=float, default=2.0)
    ap.add_argument("--frames", type=int, default=0, help="0 이면 전부 (테스트용 절단)")
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

    ag_cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": args.range_max},
        "esdf": {"voxel_size": args.voxel, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"},
    })
    ag = AG3S(ag_cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot,
             attached_parent_links=grasp_parent_links(DEFAULT_GRASP_LINKS))

    identity = CentroidIdentity()
    latch = GraspLatch()
    destination_points = None       # production 과 같은 1프레임 지연 배선
    latched_target = None

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
        grounding = debug["grounding"]
        target = cs.target

        label = identity.label(None if target is None else target.centroid)
        if target is not None and label is not None and label == latch.manipulated:
            latched_target = target

        gripper_left = float(step.state[LEFT_GRIPPER_STATE_IDX])

        placed = False
        gt_name_at_attach = None
        if latch.holding and ag.attached is not None:
            # 잠긴 조작 대상의 참 이름 — attach 시점에 한 번 정해 둔 것을 재사용(아래에서 저장)
            gt_name_at_attach = frames_out[-1].get("manipulated_gt_name") if frames_out else None
            if gt_name_at_attach:
                placed = is_placed_ground_truth(scene, gt_name_at_attach)

        score = float(getattr(target, "confidence", 0.0) or 0.0)
        runner_up = float((target.metrics or {}).get("runner_up_score", 0.0)) if target is not None else 0.0

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

        # --- 참값 진단 (latch 로직에는 안 들어간다 — 사후 확인용) ---------------------------
        contact_bodies = grasp_contact_bodies(scene, finger_body_ids)
        contact_fruit = [b for b in contact_bodies if b in FRUITS]
        manip_gt_name, manip_gt_dist = ((None, None) if target is None else
                                        nearest_fruit_name(scene, np.asarray(target.centroid)))
        # attach 프레임에서는 latched_target 기준으로 참 이름을 고정해 이후 프레임에 물려준다
        manipulated_gt_name_for_next = gt_name_at_attach
        if did_attach and latched_target is not None:
            manipulated_gt_name_for_next, _ = nearest_fruit_name(
                scene, np.asarray(latched_target.centroid))

        frames_out.append({
            "i": i, "t_step": int(step.t_step), "phase": phase,
            "gripper_left_state7": gripper_left,
            "has_target": bool(cs.has_target),
            "target_score_confidence": score,
            "runner_up_score_from_metrics": runner_up,
            "label": label,
            "latch_phase": latch.phase.value,
            "manipulated": latch.manipulated,
            "destination": latch.destination,
            "attach_this_frame": did_attach,
            "detach_this_frame": did_detach,
            "latch_note": event.note,
            "ag3s_attached_is_set": ag.attached is not None,
            "n_attached_voxels_carved": (None if cs.esdf is None else
                                        int(cs.esdf.stats.get("n_attached_voxels_carved", 0))),
            "contact_finger_bodies": contact_bodies,
            "contact_fruit": contact_fruit,
            "manip_target_nearest_fruit_gt": manip_gt_name,
            "manip_target_nearest_fruit_dist_m": manip_gt_dist,
            "manipulated_gt_name": manipulated_gt_name_for_next,
            "placed_ground_truth": placed,
        })
        if i % 10 == 0 or i == len(run) - 1:
            print(f"  [{i:3d}] t={step.t_step:4d} grip={gripper_left:.3f} label={label} "
                  f"phase={latch.phase.value:9s} manip={latch.manipulated} dest={latch.destination} "
                  f"attach={did_attach} detach={did_detach} contact_fruit={contact_fruit}")

    scene.close()

    out = {"records": str(args.records), "attention": str(args.attention),
          "prompt": run.prompt, "prompt_target": prompt_target,
          "phase_boundaries": phase_evidence,
          "wall_s": time.time() - t_wall0, "frames": frames_out}
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
