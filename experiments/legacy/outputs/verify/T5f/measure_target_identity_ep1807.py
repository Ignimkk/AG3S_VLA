"""T5f — chunk 마다 **고른 target 이 무엇이었나**, 이전(threshold 0.25) 과 이후(1 등 + confirm 3).

`outputs/verify/T5d/measure_attention_flow_ep1807.py` 를 넓힌 것이다. T5d 는 모든 cluster 의
참값 이름과 rank-1 만 냈다. 여기서 더하는 것:

  · **지금 코드**(T5e: `target_score_threshold 0.0` · `target_confirm_frames 3` ·
    `target_identity_tolerance 0.06`)로 pipeline 을 프레임 순서대로 돌려 실제로 고른 target 의
    참값 이름 · rank · confirm streak · `constraint_set.notes` 의 hold/switch 줄.
  · 같은 cluster 목록에서 **이전 동작을 그대로 유도**한다 — 옛 코드는
    `best.target_score < 0.25` 이면 `LOW_SCORE`(target 없음), 아니면 `clusters[0]`.
    clustering 은 threshold·confirm 과 무관하므로 cluster 목록이 두 조건에서 동일하고,
    이 유도는 근사가 아니라 그 분기를 그대로 다시 적용한 것이다.
  · target cluster 의 **점 수 · rms radius · 참값 이름 순도**.
  · **MuJoCo 참값 clearance** — 팔 링크 18 개의 geom 과 물체 body geom 사이 최소 부호거리
    (`mj_geomDistance`), argmin 쌍까지. 관측 qpos 한 자세에서다.

    **내가 먼저 틀렸던 것 두 가지 (첫 실행에서 잡아 고쳤다).**
    (1) `distmax` 를 1.0 m 로 주면 mesh geom 쌍에서 `mj_geomDistance` 가 **정확히 0.0** 을
        돌려준다 — 실제 거리가 0.37~0.56 m 인 쌍에서 그랬다. 0.3 m 부터 나타나고 0.1 m 이하에서는
        사라진다. 그래서 `distmax` 를 **0.10 m** 로 내리고, 대조로 0.05 m 도 같이 재서 최소값이
        일치하는지 확인한다. 0.10 m 를 넘는 거리는 saturate 되어 `100.0 mm` 로 보고된다.
    (2) 시각 전용 geom (`contype == 0 and conaffinity == 0`) 을 같이 넣으면 그쪽에서도 0.0 이
        나온다. 충돌에 참여하는 geom 만 쓴다 (팔 307 → 267, 물체 → 26).
  · T5d 의 과제 국면 규칙(`first_closed_seq 20` · `first_reopen_seq 33`) 재계산.

`benchmark/**` 는 한 줄도 고치지 않는다. 새로 촬영하지 않는다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5f/measure_target_identity_ep1807.py
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
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5f/target_identity_ep1807.json'))

#: 참값 이름으로 접는 규칙 (T5d 와 동일). 이 밖의 body 는 robot 또는 unknown.
OBJECT_BODIES = ("apple", "banana", "orange", "pear", "crate", "table")
#: 이전 조건의 score 문턱 — 옛 기본값.
OLD_THRESHOLD = 0.25


def build_body_group(model):
    """body id -> 여덟 이름 중 하나. 로봇 링크는 전부 `robot`. (T5d 와 동일)"""
    import mujoco

    names = {b: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}")
             for b in range(model.nbody)}
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

    scene = replay_scene(run)
    filter_robot = build_robot_model(scene)
    constraint_robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    group, body_names = build_body_group(scene.model)
    model = scene.model
    bid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
           for n in (*OBJECT_BODIES, "ee_finger_l1")}
    if min(bid.values()) < 0:
        raise SystemExit(f"missing bodies: {bid}")

    # --- 참값 clearance 용 geom 묶음 -------------------------------------------------------
    #: mesh geom 쌍에서 큰 distmax 가 0.0 을 만든다 (위 docstring). 0.10 m 로 재고 0.05 m 로 대조.
    DISTMAX = 0.10
    DISTMAX_CHECK = 0.05

    def collidable(g: int) -> bool:
        return int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0

    arm_body = {}
    for ln in ARM_LINKS:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ln)
        if b < 0:
            raise SystemExit(f"arm link body not found: {ln}")
        arm_body[b] = ln
    n_arm_geoms_all = sum(1 for g in range(model.ngeom)
                          if int(model.geom_bodyid[g]) in arm_body)
    arm_geoms = [(g, arm_body[int(model.geom_bodyid[g])]) for g in range(model.ngeom)
                 if int(model.geom_bodyid[g]) in arm_body and collidable(g)]
    obj_geoms = []
    n_obj_geoms_all = 0
    for n in OBJECT_BODIES:
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) == bid[n]:
                n_obj_geoms_all += 1
                if collidable(g):
                    obj_geoms.append((g, n))

    cfg = AG3SConfig.from_dict({
        "collision_backend": "esdf",
        "pointcloud": {"range_max": 2.0},
        "esdf": {"voxel_size": 0.020, "max_distance": 0.4,
                 "exclude_support_surfaces": False, "backend": "legacy"},
    })
    ag = AG3S(cfg, robot_model=filter_robot, constraint_robot_model=constraint_robot)
    live_threshold = float(cfg.clustering.target_score_threshold)
    confirm_frames = int(cfg.clustering.target_confirm_frames)
    identity_tol = float(cfg.clustering.target_identity_tolerance)

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
            obs_cam_name[obs.camera_id] = c
        phase = phase_for(step.t_step, boundaries)
        cs, debug = ag.process_multi_debug(observations, phase=phase)
        g = debug["grounding"]
        fc = debug["fusion"].cloud

        # ---- fused point 별 참값 이름 (provenance -> pixel -> segmentation) --------------
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

        d = scene.data
        world = {n: np.asarray(d.xpos[bid[n]], np.float64) for n in bid}

        def describe(cl, rank):
            idx = np.asarray(cl.point_indices, np.int64)
            labs, cnts = np.unique(point_group[idx], return_counts=True)
            order = np.argsort(-cnts)
            bodies, bcnt = np.unique(point_body[idx], return_counts=True)
            bo = np.argsort(-bcnt)
            c3 = np.asarray(cl.centroid, np.float64)
            return {
                "rank": rank,
                "id": int(cl.id),
                "name": str(labs[order[0]]),
                "name_purity": float(cnts[order[0]]) / float(idx.size),
                "name_counts": {str(labs[k]): int(cnts[k]) for k in order},
                "top_body": str(body_names.get(int(bodies[bo[0]]), "?")),
                "target_score": float(cl.target_score),
                "mean_attention": float(cl.mean_attention),
                "point_count": int(cl.point_count),
                "rms_radius_mm": float(cl.rms_radius) * 1000.0,
                "spatial_compactness": float(cl.spatial_compactness),
                "dist_from_peak_mm": float(cl.distance_from_attention_peak) * 1000.0,
                "centroid": [float(x) for x in c3],
                "dist_to_apple_gt_mm": float(np.linalg.norm(c3 - world["apple"]) * 1000.0),
            }

        clusters = [describe(cl, r) for r, cl in enumerate(g.clusters, start=1)]

        # ---- 이후(지금 코드)가 실제로 고른 것 --------------------------------------------
        tgt = cs.target
        metrics = dict(getattr(tgt, "metrics", {}) or {}) if tgt is not None else {}
        after = None
        if tgt is not None:
            chosen_rank = int(metrics.get("target_rank", 0.0)) + 1  # 0-based -> 1-based
            chosen = clusters[chosen_rank - 1] if chosen_rank - 1 < len(clusters) else None
            after = {
                "has_target": True,
                "chosen_rank": chosen_rank,
                "confirm_streak": int(metrics.get("target_confirm_streak", 0.0)),
                "name": (chosen or {}).get("name"),
                "name_purity": (chosen or {}).get("name_purity"),
                "name_counts": (chosen or {}).get("name_counts"),
                "top_body": (chosen or {}).get("top_body"),
                "target_score": float(getattr(tgt, "confidence", 0.0) or 0.0),
                "point_count": int(metrics.get("point_count", 0.0)),
                "rms_radius_mm": float(metrics.get("rms_radius", 0.0)) * 1000.0,
                "spatial_compactness": float(metrics.get("spatial_compactness", 0.0)),
                "runner_up_score": float(metrics.get("runner_up_score", 0.0)),
                "centroid": [float(x) for x in np.asarray(tgt.centroid, np.float64)],
                "dist_to_apple_gt_mm": float(
                    np.linalg.norm(np.asarray(tgt.centroid, np.float64) - world["apple"]) * 1000.0),
            }
        else:
            after = {"has_target": False, "chosen_rank": None, "name": None,
                     "confirm_streak": None, "target_score": None, "point_count": None,
                     "rms_radius_mm": None}

        # ---- 이전(threshold 0.25) 을 같은 cluster 목록에서 그대로 유도 -------------------
        if clusters and clusters[0]["target_score"] >= OLD_THRESHOLD:
            b4 = {"has_target": True, "name": clusters[0]["name"],
                  "target_score": clusters[0]["target_score"],
                  "point_count": clusters[0]["point_count"],
                  "rms_radius_mm": clusters[0]["rms_radius_mm"],
                  "status": "ok"}
        else:
            b4 = {"has_target": False, "name": None,
                  "target_score": (clusters[0]["target_score"] if clusters else None),
                  "point_count": None, "rms_radius_mm": None,
                  "status": ("low_score" if clusters else "no_cluster")}

        # ---- MuJoCo 참값 clearance: 팔 링크 geom vs 물체 geom ----------------------------
        best_d, best_pair = float("inf"), None
        per_object = {}
        n_exact_zero = 0
        for og, on in obj_geoms:
            omin, opair = float("inf"), None
            for ag_, ln in arm_geoms:
                dd = mujoco.mj_geomDistance(model, d, ag_, og, DISTMAX, None)
                if dd == 0.0:
                    n_exact_zero += 1
                if dd < omin:
                    omin, opair = dd, ln
            prev = per_object.get(on)
            if prev is None or omin < prev["mm"]:
                per_object[on] = {"mm": omin, "link": opair}
            if omin < best_d:
                best_d, best_pair = omin, (opair, on)
        # 대조: 더 작은 distmax 로 다시 재서 최소값이 같은지 본다.
        check_d, check_pair = float("inf"), None
        for og, on in obj_geoms:
            for ag_, ln in arm_geoms:
                dd = mujoco.mj_geomDistance(model, d, ag_, og, DISTMAX_CHECK, None)
                if dd < check_d:
                    check_d, check_pair = dd, (ln, on)

        st = np.asarray(step.state, np.float64).reshape(-1)
        frames.append({
            "seq": i + 1, "i": i, "t_step": int(step.t_step), "phase_injected": phase,
            "grounding_status": str(g.status.value),
            "status": str(cs.status.value),
            "n_clusters": int(len(g.clusters)),
            "n_cloud_points": int(len(fc)),
            "best_score": float(g.best_score),
            "attention_peak_name": (str(point_group[int(g.attention_peak_index)])
                                    if 0 <= int(g.attention_peak_index) < len(fc)
                                    else "not_measured"),
            "after": after,
            "before_threshold_0p25": b4,
            "notes_from_pipeline": [str(n) for n in (cs.notes or [])],
            "clusters": clusters,
            "cloud_group_counts": {str(k): int(v) for k, v in
                                   zip(*np.unique(point_group, return_counts=True))},
            "gt_clearance_mm": {
                "min_mm": best_d * 1000.0,
                "min_pair": list(best_pair) if best_pair else None,
                "saturated_at_distmax": bool(best_d >= DISTMAX - 1e-12),
                "min_mm_at_distmax_0p05": check_d * 1000.0,
                "min_pair_at_distmax_0p05": list(check_pair) if check_pair else None,
                "cross_check_agrees": bool(abs(best_d - check_d) < 1e-9
                                           or (best_d >= DISTMAX - 1e-12
                                               and check_d >= DISTMAX_CHECK - 1e-12)),
                "n_pairs_exactly_zero": n_exact_zero,
                "per_object_mm": {k: round(v["mm"] * 1000.0, 4) for k, v in per_object.items()},
                "per_object_link": {k: v["link"] for k, v in per_object.items()},
            },
            "gripper_obs_left": float(st[gcol_l]),
            "gripper_obs_right": float(st[gcol_r]),
            "apple_z_m": float(world["apple"][2]),
            "ee_apple_dist_mm": float(np.linalg.norm(world["ee_finger_l1"] - world["apple"])
                                      * 1000.0),
            "apple_crate_xy_dist_mm": float(
                np.linalg.norm((world["apple"] - world["crate"])[:2]) * 1000.0),
        })
        if i % 5 == 0 or i == n_steps - 1:
            f = frames[-1]
            a, b = f["after"], f["before_threshold_0p25"]
            print(f"  [seq {f['seq']:3d}] t={f['t_step']:4d} "
                  f"after={str(a['name']):8s} rank={a['chosen_rank']} streak={a['confirm_streak']} "
                  f"score={None if a['target_score'] is None else round(a['target_score'], 4)} "
                  f"| before={str(b['name']):8s} | n_cl={f['n_clusters']:3d} "
                  f"gt_clear={f['gt_clearance_mm']['min_mm']:.1f}mm "
                  f"{f['gt_clearance_mm']['min_pair']}", flush=True)

    scene.close()

    out = {
        "records": RECORDS,
        "attention": ATTENTION,
        "target_word": target_word,
        "attention_cell": cell,
        "cameras": list(mi.CAMERAS),
        "config_now": {"target_score_threshold": live_threshold,
                       "target_confirm_frames": confirm_frames,
                       "target_identity_tolerance": identity_tol},
        "old_threshold_used_for_derivation": OLD_THRESHOLD,
        "derivation_rule_before": ("옛 코드는 target_grounding.py 에서 best.target_score < "
                                   "target_score_threshold(0.25) 이면 LOW_SCORE(target None), "
                                   "아니면 clusters[0] 을 target 으로 삼았다. clustering 은 "
                                   "threshold·confirm 과 무관하므로 같은 cluster 목록에 그 분기를 "
                                   "다시 적용한 것이다"),
        "naming_rule": ("cluster 이름 = member point 의 MuJoCo segmentation body 다수결. point 는 "
                        "FusedPointCloud.representative -> obs_camera/obs_uv 로 원래 pixel 까지 "
                        "되짚었다. apple/banana/orange/pear/crate/table 은 그 body 이름, base "
                        "후손 링크와 *_target marker 는 robot, 그 밖은 unknown"),
        "gt_clearance_rule": (f"mj_geomDistance(distmax {DISTMAX} m) — ARM_LINKS 18 개 body 의 "
                              f"충돌 geom {len(arm_geoms)} 개(전체 {n_arm_geoms_all})와 물체 6 개 "
                              f"body 의 충돌 geom {len(obj_geoms)} 개(전체 {n_obj_geoms_all}) 사이 "
                              "최소 부호거리. 그 프레임의 관측 qpos 한 자세에서다 "
                              f"(궤적 전체가 아니다). distmax 를 넘으면 {DISTMAX * 1000} mm 로 "
                              f"saturate 된다. distmax {DISTMAX_CHECK} m 로 다시 재서 대조한다"),
        "gt_clearance_distmax_m": DISTMAX,
        "gt_clearance_distmax_check_m": DISTMAX_CHECK,
        "n_arm_geoms": len(arm_geoms),
        "n_arm_geoms_all": n_arm_geoms_all,
        "n_object_geoms": len(obj_geoms),
        "n_object_geoms_all": n_obj_geoms_all,
        "esdf_backend": "legacy",
        "voxel_m": 0.020,
        "phase_boundaries_injected": [int(x) for x in boundaries],
        "wall_s": time.time() - t0,
        "n_frames": len(frames),
        "frames": frames,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {OUT}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
