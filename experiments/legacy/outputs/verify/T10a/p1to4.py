"""T10a P1..P4 — 손가락을 막는 body 는 무엇인가. rollout 없다.

방법은 T9a/T9b 와 같다: 기록된 `qpos` 로 씬을 놓고 (물체 free joint 포함), chunk 의 action 행으로
제약 구 266 개 중심을 FK 로 구한 뒤 MuJoCo 충돌 geom **표면까지 해석적 거리**를 잰다.
`mj_geomDistance` 를 쓰지 않는다. 구 모델은 서버와 같다 (T9a 에서 시작 로그와 대조 완료).

P1 접근 구간에서 gripper 구에 가장 가까운 scene body 는 무엇인가 (사과 포함/제외 둘 다)
P2 margin 을 얼마로 내리면 정책 원본 궤적이 제약을 안 건드리나 (10/5/2/1/0 mm)
P3 trust region 한 걸음을 가득 쓴 chunk 와, 접근 구간에서 같은 일이 작게 일어나는가
P4 T9 (목적함수 켠 것) 대 T10 (끈 것) 나란히
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import collections
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))

from m3_to_refined_vs_reference import (  # noqa: E402
    ACT_COL, JOINT_ORDER, MODEL_XML, QADR, SCENE_BODIES, surface_distances)

RUNS = {
    "T10": _ag3s_workspace_path('outputs/live_test/20260928_t10/execute_ep1807/frames.jsonl'),
    "T9": _ag3s_workspace_path('outputs/live_test/20260926_t9/execute_ep1807/frames.jsonl'),
}
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T10a'))
JSON_OUT = OUT / "p1to4.json"
NPZ = OUT / "p1to4_curves.npz"
SPHERE_OPTS = {
    "sphere_spacing": 0.4, "max_spheres_per_capsule": 32, "capsule_radius_scale": 0.2,
    "capsule_extent_by_link": {"link_left_arm_5": -0.10, "link_right_arm_5": -0.10},
}
EXEC_H = 8
APPROACH_T = 200
MARGINS_MM = (10.0, 5.0, 2.0, 1.0, 0.0)
GRIPPER_L = ("ee_finger_l1", "ee_finger_l2", "ee_left")
TRUST_RADIUS_RAD = 0.15          # AG3S/trajopt/config.py:286
LEFT_ARM = tuple(f"left_arm_{i}" for i in range(7))


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    mesh_cache: dict = {}
    scene = TransportScene.attach(m, d)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS,
                                         sphere_options=SPHERE_OPTS)
    link_names = list(resolve_link_names(robot, robot.n_spheres))
    _, radii = robot.sphere_centers_numeric(np.zeros(robot.nq))
    radii = np.asarray(radii, np.float64).reshape(-1)
    S = radii.size

    def collidable(g):
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0

    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_BODIES}
    scene_geoms = [g for g in range(m.ngeom)
                   if collidable(g) and int(m.geom_bodyid[g]) in set(bid.values())]
    geom_body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
                 for g in scene_geoms]
    bodies = sorted(set(geom_body))
    body_of_geom = np.asarray([bodies.index(b) for b in geom_body], np.int64)
    G = len(scene_geoms)
    ai = bodies.index("apple")
    grip_mask = np.asarray([n in GRIPPER_L for n in link_names], bool)
    fing_mask = np.asarray([n in ("ee_finger_l1", "ee_finger_l2") for n in link_names], bool)
    print(f"{S} spheres, {len(robot.capsules)} capsules, bodies {bodies}", flush=True)

    def q_from(row, qpos):
        return np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(row[ACT_COL[n]])
                           for n in JOINT_ORDER], np.float64)

    def clearance_tensor(P):
        """(F, 2, EXEC_H, S, B) mm — 0=reference, 1=refined."""
        F = len(P)
        out = np.full((F, 2, EXEC_H, S, len(bodies)), np.nan, np.float32)
        for i, r in enumerate(P):
            qpos = np.asarray(r["qpos"], np.float64)
            d.qpos[: qpos.size] = qpos
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            for vi, key in enumerate(("actions_reference", "actions")):
                arr = np.asarray(r[key], np.float64)
                for s in range(EXEC_H):
                    c = np.asarray(robot.sphere_centers_numeric(q_from(arr[s], qpos))[0],
                                   np.float64).reshape(-1, 3)
                    dist = np.empty((G, S), np.float64)
                    for a0 in range(0, S, 1200):
                        dist[:, a0 : a0 + 1200] = surface_distances(
                            m, d, scene_geoms, c[a0 : a0 + 1200], mesh_cache)
                    cl = (dist.T - radii[:, None])          # (S, G)
                    for bi in range(len(bodies)):
                        out[i, vi, s, :, bi] = cl[:, body_of_geom == bi].min(axis=1) * 1000.0
        return out

    t0 = time.time()
    data = {}
    for run, path in RUNS.items():
        P = [json.loads(l) for l in open(path) if json.loads(l).get("kind") == "planning"]
        T = clearance_tensor(P)
        data[run] = {"P": P, "T": T,
                     "t": np.asarray([int(r["t_step"]) for r in P], np.int64)}
        print(f"  {run}: {len(P)} chunks, tensor {T.shape} [{time.time()-t0:5.1f} s]", flush=True)

    out: dict = {
        "runs": {k: v for k, v in RUNS.items()},
        "method": {
            "offline_rebuild": "기록된 qpos (물체 free joint 포함) + chunk action -> 제약 구 "
                               f"{S} 개 중심 -> MuJoCo 충돌 geom 표면까지 해석적 거리. "
                               "mj_geomDistance 미사용. T9a/T9b 와 같은 방법.",
            "sphere_options": SPHERE_OPTS, "n_spheres": int(S),
            "n_capsules": len(robot.capsules), "scene_bodies": bodies,
            "clearance_definition": "surface_distance(sphere centre, nearest geom of body) - "
                                    "sphere radius.  PRE-margin — esdf margin 10 mm 은 "
                                    "빼지 않았다 (기록의 max_violation_pair.clearance_m 은 "
                                    "post-margin 이다, T9a 참조)",
            "exec_window": EXEC_H, "approach_window": f"t_step <= {APPROACH_T}",
            "gripper_links": list(GRIPPER_L),
            "trust_radius_rad": TRUST_RADIUS_RAD,
            "trust_radius_source": "AG3S/trajopt/config.py:286",
        },
    }

    # ================= P1 — 손가락을 막는 body =============================================
    p1 = {}
    for run in RUNS:
        P, T, t = data[run]["P"], data[run]["T"], data[run]["t"]
        app = t <= APPROACH_T
        res = {}
        for vi, vname in ((0, "reference"), (1, "refined")):
            sub = T[app, vi][:, :, grip_mask, :]                 # (Fa, H, Sg, B)
            for tag, sel in (("all_bodies", np.ones(len(bodies), bool)),
                             ("apple_excluded", np.arange(len(bodies)) != ai)):
                v = sub[:, :, :, sel]
                names = [b for k, b in enumerate(bodies) if sel[k]]
                per_chunk_body = v.min(axis=(1, 2))              # (Fa, Bsel)
                near = per_chunk_body.argmin(axis=1)
                cnt = collections.Counter(names[int(k)] for k in near)
                res[f"{vname}__{tag}"] = {
                    "nearest_body_counts_per_chunk": dict(cnt.most_common()),
                    "n_chunks": int(app.sum()),
                    "min_mm_by_body": {names[k]: round(float(per_chunk_body[:, k].min()), 4)
                                       for k in range(len(names))},
                    "global_min_mm": round(float(v.min()), 4),
                    "global_min_body": names[int(np.unravel_index(
                        int(v.argmin()), v.shape)[3])],
                    "global_min_chunk_t_step": int(t[app][int(np.unravel_index(
                        int(v.argmin()), v.shape)[0])]),
                    "per_chunk": [
                        {"t_step": int(t[app][i]),
                         "nearest_body": names[int(near[i])],
                         "mm": round(float(per_chunk_body[i, int(near[i])]), 4),
                         "by_body_mm": {names[k]: round(float(per_chunk_body[i, k]), 4)
                                        for k in range(len(names))}}
                        for i in range(int(app.sum()))],
                }
        # 기록의 max_violation_pair query 를 오프라인으로 이름 짓는다
        ident = []
        for i, r in enumerate(P):
            if t[i] > APPROACH_T:
                continue
            mvp = r.get("max_violation_pair") or {}
            qi, st = mvp.get("query"), mvp.get("step")
            if qi is None or st is None or int(st) >= EXEC_H:
                continue
            qi, st = int(qi), int(st)
            if qi >= S:
                # 질의점 배열은 로봇 구 뒤에 **attached 점**이 붙는다 (T9a 에서 본
                # `attached:ee_finger_l1[k]` 라벨). 그 행은 로봇 구가 아니라 쥔 것으로 친
                # 점이므로 오프라인 구 tensor 로 이름 지을 수 없다.
                ident.append({
                    "t_step": int(t[i]), "seq": int(r["seq"]),
                    "recorded_link": mvp.get("link"),
                    "recorded_obstacle": mvp.get("obstacle"),
                    "recorded_clearance_mm_post_margin": round(
                        float(mvp["clearance_m"]) * 1000.0, 4),
                    "sphere_index": qi, "sphere_link_offline": None,
                    "link_name_matches": None, "step": st,
                    "is_attached_query_point": True,
                    "nearest_body_all": None, "nearest_mm_all": None,
                    "nearest_body_apple_excluded": None,
                    "nearest_mm_apple_excluded": None, "by_body_mm": None,
                    "note": f"query {qi} >= {S} robot spheres — attached point row"})
                continue
            row = T[i, 1, st, qi]                                # refined = 실행된 것
            k_all = int(np.nanargmin(row))
            row_na = row.copy()
            row_na[ai] = np.inf
            k_na = int(np.nanargmin(row_na))
            ident.append({
                "t_step": int(t[i]), "seq": int(r["seq"]),
                "recorded_link": mvp.get("link"),
                "recorded_obstacle": mvp.get("obstacle"),
                "recorded_clearance_mm_post_margin": round(
                    float(mvp["clearance_m"]) * 1000.0, 4),
                "sphere_index": qi, "sphere_link_offline": link_names[qi],
                "link_name_matches": bool(link_names[qi] == mvp.get("link")),
                "step": st,
                "is_attached_query_point": False,
                "nearest_body_all": bodies[k_all],
                "nearest_mm_all": round(float(row[k_all]), 4),
                "nearest_body_apple_excluded": bodies[k_na],
                "nearest_mm_apple_excluded": round(float(row_na[k_na]), 4),
                "by_body_mm": {b: round(float(row[k]), 4) for k, b in enumerate(bodies)},
                # 기록의 ESDF 거리와 오프라인 참값 거리를 나란히.
                # clearance_m 은 post-margin (T9a) 이므로 margin 을 되더하면 d_esdf - r 이다.
                "esdf_d_minus_r_mm": round(float(mvp["clearance_m"]) * 1000.0 + 10.0, 4),
                "offline_minus_esdf_mm_all_bodies": round(
                    float(row[k_all]) - (float(mvp["clearance_m"]) * 1000.0 + 10.0), 4),
                "offline_minus_esdf_mm_apple_excluded": round(
                    float(row_na[k_na]) - (float(mvp["clearance_m"]) * 1000.0 + 10.0), 4),
            })
        res["recorded_worst_row_named_offline"] = {
            "note": "기록의 max_violation_pair 는 obstacle 이 None 이다 (label 층이 destination "
                    "과 attached 만 이름 짓는다). 그 행의 query 구를 오프라인 tensor 로 "
                    "이름 지은 것이다. link 이름이 기록과 같은지도 확인했다.",
            "n_rows": len(ident),
            "n_rows_on_a_robot_sphere": int(sum(
                1 for x in ident if not x["is_attached_query_point"])),
            "n_rows_on_an_attached_point": int(sum(
                1 for x in ident if x["is_attached_query_point"])),
            "n_link_name_matches": int(sum(1 for x in ident if x["link_name_matches"])),
            "nearest_body_counts_all": dict(collections.Counter(
                x["nearest_body_all"] for x in ident
                if not x["is_attached_query_point"]).most_common()),
            "nearest_body_counts_apple_excluded": dict(collections.Counter(
                x["nearest_body_apple_excluded"] for x in ident
                if not x["is_attached_query_point"]).most_common()),
            "esdf_vs_ground_truth": {
                "definition": "esdf_d_minus_r = recorded clearance_m + margin 10 mm "
                              "(clearance_m 은 post-margin, T9a). offline 은 같은 구에서 "
                              "MuJoCo 참값 body 표면까지의 거리 - 반지름.",
                "n_rows": int(sum(1 for x in ident if not x["is_attached_query_point"])),
                "offline_minus_esdf_mm_all_bodies": {
                    "min": round(min(x["offline_minus_esdf_mm_all_bodies"] for x in ident
                                     if not x["is_attached_query_point"]), 4),
                    "max": round(max(x["offline_minus_esdf_mm_all_bodies"] for x in ident
                                     if not x["is_attached_query_point"]), 4),
                    "median": round(float(np.median([
                        x["offline_minus_esdf_mm_all_bodies"] for x in ident
                        if not x["is_attached_query_point"]])), 4)},
                "n_rows_offline_further_than_esdf": int(sum(
                    1 for x in ident if not x["is_attached_query_point"]
                    and x["offline_minus_esdf_mm_all_bodies"] > 0)),
                "note": "두 값이 다르면 그 행을 묶은 표면이 MuJoCo 참값 body 가 아니라는 "
                        "뜻이다. ESDF 는 카메라 점구름으로 만든 것이라 참값 씬과 같지 않다. "
                        "어느 쪽이 옳은지는 여기서 판정하지 않는다.",
            },
            "rows": ident,
        }
        p1[run] = res
    out["p1_what_blocks_the_fingers"] = p1

    # ================= P2 — margin sweep ==================================================
    run = "T10"
    P, T, t = data[run]["P"], data[run]["T"], data[run]["t"]
    app = t <= APPROACH_T
    ref = T[app, 0][:, :, grip_mask, :]                          # 정책 원본, gripper 구만
    sel_na = np.arange(len(bodies)) != ai
    p2 = {"window": f"approach chunks (t_step <= {APPROACH_T}), {int(app.sum())} chunks x "
                    f"{EXEC_H} steps, gripper spheres only, POLICY REFERENCE chunk",
          "clearance_is_pre_margin": True, "sweeps": {}}
    for tag, sel in (("all_bodies", np.ones(len(bodies), bool)),
                     ("apple_excluded", sel_na)):
        v = ref[:, :, :, sel]
        names = [b for k, b in enumerate(bodies) if sel[k]]
        rows = {}
        for mg in MARGINS_MM:
            slack = v - mg
            le0 = slack <= 0.0
            pairs = collections.Counter()
            for k, b in enumerate(names):
                n = int((v[:, :, :, k] - mg <= 0.0).sum())
                if n:
                    pairs[b] = n
            rows[f"{mg:g}mm"] = {
                "n_rows_slack_le0": int(le0.sum()), "n_rows_total": int(v.size),
                "n_chunks_with_any": int(le0.any(axis=(1, 2, 3)).sum()),
                "n_chunks": int(app.sum()),
                "min_slack_mm": round(float(v.min() - mg), 4),
                "feasible": bool(le0.sum() == 0),
                "rows_by_body": dict(pairs.most_common()),
            }
        crit = float(v.min())
        rows["critical_margin_mm"] = {
            "value": round(crit, 4),
            "meaning": "정책 원본 궤적의 gripper 구가 이 body 집합에 대해 가지는 최소 "
                       "clearance (pre-margin). margin 을 이 값 아래로 내리면 활성 행이 0 이 된다.",
            "argmin_body": names[int(np.unravel_index(int(v.argmin()), v.shape)[3])],
            "argmin_t_step": int(t[app][int(np.unravel_index(int(v.argmin()), v.shape)[0])]),
            "argmin_link": [n for n, g in zip(link_names, grip_mask) if g][
                int(np.unravel_index(int(v.argmin()), v.shape)[2])],
        }
        p2["sweeps"][tag] = rows
    out["p2_margin_sweep"] = p2

    # ================= P3 — trust region ==================================================
    p3 = {"trust_radius_rad": TRUST_RADIUS_RAD, "source": "AG3S/trajopt/config.py:286",
          "runs": {}}
    for run in RUNS:
        P, t = data[run]["P"], data[run]["t"]
        REFI = np.asarray([r["actions_reference"] for r in P], np.float64)
        TOUT = np.asarray([r["actions"] for r in P], np.float64)
        dq = TOUT[:, :EXEC_H] - REFI[:, :EXEC_H]
        cols = [ACT_COL[j] for j in LEFT_ARM]
        mx = np.abs(dq[:, :, cols]).max(axis=(1, 2))             # (F,) rad
        ratio = mx / TRUST_RADIUS_RAD
        app = t <= APPROACH_T
        p3["runs"][run] = {
            "n_chunks": len(P),
            "max_abs_dq_rad_max": round(float(mx.max()), 6),
            "max_abs_dq_deg_max": round(float(np.degrees(mx.max())), 4),
            "argmax_chunk": int(mx.argmax()), "argmax_t_step": int(t[int(mx.argmax())]),
            "ratio_to_trust_radius_max": round(float(ratio.max()), 6),
            "n_chunks_ratio_gt_0.95": int((ratio > 0.95).sum()),
            "n_chunks_ratio_gt_0.50": int((ratio > 0.50).sum()),
            "n_chunks_ratio_gt_0.10": int((ratio > 0.10).sum()),
            "median_ratio": round(float(np.median(ratio)), 6),
            "approach_window": {
                "n_chunks": int(app.sum()),
                "max_ratio": round(float(ratio[app].max()), 6),
                "median_ratio": round(float(np.median(ratio[app])), 6),
                "n_chunks_ratio_gt_0.95": int((ratio[app] > 0.95).sum()),
                "n_chunks_ratio_gt_0.50": int((ratio[app] > 0.50).sum()),
                "max_abs_dq_deg": round(float(np.degrees(mx[app].max())), 4),
            },
            "per_chunk_ratio": [round(float(v), 6) for v in ratio],
            "per_chunk_max_deg": [round(float(np.degrees(v)), 4) for v in mx],
        }
    # chunk 35 (T10) 상세
    P, T, t = data["T10"]["P"], data["T10"]["T"], data["T10"]["t"]
    ic = int(np.asarray(p3["runs"]["T10"]["per_chunk_ratio"]).argmax())
    r = P[ic]
    mvp = r.get("max_violation_pair") or {}
    per_joint = {j: round(float(np.degrees(np.abs(
        np.asarray(r["actions"], np.float64)[:EXEC_H, ACT_COL[j]]
        - np.asarray(r["actions_reference"], np.float64)[:EXEC_H, ACT_COL[j]]).max())), 4)
        for j in LEFT_ARM}
    tight = T[ic, 1].min(axis=(0, 2))                            # (S,) chunk 안 최소
    order = np.argsort(tight)[:8]
    p3["outlier_chunk_detail"] = {
        "run": "T10", "chunk": ic, "t_step": int(t[ic]), "seq": int(r["seq"]),
        "ratio_to_trust_radius": p3["runs"]["T10"]["per_chunk_ratio"][ic],
        "per_joint_max_abs_deg": per_joint,
        "n_joints_within_0.01deg_of_the_max": int(sum(
            1 for v in per_joint.values() if abs(v - max(per_joint.values())) < 0.01)),
        "max_deg": max(per_joint.values()),
        "max_rad": round(float(np.radians(max(per_joint.values()))), 6),
        "recorded": {"safe": r.get("safe"),
                     "trajopt_status": r["verdict"].get("trajopt_status"),
                     "max_violation_m": r["verdict"].get("max_violation_m"),
                     "notes": r["verdict"].get("notes", []),
                     "hold_reason": r.get("hold_reason", ""),
                     "mvp_link": mvp.get("link"), "mvp_obstacle": mvp.get("obstacle"),
                     "mvp_clearance_mm_post_margin": (
                         None if mvp.get("clearance_m") is None
                         else round(float(mvp["clearance_m"]) * 1000.0, 4)),
                     "timing_ms": r.get("timing_ms")},
        "tightest_offline_rows": [
            {"sphere": int(k), "link": link_names[int(k)],
             "min_clearance_mm": round(float(tight[int(k)]), 4),
             "nearest_body": bodies[int(np.nanargmin(T[ic, 1].min(axis=0)[int(k)]))]}
            for k in order],
    }
    out["p3_trust_region"] = p3

    # ================= P4 — T9 대 T10 =====================================================
    p4 = {"axes": "per chunk: min over the 8 exec steps", "runs": {}}
    for run in RUNS:
        P, T, t = data[run]["P"], data[run]["T"], data[run]["t"]
        tip_ref = T[:, 0][:, :, fing_mask, ai].min(axis=(1, 2))
        tip_ref_o = T[:, 1][:, :, fing_mask, ai].min(axis=(1, 2))
        sel_na = np.arange(len(bodies)) != ai
        grip_na = T[:, 1][:, :, grip_mask][:, :, :, sel_na]
        names = [b for k, b in enumerate(bodies) if sel_na[k]]
        near = grip_na.min(axis=(1, 2)).argmin(axis=1)
        app = t <= APPROACH_T
        p4["runs"][run] = {
            "n_chunks": len(P), "t_step": [int(v) for v in t],
            "tip_to_apple_reference_mm": [round(float(v), 4) for v in tip_ref],
            "tip_to_apple_refined_mm": [round(float(v), 4) for v in tip_ref_o],
            "tip_to_apple_reference_min_mm": round(float(tip_ref.min()), 4),
            "tip_to_apple_refined_min_mm": round(float(tip_ref_o.min()), 4),
            "tip_to_apple_reference_min_t_step": int(t[int(tip_ref.argmin())]),
            "tip_to_apple_refined_min_t_step": int(t[int(tip_ref_o.argmin())]),
            "nearest_body_refined_apple_excluded": [names[int(k)] for k in near],
            "nearest_body_counts_approach": dict(collections.Counter(
                names[int(k)] for k in near[app]).most_common()),
            "to_deviation_deg_per_chunk": p3["runs"][run]["per_chunk_max_deg"],
            "to_deviation_deg_median_all": round(float(np.median(
                p3["runs"][run]["per_chunk_max_deg"])), 4),
            "to_deviation_deg_median_approach": round(float(np.median(
                np.asarray(p3["runs"][run]["per_chunk_max_deg"])[app])), 4),
        }
    out["p4_t9_vs_t10"] = p4

    # 사용자/lead 가 이미 잰 것의 재확인
    P10 = data["T10"]["P"]
    apple0 = np.asarray(P10[0]["object_poses"]["apple"]["pos"], np.float64)
    disp = np.asarray([np.asarray(r["object_poses"]["apple"]["pos"], np.float64) - apple0
                       for r in P10]) * 1000.0
    out["recheck_of_lead_numbers"] = {
        "apple_max_lift_mm": round(float(disp[:, 2].max()), 4),
        "apple_max_total_displacement_mm": round(float(np.linalg.norm(disp, axis=1).max()), 4),
        "n_chunks_safe": sum(1 for r in P10 if r.get("safe")),
        "n_chunks": len(P10),
        "max_violation_m_max": max(float(r["verdict"].get("max_violation_m") or 0.0)
                                   for r in P10),
        "to_deviation_median_deg_all_T10": p4["runs"]["T10"]["to_deviation_deg_median_all"],
        "to_deviation_median_deg_approach_T10":
            p4["runs"]["T10"]["to_deviation_deg_median_approach"],
        "to_deviation_median_deg_all_T9": p4["runs"]["T9"]["to_deviation_deg_median_all"],
        "to_deviation_median_deg_approach_T9":
            p4["runs"]["T9"]["to_deviation_deg_median_approach"],
        "note": "lead 가 이미 낸 값의 재확인이다 (들림 0.0200 mm, 변위 0.5431 mm, safe 66/66, "
                "중앙값 1.060 -> 0.581, 접근 구간 1.060 -> 0.945)",
    }
    out["elapsed_s"] = round(time.time() - t0, 1)
    JSON_OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(
        NPZ, bodies=np.asarray(bodies), link_names=np.asarray(link_names), radii_m=radii,
        **{f"{run}__tensor": data[run]["T"] for run in RUNS},
        **{f"{run}__t": data[run]["t"] for run in RUNS})

    print("\n=== P1 T10 (refined, apple excluded) nearest body in the approach window ===")
    print(p1["T10"]["refined__apple_excluded"]["nearest_body_counts_per_chunk"])
    print("min by body:", p1["T10"]["refined__apple_excluded"]["min_mm_by_body"])
    print("=== P1 recorded worst rows named offline (T10) ===")
    for x in p1["T10"]["recorded_worst_row_named_offline"]["rows"][:16]:
        if x["is_attached_query_point"]:
            print(f"  t={x['t_step']:4d} [attached query point {x['sphere_index']}] "
                  f"rec {x['recorded_clearance_mm_post_margin']:8.3f} mm  "
                  f"link {x['recorded_link']}")
            continue
        print(f"  t={x['t_step']:4d} {x['sphere_link_offline']:14s} "
              f"rec {x['recorded_clearance_mm_post_margin']:8.3f} "
              f"-> nearest {x['nearest_body_all']:7s} {x['nearest_mm_all']:8.3f} | "
              f"no apple {x['nearest_body_apple_excluded']:7s} "
              f"{x['nearest_mm_apple_excluded']:8.3f}")
    print("=== P1 esdf vs ground truth (T10) ===")
    print(json.dumps(p1["T10"]["recorded_worst_row_named_offline"]["esdf_vs_ground_truth"],
                     indent=1, ensure_ascii=False))
    print("=== P1 T10 refined, ALL bodies ===")
    print(p1["T10"]["refined__all_bodies"]["nearest_body_counts_per_chunk"],
          p1["T10"]["refined__all_bodies"]["min_mm_by_body"])
    print("=== P2 (all bodies) ===")
    print(json.dumps(p2["sweeps"]["all_bodies"], indent=1, ensure_ascii=False)[:1400])
    print("=== P2 (apple excluded) ===")
    for k, v in p2["sweeps"]["apple_excluded"].items():
        print(" ", k, v if k == "critical_margin_mm" else
              {kk: v[kk] for kk in ("n_rows_slack_le0", "n_chunks_with_any", "min_slack_mm",
                                    "feasible", "rows_by_body")})
    print("=== P3 ===")
    print(json.dumps({r: {k: v for k, v in p3["runs"][r].items()
                          if not k.startswith("per_chunk")} for r in RUNS},
                     indent=1, ensure_ascii=False))
    print("wrote", JSON_OUT, NPZ, f"[{out['elapsed_s']} s]")


if __name__ == "__main__":
    main()
