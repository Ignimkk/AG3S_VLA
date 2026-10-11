"""T9b N1..N5 — 정책 원본(`actions_reference`) 대 TO 출력(`actions`). rollout 없다.

방법은 T9a `m0_state_tensor.py` 와 같다: 기록된 `qpos` 로 씬을 놓고 (물체 free joint 포함),
chunk 의 action 행으로 제약 구 266 개 중심을 FK 로 구한 뒤, MuJoCo 충돌 geom **표면까지
해석적 거리**를 잰다. `mj_geomDistance` 를 쓰지 않는다.

N1 link 별 사과까지 거리와 각 chunk 의 최근접 link (두 변형)
N2 손끝이 가려던 곳 대 실제로 간 곳 (mm, x·y·z)
N3 원본을 50 step 끝까지 따라가면 사과에 닿는가
N4 어느 관절이 손끝을 가장 많이 옮겼나 — 도가 아니라 **손끝 변위 기여 (mm)**, MuJoCo Jacobian
N5 gripper 열이 bit 단위로 같은가 (75 x 50 전 원소)
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))

from m3_to_refined_vs_reference import (  # noqa: E402
    ACT_COL, JOINT_ORDER, MODEL_XML, QADR, SCENE_BODIES, surface_distances)

REC = _ag3s_workspace_path('outputs/live_test/20260926_t9/execute_ep1807/frames.jsonl')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T9b'))
JSON_OUT = OUT / "n1to5.json"
NPZ = OUT / "n1to5_curves.npz"
SPHERE_OPTS = {
    "sphere_spacing": 0.4, "max_spheres_per_capsule": 32, "capsule_radius_scale": 0.2,
    "capsule_extent_by_link": {"link_left_arm_5": -0.10, "link_right_arm_5": -0.10},
}
EXEC_H = 8
GRIP_COLS = (7, 15)
LEFT_ARM = tuple(f"left_arm_{i}" for i in range(7))
RIGHT_ARM = tuple(f"right_arm_{i}" for i in range(7))
TIP_BODIES = ("ee_finger_l1", "ee_finger_l2")


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
    uniq = sorted(set(link_names))
    link_of = np.asarray([uniq.index(n) for n in link_names], np.int64)
    print(f"constraint model {S} spheres, {len(robot.capsules)} capsules "
          f"(T9a matched the server at 266/36)", flush=True)

    def collidable(g):
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0

    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_BODIES}
    scene_geoms = [g for g in range(m.ngeom)
                   if collidable(g) and int(m.geom_bodyid[g]) in set(bid.values())]
    geom_body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
                 for g in scene_geoms]
    apple_geoms = np.asarray([i for i, b in enumerate(geom_body) if b == "apple"], np.int64)
    G = len(scene_geoms)

    tip_ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in TIP_BODIES}
    apple_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
    dof = {}
    for jn in LEFT_ARM + RIGHT_ARM:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        dof[jn] = int(m.jnt_dofadr[jid])

    recs = [json.loads(l) for l in open(REC)]
    P = [r for r in recs if r["kind"] == "planning"]
    F = len(P)
    H = int(np.asarray(P[0]["actions"]).shape[0])
    REFI = np.asarray([r["actions_reference"] for r in P], np.float64)   # (F, H, 16)
    TOUT = np.asarray([r["actions"] for r in P], np.float64)
    print(f"{F} chunks x {H} steps, exec window {EXEC_H}", flush=True)

    def q_from(row, qpos):
        return np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(row[ACT_COL[n]])
                           for n in JOINT_ORDER], np.float64)

    def set_arms(qpos, row):
        d.qpos[: qpos.size] = qpos
        for n, c in ACT_COL.items():
            d.qpos[QADR[n]] = float(row[c])
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    # ---------------- N5 — gripper 열 bit 단위 (가장 싸다) --------------------------------
    n5 = {}
    for c in GRIP_COLS:
        a, b = TOUT[:, :, c], REFI[:, :, c]
        same = a == b
        n5[f"column_{c}"] = {
            "n_elements": int(a.size), "n_bitwise_identical": int(same.sum()),
            "all_identical": bool(same.all()),
            "max_abs_difference": float(np.abs(a - b).max()),
            "n_differing_elements": int((~same).sum()),
            "value_min": float(b.min()), "value_max": float(b.max()),
        }
    n5["window"] = f"all {F} chunks x {H} steps (not just the 8-step execution window)"
    n5["comparison"] = "numpy elementwise == on float64 parsed from the record"
    print("N5", {k: v["all_identical"] for k, v in n5.items() if k.startswith("column")},
          flush=True)

    # 좌우 열이 제 자리인지 확인 (lead 의 도 단위 결과 재확인만)
    dq = TOUT - REFI
    n5["sanity_joint_deltas_deg"] = {
        "left_arm_max_abs_deg_exec8": [round(float(np.degrees(np.abs(
            dq[:, :EXEC_H, ACT_COL[j]]).max())), 4) for j in LEFT_ARM],
        "right_arm_max_abs_deg_exec8": [round(float(np.degrees(np.abs(
            dq[:, :EXEC_H, ACT_COL[j]]).max())), 4) for j in RIGHT_ARM],
        "per_chunk_max_abs_deg_median_exec8": round(float(np.median(np.degrees(np.abs(
            dq[:, :EXEC_H, [ACT_COL[j] for j in LEFT_ARM]]).max(axis=(1, 2))))), 4),
        "note": "lead 가 이미 잰 값의 재확인이다 (L0~L6 4.3~5.8°, 중앙값 1.06°, 오른팔 0.04~0.10°)",
    }

    # ---------------- N1 · N2 · N4 ------------------------------------------------------
    L = len(uniq)
    per_link = {v: np.full((F, EXEC_H, L), np.nan) for v in ("reference", "refined")}
    tip_xyz = {v: np.full((F, EXEC_H, len(TIP_BODIES), 3), np.nan) for v in
               ("reference", "refined")}
    tip_apple = {v: np.full((F, EXEC_H), np.nan) for v in ("reference", "refined")}
    apple_pos = np.full((F, 3), np.nan)
    jac_contrib = np.full((F, EXEC_H, len(LEFT_ARM)), np.nan)      # mm, |J_j * dq_j|
    jac_pred = np.full((F, EXEC_H, 3), np.nan)                     # mm, sum_j J_j dq_j
    t0 = time.time()
    for i, r in enumerate(P):
        qpos = np.asarray(r["qpos"], np.float64)
        d.qpos[: qpos.size] = qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        apple_pos[i] = d.xpos[apple_bid]
        for vname, arr in (("reference", REFI), ("refined", TOUT)):
            for s in range(EXEC_H):
                c = np.asarray(robot.sphere_centers_numeric(q_from(arr[i, s], qpos))[0],
                               np.float64).reshape(-1, 3)
                dist = np.empty((G, S), np.float64)
                B = 1200
                for a0 in range(0, S, B):
                    dist[:, a0 : a0 + B] = surface_distances(
                        m, d, scene_geoms, c[a0 : a0 + B], mesh_cache)
                cl = dist[apple_geoms].min(axis=0) - radii          # (S,) clearance to apple
                for li in range(L):
                    per_link[vname][i, s, li] = cl[link_of == li].min() * 1000.0
                fin = np.asarray([n in TIP_BODIES for n in link_names], bool)
                tip_apple[vname][i, s] = cl[fin].min() * 1000.0
                set_arms(qpos, arr[i, s])
                for k, n in enumerate(TIP_BODIES):
                    tip_xyz[vname][i, s, k] = d.xpos[tip_ids[n]]
        # N4 — reference 자세에서의 Jacobian 으로 관절별 손끝 변위 기여
        for s in range(EXEC_H):
            set_arms(qpos, REFI[i, s])
            jacp = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jacp, None, tip_ids["ee_finger_l1"])
            delta = np.zeros(3)
            for k, jn in enumerate(LEFT_ARM):
                col = jacp[:, dof[jn]]
                dqj = float(TOUT[i, s, ACT_COL[jn]] - REFI[i, s, ACT_COL[jn]])
                jac_contrib[i, s, k] = np.linalg.norm(col * dqj) * 1000.0
                delta += col * dqj
            jac_pred[i, s] = delta * 1000.0
        if i % 15 == 0 or i == F - 1:
            print(f"  chunk {i+1}/{F} tip-apple ref {tip_apple['reference'][i].min():8.2f} "
                  f"to {tip_apple['refined'][i].min():8.2f} mm  [{time.time()-t0:5.1f} s]",
                  flush=True)

    # --- N1 표 -------------------------------------------------------------------------
    def nearest(vname):
        best = per_link[vname].min(axis=1)              # (F, L) chunk 안 최소
        arg = best.argmin(axis=1)
        return best, np.asarray([uniq[int(a)] for a in arg])

    b_ref, near_ref = nearest("reference")
    b_ref2, near_ref2 = b_ref, near_ref
    b_ref_ref = b_ref
    b_ref_ref2 = b_ref
    b_r, near_r = nearest("reference")
    b_t, near_t = nearest("refined")
    import collections
    n1 = {
        "metric": "per link, min over the 8 execution steps of "
                  "(surface distance to apple - sphere radius) [mm]",
        "link_axis": uniq,
        "nearest_link_counts_reference": dict(collections.Counter(near_r).most_common()),
        "nearest_link_counts_refined": dict(collections.Counter(near_t).most_common()),
        "n_chunks": F,
        "n_chunks_same_nearest_link": int((near_r == near_t).sum()),
        "n_chunks_different_nearest_link": int((near_r != near_t).sum()),
        "chunks_where_nearest_link_differs": [
            {"chunk": int(k), "t_step": int(P[int(k)]["t_step"]),
             "reference_nearest": str(near_r[k]), "reference_mm": round(float(b_r[k].min()), 4),
             "refined_nearest": str(near_t[k]), "refined_mm": round(float(b_t[k].min()), 4)}
            for k in np.nonzero(near_r != near_t)[0]],
        "global_min_mm_reference": round(float(b_r.min()), 4),
        "global_min_link_reference": str(near_r[int(b_r.min(axis=1).argmin())]),
        "global_min_chunk_reference": int(b_r.min(axis=1).argmin()),
        "global_min_mm_refined": round(float(b_t.min()), 4),
        "global_min_link_refined": str(near_t[int(b_t.min(axis=1).argmin())]),
        "global_min_chunk_refined": int(b_t.min(axis=1).argmin()),
        "per_link_min_over_all_chunks_mm": {
            uniq[li]: {"reference": round(float(b_r[:, li].min()), 4),
                       "refined": round(float(b_t[:, li].min()), 4),
                       "refined_minus_reference": round(float(b_t[:, li].min()
                                                              - b_r[:, li].min()), 4)}
            for li in range(L)},
        "closest_chunk_detail": None,
    }
    ic = int(b_r.min(axis=1).argmin())
    order = np.argsort(b_r[ic])
    n1["closest_chunk_detail"] = {
        "chunk": ic, "t_step": int(P[ic]["t_step"]),
        "ranking_reference": [{"link": uniq[int(li)], "mm": round(float(b_r[ic, li]), 4)}
                              for li in order[:6]],
        "ranking_refined": [{"link": uniq[int(li)], "mm": round(float(b_t[ic, li]), 4)}
                            for li in np.argsort(b_t[ic])[:6]],
    }

    # --- N2 ----------------------------------------------------------------------------
    diff = (tip_xyz["reference"] - tip_xyz["refined"]) * 1000.0      # (F, H, 2, 3) mm
    norm = np.linalg.norm(diff, axis=3)
    n2 = {
        "metric": "ee_finger_l1/l2 body origin, reference minus refined [mm]",
        "bodies": list(TIP_BODIES),
        "max_abs_mm": round(float(norm.max()), 4),
        "max_abs_chunk": int(np.unravel_index(int(norm.argmax()), norm.shape)[0]),
        "max_abs_step": int(np.unravel_index(int(norm.argmax()), norm.shape)[1]),
        "median_over_chunks_of_max_mm": round(float(np.median(norm.max(axis=(1, 2)))), 4),
        "mean_mm": round(float(norm.mean()), 4),
        "component_max_abs_mm": {"x": round(float(np.abs(diff[..., 0]).max()), 4),
                                 "y": round(float(np.abs(diff[..., 1]).max()), 4),
                                 "z": round(float(np.abs(diff[..., 2]).max()), 4)},
        "component_mean_signed_mm": {"x": round(float(diff[..., 0].mean()), 4),
                                     "y": round(float(diff[..., 1].mean()), 4),
                                     "z": round(float(diff[..., 2].mean()), 4)},
        "per_step_mean_mm": [round(float(norm[:, s].mean()), 4) for s in range(EXEC_H)],
        "tip_to_apple_mm": {
            "reference_min": round(float(np.nanmin(tip_apple["reference"])), 4),
            "refined_min": round(float(np.nanmin(tip_apple["refined"])), 4),
            "refined_minus_reference_at_the_reference_min_chunk": round(
                float(np.nanmin(tip_apple["refined"][int(np.nanargmin(
                    tip_apple["reference"].min(axis=1)))])
                      - np.nanmin(tip_apple["reference"])), 4)},
    }

    # --- N3 — 원본 50 step 전체 ----------------------------------------------------------
    ic3 = int(np.nanargmin(tip_apple["reference"].min(axis=1)))
    qpos = np.asarray(P[ic3]["qpos"], np.float64)
    d.qpos[: qpos.size] = qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    fin = np.asarray([n in TIP_BODIES for n in link_names], bool)
    full = np.full((H, 2), np.nan)
    tip_full = np.full((H, 3), np.nan)
    for s in range(H):
        for vi, arr in enumerate((REFI, TOUT)):
            c = np.asarray(robot.sphere_centers_numeric(q_from(arr[ic3, s], qpos))[0],
                           np.float64).reshape(-1, 3)
            dist = np.empty((G, S), np.float64)
            for a0 in range(0, S, 1200):
                dist[:, a0 : a0 + 1200] = surface_distances(
                    m, d, scene_geoms, c[a0 : a0 + 1200], mesh_cache)
            full[s, vi] = (dist[apple_geoms].min(axis=0) - radii)[fin].min() * 1000.0
        set_arms(qpos, REFI[ic3, s])
        tip_full[s] = d.xpos[tip_ids["ee_finger_l1"]]
    n3 = {
        "chunk": ic3, "t_step": int(P[ic3]["t_step"]), "horizon": H,
        "metric": "min over left fingertip spheres of (surface distance to apple - radius)",
        "reference_min_mm_over_50_steps": round(float(np.nanmin(full[:, 0])), 4),
        "reference_argmin_step": int(np.nanargmin(full[:, 0])),
        "refined_min_mm_over_50_steps": round(float(np.nanmin(full[:, 1])), 4),
        "refined_argmin_step": int(np.nanargmin(full[:, 1])),
        "reference_min_mm_in_exec8": round(float(np.nanmin(full[:EXEC_H, 0])), 4),
        "refined_min_mm_in_exec8": round(float(np.nanmin(full[:EXEC_H, 1])), 4),
        "reference_below_0mm_steps": int((full[:, 0] <= 0).sum()),
        "refined_below_0mm_steps": int((full[:, 1] <= 0).sum()),
        "series_reference_mm": [round(float(v), 4) for v in full[:, 0]],
        "series_refined_mm": [round(float(v), 4) for v in full[:, 1]],
        "note": "물체 pose 는 이 chunk 시점 값으로 50 step 내내 고정된다 (기록에 미래 pose 가 없다)",
    }

    # --- N4 ------------------------------------------------------------------------------
    per_joint_mean = jac_contrib.mean(axis=(0, 1))
    per_joint_max = jac_contrib.max(axis=(0, 1))
    per_joint_sum = jac_contrib.sum(axis=(0, 1))
    share = per_joint_sum / max(per_joint_sum.sum(), 1e-12)
    actual = np.linalg.norm(diff[:, :, 0, :], axis=2)                # l1 실제 변위 mm
    pred = np.linalg.norm(jac_pred, axis=2)
    n4 = {
        "metric": "|J_j(q_reference) * (refined_j - reference_j)| at ee_finger_l1, in mm",
        "jacobian": "mujoco.mj_jacBody translational part at the reference pose of each step",
        "joints": list(LEFT_ARM),
        "mean_contribution_mm": {j: round(float(per_joint_mean[k]), 4)
                                 for k, j in enumerate(LEFT_ARM)},
        "max_contribution_mm": {j: round(float(per_joint_max[k]), 4)
                                for k, j in enumerate(LEFT_ARM)},
        "total_contribution_share": {j: round(float(share[k]), 6)
                                     for k, j in enumerate(LEFT_ARM)},
        "largest_mean_joint": LEFT_ARM[int(per_joint_mean.argmax())],
        "largest_max_joint": LEFT_ARM[int(per_joint_max.argmax())],
        "largest_share_joint": LEFT_ARM[int(share.argmax())],
        "degree_vs_mm": {
            "max_abs_deg_by_joint": {j: round(float(np.degrees(np.abs(
                dq[:, :EXEC_H, ACT_COL[j]]).max())), 4) for j in LEFT_ARM},
            "mm_per_deg_mean": {j: round(float(
                per_joint_mean[k] / max(np.degrees(np.abs(
                    dq[:, :EXEC_H, ACT_COL[j]]).mean()), 1e-12)), 4)
                for k, j in enumerate(LEFT_ARM)},
            "note": "도 단위 크기와 손끝 mm 기여를 나란히 놓은 것이다 — 같은 1° 라도 "
                    "어깨와 손목의 손끝 변위가 다르다는 것을 이 두 줄이 보인다",
        },
        "linearisation_check": {
            "mean_predicted_mm": round(float(pred.mean()), 4),
            "mean_actual_mm": round(float(actual.mean()), 4),
            "max_abs_residual_mm": round(float(np.abs(pred - actual).max()), 4),
            "note": "predicted 는 1차 Jacobian 합, actual 은 두 자세의 실제 FK 차이다",
        },
    }

    out = {
        "record": REC, "n_chunks": F, "horizon": H, "exec_window": EXEC_H,
        "method": {
            "offline_rebuild": "기록된 qpos (물체 free joint 포함) + chunk action -> 제약 구 "
                               "266 개 중심 -> MuJoCo 충돌 geom 표면까지 해석적 거리. "
                               "mj_geomDistance 미사용. T9a m0_state_tensor.py 와 같은 방법.",
            "sphere_options": SPHERE_OPTS,
            "n_spheres": int(S), "n_capsules": len(robot.capsules),
            "clearance_definition": "surface_distance(sphere centre, apple geom) - sphere radius "
                                    "(PRE-margin; esdf_margin 10 mm 는 빼지 않았다)",
            "fingertip_point": "ee_finger_l1 / ee_finger_l2 의 MuJoCo body origin "
                               "(구 중심이 아니라 link frame 원점)",
            "action_column_map": "left_arm_0..6 -> col 0..6, left gripper col 7, "
                                 "right_arm_0..6 -> col 8..14, right gripper col 15 "
                                 "(lead 가 types.py:193-198 / :120 / :142 로 확인)",
        },
        "n1_link_to_apple": n1, "n2_fingertip_divergence": n2,
        "n3_reference_full_horizon": n3, "n4_joint_contribution_mm": n4,
        "n5_gripper_columns": n5,
        "elapsed_s": round(time.time() - t0, 1),
    }
    JSON_OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(
        NPZ, link_axis=np.asarray(uniq),
        per_link_reference=per_link["reference"], per_link_refined=per_link["refined"],
        tip_xyz_reference=tip_xyz["reference"], tip_xyz_refined=tip_xyz["refined"],
        tip_apple_reference=tip_apple["reference"], tip_apple_refined=tip_apple["refined"],
        apple_pos=apple_pos, jac_contrib=jac_contrib, jac_pred=jac_pred,
        t_step=np.asarray([int(r["t_step"]) for r in P]),
        full_horizon=full, tip_full=tip_full, joints=np.asarray(LEFT_ARM))
    print("\nN1 nearest link (reference):", n1["nearest_link_counts_reference"])
    print("N1 nearest link (refined)  :", n1["nearest_link_counts_refined"])
    print("N1 chunks with a different nearest link:", n1["n_chunks_different_nearest_link"],
          "of", F)
    print("N2 max |ref - refined| fingertip:", n2["max_abs_mm"], "mm, components",
          n2["component_max_abs_mm"])
    print("N3 chunk", n3["chunk"], "reference over 50 steps:",
          n3["reference_min_mm_over_50_steps"], "mm, refined:",
          n3["refined_min_mm_over_50_steps"], "mm")
    print("N4 mean contribution mm:", n4["mean_contribution_mm"])
    print("N5 gripper identical:", {k: v["all_identical"] for k, v in n5.items()
                                    if k.startswith("column")})
    print("wrote", JSON_OUT, NPZ)


if __name__ == "__main__":
    main()
