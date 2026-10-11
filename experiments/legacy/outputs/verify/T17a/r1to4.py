"""T17a R1..R4 — 성공한 shadow 에서 TO 는 무엇에 밀리는가. rollout 없다.

읽는 것 (쓰지 않는다):
  로컬  outputs/live_test/20260928_t17/shadow_ep1807/frames.jsonl   (planning 75, 완주)
  서버  outputs/live_test/20260928_t17/server_constraints/run_0000/chunk_*.npz

R2 가장 빡빡한 행의 표면이 **사과인가** — 그 chunk 에서 공 제외가 제자리였는지 함께
R3 사과가 아니면 무엇이고 거리가 얼마인가
R1 접근 구간에서 TO 가 무엇을 얼마나 밀었나 (관절 deg + 손끝 mm)
R4 그리퍼가 닫히는 t=120 에 손끝이 사과 표면에서 얼마인가

표면을 이름 짓는 방법 (가정을 최소로):
  거리장은 `filtered_cloud` 로 만들어진다. 그래서 `max_violation_pair.point_m` 에 가장 가까운
  **구름 점**이 곧 그 행을 묶은 표면이다. 그 점을 참값 MuJoCo body 표면과 대조해 이름을 붙인다.
  교차 확인으로 (a) 질의점에서 각 body 표면까지의 해석적 거리, (b) ESDF 가 말하는 거리
  `d = clearance_m + r + margin` 를 나란히 낸다. `mj_geomDistance` 는 쓰지 않는다.
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
    ACT_COL, MODEL_XML, QADR, SCENE_BODIES, surface_distances)

ROOT = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260928_t17'))
REC = ROOT / "shadow_ep1807" / "frames.jsonl"
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T17a'))
JSON_OUT = OUT / "r1to4.json"
NPZ_OUT = OUT / "r1to4_curves.npz"
MARGIN_MM = 10.0                 # serve_safe.log:27
APPROACH_MAX_T = 120             # 그리퍼가 닫히는 t
GRIP_COL = 7
FINGER_LINKS = ("ee_finger_l1", "ee_finger_l2")
LEFT_ARM = tuple(f"left_arm_{i}" for i in range(7))


def main() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    cache: dict = {}

    def collidable(g):
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0

    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_BODIES}
    geoms_by_body = {n: [g for g in range(m.ngeom) if collidable(g)
                         and int(m.geom_bodyid[g]) == b] for n, b in bid.items()}
    all_geoms = [g for gs in geoms_by_body.values() for g in gs]
    body_of = {g: n for n, gs in geoms_by_body.items() for g in gs}
    tip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "ee_finger_l1")
    dof = {j: int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)])
           for j in LEFT_ARM}

    P = [json.loads(l) for l in open(REC) if json.loads(l).get("kind") == "planning"]
    by_seq = {int(r["seq"]): r for r in P}
    files = sorted((ROOT / "server_constraints" / "run_0000").glob("chunk_*.npz"))
    print(f"{len(P)} planning records, {len(files)} server chunks", flush=True)

    def place(rec):
        q = np.asarray(rec["qpos"], np.float64)
        d.qpos[: q.size] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

    def body_distances(points_m):
        """`{body: (n,) 표면까지 해석적 거리 m}`. 씬은 이미 놓여 있어야 한다."""
        out = {}
        for n, gs in geoms_by_body.items():
            if gs:
                out[n] = surface_distances(m, d, gs, np.asarray(points_m, np.float64),
                                           cache).min(axis=0)
        return out

    t0 = time.time()
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        seq = int(s["t_step"])
        rec = by_seq.get(seq)
        if rec is None:
            continue
        t = int(rec["t_step"])
        place(rec)
        apple = np.asarray(rec["object_poses"]["apple"]["pos"], np.float64)
        objs = {k: np.asarray(v["pos"], np.float64) for k, v in rec["object_poses"].items()}

        mvp = s.get("to", {}).get("metrics", {}).get("max_violation_pair")
        if isinstance(mvp, str):
            mvp = json.loads(mvp.replace("'", '"').replace("None", "null")
                             .replace("True", "true").replace("False", "false"))
        radii = np.asarray(z["sphere_radii"], np.float64)
        names = list(s.get("sphere_link_names") or [])
        cloud = np.asarray(z["filtered_cloud"], np.float64)
        tpts = (np.asarray(z["target_points"], np.float64) if "target_points" in z.files
                else None)
        cen = (np.asarray(z["target_centroid"], np.float64) if "target_centroid" in z.files
               else None)

        row = {"chunk_index": int(s["chunk_index"]), "server_seq": seq, "t_step": t,
               "grounding_status": s.get("grounding_status"),
               "has_target": bool(s.get("has_target")),
               "target_n_points": int(s.get("target_n_points") or 0),
               "safe": rec.get("safe"), "hold_reason": rec.get("hold_reason", ""),
               "gripper_ref_col7": round(float(np.asarray(
                   rec["actions_reference"], np.float64)[0, GRIP_COL]), 5),
               "n_spheres": int(radii.size)}

        # ---- grounding centroid 가 어느 물체인가 -------------------------------------
        if cen is not None:
            dd = {k: float(np.linalg.norm(cen - v)) * 1000.0 for k, v in objs.items()}
            near = min(dd, key=dd.get)
            row.update(centroid_mm=[round(float(x) * 1000.0, 2) for x in cen],
                       centroid_nearest_object=near,
                       centroid_nearest_is_apple=bool(near == "apple"),
                       centroid_to_apple_mm=round(dd["apple"], 2),
                       centroid_distance_by_object_mm={k: round(v, 2) for k, v in dd.items()})
        # ---- fine 창이 사과를 담는가 ---------------------------------------------------
        lo = np.asarray(z["esdf_origin"], np.float64) * 1000.0
        hi = lo + np.asarray(z["esdf_shape"], np.int64) * float(z["esdf_voxel_size"]) * 1000.0
        ap_mm = apple * 1000.0
        row.update(fine_origin_mm=[round(float(x), 2) for x in lo],
                   fine_max_mm=[round(float(x), 2) for x in hi],
                   apple_in_fine_window=bool(np.all(ap_mm >= lo) and np.all(ap_mm < hi)),
                   apple_outside_by_mm=round(float(np.linalg.norm(
                       np.maximum(np.maximum(lo - ap_mm, ap_mm - hi), 0.0))), 2))
        # ---- 공 제외가 제자리였나 -------------------------------------------------------
        tf = s.get("esdf_stats", {}).get("target_free")
        if isinstance(tf, str):
            try:
                tf = json.loads(tf.replace("'", '"').replace("None", "null")
                                .replace("True", "true").replace("False", "false"))
            except Exception:
                tf = None
        ball = (tf or {}).get("ball") if isinstance(tf, dict) else None
        if ball:
            bc = np.asarray(ball["centre_m"], np.float64)
            row.update(ball_centre_mm=[round(float(x) * 1000.0, 2) for x in bc],
                       ball_radius_mm=round(float(ball["radius_mm"]), 3),
                       ball_centre_to_apple_mm=round(float(np.linalg.norm(bc - apple))
                                                     * 1000.0, 2),
                       ball_contains_apple_centre=bool(
                           np.linalg.norm(bc - apple) * 1000.0 <= float(ball["radius_mm"])))

        # ---- R2/R3 — 가장 빡빡한 행의 표면 ------------------------------------------------
        if mvp and mvp.get("point_m") is not None:
            p = np.asarray(mvp["point_m"], np.float64)
            qi = mvp.get("query")
            r_q = (float(radii[int(qi)]) if qi is not None and int(qi) < radii.size else None)
            d_esdf = (None if r_q is None
                      else float(mvp["clearance_m"]) * 1000.0 + r_q * 1000.0 + MARGIN_MM)
            bd = body_distances(p[None, :])
            bd_mm = {k: round(float(v[0]) * 1000.0, 3) for k, v in bd.items()}
            near_gt = min(bd_mm, key=bd_mm.get)
            # 거리장을 만든 구름에서 가장 가까운 점
            k = int(np.argmin(np.linalg.norm(cloud - p, axis=1)))
            cp = cloud[k]
            cp_dist_mm = float(np.linalg.norm(cp - p)) * 1000.0
            cbd = body_distances(cp[None, :])
            cbd_mm = {kk: round(float(v[0]) * 1000.0, 3) for kk, v in cbd.items()}
            cp_body = min(cbd_mm, key=cbd_mm.get)
            in_target = None
            if tpts is not None and tpts.size:
                in_target = bool(np.min(np.linalg.norm(tpts - cp, axis=1)) < 1e-6)
            row.update(
                mvp_link=mvp.get("link"), mvp_query=qi, mvp_step=mvp.get("step"),
                mvp_clearance_mm_post_margin=round(float(mvp["clearance_m"]) * 1000.0, 4),
                mvp_query_radius_mm=(None if r_q is None else round(r_q * 1000.0, 3)),
                mvp_point_mm=[round(float(x) * 1000.0, 2) for x in p],
                esdf_distance_mm=(None if d_esdf is None else round(d_esdf, 3)),
                gt_distance_by_body_mm=bd_mm,
                gt_nearest_body=near_gt, gt_nearest_mm=bd_mm[near_gt],
                gt_apple_mm=bd_mm.get("apple"),
                cloud_nearest_point_mm=[round(float(x) * 1000.0, 2) for x in cp],
                cloud_nearest_distance_mm=round(cp_dist_mm, 3),
                cloud_point_body=cp_body,
                cloud_point_body_distance_mm=cbd_mm[cp_body],
                cloud_point_distance_by_body_mm=cbd_mm,
                cloud_point_is_a_target_point=in_target,
                esdf_minus_cloud_mm=(None if d_esdf is None else round(d_esdf - cp_dist_mm, 3)),
                esdf_minus_gt_nearest_mm=(None if d_esdf is None
                                          else round(d_esdf - bd_mm[near_gt], 3)))
            # 거리장의 기울기를 따라 d 만큼 걸어 **그 행을 묶은 표면점**을 얻는다.
            # filtered_cloud 는 이 프레임의 구름이고 거리장은 여러 프레임이 쌓인 것이라
            # 구름 최근접점이 항상 그 표면은 아니다 — 기울기 걸음이 거리장 자신의 답이다.
            surf, used_tier = None, None
            for tier, pre in (("fine", "esdf_"), ("coarse", "esdf_coarse_")):
                if f"{pre}origin" not in z.files:
                    continue
                o = np.asarray(z[f"{pre}origin"], np.float64)
                vox = float(z[f"{pre}voxel_size"])
                sh = np.asarray(z[f"{pre}shape"], np.int64)
                dist = np.asarray(z[f"{pre}distance"], np.float32)
                idx = np.floor((p - o) / vox).astype(int)
                row[f"point_in_{tier}_window"] = bool(np.all(idx >= 0) and np.all(idx < sh))
                if surf is not None:
                    continue
                if not (np.all(idx >= 1) and np.all(idx < sh - 1)):
                    continue
                g = np.asarray([
                    float(dist[tuple(idx + e)]) - float(dist[tuple(idx - e)])
                    for e in (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))],
                    np.float64) / (2.0 * vox)
                ng = float(np.linalg.norm(g))
                if ng <= 1e-9:
                    continue
                dv = float(dist[tuple(idx)])
                surf, used_tier = p - (g / ng) * dv, tier
                sbd = body_distances(surf[None, :])
                sbd_mm = {kk: round(float(vv[0]) * 1000.0, 3) for kk, vv in sbd.items()}
                sb = min(sbd_mm, key=sbd_mm.get)
                row.update(
                    esdf_gradient_surface_mm=[round(float(x) * 1000.0, 2) for x in surf],
                    esdf_gradient_walk_mm=round(dv * 1000.0, 3),
                    esdf_gradient_tier=tier,
                    esdf_surface_body=sb, esdf_surface_body_distance_mm=sbd_mm[sb],
                    esdf_surface_distance_by_body_mm=sbd_mm)
            if surf is None:
                row.update(esdf_gradient_surface_mm=None, esdf_surface_body=None,
                           esdf_surface_body_distance_mm=None, esdf_gradient_tier=None)

            # esdf 와 esdf_free 가 이 점에서 갈라지는가 (가장 가까운 voxel 값)
            for tag, pre in (("esdf", "esdf_"), ("esdf_free", "esdf_free_")):
                if f"{pre}origin" not in z.files:
                    row[f"{tag}_at_point_mm"] = None
                    continue
                o = np.asarray(z[f"{pre}origin"], np.float64)
                v = float(z[f"{pre}voxel_size"])
                sh = np.asarray(z[f"{pre}shape"], np.int64)
                idx = np.floor((p - o) / v).astype(int)
                if np.all(idx >= 0) and np.all(idx < sh):
                    row[f"{tag}_at_point_mm"] = round(float(
                        np.asarray(z[f"{pre}distance"])[tuple(idx)]) * 1000.0, 3)
                else:
                    row[f"{tag}_at_point_mm"] = None
            a_, b_ = row.get("esdf_at_point_mm"), row.get("esdf_free_at_point_mm")
            row["esdf_free_minus_esdf_at_point_mm"] = (None if a_ is None or b_ is None
                                                       else round(b_ - a_, 3))

        # ---- R1 — TO 가 얼마나 밀었나 ------------------------------------------------------
        ref = np.asarray(rec["actions_reference"], np.float64)
        ref_o = np.asarray(rec["actions"], np.float64)
        H = 8
        dq = ref_o[:H] - ref[:H]
        per_joint = {j: round(float(np.degrees(np.abs(dq[:, ACT_COL[j]]).max())), 5)
                     for j in LEFT_ARM}
        contrib, tip_d = {}, []
        for st in range(H):
            q = np.asarray(rec["qpos"], np.float64)
            d.qpos[: q.size] = q
            for n, c in ACT_COL.items():
                d.qpos[QADR[n]] = float(ref[st, c])
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            p0 = np.asarray(d.xpos[tip_id], np.float64).copy()
            jac = np.zeros((3, m.nv))
            mujoco.mj_jacBody(m, d, jac, None, tip_id)
            for j in LEFT_ARM:
                contrib[j] = max(contrib.get(j, 0.0), float(np.linalg.norm(
                    jac[:, dof[j]] * float(dq[st, ACT_COL[j]]))) * 1000.0)
            d.qpos[: q.size] = q
            for n, c in ACT_COL.items():
                d.qpos[QADR[n]] = float(ref_o[st, c])
            mujoco.mj_forward(m, d)
            tip_d.append(float(np.linalg.norm(np.asarray(d.xpos[tip_id], np.float64) - p0))
                         * 1000.0)
        row.update(to_per_joint_max_abs_deg=per_joint,
                   to_max_abs_deg=round(max(per_joint.values()), 5),
                   to_largest_joint=max(per_joint, key=per_joint.get),
                   to_jacobian_contrib_mm={k: round(v, 4) for k, v in contrib.items()},
                   to_largest_contrib_joint=max(contrib, key=contrib.get),
                   to_tip_displacement_mm_max=round(max(tip_d), 4),
                   to_tip_displacement_mm_per_step=[round(v, 4) for v in tip_d])

        # ---- R4 — 손끝이 사과 표면에서 얼마인가 -------------------------------------------
        place(rec)
        fin = [i for i, n in enumerate(names) if n in FINGER_LINKS]
        if fin:
            cs = np.asarray(z["sphere_centres"], np.float64)[fin]
            da = body_distances(cs)["apple"] * 1000.0 - radii[fin] * 1000.0
            row["tip_to_apple_surface_mm"] = round(float(da.min()), 4)
            row["tip_centre_to_apple_surface_mm"] = round(
                float((body_distances(cs)["apple"] * 1000.0).min()), 4)
        rows.append(row)
        if len(rows) % 15 == 0:
            print(f"  {len(rows)}/{len(files)} chunks [{time.time()-t0:5.1f} s]", flush=True)

    app = [r for r in rows if r["t_step"] <= APPROACH_MAX_T]
    mv = [r for r in app if r.get("mvp_link")]
    out = {
        "record": str(REC), "server_record": str(ROOT / "server_constraints" / "run_0000"),
        "n_chunks": len(rows), "approach_window_t_step": [0, APPROACH_MAX_T],
        "n_approach_chunks": len(app), "esdf_margin_mm": MARGIN_MM,
        "method": {
            "surface_naming": "거리장은 filtered_cloud 로 만들어진다. max_violation_pair.point_m "
                              "에 가장 가까운 구름 점이 그 행을 묶은 표면이고, 그 점을 참값 "
                              "MuJoCo body 표면과 대조해 이름 붙였다. 교차 확인으로 질의점에서 "
                              "각 body 표면까지의 해석적 거리와 ESDF 가 말하는 거리 "
                              "(clearance_m + r + margin) 를 나란히 냈다.",
            "chunk_mapping": "서버 summary_json 의 t_step 은 chunk 일련번호(seq)다. "
                             "local t_step = (seq - 1) * 8.",
            "clearance_is_post_margin": "summary_json 과 max_violation_pair 의 clearance 는 "
                                        "post-margin 이다 (linearize.py:489).",
            "jacobian": "mujoco.mj_jacBody 를 reference 자세에서 받아 "
                        "|J[:, dof_j] * dq_j| (mm) 로 관절별 손끝 기여를 갈랐다.",
            "no_mj_geomDistance": True,
        },
        "r2_what_surface": {
            "n_rows_with_mvp": len(mv),
            "gt_nearest_body_counts": dict(collections.Counter(
                r["gt_nearest_body"] for r in mv).most_common()),
            "cloud_point_body_counts": dict(collections.Counter(
                r["cloud_point_body"] for r in mv).most_common()),
            "cloud_note": "filtered_cloud 는 이 프레임의 구름이고 거리장은 여러 프레임이 "
                          "쌓인 것이다 — 구름 최근접점이 그 행을 묶은 표면이라고 단정할 수 "
                          "없다. 기울기 걸음(esdf_surface_body)이 거리장 자신의 답이다.",
            "esdf_surface_body_counts": dict(collections.Counter(
                str(r.get("esdf_surface_body")) for r in mv).most_common()),
            "n_esdf_surface_is_apple": int(sum(
                1 for r in mv if r.get("esdf_surface_body") == "apple")),
            "esdf_gradient_tier_counts": dict(collections.Counter(
                str(r.get("esdf_gradient_tier")) for r in mv).most_common()),
            "n_point_in_fine_window": int(sum(
                1 for r in mv if r.get("point_in_fine_window"))),
            "n_point_in_coarse_window": int(sum(
                1 for r in mv if r.get("point_in_coarse_window"))),
            "esdf_minus_gt_apple_mm_on_apple_rows": [
                round(r["esdf_distance_mm"] - r["gt_apple_mm"], 3) for r in mv
                if r["gt_nearest_body"] == "apple"],
            "esdf_surface_body_distance_mm": {
                "min": min((r.get("esdf_surface_body_distance_mm") for r in mv
                            if r.get("esdf_surface_body_distance_mm") is not None),
                           default=None),
                "max": max((r.get("esdf_surface_body_distance_mm") for r in mv
                            if r.get("esdf_surface_body_distance_mm") is not None),
                           default=None)},
            "n_cloud_point_is_a_target_point": int(sum(
                1 for r in mv if r.get("cloud_point_is_a_target_point"))),
            "n_gt_nearest_is_apple": int(sum(1 for r in mv if r["gt_nearest_body"] == "apple")),
            "n_cloud_point_is_apple": int(sum(1 for r in mv if r["cloud_point_body"] == "apple")),
            "esdf_minus_cloud_mm": {
                "min": round(min(r["esdf_minus_cloud_mm"] for r in mv), 3),
                "median": round(float(np.median([r["esdf_minus_cloud_mm"] for r in mv])), 3),
                "max": round(max(r["esdf_minus_cloud_mm"] for r in mv), 3)},
            "grounding": {
                "n_centroid_nearest_is_apple": int(sum(
                    1 for r in app if r.get("centroid_nearest_is_apple") is True)),
                "centroid_nearest_object_counts": dict(collections.Counter(
                    str(r.get("centroid_nearest_object")) for r in app).most_common()),
                "centroid_to_apple_mm_median": round(float(np.median(
                    [r["centroid_to_apple_mm"] for r in app
                     if r.get("centroid_to_apple_mm") is not None])), 2),
                "n_apple_in_fine_window": int(sum(
                    1 for r in app if r.get("apple_in_fine_window"))),
                "n_ball_contains_apple_centre": int(sum(
                    1 for r in app if r.get("ball_contains_apple_centre"))),
                "ball_radius_mm_median": round(float(np.median(
                    [r["ball_radius_mm"] for r in app if r.get("ball_radius_mm")])), 3),
                "ball_centre_to_apple_mm_median": round(float(np.median(
                    [r["ball_centre_to_apple_mm"] for r in app
                     if r.get("ball_centre_to_apple_mm") is not None])), 2)},
            "esdf_free_minus_esdf_at_point_mm": {
                "n_rows": int(sum(1 for r in mv
                                  if r.get("esdf_free_minus_esdf_at_point_mm") is not None)),
                "n_nonzero": int(sum(1 for r in mv
                                     if (r.get("esdf_free_minus_esdf_at_point_mm") or 0) != 0)),
                "max_abs": max((abs(r.get("esdf_free_minus_esdf_at_point_mm") or 0)
                                for r in mv), default=None)},
        },
        "r2_r3_overlap": {
            "question": "사과가 그 행을 묶은 chunk 와 grounding 이 사과를 놓친 chunk 가 "
                        "같은 집합인가",
            "t_steps_surface_is_apple": [r["t_step"] for r in mv
                                         if r.get("esdf_surface_body") == "apple"],
            "t_steps_centroid_not_apple": [r["t_step"] for r in app
                                           if r.get("centroid_nearest_is_apple") is False],
            "t_steps_apple_not_in_fine_window": [r["t_step"] for r in app
                                                 if r.get("apple_in_fine_window") is False],
            "t_steps_ball_misses_apple_centre": [r["t_step"] for r in app
                                                 if r.get("ball_contains_apple_centre")
                                                 is False],
            "surface_apple_is_subset_of_centroid_not_apple": bool(
                set(r["t_step"] for r in mv if r.get("esdf_surface_body") == "apple")
                <= set(r["t_step"] for r in app
                       if r.get("centroid_nearest_is_apple") is False)),
            "ball_radius_mm_when_centroid_is_apple": sorted({
                r["ball_radius_mm"] for r in app
                if r.get("centroid_nearest_is_apple") is True and r.get("ball_radius_mm")}),
            "ball_radius_mm_when_centroid_is_not_apple": sorted({
                r["ball_radius_mm"] for r in app
                if r.get("centroid_nearest_is_apple") is False and r.get("ball_radius_mm")}),
        },
        "r3_if_not_the_apple": {
            "per_body_min_gt_distance_mm": {
                b: round(min((r["gt_distance_by_body_mm"][b] for r in mv
                              if b in r["gt_distance_by_body_mm"]), default=float("nan")), 3)
                for b in sorted(geoms_by_body)},
            "rows": [{"t_step": r["t_step"], "mvp_link": r["mvp_link"],
                      "clearance_mm_post_margin": r["mvp_clearance_mm_post_margin"],
                      "esdf_distance_mm": r["esdf_distance_mm"],
                      "gt_nearest_body": r["gt_nearest_body"],
                      "gt_nearest_mm": r["gt_nearest_mm"],
                      "gt_apple_mm": r["gt_apple_mm"],
                      "cloud_point_body": r["cloud_point_body"],
                      "cloud_nearest_distance_mm": r["cloud_nearest_distance_mm"],
                      "cloud_point_is_a_target_point": r["cloud_point_is_a_target_point"],
                      "esdf_surface_body": r.get("esdf_surface_body"),
                      "esdf_gradient_tier": r.get("esdf_gradient_tier"),
                      "point_in_fine_window": r.get("point_in_fine_window"),
                      "point_in_coarse_window": r.get("point_in_coarse_window"),
                      "esdf_surface_body_distance_mm": r.get("esdf_surface_body_distance_mm"),
                      "esdf_gradient_surface_mm": r.get("esdf_gradient_surface_mm"),
                      "gt_distance_by_body_mm": r["gt_distance_by_body_mm"]} for r in mv],
        },
        "r1_what_to_pushed": {
            "n_chunks": len(app),
            "to_max_abs_deg": {
                "median": round(float(np.median([r["to_max_abs_deg"] for r in app])), 5),
                "max": round(max(r["to_max_abs_deg"] for r in app), 5),
                "argmax_t_step": max(app, key=lambda r: r["to_max_abs_deg"])["t_step"]},
            "largest_joint_counts": dict(collections.Counter(
                r["to_largest_joint"] for r in app).most_common()),
            "largest_contrib_joint_counts": dict(collections.Counter(
                r["to_largest_contrib_joint"] for r in app).most_common()),
            "tip_displacement_mm": {
                "median": round(float(np.median(
                    [r["to_tip_displacement_mm_max"] for r in app])), 4),
                "max": round(max(r["to_tip_displacement_mm_max"] for r in app), 4),
                "argmax_t_step": max(app,
                                     key=lambda r: r["to_tip_displacement_mm_max"])["t_step"]},
            "per_joint_max_over_approach_deg": {
                j: round(max(r["to_per_joint_max_abs_deg"][j] for r in app), 5)
                for j in LEFT_ARM},
            "per_joint_max_contrib_mm": {
                j: round(max(r["to_jacobian_contrib_mm"][j] for r in app), 4)
                for j in LEFT_ARM},
        },
        "r4_gripper_close": None,
        "per_chunk": rows,
    }
    # R4 — 그리퍼 명령이 0.85 아래로 처음 내려가는 **step** 을 찾아 그 자세를 FK 로 푼다.
    # npz 의 sphere_centres 는 chunk 하나의 한 자세뿐이라 "그 순간" 이 아니다.
    close_detail = None
    for rec in P:
        ref = np.asarray(rec["actions_reference"], np.float64)
        # **실행되는 창(8 step)만 본다.** 뒤쪽 42 step 은 이 chunk 에서 실행되지 않는다.
        hit = np.nonzero(ref[:8, GRIP_COL] < 0.85)[0]
        if hit.size == 0:
            continue
        st = int(hit[0])
        place(rec)
        for n, c in ACT_COL.items():
            d.qpos[QADR[n]] = float(ref[st, c])
        mujoco.mj_forward(m, d)
        tip = np.asarray(d.xpos[tip_id], np.float64)
        apple = np.asarray(rec["object_poses"]["apple"]["pos"], np.float64)
        da = float(body_distances(tip[None, :])["apple"][0]) * 1000.0
        fin_idx = None
        close_detail = {
            "window": "execution window only (steps 0..7)",
            "t_step_of_chunk": int(rec["t_step"]), "seq": int(rec["seq"]),
            "step_in_chunk": st,
            "t_step_of_the_step": int(rec["t_step"]) + st,
            "gripper_value": round(float(ref[st, GRIP_COL]), 6),
            "gripper_value_previous_step": (round(float(ref[st - 1, GRIP_COL]), 6)
                                            if st > 0 else None),
            "ee_finger_l1_origin_to_apple_surface_mm": round(da, 4),
            "offset_tip_minus_apple_centre_mm": [round(float(x) * 1000.0, 3)
                                                 for x in (tip - apple)],
            "note": "ee_finger_l1 body 원점에서 사과 표면까지의 해석적 거리다 "
                    "(구 반지름을 빼지 않은 값). 제약 구로 재려면 반지름을 더 빼야 한다.",
        }
        break

    close = [r for r in rows if r["gripper_ref_col7"] < 0.85]
    if close:
        c0 = close[0]
        out["r4_gripper_close"] = {
            "t_step": c0["t_step"], "gripper_ref_col7": c0["gripper_ref_col7"],
            "gripper_before": next((r["gripper_ref_col7"] for r in reversed(rows)
                                    if r["t_step"] < c0["t_step"]), None),
            "tip_to_apple_surface_mm": c0.get("tip_to_apple_surface_mm"),
            "tip_centre_to_apple_surface_mm": c0.get("tip_centre_to_apple_surface_mm"),
            "reference_value_T7a_mm": -17.9552,
            "difference_from_T7a_mm": (None if c0.get("tip_to_apple_surface_mm") is None
                                       else round(c0["tip_to_apple_surface_mm"] + 17.9552, 4)),
            "min_over_all_chunks_mm": round(min(
                r["tip_to_apple_surface_mm"] for r in rows
                if r.get("tip_to_apple_surface_mm") is not None), 4),
            "min_over_all_chunks_t_step": min(
                (r for r in rows if r.get("tip_to_apple_surface_mm") is not None),
                key=lambda r: r["tip_to_apple_surface_mm"])["t_step"],
            "note": "제약 구 모델이 이 실행은 122 개다 (손끝 반지름이 T7a 와 다르면 "
                    "clearance = d - r 도 달라진다). 중심-표면 거리도 함께 낸다. "
                    "위 두 값은 서버가 chunk 와 함께 저장한 한 자세의 것이고, "
                    "'at_the_closing_step' 이 실제로 닫히는 step 의 값이다.",
            "at_the_closing_step": close_detail,
        }
    out["elapsed_s"] = round(time.time() - t0, 1)
    JSON_OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(NPZ_OUT, t_step=np.asarray([r["t_step"] for r in rows]))

    print("\n=== R2 what surface ===")
    e = out["r2_what_surface"]
    print(" gt nearest body:", e["gt_nearest_body_counts"])
    print(" cloud point body:", e["cloud_point_body_counts"])
    print(" cloud point is a target point:", e["n_cloud_point_is_a_target_point"],
          "of", e["n_rows_with_mvp"])
    print(" grounding:", json.dumps(e["grounding"], ensure_ascii=False))
    print(" esdf_free - esdf at the point:", e["esdf_free_minus_esdf_at_point_mm"])
    print("=== R3 rows ===")
    for r in out["r3_if_not_the_apple"]["rows"]:
        print(f"  t={r['t_step']:4d} {r['mvp_link']:14s} clr {r['clearance_mm_post_margin']:8.3f} "
              f"esdf_d {r['esdf_distance_mm']:8.3f} | gt {r['gt_nearest_body']:7s} "
              f"{r['gt_nearest_mm']:8.3f} apple {r['gt_apple_mm']:8.3f} | cloud "
              f"{r['cloud_point_body']:7s} at {r['cloud_nearest_distance_mm']:7.3f} "
              f"target_pt={r['cloud_point_is_a_target_point']} | "
              f"grad[{str(r.get('esdf_gradient_tier')):6s}] {str(r['esdf_surface_body']):7s} "
              + ("   n/a" if r["esdf_surface_body_distance_mm"] is None
                 else f"{r['esdf_surface_body_distance_mm']:8.3f}"))
    print("=== R2 n R3 overlap ===")
    print(json.dumps(out["r2_r3_overlap"], indent=1, ensure_ascii=False))
    print("=== R1 ===")
    print(json.dumps(out["r1_what_to_pushed"], indent=1, ensure_ascii=False))
    print("=== R4 ===")
    print(json.dumps(out["r4_gripper_close"], indent=1, ensure_ascii=False))
    print("wrote", JSON_OUT, f"[{out['elapsed_s']} s]")


if __name__ == "__main__":
    main()
