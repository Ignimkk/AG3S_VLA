"""T8e s2 — 전역 축: `capsule_radius_scale` (모든 link) × `sphere_spacing`. z_cut -100 mm 고정.

왜 spacing 을 같이 훑나: `_effective_radius` 가 구 n>=2 에서 sqrt((r_urdf*s)^2 + (spacing/2)^2)
이고 spacing = capsule 길이/(n-1) 이라 **s -> 0 이어도 spacing/2 바닥이 남는다**. 바닥을 내리는
손잡이는 `sphere_spacing` 하나뿐이다.

`max_spheres_per_capsule` 은 상한이라 기본값 8 이면 spacing < 0.5 에서 n 을 잘라 버린다.
여기서는 32 로 올려 spacing 이 실제로 듣게 한다 — 그 값을 함께 기록한다.

self-filter 모델(`build_robot_model`)은 건드리지 않는다. `sphere_options` 통로가 제약 모델
(`build_constraint_robot_model`)에만 있고, 거기서 가늘어지면 그 link 의 점이 필터를 통과해
장애물로 샌다 (`grounding_report.py` docstring). 이 측정은 제약 모델만 훑는다.

제약 행: QP 블록은 `horizon x reduction.rows_per_step` 로 **구 개수와 무관하게 고정**이다
(`linearize.py:258`, `set_attached` docstring). 구가 늘면 늘어나는 것은 **후보 행**
(horizon x 구) 과 FK jacobian 의 크기·평가 시간이다. 셋을 다 낸다.
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
    ACT_COL, JOINT_ORDER, MODEL_XML, QADR, SCENE_BODIES, SHADOW, load_planning, surface_distances)

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8e/s2_global_scale_sweep.json'))
NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8e/s2_grid.npz'))
MARGIN = 10.0
EXEC_H = 8
Z_CUT = -0.100
AUTH = ("ee_finger_l1", "ee_finger_l2", "link_left_arm_5")
ARM5_LINKS = ("link_left_arm_5", "link_right_arm_5")
PALM_LINKS = ("ee_left", "ee_right")
SCALES = (1.0, 0.6, 0.4, 0.3, 0.2)
SPACINGS = (1.0, 0.6, 0.4)
MAX_SPHERES = 32
ROWS_PER_STEP = 24
LIN_HORIZON = 8


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)
    mesh_cache: dict = {}

    def collidable(g):
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0

    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_BODIES}
    scene_geoms = [g for g in range(m.ngeom)
                   if collidable(g) and int(m.geom_bodyid[g]) in set(bid.values())]
    geom_body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
                 for g in scene_geoms]
    bodies = sorted(set(geom_body))
    body_of_geom = np.asarray([bodies.index(b) for b in geom_body], np.int64)
    G, ai = len(scene_geoms), bodies.index("apple")

    recs = load_planning(SHADOW)
    chunks = np.asarray([r["actions_reference"] for r in recs], np.float64)
    F, H = chunks.shape[0], chunks.shape[1]

    def q_from(row, qpos):
        return np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(row[ACT_COL[n]])
                           for n in JOINT_ORDER], np.float64)

    LINKS = tuple(ARM_LINKS) + PALM_LINKS

    def build(spacing, scale=None):
        opts = {"sphere_spacing": spacing, "max_spheres_per_capsule": MAX_SPHERES}
        if scale is not None and scale != 1.0:
            opts["capsule_radius_scale"] = scale
        return build_constraint_robot_model(scene, link_filter=LINKS, sphere_options=opts)

    # 현재 서버 설정의 기준점 (spacing 1.0, cap 8, 손바닥 없음) — 구 개수 대조용
    cur = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    n_current = int(cur.n_spheres)
    print(f"current constraint model (ARM_LINKS, spacing 1.0, cap 8): {n_current} spheres")

    t0 = time.time()
    grid = []
    heat = {}
    for spacing in SPACINGS:
        base = build(spacing)
        link_names = list(resolve_link_names(base, base.n_spheres))
        radii1 = np.asarray(base.sphere_centers_numeric(np.zeros(base.nq))[1],
                            np.float64).reshape(-1) * 1000.0
        local_z = np.asarray([c[2] for (_, c, _) in base._local_spheres], np.float64)
        S = len(link_names)
        is_palm = np.asarray([n in PALM_LINKS for n in link_names], bool)
        uniq = sorted(set(link_names))
        lo = np.asarray([uniq.index(n) for n in link_names], np.int64)
        auth = np.asarray([n in AUTH for n in link_names], bool)
        is_eeleft = np.asarray([n == "ee_left" for n in link_names], bool)
        cut_ok = np.asarray([not (n in ARM5_LINKS and z < Z_CUT)
                             for n, z in zip(link_names, local_z)], bool)
        n_by_link = {ln: int((np.asarray(link_names) == ln).sum()) for ln in uniq}
        print(f"\n#### spacing={spacing} cap={MAX_SPHERES}: {S} spheres with palm "
              f"({S - int(is_palm.sum())} without) · arm5 n={n_by_link['link_left_arm_5']}",
              flush=True)

        # --- 제약 행 비용: CollisionLinearizer 를 실제로 세운다 ---------------------------
        from AG3S.trajopt.linearize import CollisionLinearizer
        from AG3S.trajopt.types import ChunkLayout
        from AG3S.ag3s.robot_models import DEFAULT_RBY1_JOINTS
        layout = ChunkLayout.rby1(DEFAULT_RBY1_JOINTS)
        tb = time.time()
        lin = CollisionLinearizer(base, layout, LIN_HORIZON)
        build_s = time.time() - tb
        qz = np.zeros((int(base.nq), LIN_HORIZON))
        lin._fk_map(qz)
        te = time.time()
        for _ in range(5):
            lin._fk_map(qz)
        fk_ms = (time.time() - te) / 5.0 * 1000.0
        rowcost = {
            "linearizer_n_spheres": int(lin.n_spheres),
            "horizon_used": LIN_HORIZON,
            "candidate_rows_per_chunk": int(lin.n_spheres) * LIN_HORIZON,
            "qp_rows_per_chunk": LIN_HORIZON * ROWS_PER_STEP,
            "qp_rows_note": "QP 블록은 horizon x reduction.rows_per_step 로 구 개수와 무관하게 "
                            "고정 (linearize.py:258). 늘어나는 것은 후보 행과 FK 비용이다.",
            "jacobian_shape_H_S_3_nqopt": list(lin._jac_shape),
            "jacobian_structural_nnz": int(lin._jac_dest.size),
            "jacobian_dense_entries": int(np.prod(lin._jac_shape)),
            "linearizer_build_s": round(build_s, 3),
            "fk_map_eval_ms_mean_of_5": round(fk_ms, 3),
        }
        print(f"   linearizer: {lin.n_spheres} spheres · candidate rows "
              f"{rowcost['candidate_rows_per_chunk']} · QP rows "
              f"{rowcost['qp_rows_per_chunk']} · jac nnz {rowcost['jacobian_structural_nnz']} · "
              f"build {build_s:.2f} s · fk {fk_ms:.3f} ms", flush=True)

        # --- tensor (표면거리, mm) ---------------------------------------------------------
        sd = np.empty((F, H, S, len(bodies)), np.float32)
        for i, rec in enumerate(recs):
            qpos = np.asarray(rec["qpos"], np.float64)
            d.qpos[: qpos.size] = qpos
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            centres = np.empty((H, S, 3), np.float64)
            for s_ in range(H):
                centres[s_] = np.asarray(
                    base.sphere_centers_numeric(q_from(chunks[i, s_], qpos))[0],
                    np.float64).reshape(-1, 3)
            flat = centres.reshape(-1, 3)
            dist = np.empty((G, flat.shape[0]), np.float64)
            B = 1200
            for a in range(0, flat.shape[0], B):
                dist[:, a : a + B] = surface_distances(m, d, scene_geoms, flat[a : a + B],
                                                       mesh_cache)
            cl = dist.reshape(G, H, S).transpose(1, 2, 0)
            for bi in range(len(bodies)):
                sd[i, :, :, bi] = cl[:, :, body_of_geom == bi].min(axis=2) * 1000.0
        print(f"   tensor built [{time.time()-t0:.1f} s]", flush=True)

        def keep_for(palm_tag):
            keep_s = cut_ok.copy()
            if palm_tag == "out":
                keep_s &= ~is_palm
            keep = np.ones((S, len(bodies)), bool)
            a = auth.copy()
            if palm_tag == "in_authorized":
                a |= is_eeleft
            keep[a, ai] = False
            keep &= keep_s[:, None]
            return keep, keep_s

        for scale in SCALES:
            rob = build(spacing, scale)
            assert list(resolve_link_names(rob, rob.n_spheres)) == link_names
            rs = np.asarray(rob.sphere_centers_numeric(np.zeros(rob.nq))[1],
                            np.float64).reshape(-1) * 1000.0
            shortfall = [{"link": s_.link, "sphere_radius_mm": round(s_.sphere_radius * 1000, 4),
                          "required_radius_mm": round(s_.required_radius * 1000, 4),
                          "shortfall_mm": round(s_.shortfall_m * 1000, 4)}
                         for s_ in rob.coverage_shortfall]
            radius_by_link = {ln: sorted({round(float(v), 4) for v in rs[lo == uniq.index(ln)]})
                              for ln in uniq}
            for palm_tag in ("out", "in", "in_authorized"):
                keep, keep_s = keep_for(palm_tag)
                res = {}
                for wname, w in (("full50", slice(0, H)), ("exec8", slice(0, EXEC_H))):
                    cw = np.where(keep[None, None, :, :],
                                  sd[:, w] - rs[None, None, :, None].astype(np.float32),
                                  np.float32(np.inf))
                    le0 = (cw - MARGIN) <= 0.0
                    pairs = []
                    for li, ln in enumerate(uniq):
                        sm = lo == li
                        if not keep[sm].any():
                            continue
                        for bi, bn in enumerate(bodies):
                            if not keep[sm, bi].any():
                                continue
                            v = cw[:, :, sm, bi]
                            n = int(((v - MARGIN) <= 0.0).sum())
                            if not n:
                                continue
                            cm = v.min(axis=(1, 2))
                            pairs.append({"link": ln, "obstacle_body": bn,
                                          "n_rows_slack_le0": n, "n_rows_total": int(v.size),
                                          "n_chunks_slack_le0": int((cm - MARGIN <= 0).sum()),
                                          "min_clearance_mm": round(float(v.min()), 4),
                                          "min_slack_mm": round(float(v.min()) - MARGIN, 4)})
                    res[wname] = {
                        "n_rows_slack_le0": int(le0.sum()),
                        "n_rows_total": int(np.isfinite(cw).sum()),
                        "n_chunks_slack_le0": int(le0.any(axis=(1, 2, 3)).sum()), "n_chunks": F,
                        "global_min_slack_mm": round(float(cw.min()) - MARGIN, 4),
                        "feasible_at_10mm": bool(le0.sum() == 0),
                        "violating_pairs": sorted(pairs, key=lambda p: p["min_slack_mm"])}
                key = f"spacing_{spacing:g}__scale_{scale:g}__palm_{palm_tag}"
                cell = {
                    "key": key, "sphere_spacing": spacing, "capsule_radius_scale": scale,
                    "max_spheres_per_capsule": MAX_SPHERES, "palm": palm_tag,
                    "z_cut_m": Z_CUT,
                    "n_spheres_model": S, "n_spheres_used_after_cut": int(keep_s.sum()),
                    "n_spheres_current_model_for_reference": n_current,
                    "n_spheres_by_link": n_by_link,
                    "radius_mm_by_link": radius_by_link,
                    "n_coverage_shortfall": len(shortfall),
                    "worst_shortfall_mm": (round(max(s_["shortfall_mm"] for s_ in shortfall), 4)
                                           if shortfall else 0.0),
                    "coverage_shortfall": shortfall,
                    "row_cost": rowcost,
                    "windows": res}
                grid.append(cell)
                heat[f"{key}__exec8"] = np.asarray(
                    [res["exec8"]["n_rows_slack_le0"], res["exec8"]["n_chunks_slack_le0"],
                     res["exec8"]["global_min_slack_mm"]], np.float64)
                heat[f"{key}__full50"] = np.asarray(
                    [res["full50"]["n_rows_slack_le0"], res["full50"]["n_chunks_slack_le0"],
                     res["full50"]["global_min_slack_mm"]], np.float64)
                print(f"   {key:44s} L5 r={radius_by_link['link_left_arm_5'][0]:6.2f} "
                      f"fl1 r={radius_by_link['ee_finger_l1'][-1]:5.2f} | "
                      f"exec8 rows {res['exec8']['n_rows_slack_le0']:5d} "
                      f"chk {res['exec8']['n_chunks_slack_le0']:3d} "
                      f"min {res['exec8']['global_min_slack_mm']:8.2f} "
                      f"F{int(res['exec8']['feasible_at_10mm'])} | "
                      f"full50 rows {res['full50']['n_rows_slack_le0']:6d} "
                      f"chk {res['full50']['n_chunks_slack_le0']:3d} "
                      f"min {res['full50']['global_min_slack_mm']:8.2f} "
                      f"F{int(res['full50']['feasible_at_10mm'])} | "
                      f"short n={len(shortfall):2d} worst {cell['worst_shortfall_mm']:6.2f}",
                      flush=True)
        del sd

    # --- 위반 0 인 칸 중 가장 공격적인 것 (= 반지름 합이 가장 작은 칸) ---------------------
    def aggressive(palm_tag, w):
        cand = [c for c in grid if c["palm"] == palm_tag and c["windows"][w]["feasible_at_10mm"]]
        if not cand:
            return None
        def total_r(c):
            return sum(max(v) for v in c["radius_mm_by_link"].values())
        best = min(cand, key=total_r)
        return {"key": best["key"], "sphere_spacing": best["sphere_spacing"],
                "capsule_radius_scale": best["capsule_radius_scale"],
                "n_spheres_model": best["n_spheres_model"],
                "n_spheres_used_after_cut": best["n_spheres_used_after_cut"],
                "sum_of_max_link_radius_mm": round(total_r(best), 4),
                "radius_mm_by_link": best["radius_mm_by_link"],
                "n_coverage_shortfall": best["n_coverage_shortfall"],
                "worst_shortfall_mm": best["worst_shortfall_mm"],
                "row_cost": best["row_cost"],
                "windows": {ww: {k: best["windows"][ww][k] for k in
                                 ("n_rows_slack_le0", "n_chunks_slack_le0",
                                  "global_min_slack_mm", "feasible_at_10mm", "violating_pairs")}
                            for ww in ("exec8", "full50")}}

    picks = {f"{p}__{w}": aggressive(p, w)
             for p in ("out", "in", "in_authorized") for w in ("exec8", "full50")}

    out = {
        "method": "spacing 마다 clearance tensor 를 새로 만들고 (T8a m0 와 같은 오프라인 "
                  "재구성, mj_geomDistance 미사용) 전역 capsule_radius_scale 은 tensor 재독으로 "
                  "훑는다 (중심 불변). z_cut 은 arm_5 구를 행에서 빼는 근사.",
        "condition": {"z_cut_m": Z_CUT, "margin_mm": MARGIN,
                      "apple_rows_removed_for_links": list(AUTH),
                      "record": SHADOW, "variant": "actions_reference",
                      "n_chunks": F, "horizon": H, "bodies": bodies},
        "scales": list(SCALES), "spacings": list(SPACINGS),
        "max_spheres_per_capsule": MAX_SPHERES,
        "max_spheres_note": "기본값 8 은 spacing < 0.5 에서 n 을 잘라 spacing 이 안 듣는다. "
                            "여기서는 32 로 올렸다.",
        "self_filter_untouched": "build_robot_model(self-filter) 에는 sphere_options 통로가 "
                                 "없고 이 측정은 부르지 않았다. 제약 모델만 훑었다.",
        "n_spheres_current_constraint_model": n_current,
        "rows_per_step": ROWS_PER_STEP,
        "grid": grid,
        "most_aggressive_feasible_cell": picks,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(NPZ, **heat)
    print("\nmost aggressive feasible cells:")
    for k, v in picks.items():
        print(f"  {k}: {'none' if v is None else v['key']}")
    print("wrote", OUT, NPZ, f"[{time.time()-t0:.1f} s]")


if __name__ == "__main__":
    main()
