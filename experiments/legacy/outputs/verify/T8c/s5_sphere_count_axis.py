"""T8c s5 — 구 **개수** 축. 반지름 배율만 훑으면 `sqrt((r*s)^2 + (spacing/2)^2)` 의 바닥값에 걸린다.

`_capsule_sphere_centres` 에서
    n = ceil(cap.length / (sphere_spacing * cap.radius)) + 1,  clip(1, max_spheres_per_capsule)
이므로 `link_*_arm_5`(L 250 mm, r 75 mm) 는 기본값 `sphere_spacing=1.0` 에서 n = ceil(3.333)+1 = 5
이고 `max_spheres_per_capsule=8` 은 **걸리지 않는다**. 실제로 무엇이 듣는지 둘 다 세워 확인한다.

구 개수가 바뀌면 **중심이 바뀐다** — T8a tensor 를 재사용할 수 없다. 그래서 geometry 마다
m0 와 같은 방법으로 거리 tensor 를 다시 만든다 (기록된 qpos + chunk action, 해석적 표면거리,
`mj_geomDistance` 미사용). 그 뒤 반지름 배율은 tensor 재사용으로 훑는다.
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

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8c/s5_sphere_count_axis.json'))
MARGIN = 10.0
EXEC_H = 8
AUTH = ("ee_finger_l1", "ee_finger_l2", "link_left_arm_5")
ARM5 = ("link_left_arm_5", "link_right_arm_5")
FINGERS = ("ee_finger_l1", "ee_finger_l2")
GEOMETRIES = [(1.0, 8), (1.0, 13), (0.6, 13), (0.4, 13), (0.3, 13), (0.2, 16)]
ARM5_SCALES = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
FINGER_SCALES = (1.0, 0.35)


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

    def collidable(g):
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0
    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_BODIES}
    scene_geoms = [g for g in range(m.ngeom)
                   if collidable(g) and int(m.geom_bodyid[g]) in set(bid.values())]
    geom_body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
                 for g in scene_geoms]
    bodies = sorted(set(geom_body))
    body_of = np.asarray([bodies.index(b) for b in geom_body], np.int64)
    G = len(scene_geoms)
    ai = bodies.index("apple")

    recs = load_planning(SHADOW)
    chunks = np.asarray([r["actions_reference"] for r in recs], np.float64)
    F, H = chunks.shape[0], chunks.shape[1]

    def q_from(row, qpos):
        return np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(row[ACT_COL[n]])
                           for n in JOINT_ORDER], np.float64)

    def tensor_for(rob) -> tuple[np.ndarray, list[str], np.ndarray]:
        names = list(resolve_link_names(rob, rob.n_spheres))
        S = rob.n_spheres
        rad = np.asarray(rob.sphere_centers_numeric(np.zeros(rob.nq))[1],
                         np.float64).reshape(-1) * 1000.0
        clear = np.empty((F, H, S, len(bodies)), np.float32)
        for i, rec in enumerate(recs):
            qpos = np.asarray(rec["qpos"], np.float64)
            d.qpos[: qpos.size] = qpos
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            cen = np.empty((H, S, 3), np.float64)
            for s_ in range(H):
                cen[s_] = np.asarray(rob.sphere_centers_numeric(q_from(chunks[i, s_], qpos))[0],
                                     np.float64).reshape(-1, 3)
            flat = cen.reshape(-1, 3)
            dist = np.empty((G, flat.shape[0]), np.float64)
            for a in range(0, flat.shape[0], 1200):
                dist[:, a : a + 1200] = surface_distances(m, d, scene_geoms,
                                                          flat[a : a + 1200], mesh_cache)
            cl = dist.reshape(G, H, S).transpose(1, 2, 0) * 1000.0     # mm, radius 뒤에서 뺀다
            for b in range(len(bodies)):
                clear[i, :, :, b] = cl[:, :, body_of == b].min(axis=2)
        return clear.astype(np.float64), names, rad     # clear = surface_distance (mm)

    results = []
    for spacing, cap in GEOMETRIES:
        t0 = time.time()
        opts = {"sphere_spacing": spacing, "max_spheres_per_capsule": cap}
        rob = build_constraint_robot_model(scene, link_filter=ARM_LINKS, sphere_options=opts)
        sd, names, rad1 = tensor_for(rob)      # surface distance tensor (F,H,S,B) mm
        S = len(names)
        n_arm5 = sum(1 for n in names if n == "link_left_arm_5")
        auth = np.asarray([n in AUTH for n in names], bool)
        keep = np.ones((S, len(bodies)), bool)
        keep[auth, ai] = False
        geo = {
            "sphere_spacing": spacing, "max_spheres_per_capsule": cap,
            "n_spheres_total": S,
            "n_spheres_link_left_arm_5": n_arm5,
            "n_spheres_by_link": {ln: sum(1 for n in names if n == ln)
                                  for ln in sorted(set(names))},
            "radius_mm_link_left_arm_5_at_scale_1": round(
                float(rad1[[i for i, n in enumerate(names) if n == "link_left_arm_5"][0]]), 4),
            "tensor_seconds": round(time.time() - t0, 1),
            "combos": [],
        }
        for fs in FINGER_SCALES:
            for a in ARM5_SCALES:
                by = {}
                if a != 1.0:
                    by.update({ln: a for ln in ARM5})
                if fs != 1.0:
                    by.update({ln: fs for ln in FINGERS})
                o2 = dict(opts)
                if by:
                    o2["capsule_radius_scale_by_link"] = by
                r2 = build_constraint_robot_model(scene, link_filter=ARM_LINKS,
                                                  sphere_options=o2)
                assert list(resolve_link_names(r2, r2.n_spheres)) == names
                rs = np.asarray(r2.sphere_centers_numeric(np.zeros(r2.nq))[1],
                                np.float64).reshape(-1) * 1000.0
                clear = sd - rs[None, None, :, None]
                rec = {"arm5_scale": a, "finger_scale": fs,
                       "radius_mm_link_left_arm_5": round(
                           float(rs[[i for i, n in enumerate(names)
                                     if n == "link_left_arm_5"][0]]), 4),
                       "n_coverage_shortfall": len(r2.coverage_shortfall),
                       "worst_shortfall_mm": (round(max(x.shortfall_m
                                                        for x in r2.coverage_shortfall) * 1000.0,
                                                    4) if r2.coverage_shortfall else 0.0),
                       "windows": {}}
                for wn, w in (("full50", slice(0, H)), ("exec8", slice(0, EXEC_H))):
                    cw = np.where(keep[None, None, :, :], clear[:, w], np.inf)
                    le0 = (cw - MARGIN) <= 0.0
                    pairs = []
                    for ln in sorted(set(names)):
                        sel = np.asarray([n == ln for n in names], bool)
                        for bi, bn in enumerate(bodies):
                            if not keep[sel, bi].any():
                                continue
                            v = cw[:, :, sel, bi]
                            k = int(((v - MARGIN) <= 0.0).sum())
                            if k:
                                pairs.append({"link": ln, "obstacle_body": bn,
                                              "n_rows_slack_le0": k,
                                              "min_slack_mm": round(float(v.min()) - MARGIN, 4)})
                    rec["windows"][wn] = {
                        "n_rows_slack_le0": int(le0.sum()),
                        "n_chunks_slack_le0": int(le0.any(axis=(1, 2, 3)).sum()),
                        "min_slack_mm": round(float(cw.min()) - MARGIN, 4),
                        "feasible_at_10mm": bool(le0.sum() == 0),
                        "violating_pairs": sorted(pairs, key=lambda p: p["min_slack_mm"])}
                geo["combos"].append(rec)
                f50, e8 = rec["windows"]["full50"], rec["windows"]["exec8"]
                print(f"  sp={spacing:g} cap={cap:2d} n(L5)={n_arm5} S={S:3d} "
                      f"arm5 x{a:g} fing x{fs:g} r={rec['radius_mm_link_left_arm_5']:6.2f} | "
                      f"full50 {f50['n_rows_slack_le0']:5d}/{f50['n_chunks_slack_le0']:3d} "
                      f"{f50['min_slack_mm']:8.2f} | exec8 {e8['n_rows_slack_le0']:4d}/"
                      f"{e8['n_chunks_slack_le0']:3d} {e8['min_slack_mm']:8.2f} | "
                      f"shortfall {rec['n_coverage_shortfall']:2d} "
                      f"{rec['worst_shortfall_mm']:6.2f}", flush=True)
        results.append(geo)

    out = {
        "method": "geometry(sphere_spacing, max_spheres_per_capsule) 마다 거리 tensor 를 "
                  "m0 와 같은 방법으로 다시 만들고, 그 위에서 반지름 배율만 tensor 재사용으로 "
                  "훑는다. 구 개수가 바뀌면 중심이 바뀌므로 T8a tensor 를 쓸 수 없다.",
        "n_rule": "n = clip(ceil(cap.length / (sphere_spacing * cap.radius)) + 1, 1, "
                  "max_spheres_per_capsule)",
        "condition": {"apple_rows_removed_for_links": list(AUTH), "margin_mm": MARGIN,
                      "n_chunks": F, "horizon": H},
        "arm5_scales": list(ARM5_SCALES), "finger_scales": list(FINGER_SCALES),
        "geometries": results,
    }
    # 0 에 닿는 조합
    zero = [{"sphere_spacing": g["sphere_spacing"], "max_spheres_per_capsule":
             g["max_spheres_per_capsule"], "n_spheres_link_left_arm_5":
             g["n_spheres_link_left_arm_5"], "arm5_scale": c["arm5_scale"],
             "finger_scale": c["finger_scale"],
             "radius_mm_link_left_arm_5": c["radius_mm_link_left_arm_5"],
             "window": w, "min_slack_mm": c["windows"][w]["min_slack_mm"],
             "worst_shortfall_mm": c["worst_shortfall_mm"]}
            for g in results for c in g["combos"] for w in ("full50", "exec8")
            if c["windows"][w]["feasible_at_10mm"]]
    out["combos_reaching_zero_violations"] = zero
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("\ncombos reaching zero violations:", json.dumps(zero, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
