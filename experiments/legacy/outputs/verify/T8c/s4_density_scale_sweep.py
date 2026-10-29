"""T8c s4 — 축을 하나 더. **구 개수(density) × 반지름 배율** 2 차원.

왜 축이 필요한가 (계산이 아니라 실측):
  `_effective_radius` 가 구 n>=2 일 때 sqrt((r_urdf*s)^2 + (spacing/2)^2) 이고
  spacing = capsule 길이 / (n-1) 이다. 그래서 s -> 0 이어도 유효 반지름은 spacing/2 아래로
  안 내려간다. `link_*_arm_5` 는 URDF capsule r=75 mm L=250 mm, 기본 n=5 -> spacing/2=31.25 mm.

density 를 바꾸면 **구 중심이 달라진다** — T8a tensor 를 재사용할 수 없다. 그래서 density 마다
T8a m0 와 같은 방법으로 clearance tensor 를 새로 만든다 (해석적 point-surface, mj_geomDistance
미사용). 그 안에서 반지름 배율은 tensor 를 다시 읽는 것으로 끝난다 (중심 불변).

density 손잡이는 **전역**이다 (`sphere_spacing`, `max_spheres_per_capsule`) — link 별이 아니다.
`--max-spheres-per-capsule` 만 올려서는 arm_5 의 구가 안 늘어난다 (n 은 spacing 으로 정해지고
cap 은 상한일 뿐). 둘을 함께 준다.

길이 축(근사): `link_*_arm_5` capsule 은 link frame z 가 -0.225..+0.025 이고 r=75 mm 라
표면이 z=-0.300 까지 간다. `ee_finger_l1` 뿌리는 z=-0.2278 이므로 이 capsule 이 손목·손바닥·
손가락을 덮는다. 길이를 줄이는 flag 는 없으므로 **local z < -0.15 인 arm_5 구를 제외**하는
것으로 근사한다. 근사다 — 실제로 capsule 을 짧게 하면 남은 구의 간격·반지름도 달라진다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import math
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))

from m3_to_refined_vs_reference import (  # noqa: E402
    ACT_COL, JOINT_ORDER, MODEL_XML, QADR, SCENE_BODIES, SHADOW, load_planning, surface_distances)

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8c/s4_density_scale_sweep.json'))
NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8c/s4_density_curves.npz'))
MARGIN = 10.0
EXEC_H = 8
AUTH = ("ee_finger_l1", "ee_finger_l2", "link_left_arm_5")
ARM5_LINKS = ("link_left_arm_5", "link_right_arm_5")
FINGER_LINKS = ("ee_finger_l1", "ee_finger_l2")
DENSITIES = ((1.0, 8), (0.5, 8), (0.29, 13), (0.17, 21))
ARM5_SCALES = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1)
FINGER_SCALES = (1.0, 0.6, 0.35)
GRIPPER_Z_CUT = -0.15


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
    G = len(scene_geoms)
    ai = bodies.index("apple")

    recs = load_planning(SHADOW)
    chunks = np.asarray([r["actions_reference"] for r in recs], np.float64)
    F, H = chunks.shape[0], chunks.shape[1]

    def q_from(row, qpos):
        return np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(row[ACT_COL[n]])
                           for n in JOINT_ORDER], np.float64)

    def build(spacing, cap, by_link=None):
        opts = {"sphere_spacing": spacing, "max_spheres_per_capsule": cap}
        if by_link:
            opts["capsule_radius_scale_by_link"] = by_link
        return build_constraint_robot_model(scene, link_filter=ARM_LINKS, sphere_options=opts)

    def tensor_for(robot) -> np.ndarray:
        """(F, H, S, B) surface distance in mm (구 반지름을 빼기 전)."""
        S = robot.n_spheres
        sd = np.empty((F, H, S, len(bodies)), np.float32)
        for i, rec in enumerate(recs):
            qpos = np.asarray(rec["qpos"], np.float64)
            d.qpos[: qpos.size] = qpos
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            centres = np.empty((H, S, 3), np.float64)
            for s in range(H):
                centres[s] = np.asarray(
                    robot.sphere_centers_numeric(q_from(chunks[i, s], qpos))[0],
                    np.float64).reshape(-1, 3)
            flat = centres.reshape(-1, 3)
            dist = np.empty((G, flat.shape[0]), np.float64)
            B = 1200
            for a in range(0, flat.shape[0], B):
                dist[:, a : a + B] = surface_distances(m, d, scene_geoms, flat[a : a + B],
                                                       mesh_cache)
            cl = dist.reshape(G, H, S).transpose(1, 2, 0)
            for bi in range(len(bodies)):
                sd[i, :, :, bi] = (cl[:, :, body_of_geom == bi].min(axis=2) * 1000.0)
        return sd

    def evaluate(sd, rs_mm, link_names, keep_sphere, tag_pairs=True):
        """남은 위반. keep_sphere: (S,) bool — False 면 그 구의 모든 행을 뺀다."""
        S = len(link_names)
        auth = np.asarray([n in AUTH for n in link_names], bool)
        keep = np.ones((S, len(bodies)), bool)
        keep[auth, ai] = False
        keep &= keep_sphere[:, None]
        res = {}
        uniq = sorted(set(link_names))
        lo = np.asarray([uniq.index(n) for n in link_names], np.int64)
        for wname, w in (("full50", slice(0, H)), ("exec8", slice(0, EXEC_H))):
            cw = np.where(keep[None, None, :, :],
                          sd[:, w] - rs_mm[None, None, :, None].astype(np.float32),
                          np.float32(np.inf))
            le0 = (cw - MARGIN) <= 0.0
            r = {"n_rows_slack_le0": int(le0.sum()),
                 "n_rows_total": int(np.isfinite(cw).sum()),
                 "n_chunks_slack_le0": int(le0.any(axis=(1, 2, 3)).sum()),
                 "n_chunks": F,
                 "global_min_slack_mm": round(float(cw.min()) - MARGIN, 4),
                 "feasible_at_10mm": bool(le0.sum() == 0),
                 "per_chunk_min_slack_mm": [round(float(x), 4)
                                            for x in (cw.min(axis=(1, 2, 3)) - MARGIN)]}
            if tag_pairs:
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
                                      "n_chunks_slack_le0": int((cm - MARGIN <= 0.0).sum()),
                                      "min_clearance_mm": round(float(v.min()), 4),
                                      "min_slack_mm": round(float(v.min()) - MARGIN, 4)})
                r["violating_pairs"] = sorted(pairs, key=lambda p: p["min_slack_mm"])
            res[wname] = r
        return res

    def r_crit_groups(sd, link_names, radii1_mm, local_z, keep_sphere):
        """capsule group 별 r_crit 과 그것을 내는 최소 행."""
        S = len(link_names)
        auth = np.asarray([n in AUTH for n in link_names], bool)
        keep = np.ones((S, len(bodies)), bool)
        keep[auth, ai] = False
        keep &= keep_sphere[:, None]
        groups: dict[tuple, list] = {}
        for i, n in enumerate(link_names):
            groups.setdefault((n, round(float(radii1_mm[i]), 4)), []).append(i)
        out = {}
        for wname, w in (("full50", slice(0, H)), ("exec8", slice(0, EXEC_H))):
            per = {}
            for (ln, r0), idx in sorted(groups.items()):
                if ln not in AUTH:
                    continue
                idx = np.asarray(idx, np.int64)
                v = np.where(keep[None, None, idx, :], sd[:, w][:, :, idx, :],
                             np.float32(np.inf))
                mn = float(v.min())
                per[f"{ln}__r{r0:.4f}mm"] = {
                    "link": ln, "n_spheres": int(idx.size), "r_at_scale_1_mm": float(r0),
                    "min_surface_distance_mm": round(mn, 4),
                    "r_crit_mm": round(mn - MARGIN, 4),
                    "local_z_range_m": [round(float(local_z[idx].min()), 4),
                                        round(float(local_z[idx].max()), 4)]}
            out[wname] = per
        return out

    results = []
    curves = {}
    t0 = time.time()
    for spacing, cap in DENSITIES:
        base = build(spacing, cap)
        link_names = list(resolve_link_names(base, base.n_spheres))
        radii1 = np.asarray(base.sphere_centers_numeric(np.zeros(base.nq))[1],
                            np.float64).reshape(-1) * 1000.0
        local_z = np.asarray([c[2] for (_, c, _) in base._local_spheres], np.float64)
        S = len(link_names)
        n_by_link = {ln: int(sum(1 for n in link_names if n == ln)) for ln in sorted(set(link_names))}
        print(f"\n#### density spacing={spacing} cap={cap}: {S} spheres, "
              f"arm5 n={n_by_link['link_left_arm_5']}, "
              f"arm5 r={radii1[[i for i, n in enumerate(link_names) if n == 'link_left_arm_5'][0]]:.4f} mm",
              flush=True)
        sd = tensor_for(base)
        print(f"  tensor built [{time.time()-t0:6.1f} s]", flush=True)

        keep_all = np.ones(S, bool)
        keep_short = np.asarray(
            [not (n in ARM5_LINKS and z < GRIPPER_Z_CUT) for n, z in zip(link_names, local_z)],
            bool)
        n_dropped = int((~keep_short).sum())

        dens = {
            "sphere_spacing": spacing, "max_spheres_per_capsule": cap,
            "n_spheres_total": S, "n_spheres_by_link": n_by_link,
            "arm5_n_spheres": n_by_link["link_left_arm_5"],
            "radius_mm_by_link_at_scale_1": {
                ln: sorted({round(float(v), 4)
                            for v in radii1[[i for i, n in enumerate(link_names) if n == ln]]})
                for ln in sorted(set(link_names))},
            "arm5_local_z_m": sorted({round(float(z), 4) for n, z in zip(link_names, local_z)
                                      if n == "link_left_arm_5"}),
            "length_axis_approx": {
                "rule": f"drop arm_5 spheres with capsule-frame z < {GRIPPER_Z_CUT}",
                "n_spheres_dropped": n_dropped,
                "dropped_local_z_m": sorted({round(float(z), 4)
                                             for n, z in zip(link_names, local_z)
                                             if n in ARM5_LINKS and z < GRIPPER_Z_CUT})},
            "r_crit_groups_full_capsule": r_crit_groups(sd, link_names, radii1, local_z, keep_all),
            "r_crit_groups_shortened_capsule_approx": r_crit_groups(
                sd, link_names, radii1, local_z, keep_short),
            "combos": [], "length_axis_combos": [],
        }
        for f in FINGER_SCALES:
            for a in ARM5_SCALES:
                by = {}
                if a != 1.0:
                    by.update({ln: a for ln in ARM5_LINKS})
                if f != 1.0:
                    by.update({ln: f for ln in FINGER_LINKS})
                rob = build(spacing, cap, by or None)
                names2 = list(resolve_link_names(rob, rob.n_spheres))
                assert names2 == link_names
                rs = np.asarray(rob.sphere_centers_numeric(np.zeros(rob.nq))[1],
                                np.float64).reshape(-1) * 1000.0
                w = evaluate(sd, rs, link_names, keep_all)
                a5r = float(rs[[i for i, n in enumerate(link_names)
                                if n == "link_left_arm_5"][0]])
                entry = {
                    "key": f"n{n_by_link['link_left_arm_5']}__arm5_{a:g}__finger_{f:g}",
                    "arm5_scale": a, "finger_scale": f,
                    "radius_mm_link_left_arm_5": round(a5r, 4),
                    "radius_mm_ee_finger_l1": sorted({
                        round(float(v), 4) for v in
                        rs[[i for i, n in enumerate(link_names) if n == "ee_finger_l1"]]}),
                    "n_coverage_shortfall": len(rob.coverage_shortfall),
                    "worst_shortfall_mm": (round(max(s.shortfall_m for s in rob.coverage_shortfall)
                                                 * 1000.0, 4) if rob.coverage_shortfall else 0.0),
                    "coverage_shortfall": [{
                        "link": s.link, "sphere_radius_mm": round(s.sphere_radius * 1000.0, 4),
                        "required_radius_mm": round(s.required_radius * 1000.0, 4),
                        "shortfall_mm": round(s.shortfall_m * 1000.0, 4)}
                        for s in rob.coverage_shortfall],
                    "windows": w,
                }
                dens["combos"].append(entry)
                curves[f"{entry['key']}__full50"] = np.asarray(
                    w["full50"]["per_chunk_min_slack_mm"], np.float64)
                print(f"  n={n_by_link['link_left_arm_5']:2d} arm5 x{a:<4g} finger x{f:<5g} "
                      f"L5 r={a5r:6.2f} | full50 rows {w['full50']['n_rows_slack_le0']:6d} "
                      f"chunks {w['full50']['n_chunks_slack_le0']:3d} "
                      f"min {w['full50']['global_min_slack_mm']:8.2f} | "
                      f"exec8 rows {w['exec8']['n_rows_slack_le0']:5d} "
                      f"chunks {w['exec8']['n_chunks_slack_le0']:3d} "
                      f"min {w['exec8']['global_min_slack_mm']:8.2f} | "
                      f"shortfall {entry['worst_shortfall_mm']:6.2f} mm", flush=True)
                if f == 1.0:
                    ws = evaluate(sd, rs, link_names, keep_short)
                    dens["length_axis_combos"].append({
                        "key": entry["key"] + "__shortcapsule",
                        "arm5_scale": a, "finger_scale": f,
                        "radius_mm_link_left_arm_5": round(a5r, 4),
                        "n_spheres_dropped": n_dropped, "windows": ws})
                    print(f"      [short capsule approx, {n_dropped} arm5 spheres dropped] "
                          f"full50 rows {ws['full50']['n_rows_slack_le0']:6d} "
                          f"chunks {ws['full50']['n_chunks_slack_le0']:3d} "
                          f"min {ws['full50']['global_min_slack_mm']:8.2f} | "
                          f"exec8 rows {ws['exec8']['n_rows_slack_le0']:5d} "
                          f"min {ws['exec8']['global_min_slack_mm']:8.2f}", flush=True)
        results.append(dens)
        del sd

    out = {
        "method": "density 마다 clearance tensor 를 새로 만든다 (기록된 qpos + chunk action -> "
                  "구 중심 -> MuJoCo 충돌 geom 표면까지 해석적 거리; mj_geomDistance 미사용). "
                  "그 안에서 반지름 배율은 tensor 재독 (중심 불변).",
        "condition": {"apple_rows_removed_for_links": list(AUTH), "margin_mm": MARGIN,
                      "record": SHADOW, "variant": "actions_reference",
                      "n_chunks": F, "horizon": H, "bodies": bodies},
        "density_knobs": "sphere_spacing 과 max_spheres_per_capsule 은 전역이다 (link 별 아님). "
                         "n 은 ceil(L / (sphere_spacing * r_urdf)) + 1 로 정해지고 "
                         "max_spheres_per_capsule 은 상한일 뿐이라 cap 만 올려서는 안 늘어난다.",
        "radius_rule": "n=1: r*s + L/2 ; n>=2: sqrt((r*s)^2 + (spacing/2)^2), "
                       "spacing = L/(n-1) from the UNSCALED radius -> floor = spacing/2",
        "length_axis_is_an_approximation": (
            "capsule 길이를 줄이는 flag 가 없어 arm_5 구 중 capsule frame z < "
            f"{GRIPPER_Z_CUT} 인 것을 제외하는 것으로 근사했다. 실제로 capsule 을 짧게 하면 "
            "남은 구의 spacing 과 반지름도 함께 달라진다 — 그 효과는 여기 없다."),
        "arm5_links_scaled": list(ARM5_LINKS),
        "finger_links_scaled": list(FINGER_LINKS),
        "arm5_scales": list(ARM5_SCALES), "finger_scales": list(FINGER_SCALES),
        "densities": results,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(NPZ, **curves)
    print("\nwrote", OUT, NPZ, f"[{time.time()-t0:.1f} s]")


if __name__ == "__main__":
    main()
