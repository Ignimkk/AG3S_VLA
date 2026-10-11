"""T8e — 두 축의 **조합**: `link_*_arm_5` capsule 을 z_cut 에서 자르고, 손가락만 얇게.

고정: `arm_5` 반지름 배율 1.0 (건드리지 않는다). margin 10 mm. 사과는 authorized 셋
{ee_finger_l1, ee_finger_l2, link_left_arm_5} 에서 제외. 기록은 shadow `actions_reference`.

훑는 것: s_finger ∈ {1.0, 0.6, 0.45, 0.354, 0.30, 0.26, 0.22}  (`ee_finger_l1`·`ee_finger_l2`)
       × z_cut ∈ {none, -0.100, -0.1087}  × 손바닥 {없음, 있음}

**자르기는 근사다.** capsule 길이를 줄이는 flag 가 없으므로 `link_*_arm_5` 의 구 중
capsule frame z < z_cut 인 것을 행에서 빼는 것으로 근사한다. 남는 구의 반지름은 81.25 mm
그대로다 — 실제로 capsule 을 짧게 만들면 구 개수가 줄어 `_effective_radius` 의 spacing 항이
커져 반지름이 달라진다. 그 차이는 여기서 재지 않는다.

**손바닥**은 `ee_left`/`ee_right` 다. `ARM_LINKS` 에 없어서 지금 제약 모델에 없다.
`gap_filling_capsules` 에는 있으므로 `link_filter` 에 더하면 들어온다 (각 3 capsule).
구 집합이 커지므로 tensor 를 새로 만든다 (T8a m0 와 같은 방법, mj_geomDistance 미사용).
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

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8e/s1_cut_plus_finger.json'))
NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8e/s1_curves.npz'))
MARGIN = 10.0
EXEC_H = 8
AUTH = ("ee_finger_l1", "ee_finger_l2", "link_left_arm_5")
ARM5_LINKS = ("link_left_arm_5", "link_right_arm_5")
FINGER_LINKS = ("ee_finger_l1", "ee_finger_l2")
PALM_LINKS = ("ee_left", "ee_right")
FINGER_SCALES = (1.0, 0.6, 0.45, 0.354, 0.30, 0.26, 0.22)
Z_CUTS = (None, -0.100, -0.1087)


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

    LINKS_WITH_PALM = tuple(ARM_LINKS) + PALM_LINKS
    def build(by_link=None):
        opts = {"capsule_radius_scale_by_link": by_link} if by_link else None
        return build_constraint_robot_model(scene, link_filter=LINKS_WITH_PALM,
                                            sphere_options=opts)

    base = build()
    link_names = list(resolve_link_names(base, base.n_spheres))
    radii1 = np.asarray(base.sphere_centers_numeric(np.zeros(base.nq))[1],
                        np.float64).reshape(-1) * 1000.0
    local_z = np.asarray([c[2] for (_, c, _) in base._local_spheres], np.float64)
    S = len(link_names)
    is_palm = np.asarray([n in PALM_LINKS for n in link_names], bool)
    n_by_link = {ln: int(sum(1 for n in link_names if n == ln)) for ln in sorted(set(link_names))}
    print(f"constraint model with palm: {S} spheres ({int(is_palm.sum())} on the palm)")
    print("  palm radii mm:", sorted({round(float(v), 4) for v in radii1[is_palm]}))
    print("  arm5 local z:", sorted({round(float(z), 4) for n, z in zip(link_names, local_z)
                                     if n == "link_left_arm_5"}))

    # --- tensor (surface distance, mm) -------------------------------------------------
    t0 = time.time()
    sd = np.empty((F, H, S, len(bodies)), np.float32)
    for i, rec in enumerate(recs):
        qpos = np.asarray(rec["qpos"], np.float64)
        d.qpos[: qpos.size] = qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        centres = np.empty((H, S, 3), np.float64)
        for s in range(H):
            centres[s] = np.asarray(
                base.sphere_centers_numeric(q_from(chunks[i, s], qpos))[0],
                np.float64).reshape(-1, 3)
        flat = centres.reshape(-1, 3)
        dist = np.empty((G, flat.shape[0]), np.float64)
        B = 1200
        for a in range(0, flat.shape[0], B):
            dist[:, a : a + B] = surface_distances(m, d, scene_geoms, flat[a : a + B], mesh_cache)
        cl = dist.reshape(G, H, S).transpose(1, 2, 0)
        for bi in range(len(bodies)):
            sd[i, :, :, bi] = cl[:, :, body_of_geom == bi].min(axis=2) * 1000.0
    print(f"tensor built [{time.time()-t0:.1f} s]")

    uniq = sorted(set(link_names))
    lo = np.asarray([uniq.index(n) for n in link_names], np.int64)
    auth = np.asarray([n in AUTH for n in link_names], bool)

    def keep_mask(z_cut, palm, palm_authorized=False):
        keep_s = np.ones(S, bool)
        if z_cut is not None:
            keep_s &= np.asarray([not (n in ARM5_LINKS and z < z_cut)
                                  for n, z in zip(link_names, local_z)], bool)
        if not palm:
            keep_s &= ~is_palm
        keep = np.ones((S, len(bodies)), bool)
        a = auth.copy()
        if palm and palm_authorized:
            a |= np.asarray([n == "ee_left" for n in link_names], bool)
        keep[a, ai] = False
        keep &= keep_s[:, None]
        return keep, keep_s

    def evaluate(rs_mm, keep):
        res = {}
        for wname, w in (("full50", slice(0, H)), ("exec8", slice(0, EXEC_H))):
            cw = np.where(keep[None, None, :, :],
                          sd[:, w] - rs_mm[None, None, :, None].astype(np.float32),
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
                    pairs.append({"link": ln, "obstacle_body": bn, "n_rows_slack_le0": n,
                                  "n_rows_total": int(v.size),
                                  "n_chunks_slack_le0": int((cm - MARGIN <= 0.0).sum()),
                                  "min_clearance_mm": round(float(v.min()), 4),
                                  "min_slack_mm": round(float(v.min()) - MARGIN, 4)})
            res[wname] = {
                "n_rows_slack_le0": int(le0.sum()),
                "n_rows_total": int(np.isfinite(cw).sum()),
                "n_chunks_slack_le0": int(le0.any(axis=(1, 2, 3)).sum()), "n_chunks": F,
                "global_min_slack_mm": round(float(cw.min()) - MARGIN, 4),
                "feasible_at_10mm": bool(le0.sum() == 0),
                "violating_pairs": sorted(pairs, key=lambda p: p["min_slack_mm"]),
                "per_chunk_min_slack_mm": [round(float(x), 4)
                                           for x in (cw.min(axis=(1, 2, 3)) - MARGIN)]}
        return res

    def r_crit(keep, links):
        """capsule group 별 r_crit (배율 무관 — 중심과 표면거리만 쓴다)."""
        groups: dict[tuple, list] = {}
        for i, n in enumerate(link_names):
            if n in links:
                groups.setdefault((n, round(float(radii1[i]), 4)), []).append(i)
        out = {}
        for wname, w in (("full50", slice(0, H)), ("exec8", slice(0, EXEC_H))):
            per = {}
            for (ln, r0), idx in sorted(groups.items()):
                idx = np.asarray(idx, np.int64)
                v = np.where(keep[None, None, idx, :], sd[:, w][:, :, idx, :], np.float32(np.inf))
                mn = float(v.min())
                per[f"{ln}__r{r0:.4f}mm"] = {
                    "link": ln, "n_spheres": int(idx.size), "r_at_scale_1_mm": r0,
                    "min_surface_distance_mm": round(mn, 4),
                    "r_crit_mm": round(mn - MARGIN, 4)}
            out[wname] = per
        return out

    # --- 대조: T8a M2 를 재현하는가 (자르지 않고, 손바닥 없이, 손가락 1.0) ----------------
    keep0, _ = keep_mask(None, False)
    rs1 = radii1
    chk = evaluate(rs1, keep0)
    print("T8a M2 reproduction (no cut, no palm, finger x1): full50 rows "
          f"{chk['full50']['n_rows_slack_le0']} chunks {chk['full50']['n_chunks_slack_le0']} "
          f"min {chk['full50']['global_min_slack_mm']} | exec8 rows "
          f"{chk['exec8']['n_rows_slack_le0']} min {chk['exec8']['global_min_slack_mm']}")

    rows = []
    curves = {}
    radii_cache = {}
    for f in FINGER_SCALES:
        by = {ln: f for ln in FINGER_LINKS} if f != 1.0 else None
        rob = build(by)
        assert list(resolve_link_names(rob, rob.n_spheres)) == link_names
        rs = np.asarray(rob.sphere_centers_numeric(np.zeros(rob.nq))[1],
                        np.float64).reshape(-1) * 1000.0
        radii_cache[f] = rs
        shortfall = [{"link": s_.link, "sphere_radius_mm": round(s_.sphere_radius * 1000.0, 4),
                      "required_radius_mm": round(s_.required_radius * 1000.0, 4),
                      "shortfall_mm": round(s_.shortfall_m * 1000.0, 4)}
                     for s_ in rob.coverage_shortfall]
        fr = sorted({round(float(v), 4) for v in
                     rs[[i for i, n in enumerate(link_names) if n == "ee_finger_l1"]]})
        for z_cut in Z_CUTS:
            for palm, pauth in ((False, False), (True, False), (True, True)):
                keep, keep_s = keep_mask(z_cut, palm, pauth)
                res = evaluate(rs, keep)
                key = (f"finger_{f:g}__zcut_"
                       f"{'none' if z_cut is None else f'{abs(z_cut)*1000:g}mm'}"
                       f"__palm_{'in_authorized' if (palm and pauth) else ('in' if palm else 'out')}")
                rows.append({
                    "key": key, "finger_scale": f, "arm5_scale": 1.0,
                    "z_cut_m": z_cut, "palm_in_model": palm,
                    "apple_removed_for_ee_left": bool(palm and pauth),
                    "n_spheres_used": int(keep_s.sum()),
                    "n_arm5_spheres_kept": int(sum(1 for i, n in enumerate(link_names)
                                                   if n in ARM5_LINKS and keep_s[i])),
                    "n_arm5_spheres_total": int(sum(1 for n in link_names if n in ARM5_LINKS)),
                    "radius_mm_ee_finger_l1": fr,
                    "radius_mm_ee_finger_l2": sorted({
                        round(float(v), 4) for v in
                        rs[[i for i, n in enumerate(link_names) if n == "ee_finger_l2"]]}),
                    "n_coverage_shortfall": len(shortfall),
                    "worst_shortfall_mm": (round(max(s_["shortfall_mm"] for s_ in shortfall), 4)
                                           if shortfall else 0.0),
                    "coverage_shortfall": shortfall,
                    "windows": res})
                curves[f"{key}__full50"] = np.asarray(
                    res["full50"]["per_chunk_min_slack_mm"], np.float64)
                curves[f"{key}__exec8"] = np.asarray(
                    res["exec8"]["per_chunk_min_slack_mm"], np.float64)
                print(f"  {key:44s} finger_l1 r={fr} | "
                      f"full50 rows {res['full50']['n_rows_slack_le0']:5d} "
                      f"chunks {res['full50']['n_chunks_slack_le0']:3d} "
                      f"min {res['full50']['global_min_slack_mm']:8.2f} "
                      f"feas {int(res['full50']['feasible_at_10mm'])} | "
                      f"exec8 rows {res['exec8']['n_rows_slack_le0']:4d} "
                      f"chunks {res['exec8']['n_chunks_slack_le0']:3d} "
                      f"min {res['exec8']['global_min_slack_mm']:8.2f} "
                      f"feas {int(res['exec8']['feasible_at_10mm'])} | "
                      f"shortfall n={len(shortfall):2d} "
                      f"worst {rows[-1]['worst_shortfall_mm']:6.2f} mm", flush=True)

    # --- 0 에 닿는 가장 큰 s_finger (= 가장 덜 줄이는 값) --------------------------------
    def tag(palm, pauth):
        return "in_authorized" if (palm and pauth) else ("in" if palm else "out")

    def largest_feasible(z_cut, palm, pauth, w):
        cand = [r for r in rows if r["z_cut_m"] == z_cut and r["palm_in_model"] == palm
                and r["apple_removed_for_ee_left"] == bool(palm and pauth)
                and r["windows"][w]["feasible_at_10mm"]]
        return max((r["finger_scale"] for r in cand), default=None)

    feas = {}
    for z_cut in Z_CUTS:
        for palm, pauth in ((False, False), (True, False), (True, True)):
            for w in ("full50", "exec8"):
                k = (f"zcut_{'none' if z_cut is None else f'{abs(z_cut)*1000:g}mm'}"
                     f"__palm_{tag(palm, pauth)}__{w}")
                feas[k] = largest_feasible(z_cut, palm, pauth, w)

    # --- 남는 쌍 (가장 공격적인 s_finger) ------------------------------------------------
    surviving = {}
    for z_cut in Z_CUTS:
        for palm, pauth in ((False, False), (True, False), (True, True)):
            r = [x for x in rows if x["z_cut_m"] == z_cut and x["palm_in_model"] == palm
                 and x["apple_removed_for_ee_left"] == bool(palm and pauth)
                 and x["finger_scale"] == min(FINGER_SCALES)][0]
            k = (f"zcut_{'none' if z_cut is None else f'{abs(z_cut)*1000:g}mm'}"
                 f"__palm_{tag(palm, pauth)}")
            surviving[k] = {w: r["windows"][w]["violating_pairs"] for w in ("full50", "exec8")}

    # --- r_crit (자른 뒤) 손가락 group + 손바닥 ------------------------------------------
    keep_cut_palm, _ = keep_mask(-0.100, True, True)
    keep_cut_nopalm, _ = keep_mask(-0.100, False)
    rcrit = {
        "zcut_100mm__palm_out": r_crit(keep_cut_nopalm, FINGER_LINKS),
        "zcut_100mm__palm_in_authorized": r_crit(keep_cut_palm, FINGER_LINKS + PALM_LINKS),
        "no_cut__palm_out": r_crit(keep0, FINGER_LINKS),
    }

    out = {
        "method": "T8a m0 와 같은 오프라인 재구성 (기록된 qpos + chunk action -> 구 중심 -> "
                  "MuJoCo 충돌 geom 표면까지 해석적 거리; mj_geomDistance 미사용). 구 집합에 "
                  "손바닥 ee_left/ee_right 를 넣어 한 번 만들고, 손바닥 없는 판은 그 구를 "
                  "마스크로 뺀다. 반지름 배율은 tensor 재독으로 끝난다 (중심 불변).",
        "condition": {"apple_rows_removed_for_links": list(AUTH), "margin_mm": MARGIN,
                      "arm5_capsule_radius_scale": 1.0,
                      "finger_links_scaled": list(FINGER_LINKS),
                      "record": SHADOW, "variant": "actions_reference",
                      "n_chunks": F, "horizon": H, "bodies": bodies},
        "cut_is_an_approximation": (
            "capsule 길이를 줄이는 flag 가 없어 link_*_arm_5 의 구 중 capsule frame z < z_cut "
            "인 것을 행에서 빼는 것으로 근사했다. 남는 구의 반지름은 81.25 mm 그대로다 — "
            "실제로 capsule 을 짧게 만들면 구 개수가 줄어 반지름도 달라진다."),
        "palm_links": list(PALM_LINKS),
        "palm_capsules_source": "gap_filling_capsules(scene.model) — ee_left/ee_right 각 3 개. "
                                "ARM_LINKS 에 없어서 현재 제약 모델에는 들어 있지 않다.",
        "model": {"n_spheres_with_palm": S, "n_spheres_by_link": n_by_link,
                  "n_palm_spheres": int(is_palm.sum()),
                  "palm_radius_mm": sorted({round(float(v), 4) for v in radii1[is_palm]}),
                  "arm5_local_z_m": sorted({round(float(z), 4)
                                            for n, z in zip(link_names, local_z)
                                            if n == "link_left_arm_5"})},
        "t8a_m2_reproduction_no_cut_no_palm_finger1": {
            w: {k: chk[w][k] for k in ("n_rows_slack_le0", "n_chunks_slack_le0",
                                       "global_min_slack_mm")} for w in ("full50", "exec8")},
        "finger_scales": list(FINGER_SCALES),
        "z_cuts_m": list(Z_CUTS),
        "largest_finger_scale_reaching_zero": feas,
        "pairs_surviving_the_smallest_finger_scale": surviving,
        "r_crit_by_capsule_group": rcrit,
        "rows": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(NPZ, **curves)
    print("\nlargest finger scale reaching zero:", json.dumps(feas, ensure_ascii=False))
    print("wrote", OUT, NPZ)


if __name__ == "__main__":
    main()
