"""T8c s3 — 격자 대신 **역산**. 위반이 0 이 되는 정확한 반지름과 그 반지름을 내는 배율.

행 하나가 margin 10 mm 를 만족하려면  surface_distance - r >= 10  이어야 한다.
그러므로 한 구 group(= 한 capsule 의 구들) 이 위반 0 이 되는 **최대 반지름**은
    r_crit = min over (chunk, step, sphere in group, kept body) surface_distance  -  10 mm
이고, 이것은 격자 없이 정확하다. r_crit <= 0 이면 그 group 은 반지름을 줄여서는 풀 수 없다.

반지름 -> 배율 역산: 구가 n>=2 개인 capsule 은 r_eff(s)^2 = (r_urdf*s)^2 + (spacing/2)^2 이라
    s = sqrt(r_crit^2 - c) / r_urdf ,  c = (spacing/2)^2
이고 r_crit^2 < c 이면 **어떤 배율로도** 그 반지름에 못 내려간다 (구 사슬의 바닥값).
r_urdf 와 c 는 s1 이 실제 클래스에서 읽은 두 배율의 r_eff 로 푼다. 구가 1 개면 선형이다.
마지막에 역산한 배율로 `UrdfSphereChain` 을 다시 세워 반지름과 위반 수를 **확인**한다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))

T8A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8a'))
T8C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8c'))
OUT = T8C / "s3_critical_radius.json"
MARGIN = 10.0
EXEC_H = 8
AUTH = ("ee_finger_l1", "ee_finger_l2", "link_left_arm_5")
TARGET_LINKS = ("link_left_arm_5", "ee_finger_l1", "ee_finger_l2")


def main() -> None:
    from m3_to_refined_vs_reference import MODEL_XML

    t = np.load(T8A / "m0_clearance_tensor.npz", allow_pickle=True)
    clear1 = t["clearance_m"].astype(np.float64) * 1000.0
    bodies = [str(b) for b in t["bodies"]]
    link_names = [str(n) for n in t["link_names"]]
    r1 = np.asarray(t["radii_m"], np.float64) * 1000.0
    F, H, S, B = clear1.shape
    ai = bodies.index("apple")
    auth_sphere = np.asarray([n in AUTH for n in link_names], bool)
    sd = clear1 + r1[None, None, :, None]                 # surface distance, mm
    keep = np.ones((S, B), bool)
    keep[auth_sphere, ai] = False

    s1 = json.loads((T8C / "s1_radii_by_scale.json").read_text())
    rz = np.load(T8C / "s1_radii_by_scale.npz", allow_pickle=True)
    # capsule group = (link, r at scale 1.0) — 한 link 에 capsule 이 여럿이면 반지름이 다르다
    groups: dict[tuple[str, float], np.ndarray] = {}
    for i, n in enumerate(link_names):
        groups.setdefault((n, round(float(r1[i]), 6)), []).append(i)
    groups = {k: np.asarray(v, np.int64) for k, v in groups.items()}

    # r_eff(s) 의 두 점으로 (r_urdf, c) 를 푼다.  s=1.0 과 s=0.6 / 0.35 의 실제 값 사용.
    def curve_params(idx: np.ndarray, link: str) -> dict:
        if link == "link_left_arm_5":
            keys = ("arm5_1__finger_1", "arm5_0.6__finger_1")
            ss = (1.0, 0.6)
        else:
            keys = ("arm5_1__finger_1", "arm5_1__finger_0.6")
            ss = (1.0, 0.6)
        r_a = float(np.asarray(rz[keys[0]], np.float64)[idx[0]] * 1000.0)
        r_b = float(np.asarray(rz[keys[1]], np.float64)[idx[0]] * 1000.0)
        sa, sb = ss
        # r^2 = (ru*s)^2 + c  ->  두 식 차로 ru, c
        ru2 = (r_a ** 2 - r_b ** 2) / (sa ** 2 - sb ** 2)
        c = r_a ** 2 - ru2 * sa ** 2
        return {"n_spheres": int(idx.size), "r_at_scale_1_mm": r_a,
                "r_urdf_mm": math.sqrt(max(ru2, 0.0)),
                "floor_mm": math.sqrt(max(c, 0.0)),
                "model": "r_eff(s) = sqrt((r_urdf*s)^2 + (spacing/2)^2)" if idx.size > 1
                         else "r_eff(s) = r_urdf*s + L/2"}

    res = {}
    for wname, w in (("full50", slice(0, H)), ("exec8", slice(0, EXEC_H))):
        sdw = np.where(keep[None, None, :, :], sd[:, w], np.inf)
        per = {}
        for (ln, r0), idx in sorted(groups.items()):
            if ln not in TARGET_LINKS:
                continue
            v = sdw[:, :, idx, :]
            r_crit = float(v.min()) - MARGIN
            p = curve_params(idx, ln)
            if r_crit <= 0.0:
                s_crit, note = None, "r_crit <= 0 — 반지름을 줄여서는 풀 수 없다"
            elif p["n_spheres"] > 1 and r_crit <= p["floor_mm"]:
                s_crit, note = None, (f"r_crit {r_crit:.4f} mm <= 구 사슬 바닥값 "
                                      f"{p['floor_mm']:.4f} mm — 어떤 배율로도 못 내려간다")
            elif p["n_spheres"] > 1:
                s_crit = math.sqrt(r_crit ** 2 - p["floor_mm"] ** 2) / p["r_urdf_mm"]
                note = ""
            else:
                s_crit, note = (r_crit - p["floor_mm"]) / p["r_urdf_mm"], ""
            argmin = np.unravel_index(int(np.argmin(v)), v.shape)
            per[f"{ln}__r{r0*1000:.4f}mm"] = {
                "link": ln, "sphere_indices": [int(i) for i in idx],
                **p,
                "min_surface_distance_mm": round(float(v.min()), 4),
                "argmin_chunk": int(argmin[0]), "argmin_step": int(argmin[1]),
                "argmin_sphere": int(idx[argmin[2]]),
                "argmin_body": bodies[int(argmin[3])],
                "r_crit_mm": round(r_crit, 4),
                "scale_crit": None if s_crit is None else round(float(s_crit), 6),
                "note": note,
            }
        res[wname] = per

    out = {
        "definition": "r_crit = min over kept rows of surface_distance - margin(10 mm); "
                      "the largest sphere radius with zero violations for that capsule group",
        "condition": {"apple_rows_removed_for_links": list(AUTH), "margin_mm": MARGIN},
        "groups": res,
    }

    # --- 역산한 배율을 실제 모델로 확인 -----------------------------------------------
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)

    checks = []
    for wname in ("full50", "exec8"):
        cands = [v for v in res[wname].values()
                 if v["link"] == "link_left_arm_5" and v["scale_crit"] is not None]
        if not cands:
            continue
        s_crit = min(v["scale_crit"] for v in cands)
        s_use = math.floor(s_crit * 1000.0) / 1000.0
        by_link = {"link_left_arm_5": s_use, "link_right_arm_5": s_use}
        rob = build_constraint_robot_model(
            scene, link_filter=ARM_LINKS,
            sphere_options={"capsule_radius_scale_by_link": by_link})
        names = list(resolve_link_names(rob, rob.n_spheres))
        assert names == link_names
        rs = np.asarray(rob.sphere_centers_numeric(np.zeros(rob.nq))[1],
                        np.float64).reshape(-1) * 1000.0
        w = slice(0, H) if wname == "full50" else slice(0, EXEC_H)
        cw = np.where(keep[None, None, :, :], sd[:, w] - rs[None, None, :, None], np.inf)
        sel = np.asarray([n == "link_left_arm_5" for n in link_names], bool)
        checks.append({
            "window": wname, "scale_crit_exact": round(s_crit, 6), "scale_applied": s_use,
            "link_left_arm_5_radius_mm": round(float(rs[sel][0]), 4),
            "link_left_arm_5_rows_slack_le0": int(
                ((cw[:, :, sel, :] - MARGIN) <= 0.0).sum()),
            "link_left_arm_5_min_slack_mm": round(float(cw[:, :, sel, :].min()) - MARGIN, 4),
            "all_links_rows_slack_le0": int(((cw - MARGIN) <= 0.0).sum()),
            "all_links_min_slack_mm": round(float(cw.min()) - MARGIN, 4),
            "n_coverage_shortfall": len(rob.coverage_shortfall),
            "worst_shortfall_mm": (round(max(s.shortfall_m for s in rob.coverage_shortfall)
                                         * 1000.0, 4) if rob.coverage_shortfall else 0.0),
            "coverage_shortfall": [{
                "link": s.link, "sphere_radius_mm": round(s.sphere_radius * 1000.0, 4),
                "required_radius_mm": round(s.required_radius * 1000.0, 4),
                "shortfall_mm": round(s.shortfall_m * 1000.0, 4)}
                for s in rob.coverage_shortfall],
        })
    # --- 역산한 배율 조합 (arm5 + 손가락) 을 실제 모델로 확인 ---------------------------
    combo_checks = []
    for wname in ("full50", "exec8"):
        w = slice(0, H) if wname == "full50" else slice(0, EXEC_H)
        per = res[wname]
        by_link = {}
        blocked = []
        for ln in TARGET_LINKS:
            vals = [v["scale_crit"] for v in per.values() if v["link"] == ln]
            if any(v is None for v in vals):
                blocked.append(ln)
                continue
            s_use = math.floor(min(vals) * 1000.0) / 1000.0
            by_link[ln] = s_use
            if ln == "link_left_arm_5":
                by_link["link_right_arm_5"] = s_use
        if blocked:
            combo_checks.append({
                "window": wname, "applied": by_link, "blocked_links": blocked,
                "note": "이 link 은 r_crit <= 0 이라 배율이 정의되지 않는다 — 조합을 세우지 않는다"})
            continue
        rob = build_constraint_robot_model(
            scene, link_filter=ARM_LINKS,
            sphere_options={"capsule_radius_scale_by_link": by_link})
        rs = np.asarray(rob.sphere_centers_numeric(np.zeros(rob.nq))[1],
                        np.float64).reshape(-1) * 1000.0
        rec = {"window": wname, "applied": by_link, "blocked_links": [],
               "n_coverage_shortfall": len(rob.coverage_shortfall),
               "worst_shortfall_mm": (round(max(s_.shortfall_m for s_ in rob.coverage_shortfall)
                                            * 1000.0, 4) if rob.coverage_shortfall else 0.0),
               "coverage_shortfall": [{
                   "link": s_.link, "sphere_radius_mm": round(s_.sphere_radius * 1000.0, 4),
                   "required_radius_mm": round(s_.required_radius * 1000.0, 4),
                   "shortfall_mm": round(s_.shortfall_m * 1000.0, 4)}
                   for s_ in rob.coverage_shortfall],
               "radius_mm_by_scaled_link": {
                   ln: sorted({round(float(v), 4) for v in
                               rs[np.asarray([n == ln for n in link_names], bool)]})
                   for ln in sorted(by_link)},
               "evaluated_windows": {}}
        for ev, evw in (("full50", slice(0, H)), ("exec8", slice(0, EXEC_H))):
            cw = np.where(keep[None, None, :, :], sd[:, evw] - rs[None, None, :, None], np.inf)
            le0 = (cw - MARGIN) <= 0.0
            pairs = []
            for ln in sorted(set(link_names)):
                sel = np.asarray([n == ln for n in link_names], bool)
                for bi, bn in enumerate(bodies):
                    if not keep[sel, bi].any():
                        continue
                    v = cw[:, :, sel, bi]
                    n = int(((v - MARGIN) <= 0.0).sum())
                    if n:
                        pairs.append({"link": ln, "obstacle_body": bn, "n_rows_slack_le0": n,
                                      "min_slack_mm": round(float(v.min()) - MARGIN, 4)})
            rec["evaluated_windows"][ev] = {
                "n_rows_slack_le0": int(le0.sum()),
                "n_chunks_slack_le0": int(le0.any(axis=(1, 2, 3)).sum()),
                "min_slack_mm": round(float(cw.min()) - MARGIN, 4),
                "feasible_at_10mm": bool(le0.sum() == 0),
                "violating_pairs": sorted(pairs, key=lambda p: p["min_slack_mm"])}
        combo_checks.append(rec)
    out["inverted_scale_verified_on_the_real_model"] = checks
    out["inverted_scale_combo_verified_on_the_real_model"] = combo_checks
    print("COMBO", json.dumps(combo_checks, indent=1, ensure_ascii=False))
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for wname, per in res.items():
        print(f"--- {wname} ---")
        for k, v in per.items():
            print(f"  {k:28s} n={v['n_spheres']:2d} r@1={v['r_at_scale_1_mm']:7.3f} "
                  f"floor={v['floor_mm']:7.3f} r_crit={v['r_crit_mm']:8.3f} "
                  f"s_crit={v['scale_crit']}  {v['note']} "
                  f"(worst vs {v['argmin_body']} chunk {v['argmin_chunk']})")
    print(json.dumps(checks, indent=1, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
