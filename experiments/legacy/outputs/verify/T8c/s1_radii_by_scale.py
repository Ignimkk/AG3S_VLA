"""T8c s1 — 배율 s 마다 **유효 구 반지름**과 `coverage_shortfall` 을 실제 클래스에서 읽는다.

반지름은 배율에 선형이 아니다 (`urdf_sphere_chain._effective_radius`):
  구 1 개  -> radius*s + length/2
  구 n 개  -> sqrt((radius*s)^2 + (spacing/2)^2)     spacing 은 **원래** radius 로 정해진다
그래서 배율마다 `UrdfSphereChain` 을 다시 세워 읽는다. 추정하지 않는다.

동시에 **구 중심이 배율에 안 움직인다**는 것을 수치로 확인한다 — T8c 가 T8a tensor 를
`surface_distance = clearance + r` 로 되돌려 쓰는 근거다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))

from m3_to_refined_vs_reference import MODEL_XML  # noqa: E402

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8c/s1_radii_by_scale.json'))
NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8c/s1_radii_by_scale.npz'))
T8A_NPZ = _ag3s_workspace_path('outputs/verify/T8a/m0_clearance_tensor.npz')
ARM5_LINKS = ("link_left_arm_5", "link_right_arm_5")
FINGER_LINKS = ("ee_finger_l1", "ee_finger_l2")
ARM5_SCALES = (1.0, 0.8, 0.6, 0.5, 0.4, 0.3)
FINGER_SCALES = (1.0, 0.6, 0.35)


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene = TransportScene.attach(m, d)

    base = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    base_names = list(resolve_link_names(base, base.n_spheres))
    rng = np.random.default_rng(0)
    q_probe = rng.uniform(-0.5, 0.5, base.nq)
    c_base = np.asarray(base.sphere_centers_numeric(q_probe)[0], np.float64).reshape(-1, 3)
    r_base = np.asarray(base.sphere_centers_numeric(np.zeros(base.nq))[1], np.float64).reshape(-1)

    t8a = np.load(T8A_NPZ, allow_pickle=True)
    t8a_names = [str(x) for x in t8a["link_names"]]
    t8a_radii = np.asarray(t8a["radii_m"], np.float64)
    print("sphere order identical to T8a tensor:", t8a_names == base_names)
    print("radii identical to T8a tensor: max|d| =",
          float(np.abs(t8a_radii - r_base).max()), "m")

    combos = [(a, f) for f in FINGER_SCALES for a in ARM5_SCALES]
    entries = []
    radii_by_combo = {}
    centre_drift = []
    for a, f in combos:
        by_link = {}
        if a != 1.0:
            by_link.update({ln: a for ln in ARM5_LINKS})
        if f != 1.0:
            by_link.update({ln: f for ln in FINGER_LINKS})
        opts = {"capsule_radius_scale_by_link": by_link} if by_link else {}
        rob = build_constraint_robot_model(scene, link_filter=ARM_LINKS, sphere_options=opts)
        names = list(resolve_link_names(rob, rob.n_spheres))
        assert names == base_names, "sphere order changed with scale"
        cen, rad = rob.sphere_centers_numeric(q_probe)
        cen = np.asarray(cen, np.float64).reshape(-1, 3)
        rad = np.asarray(rad, np.float64).reshape(-1)
        drift = float(np.abs(cen - c_base).max())
        centre_drift.append(drift)
        key = f"arm5_{a:g}__finger_{f:g}"
        radii_by_combo[key] = rad
        shortfall = [{
            "link": s.link, "n_spheres": int(s.n_spheres),
            "capsule_radius_mm": round(float(s.capsule_radius) * 1000.0, 4),
            "capsule_length_mm": round(float(s.capsule_length) * 1000.0, 4),
            "sphere_radius_mm": round(float(s.sphere_radius) * 1000.0, 4),
            "required_radius_mm": round(float(s.required_radius) * 1000.0, 4),
            "shortfall_mm": round(float(s.shortfall_m) * 1000.0, 4),
        } for s in rob.coverage_shortfall]
        per_link = {}
        for ln in sorted(set(names)):
            sel = np.asarray([n == ln for n in names], bool)
            per_link[ln] = sorted({round(float(v) * 1000.0, 4) for v in rad[sel]})
        entries.append({
            "key": key, "arm5_scale": a, "finger_scale": f,
            "capsule_radius_scale_by_link": by_link,
            "max_sphere_radius": rob.max_sphere_radius,
            "n_spheres": int(rob.n_spheres),
            "centre_max_abs_drift_from_scale_1_m": drift,
            "radius_mm_by_link": per_link,
            "radius_mm_link_left_arm_5": per_link["link_left_arm_5"],
            "radius_mm_ee_finger_l1": per_link["ee_finger_l1"],
            "radius_mm_ee_finger_l2": per_link["ee_finger_l2"],
            "n_coverage_shortfall": len(shortfall),
            "worst_shortfall_mm": (round(max(s["shortfall_mm"] for s in shortfall), 4)
                                   if shortfall else 0.0),
            "coverage_shortfall": shortfall,
            "coverage_report": rob.coverage_report(),
        })
        print(f"  {key:26s} L5 r={per_link['link_left_arm_5']} mm  "
              f"fl1 r={per_link['ee_finger_l1']}  shortfall {len(shortfall)} "
              f"worst {entries[-1]['worst_shortfall_mm']} mm  centre drift {drift:.3e} m")

    out = {
        "model_xml": MODEL_XML,
        "sphere_order_identical_to_T8a_tensor": t8a_names == base_names,
        "radii_max_abs_diff_from_T8a_tensor_m": float(np.abs(t8a_radii - r_base).max()),
        "centre_invariance_check": {
            "probe_q": "uniform(-0.5, 0.5), seed 0",
            "max_abs_centre_drift_over_all_scales_m": float(max(centre_drift)),
            "note": "sphere_spacing 은 원래 capsule radius 로 정해진다 "
                    "(urdf_sphere_chain._capsule_sphere_centres) — 배율은 중심을 안 움직인다",
        },
        "radius_rule": "1 sphere: r*s + L/2 ; n spheres: sqrt((r*s)^2 + (spacing/2)^2) ; "
                       "spacing from the UNSCALED radius (urdf_sphere_chain._effective_radius)",
        "arm5_links_scaled": list(ARM5_LINKS),
        "finger_links_scaled": list(FINGER_LINKS),
        "arm5_scales": list(ARM5_SCALES),
        "finger_scales": list(FINGER_SCALES),
        "link_names": base_names,
        "combos": entries,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(NPZ, link_names=np.asarray(base_names),
                        radii_scale1_m=r_base, **radii_by_combo)
    print("wrote", OUT, NPZ)


if __name__ == "__main__":
    main()
