"""T7a s0 — offline constraint model 이 T6d 가 기록한 것과 같은가 + link -> joint chain.

두 가지를 낸다.
1. 현재 `build_constraint_robot_model(link_filter=ARM_LINKS)` 의 sphere link 이름 순서와
   반지름을 T6d 의 `m1b_query_index_identity.json` 과 대조 (구 모델이 그대로인가).
2. MuJoCo kinematic tree 를 거꾸로 올라가 각 link body 의 **joint chain** 을 만든다.
   (link -> 그 link 를 움직이는 관절 전체)
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))

MODEL_XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
T6D = _ag3s_workspace_path('outputs/verify/T6d/m1b_query_index_identity.json')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T7a/s0_model_identity.json'))


def main() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    scene = TransportScene.attach(m, d)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    link_names = list(resolve_link_names(robot, robot.n_spheres))
    _, radii = robot.sphere_centers_numeric(np.zeros(robot.nq))
    radii = np.asarray(radii, np.float64).reshape(-1)

    t6 = json.load(open(T6D))
    ref_names = t6["candidate_models"]["arms"]["ordered_sphere_link_names"]
    same_names = list(ref_names) == link_names
    per_link_r = {}
    for n, r in zip(link_names, radii):
        per_link_r.setdefault(n, []).append(round(float(r) * 1000.0, 3))

    # link -> joint chain (부모로 거슬러 올라가며 관절 이름을 모은다)
    def chain(body_name: str) -> list[str]:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        js: list[str] = []
        while b > 0:
            n_j = int(m.body_jntnum[b])
            a = int(m.body_jntadr[b])
            for k in range(n_j):
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, a + k)
                if nm:
                    js.append(nm)
            b = int(m.body_parentid[b])
        return list(reversed(js))

    chains = {}
    missing = []
    for n in sorted(set(link_names)):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        if bid < 0:
            missing.append(n)
            chains[n] = None
        else:
            chains[n] = chain(n)

    out = {
        "model_xml": MODEL_XML,
        "n_spheres": int(robot.n_spheres),
        "sphere_link_names_match_T6d_m1b": bool(same_names),
        "t6d_reference": T6D,
        "radius_mm_by_link": {k: sorted(set(v)) for k, v in per_link_r.items()},
        "n_spheres_by_link": {k: len(v) for k, v in per_link_r.items()},
        "link_bodies_missing_in_mujoco": missing,
        "joint_chain_by_link": chains,
        "sphere_link_names": link_names,
        "sphere_radii_mm": [round(float(r) * 1000.0, 4) for r in radii],
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("n_spheres", robot.n_spheres, "names_match_T6d", same_names)
    for k in sorted(chains):
        print(f"  {k:20s} n_sph={out['n_spheres_by_link'][k]:3d} r_mm={out['radius_mm_by_link'][k]} "
              f"chain={chains[k]}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
