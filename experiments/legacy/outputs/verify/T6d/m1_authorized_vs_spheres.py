"""T6d 재는 것 1 — AG3S -> cuRobo 단계.

`ClearancePolicy.margin_matrix` 의 `is_authorized = [name in authorized for name in links]`
(`AG3S/ag3s/constraints/clearance.py:280`) 는 정확한 문자열 일치다. 두 집합을 그대로 찍는다:

  authorized = ClearancePolicy.authorized_links(context)   (config.DEFAULT_CONTACT_LINKS)
  links      = ConstraintSetBuilder.sphere_link_names      (serve_safe 와 같은 경로로 만든 모델)

출력은 JSON 하나. 해석을 쓰지 않는다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))

MODEL_XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d/m1_authorized_vs_spheres.json'))


def main() -> None:
    import mujoco

    from AG3S.ag3s.config import DEFAULT_CONTACT_LINKS, AG3SConfig
    from AG3S.ag3s.constraints.clearance import ClearancePolicy, resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS,
        build_constraint_robot_model,
        build_robot_model,
    )
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene
    from AG3S.ag3s.types import ContactPolicyContext, Manipulator, Phase, SourceType

    mj_model = mujoco.MjModel.from_xml_path(MODEL_XML)
    scene = TransportScene.attach(mj_model, mujoco.MjData(mj_model))

    filter_robot = build_robot_model(scene)
    out: dict = {
        "model_xml": MODEL_XML,
        "ARM_LINKS": list(ARM_LINKS),
        "DEFAULT_CONTACT_LINKS": {k: list(v) for k, v in DEFAULT_CONTACT_LINKS.items()},
        "filter_robot": {
            "n_spheres": int(filter_robot.n_spheres),
            "unique_links": sorted(set(filter_robot.sphere_link_names)),
        },
        "constraint_models": {},
    }

    cfg = AG3SConfig.from_dict({})
    policy = ClearancePolicy.from_config(cfg.contact, cfg.geometry, cfg.support_surface)

    for label, link_filter in (("arms (ARM_LINKS)", ARM_LINKS), ("all (link_filter=None)", None)):
        robot = build_constraint_robot_model(scene, link_filter=link_filter)
        names = resolve_link_names(robot, robot.n_spheres)
        counts: dict[str, int] = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1
        entry: dict = {
            "n_spheres": int(robot.n_spheres),
            "sphere_link_names_unique": sorted(set(names)),
            "sphere_count_per_link": dict(sorted(counts.items())),
            "contexts": {},
        }
        for phase in (Phase.TRANSIT, Phase.APPROACH, Phase.PRE_GRASP, Phase.GRASP):
            for manips in (("left",), ("right",), ("left", "right")):
                ctx = ContactPolicyContext.make(phase, [Manipulator.parse(m) for m in manips])
                authorized = policy.authorized_links(ctx)
                overlap = sorted(set(names) & set(authorized))
                is_auth = np.asarray([n in authorized for n in names], bool)
                sources = [SourceType.TARGET, SourceType.OBJECT, SourceType.SUPPORT_SURFACE]
                mm = policy.margin_matrix(names, sources, context=ctx, target_grounded=True)
                target_col = mm[:, 0]
                entry["contexts"][f"{phase.value}|{'+'.join(manips)}"] = {
                    "authorized_links": sorted(authorized),
                    "n_authorized_links": len(authorized),
                    "overlap_with_sphere_links": overlap,
                    "n_overlap": len(overlap),
                    "n_spheres_authorized": int(is_auth.sum()),
                    "target_margin_unique_mm": sorted(
                        {round(float(v) * 1000.0, 6) for v in target_col}
                    ),
                    "target_margin_min_mm": float(target_col.min()) * 1000.0,
                    "target_margin_max_mm": float(target_col.max()) * 1000.0,
                    "n_spheres_relaxed_below_full": int(
                        (target_col < policy.safety_margin - 1e-12).sum()
                    ),
                }
        out["constraint_models"][label] = entry

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
