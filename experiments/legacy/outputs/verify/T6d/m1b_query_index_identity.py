"""T6d 재는 것 1 (b) — 기록의 `max_violation_pair.query` 가 어느 sphere 인가.

`max_violation_pair` 는 `{"block":"esdf","link":...,"query":k}` 를 싣는다. `query` 는
`ConstraintSetBuilder.sphere_link_names` 의 index 다. 두 후보 모델(`arms` / `all`)의 순서를
찍어 기록의 (query, link) 짝이 어느 모델과 맞는지 본다 — 서버 argv 가 기록에 없어서
모델 범위를 이 짝으로만 확정할 수 있다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

MODEL_XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
RUNS = {
    "closed_loop_approach": _ag3s_workspace_path('outputs/live_test/local_t6_from_pc/frames.jsonl'),
    "shadow_approach": _ag3s_workspace_path('outputs/live_test/local_shadow_from_pc/frames.jsonl'),
    "closed_loop_grasp": _ag3s_workspace_path('outputs/live_test/local_grasp_from_pc/frames.jsonl'),
}
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d/m1b_query_index_identity.json'))


def main() -> None:
    import mujoco

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS,
        build_constraint_robot_model,
    )
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    mj_model = mujoco.MjModel.from_xml_path(MODEL_XML)
    scene = TransportScene.attach(mj_model, mujoco.MjData(mj_model))

    models = {}
    for label, lf in (("arms", ARM_LINKS), ("all", None)):
        robot = build_constraint_robot_model(scene, link_filter=lf)
        models[label] = list(resolve_link_names(robot, robot.n_spheres))

    observed: dict[str, dict] = {}
    for run, path in RUNS.items():
        pairs: dict[str, set] = {}
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("kind") != "planning":
                    continue
                p = d.get("max_violation_pair")
                if not p or p.get("query") is None:
                    continue
                pairs.setdefault(str(p["link"]), set()).add(int(p["query"]))
        observed[run] = {k: sorted(v) for k, v in sorted(pairs.items())}

    verdict = {}
    for label, names in models.items():
        ok = True
        detail = []
        for run, pairs in observed.items():
            for link, queries in pairs.items():
                for q in queries:
                    got = names[q] if q < len(names) else "<out-of-range>"
                    if got != link:
                        ok = False
                        detail.append({"run": run, "query": q, "record_link": link, "model_link": got})
        verdict[label] = {"n_spheres": len(names), "consistent": ok, "mismatches": detail[:20]}

    out = {
        "model_xml": MODEL_XML,
        "candidate_models": {k: {"n_spheres": len(v), "ordered_sphere_link_names": v} for k, v in models.items()},
        "observed_pairs": observed,
        "which_model_matches": verdict,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for label, v in verdict.items():
        print(label, v["n_spheres"], "consistent =", v["consistent"], v["mismatches"][:4])
    print("observed:", json.dumps(observed, ensure_ascii=False))


if __name__ == "__main__":
    main()
