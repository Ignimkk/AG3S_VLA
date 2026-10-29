"""T6d 재는 것 3 (c) — 참값 clearance 를 **apple 과 그 밖**으로 가른다.

`m3` 의 최소 clearance 는 손끝이 사과에 닿는 순간 음수가 된다 (그것이 과제다). 안전 판정에
쓰이는 것은 **사과가 아닌 표면**까지의 clearance 다. 같은 metric 을 둘로 나눠 다시 낸다.

실행 창도 함께: control 기록의 `step_in_chunk` 는 0..7 뿐이므로 (실측, `m3b`), 50 스텝 chunk 중
**앞 8 스텝만** 실행됐다. 그 창에서의 값과 50 스텝 전체의 값을 따로 낸다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))

from m3_to_refined_vs_reference import (  # noqa: E402
    ACT_COL, JOINT_ORDER, MODEL_XML, QADR, SCENE_BODIES, load_planning, surface_distances)

SHADOW = _ag3s_workspace_path('outputs/live_test/local_shadow_from_pc/frames.jsonl')
RUNS = {
    "closed_loop_approach": _ag3s_workspace_path('outputs/live_test/local_t6_from_pc/frames.jsonl'),
    "closed_loop_grasp": _ag3s_workspace_path('outputs/live_test/local_grasp_from_pc/frames.jsonl'),
}
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d/m3c_clearance_split.json'))
NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d/m3c_curves.npz'))
EXEC_H = 8


def main() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    mesh_cache: dict = {}

    from AG3S.ag3s.constraints.clearance import resolve_link_names
    from AG3S.ag3s.experiments.reports.grounding_report import (
        ARM_LINKS, build_constraint_robot_model)
    from AG3S.ag3s.experiments.sources.mujoco_source import TransportScene

    scene = TransportScene.attach(m, d)
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS)
    link_names = list(resolve_link_names(robot, robot.n_spheres))
    _, radii = robot.sphere_centers_numeric(np.zeros(robot.nq))
    radii = np.asarray(radii, np.float64).reshape(-1)

    def collidable(g):
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0

    ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_BODIES}
    scene_geoms, geom_body = [], []
    for g in range(m.ngeom):
        if collidable(g) and int(m.geom_bodyid[g]) in set(ids.values()):
            scene_geoms.append(g)
            geom_body.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])))
    apple = np.asarray([i for i, b in enumerate(geom_body) if b == "apple"], np.int64)
    other = np.asarray([i for i, b in enumerate(geom_body) if b != "apple"], np.int64)
    left = np.asarray([i for i, n in enumerate(link_names) if "left" in n or "_l" in n], np.int64)

    def q_from(row, qpos):
        return np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(row[ACT_COL[n]])
                           for n in JOINT_ORDER], np.float64)

    def evaluate(rec, chunk):
        qpos = np.asarray(rec["qpos"], np.float64)
        d.qpos[: qpos.size] = qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        H = chunk.shape[0]
        no_apple = np.full(H, np.nan)
        apple_only = np.full(H, np.nan)
        worst_link = [None] * H
        for s in range(H):
            centres = np.asarray(robot.sphere_centers_numeric(q_from(chunk[s], qpos))[0],
                                 np.float64).reshape(-1, 3)
            dist = surface_distances(m, d, scene_geoms, centres, mesh_cache) - radii[None, :]
            sub = dist[other]
            no_apple[s] = float(sub.min())
            worst_link[s] = link_names[int(np.unravel_index(int(sub.argmin()), sub.shape)[1])]
            apple_only[s] = float(dist[apple].min())
        return no_apple, apple_only, worst_link

    out: dict = {"exec_horizon": EXEC_H,
                 "metric": "min over (constraint sphere, ground-truth geom) of "
                           "surface_distance - sphere_radius; apple split out",
                 "runs": {}}
    npz: dict = {}

    sh = load_planning(SHADOW)
    ref = np.asarray([r["actions_reference"] for r in sh], np.float64)
    new = np.asarray([r["actions"] for r in sh], np.float64)
    A = np.full((len(sh), 50), np.nan)
    B = np.full((len(sh), 50), np.nan)
    rows = []
    for i, rec in enumerate(sh):
        na0, ap0, wl0 = evaluate(rec, ref[i])
        na1, ap1, wl1 = evaluate(rec, new[i])
        A[i], B[i] = na0, na1
        e = slice(0, EXEC_H)
        rows.append({
            "seq": int(rec["seq"]), "t_step": int(rec["t_step"]),
            "hold_reason": rec.get("hold_reason", ""),
            "reference_no_apple_min_mm_full50": float(np.nanmin(na0)) * 1000.0,
            "refined_no_apple_min_mm_full50": float(np.nanmin(na1)) * 1000.0,
            "reference_no_apple_min_mm_exec8": float(np.nanmin(na0[e])) * 1000.0,
            "refined_no_apple_min_mm_exec8": float(np.nanmin(na1[e])) * 1000.0,
            "reference_apple_min_mm_exec8": float(np.nanmin(ap0[e])) * 1000.0,
            "refined_apple_min_mm_exec8": float(np.nanmin(ap1[e])) * 1000.0,
            "reference_worst_link_exec8": wl0[int(np.nanargmin(na0[e]))],
            "refined_worst_link_exec8": wl1[int(np.nanargmin(na1[e]))],
        })
        if i % 15 == 0:
            print(f"  chunk {i+1}/{len(sh)}")
    for key, tag in (("full50", "full50"), ("exec8", "exec8")):
        dl = [r[f"refined_no_apple_min_mm_{tag}"] - r[f"reference_no_apple_min_mm_{tag}"] for r in rows]
        out[f"shadow_no_apple_{tag}"] = {
            "n_chunks": len(rows),
            "n_refined_worse": int(sum(1 for v in dl if v < -1e-6)),
            "n_refined_better": int(sum(1 for v in dl if v > 1e-6)),
            "worst_degradation_mm": float(min(dl)),
            "best_improvement_mm": float(max(dl)),
            "chunks_refined_worse": [
                {"seq": rows[i]["seq"], "delta_mm": dl[i],
                 "reference_mm": rows[i][f"reference_no_apple_min_mm_{tag}"],
                 "refined_mm": rows[i][f"refined_no_apple_min_mm_{tag}"],
                 "refined_worst_link": rows[i].get("refined_worst_link_exec8")}
                for i in range(len(rows)) if dl[i] < -1e-6],
            "reference_global_min_mm": float(min(r[f"reference_no_apple_min_mm_{tag}"] for r in rows)),
            "refined_global_min_mm": float(min(r[f"refined_no_apple_min_mm_{tag}"] for r in rows)),
        }
        print(tag, json.dumps({k: v for k, v in out[f"shadow_no_apple_{tag}"].items()
                               if k != "chunks_refined_worse"}))
    out["shadow_chunks"] = rows
    npz["shadow_no_apple_reference_mm"] = A * 1000.0
    npz["shadow_no_apple_refined_mm"] = B * 1000.0

    for label, path in RUNS.items():
        recs = load_planning(path)
        C = np.full((len(recs), 50), np.nan)
        D2 = np.full((len(recs), 50), np.nan)
        for i, rec in enumerate(recs):
            na, ap, _ = evaluate(rec, np.asarray(rec["actions"], np.float64))
            C[i], D2[i] = na, ap
        npz[f"{label}_no_apple_mm"] = C * 1000.0
        npz[f"{label}_apple_mm"] = D2 * 1000.0
        out["runs"][label] = {
            "no_apple_min_mm_full50": float(np.nanmin(C)) * 1000.0,
            "no_apple_min_mm_exec8": float(np.nanmin(C[:, :EXEC_H])) * 1000.0,
            "apple_min_mm_full50": float(np.nanmin(D2)) * 1000.0,
            "apple_min_mm_exec8": float(np.nanmin(D2[:, :EXEC_H])) * 1000.0,
        }
        print(label, json.dumps(out["runs"][label]))

    np.savez_compressed(NPZ, **npz)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT, NPZ)


if __name__ == "__main__":
    main()
