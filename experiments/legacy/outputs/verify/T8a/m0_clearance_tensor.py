"""T8a m0 — shadow `actions_reference` 전량의 (chunk, step, sphere, body) clearance tensor.

방법은 T7a s1 과 같다 (`outputs/verify/T7a/s1_active_rows.py`):
기록된 `qpos` 로 씬을 놓고, chunk 의 action 행으로 AG3S 제약 구 120 개의 중심을 구한 뒤
MuJoCo 충돌 geom **표면까지 해석적 거리**를 잰다. `mj_geomDistance` 를 쓰지 않는다 (distmax 포화 없음).
차이는 셋:
  * shadow 의 `reference` 변형 하나만 (성공 궤적).
  * exec8 이 아니라 **horizon 50 전량**.
  * geom 을 body 별로 묶어 저장 — apple / table / crate 를 따로 물을 수 있게.

clearance = surface_distance(sphere centre, body 의 가장 가까운 충돌 geom) - sphere_radius   [m]
margin_slack(margin) = clearance - margin
box(table·crate) 는 해석적 정확값, mesh(apple·banana·orange·pear) 는 정점 최소거리 = 상한.
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

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T8a'))
NPZ = OUT / "m0_clearance_tensor.npz"
META = OUT / "m0_clearance_tensor.meta.json"


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
    S = radii.size

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
    print(f"{S} spheres · {G} scene geoms · bodies {bodies}", flush=True)
    geom_info = []
    for g, b in zip(scene_geoms, geom_body):
        t = int(m.geom_type[g])
        nv = int(m.mesh_vertnum[int(m.geom_dataid[g])]) if t == mujoco.mjtGeom.mjGEOM_MESH else 0
        geom_info.append({"geom_id": int(g), "body": b, "type": t, "mesh_verts": nv})

    recs = load_planning(SHADOW)
    chunks = np.asarray([r["actions_reference"] for r in recs], np.float64)   # (F, H, 16)
    F, H = chunks.shape[0], chunks.shape[1]
    print(f"shadow reference chunks {chunks.shape}", flush=True)

    def q_from(row, qpos):
        return np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(row[ACT_COL[n]])
                           for n in JOINT_ORDER], np.float64)

    clear = np.full((F, H, S, len(bodies)), np.nan, np.float32)
    t0 = time.time()
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
            dist[:, a : a + B] = surface_distances(m, d, scene_geoms, flat[a : a + B], mesh_cache)
        cl = dist.reshape(G, H, S).transpose(1, 2, 0) - radii[None, :, None]   # (H, S, G)
        for bi in range(len(bodies)):
            clear[i, :, :, bi] = cl[:, :, body_of_geom == bi].min(axis=2)
        if i % 10 == 0 or i == F - 1:
            ai = bodies.index("apple")
            print(f"  chunk {i+1}/{F}  min_clearance_apple={clear[i,:,:,ai].min()*1000:9.3f} mm "
                  f" min_clearance_all={clear[i].min()*1000:9.3f} mm  "
                  f"[{time.time()-t0:6.1f} s]", flush=True)

    np.savez_compressed(
        NPZ, clearance_m=clear, bodies=np.asarray(bodies),
        link_names=np.asarray(link_names), radii_m=radii,
        object_poses_present=np.asarray([bool(r.get("object_poses")) for r in recs]),
        seq=np.asarray([int(r["seq"]) for r in recs]),
        t_step=np.asarray([int(r["t_step"]) for r in recs]))
    META.write_text(json.dumps({
        "source_record": SHADOW,
        "variant": "actions_reference (shadow 의 성공 궤적)",
        "model_xml": MODEL_XML,
        "n_chunks": F, "horizon": H, "n_spheres": S, "n_scene_geoms": G,
        "bodies": bodies,
        "geoms": geom_info,
        "link_names": link_names,
        "radii_mm": [round(float(r) * 1000.0, 4) for r in radii],
        "method": "offline rebuild from recorded qpos + action chunk; analytic point-to-surface "
                  "distance to MuJoCo scene collision geoms (no mj_geomDistance, no distmax); "
                  "same as outputs/verify/T7a/s1_active_rows.py",
        "clearance_definition": "surface_distance(sphere centre, nearest collidable geom of body) "
                                "- sphere_radius",
        "mesh_note": "box geoms (table 5, crate 13) are exact; mesh geoms "
                     "(apple/banana/orange/pear) use vertex-minimum = upper bound on distance",
        "tensor_shape": "(chunk=%d, step=%d, sphere=%d, body=%d)" % (F, H, S, len(bodies)),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=1, ensure_ascii=False))
    print("wrote", NPZ, "and", META, flush=True)


if __name__ == "__main__":
    main()
