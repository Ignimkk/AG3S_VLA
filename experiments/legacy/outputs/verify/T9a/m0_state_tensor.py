"""T9a m0 — 실행 기록의 **실제 상태**를 구 × body clearance 로. rollout 없다.

T8a `m0_clearance_tensor.py` 와 같은 방법이다:
기록된 `qpos` 로 MuJoCo 씬을 놓고 (물체 free joint 포함 — `qpos` 66 에 apple/crate/orange/
pear/banana 의 free joint 가 들어 있어 `object_poses` 와 일치한다), 제약 구 중심을 구한 뒤
씬 충돌 geom **표면까지 해석적 거리**를 잰다. `mj_geomDistance` 를 쓰지 않는다.

다른 점 둘:
  * 구 모델이 이번 서버 설정이다 — `sphere_spacing 0.4` · `capsule_radius_scale 0.2` ·
    `max_spheres_per_capsule 32` · `capsule_extent_by_link {link_*_arm_5: -0.10}`.
    구 266 개 · capsule 36 개가 나와야 하고, 서버 시작 로그와 link 별 반지름을 대조한다.
  * chunk 의 계획 행이 아니라 **control 기록의 실제 `qpos`** 를 쓴다 (312 스텝) —
    "얼마나 가까워졌나" 는 계획이 아니라 실제로 간 자리의 질문이다. planning 39 개도 같이 낸다.

clearance = surface_distance(구 중심, body 의 가장 가까운 충돌 geom) - 구 반지름  [m]
box(table·crate) 는 정확, mesh(apple·banana·orange·pear) 는 정점 최소거리 = 상한.
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
    JOINT_ORDER, MODEL_XML, QADR, SCENE_BODIES, surface_distances)

REC = _ag3s_workspace_path('outputs/live_test/20260926_t8/execute_ep1807/frames.jsonl')
SERVE_LOG = _ag3s_workspace_path('outputs/live_test/20260926_t8/serve_safe_8123.log')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T9a'))
NPZ = OUT / "m0_state_tensor.npz"
META = OUT / "m0_state_tensor.meta.json"
SPHERE_OPTS = {
    "sphere_spacing": 0.4,
    "max_spheres_per_capsule": 32,
    "capsule_radius_scale": 0.2,
    "capsule_extent_by_link": {"link_left_arm_5": -0.10, "link_right_arm_5": -0.10},
}
GRIP_COL = {"left": 7, "right": 15}


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
    robot = build_constraint_robot_model(scene, link_filter=ARM_LINKS,
                                         sphere_options=SPHERE_OPTS)
    link_names = list(resolve_link_names(robot, robot.n_spheres))
    _, radii = robot.sphere_centers_numeric(np.zeros(robot.nq))
    radii = np.asarray(radii, np.float64).reshape(-1)
    S = radii.size
    per_link_r = {ln: sorted({round(float(v) * 1000.0, 2)
                              for v in radii[[i for i, n in enumerate(link_names) if n == ln]]})
                  for ln in sorted(set(link_names))}
    n_by_link = {ln: int(sum(1 for n in link_names if n == ln)) for ln in sorted(set(link_names))}
    print(f"constraint model: {S} spheres, {len(robot.capsules)} capsules", flush=True)
    for ln in sorted(per_link_r):
        print(f"   {ln:18s} n={n_by_link[ln]:3d}  r={per_link_r[ln]} mm", flush=True)
    import dataclasses
    trims = [dataclasses.asdict(t) for t in getattr(robot, "capsule_trims", ())]
    print(f"   capsule_trims: {trims}", flush=True)

    # 서버 시작 로그와 대조
    log = pathlib.Path(SERVE_LOG).read_text()
    server_line = [l for l in log.splitlines() if "지금 모델: 구" in l]
    server_spheres = None
    if server_line:
        import re
        mm = re.search(r"구 (\d+) 개 · capsule (\d+) 개", server_line[0])
        if mm:
            server_spheres = (int(mm.group(1)), int(mm.group(2)))
    print(f"   server log says {server_spheres}, this model {(S, len(robot.capsules))}", flush=True)

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

    recs = [json.loads(l) for l in open(REC)]
    plan = {int(r["seq"]): r for r in recs if r["kind"] == "planning"}
    rows = []
    for r in recs:
        if r["kind"] == "control":
            cmd = None
            pr = plan.get(int(r["chunk_seq"]))
            if pr is not None:
                a = np.asarray(pr["actions"], np.float64)
                si = int(r["step_in_chunk"])
                if 0 <= si < a.shape[0]:
                    cmd = a[si]
            rows.append({"kind": "control", "t_step": int(r["t_step"]),
                         "chunk_seq": int(r["chunk_seq"]),
                         "step_in_chunk": int(r["step_in_chunk"]),
                         "qpos": np.asarray(r["qpos"], np.float64),
                         "cmd": cmd})
        elif r["kind"] == "planning":
            rows.append({"kind": "planning", "t_step": int(r["t_step"]),
                         "chunk_seq": int(r["seq"]), "step_in_chunk": -1,
                         "qpos": np.asarray(r["qpos"], np.float64),
                         "cmd": np.asarray(r["actions"], np.float64)[0]})
    rows.sort(key=lambda x: (x["t_step"], 0 if x["kind"] == "planning" else 1))
    N = len(rows)
    print(f"{N} state rows ({sum(1 for r in rows if r['kind']=='control')} control, "
          f"{sum(1 for r in rows if r['kind']=='planning')} planning)", flush=True)

    bodies_tracked = ("apple", "crate", "orange", "pear", "banana")
    clear = np.full((N, S, len(bodies)), np.nan, np.float32)
    centres_all = np.full((N, S, 3), np.nan, np.float32)
    obj_pos = np.full((N, len(bodies_tracked), 3), np.nan, np.float64)
    ee_pos = np.full((N, 4, 3), np.nan, np.float64)     # EE_BODY_L/R + fingertip bodies
    grip_cmd = np.full((N, 2), np.nan, np.float64)
    t_steps = np.asarray([r["t_step"] for r in rows], np.int64)
    kinds = np.asarray([r["kind"] for r in rows])
    chunk_seq = np.asarray([r["chunk_seq"] for r in rows], np.int64)
    step_in_chunk = np.asarray([r["step_in_chunk"] for r in rows], np.int64)

    # MJCF 의 손 body 이름은 URDF link 이름과 다르다 (ee_left -> EE_BODY_L).
    EE_BODIES = ("EE_BODY_L", "EE_BODY_R", "ee_finger_l1", "ee_finger_l2")
    ee_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in EE_BODIES]
    assert all(i >= 0 for i in ee_ids), dict(zip(EE_BODIES, ee_ids))
    obj_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in bodies_tracked]
    def body_top_z(name):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        return max(float(d.geom_xpos[g][2]) + float(m.geom_size[g][2])
                   for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b)

    d.qpos[: rows[0]["qpos"].size] = rows[0]["qpos"]
    mujoco.mj_forward(m, d)
    crate_top_z, table_top_z = body_top_z("crate"), body_top_z("table")
    print(f"   crate rim top z = {crate_top_z*1000:.2f} mm · table top z = "
          f"{table_top_z*1000:.2f} mm", flush=True)

    t0 = time.time()
    for i, r in enumerate(rows):
        q = r["qpos"]
        d.qpos[: q.size] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        qj = np.asarray([q[QADR[n]] for n in JOINT_ORDER], np.float64)
        c = np.asarray(robot.sphere_centers_numeric(qj)[0], np.float64).reshape(-1, 3)
        centres_all[i] = c
        dist = np.empty((G, S), np.float64)
        B = 1200
        for a in range(0, S, B):
            dist[:, a : a + B] = surface_distances(m, d, scene_geoms, c[a : a + B], mesh_cache)
        cl = (dist.T - radii[:, None])                 # (S, G)
        for bi in range(len(bodies)):
            clear[i, :, bi] = cl[:, body_of_geom == bi].min(axis=1)
        for k, oid in enumerate(obj_ids):
            obj_pos[i, k] = d.xpos[oid]
        for k, eid in enumerate(ee_ids):
            ee_pos[i, k] = d.xpos[eid]
        if r["cmd"] is not None:
            grip_cmd[i] = [r["cmd"][GRIP_COL["left"]], r["cmd"][GRIP_COL["right"]]]
        if i % 60 == 0 or i == N - 1:
            ap = bodies.index("apple")
            fin = [j for j, n in enumerate(link_names) if n in ("ee_finger_l1", "ee_finger_l2")]
            print(f"  row {i+1}/{N} t={r['t_step']:4d} {r['kind']:9s} "
                  f"finger-apple {clear[i, fin, ap].min()*1000:8.2f} mm  "
                  f"apple z {obj_pos[i,0,2]*1000:8.2f} mm  "
                  f"EE_BODY_L z {ee_pos[i,0,2]*1000:8.2f} mm  [{time.time()-t0:5.1f} s]",
                  flush=True)

    np.savez_compressed(
        NPZ, clearance_m=clear, centres_m=centres_all.astype(np.float32),
        bodies=np.asarray(bodies), link_names=np.asarray(link_names), radii_m=radii,
        t_step=t_steps, kind=kinds, chunk_seq=chunk_seq, step_in_chunk=step_in_chunk,
        obj_pos=obj_pos, obj_names=np.asarray(bodies_tracked), ee_pos=ee_pos,
        ee_names=np.asarray(EE_BODIES), grip_cmd=grip_cmd,
        crate_top_z_m=np.float64(crate_top_z), table_top_z_m=np.float64(table_top_z))
    META.write_text(json.dumps({
        "record": REC, "serve_log": SERVE_LOG,
        "sphere_options": {k: (v if not isinstance(v, dict) else v)
                           for k, v in SPHERE_OPTS.items()},
        "n_spheres": int(S), "n_capsules": len(robot.capsules),
        "server_log_says_spheres_capsules": server_spheres,
        "matches_server": bool(server_spheres == (S, len(robot.capsules))),
        "n_spheres_by_link": n_by_link, "radius_mm_by_link": per_link_r,
        "scene_bodies": bodies, "n_scene_geoms": G,
        "n_state_rows": N,
        "capsule_trims": trims,
        "tracked_object_bodies": list(bodies_tracked),
        "gripper_columns": GRIP_COL,
        "ee_bodies_tracked": list(EE_BODIES),
        "ee_body_name_note": "MJCF 의 손 body 이름은 URDF link 이름과 다르다 — "
                             "제약 모델의 ee_left/ee_right 는 MJCF 에서 EE_BODY_L/EE_BODY_R 다",
        "method": "recorded qpos (includes object free joints) -> mj_forward -> constraint sphere "
                  "centres -> analytic point-to-surface distance to MuJoCo collision geoms "
                  "(no mj_geomDistance, no distmax); same as outputs/verify/T8a/"
                  "m0_clearance_tensor.py",
        "clearance_definition": "surface_distance(sphere centre, nearest collidable geom of body) "
                                "- sphere_radius",
        "mesh_note": "box geoms (table, crate) exact; mesh geoms (apple/banana/orange/pear) use "
                     "vertex-minimum = upper bound on distance",
        "crate_rim_top_z_mm": round(crate_top_z * 1000.0, 4),
        "table_top_z_mm": round(table_top_z * 1000.0, 4),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=1, ensure_ascii=False))
    print("wrote", NPZ, META, flush=True)


if __name__ == "__main__":
    main()
