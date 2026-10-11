"""T6d 재는 것 2 (d) — 그 자리를 **로봇 자신이 언제 지나갔는가**.

`esdf.time_decay` 기본값은 `1.0`(감쇠 끔)이고 cuRobo mapper 는 프레임을 누적한다
(`curobo_builder.py:202` `decay_factor`, `:296` `self._frames += 1`). 그래서 필드에 한 번
점유로 들어온 voxel 은 그 뒤 프레임에서 자유로 덮이지 않으면 남는다.

여기서 재는 것: 각 frame 의 `max_violation_pair.point_m` 에서 **그 frame 까지의 모든 frame 의
로봇 표면**까지의 최소거리. 표면은 MJCF 충돌 mesh 정점 196,632 개(로봇 body 만, 씬 물체 제외)를
그 frame 의 `qpos` 로 world 에 놓은 것이다 — 정점 기반이라 볼록 표면까지의 거리의 **상한**이다.

출력은 (query point) x (frame) 행렬의 요약: 지금 frame 에서의 거리 · 과거 전체 최소 · 그 frame.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))

MODEL_XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
RUNS = {
    "closed_loop_approach": _ag3s_workspace_path('outputs/live_test/local_t6_from_pc/frames.jsonl'),
    "shadow_approach": _ag3s_workspace_path('outputs/live_test/local_shadow_from_pc/frames.jsonl'),
    "closed_loop_grasp": _ag3s_workspace_path('outputs/live_test/local_grasp_from_pc/frames.jsonl'),
}
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d/m2d_ghost_history.json'))
NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d/m2d_ghost_history.npz'))
SCENE_BODIES = ("apple", "banana", "orange", "pear", "crate", "table")


def main() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    scene_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_BODIES}

    # 로봇 충돌 mesh 정점을 geom local 로 한 번만 모은다.
    geoms: list[tuple[int, np.ndarray]] = []
    for g in range(m.ngeom):
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if int(m.geom_bodyid[g]) in scene_ids:
            continue
        if int(m.geom_type[g]) != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        geoms.append((g, np.asarray(m.mesh_vert[a : a + n], np.float64).reshape(-1, 3)))
    n_vert = sum(v.shape[0] for _, v in geoms)
    print(f"robot mesh geoms {len(geoms)} · verts {n_vert}")

    out: dict = {"model_xml": MODEL_XML, "n_robot_mesh_geoms": len(geoms),
                 "n_robot_vertices": n_vert,
                 "method": "min distance from point to robot collision-mesh vertices, "
                           "robot posed at each recorded frame's qpos (upper bound on surface distance)",
                 "runs": {}}
    npz: dict = {}

    for run, path in RUNS.items():
        recs = []
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("kind") != "planning":
                    continue
                pair = rec.get("max_violation_pair") or {}
                if pair.get("point_m") is None:
                    continue
                recs.append(rec)
        pts = np.asarray([r["max_violation_pair"]["point_m"] for r in recs], np.float64)
        qs = [np.asarray(r["qpos"], np.float64) for r in recs]
        F, P = len(qs), pts.shape[0]
        mat = np.full((P, F), np.inf)  # [query point, robot pose frame]

        for j, q in enumerate(qs):
            d.qpos[: q.size] = q
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            world = np.empty((n_vert, 3), np.float64)
            o = 0
            for g, v in geoms:
                R = np.asarray(d.geom_xmat[g], np.float64).reshape(3, 3)
                world[o : o + v.shape[0]] = v @ R.T + np.asarray(d.geom_xpos[g], np.float64)
                o += v.shape[0]
            # (P, n_vert) 를 한 번에 만들면 메모리가 커서 point 를 덩어리로 나눈다.
            for lo in range(0, P, 8):
                hi = min(lo + 8, P)
                dist = np.linalg.norm(world[None, :, :] - pts[lo:hi, None, :], axis=2)
                mat[lo:hi, j] = dist.min(axis=1)
            if j % 15 == 0:
                print(f"  {run} pose frame {j+1}/{F}")

        npz[f"{run}__matrix_mm"] = mat * 1000.0
        rows = []
        for i, rec in enumerate(recs):
            pair = rec["max_violation_pair"]
            hist = mat[i, : i + 1]
            jmin = int(np.argmin(hist))
            rows.append({
                "seq": int(rec["seq"]),
                "t_step": int(rec["t_step"]),
                "link": pair.get("link"),
                "clearance_mm": float(pair["clearance_m"]) * 1000.0,
                "hold_reason": rec.get("hold_reason", ""),
                "self_now_mm": float(mat[i, i]) * 1000.0,
                "self_history_min_mm": float(hist.min()) * 1000.0,
                "self_history_min_at_seq": int(recs[jmin]["seq"]),
                "self_history_min_at_t_step": int(recs[jmin]["t_step"]),
                "self_future_min_mm": float(mat[i, i:].min()) * 1000.0,
            })
        out["runs"][run] = rows
        print(f"#### {run}")
        for r in rows:
            if r["hold_reason"] or r["seq"] in (1, 75):
                print(f"  seq {r['seq']:3d} {r['link']:18s} clear={r['clearance_mm']:8.3f} "
                      f"self_now={r['self_now_mm']:8.2f} hist_min={r['self_history_min_mm']:8.2f} "
                      f"@seq{r['self_history_min_at_seq']}")

    np.savez_compressed(NPZ, **npz)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT, NPZ)


if __name__ == "__main__":
    main()
