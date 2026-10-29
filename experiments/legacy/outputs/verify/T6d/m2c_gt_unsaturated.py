"""T6d 재는 것 2 (c) — saturate 없이 참값 최근접 표면까지.

`m2_gt_at_violation_points.py` 는 `mj_geomDistance` 를 쓰고 `distmax` 를 0.1 m 로 묶었다
(`T5f` 의 mesh 실패 함정). 그래서 100.5 mm 에서 전부 saturate 됐고 "무엇이 가장 가까운가" 를
못 골랐다. 여기서는 **해석적 점-표면 거리**로 다시 잰다 — `distmax` 가 없으므로 saturate 하지 않는다.

  · box / sphere / capsule / cylinder / ellipsoid / plane : 정확한 해석식
  · mesh : **정점까지의 최소거리** — 볼록 hull 까지의 거리보다 크거나 같은 상한이다
           (점이 물체 밖 100 mm 이상일 때 정점 간격(수 mm) 만큼만 헐겁다)

씬 물체는 기록의 `qpos` 안 free joint 로, 로봇은 같은 `qpos` 로 놓는다. 필드는 관측 시점의
스냅샷이므로 로봇 표면은 **그 프레임의 qpos** 가 맞다 — 질의점만 계획 horizon step 의 좌표다.
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
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d/m2c_gt_unsaturated.json'))
SCENE_BODIES = ("apple", "banana", "orange", "pear", "crate", "table")
# arm_5 구의 반지름 + 서버 --esdf-margin. m2b 실측: 81.25 mm / 50 mm.
R_ARM5_MM = 81.25
R_BY_QUERY_MM = {18: 81.25, 41: 81.25, 61: 29.21728072820923}
ESDF_MARGIN_MM = 50.0


def geom_surface_distance(m, d, g: int, p: np.ndarray, mesh_cache: dict) -> float:
    """점 `p`(world) 에서 geom `g` 표면까지. 밖이면 양수, 안이면 음수(근사)."""
    import mujoco

    t = int(m.geom_type[g])
    pos = np.asarray(d.geom_xpos[g], np.float64)
    R = np.asarray(d.geom_xmat[g], np.float64).reshape(3, 3)
    s = np.asarray(m.geom_size[g], np.float64)
    local = R.T @ (p - pos)

    if t == mujoco.mjtGeom.mjGEOM_PLANE:
        return float(local[2])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(np.linalg.norm(local) - s[0])
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        half = s[1]
        z = np.clip(local[2], -half, half)
        return float(np.linalg.norm(local - np.array([0.0, 0.0, z])) - s[0])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r_xy = float(np.hypot(local[0], local[1]))
        dr = r_xy - s[0]
        dz = abs(local[2]) - s[1]
        if dr <= 0.0 and dz <= 0.0:
            return float(max(dr, dz))
        return float(np.hypot(max(dr, 0.0), max(dz, 0.0)))
    if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        # 상한/하한이 아닌 근사: 정규화 좌표의 반지름으로 스케일. 여기서는 어차피 쓰이지 않는다.
        n = np.linalg.norm(local / s[:3])
        if n == 0.0:
            return float(-s[:3].min())
        return float((n - 1.0) * float(np.min(s[:3])))
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        q = np.abs(local) - s[:3]
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(float(np.max(q)), 0.0)
        return float(outside + inside)
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g])
        v = mesh_cache.get(mid)
        if v is None:
            a = int(m.mesh_vertadr[mid])
            n = int(m.mesh_vertnum[mid])
            v = np.asarray(m.mesh_vert[a : a + n], np.float64).reshape(-1, 3)
            mesh_cache[mid] = v
        return float(np.linalg.norm(v - local[None, :], axis=1).min())
    return float("inf")


def main() -> None:
    import mujoco

    m = mujoco.MjModel.from_xml_path(MODEL_XML)
    d = mujoco.MjData(m)
    mesh_cache: dict = {}

    def collidable(g: int) -> bool:
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0

    body_name = {b: (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}")
                 for b in range(m.nbody)}
    scene_ids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_BODIES}
    scene_geoms = [(g, body_name[int(m.geom_bodyid[g])]) for g in range(m.ngeom)
                   if collidable(g) and int(m.geom_bodyid[g]) in scene_ids]
    robot_geoms = [(g, body_name[int(m.geom_bodyid[g])]) for g in range(m.ngeom)
                   if collidable(g) and int(m.geom_bodyid[g]) not in scene_ids]
    print(f"scene {len(scene_geoms)} · robot {len(robot_geoms)} geoms")

    out: dict = {
        "model_xml": MODEL_XML,
        "method": "analytic point-to-surface; mesh = min distance to mesh vertices (upper bound)",
        "esdf_margin_mm": ESDF_MARGIN_MM,
        "sphere_radius_mm_by_query": R_BY_QUERY_MM,
        "runs": {},
    }

    for run, path in RUNS.items():
        rows = []
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("kind") != "planning":
                    continue
                pair = rec.get("max_violation_pair") or {}
                pt = pair.get("point_m")
                if pt is None:
                    continue
                qpos = np.asarray(rec["qpos"], np.float64)
                d.qpos[: qpos.size] = qpos
                d.qvel[:] = 0.0
                mujoco.mj_forward(m, d)
                p = np.asarray(pt, np.float64)

                def nearest(pairs):
                    best = (float("inf"), None)
                    per: dict[str, float] = {}
                    for g, label in pairs:
                        dd = geom_surface_distance(m, d, g, p, mesh_cache)
                        if dd < best[0]:
                            best = (dd, label)
                        if label not in per or dd < per[label]:
                            per[label] = dd
                    return best, per

                (s_min, s_lab), s_per = nearest(scene_geoms)
                (r_min, r_lab), r_per = nearest(robot_geoms)
                q = pair.get("query")
                rad = R_BY_QUERY_MM.get(int(q)) if q is not None else None
                implied = None
                if rad is not None:
                    implied = float(pair["clearance_m"]) * 1000.0 + rad + ESDF_MARGIN_MM
                rows.append({
                    "seq": int(rec["seq"]),
                    "t_step": int(rec["t_step"]),
                    "step_in_chunk": int(pair.get("step", -1)),
                    "link": pair.get("link"),
                    "query": q,
                    "sphere_radius_mm": rad,
                    "clearance_mm": float(pair["clearance_m"]) * 1000.0,
                    "implied_esdf_distance_mm": implied,
                    "required_free_radius_mm": None if rad is None else rad + ESDF_MARGIN_MM,
                    "hold_reason": rec.get("hold_reason", ""),
                    "trajopt_status": rec["verdict"]["trajopt_status"],
                    "point_m": [float(v) for v in p],
                    "gt_scene_min_mm": s_min * 1000.0,
                    "gt_scene_min_body": s_lab,
                    "gt_scene_per_body_mm": {k: round(v * 1000.0, 3) for k, v in sorted(s_per.items())},
                    "gt_robot_min_mm": r_min * 1000.0,
                    "gt_robot_min_body": r_lab,
                    "gt_robot_top5_mm": [
                        [k, round(v * 1000.0, 3)]
                        for k, v in sorted(r_per.items(), key=lambda kv: kv[1])[:5]
                    ],
                })
        out["runs"][run] = rows
        print(f"#### {run}")
        for r in rows:
            if r["hold_reason"] or r["seq"] in (1, 75):
                print(f"  seq {r['seq']:3d} t={r['t_step']:4d} step={r['step_in_chunk']:2d} "
                      f"{r['link']:18s} clear={r['clearance_mm']:8.3f} "
                      f"need={r['required_free_radius_mm']} d_esdf~{r['implied_esdf_distance_mm']:8.3f} | "
                      f"gt_scene={r['gt_scene_min_mm']:8.2f}({r['gt_scene_min_body']}) "
                      f"gt_robot={r['gt_robot_min_mm']:8.2f}({r['gt_robot_min_body']})")

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
