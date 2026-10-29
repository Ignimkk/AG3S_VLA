"""T6d 재는 것 3 — TO 단계. `refined`(=`actions`) 대 `reference`(=`actions_reference`).

shadow 기록에만 둘이 함께 있다 (`local_shadow_from_pc`). 재는 것 셋:

1. chunk 별 **관절별 수정량** `refined − reference` — 어느 관절이 가장 많이 바뀌는가.
2. **MuJoCo 참값으로 두 궤적의 최소 clearance** — `refined` 가 `reference` 보다 나빠지는가.
   metric: AG3S 제약 구 모델(arms 120 구, T6d m1b 로 서버와 같은 것임을 확인)의 구 중심에서
   **참값 씬 표면**까지 해석적 거리 − 구 반지름. saturate 없다.
   씬 표면: apple · banana · orange · pear · crate · table 의 충돌 geom (box 는 정확,
   mesh 는 정점 최소거리 = 상한).
3. **손끝이 사과로 향하는 정도**: `ee_finger_l1`/`l2` 구 중심에서 사과 표면까지 최소거리.
   closed-loop 의 `actions`(refined) 와 shadow 의 `actions_reference`(원본) 를 같은 chunk 축에.

16D action 열 ↔ MuJoCo 관절 대응은 이 기록으로 실측했다 (`m3.log` 의 첫 블록):
  col 0..6  -> left_arm_0..6   · col 7  -> left gripper
  col 8..14 -> right_arm_0..6  · col 15 -> right gripper
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))

MODEL_XML = _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml')
SHADOW = _ag3s_workspace_path('outputs/live_test/local_shadow_from_pc/frames.jsonl')
CLOSED = _ag3s_workspace_path('outputs/live_test/local_t6_from_pc/frames.jsonl')
GRASP = _ag3s_workspace_path('outputs/live_test/local_grasp_from_pc/frames.jsonl')
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d/m3_to_refined_vs_reference.json'))
NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6d/m3_curves.npz'))
SCENE_BODIES = ("apple", "banana", "orange", "pear", "crate", "table")
JOINT_ORDER = tuple([f"torso_{i}" for i in range(6)]
                    + [f"right_arm_{i}" for i in range(7)]
                    + [f"left_arm_{i}" for i in range(7)])
QADR = {**{f"torso_{i}": 5 + i for i in range(6)},
        **{f"right_arm_{i}": 11 + i for i in range(7)},
        **{f"left_arm_{i}": 20 + i for i in range(7)}}
ACT_COL = {**{f"left_arm_{i}": i for i in range(7)},
           **{f"right_arm_{i}": 8 + i for i in range(7)}}
STEP_STRIDE = 1


def surface_distances(m, d, geoms, pts: np.ndarray, mesh_cache: dict) -> np.ndarray:
    """`(len(geoms), len(pts))` — 각 geom 표면까지의 부호 있는 거리 (밖이면 양수)."""
    import mujoco

    out = np.empty((len(geoms), pts.shape[0]), np.float64)
    for i, g in enumerate(geoms):
        t = int(m.geom_type[g])
        pos = np.asarray(d.geom_xpos[g], np.float64)
        R = np.asarray(d.geom_xmat[g], np.float64).reshape(3, 3)
        s = np.asarray(m.geom_size[g], np.float64)
        local = (pts - pos) @ R
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            q = np.abs(local) - s[:3][None, :]
            outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
            inside = np.minimum(q.max(axis=1), 0.0)
            out[i] = outside + inside
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            v = mesh_cache.get(mid)
            if v is None:
                a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                v = np.asarray(m.mesh_vert[a : a + n], np.float64).reshape(-1, 3)
                mesh_cache[mid] = v
            out[i] = np.linalg.norm(local[:, None, :] - v[None, :, :], axis=2).min(axis=1)
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            out[i] = np.linalg.norm(local, axis=1) - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            z = np.clip(local[:, 2], -s[1], s[1])
            out[i] = np.linalg.norm(local - np.stack([np.zeros_like(z), np.zeros_like(z), z], 1),
                                    axis=1) - s[0]
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            dr = np.hypot(local[:, 0], local[:, 1]) - s[0]
            dz = np.abs(local[:, 2]) - s[1]
            both_in = (dr <= 0) & (dz <= 0)
            out[i] = np.where(both_in, np.maximum(dr, dz),
                              np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0)))
        elif t == mujoco.mjtGeom.mjGEOM_PLANE:
            out[i] = local[:, 2]
        else:
            out[i] = np.inf
    return out


def load_planning(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "planning":
                rows.append(rec)
    return rows


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
    fingertip_idx = np.asarray([i for i, n in enumerate(link_names)
                                if n in ("ee_finger_l1", "ee_finger_l2")], np.int64)
    print(f"constraint model {robot.n_spheres} spheres · fingertip spheres {fingertip_idx.size}")

    def collidable(g):
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0

    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_BODIES}
    scene_geoms = [g for g in range(m.ngeom)
                   if collidable(g) and int(m.geom_bodyid[g]) in set(bid.values())]
    geom_body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
                 for g in scene_geoms]
    apple_geoms = [i for i, b in enumerate(geom_body) if b == "apple"]

    def q_from(action_row: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        q = np.empty(len(JOINT_ORDER), np.float64)
        for k, name in enumerate(JOINT_ORDER):
            q[k] = (qpos[QADR[name]] if name.startswith("torso_")
                    else float(action_row[ACT_COL[name]]))
        return q

    def evaluate(rec: dict, chunk: np.ndarray):
        """(min clearance over chunk, per-step clearance, per-step fingertip-to-apple)."""
        qpos = np.asarray(rec["qpos"], np.float64)
        d.qpos[: qpos.size] = qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)   # 씬 물체를 그 프레임의 참값 pose 로 (로봇은 아래서 무시)
        steps = range(0, chunk.shape[0], STEP_STRIDE)
        clear = np.full(chunk.shape[0], np.nan)
        tip = np.full(chunk.shape[0], np.nan)
        for s in steps:
            q = q_from(chunk[s], qpos)
            centres, _ = robot.sphere_centers_numeric(q)
            centres = np.asarray(centres, np.float64).reshape(-1, 3)
            dist = surface_distances(m, d, scene_geoms, centres, mesh_cache)  # (G, S)
            clear[s] = float((dist - radii[None, :]).min())
            tip[s] = float(dist[np.ix_(apple_geoms, fingertip_idx)].min()
                           - radii[fingertip_idx].max())
        return clear, tip

    out: dict = {
        "model_xml": MODEL_XML,
        "action_column_map": {"left_arm_0..6": "col 0..6", "left_gripper": "col 7",
                              "right_arm_0..6": "col 8..14", "right_gripper": "col 15"},
        "clearance_metric": "min over (constraint sphere, ground-truth scene geom) of "
                            "surface_distance(centre) - sphere_radius; mesh = vertex upper bound",
        "n_constraint_spheres": int(robot.n_spheres),
        "step_stride": STEP_STRIDE,
    }
    npz: dict = {}

    # ---------------- 1. 관절별 수정량 --------------------------------------------------
    sh = load_planning(SHADOW)
    ref = np.asarray([r["actions_reference"] for r in sh], np.float64)   # (F, H, 16)
    refd = np.asarray([r["actions"] for r in sh], np.float64)
    delta = refd - ref
    npz["shadow_delta"] = delta
    per_joint_max = np.abs(delta).max(axis=(0, 1))
    per_joint_rms = np.sqrt((delta ** 2).mean(axis=(0, 1)))
    col_name = {**{i: f"left_arm_{i}" for i in range(7)}, 7: "left_gripper",
                **{8 + i: f"right_arm_{i}" for i in range(7)}, 15: "right_gripper"}
    out["joint_delta"] = {
        "n_chunks": int(delta.shape[0]), "horizon": int(delta.shape[1]),
        "per_column": [{"col": i, "joint": col_name[i],
                        "max_abs_rad": float(per_joint_max[i]),
                        "max_abs_deg": float(np.degrees(per_joint_max[i])),
                        "rms_rad": float(per_joint_rms[i]),
                        "rms_deg": float(np.degrees(per_joint_rms[i]))} for i in range(16)],
        "largest_max_abs_joint": col_name[int(np.argmax(per_joint_max))],
        "largest_rms_joint": col_name[int(np.argmax(per_joint_rms))],
        "n_chunks_with_any_change_gt_1e_6": int((np.abs(delta).max(axis=(1, 2)) > 1e-6).sum()),
        "per_chunk_max_abs_deg": [float(np.degrees(v)) for v in np.abs(delta).max(axis=(1, 2))],
    }
    print("largest max|delta| joint:", out["joint_delta"]["largest_max_abs_joint"],
          f"{np.degrees(per_joint_max).max():.4f} deg")

    # ---------------- 2. 참값 clearance: refined vs reference ---------------------------
    rows = []
    C_ref = np.full((len(sh), ref.shape[1]), np.nan)
    C_new = np.full((len(sh), ref.shape[1]), np.nan)
    T_ref = np.full((len(sh), ref.shape[1]), np.nan)
    T_new = np.full((len(sh), ref.shape[1]), np.nan)
    for i, rec in enumerate(sh):
        c0, t0 = evaluate(rec, ref[i])
        c1, t1 = evaluate(rec, refd[i])
        C_ref[i], C_new[i], T_ref[i], T_new[i] = c0, c1, t0, t1
        rows.append({
            "seq": int(rec["seq"]), "t_step": int(rec["t_step"]),
            "hold_reason": rec.get("hold_reason", ""),
            "trajopt_status": rec["verdict"]["trajopt_status"],
            "reference_min_clearance_mm": float(np.nanmin(c0)) * 1000.0,
            "refined_min_clearance_mm": float(np.nanmin(c1)) * 1000.0,
            "delta_min_clearance_mm": (float(np.nanmin(c1)) - float(np.nanmin(c0))) * 1000.0,
            "reference_min_tip_to_apple_mm": float(np.nanmin(t0)) * 1000.0,
            "refined_min_tip_to_apple_mm": float(np.nanmin(t1)) * 1000.0,
            "delta_min_tip_to_apple_mm": (float(np.nanmin(t1)) - float(np.nanmin(t0))) * 1000.0,
            "max_abs_joint_delta_deg": float(np.degrees(np.abs(delta[i]).max())),
        })
        if i % 10 == 0:
            print(f"  chunk {i+1}/{len(sh)} ref={rows[-1]['reference_min_clearance_mm']:8.2f} "
                  f"refined={rows[-1]['refined_min_clearance_mm']:8.2f} mm  "
                  f"tip ref={rows[-1]['reference_min_tip_to_apple_mm']:7.2f} "
                  f"refined={rows[-1]['refined_min_tip_to_apple_mm']:7.2f}")
    npz.update(shadow_clear_reference_mm=C_ref * 1000.0, shadow_clear_refined_mm=C_new * 1000.0,
               shadow_tip_reference_mm=T_ref * 1000.0, shadow_tip_refined_mm=T_new * 1000.0)
    worse = [r for r in rows if r["delta_min_clearance_mm"] < -1e-6]
    out["shadow_chunks"] = rows
    out["clearance_summary"] = {
        "n_chunks": len(rows),
        "n_refined_worse_than_reference": len(worse),
        "n_refined_better": int(sum(1 for r in rows if r["delta_min_clearance_mm"] > 1e-6)),
        "n_equal": int(sum(1 for r in rows if abs(r["delta_min_clearance_mm"]) <= 1e-6)),
        "worst_degradation_mm": min([r["delta_min_clearance_mm"] for r in rows]),
        "best_improvement_mm": max([r["delta_min_clearance_mm"] for r in rows]),
        "chunks_where_refined_worse": [
            {"seq": r["seq"], "delta_mm": r["delta_min_clearance_mm"],
             "reference_mm": r["reference_min_clearance_mm"],
             "refined_mm": r["refined_min_clearance_mm"]} for r in worse],
    }
    tip_worse = [r for r in rows if r["delta_min_tip_to_apple_mm"] > 1e-6]
    out["tip_to_apple_summary"] = {
        "n_chunks_refined_further_from_apple": len(tip_worse),
        "n_chunks_refined_closer_to_apple": int(
            sum(1 for r in rows if r["delta_min_tip_to_apple_mm"] < -1e-6)),
        "max_push_away_mm": max([r["delta_min_tip_to_apple_mm"] for r in rows]),
        "max_pull_in_mm": min([r["delta_min_tip_to_apple_mm"] for r in rows]),
        "reference_best_tip_to_apple_mm": min([r["reference_min_tip_to_apple_mm"] for r in rows]),
        "refined_best_tip_to_apple_mm": min([r["refined_min_tip_to_apple_mm"] for r in rows]),
    }
    print("refined worse in", len(worse), "chunks · worst",
          out["clearance_summary"]["worst_degradation_mm"], "mm")

    # ---------------- 3. closed-loop refined 대 shadow reference ------------------------
    for label, path in (("closed_loop_approach", CLOSED), ("closed_loop_grasp", GRASP)):
        recs = load_planning(path)
        tips = np.full((len(recs), 50), np.nan)
        clears = np.full((len(recs), 50), np.nan)
        for i, rec in enumerate(recs):
            ch = np.asarray(rec["actions"], np.float64)
            c, t = evaluate(rec, ch)
            clears[i, : c.size] = c
            tips[i, : t.size] = t
        npz[f"{label}_tip_mm"] = tips * 1000.0
        npz[f"{label}_clear_mm"] = clears * 1000.0
        out[f"{label}_executed_refined"] = {
            "n_chunks": len(recs),
            "best_tip_to_apple_mm": float(np.nanmin(tips)) * 1000.0,
            "best_tip_chunk_seq": int(recs[int(np.nanargmin(np.nanmin(tips, axis=1)))]["seq"]),
            "min_clearance_mm": float(np.nanmin(clears)) * 1000.0,
            "per_chunk_min_tip_mm": [float(v) * 1000.0 for v in np.nanmin(tips, axis=1)],
        }
        print(label, "best tip->apple", out[f"{label}_executed_refined"]["best_tip_to_apple_mm"], "mm")

    out["shadow_reference_best_tip_per_chunk_mm"] = [r["reference_min_tip_to_apple_mm"] for r in rows]
    out["shadow_refined_best_tip_per_chunk_mm"] = [r["refined_min_tip_to_apple_mm"] for r in rows]

    np.savez_compressed(NPZ, **npz)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("wrote", OUT, NPZ)


if __name__ == "__main__":
    main()
