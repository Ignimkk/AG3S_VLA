"""T7a s1 — chunk 마다 **활성 제약 행 전체**를 link · joint 축으로.

방법 (T6d m3/m3c 와 같은 방법, 새로 촬영하지 않는다):
  기록의 `qpos` 로 MuJoCo 씬을 놓고 (물체 free joint 포함), chunk 의 action 행으로
  AG3S 제약 구 모델(arms 120 구, `s0_model_identity.json` 으로 T6d 와 동일 확인)의 구 중심을
  해석적으로 구한 뒤, 씬 충돌 geom **표면까지의 해석적 거리**를 잰다. `mj_geomDistance` 를
  쓰지 않으므로 `T5f` 의 `distmax` 포화 함정이 없다.

행 하나 = (step, sphere, scene geom):
  clearance      = surface_distance - sphere_radius          [TO 의 `d_esdf - r` 에 대응]
  required_margin= collision.esdf_margin = 0.05 m            [`config.py:383`, 서빙 기본값]
  margin_slack   = clearance - required_margin               [<=0 위반, ~0 활성]

`apple` 은 authorized target 이라 ESDF 에서 파내진다 — 그래서 `no_target`(apple geom 제외) 과
`all`(apple 포함) 둘 다 낸다.
QP 가 실제로 본 행은 step 마다 가장 조이는 `reduction.rows_per_step = 24` 행이다
(`config.py:235`) — 그 24 칸의 link 점유도 센다.
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

RUNS = {
    "shadow_approach": (_ag3s_workspace_path('outputs/live_test/local_shadow_from_pc/frames.jsonl'),
                        "reference"),
    "closed_loop_approach": (_ag3s_workspace_path('outputs/live_test/local_t6_from_pc/frames.jsonl'),
                             "refined"),
    "closed_loop_grasp": (_ag3s_workspace_path('outputs/live_test/local_grasp_from_pc/frames.jsonl'),
                          "refined"),
    "closed_loop_noforearm": (_ag3s_workspace_path('outputs/live_test/local_noforearm_from_pc/frames.jsonl'),
                              "refined"),
}
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T7a/s1_active_rows.json'))
NPZ = pathlib.Path(_ag3s_workspace_path('outputs/verify/T7a/s1_curves.npz'))
S0 = _ag3s_workspace_path('outputs/verify/T7a/s0_model_identity.json')
REQUIRED_MARGIN = 0.05
ROWS_PER_STEP = 24
EXEC_H = 8
TOPN = 8
BANDS_MM = (0.0, 5.0, 10.0)
ARM_JOINTS = tuple([f"left_arm_{i}" for i in range(7)] + [f"right_arm_{i}" for i in range(7)])


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

    chains = json.load(open(S0))["joint_chain_by_link"]
    arm_chain = {k: [j for j in (v or []) if j in ARM_JOINTS] for k, v in chains.items()}
    uniq_links = sorted(set(link_names))
    link_of_sphere = np.asarray([uniq_links.index(n) for n in link_names], np.int64)
    is_left_sphere = np.asarray(
        [("left" in n) or n in ("ee_finger_l1", "ee_finger_l2") for n in link_names], bool)

    def collidable(g):
        return int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0

    bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in SCENE_BODIES}
    scene_geoms = [g for g in range(m.ngeom)
                   if collidable(g) and int(m.geom_bodyid[g]) in set(bid.values())]
    geom_body = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
                 for g in scene_geoms]
    geom_type = [int(m.geom_type[g]) for g in scene_geoms]
    is_apple = np.asarray([b == "apple" for b in geom_body], bool)
    G = len(scene_geoms)
    print(f"{S} spheres · {G} scene geoms · bodies {sorted(set(geom_body))}")
    for g, b, t in zip(scene_geoms, geom_body, geom_type):
        nv = int(m.mesh_vertnum[int(m.geom_dataid[g])]) if t == mujoco.mjtGeom.mjGEOM_MESH else 0
        print(f"   geom {g:4d} {b:8s} type={t} verts={nv}")

    def q_from(row, qpos):
        return np.asarray([qpos[QADR[n]] if n.startswith("torso_") else float(row[ACT_COL[n]])
                           for n in JOINT_ORDER], np.float64)

    def rows_for_chunk(rec, chunk):
        """(H, G, S) margin_slack in metres."""
        qpos = np.asarray(rec["qpos"], np.float64)
        d.qpos[: qpos.size] = qpos
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        H = chunk.shape[0]
        centres = np.empty((H, S, 3), np.float64)
        for s in range(H):
            centres[s] = np.asarray(
                robot.sphere_centers_numeric(q_from(chunk[s], qpos))[0], np.float64).reshape(-1, 3)
        flat = centres.reshape(-1, 3)
        dist = np.empty((G, flat.shape[0]), np.float64)
        B = 1200
        for a in range(0, flat.shape[0], B):
            dist[:, a : a + B] = surface_distances(m, d, scene_geoms, flat[a : a + B], mesh_cache)
        clear = dist.reshape(G, H, S).transpose(1, 0, 2) - radii[None, None, :]
        return clear - REQUIRED_MARGIN, clear

    def summarise(slack, clear, sel, ssel):
        """slack/clear: (H, G, S).  sel: geom mask, ssel: sphere mask. 상위 N 행 + link/joint 집계."""
        sl = slack[:, sel, :][:, :, ssel]
        cl = clear[:, sel, :][:, :, ssel]
        gidx = np.nonzero(sel)[0]
        sidx = np.nonzero(ssel)[0]
        e = slice(0, EXEC_H)
        out = {}
        for win, name in ((e, "exec8"), (slice(0, sl.shape[0]), "full50")):
            w = sl[win]
            out[f"min_margin_slack_mm_{name}"] = float(w.min()) * 1000.0
            out[f"n_rows_below_0mm_{name}"] = int((w <= 0.0).sum())
            for b in BANDS_MM:
                out[f"n_rows_below_{b:g}mm_{name}"] = int((w <= b / 1000.0).sum())
        w = sl[e]
        wc = cl[e]
        # 상위 N 행 (가장 작은 margin_slack)
        order = np.argsort(w, axis=None)[:TOPN]
        top = []
        for f in order:
            st, gg, s_ = np.unravel_index(int(f), w.shape)
            ss = int(sidx[s_])
            ln = link_names[ss]
            top.append({
                "step": int(st), "sphere_index": int(ss), "link": ln,
                "arm_joints": arm_chain.get(ln, []),
                "obstacle_body": geom_body[int(gidx[gg])],
                "geom_id": int(scene_geoms[int(gidx[gg])]),
                "clearance_mm": round(float(wc[st, gg, s_]) * 1000.0, 4),
                "required_margin_mm": REQUIRED_MARGIN * 1000.0,
                "margin_slack_mm": round(float(w[st, gg, s_]) * 1000.0, 4),
                "sphere_radius_mm": round(float(radii[ss]) * 1000.0, 4),
            })
        out["top_rows_exec8"] = top
        # link 별 최소 margin_slack (exec8)
        per_sphere = w.min(axis=(0, 1))          # (len(sidx),)
        lo = link_of_sphere[sidx]
        per_link = {}
        for li, ln in enumerate(uniq_links):
            sel_s = lo == li
            if sel_s.any():
                per_link[ln] = round(float(per_sphere[sel_s].min()) * 1000.0, 4)
        out["per_link_min_margin_slack_mm_exec8"] = per_link
        # step 마다 가장 조이는 24 행 (QP 가 보는 예산) 의 link 점유
        flatw = w.reshape(w.shape[0], -1)
        cnt24: dict[str, int] = {}
        for st in range(flatw.shape[0]):
            k = min(ROWS_PER_STEP, flatw.shape[1])
            pick = np.argpartition(flatw[st], k - 1)[:k]
            for f in pick:
                ss = int(sidx[int(f) % w.shape[2]])
                cnt24[link_names[ss]] = cnt24.get(link_names[ss], 0) + 1
        out["tightest24_link_counts_exec8"] = dict(sorted(cnt24.items()))
        return out

    meta = {
        "method": "offline rebuild from recorded qpos + action chunk; analytic point-to-surface "
                  "distance to MuJoCo scene collision geoms (no mj_geomDistance, no distmax)",
        "model_xml": MODEL_XML,
        "required_margin_m": REQUIRED_MARGIN,
        "required_margin_source": "AG3S/trajopt/config.py:383 collision.esdf_margin default, "
                                  "matching serve_safe --esdf-margin default",
        "reduction_rows_per_step": ROWS_PER_STEP,
        "exec_window_steps": EXEC_H,
        "n_spheres": int(S),
        "n_scene_geoms": G,
        "scene_bodies": sorted(set(geom_body)),
        "sphere_model_identical_to_T6d": json.load(open(S0))["sphere_link_names_match_T6d_m1b"],
        "arm_joints": list(ARM_JOINTS),
        "joint_chain_by_link": arm_chain,
        "row_definition": "clearance = surface_distance - sphere_radius ; "
                          "margin_slack = clearance - required_margin",
        "qp_slack_rows_in_record": 0,
        "qp_slack_note": "wire 기록에는 QP slack 열이 없다 (planning 기록 키: max_violation_pair "
                         "하나뿐). slack != 0 행은 기록에서 셀 수 없다.",
    }
    out: dict = {"meta": meta, "runs": {}}
    npz: dict = {}

    for run, (path, executed) in RUNS.items():
        recs = load_planning(path)
        variants = {"refined": np.asarray([r["actions"] for r in recs], np.float64)}
        if all("actions_reference" in r for r in recs):
            variants["reference"] = np.asarray(
                [r["actions_reference"] for r in recs], np.float64)
        print(f"#### {run}: {len(recs)} chunks, variants {sorted(variants)}")
        chunks = []
        curves = {v: {"all": np.full((len(recs), EXEC_H, len(uniq_links)), np.nan),
                      "no_target": np.full((len(recs), EXEC_H, len(uniq_links)), np.nan)}
                  for v in variants}
        for i, rec in enumerate(recs):
            pair = rec.get("max_violation_pair") or {}
            row = {
                "chunk": i, "seq": int(rec["seq"]), "t_step": int(rec["t_step"]),
                "executed_chunk": rec.get("executed_chunk"),
                "recorded": {
                    "trajopt_status": rec["verdict"]["trajopt_status"],
                    "max_violation_m": rec["verdict"].get("max_violation_m"),
                    "safe": rec.get("safe"), "hold_reason": rec.get("hold_reason", ""),
                    "ipc": rec.get("ipc"),
                    "max_violation_pair_link": pair.get("link"),
                    "max_violation_pair_query": pair.get("query"),
                    "max_violation_pair_clearance_mm": (None if pair.get("clearance_m") is None
                                                        else float(pair["clearance_m"]) * 1000.0),
                    "max_violation_pair_obstacle": pair.get("obstacle"),
                    "max_violation_pair_candidate_id": pair.get("candidate_id"),
                    "max_violation_pair_block": pair.get("block"),
                    "reduction_bound_note": any("rows_per_step" in n
                                                for n in rec["verdict"].get("notes", [])),
                },
                "variants": {},
            }
            for v, arr in variants.items():
                slack, clear = rows_for_chunk(rec, arr[i])
                res = {}
                for tag, sel, ssel in (
                        ("all_geoms__all_spheres", np.ones(G, bool), np.ones(S, bool)),
                        ("all_geoms__left_spheres", np.ones(G, bool), is_left_sphere),
                        ("no_target__all_spheres", ~is_apple, np.ones(S, bool)),
                        ("no_target__left_spheres", ~is_apple, is_left_sphere)):
                    res[tag] = summarise(slack, clear, sel, ssel)
                for tag, sel in (("all", np.ones(G, bool)), ("no_target", ~is_apple)):
                    sub = slack[:EXEC_H][:, sel, :].min(axis=1)     # (H, S)
                    for li in range(len(uniq_links)):
                        curves[v][tag][i, :, li] = sub[:, link_of_sphere == li].min(axis=1) * 1000.0
                row["variants"][v] = res
            chunks.append(row)
            if i % 10 == 0:
                r = row["variants"][executed]["no_target__left_spheres"]
                print(f"  {run} chunk {i+1}/{len(recs)} "
                      f"min_slack(no_target,{executed},exec8)="
                      f"{r['min_margin_slack_mm_exec8']:8.3f} mm  "
                      f"top={r['top_rows_exec8'][0]['link']}/"
                      f"{r['top_rows_exec8'][0]['obstacle_body']}")
        out["runs"][run] = {"path": path, "executed_variant": executed,
                            "n_chunks": len(recs), "chunks": chunks}
        for v in variants:
            for tag in ("all", "no_target"):
                npz[f"{run}__{v}__{tag}"] = curves[v][tag]

    out["meta"]["link_axis"] = uniq_links
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    np.savez_compressed(NPZ, link_axis=np.asarray(uniq_links), **npz)
    print("wrote", OUT)
    print("wrote", NPZ)


if __name__ == "__main__":
    main()
