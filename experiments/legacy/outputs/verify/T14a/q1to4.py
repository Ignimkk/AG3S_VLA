"""T14a Q1..Q4 — target 을 잃는가 · fine 창이 사과를 담는가 · 공이 무엇을 지웠나 · 손끝은 덜 막혔나.

읽는 것 (쓰지 않는다):
  로컬 기록   outputs/live_test/2026092?_t{12,14}/execute_ep1807/frames.jsonl
  서버 ESDF   outputs/live_test/2026092?_t{12,14}/server_constraints/run_0000/chunk_*.npz
              (계층별 거리장 esdf_* · esdf_coarse_* · esdf_free_*, target_centroid,
               target_points, seed_indices, sphere_centres/radii, clearance, summary_json)

Q2 grounding 이 target 을 잃은 chunk (우선순위 1)
Q1 fine 창 origin 이 사과·손끝을 담는가, esdf_free 창이 주 fine 창과 같은가
Q3 공 안에서 esdf 와 esdf_free 가 실제로 다른 voxel 수 (T12 형식 그대로)
Q4 접근 구간 최빡빡 행과 손끝-사과 거리, T12 와 나란히
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import collections
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, _ag3s_workspace_path(''))

RUNS = {
    "T14": _ag3s_workspace_path('outputs/live_test/20260928_t14'),
    "T12": _ag3s_workspace_path('outputs/live_test/20260928_t12'),
}
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T14a'))
JSON_OUT = OUT / "q1to4.json"
NPZ_OUT = OUT / "q1to4_curves.npz"
APPROACH = (72, 160)          # lead 가 지정한 접근 구간
FINGER_LINKS = ("ee_finger_l1", "ee_finger_l2")
APPLE_RADIUS_MM = 45.0        # T12 표와 같은 반경


def load_chunks(root: pathlib.Path, planning: list):
    """서버 기록 chunk_*.npz 를 chunk_index 순으로, **로컬 기록에 이어 붙여서**.

    서버 summary 의 `t_step` 은 로컬 기록의 `t_step` 이 아니라 **chunk 일련번호(seq)** 다
    (chunk_00014 -> server t_step 15 -> local seq 15 -> local t_step 112). 매핑은
    `reference_chunk` 와 로컬의 `actions_reference` 를 수치로 맞춰 확인한다 — 이름이 아니라
    내용으로 잇는다.
    """
    files = sorted((root / "server_constraints" / "run_0000").glob("chunk_*.npz"))
    by_seq = {int(r["seq"]): r for r in planning}
    out = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        sm = json.loads(str(z["summary_json"]))
        seq = int(sm["t_step"])          # 서버의 t_step == 로컬 seq
        rec = by_seq.get(seq)
        # **이름이 아니라 내용으로 확인한다**: reference_chunk 에 가장 가까운 로컬 기록의 seq 가
        # 서버 seq 와 같은가. 절대 오차는 실행마다 다를 수 있으므로 (T12 에서 ~0.05 rad)
        # argmin 일치를 기준으로 삼고 잔차는 숫자로 남긴다.
        best_seq, best_res = None, None
        if "reference_chunk" in z.files:
            ref = np.asarray(z["reference_chunk"], np.float64)
            for r in planning:
                res = float(np.abs(np.asarray(r["actions_reference"], np.float64) - ref).max())
                if best_res is None or res < best_res:
                    best_seq, best_res = int(r["seq"]), res
        out.append({"path": f, "summary": sm, "npz": z, "server_seq": seq,
                    "local_record": rec, "local_t_step": (None if rec is None
                                                          else int(rec["t_step"])),
                    "best_matching_local_seq": best_seq,
                    "reference_chunk_residual_rad": (None if best_res is None
                                                     else round(best_res, 6)),
                    "reference_chunk_matches_local": (best_seq == seq)})
    return out


def _centroid_identity(c, objs):
    """target centroid 가 **어느 물체에 가장 가까운가**. status 가 ok 여도 다른 것을 잡을 수 있다."""
    z = c["npz"]
    if "target_centroid" not in z.files or not objs:
        return {"centroid_nearest_object": None, "centroid_to_apple_mm": None,
                "centroid_to_nearest_mm": None, "centroid_nearest_is_apple": None}
    cen = np.asarray(z["target_centroid"], np.float64) * 1000.0
    dd = {k: float(np.linalg.norm(cen - v)) for k, v in objs.items()}
    near = min(dd, key=dd.get)
    return {"centroid_nearest_object": near,
            "centroid_to_apple_mm": round(dd.get("apple", float("nan")), 2),
            "centroid_to_nearest_mm": round(dd[near], 2),
            "centroid_distance_by_object_mm": {k: round(v, 2) for k, v in dd.items()},
            "centroid_nearest_is_apple": bool(near == "apple")}


_MJ = {}


def apple_surface_distance(rec, points_m):
    """기록된 qpos 로 씬을 놓고 `points_m` 에서 **사과 표면**까지의 해석적 거리 (m).

    T9a/T10a 와 같은 방식이다 — `mj_geomDistance` 를 쓰지 않는다.
    """
    import mujoco
    if not _MJ:
        sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T6d'))
        from m3_to_refined_vs_reference import MODEL_XML, surface_distances
        m = mujoco.MjModel.from_xml_path(MODEL_XML)
        d = mujoco.MjData(m)
        bidx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "apple")
        geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bidx
                 and (int(m.geom_contype[g]) != 0 or int(m.geom_conaffinity[g]) != 0)]
        _MJ.update(m=m, d=d, geoms=geoms, fn=surface_distances, cache={})
    m, d = _MJ["m"], _MJ["d"]
    q = np.asarray(rec["qpos"], np.float64)
    d.qpos[: q.size] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    return _MJ["fn"](m, d, _MJ["geoms"], np.asarray(points_m, np.float64),
                     _MJ["cache"]).min(axis=0)


def get_layer(z, prefix):
    """`(lo_mm, hi_mm, voxel_mm, distance)` 또는 None. 계층이 없는 chunk 가 있다."""
    ok = f"{prefix}origin" in z.files
    if not ok:
        return None
    o = np.asarray(z[f"{prefix}origin"], np.float64) * 1000.0
    n = np.asarray(z[f"{prefix}shape"], np.int64)
    v = float(z[f"{prefix}voxel_size"]) * 1000.0
    dist = (np.asarray(z[f"{prefix}distance"]) if f"{prefix}distance" in z.files else None)
    return o, o + n * v, v, dist


def layer_window(origin, shape, voxel):
    o = np.asarray(origin, np.float64) * 1000.0
    n = np.asarray(shape, np.int64)
    v = float(voxel) * 1000.0
    return o, o + n * v, v


def contains(lo, hi, p):
    p = np.asarray(p, np.float64)
    return bool(np.all(p >= lo) and np.all(p < hi))


def main() -> None:
    t0 = time.time()
    result: dict = {"runs": {k: v for k, v in RUNS.items()},
                    "approach_window_t_step": list(APPROACH),
                    "method": {
                        "source": "서버 기록의 계층별 거리장과 summary_json 을 그대로 읽는다. "
                                  "MuJoCo 를 돌리지 않고 rollout 도 하지 않는다.",
                        "window_definition": "origin + shape * voxel_size, mm",
                        "apple_ball_radius_mm": APPLE_RADIUS_MM,
                        "note_sphere_model": "이 실행들의 제약 구 개수는 npz 의 sphere_centres 에서 "
                                             "읽는다 (T9/T10 의 266 과 다를 수 있다).",
                    }}

    per_run = {}
    for run, root in RUNS.items():
        root = pathlib.Path(root)
        P = [json.loads(l) for l in open(root / "execute_ep1807" / "frames.jsonl")
             if json.loads(l).get("kind") == "planning"]
        C = load_chunks(root, P)
        O = [json.loads(l) for l in open(root / "execute_ep1807" / "frames.jsonl")
             if json.loads(l).get("kind") == "observation"]
        obs_by_t = {int(o["t_step"]): o for o in O}
        plan_by_t = {int(r["t_step"]): r for r in P}
        apple_by_t = {int(r["t_step"]): np.asarray(r["object_poses"]["apple"]["pos"],
                                                   np.float64) * 1000.0 for r in P}
        objs_by_t = {int(r["t_step"]): {k: np.asarray(v["pos"], np.float64) * 1000.0
                                        for k, v in r["object_poses"].items()} for r in P}
        nm = sum(1 for c in C if c["reference_chunk_matches_local"])
        res = [c["reference_chunk_residual_rad"] for c in C
               if c["reference_chunk_residual_rad"] is not None]
        print(f"{run}: {len(C)} server chunks, {len(P)} planning, {len(O)} observation; "
              f"the nearest local record is the same seq on {nm}/{len(C)} "
              f"(residual max {max(res) if res else None} rad)", flush=True)
        per_run[run] = {"C": C, "P": P, "obs_by_t": obs_by_t, "plan_by_t": plan_by_t,
                        "apple_by_t": apple_by_t, "objs_by_t": objs_by_t}

    # ================= Q2 — grounding 이 target 을 잃은 chunk ==============================
    q2 = {}
    for run in RUNS:
        C = per_run[run]["C"]
        obs_by_t = per_run[run]["obs_by_t"]
        rows = []
        for c in C:
            s = c["summary"]
            t = c["local_t_step"]
            o = obs_by_t.get(t, {}) if t is not None else {}
            rows.append({
                "chunk_index": int(s["chunk_index"]), "t_step": t,
                "server_seq": c["server_seq"],
                "reference_chunk_matches_local": c["reference_chunk_matches_local"],
                "reference_chunk_residual_rad": c["reference_chunk_residual_rad"],
                "server_status": s.get("status"),
                "server_grounding_status": s.get("grounding_status"),
                "has_target": bool(s.get("has_target")),
                "target_n_points": int(s.get("target_n_points") or 0),
                "target_confidence": (None if s.get("target_confidence") is None
                                      else round(float(s["target_confidence"]), 6)),
                "grounding_best_score": (None if s.get("grounding_best_score") is None
                                         else round(float(s["grounding_best_score"]), 6)),
                "n_clusters": s.get("n_clusters"),
                "local_ag3s_status": o.get("ag3s_status"),
                "local_grounding_status": o.get("grounding_status"),
                "local_reason_codes": o.get("ag3s_reason_codes"),
                **_centroid_identity(c, per_run[run]["objs_by_t"].get(t)),
            })
        lost = [r for r in rows if not r["has_target"]]
        q2[run] = {
            "n_chunks": len(rows),
            "n_chunks_without_target": len(lost),
            "t_steps_without_target": [r["t_step"] for r in lost],
            "server_seqs_without_target": [r["server_seq"] for r in lost],
            "n_reference_chunk_matched": int(sum(
                1 for r in rows if r["reference_chunk_matches_local"])),
            "chunk_indices_without_target": [r["chunk_index"] for r in lost],
            "server_status_counts": dict(collections.Counter(
                r["server_status"] for r in rows).most_common()),
            "server_grounding_status_counts": dict(collections.Counter(
                r["server_grounding_status"] for r in rows).most_common()),
            "local_ag3s_status_counts": dict(collections.Counter(
                str(r["local_ag3s_status"]) for r in rows).most_common()),
            "local_grounding_status_counts": dict(collections.Counter(
                str(r["local_grounding_status"]) for r in rows).most_common()),
            "n_chunks_centroid_nearest_is_apple": int(sum(
                1 for r in rows if r["centroid_nearest_is_apple"] is True)),
            "n_chunks_centroid_nearest_is_not_apple": int(sum(
                1 for r in rows if r["centroid_nearest_is_apple"] is False)),
            "centroid_nearest_object_counts": dict(collections.Counter(
                str(r["centroid_nearest_object"]) for r in rows).most_common()),
            "t_steps_centroid_not_apple": [r["t_step"] for r in rows
                                           if r["centroid_nearest_is_apple"] is False],
            "centroid_to_apple_mm_max": max(
                (r["centroid_to_apple_mm"] for r in rows
                 if r["centroid_to_apple_mm"] is not None), default=None),
            "centroid_to_apple_mm_median": round(float(np.median([
                r["centroid_to_apple_mm"] for r in rows
                if r["centroid_to_apple_mm"] is not None])), 2),
            "n_centroid_not_apple_in_approach": int(sum(
                1 for r in rows if r["centroid_nearest_is_apple"] is False
                and r["t_step"] is not None
                and APPROACH[0] <= r["t_step"] <= APPROACH[1])),
            "target_n_points_min": min(r["target_n_points"] for r in rows),
            "target_n_points_median": int(np.median([r["target_n_points"] for r in rows])),
            "target_confidence_min": min(
                (r["target_confidence"] for r in rows if r["target_confidence"] is not None),
                default=None),
            "n_in_approach_window": int(sum(
                1 for r in lost if r["t_step"] is not None
                and APPROACH[0] <= r["t_step"] <= APPROACH[1])),
            "per_chunk": rows,
        }
    result["q2_grounding_target_loss"] = q2

    # ================= Q1 — fine 창이 사과를 담는가 ========================================
    q1 = {}
    for run in RUNS:
        C = per_run[run]["C"]
        apple_by_t = per_run[run]["apple_by_t"]
        rows = []
        for c in C:
            z, s = c["npz"], c["summary"]
            t = c["local_t_step"]
            lo, hi, v, _ = get_layer(z, "esdf_")
            L_free = get_layer(z, "esdf_free_")
            has_free = L_free is not None
            flo, fhi = (L_free[0], L_free[1]) if has_free else (None, None)
            L_coarse = get_layer(z, "esdf_coarse_")
            has_coarse = L_coarse is not None
            clo, chi, cv = (L_coarse[0], L_coarse[1], L_coarse[2]) if has_coarse \
                else (None, None, None)
            cen = (np.asarray(z["target_centroid"], np.float64) * 1000.0
                   if "target_centroid" in z.files else None)
            apple = apple_by_t.get(t)
            # 손끝: 제약 구 중 손끝 link 의 중심 (첫 실행 step)
            names = list(s.get("sphere_link_names") or [])
            cs = np.asarray(z["sphere_centres"], np.float64) * 1000.0
            fin = [i for i, n in enumerate(names) if n in FINGER_LINKS]
            tip = cs[fin].mean(axis=0) if fin else None
            rows.append({
                "chunk_index": int(s["chunk_index"]), "t_step": t,
                "has_target": bool(s.get("has_target")),
                "fine_origin_mm": [round(float(x), 2) for x in lo],
                "fine_max_mm": [round(float(x), 2) for x in hi],
                "fine_voxel_mm": round(v, 4),
                "fine_span_mm": [round(float(x), 2) for x in (hi - lo)],
                "has_free_layer": has_free,
                "free_origin_mm": None if flo is None else [round(float(x), 2) for x in flo],
                "free_equals_fine_window": (None if flo is None else
                                            bool(np.allclose(lo, flo)
                                                 and np.allclose(hi, fhi))),
                "has_coarse_layer": has_coarse,
                "coarse_origin_mm": None if clo is None else [round(float(x), 2) for x in clo],
                "coarse_voxel_mm": None if cv is None else round(cv, 4),
                "has_target_centroid": cen is not None,
                "target_centroid_mm": None if cen is None else [round(float(x), 2) for x in cen],
                "apple_mm": None if apple is None else [round(float(x), 2) for x in apple],
                "apple_in_fine_window": (None if apple is None else contains(lo, hi, apple)),
                "apple_in_coarse_window": (None if (apple is None or clo is None)
                                           else contains(clo, chi, apple)),
                "centroid_in_fine_window": (None if cen is None else contains(lo, hi, cen)),
                "centroid_minus_apple_mm": (None if (apple is None or cen is None)
                                            else [round(float(x), 2) for x in (cen - apple)]),
                "tip_mm": None if tip is None else [round(float(x), 2) for x in tip],
                "tip_in_fine_window": None if tip is None else contains(lo, hi, tip),
                "apple_distance_outside_fine_mm": (
                    None if apple is None else round(float(np.linalg.norm(
                        np.maximum(np.maximum(lo - apple, apple - hi), 0.0))), 2)),
            })
        app = [r for r in rows if r["t_step"] is not None
               and APPROACH[0] <= r["t_step"] <= APPROACH[1]]
        q1[run] = {
            "n_chunks": len(rows),
            "n_apple_outside_fine": int(sum(1 for r in rows
                                            if r["apple_in_fine_window"] is False)),
            "t_steps_apple_outside_fine": [r["t_step"] for r in rows
                                           if r["apple_in_fine_window"] is False],
            "n_tip_outside_fine": int(sum(1 for r in rows if r["tip_in_fine_window"] is False)),
            "t_steps_tip_outside_fine": [r["t_step"] for r in rows
                                         if r["tip_in_fine_window"] is False],
            "n_apple_outside_coarse": int(sum(1 for r in rows
                                              if r["apple_in_coarse_window"] is False)),
            "n_chunks_without_coarse_layer": int(sum(
                1 for r in rows if not r["has_coarse_layer"])),
            "n_chunks_without_free_layer": int(sum(1 for r in rows if not r["has_free_layer"])),
            "t_steps_without_free_layer": [r["t_step"] for r in rows if not r["has_free_layer"]],
            "n_free_window_differs_from_fine": int(sum(
                1 for r in rows if r["free_equals_fine_window"] is False)),
            "approach_window": {
                "n_chunks": len(app),
                "n_apple_outside_fine": int(sum(1 for r in app
                                                if r["apple_in_fine_window"] is False)),
                "t_steps_apple_outside_fine": [r["t_step"] for r in app
                                               if r["apple_in_fine_window"] is False],
                "n_tip_outside_fine": int(sum(1 for r in app
                                              if r["tip_in_fine_window"] is False)),
            },
            "n_chunks_without_target_centroid": int(sum(
                1 for r in rows if not r["has_target_centroid"])),
            "centroid_minus_apple_mm_stats": {
                ax: {"min": round(float(min(r["centroid_minus_apple_mm"][k] for r in rows
                                            if r["centroid_minus_apple_mm"])), 2),
                     "max": round(float(max(r["centroid_minus_apple_mm"][k] for r in rows
                                            if r["centroid_minus_apple_mm"])), 2),
                     "median": round(float(np.median([r["centroid_minus_apple_mm"][k]
                                                      for r in rows
                                                      if r["centroid_minus_apple_mm"]])), 2)}
                for k, ax in enumerate("xyz")},
            "per_chunk": rows,
        }
    result["q1_fine_window"] = q1

    # Q1 ∩ Q2 — "창이 벗어난 chunk" 와 "centroid 가 사과가 아닌 chunk" 가 같은 집합인가
    overlap = {}
    for run in RUNS:
        a = set(q1[run]["t_steps_apple_outside_fine"])
        b = set(q2[run]["t_steps_centroid_not_apple"])
        c = set(q2[run]["t_steps_without_target"])
        overlap[run] = {
            "t_steps_apple_outside_fine": sorted(a),
            "t_steps_centroid_not_apple": sorted(b),
            "t_steps_has_target_false": sorted(x for x in c if x is not None),
            "identical_sets": bool(a == b),
            "n_in_both": len(a & b), "n_only_window": len(a - b), "n_only_centroid": len(b - a),
            "only_window": sorted(a - b), "only_centroid": sorted(b - a),
        }
    result["q1_q2_overlap"] = overlap

    # ================= Q3 — 공이 무엇을 지웠나 ============================================
    q3 = {}
    for run in RUNS:
        C = per_run[run]["C"]
        apple_by_t = per_run[run]["apple_by_t"]
        rows = []
        for c in C:
            z, s = c["npz"], c["summary"]
            t = c["local_t_step"]
            tf = s.get("esdf_stats", {}).get("target_free")
            ball = None
            if isinstance(tf, str):
                try:
                    tf = json.loads(tf.replace("'", '"').replace("None", "null")
                                    .replace("True", "true").replace("False", "false"))
                except Exception:
                    tf = None
            if isinstance(tf, dict):
                ball = tf.get("ball")
            apple = apple_by_t.get(t)
            has_free = "esdf_free_distance" in z.files
            main = np.asarray(z["esdf_distance"], np.float32).astype(np.float64)
            free = (np.asarray(z["esdf_free_distance"], np.float32).astype(np.float64)
                    if has_free else None)
            lo, hi, v = layer_window(z["esdf_origin"], z["esdf_shape"], z["esdf_voxel_size"])
            n_diff = int((main != free).sum()) if has_free else None
            entry = {"chunk_index": int(s["chunk_index"]), "t_step": t,
                     "has_free_layer": has_free,
                     "n_voxels_differing_whole_fine_layer": n_diff,
                     "fine_voxel_mm": round(v, 4),
                     "ball": (None if ball is None else
                              {k: (round(float(ball[k]) * 1000.0, 3) if k.endswith("_m")
                                   and not isinstance(ball[k], list)
                                   else ([round(float(x) * 1000.0, 2) for x in ball[k]]
                                         if isinstance(ball[k], list) else ball[k]))
                               for k in ball}),
                     "n_ball_seeds_excluded": (None if not isinstance(tf, dict)
                                               else tf.get("n_seeds_excluded")),
                     }
            # 공 안에서의 비교
            for tag, centre_mm, rad_mm in (
                    ("ball", None if ball is None
                     else np.asarray(ball["centre_m"], np.float64) * 1000.0,
                     None if ball is None else float(ball["radius_mm"])),
                    ("apple45", apple, APPLE_RADIUS_MM)):
                if centre_mm is None:
                    entry[f"{tag}_n_voxels_in_sphere"] = None
                    entry[f"{tag}_n_voxels_changed"] = None
                    continue
                for layer, dm, df, olo, ovox in (("fine", main, free, lo, v),):
                    n = np.asarray(dm).shape
                    idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1)
                    pos = olo + (idx + 0.5) * ovox
                    inside = np.linalg.norm(pos - centre_mm, axis=-1) <= rad_mm
                    entry[f"{tag}_{layer}_n_voxels_in_sphere"] = int(inside.sum())
                    if df is None or not has_free:
                        entry[f"{tag}_{layer}_n_voxels_changed"] = None
                        entry[f"{tag}_{layer}_note"] = ("esdf_free 는 fine 계층 한 겹만 기록된다 "
                                                        "— coarse 는 비교 상대가 없다")
                    else:
                        ch = inside & (np.asarray(dm) != np.asarray(df))
                        entry[f"{tag}_{layer}_n_voxels_changed"] = int(ch.sum())
                        if ch.any():
                            entry[f"{tag}_{layer}_max_abs_change_mm"] = round(float(np.abs(
                                np.asarray(dm)[ch] - np.asarray(df)[ch]).max()) * 1000.0, 3)
                            entry[f"{tag}_{layer}_median_change_mm"] = round(float(np.median(
                                np.asarray(df)[ch] - np.asarray(dm)[ch])) * 1000.0, 3)
                    break     # coarse 는 free 상대가 없어 fine 만 비교한다
            rows.append(entry)
        q3[run] = {
            "n_chunks": len(rows),
            "n_chunks_without_free_layer": int(sum(1 for r in rows if not r["has_free_layer"])),
            "n_chunks_with_any_voxel_changed": int(sum(
                1 for r in rows if (r["n_voxels_differing_whole_fine_layer"] or 0) > 0)),
            "whole_layer_diff_min": min(
                (r["n_voxels_differing_whole_fine_layer"] for r in rows
                 if r["n_voxels_differing_whole_fine_layer"] is not None), default=None),
            "whole_layer_diff_max": max(
                (r["n_voxels_differing_whole_fine_layer"] for r in rows
                 if r["n_voxels_differing_whole_fine_layer"] is not None), default=None),
            "apple45_changed_total": int(sum(
                (r.get("apple45_fine_n_voxels_changed") or 0) for r in rows)),
            "apple45_changed_max": max(
                ((r.get("apple45_fine_n_voxels_changed") or 0) for r in rows), default=0),
            "ball_changed_total": int(sum(
                (r.get("ball_fine_n_voxels_changed") or 0) for r in rows)),
            "ball_changed_max": max(
                ((r.get("ball_fine_n_voxels_changed") or 0) for r in rows), default=0),
            "per_chunk": rows,
        }
        print(f"  Q3 {run}: apple45 changed total {q3[run]['apple45_changed_total']}, "
              f"ball changed total {q3[run]['ball_changed_total']} "
              f"[{time.time()-t0:5.1f} s]", flush=True)
    result["q3_what_the_ball_erased"] = q3

    # ================= Q4 — 손끝은 덜 막혔나 ==============================================
    q4 = {}
    for run in RUNS:
        C = per_run[run]["C"]
        P = per_run[run]["P"]
        apple_by_t = per_run[run]["apple_by_t"]
        rows = []
        for c in C:
            z, s = c["npz"], c["summary"]
            t = c["local_t_step"]
            mvp = s.get("to", {}).get("metrics", {}).get("max_violation_pair")
            if isinstance(mvp, str):
                try:
                    mvp = json.loads(mvp.replace("'", '"').replace("None", "null")
                                     .replace("True", "true").replace("False", "false"))
                except Exception:
                    mvp = None
            names = list(s.get("sphere_link_names") or [])
            cs = np.asarray(z["sphere_centres"], np.float64) * 1000.0
            rad = np.asarray(z["sphere_radii"], np.float64) * 1000.0
            apple = apple_by_t.get(t)
            fin = [i for i, n in enumerate(names) if n in FINGER_LINKS]
            tip_apple = None
            if apple is not None and fin and c["local_record"] is not None:
                # 사과 표면까지 **해석적** 거리 (T9a/T10a 와 같은 방식). 기록된 qpos 로 씬을
                # 놓고, 서버가 저장한 구 중심을 그대로 쓴다.
                tip_apple = float(apple_surface_distance(
                    c["local_record"], cs[fin] / 1000.0).min() * 1000.0 - rad[fin].min())
            rows.append({
                "chunk_index": int(s["chunk_index"]), "t_step": t,
                "clearance_min_mm_post_margin": (None if s.get("clearance_min_mm") is None
                                                 else round(float(s["clearance_min_mm"]), 4)),
                "clearance_min_per_step_mm": [round(float(v), 4) for v in
                                              (s.get("clearance_min_per_step_mm") or [])],
                "n_violating_rows": s.get("n_violating_rows"),
                "mvp_link": None if not mvp else mvp.get("link"),
                "mvp_clearance_mm_post_margin": (None if not mvp else
                                                 round(float(mvp["clearance_m"]) * 1000.0, 4)),
                "n_spheres": int(cs.shape[0]),
                "tip_to_apple_sphere_surface_mm": (None if tip_apple is None
                                                   else round(tip_apple, 4)),
                "has_target": bool(s.get("has_target")),
            })
        app = [r for r in rows if r["t_step"] is not None
               and APPROACH[0] <= r["t_step"] <= APPROACH[1]]
        cl = [r["mvp_clearance_mm_post_margin"] for r in app
              if r["mvp_clearance_mm_post_margin"] is not None]
        cl_alt = [r["clearance_min_mm_post_margin"] for r in app
                  if r["clearance_min_mm_post_margin"] is not None]
        # 사과 들림은 **control 기록까지** 본다 (qpos 의 free joint 가 참값이다)
        allrec = [json.loads(l) for l in open(
            pathlib.Path(RUNS[run]) / "execute_ep1807" / "frames.jsonl")]
        qp = [np.asarray(r["qpos"], np.float64) for r in allrec if "qpos" in r]
        AP_ADR = 38     # apple free joint qpos address (T9a 에서 확인)
        z0 = qp[0][AP_ADR + 2]
        lift = max(float(q[AP_ADR + 2] - z0) * 1000.0 for q in qp)
        disp = max(float(np.linalg.norm(q[AP_ADR:AP_ADR + 3] - qp[0][AP_ADR:AP_ADR + 3]))
                   * 1000.0 for q in qp)
        q4[run] = {
            "n_chunks": len(rows),
            "approach_window": {
                "n_chunks": len(app),
                "clearance_min_mm_post_margin": round(min(cl), 4) if cl else None,
                "clearance_min_t_step": (app[int(np.argmin(cl))]["t_step"] if cl else None),
                "clearance_median_mm_post_margin": round(float(np.median(cl)), 4) if cl else None,
                "clearance_max_mm_post_margin": round(max(cl), 4) if cl else None,
                "measure": "to.metrics.max_violation_pair.clearance_m (POST-margin, "
                           "linearize.py:489) — lead 가 인용한 것과 같은 양",
                "clearance_min_mm_summary_field": round(min(cl_alt), 4) if cl_alt else None,
                "clearance_min_mm_summary_field_note": "summary_json 의 clearance_min_mm. "
                    "mvp 와 다르면 두 필드의 정의가 다르다는 뜻이고 여기서 판정하지 않는다",
                "per_chunk": [{"t_step": r["t_step"],
                               "mvp_clearance_mm_post_margin":
                                   r["mvp_clearance_mm_post_margin"],
                               "clearance_min_mm_summary_field":
                                   r["clearance_min_mm_post_margin"],
                               "mvp_link": r["mvp_link"],
                               "tip_to_apple_mm": r["tip_to_apple_sphere_surface_mm"]}
                              for r in app],
            },
            "tip_to_apple_min_mm": round(min(
                (r["tip_to_apple_sphere_surface_mm"] for r in rows
                 if r["tip_to_apple_sphere_surface_mm"] is not None), default=float("nan")), 4),
            "apple_max_lift_mm": round(lift, 4),
            "apple_max_displacement_mm": round(disp, 4),
            "apple_lift_source": f"{len(qp)} records with qpos (planning + control), "
                                 "free joint at qpos[38:41]",
            "per_chunk": rows,
        }
    result["q4_fingertip"] = q4
    result["elapsed_s"] = round(time.time() - t0, 1)

    JSON_OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    np.savez_compressed(NPZ_OUT, **{
        f"{run}__t": np.asarray([r["t_step"] for r in q4[run]["per_chunk"]]) for run in RUNS})
    print("\n=== Q2 ===")
    for run in RUNS:
        e = q2[run]
        print(f"  {run}: has_target False on {e['n_chunks_without_target']}/{e['n_chunks']} "
              f"at t={e['t_steps_without_target']} (approach {e['n_in_approach_window']}) "
              f"grounding {e['server_grounding_status_counts']}")
        print(f"       centroid nearest object {e['centroid_nearest_object_counts']} · "
              f"not-apple on {e['n_chunks_centroid_nearest_is_not_apple']} chunks "
              f"({e['n_centroid_not_apple_in_approach']} in approach) at "
              f"t={e['t_steps_centroid_not_apple']}")
        print(f"       centroid-to-apple median {e['centroid_to_apple_mm_median']} mm, "
              f"max {e['centroid_to_apple_mm_max']} mm")
    print("=== Q1 ===")
    for run in RUNS:
        e = q1[run]
        print(f"  {run}: apple outside fine {e['n_apple_outside_fine']}/{e['n_chunks']} "
              f"at t={e['t_steps_apple_outside_fine']}")
        print(f"       tip outside fine {e['n_tip_outside_fine']}, "
              f"approach apple-outside {e['approach_window']['n_apple_outside_fine']}, "
              f"free window differs {e['n_free_window_differs_from_fine']}")
        print(f"       centroid - apple {e['centroid_minus_apple_mm_stats']}")
    print("=== Q1 n Q2 ===")
    for run in RUNS:
        e = overlap[run]
        print(f"  {run}: identical sets {e['identical_sets']} · both {e['n_in_both']} · "
              f"only window {e['only_window']} · only centroid {e['only_centroid']}")
    print("=== Q3 ===")
    for run in RUNS:
        e = q3[run]
        print(f"  {run}: whole fine layer diff {e['whole_layer_diff_min']}..{e['whole_layer_diff_max']} "
              f"voxels · apple45 changed total {e['apple45_changed_total']} "
              f"(max {e['apple45_changed_max']}) · ball changed total {e['ball_changed_total']} "
              f"(max {e['ball_changed_max']})")
    print("=== Q4 ===")
    for run in RUNS:
        e = q4[run]["approach_window"]
        print(f"  {run}: approach clearance min {e['clearance_min_mm_post_margin']} mm "
              f"at t={e['clearance_min_t_step']}, median {e['clearance_median_mm_post_margin']}, "
              f"max {e['clearance_max_mm_post_margin']} · tip-apple min "
              f"{q4[run]['tip_to_apple_min_mm']} mm · apple lift "
              f"{q4[run]['apple_max_lift_mm']} mm")
    print("wrote", JSON_OUT, f"[{result['elapsed_s']} s]")


if __name__ == "__main__":
    main()
