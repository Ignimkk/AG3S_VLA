"""T9a M1..M5 — m0 tensor 와 기록에서 표를 뽑는다. MuJoCo 를 다시 돌리지 않는다.

M3 사과가 움직였나 · M1 손끝이 얼마나 가까워졌나 · M2 gripper 가 닫힌 순간 ·
M4 바구니에서 하강이 멈춘 자리 · M5 max_violation_pair 분포.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import collections
import json
import pathlib

import numpy as np

T9A = pathlib.Path(_ag3s_workspace_path('outputs/verify/T9a'))
REC = _ag3s_workspace_path('outputs/live_test/20260926_t8/execute_ep1807/frames.jsonl')
OUT = T9A / "m1to5_analysis.json"
FIGNPZ = T9A / "m1to5_curves.npz"
GRIP_CLOSE = 0.85
REF = {"closed_loop_T7a_mm": 89.8562, "shadow_T7a_mm": -17.9552}


def main() -> None:
    z = np.load(T9A / "m0_state_tensor.npz", allow_pickle=True)
    clear = z["clearance_m"].astype(np.float64) * 1000.0          # (N, S, B) mm
    centres = z["centres_m"].astype(np.float64)                   # (N, S, 3) m
    bodies = [str(b) for b in z["bodies"]]
    link_names = [str(n) for n in z["link_names"]]
    radii_mm = np.asarray(z["radii_m"], np.float64) * 1000.0
    t = np.asarray(z["t_step"], np.int64)
    kind = np.asarray([str(k) for k in z["kind"]])
    obj = np.asarray(z["obj_pos"], np.float64)
    obj_names = [str(x) for x in z["obj_names"]]
    ee = np.asarray(z["ee_pos"], np.float64)
    ee_names = [str(x) for x in z["ee_names"]]
    grip = np.asarray(z["grip_cmd"], np.float64)
    chunk_seq = np.asarray(z["chunk_seq"], np.int64)
    step_in_chunk = np.asarray(z["step_in_chunk"], np.int64)
    N = clear.shape[0]
    ai = bodies.index("apple")
    ci = bodies.index("crate")
    ti = bodies.index("table")
    fin_l = np.asarray([n in ("ee_finger_l1", "ee_finger_l2") for n in link_names], bool)
    fin_r = np.asarray([n in ("ee_finger_r1", "ee_finger_r2") for n in link_names], bool)
    palm_l = np.asarray([n == "ee_left" for n in link_names], bool)
    apple_i = obj_names.index("apple")
    crate_i = obj_names.index("crate")
    eeL = ee_names.index("EE_BODY_L")

    out: dict = {"record": REC, "n_state_rows": int(N),
                 "t_step_range": [int(t.min()), int(t.max())],
                 "reference_values_mm": REF}

    # ---------------- M3 — 사과가 움직였나 (참값) --------------------------------------
    ap = obj[:, apple_i, :]
    ap0 = ap[0]
    disp = (ap - ap0) * 1000.0
    out["m3_apple_ground_truth"] = {
        "source": "recorded qpos free joint of body 'apple' (identical to object_poses)",
        "start_pos_m": [round(float(v), 6) for v in ap0],
        "max_lift_mm": round(float(disp[:, 2].max()), 4),
        "min_z_mm_rel_start": round(float(disp[:, 2].min()), 4),
        "max_horizontal_displacement_mm": round(float(np.linalg.norm(disp[:, :2], axis=1).max()), 4),
        "max_total_displacement_mm": round(float(np.linalg.norm(disp, axis=1).max()), 4),
        "argmax_total_t_step": int(t[int(np.argmax(np.linalg.norm(disp, axis=1)))]),
        "n_rows_with_displacement_gt_1mm": int((np.linalg.norm(disp, axis=1) > 1.0).sum()),
        "z_mm_series_first_last": [round(float(ap[0, 2] * 1000), 4),
                                   round(float(ap[-1, 2] * 1000), 4)],
        "other_objects_max_total_displacement_mm": {
            n: round(float(np.linalg.norm((obj[:, k, :] - obj[0, k, :]) * 1000.0,
                                          axis=1).max()), 4)
            for k, n in enumerate(obj_names)},
    }

    # ---------------- M1 — 손끝–사과 거리 ------------------------------------------------
    d_fin_l = clear[:, fin_l, ai].min(axis=1)
    d_fin_r = clear[:, fin_r, ai].min(axis=1)
    d_palm_l = clear[:, palm_l, ai].min(axis=1)
    imin = int(np.argmin(d_fin_l))
    plan = kind == "planning"
    out["m1_finger_to_apple"] = {
        "metric": "min over (left fingertip sphere) of surface_distance(centre, apple) - radius",
        "left_min_mm": round(float(d_fin_l.min()), 4),
        "left_min_t_step": int(t[imin]),
        "left_min_kind": str(kind[imin]),
        "left_min_chunk_seq": int(chunk_seq[imin]),
        "left_min_over_planning_rows_mm": round(float(d_fin_l[plan].min()), 4),
        "left_min_over_planning_t_step": int(t[plan][int(np.argmin(d_fin_l[plan]))]),
        "right_min_mm": round(float(d_fin_r.min()), 4),
        "left_palm_min_mm": round(float(d_palm_l.min()), 4),
        "left_palm_min_t_step": int(t[int(np.argmin(d_palm_l))]),
        "n_rows_below_0mm": int((d_fin_l <= 0.0).sum()),
        "n_rows_below_10mm": int((d_fin_l <= 10.0).sum()),
        "n_rows_below_20mm": int((d_fin_l <= 20.0).sum()),
        "comparison_mm": {
            "this_run_left_min": round(float(d_fin_l.min()), 4),
            "prev_closed_loop_T7a": REF["closed_loop_T7a_mm"],
            "successful_shadow_T7a": REF["shadow_T7a_mm"],
            "this_minus_prev_closed_loop": round(float(d_fin_l.min())
                                                 - REF["closed_loop_T7a_mm"], 4),
            "this_minus_shadow": round(float(d_fin_l.min()) - REF["shadow_T7a_mm"], 4),
            "fraction_of_the_gap_closed": round(
                (REF["closed_loop_T7a_mm"] - float(d_fin_l.min()))
                / (REF["closed_loop_T7a_mm"] - REF["shadow_T7a_mm"]), 6)},
        "note_radius_changed": "T7a 의 두 기준값은 손끝 구 반지름 13.9/15.4/18.7 mm 모델에서 나온 "
                               "값이고 이번 모델의 손끝 구는 3.6/3.97/4.72 mm 다. clearance 는 "
                               "surface_distance - r 이므로 같은 자세라도 이번 모델이 "
                               "약 14 mm 더 크게 나온다. 중심-표면 거리도 함께 낸다.",
        "left_min_centre_to_apple_surface_mm": round(
            float((clear[:, fin_l, ai] + radii_mm[fin_l][None, :]).min()), 4),
    }

    # ---------------- M2 — gripper 가 닫힌 순간 ------------------------------------------
    recs = [json.loads(l) for l in open(REC)]
    m2 = {}
    for side, col in (("left", 0), ("right", 1)):
        g = grip[:, col]
        ok = np.isfinite(g)
        below = ok & (g < GRIP_CLOSE)
        if not below.any():
            m2[side] = {"closed": False, "min_command": float(np.nanmin(g))}
            continue
        ctrl_below = below & (kind == "control")
        j = int(np.nonzero(ctrl_below)[0][0]) if ctrl_below.any() \
            else int(np.nonzero(below)[0][0])
        j_any = int(np.nonzero(below)[0][0])
        fin = fin_l if side == "left" else fin_r
        dj = float(clear[j, fin, ai].min())
        sj = int(np.nonzero(fin)[0][int(np.argmin(clear[j, fin, ai]))])
        offset = (centres[j, sj] - obj[j, apple_i]) * 1000.0
        # 손 전체 (손끝 + 손바닥) 중심의 사과 중심 기준 offset
        hand = fin | (palm_l if side == "left" else
                      np.asarray([n == "ee_right" for n in link_names], bool))
        hand_c = centres[j, hand].mean(axis=0)
        hand_off = (hand_c - obj[j, apple_i]) * 1000.0
        m2[side] = {
            "closed": True, "close_threshold": GRIP_CLOSE,
            "t_step": int(t[j]), "kind": str(kind[j]),
            "chunk_seq": int(chunk_seq[j]), "step_in_chunk": int(step_in_chunk[j]),
            "command_value": round(float(g[j]), 6),
            "command_at_previous_t_step": (lambda k: None if k is None
                                            else round(float(g[k]), 6))(
                next((k for k in range(j - 1, -1, -1)
                      if t[k] < t[j] and np.isfinite(g[k])), None)),
            "previous_t_step": (lambda k: None if k is None else int(t[k]))(
                next((k for k in range(j - 1, -1, -1)
                      if t[k] < t[j] and np.isfinite(g[k])), None)),
            "finger_to_apple_surface_mm": round(dj, 4),
            "finger_centre_to_apple_surface_mm": round(dj + float(radii_mm[sj]), 4),
            "closest_sphere_index": sj, "closest_sphere_link": link_names[sj],
            "closest_sphere_radius_mm": round(float(radii_mm[sj]), 4),
            "offset_closest_sphere_minus_apple_centre_mm": {
                "x": round(float(offset[0]), 4), "y": round(float(offset[1]), 4),
                "z": round(float(offset[2]), 4),
                "norm": round(float(np.linalg.norm(offset)), 4)},
            "offset_hand_centroid_minus_apple_centre_mm": {
                "x": round(float(hand_off[0]), 4), "y": round(float(hand_off[1]), 4),
                "z": round(float(hand_off[2]), 4),
                "norm": round(float(np.linalg.norm(hand_off)), 4)},
            "apple_centre_m": [round(float(v), 6) for v in obj[j, apple_i]],
            "first_row_any_kind": {"t_step": int(t[j_any]), "kind": str(kind[j_any]),
                                   "command_value": round(float(g[j_any]), 6),
                                   "finger_to_apple_surface_mm": round(
                                       float(clear[j_any, fin, ai].min()), 4)},
            "n_rows_command_below_threshold": int(below.sum()),
            "min_command": round(float(np.nanmin(g)), 6),
            "finger_to_apple_min_while_closed_mm": round(
                float(clear[below][:, fin, ai].min()), 4),
        }
    out["m2_gripper_close"] = m2

    # ---------------- M4 — 바구니에서 하강이 멈춘 자리 ------------------------------------
    hand_z = ee[:, eeL, 2] * 1000.0
    crate_pos = obj[:, crate_i, :]
    over_crate = (np.abs(ee[:, eeL, 0] - crate_pos[:, 0]) < 0.11) & \
                 (np.abs(ee[:, eeL, 1] - crate_pos[:, 1]) < 0.11)
    first_over = int(np.nonzero(over_crate)[0][0]) if over_crate.any() else None
    plan_rows = np.nonzero(plan)[0]
    P = [r for r in recs if r["kind"] == "planning"]
    seq_of_plan = {int(r["seq"]): r for r in P}
    tail = []
    for idx in plan_rows:
        if first_over is not None and idx < first_over:
            continue
        r = seq_of_plan[int(chunk_seq[idx])]
        mvp = r.get("max_violation_pair") or {}
        tail.append({
            "t_step": int(t[idx]), "chunk_seq": int(chunk_seq[idx]),
            "hand_z_mm": round(float(hand_z[idx]), 4),
            "hand_above_crate_rim_mm": round(float(hand_z[idx] - (crate_pos[idx, 2] * 1000.0)), 4),
            "min_clearance_to_crate_mm": round(float(clear[idx, :, ci].min()), 4),
            "min_clearance_to_crate_link": link_names[int(np.argmin(clear[idx, :, ci]))],
            "min_clearance_to_table_mm": round(float(clear[idx, :, ti].min()), 4),
            "min_clearance_to_table_link": link_names[int(np.argmin(clear[idx, :, ti]))],
            "min_clearance_any_body_mm": round(float(clear[idx].min()), 4),
            "min_clearance_any_link": link_names[int(np.argmin(clear[idx].min(axis=1)))],
            "min_clearance_any_body": bodies[int(np.argmin(clear[idx].min(axis=0)))],
            "safe": r.get("safe"), "hold_reason": r.get("hold_reason", ""),
            "ag3s_reason_codes": r.get("ag3s_reason_codes", []),
            "trajopt_status": r["verdict"].get("trajopt_status"),
            "max_violation_m": r["verdict"].get("max_violation_m"),
            "mvp_link": mvp.get("link"), "mvp_obstacle": mvp.get("obstacle"),
            "mvp_block": mvp.get("block"), "mvp_step": mvp.get("step"),
            "mvp_clearance_mm": (None if mvp.get("clearance_m") is None
                                 else round(float(mvp["clearance_m"]) * 1000.0, 4)),
        })
    crate_top_from_geom = float(z["crate_top_z_m"]) * 1000.0 if "crate_top_z_m" in z.files \
        else None
    out["m4_descent_over_crate"] = {
        "over_crate_rule": "|EE_BODY_L xy - crate xy| < 110 mm on both axes",
        "first_over_crate_t_step": None if first_over is None else int(t[first_over]),
        "n_rows_over_crate": int(over_crate.sum()),
        "hand_z_mm_min_while_over_crate": (None if not over_crate.any() else
                                           round(float(hand_z[over_crate].min()), 4)),
        "hand_z_mm_at_last_row": round(float(hand_z[-1]), 4),
        "crate_body_origin_z_mm": round(float(crate_pos[-1, 2] * 1000.0), 4),
        "crate_rim_top_z_mm": (None if crate_top_from_geom is None
                               else round(crate_top_from_geom, 4)),
        "hand_above_crate_rim_mm_min": (None if crate_top_from_geom is None or not over_crate.any()
                                        else round(float(hand_z[over_crate].min())
                                                   - crate_top_from_geom, 4)),
        "planning_rows_from_first_over_crate": tail,
        "destination_margin_hypothesis": {
            "question": "쥔 것이 없으면 destination_margin 이 적용되지 않는다 — 그래서 "
                        "하강이 막혔는가",
            "planning_record_keys": sorted(P[0].keys()),
            "attached_rows_in_max_violation_pair": {
                "n_chunks_with_attached_link": int(sum(
                    1 for r in P
                    if str((r.get("max_violation_pair") or {}).get("link") or "")
                    .startswith("attached:"))),
                "n_chunks_total": len(P),
                "attached_link_labels": sorted({
                    str((r.get("max_violation_pair") or {}).get("link"))
                    for r in P
                    if str((r.get("max_violation_pair") or {}).get("link") or "")
                    .startswith("attached:")}),
                "obstacle_values_seen": sorted({
                    str((r.get("max_violation_pair") or {}).get("obstacle"))
                    for r in P}),
                "attached_clearance_mm_min": (lambda v: round(min(v), 4) if v else None)([
                    float((r.get("max_violation_pair") or {}).get("clearance_m") or 0.0) * 1000.0
                    for r in P
                    if str((r.get("max_violation_pair") or {}).get("link") or "")
                    .startswith("attached:")]),
                "attached_clearance_mm_max": (lambda v: round(max(v), 4) if v else None)([
                    float((r.get("max_violation_pair") or {}).get("clearance_m") or 0.0) * 1000.0
                    for r in P
                    if str((r.get("max_violation_pair") or {}).get("link") or "")
                    .startswith("attached:")]),
                "note": "기록에 attached 전용 키는 없다. 이 숫자는 max_violation_pair.link 의 "
                        "'attached:' 접두사와 obstacle 값에서 센 것이다 — chunk 마다 한 행만 "
                        "기록되므로 attached 점의 전체 개수는 기록에서 셀 수 없다.",
            },
            "apple_max_lift_mm": out["m3_apple_ground_truth"]["max_lift_mm"],
            "all_chunks_safe": bool(all(r.get("safe") for r in P)),
            "max_violation_m_max": max(float(r["verdict"].get("max_violation_m") or 0.0)
                                       for r in P),
            "left_hand_min_clearance_to_crate_mm_over_descent": (
                None if first_over is None else round(float(
                    clear[first_over:][:, np.asarray(
                        [n in ("ee_left", "ee_finger_l1", "ee_finger_l2") for n in link_names],
                        bool), ci].min()), 4)),
            "right_hand_min_clearance_to_crate_mm_over_descent": (
                None if first_over is None else round(float(
                    clear[first_over:][:, np.asarray(
                        [n in ("ee_right", "ee_finger_r1", "ee_finger_r2") for n in link_names],
                        bool), ci].min()), 4)),
            "esdf_margin_mm": 10.0,
        },
    }

    # ---------------- M5 — max_violation_pair 분포 ---------------------------------------
    link_c = collections.Counter()
    obs_c = collections.Counter()
    block_c = collections.Counter()
    rows = []
    for r in P:
        mvp = r.get("max_violation_pair") or {}
        link_c[mvp.get("link")] += 1
        obs_c[str(mvp.get("obstacle"))] += 1
        block_c[mvp.get("block")] += 1
        rows.append({"t_step": int(r["t_step"]), "seq": int(r["seq"]),
                     "link": mvp.get("link"), "obstacle": mvp.get("obstacle"),
                     "block": mvp.get("block"), "step": mvp.get("step"),
                     "query": mvp.get("query"),
                     "clearance_mm": (None if mvp.get("clearance_m") is None
                                      else round(float(mvp["clearance_m"]) * 1000.0, 4)),
                     "safe": r.get("safe"),
                     "trajopt_status": r["verdict"].get("trajopt_status"),
                     "max_violation_m": r["verdict"].get("max_violation_m")})
    cl = [x["clearance_mm"] for x in rows if x["clearance_mm"] is not None]
    out["m5_max_violation_pair"] = {
        "note": "unsafe 0 이므로 이것은 위반이 아니라 각 chunk 에서 가장 빡빡한 행이다",
        "n_chunks": len(P),
        "link_counts": dict(link_c.most_common()),
        "obstacle_counts": dict(obs_c.most_common()),
        "block_counts": dict(block_c.most_common()),
        "clearance_mm_min": round(min(cl), 4), "clearance_mm_max": round(max(cl), 4),
        "clearance_mm_median": round(float(np.median(cl)), 4),
        "n_chunks_safe": sum(1 for r in P if r.get("safe")),
        "n_chunks_unsafe": sum(1 for r in P if not r.get("safe")),
        "trajopt_status_counts": dict(collections.Counter(
            r["verdict"].get("trajopt_status") for r in P)),
        "max_violation_m_counts": dict(collections.Counter(
            r["verdict"].get("max_violation_m") for r in P)),
        "ee_left_in_mvp": link_c.get("ee_left", 0),
        "ee_right_in_mvp": link_c.get("ee_right", 0),
        "recorded_clearance_is_post_margin": {
            "definition": "max_violation_pair.clearance_m = d_esdf(p) - r_robot - margin. "
                          "margin 은 이미 빠져 있다 (linearize.py:489 _esdf_clearance 의 "
                          "반환식과 docstring; :704 가 그 block 원소를 _identify 에 넘기고 "
                          ":718 이 clearance_m 으로 담는다).",
            "consequence": "clearance_m >= 0 이면 위반이 아니다. "
                           "max_violation = max(0, -clearance_m) 이다.",
            "esdf_margin_mm": 10.0,
            "n_chunks_clearance_below_0": int(sum(1 for x in rows
                                                  if x["clearance_mm"] is not None
                                                  and x["clearance_mm"] < 0.0)),
            "min_clearance_mm_post_margin": round(min(
                x["clearance_mm"] for x in rows if x["clearance_mm"] is not None), 4),
            "min_clearance_mm_pre_margin_equivalent": round(min(
                x["clearance_mm"] for x in rows if x["clearance_mm"] is not None) + 10.0, 4),
            "verdict_max_violation_m_all_zero": bool(all(
                float(r["verdict"].get("max_violation_m") or 0.0) == 0.0 for r in P)),
            "n_chunks_safe_true": sum(1 for r in P if r.get("safe")),
            "consistent": True,
            "corrected_misread": {
                "what_was_written_first": "39 chunk 중 5 개(t=72·80·88·96·184)에서 "
                                          "clearance_m 이 esdf margin 10 mm 보다 작은데 "
                                          "max_violation_m 은 0.0 이라 모순으로 보였다",
                "why_it_was_wrong": "clearance_m 은 이미 margin 을 뺀 값이라 10 mm 와 "
                                    "견줄 것이 아니다. 3.8144 mm 는 margin 을 빼고도 "
                                    "3.81 mm 남았다는 뜻이고 위반이 아니다.",
                "n_chunks_clearance_below_10mm_post_margin": int(sum(
                    1 for x in rows if x["clearance_mm"] is not None
                    and x["clearance_mm"] < 10.0)),
                "t_steps_of_those_chunks": [x["t_step"] for x in rows
                                            if x["clearance_mm"] is not None
                                            and x["clearance_mm"] < 10.0],
                "note": "이 숫자 자체는 맞다 — 뜻이 '위반에 가까웠다'가 아니라 "
                        "'margin 을 뺀 뒤 10 mm 미만이 남았다'일 뿐이다. 다음 사람이 같은 "
                        "비교를 하지 않도록 남긴다.",
            },
        },
        "offline_clearance_is_pre_margin": {
            "definition": "이 측정의 오프라인 clearance 는 surface_distance - sphere_radius 로 "
                          "margin 을 빼지 않은 값이다.",
            "to_compare": "기록의 clearance_m 에 margin 10 mm 를 더하면 오프라인 값과 같은 "
                          "정의가 된다 (둘은 거리장이 다르므로 값 자체는 일치하지 않는다).",
        },
        "per_chunk": rows,
        "offline_tightest_link_counts": dict(collections.Counter(
            link_names[int(np.argmin(clear[i].min(axis=1)))] for i in plan_rows).most_common()),
        "offline_tightest_body_counts": dict(collections.Counter(
            bodies[int(np.argmin(clear[i].min(axis=0)))] for i in plan_rows).most_common()),
    }

    # ---------------- figure 용 시계열 ----------------------------------------------------
    np.savez_compressed(
        FIGNPZ, t_step=t, kind=kind, finger_left_to_apple_mm=d_fin_l,
        finger_right_to_apple_mm=d_fin_r, palm_left_to_apple_mm=d_palm_l,
        apple_pos_m=ap, apple_disp_mm=disp, hand_z_mm=hand_z,
        crate_pos_m=crate_pos, grip_cmd=grip,
        min_clearance_crate_mm=clear[:, :, ci].min(axis=1),
        min_clearance_table_mm=clear[:, :, ti].min(axis=1),
        min_clearance_any_mm=clear.min(axis=(1, 2)),
        ee_pos_m=ee, ee_names=np.asarray(ee_names))
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))

    print("=== M3 apple ground truth ===")
    print(json.dumps(out["m3_apple_ground_truth"], indent=1, ensure_ascii=False))
    print("=== M1 finger to apple ===")
    print(json.dumps(out["m1_finger_to_apple"], indent=1, ensure_ascii=False))
    print("=== M2 gripper close ===")
    print(json.dumps(out["m2_gripper_close"], indent=1, ensure_ascii=False))
    print("=== M4 descent (head) ===")
    m4 = dict(out["m4_descent_over_crate"])
    print(json.dumps({k: v for k, v in m4.items()
                      if k != "planning_rows_from_first_over_crate"},
                     indent=1, ensure_ascii=False))
    for r in m4["planning_rows_from_first_over_crate"]:
        print(f"  t={r['t_step']:4d} seq={r['chunk_seq']:3d} hand_z={r['hand_z_mm']:8.2f} "
              f"crate {r['min_clearance_to_crate_mm']:8.2f} ({r['min_clearance_to_crate_link']}) "
              f"table {r['min_clearance_to_table_mm']:8.2f} "
              f"any {r['min_clearance_any_body_mm']:8.2f} ({r['min_clearance_any_link']}/"
              f"{r['min_clearance_any_body']}) mvp={r['mvp_link']}/{r['mvp_obstacle']} "
              f"{r['mvp_clearance_mm']}")
    print("=== M5 ===")
    print(json.dumps({k: v for k, v in out["m5_max_violation_pair"].items() if k != "per_chunk"},
                     indent=1, ensure_ascii=False))
    print("wrote", OUT, FIGNPZ)


if __name__ == "__main__":
    main()
