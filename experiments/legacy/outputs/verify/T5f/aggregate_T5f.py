"""T5f — 세 raw 파일을 하나로 접는다. verify.json 의 `numbers` 가 여기서 나온다.

읽는 것:
  · `outputs/verify/T5f/target_identity_ep1807.json`  (chunk 별 cluster·target·참값 이름)
  · `outputs/verify/T5f/gt_clearance_ep1807.json`     (참값 clearance, distmax 셋 교차 확인)
  · `outputs/verify/T5f/shadow_agg_two_runs.json`     (두 live 기록의 verdict·계층·카메라 폭)
  · `outputs/verify/T5f/baseline_check.json`          (회귀 기준선)
  · `AG3S/ag3s/docs/handoff/T5d.verify.json`     (T5d 의 rank-1 구간 — 대조용)

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5f/aggregate_T5f.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
from collections import Counter

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5f'))
OUT = V / "aggregate_T5f.json"
T5D = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T5d.verify.json'))

#: T5d 의 과제 국면 규칙 그대로.
GRIPPER_CLOSED_BELOW = 0.85
LIFT_MM = 20.0
OVER_CRATE_XY_MM = 100.0


def stats(xs, nd=4):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None

    def q(p):
        i = (len(xs) - 1) * p / 100.0
        lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
    return {"n": len(xs), "min": round(xs[0], nd), "p50": round(q(50), nd),
            "max": round(xs[-1], nd), "mean": round(sum(xs) / len(xs), nd)}


def segments(pairs):
    """[(seq, name)] -> 연속 구간 목록."""
    out = []
    for seq, name in pairs:
        if out and out[-1]["name"] == name and out[-1]["seq_to"] == seq - 1:
            out[-1]["seq_to"] = seq
            out[-1]["n"] += 1
        else:
            out.append({"name": name, "seq_from": seq, "seq_to": seq, "n": 1})
    return out


def main() -> None:
    ident = json.loads((V / "target_identity_ep1807.json").read_text())
    gtc = json.loads((V / "gt_clearance_ep1807.json").read_text())
    shadow = json.loads((V / "shadow_agg_two_runs.json").read_text())
    base = json.loads((V / "baseline_check.json").read_text())
    t5d = json.loads(T5D.read_text())["numbers"]

    fr = ident["frames"]
    gt_by_seq = {f["seq"]: f for f in gtc["frames"]}

    # --- 과제 국면 (T5d 규칙 재계산) ------------------------------------------------------
    apple_z0 = fr[0]["apple_z_m"]
    closed = [f["seq"] for f in fr if f["gripper_obs_left"] < GRIPPER_CLOSED_BELOW]
    first_closed = min(closed) if closed else None
    reopen = [f["seq"] for f in fr
              if first_closed is not None and f["seq"] > first_closed
              and f["gripper_obs_left"] >= GRIPPER_CLOSED_BELOW]
    first_reopen = min(reopen) if reopen else None
    phase = {}
    for f in fr:
        s, g = f["seq"], f["gripper_obs_left"]
        lift_mm = (f["apple_z_m"] - apple_z0) * 1000.0
        if first_reopen is not None and s >= first_reopen and g >= GRIPPER_CLOSED_BELOW:
            phase[s] = "retreat"
        elif g >= GRIPPER_CLOSED_BELOW:
            phase[s] = "approach"
        elif lift_mm <= LIFT_MM:
            phase[s] = "grasp"
        elif f["apple_crate_xy_dist_mm"] > OVER_CRATE_XY_MM:
            phase[s] = "transport"
        else:
            phase[s] = "place"

    # --- 세 갈래의 chunk 별 이름 -----------------------------------------------------------
    rank1 = [(f["seq"], (f["clusters"][0]["name"] if f["clusters"] else None)) for f in fr]
    before = [(f["seq"], f["before_threshold_0p25"]["name"]) for f in fr]
    after = [(f["seq"], f["after"]["name"]) for f in fr]

    rows = []
    for f in fr:
        s = f["seq"]
        a, b = f["after"], f["before_threshold_0p25"]
        ch = f["clusters"][a["chosen_rank"] - 1] if (a["has_target"] and f["clusters"]) else None
        g10 = gt_by_seq[s]["by_distmax"]["0.10"]
        mode = "keep_or_first"
        for n in f["notes_from_pipeline"]:
            if n.startswith("target hold"):
                mode = "hold"
            elif n.startswith("target switch"):
                mode = "switch"
            elif n.startswith("target unobserved"):
                mode = "unobserved"
        rows.append({
            "seq": s, "t_step": f["t_step"], "task_phase": phase[s],
            "n_clusters": f["n_clusters"],
            "rank1_name": (f["clusters"][0]["name"] if f["clusters"] else None),
            "rank1_score": (f["clusters"][0]["target_score"] if f["clusters"] else None),
            "before_name": b["name"], "before_status": b["status"],
            "after_name": a["name"], "after_rank": a["chosen_rank"],
            "after_score": a["target_score"], "after_mode": mode,
            "after_confirm_streak": a["confirm_streak"],
            "after_point_count": a["point_count"],
            "after_rms_radius_mm": (None if a["rms_radius_mm"] is None
                                    else round(a["rms_radius_mm"], 2)),
            "after_spatial_compactness": a.get("spatial_compactness"),
            "after_name_purity": (None if ch is None else round(ch["name_purity"], 4)),
            "after_name_counts": (None if ch is None else ch["name_counts"]),
            "after_top_body": (None if ch is None else ch["top_body"]),
            "attention_peak_name": f["attention_peak_name"],
            "gt_min_clearance_mm": round(g10["min_mm_dropping_failed_zeros"], 4),
            "gt_min_pair": g10["min_pair_dropping_failed_zeros"],
            "gt_per_object_mm": g10["per_object_mm_dropping_zeros"],
            "gripper_obs_left": round(f["gripper_obs_left"], 4),
            "apple_z_m": round(f["apple_z_m"], 4),
            "ee_apple_dist_mm": round(f["ee_apple_dist_mm"], 1),
        })

    # --- table / unknown / 번진 덩어리가 target 이 된 chunk --------------------------------
    bad_name = [r["seq"] for r in rows if r["after_name"] in ("table", "unknown", "robot")]
    bad_name_before = [r["seq"] for r in rows if r["before_name"] in ("table", "unknown", "robot")]
    bad_name_rank1 = [r["seq"] for r in rows if r["rank1_name"] in ("table", "unknown", "robot")]
    # 순도가 낮은 target (member point 의 다수결 이름이 절반 이하)
    impure = [{"seq": r["seq"], "name": r["after_name"], "purity": r["after_name_purity"],
               "counts": r["after_name_counts"]}
              for r in rows if r["after_name_purity"] is not None and r["after_name_purity"] < 0.9]
    # T5e 가 단위 테스트에서 "번진 덩어리" 로 부른 모양: 10,000 점 이상 · compactness < 0.01
    flood_t5e = [r["seq"] for r in rows
                 if (r["after_point_count"] or 0) >= 10000
                 and (r["after_spatial_compactness"] or 1.0) < 0.01]
    # 그 기준에 닿지 않아도 크기·퍼짐이 물체 크기를 넘는 것
    big = [{"seq": r["seq"], "name": r["after_name"], "point_count": r["after_point_count"],
            "rms_radius_mm": r["after_rms_radius_mm"],
            "spatial_compactness": r["after_spatial_compactness"],
            "purity": r["after_name_purity"], "counts": r["after_name_counts"]}
           for r in rows if (r["after_rms_radius_mm"] or 0) >= 100.0]

    by_name_after = {}
    for nm in sorted({r["after_name"] for r in rows if r["after_name"]}):
        sel = [r for r in rows if r["after_name"] == nm]
        by_name_after[nm] = {
            "n": len(sel), "seqs": [r["seq"] for r in sel],
            "point_count": stats([r["after_point_count"] for r in sel], 1),
            "rms_radius_mm": stats([r["after_rms_radius_mm"] for r in sel], 2),
            "target_score": stats([r["after_score"] for r in sel], 4),
            "name_purity": stats([r["after_name_purity"] for r in sel], 4),
            "task_phases": sorted({r["task_phase"] for r in sel}),
        }

    # --- live violated 7 건 x 참값 clearance ----------------------------------------------
    after_run = shadow["runs"]["after_top1_confirm3"]
    before_run = shadow["runs"]["before_threshold_0p25"]
    viol_rows = []
    for v in after_run["violated_detail"]:
        r = rows[v["seq"] - 1]
        g = gt_by_seq[v["seq"]]["by_distmax"]["0.10"]
        was = None
        for n in v["notes"]:
            if "was " in n and "mm of penetration remains" in n:
                was = float(n.split("(was ")[1].split(" mm")[0])
        viol_rows.append({
            "seq": v["seq"], "t_step": v["t_step"],
            "max_violation_mm": v["max_violation_mm"],
            "violation_before_sqp_mm": was,
            "ag3s_status": v["ag3s_status"],
            "geometry_certified": v["geometry_certified"],
            "n_tiers": v["n_tiers"],
            "task_phase": r["task_phase"],
            "target_name_after": r["after_name"],
            "gt_min_clearance_mm": round(g["min_mm_dropping_failed_zeros"], 4),
            "gt_min_pair": g["min_pair_dropping_failed_zeros"],
            "gt_saturated_at_100mm": g["saturated"],
            "gt_per_object_mm": g["per_object_mm_dropping_zeros"],
            "gt_per_object_link": g["per_object_link"],
        })

    # --- 회귀 기준선 ----------------------------------------------------------------------
    bf = base["frames"]
    baseline = {
        "frames": len(bf),
        "violated_start": sum(1 for f in bf if f["clearance_before_mm"] < 0),
        "has_target": sum(1 for f in bf if f["target"]),
        "frame0_clearance_before_mm": bf[0]["clearance_before_mm"],
        "resolved": sum(1 for f in bf
                        if f["clearance_before_mm"] < 0 and f["status"] == "feasible"),
        "improved": sum(1 for f in bf if f["clearance_after_mm"] > f["clearance_before_mm"]),
        "status_counts": dict(Counter(f["status"] for f in bf)),
    }

    # --- T5d rank-1 구간과의 대조 ---------------------------------------------------------
    t5d_segs = [{"name": s["top_name"], "seq_from": s["seq_from"], "seq_to": s["seq_to"],
                 "n": s["n"]} for s in t5d["segments_by_top_name"]]
    mine_1_51 = segments([(s, n) for s, n in rank1 if s <= 51])
    same = (t5d_segs == mine_1_51)

    out = {
        "sources": {
            "target_identity": "outputs/verify/T5f/target_identity_ep1807.json",
            "gt_clearance": "outputs/verify/T5f/gt_clearance_ep1807.json",
            "shadow_agg": "outputs/verify/T5f/shadow_agg_two_runs.json",
            "baseline": "outputs/verify/T5f/baseline_check.json",
        },
        "config_now": ident["config_now"],
        "old_threshold": ident["old_threshold_used_for_derivation"],
        "task_phase_rule": {
            "gripper_closed_below": GRIPPER_CLOSED_BELOW,
            "apple_z0_m": apple_z0,
            "lift_mm": LIFT_MM,
            "over_crate_xy_mm": OVER_CRATE_XY_MM,
            "first_closed_seq": first_closed,
            "first_reopen_seq": first_reopen,
        },
        "task_phase_counts": dict(Counter(phase.values())),
        "n_chunks": len(rows),
        "name_counts": {
            "rank1": dict(Counter(n for _s, n in rank1)),
            "before_threshold_0p25": dict(Counter(str(n) for _s, n in before)),
            "after_top1_confirm3": dict(Counter(str(n) for _s, n in after)),
        },
        "has_target_offline": {
            "before_threshold_0p25": sum(1 for _s, n in before if n is not None),
            "after_top1_confirm3": sum(1 for _s, n in after if n is not None),
        },
        "segments": {
            "rank1": segments(rank1),
            "before_threshold_0p25": segments(before),
            "after_top1_confirm3": segments(after),
        },
        "t5d_cross_check": {
            "t5d_segments_seq_1_51": t5d_segs,
            "mine_rank1_segments_seq_1_51": mine_1_51,
            "identical": same,
            "t5d_offline_has_target_1_51": t5d["counts_seq_1_51"]["offline_has_target"],
            "mine_before_has_target_1_51": sum(1 for s, n in before if s <= 51 and n),
        },
        "wrong_object_as_target": {
            "table_or_unknown_or_robot_after": bad_name,
            "table_or_unknown_or_robot_before": bad_name_before,
            "table_or_unknown_or_robot_rank1": bad_name_rank1,
            "n_after": len(bad_name),
            "impure_target_purity_below_0p9": impure,
            "flood_by_t5e_unit_test_shape": flood_t5e,
            "target_rms_radius_at_or_over_100mm": big,
            "n_target_rms_radius_at_or_over_100mm": len(big),
        },
        "target_size_by_name_after": by_name_after,
        "confirm_activity": {
            "mode_counts": dict(Counter(r["after_mode"] for r in rows)),
            "hold_seqs": [r["seq"] for r in rows if r["after_mode"] == "hold"],
            "switch_seqs": [r["seq"] for r in rows if r["after_mode"] == "switch"],
            "unobserved_seqs": [r["seq"] for r in rows if r["after_mode"] == "unobserved"],
            "chosen_rank_counts": dict(Counter(r["after_rank"] for r in rows)),
            "seqs_where_hold_changed_the_name": [
                r["seq"] for r in rows
                if r["after_mode"] == "hold" and r["after_name"] != r["rank1_name"]],
        },
        "live_violated_after": viol_rows,
        "live_violated_before": before_run["violated_detail"],
        "live_counts_side_by_side": shadow["side_by_side"],
        "camera_timing": {
            "before": {"span_ms": before_run["camera_span_ms"],
                       "n_ge_100ms": before_run["n_span_at_or_over_100ms"],
                       "n_ge_90ms": before_run["n_span_at_or_over_90ms"],
                       "seqs_ge_100ms": before_run["seqs"]["camera_span_at_or_over_100ms"],
                       "mean_gap_ms": before_run["mean_gap_ms"],
                       "degraded_seqs": before_run["seqs"]["ag3s_degraded"]},
            "after": {"span_ms": after_run["camera_span_ms"],
                      "n_ge_100ms": after_run["n_span_at_or_over_100ms"],
                      "n_ge_90ms": after_run["n_span_at_or_over_90ms"],
                      "seqs_ge_100ms": after_run["seqs"]["camera_span_at_or_over_100ms"],
                      "mean_gap_ms": after_run["mean_gap_ms"],
                      "degraded_seqs": after_run["seqs"]["ag3s_degraded"]},
        },
        "gt_clearance_health": {
            "distmaxes_m": gtc["distmaxes_m"],
            "seqs_disagree_0p05_vs_0p10": gtc["seqs_disagree_0p05_vs_0p10"],
            "seqs_disagree_0p02_vs_0p05": gtc["seqs_disagree_0p02_vs_0p05"],
            "n_frames_with_failed_zero_pairs_at_0p10": sum(
                1 for f in gtc["frames"]
                if f["by_distmax"]["0.10"]["n_pairs_exactly_zero"]),
            "total_failed_zero_pairs_at_0p10": sum(
                f["by_distmax"]["0.10"]["n_pairs_exactly_zero"] for f in gtc["frames"]),
            "n_arm_geoms": gtc["n_arm_geoms"], "n_object_geoms": gtc["n_object_geoms"],
        },
        "gt_clearance_all_chunks": {
            "min_mm": stats([r["gt_min_clearance_mm"] for r in rows]),
            "n_negative": sum(1 for r in rows if r["gt_min_clearance_mm"] < 0),
            "negative_seqs": [r["seq"] for r in rows if r["gt_min_clearance_mm"] < 0],
            "pair_counts": dict(Counter(tuple(r["gt_min_pair"]) if r["gt_min_pair"] else None
                                        for r in rows).most_common()),
        },
        "baseline_check": baseline,
        "rows": rows,
    }
    # tuple key -> str
    out["gt_clearance_all_chunks"]["pair_counts"] = {
        f"{k[0]} vs {k[1]}" if k else "none": v
        for k, v in out["gt_clearance_all_chunks"]["pair_counts"].items()}

    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for k in ("name_counts", "has_target_offline", "segments", "t5d_cross_check",
              "wrong_object_as_target", "target_size_by_name_after", "confirm_activity",
              "task_phase_counts", "task_phase_rule", "gt_clearance_health",
              "gt_clearance_all_chunks", "baseline_check"):
        print(f"== {k}\n{json.dumps(out[k], indent=1, ensure_ascii=False)}")
    print("== live_violated_after")
    for v in out["live_violated_after"]:
        print(" ", json.dumps(v, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
