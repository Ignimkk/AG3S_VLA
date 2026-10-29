"""T5d 집계 — frame 표 · 구간 요약 · 실패 grounding 해부 · grace window 덮개 표.

입력은 전부 파일이다. 새로 재지 않는다.
  outputs/verify/T5d/attention_flow_ep1807.json   (offline replay, 75 frame)
  outputs/verify/T5d/apple_stages_ep1807.json     (apple point 단계별 생존, seq 1..51)
  AG3S/ag3s/docs/handoff/T5c.verify.json     (live shadow 경로의 seq 라벨)

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5d/aggregate_T5d.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np  # noqa: E402

from AG3S.trajopt.grasp_latch import CentroidIdentity, GraspLatch, LatchConfig  # noqa: E402

BASE = pathlib.Path(_ag3s_workspace_path(''))
FLOW = BASE / "outputs/verify/T5d/attention_flow_ep1807.json"
STAGES = BASE / "outputs/verify/T5d/apple_stages_ep1807.json"
T5C = BASE / "AG3S/ag3s/docs/handoff/T5c.verify.json"
OUT = BASE / "outputs/verify/T5d/aggregate_T5d.json"

SCOPE = 51  # seq 1..51 = t_step 0..400


def runs_of(seqs: list[int]) -> list[dict]:
    """정렬된 seq 리스트를 연속 구간으로 접는다."""
    out = []
    for s in sorted(seqs):
        if out and s == out[-1]["seq_to"] + 1:
            out[-1]["seq_to"] = s
            out[-1]["n"] += 1
        else:
            out.append({"seq_from": s, "seq_to": s, "n": 1})
    return out


def summ(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    return {"n": len(xs), "min": xs[0], "p50": xs[len(xs) // 2], "max": xs[-1],
            "mean": float(np.mean(xs))}


def main() -> None:
    flow = json.loads(FLOW.read_text())
    frames = flow["frames"]
    threshold = float(flow["target_score_threshold"])
    stages = json.loads(STAGES.read_text())
    stage_by_seq = {f["seq"]: f for f in stages["frames"]}
    t5c = json.loads(T5C.read_text())["numbers"]

    # ---- live shadow 경로의 실패 seq (T5c 의 라벨) ------------------------------------
    live_no_target = [s for s in t5c["no_target_chunks"]["seqs"] if s <= SCOPE]
    live_degraded_low = [d["seq"] for d in t5c["camera_timing"]["degraded_chunks"]
                         if d["grounding_status"] == "low_score" and d["seq"] <= SCOPE]
    live_failing = sorted(set(live_no_target) | set(live_degraded_low))
    live_has_target = [s for s in t5c["no_target_chunks"]["has_target_seqs"] if s <= SCOPE]

    # ---- 과제 국면 라벨 (gripper 관측 + apple 참값 z + apple<->crate xy) --------------
    closed_below = float(flow["latch_config"]["gripper_closed_below"])
    z0 = float(frames[0]["apple_z_m"])
    lift_mm = 20.0
    crate_mm = 100.0
    closed_seqs = [f["seq"] for f in frames if f["gripper_obs_left"] < closed_below]
    first_closed = min(closed_seqs) if closed_seqs else None
    reopen = next((f["seq"] for f in frames
                   if first_closed is not None and f["seq"] > first_closed
                   and f["gripper_obs_left"] >= closed_below), None)
    for f in frames:
        closed = f["gripper_obs_left"] < closed_below
        lifted = (f["apple_z_m"] - z0) * 1000.0 > lift_mm
        over_crate = f["apple_crate_xy_dist_mm"] <= crate_mm
        if not closed and (first_closed is None or f["seq"] < first_closed):
            f["task_phase"] = "approach"
        elif closed and not lifted:
            f["task_phase"] = "grasp"
        elif closed and lifted and not over_crate:
            f["task_phase"] = "transport"
        elif closed and lifted and over_crate:
            f["task_phase"] = "place"
        else:
            f["task_phase"] = "retreat"

    # ---- latch 재계산: safe_policy 가 쓰는 열 6 과 16D 의 옳은 열 7 을 나란히 --------
    latch_series = {}
    for col in (6, 7):
        latch = GraspLatch(LatchConfig())
        ident = CentroidIdentity()
        rows = []
        for f in frames:
            has = bool(f["has_target"])
            centroid = f["clusters"][0]["centroid"] if (has and f["clusters"]) else None
            label = ident.label(centroid)
            score = float(f["clusters"][0]["target_score"]) if (has and f["clusters"]) else 0.0
            runner = (float(f["clusters"][1]["target_score"])
                      if has and len(f["clusters"]) > 1 else 0.0)
            grip = f["gripper_chunk_col6"] if col == 6 else f["gripper_chunk_col7"]
            ev = latch.update(label=label, score=score, runner_up=runner,
                              gripper=grip, placed=False)
            rows.append({"seq": f["seq"], "t_step": f["t_step"], "label": label,
                         "phase": ev.phase.value, "manipulated": ev.manipulated,
                         "destination": ev.destination,
                         "attach": bool(ev.attach), "detach": bool(ev.detach)})
        latch_series[f"gripper_chunk_col{col}"] = rows

    in_scope = [f for f in frames if f["seq"] <= SCOPE]

    # ---- 재는 것 1: frame 표 + 구간 요약 --------------------------------------------
    frame_table = [{
        "seq": f["seq"], "t_step": f["t_step"],
        "top_name": f["top_name"], "top_score": f["top_score"],
        "top_point_count": f["top_point_count"],
        "apple_rank": f["apple_rank"], "apple_score": f["apple_score"],
        "n_clusters": f["n_clusters"],
        "offline_grounding_status": f["grounding_status"],
        "offline_has_target": f["has_target"],
        "live_has_target": (f["seq"] in live_has_target),
        "attention_peak_name": f["attention_peak_name"],
        "task_phase": f["task_phase"],
        "gripper_obs_left": f["gripper_obs_left"],
        "apple_z_m": f["apple_z_m"],
        "apple_crate_xy_dist_mm": f["apple_crate_xy_dist_mm"],
        "ee_apple_dist_mm": f["ee_apple_dist_mm"],
        "apple_cloud_points": f["apple_cloud_points"],
        "latch_manipulated_col6": latch_series["gripper_chunk_col6"][f["i"]]["manipulated"],
        "latch_manipulated_col7": latch_series["gripper_chunk_col7"][f["i"]]["manipulated"],
        "latch_phase_col7": latch_series["gripper_chunk_col7"][f["i"]]["phase"],
    } for f in in_scope]

    segments = []
    for row in frame_table:
        if segments and segments[-1]["top_name"] == row["top_name"]:
            seg = segments[-1]
            seg["seq_to"] = row["seq"]
            seg["n"] += 1
            seg["_scores"].append(row["top_score"])
            seg["_phases"].append(row["task_phase"])
        else:
            segments.append({"top_name": row["top_name"], "seq_from": row["seq"],
                             "seq_to": row["seq"], "n": 1,
                             "_scores": [row["top_score"]], "_phases": [row["task_phase"]]})
    for seg in segments:
        s = summ(seg.pop("_scores"))
        ph = seg.pop("_phases")
        seg["top_score"] = s
        seg["t_step_from"] = (seg["seq_from"] - 1) * 8
        seg["t_step_to"] = (seg["seq_to"] - 1) * 8
        seg["task_phases"] = sorted(set(ph), key=ph.index)

    apple_rank_hist = {}
    for row in frame_table:
        k = "absent" if row["apple_rank"] is None else str(row["apple_rank"])
        apple_rank_hist[k] = apple_rank_hist.get(k, 0) + 1

    # ---- 재는 것 2: 실패한 grounding 의 해부 ----------------------------------------
    def dissect(seq_list: list[int], tag: str) -> dict:
        rows = []
        for s in seq_list:
            f = frames[s - 1]
            st = stage_by_seq.get(s)
            rows.append({
                "seq": s, "t_step": f["t_step"], "task_phase": f["task_phase"],
                "offline_grounding_status": f["grounding_status"],
                "offline_best_score": f["best_score"],
                "threshold_gap": f["threshold_gap"],
                "top_name": f["top_name"], "top_point_count": f["top_point_count"],
                "n_clusters": f["n_clusters"],
                "apple_cluster_present": f["apple_cluster_present"],
                "apple_rank": f["apple_rank"], "apple_score": f["apple_score"],
                "apple_cloud_points": f["apple_cloud_points"],
                "apple_gt_pixels_total": f["apple_gt_pixels_total"],
                "apple_after_recon": (st["n_after_recon_total"] if st else None),
                "apple_after_self_filter": (st["n_after_filter_total"] if st else None),
                "attention_peak_name": f["attention_peak_name"],
                "latch_manipulated_col6": latch_series["gripper_chunk_col6"][s - 1]["manipulated"],
                "latch_manipulated_col7": latch_series["gripper_chunk_col7"][s - 1]["manipulated"],
                "latch_phase_col7": latch_series["gripper_chunk_col7"][s - 1]["phase"],
            })
        gaps = [r["threshold_gap"] for r in rows]
        return {
            "tag": tag,
            "n": len(rows),
            "seqs": seq_list,
            "top_name_counts": {k: sum(1 for r in rows if r["top_name"] == k)
                                for k in sorted({r["top_name"] for r in rows})},
            "n_apple_cluster_present": sum(1 for r in rows if r["apple_cluster_present"]),
            "n_apple_cluster_absent": sum(1 for r in rows if not r["apple_cluster_present"]),
            "n_apple_zero_cloud_points": sum(1 for r in rows if r["apple_cloud_points"] == 0),
            "n_apple_cloud_points_gt_zero": sum(1 for r in rows if r["apple_cloud_points"] > 0),
            "apple_rank_counts": {("absent" if r["apple_rank"] is None else str(r["apple_rank"])):
                                  sum(1 for q in rows if q["apple_rank"] == r["apple_rank"])
                                  for r in rows},
            "threshold_gap": summ(gaps),
            "n_gap_within_0p05": sum(1 for g in gaps if 0.0 < g <= 0.05),
            "n_gap_over_0p10": sum(1 for g in gaps if g > 0.10),
            "n_offline_status_ok": sum(1 for r in rows
                                       if r["offline_grounding_status"] == "ok"),
            "latch_manipulated_col7_counts": {
                str(k): sum(1 for r in rows if r["latch_manipulated_col7"] == k)
                for k in sorted({str(r["latch_manipulated_col7"]) for r in rows})},
            "latch_manipulated_col6_counts": {
                str(k): sum(1 for r in rows if r["latch_manipulated_col6"] == k)
                for k in sorted({str(r["latch_manipulated_col6"]) for r in rows})},
            "rows": rows,
        }

    offline_failing = [f["seq"] for f in in_scope if not f["has_target"]]
    dissect_live = dissect(live_failing, "live_shadow_failing_seq_1_51")
    dissect_offline = dissect(offline_failing, "offline_replay_failing_seq_1_51")

    # ---- 재는 것 3: grace window ----------------------------------------------------
    grasp_seq = first_closed
    place_seq = reopen
    radius = 2

    def window_report(seq_list: list[int], tag: str) -> dict:
        rr = runs_of(seq_list)
        for r in rr:
            r["t_step_from"] = (r["seq_from"] - 1) * 8
            r["t_step_to"] = (r["seq_to"] - 1) * 8
            r["task_phases"] = sorted(
                {frames[s - 1]["task_phase"] for s in range(r["seq_from"], r["seq_to"] + 1)},
                key=lambda p: ["approach", "grasp", "transport", "place", "retreat"].index(p))
            touches_grasp = grasp_seq is not None and not (
                r["seq_to"] < grasp_seq - radius or r["seq_from"] > grasp_seq + radius)
            touches_place = place_seq is not None and not (
                r["seq_to"] < place_seq - radius or r["seq_from"] > place_seq + radius)
            r["touches_grasp_boundary"] = bool(touches_grasp)
            r["touches_place_boundary"] = bool(touches_place)
            r["boundary_class"] = ("grasp_and_place" if touches_grasp and touches_place else
                                   "grasp" if touches_grasp else
                                   "place" if touches_place else "other")
        total_runs = len(rr)
        total_frames = sum(r["n"] for r in rr)
        cov = []
        for n in range(1, 21):
            covered_runs = [r for r in rr if r["n"] <= n]
            cov.append({
                "N": n,
                "runs_covered": len(covered_runs),
                "runs_covered_pct": (100.0 * len(covered_runs) / total_runs
                                     if total_runs else None),
                "frames_covered": sum(r["n"] for r in covered_runs),
                "frames_covered_pct": (100.0 * sum(r["n"] for r in covered_runs) / total_frames
                                       if total_frames else None),
            })
        return {
            "tag": tag,
            "boundary_radius_seq": radius,
            "grasp_boundary_seq": grasp_seq,
            "place_boundary_seq": place_seq,
            "n_failing_frames": total_frames,
            "n_runs": total_runs,
            "longest_run_n": max((r["n"] for r in rr), default=0),
            "run_length_histogram": {str(k): sum(1 for r in rr if r["n"] == k)
                                     for k in sorted({r["n"] for r in rr})},
            "runs": rr,
            "runs_by_boundary_class": {
                k: {"n_runs": sum(1 for r in rr if r["boundary_class"] == k),
                    "n_frames": sum(r["n"] for r in rr if r["boundary_class"] == k),
                    "run_lengths": [r["n"] for r in rr if r["boundary_class"] == k]}
                for k in ("grasp", "place", "grasp_and_place", "other")},
            "coverage_by_N": cov,
        }

    out = {
        "inputs": {"flow": str(FLOW.relative_to(BASE)), "stages": str(STAGES.relative_to(BASE)),
                   "t5c": str(T5C.relative_to(BASE))},
        "scope": {"seq_from": 1, "seq_to": SCOPE, "t_step_to": (SCOPE - 1) * 8,
                  "t_step_of_seq": "t_step = (seq - 1) * 8"},
        "target_score_threshold": threshold,
        "task_phase_rule": {
            "gripper_closed_below": closed_below,
            "apple_z0_m": z0,
            "lift_mm": lift_mm,
            "over_crate_xy_mm": crate_mm,
            "first_closed_seq": first_closed,
            "first_reopen_seq": reopen,
            "rule": ("approach = gripper open and seq < first_closed; grasp = closed and apple "
                     "lift <= 20 mm; transport = closed and lifted and apple<->crate xy > 100 mm; "
                     "place = closed and lifted and xy <= 100 mm; retreat = open after reopen"),
        },
        "counts_seq_1_51": {
            "n_frames": len(in_scope),
            "offline_has_target": sum(1 for f in in_scope if f["has_target"]),
            "offline_failing": len(offline_failing),
            "live_has_target": len(live_has_target),
            "live_failing": len(live_failing),
            "live_no_target": len(live_no_target),
            "live_degraded_low_score": len(live_degraded_low),
            "task_phase_counts": {k: sum(1 for f in in_scope if f["task_phase"] == k)
                                  for k in ("approach", "grasp", "transport", "place", "retreat")},
            "top_name_counts": {k: sum(1 for f in in_scope if f["top_name"] == k)
                                for k in sorted({str(f["top_name"]) for f in in_scope})},
            "apple_rank_histogram": apple_rank_hist,
            "attention_peak_name_counts": {
                k: sum(1 for f in in_scope if f["attention_peak_name"] == k)
                for k in sorted({str(f["attention_peak_name"]) for f in in_scope})},
        },
        "frame_table_seq_1_51": frame_table,
        "segments_by_top_name": segments,
        "apple_score_series_seq_1_51": [
            {"seq": f["seq"], "t_step": f["t_step"], "apple_score": f["apple_score"],
             "apple_rank": f["apple_rank"], "top_score": f["top_score"],
             "top_name": f["top_name"]} for f in in_scope],
        "dissect_live_failing": dissect_live,
        "dissect_offline_failing": dissect_offline,
        "grace_window_live": window_report(live_failing, "live_shadow_failing_seq_1_51"),
        "grace_window_offline": window_report(offline_failing, "offline_replay_failing_seq_1_51"),
        "latch_series": latch_series,
        "latch_note": ("safe_policy.py:411 은 left gripper 열을 6 으로 박아 두었다. 16D 의 "
                       "wire.gripper_columns() 는 (7, 15) 를 준다. 두 열로 각각 latch 를 "
                       "돌린 결과를 나란히 남긴다"),
        "frames_full_75": frames,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}")
    print(json.dumps(out["counts_seq_1_51"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
