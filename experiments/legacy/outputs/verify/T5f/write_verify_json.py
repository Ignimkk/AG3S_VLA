"""T5f — `handoff/T5f.verify.json` 을 aggregate 에서 그대로 만든다.

여기서 새로 계산하는 숫자는 없다. `aggregate_T5f.json` 에 있는 것만 옮긴다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5f/write_verify_json.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5f'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T5f.verify.json'))

PRE = "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u"


def main() -> None:
    a = json.loads((V / "aggregate_T5f.json").read_text())
    sh = json.loads((V / "shadow_agg_two_runs.json").read_text())
    rows = a["rows"]
    before_run = sh["runs"]["before_threshold_0p25"]
    after_run = sh["runs"]["after_top1_confirm3"]

    def pick(r):
        return {k: r[k] for k in (
            "seq", "t_step", "task_phase", "n_clusters",
            "rank1_name", "rank1_score", "before_name", "before_status",
            "after_name", "after_rank", "after_score", "after_mode",
            "after_confirm_streak", "after_point_count", "after_rms_radius_mm",
            "after_name_purity", "after_name_counts", "attention_peak_name",
            "gt_min_clearance_mm", "gt_min_pair")}

    doc = {
        "_comment": "ag3s-verifier(A2) 가 쓴다. 해석을 쓰지 않는다 — 숫자와 경로만.",
        "step": "T5f",
        "date": "2026-09-26",
        "record_consumed": {
            "live_before": "outputs/live_test/20260925_t5shadow/shadow_ep1807/",
            "live_after": "outputs/live_test/20260925_t5e_shadow/shadow_ep1807/",
            "client_logs": ["outputs/live_test/logs/t5shadow_run.log",
                            "outputs/live_test/logs/t5e_shadow_run.log"],
            "server_logs": ["outputs/live_test/logs/serve_safe_8000_shadow.log",
                            "outputs/live_test/logs/serve_safe_8000_t5e.log"],
            "offline_replay_records": "outputs/live_test/20260925_ep1807/run_0000",
            "offline_attention_npz": "AG3S/ag3s/asset/data/attention_16d_ep1807.npz",
            "regression_baseline_records": "run_16d_ep1800",
            "no_new_capture": True,
            "prompt": "Pick up the apple and place it in the basket.",
            "episode": 1807,
            "chunks": 75,
            "naming_provenance": (
                "chunk 별 target 이름은 **offline replay** 에서 나온다. live shadow 기록에는 "
                "cluster·qpos·물체 자세가 없다 (T5c 가 확인). offline replay 는 같은 ep1807 의 "
                "다른 rollout(`20260925_ep1807/run_0000`)이고 t_step 으로 맞춘다 — T5c 가 오른팔 "
                "7 관절 최대 절대차 0.00-0.02 rad 로 정렬을 확인했다. 왼팔은 shadow 기록에 없어 "
                "not_measured"),
            "naming_rule": (
                "cluster 이름 = member point 의 MuJoCo segmentation body 다수결. point 는 "
                "`FusedPointCloud.representative` -> `obs_camera`/`obs_uv` 로 원래 pixel 까지 "
                "되짚었다 (T5d 와 같은 방식). apple/banana/orange/pear/crate/table 은 그 body "
                "이름, base 후손 링크와 *_target marker 는 robot, 그 밖은 unknown"),
            "before_derivation_rule": (
                "옛 코드는 `best.target_score < target_score_threshold(0.25)` 이면 LOW_SCORE, "
                "아니면 `clusters[0]`. clustering 은 threshold·confirm 과 무관하므로 같은 "
                "cluster 목록에 그 분기를 다시 적용한 것이다 — 근사가 아니다. 교차 확인: "
                "이렇게 얻은 offline has_target 이 seq 1-51 에서 21, 전체 75 에서 29 로 "
                "T5c·T5d 의 수치와 같다"),
        },
        "runs": [
            {"id": "baseline",
             "command": f"{PRE} -m AG3S.trajopt.experiments.esdf_rollout --records "
                        "run_16d_ep1800 --attention "
                        "AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 "
                        "--voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces "
                        "field --constraint-links arms --out-json "
                        "outputs/verify/T5f/baseline_check.json",
             "venv": ".venv-ag3s",
             "raw_output": "outputs/verify/T5f/baseline_check.json",
             "stdout_log": "outputs/verify/T5f/baseline_check.log"},
            {"id": "shadow_agg_two_runs",
             "command": f"{PRE} outputs/verify/T5f/agg_two_shadow_runs.py",
             "venv": ".venv-ag3s",
             "script": "outputs/verify/T5f/agg_two_shadow_runs.py",
             "raw_output": "outputs/verify/T5f/shadow_agg_two_runs.json",
             "stdout_log": "outputs/verify/T5f/shadow_agg_two_runs.log"},
            {"id": "target_identity_ep1807",
             "command": f"{PRE} outputs/verify/T5f/measure_target_identity_ep1807.py",
             "venv": ".venv-ag3s",
             "script": "outputs/verify/T5f/measure_target_identity_ep1807.py",
             "raw_output": "outputs/verify/T5f/target_identity_ep1807.json",
             "stdout_log": "outputs/verify/T5f/target_identity_ep1807.log"},
            {"id": "gt_clearance_ep1807",
             "command": f"{PRE} outputs/verify/T5f/measure_gt_clearance_ep1807.py",
             "venv": ".venv-ag3s",
             "script": "outputs/verify/T5f/measure_gt_clearance_ep1807.py",
             "raw_output": "outputs/verify/T5f/gt_clearance_ep1807.json",
             "stdout_log": "outputs/verify/T5f/gt_clearance_ep1807.log"},
            {"id": "aggregate",
             "command": f"{PRE} outputs/verify/T5f/aggregate_T5f.py",
             "venv": ".venv-ag3s",
             "script": "outputs/verify/T5f/aggregate_T5f.py",
             "raw_output": "outputs/verify/T5f/aggregate_T5f.json",
             "stdout_log": "outputs/verify/T5f/aggregate_T5f.log"},
            {"id": "object_extents_gt",
             "command": "inline python (see outputs/verify/T5f/object_extents_gt.json)",
             "venv": ".venv-ag3s",
             "raw_output": "outputs/verify/T5f/object_extents_gt.json"},
        ],
        "code_state": {
            "repo": "benchmark",
            "head": "44fa666",
            "dirty_files": [
                "AG3S/ag3s/config.py", "AG3S/ag3s/configs/default.yaml",
                "AG3S/ag3s/configs/rby1_three_camera.yaml",
                "AG3S/ag3s/constraints/to_adapter.py",
                "AG3S/ag3s/runtime/multiview.py", "AG3S/ag3s/runtime/pipeline.py",
                "AG3S/ag3s/stages/target_grounding.py",
                "AG3S/ag3s/runtime/degradation.py",
                "AG3S/trajopt/client.py", "AG3S/trajopt/safe_policy.py",
                "AG3S/trajopt/serve_safe.py", "AG3S/trajopt/wire.py"],
            "work_root_head": "b9386c9",
            "code_changed_by_verifier": "없음 — 읽기만 했다",
            "config_in_effect": a["config_now"],
        },
        "numbers": {
            "_comment": "로그와 논문에 쓰이는 모든 수치가 여기에만 있다. 여기 없는 숫자는 쓰지 않는다.",

            "live_counts": {
                "n_chunks": 75,
                "before_threshold_0p25": {
                    "safe": before_run["counts"]["safe_true"],
                    "unsafe": before_run["counts"]["safe_false"],
                    "ag3s_ok": before_run["counts"]["ag3s_status"].get("ok", 0),
                    "ag3s_no_target": before_run["counts"]["ag3s_status"].get("no_target", 0),
                    "ag3s_degraded": before_run["counts"]["ag3s_status"].get("degraded", 0),
                    "trajopt_feasible": before_run["counts"]["trajopt_status"]["feasible"],
                    "trajopt_violated": before_run["counts"]["trajopt_status"]["violated"],
                    "geometry_certified": before_run["counts"]["geometry_certified_true"],
                    "field_two_tier_chunks": len(before_run["seqs"]["two_tiers"]),
                    "field_one_tier_chunks": len(before_run["seqs"]["one_tier"]),
                    "reason_code_combos": before_run["counts"]["reason_code_combos"],
                },
                "after_top1_confirm3": {
                    "safe": after_run["counts"]["safe_true"],
                    "unsafe": after_run["counts"]["safe_false"],
                    "ag3s_ok": after_run["counts"]["ag3s_status"].get("ok", 0),
                    "ag3s_no_target": after_run["counts"]["ag3s_status"].get("no_target", 0),
                    "ag3s_degraded": after_run["counts"]["ag3s_status"].get("degraded", 0),
                    "trajopt_feasible": after_run["counts"]["trajopt_status"]["feasible"],
                    "trajopt_violated": after_run["counts"]["trajopt_status"]["violated"],
                    "geometry_certified": after_run["counts"]["geometry_certified_true"],
                    "field_two_tier_chunks": len(after_run["seqs"]["two_tiers"]),
                    "field_one_tier_chunks": len(after_run["seqs"]["one_tier"]),
                    "reason_code_combos": after_run["counts"]["reason_code_combos"],
                },
                "lead_hand_count_agreement": (
                    "브리핑의 다섯 (safe 68/7 · no_target 0 · degraded 0 · violated 7 · "
                    "fine layer 75/75) 은 전부 측정값과 같다. 브리핑의 '이전 TO violated 1' "
                    "만 다르다 — 이전 실행의 `trajopt_status=violated` 는 61 이고, 그 중 "
                    "`max_violation_m > 0` 인 것이 1(seq 4, 3.0031 mm)이다. 나머지 60 은 "
                    "`max_violation_m = 0.0` 이고 geometry_certified=false 로 내려간 것 "
                    "(no_target 54 · degraded 6)"),
                "violated_with_nonzero_violation": {
                    "before": 1, "after": 7,
                    "before_seqs": [4],
                    "after_seqs": [v["seq"] for v in a["live_violated_after"]],
                },
            },

            "target_identity": {
                "source": "offline replay of ep1807 run_0000, 75 chunks",
                "name_counts_rank1": a["name_counts"]["rank1"],
                "name_counts_before_threshold_0p25": a["name_counts"]["before_threshold_0p25"],
                "name_counts_after_top1_confirm3": a["name_counts"]["after_top1_confirm3"],
                "has_target_offline": a["has_target_offline"],
                "table_or_unknown_or_robot_as_target_after_seqs":
                    a["wrong_object_as_target"]["table_or_unknown_or_robot_after"],
                "table_or_unknown_or_robot_as_target_after_n":
                    a["wrong_object_as_target"]["n_after"],
                "table_or_unknown_or_robot_as_target_before_seqs":
                    a["wrong_object_as_target"]["table_or_unknown_or_robot_before"],
                "table_or_unknown_or_robot_as_rank1_seqs":
                    a["wrong_object_as_target"]["table_or_unknown_or_robot_rank1"],
                "flood_by_t5e_unit_test_shape_seqs":
                    a["wrong_object_as_target"]["flood_by_t5e_unit_test_shape"],
                "flood_definition_used": (
                    "T5e 가 고친 단위 테스트가 주장하는 모양 — 10,000 점 이상이고 "
                    "spatial_compactness < 0.01"),
                "target_purity_below_0p9_chunks":
                    a["wrong_object_as_target"]["impure_target_purity_below_0p9"],
                "n_target_rms_radius_at_or_over_100mm":
                    a["wrong_object_as_target"]["n_target_rms_radius_at_or_over_100mm"],
                "seqs_target_rms_radius_at_or_over_100mm": [
                    x["seq"] for x in
                    a["wrong_object_as_target"]["target_rms_radius_at_or_over_100mm"]],
                "target_size_by_name_after": a["target_size_by_name_after"],
                "crate_target_impurity_composition": {
                    "n_chunks_with_purity_below_1": 31,
                    "non_crate_points_summed_over_49_crate_chunks": {"apple": 2464},
                    "worst_purity_chunk": {"seq": 38, "purity": 0.9167,
                                           "counts": {"crate": 1233, "apple": 112}},
                },
                "object_gt_half_diagonal_mm": {
                    "apple": 66.0, "banana": 74.1, "orange": 56.8, "pear": 80.2,
                    "crate": 404.0, "table": 1262.5,
                    "_rule": "body 의 모든 geom 을 (중심 +- 최대 size) 로 감싼 AABB 의 반대각선",
                    "_raw": "outputs/verify/T5f/object_extents_gt.json"},
                "segments": a["segments"],
                "name_flips_between_consecutive_chunks": {
                    "rank1": 9, "before_threshold_0p25": 13, "after_top1_confirm3": 3},
                "chunks_where_before_and_after_differ": 51,
                "chunks_where_before_and_after_agree": 24,
                "chunks_where_before_had_a_different_object": [
                    {"seq": 31, "before": "pear", "after": "crate"},
                    {"seq": 35, "before": "apple", "after": "crate"},
                    {"seq": 66, "before": "apple", "after": "pear"},
                    {"seq": 67, "before": "apple", "after": "pear"},
                    {"seq": 69, "before": "pear", "after": "apple"}],
                "confirm_activity": a["confirm_activity"],
            },

            "t5d_cross_check": {
                "rank1_segments_seq_1_51_identical_to_T5d":
                    a["t5d_cross_check"]["identical"],
                "t5d_segments_seq_1_51": a["t5d_cross_check"]["t5d_segments_seq_1_51"],
                "offline_has_target_seq_1_51": {
                    "t5d": a["t5d_cross_check"]["t5d_offline_has_target_1_51"],
                    "mine": a["t5d_cross_check"]["mine_before_has_target_1_51"]},
                "task_brief_said": "apple 1-17 -> crate 18-51",
                "measured_rank1": "apple 1-15 -> crate 16-30 -> pear 31-32 -> crate 33-34 "
                                  "-> apple 35 -> crate 36-51 (T5d 와 한 chunk 도 다르지 않다)",
                "measured_rank1_seq_52_75": "crate 52-62 -> pear 63-65 -> apple 66-68 "
                                            "-> pear 69 -> apple 70-75",
            },

            "violated_7_after": a["live_violated_after"],
            "violated_1_before": [
                {"seq": v["seq"], "t_step": v["t_step"],
                 "max_violation_mm": v["max_violation_mm"],
                 "ag3s_status": v["ag3s_status"]}
                for v in before_run["violated_detail"] if (v["max_violation_m"] or 0) > 0],

            "gt_clearance": {
                "rule": ("mj_geomDistance(팔 링크 충돌 geom 267 개, 물체 충돌 geom 26 개, "
                         "distmax). 그 chunk 의 관측 qpos **한 자세**에서다 — 궤적 전체가 "
                         "아니다. distmax 0.02/0.05/0.10 셋으로 재고 교차 확인했다"),
                "distmaxes_m": [0.02, 0.05, 0.10],
                "frames_disagreeing_between_distmaxes": 0,
                "quoted_at_distmax_m": 0.10,
                "saturates_at_mm": 100.0,
                "all_75_chunks": a["gt_clearance_all_chunks"]["min_mm"],
                "n_chunks_with_negative_gt_clearance":
                    a["gt_clearance_all_chunks"]["n_negative"],
                "negative_gt_clearance_seqs": a["gt_clearance_all_chunks"]["negative_seqs"],
                "closest_pair_counts": a["gt_clearance_all_chunks"]["pair_counts"],
                "violated_7_gt": [
                    {"seq": v["seq"], "gt_min_clearance_mm": v["gt_min_clearance_mm"],
                     "gt_min_pair": v["gt_min_pair"],
                     "gt_per_object_mm": v["gt_per_object_mm"]}
                    for v in a["live_violated_after"]],
                "measurement_bug_found_and_fixed": (
                    "첫 실행에서 distmax 를 1.0 m 로 주었더니 mesh geom 쌍에서 "
                    "mj_geomDistance 가 **정확히 0.0** 을 돌려주었다 — 실제 거리가 0.37~0.56 m "
                    "인 쌍에서 그랬다. distmax 0.3 m 부터 나타나고 0.1 m 이하에서 사라진다. "
                    "시각 전용 geom(contype=0 and conaffinity=0, 팔 307 중 40 개)도 같은 0.0 을 "
                    "만든다. 둘 다 고친 뒤 distmax 셋이 75 chunk 전부에서 일치한다. "
                    "고치기 전 값은 seq 1 에서 0.0 mm, 고친 뒤 +15.17 mm"),
            },

            "task_phase": {
                "rule": a["task_phase_rule"],
                "counts_seq_1_75": a["task_phase_counts"],
                "violated_7_phases": [
                    {"seq": v["seq"], "task_phase": v["task_phase"]}
                    for v in a["live_violated_after"]],
                "t5d_values_reproduced": {"first_closed_seq": 20, "first_reopen_seq": 33},
            },

            "camera_timing": {
                "limit_ms": 100,
                "before": {
                    "three_camera_capture_span_ms": a["camera_timing"]["before"]["span_ms"],
                    "n_chunks_span_ge_100ms": a["camera_timing"]["before"]["n_ge_100ms"],
                    "n_chunks_span_ge_90ms": a["camera_timing"]["before"]["n_ge_90ms"],
                    "seqs_span_ge_100ms": a["camera_timing"]["before"]["seqs_ge_100ms"],
                    "degraded_seqs": a["camera_timing"]["before"]["degraded_seqs"],
                    "mean_gap_ms": a["camera_timing"]["before"]["mean_gap_ms"]},
                "after": {
                    "three_camera_capture_span_ms": a["camera_timing"]["after"]["span_ms"],
                    "n_chunks_span_ge_100ms": a["camera_timing"]["after"]["n_ge_100ms"],
                    "n_chunks_span_ge_90ms": a["camera_timing"]["after"]["n_ge_90ms"],
                    "seqs_span_ge_100ms": a["camera_timing"]["after"]["seqs_ge_100ms"],
                    "degraded_seqs": a["camera_timing"]["after"]["degraded_seqs"],
                    "mean_gap_ms": a["camera_timing"]["after"]["mean_gap_ms"]},
                "capture_order": ["zed_left", "wrist_cam_l", "wrist_cam_r"],
                "t5c_value_for_before_reproduced": (
                    "p50 93.61 · p95 100.68 · max 107.22 ms — T5c 와 소수점까지 같다"),
            },

            "per_chunk_table": [pick(r) for r in rows],
        },
        "figures": [
            {"path": "AG3S/ag3s/docs/figures/t5f/"
                     "t5f-scene-which-object-became-target.png", "kind": "scene",
             "sidecar": "AG3S/ag3s/docs/figures/t5f/"
                        "t5f-scene-which-object-became-target.json",
             "script": "outputs/verify/T5f/make_fig1_scene.py"},
            {"path": "AG3S/ag3s/docs/figures/t5f/t5f-target-name-bands.png",
             "kind": "graph",
             "sidecar": "AG3S/ag3s/docs/figures/t5f/t5f-target-name-bands.json",
             "script": "outputs/verify/T5f/make_fig2_bands.py"},
            {"path": "AG3S/ag3s/docs/figures/t5f/t5f-tables-before-after.png",
             "kind": "table",
             "sidecar": "AG3S/ag3s/docs/figures/t5f/t5f-tables-before-after.json",
             "script": "outputs/verify/T5f/make_fig3_tables.py"},
        ],
        "baseline_check": {
            "matched": True,
            "measured": a["baseline_check"],
            "note": ("위반으로 시작 14/15 일치 · frame0 clearance_before "
                     "0.15718632962849477 mm 소수점까지 일치 · has_target 은 9/15 -> 15/15 "
                     "(지시서가 이 변경으로 바뀐다고 명시). 개수만 인용하는 셋: 해소 13 · "
                     "개선 15 · feasible 14 / violated 1 — 기준이 아니다"),
        },
        "verdict": ("회귀 기준선 셋 중 재현되는 둘(위반으로 시작 14/15 · frame0 "
                    "clearance_before +0.15718632962849477 mm)이 일치한다. has_target 은 "
                    "9/15 -> 15/15. 브리핑의 눈 센 값 다섯은 측정값과 같고, '이전 TO violated 1' "
                    "하나만 다르다 — 이전 실행의 violated 는 61 이고 그 중 max_violation > 0 인 "
                    "것이 1 이다."),
        "not_measured": [
            "refined 대 reference clearance 비교 — `frames.jsonl` 의 `executed_chunk` 는 "
            "\"reference\" 라는 이름표일 뿐이고 `actions` 배열이 없다 (`chunk_shape` 만 있다). "
            "qpos·물체 자세도 없다. 재려면 기록기가 `actions`·`actions_reference`·`qpos` 를 "
            "남겨야 한다",

            "live `violated` 7 건이 **어느 제약(어느 link 대 어느 물체)** 이었나 — 기록의 "
            "verdict 는 `max_violation_m` 과 note 세 줄만 싣고 위반 제약의 신원을 싣지 않는다. "
            "대신 그 chunk 의 MuJoCo 참값 최소 clearance 와 그 argmin 쌍을 쟀고 "
            "`numbers.gt_clearance.violated_7_gt` 에 있다 — 그것은 참값 기하의 최근접 쌍이지 "
            "SQP 가 어긴 제약이 아니다",

            "live 경로의 cluster·score·target 이름 — 서버 응답에도 두 로그에도 없다 (T5c 가 "
            "확인: `safe_policy.py:294-301` 이 status·validity·grounding_status·reasons·notes "
            "만 싣는다). 이번 STEP 의 이름은 전부 offline replay 에서 나온 것이다",

            "왼팔(작업 팔) 의 offline replay 대 live shadow 정렬 — shadow 기록에 qpos 가 없어 "
            "오른팔 7 관절로만 확인됐다 (T5c, 최대 절대차 0.00-0.02 rad)",

            "참값 clearance 의 **궤적 전체** 값 — 잰 것은 각 chunk 의 관측 qpos 한 자세다. "
            "50 x 16 chunk 를 따라간 값은 live 기록에 actions 가 없어 못 잰다",

            "camera capture 폭이 왜 두 실행에서 다른가 — 두 분포(p50 93.61 -> 74.94 ms)만 "
            "쟀다. 원인은 안 쟀다",
        ],
        "script_promotion_candidate": {
            "path": "outputs/verify/T5f/measure_gt_clearance_ep1807.py",
            "why": ("MuJoCo 참값 clearance 를 재는 최소 script. distmax 함정(1.0 m 를 주면 "
                    "mesh 쌍이 0.0 을 돌려준다)과 시각 전용 geom 제외를 안에 담고 있다. "
                    "`benchmark/` 로 올릴지는 implementer(A1)가 정한다 — verifier 는 "
                    "`benchmark/**` 에 쓰지 않는다"),
        },
    }
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
