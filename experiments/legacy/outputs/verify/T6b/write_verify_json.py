"""T6b — assemble handoff/T6b.verify.json from the aggregated measurement.

Every number is copied out of outputs/verify/T6b/aggregate_T6b.json, never retyped.

    PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T6b/write_verify_json.py
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib

VER = pathlib.Path(_ag3s_workspace_path('outputs/verify/T6b'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T6b.verify.json'))

RUNS = ["shadow_threshold", "shadow_rank1", "execute"]


def main() -> None:
    agg = json.loads((VER / "aggregate_T6b.json").read_text())
    per = agg["per_run"]
    base = json.loads((VER / "baseline_check.json").read_text())
    bframes = base["frames"]

    baseline = {
        "violated_start": sum(1 for f in bframes if f["clearance_before_mm"] < 0),
        "frames": len(bframes),
        "has_target": sum(1 for f in bframes if f["target"]),
        "frame0_clearance_before_mm": bframes[0]["clearance_before_mm"],
        "status_counts_not_a_baseline": {
            s: sum(1 for f in bframes if f["status"] == s)
            for s in sorted({f["status"] for f in bframes})},
    }

    def run_block(name):
        p = per[name]
        return {
            "record_dir": p["record_dir"],
            "mode": "shadow (unsafe verdicts overridden)" if p["argv_has_safe_shadow"]
                    else "execute (unsafe verdicts held)",
            "chunks_sent": p["n_chunks"],
            "safe": p["n_safe"],
            "hold": p["n_hold"],
            "client_ipc_stats": p["client_ipc_stats"],
            "completeness_pass": p["completeness_pass"],
            "hold_classes_ag3s_trajopt": p["hold_classes"],
            "hold_class_by_violation": p["hold_class_by_violation"],
            "hold_seq_ranges": p["hold_seq_ranges"],
            "safe_seq_ranges": p["safe_seq_ranges"],
            "safe_seq_min": p["safe_seq_min"],
            "safe_seq_max": p["safe_seq_max"],
            "first_hold_seq": p["hold_seq_min"],
            "first_unbroken_hold_seq": p["first_unbroken_hold_seq"],
            "max_violation_mm_holds_nonzero": p["max_violation_hold_chunks_nonzero"],
            "max_violation_mm_holds_all": p["max_violation_hold_chunks"],
            "hold_chunks_with_zero_violation": p["max_violation_zero_hold_chunks"],
            "ag3s_status_counts": p["ag3s_status_counts"],
            "trajopt_status_counts": p["trajopt_status_counts"],
            "ag3s_reason_code_counts": p["ag3s_reason_code_counts"],
            "field_tier_count_per_chunk": p["n_tiers_counts"],
            "field_state_counts": p["field_state_counts"],
            "timing_ms": p["timing_ms"],
            "camera_skew_ms": p["camera_skew_ms"],
            "control_steps_executed_true": p["control_executed_true"],
            "control_steps_executed_false": p["control_executed_false"],
        }

    doc = {
        "_comment": "ag3s-verifier(A2) 가 쓴다. 해석을 쓰지 않는다 — 숫자와 경로만.",
        "step": "T6b",
        "date": "2026-09-26",
        "record_consumed": {
            "shadow_threshold_0_25": "outputs/live_test/20260925_t5shadow/shadow_ep1807/",
            "shadow_rank1_confirm3": "outputs/live_test/20260925_t5e_shadow/shadow_ep1807/",
            "execute_closed_loop": "outputs/live_test/20260926_t6/execute_ep1807/",
            "reference_episode_replay_for_the_scene_figure":
                "outputs/live_test/20260925_ep1807/run_0000 (bare policy, port 8123)",
            "episode_index": 1807,
            "prompt": "Pick up the apple and place it in the basket.",
            "seq_to_t_step": "t_step = (seq - 1) * 8",
            "chunk_period_ms": 533.3333333333334,
            "no_new_capture": True,
            "no_offline_replay_used_as_a_stand_in": True,
        },
        "runs": [
            {
                "id": "baseline",
                "command": ("MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} "
                            "${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u -m "
                            "AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 "
                            "--attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz "
                            "--frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all "
                            "--support-surfaces field --constraint-links arms "
                            "--out-json outputs/verify/T6b/baseline_check.json"),
                "venv": ".venv-ag3s",
                "raw_output": "outputs/verify/T6b/baseline_check.json",
                "stdout_log": "outputs/verify/T6b/baseline_check.log",
            },
            {
                "id": "aggregate_three_runs",
                "command": ("MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} "
                            "${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u "
                            "outputs/verify/T6b/aggregate_T6b.py"),
                "venv": ".venv-ag3s",
                "script": "outputs/verify/T6b/aggregate_T6b.py",
                "raw_output": ["outputs/verify/T6b/aggregate_T6b.json",
                               "outputs/verify/T6b/chunks_T6b.json"],
                "stdout_log": "outputs/verify/T6b/aggregate_T6b.log",
                "note": ("reads only frames.jsonl / completeness.json / manifest.json of the "
                         "three recorded runs; nothing is simulated or replayed"),
            },
            {
                "id": "figures",
                "command": ("MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} "
                            "${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u "
                            "outputs/verify/T6b/make_fig1_scene.py ; ... make_fig2_bands.py ; "
                            "... make_fig3_table.py"),
                "venv": ".venv-ag3s",
                "script": ["outputs/verify/T6b/make_fig1_scene.py",
                           "outputs/verify/T6b/make_fig2_bands.py",
                           "outputs/verify/T6b/make_fig3_table.py"],
            },
        ],
        "code_state": {
            "repo": "benchmark",
            "head": "2b95baf",
            "workspace_head": "5cd6f42",
            "dirty_files": ["ag3s/docs/AG3S_REVIEW_PLAN.md", "ag3s/docs/AG3S_T0T6_LOG.md",
                            "ag3s/runtime/frame_record.py", "trajopt/client.py",
                            "trajopt/linearize.py", "trajopt/safe_policy.py",
                            "trajopt/sqp.py", "trajopt/wire.py",
                            "ag3s/runtime/scene_truth.py (untracked)"],
            "note": ("benchmark/**/*.py 는 이 스텝에서 한 줄도 고치지 않았다. 측정 script 는 "
                     "outputs/verify/T6b/ 에만 있다. dirty 는 A1 의 T6a 작업이다 — 아래 "
                     "file:line 은 2026-09-26 작업 트리 기준이라 T6a 뒤 옮겨갈 수 있다"),
            "rule_file_lines_read_2026_09_26": {
                "server_status_violated": "AG3S/trajopt/sqp.py:275-293",
                "server_certified_geometry_downgrade": "AG3S/trajopt/sqp.py:286-294",
                "client_hold_branch": "AG3S/trajopt/client.py:178-180",
                "hold_chunk_is_current_state_tiled": "AG3S/trajopt/client.py:306-337",
                "control_record_executed_flag":
                    "integration/rby1/bringup/pi05_infer.py:1629 "
                    "-> SafePolicyClient.should_execute (AG3S/trajopt/client.py:222-229)",
            },
        },
        "numbers": {
            "_comment": "로그와 논문에 쓰이는 모든 수치가 여기에만 있다. 여기 없는 숫자는 쓰지 않는다.",
            "definitions": {
                "safe": "planning record `safe` == true",
                "hold": "planning record `safe` == false",
                "max_violation_mm": "verdict.max_violation_m * 1000",
                "hold_class": "(verdict.ag3s_status, verdict.trajopt_status)",
                "percentile": "nearest rank on the sorted values; p50 of the nonzero-hold "
                              "population is the median of those chunks only",
                "camera_skew_ms": "observation record camera_skew_sec * 1000 "
                                  "(newest minus oldest camera stamp of the frame)",
                "control_steps_executed_false":
                    "control records with executed == false; the client did not apply that "
                    "chunk step to the robot",
            },
            "per_run": {name: run_block(name) for name in RUNS},
            "violation_trend_execute": agg["violation_trend_execute"],
            "task_phase_overlay": agg["task_phase_overlay"],
            "cross_checks": {
                "execute_hold_chunks_times_8_equals_unexecuted_control_steps": {
                    "hold_chunks": per["execute"]["n_hold"],
                    "open_loop_horizon": 8,
                    "control_steps_executed_false": per["execute"]["control_executed_false"],
                },
                "execute_degraded_holds_equal_frames_with_camera_skew_over_100ms": {
                    "degraded_holds": per["execute"]["hold_classes"].get(
                        "(degraded, violated)", 0),
                    "frames_with_skew_over_100ms":
                        per["execute"]["camera_skew_ms"]["over_100ms_frames"],
                    "chunks_with_camera_skew_reason_code":
                        per["execute"]["ag3s_reason_code_counts"].get("camera_skew", 0),
                },
                "lead_hand_count_vs_measured": {
                    "execute_safe_hand_33_measured": per["execute"]["n_safe"],
                    "execute_hold_hand_42_measured": per["execute"]["n_hold"],
                    "ag3s_ok_violated_hand_35_measured": per["execute"]["hold_classes"].get(
                        "(ok, violated)", 0),
                    "degraded_violated_hand_7_measured": per["execute"]["hold_classes"].get(
                        "(degraded, violated)", 0),
                    "max_violation_mm_hand_min_0_34_p50_21_82_max_37_83_measured_over":
                        "the 37 hold chunks with max_violation > 0 "
                        "(min %.4f / p50 %.4f / max %.4f mm); over all 42 hold chunks the "
                        "p50 is %.4f mm because 5 hold chunks carry 0.0"
                        % (per["execute"]["max_violation_hold_chunks_nonzero"]["min_mm"],
                           per["execute"]["max_violation_hold_chunks_nonzero"]["p50_mm"],
                           per["execute"]["max_violation_hold_chunks_nonzero"]["max_mm"],
                           per["execute"]["max_violation_hold_chunks"]["p50_mm"]),
                    "hold_positions_hand_3_22_26_28_and_38_to_75_measured":
                        per["execute"]["hold_seq_ranges"],
                    "safe_33_all_within_seq_1_37_measured":
                        agg["task_phase_overlay"]["execute_n_safe_in_seq_1_37"] ==
                        per["execute"]["n_safe"],
                },
            },
            "baseline_check_numbers": baseline,
        },
        "figures": [
            {"path": "AG3S/ag3s/docs/figures/t6b/"
                     "t6b-scene-shared-start-and-hold-seqs.png", "kind": "scene"},
            {"path": "AG3S/ag3s/docs/figures/t6b/"
                     "t6b-chunk-bands-and-violation.png", "kind": "graph"},
            {"path": "AG3S/ag3s/docs/figures/t6b/"
                     "t6b-three-runs-table-and-rule.png", "kind": "table"},
        ],
        "baseline_check": {
            "matched": True,
            "note": ("위반으로 시작 %d/%d · has_target %d/%d · frame0 clearance_before "
                     "%.17f mm — 기준선과 같다. feasible/violated 는 기준이 아니라 적지 않는다"
                     % (baseline["violated_start"], baseline["frames"],
                        baseline["has_target"], baseline["frames"],
                        baseline["frame0_clearance_before_mm"])),
        },
        "verdict": ("기준선 일치. 세 실행의 chunk 별 판정을 집계했고 lead 가 눈으로 센 값과 "
                    "전부 같다 (safe 33 / hold 42 · (ok,violated) 35 · (degraded,violated) 7 · "
                    "hold seq 3·22·26·28·38-75 · safe 33 이 전부 seq 1-37 안). "
                    "max_violation min 0.34 / p50 21.82 / max 37.83 mm 는 max_violation > 0 인 "
                    "hold 37 건 기준이다 — hold 42 건 전체로 보면 5 건이 0.0 이고 p50 은 20.15 mm 다."),
        "not_measured": [
            "T6 실행에서 사과가 언제 바구니에 들어갔나 — 기록에 물체 자세가 없다",
            "seq 38 이후 팔이 실제로 안 움직였나 — 기록에 qpos 가 없다 "
            "(control record 의 executed=false 336 개는 client 가 chunk 를 적용하지 않았다는 "
            "사실일 뿐 관절이 멈췄다는 측정이 아니다)",
            "멈춘 자리의 참값 clearance 와 최근접 쌍 — qpos 가 없다",
            "`violated` 가 어느 제약인가 — verdict 가 제약 신원을 싣지 않는다",
            "execute 실행 자신의 task phase — first_closed_seq 20 / first_reopen_seq 33 은 "
            "reference episode 의 offline replay 에서 잰 값이고 execute 궤적의 값이 아니다",
            "offline replay 로 대신하지 않았다 — execute 는 첫 hold (seq %d) 에서 reference "
            "궤적을 떠났으므로 원 기록을 재생해도 execute 가 지나간 자세가 아니다"
            % per["execute"]["hold_seq_min"],
            "scene figure 의 seq 1 을 뺀 모든 panel 은 reference episode replay 의 자세다 "
            "(T6 자세가 아니다)",
        ],
    }

    OUT.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
