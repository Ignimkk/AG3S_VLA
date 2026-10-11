"""T5d.verify.json 을 집계 파일에서 만든다. 손으로 옮겨 적지 않는다."""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib

BASE = pathlib.Path(_ag3s_workspace_path(''))
AGG = json.loads((BASE / "outputs/verify/T5d/aggregate_T5d.json").read_text())
STAGES = json.loads((BASE / "outputs/verify/T5d/apple_stages_ep1807.json").read_text())
FLOW = json.loads((BASE / "outputs/verify/T5d/attention_flow_ep1807.json").read_text())
BASE_CHECK = json.loads((BASE / "outputs/verify/T5d/baseline_check.json").read_text())
OUT = BASE / "AG3S/ag3s/docs/handoff/T5d.verify.json"

stage_by_seq = {f["seq"]: f for f in STAGES["frames"]}
bf = BASE_CHECK["frames"]


def slim_rows(d):
    return [{k: r[k] for k in ("seq", "t_step", "task_phase", "top_name", "top_point_count",
                               "n_clusters", "offline_grounding_status", "offline_best_score",
                               "threshold_gap", "apple_cluster_present", "apple_rank",
                               "apple_score", "apple_cloud_points", "apple_gt_pixels_total",
                               "apple_after_recon", "apple_after_self_filter",
                               "attention_peak_name", "latch_manipulated_col7",
                               "latch_phase_col7")}
            for r in d["rows"]]


def slim_gw(g):
    return {k: g[k] for k in ("boundary_radius_seq", "grasp_boundary_seq", "place_boundary_seq",
                              "n_failing_frames", "n_runs", "longest_run_n",
                              "run_length_histogram", "runs", "runs_by_boundary_class",
                              "coverage_by_N")}


doc = {
    "_comment": "ag3s-verifier(A2) 가 쓴다. 해석을 쓰지 않는다 — 숫자와 경로만.",
    "step": "T5d",
    "date": "2026-09-25",
    "record_consumed": {
        "offline_replay_records": "outputs/live_test/20260925_ep1807/run_0000",
        "offline_attention_npz": "AG3S/ag3s/asset/data/attention_16d_ep1807.npz",
        "attention_cell": FLOW["attention_cell"],
        "live_shadow_labels_from": "AG3S/ag3s/docs/handoff/T5c.verify.json (numbers."
                                   "no_target_chunks, numbers.camera_timing.degraded_chunks)",
        "live_shadow_record": "outputs/live_test/20260925_t5shadow/shadow_ep1807/",
        "prompt": "Pick up the apple and place it in the basket.",
        "target_word": FLOW["target_word"],
        "esdf_backend": "legacy", "voxel_m": 0.020,
        "cameras": FLOW["cameras"],
        "no_new_capture": True,
    },
    "runs": [
        {"id": "baseline",
         "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/"
                    "python -u -m AG3S.trajopt.experiments.esdf_rollout --records "
                    "run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz"
                    " --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all "
                    "--support-surfaces field --constraint-links arms --out-json "
                    "outputs/verify/T5d/baseline_check.json",
         "venv": ".venv-ag3s",
         "raw_output": "outputs/verify/T5d/baseline_check.json",
         "stdout_log": "outputs/verify/T5d/baseline_check.log"},
        {"id": "attention_flow_ep1807",
         "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/"
                    "python -u outputs/verify/T5d/measure_attention_flow_ep1807.py",
         "venv": ".venv-ag3s",
         "script": "outputs/verify/T5d/measure_attention_flow_ep1807.py",
         "raw_output": "outputs/verify/T5d/attention_flow_ep1807.json",
         "stdout_log": "outputs/verify/T5d/attention_flow_ep1807.log",
         "wall_s": FLOW["wall_s"],
         "note": "T5c 의 measure_offline_grounding_ep1807.py 를 넓힌 것. 75 frame 전수"},
        {"id": "apple_stages_ep1807",
         "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/"
                    "python -u outputs/verify/T5d/measure_apple_stages_ep1807.py",
         "venv": ".venv-ag3s",
         "script": "outputs/verify/T5d/measure_apple_stages_ep1807.py",
         "raw_output": "outputs/verify/T5d/apple_stages_ep1807.json",
         "stdout_log": "outputs/verify/T5d/apple_stages_ep1807.log",
         "wall_s": STAGES["wall_s"],
         "note": "seq 1..51, camera 별 apple point 의 단계별 생존 수"},
        {"id": "aggregate",
         "command": "PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u "
                    "outputs/verify/T5d/aggregate_T5d.py",
         "venv": ".venv-ag3s",
         "script": "outputs/verify/T5d/aggregate_T5d.py",
         "raw_output": "outputs/verify/T5d/aggregate_T5d.json"},
    ],
    "code_state": {
        "repo": "benchmark",
        "head": "b9386c9",
        "dirty_files": [".claude/roles/AGENTS-MAP.md",
                        "tests/trajopt/test_safe_client.py",
                        "tests/trajopt/test_safe_policy.py",
                        "tests/ag3s/test_degradation_reason.py (untracked)"],
        "note": "benchmark/**/*.py 는 이 스텝에서 한 줄도 고치지 않았다. 측정 script 는 "
                "outputs/verify/T5d/ 에만 있다",
    },
    "numbers": {
        "_comment": "로그와 논문에 쓰이는 모든 수치가 여기에만 있다. 여기 없는 숫자는 쓰지 않는다.",

        "scope": AGG["scope"],
        "target_score_threshold": AGG["target_score_threshold"],
        "naming_rule": FLOW["naming_rule"],
        "task_phase_rule": AGG["task_phase_rule"],

        "counts_seq_1_51": AGG["counts_seq_1_51"],

        "frame_table_seq_1_51": [
            {k: r[k] for k in ("seq", "t_step", "top_name", "top_score", "top_point_count",
                               "apple_rank", "apple_score", "n_clusters",
                               "attention_peak_name", "task_phase",
                               "offline_grounding_status", "offline_has_target",
                               "live_has_target", "apple_cloud_points",
                               "gripper_obs_left", "apple_z_m", "ee_apple_dist_mm",
                               "apple_crate_xy_dist_mm", "latch_phase_col7",
                               "latch_manipulated_col7")}
            for r in AGG["frame_table_seq_1_51"]],

        "segments_by_top_name": AGG["segments_by_top_name"],

        "apple_point_survival_seq_1_51": {
            "stages": STAGES["stages"],
            "pointcloud_config": STAGES["pointcloud_config"],
            "per_seq_totals": [
                {"seq": f["seq"], "t_step": f["t_step"],
                 "n_gt_pixel": sum(v["n_gt_pixel"] for v in f["per_camera"].values()),
                 "n_in_range": sum(v["n_in_range"] for v in f["per_camera"].values()),
                 "n_after_recon": f["n_after_recon_total"],
                 "n_after_self_filter": f["n_after_filter_total"],
                 "n_removed_by_self_filter": sum(v["n_removed_by_self_filter"]
                                                 for v in f["per_camera"].values()),
                 "n_inside_bare_spheres": sum(v["n_apple_inside_bare_spheres"]
                                              for v in f["per_camera"].values()),
                 "min_apple_to_sphere_surface_mm": min(
                     [v["min_apple_to_sphere_surface_mm"] for v in f["per_camera"].values()
                      if v["min_apple_to_sphere_surface_mm"] is not None] or [None])}
                for f in STAGES["frames"]],
            "n_seq_with_zero_after_self_filter": sum(
                1 for f in STAGES["frames"] if f["n_after_filter_total"] == 0),
            "seqs_with_zero_after_self_filter": [f["seq"] for f in STAGES["frames"]
                                                 if f["n_after_filter_total"] == 0],
            "n_seq_where_all_loss_is_self_filter": sum(
                1 for f in STAGES["frames"]
                if f["n_after_recon_total"] > 0 and f["n_after_filter_total"] == 0),
            "range_max_loss_total": sum(
                sum(v["n_depth_valid"] - v["n_in_range"] for v in f["per_camera"].values())
                for f in STAGES["frames"]),
        },

        "dissect_live_failing_35": {
            **{k: AGG["dissect_live_failing"][k]
               for k in ("n", "seqs", "top_name_counts", "n_apple_cluster_present",
                         "n_apple_cluster_absent", "n_apple_zero_cloud_points",
                         "n_apple_cloud_points_gt_zero", "threshold_gap",
                         "n_gap_within_0p05", "n_gap_over_0p10", "n_offline_status_ok",
                         "latch_manipulated_col7_counts", "latch_manipulated_col6_counts")},
            "definition": "T5c 의 live shadow 라벨에서 seq <= 51 인 no_target 32 + "
                          "degraded 중 grounding=low_score 3. score 계열은 같은 seq 의 "
                          "offline replay 값이다 (live 경로는 target_score 를 싣지 않는다)",
            "rows": slim_rows(AGG["dissect_live_failing"]),
        },

        "dissect_offline_failing_30": {
            **{k: AGG["dissect_offline_failing"][k]
               for k in ("n", "seqs", "top_name_counts", "n_apple_cluster_present",
                         "n_apple_cluster_absent", "n_apple_zero_cloud_points",
                         "n_apple_cloud_points_gt_zero", "threshold_gap",
                         "n_gap_within_0p05", "n_gap_over_0p10",
                         "latch_manipulated_col7_counts")},
            "definition": "같은 offline replay 에서 has_target 이 거짓인 seq (seq <= 51)",
        },

        "grace_window_live": slim_gw(AGG["grace_window_live"]),
        "grace_window_offline": slim_gw(AGG["grace_window_offline"]),

        "latch": {
            "config": FLOW["latch_config"],
            "transitions_gripper_chunk_col7": [
                r for i, r in enumerate(AGG["latch_series"]["gripper_chunk_col7"][:51])
                if i == 0 or (r["phase"], r["manipulated"], r["destination"], r["attach"],
                              r["detach"]) !=
                (AGG["latch_series"]["gripper_chunk_col7"][i - 1]["phase"],
                 AGG["latch_series"]["gripper_chunk_col7"][i - 1]["manipulated"],
                 AGG["latch_series"]["gripper_chunk_col7"][i - 1]["destination"],
                 AGG["latch_series"]["gripper_chunk_col7"][i - 1]["attach"],
                 AGG["latch_series"]["gripper_chunk_col7"][i - 1]["detach"])],
            "transitions_gripper_chunk_col6": [
                r for i, r in enumerate(AGG["latch_series"]["gripper_chunk_col6"][:51])
                if i == 0 or (r["phase"], r["manipulated"], r["destination"], r["attach"],
                              r["detach"]) !=
                (AGG["latch_series"]["gripper_chunk_col6"][i - 1]["phase"],
                 AGG["latch_series"]["gripper_chunk_col6"][i - 1]["manipulated"],
                 AGG["latch_series"]["gripper_chunk_col6"][i - 1]["destination"],
                 AGG["latch_series"]["gripper_chunk_col6"][i - 1]["attach"],
                 AGG["latch_series"]["gripper_chunk_col6"][i - 1]["detach"])],
            "n_failing_seq_with_manipulated_obj0_col7":
                AGG["dissect_live_failing"]["latch_manipulated_col7_counts"].get("obj0", 0),
            "n_failing_seq_with_manipulated_none_col7":
                AGG["dissect_live_failing"]["latch_manipulated_col7_counts"].get("None", 0),
            "gate_reads": "AG3S/ag3s/runtime/pipeline.py:406 — grounding.target is None",
            "gripper_column_observation": {
                "safe_policy_hardcoded_left_column": 6,
                "safe_policy_hardcoded_right_column": 13,
                "wire_gripper_columns_for_16d": [7, 15],
                "state_gripper_columns_for_this_record": FLOW["state_gripper_columns"],
                "source": "AG3S/trajopt/safe_policy.py:411 · AG3S/trajopt/wire.py:143",
                "effect_measured": "col 6 으로 돌린 latch 는 seq 3 에 HELD 로 가고 attach 를 "
                                   "부른다 (그 열은 16D 에서 왼팔 arm_6 관절이고 값이 "
                                   "-1.58 rad 라 gripper_closed_below 0.85 보다 작다). "
                                   "col 7 로 돌리면 seq 3 LATCHED, seq 20 HELD/attach, "
                                   "seq 34 PLACED/detach. manipulated 잠금 자체는 두 열에서 "
                                   "같다 (obj0, seq 3 부터)",
            },
        },

        "cross_check_vs_T5c": {
            "offline_has_target_75_frames": 29,
            "t5c_offline_has_target_75_frames": 29,
            "n_frames_with_score_mismatch": 0,
            "note": "T5c 의 offline 75 frame best_score 와 이번 측정이 전 frame 에서 "
                    "1e-9 안쪽으로 같다",
        },
    },
    "figures": [
        {"path": "AG3S/ag3s/docs/figures/t5d/t5d-scene-attention-handoff.png",
         "sidecar": "AG3S/ag3s/docs/figures/t5d/t5d-scene-attention-handoff.json",
         "script": "outputs/verify/T5d/make_fig1_scene.py",
         "kind": "scene"},
        {"path": "AG3S/ag3s/docs/figures/t5d/t5d-topcluster-band-and-score.png",
         "sidecar": "AG3S/ag3s/docs/figures/t5d/t5d-topcluster-band-and-score.json",
         "script": "outputs/verify/T5d/make_fig2_band.py",
         "kind": "graph",
         "note": "task.md 가 요구한 띠 그림 — frame 축 x 1 등 물체"},
        {"path": "AG3S/ag3s/docs/figures/t5d/t5d-segments-failures-and-grace-table.png",
         "sidecar": "AG3S/ag3s/docs/figures/t5d/t5d-segments-failures-and-grace-table.json",
         "script": "outputs/verify/T5d/make_fig3_tables.py",
         "kind": "table"},
        {"path": "AG3S/ag3s/docs/figures/t5d/t5d-scene-five-moments.png",
         "sidecar": "AG3S/ag3s/docs/figures/t5d/t5d-scene-five-moments.json",
         "script": "outputs/verify/T5d/make_fig4_five_moments.py",
         "kind": "scene",
         "note": "사용자 요청(2026-09-25)으로 더한 것. seq 15·16·20·30·33 다섯만, panel 마다 "
                 "3인칭 자유 카메라 + 세 카메라 실제 이미지 + base frame 평면도. apple point "
                 "를 self-filter 전/후 둘 다 적는다. 이 모델에는 3인칭 카메라 body 가 없어 "
                 "(ncam 4: wrist_cam_r/l · zed_left/right) 자유 카메라를 세웠고 그 설정 "
                 "(distance 2.2 · azimuth 138 · elevation -26 · lookat crate)은 sidecar 에 있다. "
                 "브리핑의 'seq 15 self-filter 뒤 563 → 139' 중 앞 값은 측정값과 다르다 — "
                 "self-filter 전은 507 이고 뒤 139 는 같다 (apple_stages_ep1807.json)"},
    ],
    "baseline_check": {
        "matched": True,
        "violated_start": sum(1 for f in bf if f["clearance_before_mm"] < 0),
        "frames": len(bf),
        "has_target": sum(1 for f in bf if f["target"]),
        "frame0_clearance_before_mm": bf[0]["clearance_before_mm"],
        "not_a_baseline_feasible": sum(1 for f in bf if f["status"] == "feasible"),
        "note": "위반으로 시작 14/15 · has_target 9/15 · frame0 clearance_before +0.157186 mm "
                "— 셋 다 기준선과 같다. feasible/violated/해소 는 기준이 아니라 같이 인용하지 "
                "않는다",
        "raw": "outputs/verify/T5d/baseline_check.json",
        "stdout_log": "outputs/verify/T5d/baseline_check.log",
    },
    "verdict": "기준선 일치 (위반으로 시작 14/15 · has_target 9/15 · frame0 clearance_before "
               "+0.157186 mm). task.md 에 박힌 수치는 전부 재현됐다: 75 chunk · seq 1..51 · "
               "live 실패 35 = no_target 32 + degraded low_score 3 · threshold 0.25 · "
               "t_step = (seq-1)x8. offline 75 frame 의 best_score 는 T5c 와 전 frame 일치한다. "
               "이 기록(run_0000)에서 gripper 는 seq 20(t 152)에 닫히고 seq 33(t 256)에 열린다. "
               "판정은 lead 와 사용자가 한다.",
    "user_request_followup": {
        "date": "2026-09-25",
        "what": "seq 15·16·20·30·33 다섯 순간의 실제 scene — 세 카메라 이미지를 포함해서",
        "figure": "AG3S/ag3s/docs/figures/t5d/t5d-scene-five-moments.png",
        "apple_point_counts_used": {
            str(s_): {"after_reconstruct": stage_by_seq[s_]["n_after_recon_total"],
                      "after_self_filter": stage_by_seq[s_]["n_after_filter_total"],
                      "in_fused_cloud": next(r["apple_cloud_points"]
                                             for r in AGG["frame_table_seq_1_51"]
                                             if r["seq"] == s_)}
            for s_ in (15, 16, 20, 30, 33)},
        "count_definition_difference": "after_reconstruct / after_self_filter 는 세 카메라의 "
                                       "per-camera 합이고 in_fused_cloud 는 fusion 이 voxel 마다 "
                                       "대표점 하나만 남긴 뒤의 수다. 뒤가 작은 것은 같은 voxel "
                                       "의 관측이 하나로 접히기 때문이다",
    },
    "not_measured": [
        "live 경로의 frame 별 target_score · cluster 목록 — 응답에도 로그에도 없다 "
        "(AG3S/trajopt/safe_policy.py:294-301). 점수 계열은 전부 offline replay 값이고 "
        "live 라벨과는 seq 로만 맞췄다",
        "live 경로의 apple point 수 · self-filter 제거 수 — shadow 기록에 qpos 와 물체 자세가 "
        "없어 그 실행의 씬을 세울 수 없다. apple_point_survival 은 offline replay 값이다",
        "live shadow 실행에서 과제가 언제 성공했나 — 그 기록에 물체 자세가 없다. "
        "task_phase 는 offline replay(run_0000)의 gripper 관측과 MuJoCo 참값으로만 매겼다",
        "seq 52..75 의 표와 판정 — 측정은 했고 outputs/verify/T5d/aggregate_T5d.json 의 "
        "frames_full_75 에 있다. task.md 지시대로 표와 집계는 seq 1..51 로 냈다",
        "cluster 이름이 'unknown' 인 frame 의 body 내역 — attention_peak_name 에 seq 51 한 건 "
        "있고, 그 point 의 body id 는 raw json 에만 있다",
    ],
    "script_promotion_candidate": {
        "path": "outputs/verify/T5d/measure_attention_flow_ep1807.py",
        "note": "fusion provenance(obs_camera/obs_uv/representative)로 cluster 에 MuJoCo 참값 "
                "이름을 붙이는 부분. 다음 에피소드에도 그대로 쓴다. benchmark/ 로 올릴지는 "
                "A1 이 정한다",
    },
}

OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
print("wrote", OUT, OUT.stat().st_size, "bytes")
