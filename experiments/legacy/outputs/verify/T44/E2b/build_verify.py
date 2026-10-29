"""Builds AG3S/ag3s/docs/handoff/T44E2b.verify.json from agg_e2b.json + timing/agg_timing.json + baseline."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2b'))
A = json.load(open(D / "agg_e2b.json")); T = json.load(open(D / "timing" / "agg_timing.json"))
B = json.load(open(D / "baseline" / "base_curobo.json"))
SC = {s: json.load(open(D / f"sign_comp_{s}.json")) for s in ("demo", "lb")}

num = {"baseline_curobo": {"violated_start": sum(f["clearance_before_mm"] < 0 for f in B["frames"]), "frames": len(B["frames"]),
                           "has_target": sum(f["target"] is not None for f in B["frames"]),
                           "frame0_clearance_before_mm": B["frames"][0]["clearance_before_mm"]},
       "margin_m": 0.010, "tie_voxels": 1.0, "owned_max_dist_voxels": [2, 3, 4], "unowned": "remove",
       "frame_lists": {"demo": "outputs/verify/T44/E2/e2_demo_posed_frames.json (64 frames, 16 episodes)",
                       "lb": "outputs/verify/T44/p/e2_frames.json fallback (12 episodes, 48 frames, history 2)"},
       "run_settings": A["settings_seen"],
       "sets": {}, "build_ms_dedicated_timing_pass": T}
for s, blk in A["sets"].items():
    num["sets"][s] = {}
    for i, b in blk.items():
        if not isinstance(b, dict):
            continue
        e = {k: b.get(k) for k in ("n_frames", "n_frames_ball_lost_distractor", "identity_resolved_counts", "criteria", "methods")}
        if "reproduction_vs_E2" in b:
            e["reproduction_vs_E2"] = {k: v for k, v in b["reproduction_vs_E2"].items() if k != "mismatches"}
        if "unowned" in b:
            e["unowned"] = b["unowned"]
        if "sign" in b:
            sg = dict(b["sign"]); sg.pop("per_frame", None)
            e["sign"] = sg
        num["sets"][s][i] = e
num["sign_composition_pooled_gt_label"] = SC

out = {
 "step": "T44E2b", "date": "2026-10-10",
 "runs": [
  {"id": "baseline_curobo", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T44/E2b/baseline/base_curobo.json", "stdout_log": "outputs/verify/T44/E2b/baseline/base_curobo.log"},
  {"id": "e2b_measure", "command": "outputs/verify/T44/E2b/run_all.sh {0,1} (2 shards concurrent, nice 19, 2 threads each)", "venv": ".venv-openpi-live",
   "script": "outputs/verify/T44/E2b/e2b_measure.py (imports outputs/verify/T44/E2/e2_measure.py and AG3S/ag3s/experiments/live/exclusion_e2.py read-only)",
   "raw_output": ["outputs/verify/T44/E2b/raw_demo/", "outputs/verify/T44/E2b/raw_lb/"], "stdout_log": ["outputs/verify/T44/E2b/run_shard0.log", "outputs/verify/T44/E2b/run_shard1.log"]},
  {"id": "e2b_timing", "command": "outputs/verify/T44/E2b/run_timing.sh (single process, no other e2b job; 10 interleaved reps after 2 warm-ups, identity gt_label)",
   "raw_output": ["outputs/verify/T44/E2b/timing/timing_demo.json", "outputs/verify/T44/E2b/timing/timing_lb.json"], "stdout_log": "outputs/verify/T44/E2b/timing/timing.log",
   "aborted_attempt_log": "outputs/verify/T44/E2b/timing/timing_aborted_pytest_overlap.log"},
  {"id": "aggregate", "command": "agg_e2b.py ; timing/agg_timing.py ; fig_e2b.py (sign_comp_*.json) ; build_verify.py",
   "raw_output": ["outputs/verify/T44/E2b/agg_e2b.json", "outputs/verify/T44/E2b/timing/agg_timing.json", "outputs/verify/T44/E2b/sign_comp_demo.json", "outputs/verify/T44/E2b/sign_comp_lb.json"]},
  {"id": "smoke", "command": "e2b_measure.py --only demo_ep0000_f068", "raw_output": "outputs/verify/T44/E2b/smoke/"}],
 "code_state": {"repo": "benchmark", "head": "498f713",
   "dirty_files_relevant": ["ag3s/config.py", "ag3s/fields/curobo_builder.py", "ag3s/runtime/pipeline.py", "ag3s/stages/attention_split.py", "ag3s/stages/target_grounding.py", "ag3s/experiments/live/exclusion_e2.py (untracked)"],
   "md5": {"ag3s/experiments/live/exclusion_e2.py": "10cca536686da40b62f71392a7230058", "ag3s/fields/curobo_builder.py": "f090db2e08ca47bd0756e6d483ada3d9",
           "ag3s/runtime/pipeline.py": "47fe851872c42b89089a204292a439df", "ag3s/fields/curobo_field.py": "d1405ca508a11431300822d46b1133cc",
           "ag3s/stages/attention_split.py": "902289bdf8a6dd5187bf8bfb7c451141", "ag3s/stages/target_grounding.py": "7bee43cc5dafee5a6b4c17b8ec34bc9d",
           "ag3s/config.py": "5f6d52b1cead80c526ce517a0dd1f3a4"},
   "third_party/curobo": _ag3s_workspace_path('third_party/curobo (read): seeds = |sdf| <= 0.9 vs or sdf < -(trunc - 1.1 vs) (builder_esdf.py:302-305); sign = TSDF sample (unobserved w <= 0.1 -> +inf -> positive) (builder_esdf.py:413-480, wp_tsdf_sample.py)')},
 "split": {"E2-demo": "train, success true; demonstration poses (same 64 frames as T44E2)", "E2-lb": "train; LB E0 success runs (same 48 frames as T44E2)"},
 "numbers": num,
 "figures": [{"path": "AG3S/ag3s/docs/figures/t44/t44e2b-scene.png", "kind": "scene"},
             {"path": "AG3S/ag3s/docs/figures/t44/t44e2b-graph.png", "kind": "graph"},
             {"path": "AG3S/ag3s/docs/figures/t44/t44e2b-table.png", "kind": "table"}],
 "baseline_check": {"matched": True, "note": "cuRobo 기준선: 위반으로 시작 10/15 · has_target 15/15 · frame0 clearance_before -9.171877401271193 mm — 기준값과 같다"},
 "verdict": {
   "baseline": "기준선 일치. d=2 재현: demo/lb × gt/gt_label 의 frame 별 recall · distractor 손실 · unowned 제거 · 제거 수가 T44E2 raw 와 불일치 0.",
   "criteria": "①–④ 통과/불통과는 numbers.sets.<set>.<identity>.criteria (④ 는 dedicated timing pass 값).",
   "확인된 것": [
     "E2 의 0.27 은 target-free 미세 tier 의 raw voxel 값, 0.80 은 main 합성 질의 (min(coarse, fine)) 값이다 — 같은 frame 에서 같은 정의로 재면 none(제외 없음 재생성) 0.2724 · ball 0.2728 · oracle 0.2724 · main fine layer 0.3184 · coarse 0.7318 · main composite 0.7955 (demo, per-hull median).",
     "(e) only=\"fine\" 재생성은 main fine layer 와 값이 같다: 112 frame ROI box 전체에서 max |diff| 0.0 mm, origin · shape 동일. sign 차이로 잡힌 수는 값 0 voxel 의 보간 결과 (-1e-6 m 이상의 음수) 이다.",
     "(b) 부호 교정이 distractor 내부를 음→양으로 바꾼 수: ball 226 · bo_d2 46 / 65,445 점 (demo); lb 0. seed 제거로 바뀐 부호 0.",
     "(d) distractor 내부 점 중 ball 안 906 / 65,445 (demo), lb 0.",
     "(a) 미세 tier 부호는 fine TSDF (truncation 30 mm) 를 따른다: observed sdf<0 → 음수 또는 0, 양수 0개; unobserved (w<=0.1) → 양수 또는 0, 음수 0개. demo 내부의 fine TSDF unobserved+unallocated 23,807 / 65,445, coarse TSDF (truncation 60 mm) 9,943.",
     "값 0 (seed voxel: |sdf|<=0.9 vs 또는 sdf<-(trunc-1.1 vs)) 이 demo 내부의 27,363 / 65,445 (`<0` 집계에서 음수가 아님).",
     "(c) target-free 질의 min(f, max(d, b)) 는 내부 전 점에서 target_free tier 가 답했다 (window_edge 0); main composite <0 인데 target-free 질의 >=0 인 점 25,493 (그중 coarse <0 25,436).",
     "oracle 도 0.2724 로 ball/owned 와 같다 — 부호 손실은 제외 방식과 무관하다."],
   "반박된 것": [
     "target-free 내부 부호 손실이 ball/ball_owned 의 seed 제외 또는 부호 교정에서 온다 — 제외 없음 재생성 (none) 이 같은 비율이다.",
     "unowned distractor seed 가 fusion_voxel_size (10 mm) 때문이다 — demo d=2 의 unowned distractor seed 422 중 raw depth 화소조차 10 mm 안에 없는 것 398, per-camera 점 10 mm 안 · fused 밖 12."],
   "모르는 것": [
     "fine truncation 을 늘리거나 target-free 질의에 coarse 를 섞을 때의 효과 (재지 않음, 코드 수정 금지).",
     "unowned distractor seed 398 개 (raw depth 10 mm 밖, distractor mesh 중앙 3.9 mm) 를 만든 TSDF 기전 (gather dilation · truncation-boundary seed · 뒷면 중 어느 것인지 분리하지 않음).",
     "closed-loop 에서 finger 구가 distractor 내부에 들어가는 빈도."]},
 "not_measured": ["closed-loop effect", "pipeline identity with real VLA attention (GT Gaussian attention)", "fine truncation / composition variants (no code change)"],
 "deviations": [
   "timing pass first attempt aborted after 1 frame (implementer pytest at ~620 % CPU running); rerun started ~06:43 UTC while a second implementer pytest (~110 % CPU) and an I2b replay ran; pod load recorded in timing/agg_timing.json pod_load",
   "measurement pass build ms (build_ms_once) not used for criterion (4)",
   "sign 'frac negative' keeps the E2 definition (value < 0); zero = |v| <= 1e-6 m reported separately; distractor depth = distance to GT mesh samples (3 mm spacing)",
   "safety query uses the median authorized finger-sphere radius of the frame (6.8 mm) at fine-lattice voxel centres inside the GT distractor convex hull within ball centre +- (r + 60 mm)"]
}
json.dump(out, open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T44E2b.verify.json'), "w"), indent=1, ensure_ascii=False, default=str)
print("ok")
