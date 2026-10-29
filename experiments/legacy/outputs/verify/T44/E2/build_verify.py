from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
E2 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2'))
A = json.load(open(E2 / "agg_e2.json")); T = json.load(open(E2 / "timing/agg_timing.json"))
V = json.load(open(E2 / "vi0/vi0_table.json")); D = json.load(open(E2 / "e2_demo_frames.json"))
B = json.load(open(E2 / "baseline/base_curobo.json"))
num = {"baseline_curobo": {"violated_start": sum(f["clearance_before_mm"] < 0 for f in B["frames"]), "frames": len(B["frames"]),
                           "has_target": sum(f["target"] is not None for f in B["frames"]), "frame0_clearance_before_mm": B["frames"][0]["clearance_before_mm"]},
       "e2_demo_list": {"n_episodes": D["n_episodes"], "n_frames": D["n_frames"], "pool_counts": D["pool_counts"],
                        "groups": [{k: g.get(k) for k in ("target_fruit", "used_arm", "bin", "n_candidates", "episode", "nearest_other", "nearest_other_xy_m", "frames")} for g in D["groups"]]},
       "margin_m": 0.010, "margin_source": "V11AB --obstacle-margin 0.010 (--esdf-margin 0)",
       "query_model": "V11AB constraint model: links gripper, capsule_radius_scale 0.05, sphere_spacing 0.4, max_spheres_per_capsule 32, finger cover both (270 spheres); authorized = active-arm ee + 2 fingers (135 spheres)",
       "sets": {}}
keep = ("n_frames", "identity_resolved_counts", "n_frames_ball_lost_distractor", "criteria", "merge", "identity_completeness_median",
        "identity_completeness_min", "nontarget_source_counts", "proximity_frames", "sign_check",
        "ball_owned_remove_distractor_lost_unowned_only_sum", "ball_owned_remove_distractor_lost_sum")
for s, blk in A["sets"].items():
    num["sets"][s] = {}
    for i in ("gt", "gt_label", "gt_cluster_only"):
        a = blk[i]
        num["sets"][s][i] = {k: a.get(k) for k in keep}
        num["sets"][s][i]["methods"] = a.get("methods")
num["build_ms_measurement_pass"] = {"note": "2 concurrent shards; see sets.*.methods.*.build_ms_*", "load": A["load"]}
num["build_ms_dedicated_timing_pass"] = T
num["v_i0"] = {"run": V["run"], "rows": V["rows"], "first_contact_per_class": V["first_contact_per_class"]}
out = {
 "step": "T44E2", "date": "2026-10-10",
 "runs": [
  {"id": "baseline_curobo", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005", "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T44/E2/baseline/base_curobo.json", "stdout_log": "outputs/verify/T44/E2/baseline/base_curobo.log"},
  {"id": "e2_demo_list", "command": "python outputs/verify/T44/E2/select_demo.py", "raw_output": "outputs/verify/T44/E2/e2_demo_frames.json", "posed_frames": "outputs/verify/T44/E2/e2_demo_posed_frames.json"},
  {"id": "e2_measure", "command": "outputs/verify/T44/E2/run_all.sh {0,1} (e2_measure.py --posed-frames e2_demo_posed_frames.json --sign-check ; e2_measure.py --lb-frames outputs/verify/T44/p/e2_frames.json --history 2 --sign-check), 2 shards concurrent, nice 19, OMP 4", "venv": ".venv-openpi-live", "raw_output": ["outputs/verify/T44/E2/raw_demo/", "outputs/verify/T44/E2/raw_lb/"], "stdout_log": ["outputs/verify/T44/E2/run_shard0.log", "outputs/verify/T44/E2/run_shard1.log"], "script": "outputs/verify/T44/E2/e2_measure.py (imports AG3S/ag3s/experiments/live/exclusion_e2.py read-only)"},
  {"id": "e2_timing", "command": "e2_timing.py --posed-frames ... ; e2_timing.py --lb-frames ... --history 2 (single process, 10 interleaved reps)", "raw_output": ["outputs/verify/T44/E2/timing/timing_demo.json", "outputs/verify/T44/E2/timing/timing_lb.json"], "stdout_log": "outputs/verify/T44/E2/timing/timing.log"},
  {"id": "aggregate", "command": "agg_e2.py ; agg_timing.py", "raw_output": ["outputs/verify/T44/E2/agg_e2.json", "outputs/verify/T44/E2/timing/agg_timing.json"]},
  {"id": "v_i0", "command": "JAX_PLATFORMS=cpu replay_contacts.py outputs/verify/T44/E2/vi0 outputs/verify/T43/V11/runs/V11AB/H/V11AB_ep1807_s18071 ; vi0_table.py", "raw_output": ["outputs/verify/T44/E2/vi0/V11AB_H_V11AB_ep1807_s18071.json", "outputs/verify/T44/E2/vi0/vi0_table.json"], "stdout_log": "outputs/verify/T44/E2/vi0/replay.log"}],
 "code_state": {"repo": "benchmark", "head": "498f713",
   "dirty_files": ["ag3s/config.py", "ag3s/constraints/to_adapter.py", "ag3s/experiments/sources/constraint_record.py", "ag3s/fields/curobo_builder.py", "ag3s/runtime/pipeline.py", "trajopt/config.py", "trajopt/linearize.py", "trajopt/serve_safe.py", "trajopt/sweep.py", "ag3s/experiments/live/exclusion_e2.py (untracked)"],
   "md5": {"ag3s/experiments/live/exclusion_e2.py": "10cca536686da40b62f71392a7230058", "ag3s/fields/curobo_builder.py": "f090db2e08ca47bd0756e6d483ada3d9", "ag3s/runtime/pipeline.py": "47fe851872c42b89089a204292a439df"}},
 "split": {"E2-demo": "train (episode_index < 1600 and meta split == train), success true; demonstration poses, no policy execution",
           "E2-lb": "train; LB E0 success runs from outputs/verify/T44/p/e2_frames.json fallback (nearest other fruit 0.109-0.117 m)"},
 "numbers": num,
 "figures": [{"path": "AG3S/ag3s/docs/figures/t44/t44e2-scene.png", "kind": "scene"},
             {"path": "AG3S/ag3s/docs/figures/t44/t44e2-scatter.png", "kind": "graph"},
             {"path": "AG3S/ag3s/docs/figures/t44/t44e2-table.png", "kind": "table"}],
 "baseline_check": {"matched": True, "note": "cuRobo 기준선: 위반으로 시작 10/15 · has_target 15/15 · frame0 clearance_before -9.171877401271193 mm — 기준값과 같다"},
 "verdict": "기준선 일치. 판정 ①–④ 의 통과/불통과는 numbers.sets.*.*.criteria 와 numbers.build_ms_dedicated_timing_pass 에 숫자로 있다.",
 "not_measured": [
   "V-I0 with T44 trace fields: V11AB record is pre-T44 (no target_free_*_ijk npz keys, no summary target_permission/target_exclusion); no server run with the T44 code was recorded. carved-surface-voxel proxy used instead.",
   "closed-loop effect (no policy execution in E2-demo)",
   "pipeline identity with real VLA attention (GT Gaussian attention on cameras that see the target)"],
 "deviations": [
   "verifier read the impl.md section 'expected results' before measuring (whole file printed at once); the list, criteria, margin and query definitions were fixed from task.md before any measurement",
   "build ms in the measurement pass were taken with 2 own shards concurrent on the same GPU; criterion (4) is also reported from a dedicated single-process timing pass",
   "identity 'gt' (harness) falls back to gt_label when no grounded cluster is >= 80 % GT target; 'gt_cluster_only' = gt frames resolved to pipeline/nearest_cluster",
   "identity_completeness reads 0.0 on 12 lb gt frames where the pipeline geometry's point_indices did not index the current cloud (nontarget source 'proximity')"]
}
json.dump(out, open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T44E2.verify.json'), "w"), indent=1, ensure_ascii=False, default=str)
print("ok")
