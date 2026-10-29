"""T44b D-merge: compose ag3s/docs/handoff/T44M.verify.json from the raw outputs (numbers and paths only)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
M = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/merge'))
A = json.load(open(M / "agg_merge.json")); W = json.load(open(M / "wrongpick.json")); D2 = json.load(open(M / "att_2d.json"))
B = json.load(open(M / "baseline/base_curobo.json"))
G = json.load(open(M / "raw_gt/demo_ep0012_f068.json"))
base = {"violated_start": sum(f["clearance_before_mm"] < 0 for f in B["frames"]), "frames": len(B["frames"]),
        "has_target": sum(f["target"] is not None for f in B["frames"]), "frame0_clearance_before_mm": B["frames"][0]["clearance_before_mm"]}
dec = {s: {k: v for k, v in A[s]["decomposition"].items()} for s in A}
cand = {s: {c: {**A[s]["candidates"][c], "wrong_pick": W[s][c]} for c in A[s]["candidates"]} for s in A}
PY = _ag3s_workspace_path('legacy/.venv-openpi-live/bin/python')
v = {
 "step": "T44M", "date": "2026-10-10",
 "runs": [
  {"id": "baseline_curobo", "command": f"MUJOCO_GL=osmesa PYTHONPATH={_ag3s_workspace_path()} XLA_PYTHON_CLIENT_PREALLOCATE=false {PY} -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T44/merge/baseline/base_curobo.json", "stdout_log": "outputs/verify/T44/merge/baseline/base_curobo.log"},
  {"id": "dump_gt", "command": "outputs/verify/T44/merge/run_dump.sh {0,1} 2 gt (dump_frames.py --attention gt; 2 shards, nice 19, 2 threads each)",
   "venv": ".venv-openpi-live", "frames": "outputs/verify/T44/E2/e2_demo_posed_frames.json (64, same as V-E2)",
   "raw_output": "outputs/verify/T44/merge/raw_gt/", "stdout_log": ["outputs/verify/T44/merge/dump_gt_shard0.log", "outputs/verify/T44/merge/dump_gt_shard1.log"]},
  {"id": "vla_attention", "command": "OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 vla_attention.py --raw raw_gt --out vla_attention.npz",
   "venv": ".venv-openpi-live", "model": "pi05_rby1_randomized_pick_place_16d_lora / rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999, return_attn_probs, one model, no server",
   "cell": "denoise step 0, agg last, layer 8, head 2 (trajopt/attention_policy.py constants), noise_seed 0, num_steps 10",
   "inputs": "16D state from posed frame (parquet observation.state), episode prompt (randomized_episodes.jsonl), policy images rendered from the posed scene 299x224 -> 224x224 reflections off (rby1_manipulation.data.recording.resize_policy_image)",
   "raw_output": "outputs/verify/T44/merge/vla_attention.npz", "stdout_log": "outputs/verify/T44/merge/vla_attention.log"},
  {"id": "dump_vla", "command": "outputs/verify/T44/merge/run_dump.sh {0,1} 2 vla vla_attention.npz (16x16 maps on all 3 cameras, as serve_safe attention_extractor passes them)",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T44/merge/raw_vla/", "stdout_log": ["outputs/verify/T44/merge/dump_vla_shard0.log", "outputs/verify/T44/merge/dump_vla_shard1.log"]},
  {"id": "analyze", "command": "JAX_PLATFORMS=cpu merge_analyze.py --raw raw_{gt,vla} --out merge_{gt,vla}.json ; agg_merge.py merge_gt.json merge_vla.json --out agg_merge.json ; wrongpick.py merge_gt.json merge_vla.json ; att_2d.py --out att_2d.json",
   "raw_output": ["outputs/verify/T44/merge/merge_gt.json", "outputs/verify/T44/merge/merge_vla.json", "outputs/verify/T44/merge/agg_merge.json",
                  "outputs/verify/T44/merge/wrongpick.json", "outputs/verify/T44/merge/att_2d.json"],
   "stdout_log": ["outputs/verify/T44/merge/agg_merge.log", "outputs/verify/T44/merge/wrongpick.log"]},
  {"id": "figures", "command": "fig_merge.py", "script": "outputs/verify/T44/merge/fig_merge.py"}],
 "code_state": {"repo": "benchmark", "head": "498f713",
   "dirty_files": ["ag3s/config.py", "ag3s/constraints/to_adapter.py", "ag3s/experiments/sources/constraint_record.py", "ag3s/fields/curobo_builder.py",
                   "ag3s/runtime/pipeline.py", "trajopt/config.py", "trajopt/linearize.py", "trajopt/serve_safe.py", "trajopt/sweep.py",
                   "ag3s/experiments/live/exclusion_e2.py (untracked)"],
   "md5": {"ag3s/stages/target_grounding.py": "1a65dc23641dfd54de6f31e8b7e98a63", "ag3s/stages/attention_split.py": "8636219b9fac3470b00f4c692e31da05",
           "ag3s/experiments/live/exclusion_e2.py": "10cca536686da40b62f71392a7230058", "ag3s/runtime/pipeline.py": "47fe851872c42b89089a204292a439df",
           "ag3s/fields/curobo_builder.py": "f090db2e08ca47bd0756e6d483ada3d9"}},
 "split": "E2-demo: train (episode_index < 1600), success, demonstration poses (16 episodes x 4 frames, pregrasp..grasp_close), no policy execution",
 "numbers": {
  "baseline_curobo": base,
  "run_config": {k: G["config"][k] for k in G["config"]},
  "definitions": {
   "parent": "grounding cluster holding the most GT-target non-support cloud points (pre-split union when a split was taken)",
   "merged": "V-E2 merge record (identities.gt.merge): target fraction < 0.8 and other-fruit points present; 34/64 frames; same set reproduced from the new dump (n_agree 64/64)",
   "cloud_gap": "min distance parent target points <-> parent other-fruit points (fused cloud, fusion voxel 10 mm)",
   "mesh_gap": "min surface distance target <-> that fruit (GT meshes, 2 mm sampling)",
   "pure_and_complete": "chosen segment precision >= 0.95 and recall >= 0.90 vs GT target points of the parent",
   "att_selection": "component with the largest normalised-attention mass (split_by_attention rule); gate = mean(top)/max mean(others) >= 2.0 where listed",
   "mean_raw_selection": "component with the largest mean raw attention",
   "oracle_selection": "component with the most GT-target points (segmentation bound)",
   "wrong_pick": "split taken and chosen segment precision < 0.5"},
  "decomposition": dec,
  "candidates": cand,
  "image_attention_target_vs_neighbour": D2["summary"]},
 "figures": [{"path": "AG3S/ag3s/docs/figures/t44/t44m-scene.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/t44/t44m-scene.json"},
             {"path": "AG3S/ag3s/docs/figures/t44/t44m-graph.png", "kind": "graph", "sidecar": "AG3S/ag3s/docs/figures/t44/t44m-graph.json"},
             {"path": "AG3S/ag3s/docs/figures/t44/t44m-table.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/t44/t44m-table.json"}],
 "baseline_check": {"matched": base["violated_start"] == 10 and base["has_target"] == 15 and base["frame0_clearance_before_mm"] == -9.171877401271193,
                    "note": f"cuRobo 기준선: 위반으로 시작 {base['violated_start']}/15 · has_target {base['has_target']}/15 · frame0 clearance_before {base['frame0_clearance_before_mm']} mm"},
 "verdict": "기준선 일치. 원인 분해와 후보 시험값은 numbers.decomposition / numbers.candidates 에 숫자로 있다.",
 "not_measured": ["closed-loop (demonstration poses only; VLA attention from offline renders of posed scenes, not from a policy rollout)",
                  "colour-region candidate on real camera colour / same-colour fruits (simulator render colour only)",
                  "candidate latency (offline numpy / networkx, not the pipeline path)",
                  "E2-lb 48 frames (not merged in V-E2)"],
 "deviations": ["task text says fusion voxel 12 mm (rby1_three_camera.yaml); the build_ag3s run uses timing.fusion_voxel_size 0.01 (dataclass default), recorded in numbers.run_config",
                "task text says ground_target uses connectivity at fusion voxel; the run uses clustering.method dbscan eps 0.03 min_points 20 (grow_region radius = eps)",
                "the verifier read T44I0I1.impl.md 'expected results' section of the I1 work earlier in this session (whole-file print); it concerns ball_owned, not the merge",
                "per-point GT body label = nearest GT depth pixel within 10 mm (V-E2 used class label within 3 mm); merged set agrees 64/64"]}
out = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T44M.verify.json'))
out.write_text(json.dumps(v, ensure_ascii=False, indent=1))
print(out, v["baseline_check"])
