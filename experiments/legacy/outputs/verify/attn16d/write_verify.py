from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, numpy as np
D = pathlib.Path(_ag3s_workspace_path('outputs/verify/attn16d'))
S = json.loads((D / "score.json").read_text()); SUM = json.loads((D / "summary.json").read_text())
rgb = {}
for n in ["ep1800_headfix", "ep1800_t2", "ep1807", "ep1850", "ep1900"]:
    g = np.load(D / f"gt_{n}.npz"); rgb[n] = {"max_per_cam": g["rgb_mad"].max(0).round(3).tolist(), "mean_per_cam": g["rgb_mad"].mean(0).round(4).tolist()}
surv = json.loads((D / "survey.json").read_text())
def pooled(ph, v):
    p = SUM[ph][v]["pooled"]
    if not p: return None
    keep = lambda c: {k: p[c][k] for k in ("rank", "rank_peak_only_best", "peak_ties", "peak", "peak_frames", "hit_b0", "hit_b0.5", "hit_b1", "lift", "mass", "gap_to_top1_peak", "peak_margin_vs_max_baseline", "failing_betas", "rule_14d_satisfied")}
    return {"n_frames": p["n_frames"], "top1": p["top1"], "top1_metrics": p["top1_metrics"], "top10": p["top10"],
            "n_cells_at_max_peak": p["n_cells_at_max_peak"], "L8h2_d0_last": keep("L8h2_d0_last"), "L7h0_d0_last": keep("L7h0_d0_last"),
            "head_average_d0_last": p["baseline_head_average_d0_last"], "uniform": p["baseline_uniform"], "top1_rule_14d": p["top1_rule"]}
def per_ep(ph, v):
    out = {}
    for e in S["config"]["episodes"]:
        x = SUM[ph][v].get(e)
        if not x: out[e] = None; continue
        out[e] = {"n_frames": x["n_frames"], "top1": x["top1"], "top1_peak": x["top1_metrics"]["peak"], "n_cells_at_max_peak": x["n_cells_at_max_peak"],
                  "L8h2_rank": x["L8h2_d0_last"]["rank"], "L8h2_peak": x["L8h2_d0_last"]["peak"], "L8h2_hit_b0.5": x["L8h2_d0_last"]["hit_b0.5"], "L8h2_lift": x["L8h2_d0_last"]["lift"],
                  "L7h0_rank": x["L7h0_d0_last"]["rank"], "L7h0_peak": x["L7h0_d0_last"]["peak"], "head_average_peak": x["baseline_head_average_d0_last"]["peak"]}
    return out
views = ["cam_high", "cam_left_wrist", "cam_right_wrist", "any"]
numbers = {
  "cells_total": 1296,
  "rank_key": S["config"]["rank_key"],
  "phase_frames": {n: S["episodes"][n]["phase_counts"] for n in S["episodes"]},
  "grasp_onset_index": {n: S["episodes"][n]["onset_index"] for n in S["episodes"]},
  "release_index": {n: S["episodes"][n]["release_index"] for n in S["episodes"]},
  "kept_frames_target_ge_200px": SUM["kept"],
  "reach": {"pooled": {v: pooled("reach", v) for v in views}, "per_episode": {v: per_ep("reach", v) for v in views},
            "stability": {v: SUM["reach"][v]["stability"] for v in views}},
  "carry": {"pooled": {v: pooled("carry", v) for v in views}, "per_episode": {v: per_ep("carry", v) for v in views},
            "stability": {v: SUM["carry"][v]["stability"] for v in views}},
  "crosscheck_coverage_640x480_pooled": SUM["crosscheck_cov640_pooled"],
  "supplement_ep1800_headfix_reach": {v: per for v, per in ((v, SUM["reach"][v].get("ep1800_headfix")) for v in views)} if False else None,
  "reproducibility_vs_asset_attention_16d_headfix": (D / "repro_headfix.txt").read_text().strip(),
  "rgb_replay_mean_abs_diff_0_255": rgb,
  "head_1_qpos_range": {k.split("/live_test/")[1]: v["head_1"] for k, v in surv.items()},
  "reference_14d": {"source": "AG3S/ag3s/docs/archive/step-verification-20260904/step-01-attention.json", "record": "run_0002", "camera": "cam_high",
                    "n_frames": 44, "n_scored": 28, "peak": 1.0, "hit_b0": 0.2857, "hit_b0.5": 0.9286, "hit_b1": 1.0, "lift": 91.82, "head_average_peak": 0.1071, "uniform_peak": 0.0},
}
sc = S["phases"]["reach"]["scopes"].get("ep1800_headfix", {})
numbers["supplement_ep1800_headfix_reach"] = {v: (None if not sc.get(v) else {"n_frames": sc[v]["n_frames"], "top1": sc[v]["top15"][0], "L8h2_d0_last": sc[v]["key_cells"]["L8h2_d0_last"], "L7h0_d0_last": sc[v]["key_cells"]["L7h0_d0_last"]}) for v in views}
bh = json.load(open(D / "baseline_HEAD_openpi_live.json"))["frames"]; bw = json.load(open(D / "baseline_openpi_live.json"))["frames"]
def bnum(fr): return {"frames": len(fr), "violated_start": sum(1 for x in fr if x["clearance_before_mm"] < 0), "has_target": sum(1 for x in fr if x["target"]), "frame0_clearance_before_mm": fr[0]["clearance_before_mm"]}
L = "outputs/live_test"
V = {
 "step": "attn16d", "date": "2026-09-29",
 "runs": [
  {"id": "attention_extract", "command": "outputs/verify/attn16d/extract.sh (python -m AG3S.ag3s.experiments.sources.pi05_attention --config pi05_rby1_randomized_pick_place_16d_lora --noise-seeds 0, default --num-denoise 10 --denoise-steps 0 4 9 --agg mean first last)",
   "venv": ".venv-openpi-live", "env": "OPENPI_DATA_HOME=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid/cache/openpi XLA_FLAGS=--xla_gpu_enable_command_buffer= XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1",
   "raw_output": [f"outputs/verify/attn16d/att_{n}.npz" for n in ["ep1800_t2", "ep1807", "ep1850", "ep1900", "ep1800_headfix"]], "stdout_log": "outputs/verify/attn16d/extract.log"},
  {"id": "ground_truth", "command": "outputs/verify/attn16d/gt.py NAME RECORD (MUJOCO_GL=osmesa)", "venv": ".venv-openpi-live",
   "raw_output": [f"outputs/verify/attn16d/gt_{n}.npz" for n in ["ep1800_t2", "ep1807", "ep1850", "ep1900", "ep1800_headfix"]], "stdout_log": "outputs/verify/attn16d/gt_*.log"},
  {"id": "score", "command": "outputs/verify/attn16d/score.py ; summarize.py ; figs.py", "venv": ".venv-openpi-live",
   "raw_output": ["outputs/verify/attn16d/score.json", "outputs/verify/attn16d/summary.json", "outputs/verify/attn16d/cubes.npz"], "stdout_log": "outputs/verify/attn16d/summarize.log"},
  {"id": "survey", "command": "outputs/verify/attn16d/survey.py", "raw_output": "outputs/verify/attn16d/survey.json", "stdout_log": "outputs/verify/attn16d/survey.log"},
  {"id": "baseline_HEAD", "command": "git archive HEAD (benchmark cee8f84) -> scratch; AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/pi05_TO_hybrid esdf_rollout --records run_16d_ep1800 --attention attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/attn16d/baseline_HEAD_openpi_live.json", "stdout_log": "outputs/verify/attn16d/baseline_HEAD_openpi_live.log"},
  {"id": "baseline_worktree", "command": "same, PYTHONPATH=${AG3S_PROJECT_ROOT} (dirty working tree)", "venv": ".venv-openpi-live",
   "raw_output": "outputs/verify/attn16d/baseline_openpi_live.json", "stdout_log": "outputs/verify/attn16d/baseline_openpi_live.log"},
 ],
 "provenance": {
  "records": {"ep1800": f"{L}/20260925_t2/run_0000/run_0000", "ep1807": f"{L}/20260925_ep1807/run_0000", "ep1850": f"{L}/20260925_t1/ep1850/run_0000", "ep1900": f"{L}/20260925_t1/ep1900/run_0000",
              "ep1800_supplement_same_episode": f"{L}/20260925_headfix16d/records/run_0000/run_0000"},
  "prompts": {n: S["episodes"][n]["prompt"] for n in S["episodes"]},
  "excluded_records": {f"{L}/20260924_long16d/run_0000": "head_1 ~ 2e-5 (before head fix)", f"{L}/20260925_t2b/run_0000/run_0000": "900-step condition, same episode as ep1800 t2",
                       f"{L}/20260925_train/*": "training episodes 0/500/1200", "outputs/verify/T28|T30|T33 *_rec": "no policy images stored (chunk npz / frames.jsonl only)"},
  "checkpoint": _ag3s_workspace_path('legacy/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999'),
  "config": "pi05_rby1_randomized_pick_place_16d_lora", "noise_seed": 0, "num_denoise": 10,
  "segmentation": "policy geometry 299x224 render -> per-body mask BILINEAR resize to 224x224 -> 14x14 patch mean; px rules on 640x480 render (target>=200, competitor>=100)",
  "phase_rule": "reach = before gripper state first < open-0.02; carry = onset .. first frame >= 0.9; after_release not scored",
 },
 "code_state": {"repo": "benchmark", "head": "cee8f84", "used_files_unmodified_vs_HEAD": ["ag3s/experiments/sources/pi05_attention.py", "ag3s/experiments/sources/policy_record.py", "ag3s/experiments/reports/attention_report.py", "ag3s/experiments/common/figstyle.py"],
                "used_file_dirty": ["ag3s/experiments/sources/mujoco_source.py (TransportScene constructor only, via replay_scene)"],
                "dirty_files_count": 28, "workspace_repo_head": "7c16a86", "openpi_head": "df12761", "pi05_TO_hybrid_head": "dc5e660"},
 "numbers": numbers,
 "figures": [{"path": "AG3S/ag3s/docs/figures/r-16d/attention-cell-16d-frames.png", "kind": "scene"},
             {"path": "AG3S/ag3s/docs/figures/r-16d/attention-cell-16d-heatmap.png", "kind": "graph"},
             {"path": "AG3S/ag3s/docs/figures/r-16d/attention-cell-16d-top15.png", "kind": "graph"},
             {"path": "AG3S/ag3s/docs/figures/r-16d/attention-cell-16d-episodes.png", "kind": "graph"},
             {"path": "AG3S/ag3s/docs/figures/r-16d/attention-cell-16d-table.png", "kind": "table"}],
 "baseline_check": {"matched": True, "HEAD_archive_openpi_live": bnum(bh), "working_tree_openpi_live": bnum(bw),
                    "expected": {"violated_start": 13, "has_target": 15, "frame0_clearance_before_mm": 0.18597810857728936},
                    "note": "HEAD archive = expected to all digits. Dirty working tree gives 14/15, frame0 -29.031048280806342, self-filter 587 spheres (HEAD 218). .venv-ag3s interpreter (${AG3S_PROJECT_ROOT}/.python/cpython-3.11) hangs in D state (NFS folio_wait) even for -S -c print(1); baseline run with .venv-openpi-live."},
 "verdict": "baseline(HEAD) 일치 / working tree 어긋남 (13/15 -> 14/15). attention 측정에 쓴 코드 4 파일은 HEAD 와 같다.",
 "not_measured": ["noise seed other than 0", "has_target through lift->ground_target for the alternative cells", "after_release frames", "cam_right_wrist in reach (target < 200 px in every frame)"],
}
out = D / "attn16d.verify.json"
assert not out.exists()
out.write_text(json.dumps(V, indent=1, default=str))
print("wrote", out)
