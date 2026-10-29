"""Builds AG3S/ag3s/docs/handoff/T44c.verify.json from agg_e2c.json + timing/agg_timing.json + repro_vs_e2b.json + baseline."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, hashlib
D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2c'))
A = json.load(open(D / "agg_e2c.json"))
T = json.load(open(D / "timing" / "agg_timing.json"))
R = json.load(open(D / "repro_vs_e2b.json"))
B = json.load(open(D / "baseline" / "base_curobo.json"))
E2B = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T44E2b.verify.json')))["numbers"]["sets"]
BM = _ag3s_workspace_path('legacy/benchmark')
md5 = lambda p: hashlib.md5(open(f"{BM}/{p}", "rb").read()).hexdigest()  # noqa: E731
head = subprocess.run(["git", "-C", BM, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()

KQ = ("n_frames", "n_queries", "n_decreased", "n_decreased_gt_1mm", "n_frames_decreased", "n_cross_margin", "n_frames_cross_margin",
      "n_off_below_margin", "n_on_below_margin", "n_on_negative", "n_off_negative", "n_on_gt_off", "n_on_lt_min_off_main",
      "n_answer_target_free_coarse", "n_decreased_shared_caused", "n_cross_margin_shared_caused", "n_decreased_near_shared",
      "frac_decreased_shared_caused", "frac_decreased_near_shared", "n_decreased_on_eq_main_1mm",
      "decrease_mm_among_decreased", "per_frame_max_decrease_mm")
KT = ("n_frames", "n_points", "n_on_negative", "n_off_negative", "n_main_negative", "n_frames_on_negative", "n_on_negative_off_nonneg",
      "n_on_negative_height_lt_20mm", "n_on_negative_height_ge_20mm", "n_points_height_lt_20mm", "n_on_negative_in_ball",
      "n_on_negative_depth_ge_10mm", "on_negative_answer_target_free", "on_negative_answer_target_free_coarse",
      "n_on_negative_coarse_free_negative", "n_decreased", "n_on_gt_off", "decrease_mm")


def pick(d, keys):
    return {k: d[k] for k in keys if k in d}


sets = {}
for s in ("demo", "lb", "demo_last2", "demo_last1"):
    so = {}
    for ident in ("gt", "gt_label"):
        g = A["sets"][s][ident]
        o = {"identity_resolved": g["identity_resolved"], "skipped": g["skipped"],
             "blocking_authorized_spheres": {m: pick(g["blocking_q"][m], KQ) for m in ("ball", "bo_d3", "oracle")},
             "blocking_grid_roi": {m: pick(g["blocking_grid_roi"][m], KQ) for m in ("ball", "bo_d3", "oracle")},
             "target_interior": {m: pick(g["target_interior"][m], KT) for m in ("ball", "bo_d3", "oracle")},
             "coarse_tier_seeds": g["coarse_seeds"]}
        if "safety" in g:
            o["safety_distractor_interior"] = g["safety"]
            o["vs_oracle_roi"] = g["vs_oracle"]
            o["gradient_at_target_free_coarse_answers"] = g["gradient"]
        so[ident] = o
    sets[s] = so

P = T["pooled"]
crit = {}
for s in ("demo", "lb"):
    for ident in ("gt", "gt_label"):
        g = A["sets"][s][ident]; e = E2B[s][ident]["criteria"]["bo_d3"]
        cs = g["coarse_seeds"]; vo = g["vs_oracle"]
        crit[f"{s}|{ident}"] = {
            "(1)_fine_tier_loss_reduction_pooled_T44E2b": e["(1)_loss_reduction_pooled"], "(1)_fine_pass_T44E2b": e["(1)_pass"],
            "(1)_coarse_tier_ball_distractor_seeds_lost": cs["ball"]["distractor_seeds_lost_sum"],
            "(1)_coarse_tier_bo_d3_distractor_seeds_lost": cs["bo_d3"]["distractor_seeds_lost_sum"],
            "(1)_coarse_tier_n_frames_ball_lost_distractor": cs["bo_d3"].get("n_frames_ball_lost_distractor"),
            "(1)_coarse_tier_loss_reduction_pooled": cs["bo_d3"].get("loss_reduction_pooled_on_hit_frames"),
            "(2)_fine_recall_median_T44E2b": e["(2)_recall_median"], "(2)_fine_ball_recall_median_T44E2b": e["(2)_ball_recall_median"],
            "(2)_fine_pass_T44E2b": e["(2)_pass"],
            "(2)_coarse_recall_median_bo_d3": cs["bo_d3"]["target_recall_median"], "(2)_coarse_recall_median_ball": cs["ball"]["target_recall_median"],
            "(2)_coarse_pass": bool(cs["bo_d3"]["target_recall_median"] >= cs["ball"]["target_recall_median"] - 0.05),
            "(3)_on_ff_q_bo_d3": vo["bo_d3|on"]["q_ff"], "(3)_on_ff_q_ball": vo["ball|on"]["q_ff"],
            "(3)_on_ff_grid_bo_d3": vo["bo_d3|on"]["grid_ff"], "(3)_on_ff_grid_ball": vo["ball|on"]["grid_ff"],
            "(3)_on_reference": "oracle with coarse on (fine oracle + coarse oracle)",
            "(3)_on_pass": bool(vo["bo_d3|on"]["q_ff"] <= vo["ball|on"]["q_ff"] and vo["bo_d3|on"]["grid_ff"] <= vo["ball|on"]["grid_ff"]),
            "(4)_p90_bo_d3_on_minus_ball_on_ms": P["criterion4_bo_d3_on_vs_ball_on_p90_increase_ms"],
            "(4)_pass": bool(P["criterion4_bo_d3_on_vs_ball_on_p90_increase_ms"] <= 10.0),
            "(4)_p90_bo_d3_on_minus_ball_off_ms": P["criterion4_bo_d3_on_vs_ball_off_p90_increase_ms"]}

ng = sum(1 for _ in [0])
out = {
    "step": "T44c-Dsign2",
    "date": "2026-10-10",
    "runs": [
        {"id": "baseline_curobo", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} XLA_PYTHON_CLIENT_PREALLOCATE=false ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python -m AG3S.trajopt.experiments.esdf_rollout --records run_16d_ep1800 --attention AG3S/ag3s/asset/data/attention_16d_ep1800.npz --frames 15 --voxel 0.020 --esdf-margin 0.05 --cameras all --support-surfaces field --constraint-links arms --esdf-backend curobo --fine-voxel 0.005 --tsdf-voxel 0.005",
         "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T44/E2c/baseline/base_curobo.json", "stdout_log": "outputs/verify/T44/E2c/baseline/base_curobo.log"},
        {"id": "e2c_measure", "command": "outputs/verify/T44/E2c/run_all.sh {0,1} (2 shards concurrent, nice 19, 1 thread each)", "venv": ".venv-openpi-live",
         "script": "outputs/verify/T44/E2c/e2c_measure.py (imports outputs/verify/T44/E2/e2_measure.py, outputs/verify/T44/E2b/e2b_measure.py, AG3S/ag3s/experiments/live/exclusion_e2.py read-only)",
         "raw_output": ["outputs/verify/T44/E2c/raw_demo/", "outputs/verify/T44/E2c/raw_lb/"],
         "stdout_log": ["outputs/verify/T44/E2c/run_shard0.log", "outputs/verify/T44/E2c/run_shard1.log"]},
        {"id": "e2c_timing", "command": "outputs/verify/T44/E2c/run_timing.sh (single process, no other job of this verifier; 2 warm-ups + 10 interleaved reps, identity gt_label)",
         "raw_output": ["outputs/verify/T44/E2c/timing/timing_demo.json", "outputs/verify/T44/E2c/timing/timing_lb.json"],
         "stdout_log": "outputs/verify/T44/E2c/timing/timing.log",
         "concurrent": ["outputs/verify/T44/E2c/timing/concurrent_procs_start.txt", "outputs/verify/T44/E2c/timing/nvidia_smi_start.txt",
                        "outputs/verify/T44/E2c/timing/concurrent_procs_end.txt", "outputs/verify/T44/E2c/timing/nvidia_smi_end.txt"]},
        {"id": "aggregate", "command": "agg_e2c.py ; timing/agg_timing.py ; repro_vs_e2b.py ; fig_e2c.py ; build_verify.py",
         "raw_output": ["outputs/verify/T44/E2c/agg_e2c.json", "outputs/verify/T44/E2c/timing/agg_timing.json", "outputs/verify/T44/E2c/repro_vs_e2b.json"]},
        {"id": "smoke", "command": "e2c_measure.py --only demo_ep0000_f068", "raw_output": "outputs/verify/T44/E2c/smoke/", "stdout_log": "outputs/verify/T44/E2c/smoke.log"}],
    "code_state": {"repo": "benchmark", "head": head,
                   "dirty_files_relevant": ["ag3s/fields/curobo_builder.py", "ag3s/fields/curobo_field.py", "ag3s/config.py", "ag3s/runtime/pipeline.py",
                                            "ag3s/experiments/sources/constraint_record.py", "trajopt/serve_safe.py", "ag3s/experiments/live/exclusion_e2.py (untracked)"],
                   "md5": {p: md5(p) for p in ("ag3s/fields/curobo_builder.py", "ag3s/fields/curobo_field.py", "ag3s/config.py",
                                               "ag3s/runtime/pipeline.py", "ag3s/experiments/live/exclusion_e2.py")}},
    "split": {"E2-demo": "train, success true; demonstration poses (same 64 frames / 16 episodes as T44E2, T44E2b; outputs/verify/T44/E2/e2_demo_posed_frames.json)",
              "E2-lb": "train; LB E0 success runs (same 48 frames as T44E2, T44E2b; outputs/verify/T44/p/e2_frames.json, history 2)",
              "demo_last1": "demo frame == grasp_close_start_frame (4th of 4 per episode), 16 frames",
              "demo_last2": "3rd + 4th demo frame per episode, 32 frames"},
    "numbers": {
        "baseline_curobo": {"violated_start": sum(f["clearance_before_mm"] < 0 for f in B["frames"]), "frames": len(B["frames"]),
                            "has_target": sum(f["target"] is not None for f in B["frames"]),
                            "frame0_clearance_before_mm": B["frames"][0]["clearance_before_mm"]},
        "settings": {"margin_m": 0.010, "owned_max_dist_voxels": 3.0, "owned_tie_voxels": 1.0, "unowned": "remove", "coarse_voxel_m": 0.02,
                     "fine_voxel_m": 0.005, "coarse_truncation_m": 0.06, "fine_truncation_m": 0.03,
                     "coarse_params_ball_owned_d3": {"band_m": 0.018, "max_dist_m_effective": 0.018, "tie_m": 0.005, "sign_threshold_m": 0.0075,
                                                     "z_min": "support z + 0.020 m"},
                     "fd_step_m": 0.001, "query_model": "V11AB (gripper links, finger cover both); authorized = DEFAULT_CONTACT_LINKS[arm]"},
        "reproduction_off_vs_T44E2b": R,
        "sets": sets,
        "criteria_1_to_4_bo_d3_vs_ball_coarse_on": crit,
        "build_ms_dedicated_timing_pass": T,
        "gradient_layer_identity_mismatch": {"source": "outputs/verify/T44/E2c/grad_mismatch.json", "n_points_checked_all_sets": 2517905, "n_mismatch_gt_1e-6": 6,
                                             "note": "all 6 at value |v| <= 1.2e-9 m (seed voxel); bo_d3 2, oracle 4; ball 0"},
    },
    "figures": [{"path": "AG3S/ag3s/docs/figures/t44/t44c-scene.png", "kind": "scene"},
                {"path": "AG3S/ag3s/docs/figures/t44/t44c-graph.png", "kind": "graph"},
                {"path": "AG3S/ag3s/docs/figures/t44/t44c-table.png", "kind": "table"}],
    "baseline_check": {"matched": True, "note": "cuRobo 기준선: 위반으로 시작 10/15 · has_target 15/15 · frame0 clearance_before -9.171877401271193 mm — 기준값과 같다"},
    "verdict": {
        "baseline": "기준선 일치.",
        "reproduction": "coarse off 의 target-free 값이 T44E2b raw 와 같은 frame 에서: distractor 내부 점 max |diff| 0.0 m (180,970 점 × 방식 3), 권한 구 / grid 는 max |diff| 5.07e-7 m (sign 차이 q 2 · grid 6–8 / 3.78M). safety off 수 (3895 점 · 54 frame, demo gt_label ball) 가 T44E2b 와 같다.",
        "facts": [
            "판정 대상 숫자는 numbers.sets.<set>.<identity> 와 numbers.criteria_1_to_4_bo_d3_vs_ball_coarse_on.",
            "timing pass 는 verifier S1 (serve_safe 2 · pi05_infer client 2) 과 겹쳐 돌았다 — pod_load 는 numbers.build_ms_dedicated_timing_pass.pooled.pod_load."]},
    "not_measured": ["closed-loop effect", "pipeline identity with real VLA attention (GT Gaussian attention)",
                     "safe_policy verdict class for target_free_coarse rows (_NOT_TARGET_TIERS, impl.md 판정 1)",
                     "target exterior shell points (0-10 mm outside the target surface) as a separate set — only authorized spheres and grid offsets",
                     "build ms on an idle pod (S1 overlap)"],
    "deviations": [
        "oracle coarse tier is verifier-defined (harness has none): GT-target-labelled coarse seeds (voxel majority >= 80 %, harness voxel_labels with hulls) removed within ball + 20 mm, z unbounded; sign set = harness make_oracle rule at 20 mm voxel (removed seeds | non-seed with d_t < d_o - 20 mm); no shared-voxel preservation. Its counts dict has no n_sign_candidates (reported 0).",
        "'_ns' coarse tiers (shared preservation disabled) built by passing no non-target points to the coarse spec; ball_owned classification still uses its OwnedEvidence non-target points.",
        "AG3S config owned_max_dist_voxels 3.0 (T44E2b used 2.0 in config; fine tiers take d from OwnedEvidence explicitly, so the fine values are the same — reproduction above).",
        "gtNT / gtT (GT nearest-surface distances of query centres) are unsigned to points (target signed by hull); the gtNT-based underestimate counts in agg_e2c.json also fire for OFF (e.g. demo gt_label q 1975/8640) and are not used in this verify.json.",
        "gradient check: <= 4000 sampled points per frame × identity × method among points answered by target_free_coarse; FD of target_free_distance (h = 1 mm) vs target_free_gradient; reference = same measure on the coarse target-free layer alone and on the main coarse layer at the same points."],
}
json.dump(out, open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T44c.verify.json'), "w"), indent=1, ensure_ascii=False)
print("written")
