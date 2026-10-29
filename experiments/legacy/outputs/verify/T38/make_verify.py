"""T38 Phase A — write AG3S/ag3s/docs/handoff/T38.profile.verify.json from the reduced summaries (numbers + paths)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import collections, json, pathlib
import numpy as np

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38'))
H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T38.profile.verify.json'))
Q = json.loads((OUT / "summary_E3b_ep1807_r2_q2.json").read_text())
B = json.loads((OUT / "summary_E3b_ep1807_r2.json").read_text())
PY = json.loads((OUT / "pyspy" / "replay_T_pyspy.raw.agg.json").read_text())
MI = {p.stem.replace("micro_kdtree_", ""): json.loads(p.read_text()) for p in (OUT / "micro").glob("micro_kdtree_*.json")}
old = json.loads(H.read_text())


def r(x, nd=1):
    if isinstance(x, dict):
        return {k: r(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [r(v, nd) for v in x]
    if isinstance(x, float):
        return round(x, nd)
    return x


def ms(d):
    return None if d is None else r({k: d[k] for k in ("median", "p90", "mean", "min", "max", "n")})


C = Q["census"]
med = lambda v: float(np.median(v))
cnt = lambda v: dict(sorted(collections.Counter(v).items(), key=lambda kv: str(kv[0])))
T = Q["pass_T"]; S = Q["pass_S"]
stage_keys = ["scene_reconstruction", "support_surface", "attention_lifting", "target_grounding", "esdf",
              "constraint_generation", "constraint_generation_minus_esdf", "sum_top_level_stages"]
F = T["functions_seq2_75"]
fsel = ["AG3S.process_multi_debug", "multiview.fuse_observations", "multiview.process_observation", "reconstruction.reconstruct",
        "reconstruction.backproject", "reconstruction.voxel_downsample", "robot_filter.filter_robot_points",
        "robot_filter.robot_sphere_mask", "attention_lifting.lift", "multiview.fuse", "support_surface.fit_support_surfaces",
        "support_surface.fit_plane_ransac", "target_grounding.ground_target", "target_grounding.grow_region",
        "target_grounding.dbscan", "target_grounding._split_groups", "TargetConfirm.select", "AG3S._depth_cameras_from",
        "AG3S._robot_mask_for", "AG3S._build_esdf", "CuroboFieldBuilder.update", "CuroboFieldBuilder._integrate",
        "CuroboFieldBuilder._build_tiers", "CuroboFieldBuilder._labels_from_sites", "CuroboFieldBuilder._free_tsdf_points",
        "CuroboFieldBuilder._camera_contribution", "AG3S._fine_window_request", "AG3S._esdf_coverage", "AG3S._constraints",
        "to_adapter.build_constraint_set", "ConstraintBuilder.build", "linearize.scene_from_constraint_set",
        "wire.unpack_camera_observations", "SafePolicy._apply_finger_state", "SafePolicy._advance_grasp",
        "SafePolicy._run_latch", "SafePolicy._verdict", "SafePolicy._record", "refiner.refine"]


def cp_rows(t, key, n=15):
    return [{"function": x["function"], "cum_ms_per_chunk": round(x["cumtime_ms"], 1), "self_ms_per_chunk": round(x["tottime_ms"], 1),
             "calls_per_chunk": round(x["ncalls_per_chunk"], 1), "where": x["device"]}
            for x in t[key] if x["device"] != "harness wrapper"][:n]


cprof = {k: {"n_chunks": t["n_profiles"], "self_time_by_where_ms_per_chunk": r(t["self_time_by_device_ms_per_chunk"]),
             "top15_cumulative": cp_rows(t, "top_cumulative"), "top15_self": cp_rows(t, "top_tottime")}
         for k, t in Q["cprofile"].items() if k != "attention_lifting"}

sizes = {}
for key in ("n_points_raw", "n_points_voxel", "n_points_fused", "n_self_filtered", "n_support_points", "n_seeds",
            "esdf_n_active_blocks", "n_points_before_fusion"):
    v = [c["sizes"][key] for c in C if c["sizes"].get(key) is not None]
    sizes[key] = {"median": med(v), "min": min(v), "max": max(v)}
gr = [c["fn_sizes"]["target_grounding.grow_region"][0] for c in C]
db = [c["fn_sizes"]["target_grounding.dbscan"][0] for c in C]
sizes["grow_region_in"] = {"median": med([g["n_in"] for g in gr]), "min": min(g["n_in"] for g in gr), "max": max(g["n_in"] for g in gr)}
sizes["grow_region_out"] = {"median": med([g["n_grown"] for g in gr]), "min": min(g["n_grown"] for g in gr), "max": max(g["n_grown"] for g in gr)}
sizes["dbscan_in"] = {"median": med([d["n_in"] for d in db]), "min": min(d["n_in"] for d in db), "max": max(d["n_in"] for d in db)}
sizes["dbscan_calls_per_chunk"] = cnt([c["fn_calls"].get("target_grounding.dbscan", 0) for c in C])
sizes["grow_region_kd_waves_per_chunk"] = cnt([c["kd_queries_by_site"].get(
    "query_ball_point AG3S/ag3s/stages/target_grounding.py:239 grow_region [target_grounding]", 0) for c in C])
sizes["n_clusters_per_chunk"] = cnt([c["fn_sizes"]["target_grounding.ground_target"][0]["n_clusters"] for c in C])
sizes["kd_builds_per_chunk"] = {"median": med([c["kd_builds"] for c in C]), "min": min(c["kd_builds"] for c in C), "max": max(c["kd_builds"] for c in C)}
sizes["kd_build_points_per_chunk_median"] = med([c["kd_build_points_total"] for c in C])
sizes["kd_queries_per_chunk_median"] = med([sum(c["kd_queries_by_site"].values()) for c in C])
sizes["cameras"] = 3
sizes["depth_px_per_chunk"] = 3 * 640 * 480
sizes["esdf_tiers_per_chunk"] = cnt([1 + c["fn_calls"].get("CuroboFieldBuilder._build_tiers", 0) for c in C])
sizes["esdf_voxels_per_tier"] = 128 ** 3
sizes["esdf_tier_note"] = "coarse 20 mm + fine 5 mm (+ target-free fine 5 mm while not holding: _build_tiers 2 calls); 3 tiers in 62 chunks, 2 in 13 (held)"
sizes["constraint_spec_params"] = 5098
sizes["selected_chunks"] = [{k: v for k, v in x.items() if k not in ("copies",)} for x in
                            json.loads(pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/t38/t38-input-sizes.json')).read_text())["rows"]]

dup = collections.Counter()
for c in C:
    for h, v in c["kd_same_input_built_more_than_once"].items():
        dup[" + ".join(sorted(set(x.split(" [")[0].split(" ")[-1] for x in v)))] += 1
spec = cnt([tuple(c["fn_sizes"]["linearize.scene_from_constraint_set"][0][k] for k in ("spec_present", "candidate_active_any", "plane_active_any")) for c in C])

copies = {"h2d_bytes_per_chunk": {"median": med([c["h2d_bytes"] for c in C]), "min": min(c["h2d_bytes"] for c in C), "max": max(c["h2d_bytes"] for c in C)},
          "d2h_bytes_per_chunk": {"median": med([c["d2h_bytes"] for c in C]), "min": min(c["d2h_bytes"] for c in C), "max": max(c["d2h_bytes"] for c in C)},
          "n_h2d_per_chunk_median": med([c["n_h2d"] for c in C]), "n_d2h_per_chunk_median": med([c["n_d2h"] for c in C]),
          "by_site_seq2": C[1]["copies"], "by_site_seq22": [c for c in C if c["seq"] == 22][0]["copies"],
          "method": "torch.Tensor.cpu/.to/.cuda/.item/.tolist and torch.as_tensor/torch.tensor(device=cuda) wrapped in census pass C; callsite = first benchmark/ frame"}
tp = Q["torchprof"]
gpu_time = {k.replace("_keyavg.json", ""): {"device_self_time_total_ms": round(v["device_self_time_total_ms"], 2),
                                            "kernel_self_time_ms": round(v["kernel_self_time_ms"], 2),
                                            "memcpy": v["memcpy"]} for k, v in tp.items()}

numbers = {
    "_comment": "all ms are per planning chunk. replay = SafePolicy.infer driven with re-rendered E3b_ep1807_r2 requests (75 chunks); "
                "stats over seq 2-75 (seq 1 builds a new cuRobo mapper after reset, reported separately). quiet run = label _q2.",
    "equivalence_to_recording": {"grasp_state_match": f"{T['grasp_state_match']}/75", "n_clusters_match": f"{T['n_clusters_match']}/75",
                                 "n_clusters_mismatch_seq_replay_recorded": T["n_clusters_mismatch"], "has_target": f"{T['has_target']}/75"},
    "stage_ms": {
        "recorded_T34_E3b_all_chunks": {"n_chunks": Q["recorded_all_449"]["n_chunks"],
                                        **{k: ms(v) for k, v in Q["recorded_all_449"]["profile_ms"].items() if k in stage_keys}},
        "recorded_T34_E3b_ep1807_r2_seq2_75": {k: ms(v) for k, v in Q["recorded_this_run"]["seq2_75"].items()},
        "replay_q2_pass_T_no_extra_sync": {k: ms(T["stages_profile_ms_seq2_75"].get(k)) for k in stage_keys},
        "replay_q2_pass_S_cuda_synchronize_at_every_stage_boundary": {k: ms(S["stages_harness_ms_seq2_75"].get(k)) for k in stage_keys},
        "replay_first_run_pass_T": {k: ms(B["pass_T"]["stages_profile_ms_seq2_75"].get(k)) for k in stage_keys},
        "note": "constraint_generation contains esdf (nested profiler stage); constraint_generation_minus_esdf = the rest. "
                "attention_lifting ~0: lifting is done inside scene_reconstruction (multiview.process_observation).",
    },
    "safe_policy_timing_ms_replay_q2_pass_T": {k: ms(v) for k, v in T["timing_seq2_75"].items() if k != "infer"},
    "to_timing_recorded_all_chunks": {k: ms(v) for k, v in Q["recorded_all_449"]["to_timing_ms"].items()},
    "first_chunk_after_reset": {
        "replay_q2_pass_T_seq1": r(T["seq1"]), "recorded_ep1807_r2_seq1": r(Q["recorded_this_run"]["seq1"]),
        "jit_first_call_warm_seq1_new_process": r(Q["warm"][0]),
        "note": "seq 1 of every episode: AG3S.reset drops the cuRobo builder (pipeline.py:368 `_esdf_builder = None`) so the mapper is rebuilt "
                "(esdf ~1.37 s). The first chunk of a new process additionally pays warp/cuRobo JIT (esdf 6.7 s, wall 9.4 s); not quoted as steady state."},
    "function_timers_ms_replay_q2_pass_T": {k: r(F[k]) for k in fsel if k in F},
    "cprofile_per_stage": cprof,
    "pyspy": {"file": "outputs/verify/T38/pyspy/replay_T_pyspy.raw.txt", "aggregate": "outputs/verify/T38/pyspy/replay_T_pyspy.raw.agg.json",
              "mode": "py-spy 0.4.2 record --format raw --rate 50 --nonblocking (all threads, no --idle); pass T only; wall median 3160 ms under sampling",
              "total_samples": PY["total_samples"], "samples_by_stage": PY["by_stage_samples"],
              "top_leaf_lines": {k: v[:6] for k, v in PY["top_leaf_per_stage"].items()},
              "first_attempt": "--idle --rate 200 (blocking) slowed the process >20x (first chunk 251 s); aborted, kept in outputs/verify/T38/pyspy_try1_idle_200hz_aborted/"},
    "input_sizes_census_pass_C_75_chunks": sizes,
    "redundancy": {
        "backproject_calls_per_chunk": cnt([c["fn_calls"].get("reconstruction.backproject") for c in C]),
        "backproject_note": "3 in scene_reconstruction (reconstruction.py:274 via multiview.process_observation) + 3 in constraint_generation "
                            "(pipeline.py:1773 _robot_mask_for) on the same depth/K/T of each camera; census hash of the back-projected points "
                            "identical between voxel_downsample input and the _robot_mask_for robot_sphere_mask input (e.g. seq 2 head 709a80f477aede2f, 139,044 pts)",
        "robot_sphere_mask_calls_per_chunk": cnt([c["fn_calls"].get("robot_filter.robot_sphere_mask") for c in C]),
        "robot_sphere_mask_note": "3 on the voxelised per-camera cloud (scene_reconstruction) + 3 on the full-resolution back-projection (_robot_mask_for); "
                                  "while held (13 chunks) +3 more held-sphere masks on the same full-res points (pipeline.py:1789). Each call builds a "
                                  "cKDTree on all points (robot_filter.py:168) and loops in Python over 587 spheres (587 query_ball_point calls per camera, "
                                  "1,761 per 3 cameras, robot_filter.py:169-172).",
        "kd_same_input_built_more_than_once_chunks_by_sites": dict(dup.most_common()),
        "kd_same_input_note": "number of (chunk, array) pairs over the 75 chunks in which an identical array (blake2b of bytes) was KD-treed more than once, keyed by the sites that built it (robot_sphere_mask 39 = 13 held chunks x 3 cameras). "
                              "_labels_from_sites builds a tree on the same seed points once per tier (coarse + fine); trajopt linearize.py:736 rebuilds a tree "
                              "on manipulated_points on every linearize/check call (9 builds per chunk before attach) from points already KD-treed in grounding "
                              "(admissibility gap / anchor_cover).",
        "depth_upload_note": "each camera depth is uploaded to GPU twice per chunk (curobo_builder.py:815 _integrate, once for the coarse mapper and once for the "
                             "fine mapper): 6 x 1,228,800 B = 7.37 MB",
        "per_camera_python_loops": "multiview.fuse_observations list comprehension over 3 cameras (multiview.py:356); _depth_cameras_from loop over 3 cameras "
                                   "(pipeline.py:1957); robot_sphere_mask python loop over 587 spheres per call; grow_region while-loop over BFS waves "
                                   "(7-22 waves); no per-cluster loop of measurable cost (_split_groups / _score_cluster < 1 ms)",
        "constraint_builder": {
            "ConstraintBuilder.build_ms_median": round(F["ConstraintBuilder.build"]["median_ms"], 3),
            "build_spec": "pipeline.py:1372 build_spec = primitive_path or bool(surfaces); support surfaces present in 75/75 chunks so the spec is built every chunk (5,098 parameters)",
            "reader": "only trajopt/linearize.py:1250 scene_from_constraint_set -> SceneSnapshot.from_spec (linearize.py:197); serve_safe trajopt config sets "
                      "collision.use_support_planes False (serve_safe.py:1335) and backend esdf, so plane_active and candidate_active are zeroed (linearize.py:1363-1374)",
            "scene_from_constraint_set_census_spec_present_candidate_active_any_plane_active_any": {str(k): v for k, v in spec.items()},
            "other_readers_in_server_path": "none found (to_adapter.refresh / summary / visualization are not called by serve_safe/safe_policy)"},
        "recorder_note": "SafePolicy._record (--record-constraints) runs after timing_ms.total: median 245 ms, mostly zipfile write (py-spy); not in profile_ms",
    },
    "host_device_copies_census": r(copies, 0),
    "gpu_device_time_torch_profiler_selected_chunks": gpu_time,
    "gpu_note": "GPU kernel time 5.7-7.4 ms and memcpy 2.7-4.4 ms per chunk (torch.profiler CUPTI, warp kernels included); the esdf stage wall (~250 ms) is CPU-side "
                "(cProfile: _labels_from_sites np.unique/sort + cKDTree, unpack_site_linear, _build_tiers D2H). Stage medians with and without "
                "torch.cuda.synchronize at stage boundaries differ by <= 40 ms (pass S vs pass T).",
    "to_note": "TO linearize median 97 ms: ESDF gradient cache np.gradient + np.stack over each 128^3 tier on first query per chunk (esdf.py:440-442, py-spy top leaves) "
               "and cKDTree on manipulated points (linearize.py:736).",
    "micro_kdtree": r(MI, 2),
    "micro_note": "same inputs, same result (`same_result`), only scipy cKDTree `workers` changed in a copy of grow_region inside outputs/verify/T38/micro_kdtree.py; "
                  "cpu_count 256. quiet = after the foreign probe.py ended; contended = while it ran.",
    "gpu_memory": {"nvidia_smi_used_before_quiet_run_mib": 0, "nvidia_smi_peak_during_quiet_run_mib": 1656,
                   "torch_max_allocated_mib": round(max(Q[f"pass_{p}"]["torch_max_alloc_mib"] for p in "TSPWX"), 1),
                   "torch_max_reserved_mib": round(max(Q[f"pass_{p}"]["torch_max_reserved_mib"] for p in "TSPWX"), 1),
                   "first_run_others_mib": "8,810-10,452 total used while another session's process (pid 981255) held 8,810 MiB",
                   "threshold_80pct_of_140GB_reached": False,
                   "note": "nvidia-smi --query-compute-apps does not list container pids; ours = total used when no other GPU process was present"},
}
runs = [
    {"id": "render", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-o4 AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/src .venv-openpi-live/bin/python outputs/verify/T38/render_requests.py E3b_ep1807_r2 75",
     "raw_output": "outputs/verify/T38/frames/", "stdout_log": "outputs/verify/T38/render_E3b_ep1807_r2.log"},
    {"id": "replay_first", "command": "MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-o4 AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/src AG3S_CLIENT_SCRIPT=${AG3S_PROJECT_ROOT}/integration/rby1/bringup/pi05_infer.py .venv-openpi-live/bin/python outputs/verify/T38/replay_profile.py E3b_ep1807_r2 75 --passes T,S,C,P,W,X --warm 4",
     "raw_output": ["outputs/verify/T38/passT_E3b_ep1807_r2.jsonl", "outputs/verify/T38/passS_E3b_ep1807_r2.jsonl", "outputs/verify/T38/passC_E3b_ep1807_r2.jsonl"],
     "stdout_log": "outputs/verify/T38/replay_E3b_ep1807_r2.log",
     "contamination": "a foreign CPU process (.venv-curobo probe.py, ~1,400 % CPU) ran from pass S seq 69 on; pass S seq 69-75 wall 5.0-10.8 s, pass C 8.7-15 s "
                      "(census also adds its own overhead). P/W/X stopped during P (kill of own pid) and re-run as _q2; partial P kept in outputs/verify/T38/contended/. "
                      "pass T (all 75) ran before the probe started. load samples: outputs/verify/T38/logs/load_samples.txt"},
    {"id": "replay_q2", "command": "same env; replay_profile.py E3b_ep1807_r2 75 --passes T,S,P,W,X --warm 4 --label _q2",
     "raw_output": ["outputs/verify/T38/pass{T,S,P,W,X}_E3b_ep1807_r2_q2.jsonl", "outputs/verify/T38/prof/", "outputs/verify/T38/torchprof/"],
     "stdout_log": "outputs/verify/T38/replay_E3b_ep1807_r2_q2.log", "alone": "no foreign process > 5 % CPU, GPU 0 MiB before (load_pre_q2.txt, logs/nvidia_smi_pre_E3b_ep1807_r2_q2.txt)"},
    {"id": "pyspy", "command": "py-spy record --format raw --rate 50 --nonblocking -o outputs/verify/T38/pyspy/replay_T_pyspy.raw.txt -- <replay_profile.py ... --passes T --label _pyspy>",
     "raw_output": "outputs/verify/T38/pyspy/", "stdout_log": "outputs/verify/T38/pyspy/replay_pyspy.log",
     "tool": "py-spy 0.4.2 wheel unpacked in the session scratchpad (not installed into any venv); attach by pid is denied (ptrace), so it runs as parent"},
    {"id": "micro_kdtree", "command": "PYTHONPATH=${AG3S_PROJECT_ROOT}-o4 .venv-openpi-live/bin/python outputs/verify/T38/micro_kdtree.py <server chunk npz> <label>",
     "raw_output": "outputs/verify/T38/micro/"},
    {"id": "reduce", "command": ".venv-openpi-live/bin/python outputs/verify/T38/analyze.py E3b_ep1807_r2 [_q2]",
     "raw_output": ["outputs/verify/T38/summary_E3b_ep1807_r2.json", "outputs/verify/T38/summary_E3b_ep1807_r2_q2.json"]},
]
out = {
    "_comment": "ag3s-verifier T38 Phase A (2026-10-01): profile of the AG3S server pipeline (serve_safe SafePolicy.infer, E3b flags) on the worktree "
                "${AG3S_PROJECT_ROOT}-o4 (branch o4-gpu-parallel = tag pre-gpu-parallel-20261001). numbers and paths only.",
    "step": "T38-profile", "date": "2026-10-01",
    "code_state": old["code_state"],
    "inputs": {"record": "outputs/verify/T34/E3b_rec/E3b_ep1807_r2 (75 planning chunks; server chunks 75-149)",
               "flags": "outputs/verify/T34/p/start_server_e3.sh (E3b): --esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper "
                        "--target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 "
                        "--max-spheres-per-capsule 32 --record-constraints; client --safe-phase approach --safe-manipulators left",
               "depth": "re-rendered per seq from frames.jsonl planning-row qpos (T32 H4 render_real.py method), wire-quantised uint16 mm",
               "attention": "server-recorded 16x16 maps per camera (float16 in the record)", "policy_chunk": "planning row actions_reference",
               "exec_feedback": "planning row exec_feedback (as sent)", "stamps": "observation row camera_stamps (sim_frozen)",
               "selected_chunks_for_profilers": {"2": "approach (searching)", "10": "pre-grasp (latched)", "15": "closing", "16": "attach (held)",
                                                 "22": "transport (held, lifted)", "28": "held at place", "40": "released (placed)"},
               "code_differences_vs_T34_server": "T34 E3b ran at benchmark cee8f84 + T33/T34 edits; worktree is b4f06ec (adds T35-T37, K1 continuity gate)"},
    "runs": runs,
    "baseline_check": old["baseline_check"],
    "numbers": numbers,
    "figures": [
        {"path": "AG3S/ag3s/docs/figures/t38/t38-scene.png", "kind": "scene", "sidecar": "AG3S/ag3s/docs/figures/t38/t38-scene.json"},
        {"path": "AG3S/ag3s/docs/figures/t38/t38-stage-bars.png", "kind": "graph", "sidecar": "AG3S/ag3s/docs/figures/t38/t38-stage-bars.json"},
        {"path": "AG3S/ag3s/docs/figures/t38/t38-function-table.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/t38/t38-function-table.json"},
        {"path": "AG3S/ag3s/docs/figures/t38/t38-input-sizes.png", "kind": "table", "sidecar": "AG3S/ag3s/docs/figures/t38/t38-input-sizes.json"},
        {"path": "AG3S/ag3s/docs/figures/t38/t38-pipeline-map.png", "kind": "diagram", "sidecar": "AG3S/ag3s/docs/figures/t38/t38-pipeline-map.json"}],
    "verdict": "기준선 일치 (legacy 14/15 · 15/15 · −29.031048280806342 ; cuRobo 10/15 · 15/15 · −9.171877401271193). "
               "replay 의 grasp state 75/75 · has_target 75/75 · n_clusters 74/75 가 기록과 같다. 해석·판정은 lead 와 사용자가 한다.",
    "not_measured": [
        "pi0.5 inference and attention extraction (stubbed with recorded outputs)",
        "websocket / msgpack transfer between client and server",
        "other 5 E3b runs (only E3b_ep1807_r2 replayed)",
        "per-chunk numbers under the foreign probe.py load beyond pass S seq 69-75 and pass C (kept, flagged)",
        "SQP iteration counts / per-frame clearances (wall-clock budget dependent, not a baseline)"],
}
H.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print("wrote", H)
