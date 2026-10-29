"""T39 — build AG3S/ag3s/docs/handoff/T39.verify.json from the partial record + analysis (numbers and paths only).
usage: build_verify.py ANALYSIS_JSON FIG_DIR [BASELINE_JSON]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, subprocess, datetime
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T39'))
H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
A = json.loads(pathlib.Path(sys.argv[1]).read_text())
FIG = pathlib.Path(sys.argv[2])
BASE = json.loads(pathlib.Path(sys.argv[3]).read_text()) if len(sys.argv) > 3 and pathlib.Path(sys.argv[3]).exists() else None
PART = json.loads((H / "T39.verify.partial.json").read_text())
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()

S = A["condition_summary"]
num = {"n_pairs": len(A["pairs"]), "paired_2x2": A["paired_2x2"]}
for c in ("E0", "E3b"):
    s = S[c]
    num[c] = {k: s[k] for k in ("n_runs", "n_success", "n_grasp", "n_place", "n_first_grasp_held", "n_hold_chunks_total", "n_runs_with_hold",
                                 "hold_kind_counts_total", "gate_action_counts_total", "verdict_reason_kind_counts_total", "ipc_counts_total",
                                 "n_round_trip_gt_chunk_period", "n_round_trip", "n_server_segment_paired", "n_server_refined_mismatch")}
    num[c]["contact_mid_z_mm"] = s["contact_mid_z_mm_stat"]
    num[c]["hold_chunks_per_run"] = s["hold_chunks_per_run"]
    num[c]["time_ms_excl_first"] = s["time_ms_pooled_excl_first"]
    num[c]["esdf_ms_excl_first"] = s["esdf_ms_pooled_excl_first"]
    num[c]["first_chunk_round_trip_ms"] = s["first_chunk_round_trip_ms"]
    num[c]["wall_s_per_run"] = s["wall_s"]
    num[c]["time_ms_excl_first_and_work_dd_window"] = s["time_ms_pooled_excl_first_and_mark_window"]
    num[c]["n_chunks_in_work_dd_window"] = s["n_chunks_in_mark_window"]
    num[c]["runs_with_chunks_in_work_dd_window"] = s["runs_with_chunks_in_mark_window"]
    num[c]["n_hold_abort_rows"] = s["n_hold_abort_rows"]
num["matrix"] = A["matrix"]
num["discordant_pairs"] = []
for kind, ps in A["discordant"].items():
    for p in ps:
        dv = p["divergence"]; ch = dv.get("E3b_chunk") or {}
        num["discordant_pairs"].append({"kind": kind, "episode": p["episode"], "seed": p["key"],
                                        "E0": p["E0"], "E3b": p["E3b"],
                                        "first_step_exec_diff_gt0": dv.get("first_step_gt0"), "first_step_exec_diff_gt_1e-3": dv.get("first_step_gt_tol"),
                                        "first_chunk_seq": dv.get("chunk_seq"), "first_chunk_t": dv.get("chunk_t"),
                                        "E3b_gate_action": ch.get("gate_action"), "E3b_hold_kinds": ch.get("hold_kinds"), "E3b_hold_reason": ch.get("hold_reason"),
                                        "E3b_max_violation_m": ch.get("max_violation_m"), "E3b_worst": ch.get("worst"),
                                        "E3b_verdict_reason_kinds": ch.get("verdict_reason_kinds"), "E3b_refined_minus_reference_rad": ch.get("refined_minus_reference_rad_rows0_7"),
                                        "E3b_manipulated": ch.get("manipulated"),
                                        "E0_predicted_minus_E3b_reference_rad": dv.get("E0_predicted_minus_E3b_reference_rad_rows0_7"),
                                        "chunk1_E0_predicted_minus_E3b_reference_max_abs": dv.get("chunk1_E0_predicted_minus_E3b_reference_max_abs"),
                                        "by_threshold": dv.get("by_threshold"), "E3b_first_hold_chunk_seq": dv.get("E3b_first_hold_chunk_seq")})
num["divergence_all_pairs"] = [{"episode": p["episode"], "seed": p["key"], "first_step_gt0": p["divergence"].get("first_step_gt0"),
                                "first_step_gt_1e-3": p["divergence"].get("first_step_gt_tol"), "chunk_seq": p["divergence"].get("chunk_seq"),
                                "E3b_gate_action": (p["divergence"].get("E3b_chunk") or {}).get("gate_action"),
                                "first_step_gt_1e-2": ((p["divergence"].get("by_threshold") or {}).get("0.01") or {}).get("first_step"),
                                "first_step_gt_1e-1": ((p["divergence"].get("by_threshold") or {}).get("0.1") or {}).get("first_step"),
                                "E3b_first_hold_chunk_seq": p["divergence"].get("E3b_first_hold_chunk_seq"),
                                "chunk1_E0_predicted_minus_E3b_reference_max_abs": p["divergence"].get("chunk1_E0_predicted_minus_E3b_reference_max_abs")}
                               for p in A["pairs"]]
num["s2_1"] = {g: {k: v[k] for k in ("gpu_peak_delta_mib", "run_wall_s", "startup_s")} for g, v in PART["s2_1"]["servers"].items()}
num["s2_1"]["bit_identical"] = {c: v["all"] for c, v in PART["s2_1"]["bit_identical"].items()}
runs = PART["runs"]
num["s2_2_runs"] = {"done": sum(1 for r in runs.values() if r["stage"] == "runs" and r["status"] == "done"),
                    "failed_attempts": sum(r["n_failed_attempts"] for r in runs.values() if r["stage"] == "runs")}
EXTRA = json.loads((T / "s23/run_health.json").read_text())
num["run_health"] = EXTRA
num["gpu_windows"] = json.loads((T / "s23/windows.json").read_text())
if BASE:
    num["regression_baseline_legacy"] = BASE
out = {
    "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
    "step": "T39", "date": datetime.date.today().isoformat(),
    "runs": [
        {"id": "S2-1", "command": "outputs/verify/T39/p/launch_s21.sh (run_chain.sh -> start_server.sh / run_client.sh)", "venv": ".venv-openpi-live",
         "raw_output": "outputs/verify/T39/s21/", "stdout_log": "outputs/verify/T39/s21/launch.out"},
        {"id": "S2-2", "command": "outputs/verify/T39/p/launch_s22.sh 4", "venv": ".venv-openpi-live",
         "raw_output": "outputs/verify/T39/runs/", "stdout_log": "outputs/verify/T39/runs/launch.out"},
        {"id": "S2-3", "command": f"analyze_t39.py {sys.argv[1]} --e0 runs/E0 --e3b runs/E3b --key seed; fig_t39.py", "venv": "shim (CPU) / .venv-openpi-live",
         "raw_output": sys.argv[1], "stdout_log": str(pathlib.Path(sys.argv[1]).with_suffix(".log"))}],
    "code_state": {"snapshot": "${AG3S_PROJECT_ROOT}-o1", "benchmark": {"branch": "o1-eval", "head": sh("git -C ${AG3S_PROJECT_ROOT}/benchmark rev-parse --short o1-eval")},
                   "pi05_TO_hybrid": {"rby1_bringup_commit": "22d8dae"}, "code_md5": "outputs/verify/T39/p/code_md5_s20.txt",
                   "xla_flags": "--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0"},
    "numbers": num,
    "figures": [{"path": str(p.relative_to(_ag3s_workspace_path(''))), "kind": ("scene" if "scene" in p.name else "table" if ("table" in p.name or "2x2" in p.name or "matrix" in p.name) else "graph")}
                for p in sorted(FIG.glob("*.png"))],
    "baseline_check": {"matched": None if not BASE else BASE.get("matched"), "note": None if not BASE else BASE.get("note")},
    "verdict": "facts only",
    "not_measured": ["regression baseline cuRobo variant (only legacy was run, after S2-2, on the snapshot code)",
                     "per-server GPU memory during S2-2 (nvidia-smi pids are not attributable; total only, s23/gpu_samples_merged_s22.txt)"],
}
(H / "T39.verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print(H / "T39.verify.json", len(out["figures"]), "figures")
