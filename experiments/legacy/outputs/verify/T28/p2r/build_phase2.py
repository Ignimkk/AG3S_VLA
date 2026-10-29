"""Merge Phase 2R numbers into AG3S/ag3s/docs/handoff/T28.verify.partial.json under key `phase2`.
usage: build_phase2.py COND [COND ...]   — (re)writes phase2.conditions.<COND> from p2r/<COND>_{rec,old}_summary.json."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, datetime

V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28'))
J = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T28.verify.partial.json'))
EPS = (1807, 1800, 1808)
FLAGS = {
    "E0": {"port": 8201, "server": "--no-safe", "client": "(no --safe-remote)"},
    "E1": {"port": 8202, "server": "--no-perception --no-limits --w-smooth 0 --w-continuity 0 --record-constraints E1_rec/server_constraints",
           "client": "--safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left --safe-gate off"},
    "E2": {"port": 8203, "server": "--no-perception --record-constraints E2_rec/server_constraints",
           "client": "--safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left --safe-gate off"},
}
KEEP = ["episode_index", "success", "grasp", "place", "grasp_t", "grasp_t_control_rows", "place_t", "apple_dz_max_mm", "t_apple_dz_max",
        "apple_dz_final_mm", "apple_final_in_crate_frame", "t_first_close_cmd", "finger_to_apple_mm_at_first_close",
        "min_finger_to_apple_surface_mm", "t_min_finger_to_apple", "min_grip_l_applied", "min_grip_l_measured_open_frac",
        "n_close_cycles_applied_lt_0p5", "hold_control_steps", "n_not_executed_control", "ipc_counts", "comms_ipc_planning_rows",
        "infer_ms", "server_timing_split_ms", "client_refined_minus_reference_max_deg", "n_planning", "n_control", "mujoco", "recorded", "mp4"]


def grid(runs):
    g = {}
    for ep in EPS:
        g[str(ep)] = [runs[k]["success"] for k in sorted(runs) if runs[k]["episode_index"] == ep]
    return {"success_by_episode": g, "n_success": sum(r["success"] for r in runs.values()),
            "n_grasp": sum(r["grasp"] for r in runs.values()), "n_runs": len(runs)}


def compact_server(sr):
    out = {}
    for k, v in sr.items():
        out[k] = {kk: vv for kk, vv in v.items() if kk != "chunks"}
        out[k]["per_chunk_max_abs_deviation_deg"] = [c["max_abs_deviation_deg"] for c in v["chunks"]]
        out[k]["per_chunk_sqp_iterations"] = [c["sqp_iterations"] for c in v["chunks"]]
        out[k]["per_chunk_sqp_returned"] = [c["sqp_returned"] for c in v["chunks"]]
        out[k]["per_chunk_limit_overshoot_refined_position"] = [c["limit_overshoot_refined"]["position"] for c in v["chunks"]]
        out[k]["per_chunk_limit_overshoot_reference_position"] = [c["limit_overshoot_reference"]["position"] for c in v["chunks"]]
        out[k]["per_chunk_initial_limit_overshoot_anchor"] = [(c["sqp_initial_limit_overshoot"] or {}).get("anchor") for c in v["chunks"]]
        out[k]["per_chunk_solve_ms"] = [c["solve_ms"] for c in v["chunks"]]
    return out


D = json.loads(J.read_text())
P2 = D.setdefault("phase2", {})
P2["_comment"] = "Phase 2R (2026-09-29): E0/E1/E2 re-run with third-person video; pooled with the 2026-09-28 unrecorded runs. Numbers and paths only."
P2["protocol"] = {
    "episodes": list(EPS), "runs_per_episode_recorded": 2,
    "client_common": "--model rby1_randomized_pick_place_16d --max-steps 600 --headless --record-frames <run_dir> --record <run_dir>/third_person_front.mp4 --view front, MUJOCO_GL=osmesa",
    "concurrency": "recorded E0/E1/E2 ran CONCURRENTLY: 3 servers (8201/8202/8203) + 3 clients (one sequential 6-run chain per condition) on one H200, launched 2026-09-29T01:27:44Z (p2r/concurrency_launch.txt). Unrecorded 2026-09-28 runs ran one condition at a time.",
    "scripts": ["outputs/verify/T28/p2r/start_server_rec.sh", "p2r/run_client_rec.sh", "p2r/run_condition.sh", "p2r/analyze_p2r.py", "p2r/build_phase2.py", "p2r/fig_p2r.py", "p2r/fig_keyframes.py"],
    "code_md5_at_start": "outputs/verify/T28/p2r/code_md5_at_start.txt (identical to E1/code_md5_at_start.txt of the unrecorded E1 runs)",
    "definitions": "success/grasp/place as P1-1 (analyze_e0.py; place = object_in_crate |x|<0.077,|y|<0.137,-0.052<z<0.075 m crate frame; grasp = apple z >= z0 + 50 mm, planning rows). "
                   "first close cmd = first control row applied_ctrl.gripper[0] < 0.5; min gripper opening = min |qpos[27]|/0.045. HOLD = control rows executed==False. "
                   "comms HOLD = client log 'comms HOLD n' and planning ipc in {timeout,stale,error}. refined-reference = server chunks.jsonl max_abs_deviation_deg "
                   "(8 planned steps x 14 arm joints) + client-side check |actions-actions_reference| over rows 0-7, arm idx 0-6,8-14. infer_ms client = planning timing_ms.policy_infer; "
                   "server split = planning verdict.timing_ms {infer, trajopt, total}; first chunk of each run excluded from stats (JIT/warm-up only in first run of each server).",
}
conds = P2.setdefault("conditions", {})
for cond in sys.argv[1:]:
    ent = {"flags": FLAGS[cond]}
    pooled = {}
    for kind in ("rec", "old"):
        p = V / "p2r" / f"{cond}_{kind}_summary.json"
        if not p.exists():
            continue
        S = json.loads(p.read_text())
        runs = {k: {kk: r.get(kk) for kk in KEEP} | {"run_dir": r["run_dir"], "client_log": r["client_log"]} for k, r in S["runs"].items()}
        e = {"summary_json": str(p), "grid": grid(S["runs"]), "runs": runs}
        if "server_records" in S:
            e["server_records"] = compact_server(S["server_records"])
        ent["recorded" if kind == "rec" else "unrecorded_2026-09-28"] = e
        for k, r in S["runs"].items():
            pooled[f"{kind}:{k}"] = r
    ent["pooled_grid"] = grid(pooled)
    ent["mp4"] = [r["mp4"] for k, r in pooled.items() if r.get("mp4")]
    cl = V / f"{cond}_rec" / "condition.log"
    ent["condition_log"] = str(cl); ent["condition_log_text"] = cl.read_text() if cl.exists() else None
    conds[cond] = ent
P2["mp4_all"] = sorted(m for c in conds.values() for m in c.get("mp4", []))
D["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
J.write_text(json.dumps(D, indent=1, ensure_ascii=False))
print("wrote", J, "phase2 conditions:", list(conds))
