"""T41c — assemble AG3S/ag3s/docs/handoff/T41c.verify.json (numbers + paths only). usage: build_verify_c.py"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, datetime, subprocess
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T41'))
H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
CMP = json.load(open(V / "c/analysis/compare_c.json"))
AC = json.load(open(V / "c/analysis/analysis_c.json"))["runs"]
BL = json.load(open(V / "c/baseline/judge.json"))
PART = json.load(open(H / "T41c.verify.partial.json"))
DRY = json.load(open(V / "c/dryrun/dryrun_cpu.json"))
IMP = {s: json.load(open(V / f"c/import_{s}.json")) for s in ("server", "client")}
FIG = "AG3S/ag3s/docs/figures/t41"
snap = (V / "c/snapshot.txt").read_text().splitlines()
md5diff = (V / "c/p/code_md5_diff_vs_t41a.txt").read_text().splitlines()
runs = CMP["runs"]
SETS = ("T40", "T40J", "T41a", "T41c")


def short(k):
    r = runs[k]; return f"{r['stage']}_ep{r['episode']}"


outcome, hold, holds, contact, dist, ident = {}, {}, {}, {}, {}, {}
for k, r in runs.items():
    s = short(k)
    for w in SETS:
        f = r.get(w)
        if f is None:
            continue
        outcome.setdefault(s, {})[w] = {x: f[x] for x in ("success", "grasp", "place", "grasp_t", "place_t", "lift_t", "n_control")}
        hold.setdefault(s, {})[w] = {x: f[x] for x in ("n_hold_chunks", "hold_t", "hold_kind_counts", "gate_action_counts", "verdict_reason_kind_counts")}
        holds.setdefault(s, {})[w] = f.get("holds")
        contact.setdefault(s, {})[w] = {"events_physics_replay": f.get("contact_events"),
                                        **{x: f[x] for x in ("obstacle_stop", "obstacle_stop_t", "client_robot_collision", "client_payload_collision",
                                                             "client_contact_steps", "client_min_robot_clearance_mm", "client_contact_pairs")},
                                        "replay_fidelity": f.get("replay")}
        dist.setdefault(s, {})[w] = {g: {p: (None if v is None else {"min_mm": v["min_mm"], "t": v["t"], "n_rows_le_0": v["n_rows_le_0"], "pair": v.get("pair")})
                                         for p, v in d.items()} for g, d in f["min_mm"].items()}
        dist[s][w]["physical_hurdle"] = f["physical_hurdle"]
    if r.get("T41c_vs_T41a_identity") is not None:
        ident[s] = r["T41c_vs_T41a_identity"]

c1982 = {w: {"outcome": outcome["C_ep1982"].get(w), "holds": holds["C_ep1982"].get(w), "contact": contact["C_ep1982"].get(w),
             "min_mm_all": {g: (v["all"] or {}).get("min_mm") for g, v in dist["C_ep1982"][w].items() if isinstance(v, dict)}}
         for w in SETS if w in outcome.get("C_ep1982", {})}

srv = sorted((V / "c/runs/E3b").glob("server_8270_*"))
head = lambda repo: subprocess.run(["git", "-C", f"{_ag3s_workspace_path()}/{repo}", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
replay_log = (V / "c/replay/replay_snapshot.log").read_text().splitlines()
out = {
    "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
    "step": "T41c",
    "date": datetime.date.today().isoformat(),
    "runs": [
        {"id": "E3b runs B 6 / C 5 / D 4 (one server, port 8270)",
         "command": "nohup outputs/verify/T41/c/p/launch_c.sh (run_chain.sh E3b 8270 ... lists/E3b_c.txt; start_server.sh = T41a flags incl. --obstacle-margin 0.010 --no-subtask-gate; run_client.sh = T41a client + --safe-hold-mode measured --viz-avoidance-arrow; --safe-gate reasons --obstacle-stop-distance 0)",
         "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T41/c/runs/E3b/{B,C,D}/E3b_ep*_s*/ + server_8270_*/server_constraints",
         "stdout_log": "outputs/verify/T41/c/runs/E3b/chain_8270.log, outputs/verify/T41/c/runs/launch.out"},
        {"id": "analysis", "command": "outputs/verify/T41/p/analyze_t41.py with T41_ROOT=outputs/verify/T41/c/runs (same script as T41a; t41a client distance code)",
         "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T41/c/analysis/analysis_c.json", "stdout_log": "outputs/verify/T41/c/analysis/analysis_c.log"},
        {"id": "HOLD / contact physics replay", "command": "outputs/verify/T41/c/p/hold_c.py (T41c runs -> hold_c.json; T40 G/D, T40 J, T41a runs -> hold_ref.json)",
         "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T41/c/analysis/{hold_c.json, hold_ref.json}", "stdout_log": "outputs/verify/T41/c/analysis/{hold_c.log, hold_ref.log}"},
        {"id": "comparison", "command": "outputs/verify/T41/c/p/compare_c.py compare_c.json", "venv": ".venv-openpi-live",
         "raw_output": "outputs/verify/T41/c/analysis/compare_c.json", "stdout_log": "outputs/verify/T41/c/analysis/compare_c.log"},
        {"id": "baseline", "command": "outputs/verify/T41/c/baseline/run_baseline.sh", "venv": ".venv-openpi-live",
         "raw_output": "outputs/verify/T41/c/baseline/{legacy,curobo}.json, judge.json", "stdout_log": "outputs/verify/T41/c/baseline/{legacy,curobo}.log"},
        {"id": "T40 H replay on the snapshot (CPU)", "command": "outputs/verify/T41/c/p/replay_snap_t41c.py (= T40/H/replay.py via T40/J/p/replay_snap.py, HY=${AG3S_PROJECT_ROOT}-t41c)",
         "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T41/c/replay/replay_snapshot.log", "stdout_log": "outputs/verify/T41/c/replay/replay_snapshot.log"},
        {"id": "client dry run (CPU)", "command": "outputs/verify/T41/c/p/dryrun_cpu.py", "venv": ".venv-openpi-live (CPU)",
         "raw_output": "outputs/verify/T41/c/dryrun/dryrun_cpu.json", "stdout_log": "outputs/verify/T41/c/dryrun/dryrun_cpu.log"},
        {"id": "figures", "command": "outputs/verify/T41/c/p/fig_t41c.py compare_c.json", "venv": ".venv-openpi-live", "raw_output": f"{FIG}/t41c-*.json", "stdout_log": "outputs/verify/T41/c/analysis/fig.log"},
    ],
    "code_state": {"snapshot": "${AG3S_PROJECT_ROOT}-t41c", "snapshot_txt": snap,
                   "code_md5": "outputs/verify/T41/c/p/code_md5_c0.txt (402 files; md5 == C-0 at chain preflight)",
                   "md5_diff_vs_t41a_snapshot": md5diff,
                   "import_check": {s: {"all_benchmark_under_snapshot": IMP[s]["all_benchmark_under_snapshot"], "file": f"outputs/verify/T41/c/import_{s}.json"} for s in IMP},
                   "dry_run": {k: DRY[k] for k in ("reached_sentinel", "hold_ctl_mode", "HOLD_MODES", "client_py")},
                   "T40H_replay_on_snapshot": replay_log,
                   "T40H_replay_vs_T40J_log": "identical apart from snapshot path and wall seconds (diff of outputs/verify/T40/J/replay_snapshot.log)",
                   "benchmark_head_now": head("benchmark"), "pi05_TO_hybrid_head_now": head("pi05_TO_hybrid")},
    "numbers": {
        "_comment": "mm / deg unless noted. All four sets re-analysed with the same scripts (analyze_t41 + hurdle_metrics with the t41a client distance code; hold_c physics replay). T40 = T40 G (B, C) / T40 D records; T40J = T40 J reruns; T41a = margin 0.010, HOLD fixed; T41c = margin 0.010, HOLD measured. B: hurdle placed virtually (r1 config), no physics.",
        "settings": {"server_flags": "T41a: E3b T40 flags + --obstacle-margin 0.010 --no-subtask-gate", "client_flags_added": "--safe-hold-mode measured --viz-avoidance-arrow",
                     "safe_gate": "reasons", "obstacle_stop_distance": 0.0, "warmup_steps": 1000, "xla_autotune": 0, "port": 8270, "seeds": "T40 / T41a seeds (lists/E3b_c.txt == T41/p/lists/E3b_a.txt)"},
        "baseline": {b: {k: BL[b][k] for k in ("violated_start", "has_target", "frame0_clearance_before_mm", "matched", "sha1")} for b in ("legacy", "curobo")},
        "summary_counts": CMP["summary_counts"],
        "outcome": outcome, "gate_hold_verdicts": hold, "hold_episodes": holds, "contact": contact, "min_distance_to_hurdle_by_group": dist,
        "C1982": c1982,
        "T41c_vs_T41a_record_identity": ident,
        "hold_definitions": json.load(open(V / "c/analysis/hold_c.json"))["definitions"],
        "resources": {"memory_rule": "working set (memory.current - inactive_file) + estimate <= 64 GiB; GPU used + 28 GB <= 130 GB",
                      "server_dir": [str(p) for p in srv], "rss_samples": "outputs/verify/T41/c/runs/E3b/server_8270_*/rss_samples.txt",
                      "gpu_samples": "outputs/verify/T41/c/runs/E3b/server_8270_*/gpu_samples.txt",
                      "first_client_after_server_start_s": None},
        "run_records": PART.get("runs"),
    },
    "figures": [{"path": f"{FIG}/t41c-scene-C.png", "kind": "scene"}, {"path": f"{FIG}/t41c-scene-D.png", "kind": "scene"},
                {"path": f"{FIG}/t41c-scene-C1982.png", "kind": "scene"}, {"path": f"{FIG}/t41c-distance.png", "kind": "graph"},
                {"path": f"{FIG}/t41c-hold.png", "kind": "graph"}, {"path": f"{FIG}/t41c-table.png", "kind": "table"}],
    "baseline_check": {"matched": bool(BL["legacy"]["matched"] and BL["curobo"]["matched"]),
                       "note": f"legacy {BL['legacy']['violated_start']} · {BL['legacy']['has_target']} · {BL['legacy']['frame0_clearance_before_mm']}; cuRobo {BL['curobo']['violated_start']} · {BL['curobo']['has_target']} · {BL['curobo']['frame0_clearance_before_mm']}; sha1 repr_json_list equal to T41a"},
    "verdict": "기준선 일치 (legacy · cuRobo). 판정은 lead 와 사용자가 한다.",
    "not_measured": ["TO row replay (replay_t41.py) was not run for T41c (server / TO unchanged vs T41a)",
                     "per-frame clearance and SQP iteration counts are not baselines (wall-clock time budget)"],
}
import re
ch = (V / "c/runs/E3b/chain_8270.log").read_text()
m1 = re.search(r"server up pid=\d+ rec=\S+ startup_s=(\d+)", ch)
out["numbers"]["resources"]["first_client_after_server_start_s"] = None if not m1 else int(m1.group(1)) + 1
out["numbers"]["resources"]["note_120s"] = "chain script (T41a copy) starts the first client right after 'server listening'; no extra 120 s wait after our own server start"
(H / "T41c.verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print("written", H / "T41c.verify.json")
