"""T41a — assemble AG3S/ag3s/docs/handoff/T41a.verify.json (numbers + paths only).
usage: build_verify_t41a.py   (reads the files below)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, datetime, subprocess
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T41'))
H = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff'))
CMP = json.load(open(V / "a/analysis/compare.json"))
RP = json.load(open(V / "a/analysis/replay_a.json"))
AA = json.load(open(V / "a/analysis/analysis_a.json"))["runs"]
BL = json.load(open(V / "a/baseline/judge.json"))
PART = json.load(open(H / "T41a.verify.partial.json"))
md5diff = (V / "p/code_md5_diff_vs_t40.txt").read_text().split()
FIG = "AG3S/ag3s/docs/figures/t41"

runs = CMP["runs"]
def short(k):
    r = runs[k]; return f"{r['stage']}_ep{r['episode']}"

outcome = {}; distance = {}; contact = {}; hold = {}; to_rows = {}; closest = {}; bit = {}
for k, r in runs.items():
    s = short(k)
    for w in ("T40", "T40J", "T41a"):
        f = r.get(w)
        if f is None:
            continue
        outcome.setdefault(s, {})[w] = {x: f[x] for x in ("success", "grasp", "place", "grasp_t", "place_t", "lift_t", "n_control")}
        hold.setdefault(s, {})[w] = {"n_hold_chunks": f["n_hold_chunks"], "hold_t": f["hold_t"], "hold_kind_counts": f["hold_kind_counts"],
                                     "gate_action_counts": f["gate_action_counts"], "verdict_reason_kind_counts": f["verdict_reason_kind_counts"]}
        contact.setdefault(s, {})[w] = {"first_t_dist_le_0": f.get("first_t_dist_le_0"), **{x: f[x] for x in ("obstacle_stop", "obstacle_stop_t", "client_robot_collision", "client_payload_collision",
                                                          "client_contact_steps", "client_min_robot_clearance_mm", "client_contact_pairs",
                                                          "contact_t", "contact_vs_lift")}}
        distance.setdefault(s, {})[w] = {g: {p: (None if v is None else {"min_mm": v["min_mm"], "t": v["t"], "n_rows_le_0": v["n_rows_le_0"]})
                                             for p, v in d.items()} for g, d in f["min_mm"].items()}
        distance[s][w]["physical_hurdle"] = f["physical_hurdle"]
    if r.get("T40_original") is not None:
        outcome[s]["T40_original_table"] = r["T40_original"]
    if r.get("T41a_TO_rows"):
        x = r["T41a_TO_rows"]
        to_rows[s] = {"by_phase": x["by_phase"], "fidelity": x["fidelity_summary"], "chunks_with_obstacle_rows_lt10mm": x["chunks_with_obstacle_lt10mm"]}
        y = r.get("T40_TO_rows_classified_by_T41a_rule")
        if y:
            to_rows[s]["T40_record_same_rule"] = {"by_phase": y["by_phase"], "chunks_with_obstacle_rows_lt10mm": y["chunks_with_obstacle_lt10mm"]}
        if x.get("closest_approach_chunk"):
            closest[s] = x["closest_approach_chunk"]
    if r.get("B_vs_T40G_bit_identity") is not None:
        bit[s] = r["B_vs_T40G_bit_identity"]

def count(stage, w, key):
    return sum(1 for k, r in runs.items() if r["stage"] == stage and (r.get(w) or {}).get(key))

summary = {st: {w: {"n": sum(1 for r in runs.values() if r["stage"] == st and r.get(w)), "episodes": sorted(r["episode"] for r in runs.values() if r["stage"] == st and r.get(w)), "success": count(st, w, "success"),
                    "grasp": count(st, w, "grasp"), "place": count(st, w, "place"),
                    "robot_contact": count(st, w, "client_robot_collision"), "payload_contact": count(st, w, "client_payload_collision"),
                    "obstacle_stop": count(st, w, "obstacle_stop")} for w in ("T40", "T40J", "T41a")} for st in ("B", "C", "D")}

B_fail = {}
for k, r in runs.items():
    if r["stage"] == "B" and not r["T41a"]["success"]:
        x = r.get("T41a_TO_rows") or {}
        a = AA[k]
        B_fail[short(k)] = {"T41a_outcome": outcome[short(k)]["T41a"], "hold": hold[short(k)]["T41a"],
                            "apple_dz_max_mm": a.get("apple_dz_max_mm"), "apple_dz_final_mm": a.get("apple_dz_final_mm"),
                            "grasp_contact_settle": a.get("contact"),
                            "T41a_obstacle_rows_lt10mm_total": sum(v.get("obstacle_lt10mm", 0) for v in x.get("by_phase", {}).values()),
                            "T41a_chunks_with_obstacle_rows_lt10mm": x.get("chunks_with_obstacle_lt10mm"),
                            "T40_record_obstacle_rows_lt10mm_by_T41a_rule": (r.get("T40_TO_rows_classified_by_T41a_rule") or {}).get("chunks_with_obstacle_lt10mm"),
                            "divergence_vs_T40G": r.get("B_vs_T40G_bit_identity")}

head = lambda repo: subprocess.run(["git", "-C", f"{_ag3s_workspace_path()}/{repo}", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
out = {
    "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
    "step": "T41a",
    "date": datetime.date.today().isoformat(),
    "runs": [
        {"id": "E3b runs B 6 / C 5 / D 4 (one server, port 8260)",
         "command": "nohup outputs/verify/T41/p/launch_a.sh  (run_chain.sh E3b 8260 ... lists/E3b_a.txt; start_server.sh = T40 E3b flags + --obstacle-margin 0.010 --no-subtask-gate; run_client.sh --obstacle-stop-distance 0 --warmup-steps 1000 --policy-seed --record ... --view front --record-frames --trajectory-out)",
         "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T41/a/runs/E3b/{B,C,D}/E3b_ep*_s*/ + server_8260_*/server_constraints",
         "stdout_log": "outputs/verify/T41/a/runs/E3b/chain_8260.log, outputs/verify/T41/a/runs/launch.out"},
        {"id": "analysis", "command": "outputs/verify/T41/p/analyze_t41.py (T41a runs; T41_ROOT=T40/G/runs for the T40 re-analysis)",
         "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T41/a/analysis/{analysis_a.json, t40g_reanalysis.json, t40j_reanalysis.json}", "stdout_log": "outputs/verify/T41/a/analysis/*.log"},
        {"id": "TO row replay", "command": "outputs/verify/T41/p/replay_t41.py", "venv": ".venv-openpi-live (CPU)",
         "raw_output": "outputs/verify/T41/a/analysis/replay_a.json (T41a), replay_t40.json (T40 records, same rule)", "stdout_log": "outputs/verify/T41/a/analysis/replay_a.log, replay_a_1828.log, replay_t40.log"},
        {"id": "comparison", "command": "outputs/verify/T41/p/compare_t41.py analysis_a.json t40g_reanalysis.json replay_a.json compare.json replay_t40.json t40j_reanalysis.json", "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T41/a/analysis/compare.json", "stdout_log": None},
        {"id": "baseline", "command": "outputs/verify/T41/a/baseline/run_baseline.sh", "venv": ".venv-openpi-live",
         "raw_output": "outputs/verify/T41/a/baseline/{legacy,curobo}.json, judge.json", "stdout_log": "outputs/verify/T41/a/baseline/{legacy,curobo}.log"},
        {"id": "figures", "command": "outputs/verify/T41/p/fig_t41a.py compare.json analysis_a.json t40g_reanalysis.json t40j_reanalysis.json", "venv": ".venv-openpi-live",
         "raw_output": f"{FIG}/t41a-*.json", "stdout_log": None},
    ],
    "code_state": {"snapshot": "${AG3S_PROJECT_ROOT}-t41a", "benchmark": "git archive 4d090ec", "pi05_TO_hybrid": "git archive bd6a002 (rby1_bringup, rby1_description, rby1_manipulation; other entries symlinked as ${AG3S_PROJECT_ROOT}-t40h)",
                   "code_md5": "outputs/verify/T41/p/code_md5_a0.txt (400 files; md5 == A-0 at chain preflight)",
                   "files_differing_from_t40_snapshot": md5diff,
                   "import_check": ["outputs/verify/T41/a/import_server.json", "outputs/verify/T41/a/import_client.json"],
                   "benchmark_head_now": head("benchmark"), "pi05_TO_hybrid_head_now": head("pi05_TO_hybrid"), "dirty_files": []},
    "numbers": {
        "_comment": "mm unless noted. T40 = T40 G (B, C) / T40 D records re-analysed with the same scripts and the t41a client distance code (H fix); T40_original_table = T40FG.verify.json outcome_table as recorded (pre-fix distance code). B: hurdle placed virtually (r1 config), no physics.",
        "settings": {"server_flags_added": "--obstacle-margin 0.010 --no-subtask-gate (support planes excluded = default)", "obstacle_stop_distance": 0.0,
                     "warmup_steps": 1000, "xla_autotune": 0, "port": 8260, "seeds": "T40 seeds (10*ep + k as in T40 G lists)",
                     "server_log_check": "TO obstacle_margin: 10.0 mm ... 지지면 제외 (present); subtask gate off (기록만)"},
        "baseline": {"legacy": {k: BL["legacy"][k] for k in ("violated_start", "has_target", "frame0_clearance_before_mm", "matched", "sha1")},
                     "curobo": {k: BL["curobo"][k] for k in ("violated_start", "has_target", "frame0_clearance_before_mm", "matched", "sha1")},
                     "reference": "legacy 14/15 · 15/15 · -29.031048280806342; cuRobo 10/15 · 15/15 · -9.171877401271193 (sha1 repr_json_list equal to T42-1 / T40 G)"},
        "summary_counts": summary,
        "outcome": outcome, "hold": hold, "contact": contact, "min_distance_to_hurdle_by_group": distance,
        "B_bit_identity_vs_T40G": bit,
        "B_failures": B_fail,
        "TO_rows_by_class_T41a": {"definitions": RP["definitions"], "per_run": to_rows},
        "closest_approach_TO_vs_true_mesh_T41a": closest,
        "T40J": "T40FG.verify.json numbers.J runs (C1995, D1967, D1982, D1995; snapshot work-t40h, margin off) re-analysed with the same scripts: outputs/verify/T41/a/analysis/t40j_reanalysis.json; per-run entries under key T40J above",
        "resources": {"memory_rule": "working set (memory.current - inactive_file) + estimate <= 64 GiB (lead, 2026-10-05); GPU used + 28 GB <= 130 GB",
                      "rss_samples": "outputs/verify/T41/a/runs/E3b/server_8260_*/rss_samples.txt", "gpu_samples": "outputs/verify/T41/a/runs/E3b/server_8260_*/gpu_samples.txt",
                      "aborted_first_chain": "outputs/verify/T41/a/runs/aborted_20261005T0721 (server started 07:21 under the old memory rule, killed before any run)"},
        "run_records": PART.get("runs"),
    },
    "figures": [{"path": f"{FIG}/t41a-scene-C.png", "kind": "scene"}, {"path": f"{FIG}/t41a-scene-D.png", "kind": "scene"},
                {"path": f"{FIG}/t41a-distance.png", "kind": "graph"}, {"path": f"{FIG}/t41a-table.png", "kind": "table"}],
    "baseline_check": {"matched": bool(BL["legacy"]["matched"] and BL["curobo"]["matched"]),
                       "note": f"legacy {BL['legacy']['violated_start']} · {BL['legacy']['has_target']} · {BL['legacy']['frame0_clearance_before_mm']}; cuRobo {BL['curobo']['violated_start']} · {BL['curobo']['has_target']} · {BL['curobo']['frame0_clearance_before_mm']}"},
    "verdict": "기준선 일치 (legacy · cuRobo). 판정은 lead 와 사용자가 한다.",
    "not_measured": ["held-object query rows are not rebuilt in the TO row replay (count reported as held_query_rows_not_rebuilt)",
                     "per-frame clearance and SQP iteration counts are not baselines (wall-clock time budget)"],
}
(H / "T41a.verify.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
print("written", H / "T41a.verify.json")
