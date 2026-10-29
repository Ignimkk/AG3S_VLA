"""T43W: write AG3S/ag3s/docs/handoff/T43W.verify.json (numbers and paths only) from analysis/numbers.json, baseline{,2}/judge.json,
import checks, replay log, the offline sweep tuning (sweep_tune/) and the server resource samples."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, glob, datetime
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/W')); OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43W.verify.json'))
num = json.loads((V / "analysis/numbers.json").read_text())
judge = {k: json.loads((V / f"{k}/judge.json").read_text()) for k in ("baseline", "baseline2")}
imp = {f"{p}_{s}": json.loads((V / f"{p}_{s}.json").read_text()) for p in ("import", "import2") for s in ("server", "client")}
res = {}
for d in sorted(glob.glob(str(V / "runs/*/server_*"))):
    d = pathlib.Path(d)
    if not (d / "gpu_samples.txt").exists():
        continue
    g = [int(l.split()[1]) for l in (d / "gpu_samples.txt").read_text().split("\n") if l.strip() and len(l.split()) > 1 and l.split()[1].isdigit()]
    rs = [l.split() for l in (d / "rss_samples.txt").read_text().split("\n") if l.strip()] if (d / "rss_samples.txt").exists() else []
    res[f"{d.parent.name}/{d.name}"] = {"gpu_used_before_mib": int((d / "gpu_used_before_mib").read_text()), "gpu_total_peak_mib": max(g) if g else None,
        "server_rss_max_gib": max([int(x[1]) for x in rs if len(x) > 1 and x[1].isdigit()], default=0) / 2**20,
        "client_rss_max_gib": max([int(x[2]) for x in rs if len(x) > 2 and x[2].isdigit()], default=0) / 2**20,
        "start": (d / "server.start").read_text().strip(), "end": (d / "server.end").read_text().strip() if (d / "server.end").exists() else None}
ld = [l.split() for l in (V / "runs/load_samples.txt").read_text().split("\n") if l.strip()]
la = [float(x[1]) for x in ld]
load_by_arm = {}
for arm in ("E0", "V1", "V3", "V4", "V5", "V6"):
    v = []
    for f in glob.glob(str(V / f"runs/{arm}/*/*.start")):
        if not pathlib.Path(f[:-6] + ".end").exists():
            continue
        s0 = datetime.datetime.fromisoformat(open(f).readline().strip()).timestamp(); e0 = datetime.datetime.fromisoformat(open(f[:-6] + ".end").readline().strip()).timestamp()
        v += [float(x[1]) for x in ld if s0 <= int(x[0]) <= e0]
    load_by_arm[arm] = {"mean": sum(v) / len(v) if v else None, "max": max(v) if v else None, "n_samples": len(v)}
det = json.loads((V / "sweep_tune/detect.json").read_text()); tim = json.loads((V / "sweep_tune/timing.json").read_text())
chosen = "m24_s4_r6"
sweep = {"chosen": {"--sweep-max-mm": 24, "--sweep-max-samples": 4, "--sweep-rows-per-step": 6, "name": chosen},
         "script": "outputs/verify/T43/W/sweep_tune/tune.py (harness tests/trajopt/_t43q_replay.py, tests/sim/_t43q_exec.py; code ${AG3S_PROJECT_ROOT}-t43v)",
         "timing_summary_ms": tim["summary"], "timing_reps": tim["reps"],
         "timing_chunks": "T43 V V2 hurdle runs (T40 seeds), seq 4,10,..,70, 47 chunks, 122-sphere model (no gripper cover), margin 0.010, recorded SQP iterations, no clock",
         "detect": [{"chunk": r["chunk"], "margin_m": r["margin_m"], "cand": r["cand"], "recorded_exec_contact": r["recorded_exec"]["contact"],
                     "recorded_plan_path_clearance_mm": r.get("recorded_plan_path_clearance_mm"), "recorded_plan_waypoint_clearance_mm": r.get("recorded_plan_waypoint_clearance_mm"),
                     "resolve_status": r["resolve"]["status"], "resolve_path_clearance_mm": r["resolve"]["path_clearance_mm"],
                     "resolved_exec_contact": r["resolved_exec"]["contact"], "resolved_exec_true_mm_min": r["resolved_exec"]["true_mm_min"]}
                    for r in det["rows"] if r["cand"] in ("off", "servo_only", "default", chosen)],
         "detect_all_candidates_file": "outputs/verify/T43/W/sweep_tune/detect.json (21 candidates x 3 chunk/margin cases)",
         "timing_file": "outputs/verify/T43/W/sweep_tune/timing.json"}
out = {
 "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
 "step": "T43W", "date": "2026-10-05",
 "runs": [
  {"id": "runs E0 / V1 / V3 / V4: P2 overbar 7 ep x 2 seeds + V3 / V4 no-obstacle 5 (chains A port 8282, B port 8283)",
   "command": "outputs/verify/T43/W/p/launch_w.sh (chain B) + p/launch_wA.sh (chain A relaunched: the first chain A was refused by its own V3 log guard, regex replaced by fixed strings)",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/W/runs/{E0,V1,V3,V4}/{H,B}/<ARM>_ep<EP>_s<SEED>/ + runs/<ARM>/server_*/server_constraints",
   "stdout_log": "outputs/verify/T43/W/runs/{launch.out, launch_A.out, chainA.out, chainB.out, <ARM>/chain_<PORT>.log}"},
  {"id": "runs V5 / V6 (T2 arms, second snapshot): P2 7 ep x 2 seeds + no-obstacle 5 (chain C port 8282 after A, chain D port 8283 after B)",
   "command": "outputs/verify/T43/W/p56/launch_w56.sh", "venv": ".venv-openpi-live",
   "raw_output": "outputs/verify/T43/W/runs/{V5,V6}/{H,B}/...", "stdout_log": "outputs/verify/T43/W/runs/{launch_56.out, chainC.out, chainD.out}"},
  {"id": "offline sweep tuning (CPU)", "command": "outputs/verify/T43/W/sweep_tune/tune.py detect | timing", "venv": ".venv-openpi-live (CPU)",
   "raw_output": "outputs/verify/T43/W/sweep_tune/{detect.json, timing.json}", "stdout_log": "outputs/verify/T43/W/sweep_tune/{detect.log, timing.log}"},
  {"id": "analysis", "command": "outputs/verify/T43/W/p/post_w.sh (analyze_t43w.py -> hold_w.py -> partial_w.py); partial_w.py --final analysis/numbers.json",
   "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T43/W/analysis/{analysis.json, hold_w.json, numbers.json}", "stdout_log": "outputs/verify/T43/W/analysis/{analysis.log, hold_w.log, partial_w.log, watch.out, watch56.out}"},
  {"id": "baselines", "command": "outputs/verify/T43/W/baseline/run_baseline.sh (snapshot 1), baseline2/run_baseline.sh (snapshot 2); judge.py", "venv": ".venv-openpi-live",
   "raw_output": "outputs/verify/T43/W/baseline{,2}/{legacy,curobo}.json, judge.json", "stdout_log": "outputs/verify/T43/W/baseline{,2}/{legacy,curobo}.log"},
  {"id": "T40 H replay on snapshot 1 (CPU)", "command": "outputs/verify/T43/W/p/replay_snap_t43w.py", "venv": ".venv-openpi-live (CPU)",
   "raw_output": "outputs/verify/T43/W/replay/replay_snapshot.log", "stdout_log": "outputs/verify/T43/W/replay/{replay_snapshot.log, diff_vs_T43V.txt}"},
  {"id": "figures", "command": "outputs/verify/T43/W/p/fig_t43w.py", "venv": ".venv-openpi-live", "raw_output": "AG3S/ag3s/docs/figures/t43/t43w-*.{png,json}", "stdout_log": "outputs/verify/T43/W/analysis/fig.log"}],
 "code_state": {
  "snapshot_1": {"path": "${AG3S_PROJECT_ROOT}-t43w", "arms": "E0 V1 V3 V4", "snapshot_txt": (V / "snapshot.txt").read_text().splitlines(),
                 "code_md5": "outputs/verify/T43/W/p/code_md5_w0.txt (405 files; == W-0 at every chain preflight)",
                 "md5_diff_vs_t43v_snapshot": (V / "p/code_md5_diff_vs_t43v.txt").read_text().split("\n")[:-1]},
  "snapshot_2": {"path": "${AG3S_PROJECT_ROOT}-t43w2", "arms": "V5 V6", "snapshot_txt": (V / "snapshot2.txt").read_text().splitlines(),
                 "code_md5": "outputs/verify/T43/W/p56/code_md5_w2.txt (405 files; == W2-0 at every chain preflight)",
                 "md5_diff_vs_snapshot_1": (V / "p56/code_md5_diff_vs_w0.txt").read_text().split("\n")[:-1]},
  "pi05_TO_hybrid_change_vs_T43V": "rby1_description and rby1_manipulation from fce4af8 (overbar slot, c7c1cc9+) instead of bd6a002; rby1_bringup fce4af8 as before",
  "import_check": {k: {"all_benchmark_under_snapshot": v["all_benchmark_under_snapshot"], "file": f"outputs/verify/T43/W/{k}.json"} for k, v in imp.items()},
  "T40H_replay_vs_T43V_log": "identical apart from snapshot path and wall seconds (outputs/verify/T43/W/replay/diff_vs_T43V.txt); snapshot 2 has the same pi05_TO_hybrid trees",
  "dirty_files": []},
 "settings": {
  "E0": "server --no-safe; client no safe flags",
  "V1": "server T34 E3b flags + --no-subtask-gate + --obstacle-margin 0.010; client --safe-hold-mode measured  (a + c)",
  "V4": "V1 + server --held-body-cover  (a + c + R2)",
  "V3": "V4 + server --servo-model --sweep-check --sweep-max-mm 24 --sweep-max-samples 4 --sweep-rows-per-step 6  (a + c + R2 + Q, lighter sweep)",
  "V5": "V4 + server --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 (snapshot 2; --refused-identity-release, --held-free-observed-mm off)",
  "V6": "V5 + server --servo-model --sweep-check --sweep-max-mm 24 --sweep-max-samples 4 --sweep-rows-per-step 6 --target-volume-exempt (snapshot 2)",
  "not_used": "--gripper-cover (R), --rest-start",
  "client_common": "V arms: --safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators <left | right for ep1834> --safe-gate reasons --safe-hold-mode measured; all arms --obstacle-stop-distance 0 --viz-avoidance-arrow --record-frames --record --trajectory-out --warmup-steps 1000 --max-steps 600",
  "obstacle_configs": "outputs/verify/T43/P2/configs/t43p2_ep<EP>_s<SEED>.json, profile t43p2_overbar",
  "seeds": {"1807": [18071, 18073], "1968": [19681, 19683], "1982": [19822, 19823], "1995": [19952, 19953], "1967": [19672, 19673], "1925": [19251, 19253], "1834": [18341, 18343]},
  "no_obstacle_seeds": {"1807": 18071, "1968": 19681, "1982": 19822, "1995": 19952, "1967": 19672},
  "xla_autotune": 0},
 "numbers": dict(num, sweep_tuning=sweep, baseline=judge["baseline"], baseline_snapshot2=judge["baseline2"],
                 resources={"servers": res, "load_avg_1min": {"mean": sum(la) / len(la), "max": max(la), "n_samples": len(la), "file": "outputs/verify/T43/W/runs/load_samples.txt",
                                                              "during_runs_by_arm": load_by_arm},
                            "rules": "working set + estimate <= 64 GiB, >= 120 s after every serve_safe start, GPU used + 28 GB <= 130 GB, <= 2 of our servers"}),
 "incident_V5_V6": {
  "what": "every V5 / V6 chunk was HOLD 'uncertified: no scene' (V5 961 / 961 HOLD verdict lines, V6 361 / 361 in serve_safe.log); chunk records carry no esdf_stats and profile_ms has collision_candidates / primitive_fitting instead of esdf",
  "reproduction": "CPU, ${AG3S_PROJECT_ROOT}-t43w2: AG3SConfig.from_dict({'collision_backend': 'esdf'}).with_overrides({'clustering': {'latched_identity_hold': True}}).collision_backend == 'primitive'; AG3SConfig.to_dict() has no 'collision_backend' key",
  "code_path": "AG3S/trajopt/serve_safe.py build_ag3s: config.with_overrides({'clustering': dict(identity)}) when any T2 identity flag is given (t43t_identity_section); AG3S/ag3s/config.py AG3SConfig.with_overrides -> to_dict (frame_id + sections only)",
  "action": "chains C / D stopped by the verifier at 2026-10-05T20:35:59Z; 16 completed runs (V5 12, V6 4) moved to outputs/verify/T43/W/runs/{V5,V6}_invalid_collision_backend/ and excluded from numbers; p56/run_chain.sh now stops a chain whose newest server chunk has no esdf_stats",
  "rerun": "needs a fixed commit and a new snapshot; scripts outputs/verify/T43/W/p56/ (change O1)"},
 "figures": [{"path": f"AG3S/ag3s/docs/figures/t43/t43w-{n}.png", "kind": k} for n, k in (("scene", "scene"), ("distance", "graph"), ("chunk-time", "graph"), ("table", "table"))],
 "baseline_check": {"matched": bool(all(judge[k][b]["matched"] for k in judge for b in ("legacy", "curobo"))),
                    "note": "; ".join(f"{k}: legacy {judge[k]['legacy']['violated_start']} / {judge[k]['legacy']['has_target']} / {judge[k]['legacy']['frame0_clearance_before_mm']}, cuRobo {judge[k]['curobo']['violated_start']} / {judge[k]['curobo']['has_target']} / {judge[k]['curobo']['frame0_clearance_before_mm']}, sha1 = T41c {judge[k]['legacy']['sha1_repr_json_list_equals_T41c']}/{judge[k]['curobo']['sha1_repr_json_list_equals_T41c']}" for k in judge)},
 "verdict": "기준선 일치 (두 snapshot 모두 legacy · cuRobo). 판정은 lead 와 사용자가 한다.",
 "not_measured": [
  "per-frame clearance and SQP iteration counts are not baselines (wall-clock time budget)",
  "overbar distance of the no-obstacle runs (B): not computed",
  "server TO / round-trip times on a shared pod (load samples in runs/load_samples.txt; other sessions ran CPU jobs up to load 77); no isolated timing run",
  "offline sweep tuning: detection measured on 3 recorded contact chunks (T40 G / T41 a C1807), timing on 47 T43 V V2 chunks; attached (held) chunks were not replayed",
  "contacts: client robot_collision / payload_collision (every sim step) and physics replay contact events (hold_w.py); overbar-metric rows <= 0 are on pre-step control-row poses only",
  "V5 / V6 (T2 arms): no valid closed-loop runs (incident_V5_V6)",
  "held-row violations: only the waypoint / path WORST row per chunk is recorded by the server; a held row that is negative but not the worst row is not counted"]}
# ---- 2026-10-06 final aggregation: V5 rerun (snapshot 3) + blind round (snapshot ${AG3S_PROJECT_ROOT}-t43b). Old E0 / V1 / V3 / V4 numbers and the V5 / V6 incident record are kept as they were;
#      numbers.json (old, 2026-10-05) == final/numbers_main.json for every E0 / V1 / V3 / V4 entry (checked key by key: no differences); numbers block below comes from final/numbers_main.json.
F = V / "final"
nm = json.loads((F / "numbers_main.json").read_text()); nb = json.loads((F / "numbers_blind.json").read_text()); agg = json.loads((F / "agg.json").read_text()); esp = json.loads((F / "esdf_presence.json").read_text())
jb = {k: json.loads((V / f"{k}/judge.json").read_text()) for k in ("baseline3", "baseline_blind")}
res_b = {}
for d in sorted(glob.glob(str(V / "runs_blind/*/server_*"))):
    d = pathlib.Path(d)
    g = [int(l.split()[1]) for l in (d / "gpu_samples.txt").read_text().split("\n") if l.strip() and len(l.split()) > 1 and l.split()[1].isdigit()]
    rs = [l.split() for l in (d / "rss_samples.txt").read_text().split("\n") if l.strip()] if (d / "rss_samples.txt").exists() else []
    res_b[f"{d.parent.name}/{d.name}"] = {"gpu_used_before_mib": int((d / "gpu_used_before_mib").read_text()), "gpu_total_peak_mib": max(g) if g else None,
        "server_rss_max_gib": max([int(x[1]) for x in rs if len(x) > 1 and x[1].isdigit()], default=0) / 2**20,
        "client_rss_max_gib": max([int(x[2]) for x in rs if len(x) > 2 and x[2].isdigit()], default=0) / 2**20,
        "start": (d / "server.start").read_text().strip(), "end": (d / "server.end").read_text().strip() if (d / "server.end").exists() else None}
out["date"] = "2026-10-05 (E0 / V1 / V3 / V4, incident); 2026-10-06 (V5 rerun, blind round, final aggregation)"
out["runs"] += [
 {"id": "V5 rerun (snapshot 3 ${AG3S_PROJECT_ROOT}-t43w3, benchmark 72ed927): P2 7 ep x 2 seeds + no-obstacle 5 (chain E port 8282, chain F port 8283)",
  "command": "outputs/verify/T43/W/p5/launch_w5.sh", "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/W/runs/V5/{H,B}/V5_ep<EP>_s<SEED>/ + runs/V5/server_*/server_constraints",
  "stdout_log": "outputs/verify/T43/W/runs/{launch_v5.out, chainE.out, chainF.out}", "ended": "all end V5 2026-10-06T02:09:38+00:00 (chains E, F rc 0)"},
 {"id": "blind round (snapshot ${AG3S_PROJECT_ROOT}-t43b, benchmark c265623 + pi05_TO_hybrid f62ab81): V5B (client --obstacle-policy-hidden) 14 + V5S (visible) 14 hurdle runs (chain G port 8282, chain H port 8283)",
  "command": "outputs/verify/T43/W/pb/launch_blind.sh", "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/W/runs_blind/{V5B,V5S}/H/<ARM>_ep<EP>_s<SEED>/ + runs_blind/<ARM>/server_*/server_constraints",
  "stdout_log": "outputs/verify/T43/W/runs_blind/{launch.out, chainG.out, chainH.out}", "ended": "all end blind 2026-10-06T03:00:22+00:00 (chains G, H rc 0)"},
 {"id": "final aggregation (CPU)", "command": "p/post_w.sh + p/post_b.sh (analyze_t43w.py, hold_w.py, partial_w.py) ; partial_w.py --final final/numbers_main.json | (T43W_AN=analysis_blind T43W_RUNS=runs_blind) final/numbers_blind.json ; p/esdf_presence_v5.py ; p/final_agg.py ; p/fig_focus.py ; p/fig_t43w_final.py ; p/build_verify_w.py",
  "venv": ".venv-openpi-live (CPU, JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa)", "raw_output": "outputs/verify/T43/W/{analysis,analysis_blind,final}/",
  "stdout_log": "outputs/verify/T43/W/{analysis,analysis_blind}/{analysis.log, hold_w.log, partial_w.log}"}]
out["code_state"]["snapshot_3"] = {"path": "${AG3S_PROJECT_ROOT}-t43w3", "arms": "V5 (rerun)", "snapshot_txt": (V / "snapshot3.txt").read_text().splitlines(), "code_md5": "outputs/verify/T43/W/p5/code_md5_w3.txt",
                                   "diff_vs_snapshot_2": (V / "snapshot3_vs_snapshot2.txt").read_text().splitlines(), "md5_diff_vs_snapshot_2": (V / "p5/code_md5_diff_vs_w2.txt").read_text().split("\n")[:-1]}
out["code_state"]["snapshot_blind"] = {"path": "${AG3S_PROJECT_ROOT}-t43b", "arms": "V5B V5S", "snapshot_txt": (V / "snapshot_blind.txt").read_text().splitlines(), "code_md5": "outputs/verify/T43/W/pb/code_md5_b0.txt",
                                       "diff_vs_snapshot_3_benchmark_py_outside_docs": (V / "snapshot_blind_vs_snapshot3.txt").read_text().splitlines(),
                                       "md5_diff_vs_snapshot_3": (V / "pb/code_md5_diff_vs_w3.txt").read_text().split("\n")[:-1]}
out["settings"]["V5_rerun"] = "V4 flags + server --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 (snapshot 3, with_overrides fix); same client flags as V4"
out["settings"]["V5S"] = "same flags as V5 rerun, code snapshot ${AG3S_PROJECT_ROOT}-t43b (differs from snapshot 3 only in the pi05_TO_hybrid trees: client pi05_infer.py, obstacle XML, pick_place_obstacles.py; see code_state.snapshot_blind)"
out["settings"]["V5B"] = "V5S + client --obstacle-policy-hidden (obstacle hidden in the policy RGB only; depth / recordings / collision unchanged)"
out["settings"]["fail_stage_definition"] = "per failed hurdle run, grasp = apple z >= initial + 50 mm (analyze_e0): before_pick = grasp False and no closing chunk; grasp_closing_only = grasp False, closing chunks, no attached chunk; grasp_attached_not_lifted = grasp False, attached chunks > 0; carry = grasp True, place False, state 'placed' never reached; place = same, state 'placed' reached"
out["numbers"] = dict(nm, sweep_tuning=sweep, baseline=judge["baseline"], baseline_snapshot2=judge["baseline2"], baseline_snapshot3=jb["baseline3"], baseline_blind_snapshot=jb["baseline_blind"],
                      resources={"servers": dict(res, **res_b), "load_avg_1min": out["numbers"]["resources"]["load_avg_1min"], "rules": out["numbers"]["resources"]["rules"]})
out["numbers"]["v5_rerun_and_blind"] = {
 "_comment": "numbers only. arms_table / pairs / focus / detail / failure_stage from final/agg.json (p/final_agg.py); per-run values in numbers.per_run (V5 = rerun) and numbers_blind.per_run (V5S / V5B)",
 "arms_table": agg["arms"], "pairs": agg["pairs"], "focus_closing_chunks_pooled": agg["focus"], "detail_per_arm": agg["V5_detail"], "failure_stage": agg["fail_stage"],
 "esdf_stats_presence": esp["summary"], "esdf_stats_presence_per_run": esp["runs"], "serve_safe_log_no_scene_lines": esp["server_logs_no_scene_lines"],
 "E0_V1_V3_V4_numbers_changed_vs_2026-10-05": "none (analysis/numbers.json vs final/numbers_main.json: summary_per_arm, pairs V3/V4/V1/E0 and per_run of E0 V1 V3 V4 compared key by key; no differences)",
 "files": {"main": "outputs/verify/T43/W/final/numbers_main.json", "blind": "outputs/verify/T43/W/final/numbers_blind.json", "agg": "outputs/verify/T43/W/final/agg.json", "esdf_presence": "outputs/verify/T43/W/final/esdf_presence.json"}}
out["numbers_blind"] = {"summary_per_arm": nb["summary_per_arm"], "pairs": nb["pairs"], "per_run": nb["per_run"]}
out["incident_V5_V6"]["rerun"] = "done 2026-10-06 as V5 rerun (snapshot 3, benchmark 72ed927; runs/V5, numbers.v5_rerun_and_blind); V6 not rerun. The incident record above (16 invalid runs in runs/{V5,V6}_invalid_collision_backend/) is unchanged"
out["figures"] += [{"path": f"AG3S/ag3s/docs/figures/t43/{n}.png", "kind": k} for n, k in (("t43w-v5-arms", "graph"), ("t43w-v5b-v5s-pairs", "table"), ("t43w-v5-fail-scene", "scene"), ("t43w-v5-focus", "graph"))]
out["baseline_check"] = {"matched": bool(out["baseline_check"]["matched"] and all(jb[k][b]["matched"] for k in jb for b in ("legacy", "curobo"))),
                         "note": out["baseline_check"]["note"] + "; " + "; ".join(f"{k}: legacy {jb[k]['legacy']['violated_start']} / {jb[k]['legacy']['has_target']} / {jb[k]['legacy']['frame0_clearance_before_mm']}, cuRobo {jb[k]['curobo']['violated_start']} / {jb[k]['curobo']['has_target']} / {jb[k]['curobo']['frame0_clearance_before_mm']}, sha1 = T41c {jb[k]['legacy']['sha1_repr_json_list_equals_T41c']}/{jb[k]['curobo']['sha1_repr_json_list_equals_T41c']}" for k in jb)}
out["verdict"] = "기준선 일치 (snapshot 1 · 2 · 3 · blind 모두 legacy · cuRobo). 판정은 lead 와 사용자가 한다."
out["not_measured"] = [x for x in out["not_measured"] if not x.startswith("V5 / V6")] + [
  "V6: no valid closed-loop runs (not rerun)",
  "V5S vs V5 differ in code snapshot (${AG3S_PROJECT_ROOT}-t43b vs ${AG3S_PROJECT_ROOT}-t43w3); no run of V5 on the blind snapshot other than V5S",
  "blind round: only V5B vs V5S (E0 / V4 blind cancelled); n = 14 pairs per arm, no no-obstacle runs in the blind round",
  "held-row violation / HOLD counts include only the server's recorded worst row per chunk; failure stage is derived from grasp / place flags and server grasp.state (definition in settings.fail_stage_definition)"]
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float)); print(OUT)
