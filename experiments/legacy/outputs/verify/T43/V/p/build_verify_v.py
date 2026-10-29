"""T43V: write AG3S/ag3s/docs/handoff/T43V.verify.json (numbers and paths only) from analysis/numbers.json, baseline/judge.json,
import checks, replay log and the server resource samples."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, glob
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V')); OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V.verify.json'))
num = json.loads((V / "analysis/numbers.json").read_text()); judge = json.loads((V / "baseline/judge.json").read_text())
imp = {s: json.loads((V / f"import_{s}.json").read_text()) for s in ("server", "client")}
res = {}
for d in sorted(glob.glob(str(V / "runs/*/server_*"))):
    d = pathlib.Path(d)
    g = [int(l.split()[1]) for l in (d / "gpu_samples.txt").read_text().split("\n") if l.strip() and len(l.split()) > 1 and l.split()[1].isdigit()]
    rs = [l.split() for l in (d / "rss_samples.txt").read_text().split("\n") if l.strip()]
    res[f"{d.parent.name}/{d.name}"] = {"gpu_used_before_mib": int((d / "gpu_used_before_mib").read_text()), "gpu_total_peak_mib": max(g) if g else None,
        "server_rss_max_gib": max(int(x[1]) for x in rs if len(x) > 1 and x[1].isdigit()) / 2**20, "client_rss_max_gib": max(int(x[2]) for x in rs if len(x) > 2) / 2**20,
        "start": (d / "server.start").read_text().strip(), "end": (d / "server.end").read_text().strip()}
ld = [l.split() for l in (V / "runs/load_samples.txt").read_text().split("\n") if l.strip()]
la = [float(x[1]) for x in ld]
import datetime
load_by_arm = {}
for arm in ("E0", "V0", "V1", "V2", "V3"):
    v = []
    for f in glob.glob(str(V / f"runs/{arm}/H/*.start")):
        s0 = datetime.datetime.fromisoformat(open(f).readline().strip()).timestamp(); e0 = datetime.datetime.fromisoformat(open(f[:-6] + ".end").readline().strip()).timestamp()
        v += [float(x[1]) for x in ld if s0 <= int(x[0]) <= e0]
    load_by_arm[arm] = {"mean": sum(v) / len(v) if v else None, "max": max(v) if v else None, "n_samples": len(v)}
out = {
 "_comment": "ag3s-verifier. numbers and paths only; no interpretation.",
 "step": "T43V", "date": "2026-10-05",
 "runs": [
  {"id": "runs: hurdle 4 ep x 2 seeds x {E0, V0, V1, V2} + V2 no-obstacle 5 (two chains, ports 8280 / 8281, <= 2 of our servers)",
   "command": "setsid nohup outputs/verify/T43/V/p/launch_v.sh (run_chain.sh ARM PORT runs/ARM LIST; start_server.sh ARM; run_client.sh ARM; lists/hurdle.txt, lists/clear.txt)",
   "venv": ".venv-openpi-live", "raw_output": "outputs/verify/T43/V/runs/{E0,V0,V1,V2}/{H,B}/<ARM>_ep<EP>_s<SEED>/ + runs/<ARM>/server_*/server_constraints",
   "stdout_log": "outputs/verify/T43/V/runs/{launch.out, chainA.out, chainB.out, <ARM>/chain_<PORT>.log}"},
  {"id": "runs V3 (added): hurdle 4 ep x 2 seeds, two chains (8280 / 8281)", "command": "setsid nohup outputs/verify/T43/V/p/launch_v3.sh", "venv": ".venv-openpi-live",
   "raw_output": "outputs/verify/T43/V/runs/V3/H/V3_ep<EP>_s<SEED>/ + runs/V3/server_*/server_constraints", "stdout_log": "outputs/verify/T43/V/runs/{launch_v3.out, chainA_v3.out, chainB_v3.out, V3/chain_<PORT>.log}"},
  {"id": "analysis", "command": "outputs/verify/T43/V/p/post_v.sh (analyze_t43v.py -> hold_v.py -> partial_v.py); partial_v.py --final analysis/numbers.json",
   "venv": ".venv-openpi-live (CPU)", "raw_output": "outputs/verify/T43/V/analysis/{analysis.json, hold_v.json, numbers.json}", "stdout_log": "outputs/verify/T43/V/analysis/{analysis.log, hold_v.log, partial_v.log, watch.out}"},
  {"id": "baseline", "command": "outputs/verify/T43/V/baseline/run_baseline.sh; judge.py", "venv": ".venv-openpi-live",
   "raw_output": "outputs/verify/T43/V/baseline/{legacy,curobo}.json, judge.json", "stdout_log": "outputs/verify/T43/V/baseline/{legacy,curobo}.log, judge.out"},
  {"id": "T40 H replay on the snapshot (CPU)", "command": "outputs/verify/T43/V/p/replay_snap_t43v.py", "venv": ".venv-openpi-live (CPU)",
   "raw_output": "outputs/verify/T43/V/replay/replay_snapshot.log", "stdout_log": "outputs/verify/T43/V/replay/{replay_snapshot.log, diff_vs_T41c.txt}"},
  {"id": "figures", "command": "outputs/verify/T43/V/p/fig_t43v.py", "venv": ".venv-openpi-live", "raw_output": "AG3S/ag3s/docs/figures/t43/t43v-*.{png,json}", "stdout_log": "outputs/verify/T43/V/analysis/fig.log"}],
 "code_state": {
  "snapshot": "${AG3S_PROJECT_ROOT}-t43v", "snapshot_txt": (V / "snapshot.txt").read_text().splitlines(),
  "benchmark_head": "f6af67c (contains 78a64ed, e2c6d1a)", "pi05_TO_hybrid": "rby1_description + rby1_manipulation bd6a002, rby1_bringup fce4af8 (trees equal to ${AG3S_PROJECT_ROOT}-t41c)",
  "code_md5": "outputs/verify/T43/V/p/code_md5_v0.txt (405 files); == V-0 at every chain preflight",
  "md5_diff_vs_t41c_snapshot": (V / "p/code_md5_diff_vs_t41c.txt").read_text().split("\n")[:-1],
  "import_check": {s: {"all_benchmark_under_snapshot": imp[s]["all_benchmark_under_snapshot"], "file": f"outputs/verify/T43/V/import_{s}.json"} for s in imp},
  "T40H_replay_vs_T41c_log": "identical apart from snapshot path and wall seconds (outputs/verify/T43/V/replay/diff_vs_T41c.txt)",
  "dirty_files": []},
 "settings": {
  "E0": "server --no-safe; client no safe flags",
  "V0": "server T34 E3b flags (--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized --esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32) + --no-subtask-gate; client --safe-hold-mode fixed",
  "V1": "V0 + server --obstacle-margin 0.010; client --safe-hold-mode measured",
  "V3": "added by lead after V2: V2 without --gripper-cover = V1 + server --held-body-cover --servo-model --sweep-check (Q defaults 12 mm / 16 / 24); server log: constraints 122 spheres, no gripper-cover line; runs/V3, ports 8280 / 8281, 16:40-17:03 UTC",
  "V2": "V1 + server --gripper-cover left --gripper-cover-max-gap 3.0 --held-body-cover --servo-model --sweep-check (no --rest-start); server log: spheres 122 -> 651",
  "client_common": "--safe-remote --safe-timeout 120 --safe-phase approach --safe-manipulators left --safe-gate reasons (V arms); all arms --obstacle-stop-distance 0 --viz-avoidance-arrow --record-frames --record --trajectory-out --warmup-steps 1000 --max-steps 600",
  "hurdle_configs": {"1807": "t43p_ep1807_s18071", "1968": "t43p_ep1968_s19681", "1982": "t43p_ep1982_s19822", "1967": "t43p_ep1967_s19672_crelax"},
  "seeds": {"1807": [18071, 18073], "1968": [19681, 19683], "1982": [19822, 19823], "1967": [19672, 19673]},
  "no_obstacle_seeds": {"1807": 18071, "1968": 19681, "1982": 19822, "1995": 19952, "1967": 19672},
  "xla_autotune": 0},
 "numbers": dict(num, resources={"servers": res, "load_avg_1min": {"mean": sum(la) / len(la), "max": max(la), "n_samples": len(la), "file": "outputs/verify/T43/V/runs/load_samples.txt", "during_hurdle_runs_by_arm": load_by_arm},
                                 "rules": "working set + estimate <= 64 GiB, >= 120 s after every serve_safe start, GPU used + 28 GB <= 130 GB, <= 2 of our servers; another verifier (T42-2) ran up to 2 servers concurrently"},
                 baseline=judge),
 "figures": [{"path": "AG3S/ag3s/docs/figures/t43/t43v-scene.png", "kind": "scene"}, {"path": "AG3S/ag3s/docs/figures/t43/t43v-distance.png", "kind": "graph"},
             {"path": "AG3S/ag3s/docs/figures/t43/t43v-chunk-time.png", "kind": "graph"}, {"path": "AG3S/ag3s/docs/figures/t43/t43v-table.png", "kind": "table"},
             {"path": "AG3S/ag3s/docs/figures/t43/t43v-v3-scene.png", "kind": "scene"}, {"path": "AG3S/ag3s/docs/figures/t43/t43v-v3-distance.png", "kind": "graph"},
             {"path": "AG3S/ag3s/docs/figures/t43/t43v-v3-chunk-time.png", "kind": "graph"}, {"path": "AG3S/ag3s/docs/figures/t43/t43v-v3-table.png", "kind": "table"}],
 "baseline_check": {"matched": bool(judge["legacy"]["matched"] and judge["curobo"]["matched"]),
                    "note": f"legacy {judge['legacy']['violated_start']} / {judge['legacy']['has_target']} / {judge['legacy']['frame0_clearance_before_mm']}; cuRobo {judge['curobo']['violated_start']} / {judge['curobo']['has_target']} / {judge['curobo']['frame0_clearance_before_mm']}; sha1 repr_json_list equal to T41c (legacy {judge['legacy']['sha1_repr_json_list_equals_T41c']}, cuRobo {judge['curobo']['sha1_repr_json_list_equals_T41c']})"},
 "verdict": "기준선 일치 (legacy · cuRobo). 판정은 lead 와 사용자가 한다.",
 "not_measured": [
  "per-frame clearance and SQP iteration counts are not baselines (wall-clock time budget)",
  "hurdle distance of the no-obstacle runs (B): no virtual hurdle (ep1995 has no T43 P placement)",
  "grasp contact height (analyze_t39 do_contact) not computed",
  "V3 ran 16:40-17:03 UTC with no other GPU job (GPU 0 MiB before start); V0-V2 / E0 ran 13:49-15:29 UTC while another verifier (T42-2) ran up to 2 servers — TO / round-trip times of V3 vs the other arms are not under equal load (runs/load_samples.txt)",
  "server TO / round-trip times were measured on a shared pod (load samples in runs/load_samples.txt); no isolated timing run",
  "contacts: client robot_collision / payload_collision (every sim step) and physics replay contact events (hold_v.py); hurdle_metrics rows <= 0 are on pre-step control-row poses only"]}
OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float)); print(OUT)
