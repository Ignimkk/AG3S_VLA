"""T43 V9 extra numbers (verifier, CPU only): G / P under the Z4 release rule (place_rel) and pairs, per-run TO solve median pairs, loadavg during the V9 round,
GPU totals per server, chain / retry / rc facts, server-log checks. usage: extra_v9.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, math, glob, re, collections
import numpy as np
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V9')); T = V.parent
ct = json.load(open(V / "ct.json")); agg = json.load(open(V / "agg.json"))
R = ct["runs"]
def mc(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)
def pair(f, A="V9", B="V8"):
    ka = {(o["episode"], o["seed"]): o for o in R.values() if o["arm"] == A}; kb = {(o["episode"], o["seed"]): o for o in R.values() if o["arm"] == B}
    ks = sorted(set(ka) & set(kb)); both = sum(ka[k][f] and kb[k][f] for k in ks); ao = sum(ka[k][f] and not kb[k][f] for k in ks); bo = sum(kb[k][f] and not ka[k][f] for k in ks)
    return {"n_pairs": len(ks), "V9_total": both + ao, "V8_total": both + bo, "both": both, "V9_only": ao, "V8_only": bo, "neither": len(ks) - both - ao - bo, "mcnemar_exact_p": mc(ao, bo),
            "V9_only_keys": [f"{e}_s{s}" for e, s in ks if ka[(e, s)][f] and not kb[(e, s)][f]], "V8_only_keys": [f"{e}_s{s}" for e, s in ks if kb[(e, s)][f] and not ka[(e, s)][f]]}
out = {"H_pairs_z4_rule": {f: pair(f) for f in ("grasp", "place_rel", "released", "success_rel", "success_ct")}}
out["H_totals"] = {a: {f: sum(o[f] for o in R.values() if o["arm"] == a) for f in ("success", "grasp", "place", "released", "place_rel", "success_rel", "success_ct")} for a in ("V8", "V9")}
# per-run TO solve median pairs (H + B)
t = agg["timing"]; pr = {}
for run in sorted(set(t["V9"]["ALL"]["per_run"]) & set(t["V8"]["ALL"]["per_run"])):
    a, b = t["V9"]["ALL"]["per_run"][run], t["V8"]["ALL"]["per_run"][run]
    pr[run] = {"V9_median_ms": a["to_solve_median_ms"], "V8_median_ms": b["to_solve_median_ms"], "ratio": a["to_solve_median_ms"] / b["to_solve_median_ms"]}
rr = [x["ratio"] for x in pr.values()]; dd = [x["V9_median_ms"] - x["V8_median_ms"] for x in pr.values()]
out["per_run_to_solve_median"] = {"runs": pr, "ratio_median": float(np.median(rr)), "ratio_min": min(rr), "ratio_max": max(rr), "diff_ms_median": float(np.median(dd)), "n_runs_V9_higher": int(sum(d > 0 for d in dd)), "n_runs": len(dd)}
# load during the V9 round
L = [l.split() for l in open(V / "raw/load_samples.txt") if len(l.split()) >= 6]
l1 = np.array([float(x[1]) for x in L]); ws = np.array([int(x[-1].split("=")[1]) for x in L])
out["loadavg_during_V9"] = {"n_samples": len(L), "load1_median": float(np.median(l1)), "load1_p90": float(np.percentile(l1, 90)), "load1_max": float(l1.max()), "working_set_gib_max": float(ws.max() / 2 ** 30),
                            "start": int(L[0][0]), "end": int(L[-1][0])}
# per-server facts
srv = {}
for d in sorted(glob.glob(str(V / "runs/V9/server_*"))):
    d = pathlib.Path(d); pk = max(int(l.split()[1]) for l in open(d / "gpu_samples.txt") if len(l.split()) == 2)
    log = (d / "serve_safe.log").read_text(errors="replace")
    srv[d.name] = {"gpu_total_used_peak_mib": pk, "gpu_used_before_mib": int(open(d / "gpu_used_before_mib").read()),
                   "cmdline": (d / "server_proc.txt").read_text().splitlines()[-1][-700:],
                   "log_gate_on": bool(re.search(r"subtask label: .* gate ON", log)),
                   "log_finger_cover_line": next((l[:260] for l in log.splitlines() if "constraint finger cover (T43 FC)" in l), None),
                   "log_identity_line": next((l.split("WARNING:root:")[-1][:200] for l in log.splitlines() if "AG3S identity (T43 T fix 3)" in l), None),
                   "log_autotune_off": "XLA autotuning off" in log,
                   "log_servo_model_line": next((l.split("WARNING:root:")[-1][:200] for l in log.splitlines() if "TO path check (T43 Q)" in l), None),
                   "cmdline_has_servo_model": "--servo-model" in (d / "server_proc.txt").read_text(), "cmdline_has_sweep_check": "--sweep-check" in (d / "server_proc.txt").read_text(),
                   "log_finger_cover_270": bool(re.search(r"constraint finger cover \(T43 FC\): 구 122 → 270 ", log)), "log_obstacle_margin_10": "TO obstacle_margin: 10.0 mm" in log,
                   "n_constraint_spheres_in_records": sorted({int(np.load(f, allow_pickle=True)["sphere_radii"].shape[0]) for f in glob.glob(str(d / "server_constraints/run_*/chunk_*.npz"))[::25]})}
out["servers"] = srv
chain = {}
for p in sorted(glob.glob(str(V / "runs/V9/chain_*.log"))):
    L = open(p, errors="replace").read().splitlines()
    chain[pathlib.Path(p).name] = {"n_end_lines": sum(l.startswith("end ") for l in L), "n_attempt2": sum("attempt=2" in l for l in L), "refused": [l[:120] for l in L if "REFUSED" in l], "died": [l[:120] for l in L if "died" in l],
                                  "code_md5_equal": sum("code md5 == V9-0" in l for l in L), "import_ok": sum("import check server+client: all under ${AG3S_PROJECT_ROOT}-t43v8" in l for l in L),
                                  "start_guard_ok": [l.split(" start guard ok ")[-1][:120] if False else l[:140] for l in L if l.startswith("start guard ok")]}
out["chains"] = chain; out["failed_dirs"] = sorted(glob.glob(str(V / "runs/V9/*/failed/*")))
out["chain_rc"] = {"E_H": 0, "E_B": 0, "F_H": 0} if "rc=0" in open(V / "runs/chainE.out").read() and "rc=0" in open(V / "runs/chainF.out").read() else "see runs/chainE.out chainF.out"
out["chain_rc_lines"] = open(V / "runs/chainE.out").read().strip().splitlines() + open(V / "runs/chainF.out").read().strip().splitlines()
# verdict reason kinds and wall
an = json.load(open(V / "analysis/analysis.json"))["runs"]
vk = collections.Counter()
for r in an.values():
    for k, v in r["verdict_kind_counts"].items(): vk[k] += v
out["verdict_kind_counts_V9"] = dict(vk)
an7 = json.load(open(T / "V8/analysis/analysis.json"))["runs"]; vk7 = collections.Counter()
for r in an7.values():
    for k, v in r["verdict_kind_counts"].items(): vk7[k] += v
out["verdict_kind_counts_V8"] = dict(vk7)
out["n_control_lt_600"] = {"V9": {k.split("/")[-1] + "/" + k.split("/")[1]: r["n_control"] for k, r in an.items() if r["n_control"] < 600}, "V8": {k.split("/")[-1] + "/" + k.split("/")[1]: r["n_control"] for k, r in an7.items() if r["n_control"] < 600}}
out["wall_s_median"] = {"V9": float(np.median([r["wall_s"] for r in an.values()])), "V8": float(np.median([r["wall_s"] for r in an7.values()]))}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
print(json.dumps({k: out[k] for k in ("H_totals", "H_pairs_z4_rule")}, default=float)[:1500]); print(out["per_run_to_solve_median"]["ratio_median"], out["per_run_to_solve_median"]["diff_ms_median"], out["per_run_to_solve_median"]["n_runs_V9_higher"], out["loadavg_during_V9"])
print(chain); print(out["n_control_lt_600"], out["wall_s_median"], out["verdict_kind_counts_V9"], out["verdict_kind_counts_V8"])
