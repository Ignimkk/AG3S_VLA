"""T44c S1 aggregation (verifier). usage: agg_s1.py OUT_JSON. Same judgement as LB (plan_lb.result): ok = analyze_t42 success (grasp and place)
AND grasp_robust robust_grasp (B). Per run <name>.robust.json cached like LB. Pass = both seeds (rep 1, 2) ok."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, collections, glob
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/LB2/p'))
import analyze_t42 as AN
import grasp_robust as GR
LB = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/LB2')); RUNS = LB / "runs" / "E0"
C = json.loads((LB / "candidates.json").read_text())
def result(ep, seed):
    name = f"E0_ep{ep}_s{seed}"; d = RUNS / name
    if not (RUNS / f"{name}.done").exists():
        return None
    rj = RUNS / f"{name}.robust.json"
    if rj.exists() and rj.stat().st_mtime >= (d / "frames.jsonl").stat().st_mtime:
        r = json.loads(rj.read_text())
    else:
        r = GR.robust(d); r.pop("series", None); rj.write_text(json.dumps(r, indent=1, default=float))
    a = AN.analyze(d)
    fm = GR.failure_mode(r)
    ok = bool(a["success"] and r["robust_grasp"])
    if not ok and fm == "ok": fm = "not_success_other"
    return {"seed": seed, "success": bool(a["success"]), "grasp": bool(a["grasp"]), "place": bool(a["place"]), "robust": bool(r["robust_grasp"]),
            "robust_strict": bool(r["robust_grasp_strict"]), "ok": ok, "failure_mode": fm if not ok else "ok",
            "drift_mm_max": r.get("drift_mm_max"), "n_close_cycles": r.get("n_close_cycles"), "run_dir": str(d)}
eps = {}; passed = []; reasons = collections.Counter(); run_reasons = collections.Counter()
for v in C["episodes"]:
    e = v["episode"]; rs = [result(e, 10 * e + k) for k in (1, 2)]
    n_done = sum(r is not None for r in rs); n_ok = sum(1 for r in rs if r and r["ok"])
    status = "pending" if n_done < 2 else ("pass" if n_ok == 2 else "fail")
    why = []
    if status == "fail":
        why = sorted({r["failure_mode"] for r in rs if r and not r["ok"]})
        for w in why: reasons[w] += 1
    for r in rs:
        if r: run_reasons[r["failure_mode"]] += 1
    eps[str(e)] = {**{k: v[k] for k in ("episode", "cell", "target_fruit", "used_arm", "bin", "nn_xy_m", "nn_other_fruit", "in_e2_demo", "layout_index")},
                   "seeds": rs, "n_runs_done": n_done, "n_ok": n_ok, "status": status, "fail_reasons": why}
    if status == "pass": passed.append(e)
inv = sorted(glob.glob(str(RUNS / "failed" / "*.client.log.*")))
out = {"definitions": __doc__, "n_candidates": len(C["episodes"]), "n_runs_done": len(list(RUNS.glob("E0_ep*.done"))), "n_runs_planned": 2 * len(C["episodes"]),
       "n_pass": len(passed), "passed": passed, "n_fail": sum(1 for x in eps.values() if x["status"] == "fail"), "n_pending": sum(1 for x in eps.values() if x["status"] == "pending"),
       "episode_fail_reasons": dict(reasons), "run_failure_modes": dict(run_reasons), "invalid_attempts_in_failed_dir": [pathlib.Path(p).name for p in inv], "episodes": eps}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
print({k: out[k] for k in ("n_runs_done", "n_pass", "n_fail", "n_pending", "episode_fail_reasons", "run_failure_modes")}, "passed", passed)
