"""T44 V-E2-3 timing aggregation (dedicated pass, single process). p90 over pooled runs."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, pathlib
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2/timing'))
recs = []
for f in ("timing_demo.json", "timing_lb.json"):
    if (T / f).exists():
        recs += [r for r in json.load(open(T / f)) if not r.get("skipped")]
q = lambda a, p: round(float(np.percentile(a, p)), 3)
out = {"definitions": __doc__, "n_frames": len(recs), "reps_per_frame": len(recs[0]["build_ms_runs"]["ball"]), "methods": {}, "by_set": {}}
def block(rr):
    o = {}
    for m in ("ball", "ball_owned_remove", "ball_owned_keep"):
        a = np.concatenate([r["build_ms_runs"][m] for r in rr])
        o[m] = {"n": int(len(a)), "median": q(a, 50), "p90": q(a, 90), "p99": q(a, 99), "max": round(float(a.max()), 3)}
    for m in ("ball_owned_remove", "ball_owned_keep"):
        o[m]["p90_increase_vs_ball_ms"] = round(o[m]["p90"] - o["ball"]["p90"], 3)
        d = np.concatenate([np.asarray(r["build_ms_runs"][m]) - np.asarray(r["build_ms_runs"]["ball"]) for r in rr])
        o[m]["paired_diff_median"], o[m]["paired_diff_p90"] = q(d, 50), q(d, 90)
    nt = np.concatenate([r["nontarget_points_ms"] for r in rr]); cl = [c for r in rr for c in r["classify_ms"] if c is not None]
    o["nontarget_points_ms"] = {"median": q(nt, 50), "p90": q(nt, 90)}
    o["classify_ms_ball_owned_remove"] = {"median": q(cl, 50), "p90": q(cl, 90)}
    la = [float(r[k]["loadavg"][0]) for r in rr for k in ("load_before", "load_after")]
    thr = [(r["load_after"]["throttled_usec"] - r["load_before"]["throttled_usec"]) / 1e6 for r in rr]
    per = [(r["load_after"]["nr_throttled"] - r["load_before"]["nr_throttled"]) / max(1, r["load_after"]["nr_periods"] - r["load_before"]["nr_periods"]) for r in rr]
    gu = [r[k].get("gpu_util_pct") for r in rr for k in ("load_before", "load_after") if r[k].get("gpu_util_pct") is not None]
    o["pod_load"] = {"loadavg1_median": q(la, 50), "loadavg1_max": max(la), "cgroup_throttled_s_per_frame_median": q(thr, 50),
                     "throttled_period_fraction_median": q(per, 50), "gpu_util_pct_median": q(gu, 50), "gpu_util_pct_max": max(gu)}
    return o
out["methods"] = block(recs)
for s in ("demo", "lb"):
    rr = [r for r in recs if r["set"] == s]
    if rr:
        out["by_set"][s] = block(rr)
json.dump(out, open(T / "agg_timing.json", "w"), indent=1)
print(json.dumps(out["methods"], indent=0)); print({s: {m: v.get("p90_increase_vs_ball_ms") for m, v in b.items() if isinstance(v, dict)} for s, b in out["by_set"].items()})
