"""T44b dedicated timing aggregation (single process, run alone among this verifier's jobs). p90 over pooled runs."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, pathlib
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2b/timing'))
recs = []
for f in ("timing_demo.json", "timing_lb.json"):
    if (T / f).exists():
        recs += [r for r in json.load(open(T / f)) if not r.get("skipped")]
q = lambda a, p: round(float(np.percentile(a, p)), 3)
M = ("ball", "bo_d2", "bo_d3", "bo_d4")
def block(rr):
    o = {}
    for m in M:
        a = np.concatenate([r["build_ms_runs"][m] for r in rr])
        o[m] = {"n": int(len(a)), "median": q(a, 50), "p90": q(a, 90), "p99": q(a, 99), "max": round(float(a.max()), 3)}
    for m in M[1:]:
        o[m]["p90_increase_vs_ball_ms"] = round(o[m]["p90"] - o["ball"]["p90"], 3)
        d = np.concatenate([np.asarray(r["build_ms_runs"][m]) - np.asarray(r["build_ms_runs"]["ball"]) for r in rr])
        o[m]["paired_diff_median"], o[m]["paired_diff_p90"] = q(d, 50), q(d, 90)
        cl = [c for r in rr for c in r["classify_ms"][m] if c is not None]
        o[m]["classify_ms_median"], o[m]["classify_ms_p90"] = q(cl, 50), q(cl, 90)
    la = [float(r[k]["loadavg"][0]) for r in rr for k in ("load_before", "load_after")]
    thr = [(r["load_after"]["throttled_usec"] - r["load_before"]["throttled_usec"]) / 1e6 for r in rr]
    gu = [r[k].get("gpu_util_pct") for r in rr for k in ("load_before", "load_after") if r[k].get("gpu_util_pct") is not None]
    o["pod_load"] = {"loadavg1_median": q(la, 50), "loadavg1_max": max(la), "cgroup_throttled_s_per_frame_median": q(thr, 50),
                     "gpu_util_pct_median": q(gu, 50), "gpu_util_pct_max": max(gu)}
    return o
out = {"definitions": __doc__, "n_frames": len(recs), "reps_per_frame": len(recs[0]["build_ms_runs"]["ball"]),
       "methods": block(recs), "by_set": {s: block([r for r in recs if r["set"] == s]) for s in ("demo", "lb") if any(r["set"] == s for r in recs)}}
json.dump(out, open(T / "agg_timing.json", "w"), indent=1)
print(json.dumps(out["methods"], indent=0))
