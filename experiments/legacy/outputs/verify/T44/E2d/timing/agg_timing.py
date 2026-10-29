"""T44c-A6 dedicated timing aggregation (single process). Per frame 10 reps after 2 warm-ups, interleaved, gt_label.
Series: fine_<m> (off tier), coarse_<m>_guard (A5), coarse_<m>_plane (A6); paired delta plane - guard per rep; p50/p90 pooled."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, pathlib
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2d/timing'))
recs = []
for f in ("timing_demo.json", "timing_lb.json"):
    if (T / f).exists(): recs += [r for r in json.load(open(T / f)) if not r.get("skipped")]
q = lambda a, p: round(float(np.percentile(a, p)), 3)  # noqa: E731
def st(v): return {"n": int(len(v)), "median": q(v, 50), "p90": q(v, 90), "p99": q(v, 99), "max": round(float(v.max()), 3)}
def block(rr):
    R = {k: np.concatenate([np.asarray(r["build_ms_runs"][k], np.float64) for r in rr]) for k in rr[0]["build_ms_runs"]}
    o = {"n_frames": len(rr)}
    for m in ("ball", "bo_d3"):
        f, g, p = R[f"fine_{m}"], R[f"coarse_{m}_guard"], R[f"coarse_{m}_plane"]
        o[m] = {"fine": st(f), "coarse_guard": st(g), "coarse_plane": st(p), "paired_plane_minus_guard": st(p - g) | {"min": round(float((p - g).min()), 3)},
                "on_guard_total": st(f + g), "on_plane_total": st(f + p), "off_total": st(f),
                "p90_increase_plane_vs_off_ms": round(q(f + p, 90) - q(f, 90), 3), "p90_increase_guard_vs_off_ms": round(q(f + g, 90) - q(f, 90), 3)}
    la = [float(r[k]["loadavg"][0]) for r in rr for k in ("load_before", "load_after")]
    o["pod_load"] = {"loadavg1_median": q(la, 50), "loadavg1_max": max(la)}
    return o
out = {"all": block(recs)}
for s in ("demo", "lb"):
    rr = [r for r in recs if r["set"] == s]
    if rr: out[s] = block(rr)
json.dump(out, open(T / "agg_timing.json", "w"), indent=1)
print(json.dumps({k: {m: out[k][m]["paired_plane_minus_guard"] for m in ("ball", "bo_d3")} for k in out}, indent=1))
