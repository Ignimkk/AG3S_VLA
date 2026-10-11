"""T44e dedicated timing aggregation. Per frame 10 reps after 2 warm-ups, interleaved, gt_label.
Series: fine_<m>, coarse_<m>_<v> (v = guard, keep, exclude, exclude_target), support_evidence (pipeline-side `_support_seed_evidence`, exclude/exclude_target only).
off_total = fine; on_total(v) = fine + coarse_v (+ support_evidence for exclude / exclude_target); p50/p90 pooled over frames x reps."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, pathlib
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2e/timing'))
recs = []
for f in ("timing_demo.json", "timing_lb.json"):
    if (T / f).exists(): recs += [r for r in json.load(open(T / f)) if not r.get("skipped")]
q = lambda a, p: round(float(np.percentile(a, p)), 3)  # noqa: E731
def st(v): return {"n": int(len(v)), "median": q(v, 50), "p90": q(v, 90), "p99": q(v, 99), "max": round(float(v.max()), 3)}
CV = ("guard", "keep", "exclude", "exclude_target")
def block(rr):
    R = {k: np.concatenate([np.asarray(r["build_ms_runs"][k], np.float64) for r in rr]) for k in rr[0]["build_ms_runs"]}
    o = {"n_frames": len(rr), "support_evidence": st(R["support_evidence"])}
    for m in ("ball", "bo_d3"):
        f = R[f"fine_{m}"]
        o[m] = {"fine": st(f)}
        for v in CV:
            c = R[f"coarse_{m}_{v}"]
            ev = R["support_evidence"] if v in ("exclude", "exclude_target") else 0.0
            o[m][f"coarse_{v}"] = st(c)
            o[m][f"on_total_{v}"] = st(f + c + ev)
            o[m][f"p90_increase_total_{v}_vs_off_ms"] = round(q(f + c + ev, 90) - q(f, 90), 3)
            if v != "keep":
                o[m][f"paired_{v}_minus_keep"] = st(c - R[f"coarse_{m}_keep"]) | {"min": round(float((c - R[f"coarse_{m}_keep"]).min()), 3)}
    la = [float(r[k]["loadavg"][0]) for r in rr for k in ("load_before", "load_after")]
    o["pod_load"] = {"loadavg1_median": q(la, 50), "loadavg1_max": max(la)}
    return o
out = {"all": block(recs)}
for s in ("demo", "lb"):
    rr = [r for r in recs if r["set"] == s]
    if rr: out[s] = block(rr)
json.dump(out, open(T / "agg_timing.json", "w"), indent=1)
print(json.dumps({k: {m: {v: out[k][m][f"coarse_{v}"]["median"] for v in CV} for m in ("ball", "bo_d3")} for k in out}, indent=1))
