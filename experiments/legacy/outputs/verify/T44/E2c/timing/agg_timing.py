"""T44c D-sign2 dedicated timing aggregation (single process, no other job of this verifier). p90 over pooled runs.
off = fine target-free tier build; on = fine + coarse target-free tier build (+ non-target points for `ball`, which the
pipeline computes only when the coarse tier is on; `ball_owned` computes them anyway)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, pathlib
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/E2c/timing'))
recs = []
for f in ("timing_demo.json", "timing_lb.json"):
    if (T / f).exists():
        recs += [r for r in json.load(open(T / f)) if not r.get("skipped")]
q = lambda a, p: round(float(np.percentile(a, p)), 3)  # noqa: E731


def block(rr):
    R = {k: np.concatenate([np.asarray(r["build_ms_runs"][k], np.float64) for r in rr])
         for k in ("fine_ball", "fine_bo_d3", "coarse_ball", "coarse_bo_d3", "nontarget_points")}
    series = {"ball|off": R["fine_ball"], "bo_d3|off": R["fine_bo_d3"],
              "ball|on": R["fine_ball"] + R["coarse_ball"] + R["nontarget_points"],
              "ball|on_excl_nontarget": R["fine_ball"] + R["coarse_ball"],
              "bo_d3|on": R["fine_bo_d3"] + R["coarse_bo_d3"],
              "coarse_ball": R["coarse_ball"], "coarse_bo_d3": R["coarse_bo_d3"], "nontarget_points": R["nontarget_points"]}
    o = {k: {"n": int(len(v)), "median": q(v, 50), "p90": q(v, 90), "p99": q(v, 99), "max": round(float(v.max()), 3)}
         for k, v in series.items()}
    o["p90_increase_on_vs_off_ms"] = {"ball": round(o["ball|on"]["p90"] - o["ball|off"]["p90"], 3),
                                      "bo_d3": round(o["bo_d3|on"]["p90"] - o["bo_d3|off"]["p90"], 3)}
    o["criterion4_bo_d3_on_vs_ball_on_p90_increase_ms"] = round(o["bo_d3|on"]["p90"] - o["ball|on"]["p90"], 3)
    o["criterion4_bo_d3_on_vs_ball_off_p90_increase_ms"] = round(o["bo_d3|on"]["p90"] - o["ball|off"]["p90"], 3)
    d = series["bo_d3|on"] - series["ball|on"]
    o["paired_bo_d3_on_minus_ball_on"] = {"median": q(d, 50), "p90": q(d, 90)}
    for k in ("coarse_ball", "coarse_bo_d3", "fine_bo_d3"):
        cl = [c for r in rr for c in r["classify_ms"][k] if c is not None]
        o[f"classify_ms_{k}"] = {"median": q(cl, 50), "p90": q(cl, 90)}
    la = [float(r[k]["loadavg"][0]) for r in rr for k in ("load_before", "load_after")]
    thr = [(r["load_after"]["throttled_usec"] - r["load_before"]["throttled_usec"]) / 1e6 for r in rr]
    gu = [r[k].get("gpu_util_pct") for r in rr for k in ("load_before", "load_after") if r[k].get("gpu_util_pct") is not None]
    gm = [r[k].get("gpu_mem_used_mib") for r in rr for k in ("load_before", "load_after") if r[k].get("gpu_mem_used_mib") is not None]
    o["pod_load"] = {"loadavg1_median": q(la, 50), "loadavg1_max": max(la), "cgroup_throttled_s_per_frame_median": q(thr, 50),
                     "gpu_util_pct_median": q(gu, 50), "gpu_util_pct_max": max(gu), "gpu_mem_used_mib_median": q(gm, 50)}
    return o


out = {"definitions": __doc__, "n_frames": len(recs), "reps_per_frame": len(recs[0]["build_ms_runs"]["fine_ball"]),
       "pooled": block(recs), "by_set": {s: block([r for r in recs if r["set"] == s]) for s in ("demo", "lb") if any(r["set"] == s for r in recs)}}
json.dump(out, open(T / "agg_timing.json", "w"), indent=1)
print(json.dumps(out["pooled"], indent=0))
