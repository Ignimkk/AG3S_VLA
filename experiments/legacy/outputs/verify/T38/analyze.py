"""T38 Phase A — reduce the replay passes to the numbers in T38.profile.verify.json (no interpretation).

reads  outputs/verify/T38/pass{warm,T,S,C,P,W,X}_<run>.jsonl, prof/*.prof, torchprof/*_keyavg.json, frames/<run>.json,
       outputs/verify/T34/E3b_rec/server_8206/server_constraints/run_0000/*.npz (recorded profile_ms, all 449 chunks)
writes outputs/verify/T38/summary_<run>.json
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import collections, glob, json, pathlib, pstats, sys
import numpy as np

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38'))
run = sys.argv[1] if len(sys.argv) > 1 else "E3b_ep1807_r2"
label = sys.argv[2] if len(sys.argv) > 2 else ""
REC = pathlib.Path(_ag3s_workspace_path('outputs/verify/T34/E3b_rec/server_8206/server_constraints/run_0000'))
STAGES = ["scene_reconstruction", "support_surface", "attention_lifting", "target_grounding", "esdf",
          "constraint_generation"]


def load(p):
    f = OUT / f"pass{p}_{run}{label}.jsonl"
    if not f.exists():
        f = OUT / f"pass{p}_{run}.jsonl"   # e.g. the census pass C lives only in the first run
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []


def stat(v):
    v = np.asarray([x for x in v if x is not None], float)
    if not v.size:
        return None
    return {"n": int(v.size), "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max())}


def stage_rows(rows, key="profile_ms"):
    out = {}
    for s in STAGES:
        out[s] = stat([(r[key] or {}).get(s) for r in rows])
    out["constraint_generation_minus_esdf"] = stat(
        [(r[key] or {}).get("constraint_generation", 0) - (r[key] or {}).get("esdf", 0) for r in rows])
    out["sum_top_level_stages"] = stat([sum((r[key] or {}).get(s, 0) for s in STAGES if s != "esdf") for r in rows])
    return out


def timing_rows(rows):
    out = {}
    for k in ("infer", "ag3s", "trajopt", "total"):
        out[k] = stat([r["timing_ms"].get(k) for r in rows])
    for k in ("linearize", "check", "assemble", "qp"):
        out["to_" + k] = stat([(r["to_timing_ms"] or {}).get(k) for r in rows])
    out["to_sum4"] = stat([sum((r["to_timing_ms"] or {}).get(k, 0) for k in ("linearize", "check", "assemble", "qp")) for r in rows])
    out["trajopt_minus_sum4"] = stat([r["timing_ms"].get("trajopt", 0) - sum((r["to_timing_ms"] or {}).get(k, 0) for k in ("linearize", "check", "assemble", "qp")) for r in rows])
    out["ag3s_minus_stages"] = stat([r["timing_ms"].get("ag3s", 0) - sum((r["profile_ms"] or {}).get(s, 0) for s in STAGES if s != "esdf") for r in rows])
    out["wall_ms"] = stat([r["wall_ms"] for r in rows])
    out["ag3s_plus_trajopt"] = stat([r["timing_ms"].get("ag3s", 0) + r["timing_ms"].get("trajopt", 0) for r in rows])
    return out


def fn_rows(rows, names=None):
    acc = collections.defaultdict(list); calls = collections.defaultdict(list)
    for r in rows:
        for k, v in r["fn"].items():
            acc[k].append(v["ms"]); calls[k].append(v["n"])
    keys = names or sorted(acc, key=lambda k: -np.median(acc[k]))
    return {k: {"median_ms": float(np.median(acc[k])), "p90_ms": float(np.percentile(acc[k], 90)),
                "calls_per_chunk_median": float(np.median(calls[k])), "chunks": len(acc[k])} for k in keys if k in acc}


def device_tag(fname, func):
    s = f"{fname} {func}"
    if "threading.py" in fname or "_thread" in func:
        return "CPU (threads: cKDTree workers=-1)"
    if "/benchmark/" in fname:
        return "CPU (python, benchmark)"
    if "curobo" in s or "/warp/" in s or "warp." in s:
        return "GPU (cuRobo/warp)"
    if "torch" in s:
        return "GPU (torch)"
    if "scipy" in s or "_ckdtree" in s or "cKDTree" in s or "csgraph" in s:
        return "CPU (scipy)"
    if "numpy" in s or "ndarray" in s:
        return "CPU (numpy)"
    if "casadi" in s:
        return "CPU (casadi)"
    if "osqp" in s:
        return "CPU (osqp)"
    if "mujoco" in s:
        return "CPU (mujoco)"
    if "T38/replay_profile.py" in fname:
        return "harness wrapper"
    return "CPU (python/builtin)"


def short(fname, line, func):
    f = fname.replace("${AG3S_PROJECT_ROOT}-o4/", "").replace(_ag3s_workspace_path('legacy/.venv-openpi-live/lib/python3.11/site-packages/'), "site:")
    f = f.replace(_ag3s_workspace_path('third_party/curobo/'), "third_party/curobo:")
    return f"{f}:{line}({func})" if line else f"{f}({func})" if f != "~" else func


def prof_table(pattern, top=15):
    files = sorted(glob.glob(str(OUT / "prof" / pattern)))
    if not files:
        return None
    st = pstats.Stats(files[0])
    for f in files[1:]:
        st.add(f)
    total_tt = sum(v[2] for v in st.stats.values())
    rows = []
    by_dev = collections.defaultdict(float)
    for (fname, line, func), (cc, nc, tt, ct, callers) in st.stats.items():
        tag = device_tag(fname, func)
        by_dev[tag] += tt
        rows.append({"function": short(fname, line, func), "ncalls": int(nc), "primitive_calls": int(cc),
                     "tottime_ms": tt * 1e3 / len(files), "cumtime_ms": ct * 1e3 / len(files), "device": tag})
    cum = sorted(rows, key=lambda r: -r["cumtime_ms"])[:top]
    tot = sorted(rows, key=lambda r: -r["tottime_ms"])[:top]
    for r in cum + tot:
        r["ncalls_per_chunk"] = r["ncalls"] / len(files)
    return {"n_profiles": len(files), "files": [pathlib.Path(f).name for f in files],
            "total_self_ms_per_chunk": total_tt * 1e3 / len(files),
            "self_time_by_device_ms_per_chunk": {k: v * 1e3 / len(files) for k, v in sorted(by_dev.items(), key=lambda kv: -kv[1])},
            "top_cumulative": cum, "top_tottime": tot}


def recorded_all():
    vals = collections.defaultdict(list); to = collections.defaultdict(list)
    for f in sorted(REC.glob("chunk_*.npz")):
        z = np.load(f, allow_pickle=False)
        s = json.loads(str(z["summary_json"]))
        for k, v in (s.get("profile_ms") or {}).items():
            vals[k].append(v)
        pm = s.get("profile_ms") or {}
        if "constraint_generation" in pm:
            vals["constraint_generation_minus_esdf"].append(pm["constraint_generation"] - pm.get("esdf", 0.0))
        tm = ((s.get("to") or {}).get("metrics") or {}).get("timing_ms")
        if isinstance(tm, str):
            tm = json.loads(tm.replace("'", '"'))
        for k, v in (tm or {}).items():
            to[k].append(v)
    return {"n_chunks": len(next(iter(vals.values()))), "profile_ms": {k: stat(v) for k, v in vals.items()},
            "to_timing_ms": {k: stat(v) for k, v in to.items()}}


def census(rows):
    out = []
    for r in rows:
        kd = r.get("kd") or []
        builds = [k for k in kd if k["op"] == "build"]
        by_site = collections.Counter((k["ours"] or k["site"], k["stage"]) for k in builds)
        qs = collections.Counter((k["op"], k["ours"] or k["site"], k["stage"]) for k in kd if k["op"] != "build")
        hashes = collections.defaultdict(list)
        for k in builds:
            hashes[k["hash"]].append(f"{k['ours'] or k['site']} [{k['stage']}]")
        dup = {h: v for h, v in hashes.items() if len(v) > 1}
        cp = collections.defaultdict(lambda: {"n": 0, "bytes": 0})
        for c in r.get("copies") or []:
            key = f"{c['dir']} {c['ours'] or c['site']} {c['kind']}"
            cp[key]["n"] += 1; cp[key]["bytes"] += c["bytes"]
        fs = r.get("fn_sizes") or {}
        out.append({
            "seq": r["seq"], "phase_label": r["phase_label"], "grasp_state": r["grasp_state"],
            "sizes": r["sizes"],
            "kd_builds": len(builds), "kd_build_points_total": int(sum(k["n"] for k in builds)),
            "kd_build_ms_total": float(sum(k["ms"] for k in builds)),
            "kd_builds_by_site": {f"{a} [{b}]": n for (a, b), n in by_site.items()},
            "kd_build_sizes_by_site": {f"{a} [{b}]": [k["n"] for k in builds if (k["ours"] or k["site"]) == a and k["stage"] == b] for (a, b) in by_site},
            "kd_queries_by_site": {f"{a} {b} [{c}]": n for (a, b, c), n in qs.items()},
            "kd_query_ms_by_site": {f"{a} {b} [{c}]": float(sum(k["ms"] for k in kd if k["op"] == a and (k["ours"] or k["site"]) == b and k["stage"] == c)) for (a, b, c) in qs},
            "kd_same_input_built_more_than_once": dup,
            "copies": dict(cp),
            "h2d_bytes": int(sum(c["bytes"] for c in r.get("copies") or [] if c["dir"] == "H2D")),
            "d2h_bytes": int(sum(c["bytes"] for c in r.get("copies") or [] if c["dir"] == "D2H")),
            "n_h2d": sum(1 for c in r.get("copies") or [] if c["dir"] == "H2D"),
            "n_d2h": sum(1 for c in r.get("copies") or [] if c["dir"] == "D2H"),
            "fn_sizes": fs,
            "fn_calls": {k: v["n"] for k, v in r["fn"].items()},
        })
    return out


def torchprof(pattern):
    res = {}
    for f in sorted(glob.glob(str(OUT / "torchprof" / pattern))):
        rows = json.loads(pathlib.Path(f).read_text())
        mem = [r for r in rows if r["key"].startswith("Memcpy") or r["key"].startswith("Memset")]
        kern_dev = sum(r["self_device_time_total_us"] for r in rows if not r["key"].startswith(("Memcpy", "Memset")))
        top = sorted(rows, key=lambda r: -r["self_device_time_total_us"])[:15]
        res[pathlib.Path(f).name] = {
            "memcpy": [{k: r[k] for k in ("key", "count", "self_device_time_total_us")} for r in mem],
            "device_self_time_total_ms": float(sum(r["self_device_time_total_us"] for r in rows)) / 1e3,
            "kernel_self_time_ms": kern_dev / 1e3,
            "top_device_ops": [{k: r[k] for k in ("key", "count", "self_device_time_total_us")} for r in top]}
    return res


def main():
    warm, T, S, C, P, W, X = (load(p) for p in ("_warm", "T", "S", "C", "P", "W", "X"))
    if not warm:
        warm = [json.loads(l) for l in (OUT / f"pass_warm_{run}{label}.jsonl").read_text().splitlines() if l.strip()]
    idx = json.loads((OUT / "frames" / f"{run}.json").read_text())
    rec_rows = [{"profile_ms": s["recorded_profile_ms"]} for s in idx["seqs"]]
    steady = lambda rows: [r for r in rows if r["seq"] != 1]
    summary = {
        "run": run,
        "recorded_all_449": recorded_all(),
        "recorded_this_run": {"all_75": stage_rows(rec_rows), "seq2_75": stage_rows(rec_rows[1:]),
                              "seq1": rec_rows[0]["profile_ms"]},
        "warm": [{"seq": r["seq"], "wall_ms": r["wall_ms"], "profile_ms": r["profile_ms"], "timing_ms": r["timing_ms"],
                  "to_timing_ms": r["to_timing_ms"]} for r in warm],
    }
    for name, rows in (("T", T), ("S", S), ("C", C), ("P", P), ("W", W), ("X", X)):
        if not rows:
            continue
        summary[f"pass_{name}"] = {
            "stages_profile_ms_seq2_75": stage_rows(steady(rows)),
            "stages_harness_ms_seq2_75": stage_rows(steady(rows), key="stage_ms_harness"),
            "timing_seq2_75": timing_rows(steady(rows)),
            "seq1": {"profile_ms": rows[0]["profile_ms"], "timing_ms": rows[0]["timing_ms"], "to_timing_ms": rows[0]["to_timing_ms"]},
            "functions_seq2_75": fn_rows(steady(rows)),
            "grasp_state_match": sum(1 for r in rows if r["grasp_state"] == r["recorded_grasp_state"]),
            "grasp_state_mismatch": [(r["seq"], r["grasp_state"], r["recorded_grasp_state"]) for r in rows if r["grasp_state"] != r["recorded_grasp_state"]],
            "n_clusters_match": sum(1 for r in rows if r["n_clusters"] == r["recorded_n_clusters"]),
            "n_clusters_mismatch": [(r["seq"], r["n_clusters"], r["recorded_n_clusters"]) for r in rows if r["n_clusters"] != r["recorded_n_clusters"]],
            "has_target": sum(1 for r in rows if r["has_target"]),
            "n_chunks": len(rows),
            "torch_max_alloc_mib": max(r["torch_max_alloc_mib"] for r in rows),
            "torch_max_reserved_mib": max(r["torch_max_reserved_mib"] for r in rows),
            "per_chunk": [{"seq": r["seq"], "phase_label": r["phase_label"], "grasp_state": r["grasp_state"],
                           "profile_ms": r["profile_ms"], "timing_ms": r["timing_ms"], "to_timing_ms": r["to_timing_ms"],
                           "stage_ms_harness": r["stage_ms_harness"]} for r in rows],
        }
    if C:
        summary["census"] = census(C)
    tabs = {}
    for st in ("scene_reconstruction", "support_surface", "target_grounding", "constraint_generation", "attention_lifting"):
        t = prof_table(f"P_seq*_{st}.prof")
        if t:
            tabs[st] = t
    t = prof_table("W_seq*_infer.prof", top=30)
    if t:
        tabs["whole_infer"] = t
    summary["cprofile"] = tabs
    summary["torchprof"] = torchprof("X_seq*_keyavg.json")
    meta = OUT / f"meta_{run}{label}.json"
    summary["meta"] = json.loads(meta.read_text()) if meta.exists() else None
    (OUT / f"summary_{run}{label}.json").write_text(json.dumps(summary, indent=1, default=str))
    print("wrote", OUT / f"summary_{run}{label}.json")


if __name__ == "__main__":
    main()
