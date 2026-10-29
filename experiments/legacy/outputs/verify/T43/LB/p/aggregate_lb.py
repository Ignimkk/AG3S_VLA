"""T43 LB aggregation (verifier, 2026-10-09). Numbers only. Reads runs/E0 (finished .done runs), candidates_lb.json, failed/ attempts.
usage: aggregate_lb.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, re, pathlib, glob, statistics
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/LB/p'))
import plan_lb as PL
import numpy as np
LB = PL.LB; RUNS = PL.RUNS; CAND = PL.CAND
out = {"groups": {}, "episodes": {}}
for (f, a), ranked in PL.groups().items():
    g = f"{f}_{a}"; tried = []; conf = []
    fm = {}
    for rank, e in enumerate(ranked):
        r = PL.reps(e)
        if all(x is None for x in r):
            continue
        info = CAND["episodes"][str(e)]
        seeds = []
        for k, x in enumerate(r, 1):
            s = 10 * e + k
            if x is None:
                continue
            seeds.append({"rep": k, "seed": s, "E0_ok_B": x["ok"], "success": x["success"], "grasp": x["grasp"], "place": x["place"],
                          "robust_B": x["robust"], "robust_strict_A": x["robust_strict"], "failure_mode": x["failure_mode"],
                          "run_dir": str(RUNS / f"E0_ep{e}_s{s}")})
            fm[x["failure_mode"]] = fm.get(x["failure_mode"], 0) + 1
        n_ok = sum(1 for s in seeds if s["E0_ok_B"])
        done3 = len(seeds) == 3
        out["episodes"][str(e)] = {"episode": e, "split": info["split"], "fruit": f, "arm": a, "rank_in_pool": rank + 1, "crowd_score_m": info["score_m"],
                                   "nn_target_m": info["nn_target_m"], "gap_target_crate_m": info["gap_target_crate_m"],
                                   "gap_crate_others_m": info["gap_crate_others_m"], "layout_index": info["layout_index"],
                                   "seeds": seeds, "n_ok_B": n_ok, "n_seeds_run": len(seeds), "three_of_three": bool(done3 and n_ok == 3),
                                   "n_ok_strict_A": sum(1 for s in seeds if s["robust_strict_A"] and s["success"]),
                                   "stage_reached": len(seeds)}
        tried.append(e)
        if done3 and n_ok == 3:
            conf.append(e)
    conf = sorted(conf, key=lambda e: (-out["episodes"][str(e)]["crowd_score_m"], e))
    rep1 = [e for e in tried if out["episodes"][str(e)]["seeds"][0]["E0_ok_B"]]
    rep12 = [e for e in rep1 if len(out["episodes"][str(e)]["seeds"]) >= 2 and out["episodes"][str(e)]["seeds"][1]["E0_ok_B"]]
    out["groups"][g] = {"pool_size": CAND["candidates"][g]["pool_size"], "tried": len(tried), "tried_episodes": tried, "rep1_ok": len(rep1),
                        "rep1_2_ok": len(rep12), "three_of_three": conf, "adopted": conf[:2], "extra_3of3_not_adopted": conf[2:],
                        "n_runs_valid": sum(len(out["episodes"][str(e)]["seeds"]) for e in tried), "failure_modes_all_valid_runs": fm}
# invalid attempts
inv = []
for f in sorted(glob.glob(str(RUNS / "failed" / "*.client.log.*"))):
    txt = pathlib.Path(f).read_text(errors="replace"); name = pathlib.Path(f).name
    se = [int(x) for x in re.findall(r"\[safe\][^\n]*?\berror (\d+)", txt)]
    lastexc = [l for l in txt.splitlines() if re.search(r"Error|Exception|exit=", l)][-2:]
    inv.append({"file": name, "safe_error_counts": se, "last_lines": lastexc,
                "keepalive_ping_timeout": "keepalive ping timeout" in txt, "connection_closed": "ConnectionClosed" in txt})
out["invalid_attempts"] = {"count": len(inv), "items": inv, "all_rerun_and_done": all((RUNS / (re.sub(r"\.client\.log\..*", "", i["file"]) + ".done")).exists() for i in inv)}
done = sorted(glob.glob(str(RUNS / "E0_ep*.done")))
out["n_valid_runs_done"] = len(done)
# pod load / memory / GPU
L = [l.split() for l in (LB / "runs" / "loadavg_samples.txt").read_text().splitlines() if l.strip()]
la1 = [float(x[1]) for x in L]; ws = [int(t.split("=")[1]) for x in L for t in x if t.startswith("wset=")]; gp = [int(t.split("=")[1]) for x in L for t in x if t.startswith("gpu_mib=")]
out["pod"] = {"n_samples": len(L), "first_sample_utc": L[0][0], "last_sample_utc": L[-1][0], "loadavg1_min": min(la1), "loadavg1_median": statistics.median(la1),
              "loadavg1_max": max(la1), "loadavg1_first": la1[0], "working_set_gib_max": max(ws) / 2**30, "gpu_used_mib_max_sampled_60s": max(gp)}
gm = 0; srv = []
for d in sorted(glob.glob(str(RUNS / "server_*"))):
    gs = pathlib.Path(d) / "gpu_samples.txt"; b = pathlib.Path(d) / "gpu_used_before_mib"
    if gs.exists():
        v = [int(l.split()[1]) for l in gs.read_text().splitlines() if len(l.split()) == 2]
        gm = max(gm, max(v)); srv.append({"dir": pathlib.Path(d).name, "gpu_total_used_mib_max": max(v),
                                          "gpu_used_before_mib": int(b.read_text()) if b.exists() else None})
out["servers"] = {"n_servers_started": len(srv), "items": srv, "gpu_total_used_mib_max_2s": gm}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=float))
for g, v in out["groups"].items():
    print(g, "tried", v["tried"], "rep1ok", v["rep1_ok"], "rep12", v["rep1_2_ok"], "3/3", v["three_of_three"], v["failure_modes_all_valid_runs"])
print("invalid", out["invalid_attempts"]["count"], "valid runs", out["n_valid_runs_done"], out["pod"], out["servers"]["gpu_total_used_mib_max_2s"])
