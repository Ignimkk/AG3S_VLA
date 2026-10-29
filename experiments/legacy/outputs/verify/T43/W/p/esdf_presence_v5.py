"""T43W V5 rerun / blind round: per run, number of server chunk records (the run's paired segment) whose summary_json has esdf_stats,
and number whose profile_ms has an 'esdf' key; plus per server log count of 'no scene' lines. Numbers only.
usage: esdf_presence_v5.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, re, sys, glob
import numpy as np
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/W'))
out = {"runs": {}, "server_logs_no_scene_lines": {}}
for an, arms in (("analysis/analysis.json", ("V4", "V5")), ("analysis_blind/analysis.json", ("V5B", "V5S"))):
    A = json.loads((V / an).read_text())["runs"]
    for k, r in A.items():
        if r["arm"] not in arms or not r.get("server_segment"):
            continue
        f0 = pathlib.Path(r["server_segment"]["first"]); n = r["server_segment"]["n"]
        i0 = int(re.search(r"chunk_(\d+)", f0.name).group(1))
        n_es = n_prof = 0; n_hold_ns = 0; keys = set()
        for i in range(i0, i0 + n):
            z = np.load(f0.parent / f"chunk_{i:05d}.npz", allow_pickle=True); s = json.loads(str(z["summary_json"]))
            n_es += int(bool(s.get("esdf_stats"))); pm = s.get("profile_ms") or {}
            n_prof += int("esdf" in pm); keys |= set(pm)
        out["runs"][k] = {"n_chunks": n, "n_with_esdf_stats": n_es, "n_with_profile_ms_esdf": n_prof}
for d in sorted(glob.glob(str(V / "runs/V5/server_*")) + glob.glob(str(V / "runs_blind/V5[BS]/server_*"))):
    p = pathlib.Path(d)
    lg = [x for x in p.glob("*.log")]
    c = 0; tot = 0
    for l in lg:
        for line in l.read_text(errors="ignore").splitlines():
            if "uncertified: no scene" in line:
                c += 1
    out["server_logs_no_scene_lines"][str(p.relative_to(V))] = {"logs": [x.name for x in lg], "no_scene_lines": c}
out["summary"] = {arm: {"runs": sum(1 for v in [x for kk, x in out["runs"].items() if kk.split("/")[2].startswith(arm + "_")]),
                        "chunks": sum(x["n_chunks"] for kk, x in out["runs"].items() if kk.split("/")[2].startswith(arm + "_")),
                        "with_esdf_stats": sum(x["n_with_esdf_stats"] for kk, x in out["runs"].items() if kk.split("/")[2].startswith(arm + "_")),
                        "with_profile_ms_esdf": sum(x["n_with_profile_ms_esdf"] for kk, x in out["runs"].items() if kk.split("/")[2].startswith(arm + "_"))}
                  for arm in ("V4", "V5", "V5B", "V5S")}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1)); print(json.dumps(out["summary"]), json.dumps(out["server_logs_no_scene_lines"]))
