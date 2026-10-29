"""T43 Z5 — per-run analysis of E0 runs without an obstacle, using the T43W/T39 rules unchanged: analyze_t39.run_metrics (success / grasp / place by the
analyze_e0 rule; HOLD chunks) and the trailing / longest HOLD counters of analyze_t43w.one. Output has the keys y_metrics.run reads.
usage: analyze_z5.py OUT_JSON GROUP RUN_DIR [RUN_DIR ...]      GROUP is the key prefix, e.g. Z5 or T39 -> key E0/B/<name> inside GROUP"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, re, sys, os
os.environ.setdefault("MUJOCO_GL", "osmesa")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/W/p'))
import analyze_t39 as A39  # noqa: E402
NAME = re.compile(r"^E0_ep(\d+)_s(\d+)$")
out = pathlib.Path(sys.argv[1]); res = {"group": sys.argv[2], "runs": {}}
for rd in map(pathlib.Path, sys.argv[3:]):
    m = NAME.match(rd.name); ep, seed = int(m.group(1)), int(m.group(2))
    r, _ = A39.run_metrics("E0", ep, "s", seed, rd, None, set(), False)
    r["arm"] = "E0"; r["stage"] = "B"
    trail = 0
    for c in reversed(r["chunks"]):
        if c["hold"]:
            trail += 1
        else:
            break
    r["trailing_hold_chunks"] = trail
    r.pop("chunks", None) if False else None
    res["runs"][f"E0/B/{rd.name}"] = r
    print(rd.name, "S" if r["success"] else "-", "g", r["grasp"], "p", r["place"], "gt", r["grasp_t"], "pt", r["place_t"], flush=True)
out.write_text(json.dumps(res, indent=1, default=float))
