"""T43 CT — list the H-stage runs with .done (re-runnable) and split into jobs of one snapshot each (<= 14 runs). usage: make_jobs.py OUT_DIR"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, sys, re
T = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43')); out = pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
ROOTS = {"E0": T/"W/runs/E0", "V1": T/"W/runs/V1", "V3": T/"W/runs/V3", "V4": T/"W/runs/V4", "V5": T/"W/runs/V5", "V5S": T/"W/runs_blind/V5S",
         "V5B": T/"W/runs_blind/V5B", "V5D": T/"Z3V/runs/V5D", "V7g": T/"V7g/runs/V7g"}
by = {}
for arm, r in ROOTS.items():
    for d in sorted((r/"H").glob(f"{arm}_ep*_s*.done")):
        rd = d.with_suffix("")
        snap = re.match(r"(/mnt/dev/[^/]+)/", json.load(open(rd/"manifest.json"))["scene"]["model_xml"]).group(1)
        by.setdefault(snap, []).append(str(rd))
for i, (snap, L) in enumerate(sorted(by.items())):
    (out/f"job_{i}.snap").write_text(snap); (out/f"job_{i}.list").write_text("\n".join(L) + "\n")
    print(i, snap, len(L))
