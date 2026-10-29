"""Merge a JSON fragment into T37.verify.partial.json under <key> (atomic write). usage: save_partial.py KEY FRAG.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, datetime
P = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T37.verify.partial.json'))
key, frag = sys.argv[1], json.loads(pathlib.Path(sys.argv[2]).read_text())
d = json.loads(P.read_text()) if P.exists() else {
    "_comment": "ag3s-verifier T37 (2026-09-30): verification of T37 K1 (continuity term off while grasp latch closing/held) — baselines, precheck, closed-loop E3a/E3b. numbers and paths only, no interpretation.",
    "step": "T37", "date": "2026-09-30"}
d[key] = frag
d["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
tmp = P.with_suffix(".tmp"); tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False)); tmp.replace(P)
print("saved", key)
