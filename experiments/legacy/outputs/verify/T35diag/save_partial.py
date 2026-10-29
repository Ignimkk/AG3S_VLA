"""Merge a JSON fragment into handoff/T35.diag.verify.json under <key> (atomic write). usage: save_partial.py KEY FRAG.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, datetime
P = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T35.diag.verify.json'))
key, frag = sys.argv[1], json.loads(pathlib.Path(sys.argv[2]).read_text())
d = json.loads(P.read_text()) if P.exists() else {
    "_comment": "ag3s-verifier T35-diag (2026-09-30): offline diagnosis of the 4 failed T34 runs (E3b ep1800 r1, E3b ep1808 r1, E3b ep1808 r2, E3a ep1800 r1) from existing records. numbers and paths only, no interpretation.",
    "step": "T35-diag", "date": "2026-09-30", "status": "partial"}
d.setdefault("numbers", {})[key] = frag
d["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
tmp = P.with_suffix(".tmp"); tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False, default=float)); tmp.replace(P)
print("saved", key)
