"""Merge a JSON fragment into T28.verify.partial.json under phase3.<key> (atomic write)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib, datetime
P = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T28.verify.partial.json'))
key, frag = sys.argv[1], json.loads(pathlib.Path(sys.argv[2]).read_text())
d = json.loads(P.read_text())
p3 = d.setdefault("phase3", {"_comment": "ag3s-verifier Phase 3 (2026-09-29): offline 0a-0f + closed-loop E3a/E3b. numbers and paths only, no interpretation."})
p3[key] = frag
p3["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
d["updated_at"] = p3["updated_at"]
tmp = P.with_suffix(".tmp"); tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False)); tmp.replace(P)
print("saved phase3." + key)
