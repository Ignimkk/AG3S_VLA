"""Merge a JSON fragment into AG3S/ag3s/docs/handoff/T31.diag.verify.json (partial saves)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, datetime, os
P = _ag3s_workspace_path('AG3S/ag3s/docs/handoff/T31.diag.verify.json')
def save(key, value):
    d = json.load(open(P)) if os.path.exists(P) else {"_comment": "ag3s-verifier. numbers and paths only.", "step": "T31-diag", "date": "2026-09-29"}
    d[key] = value
    d["updated_at"] = datetime.datetime.utcnow().isoformat() + "Z"
    json.dump(d, open(P, "w"), indent=1, ensure_ascii=False)
if __name__ == "__main__":
    save(sys.argv[1], json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None)
