"""append 한 항목을 handoff/T23.verify.partial.json 에."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, datetime, pathlib
P = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T23.verify.partial.json'))
def append(run, raw, summary):
    d = json.loads(P.read_text()) if P.exists() else {"step": "T23", "writer": "ag3s-verifier", "entries": []}
    d["entries"].append({"run": run, "raw": raw, "written": datetime.datetime.now().isoformat(timespec="seconds"), "summary": summary})
    P.write_text(json.dumps(d, indent=1, ensure_ascii=False))
if __name__ == "__main__":
    append(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]))
