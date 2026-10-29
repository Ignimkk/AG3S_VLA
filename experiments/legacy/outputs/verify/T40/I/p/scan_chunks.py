"""T40 I: map server constraint chunks -> (episode seed, t_step) for both E3b servers."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, os, sys
import numpy as np
ROOT = _ag3s_workspace_path('outputs/verify/T40/G/runs/E3b')
out = {}
for srv in sorted(glob.glob(f"{ROOT}/server_*/server_constraints/run_0000")):
    rows = []
    for f in sorted(glob.glob(f"{srv}/chunk_*.npz")):
        z = np.load(f, allow_pickle=True)
        s = json.loads(str(z["summary_json"]))
        ef = s.get("exec_feedback") or {}
        v = s.get("verdict") or {}
        to = s.get("to") or {}
        rows.append(dict(file=os.path.basename(f), keys=len(z.files),
                         has_tf=[k for k in z.files if "target" in k or "free" in k],
                         chunk_index=int(z["chunk_index"]), t_step=int(z["t_step"]),
                         seed=(s.get("policy_seed") or {}).get("episode"),
                         status=s.get("status"), grounding=s.get("grounding_status"),
                         to_status=to.get("status"),
                         clearance_min_mm=s.get("clearance_min_mm"),
                         exec_path=s.get("execution_path"),
                         verdict=[(r.get("kind"), r.get("action")) for r in v.get("reasons", [])],
                         fb_seq=ef.get("seq"), fb_n_exec=ef.get("n_exec"), fb_hold_kind=ef.get("hold_kind"),
                         fb_executed_chunk=ef.get("executed_chunk"), fb_t_step_start=ef.get("t_step_start")))
    out[srv] = rows
    print(srv, len(rows), file=sys.stderr)
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T40/I/raw/chunk_index.json'), "w"), indent=0)
