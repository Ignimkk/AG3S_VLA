"""Duplicate-run check: for each episode, compare r1 vs r2 local planning rows (same seq): max |actions diff|, max |qpos diff|,
and apple z trajectories. Identical (0.0) = the two runs are the same sample. usage: dup_check.py -> p/dup_check.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
import numpy as np
out = {}
for tag, root in (("T34_E3a", "T34/E3a_rec"), ("T34_E3b", "T34/E3b_rec"),
                  ):
    R = pathlib.Path(_ag3s_workspace_path('outputs/verify')) / root
    if not R.exists():
        continue
    for ep in (1807, 1800, 1808):
        runs = sorted(p for p in R.glob(f"*_ep{ep}_r*") if p.is_dir())
        if len(runs) < 2:
            continue
        P = {}
        for r in runs:
            rows = [json.loads(l) for l in (r / "frames.jsonl").read_text().splitlines() if l.strip()]
            P[r.name] = {x["seq"]: x for x in rows if x["kind"] == "planning"}
        names = list(P)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = P[names[i]], P[names[j]]
                common = sorted(set(a) & set(b))
                da = max((float(np.abs(np.asarray(a[s]["actions"]) - np.asarray(b[s]["actions"])).max()) for s in common if a[s].get("actions") is not None and b[s].get("actions") is not None), default=None)
                dq = max(float(np.abs(np.asarray(a[s]["qpos"]) - np.asarray(b[s]["qpos"])).max()) for s in common)
                first = next((s for s in common if float(np.abs(np.asarray(a[s]["qpos"]) - np.asarray(b[s]["qpos"])).max()) > 1e-9), None)
                out[f"{tag}_ep{ep}_{names[i].split('_')[-1]}_vs_{names[j].split('_')[-1]}"] = {
                    "n_common_planning_rows": len(common), "max_abs_actions_diff": da, "max_abs_qpos_diff": dq,
                    "first_seq_with_qpos_diff": first, "identical": bool(dq == 0.0 and (da == 0.0 or da is None))}
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T34/p/dup_check.json'), "w"), indent=1)
for k, v in out.items():
    print(k, v)
