"""V11ABC: server-recorded servo_accel (C) and deflection_rate (B) metrics per chunk (first chunk of each run excluded as in extra_v11.raw_to). numbers only. usage: c_records.py OUT_JSON"""
from common.workspace import workspace_path as _ag3s_workspace_path
import numpy as np, json, glob, ast, sys, collections
V = _ag3s_workspace_path('outputs/verify/T43/V11/runs/')
def st(v):
    v = [float(x) for x in v if x is not None]
    if not v: return {"n": 0}
    a = np.asarray(v); return {"n": len(a), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)), "max": float(a.max()), "min": float(a.min()), "mean": float(a.mean())}
out = {}
for arm in ("V11ABC", "V11AB"):
    sa = []; dr = []; n = 0; modes = collections.Counter(); bnd = 0
    for f in sorted(glob.glob(V + f"{arm}/server_*/server_constraints/run_*/chunk_*.npz")):
        z = np.load(f, allow_pickle=True)
        if int(z["t_step"]) == 1: continue
        m = (json.loads(str(z["summary_json"])).get("to") or {}).get("metrics") or {}
        n += 1
        x = m.get("servo_accel")
        if isinstance(x, str): x = ast.literal_eval(x)
        if isinstance(x, dict): sa.append(x); modes[x["mode"]] += 1
        y = m.get("deflection_rate")
        if isinstance(y, str): y = ast.literal_eval(y)
        if isinstance(y, dict): dr.append(y)
    out[arm] = {"n_chunks_excl_first": n, "servo_accel_records": len(sa), "modes": dict(modes),
                "n_over": st([x["n_over"] for x in sa]), "chunks_n_over_gt0": sum(x["n_over"] > 0 for x in sa),
                "overshoot_max_rad": st([x["overshoot_max_rad"] for x in sa]), "chunks_overshoot_gt1e-6": sum(x["overshoot_max_rad"] > 1e-6 for x in sa),
                "boundary_velocity_ratio": st([x["boundary_velocity_ratio"] for x in sa]), "chunks_boundary_velocity_ratio_gt1": sum(x["boundary_velocity_ratio"] > 1 for x in sa),
                "boundary_velocity_overshoot_rad": st([x["boundary_velocity_overshoot_rad"] for x in sa]), "chunks_boundary_velocity_overshoot_gt0": sum(x["boundary_velocity_overshoot_rad"] > 0 for x in sa),
                "deflection_rate_records": len(dr), "deflection_rate_cost": st([y["cost"] for y in dr]), "deflection_rate_max_step_change_rad": st([y["max_step_change_rad"] for y in dr])}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out)[:1500])
