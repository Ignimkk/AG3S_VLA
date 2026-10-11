"""servo_all_{V8,V9}.json -> raw/servo_summary.json (numbers only). Pools: exec rows all; exec rows from grasp_t; per stage (H / B); stop and hold counted apart."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, numpy as np, pathlib
R = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/V9/raw'))
def st(x):
    x = np.asarray([v for v in x if v is not None], float)
    return {"n": 0} if not len(x) else {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "max": float(x.max())}
out = {}
for arm in ("V8", "V9"):
    d = json.load(open(R / f"servo_all_{arm}.json"))["runs"]; o = {}
    pools = {"exec_all": lambda k, r, w: w["cat"] == "exec", "exec_from_grasp": lambda k, r, w: w["cat"] == "exec" and r["grasp_t"] is not None and w["t"] >= r["grasp_t"],
             "exec_H": lambda k, r, w: w["cat"] == "exec" and "/H/" in k, "exec_B": lambda k, r, w: w["cat"] == "exec" and "/B/" in k}
    for nm, f in pools.items():
        rows = [w for k, r in d.items() for w in r["rows"] if f(k, r, w)]
        o[nm] = {m: st([w[m] for w in rows]) for m in ("servo_err_mm", "cmd_err_mm", "servo_err_joint_deg", "cmd_err_joint_deg")}
    o["n_rows_by_cat"] = {c: sum(1 for r in d.values() for w in r["rows"] if w["cat"] == c) for c in ("exec", "hold", "stop")}
    o["hold_rows"] = {m: st([w[m] for r in d.values() for w in r["rows"] if w["cat"] == "hold"]) for m in ("servo_err_mm", "cmd_err_mm")}
    o["per_run_exec_median_servo_err_mm"] = {k: float(np.median([w["servo_err_mm"] for w in r["rows"] if w["cat"] == "exec"])) for k, r in d.items() if any(w["cat"] == "exec" for w in r["rows"])}
    o["src_counts"] = {}
    for r in d.values():
        for w in r["rows"]:
            o["src_counts"][w["src"]] = o["src_counts"].get(w["src"], 0) + 1
    out[arm] = o
(R / "servo_summary.json").write_text(json.dumps(out, indent=1))
for a, o in out.items():
    print(a, o["n_rows_by_cat"], {k: (round(v["servo_err_mm"]["median"], 3), round(v["servo_err_mm"]["p90"], 3), round(v["servo_err_mm"]["max"], 2), round(v["cmd_err_mm"]["median"], 2), v["servo_err_mm"]["n"]) for k, v in o.items() if k.startswith("exec_")},
          {k: round(v["max"], 3) for k, v in o["exec_all"].items()})
