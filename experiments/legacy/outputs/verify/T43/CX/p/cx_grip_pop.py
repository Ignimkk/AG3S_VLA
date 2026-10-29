"""T43 CX — held-period gripper command / finger opening / apple-hand force per replayed run (V11A, V11AB, V11ABC H; CX B replays), numbers only.
usage: cx_grip_pop.py OUT_JSON
held period = rows grasp_t .. t_end-1, grasp_t from V11/analysis/analysis_<ARM>.json; t_end = first row >= grasp_t with apple-hand contact 0 on 3
consecutive rows (separation, any cause incl. the intended release) else 600. Per run: mean / min / max grip_cmd (1 = open, 0 = closed), mean finger_open_norm,
mean apple-hand max normal force (N), rows held, fraction of HOLD rows, and the V11 fail stage (T43V11ABC.verify.json fail_stage for listed runs,
else analysis success / grasp)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, pathlib, sys
import numpy as np
T43 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43'))
FS = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43V11ABC.verify.json')))["numbers"]["fail_stage"]["runs"]
out = {"definitions": __doc__, "runs": []}
for arm in ("V11A", "V11AB", "V11ABC"):
    A = json.load(open(T43 / f"V11/analysis/analysis_{arm}.json"))["runs"]
    for key, a in sorted(A.items()):
        stg = key.split("/")[1]; nm = key.split("_ep")[1]
        f = T43 / ("V11/replay" if stg == "H" else "CX/replay") / f"{arm}_{stg}_{arm}_ep{nm}.npz"
        if not f.exists() or a.get("grasp_t") is None:
            continue
        z = np.load(f); gt = int(a["grasp_t"]); c = z["cnt_apple-hand"]
        te = next((t for t in range(gt, len(c) - 2) if c[t] == 0 and c[t + 1] == 0 and c[t + 2] == 0), len(c))
        g = z["grip_cmd"][gt:te]; fo = z["finger_open_norm"][gt + 1:te + 1]; F = z["maxf_apple-hand"][gt:te]; h = z["hold"][gt:te]
        ck = f"{stg}:{nm}"; stage = FS[ck][arm]["stage"] if ck in FS and arm in FS[ck] else ("success" if a.get("success") else ("no_grasp" if not a.get("grasp") else "fail_other"))
        out["runs"].append({"arm": arm, "key": key, "stage": stage, "grasp_t": gt, "t_end": te, "rows_held": te - gt, "grip_mean": float(g.mean()) if len(g) else None,
                            "grip_min": float(g.min()) if len(g) else None, "grip_max": float(g.max()) if len(g) else None,
                            "grip_first16_mean": float(g[:16].mean()) if len(g) else None,
                            "open_mean": float(fo.mean()) if len(fo) else None, "force_mean_N": float(F.mean()) if len(F) else None, "hold_frac": float(h.mean()) if len(h) else None})
        r = out["runs"][-1]; print(arm, stg, nm, stage, r["rows_held"], round(r["grip_mean"], 3), round(r["grip_first16_mean"], 3), round(r["open_mean"], 3), round(r["force_mean_N"], 2), round(r["hold_frac"], 2))
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
