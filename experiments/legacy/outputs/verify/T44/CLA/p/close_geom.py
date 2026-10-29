"""T44dA verifier: MuJoCo replay geometry at the FIRST close command of every CL run (CL/replay/<ARM>_R_<NAME>.npz, read-only).
t_c = first control row with grip_cmd < 0.5 (after >= 1 row with grip_cmd > 0.9). Reported at t_c and at t_c + 8 (one chunk later):
finger_mid - target_pos (base frame, mm; dz > 0 = finger midpoint above the target body origin), |d|, finger_open_norm, hand-target contact count,
hold rows in [t_c, t_c+16), and target_z rise at t_c+24 (mm). -> CLA/close_geom.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, glob, pathlib
import numpy as np
CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL')); out = {}
per_run = json.load(open(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T44d.verify.json')))["numbers"]["per_run"]
for f in sorted(glob.glob(str(CL / "replay" / "*_R_*.npz"))):
    name = pathlib.Path(f).stem.split("_R_")[1]; z = np.load(f)
    g = z["grip_cmd"]; opened = np.where(g > 0.9)[0]
    if not len(opened): continue
    cand = np.where((g < 0.5) & (np.arange(len(g)) > opened[0]))[0]
    if not len(cand): out[name] = {"t_c": None}; continue
    tc = int(cand[0]); T = len(g)
    def at(t):
        t = min(t, T - 1); d = (z["finger_mid"][t] - z["target_pos"][t]) * 1e3
        return {"dx": float(d[0]), "dy": float(d[1]), "dz": float(d[2]), "d": float(np.linalg.norm(d)), "open": float(z["finger_open_norm"][t]),
                "n_ht": int(z["cnt_hand-target"][t])}
    pr = per_run.get(name, {})
    out[name] = {"t_c": tc, "seq_c": int(z["seq"][tc]), "at_c": at(tc), "at_c8": at(tc + 8), "hold_rows_c_16": int(z["hold"][tc:tc + 16].sum()),
                 "target_rise_c24_mm": float((z["target_pos"][min(tc + 24, T - 1)][2] - z["target_pos"][tc][2]) * 1e3),
                 "success": pr.get("success"), "grasp": pr.get("grasp"), "first_grasp_ok": pr.get("first_grasp_ok"), "failure_mode": pr.get("failure_mode")}
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T44/CLA/close_geom.json'), "w"), indent=1)
for arm in ("C0", "C1a", "C1b", "C2"):
    R = {k: v for k, v in out.items() if k.startswith(arm + "_") and v.get("t_c") is not None}
    for lab, sel in (("all", R), ("firstok", {k: v for k, v in R.items() if v["first_grasp_ok"]}), ("not_firstok", {k: v for k, v in R.items() if not v["first_grasp_ok"]})):
        if not sel: continue
        a = lambda key, sub="at_c8": np.median([v[sub][key] for v in sel.values()])
        print(arm, lab, len(sel), "dz_c %.1f dz_c8 %.1f d_c8 %.1f dxy_c8 %.1f open_c8 %.2f hold16 %.1f rise24 %.1f" % (a("dz", "at_c"), a("dz"), a("d"),
              np.median([np.hypot(v["at_c8"]["dx"], v["at_c8"]["dy"]) for v in sel.values()]), a("open"), np.median([v["hold_rows_c_16"] for v in sel.values()]),
              np.median([v["target_rise_c24_mm"] for v in sel.values()])))
