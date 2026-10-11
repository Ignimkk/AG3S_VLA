"""T44c D-sign2: off-composition values (this run) vs T44E2b raw npz of the same frame (same definitions)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import numpy as np, glob, json, pathlib
D = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44'))
out = {"n_frames": 0, "pairs": {}}
for f in sorted(glob.glob(str(D / "E2c/raw_*/*.npz"))):
    sub = pathlib.Path(f).parent.name
    g = D / "E2b" / sub / pathlib.Path(f).name
    if not g.exists():
        continue
    a, b = np.load(f), np.load(g)
    out["n_frames"] += 1
    for I in ("gt", "gt_label"):
        for m in ("ball", "bo_d3", "oracle"):
            for mine, theirs in ((f"{I}__D_d_{m}|off", f"{I}__sg_comp_{m}"), (f"{I}__q_d_{m}|off", f"{I}__q_d_{m}"),
                                 (f"{I}__grid_d_{m}|off", f"{I}__grid_d_{m}")):
                if mine not in a.files or theirs not in b.files:
                    continue
                x, y = a[mine].astype(np.float64), b[theirs].astype(np.float64)
                k = mine.split("__")[1].split("_d_")[0] + "|" + m
                r = out["pairs"].setdefault(k, {"n": 0, "n_shape_mismatch": 0, "max_abs_diff_m": 0.0, "n_diff_gt_1e-6": 0, "n_sign_flip": 0})
                if x.shape != y.shape:
                    r["n_shape_mismatch"] += 1; continue
                r["n"] += int(len(x)); r["max_abs_diff_m"] = max(r["max_abs_diff_m"], float(np.abs(x - y).max()) if len(x) else 0.0)
                r["n_diff_gt_1e-6"] += int((np.abs(x - y) > 1e-6).sum()); r["n_sign_flip"] += int(((x > 0) != (y > 0)).sum())
json.dump(out, open(D / "E2c/repro_vs_e2b.json", "w"), indent=1)
print(json.dumps(out, indent=0))
