"""T44e A8 diag: distractor-interior points that exclude makes clearance >= margin where keep did not (bo_d3 / ball)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, numpy as np
from scipy.spatial import cKDTree
for m in ("ball", "bo_d3"):
    H, X, K, O, F, NT = [], [], [], [], 0, []
    for f in sorted(glob.glob(_ag3s_workspace_path('outputs/impl/T44/A8/raw_demo/*.npz'))):
        z = np.load(f); k = "gt_label__D_d_%s|exclude" % m
        if k not in z.files: continue
        rf = float(z["r_finger"])
        e, kp, o = (z["gt_label__D_d_%s|%s" % (m, v)].astype(float) for v in ("exclude", "keep", "off"))
        sel = (e - rf >= 0.01) & (kp - rf < 0.01)
        if sel.any():
            F += 1; H.append(z["gt_label__D_height"][sel]); X.append(e[sel]); K.append(kp[sel]); O.append(o[sel])
    h = np.concatenate(H) * 1e3; x = np.concatenate(X) * 1e3; kk = np.concatenate(K) * 1e3; oo = np.concatenate(O) * 1e3
    print(m, "frames", F, "pts", len(h), "height p10/50/90", np.percentile(h, [10, 50, 90]).round(1),
          "exclude val p50", np.median(x).round(1), "keep val p50", np.median(kk).round(1), "off val p50", np.median(oo).round(1),
          "frac off also >= margin", round(float(np.mean(oo - 0 >= 0)), 3))

print("--- D clr >= margin frames by height band (bo_d3)")
for v in ("off", "keep", "exclude"):
    row = {}
    for lo, hi in ((0, 15), (15, 1000)):
        fr = pts = 0
        for f in sorted(glob.glob(_ag3s_workspace_path('outputs/impl/T44/A8/raw_demo/*.npz'))):
            z = np.load(f); k = "gt_label__D_d_bo_d3|%s" % v
            if k not in z.files: continue
            h = z["gt_label__D_height"] * 1e3
            c = (z[k].astype(float) - float(z["r_finger"]) >= 0.01) & (h >= lo) & (h < hi)
            fr += int(c.any()); pts += int(c.sum())
        row[f"h{lo}-{hi}mm"] = (fr, pts)
    print(v, row)
