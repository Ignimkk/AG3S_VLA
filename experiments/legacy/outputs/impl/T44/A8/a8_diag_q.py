"""T44e A8 diag: authorized-sphere queries that cross the 10 mm margin (off >= m, v < m) — tier, height above support, which variant."""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, collections, numpy as np
TC = ("outside", "coarse", "fine", "target_free", "window_edge", "static", "target_free_coarse")
for v in ("keep", "exclude"):
    tiers = collections.Counter(); H = []; drop = []; frames = set()
    for f in sorted(glob.glob(_ag3s_workspace_path('outputs/impl/T44/A8/raw_demo/*.npz'))):
        z = np.load(f); r = json.load(open(f[:-4] + ".json"))
        k = "gt_label__q_d_bo_d3|%s" % v
        if k not in z.files: continue
        ra = z["auth_radii"].astype(float); xo = z["gt_label__q_d_bo_d3|off"].astype(float); x = z[k].astype(float)
        cr = (xo - ra >= 0.01) & (x - ra < 0.01)
        if not cr.any(): continue
        frames.add(r["key"])
        t = z["gt_label__q_tier_bo_d3|%s" % v][cr]
        for c in t: tiers[TC[int(c)]] += 1
        sz = r["ball"]["support_plane_z_m"]
        H.append((z["auth_centres"][cr][:, 2] - ra[cr] - sz) * 1e3); drop.append((xo - x)[cr] * 1e3)
    h = np.concatenate(H); d = np.concatenate(drop)
    print(v, "frames", len(frames), sorted(frames), "tiers", dict(tiers), "sphere bottom above support mm p10/50/90",
          np.percentile(h, [10, 50, 90]).round(1), "drop mm p50/max", np.median(d).round(1), d.max().round(1))
