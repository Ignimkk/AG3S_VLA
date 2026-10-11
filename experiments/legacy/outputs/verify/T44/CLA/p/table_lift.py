"""T44dA verifier: support-surface zero level of the served target-free field, coarse ON vs OFF, in the C1b (and C2) approach chunks
(last 4 chunks before the first close, approach_stats.json). Columns (1 mm z step from support-10 mm to support+40 mm) at the ball centre xy and at
8 xy points on a ring of radius (ball radius + 10 mm) and (ball fit radius) around it. zero level = lowest z >= support where the target-free
distance > 0, minus support z (mm). -> CLA/table_lift.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import os, sys, json, pathlib
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T44/CLA/p'))
import numpy as np, field as FD
C = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CLA')); CL = pathlib.Path(_ag3s_workspace_path('outputs/verify/T44/CL'))
ap = json.load(open(C / "approach_stats.json"))["per_run"]; out = {}
for arm in ("C1b", "C2"):
    res = []
    for name, v in ap.items():
        if not name.startswith(arm + "_"): continue
        a = json.load(open(CL / "analysis" / f"{name}.json")); files = {c["seq"]: c["server_file"] for c in a["server_chunks"]}
        for ch in v["chunks"]:
            z = np.load(files[ch["seq"]], allow_pickle=True)
            if "esdf_free_coarse_distance" not in z.files: continue
            s = json.loads(str(z["summary_json"])); tf = s["esdf_stats"]["target_free"]; tf = eval(tf) if isinstance(tf, str) else tf
            b = (tf or {}).get("ball")
            if not b: continue
            sz = b["support_plane_z_m"]; bc = np.asarray(b["centre_m"]); R = b["radius_mm"] / 1e3; Rf = b["fit"]["radius_mm"] / 1e3
            zs = np.arange(sz - 0.01, sz + 0.04, 0.001)
            xy = [bc[:2]] + [bc[:2] + rr * np.array([np.cos(t), np.sin(t)]) for rr in (Rf, R + 0.01) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
            P = np.asarray([[x, y, zz] for x, y in xy for zz in zs])
            on, off = FD.build(z, True), FD.build(z, False)
            von = on.target_free_distance(P).reshape(len(xy), -1); voff = off.target_free_distance(P).reshape(len(xy), -1)
            def lvl(V):
                o = []
                for col in V:
                    k = np.where((zs >= sz - 1e-9) & (col > 0))[0]; o.append(1e3 * (zs[k[0]] - sz) if len(k) else np.nan)
                return np.asarray(o)
            lo, lf = lvl(von), lvl(voff)
            res.append({"name": name, "seq": ch["seq"], "centre_on": lo[0], "centre_off": lf[0], "ringfit_on": float(np.nanmedian(lo[1:9])), "ringfit_off": float(np.nanmedian(lf[1:9])),
                        "ringout_on": float(np.nanmedian(lo[9:])), "ringout_off": float(np.nanmedian(lf[9:]))})
    m = lambda k: float(np.nanmedian([r[k] for r in res])) if res else None
    out[arm] = {"n_chunks": len(res), **{k + "_median_mm": m(k) for k in ("centre_on", "centre_off", "ringfit_on", "ringfit_off", "ringout_on", "ringout_off")}, "chunks": res}
    print(arm, {k: v for k, v in out[arm].items() if k != "chunks"})
json.dump(out, open(C / "table_lift.json", "w"), indent=1, default=float)
