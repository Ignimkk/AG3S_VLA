"""T44c A6 diagnostic: for residual target-interior negatives under candidate a (plane), list the 8 trilinear
corner voxels of the coarse target-free tier (height above support, seed = value 0, value)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, pathlib, collections
import numpy as np
sys.dont_write_bytecode = True
sys.path.insert(0, _ag3s_workspace_path('outputs/impl/T44/A6'))
import a6_smoke as S  # noqa: E402

KEYS = sys.argv[1].split(",")
orig = S.H.build_coarse_free_tier
store = {}


def spy(builder, ball, **kw):
    t, ms = orig(builder, ball, **kw)
    store[(kw.get("owned") is not None, kw.get("sign_floor"))] = (t.values.detach().cpu().numpy().copy(),
                                                                   np.asarray(t.origin, float), float(t.voxel_size), ball)
    return t, ms


S.H.build_coarse_free_tier = spy
out = {}
orig_run = S.run_one


def run(ctx, item, args):
    rec, npz = orig_run(ctx, item, args)
    sz = rec["ball"]["support_plane_z_m"]
    vals, org, vs, ball = store[(False, "plane")]
    P = npz["T_pts"].astype(np.float64)
    a, o = npz["T_ball_a"], npz["T_ball_off"]
    sel = (a < 0) & (o >= 0)
    rows = collections.Counter()
    for p in P[sel]:
        g = (p - org) / vs
        i0 = np.floor(g).astype(int)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ijk = i0 + [dx, dy, dz]
                    v = float(vals[tuple(ijk)])
                    c = org + ijk * vs
                    hz = round((c[2] - sz) * 1e3)
                    inb = np.linalg.norm(c - ball.centre) <= ball.radius
                    kind = "seed" if v == 0 else ("neg" if v < 0 else "pos")
                    rows[(hz, bool(inb), kind)] += 1
    out[item["key"]] = {"n": int(sel.sum()), "ball_zmin_mm": round((ball.z_min - sz) * 1e3, 1),
                        "corners": {f"h{k[0]}mm|inball={k[1]}|{k[2]}": v for k, v in sorted(rows.items())}}
    return rec, npz


S.run_one = run
sys.argv = [sys.argv[0], "--only", ",".join(KEYS), "--out", _ag3s_workspace_path('outputs/impl/T44/A6/diag_raw')]
S.main()
print(json.dumps(out, indent=1))
