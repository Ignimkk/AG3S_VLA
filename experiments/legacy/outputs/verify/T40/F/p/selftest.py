"""T40 F: check flib.Ctx.series (cached geom poses) == direct mj_kinematics per row + manager distance; timing."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, time, json
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/F/p'))
import flib as F, gjk
c = F.Ctx(sys.argv[1]); m, d = c.m, c.d
res = {}
for (x, y, yaw, h) in [(0.68, 0.23, 0.0, 0.26), (0.62, 0.12, 0.3, 0.22), (0.56, 0.25, 0.0, 0.30)]:
    t0 = time.time(); c.set_hurdle(x, y, yaw, h); ta = time.time() - t0
    t0 = time.time(); rc = c.reset_check(); tb = time.time() - t0
    R = c.runs["E0"]; rows = list(range(0, len(R["t"]), 7))
    t0 = time.time(); s, pr = c.series("E0", rows, {"A": c.A_GRIP, "B": c.B_APPLE, "C": c.C_ROBOT}, cap=0.3, want_pair=True); tc = time.time() - t0
    # direct
    c.set_hurdle(x, y, yaw, h)
    mx = 0.0; mxg = 0.0
    for j, i in enumerate(rows):
        d.qpos[:] = R["Q"][i]; mujoco.mj_kinematics(m, d)
        for k, gs in (("A", c.A_GRIP), ("B", c.B_APPLE), ("C", c.C_ROBOT)):
            best = 0.3; bg = 0.3
            for o in c.HURDLE:
                for g in gs:
                    best = min(best, c.mgr.pair_distance(int(o), int(g), 0.3))
                    if k != "C":
                        bg = min(bg, gjk.signed_dist(m, d, int(o), int(g)) if True else 0)
            mx = max(mx, abs(best - s[k][j]))
            if k != "C":
                mxg = max(mxg, abs(min(bg, 0.3) - min(s[k][j], 0.3)) if bg > 0 else 0)
    c.restore()
    res[str((x, y, yaw, h))] = {"reset_check": rc, "max_abs_cached_vs_direct_m": mx, "max_abs_vs_gjk_sep_m(AB, separated rows)": mxg,
                                "t_set_ms": 1e3 * ta, "t_reset_ms": 1e3 * tb, "t_series_ms_per_row": 1e3 * tc / len(rows)}
    print(res[str((x, y, yaw, h))], flush=True)
json.dump(res, open(f"${AG3S_PROJECT_ROOT}/outputs/verify/T40/F/raw/selftest_{sys.argv[1]}.json", "w"), indent=1)
