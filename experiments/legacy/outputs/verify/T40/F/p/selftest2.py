"""T40 F: fast set_hurdle (manager._place + mj_kinematics) vs exact (manager.activate) reset numbers, and cached-pose series vs
direct, on the captured model; peak RSS."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys, json, resource
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/F/p'))
import flib as F
c = F.Ctx(sys.argv[1]); out = {}
rng = np.random.default_rng(0)
mx = 0.0
for i in range(200):
    x, y, yaw, h = rng.uniform(0.40, 0.80), rng.uniform(-0.1, 0.33), rng.uniform(0, np.pi), rng.choice(np.arange(0.20, 0.3201, 0.02))
    c.set_hurdle(x, y, yaw, h); a = c.reset_check()
    c.set_hurdle(x, y, yaw, h, exact=True); b = c.reset_check()
    mx = max(mx, abs(a[0] - b[0]), abs(a[1] - b[1]))
out["fast_vs_exact_reset_max_abs_m_200"] = mx
c.set_hurdle(0.68, 0.23, 0.0, 0.26, exact=True); out["smoke_placement_reset"] = c.reset_check()
out["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(json.dumps(out)); json.dump(out, open(f"${AG3S_PROJECT_ROOT}/outputs/verify/T40/F/raw/selftest2_{sys.argv[1]}.json", "w"), indent=1)
