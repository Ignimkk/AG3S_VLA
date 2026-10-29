from common.workspace import workspace_path as _ag3s_workspace_path
# T43 P: cProfile of psearch.PCtx.evaluate on 36 candidates (ep1807)
import sys, cProfile, pstats, time
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import psearch as S
P = S.PCtx("1807_18071")
cands = [(u, v, dy, h) for u in (-0.06, 0.0, 0.06) for v in (0.14, 0.18, 0.22) for dy in (0, 20) for h in (0.20, 0.26)]
pr = cProfile.Profile(); t0 = time.time(); pr.enable()
for u, v, dy, h in cands:
    x, y, yaw = P.to_world(u, v, dy); P.evaluate(x, y, yaw, h)
pr.disable(); print("per cand", (time.time() - t0) / len(cands))
pstats.Stats(pr).sort_stats("tottime").print_stats(15)
