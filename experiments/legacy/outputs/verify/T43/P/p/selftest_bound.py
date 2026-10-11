"""T43 P self-test: psearch.bb_min with the segment lower bound (cap 0.06) vs the earlier rbound-sphere version (raw/old_f_<TAG>.json,
cap 0.10) on the 5 T40 F placements: every metric below 0.06 must be identical; plus a brute-force check (every pair, every row,
no bound) on 6 random candidates of ep1807 for a_min, d_min, c values.  -> raw/selftest_bound.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, time, numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import flib as F, psearch as S
out = {"vs_old_f": {}, "brute": []}
for tag in F.TAGS:
    old = json.load(open(f"{_ag3s_workspace_path()}/outputs/verify/T43/P/raw/old_f_{tag}.json"))
    P = S.PCtx(tag); t0 = time.time(); e = P.evaluate(old["x"], old["y"], old["yaw"], old["h"]); dt = time.time() - t0
    diffs = {k: abs(e[k] - old[k]) for k in e if isinstance(e[k], float) and isinstance(old.get(k), float) and old[k] < S.CAP}
    out["vs_old_f"][tag] = {"n_compared": len(diffs), "max_abs_diff": max(diffs.values()), "eval_s": dt}
    print(tag, out["vs_old_f"][tag], flush=True)
P = S.PCtx("1807_18071"); c = P.c; rng = np.random.default_rng(0)


def brute(Pc, rows, geoms):
    m, d = c.m, c.d; best = S.CAP; ft = np.zeros(6)
    for i in rows:
        for g in geoms:
            a = c.pos_in_all[int(g)]; d.geom_xpos[g] = Pc["xpos"][i][a]; d.geom_xmat[g] = Pc["xmat"][i][a]
            for h in c.HURDLE:
                best = min(best, c.mgr.pair_distance(int(h), int(g), S.CAP, ft))
    return best


for k in range(6):
    u, v, dy, h = float(rng.uniform(-0.1, 0.1)), float(rng.uniform(0.1, 0.3)), float(rng.choice(S.DYAW)), float(rng.choice(S.HS))
    x, y, yaw = P.to_world(u, v, dy); e = P.evaluate(x, y, yaw, h)
    c.set_hurdle(x, y, yaw, h)
    b = {"wu_robot": brute(P.WU, range(len(P.WU["Q"])), c.ROBOT_ALL)}
    for cond, R in c.runs.items():
        rw = P.rows[cond]
        b[f"{cond}_pre_robot"] = brute(R, rw["PRE"], c.ROBOT_ALL); b[f"{cond}_car_apple"] = brute(R, rw["CAR"], c.B_APPLE)
        b[f"{cond}_car_grip"] = brute(R, rw["CAR"], c.A_GRIP)
    c.restore()
    rec = {"u": u, "v": v, "dyaw": dy, "h": h, "bb": {q: e[q] for q in b}, "brute": b, "max_abs_diff": max(abs(e[q] - b[q]) for q in b)}
    out["brute"].append(rec); print(rec["max_abs_diff"], rec["bb"], flush=True)
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T43/P/raw/selftest_bound.json'), "w"), indent=1)
