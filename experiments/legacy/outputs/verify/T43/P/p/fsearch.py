"""T40 F (v2, analytic, lead rule 2026-10-03): hurdle placement on the recorded transport path. CPU, 1 process per call.
usage: fsearch.py TAG [TAG ...]  -> raw/fsearch_<TAG>.json

Path      : recorded apple-centre path (free-joint qpos 38:41) over control rows lift_t <= t < place_t of E0 and of E3b (T40 A
            windows), resampled every 0.01 m of 3-D arc length s from lift-off.
Candidate : bar centre xy = path point at s (run R in {E0, E3b}); bar direction = horizontal path normal (tangent from the xy
            path +-3 cm of arc; where the xy tangent is < 5 mm over that span, i.e. the vertical lift, the apple t0 -> crate
            direction is used) rotated by dyaw in {0, +-20, +-40} deg; extension (not in the lead's rule, flagged): the bar centre
            may be shifted along the bar by u in {-0.06, 0, +0.06} m (env FS_US; second pass {-0.10, -0.08, +0.08, +0.10}, suffix _u10) (the path still crosses the bar between the posts);
            bar_height h in {0.20, 0.22, ..., 0.32} m.
Order     : a -> b -> d -> c, each with an early-exit scan (flib.first_le0; same distance code); survivors get full minima.
Checks    : (a) client reset check on the captured reset scene (flib: robot_clearance > 0, object_clearance > 0);
            (c) exact, min over lift_t <= t < place_t of min(gripper, apple) distance <= 0 in E0 AND E3b; also c at h - 0.02
                (geometric only, flag c_low);
            (d) min over rows t < lift_t of distance to all robot geoms, apple, other fruits > 0 (both runs);
            (b) t0 (recorded planning row 0) robot, apple+fruits, crate > 0 (both runs).
Detour (e): for each distinct feasible bar-centre xy: kinematic max left-TCP z at that xy (workspace.max_tcp_z: SLSQP over the
            7 left-arm joints, xy tol 5 mm, approach axis within 15 deg of the recorded carry axis, no collision check) per run,
            max over runs; apple_bottom_max_est = that - median(TCP z - apple bottom z) over the carry rows of both runs;
            free_height_over_bar = apple_bottom_max_est - bar top (h + 0.012).
Selection : tier 1 = feasible and c_low and reset robot >= 0.02 and reset object >= 0.01 and pre-lift >= 0.02 and t0 >= 0.02;
            tier 2 = feasible and c_low; tier 3 = feasible. Within the best non-empty tier: largest free_height_over_bar,
            ties -> larger pre-lift minimum."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, sys, time
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import flib as F, gjk  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/raw'))
import os
DYAW = np.radians([0, 20, -20, 40, -40]); US = tuple(float(v) for v in os.environ.get("FS_US", "0,-0.06,0.06").split(",")); SUFFIX = os.environ.get("FS_SUFFIX", ""); HS = np.round(np.arange(0.20, 0.32001, 0.02), 3)


def resample(P, ds=0.01):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
    S = np.arange(0, s[-1], ds)
    return S, np.stack([np.interp(S, s, P[:, k]) for k in range(3)], 1)


def normal_dirs(S, Pts, fallback):
    out = []
    for i in range(len(S)):
        lo, hi = np.searchsorted(S, S[i] - 0.03), min(np.searchsorted(S, S[i] + 0.03), len(S) - 1)
        tg = Pts[hi, :2] - Pts[lo, :2]
        tg = tg if np.linalg.norm(tg) >= 0.005 else fallback
        out.append(math.atan2(tg[1], tg[0]) + math.pi / 2)
    return np.array(out)


class WS:
    """kinematic max TCP z (copy of T40 A workspace.py on the captured model)."""
    def __init__(self, c):
        self.c = c; m = c.m
        self.S = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_ee")
        self.ARM = [int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"left_arm_{i}")]) for i in range(7)]
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"left_arm_{i}") for i in range(7)]
        self.LO, self.HI = m.jnt_range[jid, 0], m.jnt_range[jid, 1]

    def fk(self, qb, x):
        q = qb.copy(); q[self.ARM] = x; d = self.c.d; d.qpos[:] = q; mujoco.mj_kinematics(self.c.m, d)
        return d.site_xpos[self.S].copy(), d.site_xmat[self.S].reshape(3, 3).copy()

    def axis(self, Q):
        M = np.mean([self.fk(q, q[self.ARM])[1] for q in Q], axis=0); k = int(np.argmax(np.abs(M[2, :])))
        return k, M[:, k] / np.linalg.norm(M[:, k])

    def offset(self, Q):
        c = self.c; off = []
        for q in Q:
            p, _ = self.fk(q, q[self.ARM]); s = gjk.make_support(c.m, c.d, int(c.B_APPLE[0]))
            off.append(p[2] - s(np.array([0, 0, -1.0]))[2])
        return float(np.median(off))

    def max_tcp_z(self, xy, Q, ang=15.0, tol=0.005):
        k, a = self.axis(Q); cmin = math.cos(math.radians(ang))
        idx = list(np.linspace(0, len(Q) - 1, 8).astype(int))
        dxy = [np.linalg.norm(self.fk(q, q[self.ARM])[0][:2] - xy) for q in Q]; idx.append(int(np.argmin(dxy)))
        best = None
        for i in idx:
            qb = Q[i].copy()
            cons = [{"type": "ineq", "fun": lambda x: tol ** 2 - np.sum((self.fk(qb, x)[0][:2] - xy) ** 2)},
                    {"type": "ineq", "fun": lambda x: self.fk(qb, x)[1][:, k] @ a - cmin}]
            r = minimize(lambda x: -self.fk(qb, x)[0][2], qb[self.ARM], method="SLSQP", bounds=list(zip(self.LO, self.HI)),
                         constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
            p, Rm = self.fk(qb, r.x)
            if np.linalg.norm(p[:2] - xy) <= tol + 1e-4 and Rm[:, k] @ a >= cmin - 1e-4 and (best is None or p[2] > best):
                best = float(p[2])
        return None if best is None else best - F.TABLE_Z


def check_c(c, h_override=None):
    res = {}
    for cond, R in c.runs.items():
        rows = np.where((R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"]))[0]
        s, pr = c.series(cond, rows, {"A": c.A_GRIP, "B": c.B_APPLE}, cap=0.05, want_pair=True)
        ab = np.minimum(s["A"], s["B"])
        j = int(np.argmin(ab)); hit = ab <= 0
        res[cond] = {"min_A": float(s["A"].min()), "min_B": float(s["B"].min()), "n_rows_le0": int(hit.sum()),
                     "first_contact_t": int(R["t"][rows[int(np.argmax(hit))]]) if hit.any() else None,
                     "min_pair": pr["A"][j] if s["A"][j] <= s["B"][j] else pr["B"][j]}
    return res


def main(tag):
    t0 = time.time(); c = F.Ctx(tag)
    op_apple = c.runs["E0"]["Q"][0][38:40]; crate = None
    z = np.load(F.CAPDIR / f"{tag}_reset.npz")
    bc = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_BODY, "crate"); c.restore(); crate = c.d.xpos[bc][:2].copy()
    fallback = crate - op_apple
    cands = []
    for cond, R in c.runs.items():
        mk = (R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"])
        S, Pts = resample(R["Q"][mk][:, 38:41])
        yawn = normal_dirs(S, Pts, fallback)
        for i in range(len(S)):
            for dy in DYAW:
                yaw = (yawn[i] + dy) % math.pi
                u = np.array([math.cos(yaw), math.sin(yaw)])
                for us in US:
                    xy = Pts[i, :2] + us * u
                    for h in HS:
                        cands.append({"path_run": cond, "s_m": float(S[i]), "path_pt": Pts[i].tolist(), "dyaw_deg": float(np.degrees(dy)),
                                      "u_m": us, "x": float(xy[0]), "y": float(xy[1]), "yaw": float(yaw), "h": float(h)})
    cnt = {"candidates": len(cands), "validator_xy": 0, "a": 0, "b": 0, "d": 0, "c": 0}
    feas = []
    for ci, e in enumerate(cands):
        if ci % 2000 == 0:
            print(tag, "progress", ci, len(cands), cnt, f"{time.time() - t0:.0f}s", flush=True)
        if not (0.38 <= e["x"] <= 0.85 and -0.35 <= e["y"] <= 0.35):
            continue
        cnt["validator_xy"] += 1
        c.set_hurdle(e["x"], e["y"], e["yaw"], e["h"])
        rr, ro = c.reset_check()
        if not (rr > 0 and ro > 0):
            continue
        cnt["a"] += 1; e["reset_robot"], e["reset_object"] = rr, ro
        # (b) t0 rows, early exit
        if any(F.first_le0(c, cond, [0], np.r_[c.ROBOT_ALL, c.B_APPLE, c.FRUITS, c.CRATE]) is not None for cond in c.runs):
            continue
        cnt["b"] += 1
        # (d) pre-lift, early exit (rows from lift-off backwards: the grasp approach first)
        if any(F.first_le0(c, cond, np.where((R["t"] >= 0) & (R["t"] < R["lift_t"]))[0][::-1], np.r_[c.ROBOT_ALL, c.B_APPLE, c.FRUITS]) is not None
               for cond, R in c.runs.items()):
            continue
        cnt["d"] += 1
        # (c) carry, early exit at the first contact
        if not all(F.first_le0(c, cond, np.where((R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"]))[0], np.r_[c.A_GRIP, c.B_APPLE]) is not None
                   for cond, R in c.runs.items()):
            continue
        cnt["c"] += 1
        # survivors: full values
        e["c"] = check_c(c); e["prelift_min"] = {}; e["t0_min"] = {}
        for cond, R in c.runs.items():
            rows = np.where((R["t"] >= 0) & (R["t"] < R["lift_t"]))[0]
            s = c.series(cond, rows, {"robot": c.ROBOT_ALL, "apple": c.B_APPLE, "fruits": c.FRUITS}, cap=0.05)
            e["prelift_min"][cond] = {k: float(v.min()) for k, v in s.items()}
            s = c.series(cond, [0], {"robot": c.ROBOT_ALL, "apple": c.B_APPLE, "fruits": c.FRUITS, "crate": c.CRATE}, cap=0.05)
            e["t0_min"][cond] = {k: float(v.min()) for k, v in s.items()}
        # c at h - 0.02 (geometric; below the validator range when h = 0.20)
        c.mgr._place("hurdle_0", [e["x"], e["y"], F.TABLE_Z], e["yaw"], e["h"] - 0.02); mujoco.mj_kinematics(c.m, c.d)
        cl = check_c(c); e["c_low"] = all(v["n_rows_le0"] > 0 for v in cl.values()); e["c_low_detail"] = cl
        e["prelift_overall_min"] = min(min(v.values()) for v in e["prelift_min"].values())
        e["t0_overall_min"] = min(min(v.values()) for v in e["t0_min"].values())
        feas.append(e)
    c.restore()
    # detour
    ws = WS(c); carry = {}
    for cond, R in c.runs.items():
        mk = (R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"]); carry[cond] = R["Q"][mk]
    off = ws.offset(np.vstack(list(carry.values())))
    cache = {}
    for e in feas:
        key = (round(e["x"], 3), round(e["y"], 3))
        if key not in cache:
            zs = [ws.max_tcp_z(np.array(key), Q) for Q in carry.values()]
            zs = [v for v in zs if v is not None]
            cache[key] = max(zs) if zs else None
        tz = cache[key]
        e["tcp_z_max_above_table_m"] = tz; e["tcp_minus_apple_bottom_median_m"] = off
        e["free_height_over_bar_m"] = None if tz is None else tz - off - (e["h"] + 0.012)
    c.restore()

    def tier(e):
        if e["c_low"] and e["reset_robot"] >= 0.02 and e["reset_object"] >= 0.01 and e["prelift_overall_min"] >= 0.02 and e["t0_overall_min"] >= 0.02:
            return 1
        return 2 if e["c_low"] else 3
    for e in feas:
        e["tier"] = tier(e)
    best = None
    for t in (1, 2, 3):
        pool = [e for e in feas if e["tier"] == t and e["free_height_over_bar_m"] is not None]
        if pool:
            best = max(pool, key=lambda e: (round(e["free_height_over_bar_m"], 4), e["prelift_overall_min"])); break
    out = {"tag": tag, "u_values_m": list(US), "definitions": __doc__, "counts": cnt, "n_feasible": len(feas),
           "n_by_tier": {t: sum(1 for e in feas if e["tier"] == t) for t in (1, 2, 3)}, "best": best, "feasible": feas,
           "elapsed_s": time.time() - t0}
    (RAW / f"fsearch_{tag}{SUFFIX}.json").write_text(json.dumps(out, default=float))
    print(tag, cnt, "feasible", len(feas), out["n_by_tier"], "best", None if best is None else
          {k: (round(best[k], 4) if isinstance(best[k], float) else best[k]) for k in ("path_run", "s_m", "dyaw_deg", "u_m", "x", "y", "yaw", "h", "reset_robot", "reset_object", "prelift_overall_min", "t0_overall_min", "free_height_over_bar_m", "tier")},
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
