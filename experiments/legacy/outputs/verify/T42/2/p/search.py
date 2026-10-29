"""T42-2 placement search (verifier, 2026-10-05; T40 F method generalized to five slots). CPU, one process per call.
usage: search.py EP SEED SLOT  -> raw/search_<EP>_<SLOT>.json

Phase of a slot: post_0 / fence_0 / overbar_0 = pick, hurdle_0 / crate_post_0 = place. Paths = recorded no-obstacle runs E0 and
E3b of seed SEED (plib windows).
Candidates. Pick slots (the recorded approach is a near-vertical descent onto the fruit, so a ring around the target's reset
centre c instead of the TCP path): footprint / bar centre = c + r (cos th, sin th), th every 30 deg,
  post_0       : r {0.04, 0.05, 0.06, 0.07, 0.08, 0.10}; height {0.10, 0.14, 0.18, 0.22, 0.26}; radius 0.02
  fence_0      : r as post_0; yaw = tangent of the ring + {0, +-30 deg}; bar_height {0.05, 0.08, 0.11, 0.14}; length 0.18; radius 0.008
  overbar_0    : r {0, 0.03}; yaw {0, 45, 90, 135 deg}; bar_height {0.14, 0.18, 0.22, 0.26, 0.30}; length 0.34; radius 0.012
Place slots: (1) crate-frame grid (user rule 2026-10-05, T43 P convention): u in [-0.10, 0.10] step 0.02 (crate x), v on the
target's side in [0.10, 0.30] step 0.02 (crate y); hurdle yaw = crate yaw + {0, +-10, +-20, +-30, +-45 deg}, bar_height 0.20..0.32;
crate_post height 0.10..0.30 step 0.04, radius {0.015, 0.020}; (2) path points every 0.01 m of 3-D arc length of the target-centre path over the carry window, both runs
(widened 2026-10-05 after criterion (f) left no candidate on the first, coarser grid):
  hurdle_0     : bar centre = path xy + u * bar direction, u in {0, +-0.03, +-0.06, +-0.09}; yaw = path normal + {0, +-20, +-40 deg};
                 bar_height {0.20, ..., 0.32}
  crate_post_0 : footprint = path xy + u * path normal, u in {0, +-0.02, +-0.04, +-0.06}; height {0.10, ..., 0.30 step 0.04};
                 radius {0.015, 0.020}
  path normal = horizontal normal of the xy path tangent over +-3 cm of arc (fallback: target reset xy -> crate xy direction).
  Footprint validator: 0.38 <= x <= 0.85, |y| <= 0.35 (snapshot validate_pick_place_obstacle_config).
Checks, in order a -> a' -> b -> d -> c (early-exit scans, plib distances):
  (a) reset: robot_clearance > 0 and object_clearance > 0 (client rule, stop distance 0)
  (a') (user rule 2026-10-05) every robot geom (both arms, grippers, torso, base) >= 0.030 m from the obstacle at the reset state,
      through the warmup replay (plib.Ctx.wu, every 50 of 1000 steps) and at the recorded t0 row of both runs
  (b) t0 row (both runs): ROBOT, TARGET, FRUITS, CRATE > 0
  (d) the other phase (both runs): pick slot -> transport window (carry after the 0.05 m lateral move; the lift-off column is
      shared with the grasp): ROBOT, TARGET, FRUITS > 0;
      (f) place slot -> the whole pick window (0 <= t < lift_t, approach + grasp): ROBOT, TARGET, FRUITS >= 0.030 m (user instruction
      2026-10-05: the pick must be unaffected)
  (c) own phase hit in BOTH runs: pick -> min(ROBOT, TARGET) <= 0 in the approach window; place -> min(ROBOT, TARGET) <= 0 in the
      carry window. (If a run has no carry window -- no grasp -- the place check uses the other run only; flagged.)
Survivors: full minima per group and phase; margins m = min(reset robot, reset object, t0 minimum, other-phase minimum).
Room (e), reported and used for selection:
  free_over_top_m : kinematic max TCP z at the slot's footprint xy (SLSQP over the 7 joints of the arm, xy tol 5 mm, TCP axis
                    within 15 deg of the recorded phase's mean axis, no collision check; max over runs, seeds = 8 rows of the
                    phase) minus (TCP - lowest point of gripper (pick) / of target (place), median over the phase rows) minus the
                    slot's top (posts: height; fence / overbar: bar_height + radius; hurdle: bar_height + 0.012)
  under_gap_m     : overbar only: (bar_height - radius) - target top above table at reset
  side_gap_m      : posts and crate_post: horizontal distance post surface -> target reset centre (pick) / crate outer rectangle
                    (place)
Selection (select.py): tier 1 = m >= 0.010 and room >= 0.03; tier 2 = m >= 0.005; tier 3 = feasible; room = max(free_over_top,
under_gap) for overbar, free_over_top otherwise; within a tier the largest room, ties -> larger m."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, sys, time
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/2/p'))
import plib as L  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T42/2/raw'))
PHASE = {"post_0": "pick", "fence_0": "pick", "overbar_0": "pick", "hurdle_0": "place", "crate_post_0": "place"}


def resample(P, ds=0.02):
    if len(P) < 2:
        return np.zeros(0), np.zeros((0, 3))
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1); s = np.r_[0, np.cumsum(seg)]
    S = np.arange(0, s[-1], ds)
    return S, np.stack([np.interp(S, s, P[:, k]) for k in range(3)], 1)


def normals(S, Pts, fallback):
    out = []
    for i in range(len(S)):
        lo, hi = np.searchsorted(S, S[i] - 0.03), min(np.searchsorted(S, S[i] + 0.03), len(S) - 1)
        tg = Pts[hi, :2] - Pts[lo, :2]
        tg = tg if np.linalg.norm(tg) >= 0.005 else fallback
        out.append(math.atan2(tg[1], tg[0]) + math.pi / 2)
    return np.array(out)


def top(e):
    s = e["slot"]
    if s in ("post_0", "crate_post_0"):
        return e["height"]
    if s == "hurdle_0":
        return e["bar_height"] + 0.012
    return e["bar_height"] + e["radius"]


def candidates(c, slot):
    ph = PHASE[slot]; out = []
    tgt0 = c.runs["E3b"]["Q"][0][L.AN.QADR[c.fruit]:L.AN.QADR[c.fruit] + 3]
    bc = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_BODY, "crate"); crate = c.d.xpos[bc][:2].copy()
    fb = crate - tgt0[:2]
    if ph == "pick":
        # ring around the target's reset centre (the recorded approach is a near-vertical descent onto the fruit)
        for r in ((0.0, 0.03) if slot == "overbar_0" else (0.04, 0.05, 0.06, 0.07, 0.08, 0.10)):
            for th in (np.radians(np.arange(0, 360, 30)) if r > 0 else [0.0]):
                xy = tgt0[:2] + r * np.array([math.cos(th), math.sin(th)])
                base = {"ring_r_m": r, "ring_theta_deg": float(np.degrees(th)), "path_run": None}
                if slot == "post_0":
                    for h in (0.10, 0.14, 0.18, 0.22, 0.26):
                        out.append({**base, "slot": slot, "position": [float(xy[0]), float(xy[1]), L.TABLE_Z], "yaw": 0.0, "height": h, "radius": 0.02})
                elif slot == "fence_0":
                    for dy in (0.0, math.radians(30), -math.radians(30)):
                        yaw = (th + math.pi / 2 + dy) % math.pi
                        for h in (0.05, 0.08, 0.11, 0.14):
                            out.append({**base, "slot": slot, "dyaw_deg": math.degrees(dy), "position": [float(xy[0]), float(xy[1]), L.TABLE_Z],
                                        "yaw": float(yaw), "bar_height": h, "length": 0.18, "radius": 0.008})
                else:
                    for yaw in np.radians([0, 45, 90, 135]):
                        for h in (0.14, 0.18, 0.22, 0.26, 0.30):
                            out.append({**base, "slot": slot, "position": [float(xy[0]), float(xy[1]), L.TABLE_Z], "yaw": float(yaw),
                                        "bar_height": h, "length": 0.34, "radius": 0.012})
        return out
    # crate-frame grid (user rule 2026-10-05: near the crate / within the handle span; T43 P convention): u = x_rel in
    # [-0.10, 0.10] step 0.02, v = y_rel on the target's side in [0.10, 0.30] step 0.02, crate axes from the reset crate body
    cb = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_BODY, "crate"); Rc = c.d.xmat[cb].reshape(3, 3); pc = c.d.xpos[cb].copy()
    side = 1.0 if (Rc.T @ (np.r_[tgt0[:2], pc[2]] - pc))[1] > 0 else -1.0
    cyaw = math.atan2(Rc[1, 0], Rc[0, 0])
    for u in np.round(np.arange(-0.10, 0.1001, 0.02), 3):
        for v in np.round(np.arange(0.10, 0.3001, 0.02), 3):
            p = pc + Rc @ np.array([u, side * v, 0.0])
            base = {"path_run": "grid", "s_m": None, "grid_u": float(u), "grid_v": float(side * v)}
            if slot == "crate_post_0":
                for h in (0.10, 0.14, 0.18, 0.22, 0.26, 0.30):
                    for rad in (0.015, 0.02):
                        out.append({**base, "slot": slot, "position": [float(p[0]), float(p[1]), L.TABLE_Z], "yaw": 0.0, "height": h, "radius": rad})
            else:
                for dy in np.radians([0, 10, -10, 20, -20, 30, -30, 45, -45]):
                    for h in np.round(np.arange(0.20, 0.32001, 0.02), 3):
                        out.append({**base, "slot": slot, "dyaw_deg": float(np.degrees(dy)), "position": [float(p[0]), float(p[1]), L.TABLE_Z],
                                    "yaw": float((cyaw + dy) % math.pi), "bar_height": float(h)})
    for cond, R in c.runs.items():
        rows = c.rows(cond, "carry")
        if not len(rows):
            continue
        a = L.AN.QADR[c.fruit]; P = R["Q"][rows][:, a:a + 3]
        S, Pts = resample(P, 0.01)
        if not len(S):
            continue
        yn = normals(S, Pts, fb)
        for i in range(len(S)):
            xy = Pts[i, :2]
            nrm = np.array([math.cos(yn[i]), math.sin(yn[i])])
            base = {"path_run": cond, "s_m": float(S[i]), "path_pt": Pts[i].tolist()}
            if slot == "crate_post_0":
                for u in (0.0, 0.02, -0.02, 0.04, -0.04, 0.06, -0.06):
                    p = xy + u * nrm
                    for h in (0.10, 0.14, 0.18, 0.22, 0.26, 0.30):
                        for rad in (0.015, 0.02):
                            out.append({**base, "slot": slot, "u_m": u, "position": [float(p[0]), float(p[1]), L.TABLE_Z], "yaw": 0.0, "height": h, "radius": rad})
            else:  # hurdle
                for dy in np.radians([0, 20, -20, 40, -40]):
                    yaw = (yn[i] + dy) % math.pi; ub = np.array([math.cos(yaw), math.sin(yaw)])
                    for u in (0.0, 0.03, -0.03, 0.06, -0.06, 0.09, -0.09):
                        p = xy + u * ub
                        for h in np.round(np.arange(0.20, 0.32001, 0.02), 3):
                            out.append({**base, "slot": slot, "dyaw_deg": math.degrees(dy), "u_m": u, "position": [float(p[0]), float(p[1]), L.TABLE_Z],
                                        "yaw": float(yaw), "bar_height": float(h)})
    return out


def entry(e):
    keys = ("slot", "position", "yaw") + tuple(k for k in ("height", "radius", "bar_height", "length") if k in e)
    return {k: e[k] for k in keys}


class WS:
    def __init__(self, c):
        self.c = c; m = c.m
        self.S = c.site
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{c.arm}_arm_{i}") for i in range(7)]
        self.ARM = [int(m.jnt_qposadr[j]) for j in jid]; self.LO, self.HI = m.jnt_range[jid, 0], m.jnt_range[jid, 1]

    def fk(self, qb, x):
        q = qb.copy(); q[self.ARM] = x; d = self.c.d; d.qpos[:] = q; mujoco.mj_kinematics(self.c.m, d)
        return d.site_xpos[self.S].copy(), d.site_xmat[self.S].reshape(3, 3).copy()

    def axis(self, Q):
        M = np.mean([self.fk(q, q[self.ARM])[1] for q in Q], axis=0); k = int(np.argmax(np.abs(M[2, :])))
        return k, M[:, k] / np.linalg.norm(M[:, k])

    def low_offset(self, Q, geoms):
        """median over rows of TCP z - lowest z of `geoms` (axis-aligned bound from geom_aabb corners)."""
        c = self.c; m = c.m; off = []
        for q in Q:
            p, _ = self.fk(q, q[self.ARM]); low = np.inf
            for g in geoms:
                R = c.d.geom_xmat[g].reshape(3, 3); ctr = c.d.geom_xpos[g] + R @ m.geom_aabb[g][:3]; ext = np.abs(R) @ m.geom_aabb[g][3:]
                low = min(low, ctr[2] - ext[2])
            off.append(p[2] - low)
        return float(np.median(off))

    def max_tcp_z(self, xy, Q, ang=15.0, tol=0.005):
        k, a = self.axis(Q); cmin = math.cos(math.radians(ang))
        idx = list(np.linspace(0, len(Q) - 1, 8).astype(int))
        best = None
        for i in idx:
            qb = Q[i].copy()
            cons = [{"type": "ineq", "fun": lambda x: tol ** 2 - np.sum((self.fk(qb, x)[0][:2] - xy) ** 2)},
                    {"type": "ineq", "fun": lambda x: self.fk(qb, x)[1][:, k] @ a - cmin}]
            r = minimize(lambda x: -self.fk(qb, x)[0][2], qb[self.ARM], method="SLSQP", bounds=list(zip(self.LO, self.HI)),
                         constraints=cons, options={"maxiter": 150, "ftol": 1e-9})
            p, Rm = self.fk(qb, r.x)
            if np.linalg.norm(p[:2] - xy) <= tol + 1e-4 and Rm[:, k] @ a >= cmin - 1e-4 and (best is None or p[2] > best):
                best = float(p[2])
        return None if best is None else best - L.TABLE_Z


def crate_rect_dist(c, xy):
    bc = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_BODY, "crate"); R = c.d.xmat[bc].reshape(3, 3)
    l = R.T @ (np.r_[xy, 0] - c.d.xpos[bc]); dd = np.maximum(np.abs(l[:2]) - np.array([0.09, 0.15]), 0)
    return float(np.linalg.norm(dd))


def main(ep, seed, slot):
    t0 = time.time(); c = L.Ctx(ep, seed); ph = PHASE[slot]; other = "transport" if ph == "pick" else "pick"
    dthr = 0.0 if ph == "pick" else 0.030   # (f) place slots: >= 30 mm from the whole recorded pick window (approach + grasp)
    own = "approach" if ph == "pick" else "carry"
    cands = candidates(c, slot)
    cnt = {"candidates": len(cands), "validator_xy": 0, "a": 0, "a_prime": 0, "b": 0, "d": 0, "c": 0}
    runs_c = [k for k in c.runs if len(c.rows(k, own))]
    feas = []
    for ci, e in enumerate(cands):
        if ci % 1000 == 0:
            print(ep, slot, "progress", ci, len(cands), cnt, f"{time.time() - t0:.0f}s", flush=True)
        x, y = e["position"][:2]
        if not (0.38 <= x <= 0.85 and -0.35 <= y <= 0.35):
            continue
        cnt["validator_xy"] += 1
        c.place([entry(e)])
        rr, ro = c.reset_check()
        if not (rr > 0 and ro > 0):
            continue
        cnt["a"] += 1; e["reset_robot"], e["reset_object"] = rr, ro
        # (a') user rule 2026-10-05: >= 30 mm from every robot geom at reset, through the warmup and at t0 (both runs)
        if c.first_le0(c.wu, np.arange(len(c.wu["t"])), c.ROBOT, thr=0.030) is not None or \
                any(c.first_le0(k, [0], c.ROBOT, thr=0.030) is not None for k in c.runs):
            continue
        cnt["a_prime"] += 1
        if any(c.first_le0(k, [0], np.r_[c.ROBOT, c.TARGET, c.FRUITS, c.CRATE]) is not None for k in c.runs):
            continue
        cnt["b"] += 1
        if any(c.first_le0(k, c.rows(k, other), np.r_[c.ROBOT, c.TARGET, c.FRUITS], thr=dthr) is not None for k in c.runs):
            continue
        cnt["d"] += 1
        if not runs_c or not all(c.first_le0(k, c.rows(k, own), np.r_[c.ROBOT, c.TARGET]) is not None for k in runs_c):
            continue
        cnt["c"] += 1
        e["c_runs"] = runs_c; e["mins"] = {}
        for k in c.runs:
            for phx, sets in (("t0", {"robot": c.ROBOT, "target": c.TARGET, "fruits": c.FRUITS, "crate": c.CRATE}),
                              (other, {"robot": c.ROBOT, "target": c.TARGET, "fruits": c.FRUITS}),
                              (own, {"grip": c.GRIP, "robot": c.ROBOT, "target": c.TARGET})):
                rows = c.rows(k, phx)
                if not len(rows):
                    continue
                s = c.series(k, rows, sets, cap=0.05)
                e["mins"][f"{k}:{phx}"] = {g: float(v.min()) for g, v in s.items()}
        e["t0_min"] = min(min(v.values()) for kk, v in e["mins"].items() if kk.endswith(":t0"))
        e["other_phase_min"] = min(min(v.values()) for kk, v in e["mins"].items() if kk.endswith(":" + other))
        e["margin_m"] = min(rr, ro, e["t0_min"], e["other_phase_min"])
        feas.append(e)
    c.restore()
    # room (e)
    ws = WS(c); Qs = {k: c.runs[k]["Q"][c.rows(k, own)] for k in runs_c}
    low_geoms = c.GRIP if ph == "pick" else c.TARGET
    off = ws.low_offset(np.vstack(list(Qs.values()))[::4], low_geoms) if Qs else None
    tgt_top = None
    c.restore(); a = L.AN.QADR[c.fruit]
    lo = min(c.d.geom_xpos[g][2] + (np.abs(c.d.geom_xmat[g].reshape(3, 3)) @ c.m.geom_aabb[g][3:])[2] for g in c.TARGET)
    tgt_top = max(c.d.geom_xpos[g][2] + (np.abs(c.d.geom_xmat[g].reshape(3, 3)) @ c.m.geom_aabb[g][3:])[2] for g in c.TARGET) - L.TABLE_Z
    tgt_xy = c.d.qpos[a:a + 2].copy()
    cache = {}
    for e in feas:
        key = (round(e["position"][0], 3), round(e["position"][1], 3))
        if key not in cache:
            zs = [ws.max_tcp_z(np.array(key), Q) for Q in Qs.values()]
            zs = [v for v in zs if v is not None]
            cache[key] = max(zs) if zs else None
        tz = cache[key]
        e["tcp_z_max_above_table_m"] = tz; e["tcp_minus_lowest_point_m"] = off
        e["free_over_top_m"] = None if (tz is None or off is None) else tz - off - top(e)
        if slot == "overbar_0":
            e["under_gap_m"] = e["bar_height"] - e["radius"] - tgt_top
        if slot in ("post_0", "crate_post_0"):
            xy = np.array(e["position"][:2])
            e["side_gap_m"] = (float(np.linalg.norm(xy - tgt_xy)) - e["radius"]) if slot == "post_0" else crate_rect_dist(c, xy) - e["radius"]
        rooms = [v for v in (e["free_over_top_m"], e.get("under_gap_m")) if v is not None]
        e["room_m"] = max(rooms) if rooms else None
    c.restore()
    out = {"ep": ep, "seed": seed, "slot": slot, "phase": ph, "definitions": __doc__, "counts": cnt, "n_feasible": len(feas),
           "runs": {k: {kk: R[kk] for kk in ("run_dir", "lift_t", "grasp_t", "place_t", "success")} for k, R in c.runs.items()},
           "target_top_above_table_m": tgt_top, "feasible": feas, "elapsed_s": time.time() - t0}
    (RAW / f"search_{ep}_{slot}.json").write_text(json.dumps(out, default=float))
    print(ep, slot, cnt, "feasible", len(feas), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
