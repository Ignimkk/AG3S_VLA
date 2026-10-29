"""T43 P2: overbar_0 ACROSS the crate, inside the handles. CPU, 1 process per call.
usage: p2search.py TAG [TAG ...]  -> raw/p2search_<TAG><P2_SUFFIX>.json

Slot     : overbar_0 (pi05_TO_hybrid c7c1cc9 / fce4af8): capsule bar (radius r) along the slot's local x at bar_height h above the
           table (bar centre), 2 cylinder legs r 0.012 at position +- R(yaw)[L/2, 0, 0] from the table up to the bar axis.
Frame    : crate frame (body 'crate' at the reset scene).  u = bar-centre x_rel, v = bar-centre y_rel; s = +1 when the apple starts
           at y_rel > 0 (left-arm episodes), -1 otherwise (ep1834, right arm); the grid is over |v| on the apple side (v = s*|v|).
           dyaw = overbar yaw - crate yaw; dyaw 0 = bar along the crate x axis = across the crate's short dimension (walls at
           x_rel +-0.090, the long walls); legs then stand at x_rel = u +- L/2.
Crate    : outer |x_rel| <= 0.090, |y_rel| <= 0.150; interior |x_rel| < 0.082, |y_rel| < 0.142; wall top 0.120 m above the table;
           handles |y_rel| 0.150..0.210, top 0.157 (probe.json).
Grid     : |v| in P2_V (default 0.02..0.13 step 0.01), u in P2_U (default -0.02, 0, 0.02), dyaw in P2_DYAW (default 0, +-5 deg),
           L in P2_L (default 0.22..0.34 step 0.03), h in P2_H (default 0.14..0.30 step 0.01), r in P2_R (default 0.012).
Rows     : as T43 P: WU = reset scene + 1000-step warmup replay every 50 steps; T0 = recorded planning row 0 of E0 and E3b;
           PRE = control rows t < lift_t; CAR = control rows lift_t <= t < place_t.
Metric   : min pair_distance (manager, certified) over rows and (overbar geom, set geom) pairs, branch-and-bound, cap 0.06 m.
Criteria : a' = min(ROBOT over WU and T0) >= 0.030;
           b  = CRATE, APPLE, FRUITS over WU row 0 and T0 > 0, client reset check (ARMS > 0.02, object > 0), and legs_outside_crate
                (each leg circle, r 0.012, lies outside the crate outer footprint);
           c  = min(GRIP, APPLE) over CAR <= 0 in E0 AND E3b; c_apple = min(APPLE) over CAR <= 0 in E0 AND E3b;
           d  = min(ROBOT, APPLE) over PRE >= 0.030 (both runs);
           e  = free_height_over_bar >= 0.05 m: kinematic max TCP z of the used arm (SLSQP, approach axis within 15 deg of the
                recorded carry axis, no collision check) at the crossing point (where the recorded apple path crosses y_rel = v,
                projected on the bar axis), max over E0/E3b, minus the median TCP - apple-bottom offset over the carry rows, minus
                the bar top (h + r); drop_gap_m = (|v| - r) + 0.142 = y distance from the bar's crate-centre edge to the far
                interior wall face (reported, with the apple's horizontal size)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, os, pathlib, sys, time
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P2/p'))
import p2lib as F  # noqa: E402

CAP = 0.06
WALL_X, WALL_Y, IN_X, IN_Y = 0.090, 0.150, 0.082, 0.142


def _rng(env, default):
    a = [float(x) for x in os.environ.get(env, default).split(",")]
    return a


def grid(env, lo, hi, step):
    v = os.environ.get(env)
    if v:
        return [float(x) for x in v.split(",")]
    return list(np.round(np.arange(lo, hi + 1e-6, step), 4))


VS = grid("P2_V", 0.02, 0.13, 0.01); US = grid("P2_U", -0.02, 0.02, 0.02); DYAW = grid("P2_DYAW", -5, 5, 5)
LS = grid("P2_L", 0.22, 0.34, 0.03); HS = grid("P2_H", 0.14, 0.30, 0.01); RS = grid("P2_R", 0.012, 0.012, 0.01)
SUFFIX = os.environ.get("P2_SUFFIX", "")


class PCtx:
    def __init__(self, tag):
        self.c = c = F.Ctx(tag); m, d = c.m, c.d
        c.restore()
        bc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")
        self.crate_xy = d.xpos[bc][:2].copy(); Rc = d.xmat[bc].reshape(3, 3)
        self.crate_yaw = math.atan2(Rc[1, 0], Rc[0, 0])
        ap = c.q_reset[38:40]; self.side = 1.0 if self.to_crate(*ap)[1] > 0 else -1.0
        z = np.load(F.CAPDIR / f"{tag}_reset.npz")
        c.restore(); d.qvel[:] = z["qvel"]; d.ctrl[:] = z["ctrl"]
        if m.na:
            d.act[:] = z["act"]
        Q = [c.q_reset.copy()]
        for k in range(1, 1001):
            mujoco.mj_step(m, d)
            if k % 50 == 0:
                Q.append(d.qpos.copy())
        self.wu_final_vs_row0 = {cond: float(np.abs(Q[-1] - R["Q"][0]).max()) for cond, R in c.runs.items()}
        Q = np.array(Q); xp, xm = c.pose_cache(Q)
        self.WU = {"Q": Q, "xpos": xp, "xmat": xm}
        c.restore()
        self.rows = {}
        for cond, R in c.runs.items():
            self.rows[cond] = {"T0": np.array([0]), "PRE": np.where((R["t"] >= 0) & (R["t"] < R["lift_t"]))[0],
                               "CAR": np.where((R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"]))[0]}
        # recorded apple path in the crate frame (carry rows)
        self.apath = {}
        for cond, R in c.runs.items():
            A = R["Q"][self.rows[cond]["CAR"]][:, 38:41]
            self.apath[cond] = np.array([self.to_crate(a[0], a[1]) for a in A])

    def to_world(self, u, v, dyaw_deg):
        cy, sy = math.cos(self.crate_yaw), math.sin(self.crate_yaw)
        return self.crate_xy[0] + cy * u - sy * v, self.crate_xy[1] + sy * u + cy * v, self.crate_yaw + math.radians(dyaw_deg)

    def to_crate(self, x, y):
        cy, sy = math.cos(self.crate_yaw), math.sin(self.crate_yaw); dx, dy = x - self.crate_xy[0], y - self.crate_xy[1]
        return cy * dx + sy * dy, -sy * dx + cy * dy

    def bb_multi(self, P, rows, sets, cap=CAP):
        """T43 P psearch.bb_multi with the overbar geoms (capsule / cylinders: axis = geom z, half-length size[1])."""
        c = self.c; m, d = c.m, c.d
        H = c.OB
        ax = d.geom_xmat[H].reshape(-1, 3, 3)[:, :, 2]; hl = m.geom_size[H, 1][:, None]
        A0 = d.geom_xpos[H] - ax * hl; AB = 2 * ax * hl; hr = m.geom_size[H, 0]
        ends = np.vstack([A0, A0 + AB]); hc = 0.5 * (ends.min(0) + ends.max(0))
        Rh = float(np.max(np.linalg.norm(ends - hc, axis=1) + np.r_[hr, hr]))
        allg = np.unique(np.concatenate([np.asarray(v, int) for v in sets.values()]))
        ix = np.array([c.pos_in_all[int(g)] for g in allg], int); gr = m.geom_rbound[c.ALL[ix]]
        X = P["xpos"][rows][:, ix]
        coarse = np.linalg.norm(X - hc, axis=2) - gr[None] - Rh
        rr, gg = np.nonzero(coarse <= cap)
        W = X[rr, gg][:, None, :] - A0[None]
        tt = np.clip((W * AB[None]).sum(-1) / (AB * AB).sum(-1)[None], 0.0, 1.0)
        gapN = np.linalg.norm(W - tt[..., None] * AB[None], axis=2) - gr[gg][:, None] - hr[None]
        col = {int(g): j for j, g in enumerate(allg)}
        out = {}; ft = np.zeros(6)
        for name, geoms in sets.items():
            sel = np.isin(gg, [col[int(g)] for g in geoms])
            G = gapN[sel]; R_ = rr[sel]; Gi = gg[sel]
            flat = G.ravel(); near = np.flatnonzero(flat <= cap)
            best, arg = cap, None
            for k in near[np.argsort(flat[near], kind="stable")]:
                if flat[k] > best:
                    break
                n, hi = divmod(int(k), len(H))
                a = ix[Gi[n]]; g = int(c.ALL[a]); i = int(rows[R_[n]])
                d.geom_xpos[g] = P["xpos"][i][a]; d.geom_xmat[g] = P["xmat"][i][a]
                v = c.mgr.pair_distance(int(H[hi]), g, max(best, 1e-6), ft)
                if v < best:
                    best, arg = float(v), (i, c.gname(H[hi]), c.gname(g))
            out[name] = (best, arg)
        return out

    def bb_min(self, P, rows, geoms, cap=CAP):
        return self.bb_multi(P, rows, {"x": geoms}, cap)["x"]

    def evaluate(self, x, y, yaw, h, L, r):
        c = self.c
        c.set_overbar(x, y, yaw, h, L, r)
        e = {}
        ra, ro = c.reset_check(); e["client_reset_arms"] = ra; e["client_reset_object"] = ro
        wrows = np.arange(len(self.WU["Q"]))
        e["wu_robot"], e["wu_robot_arg"] = self.bb_min(self.WU, wrows, c.ROBOT_ALL)
        r0 = self.bb_multi(self.WU, np.array([0]), {"robot": c.ROBOT_ALL, "crate": c.CRATE, "apple": c.B_APPLE, "fruits": c.FRUITS})
        e["reset_robot"] = r0["robot"][0]; e["reset_crate"], e["reset_crate_arg"] = r0["crate"]
        e["reset_apple"] = r0["apple"][0]; e["reset_fruits"] = r0["fruits"][0]
        for cond, R in c.runs.items():
            rw = self.rows[cond]
            t0 = self.bb_multi(R, rw["T0"], {"robot": c.ROBOT_ALL, "crate": c.CRATE, "apple": c.B_APPLE, "fruits": c.FRUITS})
            for k in ("robot", "crate", "apple", "fruits"):
                e[f"{cond}_t0_{k}"] = t0[k][0]
            pre = self.bb_multi(R, rw["PRE"], {"robot": c.ROBOT_ALL, "apple": c.B_APPLE})
            car = self.bb_multi(R, rw["CAR"], {"grip": c.A_GRIP, "apple": c.B_APPLE})
            e[f"{cond}_pre_robot"], e[f"{cond}_pre_robot_arg"] = pre["robot"]
            e[f"{cond}_pre_apple"], e[f"{cond}_pre_apple_arg"] = pre["apple"]
            e[f"{cond}_car_grip"], e[f"{cond}_car_grip_arg"] = car["grip"]
            e[f"{cond}_car_apple"], e[f"{cond}_car_apple_arg"] = car["apple"]
        conds = list(c.runs)
        e["a_min"] = min(e["wu_robot"], *[e[f"{k}_t0_robot"] for k in conds])
        e["b_crate_min"] = min(e["reset_crate"], *[e[f"{k}_t0_crate"] for k in conds])
        e["b_obj_min"] = min(e["reset_apple"], e["reset_fruits"], *[e[f"{k}_t0_{s}"] for k in conds for s in ("apple", "fruits")])
        e["d_min"] = min(min(e[f"{k}_pre_robot"], e[f"{k}_pre_apple"]) for k in conds)
        e["c_min_per_run"] = {k: min(e[f"{k}_car_grip"], e[f"{k}_car_apple"]) for k in conds}
        e["c_apple_per_run"] = {k: e[f"{k}_car_apple"] for k in conds}
        c.restore()
        return e

    def legs(self, u, v, dyaw_deg, L):
        a = math.radians(dyaw_deg); return [(u - L / 2 * math.cos(a), v - L / 2 * math.sin(a)), (u + L / 2 * math.cos(a), v + L / 2 * math.sin(a))]

    def crossing(self, v):
        """x_rel where each run's recorded apple carry path first crosses y_rel = v (None if it does not)."""
        out = {}
        for cond, A in self.apath.items():
            s = np.sign(A[:, 1] - v); k = np.where(s[:-1] * s[1:] <= 0)[0]
            if not len(k):
                out[cond] = None; continue
            i = int(k[0]); y0, y1 = A[i, 1], A[i + 1, 1]; f = 0.0 if y1 == y0 else (v - y0) / (y1 - y0)
            out[cond] = float(A[i, 0] + f * (A[i + 1, 0] - A[i, 0]))
        return out


def leg_outside(xl, yl):
    """leg circle (r 0.012) entirely outside the crate's outer rectangle."""
    dx = max(abs(xl) - WALL_X, 0.0); dy = max(abs(yl) - WALL_Y, 0.0)
    return math.hypot(dx, dy) > F.LEG_R


def flags(e):
    e["pass_a"] = e["a_min"] >= 0.030
    e["pass_b"] = (e["b_crate_min"] > 0 and e["b_obj_min"] > 0 and e["client_reset_arms"] > 0.02 and e["client_reset_object"] > 0
                   and e["legs_outside_crate"])
    e["pass_c"] = all(v <= 0 for v in e["c_min_per_run"].values())
    e["pass_c_apple"] = all(v <= 0 for v in e["c_apple_per_run"].values())
    e["pass_d"] = e["d_min"] >= 0.030
    return e


def main(tag):
    t0 = time.time(); P = PCtx(tag)
    cands = []
    for av in VS:
        v = P.side * av
        for u in US:
            for dy in DYAW:
                x, y, yaw = P.to_world(u, v, dy)
                for L in LS:
                    for h in HS:
                        for r in RS:
                            cands.append({"u": u, "v": v, "dyaw_deg": dy, "x": x, "y": y, "yaw": yaw, "h": h, "L": L, "r": r})
    print(tag, "side", P.side, "candidates", len(cands), "warmup final vs row0", P.wu_final_vs_row0, flush=True)
    out = []
    for i, e in enumerate(cands):
        if i % 1000 == 0:
            print(tag, "progress", i, len(cands), f"{time.time() - t0:.0f}s", flush=True)
        e.update(P.evaluate(e["x"], e["y"], e["yaw"], e["h"], e["L"], e["r"]))
        lg = P.legs(e["u"], e["v"], e["dyaw_deg"], e["L"])
        e["legs_crate_frame"] = lg; e["legs_outside_crate"] = all(leg_outside(*p) for p in lg)
        e["bar_over_interior"] = abs(e["v"]) + e["r"] <= IN_Y
        e["inside_handles"] = abs(e["v"]) + e["r"] < 0.192
        e["drop_gap_m"] = abs(e["v"]) - e["r"] + IN_Y
        flags(e); out.append(e)
    # e
    c = P.c; ws = F.WS(c); carry = {cond: R["Q"][P.rows[cond]["CAR"]] for cond, R in c.runs.items()}
    off = ws.offset(np.vstack(list(carry.values()))); cache = {}
    for e in out:
        e["free_height_over_bar_m"] = None
        if not (e["pass_a"] and e["pass_b"] and e["pass_d"] and e["pass_c"]):
            continue
        cr = P.crossing(e["v"]); xs = [v for v in cr.values() if v is not None]
        xr = float(np.median(xs)) if xs else e["u"]
        a = math.radians(e["dyaw_deg"])  # project onto the bar axis (through (u, v)), clip to the bar
        tpar = np.clip((xr - e["u"]) * math.cos(a), -e["L"] / 2, e["L"] / 2)
        pu, pv = e["u"] + tpar * math.cos(a), e["v"] + tpar * math.sin(a)
        px, py, _ = P.to_world(pu, pv, 0)
        key = (round(px, 3), round(py, 3))
        if key not in cache:
            zs = [z for z in (ws.max_tcp_z(np.array(key), Q) for Q in carry.values()) if z is not None]
            cache[key] = max(zs) if zs else None
        tz = cache[key]
        e["crossing_crate_frame"] = [pu, pv]; e["crossing_per_run_x_rel"] = cr
        e["tcp_z_max_above_table_m"] = tz; e["tcp_minus_apple_bottom_median_m"] = off
        e["free_height_over_bar_m"] = None if tz is None else tz - off - (e["h"] + e["r"])
    for e in out:
        e["pass_e"] = bool(e["free_height_over_bar_m"] is not None and e["free_height_over_bar_m"] >= 0.05)
    c.restore()
    keys = ("pass_a", "pass_b", "pass_c", "pass_c_apple", "pass_d", "pass_e", "legs_outside_crate", "bar_over_interior")
    res = {"tag": tag, "definitions": __doc__, "side": P.side, "crate_xy": P.crate_xy.tolist(), "crate_yaw_rad": P.crate_yaw,
           "warmup_final_vs_planning_row0_maxabs_qpos": P.wu_final_vs_row0, "n_candidates": len(out),
           "windows": {cond: {k: R[k] for k in ("lift_t", "grasp_t", "place_t", "source", "run_dir")} for cond, R in c.runs.items()},
           "grid": {"v_abs": VS, "u": US, "dyaw": DYAW, "L": LS, "h": HS, "r": RS},
           "counts": {k: int(sum(bool(e[k]) for e in out)) for k in keys}, "elapsed_s": time.time() - t0, "candidates": out}
    res["counts"]["all_abcde"] = sum(e["pass_a"] and e["pass_b"] and e["pass_c"] and e["pass_d"] and e["pass_e"] for e in out)
    res["counts"]["all_abd"] = sum(e["pass_a"] and e["pass_b"] and e["pass_d"] for e in out)
    (F.RAW / f"p2search_{tag}{SUFFIX}.json").write_text(json.dumps(res, default=float))
    print(tag, res["counts"], f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
