"""T43 P: hurdle re-placement near the crate / inside the crate handle, with no effect on the robot's reset pose.
CPU, 1 process per call.  usage: psearch.py TAG [TAG ...]  -> raw/psearch_<TAG>.json

Code     : pi05_TO_hybrid bd6a002 (snapshot ${AG3S_PROJECT_ROOT}-t40h): PickPlaceObstacleManager.pair_distance (mj_geomDistance checked
           by convex_distance.certify, exact GJK fallback; capsule-box exact).  Red/white hurdle_0.  Reset scene = the client's
           pi05_infer.main() state at the obstacle reset check (flib.capture_reset, bd6a002).
Frame    : crate frame = body 'crate' at the reset scene (origin = crate body centre, axes = crate body x/y).  u = bar-centre x_rel,
           v = bar-centre y_rel (+y = the apple / left-handle side), dyaw = hurdle yaw - crate yaw (0 = bar parallel to the crate's
           +y wall).  Crate (all episodes): walls |x_rel| <= 0.094, |y_rel| <= 0.150; left handle bar y_rel 0.192..0.208,
           x_rel -0.050..0.050; handle posts x_rel +-0.058; handle arms y_rel 0.150..0.200, z 73..87 mm above table.
Grid     : u in [-0.10, 0.10], v in [0.10, 0.30] (env PS_U / PS_V / PS_DYAW override for the _ext pass), step PS_STEP (coarse pass 0.02 m; refine.py for the local pass); dyaw in {0, +-10, +-20, +-30, +-45} deg;
           bar_height h in {0.20, 0.22, ..., 0.32} m (validator range); validator xy box enforced.
Rows     : WU  = reset scene (row 0) + warmup replay (1000 mj_step from the captured reset state, hurdle parked) every 50 steps;
           T0  = recorded planning row 0 of E0 and of E3b (state after the 1000-step warmup);
           PRE = control rows 0 <= t < lift_t (approach + grasp; T40 A windows), E0 and E3b;
           CAR = control rows lift_t <= t < place_t (transport), E0 and E3b.
Geom sets: ROBOT = every collision geom under body 'base' (both arms, both grippers, torso, head, base); ARMS = the client's
           manager.robot_geom_ids (arms + grippers); GRIP = EE_BODY_L + ee_finger_l1/l2; APPLE; FRUITS = banana/orange/pear;
           CRATE = crate collision geoms (floor, walls, handles).
Metric   : min over the given rows and geom pairs (hurdle collision geom x set geom) of pair_distance, branch-and-bound on the
           lower bound dist(geom centre, hurdle geom axis segment) - hurdle radius - geom rbound (exact for every value below cap = 0.06 m; values >= cap are reported as cap).
Criteria : a' = min(ROBOT over WU and T0) >= 0.030;
           b  = CRATE, APPLE, FRUITS over WU row 0 and T0 > 0 (+ client reset check: ARMS > 0.02 stop distance, object > 0);
                flag post_in_crate = a foot centre lies inside the crate outer footprint (|x_rel| <= 0.094 and |y_rel| <= 0.150);
           c  = min(GRIP, APPLE) over CAR <= 0 in E0 AND E3b (T40 F definition);
           c_apple = min(APPLE) over CAR <= 0 in E0 AND E3b;  c_relaxed ("blocks the apple") = min(APPLE) over CAR <= 0 in E0 OR E3b;
           d  = min(ROBOT, APPLE) over PRE >= 0.030 (both runs);
           e  = free_height_over_bar (fsearch.WS: kinematic max left-TCP z above the bar xy, approach axis within 15 deg of the
                recorded carry axis, minus the median TCP-apple-bottom offset, minus the bar top h + 0.012) >= 0.05 m; computed only
                for candidates passing a', b, d and (c or c_relaxed) (fix_e.py fills the files written before this gate was widened)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, os, pathlib, sys, time
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import flib as F  # noqa: E402
from fsearch import WS  # noqa: E402

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/raw'))
CAP = 0.06
STEP = float(os.environ.get("PS_STEP", "0.02")); SUFFIX = os.environ.get("PS_SUFFIX", "")
_U = [float(x) for x in os.environ.get("PS_U", "-0.10,0.10").split(",")]; _V = [float(x) for x in os.environ.get("PS_V", "0.10,0.30").split(",")]
US = np.round(np.arange(_U[0], _U[1] + 1e-5, STEP), 3)
VS = np.round(np.arange(_V[0], _V[1] + 1e-5, STEP), 3)
DYAW = [int(x) for x in os.environ.get("PS_DYAW", "0,10,-10,20,-20,30,-30,45,-45").split(",")]
HS = np.round(np.arange(0.20, 0.32001, 0.02), 3)
WALL_X, WALL_Y = 0.094, 0.150


class PCtx:
    def __init__(self, tag):
        self.c = c = F.Ctx(tag); m, d = c.m, c.d
        c.restore()
        bc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate")
        self.crate_xy = d.xpos[bc][:2].copy(); Rc = d.xmat[bc].reshape(3, 3)
        self.crate_yaw = math.atan2(Rc[1, 0], Rc[0, 0])
        self.ARMS = c.ARMS
        # warmup replay
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

    def to_world(self, u, v, dyaw_deg):
        cy, sy = math.cos(self.crate_yaw), math.sin(self.crate_yaw)
        x = self.crate_xy[0] + cy * u - sy * v; y = self.crate_xy[1] + sy * u + cy * v
        return x, y, (self.crate_yaw + math.radians(dyaw_deg))

    def to_crate(self, x, y):
        cy, sy = math.cos(self.crate_yaw), math.sin(self.crate_yaw); dx, dy = x - self.crate_xy[0], y - self.crate_xy[1]
        return cy * dx + sy * dy, -sy * dx + cy * dy

    def bb_multi(self, P, rows, sets, cap=CAP):
        """branch-and-bound min pair_distance(hurdle geom, geom) over rows of pose cache P (dict xpos/xmat over c.ALL), for each
        geom set in `sets` (name -> geom ids).  returns name -> (value, (row index into P, hurdle geom name, geom name) or None);
        value capped at cap.  Lower bound: every hurdle geom is a capsule or cylinder, i.e. inside the capsule (axis segment,
        radius size[0]); bound = dist(other geom centre, axis segment) - size[0] - rbound(other) <= true distance.  A coarse
        bound (one sphere around all hurdle segments) skips (row, geom) entries before the per-segment bound."""
        c = self.c; m, d = c.m, c.d
        H = c.HURDLE
        ax = d.geom_xmat[H].reshape(-1, 3, 3)[:, :, 2]; hl = m.geom_size[H, 1][:, None]
        A0 = d.geom_xpos[H] - ax * hl; AB = 2 * ax * hl; hr = m.geom_size[H, 0]
        ends = np.vstack([A0, A0 + AB]); hc = 0.5 * (ends.min(0) + ends.max(0))
        Rh = float(np.max(np.linalg.norm(ends - hc, axis=1) + np.r_[hr, hr]))
        allg = np.unique(np.concatenate([np.asarray(v, int) for v in sets.values()]))
        ix = np.array([c.pos_in_all[int(g)] for g in allg], int); gr = m.geom_rbound[c.ALL[ix]]
        X = P["xpos"][rows][:, ix]  # (R, G, 3)
        coarse = np.linalg.norm(X - hc, axis=2) - gr[None] - Rh
        rr, gg = np.nonzero(coarse <= cap)
        W = X[rr, gg][:, None, :] - A0[None]  # (N, H, 3)
        tt = np.clip((W * AB[None]).sum(-1) / (AB * AB).sum(-1)[None], 0.0, 1.0)
        gapN = np.linalg.norm(W - tt[..., None] * AB[None], axis=2) - gr[gg][:, None] - hr[None]  # (N, H)
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

    def evaluate(self, x, y, yaw, h, full=True):
        c = self.c
        c.set_hurdle(x, y, yaw, h)
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


def flags(e):
    e["pass_a"] = e["a_min"] >= 0.030
    e["pass_b"] = e["b_crate_min"] > 0 and e["b_obj_min"] > 0 and e["client_reset_arms"] > 0.02 and e["client_reset_object"] > 0
    e["pass_c"] = all(v <= 0 for v in e["c_min_per_run"].values())
    e["pass_c_apple"] = all(v <= 0 for v in e["c_apple_per_run"].values())
    e["pass_c_relaxed"] = any(v <= 0 for v in e["c_apple_per_run"].values())
    e["pass_d"] = e["d_min"] >= 0.030
    return e


def foot_centres(x, y, yaw):
    u = np.array([math.cos(yaw), math.sin(yaw)]); p = np.array([x, y])
    return p - 0.12 * u, p + 0.12 * u


def main(tag):
    t0 = time.time(); P = PCtx(tag)
    cands = []
    for u in US:
        for v in VS:
            for dy in DYAW:
                x, y, yaw = P.to_world(float(u), float(v), dy)
                if not (0.38 <= x <= 0.85 and -0.35 <= y <= 0.35):
                    continue
                for h in HS:
                    cands.append({"u": float(u), "v": float(v), "dyaw_deg": dy, "x": x, "y": y, "yaw": yaw, "h": float(h)})
    print(tag, "candidates", len(cands), f"crate xy {P.crate_xy.round(4)} yaw {math.degrees(P.crate_yaw):.3f} deg",
          "warmup final vs row0", P.wu_final_vs_row0, flush=True)
    run(tag, P, cands, SUFFIX, t0)


def run(tag, P, cands, suffix, t0, extra=None):
    c = P.c
    out = []
    for i, e in enumerate(cands):
        if i % 2000 == 0:
            print(tag, "progress", i, len(cands), f"{time.time() - t0:.0f}s", flush=True)
        r = P.evaluate(e["x"], e["y"], e["yaw"], e["h"])
        e.update(r)
        fl, fr = foot_centres(e["x"], e["y"], e["yaw"])
        e["post_in_crate"] = any(abs(a) <= WALL_X and abs(b) <= WALL_Y for a, b in (P.to_crate(*fl), P.to_crate(*fr)))
        e["inside_handle"] = e["v"] <= 0.208
        flags(e); out.append(e)
    # e: detour room for candidates passing a, b, d and c_relaxed
    ws = WS(c); carry = {cond: R["Q"][P.rows[cond]["CAR"]] for cond, R in c.runs.items()}
    off = ws.offset(np.vstack(list(carry.values()))); cache = {}
    for e in out:
        e["free_height_over_bar_m"] = None
        if not (e["pass_a"] and e["pass_b"] and e["pass_d"] and (e["pass_c"] or e["pass_c_relaxed"])):
            continue
        key = (round(e["x"], 3), round(e["y"], 3))
        if key not in cache:
            zs = [z for z in (ws.max_tcp_z(np.array(key), Q) for Q in carry.values()) if z is not None]
            cache[key] = max(zs) if zs else None
        tz = cache[key]
        e["tcp_z_max_above_table_m"] = tz; e["tcp_minus_apple_bottom_median_m"] = off
        e["free_height_over_bar_m"] = None if tz is None else tz - off - (e["h"] + 0.012)
    for e in out:
        e["pass_e"] = e["free_height_over_bar_m"] is not None and e["free_height_over_bar_m"] >= 0.05
    c.restore()
    res = {"tag": tag, "definitions": __doc__, "crate_xy": P.crate_xy.tolist(), "crate_yaw_rad": P.crate_yaw,
           "warmup_final_vs_planning_row0_maxabs_qpos": P.wu_final_vs_row0, "n_candidates": len(out),
           "counts": {k: int(sum(e[k] for e in out)) for k in ("pass_a", "pass_b", "pass_c", "pass_c_apple", "pass_c_relaxed", "pass_d", "pass_e", "post_in_crate")},
           "elapsed_s": time.time() - t0, **(extra or {}), "candidates": out}
    full = [e for e in out if e["pass_a"] and e["pass_b"] and e["pass_c"] and e["pass_d"] and e["pass_e"]]
    rel = [e for e in out if e["pass_a"] and e["pass_b"] and e["pass_c_relaxed"] and e["pass_d"] and e["pass_e"]]
    res["counts"]["all_abcde"] = len(full); res["counts"]["all_ab_crelaxed_de"] = len(rel)
    (RAW / f"psearch_{tag}{suffix}.json").write_text(json.dumps(res, default=float))
    print(tag, res["counts"], f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
