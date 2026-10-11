"""T40 A step 3-4: candidate placement search + exact replay (verifier, CPU, MuJoCo only).

Window: lift_t = first control row after the last row (<= grasp_t) whose apple centre is < z0 + 2 mm; carry window
[lift_t, place_t] (grasp_t / place_t = T39 analysis.json, planning t_step).  'carry' in the task = grasp_t..place_t is
reported separately; the search uses [lift_t, place_t] (adds the vertical lift rows grasp_t excludes).
Proxy (search only): apple = sphere r 33.3 mm at the recorded apple centre; bollard = post cylinder r27 z[10,287] mm +
foot r45 z[0,20] mm; divider = box 208x36x80 mm (feet ignored). Score of a placement = max(dmin_E0, dmin_E3b) (proxy
min distance over the window; lower = blocks both runs deeper). With --either: score = min(dmin_E0, dmin_E3b)
(blocks at least one recorded path); output file suffix _either.
Candidates:
  bollard : 2.5 mm grid, x in [0.38, 0.85], y in [-0.35, 0.35] (validator bounds), yaw 0.
  divider : (i) every recorded apple xy in the window of either run, yaw = local xy tangent (+-2 rows) + 90 deg
            (tangent < 1 mm/row -> direction apple t0 -> crate centre); (ii) 2.5 mm grid, yaw = (apple t0 -> crate) + 90 deg.
Hard filters (GJK separation, t40a_lib / gjk.py, at the t=0 scene of BOTH runs, i.e. planning row 0 qpos):
  obstacle <-> each fruit (apple, banana, orange, pear) > 20 mm, <-> crate > 20 mm, <-> all robot geoms > 20 mm;
  obstacle origin xy not inside the crate outer wall footprint (|dx| < 0.094, |dy| < 0.154 crate frame).
(v2) the first K = 5 candidates in proxy-score order that pass the hard filters are listed, whether or not the proxy
blocks (the proxy is apple-only; the exact replay also counts the gripper). They are listed (stage 1, all 22 pairs); with --replay they get the exact replay, every control row 0..599 of E0 and E3b: min distance obstacle <->
A_gripper / B_apple / C_robot (t40a_lib, GJK signed distance). Best = lowest max (--either: min) over the two runs of
the window minimum of min(A, B); blocking = that value <= 0."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, sys, pathlib
import numpy as np, mujoco
import t40a_lib as L

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/A/raw'))
AR = 0.0333
K = 5
CLR = 0.020
EITHER = False  # --either: score = min(dmin_E0, dmin_E3b) (blocks at least one recorded path); default max (blocks both)


def proxy_bollard(c, P):  # c (N,2) centres, P (F,3) apple centres -> (N,) min distance
    out = np.full(len(c), np.inf)
    for (R, z0, z1) in ((0.027, 0.010, 0.287), (0.045, 0.0, 0.020)):
        z0 += L.TABLE_Z; z1 += L.TABLE_Z
        dxy = np.linalg.norm(c[:, None, :] - P[None, :, :2], axis=2) - R
        dz = np.maximum(np.maximum(z0 - P[:, 2], P[:, 2] - z1), 0)[None, :]
        inside = np.minimum(np.minimum(P[:, 2] - z0, z1 - P[:, 2]), 0)  # unused
        dist = np.where(dxy > 0, np.sqrt(np.maximum(dxy, 0) ** 2 + dz ** 2),
                        np.where(dz > 0, dz, np.maximum(dxy, -np.minimum(P[:, 2] - z0, z1 - P[:, 2])[None, :]))) - AR
        out = np.minimum(out, dist.min(axis=1))
    return out


def proxy_divider(c, yaw, P):
    h = np.array([0.104, 0.018, 0.040])
    out = np.empty(len(c))
    for i in range(len(c)):
        cy, sy = math.cos(yaw[i]), math.sin(yaw[i])
        rel = P - np.array([c[i, 0], c[i, 1], L.TABLE_Z + 0.040])
        loc = np.stack([cy * rel[:, 0] + sy * rel[:, 1], -sy * rel[:, 0] + cy * rel[:, 1], rel[:, 2]], 1)
        q = np.abs(loc) - h
        dist = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - AR
        out[i] = dist.min()
    return out


def t0_check(slot, xy, yaw, runs):
    """full t=0 clearance values (GJK signed distance, no early stop)."""
    res = {}
    for cond, R in runs.items():
        L.set_q(R["q0"]); L.place(slot, xy, yaw); mujoco.mj_kinematics(L.m, L.d)
        r = {"apple": L.min_dist(L.OB[slot], L.B_APPLE)}
        for n, g in L.F_FRUITS.items():
            r[n] = L.min_dist(L.OB[slot], g)
        r["crate"] = L.min_dist(L.OB[slot], L.CR_CRATE)
        r["robot"] = L.min_dist(L.OB[slot], L.ROBOT_ALL)
        res[cond] = r
    return res


_CTX = {}


def _ctx(cond, R):
    key = (cond, id(R))
    if key not in _CTX:
        L.set_q(R["q0"])
        sets = [("apple", L.B_APPLE)] + list(L.F_FRUITS.items()) + [("crate", L.CR_CRATE), ("robot", L.ROBOT_ALL)]
        _CTX[key] = [(n, [(int(g), L.d.geom_xpos[g].copy(), float(L.m.geom_rbound[g]), L.gjk.make_support(L.m, L.d, int(g))) for g in gs]) for n, gs in sets]
    return _CTX[key]


def t0_fast(slot, xy, yaw, runs):
    """filter: True if every obstacle geom is > CLR from every scene geom at both runs' t=0 (bounding-sphere accept,
    support-point upper-bound reject, GJK otherwise)."""
    for cond, R in runs.items():
        ctx = _ctx(cond, R)
        L.set_q(R["q0"]); L.place(slot, xy, yaw); mujoco.mj_kinematics(L.m, L.d)
        obs = [(L.d.geom_xpos[o].copy(), float(L.m.geom_rbound[o]), L.gjk.make_support(L.m, L.d, int(o))) for o in L.OB[slot]]
        for n, lst in ctx:
            for g, cg, rg, sg in lst:
                for co, ro, so in obs:
                    v = cg - co
                    if np.linalg.norm(v) - ro - rg > CLR:
                        continue
                    if np.linalg.norm(sg(-v) - so(v)) <= CLR:
                        return False
                    if L.gjk.gjk_dist(so, sg, v + 1e-9) <= CLR:
                        return False
    return True


def in_crate_fp(xy, op):
    c = np.array(op["crate"]["pos"]); q = op["crate"]["quat"]; yaw = 2 * math.atan2(q[3], q[0])
    dx, dy = xy[0] - c[0], xy[1] - c[1]
    lx, ly = math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy
    return abs(lx) < 0.094 and abs(ly) < 0.154


def replay(slot, xy, yaw, R):
    L.place(slot, xy, yaw)
    A, B, C, Cp = [], [], [], []
    for q in R["q"]:
        L.set_q(q)
        A.append(L.min_dist(L.OB[slot], L.A_GRIP)); B.append(L.min_dist(L.OB[slot], L.B_APPLE))
        c, p = L.min_dist(L.OB[slot], L.C_ROBOT, want_pair=True); C.append(c); Cp.append(p)
    return np.array(A), np.array(B), np.array(C), Cp


def window(R, grasp_t, place_t):
    z = R["q"][:, 40]; z0 = R["q0"][40]
    gi = int(np.searchsorted(R["t"], grasp_t))
    k = gi
    while k > 0 and z[k - 1] >= z0 + 0.002:
        k -= 1
    return int(R["t"][k]), grasp_t, place_t


def summarize(t, A, B, C, Cp, lift_t, grasp_t, place_t):
    ph = {"approach": t < lift_t, "lift": (t >= lift_t) & (t < grasp_t), "carry_grasp_to_place": (t >= grasp_t) & (t <= place_t),
          "window_lift_to_place": (t >= lift_t) & (t <= place_t), "after_place": t > place_t, "all": t >= 0}
    out = {}
    for k, mk in ph.items():
        if not mk.any():
            out[k] = None; continue
        e = {}
        for nm, arr in (("A_gripper", A), ("B_apple", B), ("C_robot", C)):
            i = int(np.argmin(np.where(mk, arr, np.inf)))
            e[nm] = {"min_m": float(arr[i]), "t": int(t[i]), "n_rows_le_0": int(((arr <= 0) & mk).sum())}
            if nm == "C_robot":
                e[nm]["pair"] = Cp[i]
        out[k] = e
    return out


def run_pair(ent, do_replay=False):
    ep, seed = ent["episode"], ent["seed"]
    runs, win = {}, {}
    for cond in ("E0", "E3b"):
        rr = ent["runs"][cond]
        runs[cond] = L.load_run(rr["run_dir"])
        win[cond] = window(runs[cond], rr["grasp_t"], rr["place_t"])
    P = {c: runs[c]["q"][(runs[c]["t"] >= win[c][0]) & (runs[c]["t"] <= win[c][2]), 38:41] for c in runs}
    op = runs["E0"]["object_poses_t0"]
    a0 = np.array(op["apple"]["pos"][:2]); cc = np.array(op["crate"]["pos"][:2])
    gdir = cc - a0; yaw_perp_global = math.atan2(gdir[1], gdir[0]) + math.pi / 2
    gx, gy = np.meshgrid(np.arange(0.38, 0.85001, 0.0025), np.arange(-0.35, 0.35001, 0.0025))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    cands = {}
    # bollard
    AGG = np.minimum if EITHER else np.maximum
    sc = AGG(proxy_bollard(grid, P["E0"]), proxy_bollard(grid, P["E3b"]))
    cands["bollard_0"] = [(grid[i], 0.0, float(sc[i])) for i in np.argsort(sc)]
    # divider
    dc, dy = [], []
    for c in runs:
        Q = P[c]
        for i in range(len(Q)):
            tg = Q[min(i + 2, len(Q) - 1), :2] - Q[max(i - 2, 0), :2]
            yaw = (math.atan2(tg[1], tg[0]) if np.linalg.norm(tg) > 0.004 else math.atan2(gdir[1], gdir[0])) + math.pi / 2
            dc.append(Q[i, :2]); dy.append(yaw)
    dc = np.concatenate([np.array(dc), grid]); dy = np.concatenate([np.array(dy), np.full(len(grid), yaw_perp_global)])
    sc = AGG(proxy_divider(dc, dy, P["E0"]), proxy_divider(dc, dy, P["E3b"]))
    cands["divider_0"] = [(dc[i], float(dy[i]), float(sc[i])) for i in np.argsort(sc)]
    out = {"mode": "either" if EITHER else "both", "episode": ep, "seed": seed, "windows": {c: dict(zip(("lift_t", "grasp_t", "place_t"), win[c])) for c in win},
           "crate_xy": cc.tolist(), "apple_t0_xy": a0.tolist(), "kinds": {}}
    for slot, lst in cands.items():
        n_block_proxy = sum(1 for x in lst if x[2] <= 0)
        kept, n_checked, rej = [], 0, {"t0_clearance": 0, "in_crate": 0}
        for xy, yaw, s in lst:
            if len(kept) >= K or n_checked >= 40000:
                break
            n_checked += 1
            if in_crate_fp(xy, op):
                rej["in_crate"] += 1; continue
            if not t0_fast(slot, xy, yaw, runs):
                rej["t0_clearance"] += 1; continue
            kept.append((xy, yaw, s, t0_check(slot, xy, yaw, runs)))
        best_unconstrained = lst[0]
        ev = []
        for xy, yaw, s, t0 in kept:
            if not do_replay:
                ev.append({"xy": [float(xy[0]), float(xy[1])], "yaw_rad": yaw, "proxy_score_m": s, "t0_clearance_m": t0}); continue
            e = {"xy": [float(xy[0]), float(xy[1])], "yaw_rad": yaw, "proxy_score_m": s, "t0_clearance_m": t0, "replay": {}}
            for c in runs:
                A, B, C, Cp = replay(slot, xy, yaw, runs[c])
                e["replay"][c] = summarize(runs[c]["t"], A, B, C, Cp, *win[c])
                e["replay"][c]["_series"] = {"t": runs[c]["t"].tolist(), "A": A.tolist(), "B": B.tolist(), "C": C.tolist()}
            w = [min(e["replay"][c]["window_lift_to_place"]["A_gripper"]["min_m"], e["replay"][c]["window_lift_to_place"]["B_apple"]["min_m"]) for c in runs]
            e["exact_score_m"] = (min if EITHER else max)(w)
            e["blocks_E0"] = w[0] <= 0; e["blocks_E3b"] = w[1] <= 0
            ev.append(e)
        ev.sort(key=lambda e: e.get("exact_score_m", e["proxy_score_m"]))
        out["kinds"][slot] = {"n_candidates": len(lst), "n_proxy_blocking_both": n_block_proxy, "n_checked": n_checked,
                              "rejected": rej, "n_t0_feasible_listed": len(kept),
                              "n_proxy_blocking_among_listed": sum(1 for x in kept if x[2] <= 0),
                              "best_unconstrained_proxy": {"xy": best_unconstrained[0].tolist(), "yaw": best_unconstrained[1], "score_m": best_unconstrained[2]},
                              "evaluated": ev, "best": ev[0] if ev else None,
                              "best_is_blocking_candidate": bool(ev) and ("exact_score_m" in ev[0]) and ev[0]["exact_score_m"] <= 0}
    return out


if __name__ == "__main__":
    allp = json.load(open(RAW / "carry_paths_22pairs.json"))["pairs"]
    do_replay = "--replay" in sys.argv
    EITHER = "--either" in sys.argv
    sel = [a for a in sys.argv[1:] if not a.startswith("--")]
    for ent in allp:
        tag = f"{ent['episode']}_{ent['seed']}"
        if sel and tag not in sel:
            continue
        o = run_pair(ent, do_replay)
        o["mj_check"] = dict(L.MJ_CHECK)
        (RAW / f"{'replay' if do_replay else 'search'}_{tag}{'_either' if EITHER else ''}.json").write_text(json.dumps(o))
        for slot, k in o["kinds"].items():
            b = k["best"]
            print(tag, slot, "proxy_block_both", k["n_proxy_blocking_both"], "checked", k["n_checked"], "rej", k["rejected"],
                  "listed", k["n_t0_feasible_listed"], "proxy_blocking_listed", k["n_proxy_blocking_among_listed"],
                  "best(exact<=0 both or either per mode)" if k["best_is_blocking_candidate"] else "best(not blocking)",
                  None if b is None else (np.round(b["xy"], 3).tolist(), round(math.degrees(b["yaw_rad"]), 1), "proxy", round(1e3 * b["proxy_score_m"], 1), "exact", round(1e3 * b.get("exact_score_m", np.nan), 1),
                                          "t0min", round(1e3 * min(min(v.values()) for v in b["t0_clearance_m"].values()), 1)), flush=True)
