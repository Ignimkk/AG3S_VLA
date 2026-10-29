"""T43 P2 step 2: choose one overbar placement per episode, write the config (red rgba), cross-check.
usage: select_p2.py TAG [TAG ...]  -> configs/t43p2_ep<EP>_s<SEED>.json, raw/select_p2_<TAG>.json

Pool     : raw/p2search_<TAG>.json (+ _refine when present).
Main     : pass a', b (incl. legs outside the crate), c (GRIP or APPLE hits the bar in E0 AND E3b), d, e.
Ranking  : (1) margin = min(a_min, d_min) capped at 0.06 m, rounded to 1 mm (larger first); (2) c_apple (the apple itself hits
           the bar in both runs) first; (3) place_spot_clear first: in both runs the recorded final apple centre lies at least
           r + 0.035 m (apple half-size) beyond the bar axis toward the crate centre (crate-frame y); (4) |dyaw| smaller first;
           (5) depth_ok first: max over runs of min(GRIP, APPLE) over CAR <= -0.020 m (the recorded path is >= 20 mm inside the
           bar in both runs); (6) larger |v| (closer to the apple-side end wall); (7) smaller |u| (legs symmetric about the
           crate); (8) crate clearance b_crate_min capped at 0.030 m, rounded to 1 mm; (9) deeper blocking; (10) larger
           free_height_over_bar.  (v1 of this ranking, raw/select_v1/, put the blocking depth rounded to 5 mm before |v| and
           had no |u| term.)
Trade-off: when no candidate passes: best value of each criterion with the others held (a_best_given_bcd, d_best_given_abc,
           c_miss_best_given_abd = smallest max-over-runs CAR minimum among a', b, d passers) and a maximin candidate (worst slack
           of min(a_min - 0.030, d_min - 0.030, b_crate_min, b_obj_min, client_reset_arms - 0.020, -max c) among b passers).
Cross-check of the written config: (1) manager.activate (the client's call) vs the search's _place path: robot/object reset
           clearance and every overbar geom pose; (2) brute force (every row, every overbar x set geom pair, no bound) minima for
           a_min, d_min, b_crate_min, CAR per run, vs the search values; (3) independent numpy GJK (T40 A gjk.signed_dist) at the
           argmin pair of a_min, d_min, b_crate_min and each run's CAR minimum; (4) dryrun_p2.py = the fce4af8 client's own reset
           check (pi05_infer.main() up to the policy call, --obstacle-stop-distance 0.02)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, subprocess, sys
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P2/p'))
import p2lib as F  # noqa: E402
import p2search as S  # noqa: E402

CFG = F.ROOT / "configs"; PY = _ag3s_workspace_path('legacy/.venv-openpi-live/bin/python')
KEYS = ("u", "v", "dyaw_deg", "x", "y", "yaw", "h", "L", "r", "a_min", "wu_robot_arg", "d_min", "b_crate_min", "reset_crate_arg", "b_obj_min",
        "client_reset_arms", "client_reset_object", "c_min_per_run", "c_apple_per_run", "free_height_over_bar_m",
        "tcp_z_max_above_table_m", "tcp_minus_apple_bottom_median_m", "crossing_crate_frame", "drop_gap_m", "legs_crate_frame", "place_spot_clear",
        "legs_outside_crate", "bar_over_interior", "inside_handles", "pass_a", "pass_b", "pass_c", "pass_c_apple", "pass_d", "pass_e")


def short(e):
    return None if e is None else {k: e.get(k) for k in KEYS} | {"margin": e["margin"]}


def load(tag):
    pool, src = [], {}
    for suf in ("", "_refine"):
        f = F.RAW / f"p2search_{tag}{suf}.json"
        if f.exists():
            d = json.loads(f.read_text()); src[suf or "coarse"] = {"n": d["n_candidates"], "counts": d["counts"], "grid": d["grid"]}
            pool += d["candidates"]
    side = d["side"]; cyaw = d["crate_yaw_rad"]; cxy = np.array(d["crate_xy"])
    yfin = {}
    for cond, w in d["windows"].items():
        rows = [json.loads(l) for l in open(pathlib.Path(w["run_dir"]) / "frames.jsonl") if '"control"' in l]
        q = rows[-1]["qpos"]; dx, dy = q[38] - cxy[0], q[39] - cxy[1]
        yfin[cond] = -math.sin(cyaw) * dx + math.cos(cyaw) * dy
    for e in pool:
        e["margin"] = min(e["a_min"], e["d_min"])
        e["place_spot_clear"] = all(side * (e["v"] - y) >= e["r"] + 0.035 for y in yfin.values())
    return pool, src, d | {"apple_final_y_rel": yfin}


def key_main(e):
    cmax = max(e["c_min_per_run"].values())
    return (round(min(e["margin"], 0.06), 3), e["pass_c_apple"], e["place_spot_clear"], -abs(e["dyaw_deg"]), cmax <= -0.020,
            round(abs(e["v"]), 3), -round(abs(e["u"]), 3), round(min(e["b_crate_min"], 0.030), 3), -cmax, e["free_height_over_bar_m"] or -1)


def tradeoff(pool):
    t = {}
    sel = lambda f: [e for e in pool if f(e)]
    P = sel(lambda e: e["pass_b"] and e["pass_c"] and e["pass_d"]); t["a_best_given_bcd"] = max((e["a_min"] for e in P), default=None)
    P = sel(lambda e: e["pass_a"] and e["pass_b"] and e["pass_c"]); t["d_best_given_abc"] = max((e["d_min"] for e in P), default=None)
    P = sel(lambda e: e["pass_a"] and e["pass_b"] and e["pass_d"])
    t["c_miss_best_given_abd"] = min((max(e["c_min_per_run"].values()) for e in P), default=None)
    t["c_apple_miss_best_given_abd"] = min((max(e["c_apple_per_run"].values()) for e in P), default=None)
    P = sel(lambda e: e["pass_a"] and e["pass_b"] and e["pass_c"] and e["pass_d"])
    t["e_best_given_abcd"] = max((e["free_height_over_bar_m"] for e in P if e["free_height_over_bar_m"] is not None), default=None)
    t["n_abcd"] = len(P)

    def slack(e):
        s = {"a": e["a_min"] - 0.030, "d": e["d_min"] - 0.030, "b_crate": e["b_crate_min"], "b_obj": e["b_obj_min"],
             "reset_arms": e["client_reset_arms"] - 0.020, "c": -max(e["c_min_per_run"].values())}
        k = min(s, key=s.get); return s[k], k
    P = sel(lambda e: e["legs_outside_crate"])
    if P:
        best = max(P, key=lambda e: slack(e)[0]); t["maximin"] = short(best) | {"slack": slack(best)}
    return t


def brute(P, c, rows_P, geoms):
    """min over rows of every (overbar geom, geom) pair_distance (no bound), cap 0.06; returns value, (row, ob, geom)."""
    m, d = c.m, c.d; best, arg = S.CAP, None; ft = np.zeros(6)
    for i in rows_P:
        for g in geoms:
            a = c.pos_in_all[int(g)]; d.geom_xpos[g] = P["xpos"][i][a]; d.geom_xmat[g] = P["xmat"][i][a]
            for o in c.OB:
                v = c.mgr.pair_distance(int(o), int(g), S.CAP, ft)
                if v < best:
                    best, arg = float(v), (int(i), c.gname(o), c.gname(g))
    return best, arg


def gjk_at(P, c, i, ob_name, g_name):
    from gjk import signed_dist
    m, d = c.m, c.d
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g_name); o = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, ob_name)
    a = c.pos_in_all[int(g)]; d.geom_xpos[g] = P["xpos"][i][a]; d.geom_xmat[g] = P["xmat"][i][a]
    return float(signed_dist(m, d, o, g)), float(c.mgr.pair_distance(o, g, S.CAP, np.zeros(6)))


def crosscheck(PC, e):
    c = PC.c; out = {}
    c.set_overbar(e["x"], e["y"], e["yaw"], e["h"], e["L"], e["r"]); pa = c.d.geom_xpos[c.OB].copy(); ma = c.d.geom_xmat[c.OB].copy()
    r_place = c.reset_check()
    c.set_overbar(e["x"], e["y"], e["yaw"], e["h"], e["L"], e["r"], exact=True); pb = c.d.geom_xpos[c.OB].copy(); mb = c.d.geom_xmat[c.OB].copy()
    r_act = c.reset_check()
    out["activate_vs_place"] = {"geom_xpos_maxabs": float(np.abs(pa - pb).max()), "geom_xmat_maxabs": float(np.abs(ma - mb).max()),
                                "reset_check_place": r_place, "reset_check_activate": r_act,
                                "ob_geom_size": c.m.geom_size[c.OB, :2].tolist(), "ob_rgba": c.m.geom_rgba[list(c.mgr._visual_geoms[F.SLOT])].tolist()}
    W = PC.WU; allw = np.arange(len(W["Q"]))
    bf = {}
    bf["wu_robot"] = brute(W, c, allw, c.ROBOT_ALL)
    bf["reset_crate"] = brute(W, c, [0], c.CRATE)
    for cond, R in c.runs.items():
        rw = PC.rows[cond]
        bf[f"{cond}_t0_robot"] = brute(R, c, rw["T0"], c.ROBOT_ALL); bf[f"{cond}_t0_crate"] = brute(R, c, rw["T0"], c.CRATE)
        bf[f"{cond}_pre"] = min(brute(R, c, rw["PRE"], c.ROBOT_ALL), brute(R, c, rw["PRE"], c.B_APPLE), key=lambda t: t[0])
        bf[f"{cond}_car"] = min(brute(R, c, rw["CAR"], c.A_GRIP), brute(R, c, rw["CAR"], c.B_APPLE), key=lambda t: t[0])
    conds = list(c.runs)
    out["bruteforce"] = {"a_min": min(bf["wu_robot"][0], *[bf[f"{k}_t0_robot"][0] for k in conds]),
                         "d_min": min(bf[f"{k}_pre"][0] for k in conds),
                         "b_crate_min": min(bf["reset_crate"][0], *[bf[f"{k}_t0_crate"][0] for k in conds]),
                         "c_min_per_run": {k: bf[f"{k}_car"][0] for k in conds}, "args": {k: v[1] for k, v in bf.items()}}
    out["search"] = {"a_min": e["a_min"], "d_min": e["d_min"], "b_crate_min": e["b_crate_min"], "c_min_per_run": e["c_min_per_run"]}
    g = {}
    for k, (val, arg) in bf.items():
        if arg is None:
            continue
        cond = k.split("_")[0]; P = W if cond not in c.runs else c.runs[cond]
        g[k] = {"pair": arg, "manager": val, "gjk_signed": gjk_at(P, c, arg[0], arg[1], arg[2])[0]}
    out["gjk_at_argmin"] = g
    c.restore()
    return out


def dryrun(ep, cfg, prof, tag):
    o = F.RAW / f"dryrun_p2_{tag}.json"
    subprocess.run([PY, str(F.ROOT / "p/dryrun_p2.py"), str(ep), str(cfg), prof, str(o)], check=False, capture_output=True, text=True,
                   env={"MUJOCO_GL": "osmesa", "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": _ag3s_workspace_path(''),
                        "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
    return json.loads(o.read_text()) if o.exists() else None


def main(tag):
    pool, src, last = load(tag)
    P = [e for e in pool if e["pass_a"] and e["pass_b"] and e["pass_c"] and e["pass_d"] and e["pass_e"]]
    best = max(P, key=key_main) if P else None
    res = {"tag": tag, "selection": __doc__, "sources": src, "n_pool": len(pool), "n_pass_main": len(P), "chosen": short(best),
           "tradeoff": tradeoff(pool), "side": last["side"], "crate_xy": last["crate_xy"], "crate_yaw_rad": last["crate_yaw_rad"], "windows": last["windows"], "apple_final_y_rel": last["apple_final_y_rel"]}
    ep, seed = tag.split("_")
    if best is not None:
        PC = S.PCtx(tag)
        res["crosscheck"] = crosscheck(PC, best)
        prof = "t43p2_overbar"
        cfg = CFG / f"t43p2_ep{ep}_s{seed}.json"
        F.write_config(cfg, prof, best["x"], best["y"], best["yaw"], best["h"], best["L"], best["r"],
                       f"T43 P2: overbar_0 across the crate inside the handles (ep{ep}); red rgba {F.RED}")
        res["config"] = str(cfg); res["dryrun"] = dryrun(ep, cfg, prof, tag)
    (F.RAW / f"select_p2_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(tag, "n_pass", len(P), json.dumps(short(best), default=float)[:600], flush=True)
    if best is not None:
        print("  crosscheck", json.dumps({k: res["crosscheck"][k] for k in ("bruteforce", "search")}, default=float)[:800])
        print("  dryrun", json.dumps(res["dryrun"], default=float)[:400], flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
