"""T43 P step 2: choose one placement per episode from raw/psearch_<TAG>[_refine].json, write configs, cross-check.
usage: select_p.py TAG [TAG ...]  -> configs/t43p_ep<EP>_s<SEED>[_crelax].json, raw/select_p_<TAG>.json

Pool      : psearch_<TAG>.json (coarse, 0.02 m) + psearch_<TAG>_refine.json (refine.py) + psearch_<TAG>_ext.json (run_ext.sh) when present.
margin    : min(a_min, d_min) (reset/warmup/t0 robot clearance, approach+grasp clearance of robot and apple).
Main      : pass a', b, c, d, e (psearch definitions).  Relaxed: c replaced by c_relaxed (apple hits the bar in E0 or E3b).
Ranking   : (1) margin, capped at the 0.06 m distance cap and rounded to 1 mm; (2) main only: the apple itself hits the bar in both
            runs (c_apple) first; (3) crate clearance b_crate_min capped at 0.010 m, rounded to 1 mm; (4) deepest blocking (main:
            smaller max over runs of min(GRIP, APPLE) over CAR; relaxed: smaller min over runs of APPLE over CAR); (5) larger
            free_height_over_bar.
Trade-off : always reported (best value of each criterion with the other criteria held):
            a_best_given_bcd, d_best_given_abc, margin_best_given_bc (a and d both free), c_miss_best_given_abd (smallest
            max-over-runs min(GRIP, APPLE) over CAR among a', b, d passers; > 0 = the recorded path misses the bar by that much),
            c_apple_miss_best_given_abd (same with APPLE only, min over runs = relaxed c), n_post_in_crate_given_acd;
            maximin_given_c / maximin_given_crelaxed: the candidate maximising the worst slack min(a_min - 0.030, d_min - 0.030,
            b_crate_min, b_obj_min, client_reset_arms - 0.020) among c (resp. c_relaxed) passers, with the limiting criterion.
Cross-check of each written config: (1) independent numpy GJK (gjk.py, T40 A) at the argmin row/pair of a_min, d_min, b_crate,
            and the CAR minima vs the client distance; (2) dryrun.py = the client's own reset check (bd6a002 pi05_infer.main() up to the
            policy call, default --obstacle-stop-distance 0.02)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, sys
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import flib as F, gjk  # noqa: E402
import psearch as S  # noqa: E402

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/raw')); CFG = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/configs'))
PY = _ag3s_workspace_path('legacy/.venv-openpi-live/bin/python')
KEYS = ("u", "v", "dyaw_deg", "x", "y", "yaw", "h", "a_min", "d_min", "b_crate_min", "b_obj_min", "client_reset_arms", "client_reset_object",
        "c_min_per_run", "c_apple_per_run", "free_height_over_bar_m", "post_in_crate", "inside_handle",
        "pass_a", "pass_b", "pass_c", "pass_c_apple", "pass_c_relaxed", "pass_d", "pass_e")


def short(e):
    return None if e is None else {k: e.get(k) for k in KEYS} | {"margin": e["margin"]}


def load(tag):
    pool, src = [], {}
    for suf in ("", "_refine", "_ext"):
        f = RAW / f"psearch_{tag}{suf}.json"
        if f.exists():
            d = json.loads(f.read_text()); src[suf or "coarse"] = {"n": d["n_candidates"], "counts": d["counts"]}
            pool += d["candidates"]
    for e in pool:
        e["margin"] = min(e["a_min"], e["d_min"])
    return pool, src


def pick(pool, relaxed):
    if relaxed:
        P = [e for e in pool if e["pass_a"] and e["pass_b"] and e["pass_c_relaxed"] and e["pass_d"] and e["pass_e"]]
        key = lambda e: (round(min(e["margin"], 0.06), 3), round(min(e["b_crate_min"], 0.010), 3), -min(e["c_apple_per_run"].values()), e["free_height_over_bar_m"])
    else:
        P = [e for e in pool if e["pass_a"] and e["pass_b"] and e["pass_c"] and e["pass_d"] and e["pass_e"]]
        key = lambda e: (round(min(e["margin"], 0.06), 3), e["pass_c_apple"], round(min(e["b_crate_min"], 0.010), 3), -max(e["c_min_per_run"].values()), e["free_height_over_bar_m"])
    return (max(P, key=key) if P else None), len(P)


def maximin(P):
    def sl(e):
        return {"a": e["a_min"] - 0.030, "d": e["d_min"] - 0.030, "b_crate": e["b_crate_min"], "b_obj": e["b_obj_min"],
                "client_arms": e["client_reset_arms"] - 0.020}
    if not P:
        return None
    e = max(P, key=lambda e: min(sl(e).values())); s = sl(e); w = min(s, key=s.get)
    return {"limiting": w, "worst_slack_m": s[w], "slacks_m": s, "candidate": short(e)}


def tradeoff(pool):
    def best(P, f, mx=True):
        P = list(P)
        return short((max if mx else min)(P, key=f)) if P else None
    return {
        "a_best_given_bcd": best((e for e in pool if e["pass_b"] and e["pass_c"] and e["pass_d"]), lambda e: e["a_min"]),
        "d_best_given_abc": best((e for e in pool if e["pass_a"] and e["pass_b"] and e["pass_c"]), lambda e: e["d_min"]),
        "margin_best_given_bc": best((e for e in pool if e["pass_b"] and e["pass_c"]), lambda e: e["margin"]),
        "c_miss_best_given_abd": best((e for e in pool if e["pass_a"] and e["pass_b"] and e["pass_d"]), lambda e: max(e["c_min_per_run"].values()), mx=False),
        "c_apple_miss_best_given_abd": best((e for e in pool if e["pass_a"] and e["pass_b"] and e["pass_d"]), lambda e: min(e["c_apple_per_run"].values()), mx=False),
        "margin_best_given_b_crelaxed": best((e for e in pool if e["pass_b"] and e["pass_c_relaxed"]), lambda e: e["margin"]),
        "maximin_given_c": maximin([e for e in pool if e["pass_c"]]),
        "maximin_given_crelaxed": maximin([e for e in pool if e["pass_c_relaxed"]]),
        "n_post_in_crate_given_acd": sum(1 for e in pool if e["post_in_crate"] and e["pass_a"] and e["pass_c"] and e["pass_d"]),
        "n_post_in_crate_given_a_crelaxed_d": sum(1 for e in pool if e["post_in_crate"] and e["pass_a"] and e["pass_c_relaxed"] and e["pass_d"]),
    }


def gjk_at(P, e):
    """independent GJK at the argmin row/pair of each minimum (exact re-evaluation of the chosen placement)."""
    c = P.c; out = {}
    ev = P.evaluate(e["x"], e["y"], e["yaw"], e["h"])
    c.set_hurdle(e["x"], e["y"], e["yaw"], e["h"], exact=True)
    items = [("wu_robot", P.WU, ev["wu_robot"], ev["wu_robot_arg"]), ("reset_crate", P.WU, ev["reset_crate"], ev["reset_crate_arg"])]
    for cond, R in c.runs.items():
        for k in ("pre_robot", "pre_apple", "car_grip", "car_apple"):
            items.append((f"{cond}_{k}", R, ev[f"{cond}_{k}"], ev[f"{cond}_{k}_arg"]))
    for name, Pc, v, arg in items:
        if arg is None:
            out[name] = {"client_m": v, "note": f">= cap {S.CAP}"}; continue
        i, hn, gn = arg
        d = c.d; d.qpos[:] = Pc["Q"][i]; mujoco.mj_kinematics(c.m, d)
        ids = [mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_GEOM, n) if not n.startswith("geom#") else int(n[5:]) for n in (hn, gn)]
        g = gjk.signed_dist(c.m, d, ids[0], ids[1]) if v > 0 else -gjk.pen_depth(gjk.make_support(c.m, d, ids[0]), gjk.make_support(c.m, d, ids[1]))
        out[name] = {"client_m": v, "row": int(i), "pair": [hn, gn], "gjk_m": float(g), "abs_diff_m": abs(float(g) - v) if v > 0 else None}
    c.set_hurdle(e["x"], e["y"], e["yaw"], e["h"], exact=True); out["reset_exact_activate"] = c.reset_check()
    c.restore()
    return out, ev


def write(tag, e, kind, P):
    ep, seed = tag.split("_")
    prof = f"t43p_ep{ep}_s{seed}" + ("_crelax" if kind == "relaxed" else ""); path = CFG / f"{prof}.json"
    desc = (f"T43 P ({kind}): hurdle near the crate, crate frame u={e['u']:+.3f} v={e['v']:+.3f} m, dyaw={e['dyaw_deg']:+.0f} deg, "
            f"bar_height {e['h']:.2f} m; reset/warmup robot clearance {e['a_min'] * 1000:.1f} mm, approach+grasp {e['d_min'] * 1000:.1f} mm")
    F.write_config(path, prof, e["x"], e["y"], e["yaw"], e["h"], desc)
    g, ev = gjk_at(P, e)
    dj = RAW / f"dryrun_{prof}.json"
    r = subprocess.run([PY, _ag3s_workspace_path('outputs/verify/T43/P/p/dryrun.py'), ep, str(path), prof, str(dj)], capture_output=True, text=True,
                       env={"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "osmesa", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
    dry = json.loads(dj.read_text()) if dj.exists() else {"stderr": r.stderr[-2000:]}
    return {"config": str(path), "profile": prof, "gjk_check": g, "reeval": {k: v for k, v in ev.items()}, "dryrun": dry}


def main(tag):
    pool, src = load(tag)
    res = {"tag": tag, "definitions": __doc__, "sources": src, "n_pool": len(pool)}
    best, n = pick(pool, False); bestr, nr = pick(pool, True)
    res["n_main"] = n; res["n_relaxed"] = nr; res["chosen"] = short(best); res["chosen_relaxed"] = short(bestr)
    res["tradeoff"] = tradeoff(pool)
    P = S.PCtx(tag)
    if best is not None:
        res["main"] = write(tag, best, "main", P)
    if bestr is not None and (best is None or (bestr["x"], bestr["y"], bestr["yaw"], bestr["h"]) != (best["x"], best["y"], best["yaw"], best["h"])):
        res["relaxed"] = write(tag, bestr, "relaxed", P)
    (RAW / f"select_p_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(tag, "main", n, short(best), "relaxed", nr, short(bestr), flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
