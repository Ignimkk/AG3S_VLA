"""T40 F step 2: choose one placement per episode from raw/fsearch_<TAG>[ _u10].json, write the config, cross-check.
usage: select_write.py TAG [TAG ...]  -> configs/t40f_ep<EP>_s<SEED>_r1.json, raw/select_<TAG>.json

Margins m = min(reset robot clearance, reset object clearance, pre-lift minimum (robot+apple+fruits, both runs), t0 minimum).
Tiers (no candidate of any episode reached the lead's 20 mm, so the thresholds are 10 and 5 mm):
  T1: c_low (still blocks with the bar 20 mm lower) and m >= 0.010;  T2: c_low and m >= 0.005;  T3: c_low;  T4: feasible.
Within the best non-empty tier: largest free_height_over_bar (detour room, fsearch definition), ties -> larger m.
Pool = fsearch_<TAG>.json (u in {0, +-0.06}) plus fsearch_<TAG>_u10.json (u in {+-0.08, +-0.10}) when present.
Cross-check of the chosen placement: (1) GJK (gjk.signed_dist) at the argmin row/pair of each reported minimum vs the client
distance; (2) the client's own reset check: dryrun.py (snapshot pi05_infer.main() up to the policy call)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, sys
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import flib as F, gjk  # noqa: E402

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/raw')); CFG = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/P/configs'))
PY = _ag3s_workspace_path('legacy/.venv-openpi-live/bin/python')


def margin(e):
    return min(e["reset_robot"], e["reset_object"], e["prelift_overall_min"], e["t0_overall_min"])


def tier(e):
    m = margin(e)
    if e["c_low"] and m >= 0.010:
        return 1
    if e["c_low"] and m >= 0.005:
        return 2
    return 3 if e["c_low"] else 4


def gjk_check(c, e):
    c.set_hurdle(e["x"], e["y"], e["yaw"], e["h"], exact=True)
    out = {}
    for cond, R in c.runs.items():
        for ph, rows, sets in (("carry", np.where((R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"]))[0], {"A": c.A_GRIP, "B": c.B_APPLE}),
                               ("prelift", np.where((R["t"] >= 0) & (R["t"] < R["lift_t"]))[0], {"robot": c.ROBOT_ALL, "apple": c.B_APPLE, "fruits": c.FRUITS}),
                               ("t0", np.array([0]), {"robot": c.ROBOT_ALL, "apple": c.B_APPLE, "fruits": c.FRUITS, "crate": c.CRATE})):
            s, pr = c.series(cond, rows, sets, cap=0.30, want_pair=True)
            for k in sets:
                j = int(np.argmin(s[k])); v = float(s[k][j]); pair = pr[k][j]
                g_val = None
                if pair is not None:
                    d = c.d; d.qpos[:] = R["Q"][rows[j]]; mujoco.mj_kinematics(c.m, d)
                    ids = [mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_GEOM, n) if not n.startswith("geom#") else int(n[5:]) for n in pair]
                    g_val = gjk.signed_dist(c.m, d, ids[0], ids[1]) if v > 0 else float(gjk.gjk_dist(gjk.make_support(c.m, d, ids[0]), gjk.make_support(c.m, d, ids[1]), d.geom_xpos[ids[1]] - d.geom_xpos[ids[0]] + 1e-9))
                out[f"{cond}:{ph}:{k}"] = {"client_m": v, "t": int(R["t"][rows[j]]), "pair": pair,
                                           "gjk_m": g_val, "gjk_note": "separation (0 if overlapping)" if v <= 0 else "signed"}
    c.set_hurdle(e["x"], e["y"], e["yaw"], e["h"], exact=True)
    out["reset_exact_activate"] = c.reset_check()
    c.restore()
    return out


def main(tag):
    ep, seed = tag.split("_")
    pool, src = [], {}
    for suf in ("", "_u10"):
        f = RAW / f"fsearch_{tag}{suf}.json"
        if f.exists():
            d = json.loads(f.read_text()); src[suf or "base"] = {"counts": d["counts"], "n_feasible": d["n_feasible"], "u_values_m": d.get("u_values_m")}
            pool += d["feasible"]
    for e in pool:
        e["margin_m"] = margin(e); e["tier_sel"] = tier(e)
    res = {"tag": tag, "sources": src, "n_pool": len(pool), "n_by_tier": {t: sum(1 for e in pool if e["tier_sel"] == t) for t in (1, 2, 3, 4)}}
    best = None
    for t in (1, 2, 3, 4):
        P = [e for e in pool if e["tier_sel"] == t and e["free_height_over_bar_m"] is not None]
        if P:
            best = max(P, key=lambda e: (round(e["free_height_over_bar_m"], 4), e["margin_m"])); break
    res["chosen"] = best
    if best is None:
        res["dropped"] = True; res["reason"] = "no candidate satisfies (a)-(d)"
    else:
        prof = f"t40f_ep{ep}_s{seed}_r1"; path = CFG / f"{prof}.json"
        desc = (f"T40 F round 1: hurdle on the recorded transport path of {best['path_run']} at s={best['s_m']:.2f} m, "
                f"dyaw={best['dyaw_deg']:+.0f} deg, u={best['u_m']:+.2f} m; tier {best['tier_sel']}")
        F.write_config(path, prof, best["x"], best["y"], best["yaw"], best["h"], desc)
        res["config"] = str(path); res["profile"] = prof
        c = F.Ctx(tag)
        res["gjk_check"] = gjk_check(c, best)
        dj = RAW / f"dryrun_{prof}.json"
        r = subprocess.run([PY, _ag3s_workspace_path('outputs/verify/T43/P/p/dryrun.py'), ep, str(path), prof, str(dj)], capture_output=True, text=True,
                           env={"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "osmesa", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
        res["dryrun"] = json.loads(dj.read_text()) if dj.exists() else {"stderr": r.stderr[-2000:]}
    (RAW / f"select_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    b = best
    print(tag, res["n_by_tier"], None if b is None else {k: (round(b[k], 4) if isinstance(b[k], float) else b[k]) for k in
          ("path_run", "s_m", "dyaw_deg", "u_m", "x", "y", "yaw", "h", "margin_m", "free_height_over_bar_m", "tier_sel")},
          (res.get("dryrun") or {}).get("obstacles_line"), flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
