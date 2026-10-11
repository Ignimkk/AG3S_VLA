"""T40 D: one adjustment round (round 2) for the episodes whose E3b run failed with the round-1 hurdle. CPU.
usage: dselect.py TAG [TAG ...]  -> configs/t40f_ep<EP>_s<SEED>_r2.json, raw/dselect_<TAG>.json

Every round-1 E3b failure was an obstacle stop BEFORE lift-off (gripper/finger <-> bar during the approach, or the 1995 stop with
clearance 0.0 and no contact); so round 2 keeps (a)-(d) and (c) from the same F pool (fsearch_<TAG>[ _u10].json: bar centre on the
recorded T39 path, blocks both recorded transports) and maximises the pre-lift room, now also against the round-1 closed-loop
approaches:
  m_T39 = min(reset robot, reset object, T39 pre-lift minimum, t0 minimum)            (F numbers)
  m_C   = min over the round-1 C runs (E0 and E3b) of the hurdle <-> ROBOT_ALL distance over every recorded control row before
          that run's lift-off (or before its end when it never lifted), client distance code (cap 0.05)
  score = min(m_T39, m_C); choose the max score; ties -> lower bar_height. The round-1 placement is excluded.
The chosen placement is re-checked: GJK at the argmins (select_write.gjk_check) and the client dry run (dryrun.py)."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib, subprocess, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/F/p'))
import flib as F, select_write as SW  # noqa: E402

RAW = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/F/raw')); CFG = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/F/configs'))
G = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/G/runs')); AN = json.loads(pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/G/analysis_r1.json')).read_text())


def main(tag):
    ep, seed = tag.split("_")
    c = F.Ctx(tag)
    for cond in ("E0", "E3b"):
        rd = G / cond / "C" / f"{cond}_ep{ep}_s{seed}"
        rows = [json.loads(l) for l in (rd / "frames.jsonl").read_text().splitlines() if l.strip()]
        ctrl = [r for r in rows if r["kind"] == "control"]
        Q = np.array([r["qpos"] for r in ctrl]); t = np.array([r["t_step"] for r in ctrl])
        lt = AN["runs"][f"{cond}/C/{cond}_ep{ep}_s{seed}"]["hurdle"]["lift_t"]
        xp, xm = c.pose_cache(Q)
        c.runs[f"C_{cond}"] = {"t": t, "Q": Q, "xpos": xp, "xmat": xm, "lift_t": lt if lt is not None else int(t[-1]) + 1}
    c.restore()
    r1 = json.loads((RAW / f"select_{tag}.json").read_text())["chosen"]
    pool = []
    for suf in ("", "_u10"):
        f = RAW / f"fsearch_{tag}{suf}.json"
        if f.exists():
            pool += json.loads(f.read_text())["feasible"]
    ev = []
    for e in pool:
        if all(abs(e[k] - r1[k]) < 1e-9 for k in ("x", "y", "yaw", "h")):
            continue
        c.set_hurdle(e["x"], e["y"], e["yaw"], e["h"])
        mc = {}
        for cond in ("C_E0", "C_E3b"):
            R = c.runs[cond]; rows = np.where(R["t"] < R["lift_t"])[0]
            mc[cond] = float(c.series(cond, rows, {"robot": c.ROBOT_ALL}, cap=0.05)["robot"].min())
        e = dict(e); e["m_T39"] = SW.margin(e); e["m_C"] = mc; e["score"] = min(e["m_T39"], *mc.values())
        ev.append(e)
    c.restore()
    # round-1 placement under the same score
    c.set_hurdle(r1["x"], r1["y"], r1["yaw"], r1["h"])
    r1mc = {cond: float(c.series(cond, np.where(c.runs[cond]["t"] < c.runs[cond]["lift_t"])[0], {"robot": c.ROBOT_ALL}, cap=0.05)["robot"].min())
            for cond in ("C_E0", "C_E3b")}
    c.restore()
    res = {"tag": tag, "definitions": __doc__, "n_pool_excl_r1": len(ev), "round1": {k: r1[k] for k in ("x", "y", "yaw", "h")},
           "round1_m_T39": SW.margin(r1), "round1_m_C": r1mc, "round1_score": min(SW.margin(r1), *r1mc.values())}
    best = max(ev, key=lambda e: (round(e["score"], 5), -e["h"])) if ev else None
    res["chosen"] = best
    if best is not None:
        prof = f"t40f_ep{ep}_s{seed}_r2"; path = CFG / f"{prof}.json"
        F.write_config(path, prof, best["x"], best["y"], best["yaw"], best["h"],
                       f"T40 D round 2: max pre-lift room incl. round-1 closed-loop approaches; path {best['path_run']} s={best['s_m']:.2f} m, "
                       f"dyaw={best['dyaw_deg']:+.0f} deg, u={best['u_m']:+.2f} m")
        res["config"] = str(path); res["profile"] = prof
        c2 = F.Ctx(tag); res["gjk_check"] = SW.gjk_check(c2, best)
        dj = RAW / f"dryrun_{prof}.json"
        subprocess.run([SW.PY, _ag3s_workspace_path('outputs/verify/T40/F/p/dryrun.py'), ep, str(path), prof, str(dj)], capture_output=True, text=True,
                       env={"JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": "", "MUJOCO_GL": "osmesa", "OMP_NUM_THREADS": "1", "PATH": "/usr/bin:/bin"})
        res["dryrun"] = json.loads(dj.read_text()) if dj.exists() else None
    (RAW / f"dselect_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    b = best
    print(tag, "r1 score", round(1e3 * res["round1_score"], 1), {k: round(1e3 * v, 1) for k, v in r1mc.items()}, "->",
          None if b is None else {k: (round(b[k], 4) if isinstance(b[k], float) else b[k]) for k in ("x", "y", "yaw", "h", "m_T39", "score", "c_low")},
          None if b is None else {k: round(1e3 * v, 1) for k, v in b["m_C"].items()}, (res.get("dryrun") or {}).get("obstacles_line"), flush=True)


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
