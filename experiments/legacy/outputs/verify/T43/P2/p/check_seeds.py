"""T43 P2: the chosen overbar against EVERY recorded E0 / E3b run of the episode (all policy seeds in outputs/verify/T39/runs and
outputs/verify/T42/1/runs/*/noobs with a .done marker), not only the search seed.  Per run: T42-1 analyze rule (grasp_t, place_t,
success), lift_t (T40 A rule), and with the config's overbar (manager._place = activate geometry, cross-checked in select_p2):
  car_min  = min(GRIP, APPLE) over control rows lift_t <= t < place_t (<= 0: the recorded path hits the bar), car_apple = APPLE only;
  pre_min  = min(ROBOT, APPLE) over control rows t < lift_t (criterion d: >= 0.030);  t0_robot = recorded planning row 0.
Distances: branch-and-bound p2search.PCtx.bb_multi (certified pair_distance), cap 0.06 m.
usage: check_seeds.py TAG [TAG ...]  -> raw/check_seeds_<TAG>.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import glob, json, os, pathlib, sys
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P2/p')); sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/p'))
import p2lib as F  # noqa: E402
import p2search as S  # noqa: E402
import analyze_t42 as AN  # noqa: E402


def runs_of(ep):
    out = {}
    for cond in ("E0", "E3b"):
        for root in (f"${AG3S_PROJECT_ROOT}/outputs/verify/T39/runs/{cond}", f"${AG3S_PROJECT_ROOT}/outputs/verify/T42/1/runs/{cond}/noobs"):
            for d in sorted(glob.glob(f"{root}/{cond}_ep{ep}_s*")):
                if os.path.isdir(d) and os.path.exists(d + ".done"):
                    out.setdefault(pathlib.Path(d).name, d)
    return out


def main(tag):
    sel = json.loads((F.RAW / f"select_p2_{tag}.json").read_text()); e = sel["chosen"]
    PC = S.PCtx(tag); c = PC.c
    c.set_overbar(e["x"], e["y"], e["yaw"], e["h"], e["L"], e["r"])
    res = {"tag": tag, "definitions": __doc__, "config": sel.get("config"), "runs": {}}
    # uncapped (cap 0.30 m) a' and d of the search seed: reset + warmup rows and t0 (a'), pre-lift rows (d)
    W = PC.WU; ua = PC.bb_min(W, np.arange(len(W["Q"])), c.ROBOT_ALL, cap=0.30)
    uargs = {"warmup": ua}
    for cond, R in c.runs.items():
        uargs[f"{cond}_t0"] = PC.bb_min(R, PC.rows[cond]["T0"], c.ROBOT_ALL, cap=0.30)
        uargs[f"{cond}_pre_robot"] = PC.bb_min(R, PC.rows[cond]["PRE"], c.ROBOT_ALL, cap=0.30)
        uargs[f"{cond}_pre_apple"] = PC.bb_min(R, PC.rows[cond]["PRE"], c.B_APPLE, cap=0.30)
    res["uncapped_search_seed"] = {"a_min": min(uargs["warmup"][0], *[uargs[f"{k}_t0"][0] for k in c.runs]),
                                   "d_min": min(min(uargs[f"{k}_pre_robot"][0], uargs[f"{k}_pre_apple"][0]) for k in c.runs),
                                   "args": uargs}
    print(tag, "uncapped", {k: v for k, v in res["uncapped_search_seed"].items() if k != "args"}, flush=True)
    for name, rd in runs_of(c.ep).items():
        cond = name.split("_")[0]
        a = AN.analyze(rd)
        rows = [json.loads(l) for l in (pathlib.Path(rd) / "frames.jsonl").read_text().splitlines() if l.strip()]
        ctrl = [r for r in rows if r["kind"] == "control"]; plan = [r for r in rows if r["kind"] == "planning"]
        Q = np.vstack([np.array(plan[0]["qpos"])[None], np.array([r["qpos"] for r in ctrl])]); t = np.r_[-1, [r["t_step"] for r in ctrl]]
        r = {"run_dir": rd, "success": a["success"], "grasp_t": a["grasp_t"], "place_t": a["place_t"]}
        xp, xm = c.pose_cache(Q); c.set_overbar(e["x"], e["y"], e["yaw"], e["h"], e["L"], e["r"]); P = {"xpos": xp, "xmat": xm}
        r["t0_robot"] = PC.bb_min(P, np.array([0]), c.ROBOT_ALL)[0]
        if a["grasp_t"] is None:
            r["lift_t"] = None
            pre = np.where(t >= 0)[0]
            r["pre_min_all_rows"] = min(PC.bb_min(P, pre, c.ROBOT_ALL)[0], PC.bb_min(P, pre, c.B_APPLE)[0])
        else:
            z = Q[:, 40]; ok = np.where((t >= 0) & (t <= a["grasp_t"]) & (z < z[0] + 0.002))[0]; lift = int(t[ok[-1] + 1]); r["lift_t"] = lift
            pre = np.where((t >= 0) & (t < lift))[0]
            r["pre_min"] = min(PC.bb_min(P, pre, c.ROBOT_ALL)[0], PC.bb_min(P, pre, c.B_APPLE)[0])
            end = a["place_t"] if a["place_t"] is not None else int(t[-1]) + 1
            car = np.where((t >= lift) & (t < end))[0]
            g = PC.bb_multi(P, car, {"grip": c.A_GRIP, "apple": c.B_APPLE})
            r["car_min"] = min(g["grip"][0], g["apple"][0]); r["car_apple"] = g["apple"][0]
            r["car_window"] = [lift, end]
        res["runs"][name] = r
        print(tag, name, {k: (round(v * 1000, 1) if isinstance(v, float) else v) for k, v in r.items() if k != "run_dir"}, flush=True)
    c.restore()
    (F.RAW / f"check_seeds_{tag}.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    for t in sys.argv[1:]:
        main(t)
