"""T40 F step 1: exhaustive hurdle placement search for one episode (worker slice). CPU only.
usage: search.py TAG WORKER NWORKERS  -> raw/search/<TAG>_w<WORKER>.json

Grid: x in [0.38, 0.85] step 0.01, y in [-0.15, 0.35] step 0.01, yaw in [0, 180) step 7.5 deg (the hurdle is symmetric
under yaw + 180), bar_height in {0.20, 0.22, ..., 0.32}. Worker k takes the grid points with index % N == k.
xy prefilter: some point of the hurdle footprint (bar segment between post centres, +-0.04 m) is within 0.06 m (xy) of a
recorded apple-centre xy in [lift_t, place_t] of BOTH runs (cheap; a hurdle farther away cannot touch the carried apple
or the gripper in xy).
Stages (first failure rejects; counts reported):
  a  reset check, client code: robot_clearance() > 0 and object_clearance() > 0 (stop distance 0, pi05_infer.py:1165-1173)
  c  carry block: min over rows lift_t <= t < place_t of min(A_GRIP, B_APPLE) distance <= 0 in E0 AND E3b (cap 0.02)
  d  pre-lift: min over rows t < lift_t (control rows 0 .. lift_t-1) of distance to ROBOT_ALL, B_APPLE, FRUITS > 0 (cap 0.05)
  b  t0 (recorded planning row 0 qpos, after warmup, both runs): ROBOT_ALL > 0, apple+fruits > 0, crate > 0 (cap 0.05)
Survivors are written with: reset (robot, object), carry minima per run (A, B, cap 0.02), first carry row with contact per run,
pre-lift min per run, t0 minima per run."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, pathlib, sys, time
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/F/p'))
import flib as F

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T40/F/raw/search')); OUT.mkdir(parents=True, exist_ok=True)


def main(tag, k, n):
    c = F.Ctx(tag)
    paths = {}
    for cond, R in c.runs.items():
        mk = (R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"])
        import mujoco
        b = mujoco.mj_name2id(c.m, mujoco.mjtObj.mjOBJ_BODY, "apple")
        paths[cond] = R["Q"][mk][:, 38:40]  # apple free joint x, y
    xs = np.round(np.arange(0.38, 0.85001, 0.01), 4); ys = np.round(np.arange(-0.15, 0.35001, 0.01), 4)
    yaws = np.radians(np.arange(0, 180, 7.5)); hs = np.round(np.arange(0.20, 0.32001, 0.02), 4)
    cnt = {"grid": 0, "slice": 0, "xy_prefilter": 0, "a": 0, "c": 0, "d": 0, "b": 0}
    keep = []
    t0 = time.time(); gi = -1
    sets_ab = {"A": c.A_GRIP, "B": c.B_APPLE}
    sets_pre = {"robot": c.ROBOT_ALL, "apple": c.B_APPLE, "fruits": c.FRUITS}
    sets_t0 = {"robot": c.ROBOT_ALL, "apple": c.B_APPLE, "fruits": c.FRUITS, "crate": c.CRATE}
    for x in xs:
        for y in ys:
            for yaw in yaws:
                a_, b_ = F.hurdle_xy_segments(x, y, yaw)
                okxy = all(F.seg_point_dist(a_, b_, P)[0].min() <= 0.06 for P in paths.values())
                for h in hs:
                    gi += 1; cnt["grid"] += 1
                    if gi % n != k:
                        continue
                    cnt["slice"] += 1
                    if not okxy:
                        continue
                    cnt["xy_prefilter"] += 1
                    c.set_hurdle(x, y, yaw, h)
                    rr, ro = c.reset_check()
                    if not (rr > 0 and ro > 0):
                        continue
                    cnt["a"] += 1
                    e = {"x": float(x), "y": float(y), "yaw": float(yaw), "h": float(h), "reset_robot": rr, "reset_object": ro, "runs": {}}
                    ok = True
                    for cond, R in c.runs.items():
                        rows = np.where((R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"]))[0]
                        s = c.series(cond, rows, sets_ab, cap=0.02)
                        ab = np.minimum(s["A"], s["B"])
                        if ab.min() > 0:
                            ok = False; break
                        j = int(np.argmax(ab <= 0))
                        e["runs"][cond] = {"carry_min_A": float(s["A"].min()), "carry_min_B": float(s["B"].min()),
                                           "first_contact_t": int(R["t"][rows[j]]), "n_rows_le0": int((ab <= 0).sum())}
                    if not ok:
                        continue
                    cnt["c"] += 1
                    for cond, R in c.runs.items():
                        rows = np.where((R["t"] >= 0) & (R["t"] < R["lift_t"]))[0]
                        s = c.series(cond, rows, sets_pre, cap=0.05)
                        mins = {kk: float(v.min()) for kk, v in s.items()}
                        e["runs"][cond]["prelift_min"] = mins
                        if min(mins.values()) <= 0:
                            ok = False; break
                    if not ok:
                        continue
                    cnt["d"] += 1
                    for cond, R in c.runs.items():
                        s = c.series(cond, [0], sets_t0, cap=0.05)
                        mins = {kk: float(v.min()) for kk, v in s.items()}
                        e["runs"][cond]["t0_min"] = mins
                        if min(mins.values()) <= 0:
                            ok = False
                    if not ok:
                        continue
                    cnt["b"] += 1
                    keep.append(e)
        print(tag, k, "x", x, cnt, f"{time.time() - t0:.0f}s", flush=True)
    c.restore()
    (OUT / f"{tag}_w{k}.json").write_text(json.dumps({"tag": tag, "worker": k, "n_workers": n, "counts": cnt, "kept": keep,
                                                       "elapsed_s": time.time() - t0, "definitions": __doc__}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
