"""T40 F: why does (d) reject? For the (a)+(b)-passing analytic candidates of one episode: per run, the first pre-lift row
(scanning from lift-off backwards, as fsearch) with contact, its t and the hurdle/robot geom pair; counts by run and body.
usage: diag_d.py TAG [STRIDE]  -> raw/diag_d_<TAG>.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, math, sys, collections
import numpy as np, mujoco
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T40/F/p'))
import flib as F, fsearch as FS
tag = sys.argv[1]; stride = int(sys.argv[2]) if len(sys.argv) > 2 else 5
c = F.Ctx(tag); m = c.m
bc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "crate"); c.restore(); crate = c.d.xpos[bc][:2].copy()
fallback = crate - c.runs["E0"]["Q"][0][38:40]
body = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g]))
res = []; cnt = collections.Counter(); k = 0
for cond, R in c.runs.items():
    mk = (R["t"] >= R["lift_t"]) & (R["t"] < R["place_t"])
    S, Pts = FS.resample(R["Q"][mk][:, 38:41]); yawn = FS.normal_dirs(S, Pts, fallback)
    for i in range(0, len(S), 2):
        for dy in FS.DYAW:
            yaw = (yawn[i] + dy) % math.pi
            for h in (0.20, 0.26, 0.32):
                k += 1
                if k % stride:
                    continue
                x, y = Pts[i, :2]
                if not (0.38 <= x <= 0.85 and -0.35 <= y <= 0.35):
                    continue
                c.set_hurdle(x, y, yaw, h); rr, ro = c.reset_check()
                if not (rr > 0 and ro > 0):
                    continue
                e = {"path_run": cond, "s": float(S[i]), "x": float(x), "y": float(y), "yaw_deg": math.degrees(yaw), "h": h, "first": {}}
                for c2, R2 in c.runs.items():
                    rows = np.where((R2["t"] >= 0) & (R2["t"] < R2["lift_t"]))[0][::-1]
                    s, pr = c.series(c2, rows, {"x": np.r_[c.ROBOT_ALL, c.B_APPLE, c.FRUITS]}, cap=0.02, want_pair=True)
                    hit = np.where(s["x"] <= 0)[0]
                    if len(hit):
                        j = hit[0]; t = int(R2["t"][rows[j]])
                        gname = pr["x"][j][1]; g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gname) if not gname.startswith("geom#") else int(gname[5:])
                        e["first"][c2] = {"t_latest_contact_before_lift": t, "lift_t": R2["lift_t"], "pair": pr["x"][j], "body": body(g),
                                          "n_rows_le0": int(len(hit)), "t_all": sorted(int(R2["t"][rows[q]]) for q in hit)[:3] + ["..."]}
                        cnt[(c2, body(g))] += 1
                res.append(e)
out = {"tag": tag, "n": len(res), "counts_run_body": {f"{a}|{b}": v for (a, b), v in cnt.most_common()}, "cands": res}
json.dump(out, open(f"${AG3S_PROJECT_ROOT}/outputs/verify/T40/F/raw/diag_d_{tag}.json", "w"), indent=1, default=str)
print(tag, len(res), cnt.most_common(12))
