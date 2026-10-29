"""T30 #2 occupancy near the GT apple, table excluded (supersedes the occupancy numbers inside f3_replay.json, which include table voxels
under the apple). For every E3 chunk in f3_replay.json:
  region  = voxel centres within 34 + 15 mm of the GT apple centre AND world z > apple z - 34 mm + 5 mm (above the table top + 1 voxel;
            the apple rests on the table in pre-grasp chunks)
  occupied = recorded layer value d <= 0
  (a) tf_recorded           : recorded target-free layer (esdf_free_distance; built with the RECORDED (pre-T30) ball) occupied in region
  (b) fine_out_old_ball     : recorded fine layer occupied in region, outside the replayed OLD-rule ball (compare with (a))
  (c) fine_out_new_ball     : recorded fine layer occupied in region, outside the replayed NEW-rule (T30) ball
  (d) fine_in_region        : recorded fine layer occupied in region
World z of a base-frame voxel: base pose from the paired planning row qpos (T28 P1-2 helper).
usage: f3_occupancy.py -> off2/f3_occupancy.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T28/P1-2'))
import p12_records as P
import mujoco
R = json.load(open(_ag3s_workspace_path('outputs/verify/T30/off2/f3_replay.json')))
T28 = pathlib.Path(_ag3s_workspace_path('outputs/verify/T28'))
m, d, probe, objg, _ = P.build()
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
RAD = 0.034 + 0.015


def occ(grid, origin, vox, a_b, Rw, tw, zcut):
    g = np.asarray(grid, np.float32); o = np.asarray(origin, float)
    lo = np.clip(np.floor((a_b - RAD - o) / vox).astype(int), 0, g.shape); hi = np.clip(np.ceil((a_b + RAD - o) / vox).astype(int) + 1, 0, g.shape)
    if np.any(hi <= lo):
        return np.zeros((0, 3))
    idx = np.argwhere(np.isfinite(g[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]) & (g[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] <= 0)) + lo
    c = o + idx * vox
    c = c[np.linalg.norm(c - a_b, axis=1) <= RAD]
    zw = (c @ Rw.T + tw)[:, 2]
    return c[zw > zcut]


out = {"definitions": __doc__, "runs": {}}
for label, rows in R["runs"].items():
    if not label.startswith("E3"):
        continue
    cond = label[:3]
    fr = [json.loads(l) for l in (T28 / f"{cond}_rec/{label}/frames.jsonl").read_text().splitlines() if l.strip()]
    plan = {x["seq"]: x for x in fr if x["kind"] == "planning"}
    L = []
    for e in rows:
        z = np.load(e["file"], allow_pickle=True)
        pr = plan[e["seq"]]
        P.set_q(m, d, pr["qpos"])
        Rw = d.xmat[bid].reshape(3, 3).copy(); tw = d.xpos[bid].copy()
        aw = np.asarray(pr["object_poses"]["apple"]["pos"], float)
        a_b = Rw.T @ (aw - tw)
        zcut = aw[2] - 0.034 + 0.005
        r = {"t": e["t"], "phase": e["phase"], "frozen": e["frozen"]}
        fine = occ(z["esdf_distance"], z["esdf_origin"], float(z["esdf_voxel_size"]), a_b, Rw, tw, zcut)
        r["fine_in_region"] = int(len(fine))
        for rule in ("old", "new"):
            b = e[rule]["ball"]
            r[f"fine_out_{rule}_ball"] = int(len(fine)) if not b else int((np.linalg.norm(fine - np.asarray(b["centre_b"]), axis=1) > b["radius_mm"] / 1000).sum())
            r[f"{rule}_whole_inside"] = bool(b and b["whole_inside"])
        if "esdf_free_distance" in z.files:
            r["tf_recorded"] = int(len(occ(z["esdf_free_distance"], z["esdf_free_origin"], float(z["esdf_free_voxel_size"]), a_b, Rw, tw, zcut)))
        L.append(r)
    out["runs"][label] = L
    print(label, len(L), flush=True)


def st(v):
    v = [x for x in v if x is not None]
    return None if not v else {"n": len(v), "n_gt0": sum(1 for x in v if x > 0), "median": float(np.median(v)), "max": int(max(v)), "sum": int(sum(v))}


allr = [r for L in out["runs"].values() for r in L]
summ = {}
for gname, sel in (("pre_grasp_nonfrozen", lambda r: r["phase"] == "pre_grasp" and not r["frozen"]), ("frozen", lambda r: r["frozen"])):
    G = [r for r in allr if sel(r)]
    s = {"n": len(G), "fine_in_region": st([r["fine_in_region"] for r in G]), "tf_recorded": st([r.get("tf_recorded") for r in G]),
         "fine_out_old_ball": st([r["fine_out_old_ball"] for r in G]),
         "tf_recorded_vs_fine_out_old_ball_abs_diff": st([abs(r["tf_recorded"] - r["fine_out_old_ball"]) for r in G if "tf_recorded" in r])}
    for rule in ("old", "new"):
        U = [r for r in G if not r[f"{rule}_whole_inside"]]
        s[f"{rule}_rule_uncovered"] = {"n_chunks": len(U), "fine_out_ball": st([r[f"fine_out_{rule}_ball"] for r in U]),
                                       "n_chunks_fine_out_ball_gt0": sum(1 for r in U if r[f"fine_out_{rule}_ball"] > 0)}
        if rule == "old":
            s["old_rule_uncovered"]["tf_recorded"] = st([r.get("tf_recorded") for r in U])
    summ[gname] = s
out["summary"] = summ
json.dump(out, open(_ag3s_workspace_path('outputs/verify/T30/off2/f3_occupancy.json'), "w"), indent=1)
print(json.dumps(summ, indent=1))
