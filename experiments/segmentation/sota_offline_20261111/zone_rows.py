#!/usr/bin/env python3
"""(2026-11-12 dev-57, offline CPU, GT = scoring only) zone_rows.py -- WHAT do the zone-released hand / finger rows penetrate?
LATCHED / CLOSING calls: authorized spheres within 100 mm (horizontal) of the manipulated centroid whose recorded clearance
(min over the 5 executed steps) is < 0, labelled with the real body of the nearest non-support cloud point (request pose).
(derived from target_truth.py) -- original docstring:  WHICH real body is the recognised target?  For every server chunk of
every episode in a run: the chunk's `target_points` (the recognised target / manipulated object geometry) are labelled with
the body whose geometry (mesh vertices; primitives sampled on their box) is nearest in the sim state of that call, and
compared with the task's object of interest (BDDL `obj_of_interest[0]`).
usage: .venv-libero/bin/python target_truth.py --run-dir RUN [--episodes 5,6] --out FILE.json"""
import argparse, ast, collections, glob, json, os, pathlib, sys

PROJECT_ROOT = pathlib.Path("/mnt/dev/AG3S_VLA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.append(str(PROJECT_ROOT))
    import numpy as np
    import mujoco
    from scipy.spatial import cKDTree
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "runners"))
    from libero_safe_client import activate_safelibero
    activate_safelibero()
    from benchmark.adapters.libero_panda import env_history
    run = pathlib.Path(a.run_dir).resolve()
    want_eps = {int(x) for x in a.episodes.split(",") if x}
    rows = [r for r in env_history.read_rows(run) if not want_eps or int(r["episode_idx"]) in want_eps]
    wanted = {env_history.stem_of(r) for r in rows}
    env_history.run_history(run)
    eps = [json.loads(l) for l in open(run / "episodes.jsonl") if l.strip()]
    per, cur, last = [], [], None
    for c in sorted(glob.glob(str(run / "server_constraints/run_*/chunk_*.npz"))):
        ts = int(np.load(c)["t_step"])
        if last is not None and ts <= last:
            per.append(cur)
            cur = []
        cur.append((ts, c))
        last = ts
    per.append(cur)
    out = {}
    for row, env, _ in env_history.episode_envs(run, wanted):
        stem = env_history.stem_of(row)
        ei = int(row["episode_idx"])
        k = [i for i, e in enumerate(eps) if str(e.get("npz", "")).endswith("/" + stem + ".npz")][0]   # multi-task runs repeat episode_idx
        sim = env.sim
        m, d = sim.model._model, sim.data._data
        goal = None
        for obj in (env, getattr(env, "env", None)):
            for name in ("obj_of_interest",):
                v = getattr(obj, name, None) if obj is not None else None
                if v:
                    goal = list(v)
        g = np.load(str(run / "episodes" / (stem + "_geom.npz")))
        bname = lambda gi: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
        local = {}
        for gi in range(m.ngeom):
            b = bname(gi)
            if not b or b in ("world",) or "table" in b or "floor" in b or "wall" in b:
                continue
            if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = int(m.geom_dataid[gi])
                v = np.asarray(m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]], float)
                local[gi] = v[:: max(1, len(v) // 800)]
            else:
                s = np.asarray(m.geom_size[gi], float)
                t = int(m.geom_type[gi])
                half = (s if t == mujoco.mjtGeom.mjGEOM_BOX else np.array([s[0], s[0], s[1] if t in (
                    mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE) else s[0]]))
                u = np.linspace(-1, 1, 7)
                grid = np.array([[x, y, z] for x in u for y in u for z in u if max(abs(x), abs(y), abs(z)) == 1.0])
                local[gi] = grid * half[None, :]
        calls = []
        tgt = (goal or [None])[0]
        for ts, cpath in per[k]:
            z = np.load(cpath)
            sm = json.loads(str(z["summary_json"]))
            state = (sm.get("grasp") or {}).get("state")
            man = sm.get("manipulated") or {}
            if state not in ("latched", "closing") or man.get("centroid") is None or "sphere_centres" not in z.files:
                continue
            names = sm.get("sphere_link_names") or []
            cen = np.asarray(z["sphere_centres"], float)
            rad = np.asarray(z["sphere_radii"], float)
            cl = np.asarray(z["clearance"], float)
            n = min(len(names), len(cen), cl.shape[1])
            auth = np.array([names[j] in ("panda_hand", "panda_leftfinger", "panda_rightfinger") for j in range(n)])
            horiz = np.linalg.norm(cen[:n, :2] - np.asarray(man["centroid"], float)[None, :2], axis=1)
            cmin = np.nanmin(cl[:5, :n], axis=0)
            sel = np.flatnonzero(auth & (horiz <= 0.10) & (cmin < float(os.environ.get('ZONE_THR_M', '0.0'))))   # ZONE_THR_M: clearance threshold (m)
            if not len(sel):
                continue
            f = 20 + 5 * ts
            i = int(np.argmin(np.abs(np.asarray(g["obs_index"]) - f)))
            sim.set_state_from_flattened(np.asarray(g["sim_state_image"][i]).copy())
            sim.forward()
            T = np.asarray(g["T_world_base"][i])
            cloud = z["filtered_cloud"].astype(float)[~z["support_mask"].astype(bool)]
            if not len(cloud):
                continue
            dn, jn = cKDTree(cloud).query(cen[sel])
            pw = cloud[jn] @ T[:3, :3].T + T[:3, 3]
            best = np.full(len(pw), np.inf)
            owner = np.full(len(pw), -1)
            for gi, v in local.items():
                R = np.asarray(d.geom_xmat[gi], float).reshape(3, 3)
                vw = v @ R.T + np.asarray(d.geom_xpos[gi], float)
                dist = cKDTree(vw).query(pw)[0]
                better = dist < best
                best[better], owner[better] = dist[better], gi
            for q, j in enumerate(sel):
                if dn[q] > rad[j] + 0.03:
                    oc = "far"
                else:
                    b = bname(int(owner[q])).replace("_main", "")
                    oc = ("target" if tgt and b.startswith(tgt) else "obstacle" if "obstacle" in b
                          else "robot" if ("gripper" in b or "robot" in b) else "other:" + b)
                calls.append({"seq": ts, "state": state, "link": "palm" if names[j] == "panda_hand" else "finger",
                              "band": "0-60" if horiz[j] <= 0.06 else "60-100", "clear_mm": round(1000 * float(cmin[j]), 1), "owner": oc})
        e = eps[k]
        sl = e.get("safelibero") or {}
        out[stem] = {"task_id": e.get("task_id"), "init": e.get("init_state_index"), "goal": goal,
                     "success": bool(e.get("official_success")), "collided": bool(sl.get("collided")), "rows": calls}
        cnt = collections.Counter((r["link"], r["band"], r["owner"].split(":")[0]) for r in calls)
        print(stem, "S" if out[stem]["success"] else "F", "+C" if out[stem]["collided"] else "", dict(cnt), flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
