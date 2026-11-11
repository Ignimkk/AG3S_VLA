#!/usr/bin/env python3
"""(2026-11-11 dev-57, offline CPU, GT = scoring only) WHICH real body is the recognised target?  For every server chunk of
every episode in a run: the chunk's `target_points` (the recognised target / manipulated object geometry) are labelled with
the body whose geometry (mesh vertices; primitives sampled on their box) is nearest in the sim state of that call, and
compared with the task's object of interest (BDDL `obj_of_interest[0]`).
usage: .venv-libero/bin/python target_truth.py --run-dir RUN [--episodes 5,6] --out FILE.json"""
import argparse, ast, collections, glob, json, pathlib, sys

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
        for ts, cpath in per[k]:
            z = np.load(cpath)
            sm = json.loads(str(z["summary_json"]))
            rec = {"seq": ts, "grasp": (sm.get("grasp") or {}).get("state"), "has_target": bool(sm.get("has_target")),
                   "grounding": sm.get("grounding_status")}
            tp = np.asarray(z["target_points"], float).reshape(-1, 3) if "target_points" in z.files else np.zeros((0, 3))
            if len(tp):
                f = 20 + 5 * ts
                i = int(np.argmin(np.abs(np.asarray(g["obs_index"]) - f)))
                sim.set_state_from_flattened(np.asarray(g["sim_state_image"][i]).copy())
                sim.forward()
                T = np.asarray(g["T_world_base"][i])
                nw = tp @ T[:3, :3].T + T[:3, 3]
                best = np.full(len(nw), np.inf)
                owner = np.full(len(nw), -1)
                for gi, v in local.items():
                    R = np.asarray(d.geom_xmat[gi], float).reshape(3, 3)
                    vw = v @ R.T + np.asarray(d.geom_xpos[gi], float)
                    dist = cKDTree(vw).query(nw)[0]
                    better = dist < best
                    best[better], owner[better] = dist[better], gi
                cnt = collections.Counter(bname(int(o)).replace("_main", "") for o in owner)
                rec.update(n=int(len(tp)), owners=cnt.most_common(3), median_dist_mm=round(1000 * float(np.median(best)), 1),
                           centroid_m=[round(float(x), 3) for x in tp.mean(0)])
            calls.append(rec)
        e = eps[k]
        sl = e.get("safelibero") or {}
        out[stem] = {"task_id": e.get("task_id"), "init": e.get("init_state_index"), "goal": goal,
                     "success": bool(e.get("official_success")), "collided": bool(sl.get("collided")), "calls": calls}
        tgt = (goal or [None])[0]
        pre = [c for c in calls if c.get("grasp") != "held" and c.get("owners")]
        okc = sum(1 for c in pre if tgt and c["owners"][0][0].startswith(tgt))
        print(stem, "goal", goal, "pre-grasp calls with target", len(pre), "top owner == goal", okc,
              collections.Counter(c["owners"][0][0] for c in pre).most_common(3), flush=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
