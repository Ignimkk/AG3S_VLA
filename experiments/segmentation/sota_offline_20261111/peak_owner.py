#!/usr/bin/env python3
"""(2026-11-11 dev-57, offline CPU, GT = scoring only) WHICH real body do the cloud points next to the worst obstacle-object
row belong to?  row_owner.py ranks bodies around the row's (future) sphere centre -- while holding, the held object is always
nearest to a hand row, so it cannot tell "the held object's own image" from a real obstacle. Here: the server's filtered,
non-support cloud points within `--reach` of the worst row point (the surface that row is measured against), each labelled
with the body whose geometry (mesh vertices; primitives sampled on their box) is nearest in the sim state of that call.
(peak_owner.py = body under the attention peak per call; from surface_owner_task.py = surface_owner.py + --task / --plain-libero for the LIBERO guard runs, 2026-11-11 13:3x)
usage: .venv-libero/bin/python surface_owner_task.py --run-dir RUN --episode E [--only-held] --out FILE.json"""
import argparse, ast, collections, glob, json, os, pathlib, sys

PROJECT_ROOT = pathlib.Path("/mnt/dev/AG3S_VLA")


def lit(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return {}
    return x or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reach", type=float, default=0.06)
    ap.add_argument("--only-held", action="store_true")
    ap.add_argument("--task", type=int, default=None, help="task id (runs with several tasks, e.g. the LIBERO guard)")
    ap.add_argument("--plain-libero", action="store_true", help="LIBERO (not SafeLIBERO) run")
    a = ap.parse_args()
    sys.path.append(str(PROJECT_ROOT))
    import numpy as np
    import mujoco
    from scipy.spatial import cKDTree
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "runners"))
    if not a.plain_libero:
        from libero_safe_client import activate_safelibero
        activate_safelibero()
    from benchmark.adapters.libero_panda import env_history
    run = pathlib.Path(a.run_dir).resolve()
    rows = [r for r in env_history.read_rows(run) if int(r["episode_idx"]) == a.episode
            and (a.task is None or int(r["task_id"]) == a.task)]
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
    k = [i for i, e in enumerate(eps) if int(e["episode_idx"]) == a.episode
         and (a.task is None or int(e["task_id"]) == a.task)][0]
    out = []
    for row, env, _ in env_history.episode_envs(run, wanted):
        stem = env_history.stem_of(row)
        sim = env.sim
        m, d = sim.model._model, sim.data._data
        g = np.load(str(run / "episodes" / (stem + "_geom.npz")))
        bname = lambda gi: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
        local = {}                                           # geom id -> sample points in the geom frame
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
        for ts, cpath in per[k]:
            z = np.load(cpath)
            sm = json.loads(str(z["summary_json"]))
            if a.only_held and (sm.get("grasp") or {}).get("state") != "held":
                continue
            ac = z["attention_cloud"].astype(float)
            raw = z["attention_raw"].astype(float).reshape(-1) if "attention_raw" in z.files else z["attention_values"].astype(float).reshape(-1)
            sup = z["support_mask"].astype(bool).reshape(-1)
            if len(raw) != len(ac) or len(sup) != len(ac):
                out.append({"seq": ts, "why": f"shape {len(ac)} {len(raw)} {len(sup)}"}); continue
            ok = np.isfinite(raw) & ~sup
            if not ok.any():
                continue
            j = np.flatnonzero(ok)[int(np.argmax(raw[ok]))]
            f = 20 + 5 * ts
            i = int(np.argmin(np.abs(np.asarray(g["obs_index"]) - f)))
            sim.set_state_from_flattened(np.asarray(g["sim_state_image"][i]).copy())
            sim.forward()
            T = np.asarray(g["T_world_base"][i])
            near = ac[ok][np.linalg.norm(ac[ok] - ac[j], axis=1) <= 0.02]
            nw = near @ T[:3, :3].T + T[:3, 3]
            best = np.full(len(nw), np.inf)
            owner = np.full(len(nw), -1)
            for gi, v in local.items():
                R = np.asarray(d.geom_xmat[gi], float).reshape(3, 3)
                vw = v @ R.T + np.asarray(d.geom_xpos[gi], float)
                dist = cKDTree(vw).query(nw)[0]
                better = dist < best
                best[better], owner[better] = dist[better], gi
            cnt = collections.Counter(bname(int(o)).replace("_main", "") for o in owner)
            med = float(np.median(raw[~sup])) if (~sup).any() else 0.0
            tp = z["target_points"] if "target_points" in z.files and len(z["target_points"]) else None
            out.append({"seq": ts, "state": (sm.get("grasp") or {}).get("state"), "peak_owner": cnt.most_common(1)[0][0],
                        "peak_ratio": round(float(raw[j] / med), 1) if med > 0 else None,
                        "peak_to_target_mm": None if tp is None else round(1000 * float(np.linalg.norm(tp.mean(axis=0) - ac[j])), 0)})
            print(json.dumps(out[-1]), flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "adapters" / "libero_panda"))
    import client_env  # noqa: E402

    client_env.reexec_with_client_environment()
    main()
