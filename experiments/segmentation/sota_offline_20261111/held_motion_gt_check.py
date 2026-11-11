#!/usr/bin/env python3
"""(2026-11-11 dev-57, offline CPU, GT = scoring only) SL-SOTA5 safety check: replay SL-PE8's tracker on a run's server
records with the current reach and with a reach floor, and label every point it removes with the real body it lies on
(sim state of that call; mesh vertices / primitive boxes, as surface_owner.py). Wanted: more held-object points removed,
no obstacle points removed.
usage: .venv-libero/bin/python held_motion_gt_check.py --run-dir RUN [--run-dir ...] --reach-min-mm 150 --out FILE.json"""
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
    ap.add_argument("--run-dir", required=True, action="append")
    ap.add_argument("--reach-min-mm", type=float, default=150.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tracker", default=str(PROJECT_ROOT / "AG3S/ag3s/stages/held_motion.py"))
    a = ap.parse_args()
    sys.path.append(str(PROJECT_ROOT))
    import numpy as np
    import mujoco
    from scipy.spatial import cKDTree
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "runners"))
    from libero_safe_client import activate_safelibero
    activate_safelibero()
    from benchmark.adapters.libero_panda import env_history
    import importlib.util   # the tracker file alone (numpy / scipy only); --tracker picks the checkout that has reach_min_m
    spec = importlib.util.spec_from_file_location("held_motion_under_test", a.tracker)
    hm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hm)
    HeldMotionTracker = hm.HeldMotionTracker

    def kabsch(A, Bp):
        ca, cb = A.mean(0), Bp.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (Bp - cb))
        R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, cb - R @ ca
        return T

    out = []
    for run_dir in a.run_dir:
        run = pathlib.Path(run_dir).resolve()
        rows = list(env_history.read_rows(run))
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
        for row, env, _ in env_history.episode_envs(run, wanted):
            stem = env_history.stem_of(row)
            k = [i for i, e in enumerate(eps) if int(e["episode_idx"]) == int(row["episode_idx"])][0]
            obstacle = str((row.get("safelibero") or {}).get("obstacle"))
            sim = env.sim
            m, d = sim.model._model, sim.data._data
            g = np.load(str(run / "episodes" / (stem + "_geom.npz")))
            bname = lambda gi: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
            local = {}
            for gi in range(m.ngeom):
                b = bname(gi)
                if not b or b == "world" or "table" in b or "floor" in b or "wall" in b:
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
            cs = [json.loads(x) for x in open(run / "episodes" / (stem + "_calls.jsonl"))]
            trackers = {"now": HeldMotionTracker(), "floor": HeldMotionTracker(reach_min_m=a.reach_min_mm / 1e3)}
            tot = {n: collections.Counter() for n in trackers}
            ref, held_calls = None, 0
            ch = dict(per[k])
            for c in cs:
                q = c["seq"]
                if q not in ch:
                    continue
                z = np.load(ch[q])
                names = [str(n) for n in json.loads(str(z["summary_json"])).get("sphere_link_names", [])]
                sc = z["sphere_centres"].astype(float)
                hi = [i for i, n in enumerate(names) if n == "panda_hand"]
                if len(hi) < 3:
                    continue
                if ref is None:
                    ref = sc[hi].copy()
                T = kabsch(ref, sc[hi])
                gr = c.get("grasp") or {}
                tip = lit(lit(gr.get("tip_alignment")).get("now")).get("tip_m")
                held = gr.get("state") == "held"
                cloud = z["filtered_cloud"].astype(float)[~z["support_mask"].astype(bool)]
                tgt = z["target_points"].astype(float) if "target_points" in z.files and len(z["target_points"]) else None
                masks = {n: tr.update(cloud, T, None if tip is None else np.asarray(tip, float), held, None if held else tgt)
                         for n, tr in trackers.items()}
                if not held or not any(mk.any() for mk in masks.values()):
                    held_calls += int(held)
                    continue
                held_calls += 1
                i = int(np.argmin(np.abs(np.asarray(g["obs_index"]) - (20 + 5 * q))))
                sim.set_state_from_flattened(np.asarray(g["sim_state_image"][i]).copy())
                sim.forward()
                Tw = np.asarray(g["T_world_base"][i])
                sel = masks["now"] | masks["floor"]
                pw = cloud[sel] @ Tw[:3, :3].T + Tw[:3, 3]
                best = np.full(len(pw), np.inf)
                owner = np.full(len(pw), -1)
                for gi, v in local.items():
                    R = np.asarray(d.geom_xmat[gi], float).reshape(3, 3)
                    dist = cKDTree(v @ R.T + np.asarray(d.geom_xpos[gi], float)).query(pw)[0]
                    better = dist < best
                    best[better], owner[better] = dist[better], gi
                lab = np.array([bname(int(o)) for o in owner])
                for n, mk in masks.items():
                    for b in lab[mk[sel]]:
                        key = ("obstacle" if obstacle in b else "robot" if b.startswith(("robot0_", "gripper0_")) else
                               b.replace("_main", ""))
                        tot[n][key] += 1
            rec = {"run": run.name[:40], "task": int(row["task_id"]), "init": row.get("init_state_index"), "obstacle": obstacle,
                   "held_calls": held_calls, **{n: dict(v.most_common(6)) for n, v in tot.items()}}
            out.append(rec)
            print(json.dumps(rec), flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "adapters" / "libero_panda"))
    import client_env  # noqa: E402

    client_env.reexec_with_client_environment()
    main()
