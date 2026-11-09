"""(.venv-ag3s, CPU) SL-PE safety metric (2026-11-09, dev-30; asked by dev-38): per SafeLIBERO run,
(a) pre-held calls whose grasp-seed cut (split_at_seed on the recorded cloud, as the server) lies on the SafeLIBERO obstacle body
    (`episodes.jsonl` safelibero.obstacle), and
(b) calls where the server's identity (recorded target points) lies on that obstacle -- then the obstacle is out of the
    obstacle-object set (dev-38 `scene_obstacles` recomputed: obstacle voxels in the set).
Writes safety_<tag>.json and prints per-episode counts. Run: SCENE_DIR=<scene24S> SAFELIBERO_PATCH=1 python pe_safety.py '<glob>' <tag>"""
import collections
import json
import pathlib
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pe_signal as PS  # noqa: E402
sys.path.insert(0, "/mnt/dev/AG3S_VLA")
from AG3S.ag3s.stages.close_seed import split_at_seed  # noqa: E402

Q = PS.Q
OUT = pathlib.Path(__file__).resolve().parent


def body_majority(points_world, tree, bodies):
    if points_world is None or not len(points_world):
        return None
    d, j = tree.query(points_world)
    c = collections.Counter(bodies[j][d <= 0.015].tolist())
    return c.most_common(1)[0][0] if c else None


def main():
    run = next(p for p in sorted(Q.E.RUNS.glob(sys.argv[1])) if not p.name.startswith("INVALID"))
    tag = sys.argv[2]
    eps = [json.loads(l) for l in open(run / "episodes.jsonl")]
    order = [f"{e['suite']}_t{e['task_id']}_e{e['episode_idx']}" for e in eps]
    obst = {o: (e.get("safelibero") or {}).get("obstacle") for o, e in zip(order, eps)}
    coll = {o: (e.get("safelibero") or {}) for o, e in zip(order, eps)}
    ch = PS.episode_chunks(run, order)
    res = {}
    for stem in order:
        if not (Q.SCENE / f"{stem}_scene.xml").exists() or not obst.get(stem):
            continue
        info, m, rg, cache, stc = PS.scene_for(stem)
        d = mujoco.MjData(m)
        bn = [m.body(i).name for i in range(m.nbody)]
        gbody = np.array([bn[b] for b in m.geom_bodyid])
        geom = np.load(run / "episodes" / f"{stem}_geom.npz")
        grow = {int(o): i for i, o in enumerate(geom["obs_index"])}
        calls = {int(json.loads(l)["seq"]): json.loads(l) for l in open(run / "episodes" / f"{stem}_calls.jsonl")}
        rows = []
        for ts, cp in ch.get(stem, []):
            c = calls[ts]
            g = c.get("grasp") or {}
            if g.get("state") not in PS.PRE:
                continue
            i = grow[int(c["obs_index"])]
            d.qpos[:] = geom["sim_state_image"][i][1:1 + m.nq]
            mujoco.mj_forward(m, d)
            S1, G1 = stc.world(d)
            S2, G2 = cache.world(d)
            tree = cKDTree(np.concatenate([S1, S2]))
            bodies = gbody[np.concatenate([G1, G2])]
            Tw = geom["T_world_base"][i]
            W = lambda X: np.asarray(X, float).reshape(-1, 3) @ Tw[:3, :3].T + Tw[:3, 3]
            z = np.load(cp)
            row = {"seq": ts, "latch": g.get("state")}
            pc = (g.get("tip_alignment") or {}).get("plan_close")
            if pc:
                idx, _ = split_at_seed(z["attention_cloud"].astype(float), pc["tip_m"], exclude_mask=z["support_mask"])
                if idx is not None:
                    row["seed_body"] = body_majority(W(z["attention_cloud"][idx]), tree, bodies)
            P = z["target_points"].astype(float) if "target_points" in z.files else None
            row["identity_body"] = body_majority(W(P), tree, bodies) if P is not None and len(P) else None
            ob = Q.scene_obstacles(z["filtered_cloud"].astype(float), z["support_mask"].astype(bool), P, None)
            dd, jj = tree.query(W(ob)) if len(ob) else (np.zeros(0), np.zeros(0, int))
            row["obstacle_voxels_in_set"] = int(sum(str(bodies[j]).startswith(obst[stem]) for j, x in zip(jj, dd) if x <= 0.015))
            rows.append(row)
        on = lambda b: b is not None and str(b).startswith(obst[stem])
        res[stem] = {"obstacle": obst[stem], "collided": coll[stem].get("collided"),
                     "max_displacement_mm": round(1000 * float(coll[stem].get("max_displacement_m") or 0), 1),
                     "seed_on_obstacle_calls": [r["seq"] for r in rows if on(r.get("seed_body"))],
                     "identity_on_obstacle_calls": [r["seq"] for r in rows if on(r.get("identity_body"))],
                     "identity_on_obstacle_and_obstacle_out_of_set": [
                         r["seq"] for r in rows if on(r.get("identity_body")) and r["obstacle_voxels_in_set"] == 0],
                     "rows": rows}
        r = res[stem]
        print(tag, stem[-6:], r["obstacle"], "| seed on obstacle", r["seed_on_obstacle_calls"], "| identity on obstacle",
              r["identity_on_obstacle_calls"], "| out of set", r["identity_on_obstacle_and_obstacle_out_of_set"],
              "| collided", r["collided"], r["max_displacement_mm"], "mm", flush=True)
    (OUT / f"safety_{tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
