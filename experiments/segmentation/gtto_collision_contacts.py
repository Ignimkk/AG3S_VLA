#!/usr/bin/env python3
"""SafeLIBERO collision analysis (2026-11-08): WHAT touched the obstacle when the episode recorded `safelibero.collide_t`?

Rebuilds each collided episode's env with the rollout's history (`env_history.episode_envs`, as panda_geom_snapshot.py), restores
the recorded `sim_state_image` at the frames around `collide_t` (collide_t-6 .. collide_t+2), runs mj_forward and lists the active
MuJoCo contacts whose bodies include the obstacle (`safelibero.obstacle`): the other body (robot link / the grasped target / the
table ...) and the penetration depth. Run with .venv-libero (Python 3.8; re-execs into the client environment like
panda_geom_snapshot.py):
  LIBERO_CONFIG_PATH=<checkout>/.runtime/libero .venv-libero/bin/python experiments/segmentation/gtto_collision_contacts.py \
      --run-dir RUN --out FILE.json"""
import argparse
import json
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, action="append")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.append(str(PROJECT_ROOT))
    import numpy as np
    import mujoco
    if any("safelibero" in r for r in a.run_dir):
        sys.path.append(str(PROJECT_ROOT / "benchmark" / "runners"))
        from libero_safe_client import activate_safelibero
        activate_safelibero()
    from benchmark.adapters.libero_panda import env_history

    out = []
    for run_dir in a.run_dir:
        run = pathlib.Path(run_dir).resolve()
        rows = list(env_history.read_rows(run))
        wanted = {env_history.stem_of(r) for r in rows if (r.get("safelibero") or {}).get("collided")}
        if not wanted:
            continue
        env_history.run_history(run)
        for row, env, _note in env_history.episode_envs(run, wanted):
            stem = env_history.stem_of(row)
            sl = row.get("safelibero") or {}
            obstacle = str(sl.get("obstacle"))
            sim = env.sim
            m, d = sim.model._model, sim.data._data
            g = np.load(str(run / "episodes" / (stem + "_geom.npz")))
            idx = {int(f): i for i, f in enumerate(np.asarray(g["obs_index"]).reshape(-1))}
            ct = int(sl.get("collide_t"))
            frames = []
            ob_ids = [i for i in range(m.nbody) if obstacle in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")]
            hand_ids = [i for i in range(m.nbody) if "gripper0" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")]
            track = []
            for f in sorted(idx):
                if f > ct + 2:
                    break
                sim.set_state_from_flattened(np.asarray(g["sim_state_image"][idx[f]]).copy())
                sim.forward()
                p0 = np.asarray(d.xpos[ob_ids[0]]).copy()
                hd = min(float(np.linalg.norm(np.asarray(d.xpos[h]) - p0)) for h in hand_ids) if hand_ids else None
                track.append({"obs": f, "pos": [round(float(v), 5) for v in p0], "hand_dist_m": None if hd is None else round(hd, 4)})
            # minimum robot <-> obstacle geom distance (mj_geomDistance, 50 mm cap) at collide_t-4 .. collide_t+1: a touch
            # between two recorded frames leaves no contact at either frame, but the gap shrinks to ~0 there
            coll = [gi for gi in range(m.ngeom) if int(m.geom_contype[gi]) != 0 or int(m.geom_conaffinity[gi]) != 0]
            ob_geoms = [gi for gi in coll if int(m.geom_bodyid[gi]) in ob_ids]
            rb_geoms = [gi for gi in coll
                        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or "").startswith(("robot0_", "gripper0_"))]
            fromto = np.zeros(6)
            gaps = []
            for f in range(ct - 4, ct + 2):
                if f not in idx:
                    continue
                sim.set_state_from_flattened(np.asarray(g["sim_state_image"][idx[f]]).copy())
                sim.forward()
                best = (0.05, None)
                for a_ in rb_geoms:
                    for b_ in ob_geoms:
                        dd = mujoco.mj_geomDistance(m, d, a_, b_, 0.05, fromto)
                        if dd < best[0]:
                            best = (dd, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[a_])))
                gaps.append({"obs": f, "min_gap_mm": round(1000.0 * best[0], 2), "robot_body": best[1]})
            p_first = np.asarray(track[0]["pos"])
            for t in track:
                t["disp_mm"] = round(1000.0 * float(np.linalg.norm(np.asarray(t["pos"]) - p_first)), 2)
            for f in range(6, ct + 3):
                if f not in idx:
                    continue
                sim.set_state_from_flattened(np.asarray(g["sim_state_image"][idx[f]]).copy())
                sim.forward()
                hits = []
                for i in range(d.ncon):
                    c = d.contact[i]
                    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1])) or ""
                    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2])) or ""
                    if obstacle in b1 or obstacle in b2:
                        other = b2 if obstacle in b1 else b1
                        hits.append({"other": other, "depth_mm": round(-1000.0 * float(c.dist), 2)})
                robot_any = []
                for i in range(d.ncon):
                    c = d.contact[i]
                    b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1])) or ""
                    b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2])) or ""
                    r1, r2 = b1.startswith(("robot0_", "gripper0_")), b2.startswith(("robot0_", "gripper0_"))
                    if r1 != r2:
                        robot_any.append({"robot": b1 if r1 else b2, "other": b2 if r1 else b1,
                                          "depth_mm": round(-1000.0 * float(c.dist), 2)})
                frames.append({"obs": f, "contacts": hits, "robot_contacts": robot_any})
            out.append({"run": run.name, "stem": stem, "obstacle": obstacle, "collide_t": ct, "frames": frames, "track": track, "robot_gap": gaps})
            print(json.dumps({"stem": stem, "collide_t": ct,
                              "others": sorted({h["other"] for fr in frames for h in fr["contacts"]})}), flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "adapters" / "libero_panda"))
    import client_env  # noqa: E402

    client_env.reexec_with_client_environment()
    main()
