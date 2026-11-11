#!/usr/bin/env python3
"""(2026-11-11 dev-57, offline CPU, GT = scoring only) Which real object is next to the worst obstacle-object row of each call?
Rebuilds the env (as first_contact.py), sets the state of the call's frame, and for the recorded worst row point
(`passthrough_clear.obstacle_rows.worst.point_m`, base frame) lists the nearest non-robot body (point <-> geom distance by
mj_geomDistance against a 1 mm probe sphere is not available for free points, so: nearest collision geom by
|p - geom centre| - geom rbound, then refined per geom type is skipped -- a ranking, not a distance).
usage: .venv-libero/bin/python row_owner.py --run-dir RUN --episode E --out FILE.json"""
import argparse, ast, glob, json, os, pathlib, sys

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
    a = ap.parse_args()
    sys.path.append(str(PROJECT_ROOT))
    import numpy as np
    import mujoco
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "runners"))
    from libero_safe_client import activate_safelibero
    activate_safelibero()
    from benchmark.adapters.libero_panda import env_history
    run = pathlib.Path(a.run_dir).resolve()
    rows = [r for r in env_history.read_rows(run) if int(r["episode_idx"]) == a.episode]
    wanted = {env_history.stem_of(r) for r in rows}
    env_history.run_history(run)
    # chunks of this episode
    eps = [json.loads(l) for l in open(run / "episodes.jsonl") if l.strip()]
    per, cur, last = [], [], None
    for c in sorted(glob.glob(str(run / "server_constraints/run_*/chunk_*.npz"))):
        z = np.load(c)
        ts = int(z["t_step"])
        if last is not None and ts <= last:
            per.append(cur)
            cur = []
        cur.append((ts, json.loads(str(z["summary_json"]))))
        last = ts
    per.append(cur)
    k = [i for i, e in enumerate(eps) if int(e["episode_idx"]) == a.episode][0]
    out = []
    for row, env, _ in env_history.episode_envs(run, wanted):
        stem = env_history.stem_of(row)
        sim = env.sim
        m, d = sim.model._model, sim.data._data
        g = np.load(str(run / "episodes" / (stem + "_geom.npz")))
        bname = lambda gi: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
        col = [gi for gi in range(m.ngeom) if (m.geom_contype[gi] or m.geom_conaffinity[gi])
               and not bname(gi).startswith(("robot0_", "gripper0_"))]
        for ts, sm in per[k]:
            mt = (sm.get("to") or {}).get("metrics") or {}
            pc = lit(mt.get("passthrough_clear"))
            w = (pc.get("obstacle_rows") or {}).get("worst") or {}
            if not w.get("point_m"):
                continue
            f = 20 + 5 * ts
            i = int(np.argmin(np.abs(np.asarray(g["obs_index"]) - f)))
            sim.set_state_from_flattened(np.asarray(g["sim_state_image"][i]).copy())
            sim.forward()
            pw = (np.asarray(g["T_world_base"][i]) @ np.r_[np.asarray(w["point_m"], float), 1.0])[:3]
            score = sorted(((float(np.linalg.norm(d.geom_xpos[gi] - pw) - m.geom_rbound[gi]), bname(gi)) for gi in col))
            near = []
            for s, b in score:
                if b not in [x[1] for x in near]:
                    near.append((round(1000 * s), b))
                if len(near) == 3:
                    break
            out.append({"seq": ts, "applied": bool(pc.get("applied")), "link": w.get("link"), "step": w.get("step"),
                        "violation_mm": round(1000 * float((pc.get("obstacle_rows") or {}).get("violation_m") or 0), 1),
                        "grasp": (sm.get("grasp") or {}).get("state"), "nearest_bodies": near})
            print(json.dumps(out[-1]), flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "adapters" / "libero_panda"))
    import client_env  # noqa: E402

    client_env.reexec_with_client_environment()
    main()
