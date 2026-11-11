#!/usr/bin/env python3
"""(2026-11-11 dev-57, offline CPU, GT = scoring only) SafeLIBERO collided episodes: WHAT touched the obstacle first, and how
close did the robot really get?  Rebuilds each env from the rollout history (as qe_csc_contacts.py) and, per control step:
  - MuJoCo contacts of the active obstacle with anything except its support (body + geom names),
  - the true minimum distance robot geoms <-> obstacle geoms (mj_geomDistance, cap 0.2 m) and the robot geom that is nearest.
Per collided episode it prints the first toucher (robot link / target / other object), the frame, and the true robot distance
in the 12 frames up to collide_t. Run with .venv-libero:
  LIBERO_CONFIG_PATH=<checkout>/.runtime/libero .venv-libero/bin/python first_contact.py --run-dir RUN [...] --out FILE.json"""
import argparse
import json
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path("/mnt/dev/AG3S_VLA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, action="append")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wait", type=int, default=20)
    ap.add_argument("--all", action="store_true", help="also the episodes without a collision (true min distance only)")
    a = ap.parse_args()
    sys.path.append(str(PROJECT_ROOT))
    import numpy as np
    import mujoco
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "runners"))
    from libero_safe_client import activate_safelibero
    activate_safelibero()
    from benchmark.adapters.libero_panda import env_history
    from benchmark.adapters.libero_panda import task_target as tt
    _F = PROJECT_ROOT / "benchmark/SafeLIBERO/safelibero/libero/libero"
    tt.LIBERO_ROOT, tt.BDDL_ROOT = _F, _F / "bddl_files"
    tt.TASK_MAP_FILE, tt.BENCHMARK_INIT_FILE = _F / "benchmark/libero_suite_task_map.py", _F / "benchmark/__init__.py"

    out = []
    for run_dir in a.run_dir:
        run = pathlib.Path(run_dir).resolve()
        rows = list(env_history.read_rows(run))
        wanted = {env_history.stem_of(r) for r in rows
                  if a.all or (r.get("safelibero") or {}).get("collided")}
        if not wanted:
            continue
        env_history.run_history(run)
        for row, env, _note in env_history.episode_envs(run, wanted):
            stem = env_history.stem_of(row)
            sl = row.get("safelibero") or {}
            obstacle = str(sl.get("obstacle"))
            try:
                target = tt.task_target(row["suite"], int(row["task_id"]))["target"]
            except Exception:   # noqa: BLE001
                target = None
            sim = env.sim
            m, d = sim.model._model, sim.data._data
            bname = lambda gi: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
            gname = lambda gi: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gi)) or f"geom{int(gi)}"
            # collision geoms only (contype / conaffinity != 0)
            col = [gi for gi in range(m.ngeom) if (m.geom_contype[gi] or m.geom_conaffinity[gi])]
            obst_g = [gi for gi in col if obstacle in bname(gi)]
            robot_g = [gi for gi in col if bname(gi).startswith(("robot0_", "gripper0_"))]
            g = np.load(str(run / "episodes" / (stem + "_geom.npz")))
            idx = {int(f): i for i, f in enumerate(np.asarray(g["obs_index"]).reshape(-1))}
            ct = sl.get("collide_t")
            first, touch_log, dist_log = None, [], {}
            fromto = np.zeros(6)
            for f in sorted(idx):
                if f < a.wait:
                    continue
                sim.set_state_from_flattened(np.asarray(g["sim_state_image"][idx[f]]).copy())
                sim.forward()
                others = set()
                for i in range(d.ncon):
                    c = d.contact[i]
                    b1, b2 = bname(c.geom1), bname(c.geom2)
                    if (obstacle in b1) == (obstacle in b2):
                        continue
                    og = c.geom2 if obstacle in b1 else c.geom1
                    ob = bname(og)
                    if ob.startswith(("robot0_", "gripper0_")) or (target and ob.startswith(target)) or (
                            first is not None) or (ct is not None and abs(f - ct) <= 3):
                        others.add((ob, gname(og)))
                movers = {o for o in others if o[0].startswith(("robot0_", "gripper0_")) or (target and o[0].startswith(target))}
                if movers and first is None:
                    first = f
                if movers and len(touch_log) < 6:
                    touch_log.append({"f": f, "touch": sorted(f"{b}/{gn}" for b, gn in movers)})
                near = (ct is not None and ct - 12 <= f <= ct + 2) or (ct is None)
                if near and obst_g and robot_g:
                    best, who, pts = 0.2, None, None
                    for rg in robot_g:
                        for og in obst_g:
                            dd = mujoco.mj_geomDistance(m, d, rg, og, best, fromto)
                            if dd < best:
                                best, who = float(dd), f"{bname(rg)}/{gname(rg)}"
                                pts = [round(float(v), 4) for v in fromto]   # world: nearest point on the robot geom, on the obstacle geom
                    dist_log[f] = (round(1000 * best, 1), who, pts if who else None)
            rec = {"run": run.name, "stem": stem, "task": int(row["task_id"]), "init": row.get("init_state_index"),
                   "obstacle": obstacle, "target": target, "collided": bool(sl.get("collided")), "collide_t": ct,
                   "official_success": bool(row.get("official_success")),
                   "max_displacement_mm": round(1000 * float(sl.get("max_displacement_m") or 0.0), 1),
                   "first_contact_obs": first, "touch_log": touch_log}
            if ct is not None:
                rec["true_robot_dist_mm"] = {str(f): v for f, v in sorted(dist_log.items())}
            else:
                vals = [v for v in dist_log.values()]
                rec["true_robot_dist_min_mm"] = min(vals) if vals else None
            out.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "adapters" / "libero_panda"))
    import client_env  # noqa: E402

    client_env.reexec_with_client_environment()
    main()
