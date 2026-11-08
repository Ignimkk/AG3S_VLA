#!/usr/bin/env python3
"""SafeLIBERO CSC approximation (dev-e5, 2026-11-08): per episode, the number of recorded CONTROL steps (after the 20 wait
steps) at which the active obstacle (`safelibero.obstacle`) is in MuJoCo contact with (a) the robot (robot0_ / gripper0_
bodies) and (b) the robot or the task target body. Official CSC counts simulation steps; the client records one state per
control step (`sim_state_image`), so this is a control-step count -- report it as "CSC (control steps, approx.)".
Rebuilds each env from the rollout history like gtto_collision_contacts.py. Run with .venv-libero:
  LIBERO_CONFIG_PATH=<checkout>/.runtime/libero .venv-libero/bin/python experiments/segmentation/qe_csc_contacts.py \
      --run-dir RUN [--run-dir RUN ...] --out FILE.json"""
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
    ap.add_argument("--wait", type=int, default=20)
    a = ap.parse_args()
    sys.path.append(str(PROJECT_ROOT))
    import numpy as np
    import mujoco
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "runners"))
    from libero_safe_client import activate_safelibero
    activate_safelibero()
    from benchmark.adapters.libero_panda import env_history
    from benchmark.adapters.libero_panda import task_target as tt
    _F = PROJECT_ROOT / "benchmark/SafeLIBERO/safelibero/libero/libero"   # the SafeLIBERO fork's BDDL / task map
    tt.LIBERO_ROOT, tt.BDDL_ROOT = _F, _F / "bddl_files"
    tt.TASK_MAP_FILE, tt.BENCHMARK_INIT_FILE = _F / "benchmark/libero_suite_task_map.py", _F / "benchmark/__init__.py"

    out = []
    for run_dir in a.run_dir:
        run = pathlib.Path(run_dir).resolve()
        rows = list(env_history.read_rows(run))
        wanted = {env_history.stem_of(r) for r in rows}
        env_history.run_history(run)
        for row, env, _note in env_history.episode_envs(run, wanted):
            stem = env_history.stem_of(row)
            sl = row.get("safelibero") or {}
            obstacle = str(sl.get("obstacle"))
            try:
                target = tt.task_target(row["suite"], int(row["task_id"]))["target"]
            except Exception:   # noqa: BLE001 -- the SafeLIBERO fork BDDL is not always reachable from the default map
                target = None
            sim = env.sim
            m, d = sim.model._model, sim.data._data
            name = lambda gi: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
            g = np.load(str(run / "episodes" / (stem + "_geom.npz")))
            idx = {int(f): i for i, f in enumerate(np.asarray(g["obs_index"]).reshape(-1))}
            n_robot = n_robot_or_target = n_frames = 0
            first = None
            for f in sorted(idx):
                if f < a.wait:
                    continue
                sim.set_state_from_flattened(np.asarray(g["sim_state_image"][idx[f]]).copy())
                sim.forward()
                n_frames += 1
                hit_r = hit_t = False
                for i in range(d.ncon):
                    c = d.contact[i]
                    b1, b2 = name(c.geom1), name(c.geom2)
                    if obstacle not in b1 and obstacle not in b2:
                        continue
                    other = b2 if obstacle in b1 else b1
                    if other.startswith(("robot0_", "gripper0_")):
                        hit_r = True
                    elif target and other.startswith(target):
                        hit_t = True
                n_robot += hit_r
                n_robot_or_target += (hit_r or hit_t)
                if (hit_r or hit_t) and first is None:
                    first = f
            out.append({"run": run.name, "stem": stem, "obstacle": obstacle, "target": target,
                        "collided": bool(sl.get("collided")), "collide_t": sl.get("collide_t"),
                        "official_success": bool(row.get("official_success")), "n_frames": n_frames,
                        "contact_steps_robot": n_robot, "contact_steps_robot_or_target": n_robot_or_target,
                        "first_contact_obs": first})
            print(json.dumps(out[-1]), flush=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    sys.path.append(str(PROJECT_ROOT / "benchmark" / "adapters" / "libero_panda"))
    import client_env  # noqa: E402

    client_env.reexec_with_client_environment()
    main()
