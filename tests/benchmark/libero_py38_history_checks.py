"""SL4b env-history checks on real LIBERO -- run by ``.venv-libero`` (Python 3.8, no pytest).

Driven by ``tests/benchmark/test_panda_sl4b_history.py`` via subprocess::

    .venv-libero/bin/python tests/benchmark/libero_py38_history_checks.py --run <geometry run> --task 0 \\
        --episodes 1,2,3 --frames 2 --out DIR

For the first ``--frames`` policy-call frames of each listed episode of ``--task`` it restores the recorded
image-instant state and compares the re-rendered RGB (both cameras) with the recorded frames, in three envs:

* ``history`` -- ``env_history.episode_envs`` (the rollout's env history; what the exports use);
* ``fresh``   -- a new env + one ``start_episode`` per episode (the pre-fix exports);
* ``replay_last`` -- the replay path for the last listed episode alone: ``libero_replay.source_history_resets``
  + ``start_episode`` of those source episodes (``run_tasks``' ``prepare_episode``), then its own start.

Also records each env's fixture ``body_pos`` (cabinet / stove roots). Writes ``DIR/history_checks.json``.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "benchmark/runners"))

import numpy as np  # noqa: E402

from benchmark.adapters.libero_panda import env_history  # noqa: E402
from benchmark.runners import libero_pi05_client as client  # noqa: E402

FIXTURE_ROOTS = ("wooden_cabinet_1_main", "flat_stove_1_main")


def frame_check(env, run, row, n_frames):
    """Per call frame (first ``n_frames``): {camera: equal} after restoring ``sim_state_image``."""
    stem = env_history.stem_of(row)
    e = np.load(str(pathlib.Path(run) / row["npz"]))
    g = np.load(str(pathlib.Path(run) / "episodes/{}_geom.npz".format(stem)))
    sim_state = np.asarray(g["sim_state_image"])
    obs_index = [int(x) for x in np.asarray(g["obs_index"])]
    images = {key: np.asarray(e[key]) for _, key in env_history.RGB_CHECK_CAMERAS}
    out = []
    for f in [int(t) for t in np.asarray(e["chunk_t"])][:n_frames]:
        env.sim.set_state_from_flattened(sim_state[obs_index.index(f)].copy())
        env.sim.forward()
        out.append(env_history.rgb_rerender_equal(env.sim, images, f))
    return out


def fixtures(env):
    m = env.sim.model
    return {name: [float(x) for x in m.body_pos[m.body_name2id(name)]] for name in FIXTURE_ROOTS
            if name in m.body_names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--task", type=int, default=0)
    parser.add_argument("--episodes", default="1,2,3")
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run = pathlib.Path(args.run)
    episodes = [int(x) for x in args.episodes.split(",")]
    blocks = [b for b in env_history.run_history(run) if b["task_id"] == args.task]
    rows = {int(r["episode_idx"]): r for r in blocks[0]["rows"]}
    wanted = set(env_history.stem_of(rows[k]) for k in episodes)
    result = {"run": str(run), "task": args.task, "episodes": episodes, "history": {}, "fresh": {}}

    for row, env, note in env_history.episode_envs(run, wanted):
        result["history"][env_history.stem_of(row)] = {"resets_before": note["resets_before"], "fixtures": fixtures(env),
                                                       "frames": frame_check(env, run, row, args.frames)}

    from libero.libero import benchmark as lb

    suite = lb.get_benchmark_dict()[blocks[0]["suite"]]()
    task, init = suite.get_task(args.task), suite.get_task_init_states(args.task)
    for k in episodes:
        env, _ = client.make_task_env(task, blocks[0]["seed"], blocks[0]["capture_geometry"])
        client.start_episode(env, init, k)
        result["fresh"][env_history.stem_of(rows[k])] = {"fixtures": fixtures(env),
                                                         "frames": frame_check(env, run, rows[k], args.frames)}
        env.close()

    import libero_replay

    last = episodes[-1]
    resets = libero_replay.source_history_resets(env_history.run_history(run), [args.task], [last])[(args.task, last)]
    env, _ = client.make_task_env(task, blocks[0]["seed"], blocks[0]["capture_geometry"])
    for skipped in resets:
        client.start_episode(env, init, skipped)
    client.start_episode(env, init, last)
    result["replay_last"] = {"stem": env_history.stem_of(rows[last]), "source_history_resets": resets,
                             "fixtures": fixtures(env), "frames": frame_check(env, run, rows[last], args.frames)}
    env.close()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "history_checks.json").write_text(json.dumps(result, indent=1))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
