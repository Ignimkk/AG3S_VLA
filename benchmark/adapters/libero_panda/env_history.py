"""Reproduce a run's LIBERO env history for offline re-rendering (SL4b). Python 3.8 compatible.

Why: the client (``libero_pi05_client.run_tasks``) builds **one env per task** (``env.seed`` =
``np.random.seed``) and starts every episode with ``env.reset()`` + ``set_init_state``
(``libero_pi05_client.start_episode``). LIBERO's reset is a hard reset that re-samples the **fixture**
placement (``model.body_pos`` of ``wooden_cabinet_1`` / ``flat_stove_1`` / ...: no joints, so not in the
recorded simulator state) from the global numpy RNG. Episode k of a task therefore shows the fixtures of the
(k+1)-th reset of that task's env; a fresh env with one reset reproduces only the task's first episode
(SL4b verifier finding: init 2-4 re-rendered with ~1 cm shifted cabinet / stove).

What is reproduced: per task block, the env built by ``libero_pi05_client.make_task_env`` with the run's seed
and geometry flag, and ``start_episode`` for every episode of the block **in the recorded order** (the
episodes' steps are not replayed: robosuite's step path draws no numpy random numbers with the default
identity corrupters / zero delayers, and the hard reset rebuilds the simulator, so only the number of resets
carries over). Callers prove it per frame: :func:`rgb_rerender_equal` must hold for every exported call frame
(``scene_export`` / ``phase_export`` raise otherwise).

Order source: ``episodes.jsonl`` line order (the client appends and fsyncs one line per finished episode), checked
to equal the manifest's ``settings.protocol`` ``task_ids x episode_ids`` (the client loop) and to have
non-decreasing ``started_at``. Anything else is refused -- the order is never assumed.
"""
from __future__ import annotations

import json
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
#: (robosuite camera, recorded obs key) of the RGB hard check -- the two cameras the client records.
RGB_CHECK_CAMERAS = (("agentview", "obs_agentview_image"), ("robot0_eye_in_hand", "obs_robot0_eye_in_hand_image"))
RGB_RESOLUTION = 256


def stem_of(row):
    return "{}_t{}_e{}".format(row["suite"], int(row["task_id"]), int(row["episode_idx"]))


def read_rows(run):
    run = pathlib.Path(run)
    return [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines() if line.strip()]


def read_protocol(run):
    """``settings.protocol`` of the run manifest (``None`` if the run has none)."""
    path = pathlib.Path(run) / "run_manifest.json"
    if not path.is_file():
        return None
    return (json.loads(path.read_text()).get("settings") or {}).get("protocol")


def history_blocks(rows, protocol):
    """Split the recorded episodes into the client's env blocks (one env per ``task_ids`` entry).

    ``rows``: ``episodes.jsonl`` rows in line order; ``protocol``: the manifest's ``settings.protocol``.
    Returns ``[{"block": i, "suite", "task_id", "seed", "capture_geometry", "rows": [row, ...]}]`` in execution
    order. Raises ``ValueError`` unless the line order is exactly the client loop over the manifest's
    ``task_ids x episode_ids`` with one suite / seed / geometry flag per block and non-decreasing ``started_at``.
    """
    if not protocol or "task_ids" not in protocol or "episode_ids" not in protocol:
        raise ValueError("run manifest has no settings.protocol task_ids / episode_ids: the env history is unknown")
    task_ids = [int(t) for t in protocol["task_ids"]]
    episode_ids = [int(e) for e in protocol["episode_ids"]]
    expected = [(t, e) for t in task_ids for e in episode_ids]
    observed = [(int(r["task_id"]), int(r["episode_idx"])) for r in rows]
    if observed != expected:
        raise ValueError("episodes.jsonl order {} is not the client loop over task_ids {} x episode_ids {} "
                         "(incomplete or mixed run): the env history is unknown".format(observed, task_ids, episode_ids))
    started = [r.get("started_at") for r in rows]
    if all(s is not None for s in started) and any(b < a for a, b in zip(started, started[1:])):
        raise ValueError("episodes.jsonl lines are not in started_at order")
    blocks = []
    n = len(episode_ids)
    for i, task_id in enumerate(task_ids):
        block_rows = rows[i * n:(i + 1) * n]
        for key in ("suite", "seed"):
            values = set(json.dumps(r.get(key)) for r in block_rows)
            if len(values) != 1:
                raise ValueError("task block {} (task {}) mixes {} values {}".format(i, task_id, key, sorted(values)))
        geometry = set(bool(r.get("geometry")) for r in block_rows)
        if len(geometry) != 1:
            raise ValueError("task block {} (task {}) mixes geometry capture on/off".format(i, task_id))
        if protocol.get("seed") is not None and int(block_rows[0]["seed"]) != int(protocol["seed"]):
            raise ValueError("task block {}: row seed {} != manifest seed {}".format(i, block_rows[0]["seed"],
                                                                                     protocol["seed"]))
        if protocol.get("suite") is not None and block_rows[0]["suite"] != protocol["suite"]:
            raise ValueError("task block {}: row suite {} != manifest suite {}".format(i, block_rows[0]["suite"],
                                                                                       protocol["suite"]))
        blocks.append({"block": i, "suite": block_rows[0]["suite"], "task_id": task_id, "seed": int(block_rows[0]["seed"]),
                       "capture_geometry": geometry.pop(), "rows": block_rows})
    return blocks


def run_history(run):
    """:func:`history_blocks` of a run directory."""
    return history_blocks(read_rows(run), read_protocol(run))


def plan(blocks, wanted_stems=None):
    """Which resets to repeat: ``[(block, [(row, wanted), ...])]`` up to the last wanted episode of each block.

    Blocks without a wanted episode are skipped; resets after the last wanted episode are not needed.
    ``wanted_stems=None`` = every episode."""
    out = []
    for block in blocks:
        flags = [wanted_stems is None or stem_of(r) in wanted_stems for r in block["rows"]]
        if not any(flags):
            continue
        last = max(i for i, f in enumerate(flags) if f)
        out.append((block, list(zip(block["rows"][:last + 1], flags[:last + 1]))))
    return out


def history_note(block, position):
    """Provenance of the env state an episode is exported from."""
    rows = block["rows"]
    return {"order_source": "episodes.jsonl line order == manifest protocol task_ids x episode_ids (client loop)",
            "task_block": int(block["block"]), "seed": int(block["seed"]),
            "capture_geometry_env": bool(block["capture_geometry"]),
            "resets_before": int(position), "reset_index": int(position) + 1,
            "preceding_episodes": [stem_of(r) for r in rows[:position]],
            "reproduced": "libero_pi05_client.make_task_env + start_episode per preceding episode (steps not replayed)"}


def _client():
    """``libero_pi05_client`` (its ``make_task_env`` / ``start_episode`` are the rollout's own code)."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))  # appended: never shadows an installed package
    from benchmark.runners import libero_pi05_client as client

    return client


def episode_envs(run, wanted_stems=None):
    """Yield ``(row, env, note)`` at the start of every wanted episode of ``run``, after the client's env history.

    ``env`` is the task's env right after ``start_episode`` of that episode (LIBERO, ``.venv-libero``); it stays
    valid until the next item is requested and is closed when its block ends."""
    from libero.libero import benchmark as lb

    client = _client()
    for block, items in plan(run_history(run), wanted_stems):
        suite = lb.get_benchmark_dict()[block["suite"]]()
        task = suite.get_task(block["task_id"])
        initial_states = suite.get_task_init_states(block["task_id"])
        env, _ = client.make_task_env(task, block["seed"], block["capture_geometry"])
        try:
            for position, (row, wanted) in enumerate(items):
                client.start_episode(env, initial_states, int(row["episode_idx"]))
                if wanted:
                    yield row, env, history_note(block, position)
        finally:
            env.close()


def rgb_rerender_equal(sim, episode, frame, resolution=RGB_RESOLUTION):
    """``{robosuite camera: bool}``: RGB rendered at the current sim state == the recorded obs image of ``frame``."""
    import numpy as np

    out = {}
    for source, key in RGB_CHECK_CAMERAS:
        rgb = sim.render(camera_name=source, height=resolution, width=resolution)
        out[source] = bool(np.array_equal(rgb, episode[key][frame]))
    return out


def require_rgb_equal(stem, counts, frames_checked):
    """Raise unless every checked call frame re-rendered bit-identical on every camera (``counts`` per camera)."""
    bad = {cam: (frames_checked - n) for cam, n in counts.items() if n != frames_checked}
    if bad:
        raise ValueError("{}: re-rendered RGB differs from the recorded frames ({} of {} call frames per camera): "
                         "the env history / fixture placement is not the rollout's".format(stem, bad, frames_checked))
