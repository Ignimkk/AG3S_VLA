"""Policy-free action replay of a recorded LIBERO episode (SL1). Runs under .venv-libero -- keep 3.8 syntax.

Reads ``episodes/<stem>.npz`` of an existing run (the client's record) and re-runs the *same* episode
loop of ``libero_pi05_client`` (same seed order, env creation, ``set_init_state``, dummy wait, replan
cadence, ``done`` handling) with a :class:`ReplayPolicy` that answers the i-th policy call with the
i-th recorded action chunk. The executed actions are therefore the recorded ones; before running,
the recorded ``action`` array is checked to be exactly what the chunk/replan rule yields from
``action_chunks``. Outputs go to a *new* run (allocated by ``launch_libero_replay.py`` with
``common.run_artifacts.create_run``) in the client's format, optionally with ``--capture-geometry``.

If the replay diverges (e.g. ``done`` later than recorded) the policy runs out of chunks and the
episode ends as ``execution_error`` with that reason -- the replay never invents actions.

Env history (SL4b): LIBERO re-samples the fixture placement on every ``env.reset()`` of a task's env, so episode k
of a source task is only reproduced after the same resets. Before each replayed episode the source episodes of
that task that ran before it and are not replayed get their ``start_episode`` (reset + init state) on the replay
env (``source_history_resets``; source order from ``env_history.run_history``). Replaying the source's own
task/episode list adds nothing (the SL0/SL1/SL3 replays are unchanged); an order the source never ran is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pathlib
import sys

import numpy as np

try:  # .venv-ag3s unit tests import this module as part of the package
    from benchmark.runners import libero_pi05_client as client_mod
except ImportError:  # .venv-libero script: benchmark/runners is sys.path[0]
    import libero_pi05_client as client_mod
try:
    from benchmark.adapters.libero_panda import env_history
except ImportError:  # .venv-libero script: append (never prepend) the checkout root, as the client does
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
    from benchmark.adapters.libero_panda import env_history


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def executed_actions_from_chunks(source, replan_steps):
    """Actions the client loop executes given the recorded chunks (``chunk_index``/``is_wait``)."""
    is_wait = np.asarray(source["is_wait"], bool)
    chunk_index = np.asarray(source["chunk_index"], np.int64)
    chunks = np.asarray(source["action_chunks"], np.float64)
    out = np.zeros((len(is_wait), 7), np.float64)
    position = {}
    for step, (wait, chunk) in enumerate(zip(is_wait, chunk_index)):
        if wait:
            out[step] = client_mod.LIBERO_DUMMY_ACTION
            continue
        k = position.get(int(chunk), 0)
        if k >= replan_steps:
            raise ValueError("chunk {} used for more than replan_steps={} actions".format(chunk, replan_steps))
        out[step] = chunks[int(chunk)][k]
        position[int(chunk)] = k + 1
    return out


def check_replayable(source, replan_steps):
    """Raise unless the recorded ``action`` array is exactly reproduced from ``action_chunks``."""
    expected = executed_actions_from_chunks(source, replan_steps)
    recorded = np.asarray(source["action"], np.float64)
    if expected.shape != recorded.shape or not np.array_equal(expected, recorded):
        bad = np.flatnonzero(np.any(expected != recorded, axis=1)) if expected.shape == recorded.shape else []
        raise ValueError("recorded actions are not the chunk/replan rule applied to action_chunks "
                         "(first differing step: {})".format(int(bad[0]) if len(bad) else "shape"))
    chunk_index = np.asarray(source["chunk_index"], np.int64)
    used = sorted(set(int(c) for c in chunk_index if c >= 0))
    if used != list(range(len(np.asarray(source["action_chunks"])))):
        raise ValueError("recorded chunks are not consumed in order 0..n-1")


class ReplayPolicy:
    """Answers the i-th ``infer`` with the i-th recorded chunk; never invents actions."""

    def __init__(self, chunks, metadata):
        self._chunks = np.asarray(chunks, np.float64)
        self._metadata = dict(metadata)
        self.calls = 0

    def infer(self, element):
        if self.calls >= len(self._chunks):
            raise RuntimeError("replay exhausted: source episode recorded only {} policy calls".format(len(self._chunks)))
        chunk = np.array(self._chunks[self.calls], copy=True)
        self.calls += 1
        return {"actions": chunk}

    def get_server_metadata(self):
        return dict(self._metadata)


def read_source_rows(source_run: pathlib.Path):
    rows = {}
    with open(str(source_run / "episodes.jsonl")) as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                rows[(int(row["task_id"]), int(row["episode_idx"]))] = row
    return rows


def load_source_episode(source_run: pathlib.Path, row):
    npz_path = (source_run / row["npz"]).resolve()
    if source_run not in npz_path.parents:
        raise ValueError("source npz outside the source run: {}".format(npz_path))
    with np.load(str(npz_path), allow_pickle=False) as data:
        arrays = {key: np.array(data[key]) for key in ("action", "action_chunks", "chunk_index", "is_wait", "done")}
    return npz_path, arrays


def source_history_resets(blocks, task_ids, episode_ids):
    """``{(task_id, episode_idx): [source episode_idx, ...]}``: source episodes of the task that ran before the
    replayed one (and after the previous replayed one) and are not replayed -- their resets must be repeated.

    ``blocks``: ``env_history.run_history(source)``. Raises ``ValueError`` if a task has several source env blocks,
    ids repeat, or the replay order is not a subsequence of the source order."""
    order = {}
    for block in blocks:
        if block["task_id"] in order:
            raise ValueError("source run built several envs for task {}: env history ambiguous".format(block["task_id"]))
        order[block["task_id"]] = [int(r["episode_idx"]) for r in block["rows"]]
    if len(set(task_ids)) != len(task_ids) or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("replay task / episode ids repeat")
    out = {}
    for task_id in task_ids:
        source = order.get(task_id)
        if source is None:
            raise ValueError("source run has no task {}".format(task_id))
        last = -1
        for episode_idx in episode_ids:
            if episode_idx not in source:
                raise ValueError("source run has no episode task {} / episode {}".format(task_id, episode_idx))
            pos = source.index(episode_idx)
            if pos <= last:
                raise ValueError("replay order {} of task {} is not the source order {}: env history differs".format(
                    list(episode_ids), task_id, source))
            out[(task_id, episode_idx)] = source[last + 1:pos]
            last = pos
    return out


def check_protocol(args, row):
    for name, value in (("suite", args.task_suite_name), ("seed", args.seed), ("num_steps_wait", args.num_steps_wait)):
        if row.get(name) != value:
            raise ValueError("replay {}={!r} differs from the source episode ({!r})".format(name, value, row.get(name)))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="new run root allocated by common.run_artifacts")
    parser.add_argument("--source-run-dir", required=True, help="existing run whose episodes are replayed (read-only)")
    parser.add_argument("--task-suite-name", default=client_mod.DEFAULT_TASK_SUITE, choices=sorted(client_mod.MAX_STEPS))
    parser.add_argument("--task-ids", type=client_mod._int_list, default=[0])
    parser.add_argument("--episode-ids", type=client_mod._int_list, default=[0])
    parser.add_argument("--seed", type=int, default=client_mod.DEFAULT_SEED)
    parser.add_argument("--resize-size", type=int, default=client_mod.DEFAULT_RESIZE_SIZE)
    parser.add_argument("--replan-steps", type=int, default=client_mod.DEFAULT_REPLAN_STEPS)
    parser.add_argument("--num-steps-wait", type=int, default=client_mod.DEFAULT_NUM_STEPS_WAIT)
    client_mod.add_geometry_arguments(parser)
    return parser


def replay(args) -> int:
    run_dir = client_mod.validate_run_dir(args.run_dir)
    source_run = client_mod.validate_run_dir(args.source_run_dir)
    if run_dir == source_run:
        raise ValueError("--run-dir must be a new run, not the source run")
    client_mod.check_geometry_arguments(args)
    rows = read_source_rows(source_run)
    sources = {}
    for task_id in args.task_ids:
        for episode_idx in args.episode_ids:
            row = rows.get((task_id, episode_idx))
            if row is None:
                raise ValueError("source run has no episode task {} / episode {}".format(task_id, episode_idx))
            check_protocol(args, row)
            npz_path, arrays = load_source_episode(source_run, row)
            check_replayable(arrays, args.replan_steps)
            sources[(task_id, episode_idx)] = (row, npz_path, arrays)
    resets = source_history_resets(env_history.run_history(source_run), list(args.task_ids), list(args.episode_ids))

    client_mod.require_libero_config()
    from libero.libero import benchmark

    (run_dir / "episodes").mkdir(exist_ok=True)
    (run_dir / "videos").mkdir(exist_ok=True)
    jsonl = open(str(run_dir / "episodes.jsonl"), "x")

    # Same order of RNG seeding / suite / env creation as the client (official example).
    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = client_mod.MAX_STEPS[args.task_suite_name]

    def policy_for(task_id, episode_idx):
        row, npz_path, arrays = sources[(task_id, episode_idx)]
        done = np.flatnonzero(arrays["done"])
        info = {
            "source_run_dir": str(source_run),
            "source_run_id": source_run.name,
            "source_npz": str(npz_path.relative_to(source_run)),
            "source_npz_sha256": sha256_file(npz_path),
            "source_status": row.get("status"),
            "source_policy_calls": int(len(arrays["action_chunks"])),
            "source_env_steps_including_wait": int(len(arrays["action"])),
            "source_done_step_index": [int(i) for i in done],
            "executed_actions": "recorded action_chunks[i][:replan_steps] (checked equal to recorded action)",
            "source_history_resets": resets[(task_id, episode_idx)],
        }
        policy = ReplayPolicy(arrays["action_chunks"], {"replay_of": info["source_run_id"], "source_npz": info["source_npz"]})
        return policy, "replay (recorded action chunks; no policy server)", {"replay": info}

    def prepare_episode(env, initial_states, task_id, episode_idx):
        for skipped in resets[(task_id, episode_idx)]:  # source episodes not replayed: same reset + init state
            client_mod.start_episode(env, initial_states, skipped)

    with jsonl:
        errors = client_mod.run_tasks(args, run_dir, jsonl, task_suite, max_steps, policy_for, prepare_episode)
    return 1 if errors else 0


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return replay(args)


if __name__ == "__main__":
    raise SystemExit(main())
