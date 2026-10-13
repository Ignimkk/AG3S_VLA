"""SL4b fix: the scene / phase exports and the action replay reproduce the rollout's LIBERO env history.

LIBERO re-samples the fixture placement (cabinet / stove ``body_pos``; not in the recorded sim state) from the global
numpy RNG on every ``env.reset()``; the client resets one env per task once per episode. A fresh env with one reset
therefore shows episode k > first of a task with the wrong fixtures. CPU only: a fake env with the same RNG rule
(no LIBERO) and, when ``.venv-libero`` and the SL4b dev run are present, the real thing on a small case.
"""
import ast
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from benchmark.adapters.libero_panda import client_env, env_history
from benchmark.runners import libero_pi05_client as client
from benchmark.runners import libero_replay
from common.workspace import PROJECT_ROOT, shared_storage_root

from tests.benchmark.test_libero_pi05_sl0 import FakeEnv, FakePolicy  # noqa: F401

LIBERO_PYTHON = PROJECT_ROOT / ".venv-libero/bin/python"
PY38_HISTORY = PROJECT_ROOT / "tests/benchmark/libero_py38_history_checks.py"
SL4B_DEV_RUN = "20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4"
SUITE = "libero_spatial"


# --- fake LIBERO with the fixture rule ---------------------------------------------------------------------
class FixtureEnv(FakeEnv):
    """``env.seed`` = ``np.random.seed``; every reset draws a new fixture offset; steps draw nothing (robosuite)."""

    def __init__(self, seed, done_at=10 + 4):
        super().__init__(done_at=done_at)
        np.random.seed(seed)
        self.fixture, self.resets, self.starts, self.closed = None, 0, [], False

    def reset(self):
        super().reset()
        self.resets += 1
        self.fixture = float(np.random.uniform(-0.02, 0.02))

    def set_init_state(self, state):
        self.starts.append((state, self.fixture))
        return self._obs()

    def close(self):
        self.closed = True


class FakeSuite:
    def get_task(self, task_id):
        return types.SimpleNamespace(language="task {}".format(task_id))

    def get_task_init_states(self, task_id):
        return list(range(5))  # init state k is just the label k


def _rows(task_ids, episode_ids, **extra):
    return [dict({"suite": SUITE, "task_id": t, "episode_idx": e, "seed": 7,
                  "started_at": "2026-10-13T09:{:02d}:{:02d}".format(i // 60, i % 60),
                  "npz": "episodes/{}_t{}_e{}.npz".format(SUITE, t, e)}, **extra)
            for i, (t, e) in enumerate((t, e) for t in task_ids for e in episode_ids)]


def _protocol(task_ids, episode_ids):
    return {"suite": SUITE, "seed": 7, "task_ids": list(task_ids), "episode_ids": list(episode_ids)}


def _write_run(path, task_ids, episode_ids):
    path.mkdir(parents=True)
    (path / "run_manifest.json").write_text(json.dumps({"settings": {"protocol": _protocol(task_ids, episode_ids)}}))
    (path / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _rows(task_ids, episode_ids)))
    return path


@pytest.fixture
def fake_libero(monkeypatch):
    """``_get_libero_env`` -> FixtureEnv (seeded like LIBERO) and a fake ``libero.libero.benchmark``."""
    made = []

    def get_env(task, resolution, seed):
        env = FixtureEnv(seed)
        made.append(env)
        return env, task.language

    monkeypatch.setattr(client, "_get_libero_env", get_env)
    bench = types.ModuleType("libero.libero.benchmark")
    bench.get_benchmark_dict = lambda: {SUITE: FakeSuite}
    inner = types.ModuleType("libero.libero")
    inner.benchmark = bench
    outer = types.ModuleType("libero")
    outer.libero = inner
    for name, module in (("libero", outer), ("libero.libero", inner), ("libero.libero.benchmark", bench)):
        monkeypatch.setitem(sys.modules, name, module)
    return made


# --- order rules -------------------------------------------------------------------------------------------
def test_history_blocks_follow_the_client_loop_and_refuse_an_unknown_order():
    rows = _rows([0, 3], [1, 2, 3, 4])
    blocks = env_history.history_blocks(rows, _protocol([0, 3], [1, 2, 3, 4]))
    assert [(b["block"], b["task_id"], b["seed"], b["capture_geometry"]) for b in blocks] == [(0, 0, 7, False),
                                                                                              (1, 3, 7, False)]
    assert [[r["episode_idx"] for r in b["rows"]] for b in blocks] == [[1, 2, 3, 4], [1, 2, 3, 4]]
    with pytest.raises(ValueError, match="no settings.protocol"):
        env_history.history_blocks(rows, {"suite": SUITE, "seed": 7})
    with pytest.raises(ValueError, match="not the client loop"):  # line order differs from the loop
        env_history.history_blocks([rows[1], rows[0]] + rows[2:], _protocol([0, 3], [1, 2, 3, 4]))
    with pytest.raises(ValueError, match="not the client loop"):  # incomplete run
        env_history.history_blocks(rows[:-1], _protocol([0, 3], [1, 2, 3, 4]))
    with pytest.raises(ValueError, match="not the client loop"):  # manifest order differs
        env_history.history_blocks(rows, _protocol([0, 3], [2, 1, 3, 4]))
    late = [dict(r) for r in rows]
    late[2]["started_at"] = "2026-10-13T08:00:00"
    with pytest.raises(ValueError, match="started_at"):
        env_history.history_blocks(late, _protocol([0, 3], [1, 2, 3, 4]))
    mixed = [dict(r) for r in rows]
    mixed[1]["seed"] = 8
    with pytest.raises(ValueError, match="seed"):
        env_history.history_blocks(mixed, _protocol([0, 3], [1, 2, 3, 4]))
    geo = [dict(r) for r in rows]
    geo[0]["geometry"] = {"npz": "x"}
    with pytest.raises(ValueError, match="geometry"):
        env_history.history_blocks(geo, _protocol([0, 3], [1, 2, 3, 4]))


def test_plan_repeats_resets_up_to_the_last_wanted_episode():
    blocks = env_history.history_blocks(_rows([0, 3], [1, 2, 3, 4]), _protocol([0, 3], [1, 2, 3, 4]))
    plan = env_history.plan(blocks, {"libero_spatial_t3_e3", "libero_spatial_t3_e1"})
    assert len(plan) == 1 and plan[0][0]["task_id"] == 3
    assert [(r["episode_idx"], wanted) for r, wanted in plan[0][1]] == [(1, True), (2, False), (3, True)]
    full = env_history.plan(blocks)
    assert [[r["episode_idx"] for r, _ in items] for _, items in full] == [[1, 2, 3, 4], [1, 2, 3, 4]]
    note = env_history.history_note(blocks[1], 2)
    assert note["resets_before"] == 2 and note["reset_index"] == 3
    assert note["preceding_episodes"] == ["libero_spatial_t3_e1", "libero_spatial_t3_e2"]


def test_replay_source_history_resets_rules():
    blocks = env_history.history_blocks(_rows([0, 1], [1, 2, 3, 4]), _protocol([0, 1], [1, 2, 3, 4]))
    assert libero_replay.source_history_resets(blocks, [0, 1], [1, 2, 3, 4]) == {
        (t, e): [] for t in (0, 1) for e in (1, 2, 3, 4)}  # the source's own list: nothing added
    assert libero_replay.source_history_resets(blocks, [1], [3]) == {(1, 3): [1, 2]}
    assert libero_replay.source_history_resets(blocks, [0], [2, 4]) == {(0, 2): [1], (0, 4): [3]}
    with pytest.raises(ValueError, match="not the source order"):
        libero_replay.source_history_resets(blocks, [0], [3, 2])
    with pytest.raises(ValueError, match="repeat"):
        libero_replay.source_history_resets(blocks, [0], [2, 2])
    with pytest.raises(ValueError, match="no episode"):
        libero_replay.source_history_resets(blocks, [0], [0])
    with pytest.raises(ValueError, match="no task"):
        libero_replay.source_history_resets(blocks, [5], [1])
    twice = env_history.history_blocks(_rows([0, 0], [1]), _protocol([0, 0], [1]))
    with pytest.raises(ValueError, match="several envs"):
        libero_replay.source_history_resets(twice, [0], [1])


def test_rgb_check_fails_loudly():
    env_history.require_rgb_equal("s", {"agentview": 3, "robot0_eye_in_hand": 3}, 3)
    with pytest.raises(ValueError, match="re-rendered RGB differs"):
        env_history.require_rgb_equal("s", {"agentview": 0, "robot0_eye_in_hand": 3}, 3)


def test_env_history_and_exports_are_python38():
    for path in ("benchmark/adapters/libero_panda/env_history.py", "benchmark/adapters/libero_panda/scene_export.py",
                 "benchmark/adapters/libero_panda/phase_export.py", "benchmark/runners/libero_replay.py",
                 "benchmark/runners/libero_pi05_client.py", "tests/benchmark/libero_py38_history_checks.py"):
        ast.parse((PROJECT_ROOT / path).read_text(), feature_version=(3, 8))


# --- the shared helpers are the client's loop ----------------------------------------------------------------
def test_start_episode_is_reset_then_init_state(fake_libero):
    env, _ = client.make_task_env(FakeSuite().get_task(0), 7, False)
    assert fake_libero == [env] and env.resets == 0  # building the env does not reset it
    obs = client.start_episode(env, [10, 11, 12], 2)
    assert env.resets == 1 and env.starts == [(12, env.fixture)] and "robot0_eef_pos" in obs


def _run_client(tmp_path, monkeypatch, run, task_ids, episode_ids, policy_for=None, prepare_episode=None):
    pytest.importorskip("openpi_client.image_tools")
    (run / "episodes").mkdir(parents=True, exist_ok=True)
    (run / "videos").mkdir(exist_ok=True)
    monkeypatch.setattr(client, "_save_video", lambda frames, final: final.write_bytes(b"mp4"))
    args = client.build_parser().parse_args(["--run-dir", "x", "--port", "1", "--task-ids",
                                             ",".join(map(str, task_ids)), "--episode-ids", ",".join(map(str, episode_ids))])
    policy = policy_for or (lambda t, e: (type("P", (FakePolicy,), {"get_server_metadata": lambda self: {}})(), "f", {}))
    with open(run / "episodes.jsonl", "x") as jsonl:
        assert client.run_tasks(args, run, jsonl, FakeSuite(), 30, policy, prepare_episode) == 0
    return [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]


def test_multi_episode_reset_history_is_reproduced_by_exports_and_replay(tmp_path, monkeypatch, fake_libero):
    """Rollout (``run_tasks``, 2 tasks x episodes 1-3) -> export envs (``episode_envs``) -> replay of one episode."""
    run = tmp_path / "run"
    rows = _run_client(tmp_path, monkeypatch, run, [0, 4], [1, 2, 3])
    (run / "run_manifest.json").write_text(json.dumps({"settings": {"protocol": _protocol([0, 4], [1, 2, 3])}}))
    rollout = {}
    for env, task_id in zip(fake_libero, (0, 4)):
        assert env.resets == 3
        for episode_idx, fixture in env.starts:
            rollout[(task_id, episode_idx)] = fixture
    assert len(set(rollout.values())) == 3  # one fixture per reset index; same seed -> same sequence per task
    assert [(r["task_id"], r["episode_idx"]) for r in rows] == [(0, 1), (0, 2), (0, 3), (4, 1), (4, 2), (4, 3)]

    # exports: every episode, and a subset that skips e1 / e2 of task 4 (their resets are still repeated)
    del fake_libero[:]
    seen = {}
    for row, env, note in env_history.episode_envs(run):
        seen[(row["task_id"], row["episode_idx"])] = env.starts[-1]
        assert note["resets_before"] == row["episode_idx"] - 1 and env.resets == row["episode_idx"]
    assert seen == {k: (k[1], v) for k, v in rollout.items()}
    assert all(env.closed for env in fake_libero) and len(fake_libero) == 2  # one env per task block
    del fake_libero[:]
    subset = list(env_history.episode_envs(run, {"libero_spatial_t4_e3"}))
    assert len(subset) == 1 and subset[0][1].starts == [(1, rollout[(4, 1)]), (2, rollout[(4, 2)]), (3, rollout[(4, 3)])]
    assert len(fake_libero) == 1  # task 0 has no wanted episode: no env built

    # the pre-fix exports: a fresh env with one reset shows the first episode's fixtures for every episode
    for k in (1, 2, 3):
        fresh, _ = client.make_task_env(FakeSuite().get_task(4), 7, False)
        client.start_episode(fresh, FakeSuite().get_task_init_states(4), k)
        assert (fresh.fixture == rollout[(4, k)]) is (k == 1)

    # replay of task 4 / episode 3 alone through run_tasks' prepare_episode: the source resets are repeated
    resets = libero_replay.source_history_resets(env_history.run_history(run), [4], [3])
    assert resets == {(4, 3): [1, 2]}
    del fake_libero[:]

    def prepare(env, initial_states, task_id, episode_idx):
        for skipped in resets[(task_id, episode_idx)]:
            client.start_episode(env, initial_states, skipped)

    _run_client(tmp_path, monkeypatch, tmp_path / "replay", [4], [3], prepare_episode=prepare)
    assert fake_libero[0].starts[-1] == (3, rollout[(4, 3)]) and fake_libero[0].resets == 3


def test_run_tasks_without_prepare_is_the_client_loop(tmp_path, monkeypatch, fake_libero):
    """``prepare_episode=None`` (the client): exactly one reset + init state per episode, nothing else."""
    _run_client(tmp_path, monkeypatch, tmp_path / "run", [0], [2, 0, 4])
    env = fake_libero[0]
    assert env.resets == 3 and [s for s, _ in env.starts] == [2, 0, 4]


# --- the real thing (.venv-libero + the SL4b dev run) ----------------------------------------------------------
@pytest.fixture(scope="module")
def real_history(tmp_path_factory):
    run = shared_storage_root() / "results/benchmark" / SL4B_DEV_RUN
    if not (LIBERO_PYTHON.exists() and (run / "episodes.jsonl").is_file()):
        pytest.skip("needs .venv-libero and the SL4b dev run")
    out = tmp_path_factory.mktemp("sl4b_history")
    proc = subprocess.run([str(LIBERO_PYTHON), str(PY38_HISTORY), "--run", str(run), "--task", "0", "--episodes", "1,3",
                           "--frames", "2", "--out", str(out)], capture_output=True, text=True, timeout=900,
                          env=client_env.client_environment())
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads((out / "history_checks.json").read_text())


def _all_equal(frames):
    return all(all(f.values()) for f in frames)


def test_real_libero_history_reproduces_the_recorded_frames(real_history):
    h, fresh = real_history["history"], real_history["fresh"]
    assert {s: v["resets_before"] for s, v in h.items()} == {"libero_spatial_t0_e1": 0, "libero_spatial_t0_e3": 2}
    assert all(_all_equal(v["frames"]) for v in h.values())  # both cameras, bit-identical
    # fresh env (pre-fix exports): only the task's first episode re-renders; e3 shows e1's fixtures
    assert _all_equal(fresh["libero_spatial_t0_e1"]["frames"])
    assert not any(f["agentview"] for f in fresh["libero_spatial_t0_e3"]["frames"])
    assert fresh["libero_spatial_t0_e3"]["fixtures"] == fresh["libero_spatial_t0_e1"]["fixtures"]
    assert h["libero_spatial_t0_e3"]["fixtures"] != h["libero_spatial_t0_e1"]["fixtures"]
    # replay of e3 alone repeats the source resets of e1, e2
    r = real_history["replay_last"]
    assert r["stem"] == "libero_spatial_t0_e3" and r["source_history_resets"] == [1, 2]
    assert _all_equal(r["frames"]) and r["fixtures"] == h["libero_spatial_t0_e3"]["fixtures"]


def test_real_run_order_is_the_client_loop():
    run = shared_storage_root() / "results/benchmark" / SL4B_DEV_RUN
    if not (run / "episodes.jsonl").is_file():
        pytest.skip("SL4b dev run not in this storage root")
    blocks = env_history.run_history(run)
    assert [b["task_id"] for b in blocks] == list(range(10))
    assert all([int(r["episode_idx"]) for r in b["rows"]] == [1, 2, 3, 4] and b["capture_geometry"] for b in blocks)
