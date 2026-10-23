"""BM-SETUP-B — LIBERO-Safety client (CPU, no LIBERO import; runs under .venv-ag3s and .venv-libero-safety)."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark/runners"))

import libero_pi05_client as base  # noqa: E402
import libero_safety_client as lsc  # noqa: E402

LS = ROOT / "benchmark/LIBERO-Safety"


class FakeEnv:
    """Deterministic env: obs from a counter, optional cost at given steps, success at ``done_at``."""

    def __init__(self, cost_at=(), done_at=None, wait_cost_at=()):
        self.n = 0
        self.cost_at, self.done_at, self.wait_cost_at = set(cost_at), done_at, set(wait_cost_at)
        self.actions = []

    def _obs(self):
        k = float(self.n)
        img = (np.arange(256 * 256 * 3).reshape(256, 256, 3) + self.n) % 255
        return {"agentview_image": img.astype(np.uint8), "robot0_eye_in_hand_image": img[::-1].astype(np.uint8),
                "robot0_eef_pos": np.array([k * 1e-3, 0.0, 1.0]), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_joint_pos": np.zeros(7), "robot0_gripper_qpos": np.array([0.04, -0.04])}

    def reset(self):
        self.n = 0

    def set_init_state(self, s):
        return self._obs()

    def step(self, a):
        self.actions.append(list(a))
        self.n += 1
        done = self.done_at is not None and self.n == self.done_at
        cost = {}
        if self.n in self.cost_at or self.n in self.wait_cost_at:
            cost = {"checkrobotcontact": 1, "checkcontact": 0}
        elif not done:
            cost = {"checkrobotcontact": 0, "checkcontact": 0}
        return self._obs(), 0.0, done, {"cost": cost}


class FakeClient:
    def __init__(self, scale=0.1):
        self.calls = 0
        self.scale = scale

    def infer(self, element):
        self.calls += 1
        return {"actions": np.full((10, 7), self.scale * self.calls)}


class FakeMonitor(lsc.ConstraintMonitor):
    """ConstraintMonitor without an env (official cost bookkeeping only)."""

    def __init__(self, wait):
        self.inner = None
        self.wait = wait
        self.constraints, self.labels, self.robot_ids, self.obj_ids = [], [], set(), {}
        self.t = 0
        self.official_first, self.official_wait, self.official_steps = None, [], 0
        self.full_first, self.geom_first, self.cost_trace = {}, {}, []

    def evaluate_full(self):
        return {}

    def evaluate_geom(self):
        return {}


def _args(**kw):
    a = dict(num_steps_wait=10, resize_size=224, replan_steps=5, image_transform="rot180", action_adapter="none",
             violation_termination="official")
    a.update(kw)
    return argparse.Namespace(**a)


def test_loop_matches_base_without_monitor():
    """No monitor / rot180 / no adapter: same env actions and recorder arrays as libero_pi05_client.run_episode."""
    pytest.importorskip("openpi_client")
    e1, e2 = FakeEnv(done_at=37), FakeEnv(done_at=37)
    s1, r1, im1 = base.run_episode(e1, [None], 0, "task", FakeClient(), _args(), 50)
    s2, r2, im2, term = lsc.run_episode(e2, [None], 0, "task", FakeClient(), _args(), 50)
    assert e1.actions == e2.actions and not term
    a1, a2 = r1.arrays(), r2.arrays()
    assert a1.keys() == a2.keys()
    for k in a1:
        np.testing.assert_array_equal(a1[k], a2[k])
    assert len(im1) == len(im2) and all(np.array_equal(x, y) for x, y in zip(im1, im2))
    for k in ("official_success", "status", "env_steps_including_wait", "loop_t_final", "policy_calls"):
        assert s1[k] == s2[k]


def test_violation_terminates_at_policy_step_only():
    pytest.importorskip("openpi_client")
    env = FakeEnv(cost_at={18}, wait_cost_at={3})
    mon = FakeMonitor(10)
    s, rec, _, term = lsc.run_episode(env, [None], 0, "task", FakeClient(), _args(), 50, monitor=mon)
    assert term and s["status"] == "violation" and not s["official_success"]
    assert len(env.actions) == 18  # stopped right after the violating step
    r = mon.record(s["official_success"], term)
    assert r["official_first_violation"] == [17, ["checkrobotcontact"]]   # monitor t counts from 0
    assert r["official_violation_during_wait"] == [[2, ["checkrobotcontact"]]]
    assert r["collided"] and not r["safe_success"]


def test_violation_recorded_without_termination():
    pytest.importorskip("openpi_client")
    env = FakeEnv(cost_at={18, 19})
    mon = FakeMonitor(10)
    s, _, _, term = lsc.run_episode(env, [None], 0, "task", FakeClient(), _args(violation_termination="none"), 30,
                                    monitor=mon)
    assert not term and s["status"] == "max_steps" and len(env.actions) == 40
    assert mon.record(False, term)["official_violation_policy_steps"] == 2


def test_action_adapter_and_transforms():
    a = np.array([1.0, -0.5, 0.2, 0.4, -0.2, 0.1, 1.0])
    b = lsc.adapt_action(a, "libero_to_safety")
    fit = json.loads((ROOT / "robots/panda/description/osc_response_fit.json").read_text())
    np.testing.assert_allclose(b[:3], a[:3] * 0.05 * fit["pos"]["steady_state_gain"])
    np.testing.assert_allclose(b[3:6], a[3:6] * 0.5 * fit["rot"]["steady_state_gain"])
    assert b[6] == a[6]
    np.testing.assert_array_equal(lsc.adapt_action(a, "none"), a)
    x = np.arange(24).reshape(2, 4, 3)
    np.testing.assert_array_equal(lsc.policy_image(x, "rot180"), x[::-1, ::-1])
    np.testing.assert_array_equal(lsc.policy_image(x, "vflip"), x[::-1])
    np.testing.assert_array_equal(lsc.policy_image(x, "none"), x)


def test_fork_gain_record_matches_constant():
    path = ROOT / "outputs/verify/BM-SETUP-B/controller/fork_osc_gain.json"
    if not path.exists():
        pytest.skip("fork gain record not present")
    gains = [m["gain"] for m in json.loads(path.read_text())["measurements"]]
    assert all(abs(g - lsc.REALISED_GAIN["libero_safety_fork"]["pos"]) < 0.06 for g in gains)


def test_adapter_uses_fork_controller_constants():
    cfg = json.loads((LS / "third_party/robosuite-1.4/robosuite/controllers/config/osc_pose.json").read_text())
    fork = lsc.OSC_OUTPUT_MAX["libero_safety_fork"]
    assert cfg["output_max"] == [fork["pos"]] * 3 + [fork["ori"]] * 3 and cfg["kp"] == fork["kp"]


def test_ldlj_amplitude_invariant():
    t = np.linspace(0, 1, 101)
    s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5   # minimum jerk
    x = np.stack([s, 0 * s, 0 * s], 1)
    a, b = lsc.ldlj(x, 0.01), lsc.ldlj(3.0 * x, 0.01)
    assert a is not None and abs(a - b) < 1e-9
    noisy = x + np.random.default_rng(0).normal(scale=1e-3, size=x.shape)
    assert lsc.ldlj(noisy, 0.01) < a     # less smooth -> more negative
    assert lsc.ldlj(x[:3], 0.01) is None


def test_level_suite_paths():
    from collections import namedtuple

    Task = namedtuple("Task", "name language problem problem_folder bddl_file init_states_file level level_id")

    class B:
        def get_all_tasks_by_level(self, level):
            return [Task("t%d" % i, "l", "vla_safety", "human_safety", "t%d.bddl" % i, "t%d.pruned_init" % i, level, i)
                    for i in range(5)]

        def get_task_init_states(self, level, i):
            return ("init", level, i)

    s = lsc.LevelSuite(B(), 2)
    assert s.n_tasks == 5 and s.get_task(3).problem_folder == "human_safety/L2"
    assert s.get_task_init_states(3) == ("init", 2, 3)


def test_task_map_matches_constants():
    """vla_safety_task_map: 5 tasks per level for every supported suite; init files exist (50 each, checked in probe)."""
    ns = {}
    exec((LS / "libero/libero/benchmark/vla_safety_task_map.py").read_text(), ns)
    m = ns["vla_safety_task_map"]
    for suite in lsc.SUITES:
        assert sorted(m[suite]) == list(lsc.LEVELS)
        for level in lsc.LEVELS:
            assert len(m[suite][level]) == lsc.TASKS_PER_LEVEL
            for task in m[suite][level]:
                assert (LS / "libero/libero/bddl_files/{}/L{}/{}.bddl".format(suite, level, task)).is_file()
                assert (LS / "libero/libero/init_files/{}/L{}/{}.pruned_init".format(suite, level, task)).is_file()
    assert not (LS / "libero/libero/init_files/reasoning_safety").exists()
    assert "reasoning_safety" in lsc.UNSUPPORTED_SUITES


def test_summarize():
    def row(ok, viol, steps):
        return {"official_success": ok, "libero_safety": {"safe_success": ok and not viol, "terminated_by_violation": viol,
                                                          "official_violation": viol, "full_violation": viol,
                                                          "geom_robot_contact": False, "execution_steps": steps,
                                                          "ldlj_eef": -10.0}}
    s = lsc.summarize([row(True, False, 100), row(False, True, 50), row(False, False, 600), row(True, False, 120)])
    assert s["n"] == 4 and s["SR"] == 0.5 and s["CR"] == 0.25 and s["mean_execution_steps"] == 217.5


def test_parser_defaults_per_checkpoint():
    p = lsc.build_parser()
    a = lsc.resolve_defaults(p.parse_args(["--run-dir", "x", "--port", "1", "--task-suite-name", "human_safety",
                                           "--level", "1", "--policy-checkpoint", "pi05_libero"]))
    assert a.action_adapter == "libero_to_safety" and a.num_steps_wait == lsc.NUM_STEPS_WAIT
    a = lsc.resolve_defaults(p.parse_args(["--run-dir", "x", "--port", "1", "--task-suite-name", "human_safety",
                                           "--level", "1", "--policy-checkpoint", "pi05_libero_safety"]))
    assert a.action_adapter == "none" and a.violation_termination == "official"
    with pytest.raises(SystemExit):
        p.parse_args(["--run-dir", "x", "--port", "1", "--task-suite-name", "reasoning_safety", "--level", "0",
                      "--policy-checkpoint", "pi05_libero"])


def test_wrapper_launcher_argv_and_refusals():
    from benchmark.runners import launch_libero_safety_paired as w

    p = w.build_parser()
    a = p.parse_args(["--task-suite-name", "human_safety", "--level", "2", "--task-ids", "0,4", "--episode-ids", "0,4",
                      "--run-id", "x", "--policy-checkpoint", "pi05_libero", "--conversion", "B"])
    w.check(a, ["vla_only", "vla_v11ab"])                      # pi05_libero: allowed (openpi config exists)
    argv = w.make_client_argv(a)("RUN", 1234)
    assert argv[0].endswith(".venv-libero-safety/bin/python") and argv[1].endswith("libero_safety_client.py")
    assert argv[argv.index("--level") + 1] == "2" and argv[argv.index("--policy-checkpoint") + 1] == "pi05_libero"
    a.episode_ids = [5]
    with pytest.raises(ValueError):                             # dev split = init 0-4
        w.check(a, ["vla_only"])
    a.data_split = "test"
    w.check(a, ["vla_only"])
    b = p.parse_args(["--task-suite-name", "human_safety", "--level", "0", "--task-ids", "0", "--episode-ids", "0",
                      "--run-id", "x", "--policy-checkpoint", "pi05_libero_safety", "--conversion", "B"])
    if not w.openpi_has_config("pi05_libero_safety"):
        with pytest.raises(ValueError, match="H1"):
            w.check(b, ["vla_only"])
    else:
        with pytest.raises(ValueError, match="H2"):
            w.check(b, ["vla_v11ab"])
    srv = w.make_server_argv("pi05_libero_safety")
    assert srv is not None
