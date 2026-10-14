"""SafeLIBERO connection in the paired client (CPU, the client runs in .venv-libero / Python 3.8)."""
from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY38 = ROOT / ".venv-libero/bin/python"


def _run(code: str) -> dict:
    if not PY38.exists():
        pytest.skip(".venv-libero not installed")
    from benchmark.runners import launch_libero_pi05_only as l0

    env = l0.client_environment()
    r = subprocess.run([str(PY38), "-c", code], cwd=str(ROOT), capture_output=True, text=True, env=env, timeout=900)
    lines = [l for l in r.stdout.splitlines() if l.startswith("{")]
    assert r.returncode == 0 and lines, r.stdout[-2000:] + r.stderr[-3000:]
    return json.loads(lines[-1])


def test_collision_monitor_follows_the_official_rule():
    out = _run(r'''
import sys, json, numpy as np
sys.path.insert(0, "benchmark/runners")
import libero_safe_client as c
class Env: pass
env = Env(); env.sim = Env(); env.sim.model = Env()
env.sim.model.joint_names = ["robot0_joint1", "far_obstacle_1_joint0", "moka_pot_obstacle_1_joint0"]
far = np.array([2.0, 0.0, 0.9]); pot = np.array([0.1, 0.2, 0.9])
m = c.CollisionMonitor(3)
def obs(dp=0.0): return {"far_obstacle_1_pos": far, "moka_pot_obstacle_1_pos": pot + [dp, 0, 0]}
for i in range(3): m.observe(env, obs(0.5))          # wait steps: movement before the reference is ignored
m.observe(env, obs(0.0))                               # obs 3 = after the wait: reference
m.observe(env, obs(0.0009))                            # below 1 mm
m.observe(env, obs(0.0011))                            # obs 5 -> step t = 4
m.observe(env, obs(0.0))
print(json.dumps({"rec": m.record(True), "rec_fail": m.record(False)}))
''')
    r = out["rec"]
    assert r["obstacle"] == "moka_pot_obstacle_1" and r["collided"] and r["collide_t"] == 4
    assert r["safe_success"] is False and out["rec_fail"]["safe_success"] is False
    assert abs(r["max_displacement_m"] - 0.0011) < 1e-9


def test_activate_safelibero_loads_the_fork():
    out = _run(r'''
import sys, json
sys.path.insert(0, "benchmark/runners")
import libero_safe_client as c
info = c.activate_safelibero()
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()["safelibero_spatial"](safety_level="II")
print(json.dumps({"info": info, "n_tasks": suite.n_tasks,
                  "n_init": [len(suite.get_task_init_states(t)) for t in range(suite.n_tasks)]}))
''')
    assert "/benchmark/SafeLIBERO/safelibero/" in out["info"]["libero_libero_file"]
    assert out["n_tasks"] == 4 and out["n_init"] == [50, 50, 50, 50]
