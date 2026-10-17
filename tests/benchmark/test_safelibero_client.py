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


def test_launcher_records_the_safelibero_protocol():
    from benchmark.runners import launch_libero_paired as lp

    def args(*extra):
        a = lp.build_parser().parse_args(["--task-ids", "0", "--episode-ids", "0", "--run-id", "x",
                                          "--conversion", "B", *extra])
        for name, value in (("resize_size", 224), ("replan_steps", 5), ("num_steps_wait", 10)):
            setattr(a, name, getattr(a, name, value))
        return a

    sl = args("--task-suite-name", "safelibero_spatial", "--safety-level", "II", "--data-split", "d")
    p, r = lp.protocol_settings(sl), lp.runtime_schedule(sl)
    assert (p["suite"], p["safety_level"], p["max_steps"], p["num_steps_wait"]) == ("safelibero_spatial", "II", 280, 20)
    assert r["control_freq_hz"] == 20 and r["num_steps_wait"] == 20 and "SafeLIBERO" in r["control_freq_source"]
    lib = args()
    assert lp.protocol_settings(lib)["max_steps"] == 220 and lp.protocol_settings(lib)["num_steps_wait"] == 10


def test_ag3s_version_flags_reach_the_server_argv():
    from AG3S.adapters.panda import profile
    from benchmark.runners import launch_libero_paired as lp

    base = lp.server_argv("vla_v11ab", "<ckpt>", 1)
    pb1 = lp.server_argv("vla_v11ab", "<ckpt>", 1, "V11AB-pb1")
    assert "--reach-at-capture" not in base and pb1 == base + ["--reach-at-capture"]
    assert lp.server_argv("vla_only", "<ckpt>", 1, "V11AB-pb1") == lp.server_argv("vla_only", "<ckpt>", 1)
    args = profile.parse_v11ab_args(argv=profile.v11ab_server_argv() + profile.version_server_flags("V11AB-pb1"))
    assert profile.v11ab_build_kwargs(args)["reach_at_capture"] is True
    assert profile.v11ab_build_kwargs(profile.parse_v11ab_args())["reach_at_capture"] is False
    with pytest.raises(ValueError):
        profile.version_server_flags("V99")


def test_pb2_flags_reach_the_trajopt_config():
    from AG3S.adapters.panda import profile
    from AG3S.trajopt import serve_safe as ss

    args = profile.parse_v11ab_args(argv=profile.v11ab_server_argv() + profile.version_server_flags("V11AB-pb2"))
    cfg = ss.trajopt_config_from_args(args)
    assert cfg.sqp.minimal_intervention == pytest.approx(0.010)
    assert profile.v11ab_build_kwargs(args)["reach_at_capture"] is True
    assert ss.trajopt_config_from_args(profile.parse_v11ab_args()).sqp.minimal_intervention is None


def test_pb3_adds_the_arm_links_to_the_constraint_model():
    from AG3S.adapters.panda import model as pm
    from AG3S.adapters.panda import profile

    args = profile.parse_v11ab_args(argv=profile.v11ab_server_argv() + profile.version_server_flags("V11AB-pb3"))
    kw = profile.v11ab_build_kwargs(args)
    assert kw["links"] == "arms" and kw["reach_at_capture"] is True
    assert profile.v11ab_build_kwargs(profile.parse_v11ab_args())["links"] == "gripper"
    arm = pm.build_panda_constraint_model(pm.panda_constraint_link_filter("arms"))
    grip = pm.build_panda_constraint_model(pm.panda_constraint_link_filter("gripper"))
    assert arm.n_spheres > grip.n_spheres
    assert {"panda_link5", "panda_link6", "panda_link7"} <= set(arm.sphere_link_names)


def test_pb4_pb5_flags_reach_safe_policy():
    from types import SimpleNamespace

    from AG3S.adapters.panda import profile
    from AG3S.trajopt import serve_safe as ss

    def applied(version):
        args = profile.parse_v11ab_args(argv=profile.v11ab_server_argv() + profile.version_server_flags(version))
        safe = SimpleNamespace()
        return ss.apply_grasp_contact_flags(safe, args)

    assert applied("V11AB") == {"latch_phase": False, "held_hand_exempt": False, "grasp_zone_allowance_m": None,
                                "grasp_zone_radius_m": 0.1}
    assert applied("V11AB-pb4")["held_hand_exempt"] is True and applied("V11AB-pb4")["grasp_zone_allowance_m"] is None
    pb5 = applied("V11AB-pb5")
    assert pb5["held_hand_exempt"] and pb5["latch_phase"] and pb5["grasp_zone_allowance_m"] == pytest.approx(0.010)
