"""BM-SETUP-A: the CPU suite probe (``benchmark/evaluators/suite_probe.py``, Python 3.8) on two non-spatial scenes.

SafeLIBERO Object Level II task 0 (floor scene, robot base at world z 0): adapter conventions hold, the official collision
rule does not fire while the robot holds still and fires (with a MuJoCo robot-obstacle contact) when the EE is driven
into the active obstacle. LIBERO 10 task 5 (study scene): the ``study_table`` support resolves and lies inside the
Panda ESDF box.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PY38 = ROOT / ".venv-libero/bin/python"


def _probe(tmp_path, *argv) -> dict:
    if not PY38.exists():
        pytest.skip(".venv-libero not installed")
    from benchmark.runners import launch_libero_pi05_only as l0

    out = tmp_path / "probe.json"
    r = subprocess.run([str(PY38), "benchmark/evaluators/suite_probe.py", *argv, "--out", str(out)], cwd=str(ROOT),
                       capture_output=True, text=True, env=l0.client_environment(), timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    return json.loads(out.read_text())


def _conventions_hold(t):
    for f in t["frames"].values():
        assert f["rgb_is_raw_row_flip"] and f["policy_is_opencv_mirror"] and f["K_pixel_centre"]
        assert f["T_base_cam_rigid"] and f["depth_finite_frac"] == 1.0
    assert t["robot"]["robot_state_dim"] == 9 and t["robot"]["policy_state_dim"] == 8
    assert t["robot"]["controller"] == "OperationalSpaceController" and t["robot"]["control_freq_hz"] == 20.0


def test_safelibero_object_floor_scene_collision_rule(tmp_path):
    d = _probe(tmp_path, "--suite", "safelibero_object", "--safety-level", "II", "--task-ids", "0")
    t = d["tasks"][0]
    assert t["ok"], t.get("traceback")
    _conventions_hold(t)
    assert d["num_steps_wait"] == 20 and t["n_init_states"] == 50
    assert t["support"]["fixture"] == "floor" and t["support"]["above_esdf_floor"]
    assert t["obstacle"]["active"] == "moka_pot_obstacle_1"
    assert t["hold"]["collided"] is False and t["hold"]["max_displacement_m"] < 1e-3
    dr = t["drive"]
    assert dr["collided"] and dr["first_robot_contact_step"] is not None
    assert dr["first_robot_contact_step"] <= dr["drive_steps"] and dr["robot_bodies_in_contact_substeps"]


def test_libero_10_study_scene_support(tmp_path):
    d = _probe(tmp_path, "--suite", "libero_10", "--task-ids", "5")
    t = d["tasks"][0]
    assert t["ok"], t.get("traceback")
    _conventions_hold(t)
    assert t["unresolved"] == [] and t["support"]["fixture"] == "study_table"
    assert -0.10 < t["support"]["z_base_median_m"] < 0.0
    assert t["target_rule"]["target"] == "black_book_1" and t["target_is_body"]
